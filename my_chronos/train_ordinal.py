import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from chronos import ChronosConfig, ChronosModel, MeanScaleUniformBins
from ordinal_head import ProportionalOddsHead
import warnings
import numpy as np
from datasets import load_dataset
import gc
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, series_list, context_length, prediction_length):
        self.series_list = series_list
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def __len__(self):
        return len(self.series_list)
    
    def __getitem__(self, idx):
        series = self.series_list[idx]
        context = series[:self.context_length]
        label = series[self.context_length:self.context_length + self.prediction_length]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class FixedOrdinalModel(ChronosModel):
    def __init__(self, config, model, use_ordinal_head=False):
        super().__init__(config, model)
        self.use_ordinal_head = use_ordinal_head
        self.ordinal_head = None
        self.num_bins = config.n_tokens - config.n_special_tokens
        self.centers = torch.linspace(-15, 15, self.num_bins)
    
    def _init_ordinal_head(self):
        if self.ordinal_head is None and self.use_ordinal_head:
            self.ordinal_head = ProportionalOddsHead(
                hidden_dim=self.model.config.d_model,
                num_bins=self.num_bins,
                dropout_p=0.1,
            )
    
    def forward_ordinal(self, input_ids, attention_mask, prediction_length=12):
        self._init_ordinal_head()
        batch_size = input_ids.size(0)
        
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=input_ids.device,
            dtype=torch.long,
        )
        
        all_probs = []
        
        for step in range(prediction_length):
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
            
            probs_step = self.ordinal_head(decoder_outputs.last_hidden_state)
            probs_last = probs_step[:, -1:, :]
            all_probs.append(probs_last)
            
            centers = self.centers.to(input_ids.device)
            expectation = torch.sum(probs_last * centers.reshape(1, 1, -1), dim=-1, keepdim=True)
            next_token = torch.clamp(torch.round(expectation + 15), 0, self.num_bins - 1).long()
            next_token = next_token.squeeze(1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        return torch.stack(all_probs, dim=1)
    
    def forward_expectation(self, input_ids, attention_mask, prediction_length=12):
        probs = self.forward_ordinal(input_ids, attention_mask, prediction_length)
        centers = self.centers.to(input_ids.device)
        expectation = torch.sum(probs * centers.reshape(1, 1, 1, -1), dim=-1)
        return expectation

def collate_fn(batch):
    contexts, labels = zip(*batch)
    max_context_len = max(c.size(0) for c in contexts)
    max_label_len = max(l.size(0) for l in labels)
    
    padded_contexts = []
    for c in contexts:
        if c.size(0) < max_context_len:
            pad = torch.zeros(max_context_len - c.size(0))
            c = torch.cat([c, pad])
        padded_contexts.append(c)
    
    padded_labels = []
    for l in labels:
        if l.size(0) < max_label_len:
            pad = torch.zeros(max_label_len - l.size(0))
            l = torch.cat([l, pad])
        padded_labels.append(l)
    
    return torch.stack(padded_contexts), torch.stack(padded_labels)

def prepare_data(dataset_name, max_series, context_length, prediction_length):
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset("autogluon/chronos_datasets", dataset_name, split="train")
        df = dataset.to_pandas()
    except:
        local_path = Path(f"./data/{dataset_name}_train.parquet")
        if local_path.exists():
            df = pd.read_parquet(local_path)
            print(f"Loaded from local file: {local_path}")
        else:
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    train_series = []
    required_len = context_length + prediction_length
    
    for series_id in df['id'].unique()[:max_series]:
        series = df[df['id'] == series_id]['target'].values[0]
        if isinstance(series, np.ndarray):
            series = series.tolist()
        if len(series) > required_len:
            series = series[:required_len]
            train_series.append(series)
    
    print(f"Prepared {len(train_series)} series")
    return train_series

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_experiments.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    
    experiments = configs['experiments']
    
    if args.experiment:
        experiments = [e for e in experiments if e['name'] == args.experiment]
    
    for exp in experiments:
        print(f"{exp['name']}")
        
        dataset_name = args.dataset or exp.get('dataset', 'm4_hourly')
        max_series = args.max_series or exp.get('max_series', 100)
        n_tokens = args.n_tokens or exp.get('n_tokens', 66)
        epochs = args.epochs or exp.get('epochs', 100)
        
        config = {
            'dataset': dataset_name,
            'model_id': 'amazon/chronos-t5-small',
            'context_length': 64,
            'prediction_length': 12,
            'max_series': max_series,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'epochs': epochs,
            'output_dir': f"./ordinal_model_checkpoints_{exp['name']}",
            'freq': 'H',
            'n_tokens': n_tokens,
            'n_special_tokens': 2,
            'batch_size': 4,
            'num_workers': 0,
        }
        
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        train_series = prepare_data(
            dataset_name,
            max_series,
            config['context_length'],
            config['prediction_length']
        )
        
        if len(train_series) == 0:
            print(f"ERROR: No data for {dataset_name}")
            continue
        
        dataset = TimeSeriesDataset(train_series, config['context_length'], config['prediction_length'])
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f"Created DataLoader with {len(dataloader)} batches per epoch")
        
        print("Loading base model...")
        inner_model = AutoModelForSeq2SeqLM.from_pretrained(config['model_id'])
        
        chronos_config = ChronosConfig(
            tokenizer_class='MeanScaleUniformBins',
            tokenizer_kwargs={'low_limit': -15, 'high_limit': 15},
            context_length=config['context_length'],
            prediction_length=config['prediction_length'],
            n_tokens=config['n_tokens'],
            n_special_tokens=config['n_special_tokens'],
            pad_token_id=0,
            eos_token_id=1,
            use_eos_token=True,
            model_type='seq2seq',
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )
        
        model = FixedOrdinalModel(
            config=chronos_config,
            model=inner_model,
            use_ordinal_head=True
        )
        model = model.to('cpu')
        
        tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=chronos_config)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        model.train()
        best_loss = float('inf')
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nStarting training for {config['epochs']} epochs...")
        print(f"Total batches per epoch: {len(dataloader)}\n")
        
        for epoch in range(config['epochs']):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
            
            for contexts, labels in progress_bar:
                try:
                    batch_token_ids = []
                    batch_attention_masks = []
                    batch_scales = []
                    
                    for i in range(contexts.size(0)):
                        token_ids, attention_mask, scale = tokenizer.context_input_transform(contexts[i].unsqueeze(0))
                        batch_token_ids.append(token_ids)
                        batch_attention_masks.append(attention_mask)
                        batch_scales.append(scale)
                    
                    token_ids = torch.cat(batch_token_ids, dim=0)
                    attention_mask = torch.cat(batch_attention_masks, dim=0)
                    
                    optimizer.zero_grad()
                    
                    predictions = model.forward_expectation(token_ids, attention_mask, config['prediction_length'])
                    predictions = predictions[:, :config['prediction_length']].squeeze(-1)
                    
                    labels_scaled = []
                    for i in range(labels.size(0)):
                        scale_val = batch_scales[i].squeeze().detach().item()
                        if scale_val > 0:
                            label_scaled = labels[i] / scale_val
                        else:
                            label_scaled = labels[i]
                        labels_scaled.append(label_scaled)
                    labels = torch.stack(labels_scaled)
                    labels = labels[:, :config['prediction_length']]
                    
                    loss = criterion(predictions, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/num_batches:.4f}'
                    })
                    
                    del token_ids, attention_mask, predictions, labels, loss
                    gc.collect()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nWARNING: Out of memory, skipping batch...")
                        gc.collect()
                        continue
                    else:
                        raise e
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.6f}")
                
                scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_dir = output_dir / 'best_model'
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    model.model.save_pretrained(checkpoint_dir)
                    if model.ordinal_head:
                        torch.save(model.ordinal_head.state_dict(), checkpoint_dir / 'ordinal_head.pt')
                    print(f"Best model saved to {checkpoint_dir} (loss: {avg_loss:.6f})")
            
            gc.collect()
        
        final_dir = output_dir / 'final_model'
        final_dir.mkdir(exist_ok=True)
        model.model.save_pretrained(final_dir)
        if model.ordinal_head:
            torch.save(model.ordinal_head.state_dict(), final_dir / 'ordinal_head.pt')
        print(f"\nFinal model saved to {final_dir}")

if __name__ == '__main__':
    train()