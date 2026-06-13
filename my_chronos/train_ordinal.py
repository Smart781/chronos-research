import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from chronos import ChronosConfig, ChronosModel, MeanScaleUniformBins
from ordinal_head import ProportionalOddsHead, OrdinalCrossEntropyLoss
import warnings
import numpy as np
from datasets import load_dataset
import gc
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use-ordinal-head", action="store_true")
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()

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
        context = torch.tensor(context, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return context, label

def load_and_prepare_data(config):
    print(f"Loading dataset: {config['dataset']}")
    dataset = load_dataset("autogluon/chronos_datasets", config['dataset'], split="train")
    df = dataset.to_pandas()
    max_series = config.get('max_series', 30)
    series_ids = df['id'].unique()[:max_series]
    train_data = []
    required_len = config['context_length'] + config['prediction_length']
    for series_id in tqdm(series_ids, desc="Loading series"):
        series = df[df['id'] == series_id]['target'].values[0]
        if isinstance(series, np.ndarray):
            series = series.tolist()
        if len(series) > required_len:
            series = series[:required_len]
            train_data.append(series)
    print(f"Prepared {len(train_data)} series")
    return train_data

class OrdinalChronosModel(ChronosModel):
    def __init__(self, config, model, use_ordinal_head=False):
        super().__init__(config, model)
        self.use_ordinal_head = use_ordinal_head
        self.ordinal_head = None
        self.model.config.use_cache = True
        self.model.config.output_attentions = False
        self.model.config.output_hidden_states = False
    
    def _init_ordinal_head(self):
        if self.ordinal_head is None and self.use_ordinal_head:
            num_bins = self.config.n_tokens - self.config.n_special_tokens
            self.ordinal_head = ProportionalOddsHead(
                hidden_dim=self.model.config.d_model,
                num_bins=num_bins,
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
            
            next_token = torch.argmax(probs_last, dim=-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            del decoder_outputs
        
        return torch.stack(all_probs, dim=1)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        if self.use_ordinal_head:
            return self.forward_ordinal(input_ids, attention_mask, kwargs.get('prediction_length', 12))
        return super().forward(input_ids, attention_mask, **kwargs)

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

def train():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.num_epochs:
        config['epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    config.setdefault('n_tokens', 32)
    config.setdefault('n_special_tokens', 2)
    config.setdefault('batch_size', 1)
    config.setdefault('num_workers', 0)
    config.setdefault('max_series', 20)
    config.setdefault('context_length', 64)
    config.setdefault('prediction_length', 12)
    
    print("="*60)
    print("CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  use_ordinal_head: {args.use_ordinal_head}")
    print(f"  device: CPU (using {mp.cpu_count()} cores)")
    print("="*60)
    
    train_series = load_and_prepare_data(config)
    if len(train_series) == 0:
        print("ERROR: No training data! Exiting.")
        return
    
    dataset = TimeSeriesDataset(train_series, config['context_length'], config['prediction_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,
        collate_fn=collate_fn
    )
    print(f"Created DataLoader with {len(dataloader)} batches per epoch")
    print("\nLoading base model...")
    
    inner_model = AutoModelForSeq2SeqLM.from_pretrained(config['model_id'], low_cpu_mem_usage=True)
    
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
    
    model = OrdinalChronosModel(
        config=chronos_config,
        model=inner_model,
        use_ordinal_head=args.use_ordinal_head
    )
    model = model.to('cpu')
    
    num_bins = config['n_tokens'] - config['n_special_tokens']
    criterion = OrdinalCrossEntropyLoss(distance_weight=args.distance_weight)
    print(f"\nUsing OrdinalCrossEntropyLoss with {num_bins} bins")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=chronos_config)
    model.train()
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Total batches per epoch: {len(dataloader)}\n")
    
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        total_loss = 0
        num_batches = 0
        print(f"{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (contexts, labels) in enumerate(progress_bar):
            try:
                batch_token_ids = []
                batch_attention_masks = []
                batch_scales = []
                
                for i in range(contexts.size(0)):
                    context = contexts[i]
                    token_ids, attention_mask, scale = tokenizer.context_input_transform(context.unsqueeze(0))
                    batch_token_ids.append(token_ids)
                    batch_attention_masks.append(attention_mask)
                    batch_scales.append(scale)
                
                token_ids = torch.cat(batch_token_ids, dim=0)
                attention_mask = torch.cat(batch_attention_masks, dim=0)
                
                batch_label_ids = []
                for i in range(labels.size(0)):
                    label = labels[i]
                    label_ids, _ = tokenizer.label_input_transform(label.unsqueeze(0), batch_scales[i])
                    batch_label_ids.append(label_ids)
                
                label_ids = torch.cat(batch_label_ids, dim=0)
                
                optimizer.zero_grad()
                
                logits = model.forward_ordinal(token_ids, attention_mask, config['prediction_length'])
                logits = logits[:, :config['prediction_length'], :]
                label_ids_for_loss = label_ids.unsqueeze(1)[:, :config['prediction_length'], :]
                loss = criterion(logits, label_ids_for_loss, num_bins)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
                
                del token_ids, attention_mask, label_ids, logits, loss
                
                if batch_idx % 5 == 0:
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nWARNING: Out of memory at batch {batch_idx}, skipping...")
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.6f}")
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_dir = Path(config['output_dir']) / 'best_model'
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                model.model.save_pretrained(checkpoint_dir)
                if args.use_ordinal_head and model.ordinal_head:
                    torch.save(model.ordinal_head.state_dict(), checkpoint_dir / 'ordinal_head.pt')
                print(f"Best model saved to {checkpoint_dir}")
            
            if (epoch + 1) % 10 == 0:
                checkpoint_dir = Path(config['output_dir']) / f'checkpoint_epoch_{epoch+1}'
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                model.model.save_pretrained(checkpoint_dir)
                if args.use_ordinal_head and model.ordinal_head:
                    torch.save(model.ordinal_head.state_dict(), checkpoint_dir / 'ordinal_head.pt')
                print(f"Checkpoint saved to {checkpoint_dir}\n")
        
        gc.collect()
    
    output_dir = Path(config['output_dir']) / 'final_model'
    output_dir.mkdir(exist_ok=True, parents=True)
    model.model.save_pretrained(output_dir)
    if args.use_ordinal_head and model.ordinal_head:
        torch.save(model.ordinal_head.state_dict(), output_dir / 'ordinal_head.pt')
    print(f"\nFinal model saved to {output_dir}")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()