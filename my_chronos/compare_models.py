import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from chronos import ChronosPipeline, ChronosConfig, MeanScaleUniformBins
from ordinal_head import ProportionalOddsHead
import warnings
from train_ordinal import FixedOrdinalModel
from datasets import load_dataset
import yaml
import argparse
warnings.filterwarnings('ignore')

def smape(actual, forecast):
    return 100 * np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + 1e-8))

def load_data(dataset_name="m4_hourly"):
    local_path = Path(f"./data/{dataset_name}_train.parquet")
    if local_path.exists():
        print(f"Loading from local file: {local_path}")
        df = pd.read_parquet(local_path)
        return df
    
    print(f"Loading from Hugging Face datasets: {dataset_name}")
    try:
        dataset = load_dataset(
            "autogluon/chronos_datasets", 
            dataset_name, 
            split="train",
            download_mode="reuse_dataset_if_exists"
        )
        df = dataset.to_pandas()
        local_path.parent.mkdir(exist_ok=True)
        df.to_parquet(local_path)
        print(f"Saved to {local_path} for future use")
        return df
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        raise

def plot_comparison(forecast_orig, forecast_ordinal, test_vals, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(test_vals, 'o-', label='Actual', color='black', linewidth=2, markersize=8)
    axes[0].plot(forecast_orig, 's-', label='Original Chronos', color='#2E86AB', linewidth=2, markersize=6)
    axes[0].plot(forecast_ordinal, '^-', label='Ordinal Chronos', color='#A23B72', linewidth=2, markersize=6)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{title} - Forecast Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    error_orig = np.abs(forecast_orig - test_vals)
    error_ord = np.abs(forecast_ordinal - test_vals)
    
    x = np.arange(len(test_vals))
    width = 0.35
    
    axes[1].bar(x - width/2, error_orig, width, label='Original Chronos', color='#2E86AB', alpha=0.7)
    axes[1].bar(x + width/2, error_ord, width, label='Ordinal Chronos', color='#A23B72', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f'{title} - Error Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def test_model(checkpoint_path, dataset_name="m4_hourly", n_tokens=66, n_test_series=20, save_plots=True):
    device = "cpu"
    
    print("Loading original Chronos model...")
    pipeline_orig = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        dtype=torch.float32,
    )
    
    print("Loading ordinal model...")
    inner_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    
    chronos_config = ChronosConfig(
        tokenizer_class='MeanScaleUniformBins',
        tokenizer_kwargs={'low_limit': -15, 'high_limit': 15},
        context_length=64,
        prediction_length=12,
        n_tokens=n_tokens,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type='seq2seq',
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
    )
    
    model_ordinal = FixedOrdinalModel(
        config=chronos_config,
        model=inner_model,
        use_ordinal_head=True
    ).to(device)
    model_ordinal.eval()
    
    num_bins = n_tokens - 2
    model_ordinal.ordinal_head = ProportionalOddsHead(
        hidden_dim=inner_model.config.d_model,
        num_bins=num_bins,
    ).to(device)
    
    ordinal_head_path = f"{checkpoint_path}/ordinal_head.pt"
    if Path(ordinal_head_path).exists():
        model_ordinal.ordinal_head.load_state_dict(torch.load(ordinal_head_path, map_location=device))
        print("Ordinal head loaded successfully")
    else:
        print(f"Warning: ordinal_head.pt not found")
    
    tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=chronos_config)
    
    print(f"Loading dataset: {dataset_name}")
    df = load_data(dataset_name)
    
    context_length = 64
    prediction_length = 12
    
    test_series_list = []
    for series_id in df['id'].unique()[:n_test_series]:
        series = df[df['id'] == series_id]['target'].values[0]
        if isinstance(series, np.ndarray):
            series = series.tolist()
        if len(series) > context_length + prediction_length:
            series = series[:context_length + prediction_length]
            test_series_list.append(series)
    
    print(f"Testing on {len(test_series_list)} series")
    
    results_orig = []
    results_ordinal = []
    centers = np.linspace(-15, 15, num_bins)
    
    all_forecasts_orig = []
    all_forecasts_ord = []
    all_test_vals = []
    
    for idx, series in enumerate(test_series_list):
        context_vals = series[:-prediction_length]
        test_vals = np.array(series[-prediction_length:])
        
        with torch.no_grad():
            forecast_orig = pipeline_orig.predict(
                torch.tensor(context_vals, dtype=torch.float32),
                prediction_length=prediction_length,
                num_samples=20
            )
            forecast_np = forecast_orig.numpy()
            if forecast_np.ndim == 3:
                forecast_np = forecast_np[0]
            median_orig = np.median(forecast_np, axis=0)
        
        with torch.no_grad():
            context_tensor = torch.tensor(context_vals, dtype=torch.float32).unsqueeze(0).to(device)
            token_ids, attention_mask, scale = tokenizer.context_input_transform(context_tensor)
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            probs = model_ordinal.forward_ordinal(token_ids, attention_mask, prediction_length)
            probs_np = probs.squeeze().cpu().numpy()
            
            forecast_normalized = np.sum(probs_np * centers.reshape(1, -1), axis=1)
            scale_val = scale.squeeze().cpu().numpy()
            forecast_ordinal = forecast_normalized * scale_val
            
            all_forecasts_orig.append(median_orig)
            all_forecasts_ord.append(forecast_ordinal)
            all_test_vals.append(test_vals)
            
            if idx == 0:
                print(f"\nDebug first series:")
                print(f"  Scale from tokenizer: {scale_val:.4f}")
                print(f"  Forecast normalized: {forecast_normalized[:5]}")
                print(f"  Forecast ordinal: {forecast_ordinal[:5]}")
                print(f"  Test values (first 5): {test_vals[:5]}")
                print(f"  Original forecast: {median_orig[:5]}")
        
        results_orig.append({
            'mae': mean_absolute_error(test_vals, median_orig),
            'smape': smape(test_vals, median_orig),
        })
        
        results_ordinal.append({
            'mae': mean_absolute_error(test_vals, forecast_ordinal),
            'smape': smape(test_vals, forecast_ordinal),
        })
        
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(test_series_list)} series")
    
    avg_mae_orig = np.mean([r['mae'] for r in results_orig])
    avg_smape_orig = np.mean([r['smape'] for r in results_orig])
    avg_mae_ord = np.mean([r['mae'] for r in results_ordinal])
    avg_smape_ord = np.mean([r['smape'] for r in results_ordinal])
    
    print(f"\n{'Model':<20} {'MAE':<12} {'sMAPE (%)':<12}")
    print(f"{'Original Chronos':<20} {avg_mae_orig:<12.3f} {avg_smape_orig:<12.2f}")
    print(f"{'Ordinal Chronos':<20} {avg_mae_ord:<12.3f} {avg_smape_ord:<12.2f}")
    
    improvement_mae = ((avg_mae_orig - avg_mae_ord) / avg_mae_orig) * 100
    improvement_smape = ((avg_smape_orig - avg_smape_ord) / avg_smape_orig) * 100
    
    print(f"\n{'Improvement':<20} {improvement_mae:+.1f}% {improvement_smape:+.1f}%")
    
    if save_plots:
        plots_dir = Path("./comparison_plots")
        plots_dir.mkdir(exist_ok=True)
        
        for i in range(min(4, len(all_test_vals))):
            title = f"{dataset_name} - Series {i+1}"
            save_path = plots_dir / f"{dataset_name}_series_{i+1}.png"
            plot_comparison(
                all_forecasts_orig[i],
                all_forecasts_ord[i],
                all_test_vals[i],
                title,
                save_path
            )
        
        all_forecasts_orig_flat = np.concatenate(all_forecasts_orig)
        all_forecasts_ord_flat = np.concatenate(all_forecasts_ord)
        all_test_vals_flat = np.concatenate(all_test_vals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(all_test_vals_flat, all_forecasts_orig_flat, alpha=0.5, label='Original Chronos', color='#2E86AB')
        axes[0].scatter(all_test_vals_flat, all_forecasts_ord_flat, alpha=0.5, label='Ordinal Chronos', color='#A23B72')
        axes[0].plot([all_test_vals_flat.min(), all_test_vals_flat.max()], 
                    [all_test_vals_flat.min(), all_test_vals_flat.max()], 
                    'k--', label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{dataset_name} - Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        errors_orig = np.abs(all_forecasts_orig_flat - all_test_vals_flat)
        errors_ord = np.abs(all_forecasts_ord_flat - all_test_vals_flat)
        
        axes[1].boxplot([errors_orig, errors_ord], labels=['Original Chronos', 'Ordinal Chronos'])
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title(f'{dataset_name} - Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = plots_dir / f"{dataset_name}_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to {save_path}")
        plt.show()
    
    return {
        'model': checkpoint_path,
        'dataset': dataset_name,
        'mae_orig': avg_mae_orig,
        'mae_ord': avg_mae_ord,
        'smape_orig': avg_smape_orig,
        'smape_ord': avg_smape_ord,
        'improvement_mae': improvement_mae,
        'improvement_smape': improvement_smape,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving plots")
    args = parser.parse_args()
    
    with open('config_experiments.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    results = []
    save_plots = not args.no_plots
    
    if args.experiment:
        exp = next((e for e in configs['experiments'] if e['name'] == args.experiment), None)
        if exp:
            checkpoint = f"./ordinal_model_checkpoints_{exp['name']}/best_model"
            if Path(checkpoint).exists():
                result = test_model(checkpoint, exp['dataset'], exp['n_tokens'], save_plots=save_plots)
                results.append(result)
            else:
                print(f"Checkpoint not found: {checkpoint}")
    elif args.all:
        for exp in configs['experiments']:
            checkpoint = f"./ordinal_model_checkpoints_{exp['name']}/best_model"
            if Path(checkpoint).exists():
                result = test_model(checkpoint, exp['dataset'], exp['n_tokens'], save_plots=save_plots)
                results.append(result)
            else:
                print(f"Checkpoint not found: {checkpoint}")
    else:
        checkpoint = "./ordinal_model_checkpoints_final/best_model"
        if Path(checkpoint).exists():
            result = test_model(checkpoint, "m4_hourly", 66, save_plots=save_plots)
            results.append(result)
        else:
            print("Checkpoint not found. Use --experiment or --all")
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        if len(results) > 1 and save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            exp_names = [r['dataset'] for r in results]
            mae_orig = [r['mae_orig'] for r in results]
            mae_ord = [r['mae_ord'] for r in results]
            smape_orig = [r['smape_orig'] for r in results]
            smape_ord = [r['smape_ord'] for r in results]
            
            x = np.arange(len(exp_names))
            width = 0.35
            
            axes[0].bar(x - width/2, mae_orig, width, label='Original Chronos', color='#2E86AB')
            axes[0].bar(x + width/2, mae_ord, width, label='Ordinal Chronos', color='#A23B72')
            axes[0].set_xlabel('Dataset')
            axes[0].set_ylabel('MAE')
            axes[0].set_title('MAE Comparison Across Datasets')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(exp_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].bar(x - width/2, smape_orig, width, label='Original Chronos', color='#2E86AB')
            axes[1].bar(x + width/2, smape_ord, width, label='Ordinal Chronos', color='#A23B72')
            axes[1].set_xlabel('Dataset')
            axes[1].set_ylabel('sMAPE (%)')
            axes[1].set_title('sMAPE Comparison Across Datasets')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(exp_names, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = Path("./comparison_plots") / "all_experiments_summary.png"
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
            plt.show()

if __name__ == '__main__':
    main()