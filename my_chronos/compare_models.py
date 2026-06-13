import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from chronos import ChronosPipeline, ChronosConfig, MeanScaleUniformBins
from ordinal_head import ProportionalOddsHead
import warnings
from train_ordinal import OrdinalChronosModel
warnings.filterwarnings('ignore')

def smape(actual, forecast):
    return 100 * np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + 1e-8))

device = "cpu"
print(f"Device: {device}")

pipeline_orig = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device,
    torch_dtype=torch.float32,
)

checkpoint_path = "./ordinal_model_checkpoints_fixed/best_model"

inner_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

chronos_config = ChronosConfig(
    tokenizer_class='MeanScaleUniformBins',
    tokenizer_kwargs={'low_limit': -15, 'high_limit': 15},
    context_length=64,
    prediction_length=12,
    n_tokens=32,
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

model_ordinal = OrdinalChronosModel(
    config=chronos_config,
    model=inner_model,
    use_ordinal_head=True
).to(device)
model_ordinal.eval()

num_bins = 30
model_ordinal.ordinal_head = ProportionalOddsHead(
    hidden_dim=inner_model.config.d_model,
    num_bins=num_bins,
).to(device)

ordinal_head_path = f"{checkpoint_path}/ordinal_head.pt"
model_ordinal.ordinal_head.load_state_dict(torch.load(ordinal_head_path, map_location=device))
print("Ordinal head loaded successfully")

tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=chronos_config)

dataset = load_dataset("autogluon/chronos_datasets", "m4_hourly", split="train")
df = dataset.to_pandas()

context_length = 64
prediction_length = 12
n_test_series = 20

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
        pred_tokens = torch.argmax(probs, dim=-1)
        pred_tokens_flat = pred_tokens.squeeze().cpu().numpy()
        scale_val = scale.squeeze().cpu().numpy()
        
        centers = np.linspace(-15, 15, num_bins)
        forecast_ordinal = centers[pred_tokens_flat] * scale_val
    
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
    
    if idx == 0:
        print(f"\nDebug first series:")
        print(f"  Test values (first 5): {test_vals[:5]}")
        print(f"  Ordinal forecast (first 5): {forecast_ordinal[:5]}")
        print(f"  Original forecast (first 5): {median_orig[:5]}")

avg_mae_orig = np.mean([r['mae'] for r in results_orig])
avg_smape_orig = np.mean([r['smape'] for r in results_orig])

avg_mae_ord = np.mean([r['mae'] for r in results_ordinal])
avg_smape_ord = np.mean([r['smape'] for r in results_ordinal])

print(f"\n{'Model':<20} {'MAE':<12} {'sMAPE (%)':<12}")
print("-" * 45)
print(f"{'Original Chronos':<20} {avg_mae_orig:<12.3f} {avg_smape_orig:<12.2f}")
print(f"{'Ordinal Chronos':<20} {avg_mae_ord:<12.3f} {avg_smape_ord:<12.2f}")

improvement_mae = ((avg_mae_orig - avg_mae_ord) / avg_mae_orig) * 100
improvement_smape = ((avg_smape_orig - avg_smape_ord) / avg_smape_orig) * 100

print(f"\n{'Improvement':<20} {improvement_mae:+.1f}% {improvement_smape:+.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['Original', 'Ordinal']
mae_values = [avg_mae_orig, avg_mae_ord]
smape_values = [avg_smape_orig, avg_smape_ord]

axes[0].bar(models, mae_values, color=['#2E86AB', '#A23B72'])
axes[0].set_ylabel('MAE')
axes[0].set_title('Mean Absolute Error')
for i, v in enumerate(mae_values):
    axes[0].text(i, v + 5, f'{v:.1f}', ha='center')

axes[1].bar(models, smape_values, color=['#2E86AB', '#A23B72'])
axes[1].set_ylabel('sMAPE (%)')
axes[1].set_title('sMAPE')
for i, v in enumerate(smape_values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
plt.show()