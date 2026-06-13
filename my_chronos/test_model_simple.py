import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from chronos import ChronosConfig, MeanScaleUniformBins
from train_ordinal import OptimizedChronosModel, SimpleHead
import warnings
warnings.filterwarnings('ignore')

device = "cpu"
checkpoint_path = "./ordinal_model_checkpoints_medium/best_model"

print("="*60)
print("MODEL ADEQUACY CHECK")
print("="*60)

# Загрузка модели
inner_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

chronos_config = ChronosConfig(
    tokenizer_class='MeanScaleUniformBins',
    tokenizer_kwargs={'low_limit': -15, 'high_limit': 15},
    context_length=128,
    prediction_length=24,
    n_tokens=64,
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

model_ordinal = OptimizedChronosModel(
    config=chronos_config,
    model=inner_model,
    use_ordinal_head=True
).to(device)
model_ordinal.eval()

# Загрузка головы
model_ordinal.value_head = SimpleHead(hidden_dim=inner_model.config.d_model).to(device)
model_ordinal.value_head.load_state_dict(torch.load(f"{checkpoint_path}/value_head.pt", map_location=device))
print("✓ Model loaded")

tokenizer = MeanScaleUniformBins(low_limit=-15, high_limit=15, config=chronos_config)

# Тестовые данные
test_context = [350.0, 355.0, 360.0, 365.0, 370.0, 375.0, 380.0, 385.0]
print(f"\nTest context: {test_context}")

# Прогноз
with torch.no_grad():
    context_tensor = torch.tensor(test_context, dtype=torch.float32).unsqueeze(0).to(device)
    token_ids, attention_mask, scale = tokenizer.context_input_transform(context_tensor)
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    predictions = model_ordinal.forward_value(token_ids, attention_mask, 5)
    predictions = predictions.squeeze(-1).cpu().numpy()[0]
    scale_val = scale.squeeze().cpu().numpy()
    forecast = predictions * scale_val

print(f"Predictions (raw): {predictions[:5]}")
print(f"Scale: {scale_val}")
print(f"Forecast: {forecast[:5]}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if np.std(forecast) < 1:
    print("❌ Model outputs constant values - NOT LEARNING!")
    print(f"   All predictions = {forecast[0]:.2f}")
    print(f"   Expected range: 350-400")
    
    print("\nPossible reasons:")
    print("1. Learning rate too high (0.001)")
    print("2. Model architecture too simple")
    print("3. Need to train on scaled data differently")
    print("4. Gradient vanishing/exploding")
    
    print("\nSuggestions:")
    print("1. Try learning_rate = 0.0001")
    print("2. Add layer normalization")
    print("3. Use different loss function")
    print("4. Train on normalized data")
    
else:
    print("✓ Model shows variation - might be learning")
    print(f"   Prediction range: {forecast.min():.2f} - {forecast.max():.2f}")
    print(f"   Std deviation: {np.std(forecast):.2f}")