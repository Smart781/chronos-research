from datasets import load_dataset
import pandas as pd
from pathlib import Path

datasets = [
    "m4_hourly",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
]

Path("./data").mkdir(exist_ok=True)

for dataset_name in datasets:
    print(f"\nDownloading {dataset_name}...")
    try:
        dataset = load_dataset("autogluon/chronos_datasets", dataset_name, split="train")
        df = dataset.to_pandas()
        local_path = Path(f"./data/{dataset_name}_train.parquet")
        df.to_parquet(local_path)
        print(f"Saved to {local_path}")
        print(f"  Rows: {len(df)}, Series: {df['id'].nunique()}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")