import requests
import pandas as pd
from pathlib import Path

url = "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/m4_hourly/train-00000-of-00001.parquet"
Path("./data").mkdir(exist_ok=True)

response = requests.get(url, stream=True)
response.raise_for_status()

with open("./data/m4_hourly_train.parquet", 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)