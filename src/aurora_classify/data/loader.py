import pandas as pd
from pathlib import Path
from typing import Tuple

from ..config import DATA_PATH


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw product specification dataset.

    Expected columns (anonymized):
    - brand, model_name, device_type, display_spec, processor,
      ram_config, storage_config, graphics, os, weight_kg, price_euros

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place your CSV in data/raw/.")
    
    df = pd.read_csv(path, encoding="utf-8")
    return df