from typing import List, Dict
import pandas as pd

def compare_models(results: List[Dict]) -> pd.DataFrame:
    """Convert list of result dicts into a nice comparison table."""
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)
    return df