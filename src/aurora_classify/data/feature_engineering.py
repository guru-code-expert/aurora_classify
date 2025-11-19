import numpy as np
import pandas as pd

def create_tier_category(df: pd.DataFrame, low_thresh: float = 700.0, high_thresh: float = 1300.0) -> pd.DataFrame:
    """
    Create a three-level tier category based on price.

    Categories:
        0 → Budget
        1 → Mid-range
        2 → Premium
    """
    conditions = [
        df["price_euros"] < low_thresh,
        df["price_euros"].between(low_thresh, high_thresh),
    ]
    choices = [0, 1]  # Budget, Mid-range
    df["tier_category"] = np.select(conditions, choices, default=2)  # Premium
    return df