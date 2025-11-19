import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import os

from ..config import PROJECT_ROOT, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class DataPreprocessor:
    """Handles cleaning, encoding and scaling of the dataset."""
    
    def __init__(self):
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def clean_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove any non-numeric suffix from weight column and convert to float."""
        df = df.copy()
        df["weight_kg"] = df["weight_kg"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
        return df

    def fit(self, df: pd.DataFrame):
        df = self.clean_weight(df)
        
        cat_data = df[CATEGORICAL_FEATURES]
        self.cat_encoder.fit(cat_data)
        
        num_data = df[NUMERICAL_FEATURES]
        self.scaler.fit(num_data)
        
        self.is_fitted = True

    def transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = self.clean_weight(df)
        
        cat_encoded = self.cat_encoder.transform(df[CATEGORICAL_FEATURES])
        cat_df = pd.DataFrame(cat_encoded, columns=CATEGORICAL_FEATURES, index=df.index)
        
        num_scaled = self.scaler.transform(df[NUMERICAL_FEATURES])
        num_df = pd.DataFrame(num_scaled, columns=NUMERICAL_FEATURES, index=df.index)
        
        processed = pd.concat([cat_df, num_df], axis=1)
        return processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, fit=True)

    def save(self, path: os.PathLike = PROJECT_ROOT / "models" / "preprocessor.joblib"):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: os.PathLike):
        return joblib.load(path)