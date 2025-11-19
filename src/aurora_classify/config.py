from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "product_specs.csv"

# Modeling
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "tier_category"

# Feature columns (anonymized)
CATEGORICAL_FEATURES = [
    "brand", "model_name", "device_type", "display_spec", 
    "processor", "ram_config", "storage_config", "graphics", "os"
]
NUMERICAL_FEATURES = ["weight_kg"]