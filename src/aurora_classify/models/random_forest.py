from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("RandomForestClassifier(n_estimators=75, random_state=42")

    def build(self):
        return RandomForestClassifier(n_estimators=75, random_state=42)