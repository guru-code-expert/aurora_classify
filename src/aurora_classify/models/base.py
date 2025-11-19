from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def build(self):
        pass

    def fit(self, X_train, y_train):
        self.model = self.build()
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> dict:
        preds = self.model.predict(X_test)
        return {
            "model": self.name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='weighted'),
            "recall": recall_score(y_test, preds, average='weighted'),
            "f1": f1_score(y_test, preds, average='weighted'),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }