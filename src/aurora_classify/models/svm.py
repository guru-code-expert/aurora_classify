from sklearn.svm import SVC
from .base import BaseModel

class SVMModel(BaseModel):
    def __init__(self):
        super().__init__("SVM")

    def build(self):
        return SVC(random_state=42)