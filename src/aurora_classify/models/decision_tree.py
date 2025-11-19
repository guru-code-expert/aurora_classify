from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__("DecisionTree")

    def build(self):
        return DecisionTreeClassifier(max_depth=11, random_state=42)