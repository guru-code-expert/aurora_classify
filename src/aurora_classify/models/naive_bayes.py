from sklearn.naive_bayes import GaussianNB
from .base import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__("NaiveBayes")

    def build(self):
        return GaussianNB()