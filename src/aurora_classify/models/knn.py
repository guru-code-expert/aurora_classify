from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors: int = 9):
        super().__init__(f"KNN(k={n_neighbors})")
        self.n_neighbors = n_neighbors

    def build(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)