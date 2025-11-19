import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loader import load_dataset
from ..data.feature_engineering import create_tier_category
from ..data.preprocessor import DataPreprocessor
from ..models.random_forest import RandomForestModel
from ..models.knn import KNNModel
from ..models.decision_tree import DecisionTreeModel
from ..models.naive_bayes import NaiveBayesModel
from ..models.svm import SVMModel
from ..utils.metrics import plot_knn_accuracy
from .evaluator import compare_models

class ModelTrainer:
    """Orchestrates the full training & evaluation pipeline."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = [
            RandomForestModel(),
            KNNModel(n_neighbors=9),
            DecisionTreeModel(),
            NaiveBayesModel(),
            SVMModel()
        ]
        self.results = []

    def run_experiments(self):
        # 1. Load data
        df = load_dataset()
        
        # 2. Feature engineering
        df = create_tier_category(df, low_thresh=700, high_thresh=1300)
        
        feature_cols = [col for col in df.columns if col != "price_euros"]
        X = df[feature_cols]
        y = df["tier_category"]

        # 3. Preprocessing
        X_processed = self.preprocessor.fit_transform(X)

        # 4. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training multiple models...\n")
        for model in self.models:
            print(f"Training {model.name}...")
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            self.results.append(metrics)
            print(f"{model.name} Accuracy: {metrics['accuracy']:.4f}")

        # 5. Show comparison
        comparison = compare_models(self.results)
        print("\n=== Model Comparison ===")
        print(comparison[['model', 'accuracy', 'f1']])

        # 6. Save best model & preprocessor
        best_model = self.models[self.results.index(max(self.results, key=lambda x: x['accuracy']))]
        joblib.dump(best_model.model, "models/best_random_forest.joblib")
        self.preprocessor.save("models/preprocessor.joblib")

        print("\nBest model and preprocessor saved to models/")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_experiments()