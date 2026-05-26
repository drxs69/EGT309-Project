"""Evaluation utilities for trained machine learning models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Evaluates classification models and saves metrics/visuals."""

    def __init__(self, visuals_dir: str = "visuals") -> None:
        self.visuals_dir = Path(visuals_dir)
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model_name: str, model, x_test, y_test) -> Dict[str, float]:
        """Calculate key classification metrics."""
        predictions = model.predict(x_test)

        return {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }

    def print_report(self, model_name: str, model, x_test, y_test) -> None:
        """Print detailed classification report for code walkthrough and checking."""
        predictions = model.predict(x_test)
        print(f"\n===== Classification Report: {model_name} =====")
        print(classification_report(y_test, predictions, zero_division=0))

    def save_confusion_matrix(self, model_name: str, model, x_test, y_test) -> None:
        """Save confusion matrix visual for each model."""
        predictions = model.predict(x_test)
        labels = sorted(pd.Series(y_test).unique())
        matrix = confusion_matrix(y_test, predictions, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")
        output_path = self.visuals_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def save_metrics(metrics: List[Dict[str, float]], output_path: str) -> pd.DataFrame:
        """Save all model metrics into a CSV file."""
        results_df = pd.DataFrame(metrics).sort_values(by="f1_macro", ascending=False)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        return results_df
