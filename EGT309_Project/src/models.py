"""Model definitions for the ElderGuard activity prediction task."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """Creates machine learning models used in Week 7 training."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def create_models(self) -> Dict[str, Any]:
        """Return at least three models for fair comparison."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=1,
            ),
        }
