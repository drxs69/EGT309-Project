"""
model_training.py
-----------------
Defines and trains the three required models:
1. Random Forest
2. Gradient Boosting
3. Logistic Regression

Accuracy improvement notes:
- Random Forest no longer uses class_weight="balanced" because that reduced
  overall accuracy on this imbalanced dataset.
- Random Forest is allowed full-depth trees and uses sqrt feature sampling.
- Gradient Boosting uses a moderate number of boosting rounds to balance
  accuracy and runtime.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from config import RF_PARAMS, GB_PARAMS, LR_PARAMS, CV_FOLDS, MODEL_DIR


def build_models() -> dict:
    """Instantiate only the three models required by the project brief."""
    return {
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
        "GradientBoosting": GradientBoostingClassifier(**GB_PARAMS),
        "LogisticRegression": LogisticRegression(**LR_PARAMS),
    }


def train_and_evaluate(X_train, y_train, models: dict, cv_folds: int = CV_FOLDS) -> dict:
    """
    Train each model and optionally compute cross-validated accuracy.

    If cv_folds is 0 or 1, cross-validation is skipped to keep the pipeline
    quick. Final accuracy is still calculated later on the held-out test set.
    """
    results = {}
    for name, model in models.items():
        print(f"\n[Training] Fitting {name} …")
        model.fit(X_train, y_train)

        if cv_folds and cv_folds > 1:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=1,
            )
            mean_cv = cv_scores.mean()
            std_cv = cv_scores.std()
            print(f"[Training] {name} CV accuracy: {mean_cv:.4f} ± {std_cv:.4f}")
        else:
            cv_scores = np.array([])
            print(f"[Training] {name} fitted. CV skipped; using held-out test evaluation.")

        results[name] = {"model": model, "cv_scores": cv_scores}

    return results


def save_models(results: dict, model_dir: str = MODEL_DIR) -> None:
    """Persist each fitted model to disk using joblib."""
    os.makedirs(model_dir, exist_ok=True)
    for name, info in results.items():
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(info["model"], path)
        print(f"[Training] Saved {name} → {path}")


def load_model(name: str, model_dir: str = MODEL_DIR):
    """Load a previously saved model by name."""
    path = os.path.join(model_dir, f"{name}.joblib")
    return joblib.load(path)
