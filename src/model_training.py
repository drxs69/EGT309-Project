"""
model_training.py
-----------------
Defines, trains, and cross-validates the three chosen models:

1. Random Forest Classifier
   - Ensemble of decision trees; handles non-linear interactions and
     high-cardinality feature spaces well.  Naturally provides feature
     importances.  Robust to outliers and does not require feature
     scaling, but we scale anyway for consistency with LR.

2. Gradient Boosting Classifier (sklearn GBM)
   - Sequential boosting typically outperforms RF on tabular data by
     correcting residual errors.  Provides its own feature importances
     and generally produces the best predictive accuracy on structured
     data of this kind.

3. Logistic Regression (multinomial)
   - Simple, interpretable linear baseline.  Fast to train and
     provides calibrated probability estimates.  Useful for
     benchmarking and explaining coefficients.

All three models are well-suited to multi-class classification.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from config import RF_PARAMS, GB_PARAMS, LR_PARAMS, CV_FOLDS, MODEL_DIR


def build_models() -> dict:
    """Instantiate all three classifiers with configured hyperparameters."""
    return {
        "RandomForest":       RandomForestClassifier(**RF_PARAMS),
        "GradientBoosting":   GradientBoostingClassifier(**GB_PARAMS),
        "LogisticRegression": LogisticRegression(**LR_PARAMS),
    }


def train_and_evaluate(
    X_train, y_train, models: dict, cv_folds: int = CV_FOLDS
) -> dict:
    """
    Train each model and compute cross-validated accuracy on the
    training set.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    models  : dict  {name: estimator}
    cv_folds: int

    Returns
    -------
    dict  {name: {'model': fitted_estimator, 'cv_scores': np.ndarray}}
    """
    results = {}
    for name, model in models.items():
        print(f"\n[Training] Fitting {name} …")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1
        )
        mean_cv = cv_scores.mean()
        std_cv  = cv_scores.std()
        print(f"[Training] {name} CV accuracy: {mean_cv:.4f} ± {std_cv:.4f}")
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
