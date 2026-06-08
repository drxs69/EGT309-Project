"""
model_evaluation.py
-------------------
Evaluates trained models on the held-out test set and produces:
  - Classification report (precision, recall, F1 per class)
  - Overall accuracy
  - Confusion matrix
  - Feature importance ranking (for tree-based models)
  - Comparative summary table

Metric choice justification
----------------------------
The dataset is imbalanced (Low Activity ~52 %, Moderate ~28 %,
High ~11 %).  Accuracy alone can be misleading; therefore we also
report **macro-averaged F1** which treats all three classes equally and
is informative in an elder-care context where missing a High Activity
episode (or a sudden drop to inactivity) carries real health risk.

Recall for High Activity is highlighted specifically: a false negative
(missing a distress episode) is more costly than a false positive.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from config import MODEL_DIR


def evaluate_model(model, X_test, y_test, class_names: list) -> dict:
    """
    Generate evaluation metrics for one model.

    Returns a dict with accuracy, macro_f1, report string, and
    confusion matrix array.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mac_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy":   acc,
        "macro_f1":   mac_f1,
        "report":     report,
        "confusion":  cm,
        "y_pred":     y_pred,
    }


def evaluate_all(results: dict, X_test, y_test, label_encoder) -> dict:
    """Evaluate all trained models and print a comparison table."""
    class_names = list(label_encoder.classes_)
    eval_results = {}

    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    rows = []
    for name, info in results.items():
        ev = evaluate_model(info["model"], X_test, y_test, class_names)
        eval_results[name] = ev
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {ev['accuracy']:.4f}")
        print(f"  Macro F1 : {ev['macro_f1']:.4f}")
        print(ev["report"])
        rows.append({
            "Model":    name,
            "Accuracy": round(ev["accuracy"], 4),
            "Macro F1": round(ev["macro_f1"],  4),
        })

    summary = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    print("\n--- Summary ---")
    print(summary.to_string(index=False))
    return eval_results


def plot_confusion_matrices(eval_results: dict, label_encoder, output_dir: str) -> None:
    """Save a confusion-matrix heatmap for every model."""
    class_names = list(label_encoder.classes_)
    os.makedirs(output_dir, exist_ok=True)

    for name, ev in eval_results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            ev["confusion"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix – {name}\n(Accuracy={ev['accuracy']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        path = os.path.join(output_dir, f"cm_{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Evaluation] Saved confusion matrix → {path}")


def plot_feature_importances(
    results: dict,
    feature_names: list,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    Plot top-N feature importances for tree-based models.
    Logistic Regression uses absolute coefficient magnitudes.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, info in results.items():
        model = info["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = f"Feature Importances – {name}"
        elif hasattr(model, "coef_"):
            # For multi-class LR take mean of absolute values across classes
            importances = np.abs(model.coef_).mean(axis=0)
            title = f"Mean |Coefficient| – {name}"
        else:
            continue

        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = os.path.join(output_dir, f"fi_{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Evaluation] Saved feature importance plot → {path}")
