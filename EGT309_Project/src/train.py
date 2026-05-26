"""Training pipeline for Week 7 model training, tuning, and model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from evaluate import ModelEvaluator
from models import ModelFactory
from preprocessing import DataCleaner, PreprocessorFactory


class ModelTrainer:
    """Handles model training, hyperparameter tuning, evaluation, and saving outputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.target_column = config["columns"]["target"]
        self.drop_columns = config["columns"].get("drop_columns", [])
        self.training_config = config["training"]
        self.outputs = config["outputs"]

    def prepare_data(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Clean data and split it into train/test sets."""
        cleaner = DataCleaner(target_column=self.target_column, drop_columns=self.drop_columns)
        cleaned_df = cleaner.clean(dataframe)
        features, target = cleaner.split_features_target(cleaned_df)

        return train_test_split(
            features,
            target,
            test_size=self.training_config["test_size"],
            random_state=self.training_config["random_state"],
            stratify=target,
        )

    def train_all_models(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Train, tune, evaluate, and save the best model."""
        results_dir = Path(self.outputs["results_dir"])
        visuals_dir = Path(self.outputs["visuals_dir"])
        saved_model_dir = Path(self.outputs["saved_model_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        saved_model_dir.mkdir(parents=True, exist_ok=True)

        x_train, x_test, y_train, y_test = self.prepare_data(dataframe)
        preprocessor = PreprocessorFactory.create_preprocessor(x_train)
        models = ModelFactory(random_state=self.training_config["random_state"]).create_models()
        evaluator = ModelEvaluator(visuals_dir=str(visuals_dir))

        metrics = []
        fitted_models = {}

        for model_name, model in models.items():
            config_key = model_name.lower().replace(" ", "_")
            model_config = self.config["models"].get(config_key, {})
            if not model_config.get("enabled", True):
                continue

            print(f"\n==============================")
            print(f"MODEL TEST: {model_name}")
            print(f"==============================")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            param_grid = model_config.get("params", {})
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=self.training_config["scoring"],
                cv=self.training_config["cv_folds"],
                n_jobs=1,
            )
            grid_search.fit(x_train, y_train)

            best_pipeline = grid_search.best_estimator_
            fitted_models[model_name] = best_pipeline

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")

            model_metrics = evaluator.evaluate(model_name, best_pipeline, x_test, y_test)
            model_metrics["best_cv_score"] = grid_search.best_score_
            metrics.append(model_metrics)

            evaluator.print_report(model_name, best_pipeline, x_test, y_test)
            evaluator.save_confusion_matrix(model_name, best_pipeline, x_test, y_test)

        results_df = evaluator.save_metrics(metrics, self.outputs["model_results"])
        best_model_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_model_name]

        joblib.dump(best_model, self.outputs["best_model"])
        print(f"\nBest model saved: {best_model_name} -> {self.outputs['best_model']}")

        self.save_feature_importance(best_model_name, best_model, x_test, y_test)
        return results_df

    def save_feature_importance(self, model_name: str, model, x_test, y_test) -> None:
        """Use permutation importance so feature importance works for all model types."""
        print("\nCalculating permutation feature importance for the best model...")
        importance = permutation_importance(
            model,
            x_test,
            y_test,
            scoring=self.training_config["scoring"],
            n_repeats=5,
            random_state=self.training_config["random_state"],
            n_jobs=1,
        )

        importance_df = pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
                "model": model_name,
            }
        ).sort_values(by="importance_mean", ascending=False)

        importance_df.to_csv(self.outputs["feature_importance"], index=False)
        print(f"Feature importance saved to {self.outputs['feature_importance']}")
