"""Data cleaning and preprocessing classes for model training."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataCleaner:
    """Cleans labels, removes duplicates, and separates target/features."""

    def __init__(self, target_column: str, drop_columns: List[str] | None = None) -> None:
        self.target_column = target_column
        self.drop_columns = drop_columns or []

    @staticmethod
    def normalise_activity_label(value: object) -> object:
        """Standardise inconsistent target labels such as LowActivity and Low_Activity."""
        if pd.isna(value):
            return value

        cleaned = str(value).strip().replace("_", " ")
        replacements = {
            "LowActivity": "Low Activity",
            "ModerateActivity": "Moderate Activity",
            "HighActivity": "High Activity",
            "Lowactivity": "Low Activity",
            "Moderateactivity": "Moderate Activity",
            "Highactivity": "High Activity",
        }
        cleaned = replacements.get(cleaned, cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned.title()

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning while keeping decisions easy to explain."""
        cleaned_df = dataframe.copy()
        cleaned_df = cleaned_df.drop_duplicates()

        if self.target_column not in cleaned_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")

        cleaned_df[self.target_column] = cleaned_df[self.target_column].apply(
            self.normalise_activity_label
        )
        cleaned_df = cleaned_df.dropna(subset=[self.target_column])
        return cleaned_df

    def split_features_target(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split cleaned data into input features X and target y."""
        features = dataframe.drop(columns=[self.target_column])
        existing_drop_columns = [col for col in self.drop_columns if col in features.columns]
        features = features.drop(columns=existing_drop_columns)
        target = dataframe[self.target_column]
        return features, target


class PreprocessorFactory:
    """Creates a reusable scikit-learn preprocessor for numerical and categorical columns."""

    @staticmethod
    def identify_columns(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical feature columns."""
        numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_columns, categorical_columns

    @staticmethod
    def create_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing steps without manually hardcoding column names."""
        numeric_columns, categorical_columns = PreprocessorFactory.identify_columns(features)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ]
        )
        return preprocessor
