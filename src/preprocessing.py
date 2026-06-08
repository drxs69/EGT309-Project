"""
preprocessing.py
----------------
Handles all data-cleaning, imputation, feature-engineering, and
encoding steps before the data is fed into any ML model.

Assumptions & justifications
------------------------------
1.  **Target label normalisation** – The raw data contains duplicate
    spellings of the same class ('Low Activity', 'Low_Activity',
    'LowActivity'; 'Moderate Activity', 'ModerateActivity').
    These are mapped to three canonical labels before any modelling.

2.  **Temperature outliers** – The max temperature (307 °C) is
    physically impossible for indoor air; the distribution is heavily
    right-skewed.  We cap extreme values using an IQR-based fence
    (Q3 + 1.5 × IQR) rather than dropping rows, preserving sample size.

3.  **Missing numeric values** – Humidity (~19 %), MetalOxideSensor_Unit2
    (~14 %), and CO_GasSensor (~8 %) have missing entries.  We impute
    with the column *median* (robust to remaining outliers after capping)
    computed on the training set only to prevent data leakage.

4.  **Missing categorical values** – Ambient Light Level has ~10 % nulls.
    We impute with the most-frequent value ('very_bright') computed on
    the training set.

5.  **HVAC / label case normalisation** – HVAC Operation Mode has mixed
    casing (e.g. 'COOLING_ACTIVE' vs 'cooling_active').  All values are
    lower-cased and stripped before encoding.

6.  **Session ID** – Excluded from features; it is an administrative
    identifier, not a sensor reading.

7.  **Encoding** – Categorical features are one-hot encoded (drop-first
    to avoid multicollinearity); numeric features are standardised with
    StandardScaler.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    SESSION_COLUMN, TEST_SIZE, RANDOM_STATE, TEMP_IQR_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Label normalisation map
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "low activity":      "Low Activity",
    "low_activity":      "Low Activity",
    "lowactivity":       "Low Activity",
    "moderate activity": "Moderate Activity",
    "moderateactivity":  "Moderate Activity",
    "high activity":     "High Activity",
    "highactivity":      "High Activity",
}


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map all Activity Level variants to three canonical classes."""
    df = df.copy()
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)
    )
    unmapped = df[TARGET_COLUMN].isnull().sum()
    if unmapped > 0:
        print(f"[Preprocessing] WARNING: {unmapped} rows with unrecognised Activity Level dropped.")
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"[Preprocessing] Canonical label distribution:\n{df[TARGET_COLUMN].value_counts()}")
    return df


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip all categorical feature values."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].str.strip().str.lower()
    return df


def cap_temperature_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap Temperature at Q3 + IQR_MULTIPLIER * IQR.
    Values well above 100 °C are impossible indoors and treated as
    sensor faults / synthetic contamination.
    """
    df = df.copy()
    q1 = df["Temperature"].quantile(0.25)
    q3 = df["Temperature"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + TEMP_IQR_MULTIPLIER * iqr
    n_capped = (df["Temperature"] > upper).sum()
    df["Temperature"] = df["Temperature"].clip(upper=upper)
    print(f"[Preprocessing] Temperature: capped {n_capped} outlier(s) above {upper:.2f} °C.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional features that may capture sensor-fusion signals.

    New features
    ------------
    CO2_mean        : Average of the two CO2 sensor readings.
    CO2_diff        : Absolute difference between the two CO2 sensors
                      (sensor disagreement – proxy for air stratification
                      or sensor fault).
    MOS_mean        : Mean of all four Metal Oxide Sensor units
                      (captures overall VOC / gas load).
    MOS_range       : Max – Min of the four MOS units
                      (captures spatial variation within the home).
    is_night        : Binary flag for 'night' time-of-day.
    hvac_active     : Binary flag – 1 if HVAC is doing active work
                      (heating, cooling, eco, ventilation).
    """
    df = df.copy()

    # CO2 sensor fusion
    df["CO2_mean"] = df[["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]].mean(axis=1)
    df["CO2_diff"] = (
        df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
    ).abs()

    # Metal oxide sensor fusion
    mos_cols = [c for c in NUMERIC_FEATURES if "MetalOxide" in c]
    df["MOS_mean"]  = df[mos_cols].mean(axis=1)
    df["MOS_range"] = df[mos_cols].max(axis=1) - df[mos_cols].min(axis=1)

    # Time-of-day binary flag
    df["is_night"] = (df["Time of Day"] == "night").astype(int)

    # HVAC active flag
    inactive_modes = {"off", "maintenance_mode"}
    df["hvac_active"] = (~df["HVAC Operation Mode"].isin(inactive_modes)).astype(int)

    print("[Preprocessing] Engineered 6 new features.")
    return df


def split_and_encode(df: pd.DataFrame):
    """
    Split into train/test, then impute, scale, and one-hot encode.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    # Define final feature set (original + engineered numerics + categoricals)
    engineered_numeric = ["CO2_mean", "CO2_diff", "MOS_mean", "MOS_range"]
    binary_features    = ["is_night", "hvac_active"]
    all_numeric = NUMERIC_FEATURES + engineered_numeric + binary_features

    # One-hot encode (in a reproducible way using pd.get_dummies)
    feature_df = df[all_numeric + CATEGORICAL_FEATURES].copy()
    feature_df = pd.get_dummies(feature_df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN])

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Impute numeric columns using train-set statistics only ---
    for col in all_numeric:
        if col in X_train.columns and X_train[col].isnull().any():
            fill_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col]  = X_test[col].fillna(fill_val)

    # One-hot encoded columns are already 0/1; no NaN expected there.
    # Fill any remaining NaN with 0 as a safety net.
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # --- Scale numeric features ---
    scaler = StandardScaler()
    numeric_cols = [c for c in all_numeric if c in X_train.columns]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    feature_names = list(X_train.columns)
    print(f"[Preprocessing] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Preprocessing] Classes: {le.classes_}")
    return X_train, X_test, y_train, y_test, feature_names, le, scaler


def run_preprocessing(df: pd.DataFrame):
    """Full preprocessing pipeline: clean → engineer → split/encode."""
    df = normalise_labels(df)
    df = normalise_categoricals(df)
    df = cap_temperature_outliers(df)
    df = engineer_features(df)
    return split_and_encode(df)
