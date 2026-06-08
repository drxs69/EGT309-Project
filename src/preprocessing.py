"""
preprocessing.py
----------------
Handles all data-cleaning, imputation, feature-engineering, and
encoding steps before the data is fed into any ML model.

Main accuracy fixes added:
1. Split the dataset before fitting imputers so test data does not leak into
   training preprocessing statistics.
2. Impute categorical missing values using the training-set mode before
   one-hot encoding.
3. Keep all one-hot categories instead of drop_first=True. Tree models do not
   need drop-first encoding, and keeping all categories preserves signal.
4. Cap impossible humidity values to the physical 0–100% range.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, TEMP_IQR_MULTIPLIER,
    HUMIDITY_MIN, HUMIDITY_MAX,
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
    Values well above normal indoor temperature are treated as sensor faults.
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


def cap_humidity_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap Humidity to the physically valid 0–100% range."""
    df = df.copy()
    n_capped = ((df["Humidity"] < HUMIDITY_MIN) | (df["Humidity"] > HUMIDITY_MAX)).sum()
    df["Humidity"] = df["Humidity"].clip(lower=HUMIDITY_MIN, upper=HUMIDITY_MAX)
    print(f"[Preprocessing] Humidity: capped {n_capped} invalid value(s) outside {HUMIDITY_MIN}–{HUMIDITY_MAX}%.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional sensor-fusion features.
    """
    df = df.copy()

    df["CO2_mean"] = df[["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]].mean(axis=1)
    df["CO2_diff"] = (
        df["CO2_InfraredSensor"] - df["CO2_ElectroChemicalSensor"]
    ).abs()

    mos_cols = [c for c in NUMERIC_FEATURES if "MetalOxide" in c]
    df["MOS_mean"] = df[mos_cols].mean(axis=1)
    df["MOS_range"] = df[mos_cols].max(axis=1) - df[mos_cols].min(axis=1)

    df["is_night"] = (df["Time of Day"] == "night").astype(int)

    inactive_modes = {"off", "maintenance_mode"}
    df["hvac_active"] = (~df["HVAC Operation Mode"].isin(inactive_modes)).astype(int)

    print("[Preprocessing] Engineered 6 new features.")
    return df


def split_and_encode(df: pd.DataFrame):
    """
    Split into train/test, then impute, scale, and one-hot encode.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler
    """
    engineered_numeric = ["CO2_mean", "CO2_diff", "MOS_mean", "MOS_range"]
    binary_features = ["is_night", "hvac_active"]
    all_numeric = NUMERIC_FEATURES + engineered_numeric + binary_features

    feature_df = df[all_numeric + CATEGORICAL_FEATURES].copy()

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Impute numeric columns using train-set median only.
    for col in all_numeric:
        fill_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

    # Impute categorical columns using train-set mode only.
    for col in CATEGORICAL_FEATURES:
        mode_val = X_train[col].mode(dropna=True)[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    # One-hot encode after split and align test columns to train columns.
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale numeric features. Tree models do not require it, but LR benefits.
    scaler = StandardScaler()
    numeric_cols = [c for c in all_numeric if c in X_train.columns]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    feature_names = list(X_train.columns)
    print(f"[Preprocessing] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Preprocessing] Classes: {le.classes_}")
    return X_train, X_test, y_train, y_test, feature_names, le, scaler


def run_preprocessing(df: pd.DataFrame):
    """Full preprocessing pipeline: clean → engineer → split/encode."""
    df = normalise_labels(df)
    df = normalise_categoricals(df)
    df = cap_temperature_outliers(df)
    df = cap_humidity_outliers(df)
    df = engineer_features(df)
    return split_and_encode(df)
