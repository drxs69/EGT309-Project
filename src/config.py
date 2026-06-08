"""
config.py
---------
Central configuration for the ElderGuard Analytics ML pipeline.
All hyperparameters, paths, and settings are defined here so that
the pipeline can be easily reconfigured without touching model code.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "gas_monitoring.db")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
TABLE_NAME = "gas_monitoring"
TARGET_COLUMN = "Activity Level"
SESSION_COLUMN = "Session ID"          # identifier – not a predictive feature

# Temperature IQR multiplier for outlier capping
TEMP_IQR_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "CO2_InfraredSensor",
    "CO2_ElectroChemicalSensor",
    "MetalOxideSensor_Unit1",
    "MetalOxideSensor_Unit2",
    "MetalOxideSensor_Unit3",
    "MetalOxideSensor_Unit4",
    "CO_GasSensor",
]

CATEGORICAL_FEATURES = [
    "Time of Day",
    "HVAC Operation Mode",
    "Ambient Light Level",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

# Random Forest
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Gradient Boosting
GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

# Logistic Regression
LR_PARAMS = {
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
    "C": 0.5,
    "solver": "lbfgs",
    "class_weight": "balanced",
}

# Cross-validation folds
CV_FOLDS = 5
