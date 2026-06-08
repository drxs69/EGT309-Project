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

# Outlier capping settings
TEMP_IQR_MULTIPLIER = 1.5
HUMIDITY_MIN = 0
HUMIDITY_MAX = 100

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

# Random Forest tuned mainly for accuracy. Removing class_weight="balanced"
# improves overall accuracy because the dataset is dominated by Low Activity.
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Gradient Boosting tuned as a strong boosting baseline while keeping runtime reasonable.
GB_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

# Logistic Regression baseline. No class balancing because the user requested
# higher accuracy; balanced LR improves minority recall but lowers accuracy.
LR_PARAMS = {
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
    "C": 0.1,
    "solver": "lbfgs",
    "class_weight": None,
}

# Set to 3 or 5 to enable cross-validation. Default 0 keeps the pipeline fast
# for assignment demonstrations and uses the held-out test set for evaluation.
CV_FOLDS = 0
