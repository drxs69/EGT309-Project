"""
pipeline.py
-----------
End-to-end entry point for the ElderGuard Analytics ML pipeline.

Usage
-----
    python pipeline.py                          # run with defaults
    python pipeline.py --data ../data/gas_monitoring.db
    python pipeline.py --rf_n_estimators 300 --gb_learning_rate 0.05
    python pipeline.py --skip_train             # evaluate saved models only

The pipeline executes in five stages:
    1. Data ingestion
    2. Preprocessing & feature engineering
    3. Model training (Random Forest, Gradient Boosting, Logistic Regression)
    4. Model evaluation (test-set metrics + confusion matrices + feature importances)
    5. Artefact persistence (trained models saved to saved_model/)
"""

import argparse
import os
import sys

# Ensure src/ is on the import path when called from the project root
sys.path.insert(0, os.path.dirname(__file__))

import config
from data_ingestion  import load_data
from preprocessing   import run_preprocessing
from model_training  import build_models, train_and_evaluate, save_models
from model_evaluation import evaluate_all, plot_confusion_matrices, plot_feature_importances


def parse_args():
    parser = argparse.ArgumentParser(
        description="ElderGuard Analytics – Activity Level Prediction Pipeline"
    )
    parser.add_argument("--data",              type=str,   default=config.DATA_PATH)
    parser.add_argument("--test_size",         type=float, default=config.TEST_SIZE)
    parser.add_argument("--rf_n_estimators",   type=int,   default=config.RF_PARAMS["n_estimators"])
    parser.add_argument("--gb_n_estimators",   type=int,   default=config.GB_PARAMS["n_estimators"] )
    parser.add_argument("--gb_learning_rate",  type=float, default=config.GB_PARAMS["learning_rate"])
    parser.add_argument("--lr_C",              type=float, default=config.LR_PARAMS["C"])
    parser.add_argument("--skip_train",        action="store_true",
                        help="Skip training and load saved models for evaluation.")
    return parser.parse_args()


def apply_cli_overrides(args) -> None:
    """Override config values with any CLI arguments supplied."""
    config.DATA_PATH = args.data
    config.TEST_SIZE = args.test_size
    config.RF_PARAMS["n_estimators"]  = args.rf_n_estimators
    config.GB_PARAMS["n_estimators"]  = args.gb_n_estimators
    config.GB_PARAMS["learning_rate"] = args.gb_learning_rate
    config.LR_PARAMS["C"]             = args.lr_C


def main():
    args = parse_args()
    apply_cli_overrides(args)

    print("\n========================================")
    print("  ElderGuard Analytics – ML Pipeline")
    print("========================================\n")

    # ------------------------------------------------------------------
    # Stage 1 – Ingest
    # ------------------------------------------------------------------
    raw_df = load_data(db_path=config.DATA_PATH)

    # ------------------------------------------------------------------
    # Stage 2 – Preprocess
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, feature_names, le, _ = run_preprocessing(raw_df)

    # ------------------------------------------------------------------
    # Stage 3 – Train
    # ------------------------------------------------------------------
    models = build_models()
    if not args.skip_train:
        results = train_and_evaluate(X_train, y_train, models)
        save_models(results, config.MODEL_DIR)
    else:
        from model_training import load_model
        results = {
            name: {"model": load_model(name, config.MODEL_DIR), "cv_scores": []}
            for name in models
        }
        print("[Pipeline] Loaded saved models from disk.")

    # ------------------------------------------------------------------
    # Stage 4 – Evaluate
    # ------------------------------------------------------------------
    output_dir = os.path.join(config.MODEL_DIR, "plots")
    eval_results = evaluate_all(results, X_test, y_test, le)
    plot_confusion_matrices(eval_results, le, output_dir)
    plot_feature_importances(results, feature_names, output_dir)

    print("\n[Pipeline] Complete. Artefacts saved to:", config.MODEL_DIR)


if __name__ == "__main__":
    main()
