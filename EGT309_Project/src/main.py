"""Main entry point for the ElderGuard Week 7 ML pipeline."""

from __future__ import annotations

import argparse

from config import ConfigLoader
from data_loader import SQLiteDataLoader
from train import ModelTrainer


def parse_arguments() -> argparse.Namespace:
    """Read command line arguments for configurable execution."""
    parser = argparse.ArgumentParser(description="Run ElderGuard machine learning pipeline.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete data loading, training, tuning, and evaluation pipeline."""
    print("========================================")
    print("TEST 1: LOAD CONFIGURATION")
    print("========================================")
    args = parse_arguments()
    config = ConfigLoader(args.config).load()

    print("========================================")
    print("TEST 2: LOAD DATASET FROM SQLITE")
    print("========================================")
    loader = SQLiteDataLoader(config["database"]["path"])
    print("Available tables:", loader.list_tables())
    dataframe = loader.load_table(config["database"]["table_name"])
    print("Dataset shape:", dataframe.shape)

    print("========================================")
    print("TEST 3: TRAIN, TUNE, AND EVALUATE MODELS")
    print("========================================")
    trainer = ModelTrainer(config)
    results = trainer.train_all_models(dataframe)

    print("========================================")
    print("TEST 4: FINAL MODEL RESULTS")
    print("========================================")
    print(results)


if __name__ == "__main__":
    main()
