"""
data_ingestion.py
-----------------
Responsible for loading the raw dataset from the SQLite database
and performing an initial sanity check on the retrieved data.
"""

import sqlite3
import pandas as pd
from config import DATA_PATH, TABLE_NAME # from config.py


def load_data(db_path: str = DATA_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    """
    Connect to the SQLite database and load the full monitoring table.

    Parameters
    ----------
    db_path : str
        Relative or absolute path to the .db file.
    table : str
        Name of the table to query.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as stored in the database.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:                # always closes the connection even if the query crashes to prevent connection leaks
        conn.close()

    print(f"[Ingestion] Loaded {len(df):,} rows × {df.shape[1]} columns from '{table}'.") # print row n column count with a confirmation msg
    return df


def basic_info(df: pd.DataFrame) -> None:   # prints column types, missing value counts, and the target class distribution
    """Print a quick summary of the loaded dataframe for validation."""
    print("\n[Ingestion] Column dtypes:\n", df.dtypes)
    print("\n[Ingestion] Null counts:\n", df.isnull().sum())
    print("\n[Ingestion] Target distribution:\n", df["Activity Level"].value_counts())


if __name__ == "__main__": # This block only runs if you execute data_ingestion.py directly (python src/data_ingestion.py). It won't run when the file is imported by pipeline.py. This is a standard Python pattern that lets a module be both importable and runnable standalone which is good for quick sanity checks without running the full pipeline.
    raw = load_data()
    basic_info(raw)
