"""Database loading logic for the ElderGuard project."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import pandas as pd


class SQLiteDataLoader:
    """Loads data from the SQLite database into a pandas DataFrame."""

    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)

    def list_tables(self) -> List[str]:
        """Return all table names in the SQLite database."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

        with sqlite3.connect(self.database_path) as connection:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query, connection)["name"].tolist()
        return tables

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load a selected database table into a DataFrame."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

        with sqlite3.connect(self.database_path) as connection:
            dataframe = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        return dataframe
