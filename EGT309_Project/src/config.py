"""Configuration utilities for the ElderGuard pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Loads external JSON configuration so paths and model settings are not hardcoded."""

    def __init__(self, config_path: str = "config.json") -> None:
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        """Read the JSON config file and return it as a dictionary."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
