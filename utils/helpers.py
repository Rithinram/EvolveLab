"""
EvolveLab — Utility Helpers
Common functions used across the system.
"""

import json
import os
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("evolvelab")


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the entire application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load JSON configuration file with environment variable overrides."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.json"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Apply environment variable overrides
    env_overrides = {
        "POPULATION_SIZE": ("evolution", "population_size", int),
        "MAX_GENERATIONS": ("evolution", "max_generations", int),
        "ELITE_COUNT": ("evolution", "elite_count", int),
        "TOURNAMENT_SIZE": ("evolution", "tournament_size", int),
        "ACCURACY_WEIGHT": ("fitness", "accuracy_weight", float),
        "COST_WEIGHT": ("fitness", "cost_weight", float),
        "DATABASE_URL": ("database", "url", str),
    }

    for env_key, (section, key, type_fn) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                config[section][key] = type_fn(val)
            except (ValueError, KeyError):
                pass

    return config


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def safe_json_dumps(obj: Any) -> str:
    """JSON serialize with fallback for non-serializable objects."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def truncate_string(s: str, max_length: int = 200) -> str:
    """Truncate a string with ellipsis if too long."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def generate_short_id(full_id: str) -> str:
    """Return first 8 characters of a UUID for display."""
    return full_id[:8] if len(full_id) >= 8 else full_id
