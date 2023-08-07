"""Data processing pipeline for a nucleate pool boiling apparatus."""
from pathlib import Path

# Monkeypatch these when testing.
PROJECT_DIR = Path()
"""Base directory for the project."""
PARAMS_FILE = PROJECT_DIR / "params.yaml"
"""Path to the project parameters file."""
PROJECT_CONFIG = PROJECT_DIR / "config"
"""Configuration directory for the project."""
TRIAL_CONFIG = PROJECT_CONFIG / "trials.yaml"
"""Path to the trials configuration file."""
AXES_CONFIG = PROJECT_CONFIG / "axes.yaml"
"""Path to the axes configuration file."""
AXES_ENUM_FILE = Path("src/boilerdata/axes_enum.py")
"""Path to the dynamic axes enum file which provides autocomplete."""
DATA_DIR = PROJECT_DIR / Path("data")
"""Data directory."""
