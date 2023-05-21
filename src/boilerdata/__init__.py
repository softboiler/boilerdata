"""Data processing pipeline for a nucleate pool boiling apparatus."""
from pathlib import Path

# Monkeypatch these when testing.
GIT_BASE = Path(".")
"""Base directory for git-tracked files."""
DVC_BASE = Path(".")
"""Base directory for DVC-tracked files."""
PARAMS_FILE = Path("params.yaml")
"""Path to the project parameters file."""
AXES_CONFIG = Path("config/axes.yaml")
"""Path to the axes configuration file."""
AXES_ENUM_FILE = Path("src/boilerdata/axes_enum.py")
"""Path to the dynamic axes enum file which provides autocomplete."""
