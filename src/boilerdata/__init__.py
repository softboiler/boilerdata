"""Data processing pipeline for a nucleate pool boiling apparatus."""

from pathlib import Path

PROJECT_PATH = Path()


def get_params_file():
    return PROJECT_PATH / "params.yaml"
