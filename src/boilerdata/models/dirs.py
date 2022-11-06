from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import DirectoryPath, FilePath, validator

from boilerdata.models.common import (
    MyBaseModel,
    StrPath,
    allow_extra,
    default_axes_enum_file,
)


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    class Config(MyBaseModel.Config):
        @staticmethod
        def schema_extra(schema: dict[str, Any], model: Dirs):
            for prop in schema.get("properties", {}).values():
                if default := prop.get("default"):
                    prop["default"] = default.replace("\\", "/")

    # ! DIRECTORY PER TRIAL
    # Don't validate this here. Handle when initializing Project.
    per_trial: Optional[Path] = None

    # ! BASE DIRECTORY
    base: DirectoryPath = Path(".")

    # ! CONFIG
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = base / "config"

    # ! PACKAGE
    package: DirectoryPath = base / "src/boilerdata"
    stages: DirectoryPath = package / "stages"
    models: DirectoryPath = package / "models"

    # ! DATA
    data: DirectoryPath = base / "data"

    # ! AXES
    axes: DirectoryPath = data / "axes"
    file_axes_enum_copy = axes / "axes_enum.py"
    file_originlab_coldes: Path = axes / "originlab_coldes.txt"

    # ! SCHEMA
    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = data / "schema"

    # ! LITERATURE
    literature: DirectoryPath = data / "literature"

    # ! LITERATURE RESULTS
    literature_results: DirectoryPath = data / "literature_results"
    file_literature_results: Path = literature_results / "lit.csv"

    # ! MODEL FUNCTION
    modelfun: DirectoryPath = data / "modelfun"
    file_model: Path = modelfun / "model.dillpickle"

    # ! TRIALS
    trials: DirectoryPath = data / "curves"

    # ! RUNS
    runs: DirectoryPath = data / "runs"
    file_runs: Path = runs / "runs.csv"

    # ! RESULTS
    results: DirectoryPath = data / "results"
    file_results: Path = results / "results.csv"

    # ! PLOTTER
    plotter: DirectoryPath = data / "plotter"
    file_plotter: FilePath = plotter / "results.opju"

    # ! ORIGINLAB RESULTS
    originlab_results: DirectoryPath = data / "originlab_results"
    file_originlab_results: Path = originlab_results / "originlab_results.csv"

    # ! PLOTS
    metrics: DirectoryPath = data / "metrics"
    new_fits: DirectoryPath = metrics / "new_fits"
    file_pipeline_metrics_plot: Path = metrics / "pipeline_metrics.png"
    file_pipeline_metrics: Path = metrics / "pipeline_metrics.json"

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "axes",
        "project_schema",
        "literature_results",
        "modelfun",
        "new_fits",
        "runs",
        "results",
        "originlab_results",
        "metrics",
        always=True,
        pre=True,
    )
    def validate_output_directories(cls, directory: StrPath):
        """Re-create designated output directories each run, for reproducibility."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def __init__(self, **data):
        super().__init__(**data)
        with allow_extra(self):
            # ! THIS FILE PATH IS DYNAMICALLY GENERATED BASED ON THE PACKAGE DIRECTORY
            self.file_axes_enum: Path = default_axes_enum_file
