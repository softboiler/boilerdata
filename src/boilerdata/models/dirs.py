from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, FilePath, validator

from boilerdata.models.common import MyBaseModel, StrPath, default_axes_enum_file


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    # ! DIRECTORY PER TRIAL
    # Don't validate this here. Handle when initializing Project.
    per_trial: Optional[Path] = None

    # ! THIS FILE IS FIXED
    file_axes_enum: Path = default_axes_enum_file

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

    # ! SCHEMA
    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = data / "schema"
    file_originlab_coldes: Path = project_schema / "originlab_coldes.txt"

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

    # ! METRICS
    metrics: DirectoryPath = data / "metrics"
    file_pipeline_metrics: Path = metrics / "pipeline_metrics.json"

    # ! PLOTS
    plots: DirectoryPath = data / "plots"
    new_fits: DirectoryPath = plots / "new_fits"
    file_pipeline_metrics_plot: Path = plots / "pipeline_metrics.png"

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "project_schema",
        "literature_results",
        "modelfun",
        "new_fits",
        "runs",
        "results",
        "originlab_results",
        "metrics",
        "plots",
        always=True,
        pre=True,
    )
    def validate_output_directories(cls, directory: StrPath):
        """Re-create designated output directories each run, for reproducibility."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
