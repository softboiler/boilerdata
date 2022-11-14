from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import DirectoryPath, FilePath, validator

from boilerdata.models.common import MyBaseModel, StrPath


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

    # ! PROJECT FILE
    file_proj: FilePath = base / "params.yaml"

    # ! CONFIG
    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = base / "config"

    # ! PACKAGE
    package: DirectoryPath = base / "src/boilerdata"
    stages: DirectoryPath = package / "stages"
    notebooks: DirectoryPath = stages / "notebooks"
    models: DirectoryPath = package / "models"
    validation: FilePath = package / "validation.py"
    file_axes_enum: FilePath = package / "axes_enum.py"

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

    # ! PLOT CONFIG
    plot_config: DirectoryPath = config / "plotting"
    mpl_base: FilePath = plot_config / "base.mplstyle"
    mpl_hide_title: FilePath = plot_config / "hide_title.mplstyle"

    # ! TOP-LEVEL METRICS DIR
    metrics: DirectoryPath = data / "metrics"

    # ! PLOTS
    plots: DirectoryPath = metrics / "plots"
    plot_new_fit_0: Path = plots / "new_fit_0.png"
    plot_new_fit_1: Path = plots / "new_fit_1.png"
    plot_new_fit_2: Path = plots / "new_fit_2.png"
    plot_error_T_s: Path = plots / "error_T_s.png"  # noqa: N815
    plot_error_q_s: Path = plots / "error_q_s.png"
    plot_error_h_a: Path = plots / "error_h_a.png"

    # ! TABLES
    tables: DirectoryPath = metrics / "tables"
    file_pipeline_metrics: Path = tables / "pipeline_metrics.json"

    # ! STAGES
    stage_setup: FilePath = stages / "setup.py"
    stage_axes: FilePath = stages / "axes.py"
    stage_literature: FilePath = stages / "literature.py"
    stage_metrics: FilePath = notebooks / "metrics.ipynb"
    stage_modelfun: FilePath = notebooks / "modelfun.ipynb"
    stage_originlab: FilePath = stages / "originlab.py"
    stage_pipeline: FilePath = stages / "pipeline.py"
    stage_runs: FilePath = stages / "runs.py"
    stage_schema: FilePath = stages / "schema.py"

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "axes",
        "literature_results",
        "metrics",
        "modelfun",
        "originlab_results",
        "plots",
        "project_schema",
        "results",
        "runs",
        "tables",
        always=True,
        pre=True,
    )
    def validate_output_directories(cls, directory: StrPath):
        """Re-create designated output directories each run, for reproducibility."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
