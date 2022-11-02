from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, Field, FilePath, validator

from boilerdata.models.common import MyBaseModel, StrPath, default_axes_enum_file


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    # ! DIRECTORY PER TRIAL
    # Don't validate this here. Handle when initializing Project.
    per_trial: Optional[Path] = Field(default=None)

    # ! THIS FILE IS FIXED
    file_axes_enum: Path = Field(default=default_axes_enum_file)

    # ! BASE DIRECTORY
    base: DirectoryPath = Field(default=Path("."))

    # ! CONFIG

    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = Field(
        default=base.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "config",
    )

    # ! DATA

    data: DirectoryPath = Field(
        default=base.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "data",
    )

    # ! SCHEMA

    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "schema",
    )

    file_originlab_coldes: Path = Field(
        default=project_schema.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "originlab_coldes.txt",
    )

    # ! LITERATURE

    literature: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "literature",
    )

    # ! LITERATURE RESULTS

    literature_results: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "literature_results",
    )

    file_literature_results: Path = Field(
        default=literature_results.default  # pyright: ignore [reportGeneralTypeIssues]  # Validator makes it a Path
        / "lit.csv",
    )

    # ! MODEL FUNCTION

    model: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "modelfun",
    )

    file_model: Path = Field(
        default=model.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "model.dillpickle",
    )

    # ! PLOTS

    plots: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "plots",
    )

    # ? NEW_FITS

    new_fits: DirectoryPath = Field(
        default=plots.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "new_fits",
    )

    # ! TRIALS

    trials: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "curves",
    )

    # ! RUNS

    runs: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "runs",
    )

    file_runs: Path = Field(
        default=runs.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "runs.csv",
    )

    # ! RESULTS

    results: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "results",
    )

    file_results: Path = Field(
        default=results.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "results.csv",
    )

    # ! PLOTTER

    plotter: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "plotter",
    )

    file_plotter: FilePath = Field(
        default=plotter.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "results.opju",
    )

    # ! ORIGINLAB RESULTS

    originlab_results: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "originlab_results",
    )

    file_originlab_results: Path = Field(
        default=originlab_results.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "originlab_results.csv",
    )

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "project_schema",
        "literature_results",
        "model",
        "plots",
        "new_fits",
        "runs",
        "results",
        "originlab_results",
        always=True,
        pre=True,
    )
    def validate_output_directories(cls, directory: StrPath):
        """Re-create designated output directories each run, for reproducibility."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
