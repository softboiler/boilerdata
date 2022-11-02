from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, Field, validator

from boilerdata.models.common import MyBaseModel, StrPath, default_axes_enum_file


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    # ! FIXED DIRECTORY

    base: DirectoryPath = Field(
        default=Path("."),
        description="The base directory.",
    )

    # ! DIRECTORIES

    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = Field(
        default=base.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "config",
        description="The config directory.",
    )

    data: DirectoryPath = Field(
        default=base.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "data",
        description="The data directory.",
    )

    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "schema",
        description="The schema directory.",
    )

    literature: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "literature",
        description="The directory containing literature figures.",
    )

    literature_results: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "literature_results",
        description="The directory containing processed literature data.",
    )

    model: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "modelfun",
        description="The directory containing a pickled model function.",
    )

    plots: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "plots",
        description="The directory containing plots.",
    )

    new_fits: DirectoryPath = Field(
        default=plots.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "new_fits",
        description="The directory in which model fit plots will go for new runs.",
    )

    trials: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "curves",
        description="The directory containing raw experimental trial data.",
    )

    runs: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "runs",
        description="The directory containing reduced experimental trial data.",
    )

    plotter: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "plotter",
        description="The directory containing the OriginLab project.",
    )

    results: DirectoryPath = Field(
        default=data.default  # pyright: ignore [reportGeneralTypeIssues]  # pydantic
        / "results",
        description="The results directory.",
    )

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "config",
        "data",
        "project_schema",
        "literature",
        "literature_results",
        "model",
        "plots",
        "new_fits",
        "trials",
        "runs",
        "plotter",
        "results",
        always=True,
        pre=True,
    )
    def validate_directory(cls, directory: StrPath):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # ! DIRECTORY PER TRIAL

    # Don't validate this here. Handle when initializing Project.
    per_trial: Optional[Path] = Field(
        default=None,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )

    # ! FIXED FILE
    axes_enum_file: Path = Field(
        default=default_axes_enum_file,
        description="The path to the axes enum file.",
    )

    # ! FILES
    literature_results_file: Path = Field(
        default=literature_results.default / "lit.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the literature results file. Default: lit.csv",
    )

    originlab_coldes_file: Path = Field(
        default=project_schema.default / "originlab_coldes.txt",  # type: ignore  # Validator makes it a Path
        description="The path to which the OriginLab column designation string will be written. Default: coldes.txt",
    )

    model_file: Path = Field(
        default=model.default / "model.dillpickle",  # type: ignore  # Validator makes it a Path
        description="The path to the pickled model function. Default: model.dillpickle",
    )

    runs_file: Path = Field(
        default=runs.default / "runs.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the runs. Default: runs.csv",
    )

    simple_results_file: Path = Field(
        default=results.default / "results.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the simple results file. Default: results.csv",
    )

    originlab_results_file: Path = Field(
        default=results.default / "originlab_results.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the results file to be parsed by OriginLab. Default: originlab_results.csv",
    )

    plotter_file: Path = Field(
        default=project_schema.default / "results.opju",  # type: ignore  # Validator makes it a Path
        description="The path to the OriginLab plotter file. Default: results.opju",
    )

    # "always" so it'll run even if not in YAML
    @validator(
        "literature_results_file",
        "originlab_coldes_file",
        "model_file",
        "runs_file",
        "simple_results_file",
        "originlab_results_file",
        "plotter_file",
        always=True,
    )
    def validate_files(cls, file: Path):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        return file
