from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, Field, validator

from boilerdata.models.common import MyBaseModel, StrPath


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    # ! DIRECTORIES

    base: DirectoryPath = Field(
        default=Path("."),
        description="The base directory.",
    )

    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = Field(
        default=base.default / "config",  # type: ignore  # Validator makes it a Path
        description="The config directory.",
    )

    data: DirectoryPath = Field(
        default=base.default / "data",  # type: ignore  # Validator makes it a Path
        description="The data directory.",
    )

    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = Field(
        default=data.default / "schema",  # type: ignore  # Validator makes it a Path
        description="The schema directory.",
    )

    # Can't be "schema", which is a special member of BaseClass
    project_schema_old: DirectoryPath = Field(
        default=data.default / "schema_old",  # type: ignore  # Validator makes it a Path
        description="The old schema. Done to eliminate a data pipeline cycle.",
    )

    trials: DirectoryPath = Field(
        default=data.default / "curves",  # type: ignore  # Validator makes it a Path
        description="The trials directory.",
    )

    runs: DirectoryPath = Field(
        default=data.default / "runs",  # type: ignore  # Validator makes it a Path
        description="The runs directory.",
    )

    results: DirectoryPath = Field(
        default=data.default / "results",  # type: ignore  # Validator makes it a Path
        description="The results directory.",
    )

    new_fits: DirectoryPath = Field(
        default=results.default / "new_fits",  # type: ignore  # Validator makes it a Path
        description="The directory in which fit plots will go for new runs.",
    )

    # "always" so it'll run even if not in YAML
    # "pre" because dir must exist pre-validation
    @validator(
        "config",
        "data",
        "project_schema",
        "trials",
        "runs",
        "results",
        "new_fits",
        always=True,
        pre=True,
    )
    def validate_directory(cls, v: StrPath):
        directory = Path(v)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # ! DIRECTORY PER TRIAL

    # Don't validate this here. Handle when initializing Project.
    per_trial: Optional[Path] = Field(
        default=None,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )

    # ! FILES

    runs_file: Path = Field(
        default=runs.default / "runs.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the runs. Must be relative to the results directory. Default: runs.csv",
    )
    simple_results_file: Path = Field(
        default=results.default / "results.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the simple results file. Must be relative to the results directory. Default: results.csv",
    )
    originlab_results_file: Path = Field(
        default=results.default / "originlab_results.csv",  # type: ignore  # Validator makes it a Path
        description="The path to the results file to be parsed by OriginLab. Must be relative to the results directory. Default: originlab_results.csv",
    )
    originlab_coldes_file: Path = Field(
        default=project_schema.default / "originlab_coldes.txt",  # type: ignore  # Validator makes it a Path
        description="The path to which the OriginLab column designation string will be written. Must be relative to the results directory. Default: coldes.txt",
    )

    # "always" so it'll run even if not in YAML
    @validator(
        "runs_file",
        "simple_results_file",
        "originlab_results_file",
        "originlab_coldes_file",
        always=True,
    )
    def validate_files(cls, v: Path):
        file = Path(v)
        file.parent.mkdir(parents=True, exist_ok=True)
        return file
