from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, Field, validator

from boilerdata.models.common import MyBaseModel, StrPath


class Dirs(MyBaseModel):
    """Directories relevant to the project."""

    # ! BASE

    base: DirectoryPath = Field(
        default=...,
        description="The base directory for the project data.",
    )

    # ! DIRECTORIES

    # Careful, "Config" is a special member of BaseClass
    config: DirectoryPath = Field(
        default=...,
        description="Relative path from the base directory to the config directory.",
    )

    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = Field(
        default=...,
        description="Relative path from the base directory to the schema directory.  Will be created if missing.",
    )

    trials: DirectoryPath = Field(
        default=...,
        description="The directory in which the individual trials are. Must be relative to the base directory.",
    )

    results: DirectoryPath = Field(
        default=...,
        description="The directory in which the results will go. Must be relative to the base directory. Will be created if missing.",
    )

    # "pre" because dir must exist pre-validation
    @validator("config", "project_schema", "trials", "results", pre=True)
    def validate_directory(cls, v: StrPath, values: dict[str, Path]):
        if Path(v).is_absolute():
            raise ValueError("Must be relative to the base directory.")
        directory = values["base"] / v
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
        default="runs.csv",
        description="The path to the runs. Must be relative to the results directory. Default: runs.csv",
    )
    results_file: Path = Field(
        default="results.csv",
        description="The path to the results file. Must be relative to the results directory. Default: results.csv",
    )
    coldes_file: Path = Field(
        default="coldes.txt",
        description="The path to which the OriginLab column designation string will be written. Must be relative to the results directory. Default: coldes.txt",
    )

    # "always" so it'll run even if not in YAML
    @validator("results_file", "coldes_file", "runs_file", always=True)
    def validate_files(cls, file: Path, values: dict[str, Path]):
        if file.is_absolute():
            raise ValueError("Must be relative to the results directory.")
        file = values["results"] / file
        file.parent.mkdir(parents=True, exist_ok=True)
        return file
