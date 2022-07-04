from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, Field, validator

from boilerdata.models.common import MyBaseModel, StrPath, expanduser2


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
        description="The directory in which the config files are. Must be relative to the base directory or an absolute path that exists.",
    )
    # Can't be "schema", which is a special member of BaseClass
    project_schema: DirectoryPath = Field(
        default=...,
        description="The directory in which the schema are. Must be relative to the base directory or an absolute path that exists.",
    )

    # "pre" because dir must exist pre-validation
    @validator("config", "project_schema", pre=True)
    def validate_configs(cls, v: StrPath, values: dict[str, Path]):
        v = expanduser2(v)
        return v if v.is_absolute() else values["base"] / v

    # ! TRIALS

    trials: DirectoryPath = Field(
        default=...,
        description="The directory in which the individual trials are. Must be relative to the base directory or an absolute path that exists.",
    )
    per_trial: Optional[Path] = Field(
        default=None,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )
    results: DirectoryPath = Field(
        default=...,
        description="The directory in which the results will go. Must be relative to the base directory or an absolute path that exists. Will be created if it is relative to the base directory.",
    )

    @validator("trials", pre=True)  # "pre" because dir must exist pre-validation
    def validate_trials(cls, trials: StrPath, values: dict[str, Path]):
        trials = expanduser2(trials)
        return trials if trials.is_absolute() else values["base"] / trials

    @validator("results", pre=True)  # "pre" because dir must exist pre-validation
    def validate_results(cls, results: StrPath, values: dict[str, Path]):
        if expanduser2(results).is_absolute():
            return results
        results = values["base"] / results
        results.mkdir(parents=True, exist_ok=True)
        return results

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
            raise ValueError("The file must be relative to the results directory.")
        file = values["results"] / file
        file.parent.mkdir(parents=True, exist_ok=True)
        return file
