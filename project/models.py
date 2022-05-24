from datetime import date
from enum import auto
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, DirectoryPath, Extra, Field, validator

from boilerdata.enums import GetNameEnum
from boilerdata.utils import expanduser2, load_config

# * -------------------------------------------------------------------------------- * #
# * PROJECT


class Fit(BaseModel):
    """Configure the linear regression of thermocouple temperatures vs. position."""

    thermocouple_pos: list[float] = Field(
        default=...,
        description="Thermocouple positions.",
    )
    do_plot: bool = Field(False, description="Whether to plot the linear regression.")

    @validator("thermocouple_pos")
    def _(cls, thermocouple_pos):
        return np.array(thermocouple_pos)


class Dirs(BaseModel, extra=Extra.forbid):
    """Directories relevant to the project."""

    base: DirectoryPath = Field(
        default=...,
        description="The base directory for the project data.",
    )
    config: DirectoryPath = Field(
        default=...,
        description="The directory in which the config files are. Must be relative to the base directory or an absolute path that exists.",
    )
    trials: DirectoryPath = Field(
        default=...,
        description="The directory in which the individual trials are. Must be relative to the base directory or an absolute path that exists.",
    )
    results: DirectoryPath = Field(
        default=...,
        description="The directory in which the results will go. Must be relative to the base directory or an absolute path that exists. Will be created if it is relative to the base directory.",
    )
    results_file: Path = Field(
        default="results.csv",
        description="The path to the results file. Must be relative to the results directory. Default: results.csv",
    )
    per_trial: Path = Field(
        default=...,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )

    @validator("trials", pre=True)  # "pre" because dir must exist pre-validation
    def validate_trials(cls, trials, values):
        trials = expanduser2(trials)
        return trials if trials.is_absolute() else values["base"] / trials

    @validator("config", pre=True)  # "pre" because dir must exist pre-validation
    def validate_configs(cls, config, values):
        config = expanduser2(config)
        return config if config.is_absolute() else values["base"] / config

    @validator("results", pre=True)  # "pre" because dir must exist pre-validation
    def validate_results(cls, results, values):
        if (results := expanduser2(results)).is_absolute():
            return results
        results.mkdir(parents=True, exist_ok=True)
        return values["base"] / results

    @validator("results_file", always=True)  # "always" so it'll run even if not in YAML
    def validate_results_file(cls, results_file: Path, values):
        if results_file.is_absolute():
            raise ValueError(
                "The results file path must be given relative to the results directory."
            )
        if results_file.suffix != ".csv":
            raise ValueError("The supplied results file is not a CSV.")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        return values["results"] / results_file


# Extra fields are allowed so we can pack trials and columns into this
class Project(BaseModel, extra=Extra.allow):
    """Configuration for the package."""

    dirs: Dirs
    fit: Fit

    def __init__(self, **data):
        super().__init__(**data)
        self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
        self.columns = load_config(self.dirs.config / "columns.yaml", Columns).columns


# * -------------------------------------------------------------------------------- * #
# * TRIALS


class Rod(GetNameEnum):
    """The rod used in this trial."""

    W = auto()
    X = auto()
    Y = auto()


class Coupon(GetNameEnum):
    """The coupon attached to the rod for this trial."""

    A1 = auto()
    A2 = auto()
    A3 = auto()
    A4 = auto()
    A6 = auto()
    A7 = auto()
    A9 = auto()


class Sample(GetNameEnum):
    """The sample attached to the coupon in this trial."""

    NA = auto()  # If no sample is attached to the coupon.
    B3 = auto()


class Group(GetNameEnum):
    """The group that this sample belongs to."""

    control = auto()
    porous = auto()
    hybrid = auto()


class Joint(GetNameEnum):
    """The method used to join parts of the sample in this trial."""

    paste = auto()
    epoxy = auto()
    solder = auto()


class Trial(BaseModel, extra=Extra.forbid):
    """A trial."""

    date: date
    rod: Rod
    coupon: Coupon
    sample: Sample
    group: Group
    joint: Joint
    monotonic: bool = Field(
        default=...,
        description="Whether the boiling curve is monotonic.",
    )
    comment: str

    def get_path(self, project: Project) -> Path:
        """Get the path to the data for this trial."""
        return project.dirs.trials / self.date.isoformat() / project.dirs.per_trial


class Trials(BaseModel):
    """The trials."""

    trials: list[Trial]


# * -------------------------------------------------------------------------------- * #
# * COLUMNS


class Column(BaseModel):
    """Metadata for a column in the dataframe."""

    pretty_name: Optional[str] = Field(
        default=None,
        description="The column name.",
    )
    units: str = Field(
        default=...,
        description="The units for this column's values.",
    )
    originlab_column_designation: str = Field(
        default=None,
        description="The column designation for plotting in OriginLab.",
    )

    # "always" so it'll run even if not in YAML
    @validator("originlab_column_designation", pre=True, always=True)
    def _(cls, v):
        return v or "N"


class Columns(BaseModel):
    """Columns in the dataframe."""

    columns: dict[str, Column]

    def generate_originlab_column_designation_string(self) -> str:
        return "N" + "".join(
            [column.originlab_column_designation for column in self.columns.values()]
        )
