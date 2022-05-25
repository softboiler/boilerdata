from datetime import date
from enum import auto
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, DirectoryPath, Field, validator

from boilerdata.enums import NameEnum
from boilerdata.utils import StrPath, allow_extra, expanduser2, load_config

# * -------------------------------------------------------------------------------- * #
# * DIRS


class Dirs(BaseModel):
    """Directories relevant to the project."""

    base: DirectoryPath = Field(
        default=...,
        description="The base directory for the project data.",
    )

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
    def validate_trials(cls, trials: StrPath, values: dict[str, Path]):
        trials = expanduser2(trials)
        return trials if trials.is_absolute() else values["base"] / trials

    @validator(
        "config", "project_schema", pre=True
    )  # "pre" because dir must exist pre-validation
    def validate_configs(cls, v: StrPath, values: dict[str, Path]):
        v = expanduser2(v)
        return v if v.is_absolute() else values["base"] / v

    @validator("results", pre=True)  # "pre" because dir must exist pre-validation
    def validate_results(cls, results: StrPath, values: dict[str, Path]):
        if expanduser2(results).is_absolute():
            return results
        results = values["base"] / results
        results.mkdir(parents=True, exist_ok=True)
        return values["base"] / results

    @validator("results_file", always=True)  # "always" so it'll run even if not in YAML
    def validate_results_file(cls, results_file: Path, values: dict[str, Path]):
        if results_file.is_absolute():
            raise ValueError("The file must be relative to the results directory.")
        if results_file.suffix != ".csv":
            raise ValueError("The supplied results file is not a CSV.")
        results_file = values["results"] / results_file
        results_file.parent.mkdir(parents=True, exist_ok=True)
        return results_file


# * -------------------------------------------------------------------------------- * #
# * PARAMS


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


class Params(BaseModel):
    """Parameters of the pipeline."""

    records_to_average: int = Field(
        default=60,
        description="The number of records over which to average in a given trial.",
    )


# * -------------------------------------------------------------------------------- * #
# * TRIALS


class Rod(NameEnum):
    """The rod used in this trial."""

    W = auto()
    X = auto()
    Y = auto()


class Coupon(NameEnum):
    """The coupon attached to the rod for this trial."""

    A1 = auto()
    A2 = auto()
    A3 = auto()
    A4 = auto()
    A6 = auto()
    A7 = auto()
    A9 = auto()


class Sample(NameEnum):
    """The sample attached to the coupon in this trial."""

    NA = auto()  # If no sample is attached to the coupon.
    B3 = auto()


class Group(NameEnum):
    """The group that this sample belongs to."""

    control = auto()
    porous = auto()
    hybrid = auto()


class Joint(NameEnum):
    """The method used to join parts of the sample in this trial."""

    paste = auto()
    epoxy = auto()
    solder = auto()


class Trial(BaseModel):
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

    def get_path(self, dirs: Dirs):
        """Get the path to the data for this trial."""

        with allow_extra(self):
            self.path = dirs.trials / self.date.isoformat() / dirs.per_trial


class Trials(BaseModel):
    """The trials."""

    trials: list[Trial]


# * -------------------------------------------------------------------------------- * #
# * COLUMNS

# sourcery skip: avoid-builtin-shadow
class PandasDtype(NameEnum):
    float = auto()  # noqa: A003
    int = auto()  # noqa: A003
    bool = auto()  # noqa: A003
    timedelta64ns = "timedelta64[ns]"
    datetime64ns = "datetime64[ns]"
    string = auto()
    boolean = auto()
    category = auto()
    Sparse = auto()
    interval = auto()
    Int8 = auto()
    Int16 = auto()
    Int32 = auto()
    Int64 = auto()
    UInt8 = auto()
    UInt16 = auto()
    Uint32 = auto()
    UInt64 = auto()


class Column(BaseModel):
    """Metadata for a column in the dataframe."""

    index: bool = Field(  # Validator ensures this is set even if omitted.
        default=False,
        description="Whether this column is to be the index.",
    )
    dtype: PandasDtype = Field(
        default=PandasDtype.float,
        description="The Pandas data type to be used to represent this column.",
    )
    units: str = Field(
        default="",
        description="The units for this column's values.",
    )
    source: Optional[str] = Field(
        default=None,
        description="The name of the input column that this column is based off of.",
    )
    originlab_coldes: str = Field(  # Validator ensures this is set even if omitted.
        default=None,
        description="The column designation for plotting in OriginLab.",
    )
    pretty_name: Optional[str] = Field(
        default=None,
        description="The column name.",
    )

    # "always" so it'll run even if not in YAML
    @validator("index", pre=True, always=True)
    def validate_index(cls, index):
        return index or False

    @validator("source")
    def validate_source(cls, source, values):
        return f"{source} ({values['units']})" if values["units"] else source

    # "always" so it'll run even if not in YAML
    @validator("originlab_coldes", pre=True, always=True)
    def validate_coldes(cls, v):
        return v or "N"

    def __init__(self, **data):
        super().__init__(**data)
        self.name: str = ""  # Should never stay this way. Columns.__init__() sets it.


class Columns(BaseModel):
    """Columns in the dataframe."""

    columns: dict[str, Column]

    def __init__(self, **data):
        super().__init__(**data)

        for name, column in self.columns.items():
            column.name = name


# * -------------------------------------------------------------------------------- * #
# * PROJECT


class Project(BaseModel):
    """Configuration for the package."""

    dirs: Dirs
    params: Params
    fit: Fit

    def __init__(self, **data):
        super().__init__(**data)

        with allow_extra(self):
            self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
            self.columns = load_config(
                self.dirs.config / "columns.yaml", Columns
            ).columns

        for trial in self.trials:
            trial.get_path(self.dirs)

    def get_source_cols(self) -> list[Column]:
        return [column for column in self.columns.values() if column.source]

    def get_index(self) -> Column:
        index_cols = [column for column in self.columns.values() if column.index]
        match index_cols:
            case [index]:
                return index
            case []:
                raise ValueError("One column must be designated as the index.")
            case [*indices]:
                indices = [index.name for index in indices]
                raise ValueError(
                    f"Only one column may be designated as the index. You specified the following: {', '.join(indices)}"
                )

    def get_originlab_coldes(self) -> str:
        return "N" + "".join(
            [column.originlab_coldes for column in self.columns.values()]
        )
