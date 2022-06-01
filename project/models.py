import datetime
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field, FilePath, validator

from boilerdata.typing import NpNDArray
from boilerdata.utils import StrPath, expanduser2, load_config
from enums import Coupon, Group, Joint, OriginLabColdes, PandasDtype, Rod, Sample

# * -------------------------------------------------------------------------------- * #
# * BASE


class MyBaseModel(
    BaseModel,
    use_enum_values=True,  # To use enums for schemas, but not in code
    arbitrary_types_allowed=True,  # To use Numpy types
):
    pass


# * -------------------------------------------------------------------------------- * #
# * DIRS


class Dirs(MyBaseModel):
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
    per_trial: Path = Field(
        default=...,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )
    results: DirectoryPath = Field(
        default=...,
        description="The directory in which the results will go. Must be relative to the base directory or an absolute path that exists. Will be created if it is relative to the base directory.",
    )
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

    # "always" so it'll run even if not in YAML
    @validator("results_file", "coldes_file", "runs_file", always=True)
    def validate_files(cls, file: Path, values: dict[str, Path]):
        if file.is_absolute():
            raise ValueError("The file must be relative to the results directory.")
        file = values["results"] / file
        file.parent.mkdir(parents=True, exist_ok=True)
        return file


# * -------------------------------------------------------------------------------- * #
# * PARAMS


class Params(MyBaseModel):
    """Parameters of the pipeline."""

    refetch_runs: bool = Field(
        default=False,
        description="Fetch the runs from their source files again even if there are no new runs.",
    )
    records_to_average: int = Field(
        default=60,
        description="The number of records over which to average in a given trial.",
    )
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )


# * -------------------------------------------------------------------------------- * #
# * COLUMNS


class Column(MyBaseModel):
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
    meta: bool = Field(
        default=False,
        description="Whether this column is informed by the trials config.",
    )

    originlab_coldes: OriginLabColdes = Field(
        default="N",
        description="The column designation for plotting in OriginLab.",
    )

    # Can't be None. Set in Columns.__init__()
    name: str = Field(default=None)

    # Can be None, but never accessed directly.
    pretty_name_: Optional[str] = Field(
        default=None,
        alias="pretty_name",  # So we can make this a dynamic property below
        description="The pretty version of the column name.",
    )

    # "always" so it'll run even if not in YAML
    @validator("index", pre=True, always=True)
    def validate_index(cls, index):
        return index or False

    @validator("source")
    def validate_source(cls, source, values):
        return f"{source} ({values['units']})" if values["units"] else source

    def __init__(self, **data):
        super().__init__(**data)

    @property
    def pretty_name(self):
        return self.pretty_name_ or self.name


class Columns(MyBaseModel):
    """Columns in the dataframe."""

    cols: dict[str, Column]

    def __init__(self, **data):
        super().__init__(**data)

        # Set name as convenience property
        for name, col in self.cols.items():
            col.name = name


# * -------------------------------------------------------------------------------- * #
# * GEOMETRY


class Geometry(MyBaseModel):
    """The geometry."""

    rods: dict[Rod, NpNDArray] = Field(
        default=...,
        description="Distance of each thermocouple from the cool side of the rod, starting with TC1. Fifth thermocouple may be omitted. Input: inch. Output: meter.",
        exclude=True,
    )
    coupons: dict[Coupon, float] = Field(
        default=...,
        description="Length of the coupon. Input: inch. Output: meter.",
    )
    _in_p_m: float = 39.3701  # (in/m) Conversion factor

    @validator("rods", pre=True)
    def validate_rods(cls, rods):
        return {rod: np.array(values) / cls._in_p_m for rod, values in rods.items()}

    @validator("coupons")
    def validate_coupons(cls, coupons):
        return {coupon: value / cls._in_p_m for coupon, value in coupons.items()}


# * -------------------------------------------------------------------------------- * #
# * TRIALS


class Trial(MyBaseModel):
    """A trial."""

    date: datetime.date = Field(
        default=..., description="The date of the trial.", exclude=True
    )
    group: Group
    rod: Rod
    coupon: Coupon
    sample: Sample
    joint: Joint
    good: bool = Field(
        default=True,
        description="Whether the boiling curve is good.",
    )
    new: bool = Field(
        default=False, description="Whether this is newly-collected data."
    )
    comment: str

    # Can't be None. Set in Project.__init__()
    path: DirectoryPath = Field(default=None, exclude=True)
    run_files: list[FilePath] = Field(default=None, exclude=True)
    run_index: list[tuple[pd.Timestamp, pd.Timestamp]] = Field(
        default=None, exclude=True
    )
    thermocouple_pos: NpNDArray = Field(default=None, exclude=True)

    @property
    def trial(self):
        return pd.Timestamp(self.date)

    def set_paths(self, dirs: Dirs):
        """Get the path to the data for this trial. Called during project setup."""

        self.path = dirs.trials / str(self.trial.date()) / dirs.per_trial
        self.run_files = sorted(self.path.glob("*.csv"))

    def set_index(self):
        """Get the multiindex for all runs. Called during project setup."""

        run_re = re.compile(r"(?P<date>.*)T(?P<time>.*)")
        run_index: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for run_file in self.run_files:

            run_time = run_file.stem.removeprefix("results_")

            if m := run_re.match(run_time):
                run_time = f"{m['date']}T{m['time'].replace('-', ':')}"
            else:
                raise AttributeError(f"Could not parse run time: {run_time}")

            trial_date = self.trial.isoformat()  # for consistency across datetimes
            run_index.append(
                tuple(
                    pd.Timestamp.fromisoformat(item) for item in [trial_date, run_time]
                )
            )
        self.run_index = run_index

    def set_geometry(self, geometry: Geometry):
        """Get relevant geometry for the trial."""
        self.thermocouple_pos = geometry.rods[self.rod] + geometry.coupons[self.coupon]  # type: ignore


class Trials(MyBaseModel):
    """The trials."""

    trials: list[Trial]


# * -------------------------------------------------------------------------------- * #
# * PROJECT


class Project(MyBaseModel):
    """Configuration for the package."""

    dirs: Dirs
    geometry: Geometry
    params: Params

    # These can't be None, as they are set in Project.__init__()
    trials: list[Trial] = Field(default=None)
    cols: dict[str, Column] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
        cols = load_config(self.dirs.config / "columns.yaml", Columns)
        self.cols = cols.cols

        for trial in self.trials:
            trial.set_paths(self.dirs)
            trial.set_geometry(self.geometry)
            trial.set_index()

    def get_index(self) -> list[Column]:
        index_cols = [col for col in self.cols.values() if col.index]
        match index_cols:
            case []:
                raise ValueError("One column must be designated as the index.")
            case [index]:
                return [index]
            case [*indices]:
                return indices

    def get_col_index(self) -> pd.MultiIndex:

        cols = {name: col for name, col in self.cols.items() if not col.index}

        # Rename columns and extract them into a row
        quantity = pd.DataFrame(
            cols.keys(),
            index=cols.keys(),
            dtype=PandasDtype.string,
        ).rename({0: "quantity"}, axis="columns")
        units = pd.DataFrame(
            {
                name: pd.Series(col.units, index=["units"])
                for name, col in cols.items()
                if not col.index
            },
            dtype=PandasDtype.string,
        ).T
        return pd.MultiIndex.from_frame(pd.concat([quantity, units], axis="columns"))

    def get_originlab_coldes(self) -> str:
        return "".join([column.originlab_coldes for column in self.cols.values()])
