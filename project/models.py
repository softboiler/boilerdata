from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, DirectoryPath, Field, validator

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
    @validator("results_file", "coldes_file", always=True)
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

    # Can't be None. Set in Project.__init__()
    path: DirectoryPath = Field(default=None)
    thermocouple_pos: NpNDArray = Field(default=None, exclude=True)

    def get_path(self, dirs: Dirs):
        """Get the path to the data for this trial. Called during project setup."""
        self.path = dirs.trials / self.date.isoformat() / dirs.per_trial

    def get_geometry(self, geometry: Geometry):
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
        self.cols = load_config(self.dirs.config / "columns.yaml", Columns).cols

        for trial in self.trials:
            trial.get_path(self.dirs)
            trial.get_geometry(self.geometry)

    def get_index(self) -> Column:
        index_cols = [col for col in self.cols.values() if col.index]
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
        return "".join([column.originlab_coldes for column in self.cols.values()])
