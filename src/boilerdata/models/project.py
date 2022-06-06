from typing import Optional

import pandas as pd
from pydantic import Field, validator

from boilerdata.axes import Axes as A  # noqa: N817
from boilerdata.enums import OriginLabColdes, PandasDtype
from boilerdata.models.common import MyBaseModel, load_config
from boilerdata.models.dirs import Dirs
from boilerdata.models.geometry import Geometry
from boilerdata.models.trials import Trial, Trials


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
    water_temps: list[A] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="The water temperature measurements.",
    )
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )


# * -------------------------------------------------------------------------------- * #
# * AXES


class Axis(MyBaseModel):
    """Metadata for a column in the dataframe."""

    # ! COMMON FIELDS

    name: str = Field(
        default=...,
        description="The name of the column.",
    )

    dtype: PandasDtype = Field(
        default=PandasDtype.float,
        description="The Pandas data type to be used to represent this column.",
    )

    units: str = Field(
        default="",
        description="The units for this column's values.",
    )

    # ! COLUMNS IN SOURCE DATA

    source: Optional[str] = Field(
        default=None,
        description="The name of the input column that this column is based off of.",
    )

    @validator("source")
    def validate_source(cls, source, values):
        return f"{source} ({values['units']})" if values["units"] else source

    # ! INDEX

    index: bool = Field(
        default=False,
        description="Whether this column is to be the index.",
    )

    @validator("index", pre=True, always=True)
    def validate_index(cls, index):
        return index or False

    # ! META

    meta: bool = Field(
        default=False,
        description="Whether this column is informed by the trials config.",
    )

    # ! ORIGINLAB

    originlab_coldes: OriginLabColdes = Field(
        default="N",
        description="The column designation for plotting in OriginLab.",
    )

    # ! PRETTY NAME

    # Can be None, but never accessed directly.
    pretty_name_: Optional[str] = Field(
        default=None,
        alias="pretty_name",  # So we can make this a dynamic property below
        description="The pretty version of the column name.",
    )

    @property
    def pretty_name(self):
        return self.pretty_name_ or self.name


class Axes(MyBaseModel):
    """Columns in the dataframe."""

    all: list[Axis]  # noqa: A003

    @property
    def index(self) -> list[Axis]:
        index = [ax for ax in self.all if ax.index]
        if index and index[-1].source:
            return index
        elif index:
            raise ValueError("The last (or only) index must have a source.")
        else:
            raise ValueError("No index.")

    @property
    def cols(self) -> list[Axis]:
        return [ax for ax in self.all if not ax.index]

    @property
    def source(self) -> list[Axis]:
        if source := [ax for ax in self.all if ax.source]:
            return source
        else:
            raise ValueError("No source columns.")

    @property
    def meta(self) -> list[Axis]:
        return [ax for ax in self.all if ax.meta]

    def get_col_index(self) -> pd.MultiIndex:

        # Rename columns and extract them into a row
        quantity = pd.DataFrame(
            get_names(self.cols),
            index=get_names(self.cols),
            dtype=PandasDtype.string,
        ).rename({0: "quantity"}, axis="columns")

        units = pd.DataFrame(
            {col.name: pd.Series(col.units, index=["units"]) for col in self.cols},
            dtype=PandasDtype.string,
        ).T

        return pd.MultiIndex.from_frame(pd.concat([quantity, units], axis="columns"))

    def get_originlab_coldes(self) -> str:
        return "".join([ax.originlab_coldes for ax in self.all])


# * -------------------------------------------------------------------------------- * #
# * PROJECT


class Project(MyBaseModel):
    """Configuration for the package."""

    dirs: Dirs
    geometry: Geometry
    params: Params

    # These can't be None, as they are set in Project.__init__()
    trials: list[Trial] = Field(default=None)
    axes: Axes = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        # Get the Columns instance
        self.axes = load_config(self.dirs.config / "axes.yaml", Axes)

        # Get the trials field of the Trials instance. Ensure trials are populated.
        self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
        for trial in self.trials:
            trial.setup(self.dirs, self.geometry)


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_project():
    return load_config("src/boilerdata/config/project.yaml", Project)


def get_names(axes: list[Axis]) -> list[str]:
    """Get names of the axes."""
    return [ax.name for ax in axes]


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    return df.assign(**{name: df[name].astype(dtype) for name, dtype in dtypes.items()})
