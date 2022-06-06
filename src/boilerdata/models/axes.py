from typing import Optional

import pandas as pd
from pydantic import Field, validator

from boilerdata.enums import OriginLabColdes, PandasDtype
from boilerdata.models.common import MyBaseModel

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
# * HELPER FUNCTIONS


def get_names(axes: list[Axis]) -> list[str]:
    """Get names of the axes."""
    return [ax.name for ax in axes]
