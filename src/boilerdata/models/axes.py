"""Hello."""

from pathlib import Path

import pandas as pd
from boilercore.models import YamlModel
from pydantic.v1 import BaseModel, Field, validator

from boilerdata.types import OriginLabColdes, PandasAggfun, PandasDtype


class Axis(BaseModel):
    """Metadata for a column in the dataframe."""

    # ! COMMON FIELDS

    name: str = Field(default=..., description="The name of the column.")

    dtype: PandasDtype = Field(
        default="float",
        description="The Pandas data type to be used to represent this column.",
    )

    units: str = Field(default="", description="The units for this column's values.")

    # ! AGGREGATION

    agg: PandasAggfun = Field(
        default="mean", description="The aggregation method to use for this column."
    )

    @validator("agg", always=True)
    @classmethod
    def validate_agg(cls, agg, values):
        return (lambda ser: ser.iloc[0]) if values["dtype"] == "category" else agg

    # ! COLUMNS IN SOURCE DATA

    source: str | None = Field(
        default=None,
        description="The name of the input column that this column is based off of.",
    )

    @validator("source")
    @classmethod
    def validate_source(cls, source, values):
        return f"{source} ({values['units']})" if values["units"] else source

    # ! INDEX

    index: bool = Field(
        default=False, description="Whether this column is to be the index."
    )

    @validator("index", pre=True, always=True)
    @classmethod
    def validate_index(cls, index):
        return index or False

    # ! META

    meta: bool = Field(
        default=False,
        description="Whether this column is informed by the trials config.",
    )

    # ! ORIGINLAB

    originlab_coldes: OriginLabColdes = Field(
        default="N", description="The column designation for plotting in OriginLab."
    )

    # ! PRETTY NAME

    # Can be None, but never accessed directly.
    pretty_name_: str | None = Field(
        default=None,
        alias="pretty_name",  # So we can make this a dynamic property below
        description="The pretty version of the column name.",
    )

    @property
    def pretty_name(self):
        return self.pretty_name_ or self.name


class Axes(YamlModel):
    """Columns in the dataframe."""

    all: list[Axis]

    def __getitem__(self, key):
        return next(axis for axis in self.all if axis.name == key)

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
    def source_cols(self) -> list[Axis]:
        return [col for col in self.source if not col.index]

    @property
    def meta(self) -> list[Axis]:
        return [ax for ax in self.all if ax.meta]

    @property
    def aggs(self) -> dict[str, pd.NamedAgg]:
        return {
            ax.name: pd.NamedAgg(column=ax.name, aggfunc=ax.agg) for ax in self.cols
        }

    def get_col_index(self) -> pd.MultiIndex:
        # Rename columns and extract them into a row
        quantity = pd.DataFrame(
            [ax.name for ax in self.cols],
            index=[ax.name for ax in self.cols],
            dtype="string[pyarrow]",
        ).rename(axis="columns", mapper={0: "quantity"})

        units = pd.DataFrame(
            {col.name: pd.Series(col.units, index=["units"]) for col in self.cols},
            dtype="string[pyarrow]",
        ).T

        return pd.MultiIndex.from_frame(
            pd.concat(axis="columns", objs=[quantity, units])
        )

    def get_originlab_coldes(self) -> str:
        return "".join([
            ax.originlab_coldes for ax in self.all if ax.originlab_coldes != "Q"
        ])

    def __init__(self, data_file: Path):
        super().__init__(data_file)
