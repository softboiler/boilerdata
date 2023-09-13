"""Types used throughout this package."""

from typing import Literal

Group = Literal["control", "porous", "hybrid"]
"""The group that this sample belongs to."""

Joint = Literal["paste", "epoxy", "solder", "none"]
"""The method used to join parts of the sample in this trial."""

Sample = Literal["B3"]
"""The sample attached to the coupon in this trial."""

PandasAggfun = Literal[
    "mean",
    "sum",
    "size",
    "count",
    "std",
    "var",
    "sem",
    "describe",
    "first",
    "last",
    "nth",
    "min",
    "max",
]
"""Valid built-in aggregation functions in Pandas.

See also: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#aggregation
"""

PandasDtype = Literal[
    "object",
    "float",
    "int",
    "bool",
    "timedelta64[ns]",
    "datetime64[ns]",
    "string[pyarrow]",
    "boolean",
    "category",
    "Sparse",
    "interval",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "Uint32",
    "UInt64",
]
"""Valid data types for Pandas objects.

    See also: https://pandas.pydata.org/docs/user_guide/basics.html#dtypes

"""

OriginLabColdes = Literal["X", "Y", "Z", "M", "E", "L", "G", "S", "N", "Q"]
"""Valid column designations for plotting in OriginLab.

    Designations:
        X: x-axis
        Y: y-axis
        Z: z-axis
        M: x-axis error
        E: y-axis error
        L: Label
        G: Group
        S: Subject
        N: None (Disregard)
        Q: Omit (If this column isn't in the output at all)
"""
