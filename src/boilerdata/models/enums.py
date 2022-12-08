"""Enums to be used in models."""

from enum import Enum, auto


class NameEnum(Enum):
    """Enum names get assigned to values when `auto()` is used."""

    @staticmethod
    def _generate_next_value_(name: str, *_) -> str:
        return name


class GetNameEnum(NameEnum):
    """When getting a value from an enum, return the name."""

    def __get__(self, *_):
        return self.name


class GetValueNameEnum(NameEnum):
    """When getting a value from an enum, return the value."""

    def __get__(self, *_):
        return self.name


class Rod(GetNameEnum):
    """The rod used in this trial."""

    W = auto()
    X = auto()
    Y = auto()
    R = auto()


class Coupon(GetNameEnum):
    """The coupon attached to the rod for this trial."""

    A0 = auto()
    A1 = auto()
    A2 = auto()
    A3 = auto()
    A4 = auto()
    A5 = auto()
    A6 = auto()
    A7 = auto()
    A8 = auto()
    A9 = auto()


class Sample(GetNameEnum):
    """The sample attached to the coupon in this trial."""

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
    none = auto()


class FitMethod(GetNameEnum):
    """Valid methods for curve fitting."""

    lm = auto()
    trf = auto()
    dogbox = auto()


class PandasDtype(GetNameEnum):
    """Valid data types for Pandas objects.

    See also: https://pandas.pydata.org/docs/user_guide/basics.html#dtypes
    """

    object = auto()  # noqa: A003
    float = auto()  # noqa: A003
    int = auto()  # noqa: A003
    bool = auto()  # noqa: A003
    timedelta64ns = "timedelta64[ns]"
    datetime64ns = "datetime64[ns]"
    string = "string[pyarrow]"
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


class PandasAggfun(GetNameEnum):
    """Valid built-in aggregation functions in Pandas.

    See also: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#aggregation
    """

    mean = auto()
    sum = auto()  # noqa: A003
    size = auto()
    count = auto()
    std = auto()
    var = auto()
    sem = auto()
    describe = auto()
    first = auto()
    last = auto()
    nth = auto()
    min = auto()  # noqa: A003
    max = auto()  # noqa: A003


class OriginLabColdes(GetNameEnum):
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

    X = auto()
    Y = auto()
    Z = auto()
    M = auto()  # xEr+-
    E = auto()  # yEr+-
    L = auto()  # Label
    G = auto()  # Group
    S = auto()  # Subject
    N = auto()  # None (Disregard)
    Q = auto()  # Omit (If this column isn't in the output at all)
