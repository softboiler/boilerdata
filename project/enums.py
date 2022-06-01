"""Enums for trials and geometry."""

from enum import auto

from boilerdata.enums import GetNameEnum


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


# sourcery skip: avoid-builtin-shadow
class PandasDtype(GetNameEnum):
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


class OriginLabColdes(GetNameEnum):
    X = auto()
    Y = auto()
    Z = auto()
    M = auto()  # xEr+-
    E = auto()  # yEr+-
    L = auto()  # Label
    G = auto()  # Group
    S = auto()  # Subject
    N = auto()  # None
