"""Manipulate trials."""

from enum import auto, unique

from pydantic import BaseModel, Field

from boilerdata.enums import GetNameEnum


@unique
class Rod(GetNameEnum):
    """The rod used in this trial."""

    W = auto()
    X = auto()
    Y = auto()


@unique
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


@unique
class Sample(GetNameEnum):
    """The sample attached to the coupon in this trial."""

    NA = auto()  # If no sample is attached to the coupon.
    B1 = auto()
    B2 = auto()
    B3 = auto()


@unique
class SampleType(GetNameEnum):
    """The type of sample studied in this trial."""

    control = auto()
    porous = auto()
    hybrid = auto()


@unique
class Joint(GetNameEnum):
    """The method used to join parts of the sample in this trial."""

    paste = auto()
    epoxy = auto()
    solder = auto()


class Trial(BaseModel):
    """A trial."""

    name: str
    rod: Rod
    coupon: Coupon
    sample: Sample
    sample_type: SampleType
    good: bool = Field(..., description="Whether this trial was deemed a success.")
    joint: Joint


class Trials(BaseModel):
    """The trials."""

    trials: list[Trial]
