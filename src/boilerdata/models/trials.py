"""Manipulate trials."""

from datetime import date
from enum import auto, unique

from pydantic import BaseModel, Extra, Field

from boilerdata.enums import GetNameEnum

# * -------------------------------------------------------------------------------- * #
# * ENUMS


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
    A6 = auto()
    A7 = auto()
    A9 = auto()


@unique
class Sample(GetNameEnum):
    """The sample attached to the coupon in this trial."""

    NA = auto()  # If no sample is attached to the coupon.
    B3 = auto()


@unique
class Group(GetNameEnum):
    """The group that this sample belongs to."""

    control = auto()
    porous = auto()
    hybrid = auto()


@unique
class Joint(GetNameEnum):
    """The method used to join parts of the sample in this trial."""

    paste = auto()
    epoxy = auto()
    solder = auto()


# * -------------------------------------------------------------------------------- * #
# * MODELS


class Trial(BaseModel, extra=Extra.forbid):
    """A trial."""

    date: date
    name: str
    rod: Rod
    coupon: Coupon
    sample: Sample
    group: Group
    monotonic: bool = Field(..., description="Whether the boiling curve is monotonic.")
    joint: Joint


class Trials(BaseModel):
    """The trials."""

    trials: list[Trial]
