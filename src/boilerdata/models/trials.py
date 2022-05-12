"""Manipulate trials."""

from datetime import date
from enum import auto, unique

from pydantic import BaseModel, Field

from boilerdata.enums import GetNameEnum
from boilerdata.models import ExtraForbidBaseModel

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


class Trial(ExtraForbidBaseModel):
    """A trial."""

    date: date
    name: str
    rod: Rod
    coupon: Coupon
    sample: Sample
    group: Group
    monotonic: bool = Field(..., description="Whether the boiling curve is monotonic.")
    joint: Joint
    hello: str


class Trials(BaseModel):
    """The trials."""

    trials: list[Trial]


def get_trials():
    pass
