"""Manipulate trials."""

from enum import auto, unique
from pathlib import Path
from boilerdata.enums import GetNameEnum

from pydantic import BaseModel
import toml


@unique
class Rod(GetNameEnum):
    """The rod."""

    W = auto()
    X = auto()
    Y = auto()


@unique
class Coupon(GetNameEnum):
    """The coupon."""

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
    """The sample."""

    B1 = auto()
    B2 = auto()
    B3 = auto()


@unique
class SampleType(GetNameEnum):
    """The sample type."""

    control = auto()
    porous = auto()
    hybrid = auto()


@unique
class Joint(GetNameEnum):
    """The joint."""

    paste = auto()
    epoxy = auto()
    solder = auto()


class EnumValueBaseModel(BaseModel):
    class Config:
        use_enum_values = True


class Trial(BaseModel):
    """A trial."""

    name: str
    rod: Rod
    coupon: Coupon
    sample: Sample
    sample_type: SampleType
    good: bool
    joint: Joint


class Trials(BaseModel):
    """Trials."""

    trials: list[Trial]


def get_trials():
    a = {
        "trials": [
            {"test": 100, "best": 200, "rest": 300},
            {"test": 100, "best": 200, "rest": 300},
        ]
    }
    Path("test.toml").write_text(toml.dumps(a))
