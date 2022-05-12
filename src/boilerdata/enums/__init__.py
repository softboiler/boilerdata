"""Enums to be used in models."""

from enum import Enum, auto, unique


class NameEnum(Enum):
    """Enum names get assigned to values when `auto()` is used."""

    @staticmethod
    def _generate_next_value_(name: str, *_) -> str:
        return name


class GetNameEnum(Enum):
    """When getting a value from an enum, return the name."""

    @staticmethod
    def _generate_next_value_(name: str, *_) -> str:
        return name

    def __get__(self, *_):
        return self.name


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
