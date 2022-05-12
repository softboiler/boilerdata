"""Enums to be used in models."""

from enum import Enum


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
