"""Enums to be used in models."""

from enum import Enum
from pathlib import Path
from textwrap import dedent


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


def generate_columns_enum(columns: list[str], path: Path):
    """Given a list of column names, generate a Python script with columns as enums."""
    text = dedent(
        """\
        # flake8: noqa

        from enum import auto

        from boilerdata.enums import GetNameEnum


        class Columns(GetNameEnum):
        """
    )
    for label in columns:
        text += f"    {label} = auto()\n"
    path.write_text(text, encoding="utf-8")
