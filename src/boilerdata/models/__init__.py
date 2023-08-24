"""Parameter models for this project."""

from __future__ import annotations

from types import EllipsisType
from typing import TypeVar

T = TypeVar("T")


def default_opt(default: T, optional: bool = False) -> EllipsisType | T:
    """Has a default that will be passed to a Pydantic model if optional.
    It is useful to set `optional` to `True` when actively developing a parameter, then
    revert it to `False` when that parameter is going to always be coming from a
    configuration file.
    """
    return default if optional else ...
