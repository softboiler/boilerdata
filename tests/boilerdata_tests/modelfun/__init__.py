"""Modelfun tests."""

from typing import NamedTuple

from numpy.typing import ArrayLike


class MFParam(NamedTuple):
    id_: str
    y: ArrayLike
    expected: dict[str, float]
