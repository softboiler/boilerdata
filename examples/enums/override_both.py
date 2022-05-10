"""This does both."""

from enum import Enum, auto, unique


@unique
class C(Enum):
    @staticmethod
    def _generate_next_value_(name, *_):
        return name

    T1 = auto()
    T2 = auto()
    T3 = auto()

    def __get__(self, *_):
        return self.name


print(C.T1)
