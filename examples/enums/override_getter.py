"""Makes the Enum return the name of the key when called, eg: `C.T1` yields `"T1`."""

from enum import Enum, auto, unique


@unique
class C(Enum):
    T1 = auto()
    T2 = auto()
    T3 = auto()

    def __get__(self, *_):
        return self.name


print(C.T1)
