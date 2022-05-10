"""Makes the Enum return its keys as its values."""

from enum import Enum, auto, unique


@unique
class C(Enum):
    @staticmethod
    def _generate_next_value_(name, *_):
        return name

    T1 = auto()
    T2 = auto()
    T3 = auto()


print(C.T1)
