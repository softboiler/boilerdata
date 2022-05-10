from enum import Enum, auto, unique


@unique
class C(Enum):
    T1 = auto()
    T2 = auto()
    T3 = auto()

    def __get__(self, *_):
        return self.name


C.T1
