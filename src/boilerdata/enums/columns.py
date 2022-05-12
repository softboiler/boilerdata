from enum import auto, unique

from boilerdata.enums import NameEnum


@unique
class Column(NameEnum):
    T1 = auto()
