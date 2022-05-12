from enum import auto

from boilerdata.enums import NameEnum
from boilerdata.trials import Trials


class Model(NameEnum):
    Trials = auto()


model_from_cli = {Model.Trials: Trials}
