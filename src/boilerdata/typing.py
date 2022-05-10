from os import PathLike
from typing import TypeVar

from pydantic.main import ModelMetaclass

StrPath = str | PathLike[str]
# Needs to be "ModelMetaclass" and not "BaseModel" for some weird reason...
PydanticModel = TypeVar("PydanticModel", bound=ModelMetaclass)
