from enum import Enum, auto, unique
from pathlib import Path

from pydantic import BaseModel, Field


@unique
class MyEnum(Enum):
    """Docstring for MyEnum."""

    @staticmethod
    def _generate_next_value_(name, *_):
        return name

    T1 = auto()
    T2 = auto()
    T3 = auto()


class EnumModel(BaseModel):
    """Docstring for EnumModel."""

    class Config:
        use_enum_values = True

    my_enum: MyEnum = Field(..., description="Description for my_enum.")


class MyModel(BaseModel):
    """Docstring for MyModel."""

    my_list: list[EnumModel] = Field(..., description="Description for my_list.")


if __name__ == "__main__":
    Path("examples/enums/test_schema.json").write_text(MyModel.schema_json(indent=2))
    my_model = MyModel(
        my_list=[EnumModel(my_enum=enum) for enum in [MyEnum.T1, MyEnum.T2, MyEnum.T3]]
    )
    print(my_model)
