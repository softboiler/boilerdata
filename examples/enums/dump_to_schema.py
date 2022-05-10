from enum import Enum, unique, auto
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


class MyModel(BaseModel):
    """Docstring for MyModel"""

    class Config:
        use_enum_values = True

    my_enum: MyEnum = Field(..., description="Description for this field.")


Path("examples/enums/test_schema.json").write_text(MyModel.schema_json(indent=2))
print(MyModel(my_enum=MyEnum.T2).my_enum)
print(MyModel(my_enum=MyEnum.T4))  # type: ignore  # should fail
