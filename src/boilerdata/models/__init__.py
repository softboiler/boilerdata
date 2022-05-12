from pydantic import BaseModel


class EnumValueBaseModel(BaseModel):
    class Config:
        use_enum_values = True
