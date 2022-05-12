from pydantic import BaseModel, Extra


class EnumValueBaseModel(BaseModel):
    class Config:
        use_enum_values = True


class ExtraForbidBaseModel(BaseModel):
    class Config:
        extra = Extra.forbid
