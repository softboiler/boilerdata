"""Manipulate trials."""

from pathlib import Path

from pydantic import BaseModel
import toml

from boilerdata.enums import Coupon, Joint, Rod, Sample, SampleType


class EnumValueBaseModel(BaseModel):
    class Config:
        use_enum_values = True


class Trial(EnumValueBaseModel):
    """A trial."""

    name: str
    rod: Rod
    coupon: Coupon
    sample: Sample
    sample_type: SampleType
    good: bool
    joint: Joint


class Trials(BaseModel):
    """Trials."""

    trials: list[Trial]


def get_trials():
    a = {
        "trials": [
            {"test": 100, "best": 200, "rest": 300},
            {"test": 100, "best": 200, "rest": 300},
        ]
    }
    Path("test.toml").write_text(toml.dumps(a))
