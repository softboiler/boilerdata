import numpy as np
from pydantic import Field, validator

from boilerdata.models.common import MyBaseModel, NpNDArray
from boilerdata.models.enums import Coupon, Rod


class Geometry(MyBaseModel):
    """The geometry."""

    _in_p_m: float = 39.3701  # (in/m) Conversion factor

    # ! DIAMETER

    diameter: float = Field(
        default=...,
        description="The common diameter of all rods.",
    )

    @validator("diameter")
    def validate_diameter(cls, diameter):
        return diameter / cls._in_p_m

    # ! RODS

    rods: dict[Rod, NpNDArray] = Field(
        default=...,
        description="Distance of each thermocouple from the cool side of the rod, starting with TC1. Fifth thermocouple may be omitted. Input: inch. Output: meter.",
        # exclude=True,
    )

    @validator("rods", pre=True)
    def validate_rods(cls, rods):
        return {rod: np.array(values) / cls._in_p_m for rod, values in rods.items()}

    # ! COUPONS

    coupons: dict[Coupon, float] = Field(
        default=...,
        description="Length of the coupon. Input: inch. Output: meter.",
    )

    @validator("coupons")
    def validate_coupons(cls, coupons):
        return {coupon: value / cls._in_p_m for coupon, value in coupons.items()}
