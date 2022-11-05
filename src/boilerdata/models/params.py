from __future__ import annotations

import numpy as np
from pydantic import Field, validator

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel, allow_extra


class Params(MyBaseModel):
    """Parameters that can vary."""

    records_to_average: int = Field(
        default=5,
        description="The number of records over which to average in a given trial.",
    )

    model_inputs: dict[str, float] = Field(
        default=dict(
            r=0.0047625,  # (m)
            T_infa=25.0,  # (C)
            T_infw=100.0,  # (C)
            x_s=0,  # (m)
            x_wa=0.0381,  # (m)
            h_w=0,  # (W/m^2-K)  # TODO: Move this back
        ),
        description="The inputs to the model to be fitted.",
    )

    @validator("model_inputs", always=True)
    def validate_model_inputs(cls, model_inputs) -> dict[str, float]:
        """Substitute this instead of zero to avoid division by zero"""
        eps = float(np.finfo(float).eps)
        return {
            k: eps if v == 0 and "x" not in k else v for k, v in model_inputs.items()
        }

    # Reason: pydantic: use_enum_values
    model_params: list[A] = [
        A.T_s,
        A.q_s,
        # A.k,
        A.h_a,
        # A.h_w,  # TODO: Uncomment this
    ]  # type: ignore
    fixed_params: list[A] = [
        A.k,
        # A.h_w,  # TODO: Uncomment this
    ]  # type: ignore

    water_temps: list[A] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="Water temperature measurements.",
    )
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )
    plots: list[str] = Field(
        default=["lit_"],
        description="List of plots to save.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        with allow_extra(self):
            self.free_params = [
                p for p in self.model_params if p not in self.fixed_params
            ]
            self.free_errors = self.get_model_errors(self.free_params)
            self.model_errors = self.get_model_errors(self.model_params)
            self.fixed_errors = self.get_model_errors(self.fixed_params)
            self.params_and_errors = self.model_params + self.model_errors

    def get_model_errors(self, model_params) -> list[str]:
        return [f"{param}_err" for param in model_params]
