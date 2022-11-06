from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from pydantic import Field, validator

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel, allow_extra

bound: TypeAlias = float | Literal["-inf", "inf"]


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
        ),
        description="The inputs to the model to be fitted.",
    )

    # Reason: pydantic: use_enum_values
    model_params: list[A] = [
        A.T_s,
        A.q_s,
        A.k,
        A.h_a,
        A.h_w,
    ]  # type: ignore

    # ! MODEL BOUNDS

    # Reason: pydantic: use_enum_values
    model_bounds: dict[A, tuple[bound, bound]] = {
        A.T_s: (95, "inf"),  # (C) T_s
        A.q_s: (0, "inf"),  # (W/m^2) q_s
        A.k: (350, 450),  # (W/m-K) k
        A.h_a: (0, "inf"),  # (W/m^2-K) h_a
        A.h_w: (0, "inf"),  # (W/m^2-K) h_w
    }  # type: ignore

    @validator("model_bounds", always=True)
    def validate_model_bounds(cls, model_bounds) -> dict[A, tuple[float, float]]:
        """Substitute inf for np.inf"""
        for param, b in model_bounds.items():
            b0 = -np.inf if isinstance(b[0], str) and "inf" in b[0] else b[0]
            b1 = np.inf if isinstance(b[0], str) and "inf" in b[1] else b[1]
            model_bounds[param] = (b0, b1)
        return model_bounds

    # ! FIXED PARAMS

    fixed_params: list[A] = [
        A.k,
        A.h_w,
    ]  # type: ignore

    @validator("fixed_params", always=True, each_item=True)
    def validate_each_fixed_param(cls, param, values):
        """Substitute inf for np.inf"""
        if param in values["model_params"]:
            return param
        raise ValueError(f"Fixed parameter {param} not in model parameters")

    # ! INITIAL VALUES

    initial_values: dict[A, float] = {
        A.T_s: 95,  # (C) T_s
        A.q_s: 0,  # (W/m^2) q_s
        A.k: 400,  # (W/m-K) k
        A.h_a: 0,  # (W/m^2-K) h_a
        A.h_w: 0,  # (W/m^2-K) h_w
    }  # type: ignore

    @validator("initial_values", always=True)
    def validate_initial_values(cls, model_inputs) -> dict[str, float]:
        """Substitute near zero instead of zero to avoid division by zero."""
        eps = float(np.finfo(float).eps)
        params_to_check = (A.h_a, A.h_w)
        return {
            param: eps if v == 0 and param in params_to_check else v
            for param, v in model_inputs.items()
        }

    # !

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
