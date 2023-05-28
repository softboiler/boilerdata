from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from pydantic import Field, validator

from boilerdata import AXES_CONFIG, PARAMS_FILE, TRIAL_CONFIG
from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models import (
    ProjectModel,
    SynchronizedPathsYamlModel,
    allow_extra,
    default_opt,
)
from boilerdata.models.axes import Axes
from boilerdata.models.enums import FitMethod
from boilerdata.models.geometry import Geometry
from boilerdata.models.paths import Paths, ProjectPaths
from boilerdata.models.trials import Trial, Trials

bound: TypeAlias = float | Literal["-inf", "inf"]


class Params(SynchronizedPathsYamlModel):
    """The global project configuration."""

    Config = ProjectModel.Config  # type: ignore

    records_to_average: int = Field(
        default=default_opt(5),
        description="The number of records over which to average in a given trial.",
    )

    fit_method: FitMethod = Field(default=default_opt(FitMethod.trf, optional=False))

    model_params: list[A] = Field(
        default=default_opt([A.T_s, A.q_s, A.k, A.h_a, A.h_w]),
        description="Parameters that can vary in the model. Some will be fixed.",
    )

    model_inputs: dict[str, float] = Field(
        default=default_opt(
            dict(
                r=0.0047625,  # (m)
                T_infa=25.0,  # (C)
                T_infw=100.0,  # (C)
                x_s=0,  # (m)
                x_wa=0.0381,  # (m)
            )
        ),
        description="Inputs to the symbolic model float evaluation stage.",
    )

    # ! MODEL BOUNDS

    model_bounds: dict[A, tuple[bound, bound]] = Field(
        default=default_opt(
            {
                A.T_s: (95, "inf"),  # (C) T_s
                A.q_s: (0, "inf"),  # (W/m^2) q_s
                A.k: (350, 450),  # (W/m-K) k
                A.h_a: (0, "inf"),  # (W/m^2-K) h_a
                A.h_w: (0, "inf"),  # (W/m^2-K) h_w
            }
        ),
        description="Bounds for the model parameters. Not used if parameter is fixed.",
    )

    @validator("model_bounds", always=True)
    def validate_model_bounds(cls, model_bounds) -> dict[A, tuple[float, float]]:
        """Substitute inf for np.inf."""
        for param, b in model_bounds.items():
            b0 = -np.inf if isinstance(b[0], str) and "-inf" in b[0] else b[0]
            b1 = np.inf if isinstance(b[0], str) and "inf" in b[1] else b[1]
            model_bounds[param] = (b0, b1)
        return model_bounds

    # ! FIXED PARAMS

    fixed_params: list[A] = Field(
        default=default_opt(
            [
                A.k,
                A.h_w,
            ]
        ),
        description="Parameters to fix. Evaluated before fitting, overridable in code.",
    )

    @validator("fixed_params", always=True, each_item=True)
    def validate_each_fixed_param(cls, param, values):
        """Check that the fixed parameter is one of the model parameters."""
        if param in values["model_params"]:
            return param
        raise ValueError(f"Fixed parameter {param} not in model parameters")

    # ! INITIAL VALUES

    initial_values: dict[A, float] = Field(
        default=default_opt(
            {
                A.T_s: 95,  # (C) T_s
                A.q_s: 0,  # (W/m^2) q_s
                A.k: 400,  # (W/m-K) k
                A.h_a: 0,  # (W/m^2-K) h_a
                A.h_w: 0,  # (W/m^2-K) h_w
            }
        ),
        description="Initial guess for free parameters, constant value otherwise.",
    )

    @validator("initial_values", always=True)
    def validate_initial_values(cls, model_inputs) -> dict[str, float]:
        """Avoid division by zero in select parameters."""
        eps = float(np.finfo(float).eps)
        params_to_check = [A.h_a, A.h_w]
        return {
            param: eps if v == 0 and param in params_to_check else v
            for param, v in model_inputs.items()
        }

    # ! EXCLUDED FROM PARAMS FILE

    copper_temps: list[A] = Field(
        default=[A.T_1, A.T_2, A.T_3, A.T_4, A.T_5],
        description="Copper temperature measurements.",
        exclude=True,
    )

    water_temps: list[A] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="Water temperature measurements.",
        exclude=True,
    )

    # ! PLOTTING

    do_plot: bool = Field(
        default=default_opt(False),
        description="Whether to plot the fits of the individual runs.",
    )

    geometry: Geometry = Field(default_factory=Geometry)
    project_paths: ProjectPaths = Field(default_factory=Paths)
    paths: Paths = Field(default_factory=Paths)

    def __init__(self):
        super().__init__(PARAMS_FILE)
        with allow_extra(self):
            self.axes = Axes(AXES_CONFIG)
            self.trials = Trials(TRIAL_CONFIG).trials
            self.free_params = [
                p for p in self.model_params if p not in self.fixed_params
            ]
            self.free_errors = self.get_model_errors(self.free_params)
            self.model_errors = self.get_model_errors(self.model_params)
            self.fixed_errors = self.get_model_errors(self.fixed_params)
            self.params_and_errors = self.model_params + self.model_errors
            self.fixed_values = {
                k: v for k, v in self.initial_values.items() if k in self.fixed_params
            }
        for trial in self.trials:
            trial.setup(self.paths, self.geometry, self.copper_temps)

    # ! METHODS

    def get_trial(self, timestamp: pd.Timestamp) -> Trial:
        """Get a trial by its timestamp."""
        for trial in self.trials:
            if trial.timestamp == timestamp:
                return trial
        raise ValueError(f"Trial '{timestamp.date()}' not found.")

    def get_model_errors(self, params) -> list[str]:
        return [f"{param}_err" for param in params]


PARAMS = Params()
"""All project parameters, including paths."""
