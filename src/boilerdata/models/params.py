"""Project parameters."""

from os import environ
from pathlib import Path
from typing import Any, Literal, TypeAlias

import pandas as pd
from boilercore.models import SynchronizedPathsYamlModel
from boilercore.models.geometry import Geometry
from boilercore.models.params import FitParams
from pydantic import Extra, Field

from boilerdata import get_params_file
from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.axes import Axes
from boilerdata.models.paths import Paths
from boilerdata.models.trials import Trial, Trials

PARAMS_FILE = get_params_file()

Bound: TypeAlias = float | Literal["-inf", "inf"]


class Params(SynchronizedPathsYamlModel, extra=Extra.allow):
    """Global project parameters."""

    fit: FitParams = Field(
        default_factory=FitParams, description="Parameters for model fit."
    )

    records_to_average: int = Field(
        default=5,
        description="The number of records over which to average in a given trial.",
    )

    # ! EXCLUDED FROM PARAMS FILE

    copper_temps: list[str] = Field(
        default=[A.T_1, A.T_2, A.T_3, A.T_4, A.T_5],
        description="Copper temperature measurements.",
        exclude=True,
    )

    water_temps: list[str] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="Water temperature measurements.",
        exclude=True,
    )

    # ! PLOTTING
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )

    geometry: Geometry = Field(default_factory=Geometry)
    paths: Paths = Field(default_factory=Paths)

    def __init__(self, data_file: Path = PARAMS_FILE, **kwargs):
        super().__init__(data_file, **kwargs)
        self.axes = Axes(self.paths.axes_config)
        self.trials = Trials(self.paths.trials_config).trials
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


def init() -> tuple[Params, Any, Any, Any]:
    """Parameters and associated project setup.

    Assigned to module constants at the end of this module."""
    params = Params()

    # Override the default app folder
    environ["DYNACONF_APP_FOLDER"] = environ["APP_FOLDER_FOR_DYNACONF"] = str(
        params.paths.propshop
    )
    from propshop import get_prop
    from propshop.library import Mat, Prop

    return params, get_prop, Mat, Prop


PARAMS, get_prop, Mat, Prop = init()
"""All project parameters, including paths."""
