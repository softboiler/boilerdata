from pathlib import Path

import numpy as np
from pydantic import BaseModel, DirectoryPath, Extra, Field, validator


class Fit(BaseModel):
    """Configure the linear regression of thermocouple temperatures vs. position."""

    thermocouple_pos: list[float] = Field(..., description="Thermocouple positions.")
    do_plot: bool = Field(False, description="Whether to plot the linear regression.")

    @validator("thermocouple_pos")
    def _(cls, thermocouple_pos):
        return np.array(thermocouple_pos)


class Project(BaseModel, extra=Extra.forbid):
    """Configuration for the package."""

    base: DirectoryPath = Field(
        ...,
        description="The base directory for the project data.",
    )
    trials: DirectoryPath = Field(
        ...,
        description="The directory in which the individual trials are. Must be relative to the base directory.",
    )
    directory_per_trial: Path = Field(
        ...,
        description="The directory in which the data are for a given trial. Must be relative to a trial folder, and all trials must share this pattern.",
    )
    fit: Fit

    @validator("trials", pre=True)
    def _(cls, trials, values):
        return values["base"] / Path(trials)
