import pandas as pd
from pydantic import Field, validator

from boilerdata.models.axes import Axes
from boilerdata.models.common import MyBaseModel, StrPath, load_config
from boilerdata.models.dirs import Dirs
from boilerdata.models.geometry import Geometry
from boilerdata.models.params import Params
from boilerdata.models.trials import Trial, Trials


class Project(MyBaseModel):
    """The global project configuration."""

    geometry: Geometry = Field(default_factory=Geometry)
    params: Params = Field(default_factory=Params)
    dirs: Dirs = Field(default_factory=Dirs)

    # ! AXES
    # Axes are specified separately and loading depends on this model's attributes.
    axes: Axes = Field(default=None)

    @validator("axes", always=True, pre=True)
    def validate_axes(cls, _, values):
        return load_config(values["dirs"].config / "axes.yaml", Axes)

    # ! TRIALS
    # Trials are specified separately and loading depends on this model's attributes.
    trials: list[Trial] = Field(default=None)

    @validator("trials", always=True, pre=True)
    def validate_trials(cls, _, values):
        trials = load_config(values["dirs"].config / "trials.yaml", Trials).trials
        for trial in trials:
            trial.setup(values["dirs"], values["geometry"])
        return trials

    # ! METHODS

    def get_trial(self, timestamp: pd.Timestamp) -> Trial:
        """Get a trial by its timestamp."""
        for trial in self.trials:
            if trial.timestamp == timestamp:
                return trial
        raise ValueError(f"Trial '{timestamp.date()}' not found.")

    @classmethod
    def get_project(cls, proj: StrPath = "params.yaml"):
        return load_config(proj, cls)
