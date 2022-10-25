import pandas as pd
from pydantic import Field, validator

from boilerdata.models.axes import Axes
from boilerdata.models.common import MyBaseModel, StrPath, load_config
from boilerdata.models.dirs import Dirs
from boilerdata.models.geometry import Geometry
from boilerdata.models.params import Params
from boilerdata.models.trials import Trial, Trials


class Project(MyBaseModel):
    """Configuration for the package."""

    geometry: Geometry
    params: Params = Field(default=Params())

    # ! DIRS
    # Use validator instead of `Field(default=Dirs())` to delay directory-creating
    # side-effects of calling default `Dirs()`.
    dirs: Dirs = Field(default=None)

    @validator("dirs", always=True, pre=True)
    def validate_dirs(cls, v):
        return v or Dirs()

    # ! AXES
    axes: Axes = Field(default=None)

    @validator("axes", always=True, pre=True)
    def validate_axes(cls, v, values):
        return load_config(values["dirs"].config / "axes.yaml", Axes)

    # ! TRIALS
    trials: list[Trial] = Field(default=None)

    @validator("trials", always=True, pre=True)
    def validate_trials(cls, v, values):
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
    def get_project(cls, proj: StrPath = "config/project.yaml"):
        return load_config(proj, cls)
