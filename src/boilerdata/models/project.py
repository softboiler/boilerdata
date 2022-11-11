import pandas as pd
from pydantic import Field

from boilerdata.models.axes import Axes
from boilerdata.models.common import MyBaseModel, StrPath, allow_extra, load_config
from boilerdata.models.dirs import Dirs
from boilerdata.models.geometry import Geometry
from boilerdata.models.params import Params
from boilerdata.models.trials import Trial, Trials


class Project(MyBaseModel):
    """The global project configuration."""

    geometry: Geometry = Field(default_factory=Geometry)
    params: Params = Field(default_factory=Params)
    dirs: Dirs = Field(default_factory=Dirs)

    def __init__(self, **data):
        super().__init__(**data)
        with allow_extra(self):
            # Set up axes
            self.axes = load_config(self.dirs.config / "axes.yaml", Axes)
            # Set up trials
            trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
            for trial in trials:
                trial.setup(self.dirs, self.geometry)
            self.trials = trials

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
