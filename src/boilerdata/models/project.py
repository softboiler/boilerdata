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

    # Use validator instead of `Field(default=Dirs())` to delay directory-creating
    # side-effects of calling default `Dirs()`.
    dirs: Dirs = Field(default=None)

    # "always" so it'll run even if not in YAML
    # "pre" because this must exist pre-validation
    @validator("dirs", always=True, pre=True)
    def validate_dirs(cls, v):
        return v or Dirs()

    geometry: Geometry
    params: Params = Field(default=Params())

    # These can't be None, as they are set in Project.__init__()
    axes: Axes = Field(default=None)
    trials: list[Trial] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.axes = load_config(self.dirs.config / "axes.yaml", Axes)
        self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
        for trial in self.trials:
            trial.setup(self.dirs, self.geometry)

    def get_trial(self, timestamp: pd.Timestamp) -> Trial:
        """Get a trial by its timestamp."""
        for trial in self.trials:
            if trial.timestamp == timestamp:
                return trial
        raise ValueError(f"Trial '{timestamp.date()}' not found.")

    @classmethod
    def get_project(cls, proj: StrPath = "config/project.yaml"):
        return load_config(proj, cls)
