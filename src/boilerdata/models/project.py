from pydantic import Field

from boilerdata.models.axes import Axes
from boilerdata.models.common import MyBaseModel, StrPath, load_config
from boilerdata.models.dirs import Dirs
from boilerdata.models.geometry import Geometry
from boilerdata.models.params import Params
from boilerdata.models.trials import Trial, Trials


class Project(MyBaseModel):
    """Configuration for the package."""

    dirs: Dirs
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

    @classmethod
    def get_project(cls, proj: StrPath = "config/project.yaml"):
        return load_config(proj, cls)
