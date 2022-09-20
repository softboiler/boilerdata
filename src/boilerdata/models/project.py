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
    params: Params

    # These can't be None, as they are set in Project.__init__()
    trials: list[Trial] = Field(default=None)
    axes: Axes = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        # Get the Columns instance
        self.axes = load_config(self.dirs.config / "axes.yaml", Axes)

        # Get the trials field of the Trials instance. Ensure trials are populated.
        self.trials = load_config(self.dirs.config / "trials.yaml", Trials).trials
        for trial in self.trials:
            trial.setup(self.dirs, self.geometry)

    @classmethod
    def get_project(cls, proj: StrPath = "config/project.yaml"):
        return load_config(proj, cls)
