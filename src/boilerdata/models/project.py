import pandas as pd
from pydantic import Field
from ruamel.yaml import YAML

from boilerdata import AXES_CONFIG, PARAMS_FILE, TRIAL_CONFIG
from boilerdata.models import SynchronizedPathsYamlModel
from boilerdata.models.axes import Axes
from boilerdata.models.common import allow_extra
from boilerdata.models.dirs import Dirs, ProjectPaths
from boilerdata.models.geometry import Geometry
from boilerdata.models.params import Params
from boilerdata.models.trials import Trial, Trials

yaml = YAML()
yaml.indent(2)


class Project(SynchronizedPathsYamlModel):
    """The global project configuration."""

    geometry: Geometry = Field(default_factory=Geometry)
    params: Params = Field(default_factory=Params)
    project_paths: ProjectPaths = Field(default_factory=Dirs)
    dirs: Dirs = Field(default_factory=Dirs)

    def __init__(self):
        super().__init__(PARAMS_FILE)
        with allow_extra(self):
            self.axes = Axes(**yaml.load(AXES_CONFIG))
            trials = Trials(**yaml.load(TRIAL_CONFIG)).trials
            for trial in trials:
                trial.setup(self.dirs, self.geometry, self.params.copper_temps)
            self.trials = trials

    # ! METHODS

    def get_trial(self, timestamp: pd.Timestamp) -> Trial:
        """Get a trial by its timestamp."""
        for trial in self.trials:
            if trial.timestamp == timestamp:
                return trial
        raise ValueError(f"Trial '{timestamp.date()}' not found.")


PROJ = Project()
"""All project parameters, including paths."""
