from dataclasses import asdict
from pathlib import Path

from dynaconf import Dynaconf
import numpy as np
from pydantic import DirectoryPath, validator
from pydantic.dataclasses import dataclass


@dataclass
class FitParams:
    thermocouple_pos: list[float]
    do_plot: bool

    @validator("thermocouple_pos")
    def _(cls, thermocouple_pos):
        return np.array(thermocouple_pos)


@dataclass
class Config:
    data_path: DirectoryPath
    fit_params: FitParams

    @validator("fit_params")
    def _(cls, param):
        return asdict(param)


CONFIG = Path("boilerdata.toml")
if CONFIG.exists():
    raw_config = Dynaconf(settings_files=[Path("boilerdata.toml")])
else:
    raise FileNotFoundError("Configuration file boilerdata.toml not found.")

config = Config(data_path=raw_config.data_path, fit_params=FitParams(**raw_config.fit))


def write_schema(directory: str):
    (Path(directory) / "boilerdata.toml.json").write_text(
        config.__pydantic_model__.schema_json()
    )
