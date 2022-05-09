from pathlib import Path

import numpy as np
from dynaconf import Dynaconf
from pydantic import BaseModel, DirectoryPath, Field, validator


def parse_tilde_in_path(path: str) -> Path:
    return Path.home() / path.lstrip("~/") if path.startswith("~/") else Path(path)


APP_FOLDER = Path(parse_tilde_in_path("~/.boilerdata"))


class Fit(BaseModel):
    """Configure the linear regression of thermocouple temperatures vs. position."""

    thermocouple_pos: list[float] = Field(
        ...,
        description="Thermocouple positions.",
    )
    do_plot: bool = Field(False, description="Whether to plot the linear regression.")

    @validator("thermocouple_pos")
    def _(cls, thermocouple_pos):
        return np.array(thermocouple_pos)


class Boilerdata(BaseModel):
    """Configuration for the package."""

    data: DirectoryPath = Field(
        ...,
        description='Absolute or relative path to a folder containing a subfolder "raw" which has CSVs of experimental runs.',
    )
    fit: Fit


def load_config(path: str = "boilerdata.toml"):
    """Load the configuration file."""

    config = Path(path)
    if config.exists():
        raw_config = Dynaconf(settings_files=[config])
    else:
        raise FileNotFoundError(f"Configuration file {config.name} not found.")

    APP_FOLDER.mkdir(parents=False, exist_ok=True)

    return Boilerdata(data=raw_config.get("data"), fit=Fit(**raw_config.fit))


def write_schema(directory: str):
    (Path(directory) / "boilerdata.toml.json").write_text(
        Boilerdata.schema_json(indent=2)
    )
