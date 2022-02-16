from pathlib import Path

from dynaconf import Dynaconf
from pydantic.dataclasses import dataclass

import boilerdata

DEFAULT_CONFIG_FILENAME = "defaults.toml"
USER_CONFIG_FILENAME = "boilerdata.toml"
default_path = Path(boilerdata.__path__[0]) / DEFAULT_CONFIG_FILENAME  # type: ignore
user_path = next(Path().rglob(USER_CONFIG_FILENAME), Path(USER_CONFIG_FILENAME))
raw_config = Dynaconf(settings_files=[default_path, user_path])


@dataclass
class Config:
    """A validated configuration."""

    pass


config = Config()
