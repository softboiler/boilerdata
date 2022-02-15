from pathlib import Path
from dynaconf import Dynaconf
from pydantic import FilePath, validator
from pydantic.dataclasses import dataclass

import boilerdata

default_path = Path(boilerdata.__path__[0]) / "default_settings.toml"  # type: ignore
user_path = next(Path().rglob("settings.toml"), Path())
_settings = Dynaconf(
    envvar_prefix="BOILERDATA", settings_files=[default_path, user_path]
)


@dataclass
class Settings:
    ees: FilePath

    @validator("ees")
    def validate_ees(cls, ees):
        if ees.name != "EES.exe":
            raise ValueError("Filename must be 'EES.exe'.")
        return ees


settings = Settings(ees=_settings.ees)
...
