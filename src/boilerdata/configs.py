from pathlib import Path
from typing import TypeVar

import toml
from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from boilerdata.typing import StrPath


def expanduser2(path: str) -> Path:
    """Expand the "~" user construction.

    Unlike the builtin `posixpath.expanduser`, this always works on Windows, and returns
    a `pathlib.Path` object.

    Parameters
    ----------
    path: str
        A string that may contain "~" at the start.

    Returns
    -------
    pathlib.Path
        The path after user expansion.

    """
    home = "~/"
    return Path.home() / path.lstrip(home) if path.startswith(home) else Path(path)


def get_path(path: StrPath) -> Path:
    """Generate `pathlib.Path` from various inputs.

    Handle the "~" user construction if necessary and return a `pathlib.Path` object.

    Parameters
    ----------
    path: str | PathLike[str]
        The path.

    Returns
    -------
    pathlib.Path
        The path after handling.

    """
    if isinstance(path, str):
        path = expanduser2(path)
    return Path(path)


APP_FOLDER = Path(get_path("~/.boilerdata"))

PydanticModel = TypeVar("PydanticModel", bound=ModelMetaclass)


def load_config(path: StrPath, model: PydanticModel) -> PydanticModel:
    """Load a configuration file."""

    config = Path(path)
    if config.exists():
        raw_config = toml.load(config)
    else:
        raise FileNotFoundError(f"Configuration file {config.name} not found.")

    APP_FOLDER.mkdir(parents=False, exist_ok=True)

    return model(**{key: raw_config.get(key) for key in model.__fields__.keys()})  # type: ignore


def write_schema(directory: str, model: type[BaseModel]):
    (Path(directory) / "boilerdata.toml.json").write_text(model.schema_json(indent=2))
