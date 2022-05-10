from pathlib import Path
from typing import Optional, TypeVar

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


def get_file(path: StrPath) -> Path:
    """Generate `pathlib.Path` to a file that exists.

    Handle the "~" user construction if necessary and return a `pathlib.Path` object.
    Raise exception if the file is not found.

    Parameters
    ----------
    path: StrPath
        The path.

    Returns
    -------
    pathlib.Path
        The path after handling.

    Raises
    ------
    FleNotFoundError
        If the file doesn't exist or does not refer to a file.

    """
    path = expanduser2(path) if isinstance(path, str) else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    elif not path.is_file():
        raise FileNotFoundError(f"The path '{path}' does not refer to a file.")
    else:
        return path


PydanticModel = TypeVar("PydanticModel", bound=ModelMetaclass)


def load_config(
    path: StrPath, model: PydanticModel
) -> tuple[PydanticModel, Optional[str]]:
    """Load a TOML file into a Pydantic model.

    Given a path to a TOML file, automatically unpack its fields into the provided
    Pydantic model. Also return the schema directive at the top of the TOML file, if it
    happens to have one.

    Parameters
    ----------
    path: StrPath
        The path to a TOML file.
    model: pydantic.BaseModel
        The Pydantic model to which the contents of the TOML file will be passed.

    Returns
    -------
    pydantic.BaseModel
        An instance of the Pydantic model after validation.
    Optional[str]
        The schema directive in the TOML file, if it had one.

    """
    path = get_file(path)
    if path.suffix != ".toml":
        raise ValueError(f"The path '{path}' does not refer to a TOML file.")

    with open(path) as file:
        if (first_line := file.readline()).startswith("#:"):
            schema_directive = first_line
        else:
            schema_directive = None

    raw_config = toml.load(path)
    return (
        model(**{key: raw_config.get(key) for key in model.__fields__.keys()}),  # type: ignore
        schema_directive,
    )


def write_schema(directory: str, model: type[BaseModel]):
    (Path(directory) / "boilerdata.toml.json").write_text(model.schema_json(indent=2))
