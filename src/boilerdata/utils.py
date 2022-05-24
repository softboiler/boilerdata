"""Configuration utilities for loading and dumping Pydantic models and their schema."""

from os import PathLike
from pathlib import Path

from pydantic import BaseModel, MissingError, ValidationError
import yaml

StrPath = str | PathLike[str]


def expanduser2(path: StrPath) -> Path:
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
    if isinstance(path, str) and path.startswith(home):
        return Path.home() / path.lstrip(home)
    else:
        return Path(path)


def get_file(path: StrPath, create: bool = False) -> Path:
    """Generate `pathlib.Path` to a file that exists.

    Handle the "~" user construction if necessary and return a `pathlib.Path` object.
    Raise exception if the file is not found.

    Parameters
    ----------
    path: StrPath
        The path.
    create: bool
        Whether a file should be created at the path if it doesn't already exist.

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
        if create:
            path.touch()
        else:
            raise FileNotFoundError(f"The path '{path}' does not exist.")
    elif not path.is_file():
        raise FileNotFoundError(f"The path '{path}' does not refer to a file.")
    return path


def load_config(path: StrPath, model):
    """Load a YAML file into a Pydantic model.

    Given a path to a YAML file, automatically unpack its fields into the provided
    Pydantic model. Also return the schema directive at the top of the YAML file, if it
    happens to have one.

    Parameters
    ----------
    path: StrPath
        The path to a YAML file.
    model: type[pydantic.BaseModel]
        The Pydantic model class to which the contents of the YAML file will be passed.

    Returns
    -------
    pydantic.BaseModel
        An instance of the Pydantic model after validation.

    Raises
    ------
    ValueError
        If the path does not refer to a YAML file, or the YAML file is empty.
    ValidationError
        If the configuration file is missing a required field.
    """
    path = get_file(path)
    if path.suffix != ".yaml":
        raise ValueError(f"The path '{path}' does not refer to a YAML file.")

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw_config:
        raise ValueError("The configuration file is empty.")

    try:
        config = model(**{key: raw_config.get(key) for key in raw_config.keys()})
    except ValidationError as exception:
        addendum = "\n  The field may be undefined in the configuration file."
        for error in exception.errors():
            if error["msg"] == MissingError.msg_template:
                error["msg"] += addendum
        raise exception
    return config


# Can't type annotate `model` for some reason
def dump_model(path: StrPath, model):
    """Dump a Pydantic model to a YAML file.

    Given a path to a YAML file, write a Pydantic model to the file. Optionally add a
    schema directive at the top of the file. Create the file if it doesn't exist.

    Parameters
    ----------
    path: StrPath
        The path to a YAML file. Will create it if it doesn't exist.
    model: type[pydantic.BaseModel]
        An instance of the Pydantic model to dump.
    """
    path = get_file(path, create=True)
    # ensure one \n and no leading \n, Pydantic sometimes does more
    path.write_text(
        yaml.safe_dump(model.dict(exclude_none=True), sort_keys=False), encoding="utf-8"
    )


def write_schema(path: StrPath, model: type[BaseModel]):
    """Write a Pydantic model schema to a JSON file.

    Given a path to a JSON file, write a Pydantic model schema to the file. Create the
    file if it doesn't exist.

    Parameters
    ----------
    path: StrPath
        The path to a JSON file. Will create it if it doesn't exist.
    model: type[pydantic.BaseModel]
        The Pydantic model class to get the schema from.
    """
    path = get_file(path, create=True)
    if path.suffix != ".json":
        raise ValueError(f"The path '{path}' does not refer to a JSON file.")
    path.write_text(model.schema_json(indent=2) + "\n", encoding="utf-8")
