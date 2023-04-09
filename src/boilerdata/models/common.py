from contextlib import contextmanager
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Extra, MissingError, ValidationError

from boilerdata.common import StrPath, get_file

NpNDArray = np.ndarray[Any, Any]
NpFloating = np.floating[Any]


class MyBaseModel(BaseModel):
    """Base model for all Pydantic models used in this project."""

    class Config:
        """Model configuration.

        Accessing enums yields their values, and allowing arbitrary types enables
        using Numpy types in fields.
        """

        # Don't specify as class kwargs for easier overriding, and "extra" acted weird.
        use_enum_values = True  # To use enums in schema, but not in code
        arbitrary_types_allowed = True  # To use Numpy types
        extra = Extra.forbid  # To forbid extra fields


@contextmanager
def allow_extra(model: BaseModel):
    """Temporarily allow extra properties to be set on a Pydantic model.

    This is useful when writing a custom `__init__`, where not explicitly allowing extra
    properties will result in errors, but you don't want to allow extra properties
    forevermore.

    Args:
        model: The model to allow extras on.
    """

    # Store the current value of the attribute or note its absence
    try:
        original_config = model.Config.extra
    except AttributeError:
        original_config = None
    model.Config.extra = Extra.allow

    # Yield the temporarily changed config, resetting or deleting it when done
    try:
        yield
    finally:
        if original_config:
            model.Config.extra = original_config
        else:
            del model.Config.extra


# * -------------------------------------------------------------------------------- * #
# * COMMON


def load_config(path: StrPath, model):
    """Load a YAML file into a Pydantic model.

    Given a path to a YAML file, automatically unpack its fields into the provided
    Pydantic model.

    Args:
        path: The path to a YAML file.
        model: The Pydantic model class to which the YAML file contents will be passed.

    Returns:
        pydantic.BaseModel: An instance of the Pydantic model after validation.

    Raises:
        ValueError: The path does not refer to a valid YAML file.
    ValidationError: If the configuration file is missing a required field.
    """
    path = get_file(path)
    if path.suffix != ".yaml":
        raise ValueError(f"The path '{path}' does not refer to a YAML file.")

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw_config:
        raise ValueError("The configuration file is empty.")

    try:
        config = model(**{key: raw_config.get(key) for key in raw_config})
    except ValidationError as exception:
        addendum = "\n  The field may be undefined in the configuration file."
        for error in exception.errors():
            if error["msg"] == MissingError.msg_template:
                error["msg"] += addendum
        raise
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
