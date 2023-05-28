"""Generate schema for the project."""

import re

from pydantic import BaseModel

from boilerdata.common import StrPath, get_file
from boilerdata.models.axes import Axes
from boilerdata.models.params import PARAMS, Params
from boilerdata.models.trials import Trials


def main():
    models = [Params, Trials, Axes]
    for model in models:
        write_schema(
            PARAMS.paths.project_schema
            / f"{to_snake_case(model.__name__)}_schema.json",
            model,
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


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


if __name__ == "__main__":
    main()
