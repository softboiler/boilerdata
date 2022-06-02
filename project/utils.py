"""Update all schema."""

from pathlib import Path
import re

from pydantic import ValidationError

from boilerdata.enums import generate_axes_enum
from boilerdata.utils import load_config, write_schema
from models import Axes, Project, Trials, get_names

models = [Project, Trials, Axes]


def update_schema():

    try:
        proj = get_project()
        path = proj.dirs.project_schema
        generate_axes_enum(list(get_names(proj.axes.all)), Path("project/axes.py"))
    except ValidationError as exception:
        path = Path("project/schema")
        print(
            f"Schema didn't validate, using default schema path: {path}.",
            "Axes enum not generated.",
            "Message from caught ValidationError:",
            exception,
            sep="\n",
        )

    for model in models:

        write_schema(
            path / f"{to_snake_case(model.__name__)}_schema.json",
            model,
        )


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


def get_project():
    return load_config("project/config/project.yaml", Project)


if __name__ == "__main__":
    update_schema()
