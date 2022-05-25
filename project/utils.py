"""Update all schema."""

import re

from pydantic import Extra

from boilerdata.enums import generate_columns_enum
from boilerdata.utils import load_config, write_schema
from models import Columns, Project, Trials

models = [Project, Trials, Columns]


def update_schema():

    project = get_project()

    for model in models:

        # Forbid extra properties when writing the schema.
        model.Config.extra = Extra.forbid

        write_schema(
            project.dirs.project_schema
            / f"{to_snake_case(model.__name__)}_schema.json",
            model,
        )

    generate_columns_enum(
        list(project.columns.keys()), project.dirs.config / "columns.py"
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