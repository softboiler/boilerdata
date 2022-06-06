"""Configuration utilities for Pydantic models and their schema."""

from pathlib import Path
import re
from textwrap import dedent

from pydantic import ValidationError

from boilerdata.models.axes import Axes
from boilerdata.models.common import write_schema
from boilerdata.models.project import Project, Trials

models = [Project, Trials, Axes]


def update_schema():

    try:
        proj = Project.get_project()
        path = proj.dirs.project_schema
        generate_axes_enum(
            list(Axes.get_names(proj.axes.all)),
            Path("src/boilerdata/models/axes_enum.py"),
        )
    except ValidationError as exception:
        path = Path("src/boilerdata/schema")
        print(
            f"Schema didn't validate, using default schema path: {path}.",
            "Axes enum not generated.",
            "Message from caught ValidationError:",
            exception,
            sep="\n",
        )

    for model in models:

        write_schema(path / f"{to_snake_case(model.__name__)}_schema.json", model)


def generate_axes_enum(axes: list[str], path: Path):
    """Given a list of axis names, generate a Python script with axes as enums."""
    text = dedent(
        """\
        # flake8: noqa

        from enum import auto

        from boilerdata.models.enums import GetNameEnum


        class AxesEnum(GetNameEnum):
        """
    )
    for label in axes:
        text += f"    {label} = auto()\n"
    path.write_text(text, encoding="utf-8")


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


if __name__ == "__main__":
    update_schema()
