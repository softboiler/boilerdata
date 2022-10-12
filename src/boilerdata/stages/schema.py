"""Configuration utilities for Pydantic models and their schema."""

from pathlib import Path
import re
from shutil import copytree, rmtree
from textwrap import dedent

from boilerdata.models.axes import Axes
from boilerdata.models.common import write_schema
from boilerdata.models.project import Project
from boilerdata.models.trials import Trials


def main():

    models = [Project, Trials, Axes]
    proj = Project.get_project()

    for model in models:
        write_schema(
            proj.dirs.project_schema / f"{to_snake_case(model.__name__)}_schema.json",
            model,
        )
    generate_axes_enum(
        [ax.name for ax in proj.axes.all], Path("src/boilerdata/axes_enum.py")
    )
    proj.dirs.originlab_coldes_file.write_text(proj.axes.get_originlab_coldes())

    rmtree(proj.dirs.project_schema_old)
    copytree(proj.dirs.project_schema, proj.dirs.project_schema_old)


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
    main()
