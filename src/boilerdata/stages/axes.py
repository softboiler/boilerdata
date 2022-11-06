"""Configuration utilities for Pydantic models and their schema."""

from pathlib import Path
from shutil import copy
from textwrap import dedent

from pydantic import MissingError, ValidationError
import yaml

from boilerdata.models.axes import Axes
from boilerdata.models.common import get_file, load_config
from boilerdata.models.dirs import Dirs


def main(dirs: Dirs):
    axes = load_config(dirs.config / "axes.yaml", Axes)
    generate_axes_enum([ax.name for ax in axes.all], dirs.file_axes_enum_copy)
    copy(dirs.file_axes_enum_copy, dirs.file_axes_enum)
    dirs.file_originlab_coldes.write_text(axes.get_originlab_coldes())


def generate_axes_enum(axes: list[str], path: Path) -> None:
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


if __name__ == "__main__":
    # Repeats some logic from load_config() to sidestep cyclic dependencies.
    path = get_file("params.yaml")
    if path.suffix != ".yaml":
        raise ValueError(f"The path '{path}' does not refer to a YAML file.")
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw_config:
        raise ValueError("The configuration file is empty.")
    try:
        dirs = Dirs(**raw_config["dirs"])
    except ValidationError as exception:
        addendum = "\n  The field may be undefined in the configuration file."
        for error in exception.errors():
            if error["msg"] == MissingError.msg_template:
                error["msg"] += addendum
        raise exception
    main(dirs)
