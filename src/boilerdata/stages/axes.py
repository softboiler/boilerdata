"""Generate `axes_enum.py`."""

from pathlib import Path
from shutil import copy
from textwrap import dedent

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
    # Repeats boilerdata.models.common.load_config() logic to avoid cyclic dependencies.
    path = get_file("params.yaml")
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    dirs = Dirs(**raw_config["dirs"])
    main(dirs)
