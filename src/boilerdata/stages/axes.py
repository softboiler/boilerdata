"""Generate `axes_enum.py`."""

from pathlib import Path
from shutil import copy
from textwrap import dedent

from boilerdata import AXES_ENUM_FILE
from boilerdata.models.params import PARAMS


def main():
    axes_enum_copy = PARAMS.paths.axes / AXES_ENUM_FILE.name
    generate_axes_enum([ax.name for ax in PARAMS.axes.all], axes_enum_copy)
    copy(axes_enum_copy, AXES_ENUM_FILE)
    PARAMS.paths.file_originlab_coldes.write_text(PARAMS.axes.get_originlab_coldes())


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
    main()
