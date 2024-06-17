"""Generate `axes_enum.py`."""

from pathlib import Path
from shutil import copy
from textwrap import dedent

from boilerdata.models.params import PARAMS


def main():  # noqa: D103
    generate_axes_enum([ax.name for ax in PARAMS.axes.all], PARAMS.paths.axes_enum_copy)
    copy(PARAMS.paths.axes_enum_copy, PARAMS.paths.axes_enum)
    PARAMS.paths.file_originlab_coldes.write_text(
        encoding="utf-8", data=PARAMS.axes.get_originlab_coldes()
    )


def generate_axes_enum(axes: list[str], path: Path) -> None:
    """Given a list of axis names, generate a Python script with axes as enums."""
    text = dedent(
        '''\
        """Auto-generated enums for auto-complete."""

        from enum import auto

        from boilerdata.types import GetNameEnum


        class AxesEnum(GetNameEnum):
        '''
    )
    for label in axes:
        text += f"    {label} = auto()\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
