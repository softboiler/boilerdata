# # Necessary as long as a line marked "triggered only locally" is in this file
# pyright: reportUnnecessaryTypeIgnoreComment=none

from os import chdir
from pathlib import Path

import pandas as pd

from boilerdata.models.project import Project

FLOAT_SPEC = "#.4g"
pd.options.display.min_rows = pd.options.display.max_rows = 50
pd.options.display.float_format = f"{{:{FLOAT_SPEC}}}".format  # type: ignore  # pandas

# * -------------------------------------------------------------------------------- * #
# * NOTEBOOK SETUP


def chdir_to_nearest_git_root(max_depth: int = 7) -> None:
    """Change the working directory to the nearest git root."""
    original_cwd = Path.cwd()
    if (original_cwd / Path(".git")).exists():
        return
    eventual_cwd = original_cwd.parent
    current_drive_root = Path(original_cwd.anchor)
    for _ in range(max_depth + 1):
        if eventual_cwd == current_drive_root:
            raise RuntimeError(
                "Couldn't find git project folder above drive root.\n"
                f"Original CWD: {original_cwd}\n"
                f"Stopped at : {eventual_cwd}\n"
            )
        if (eventual_cwd / Path(".git")).exists():
            chdir(eventual_cwd)
            return
        eventual_cwd = eventual_cwd.parent
    raise RuntimeError(
        f"Couldn't find git project folder above max depth of {max_depth}.\n"
        f"Original CWD: {original_cwd}\n"
        f"Stopped at : {eventual_cwd}\n"
    )


def chdir_to_nearest_git_root_and_get_project():
    """Ensure this notebook runs at project root regardless of how it is executed."""
    chdir_to_nearest_git_root()
    return Project.get_project()


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def add_units(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Make the columns a multi-index representing units."""
    cols = proj.axes.get_col_index()
    quantity = cols.get_level_values("quantity")
    units = cols.get_level_values("units")

    old = (col.name for col in proj.axes.cols)
    new = (add_unit(q, u) for q, u in zip(quantity, units))
    return df.rename(axis="columns", mapper=dict(zip(old, new)))


def add_unit(quantity: str, units: str) -> str:
    return f"{quantity} ({units})" if units else quantity


def tex_wrap(df: pd.DataFrame) -> pd.DataFrame:
    """Wrap all column titles in LaTeX flags ($)."""
    mapper: dict[str, str] = {}
    for src_col in df.columns:
        col = f"${handle_subscript(src_col)}$"
        mapper[src_col] = col
    return df.rename(axis="columns", mapper=mapper)


def handle_subscript(val: str) -> str:
    """Wrap everything after the first underscore and replace others with commas."""
    if "_" not in val:
        return val
    parts = val.split("_")
    return f"{parts[0]}_" + "{" + ",".join(parts[1:]) + "}"
