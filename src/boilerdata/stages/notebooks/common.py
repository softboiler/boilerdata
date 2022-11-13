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
