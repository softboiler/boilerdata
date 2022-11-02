from contextlib import contextmanager
from pathlib import Path
import re
from time import sleep

import originpro as op
import pandas as pd

from boilerdata.models.project import Project
from boilerdata.stages.common import get_results


def main(proj: Project):

    (
        get_results(proj)
        .pipe(transform_for_originlab, proj)
        .to_csv("data/results/originlab_results.csv", index=False, encoding="utf-8")
    )

    with open_originlab(proj.dirs.file_plotter):
        gp = op.find_graph(proj.params.plots[0])
        fig = gp.save_fig(get_path(proj.dirs.plots, mkdirs=True), type="png", width=800)
        if not fig:
            raise RuntimeError("Failed to save figure.")


def transform_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels.

    Explicitly set all dtypes to string to avoid data rendering issues, especially with
    dates. Convert super/subscripts in units to their OriginLab representation. Reset
    the index to avoid the extra row between units and data indicating index axis names.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """

    superscript = re.compile(r"\^(.*)")
    superscript_repl = r"\+(\1)"
    subscript = re.compile(r"\_(.*)")
    subscript_repl = r"\-(\1)"

    cols = proj.axes.get_col_index()
    quantity = cols.get_level_values("quantity").map(
        {col.name: col.pretty_name for col in proj.axes.all}
    )
    units = cols.get_level_values("units")
    indices = [
        index.to_series()
        .reset_index(drop=True)
        .replace(
            superscript,  # pyright: ignore [reportGeneralTypeIssues]  # pandas
            superscript_repl,
        )
        .replace(subscript, subscript_repl)
        for index in (quantity, units)
    ]
    cols = pd.MultiIndex.from_frame(pd.concat(axis="columns", objs=indices))
    return df.set_axis(axis="columns", labels=cols).reset_index()


@contextmanager
def open_originlab(file, readonly=True):
    """Open an OriginLab file."""
    if not Path(file).exists():
        raise FileNotFoundError(f"File not found: {file}")
    op.set_show(True)  # required
    file = op.open(file=get_path(file), readonly=readonly)
    sleep(5)  # wait for data sources to update upon book opening
    yield file
    op.exit()


def get_path(file, mkdirs=False):
    """Return the absolute path of a file for OriginLab interoperation."""
    path = Path(file)
    if mkdirs:
        path.mkdir(parents=True, exist_ok=True)
    return str(Path(file).resolve())


if __name__ == "__main__":
    main(Project.get_project())
