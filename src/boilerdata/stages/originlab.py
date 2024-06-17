"""Produce OriginLab plots."""

import re
from contextlib import contextmanager
from pathlib import Path
from time import sleep

import originpro as op  # type: ignore  # Not installed in CI
import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS, Params


def main():  # noqa: D103
    (
        pd.read_csv(
            PARAMS.paths.file_results,
            index_col=(index := [A.trial, A.run]),
            parse_dates=index,
            dtype={col.name: col.dtype for col in PARAMS.axes.cols},
        )
        .pipe(transform_for_originlab, PARAMS)
        .to_csv(PARAMS.paths.file_originlab_results, index=False, encoding="utf-8")
    )

    with open_originlab(PARAMS.paths.file_plotter):
        for shortname, file in PARAMS.paths.originlab_plot_files.items():
            gp = op.find_graph(shortname)
            fig = gp.save_fig(get_path(file), type="png")
            if not fig:
                raise RuntimeError("Failed to save figure.")


def transform_for_originlab(df: pd.DataFrame, params: Params) -> pd.DataFrame:
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

    cols = params.axes.get_col_index()
    quantity = cols.get_level_values("quantity").map({
        col.name: col.pretty_name for col in params.axes.all
    })
    units = cols.get_level_values("units")
    indices = [
        index.to_series()
        .reset_index(drop=True)
        .replace(superscript, superscript_repl)  # type: ignore  # pandas
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
    main()
