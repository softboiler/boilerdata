# * -------------------------------------------------------------------------------- * #
# * GET RUNS


from datetime import datetime
from pathlib import Path

import pandas as pd

from boilerdata.models.common import set_dtypes
from boilerdata.models.project import Project

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):
    (
        pd.DataFrame(
            columns=[ax.name for ax in proj.axes.cols], data=get_runs(proj)
        ).to_csv(proj.dirs.runs_file, encoding="utf-8")
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def get_runs(proj: Project) -> pd.DataFrame:
    """Get runs from all trials."""

    # Get runs and multiindex
    dtypes = {col.name: col.dtype for col in proj.axes.source if not col.index}
    runs: list[pd.DataFrame] = []
    multiindex: list[tuple[datetime, datetime, datetime]] = []
    for trial in proj.trials:
        for file, run_index in zip(trial.run_files, trial.run_index):
            run = get_run(proj, file)
            runs.append(run)
            multiindex.extend(
                tuple((*run_index, record_time) for record_time in run.index)
            )

    return (
        pd.concat(runs)
        .set_index(
            pd.MultiIndex.from_tuples(
                multiindex, names=[idx.name for idx in proj.axes.index]
            )
        )
        .pipe(set_dtypes, dtypes)
    )


def get_run(proj: Project, run: Path) -> pd.DataFrame:
    """Get data for a single run."""

    # Get source columns
    index = proj.axes.index[-1].source  # Get the last index, associated with source
    source_col_names = [col.source for col in proj.axes.source_cols]
    source_dtypes = {col.source: col.dtype for col in proj.axes.source_cols}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    df = pd.DataFrame(
        columns=source_col_names,
        data=pd.read_csv(
            run,
            # Allow source cols to be missing (such as T_6)
            usecols=lambda col: col in [index, *source_col_names],
            index_col=index,
            parse_dates=[index],  # type: ignore  # Upstream issue w/ pandas-stubs
            dtype=source_dtypes,  # type: ignore  # Upstream issue w/ pandas-stubs
            encoding="utf-8",
        )
        # Rarely a run has an all NA record at the end
    ).dropna(how="all")

    # Need "df" defined so we can call "df.index.dropna()"
    return (
        df.reindex(index=df.index.dropna())  # A run can have an NA index at the end
        .dropna(how="all")  # A CSV can have an all NA record at the end
        .tail(proj.params.records_to_average)
        .pipe(rename_columns, proj)
    )


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in proj.axes.cols})


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
