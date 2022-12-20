from pathlib import Path

import pandas as pd

from boilerdata.models.project import Project


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
            # Allow source cols to be missing (such as certain thermocouples)
            usecols=lambda col: col in [index, *source_col_names],
            index_col=index,
            parse_dates=[index],  # type: ignore  # pandas
            dtype=source_dtypes,  # type: ignore  # pandas
            encoding="utf-8",
        )
        # Rarely a run has an all NA record at the end
    ).dropna(how="all")

    # Need "df" defined so we can call "df.index.dropna()". Repeat `dropna` because a
    # run can have an NA index at the end and a CSV can have an all NA record at the end
    return (
        df.reindex(index=df.index.dropna()).dropna(how="all").pipe(rename_columns, proj)
    )


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in proj.axes.cols})
