"""Get runs from all trials."""

from datetime import datetime

import pandas as pd

from boilerdata.models.params import PARAMS, Params
from boilerdata.stages import get_run, set_dtypes


def main():  # noqa: D103
    (
        pd.DataFrame(
            columns=[ax.name for ax in PARAMS.axes.cols], data=get_runs(PARAMS)
        ).to_csv(PARAMS.paths.file_runs, encoding="utf-8")
    )


def get_runs(params: Params) -> pd.DataFrame:
    """Get runs from all trials."""
    # Get runs and multiindex
    dtypes = {col.name: col.dtype for col in params.axes.source if not col.index}
    runs: list[pd.DataFrame] = []
    multiindex: list[tuple[datetime, datetime, datetime]] = []
    for trial in params.trials:
        for file, run_index in zip(trial.run_files, trial.run_index, strict=True):
            run = get_run(params, file).tail(params.records_to_average)
            runs.append(run)
            multiindex.extend(
                tuple((*run_index, record_time) for record_time in run.index)
            )

    return (
        pd.concat(runs)
        .set_index(
            pd.MultiIndex.from_tuples(
                multiindex, names=[idx.name for idx in params.axes.index]
            )
        )
        .pipe(set_dtypes, dtypes)
    )


if __name__ == "__main__":
    main()
