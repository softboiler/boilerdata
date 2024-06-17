"""Parse benchmark runs."""

from pathlib import Path

import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS, Params
from boilerdata.stages import get_run


def main():  # noqa: D103
    pd.DataFrame(data=get_benchmarks(PARAMS)).to_csv(
        PARAMS.paths.file_benchmarks_parsed, encoding="utf-8"
    )


def get_benchmarks(params: Params) -> pd.DataFrame:
    """Get runs from all trials."""
    benchmarks = [
        get_run(params, benchmark).pipe(parse_benchmark, params)
        for benchmark in Path(params.paths.benchmarks).glob("*.csv")
    ]
    return pd.concat(benchmarks)


def parse_benchmark(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Get all temperatures when base temperature has risen 90% of its total change."""
    threshold = 0.9
    df = df[[A.T_0, *params.copper_temps]]
    base = df[A.T_0]
    start = base.head(10).mean()
    end = base.tail(10).mean()
    base_normalized = (base - start) / (end - start)
    time_of_90_rise = (base_normalized > threshold).idxmax()
    return df.loc[[time_of_90_rise], :]


if __name__ == "__main__":
    main()
