from pathlib import Path

import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import PROJ, Project
from boilerdata.stages.prep.common import get_run


def main():
    pd.DataFrame(data=get_benchmarks(PROJ)).to_csv(
        PROJ.dirs.file_benchmarks_parsed, encoding="utf-8"
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def get_benchmarks(proj: Project) -> pd.DataFrame:
    """Get runs from all trials."""
    benchmarks = [
        get_run(proj, benchmark).pipe(parse_benchmark, proj)
        for benchmark in Path(proj.dirs.benchmarks).glob("*.csv")
    ]
    return pd.concat(benchmarks)


def parse_benchmark(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get all temperatures when base temperature has risen 90% of its total change."""
    threshold = 0.9
    df = df[[A.T_0, *proj.params.copper_temps]]  # type: ignore  # use_enum_values
    base = df[A.T_0]
    start = base.head(10).mean()
    end = base.tail(10).mean()
    base_normalized = (base - start) / (end - start)
    time_of_90_rise = (base_normalized > threshold).idxmax()
    return df.loc[[time_of_90_rise], :]  # type: ignore  # pyright 1.1.308


# * -------------------------------------------------------------------------------- * #


if __name__ == "__main__":
    main()
