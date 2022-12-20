from pathlib import Path

import numpy as np
import pandas as pd

from boilerdata.models.project import Project
from boilerdata.stages.common import set_dtypes
from boilerdata.stages.prep.common import get_run


def main(proj: Project):
    pd.DataFrame(data=get_benchmarks(proj)).to_csv(
        proj.dirs.file_benchmarks_parsed, encoding="utf-8"
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def get_benchmarks(proj: Project) -> pd.DataFrame:
    """Get runs from all trials."""
    dtypes = {col.name: col.dtype for col in proj.axes.source if not col.index}
    benchmarks = [
        get_run(proj, benchmark).pipe(get_statistics, proj)
        for benchmark in Path(proj.dirs.benchmarks).glob("*.csv")
    ]
    return pd.concat(benchmarks).pipe(set_dtypes, dtypes)


def get_statistics(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    start = df.index[0]
    df.index = (df.index - df.index[0]).total_seconds()
    df = (df - df.min()) / (df.max() - df.min())
    rise_time = df[df > 0.9].idxmax()
    rise_time.name = start
    return pd.DataFrame(rise_time).T


def exponential(x, tau):
    return np.exp(x / tau)


# * -------------------------------------------------------------------------------- * #


if __name__ == "__main__":
    main(Project.get_project())
