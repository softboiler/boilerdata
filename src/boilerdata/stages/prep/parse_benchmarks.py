# pyright: reportConstantRedefinition=false

from pathlib import Path

import numpy as np
import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project
from boilerdata.stages.prep.common import get_run


def main(proj: Project):
    pd.DataFrame(data=get_benchmarks(proj)).to_csv(
        proj.dirs.file_benchmarks_parsed, encoding="utf-8"
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def get_benchmarks(proj: Project) -> pd.DataFrame:
    """Get runs from all trials."""
    benchmarks = [
        get_run(proj, benchmark).pipe(get_statistics, proj)
        for benchmark in Path(proj.dirs.benchmarks).glob("*.csv")
    ]
    return pd.concat(benchmarks)


def get_statistics(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    df = df[[A.T_0, *proj.params.copper_temps]]  # type: ignore  # use_enum_values
    start = df.index[0]
    df.index = (df.index - start).total_seconds()  # type: ignore  # pandas
    normalized_temps = (df - df.min()) / (df.max() - df.min())
    rise_times = normalized_temps[normalized_temps > 0.9].idxmax().rename("rise_time")
    temps_at_rise_time = df.loc[rise_times[A.T_0], :].rename("temps_at_rise_time")  # type: ignore  # pandas
    return pd.concat([rise_times, temps_at_rise_time], axis="columns").set_axis(
        axis="index",
        labels=pd.MultiIndex.from_product(
            [[start], rise_times.index], names=["time", "temperature"]
        ),
    )


def exponential(x, tau):
    return np.exp(x / tau)


# * -------------------------------------------------------------------------------- * #


if __name__ == "__main__":
    main(Project.get_project())
