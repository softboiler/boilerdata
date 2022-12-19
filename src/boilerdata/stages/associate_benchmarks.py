import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project


def main(proj: Project):

    (
        pd.read_csv(
            proj.dirs.file_runs,
            index_col=(index_col := [A.trial, A.run, A.time]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in proj.axes.cols},
        ).to_csv(proj.dirs.file_runs_with_benchmarks, encoding="utf-8")
    )


if __name__ == "__main__":
    main(Project.get_project())
