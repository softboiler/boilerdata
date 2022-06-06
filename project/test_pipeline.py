import pandas as pd
from pandas.testing import assert_frame_equal

from boilerdata.models.project import Project
from boilerdata.pipeline import get_df, pipeline

CI = "Skip on CI."


def test_pipeline(tmp_proj):
    """Ensure the same result is coming out of the pipeline as before."""

    old_commit = "3c9fa84fe2c5c71ed58b9f9b53a23194eab85976"

    common_read_csv_params = dict(
        skiprows=[1],  # Skip the "units" row so dtype detection works properly
        encoding="utf-8",
    )

    tmp_proj.params.refetch_runs = True
    pipeline(tmp_proj)

    old = pd.read_csv(get_old_file(old_commit), **common_read_csv_params).reset_index(
        drop=True
    )
    new = pd.read_csv(tmp_proj.dirs.results_file, **common_read_csv_params).reset_index(
        drop=True
    )
    assert_frame_equal(old, new)


def test_get_df(tmp_proj):
    """Ensure the same dataframe comes from disk and from fetching runs."""

    tmp_proj.params.refetch_runs = True
    fetched = get_df(tmp_proj)

    tmp_proj.params.refetch_runs = False
    loaded = get_df(tmp_proj)

    assert_frame_equal(fetched, loaded)


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_old_file(old_commit):
    project = Project.get_project()
    try:
        return next((project.dirs.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception
