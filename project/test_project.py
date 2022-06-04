from os import getenv
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m
import yaml

from migrate import migrate_1, migrate_2, migrate_3
from pipeline import get_df, pipeline
from utils import get_project

CI = "Skip on CI."


# * -------------------------------------------------------------------------------- * #
# * PIPELINE


@m.skipif(bool(getenv("CI")), reason=CI)
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


@m.skipif(bool(getenv("CI")), reason=CI)
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
    project = get_project()
    try:
        return next((project.dirs.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception


# * -------------------------------------------------------------------------------- * #
# * MIGRATIONS


@m.skip("outdated")
@m.skipif(bool(getenv("CI")), reason=CI)
def test_migrate_3(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project = get_project()
    project.results_file = get_old_file(old_commit)
    migrate_3(
        project,
        schema_path := tmp_path / "columns_schema.json",
        columns_path := tmp_path / "columns.yaml",
    )

    result = schema_path.read_text(encoding="utf-8")
    expected = Path("project/tests/migrate/migrate_3/columns_schema.json").read_text(
        encoding="utf-8"
    )
    assert result == expected

    result = yaml.safe_load(columns_path.read_text(encoding="utf-8"))
    expected = yaml.safe_load(
        Path("project/tests/migrate/migrate_3/columns.yaml").read_text(encoding="utf-8")
    )
    assert result == expected


@m.skip("outdated")
@m.skipif(bool(getenv("CI")), reason=CI)
def test_migrate_2(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project = get_project()
    project.results_file = get_old_file(old_commit)
    migrate_2(project, path := tmp_path / "columns.py")
    result = path.read_text(encoding="utf-8")
    expected = Path("project/tests/migrate/migrate_2/columns.py").read_text(
        encoding="utf-8"
    )
    assert result == expected


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text(encoding="utf-8"))
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text(encoding="utf-8"))
    assert result == expected
