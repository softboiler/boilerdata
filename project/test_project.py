from math import inf
from os import getenv
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m, raises
import yaml

from migrate import migrate_1, migrate_2, migrate_3
from pipeline import pipeline
from utils import get_project

CI = "Skip on CI."


# * -------------------------------------------------------------------------------- * #
# * PIPELINE


@m.skipif(bool(getenv("CI")), reason=CI)
def test_get_steady_state_raises(tmp_proj):
    tmp_proj.params.records_to_average = inf
    with raises(ValueError, match="not enough records"):
        pipeline(tmp_proj)


@m.skipif(bool(getenv("CI")), reason=CI)
def test_pipeline(tmp_proj):
    """Ensure the same result is coming out of the pipeline as before."""

    old_commit = "c1a6cbf1d62ce59c44b8ed4dc1e526d21e84f403"

    common_read_csv_params = dict(
        skiprows=[1],  # Skip the "units" row so dtype detection works properly
        encoding="utf-8",
    )

    pipeline(tmp_proj)

    old = pd.read_csv(get_old_file(old_commit), **common_read_csv_params)
    new = pd.read_csv(tmp_proj.dirs.results_file, **common_read_csv_params)
    assert_frame_equal(old, new)


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
