from math import inf
from os import getenv
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m, raises
import yaml

from config.columns import Columns as C  # noqa: N817
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
def test_run(tmp_proj):
    """Ensure the same result is coming out of the pipeline as before."""

    old_commit = "ccd7affba00d0cf6b4b80b94a638dcfa3d3404f5"

    rest_of_read_csv_params = dict(
        usecols=[col.pretty_name for col in tmp_proj.cols.values()],
        index_col=tmp_proj.get_index().name,
        parse_dates=[C.date],  # Can't handle date column in dtypes parameter below
        dtype={
            name: col.dtype for name, col in tmp_proj.cols.items() if name != C.date
        },
        skiprows=[1],  # Skip the "units" row as it won't be coerced to dtype nicely
    )

    pipeline(tmp_proj)

    old = pd.read_csv(get_old_file(old_commit), **rest_of_read_csv_params)
    new = pd.read_csv(tmp_proj.dirs.results_file, **rest_of_read_csv_params)
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
