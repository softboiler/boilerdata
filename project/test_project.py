from os import getenv
from pathlib import Path
from shutil import copy

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m
import yaml

from migrate import migrate_1, migrate_2, migrate_3
from models import Project
from pipeline import get_defaults, main

CI = "Skip on CI."


# * -------------------------------------------------------------------------------- * #
# * MIGRATIONS


@m.skipif(bool(getenv("CI")), reason=CI)
def test_migrate_3(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project, _ = get_defaults()
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


@m.skipif(bool(getenv("CI")), reason=CI)
def test_migrate_2(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project, _ = get_defaults()
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


# * -------------------------------------------------------------------------------- * #
# * PIPELINE


@m.skip("slow")
@m.skipif(bool(getenv("CI")), reason=CI)
def test_run(tmp_path):
    """Ensure the same result is coming out of the pipeline as before."""

    old_commit = "ddb93d9463f116187cf8b57d914c2afef48c7313"
    project, trials = get_defaults()

    for csv in project.trials.glob(f"**/{project.directory_per_trial}/**/*.csv"):
        dst = tmp_path / csv.relative_to(project.base)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy(csv, dst)

    new_project = Project(
        base=tmp_path,
        trials=tmp_path / (project.trials.relative_to(project.base)),
        results=tmp_path / (project.results.relative_to(project.base)),
        directory_per_trial=project.directory_per_trial,
        fit=project.fit,
    )
    main(new_project, trials)

    old = get_old_data(old_commit)
    new = pd.read_csv(new_project.results_file, usecols=old.columns)
    assert_frame_equal(old, new)


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_old_data(old_commit):
    return pd.read_csv(get_old_file(old_commit))


def get_old_file(old_commit):
    project, _ = get_defaults()
    try:
        return next((project.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception
