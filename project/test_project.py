from os import getenv
from pathlib import Path
from shutil import copy

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m
import yaml

from migrate import migrate_1
from models import Project
from pipeline import get_defaults, main


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text())
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text())
    assert result == expected


def test_migrate_2():
    ...


@m.skipif(bool(getenv("CI")), reason="Skip on CI.")
def test_run(tmp_path):
    """Ensure the same result is coming out of the pipeline as before."""

    project, trials = get_defaults()
    old_commit = "ddb93d9463f116187cf8b57d914c2afef48c7313"
    try:
        old_file = next((project.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception

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

    old = pd.read_csv(old_file)
    new = pd.read_csv(new_project.results_file, usecols=old.columns)
    assert_frame_equal(old, new)
