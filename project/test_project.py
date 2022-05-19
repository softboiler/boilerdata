from pathlib import Path
from shutil import copy

import pandas as pd
from pandas.testing import assert_frame_equal
import yaml

from migrate import migrate_1
from models import Project
from pipeline import PROJECT, run


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text())
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text())
    assert result == expected


def test_run(tmp_path):
    """Ensure the same result is coming out of the pipeline as before."""
    for csv in PROJECT.trials.glob(f"**/{PROJECT.directory_per_trial}/**/*.csv"):
        dst = tmp_path / csv.relative_to(PROJECT.base)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy(csv, dst)
    project = Project(
        base=tmp_path,
        trials=tmp_path / (PROJECT.trials.relative_to(PROJECT.base)),
        directory_per_trial=PROJECT.directory_per_trial,
        fit=PROJECT.fit,
    )
    run(project)
    assert_frame_equal(
        pd.read_csv(project.base / "results.csv"),
        pd.read_csv(PROJECT.base / "results.csv"),
    )
