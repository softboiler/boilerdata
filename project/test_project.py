from pathlib import Path
from shutil import copy

import pandas as pd
from pandas.testing import assert_frame_equal
import yaml

from migrate import migrate_1
from models import Project
from pipeline import PROJECT, main


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text())
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text())
    assert result == expected


def test_run(tmp_path):
    """Ensure the same result is coming out of the pipeline as before."""

    old_commit = "ddb93d9463f116187cf8b57d914c2afef48c7313"
    try:
        old_file = next((PROJECT.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception

    for csv in PROJECT.trials.glob(f"**/{PROJECT.directory_per_trial}/**/*.csv"):
        dst = tmp_path / csv.relative_to(PROJECT.base)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy(csv, dst)

    new_project = Project(
        base=tmp_path,
        trials=tmp_path / (PROJECT.trials.relative_to(PROJECT.base)),
        results=tmp_path / (PROJECT.results.relative_to(PROJECT.base)),
        directory_per_trial=PROJECT.directory_per_trial,
        fit=PROJECT.fit,
    )
    main(new_project)

    old = pd.read_csv(old_file)
    new = pd.read_csv(new_project.results_file, usecols=old.columns)
    assert_frame_equal(old, new)
