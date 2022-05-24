from os import getenv
from pathlib import Path
from shutil import copy

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark as m
import yaml

from migrate import migrate_1, migrate_2, migrate_3
from models import Coupon, Group, Joint, Project, Rod, Sample
from pipeline import get_defaults, main as pipeline_main

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

    pd.options.mode.string_storage = "pyarrow"
    old_commit = "02731650edf4ea0f4e08f1148f2bed57ed2515ca"
    common_records_count = 322
    dtypes = {
        "Run": pd.StringDtype(),
        "V": float,
        "T0cal": float,
        "T1cal": float,
        "T2cal": float,
        "T3cal": float,
        "T4cal": float,
        "T5cal": float,
        "dT_dx": float,
        "TLfit": float,
        "rvalue": float,
        "pvalue": float,
        "stderr": float,
        "intercept_stderr": float,
        "dT_dx_err": float,
        "k": float,
        # "q": float,  #! These units are different.
        # "q_err": float,
        # "Q": float,
        # "∆T": float,  #! These names are different.
        # "∆T_err": float,
        "date": pd.StringDtype(),  # Will become datetime64[ns] after parsing
        "rod": pd.CategoricalDtype([cat.name for cat in Rod]),
        "coupon": pd.CategoricalDtype([cat.name for cat in Coupon]),
        "sample": pd.CategoricalDtype([cat.name for cat in Sample]),
        "group": pd.CategoricalDtype([cat.name for cat in Group]),
        "joint": pd.CategoricalDtype([cat.name for cat in Joint]),
        "monotonic": bool,
        "comment": pd.StringDtype(),
    }

    read_csv_params = dict(
        index_col="Run",
        skiprows=[1],
        nrows=common_records_count,
        usecols=dtypes.keys(),
        dtype=dtypes,
        parse_dates=["date"],
    )

    col_order = list(dtypes.keys())[1:]
    project, trials = get_defaults()
    old = pd.read_csv(get_old_file(old_commit), **read_csv_params)[col_order]

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
    pipeline_main(new_project, trials)
    new = pd.read_csv(new_project.results_file, **read_csv_params)[col_order]

    assert_frame_equal(old, new)


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_old_file(old_commit):
    project, _ = get_defaults()
    try:
        return next((project.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception
