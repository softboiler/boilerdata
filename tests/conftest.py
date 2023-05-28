"""Test configuration."""

from pathlib import Path
from shutil import copy, copytree
from sys import path

import pytest

from tests import NOTEBOOK_STAGES, get_nb_content

TEST_DATA = Path("tests/data")


@pytest.fixture()
def tmp_project(monkeypatch, tmp_path: Path) -> Path:
    """Produce a temporary project directory."""

    monkeypatch.setenv("DYNACONF_APP_FOLDER", f"{TEST_DATA / '.propshop'}")
    import boilerdata

    test_params = tmp_path / "params.yaml"
    monkeypatch.setattr(boilerdata, "PARAMS_FILE", test_params)
    copy("params.yaml", test_params)

    test_config = tmp_path / "config"
    monkeypatch.setattr(boilerdata, "AXES_CONFIG", test_config / "axes.yaml")
    monkeypatch.setattr(boilerdata, "TRIAL_CONFIG", test_config / "trials.yaml")
    copytree(TEST_DATA / "config", test_config)

    test_data = tmp_path / "data"
    monkeypatch.setattr(boilerdata, "DATA_DIR", test_data)
    copytree(TEST_DATA / "data", test_data)

    return tmp_path


@pytest.fixture()
def _tmp_project_with_nb_stages(tmp_project: Path):
    """Enable importing of notebook stages like `importlib.import_module("stage")`."""
    path.insert(0, str(tmp_project))  # For importing tmp_project stages in tests
    for nb in NOTEBOOK_STAGES:
        (tmp_project / nb.with_suffix(".py").name).write_text(
            encoding="utf-8", data=get_nb_content(nb)
        )
