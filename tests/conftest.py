"""Test configuration."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from shutil import copy, copytree

import pytest
from boilercore.testing import make_tmp_project_with_nb_stages

from tests import NOTEBOOK_STAGES

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


_tmp_project_with_nb_stages = pytest.fixture(
    make_tmp_project_with_nb_stages(NOTEBOOK_STAGES)
)
