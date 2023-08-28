"""Test configuration."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from shutil import copy, copytree

import pytest
from boilercore.testing import change_workdir_and_prepend, make_tmp_nbs_content

from tests import NOTEBOOK_STAGES


@pytest.fixture(autouse=True)
def _tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Produce a temporary project directory."""
    monkeypatch.setenv("DYNACONF_APP_FOLDER", f"{tmp_path / '.propshop'}")
    orig_workdir = change_workdir_and_prepend(tmp_path, monkeypatch)
    copy(orig_workdir / "params.yaml", tmp_path / "params.yaml")
    copytree(orig_workdir / "tests" / "data", tmp_path, dirs_exist_ok=True)
    make_tmp_nbs_content(NOTEBOOK_STAGES, tmp_path, orig_workdir)
