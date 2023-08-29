"""Test configuration."""

from pathlib import Path

import pytest
from boilercore.testing import get_tmp_project, make_tmp_nbs_content

from tests import NOTEBOOK_STAGES


@pytest.fixture()
def _tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Produce a temporary project directory."""
    with get_tmp_project(tmp_path, monkeypatch):
        ...


@pytest.fixture()
def _tmp_nbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Produce a temporary project directory."""
    with get_tmp_project(tmp_path, monkeypatch):
        make_tmp_nbs_content(NOTEBOOK_STAGES, tmp_path)
