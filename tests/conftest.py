"""Test configuration."""

from pathlib import Path

import pytest
from boilercore.testing import get_nb_client, tmp_workdir
from ploomber_engine.ipython import PloomberClient

from tests import NOTEBOOK_STAGES


@pytest.fixture()
def _tmp_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Produce a temporary project directory."""
    tmp_workdir(tmp_path, monkeypatch)


@pytest.fixture(params=NOTEBOOK_STAGES)
def nb_client(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PloomberClient:
    """Run a notebook client in a temporary project directory."""
    return get_nb_client(request, tmp_path, monkeypatch)
