"""Test configuration."""

from pathlib import Path

import pytest
from boilercore import filter_certain_warnings
from boilercore.testing import get_nb_client, get_session_path
from ploomber_engine.ipython import PloomberClient

import boilerdata
from tests import NOTEBOOK_STAGES


@pytest.fixture(scope="session", autouse=True)
def session_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Set the project directory."""
    return get_session_path(tmp_path_factory, boilerdata)


# Can't be session scope
@pytest.fixture(autouse=True)
def _filter_certain_warnings():
    """Filter certain warnings."""
    filter_certain_warnings()


@pytest.fixture(params=NOTEBOOK_STAGES)
def nb_client(request: pytest.FixtureRequest, session_path: Path) -> PloomberClient:
    """Run a notebook client in a temporary project directory."""
    return get_nb_client(request, session_path)
