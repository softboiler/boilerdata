"""Test configuration."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pytest
from boilercore import filter_certain_warnings
from boilercore.notebooks.namespaces import get_nb_client
from boilercore.testing import get_session_path
from ploomber_engine.ipython import PloomberClient

import boilerdata
from boilerdata_tests import nbs_to_execute, stages


# Can't be session scope
@pytest.fixture(autouse=True)
def _filter_certain_warnings():
    """Filter certain warnings."""
    filter_certain_warnings(package="boilerdata")


@pytest.fixture(autouse=True, scope="session")
def project_session_path(tmp_path_factory) -> Path:
    """Project session path."""
    return get_session_path(tmp_path_factory, boilerdata)


@pytest.fixture
def params(project_session_path):
    """Parameters."""
    from boilerdata.models.params import PARAMS  # noqa: PLC0415

    return PARAMS


@pytest.fixture(params=stages)
def stage(request) -> str:
    """Stage module name."""
    return request.param


@pytest.fixture
def main(stage) -> Callable[..., None]:  # noqa: D103
    return import_module(stage).main


@pytest.fixture(params=nbs_to_execute)
def nb_to_execute(request) -> Path:
    """Path to a notebook that should be executed only."""
    return request.param


@pytest.fixture
def nb_client_to_execute(nb_to_execute) -> PloomberClient:
    """Notebook client to be executed only."""
    return get_nb_client(nb_to_execute.read_text(encoding="utf-8"))
