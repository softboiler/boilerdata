from types import SimpleNamespace
from warnings import catch_warnings, simplefilter

import pytest
from boilercore.testing import get_nb_client, get_nb_namespace
from dill import UnpicklingWarning, loads

from boilerdata_tests import MODELFUN


@pytest.fixture(scope="module")
def ns(project_session_path) -> SimpleNamespace:
    """Namespace for the modelfun notebook."""
    return get_nb_namespace(get_nb_client(MODELFUN, project_session_path))


@pytest.fixture()
def notebook_model(ns):
    """Notebook model."""
    return ns.model_for_pickling.basic


@pytest.fixture()
def unpickled_model(ns):
    """Unpickled model."""
    with catch_warnings():
        simplefilter("ignore", UnpicklingWarning)
        return loads(ns.pickled_model).basic


@pytest.fixture()
def stage_model():
    """Model as loaded prior to running stages."""
    from boilerdata.stages import MODEL

    return MODEL


@pytest.fixture(params=["notebook_model", "unpickled_model", "stage_model"])
def model(request):
    """Model."""
    return request.getfixturevalue(request.param)
