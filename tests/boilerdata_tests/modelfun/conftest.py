from types import SimpleNamespace
from warnings import catch_warnings, simplefilter

import pytest
from boilercore.modelfun import fix_model
from boilercore.testing import get_nb_client, get_nb_namespace
from dill import UnpicklingWarning, loads

from boilerdata_tests import MODELFUN


@pytest.fixture(scope="module")
def ns(project_session_path) -> SimpleNamespace:
    """Namespace for the modelfun notebook."""
    return get_nb_namespace(get_nb_client(MODELFUN, project_session_path))


@pytest.fixture()
def notebook_model(ns):
    """Notebook models."""
    return fix_model(ns.model_for_pickling.for_ufloat)


@pytest.fixture()
def unpickled_model(ns):
    """Unpickled models."""
    with catch_warnings():
        simplefilter("ignore", UnpicklingWarning)
        unpickled_model = loads(ns.pickled_model)
        return fix_model(unpickled_model.for_ufloat)


@pytest.fixture()
def stage_model(project_session_path):
    """Models as loaded prior to running stages."""
    from boilerdata.stages import MODEL_WITH_UNCERTAINTY

    return MODEL_WITH_UNCERTAINTY


@pytest.fixture(params=["notebook_model", "unpickled_model", "stage_model"])
def model(request):
    """Model."""
    return request.getfixturevalue(request.param)
