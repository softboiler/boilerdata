from types import SimpleNamespace
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest
from dill import UnpicklingWarning, loads
from sympy import Eq

from boilerdata_tests import MODELFUN

pytestmark = [pytest.mark.slow]


@pytest.fixture()
def ns(project_session_path) -> SimpleNamespace:
    """Namespace for the modelfun notebook."""
    return get_nb_namespace(get_nb_client(MODELFUN, project_session_path))


@pytest.fixture()
def unpickled_model(ns):
    """Unpickled model."""
    with catch_warnings():
        simplefilter("ignore", UnpicklingWarning)
        return loads(ns.pickled_model)


def test_ode(ns):
    """Verify the solution to the ODE by substitution."""
    # Don't subs/simplify the lhs then try equating to zero. Doesn't work. "Truth value of
    # relational" issue. Here we subs/simplify the whole ODE equation.
    ode, T, x, T_int_expr = ns.ode, ns.T, ns.x, ns.T_int_expr  # noqa: N806
    assert ode.subs(T(x), T_int_expr).simplify()


def test_temperature_continuous(ns):
    """Test that temperature is continuous at the domain transition."""
    T_wa_expr_w, T_wa_expr_a = ns.T_wa_expr_w, ns.T_wa_expr_a  # noqa: N806
    q_wa_expr_w, q_wa_expr_a = ns.q_wa_expr_w, ns.q_wa_expr_a
    assert Eq(T_wa_expr_w, T_wa_expr_a).simplify()


def test_temperature_gradient_continuous(ns):
    """Test that the temperature gradient is continuous at the domain transition."""
    T_wa_expr_w, T_wa_expr_a = ns.T_wa_expr_w, ns.T_wa_expr_a  # noqa: N806
    q_wa_expr_w, q_wa_expr_a = ns.q_wa_expr_w, ns.q_wa_expr_a
    assert Eq(q_wa_expr_w, q_wa_expr_a).simplify()


def test_pickle_roundtrip_basic(ns, unpickled_model):
    """Test that the unpickled basic model matches the original model."""
    assert np.allclose(
        ns.model_evaluated_at_x_smooth, unpickled_model.basic(**ns.model_kwargs)
    )


def test_pickle_roundtrip_ufloat(ns, unpickled_model):
    """Test that the unpickled model for ufloats matches the original model."""
    assert np.allclose(
        ns.model_evaluated_at_x_smooth, unpickled_model.for_ufloat(**ns.model_kwargs)
    )
