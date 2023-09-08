import numpy as np
import pytest
from sympy import Eq

pytestmark = [pytest.mark.slow]


@pytest.fixture()
def ns(modelfun_namespace):
    return modelfun_namespace


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


def test_pickle_roundtrip_basic(ns):
    """Test that the unpickled basic model matches the original model."""
    assert np.allclose(
        ns.model_evaluated_at_x_smooth, ns.unpickled_model.basic(**ns.model_kwargs)
    )


def test_pickle_roundtrip_ufloat(ns):
    """Test that the unpickled model for ufloats matches the original model."""
    assert np.allclose(
        ns.model_evaluated_at_x_smooth, ns.unpickled_model.for_ufloat(**ns.model_kwargs)
    )
