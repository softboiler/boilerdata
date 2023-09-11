from types import SimpleNamespace
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest
from boilercore.testing import get_nb_client, get_nb_namespace
from dill import UnpicklingWarning, loads
from sympy import Eq

from boilerdata_tests import MODELFUN

pytestmark = [pytest.mark.slow]


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


def test_model_fit(model):
    """Test that the model fit is as expected."""
    assert np.allclose(
        model(
            x=np.linspace(0, 0.10),
            T_s=105,  # (C)
            q_s=20,  # (W/cm^2)
            h_a=100,  # (W/m^2-K)
            h_w=np.finfo(float).eps,  # (W/m^2-K)
            k=400,  # (W/m-K)
        ),
        np.array(
            [
                # fmt: off
                105.        , 106.02040672, 107.04081917, 108.06122589,
                109.08163071, 110.10204124, 111.12244987, 112.14286041,
                113.16326523, 114.18367195, 115.2040844 , 116.22449112,
                117.24489594, 118.26530647, 119.2857151 , 120.30612183,
                121.32653046, 122.34693718, 123.36734962, 124.39013155,
                125.4467062 , 126.5472041 , 127.69210646, 128.88191394,
                130.11714679, 131.39834518, 132.72606933, 134.10089984,
                135.52343788, 136.99430552, 138.51414591, 140.08362367,
                141.70342508, 143.37425846, 145.09685442, 146.87196622,
                148.70037008, 150.58286552, 152.52027572, 154.51344787,
                156.56325353, 158.67058905, 160.83637592, 163.06156119,
                165.3471179 , 167.69404545, 170.10337013, 172.57614547,
                175.11345277, 177.71640154
                # fmt: on
            ]
        ),
    )


def test_ode(ns):
    """Verify the solution to the ODE by substitution."""
    # Don't subs/simplify the lhs then try equating to zero. Doesn't work. "Truth value of
    # relational" issue. Here we subs/simplify the whole ODE equation.
    ode, T, x, T_int_expr = ns.ode, ns.T, ns.x, ns.T_int_expr  # noqa: N806
    assert ode.subs(T(x), T_int_expr).simplify()


@pytest.mark.parametrize(
    "group_name",
    [
        "params",
        "inputs",
        "intermediate_vars",
        "functions",
    ],
)
def test_syms(group_name: str):
    """Test that declared symbolic variables are assigned to the correct symbols."""
    from boilerdata import syms

    module_vars = vars(syms)
    sym_group = module_vars[group_name]
    symvars = {
        var: sym
        for var, sym in module_vars.items()
        if var in [group_sym.name for group_sym in sym_group]
    }
    assert all(var == sym.name for var, sym in symvars.items())


def test_temperature_continuous(ns):
    """Test that temperature is continuous at the domain transition."""
    T_wa_expr_w, T_wa_expr_a = ns.T_wa_expr_w, ns.T_wa_expr_a  # noqa: N806
    assert Eq(T_wa_expr_w, T_wa_expr_a).simplify()


def test_temperature_gradient_continuous(ns):
    """Test that the temperature gradient is continuous at the domain transition."""
    q_wa_expr_w, q_wa_expr_a = ns.q_wa_expr_w, ns.q_wa_expr_a
    assert Eq(q_wa_expr_w, q_wa_expr_a).simplify()
