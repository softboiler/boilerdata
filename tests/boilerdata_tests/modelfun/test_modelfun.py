import numpy as np
import pytest
from boilercore.fits import fit_to_model
from sympy import Eq


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


@pytest.mark.slow()
def test_forward_model(model):
    """Test that the model evaluates to the expected output for known input."""
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


@pytest.mark.slow()
def test_model_fit(model):
    """Test that the model fit is as expected."""
    # grp.index.get_level_values(A.run)[0] == pd.Timestamp.fromisoformat("2022-09-14T10:21:00")
    from boilerdata.models.params import PARAMS

    fitted_params, errors = fit_to_model(
        model_bounds=PARAMS.fit.model_bounds,
        initial_values=PARAMS.fit.initial_values,
        free_params=PARAMS.fit.free_params,
        fit_method=PARAMS.fit.fit_method,
        model=model,
        confidence_interval_95=2.2621571627409915,
        x=np.array(
            [
                # fmt: off
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999,
                    0.10413994, 0.09207495, 0.08000996, 0.06794496, 0.02412999
                # fmt: on
            ]
        ),
        y=np.array(
            [
                # fmt: off
                93.9069161621464,  93.2765083498772, 94.47662235676236, 94.84253592682008,
                96.29970955951876, 93.88705994776728, 93.26568781919634, 94.47933457413208,
                94.84975318177332, 96.29501933917008, 93.89518329290117,  93.2720055388083,
                94.47751110595696, 94.84433832311552,  96.2864294704602, 93.91398193970622,
                93.28807934906952, 94.48729500872852, 94.85050481937738, 96.28410336442016,
                93.91938473057355,  93.2926051336643, 94.50263359396634, 94.85321224870628,
                96.28180006334132, 93.91593614065825,  93.2819224441386, 94.49737005347764,
                94.84884814873986, 96.26133641151208, 93.91231895301372,  93.2819224441386,
                94.4991781983908, 94.84703808267297, 96.26061425440815, 93.90885503603212,
                93.28206794313571, 94.49572280281524, 94.85262934607456, 96.25910152531677,
                93.93685758614453, 93.29650297521884,  94.4975309477284, 94.84811185067876,
                96.26388296551016
                # fmt: on
            ]
        ),
        y_errors=np.array(
            [
                # fmt: off
                2.2, 2.2, 2.2, 2.2, 1. , 2.2, 2.2, 2.2, 2.2, 1. , 2.2, 2.2, 2.2,
                2.2, 1. , 2.2, 2.2, 2.2, 2.2, 1. , 2.2, 2.2, 2.2, 2.2, 1. , 2.2,
                2.2, 2.2, 2.2, 1. , 2.2, 2.2, 2.2, 2.2, 1. , 2.2, 2.2, 2.2, 2.2,
                1. , 2.2, 2.2, 2.2, 2.2, 1.
                # fmt: on
            ]
        ),
        fixed_values={"k": 392.9526858487623, "h_w": 2.220446049250313e-16},
    )
    assert {
        "T_s": 95.30675253439594,
        "q_s": 3.0614461894194567e-18,
        "h_a": 6.104766706118163e-18,
        "T_s_err": 1.718480494798554,
        "q_s_err": 2.0841157474285628,
        "h_a_err": 29.9641043335052,
    } == pytest.approx(
        dict(
            zip(
                [*PARAMS.fit.free_params, *PARAMS.fit.free_errors],
                np.concatenate([fitted_params, errors]),
                strict=True,
            )
        )
    )


@pytest.mark.slow()
def test_ode(ns):
    """Verify the solution to the ODE by substitution."""
    # Don't subs/simplify the lhs then try equating to zero. Doesn't work. "Truth value of
    # relational" issue. Here we subs/simplify the whole ODE equation.
    ode, T, x, T_int_expr = ns.ode, ns.T, ns.x, ns.T_int_expr  # noqa: N806
    assert ode.subs(T(x), T_int_expr).simplify()


@pytest.mark.slow()
def test_temperature_continuous(ns):
    """Test that temperature is continuous at the domain transition."""
    T_wa_expr_w, T_wa_expr_a = ns.T_wa_expr_w, ns.T_wa_expr_a  # noqa: N806
    assert Eq(T_wa_expr_w, T_wa_expr_a).simplify()


@pytest.mark.slow()
def test_temperature_gradient_continuous(ns):
    """Test that the temperature gradient is continuous at the domain transition."""
    q_wa_expr_w, q_wa_expr_a = ns.q_wa_expr_w, ns.q_wa_expr_a
    assert Eq(q_wa_expr_w, q_wa_expr_a).simplify()
