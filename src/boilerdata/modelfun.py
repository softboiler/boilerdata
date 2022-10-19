from sympy import Derivative, Eq, Function, dsolve, lambdify, linsolve
from sympy.abc import a, b, c, f, g, x

y = Function("y")
ode_solution = dsolve(
    eq=Derivative(y(x), x) + f * x + g,
    fun=y(x),
    ics={y(0): c},
)
_model = (
    linsolve(
        # The system of equations we want to solve
        (
            ode_solution,  # `.args[0].args[0]` selects this equation after solving
            Eq(a, -f / 2),
            Eq(b, -g),
        ),
        # The variables we want to eliminate
        (
            y(x),
            f,
            g,
        ),
    )
    .args[0]
    .args[0]
)

_slope = _model.diff(x)  # type: ignore

model = lambdify((x, a, b, c), _model)
slope = lambdify((x, a, b, c), _slope)
...
