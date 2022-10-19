from sympy import Derivative, Function, dsolve, lambdify
from sympy.abc import a, b, c, x

y = Function("y")
_model = dsolve(
    eq=Derivative(y(x), x) + a * x + b,
    fun=y(x),
    ics={y(0): c},
).rhs  # type: ignore
_slope = _model.diff(x)  # type: ignore

model = lambdify((x, a, b, c), _model)
slope = lambdify((x, a, b, c), _slope)
...
