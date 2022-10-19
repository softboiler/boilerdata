from matplotlib import pyplot as plt
import numpy as np
from sympy import Function, dsolve, lambdify, pi, symbols
from sympy.abc import h, k, q, r, x

cm_p_m = 100  # (cm/m) Conversion factor
cm2_p_m2 = cm_p_m**2  # ((cm/m)^2) Conversion factor

T_inf, T_s = symbols("T_inf, T_s")
T = Function("T")
P = 2 * pi * r
A_c = pi * r**2
soln = dsolve(
    eq=T(x).diff(x, 2) - h * P / k / A_c * (T(x) - T_inf),  # type: ignore  # issue w/ sympy
    fun=T(x),
    ics={
        T(0): T_s,
        T(x).diff(x).subs(x, 0): (q * cm2_p_m2) / k,  # type: ignore  # issue w/ sympy
    },
)


def get_model_fun():
    model = soln.rhs.subs(  # type: ignore  # issue w/ sympy
        {
            h: 20,
            k: 398,
            r: 0.0047625,
            T_inf: 20,
        }
    )
    return lambdify((x, T_s, q), model)


def plot_model_fun(model, file):
    fig, ax = plt.subplots(layout="constrained")
    x_smooth = np.linspace(0, 0.14)
    ax.plot(x_smooth, model(x_smooth, 100, 1))
    fig.savefig(file, dpi=300)


if __name__ == "__main__":
    get_model_fun()
