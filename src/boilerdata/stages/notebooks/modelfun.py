# pyright 1.1.308, local/CI difference
from IPython.core.display import Math  # type: ignore
from IPython.display import display  # type: ignore
from sympy import FiniteSet
from sympy.printing.latex import latex


def math_mod(expr, long_frac_ratio=3, **kwargs):
    return Math(latex(expr, long_frac_ratio=long_frac_ratio, **kwargs))


def disp(title, *exprs, **kwargs):
    print(f"{title}:")
    display(*(math_mod(expr, **kwargs) for expr in exprs))


def disp_free(title, eqn, **kwargs):
    disp(title, eqn, **kwargs)
    disp("Free symbols", FiniteSet(*eqn.rhs.free_symbols), **kwargs)
