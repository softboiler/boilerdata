from functools import wraps
from pathlib import Path
from typing import Any, Callable
import warnings

import dill  # noqa: S403  # Only unpickling an internal object.
import numpy as np

from boilerdata.models.project import Project


def fix_model(f) -> Callable[..., Any]:
    """Fix edge-cases of lambdify where all inputs must be arrays.

    See the notes section in the link below where it says, "However, in some cases
    the generated function relies on the input being a numpy array."

    https://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(
            *(np.array(arg) for arg in args),
            **{k: np.array(v) for k, v in kwargs.items()}
        )

        return result if result.size > 1 else result.item()

    return wrapper


model_file = Project.get_project().dirs.file_model
file_bytes = Path(model_file).read_bytes()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", dill.UnpicklingWarning)
    unpickled_model = dill.loads(file_bytes)  # noqa: S301  # Known unpickling.
model = unpickled_model.basic
model_with_uncertainty = fix_model(unpickled_model.for_ufloat)
