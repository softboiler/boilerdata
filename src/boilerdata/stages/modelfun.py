import warnings
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import dill
import numpy as np

from boilerdata.models.params import PARAMS


def get_model():
    """Unpickle the model function for fitting data."""
    model_file = PARAMS.paths.file_model
    file_bytes = Path(model_file).read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dill.UnpicklingWarning)
        unpickled_model = dill.loads(file_bytes)
    return unpickled_model.basic, fix_model(unpickled_model.for_ufloat)


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
            **{k: np.array(v) for k, v in kwargs.items()},
        )

        return result if result.size > 1 else result.item()

    return wrapper


MODEL, MODEL_WITH_UNCERTAINTY = get_model()
