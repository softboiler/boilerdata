from contextlib import contextmanager
from typing import Any, Mapping

from IPython.display import Markdown, display
import matplotlib as mpl
import numpy as np
import pandas as pd
from uncertainties import ufloat

from boilerdata.models.project import Project

# * -------------------------------------------------------------------------------- * #
# * MODULE VARIABLES

idxs = pd.IndexSlice
"""Use to slice pd.MultiIndex indices."""

# * -------------------------------------------------------------------------------- * #
# * PLOTTING CONTEXTS


@contextmanager
def manual_subplot_spacing():
    """Context manager that allows custom spacing of subplots."""
    with mpl.rc_context({"figure.autolayout": False}):
        try:
            yield
        finally:
            ...


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


def display_named(*args: tuple[Any, str]):
    """Display objects with names above them."""
    for elem, name in args:
        display(Markdown(f"##### {name}"))
        display(elem)


def tex_wrap(df: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Wrap column titles in LaTeX flags if they contain underscores ($)."""
    mapper: dict[str, str] = {}
    for src_col in df.columns:
        col = f"${handle_subscript(src_col)}$" if "_" in src_col else src_col
        mapper[src_col] = col
    return df.rename(axis="columns", mapper=mapper), mapper


def add_units(
    df: pd.DataFrame, proj: Project
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Make the columns a multi-index representing units."""
    cols = proj.axes.get_col_index()
    quantity = cols.get_level_values("quantity")
    units = cols.get_level_values("units")

    old = (col.name for col in proj.axes.cols)
    new = (add_unit(q, u) for q, u in zip(quantity, units))
    mapper = dict(zip(old, new))
    return df.rename(axis="columns", mapper=mapper), mapper


def get_params_mapping_with_uncertainties(
    grp: pd.DataFrame, proj: Project
) -> dict[str, Any]:
    """Get a mapping of parameter names to values with uncertainty."""
    model_params_and_errors = proj.params.params_and_errors
    # Reason: pydantic: use_enum_values
    params: list[str] = proj.params.model_params  # type: ignore
    param_errors: list[str] = proj.params.model_errors
    u_params = [
        ufloat(param, err, tag)
        for param, err, tag in zip(
            grp[params], grp[param_errors], model_params_and_errors
        )
    ]
    return dict(zip(model_params_and_errors, u_params))


def get_params_mapping(grp: pd.DataFrame, params: list[Any]) -> dict[str, Any]:
    """Get a mapping of parameter names to values."""
    # Reason: pydantic: use_enum_values
    return dict(zip(params, grp[params]))


def model_with_error(model, x, u_params):
    """Evaluate the model for x and return y with errors."""
    u_x = [ufloat(v, 0, "x") for v in x]
    u_y = model(u_x, **u_params)
    y = np.array([v.nominal_value for v in u_y])
    y_min = y - [v.std_dev for v in u_y]
    y_max = y + [v.std_dev for v in u_y]
    return y, y_min, y_max


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def handle_subscript(val: str) -> str:
    """Wrap everything after the first underscore and replace others with commas."""
    quantity, units = sep_unit(val)
    parts = quantity.split("_")
    quantity = f"{parts[0]}_" + "{" + ",".join(parts[1:]) + "}"
    return add_unit(quantity, units, tex=True)


def add_unit(quantity: str, units: str, tex: bool = False) -> str:
    """Append units to a quantity."""
    if not tex:
        return f"{quantity} ({units})" if units else quantity
    units = units.replace("-", r"{\cdot}")
    return rf"{quantity}\;({units})" if units else quantity


def sep_unit(val: str) -> tuple[str, str]:
    """Split a quantity and its units."""
    quantity, units = val.split(" (")
    units = units.removesuffix(")")
    return quantity, units
