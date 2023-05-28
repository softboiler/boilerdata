# type: ignore pyright 1.1.308, local/CI differences, below
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from IPython.core.display import Markdown, Math  # type: ignore
from IPython.display import display  # type: ignore
from matplotlib import pyplot as plt
from sympy import FiniteSet, Function, symbols
from sympy.printing.latex import latex
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS, Params
from boilerdata.models.trials import Trial

idxs = pd.IndexSlice
"""Use to slice pd.MultiIndex indices."""

# * -------------------------------------------------------------------------------- * #
# * SYMBOLIC PARAMETER GROUPS

params = symbols(["x", *PARAMS.free_params, *PARAMS.fixed_params])
(
    x,
    T_s,
    q_s,
    h_a,
    k,
    h_w,
) = params

inputs = symbols(list(PARAMS.model_inputs.keys()))
(
    r,
    T_infa,
    T_infw,
    x_s,
    x_wa,
) = inputs

intermediate_vars = symbols(
    """
    h,
    q_0,
    q_wa,
    T_0,
    T_inf,
    T_wa,
    x_0,
    """
)
(
    h,  # (W/m^2-K) Convection heat transfer coefficient
    q_0,  # (W/m^2) q at x_0, the LHS of a general domain
    q_wa,  # (W/m^2) q at the domain interface
    T_0,  # (C) T at x_0, the LHS of a general domain
    T_inf,  # (C) Ambient temperature
    T_wa,  # (C) T at the domain interface
    x_0,  # (m) x at the LHS of a general domain
) = intermediate_vars

functions = symbols(
    """
    T*,
    T_a,
    T_w,
    T,
    """,
    cls=Function,  # type: ignore  # sympy
)
(
    T_int,  # (T*, C) The general solution to the ODE
    T_a,  # (C) Solution in air
    T_w,  # (C) Solution in water
    T,  # (C) The piecewise combination of the two above solutions
) = functions


# * -------------------------------------------------------------------------------- * #
# * DATA MANIPULATION


def get_run(params: Params, run: Path) -> pd.DataFrame:
    """Get data for a single run."""

    # Get source columns
    index = params.axes.index[-1].source  # Get the last index, associated with source
    source_col_names = [col.source for col in params.axes.source_cols]
    source_dtypes = {col.source: col.dtype for col in params.axes.source_cols}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    df = pd.DataFrame(
        columns=source_col_names,
        data=pd.read_csv(
            run,
            # Allow source cols to be missing (such as certain thermocouples)
            usecols=lambda col: col in [index, *source_col_names],
            index_col=index,
            parse_dates=[index],  # type: ignore  # pandas
            dtype=source_dtypes,
            encoding="utf-8",
        )
        # Rarely a run has an all NA record at the end
    ).dropna(how="all")

    # Need "df" defined so we can call "df.index.dropna()". Repeat `dropna` because a
    # run can have an NA index at the end and a CSV can have an all NA record at the end
    return (
        df.reindex(index=df.index.dropna())
        .dropna(how="all")
        .pipe(rename_columns, params)
    )


def get_trial(df: pd.DataFrame, params: Params) -> Trial:
    """Get the trial represented in a given dataframe, verifying it is the only one."""
    trials_in_df = df.index.get_level_values(A.trial)
    if trials_in_df.nunique() > 1:
        raise RuntimeError("There was more than one trial when getting the trial.")
    trial = pd.Timestamp(trials_in_df[0].date())  # type: ignore  # pandas
    return params.get_trial(trial)


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    return (
        df
        if df.empty
        else df.assign(
            **{name: df[name].astype(dtype) for name, dtype in dtypes.items()}  # type: ignore
        )
    )


def set_proj_dtypes(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in params.axes.cols})


def per_index(
    df: pd.DataFrame,
    level: str | list[str],
    per_index_func,
    params: Params,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Group dataframe by index and apply a function to the groups, setting dtypes."""
    df = (
        df.groupby(level=level, sort=False, group_keys=False)  # type: ignore
        .apply(per_index_func, params, *args, **kwargs)
        .pipe(set_proj_dtypes, params)
    )
    return df


def per_trial(
    df: pd.DataFrame,
    per_trial_func,
    params: Params,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual trials."""
    df = per_index(df, A.trial, per_trial_func, params, *args, **kwargs)
    return df


def per_run(
    df: pd.DataFrame,
    per_run_func,
    params: Params,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual runs."""
    df = per_index(df, [A.trial, A.run], per_run_func, params, *args, **kwargs)
    return df


def add_units(
    df: pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Make the columns a multi-index representing units."""
    cols = params.axes.get_col_index()
    quantity = cols.get_level_values("quantity")
    units = cols.get_level_values("units")

    old = (col.name for col in params.axes.cols)
    new = (add_unit(q, u) for q, u in zip(quantity, units, strict=True))  # type: ignore  # pyright 1.1.310, pandas
    mapper = dict(zip(old, new, strict=True))
    return df.rename(axis="columns", mapper=mapper), mapper


# * -------------------------------------------------------------------------------- * #
# * DISPLAY


def set_format():
    """Set up formatting for interactive notebook sessions.
    The triple curly braces in the f-string allows the format function to be dynamically
    specified by a given float specification. The intent is clearer this way, and may be
    extended in the future by making `float_spec` a parameter.
    """
    float_spec = ":#.4g"
    pd.options.display.min_rows = pd.options.display.max_rows = 50
    pd.options.display.float_format = f"{{{float_spec}}}".format


def disp_named(*args: tuple[Any, str]):
    """Display objects with names above them."""
    for elem, name in args:
        display(Markdown(f"##### {name}"))
        display(elem)


def disp_free(title, eqn, **kwargs):
    disp(title, eqn, **kwargs)
    disp("Free symbols", FiniteSet(*eqn.rhs.free_symbols), **kwargs)


def disp(title, *exprs, **kwargs):
    print(f"{title}:")
    display(*(math_mod(expr, **kwargs) for expr in exprs))


def math_mod(expr, long_frac_ratio=3, **kwargs):
    return Math(latex(expr, long_frac_ratio=long_frac_ratio, **kwargs))


# * -------------------------------------------------------------------------------- * #
# * PLOTTING


@contextmanager
def manual_subplot_spacing():
    """Context manager that allows custom spacing of subplots."""
    with mpl.rc_context({"figure.autolayout": False}):
        try:
            yield
        finally:
            ...


def tex_wrap(df: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Wrap column titles in LaTeX flags if they contain underscores ($)."""
    mapper: dict[str, str] = {}
    for src_col in df.columns:
        col = f"${handle_subscript(src_col)}$" if "_" in src_col else src_col
        mapper[src_col] = col
    return df.rename(axis="columns", mapper=mapper), mapper


def plot_new_fits(grp: pd.DataFrame, params: Params, model):
    """Plot model fits for trials marked as new."""

    trial = get_trial(grp, params)
    if not trial.new:
        return grp

    tcs, tc_errors = get_tcs(trial)
    x_unique = list(trial.thermocouple_pos.values())
    figs = dict(
        zip(
            [
                params.paths.plot_new_fit_0,
                params.paths.plot_new_fit_1,
                params.paths.plot_new_fit_2,
            ],
            grp.iloc[
                [
                    0,
                    (len(grp) // 2) - 1,
                    len(grp) - 1,
                ]
            ].iterrows(),
            strict=True,
        )
    )

    for fig_dst, (ser_name, ser) in figs.items():
        y_unique = ser[tcs]

        # Plot setup
        fig, ax = plt.subplots(layout="constrained")
        run = ser_name[-1].isoformat()  # type: ignore  # pandas
        ax.margins(0, 0)
        ax.set_title(f"{run = }")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("T (C)")

        # Initial plot boundaries
        x_bounds = np.array([0, trial.thermocouple_pos[A.T_1]])
        y_bounds = model(x_bounds, **get_params_mapping(ser, params.model_params))
        ax.plot(
            x_bounds,
            y_bounds,
            "none",
        )

        # Measurements
        measurements_color = [0.2, 0.2, 0.2]
        ax.plot(
            x_unique,
            y_unique,
            ".",
            label="Measurements",
            color=measurements_color,
            markersize=10,
        )
        ax.errorbar(
            x=x_unique,
            y=y_unique,
            yerr=ser[tc_errors],
            fmt="none",
            color=measurements_color,
        )

        # Confidence interval
        (xlim_min, xlim_max) = ax.get_xlim()
        pad = 0.025 * (xlim_max - xlim_min)
        x_padded = np.linspace(xlim_min - pad, xlim_max + pad, 200)

        y_padded, y_padded_min, y_padded_max = get_model_with_error(
            model, x_padded, get_params_mapping_with_uncertainties(ser, params)
        )
        ax.plot(
            x_padded,
            y_padded,
            "--",
            label="Model Fit",
        )
        ax.fill_between(
            x=x_padded,
            y1=y_padded_min,  # type: ignore  # pydantic: use_enum_values # Only in CI
            y2=y_padded_max,  # type: ignore  # matplotlib
            color=[0.8, 0.8, 0.8],
            edgecolor=[1, 1, 1],
            label="95% CI",
        )

        # Extrapolation
        ax.plot(
            0,
            ser[A.T_s],
            "x",
            label="Extrapolation",
            color=[1, 0, 0],
        )

        # Finishing
        ax.legend()
        fig.savefig(
            fig_dst,  # type: ignore  # matplotlib
            dpi=300,
        )


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def rename_columns(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in params.axes.cols})


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


def get_params_mapping(
    grp: pd.Series | pd.DataFrame, model_params: list[Any]  # type: ignore  # pandas
) -> dict[str, Any]:
    """Get a mapping of parameter names to values."""
    # Reason: pydantic: use_enum_values
    return dict(zip(model_params, grp[model_params], strict=True))


def get_params_mapping_with_uncertainties(
    grp: pd.Series | pd.DataFrame, params: Params  # type: ignore  # pandas
) -> dict[str, Any]:
    """Get a mapping of parameter names to values with uncertainty."""
    # Reason: pydantic: use_enum_values
    model_params: list[str] = params.model_params  # type: ignore
    param_errors: list[str] = params.model_errors  # type: ignore
    model_param_uncertainties = [
        ufloat(model_param, model_error, tag)
        for model_param, model_error, tag in zip(
            grp[model_params], grp[param_errors], model_params, strict=True
        )
    ]
    return dict(zip(model_params, model_param_uncertainties, strict=True))


def get_model_with_error(model, x, model_param_uncertainties):
    """Evaluate the model for x and return y with errors."""
    u_x = [ufloat(v, 0, "x") for v in x]
    u_y = model(u_x, **model_param_uncertainties)
    y = np.array([v.nominal_value for v in u_y])
    y_min = y - [v.std_dev for v in u_y]  # type: ignore # pyright 1.1.308, local/CI difference
    y_max = y + [v.std_dev for v in u_y]
    return y, y_min, y_max


def get_tcs(trial: Trial) -> tuple[list[str], list[str]]:
    """Get the thermocouple names and their error names for this trial."""
    tcs = list(trial.thermocouple_pos.keys())
    tc_errors = [tc + "_err" for tc in tcs]
    return tcs, tc_errors
