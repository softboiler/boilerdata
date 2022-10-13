"""Pipeline functions."""

from __future__ import annotations

from pathlib import Path
import re
from shutil import copy
from types import ModuleType
from typing import Callable

import janitor  # type: ignore  # Magically registers methods on Pandas objects
from numpy import typing as npt
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from pyXSteam.XSteam import XSteam
from scipy.constants import convert_temperature
from scipy.optimize import curve_fit
from scipy.stats import norm

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import set_dtypes
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.validation import handle_invalid_data, validate_initial_df

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):
    (
        pd.read_csv(
            proj.dirs.runs_file,
            index_col=(index_col := [A.trial, A.run, A.time]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in proj.axes.cols},
        )
        .pipe(handle_invalid_data, validate_initial_df)
        .pipe(per_trial, fit, proj)  # Need thermocouple spacing trial-by-trial
        .pipe(agg_and_get_95_ci, proj)
        .also(plot_fits, proj)
        .pipe(per_trial, get_heat_transfer, proj)  # Water temp varies across trials
        .pipe(per_trial, assign_metadata, proj)  # Distinct per trial
        # .pipe(validate_final_df)  # TODO: Uncomment in main
        .also(lambda df: df.to_csv(proj.dirs.simple_results_file, encoding="utf-8"))
        .pipe(transform_for_originlab, proj)
        .to_csv(proj.dirs.originlab_results_file, index=False, encoding="utf-8")
    )
    proj.dirs.originlab_coldes_file.write_text(proj.axes.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * MODEL


def model(x, a, b, c):
    return a * np.exp(b * x) + c


def slope(x, a, b, c):
    return a * b * np.exp(b * x)


# * -------------------------------------------------------------------------------- * #
# * STAGES


def agg_and_get_95_ci(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Take means of numeric columns and get 95% confidence interval of fits."""
    confidence_interval_95 = abs(norm.ppf(0.025))
    complex_agg_overrides = {
        A.T_s_err: pd.NamedAgg(column=A.T_s, aggfunc="sem"),
        A.dT_dx_err: pd.NamedAgg(column=A.dT_dx, aggfunc="sem"),
    }
    aggs = proj.axes.aggs | complex_agg_overrides
    df = (
        df.groupby(level=[A.trial, A.run])  # type: ignore  # Upstream issue w/ pandas-stubs
        .agg(**aggs)
        .assign(
            **{
                A.T_s_err: lambda df: df[A.T_s_err] * confidence_interval_95,
                A.dT_dx_err: lambda df: df[A.dT_dx_err] * confidence_interval_95,
            }
        )
    )
    return df


def plot_fits(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Plot the fits."""
    df = per_trial(df, plot_fit, proj)
    if proj.params.do_plot and (new_fits := sorted(proj.dirs.new_fits.iterdir())):
        latest_new_fit = new_fits[-1]
        copy(latest_new_fit, Path("data/plots/latest_new_fit.png"))
    return df


# * -------------------------------------------------------------------------------- * #
# * PER-TRIAL STAGES


def fit(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""
    trial = proj.get_trial(df.name)  # type: ignore  # Group name is the trial
    return df.assign(
        **df[list(trial.thermocouple_pos.keys())].apply(
            axis="columns",
            func=fit_ser,
            x=list(trial.thermocouple_pos.values()),
            index=[*proj.params.model_params, A.T_s, A.dT_dx],
            proj=proj,
        )  # type: ignore  # Upstream issue w/ pandas-stubs
    )


def plot_fit(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Plot the goodness of fit for each run in the trial."""

    trial = proj.get_trial(df.name)  # type: ignore  # Group name is the trial

    if not trial.new:
        return df

    from matplotlib import pyplot as plt

    # Reason: Enum incompatible with str, but we have use_enum_values from Pydantic
    df.apply(
        axis="columns", func=plot_fit_ser, trial=trial, proj=proj, plt=plt
    )  # type: ignore  # Upstream issue w/ pandas-stubs
    return df


def get_heat_transfer(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""
    trial = proj.get_trial(df.name)  # type: ignore  # Group name is the trial
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # ((cm/m)^2) Conversion factor
    diameter = proj.geometry.diameter * cm_p_m  # (cm)
    cross_sectional_area = np.pi / 4 * diameter**2  # (cm^2)
    get_saturation_temp = XSteam(XSteam.UNIT_SYSTEM_FLS).tsat_p  # A lookup function

    T_w_avg = df[[A.T_w1, A.T_w2, A.T_w3]].mean(axis="columns")  # noqa: N806
    T_w_p = convert_temperature(  # noqa: N806
        df[A.P].apply(get_saturation_temp), "F", "C"
    )

    return df.assign(
        **{
            A.k: lambda df: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature((df[A.T_s] + df[A.T_1]) / 2, "C", "K"),
            ),
            A.T_w: lambda df: (T_w_avg + T_w_p) / 2,
            A.T_w_diff: lambda df: abs(T_w_avg - T_w_p),
            # Not negative due to reversed x-coordinate
            A.q: lambda df: df[A.k] * df[A.dT_dx] / cm2_p_m2,
            A.q_err: lambda df: (df[A.k] * df[A.dT_dx_err]) / cm2_p_m2,
            A.Q: lambda df: df[A.q] * cross_sectional_area,
            # Explicitly index the trial to catch improper application of the mean
            A.DT: lambda df: (df[A.T_s] - df.loc[trial.date.isoformat(), A.T_w].mean()),
            A.DT_err: lambda df: df[A.T_s_err],
        }
    )


def assign_metadata(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Assign metadata columns to the dataframe."""
    trial = proj.get_trial(df.name)  # type: ignore  # Group name is the trial
    # Need to re-apply categorical dtypes
    df = df.assign(
        **{
            field: value
            for field, value in trial.dict().items()  # Dict call avoids excluded properties
            if field not in [idx.name for idx in proj.axes.index]
        }
    )
    return df


# * -------------------------------------------------------------------------------- * #
# * PER-RECORD FUNCTIONS


def fit_ser(
    y: pd.Series[float], x: npt.ArrayLike, index: list[str], proj: Project
) -> pd.Series[float]:
    """Perform linear regression of a series of y's with respect to given x's."""
    try:
        params, _ = curve_fit(model, x, y)
    except RuntimeError:
        params = np.full(len(proj.params.model_params), np.nan)
    return pd.Series([*params, model(0, *params), slope(0, *params)], index=index)


def plot_fit_ser(ser: pd.Series[float], trial: Trial, proj: Project, plt: ModuleType):
    """Plot the goodness of fit for a series of temperatures and positions."""
    run = ser.name[-1].isoformat()  # type: ignore  # Issue w/ upstream pandas-stubs
    run_file = proj.dirs.new_fits / f"{run.replace(':', '-')}.png"
    plt.figure()
    plt.title(f"{run}")
    plt.xlabel("x (m)")
    plt.ylabel("T (C)")
    # Reason: Enum incompatible with str, but we have use_enum_values from Pydantic
    plt.plot(
        trial.thermocouple_pos.values(),
        ser[list(trial.thermocouple_pos.keys())],
        "*",
        label="Measurements",
        color=[0.2, 0.2, 0.2],
    )
    x_smooth = np.linspace(0, trial.thermocouple_pos[A.T_1])
    plt.plot(
        x_smooth,
        model(x_smooth, *ser[proj.params.model_params]),  # type: ignore # Upstream issue w/ pandas-stubs
        "--",
        label="Fit",
    )
    plt.plot(0, ser[A.T_s], "x", label="Extrapolation", color=[1, 0, 0])
    plt.legend()
    plt.savefig(run_file)


# * -------------------------------------------------------------------------------- * #
# * POST-PROCESSING


def transform_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels.

    Explicitly set all dtypes to string to avoid data rendering issues, especially with
    dates. Convert super/subscripts in units to their OriginLab representation. Reset
    the index to avoid the extra row between units and data indicating index axis names.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """

    superscript = re.compile(r"\^(.*)")
    superscript_repl = r"\+(\1)"
    subscript = re.compile(r"\_(.*)")
    subscript_repl = r"\-(\1)"

    cols = proj.axes.get_col_index()
    quantity = cols.get_level_values("quantity").map(
        {col.name: col.pretty_name for col in proj.axes.all}
    )
    units = cols.get_level_values("units")
    indices = [
        index.to_series()
        .reset_index(drop=True)
        .replace(superscript, superscript_repl)  # type: ignore  # Upstream issue w/ pandas-stubs
        .replace(subscript, subscript_repl)
        for index in (quantity, units)
    ]
    cols = pd.MultiIndex.from_frame(pd.concat(axis="columns", objs=indices))
    return df.set_axis(axis="columns", labels=cols).reset_index()


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def per_trial(
    df: pd.DataFrame,
    per_trial_func: Callable[[pd.DataFrame, Project], pd.DataFrame],
    proj: Project,
):
    """Apply a function to individual trials."""
    return (
        df.groupby(level=A.trial, sort=False, group_keys=False)
        .apply(per_trial_func, proj)  # type: ignore  # Issue with upstream pandas-stubs
        .pipe(set_proj_dtypes, proj)
    )


def set_proj_dtypes(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in proj.axes.cols})


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
