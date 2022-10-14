"""Pipeline functions."""

from __future__ import annotations

from pathlib import Path
import re
from shutil import copy

import janitor  # type: ignore  # Magically registers methods on Pandas objects
from matplotlib import pyplot as plt
from more_itertools import divide, roundrobin
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from pyXSteam.XSteam import XSteam
from scipy.constants import convert_temperature
from scipy.optimize import curve_fit
from scipy.stats import t
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import set_dtypes
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.validation import handle_invalid_data, validate_initial_df

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):

    # Used in a few stages below
    confidence_interval_95 = t.interval(0.95, proj.params.records_to_average)[1]

    (
        pd.read_csv(
            proj.dirs.runs_file,
            index_col=(index_col := [A.trial, A.run, A.time]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in proj.axes.cols},
        )
        .pipe(handle_invalid_data, validate_initial_df)
        # Need thermocouple spacing run-to-run
        .pipe(per_run, fit, proj, confidence_interval_95)
        .pipe(per_trial, agg_over_runs, proj, confidence_interval_95)
        .also(plot_fits, proj)
        .pipe(per_trial, get_heat_transfer, proj)  # Water temp varies across trials
        .pipe(per_trial, assign_metadata, proj)  # Distinct per trial
        # .pipe(validate_final_df)  # TODO: Uncomment in main
        .also(lambda df: df.to_csv(proj.dirs.simple_results_file, encoding="utf-8"))
        .pipe(transform_for_originlab, proj)
        .to_csv(proj.dirs.originlab_results_file, index=False, encoding="utf-8")
    )
    proj.dirs.originlab_coldes_file.write_text(proj.axes.get_originlab_coldes())


def agg_over_runs(
    grp: pd.DataFrame,
    proj: Project,
    confidence_interval_95: float,
) -> pd.DataFrame:
    trial = proj.get_trial(pd.Timestamp(grp.name.date()))
    tcs, tc_errors = get_tcs(trial)
    complex_agg_overrides = {
        tc_err: pd.NamedAgg(column=tc, aggfunc="sem")
        for tc, tc_err in zip(tcs, tc_errors)
    }
    aggs = proj.axes.aggs | complex_agg_overrides
    grp = (
        grp.groupby(level=[A.trial, A.run])  # type: ignore  # Upstream issue w/ pandas-stubs
        .agg(**aggs)
        .assign(
            **{
                tc_err: lambda df: df[tc_err] * confidence_interval_95
                for tc_err in tc_errors
            }
        )
    )
    return grp


def plot_fits(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get the latest new model fit plot."""
    if proj.params.do_plot:
        per_run(df, plot_new_fits, proj)
        if new_fits := sorted(proj.dirs.new_fits.iterdir()):
            latest_new_fit = new_fits[-1]
            copy(latest_new_fit, Path("data/plots/latest_new_fit.png"))
    return df


def plot_new_fits(grp: pd.DataFrame, proj: Project):
    """Plot model fits for trials marked as new."""
    trial = proj.get_trial(pd.Timestamp(grp.name[0].date()))
    if not trial.new:
        return grp

    ser = grp.squeeze()
    tcs, tc_errors = get_tcs(trial)
    x_unique = list(trial.thermocouple_pos.values())
    y_unique = ser[tcs]
    u_params = np.array(
        [ufloat(param, err, tag) for param, err, tag in zip_params(ser, proj)]
    )

    # Plot setup
    fig, ax = plt.subplots(layout="constrained")

    run = ser.name[-1].isoformat()
    run_file = proj.dirs.new_fits / f"{run.replace(':', '-')}.png"

    ax.margins(0, 0)
    ax.set_title(f"{run = }")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("T (C)")

    # Initial plot boundaries
    x_bounds = np.array([0, trial.thermocouple_pos[A.T_1]])
    y_bounds = model(x_bounds, *[param.nominal_value for param in u_params])
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
    x_padded = np.linspace(xlim_min - pad, xlim_max + pad)
    y_padded, y_padded_min, y_padded_max = model_with_error(x_padded, u_params)
    ax.plot(
        x_padded,
        y_padded,
        "--",
        label="Model Fit",
    )
    ax.fill_between(
        x=x_padded,
        y1=y_padded_min,
        y2=y_padded_max,  # type: ignore
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
    fig.savefig(run_file, dpi=300)  # type: ignore  # Issue w/ matplotlib stubs


# * -------------------------------------------------------------------------------- * #
# * MODEL FIT AND RUN AGGREGATION


def model(x, a, b, c):
    return a * x**2 + b * x + c


def slope(x, a, b, c):
    return 2 * a * x + b


def fit(
    grp: pd.DataFrame,
    proj: Project,
    confidence_interval_95: float,
) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Make timestamp explicit due to deprecation warning
    trial = proj.get_trial(pd.Timestamp(grp.name[0].date()))

    # Setup
    x_unique = list(trial.thermocouple_pos.values())
    y_unique = grp[get_tcs(trial)[0]]
    x = np.tile(x_unique, proj.params.records_to_average)
    y = y_unique.stack()
    model_params = proj.params.model_params
    index = [*model_params, *proj.params.model_outs]

    # Fit
    try:
        params, pcov = curve_fit(model, x, y)
    except RuntimeError:
        params = np.full(len(model_params), np.nan)
        pcov = np.full(len(model_params), np.nan)

    # Confidence interval
    param_standard_errors = np.sqrt(np.diagonal(pcov))
    param_errors = param_standard_errors * confidence_interval_95

    # Uncertainties
    u_params = np.array(
        [
            ufloat(param, err, tag)
            for param, err, tag in zip(params, param_errors, model_params)
        ]
    )
    u_x_0 = ufloat(0, 0, "x")
    u_y_0 = model(u_x_0, *u_params)
    u_dy_dx_0 = slope(u_x_0, *u_params)
    outs = (
        u_y_0.nominal_value,
        u_y_0.std_dev,
        u_dy_dx_0.nominal_value,
        u_dy_dx_0.std_dev,
    )

    # Agg
    grp = grp.assign(**pd.Series([*roundrobin(params, param_errors), *outs], index=index))  # type: ignore  # Issue w/ pandas-stubs
    return grp


def zip_params(grp: pd.DataFrame, proj: Project):
    model_params = proj.params.model_params
    param_pairs = [list(i) for i in divide(len(model_params) // 2, model_params)]
    params = [grp[pair[0]].squeeze() for pair in param_pairs]  # type: ignore
    param_errors = [grp[pair[1]].squeeze() for pair in param_pairs]  # type: ignore
    return zip(params, param_errors, model_params)


def model_with_error(x, u_params):
    """Evaluate the model for x and return y with errors."""
    u_x = np.array([ufloat(v, 0, "x") for v in x])
    u_y = model(u_x, *u_params)
    y = np.array([y.nominal_value for y in u_y])
    y_min = y - [y.std_dev for y in u_y]  # type: ignore
    y_max = y + [y.std_dev for y in u_y]
    return y, y_min, y_max


# * -------------------------------------------------------------------------------- * #
# * TRIAL OPERATIONS


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
        .replace(superscript, superscript_repl)  # type: ignore  # Issue w/ pandas-stubs
        .replace(subscript, subscript_repl)
        for index in (quantity, units)
    ]
    cols = pd.MultiIndex.from_frame(pd.concat(axis="columns", objs=indices))
    return df.set_axis(axis="columns", labels=cols).reset_index()


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def per_trial(
    df: pd.DataFrame,
    per_trial_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual trials."""
    df = per_index(df, A.trial, per_trial_func, proj, *args, **kwargs)
    return df


def per_run(
    df: pd.DataFrame,
    per_run_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual runs."""
    df = per_index(df, [A.trial, A.run], per_run_func, proj, *args, **kwargs)
    return df


def per_index(
    df: pd.DataFrame,
    level: str | list[str],
    per_index_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    df = (
        df.groupby(level=level, sort=False, group_keys=False)  # type: ignore  # Issue w/ pandas-stubs
        .apply(per_index_func, proj, *args, **kwargs)
        .pipe(set_proj_dtypes, proj)
    )
    return df


def set_proj_dtypes(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in proj.axes.cols})


def get_tcs(trial: Trial) -> tuple[list[str], list[str]]:
    """Get the thermocouple names and their error names for this trial."""
    tcs = list(trial.thermocouple_pos.keys())
    tc_errors = [tc + "_err" for tc in tcs]
    return tcs, tc_errors


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
