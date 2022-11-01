"""Pipeline functions."""

# # Necessary as long as a line marked "triggered only in CI" is in this file
# pyright: reportUnnecessaryTypeIgnoreComment=none


from functools import (
    wraps,  # pyright: ignore [reportUnusedImport]  # Needed for unpickled model
)
from pathlib import Path
import re
from shutil import copy

import janitor  # pyright: ignore [reportUnusedImport]  # Registers methods on Pandas objects
from matplotlib import pyplot as plt
import numpy as np  # pyright: ignore [reportUnusedImport]  # Also needed for unpickled model
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from pyXSteam.XSteam import XSteam
from scipy.constants import convert_temperature
from scipy.optimize import curve_fit
from scipy.stats import t
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.modelfun import model
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.utils import get_tcs, model_with_error, per_run, per_trial, zip_params
from boilerdata.validation import (
    handle_invalid_data,
    validate_final_df,
    validate_initial_df,
)

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):

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
        .pipe(per_run, fit, proj, model, confidence_interval_95)
        .pipe(per_trial, agg_over_runs, proj, confidence_interval_95)
        .pipe(plot_fits, proj, model)
        .pipe(per_trial, get_heat_transfer, proj)  # Water temp varies across trials
        .pipe(per_trial, assign_metadata, proj)  # Distinct per trial
        .pipe(validate_final_df)
        .also(lambda df: df.to_csv(proj.dirs.simple_results_file, encoding="utf-8"))
        .pipe(transform_for_originlab, proj)
        .to_csv(proj.dirs.originlab_results_file, index=False, encoding="utf-8")
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def fit(
    grp: pd.DataFrame,
    proj: Project,
    model,
    confidence_interval_95: float,
) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Make timestamp explicit due to deprecation warning
    trial = proj.get_trial(pd.Timestamp(grp.name[0].date()))

    # Get coordinates and model parameters
    model_params = proj.params.model_params
    _, tc_errors = get_tcs(trial)

    # Assign thermocouple errors
    k_type_error = 2.2
    t_type_error = 1.0
    grp = grp.assign(
        **(
            {tc_error: k_type_error for tc_error in tc_errors}
            | {A.T_5_err: t_type_error}
        )
    )
    x, y, y_errors = fit_setup(grp, proj, trial)

    # Perform fit
    try:
        model_params_fitted, pcov = curve_fit(
            model,
            x,
            y,
            sigma=y_errors,
            absolute_sigma=True,
            bounds=(0, np.inf),
        )
    except RuntimeError:
        dim = len(model_params) // 2
        model_params_fitted = np.full(dim, np.nan)
        pcov = np.full((dim, dim), np.nan)

    # Compute confidence interval
    param_standard_errors = np.sqrt(np.diagonal(pcov))
    param_errors = param_standard_errors * confidence_interval_95

    # Assign the same fit to all time slots in the run. Will be agged later.
    grp = grp.assign(
        **pd.Series(
            np.concatenate([model_params_fitted, param_errors]), index=model_params
        )  # pyright: ignore [reportGeneralTypeIssues]  # pandas
    )
    return grp


def fit_setup(grp: pd.DataFrame, proj: Project, trial: Trial):
    tcs, tc_errors = get_tcs(trial)
    x_unique = list(trial.thermocouple_pos.values())
    x = np.tile(x_unique, proj.params.records_to_average)
    y = grp[tcs].stack()
    y_errors = grp[tc_errors].stack()
    return x, y, y_errors


def agg_over_runs(
    grp: pd.DataFrame,
    proj: Project,
    confidence_interval_95: float,
) -> pd.DataFrame:

    trial = proj.get_trial(pd.Timestamp(grp.name.date()))
    _, tc_errors = get_tcs(trial)
    grp = (
        grp.groupby(
            level=[
                A.trial,
                A.run,
            ],  # pyright: ignore [reportGeneralTypeIssues]  # pandas
            dropna=False,
        )
        .agg(
            **(
                # Take the default agg for all cols
                proj.axes.aggs
                # Override the agg for cols with duplicates in a run to take the first
                | {
                    col: pd.NamedAgg(
                        column=col,  # pyright: ignore [reportGeneralTypeIssues]  # pydantic: use_enum_values
                        aggfunc="first",
                    )
                    for col in (tc_errors + proj.params.model_params)
                }
            )
        )
        .assign(
            **{
                tc_error: lambda df: df[tc_error]
                * confidence_interval_95
                / np.sqrt(proj.params.records_to_average)
                for tc_error in tc_errors
            }
        )
    )
    return grp


def plot_fits(df: pd.DataFrame, proj: Project, model) -> pd.DataFrame:
    """Get the latest new model fit plot."""
    if proj.params.do_plot:
        per_run(df, plot_new_fits, proj, model)
        if figs_src := sorted(proj.dirs.new_fits.iterdir()):
            figs_src = (figs_src[0], figs_src[-1])
            figs_dst = (
                Path(f"data/plots/{num}_new_fit.png") for num in ("first", "last")
            )
            for fig_src, fig_dst in zip(figs_src, figs_dst):
                copy(fig_src, fig_dst)
    return df


def plot_new_fits(grp: pd.DataFrame, proj: Project, model):
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
    # model = uncertainties.wrap(model)
    (xlim_min, xlim_max) = ax.get_xlim()
    pad = 0.025 * (xlim_max - xlim_min)
    x_padded = np.linspace(xlim_min - pad, xlim_max + pad)
    y_padded, y_padded_min, y_padded_max = model_with_error(model, x_padded, u_params)
    ax.plot(
        x_padded,
        y_padded,
        "--",
        label="Model Fit",
    )
    ax.fill_between(
        x=x_padded,
        y1=y_padded_min,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib, triggered only in CI
        y2=y_padded_max,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
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
        run_file,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
        dpi=300,
    )


def get_heat_transfer(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""
    trial = proj.get_trial(
        df.name  # pyright: ignore [reportGeneralTypeIssues]  # pandas
    )
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
                convert_temperature((df[A.T_1] + df[A.T_5]) / 2, "C", "K"),
            ),
            A.T_w: lambda df: (T_w_avg + T_w_p) / 2,
            A.T_w_diff: lambda df: abs(T_w_avg - T_w_p),
            # Explicitly index the trial to catch improper application of the mean
            A.DT: lambda df: (df[A.T_s] - df.loc[trial.date.isoformat(), A.T_w].mean()),
            A.DT_err: lambda df: df[A.T_s_err],
        }
    )


def assign_metadata(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Assign metadata columns to the dataframe."""
    trial = proj.get_trial(
        df.name  # pyright: ignore [reportGeneralTypeIssues]  # pandas
    )
    # Need to re-apply categorical dtypes
    df = df.assign(
        **{
            field: value
            for field, value in trial.dict().items()  # Dict call avoids excluded properties
            if field not in [idx.name for idx in proj.axes.index]
        }
    )
    return df


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
        .replace(
            superscript,  # pyright: ignore [reportGeneralTypeIssues]  # pandas
            superscript_repl,
        )
        .replace(subscript, subscript_repl)
        for index in (quantity, units)
    ]
    cols = pd.MultiIndex.from_frame(pd.concat(axis="columns", objs=indices))
    return df.set_axis(axis="columns", labels=cols).reset_index()


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
