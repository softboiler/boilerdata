from functools import partial

import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from pyXSteam.XSteam import XSteam
from scipy.constants import convert_temperature
from scipy.optimize import curve_fit
from scipy.stats import t

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.stages.common import get_tcs, get_trial, per_run, per_trial
from boilerdata.stages.modelfun import model
from boilerdata.validation import handle_invalid_data, validate_initial_df

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):

    confidence_interval_95 = t.interval(0.95, proj.params.records_to_average)[1]

    (
        pd.read_csv(
            proj.dirs.file_runs,
            index_col=(index_col := [A.trial, A.run, A.time]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in proj.axes.cols},
        )
        .pipe(handle_invalid_data, validate_initial_df)
        .pipe(get_properties, proj)
        .pipe(per_run, fit, proj, model, confidence_interval_95)
        .pipe(per_trial, agg_over_runs, proj, confidence_interval_95)  # TCs may vary
        .pipe(per_trial, get_superheat, proj)  # Water temp varies across trials
        .pipe(per_trial, assign_metadata, proj)  # Metadata is distinct per trial
        # .pipe(validate_final_df)  # TODO: Uncomment
        .to_csv(proj.dirs.file_results, encoding="utf-8")
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def get_properties(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get properties."""

    get_saturation_temp = XSteam(XSteam.UNIT_SYSTEM_FLS).tsat_p  # A lookup function

    T_w_avg = df[[A.T_w1, A.T_w2, A.T_w3]].mean(axis="columns")  # noqa: N806
    T_w_p = convert_temperature(  # noqa: N806
        df[A.P].apply(get_saturation_temp), "F", "C"
    )

    return df.assign(
        **(
            {
                A.k: lambda df: get_prop(
                    Mat.COPPER,
                    Prop.THERMAL_CONDUCTIVITY,
                    convert_temperature((df[A.T_1] + df[A.T_5]) / 2, "C", "K"),
                ),
                A.h_w: float(np.finfo(float).eps),  # TODO: We wrapped this in float
                A.T_w: lambda df: (T_w_avg + T_w_p) / 2,
                A.T_w_diff: lambda df: abs(T_w_avg - T_w_p),
            }
            | {k: 0 for k in proj.params.fixed_errors}  # Zero error for fixed params
        )
    )


def fit(
    grp: pd.DataFrame,
    proj: Project,
    model,
    confidence_interval_95: float,
) -> pd.DataFrame:
    """Fit the data to a model function."""

    # Make timestamp explicit due to deprecation warning
    trial = get_trial(grp, proj)

    # Get coordinates and model parameters
    fixed_param_values = dict(
        zip(
            proj.params.fixed_params,
            grp[proj.params.fixed_params].mean(),  # type: ignore  # pydantic: use_enum_values
        )
    )
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

    # Prepare for fitting
    x, y, y_errors = fit_setup(grp, proj, trial)
    T_w = grp.T_w.mean()  # noqa: N806  # Mean water temperature
    T_s_bnd = (T_w, np.inf)  # noqa: N806  # Boiling surface temp bounds
    q_s_bnd = (0, np.inf)  # Surface heat flux bounds
    h_a_bnd = (0, np.inf)  # Convection heat transfer coefficient bounds

    # Perform fit  # ! Depends on the order of the parameters
    try:
        fitted_params, pcov = curve_fit(
            partial(model, **fixed_param_values),
            x,
            y,
            sigma=y_errors,
            absolute_sigma=True,
            p0=(T_w, 1, 1),
            bounds=tuple(zip(T_s_bnd, q_s_bnd, h_a_bnd)),
        )
    except RuntimeError:
        dim = len(proj.params.free_params)
        fitted_params = np.full(dim, np.nan)
        pcov = np.full((dim, dim), np.nan)

    # Compute confidence interval
    standard_errors = np.sqrt(np.diagonal(pcov))
    errors = standard_errors * confidence_interval_95

    # Assign the same fit to all time slots in the run. Will be agged later.
    grp = grp.assign(
        **pd.Series(
            np.concatenate([fitted_params, errors]),
            index=proj.params.free_params + proj.params.free_errors,
        )  # pyright: ignore [reportGeneralTypeIssues]  # pandas
    )
    return grp


def fit_setup(grp: pd.DataFrame, proj: Project, trial: Trial):
    """Reshape vectors to be passed to the curve fit."""
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
    """Aggregate properties over each run. Runs per trial because TCs can vary."""
    trial = get_trial(grp, proj)
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
                    for col in (tc_errors + proj.params.params_and_errors)
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


def get_superheat(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""
    # Explicitly index the trial to catch improper application of the mean
    trial = get_trial(df, proj)
    return df.assign(
        **{
            A.DT: lambda df: (df[A.T_s] - df.loc[trial.date.isoformat(), A.T_w].mean()),
            A.DT_err: lambda df: df[A.T_s_err],
        }
    )


def assign_metadata(grp: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Assign metadata columns to the dataframe."""
    trial = get_trial(grp, proj)
    # Need to re-apply categorical dtypes
    grp = grp.assign(
        **{
            field: value
            for field, value in trial.dict().items()  # Dict call avoids excluded properties
            if field not in [idx.name for idx in proj.axes.index]
        }
    )
    return grp


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
