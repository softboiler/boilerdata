"""Pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from boilercore.fits import fit_from_params
from boilercore.models.trials import Trial
from pyXSteam.XSteam import XSteam
from scipy.constants import convert_temperature
from scipy.stats import t

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS, Mat, Params, Prop, get_prop
from boilerdata.stages import MODEL, get_tcs, get_trial, per_run, per_trial
from boilerdata.validation import (
    handle_invalid_data,
    validate_final_df,
    validate_initial_df,
)


def main():  # noqa: D103
    confidence_interval_95 = t.interval(0.95, PARAMS.records_to_average)[1]

    (
        pd.read_csv(
            PARAMS.paths.file_runs,
            index_col=(index_col := [A.trial, A.run, A.time]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in PARAMS.axes.cols},
        )
        .pipe(handle_invalid_data, validate_initial_df)
        .pipe(get_properties, PARAMS)
        .pipe(per_run, fit, PARAMS, MODEL, confidence_interval_95)
        .pipe(per_trial, agg_over_runs, PARAMS, confidence_interval_95)  # TCs may vary
        .pipe(per_trial, get_superheat, PARAMS)  # Water temp varies across trials
        .pipe(per_trial, assign_metadata, PARAMS)  # Metadata is distinct per trial
        .pipe(validate_final_df)
        .to_csv(PARAMS.paths.file_results, encoding="utf-8")
    )


def get_properties(df: pd.DataFrame, params: Params) -> pd.DataFrame:
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
                A.T_w: lambda df: (T_w_avg + T_w_p) / 2,
                A.T_w_diff: lambda df: abs(T_w_avg - T_w_p),
            }
            | dict.fromkeys(params.fit.fixed_errors, 0)  # Zero error for fixed params
        )
    )


def fit(
    grp: pd.DataFrame, params: Params, model: Any, confidence_interval_95: float
) -> pd.DataFrame:
    """Fit the data to a model function."""
    trial = get_trial(grp, params)
    # Assign thermocouple errors trial-by-trial (since they can vary)
    _, tc_errors = get_tcs(trial)
    k_type_error = 2.2
    t_type_error = 1.0
    # Need to assign here (not at the end) because these are used in the model fit
    grp = grp.assign(
        **(dict.fromkeys(tc_errors, k_type_error) | {A.T_5_err: t_type_error})
    )
    # Prepare for fitting
    x, y, y_errors = fit_setup(grp, params, trial)
    # Get fixed values
    fixed_values: dict[str, float] = params.fit.fixed_values
    for key in fixed_values:
        if not all(grp[key].isna()):
            fixed_values[key] = grp[key].mean()
    # Get bounds/guesses and override some. Can't do it earlier because of the override.
    fits, errors = fit_from_params(
        model=model,
        params=params.fit,
        x=x,
        y=y,
        y_errors=y_errors,
        confidence_interval=confidence_interval_95,
    )
    grp = grp.assign(
        **{key: fixed_values[key] for key in fixed_values if all(grp[key].isna())},
        **fits,
        **errors,
    )
    return grp


def fit_setup(grp: pd.DataFrame, params: Params, trial: Trial):
    """Reshape vectors to be passed to the curve fit."""
    tcs, tc_errors = get_tcs(trial)
    x_unique = list(trial.thermocouple_pos.values())
    x = np.tile(x_unique, params.records_to_average)
    y = grp[tcs].stack()
    y_errors = grp[tc_errors].stack()
    return x, y, y_errors


def agg_over_runs(
    grp: pd.DataFrame, params: Params, confidence_interval_95: float
) -> pd.DataFrame:
    """Aggregate properties over each run. Runs per trial because TCs can vary."""
    trial = get_trial(grp, params)
    _, tc_errors = get_tcs(trial)
    grp = (
        grp.groupby(level=[A.trial, A.run], dropna=False)  # type: ignore  # pandas
        .agg(
            **(  # type: ignore  # pandas-stubs 2.0.2
                # Take the default agg for all cols
                params.axes.aggs
                # Override the agg for cols with duplicates in a run to take the first
                | {
                    col: pd.NamedAgg(column=col, aggfunc="first")
                    for col in (tc_errors + params.fit.params_and_errors)
                }
            )
        )
        .assign(**{
            tc_error: lambda df: df[tc_error]  # noqa: B023  # False positive
            * confidence_interval_95
            / np.sqrt(params.records_to_average)
            for tc_error in tc_errors
        })
    )
    return grp


def get_superheat(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""
    # Explicitly index the trial to catch improper application of the mean
    trial = get_trial(df, params)
    return df.assign(**{
        A.DT: lambda df: (df[A.T_s] - df.loc[trial.date.isoformat(), A.T_w].mean()),
        A.DT_err: lambda df: df[A.T_s_err],
    })


def assign_metadata(grp: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Assign metadata columns to the dataframe."""
    trial = get_trial(grp, params)
    # Need to re-apply categorical dtypes
    grp = grp.assign(**{
        field: value
        for field, value in trial.dict().items()  # Dict call avoids excluded properties
        if field not in [idx.name for idx in params.axes.index]
    })
    return grp


if __name__ == "__main__":
    main()
