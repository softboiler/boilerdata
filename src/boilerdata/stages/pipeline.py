"""Pipeline functions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from types import ModuleType

from numpy import typing as npt
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from scipy.constants import convert_temperature
from scipy.stats import linregress, norm

from boilerdata.models.axes import Axes
from boilerdata.models.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import set_dtypes
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.validation import (
    handle_invalid_data,
    validate_final_df,
    validate_runs_df,
)

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):

    # Get dataframe of all runs and reduce to steady-state
    runs_df = get_runs(proj)
    runs_df = handle_invalid_data(proj, runs_df, validate_runs_df)

    df = pd.DataFrame(columns=Axes.get_names(proj.axes.cols)).assign(
        **get_steady_state(runs_df, proj)  # type: ignore  # All DataFrames from CSV guarantees str keys, not expressible in types.
    )

    # Perform fits and compute heat transfer for each trial
    for trial in proj.trials:
        trial_df = df.xs(trial.trial, level=A.trial, drop_level=False)
        df.update(
            trial_df.pipe(assign_metadata, proj, trial)
            .pipe(fit, proj, trial)
            .pipe(get_heat_transfer, proj, trial)
        )
        if proj.params.do_plot:
            plot_fit_apply(trial_df, proj, trial)

    # Set dtypes after update. https://github.com/pandas-dev/pandas/issues/4094
    dtypes = {col.name: col.dtype for col in proj.axes.cols}
    df = df.pipe(set_dtypes, dtypes)
    df = handle_invalid_data(proj, df, validate_final_df)

    # Write a simple version of results to CSV for quick-reference
    df.to_csv(proj.dirs.simple_results_file, encoding="utf-8")

    # Post-process the dataframe for writing to OriginLab-flavored CSV
    df.pipe(transform_for_originlab, proj).to_csv(
        proj.dirs.originlab_results_file, index=False, encoding="utf-8"
    )
    proj.dirs.originlab_coldes_file.write_text(proj.axes.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * GET RUNS


def get_runs(proj: Project) -> pd.DataFrame:
    """Get runs from all the data CSVs."""

    # Get runs and multiindex
    dtypes = {col.name: col.dtype for col in proj.axes.source if not col.index}
    runs: list[pd.DataFrame] = []
    multiindex: list[tuple[datetime, datetime, datetime]] = []
    for trial in proj.trials:
        for file, run_index in zip(trial.run_files, trial.run_index):
            run = get_run(proj, trial, file)
            runs.append(run)
            multiindex.extend(
                tuple((*run_index, record_time) for record_time in run.index)
            )

    # Concatenate results from all runs and set the multiindex
    df = pd.concat(runs).set_index(
        pd.MultiIndex.from_tuples(
            multiindex, names=[idx.name for idx in proj.axes.index]
        )
    )

    # Ensure appropriate dtypes for each column. Validate number of records in each run.
    df = set_dtypes(df, dtypes)

    # Write the runs to disk for faster fetching later
    df.to_csv(proj.dirs.runs_file, encoding="utf-8")

    return df


def get_run(proj: Project, trial: Trial, run: Path) -> pd.DataFrame:

    # Get source columns
    index = proj.axes.index[-1].source  # Get the last index, associated with source
    source_cols = [col for col in proj.axes.source if not col.index]
    source_col_names = [col.source for col in source_cols]
    source_dtypes = {col.source: col.dtype for col in source_cols}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    df = (
        pd.DataFrame(columns=source_col_names)
        .assign(
            **pd.read_csv(
                run,
                # Allow source cols to be missing (such as T_6)
                usecols=lambda col: col in [index, *source_col_names],
                index_col=index,
                parse_dates=[index],  # type: ignore  # Upstream issue w/ pandas-stubs
                dtype=source_dtypes,  # type: ignore  # Upstream issue w/ pandas-stubs
                encoding="utf-8",
            )
        )
        .dropna(how="all")  # Rarely a run has an all NA record at the end
    )

    # Need "df" defined so we can call "df.index.dropna()"
    return (
        df.reindex(index=df.index.dropna())  # A run can have an NA index at the end
        .dropna(how="all")  # A CSV can have an all NA record at the end
        .tail(proj.params.records_to_average)
        .pipe(rename_columns, proj)
    )


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in proj.axes.cols})


# * -------------------------------------------------------------------------------- * #
# * ASSIGN METADATA


def assign_metadata(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:

    # Get metadata
    metadata = {
        field: value
        for field, value in trial.dict().items()  # Dict call avoids excluded properties
        if field not in [idx.name for idx in proj.axes.index]
    }

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.

    return df.assign(**metadata)


# * -------------------------------------------------------------------------------- * #
# * REDUCE DATA


def get_steady_state(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get the steady-state values for each run."""
    cols_to_mean = [
        col.name for col in proj.axes.all if col.source and col.dtype == "float"
    ]
    means = df[cols_to_mean].groupby(level=A.run, sort=False).transform("mean")
    return df.assign(**means).droplevel(A.time)[:: proj.params.records_to_average]


# * -------------------------------------------------------------------------------- * #
# * MAIN PIPELINE


def fit(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""
    df = df.pipe(
        linregress_apply,
        proj=proj,
        trial=trial,
        temperature_cols=df[list(trial.thermocouple_pos.keys())],
        result_cols=[A.dT_dx, A.dT_dx_err, A.T_s, A.T_s_err, A.rvalue, A.pvalue],
    )
    return df


def get_heat_transfer(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""

    # Constants
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # ((cm/m)^2) Conversion factor
    diameter = proj.geometry.diameter * cm_p_m  # (cm)
    cross_sectional_area = np.pi / 4 * diameter**2  # (cm^2)

    # Temperatures
    trial_water_temp = df[proj.params.water_temps].mean().mean()  # type: ignore  # Due to use_enum_values
    midpoint_temps = (trial_water_temp + df[A.T_1]) / 2

    return df.assign(
        **{
            A.k: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(midpoint_temps, "C", "K"),
            ),
            # no negative due to reversed x-coordinate
            A.q: lambda df: df[A.k] * df[A.dT_dx] / cm2_p_m2,
            A.q_err: lambda df: (df[A.k] * df[A.dT_dx_err]) / cm2_p_m2,
            A.Q: lambda df: df[A.q] * cross_sectional_area,
            A.DT: lambda df: (df[A.T_s] - trial_water_temp),
            A.DT_err: lambda df: df[A.T_s_err],
        }
    )


# * -------------------------------------------------------------------------------- * #
# * LINEAR REGRESSION


def linregress_apply(
    df: pd.DataFrame,
    proj: Project,
    trial: Trial,
    temperature_cols: pd.DataFrame,
    result_cols: list[str],
) -> pd.DataFrame:
    """Apply linear regression across the temperature columns."""
    return df.assign(
        **temperature_cols.apply(
            axis="columns",
            func=linregress_ser,
            x=list(trial.thermocouple_pos.values()),
            repeats_per_pair=proj.params.records_to_average,
            regression_stats=result_cols,
        )  # type: ignore  # Upstream issue w/ pandas-stubs
    )


def linregress_ser(
    series_of_y: pd.Series[float],
    x: npt.ArrayLike,
    repeats_per_pair: int,
    regression_stats: list[str],
) -> pd.Series[float]:
    """Perform linear regression of a series of y's with respect to given x's.

    Given x-values and a series of y-values, return a series of linear regression
    statistics.
    """
    # Assume the ordered pairs are repeated with zero standard deviation in x and y
    x = np.repeat(x, repeats_per_pair)
    y = series_of_y.repeat(repeats_per_pair)
    r = linregress(x, y)

    # Confidence interval
    confidence_interval_95 = abs(norm.ppf(0.025))
    slope_err = confidence_interval_95 * r.stderr  # type: ignore  # Issue w/ upstream scipy
    int_err = confidence_interval_95 * r.intercept_stderr  # type: ignore  # Issue w/ upstream scipy

    # Unpacking would drop r.intercept_stderr, so we have to do it this way.
    # See "Notes" section of SciPy documentation for more info.
    return pd.Series(
        [r.slope, slope_err, r.intercept, int_err, r.rvalue, r.pvalue],  # type: ignore  # Issue w/ upstream scipy
        index=regression_stats,
    )


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
# * PLOTTING


def plot_fit_apply(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Plot the goodness of fit for each run in the trial."""
    from matplotlib import pyplot as plt

    # Reason: Enum incompatible with str, but we have use_enum_values from Pydantic
    df.apply(
        axis="columns",
        func=plot_fit_ser,
        proj=proj,
        trial=trial,
        plt=plt,
    )  # type: ignore  # Upstream issue w/ pandas-stubs
    plt.show()
    return df


def plot_fit_ser(ser: pd.Series[float], proj: Project, trial: Trial, plt: ModuleType):
    """Plot the goodness of fit for a series of temperatures and positions."""
    plt.figure()
    plt.title("Temperature Profile in Post")
    plt.xlabel("x (m)")
    plt.ylabel("T (C)")
    # Reason: Enum incompatible with str, but we have use_enum_values from Pydantic
    plt.plot(
        trial.thermocouple_pos.values(),
        ser[list(trial.thermocouple_pos.keys())],
        "*",
        label="Measured Temperatures",
        color=[0.2, 0.2, 0.2],
    )
    x_smooth = np.linspace(0, trial.thermocouple_pos[A.T_1])
    plt.plot(
        x_smooth,
        ser[A.T_s] + ser[A.dT_dx] * x_smooth,
        "--",
        label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
    )
    plt.plot(
        0, ser[A.T_s], "x", label="Extrapolated Surface Temperature", color=[1, 0, 0]
    )
    plt.legend()


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
