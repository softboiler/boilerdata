"""Pipeline functions."""

from __future__ import annotations

from datetime import datetime
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
from scipy.constants import convert_temperature
from scipy.stats import linregress, norm

from boilerdata.models.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import set_dtypes
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial
from boilerdata.validation import (
    handle_invalid_data,
    validate_final_df,
    validate_initial_df,
)

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):
    (
        pd.DataFrame(columns=[ax.name for ax in proj.axes.cols], data=get_runs(proj))
        .pipe(set_proj_dtypes, proj)
        .pipe(handle_invalid_data, validate_initial_df)
        .pipe(per_trial, proj, fit)
        .pipe(agg_and_get_95_ci, proj)
        .also(plot_fits, proj)
        .pipe(get_heat_transfer, proj)
        .pipe(per_trial, proj, assign_metadata)
        .pipe(validate_final_df)
        .also(lambda df: df.to_csv(proj.dirs.simple_results_file, encoding="utf-8"))
        .pipe(transform_for_originlab, proj)
        .to_csv(proj.dirs.originlab_results_file, index=False, encoding="utf-8")
    )
    proj.dirs.originlab_coldes_file.write_text(proj.axes.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * GET RUNS


def get_runs(proj: Project) -> pd.DataFrame:
    """Get runs from all trials."""

    # Get runs and multiindex
    dtypes = {col.name: col.dtype for col in proj.axes.source if not col.index}
    runs: list[pd.DataFrame] = []
    multiindex: list[tuple[datetime, datetime, datetime]] = []
    for trial in proj.trials:
        for file, run_index in zip(trial.run_files, trial.run_index):
            run = get_run(proj, file)
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

    # Ensure appropriate dtypes for each column.
    df = set_dtypes(df, dtypes)

    # Write the runs to disk for faster fetching later
    df.to_csv(proj.dirs.runs_file, encoding="utf-8")

    return df


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
        .pipe(set_proj_dtypes, proj)
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
    df = per_trial(df, proj, plot_fit)
    if new_fits := sorted(proj.dirs.new_fits.iterdir()):
        latest_new_fit = new_fits[-1]
        copy(latest_new_fit, Path("data/plots/latest_new_fit.png"))
    return df


def get_heat_transfer(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""

    # Constants
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # ((cm/m)^2) Conversion factor
    diameter = proj.geometry.diameter * cm_p_m  # (cm)
    cross_sectional_area = np.pi / 4 * diameter**2  # (cm^2)

    # Temperatures
    water_temp = df[proj.params.water_temps].mean().mean()  # type: ignore  # Due to use_enum_values
    midpoint_temps = (df[A.T_s] + df[A.T_1]) / 2

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
            A.DT: lambda df: (df[A.T_s] - water_temp),
            A.DT_err: lambda df: df[A.T_s_err],
        }
    )


# * -------------------------------------------------------------------------------- * #
# * PER-TRIAL STAGES


def fit(df: pd.DataFrame, trial: Trial, _) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""
    return df.assign(
        **df[list(trial.thermocouple_pos.keys())].apply(
            axis="columns",
            func=linregress_ser,
            x=list(trial.thermocouple_pos.values()),
            regression_stats=[A.dT_dx, A.T_s],
        )  # type: ignore  # Upstream issue w/ pandas-stubs
    )


def plot_fit(df: pd.DataFrame, trial: Trial, proj: Project) -> pd.DataFrame:
    """Plot the goodness of fit for each run in the trial."""

    if not trial.new:
        return df

    from matplotlib import pyplot as plt

    # Reason: Enum incompatible with str, but we have use_enum_values from Pydantic
    df.apply(
        axis="columns",
        func=plot_fit_ser,
        trial=trial,
        proj=proj,
        plt=plt,
    )  # type: ignore  # Upstream issue w/ pandas-stubs
    return df


def assign_metadata(df: pd.DataFrame, trial: Trial, proj: Project) -> pd.DataFrame:
    """Assign metadata columns to the dataframe."""
    return df.assign(
        **{
            field: value
            for field, value in trial.dict().items()  # Dict call avoids excluded properties
            if field not in [idx.name for idx in proj.axes.index]
        }
    )


# * -------------------------------------------------------------------------------- * #
# * PER-RECORD FUNCTIONS


def linregress_ser(
    y: pd.Series[float],
    x: npt.ArrayLike,
    regression_stats: list[str],
) -> pd.Series[float]:
    """Perform linear regression of a series of y's with respect to given x's."""
    r = linregress(x, y)
    # Unpacking is weird with linregress.
    # See "Notes" section: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    return pd.Series([r.slope, r.intercept], index=regression_stats)  # type: ignore  # Issue w/ upstream scipy


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
        ser[A.T_s] + ser[A.dT_dx] * x_smooth,
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


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in proj.axes.cols})


def set_proj_dtypes(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in proj.axes.cols})


def per_trial(
    df: pd.DataFrame,
    proj: Project,
    per_trial_func: Callable[[pd.DataFrame, Trial, Project], pd.DataFrame],
):
    """Apply a function to cross-sections of the dataframe corresponding to trials."""
    for trial in proj.trials:
        trial_df = df.xs(trial.trial, level=A.trial, drop_level=False)
        df.update(per_trial_func(trial_df, trial, proj))  # type: ignore

    # Set dtypes after update. https://github.com/pandas-dev/pandas/issues/4094
    return set_proj_dtypes(df, proj)


def get_run(proj: Project, run: Path) -> pd.DataFrame:
    """Get data for a single run."""

    # Get source columns
    index = proj.axes.index[-1].source  # Get the last index, associated with source
    source_col_names = [col.source for col in proj.axes.source_cols]
    source_dtypes = {col.source: col.dtype for col in proj.axes.source_cols}

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


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
