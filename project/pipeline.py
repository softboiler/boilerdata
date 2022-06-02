"""Pipeline functions."""

from datetime import datetime
from pathlib import Path
import re
from types import ModuleType

from numpy import typing as npt
import numpy as np
import pandas as pd
from pint import UnitRegistry
from pint_pandas import PintType
from propshop import get_prop
from propshop.library import Mat, Prop
from scipy.constants import convert_temperature
from scipy.stats import linregress

from axes import Axes as A  # noqa: N817
from constants import TEMPS_TO_REGRESS, WATER_TEMPS
from models import Project, Trial, get_names, set_dtypes
from utils import get_project
from validation import validate_df, validate_runs_df

u = UnitRegistry()
Q = u.Quantity
u.load_definitions(Path("project/config/units.txt"))
PintType.ureg = u

# * -------------------------------------------------------------------------------- * #
# * MAIN


# TODO: Refactor into parallel pipeline
# # Quantify the columns with units
# df.columns = multi_index  # Set the multi-index so quantify can work
# df = df.pint.quantify()  # Changes column dtypes to unit-aware dtypes
# df.columns = multi_index.droplevel(-1)  # Go back to the simple index


def pipeline(proj: Project):

    # Get dataframe of all runs and reduce to steady-state
    runs_df = validate_runs_df(get_df(proj))
    df = pd.DataFrame(columns=get_names(proj.axes.cols)).assign(**get_steady_state(runs_df, proj))  # type: ignore

    # Perform fits and compute heat transfer for each trial
    for trial in proj.trials:
        df.update(
            df.xs(trial.trial, level=A.trial, drop_level=False)
            .pipe(fit, proj, trial)
            .pipe(get_heat_transfer, proj, trial)
        )
    # Set dtypes after update. https://github.com/pandas-dev/pandas/issues/4094
    dtypes = {col.name: col.dtype for col in proj.axes.cols}
    df = validate_df(df.pipe(set_dtypes, dtypes))

    # Post-process the dataframe for writing to OriginLab-flavored CSV
    df.pipe(transform_for_originlab, proj).to_csv(
        proj.dirs.results_file,
        index=False,
        encoding="utf-8",
    )
    proj.dirs.coldes_file.write_text(proj.axes.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * GET DATAFRAME OF ALL RUNS AND TIMES


def get_df(proj: Project) -> pd.DataFrame:
    """Get the dataframe of all runs."""

    dtypes = {
        col.name: col.dtype
        for col in (proj.axes.source + proj.axes.meta)
        if not col.index
    }

    # Fetch all runs from their original CSVs if needed.
    if any(trial.new for trial in proj.trials) or proj.params.refetch_runs:
        return get_runs(proj, dtypes)

    # Otherwise, reload the dataframe last written by `get_runs()`
    index_cols = list(range(len(proj.axes.index)))

    return pd.read_csv(
        proj.dirs.runs_file,
        index_col=index_cols,
        dtype=dtypes,
        parse_dates=index_cols,
        encoding="utf-8",
    )


def get_runs(proj: Project, dtypes: dict[str, str]) -> pd.DataFrame:
    """Get runs from all the data CSVs."""

    # Get runs and multiindex
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

    # Get metadata
    metadata = {
        k: v
        for k, v in trial.dict().items()  # Dict call avoids excluded properties
        if k not in [idx.name for idx in proj.axes.index]
    }

    # Get source columns
    index = proj.axes.index[-1].source  # Get the last index, associated with source
    source_cols = [col for col in proj.axes.source if not col.index]
    source_col_names = [col.source for col in source_cols]
    source_dtypes = {col.source: col.dtype for col in source_cols}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    df = (
        pd.DataFrame(columns=get_names(proj.axes.meta) + source_col_names)  # type: ignore  # Guarded in source property
        .assign(
            **pd.read_csv(
                run,
                usecols=[index, *source_col_names],  # type: ignore  # Guarded in source property
                index_col=index,
                parse_dates=[index],  # type: ignore  # Guarded in index property
                dtype=source_dtypes,
                encoding="utf-8",
            ),
            **metadata,
        )
        .dropna(how="all")  # Rarely a run has an all NA record at the end
    )
    # Need "df" defined so we can call "df.index.dropna()"
    return (
        df.reindex(index=df.index.dropna())  # Rarely a run has an NA index at the end
        .tail(proj.params.records_to_average)
        .pipe(rename_columns, proj)
    )


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units into a MultiIndex."""

    return df.rename(
        {col.source: col.name for col in proj.axes.cols},
        axis="columns",
    )


# * -------------------------------------------------------------------------------- * #
# * GET REDUCED DATA FOR EACH RUN


def get_steady_state(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get the steady-state values for each run."""
    cols_to_mean = [
        col.name for col in proj.axes.all if col.source and col.dtype == "float"
    ]
    means = df[cols_to_mean].groupby(level=A.run, sort=False).transform("mean")
    return df.assign(**means).droplevel(A.time)[:: proj.params.records_to_average]  # type: ignore


# * -------------------------------------------------------------------------------- * #
# * PERFORM FITS AND COMPUTE HEAT TRANSFER


def fit(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    df = df.pipe(
        linregress_apply,
        proj=proj,
        trial=trial,
        temperature_cols=df[TEMPS_TO_REGRESS],
        result_cols=[
            A.dT_dx,
            A.TLfit,
            A.rvalue,
            A.pvalue,
            A.stderr,
            A.intercept_stderr,
        ],
    )

    return df


def plot_fit_apply(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Plot the goodness of fit for each run in the trial."""

    if proj.params.do_plot:
        import matplotlib
        from matplotlib import pyplot as plt

        matplotlib.use("QtAgg")

        df.apply(axis="columns", func=plot_fit_ser, proj=proj, trial=trial, plt=plt)
        plt.show()

    return df


def get_heat_transfer(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Calculate heat transfer and superheat based on one-dimensional approximation."""

    # Constants
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # ((cm/m)^2) Conversion factor
    diameter = 0.009525 * cm_p_m  # (cm) 3/8"
    cross_sectional_area = np.pi / 4 * diameter**2  # (cm^2)

    return df.assign(
        **{
            A.dT_dx_err: lambda df: (4 * df["stderr"]).abs(),
            A.k: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(
                    df[TEMPS_TO_REGRESS].mean(axis="columns"), "C", "K"
                ),
            ),
            # no negative due to reversed x-coordinate
            A.q: lambda df: df[A.k] * df[A.dT_dx] / cm2_p_m2,
            A.q_err: lambda df: (df[A.k] * df[A.dT_dx_err]).abs() / cm2_p_m2,
            A.Q: lambda df: df[A.q] * cross_sectional_area,
            A.DT: lambda df: (df[A.TLfit] - df[WATER_TEMPS].mean().mean()),
            A.DT_err: lambda df: (4 * df[A.intercept_stderr]).abs(),
        }
    )


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
            x=trial.thermocouple_pos,
            repeats_per_pair=proj.params.records_to_average,
            regression_stats=result_cols,
        ),  # type: ignore
    )


# * -------------------------------------------------------------------------------- * #
# * POST-PROCESSING

SUPERSCRIPT = re.compile(r"\^(.*)")
SUPERSCRIPT_REPL = r"\+(\1)"
SUBSCRIPT = re.compile(r"\_(.*)")
SUBSCRIPT_REPL = r"\-(\1)"


def transform_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels.

    Explicitly set all dtypes to string to avoid data rendering issues, especially with
    dates.

    Convert super/subscripts in units to their OriginLab representation.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """

    cols = proj.axes.get_col_index()
    quantity = cols.get_level_values("quantity").map(
        {col.name: col.pretty_name for col in proj.axes.all}
    )
    units = cols.get_level_values("units")
    indices = [
        index.to_series()
        .reset_index(drop=True)
        .replace(SUPERSCRIPT, SUPERSCRIPT_REPL)
        .replace(SUBSCRIPT, SUBSCRIPT_REPL)
        for index in (quantity, units)
    ]
    cols = pd.MultiIndex.from_frame(pd.concat(indices, axis="columns"))

    return df.set_axis(cols, axis="columns").reset_index()  # type: ignore


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS OPERATING ON SERIES


def linregress_ser(
    series_of_y: pd.Series,
    x: npt.ArrayLike,
    repeats_per_pair: int,
    regression_stats: list[str],
) -> pd.Series:
    """Perform linear regression of a series of y's with respect to given x's.

    Given x-values and a series of y-values, return a series of linear regression
    statistics.
    """
    # Assume the ordered pairs are repeated with zero standard deviation in x and y
    x = np.repeat(x, repeats_per_pair)
    y = np.repeat(series_of_y, repeats_per_pair)
    r = linregress(x, y)

    # Unpacking would drop r.intercept_stderr, so we have to do it this way.
    # See "Notes" section of SciPy documentation for more info.
    return pd.Series(
        [r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr],
        index=regression_stats,
    )


def plot_fit_ser(
    ser: pd.Series,
    proj: Project,
    trial: Trial,
    temps_to_regress: list[str],
    plt: ModuleType,
):
    """Plot the goodness of fit for a series of temperatures and positions."""
    plt.figure()
    plt.title("Temperature Profile in Post")
    plt.xlabel("x (m)")
    plt.ylabel("T (C)")
    plt.plot(
        trial.thermocouple_pos,
        ser[temps_to_regress],  # type: ignore
        "*",
        label="Measured Temperatures",
        color=[0.2, 0.2, 0.2],
    )
    x_smooth = np.linspace(0, trial.thermocouple_pos[-1])
    plt.plot(
        x_smooth,
        ser[A.TLfit] + ser[A.dT_dx] * x_smooth,
        "--",
        label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
    )
    plt.plot(
        0, ser[A.TLfit], "x", label="Extrapolated Surface Temperature", color=[1, 0, 0]
    )
    plt.legend()


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    pipeline(get_project())
