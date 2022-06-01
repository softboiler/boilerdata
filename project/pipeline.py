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

from columns import Columns as C  # noqa: N817
from constants import TEMPS_TO_REGRESS, WATER_TEMPS
from models import Project, Trial
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

    runs_df = validate_runs_df(get_df(proj))

    cols = [
        col for col in proj.cols if col not in [col.name for col in proj.get_index()]
    ]

    df = pd.DataFrame(columns=cols).assign(**get_steady_state(runs_df, proj))  # type: ignore

    for trial in proj.trials:
        df.update(
            df.xs(trial.trial, level=C.trial, drop_level=False)
            .pipe(fit, proj, trial)
            .pipe(get_heat_transfer, proj, trial)
        )

    # Set dtypes after update. https://github.com/pandas-dev/pandas/issues/4094
    dtypes = {name: col.dtype for name, col in proj.cols.items() if name in cols}
    df = validate_df(df.pipe(set_dtypes, dtypes))

    # Post-process the dataframe for writing to OriginLab-flavored CSV
    df.pipe(transform_for_originlab, proj).to_csv(
        proj.dirs.results_file,
        index=False,
        encoding="utf-8",
    )
    proj.dirs.coldes_file.write_text(proj.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * GET DATAFRAME OF ALL RUNS AND TIMES


def get_df(proj: Project) -> pd.DataFrame:
    """Get the dataframe of all runs."""

    dtypes = {
        name: col.dtype
        for name, col in proj.cols.items()
        if col.meta or col.source and not col.index
    }

    # Fetch all runs from their original CSVs if needed.
    if any(trial.new for trial in proj.trials) or proj.params.refetch_runs:
        return get_runs(proj, dtypes)

    # Otherwise, reload the dataframe last written by `get_runs()`
    index_cols = list(range(len(proj.get_index())))
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
                tuple(
                    (*run_index, datetime.fromisoformat(record_time))
                    for record_time in run.index
                )
            )

    # Concatenate results from all runs and set the multiindex
    df = pd.concat(runs).set_index(
        pd.MultiIndex.from_tuples(
            multiindex, names=[idx.name for idx in proj.get_index()]
        )
    )

    # Ensure appropriate dtypes for each column. Validate number of records in each run.
    df = set_dtypes(df, dtypes)

    # Write the runs to disk for faster fetching later
    df.to_csv(proj.dirs.runs_file, encoding="utf-8")

    return df


def get_run(proj: Project, trial: Trial, run: Path) -> pd.DataFrame:

    # Get metadata
    indices = proj.get_index()
    meta_cols = {name: col for name, col in proj.cols.items() if col.meta}
    metadata = {
        k: v for k, v in trial.dict().items() if k not in [col.name for col in indices]
    }

    # Get source columns
    index = proj.get_index()[-1].source
    src_cols = {
        col.source: col for col in proj.cols.values() if col.source and not col.index
    }
    src_dtypes = {name: col.dtype for name, col in src_cols.items()}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    return (
        pd.DataFrame(columns=(meta_cols | src_cols).keys())
        .assign(
            **pd.read_csv(
                run,
                usecols=[index, *src_cols.keys()],  # type: ignore
                index_col=index,
                dtype=src_dtypes,
                encoding="utf-8",
            ),
            **metadata,
        )
        .dropna()  # Rarely a run has a NaN record at the end
        .tail(proj.params.records_to_average)
        .pipe(rename_columns, proj)
    )


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units into a MultiIndex."""

    return df.rename(
        {col.source: name for name, col in proj.cols.items() if not col.index},
        axis="columns",
    )


# * -------------------------------------------------------------------------------- * #
# * GET REDUCED DATA FOR EACH RUN


def get_steady_state(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Get the steady-state values for each run."""
    cols_to_mean = [
        name for name, col in proj.cols.items() if col.source and col.dtype == "float"
    ]
    means = df[cols_to_mean].groupby(level=C.run, sort=False).transform("mean")
    return df.assign(**means).droplevel(C.time)[:: proj.params.records_to_average]  # type: ignore


def fit(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Main pipeline
    df = df.pipe(
        linregress_apply,
        proj=proj,
        trial=trial,
        temperature_cols=df[TEMPS_TO_REGRESS],
        result_cols=[
            C.dT_dx,
            C.TLfit,
            C.rvalue,
            C.pvalue,
            C.stderr,
            C.intercept_stderr,
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
            C.dT_dx_err: lambda df: (4 * df["stderr"]).abs(),
            C.k: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(
                    df[TEMPS_TO_REGRESS].mean(axis="columns"), "C", "K"
                ),
            ),
            # no negative due to reversed x-coordinate
            C.q: lambda df: df[C.k] * df[C.dT_dx] / cm2_p_m2,
            C.q_err: lambda df: (df[C.k] * df[C.dT_dx_err]).abs() / cm2_p_m2,
            C.Q: lambda df: df[C.q] * cross_sectional_area,
            C.DT: lambda df: (df[C.TLfit] - df[WATER_TEMPS].mean().mean()),
            C.DT_err: lambda df: (4 * df[C.intercept_stderr]).abs(),
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

    columns = proj.get_col_index()
    quantity = columns.get_level_values("quantity").map(
        {name: col.pretty_name for name, col in proj.cols.items()}
    )
    units = columns.get_level_values("units")
    indices = [
        index.to_series()
        .reset_index(drop=True)
        .replace(SUPERSCRIPT, SUPERSCRIPT_REPL)
        .replace(SUBSCRIPT, SUBSCRIPT_REPL)
        for index in (quantity, units)
    ]
    columns = pd.MultiIndex.from_frame(pd.concat(indices, axis="columns"))

    return df.set_axis(columns, axis="columns").reset_index()  # type: ignore


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
        ser[C.TLfit] + ser[C.dT_dx] * x_smooth,
        "--",
        label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
    )
    plt.plot(
        0, ser[C.TLfit], "x", label="Extrapolated Surface Temperature", color=[1, 0, 0]
    )
    plt.legend()


# * -------------------------------------------------------------------------------- * #
# * OTHER HELPER FUNCTIONS


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    return df.assign(**{name: df[name].astype(dtype) for name, dtype in dtypes.items()})


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    pipeline(get_project())
