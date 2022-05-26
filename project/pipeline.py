"""Pipeline functions."""

import json
from pathlib import Path
import re
from types import ModuleType

from numpy import typing as npt
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from scipy.constants import convert_temperature
from scipy.stats import linregress

from config.columns import Columns as C  # noqa: N817
from models import Project
from utils import get_project

# * -------------------------------------------------------------------------------- * #
# * MAIN

pd.options.mode.string_storage = "pyarrow"

UNITS_INDEX = "units"


def pipeline(proj: Project):

    # Column names
    temps_to_regress = [C.T_1, C.T_2, C.T_3, C.T_4, C.T_5]
    water_temps = [C.T_w1, C.T_w2, C.T_w3]

    # Reduce data from CSV's of runs within trials, to single df w/ trials as records
    dfs = [
        get_steady_state(trial.path, proj)  # Reduce many CSV's to one df
        .pipe(rename_columns, proj)  # Pull units out of columns for cleaner ref
        .pipe(fit, proj, temps_to_regress)
        .pipe(plot_fit_apply, proj, temps_to_regress)
        .pipe(get_heat_transfer, temps_to_regress, water_temps)
        .assign(**json.loads(trial.json()))
        for trial in proj.trials
        if trial.monotonic
    ]

    # Post-process the dataframe for writing to OriginLab-flavored CSV
    (
        pd.concat(dfs)
        .pipe(set_units_row_for_originlab, proj)
        .pipe(transform_units_for_originlab)
        .pipe(prettify_for_originlab, proj)
        .to_csv(proj.dirs.results_file, index_label=proj.get_index().name)
    )


# * -------------------------------------------------------------------------------- * #
# * PER-TRIAL STAGES


def get_steady_state(path: Path, proj: Project) -> pd.DataFrame:
    """Get steady-state values for the trial."""

    files: list[Path] = sorted(path.glob("*.csv"))
    run_names: list[str] = [file.stem for file in files]
    runs: list[pd.DataFrame] = []
    for file in files:
        run = get_run(file, proj)
        if len(run) < proj.params.records_to_average:
            raise ValueError(
                f"There are not enough records in {file.name} to compute steady-state."
            )
        runs.append(run)

    runs_steady_state: list[pd.Series] = [
        df.iloc[-proj.params.records_to_average :, :].mean() for df in runs
    ]
    return pd.DataFrame(runs_steady_state, index=run_names).rename_axis(
        proj.get_index().name, axis="index"
    )


# * -------------------------------------------------------------------------------- * #
# * PRIMARY STAGES


def rename_columns(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Remove units from column labels."""

    return df.rename(
        {col.source: name for name, col in proj.cols.items() if not col.index},
        axis="columns",
    )


def fit(df: pd.DataFrame, proj: Project, temps_to_regress: list[str]) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Main pipeline
    df = df.pipe(
        linregress_apply,
        proj,
        df[temps_to_regress],
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


def plot_fit_apply(
    df: pd.DataFrame, proj: Project, temps_to_regress: list[str]
) -> pd.DataFrame:
    """Plot the goodness of fit for each run in the trial."""

    if proj.fit.do_plot:
        import matplotlib
        from matplotlib import pyplot as plt

        matplotlib.use("QtAgg")

        df.apply(plot_fit_ser, args=(proj, temps_to_regress, plt), axis="columns")
        plt.show()

    return df


def get_heat_transfer(
    df: pd.DataFrame, temps_to_regress: list[str], water_temps: list[str]
) -> pd.DataFrame:
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
                    df[temps_to_regress].mean(axis="columns"), "C", "K"
                ),
            ),
            # no negative due to reversed x-coordinate
            C.q: lambda df: df[C.k] * df[C.dT_dx] / cm2_p_m2,
            C.q_err: lambda df: (df[C.k] * df[C.dT_dx_err]).abs() / cm2_p_m2,
            C.Q: lambda df: df[C.q] * cross_sectional_area,
            C.DT: lambda df: (df[C.TLfit] - df[water_temps].mean().mean()),
            C.DT_err: lambda df: (4 * df["intercept_stderr"]).abs(),
        }
    )


# * -------------------------------------------------------------------------------- * #
# * FINISHING STAGES


def set_units_row_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels."""
    units_row = pd.DataFrame(
        {
            name: pd.Series(col.units, index=[UNITS_INDEX])
            for name, col in proj.cols.items()
            if not col.index
        }
    )

    return pd.concat([units_row, df])


SUPERSCRIPT = re.compile(r"\^(.*)")
SUPERSCRIPT_REPL = r"\+(\1)"
SUBSCRIPT = re.compile(r"\_(.*)")
SUBSCRIPT_REPL = r"\-(\1)"


def transform_units_for_originlab(df: pd.DataFrame) -> pd.DataFrame:
    """Convert super/subscripts in units to their OriginLab representation.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """
    units = (
        df.loc[UNITS_INDEX, :]
        .replace(SUPERSCRIPT, SUPERSCRIPT_REPL)
        .replace(SUBSCRIPT, SUBSCRIPT_REPL)
    )
    df.loc[UNITS_INDEX, :] = units
    return df


def prettify_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Rename columns with Greek symbols, superscripts, and subscripts transformed.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """
    return df.rename(
        {
            name: replace_for_originlab(col.pretty_name)
            for name, col in proj.cols.items()
        },
        axis="columns",
    )


def replace_for_originlab(text: str) -> str:
    """Transform superscripts and subscripts to their OriginLab representation.

    See: <https://www.originlab.com/doc/en/Origin-Help/Escape-Sequences>
    """
    return SUBSCRIPT.sub(SUBSCRIPT_REPL, SUPERSCRIPT.sub(SUPERSCRIPT_REPL, text))


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_run(file: Path, proj: Project) -> pd.DataFrame:
    """Get an individual run in a trial."""
    source_cols = {name: col for name, col in proj.cols.items() if col.source}
    dtypes = {name: col.dtype for name, col in source_cols.items()}
    return pd.read_csv(
        file,
        usecols=[col.source for col in source_cols.values()],  # pyright: ignore
        index_col=proj.get_index().source,
        dtype=dtypes,
    )


def linregress_apply(
    df: pd.DataFrame,
    proj: Project,
    temperature_cols: pd.DataFrame,
    result_cols: list[str],
) -> pd.DataFrame:
    """Apply linear regression across the temperature columns."""
    return pd.concat(
        axis="columns",
        objs=[
            df,
            temperature_cols.apply(
                axis="columns",
                func=lambda ser: linregress_ser(
                    x=proj.fit.thermocouple_pos,
                    series_of_y=ser,
                    repeats_per_pair=proj.params.records_to_average,
                    regression_stats=result_cols,
                ),
            ),
        ],
    )


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def linregress_ser(
    x: npt.ArrayLike,
    series_of_y: pd.Series,
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
    ser: pd.Series, proj: Project, temps_to_regress: list[str], plt: ModuleType
):
    """Plot the goodness of fit for a series of temperatures and positions."""
    plt.figure()
    plt.title("Temperature Profile in Post")
    plt.xlabel("x (m)")
    plt.ylabel("T (C)")
    plt.plot(
        proj.fit.thermocouple_pos,
        ser[temps_to_regress],  # type: ignore
        "*",
        label="Measured Temperatures",
        color=[0.2, 0.2, 0.2],
    )
    x_smooth = np.linspace(0, proj.fit.thermocouple_pos[-1])
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

if __name__ == "__main__":
    pipeline(get_project())
