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
from constants import TEMPS_TO_REGRESS, UNITS_INDEX, WATER_TEMPS
from df_schema import df_schema
from enums import PandasDtype
from models import Project, Trial
from utils import get_project

u = UnitRegistry()
Q = u.Quantity
u.load_definitions(Path("project/config/units.txt"))
PintType.ureg = u

# * -------------------------------------------------------------------------------- * #
# * MAIN


def pipeline(proj: Project):

    df = get_df(proj)

    # Reduce data from CSV's of runs within trials, to single df w/ trials as records
    # dfs = [
    #     get_run(proj, trial)
    #     .pipe(get_steady_state, proj, trial)  # Reduce many CSV's to one df
    #     .pipe(rename_columns, proj, trial)  # Pull units out of columns for cleaner ref
    #     .pipe(fit, proj, trial)
    #     .pipe(plot_fit_apply, proj, trial)
    #     .pipe(get_heat_transfer, proj, trial)
    #     for trial in proj.trials
    # ]

    # Validate the dataframe
    df = df_schema(concat_with_dtypes(dfs, proj))

    # Post-process the dataframe for writing to OriginLab-flavored CSV
    df = (
        df.pipe(set_units_row_for_originlab, proj)
        .pipe(transform_units_for_originlab, proj)
        .pipe(prettify_for_originlab, proj)
    )
    df.to_csv(proj.dirs.results_file, index_label=proj.get_index().name)
    proj.dirs.coldes_file.write_text(proj.get_originlab_coldes())


# * -------------------------------------------------------------------------------- * #
# * AGG PER RUN


def get_df(proj: Project) -> pd.DataFrame:
    """Get an individual run in a trial."""

    index = proj.get_index()
    dtypes = {
        name: col.dtype
        for name, col in proj.cols.items()
        if col.meta or col.source and not col.index
    }

    if any(trial.new for trial in proj.trials) or proj.params.refetch_runs:

        run_re = re.compile(r"(?P<date>.*)T(?P<time>.*)")
        runs: list[pd.DataFrame] = []

        multi_index: list[tuple[datetime, datetime, datetime]] = []
        for trial in proj.trials:
            for run in trial.runs:
                run_time = run.stem.removeprefix("results_")

                if m := run_re.match(run_time):
                    run_time = datetime.fromisoformat(
                        f"{m['date']}T{m['time'].replace('-', ':')}"
                    )
                else:
                    raise AttributeError(f"Could not parse run time: {run_time}")

                run = get_run(proj, trial, run)
                runs.append(run)
                multi_index.extend(
                    (
                        datetime.fromisoformat(str(trial.trial)),
                        run_time,
                        datetime.fromisoformat(time),
                    )
                    for time in run.index
                )

        df = pd.concat(runs)
        df.index = pd.MultiIndex.from_tuples(
            multi_index, names=[idx.name for idx in index]
        )
        df = df.assign(**{name: df[name].astype(dtypes[name]) for name in dtypes})
        df.to_csv(proj.dirs.runs_file)
    else:
        index_cols = list(range(len(index)))
        df = pd.read_csv(
            proj.dirs.runs_file,
            index_col=index_cols,
            dtype=dtypes,
            parse_dates=index_cols,
            encoding="utf-8",
        )

    return df


# TODO: Check output for NaN with Pandera like in run "2022-03-29T14-01-41".
def get_run(proj: Project, trial: Trial, run: Path) -> pd.DataFrame:

    # Metadata
    indices = proj.get_index()
    meta_cols = {name: col for name, col in proj.cols.items() if col.meta}
    metadata = {
        k: v for k, v in trial.dict().items() if k not in [col.name for col in indices]
    }

    # Columns
    index = proj.get_index()[-1].source
    src_cols = {
        col.source: col for col in proj.cols.values() if col.source and not col.index
    }

    # Data types
    src_dtypes = {name: col.dtype for name, col in src_cols.items()}

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


# ! TODO: Refactor into parallel pipeline
# # Quantify the columns with units
# df.columns = multi_index  # Set the multi-index so quantify can work
# df = df.pint.quantify()  # Changes column dtypes to unit-aware dtypes
# df.columns = multi_index.droplevel(-1)  # Go back to the simple index


# * -------------------------------------------------------------------------------- * #
# * PIPELINE


def get_steady_state(df: pd.DataFrame, proj: Project, trial: Trial) -> pd.DataFrame:
    """Get steady-state values for the trial."""

    return df.mean()

    # TODO: Can't take the mean like we were able to before.
    runs_steady_state: list[pd.Series] = [
        df.tail(proj.params.records_to_average).mean() for df in runs
    ]  # type: ignore
    return pd.DataFrame(runs_steady_state, index=run_names).rename_axis(
        proj.get_index().name, axis="index"
    )


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
    return pd.concat(
        axis="columns",
        objs=[
            df,
            temperature_cols.apply(
                axis="columns",
                func=linregress_ser,
                x=trial.thermocouple_pos,
                repeats_per_pair=proj.params.records_to_average,
                regression_stats=result_cols,
            ),
        ],
    )


def concat_with_dtypes(dfs: list[pd.DataFrame], proj: Project) -> pd.DataFrame:
    """Concatenate trials and set data types properly."""

    df = pd.concat(dfs)
    sers = (item[1] for item in df.items())
    dtypes = {name: col.dtype for name, col in proj.cols.items() if not col.index}
    return df.assign(
        **{
            name: ser.astype(dtype)
            for name, dtype, ser in zip(dtypes, dtypes.values(), sers)
        }
    ).set_index(df.index.astype(proj.get_index().dtype))


# * -------------------------------------------------------------------------------- * #
# * POST-PROCESSING

SUPERSCRIPT = re.compile(r"\^(.*)")
SUPERSCRIPT_REPL = r"\+(\1)"
SUBSCRIPT = re.compile(r"\_(.*)")
SUBSCRIPT_REPL = r"\-(\1)"


# TODO: This will need to be a bit different now that we put units in a MultiIndex. We
# should instead stuff that MultiIndex into the `project` object and just fetch it here
# to glue onto the dataframe. That way we don't have to carry it around teh pipeline to
# facilitate "Data preview" in VSCode rendering it nicely.
def set_units_row_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels.

    Explicitly set all dtypes to string to avoid data rendering issues, especially with
    dates.
    """
    units_row = pd.DataFrame(
        {
            name: pd.Series(col.units, index=[UNITS_INDEX])
            for name, col in proj.cols.items()
            if not col.index
        },
        dtype=PandasDtype.string,
    )
    return pd.concat([units_row, df.astype(PandasDtype.string)])


def transform_units_for_originlab(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
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

if __name__ == "__main__":
    pipeline(get_project())
