"""Pipeline functions."""

import json
from pathlib import Path
import re

from numpy import typing as npt
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from scipy.constants import convert_temperature
from scipy.stats import linregress

from boilerdata.utils import load_config
from columns import Columns as C  # noqa: N817
from models import Project


def get_project():
    return load_config("project/config/project.yaml", Project)


def main(project: Project):

    POINTS_TO_AVERAGE = 60  # noqa: N806

    dfs: list[pd.DataFrame] = []
    for trial in project.trials:
        if trial.monotonic:
            df = (
                get_steady_state(trial.path, POINTS_TO_AVERAGE)
                .pipe(rename_columns, project)
                .pipe(run_one, project, POINTS_TO_AVERAGE)
                .assign(**json.loads(trial.json()))  # Assign trial metadata
            )
            dfs.append(df)
    pd.concat(dfs).pipe(set_units_row, project).pipe(
        transform_units_for_originlab
    ).pipe(prettify, project).to_csv(project.dirs.results_file, index_label="Run")


def get_steady_state(path: Path, points_to_average: int) -> pd.DataFrame:
    """Get steady-state values for the run."""
    files: list[Path] = sorted(path.glob("*.csv"))
    run_names: list[str] = [file.stem for file in files]
    runs_full: list[pd.DataFrame] = [pd.read_csv(file, index_col=0) for file in files]
    runs_steady_state: list[pd.Series] = [
        df.iloc[-points_to_average:, :].mean() for df in runs_full
    ]
    return pd.DataFrame(runs_steady_state, index=run_names)


def rename_columns(df: pd.DataFrame, project: Project) -> pd.DataFrame:
    """Remove units from column labels."""
    columns_mapping = dict(zip(df.columns, project.columns.keys()))
    return df.rename(columns_mapping, axis="columns")


def run_one(df: pd.DataFrame, project: Project, points_to_average: int) -> pd.DataFrame:
    return df.pipe(fit, project, points_to_average)


def set_units_row(df: pd.DataFrame, project: Project) -> pd.DataFrame:
    """Move units out of column labels and into a row just below the column labels."""
    units_row = pd.DataFrame(
        {
            name: pd.Series(column.units, index=["Units"])
            # Don't simplify to "columns.items()" because df.columns are prettified
            for name, column in zip(df.columns, project.columns.values())
        }
    )

    return pd.concat([units_row, df])


def transform_units_for_originlab(df: pd.DataFrame) -> pd.DataFrame:
    units = df.loc["Units", :].replace(re.compile(r"\^(\d)"), r"\+(\1)")
    df.loc["Units", :] = units
    return df


def prettify(df: pd.DataFrame, project: Project) -> pd.DataFrame:
    return df.rename(
        {
            "DT": project.columns["DT"].pretty_name,
            "DT_err": project.columns["DT_err"].pretty_name,
        },
        axis="columns",
    )


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


def fit(
    df: pd.DataFrame,
    project: Project,
    points_to_average: int,
):
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Constants
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # (cm/m)^2 Conversion factor
    diameter = 0.009525 * cm_p_m  # (cm) 3/8"

    # Column names
    temps_to_regress = [C.T1cal, C.T2cal, C.T3cal, C.T4cal, C.T5cal]
    water_temps = [C.Tw1cal, C.Tw2cal, C.Tw3cal]

    # Computed values
    temperature_cols: pd.DataFrame = df.loc[:, temps_to_regress]
    water_temp_cols: pd.DataFrame = df.loc[:, water_temps]
    cross_sectional_area = np.pi / 4 * diameter**2

    # Main pipeline
    df = df.pipe(
        lambda df: pd.concat(  # Compute regression stats for each run
            axis="columns",
            objs=[
                df,
                temperature_cols.apply(
                    axis="columns",
                    func=lambda ser: linregress_series(
                        project.fit.thermocouple_pos,
                        ser,
                        points_to_average,
                        (C.dT_dx, C.TLfit),
                    ),
                ),
            ],
        )
    ).assign(
        **{
            C.dT_dx_err: lambda df: (4 * df["stderr"]).abs(),
            C.k: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(temperature_cols.mean(axis="columns"), "C", "K"),
            ),
            # no negative due to reversed x-coordinate
            C.q: lambda df: df[C.k] * df[C.dT_dx] / cm2_p_m2,
            C.q_err: lambda df: (df[C.k] * df[C.dT_dx_err]).abs() / cm2_p_m2,
            C.Q: lambda df: df[C.q] * cross_sectional_area,
            C.DT: lambda df: (df[C.TLfit] - water_temp_cols.mean().mean()),
            C.DT_err: lambda df: (4 * df["intercept_stderr"]).abs(),
        }
    )

    # Plotting
    if project.fit.do_plot:

        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib import pyplot as plt

        def plot(ser):
            plt.figure()
            plt.title("Temperature Profile in Post")
            plt.xlabel("x (m)")
            plt.ylabel("T (C)")
            plt.plot(
                project.fit.thermocouple_pos,
                ser.loc[temps_to_regress],
                "*",
                label="Measured Temperatures",
                color=[0.2, 0.2, 0.2],
            )
            x_smooth = np.linspace(thermocouple_pos[0], thermocouple_pos[-1])  # type: ignore
            plt.plot(
                x_smooth,
                ser[C.TLfit] + ser[C.dT_dx] * x_smooth,
                "--",
                label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
            )
            plt.plot(
                0,
                ser[C.TLfit],
                "x",
                label="Extrapolated Surface Temperature",
                color=[1, 0, 0],
            )
            plt.legend()

        df.apply(plot, axis="columns")
        plt.show()

    return df


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def linregress_series(
    x: npt.ArrayLike,
    series_of_y: pd.Series,
    repeats_per_pair: int,
    label: tuple[str, str] = ("slope", "intercept"),
    prefix: str = "",
) -> pd.Series:
    """Perform linear regression of a series of y's with respect to given x's.

    Given x-values and a series of y-values, return a series of linear regression
    statistics.
    """
    labels = [*label, "rvalue", "pvalue", "stderr", "intercept_stderr"]
    if prefix:
        labels = [*label, *("_".join(label) for label in labels[-4:])]

    # Assume the ordered pairs are repeated with zero standard deviation in x and y
    x = np.repeat(x, repeats_per_pair)
    y = np.repeat(series_of_y, repeats_per_pair)
    r = linregress(x, y)

    # Unpacking would drop r.intercept_stderr, so we have to do it this way.
    # See "Notes" section of SciPy documentation for more info.
    return pd.Series(
        [r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr],
        index=labels,
    )


if __name__ == "__main__":
    main(get_project())
