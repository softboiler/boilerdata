"""Pipeline functions."""

import json
from pathlib import Path

from numpy import typing as npt
import numpy as np
import pandas as pd
from propshop import get_prop
from propshop.library import Mat, Prop
from scipy.constants import convert_temperature
from scipy.stats import linregress

from boilerdata.utils import load_config
from models import Fit, Project, Trials

PROJECT = load_config("project/config/project.yaml", Project)
TRIALS = load_config("project/config/trials.yaml", Trials)


def run(project: Project = PROJECT, trials: Trials = TRIALS):
    dfs: list[pd.DataFrame] = []
    for trial in trials.trials:
        if trial.monotonic:
            df = run_one(trial.get_path(project), project.fit).assign(
                **json.loads(trial.json())
            )
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(project.base / "results.csv", index_label="Run")


def run_one(path: Path, fit_params: Fit) -> pd.DataFrame:

    points_to_average = 60
    files: list[Path] = sorted(path.glob("*.csv"))
    run_names: list[str] = [file.stem for file in files]
    runs_full: list[pd.DataFrame] = [pd.read_csv(file, index_col=0) for file in files]
    runs_steady_state: list[pd.Series] = [
        df.iloc[-points_to_average:, :].mean() for df in runs_full
    ]
    df: pd.DataFrame = pd.DataFrame(runs_steady_state, index=run_names).pipe(
        fit, **fit_params.dict(), points_averaged=points_to_average
    )
    return df


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


def fit(
    df: pd.DataFrame,
    thermocouple_pos: npt.ArrayLike,
    do_plot: bool,
    points_averaged: int,
):
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # Constants
    diameter = 0.009525  # (m) 3/8"
    cm_p_m = 100  # (cm/m) Conversion factor
    cm2_p_m2 = cm_p_m**2  # (cm/m)^2 Conversion factor

    # Column names
    slope = "dT/dx (K/m)"  # slope
    extrap_surf_temp = "TLfit (C)"
    slope_err = "dT/dx_err (K/m)"  # total magnitude of the error bar for slope
    k = "k (W/m-K)"  # thermal conductivity
    q = "q (W/m^2)"  # heat flux
    q_err = "q_err (W/m^2)"  # total magnitude of the error bar for heat flux
    inter_err = "∆T_err (K)"  # total magnitude of the error bar for slope
    temps_to_regress = ["T1cal (C)", "T2cal (C)", "T3cal (C)", "T4cal (C)", "T5cal (C)"]
    water_temps = ["Tw1cal (C)", "Tw2cal (C)", "Tw3cal (C)"]

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
                        thermocouple_pos,
                        ser,
                        points_averaged,
                        (slope, extrap_surf_temp),
                    ),
                ),
            ],
        )
    ).assign(
        **{
            slope_err: lambda df: (4 * df["stderr"]).abs(),
            k: get_prop(
                Mat.COPPER,
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(temperature_cols.mean(axis="columns"), "C", "K"),
            ),
            q: lambda df: df[k] * df[slope],  # no negative due to reversed x-coordinate
            q_err: lambda df: (df[k] * df[slope_err]).abs(),
            "Q (W)": lambda df: df[q] * cross_sectional_area,
            # "∆T (K)": lambda df: (
            #     df[extrap_surf_temp] - water_temp_cols.mean(axis="columns")
            # ),
            "∆T (K)": lambda df: (df[extrap_surf_temp] - water_temp_cols.mean().mean()),
            inter_err: lambda df: (4 * df["intercept_stderr"]).abs(),
            "q (W/cm^2)": lambda df: df[q] / cm2_p_m2,
            "q_err (W/cm^2)": lambda df: df[q_err] / cm2_p_m2,
        }
    )

    # Plotting
    if do_plot:

        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib import pyplot as plt

        def plot(ser):
            plt.figure()
            plt.title("Temperature Profile in Post")
            plt.xlabel("x (m)")
            plt.ylabel("T (C)")
            plt.plot(
                thermocouple_pos,
                ser.loc[temps_to_regress],
                "*",
                label="Measured Temperatures",
                color=[0.2, 0.2, 0.2],
            )
            x_smooth = np.linspace(thermocouple_pos[0], thermocouple_pos[-1])  # type: ignore
            plt.plot(
                x_smooth,
                ser[extrap_surf_temp] + ser[slope] * x_smooth,
                "--",
                label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
            )
            plt.plot(
                0,
                ser[extrap_surf_temp],
                "x",
                label="Extrapolated Surface Temperature",
                color=[1, 0, 0],
            )
            plt.legend()

        df.apply(plot, axis="columns")
        plt.show()

    # # Move units out of column labels and into a row just below the column labels
    # labels = (get_label(label) for label in df.columns)
    # units = (get_units(label) for label in df.columns)
    # columns_mapping = {old: new for old, new in zip(df.columns, labels)}
    # units_row = pd.DataFrame(
    #     {
    #         label: pd.Series(unit, index=["Units"])
    #         for label, unit in zip(df.columns, units)
    #     }
    # )
    # df = pd.concat([units_row, df]).rename(columns=columns_mapping)

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


def get_label(label: str) -> str:
    return label.split()[0]


def get_units(label: str) -> str:
    match label.split():
        case label, units:
            return units.strip("()")
        case _:
            return ""


if __name__ == "__main__":
    run()
