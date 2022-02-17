"""Pipeline functions."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from dynaconf import Dynaconf
from numpy import typing as npt
from propshop import get_prop
from propshop.library import Mat, Prop
from pydantic import DirectoryPath, validator
from pydantic.dataclasses import dataclass
from scipy.constants import convert_temperature
from scipy.stats import linregress

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main():
    config = configure()
    data: Path = config.data_path  # type: ignore
    files: list[Path] = sorted((data / "raw").glob("*.csv"))
    run_names: list[str] = [file.stem for file in files]
    runs_full: list[pd.DataFrame] = [pd.read_csv(file, index_col=0) for file in files]
    runs_steady_state: list[pd.Series] = [df.iloc[-80:, :].mean() for df in runs_full]
    df = pd.DataFrame(runs_steady_state, index=run_names).pipe(
        fit, **config.fit_params  # type: ignore
    )
    df.to_csv(data / "fitted.csv", index_label="Run")


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


def fit(
    df: pd.DataFrame,
    x: float,
    T_p_str: list[str],  # noqa: N803
    material: str,
    k: str,
    T_b_str: str,
    T_L_str: str,
    slope: str,
    L: float,
    D: float,
    do_plot: bool = False,
):
    """Fit the data assuming one-dimensional, steady-state conduction."""

    temperature_cols = df.loc[:, T_p_str]
    cross_sectional_area = np.pi / 4 * D**2

    df = df.pipe(
        lambda df: pd.concat(
            axis="columns",
            objs=[
                df,
                temperature_cols.apply(
                    axis="columns",
                    func=lambda ser_: linregress_series(x, ser_, (slope, T_b_str)),
                ),
            ],
        )
    ).assign(
        **{
            "k (W/m-K)": get_prop(
                Mat[material],
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(temperature_cols.mean(axis="columns"), "C", "K"),
            ),
            T_L_str: lambda df: df[T_b_str] + df[slope] * L,
            "q (W/m^2)": lambda df: -df[k] * df[slope],
            "q_err (W/m^2)": lambda df: (df[k] * 4 * df["stderr"]).abs(),
            "Q (W)": lambda df: df["q (W/m^2)"] * cross_sectional_area,
        }
    )

    if do_plot:

        from matplotlib import pyplot as plt

        def plot(ser):
            plt.figure()
            plt.title(f"Temperature Profile in {material.title()} Post")
            plt.xlabel("x (m)")
            plt.ylabel("T (C)")
            plt.plot(
                x,
                ser.loc[T_p_str],
                "*",
                label="Measured Temperatures",
                color=[0.2, 0.2, 0.2],
            )
            x_smooth = np.linspace(0, L)
            plt.plot(
                x_smooth,
                ser[T_b_str] + ser[slope] * x_smooth,
                "--",
                label=f"Linear Regression $(r^2={round(ser.rvalue**2,4)})$",
            )
            plt.plot(
                L,
                ser[T_L_str],
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

    # Unpacking would drop r.intercept_stderr, so we have to do it this way.
    # See "Notes" section of SciPy documentation for more info.
    r = linregress(x, series_of_y)
    return pd.Series(
        [r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr],
        index=labels,
    )


# * -------------------------------------------------------------------------------- * #
# * CONFIGURE


def configure():
    @dataclass
    class FitParams:
        x: list[float]
        T_p_str: list[str]
        material: str
        k: str
        T_b_str: str
        T_L_str: str
        slope: str
        L: float
        D: float
        do_plot: bool

        _ = validator("material", allow_reuse=True)(lambda string: string.upper())

    @dataclass
    class Config:
        data_path: DirectoryPath
        fit_params: FitParams

        @validator("fit_params")
        def _(cls, param):
            return asdict(param)

    raw_config = Dynaconf(settings_files=["examples/config.yaml"])
    return Config(
        data_path=raw_config.data_path, fit_params=FitParams(**raw_config.fit)
    )


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main()
