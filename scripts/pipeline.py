"""Pipeline functions."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess  # noqa: S404  # only used for hardcoded calls
from dataclasses import asdict
from time import sleep
from typing import Any

import numpy as np
from numpy import typing as npt
import pandas as pd
from dynaconf import Dynaconf
from pydantic import DirectoryPath, FilePath, validator
from pydantic.dataclasses import dataclass
from scipy.stats import linregress

# * -------------------------------------------------------------------------------- * #
# * VALIDATION


@dataclass
class Paths:
    data: DirectoryPath
    ees: FilePath
    ees_workdir: DirectoryPath

    @validator("ees")
    def validate_ees(cls, ees):
        if ees.name != "EES.exe":
            raise ValueError("Filename must be 'EES.exe'.")
        return ees

    @validator("ees_workdir")
    def validate_ees_workdir(cls, ees_workdir):
        # TODO: Look for "in.dat", "out.dat", and "get_thermal_conductivity.ees"
        return ees_workdir


@dataclass
class FitParams:
    x: list[float]
    T_p_str: list[str]
    material: str
    T_b_str: str
    T_L_str: str
    L: float
    D: float
    do_plot: bool


@dataclass
class GetSuperheatParams:
    T_b_str: str
    T_L_str: str
    T_w_str: list[str]
    material: str
    L: float


params = Dynaconf(settings_files=["scripts/parameters.yaml"])
paths = asdict(Paths(**params.paths))
fit_params = asdict(FitParams(**params.fit))
get_superheat_params = asdict(GetSuperheatParams(**params.get_superheat))


# * -------------------------------------------------------------------------------- * #
# * MAIN


def main():

    data = paths["data"]
    files = sorted((data / "raw").glob("*.csv"))
    stems = [file.stem for file in files]
    dfs = [pd.read_csv(file, index_col=0) for file in files]
    srs = [df.iloc[-80:].mean() for df in dfs]
    df = pd.concat(srs, keys=stems, axis="columns").T
    df = df.pipe(fit, **fit_params).pipe(get_superheat, **get_superheat_params)
    df.to_csv(data / "fitted.csv", index_label="From Dataset")


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


class StrictDict(dict[Any, Any]):
    """Dict that doesn't allow new keys to be set after construction."""

    def __setitem__(self, key: Any, value: Any):
        if key not in self:
            raise KeyError(f'Cannot add "{key}" key to StrictDict with immutable keys.')
        dict.__setitem__(self, key, value)


def fit(
    df: pd.DataFrame | pd.Series,
    x: float,
    T_p_str: list[str],  # noqa: N803
    material: str,
    L: float,
    D: float,
    T_b_str: str,
    T_L_str: str,
    wait: float = 7,
    do_plot: bool = False,
):
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # ! Inputs
    # get temperatures along the post as a numpy array
    T_p_arr: npt.NDArray[np.floating] = df.loc[:, T_p_str].values  # noqa: N806
    # get average post temperature for each run, for property estimation
    T_p_avg_arr: npt.NDArray[np.floating] = (  # noqa: N806
        df.loc[:, T_p_str].mean(axis="columns").values
    )
    # post geometry
    A = np.pi / 4 * D**2  # noqa: N806

    # ! Property Lookup
    k_arr = get_thermal_conductivity(material, T_p_avg_arr)

    # keys for a results dict that will become a DataFrame
    keys = [
        T_b_str,
        T_L_str,
        "Q (W)",
        "q (W/m^2)",
        "rval",
        "pval",
        "stderr",
        "k (W/m-K)",
    ]
    # preallocate numpy arrays
    results = {key: np.full_like(k_arr, np.nan) for key in keys}

    j = StrictDict(dict.fromkeys(keys))  # ensure key order for later mapping

    # perform a curve fit for each experimental run
    for i, (T_p, k) in enumerate(zip(T_p_arr, k_arr)):  # noqa: N806

        # ! Fit Assuming 1D Conduction
        # linear regression of the temperature profile
        (  # noqa: N806
            dTdx,
            j[T_b_str],
            j["rval"],
            j["pval"],
            j["stderr"],
        ) = linregress(
            x, T_p
        )  # noqa: N806
        # ? q and DT
        j[T_L_str] = j[T_b_str] + dTdx * L
        j["Q (W)"] = -k * A * dTdx
        j["q (W/m^2)"] = -k * dTdx

        # ! Record k
        j["k (W/m-K)"] = k

        # ! Plot
        if do_plot:
            from matplotlib import pyplot as plt

            x_plt = np.linspace(0, L)
            plt.plot(x_plt, j[T_b_str] + dTdx * x_plt, "--", label="1D, SS Cond.")
            plt.plot(x, T_p, "*", color=[0.25, 0.25, 0.25], label="Exp. Data")
            # ? Plot Setup
            plt.title("1D-Conduction Fit to Data")
            plt.xlabel("x (m)")
            plt.ylabel("T (C)")
            plt.legend()
            plt.draw()

        # map results from this iteration to the overall results dict
        for key, value in j.items():
            results[key][i] = value
    # ! Generate Results DataFrame
    df_results = pd.DataFrame(index=df.index, data=results)

    # concatenate original DataFrame with results
    df = pd.concat([df, df_results], axis="columns")

    return df


def get_superheat(
    df: pd.DataFrame,
    T_b_str: str,  # noqa: N803
    T_L_str: str,
    T_w_str: list[str],
    material: str,
    L: float,
    wait: float = 7,
) -> pd.DataFrame:
    """Fit the data assuming one-dimensional, steady-state conduction."""

    # ! Inputs
    # get temperatures along the post as a numpy array
    T_b_arr = df.loc[:, T_b_str].values  # noqa: N806
    # get average water temperature, part of superheat calculation
    T_w_avg = np.mean(df.loc[:, T_w_str].mean(axis="columns").values)  # noqa: N806
    # post geometry

    # ! Property Lookup
    k_arr = get_thermal_conductivity(material, T_b_arr)

    k_str = f"k_{material} (W/m-K)"

    # keys for a results dict that will become a DataFrame
    keys = [
        T_L_str,
        "DT (C)",
        k_str,
    ]
    # preallocate numpy arrays
    results = {key: np.full_like(k_arr, np.nan) for key in keys}

    # prepare an index for mapping results
    j = StrictDict(dict.fromkeys(keys))  # ensure key order for later mapping

    # get the superheat for each experimental run
    for i, (T_b, k) in enumerate(zip(T_b_arr, k_arr)):  # noqa: N806
        # ? q and DT
        dTdx = -df.at[df.index[i], "q (W/m^2)"] / k  # noqa: N806
        j[T_L_str] = T_b + dTdx * L
        j["DT (C)"] = j[T_L_str] - T_w_avg

        # ! Record k
        j[k_str] = k

        # map results from this iteration to the overall results dict
        for key, value in j.items():
            results[key][i] = value

    # ! Generate Results DataFrame
    df_results = pd.DataFrame(index=df.index, data=results)

    # concatenate original DataFrame with results
    df = pd.concat([df, df_results], axis="columns")

    return df


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_thermal_conductivity(
    material: str, temperatures: npt.NDArray[np.floating], wait: float = 7
):
    """Get thermal conductivity."""

    @contextmanager
    def change_directory(path: str):
        """Context manager for changing working directory."""
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    with change_directory(paths["ees_workdir"]):

        # write post material, number of runs, and average post temperatures to in.dat
        with open("in.dat", "w+") as f:
            print(material, len(temperatures), *temperatures, file=f)
        # Invoke EES to write thermal conductivities to out.dat given contents of in.dat
        subprocess.Popen(  # noqa: S603, S607  # hardcoded
            [
                "pwsh",
                "-Command",
                f"{paths['ees']}",
                f"{Path('get_thermal_conductivity.ees').resolve()}",
                "/solve",
            ]
        )
        sleep(wait)  # Wait long enough for EES to finish
        # EES should have written to out.dat
        with open("out.dat", "r") as f:
            k_str = f.read().split("\t")
            thermal_conductivity: npt.NDArray[np.floating] = np.array(
                k_str, dtype=np.float64
            )

    return thermal_conductivity


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main()
