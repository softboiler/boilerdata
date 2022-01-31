"""Pipeline functions."""

import logging
import subprocess
from os import environ
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
from dynaconf import Dynaconf
from scipy.stats import linregress

from pdfittings import tap, tee

ENABLE_FITTINGS = True
logging.basicConfig(level=logging.INFO)
pdobj = pd.DataFrame | pd.Series

parameters = Dynaconf(settings_files=["parameters.yaml"])


def main():
    path = Path(
        "G:/My Drive/Blake/School/Grad/Projects/18.09 Nucleate Pool Boiling/Data/Boiling Curves/22.01.19 Copper X-A6-B3-1/data2"
    )
    files = sorted((path / "raw").glob("*.csv"))
    stems = [file.stem for file in files]
    dfs = [pd.read_csv(file, index_col=0) for file in files]
    srs = [
        df.pipe(take_last, rows=80)
        .pipe(tee, enable=ENABLE_FITTINGS, preview=skip_preview)
        .mean()
        for df in dfs
    ]
    df = pd.concat(srs, keys=stems, axis="columns")
    df = df.T.pipe(fit, **parameters.fit).pipe(
        get_superheat, **parameters.get_superheat
    )
    df.to_csv(path / "fitted.csv", index_label="From Dataset")


def skip_preview(df: pdobj) -> str:
    return ""


# * -------------------------------------------------------------------------------- * #
# * FUNCTIONS


@tap(enable=ENABLE_FITTINGS, preview=skip_preview)
def take_last(df: pdobj, rows: int = 100) -> pdobj:
    """Reduce data to the last 80 data points."""
    df = df.iloc[-rows:]
    return df


def fit(
    df: pd.DataFrame,
    x: float,
    T_p_str: list[str],
    material: str,
    L: float,
    D: float,
    T_b_str: str,
    T_L_str: str,
    wait: float = 7,
    do_plot: bool = False,
) -> pd.DataFrame:
    """
    Fit the data assuming one-dimensional, steady-state conduction.
    """

    # ! Environment Variables
    EESIN = "EESIN"
    EESOUT = "EESOUT"
    EES = "EES"
    EESFILE = "EESFILE"

    # ! Inputs
    # get temperatures along the post as a numpy array
    T_p_arr = df.loc[:, T_p_str].values
    # get average post temperature for each run, for property estimation
    T_p_avg_arr = df.loc[:, T_p_str].mean(axis="columns").values
    # post geometry
    A = np.pi / 4 * D ** 2

    # ! Property Lookup
    # write post material, number of runs, and average post temperatures to IN.DAT
    with open(environ[EESIN], "w+") as f:
        print(material, len(T_p_avg_arr), *T_p_avg_arr, file=f)
    # call on PowerShell to invoke EES to write thermal conductivities to OUT.DAT given
    # the contents of IN.DAT
    subprocess.Popen(
        ["pwsh.exe", "-Command", "& $env:" + EES + " $pwd\\$env:" + EESFILE + " /solve"]
    )
    # Since the subprocess command finishes before EES does, we have to wait long enough
    # for EES to finish its job. We could hook into the process, but waiting is fine.
    sleep(wait)
    # EES should have written to OUT.DAT. get the properties from it
    with open(environ[EESOUT], "r") as f:
        k_str = f.read().split("\t")
        k_arr = np.array(k_str, dtype=np.float64)

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

    class StrictDict(dict):
        """Dict that doesn't allow new keys to be set after construction."""

        def __setitem__(self, key, value):
            if key not in self:
                raise KeyError(
                    f'Cannot add "{key}" key to StrictDict with immutable keys.'
                )
            dict.__setitem__(self, key, value)

    # prepare an index for mapping results
    i = 0
    j = StrictDict(dict.fromkeys(keys))  # ensure key order for later mapping

    # perform a curve fit for each experimental run
    for (T_p, k) in zip(T_p_arr, k_arr):

        # ! Fit Assuming 1D Conduction
        # linear regression of the temperature profile
        (dTdx, j[T_b_str], j["rval"], j["pval"], j["stderr"]) = linregress(x, T_p)
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
        # increment the index
        i += 1

    # ! Generate Results DataFrame
    df_results = pd.DataFrame(index=df.index, data=results)

    # concatenate original DataFrame with results
    df = pd.concat([df, df_results], axis="columns")

    return df


def get_superheat(
    df: pd.DataFrame,
    T_b_str: str,
    T_L_str: str,
    T_w_str: list[str],
    material: str,
    L: float,
    D: float,
    wait: float = 7,
) -> pd.DataFrame:
    """
    Fit the data assuming one-dimensional, steady-state conduction.
    """

    # ! Environment Variables
    EESIN = "EESIN"
    EESOUT = "EESOUT"
    EES = "EES"
    EESFILE = "EESFILE"

    # ! Inputs
    # get temperatures along the post as a numpy array
    T_b_arr = df.loc[:, T_b_str].values
    # get average water temperature, part of superheat calculation
    T_w_avg = np.mean(df.loc[:, T_w_str].mean(axis="columns").values)
    # post geometry

    # ! Property Lookup
    # write post material, number of runs, and base temperatures to IN.DAT
    with open(environ[EESIN], "w+") as f:
        print(material, len(T_b_arr), *T_b_arr, file=f)
    # call on PowerShell to invoke EES to write thermal conductivities to OUT.DAT given
    # the contents of IN.DAT
    subprocess.Popen(
        ["pwsh.exe", "-Command", "& $env:" + EES + " $pwd\\$env:" + EESFILE + " /solve"]
    )
    # Since the subprocess command finishes before EES does, we have to wait long enough
    # for EES to finish its job. We could hook into the process, but waiting is fine.
    sleep(wait)
    # EES should have written to OUT.DAT. get the properties from it
    with open(environ[EESOUT], "r") as f:
        k_str = f.read().split("\t")
        k_arr = np.array(k_str, dtype=np.float64)

    k_str = f"k_{material} (W/m-K)"

    # keys for a results dict that will become a DataFrame
    keys = [
        T_L_str,
        "DT (C)",
        k_str,
    ]
    # preallocate numpy arrays
    results = {key: np.full_like(k_arr, np.nan) for key in keys}

    class StrictDict(dict):
        """Dict that doesn't allow new keys to be set after construction."""

        def __setitem__(self, key, value):
            if key not in self:
                raise KeyError(
                    f'Cannot add "{key}" key to StrictDict with immutable keys.'
                )
            dict.__setitem__(self, key, value)

    # prepare an index for mapping results
    j = StrictDict(dict.fromkeys(keys))  # ensure key order for later mapping

    # get the superheat for each experimental run
    for i, (T_b, k) in enumerate(zip(T_b_arr, k_arr)):
        # ? q and DT
        dTdx = -df.at[df.index[i], "q (W/m^2)"] / k
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

if __name__ == "__main__":
    main()
