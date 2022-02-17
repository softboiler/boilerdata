"""Pipeline functions."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dynaconf import Dynaconf
from propshop import get_prop
from propshop.library import Mat, Prop
from pydantic import DirectoryPath, validator
from pydantic.dataclasses import dataclass
from scipy.stats import linregress
from scipy.constants import convert_temperature


# * -------------------------------------------------------------------------------- * #
# * CONFIGURE


def material_validator(*fields):
    return validator(*fields, allow_reuse=True)(lambda string: string.upper())


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

    _ = material_validator("material")


@dataclass
class GetSuperheatParams:
    T_b_str: str
    T_L_str: str
    T_w_str: list[str]
    material: str
    L: float

    _ = material_validator("material")


@dataclass
class Config:
    data_path: DirectoryPath
    fit_params: FitParams
    get_superheat_params: GetSuperheatParams

    @validator("fit_params", "get_superheat_params")
    def _(cls, param):
        return asdict(param)


raw_config = Dynaconf(settings_files=["examples/parameters.yaml"])
config = Config(
    data_path=raw_config.data_path,
    fit_params=FitParams(**raw_config.fit),
    get_superheat_params=GetSuperheatParams(**raw_config.get_superheat),
)

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main():

    data: Path = config.data_path  # type: ignore
    files: list[Path] = sorted((data / "raw").glob("*.csv"))
    stems: list[str] = [file.stem for file in files]
    runs: list[pd.DataFrame] = [pd.read_csv(file, index_col=0) for file in files]
    steady_state_per_run: list[pd.Series] = [df_.iloc[-80:, :].mean() for df_ in runs]
    (
        pd.DataFrame(steady_state_per_run, index=stems)
        .pipe(fit, **config.fit_params)  # type: ignore
        .pipe(get_superheat, **config.get_superheat_params)  # type: ignore
    ).to_csv(data / "fitted.csv", index_label="From Dataset")


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

    # Get thermal conductivities
    temps_per_run = df.loc[:, T_p_str]
    df = df.assign(
        **{
            "k (W/m-K)": get_prop(
                Mat[material],
                Prop.THERMAL_CONDUCTIVITY,
                convert_temperature(temps_per_run.mean(axis="columns"), "C", "K"),
            )
        }
    )

    # Perform regression along the post temperatures
    def linregress_cols(series: pd.Series) -> pd.Series:
        return pd.Series(
            linregress(x, series),
            index=["dTdx (K/m)", T_b_str, "rvalue", "pvalue", "stderr"],
        )

    runs = df.loc[:, T_p_str].T
    regressions_per_run = runs.agg(linregress_cols).T
    df = pd.concat([df, regressions_per_run], axis="columns")
    # keys for a results dict that will become a DataFrame
    keys = [
        T_b_str,
        T_L_str,
        "Q (W)",
        "q (W/m^2)",
        "rval",
        "pval",
        "stderr",
    ]

    k_arr = df.loc[:, "k (W/m-K)"]
    # preallocate numpy arrays
    results = {key: np.full_like(k_arr, np.nan) for key in keys}

    j = StrictDict(dict.fromkeys(keys))  # ensure key order for later mapping

    # ! Inputs
    # get temperatures along the post as a numpy array
    T_p_arr = df.loc[:, T_p_str].values  # noqa: N806
    # post geometry
    A = np.pi / 4 * D**2  # noqa: N806

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
    k_arr = get_prop(
        Mat[material], Prop.THERMAL_CONDUCTIVITY, convert_temperature(T_b_arr, "C", "K")
    )

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

if __name__ == "__main__":
    main()
