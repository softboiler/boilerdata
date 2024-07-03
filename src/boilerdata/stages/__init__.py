"""Stages."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import matplotlib as mpl
import pandas as pd
from boilercore.fits import plot_fit
from boilercore.modelfun import get_model
from boilercore.models.trials import Trial
from matplotlib import pyplot as plt

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS, Params

idxs = pd.IndexSlice
"""Use to slice pd.MultiIndex indices."""

MODEL, MODEL_WITH_UNCERTAINTY = get_model(PARAMS.paths.model_functions)

# * -------------------------------------------------------------------------------- * #
# * DATA MANIPULATION


def get_run(params: Params, run: Path) -> pd.DataFrame:
    """Get data for a single run."""
    # Get source columns
    index = params.axes.index[-1].source  # Get the last index, associated with source
    source_col_names = [col.source for col in params.axes.source_cols]
    source_dtypes = {col.source: col.dtype for col in params.axes.source_cols}

    # Assign columns from CSV and metadata to the structured dataframe. Get the tail.
    df = pd.DataFrame(
        columns=source_col_names,
        data=pd.read_csv(
            run,
            # Allow source cols to be missing (such as certain thermocouples)
            usecols=lambda col: col in [index, *source_col_names],
            index_col=index,
            parse_dates=[index],  # type: ignore  # pandas
            dtype=source_dtypes,
            encoding="utf-8",
        ),
        # Rarely a run has an all NA record at the end
    ).dropna(how="all")

    # Need "df" defined so we can call "df.index.dropna()". Repeat `dropna` because a
    # run can have an NA index at the end and a CSV can have an all NA record at the end
    return (
        df.reindex(index=df.index.dropna())
        .dropna(how="all")
        .pipe(rename_columns, params)
    )


def get_trial(df: pd.DataFrame, params: Params) -> Trial:
    """Get the trial represented in a given dataframe, verifying it is the only one."""
    trials_in_df = df.index.get_level_values(A.trial)
    if trials_in_df.nunique() > 1:
        raise RuntimeError("There was more than one trial when getting the trial.")
    trial = pd.Timestamp(trials_in_df[0].date())  # type: ignore  # pandas
    return params.get_trial(trial)


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    return (
        df
        if df.empty
        else df.assign(
            **{name: df[name].astype(dtype) for name, dtype in dtypes.items()}  # type: ignore
        )
    )


def set_proj_dtypes(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in params.axes.cols})


def per_index(
    df: pd.DataFrame,
    level: str | list[str],
    per_index_func,
    params: Params,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Group dataframe by index and apply a function to the groups, setting dtypes."""
    df = (
        df.groupby(level=level, sort=False, group_keys=False)  # type: ignore
        .apply(per_index_func, params, *args, **kwargs)
        .pipe(set_proj_dtypes, params)
    )
    return df


def per_trial(
    df: pd.DataFrame, per_trial_func, params: Params, *args, **kwargs
) -> pd.DataFrame:
    """Apply a function to individual trials."""
    df = per_index(df, A.trial, per_trial_func, params, *args, **kwargs)
    return df


def per_run(
    df: pd.DataFrame, per_run_func, params: Params, *args, **kwargs
) -> pd.DataFrame:
    """Apply a function to individual runs."""
    df = per_index(df, [A.trial, A.run], per_run_func, params, *args, **kwargs)
    return df


def add_units(
    df: pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Make the columns a multi-index representing units."""
    cols = params.axes.get_col_index()
    quantity = cols.get_level_values("quantity")
    units = cols.get_level_values("units")

    old = (col.name for col in params.axes.cols)
    new = (add_unit(q, u) for q, u in zip(quantity, units, strict=True))
    mapper = dict(zip(old, new, strict=True))
    return df.rename(axis="columns", mapper=mapper), mapper


# * -------------------------------------------------------------------------------- * #
# * PLOTTING


@contextmanager
def manual_subplot_spacing():
    """Context manager that allows custom spacing of subplots."""
    with mpl.rc_context({"figure.autolayout": False}):
        yield


def tex_wrap(df: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, str]]:
    """Wrap column titles in LaTeX flags if they contain underscores ($)."""
    mapper: dict[str, str] = {}
    for src_col in df.columns:
        col = f"${handle_subscript(src_col)}$" if "_" in src_col else src_col
        mapper[src_col] = col
    return df.rename(axis="columns", mapper=mapper), mapper


def plot_new_fits(grp: pd.DataFrame, params: Params, model):
    """Plot model fits for trials marked as new."""
    trial = get_trial(grp, params)
    if not trial.plot:
        return grp

    tcs, tc_errors = get_tcs(trial)
    x = list(trial.thermocouple_pos.values())
    for fig_dst, (ser_name, ser) in dict(
        zip(
            [
                params.paths.plot_new_fit_0,
                params.paths.plot_new_fit_1,
                params.paths.plot_new_fit_2,
            ],
            grp.iloc[[0, (len(grp) // 2) - 1, len(grp) - 1]].iterrows(),
            strict=True,
        )
    ).items():
        fig, ax = plt.subplots(layout="constrained")
        plot_fit(
            ax=ax,
            run=ser_name[-1].isoformat(),  # type: ignore  # pandas
            x=x,
            y=ser[tcs],  # type: ignore  # pandas
            y_errors=ser[tc_errors],  # type: ignore  # pandas
            y_0=ser[A.T_s],
            model=model,
            params=get_params_mapping(ser, params.fit.model_params),
            errors=get_params_mapping(ser, params.fit.model_errors),
        )
        if params.do_plot:
            fig.savefig(fig_dst, dpi=300)  # type: ignore  # matplotlib


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def rename_columns(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Rename source columns."""
    return df.rename(columns={col.source: col.name for col in params.axes.cols})


def handle_subscript(val: str) -> str:
    """Wrap everything after the first underscore and replace others with commas."""
    quantity, units = sep_unit(val)
    parts = quantity.split("_")
    quantity = f"{parts[0]}_" + "{" + ",".join(parts[1:]) + "}"
    return add_unit(quantity, units, tex=True)


def add_unit(quantity: str, units: str, tex: bool = False) -> str:
    """Append units to a quantity."""
    if not tex:
        return f"{quantity} ({units})" if units else quantity
    units = units.replace("-", r"{\cdot}")
    return rf"{quantity}\;({units})" if units else quantity


def sep_unit(val: str) -> tuple[str, str]:
    """Split a quantity and its units."""
    quantity, units = val.split(" (")
    units = units.removesuffix(")")
    return quantity, units


def get_tcs(trial: Trial) -> tuple[list[str], list[str]]:
    """Get the thermocouple names and their error names for this trial."""
    tcs = list(trial.thermocouple_pos.keys())
    tc_errors = [tc + "_err" for tc in tcs]
    return tcs, tc_errors


T = TypeVar("T")


def get_params_mapping(ser: pd.Series, model_params: Sequence[T]) -> dict[str, T]:  # type: ignore  # pandas
    """Get a mapping of parameter names to values."""
    return dict(zip(model_params, ser[model_params], strict=True))  # type: ignore  # pandas
