# # Necessary as long as a line marked "triggered only locally" is in this file
# pyright: reportUnnecessaryTypeIgnoreComment=none

from typing import Any

import numpy as np
import pandas as pd
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial


def get_trial(df: pd.DataFrame, proj: Project):
    """Get the trial represented in a given dataframe, verifying it is the only one."""
    trials_in_df = df.index.get_level_values(A.trial)
    if trials_in_df.nunique() > 1:
        raise RuntimeError("There was more than one trial when getting the trial.")
    trial = pd.Timestamp(
        trials_in_df[0].date()  # pyright: ignore [reportGeneralTypeIssues]  # pandas
    )
    return proj.get_trial(trial)


def per_trial(
    df: pd.DataFrame,
    per_trial_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual trials."""
    df = per_index(df, A.trial, per_trial_func, proj, *args, **kwargs)
    return df


def per_run(
    df: pd.DataFrame,
    per_run_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Apply a function to individual runs."""
    df = per_index(df, [A.trial, A.run], per_run_func, proj, *args, **kwargs)
    return df


def per_index(
    df: pd.DataFrame,
    level: str | list[str],
    per_index_func,
    proj: Project,
    *args,
    **kwargs,
) -> pd.DataFrame:
    df = (
        df.groupby(
            level=level,  # pyright: ignore [reportGeneralTypeIssues]
            sort=False,
            group_keys=False,
        )
        .apply(per_index_func, proj, *args, **kwargs)
        .pipe(set_proj_dtypes, proj)
    )
    return df


def set_proj_dtypes(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in proj.axes.cols})


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    return df.assign(**{name: df[name].astype(dtype) for name, dtype in dtypes.items()})


def get_tcs(trial: Trial) -> tuple[list[str], list[str]]:
    """Get the thermocouple names and their error names for this trial."""
    tcs = list(trial.thermocouple_pos.keys())
    tc_errors = [tc + "_err" for tc in tcs]
    return tcs, tc_errors


def get_params_mapping(grp: pd.DataFrame, params: list[Any]) -> dict[str, Any]:
    """Get a mapping of parameter names to values."""
    # Reason: pydantic: use_enum_values
    return dict(zip(params, grp[params]))  # type: ignore


def get_params_mapping_with_uncertainties(
    grp: pd.DataFrame, proj: Project
) -> dict[str, Any]:
    """Get a mapping of parameter names to values with uncertainty."""
    model_params_and_errors = proj.params.params_and_errors
    # Reason: pydantic: use_enum_values
    params: list[str] = proj.params.model_params  # type: ignore
    param_errors: list[str] = proj.params.model_errors  # type: ignore
    u_params = [
        ufloat(param, err, tag)
        for param, err, tag in zip(
            grp[params], grp[param_errors], model_params_and_errors
        )
    ]
    return dict(zip(model_params_and_errors, u_params))


def model_with_error(model, x, u_params):
    """Evaluate the model for x and return y with errors."""
    u_x = [ufloat(v, 0, "x") for v in x]
    u_y = model(u_x, **u_params)
    y = np.array([v.nominal_value for v in u_y])
    y_min = y - [
        v.std_dev for v in u_y
    ]  # pyright: ignore [reportGeneralTypeIssues]  # uncertainties, triggered only locally
    y_max = y + [v.std_dev for v in u_y]
    return y, y_min, y_max


def get_results(proj: Project):
    return pd.read_csv(
        proj.dirs.file_results,
        index_col=(index_col := [A.trial, A.run]),
        parse_dates=index_col,
        dtype={col.name: col.dtype for col in proj.axes.cols},
    )
