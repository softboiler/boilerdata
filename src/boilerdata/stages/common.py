# # Necessary as long as a line marked "triggered only locally" is in this file
# pyright: reportUnnecessaryTypeIgnoreComment=none

from os import chdir
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial


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


def zip_params(grp: pd.DataFrame, proj: Project):
    model_params = proj.params.model_params
    params = [
        param
        for param in model_params
        if "err"
        not in param  # pyright: ignore [reportGeneralTypeIssues]  # pydantic: use_enum_values
    ]
    param_errors = [
        param
        for param in model_params
        if "err"
        in param  # pyright: ignore [reportGeneralTypeIssues]  # pydantic: use_enum_values
    ]
    return zip(
        grp[
            params
        ],  # pyright: ignore [reportGeneralTypeIssues]  # pydantic: use_enum_values
        grp[
            param_errors
        ],  # pyright: ignore [reportGeneralTypeIssues]  # pydantic: use_enum_values
        model_params,
    )


def model_with_error(model, x, u_params):
    """Evaluate the model for x and return y with errors."""
    u_x = [ufloat(v, 0, "x") for v in x]
    u_y = model(u_x, *u_params)
    y = np.array([v.nominal_value for v in u_y])
    y_min = y - [
        v.std_dev for v in u_y
    ]  # pyright: ignore [reportGeneralTypeIssues]  # uncertainties, triggered only locally
    y_max = y + [v.std_dev for v in u_y]
    return y, y_min, y_max


def chdir_to_nearest_git_root(max_depth: int = 7) -> None:
    """Change the working directory to the nearest git root."""
    original_cwd = Path.cwd()
    if (original_cwd / Path(".git")).exists():
        return
    eventual_cwd = original_cwd.parent
    current_drive_root = Path(original_cwd.anchor)
    for _ in range(max_depth + 1):
        if eventual_cwd == current_drive_root:
            raise RuntimeError(
                "Couldn't find git project folder above drive root.\n"
                f"Original CWD: {original_cwd}\n"
                f"Stopped at : {eventual_cwd}\n"
            )
        if (eventual_cwd / Path(".git")).exists():
            chdir(eventual_cwd)
            return
        eventual_cwd = eventual_cwd.parent
    raise RuntimeError(
        f"Couldn't find git project folder above max depth of {max_depth}.\n"
        f"Original CWD: {original_cwd}\n"
        f"Stopped at : {eventual_cwd}\n"
    )


def get_results(proj: Project):
    return pd.read_csv(
        proj.dirs.file_results,
        index_col=(index_col := [A.trial, A.run]),
        parse_dates=index_col,
        dtype={col.name: col.dtype for col in proj.axes.cols},
    )
