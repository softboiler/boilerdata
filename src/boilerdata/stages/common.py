import pandas as pd

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project
from boilerdata.models.trials import Trial


def get_trial(df: pd.DataFrame, proj: Project) -> Trial:
    """Get the trial represented in a given dataframe, verifying it is the only one."""
    trials_in_df = df.index.get_level_values(A.trial)
    if trials_in_df.nunique() > 1:
        raise RuntimeError("There was more than one trial when getting the trial.")
    trial = pd.Timestamp(trials_in_df[0].date())  # type: ignore  # pandas
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
    """Group dataframe by index and apply a function to the groups, setting dtypes."""
    df = (
        df.groupby(level=level, sort=False, group_keys=False)  # type: ignore
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
