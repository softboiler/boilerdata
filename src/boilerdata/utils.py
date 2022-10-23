import numpy as np
import pandas as pd

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
        df.groupby(level=level, sort=False, group_keys=False, dropna=False)  # type: ignore  # Issue w/ pandas-stubs
        .apply(per_index_func, proj, *args, **kwargs)
        .pipe(set_proj_dtypes, proj)
    )
    return df


def set_proj_dtypes(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
    """Set project-specific dtypes for the dataframe."""
    return set_dtypes(df, {col.name: col.dtype for col in proj.axes.cols})


def set_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Set column datatypes in a dataframe."""
    if all(df.isna()):
        return df
    else:
        return df.assign(
            **{name: df[name].astype(dtype) for name, dtype in dtypes.items()}
        )


def get_tcs(trial: Trial) -> tuple[list[str], list[str]]:
    """Get the thermocouple names and their error names for this trial."""
    tcs = list(trial.thermocouple_pos.keys())
    tc_errors = [tc + "_err" for tc in tcs]
    return tcs, tc_errors


def zip_params(grp: pd.DataFrame, proj: Project):
    model_params = proj.params.model_params
    params = [param for param in model_params if "err" not in param]  # type: ignore  # Due to use_enum_values
    param_errors = [param for param in model_params if "err" in param]  # type: ignore  # Due to use_enum_values
    return zip(grp[params], grp[param_errors], model_params)  # type: ignore  # Due to use_enum_values


def model_with_error(model, x, u_params):
    """Evaluate the model for x and return y with errors."""
    # Can't vectorize the below operation due to the error:
    #     TypeError: loop of ufunc does not support argument 0 of type AffineScalarFunc
    #     which has no callable exp method
    u_y = model(x, *u_params)
    y = np.array([y.nominal_value for y in u_y])
    y_min = y - [y.std_dev for y in u_y]  # type: ignore  # Due to unknown array type
    y_max = y + [y.std_dev for y in u_y]
    return y, y_min, y_max
