# # # Necessary as long as a line marked "triggered only in CI" is in this file
# # pyright: reportUnnecessaryTypeIgnoreComment=none


# import json
# from shutil import copy
# import warnings

# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd

# import janitor  # pyright: ignore [reportUnusedImport]  # adds methods to pd.DataFrame


# from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
# from boilerdata.models.project import Project
# from boilerdata.stages.common import (
#     get_params_mapping,
#     get_params_mapping_with_uncertainties,
#     get_tcs,
#     get_trial,
#     model_with_error,
#     per_run,
# )
# from boilerdata.stages.modelfun import model_with_uncertainty

# # * -------------------------------------------------------------------------------- * #
# # * MAIN


# def main(proj: Project):

#     meta = [col.name for col in proj.axes.meta]
#     errors = proj.params.free_errors
#     cols = meta + errors
#     dtypes = {col.name: col.dtype for col in proj.axes.cols if col.name in cols}
#     (
#         pd.read_csv(
#             proj.dirs.file_results,
#             index_col=(index := [A.trial, A.run]),
#             # Allow source cols to be missing (such as T_6)
#             usecols=lambda col: col in index + cols,
#             parse_dates=index,
#             dtype=dtypes,  # pyright: ignore [reportGeneralTypeIssues]  # pandas
#         )
#         .also(plot_error_distribution, proj, errors)
#         .also(plot_error_groups, proj, errors)
#         .also(write_metrics, proj)
#         .also(plot_fits, proj, model_with_uncertainty)
#     )


# # * -------------------------------------------------------------------------------- * #
# # * PLOTS


# def plot_error_distribution(df: pd.DataFrame, proj: Project, errors: list[str]):
#     """Produce a box-plot showing the error by joint type in each free parameter."""
#     fig, ax = plt.subplots(layout="constrained")
#     df = df[[A.joint, *errors]].pipe(add_units, proj)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", UserWarning)
#         df.plot.box(by=A.joint, ax=ax)
#     fig.savefig(
#         proj.dirs.file_pipeline_metrics_plot,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
#         dpi=300,
#     )


# def plot_error_groups(df: pd.DataFrame, proj: Project, errors: list[str]):
#     fig, ax = plt.subplots(layout="constrained")
#     df = df[[A.joint, *errors]].pipe(add_units, proj)
#     ...


# # * -------------------------------------------------------------------------------- * #
# # * METRICS


# def write_metrics(df: pd.DataFrame, proj: Project):
#     """Compute summary metrics of the model fit and write them to a file."""

#     # sourcery skip: merge-dict-assign
#     fits: list[str] = proj.params.model_params  # type: ignore
#     errors: list[str] = proj.params.model_errors  # type: ignore
#     first_fit = fits[0]

#     def strip_err(df: pd.DataFrame) -> pd.DataFrame:
#         """Strip the "err" suffix from the column names."""
#         return df.rename(axis="columns", mapper=lambda col: col.removesuffix("_err"))

#     # Reason: pydantic: use_enum_values
#     error_ratio = df[errors].pipe(strip_err) / df[fits]
#     error_normalized = (df[errors] / df[errors].max()).pipe(strip_err)

#     # Compute the rate of failures to fit the model
#     metrics: dict[str, float] = {}
#     metrics["fit_failure_rate"] = df[first_fit].isna().sum() / len(df)

#     # Compute the median and spread of the error two ways
#     metric_dfs = {"err_ratio": error_ratio, "err_norm": error_normalized}
#     for err_tag, err_df in metric_dfs.items():
#         for agg in ["median", "std"]:
#             metrics |= {
#                 f"{k}_{err_tag}_{agg}": v for k, v in err_df.agg(agg).to_dict().items()
#             }
#     metrics |= {k: 0 for k, v in metrics.items() if np.isnan(v)}
#     proj.dirs.file_pipeline_metrics.write_text(json.dumps(metrics, indent=2))


# # * -------------------------------------------------------------------------------- * #
# # * PLOT FITS


# def plot_fits(df: pd.DataFrame, proj: Project, model) -> pd.DataFrame:
#     """Get the latest new model fit plot."""
#     if proj.params.do_plot:
#         per_run(df, plot_new_fits, proj, model)
#         if figs_src := sorted(proj.dirs.new_fits.iterdir()):
#             figs_src = (
#                 figs_src[0],
#                 figs_src[len(figs_src) // 2],
#                 figs_src[-1],
#             )
#             figs_dst = (
#                 proj.dirs.file_new_fit_0,
#                 proj.dirs.file_new_fit_1,
#                 proj.dirs.file_new_fit_2,
#             )
#             for fig_src, fig_dst in zip(figs_src, figs_dst):
#                 copy(fig_src, fig_dst)
#     return df


# def plot_new_fits(grp: pd.DataFrame, proj: Project, model):
#     """Plot model fits for trials marked as new."""

#     trial = get_trial(grp, proj)
#     if not trial.new:
#         return grp

#     ser = grp.squeeze()
#     tcs, tc_errors = get_tcs(trial)
#     x_unique = list(trial.thermocouple_pos.values())
#     y_unique = ser[tcs]

#     # Plot setup
#     fig, ax = plt.subplots(layout="constrained")

#     run = ser.name[-1].isoformat()
#     run_file = proj.dirs.new_fits / f"{run.replace(':', '-')}.png"

#     ax.margins(0, 0)
#     ax.set_title(f"{run = }")
#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("T (C)")

#     # Initial plot boundaries
#     x_bounds = np.array([0, trial.thermocouple_pos[A.T_1]])

#     y_bounds = model(x_bounds, **get_params_mapping(ser, proj.params.model_params))
#     ax.plot(
#         x_bounds,
#         y_bounds,
#         "none",
#     )

#     # Measurements
#     measurements_color = [0.2, 0.2, 0.2]
#     ax.plot(
#         x_unique,
#         y_unique,
#         ".",
#         label="Measurements",
#         color=measurements_color,
#         markersize=10,
#     )
#     ax.errorbar(
#         x=x_unique,
#         y=y_unique,
#         yerr=ser[tc_errors],
#         fmt="none",
#         color=measurements_color,
#     )

#     # Confidence interval
#     (xlim_min, xlim_max) = ax.get_xlim()
#     pad = 0.025 * (xlim_max - xlim_min)
#     x_padded = np.linspace(xlim_min - pad, xlim_max + pad)

#     y_padded, y_padded_min, y_padded_max = model_with_error(
#         model, x_padded, get_params_mapping_with_uncertainties(ser, proj)
#     )
#     ax.plot(
#         x_padded,
#         y_padded,
#         "--",
#         label="Model Fit",
#     )
#     ax.fill_between(
#         x=x_padded,
#         y1=y_padded_min,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib, triggered only in CI
#         y2=y_padded_max,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
#         color=[0.8, 0.8, 0.8],
#         edgecolor=[1, 1, 1],
#         label="95% CI",
#     )

#     # Extrapolation
#     ax.plot(
#         0,
#         ser[A.T_s],
#         "x",
#         label="Extrapolation",
#         color=[1, 0, 0],
#     )

#     # Finishing
#     ax.legend()
#     fig.savefig(
#         run_file,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
#         dpi=300,
#     )


# # * -------------------------------------------------------------------------------- * #
# # * HELPER FUNCTIONS


# def add_units(df: pd.DataFrame, proj: Project) -> pd.DataFrame:
#     """Make the columns a multi-index representing units."""
#     cols = proj.axes.get_col_index()
#     quantity = cols.get_level_values("quantity")
#     units = cols.get_level_values("units")

#     old = (col.name for col in proj.axes.cols)
#     new = (add_unit(q, u) for q, u in zip(quantity, units))
#     return df.rename(axis="columns", mapper=dict(zip(old, new)))


# def add_unit(quantity: str, units: str) -> str:
#     return f"{quantity} ({units})" if units else quantity


# # * -------------------------------------------------------------------------------- * #

# if __name__ == "__main__":
#     main(Project.get_project())
