# # Necessary as long as a line marked "triggered only in CI" is in this file
# pyright: reportUnnecessaryTypeIgnoreComment=none


from pathlib import Path
from shutil import copy

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.modelfun import model_with_uncertainty
from boilerdata.models.project import Project
from boilerdata.utils import get_tcs, model_with_error, per_run, zip_params

# * -------------------------------------------------------------------------------- * #
# * MAIN


def main(proj: Project):

    (
        pd.read_csv(
            proj.dirs.simple_results_file,
            index_col=(index_col := [A.trial, A.run]),
            parse_dates=index_col,
            dtype={col.name: col.dtype for col in proj.axes.cols},
        ).pipe(plot_fits, proj, model_with_uncertainty)
    )


# * -------------------------------------------------------------------------------- * #
# * STAGES


def plot_fits(df: pd.DataFrame, proj: Project, model) -> pd.DataFrame:
    """Get the latest new model fit plot."""
    if proj.params.do_plot:
        per_run(df, plot_new_fits, proj, model)
        if figs_src := sorted(proj.dirs.new_fits.iterdir()):
            figs_src = (figs_src[0], figs_src[-1])
            figs_dst = (
                Path(f"data/plots/{num}_new_fit.png") for num in ("first", "last")
            )
            for fig_src, fig_dst in zip(figs_src, figs_dst):
                copy(fig_src, fig_dst)
    return df


def plot_new_fits(grp: pd.DataFrame, proj: Project, model):
    """Plot model fits for trials marked as new."""

    trial = proj.get_trial(pd.Timestamp(grp.name[0].date()))
    if not trial.new:
        return grp

    ser = grp.squeeze()
    tcs, tc_errors = get_tcs(trial)
    x_unique = list(trial.thermocouple_pos.values())
    y_unique = ser[tcs]
    u_params = np.array(
        [ufloat(param, err, tag) for param, err, tag in zip_params(ser, proj)]
    )

    # Plot setup
    fig, ax = plt.subplots(layout="constrained")

    run = ser.name[-1].isoformat()
    run_file = proj.dirs.new_fits / f"{run.replace(':', '-')}.png"

    ax.margins(0, 0)
    ax.set_title(f"{run = }")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("T (C)")

    # Initial plot boundaries
    x_bounds = np.array([0, trial.thermocouple_pos[A.T_1]])
    y_bounds = model(x_bounds, *[param.nominal_value for param in u_params])
    ax.plot(
        x_bounds,
        y_bounds,
        "none",
    )

    # Measurements
    measurements_color = [0.2, 0.2, 0.2]
    ax.plot(
        x_unique,
        y_unique,
        ".",
        label="Measurements",
        color=measurements_color,
        markersize=10,
    )
    ax.errorbar(
        x=x_unique,
        y=y_unique,
        yerr=ser[tc_errors],
        fmt="none",
        color=measurements_color,
    )

    # Confidence interval
    # model = uncertainties.wrap(model)
    (xlim_min, xlim_max) = ax.get_xlim()
    pad = 0.025 * (xlim_max - xlim_min)
    x_padded = np.linspace(xlim_min - pad, xlim_max + pad)
    y_padded, y_padded_min, y_padded_max = model_with_error(model, x_padded, u_params)
    ax.plot(
        x_padded,
        y_padded,
        "--",
        label="Model Fit",
    )
    ax.fill_between(
        x=x_padded,
        y1=y_padded_min,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib, triggered only in CI
        y2=y_padded_max,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
        color=[0.8, 0.8, 0.8],
        edgecolor=[1, 1, 1],
        label="95% CI",
    )

    # Extrapolation
    ax.plot(
        0,
        ser[A.T_s],
        "x",
        label="Extrapolation",
        color=[1, 0, 0],
    )

    # Finishing
    ax.legend()
    fig.savefig(
        run_file,  # pyright: ignore [reportGeneralTypeIssues]  # matplotlib
        dpi=300,
    )


# * -------------------------------------------------------------------------------- * #

if __name__ == "__main__":
    main(Project.get_project())
