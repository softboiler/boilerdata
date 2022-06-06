"""Try regressing average versus actual thermocouple temperatures.

An exploration of the significance of taking the mean of multiple temperature
measurements versus the actual temperatures. In the "Estimate" approach, the mean of
`rep = proj.params.records_to_average` different temperatures of each thermocouple is
taken. Then this mean is repeated `rep` times before linear regression. This is meant to
represent the multiple samples taken at each thermocouple used to compute the
regression, and should give a representative `stderror` and `intercept_stderror`.

In the "Actual" approach, take `rep` number of actual temperatures during steady-state,
and perform the linear regression with those. It can be seen that the estimate
underestimates `pvalue`, `stderr`, and `intercept_stderr` slightly, but this isn't
significant enough to worry about.

The highest average standard deviation for any thermocouple is only about 0.15K,
resulting in a 95% CI of about 0.30K. The K-type thermocouples have about 30% higher
standard deviation than the T-types. Nominal error for K-type thermocouples is no more
than +/- 2.5C in the measured range, which is assumed to be a 95% CI for a single
measurement. Since error goes by `1/sqrt(rep)`, that amounts to to ~0.30K 95% CI.

The result of `(reg_true_df.int95 / reg_true_df.TLfit).mean()` is the average relative
error of the intercept at the 95% CI assuming a student's t-distribution. For all trials
thus far, the mean relative error is ~0.006 of the temperature in Celsius, or about 0.5K
if the intercept is near 100C.

The relevant value for plotting error bars going forward should be the 95% CI computed
in the fashion just described. The additional contribution of measurement error in the
regression is assumed to be negligible since it affects `stderr` and `intercept_stderr`
by 1e-5.


| Stat             | (Est-Act)/Act |
| :--------------- | :------------ |
| dT_dx            | +3.58E-13     |
| TLfit            | +2.19E-15     |
| rvalue           | +2.46E-01     |
| pvalue           | -2.92E-03     |
| stderr           | -1.01E-05     |
| intercept_stderr | -1.01E-05     |

"""
from numpy import typing as npt
import numpy as np
import pandas as pd
from scipy.stats import linregress, t

from boilerdata.axes_enum import Axes as A  # noqa: N817
from boilerdata.pipeline import get_df


def main():
    proj = Project.get_project()
    all_temps = get_df(proj)[proj.trials[0].thermocouple_pos.keys()]

    # ! Takes awhile
    # temps.groupby(level=A.run).describe().to_csv(
    #     proj.dirs.results / "Explorations/try_00_act_vs_est_describe.csv"
    # )

    # ! https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    # Two-sided inverse Students t-distribution
    # p - probability, df - degrees of freedom
    tinv = lambda p, df: abs(t.ppf(p / 2, df))  # noqa: E731

    reg_trues: list[pd.Series] = []
    regs: list[pd.Series] = []
    for trial in proj.trials:
        x = list(trial.thermocouple_pos.values())
        rep = proj.params.records_to_average
        res_index = [A.dT_dx, A.T_s, A.rvalue, A.pvalue, "stderr", "intercept_stderr"]

        runs_temps = all_temps.xs(trial.trial, level=A.trial)

        # Could use "apply" here, but it's okay.
        for _, temps in runs_temps.groupby(level=A.run):
            temps = temps.droplevel(level=A.run)

            y_true = temps.T.stack()
            reg_true = linregress_ser(y_true, x, rep, res_index)

            y_mean = temps.mean().repeat(rep)
            reg_mean = linregress_ser(y_mean, x, rep, res_index)

            # ! Degrees of freedom are reps times number of thermocouples minus 2
            ts = tinv(0.05, rep * 5 - 2)
            reg_true["slope95"] = ts * reg_true.stderr
            reg_true["int95"] = ts * reg_true.intercept_stderr

            reg_trues.append(reg_true)
            regs.append((reg_mean - reg_true) / reg_true)

    reg_true_df = pd.concat(reg_trues, axis="columns").T[["TLfit", "int95"]]
    reg_true_df = reg_true_df[reg_true_df.TLfit < 120]
    print((reg_true_df.int95 / reg_true_df.TLfit).mean())

    # ! Don't need to rewrite this
    # regs_df = pd.concat(regs, axis="columns").T
    # regs_df.max().to_csv(proj.dirs.results / "Explorations/try_00_act_vs_est.csv")


def linregress_ser(
    series_of_y: pd.Series,
    x: npt.ArrayLike,
    repeats_per_pair: int,
    regression_stats: list[str],
) -> pd.Series:
    """This implementation doesn't repeat the y-values by default."""

    # Assume the ordered pairs are repeated with zero standard deviation in x and y
    x = np.repeat(x, repeats_per_pair)
    r = linregress(x, series_of_y)

    # Unpacking would drop r.intercept_stderr, so we have to do it this way.
    # See "Notes" section of SciPy documentation for more info.
    return pd.Series(
        [r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr],
        index=regression_stats,
    )


if __name__ == "__main__":
    main()
