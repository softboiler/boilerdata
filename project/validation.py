from pandera import Check, Column, DataFrameSchema, Index, MultiIndex

from axes import Axes as A  # noqa: N817
from utils import get_project

proj = get_project()
c = {ax.name: ax for ax in proj.axes.all}
tc_submerged_and_boiling = Check.in_range(95, 101)  # (C)

validate_runs_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    checks=(
        # Check that every run was tailed properly
        Check(
            lambda df: all(
                df.groupby(level=A.run, sort=False).apply(len)
                == proj.params.records_to_average
            )
        )
    ),
    index=MultiIndex(
        [
            Index(name=A.trial, dtype=c[A.trial].dtype),
            Index(name=A.run, dtype=c[A.run].dtype),
            Index(name=A.time, dtype=c[A.time].dtype),
        ]
    ),
    columns={
        "group": Column(c[A.group].dtype),
        "rod": Column(c[A.rod].dtype),
        "coupon": Column(c[A.coupon].dtype),
        "sample": Column(c[A.sample].dtype, nullable=True),
        "joint": Column(c[A.joint].dtype),
        "good": Column(c[A.good].dtype),
        "new": Column(c[A.new].dtype),
        "V": Column(c[A.V].dtype),
        "I": Column(c[A.I].dtype),
        "T_0": Column(c[A.T_0].dtype),
        "T_1": Column(c[A.T_1].dtype),
        "T_2": Column(c[A.T_2].dtype),
        "T_3": Column(c[A.T_3].dtype),
        "T_4": Column(c[A.T_4].dtype),
        "T_5": Column(c[A.T_5].dtype),
        "T_w1": Column(c[A.T_w1].dtype, tc_submerged_and_boiling),
        "T_w2": Column(c[A.T_w2].dtype, tc_submerged_and_boiling),
        "T_w3": Column(c[A.T_w3].dtype, tc_submerged_and_boiling),
        # "P": Column(c[A.P].dtype, Check.greater_than(12)),  # Strict check on pressure
        "P": Column(c[A.P].dtype),
    },
)

validate_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(
        [
            Index(name=A.trial, dtype=c[A.trial].dtype),
            Index(name=A.run, dtype=c[A.run].dtype),
            # Index(name=A.time, dtype=c[A.time].dtype),  # Not in the reduced df
        ]
    ),
    columns={
        "group": Column(c[A.group].dtype),
        "rod": Column(c[A.rod].dtype),
        "coupon": Column(c[A.coupon].dtype),
        "sample": Column(c[A.sample].dtype, nullable=True),
        "joint": Column(c[A.joint].dtype),
        "good": Column(c[A.good].dtype),
        "new": Column(c[A.new].dtype),
        "V": Column(c[A.V].dtype),
        "I": Column(c[A.I].dtype),
        "T_0": Column(c[A.T_0].dtype),
        "T_1": Column(c[A.T_1].dtype),
        "T_2": Column(c[A.T_2].dtype),
        "T_3": Column(c[A.T_3].dtype),
        "T_4": Column(c[A.T_4].dtype),
        "T_5": Column(c[A.T_5].dtype),
        "T_w1": Column(c[A.T_w1].dtype, tc_submerged_and_boiling),
        "T_w2": Column(c[A.T_w2].dtype, tc_submerged_and_boiling),
        "T_w3": Column(c[A.T_w3].dtype, tc_submerged_and_boiling),
        # "P": Column(c[A.P].dtype, Check.greater_than(12)),  # Strict check on pressure
        "P": Column(c[A.P].dtype),
        "dT_dx": Column(c[A.dT_dx].dtype),
        "dT_dx_err": Column(c[A.dT_dx_err].dtype),
        "T_s": Column(c[A.T_s].dtype),
        "T_s_err": Column(c[A.T_s_err].dtype),
        "rvalue": Column(c[A.rvalue].dtype),
        "pvalue": Column(c[A.pvalue].dtype),
        "k": Column(c[A.k].dtype),
        "q": Column(c[A.q].dtype),
        "q_err": Column(c[A.q_err].dtype),
        "Q": Column(c[A.Q].dtype),
        "DT": Column(c[A.DT].dtype),
        "DT_err": Column(c[A.DT_err].dtype),
    },
)
