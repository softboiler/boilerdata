from pandera import Check, Column, DataFrameSchema, Index, MultiIndex

from axes import Axes as A  # noqa: N817
from utils import get_project

proj = get_project()
c = {ax.name: ax for ax in proj.axes.all}
tc_submerged_and_boiling = Check.in_range(95, 101)  # (C)

initial_index = [
    Index(name=A.trial, dtype=c[A.trial].dtype),
    Index(name=A.run, dtype=c[A.run].dtype),
    Index(name=A.time, dtype=c[A.time].dtype),
]

initial_cols = {
    A.group: Column(c[A.group].dtype),
    A.rod: Column(c[A.rod].dtype),
    A.coupon: Column(c[A.coupon].dtype),
    A.sample: Column(c[A.sample].dtype, nullable=True),
    A.joint: Column(c[A.joint].dtype),
    A.top_of_coupon_tc: Column(c[A.top_of_coupon_tc].dtype),
    A.good: Column(c[A.good].dtype),
    A.new: Column(c[A.new].dtype),
    A.V: Column(c[A.V].dtype),
    A.I: Column(c[A.I].dtype),
    A.T_0: Column(c[A.T_0].dtype),
    A.T_1: Column(c[A.T_1].dtype),
    A.T_2: Column(c[A.T_2].dtype),
    A.T_3: Column(c[A.T_3].dtype),
    A.T_4: Column(c[A.T_4].dtype),
    A.T_5: Column(c[A.T_5].dtype),
    A.T_6: Column(c[A.T_6].dtype, nullable=True),  # Some trials don't have it
    A.T_w1: Column(c[A.T_w1].dtype, tc_submerged_and_boiling),
    A.T_w2: Column(c[A.T_w2].dtype, tc_submerged_and_boiling),
    A.T_w3: Column(c[A.T_w3].dtype, tc_submerged_and_boiling),
    A.P: Column(c[A.P].dtype),
    # A.P: Column(c[A.P].dtype, Check.greater_than(12)),  # Strict check on pressure
}

validate_runs_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(initial_index),
    columns=initial_cols,
    checks=(
        # Check that every run was tailed properly
        Check(
            lambda df: all(
                df.groupby(level=A.run, sort=False).apply(len)
                == proj.params.records_to_average
            )
        )
    ),
)

validate_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(initial_index[:-1]),  # the final index is dropped by now
    columns={
        **initial_cols,
        A.dT_dx: Column(c[A.dT_dx].dtype),
        A.dT_dx_err: Column(c[A.dT_dx_err].dtype),
        A.T_s: Column(c[A.T_s].dtype),
        A.T_s_err: Column(c[A.T_s_err].dtype),
        A.rvalue: Column(c[A.rvalue].dtype),
        A.pvalue: Column(c[A.pvalue].dtype),
        A.k: Column(c[A.k].dtype),
        A.q: Column(c[A.q].dtype),
        A.q_err: Column(c[A.q_err].dtype),
        A.Q: Column(c[A.Q].dtype),
        A.DT: Column(c[A.DT].dtype),
        A.DT_err: Column(c[A.DT_err].dtype),
    },
)
