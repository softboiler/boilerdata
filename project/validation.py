from pandera import Check, Column, DataFrameSchema, Index, MultiIndex

from columns import Columns as C  # noqa: N817
from utils import get_project

proj = get_project()
c = proj.cols
tc_submerged_and_boiling = Check.in_range(95, 101)  # (C)

validate_runs_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    checks=(
        # Check that every run was tailed properly
        Check(
            lambda df: all(
                df.groupby(level=C.run, sort=False).apply(len)
                == proj.params.records_to_average
            )
        )
    ),
    index=MultiIndex(
        [
            Index(name=C.trial, dtype=c[C.trial].dtype),
            Index(name=C.run, dtype=c[C.run].dtype),
            Index(name=C.time, dtype=c[C.time].dtype),
        ]
    ),
    columns={
        "group": Column(c[C.group].dtype),
        "rod": Column(c[C.rod].dtype),
        "coupon": Column(c[C.coupon].dtype),
        "sample": Column(c[C.sample].dtype, nullable=True),
        "joint": Column(c[C.joint].dtype),
        "good": Column(c[C.good].dtype),
        "new": Column(c[C.new].dtype),
        "comment": Column(c[C.comment].dtype, nullable=True),
        "V": Column(c[C.V].dtype),
        "I": Column(c[C.I].dtype),
        "T_0": Column(c[C.T_0].dtype),
        "T_1": Column(c[C.T_1].dtype),
        "T_2": Column(c[C.T_2].dtype),
        "T_3": Column(c[C.T_3].dtype),
        "T_4": Column(c[C.T_4].dtype),
        "T_5": Column(c[C.T_5].dtype),
        "T_w1": Column(c[C.T_w1].dtype, tc_submerged_and_boiling),
        "T_w2": Column(c[C.T_w2].dtype, tc_submerged_and_boiling),
        "T_w3": Column(c[C.T_w3].dtype, tc_submerged_and_boiling),
        # "P": Column(c[C.P].dtype, Check.greater_than(12)),  # Strict check on pressure
        "P": Column(c[C.P].dtype),
    },
)

validate_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(
        [
            Index(name=C.trial, dtype=c[C.trial].dtype),
            Index(name=C.run, dtype=c[C.run].dtype),
            # Index(name=C.time, dtype=c[C.time].dtype),  # Not in the reduced df
        ]
    ),
    columns={
        "group": Column(c[C.group].dtype),
        "rod": Column(c[C.rod].dtype),
        "coupon": Column(c[C.coupon].dtype),
        "sample": Column(c[C.sample].dtype, nullable=True),
        "joint": Column(c[C.joint].dtype),
        "good": Column(c[C.good].dtype),
        "new": Column(c[C.new].dtype),
        "comment": Column(c[C.comment].dtype, nullable=True),
        "V": Column(c[C.V].dtype),
        "I": Column(c[C.I].dtype),
        "T_0": Column(c[C.T_0].dtype),
        "T_1": Column(c[C.T_1].dtype),
        "T_2": Column(c[C.T_2].dtype),
        "T_3": Column(c[C.T_3].dtype),
        "T_4": Column(c[C.T_4].dtype),
        "T_5": Column(c[C.T_5].dtype),
        "T_w1": Column(c[C.T_w1].dtype, tc_submerged_and_boiling),
        "T_w2": Column(c[C.T_w2].dtype, tc_submerged_and_boiling),
        "T_w3": Column(c[C.T_w3].dtype, tc_submerged_and_boiling),
        # "P": Column(c[C.P].dtype, Check.greater_than(12)),  # Strict check on pressure
        "P": Column(c[C.P].dtype),
        "dT_dx": Column(c[C.dT_dx].dtype),
        "TLfit": Column(c[C.TLfit].dtype),
        "rvalue": Column(c[C.rvalue].dtype),
        "pvalue": Column(c[C.pvalue].dtype),
        "stderr": Column(c[C.stderr].dtype),
        "intercept_stderr": Column(c[C.intercept_stderr].dtype),
        "dT_dx_err": Column(c[C.dT_dx_err].dtype),
        "k": Column(c[C.k].dtype),
        "q": Column(c[C.q].dtype),
        "q_err": Column(c[C.q_err].dtype),
        "Q": Column(c[C.Q].dtype),
        "DT": Column(c[C.DT].dtype),
        "DT_err": Column(c[C.DT_err].dtype),
    },
)
