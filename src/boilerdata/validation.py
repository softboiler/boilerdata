import pandas as pd
from pandera import Check, Column, DataFrameSchema, Index, MultiIndex
from pandera.errors import SchemaError

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project

proj = Project.get_project()
c = {ax.name: ax for ax in proj.axes.all}

# * -------------------------------------------------------------------------------- * #
# * HANDLING AND CHECKS

columns_to_automatically_handle = [*proj.params.water_temps, A.P]
water_tc_in_range = Check.in_range(95, 101)  # (C)
pressure_in_range = Check.in_range(12, 15)  # (psi)
water_temps_agree = Check.less_than(1.5)  # (C)

# * -------------------------------------------------------------------------------- * #
# * INDEX AND COLUMNS

initial_index = [
    Index(name=A.trial, dtype=c[A.trial].dtype),
    Index(name=A.run, dtype=c[A.run].dtype),
    Index(name=A.time, dtype=c[A.time].dtype),
]

meta_cols = {
    A.group: Column(c[A.group].dtype),
    A.rod: Column(c[A.rod].dtype),
    A.coupon: Column(c[A.coupon].dtype),
    A.sample: Column(c[A.sample].dtype, nullable=True),
    A.joint: Column(c[A.joint].dtype),
    A.sixth_tc: Column(c[A.sixth_tc].dtype),
    A.good: Column(c[A.good].dtype),
    A.new: Column(c[A.new].dtype),
}

source_cols = {
    A.V: Column(c[A.V].dtype, nullable=True),  # Not used
    A.I: Column(c[A.I].dtype, nullable=True),  # Not used
    A.T_0: Column(c[A.T_0].dtype),
    A.T_1: Column(c[A.T_1].dtype),
    A.T_1_err: Column(c[A.T_1_err].dtype, nullable=True),
    A.T_2: Column(c[A.T_2].dtype),
    A.T_2_err: Column(c[A.T_2_err].dtype, nullable=True),
    A.T_3: Column(c[A.T_3].dtype),
    A.T_3_err: Column(c[A.T_3_err].dtype, nullable=True),
    A.T_4: Column(c[A.T_4].dtype),
    A.T_4_err: Column(c[A.T_4_err].dtype, nullable=True),
    A.T_5: Column(c[A.T_5].dtype),
    A.T_5_err: Column(c[A.T_5_err].dtype, nullable=True),
    A.T_6: Column(c[A.T_6].dtype, nullable=True),  # Some trials don't have it
    A.T_w1: Column(c[A.T_w1].dtype, water_tc_in_range),
    A.T_w2: Column(c[A.T_w2].dtype, water_tc_in_range),
    A.T_w3: Column(c[A.T_w3].dtype, water_tc_in_range),
    A.P: Column(c[A.P].dtype, pressure_in_range),
}

computed_cols = {
    A.T_w: Column(c[A.T_w].dtype),
    A.T_w_diff: Column(c[A.T_w_diff].dtype, checks=water_temps_agree),
    A.a: Column(c[A.a].dtype),
    A.a_err: Column(c[A.a_err].dtype),
    A.b: Column(c[A.b].dtype),
    A.b_err: Column(c[A.b_err].dtype),
    A.c: Column(c[A.c].dtype),
    A.c_err: Column(c[A.c_err].dtype),
    A.dT_dx: Column(c[A.dT_dx].dtype),
    A.dT_dx_err: Column(c[A.dT_dx_err].dtype),
    A.T_s: Column(c[A.T_s].dtype),
    A.T_s_err: Column(c[A.T_s_err].dtype),
    A.k: Column(c[A.k].dtype),
    A.q: Column(c[A.q].dtype),
    A.q_err: Column(c[A.q_err].dtype),
    A.Q: Column(c[A.Q].dtype),
    A.DT: Column(c[A.DT].dtype),
    A.DT_err: Column(c[A.DT_err].dtype),
}

# * -------------------------------------------------------------------------------- * #
# * VALIDATION

validate_initial_df = DataFrameSchema(
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(initial_index),
    columns=source_cols,
)

validate_final_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(initial_index[:-1]),  # the final index is dropped by now
    columns=meta_cols | source_cols | computed_cols,
)


# * -------------------------------------------------------------------------------- * #
# * HANDLING


def handle_invalid_data(df: pd.DataFrame, validator: DataFrameSchema) -> pd.DataFrame:
    validation_error = True
    while validation_error:
        try:
            df = validator(df)
        except SchemaError as exc:
            if (
                # It can be a dataframe with ambiguous existence and truthiness
                exc.check_output is False
                or exc.check_output is None
                # Only handle certain columns
                or exc.check_output.name not in columns_to_automatically_handle
            ):
                raise
            failed = exc.check_output
            df = df.assign(
                **{failed.name: (lambda df: df[failed.name].where(failed).ffill())}
            )
            continue
        else:
            validation_error = False
    return df
