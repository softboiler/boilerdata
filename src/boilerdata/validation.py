"""Validation."""

import pandas as pd
from pandera import Check, Column, DataFrameSchema, Index, MultiIndex
from pandera.errors import SchemaError

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.params import PARAMS

c = {ax.name: ax for ax in PARAMS.axes.all}

# * -------------------------------------------------------------------------------- * #
# * HANDLING AND CHECKS

columns_to_automatically_handle = [*PARAMS.water_temps, A.P]
water_tc_in_range = Check.in_range(95, 101)  # (C)
pressure_in_range = Check.in_range(12, 15)  # (psi)
water_temps_agree = Check.less_than(1.6)  # (C)

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
    A.good: Column(c[A.good].dtype),
    A.plot: Column(c[A.plot].dtype),
}

runs_cols = {
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
    A.T_w1: Column(c[A.T_w1].dtype, water_tc_in_range),
    A.T_w2: Column(c[A.T_w2].dtype, water_tc_in_range),
    A.T_w3: Column(c[A.T_w3].dtype, water_tc_in_range),
    A.P: Column(c[A.P].dtype, pressure_in_range),
}


# Model fits are nullable because they may not converge. Nullability propagates to other
# columns downstream.
model_cols = {
    col: Column(c[col].dtype, nullable=True) for col in PARAMS.fit.params_and_errors
}

computed_cols = {
    A.T_w: Column(c[A.T_w].dtype),
    A.T_w_diff: Column(c[A.T_w_diff].dtype, checks=water_temps_agree),
    **model_cols,
    A.DT: Column(c[A.DT].dtype, nullable=True),
    A.DT_err: Column(c[A.DT_err].dtype, nullable=True),
}

# * -------------------------------------------------------------------------------- * #
# * VALIDATION

# We know that `meta_cols | runs_cols | computed_cols` are in the DataFrame, but we
# don't check for their presence (nor do we specify `strict`) because they're all null
# right now. We can make sure `model_cols` are here, though.
validate_initial_df = DataFrameSchema(
    unique_column_names=True,
    index=MultiIndex(initial_index),
    columns=runs_cols | model_cols,
)

validate_final_df = DataFrameSchema(
    strict=True,
    unique_column_names=True,
    index=MultiIndex(initial_index[:-1]),  # the final index is dropped by now
    columns=meta_cols | runs_cols | computed_cols,
)


# * -------------------------------------------------------------------------------- * #
# * HANDLING


def handle_invalid_data(df: pd.DataFrame, validator: DataFrameSchema) -> pd.DataFrame:
    """Handle invalid data."""
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
            df = df.assign(**{
                failed.name: (
                    lambda df: df[failed.name].where(failed).ffill()  # noqa: B023
                )
            })
            continue
        else:
            validation_error = False
    return df
