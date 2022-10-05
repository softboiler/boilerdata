from pandera import Check, Column, DataFrameSchema, Index, MultiIndex
from pandera.errors import SchemaError

from boilerdata.models.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.project import Project

proj = Project.get_project()
c = {ax.name: ax for ax in proj.axes.all}
tc_submerged_and_boiling = Check.in_range(95, 101)  # (C)

# * -------------------------------------------------------------------------------- * #
# * INDEX AND COLUMNS

initial_index = [
    Index(name=A.trial, dtype=c[A.trial].dtype),
    Index(name=A.run, dtype=c[A.run].dtype),
    Index(name=A.time, dtype=c[A.time].dtype),
]

source_cols = {
    A.V: Column(c[A.V].dtype, nullable=True),  # Not used
    A.I: Column(c[A.I].dtype, nullable=True),  # Not used
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
    A.P: Column(c[A.P].dtype, Check.greater_than(12)),
}

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

computed_cols = {
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
}

# * -------------------------------------------------------------------------------- * #
# * VALIDATION


def all_tailed_properly(df):
    return all(tailed_properly(df))


def tailed_properly(df):
    return (
        df.iloc[:, 0].groupby(level=A.run).transform(len)
        == proj.params.records_to_average
    )


validate_runs_df = DataFrameSchema(
    strict=True,
    ordered=True,
    unique_column_names=True,
    index=MultiIndex(initial_index),
    columns=source_cols,
    # checks=Check(all_tailed_properly),  # TODO: Don't check once strategy changes
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


def handle_invalid_data(proj, df, validator):
    catch_and_ffill = [*proj.params.water_temps, A.P]
    validation_error = True
    while validation_error:
        try:
            df = validator(df)
        except SchemaError as exc:
            # It can be a dataframe with ambiguous existence and truthiness
            if exc.check_output is False or exc.check_output is None:
                raise
            failed = exc.check_output
            if failed.name in catch_and_ffill:
                df = df.assign(
                    **{failed.name: (lambda df: df[failed.name].where(failed).ffill())}
                )
            continue
        else:
            validation_error = False
    return df
