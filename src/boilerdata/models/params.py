from pydantic import Field

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel


class Params(MyBaseModel):
    """Parameters of the pipeline."""

    records_to_average: int = Field(
        default=5,
        description="The number of records over which to average in a given trial.",
    )
    model_params: list[A] = Field(
        default=[
            A.a,
            A.b,
            A.c,
            A.a_err,
            A.b_err,
            A.c_err,
        ],
        description="The parameters of the model to be fitted.",
    )
    model_outs: list[A] = Field(
        default=[
            A.T_s,
            A.dT_dx,
            A.T_s_err,
            A.dT_dx_err,
        ],
        description="The parameters of the model to be fitted.",
    )
    water_temps: list[A] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="The water temperature measurements.",
    )
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )
