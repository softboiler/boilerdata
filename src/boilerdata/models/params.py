from pydantic import Field

from boilerdata.models.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel


class Params(MyBaseModel):
    """Parameters of the pipeline."""

    records_to_average: int = Field(
        default=60,
        description="The number of records over which to average in a given trial.",
    )
    water_temps: list[A] = Field(
        default=[A.T_w1, A.T_w2, A.T_w3],
        description="The water temperature measurements.",
    )
    do_plot: bool = Field(
        default=False,
        description="Whether to plot the fits of the individual runs.",
    )
