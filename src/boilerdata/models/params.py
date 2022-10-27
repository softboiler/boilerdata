import numpy as np
from pydantic import Field, validator

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel


class Params(MyBaseModel):
    """Parameters of the pipeline."""

    records_to_average: int = Field(
        default=5,
        description="The number of records over which to average in a given trial.",
    )
    model_inputs: dict[str, float] = Field(
        default=dict(
            h_w=0,  # (W/m^2-K)
            k=400.0,  # (W/m-K)
            r=0.0047625,  # (m)
            T_infa=25.0,  # (C)
            T_infw=100.0,  # (C)
            x_s=0,  # (m)
            x_wa=0.0381,  # (m)
        ),
        description="The inputs to the model to be fitted.",
    )

    @validator("model_inputs", always=True)
    def validate_model_inputs(cls, model_inputs):
        # Substitute this instead of zero to avoid division by zero
        eps = np.finfo(float).eps
        return {
            k: eps if v == 0 and "x" not in k else v for k, v in model_inputs.items()
        }

    model_params: list[A] = Field(
        default=[
            A.T_s,
            A.q,
            A.h_a,
            A.T_s_err,
            A.q_err,
            A.h_a_err,
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
