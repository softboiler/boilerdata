"""Generate schema for the project."""

import re

from boilerdata.models.axes import Axes
from boilerdata.models.common import write_schema
from boilerdata.models.params import PARAMS, Params
from boilerdata.models.trials import Trials


def main():
    models = [Params, Trials, Axes]
    for model in models:
        write_schema(
            PARAMS.paths.project_schema
            / f"{to_snake_case(model.__name__)}_schema.json",
            model,
        )


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


if __name__ == "__main__":
    main()
