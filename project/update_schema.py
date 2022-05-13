"""Update all schema."""

import re

from models.project import Project
from models.trials import Trials

from boilerdata.utils import write_schema

models = [Project, Trials]


def main():
    for model in models:
        write_schema(
            f"project/schema/{to_snake_case(model.__name__)}_schema.json", model
        )


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


if __name__ == "__main__":
    main()
