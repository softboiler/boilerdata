from enum import auto
import re

from typer import Argument, Typer

from boilerdata.enums import NameEnum
from boilerdata.models import trials
from boilerdata.models.project import Project
from boilerdata.utils import write_schema

app = Typer()


class Model(NameEnum):
    all_models = "all"
    project = auto()
    trials = auto()


all_models = {Model.project: Project, Model.trials: trials.Trials}


@app.command()
def update_schema(
    model: Model = Argument(..., help='The model, or "all".', case_sensitive=False),
):
    """
    Given a Pydantic model named e.g. "Model", write its JSON schema to
    "schema/Model.json".
    """

    if model == Model.all_models:
        for model in all_models.keys():
            update_schema(model)
    else:
        write_schema(
            f"project/schema/{to_snake_case(all_models[model].__name__)}_schema.json".lower(),
            all_models[model],
        )


# https://github.com/samuelcolvin/pydantic/blob/4f4e22ef47ab04b289976bb4ba4904e3c701e72d/pydantic/utils.py#L127-L131
def to_snake_case(v: str) -> str:
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", v)
    v = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", v)

    return v.replace("-", "_").lower()


if __name__ == "__main__":
    app()
