"""CLI for boilerdata."""

from enum import auto
from types import ModuleType

from typer import Argument, Typer
from typer.main import get_command_name

from boilerdata.enums import NameEnum
from boilerdata.models import trials
from boilerdata.models.configs import Config
from boilerdata.utils import write_schema


def add_typer_autoname(app: Typer, module: ModuleType):
    """Add a subcommand to a Typer app, inferring its name from the module name.

    Given a Typer app and a module, add a subcommand to the Typer app with the same name
    as the module itself, all lowercase and with underscores swapped for dashes.

    Parameters
    ----------
    app: Typer
        The app to which a subcommand should be added.
    module: ModuleType
        The module which will be added as an automatically-named subcommand.
    """
    name = get_command_name(module.__name__.split(".")[-1])
    app.add_typer(module.app, name=name)


# This flattens nested namespaces
app = Typer()
for module in [trials]:
    add_typer_autoname(app, module)

# * -------------------------------------------------------------------------------- * #
# * UTILS

app_utils = Typer()
app.add_typer(app_utils, name="utils")


class Model(NameEnum):
    all_models = "all"
    config = auto()
    trials = auto()


all_models = {Model.config: Config, Model.trials: trials.Trials}


@app_utils.command("schema")
def write_schema_cli(
    model: Model = Argument(..., help='The model, or "all".', case_sensitive=False),
):
    """
    Given a Pydantic model named e.g. "Model", write its JSON schema to
    "schema/Model.json".
    """

    if model == Model.all_models:
        for model in all_models.keys():
            write_schema_cli(model)
    else:
        write_schema(f"schema/{model.name}_schema.json".lower(), all_models[model])
