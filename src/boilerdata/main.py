"""CLI for boilerdata."""

from types import ModuleType

from typer import Option, Typer
from typer.main import get_command_name

from boilerdata import configs, pipeline, trials
from boilerdata.models import Model, model_from_cli


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


app = Typer()
for module in [configs, pipeline, trials]:
    add_typer_autoname(app, module)


@app.command("schema")
def write_schema_cli(
    model: Model = Option(
        ..., help="The Pydantic model.", case_sensitive=False, show_choices=True
    )
):
    """
    Given a Pydantic model named e.g. "Model", write its JSON schema to
    "schema/Model.json".
    """
    configs.write_schema(f"schema/{model.name}_schema.json", model_from_cli[model])
