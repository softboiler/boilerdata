"""CLI for boilerdata."""

import typer

from boilerdata import trials, pipeline

app = typer.Typer()
app.add_typer(trials.app, name=trials.__name__.split(".")[-1])
app.add_typer(pipeline.app, name=pipeline.__name__.split(".")[-1])
