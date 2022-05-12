"""Manipulate trials."""

from pathlib import Path

from pydantic import BaseModel
import toml
import typer

from boilerdata.enums import NameEnum
from boilerdata.enums.rods import Rod


class Trial(BaseModel):
    """Trial docstring."""

    name: str
    rod: Rod
    upper: NameEnum
    sample: NameEnum


class Trials(BaseModel):
    """Trials docstring."""

    trials: list[Trial]


app = typer.Typer()


@app.command("get")
def get_trials():
    a = {
        "trials": [
            {"test": 100, "best": 200, "rest": 300},
            {"test": 100, "best": 200, "rest": 300},
        ]
    }
    Path("test.toml").write_text(toml.dumps(a))
