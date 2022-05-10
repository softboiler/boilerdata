"""Manipulate trials."""

from pathlib import Path
import toml
import typer
from pydantic import BaseModel


class Trials(BaseModel):
    pass


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
