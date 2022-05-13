from pathlib import Path

from pydantic import BaseModel, DirectoryPath, Extra, Field


class Project(BaseModel, extra=Extra.forbid):
    """Base configuration for boilerdata."""

    trials: DirectoryPath = Field(
        ..., description="The path to a directory containing trials."
    )
    data: Path = Field(
        ..., description="The relative path inside each trial directory to its data."
    )
