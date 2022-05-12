from pydantic import BaseModel, DirectoryPath, Field


class Config(BaseModel):
    """Base configuration for boilerdata."""

    trials: DirectoryPath = Field(..., description="The path to the trials directory.")
