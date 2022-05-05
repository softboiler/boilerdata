"""Data processing pipeline for a nucleate pool boiling apparatus."""

__version__ = "0.0.0"

from boilerdata.api import *  # type: ignore

from rich import pretty, traceback

pretty.install()
traceback.install()
