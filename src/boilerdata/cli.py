"""CLI for boilerdata."""

import fire
from boilerdata import api, configs


def main():
    """Entry-point for the CLI."""
    fire.Fire({"run": api.run, "schema": configs.write_schema})
