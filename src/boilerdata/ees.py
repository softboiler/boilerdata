"""Utility functions for interoperating with EES."""

from pathlib import Path
from subprocess import run  # noqa: S404  # internal use only
from tempfile import TemporaryDirectory

from boilerdata.config import settings


def run_script(text: str):
    """Run an EES script."""
    with TemporaryDirectory() as tempdir:
        file = Path(tempdir) / "file"
        file.write_text(text)
        run(  # noqa: S603  # internal use only
            [f"{settings.ees}", f"{file.resolve()}", "/Solve"]  # type: ignore
        )
