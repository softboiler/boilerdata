"""Utility functions for interoperating with EES."""

from pathlib import Path
from subprocess import run  # noqa: S404  # only used for hardcoded calls
from tempfile import TemporaryDirectory

EES_ROOT = Path("C:/EES32")
EES_PATH = EES_ROOT / "EES.exe"


def run_script(text: str):
    """Run an EES script."""
    with TemporaryDirectory() as tempdir:
        file = Path(tempdir) / "file"
        file.write_text(text)
        run([f"{EES_PATH}", f"{file.resolve()}", "/Solve"])  # noqa: S603
