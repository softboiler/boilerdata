from contextlib import contextmanager
from os import chdir, getcwd
from pathlib import Path
from shutil import copytree

import pandas as pd

from boilerdata import pipeline

DATA = Path("tests/data")
RESULT = Path("fitted.csv")


@contextmanager
def working_directory(path: Path):
    original_working_directory = getcwd()
    try:
        chdir(path)
        yield
    finally:
        chdir(original_working_directory)


def test_run(tmp_path):
    """Ensure the same result is coming out of the pipeline as before."""
    test_data = tmp_path / "data"
    copytree(DATA, test_data)
    with working_directory(test_data):
        pipeline.run()
    result = pd.read_csv(test_data / RESULT)
    expected = pd.read_csv(DATA / RESULT)
    pd.testing.assert_frame_equal(result, expected)
