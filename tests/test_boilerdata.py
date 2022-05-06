from contextlib import contextmanager
from filecmp import cmp
from os import chdir, getcwd
from pathlib import Path
from shutil import copytree
import boilerdata
from boilerdata.configs import write_schema
import pandas as pd

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


def test_write_schema(tmpdir):
    """Ensure the schema can be written and is up to date."""
    write_schema(tmpdir)
    schema = next(Path(tmpdir).iterdir())
    expected_schema = Path("schema/boilerdata.toml.json")
    assert cmp(schema, expected_schema)


def test_run(tmpdir):
    """Ensure the same result is coming out of the pipeline as before."""
    test_data = tmpdir / "data"
    copytree(DATA, test_data)
    with working_directory(test_data):
        boilerdata.run()
    result = pd.read_csv(test_data / RESULT)
    expected = pd.read_csv(DATA / RESULT)
    pd.testing.assert_frame_equal(result, expected)
