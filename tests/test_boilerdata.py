from filecmp import cmp
from pathlib import Path
from boilerdata.configs import write_schema

DATA = Path("tests/data")


def test_write_schema(tmpdir):
    """Ensure the schema can be written and is up to date."""
    write_schema(tmpdir)
    schema = next(Path(tmpdir).iterdir())
    expected_schema = Path("schema/boilerdata.toml.json")
    assert cmp(schema, expected_schema)


# def test_write_schema(tmpdir):
#     data = tmpdir / "data"
#     copytree(DATA, data)
#     write_schema(tmpdir)
#     ...
