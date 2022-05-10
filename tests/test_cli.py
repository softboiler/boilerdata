import filecmp
from pathlib import Path
from boilerdata.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["trials", "test"])
    assert result.exit_code == 0


def test_write_schema(tmp_path):
    """Ensure the schema can be written and is up to date."""
    runner.invoke(app, ["pipeline", "schema", str(tmp_path)])
    schema = next(tmp_path.iterdir())
    expected_schema = Path("schema/boilerdata.toml.json")
    assert filecmp.cmp(schema, expected_schema)
