from pathlib import Path

from typer.testing import CliRunner

from boilerdata.main import app

runner = CliRunner()


def test_app_help():
    assert runner.invoke(app, ["--help"]).exit_code == 0


# TODO: Move this to pipeline tests
def test_write_schema(tmp_path):
    """Ensure the schema can be written and is up to date."""
    schema_path = tmp_path / "test.json"
    runner.invoke(app, ["pipeline", "schema", str(schema_path)])
    expected_schema = Path("schema/boilerdata.toml.json")
    assert schema_path.read_text() == expected_schema.read_text()
