from pathlib import Path

from pytest import mark as m
from typer.testing import CliRunner

from boilerdata.main import app

runner = CliRunner()


def test_app_help():
    assert runner.invoke(app, ["--help"]).exit_code == 0


@m.xfail
def test_write_schema(tmp_path):
    """Ensure the schema can be written and is up to date."""
    schema_path = tmp_path / "test.json"
    runner.invoke(app, ["pipeline", "schema", str(schema_path)])
    expected_schema = Path("schema/boilerdata.toml.json")
    assert schema_path.read_text() == expected_schema.read_text()


def test_schema_missing_required_arg_raises():
    assert 0 != runner.invoke(app, ["utils", "schema", "what"]).exit_code


def test_schema_all():
    assert 0 == runner.invoke(app, ["utils", "schema", "all"]).exit_code


def test_schema_other_than_all():
    assert 0 == runner.invoke(app, ["utils", "schema", "config"]).exit_code
