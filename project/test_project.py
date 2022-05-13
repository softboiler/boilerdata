from schema.update import app
from typer.testing import CliRunner

runner = CliRunner()


def test_schema_missing_required_arg_raises():
    assert 0 != runner.invoke(app, []).exit_code


def test_schema_all():
    assert 0 == runner.invoke(app, ["all"]).exit_code


def test_schema_other_than_all():
    assert 0 == runner.invoke(app, ["project"]).exit_code
