from pytest import mark as m
from typer.testing import CliRunner

from boilerdata.__main__ import app

runner = CliRunner()


@m.skip("Fails if no commands are registered.")
def test_app_help():
    assert runner.invoke(app, ["--help"]).exit_code == 0
