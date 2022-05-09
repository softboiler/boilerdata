from boilerdata.__main__ import app
from typer.testing import CliRunner

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["trials", "test"])
    assert result.exit_code == 0
