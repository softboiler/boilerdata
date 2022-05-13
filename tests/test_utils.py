import re

from pydantic import BaseModel
from pytest import mark as m, raises
from typer.testing import CliRunner
import yaml

from boilerdata.utils import dump_model, load_config, write_schema

runner = CliRunner()


class UserModel(BaseModel):
    test: str
    other: str


USER_MODEL_INSTANCE = UserModel(test="hello", other="world")

USER_MODEL_YAML = "test: hello\nother: world\n"

USER_MODEL_MISSING_KEY_YAML = "test: hello\n"

SCHEMA_JSON = """\
{
  "title": "UserModel",
  "type": "object",
  "properties": {
    "test": {
      "title": "Test",
      "type": "string"
    },
    "other": {
      "title": "Other",
      "type": "string"
    }
  },
  "required": [
    "test",
    "other"
  ]
}
\
"""


@m.parametrize("test_id, file", [("does_not_exist", "file"), ("not_a_file", "")])
def test_load_config_raises(test_id, file, tmp_path):
    with raises(FileNotFoundError):
        load_config(tmp_path / file, UserModel)


def test_load_config_raises_not_yaml(tmp_path):
    file = tmp_path / "test.not_yaml"
    file.touch()
    with raises(ValueError, match=re.compile("yaml file", re.IGNORECASE)):
        load_config(file, UserModel)


def test_load_config_raises_value_error(tmp_path):
    user_model_path = tmp_path / "test.yaml"
    user_model_path.write_text("\n")
    # Can't check for ValidationError directly for some reason
    with raises(ValueError, match=re.compile("file is empty", re.IGNORECASE)):
        load_config(user_model_path, UserModel)


def test_load_config_raises_validation(tmp_path):
    user_model_path = tmp_path / "test.yaml"
    user_model_path.write_text(USER_MODEL_MISSING_KEY_YAML)
    # Can't check for ValidationError directly for some reason
    with raises(Exception, match=re.compile("validation error", re.IGNORECASE)):
        load_config(user_model_path, UserModel)


def test_load_config(tmp_path):
    user_model_path = tmp_path / "user_model.yaml"
    user_model_path.write_text(USER_MODEL_YAML)
    config = load_config(user_model_path, UserModel)
    assert yaml.safe_load(user_model_path.read_text()) == config.dict()


def test_dump_model(tmp_path):
    user_model_path = tmp_path / "test.toml"
    dump_model(user_model_path, USER_MODEL_INSTANCE)
    assert user_model_path.read_text() == USER_MODEL_YAML


def test_write_schema_raises_not_json(tmp_path):
    file = tmp_path / "test.not_json"
    with raises(ValueError):
        write_schema(file, UserModel)


def test_write_schema(tmp_path):
    """Ensure the schema can be written and is up to date."""
    schema_path = tmp_path / "test.json"
    write_schema(schema_path, UserModel)
    assert schema_path.read_text() == SCHEMA_JSON
