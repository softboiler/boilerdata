from pydantic import BaseModel
from pytest import mark as m, raises
import toml
from typer.testing import CliRunner

from boilerdata.utils import dump_model, load_config, write_schema

runner = CliRunner()


class UserModel(BaseModel):
    test: str


USER_MODEL_INSTANCE = UserModel(test="hello")

USER_MODEL_TOML = """\
#:schema schema.json

test = "hello"
\
"""

USER_MODEL_MISSING_KEY_TOML = """\
#:schema schema.json
\
"""

USER_MODEL_TOML_NO_SCHEMA = 'test = "hello"\n'

SCHEMA_JSON = """\
{
  "title": "UserModel",
  "type": "object",
  "properties": {
    "test": {
      "title": "Test",
      "type": "string"
    }
  },
  "required": [
    "test"
  ]
}
\
"""


@m.parametrize("test_id, file", [("does_not_exist", "file"), ("not_a_file", "")])
def test_load_config_raises(test_id, file, tmp_path):
    with raises(FileNotFoundError):
        load_config(tmp_path / file, UserModel)


def test_load_config_raises_not_toml(tmp_path):
    file = tmp_path / "test.not_toml"
    file.touch()

    with raises(ValueError):
        load_config(file, UserModel)


def test_load_config_raises_validation(tmp_path):
    user_model_path = tmp_path / "test.toml"
    user_model_path.write_text(USER_MODEL_MISSING_KEY_TOML)
    # Can't check for ValidationError directly for some reason
    with raises(Exception, match="validation error"):
        load_config(user_model_path, UserModel)


@m.parametrize(
    "test_id, user_model, expected_schema_directive",
    [
        ("schema", USER_MODEL_TOML, "#:schema schema.json"),
        ("no_schema", USER_MODEL_TOML_NO_SCHEMA, None),
    ],
)
def test_load_config(test_id, user_model, expected_schema_directive, tmp_path):
    user_model_path = tmp_path / "user_model.toml"
    user_model_path.write_text(user_model)
    config, schema_directive = load_config(user_model_path, UserModel)
    assert toml.load(user_model_path) == config.dict()  # type: ignore
    assert schema_directive == expected_schema_directive


def test_dump_model(tmp_path, capfd):
    user_model_path = tmp_path / "test.toml"
    dump_model(user_model_path, USER_MODEL_INSTANCE)
    assert user_model_path.read_text() == USER_MODEL_TOML_NO_SCHEMA


def test_write_schema_raises_not_json(tmp_path):
    file = tmp_path / "test.not_json"
    with raises(ValueError):
        write_schema(file, UserModel)


def test_write_schema(tmp_path):
    """Ensure the schema can be written and is up to date."""
    schema_path = tmp_path / "test.json"
    write_schema(schema_path, UserModel)
    assert schema_path.read_text() == SCHEMA_JSON
