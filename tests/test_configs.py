from boilerdata.configs import load_config
from pydantic import BaseModel
from pytest import raises
from pytest import mark as m
import toml


class UserModel(BaseModel):
    test: str


USER_MODEL = """\
#:schema schema.json

test = "hello"
"""

USER_MODEL_NO_SCHEMA = 'test = "hello"'

SCHEMA = """\
{
  "title": "UserModel",
  "type": "object",
  "properties": { "test": { "title": "Test", "type": "string" } },
  "required": ["test"]
}
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


@m.parametrize(
    "test_id, user_model, expected_schema_directive",
    [
        ("schema", USER_MODEL, "#:schema schema.json"),
        ("no_schema", USER_MODEL_NO_SCHEMA, None),
    ],
)
def test_load_config(test_id, user_model, expected_schema_directive, tmp_path):
    user_model_path = tmp_path / "user_model.toml"
    user_model_path.write_text(user_model)
    config, schema_directive = load_config(user_model_path, UserModel)
    assert toml.load(user_model_path) == config.dict()  # type: ignore
    assert schema_directive == expected_schema_directive
