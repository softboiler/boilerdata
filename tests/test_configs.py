from boilerdata.configs import load_config
from pydantic import BaseModel
from pytest import raises
from pytest import mark as m


class UserModel(BaseModel):
    test: str


@m.parametrize("test_id, file", [("does_not_exist", "file"), ("not_a_file", "")])
def test_load_config_raises(test_id, file, tmp_path):
    with raises(FileNotFoundError):
        load_config(tmp_path / file, UserModel)


def test_load_config_raises_not_toml(tmp_path):
    file = tmp_path / "test.not_toml"
    file.touch()

    with raises(ValueError):
        load_config(file, UserModel)
