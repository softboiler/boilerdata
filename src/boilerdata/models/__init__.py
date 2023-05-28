"""Parameter models for this project."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import EllipsisType
from typing import Any, TypeVar

from pydantic import BaseModel, Extra, validator
from ruamel.yaml import YAML

YAML_INDENT = 2
yaml = YAML()
yaml.indent(mapping=YAML_INDENT, sequence=YAML_INDENT, offset=YAML_INDENT)
yaml.preserve_quotes = True  # type: ignore


@contextmanager
def allow_extra(model: BaseModel):
    """Temporarily allow extra properties to be set on a Pydantic model.
    This is useful when writing a custom `__init__`, where not explicitly allowing extra
    properties will result in errors, but you don't want to allow extra properties
    forevermore.

    Args:
        model: The model to allow extras on.
    """

    # Store the current value of the attribute or note its absence
    try:
        original_config = model.Config.extra
    except AttributeError:
        original_config = None
    model.Config.extra = Extra.allow

    # Yield the temporarily changed config, resetting or deleting it when done
    try:
        yield
    finally:
        if original_config:
            model.Config.extra = original_config
        else:
            del model.Config.extra


T = TypeVar("T")


def default_opt(default: T, optional: bool = False) -> EllipsisType | T:
    """Has a default that will be passed to a Pydantic model if optional.
    It is useful to set `optional` to `True` when actively developing a parameter, then
    revert it to `False` when that parameter is going to always be coming from a
    configuration file.
    """
    return default if optional else ...


class ProjectModel(BaseModel):
    """Model configuration for this project."""

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        extra = Extra.forbid


class YamlModel(BaseModel):
    """Model of a YAML file with automatic schema generation.

    Updates a JSON schema next to the YAML file with each initialization.
    """

    class Config:
        extra = Extra.forbid

    def __init__(self, data_file: Path):
        """Initialize and update the schema."""
        params = self.get_params(data_file)
        self.update_schema(data_file)
        super().__init__(**params)

    def get_params(self, data_file: Path) -> dict[str, Any]:
        """Get parameters from file."""
        return (
            yaml.load(data_file)
            if data_file.exists() and data_file.read_text(encoding="utf-8")
            else {}
        )

    def update_schema(self, data_file: Path):
        schema_file = data_file.with_name(f"{data_file.stem}_schema.json")
        schema_file.write_text(
            encoding="utf-8", data=f"{self.schema_json(indent=YAML_INDENT)}\n"
        )


class SynchronizedPathsYamlModel(YamlModel):
    """Model of a YAML file that synchronizes paths back to the file.

    For example, synchronize complex path structures back to `params.yaml` DVC files for
    pipeline orchestration.
    """

    def __init__(self, data_file: Path):
        """Initialize, update the schema, and synchronize paths in the file."""
        super().__init__(data_file)

    def get_params(self, data_file: Path) -> dict[str, Any]:
        """Get parameters from file, synchronizing paths in the file."""
        params = (
            yaml.load(data_file)
            if data_file.exists() and data_file.read_text(encoding="utf-8")
            else {}
        )
        params |= self.get_paths()
        yaml.dump(params, data_file)
        return params

    def get_paths(self) -> dict[str, dict[str, str]]:
        """Get all paths specified in paths-type models."""
        maybe_excludes = self.__exclude_fields__
        excludes = set(maybe_excludes.keys()) if maybe_excludes else set()
        defaults: dict[str, dict[str, str]] = {}
        for key, field in self.__fields__.items():
            type_ = field.type_
            try:
                is_paths_model = issubclass(type_, DefaultPathsModel)
            except TypeError:
                is_paths_model = False
            if is_paths_model and key not in excludes:
                defaults[key] = type_.get_paths()
        return defaults


class DefaultPathsModel(BaseModel):
    """All fields must be path-like and have defaults specified in this model."""

    class Config:
        extra = Extra.forbid

        @staticmethod
        def schema_extra(schema: dict[str, Any], model: type[DefaultPathsModel]):
            """Replace backslashes with forward slashes in paths."""
            if schema.get("required"):
                raise TypeError(
                    f"Defaults must be specified in {model}, derived from {DefaultPathsModel}."
                )
            for (field, prop), type_ in zip(
                schema["properties"].items(),
                (field.type_ for field in model.__fields__.values()),
                strict=True,
            ):
                if not issubclass(type_, Path):
                    raise TypeError(
                        f"Field <{field}> is not Path-like in {model}, derived from {DefaultPathsModel}."
                    )
                default = prop.get("default")
                if isinstance(default, list | tuple):
                    default = [item.replace("\\", "/") for item in default]
                elif isinstance(default, dict):
                    default = {
                        key: value.replace("\\", "/") for key, value in default.items()
                    }
                else:
                    default = default.replace("\\", "/")
                prop["default"] = default

    @validator("*", always=True, pre=True, each_item=True)
    def check_pathlike(cls, value, field):
        """Check that the value is path-like."""
        if not issubclass(field.type_, Path):
            raise TypeError(
                f"Field is not Path-like in {cls}, derived from {DefaultPathsModel}."
            )
        return value

    @classmethod
    def get_paths(cls) -> dict[str, Any]:
        """Get the paths for this model."""
        return {
            key: value["default"] for key, value in cls.schema()["properties"].items()
        }


class CreatePathsModel(DefaultPathsModel):
    """Parent directories will be created for all fields in this model."""

    @validator("*", always=True, pre=True, each_item=True)
    def create_directories(cls, value):
        """Create directories associated with each value."""
        path = Path(value)
        if path.is_file():
            return value
        directory = path.parent if path.suffix else path
        directory.mkdir(parents=True, exist_ok=True)
        return value
