from pathlib import Path

from dump_schema import MyModel

from boilerdata.configs import dump_model, load_config

path = Path("examples/enums")

(config, schema_directive) = load_config(path / "test_in.toml", MyModel)
result = load_config(path / "test_in.toml", MyModel)

dump_model(path / "test_out.toml", *result)
