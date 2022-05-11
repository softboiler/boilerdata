from pathlib import Path
from boilerdata.configs import load_config, dump_model

from dump_schema import MyModel

path = Path("examples/enums")

(config, schema_directive) = load_config(path / "test_in.toml", MyModel)
result = load_config(path / "test_in.toml", MyModel)

dump_model(path / "test_out.toml", *result)
