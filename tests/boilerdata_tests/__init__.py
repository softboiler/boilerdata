"""Helper functions for tests."""

from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet
from boilercore.paths import get_module_rel, walk_module_paths, walk_modules

BOILERDATA = Path("src") / "boilerdata"
STAGES_DIR = BOILERDATA / "stages"
stages: list[ParameterSet] = []
for module in (f"boilerdata.{module}" for module in walk_modules(STAGES_DIR)):
    rel = get_module_rel(module, "stages")
    stages.append(
        pytest.param(
            module, id=rel, marks=[pytest.mark.skip] if rel in {"originlab"} else []
        )
    )
nbs_to_execute: list[ParameterSet] = [
    pytest.param(path, id=str(path.relative_to(STAGES_DIR)))
    for path in list(walk_module_paths(STAGES_DIR, suffixes=[".ipynb"]))
]
