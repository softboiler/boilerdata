"""Tests."""

from pathlib import Path
from typing import Any

import pytest
from boilercore.paths import get_module_rel, walk_module_paths, walk_modules

BOILERDATA = Path("src") / "boilerdata"
STAGES_DIR = BOILERDATA / "stages"
STAGES: list[Any] = []
for module in walk_modules(STAGES_DIR, BOILERDATA):
    rel = get_module_rel(module, "stages")
    STAGES.append(
        pytest.param(
            module, id=rel, marks=[pytest.mark.skip] if rel in {"originlab"} else []
        )
    )
NOTEBOOK_STAGES = [
    pytest.param(path, id=str(path.relative_to(STAGES_DIR)))
    for path in walk_module_paths(STAGES_DIR, BOILERDATA, suffix=".ipynb")
]
