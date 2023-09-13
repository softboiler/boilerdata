"""Tests."""

from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet
from boilercore.paths import get_module_rel, walk_module_paths, walk_modules


def approx(*args):
    """Approximate equality with a relative tolerance of 1e-3."""
    return pytest.approx(*args, rel=1e-3)


BOILERDATA = Path("src") / "boilerdata"
STAGES_DIR = BOILERDATA / "stages"

stages: list[ParameterSet] = []
for module in walk_modules(STAGES_DIR, BOILERDATA):
    rel = get_module_rel(module, "stages")
    stages.append(
        pytest.param(
            module, id=rel, marks=[pytest.mark.skip] if rel in {"originlab"} else []
        )
    )

nbs = list(walk_module_paths(STAGES_DIR, BOILERDATA, suffix=".ipynb"))
MODELFUN = Path("src/boilerdata/stages/modelfun.ipynb")
nbs_to_check = [MODELFUN]
nbs_to_execute: list[ParameterSet] = [
    pytest.param(path, id=str(path.relative_to(STAGES_DIR)))
    for path in nbs
    if path not in nbs_to_check
]
