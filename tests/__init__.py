"""Helper functions for tests."""

from pathlib import Path
from typing import Any

import pytest
from boilercore.testing import get_module_rel, walk_modules

BOILERDATA = Path("src") / "boilerdata"
STAGES_DIR = BOILERDATA / "stages"
NOTEBOOK_STAGES = list(STAGES_DIR.glob("[!__]*.ipynb"))
STAGES: list[Any] = []
for module in walk_modules(STAGES_DIR, BOILERDATA):
    rel_to_stages = get_module_rel(module, "stages")
    if rel_to_stages in {"common", "literature", "modelfun", "originlab"}:
        marks = [pytest.mark.skip]
    else:
        marks = []
    STAGES.append(pytest.param(module, marks=marks))
