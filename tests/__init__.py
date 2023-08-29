"""Helper functions for tests."""


from pathlib import Path

import pytest
from boilercore.testing import get_module_rel, walk_modules

BOILERDATA = Path("src") / "boilerdata"
STAGES_DIR = BOILERDATA / "stages"
NOTEBOOK_STAGES = list(STAGES_DIR.glob("[!__]*.ipynb"))
STAGES = [
    pytest.param(
        module,
        marks=[pytest.mark.skip]
        if get_module_rel(module, "stages") in {"originlab"}
        else [],
    )
    for module in walk_modules(STAGES_DIR, BOILERDATA)
]
