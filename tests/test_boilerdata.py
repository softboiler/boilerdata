"""Test the pipeline."""

from pathlib import Path

import pytest

TEST_DATA = Path("tests/data")


@pytest.mark.slow()
@pytest.mark.parametrize(
    "module",
    [
        "parse_benchmarks",
        "pipeline",
        "runs",
    ],
)
def test_boilerdata(module, patched_modules):
    patched_modules[module].main()
