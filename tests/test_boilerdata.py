"""Test the pipeline."""

from dataclasses import dataclass
import platform
from pathlib import Path
from unittest.mock import PropertyMock

import pytest
from testbook import testbook

if platform.system() == "Windows":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

TEST_DATA = Path("tests/data")


def test_pest(patched_modules):
    ...


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
    patched_modules(1)[module].main()


def test_book(patched_modules):
    Params = patched_modules[0]

    with testbook("pest.ipynb") as tb:  # noqa: SIM117
        tb.execute_cell(0)
        with tb.patch("__main__.Params", create=Params, spec=Params):
            tb.execute_cell(1)
            print(tb.ref("hello")())
