from pathlib import Path
from shutil import copy, copytree
from types import ModuleType

import pytest

TEST_DATA = Path("tests/data")


@pytest.fixture()
def patched_modules(monkeypatch, tmp_path) -> dict[str, ModuleType]:
    """Test the pipeline by patching constants before importing stages."""

    monkeypatch.setenv("DYNACONF_APP_FOLDER", f"{TEST_DATA / '.propshop'}")

    import boilerdata

    test_params = tmp_path / "params.yaml"
    monkeypatch.setattr(boilerdata, "PARAMS_FILE", test_params)
    copy("params.yaml", test_params)

    test_config = tmp_path / "config"
    monkeypatch.setattr(boilerdata, "AXES_CONFIG", test_config / "axes.yaml")
    monkeypatch.setattr(boilerdata, "TRIAL_CONFIG", test_config / "trials.yaml")
    copytree(TEST_DATA / "config", test_config)

    test_data = tmp_path / "data"
    monkeypatch.setattr(boilerdata, "DATA_DIR", test_data)
    copytree(TEST_DATA / "data", test_data)

    from boilerdata.stages import pipeline
    from boilerdata.stages.prep import parse_benchmarks, runs

    return {
        module.__name__.removeprefix(f"{module.__package__}."): module
        for module in (
            parse_benchmarks,
            pipeline,
            runs,
        )
    }
