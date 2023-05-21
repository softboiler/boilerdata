"""Test the pipeline."""

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from shutil import copytree
from types import ModuleType

import pandas as pd
import pytest

TEST_ROOT = Path("tests/test_root")


@pytest.mark.slow()
def test_pipeline(check, monkeypatch, tmp_path):
    """Test the pipeline."""

    def main():
        copytree(TEST_ROOT, tmp_path, dirs_exist_ok=True)
        stages = get_stages()
        for stage in stages:
            skip_asserts = ("schema",)
            if stage.name in skip_asserts:
                continue
            for result, expected in stage.expectations.items():
                with check:
                    assert_stage_result(result, expected)

    def get_stages():
        """Test the pipeline by patching constants before importing stages."""

        monkeypatch.setenv("DYNACONF_APP_FOLDER", "tests/.propshop")

        import boilerdata

        monkeypatch.setattr(boilerdata, "DVC_BASE", tmp_path)
        monkeypatch.setattr(boilerdata, "PARAMS_FILE", tmp_path / "params.yaml")

        from boilerdata.models.project import Project

        proj = Project.get_project()

        from boilerdata.stages import pipeline, schema
        from boilerdata.stages.prep import parse_benchmarks, runs

        @dataclass
        class Stage:
            """Results of running a pipeline stage.

            Args:
                module: The module corresponding to this pipeline stage.
                result_paths: The directories or a single file produced by the stage.
                tmp_path: The results directory.

            Attributes:
                name: The name of the pipeline stage.
                expectations: A mapping from resulting to expected files.
            """

            module: InitVar[ModuleType]
            result_paths: InitVar[tuple[Path, ...]]
            tmp_path: InitVar[Path]

            name: str = field(init=False)
            expectations: dict[Path, Path] = field(init=False)

            def __post_init__(
                self, module: ModuleType, result_paths: tuple[Path, ...], tmp_path: Path
            ):
                self.name = module.__name__.removeprefix(f"{module.__package__}.")
                module.main(proj)
                results: list[Path] = []
                expectations: list[Path] = []
                for path in result_paths:
                    expected = TEST_ROOT / path.relative_to(tmp_path)
                    if expected.is_dir():
                        results.extend(sorted(path.iterdir()))
                        expectations.extend(sorted(expected.iterdir()))
                    else:
                        results.append(path)
                        expectations.append(expected)
                self.expectations = dict(zip(results, expectations, strict=True))

        return [
            Stage(module, result_paths, tmp_path)
            for module, result_paths in {
                schema: (proj.dirs.project_schema,),
                runs: (proj.dirs.runs,),
                parse_benchmarks: (proj.dirs.benchmarks_parsed,),
                pipeline: (proj.dirs.results,),
            }.items()
        ]

    main()


def assert_stage_result(result_file: Path, expected_file: Path):
    """Assert that the result of a stage is as expected.

    Args:
        result_file: The file produced by the stage.
        expected_file: The file that the stage should produce.

    Raises:
        AssertionError: If the result is not as expected.
    """
    if expected_file.suffix == ".csv":
        result_df = pd.read_csv(result_file)
        expected_df = pd.read_csv(expected_file)
        if any(part in expected_file.stem for part in ["benchmarks", "runs"]):
            pd.testing.assert_series_equal(
                result_df.T_4, expected_df.T_4, rtol=1, atol=1
            )
        else:
            pd.testing.assert_series_equal(result_df.T_s, expected_df.T_s)
    else:
        assert result_file.read_bytes() == expected_file.read_bytes()
