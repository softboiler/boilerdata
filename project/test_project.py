from pathlib import Path

import yaml

from migrate import migrate_1


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text())
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text())
    assert result == expected


# DATA = Path("tests/data")
# RESULT = Path("fitted.csv")

# @contextmanager
# def working_directory(path: Path):
#     original_working_directory = getcwd()
#     try:
#         chdir(path)
#         yield
#     finally:
#         chdir(original_working_directory)

# @m.skip
# def test_run(tmp_path):
#     """Ensure the same result is coming out of the pipeline as before."""
#     test_data = tmp_path / "data"
#     copytree(DATA, test_data)
#     with working_directory(test_data):
#         pipeline.run()
#     result = pd.read_csv(test_data / RESULT)
#     expected = pd.read_csv(DATA / RESULT)
#     pd.testing.assert_frame_equal(result, expected)
