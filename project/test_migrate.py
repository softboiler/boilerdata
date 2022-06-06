from pathlib import Path

from pytest import mark as m
import yaml

from boilerdata.models.project import Project
from migrate import migrate_1, migrate_2, migrate_3

# * -------------------------------------------------------------------------------- * #
# * MIGRATIONS


@m.skip("outdated")
def test_migrate_3(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project = Project.get_project()
    project.results_file = get_old_file(old_commit)
    migrate_3(
        project,
        schema_path := tmp_path / "columns_schema.json",
        columns_path := tmp_path / "columns.yaml",
    )

    result = schema_path.read_text(encoding="utf-8")
    expected = Path("project/tests/migrate/migrate_3/columns_schema.json").read_text(
        encoding="utf-8"
    )
    assert result == expected

    result = yaml.safe_load(columns_path.read_text(encoding="utf-8"))
    expected = yaml.safe_load(
        Path("project/tests/migrate/migrate_3/columns.yaml").read_text(encoding="utf-8")
    )
    assert result == expected


@m.skip("outdated")
def test_migrate_2(tmp_path):
    old_commit = "b4ddee04a3d7aee2a81eaed68f3b016873f924d1"
    project = Project.get_project()
    project.results_file = get_old_file(old_commit)
    migrate_2(project, path := tmp_path / "columns.py")
    result = path.read_text(encoding="utf-8")
    expected = Path("project/tests/migrate/migrate_2/columns.py").read_text(
        encoding="utf-8"
    )
    assert result == expected


def test_migrate_1(tmp_path):
    base = Path("project/tests/migrate/migrate_1")
    migrate_1(base / "project.yaml", tmp_path / "trials.yaml")
    result = yaml.safe_load((tmp_path / "trials.yaml").read_text(encoding="utf-8"))
    expected = yaml.safe_load(Path(base / "trials.yaml").read_text(encoding="utf-8"))
    assert result == expected


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_old_file(old_commit):
    project = Project.get_project()
    try:
        return next((project.dirs.results / "Old").glob(f"results_*_{old_commit}.csv"))
    except StopIteration as exception:
        raise StopIteration("Old results file is missing.") from exception
