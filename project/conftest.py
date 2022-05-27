from pytest import fixture

from boilerdata.utils import allow_extra
from utils import get_project


@fixture
def tmp_proj(tmp_path):
    old_proj = get_project()
    new_results_file = tmp_path / old_proj.dirs.results_file.relative_to(
        old_proj.dirs.base
    )
    new_results_file.parent.mkdir()  # Needed because copying won't run the validators
    tmp_proj = old_proj.copy(
        deep=True,
        update=dict(
            dirs=old_proj.dirs.copy(
                update=dict(
                    results_file=new_results_file,
                )
            )
        ),
    )

    # In case the tmp_path needs to be accessed by the test
    with allow_extra(tmp_proj):
        tmp_proj.dirs.tmp_path = tmp_path

    return tmp_proj
