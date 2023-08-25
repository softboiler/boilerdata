"""Tests."""

from importlib import import_module

import pytest

from tests import NOTEBOOK_STAGES, STAGES


@pytest.mark.usefixtures("tmp_project")
@pytest.mark.parametrize(
    "group_name",
    [
        "params",
        "inputs",
        "intermediate_vars",
        "functions",
    ],
)
def test_syms(group_name: str):
    """Test that declared symbolic variables are assigned to the correct symbols."""
    from boilerdata import stages

    module_vars = vars(stages)
    sym_group = module_vars[group_name]
    symvars = {
        var: sym
        for var, sym in module_vars.items()
        if var in [group_sym.name for group_sym in sym_group]
    }
    assert all(var == sym.name for var, sym in symvars.items())


@pytest.mark.usefixtures("tmp_project")
@pytest.mark.parametrize("stage", STAGES)
def test_stages(stage: str):
    """Test that stages can run."""
    import_module(stage).main()


@pytest.mark.slow()
@pytest.mark.usefixtures("_tmp_project_with_nb_stages")
@pytest.mark.parametrize(
    "stage",
    [stage.stem for stage in NOTEBOOK_STAGES],
)
def test_nb_stages(stage: str):
    """Test that notebook pipeline stages can run."""
    import_module(stage)
