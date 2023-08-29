"""Tests."""

from importlib import import_module

import pytest
from ploomber_engine.ipython import PloomberClient

from tests import STAGES


@pytest.mark.slow()
def test_nb_stages(nb_client: PloomberClient):
    """Test that notebook pipeline stages can run."""
    nb_client.execute()


@pytest.mark.slow()
@pytest.mark.usefixtures("_tmp_workdir")
@pytest.mark.parametrize("stage", STAGES)
def test_stages(stage: str):
    """Test that stages can run."""
    import_module(stage).main()


@pytest.mark.usefixtures("_tmp_workdir")
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
    from boilerdata import syms

    module_vars = vars(syms)
    sym_group = module_vars[group_name]
    symvars = {
        var: sym
        for var, sym in module_vars.items()
        if var in [group_sym.name for group_sym in sym_group]
    }
    assert all(var == sym.name for var, sym in symvars.items())
