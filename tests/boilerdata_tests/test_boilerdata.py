import pytest


@pytest.mark.slow()
def execute_nb(nb_client_to_execute):
    """Execute a notebook."""
    nb_client_to_execute.execute()


@pytest.mark.slow()
def test_stage(main):
    """Test a stage."""
    main()


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
