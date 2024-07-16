"""Tests."""

import pytest


@pytest.mark.slow
def test_execute_nb(nb_client_to_execute):
    """Execute a notebook."""
    nb_client_to_execute.execute()


@pytest.mark.slow
def test_stage(main):
    """Test a stage."""
    main()
