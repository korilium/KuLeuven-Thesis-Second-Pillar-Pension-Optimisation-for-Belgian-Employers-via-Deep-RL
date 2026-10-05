"""Shared small-grid fixtures: big enough to exercise every branch, small enough
that the whole suite runs in well under a minute."""
import numpy as np
import pytest

import pension.dp as dp
from pension.params import DEFAULT


@pytest.fixture(scope="session")
def grid():
    return dict(Fg=dp.make_F_grid(n=31), rg=dp.make_rho_grid(n=25), ag=dp.make_a_grid(n=7), nq=3)


@pytest.fixture(scope="session")
def entry():
    """The new-plan entry cohort of the suites, 2000 careers."""
    R0, L0, S0 = dp.new_plan_init(2000, np.random.default_rng(7))
    return dict(R0=R0, L0=L0, S0=S0)


@pytest.fixture(scope="session")
def p():
    return DEFAULT
