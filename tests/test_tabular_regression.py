"""The tabular rung is unchanged: it reproduces tests/regression/golden/tabular.npz
(recorded by tests/regression/make_golden_tabular.py), bit for bit."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression"))
import make_golden_tabular as golden   # noqa: E402


def test_tabular_matches_golden():
    ref, cur = np.load(golden.PATH), golden.run()
    assert set(ref.files) == set(cur)
    for k in ref.files:
        assert np.array_equal(ref[k], cur[k]), k
