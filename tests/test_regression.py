"""The refactor contract: the current code reproduces the golden outputs recorded
from the pre-refactor model (tests/regression/make_golden.py).

Policies and every action-valued output must be IDENTICAL; values may move by
floating-point reassociation only (<= 1e-12 relative).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression"))
import make_golden as golden   # noqa: E402

RTOL = 1e-12


def _current(model, objective):
    """The same case through the CURRENT API. This adapter is the one place that
    follows the refactor; the recorded files never change."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Envs"))
    import DynPro as dp
    saved = dp.RATE_MODEL
    try:
        return golden.run(dp, model, objective)
    finally:
        dp.RATE_MODEL = saved


@pytest.mark.parametrize("model,objective", golden.CASES, ids=[f"{m}-{o}" for m, o in golden.CASES])
def test_matches_golden(model, objective):
    ref = np.load(golden.path(model, objective))
    cur = _current(model, objective)
    assert set(ref.files) == set(cur), set(ref.files) ^ set(cur)
    for k in ref.files:
        a, b = ref[k], np.asarray(cur[k])
        assert a.shape == b.shape, (k, a.shape, b.shape)
        name = k.split("_", 1)[-1] if k.startswith(("opt_", "flat_")) else k
        if name in golden.ACTION_KEYS:
            assert np.array_equal(a, b, equal_nan=True), f"{k}: actions differ"
        else:
            scale = np.maximum(np.abs(a), 1e-300)
            rel = np.nanmax(np.abs(a - b) / scale) if a.size else 0.0
            assert rel <= RTOL, f"{k}: max relative diff {rel:.2e}"
