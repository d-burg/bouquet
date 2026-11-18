"""The in-process speed options as the default, without the solver: table inputs copied when their functions are
made, the coefficient memo only on inputs known to be pure, the options in effect in every fit and series record,
and the default itself (``bouquet.kinrecon.speed``)."""

import numpy as np
import pytest

from bouquet.kinrecon import speed as SP


# ---------------------------------------------------------------------------------------------------- 1. table copies
@pytest.mark.parametrize("form", ["2", "3"])
@pytest.mark.parametrize("memo", [False, True])
def test_profile_table_changed_in_place_afterwards_changes_no_value(form, memo):
    from bouquet.kinrecon.current import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    times = np.array([0.0, 0.02])
    table = np.vstack((vals, 2 * vals))
    spec = (rho, vals) if form == "2" else (times, rho, table)
    pts = np.linspace(0.05, 0.95, 37)
    with SP.speed_scope(SP.SpeedSettings(memo=memo, mesh_index=False, reuse_stats=False)):
        f = DF.as_profile_function(spec, log=True)
        before = [f(t, pts) for t in (0.004, 0.011)]
        # the caller changes every table array in place after the function was made
        times[1] = 2.0
        rho[:] = rho[::-1] * 0.5
        vals *= 3.0
        table *= 5.0
        after = [f(t, pts) for t in (0.004, 0.011)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_time_table_changed_in_place_afterwards_changes_no_value():
    from bouquet.kinrecon.current import diffusion as DF
    times, values = np.array([0.0, 0.01, 0.02]), np.array([1.0e6, 1.2e6, 1.5e6])
    for f in (DF.as_time_function((times, values)), DF.IpBoundary((times, values))):
        before = [f(t) for t in (0.004, 0.013)]
        saved = times.copy(), values.copy()
        times[1], values[:] = 0.019, 7.0
        assert [f(t) for t in (0.004, 0.013)] == before
        times[:], values[:] = saved
