"""Positive frame, residual sign and current sheets of non-monotonic q in ``bouquet.kinrecon.current.sawtooth``.

These tests target defects found by an independent review (they fail on the code before the fix):

* the periodic-orbit criterion (P1) divided by a depth that is negative for Ip < 0, so a negative residual
  passed the 1e-9 test vacuously; the positive frame is now enforced and the residual is non-negative;
* for non-monotonic q the slope jump at the new position of the axis level chi(0) was never smoothed.

Bounds and their sources:
* frame checks are exceptions (no tolerance);
* the residual reported by the orbit must equal the independently recomputed one-cycle change of the state
  (real stepping), divided by |psi(rho_mix) - psi(0)|, to round-off (1e-12 relative) and be >= 0;
* sheet smoothing: the slope jump of dpsi/drho at the axis-level position is measured by one-sided quadratic fits
  of the face gradients (an estimator independent of the operator's internals); the position itself comes from an
  independent sort-based rearrangement of chi on a fine grid.  Before the fix the smoothed (w = 0.05) output kept
  the whole unsmoothed jump (ratio 1.0); the bound requires at least 95 % of it to be cancelled (ratio <= 0.05).
  The measured ratio after the fix is 0.5-0.7 %, the resolution of the estimator.  The sheet listed in the
  record must sit at the independent position to 1e-3 in rho (the interpolant error at 400 cells is ~h^2 = 6e-6
  in s).
"""
import numpy as np
import pytest
from scipy.integrate import cumulative_simpson

from bouquet.kinrecon.current import (CurrentDiffusion, CurrentState, FluxSurfaceGeometry, IpBoundary,
                                      PositiveFrameError)
from bouquet.kinrecon.current.sawtooth import _anderson, reconnect, reconnect_psi, solve_periodic_orbit

PHI_B = 1.0
CF = PHI_B / (2.0 * np.pi)
N = 400
RF = np.linspace(0.0, 1.0, N + 1)
RC = 0.5 * (RF[1:] + RF[:-1])
S_FINE = np.linspace(0.0, 1.0, 400001)

CASES = {
    # q0 > 1 with a deep off-axis q < 1 region ('reversed'); minimum of chi interior, axis level below the top
    "reversed": lambda s: 1.15 - 0.4 * np.exp(-((np.sqrt(s) - 0.3) / 0.1) ** 2) + 1.5 * s,
    # q0 < 1 with a second q < 1 region ('multiple'); interior minimum of chi below chi(0)
    "multiple": lambda s: 0.95 + 2.0 * s - 0.3 * np.exp(-((np.sqrt(s) - 0.4) / 0.05) ** 2),
}


def _psi(qfun):
    psi_f = CF * cumulative_simpson(1.0 / qfun(S_FINE), x=S_FINE, initial=0.0)
    return np.interp(RC ** 2, S_FINE, psi_f), psi_f


def _slope_jump(post, rho_k, m=6):
    g = np.diff(post) / np.diff(RC)
    xf = RF[1:-1]
    i = np.searchsorted(xf, rho_k)
    left = np.polyfit(xf[i - m - 1:i - 1], g[i - m - 1:i - 1], 2)
    right = np.polyfit(xf[i + 1:i + m + 1], g[i + 1:i + m + 1], 2)
    return np.polyval(right, rho_k) - np.polyval(left, rho_k)


def _axis_level_position(psi_f, rho_mix):
    """rho of mu(chi(0)) on D = [0, s_mix], by counting fine samples (independent of the operator)."""
    chi = psi_f - CF * S_FINE
    sm = rho_mix ** 2
    sel = S_FINE <= sm
    return float(np.sqrt(np.mean(chi[sel] > chi[0]) * sm))


@pytest.mark.parametrize("kind", sorted(CASES))
def test_axis_level_sheet_is_smoothed(kind):
    psi_c, psi_f = _psi(CASES[kind])
    post0, rec0 = reconnect_psi(psi_c, RF, PHI_B, 1.0, 0.0)
    postw, recw = reconnect_psi(psi_c, RF, PHI_B, 1.0, 0.05)
    assert recw.topology == kind
    rk = _axis_level_position(psi_f, rec0.rho_mix)
    j0 = _slope_jump(post0, rk)
    jw = _slope_jump(postw, rk)
    scale = np.interp(rk, RF[1:-1], np.diff(postw) / np.diff(RC))
    assert abs(j0) > 5e-3 * abs(scale)                 # the fixture really has a sheet there (1.3-2.5 % of psi')
    assert abs(jw) <= 0.05 * abs(j0), (j0, jw)
    # the record lists that sheet at the independent position, with the unsmoothed jump's sign
    near = [s for s in recw.sheets if abs(s["rho"] - rk) < 1e-3]
    assert len(near) == 1 and np.sign(near[0]["jump_dpsi_drho"]) == np.sign(j0)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_every_listed_sheet_is_smoothed(kind):
    """No residual slope jump at any sheet position of the record (w = 0.05), same estimator and bound."""
    psi_c, _ = _psi(CASES[kind])
    post0, _ = reconnect_psi(psi_c, RF, PHI_B, 1.0, 0.0)
    postw, recw = reconnect_psi(psi_c, RF, PHI_B, 1.0, 0.05)
    for sh in recw.sheets:
        j0 = _slope_jump(post0, sh["rho"])
        jw = _slope_jump(postw, sh["rho"])
        assert abs(jw) <= 0.05 * abs(j0), (sh, j0, jw)


# ----------------------------------------------------------------------------- positive frame
def _cylinder_solver(ip):
    geo = FluxSurfaceGeometry.cylinder(0.6, 2.0, 2.0, n=401)
    sig = 1.0e8 * (0.033 + 0.967 * (1 - geo.rho ** 2) ** 1.5) ** 1.5
    return CurrentDiffusion(geo, (geo.rho, sig), IpBoundary(ip))


def test_negative_ip_is_refused_everywhere():
    sv = _cylinder_solver(-0.8e6)
    rf = np.linspace(0.0, 1.0, 201)
    with pytest.raises(PositiveFrameError):
        sv.steady_state(rf)
    with pytest.raises(PositiveFrameError):
        solve_periodic_orbit(sv, rf, 0.5, 0.05)
    # a state outside the positive frame (psi falling outward) is refused by the operator itself
    psi_c, _ = _psi(CASES["multiple"])
    with pytest.raises(PositiveFrameError):
        reconnect_psi(-psi_c, RF, PHI_B, 1.0, 0.05)
    with pytest.raises(PositiveFrameError):
        reconnect_psi(psi_c, RF, -PHI_B, 1.0, 0.05)
    with pytest.raises(PositiveFrameError):
        reconnect(CurrentState(RF, -psi_c, 0.0), PHI_B)
    # zero Ip is allowed for the linear solver (not for the orbit)
    sv0 = _cylinder_solver(0.0)
    sv0.steady_state(rf)
    with pytest.raises(PositiveFrameError):
        solve_periodic_orbit(sv0, rf, 0.5, 0.05)


def test_residual_is_nonnegative_and_independently_recomputed():
    sv = _cylinder_solver(0.8e6)
    rf = np.linspace(0.0, 1.0, 201)
    o = solve_periodic_orbit(sv, rf, 0.37, 0.05)
    assert o.converged
    assert o.residual >= 0.0 and o.residual_stepping >= 0.0
    assert all(h >= 0.0 for h in o.residual_history)
    # one real cycle from the stored post-crash state, normalized by |psi(rho_mix) - psi(0)| of the pre-crash
    # state computed here from the record's own definition (positive in the positive frame)
    pre = sv.advance(CurrentState(rf, o.z, 0.0, 0.0), 0.05, nsteps=o.nsteps)
    post, rec = reconnect(pre, sv.geometry_at(0.0).phi_b, 0.37)
    assert rec.depth > 0
    change = np.max(np.abs((post.psi - post.psi[-1]) - o.z)) / abs(rec.depth)
    assert change == pytest.approx(o.residual_stepping, rel=1e-12, abs=1e-18)


def test_anderson_plain_step_failure_is_reported_not_raised():
    """A ValueError from the physical map in a plain (non-accelerated) Anderson step ends the iteration with the
    best iterate; the caller then reports non-convergence (previously the exception aborted the solve)."""
    calls = {"n": 0}

    def resid(z):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("the reconnection zone reaches the last two cells")
        r = np.full_like(z, 1e-3)
        return r, z, {}, 1e-3

    z, nrm, n_ev, _, _ = _anderson(resid, np.zeros(5), 1e-3, 1e-9, 50, [], False)
    assert nrm == pytest.approx(1e-3) and n_ev == 2 and np.array_equal(z, np.zeros(5))
