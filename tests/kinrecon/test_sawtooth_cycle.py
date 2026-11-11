"""Periodic sawtooth state of ``bouquet.kinrecon.current.sawtooth`` on a synthetic cylinder.

Bounds and their sources:
* the cycle criterion (P1) is the decided 1e-9, for both the matrix map and real stepping;
* the one-period matrix map equals ``advance`` to round-off (1e-12 relative; the prototype measured 3.6e-15);
* Newton converges in at most 6 iterations away from the q0 = 1 kink (the prototype's measured 2-6);
* Newton vs brute-force cycle stepping: both are fixed points of the same map to 1e-9, so they differ by at most
  ~1e-9/(1 - r) with r the contraction rate of the map; r <= 0.99 in every case of the development checks, so the
  bound is 1e-7 of the mixing-region flux depth;
* smoothness: symmetric finite-difference asymmetry |y+ + y- - 2 y0| / |y+ - y-| <= 0.1 at a 1e-2 relative step
  (acceptance rule of the scoping report; the prototype measured <= 0.073);
* onset of q = 1 (relaxed q0 -> 1-): l_i is C1 (its change is O((1 - q0)^2) or smaller), so its one-sided
  difference quotient from the onset must shrink when the step shrinks fourfold: ratio <= 0.5 (exponent >= 1.5;
  2 expected); q0 itself has an intrinsic slope break there and is not bounded;
* window bookkeeping: identities of the trapezoid rule, to 1e-12 relative (round-off).
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from bouquet.kinrecon.current import CurrentDiffusion, CurrentState, FluxSurfaceGeometry, IpBoundary
from bouquet.kinrecon.current.sawtooth import (CYCLE_TOL, _affine_power, _step_matrices, reconnect,
                                               solve_periodic_orbit)

A, R0, B0 = 0.6, 2.0, 2.0
RF = np.linspace(0.0, 1.0, 201)


def make_solver(ip=0.8e6, sig_scale=1.0):
    """Synthetic periodic cylinder; conductivity ~ Te^1.5 with a peaked Te shape (arbitrary units)."""
    geo = FluxSurfaceGeometry.cylinder(A, R0, B0, n=401)
    rho = geo.rho
    sig = 1.0e8 * sig_scale * (0.033 + 0.967 * (1 - rho ** 2) ** 1.5) ** 1.5
    return CurrentDiffusion(geo, (rho, sig), IpBoundary(ip))


def q_at(solver, rhos):
    return lambda s: np.interp(rhos, RF, solver.profiles(s)["q"])


@pytest.fixture(scope="module")
def base():
    sv = make_solver()
    return sv, solve_periodic_orbit(sv, RF, 0.37, 0.05)


def test_matrix_map_matches_advance():
    sv = make_solver()
    S, s0 = _step_matrices(sv, RF, 0.05 / 40, 0.0, "trbdf2")
    M, c = _affine_power(S, s0, 40)
    x = sv.steady_state(RF).psi + 1e-3 * np.cos(3 * np.pi * 0.5 * (RF[1:] + RF[:-1]))
    ref = sv.advance(CurrentState(RF, x, 0.0, 0.0), 0.05, nsteps=40).psi
    assert np.max(np.abs(M @ x + c - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_periodic_orbit_converges(base):
    sv, o = base
    assert o.converged and o.residual < CYCLE_TOL and o.residual_stepping < CYCLE_TOL
    assert o.n_newton <= 6 and o.n_plain_cycles == 0
    assert o.crash.active and o.crash.topology == "monotonic"
    assert len(o.states) == 41 and o.states[-1].time == pytest.approx(0.05, abs=1e-15)
    # the stored orbit closes: reconnecting the pre-crash state returns the post-crash state (gauge removed)
    post, _ = reconnect(o.pre_crash, sv.geometry_at(0.0).phi_b, 0.37)
    d = (post.psi - post.psi[-1]) - (o.post_crash.psi - o.post_crash.psi[-1])
    assert np.max(np.abs(d)) / o.crash.depth < CYCLE_TOL
    with pytest.raises(ValueError):
        solve_periodic_orbit(sv, RF, 0.37, 0.05, tol=1e-8)          # looser than the decided criterion


def test_orbit_agrees_with_brute_force_cycles(base):
    sv, o = base
    phi_b = sv.geometry_at(0.0).phi_b
    st = sv.steady_state(RF)
    z = st.psi - st.psi[-1]
    for k in range(3000):
        pre = sv.advance(CurrentState(RF, z, 0.0, 0.0), 0.05, nsteps=40)
        post, rec = reconnect(pre, phi_b, 0.37)
        zn = post.psi - post.psi[-1]
        change = np.max(np.abs(zn - z)) / rec.depth
        z = zn
        if change < 1e-12:
            break
    assert change < 1e-12
    zo = o.post_crash.psi - o.post_crash.psi[-1]
    assert np.max(np.abs(z - zo)) / o.crash.depth <= 1e-7


def test_fast_full_reconnection_corner():
    """f >= 0.75 with a period <= 10 ms: the corner where the single-surface prototype had no fixed point
    (period-2 orbit).  The pre-crash q has more than one q < 1 region here; the orbit must converge to the
    decided criterion and be a true fixed point (period 1)."""
    sv = make_solver()
    phi_b = sv.geometry_at(0.0).phi_b
    for f, tau in ((1.0, 0.005), (0.75, 0.01)):
        o = solve_periodic_orbit(sv, RF, f, tau)
        assert o.converged, (f, tau, o.residual)
        z = o.post_crash.psi - o.post_crash.psi[-1]
        for _ in range(2):                                         # two further cycles stay on the orbit
            pre = sv.advance(CurrentState(RF, z, 0.0, 0.0), tau, nsteps=40)
            post, rec = reconnect(pre, phi_b, f)
            zn = post.psi - post.psi[-1]
            assert np.max(np.abs(zn - z)) / rec.depth < CYCLE_TOL
            z = zn


def _outputs(sv, o):
    return np.concatenate(([o.cycle_average(lambda s: sv.profiles(s)["q"][0]),
                            o.cycle_average(lambda s: sv.profiles(s)["li3"])],
                           o.cycle_average(q_at(sv, [0.05, 0.2, 0.4]))))


@pytest.mark.parametrize("which", ["f", "conductivity", "period"])
def test_finite_difference_smoothness(base, which):
    sv0, o0 = base
    y0 = _outputs(sv0, o0)
    h = 1e-2
    ys = []
    for e in (h, -h):
        if which == "f":
            sv, f, tau = sv0, 0.37 * (1 + e), 0.05
        elif which == "conductivity":                               # a Te scale: sigma x lambda^1.5
            sv, f, tau = make_solver(sig_scale=(1 + e) ** 1.5), 0.37, 0.05
        else:
            sv, f, tau = sv0, 0.37, 0.05 * (1 + e)
        o = solve_periodic_orbit(sv, RF, f, tau, z0=o0.z)
        assert o.converged
        ys.append(_outputs(sv, o))
    asym = np.abs(ys[0] + ys[1] - 2 * y0) / np.abs(ys[0] - ys[1])
    assert np.all(asym <= 0.1), asym


def test_onset_li_is_c1():
    def relaxed_q0(ip):
        sv = make_solver(ip=ip)
        return sv.profiles(sv.steady_state(RF))["q"][0]

    ip_star = brentq(lambda ip: relaxed_q0(ip) - 1.0, 0.5e6, 0.7e6, xtol=1e-3)

    def li(ip):
        sv = make_solver(ip=ip)
        o = solve_periodic_orbit(sv, RF, 0.37, 0.05)
        assert o.converged
        return o.cycle_average(lambda s: sv.profiles(s)["li3"])

    l0 = li(ip_star * (1 - 1e-6))                                   # just below onset: no crash
    # steps inside the asymptotic range (the l_i curve turns over a few percent past onset)
    d1 = (li(ip_star * (1 + 1e-3)) - l0) / 1e-3
    d2 = (li(ip_star * (1 + 2.5e-4)) - l0) / 2.5e-4
    assert abs(d2) <= 0.5 * abs(d1)


def test_window_bookkeeping(base):
    sv, o = base
    g = q_at(sv, [0.0, 0.1, 0.3])
    cyc = o.cycle_average(g)
    vals = np.array([g(s) for s in o.states])
    tol = 1e-12 * np.max(np.abs(cyc))
    for phase in (0.0, 0.3, 0.77):
        for k in (1, 2):                                           # whole periods give the cycle average
            assert np.max(np.abs(o.window_average(g, k * o.period, phase, values=vals) - cyc)) <= tol
    # a window inside one cycle equals the trapezoid rule on the stored states between the same phases
    t = o.times - o.times[0]
    w = o.window_average(None, 0.5 * o.period, 0.25, values=vals)
    i0, i1 = 10, 30
    ref = np.trapezoid(vals[i0:i1 + 1], t[i0:i1 + 1], axis=0) / (t[i1] - t[i0])
    assert np.max(np.abs(w - ref)) <= tol
    # the range over window starts brackets the cycle average and collapses for whole-period windows
    rng = o.window_average_over_phases(g, 0.3 * o.period)
    assert np.all(rng["min"] <= cyc) and np.all(rng["max"] >= cyc)
    rng = o.window_average_over_phases(g, o.period)
    assert np.max(rng["max"] - rng["min"]) <= tol
    assert o.window_average(lambda s: 1.0, 0.123) == pytest.approx(1.0, rel=1e-14)
    # phase states: stored phases are returned as stored
    assert np.array_equal(o.phase_state(0.5).psi, o.states[20].psi)
