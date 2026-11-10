"""Numerical properties of ``bouquet.kinrecon.current.diffusion``.

Time-convergence order of each scheme (TR-BDF2 and Crank-Nicolson: 2,
backward Euler: 1) with time-varying Ip, conductivity and non-inductive
current; exact current conservation under the Ip condition; steady state ==
long-time limit; state serialization round trips; sawtooth hooks; the
moving-boundary (dPhi_b/dt) term.  Synthetic geometry and profiles only.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon.current import (CurrentDiffusion, CurrentState, FluxSurfaceGeometry,  # noqa: E402
                                      GeometrySchedule, IpBoundary, VloopBoundary)

GEO = FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=301, fpol=lambda x: 3.4 * (1.0 - 0.02 * x ** 2))
T_END = 0.4


def sigma_t(t, rho):
    return (4.0e6 * (1.0 - 0.9 * rho ** 2) ** 1.5 + 1.0e5) * (1.0 + 0.3 * np.sin(2 * np.pi * t / T_END))


def jni_t(t, rho):
    return 2.0e5 * (1.0 + t / T_END) * np.exp(-((rho - 0.6) / 0.15) ** 2)


def ip_t(t):
    return 1.0e6 * (1.0 + 0.5 * t / T_END + 0.1 * np.sin(5.0 * t / T_END))


def _solver(bc=None, **kw):
    return CurrentDiffusion(GEO, sigma_t, bc or IpBoundary(ip_t), j_ni=jni_t, **kw)


def _start(sol, n=40):
    return sol.steady_state(CurrentState.uniform_faces(n), t=0.0)


@pytest.mark.parametrize("scheme,theta,order", [("trbdf2", None, 2.0), ("theta", 0.5, 2.0), ("theta", 1.0, 1.0)])
@pytest.mark.parametrize("bc", ["ip", "vloop"])
def test_time_convergence_order(scheme, theta, order, bc):
    boundary = IpBoundary(ip_t) if bc == "ip" else VloopBoundary(lambda t: 1.5 + 0.5 * np.sin(4 * t / T_END))
    sol = _solver(boundary)
    st0 = _start(sol)
    kw = dict(scheme=scheme, theta=theta if theta is not None else 0.5)
    # reference: the semi-discrete solution, approximated by 5120 TR-BDF2 steps for every scheme
    # (a 5120-step backward-Euler reference would itself carry ~6 % of the finest-step error)
    ref = sol.advance(st0, T_END, nsteps=5120, scheme="trbdf2")
    errs = []
    for ns in (40, 80, 160, 320):
        s = sol.advance(st0, T_END, nsteps=ns, **kw)
        errs.append(max(np.max(np.abs(s.psi - ref.psi)), abs(s.psi_boundary - ref.psi_boundary)))
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    # theory: p = 2 (TR-BDF2, CN), p = 1 (backward Euler); bound +-0.1, fixed before running
    assert order - 0.1 <= p[-1] <= order + 0.1, (errs, p)


def test_ip_boundary_conserves_total_current():
    sol = _solver()
    s = _start(sol)
    for k in range(1, 11):
        s = sol.advance(s, k * T_END / 10, nsteps=7)
        p = sol.profiles(s)
        assert abs(p["ip"] / ip_t(s.time) - 1.0) < 1e-13
        # volume integral of the current density: int <J_phi/R> V'/(2 pi) drho = Ip (telescoping, round-off)
        total = np.sum(p["jt_over_r"] * GEO.evaluate(p["rho_cells"], ["dvol_drho"])["dvol_drho"]
                       * np.diff(p["rho_faces"])) / (2 * np.pi)
        assert abs(total / ip_t(s.time) - 1.0) < 1e-12


@pytest.mark.parametrize("scheme,theta", [("trbdf2", 0.5), ("theta", 1.0)])
def test_steady_state_is_long_time_limit(scheme, theta):
    sol = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, IpBoundary(1.2e6),
                           j_ni=lambda t, r: 1.5e5 * np.exp(-((r - 0.6) / 0.15) ** 2))
    rf = CurrentState.uniform_faces(60)
    ss = sol.steady_state(rf)
    # start far from relaxed: flat q = 3 (hollow-ish current)
    st = CurrentState.from_q(GEO, np.linspace(0, 1, 30), 3.0 + 0.0 * np.linspace(0, 1, 30), rf)
    tau = 4e-7 * np.pi * 2.1e6 * 0.6 ** 2      # ~ core resistive time [s]
    out = sol.advance(st, 60 * tau, dt=0.1 * tau, scheme=scheme, theta=theta)
    a, b = sol.profiles(out), sol.profiles(ss)
    np.testing.assert_allclose(a["q"], b["q"], rtol=1e-9)
    np.testing.assert_allclose(a["v_loop"], ss.meta["v_loop"], rtol=1e-9)
    # psi up to its (time-growing) offset
    np.testing.assert_allclose(out.psi - out.psi_boundary, ss.psi - ss.psi_boundary, rtol=1e-9, atol=0)


def test_vloop_steady_state_holds_current():
    sol_ip = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, IpBoundary(1.0e6))
    ss = sol_ip.steady_state(CurrentState.uniform_faces(50))
    sol_v = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, VloopBoundary(ss.meta["v_loop"]))
    out = sol_v.advance(ss, 5.0, nsteps=50)
    assert abs(sol_v.profiles(out)["ip"] / 1.0e6 - 1.0) < 1e-10
    ss_v = sol_v.steady_state(CurrentState.uniform_faces(50))
    assert abs(ss_v.meta["ip"] / 1.0e6 - 1.0) < 1e-12


@pytest.mark.parametrize("suffix", [".json", ".npz", ".h5"])
def test_state_round_trip_and_restart(tmp_path, suffix):
    sol = _solver()
    s0 = _start(sol)
    s0.meta["note"] = "synthetic"
    s1 = sol.advance(s0, 0.1, nsteps=10)
    path = tmp_path / ("state" + suffix)
    s1.save(path)
    r = CurrentState.load(path)
    np.testing.assert_array_equal(r.psi, s1.psi)
    np.testing.assert_array_equal(r.rho_faces, s1.rho_faces)
    assert r.psi_boundary == s1.psi_boundary and r.time == s1.time and r.meta == s1.meta
    # persistence: continuing from the reloaded state is bit-identical to continuing from the in-memory one
    np.testing.assert_array_equal(sol.advance(s1, 0.2, nsteps=10).psi, sol.advance(r, 0.2, nsteps=10).psi)
    assert CurrentState.from_dict(s1.to_dict()).psi.tolist() == s1.psi.tolist()
    # splitting an interval in two calls equals one call with the same steps up to round-off: the
    # stage times t0 + k h differ in the last bit between the two time grids (0.1 + 0.01 k vs 0.01 (10 + k))
    through = sol.advance(s0, 0.2, nsteps=20)
    restarted = sol.advance(r, 0.2, nsteps=10)
    np.testing.assert_allclose(restarted.psi, through.psi, rtol=0, atol=1e-13 * np.max(np.abs(through.psi)))


def test_sawtooth_hooks_are_called_and_identity_is_neutral():
    calls = []

    def post(state, t, solver):
        calls.append(t)
        return state

    sol = _solver()
    s0 = _start(sol)
    plain = sol.advance(s0, 0.1, nsteps=8)
    hooked = sol.advance(s0, 0.1, nsteps=8, post_step=post)
    assert len(calls) == 8 and abs(calls[-1] - 0.1) < 1e-15
    np.testing.assert_array_equal(plain.psi, hooked.psi)
    ident = _solver(resistivity_modifier=lambda t, rho, eta, state: eta)
    np.testing.assert_allclose(ident.advance(s0, 0.1, nsteps=8).psi, plain.psi, rtol=0, atol=1e-12 * np.ptp(plain.psi))

    # a Jardin-type flattening of eta inside rho < 0.3 must change the core current and leave Ip fixed
    def flatten(t, rho, eta, state):
        out = eta.copy()
        out[rho < 0.3] = np.interp(0.3, rho, eta)
        return out

    mod = _solver(resistivity_modifier=flatten)
    m = mod.advance(s0, 0.1, nsteps=8)
    pm, pp = mod.profiles(m), sol.profiles(plain)
    assert abs(pm["ip"] / pp["ip"] - 1) < 1e-13
    assert np.max(np.abs(pm["q"][:5] - pp["q"][:5])) > 1e-4


def test_moving_boundary_term_frozen_flux():
    """Nearly ideal plasma (sigma huge), Phi_b decreasing: psi is frozen to the Phi surfaces,
    psi(rho, t) = psi0(rho sqrt(Phi_b(t)/Phi_b(0))) (equation 7 of the module)."""
    g0 = FluxSurfaceGeometry.cylinder(0.5, 3.0, 2.0, n=401)
    g1 = FluxSurfaceGeometry.cylinder(0.5, 3.0, 1.6, n=401)   # B0 ramp-down: Phi_b falls 20 %
    sched = GeometrySchedule([0.0, 1.0], [g0, g1])
    x = np.linspace(0, 1, 2001)
    psi0 = lambda r: 0.3 * r ** 2 - 0.1 * r ** 4  # noqa: E731
    errs = []
    for n in (50, 100, 200):
        rf = CurrentState.uniform_faces(n)
        st = CurrentState.from_psi(x, psi0(x), rf)
        sol = CurrentDiffusion(sched, 1.0e16, VloopBoundary(0.0))
        out = sol.advance(st, 1.0, nsteps=400)
        exact = psi0(out.rho_cells * np.sqrt(1.6 / 2.0))
        errs.append(np.max(np.abs(out.psi - exact)) / np.max(np.abs(exact)))
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    # central differences in space: second order (theory); bound fixed before running
    assert 1.8 <= p[-1] <= 2.2, (errs, p)
