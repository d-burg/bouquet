"""Analytic checks of ``bouquet.kinrecon.current.diffusion``.

Cylindrical limit (``FluxSurfaceGeometry.cylinder``): equation (6) of the
module reduces to ``dpsi/dt = (eta/mu0) (1/r) d/dr (r dpsi/dr)``, whose
eigenmodes are ``J0(j r/a) exp(-eta j^2 t/(mu0 a^2))`` with ``J1(j) = 0`` under
the Ip condition (zero flux ``dpsi/dr(a) = 0`` for Ip = 0) and ``J0(j) = 0``
under the loop-voltage condition with ``V = 0``.  Steady state: ``J_phi =
sigma E`` with uniform ``E``, i.e. ``<J.B> / (sigma F <1/R^2>)`` uniform in any
geometry.  All numbers synthetic.
"""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, jn_zeros

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon.current import (MU0, CurrentDiffusion, CurrentState, FluxSurfaceGeometry,  # noqa: E402
                                      IpBoundary, VloopBoundary)

A, R0, B0 = 0.5, 1.0e4, 2.0           # R0/a = 2e4: toroidal corrections ~ (a/R0)^2 = 2.5e-9
SIGMA = 1.0e6
CYL = FluxSurfaceGeometry.cylinder(A, R0, B0, n=401)


def _mode_run(bc_kind, n_cells, nsteps=2000):
    zero = jn_zeros(1, 1)[0] if bc_kind == "ip" else jn_zeros(0, 1)[0]
    lam = zero ** 2 / (MU0 * SIGMA * A ** 2)
    rf = CurrentState.uniform_faces(n_cells)
    x = np.linspace(0.0, 1.0, 4001)
    st = CurrentState.from_psi(x, j0(zero * x), rf)
    bc = IpBoundary(0.0) if bc_kind == "ip" else VloopBoundary(0.0)
    sol = CurrentDiffusion(CYL, SIGMA, bc)
    t_end = 0.5 / lam
    out = sol.advance(st, t_end, nsteps=nsteps)
    exact = j0(zero * st.rho_cells) * np.exp(-lam * t_end)
    num = out.psi
    if bc_kind == "ip":     # the Neumann problem fixes psi only up to a constant: compare mean-free parts
        num, exact = num - num.mean(), exact - exact.mean()
    return np.max(np.abs(num - exact)) / np.max(np.abs(exact)), out, sol, lam, zero


@pytest.mark.parametrize("bc_kind", ["ip", "vloop"])
def test_decay_eigenmode_and_spatial_order(bc_kind):
    errs = [_mode_run(bc_kind, n)[0] for n in (16, 32, 64, 128)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    # cell-centered FV + midpoint quadrature: second order (theory); bound fixed before running
    assert 1.9 <= orders[-1] <= 2.1, orders
    assert errs[-1] < 1e-4


def test_vloop_boundary_current_second_order():
    """Under the loop-voltage condition Ip is an output, I(1) = V' g2 psi'(1)/(2 pi mu0); the
    one-sided boundary gradient must keep it second order (theory: 2; bound fixed before running)."""
    from scipy.special import j1
    errs = []
    for n in (16, 32, 64, 128):
        _, out, sol, lam, zero = _mode_run("vloop", n)
        ip = sol.profiles(out)["ip"]
        t_end = out.time
        exact = 2 * np.pi / (MU0 * R0) * (-zero * j1(zero)) * np.exp(-lam * t_end)
        errs.append(abs(ip / exact - 1))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert 1.9 <= orders[-1] <= 2.1, (errs, orders)


@pytest.mark.parametrize("bc_kind", ["ip", "vloop"])
def test_decay_rate(bc_kind):
    """Measured exponential decay rate of the eigenmode equals eta j^2/(mu0 a^2)."""
    zero = jn_zeros(1, 1)[0] if bc_kind == "ip" else jn_zeros(0, 1)[0]
    lam = zero ** 2 / (MU0 * SIGMA * A ** 2)
    rf = CurrentState.uniform_faces(128)
    x = np.linspace(0.0, 1.0, 4001)
    st = CurrentState.from_psi(x, j0(zero * x), rf)
    sol = CurrentDiffusion(CYL, SIGMA, IpBoundary(0.0) if bc_kind == "ip" else VloopBoundary(0.0))
    t1, t2 = 0.25 / lam, 1.25 / lam
    s1 = sol.advance(st, t1, nsteps=1000)
    s2 = sol.advance(s1, t2, nsteps=4000)
    # amplitude = projection on the mode (weight r dr) of the mean-free profile
    w = st.rho_cells * np.diff(rf)
    mode = j0(zero * st.rho_cells)
    amp = [np.sum(w * (s.psi - (np.sum(w * s.psi) / w.sum() if bc_kind == "ip" else 0.0)) * mode) for s in (s1, s2)]
    lam_num = np.log(amp[0] / amp[1]) / (t2 - t1)
    # spatial error at 128 cells is ~1e-5 (previous test); the rate error is of the same order
    assert abs(lam_num / lam - 1.0) < 1e-4


def test_steady_state_current_proportional_to_conductivity_cylinder():
    def sigma(t, rho):
        return 1.0e7 * (1.0 - 0.95 * rho ** 2) ** 1.5 + 2.0e4

    ip = 1.0e6
    sol = CurrentDiffusion(CYL, sigma, IpBoundary(ip))
    vs = []
    for n in (25, 50, 100, 200):
        ss = sol.steady_state(CurrentState.uniform_faces(n))
        p = sol.profiles(ss)
        # discrete identity: <J.B>/(sigma F <1/R^2>) uniform, j_phi = sigma E (round-off)
        e_par = p["jdotb"] / (p["conductivity"] * B0 * R0 / R0 ** 2)
        # round-off bound: <J.B> is rebuilt from differences of psi, so the attainable relative
        # accuracy is ~ eps * max|psi - psi_b| / min|delta psi| (cancellation), times a safety factor 64
        dpsi = np.abs(np.diff(np.concatenate((ss.psi, [ss.psi_boundary]))))
        roundoff = 64 * np.finfo(float).eps * np.max(np.abs(ss.psi - ss.psi_boundary)) / dpsi.min()
        assert np.ptp(e_par) / np.mean(e_par) < roundoff
        assert abs(p["ip"] / ip - 1.0) < 1e-14
        vs.append(ss.meta["v_loop"])
    # analytic: Ip = (V / (2 pi R0)) int sigma 2 pi r dr
    integral = quad(lambda r: sigma(0, r / A) * 2 * np.pi * r, 0, A, epsabs=0, epsrel=1e-13)[0]
    v_exact = ip * 2 * np.pi * R0 / integral
    errs = np.abs(np.array(vs) / v_exact - 1.0)
    orders = np.log2(errs[:-1] / errs[1:])
    assert 1.9 <= orders[-1] <= 2.1, (errs, orders)


def test_uniform_current_cylinder_q_and_li():
    """Uniform sigma, no J_NI: flat j_phi, so q = 2 pi a^2 B0 / (mu0 Ip R0) at every radius and
    l_i(3) = 2 int B_p^2 dV / (mu0^2 Ip^2 R0) = 1/2 exactly (textbook cylinder values)."""
    ip = 5.0e5
    sol = CurrentDiffusion(CYL, SIGMA, IpBoundary(ip))
    q_exact = 2 * np.pi * A ** 2 * B0 / (MU0 * ip * R0)
    errs = []
    for n in (25, 50, 100, 200):
        p = sol.profiles(sol.steady_state(CurrentState.uniform_faces(n)))
        np.testing.assert_allclose(p["q"][1:], q_exact, rtol=1e-9)      # exact for linear psi' (round-off)
        errs.append(abs(p["li3"] / 0.5 - 1))
    # l_i from the trapezoid rule over faces: second order (theory); bound fixed before running
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert errs[-1] < 1e-4 and (errs[-1] < 1e-12 or 1.9 <= orders[-1] <= 2.1), (errs, orders)


def test_steady_state_ohmic_shape_toroidal():
    """Circular torus with a non-inductive current: <J_ohm.B> = sigma F <1/R^2> V / 2pi exactly."""
    g = FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=401, fpol=lambda x: 3.4 * (1 - 0.02 * x ** 2))
    sol = CurrentDiffusion(g, lambda t, r: 5e6 * (1 - 0.9 * r ** 2) + 1e5, IpBoundary(8e5),
                           j_ni=lambda t, r: 3e5 * np.exp(-((r - 0.75) / 0.08) ** 2))
    ss = sol.steady_state(CurrentState.uniform_faces(120))
    p = sol.profiles(ss)
    gc = g.evaluate(p["rho_cells"])
    shape = p["conductivity"] * gc["fpol"] * gc["avg_inv_r2"] * ss.meta["v_loop"] / (2 * np.pi)
    np.testing.assert_allclose(p["jdotb_ohmic"], shape, rtol=1e-10)
    np.testing.assert_allclose(p["v_loop"], ss.meta["v_loop"], rtol=1e-10)


def _cylinder_equilibrium(n_geo=2001):
    """Periodic-cylinder force-balance equilibrium with a diamagnetic B_z(r) (exact, analytic).

    J_z = j0 (1 - r^2/a^2), B_theta = mu0 j0 (r/2 - r^3/(4 a^2)), B_z = B0 (1 - 0.05 r^2/a^2),
    J_theta = -B_z'/mu0, dp/dr = J_theta B_z - J_z B_theta (radial force balance).  rho = sqrt(Phi/Phi_b) with
    Phi(r) = 2 pi B0 (r^2/2 - 0.05 r^4/(4 a^2)); F = R0 B_z, <1/R^2> = 1/R0^2, <R> = R0, <1/R> = 1/R0,
    <B^2> = B_z^2 + B_theta^2, psi(r) = R0 int B_theta dr.  In this geometry the identity of
    ``profiles()['jphi_avg']`` reduces exactly to <J_phi> = J_z:
    (B_z/B^2)(J_z B_z + J_theta B_theta) - (p_r / (R0 B_theta)) R0 B_theta^2 / B^2 = J_z.
    """
    from scipy.optimize import brentq
    a, Rc, Bc, j0, bz = 0.5, 1.0e4, 2.0, 1.0e6, 0.05

    def B_z(r):
        return Bc * (1.0 - bz * r ** 2 / a ** 2)

    def B_t(r):
        return MU0 * j0 * (r / 2.0 - r ** 3 / (4.0 * a ** 2))

    def J_z(r):
        return j0 * (1.0 - r ** 2 / a ** 2)

    def dp_dr(r):
        return -B_z(r) * (-2.0 * Bc * bz * r / a ** 2) / MU0 - J_z(r) * B_t(r)

    def phi(r):
        return 2.0 * np.pi * Bc * (r ** 2 / 2.0 - bz * r ** 4 / (4.0 * a ** 2))

    phib = phi(a)
    rho = np.linspace(0.0, 1.0, n_geo)
    r = np.array([0.0] + [brentq(lambda x: phi(x) - phib * v ** 2, 0.0, a, xtol=1e-15, rtol=1e-15)
                          for v in rho[1:-1]] + [a])
    dr_drho = np.empty(n_geo)
    dr_drho[1:] = rho[1:] * phib / (np.pi * r[1:] * B_z(r[1:]))
    dr_drho[0] = np.sqrt(phib / (np.pi * Bc))
    one = np.ones(n_geo)
    geo = FluxSurfaceGeometry(rho=rho, dvol_drho=4 * np.pi ** 2 * Rc * r * dr_drho,
                              avg_grad_rho2_r2=1.0 / (dr_drho * Rc) ** 2, avg_inv_r2=one / Rc ** 2, fpol=Rc * B_z(r),
                              phi_b=phib, B0=Bc, R0=Rc, avg_inv_r=one / Rc, avg_r=Rc * one,
                              avg_b2=B_z(r) ** 2 + B_t(r) ** 2, meta=dict(r=r))
    psi = Rc * MU0 * j0 * (r ** 2 / 4.0 - r ** 4 / (16.0 * a ** 2))
    ip = 2.0 * np.pi * a * B_t(a) / MU0
    r_of_rho = lambda x: np.interp(x, rho, r)  # noqa: E731  (only for evaluating exact values at cells)
    return geo, rho, psi, ip, J_z, dp_dr, dr_drho, r_of_rho


def test_jphi_avg_cylinder_equilibrium():
    """<J_phi> output (the module's <J_phi> relation) against an exact force-balance equilibrium.

    Previously untested.  The state is the exact psi sampled onto the cells (not a discrete solution).
    Interior cells: second order (bound 1.9-2.1, and < 1e-3 of max J_z at 200 cells; fixed before running).
    Last cell: first order (bound 0.9-1.1).  [A first run asserted second order over all cells and failed
    at the last cell: there the exact Ip boundary flux meets the O(h^2) error of the central-difference
    face flux at rho_{n-1/2}, which no longer cancels across the cell, so <J.B> -- and with it <J_phi> --
    is O(h) in the last cell whenever the state is not a discrete solution of the scheme (a sampled
    equilibrium, or a state after a post_step crash).  Measured 1.1e-3, 6.1e-4, 3.1e-4 at 50/100/200 cells.]"""
    from scipy.interpolate import PchipInterpolator
    geo, rho, psi, ip, J_z, dp_dr, dr_drho, r_of_rho = _cylinder_equilibrium()
    drdrho = PchipInterpolator(rho, dr_drho)
    rr = PchipInterpolator(rho, geo.meta["r"])
    e_in, e_last = [], []
    for n in (25, 50, 100, 200):
        rf = CurrentState.uniform_faces(n)
        st = CurrentState.from_psi(rho, psi, rf)
        sol = CurrentDiffusion(geo, 1.0e7, IpBoundary(ip))
        rcells = st.rho_cells
        p = sol.profiles(st, dpressure_drho=lambda x: dp_dr(rr(x)) * drdrho(x))
        exact = J_z(rr(rcells))
        err = np.abs(p["jphi_avg"] - exact) / np.max(np.abs(exact))
        e_in.append(np.max(err[:-1]))
        e_last.append(err[-1])
    o_in = np.log2(np.array(e_in[:-1]) / np.array(e_in[1:]))
    o_last = np.log2(np.array(e_last[:-1]) / np.array(e_last[1:]))
    assert e_in[-1] < 1e-3, e_in
    assert 1.9 <= o_in[-1] <= 2.1, (e_in, o_in)
    assert 0.9 <= o_last[-1] <= 1.1, (e_last, o_last)
