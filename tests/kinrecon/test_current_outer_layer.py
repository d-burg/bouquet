"""Lumped outer layer and domain edge of ``bouquet.kinrecon.current``.

Synthetic, separatrix-like geometry tabulated on psi_N (what an equilibrium
code returns), with the logarithmic X-point singularity of a diverted
boundary built in:

    q(x)        = 1.1 + 2.2 x^2 + 0.5 ln(1/(1 - x))       (b = 0.5: inside the measured LSN range 0.38-0.77)
    dV/dpsi     = 4 pi^2 q / (F <1/R^2>)                  (geometric q identity; diverges like q)
    <B_p^2>     = 2 pi mu0 I_ref / (dV/dpsi)              (Ampere with a regular reference current I_ref)
    F, <1/R^2>  regular;  sigma, <J_NI.B> regular, with an edge-pedestal bootstrap peak at x = 0.975

The exact fully relaxed state is a volume quadrature on these surfaces
(no metric): ``I/F = (V/2pi) A(x) + B(x)``, ``A = int sigma <1/R^2>/(2 pi F) dV``,
``B = int <J_NI.B>/(2 pi F^2) dV``, ``V`` fixed by ``I(1) = Ip``; on the fixed
surfaces ``q = q_ref I_ref / I`` and ``dpsi/dx = dpsi_ref I / I_ref`` (as in the
domain-edge experiment).

Bounds are taken from the domain-edge experiment report (section numbers in
each assertion; never adjusted to the measured values here).  All numbers are
synthetic.
"""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon.current import (MU0, CurrentDiffusion, CurrentState, FluxSurfaceGeometry,  # noqa: E402
                                      IpBoundary, OuterLayer, OuterLayerGeometry, VloopBoundary)

R0, B0, F0, DPSI, IP = 1.7, 2.0, 3.4, 0.18, 1.2e6
CUTOFFS = (0.99, 0.995, 0.999)
NCELLS, NRHO = 400, 401        # as in the experiment (its section 1.1)


def F(x):
    return F0 * (1.0 - 0.02 * x)


def g1(x):
    return (1.0 + 0.08 * x) / R0 ** 2


def q_ref(x):
    # 1 - x floored at 1e-300 only so that a quadrature node rounding to x = 1 stays finite
    return 1.1 + 2.2 * x ** 2 - 0.5 * np.log(np.maximum(1.0 - x, 1e-300))


def dvdpsi(x):
    return 4.0 * np.pi ** 2 * q_ref(x) / (F(x) * g1(x))


def i_ref(x):
    return IP * (1.6 * x - 0.6 * x ** 2)


def bp2(x):
    return 2.0 * np.pi * MU0 * i_ref(x) / dvdpsi(x)


def sigma(x):
    return 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - x ** 2, 0.0, None) ** 1.5)


def jni(x):
    return B0 * (1.5e5 * np.exp(-(x / 0.35) ** 2) + 4.0e5 * np.exp(-((x - 0.975) / 0.02) ** 2))


# ----------------------------------------------------------------------------- exact relaxed state
def _exact():
    xs = np.unique(np.concatenate((np.linspace(0.0, np.sqrt(0.99), 1500) ** 2, 1.0 - np.geomspace(1e-2, 1e-10, 500),
                                   [1.0])))

    def q_(f, a, b):
        return quad(f, a, b, epsabs=0.0, epsrel=1e-12, limit=200)[0]

    fa = lambda x: sigma(x) * g1(x) / (2 * np.pi * F(x)) * DPSI * dvdpsi(x)  # noqa: E731
    fb = lambda x: jni(x) / (2 * np.pi * F(x) ** 2) * DPSI * dvdpsi(x)  # noqa: E731
    A = np.concatenate(([0.0], np.cumsum([q_(fa, a, b) for a, b in zip(xs[:-1], xs[1:])])))
    B = np.concatenate(([0.0], np.cumsum([q_(fb, a, b) for a, b in zip(xs[:-1], xs[1:])])))
    V = 2 * np.pi * (IP / F(1.0) - B[-1]) / A[-1]
    inew = F(xs) * (V / (2 * np.pi) * A + B)
    ratio = np.empty_like(xs)
    ratio[1:] = inew[1:] / i_ref(xs[1:])
    ratio[0] = ratio[1] + (ratio[1] - ratio[2]) * xs[1] / (xs[2] - xs[1])     # regular on axis
    anti = PchipInterpolator(xs, ratio).antiderivative()
    psin_new = (anti(xs) - anti(0.0)) / (anti(1.0) - anti(0.0))
    x95 = brentq(lambda x: float(np.interp(x, xs, psin_new)) - 0.95, 0.5, 0.999)
    x95 = brentq(lambda x: float((anti(x) - anti(0.0)) / (anti(1.0) - anti(0.0))) - 0.95, x95 - 0.01, x95 + 0.01)
    with np.errstate(divide="ignore"):
        qn = q_ref(xs) / ratio
    q_new = PchipInterpolator(xs[1:-1], qn[1:-1])
    e2 = np.empty_like(xs)
    e2[1:] = inew[1:] ** 2 / i_ref(xs[1:])
    e2[0] = 0.0
    bp2_int = 2 * np.pi * MU0 * DPSI * float(PchipInterpolator(xs, e2).antiderivative()(1.0))
    jb_ohm = PchipInterpolator(xs, sigma(xs) * F(xs) * g1(xs) * V / (2 * np.pi))
    return dict(V=V, q=q_new, q95=float(q_new(x95)), li3=2 * bp2_int / (MU0 ** 2 * IP ** 2 * R0), jb_ohm=jb_ohm,
                I=PchipInterpolator(xs, inew))


EXACT = _exact()


# ----------------------------------------------------------------------------- truncated problem
def _domain_samples(c, n_in=300, n_edge=100):
    """sqrt-uniform interior + geometric clustering toward the cutoff (the experiment's sample grid)."""
    a = np.linspace(np.sqrt(1e-3), np.sqrt(0.97 * c), n_in) ** 2
    d = np.geomspace(0.03 * c, 1e-3 * (1 - c), n_edge)
    return np.concatenate((a[:-1], c - d, [c]))


def _layer_samples(c, n=40, pad=1e-4):
    xl = 1.0 - (1.0 - c) * np.geomspace(1.0, pad, n)
    xl[0] = c
    return xl


def build(c, boundary=None, outer="lumped"):
    x = _domain_samples(c)
    geo = FluxSurfaceGeometry.from_psi_coordinates(x, 0.0, DPSI, dvdpsi(x), bp2(x), g1(x), F(x), B0=B0, R0=R0,
                                                   nrho=NRHO, psi_n_edge=c)
    xl = _layer_samples(c)
    lg = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, DPSI, dvdpsi(xl), g1(xl), F(xl), avg_grad_psi2_r2=bp2(xl),
                                                 fpol_lcfs=F(1.0))
    rs = np.concatenate(([0.0], geo.meta["rho_samples"]))
    xs = np.concatenate(([0.0], x))
    layer = OuterLayer.from_profiles(lg, (xl, sigma(xl)), (xl, jni(xl))) if outer == "lumped" else outer
    sol = CurrentDiffusion(geo, (rs, sigma(xs)), boundary or IpBoundary(IP), j_ni=(rs, jni(xs)), outer_layer=layer)
    return sol, PchipInterpolator(rs, xs)


def _interior(p, r2x, xmax=0.98):
    xf = r2x(p["rho_faces"])
    sel = (xf >= 0.02) & (xf <= xmax)
    xc = r2x(p["rho_cells"])
    selc = (xc >= 0.02) & (xc <= xmax)
    return xf[sel], p["q"][sel], xc[selc], p["jdotb_ohmic"][selc]


@pytest.fixture(scope="module")
def steady():
    out = {}
    for c in CUTOFFS:
        sol, r2x = build(c)
        ss = sol.steady_state(CurrentState.uniform_faces(NCELLS))
        out[c] = (sol, r2x, ss, sol.profiles(ss))
    return out


# ----------------------------------------------------------------------------- steady state
@pytest.mark.parametrize("c", CUTOFFS)
def test_lumped_layer_steady_state_matches_exact_relaxed_state(steady, c):
    sol, r2x, ss, p = steady[c]
    xf, q, xc, jb = _interior(p, r2x)
    # interior q: <= 1e-3, the experiment's bound "for every cutoff from 0.95 to 0.99999 in all 10 equilibria"
    # (section 0, item 1; code vs exact quadrature 2e-5 to 6e-4, section 2.4)
    assert np.max(np.abs(q / EXACT["q"](xf) - 1.0)) <= 1e-3
    # <J.B>_ohm relative to its maximum, V_loop: <= 9.3e-4 (section 2.4 table, max over 10 cases)
    je = EXACT["jb_ohm"](xc)
    assert np.max(np.abs(jb - je)) / np.max(np.abs(je)) <= 9.3e-4
    assert abs(ss.meta["v_loop"] / EXACT["V"] - 1.0) <= 9.3e-4
    # full-plasma q95 <= 6.8e-4 and l_i(3) <= 1.4e-3 (section 2.4 table)
    assert abs(p["q95"] / EXACT["q95"] - 1.0) <= 6.8e-4
    assert abs(p["li3"] / EXACT["li3"] - 1.0) <= 1.4e-3
    # the edge quantities say which surface they refer to
    assert p["ip"] == IP and p["psi_n_edge"] == c and p["edge_treatment"] == "lumped"
    assert "last closed surface" in p["surfaces"]["ip"] and str(c) in p["surfaces"]["ip_domain_edge"]
    assert 0.0 < p["outer_layer_current"] < 0.01 * IP


@pytest.mark.parametrize("c", CUTOFFS)
def test_lumped_layer_error_is_below_half_the_effect_of_a_halved_layer(steady, c):
    """Discrimination bounds derived on this fixture (added after an independent review showed that dropping
    the layer entirely at c = 0.999 passed every bound above).  For each quantity the lumped layer's error
    against the exact relaxed state must be at most HALF the error produced by a layer whose current is halved
    (conductance and source x 0.5): a 25 % error in the layer current would then be detected.  The layer
    current itself is compared with the exact quadrature (IP - I_exact(c)), not with the code's own formula,
    under the same rule.  Measured margins: factor 2-60 (q at c = 0.999 is the tightest, 5.5e-5 vs 1.1e-4)."""
    sol, r2x, ss, p = steady[c]
    xl = _layer_samples(c)
    lg = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, DPSI, dvdpsi(xl), g1(xl), F(xl), avg_grad_psi2_r2=bp2(xl),
                                                 fpol_lcfs=F(1.0))
    half = OuterLayer.from_profiles(lg, (xl, 0.5 * sigma(xl)), (xl, 0.5 * jni(xl)))
    sol_h, r2x_h = build(c, outer=half)
    ss_h = sol_h.steady_state(CurrentState.uniform_faces(NCELLS))
    p_h = sol_h.profiles(ss_h)

    def errors(p_, ss_, r2x_):
        xf, q, xc, jb = _interior(p_, r2x_)
        je = EXACT["jb_ohm"](xc)
        lay_ex = IP - float(EXACT["I"](c))
        return dict(q=np.max(np.abs(q / EXACT["q"](xf) - 1.0)), jb=np.max(np.abs(jb - je)) / np.max(np.abs(je)),
                    V=abs(ss_.meta["v_loop"] / EXACT["V"] - 1.0), q95=abs(p_["q95"] / EXACT["q95"] - 1.0),
                    li3=abs(p_["li3"] / EXACT["li3"] - 1.0), layer=abs(p_["outer_layer_current"] / lay_ex - 1.0))

    good, bad = errors(p, ss, r2x), errors(p_h, ss_h, r2x_h)
    for k in good:
        assert good[k] <= 0.5 * bad[k], (k, good[k], bad[k])


def test_interior_independent_of_cutoff(steady):
    """Cutoff-to-cutoff interior q differences <= 1e-3 (experiment section 0 item 1, same bound)."""
    xs = np.linspace(0.02, 0.98, 400)
    qs = {}
    for c in CUTOFFS:
        sol, r2x, ss, p = steady[c]
        qs[c] = PchipInterpolator(r2x(p["rho_faces"]), p["q"])(xs)
    for c in CUTOFFS[:-1]:
        assert np.max(np.abs(qs[c] / qs[CUTOFFS[-1]] - 1.0)) <= 1e-3


def test_dropping_the_layer_is_detected():
    """Discriminating power: treatment (iii) (Ip applied at the cutoff, layer dropped; outer_layer='none')
    at c = 0.99 biases V and interior q by more than the bounds above (experiment section 2.4: 0.2-1.1 %)."""
    sol, r2x = build(0.99, outer="none")
    ss = sol.steady_state(CurrentState.uniform_faces(NCELLS))
    p = sol.profiles(ss)
    xf, q, _, _ = _interior(p, r2x)
    assert abs(ss.meta["v_loop"] / EXACT["V"] - 1.0) > 9.3e-4
    assert np.max(np.abs(q / EXACT["q"](xf) - 1.0)) > 1e-3
    assert p["edge_treatment"] == "none" and "no outer layer" in p["surfaces"]["ip"]


def test_truncated_geometry_requires_an_explicit_edge_treatment():
    x = _domain_samples(0.995)
    geo = FluxSurfaceGeometry.from_psi_coordinates(x, 0.0, DPSI, dvdpsi(x), bp2(x), g1(x), F(x), B0=B0, R0=R0,
                                                   psi_n_edge=0.995)
    with pytest.raises(ValueError, match="outer_layer"):
        CurrentDiffusion(geo, 1e7, IpBoundary(IP))
    CurrentDiffusion(geo, 1e7, IpBoundary(IP), outer_layer="none")        # previous behavior, explicit
    wrong = OuterLayer(0.99, F(1.0), 1.0)
    with pytest.raises(ValueError, match="starts at"):
        CurrentDiffusion(geo, 1e7, IpBoundary(IP), outer_layer=wrong)


# ----------------------------------------------------------------------------- conservation and time dependence
def test_total_current_conserved_with_layer_ip_ramp():
    """Ip(t) applies at the last closed surface: the domain current (volume integral of <J_phi/R>) plus the
    layer current from the layer's own Ohm's law, F_lcfs (I(c)/F(c) + A V/2pi + B) - I(c), equals Ip(t) to
    round-off (bounds 1e-12 / 1e-13 as the existing conservation test)."""
    ip_t = lambda t: IP * (1.0 + 0.05 * min(t / 0.05, 1.0))  # noqa: E731
    sol, r2x = build(0.995, boundary=IpBoundary(ip_t))
    s = sol.steady_state(CurrentState.uniform_faces(NCELLS))
    lay_a, lay_b = sol.outer_layer.integrals(0.0)
    for k in range(1, 9):
        s = sol.advance(s, 0.01 * k, dt=5e-4)
        p = sol.profiles(s)
        # (p["ip"] is the boundary value itself when a layer is attached -- not checked, that would be a
        # tautology; the independent check is the volume integral plus the layer's Ohm's law below)
        vp = sol.geometry_at(s.time).evaluate(p["rho_cells"], ["dvol_drho"])["dvol_drho"]
        i_c = np.sum(p["jt_over_r"] * vp * np.diff(p["rho_faces"])) / (2 * np.pi)
        assert abs(i_c / p["ip_domain_edge"] - 1.0) < 1e-12
        f_c = sol.geometry_at(s.time).fpol[-1]
        layer = F(1.0) * (i_c / f_c + lay_a * p["v_loop_boundary"] / (2 * np.pi) + lay_b) - i_c
        assert abs((i_c + layer) / ip_t(s.time) - 1.0) < 1e-12
        assert abs(layer / p["outer_layer_current"] - 1.0) < 1e-9


def test_time_dependent_interior_independent_of_cutoff():
    """5 % Ip ramp in 50 ms (the experiment's scenario b), TR-BDF2, dt = 0.5 ms, 400 cells.  The experiment
    measured the implicit lumped layer within 2.6e-4 of the finest cutoff in interior q and <J.B> for every
    c >= 0.98 (section 2.5); two cutoffs therefore agree to 2 x 2.6e-4 = 5.2e-4 (triangle inequality)."""
    ip_t = lambda t: IP * (1.0 + 0.05 * min(t / 0.05, 1.0))  # noqa: E731
    xs = np.linspace(0.02, 0.9, 300)
    res = {}
    for c in (0.99, 0.999):
        sol, r2x = build(c, boundary=IpBoundary(ip_t))
        s = sol.steady_state(CurrentState.uniform_faces(NCELLS))
        snaps = []
        for t in (0.005, 0.02, 0.05, 0.1, 0.2):
            s = sol.advance(s, t, dt=5e-4)
            p = sol.profiles(s)
            snaps.append((PchipInterpolator(r2x(p["rho_faces"]), p["q"])(xs),
                          PchipInterpolator(r2x(p["rho_cells"]), p["jdotb"], extrapolate=True)(xs)))
        res[c] = snaps
    for (qa, ja), (qb, jb) in zip(res[0.99], res[0.999]):
        assert np.max(np.abs(qa / qb - 1.0)) <= 5.2e-4
        assert np.max(np.abs(ja - jb)) / np.max(np.abs(jb)) <= 5.2e-4


@pytest.mark.parametrize("scheme,theta", [("trbdf2", 0.5), ("theta", 1.0)])
def test_steady_state_is_long_time_limit_with_layer(scheme, theta):
    """Same criterion as the untruncated test: the state (psi - psi_b), q and the layer current agree to 1e-9
    after 60 core resistive times.

    V_loop(rho) is a second difference of psi divided by sigma: it is checked against the error that the same
    1e-9 state criterion implies through the discrete formula, |dV_i| <= 2 pi / (sigma F <1/R^2>)_i *
    2 pi F_i^2 / (V'_i h) * (k_{i-1/2}/d_{i-1/2} + k_{i+1/2}/d_{i+1/2}) * 2e-9 max|psi - psi_b| (propagation of
    the 1e-9 bound, not a separate tolerance).  [A first run asserted V_loop to 1e-9 directly and measured
    4-7e-9 while psi - psi_b agreed to 2e-11: amplification of the state error by the second difference over
    the steep edge conductivity, after a 2700 s run whose flux offset grows to ~90 Wb/rad.]"""
    sol, r2x = build(0.995)
    rf = CurrentState.uniform_faces(80)
    ss = sol.steady_state(rf)
    st = CurrentState.from_q(sol.geometry_at(0.0), np.linspace(0, 1, 30), np.full(30, 3.0), rf)
    tau = MU0 * 1.0e8 * 0.6 ** 2
    out = sol.advance(st, 60 * tau, dt=0.1 * tau, scheme=scheme, theta=theta)
    a, b = sol.profiles(out), sol.profiles(ss)
    np.testing.assert_allclose(a["q"], b["q"], rtol=1e-9)
    np.testing.assert_allclose(a["outer_layer_current"], ss.meta["outer_layer_current"], rtol=1e-9)
    np.testing.assert_allclose(out.psi - out.psi_boundary, ss.psi - ss.psi_boundary, rtol=1e-9, atol=0)
    c = sol.coefficients(0.0, rf)
    rc = 0.5 * (rf[1:] + rf[:-1])
    kd = np.zeros(rf.size)
    kd[1:-1] = c.k_f[1:-1] / np.diff(rc)
    # cutoff face (Ip condition with the layer, eq. 8): Gamma_c = Gamma_b - A rate - B with
    # rate = (Gamma_b - Gamma_{n-1/2} - s - B)/(m + A), so |dGamma_c| = A/(m + A) |dGamma_{n-1/2}| <= |dGamma_{n-1/2}|
    kd[-1] = kd[-2]
    dpsi_abs = 2e-9 * np.max(np.abs(ss.psi - ss.psi_boundary))
    dv = (2 * np.pi / (c.sigma_c * c.geo_c["fpol"] * c.geo_c["avg_inv_r2"]) * 2 * np.pi * c.geo_c["fpol"] ** 2
          / (c.geo_c["dvol_drho"] * np.diff(rf)) * (kd[:-1] + kd[1:]) * dpsi_abs)
    assert np.all(np.abs(a["v_loop"] - ss.meta["v_loop"]) <= dv)



def test_vloop_boundary_with_layer():
    """V(t) applies at the last closed surface.  Steady state: the Ip-mode loop voltage returns Ip (exact
    algebra, 1e-12); in time the total current holds at the steady value (1e-10, as the untruncated test)."""
    sol, r2x = build(0.995)
    rf = CurrentState.uniform_faces(NCELLS)
    ss = sol.steady_state(rf)
    v = ss.meta["v_loop"]
    sol_v, _ = build(0.995, boundary=VloopBoundary(v))
    ss_v = sol_v.steady_state(rf)
    assert abs(ss_v.meta["ip"] / IP - 1.0) < 1e-12
    assert abs(ss_v.meta["outer_layer_current"] / ss.meta["outer_layer_current"] - 1.0) < 1e-9
    out = sol_v.advance(ss, 2.0, nsteps=40)
    p = sol_v.profiles(out)
    assert abs(p["ip"] / IP - 1.0) < 1e-10
    assert p["v_loop_boundary"] == v and "last closed surface" in p["surfaces"]["v_loop_boundary"]


# ----------------------------------------------------------------------------- convergence orders with a layer
GEO = FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=301, fpol=lambda x: 3.4 * (1.0 - 0.02 * x ** 2))
T_END = 0.4


def _sigma_t(t, rho):
    return (4.0e6 * (1.0 - 0.9 * rho ** 2) ** 1.5 + 1.0e5) * (1.0 + 0.3 * np.sin(2 * np.pi * t / T_END))


def _layer_t():
    # a layer with time-varying conductance and source, as the kinetic profiles would make it
    return OuterLayer(0.995, 3.4 * 0.98, lambda t: 40.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / T_END)),
                      lambda t: 2.0e3 * (1.0 + t / T_END))


@pytest.mark.parametrize("scheme,theta,order", [("trbdf2", 0.5, 2.0), ("theta", 0.5, 2.0), ("theta", 1.0, 1.0)])
@pytest.mark.parametrize("bc", ["ip", "vloop"])
def test_time_convergence_order_with_layer(scheme, theta, order, bc):
    """Same bounds as the untruncated test (2 +- 0.1, 1 +- 0.1; theory)."""
    boundary = (IpBoundary(lambda t: 1.0e6 * (1.0 + 0.5 * t / T_END + 0.1 * np.sin(5.0 * t / T_END))) if bc == "ip"
                else VloopBoundary(lambda t: 1.5 + 0.5 * np.sin(4 * t / T_END)))
    sol = CurrentDiffusion(GEO, _sigma_t, boundary, j_ni=lambda t, r: 2.0e5 * np.exp(-((r - 0.6) / 0.15) ** 2),
                           outer_layer=_layer_t())
    st0 = sol.steady_state(CurrentState.uniform_faces(40))
    ref = sol.advance(st0, T_END, nsteps=5120, scheme="trbdf2")
    errs = []
    for ns in (40, 80, 160, 320):
        s = sol.advance(st0, T_END, nsteps=ns, scheme=scheme, theta=theta)
        errs.append(max(np.max(np.abs(s.psi - ref.psi)), abs(s.psi_boundary - ref.psi_boundary)))
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert order - 0.1 <= p[-1] <= order + 0.1, (errs, p)


def test_spatial_order_truncated_cylinder_with_layer():
    """Cylinder of radius a truncated at r_c = 0.95 a: the truncated domain (itself a cylinder of radius r_c)
    plus the analytic layer must give the full-domain steady loop voltage, Ip = (V/(2 pi R0)) int_0^a sigma
    2 pi r dr, with the second-order error of the domain's midpoint quadrature (theory: 2; bound as the
    untruncated test, 1.9-2.1)."""
    a, R0c, B0c, ip = 0.5, 1.0e4, 2.0, 1.0e6
    rc_ = 0.95 * a

    def sig_r(r):
        return 1.0e7 * (1.0 - 0.95 * (r / a) ** 2) ** 1.5 + 2.0e4

    cyl = FluxSurfaceGeometry.cylinder(rc_, R0c, B0c, n=401)
    cyl.meta["psi_n_edge"] = 0.95       # labels the truncation (any value < 1 works here)
    # layer: A = int sigma <1/R^2>/(2 pi F) dV = (2 pi / (B0 R0^2)) int sigma r dr; no J_NI
    A = 2 * np.pi / (B0c * R0c ** 2) * quad(lambda r: sig_r(r) * r, rc_, a, epsabs=0, epsrel=1e-13)[0]
    layer = OuterLayer(0.95, B0c * R0c, A, 0.0)
    sol = CurrentDiffusion(cyl, lambda t, rho: sig_r(rho * rc_), IpBoundary(ip), outer_layer=layer)
    v_exact = ip * 2 * np.pi * R0c / quad(lambda r: sig_r(r) * 2 * np.pi * r, 0, a, epsabs=0, epsrel=1e-13)[0]
    errs = []
    for n in (25, 50, 100, 200):
        ss = sol.steady_state(CurrentState.uniform_faces(n))
        errs.append(abs(ss.meta["v_loop"] / v_exact - 1.0))
        assert abs(sol.profiles(ss)["ip"] / ip - 1.0) < 1e-14
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert 1.9 <= orders[-1] <= 2.1, (errs, orders)
