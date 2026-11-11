"""Moving boundary with finite conductivity, and input guards, of ``bouquet.kinrecon.current.diffusion``.

Added after an independent review:

* the loop-voltage boundary row used a two-point psi'(1) in the moving-boundary term (7), which made the
  scheme first order in space whenever dPhi_b/dt != 0 and the conductivity is finite; the existing frozen-flux
  test (sigma = 1e16) could not see it.  Manufactured solution below: shaped metric, steep edge conductivity,
  Phi_b falling 30 % in 50 ms, time-varying everything; the source <J_NI.B> is computed from the exact solution.
  Bound: observed spatial order 2 +- 0.1 between the two finest grids (theory: 2; the same convention as the
  other order tests).  Before the fix the loop-voltage case gave 1.00.
* silent behaviors are refused: a static geometry carrying dPhi_b/dt, tables evaluated outside their range,
  Z_eff < 1, grids with fewer than 3 cells, negative Ip.
"""
import numpy as np
import pytest

from bouquet.kinrecon.current import (MU0, CurrentDiffusion, CurrentState, FluxSurfaceGeometry, IpBoundary,
                                      PositiveFrameError, VloopBoundary, as_profile_function, as_time_function)
from bouquet.kinrecon.current.conductivity import f33_effective_trapped_fraction

R0, T = 1.7, 0.05
V0 = 4 * np.pi ** 2 * R0 * 0.6 ** 2 * 2.0
G20 = 1.0 / (0.6 * R0) ** 2
PHIB0 = 1.2
RHO = np.linspace(0.0, 1.0, 4001)


def Vp(r):
    return V0 * r * (1 + 0.8 * r ** 2 + 2.0 * r ** 6)


def g2(r):
    return G20 * (1 + 3 * r ** 2 + 20.0 * r ** 8)


def g1(r):
    return (1 - 0.3 * r ** 2) / R0 ** 2


def F0t(t):
    return 3.4 * (1 - 0.3 * t / T)


def Ff(t, r):
    return F0t(t) * (1 - 0.05 * r ** 2)


def phib(t):
    return PHIB0 * F0t(t) / 3.4


def dphib(t):
    return PHIB0 * (-0.3 * 3.4 / T) / 3.4


def kk(t, r):
    return Vp(r) * g2(r) / (2 * np.pi * MU0 * Ff(t, r))


def dkk(t, r, h=1e-6):
    return (kk(t, r + h) - kk(t, r - h)) / (2 * h)


def sigma(t, r):
    r = np.asarray(r, dtype=float)
    return 2e7 * (1 - 0.6 * r ** 2) * (1e-2 + (1 - 1e-2) * 0.5 * (1 - np.tanh((r - 0.93) / 0.04)))


SC = 1.0


def amp(t):
    return SC * (1 + 0.4 * np.sin(3 * t / T))


def psi(t, r):
    return amp(t) * (0.1 * r ** 2 + 0.5 * r ** 4 - 0.35 * r ** 6)


def dpsi(t, r):
    return amp(t) * (0.2 * r + 2.0 * r ** 3 - 2.1 * r ** 5)


def d2psi(t, r):
    return amp(t) * (0.2 + 6.0 * r ** 2 - 10.5 * r ** 4)


def psidot_phi(t, r):
    psidot_rho = SC * (1.2 / T * np.cos(3 * t / T)) * (0.1 * r ** 2 + 0.5 * r ** 4 - 0.35 * r ** 6)
    return psidot_rho - r * dphib(t) / (2 * phib(t)) * dpsi(t, r)


def jni(t, r):
    r = np.asarray(r, dtype=float)
    a = Vp(r) * g1(r) * sigma(t, r) / (2 * np.pi * Ff(t, r))
    src = dkk(t, r) * dpsi(t, r) + kk(t, r) * d2psi(t, r) - a * psidot_phi(t, r)
    return src * 2 * np.pi * Ff(t, r) ** 2 / Vp(r)


def ip_edge(t):
    return Ff(t, 1.0) * kk(t, 1.0) * dpsi(t, 1.0)


SC = 1e6 / ip_edge(0.0)


def geo(t):
    return FluxSurfaceGeometry(rho=RHO, dvol_drho=Vp(RHO), avg_grad_rho2_r2=g2(RHO), avg_inv_r2=g1(RHO),
                               fpol=Ff(t, RHO), phi_b=phib(t), B0=2.0, R0=R0, dphib_dt=dphib(t))


def _error(n, mode, nsteps=800):
    rf = np.linspace(0, 1, n + 1)
    rc = 0.5 * (rf[1:] + rf[:-1])
    if mode == "ip":
        sol = CurrentDiffusion(geo, sigma, IpBoundary(ip_edge), j_ni=jni)
    else:
        sol = CurrentDiffusion(geo, sigma, VloopBoundary(lambda t: 2 * np.pi * psidot_phi(t, 1.0)), j_ni=jni)
    out = sol.advance(CurrentState(rf, psi(0, rc), psi(0, 1.0), 0.0), T, nsteps=nsteps)
    return np.max(np.abs(out.psi - psi(T, rc))) / np.ptp(psi(T, rc))


@pytest.mark.parametrize("mode", ["vloop", "ip"])
def test_moving_boundary_second_order_with_finite_conductivity(mode):
    errs = [_error(n, mode) for n in (50, 100, 200)]
    order = np.log2(errs[1] / errs[2])
    assert 1.9 <= order <= 2.1, (errs, order)


# ----------------------------------------------------------------------------- guards
def test_static_geometry_with_dphib_dt_is_refused():
    g = geo(0.0)
    with pytest.raises(ValueError, match="dphib_dt"):
        CurrentDiffusion(g, 1e7, IpBoundary(1e6))
    d = g.to_dict()
    d["dphib_dt"] = 0.0
    CurrentDiffusion(FluxSurfaceGeometry.from_dict(d), 1e7, IpBoundary(1e6))


def test_tables_are_not_extrapolated_silently():
    f = as_profile_function((np.linspace(0, 0.9, 10), np.linspace(2.0, 1.0, 10)), log=True)
    f(0.0, np.linspace(0, 0.9, 7))
    with pytest.raises(ValueError, match="outside the tabulated range"):
        f(0.0, np.array([0.5, 0.95]))
    g = as_profile_function((np.linspace(0, 0.9, 10), np.linspace(2.0, 1.0, 10)), log=True, extrapolate=True)
    assert np.isfinite(g(0.0, np.array([1.0])))[0]
    h = as_profile_function((np.array([0.0, 1.0]), np.linspace(0, 1, 5), np.ones((2, 5))))
    with pytest.raises(ValueError, match="time"):
        h(1.5, np.array([0.5]))
    tf = as_time_function(([0.0, 1.0], [1.0, 2.0]))
    assert tf(0.5) == 1.5
    with pytest.raises(ValueError, match="outside the tabulated range"):
        tf(-1.0)
    assert as_time_function(([0.0, 1.0], [1.0, 2.0]), extrapolate=True)(2.0) == 2.0
    # the solver refuses a conductivity table that stops short of the grid, unless asked to extrapolate
    cyl = FluxSurfaceGeometry.cylinder(0.6, 2.0, 2.0, n=101)
    rf = np.linspace(0, 1, 41)
    sol = CurrentDiffusion(cyl, (np.linspace(0, 0.9, 10), np.full(10, 1e7)), IpBoundary(1e6))
    with pytest.raises(ValueError, match="conductivity table"):
        sol.steady_state(rf)
    sol = CurrentDiffusion(cyl, (np.linspace(0, 0.9, 10), np.full(10, 1e7)), IpBoundary(1e6),
                           extrapolate_inputs=True)
    sol.steady_state(rf)
    assert sol.settings["extrapolate_inputs"] is True
    with pytest.raises(ValueError, match="outside the tabulated range"):
        IpBoundary(([0.0, 1.0], [1e6, 1.1e6]))(2.0)


def test_zeff_below_one_is_refused_not_clipped():
    with pytest.raises(ValueError, match="Z_eff"):
        f33_effective_trapped_fraction(0.5, 0.1, 0.8)
    f33_effective_trapped_fraction(0.5, 0.1, 1.0)


def test_grid_with_fewer_than_three_cells_is_refused():
    cyl = FluxSurfaceGeometry.cylinder(0.6, 2.0, 2.0, n=101)
    sol = CurrentDiffusion(cyl, 1e7, IpBoundary(1e6))
    for n in (1, 2):
        with pytest.raises(ValueError, match="at least 3 cells"):
            sol.steady_state(np.linspace(0, 1, n + 1))
    sol.steady_state(np.linspace(0, 1, 4))


def test_negative_ip_is_refused_and_zero_allowed():
    cyl = FluxSurfaceGeometry.cylinder(0.6, 2.0, 2.0, n=101)
    with pytest.raises(PositiveFrameError):
        CurrentDiffusion(cyl, 1e7, IpBoundary(-1e6)).steady_state(np.linspace(0, 1, 21))
    with pytest.raises(PositiveFrameError):
        IpBoundary(lambda t: float("nan"))(0.0)
    CurrentDiffusion(cyl, 1e7, IpBoundary(0.0)).steady_state(np.linspace(0, 1, 21))
