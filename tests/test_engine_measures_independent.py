"""The engine's measures checked against INDEPENDENT computations.

The toy stand-in measures with the package's own ``Ip_fsa_weights`` /
``li_value``, so the engine and the toy would agree on a wrong measure (the
review's surviving mutant U1: ``Ip_fsa_weights`` 1 % off passed every engine
test).  Here:

* ``Ip_fsa_weights`` (the jphi-linterp affine Ip measure) against a direct
  2-D area integral of the Grad-Shafranov toroidal current density
  ``j_phi = R p' + F F' / (mu0 R)`` over an analytic cross-section
  (concentric circular surfaces), with ``F F'`` from the jphi-linterp
  relation ``J = <R> p' + F F' <1/R> / mu0``;
* the engine's l_i row criterion (``EngineRows``) binds exactly at its
  tolerance: an error at the tolerance passes, 1.5x fails (mutants E5 /
  E5b, never binding on the toy, where another criterion always binds
  first).

Solver-free.
"""
import numpy as np
import pytest

MU0 = 4.0e-7 * np.pi


def _circular(n=401, R0=1.7, a=0.6, dpsi=0.3):
    """Flux-surface averages of concentric circles r = a sqrt(psi_N) about
    R0.  |grad psi| is constant on a surface, so the FSA weight
    dl / B_p = R dl / |grad psi| is R dtheta: <f> = int f R dtheta /
    (2 pi R0)."""
    x = np.linspace(0.0, 1.0, n)
    r = a * np.sqrt(x)
    R_avg = R0 + r ** 2 / (2.0 * R0)
    inv_R = np.full_like(x, 1.0 / R0)
    inv_R2 = 1.0 / (R0 * np.sqrt(R0 ** 2 - r ** 2))
    # dV/dpsi = oint 2 pi R dl / |grad psi| = 2 pi^2 R0 a^2 / dpsi
    dV = np.full_like(x, 2.0 * np.pi ** 2 * R0 * a ** 2 / dpsi)
    return x, dict(psi_N=x, R_avg=R_avg, inv_R=inv_R, inv_R2=inv_R2,
                   dV_dpsi=dV, dpsi_dpsiN=dpsi), (R0, a, dpsi)


def _profiles(x):
    J = 1.1e6 * (1.0 - x ** 1.5) ** 1.3 + 4.0e4
    p = 5.0e4 * (1.0 - x) ** 2
    return J, p


def _direct_Ip(R0, a, dpsi, Jf, pf):
    """Gauss-Legendre area integral of j_phi(R, psi) over the disc."""
    from numpy.polynomial.legendre import leggauss
    nr, nt = 400, 256
    ur, wr = leggauss(nr)
    r = 0.5 * a * (ur + 1.0)
    wr = 0.5 * a * wr
    th = 2.0 * np.pi * (np.arange(nt) + 0.5) / nt
    R = R0 + r[:, None] * np.cos(th)[None, :]
    xr = (r / a) ** 2
    # p' per Wb: dp/dpsi = (dp/dpsi_N) / dpsi
    h = 1e-6
    pp = (pf(np.clip(xr + h, 0, 1)) - pf(np.clip(xr - h, 0, 1))) / (
        np.clip(xr + h, 0, 1) - np.clip(xr - h, 0, 1)) / dpsi
    R_avg = R0 + r ** 2 / (2.0 * R0)
    inv_R = 1.0 / R0
    ffp_mu0 = (Jf(xr) - R_avg * pp) / inv_R            # F F' / mu0
    jphi = R * pp[:, None] + ffp_mu0[:, None] / R
    return float(np.sum(jphi * r[:, None] * wr[:, None]) * (2.0 * np.pi / nt))


def test_the_ip_measure_is_the_area_integral_of_the_current_density():
    from bouquet.utils import Ip_fsa_weights
    x, geom, (R0, a, dpsi) = _circular()
    J, p = _profiles(x)
    geom["pprime"] = np.gradient(p, x) / dpsi
    w, c = Ip_fsa_weights(geom)
    Ip_pkg = float(np.trapezoid(w * J, x) + c)

    def Jf(xx):
        return np.interp(xx, x, J)

    def pf(xx):
        return 5.0e4 * (1.0 - xx) ** 2

    Ip_dir = _direct_Ip(R0, a, dpsi, Jf, pf)
    assert abs(Ip_pkg - Ip_dir) / abs(Ip_dir) < 1e-6, (Ip_pkg, Ip_dir)  # 1.6e-8 measured
    # the affine P' term is a real part of it (not a vanishing check)
    assert abs(c) > 1e-3 * abs(Ip_dir)


@pytest.mark.parametrize("err, ok", [(0.5, True), (1.0, True),
                                     (1.5, False), (-1.0, True),
                                     (-1.5, False)])
def test_the_li_row_criterion_binds_at_its_tolerance(err, ok):
    from types import SimpleNamespace
    from bouquet.engine import EngineRows, gfile_li_row_tol
    tol = gfile_li_row_tol()
    eng = SimpleNamespace(_pending=dict(li_predicted_plus_d=0.0))
    rows = EngineRows(eng, li_tol=tol)
    got, never, txt = rows.observe(0, dict(li=err * tol))
    assert got is ok and never is None
    assert rows.log["li_row_ok"] == [ok]
    assert rows.log["li_row_error"] == [err * tol]
