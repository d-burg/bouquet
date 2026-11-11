"""Reconnection operator of ``bouquet.kinrecon.current.sawtooth`` (synthetic / analytic inputs only).

Bounds and their sources:
* cylinder q = q0 + C s: rho_mix/rho_1 is the root of ln(1 + x/q0) = x (x = C s_mix), computed here
  independently; agreement to 1e-4 relative = the four digits to which the development prototype reproduced the
  scoping table (1.585, 1.500, 1.439, 1.417, 1.4144).  The exact roots are also checked against that table to half
  a unit of its last digit, except q0 = 0.7: the exact root 1.500524 rounds to 1.501, so the table's "1.500" is a
  truncation (found by this test; the stricter exact-root check applies to every entry);
* post-crash q(0) = 1 exactly in the continuum; |q0_post - 1| <= 3e-6 is the prototype's measured worst case;
* conservation residuals are exactly zero by construction (unchanged cells are copied, not recomputed);
* the identity cases are bitwise;
* onset: |psi_post - psi| = O((1 - q0)^2) and rho_mix = O((1 - q0)^(1/2)) (derived in the module docstring);
  fitted exponents within 2 +- 0.1 and 0.5 +- 0.05 over 1 - q0 in [1e-3, 1e-2];
* continuity through the change of topology (q0 crossing 1 with an off-axis q < 1 region): a Lipschitz map gives
  ||K(p + d) - K(p - d)|| proportional to d, so shrinking d tenfold must shrink the change tenfold; bound 0.12
  (0.1 plus an allowance of 0.02 for the O(d) curvature term); a jump would give ~1.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from bouquet.kinrecon.current import CurrentState
from bouquet.kinrecon.current.sawtooth import ReconnectionRecord, reconnect, reconnect_psi

PHI_B = 1.0
CF = PHI_B / (2.0 * np.pi)


def grid(n):
    rf = np.linspace(0.0, 1.0, n + 1)
    return rf, 0.5 * (rf[1:] + rf[:-1])


def psi_cells(qfun, rc):
    """psi(s) = c int_0^s ds'/q(s') (eq. S1), by adaptive quadrature."""
    return np.array([CF * quad(lambda x: 1.0 / qfun(x), 0.0, s, epsabs=0, epsrel=1e-13, limit=200)[0]
                     for s in rc ** 2])


TABLE = {0.5: (1.585, 5e-4), 0.7: (1.500, 5e-4), 0.9: (1.439, 5e-4), 0.99: (1.417, 5e-4), 0.999: (1.4144, 5e-5)}


@pytest.mark.parametrize("q0", sorted(TABLE))
def test_cylinder_mixing_ratio(q0):
    C = 3.0
    rf, rc = grid(400)
    psi = psi_cells(lambda s: q0 + C * s, rc)
    post, rec = reconnect_psi(psi, rf, PHI_B, f=1.0, sheet_width=0.0)
    x = brentq(lambda x: np.log1p(x / q0) - x, (1 - q0) * 1.0001, 50.0, xtol=1e-15)
    exact = np.sqrt(x / (1.0 - q0))
    ratio = rec.rho_mix / rec.rho_q1_outer
    assert rec.topology == "monotonic"
    assert abs(rec.rho_q1_outer / np.sqrt((1 - q0) / C) - 1) < 1e-4
    assert abs(ratio / exact - 1) < 1e-4
    table, half_unit = TABLE[q0]
    if q0 != 0.7:
        assert abs(exact - table) <= half_unit and abs(ratio - table) <= half_unit
    else:
        assert abs(exact - table) < 2 * half_unit
    assert abs(rec.q0_post - 1.0) <= 3e-6
    # q_new >= 1 inside the mixing region (from the post-crash cell differences)
    q_faces = PHI_B * rf[1:-1] / (np.pi * np.diff(post) / np.diff(rc))
    inside = rf[1:-1] < rec.rho_mix
    assert np.all(q_faces[inside] >= 1.0 - 3e-6)


def _states():
    rf, rc = grid(400)
    mono = psi_cells(lambda s: 0.7 + 2.0 * s, rc)
    rev = psi_cells(lambda s: 1.02 - 0.06 * np.exp(-((np.sqrt(s) - 0.15) / 0.08) ** 2) + 2.0 * s, rc)
    multi = psi_cells(lambda s: 0.95 + 2.0 * s - 0.3 * np.exp(-((np.sqrt(s) - 0.4) / 0.05) ** 2), rc)
    return rf, rc, dict(monotonic=mono, reversed=rev, multiple=multi)


@pytest.mark.parametrize("kind", ["monotonic", "reversed", "multiple"])
@pytest.mark.parametrize("f", [1.0, 0.37])
@pytest.mark.parametrize("w", [0.0, 0.05, 0.1])
def test_conservation_exact(kind, f, w):
    rf, rc, states = _states()
    psi = states[kind]
    st = CurrentState(rf, psi, float(psi[-1]) + 0.01, 0.3)
    out, rec = reconnect(st, PHI_B, f=f, sheet_width=w)
    assert isinstance(rec, ReconnectionRecord) and rec.active and rec.topology == kind
    assert rec.psi_outside_max_change == 0.0
    assert rec.current_outside_max_rel_change == 0.0
    assert rec.psi_boundary_change == 0.0
    assert out.psi_boundary == st.psi_boundary and out.time == st.time
    assert np.array_equal(out.rho_faces, st.rho_faces)          # toroidal flux: same grid
    outside = (rc >= rec.zone_outer) | (rc <= rec.zone_inner)
    assert np.array_equal(out.psi[outside], psi[outside])
    assert rec.n_changed_cells == np.count_nonzero(out.psi != psi)
    assert rec.dpsi_max > 0


def test_helical_flux_measure_conserved_full_reconnection():
    """f = 1, no smoothing: the toroidal-flux measure of {chi > L} is preserved (rearrangement, eq. K2).

    Checked with an independent linear interpolation of the cell values on two grids; the discrepancy is an
    interpolation error and must fall at least 3x when the grid is doubled (second order: 4x)."""
    errs = []
    for n in (200, 400):
        rf, rc = grid(n)
        psi = psi_cells(lambda s: 0.8 + 2.0 * s, rc)
        post, rec = reconnect_psi(psi, rf, PHI_B, 1.0, 0.0)
        smax = (rec.rho_mix * 1.05) ** 2
        xs = np.linspace(0.0, smax, 400001)
        s_nodes = np.concatenate(([0.0], rc ** 2))

        def chi_of(p):
            axis = p[0] - (p[1] - p[0]) * rc[0] ** 2 / (rc[1] ** 2 - rc[0] ** 2)
            return np.interp(xs, s_nodes, np.concatenate(([axis], p)) - CF * s_nodes)

        c0, c1 = chi_of(psi), chi_of(post)
        levels = np.linspace(c0.min(), c0.max(), 41)[1:-1]
        m0 = np.array([np.mean(c0 > L) for L in levels]) * smax
        m1 = np.array([np.mean(c1 > L) for L in levels]) * smax
        errs.append(np.max(np.abs(m1 - m0)) / smax)
    assert errs[1] < errs[0] / 3.0
    assert errs[1] < 1e-3


def test_identity_cases():
    rf, rc = grid(300)
    psi = psi_cells(lambda s: 1.05 + 2.0 * s, rc)                # q >= 1 everywhere
    post, rec = reconnect_psi(psi, rf, PHI_B, 1.0)
    assert not rec.active and rec.topology == "none" and np.array_equal(post, psi)
    psi_rs = psi_cells(lambda s: 1.3 - 0.2 * s + 1.5 * s * s, rc)  # reversed shear with q_min > 1
    post, rec = reconnect_psi(psi_rs, rf, PHI_B, 1.0)
    assert not rec.active and np.array_equal(post, psi_rs)
    psi_lo = psi_cells(lambda s: 0.6 + 2.0 * s, rc)               # q < 1 but f = 0
    post, rec = reconnect_psi(psi_lo, rf, PHI_B, 0.0)
    assert rec.active and np.array_equal(post, psi_lo)


def test_partial_reconnection_is_the_psi_blend():
    rf, rc = grid(300)
    psi = psi_cells(lambda s: 0.7 + 2.0 * s, rc)
    p1, _ = reconnect_psi(psi, rf, PHI_B, 1.0)
    pf, rec = reconnect_psi(psi, rf, PHI_B, 0.37)
    assert np.max(np.abs(pf - (psi + 0.37 * (p1 - psi)))) <= 4 * np.finfo(float).eps * np.max(np.abs(psi))
    # iota_post(0) = (1 - f) iota(0) + f, from the same interpolant (K5)
    assert abs(1.0 / rec.q0_post - (0.63 / rec.q0_pre + 0.37)) < 1e-5


@pytest.mark.parametrize("w", [0.0, 0.05])
def test_onset_scaling(w):
    rf, rc = grid(800)
    eps = np.geomspace(1e-3, 1e-2, 7)
    dpsi, rmix = [], []
    for e in eps:
        psi = psi_cells(lambda s, e=e: 1.0 - e + 0.5 * s, rc)
        post, rec = reconnect_psi(psi, rf, PHI_B, 1.0, w)
        dpsi.append(rec.dpsi_max)
        rmix.append(rec.rho_mix)
    p_psi = np.polyfit(np.log(eps), np.log(dpsi), 1)[0]
    p_mix = np.polyfit(np.log(eps), np.log(rmix), 1)[0]
    assert abs(p_psi - 2.0) < 0.1
    assert abs(p_mix - 0.5) < 0.05
    # and at q0 >= 1 (monotonic q) the operator is the identity
    psi = psi_cells(lambda s: 1.0 + 1e-6 + 0.5 * s, rc)
    assert not reconnect_psi(psi, rf, PHI_B, 1.0, w)[1].active


def test_continuity_through_topology_change():
    """q0 crossing 1 with an off-axis q < 1 dip (monotonic <-> reversed topology)."""
    rf, rc = grid(400)

    def family(p):
        return psi_cells(lambda s: p - 0.04 * np.exp(-((np.sqrt(s) - 0.12) / 0.06) ** 2) + 1.5 * s, rc)

    # find the parameter where the topology changes (axis slope of chi = 0)
    def axis_kind(p):
        return reconnect_psi(family(p), rf, PHI_B, 1.0)[1].topology

    lo, hi = 0.99, 1.02
    assert axis_kind(lo) == "monotonic" and axis_kind(hi) == "reversed"
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if axis_kind(mid) == "monotonic" else (lo, mid)
    p_star = 0.5 * (lo + hi)
    for w in (0.0, 0.05):
        ch = []
        for d in (1e-4, 1e-5):
            a = reconnect_psi(family(p_star + d), rf, PHI_B, 1.0, w)[0]
            b = reconnect_psi(family(p_star - d), rf, PHI_B, 1.0, w)[0]
            ch.append(np.max(np.abs(a - b)))
        assert ch[1] / ch[0] <= 0.12


def test_input_validation():
    rf, rc = grid(100)
    psi = psi_cells(lambda s: 0.7 + 2.0 * s, rc)
    with pytest.raises(ValueError):
        reconnect_psi(psi, rf, PHI_B, 1.2)
    with pytest.raises(ValueError):
        reconnect_psi(psi, rf, PHI_B, 1.0, sheet_width=-0.1)
    deep = psi_cells(lambda s: 0.3 + 0.4 * s, rc)                  # q < 1 everywhere
    with pytest.raises(ValueError):
        reconnect_psi(deep, rf, PHI_B, 1.0)
