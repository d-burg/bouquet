"""``bouquet.kinrecon.current.geometry``: identities, coordinate change, serialization.

Analytic geometries only (periodic cylinder; concentric circles in a torus).
"""
import ast
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon.current import FluxSurfaceGeometry  # noqa: E402
from bouquet.kinrecon.current import geometry as geo_mod  # noqa: E402


def _circ(n=401):
    return FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=n, fpol=lambda x: 3.4 * (1.0 - 0.03 * x ** 2))


def test_toroidal_flux_identity_cylinder_and_circle():
    # Phi(rho) = (1/2pi) int F <1/R^2> V' = Phi_b rho^2 (Lyons eq. 13); exact up to PCHIP quadrature
    assert FluxSurfaceGeometry.cylinder(0.5, 3.0, 2.0).consistency()["max_rel_error"] < 1e-13
    g = _circ()
    c = g.consistency()
    assert c["max_rel_error"] < 1e-8
    np.testing.assert_allclose(c["phi_b_from_geometry"], g.phi_b, rtol=1e-8)


def test_circular_large_aspect_ratio_tends_to_cylinder():
    a, B0 = 0.5, 2.0
    errs = []
    for R0 in (50.0, 500.0):
        g = FluxSurfaceGeometry.circular(a, R0, B0, n=101)
        c = FluxSurfaceGeometry.cylinder(a, R0, B0, n=101)
        errs.append(np.max(np.abs(g.avg_grad_rho2_r2 / c.avg_grad_rho2_r2 - 1)))
    # toroidal corrections scale as (a/R0)^2: a 10x larger R0 gives ~100x smaller error
    assert errs[1] < errs[0] / 50.0


def _psi_tabulation(g, npsi):
    """Tabulate g on normalized poloidal flux for a prescribed q(rho), as an equilibrium code would."""
    rho = g.rho
    q = 1.1 + 2.5 * rho ** 2
    dpsi_drho = g.phi_b * rho / (np.pi * q)
    from scipy.interpolate import PchipInterpolator
    anti = PchipInterpolator(rho, dpsi_drho).antiderivative()
    psi = anti(rho) - anti(0.0)
    psi_n = psi / psi[-1]
    # sampled uniformly in sqrt(psi_n) (~ uniform in rho near the axis), as from_tokamaker does
    target = np.linspace(1.0 / npsi, 1.0, npsi) ** 2
    rho_t = np.interp(target, psi_n, rho)
    # psi_n -> rho is smooth only in sqrt(psi_n); invert with Newton on the PCHIP antiderivative
    for _ in range(30):
        f = (anti(rho_t) - anti(0.0)) / psi[-1] - target
        rho_t -= f / (PchipInterpolator(rho, dpsi_drho)(rho_t) / psi[-1])
    ev = g.evaluate(rho_t)
    dpd = PchipInterpolator(rho, dpsi_drho)(rho_t)
    return dict(psi_n=target, psi_axis=0.0, psi_boundary=psi[-1], dvol_dpsi=ev["dvol_drho"] / dpd,
                avg_grad_psi2_r2=dpd ** 2 * ev["avg_grad_rho2_r2"], avg_inv_r2=ev["avg_inv_r2"], fpol=ev["fpol"])


def test_from_psi_coordinates_recovers_rho_geometry():
    g = _circ(n=2001)
    errs = []
    for npsi in (50, 100, 200):
        tab = _psi_tabulation(g, npsi)
        h = FluxSurfaceGeometry.from_psi_coordinates(**tab, B0=g.B0, R0=g.R0, nrho=101)
        ref = g.evaluate(h.rho)
        e = max(np.max(np.abs(h.avg_grad_rho2_r2 / ref["avg_grad_rho2_r2"] - 1)),
                np.max(np.abs(h.dvol_drho[1:] / ref["dvol_drho"][1:] - 1)),
                abs(h.phi_b / g.phi_b - 1))
        errs.append(e)
    # PCHIP integration/resampling is at least second order: each doubling must cut the error >= 4x
    assert errs[1] < errs[0] / 4.0 and errs[2] < errs[1] / 4.0, errs
    assert errs[2] < 1e-4


def test_dict_round_trip_and_validation():
    g = _circ(n=51)
    h = FluxSurfaceGeometry.from_dict(g.to_dict())
    for name in ("rho", "dvol_drho", "avg_grad_rho2_r2", "avg_inv_r2", "fpol", "avg_inv_r", "avg_r"):
        np.testing.assert_array_equal(getattr(g, name), getattr(h, name))
    d = g.to_dict()
    d["fpol"] = -d["fpol"]
    with pytest.raises(ValueError):
        FluxSurfaceGeometry.from_dict(d)
    d = g.to_dict()
    d["rho"] = d["rho"] * 0.99
    with pytest.raises(ValueError):
        FluxSurfaceGeometry.from_dict(d)


def test_no_equilibrium_solver_import():
    """The current package must run without OFT: from_tokamaker only calls methods of the object it is given."""
    pkg = os.path.dirname(geo_mod.__file__)
    for mod in ("geometry", "conductivity", "diffusion", "__init__"):
        tree = ast.parse(open(os.path.join(pkg, mod + ".py")).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("OpenFUSIONToolkit") for a in node.names), mod
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("OpenFUSIONToolkit"), mod


# ----------------------------------------------------------------------------- domain-edge rule (R29)
from bouquet.kinrecon.current import (OuterLayerGeometry, check_psi_samples, choose_psi_n_edge,  # noqa: E402
                                      PSI_N_EDGE_DEFAULT, PSI_N_EDGE_FALLBACK, PSI_N_EDGE_MAX)


def test_edge_rule_default_and_fallbacks():
    c, why = choose_psi_n_edge()
    assert c == PSI_N_EDGE_DEFAULT == 0.995 and "default" in why and "not checked" in why
    assert choose_psi_n_edge(grid_shape=(129, 129)).psi_n_edge == 0.995
    for shape in ((65, 65), (33, 33), (65, 129)):
        c, why = choose_psi_n_edge(grid_shape=shape)
        assert c == PSI_N_EDGE_FALLBACK == 0.99 and "65^2" in why
    assert choose_psi_n_edge(route_disagreement=9.9e-4).psi_n_edge == 0.995
    c, why = choose_psi_n_edge(route_disagreement=1.5e-3)
    assert c == 0.99 and "disagree" in why
    ch = choose_psi_n_edge(grid_shape=(257, 257), route_disagreement=2e-4)
    assert ch.psi_n_edge == 0.995 and ch.checks["route_disagreement"]["value"] == 2e-4
    # closed with margin (experiment recommendation 1): X-point 1e-3 beyond the cutoff
    assert choose_psi_n_edge(psi_n_xpoint=1.0000001).psi_n_edge == 0.995
    assert choose_psi_n_edge(psi_n_xpoint=0.9955).psi_n_edge == 0.99
    with pytest.raises(ValueError, match="no cutoff"):
        choose_psi_n_edge(psi_n_xpoint=0.9905)


def test_edge_rule_never_beyond_0p999():
    assert PSI_N_EDGE_MAX == 0.999
    c, why = choose_psi_n_edge(requested=0.999)
    assert c == 0.999 and "requested" in why and "beyond the rule" in why
    for bad in (0.9995, 1.0, 0.0):
        with pytest.raises(ValueError):
            choose_psi_n_edge(requested=bad)
    c, why = choose_psi_n_edge(requested=0.98, grid_shape=(65, 65))
    assert c == 0.98 and "0.99" in why


# ----------------------------------------------------------------------------- duplicate-sample guard
def test_duplicate_samples_exact_repeat_is_merged():
    x = np.linspace(0.01, 1.0, 50) ** 2
    v = np.sin(x)
    xd = np.insert(x, 21, x[20] + 1e-16)
    vd = np.insert(v, 21, v[20])
    xo, arr, merged = check_psi_samples(xd, dict(v=vd))
    np.testing.assert_array_equal(xo, x)
    np.testing.assert_array_equal(arr["v"], v)
    assert merged == [21]


def test_duplicate_samples_with_different_values_raise():
    """Two samples 1e-16 apart from concatenated, separately traced grids (the case the domain-edge
    experiment hit): the values differ at the tracing-noise level, so the guard refuses to choose."""
    x = np.linspace(0.01, 1.0, 50) ** 2
    v = np.sin(x)
    xd = np.insert(x, 21, x[20] + 1e-16)
    vd = np.insert(v, 21, v[20] * (1 + 1e-7))
    with pytest.raises(ValueError, match="cannot be resolved"):
        check_psi_samples(xd, dict(v=vd))
    g = _circ(n=201)
    tab = _psi_tabulation(g, 100)
    i = 60
    for k in ("psi_n", "dvol_dpsi", "avg_grad_psi2_r2", "avg_inv_r2", "fpol"):
        tab[k] = np.insert(tab[k], i + 1, tab[k][i] * (1 + 3e-8) if k != "psi_n" else tab[k][i] + 1e-16)
    with pytest.raises(ValueError, match="cannot be resolved"):
        FluxSurfaceGeometry.from_psi_coordinates(**tab, B0=g.B0, R0=g.R0)


def test_isolated_near_duplicate_raises_and_graded_grids_pass():
    x = np.linspace(0.05, 0.95, 40)
    xd = np.sort(np.append(x, x[10] + 1e-6))      # gap 1e-6 between gaps of 0.023
    with pytest.raises(ValueError, match="isolated near-duplicate"):
        check_psi_samples(xd, dict(v=np.sin(xd)))
    # geometric clustering toward a cutoff and sqrt-uniform spacing are accepted unchanged
    c = 0.995
    graded = np.concatenate((np.linspace(np.sqrt(1e-3), np.sqrt(0.97 * c), 300)[:-1] ** 2,
                             c - np.geomspace(0.03 * c, 1e-3 * (1 - c), 100), [c]))
    xo, _, merged = check_psi_samples(graded)
    assert merged == [] and xo.size == graded.size
    with pytest.raises(ValueError, match="increase"):
        check_psi_samples(x[::-1])


def test_from_psi_coordinates_truncated_at_cutoff():
    """psi_n_edge: rho is normalized to Phi(c); the result equals the untruncated construction restricted to
    psi_N <= c and renormalized (same samples), and the last sample must be the cutoff itself."""
    g = _circ(n=2001)
    tab = _psi_tabulation(g, 200)
    c = tab["psi_n"][150]
    sub = {k: (v[:151] if isinstance(v, np.ndarray) else v) for k, v in tab.items()}
    h = FluxSurfaceGeometry.from_psi_coordinates(**sub, B0=g.B0, R0=g.R0, nrho=101, psi_n_edge=c)
    assert h.meta["psi_n_edge"] == c and "truncated" in h.meta["edge_treatment"]
    np.testing.assert_array_equal(h.meta["psi_n_samples"], sub["psi_n"])
    # Phi(c) from the full construction at the same samples
    full = FluxSurfaceGeometry.from_psi_coordinates(**tab, B0=g.B0, R0=g.R0, nrho=101)
    rho_c = full.meta["rho_samples"][150]
    # the two constructions differ only by the PCHIP end-slope at the cutoff (one-sided vs interior), i.e. by
    # the interpolation error of the construction, bounded by 1e-4 at npsi = 200 in
    # test_from_psi_coordinates_recovers_rho_geometry
    np.testing.assert_allclose(h.phi_b, full.phi_b * rho_c ** 2, rtol=1e-4)
    np.testing.assert_allclose(h.meta["rho_samples"], full.meta["rho_samples"][:151] / rho_c, rtol=1e-4)
    bad = dict(sub)
    with pytest.raises(ValueError, match="cutoff surface"):
        FluxSurfaceGeometry.from_psi_coordinates(**bad, B0=g.B0, R0=g.R0, psi_n_edge=c + 1e-6)
    assert full.meta["psi_n_edge"] == 1.0


def test_outer_layer_volume_tail_is_exact_for_log_divergence():
    """dV/dpsi_N = a + b ln(1/(1 - x)) (the X-point form): the fitted tail beyond the last sample is exact."""
    a, b, c, dpsi = 30.0, 4.0, 0.995, 1.0
    xl = 1.0 - (1.0 - c) * np.geomspace(1.0, 1e-2, 30)
    vps = (a + b * np.log(1.0 / (1.0 - xl))) / dpsi
    lay = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, dpsi, vps, np.full(30, 0.3), np.full(30, 3.4))
    d = 1.0 - xl[-1]
    np.testing.assert_allclose(lay.volume_tail, d * (a + b * (np.log(1.0 / d) + 1.0)), rtol=1e-10)
    exact_total = (1 - c) * (a + b * (np.log(1.0 / (1 - c)) + 1.0))
    # samples-part accuracy: PCHIP on 30 geometric samples (the integral of a log is smooth in this spacing)
    assert abs(lay.total_volume / exact_total - 1.0) < 1e-4
    assert lay.psi_n_edge == c and lay.fpol_lcfs == 3.4 and "log fit" in lay.meta["tail_method"]
    # integral of a constant density is the volume
    np.testing.assert_allclose(lay.integral(np.ones(30)), lay.total_volume, rtol=1e-14)
