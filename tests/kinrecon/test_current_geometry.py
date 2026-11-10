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
