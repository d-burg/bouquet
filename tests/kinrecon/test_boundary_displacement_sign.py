"""Sign of the boundary displacement (:meth:`bouquet.kinrecon.forward.ForwardModel.boundary_distance`), without OFT:
an analytic circular equilibrium of minor radius ``a`` stands in for the solver.  A reference (anchor) circle of
radius ``a - d`` lies INSIDE the solved boundary, so the solved boundary is OUTSIDE the anchor and every signed
displacement (normal mean, outboard and inboard midplane) is ``+d`` (first order in ``d``); with ``a + d`` it is
``-d``.  Bounds (fixed before the first run): midplane crossings to 0.01 mm (bisection on a 1 mm scan), the
first-order normal mean to 2 % of ``d`` (its second-order error is d/(2a) = 1 %)."""
import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as DG
from bouquet.kinrecon import forward as FW

R0, A, PSI_A, PSI_B = 1.7, 0.5, 0.2, -0.3       # psi falls outward here: the convention must not depend on it


class _Eval:
    def __init__(self, field):
        self.field = field

    def eval(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        r, z = pts[:, 0], pts[:, 1]
        psi = PSI_A + (PSI_B - PSI_A) * ((r - R0) ** 2 + z ** 2) / A ** 2
        if self.field == "PSI":
            return psi[:, None]
        dpsi_dr = (PSI_B - PSI_A) * 2.0 * (r - R0) / A ** 2
        dpsi_dz = (PSI_B - PSI_A) * 2.0 * z / A ** 2
        return np.column_stack((-dpsi_dz / r, np.full_like(r, 1.0), dpsi_dr / r))


class _GS:
    psi_bounds = (PSI_B, PSI_A)
    o_point = (R0, 0.0)
    lim_contour = None

    def get_field_eval(self, field):
        return _Eval(field)

    def abspsi_to_normalized(self, psi):
        return (np.asarray(psi, float) - PSI_A) / (PSI_B - PSI_A)

    def get_xpoints(self):
        return None, None


def _model(radius):
    fm = object.__new__(FW.ForwardModel)
    th = np.linspace(0.0, 2.0 * np.pi, 181)[:-1]
    fm.gs = _GS()
    fm.reference_boundary = np.column_stack((R0 + radius * np.cos(th), radius * np.sin(th)))
    fm.reference_points = {}
    fm.settings = FW.ForwardSettings()
    return fm


@pytest.mark.parametrize("d_mm", [10.0, -10.0])
def test_positive_displacement_means_the_solved_boundary_is_outside_the_anchor(monkeypatch, d_mm):
    monkeypatch.setattr(DG, "on_mesh", lambda eq, pts: np.ones(np.asarray(pts).reshape(-1, 2).shape[0], bool))
    out = _model(A - 1e-3 * d_mm).boundary_distance()
    d = out["displacement_mm"]
    assert "error" not in d, d
    assert d["outboard_midplane"] == pytest.approx(d_mm, abs=0.01)
    assert d["inboard_midplane"] == pytest.approx(d_mm, abs=0.01)
    assert d["mean"] == pytest.approx(d_mm, rel=0.02)
    assert d["rms"] == pytest.approx(abs(d_mm), rel=0.02) and d["max"] == pytest.approx(abs(d_mm), rel=0.02)
    assert d["sign"].startswith("positive: the solved boundary lies outside")
