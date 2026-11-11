"""Finite-length poloidal-field probes (EFIT convention) in ``bouquet.kinrecon.diagnostics``.

Hand calculations on an analytic stand-in field: the averaging points are written out
explicitly here (independently of ``probe_segment_points``) and the field is evaluated term by
term.  Synthetic numbers only; no OFT needed.
"""
import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as diag
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
from bouquet.kinrecon.score import score

R0, A = 1.7, 0.5
PSI_AX, PSI_B = -1.0, -0.2
F0 = -3.4


def _B(R, Z):
    """Analytic (B_R, B_phi, B_Z) of psi = PSI_AX + (PSI_B - PSI_AX) ((R-R0)^2 + Z^2)/A^2."""
    c = (PSI_B - PSI_AX) / A ** 2
    return -2 * c * Z / R, F0 / R, 2 * c * (R - R0) / R


class _Interp:
    def eval(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        br, bp, bz = _B(pts[:, 0], pts[:, 1])
        return np.column_stack((br, bp, bz))


class FakeEq:
    def __init__(self):
        # mesh: the rectangle 1.0 <= R <= 2.4, -1 <= Z <= 1
        self.r = np.array([[1.0, -1.0, 0.0], [2.4, -1.0, 0.0], [2.4, 1.0, 0.0], [1.0, 1.0, 0.0]])
        self.lc = np.array([[0, 1, 2], [0, 2, 3]])
        self.lim_contour = self.r[:, :2]

    def get_field_eval(self, kind):
        assert kind.upper() == "B"
        return _Interp()


def _bn(R, Z, nr, nz):
    br, _, bz = _B(R, Z)
    return br * nr + bz * nz


def test_two_point_midpoint_rule_by_hand_along_and_across():
    eq = FakeEq()
    c = np.array([2.2, 0.3])
    n = np.array([0.6, 0.8])                       # unit probe direction
    L = 0.2
    # n_avg = 2: offsets (k - 1/2) L / 2 for k = 0, 1 -> -L/4, +L/4
    p1, p2 = c - 0.05 * n, c + 0.05 * n           # along n (L > 0)
    expect_along = 0.5 * (_bn(*p1, *n) + _bn(*p2, *n))
    t = np.array([-0.8, 0.6])                      # n rotated by +90 degrees (L < 0)
    q1, q2 = c - 0.05 * t, c + 0.05 * t
    expect_across = 0.5 * (_bn(*q1, *n) + _bn(*q2, *n))
    geo = {"r": [c[0], c[0]], "z": [c[1], c[1]], "direction_r": [n[0], n[0]],
           "direction_z": [n[1], n[1]], "length": [L, -L]}
    syn = diag.bpol_probe(eq, geo, n_avg=2)
    assert syn.value[0] == pytest.approx(expect_along, rel=1e-14)
    assert syn.value[1] == pytest.approx(expect_across, rel=1e-14)
    assert list(syn.reason) == ["", ""]
    # the two placements really differ from the point value here
    point = _bn(*c, *n)
    assert abs(expect_along - point) > 1e-6 and abs(expect_across - point) > 1e-6


def test_default_25_points_by_hand_and_against_the_segment_integral():
    eq = FakeEq()
    c = np.array([1.3, -0.1])
    n = np.array([0.0, 1.0])                       # vertical probe, horizontal segment for L < 0
    L = -0.14
    t = np.array([-1.0, 0.0])
    offs = [(k - 12) * abs(L) / 25 for k in range(25)]
    expect = np.mean([_bn(*(c + s * t), *n) for s in offs])
    syn = diag.bpol_probe(eq, {"r": [c[0]], "z": [c[1]], "direction_r": [n[0]], "direction_z": [n[1]],
                               "length": [L]})
    assert syn.value[0] == pytest.approx(expect, rel=1e-14)
    # B_Z = 2 c (R - R0)/R along the segment: exact mean = 2c (1 - R0 ln(R2/R1)/(R2 - R1))
    cc = (PSI_B - PSI_AX) / A ** 2
    r1, r2 = c[0] - 0.07, c[0] + 0.07
    exact = 2 * cc * (1 - R0 * np.log(r2 / r1) / (r2 - r1))
    assert syn.value[0] == pytest.approx(exact, rel=1e-4)       # midpoint-rule error O((L/R)^2/n^2)


def test_point_model_zero_length_and_missing_length_agree():
    eq = FakeEq()
    c, n = (2.0, 0.1), (0.6, 0.8)
    point = _bn(*c, *n)
    base = {"r": [c[0]], "z": [c[1]], "direction_r": [n[0]], "direction_z": [n[1]]}
    assert diag.bpol_probe(eq, base).value[0] == pytest.approx(point, rel=1e-14)
    assert diag.bpol_probe(eq, dict(base, length=[0.0])).value[0] == pytest.approx(point, rel=1e-14)
    assert diag.bpol_probe(eq, dict(base, length=[0.2]), model="point").value[0] == pytest.approx(point,
                                                                                                rel=1e-14)
    with pytest.raises(ValueError):
        diag.bpol_probe(eq, base, model="averaged")


def test_segment_leaving_the_mesh_is_not_modelable():
    eq = FakeEq()
    # center on the mesh (R = 2.35), segment along +R with |L| = 0.2: the outer end is at R = 2.45
    geo = {"r": [2.35, 2.35], "z": [0.0, 0.0], "direction_r": [1.0, 1.0], "direction_z": [0.0, 0.0],
           "length": [0.2, 0.08]}
    syn = diag.bpol_probe(eq, geo)
    assert np.isnan(syn.value[0]) and syn.reason[0] == diag.REASON_SEGMENT_OFF_MESH
    # a shorter segment (ends at 2.31 and 2.39) stays inside and is modeled
    assert np.isfinite(syn.value[1]) and syn.reason[1] == ""
    # with the point model the same probe is modeled at its center
    assert np.isfinite(diag.bpol_probe(eq, geo, model="point").value[0])
    # end points are checked even when every midpoint is inside: L = 0.10 + tiny puts the
    # outermost midpoint at R = 2.398 but the end at 2.40 + eps
    geo2 = dict(geo, r=[2.35], z=[0.0], direction_r=[1.0], direction_z=[0.0], length=[0.1000001])
    s2 = diag.bpol_probe(eq, geo2)
    assert np.isnan(s2.value[0]) and s2.reason[0] == diag.REASON_SEGMENT_OFF_MESH


def test_segment_points_helper_and_score_meta():
    pts, ends = diag.probe_segment_points({"r": [2.0], "z": [0.0], "direction_r": [0.0],
                                           "direction_z": [2.0], "length": [0.5]}, n_avg=5)
    np.testing.assert_allclose(pts[0, :, 1], [-0.2, -0.1, 0.0, 0.1, 0.2], atol=1e-15)
    np.testing.assert_allclose(ends[0], [[2.0, -0.25], [2.0, 0.25]], atol=1e-15)
    eq = FakeEq()
    geo = {"r": [2.2], "z": [0.3], "direction_r": [0.6], "direction_z": [0.8], "length": [0.2]}
    avg = diag.bpol_probe(eq, geo).value
    g = ConstraintGroup(name="bpol_probe", labels=["p0"], measured=avg, sigma=np.array([1e-3]),
                        valid=np.array([True]), geometry=geo, meta={"sigma_model": "test"})
    tab = score(eq, MeasurementSet("synthetic", 1.0, 0.01, {"bpol_probe": g}))
    assert tab.chi2("bpol_probe") == 0.0
    assert any("finite-length" in s for s in tab.notes)
    g.meta["probe_model"] = "point"
    tab = score(eq, MeasurementSet("synthetic", 1.0, 0.01, {"bpol_probe": g}))
    assert tab.chi2("bpol_probe") > 0.0
