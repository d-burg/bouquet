"""The boundary (isoflux) constraint per unit length (#84).

The unified engine resamples the source boundary to ``N_ISOFLUX`` points by
arc length (closed, counter-clockwise, X-points kept exactly) and weights each
point by the arc length it represents, the weights summing to
``ISOFLUX_WEIGHT_TOTAL`` -- one constant for the reconstruction and the draws,
so the strength no longer follows the source's point count.  A
``SolverConfig.isoflux_pts`` set is used as given; the legacy route keeps the
source boundary at ``ISOFLUX_WEIGHT`` per point.  Synthetic geometry and the
example data only; no solver.
"""
import copy
import os

import numpy as np
import pytest

from bouquet.solver_geometry import (
    ISOFLUX_RULE_LEGACY, ISOFLUX_RULE_STANDARD, ISOFLUX_RULE_USER, ISOFLUX_WEIGHT,
    ISOFLUX_WEIGHT_TOTAL, N_ISOFLUX, XPOINT_TURN_DEG, case_geometry, find_xpoints,
    per_length_weights, resample_boundary, standard_isoflux, standard_isoflux_for)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")


def _shape(m, corner=0.8, roll=0, top_corner=0.0):
    """A D-shaped closed contour sampled at ``m`` points (no repeated point): with ``corner`` > 0
    an X-point corner at the bottom (exactly a sample point), with ``top_corner`` > 0 a second one
    at the top; rolled by ``roll`` samples so the source starts anywhere."""
    assert m % 2 == 0
    phi = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    th = phi - np.pi / 2
    R = 1.7 + 0.6 * np.cos(th + 0.4 * np.sin(th))
    Z = 1.02 * np.sin(th)
    Z = Z - corner * (1.0 - np.sin(phi / 2))                     # bottom corner at phi = 0
    if top_corner:
        Z = Z + top_corner * (1.0 - np.abs(np.cos(phi / 2)))    # top corner at phi = pi
    return np.roll(np.column_stack([R, Z]), roll, axis=0)


def _spacing(pts):
    return np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)


def _signed_area(pts):
    return 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])


# ── the resampling ────────────────────────────────────────────────────────────────────────────────
def test_the_resampling_has_the_fixed_count_closed_and_uniform_in_arc_length():
    src = _shape(400, corner=0.0)                                  # smooth, no X-point
    pts, arc, info = resample_boundary(src)
    assert pts.shape == (N_ISOFLUX, 2) and len(arc) == N_ISOFLUX
    assert info["n"] == N_ISOFLUX and info["source_n"] == 400 and info["x_points"] == []
    # a closed contour, no repeated closing point: the gap back to the first point is one spacing
    d = _spacing(pts)
    assert np.all(d > 0)
    assert d[-1] == pytest.approx(np.mean(d), rel=1e-3)
    # uniform in arc length (the chords of a uniform arc-length spacing vary only with curvature)
    assert np.std(d) / np.mean(d) < 1e-3
    np.testing.assert_allclose(arc, info["boundary_length_m"] / N_ISOFLUX, rtol=1e-12)
    assert np.sum(arc) == pytest.approx(info["boundary_length_m"], rel=1e-12)
    # the chords of the resampled polygon fall short of the curve's arc length by its sagitta only
    L = info["boundary_length_m"]
    assert L * (1.0 - 2e-4) < np.sum(d) <= L * (1.0 + 1e-12)
    # counter-clockwise, first point the outermost point
    assert _signed_area(pts) > 0
    assert pts[0, 0] == pytest.approx(np.max(pts[:, 0]), abs=1e-9)
    # a clockwise source with its closing point repeated gives the same set
    cw = np.vstack([src[::-1], src[-1:]])
    np.testing.assert_allclose(resample_boundary(cw)[0], pts, rtol=0, atol=1e-9)


def test_the_x_point_is_kept_exactly_and_the_outermost_point_is_the_reference():
    src = _shape(200, corner=0.8, roll=37)
    P, xp = find_xpoints(src)
    assert len(xp) == 1 and xp[0]["origin"] == "turning angle"
    assert xp[0]["turn_deg"] > XPOINT_TURN_DEG
    corner = _shape(200, corner=0.8)[0]                            # phi = 0, exactly
    pts, arc, info = resample_boundary(src)
    assert np.any(np.all(pts == corner, axis=1))                   # exact, not interpolated
    assert info["x_points"][0]["R"] == corner[0] and info["x_points"][0]["Z"] == corner[1]
    # TokaMaker's isoflux reference (the first point) is the outermost point, never the X-point
    assert info["first_point"]["kind"].startswith("outermost point")
    assert pts[0, 0] == np.max(pts[:, 0]) and tuple(pts[0]) != tuple(corner)
    # spacing stays uniform right up to the corner on both sides
    d = _spacing(pts)
    assert np.std(d) / np.mean(d) < 5e-3
    # the smooth contour has none
    assert find_xpoints(_shape(200, corner=0.0))[1] == []


def test_two_x_points_split_the_points_by_length_and_both_are_exact():
    src = _shape(400, corner=0.8, top_corner=0.9, roll=11)
    pts, arc, info = resample_boundary(src)
    assert len(info["x_points"]) == 2
    raw = _shape(400, corner=0.8, top_corner=0.9)
    for xy in (raw[0], raw[200]):                                  # phi = 0 and phi = pi
        assert np.any(np.all(pts == xy, axis=1)), "an X-point was not kept exactly"
    assert pts[0, 0] == np.max(pts[:, 0])                         # the reference: outermost
    assert pts.shape == (N_ISOFLUX, 2)
    assert np.sum(arc) == pytest.approx(info["boundary_length_m"], rel=1e-12)


def test_a_saddle_target_marks_a_corner_the_angle_test_misses():
    src = _shape(200, corner=0.25)                                 # a ~25 deg corner: below the test
    assert find_xpoints(src)[1] == []
    corner = src[0]
    P, xp = find_xpoints(src, saddle_targets=[[corner[0] + 0.004, corner[1] - 0.003]])
    assert len(xp) == 1 and xp[0]["origin"] == "saddle target 0"
    assert xp[0]["xpoint_match"] == "nearest"                      # nothing above the angle test
    pts, _, info = resample_boundary(src, saddle_targets=[[corner[0] + 0.004, corner[1] - 0.003]])
    assert np.any(np.all(pts == corner, axis=1))
    # a saddle target far from the boundary marks nothing
    assert find_xpoints(src, saddle_targets=[[1.7, 0.0]])[1] == []


@pytest.mark.parametrize("offset_mm", [6.0, 12.0, 30.0])
def test_a_saddle_target_off_the_x_point_still_marks_the_x_point(offset_mm):
    """A 256-point single null with a 90 deg corner, vertex spacing ~17 mm, and a saddle target
    moved off the X-point along the outboard side: the corner (the sharpest vertex in the match
    radius), not the nearest vertex, is marked and kept exactly."""
    src = _shape(256, corner=1.0)
    src = np.array([1.7, 0.0]) + 0.63 * (src - np.array([1.7, 0.0]))   # spacing ~17 mm
    P, xp0 = find_xpoints(src)
    c = xp0[0]["index"]
    assert np.all(P[c] == src[0]) and xp0[0]["turn_deg"] > 80.0
    leg = (P[c + 1] - P[c]) / np.hypot(*(P[c + 1] - P[c]))          # along the outboard edge
    tgt = P[c] + 1e-3 * offset_mm * leg
    nearest = int(np.argmin(np.hypot(*(P - tgt).T)))
    if offset_mm >= 12.0:
        assert nearest != c                                            # what nearest-vertex did
    _, xp = find_xpoints(src, saddle_targets=[tgt])
    assert [x["index"] for x in xp] == [c]
    assert xp[0]["origin"] == "saddle target 0" and xp[0]["xpoint_match"] == "largest turn"
    pts, _, info = resample_boundary(src, saddle_targets=[tgt])
    assert np.any(np.all(pts == P[c], axis=1))
    assert info["x_points"][0]["xpoint_match"] == "largest turn"


def test_a_degenerate_boundary_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        resample_boundary(np.array([[1.0, 0.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        resample_boundary(np.array([[1.0, 0.0], [2.0, np.nan], [1.5, 1.0]]))


# ── the density-aware spline breaks ───────────────────────────────────────────────────────────────
_DATA = os.path.join(_HERE, "data")


def _fixture_boundaries():
    """The shipped fixtures' source boundaries: the synthetic g-files, the synthetic dd's slices
    and the D3D-like example (and every third point of it)."""
    import gzip
    import json
    from bouquet.io.geqdsk import read_geqdsk
    out = {}
    for f in ("g_synthetic.geqdsk", "g_synthetic_pfast.geqdsk", "g_synthetic_lmode.geqdsk",
              "d3dlike.geqdsk"):
        g = read_geqdsk(os.path.join(_DATA, f))
        out[f] = np.column_stack([g.boundary_R, g.boundary_Z])
    with gzip.open(os.path.join(_DATA, "dd_synthetic.json.gz"), "rt") as fh:
        dd = json.load(fh)
    for i, ts in enumerate(dd["equilibrium"]["time_slice"]):
        o = ts["boundary"]["outline"]
        out[f"dd slice {i}"] = np.column_stack([o["r"], o["z"]])
    out["example"] = _gfile_boundary()
    out["example, every third point"] = out["example"][::3]
    return out


def _old_kinks(P):
    """The pre-fix break rule: the vertex's own turn above SPLINE_BREAK_DEG, nothing else."""
    from bouquet.solver_geometry import SPLINE_BREAK_DEG, _vertex_turns
    return np.flatnonzero(_vertex_turns(P) > SPLINE_BREAK_DEG)


def test_the_density_aware_break_rule_leaves_the_fixtures_identical(monkeypatch):
    """The relative test only removes breaks a coarse smooth source would otherwise get: on every
    shipped fixture boundary the break vertices and the resampled set are bit-identical to the
    fixed-angle rule's."""
    import bouquet.solver_geometry as sg
    for name, b in _fixture_boundaries().items():
        P = sg._open_ccw(b)
        np.testing.assert_array_equal(sg._local_kinks(P), _old_kinks(P), err_msg=name)
        new = sg.resample_boundary(b)
        with monkeypatch.context() as m:
            m.setattr(sg, "_local_kinks", _old_kinks)
            old = sg.resample_boundary(b)
        np.testing.assert_array_equal(new[0], old[0], err_msg=name)
        np.testing.assert_array_equal(new[1], old[1], err_msg=name)
        assert new[2]["spline_breaks"] == old[2]["spline_breaks"], name


def _smooth_d(m):
    """A smooth D (R0 1.7 m, a 0.6 m, kappa 1.8, delta 0.5, no X-point) sampled at ``m`` points."""
    th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    return np.column_stack([1.7 + 0.6 * np.cos(th + 0.5 * np.sin(th)), 1.8 * 0.6 * np.sin(th)])


@pytest.mark.parametrize("m, dev_mm", [(16, 7.0), (20, 3.0), (24, 1.5), (30, 0.5)])
def test_a_coarse_smooth_source_is_not_cut_into_a_polygon(m, dev_mm):
    """At 16-30 source points the tips of a smooth D turn by 29-46 deg per vertex; the fixed angle
    alone broke the spline there and left the set on the chords (6-24 mm inside the curve).  Now
    no vertex breaks and the set lies on the periodic spline (its own error: 6.3 / 2.6 / 1.1 /
    0.43 mm at 16 / 20 / 24 / 30 points)."""
    from bouquet.solver_geometry import _local_kinks, _open_ccw
    src = _smooth_d(m)
    assert len(_local_kinks(_open_ccw(src))) == 0
    assert len(_old_kinks(_open_ccw(src))) > 0                       # what the fixed angle did
    pts, _, info = resample_boundary(src)
    assert info["spline_breaks"] == 0 and info["x_points"] == []
    true = _smooth_d(2 * 10**5)
    seg_d = []
    for p in pts:                                                    # distance to the true curve
        seg_d.append(np.min(np.hypot(true[:, 0] - p[0], true[:, 1] - p[1])))
    assert 1e3 * max(seg_d) < dev_mm


def test_a_90_degree_corner_still_breaks_the_spline():
    from bouquet.solver_geometry import XPOINT_TURN_DEG, _local_kinks, _open_ccw
    src = _shape(256, corner=1.0)                                    # ~90 deg over the window
    P = _open_ccw(src)
    corner = int(np.flatnonzero(np.all(P == src[0], axis=1))[0])
    assert corner in set(_local_kinks(P).tolist())                   # by the break rule itself
    _, xp = find_xpoints(src)
    assert [x["index"] for x in xp] == [corner] and xp[0]["turn_deg"] > XPOINT_TURN_DEG
    pts, _, info = resample_boundary(src)
    assert info["spline_breaks"] >= 1
    assert np.any(np.all(pts == src[0], axis=1))


# ── the weights ───────────────────────────────────────────────────────────────────────────────────
def test_the_weights_sum_to_the_total_and_follow_the_arc_length():
    w = per_length_weights(np.array([1.0, 2.0, 3.0, 2.0]))
    assert np.sum(w) == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-14)
    np.testing.assert_allclose(w / np.array([1.0, 2.0, 3.0, 2.0]), w[0], rtol=1e-14)
    # two X-points: the two stretches are spaced differently, the weights per unit length are not
    pts, w, info = standard_isoflux(_shape(400, corner=0.8, top_corner=0.9))
    _, arc, _ = resample_boundary(_shape(400, corner=0.8, top_corner=0.9))
    assert np.sum(w) == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    assert info["weight_total"] == ISOFLUX_WEIGHT_TOTAL
    np.testing.assert_allclose(w / arc, ISOFLUX_WEIGHT_TOTAL / info["boundary_length_m"],
                               rtol=1e-12)
    with pytest.raises(ValueError):
        per_length_weights(np.array([1.0, 0.0]))


@pytest.mark.parametrize("corner", [0.8, 0.0])
def test_the_constraint_does_not_depend_on_the_sources_point_count(corner):
    """A 100-point and a 10 000-point sampling of the same contour, starting at different places:
    the same total weight and (near-)identical point sets and weights."""
    a_pts, a_w, a_info = standard_isoflux(_shape(100, corner=corner, roll=17))
    b_pts, b_w, b_info = standard_isoflux(_shape(10000, corner=corner, roll=4321))
    assert a_info["source_n"] == 100 and b_info["source_n"] == 10000
    assert np.sum(a_w) == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    assert np.sum(b_w) == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    assert a_pts.shape == b_pts.shape == (N_ISOFLUX, 2)
    assert np.max(np.hypot(*(a_pts - b_pts).T)) < 2e-5               # 20 micrometres
    np.testing.assert_allclose(a_w, b_w, rtol=1e-9)
    assert a_info["boundary_length_m"] == pytest.approx(b_info["boundary_length_m"], rel=1e-6)
    # ... where the old rule (every source point at one weight) differed by the point count
    assert 10000 * ISOFLUX_WEIGHT / (100 * ISOFLUX_WEIGHT) == 100


# ── the solver geometry: rule, record, user set, legacy ───────────────────────────────────────────
def _gsrc():
    from bouquet.config import ReconstructionSource
    return ReconstructionSource(
        geqdsk_path=os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles_path=os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk"))


def _sc():
    from bouquet.config import SolverConfig
    return SolverConfig(mesh_path=os.path.join(_EX, "DIIID_mesh.h5"))


def _gfile_boundary():
    from bouquet.io.geqdsk import read_geqdsk
    g = read_geqdsk(os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"))
    return np.column_stack([g.boundary_R, g.boundary_Z])


def test_the_record_stamps_the_standard_constraint():
    geo = case_geometry(_gsrc(), _sc())
    rec = geo.record()
    src = _gfile_boundary()
    assert rec["isoflux_rule"] == ISOFLUX_RULE_STANDARD
    assert rec["isoflux_n"] == N_ISOFLUX and len(geo.iso_pts) == N_ISOFLUX
    assert rec["isoflux_weight_total"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    assert rec["isoflux_weights"]["total"] == rec["isoflux_weight_total"]
    assert rec["isoflux_source_n"] == len(src)
    rs = rec["isoflux_resampling"]
    assert rs["method"] == "arc_length_resample/1" and rs["n"] == N_ISOFLUX
    assert rs["source_n"] == len(src)
    assert rs["weight_total"] == ISOFLUX_WEIGHT_TOTAL
    assert rec["boundary_length_m"] == rs["boundary_length_m"] > 0
    # the example g-file is diverted: its X-point is kept, exactly a source vertex
    assert len(rs["x_points"]) == 1
    xp = (rs["x_points"][0]["R"], rs["x_points"][0]["Z"])
    assert np.any(np.all(src == xp, axis=1))
    assert np.any(np.all(geo.iso_pts == xp, axis=1))
    # JSON-safe (it is archived as solver_geometry_json)
    import json
    assert json.loads(json.dumps(rec))["isoflux_resampling"]["x_points"][0]["R"] == xp[0]


def test_a_user_supplied_set_is_used_as_given():
    sc = _sc()
    pts = _shape(40, corner=0.0)
    w = np.linspace(1.0, 2.0, 40)
    sc.isoflux_pts, sc.isoflux_weights = pts, w
    geo = case_geometry(_gsrc(), sc)
    assert geo.iso_pts is pts and geo.iso_w is w                   # untouched, not resampled
    rec = geo.record()
    assert rec["isoflux_rule"] == ISOFLUX_RULE_USER
    assert rec["isoflux_origin"] == "SolverConfig.isoflux_pts"
    assert rec["isoflux_n"] == 40 and rec["isoflux_resampling"] is None
    assert rec["isoflux_weight_total"] == pytest.approx(np.sum(w))
    # ... on the legacy route too
    assert case_geometry(_gsrc(), sc, standard=False).iso_pts is pts


def test_the_installed_total_omits_the_reference_points_weight():
    """TokaMaker's set_isoflux does not apply weights[0] (the reference point's): the record and
    the stamp carry the installed total, the nominal total minus w[0], next to the nominal one."""
    from bouquet.solver_geometry import isoflux_stamp
    geo = case_geometry(_gsrc(), _sc())
    rec = geo.record()
    w = np.asarray(geo.iso_w)
    assert rec["isoflux_weight_total"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    assert rec["isoflux_weight_installed_total"] == pytest.approx(
        rec["isoflux_weight_total"] - w[0], rel=1e-12)
    assert rec["isoflux_weights"]["installed_total"] == rec["isoflux_weight_installed_total"]
    st = isoflux_stamp(geo.iso_pts, geo.iso_w, "x")
    assert st["weight_installed_total"] == pytest.approx(st["weight_total"] - w[0], rel=1e-12)
    # no X-point: uniform weights, installed = 2e5 x 255/256
    _, w0, _ = standard_isoflux(_shape(200, corner=0.0))
    st0 = isoflux_stamp(np.zeros((N_ISOFLUX, 2)), w0, "x")
    assert st0["weight_installed_total"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL * 255 / 256,
                                                         rel=1e-12)


def test_a_user_set_without_weights_gets_the_standard_per_length_weights():
    """Points without weights are never left at TokaMaker's 1 per point on the unified engine:
    per-length weights over the points as given, summing to the standard total, stamped, with a
    UserWarning naming the config field."""
    from bouquet.engine import _geometry_isoflux
    from bouquet.solver_geometry import ISOFLUX_RULE_USER_PER_LENGTH, isoflux_stamp
    sc = _sc()
    pts = _shape(40, corner=0.0)
    sc.isoflux_pts = pts
    with pytest.warns(UserWarning, match="SolverConfig.isoflux_weights"):
        geo = case_geometry(_gsrc(), sc)
    assert geo.iso_pts is pts                                        # the points: as given
    assert np.sum(geo.iso_w) == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    d = _spacing(pts)                                                # closed: half each side
    np.testing.assert_allclose(geo.iso_w, ISOFLUX_WEIGHT_TOTAL * 0.5 * (d + np.roll(d, 1))
                               / np.sum(d), rtol=1e-12)
    rec = geo.record()
    assert rec["isoflux_rule"] == ISOFLUX_RULE_USER_PER_LENGTH
    assert rec["isoflux_weight_total"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    inst = _geometry_isoflux(geo)                                    # what the engine installs
    np.testing.assert_array_equal(inst["w"], geo.iso_w)
    st = isoflux_stamp(inst["pts"], inst["w"], inst["origin"])
    assert st["weight_total"] == pytest.approx(ISOFLUX_WEIGHT_TOTAL, rel=1e-12)
    # an open arc (the upper half): its two end points represent half of their one spacing
    half = pts[:21]
    sc.isoflux_pts = half
    with pytest.warns(UserWarning):
        w = case_geometry(_gsrc(), sc).iso_w
    dh = _spacing(half)[:-1]
    np.testing.assert_allclose(w, ISOFLUX_WEIGHT_TOTAL * 0.5 * (np.r_[0.0, dh] + np.r_[dh, 0.0])
                               / np.sum(dh), rtol=1e-12)


def test_a_user_set_with_weights_is_stamped_as_given_and_never_none():
    from bouquet.solver_geometry import isoflux_stamp
    sc = _sc()
    pts, w = _shape(40, corner=0.0), np.linspace(1.0, 2.0, 40)
    sc.isoflux_pts, sc.isoflux_weights = pts, w
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)                        # no per-length warning
        geo = case_geometry(_gsrc(), sc)
    assert geo.iso_w is w
    assert isoflux_stamp(geo.iso_pts, geo.iso_w, "x")["weight_total"] == pytest.approx(np.sum(w))
    # the legacy route keeps TokaMaker's 1 per point for a set without weights, stamped as such
    sc.isoflux_weights = None
    geo = case_geometry(_gsrc(), sc, standard=False)
    assert geo.iso_w is None
    assert geo.record()["isoflux_weight_total"] == pytest.approx(40.0)
    assert geo.record()["isoflux_weights"]["min"] == geo.record()["isoflux_weights"]["max"] == 1.0
    assert isoflux_stamp(pts, None, "x")["weight_total"] == pytest.approx(40.0)


def test_the_legacy_route_keeps_the_source_boundary_per_point():
    geo = case_geometry(_gsrc(), _sc(), standard=False)
    src = _gfile_boundary()
    np.testing.assert_array_equal(geo.iso_pts, src)
    assert np.all(geo.iso_w == ISOFLUX_WEIGHT)
    rec = geo.record()
    assert rec["isoflux_rule"] == ISOFLUX_RULE_LEGACY
    assert rec["isoflux_origin"] == "source boundary" and rec["isoflux_resampling"] is None
    assert rec["isoflux_n"] == len(src)
    assert rec["isoflux_weight_total"] == pytest.approx(len(src) * ISOFLUX_WEIGHT)


def test_which_routes_hold_the_standard_constraint():
    from bouquet.config import GenerationConfig
    assert standard_isoflux_for(None) is True
    for sm, want in (("engine", True), ("legacy", False), ("swb", False)):
        g = GenerationConfig()
        g.solve_method = sm
        assert standard_isoflux_for(g) is want, sm


def test_the_saddle_targets_are_installed_as_configured_and_mark_the_x_point():
    sc = _sc()
    src = _gfile_boundary()
    geo0 = case_geometry(_gsrc(), sc)
    x0 = geo0.record()["isoflux_resampling"]["x_points"][0]
    xp = np.array([[x0["R"] + 0.003, x0["Z"] - 0.002]])
    sc.saddle_targets, sc.saddle_weights = xp, np.array([7.0])
    geo = case_geometry(_gsrc(), sc)
    np.testing.assert_array_equal(geo.saddle_pts, xp)              # the saddles unchanged
    np.testing.assert_array_equal(geo.saddle_w, [7.0])
    rs = geo.record()["isoflux_resampling"]
    assert [x["origin"] for x in rs["x_points"]] == ["saddle target 0"]
    np.testing.assert_array_equal(geo.iso_pts, geo0.iso_pts)      # the same source vertex
    assert np.any(np.all(src == (x0["R"], x0["Z"]), axis=1))


def test_a_coarser_writer_of_the_same_boundary_gets_the_same_constraint_strength():
    """Every third point of the example g-file's boundary: the same count and total weight, and
    nearly the same length -- before #84 the coarser writer's boundary was held 3x more weakly."""
    g1 = case_geometry(_gsrc(), _sc())
    p_sub, w_sub, i_sub = standard_isoflux(_gfile_boundary()[::3])
    assert len(p_sub) == N_ISOFLUX
    assert np.sum(w_sub) == pytest.approx(np.sum(g1.iso_w), rel=1e-12)
    assert i_sub["boundary_length_m"] == pytest.approx(
        g1.record()["boundary_length_m"], rel=5e-3)


# ---------------------------------------------------------------------------
# SolverConfig.isoflux_weight_total: validated, and an ignored override is said
# (review of #94, C2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, 0.0, -2e5, float("nan"), float("inf"), True, False,
                                 "2e5", [2e5]])
def test_the_total_override_is_refused_unless_finite_and_positive(bad):
    from bouquet.config import SolverConfig
    with pytest.raises(ValueError, match="isoflux_weight_total"):
        SolverConfig(mesh_path="m.h5", isoflux_weight_total=bad)


@pytest.mark.parametrize("good", [None, 2e5, 300000, np.float64(1.5e5), np.int64(4)])
def test_the_total_override_accepts_none_or_a_positive_number(good):
    from bouquet.config import SolverConfig
    sc = SolverConfig(mesh_path="m.h5", isoflux_weight_total=good)
    assert sc.isoflux_weight_total == (None if good is None else float(good))


def test_an_override_with_user_weights_is_warned_and_recorded_as_ignored():
    sc = _sc()
    pts, w = _shape(40, corner=0.0), np.linspace(1.0, 2.0, 40)
    sc.isoflux_pts, sc.isoflux_weights = pts, w
    sc.isoflux_weight_total = 3e5
    with pytest.warns(UserWarning, match=r"isoflux_weight_total=300000 is IGNORED: "
                                         r"SolverConfig.isoflux_weights is given"):
        geo = case_geometry(_gsrc(), sc)
    assert geo.iso_w is w                                            # installed as given
    bal = geo.record()["isoflux_balance"]
    assert bal["mode"] == "as given"
    assert bal["isoflux_weight_total_ignored"]["value"] == 3e5


def test_an_override_on_the_legacy_route_is_warned_as_ignored():
    sc = _sc()
    sc.isoflux_weight_total = 3e5
    with pytest.warns(UserWarning, match=r"IGNORED: the legacy route installs the boundary at "
                                         r"ISOFLUX_WEIGHT=500 per point"):
        geo = case_geometry(_gsrc(), sc, standard=False)
    assert np.all(geo.iso_w == ISOFLUX_WEIGHT)


def test_an_override_on_the_standard_set_is_not_warned():
    import warnings as _w
    sc = _sc()
    sc.isoflux_weight_total = 3e5
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        geo = case_geometry(_gsrc(), sc)
    assert geo.record()["isoflux_balance"]["mode"] == "override"
    assert float(np.sum(geo.iso_w)) == pytest.approx(3e5, rel=0.05)


def test_open_boundary_refused():
    """An open trace (a separatrix traced strike point to strike point) is refused, not closed with a chord."""
    from bouquet.solver_geometry import _open_ccw
    src = _shape(200)
    _open_ccw(src)                                          # closed: accepted
    t = np.linspace(0.0, 1.0, 30)[:, None]
    legs = np.vstack([src[0] + (1 - t) * [-0.25, -0.4], src, src[0] + t * [0.3, -0.5]])
    with pytest.raises(ValueError, match="open curve"):
        resample_boundary(legs)
    with pytest.raises(ValueError, match="open curve"):
        _open_ccw(src[:120])                                # a closed contour with an arc missing
