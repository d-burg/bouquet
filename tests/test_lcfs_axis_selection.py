"""The boundary metric measures against the PLASMA boundary: of the curves
of the psi = psi_LCFS level set, the innermost one that goes around the
magnetic axis -- never another closed loop of the same level.

The "longest closed segment" rule (issue #33) cannot tell the boundary from
a closed loop elsewhere on the mesh.  On a diverted state a linear
contouring at the X-point flux value can return the boundary JOINED to its
two divertor legs as one open segment; the only closed segment left is then
a far loop (around a coil), and the boundary "rms" came out in metres.

Solver-free.  A synthetic flux map with the three ingredients:

* a plasma well with an X-point below it (``x^2 + z^2 + 2 z^3 / (3 z_x)``:
  O-point at the origin, saddle at ``(0, -z_x)``, flux ``z_x^2 / 3`` there),
* divertor legs running out of the mesh, and
* a far dip (a "coil") whose own closed loop crosses the same level,

contoured by matplotlib on a triangular mesh, exactly as the measurement
does, at a level just INSIDE the saddle value (the boundary closes; the far
loop is the longer closed curve) and just OUTSIDE it (the failing geometry:
leg -> boundary -> leg in one open segment, the far loop the only closed
one).
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bouquet.utils import (OpenLCFSContourWarning, magnetic_axis_of,  # noqa
                           select_closed_lcfs)

R0, ZX = 1.7, 1.0                    # axis major radius, X-point depth [m]
PSI_X = ZX ** 2 / 3.0                # flux at the saddle
COIL = (R0 + 2.6, 0.2)               # the far dip's centre
AXIS = (R0, 0.0)


def _psi(R, Z):
    x = R - R0
    plasma = x ** 2 + Z ** 2 + 2.0 * Z ** 3 / (3.0 * ZX)
    coil = -60.0 * np.exp(-((R - COIL[0]) ** 2 + (Z - COIL[1]) ** 2)
                          / 0.55 ** 2)
    return plasma + coil


def _mesh(nr=261, nz=221):
    r = np.linspace(R0 - 1.2, R0 + 3.9, nr)
    z = np.linspace(-1.9, 1.3, nz)
    RR, ZZ = np.meshgrid(r, z)
    tri = mtri.Triangulation(RR.ravel(), ZZ.ravel())
    return np.column_stack([RR.ravel(), ZZ.ravel()]), tri.triangles


def _segments(level):
    rz, lc = _mesh()
    fig, ax = plt.subplots(1, 1)
    try:
        cs = ax.tricontour(rz[:, 0], rz[:, 1], lc, _psi(rz[:, 0], rz[:, 1]),
                           levels=[level])
        return [np.asarray(v) for seg in cs.allsegs for v in seg
                if len(v) > 4]
    finally:
        plt.close(fig)


def _true_boundary(n=400):
    """The separatrix loop of the plasma well alone (analytic)."""
    z = np.linspace(-ZX, ZX / 2.0, n)        # z_top solves the cubic: z_x/2
    x2 = np.clip(PSI_X - z ** 2 - 2.0 * z ** 3 / (3.0 * ZX), 0.0, None)
    x = np.sqrt(x2)
    return np.vstack([np.column_stack([R0 + x, z]),
                      np.column_stack([R0 - x[::-1], z[::-1]])])


def _is_closed(s):
    return np.hypot(*(s[0] - s[-1])) <= 1e-3 * np.hypot(*np.ptp(s, axis=0))


def _near_plasma(curve):
    """Every vertex within the plasma well's bounding box (+ 5 cm)."""
    return (np.all(np.abs(curve[:, 0] - R0) < ZX / np.sqrt(3.0) + 0.05)
            and np.all(curve[:, 1] > -ZX - 0.05)
            and np.all(curve[:, 1] < ZX / 2.0 + 0.05))


INSIDE = PSI_X * (1.0 - 2e-3)
OUTSIDE = PSI_X * (1.0 + 2e-3)


class TestTheFixtureHasTheDefect:
    def test_inside_the_far_loop_is_the_longest_closed_curve(self):
        segs = _segments(INSIDE)
        closed = [s for s in segs if _is_closed(s)]
        assert len(closed) >= 2, "a boundary and a far loop are needed"
        legacy = select_closed_lcfs(segs)
        assert not _near_plasma(legacy), (
            "the fixture does not reproduce the defect: the longest closed "
            "curve is the boundary")
        assert np.hypot(*(legacy.mean(axis=0) - COIL)) < 0.5

    def test_outside_the_boundary_is_joined_to_its_legs(self):
        segs = _segments(OUTSIDE)
        closed = [s for s in segs if _is_closed(s)]
        assert len(closed) == 1, "only the far loop closes"
        assert not _near_plasma(closed[0])
        # ... and it is what the longest-closed rule returns, silently
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenLCFSContourWarning)
            legacy = select_closed_lcfs(segs)
        assert legacy is closed[0]


class TestAxisRule:
    @pytest.mark.parametrize("level", [INSIDE, OUTSIDE],
                             ids=["closed-boundary", "boundary-with-legs"])
    def test_the_boundary_is_returned(self, level):
        got = select_closed_lcfs(_segments(level), axis=AXIS)
        assert got is not None
        assert _near_plasma(got)
        np.testing.assert_array_equal(got[0], got[-1])      # closed
        # it is the separatrix loop, to the mesh
        from scipy.spatial import cKDTree
        d, _ = cKDTree(got).query(_true_boundary())
        # PROVENANCE of 0.012 m (set with the test in 884f9c6 without a
        # stated origin; recorded 2026-10-04, the number unchanged): "to the
        # mesh" -- below one cell of _mesh()'s grid (dR = 19.6 mm,
        # dZ = 14.5 mm), so the true separatrix is distinguished from any
        # other level-set curve; measured 4.0-4.1 mm at both levels
        assert np.sqrt(np.mean(d ** 2)) < 0.012
        # and it goes all the way around: both sides, top and X-point
        assert got[:, 0].min() < R0 - 0.5 and got[:, 0].max() > R0 + 0.5
        assert got[:, 1].max() > 0.45 and got[:, 1].min() < -0.9

    def test_the_two_levels_give_the_same_boundary(self):
        from scipy.spatial import cKDTree
        a = select_closed_lcfs(_segments(INSIDE), axis=AXIS)
        b = select_closed_lcfs(_segments(OUTSIDE), axis=AXIS)
        d, _ = cKDTree(a).query(b)
        # away from the X-point the two curves are a level step apart; at
        # the X-point the joined curve passes the saddle on the other side
        assert np.median(d) < 0.01 and np.max(d) < 0.08

    def test_the_legs_are_not_part_of_it(self):
        got = select_closed_lcfs(_segments(OUTSIDE), axis=AXIS)
        assert got[:, 1].min() > -ZX - 0.05

    def test_innermost_of_two_curves_around_the_axis(self):
        def ring(r, n=80):
            tt = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            p = np.column_stack([R0 + r * np.cos(tt), r * np.sin(tt)])
            return np.vstack([p, p[:1]])

        inner, outer = ring(0.5, 40), ring(1.5, 300)
        got = select_closed_lcfs([outer, inner], axis=AXIS)
        assert len(got) == len(inner)

    def test_no_curve_around_the_axis_is_none_and_loud(self):
        t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        far = np.column_stack([COIL[0] + 0.4 * np.cos(t),
                               COIL[1] + 0.4 * np.sin(t)])
        far = np.vstack([far, far[:1]])
        leg = np.column_stack([R0 + 3.0 + 0.0 * t, np.linspace(-1, 1, 64)])
        with pytest.warns(OpenLCFSContourWarning, match="magnetic axis"):
            assert select_closed_lcfs([far, leg], axis=AXIS) is None

    def test_a_bad_axis_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            select_closed_lcfs(_segments(INSIDE), axis=(np.nan, 0.0))

    def test_without_an_axis_the_historical_rule_is_unchanged(self):
        segs = _segments(INSIDE)
        closed = [s for s in segs if _is_closed(s)]
        assert select_closed_lcfs(segs) is max(closed, key=len)


class _FakeSolver:
    """Just what ``engine._lcfs_deviation_mm`` reads."""

    def __init__(self, level, o_point=AXIS):
        self.r, self.lc = _mesh()
        self.r = np.column_stack([self.r, np.zeros(len(self.r))])
        self.psi_bounds = (level, 0.0)
        self.o_point = np.asarray(o_point, dtype=float)

    def get_psi(self, normalized=True):
        assert normalized is False
        return _psi(self.r[:, 0], self.r[:, 1])


class TestTheEngineMeasure:
    @pytest.mark.parametrize("level", [INSIDE, OUTSIDE],
                             ids=["closed-boundary", "boundary-with-legs"])
    def test_millimetres_not_metres(self, level):
        from bouquet.engine import _lcfs_deviation_mm
        rms, mx = _lcfs_deviation_mm(_FakeSolver(level), _true_boundary())
        assert rms < 12.0 and mx < 60.0

    @pytest.mark.parametrize("level", [INSIDE, OUTSIDE],
                             ids=["closed-boundary", "boundary-with-legs"])
    def test_regression_the_longest_closed_rule_reads_metres(self, level,
                                                             monkeypatch):
        """Negative control: with the axis withheld the same measurement is
        the far loop's distance."""
        import bouquet.utils as U
        from bouquet.engine import _lcfs_deviation_mm
        monkeypatch.setattr(U, "magnetic_axis_of", lambda mygs: None)
        rms, _mx = _lcfs_deviation_mm(_FakeSolver(level), _true_boundary())
        assert rms > 1500.0

    def test_no_axis_reported_falls_back_to_the_historical_rule(self):
        assert magnetic_axis_of(_FakeSolver(INSIDE, (-1.0, 0.0))) is None
        assert magnetic_axis_of(object()) is None
        assert magnetic_axis_of(_FakeSolver(INSIDE)) == AXIS


class TestTheDrawFilterDoesNotUseIt:
    """The selector is a measurement.  The draw filter traces its own
    surface (``safe_trace_surf``); no filter, sampler or draw-path module
    may come to depend on the contour selected here."""

    @pytest.mark.parametrize("name", ["filtering.py", "sampling.py",
                                      "engine_draws.py", "jbs_loop.py",
                                      "parallel.py", "stats.py"])
    def test_module_never_names_the_selector(self, name):
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "bouquet" / name
        if not p.exists():
            pytest.skip(f"{name} is not on this line")
        src = p.read_text()
        for word in ("select_closed_lcfs", "_lcfs_deviation_mm",
                     "magnetic_axis_of", "boundary_max_dev_mm"):
            assert word not in src, f"{name} reads {word}"
        # ``boundary_rms_mm`` is ALSO the name of the per-draw HDF5 attr that
        # filter_boundaries writes from its OWN trace (main, #63) and the band
        # provenance reads back. That attr is not the reconstruction-quality
        # record's key, so it is exempt -- as an HDF5 attribute key only;
        # every other code use of the name (a dict/record key, an attribute,
        # a variable, a keyword) still fails.
        bad = _non_attr_uses(src, "boundary_rms_mm")
        assert not bad, f"{name} reads boundary_rms_mm outside HDF5 attrs: lines {bad}"

    def test_the_attr_exemption_still_catches_record_reads(self):
        """The exemption above is narrow: it admits HDF5 attribute keys and
        nothing else."""
        w = "boundary_rms_mm"
        for code in ('x = q.get("boundary_rms_mm")',
                     'x = q["boundary_rms_mm"]',
                     'x = rec.boundary_rms_mm',
                     'f(boundary_rms_mm=1.0)',
                     'boundary_rms_mm = 1.0',
                     'ok = "boundary_rms_mm" in q'):
            assert _non_attr_uses(code, w), code
        for code in ('hf[gp].attrs["boundary_rms_mm"] = 1.0',
                     'a = hf[gp].attrs\nx = a["boundary_rms_mm"]',
                     'a = hf[gp].attrs\nok = "boundary_rms_mm" in a',
                     'a = hf[gp].attrs\nfor k in ("x", "boundary_rms_mm"):\n'
                     '    if k in a:\n        y = a[k]',
                     '"""docstring naming boundary_rms_mm"""',
                     'x = {"draw_boundary_rms_mm": 1}'):
            assert not _non_attr_uses(code, w), code


def _non_attr_uses(src, word):
    """Line numbers where *word* is used as code other than as an HDF5
    attribute key (``<x>.attrs[word]``, ``word in <x>.attrs``, or a tuple of
    keys a ``for`` loop tests against / indexes ``<x>.attrs`` with), where a
    name bound to ``<expr>.attrs`` counts as ``<x>.attrs``. Docstrings,
    comments and longer names that merely contain *word* are not uses."""
    import ast
    tree = ast.parse(src)
    attrs_names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Attribute)
                   and n.value.attr == "attrs"
                   for t in n.targets if isinstance(t, ast.Name)}

    def is_attrs(n):
        return ((isinstance(n, ast.Attribute) and n.attr == "attrs")
                or (isinstance(n, ast.Name) and n.id in attrs_names))

    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[c] = n

    def loop_tests_attrs(for_node):
        if not isinstance(for_node.target, ast.Name):
            return False
        k = for_node.target.id
        for n in ast.walk(for_node):
            if (isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
                    and n.left.id == k and len(n.ops) == 1
                    and isinstance(n.ops[0], ast.In)
                    and is_attrs(n.comparators[0])):
                return True
        return False

    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and n.value == word:
            p = parent.get(n)
            ok = ((isinstance(p, ast.Subscript) and p.slice is n
                   and is_attrs(p.value))
                  or (isinstance(p, ast.Compare) and p.left is n
                      and len(p.ops) == 1 and isinstance(p.ops[0], ast.In)
                      and is_attrs(p.comparators[0]))
                  or (isinstance(p, ast.Tuple)
                      and isinstance(parent.get(p), ast.For)
                      and parent[p].iter is p and loop_tests_attrs(parent[p])))
            if not ok:
                bad.append(n.lineno)
        elif ((isinstance(n, ast.Attribute) and n.attr == word)
              or (isinstance(n, ast.Name) and n.id == word)
              or (isinstance(n, ast.keyword) and n.arg == word)
              or (isinstance(n, ast.arg) and n.arg == word)):
            bad.append(getattr(n, "lineno", -1))
    return sorted(bad)
