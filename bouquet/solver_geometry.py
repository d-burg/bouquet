"""The per-baseline solver geometry: vacuum F0, LCFS boundary, isoflux and saddle constraints.

The mesh, regions, coil mode and VSC are fixed when the solver is set up; everything here belongs to one
baseline, so one solver can run many baselines (:meth:`bouquet.run.Bouquet.point_solver`).

The boundary (isoflux) constraint of the unified engine is standardised per unit length (#84): the
source boundary is resampled to :data:`N_ISOFLUX` points by arc length, its X-points kept exactly,
and each point weighted by the arc length it represents, the weights summing to
:data:`ISOFLUX_WEIGHT_TOTAL` (:func:`standard_isoflux`).  The reconstruction solves under that set
and every engine draw's homotopy holds the same set.  A ``SolverConfig.isoflux_pts`` set is used as
given; the legacy route keeps the source boundary at :data:`ISOFLUX_WEIGHT` per point.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

#: The LEGACY route's per-point isoflux weight: under ``solve_method="legacy"`` / ``"swb"`` the
#: reset installs the source boundary as given, every point at this weight, as before #84 (the legacy
#: g-file reconstruction then installs its own set at 200, ``baseline.py``; the legacy draws
#: re-trace).  The unified engine holds the standard set below instead.
ISOFLUX_WEIGHT = 500.0

#: The number of points the unified engine's boundary (isoflux) constraint holds, whatever the
#: source carried (#84): the source boundary is resampled to ``N_ISOFLUX`` points by arc length.
N_ISOFLUX = 256

#: The TOTAL isoflux weight of the unified engine's boundary constraint (#84), ONE constant for the
#: reconstruction and every draw (the draws hold the reconstruction's own set), unless
#: ``SolverConfig.isoflux_weight_total`` overrides it (stamped as mode "override").  Each point
#: carries ``total x (the arc length it represents) / (the boundary length)``, so the constraint
#: strength depends on the boundary itself, not on how many points the g-file writer, the IDS
#: writer or the mesh gave it.
#:
#: Why 2e5: a constant inside a MEASURED WINDOW (docs/engine.md, "The boundary constraint: a
#: constant total inside a measured window").  Only the ratio of this total to the draws' coil
#: regularisation matters to the solution.  The measured window is about 1.5e5-4e5: below it a
#: diverted g-file case's vertical-control drift grows as the boundary weakens and its in-spec
#: draws fall (7 of 10 at 2e5, 3 at 1e5, 1 at 5e4; before #84, at the route's ~2e4, its sigma=0
#: draw missed the archived-stage bootstrap gate); at 4e5 the boundary is already the stiffer
#: side (partition 0.3-0.9) and that case keeps only 5 of 10 in spec.  The balance-crossing band
#: is 1.8e5-3.8e5: the total at which a boundary miss equal to the boundary cut and a coil drift
#: equal to the coil cut cost the solver the same (partition = 1, the vertical-control pair
#: excluded), measured on six equilibria (three experimental, three synthetic), at 2 % and 10 %
#: probe amplitude, with and without the homotopy's coil bounds.  Every sigma=0 draw passes its
#: gates from 5e4 to 4e5.  2e5 sits inside both.  The partition at the installed total is
#: measured on every reconstruction and reported (:mod:`bouquet.boundary_balance`; it never
#: changes the total).  A calibrated per-case root find was tried and rejected: on cases where
#: the boundary and the vertical-control pair move as one rigid displacement the partition is
#: flat in the total over most of the window, so the root find lands anywhere on the plateau.
#:
#: This is the NOMINAL total, the sum of all the set's weights.  TokaMaker's ``set_isoflux`` measures
#: every other point's flux against the set's first point (the reference) and does not apply that
#: point's own weight (``weights[0]`` is discarded), so the total actually installed is the nominal
#: total minus ``w[0]`` (199 219 for a 256-point set without X-points).  The measurements above were
#: made with exactly this installation; the records stamp both totals
#: (``isoflux_weight_total`` / ``isoflux_weight_installed_total``, and the isoflux stamps'
#: ``weight_total`` / ``weight_installed_total``).
ISOFLUX_WEIGHT_TOTAL = 2.0e5

#: X-point detection on the source boundary (#84; X-points are kept as exact resampled points): a
#: source vertex whose turning angle -- between the chords to the boundary points one window
#: (``XPOINT_WINDOW_FRAC`` x the length) before and after it, so the test does not depend on the
#: sampling density -- exceeds ``XPOINT_TURN_DEG``; of candidates within two windows of each other
#: only the sharpest is kept.  An X-point's corner turns the boundary by ~70-90 deg over that window,
#: a smooth boundary by far less (measured on the synthetic fixtures: X-points 68-84 deg, elsewhere
#: <= 15 deg away from the corner).  A saddle target (``SolverConfig.saddle_targets``) marks an
#: X-point when source vertices lie within ``XPOINT_SADDLE_MATCH_FRAC`` x the length of it: the
#: one of them with the largest windowed turning angle, if that exceeds ``XPOINT_TURN_DEG`` (so a
#: target a centimetre or two off the X-point still marks the X-point, not a leg vertex), else the
#: nearest one (stamped per X-point as ``xpoint_match`` = "largest turn" / "nearest").  A false
#: detection costs nothing (one point is kept exactly); a missed one leaves the corner to the
#: spline.
#: A source boundary whose closing gap (last point to first) exceeds this many times its largest
#: other spacing is an open curve, and :func:`_open_ccw` refuses it rather than close it.
OPEN_BOUNDARY_GAP_FACTOR = 10.0
XPOINT_TURN_DEG = 45.0
XPOINT_WINDOW_FRAC = 0.01
XPOINT_SADDLE_MATCH_FRAC = 0.01

#: The resampling's interpolant is broken (two open splines joined, no smoothing across) at the
#: X-points and at any source vertex where the polygon itself turns by more than this between its
#: neighbouring edges: a corner the X-point test keeps only one vertex of (a corner the writer cut
#: in two) would otherwise make the cubic overshoot by millimetres.  Smooth boundaries turn by
#: <= ~5 deg per vertex at 257 points (<= ~13 deg at 100); the cut corners of the fixtures 28-39 deg.
#:
#: The per-vertex turn depends on the sampling density (a smooth D sampled at 16-30 points turns by
#: 29-46 deg at its tips), so the fixed angle alone would break a coarse smooth source at every
#: high-curvature vertex and leave it a polygon, centimetres inside the curve.  A vertex is
#: therefore a break only when its turn ALSO exceeds :data:`SPLINE_BREAK_MEDIAN_RATIO` x the median
#: turn of the vertices around it (itself and :data:`SPLINE_BREAK_MEDIAN_WINDOW` on either side).
#: A corner is a local outlier against its neighbours, a coarse smooth boundary's tip is not: its
#: neighbours turn almost as much (measured: smooth D shapes, kappa 1.8, delta +/-0.5, 12-100 points,
#: turn / local median <= 2.8; the fixtures' corners and X-points 14-160).  A neighbourhood median
#: rather than the whole boundary's, because the curvature of a shaped boundary varies round it.
SPLINE_BREAK_DEG = 20.0
SPLINE_BREAK_MEDIAN_RATIO = 3.0
SPLINE_BREAK_MEDIAN_WINDOW = 3

#: The resampling method's name and version, stamped in the solver geometry record.
ISOFLUX_RESAMPLING_METHOD = "arc_length_resample/1"

#: How the isoflux set was made, stamped as ``isoflux_rule`` in the solver geometry record.
ISOFLUX_RULE_STANDARD = ("per unit length (#84): the source boundary resampled by arc length, "
                         "each point weighted by the arc length it represents")
ISOFLUX_RULE_LEGACY = ("per point (legacy route): the source boundary as given, every point at "
                       "ISOFLUX_WEIGHT")
ISOFLUX_RULE_USER = "as given: SolverConfig.isoflux_pts / isoflux_weights (not resampled)"
ISOFLUX_RULE_USER_PER_LENGTH = (
    "as given: SolverConfig.isoflux_pts (not resampled), isoflux_weights=None so each point "
    "weighted by the arc length it represents along the points in the order given, summing to "
    "ISOFLUX_WEIGHT_TOTAL")

#: The F0 rule of a reused solver, stamped in every :meth:`CaseGeometry.record`: since #76 every
#: baseline (each ``run_slices`` / ``set_slice`` slice, each ``set_case`` case, every swb per-solve
#: reset) is solved at ITS OWN vacuum F0; before, a reused solver kept the first baseline's F0.
F0_RULE = "per-baseline: each baseline solves at its own F0 (since #76)"


def isoflux_fingerprint(pts, weights) -> Optional[str]:
    """A short sha256 of an isoflux pattern (its points AND weights, float64, C order): the
    identity the solver geometry record, the reconstruction and every engine draw's homotopy stamp,
    so that "the draw held the reconstruction's own isoflux" is a checkable equality.  ``None``
    without points."""
    if pts is None:
        return None
    import hashlib
    p = np.ascontiguousarray(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
    w = np.ascontiguousarray(np.broadcast_to(
        np.asarray(1.0 if weights is None else weights, dtype=np.float64), (len(p),)))
    h = hashlib.sha256(p.tobytes())
    h.update(w.tobytes())
    return h.hexdigest()[:16]


def isoflux_stamp(pts, weights, origin) -> dict:
    """The plain-data stamp of an isoflux pattern: ``n`` points, the weight (``weight_min`` /
    ``weight_max`` / ``weight_total``, and ``weight_installed_total``, the total without the
    reference point's weight, which TokaMaker does not apply; ``weights=None`` is stamped as
    TokaMaker installs it, 1 per point), its ``origin`` and :func:`isoflux_fingerprint` (``sha256``)."""
    n = 0 if pts is None else int(len(pts))
    # weights None: TokaMaker's set_isoflux installs 1 per point, so that is what is stamped
    w = (None if pts is None else np.ones(n) if weights is None else
         np.atleast_1d(np.asarray(weights, dtype=float)))
    wn = None if w is None or not w.size else np.broadcast_to(w, (n,))
    return {"n": n,
            "weight_min": None if w is None or not w.size else float(w.min()),
            "weight_max": None if w is None or not w.size else float(w.max()),
            "weight_total": None if wn is None else float(np.sum(wn)),
            # what TokaMaker applies: the reference point's (the first point's) weight is not
            "weight_installed_total": None if wn is None else float(np.sum(wn[1:])),
            "origin": None if origin is None else str(origin),
            "sha256": isoflux_fingerprint(pts, weights)}


# ── the standard boundary constraint (#84) ─────────────────────────────────────────────────────────
def _open_ccw(boundary_RZ):
    """The source boundary as an OPEN polygon (no repeated closing point, no repeated vertices),
    counter-clockwise in (R, Z)."""
    P = np.asarray(boundary_RZ, dtype=np.float64).reshape(-1, 2)
    if len(P) < 3:
        raise ValueError(f"the source boundary has {len(P)} points: at least 3 are needed")
    if not np.all(np.isfinite(P)):
        raise ValueError("the source boundary has non-finite points")
    d = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    scale = float(np.sum(d))
    if scale <= 0.0:
        raise ValueError("the source boundary is degenerate (zero length)")
    P = P[d > 1e-12 * scale]          # drops repeated vertices and the repeated closing point
    if len(P) < 3:
        raise ValueError("the source boundary has fewer than 3 distinct points")
    # an OPEN trace (an IMAS boundary.outline running strike point -> LCFS -> strike point) is not
    # a closed contour: closing it puts the chord between its ends -- across the private-flux
    # region -- into the isoflux set, a constraint no coil set meets (SPARC IDS: the first GS pass
    # cycled with period 1/urf to maxits)
    d = np.hypot(*np.diff(P, axis=0).T)
    gap = float(np.hypot(*(P[-1] - P[0])))
    if gap > OPEN_BOUNDARY_GAP_FACTOR * float(np.max(d)):
        raise ValueError(
            f"the source boundary is an open curve, not a closed LCFS: its ends ({P[0, 0]:.4g}, "
            f"{P[0, 1]:.4g}) and ({P[-1, 0]:.4g}, {P[-1, 1]:.4g}) m are {gap:.3g} m apart against "
            f"a largest spacing of {float(np.max(d)):.3g} m elsewhere (factor > "
            f"{OPEN_BOUNDARY_GAP_FACTOR:g}); a separatrix traced leg to leg must be clipped to its "
            "closed part at the X-point first (or give ImasSource.LCFS_geqdsk)")
    area = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    return P if area > 0 else P[::-1].copy()


def _polyline_s(P):
    """Cumulative chord length round the CLOSED polygon ``P`` (open form): ``len(P)+1`` values."""
    d = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def find_xpoints(boundary_RZ, saddle_targets=None):
    """The source boundary's X-points: ``(P, xpts)`` with ``P`` the boundary as an open
    counter-clockwise polygon and ``xpts`` a list of ``dict(index, R, Z, origin, turn_deg)`` (vertex
    indices into ``P``, sorted).  Rules and constants: :data:`XPOINT_TURN_DEG`,
    :data:`XPOINT_WINDOW_FRAC`, :data:`XPOINT_SADDLE_MATCH_FRAC`."""
    P = _open_ccw(boundary_RZ)
    s = _polyline_s(P)
    L = float(s[-1])
    Q = np.vstack([P, P[:1]])
    h = XPOINT_WINDOW_FRAC * L

    def at(t):
        t = np.mod(t, L)
        return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])
    a = P - at(s[:-1] - h)
    b = at(s[:-1] + h) - P
    cosang = np.sum(a * b, axis=1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-300)
    turn = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    def apart(i, j):
        g = abs(s[i] - s[j])
        return min(g, L - g) > 2.0 * h
    found = []
    if saddle_targets is not None:
        for k, xy in enumerate(np.asarray(saddle_targets, dtype=np.float64).reshape(-1, 2)):
            dist = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1])
            near = np.flatnonzero(dist <= XPOINT_SADDLE_MATCH_FRAC * L)
            if not near.size:
                continue
            # the sharpest vertex inside the match radius (a target off the X-point must not
            # mark a leg vertex in its place); none above XPOINT_TURN_DEG: the nearest vertex
            j = near[np.argsort(-turn[near], kind="stable")[0]]
            i, match = ((int(j), "largest turn") if turn[j] > XPOINT_TURN_DEG else
                        (int(np.argmin(dist)), "nearest"))
            if all(apart(i, f["index"]) for f in found):
                found.append(dict(index=i, origin=f"saddle target {k}", turn_deg=float(turn[i]),
                                  saddle_distance_m=float(dist[i]), xpoint_match=match))
    for i in np.argsort(-turn, kind="stable"):
        if turn[i] <= XPOINT_TURN_DEG:
            break
        if all(apart(int(i), f["index"]) for f in found):
            found.append(dict(index=int(i), origin="turning angle", turn_deg=float(turn[i])))
    for f in found:
        f["R"], f["Z"] = float(P[f["index"], 0]), float(P[f["index"], 1])
    return P, sorted(found, key=lambda f: f["index"])


def _spline(t, xy, periodic):
    """Interpolating splines ``u -> (R, Z)`` through ``xy`` at parameter ``t`` (cubic where there
    are enough points, else of the highest degree they allow)."""
    from scipy import interpolate
    m = len(t)
    k = min(3, m - 2) if periodic else min(3, m - 1)
    if k < 1:
        raise ValueError("too few boundary points for a spline")
    tr = interpolate.splrep(t, xy[:, 0], k=k, per=periodic, s=0)
    tz = interpolate.splrep(t, xy[:, 1], k=k, per=periodic, s=0)
    return lambda u: np.column_stack([interpolate.splev(u, tr), interpolate.splev(u, tz)])


def _vertex_turns(P):
    """The turning angle [deg] of the polygon ``P`` (open, closed round) at each vertex, between
    its two neighbouring edges."""
    a = P - np.roll(P, 1, axis=0)
    b = np.roll(P, -1, axis=0) - P
    c = np.sum(a * b, axis=1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _local_kinks(P):
    """Vertex indices of ``P`` (open, closed round) where the polygon itself turns by more than
    :data:`SPLINE_BREAK_DEG` between its two neighbouring edges AND by more than
    :data:`SPLINE_BREAK_MEDIAN_RATIO` x the median turn of the vertices within
    :data:`SPLINE_BREAK_MEDIAN_WINDOW` of it (itself included), so a coarse but smooth source is
    not cut into a polygon."""
    turn = _vertex_turns(P)
    k = SPLINE_BREAK_MEDIAN_WINDOW
    idx = (np.arange(len(P))[:, None] + np.arange(-k, k + 1)[None, :]) % len(P)
    local = np.median(turn[idx], axis=1)
    return np.flatnonzero((turn > SPLINE_BREAK_DEG) & (turn > SPLINE_BREAK_MEDIAN_RATIO * local))


class _Curve:
    """The interpolated closed boundary: one periodic spline, or a chain of open splines joined at
    break vertices; ``S`` the arc length from the chain's start, ``at(S)`` the point there."""

    def __init__(self, P, breaks, m_dense):
        s = _polyline_s(P)
        nP = len(P)
        self.pieces = []
        if not len(breaks):
            f = _spline(s, np.vstack([P, P[:1]]), periodic=True)
            self._add(f, s[-1], m_dense)
        else:
            br = sorted(int(b) for b in breaks)
            for j, ia in enumerate(br):
                ib = br[(j + 1) % len(br)]
                idx = np.arange(ia, ib + 1) if ib > ia else np.arange(ia, ib + nP + 1)
                V = P[idx % nP]
                t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(V, axis=0).T))])
                self._add(_spline(t, V, periodic=False), t[-1],
                          max(16, int(np.ceil(m_dense * t[-1] / s[-1]))), start=ia, x=V[0])
        self.S0 = np.concatenate([[0.0], np.cumsum([p["L"] for p in self.pieces])])
        self.L = float(self.S0[-1])

    def _add(self, f, t1, m, start=None, x=None):
        u = np.linspace(0.0, t1, m + 1)
        S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(f(u), axis=0).T))])
        self.pieces.append(dict(f=f, u=u, S=S, L=float(S[-1]), start=start, x=x))

    def start_of(self, vertex):
        """The arc length at which the piece starting at break ``vertex`` starts."""
        j = [p["start"] for p in self.pieces].index(int(vertex))
        return float(self.S0[j])

    def at(self, S):
        S = np.mod(np.atleast_1d(np.asarray(S, dtype=float)), self.L)
        out = np.empty((len(S), 2))
        j = np.clip(np.searchsorted(self.S0, S, side="right") - 1, 0, len(self.pieces) - 1)
        for k in np.unique(j):
            p, sel = self.pieces[k], j == k
            out[sel] = p["f"](np.interp(S[sel] - self.S0[k], p["S"], p["u"]))
        return out

    def outermost(self):
        """The arc length of the curve's largest-R point (refined on its piece)."""
        from scipy.optimize import minimize_scalar
        best = None
        for k, p in enumerate(self.pieces):
            R = p["f"](p["u"])[:, 0]
            i = int(np.argmax(R))
            if best is None or R[i] > best[0]:
                best = (R[i], k, i)
        _, k, i = best
        p = self.pieces[k]
        lo, hi = p["u"][max(i - 1, 0)], p["u"][min(i + 1, len(p["u"]) - 1)]
        if hi > lo:
            r = minimize_scalar(lambda t: -float(p["f"](np.atleast_1d(t))[0, 0]),
                                bounds=(lo, hi), method="bounded",
                                options=dict(xatol=1e-12 * max(p["u"][-1], 1e-300)))
            ui = float(r.x)
        else:
            ui = float(p["u"][i])
        return float(self.S0[k] + np.interp(ui, p["u"], p["S"]))


def resample_boundary(boundary_RZ, n=N_ISOFLUX, saddle_targets=None):
    """The source boundary resampled to ``n`` points by arc length: a closed contour (no repeated
    point), counter-clockwise, its X-points (:func:`find_xpoints`) kept EXACTLY (#84).

    An interpolating cubic spline is put through the source points -- periodic, or, where the
    source polygon has corners (the X-points, and any vertex where it turns by more than
    :data:`SPLINE_BREAK_DEG`), a chain of open splines joined at those vertices so no corner is
    smoothed or overshot (the reasoning of :func:`bouquet.io.geqdsk._resample_contour`) --, its arc
    length is measured on dense samples, and the points are spaced uniformly in arc length: each
    stretch between two X-points gets a share of ``n`` proportional to its length, its starting
    X-point first (the spacing starts at the lowest X-point).  The first point of the set -- the
    point TokaMaker measures every other isoflux point's flux against (``set_isoflux``'s reference
    point) -- is the outermost (largest-R) point: on the curve itself without X-points, of the
    spaced set with them, so it does not depend on the source's sampling.  It is deliberately NOT
    the X-point: the flux at a saddle is the worst-determined flux on the boundary, and with the
    X-point as the reference a reversed-Ip near-double-null IDS slice's reconstruction stopped at
    maxits, while the same set with the outermost point first converged and reproduced at sigma=0
    (the pre-#84 sets started at the source's first point, the outboard midplane on the sources
    we read).  Returns ``(pts, arc, info)``: ``arc[i]`` is the arc length point ``i`` represents
    (half the spacing to each neighbour; the sum is the boundary length) and ``info`` the record
    (method, n, the X-points kept, the source's point count, the lengths)."""
    n = int(n)
    if n < 8:
        raise ValueError(f"n={n}: the isoflux resampling needs at least 8 points")
    n_src = int(len(np.asarray(boundary_RZ).reshape(-1, 2)))
    P, xpts = find_xpoints(boundary_RZ, saddle_targets)
    s = _polyline_s(P)
    xi = [x["index"] for x in xpts]
    breaks = sorted(set(xi) | set(int(i) for i in _local_kinks(P)))
    C = _Curve(P, breaks, 64 * n)
    L = C.L
    if not xpts:
        pts = C.at(C.outermost() + L * np.arange(n) / n)
        arc = np.full(n, L / n)
        anchor = "outermost point (largest R)"
    else:
        first = min(range(len(xpts)), key=lambda j: (xpts[j]["Z"], xpts[j]["index"]))
        order = xpts[first:] + xpts[:first]
        Sx = np.array([C.start_of(x["index"]) for x in order])
        Lj = np.mod(np.roll(Sx, -1) - Sx, L)
        if len(order) == 1:
            Lj = np.array([L])
        # each stretch's share of n by its length (largest remainder; its X-point counts)
        want = n * Lj / L
        cnt = np.maximum(np.floor(want).astype(int), 1)
        while cnt.sum() < n:
            cnt[int(np.argmax(want - cnt))] += 1
        while cnt.sum() > n:
            cnt[int(np.argmax(np.where(cnt > 1, cnt - want, -np.inf)))] -= 1
        pts, arc = [], []
        for j, x in enumerate(order):
            d, d_prev = Lj[j] / cnt[j], Lj[j - 1] / cnt[j - 1]
            p = C.at(Sx[j] + d * np.arange(cnt[j]))
            p[0] = P[x["index"]]                   # the X-point: exactly the source's vertex
            pts.append(p)
            arc.append(np.concatenate([[0.5 * (d_prev + d)], np.full(cnt[j] - 1, d)]))
        pts, arc = np.vstack(pts), np.concatenate(arc)
        # the reference point: the outermost resampled point, NOT the X-point (see the docstring)
        k = int(np.argmax(pts[:, 0]))
        pts, arc = np.roll(pts, -k, axis=0), np.roll(arc, -k)
        anchor = "outermost point (largest R) of the set spaced from the lowest X-point"
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    arc = np.asarray(arc, dtype=np.float64)
    info = dict(
        method=ISOFLUX_RESAMPLING_METHOD,
        description=("interpolating cubic spline through the source points (periodic, or open "
                     "splines joined at the X-points and the polygon's corners), uniform "
                     "arc-length spacing between X-points, X-points kept exactly; "
                     "counter-clockwise; first point: " + anchor),
        n=n, source_n=n_src, source_n_distinct=int(len(P)),
        boundary_length_m=float(L), source_polygon_length_m=float(s[-1]),
        first_point=dict(kind=anchor, R=float(pts[0, 0]), Z=float(pts[0, 1])),
        x_points=[{k: v for k, v in x.items() if k != "index"} for x in xpts],
        x_point_rule=dict(turn_deg=XPOINT_TURN_DEG, window_frac=XPOINT_WINDOW_FRAC,
                          saddle_match_frac=XPOINT_SADDLE_MATCH_FRAC),
        spline_breaks=len(breaks), spline_break_deg=SPLINE_BREAK_DEG,
        spline_break_median_ratio=SPLINE_BREAK_MEDIAN_RATIO,
        spline_break_median_window=SPLINE_BREAK_MEDIAN_WINDOW,
        spacing_m=dict(min=float(np.min(arc)), max=float(np.max(arc))))
    return pts, arc, info


def per_length_weights(arc, w_total=ISOFLUX_WEIGHT_TOTAL):
    """Weights ``w_total x arc_i / sum(arc)``: per unit length, summing to ``w_total`` (#84)."""
    arc = np.asarray(arc, dtype=np.float64)
    if arc.ndim != 1 or not arc.size or not np.all(np.isfinite(arc)) or np.any(arc <= 0):
        raise ValueError("per_length_weights needs positive, finite arc lengths")
    return float(w_total) * arc / float(np.sum(arc))


def user_isoflux_weights(pts, w_total=ISOFLUX_WEIGHT_TOTAL):
    """Per-length weights for a user isoflux set given without weights (``SolverConfig.isoflux_pts``
    with ``isoflux_weights=None``): each point weighted by the arc length it represents along the
    points IN THE ORDER GIVEN (half the distance to each neighbour), summing to ``w_total``.  The
    set is taken as a closed contour when its closing gap (last point to first) is no longer than
    its longest other spacing, else as an open arc whose two end points represent half of their one
    spacing.  Returns ``(weights, closed)``."""
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(P) < 2 or not np.all(np.isfinite(P)):
        raise ValueError("SolverConfig.isoflux_pts needs at least 2 finite points to be weighted "
                         "per unit length: give isoflux_weights")
    d = np.hypot(*np.diff(P, axis=0).T)
    gap = float(np.hypot(*(P[0] - P[-1])))
    closed = len(P) > 2 and gap <= float(np.max(d))
    if closed:
        dd = np.concatenate([d, [gap]])
        arc = 0.5 * (dd + np.roll(dd, 1))
    else:
        arc = 0.5 * (np.concatenate([[0.0], d]) + np.concatenate([d, [0.0]]))
    if np.any(arc <= 0):
        raise ValueError("SolverConfig.isoflux_pts has repeated points, so it cannot be weighted "
                         "per unit length: give isoflux_weights")
    return per_length_weights(arc, w_total), bool(closed)


def standard_isoflux(boundary_RZ, saddle_targets=None, n=N_ISOFLUX, w_total=ISOFLUX_WEIGHT_TOTAL):
    """The unified engine's standard boundary constraint (#84): ``(pts, weights, info)`` --
    :func:`resample_boundary` to ``n`` points and :func:`per_length_weights` summing to
    ``w_total``; ``info`` gains ``weight_total``."""
    pts, arc, info = resample_boundary(boundary_RZ, n=n, saddle_targets=saddle_targets)
    w = per_length_weights(arc, w_total)
    return pts, w, dict(info, weight_total=float(w_total))


def standard_isoflux_for(generation) -> bool:
    """Whether a run with this ``GenerationConfig`` holds the standard (#84) boundary constraint:
    under the unified engine's solve method, yes; ``"legacy"`` and ``"swb"`` keep the source
    boundary at :data:`ISOFLUX_WEIGHT` per point (no generation config: the default, yes)."""
    if generation is None:
        return True
    from .config import solve_method_of
    return solve_method_of(generation) == "engine"


@dataclass(frozen=True)
class CaseGeometry:
    F0: float
    boundary_RZ: Optional[np.ndarray]
    iso_pts: Optional[np.ndarray]
    iso_w: Optional[np.ndarray]
    saddle_pts: Optional[np.ndarray] = None
    saddle_w: Optional[np.ndarray] = None
    # provenance (record()): where F0, the boundary and the isoflux targets came from
    F0_source: Optional[float] = None
    F0_origin: str = "source"
    boundary_origin: Optional[str] = None
    iso_origin: str = "source boundary"
    # how the isoflux set was made (ISOFLUX_RULE_*) and, for the standard set, the resampling
    iso_rule: Optional[str] = None
    iso_resampling: Optional[dict] = None
    # how the total was chosen: mode "constant" (ISOFLUX_WEIGHT_TOTAL), "override"
    # (SolverConfig.isoflux_weight_total) or "as given"; after the reconstruction also the
    # report-only partition (bouquet.boundary_balance), under "partition"
    iso_balance: Optional[dict] = None

    def record(self) -> dict:
        """What every reset installs for this baseline, as plain data (archived as the ``_baseline``
        attr ``solver_geometry_json``): ``solver_F0`` and where it came from, the isoflux targets
        (count, weights and their total, the rule, the resampling: method, N, the X-points kept,
        the source's point count and the boundary length) and the saddle constraints.  The legacy
        g-file reconstruction installs its own isoflux afterwards (recorded in ``Baseline.recon``);
        this is the reset state every solve starts from."""
        def _w(w, n):
            if not n:
                return None
            if w is None:                      # TokaMaker's set_isoflux: 1 per point
                w = np.ones(int(n))
            w = np.atleast_1d(np.asarray(w, dtype=float))
            wn = np.broadcast_to(w, (int(n),))
            return {"n": int(n), "min": float(w.min()), "max": float(w.max()),
                    "total": float(np.sum(wn)),
                    "installed_total": float(np.sum(wn[1:]))} if w.size else None
        n_iso = 0 if self.iso_pts is None else int(len(self.iso_pts))
        n_sad = 0 if self.saddle_pts is None else int(len(self.saddle_pts))
        ww = _w(self.iso_w, n_iso)
        rs = self.iso_resampling
        b = None if self.boundary_RZ is None else np.asarray(self.boundary_RZ, float).reshape(-1, 2)
        return {
            "solver_F0": float(self.F0),
            "F0_source": None if self.F0_source is None else float(self.F0_source),
            "F0_origin": str(self.F0_origin),
            "F0_rule": F0_RULE,
            "boundary_origin": self.boundary_origin,
            "isoflux_origin": str(self.iso_origin) if n_iso else None,
            "isoflux_rule": self.iso_rule if n_iso else None,
            "isoflux_n": n_iso,
            "isoflux_weights": ww,
            "isoflux_weight_total": None if ww is None else ww["total"],
            # the total TokaMaker applies: set_isoflux discards the reference point's weight w[0]
            "isoflux_weight_installed_total": None if ww is None else ww["installed_total"],
            # the source boundary's own point count and length; the resampling (None: as given)
            "isoflux_source_n": None if b is None else int(len(b)),
            "boundary_length_m": (rs["boundary_length_m"] if rs else
                                  None if b is None or len(b) < 3 else
                                  float(_polyline_s(_open_ccw(b))[-1])),
            "isoflux_resampling": None if rs is None else dict(rs),
            # the total's provenance (constant, override, as given) and the measured partition
            "isoflux_weight_mode": (None if not n_iso or self.iso_balance is None
                                    else self.iso_balance.get("mode")),
            "isoflux_balance": None if self.iso_balance is None else dict(self.iso_balance),
            # the pattern's identity (points and weights): what the unified engine's draws compare
            # their homotopy isoflux against (engine_draws.GenerateEngineDraws.homotopy_isoflux)
            "isoflux_sha256": isoflux_fingerprint(self.iso_pts, self.iso_w) if n_iso else None,
            "saddle_origin": "SolverConfig.saddle_targets" if n_sad else None,
            "saddle_n": n_sad,
            "saddle_points": None if not n_sad else np.asarray(self.saddle_pts, float).tolist(),
            "saddle_weights": None if self.saddle_w is None else
            np.atleast_1d(np.asarray(self.saddle_w, float)).tolist(),
        }

    def apply(self, mygs):
        """Install this geometry on ``mygs`` (after a reset to the clean equilibrium)."""
        mygs.set_profiles(foffset=float(self.F0))
        if self.iso_pts is not None:
            mygs.set_isoflux(self.iso_pts, weights=self.iso_w)
        if self.saddle_pts is not None:
            mygs.set_saddle_constraints(self.saddle_pts, weights=self.saddle_w)


def with_partition(geom: CaseGeometry, partition: dict) -> CaseGeometry:
    """*geom* with the report-only partition record *partition* (:mod:`bouquet.boundary_balance`)
    added to its ``iso_balance`` under ``"partition"``.  The points and weights are unchanged: the
    partition is a measurement at the installed total, never a reason to change it."""
    import dataclasses
    bal = dict(geom.iso_balance or {})
    bal["partition"] = dict(partition)
    return dataclasses.replace(geom, iso_balance=bal)


def case_geometry(source, solver, standard=True) -> CaseGeometry:
    """The :class:`CaseGeometry` of ``source`` (g-file: R_center*B_center and its boundary; IDS:
    |r0*b0(t)| and the boundary outline or ``LCFS_geqdsk``), with ``solver.F0`` / ``isoflux_pts`` /
    ``saddle_targets`` overriding.

    The isoflux set: ``solver.isoflux_pts`` (with ``isoflux_weights``) as given, never resampled
    (without ``isoflux_weights``, under ``standard``, its points get per-length weights summing
    to :data:`ISOFLUX_WEIGHT_TOTAL`, :func:`user_isoflux_weights`, with a UserWarning; the legacy
    route installs them at TokaMaker's 1 per point, as before, and the record says so);
    else, with ``standard`` (the unified engine, :func:`standard_isoflux_for`), the
    standard set of #84 (:func:`standard_isoflux`: :data:`N_ISOFLUX` points by arc length, X-points
    exact -- the saddle targets mark them too --, weights per unit length summing to
    ``solver.isoflux_weight_total`` when set (an override), else to the constant
    :data:`ISOFLUX_WEIGHT_TOTAL`; else (the legacy route) the source boundary as given, every
    point at :data:`ISOFLUX_WEIGHT`.  The saddle constraints are installed as configured."""
    from .config import ImasSource, ReconstructionSource
    if isinstance(source, ReconstructionSource):
        from .io.geqdsk import read_geqdsk
        g = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
        F0_src = abs(g.R_center * g.B_center)
        boundary_RZ = np.column_stack([g.boundary_R, g.boundary_Z])
        F0_from, b_from = "source: g-file |R_center*B_center|", "source: g-file boundary"
    elif isinstance(source, ImasSource):
        from .io.imas import read_imas_geometry
        F0_src, boundary_RZ = read_imas_geometry(source)
        F0_from = "source: IDS |r0*b0(t)|"
        b_from = ("source: LCFS_geqdsk boundary" if getattr(source, "LCFS_geqdsk", None)
                  else "source: IDS boundary.outline")
    else:
        raise TypeError(f"unknown baseline source type {type(source).__name__}")
    if F0_src is None and solver.F0 is None:
        raise ValueError(f"{type(source).__name__} gives no vacuum F0: set SolverConfig.F0")
    F0 = F0_src if solver.F0 is None else float(solver.F0)
    if solver.F0 is not None:
        F0_from = "SolverConfig.F0"
    if F0_src is not None and abs(F0 - F0_src) > 1e-3 * abs(F0_src):
        warnings.warn(f"SolverConfig.F0={F0:.4f} overrides the source's F0={F0_src:.4f}; the engine's IDS "
                      "contract still uses the source's vacuum field", UserWarning, stacklevel=2)
    sad = sad_w = None
    if solver.saddle_targets is not None:
        sad = np.asarray(solver.saddle_targets, dtype=np.float64).reshape(-1, 2)
        sad_w = None if solver.saddle_weights is None else np.asarray(solver.saddle_weights, dtype=np.float64)
    iso_pts, iso_w = solver.isoflux_pts, solver.isoflux_weights
    iso_from, iso_rule, iso_rs = "SolverConfig.isoflux_pts", ISOFLUX_RULE_USER, None
    w_over = getattr(solver, "isoflux_weight_total", None)
    w_total = ISOFLUX_WEIGHT_TOTAL if w_over is None else float(w_over)
    iso_bal = None
    if iso_pts is not None and iso_w is None and standard:
        # never TokaMaker's default of 1 per point (2e5/256 ~ 780x weaker than the standard set);
        # the legacy route is unchanged (1 per point, stamped as such by record())
        iso_w, closed = user_isoflux_weights(iso_pts, w_total)
        iso_rule = ISOFLUX_RULE_USER_PER_LENGTH
        iso_from = (f"SolverConfig.isoflux_pts, isoflux_weights=None: weight per unit length along "
                    f"the {'closed contour' if closed else 'open arc'} of the points as given "
                    f"(total {w_total:g})")
        iso_bal = dict(mode="override" if w_over is not None else "as given", total=w_total)
        warnings.warn("SolverConfig.isoflux_pts was given without SolverConfig.isoflux_weights: the "
                      "points received the standard per-length weights (each point weighted by the "
                      "arc length it represents, summing to ISOFLUX_WEIGHT_TOTAL or "
                      "SolverConfig.isoflux_weight_total), not TokaMaker's default of 1 per point; "
                      "set isoflux_weights to choose them",
                      UserWarning, stacklevel=2)
    elif iso_pts is not None:
        iso_bal = dict(mode="as given", total=float(np.sum(np.broadcast_to(
            np.asarray(1.0 if iso_w is None else iso_w, dtype=float), (len(iso_pts),)))))
    # an override that this route does not use: say so (warned once per
    # message by the default filter) and, where there is a balance record,
    # record it (review of #94, C2)
    _w_ignored = None
    if w_over is not None:
        if solver.isoflux_pts is not None and solver.isoflux_weights is not None:
            _w_ignored = ("SolverConfig.isoflux_weights is given, and those weights are "
                          "installed as given")
        elif not standard:
            _w_ignored = ("the legacy route installs the boundary at "
                          + ("1 per point" if solver.isoflux_pts is not None
                             else f"ISOFLUX_WEIGHT={ISOFLUX_WEIGHT:g} per point"))
    if _w_ignored is not None:
        warnings.warn(f"SolverConfig.isoflux_weight_total={w_over:g} is IGNORED: {_w_ignored}",
                      UserWarning, stacklevel=2)
        if iso_bal is not None:
            iso_bal["isoflux_weight_total_ignored"] = dict(value=float(w_over), why=_w_ignored)
    if iso_pts is None:
        if standard:
            iso_pts, iso_w, iso_rs = standard_isoflux(boundary_RZ, saddle_targets=sad,
                                                      w_total=w_total)
            if w_over is None:
                iso_from = (f"source boundary, resampled to {N_ISOFLUX} points by arc length, weight "
                            f"per unit length (total {w_total:g})")
                iso_bal = dict(mode="constant", total=w_total,
                               source="solver_geometry.ISOFLUX_WEIGHT_TOTAL")
            else:
                iso_from = (f"source boundary, resampled to {N_ISOFLUX} points by arc length, weight "
                            f"per unit length (total {w_total:g}: SolverConfig.isoflux_weight_total)")
                iso_bal = dict(mode="override", total=w_total,
                               source="SolverConfig.isoflux_weight_total")
            iso_rule = ISOFLUX_RULE_STANDARD
        else:
            iso_pts, iso_w = boundary_RZ, np.full(len(boundary_RZ), ISOFLUX_WEIGHT)
            iso_from, iso_rule = "source boundary", ISOFLUX_RULE_LEGACY
    return CaseGeometry(F0=F0, boundary_RZ=boundary_RZ, iso_pts=iso_pts, iso_w=iso_w,
                        saddle_pts=sad, saddle_w=sad_w, F0_source=F0_src, F0_origin=F0_from,
                        boundary_origin=b_from, iso_origin=iso_from, iso_rule=iso_rule,
                        iso_resampling=iso_rs, iso_balance=iso_bal)
