"""Synthetic diagnostics evaluated on a solved TokaMaker equilibrium.

One function per measurement group of :data:`interfaces.GROUP_IDS_PATHS`.
Each takes the equilibrium and the group's ``geometry`` dictionary (see
:class:`interfaces.ConstraintGroup`) and returns a :class:`Synthetic` pair
``(value, reason)``: ``value`` is the modeled signal per channel, NaN where it
cannot be evaluated, and ``reason`` is the matching per-channel explanation
(an empty string where the value is good).  No channel is ever dropped
without a recorded reason.

Nothing here changes the equilibrium: only read-only TokaMaker calls are made
(``get_field_eval``, ``get_globals``, ``get_coil_currents``, ``psi_bounds``,
``o_point``, ``get_xpoints``, ``abspsi_to_normalized``).

The equilibrium may be a ``TokaMaker`` object (its active equilibrium is used)
or a ``TokaMaker_equilibrium`` returned by ``TokaMaker.solve()`` /
``TokaMaker.copy_eq()``.

Units and conventions (TokaMaker's own, i.e. what the OFT reconstruction
module's constraints use; references are to OFT upstream ``f1492c4``)
------------------------------------------------------------------------------
=====================  ===========  ==========================================
group                  unit         model
=====================  ===========  ==========================================
``flux_loop``          Wb           ``-2*pi*psi_TM(R, Z)``: IMAS
                                    ``magnetics.flux_loop[:].flux`` (COCOS
                                    11, full loop flux, 2*pi included).
                                    TokaMaker ``psi_TM`` is per radian with
                                    the opposite sign (see :func:`flux_loop`);
                                    OFT ``fit_flux_eval`` returns
                                    ``+2*pi*psi_TM``.  Relative to a reference
                                    loop when ``flux_reference`` is set.
``bpol_probe``         T            ``B_R n_R + B_Z n_Z`` with ``(B_R, B_phi,
                                    B_Z) = (-dpsi/dZ, F, dpsi/dR)/R`` in the
                                    right-handed ``(R, phi, Z)`` frame, as
                                    OFT ``fit_field_eval`` with a zero
                                    toroidal normal component.  IMAS
                                    ``magnetics.b_field_pol_probe[:].field``,
                                    direction from ``poloidal_angle``.
                                    Averaged over the probe's signed
                                    ``length`` (EFIT convention) when the
                                    geometry carries one (default).
``ip``                 A            signed toroidal plasma current from
                                    ``get_globals`` (``gs_comp_globals``,
                                    divided by mu0 in
                                    ``tokamaker_get_globals``).  OFT's
                                    ``Ip_con`` accepts only positive values.
``diamagnetic_flux``   Wb           ``int_plasma (B_phi - F0/R) dA``, OFT
                                    ``gs_dflux`` via ``get_globals``.
``pf_current``         A (per turn) ``get_coil_currents()``:
                                    ``(coil_currs + coil_vcont*vcontrol)/mu0``
                                    (``tokamaker_get_coil_currents``), i.e.
                                    INCLUDING the virtual vertical-stability
                                    coil.  OFT ``fit_coil_eval`` omits the VSC
                                    part.  IMAS ``pf_active.coil[:].current``.
``mse``                rad (or 1)   9-coefficient IMAS form, see :func:`mse`.
``thomson_ne``         m^-3         ``n_e(psi_N(R, Z))``, see :func:`thomson`.
``thomson_te``         eV           ``T_e(psi_N(R, Z))``, see :func:`thomson`.
=====================  ===========  ==========================================

Sign conventions: measurements are expected in the IMAS convention (COCOS 11,
right-handed ``(R, phi, Z)`` with phi counter-clockwise seen from above).  Field
components, currents and the diamagnetic flux are physical quantities and
TokaMaker's ``(R, phi, Z)`` frame is that same right-handed frame, so they are
used as returned.  Poloidal flux is the one quantity whose sign differs:
``psi_IMAS = -2*pi*psi_TM`` (:func:`flux_loop`).  MSE alone accepts a stated
``(sign_pol, sign_tor)`` orientation map.

Off-mesh points: OFT's reconstruction constraints silently move an off-mesh
sensor to the nearest boundary node (``fit_flux_eval`` / ``fit_field_eval``,
``grad_shaf_fit.F90``).  Here such a channel is NOT moved: its value is NaN
with reason :data:`REASON_OFF_MESH`.  The test is an explicit point-in-
triangle search on the mesh (:func:`on_mesh`), so it does not depend on how a
given OFT build reports a failed interpolation.
"""

from typing import NamedTuple

import numpy as np

#: Reason recorded for a channel whose location is not inside any mesh cell.
REASON_OFF_MESH = "off the TokaMaker mesh: no field can be evaluated there"
#: Reason recorded for a Thomson channel outside the confined plasma.
REASON_OUTSIDE_LCFS = ("outside the last closed flux surface: a profile given "
                       "as a function of normalized psi is not defined there")
#: Reason recorded for a non-finite model value at an on-mesh point.
REASON_NONFINITE = "the model evaluated to a non-finite value"
#: Reason recorded for a channel with non-finite or non-physical geometry.
REASON_BAD_GEOMETRY = "non-finite or non-physical (R <= 0) channel geometry"

TWO_PI = 2.0 * np.pi


class Synthetic(NamedTuple):
    """Modeled signal per channel and the reason for each NaN entry."""

    value: np.ndarray
    reason: np.ndarray


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tokamaker(eq):
    """The ``TokaMaker`` object that owns *eq* (mesh, limiter, coil sets)."""
    return eq if hasattr(eq, "lim_contour") else eq._tMaker


def _points(geometry):
    """``(n, 2)`` array of ``(R, Z)`` from ``geometry['r']``, ``['z']``."""
    r = np.atleast_1d(np.asarray(geometry["r"], dtype=float)).ravel()
    z = np.atleast_1d(np.asarray(geometry["z"], dtype=float)).ravel()
    if r.shape != z.shape:
        raise ValueError("geometry 'r' and 'z' must have the same length")
    return np.column_stack((r, z))


def _reasons(n):
    return np.full(n, "", dtype=object)


def on_mesh(eq, pts):
    """True for each ``(R, Z)`` of *pts* that lies inside a mesh triangle.

    Barycentric point-in-triangle test against the TokaMaker mesh
    (``TokaMaker.r``, ``TokaMaker.lc``; linear triangles, the geometric mesh
    of any FE order), with a relative tolerance of 1e-10 so points on shared
    edges count as inside.
    """
    gs = _tokamaker(eq)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    xy = np.asarray(gs.r)[:, :2]
    tri = np.asarray(gs.lc)
    # the mesh of a TokaMaker object never changes: results are cached per (mesh, point set)
    key = (xy.shape, float(np.sum(xy)), float(np.sum(xy[:, 0] * np.arange(xy.shape[0]))), tri.shape,
           pts.tobytes())
    hit = _ON_MESH_CACHE.get(key)
    if hit is not None:
        return hit.copy()
    out = _on_mesh_compute(xy, tri, pts)
    if len(_ON_MESH_CACHE) > 256:
        _ON_MESH_CACHE.clear()
    _ON_MESH_CACHE[key] = out.copy()
    return out


_ON_MESH_CACHE = {}


def _on_mesh_compute(xy, tri, pts):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    out = np.zeros(pts.shape[0], dtype=bool)
    tol = 1e-10
    for k, p in enumerate(pts):
        if not np.all(np.isfinite(p)):
            continue
        cand = np.nonzero(np.all((p >= lo - 1e-12) & (p <= hi + 1e-12), axis=1))[0]
        if cand.size == 0:
            continue
        d = p - a[cand]
        s = (d[:, 0] * v1[cand, 1] - v1[cand, 0] * d[:, 1]) / den[cand]
        t = (v0[cand, 0] * d[:, 1] - d[:, 0] * v0[cand, 1]) / den[cand]
        out[k] = bool(np.any((s >= -tol) & (t >= -tol) & (s + t <= 1.0 + tol)))
    return out


def _eval_field(eq, field, pts):
    """Evaluate a TokaMaker field (``'B'`` or ``'PSI'``) at *pts*.

    Returns ``(values (n, dim), reason)``.  Off-mesh points and bad geometry
    get NaN and a reason; the interpolator is created fresh on every call
    (an interpolator bound to a previous solve must not be reused) and freed
    before returning.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    dim = 3 if field.upper() == "B" else 1
    vals = np.full((n, dim), np.nan)
    reason = _reasons(n)
    good = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] > 0.0)
    reason[~good] = REASON_BAD_GEOMETRY
    inside = np.zeros(n, dtype=bool)
    inside[good] = on_mesh(eq, pts[good])
    reason[good & ~inside] = REASON_OFF_MESH
    if np.any(inside):
        fe = eq.get_field_eval(field)
        try:
            v = np.asarray(fe.eval(np.ascontiguousarray(pts[inside])), dtype=float)
        finally:
            del fe
        vals[inside] = v.reshape(-1, dim)
    bad = inside & ~np.all(np.isfinite(vals), axis=1)
    vals[bad] = np.nan
    reason[bad] = REASON_NONFINITE
    return vals, reason


def field_at(eq, pts):
    """``(B, reason)``: ``(B_R, B_phi, B_Z)`` [T] at *pts*.

    TokaMaker ``get_field_eval('B')`` (Fortran ``gs_b_interp``):
    ``B = psiscale*(-dpsi/dZ, F(psi), dpsi/dR)/R`` in the right-handed
    ``(R, phi, Z)`` frame, the same expression OFT's ``fit_field_eval`` uses
    for its Mirnov constraint.
    """
    return _eval_field(eq, "B", pts)


def psi_at(eq, pts):
    """``(psi, reason)``: poloidal flux per radian [Wb/rad] at *pts*.

    TokaMaker ``get_field_eval('PSI')`` (``psiscale*psi``), in the same
    absolute gauge as ``get_psi(normalized=False)`` and ``psi_bounds``.
    """
    v, reason = _eval_field(eq, "PSI", pts)
    return v[:, 0], reason


def psi_n_at(eq, pts):
    """``(psi_N, reason)``: normalized poloidal flux at *pts*.

    ``psi_N = (psi - psi_axis)/(psi_boundary - psi_axis)`` via TokaMaker's own
    ``abspsi_to_normalized`` (0 on axis, 1 on the last closed flux surface),
    defined everywhere on the mesh -- inside or outside the plasma.
    """
    psi, reason = psi_at(eq, pts)
    return np.asarray(eq.abspsi_to_normalized(psi), dtype=float), reason


def _inside_polygon(pts, poly):
    """Even-odd ray-casting point-in-polygon test (closed or open polygon)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    out = np.zeros(pts.shape[0], dtype=bool)
    for k, (px, py) in enumerate(pts):
        crosses = (y0 > py) != (y1 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        out[k] = bool(np.count_nonzero(crosses & (px < xint)) % 2)
    return out


def inside_plasma(eq, pts, psi_n=None):
    """True where *pts* lie inside the confined plasma (the LCFS).

    Mirrors OFT's own in-plasma test (``gs_test_bounds``,
    ``grad_shaf.F90``, plus its ``psi > plasma_bounds(1)`` condition):

    * ``psi_N <= 1`` (normalized flux inside the boundary value);
    * on the plasma side of every X-point: ``(p - x_i).(o - x_i) > 0`` with
      ``o`` the O-point (OFT ``x_vecs = o_point - x_points``);
    * inside the limiter contour (``TokaMaker.lim_contour``; OFT uses the
      plasma region's bounding box and region id, which the limiter polygon
      tightens).

    The X-point half-planes exclude the private-flux region and the region
    beyond the X-point, where ``psi_N < 1`` too.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if psi_n is None:
        psi_n, _ = psi_n_at(eq, pts)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(psi_n) & (psi_n <= 1.0)
    o = np.asarray(eq.o_point, dtype=float)
    xpts, _ = eq.get_xpoints()
    if xpts is not None:
        for x in np.asarray(xpts, dtype=float).reshape(-1, 2):
            ok &= (pts - x) @ (o - x) > 0.0
    lim = np.asarray(_tokamaker(eq).lim_contour, dtype=float)
    if lim.ndim == 2 and lim.shape[0] >= 3:
        ok &= _inside_polygon(pts, lim)
    return ok


# ---------------------------------------------------------------------------
# magnetics
# ---------------------------------------------------------------------------
#: ``psi_IMAS = PSI_TM_TO_IMAS * psi_TM``; see :func:`flux_loop`.
PSI_TM_TO_IMAS = -TWO_PI


def flux_loop(eq, geometry, reference=None):
    """Poloidal flux through toroidal flux loops [Wb], IMAS convention.

    ``model = psi_IMAS(R, Z) = -2*pi*psi_TM(R, Z)``, the IMAS
    ``magnetics.flux_loop[:].flux`` quantity (COCOS 11: full flux, 2*pi
    included; O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184, 293
    (2013), Table I).  TokaMaker's ``psi_TM`` is per radian and DEcreases
    outward for a current along +phi (its ``B_Z = (1/R) dpsi/dR``,
    ``gs_b_interp`` in ``grad_shaf.F90``), whereas COCOS 11 flux increases
    outward for that current; hence the factor ``-2*pi``.  Confirmed on
    measured data: loop fluxes stored in the IMAS convention match
    ``-2*pi*psi_TM`` of a TokaMaker re-solve.  OFT's own flux constraint
    (``fit_flux_eval``, ``grad_shaf_fit.F90``) evaluates ``+2*pi*psi_TM``.

    ``geometry``: ``r``, ``z`` [m], one point per loop (a multi-point loop must
    be reduced by the adapter).  ``reference``: optional ``(r, z)`` of a
    reference loop; the model is then ``psi_IMAS,i - psi_IMAS,ref`` (group
    ``meta['flux_reference']``, for data measured relative to one loop).  An
    off-mesh reference makes every channel NaN with that reason.
    """
    psi, reason = psi_at(eq, _points(geometry))
    val = PSI_TM_TO_IMAS * psi
    if reference is not None:
        pref, rref = psi_at(eq, np.asarray(reference, dtype=float).reshape(1, 2))
        if rref[0]:
            val = np.full(val.shape, np.nan)
            reason[reason == ""] = "reference loop: " + rref[0]
        else:
            val = val - PSI_TM_TO_IMAS * pref[0]
    return Synthetic(val, reason)


#: Probe models of :func:`bpol_probe`.
PROBE_MODELS = ("finite_length", "point")
#: Default number of averaging points along a finite-length probe (EFIT ``NSMP2``).
PROBE_N_AVG_DEFAULT = 25
#: Reason recorded for a finite-length probe whose segment leaves the mesh.
REASON_SEGMENT_OFF_MESH = ("finite-length probe: part of the averaging segment lies off the "
                           "TokaMaker mesh; the channel is not modelable (never truncated)")
#: Convention text stored with probe lengths (machine bundles, group meta, records).
PROBE_LENGTH_CONVENTION = ("EFIT: signed length SMP2 [m]; > 0 the segment lies along the probe "
                           "direction (AMP2), < 0 perpendicular to it (saddle loops); B.n averaged "
                           "at n_avg midpoints (k - (n_avg-1)/2) |L| / n_avg, k = 0..n_avg-1")


def probe_segment_points(geometry, n_avg=PROBE_N_AVG_DEFAULT):
    """Averaging points of finite-length poloidal-field probes.

    EFIT convention (OMFIT ``omfit_efund.py``, T. Strait 2016 note; ``mhdin.dat``
    ``&IN5 NSMP2 = 25``): a probe of signed length ``L`` (EFIT ``SMP2``) centered
    at ``(r, z)`` measures the field component along its direction averaged over a
    straight segment of length ``|L|``; for ``L > 0`` the segment lies along the
    probe direction ``n``, for ``L < 0`` perpendicular to it (saddle loops; their
    toroidal extent is irrelevant under axisymmetry).  The averaging rule is the
    midpoint rule with ``n_avg`` points at offsets ``(k - (n_avg - 1)/2) |L| / n_avg``,
    ``k = 0 .. n_avg - 1``; this placement was determined numerically against
    EFIT's own calculated probe signals (1e-4 sigma rms with midpoints vs 0.023 sigma with
    endpoint-inclusive placement; measured).

    Returns ``(points (n, n_avg, 2), ends (n, 2, 2))``: the averaging points and
    the two segment end points ``center -/+ |L|/2 t`` of every probe (``t`` the
    segment's unit tangent).  A probe with ``L = 0`` (or no ``length`` key)
    has all its points at the center.
    """
    pts = _points(geometry)
    n = pts.shape[0]
    nr = np.atleast_1d(np.asarray(geometry["direction_r"], dtype=float)).ravel()
    nz = np.atleast_1d(np.asarray(geometry["direction_z"], dtype=float)).ravel()
    norm = np.hypot(nr, nz)
    if np.any(~np.isfinite(norm) | (norm <= 0.0)):
        raise ValueError("bpol_probe: probe direction must be a finite, non-zero vector")
    nr, nz = nr / norm, nz / norm
    L = np.zeros(n) if "length" not in geometry else \
        np.atleast_1d(np.asarray(geometry["length"], dtype=float)).ravel()
    if L.shape != (n,):
        raise ValueError("bpol_probe: 'length' must have one entry per probe")
    if not np.all(np.isfinite(L)):
        raise ValueError("bpol_probe: 'length' must be finite (0 for a point probe)")
    n_avg = int(n_avg)
    if n_avg < 1:
        raise ValueError("n_avg must be >= 1")
    # segment tangent: along n for L > 0, perpendicular (n rotated by +90 deg) for L < 0
    tr = np.where(L >= 0.0, nr, -nz)
    tz = np.where(L >= 0.0, nz, nr)
    off = (np.arange(n_avg) - 0.5 * (n_avg - 1)) / n_avg          # midpoints, in units of |L|
    s = np.abs(L)[:, None] * off[None, :]
    points = np.stack((pts[:, 0:1] + s * tr[:, None], pts[:, 1:2] + s * tz[:, None]), axis=-1)
    half = 0.5 * np.abs(L)
    ends = np.stack((np.column_stack((pts[:, 0] - half * tr, pts[:, 1] - half * tz)),
                     np.column_stack((pts[:, 0] + half * tr, pts[:, 1] + half * tz))), axis=1)
    return points, ends


def probe_signals_from_field(field, geometry, model="finite_length", n_avg=PROBE_N_AVG_DEFAULT):
    """Probe signals ``B . n`` [T] from a poloidal-field function, by the same rule as :func:`bpol_probe`.

    ``field(pts)`` takes ``(N, 2)`` points ``(R, Z)`` and returns ``(B_R, B_Z)`` (two length-N arrays).  This
    is the probe model without a TokaMaker equilibrium -- e.g. for checking a machine bundle against another
    code's calculated probe signals from that code's flux map.  Finite-length probes use
    :func:`probe_segment_points` and the mean over the ``n_avg`` midpoints; ``model="point"`` or a geometry
    without ``length`` gives the field at the centers.  No mesh test is made.
    """
    if model not in PROBE_MODELS:
        raise ValueError(f"probe_signals_from_field: model must be one of {PROBE_MODELS}")
    pts = _points(geometry)
    nr = np.atleast_1d(np.asarray(geometry["direction_r"], dtype=float)).ravel()
    nz = np.atleast_1d(np.asarray(geometry["direction_z"], dtype=float)).ravel()
    norm = np.hypot(nr, nz)
    if nr.shape != (pts.shape[0],) or np.any(~np.isfinite(norm) | (norm <= 0.0)):
        raise ValueError("probe_signals_from_field: one finite, non-zero direction per probe")
    nr, nz = nr / norm, nz / norm
    if model == "point" or "length" not in geometry:
        br, bz = (np.asarray(x, dtype=float) for x in field(pts))
        return br * nr + bz * nz
    seg, _ = probe_segment_points(geometry, n_avg)
    n, m = seg.shape[0], seg.shape[1]
    br, bz = (np.asarray(x, dtype=float).reshape(n, m) for x in field(seg.reshape(-1, 2)))
    return np.mean(br * nr[:, None] + bz * nz[:, None], axis=1)


def probe_reproduction(model, reference, sigma, tolerance_rms):
    """Per-signal agreement of modeled probe signals with a reference (e.g. EFIT's calculated signals).

    ``z = (model - reference) / sigma`` over channels with finite values and ``sigma > 0``; returns ``{"z",
    "rms", "max_abs", "n", "tolerance_rms", "passed"}`` with ``passed = rms <= tolerance_rms``.  The tolerance
    is the caller's stated criterion; it is recorded, not chosen here.
    """
    model, reference, sigma = (np.asarray(x, dtype=float).ravel() for x in (model, reference, sigma))
    use = np.isfinite(model) & np.isfinite(reference) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(use):
        raise ValueError("probe_reproduction: no channel with finite values and sigma > 0")
    z = np.full(model.shape, np.nan)
    z[use] = (model[use] - reference[use]) / sigma[use]
    rms = float(np.sqrt(np.mean(z[use] ** 2)))
    return {"z": z, "rms": rms, "max_abs": float(np.max(np.abs(z[use]))), "n": int(use.sum()),
            "tolerance_rms": float(tolerance_rms), "passed": bool(rms <= tolerance_rms)}


def bpol_probe(eq, geometry, model="finite_length", n_avg=PROBE_N_AVG_DEFAULT):
    """Poloidal-field probe signal ``B . n`` [T].

    ``n = (direction_r, direction_z)`` is the probe's unit normal in the
    poloidal plane.  IMAS DD 3.41.0
    ``magnetics.b_field_pol_probe[:].poloidal_angle`` is measured CLOCKWISE
    from the +R direction in the (R, Z) plane, so ``n = (cos a, -sin a)``
    (``a = -pi/2`` points along +Z); the conversion is
    :func:`bouquet.kinrecon.measurements.probe_direction`.  The point form is
    identical to OFT ``fit_field_eval`` (``grad_shaf_fit.F90``) with a zero
    toroidal normal component.  The direction vector is normalized here and
    must be non-zero.

    ``model="finite_length"`` (default): when ``geometry`` carries a signed
    ``length`` [m] (EFIT ``SMP2``; IMAS ``b_field_pol_probe[:].length``), the
    signal is ``B . n`` averaged over the probe's segment by the EFIT midpoint
    rule (:func:`probe_segment_points`, ``n_avg`` points, default 25 = EFIT
    ``NSMP2``).  Probes with ``length = 0`` and geometries without ``length``
    are point probes.  If any averaging point OR either segment end lies off
    the mesh the channel is NaN with :data:`REASON_SEGMENT_OFF_MESH` -- a
    segment is never silently truncated.  ``model="point"``: the field at the
    center only (the earlier behavior).  The evidence that EFIT's calculated
    signals are length-averaged (and that the point model misses them by up to
    1.5 sigma, about +0.13 chi^2 per probe on fine-grid cases) is a comparison with
    EFIT's own calculated probe signals.
    """
    if model not in PROBE_MODELS:
        raise ValueError(f"bpol_probe: model must be one of {PROBE_MODELS}")
    pts = _points(geometry)
    nr = np.atleast_1d(np.asarray(geometry["direction_r"], dtype=float)).ravel()
    nz = np.atleast_1d(np.asarray(geometry["direction_z"], dtype=float)).ravel()
    if nr.shape != (pts.shape[0],) or nz.shape != (pts.shape[0],):
        raise ValueError("bpol_probe: direction arrays must match 'r'")
    norm = np.hypot(nr, nz)
    if np.any(~np.isfinite(norm) | (norm <= 0.0)):
        raise ValueError("bpol_probe: probe direction must be a finite, non-zero vector")
    has_len = "length" in geometry
    if model == "point" or not has_len:
        B, reason = field_at(eq, pts)
        return Synthetic((B[:, 0] * nr + B[:, 2] * nz) / norm, reason)
    seg, ends = probe_segment_points(geometry, n_avg)
    n, m = seg.shape[0], seg.shape[1]
    B, reason_pts = field_at(eq, seg.reshape(-1, 2))
    bn = (B[:, 0].reshape(n, m) * (nr / norm)[:, None] + B[:, 2].reshape(n, m) * (nz / norm)[:, None])
    reason_pts = reason_pts.reshape(n, m)
    good = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] > 0.0)
    end_ok = np.ones(n, dtype=bool)
    flat_ends = ends.reshape(-1, 2)
    fin = np.all(np.isfinite(flat_ends), axis=1) & (flat_ends[:, 0] > 0.0)
    on = np.zeros(flat_ends.shape[0], dtype=bool)
    on[fin] = on_mesh(eq, flat_ends[fin])
    end_ok = on.reshape(n, 2).all(axis=1)
    val = np.full(n, np.nan)
    reason = _reasons(n)
    for i in range(n):
        if not good[i]:
            reason[i] = REASON_BAD_GEOMETRY
            continue
        bad = [r for r in reason_pts[i] if r]
        if bad and all(r == REASON_NONFINITE for r in bad):
            reason[i] = REASON_NONFINITE
            continue
        if bad or not end_ok[i]:
            reason[i] = REASON_SEGMENT_OFF_MESH if np.any(np.abs(seg[i] - pts[i]) > 0) else (bad[0] if bad else
                                                                                               REASON_OFF_MESH)
            continue
        val[i] = float(np.mean(bn[i]))
    return Synthetic(val, reason)


def ip(eq, geometry=None):
    """Signed toroidal plasma current [A], length-1 array.

    ``get_globals()[0]``: Fortran ``gs_comp_globals`` divided by mu0
    (``tokamaker_get_globals``); the same integral OFT's ``fit_itor_eval``
    uses (``gs_itor_nl``/mu0).  IMAS ``magnetics.ip[0].data``.
    """
    return Synthetic(np.array([float(eq.get_globals()[0])]), _reasons(1))


def diamagnetic_flux(eq, geometry=None):
    """Diamagnetic flux [Wb], length-1 array.

    The toroidal-flux change due to the plasma, ``int_plasma (B_phi - F0/R)
    dR dZ`` -- OFT ``gs_dflux`` (``grad_shaf.F90``) via ``get_globals()[4]``,
    the same quantity OFT's ``dFlux_con`` evaluates.  IMAS
    ``magnetics.diamagnetic_flux[0].data`` [Wb].
    """
    return Synthetic(np.array([float(eq.get_globals()[4])]), _reasons(1))


def pf_current(eq, geometry):
    """Coil-set currents [A per turn] for ``geometry['coil_names']``.

    TokaMaker ``get_coil_currents()[0]``: ``(coil_currs +
    coil_vcont*vcontrol_val)/mu0`` (``tokamaker_get_coil_currents``), i.e.
    the physical current including any virtual vertical-stability-coil
    contribution.  OFT's own coil constraint (``fit_coil_eval``,
    ``grad_shaf_fit.F90``) evaluates ``coil_currs/mu0`` only and so differs
    whenever the VSC is active.  IMAS ``pf_active.coil[:].current`` is the
    current in one turn.  An unknown coil name raises ``KeyError``.
    """
    names = list(geometry["coil_names"])
    currents, _ = eq.get_coil_currents()
    missing = [n for n in names if n not in currents]
    if missing:
        raise KeyError(f"pf_current: coil sets not in the equilibrium: {missing}")
    return Synthetic(np.array([float(currents[n]) for n in names]), _reasons(len(names)))


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------
#: MSE output quantities: the IMAS polarisation angle [rad] or its tangent.
MSE_QUANTITIES = ("polarisation_angle", "tan_polarisation_angle")


def mse_from_field(B, coefficients, e_r=0.0, e_z=0.0, sign_pol=1.0,
                   sign_tor=1.0, quantity="polarisation_angle"):
    """MSE polarisation angle from the local fields and 9 geometric coefficients.

    IMAS ``mse.channel[:].active_spatial_resolution[:].geometric_coefficients``
    order (numerator B_Z, B_R, B_phi, E_R; denominator B_Z, B_R, B_phi, E_R,
    E_Z)::

        tan(gamma) = (c0 B_Z + c1 B_R + c2 B_phi + c3 E_R)
                   / (c4 B_Z + c5 B_R + c6 B_phi + c7 E_R + c8 E_Z)

    This generalizes the EFIT form ``(A1 B_Z + A5 E_R)/(A2 B_phi + A3 B_R +
    A4 B_Z + A6 E_Z + A7 E_R)`` (k-file ``AA1GAM`` .. ``AA7GAM``; B. W. Rice
    et al., Phys. Rev. Lett. 79, 2694 (1997)) with ``c0=A1, c3=A5, c4=A4,
    c5=A3, c6=A2, c7=A7, c8=A6`` and ``c1=c2=0``.

    *B* is ``(n, 3)`` = ``(B_R, B_phi, B_Z)`` [T] in the right-handed
    ``(R, phi, Z)`` frame the coefficients are defined in; ``e_r``, ``e_z``
    [V/m] are scalars or per-channel arrays.  ``sign_pol`` multiplies
    ``B_R`` and ``B_Z``, ``sign_tor`` multiplies ``B_phi``: a STATED map from
    the equilibrium's current/field orientation to the discharge's, never a
    fit.  ``quantity`` selects ``gamma = arctan(tan gamma)`` [rad], in
    (-pi/2, pi/2), or ``tan gamma``.
    """
    if quantity not in MSE_QUANTITIES:
        raise ValueError(f"mse: quantity must be one of {MSE_QUANTITIES}")
    B = np.asarray(B, dtype=float).reshape(-1, 3)
    c = np.asarray(coefficients, dtype=float).reshape(-1, 9)
    if c.shape[0] != B.shape[0]:
        raise ValueError("mse: one row of 9 coefficients per channel is required")
    br, bphi, bz = sign_pol * B[:, 0], sign_tor * B[:, 1], sign_pol * B[:, 2]
    er = np.broadcast_to(np.asarray(e_r, dtype=float), (B.shape[0],))
    ez = np.broadcast_to(np.asarray(e_z, dtype=float), (B.shape[0],))
    num = c[:, 0] * bz + c[:, 1] * br + c[:, 2] * bphi + c[:, 3] * er
    den = c[:, 4] * bz + c[:, 5] * br + c[:, 6] * bphi + c[:, 7] * er + c[:, 8] * ez
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = num / den
    return np.arctan(tg) if quantity == "polarisation_angle" else tg


def mse(eq, geometry, e_r=None, e_z=None, sign_pol=1.0, sign_tor=1.0,
        quantity="polarisation_angle"):
    """Synthetic MSE signal at each channel's ``(r, z)``; see :func:`mse_from_field`.

    ``geometry``: ``r``, ``z`` [m] and ``coefficients`` ``(n, 9)``.  ``e_r``
    and ``e_z`` [V/m] may be ``None`` (zero), a per-channel array, or a
    callable ``f(r, z)`` returning per-channel values.  E_r enters the
    numerator through ``c3`` and the denominator through ``c7``; neglecting a
    real E_r biases the inferred B_Z by ``(c3/c0) E_R`` to first order.
    """
    pts = _points(geometry)
    n = pts.shape[0]

    def _field(e):
        if e is None:
            return np.zeros(n)
        v = e(pts[:, 0], pts[:, 1]) if callable(e) else e
        v = np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
        return v

    B, reason = field_at(eq, pts)
    er, ez = _field(e_r), _field(e_z)
    val = mse_from_field(B, geometry["coefficients"], er, ez, sign_pol, sign_tor, quantity)
    bad = (reason == "") & ~(np.isfinite(val) & np.isfinite(er) & np.isfinite(ez))
    val = np.where(bad, np.nan, val)
    reason[bad] = REASON_NONFINITE
    return Synthetic(val, reason)


# ---------------------------------------------------------------------------
# Thomson scattering
# ---------------------------------------------------------------------------
#: Treatments of Thomson channels outside the last closed flux surface.
THOMSON_OUTSIDE = ("exclude", "extrapolate")


def thomson(eq, geometry, profile, outside="exclude"):
    """A kinetic profile given as ``profile(psi_N)`` evaluated at each channel.

    ``psi_N`` comes from ``psi(R, Z)`` (:func:`psi_n_at`) -- the equilibrium
    maps the channel onto a flux surface; the profile is not moved.  Channels
    outside the LCFS (:func:`inside_plasma`) are handled explicitly:

    * ``outside='exclude'`` (default): NaN with :data:`REASON_OUTSIDE_LCFS`;
    * ``outside='extrapolate'``: the profile is called with the channel's
      ``psi_N`` (> 1 in the scrape-off layer; < 1 in the private-flux region,
      which this option therefore maps onto a core surface -- use only with a
      profile and channel set for which that is intended).

    Units are the profile's: m^-3 for ``thomson_ne`` (IMAS
    ``thomson_scattering.channel[:].n_e``), eV for ``thomson_te`` (``t_e``).
    """
    if outside not in THOMSON_OUTSIDE:
        raise ValueError(f"thomson: outside must be one of {THOMSON_OUTSIDE}")
    pts = _points(geometry)
    psin, reason = psi_n_at(eq, pts)
    val = np.full(pts.shape[0], np.nan)
    ok = reason == ""
    inside = np.zeros(pts.shape[0], dtype=bool)
    inside[ok] = inside_plasma(eq, pts[ok], psin[ok])
    if outside == "exclude":
        reason[ok & ~inside] = REASON_OUTSIDE_LCFS
        use = ok & inside
        # psi_N of a point inside the plasma lies in [0, 1] up to FE
        # interpolation error at the axis; clip that rounding only.
        x = np.clip(psin[use], 0.0, 1.0)
    else:
        use = ok
        x = np.maximum(psin[use], 0.0)
    if np.any(use):
        val[use] = np.broadcast_to(np.asarray(profile(x), dtype=float), x.shape)
    bad = use & ~np.isfinite(val)
    reason[bad] = REASON_NONFINITE
    return Synthetic(val, reason)


def thomson_ne(eq, geometry, n_e, outside="exclude"):
    """Thomson electron density [m^-3]; ``n_e`` is a callable of psi_N."""
    return thomson(eq, geometry, n_e, outside)


def thomson_te(eq, geometry, t_e, outside="exclude"):
    """Thomson electron temperature [eV]; ``t_e`` is a callable of psi_N."""
    return thomson(eq, geometry, t_e, outside)


# ---------------------------------------------------------------------------
# not yet implemented
# ---------------------------------------------------------------------------
def q(eq, geometry):
    """Safety-factor constraints (``equilibrium...constraints.q``). NOT IMPLEMENTED.

    Planned: ``get_q`` at the constraint's psi_N (OFT ``q_con`` semantics).
    """
    raise NotImplementedError("kinrecon.diagnostics.q is a stub (not implemented)")


def vloop(eq, geometry):
    """Loop voltage (``magnetics.flux_loop[:].voltage``). NOT IMPLEMENTED.

    Needs the current-diffusion model; a static equilibrium has no loop voltage of
    its own.
    """
    raise NotImplementedError("kinrecon.diagnostics.vloop is a stub (needs the current-diffusion model)")
