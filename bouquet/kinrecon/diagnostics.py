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


def bpol_probe(eq, geometry):
    """Poloidal-field probe signal ``B . n`` [T].

    ``n = (direction_r, direction_z)`` is the probe's unit normal in the
    poloidal plane.  IMAS DD 3.41.0
    ``magnetics.b_field_pol_probe[:].poloidal_angle`` is measured CLOCKWISE
    from the +R direction in the (R, Z) plane, so ``n = (cos a, -sin a)``
    (``a = -pi/2`` points along +Z); the conversion is
    :func:`bouquet.kinrecon.measurements.probe_direction`.  Identical to OFT ``fit_field_eval``
    (``grad_shaf_fit.F90``) with a zero toroidal normal component.  The
    direction vector is normalized here and must be non-zero.
    """
    pts = _points(geometry)
    nr = np.atleast_1d(np.asarray(geometry["direction_r"], dtype=float)).ravel()
    nz = np.atleast_1d(np.asarray(geometry["direction_z"], dtype=float)).ravel()
    if nr.shape != (pts.shape[0],) or nz.shape != (pts.shape[0],):
        raise ValueError("bpol_probe: direction arrays must match 'r'")
    norm = np.hypot(nr, nz)
    if np.any(~np.isfinite(norm) | (norm <= 0.0)):
        raise ValueError("bpol_probe: probe direction must be a finite, non-zero vector")
    B, reason = field_at(eq, pts)
    return Synthetic((B[:, 0] * nr + B[:, 2] * nz) / norm, reason)


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
    raise NotImplementedError("kinrecon.diagnostics.q is a stub (P0b)")


def vloop(eq, geometry):
    """Loop voltage (``magnetics.flux_loop[:].voltage``). NOT IMPLEMENTED.

    Needs the current-diffusion model (P0c / L2); a static equilibrium has no
    loop voltage of its own.
    """
    raise NotImplementedError("kinrecon.diagnostics.vloop is a stub (needs L2)")
