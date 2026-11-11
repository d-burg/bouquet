r"""Flux-surface-averaged geometry for 1-D current diffusion.

:class:`FluxSurfaceGeometry` is a plain container of the flux-surface
quantities that the poloidal-flux diffusion equation (:mod:`.diffusion`)
needs, tabulated on the normalized toroidal-flux coordinate

.. code-block:: text

    rho = sqrt(Phi / Phi_b),     Phi(rho) = toroidal flux inside the surface [Wb]

(IMAS ``rho_tor_norm``).  Every coefficient of the equation comes from the
equilibrium GEOMETRY (surface averages of the metric), never from the current
or q profile.  This is the deliberate difference from QED.jl, which back-solves
``<|grad rho|^2/R^2>`` from ``j_tor`` and ``q`` (QED ``initialize.jl``).

Stored quantities (SI; ``<.>`` is the flux-surface average, ``<A> = (dV/drho)^-1
d/drho int A dV``):

========================  ==========================================  ==========
field                     definition                                  units
========================  ==========================================  ==========
``rho``                   rho_tor_norm grid, ``rho[0] = 0``,          --
                          ``rho[-1] = 1``, strictly increasing
``dvol_drho``             ``V' = dV/drho``                            m^3
``avg_grad_rho2_r2``      ``g2 = <|grad rho|^2 / R^2>``               m^-4
``avg_inv_r2``            ``g1 = <1/R^2>``                            m^-2
``fpol``                  ``F = R B_phi`` (stored positive)           T m
``phi_b``                 toroidal flux inside the boundary           Wb
``B0``, ``R0``            reference field and major radius (used      T, m
                          only for ``j_par = <j.B>/B0`` and l_i(3))
``avg_inv_r``             ``<1/R>`` (optional; IMAS ``j_tor``)        m^-1
``avg_r``                 ``<R>`` (optional; ``<j_phi>``)             m
``avg_b2``                ``<B^2>`` (optional; ``<j_phi>``)           T^2
``trapped_fraction``      ``f_t`` (optional; conductivity)            --
``eps``, ``r_geo``        surface inverse aspect ratio and major      --, m
                          radius (optional; collisionality)
``dphib_dt``              ``dPhi_b/dt`` (scalar)                      Wb/s
========================  ==========================================  ==========

The containers are filled from plain arrays.  Helpers:

* :meth:`FluxSurfaceGeometry.cylinder` -- the exact large-aspect-ratio
  (periodic-cylinder) metric, for analytic tests;
* :meth:`FluxSurfaceGeometry.circular` -- concentric circular surfaces in a
  torus, exact metric for any aspect ratio (a geometric, not an equilibrium,
  test case);
* :meth:`FluxSurfaceGeometry.from_psi_coordinates` -- change of coordinate
  from poloidal-flux tabulations (what equilibrium codes return);
* :func:`from_tokamaker` -- reads a TokaMaker equilibrium; the only place in
  this subpackage that touches OpenFUSIONToolkit (through the methods of the
  equilibrium object passed in; nothing is imported);
* :func:`choose_psi_n_edge` -- the domain-edge rule (below), returning the
  cutoff AND the reason;
* :class:`OuterLayerGeometry` -- volume-based tabulation of the layer between
  the cutoff surface and the last closed surface, for the lumped outer-layer
  element of :mod:`.diffusion`.

Domain edge (project ruling R29; evidence in the domain-edge experiment of
the kinetic-reconstruction record)
-----------------------------------------------------------------------------
On a diverted boundary the rho_tor metric is singular at the separatrix: q,
``drho/dpsi_N`` and ``<|grad rho|^2/R^2>`` grow like ``a + b ln(1/(1 - psi_N))``
(standard X-point result, ``oint dl/B_p`` with ``B_p`` linear in the distance to
the null; measured b = 0.4-2.8 on the equilibria examined), while ``V'``,
``<1/R^2>``, ``F`` and ``f_t`` stay finite.  The diffusion domain therefore ends
at a closed surface ``psi_N = c`` inside the boundary:

* ``c = 0.995`` by default;
* ``c = 0.99`` when the 2-D psi grid is 65 x 65 or coarser, or when two
  independent geometry routes disagree by more than 1e-3 (relative, in q,
  dV/dpsi or <B_p^2>) at 0.995;
* never beyond 0.999 (beyond it independent geometry routes disagree at the
  percent level and the layer outside holds < 0.1 % of Ip).

The current between ``c`` and the last closed surface is NOT dropped: it is a
lumped element with uniform parallel electric field, attached to the last
cell of the diffusion grid (:class:`.diffusion.OuterLayer`), and the Ip
boundary condition applies at the last closed surface.  ``rho`` of the domain
is normalized to the toroidal flux inside ``c``
(``rho = sqrt(Phi / Phi(c))``); ``meta['psi_n_edge']`` records ``c``.

Geometric identities (used by :meth:`FluxSurfaceGeometry.consistency`):

.. code-block:: text

    Phi(rho) = (1/2pi) int_0^rho F <1/R^2> V' drho'          (Lyons eq. 13, with V' = dV/drho)
    Phi(rho) = Phi_b rho^2                                    (definition of rho)
    q        = dPhi/dPsi_pol = F <1/R^2> (dV/dpsi) / (4 pi^2)   (psi = Psi_pol / 2pi)

References: B. C. Lyons, "Neoclassical current diffusion solver",
``equations_readme.pdf`` in QED.jl (General Atomics, Apache-2.0); S. C.
Jardin, *Computational Methods in Plasma Physics* (CRC Press, 2010),
flux-surface-averaged transport equations.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["FluxSurfaceGeometry", "OuterLayerGeometry", "EdgeChoice", "choose_psi_n_edge", "check_psi_samples",
           "from_tokamaker", "PSI_N_EDGE_DEFAULT", "PSI_N_EDGE_FALLBACK", "PSI_N_EDGE_MAX"]

MU0 = 4.0e-7 * np.pi  # vacuum permeability as fixed in QED.jl and OFT [H/m]

_REQUIRED = ("rho", "dvol_drho", "avg_grad_rho2_r2", "avg_inv_r2", "fpol")
_OPTIONAL_PROFILES = ("avg_inv_r", "avg_r", "avg_b2", "trapped_fraction", "eps", "r_geo")

# ----------------------------------------------------------------------------- domain-edge rule (R29)
PSI_N_EDGE_DEFAULT = 0.995    # default cutoff
PSI_N_EDGE_FALLBACK = 0.99    # coarse 2-D grid, or geometry routes disagree at the default
PSI_N_EDGE_MAX = 0.999        # never beyond
COARSE_GRID_MAX = 65          # "2-D grid <= 65^2": either grid dimension <= 65 points
ROUTE_TOLERANCE = 1.0e-3      # relative disagreement of two geometry routes at the default cutoff
XPOINT_MARGIN = 1.0e-3        # closed with margin: psi_N(X) - c >= 1e-3 (experiment report, recommendation item 1)

# ----------------------------------------------------------------------------- sample guard (from_psi_coordinates)
MIN_SAMPLE_GAP = 1.0e-10      # psi_N gap below which two samples are treated as the same surface
MERGE_RTOL = 1.0e-9           # ... and merged only if every tabulated value agrees to this relative tolerance
ISOLATED_GAP_RATIO = 1.0e-3   # a gap < 1e-3 x both neighboring gaps is an isolated near-duplicate: refused


@dataclass(frozen=True)
class EdgeChoice:
    """Result of :func:`choose_psi_n_edge`: the cutoff, the reason (human-readable) and the checks made.

    Unpacks as ``psi_n_edge, reason = choose_psi_n_edge(...)``.
    """

    psi_n_edge: float
    reason: str
    checks: Dict[str, object]

    def __iter__(self):
        yield self.psi_n_edge
        yield self.reason


def choose_psi_n_edge(grid_shape=None, route_disagreement=None, psi_n_xpoint=None, requested=None):
    r"""Diffusion-domain cutoff ``psi_N = c`` by the R29 rule, with the reason.

    Parameters
    ----------
    grid_shape : (int, int), optional
        Shape of the 2-D psi grid the geometry was traced from (g-file
        ``nw x nh``, IDS ``profiles_2d``).  ``None`` (e.g. a finite-element
        TokaMaker state, which has no such grid): the grid criterion does not
        apply, and the reason says so.  "65^2 or coarser" is taken as either
        dimension <= 65 (the conservative reading; the evidence was 65 x 65).
    route_disagreement : float, optional
        Largest relative disagreement, at ``psi_N = 0.995``, between two
        independent geometry routes (e.g. TokaMaker ``get_fsa`` vs a trace of
        the 2-D psi, or IDS ``profiles_1d`` vs a trace of ``profiles_2d``) in
        q, dV/dpsi or <B_p^2>.  ``None``: not checked (stated in the reason).
    psi_n_xpoint : float, optional
        psi_N of the first X-point inside the limiter (file normalization).  When
        given, the cutoff must satisfy ``psi_N(X) - c >= 1e-3`` (closed with
        margin; recommendation item 1 of the domain-edge experiment, an
        addition to the R29 text).  If even 0.99 fails, a ``ValueError`` is
        raised: no admissible cutoff exists and the caller must decide.
    requested : float, optional
        A cutoff chosen by the caller.  It is validated (``0 < c <= 0.999``;
        anything beyond 0.999 raises) and returned with reason
        ``"requested"``; if the rule would have chosen a smaller value the
        reason says so (it is not overridden silently in either direction).

    Returns
    -------
    EdgeChoice
        ``psi_n_edge`` in {0.995, 0.99} (or the validated request), ``reason``
        and ``checks`` (what was and was not checked).

    Source: project ruling R29 (2026-11-10) and the domain-edge experiment
    report it rests on (sections 2.2, 2.4 and 8).
    """
    checks = {}
    triggers = []
    if grid_shape is not None:
        shape = tuple(int(v) for v in grid_shape)
        coarse = min(shape) <= COARSE_GRID_MAX
        checks["grid_shape"] = dict(shape=shape, coarse=coarse)
        if coarse:
            triggers.append(f"2-D psi grid {shape[0]}x{shape[1]} is {COARSE_GRID_MAX}^2 or coarser")
    else:
        checks["grid_shape"] = "not applicable / not given"
    if route_disagreement is not None:
        d = float(route_disagreement)
        if not np.isfinite(d) or d < 0:
            raise ValueError("route_disagreement must be a finite, non-negative relative difference")
        checks["route_disagreement"] = dict(value=d, tolerance=ROUTE_TOLERANCE, at=PSI_N_EDGE_DEFAULT)
        if d > ROUTE_TOLERANCE:
            triggers.append(f"geometry routes disagree by {d:.2e} > {ROUTE_TOLERANCE:g} at psi_N = {PSI_N_EDGE_DEFAULT}")
    else:
        checks["route_disagreement"] = "not checked (no second geometry route given)"
    if psi_n_xpoint is not None:
        xp = float(psi_n_xpoint)
        checks["psi_n_xpoint"] = dict(value=xp, margin=XPOINT_MARGIN)
        if xp - PSI_N_EDGE_FALLBACK < XPOINT_MARGIN:
            raise ValueError(f"X-point at psi_N = {xp:.6f}: no cutoff in {{{PSI_N_EDGE_DEFAULT}, {PSI_N_EDGE_FALLBACK}}} "
                             f"is closed with margin {XPOINT_MARGIN:g}; choose the domain explicitly")
        if xp - PSI_N_EDGE_DEFAULT < XPOINT_MARGIN:
            triggers.append(f"X-point at psi_N = {xp:.6f} is within {XPOINT_MARGIN:g} of {PSI_N_EDGE_DEFAULT}")
    else:
        checks["psi_n_xpoint"] = "not checked (no X-point location given)"
    rule = PSI_N_EDGE_FALLBACK if triggers else PSI_N_EDGE_DEFAULT
    rule_reason = ("fallback (R29): " + "; ".join(triggers)) if triggers else "default (R29)"
    unchecked = [k for k, v in checks.items() if isinstance(v, str)]
    if unchecked:
        rule_reason += "; not checked: " + ", ".join(unchecked)
    if requested is not None:
        c = float(requested)
        if not 0.0 < c <= PSI_N_EDGE_MAX:
            raise ValueError(f"psi_n_edge = {c} is outside (0, {PSI_N_EDGE_MAX}] (R29: never beyond {PSI_N_EDGE_MAX})")
        reason = f"requested by the caller (rule would give {rule}: {rule_reason})"
        if c > rule:
            reason += "; WARNING: the request is beyond the rule's choice"
        checks["rule_choice"] = rule
        return EdgeChoice(c, reason, checks)
    return EdgeChoice(rule, rule_reason, checks)


def check_psi_samples(psi_n, arrays=None, min_gap=MIN_SAMPLE_GAP, merge_rtol=MERGE_RTOL,
                      isolated_ratio=ISOLATED_GAP_RATIO):
    r"""Validate a psi_N sample set; merge exact repeats, refuse near-duplicates.

    Criterion (stated once, applied everywhere a psi_N tabulation enters this
    subpackage):

    1. Samples must not decrease.
    2. Two consecutive samples closer than ``min_gap`` (1e-10 in psi_N, i.e.
       relative to the [0, 1] range) are the same surface.  They are merged
       (the second is dropped) only if every tabulated value in ``arrays``
       agrees to ``merge_rtol`` (1e-9 relative); otherwise a ``ValueError`` is
       raised -- the two values cannot both be right and the function does not
       guess which one is.
    3. After merging, an *isolated* gap -- one shorter than ``isolated_ratio``
       (1e-3) times every neighboring gap -- is refused (``ValueError``).  PCHIP
       derivatives are weighted harmonic means of the adjacent secants, so the
       secant over such a gap amplifies tracing noise by the gap ratio; the
       domain-edge experiment found two samples 1e-16 apart (from concatenated
       grids) silently giving 2-5e-3 errors in q.  Smoothly graded grids
       (geometric clustering, square-root spacing) have neighboring gap ratios
       of order one and pass.

    Returns ``(psi_n, arrays, merged)`` with ``merged`` the dropped indices
    (positions in the input).
    """
    x = np.asarray(psi_n, dtype=float)
    arrays = {k: np.asarray(v, dtype=float) for k, v in (arrays or {}).items()}
    for k, v in arrays.items():
        if v.shape != x.shape:
            raise ValueError(f"{k} has shape {v.shape}, psi_n has {x.shape}")
    if x.ndim != 1 or x.size < 2:
        raise ValueError("psi_n must be 1-D with at least 2 samples")
    if np.any(~np.isfinite(x)):
        raise ValueError("psi_n contains non-finite values")
    gaps = np.diff(x)
    if np.any(gaps < 0):
        i = int(np.argmax(gaps < 0))
        raise ValueError(f"psi_n must increase: sample {i + 1} ({x[i + 1]!r}) < sample {i} ({x[i]!r})")
    keep = np.ones(x.size, dtype=bool)
    merged = []
    for i in np.nonzero(gaps < min_gap)[0]:
        bad = []
        for k, v in arrays.items():
            a, b = v[i], v[i + 1]
            scale = max(abs(a), abs(b))
            if abs(a - b) > merge_rtol * scale:
                bad.append(f"{k}: {a!r} vs {b!r}")
        if bad:
            raise ValueError(f"psi_n samples {i} and {i + 1} are {gaps[i]:.1e} apart (< {min_gap:g}: the same surface) "
                             f"but their values differ ({'; '.join(bad[:3])}); remove one of them -- "
                             "the duplicate cannot be resolved automatically")
        keep[i + 1] = False
        merged.append(int(i + 1))
    x = x[keep]
    arrays = {k: v[keep] for k, v in arrays.items()}
    g = np.diff(x)
    if g.size >= 2:
        for i in range(g.size):
            nb = [g[j] for j in (i - 1, i + 1) if 0 <= j < g.size]
            if g[i] < isolated_ratio * min(nb):
                raise ValueError(f"psi_n samples {x[i]!r} and {x[i + 1]!r} are {g[i]:.2e} apart, less than "
                                 f"{isolated_ratio:g} x the neighboring spacing ({min(nb):.2e}): an isolated "
                                 "near-duplicate (e.g. from concatenating two sample grids). Remove or respace it.")
    return x, arrays, merged


@dataclass
class FluxSurfaceGeometry:
    """Flux-surface-averaged geometry on a ``rho_tor_norm`` grid (see module docstring)."""

    rho: np.ndarray
    dvol_drho: np.ndarray
    avg_grad_rho2_r2: np.ndarray
    avg_inv_r2: np.ndarray
    fpol: np.ndarray
    phi_b: float
    B0: float
    R0: float
    avg_inv_r: Optional[np.ndarray] = None
    avg_r: Optional[np.ndarray] = None
    avg_b2: Optional[np.ndarray] = None
    trapped_fraction: Optional[np.ndarray] = None
    eps: Optional[np.ndarray] = None
    r_geo: Optional[np.ndarray] = None
    dphib_dt: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for name in _REQUIRED + _OPTIONAL_PROFILES:
            val = getattr(self, name)
            if val is None:
                continue
            arr = np.array(val, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D")
            setattr(self, name, arr)
        n = self.rho.size
        if n < 4:
            raise ValueError("geometry needs at least 4 radial points")
        for name in _REQUIRED + _OPTIONAL_PROFILES:
            val = getattr(self, name)
            if val is not None and val.size != n:
                raise ValueError(f"{name} has {val.size} points, rho has {n}")
            if val is not None and not np.all(np.isfinite(val)):
                raise ValueError(f"{name} contains non-finite values")
        if abs(self.rho[0]) > 1e-12 or abs(self.rho[-1] - 1.0) > 1e-12 or np.any(np.diff(self.rho) <= 0):
            raise ValueError("rho must increase strictly from 0 to 1 (rho_tor_norm)")
        if np.any(self.dvol_drho < 0) or np.any(self.dvol_drho[1:] <= 0):
            raise ValueError("dvol_drho must be positive away from the axis")
        if np.any(self.avg_grad_rho2_r2 <= 0) or np.any(self.avg_inv_r2 <= 0):
            raise ValueError("<|grad rho|^2/R^2> and <1/R^2> must be positive")
        if np.any(self.fpol <= 0):
            raise ValueError("fpol must be positive: store |F| (the solver works in the frame F > 0; "
                             "see diffusion module docstring for the sign convention)")
        for name in ("phi_b", "B0", "R0"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, v)
        self.dphib_dt = float(self.dphib_dt)

    # ------------------------------------------------------------------ resampling
    def interpolator(self, name):
        """PCHIP interpolator of profile ``name`` in rho (shape-preserving, no overshoot)."""
        val = getattr(self, name)
        if val is None:
            raise KeyError(f"geometry has no {name!r}")
        return PchipInterpolator(self.rho, val, extrapolate=True)

    def evaluate(self, rho, names=None):
        """Dict of profiles resampled at ``rho`` (PCHIP).  ``names`` defaults to all present."""
        rho = np.asarray(rho, dtype=float)
        if names is None:
            names = [n for n in _REQUIRED[1:] + _OPTIONAL_PROFILES if getattr(self, n) is not None]
        return {n: self.interpolator(n)(rho) for n in names}

    def resample(self, rho_new):
        """A new geometry on grid ``rho_new`` (must run from 0 to 1)."""
        d = {n: getattr(self, n) for n in ("phi_b", "B0", "R0", "dphib_dt")}
        d.update(self.evaluate(rho_new))
        d["dvol_drho"] = np.where(np.asarray(rho_new) == 0.0, 0.0, d["dvol_drho"])
        return FluxSurfaceGeometry(rho=np.asarray(rho_new, dtype=float), meta=dict(self.meta), **d)

    # ------------------------------------------------------------------ identities
    def toroidal_flux(self):
        """Phi(rho) [Wb] integrated from the geometry: (1/2pi) int F <1/R^2> V' drho (Lyons eq. 13).

        Integrated with the PCHIP antiderivative of the integrand on the stored grid.
        """
        integrand = self.fpol * self.avg_inv_r2 * self.dvol_drho / (2.0 * np.pi)
        anti = PchipInterpolator(self.rho, integrand).antiderivative()
        return anti(self.rho) - anti(0.0)

    def consistency(self):
        """Check the geometric identity Phi(rho) = Phi_b rho^2.

        Returns a dict with ``phi_b_from_geometry`` and ``max_rel_error`` =
        max |Phi_geom(rho) - Phi_b rho^2| / Phi_b.  A value well above the
        quadrature error of the input grid means the stored averages, F or
        Phi_b do not describe one equilibrium.
        """
        phi = self.toroidal_flux()
        err = np.max(np.abs(phi - self.phi_b * self.rho ** 2)) / self.phi_b
        return dict(phi_b_from_geometry=float(phi[-1]), max_rel_error=float(err))

    def q_geometric(self, psi_prime):
        """Safety factor from geometry and dpsi/drho: q = dPhi/drho / (2 pi dpsi/drho), dPhi/drho = 2 Phi_b rho."""
        return 2.0 * self.phi_b * self.rho / (2.0 * np.pi * np.asarray(psi_prime))

    # ------------------------------------------------------------------ (de)serialization
    def to_dict(self):
        """Plain dict (numpy arrays kept; ``meta`` copied)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = dict(v) if f.name == "meta" else (None if v is None else (np.array(v) if isinstance(v, np.ndarray) else v))
        return out

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d.get(f.name) for f in fields(cls) if f.name in d})

    # ------------------------------------------------------------------ analytic geometries
    @classmethod
    def cylinder(cls, a, R0, B0, n=201):
        r"""Periodic-cylinder metric (large-aspect-ratio limit), exact.

        Minor radius ``a``, length ``2 pi R0``, uniform axial field ``B0``:
        ``rho = r/a``, ``V' = 4 pi^2 R0 a^2 rho``, ``<|grad rho|^2/R^2> =
        1/(a R0)^2``, ``<1/R^2> = 1/R0^2``, ``F = B0 R0``, ``Phi_b = pi B0 a^2``.
        In this geometry the diffusion equation reduces to
        ``dpsi/dt = (eta/mu0) (1/r) d/dr (r dpsi/dr) - ...`` (see :mod:`.diffusion`).
        """
        rho = np.linspace(0.0, 1.0, n)
        one = np.ones(n)
        return cls(rho=rho, dvol_drho=4.0 * np.pi ** 2 * R0 * a ** 2 * rho, avg_grad_rho2_r2=one / (a * R0) ** 2,
                   avg_inv_r2=one / R0 ** 2, fpol=B0 * R0 * one, phi_b=np.pi * B0 * a ** 2, B0=B0, R0=R0,
                   avg_inv_r=one / R0, avg_r=R0 * one, eps=a * rho / R0, r_geo=R0 * one,
                   meta=dict(kind="cylinder", a=a))

    @classmethod
    def circular(cls, a, R0, B0, n=201, fpol=None):
        r"""Concentric circular flux surfaces ``R = R0 + r cos(theta)``, exact metric.

        With ``s = sqrt(R0^2 - r^2)`` and ``|grad r| = 1`` the surface averages
        (weight ``dV = 2 pi R r dr dtheta``) are exactly

        .. code-block:: text

            dV/dr = 4 pi^2 R0 r,  <1/R^2> = <|grad r|^2/R^2> = 1/(R0 s),
            <1/R> = 1/R0,         <R> = R0 + r^2/(2 R0),
            Phi(r) = int_0^r 2 pi F(r') r'/s' dr'   ( = 2 pi F (R0 - s) for constant F)

        ``fpol`` is a callable F(r/a) [T m] (default constant ``B0 R0``).  This
        is a geometric test case (the surfaces are not a Grad-Shafranov
        solution); every identity of the module holds exactly.
        """
        from scipy.integrate import quad
        from scipy.optimize import brentq

        if fpol is None:
            def fpol(x):
                return B0 * R0 + 0.0 * np.asarray(x)

        def dphi_dr(r):
            return 2.0 * np.pi * fpol(r / a) * r / np.sqrt(R0 ** 2 - r ** 2)

        def phi_of_r(r):
            return quad(dphi_dr, 0.0, r, epsabs=0.0, epsrel=1e-13, limit=200)[0]

        phi_b = phi_of_r(a)
        rho = np.linspace(0.0, 1.0, n)
        r = np.empty(n)
        r[0], r[-1] = 0.0, a
        for k in range(1, n - 1):
            target = phi_b * rho[k] ** 2
            r[k] = brentq(lambda x: phi_of_r(x) - target, 0.0, a, xtol=1e-15 * a, rtol=1e-15)
        s = np.sqrt(R0 ** 2 - r ** 2)
        f_r = np.asarray(fpol(r / a), dtype=float) * np.ones(n)
        # drho/dr = (dPhi/dr) / (2 rho Phi_b); on axis the limit is sqrt(pi F(0) / (R0 Phi_b)).
        drho_dr = np.empty(n)
        drho_dr[1:] = dphi_dr(r[1:]) / (2.0 * rho[1:] * phi_b)
        drho_dr[0] = np.sqrt(np.pi * f_r[0] / (R0 * phi_b))
        inv_r2 = 1.0 / (R0 * s)
        return cls(rho=rho, dvol_drho=4.0 * np.pi ** 2 * R0 * r / drho_dr, avg_grad_rho2_r2=drho_dr ** 2 * inv_r2,
                   avg_inv_r2=inv_r2, fpol=f_r, phi_b=phi_b, B0=B0, R0=R0, avg_inv_r=np.ones(n) / R0,
                   avg_r=R0 + r ** 2 / (2.0 * R0), eps=r / R0, r_geo=R0 * np.ones(n),
                   meta=dict(kind="circular", a=a, r=r))

    # ------------------------------------------------------------------ change of coordinate
    @classmethod
    def from_psi_coordinates(cls, psi_n, psi_axis, psi_boundary, dvol_dpsi, avg_grad_psi2_r2, avg_inv_r2, fpol,
                             B0, R0, phi_b=None, nrho=201, extras=None, meta=None, psi_n_edge=None):
        r"""Build the geometry from tabulations on normalized poloidal flux.

        Parameters
        ----------
        psi_n : array (m,)
            Normalized poloidal flux of the samples, increasing in (0, 1]; the
            axis point itself is not needed (it cannot be traced).  Checked
            by :func:`check_psi_samples`: exact repeats (gap < 1e-10 with
            identical values) are merged, any other near-duplicate raises
            ``ValueError``.
        psi_axis, psi_boundary : float
            Poloidal flux per radian [Wb/rad] on axis and at the last closed
            surface (psi_N = 1) (either sign).
        dvol_dpsi, avg_grad_psi2_r2, avg_inv_r2, fpol : arrays (m,)
            ``dV/dpsi`` [m^3 rad/Wb], ``<|grad psi|^2/R^2> = <B_p^2>`` [T^2],
            ``<1/R^2>`` [m^-2], ``F`` [T m].  Signs are dropped (magnitudes).
        B0, R0 : float
            Reference field and major radius.
        phi_b : float, optional
            Toroidal flux inside the domain edge [Wb] (inside ``psi_n_edge``
            when that is given).  Default: the geometric integral below; with
            ``psi_n_edge=None`` and a last sample inside psi_N = 1 it is
            extrapolated (PCHIP) to psi_N = 1 -- the previous behavior, kept
            for untruncated domains (limited plasmas, analytic tests) and
            recorded in ``meta['edge_treatment']``.
        nrho : int
            Size of the uniform output grid in rho.
        extras : dict, optional
            Further profiles on ``psi_n`` named as the optional fields of this
            class (e.g. ``avg_inv_r``, ``trapped_fraction``); resampled.
        psi_n_edge : float, optional
            Domain cutoff ``c`` (R29; see :func:`choose_psi_n_edge`).  The last
            sample must be the cutoff surface itself (``|psi_n[-1] - c| <=
            1e-12``; nothing is extrapolated) and ``rho`` is normalized to the
            toroidal flux inside it, ``rho = sqrt(Phi / Phi(c))``.  The layer
            ``c < psi_N < 1`` is described separately by
            :class:`OuterLayerGeometry`.

        ``meta['psi_n_samples']`` / ``meta['rho_samples']`` keep the
        psi_n -> rho map of the samples (psi_n in the equilibrium's own
        normalization, 1 at the last closed surface), for mapping kinetic
        profiles given on psi_n onto rho.  ``meta['psi_n_edge']`` is the
        psi_N of the domain edge (1.0 when not truncated).

        Method (all geometric; q is not used):

        .. code-block:: text

            dPhi/dpsi = F <1/R^2> dV/dpsi / (2 pi)                 (Lyons eq. 13)
            Phi(psi_n) = |psi_b - psi_a| int_0^psi_n dPhi/dpsi dpsi_n'  (integrand extrapolated
                                                                     linearly to the axis)
            rho = sqrt(Phi / Phi_b),  drho/dpsi = (dPhi/dpsi) / (2 rho Phi_b)
            V'_rho = (dV/dpsi) / (drho/dpsi),  <|grad rho|^2/R^2> = (drho/dpsi)^2 <|grad psi|^2/R^2>

        Near the axis ``<|grad psi|^2/R^2>`` vanishes like ``rho^2`` and
        ``(drho/dpsi)^2`` diverges like ``rho^-2``; their product is finite and
        its axis value is extrapolated as ``g2(rho) = c0 + c1 rho^2`` from the
        two innermost samples.  All profiles are then PCHIP-resampled in rho.
        """
        tab = dict(dvol_dpsi=dvol_dpsi, avg_grad_psi2_r2=avg_grad_psi2_r2, avg_inv_r2=avg_inv_r2, fpol=fpol)
        tab.update({"extra:" + k: v for k, v in (extras or {}).items()})
        psi_true, tab, merged = check_psi_samples(psi_n, tab)
        extras = {k[6:]: v for k, v in tab.items() if k.startswith("extra:")}
        if psi_true[0] <= 0 or psi_true[-1] > 1:
            raise ValueError("psi_n must lie within (0, 1]")
        dpsi = abs(float(psi_boundary) - float(psi_axis))
        if psi_n_edge is None:
            c = 1.0
            if psi_true[-1] == 1.0:
                edge_treatment = "domain ends at the last sample, psi_N = 1"
            elif phi_b is None:
                edge_treatment = (f"untruncated (previous behavior): Phi_b extrapolated (PCHIP) from the last sample "
                                  f"psi_N = {psi_true[-1]:.6g} to psi_N = 1")
            else:
                edge_treatment = f"untruncated: Phi_b given by the caller; last sample psi_N = {psi_true[-1]:.6g}"
        else:
            c = float(psi_n_edge)
            if not 0.0 < c <= 1.0:
                raise ValueError("psi_n_edge must lie in (0, 1]")
            if abs(psi_true[-1] - c) > 1e-12:
                raise ValueError(f"the last sample (psi_N = {psi_true[-1]!r}) must be the cutoff surface psi_n_edge = {c!r}: "
                                 "sample the cutoff surface itself (nothing is extrapolated)")
            psi_true = psi_true.copy()
            psi_true[-1] = c
            edge_treatment = (f"truncated at psi_N = {c:g} (R29): rho normalized to Phi(psi_N = {c:g}); the layer outside "
                              "is the lumped outer element (see OuterLayerGeometry / diffusion.OuterLayer)")
        psi_n = psi_true / c          # normalized to the domain edge
        dpsi = dpsi * c               # flux from the axis to the domain edge
        vps = np.abs(np.asarray(tab["dvol_dpsi"], dtype=float))
        gp = np.abs(np.asarray(tab["avg_grad_psi2_r2"], dtype=float))
        g1 = np.asarray(tab["avg_inv_r2"], dtype=float)
        F = np.abs(np.asarray(tab["fpol"], dtype=float))
        dphi_dpsi = F * g1 * vps / (2.0 * np.pi)
        # integrand on psi_n with a linear extrapolation to the axis
        x = np.concatenate(([0.0], psi_n))
        slope = (dphi_dpsi[1] - dphi_dpsi[0]) / (psi_n[1] - psi_n[0])
        y = np.concatenate(([dphi_dpsi[0] - slope * psi_n[0]], dphi_dpsi)) * dpsi
        anti = PchipInterpolator(x, y, extrapolate=True).antiderivative()
        phi = anti(psi_n) - anti(0.0)
        if phi_b is None:
            phi_b = float(anti(1.0) - anti(0.0))
        rho_s = np.sqrt(phi / phi_b)
        drho_dpsi = dphi_dpsi / (2.0 * rho_s * phi_b)
        vp_rho = vps / drho_dpsi
        g2 = drho_dpsi ** 2 * gp
        # axis values: V' = 0, g2 even in rho
        c1 = (g2[1] - g2[0]) / (rho_s[1] ** 2 - rho_s[0] ** 2)
        g2_axis = g2[0] - c1 * rho_s[0] ** 2

        def axis_even(v):
            v = np.asarray(v, dtype=float)
            c = (v[1] - v[0]) / (rho_s[1] ** 2 - rho_s[0] ** 2)
            return v[0] - c * rho_s[0] ** 2

        rr = np.concatenate(([0.0], rho_s))
        rho_out = np.linspace(0.0, 1.0, nrho)

        def onto(v_samples, v_axis):
            return PchipInterpolator(rr, np.concatenate(([v_axis], v_samples)), extrapolate=True)(rho_out)

        prof = dict(dvol_drho=onto(vp_rho, 0.0), avg_grad_rho2_r2=onto(g2, g2_axis),
                    avg_inv_r2=onto(g1, axis_even(g1)), fpol=onto(F, axis_even(F)))
        prof["dvol_drho"][0] = 0.0
        for name, v in (extras or {}).items():
            if name not in _OPTIONAL_PROFILES:
                raise KeyError(f"unknown optional profile {name!r}")
            v = np.asarray(v, dtype=float)
            v_axis = 0.0 if name in ("trapped_fraction", "eps") else axis_even(v)
            prof[name] = onto(v, v_axis)
        m = dict(meta or {})
        m.setdefault("source", "psi_coordinates")
        m.update(rho_last_sample=float(rho_s[-1]), psi_n_last_sample=float(psi_true[-1]),
                 psi_n_samples=psi_true.copy(), rho_samples=rho_s.copy(), psi_n_edge=c, edge_treatment=edge_treatment,
                 merged_duplicate_samples=merged)
        return cls(rho=rho_out, phi_b=float(phi_b), B0=B0, R0=R0, meta=m, **prof)


@dataclass
class OuterLayerGeometry:
    r"""Volume-based tabulation of the outer layer ``psi_n_edge < psi_N < 1``.

    The lumped outer-layer element of :mod:`.diffusion` needs only volume
    integrals over the layer (domain-edge experiment, recommendation 8;
    ruling R29):

    .. code-block:: text

        A_layer = int sigma_par <1/R^2> / (2 pi F) dV        (uniform-E_par ohmic conductance)
        B_layer = int <J_NI.B> / (2 pi F^2) dV               (non-inductive source)

    (the integrals of the coefficients ``a`` and ``S`` of the diffusion
    equation over the layer, written in volume: ``a drho = sigma <1/R^2>
    dV/(2 pi F)``).  ``dV``, ``<1/R^2>``, ``F``, ``sigma`` and ``J_NI`` are all
    regular through a separatrix, so no metric is needed.  Integrals use the
    trapezoid rule in enclosed volume on the samples, plus the volume between
    the last sample and psi_N = 1 (``volume_tail``) weighted by the last
    sample's integrand.

    Fields
    ------
    psi_n : (m,)        samples; ``psi_n[0]`` is the cutoff, increasing, ``<= 1``
    volume : (m,)       ``V(psi_n) - V(psi_n[0])`` [m^3]
    avg_inv_r2 : (m,)   ``<1/R^2>`` [m^-2]
    fpol : (m,)         ``|F|`` [T m]
    fpol_lcfs           ``|F|`` at psi_N = 1 [T m] (Ip is applied there: outer flux ``I/F = Ip/F_lcfs``)
    volume_tail         volume between ``psi_n[-1]`` and psi_N = 1 [m^3]
    ampere_current      optional ``I_ref(psi_n) = (dV/dpsi) <B_p^2> / (2 pi mu0)`` [A], the reference
                        equilibrium's enclosed toroidal current (Ampere; regular at a separatrix)
    dpsi_ref            optional ``|psi(1) - psi_axis|`` of the reference equilibrium [Wb/rad]
    extras              further profiles on the samples (e.g. ``trapped_fraction``, ``eps``, ``r_geo``,
                        ``q``) for evaluating the layer conductivity; not used here
    meta                provenance (tail method, ...)

    ``ampere_current`` and ``dpsi_ref`` allow full-plasma q95 and l_i(3) on the
    reference surfaces (fixed-surface approximation: on fixed surfaces
    ``dpsi/dpsi_N`` scales with the enclosed current), as in the domain-edge
    experiment.
    """

    psi_n: np.ndarray
    volume: np.ndarray
    avg_inv_r2: np.ndarray
    fpol: np.ndarray
    fpol_lcfs: float
    volume_tail: float = 0.0
    ampere_current: Optional[np.ndarray] = None
    dpsi_ref: Optional[float] = None
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("psi_n", "volume", "avg_inv_r2", "fpol"):
            setattr(self, name, np.array(getattr(self, name), dtype=float))
        x = self.psi_n
        if x.ndim != 1 or x.size < 2:
            raise ValueError("the outer layer needs at least 2 samples (the cutoff and one more)")
        if np.any(np.diff(x) <= 0) or x[0] <= 0 or x[-1] > 1:
            raise ValueError("outer-layer psi_n must increase strictly within (0, 1]")
        for name in ("volume", "avg_inv_r2", "fpol"):
            v = getattr(self, name)
            if v.shape != x.shape or not np.all(np.isfinite(v)):
                raise ValueError(f"outer-layer {name} must be finite with one value per sample")
        if abs(self.volume[0]) > 0 or np.any(np.diff(self.volume) <= 0):
            raise ValueError("outer-layer volume must start at 0 at the cutoff and increase")
        if np.any(self.avg_inv_r2 <= 0) or np.any(self.fpol <= 0):
            raise ValueError("outer-layer <1/R^2> and |F| must be positive")
        self.fpol_lcfs = float(self.fpol_lcfs)
        self.volume_tail = float(self.volume_tail)
        if not (np.isfinite(self.fpol_lcfs) and self.fpol_lcfs > 0) or not (np.isfinite(self.volume_tail)
                                                                             and self.volume_tail >= 0):
            raise ValueError("fpol_lcfs must be positive and volume_tail non-negative")
        if self.ampere_current is not None:
            self.ampere_current = np.array(self.ampere_current, dtype=float)
            if self.ampere_current.shape != x.shape or np.any(~np.isfinite(self.ampere_current)) \
                    or np.any(self.ampere_current <= 0):
                raise ValueError("ampere_current must be positive and finite with one value per sample")
        if self.dpsi_ref is not None:
            self.dpsi_ref = float(self.dpsi_ref)
        self.extras = {k: np.array(v, dtype=float) for k, v in (self.extras or {}).items()}

    @property
    def psi_n_edge(self):
        """The cutoff surface (first sample)."""
        return float(self.psi_n[0])

    @property
    def total_volume(self):
        """Volume of the layer, cutoff to psi_N = 1 [m^3]."""
        return float(self.volume[-1] + self.volume_tail)

    def cumulative(self, values):
        """``int_c^{psi_n[k]} values dV`` at the samples (trapezoid in volume)."""
        v = np.asarray(values, dtype=float) * np.ones_like(self.psi_n)
        return np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(self.volume))))

    def integral(self, values):
        """``int_c^1 values dV``: :meth:`cumulative` to the last sample plus ``values[-1] * volume_tail``."""
        v = np.asarray(values, dtype=float) * np.ones_like(self.psi_n)
        return float(self.cumulative(v)[-1] + v[-1] * self.volume_tail)

    def conductance_density(self, sigma):
        """``sigma <1/R^2> / (2 pi F)`` at the samples (integrand of A_layer)."""
        return np.asarray(sigma, dtype=float) * self.avg_inv_r2 / (2.0 * np.pi * self.fpol)

    def source_density(self, jni):
        """``<J_NI.B> / (2 pi F^2)`` at the samples (integrand of B_layer)."""
        return np.asarray(jni, dtype=float) / (2.0 * np.pi * self.fpol ** 2)

    @classmethod
    def from_psi_coordinates(cls, psi_n, psi_axis, psi_boundary, dvol_dpsi, avg_inv_r2, fpol,
                             avg_grad_psi2_r2=None, fpol_lcfs=None, extras=None, meta=None, n_tail_fit=6):
        r"""Layer tabulation from samples on normalized poloidal flux.

        ``psi_n[0]`` must be the cutoff surface; the samples run outward
        (checked by :func:`check_psi_samples`) and may stop short of psi_N = 1
        (on a separatrix ``dV/dpsi`` is infinite there).  Arguments as in
        :meth:`FluxSurfaceGeometry.from_psi_coordinates` (magnitudes are
        used); ``fpol_lcfs`` defaults to the PCHIP extrapolation of ``|F|`` to
        psi_N = 1 (F is regular).

        Volume: ``V(psi_n) = |psi_b - psi_a| int dV/dpsi dpsi_n`` by the PCHIP
        antiderivative on the samples.  Beyond the last sample ``x_m < 1``,
        ``dV/dpsi_n`` is fitted as ``a + b ln(1/(1 - x))`` (the logarithmic
        X-point divergence; regular boundaries give b ~ 0) on the last
        ``n_tail_fit`` samples and integrated exactly,
        ``V_tail = d (a + b (ln(1/d) + 1))`` with ``d = 1 - x_m``.  With fewer
        than 3 samples, or a negative fitted tail, ``dV/dpsi_n`` is held at its
        last value instead; ``meta['tail_method']`` and
        ``meta['tail_volume_fraction']`` record what was done.
        """
        tab = dict(dvol_dpsi=dvol_dpsi, avg_inv_r2=avg_inv_r2, fpol=fpol)
        if avg_grad_psi2_r2 is not None:
            tab["avg_grad_psi2_r2"] = avg_grad_psi2_r2
        tab.update({"extra:" + k: v for k, v in (extras or {}).items()})
        x, tab, merged = check_psi_samples(psi_n, tab)
        dpsi = abs(float(psi_boundary) - float(psi_axis))
        vps = np.abs(tab["dvol_dpsi"])
        if not np.all(np.isfinite(vps)):
            raise ValueError("dV/dpsi must be finite at every layer sample (stop the samples short of a separatrix)")
        dvdx = dpsi * vps
        anti = PchipInterpolator(x, dvdx).antiderivative()
        vol = anti(x) - anti(x[0])
        d = 1.0 - x[-1]
        m = dict(meta or {})
        m["merged_duplicate_samples"] = merged
        if d <= 0.0:
            tail, method = 0.0, "none (last sample at psi_N = 1)"
        else:
            k = min(int(n_tail_fit), x.size)
            tail = None
            if k >= 3:
                L = np.log(1.0 / (1.0 - x[-k:]))
                b, a = np.polyfit(L, dvdx[-k:], 1)
                tail = d * (a + b * (np.log(1.0 / d) + 1.0))
                method = f"log fit a + b ln(1/(1-x)) on the last {k} samples (a = {a:.4g}, b = {b:.4g} m^3)"
                if not np.isfinite(tail) or tail < 0:
                    tail = None
            if tail is None:
                tail = d * dvdx[-1]
                method = "constant dV/dpsi_n beyond the last sample"
        m.update(tail_method=method, tail_volume_fraction=float(tail / (vol[-1] + tail)), dpsi_ref=dpsi)
        F = np.abs(tab["fpol"])
        if fpol_lcfs is None:
            fpol_lcfs = float(PchipInterpolator(x, F, extrapolate=True)(1.0))
            m["fpol_lcfs"] = "PCHIP extrapolation of |F| to psi_N = 1"
        amp = None
        if "avg_grad_psi2_r2" in tab:
            amp = vps * np.abs(tab["avg_grad_psi2_r2"]) / (2.0 * np.pi * MU0)
        ex = {k[6:]: v for k, v in tab.items() if k.startswith("extra:")}
        return cls(psi_n=x, volume=vol, avg_inv_r2=tab["avg_inv_r2"], fpol=F, fpol_lcfs=abs(float(fpol_lcfs)),
                   volume_tail=float(tail), ampere_current=amp, dpsi_ref=dpsi, extras=ex, meta=m)


def from_tokamaker(gs, R0, B0=None, npsi=200, psi_pad_axis=1.0e-3, psi_n_edge="auto", nrho=201,
                   with_trapped_fraction=True, n_layer=40, layer_pad=1.0e-2, route_disagreement=None,
                   psi_pad_edge=None):
    r"""Fill a :class:`FluxSurfaceGeometry` (and its outer layer) from a solved TokaMaker equilibrium.

    The only function in ``bouquet.kinrecon.current`` that touches
    OpenFUSIONToolkit, and only through the methods of the equilibrium object
    it is given (no import).  It calls ``gs.get_fsa`` (surface averages traced in
    one pass: ``dV/dPsi``, ``<Bp^2> = <|grad psi|^2/R^2>``, ``<1/R^2>``,
    ``<1/R>``, ``<R>``, ``F``, ``R_min``, ``R_max``), ``gs.get_profiles`` (F at
    psi_N = 1) and optionally ``gs.sauter_fc`` (``f_t = 1 - f_c``), then
    :meth:`FluxSurfaceGeometry.from_psi_coordinates`.

    Domain edge (R29; module docstring)
    -----------------------------------
    ``psi_n_edge="auto"`` (default): ``c`` from :func:`choose_psi_n_edge`
    (0.995 unless ``route_disagreement`` -- measured by the caller between
    ``get_fsa`` and an independent route at 0.995 -- exceeds 1e-3; a
    TokaMaker state has no 2-D grid, so the grid criterion does not apply).
    A number: that cutoff (validated: never beyond 0.999).  The domain samples
    run (uniform in sqrt(psi_N)) from ``psi_pad_axis`` to exactly ``c``; ``rho``
    is normalized to Phi(c).  The layer ``c < psi_N < 1`` is traced on
    ``n_layer`` samples clustered geometrically from ``c`` to
    ``1 - layer_pad (1 - c)`` and returned as an :class:`OuterLayerGeometry`
    in ``meta['outer_layer_geometry']`` (with ``f_t``, ``eps``, ``r_geo`` and
    TokaMaker's ``q`` in its ``extras``, for evaluating the layer
    conductivity).  ``meta['psi_n_edge']``, ``meta['edge_choice']`` (the
    reason) and ``meta['edge_checks']`` record the choice.

    ``psi_n_edge=None``: the previous behavior, kept as an explicit option --
    surfaces up to ``1 - psi_pad_edge`` (default 1e-3) and Phi_b extrapolated
    (PCHIP) to psi_N = 1, no outer layer; the extrapolated fraction of Phi_b
    is recorded as ``meta['phi_b_extrapolated_fraction']``.  On a diverted
    boundary this under-represents the logarithmic q rise (domain-edge
    experiment, issue 4) and is not R29-conformant.

    ``R0`` (reference major radius) must be given; ``B0`` defaults to
    ``|F0| / R0`` with TokaMaker's vacuum ``F0``.  ``<B^2> = <B_p^2> + F^2
    <1/R^2>`` (exact, F a flux function).  ``r_geo = (R_max + R_min)/2`` and
    ``eps = (R_max - R_min)/(R_max + R_min)`` per surface.

    Self-check stored in ``meta``: TokaMaker's own q against the geometric
    ``q = F <1/R^2> dV/dpsi / (4 pi^2)`` (max relative difference over the
    domain samples).  It is a diagnostic only; nothing is fitted to it.
    """
    legacy = psi_n_edge is None
    if legacy:
        pad = 1.0e-3 if psi_pad_edge is None else float(psi_pad_edge)
        choice = None
        c = 1.0 - pad
        psi_n = np.linspace(np.sqrt(psi_pad_axis), np.sqrt(c), npsi) ** 2
    else:
        if psi_pad_edge is not None:
            raise ValueError("psi_pad_edge applies only to psi_n_edge=None (previous behavior)")
        if isinstance(psi_n_edge, str):
            if psi_n_edge != "auto":
                raise ValueError("psi_n_edge must be 'auto', a number in (0, 0.999], or None")
            choice = choose_psi_n_edge(route_disagreement=route_disagreement)
        else:
            choice = choose_psi_n_edge(route_disagreement=route_disagreement, requested=psi_n_edge)
        c = choice.psi_n_edge
        # uniform in sqrt(psi_n), i.e. close to uniform in rho near the axis (uniform psi_n
        # samples would leave the axis region resolved only as npsi^-1/2 in rho); last sample = c exactly
        psi_n = np.linspace(np.sqrt(psi_pad_axis), np.sqrt(c), npsi) ** 2
        psi_n[-1] = c

    def trace(x):
        fsa = gs.get_fsa(psi=x)
        if np.any(~np.isfinite(fsa["dV/dPsi"])):
            bad = np.asarray(x)[~np.isfinite(fsa["dV/dPsi"])]
            raise RuntimeError(f"TokaMaker flux-surface trace failed on {bad.size} surfaces (psi_N {bad.min():.6g} ... "
                               f"{bad.max():.6g})")
        F = np.abs(fsa["F"])
        g1 = fsa["<1/R^2>"]
        bp2 = fsa["<Bp^2>"]
        ex = dict(avg_inv_r=fsa["<1/R>"], avg_r=fsa["<R>"], avg_b2=bp2 + F ** 2 * g1,
                  r_geo=0.5 * (fsa["R_min"] + fsa["R_max"]),
                  eps=(fsa["R_max"] - fsa["R_min"]) / (fsa["R_max"] + fsa["R_min"]))
        if with_trapped_fraction:
            _, fc, _, _, _ = gs.sauter_fc(psi=x, return_eps=True)
            ex["trapped_fraction"] = 1.0 - np.asarray(fc)
        return fsa, F, g1, np.abs(fsa["dV/dPsi"]), bp2, ex

    fsa, F, g1, vps, bp2, extras = trace(psi_n)
    q_geo = F * g1 * vps / (4.0 * np.pi ** 2)
    q_tm = np.abs(np.asarray(fsa["q"]))
    if B0 is None:
        B0 = abs(float(fsa["F0"])) / R0
    meta = dict(source="tokamaker", q_geometric_vs_tokamaker_max_rel=float(np.max(np.abs(q_geo / q_tm - 1.0))))
    geom = FluxSurfaceGeometry.from_psi_coordinates(
        psi_n, fsa["psi_axis"], fsa["psi_boundary"], vps, bp2, g1, F, B0=B0, R0=R0, nrho=nrho, extras=extras,
        meta=meta, psi_n_edge=None if legacy else c)
    if legacy:
        # fraction of Phi_b that comes from the extrapolated edge strip
        dphi = F * g1 * vps / (2.0 * np.pi) * abs(fsa["psi_boundary"] - fsa["psi_axis"])
        phi_in = np.trapezoid(dphi, psi_n)
        geom.meta["phi_b_extrapolated_fraction"] = float(1.0 - phi_in / geom.phi_b)
        geom.meta["edge_choice"] = "previous behavior (psi_n_edge=None): extrapolated to psi_N = 1, no outer layer"
        return geom
    geom.meta["edge_choice"] = choice.reason
    geom.meta["edge_checks"] = choice.checks
    # outer layer c < psi_N < 1: geometric clustering toward the boundary
    xl = 1.0 - (1.0 - c) * np.geomspace(1.0, float(layer_pad), int(n_layer))
    xl[0] = c
    lf, lF, lg1, lvps, lbp2, lex = trace(xl)
    lex = {k: lex[k] for k in ("trapped_fraction", "eps", "r_geo") if k in lex}
    lex["q"] = np.abs(np.asarray(lf["q"]))
    f_lcfs = abs(float(np.asarray(gs.get_profiles(psi=np.array([1.0]))[1])[0]))
    layer = OuterLayerGeometry.from_psi_coordinates(
        xl, lf["psi_axis"], lf["psi_boundary"], lvps, lg1, lF, avg_grad_psi2_r2=lbp2, fpol_lcfs=f_lcfs, extras=lex,
        meta=dict(source="tokamaker", fpol_lcfs="TokaMaker get_profiles at psi_N = 1"))
    geom.meta["outer_layer_geometry"] = layer
    return geom
