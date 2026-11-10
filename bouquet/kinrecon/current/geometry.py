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
  equilibrium object passed in; nothing is imported).

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

__all__ = ["FluxSurfaceGeometry", "from_tokamaker"]

MU0 = 4.0e-7 * np.pi  # vacuum permeability as fixed in QED.jl and OFT [H/m]

_REQUIRED = ("rho", "dvol_drho", "avg_grad_rho2_r2", "avg_inv_r2", "fpol")
_OPTIONAL_PROFILES = ("avg_inv_r", "avg_r", "avg_b2", "trapped_fraction", "eps", "r_geo")


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
                             B0, R0, phi_b=None, nrho=201, extras=None, meta=None):
        r"""Build the geometry from tabulations on normalized poloidal flux.

        Parameters
        ----------
        psi_n : array (m,)
            Normalized poloidal flux of the samples, strictly increasing in
            (0, 1]; the axis point itself is not needed (it cannot be traced).
        psi_axis, psi_boundary : float
            Poloidal flux per radian [Wb/rad] on axis and boundary (either sign).
        dvol_dpsi, avg_grad_psi2_r2, avg_inv_r2, fpol : arrays (m,)
            ``dV/dpsi`` [m^3 rad/Wb], ``<|grad psi|^2/R^2> = <B_p^2>`` [T^2],
            ``<1/R^2>`` [m^-2], ``F`` [T m].  Signs are dropped (magnitudes).
        B0, R0 : float
            Reference field and major radius.
        phi_b : float, optional
            Toroidal flux inside the boundary [Wb].  Default: the geometric
            integral below, extrapolated (PCHIP) to ``psi_n = 1`` when the
            last sample is inside the boundary.
        nrho : int
            Size of the uniform output grid in rho.
        extras : dict, optional
            Further profiles on ``psi_n`` named as the optional fields of this
            class (e.g. ``avg_inv_r``, ``trapped_fraction``); resampled.

        ``meta['psi_n_samples']`` / ``meta['rho_samples']`` keep the
        psi_n -> rho map of the samples (for mapping kinetic profiles given on
        psi_n onto rho).

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
        psi_n = np.asarray(psi_n, dtype=float)
        if np.any(np.diff(psi_n) <= 0) or psi_n[0] <= 0 or psi_n[-1] > 1:
            raise ValueError("psi_n must increase strictly within (0, 1]")
        dpsi = abs(float(psi_boundary) - float(psi_axis))
        vps = np.abs(np.asarray(dvol_dpsi, dtype=float))
        gp = np.abs(np.asarray(avg_grad_psi2_r2, dtype=float))
        g1 = np.asarray(avg_inv_r2, dtype=float)
        F = np.abs(np.asarray(fpol, dtype=float))
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
        m.update(source="psi_coordinates", rho_last_sample=float(rho_s[-1]), psi_n_last_sample=float(psi_n[-1]),
                 psi_n_samples=psi_n.copy(), rho_samples=rho_s.copy())
        return cls(rho=rho_out, phi_b=float(phi_b), B0=B0, R0=R0, meta=m, **prof)


def from_tokamaker(gs, R0, B0=None, npsi=200, psi_pad_axis=1.0e-3, psi_pad_edge=1.0e-3, nrho=201,
                   with_trapped_fraction=True):
    r"""Fill a :class:`FluxSurfaceGeometry` from a solved TokaMaker equilibrium.

    The only function in ``bouquet.kinrecon.current`` that touches
    OpenFUSIONToolkit, and only through the methods of the equilibrium object
    it is given (no import).  It calls ``gs.get_fsa`` (surface averages traced in
    one pass: ``dV/dPsi``, ``<Bp^2> = <|grad psi|^2/R^2>``, ``<1/R^2>``,
    ``<1/R>``, ``<R>``, ``F``, ``R_min``, ``R_max``) and optionally
    ``gs.sauter_fc`` (``f_t = 1 - f_c``), then
    :meth:`FluxSurfaceGeometry.from_psi_coordinates`.

    ``R0`` (reference major radius) must be given; ``B0`` defaults to
    ``|F0| / R0`` with TokaMaker's vacuum ``F0``.  ``<B^2> = <B_p^2> + F^2
    <1/R^2>`` (exact, F a flux function).  ``r_geo = (R_max + R_min)/2`` and
    ``eps = (R_max - R_min)/(R_max + R_min)`` per surface.

    Self-check stored in ``meta``: TokaMaker's own q against the geometric
    ``q = F <1/R^2> dV/dpsi / (4 pi^2)`` (max relative difference).  It is a
    diagnostic only; nothing is fitted to it.  Edge: surfaces are sampled up
    to ``psi_n = 1 - psi_pad_edge`` and Phi_b is extrapolated to the
    boundary (PCHIP), which on a diverted boundary under-represents the
    logarithmic q divergence; the relative effect is recorded as
    ``meta['phi_b_extrapolated_fraction']``.

    Not verified against a live TokaMaker solve in the P0c work (see report).
    """
    # uniform in sqrt(psi_n), i.e. close to uniform in rho near the axis (uniform psi_n
    # samples would leave the axis region resolved only as npsi^-1/2 in rho)
    psi_n = np.linspace(np.sqrt(psi_pad_axis), np.sqrt(1.0 - psi_pad_edge), npsi) ** 2
    fsa = gs.get_fsa(psi=psi_n)
    if np.any(~np.isfinite(fsa["dV/dPsi"])):
        raise RuntimeError("TokaMaker flux-surface trace failed on some surfaces")
    F = np.abs(fsa["F"])
    g1 = fsa["<1/R^2>"]
    vps = np.abs(fsa["dV/dPsi"])
    bp2 = fsa["<Bp^2>"]
    extras = dict(avg_inv_r=fsa["<1/R>"], avg_r=fsa["<R>"], avg_b2=bp2 + F ** 2 * g1,
                  r_geo=0.5 * (fsa["R_min"] + fsa["R_max"]),
                  eps=(fsa["R_max"] - fsa["R_min"]) / (fsa["R_max"] + fsa["R_min"]))
    if with_trapped_fraction:
        _, fc, _, _, _ = gs.sauter_fc(psi=psi_n, return_eps=True)
        extras["trapped_fraction"] = 1.0 - np.asarray(fc)
    q_geo = F * g1 * vps / (4.0 * np.pi ** 2)
    q_tm = np.abs(np.asarray(fsa["q"]))
    if B0 is None:
        B0 = abs(float(fsa["F0"])) / R0
    geom = FluxSurfaceGeometry.from_psi_coordinates(
        psi_n, fsa["psi_axis"], fsa["psi_boundary"], vps, bp2, g1, F, B0=B0, R0=R0, nrho=nrho, extras=extras,
        meta=dict(source="tokamaker", q_geometric_vs_tokamaker_max_rel=float(np.max(np.abs(q_geo / q_tm - 1.0)))))
    # fraction of Phi_b that comes from the extrapolated edge strip
    dphi = F * g1 * vps / (2.0 * np.pi) * abs(fsa["psi_boundary"] - fsa["psi_axis"])
    phi_in = np.trapezoid(dphi, psi_n)
    geom.meta["phi_b_extrapolated_fraction"] = float(1.0 - phi_in / geom.phi_b)
    return geom
