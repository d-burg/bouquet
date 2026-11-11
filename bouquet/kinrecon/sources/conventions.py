r"""Units, frames and the ONE fast-ion pressure convention of :mod:`bouquet.kinrecon.sources`.

Everything a provider returns is a plain array on a stated radial grid.  This module fixes
what those arrays mean and holds the only conversion functions between frames.

Radial grid
-----------
``rho`` is the normalized toroidal-flux coordinate (IMAS ``rho_tor_norm``)

.. code-block:: text

    rho = sqrt(Phi / Phi_b)                                                            (C1)

the coordinate of :class:`bouquet.kinrecon.current.geometry.FluxSurfaceGeometry`.  When a
caller needs psi_N (the poloidal-flux coordinate of the solve), :func:`rho_to_psi_n` /
:func:`psi_n_to_rho` map through a tabulation supplied by the caller from its OWN
equilibrium (never inferred here).

Driven (non-inductive) current
------------------------------
The primary quantity is the flux-surface average ``<J_CD . B>`` [A T / m^2], the source term
of the parallel Ohm's law (Sauter, Angioni and Lin-Liu 1999, eq. 5; checked against the
paper for :mod:`..current.conductivity`) and what :class:`..current.diffusion.CurrentDiffusion`
takes as ``j_ni``.  It is signed RELATIVE TO THE PLASMA CURRENT: co-Ip drive is positive
(the "positive frame" of :mod:`..current.diffusion`).

Derived frames (all exact identities once the current is a pure parallel current,
``J_CD = sigma(psi) B``, which holds for a divergence-free current with no pressure-driven
Pfirsch-Schlueter part; a non-inductive source current is of that form by construction):

.. code-block:: text

    j_par        = <J.B> / B0                         IMAS j_parallel        [A/m^2]   (C2)
    <J_phi / R>  = <J.B> F <1/R^2> / <B^2>                                   [A/m^3]   (C3)
    j_R0         = R0 <J_phi / R>                     transport-code frame   [A/m^2]   (C4)
                   (ONETWO "<Jtor R0/R>": integrates to I with dV / (2 pi R0))
    j_tor        = <J_phi / R> / <1/R>                IMAS j_tor             [A/m^2]   (C5)
    <J_phi>      = <J.B> F <1/R> / <B^2>                                     [A/m^2]   (C6)
    I(rho)       = (1 / 2 pi) int_0^rho <J_phi / R> dV                       [A]       (C7)

(C3) and (C6) follow from ``J = sigma B`` with ``B_phi = F/R``: ``<J.B> = sigma <B^2>`` and
``J_phi = sigma F / R``.  (C7) is Ampere's law averaged over a surface (Lyons, QED.jl
``equations_readme.pdf`` eq. 28; the same relation is :mod:`..current.diffusion` eq. 5).
The IMAS definition of ``j_parallel`` as ``<J.B>/B0`` is the data-dictionary text
(``core_sources.source[:].profiles_1d[:].j_parallel``; checked against the DD
documentation strings reproduced in OMAS).

Fast-ion pressure: the one convention
-------------------------------------
With TRUE directional pressures ``P_par`` and ``P_perp`` of a gyrotropic fast-ion population,
the scalar that ``kinrecon`` puts into the isotropic Grad-Shafranov pressure (and hence P')
is the isotropic-equivalent pressure

.. code-block:: text

    p_fast = (P_par + 2 P_perp) / 3 = (2/3) W_fast,    W_fast = P_par / 2 + P_perp      (C8)

(``W_fast`` = kinetic-energy density; (C8) is one third of the trace of the pressure
tensor).  This is the convention of kinetic EFIT (``beam pressure = (2/3) W_beam``), of the
ONETWO statefile (``pressb = (2/3) wbeam``) and of the AutoONETWO ``PRESS_BEAM`` node
(measured: ``PRESS_BEAM = (2/3) WBEAM`` to 1e-5).

How each stored form maps onto (C8):

=================================  ===================================================
stored form                        p_fast
=================================  ===================================================
isotropic-equivalent scalar        as is
energy density W                   (2/3) W
true (P_par, P_perp)               (P_par + 2 P_perp) / 3
IMAS components (p_par, p_perp)    p_par + 2 p_perp   (each component = TRUE/3)
=================================  ===================================================

IMAS components: OMAS and IMAS.jl form the total pressure as
``p_thermal + pressure_fast_parallel + 2 pressure_fast_perpendicular``, so each stored
component holds ONE THIRD of the corresponding true pressure (an isotropic p_fast is stored
as p_fast/3 in both).  Every OMFIT export (ONETWO, RABBIT, FREYA) writes ``p/3``; the
TRANSP plasma-state export writes true pressures (report 26, B.2/B.4 of the project record).
The DD itself cannot tell the two apart, so readers must be TOLD which one a file uses
(:mod:`.imas` refuses to guess).

Where (C8) stops being right: an anisotropic population enters a force balance through
``P_perp`` and ``P_par`` separately (anisotropic Grad-Shafranov); (C8) is the standard
isotropic approximation, adequate for tangential NBI (``P_par / P_perp`` of order 1-2) and
poor for ICRF minority tails (``P_perp >> P_par``).  Providers return the true components
when they can compute them so a caller can judge.

Units: SI throughout (``[Pa] = [J/m^3]``, densities [m^-3], currents [A/m^2], fields [T],
energies at interfaces in eV).
"""

import hashlib

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = [
    "E_CHARGE", "M_PROTON", "M_ELECTRON", "EPS0", "M_AMU", "C_LIGHT",
    "RADIAL_COORDINATE", "CURRENT_FRAME", "PRESSURE_CONVENTION",
    "CURRENT_FRAMES", "PRESSURE_FORMS",
    "AmbiguousConventionError",
    "p_fast_from_true", "p_fast_from_imas_components", "p_fast_from_energy_density",
    "imas_components_from_true", "imas_components_from_isotropic",
    "jpar_from_jdotb", "jdotb_from_jpar", "jphi_over_r_from_jdotb", "jdotb_from_jphi_over_r",
    "toroidal_frames", "enclosed_current", "rho_to_psi_n", "psi_n_to_rho", "array_sha256",
]

# CODATA 2018 exact / recommended values (SI)
E_CHARGE = 1.602176634e-19       # C (exact)
M_PROTON = 1.67262192369e-27     # kg
M_ELECTRON = 9.1093837015e-31    # kg
M_AMU = 1.66053906660e-27        # kg (unified atomic mass unit)
EPS0 = 8.8541878128e-12          # F/m
C_LIGHT = 299792458.0            # m/s (exact)

RADIAL_COORDINATE = "rho_tor_norm"
CURRENT_FRAME = "jdotb"          # <J_CD . B> [A T/m^2], signed relative to Ip
PRESSURE_CONVENTION = "isotropic_equivalent"   # (C8)

#: Current frames a caller may declare for input arrays (see (C2)-(C6)).
CURRENT_FRAMES = {
    "jdotb": "<J.B> [A T/m^2]",
    "jpar_b0": "<J.B>/B0 (IMAS j_parallel) [A/m^2]; needs B0",
    "j_r0": "R0 <J_phi/R> (ONETWO '<Jtor R0/R>') [A/m^2]; needs R0 and geometry (F, <1/R^2>, <B^2>)",
    "j_tor": "<J_phi/R>/<1/R> (IMAS j_tor) [A/m^2]; needs geometry (F, <1/R^2>, <1/R>, <B^2>)",
    "jphi_avg": "<J_phi> [A/m^2]; needs geometry (F, <1/R>, <B^2>)",
}

#: Pressure forms a caller may declare (see the table in the module docstring).
PRESSURE_FORMS = {
    "isotropic_equivalent": "p_fast = (P_par + 2 P_perp)/3 [Pa]",
    "energy_density": "W_fast [J/m^3]; p_fast = (2/3) W",
    "true_components": "(P_par, P_perp) true pressures [Pa]",
    "imas_components": "(p_par, p_perp) IMAS components, each = true/3 [Pa]",
}


class AmbiguousConventionError(ValueError):
    """An input whose unit, frame, sign or pressure convention was not declared (refused)."""


# ----------------------------------------------------------------------------- pressure (C8)
def p_fast_from_true(p_par, p_perp):
    """Isotropic-equivalent p_fast = (P_par + 2 P_perp)/3 from TRUE pressures (eq. C8)."""
    return (np.asarray(p_par, float) + 2.0 * np.asarray(p_perp, float)) / 3.0


def p_fast_from_imas_components(p_par_comp, p_perp_comp):
    """p_fast = p_par + 2 p_perp from IMAS one-third components (eq. C8 table; OMAS/IMAS.jl sum)."""
    return np.asarray(p_par_comp, float) + 2.0 * np.asarray(p_perp_comp, float)


def p_fast_from_energy_density(w):
    """p_fast = (2/3) W from the kinetic-energy density W (eq. C8)."""
    return (2.0 / 3.0) * np.asarray(w, float)


def imas_components_from_true(p_par, p_perp):
    """IMAS components (P_par/3, P_perp/3) from TRUE pressures (eq. C8 table)."""
    return np.asarray(p_par, float) / 3.0, np.asarray(p_perp, float) / 3.0


def imas_components_from_isotropic(p_fast):
    """IMAS components of an isotropic p_fast: (p_fast/3, p_fast/3) (eq. C8 table)."""
    p = np.asarray(p_fast, float) / 3.0
    return p, p.copy()


# ----------------------------------------------------------------------------- current frames
def jpar_from_jdotb(jdotb, B0):
    """IMAS ``j_parallel`` = <J.B>/B0 (eq. C2).  ``B0`` > 0 is the reference vacuum field magnitude."""
    B0 = float(B0)
    if not np.isfinite(B0) or B0 <= 0:
        raise ValueError("B0 must be a positive magnitude (signs are carried relative to Ip)")
    return np.asarray(jdotb, float) / B0


def jdotb_from_jpar(jpar, B0):
    """<J.B> = j_parallel B0 (eq. C2)."""
    B0 = float(B0)
    if not np.isfinite(B0) or B0 <= 0:
        raise ValueError("B0 must be a positive magnitude (signs are carried relative to Ip)")
    return np.asarray(jpar, float) * B0


def _geo_arrays(geometry, rho, names):
    """Profiles ``names`` of a FluxSurfaceGeometry-like object at ``rho``; refuses missing ones."""
    out = {}
    for n in names:
        val = getattr(geometry, n, None)
        if val is None:
            raise AmbiguousConventionError(f"frame conversion needs geometry field {n!r}, which is absent")
        out[n] = PchipInterpolator(geometry.rho, np.asarray(val, float), extrapolate=True)(rho)
    return out


def jphi_over_r_from_jdotb(jdotb, rho, geometry):
    """<J_phi/R> = <J.B> F <1/R^2>/<B^2> for a pure parallel current (eq. C3)."""
    g = _geo_arrays(geometry, rho, ("fpol", "avg_inv_r2", "avg_b2"))
    return np.asarray(jdotb, float) * g["fpol"] * g["avg_inv_r2"] / g["avg_b2"]


def jdotb_from_jphi_over_r(jphi_r, rho, geometry):
    """Inverse of :func:`jphi_over_r_from_jdotb` (eq. C3)."""
    g = _geo_arrays(geometry, rho, ("fpol", "avg_inv_r2", "avg_b2"))
    return np.asarray(jphi_r, float) * g["avg_b2"] / (g["fpol"] * g["avg_inv_r2"])


def toroidal_frames(jdotb, rho, geometry):
    """All toroidal-current equivalents of ``<J.B>`` (eqs. C3-C7) as a dict of arrays on ``rho``.

    Keys: ``jphi_over_r`` (C3) [A/m^3], ``j_r0`` (C4), ``j_tor`` (C5, needs ``avg_inv_r``),
    ``jphi_avg`` (C6, needs ``avg_inv_r``), ``current_enclosed`` (C7) [A].  ``geometry`` is a
    :class:`..current.geometry.FluxSurfaceGeometry` (or any object with the same fields)
    with ``avg_b2`` present.
    """
    rho = np.asarray(rho, float)
    jr = jphi_over_r_from_jdotb(jdotb, rho, geometry)
    out = {"jphi_over_r": jr, "j_r0": float(geometry.R0) * jr}
    if getattr(geometry, "avg_inv_r", None) is not None:
        g = _geo_arrays(geometry, rho, ("fpol", "avg_inv_r", "avg_b2"))
        out["j_tor"] = jr / g["avg_inv_r"]
        out["jphi_avg"] = np.asarray(jdotb, float) * g["fpol"] * g["avg_inv_r"] / g["avg_b2"]
    dv = PchipInterpolator(geometry.rho, geometry.dvol_drho, extrapolate=True)(rho)
    out["current_enclosed"] = enclosed_current(rho, jr, dv)
    return out


def enclosed_current(rho, jphi_over_r, dvol_drho):
    """I(rho) = (1/2pi) int_0^rho <J_phi/R> V' drho (eq. C7), trapezoidal, I(rho[0]) = 0."""
    rho = np.asarray(rho, float)
    f = np.asarray(jphi_over_r, float) * np.asarray(dvol_drho, float) / (2.0 * np.pi)
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rho))])


# ----------------------------------------------------------------------------- radial coordinate (C1)
def _check_table(psi_n_table, rho_table):
    p = np.asarray(psi_n_table, float)
    r = np.asarray(rho_table, float)
    if p.shape != r.shape or p.ndim != 1 or p.size < 4:
        raise ValueError("psi_n / rho tables must be 1-D, same length, >= 4 points")
    if np.any(np.diff(p) <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("psi_n and rho tables must both increase strictly (one surface per entry)")
    return p, r


def _snap(x, table):
    """Snap values within 1e-12 of the table ends onto them (round-off); outside stays NaN later."""
    x = np.array(x, dtype=float)
    for end in (table[0], table[-1]):
        x = np.where(np.abs(x - end) <= 1e-12 * max(1.0, abs(end)), end, x)
    return x


def rho_to_psi_n(rho, psi_n_table, rho_table):
    """psi_N at ``rho`` through the caller's own equilibrium tabulation (monotone PCHIP; NaN outside)."""
    p, r = _check_table(psi_n_table, rho_table)
    return PchipInterpolator(r, p, extrapolate=False)(_snap(rho, r))


def psi_n_to_rho(psi_n, psi_n_table, rho_table):
    """rho_tor_norm at ``psi_n`` through the caller's own equilibrium tabulation (monotone PCHIP; NaN outside)."""
    p, r = _check_table(psi_n_table, rho_table)
    return PchipInterpolator(p, r, extrapolate=False)(_snap(psi_n, p))


def array_sha256(*arrays):
    """sha256 of the float64 bytes of the given arrays (provenance of numeric inputs)."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()
