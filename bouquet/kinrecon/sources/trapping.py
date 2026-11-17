r"""Zero-orbit-width fast-ion trapping for the beam-driven current (an option, off by default).

Used by :class:`.beam.SimpleBeamModel` with ``trapping="zero_orbit_width"``.  The default
(``trapping="none"``) does not call this module and is unchanged: every birth carries its birth
pitch into the fast-ion current (S11 of :mod:`.slowing_down`), which counts fast ions born on
trapped orbits as passing (module docstring of :mod:`.beam`).

Equations (derived here for the zero-orbit-width limit)
-------------------------------------------------------
A fast ion born at a point where ``|B| = B_b`` with pitch ``xi0 = v_par / v`` (signed relative to
Ip) keeps its energy and its magnetic moment along its orbit between collisions.  In the
zero-orbit-width limit it stays on the flux surface of its birth point, where

.. code-block:: text

    xi(l)^2 = 1 - lambda B(l),        lambda = (1 - xi0^2) / B_b                              (T1)

(``l`` is the poloidal arc length on the surface).  The ion is TRAPPED (it bounces) when
``xi^2`` reaches zero somewhere on the surface:

.. code-block:: text

    trapped   <=>   lambda B_max >= 1,        B_max = max over the surface of |B|            (T2)

A passing ion transits the surface.  Its poloidal speed is ``v_pol = v |xi| B_p / B``, so the
time it spends on ``dl`` is ``dl B / (v |xi| B_p)`` and its transit-time-averaged pitch is

.. code-block:: text

    <xi>_t = sign(xi0) oint (B / B_p) dl / oint (B / (B_p |xi|)) dl                           (T3)

The driven-current option replaces the birth pitch in S11 by

.. code-block:: text

    xi_cd = 0          for a trapped birth (T2)
    xi_cd = <xi>_t     for a passing birth (T3)                                               (T4)

and leaves everything else of S11 (the slowing down and the cylindrical pitch-angle decay
``I_1``) as it is.  T1-T2 are the standard zero-orbit-width trapping condition (e.g. Helander
and Sigmar, *Collisional Transport in Magnetized Plasmas* (CUP 2002), ch. 7; recalled); a
bounce-averaged slowing-down model of the beam-driven current is Cordey, Nucl. Fusion 16, 499
(1976), and its use with trapped-electron shielding Start, Cordey and Jones, Plasma Phys. 22,
303 (1980) (both recalled, not checked); T3-T4 are a reduced form of that treatment written for
this module.

Analytic check used by the tests: for ``|B| = B_0 R_0 / R`` (vacuum toroidal field, the large
aspect-ratio limit) on a circular surface of inverse aspect ratio ``eps``, ions born at the
outboard midplane with an isotropic pitch distribution are trapped with the fraction

.. code-block:: text

    f_trapped = sqrt(1 - B_min / B_max) = sqrt(2 eps / (1 + eps))  ->  sqrt(2 eps)              (T5)

which goes to zero on the magnetic axis (``B_max / B_min -> 1``).

What T4 does NOT include (recorded as limitations by the beam model):

* pitch-angle scattering across the trapped-passing boundary during the slowing down: a passing
  ion scattered into the trapped region keeps carrying current in S11, and a trapped birth
  scattered out of it never does -- T4 brackets the bounce-averaged result, it does not replace
  it;
* finite orbit width (banana width, potato orbits near the axis, the shift of passing orbits),
  which is not small for beam ions in the core;
* first-orbit (prompt) loss, in particular of counter-injected ions: with co and counter
  injection together the net current is a small difference and is ill-conditioned without it
  (the model then records a flag; the loss itself is not modeled);
* the ``B`` weighting of ``<J.B>``: the 1-D current model equates the fast-ion parallel current
  with ``<J.B>/B0`` (module docstring of :mod:`.beam`); T3 is the time average of ``xi``,
  not of ``xi B``.

Numerics: each birth is classified on the NEAREST surface of the caller's table
(:class:`FluxSurfaceField`), with ``B_b`` evaluated at the birth point; the table's resolution
is the caller's and is recorded.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .conventions import array_sha256

__all__ = ["FluxSurfaceField", "classify_births", "isotropic_trapped_fraction"]


@dataclass
class FluxSurfaceField:
    """|B| on traced flux surfaces, for the trapping classification (module docstring).

    ``rho`` -- rho_tor_norm of each traced surface (strictly increasing); ``B`` -- per surface,
    ``|B|`` [T] at sample points along the surface; ``w`` -- per surface, the poloidal
    transit weight ``dl / |B_p|`` of each sample (any common factor per surface; T3 is a ratio);
    ``b_of_rz(R, Z)`` -- ``|B|`` [T] at points (1-D arrays) from the same equilibrium.
    """

    rho: np.ndarray
    B: Sequence[np.ndarray]
    w: Sequence[np.ndarray]
    b_of_rz: Callable

    def __post_init__(self):
        self.rho = np.asarray(self.rho, float)
        if self.rho.ndim != 1 or self.rho.size < 1 or np.any(np.diff(self.rho) <= 0):
            raise ValueError("FluxSurfaceField: rho must be a strictly increasing 1-D array")
        if len(self.B) != self.rho.size or len(self.w) != self.rho.size:
            raise ValueError("FluxSurfaceField: one B and one w array per surface")
        self.B = [np.asarray(b, float) for b in self.B]
        self.w = [np.asarray(x, float) for x in self.w]
        for b, x in zip(self.B, self.w):
            if b.shape != x.shape or b.size < 2 or np.any(~np.isfinite(b)) or np.any(b <= 0) \
                    or np.any(~np.isfinite(x)) or np.any(x < 0) or x.sum() <= 0:
                raise ValueError("FluxSurfaceField: B > 0 and w >= 0 (finite, same shape, not all zero) required")
        self.B_max = np.array([b.max() for b in self.B])
        self.B_min = np.array([b.min() for b in self.B])

    def record(self):
        return {"n_surfaces": int(self.rho.size), "rho_min": float(self.rho[0]), "rho_max": float(self.rho[-1]),
                "samples_per_surface_min": int(min(b.size for b in self.B)),
                "sha256": array_sha256(self.rho, *self.B, *self.w),
                "rule": "each birth classified on the nearest traced surface; B at the birth point from b_of_rz"}

    def nearest(self, rho):
        """Index of the nearest traced surface for each rho."""
        rho = np.asarray(rho, float)
        if self.rho.size == 1:
            return np.zeros(rho.shape, int)
        return np.searchsorted(0.5 * (self.rho[1:] + self.rho[:-1]), rho)


def classify_births(field, rho, R, Z, xi0, chunk=1 << 18):
    r"""Trapped mask and the current-carrying pitch ``xi_cd`` (T4) of births at ``(rho, R, Z)`` with pitch ``xi0``.

    Returns ``(trapped, xi_cd)`` (arrays of the births' shape).
    """
    rho, R, Z = (np.asarray(a, float).ravel() for a in (rho, R, Z))
    xi0 = np.asarray(xi0, float).ravel()
    if not (rho.size == R.size == Z.size == xi0.size):
        raise ValueError("classify_births: rho, R, Z and xi0 must have one value per birth")
    if np.any(np.abs(xi0) > 1.0 + 1e-12):
        raise ValueError("|xi0| must be <= 1")
    Bb = np.asarray(field.b_of_rz(R, Z), float).ravel()
    if Bb.size != rho.size or np.any(~np.isfinite(Bb)) or np.any(Bb <= 0):
        raise ValueError("classify_births: b_of_rz must give a finite |B| > 0 at every birth point")
    lam = np.clip(1.0 - xi0 ** 2, 0.0, None) / Bb                                  # T1
    idx = field.nearest(rho)
    trapped = lam * field.B_max[idx] >= 1.0                                          # T2
    xi_cd = np.zeros(rho.size)
    for i in np.unique(idx[~trapped]):
        sel = np.flatnonzero((idx == i) & ~trapped)
        Bs, ws = field.B[i], field.w[i]
        bw = Bs * ws
        for c in range(0, sel.size, max(1, chunk // max(1, Bs.size))):
            j = sel[c:c + max(1, chunk // max(1, Bs.size))]
            arg = 1.0 - lam[j, None] * Bs[None, :]                                    # xi(l)^2 > 0 (passing)
            xi_cd[j] = np.sign(xi0[j]) * bw.sum() / np.sum(bw[None, :] / np.sqrt(arg), axis=1)   # T3
    return trapped, xi_cd


def isotropic_trapped_fraction(B_birth, B_max):
    """Fraction of an isotropic pitch distribution born at ``B_birth`` that is trapped (T5 in general form)."""
    r = np.asarray(B_birth, float) / np.asarray(B_max, float)
    return np.sqrt(np.clip(1.0 - r, 0.0, None))
