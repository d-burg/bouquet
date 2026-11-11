r"""Electron shielding of a fast-ion current, with the trapped-electron correction.

A fast-ion parallel current ``J_fast`` (:func:`.slowing_down.classical_moments`, eq. S11) drags
the electrons, whose return current cancels part of it.  The net driven current is

.. code-block:: text

    J_CD = J_fast F,      F = 1 - (Z_b / Z_eff) (1 - G)                                    (N1)

(Ohkawa's shielding factor with Start and Cordey's trapped-electron correction G; Start and
Cordey, Phys. Fluids 23, 1477 (1980); recalled).  Trapped electrons cannot carry the return
current, so G grows from 0 (no trapping) to 1 (all electrons trapped, no shielding).

G models (``model`` argument):

``"redl_l31"`` (default)
    ``G = L31(X = f_t, Z_eff)`` with the collisionless Redl et al. fit
    ``L31 = (1 + 0.15/d) X - (0.22/d) X^2 + (0.01/d) X^3 + (0.06/d) X^4``, ``d = Z^1.2 - 0.71``
    (Redl et al., Phys. Plasmas 28, 022502 (2021), eqs. 10-11; coefficients checked against
    the independent implementation in OFT ``TokaMaker/bootstrap.py``).  The identification
    ``G = L31`` (the trapped-electron correction to beam-driven current equals the bootstrap
    coefficient L31 in general geometry, in the collisionless limit) is Lin-Liu and Hinton,
    Phys. Plasmas 4, 4179 (1997) (recalled; it follows from the adjoint/reciprocity property
    of the electron response, cf. Hirshman, Phys. Fluids 31, 3150 (1988)).  Needs ``ft``.
``"sauter_l31"``
    Same identification with the Sauter, Angioni and Lin-Liu, Phys. Plasmas 6, 2834 (1999),
    eq. 14 collisionless fit ``L31 = (1 + 1.4/(Z+1)) X - (1.9/(Z+1)) X^2 + (0.3/(Z+1)) X^3
    + (0.2/(Z+1)) X^4`` (recalled coefficients).  Needs ``ft``.
``"start_cordey_fit"``
    The large-aspect-ratio fit used by FUSE ``ActorSimpleNB``:
    ``G = (1.55 + 0.85/Z) sqrt(eps) - (0.20 + 1.55/Z) eps`` (commonly attributed to Start and
    Cordey 1980 / Wesson, *Tokamaks*; attribution not verified).  Needs ``eps``.  At small eps
    it agrees with ``"sauter_l31"`` (f_t ~ 1.46 sqrt(eps)) to <10 % for Z = 1-2 (tested).
``"none"``
    ``G = 0`` (cylindrical Ohkawa factor).

Collisional detrapping (finite nu_e*) reduces the effective trapped fraction; it is not
included (the reciprocity is established for the collisionless limit only), which
overestimates G, and hence J_CD, in the edge.  Recorded as a limitation.
"""

import numpy as np

__all__ = ["trapped_electron_correction", "shielding_factor", "SHIELDING_MODELS"]

SHIELDING_MODELS = ("redl_l31", "sauter_l31", "start_cordey_fit", "none")


def _l31_redl(x, z):
    d = z ** 1.2 - 0.71
    return (1 + 0.15 / d) * x - (0.22 / d) * x ** 2 + (0.01 / d) * x ** 3 + (0.06 / d) * x ** 4


def _l31_sauter(x, z):
    zp = z + 1.0
    return (1 + 1.4 / zp) * x - (1.9 / zp) * x ** 2 + (0.3 / zp) * x ** 3 + (0.2 / zp) * x ** 4


def trapped_electron_correction(zeff, model="redl_l31", ft=None, eps=None):
    """G of eq. N1 for the chosen ``model`` (module docstring)."""
    z = np.asarray(zeff, float)
    if np.any(z < 1.0 - 1e-9):
        raise ValueError("Z_eff must be >= 1")
    if model in ("redl_l31", "sauter_l31"):
        if ft is None:
            raise ValueError(f"model {model!r} needs the trapped fraction ft")
        x = np.asarray(ft, float)
        if np.any((x < 0) | (x > 1)):
            raise ValueError("ft must lie in [0, 1]")
        return _l31_redl(x, z) if model == "redl_l31" else _l31_sauter(x, z)
    if model == "start_cordey_fit":
        if eps is None:
            raise ValueError("model 'start_cordey_fit' needs eps")
        e = np.asarray(eps, float)
        return (1.55 + 0.85 / z) * np.sqrt(np.maximum(e, 0.0)) - (0.20 + 1.55 / z) * e
    if model == "none":
        return np.zeros_like(z)
    raise ValueError(f"shielding model must be one of {SHIELDING_MODELS}")


def shielding_factor(zeff, Z_b, model="redl_l31", ft=None, eps=None):
    """F = 1 - (Z_b/Z_eff)(1 - G) (eq. N1)."""
    z = np.asarray(zeff, float)
    G = trapped_electron_correction(z, model, ft=ft, eps=eps)
    return 1.0 - (Z_b / z) * (1.0 - G)
