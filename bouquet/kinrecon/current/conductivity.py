"""Parallel electrical conductivity: Spitzer and neoclassical (Redl 2021 / Sauter 1999).

The neoclassical Ohm's law closes the current-diffusion equation
(:mod:`.diffusion`).  In flux-surface-averaged form (Sauter, Angioni and
Lin-Liu 1999, eq. 5)

.. code-block:: text

    <j . B> = sigma_neo <E . B> + <j_NI . B>

where ``<j_NI . B>`` collects every non-inductive current (bootstrap, beam,
RF).  This module provides ``sigma_neo`` as the Spitzer conductivity times a
trapped-particle correction ``F33``:

.. code-block:: text

    sigma_Spitzer = 1.9012e4 Te^(3/2) / (Z N(Z) ln(Lambda_e))  [1/(Ohm m)]   (S99 18a)
    N(Z)          = 0.58 + 0.74 / (0.76 + Z)                                (S99 18a)
    nu_e*         = 6.921e-18 q R n_e Z ln(Lambda_e) / (Te^2 eps^(3/2))      (S99 18b)
    ln(Lambda_e)  = 31.3 - ln(sqrt(n_e) / Te)                               (S99 18d)
    sigma_neo     = sigma_Spitzer F33(X = f_t,eff^33)                       (S99 13a)

with ``Te`` in eV, ``n_e`` in m^-3, ``Z = Z_eff``, ``q`` the safety factor,
``R`` the flux-surface major radius [m], ``eps`` the local inverse aspect
ratio and ``f_t`` the trapped-particle fraction (S99 eq. 12; see
:func:`trapped_fraction`).

Two fits of ``F33`` and of the effective trapped fraction are offered:

``"redl2021"`` (default) -- A. Redl, C. Angioni, E. Belli, O. Sauter, ASDEX
Upgrade Team and EUROfusion MST1 Team, "A new set of analytical formulae for
the computation of the bootstrap current and the neoclassical conductivity in
tokamaks", Phys. Plasmas 28, 022502 (2021), refit to NEO:

.. code-block:: text

    f_t,eff^33 = f_t / (1 + 0.25 (1 - 0.7 f_t) sqrt(nu_e*) (1 + 0.45 sqrt(Z - 1))
                          + 0.61 (1 - 0.41 f_t) nu_e* / sqrt(Z))
    F33(X)     = 1 - (1 + 0.21/Z) X + (0.54/Z) X^2 - (0.33/Z) X^3

``"sauter1999"`` -- O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6,
2834 (1999), eqs. 13a-b:

.. code-block:: text

    f_t,eff^33 = f_t / (1 + (0.55 - 0.1 f_t) sqrt(nu_e*) + 0.45 (1 - f_t) nu_e* / Z^(3/2))
    F33(X)     = 1 - (1 + 0.36/Z) X + (0.59/Z) X^2 - (0.23/Z) X^3

Verification status (details in the P0c report): the Sauter-1999 formulas
(13a-b, 18a-e) were checked term by term against the paper.  The Redl-2021
primary text was not accessible; its ``F33``/``f_t,eff^33`` coefficients are
checked numerically against the authors' own published fit values
(``sigma_new/sigma_Spitzer`` columns of the Zenodo dataset
doi:10.5281/zenodo.4110606, Z_eff = 1.8, three trapped fractions, 18 points;
see ``tests/kinrecon/test_current_conductivity.py``), and agree with the
independent implementations in IMAS.jl (``neo_conductivity``) and OFT's
``TokaMaker.bootstrap``.

Collisionality used with the Redl fits (settled from the primary text): Redl
et al. 2021 keep Sauter's definition -- their section II.B states that the
collisionality varies "according to Eqs. (18b) and (18c) in Ref. 4", Ref. 4
being Sauter, Angioni and Lin-Liu, Phys. Plasmas 6, 2834 (1999) -- and S99
eq. (18b) contains ``Z`` (= Z_eff): ``nu_e* = 6.921e-18 q R n_e Z ln(Lambda_e) /
(Te^2 eps^1.5)``; S99 section II also writes it as ``0.012 n_e20 Z_eff q R /
(eps^1.5 T_ekeV^2)``.  So :func:`nu_e_star` includes Z_eff for both models, as
IMAS.jl ``nuestar`` does.  Saxena et al., arXiv:2507.05166, appendix A
(eq. A4a), was found in an earlier reading to restate nu_e* without Z_eff
(not re-read for this note); such a restatement disagrees with
both primary texts and is not followed.  (The authors' Zenodo dataset cannot
decide this: it tabulates the fits against nu_e* as an input coordinate.)

Units are SI except temperatures, which are in eV as in the source papers.
"""

import numpy as np

__all__ = [
    "coulomb_log_e",
    "spitzer_conductivity",
    "nu_e_star",
    "f33_effective_trapped_fraction",
    "f33",
    "neoclassical_conductivity",
    "trapped_fraction",
]

_MODELS = ("redl2021", "sauter1999")


def coulomb_log_e(ne, te_ev, model="sauter"):
    r"""Electron Coulomb logarithm ln(Lambda_e).

    ``model="sauter"`` (default): ``31.3 - ln(sqrt(n_e)/Te)`` (Sauter 1999
    eq. 18d; the definition Redl 2021 keeps), ``n_e`` in m^-3, ``Te`` in eV.

    ``model="nrl"``: ``23.5 - ln(sqrt(n_e[cm^-3]) Te^(-5/4))
    - sqrt(1e-5 + (ln Te - 2)^2 / 16)``, the NRL Plasma Formulary
    electron-electron form, as used by IMAS.jl ``lnLambda_e`` and OFT
    ``calculate_ln_lambda(..., electron_lnLambda_model='NRL')`` (formula
    recalled from the Formulary; checked against those two codes only).

    No floor is applied (OFT clamps at 10; a floor is a modeling choice left
    to the caller).
    """
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te_ev, dtype=float)
    if model == "sauter":
        return 31.3 - np.log(np.sqrt(ne) / te)
    if model == "nrl":
        return 23.5 - np.log(np.sqrt(ne / 1.0e6) * te ** (-1.25)) - np.sqrt(1.0e-5 + (np.log(te) - 2.0) ** 2 / 16.0)
    raise ValueError(f"unknown Coulomb-log model {model!r} (use 'sauter' or 'nrl')")


def spitzer_conductivity(ne, te_ev, zeff, ln_lambda=None):
    r"""Spitzer parallel conductivity [1/(Ohm m)] (Sauter 1999 eq. 18a).

    ``sigma = 1.9012e4 Te^1.5 / (Z N(Z) ln(Lambda_e))``,
    ``N(Z) = 0.58 + 0.74/(0.76 + Z)``; ``Te`` in eV.  ``ln_lambda`` defaults
    to :func:`coulomb_log_e` (``model="sauter"``).
    """
    te = np.asarray(te_ev, dtype=float)
    z = np.asarray(zeff, dtype=float)
    if ln_lambda is None:
        ln_lambda = coulomb_log_e(ne, te)
    n_z = 0.58 + 0.74 / (0.76 + z)
    return 1.9012e4 * te ** 1.5 / (z * n_z * np.asarray(ln_lambda, dtype=float))


def nu_e_star(q, R, ne, te_ev, zeff, eps, ln_lambda=None):
    r"""Electron collisionality nu_e* (Sauter 1999 eq. 18b; also the definition Redl 2021 uses).

    ``nu_e* = 6.921e-18 |q| R n_e Z ln(Lambda_e) / (Te^2 eps^1.5)``, ``Z`` =
    Z_eff (S99 eq. 18b, checked against the paper; Redl 2021 section II.B
    refers to S99 eqs. 18b-c, see the module docstring), with ``R``
    the flux-surface major radius [m] and ``eps`` the local inverse aspect
    ratio.  The papers leave the surface definitions of ``R`` and ``eps`` to
    the user; IMAS.jl and the OFT Python bootstrap both use
    ``R = (R_max + R_min)/2`` and ``eps = (R_max - R_min)/(R_max + R_min)`` of
    each surface, which is the recommended choice here.  ``|q|`` is used so
    the result does not depend on the helicity convention.
    """
    if ln_lambda is None:
        ln_lambda = coulomb_log_e(ne, te_ev)
    te = np.asarray(te_ev, dtype=float)
    return (6.921e-18 * np.abs(np.asarray(q, dtype=float)) * np.asarray(R, dtype=float)
            * np.asarray(ne, dtype=float) * np.asarray(zeff, dtype=float) * np.asarray(ln_lambda, dtype=float)
            / (te ** 2 * np.asarray(eps, dtype=float) ** 1.5))


def f33_effective_trapped_fraction(ft, nu_e, zeff, model="redl2021"):
    r"""Collisionality-reduced trapped fraction f_t,eff^33.

    ``"redl2021"``: ``f_t / (1 + 0.25(1 - 0.7 f_t) sqrt(nu)(1 + 0.45 sqrt(Z-1))
    + 0.61(1 - 0.41 f_t) nu / sqrt(Z))`` (Redl et al. 2021; coefficients
    verified against the authors' dataset, see module docstring).

    ``"sauter1999"``: ``f_t / (1 + (0.55 - 0.1 f_t) sqrt(nu) + 0.45(1 - f_t)
    nu / Z^1.5)`` (Sauter 1999 eq. 13b).
    """
    ft = np.asarray(ft, dtype=float)
    nu = np.asarray(nu_e, dtype=float)
    z = np.asarray(zeff, dtype=float)
    if model == "redl2021":
        den = (1.0 + 0.25 * (1.0 - 0.7 * ft) * np.sqrt(nu) * (1.0 + 0.45 * np.sqrt(np.maximum(z - 1.0, 0.0)))
               + 0.61 * (1.0 - 0.41 * ft) * nu / np.sqrt(z))
    elif model == "sauter1999":
        den = 1.0 + (0.55 - 0.1 * ft) * np.sqrt(nu) + 0.45 * (1.0 - ft) * nu / z ** 1.5
    else:
        raise ValueError(f"unknown conductivity model {model!r}; use one of {_MODELS}")
    return ft / den


def f33(x, zeff, model="redl2021"):
    r"""Neoclassical conductivity factor F33(X) = sigma_neo / sigma_Spitzer.

    ``"redl2021"``: ``1 - (1 + 0.21/Z) X + 0.54/Z X^2 - 0.33/Z X^3``;
    ``"sauter1999"``: ``1 - (1 + 0.36/Z) X + 0.59/Z X^2 - 0.23/Z X^3``
    (Sauter 1999 eq. 13a).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(zeff, dtype=float)
    if model == "redl2021":
        c1, c2, c3 = 0.21, 0.54, 0.33
    elif model == "sauter1999":
        c1, c2, c3 = 0.36, 0.59, 0.23
    else:
        raise ValueError(f"unknown conductivity model {model!r}; use one of {_MODELS}")
    return 1.0 - (1.0 + c1 / z) * x + (c2 / z) * x ** 2 - (c3 / z) * x ** 3


def neoclassical_conductivity(ne, te_ev, zeff, ft, q, R, eps, model="redl2021", ln_lambda_model="sauter",
                              nu_e=None):
    r"""Neoclassical parallel conductivity sigma_neo [1/(Ohm m)].

    ``sigma_neo = sigma_Spitzer(n_e, Te, Z) F33(f_t,eff^33(f_t, nu_e*, Z))``
    (Sauter 1999 eq. 13a with the chosen ``model``'s F33 and f_t,eff).

    Parameters
    ----------
    ne, te_ev, zeff
        Electron density [m^-3], electron temperature [eV], Z_eff.
    ft
        Trapped-particle fraction (Sauter 1999 eq. 12), e.g. ``1 - f_c`` from
        TokaMaker ``sauter_fc`` or :func:`trapped_fraction`.
    q, R, eps
        Safety factor, surface major radius [m], local inverse aspect ratio;
        used only for ``nu_e*`` (see :func:`nu_e_star`).  Ignored when
        ``nu_e`` is given.
    model
        ``"redl2021"`` (default) or ``"sauter1999"``.
    ln_lambda_model
        ``"sauter"`` (default, S99 18d) or ``"nrl"``; see :func:`coulomb_log_e`.
    nu_e
        Optional precomputed nu_e*.

    Returns
    -------
    sigma_neo : ndarray
    parts : dict
        ``sigma_spitzer``, ``F33``, ``ft_eff``, ``nu_e_star``, ``ln_lambda_e``.
    """
    ln_l = coulomb_log_e(ne, te_ev, model=ln_lambda_model)
    sig_sp = spitzer_conductivity(ne, te_ev, zeff, ln_lambda=ln_l)
    if nu_e is None:
        nu_e = nu_e_star(q, R, ne, te_ev, zeff, eps, ln_lambda=ln_l)
    x = f33_effective_trapped_fraction(ft, nu_e, zeff, model=model)
    fac = f33(x, zeff, model=model)
    return sig_sp * fac, dict(sigma_spitzer=sig_sp, F33=fac, ft_eff=x, nu_e_star=np.asarray(nu_e, dtype=float),
                              ln_lambda_e=ln_l)


def trapped_fraction(b_mod, weight, nlambda=2001):
    r"""Trapped-particle fraction of one flux surface (Sauter 1999 eq. 12).

    ``f_t = 1 - (3/4) <B^2> int_0^{1/B_max} lambda dlambda / <sqrt(1 - lambda B)>``

    Parameters
    ----------
    b_mod : array, shape (n,)
        |B| at points along the surface [T].
    weight : array, shape (n,)
        Flux-surface-average weights at those points, proportional to
        ``dl / B_p`` (normalized internally).
    nlambda : int
        Number of trapezoidal nodes in ``lambda B_max`` on [0, 1].

    The integrand of the lambda integral has an integrable square-root edge at
    ``lambda = 1/B_max``; the trapezoid on a uniform grid converges at
    O(nlambda^-1.5) there, adequate at the default resolution (relative error
    below 1e-4 for the tested circular cases).
    """
    b = np.asarray(b_mod, dtype=float)
    w = np.asarray(weight, dtype=float)
    w = w / w.sum()
    bmax = b.max()
    h = b / bmax
    lam = np.linspace(0.0, 1.0, nlambda)
    den = np.array([np.sum(w * np.sqrt(np.clip(1.0 - lv * h, 0.0, None))) for lv in lam])
    integ = np.trapezoid(lam / den, lam)
    return 1.0 - 0.75 * np.sum(w * h ** 2) * integ
