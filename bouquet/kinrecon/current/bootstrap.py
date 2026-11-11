r"""Bootstrap current <j_BS . B> from the Redl et al. (2021) fits, evaluated in Python.

Source: A. Redl, C. Angioni, E. Belli, O. Sauter, ASDEX Upgrade Team and EUROfusion MST1 Team,
"A new set of analytical formulae for the computation of the bootstrap current and the
neoclassical conductivity in tokamaks", Phys. Plasmas 28, 022502 (2021): the L31 (eqs. 10-11),
L32 (eqs. 12-16) and alpha (eqs. 20-21) fits, used in Sauter's (1999) "jboot1" form

.. code-block:: text

    <j_BS . B> = -I(psi) p_e [ L31 (1/p_e) dp/dpsi + L32 (1/T_e) dT_e/dpsi
                               + L34 alpha (1 - R_pe)/R_pe (1/T_i) dT_i/dpsi ]           (S99 eq. 5)

with ``I(psi) = F = R B_phi``, ``R_pe = p_e / p``, ``p = p_e + p_i`` (thermal), L34 = L31 (Redl
section II.C).  The transcription follows, term for term, OFT's Python implementation
``OpenFUSIONToolkit.TokaMaker.bootstrap.redl_bootstrap`` (``formula_form='jboot1'``,
``use_legacy_L34=False``), which agrees with TokaMaker's in-solve Fortran Redl on the same
equilibria to 4-5e-4 of the peak (the solver-noise level; measured).
``tests/kinrecon/test_redl_bootstrap.py`` checks this module against that function to round-off
whenever OFT is importable.

Collisionality and Coulomb logarithms (the defaults reproduce the earlier prototype of this
package; each is a named option of :func:`redl_jdotb`):

* ``ln Lambda_e``: NRL form ``23.5 - ln(sqrt(n_e[cm^-3]) T_e^-5/4) - sqrt(1e-5 + (ln T_e - 2)^2/16)``,
  floored at 10 (OFT ``calculate_ln_lambda``, ``electron_lnLambda_model='NRL'``);
* ``ln Lambda_ii``: ``30 - ln(Z^3 sqrt(n_i) / T_i^1.5)`` with ``Z = max(n_e/n_i, 1)``, floored at
  10 (OFT ``ion_lnLambda_model='Zavg'``);
* ``nu_e* = 6.921e-18 |q| R n_e Z_eff ln Lambda_e / (T_e^2 eps^1.5)`` (Sauter 1999 eq. 18b; the
  definition Redl 2021 keeps);
* ``nu_i* = 4.90e-18 |q| R n_i Z_i^4 ln Lambda_ii / (T_i^2 eps^1.5)`` with
  ``Z_i^4 = (n_e / n_i) Z_eff`` (``ion_collisionality="spike"``, named after that prototype:
  ``Z_ion = ((n_e/n_i) Z_eff)^(1/4)``); ``"zeff"`` uses ``Z_eff^4`` and ``"unity"`` 1, as OFT's
  ``ion_collisionality_model`` options.  Sauter 1999 eq. 18c defines nu_i* with Z^4 of the main
  ion; the choice matters only through alpha.

``R`` is the surface major radius ``(R_max + R_min)/2`` and ``eps = (R_max - R_min)/(R_max +
R_min)`` (as OFT and IMAS.jl).  Units: SI, temperatures in eV, psi per radian.  Sign convention:
the "positive frame" of :mod:`.diffusion` -- pass ``|F|`` and derivatives with respect to a psi
that INCREASES outward (``dpsi > 0``); a co-current bootstrap then comes out positive.
"""

import numpy as np

EC = 1.602176634e-19

__all__ = ["coulomb_logs", "collisionalities", "redl_coefficients", "redl_jdotb", "ION_COLLISIONALITY"]

#: Ion-collisionality charge models (module docstring).
ION_COLLISIONALITY = ("spike", "zeff", "unity")


def coulomb_logs(te, ti, ne, ni):
    """``(ln Lambda_e, ln Lambda_ii)``: NRL electron form and the ``Zavg`` ion form, both floored at
    10 (OFT ``calculate_ln_lambda(..., 'NRL', 'Zavg')``; NRL Plasma Formulary, recalled)."""
    te, ti, ne, ni = (np.asarray(x, dtype=float) for x in (te, ti, ne, ni))
    ln_e = 23.5 - np.log(np.sqrt(ne / 1e6) * te ** (-1.25)) - np.sqrt(1e-5 + (np.log(te) - 2.0) ** 2 / 16.0)
    z = np.clip(ne / ni, 1.0, None)
    ln_ii = 30.0 - np.log(z ** 3 * np.sqrt(ni) / ti ** 1.5)
    return np.maximum(ln_e, 10.0), np.maximum(ln_ii, 10.0)


def collisionalities(q, R, eps, te, ti, ne, ni, zeff, ln_e, ln_ii, ion_collisionality="spike"):
    """``(nu_e*, nu_i*)`` (module docstring)."""
    if ion_collisionality not in ION_COLLISIONALITY:
        raise ValueError(f"ion_collisionality must be one of {ION_COLLISIONALITY}")
    q, R, eps, te, ti, ne, ni, zeff = (np.asarray(x, dtype=float) for x in (q, R, eps, te, ti, ne, ni, zeff))
    nu_e = 6.921e-18 * np.abs(q) * R * ne * zeff * ln_e / (te ** 2 * eps ** 1.5)
    if ion_collisionality == "spike":
        z4 = (ne / ni) * zeff
    elif ion_collisionality == "zeff":
        z4 = zeff ** 4
    else:
        z4 = np.ones_like(zeff)
    nu_i = 4.90e-18 * np.abs(q) * R * ni * z4 * ln_ii / (ti ** 2 * eps ** 1.5)
    return nu_e, nu_i


def redl_coefficients(ft, zeff, nu_e, nu_i):
    """Redl 2021 ``L31``, ``L32``, ``L34 (= L31)`` and ``alpha`` (eqs. 10-16, 20-21)."""
    fT, Zeff, nue, nui = (np.asarray(x, dtype=float) for x in (ft, zeff, nu_e, nu_i))
    d1 = (0.67 * (1 - 0.7 * fT) * np.sqrt(nue)) / (0.56 + 0.44 * Zeff)
    d2 = ((0.52 + 0.086 * np.sqrt(nue)) * (1 + 0.87 * fT) * nue) / (1 + 1.13 * np.sqrt(np.maximum(Zeff - 1, 0.0)))
    X31 = fT / (1 + d1 + d2)
    dZ = Zeff ** 1.2 - 0.71
    L31 = (1 + 0.15 / dZ) * X31 - (0.22 / dZ) * X31 ** 2 + (0.01 / dZ) * X31 ** 3 + (0.06 / dZ) * X31 ** 4
    dee_2 = (0.23 * (1 - 0.96 * fT) * np.sqrt(nue)) / np.sqrt(Zeff)
    dee_3 = ((0.13 * (1 - 0.38 * fT) * nue / Zeff ** 2)
             * (np.sqrt(1 + 2 * np.sqrt(np.maximum(Zeff - 1, 0.0)))
                + fT ** 2 * np.sqrt((0.075 + 0.25 * (Zeff - 1) ** 2) * nue)))
    X32e = fT / (1 + dee_2 + dee_3)
    F32_ee = ((0.1 + 0.6 * Zeff) / (Zeff * (0.77 + 0.63 * (1 + np.maximum(Zeff - 1, 0) ** 1.1)))
              * (X32e - X32e ** 4)
              + 0.7 / (1 + 0.2 * Zeff) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4))
              + 1.3 / (1 + 0.5 * Zeff) * X32e ** 4)
    dei_2 = (0.87 * (1 + 0.39 * fT) * np.sqrt(nue)) / (1 + 2.95 * (Zeff - 1) ** 2)
    dei_3 = 1.53 * (1 - 0.37 * fT) * nue * (2 + 0.375 * (Zeff - 1))
    X32i = fT / (1 + dei_2 + dei_3)
    F32_ei = (-(0.4 + 1.93 * Zeff) / (Zeff * (0.8 + 0.6 * Zeff)) * (X32i - X32i ** 4)
              + 5.5 / (1.5 + 2 * Zeff) * (X32i ** 2 - X32i ** 4 - 0.8 * (X32i ** 3 - X32i ** 4))
              - 1.3 / (1 + 0.5 * Zeff) * X32i ** 4)
    L32 = F32_ee + F32_ei
    alpha0 = (-(0.62 + 0.055 * (Zeff - 1)) / (0.53 + 0.17 * (Zeff - 1)) * (1 - fT)
              / (1 - (0.31 - 0.065 * (Zeff - 1)) * fT - 0.25 * fT ** 2))
    alpha = (((alpha0 + 0.7 * Zeff * np.sqrt(fT) * np.sqrt(nui)) / (1 + 0.18 * np.sqrt(nui))
              - 0.002 * nui ** 2 * fT ** 6) / (1 + 0.004 * nui ** 2 * fT ** 6))
    return dict(L31=L31, L32=L32, L34=L31, alpha=alpha, alpha0=alpha0)


def redl_jdotb(psi_n, te, ti, ne, ni, zeff, q, R, eps, ft, fpol, dpsi, ion_collisionality="spike",
               gradient="numpy_second_order"):
    r"""<j_BS . B> [A T / m^2] on the given psi_N samples (module docstring).

    Parameters
    ----------
    psi_n : array (n,)
        Normalized poloidal flux of the samples (strictly increasing).
    te, ti : array (n,)  [eV];  ne, ni : array (n,)  [m^-3];  zeff : array (n,)
        Thermal kinetic profiles on the samples.
    q, R, eps, ft, fpol : array (n,)
        Safety factor, surface major radius [m], inverse aspect ratio, trapped fraction,
        ``|F|`` [T m].
    dpsi : float
        ``|psi_boundary - psi_axis|`` [Wb/rad]; derivatives are ``d/dpsi = d/dpsi_N / dpsi``.
    gradient : str
        ``"numpy_second_order"``: ``numpy.gradient(y, psi_n, edge_order=2)`` (the prototype's choice).

    Returns ``(jdotb, parts)`` with ``parts`` holding ``L31``, ``L32``, ``alpha``, ``nu_e_star``,
    ``nu_i_star``, ``ln_lambda_e``, ``ln_lambda_ii`` and the settings used.
    """
    if gradient != "numpy_second_order":
        raise ValueError("only gradient='numpy_second_order' is implemented")
    x = np.asarray(psi_n, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("psi_n must increase strictly")
    te, ti, ne, ni, zeff, q, R, eps, ft, F = (np.asarray(v, dtype=float)
                                              for v in (te, ti, ne, ni, zeff, q, R, eps, ft, fpol))
    if not (np.isfinite(dpsi) and dpsi > 0):
        raise ValueError("dpsi must be |psi_b - psi_a| > 0")

    def d(y):
        return np.gradient(y, x, edge_order=2) / float(dpsi)

    ln_e, ln_ii = coulomb_logs(te, ti, ne, ni)
    nu_e, nu_i = collisionalities(q, R, eps, te, ti, ne, ni, zeff, ln_e, ln_ii, ion_collisionality)
    c = redl_coefficients(ft, zeff, nu_e, nu_i)
    pe, pi = EC * ne * te, EC * ni * ti
    rpe = pe / (pe + pi)
    dte, dti, dne, dni = d(te), d(ti), d(ne), d(ni)
    dp = (ne * dte + te * dne + ni * dti + ti * dni) * EC
    jb = -np.abs(F) * pe * (c["L31"] * dp / pe + c["L32"] * dte / te
                            + c["L34"] * c["alpha"] * (1 - rpe) / rpe * dti / ti)
    parts = dict(c, nu_e_star=nu_e, nu_i_star=nu_i, ln_lambda_e=ln_e, ln_lambda_ii=ln_ii,
                 settings=dict(model="redl2021", form="jboot1 (Sauter 1999 eq. 5), L34 = L31",
                               ln_lambda_e="NRL, floor 10", ln_lambda_ii="Zavg, floor 10",
                               ion_collisionality=ion_collisionality, gradient=gradient,
                               frame="positive (|F|, psi increasing outward)"))
    return jb, parts
