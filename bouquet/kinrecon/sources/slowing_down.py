r"""Classical fast-ion slowing down: tau_s, critical energy, distribution moments, heating split.

Steady-state, collisional (Fokker-Planck) slowing down of a mono-energetic source of fast
ions (mass ``m_b = A_b m_u``, charge ``Z_b e``, birth speed ``v0``, birth pitch ``xi0 = v_par/v``)
in a Maxwellian plasma with ``v_thi << v << v_the``.  No orbit width, no charge-exchange
loss, no anomalous transport, no beam-beam collisions (see LIMITATIONS).  Each energy
component of a beam is a separate source born at its OWN energy (``E_k = E_full / k``).

Equations (derived for this module from the cited collision operators)
----------------------------------------------------------------------
Drag on electrons (``v << v_the``) and energy loss to ions (``v >> v_thi``) give

.. code-block:: text

    dE/dt = -(2E/tau_s) [1 + (v_c/v)^3]                                                (S1)
    tau_s = 3 (2 pi)^(3/2) eps0^2 m_b T_e^(3/2) / (Z_b^2 e^4 n_e m_e^(1/2) lnL_e)        (S2)
    v_c^3 = (3 sqrt(pi)/4) v_te^3 sum_j (m_e/m_j) (n_j Z_j^2 / n_e) (lnL_j / lnL_e),
            v_te = sqrt(2 T_e / m_e)                                                    (S3)
    E_c   = m_b v_c^2 / 2  (= 14.8 A_b T_e [sum_j n_j Z_j^2/(n_e A_j)]^(2/3) when lnL_j = lnL_e)  (S4)

(Stix, Plasma Phys. 14, 367 (1972); Spitzer's tau_s; recalled, equation numbers in Stix not
verified.  S2 reproduces the textbook ``6.27e8 A_b T_e[eV]^1.5 / (Z_b^2 n_e[cm^-3] lnL)`` s
and S4 the textbook 14.8 coefficient -- both checked numerically in the tests.)
Pitch-angle scattering on ions at the rate

.. code-block:: text

    nu_D = sum_j Z_b^2 Z_j^2 e^4 n_j lnL_j / (4 pi eps0^2 m_b^2 v^3) = R_D v_c^3 / (tau_s v^3),
    R_D  = sum_j n_j Z_j^2 lnL_j / ( sum_j n_j Z_j^2 lnL_j m_b/m_j )   (= Z_eff/[Z] of Gaffey)  (S5)

(deflection frequency of a fast test particle on slow field particles: Helander and Sigmar,
*Collisional Transport in Magnetized Plasmas* (CUP 2002), ch. 3; NRL Plasma Formulary
``nu_perp = 2 nu_D``; recalled).  The steady-state solution of the Fokker-Planck equation
with drag (S1), pitch-angle scattering (S5) and the source ``S delta(v - v0)
delta(xi - xi0)/(2 pi v^2)``, expanded in Legendre polynomials ``P_l(xi)``, is

.. code-block:: text

    f_l(v) = (2l+1) P_l(xi0) S tau_s / (4 pi (v^3 + v_c^3)) * [ v^3 (v0^3 + v_c^3) / (v0^3 (v^3 + v_c^3)) ]^(alpha_l),
    alpha_l = l (l+1) R_D / 6,      0 < v < v0                                          (S6)

(Gaffey, J. Plasma Phys. 16, 149 (1976); Cordey, Nucl. Fusion 16, 499 (1976); recalled.
FUSE/IMAS.jl ``ion_momentum_fraction`` has the l = 1 exponent ``A_i Z_eff / (3 A_b Z_b)``,
which equals ``alpha_1`` for a single ion mass; S5-S6 use the exact species sums.)
Moments, with ``x = v/v0``, ``w = v_c/v0 = sqrt(E_c/E0)``:

.. code-block:: text

    n_fast  = S tau_s N(w),          N(w) = int_0^1 x^2/(x^3+w^3) dx = (1/3) ln(1 + w^-3)   (S7)
    W_fast  = S E0 tau_s J(w),       J(w) = int_0^1 x^4/(x^3+w^3) dx                       (S8)
    J(w)    = 1/2 - w^2 [ (1/6) ln((1 - w + w^2)/(1 + w)^2)
                          + (1/sqrt3)(atan((2 - w)/(w sqrt3)) + pi/6) ]                    (S9)
    p_fast  = (2/3) W_fast                                         (isotropic equivalent, C8)
    P_par - P_perp = 2 P_2(xi0) S E0 tau_s K_2(w),
             K_2 = int_0^1 x^4/(x^3+w^3) g(x)^(alpha_2) dx,  g = x^3 (1+w^3)/(x^3+w^3)      (S10)
    P_par = p_fast + (2/3)(P_par - P_perp),   P_perp = p_fast - (1/3)(P_par - P_perp)
    J_fast  = Z_b e S tau_s v0 xi0 I_1(w),  I_1 = int_0^1 x^3/(x^3+w^3) g(x)^(alpha_1) dx  (S11)
    q_e     = S E0 G_e,  q_i = S E0 (1 - G_e),   G_e = 2 J(w)                              (S12)
    W_fast  = tau_s q_e / 2                     (local identity: S8 with S12)               (S13)

S9 is the closed form of S8 (the same expression as the Estrada et al., Phys. Plasmas 13,
112303 (2006) ``I_4`` integral used by IMAS.jl, derived independently here).  S12 is the
Sivukhin split (Sivukhin, Rev. Plasma Phys. 4 (1966); recalled) written as an integral over
the slowing-down trajectory; ``G_e`` integrates S1's electron fraction ``x^3/(x^3+w^3)``
with ``dE = 2 E0 x dx``.  S13 is exact for the classical model, independent of the energy
mix and of the deposition, and is the first test applied to any transport-code output.
Limits used by the tests: ``J -> 1/2`` (w -> 0, electron drag only) and ``J -> 1/(5 w^3)``
(w -> inf, ion drag only); ``I_1 -> 1`` for w -> 0 (momentum decays as exp(-t/tau_s)).
The absolute normalization of S11 contains NO numerical factor: ``J_fast`` is the parallel
current density of the fast ions in SI units.  S6 is the CYLINDRICAL solution: toroidal
trapping of the fast ions (which removes the bounce-averaged parallel current of trapped
fast ions) is not included, so S11 is an upper estimate of the fast-ion current.

Coulomb logarithms (``ln_lambda_e`` / ``ln_lambda_i`` settings)
------------------------------------------------------------------
* ``"nrl_ei"`` (default for electrons): ``lnL_e = 24 - ln(sqrt(n_e[cm^-3]) / T_e[eV])``,
  NRL Plasma Formulary ion-electron form for ``T_e > 10 Z^2 eV`` (recalled);
* ``"sauter"``: ``31.3 - ln(sqrt(n_e)/T_e)`` (:func:`..current.conductivity.coulomb_log_e`);
* ``"nrl_beam_ion"`` (default for ions): counter-streaming ions in warm electrons,
  ``lnL_j = 43 - ln[ Z_b Z_j (A_b + A_j) / (A_b A_j beta^2) sqrt(n_e[cm^-3]/T_e[eV]) ]``,
  ``beta = v0/c`` at the birth speed (NRL Plasma Formulary; recalled);
* ``"equal_to_electron"``: ``lnL_j = lnL_e`` (the convention behind the 14.8 coefficient of
  S4 and FUSE's default; FUSE's own comment estimates the resulting E_c too low by a factor
  ``(lnL_i/lnL_e)^(2/3) ~ 1.2``).

LIMITATIONS (also recorded in provenance by the models that call this module): no finite
orbit width (deposition-to-flux-surface mapping is the caller's), no prompt/first-orbit loss,
no fast-ion charge-exchange loss, no anomalous fast-ion transport (Alfven eigenmodes,
sawteeth, tearing modes), no beam-beam or beam-fast-ion collisions, no time history
(steady state at each slice; valid when the source changes slowly compared with
``tau_th = tau_s N(w)``), electron pitch-angle scattering neglected (``m_e/m_b`` small),
thermalization taken to v = 0 (energy below ~ T_i is a few percent of E0 for E0 >> T_i).

Fast-ion density lower limit: ``n_fast`` (S7) counts the slowing-down distribution down to
v = 0; no thermalization cutoff is applied, so ions that have already joined the thermal
population are counted as fast.  The share below 1.5 T_i is small for the full-energy
component (about 2 % for an 80 keV deuterium beam at T_i = 3-5 keV) but not for the E/3
component (8-17 % in the same conditions; independent review).  Stored energy and pressure are
much less affected (weighted by energy).  The linear electron drag (S1) assumes v << v_te;
at the edge (T_e of 100-300 eV) v_b/v_te reaches 0.3-0.5 for 80 keV deuterium and the drag is
overstated by roughly 10-20 % there (not corrected).
"""

from dataclasses import dataclass

import numpy as np

from ..current.conductivity import coulomb_log_e
from .conventions import C_LIGHT, E_CHARGE, EPS0, M_AMU, M_ELECTRON

__all__ = ["IonSpecies", "ln_lambda_electron", "ln_lambda_beam_ion", "slowing_down_time", "critical_velocity",
           "critical_energy", "deflection_ratio", "density_integral", "energy_integral", "pitch_integral",
           "classical_moments", "ELECTRON_LNL_MODELS", "ION_LNL_MODELS"]

ELECTRON_LNL_MODELS = ("nrl_ei", "sauter")
ION_LNL_MODELS = ("nrl_beam_ion", "equal_to_electron")


@dataclass(frozen=True)
class IonSpecies:
    """A thermal ion species: mass number ``A`` [u], charge ``Z``, density ``n`` [m^-3] (array on rho)."""

    name: str
    A: float
    Z: float
    n: np.ndarray

    def __post_init__(self):
        if not (self.A > 0 and self.Z > 0):
            raise ValueError(f"{self.name}: A and Z must be positive")
        n = np.asarray(self.n, float)
        if np.any(n < 0) or not np.all(np.isfinite(n)):
            raise ValueError(f"{self.name}: density must be finite and >= 0")
        object.__setattr__(self, "n", n)


def _pos(x, name):
    a = np.asarray(x, float)
    if np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return a


def ln_lambda_electron(ne, te_ev, model="nrl_ei"):
    """Electron Coulomb logarithm seen by a fast ion (module docstring; ``n_e`` m^-3, ``T_e`` eV)."""
    ne, te = _pos(ne, "n_e"), _pos(te_ev, "T_e")
    if model == "nrl_ei":
        return 24.0 - np.log(np.sqrt(ne * 1e-6) / te)
    if model == "sauter":
        return coulomb_log_e(ne, te, model="sauter")
    raise ValueError(f"ln_lambda_e model must be one of {ELECTRON_LNL_MODELS}")


def ln_lambda_beam_ion(ne, te_ev, A_b, Z_b, E0_ev, ion, model="nrl_beam_ion", ln_lambda_e=None):
    """Beam-ion Coulomb logarithm for target species ``ion`` (module docstring)."""
    if model == "equal_to_electron":
        if ln_lambda_e is None:
            raise ValueError("equal_to_electron needs ln_lambda_e")
        return np.asarray(ln_lambda_e, float)
    if model != "nrl_beam_ion":
        raise ValueError(f"ln_lambda_i model must be one of {ION_LNL_MODELS}")
    ne, te = _pos(ne, "n_e"), _pos(te_ev, "T_e")
    v0 = np.sqrt(2.0 * E0_ev * E_CHARGE / (A_b * M_AMU))
    beta2 = (v0 / C_LIGHT) ** 2
    arg = Z_b * ion.Z * (A_b + ion.A) / (A_b * ion.A * beta2) * np.sqrt(ne * 1e-6 / te)
    return 43.0 - np.log(arg)


def slowing_down_time(ne, te_ev, A_b, Z_b, ln_lambda_e):
    """Spitzer electron slowing-down time tau_s [s] (eq. S2)."""
    ne, te = _pos(ne, "n_e"), _pos(te_ev, "T_e")
    m_b = A_b * M_AMU
    te_j = te * E_CHARGE
    return (3.0 * (2.0 * np.pi) ** 1.5 * EPS0 ** 2 * m_b * te_j ** 1.5
            / (Z_b ** 2 * E_CHARGE ** 4 * ne * np.sqrt(M_ELECTRON) * np.asarray(ln_lambda_e, float)))


def critical_velocity(ne, te_ev, ions, ln_lambda_e, ln_lambda_i):
    """Critical speed v_c [m/s] (eq. S3); ``ln_lambda_i`` is a list (one array per ion species)."""
    ne, te = _pos(ne, "n_e"), _pos(te_ev, "T_e")
    vte3 = (2.0 * te * E_CHARGE / M_ELECTRON) ** 1.5
    s = sum((M_ELECTRON / (ion.A * M_AMU)) * ion.n * ion.Z ** 2 / ne * (np.asarray(li, float) / ln_lambda_e)
            for ion, li in zip(ions, ln_lambda_i))
    return (0.75 * np.sqrt(np.pi) * vte3 * s) ** (1.0 / 3.0)


def critical_energy(ne, te_ev, ions, A_b, ln_lambda_e, ln_lambda_i):
    """Critical energy E_c [eV] = m_b v_c^2 / 2 (eq. S4)."""
    vc = critical_velocity(ne, te_ev, ions, ln_lambda_e, ln_lambda_i)
    return 0.5 * A_b * M_AMU * vc ** 2 / E_CHARGE


def deflection_ratio(ions, A_b, ln_lambda_i):
    """R_D = sum n Z^2 lnL / sum (n Z^2 lnL m_b/m_j) (eq. S5; Gaffey's Z_eff/[Z])."""
    num = sum(ion.n * ion.Z ** 2 * np.asarray(li, float) for ion, li in zip(ions, ln_lambda_i))
    den = sum(ion.n * ion.Z ** 2 * np.asarray(li, float) * A_b / ion.A for ion, li in zip(ions, ln_lambda_i))
    return num / den


def density_integral(w):
    """N(w) = (1/3) ln(1 + w^-3) (eq. S7)."""
    w = _pos(w, "w")
    return np.log1p(w ** -3) / 3.0


def energy_integral(w):
    """J(w) = int_0^1 x^4/(x^3 + w^3) dx in closed form (eq. S9); series for w < 1e-3."""
    w = np.atleast_1d(_pos(w, "w"))
    small, large = w < 1e-3, w > 20.0
    mid = ~(small | large)
    out = np.empty_like(w)
    ws = w[mid]
    out[mid] = 0.5 - ws ** 2 * (np.log((1.0 - ws + ws ** 2) / (1.0 + ws) ** 2) / 6.0
                                 + (np.arctan((2.0 - ws) / (ws * np.sqrt(3.0))) + np.pi / 6.0) / np.sqrt(3.0))
    # w -> 0: J = 1/2 - w^3 int_0^1 x/(x^3+w^3) dx = 1/2 - w^2 (2 pi / (3 sqrt 3)) + w^3 + O(w^6)
    wsm = w[small]
    out[small] = 0.5 - wsm ** 2 * 2.0 * np.pi / (3.0 * np.sqrt(3.0)) + wsm ** 3
    # w -> inf: x^4/(x^3+w^3) = (x^4/w^3)(1 - x^3/w^3 + x^6/w^6 - ...)
    wl3 = w[large] ** -3
    out[large] = wl3 / 5.0 - wl3 ** 2 / 8.0 + wl3 ** 3 / 11.0
    return out


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def pitch_integral(w, alpha, power):
    """int_0^1 x^power/(x^3+w^3) [x^3 (1+w^3)/(x^3+w^3)]^alpha dx (S10 with power 4, S11 with power 3).

    Composite Gauss-Legendre (48 nodes per panel) on panels split at 0.25 w, w and 4 w,
    vectorized over arrays ``w`` and ``alpha`` of one shape; tested against the closed forms at
    ``alpha = 0`` and against adaptive quadrature at ``alpha > 0`` (relative error ~1e-10 for
    w >~ 0.01, worst measured 1.2e-7 at w = 1e-3).
    """
    w = np.atleast_1d(_pos(w, "w"))
    alpha = np.broadcast_to(np.asarray(alpha, float), w.shape)
    edges = np.stack([np.zeros_like(w), np.minimum(0.25 * w, 1.0), np.minimum(w, 1.0), np.minimum(4.0 * w, 1.0),
                      np.ones_like(w)], axis=-1)
    total = np.zeros_like(w)
    w3 = w ** 3
    for i in range(4):
        a, b = edges[..., i], edges[..., i + 1]
        half = 0.5 * (b - a)
        x = a[..., None] + half[..., None] * (_GL_X + 1.0)
        x3 = x ** 3
        g = x3 * (1.0 + w3[..., None]) / (x3 + w3[..., None])
        f = x ** power / (x3 + w3[..., None]) * g ** alpha[..., None]
        total += half * np.sum(f * _GL_W, axis=-1)
    return total


def classical_moments(S, E0_ev, ne, te_ev, ions, A_b, Z_b, xi0=None, ln_lambda_e="nrl_ei",
                      ln_lambda_i="nrl_beam_ion", p2_xi0=None):
    r"""Steady-state moments of the classical slowing-down distribution of ONE energy component.

    Parameters
    ----------
    S
        Birth rate density of this component [m^-3 s^-1] (array on rho).
    E0_ev
        Birth energy of this component [eV] (scalar: each component at its OWN energy).
    ne, te_ev
        Electron density [m^-3] and temperature [eV].
    ions
        List of :class:`IonSpecies` (thermal ions, densities on the same grid).
    A_b, Z_b
        Fast-ion mass number and charge.
    xi0
        Birth pitch ``v_par/v`` relative to the co-Ip direction (scalar or array); ``None`` skips
        the current and anisotropy moments.
    ln_lambda_e, ln_lambda_i
        Coulomb-logarithm models (module docstring).
    p2_xi0
        Birth-weighted mean of ``P_2(xi0)`` when births at one radius have a spread of pitches
        (the moments are linear in ``P_l(xi0)``, so ``<xi0>`` and ``<P_2(xi0)>`` are what enter
        S11 and S10); default ``P_2(xi0)``.

    Returns a dict: ``n_fast`` [m^-3] (S7), ``W`` [J/m^3] (S8), ``p_fast`` [Pa] (C8),
    ``q_e``, ``q_i`` [W/m^3] (S12), ``tau_s``, ``tau_th`` [s], ``E_c`` [eV], ``w``, and with
    ``xi0``: ``j_fast`` [A/m^2] (S11), ``p_par``, ``p_perp`` [Pa] (S10).
    """
    S = np.asarray(S, float)
    if np.any(S < 0) or not np.all(np.isfinite(S)):
        raise ValueError("S must be finite and >= 0")
    if not (E0_ev > 0):
        raise ValueError("E0 must be positive")
    lle = ln_lambda_electron(ne, te_ev, ln_lambda_e)
    lli = [ln_lambda_beam_ion(ne, te_ev, A_b, Z_b, E0_ev, ion, ln_lambda_i, lle) for ion in ions]
    tau_s = slowing_down_time(ne, te_ev, A_b, Z_b, lle)
    Ec = critical_energy(ne, te_ev, ions, A_b, lle, lli)
    w = np.sqrt(Ec / E0_ev)
    Nw, Jw = density_integral(w), energy_integral(w)
    E0_j = E0_ev * E_CHARGE
    out = {"tau_s": tau_s, "E_c": Ec, "w": w, "tau_th": tau_s * Nw,
           "n_fast": S * tau_s * Nw, "W": S * E0_j * tau_s * Jw}
    out["p_fast"] = (2.0 / 3.0) * out["W"]
    out["q_e"] = S * E0_j * 2.0 * Jw
    out["q_i"] = S * E0_j * (1.0 - 2.0 * Jw)
    if xi0 is not None:
        xi0 = np.broadcast_to(np.asarray(xi0, float), S.shape)
        if np.any(np.abs(xi0) > 1.0 + 1e-12):
            raise ValueError("|xi0| must be <= 1")
        rd = deflection_ratio(ions, A_b, lli)
        v0 = np.sqrt(2.0 * E0_j / (A_b * M_AMU))
        I1 = pitch_integral(w, rd / 3.0, 3)          # alpha_1 = 2 R_D / 6
        K2 = pitch_integral(w, rd, 4)                # alpha_2 = 6 R_D / 6
        out["j_fast"] = Z_b * E_CHARGE * S * tau_s * v0 * xi0 * I1
        p2 = 0.5 * (3.0 * xi0 ** 2 - 1.0) if p2_xi0 is None else np.broadcast_to(np.asarray(p2_xi0, float), S.shape)
        dp = 2.0 * p2 * S * E0_j * tau_s * K2
        out["p_par"] = out["p_fast"] + (2.0 / 3.0) * dp
        out["p_perp"] = out["p_fast"] - dp / 3.0
        out["R_D"] = rd
    return out
