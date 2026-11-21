"""Physics-informed critical-gradient stiffness and GP correlation length scales.

Computes radially varying 1-D non-stationary Gibbs correlation length arrays
``ell(x)`` and local inverse-scale-length excursion bounds ``Delta(R/L_f)(x)``
for ``ne``, ``te``, ``ni``, ``ti``, ``zeff``, and ``j_phi`` across two tiers:

* **Tier 1 (`tier1_analytical`):** Pure-NumPy analytical critical-gradient and
  transport scaling laws:
  - ``Te``: Jenko et al. (Phys. Plasmas 8, 4096, 2001) ETG + Peeters et al.
    (Phys. Plasmas 12, 022505, 2005) TEM + Garbet et al. (PPCF 46, 1351, 2004)
    critical-gradient subcritical margin + stiffness ``Q_s / (chi_stiff * Q_gB)``.
  - ``Ti``: Romanelli (PoFB 1, 1018, 1989) / Kotschenreuther et al. (PoP 2,
    2381, 1995) dilution-stabilized ITG threshold + Waltz, Candy & Rosenbluth
    (PoP 9, 1938, 2002) finite-``rho_star`` profile shearing + Citrin et al.
    (PRL 111, 155001, 2013) electromagnetic / fast-ion stiffness reduction.
  - ``ne, ni``: Angioni et al. (Nucl. Fusion 47, 1326, 2007) collisionality
    density-peaking spread.
  - ``Zeff``: Explicit user prior ``aux_length_scales['zeff']`` preserved
    verbatim (cf. Fajardo, Angioni, Dux & Manas, PPCF 64, 044002, 2022).
  - ``j_phi``: Explicit user prior ``j_ls`` preserved verbatim unless
    ``current_evolution_time_s > 0`` activates Sauter et al. (PoP 6, 2834, 1999)
    neoclassical resistive skin depth with smooth ``tanh`` sawtooth blending
    across ``q = 1``.

* **Tier 2 (`tier2_tglfnn` / `tier2_qlknn`):** Evaluates local quasilinear
  turbulent fluxes and exact stiffness Jacobians ``dQ_s / d(R/L_Ts)`` and
  ``dGamma_e / d(R/L_ne)`` via :mod:`bouquet.surrogates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import integrate

from .geometry.miller import (
    MillerGeometry,
    build_tglf_inputs,
    gyrobohm_flux_scales,
    miller_from_baseline,
)
from .physics import ELEMENTARY_CHARGE

_MU0 = 4.0e-7 * np.pi
_EPS0 = 8.8541878128e-12
_ME_KG = 9.1093837015e-31
_MP_KG = 1.67262192369e-27
_GH7_X, _GH7_W = np.polynomial.hermite.hermgauss(7)
_Z_NODES = np.sqrt(2.0) * _GH7_X
_Z_WEIGHTS = _GH7_W / np.sqrt(np.pi)


@dataclass
class PedestalWindow:
    """Detected H-mode pedestal geometry on ``psi_N``."""

    has_pedestal: bool
    psi_ped_top: float
    psi_sym: float
    width_psiN: float
    eped_width_psiN: float
    beta_p_ped: float


@dataclass
class PhysicsStiffnessResult:
    """Resolved non-stationary length scales and critical-gradient diagnostics."""

    mode: str
    psi_kin: np.ndarray
    psi_j: np.ndarray
    rho_kin: np.ndarray
    pedestal: PedestalWindow
    # Per-channel non-stationary Gibbs length scales on psi_kin (and psi_j for j_ls)
    ne_ls: np.ndarray
    te_ls: np.ndarray
    ni_ls: np.ndarray
    ti_ls: np.ndarray
    zeff_ls: np.ndarray
    j_ls: np.ndarray
    # Shared legacy-compatible aliases (harmonic mean of e/i where scalar slot used)
    n_ls: np.ndarray
    t_ls: np.ndarray
    # Baseline & critical inverse scale lengths R0 / L_s on psi_kin
    rlne_base: np.ndarray
    rlte_base: np.ndarray
    rlni_base: np.ndarray
    rlti_base: np.ndarray
    rlte_crit: np.ndarray
    rlti_crit: np.ndarray
    # Realizable 1-sigma inverse scale length excursions Delta(R0 / L_s) on psi_kin
    dkappa_ne: np.ndarray
    dkappa_te: np.ndarray
    dkappa_ni: np.ndarray
    dkappa_ti: np.ndarray
    # Regional gradient bounds [kappa_min, kappa_max] in R0 / L_s units
    kappa_bounds: Dict[str, tuple]
    # Gyro-Bohm and surrogate metadata
    Q_gB_MW_m2: np.ndarray
    surrogate_meta: Optional[dict] = None
    gibbs_ls_form: str = "chilenski_tanh"

    @property
    def dkappa_allow(self) -> Dict[str, np.ndarray]:
        """Channel dictionary of realizable 1-sigma inverse scale-length excursions."""
        return {
            "ne": self.dkappa_ne,
            "te": self.dkappa_te,
            "ni": self.dkappa_ni,
            "ti": self.dkappa_ti,
        }

    def summary_dict(self) -> dict:
        """Compact JSON-safe provenance summary for baseline / archive records."""
        core = (self.rho_kin >= 0.25) & (self.rho_kin <= 0.75)
        if not np.any(core):
            core = slice(None)
        return {
            "mode": str(self.mode),
            "gibbs_ls_form": str(self.gibbs_ls_form),
            "has_pedestal": bool(self.pedestal.has_pedestal),
            "psi_ped_top": float(self.pedestal.psi_ped_top),
            "ped_width_psiN": float(self.pedestal.width_psiN),
            "eped_width_psiN": float(self.pedestal.eped_width_psiN),
            "core_median_ls": {
                "ne": float(np.median(self.ne_ls[core])),
                "te": float(np.median(self.te_ls[core])),
                "ni": float(np.median(self.ni_ls[core])),
                "ti": float(np.median(self.ti_ls[core])),
                "zeff": float(np.median(self.zeff_ls[core])),
                "jphi": float(np.median(self.j_ls)),
            },
            "ped_min_ls": {
                "ne": float(np.min(self.ne_ls)),
                "te": float(np.min(self.te_ls)),
                "jphi": float(np.min(self.j_ls)),
            },
            "core_median_dkappa": {
                "te": float(np.median(self.dkappa_te[core])),
                "ti": float(np.median(self.dkappa_ti[core])),
                "ne": float(np.median(self.dkappa_ne[core])),
            },
            "surrogate": self.surrogate_meta,
        }


def bosch_hale_dt_reactivity(ti_eV: np.ndarray) -> np.ndarray:
    """Bosch-Hale DT thermal reactivity ``<sigma*v>`` [m^3/s] for 50-50 D-T.

    Reference: H.-S. Bosch & G. M. Hale, Nucl. Fusion 32, 611 (1992), Table VII.
    Valid across 0.2 keV <= T_i <= 100 keV.
    """
    ti_keV = np.clip(np.asarray(ti_eV, dtype=float) * 1.0e-3, 0.2, 100.0)
    bg = 34.3827
    mrc2 = 1124656.0
    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.35000e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5

    numer = ti_keV * (c2 + ti_keV * (c4 + ti_keV * c6))
    denom = 1.0 + ti_keV * (c3 + ti_keV * (c5 + ti_keV * c7))
    theta = ti_keV / np.maximum(1.0 - numer / denom, 1e-6)
    xi = ((bg ** 2) / (4.0 * theta)) ** (1.0 / 3.0)
    sv_cm3_s = c1 * theta * np.sqrt(xi / (mrc2 * (ti_keV ** 3))) * np.exp(-3.0 * xi)
    return sv_cm3_s * 1.0e-6


def _diluted_fuel_density(
    ne_m3: np.ndarray,
    ni_m3: np.ndarray,
    zeff: np.ndarray,
    Z_imp: float = 6.0,
    z_fast_m3: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve thermal hydrogenic fuel density ``n_D + n_T`` [m^-3].

    When ``ni_m3`` is already distinct from ``ne_m3``, preserves ``ni_m3``.
    When ``ni_m3 == ne_m3`` on reconstructions where ``Z_eff > 1`` was not yet
    applied to ``n_i``, dilutes via quasineutrality ``(Z_imp - Z_eff)/(Z_imp - 1)``.
    """
    ne = np.maximum(np.asarray(ne_m3, dtype=float), 0.0)
    ni = np.maximum(np.asarray(ni_m3, dtype=float), 0.0)
    z_imp = max(float(Z_imp), 2.0)
    zf = np.clip(np.asarray(zeff, dtype=float), 1.0, z_imp - 1.0e-3)
    zf_fast = np.zeros_like(ne) if z_fast_m3 is None else np.maximum(np.asarray(z_fast_m3, dtype=float), 0.0)
    if np.allclose(ni, ne, rtol=1.0e-4, atol=1.0e14) and np.any(zf > 1.0 + 1.0e-4):
        f_dil = np.clip((z_imp - zf) / max(z_imp - 1.0, 1.0), 0.05, 1.0)
        return np.maximum(ne - zf_fast, 0.0) * f_dil
    return ni


def detect_pedestal_window(
    psi_N: np.ndarray,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    miller: MillerGeometry,
    Ip_A: float,
    *,
    zeff: Optional[np.ndarray] = None,
    Z_imp: float = 6.0,
) -> PedestalWindow:
    """Detect H-mode pedestal top and width on poloidal flux ``psi_N``
    via finite-difference HWHM and EPED1 ``0.076 * sqrt(beta_p,ped)`` scaling."""
    psi = np.asarray(psi_N, dtype=float)
    ne_pos = np.maximum(ne_m3, 0.0)
    te_pos = np.maximum(te_eV, 0.0)
    ni_pos = np.maximum(ni_m3, 0.0)
    ti_pos = np.maximum(ti_eV, 0.0)
    z_imp = max(float(Z_imp), 2.0)
    if zeff is not None:
        ni_eff = _diluted_fuel_density(ne_pos, ni_pos, zeff, Z_imp=z_imp)
    else:
        ni_eff = ni_pos
    nz_pos = np.maximum((ne_pos - ni_eff) / z_imp, 0.0)
    p_th = ELEMENTARY_CHARGE * (ne_pos * te_pos + (ni_eff + nz_pos) * ti_pos)
    dp_dpsi = -np.gradient(p_th, psi)

    edge_mask = (psi >= 0.80) & (psi <= 0.998)
    core_mask = (psi >= 0.40) & (psi <= 0.75)
    edge_peak = float(np.max(dp_dpsi[edge_mask])) if np.any(edge_mask) else 0.0
    core_med = float(np.median(np.maximum(dp_dpsi[core_mask], 1e-6))) if np.any(core_mask) else 1.0

    # Estimate poloidal field at LCFS for EPED beta_p,ped = p_ped / (Bp_edge^2 / 2*mu0),
    # preferring exact shaped flux-surface area S_LCFS / (2*pi*R_maj) when available.
    s_lcfs = float(miller.surface_m2[-1]) if miller.surface_m2.size else 0.0
    r_maj_lcfs = float(miller.R_maj_m[-1]) if miller.R_maj_m.size else float(miller.R0_m)
    if s_lcfs > 0.1 and r_maj_lcfs > 0.1:
        perim_lcfs = s_lcfs / (2.0 * np.pi * r_maj_lcfs)
    else:
        perim_lcfs = 2.0 * np.pi * miller.a_m * np.sqrt(0.5 * (1.0 + float(miller.kappa[-1]) ** 2))
    bp_lcfs = max(_MU0 * abs(float(Ip_A)) / max(perim_lcfs, 0.1), 0.05)

    has_ped = edge_peak > 2.2 * max(core_med, 1.0)
    if has_ped:
        edge_indices = np.where(edge_mask)[0]
        i_sym = int(edge_indices[int(np.argmax(dp_dpsi[edge_indices]))])
        psi_sym = float(psi[i_sym])
        half_ht = 0.5 * (edge_peak + core_med)
        # Walk inward from symmetry point until -dp/dpsi drops below half_ht
        i_in = i_sym
        while i_in > 0 and psi[i_in] > 0.70 and dp_dpsi[i_in] > half_ht:
            i_in -= 1
        half_w_meas = max(psi_sym - float(psi[i_in]), 0.008)
        psi_ped_cand = float(np.clip(psi_sym - 1.8 * half_w_meas, 0.72, 0.985))
        p_ped = float(np.interp(psi_ped_cand, psi, p_th))
        beta_p_ped = max(p_ped / ((bp_lcfs ** 2) / (2.0 * _MU0)), 1e-3)
        eped_w = float(np.clip(0.076 * np.sqrt(beta_p_ped), 0.015, 0.22))
        width_psi = float(np.clip(1.0 - psi_ped_cand, 0.6 * eped_w, 1.8 * eped_w))
        psi_ped_top = float(np.clip(1.0 - width_psi, 0.72, 0.985))
    else:
        p_92 = float(np.interp(0.92, psi, p_th))
        beta_p_ped = max(p_92 / ((bp_lcfs ** 2) / (2.0 * _MU0)), 1e-3)
        eped_w = float(np.clip(0.076 * np.sqrt(beta_p_ped), 0.015, 0.22))
        psi_sym = 0.97
        psi_ped_top = 0.95
        width_psi = 0.05

    return PedestalWindow(
        has_pedestal=has_ped,
        psi_ped_top=psi_ped_top,
        psi_sym=psi_sym,
        width_psiN=width_psi,
        eped_width_psiN=eped_w,
        beta_p_ped=beta_p_ped,
    )


def _braginskii_ei_exchange_W_m3_eV(
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    zeff: np.ndarray,
    A_main: float = 2.0,
    *,
    ni_m3: Optional[np.ndarray] = None,
    Z_imp: float = 6.0,
    A_imp: Optional[float] = None,
) -> np.ndarray:
    """Exact multi-species SI Braginskii electron-ion Coulomb heat-exchange rate [W m^-3 eV^-1].

    Computes ``alpha_ei = sum_s 3 * (m_e / m_s) * n_e * e / tau_es``, which scales
    with ``sum_s n_s Z_s^2 / A_s`` (Braginskii 1965; Hinton & Hazeltine, Rev. Mod.
    Phys. 48, 239, 1976) rather than ``n_e Z_eff / A_main``.
    """
    ne_s = np.maximum(np.asarray(ne_m3, dtype=float), 1.0e15)
    te_s = np.maximum(np.asarray(te_eV, dtype=float), 10.0)
    z_imp = max(float(Z_imp), 2.0)
    a_main = max(float(A_main), 1.0)
    a_imp = max(float(A_imp) if A_imp is not None else 2.0 * z_imp, 1.0)
    zf_s = np.clip(np.asarray(zeff, dtype=float), 1.0, z_imp - 1.0e-3)
    if ni_m3 is not None:
        ni_s = np.clip(_diluted_fuel_density(ne_s, ni_m3, zf_s, Z_imp=z_imp), 0.0, ne_s)
    else:
        ni_s = ne_s * np.clip((z_imp - zf_s) / max(z_imp - 1.0, 1.0), 0.0, 1.0)
    nz_s = np.maximum((ne_s - ni_s) / z_imp, 0.0)

    ln_lam = np.maximum(24.0 - np.log(np.sqrt(ne_s * 1.0e-6) / te_s), 10.0)
    tau_e1 = (
        12.0 * (np.pi ** 1.5) * (_EPS0 ** 2) * np.sqrt(_ME_KG)
        * ((ELEMENTARY_CHARGE * te_s) ** 1.5)
    ) / (
        np.sqrt(2.0) * ne_s * (ELEMENTARY_CHARGE ** 4) * ln_lam
    )
    # Multi-species mass-weighted factor: sum_s (n_s / n_e) * Z_s^2 / A_s
    z2_over_a = (ni_s / a_main + nz_s * (z_imp ** 2) / a_imp) / ne_s
    return 3.0 * (_ME_KG / _MP_KG) * z2_over_a * ne_s * ELEMENTARY_CHARGE / np.maximum(tau_e1, 1.0e-12)


def _exact_ell_psi_from_ell_r(
    psi_N: np.ndarray,
    r_min_m: np.ndarray,
    a_m: float,
    ell_r_m: np.ndarray,
) -> np.ndarray:
    """Map physical minor-radius correlation length ``ell_r`` [m] onto run coordinate
    ``psi_N`` (or ``Phi_N``) via exact 7-point Gauss-Hermite quadrature of
    ``E[(x(|r + ell_r Z|) - x(r))^2]^(1/2)`` for ``Z ~ N(0, 1)``.

    Preserves positivity on the magnetic axis (``r = 0``, where ``dx/dr = 0``
    so first-order ``|dx/dr| * ell_r`` spuriously vanishes while ``x(ell_r) ~ (ell_r/a)^2``
    is finite), linearly extrapolates beyond the outer knot with guarded boundary
    slope, and allows ``ell_x > 1`` naturally without ceilings or filters.
    """
    psi = np.asarray(psi_N, dtype=float)
    r_m = np.asarray(r_min_m, dtype=float)
    a = max(float(a_m), 1.0e-4)
    ell_r = np.maximum(np.asarray(ell_r_m, dtype=float), 1.0e-6)
    r_top = max(float(r_m[-1]), 1.0e-4) if r_m.size else a

    slope_a = 2.0 / a
    if psi.size >= 2:
        dr = np.diff(r_m)
        good = np.where(dr > 1.0e-4 * a)[0]
        if good.size > 0:
            k = int(good[-1])
            raw_slope = (psi[k + 1] - psi[k]) / dr[k]
            slope_a = float(np.clip(raw_slope, 0.2 / a, 10.0 / a))

    r_queries = np.abs(r_m[:, None] + ell_r[:, None] * _Z_NODES[None, :])
    psi_in = np.interp(np.minimum(r_queries, r_top), r_m, psi)
    psi_q = np.where(r_queries <= r_top, psi_in, psi[-1] + slope_a * (r_queries - r_top))
    var_psi = np.sum(_Z_WEIGHTS[None, :] * ((psi_q - psi[:, None]) ** 2), axis=1)
    return np.sqrt(np.maximum(var_psi, 1.0e-8))


def _estimate_power_balance_fluxes(
    mg: MillerGeometry,
    inp: dict,
    ne: np.ndarray,
    te: np.ndarray,
    ni: np.ndarray,
    ti: np.ndarray,
    Ip_A: float = 1.0e6,
    chi_e_base: Optional[np.ndarray] = None,
    chi_i_base: Optional[np.ndarray] = None,
    *,
    tritium_fraction: float = 0.0,
    n_fuel: Optional[np.ndarray] = None,
) -> tuple:
    """Compute steady-state electron and ion surface heat fluxes [W/m^2] from
    baseline ``chi_s`` when available, else Martin (2008) H-mode L-H threshold
    loss power ``P_loss = max(P_alpha,tot, 1.8 * P_LH)`` partitioned between
    isotope-gated Bosch-Hale DT alpha self-heating and auxiliary power."""
    surf = np.maximum(mg.surface_m2, 0.1)
    vol = mg.volume_m3

    f_trit = float(np.clip(float(tritium_fraction), 0.0, 1.0))
    if 0.0 < f_trit < 1.0:
        nf = np.maximum(ni if n_fuel is None else n_fuel, 0.0)
        sv_dt = bosch_hale_dt_reactivity(ti)
        p_alpha_W_m3 = f_trit * (1.0 - f_trit) * (nf ** 2) * sv_dt * (3.52e6 * ELEMENTARY_CHARGE)
        P_alpha_cum = integrate.cumulative_trapezoid(p_alpha_W_m3, vol, initial=0.0)
        p_alpha_tot_W = float(P_alpha_cum[-1]) if P_alpha_cum.size else 0.0
    else:
        P_alpha_cum = np.zeros_like(vol, dtype=float)
        p_alpha_tot_W = 0.0

    # Martin et al. (J. Phys.: Conf. Ser. 123, 012033, 2008) ITPA L-H threshold scaling:
    #   P_LH [MW] = 0.0488 * n20_line^0.717 * B0^0.803 * S_LCFS^0.941,
    # where n20_line = (1/a) * int_0^a n_e(r) dr is the chord line-averaged density.
    a_safe = max(float(mg.a_m), 1.0e-4)
    r_m = np.asarray(mg.r_min_m, dtype=float)
    if r_m.size >= 2:
        r_chord = r_m
        ne_chord = np.asarray(ne, dtype=float)
        if r_chord[0] > 1.0e-6 * a_safe:
            r_chord = np.concatenate(([0.0], r_chord))
            ne_chord = np.concatenate(([ne_chord[0]], ne_chord))
        if r_chord[-1] < a_safe - 1.0e-6 * a_safe:
            r_chord = np.concatenate((r_chord, [a_safe]))
            ne_chord = np.concatenate((ne_chord, [ne_chord[-1]]))
        ne_line = float(np.trapezoid(ne_chord, r_chord)) / a_safe
    else:
        ne_line = float(np.mean(ne))
    n20_line = max(ne_line * 1.0e-20, 0.1)
    s_lcfs = max(float(mg.surface_m2[-1]), 5.0)
    p_lh_W = 0.0488e6 * (n20_line ** 0.717) * (max(abs(float(mg.B0_T)), 0.2) ** 0.803) * (s_lcfs ** 0.941)
    p_loss_W = max(p_alpha_tot_W, 1.8 * p_lh_W)
    p_aux_tot_W = max(p_loss_W - p_alpha_tot_W, 0.0)
    core_dep = 1.0 - np.exp(-((np.maximum(mg.rho_tor_norm, 0.05) / 0.38) ** 2))
    P_aux_cum = p_aux_tot_W * core_dep

    if chi_e_base is not None and np.any(np.asarray(chi_e_base) > 0):
        dte_dr = np.maximum(np.abs(inp["RLTS_1"]) * te / a_safe, 1.0)
        Qe_phys = np.maximum(ne * ELEMENTARY_CHARGE * np.asarray(chi_e_base, float) * dte_dr, 1e2)
    else:
        # Stix (Plasma Phys. 1972): 3.52 MeV DT alphas heat electrons (~80%) > ions (~20%)
        Qe_phys = (0.80 * P_alpha_cum + 0.50 * P_aux_cum) / surf

    if chi_i_base is not None and np.any(np.asarray(chi_i_base) > 0):
        dti_dr = np.maximum(np.abs(inp["RLTS_2"]) * ti / a_safe, 1.0)
        Qi_phys = np.maximum(ni * ELEMENTARY_CHARGE * np.asarray(chi_i_base, float) * dti_dr, 1e2)
    else:
        Qi_phys = (0.20 * P_alpha_cum + 0.50 * P_aux_cum) / surf

    return Qe_phys, Qi_phys, p_alpha_tot_W


def _tier1_analytical_thresholds(
    mg: MillerGeometry,
    psi_kin_pol: np.ndarray,
    ne: np.ndarray,
    te: np.ndarray,
    ni: np.ndarray,
    ti: np.ndarray,
    zeff: np.ndarray,
    p_fast: Optional[np.ndarray] = None,
    chi_e_base: Optional[np.ndarray] = None,
    chi_i_base: Optional[np.ndarray] = None,
    Ip_A: float = 1.0e6,
    flux_uncertainty_rel: float = 0.30,
    *,
    tritium_fraction: float = 0.0,
    Z_imp: float = 6.0,
    z_fast: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate Tier 1 analytical critical gradients and allowable excursions."""
    f_trit = float(np.clip(float(tritium_fraction), 0.0, 1.0))
    A_main = 2.0 * (1.0 - f_trit) + 3.0 * f_trit
    n_fuel = _diluted_fuel_density(ne, ni, zeff, Z_imp=Z_imp, z_fast_m3=z_fast)

    inp = build_tglf_inputs(
        mg, psi_kin_pol, ne, te, ni, ti, zeff,
        Z_imp=Z_imp, A_main=A_main, z_fast_m3=z_fast,
    )
    gb = gyrobohm_flux_scales(mg, ne, te)

    R_over_a = float(mg.R0_m / max(mg.a_m, 1e-4))
    eps = np.clip(mg.r_min_m / np.maximum(mg.R_maj_m, 0.1), 1e-3, 0.65)
    q = np.maximum(mg.q, 0.7)
    s_hat = np.clip(mg.s_hat, -5.0, 5.0)
    shat_over_q = np.maximum(np.abs(s_hat) / q, 0.05)
    kappa_term = np.maximum(1.0 + 0.3 * mg.kappa * mg.s_kappa, 0.5)
    eps_term = np.maximum(1.0 - 1.5 * eps, 0.25)

    rlne = inp["RLNS_1"] * R_over_a
    rlte = inp["RLTS_1"] * R_over_a
    rlni = inp["RLNS_2"] * R_over_a
    rlti = inp["RLTS_2"] * R_over_a

    tau_ei = np.clip(te / np.maximum(ti, 1.0), 0.2, 5.0)
    tau_ie = 1.0 / tau_ei
    zeff_c = np.clip(zeff, 1.0, 6.0)
    f_i = np.clip(n_fuel / np.maximum(ne, 1.0e16), 0.1, 1.0)

    # 1. Te critical gradient: min(ETG [Jenko PoP 2001], TEM [Peeters PoP 2005])
    rlte_etg = np.maximum(
        (1.0 + zeff_c * tau_ei) * (1.33 + 1.91 * shat_over_q) * eps_term * kappa_term,
        0.8 * np.maximum(rlne, 0.0),
    )
    ne_19 = np.maximum(ne, 1e17) * 1.0e-19
    te_keV = np.maximum(te, 20.0) * 1.0e-3
    nu_eff = 0.1 * zeff_c * ne_19 * mg.R_maj_m / np.maximum(te_keV ** 2, 1e-4)
    rlte_tem = (
        ((0.357 * np.sqrt(eps) + 0.271) / np.sqrt(eps))
        * np.maximum(
            4.90 - 1.31 * np.clip(rlne, -2.0, 6.0) + 2.68 * np.maximum(s_hat, 0.0)
            + np.log(1.0 + 20.0 * np.maximum(nu_eff, 0.0)),
            1.5,
        )
    )
    rlte_crit = np.clip(np.minimum(rlte_etg, rlte_tem), 1.2, 18.0)

    # 2. Ti critical gradient: toroidal ITG [Romanelli PoFB 1989 / Kotschenreuther PoP 1995]
    #    with main-ion dilution stabilization (1 + T_i / (f_i * T_e))
    rlti_itg = np.maximum(
        (1.0 + tau_ie / f_i) * (1.33 + 1.91 * shat_over_q) * eps_term * kappa_term,
        0.8 * np.maximum(rlni, 0.0) + 0.8 * (1.0 + tau_ie),
    )
    rlti_crit = np.clip(rlti_itg, 1.2, 18.0)

    # 3. Local heat flux in gyro-Bohm units Q_s / Q_gB
    Q_gB = np.maximum(gb["Q_gB_imas_W_m2"], 1.0)
    Qe_phys, Qi_phys, p_alpha_tot = _estimate_power_balance_fluxes(
        mg, inp, ne, te, ni, ti, Ip_A=Ip_A,
        chi_e_base=chi_e_base, chi_i_base=chi_i_base,
        tritium_fraction=f_trit, n_fuel=n_fuel,
    )
    Qe_gB_est = Qe_phys / Q_gB
    Qi_gB_est = Qi_phys / Q_gB

    # Electromagnetic + fast-ion stiffness reduction (Citrin PRL 2013)
    p_th = np.maximum(ELEMENTARY_CHARGE * (ne * te + ni * ti), 1.0)
    f_fast = (np.zeros_like(psi_kin_pol) if p_fast is None
              else np.clip(np.asarray(p_fast, dtype=float) / p_th, 0.0, 2.0))
    em_relief = 1.0 + (inp["BETAE"] / 0.010) * (1.0 + 1.5 * f_fast)

    # Garbet et al. (PPCF 46, 1351, 2004, Eq. 5) critical-gradient stiffness factor:
    # Q_s = n_s T_s c_s (rho_s / R_0)^2 * chi_s0 * (R_0/L_Ts - kappa_c) * H(R_0/L_Ts - kappa_c),
    # with chi_s0 ~ q^(3/2) / |s_hat| and geometric conversion (a / R_0)^2 to GACODE Q_gB.
    # Waltz, Candy & Rosenbluth (Phys. Plasmas 9, 1938, 2002) finite-rho_star profile shearing
    # reduces effective stiffness at large rho_star and saturates to 1.0 (pure gyro-Bohm) as rho_star -> 0.
    a_over_R_sq = (1.0 / R_over_a) ** 2
    chi_garbet = (q ** 1.5) / np.maximum(np.abs(s_hat), 0.25)
    rho_star_factor = np.clip(1.0 - gb["rho_star"] / 0.015, 0.25, 1.0)
    stiff_slope_e = a_over_R_sq * chi_garbet * rho_star_factor
    stiff_slope_i = a_over_R_sq * (1.25 * chi_garbet * rho_star_factor) / em_relief

    f_flux = max(float(flux_uncertainty_rel), 1.0e-3)
    in_core = (mg.rho_tor_norm >= 0.18) & (mg.rho_tor_norm <= 0.88)
    subcrit_te = np.maximum(rlte_crit - rlte, 0.0)
    subcrit_ti = np.maximum(rlti_crit - rlti, 0.0)
    dk_te_raw = subcrit_te + (f_flux * Qe_gB_est) / stiff_slope_e
    dk_ti_raw = subcrit_ti + (f_flux * Qi_gB_est) / stiff_slope_i
    if np.any(in_core):
        dkappa_te = np.interp(psi_kin_pol, psi_kin_pol[in_core], dk_te_raw[in_core])
        dkappa_ti = np.interp(psi_kin_pol, psi_kin_pol[in_core], dk_ti_raw[in_core])
    else:
        dkappa_te, dkappa_ti = dk_te_raw, dk_ti_raw

    # Angioni et al. (Nucl. Fusion 47, 1326, 2007) empirical density-peaking RMSE (~0.75 in R/L_ne)
    dkappa_ne = 0.75 + 0.35 * np.log1p(1.0 / np.maximum(nu_eff, 0.02))
    dkappa_ni = dkappa_ne.copy()

    return {
        "rlne_base": rlne,
        "rlte_base": rlte,
        "rlni_base": rlni,
        "rlti_base": rlti,
        "rlte_crit": rlte_crit,
        "rlti_crit": rlti_crit,
        "dkappa_ne": np.maximum(dkappa_ne, 1.0e-3),
        "dkappa_te": np.maximum(dkappa_te, 1.0e-3),
        "dkappa_ni": np.maximum(dkappa_ni, 1.0e-3),
        "dkappa_ti": np.maximum(dkappa_ti, 1.0e-3),
        "Qe_phys_W_m2": Qe_phys,
        "Qi_phys_W_m2": Qi_phys,
        "Q_gB_W_m2": Q_gB,
        "Q_gB_MW_m2": Q_gB * 1.0e-6,
        "A_main": A_main,
        "n_fuel": n_fuel,
        "p_alpha_tot_W": p_alpha_tot,
    }


def _map_dkappa_to_gibbs_ls(
    psi_N: np.ndarray,
    mg: MillerGeometry,
    prof: np.ndarray,
    sigma_prof: np.ndarray,
    dkappa_allow: np.ndarray,
    ped: PedestalWindow,
    *,
    gibbs_ls_form: str = "chilenski_tanh",
) -> np.ndarray:
    """Convert allowable ``Delta(R0/L_f)`` into an unclamped Gibbs length scale ``ell(x)``.

    Uses:
      1. Exact GP derivative-variance identity ``Std(df/dr) = sigma_f / ell_r``
         (prefactor 1.0, giving ``ell_r = R0 * (sigma_f / f) / Delta(R0/L_f)``),
      2. Exact 7-point Gauss-Hermite expectation mapping ``ell_r`` onto the run
         coordinate ``x`` (``psi_N`` or ``Phi_N``), and
      3. Either the Chilenski et al. (Nucl. Fusion 55, 023012, 2015, Eq. 30)
         linear two-plateau ``tanh`` form ``(1 - g) * ell_core + g * ell_ped``
         with transition width ``max(ell_ped, 0.03)`` (``gibbs_ls_form='chilenski_tanh'``)
         or the raw surface-by-surface curve (``gibbs_ls_form='pointwise'``).
    """
    if gibbs_ls_form not in ("chilenski_tanh", "pointwise"):
        raise ValueError(
            f"Unknown gibbs_ls_form={gibbs_ls_form!r}; expected "
            "'chilenski_tanh' or 'pointwise'"
        )
    f_safe = np.maximum(np.abs(np.asarray(prof, dtype=float)), 1.0e-30)
    sig_rel = np.maximum(np.asarray(sigma_prof, dtype=float) / f_safe, 1.0e-6)
    ell_r_m = float(mg.R0_m) * sig_rel / np.maximum(dkappa_allow, 1.0e-6)
    ell_core_pt = _exact_ell_psi_from_ell_r(psi_N, mg.r_min_m, mg.a_m, ell_r_m)

    if gibbs_ls_form == "chilenski_tanh":
        core_mask = (mg.rho_tor_norm >= 0.25) & (mg.rho_tor_norm <= 0.75)
        c_val = float(np.median(ell_core_pt[core_mask])) if np.any(core_mask) else float(np.median(ell_core_pt))
        ell_core = np.full_like(psi_N, max(c_val, 1.0e-6))
    else:
        ell_core = ell_core_pt

    if ped.has_pedestal:
        ell_ped = max(float(ped.width_psiN), 1.0e-3)
        w_half = 0.5 * max(ell_ped, 0.03)
        gate = 0.5 * (1.0 + np.tanh((psi_N - ped.psi_ped_top) / w_half))
        return np.maximum((1.0 - gate) * ell_core + gate * ell_ped, 1.0e-6)
    return ell_core


def _resolve_poloidal_mapping(baseline, source, raw_coord: str, x_kin: np.ndarray, x_j: np.ndarray):
    """Map run-coordinate grids ``(x_kin, x_j)`` to true poloidal flux ``(psi_kin_pol, psi_j_pol)``
    and return ``(psi_kin_pol, psi_j_pol, mg_kin, pol_to_run_fn)``."""
    if raw_coord == "psi_n":
        mg_kin = miller_from_baseline(baseline, source=source, psi_N_out=x_kin)
        return x_kin, x_j, mg_kin, (lambda p: p)
    if raw_coord not in ("phi_n", "rho_tor"):
        raise ValueError(
            f"Unknown baseline/source coord={raw_coord!r}; expected 'psi_n', 'phi_n', or 'rho_tor'"
        )

    psi_map = getattr(baseline, "psi_map", None)
    if callable(psi_map):
        psi_kin_pol = np.clip(np.asarray(psi_map(x_kin), dtype=float), 0.0, 1.0)
        psi_j_pol = np.clip(np.asarray(psi_map(x_j), dtype=float), 0.0, 1.0)
        x_dense = np.linspace(0.0, 1.0, 513)
        p_dense = np.clip(np.asarray(psi_map(x_dense), dtype=float), 0.0, 1.0)
        order = np.argsort(p_dense)
        p_sorted, x_sorted = p_dense[order], x_dense[order]
        mg_kin = miller_from_baseline(baseline, source=source, psi_N_out=psi_kin_pol)
        return psi_kin_pol, psi_j_pol, mg_kin, (lambda p: np.interp(p, p_sorted, x_sorted))

    if isinstance(psi_map, (tuple, list)) and len(psi_map) == 2:
        p_nodes = np.asarray(psi_map[0], dtype=float)
        x_nodes = np.asarray(psi_map[1], dtype=float)
        psi_kin_pol = np.clip(np.interp(x_kin, x_nodes, p_nodes), 0.0, 1.0)
        psi_j_pol = np.clip(np.interp(x_j, x_nodes, p_nodes), 0.0, 1.0)
        mg_kin = miller_from_baseline(baseline, source=source, psi_N_out=psi_kin_pol)
        return psi_kin_pol, psi_j_pol, mg_kin, (lambda p: np.interp(p, p_nodes, x_nodes))

    # Fallback when psi_map is None: invert (mg.rho_tor_norm**2, mg.psi_N) from native MillerGeometry
    aux = getattr(baseline, "aux", None) or {}
    if isinstance(aux.get("miller_geometry"), MillerGeometry):
        mg_ref = aux["miller_geometry"]
    elif getattr(baseline, "eqdsk_bytes", None) or (
        source is not None and (getattr(source, "ids_path", None) or getattr(source, "geqdsk_path", None))
    ):
        mg_ref = miller_from_baseline(baseline, source=source, psi_N_out=None)
    else:
        mg_ref = miller_from_baseline(
            baseline, source=source, psi_N_out=np.linspace(0.0, 1.0, max(x_kin.size, 257))
        )

    x_ref = (mg_ref.rho_tor_norm ** 2) if raw_coord == "phi_n" else mg_ref.rho_tor_norm
    psi_kin_pol = np.clip(np.interp(x_kin, x_ref, mg_ref.psi_N), 0.0, 1.0)
    psi_j_pol = np.clip(np.interp(x_j, x_ref, mg_ref.psi_N), 0.0, 1.0)
    mg_kin = miller_from_baseline(baseline, source=source, psi_N_out=psi_kin_pol)
    return psi_kin_pol, psi_j_pol, mg_kin, (lambda p: np.interp(p, mg_ref.psi_N, x_ref))


def compute_physics_length_scales(
    baseline,
    unc_dict: dict,
    *,
    source=None,
    mode: str = "tier1_analytical",
    tglfnn_model: str = "sat2_em_d3d_azf-1_withnegD",
    current_evolution_time_s: Optional[float] = None,
    flux_uncertainty_rel: float = 0.30,
    gibbs_ls_form: str = "chilenski_tanh",
    tritium_fraction: Optional[float] = None,
) -> PhysicsStiffnessResult:
    """Compute non-stationary ``ell(x)`` and critical-gradient bounds for ``baseline``.

    Parameters
    ----------
    baseline : Baseline
        Resolved baseline equilibrium and kinetics.
    unc_dict : dict
        Partially or fully resolved uncertainty dict carrying ``sigma_ne``,
        ``sigma_te``, ``sigma_ni``, ``sigma_ti`` on ``baseline.psi_N_kinetic``
        and ``sigma_jphi`` on ``baseline.psi_N``.
    source : BaselineSource, optional
        Source config (``ImasSource`` or ``ReconstructionSource``) for exact
        Miller geometry lookup.
    mode : {'tier1_analytical', 'tier2_tglfnn', 'tier2_qlknn'}
        Physics fidelity tier.
    flux_uncertainty_rel : float, default 0.30
        Explicit fractional surface-flux uncertainty ``delta_Q / Q`` used when
        inverting transport flux curves.
    gibbs_ls_form : {'chilenski_tanh', 'pointwise'}, default 'chilenski_tanh'
        Functional form mapping allowable ``Delta(R0/L_s)`` into ``ell(x)``
        for level-space Gibbs GP sampling (see :func:`_map_dkappa_to_gibbs_ls`).
    tritium_fraction : float, optional
        Fuel tritium fraction ``n_T / (n_D + n_T)`` in ``[0, 1]``. Defaults to
        ``unc_dict.get('tritium_fraction')`` or ``baseline.tritium_fraction``,
        or ``0.0`` (pure Deuterium: ``A_main = 2.0``, ``P_alpha = 0.0``).
    """
    x_kin = np.asarray(baseline.psi_N_kinetic, dtype=float)
    x_j = np.asarray(baseline.psi_N, dtype=float)
    ne = np.asarray(baseline.ne, dtype=float)
    te = np.asarray(baseline.te, dtype=float)
    ni = np.asarray(baseline.ni, dtype=float)
    ti = np.asarray(baseline.ti, dtype=float)
    zeff = np.asarray(
        (unc_dict.get("aux_baselines") or {}).get("zeff", baseline.Zeff),
        dtype=float,
    )

    sig_ne = np.asarray(unc_dict["sigma_ne"], dtype=float)
    sig_te = np.asarray(unc_dict["sigma_te"], dtype=float)
    sig_ni = np.asarray(unc_dict["sigma_ni"], dtype=float)
    sig_ti = np.asarray(unc_dict["sigma_ti"], dtype=float)

    if tritium_fraction is not None:
        f_trit = float(np.clip(float(tritium_fraction), 0.0, 1.0))
    elif unc_dict.get("tritium_fraction") is not None:
        f_trit = float(np.clip(float(unc_dict["tritium_fraction"]), 0.0, 1.0))
    elif getattr(baseline, "tritium_fraction", None) is not None:
        f_trit = float(np.clip(float(getattr(baseline, "tritium_fraction")), 0.0, 1.0))
    else:
        f_trit = 0.0

    b_coord = str(getattr(baseline, "coord", None) or "").strip().lower()
    s_coord = str(getattr(source, "coord", None) or "").strip().lower()
    if b_coord and b_coord != "psi_n":
        raw_coord = b_coord
    elif s_coord and s_coord != "psi_n":
        raw_coord = s_coord
    else:
        raw_coord = b_coord or s_coord or "psi_n"

    psi_kin_pol, psi_j_pol, mg_kin, pol_to_run = _resolve_poloidal_mapping(
        baseline, source, raw_coord, x_kin, x_j
    )

    Z_imp = float(getattr(baseline, "Z_imp", None) or 6.0)
    Ip_A = float(getattr(baseline, "Ip_target", 1.0e6))
    ped_pol = detect_pedestal_window(
        psi_kin_pol, ne, te, ni, ti, mg_kin, Ip_A, zeff=zeff, Z_imp=Z_imp
    )
    if raw_coord != "psi_n":
        psi_ped_run = float(np.clip(pol_to_run(ped_pol.psi_ped_top), 0.0, 1.0))
        psi_sym_run = float(np.clip(pol_to_run(ped_pol.psi_sym), 0.0, 1.0))
        x_lcfs_run = float(np.clip(pol_to_run(1.0), min(psi_ped_run + 1.0e-3, 1.0), 1.0))
        width_run = max(x_lcfs_run - psi_ped_run, 1.0e-3)
        ped = PedestalWindow(
            has_pedestal=ped_pol.has_pedestal,
            psi_ped_top=psi_ped_run,
            psi_sym=psi_sym_run,
            width_psiN=width_run,
            eped_width_psiN=ped_pol.eped_width_psiN,
            beta_p_ped=ped_pol.beta_p_ped,
        )
    else:
        ped = ped_pol

    aux_base = unc_dict.get("aux_baselines") or getattr(baseline, "aux", None) or {}
    chi_e_b = aux_base.get("chi_e")
    chi_i_b = aux_base.get("chi_i")
    p_fast = getattr(baseline, "p_fast", None)
    if p_fast is not None and np.shape(p_fast) != x_kin.shape:
        p_fast = np.interp(x_kin, x_j, np.asarray(p_fast, dtype=float))
    z_fast = getattr(baseline, "z_fast", None)
    if z_fast is not None and np.shape(z_fast) != x_kin.shape:
        z_fast = np.interp(x_kin, x_j, np.asarray(z_fast, dtype=float))

    t1 = _tier1_analytical_thresholds(
        mg_kin, psi_kin_pol, ne, te, ni, ti, zeff,
        p_fast=p_fast, chi_e_base=chi_e_b, chi_i_base=chi_i_b,
        Ip_A=Ip_A, flux_uncertainty_rel=flux_uncertainty_rel,
        tritium_fraction=f_trit, Z_imp=Z_imp, z_fast=z_fast,
    )

    surr_meta = None
    if mode == "tier1_analytical":
        rlne_b, rlte_b = t1["rlne_base"], t1["rlte_base"]
        rlni_b, rlti_b = t1["rlni_base"], t1["rlti_base"]
        rlte_c, rlti_c = t1["rlte_crit"], t1["rlti_crit"]
        dk_ne, dk_te = t1["dkappa_ne"], t1["dkappa_te"]
        dk_ni, dk_ti = t1["dkappa_ni"], t1["dkappa_ti"]
    elif mode == "tier2_tglfnn":
        from .surrogates.tglfnn import evaluate_tglfnn_stiffness, solve_tglfnn_flux_excursions
        omega_tor = aux_base.get("omega_tor")
        A_main = float(t1["A_main"])
        n_fuel = t1["n_fuel"]
        stiff = evaluate_tglfnn_stiffness(
            mg_kin, psi_kin_pol, ne, te, n_fuel, ti, zeff,
            Z_imp=Z_imp, A_main=A_main, z_fast_m3=z_fast,
            omega_tor_rad_s=omega_tor, model_spec=tglfnn_model,
        )
        Qe_pb_gB = t1["Qe_phys_W_m2"] / t1["Q_gB_W_m2"]
        Qi_pb_gB = t1["Qi_phys_W_m2"] / t1["Q_gB_W_m2"]
        exc = solve_tglfnn_flux_excursions(
            mg_kin, psi_kin_pol, ne, te, n_fuel, ti, zeff,
            Qe_pb_gB, Qi_pb_gB,
            flux_uncertainty_rel=flux_uncertainty_rel,
            Z_imp=Z_imp, A_main=A_main, z_fast_m3=z_fast,
            omega_tor_rad_s=omega_tor, model_spec=tglfnn_model,
        )
        rlne_b, rlte_b = stiff.rlne_base, stiff.rlte_base
        rlni_b, rlti_b = t1["rlni_base"], stiff.rlti_base
        rlte_c, rlti_c = stiff.rlte_crit, stiff.rlti_crit

        # Exact multi-species SI Braginskii electron-ion Coulomb heat-exchange coupling:
        # fraction of species temperature variance energetically free to decouple
        # without exceeding available surface heat-flux uncertainty.
        alpha_ei = _braginskii_ei_exchange_W_m3_eV(
            ne, te, zeff, A_main=A_main, ni_m3=n_fuel, Z_imp=Z_imp
        )
        L_vs = mg_kin.volume_m3 / np.maximum(mg_kin.surface_m2, 0.1)
        qe_std = np.asarray(exc.get("Qe_gB_std_pb", stiff.Qe_gB_std), dtype=float)
        qi_std = np.asarray(exc.get("Qi_gB_std_pb", stiff.Qi_gB_std), dtype=float)
        dQe_W = float(flux_uncertainty_rel) * t1["Qe_phys_W_m2"] + qe_std * t1["Q_gB_W_m2"]
        dQi_W = float(flux_uncertainty_rel) * t1["Qi_phys_W_m2"] + qi_std * t1["Q_gB_W_m2"]
        eta_e = dQe_W / np.maximum(dQe_W + L_vs * alpha_ei * sig_te, 1.0e-12)
        eta_i = dQi_W / np.maximum(dQi_W + L_vs * alpha_ei * sig_ti, 1.0e-12)

        dk_te = np.maximum(eta_e * exc["dkappa_te_free"] + (1.0 - eta_e) * exc["dkappa_t_coupled"], 1.0e-3)
        dk_ti = np.maximum(eta_i * exc["dkappa_ti_free"] + (1.0 - eta_i) * exc["dkappa_t_coupled"], 1.0e-3)
        dk_ne = np.maximum(exc["dkappa_ne"], 1.0e-3)
        dk_ni = dk_ne.copy()
        surr_meta = {
            "engine": "tglfnn",
            "model": stiff.model_name,
            "flux_uncertainty_rel": float(flux_uncertainty_rel),
            "tritium_fraction": float(f_trit),
            "core_median_Qe_gB": float(np.median(stiff.Qe_gB[(mg_kin.rho_tor_norm >= 0.3) & (mg_kin.rho_tor_norm <= 0.75)])),
            "core_median_Qi_gB": float(np.median(stiff.Qi_gB[(mg_kin.rho_tor_norm >= 0.3) & (mg_kin.rho_tor_norm <= 0.75)])),
        }
    elif mode == "tier2_qlknn":
        from .surrogates.qlknn_backend import evaluate_qlknn_stiffness
        qres = evaluate_qlknn_stiffness(
            mg_kin, psi_kin_pol, ne, te, t1["n_fuel"], ti, zeff, Z_imp=Z_imp
        )
        rlne_b, rlte_b = t1["rlne_base"], t1["rlte_base"]
        rlni_b, rlti_b = t1["rlni_base"], t1["rlti_base"]
        rlte_c, rlti_c = qres["rlte_crit"], qres["rlti_crit"]
        f_flux = max(float(flux_uncertainty_rel), 1.0e-3)
        Qe_pb_gB = t1["Qe_phys_W_m2"] / t1["Q_gB_W_m2"]
        Qi_pb_gB = t1["Qi_phys_W_m2"] / t1["Q_gB_W_m2"]
        dk_te = np.maximum((f_flux * Qe_pb_gB) / np.maximum(qres["dQe_dRLTe_gB"], 1.0e-3), 1.0e-3)
        dk_ti = np.maximum((f_flux * Qi_pb_gB) / np.maximum(qres["dQi_dRLTi_gB"], 1.0e-3), 1.0e-3)
        dk_ne, dk_ni = t1["dkappa_ne"], t1["dkappa_ni"]
        surr_meta = {
            "engine": "qlknn",
            "flux_uncertainty_rel": float(flux_uncertainty_rel),
            "tritium_fraction": float(f_trit),
        }
    else:
        raise ValueError(
            f"Unknown length_scale_mode={mode!r}; expected 'manual', "
            "'tier1_analytical', 'tier2_tglfnn', or 'tier2_qlknn'"
        )

    ne_ls = _map_dkappa_to_gibbs_ls(
        x_kin, mg_kin, ne, sig_ne, dk_ne, ped, gibbs_ls_form=gibbs_ls_form
    )
    te_ls = _map_dkappa_to_gibbs_ls(
        x_kin, mg_kin, te, sig_te, dk_te, ped, gibbs_ls_form=gibbs_ls_form
    )
    ni_ls = _map_dkappa_to_gibbs_ls(
        x_kin, mg_kin, ni, sig_ni, dk_ni, ped, gibbs_ls_form=gibbs_ls_form
    )
    ti_ls = _map_dkappa_to_gibbs_ls(
        x_kin, mg_kin, ti, sig_ti, dk_ti, ped, gibbs_ls_form=gibbs_ls_form
    )

    # Z_eff has no critical-gradient law in TGLF-NN / QLKNN (which output only
    # OUT_G_elec, OUT_P_ions, OUT_Q_elec, OUT_Q_ions and no impurity flux):
    # preserve the explicit user prior aux_length_scales.get('zeff', 0.4) verbatim.
    zeff_prior = (unc_dict.get("aux_length_scales") or {}).get("zeff", 0.4)
    zeff_ls = np.broadcast_to(np.asarray(zeff_prior, dtype=float), x_kin.shape).copy()

    # j_phi (inductive current) length scale on x_j:
    # Preserve the user's explicit j_ls prior verbatim unless an explicit
    # resistive diffusion timescale current_evolution_time_s > 0 is set.
    if current_evolution_time_s is not None and float(current_evolution_time_s) > 0.0:
        mg_j = miller_from_baseline(baseline, source=source, psi_N_out=psi_j_pol)
        te_j = np.maximum(np.interp(psi_j_pol, psi_kin_pol, te), 10.0)
        ne_j = np.maximum(np.interp(psi_j_pol, psi_kin_pol, ne), 1.0e15)
        zeff_j = np.maximum(np.interp(psi_j_pol, psi_kin_pol, zeff), 1.0)
        ln_lam_j = np.maximum(24.0 - np.log(np.sqrt(ne_j * 1.0e-6) / te_j), 10.0)
        tau_e_j = (
            12.0 * (np.pi ** 1.5) * (_EPS0 ** 2) * np.sqrt(_ME_KG)
            * ((ELEMENTARY_CHARGE * te_j) ** 1.5)
        ) / (
            np.sqrt(2.0) * ne_j * zeff_j * (ELEMENTARY_CHARGE ** 4) * ln_lam_j
        )
        # Sauter, Angioni & Lin-Liu (Phys. Plasmas 6, 2834, 1999, Eqs. 13-14)
        # Z_eff-dependent Spitzer + neoclassical trapped-particle conductivity [S/m]
        n_zeff = 0.58 + 0.74 / (0.76 + zeff_j)
        sigma_spitzer = (1.96 / n_zeff) * ne_j * (ELEMENTARY_CHARGE ** 2) * tau_e_j / _ME_KG
        eps_j = np.clip(mg_j.r_min_m / np.maximum(mg_j.R_maj_m, 0.1), 0.0, 0.65)
        v_te_j = np.sqrt(2.0 * ELEMENTARY_CHARGE * te_j / _ME_KG)
        nu_star_e = (
            mg_j.R_maj_m * np.maximum(mg_j.q, 0.5)
            / (np.maximum(eps_j, 1.0e-3) ** 1.5 * np.maximum(v_te_j * tau_e_j, 1.0e-6))
        )
        f_trap = np.clip(1.46 * np.sqrt(eps_j) - 0.46 * (eps_j ** 1.5), 0.0, 0.90)
        f_trap_eff = f_trap / (
            1.0 + (0.55 - 0.1 * f_trap) * np.sqrt(nu_star_e)
            + 0.45 * (1.0 - f_trap) * nu_star_e / (zeff_j ** 1.5)
        )
        neo_fac = np.maximum(
            1.0 - (1.0 + 0.36 / zeff_j) * f_trap_eff
            + (0.59 / zeff_j) * (f_trap_eff ** 2)
            - (0.23 / zeff_j) * (f_trap_eff ** 3),
            0.05,
        )
        sigma_neo = sigma_spitzer * neo_fac
        skin_m = np.sqrt(float(current_evolution_time_s) / (_MU0 * np.maximum(sigma_neo, 1.0)))
        j_ls = _exact_ell_psi_from_ell_r(x_j, mg_j.r_min_m, mg_j.a_m, skin_m)
        saw_mask = mg_j.q <= 1.0
        if np.any(saw_mask):
            if np.all(saw_mask):
                j_ls = np.maximum(j_ls, max(float(x_j[-1]), 1.0e-3))
            else:
                i_last = int(np.where(saw_mask)[0][-1])
                if i_last + 1 < x_j.size and mg_j.q[i_last + 1] > mg_j.q[i_last]:
                    frac = (1.0 - float(mg_j.q[i_last])) / float(mg_j.q[i_last + 1] - mg_j.q[i_last])
                    x_q1 = float(x_j[i_last] + np.clip(frac, 0.0, 1.0) * (x_j[i_last + 1] - x_j[i_last]))
                else:
                    x_q1 = float(x_j[i_last])
                if x_q1 > 0.0:
                    ell_q1 = float(np.interp(x_q1, x_j, j_ls))
                    w_q1 = max(0.35 * x_q1, ell_q1, 0.04)
                    g_saw = 0.5 * (1.0 - np.tanh((x_j - x_q1) / w_q1))
                    j_saw = np.maximum(j_ls, x_q1)
                    j_ls = (1.0 - g_saw) * j_ls + g_saw * j_saw
    else:
        j_prior = unc_dict.get("j_ls", 0.25)
        j_ls = np.broadcast_to(np.asarray(j_prior, dtype=float), x_j.shape).copy()

    # Regional core gradient bounds [kappa_min, kappa_max] in R0 / L_s units:
    # Symmetric +-half_w around rl_base guarantees exact zero truncation bias on flat
    # cores while half_w >= rl_base + hollow_allowance guarantees kappa_min <= -hollow_allowance < 0.
    def _bounds_for(rl_base: np.ndarray, dk: np.ndarray, hollow_allowance: float):
        half_w = np.maximum(3.5 * dk, np.maximum(rl_base, 0.0) + hollow_allowance)
        return (rl_base - half_w, rl_base + half_w)

    kappa_bounds = {
        "ne": _bounds_for(rlne_b, dk_ne, hollow_allowance=5.0),
        "ni": _bounds_for(rlni_b, dk_ni, hollow_allowance=5.0),
        "te": _bounds_for(rlte_b, dk_te, hollow_allowance=3.5),
        "ti": _bounds_for(rlti_b, dk_ti, hollow_allowance=2.5),
    }

    n_ls = 1.0 / np.sqrt(0.5 * (1.0 / ne_ls ** 2 + 1.0 / ni_ls ** 2))
    t_ls = 1.0 / np.sqrt(0.5 * (1.0 / te_ls ** 2 + 1.0 / ti_ls ** 2))

    return PhysicsStiffnessResult(
        mode=mode,
        psi_kin=x_kin,
        psi_j=x_j,
        rho_kin=mg_kin.rho_tor_norm.copy(),
        pedestal=ped,
        ne_ls=ne_ls,
        te_ls=te_ls,
        ni_ls=ni_ls,
        ti_ls=ti_ls,
        zeff_ls=zeff_ls,
        j_ls=j_ls,
        n_ls=n_ls,
        t_ls=t_ls,
        rlne_base=rlne_b,
        rlte_base=rlte_b,
        rlni_base=rlni_b,
        rlti_base=rlti_b,
        rlte_crit=rlte_c,
        rlti_crit=rlti_c,
        dkappa_ne=dk_ne,
        dkappa_te=dk_te,
        dkappa_ni=dk_ni,
        dkappa_ti=dk_ti,
        kappa_bounds=kappa_bounds,
        Q_gB_MW_m2=t1["Q_gB_MW_m2"],
        surrogate_meta=surr_meta,
        gibbs_ls_form=str(gibbs_ls_form),
    )
