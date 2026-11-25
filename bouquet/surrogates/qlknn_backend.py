"""Pure-NumPy ``QLKNN`` (QuaLiKiz Neural Network) electrostatic surrogate adapter.

Evaluates ``QLKNN-hyper-10D`` JSON MLP weight committees (van de Plassche et al.,
Phys. Plasmas 27, 022310, 2020) across ITG, TEM, and ETG branches (including
``_div_`` ratio networks), converts QuaLiKiz gyro-Bohm fluxes into GACODE
``Q_gB_imas_W_m2`` units, and emits an explicit ``UserWarning`` when evaluated
in finite-``beta_e`` regimes (``beta_e > 0.005``) where electromagnetic
``TGLF-NN`` (SAT2/SAT3 EM) should be preferred.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Optional

import numpy as np

from ..physics import ELEMENTARY_CHARGE

# CODATA 2018 SI constants
_EPS0 = 8.8541878128e-12
_ME_KG = 9.1093837015e-31
_MP_KG = 1.67262192369e-27
_MU0 = 4.0e-7 * np.pi


class QLKNNJSONModel:
    """Pure-NumPy evaluator for ``QLKNN-hyper-10D`` JSON MLP files."""

    def __init__(self, json_path: str):
        with open(json_path, "r") as fh:
            spec = json.load(fh)
        self.feature_names = list(spec["feature_names"])
        self.target_names = list(spec["target_names"])
        self._f_min = np.asarray([spec["feature_min"][k] for k in self.feature_names], dtype=float)
        self._f_max = np.asarray([spec["feature_max"][k] for k in self.feature_names], dtype=float)
        self._f_scale = np.asarray([spec["settings"]["feature_prescale_factor"][k] for k in self.feature_names], dtype=float)
        self._f_bias = np.asarray([spec["settings"]["feature_prescale_bias"][k] for k in self.feature_names], dtype=float)
        self._t_scale = np.asarray([spec["settings"]["target_prescale_factor"][k] for k in self.target_names], dtype=float)
        self._t_bias = np.asarray([spec["settings"]["target_prescale_bias"][k] for k in self.target_names], dtype=float)

        self.layers = []
        idx = 1
        while f"layer{idx}/weights:0" in spec or f"hidden{idx}/weights:0" in spec:
            pfx = f"layer{idx}" if f"layer{idx}/weights:0" in spec else f"hidden{idx}"
            W = np.asarray(spec[f"{pfx}/weights:0"], dtype=float)
            b = np.asarray(spec[f"{pfx}/biases:0"], dtype=float)
            act = (
                spec["hidden_activation"][idx - 1]
                if idx - 1 < len(spec.get("hidden_activation", []))
                else spec.get("output_activation", "none")
            )
            self.layers.append((W, b, act))
            idx += 1
        if "output/weights:0" in spec:
            self.layers.append((
                np.asarray(spec["output/weights:0"], dtype=float),
                np.asarray(spec["output/biases:0"], dtype=float),
                spec.get("output_activation", "none"),
            ))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=float)
        squeeze_out = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            squeeze_out = True
        h = x_arr * self._f_scale[None, :] + self._f_bias[None, :]
        for W, b, act in self.layers:
            h = h @ W + b
            act_norm = "none" if act is None else str(act).strip().lower()
            if act_norm in ("none", "", "linear", "identity"):
                pass
            elif act_norm == "tanh":
                h = np.tanh(h)
            elif act_norm == "relu":
                h = np.maximum(h, 0.0)
            elif act_norm == "elu":
                neg = np.minimum(h, 0.0)
                h = np.where(h > 0.0, h, np.expm1(np.maximum(neg, -80.0)))
            elif act_norm == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -80.0, 80.0)))
            elif act_norm == "softplus":
                h = np.log1p(np.exp(-np.abs(h))) + np.maximum(h, 0.0)
            else:
                raise ValueError(f"Unsupported QLKNN activation {act!r}")
        out = (h - self._t_bias[None, :]) / np.where(
            self._t_scale[None, :] != 0.0, self._t_scale[None, :], 1.0
        )
        return out[0] if squeeze_out else out


def z_safe(zeff) -> np.ndarray:
    return np.maximum(np.asarray(zeff, dtype=float), 1.0)


def _resolve_qlknn_dir(model_dir: Optional[str]) -> str:
    candidates = []
    if model_dir:
        candidates.append(os.path.expanduser(model_dir))
    env_dir = os.environ.get("BOUQUET_QLKNN_DIR")
    if env_dir:
        candidates.append(os.path.expanduser(env_dir))
    candidates.append(os.path.expanduser("~/.cache/bouquet/qlknn"))
    candidates.append(os.path.expanduser("~/.cache/bouquet/models/qlknn_hyper_10d"))
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "No QLKNN JSON network directory found. Pass model_dir=... or set "
        "BOUQUET_QLKNN_DIR to a directory of QLKNN-hyper-10D JSON weights "
        "(or place them in ~/.cache/bouquet/qlknn). For zero-config surrogate "
        "evaluation, use length_scale_mode='tier2_tglfnn'."
    )


def evaluate_qlknn_stiffness(
    miller,
    psi_N: np.ndarray,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    zeff: np.ndarray,
    *,
    Z_imp: float = 6.0,
    omega_tor_rad_s: Optional[np.ndarray] = None,
    model_dir: Optional[str] = None,
    q_crit_gb_threshold: float = 0.08,
) -> Dict[str, np.ndarray]:
    """Evaluate QLKNN 10D fluxes, stiffness slopes, and critical gradients in GACODE GB units."""
    from ..geometry.miller import _safe_gradient, build_tglf_inputs, gyrobohm_flux_scales

    mg = miller.interpolate(psi_N)
    inp = build_tglf_inputs(
        mg, psi_N, ne_m3, te_eV, ni_m3, ti_eV, zeff,
        Z_imp=Z_imp, omega_tor_rad_s=omega_tor_rad_s,
    )
    core_mask = mg.rho_tor_norm <= 0.85
    max_betae = float(np.max(inp["BETAE"][core_mask] if np.any(core_mask) else inp["BETAE"]))
    if max_betae > 5.0e-3:
        warnings.warn(
            f"QLKNN electrostatic surrogate requested on a plasma with core "
            f"beta_e={max_betae:.4f} > 0.005. Electromagnetic stabilization "
            "(ITG/KBM) is neglected in electrostatic QLKNN; prefer "
            "length_scale_mode='tier2_tglfnn' for SPARC/ARC/high-beta regimes.",
            UserWarning,
            stacklevel=2,
        )

    R_over_a = float(mg.R0_m / max(mg.a_m, 1e-4))
    ne_safe = np.maximum(np.asarray(ne_m3, dtype=float), 1.0e17)
    ni_safe = np.maximum(np.asarray(ni_m3, dtype=float), 1.0e15)
    te_safe = np.maximum(np.asarray(te_eV, dtype=float), 10.0)
    ti_safe = np.maximum(np.asarray(ti_eV, dtype=float), 10.0)
    zeff_safe = z_safe(zeff)

    # Exact CODATA Braginskii electron collision frequency nu_e = 1 / tau_e [s^-1]
    eps = np.maximum(mg.r_min_m / np.maximum(mg.R_maj_m, 0.1), 1e-3)
    ne_19 = ne_safe * 1.0e-19
    te_keV = te_safe * 1.0e-3
    ln_lam = np.clip(15.2 - 0.5 * np.log(ne_19 / 10.0) + np.log(te_keV), 8.0, 25.0)
    tau_e = (
        12.0 * (np.pi ** 1.5) * (_EPS0 ** 2) * np.sqrt(_ME_KG)
        * ((ELEMENTARY_CHARGE * te_safe) ** 1.5)
    ) / (
        np.sqrt(2.0) * ne_safe * zeff_safe * (ELEMENTARY_CHARGE ** 4) * ln_lam
    )
    nu_e = 1.0 / np.maximum(tau_e, 1.0e-30)
    v_the = np.sqrt(ELEMENTARY_CHARGE * te_safe / _ME_KG)
    nu_star = np.maximum(nu_e * mg.q * mg.R_maj_m / ((eps ** 1.5) * v_the), 1.0e-6)

    # Optional 10D/11D QuaLiKiz rotation & MHD-alpha features
    c_ref_qlk = np.sqrt(ELEMENTARY_CHARGE * 1.0e3 / _MP_KG)
    r0_safe = max(float(mg.R0_m), 1.0e-3)
    b0_safe = max(abs(float(mg.B0_T)), 1.0e-4)
    a_safe = max(float(mg.a_m), 1.0e-4)
    if omega_tor_rad_s is not None:
        w_tor = np.asarray(omega_tor_rad_s, dtype=float)
        dw_dr = _safe_gradient(w_tor, mg.r_min_m)
        q_signed = (-1.0 if float(mg.sign_q) < 0.0 else 1.0) * mg.q
        gamma_e_qlk = -(mg.r_min_m / q_signed) * dw_dr * (r0_safe / c_ref_qlk)
        mach_tor = mg.R_maj_m * w_tor / c_ref_qlk
        au_tor = -(r0_safe / c_ref_qlk) * mg.R_maj_m * dw_dr
    else:
        gamma_e_qlk = np.zeros_like(mg.psi_N)
        mach_tor = np.zeros_like(mg.psi_N)
        au_tor = np.zeros_like(mg.psi_N)

    p_tot_Pa = ELEMENTARY_CHARGE * (ne_safe * te_safe + ni_safe * ti_safe)
    dp_dr = _safe_gradient(p_tot_Pa, mg.r_min_m)
    alpha_mhd = np.maximum(-(mg.q ** 2) * r0_safe * (2.0 * _MU0 / (b0_safe ** 2)) * dp_dr, 0.0)

    features_10d: Dict[str, np.ndarray] = {
        "Zeff": np.clip(zeff_safe, 1.0, 3.0),
        "Ati": inp["RLTS_2"] * R_over_a,
        "Ate": inp["RLTS_1"] * R_over_a,
        "An": inp["RLNS_1"] * R_over_a,
        "Ani": inp["RLNS_2"] * R_over_a,
        "Normni": np.clip(ni_safe / ne_safe, 0.1, 1.0),
        "q": np.clip(mg.q, 0.66, 15.0),
        "smag": np.clip(mg.s_hat, -1.0, 5.0),
        "x": np.clip(mg.r_min_m / a_safe, 0.05, 0.99),
        "Ti_Te": np.clip(ti_safe / te_safe, 0.25, 2.5),
        "logNustar": np.log10(nu_star),
        "gammaE": np.clip(gamma_e_qlk, -2.0, 2.0),
        "Machtor": mach_tor,
        "Autor": au_tor,
        "alpha": np.clip(alpha_mhd, 0.0, 5.0),
    }

    # Exact conversion from QuaLiKiz gyro-Bohm heat flux units
    #   q_{s,GB}^{QLK} = n_s (e T_s) sqrt(m_p) (e T_e)^(3/2) / (e^2 B_0^2 a R_0)
    # into GACODE surface gyro-Bohm units Q_gB_imas_W_m2 = Q_gB_W_m2 * vprime_miller.
    gb = gyrobohm_flux_scales(mg, ne_safe, te_safe)
    q_gb_gacode_imas = np.maximum(gb["Q_gB_imas_W_m2"], 1.0e-30)
    chi_gb_qlk = (
        np.sqrt(_MP_KG) * ((ELEMENTARY_CHARGE * te_safe) ** 1.5)
        / ((ELEMENTARY_CHARGE ** 2) * (b0_safe ** 2) * a_safe)
    )
    q_gb_qlk_e = ne_safe * (ELEMENTARY_CHARGE * te_safe) * chi_gb_qlk / r0_safe
    q_gb_qlk_i = ni_safe * (ELEMENTARY_CHARGE * ti_safe) * chi_gb_qlk / r0_safe
    conv_e = q_gb_qlk_e / q_gb_gacode_imas
    conv_i = q_gb_qlk_i / q_gb_gacode_imas

    resolved_dir = _resolve_qlknn_dir(model_dir)
    return _eval_from_json_dir(
        resolved_dir,
        features_10d,
        conv_e=conv_e,
        conv_i=conv_i,
        q_gb_gacode_imas=q_gb_gacode_imas,
        q_crit_gb_threshold=q_crit_gb_threshold,
    )


def _load_optional_net(model_dir: str, filename: str) -> Optional[QLKNNJSONModel]:
    p = os.path.join(model_dir, filename)
    return QLKNNJSONModel(p) if os.path.isfile(p) else None


def _eval_net(net: QLKNNJSONModel, feat: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.column_stack([feat[k] for k in net.feature_names])
    y = net(X)
    return y[:, 0] if y.ndim == 2 else y


def _eval_from_json_dir(
    model_dir: str,
    feat: Dict[str, np.ndarray],
    *,
    conv_e: np.ndarray,
    conv_i: np.ndarray,
    q_gb_gacode_imas: np.ndarray,
    q_crit_gb_threshold: float = 0.08,
) -> Dict[str, np.ndarray]:
    """Sum ITG + TEM + ETG QLKNN committees (including _div_ ratio networks) and scan thresholds."""
    nets = {
        "efi_itg": _load_optional_net(model_dir, "efiITG_GB.json"),
        "efe_itg": _load_optional_net(model_dir, "efeITG_GB.json"),
        "efe_div_efi_itg": _load_optional_net(model_dir, "efeITG_GB_div_efiITG_GB.json"),
        "efi_div_efe_itg": _load_optional_net(model_dir, "efiITG_GB_div_efeITG_GB.json"),
        "efe_tem": _load_optional_net(model_dir, "efeTEM_GB.json"),
        "efi_tem": _load_optional_net(model_dir, "efiTEM_GB.json"),
        "efi_div_efe_tem": _load_optional_net(model_dir, "efiTEM_GB_div_efeTEM_GB.json"),
        "efe_div_efi_tem": _load_optional_net(model_dir, "efeTEM_GB_div_efiTEM_GB.json"),
        "efe_etg": _load_optional_net(model_dir, "efeETG_GB.json"),
        "efe_tot": _load_optional_net(model_dir, "efe_GB.json"),
        "efi_tot": _load_optional_net(model_dir, "efi_GB.json"),
    }

    has_e = any(
        nets[k] is not None
        for k in ("efe_itg", "efe_tem", "efe_etg", "efe_tot")
    ) or (nets["efe_div_efi_itg"] is not None and nets["efi_itg"] is not None) or (
        nets["efe_div_efi_tem"] is not None and nets["efi_tem"] is not None
    )
    has_i = any(
        nets[k] is not None
        for k in ("efi_itg", "efi_tem", "efi_tot")
    ) or (nets["efi_div_efe_tem"] is not None and nets["efe_tem"] is not None) or (
        nets["efi_div_efe_itg"] is not None and nets["efe_itg"] is not None
    )
    if not (has_e and has_i):
        raise FileNotFoundError(
            f"Expected at least one electron (e.g. 'efeETG_GB.json', 'efeTEM_GB.json', "
            f"'efeITG_GB.json') and one ion (e.g. 'efiITG_GB.json', 'efiTEM_GB.json') "
            f"QLKNN JSON network in '{model_dir}'"
        )

    def _predict_qlk_fluxes(f_dict: Dict[str, np.ndarray]) -> tuple:
        n_rows = f_dict["Ate"].size
        qe = np.zeros(n_rows, dtype=float)
        qi = np.zeros(n_rows, dtype=float)

        # ITG branch
        qi_itg = np.maximum(_eval_net(nets["efi_itg"], f_dict), 0.0) if nets["efi_itg"] is not None else None
        qe_itg = np.maximum(_eval_net(nets["efe_itg"], f_dict), 0.0) if nets["efe_itg"] is not None else None
        if qe_itg is None and qi_itg is not None and nets["efe_div_efi_itg"] is not None:
            qe_itg = np.maximum(_eval_net(nets["efe_div_efi_itg"], f_dict) * qi_itg, 0.0)
        if qi_itg is None and qe_itg is not None and nets["efi_div_efe_itg"] is not None:
            qi_itg = np.maximum(_eval_net(nets["efi_div_efe_itg"], f_dict) * qe_itg, 0.0)
        if qe_itg is not None:
            qe += qe_itg
        if qi_itg is not None:
            qi += qi_itg

        # TEM branch
        qe_tem = np.maximum(_eval_net(nets["efe_tem"], f_dict), 0.0) if nets["efe_tem"] is not None else None
        qi_tem = np.maximum(_eval_net(nets["efi_tem"], f_dict), 0.0) if nets["efi_tem"] is not None else None
        if qi_tem is None and qe_tem is not None and nets["efi_div_efe_tem"] is not None:
            qi_tem = np.maximum(_eval_net(nets["efi_div_efe_tem"], f_dict) * qe_tem, 0.0)
        if qe_tem is None and qi_tem is not None and nets["efe_div_efi_tem"] is not None:
            qe_tem = np.maximum(_eval_net(nets["efe_div_efi_tem"], f_dict) * qi_tem, 0.0)
        if qe_tem is not None:
            qe += qe_tem
        if qi_tem is not None:
            qi += qi_tem

        # ETG branch (electron heat transport only)
        if nets["efe_etg"] is not None:
            qe += np.maximum(_eval_net(nets["efe_etg"], f_dict), 0.0)

        # Combined total-flux networks if supplied instead of mode-split networks
        if nets["efe_tot"] is not None and qe_itg is None and qe_tem is None and nets["efe_etg"] is None:
            qe += np.maximum(_eval_net(nets["efe_tot"], f_dict), 0.0)
        if nets["efi_tot"] is not None and qi_itg is None and qi_tem is None:
            qi += np.maximum(_eval_net(nets["efi_tot"], f_dict), 0.0)

        return qe, qi

    m_pts = feat["Ate"].size
    qe0_qlk, qi0_qlk = _predict_qlk_fluxes(feat)
    qe0 = qe0_qlk * conv_e
    qi0 = qi0_qlk * conv_i

    # Vectorized 25-point threshold scan across Ate / Ati in [0.2, max(15.0, 1.3 * A_base)]
    # to find true critical-gradient crossings even when baseline profiles are subcritical/hollow.
    n_scan = 25
    u_scan = np.linspace(0.0, 1.0, n_scan)[None, :]
    hi_ate = np.maximum(15.0, 1.3 * feat["Ate"])[:, None]
    hi_ati = np.maximum(15.0, 1.3 * feat["Ati"])[:, None]
    grid_ate = 0.2 + (hi_ate - 0.2) * u_scan
    grid_ati = 0.2 + (hi_ati - 0.2) * u_scan

    feat_scan_e = {k: np.repeat(v, n_scan) for k, v in feat.items()}
    feat_scan_e["Ate"] = grid_ate.ravel()
    qe_scan_qlk, _ = _predict_qlk_fluxes(feat_scan_e)
    qe_curve = qe_scan_qlk.reshape(m_pts, n_scan) * conv_e[:, None]

    feat_scan_i = {k: np.repeat(v, n_scan) for k, v in feat.items()}
    feat_scan_i["Ati"] = grid_ati.ravel()
    _, qi_scan_qlk = _predict_qlk_fluxes(feat_scan_i)
    qi_curve = qi_scan_qlk.reshape(m_pts, n_scan) * conv_i[:, None]

    def _find_crit(grid_A: np.ndarray, q_curve: np.ndarray) -> np.ndarray:
        rl_crit = np.empty(m_pts, dtype=float)
        for i in range(m_pts):
            above = np.where(q_curve[i] >= q_crit_gb_threshold)[0]
            if above.size > 0 and above[0] > 0:
                k = int(above[0])
                q_lo, q_hi = float(q_curve[i, k - 1]), float(q_curve[i, k])
                x_lo, x_hi = float(grid_A[i, k - 1]), float(grid_A[i, k])
                w = np.clip((q_crit_gb_threshold - q_lo) / max(q_hi - q_lo, 1e-9), 0.0, 1.0)
                rl_crit[i] = x_lo + w * (x_hi - x_lo)
            elif above.size == 0:
                rl_crit[i] = float(grid_A[i, -1])
            else:
                dq_end = (float(q_curve[i, -1] - q_curve[i, -2])
                          / max(float(grid_A[i, -1] - grid_A[i, -2]), 1e-9))
                rl_crit[i] = max(0.5, float(grid_A[i, -1]) - float(q_curve[i, -1]) / max(dq_end, 0.1))
        return np.clip(rl_crit, 0.5, 25.0)

    rlte_crit = _find_crit(grid_ate, qe_curve)
    rlti_crit = _find_crit(grid_ati, qi_curve)

    # Stiffness derivatives dQ / d(R0/L_Ts) in GACODE Q_gB_imas units.
    # Evaluate around max(A_base, A_crit) when subcritical so stiffness reflects
    # the supercritical turbulent slope rather than a subcritical zero plateau.
    h = 0.25
    ate_eval = np.where(qe0 > 1.0e-6, feat["Ate"], np.maximum(feat["Ate"], rlte_crit))
    ati_eval = np.where(qi0 > 1.0e-6, feat["Ati"], np.maximum(feat["Ati"], rlti_crit))

    f_ep = dict(feat, Ate=ate_eval + h)
    f_em = dict(feat, Ate=ate_eval - np.where(qe0 > 1.0e-6, h, 0.0))
    qe_p_qlk, _ = _predict_qlk_fluxes(f_ep)
    qe_m_qlk, _ = _predict_qlk_fluxes(f_em)
    step_e = np.where(qe0 > 1.0e-6, 2.0 * h, h)
    dqe = np.maximum(((qe_p_qlk - qe_m_qlk) / step_e) * conv_e, 1.0e-3)

    f_ip = dict(feat, Ati=ati_eval + h)
    f_im = dict(feat, Ati=ati_eval - np.where(qi0 > 1.0e-6, h, 0.0))
    _, qi_p_qlk = _predict_qlk_fluxes(f_ip)
    _, qi_m_qlk = _predict_qlk_fluxes(f_im)
    step_i = np.where(qi0 > 1.0e-6, 2.0 * h, h)
    dqi = np.maximum(((qi_p_qlk - qi_m_qlk) / step_i) * conv_i, 1.0e-3)

    return {
        "Qe_gB": qe0,
        "Qi_gB": qi0,
        "Qe_W_m2": qe0 * q_gb_gacode_imas,
        "Qi_W_m2": qi0 * q_gb_gacode_imas,
        "dQe_dRLTe_gB": dqe,
        "dQi_dRLTi_gB": dqi,
        "rlte_crit": rlte_crit,
        "rlti_crit": rlti_crit,
    }
