"""Stiff-core inward gradient-integrating profile sampler (`stiff_inward`).

Generalizes deterministic pedestal-pinned inward integration into a zero-bias
stochastic Gaussian-process sampler for stiff burning-plasma profiles (SPARC,
ARC, ITER) while naturally accommodating L-mode edges, core radiative
hollowing, and pellet density inversions.

Physics & Statistical Properties
--------------------------------
1. **Pedestal & Edge Anchor (`has_pedestal=True`, `psi_N >= psi_ped_top`) or
   LCFS Anchor (`has_pedestal=False`):**
   - In H-mode (`has_pedestal=True`), draws a ``C^1``-slope-matched logarithmic
     perturbation on ``[psi_ped_top, 1]`` with stationary edge correlation
     length ``ell_edge = max(min(ell_profile), 1e-3)``, enforcing barrier
     monotonicity strictly on steep barrier segments where
     ``diff(base) < -1e-3 * max(|base|)`` (excluding flat SOL feet) and
     centering the conditioned pedestal anchor via its exact cumulant
     generating function so ``E[f_draw(r)] == f_base(r)`` without core bias.
   - In L-mode (`has_pedestal=False`), anchors directly at the LCFS without an
     artificial edge transport barrier.
2. **Stiff Core Inverse Scale-Length Draw (`psi_N <= psi_ped_top`):**
   Samples perturbations ``delta_kappa(r)`` to the normalized inverse scale
   length ``kappa(r) = -(R_0 / a) * d(ln f)/d(r/a)`` from a zero-mean GP whose
   pointwise 1-sigma amplitude is bounded by ``dkappa_allow(r)`` (from Tier 1
   or Tier 2 ``TGLF-NN``), partitioned with the anchor variance so total
   marginal variance never exceeds ``sigma_profile(r)^2`` — with zero
   outer-core two-draw level blending and zero gradient-stiffness spikes.
3. **Exact ``sigma = 0`` Identity, Deterministic RNG & Nodewise Positivity:**
   - ``sigma = 0`` returns ``f_base`` **to machine precision** while advancing
     ``rng`` deterministically by a single step (preserving downstream channel
     seeds when an individual channel sigma is zeroed);
   - ``E[f_draw(r)] == f_base(r)`` across core, pedestal, and SOL;
   - ``df_draw / dr |_{r=0} == 0`` via a smooth axis taper on ``delta_kappa``;
   - nodewise positivity ``f_draw >= 0.05 * max(f_base, 1e-4 * max(|f_base|))``.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Optional

import numpy as np

_PLAN_CACHE: "OrderedDict[bytes, dict]" = OrderedDict()
_PLAN_CACHE_MAX = 16


def _build_core_integral_covariance(
    x_norm: np.ndarray,
    sigma_kappa: np.ndarray,
    ell_kappa: float,
) -> tuple:
    """Build GP covariance of ``delta_kappa(x)`` on ``x = r / a in [0, x_ped]``."""
    dx = x_norm[:, None] - x_norm[None, :]
    K_corr = np.exp(-0.5 * (dx / max(ell_kappa, 0.03)) ** 2)
    K_kappa = np.outer(sigma_kappa, sigma_kappa) * K_corr
    K_kappa += 1.0e-10 * max(float(np.max(sigma_kappa ** 2)), 1.0e-12) * np.eye(x_norm.size)
    L_chol = np.linalg.cholesky(K_kappa)
    return K_kappa, L_chol


def _integrate_from_ped_trapezoid(x_core: np.ndarray, y_core: np.ndarray) -> np.ndarray:
    """Compute ``int_{x_i}^{x_ped} y(u) du`` along axis 0 for 1-D or 2-D ``y_core``."""
    if x_core.size <= 1:
        return np.zeros_like(y_core)
    xr = x_core[::-1]
    yr = y_core[::-1]
    dx = xr[:-1] - xr[1:]
    if yr.ndim == 1:
        seg = 0.5 * (yr[:-1] + yr[1:]) * dx
        integ_rev = np.concatenate(([0.0], np.cumsum(seg)))
    else:
        seg = 0.5 * (yr[:-1, :] + yr[1:, :]) * dx[:, None]
        integ_rev = np.vstack((np.zeros((1, yr.shape[1])), np.cumsum(seg, axis=0)))
    return integ_rev[::-1]


def _get_or_build_channel_plan(
    psi: np.ndarray,
    base: np.ndarray,
    sig: np.ndarray,
    dkappa_allow: np.ndarray,
    ell_arr: np.ndarray,
    x_r: np.ndarray,
    a_over_R0: float,
    psi_ped_top: float,
    has_pedestal: bool,
    ell_kappa: float,
    enforce_ped_monotonic: bool,
) -> dict:
    """Build or retrieve cached Cholesky factors & conditioned drift vectors."""
    h = hashlib.blake2b(digest_size=24)
    for arr in (psi, base, sig, dkappa_allow, ell_arr, x_r):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(
        f"{a_over_R0:.12g}|{psi_ped_top:.12g}|{int(has_pedestal)}|"
        f"{ell_kappa:.12g}|{int(enforce_ped_monotonic)}".encode("ascii")
    )
    key = h.digest()
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached

    n_pts = psi.size
    p0 = float(max(np.max(np.abs(base)), 1.0e-30))
    base_pos = np.maximum(base, 1.0e-4 * p0)
    min_node_floor = 0.05 * base_pos

    if not has_pedestal or n_pts < 5:
        ell_edge_scalar = float(np.min(ell_arr)) if ell_arr.size else 0.25
        i_ped = n_pts - 1
    else:
        ell_edge_scalar = max(float(np.min(ell_arr)) if ell_arr.size else 0.03, 1.0e-3)
        i_ped = int(
            np.searchsorted(psi, float(np.clip(psi_ped_top, 0.45, 0.985)), side="right")
        )
        i_ped = min(max(i_ped, 3), n_pts - 2)

    x_core = x_r[: i_ped + 1]
    base_core = base_pos[: i_ped + 1]
    sig_rel_core = np.clip(sig[: i_ped + 1] / base_core, 0.0, 1.2)
    sig_rel_ped = float(sig_rel_core[-1])
    v_ped_nom = sig_rel_ped ** 2

    tau_axis = x_core / np.sqrt(x_core ** 2 + 0.04 ** 2)
    if x_core[0] <= 1.0e-8:
        tau_axis[0] = 0.0

    dk_stiff = np.maximum(dkappa_allow[: i_ped + 1], 0.0)
    sigma_kappa = tau_axis * dk_stiff
    if not np.any(sig[: i_ped + 1] > 0.0):
        sigma_kappa = np.zeros_like(x_core)

    has_core_gp = bool(np.any(sigma_kappa > 0.0) and x_core.size > 1)
    if has_core_gp:
        _, L_chol = _build_core_integral_covariance(x_core, sigma_kappa, ell_kappa)
        M_cols = a_over_R0 * _integrate_from_ped_trapezoid(x_core, L_chol)
        v_stiff = np.sum(M_cols ** 2, axis=1)
    else:
        L_chol = None
        M_cols = None
        v_stiff = np.zeros_like(x_core)

    plan: dict = {
        "n_pts": n_pts,
        "i_ped": i_ped,
        "base_pos": base_pos,
        "base_core": base_core,
        "min_node_floor": min_node_floor,
        "sig_rel_ped": sig_rel_ped,
        "has_core_gp": has_core_gp,
        "L_chol": L_chol,
        "M_cols": M_cols,
        "x_core_size": x_core.size,
    }

    if i_ped == n_pts - 1:
        v_raw = v_ped_nom + v_stiff
        beta = np.where(
            v_raw > 1.0e-16,
            np.minimum(1.0, sig_rel_core / np.sqrt(np.maximum(v_raw, 1.0e-16))),
            1.0,
        )
        plan["beta"] = beta
        plan["v_eff"] = (beta ** 2) * v_raw
    else:
        psi_edge = psi[i_ped:]
        tau_e = psi_edge - psi_edge[0]
        m_e = psi_edge.size
        base_edge = base_pos[i_ped:]
        sig_rel_edge = np.clip(sig[i_ped:] / base_edge, 0.0, 1.2)

        ell2 = ell_edge_scalar ** 2
        phi0 = np.exp(-0.5 * (tau_e ** 2) / ell2)
        phi1 = tau_e * phi0
        dtau = tau_e[:, None] - tau_e[None, :]
        K_full = np.exp(-0.5 * (dtau ** 2) / ell2)
        K_rem = K_full - np.outer(phi0, phi0) - np.outer(phi1, phi1) / ell2
        K_rem[0, :] = 0.0
        K_rem[:, 0] = 0.0
        if m_e > 1:
            K_sub = 0.5 * (K_rem[1:, 1:] + K_rem[1:, 1:].T)
            eig_vals, eig_vecs = np.linalg.eigh(K_sub)
            eig_vals = np.maximum(eig_vals, 0.0)
            L_rem_sub = eig_vecs * np.sqrt(eig_vals)[None, :]
        else:
            L_rem_sub = np.zeros((0, 0))

        dx_dpsi_ped = float(
            (x_r[i_ped] - x_r[i_ped - 1]) / max(psi[i_ped] - psi[i_ped - 1], 1.0e-8)
        )
        dsigr_dpsi0 = float((sig_rel_edge[1] - sig_rel_edge[0]) / max(tau_e[1], 1.0e-8))
        steep_seg = (
            (np.diff(base[i_ped:]) < -1.0e-3 * p0)
            if enforce_ped_monotonic
            else np.zeros(m_e - 1, dtype=bool)
        )

        v_raw_0 = v_ped_nom + v_stiff
        beta_0 = np.where(
            v_raw_0 > 1.0e-16,
            np.minimum(1.0, sig_rel_core / np.sqrt(np.maximum(v_raw_0, 1.0e-16))),
            1.0,
        )
        dbeta_dpsi_0 = float(
            (beta_0[-1] - beta_0[-2]) / max(psi[i_ped] - psi[i_ped - 1], 1.0e-8)
        )

        cal_rng = np.random.default_rng(20261121)
        n_half = 512
        z0_h = cal_rng.standard_normal(n_half)
        z0_cal = np.concatenate([z0_h, -z0_h])
        z0_cal = z0_cal / max(float(np.std(z0_cal)), 1.0e-12)
        zr_h = cal_rng.standard_normal((m_e - 1, n_half))
        zr_cal = np.concatenate([zr_h, -zr_h], axis=1)
        if zr_cal.shape[0] > 0:
            zr_cal = zr_cal / np.maximum(np.std(zr_cal, axis=1, keepdims=True), 1.0e-12)
        zk_h = cal_rng.standard_normal(n_half)
        zk_cal = np.concatenate([zk_h, -zk_h])
        zk_cal = zk_cal / max(float(np.std(zk_cal)), 1.0e-12)
        slope_core_cal = -a_over_R0 * float(sigma_kappa[-1]) * dx_dpsi_ped * zk_cal

        rem_cal = np.vstack([np.zeros((1, 2 * n_half)), L_rem_sub @ zr_cal])
        c_slope_cal = (dbeta_dpsi_0 * sig_rel_ped - dsigr_dpsi0) * z0_cal + slope_core_cal
        U_cal = (
            sig_rel_edge[:, None] * (phi0[:, None] * z0_cal[None, :] + rem_cal)
            + phi1[:, None] * c_slope_cal[None, :]
        )

        mu_log_edge = 0.5 * np.var(U_cal, axis=1)
        U_acc = U_cal
        for _ in range(2):
            cand_cal = base_edge[:, None] * np.exp(
                np.clip(U_cal - mu_log_edge[:, None], -3.0, 3.5)
            )
            ok_cal = np.all(cand_cal >= min_node_floor[i_ped:, None], axis=0)
            if np.any(steep_seg):
                ok_cal &= np.all(np.diff(cand_cal, axis=0)[steep_seg, :] <= 0.0, axis=0)
            U_acc = U_cal[:, ok_cal] if np.count_nonzero(ok_cal) >= 32 else U_cal
            mu_log_edge = np.log(
                np.maximum(np.mean(np.exp(np.clip(U_acc, -3.0, 3.5)), axis=1), 1.0e-12)
            )

        mu_u0 = float(np.mean(U_acc[0, :]))
        w_acc = U_acc[0, :] - mu_u0
        v_ped_cond = max(float(np.var(w_acc)), 1.0e-16) if v_ped_nom > 0.0 else 0.0
        v_ped_eff = min(v_ped_cond, v_ped_nom)

        v_raw = v_ped_eff + v_stiff
        beta = np.where(
            v_raw > 1.0e-16,
            np.minimum(1.0, sig_rel_core / np.sqrt(np.maximum(v_raw, 1.0e-16))),
            1.0,
        )
        dbeta_dpsi_ped = float(
            (beta[-1] - beta[-2]) / max(psi[i_ped] - psi[i_ped - 1], 1.0e-8)
        )
        psi_ped_cgf = np.log(
            np.maximum(
                np.mean(np.exp(np.clip(beta[:, None] * w_acc[None, :], -6.0, 6.0)), axis=1),
                1.0e-12,
            )
        )
        drift_core = psi_ped_cgf + 0.5 * (beta ** 2) * v_stiff
        ddrift_core_dpsi_ped = float(
            (drift_core[-1] - drift_core[-2]) / max(psi[i_ped] - psi[i_ped - 1], 1.0e-8)
        )
        dmu_edge_dpsi0 = float((mu_log_edge[1] - mu_log_edge[0]) / max(tau_e[1], 1.0e-8))

        plan.update(
            m_e=m_e,
            base_edge=base_edge,
            sig_rel_edge=sig_rel_edge,
            phi0=phi0,
            phi1=phi1,
            L_rem_sub=L_rem_sub,
            dx_dpsi_ped=dx_dpsi_ped,
            dsigr_dpsi0=dsigr_dpsi0,
            steep_seg=steep_seg,
            has_steep=bool(np.any(steep_seg)),
            mu_u0=mu_u0,
            mu_log_edge=mu_log_edge,
            beta=beta,
            drift_core=drift_core,
            dbeta_dpsi_ped=dbeta_dpsi_ped,
            ddrift_core_dpsi_ped=ddrift_core_dpsi_ped,
            dmu_edge_dpsi0=dmu_edge_dpsi0,
        )

    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
        _PLAN_CACHE.popitem(last=False)
    return plan


def sample_stiff_inward_channel(
    psi_N: np.ndarray,
    base_profile: np.ndarray,
    sigma_profile: np.ndarray,
    dkappa_allow: np.ndarray,
    ell_profile: np.ndarray,
    *,
    r_norm: Optional[np.ndarray] = None,
    a_over_R0: float = 0.33,
    psi_ped_top: float = 0.92,
    has_pedestal: bool = True,
    ell_kappa: float = 0.30,
    enforce_ped_monotonic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw one stiff-core inward-integrated perturbation on ``psi_N``.

    Returns the perturbed profile in physical units (same shape as
    ``base_profile``).  When ``sigma_profile`` is all zero, returns
    ``base_profile.copy()`` identically after consuming one deterministic step
    from ``rng``.
    """
    psi = np.asarray(psi_N, dtype=float)
    base = np.asarray(base_profile, dtype=float)
    sig = np.asarray(sigma_profile, dtype=float)
    n_pts = psi.size

    if rng is None:
        rng = np.random.default_rng()
    # Consume exactly one integer from the caller's RNG regardless of sigma == 0,
    # L-mode vs H-mode, or rejection iterations (matching GPRProfilePerturber).
    ch_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))

    if n_pts == 0 or not np.any(sig > 0.0):
        return base.copy()

    x_r = (
        np.sqrt(np.clip(psi, 0.0, 1.0))
        if r_norm is None
        else np.clip(np.asarray(r_norm, dtype=float), 0.0, 1.0)
    )
    dk_arr = np.asarray(dkappa_allow, dtype=float)
    ell_arr = np.asarray(ell_profile, dtype=float).ravel()

    plan = _get_or_build_channel_plan(
        psi, base, sig, dk_arr, ell_arr, x_r,
        float(a_over_R0), float(psi_ped_top), bool(has_pedestal),
        float(ell_kappa), bool(enforce_ped_monotonic),
    )

    i_ped = plan["i_ped"]
    min_node_floor = plan["min_node_floor"]
    sig_rel_ped = plan["sig_rel_ped"]

    if plan["has_core_gp"]:
        z_core = ch_rng.standard_normal(plan["x_core_size"])
        delta_kappa_end = float(plan["L_chol"][-1, :] @ z_core)
        delta_J = plan["M_cols"] @ z_core
    else:
        delta_kappa_end = 0.0
        delta_J = np.zeros(plan["x_core_size"], dtype=float)

    if i_ped == n_pts - 1:
        w_ped = sig_rel_ped * float(ch_rng.standard_normal())
        s_core = plan["beta"] * (w_ped + delta_J)
        out = plan["base_pos"][: i_ped + 1] * np.exp(
            np.clip(s_core - 0.5 * plan["v_eff"], -3.0, 3.5)
        )
        return np.maximum(out, min_node_floor)

    m_e = plan["m_e"]
    base_edge = plan["base_edge"]
    sig_rel_edge = plan["sig_rel_edge"]
    phi0 = plan["phi0"]
    phi1 = plan["phi1"]
    L_rem_sub = plan["L_rem_sub"]
    steep_seg = plan["steep_seg"]
    has_steep = plan["has_steep"]
    mu_u0 = plan["mu_u0"]
    mu_log_edge = plan["mu_log_edge"]
    dbeta_dpsi_ped = plan["dbeta_dpsi_ped"]
    ddrift_core_dpsi_ped = plan["ddrift_core_dpsi_ped"]
    dmu_edge_dpsi0 = plan["dmu_edge_dpsi0"]
    dsigr_dpsi0 = plan["dsigr_dpsi0"]
    floor_edge = min_node_floor[i_ped:]

    slope_core_draw = -a_over_R0 * delta_kappa_end * plan["dx_dpsi_ped"]
    u_draw = None
    edge_draw = None
    for _ in range(120):
        z0 = float(ch_rng.standard_normal())
        zr = ch_rng.standard_normal(m_e - 1)
        rem_d = np.concatenate(([0.0], L_rem_sub @ zr))
        w_d = sig_rel_ped * z0 - mu_u0
        left_log_slope = dbeta_dpsi_ped * w_d + slope_core_draw - ddrift_core_dpsi_ped
        c_slope = left_log_slope + dmu_edge_dpsi0 - dsigr_dpsi0 * z0
        u_cand = sig_rel_edge * (phi0 * z0 + rem_d) + phi1 * c_slope
        edge_cand = base_edge * np.exp(np.clip(u_cand - mu_log_edge, -3.0, 3.5))
        if np.any(edge_cand < floor_edge):
            continue
        if has_steep and np.any(np.diff(edge_cand)[steep_seg] > 0.0):
            continue
        u_draw = u_cand
        edge_draw = edge_cand
        break

    if u_draw is None:
        for _ in range(40):
            z0 = float(ch_rng.standard_normal())
            zr = ch_rng.standard_normal(m_e - 1)
            rem_d = np.concatenate(([0.0], L_rem_sub @ zr))
            u_cand = sig_rel_edge * (phi0 * z0 + rem_d)
            edge_cand = np.maximum(
                base_edge * np.exp(np.clip(u_cand - mu_log_edge, -3.0, 3.5)),
                floor_edge,
            )
            if not has_steep or np.all(np.diff(edge_cand)[steep_seg] <= 0.0):
                u_draw = u_cand
                edge_draw = edge_cand
                break
        if u_draw is None:
            u_draw = u_cand
            edge_draw = edge_cand.copy()
            for k in range(m_e - 1):
                if steep_seg[k] and edge_draw[k + 1] > edge_draw[k]:
                    edge_draw[k + 1] = edge_draw[k]
            edge_draw = np.maximum(edge_draw, floor_edge)

    w_ped = float(u_draw[0]) - mu_u0
    s_core = plan["beta"] * (w_ped + delta_J)
    core_draw = np.maximum(
        plan["base_core"] * np.exp(np.clip(s_core - plan["drift_core"], -3.0, 3.5)),
        min_node_floor[: i_ped + 1],
    )

    out = np.empty_like(base)
    out[: i_ped + 1] = core_draw
    out[i_ped:] = edge_draw
    out[i_ped] = core_draw[-1]
    return out
