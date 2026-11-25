"""Local flux-surface Miller parameterization and GACODE/TGLF input builder.

Extracts radial Miller shaping profiles (Miller et al., Phys. Plasmas 5, 973,
1998; Candy, Plasma Phys. Control. Fusion 51, 105009, 2009) and dimensionless
TGLF / QuaLiKiz local parameters directly from:

1. IMAS/OMAS ``dd`` dictionaries or JSON paths (:func:`miller_from_imas_dd`),
2. GEQDSK g-files, raw bytes, or :class:`~bouquet.io.geqdsk.GEQDSKEquilibrium`
   instances (:func:`miller_from_geqdsk`), or
3. Any :class:`~bouquet.baseline.Baseline` (:func:`miller_from_baseline`), with
   a smooth analytical shaped-tokamak fallback when only 1-D kinetic/current
   profiles are provided in solver-free unit tests.

Unit conventions in :func:`build_tglf_inputs` and :func:`gyrobohm_flux_scales`
follow GACODE / ``TurbulentTransport.jl`` (``InputTGLF`` with ``UNITS='CGYRO'``
/ ``'GYRO'`` gyro-Bohm normalizations) so pre-trained ``TGLF-NN`` ensembles can
be evaluated directly in Python without FUSE or GACODE binaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
import tempfile
from typing import Optional

import numpy as np
from scipy import integrate

from ..physics import ELEMENTARY_CHARGE

# Exact CGS constants matching IMAS.cgs / GACODE/src/physics.jl
_CGS_E_STATC = 4.80320425e-10          # statcoulomb
_CGS_K_ERG_EV = 1.602176634e-12        # erg / eV
_CGS_MP_G = 1.67262192369e-24          # proton mass [g]
_CGS_MD_G = 3.3435837724e-24           # deuteron mass [g]
_CGS_ME_G = 9.1093837015e-28           # electron mass [g]
_CGS_C_CM_S = 2.99792458e10            # speed of light [cm/s]
_M_TO_CM = 100.0
_M3_TO_CM3 = 1.0e6
_T_TO_GAUSS = 1.0e4
_PA_TO_DYN_CM2 = 10.0
_MU0 = 4.0e-7 * np.pi


@dataclass
class MillerGeometry:
    """1-D local Miller equilibrium geometry on normalized poloidal flux ``psi_N``."""

    psi_N: np.ndarray
    rho_tor_norm: np.ndarray
    r_min_m: np.ndarray          # midplane half-width r = (R_out - R_in)/2 [m]
    R_maj_m: np.ndarray          # midplane major radius R0 = (R_out + R_in)/2 [m]
    Z_maj_m: np.ndarray          # elevation midpoint Z0 = (Z_max + Z_min)/2 [m]
    a_m: float                   # LCFS minor radius r_min(psi_N=1) [m]
    R0_m: float                  # reference magnetic/geometric axis major radius [m]
    B0_T: float                  # reference vacuum toroidal field magnitude |B0| [T]
    kappa: np.ndarray            # elongation
    delta: np.ndarray            # average triangularity (delta_u + delta_l)/2
    zeta: np.ndarray             # average squareness
    s_kappa: np.ndarray          # (r / kappa) * d(kappa)/dr
    s_delta: np.ndarray          # r * d(delta)/dr
    s_zeta: np.ndarray           # r * d(zeta)/dr
    drmaj_dr: np.ndarray         # d(R_maj)/dr (Shafranov shift derivative)
    dzmaj_dr: np.ndarray         # d(Z_maj)/dr
    q: np.ndarray                # |q(psi_N)| safety factor magnitude
    dq_dr_m: np.ndarray          # d|q|/dr [1/m]
    s_hat: np.ndarray            # (r / q) * dq/dr magnetic shear
    B_unit_T: np.ndarray         # GACODE B_unit = (1 / (2*pi*r)) * dPhi_tor/dr [T]
    vprime_miller: np.ndarray    # (dV/dr) / surface (dimensionless Miller factor)
    surface_m2: np.ndarray       # flux-surface area S(psi_N) [m^2]
    volume_m3: np.ndarray        # enclosed volume V(psi_N) [m^3]
    drho_dpsiN: np.ndarray       # d(rho_tor_norm)/d(psi_N) coordinate Jacobian
    dr_dpsiN_m: np.ndarray       # d(r_min_m)/d(psi_N) [m]
    provenance: str = "unknown"
    sign_b0: float = 1.0
    sign_q: float = 1.0

    def interpolate(self, psi_N_target: np.ndarray) -> "MillerGeometry":
        """Interpolate all 1-D radial geometry fields onto ``psi_N_target``."""
        pt = np.asarray(psi_N_target, dtype=float)
        if pt.shape == self.psi_N.shape and np.allclose(pt, self.psi_N, atol=1e-12):
            return self

        def _itp(y):
            return np.interp(pt, self.psi_N, np.asarray(y, dtype=float))

        return MillerGeometry(
            psi_N=pt.copy(),
            rho_tor_norm=_itp(self.rho_tor_norm),
            r_min_m=_itp(self.r_min_m),
            R_maj_m=_itp(self.R_maj_m),
            Z_maj_m=_itp(self.Z_maj_m),
            a_m=float(self.a_m),
            R0_m=float(self.R0_m),
            B0_T=float(self.B0_T),
            kappa=_itp(self.kappa),
            delta=_itp(self.delta),
            zeta=_itp(self.zeta),
            s_kappa=_itp(self.s_kappa),
            s_delta=_itp(self.s_delta),
            s_zeta=_itp(self.s_zeta),
            drmaj_dr=_itp(self.drmaj_dr),
            dzmaj_dr=_itp(self.dzmaj_dr),
            q=_itp(self.q),
            dq_dr_m=_itp(self.dq_dr_m),
            s_hat=_itp(self.s_hat),
            B_unit_T=_itp(self.B_unit_T),
            vprime_miller=_itp(self.vprime_miller),
            surface_m2=_itp(self.surface_m2),
            volume_m3=_itp(self.volume_m3),
            drho_dpsiN=_itp(self.drho_dpsiN),
            dr_dpsiN_m=_itp(self.dr_dpsiN_m),
            provenance=self.provenance,
            sign_b0=float(self.sign_b0),
            sign_q=float(self.sign_q),
        )


def _safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1-D derivative dy/dx handling non-strictly-monotonic edge knots safely."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return np.zeros_like(y)
    # Ensure strictly increasing coordinate for np.gradient
    dx = np.diff(x)
    if np.all(dx > 1e-14):
        return np.gradient(y, x)
    x_mono = x.copy()
    for i in range(1, x_mono.size):
        if x_mono[i] <= x_mono[i - 1] + 1e-12:
            x_mono[i] = x_mono[i - 1] + 1e-12
    return np.gradient(y, x_mono)


def _over_r_limit(numer_df_dr: np.ndarray, f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Compute ``(1 / r) * df/dr`` with exact L'Hopital ``2 * df/d(r^2)`` on axis."""
    out = np.zeros_like(r, dtype=float)
    a_ref = max(float(r[-1]), 1e-6)
    interior = r > 1e-6 * a_ref
    out[interior] = numer_df_dr[interior] / r[interior]
    if np.any(~interior) and r.size >= 3:
        r2 = r ** 2
        if r2[2] > r2[0] + 1e-16:
            slope_r2 = (f[2] - f[0]) / (r2[2] - r2[0])
        else:
            slope_r2 = 0.0
        out[~interior] = 2.0 * slope_r2
    return out


def _luce_squareness_contour(r_c: np.ndarray, z_c: np.ndarray) -> float:
    """Average 4-quadrant squareness zeta (Luce, PPCF 55, 095009, 2013)."""
    r_c = np.asarray(r_c, dtype=float)
    z_c = np.asarray(z_c, dtype=float)
    if r_c.size < 12:
        return 0.0
    r_max, r_min = float(np.max(r_c)), float(np.min(r_c))
    a = 0.5 * (r_max - r_min)
    if a <= 1e-8:
        return 0.0
    i_zmax = int(np.argmax(z_c))
    i_zmin = int(np.argmin(z_c))
    z_max, r_zmax = float(z_c[i_zmax]), float(r_c[i_zmax])
    z_min, r_zmin = float(z_c[i_zmin]), float(r_c[i_zmin])
    # Midplane elevation approximated by midpoint of extrema
    z0 = 0.5 * (z_max + z_min)
    zetas = []
    # Four quadrants: (R_ext, Z_ext, R_at_Zext)
    quads = [
        (r_max, z_max, r_zmax, +1, +1),  # upper outer
        (r_min, z_max, r_zmax, -1, +1),  # upper inner
        (r_min, z_min, r_zmin, -1, -1),  # lower inner
        (r_max, z_min, r_zmin, +1, -1),  # lower outer
    ]
    cos45 = np.sqrt(0.5)
    for r_lim, z_lim, r_vert, sr, sz in quads:
        dr_q = abs(r_lim - r_vert)
        dz_q = abs(z_lim - z0)
        if dr_q <= 1e-8 or dz_q <= 1e-8:
            continue
        mask = ((r_c - r_vert) * sr >= 0.0) & ((z_c - z0) * sz >= 0.0)
        if np.count_nonzero(mask) < 3:
            continue
        u = np.abs(r_c[mask] - r_vert) / dr_q
        v = np.abs(z_c[mask] - z0) / dz_q
        # Diagonal u == v in normalized quadrant coordinates:
        # ellipse intersects at (cos45, cos45), distance L_OE = 1.0 in (u/cos45) metric
        idx = int(np.argmin(np.abs(u - v)))
        l_od = np.hypot(u[idx], v[idx])
        l_oe = 1.0  # hypot(cos45, cos45) == 1.0
        l_ec = np.hypot(1.0 - cos45, 1.0 - cos45)  # sqrt(2) - 1
        if l_ec > 1e-12:
            zetas.append(float(np.clip((l_od - l_oe) / l_ec, -0.5, 0.8)))
    return float(np.mean(zetas)) if zetas else 0.0


def _finalize_miller(
    psi_N: np.ndarray,
    phi_tor_Wb: np.ndarray,
    r_min_m: np.ndarray,
    R_maj_m: np.ndarray,
    Z_maj_m: np.ndarray,
    kappa: np.ndarray,
    delta: np.ndarray,
    zeta: np.ndarray,
    q_abs: np.ndarray,
    B0_T: float,
    R0_m: float,
    volume_m3: Optional[np.ndarray] = None,
    surface_m2: Optional[np.ndarray] = None,
    provenance: str = "miller",
    sign_b0: Optional[float] = None,
    sign_q: Optional[float] = None,
) -> MillerGeometry:
    """Compute consistent radial derivatives, B_unit, and Jacobians on psi_N."""
    psi_N = np.asarray(psi_N, dtype=float)
    r_min_m = np.maximum(np.asarray(r_min_m, dtype=float), 0.0)
    r_min_m[0] = 0.0 if psi_N[0] <= 1e-8 else r_min_m[0]
    a_m = float(max(r_min_m[-1], 1e-4))
    kappa = np.maximum(np.asarray(kappa, dtype=float), 0.5)
    delta = np.clip(np.asarray(delta, dtype=float), -0.95, 0.95)
    zeta = np.clip(np.asarray(zeta, dtype=float), -0.8, 0.8)
    q_in = np.asarray(q_abs, dtype=float)
    resolved_sign_b0 = (
        (-1.0 if float(B0_T) < 0.0 else 1.0)
        if sign_b0 is None
        else (-1.0 if float(sign_b0) < 0.0 else 1.0)
    )
    resolved_sign_q = (
        (-1.0 if float(np.median(q_in)) < 0.0 else 1.0)
        if sign_q is None
        else (-1.0 if float(sign_q) < 0.0 else 1.0)
    )
    q_abs = np.maximum(np.abs(q_in), 0.05)

    phi_abs = np.abs(np.asarray(phi_tor_Wb, dtype=float) - float(phi_tor_Wb[0]))
    phi_end = float(max(phi_abs[-1], 1e-12))
    rho_tor_norm = np.sqrt(np.clip(phi_abs / phi_end, 0.0, 1.0))

    dkappa_dr = _safe_gradient(kappa, r_min_m)
    ddelta_dr = _safe_gradient(delta, r_min_m)
    dzeta_dr = _safe_gradient(zeta, r_min_m)
    drmaj_dr = _safe_gradient(R_maj_m, r_min_m)
    dzmaj_dr = _safe_gradient(Z_maj_m, r_min_m)
    dq_dr_m = _safe_gradient(q_abs, r_min_m)

    s_kappa = (r_min_m / kappa) * dkappa_dr
    s_delta = r_min_m * ddelta_dr
    s_zeta = r_min_m * dzeta_dr
    s_hat = (r_min_m / q_abs) * dq_dr_m
    if psi_N[0] <= 1e-8:
        s_kappa[0] = 0.0
        s_delta[0] = 0.0
        s_zeta[0] = 0.0
        s_hat[0] = 0.0
        drmaj_dr[0] = 0.0
        dzmaj_dr[0] = 0.0

    # B_unit = (1 / (2*pi*r)) * dPhi/dr  [T]  (GACODE/src/physics.jl line 52)
    dphi_dr = _safe_gradient(phi_abs, r_min_m)
    B_unit_T = _over_r_limit(dphi_dr / (2.0 * np.pi), phi_abs / (2.0 * np.pi), r_min_m)
    B_unit_T = np.maximum(np.abs(B_unit_T), 1e-3 * max(abs(B0_T), 0.1))

    if volume_m3 is None:
        volume_m3 = 2.0 * (np.pi ** 2) * R_maj_m * (r_min_m ** 2) * kappa
    else:
        volume_m3 = np.maximum(np.abs(np.asarray(volume_m3, dtype=float)), 0.0)

    if surface_m2 is None:
        # Ramanujan-like elongated torus surface approximation
        perim = 2.0 * np.pi * r_min_m * np.sqrt(0.5 * (1.0 + kappa ** 2))
        surface_m2 = 2.0 * np.pi * R_maj_m * perim
    else:
        surface_m2 = np.maximum(np.abs(np.asarray(surface_m2, dtype=float)), 0.0)

    dv_dr = _safe_gradient(volume_m3, r_min_m)
    vprime_miller = np.ones_like(psi_N)
    valid_s = surface_m2 > 1e-8
    vprime_miller[valid_s] = dv_dr[valid_s] / surface_m2[valid_s]
    if not valid_s[0] and vprime_miller.size > 1:
        vprime_miller[0] = vprime_miller[1]
    vprime_miller = np.clip(vprime_miller, 0.2, 5.0)

    # Coordinate Jacobians: d(rho_tor_norm)/d(psi_N) and d(r_min)/d(psi_N)
    # Note near axis rho ~ sqrt(psi_N), so regularize at inner half-step
    drho_dpsiN = _safe_gradient(rho_tor_norm, psi_N)
    dr_dpsiN_m = _safe_gradient(r_min_m, psi_N)
    drho_dpsiN = np.maximum(drho_dpsiN, 1e-3)
    dr_dpsiN_m = np.maximum(dr_dpsiN_m, 1e-4 * a_m)

    return MillerGeometry(
        psi_N=psi_N,
        rho_tor_norm=rho_tor_norm,
        r_min_m=r_min_m,
        R_maj_m=np.asarray(R_maj_m, dtype=float),
        Z_maj_m=np.asarray(Z_maj_m, dtype=float),
        a_m=a_m,
        R0_m=float(R0_m),
        B0_T=float(abs(B0_T)),
        kappa=kappa,
        delta=delta,
        zeta=zeta,
        s_kappa=s_kappa,
        s_delta=s_delta,
        s_zeta=s_zeta,
        drmaj_dr=drmaj_dr,
        dzmaj_dr=dzmaj_dr,
        q=q_abs,
        dq_dr_m=dq_dr_m,
        s_hat=s_hat,
        B_unit_T=B_unit_T,
        vprime_miller=vprime_miller,
        surface_m2=surface_m2,
        volume_m3=volume_m3,
        drho_dpsiN=drho_dpsiN,
        dr_dpsiN_m=dr_dpsiN_m,
        provenance=provenance,
        sign_b0=resolved_sign_b0,
        sign_q=resolved_sign_q,
    )


def _arr1d_checked(mapping: dict, key: str, expected_len: int) -> Optional[np.ndarray]:
    """Return 1-D float array ``mapping[key]`` when non-None and of ``expected_len``."""
    if not isinstance(mapping, dict):
        return None
    val = mapping.get(key)
    if val is None:
        return None
    arr = np.asarray(val, dtype=float)
    return arr if arr.ndim == 1 and arr.size == expected_len else None


def miller_from_imas_dd(
    dd_or_path,
    *,
    time: Optional[float] = None,
    psi_N_out: Optional[np.ndarray] = None,
) -> MillerGeometry:
    """Extract :class:`MillerGeometry` from an IMAS/OMAS ``dd`` dict or JSON path."""
    if isinstance(dd_or_path, (str, os.PathLike)):
        from ..io.imas import _load_dd
        dd = _load_dd(dd_or_path)
        src_name = f"imas:{os.path.basename(os.fspath(dd_or_path))}"
    else:
        dd = dd_or_path
        src_name = "imas:dict"

    eq = dd.get("equilibrium") or {}
    raw_slices = eq.get("time_slice") or []
    if isinstance(raw_slices, dict):
        slices = [raw_slices[k] for k in sorted(raw_slices.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
    else:
        slices = list(raw_slices)
    if not slices:
        raise ValueError("miller_from_imas_dd: dd['equilibrium']['time_slice'] is empty")

    if time is None or len(slices) == 1:
        eqt = slices[0] if len(slices) == 1 else slices[-1]
        t_idx = 0 if len(slices) == 1 else len(slices) - 1
    else:
        eq_times = eq.get("time")
        if eq_times is not None and len(eq_times) == len(slices):
            times = [float(t) for t in eq_times]
        else:
            times = [float(s.get("time", i)) for i, s in enumerate(slices)]
        t_idx = int(np.argmin(np.abs(np.asarray(times) - float(time))))
        eqt = slices[t_idx]

    eq1d = eqt.get("profiles_1d") or {}
    if eq1d.get("psi") is None:
        raise ValueError("miller_from_imas_dd: equilibrium profiles_1d['psi'] is missing")
    psi = np.asarray(eq1d["psi"], dtype=float)
    if psi.ndim != 1 or psi.size < 2 or abs(float(psi[-1] - psi[0])) <= 0.0:
        raise ValueError("miller_from_imas_dd: equilibrium profiles_1d['psi'] is degenerate")
    n_psi = psi.size
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])

    q_raw = _arr1d_checked(eq1d, "q", n_psi)
    if q_raw is None:
        raise ValueError("miller_from_imas_dd: equilibrium profiles_1d['q'] is missing or mismatched")
    sign_q = -1.0 if float(np.median(q_raw)) < 0.0 else 1.0
    q_abs = np.maximum(np.abs(q_raw), 0.05)

    # Compute phi_tor_Wb and rho_tor_norm BEFORE r_min_m fallback so equilibria
    # without r_inboard/r_outboard set r_min = a_LCFS * rho_tor_norm (making
    # B_unit = Phi_end / (pi * a^2) smooth across the separatrix instead of
    # diverging proportional to q(psi_N)).
    phi_1d = _arr1d_checked(eq1d, "phi", n_psi)
    rho_1d = _arr1d_checked(eq1d, "rho_tor_norm", n_psi)
    sqrt_psi = np.sqrt(np.clip(psi_N, 0.0, 1.0))
    q_varies = float(np.ptp(q_abs)) > 1.0e-3 * max(float(np.mean(q_abs)), 0.1)
    rho_1d_valid = (
        rho_1d is not None
        and np.all(np.isfinite(rho_1d))
        and not (q_varies and np.allclose(rho_1d, sqrt_psi, rtol=0.0, atol=1.0e-5))
    )

    if phi_1d is not None:
        phi_tor_Wb = phi_1d
        phi_abs = np.abs(phi_tor_Wb - float(phi_tor_Wb[0]))
        phi_end = max(float(phi_abs[-1]), 1.0e-12)
        rho_for_rmin = (
            np.clip(rho_1d, 0.0, 1.0)
            if rho_1d_valid
            else np.sqrt(np.clip(phi_abs / phi_end, 0.0, 1.0))
        )
    else:
        # In IMAS DD v3 (COCOS 11), psi is in Wb, so dPhi/dPsi = q
        dpsi = abs(float(psi[-1] - psi[0]))
        phi_from_q = integrate.cumulative_trapezoid(q_abs, psi_N * dpsi, initial=0.0)
        phi_end = max(float(phi_from_q[-1]), 1.0e-12)
        if rho_1d_valid:
            rho_for_rmin = np.clip(rho_1d, 0.0, 1.0)
            phi_tor_Wb = phi_end * (rho_for_rmin ** 2)
        else:
            phi_tor_Wb = phi_from_q
            rho_for_rmin = np.sqrt(np.clip(phi_from_q / phi_end, 0.0, 1.0))

    bdry = eqt.get("boundary") or {}
    outline = bdry.get("outline") or {}
    outline_geo = None
    outline_zeta = 0.0
    if isinstance(outline, dict) and outline.get("r") is not None and outline.get("z") is not None:
        rb = np.asarray(outline["r"], dtype=float)
        zb = np.asarray(outline["z"], dtype=float)
        if rb.ndim == 1 and zb.ndim == 1 and rb.size == zb.size and rb.size >= 8:
            from ..io.geqdsk import _flux_geometry
            outline_geo = _flux_geometry(rb, zb)
            outline_zeta = _luce_squareness_contour(rb, zb)

    r_in = _arr1d_checked(eq1d, "r_inboard", n_psi)
    r_out = _arr1d_checked(eq1d, "r_outboard", n_psi)
    bdry_ga = bdry.get("geometric_axis") if isinstance(bdry.get("geometric_axis"), dict) else {}
    gq = eqt.get("global_quantities") or {}
    gq_ax = gq.get("magnetic_axis") if isinstance(gq.get("magnetic_axis"), dict) else {}

    if r_in is not None and r_out is not None:
        r_min_m = 0.5 * (r_out - r_in)
        R_maj_m = 0.5 * (r_out + r_in)
    else:
        # Fallback for MOSAIC / OMAS equilibria without 1D r_inboard/r_outboard.
        # Prefer explicit closed-LCFS boundary scalars over outline.r when present.
        if bdry.get("minor_radius") is not None and float(bdry["minor_radius"]) > 0.0:
            a_lcfs = float(bdry["minor_radius"])
        elif outline_geo is not None and float(outline_geo["a"]) > 0.0:
            a_lcfs = float(outline_geo["a"])
        else:
            a_lcfs = 0.57

        if bdry_ga.get("r") is not None and float(bdry_ga["r"]) > 0.0:
            r_geo = float(bdry_ga["r"])
        elif outline_geo is not None and float(outline_geo["R"]) > 0.0:
            r_geo = float(outline_geo["R"])
        else:
            r_geo = 1.85

        gm8 = _arr1d_checked(eq1d, "gm8", n_psi)
        if gq_ax.get("r") is not None:
            r_ax = float(gq_ax["r"])
        elif gm8 is not None and float(gm8[0]) > 0.0:
            r_ax = float(gm8[0])
        else:
            r_ax = r_geo + 0.05 * a_lcfs

        r_min_m = a_lcfs * rho_for_rmin
        R_maj_m = r_ax + (r_geo - r_ax) * (rho_for_rmin ** 2)

    eq1d_ga = eq1d.get("geometric_axis") if isinstance(eq1d.get("geometric_axis"), dict) else {}
    z_maj_1d = _arr1d_checked(eq1d_ga, "z", n_psi)
    if z_maj_1d is not None:
        Z_maj_m = z_maj_1d
    else:
        z_ax = float(gq_ax["z"]) if gq_ax.get("z") is not None else 0.0
        if bdry_ga.get("z") is not None:
            z_edge = float(bdry_ga["z"])
        elif outline_geo is not None:
            z_edge = float(outline_geo["Z"])
        else:
            z_edge = 0.0
        Z_maj_m = z_ax + (z_edge - z_ax) * (rho_for_rmin ** 2)

    kappa_1d = _arr1d_checked(eq1d, "elongation", n_psi)
    if kappa_1d is not None:
        kappa = kappa_1d
    else:
        if bdry.get("elongation") is not None:
            k_edge = float(bdry["elongation"])
        elif outline_geo is not None and float(outline_geo["kappa"]) > 0.5:
            k_edge = float(outline_geo["kappa"])
        else:
            k_edge = 1.85
        k_core = min(1.25, k_edge)
        kappa = k_core + (k_edge - k_core) * (rho_for_rmin ** 2)

    tri_u_1d = _arr1d_checked(eq1d, "triangularity_upper", n_psi)
    tri_l_1d = _arr1d_checked(eq1d, "triangularity_lower", n_psi)
    if tri_u_1d is not None and tri_l_1d is not None:
        delta = 0.5 * (tri_u_1d + tri_l_1d)
    elif bdry.get("triangularity") is not None:
        delta = float(bdry["triangularity"]) * (rho_for_rmin ** 2)
    elif bdry.get("triangularity_upper") is not None or bdry.get("triangularity_lower") is not None:
        tu = float(bdry["triangularity_upper"]) if bdry.get("triangularity_upper") is not None else 0.0
        tl = float(bdry["triangularity_lower"]) if bdry.get("triangularity_lower") is not None else tu
        delta = 0.5 * (tu + tl) * (rho_for_rmin ** 2)
    elif outline_geo is not None:
        delta = float(outline_geo["delta"]) * (rho_for_rmin ** 2)
    else:
        delta = np.zeros_like(psi_N)

    sq_keys = (
        "squareness_lower_inner",
        "squareness_lower_outer",
        "squareness_upper_inner",
        "squareness_upper_outer",
    )
    sq_1d = [_arr1d_checked(eq1d, k, n_psi) for k in sq_keys]
    if all(arr is not None for arr in sq_1d):
        zeta = 0.25 * sum(arr for arr in sq_1d if arr is not None)
    elif all(bdry.get(k) is not None for k in sq_keys):
        z_edge = 0.25 * sum(float(bdry[k]) for k in sq_keys)
        zeta = z_edge * (rho_for_rmin ** 3)
    elif outline_geo is not None:
        zeta = float(outline_zeta) * (rho_for_rmin ** 3)
    else:
        zeta = np.zeros_like(psi_N)

    vac = ((eq.get("vacuum_toroidal_field") or {})
           if "vacuum_toroidal_field" in eq else {})
    gvac = gq.get("vacuum_toroidal_field") or {}
    f_dia = _arr1d_checked(eq1d, "f", n_psi)
    if gvac.get("b0") is not None and np.size(gvac["b0"]) > 0:
        B0_signed = float(np.ravel(gvac["b0"])[0])
        R0_m = float(gvac["r0"]) if gvac.get("r0") is not None else float(R_maj_m[0])
    elif vac.get("b0") is not None and np.size(vac["b0"]) > 0:
        b0_arr = np.ravel(vac["b0"])
        B0_signed = float(b0_arr[min(t_idx, b0_arr.size - 1)])
        R0_m = float(vac["r0"]) if vac.get("r0") is not None else float(R_maj_m[0])
    elif f_dia is not None:
        R0_m = float(R_maj_m[0])
        B0_signed = float(f_dia[-1]) / max(R0_m, 0.1)
    else:
        R0_m = float(R_maj_m[0])
        B0_signed = 1.0

    vol = _arr1d_checked(eq1d, "volume", n_psi)
    surf = _arr1d_checked(eq1d, "surface", n_psi)

    mg = _finalize_miller(
        psi_N=psi_N,
        phi_tor_Wb=phi_tor_Wb,
        r_min_m=r_min_m,
        R_maj_m=R_maj_m,
        Z_maj_m=Z_maj_m,
        kappa=kappa,
        delta=delta,
        zeta=zeta,
        q_abs=q_abs,
        B0_T=B0_signed,
        R0_m=R0_m,
        volume_m3=vol,
        surface_m2=surf,
        provenance=src_name,
        sign_b0=(-1.0 if B0_signed < 0.0 else 1.0),
        sign_q=sign_q,
    )
    return mg if psi_N_out is None else mg.interpolate(psi_N_out)


def miller_from_geqdsk(
    eqdsk_or_path_or_bytes,
    *,
    psi_N_out: Optional[np.ndarray] = None,
    cocos: int = 1,
) -> MillerGeometry:
    """Extract :class:`MillerGeometry` from a GEQDSK file, bytes, or object."""
    from ..io.geqdsk import GEQDSKEquilibrium

    cocos_int = int(cocos)
    if isinstance(eqdsk_or_path_or_bytes, GEQDSKEquilibrium):
        geq = eqdsk_or_path_or_bytes
        prov = "geqdsk:instance"
    elif isinstance(eqdsk_or_path_or_bytes, (bytes, bytearray)):
        geq = GEQDSKEquilibrium.from_bytes(bytes(eqdsk_or_path_or_bytes), cocos=cocos_int)
        prov = "geqdsk:bytes"
    else:
        path = os.fspath(eqdsk_or_path_or_bytes)
        geq = GEQDSKEquilibrium(path, cocos=cocos_int)
        prov = f"geqdsk:{os.path.basename(path)}"

    geq._trace_surfaces()
    geo = geq._cache["geo"]
    avg = geq._cache["avg"]
    contours = geq._cache["contours"]
    psi_N = np.asarray(geq.psi_N, dtype=float)

    r_min_m = np.asarray(geo["a"], dtype=float)
    R_maj_m = np.asarray(geo["R"], dtype=float)
    Z_maj_m = np.asarray(geo["Z"], dtype=float)
    kappa = np.asarray(geo["kappa"], dtype=float)
    delta = np.asarray(geo["delta"], dtype=float)
    q_raw = np.asarray(avg["q"], dtype=float)
    sign_q = -1.0 if float(np.median(q_raw)) < 0.0 else 1.0
    q_abs = np.abs(q_raw)

    zeta = np.zeros_like(psi_N)
    for k, cd in enumerate(contours):
        if cd is None:
            continue
        if isinstance(cd, dict) and "R" in cd and "Z" in cd and len(cd["R"]) >= 12:
            zeta[k] = _luce_squareness_contour(cd["R"], cd["Z"])
        else:
            arr = np.asarray(cd, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 12:
                zeta[k] = _luce_squareness_contour(arr[:, 0], arr[:, 1])
    zeta[0] = 0.0

    # Convert poloidal flux excursion to Wb regardless of GEQDSK COCOS exp_Bp
    exp_bp = float((getattr(geq, "_cocos", None) or {}).get("exp_Bp", 0))
    dpsi_wb = ((2.0 * np.pi) ** (1.0 - exp_bp)) * abs(float(geq.psi_boundary - geq.psi_axis))
    phi_tor_Wb = integrate.cumulative_trapezoid(
        q_abs, psi_N * dpsi_wb, initial=0.0
    )
    vol = np.asarray(geo["vol"], dtype=float)
    if "surfArea" in geo and np.any(np.asarray(geo["surfArea"]) > 0):
        surf = np.asarray(geo["surfArea"], dtype=float)
    else:
        surf = 2.0 * np.pi * R_maj_m * np.asarray(geo["perimeter"], dtype=float)
    R0_m = float(geq.R_center if geq.R_center else R_maj_m[0])
    B0_signed = float(geq.B_center) if geq.B_center else float(geq._raw["FPOL"][-1]) / R0_m

    mg = _finalize_miller(
        psi_N=psi_N,
        phi_tor_Wb=phi_tor_Wb,
        r_min_m=r_min_m,
        R_maj_m=R_maj_m,
        Z_maj_m=Z_maj_m,
        kappa=kappa,
        delta=delta,
        zeta=zeta,
        q_abs=q_abs,
        B0_T=B0_signed,
        R0_m=R0_m,
        volume_m3=vol,
        surface_m2=surf,
        provenance=prov,
        sign_b0=(-1.0 if B0_signed < 0.0 else 1.0),
        sign_q=sign_q,
    )
    return mg if psi_N_out is None else mg.interpolate(psi_N_out)


def miller_from_baseline(
    baseline,
    source=None,
    psi_N_out: Optional[np.ndarray] = None,
    *,
    R0_fallback: float = 1.70,
    a_fallback: float = 0.60,
    B0_fallback: float = 2.0,
    kappa_edge_fallback: float = 1.75,
    delta_edge_fallback: float = 0.45,
) -> MillerGeometry:
    """Resolve :class:`MillerGeometry` from a :class:`~bouquet.baseline.Baseline`.

    Resolution precedence:
    1. Cached ``baseline.aux['miller_geometry']`` if already present;
    2. ``ImasSource.ids_path`` when ``source`` is an :class:`ImasSource`;
    3. ``baseline.eqdsk_bytes`` or ``ReconstructionSource.geqdsk_path``;
    4. Analytical shaped-tokamak equilibrium fallback calibrated to
       ``baseline.Ip_target`` and ``baseline.l_i_target`` (for solver-free
       synthetic unit tests).
    """
    aux = getattr(baseline, "aux", None) or {}
    if isinstance(aux.get("miller_geometry"), MillerGeometry):
        mg = aux["miller_geometry"]
        return mg if psi_N_out is None else mg.interpolate(psi_N_out)

    from ..config import ImasSource, ReconstructionSource

    if isinstance(source, ImasSource) and getattr(source, "ids_path", None):
        if os.path.exists(source.ids_path):
            return miller_from_imas_dd(
                source.ids_path,
                time=getattr(source, "time", None),
                psi_N_out=psi_N_out,
            )

    src_cocos = int(getattr(source, "cocos", 1) or 1)
    eq_bytes = getattr(baseline, "eqdsk_bytes", None)
    if eq_bytes:
        return miller_from_geqdsk(eq_bytes, psi_N_out=psi_N_out, cocos=src_cocos)

    if isinstance(source, ReconstructionSource) and getattr(source, "geqdsk_path", None):
        if os.path.exists(source.geqdsk_path):
            return miller_from_geqdsk(source.geqdsk_path, psi_N_out=psi_N_out, cocos=src_cocos)

    # Analytical fallback for synthetic solver-free tests
    psi_N = (np.asarray(psi_N_out, dtype=float)
             if psi_N_out is not None
             else np.asarray(baseline.psi_N_kinetic, dtype=float))
    rho = np.sqrt(np.clip(psi_N, 0.0, 1.0))
    r_min_m = a_fallback * rho
    # Approximate Shafranov shift ~ -0.08 * a * rho^2
    R_maj_m = R0_fallback + 0.06 * a_fallback * (1.0 - rho ** 2)
    Z_maj_m = np.zeros_like(psi_N)
    kappa = 1.25 + (kappa_edge_fallback - 1.25) * (rho ** 2)
    delta = delta_edge_fallback * (rho ** 2)
    zeta = -0.04 * (rho ** 3)
    ip_ma = max(abs(float(getattr(baseline, "Ip_target", 1.0e6))) * 1e-6, 0.2)
    b0_abs = abs(float(B0_fallback))
    q95_est = max(
        2.1,
        (5.0 * (a_fallback ** 2) * b0_abs / (R0_fallback * ip_ma))
        * 0.5 * (1.0 + kappa_edge_fallback ** 2),
    )
    q0_est = 1.02
    q_abs = q0_est + (q95_est - q0_est) * (psi_N ** 1.6)
    phi_tor_Wb = np.pi * b0_abs * (r_min_m ** 2) * kappa
    return _finalize_miller(
        psi_N=psi_N,
        phi_tor_Wb=phi_tor_Wb,
        r_min_m=r_min_m,
        R_maj_m=R_maj_m,
        Z_maj_m=Z_maj_m,
        kappa=kappa,
        delta=delta,
        zeta=zeta,
        q_abs=q_abs,
        B0_T=B0_fallback,
        R0_m=R0_fallback,
        provenance="analytical_fallback",
        sign_b0=(-1.0 if float(B0_fallback) < 0.0 else 1.0),
        sign_q=1.0,
    )


def gyrobohm_flux_scales(
    miller: MillerGeometry,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
) -> dict:
    """Compute GACODE gyro-Bohm normalization factors on ``miller.psi_N``.

    Matches ``GACODE/src/physics.jl`` and ``GACODE/src/fluxes.jl`` exactly:
    - ``c_s_cm_s``: deuterium sound speed ``sqrt(k * Te / m_D)`` [cm/s]
    - ``rho_s_cm``: sound gyroradius ``c_s * m_D * c / (e * B_unit_G)`` [cm]
    - ``Q_gB_W_m2``: gyro-Bohm energy flux [W/m^2] (before ``vprime_miller``)
    - ``Q_gB_imas_W_m2``: ``Q_gB_W_m2 * vprime_miller`` [W/m^2] (IMAS surface flux)
    - ``Gamma_gB_m2_s``: gyro-Bohm particle flux [1/(m^2 s)]
    """
    ne_cgs = np.maximum(np.asarray(ne_m3, dtype=float), 1e16) / _M3_TO_CM3
    te_safe = np.maximum(np.asarray(te_eV, dtype=float), 1.0)
    bunit_G = np.maximum(np.abs(miller.B_unit_T), 1e-4) * _T_TO_GAUSS
    a_cm = max(float(miller.a_m) * _M_TO_CM, 1e-2)

    c_s_cm_s = np.sqrt(_CGS_K_ERG_EV * te_safe / _CGS_MD_G)
    rho_s_cm = c_s_cm_s / (_CGS_E_STATC * bunit_G) * (_CGS_MD_G * _CGS_C_CM_S)
    # Erg/(cm^2 s) -> W/m^2: 1e-7 * 1e4 = 1e-3
    q_gb_w_m2 = (
        ne_cgs * _CGS_K_ERG_EV * te_safe * c_s_cm_s
        * ((rho_s_cm / a_cm) ** 2) * 1.0e-3
    )
    gamma_gb = q_gb_w_m2 / (ELEMENTARY_CHARGE * te_safe)
    return {
        "c_s_cm_s": c_s_cm_s,
        "c_s_m_s": c_s_cm_s * 1.0e-2,
        "rho_s_cm": rho_s_cm,
        "rho_s_m": rho_s_cm * 1.0e-2,
        "rho_star": rho_s_cm / a_cm,
        "Q_gB_W_m2": q_gb_w_m2,
        "Q_gB_imas_W_m2": q_gb_w_m2 * miller.vprime_miller,
        "Gamma_gB_m2_s": gamma_gb,
        "Gamma_gB_imas_m2_s": gamma_gb * miller.vprime_miller,
    }


def build_tglf_inputs(
    miller: MillerGeometry,
    psi_N: np.ndarray,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    zeff: np.ndarray,
    *,
    Z_imp: float = 6.0,
    A_main: float = 2.0,
    z_fast_m3: Optional[np.ndarray] = None,
    Z_fast: float = 2.0,
    omega_tor_rad_s: Optional[np.ndarray] = None,
    p_thermal_Pa: Optional[np.ndarray] = None,
) -> dict:
    """Build 1-D radial arrays of all dimensionless ``InputTGLF`` parameters.

    Follows ``TurbulentTransport.jl/src/tglf.jl`` lines 124-231 so that every
    feature consumed by 3-species ``TGLF-NN`` committees (e.g.,
    ``sat2_em_d3d_azf-1_withnegD``) is in the exact GACODE normalization.
    """
    _ = A_main  # retained in signature; GACODE c_s and XNUE normalize to fixed m_D (A=2)
    mg = miller.interpolate(psi_N)
    r_cm = mg.r_min_m * _M_TO_CM
    Rmaj_cm = mg.R_maj_m * _M_TO_CM
    a_cm = max(float(mg.a_m) * _M_TO_CM, 1e-2)
    bunit_G = np.maximum(mg.B_unit_T, 1e-4) * _T_TO_GAUSS

    ne = np.maximum(np.asarray(ne_m3, dtype=float), 1e16)
    te = np.maximum(np.asarray(te_eV, dtype=float), 1.0)
    ni = np.maximum(np.asarray(ni_m3, dtype=float), 1e15)
    ti = np.maximum(np.asarray(ti_eV, dtype=float), 1.0)
    zf_eff = np.clip(np.asarray(zeff, dtype=float), 1.0, max(float(Z_imp) - 1e-3, 1.01))
    Z_imp_f = max(float(Z_imp), 2.0)

    ne_cgs = ne / _M3_TO_CM3
    ni_cgs = ni / _M3_TO_CM3

    # Smooth single-branch thermal quasineutrality from Z_eff (accounting for
    # fast-ion charge density z_fast = Z_fast * n_fast): avoids step discontinuities
    # in RLNS_3 when independent (ne, ni) draws approach ne == ni + z_fast, while
    # preserving TGLF thermal quasineutrality AS_2 + Z_imp * AS_3 == 1 - z_fast / ne.
    zf_fast = (
        np.zeros_like(ne)
        if z_fast_m3 is None
        else np.clip(np.asarray(z_fast_m3, dtype=float), 0.0, 0.90 * ne)
    )
    nz_zeff_raw = np.maximum(
        ne * np.maximum(zf_eff - 1.0, 0.0) - (float(Z_fast) - 1.0) * zf_fast,
        0.0,
    ) / (Z_imp_f * (Z_imp_f - 1.0))
    nz = np.hypot(nz_zeff_raw, 1.0e-6 * ne)
    nz_cgs = nz / _M3_TO_CM3
    as_2 = np.clip((ne - zf_fast - Z_imp_f * nz_zeff_raw) / ne, 0.02, 1.0)
    as_3 = np.clip(nz / ne, 1.0e-6, 0.5)

    def _a_over_L(prof: np.ndarray) -> np.ndarray:
        # -a * d(ln prof) / dr_min
        dln_dr = _safe_gradient(np.log(np.maximum(prof, 1e-30)), r_cm)
        val = -a_cm * dln_dr
        if psi_N[0] <= 1e-8:
            val[0] = 0.0
        return val

    rlns_1 = _a_over_L(ne_cgs)
    rlts_1 = _a_over_L(te)
    rlns_2 = _a_over_L(ni_cgs)
    rlts_2 = _a_over_L(ti)
    rlns_3 = _a_over_L(nz_cgs)
    rlts_3 = rlts_2.copy()

    gb = gyrobohm_flux_scales(mg, ne, te)
    c_s_cm = gb["c_s_cm_s"]
    rho_s_cm = gb["rho_s_cm"]

    # BETAE, XNUE, DEBYE (tglf.jl lines 187-191; sqrt(2) matches fixed m_D in c_s)
    betae = np.maximum(8.0 * np.pi * ne_cgs * _CGS_K_ERG_EV * te / (bunit_G ** 2), 1e-8)
    coulomb_log = np.clip(24.0 - np.log(np.sqrt(ne_cgs) / te), 8.0, 25.0)
    xnue = np.maximum(
        (a_cm / c_s_cm)
        * np.sqrt(2.0)
        * (_CGS_E_STATC ** 4)
        * np.pi
        * ne_cgs
        * coulomb_log
        / (np.sqrt(_CGS_ME_G) * ((_CGS_K_ERG_EV * te) ** 1.5)),
        1e-8,
    )
    debye = np.maximum(7.43e2 * np.sqrt(te / ne_cgs) / np.maximum(rho_s_cm, 1e-12), 1e-8)

    # Rotation / ExB shear preserving sign(B0), sign(q), SIGN_IT (tglf.jl lines 139-161)
    sign_b0 = -1.0 if float(mg.sign_b0) < 0.0 else 1.0
    sign_q = -1.0 if float(mg.sign_q) < 0.0 else 1.0
    sign_it = sign_b0 * sign_q
    if omega_tor_rad_s is not None:
        w0 = -np.asarray(omega_tor_rad_s, dtype=float)
        w0p = _safe_gradient(w0, r_cm)
        gamma_p = -Rmaj_cm * w0p
        q_signed = sign_q * mg.q
        gamma_e = -(r_cm / q_signed) * w0p
        mach = Rmaj_cm * w0 / c_s_cm
        vpar_1 = -sign_it * mach
        vpar_shear_1 = -sign_it * (a_cm / c_s_cm) * gamma_p
        vexb_shear = -(a_cm / c_s_cm) * gamma_e
    else:
        vpar_1 = np.zeros_like(psi_N)
        vpar_shear_1 = np.zeros_like(psi_N)
        vexb_shear = np.zeros_like(psi_N)

    # Thermal pressure for P_PRIME_LOC (tglf.jl lines 223-230)
    if p_thermal_Pa is None:
        p_thermal_Pa = ELEMENTARY_CHARGE * (ne * te + (ni + nz) * ti)
    p_dyn = np.asarray(p_thermal_Pa, dtype=float) * _PA_TO_DYN_CM2
    dp_dr_dyn = _safe_gradient(p_dyn, r_cm)
    over_r_dp = _over_r_limit(dp_dr_dyn, p_dyn, r_cm)
    p_prime_loc = mg.q * (a_cm ** 2) / (bunit_G ** 2) * over_r_dp

    dq_dr_cm = _safe_gradient(mg.q, r_cm)
    over_r_dq = _over_r_limit(dq_dr_cm, mg.q, r_cm)
    q_prime_loc = mg.q * (a_cm ** 2) * over_r_dq

    rmin_loc = np.clip(r_cm / a_cm, 0.0, 1.0)
    rmaj_loc = Rmaj_cm / a_cm

    return {
        "AS_1": np.ones_like(psi_N),
        "AS_2": as_2,
        "AS_3": as_3,
        "TAUS_1": np.ones_like(psi_N),
        "TAUS_2": np.clip(ti / te, 0.05, 20.0),
        "TAUS_3": np.clip(ti / te, 0.05, 20.0),
        "RLNS_1": rlns_1,
        "RLNS_2": rlns_2,
        "RLNS_3": rlns_3,
        "RLTS_1": rlts_1,
        "RLTS_2": rlts_2,
        "RLTS_3": rlts_3,
        "BETAE": betae,
        "DEBYE": debye,
        "XNUE": xnue,
        "ZEFF": zf_eff,
        "RMIN_LOC": rmin_loc,
        "RMAJ_LOC": rmaj_loc,
        # TurbulentTransport.jl/src/tglf.jl lines 194, 200 fix ZMAJ_LOC = DZMAJDX_LOC = 0
        "ZMAJ_LOC": np.zeros_like(psi_N),
        "DRMAJDX_LOC": mg.drmaj_dr,
        "DZMAJDX_LOC": np.zeros_like(psi_N),
        "KAPPA_LOC": mg.kappa,
        "DELTA_LOC": mg.delta,
        "ZETA_LOC": mg.zeta,
        "S_KAPPA_LOC": mg.s_kappa,
        "S_DELTA_LOC": mg.s_delta,
        "S_ZETA_LOC": mg.s_zeta,
        "Q_LOC": mg.q,
        "Q_PRIME_LOC": q_prime_loc,
        "P_PRIME_LOC": p_prime_loc,
        "VEXB_SHEAR": vexb_shear,
        "VPAR_1": vpar_1,
        "VPAR_SHEAR_1": vpar_shear_1,
        "SIGN_BT": np.full_like(psi_N, sign_b0),
        "SIGN_IT": np.full_like(psi_N, sign_it),
        # Convenience aliases for downstream stiffness calculations:
        "a_over_R": float(mg.a_m / max(mg.R0_m, 1e-3)),
        "R_over_a_loc": mg.R_maj_m / max(mg.a_m, 1e-4),
        "s_hat": mg.s_hat,
        "Q_gB_imas_W_m2": gb["Q_gB_imas_W_m2"],
        "Gamma_gB_imas_m2_s": gb["Gamma_gB_imas_m2_s"],
    }
