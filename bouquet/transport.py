"""Per-equilibrium chi_perp and tau_E from FUSE's summed sources.

IMAS.jl 7.4.0 recipes (``calculate_diffusivities``, ``tau_e_thermal``,
``tau_e_h98``, ``tau_e_ds03``) evaluated on bouquet's own profiles and on the
equilibrium's own TokaMaker geometry; the sources (:func:`bouquet.io.imas.
dd_source_totals`) are FUSE's at the dd step, the same for every draw of a scan.
"""
import numpy as np

from .physics import ELEMENTARY_CHARGE as E_CHARGE

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Archived per equilibrium (draw group or ``_baseline``): datasets and attrs.
TRANSPORT_PROFILES = {"chi_perp_e": "m^2 s^-1", "chi_perp_i": "m^2 s^-1"}
TRANSPORT_SCALARS = {"tau_E_th": "s", "tau_E": "s", "tau_98": "s", "H_98": "",
                     "tau_DS03": "s", "W_th": "J", "W_fast": "J", "P_net": "W"}


def flux_geometry(mygs, b0, npsi=65, psi_pad=1e-3):
    """The live equilibrium's geometry on ``psi_N`` (TokaMaker ``get_fsa``):
    volume ``V`` [m^3], surface area ``S`` [m^2] (= <|grad psi|> dV/dpsi),
    ``phi_N``, ``rho_tor`` [m] (sqrt(Phi / pi |b0|)), ``R_min``/``R_max`` [m],
    with the axis point prepended; scalars ``a``, ``R_geo``, ``kappa``."""
    f = mygs.get_fsa(psi=np.linspace(psi_pad, 1.0 - psi_pad, int(npsi)))
    s = np.abs(np.asarray(f["psi"]) - f["psi_axis"])          # Wb/rad from the axis
    dVds = np.abs(np.asarray(f["dV/dPsi"]))
    q = np.abs(np.asarray(f["q"]))
    s = np.concatenate(([0.0], s))
    dVds = np.concatenate(([dVds[0]], dVds))
    q = np.concatenate(([q[0]], q))
    V = np.concatenate(([0.0], np.cumsum(0.5 * (dVds[1:] + dVds[:-1]) * np.diff(s))))
    phi = 2.0 * np.pi * np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(s))))
    S = np.concatenate(([0.0], np.asarray(f["<|grad psi|>"]) * np.abs(f["dV/dPsi"])))
    rmin = np.concatenate(([f["R_axis"]], f["R_min"]))
    rmax = np.concatenate(([f["R_axis"]], f["R_max"]))
    zr = (f["Z_max"][-1] - f["Z_min"][-1]) / (f["R_max"][-1] - f["R_min"][-1])
    return dict(psi_N=np.concatenate(([0.0], f["psi_norm"])), phi_N=phi / phi[-1],
                rho_tor=np.sqrt(phi / (np.pi * abs(b0))), V=V, S=S, R_min=rmin,
                R_max=rmax, a=0.5 * (rmax[-1] - rmin[-1]),
                R_geo=0.5 * (rmax[-1] + rmin[-1]), kappa=float(zr))


def transport_record(geom, x_kin, ne, te, zeff, ti, x_eq, p_thermal, src, Ip,
                     coord="psi_n", p_fast=None, main_ion_mass=2.0):
    """chi_perp_e / chi_perp_i [m^2/s] on the kinetic nodes *x_kin* and the tau
    scalars of one equilibrium (:data:`TRANSPORT_SCALARS`).

    Power balance, conduction only: chi_s = L_eff (Q_s - 5/2 e T_s Gamma_s) /
    (S n_s e T_s dlnT_s/drho_N), L_eff = rho_tor,b (dV/drho_tor) / S, and the
    main ions as IMAS.jl 7.4: n_i = n_e/Z_eff, Gamma_i = Gamma_e/Z_eff (*zeff* on
    *x_kin*).  tau_E,th = W_th/P_net, tau_E = (W_th + W_fast)/P_net,
    W = 3/2 int p dV; tau_98 (IPB98y2), tau_DS03 as IMAS.jl, H_98 = tau_E,th/tau_98.
    *x_kin* / *x_eq* are in the run coordinate *coord* ("psi_n" or "phi_n");
    temperatures in eV."""
    key = "phi_N" if coord == "phi_n" else "psi_N"
    g = {k: np.interp(np.asarray(x_kin, float), geom[key], geom[k])
         for k in ("phi_N", "rho_tor", "V", "S", "R_min", "R_max")}
    ne, te, zeff, ti = (np.asarray(v, dtype=float) for v in (ne, te, zeff, ti))
    ni = ne / zeff
    rho_n = np.sqrt(np.clip(g["phi_N"], 0.0, None))
    rho_b = geom["rho_tor"][-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        L_eff = rho_b * np.gradient(g["V"], g["rho_tor"]) / g["S"]
        L_eff[0] = L_eff[1]
        rho_src = np.asarray(src["rho_tor_norm"], dtype=float)
        Qe, Qi, Ge = (np.interp(rho_n, rho_src, np.asarray(src[k], float))
                      for k in ("Q_e", "Q_i", "Gamma_e"))
        Gi = Ge / zeff

        def chi(Q, G, n, T):
            dlnT = -np.gradient(T, rho_n) / T
            c = L_eff * (Q - 2.5 * E_CHARGE * T * G) / (g["S"] * n * E_CHARGE * T * dlnT)
            c[0] = c[1]
            return c
        rec = dict(chi_perp_e=chi(Qe, Ge, ne, te), chi_perp_i=chi(Qi, Gi, ni, ti))
    V_eq = np.interp(np.asarray(x_eq, float), geom[key], geom["V"])
    W_th = 1.5 * _trapz(np.asarray(p_thermal, float), V_eq)
    W_fast = (0.0 if p_fast is None else
              1.5 * _trapz(np.asarray(p_fast, float), g["V"]))
    P = float(src["P_net"])
    tau = (lambda W: W / P if P > 0 else np.inf)
    a, R0, b0 = geom["a"], float(src["r0"]), float(src["b0"])
    width = g["R_max"] - g["R_min"]
    nel = _trapz(ne, width) / width[-1]
    ne_vol = _trapz(ne, g["V"]) / g["V"][-1]
    kappa_areal = geom["V"][-1] / (2 * np.pi * geom["R_geo"]) / (np.pi * a ** 2)
    Ip_MA, P_MW = abs(float(Ip)) / 1e6, P / 1e6
    if P > 0:
        tau_98 = (0.0562 * Ip_MA ** 0.93 * b0 ** 0.15 * P_MW ** -0.69
                  * (nel / 1e19) ** 0.41 * main_ion_mass ** 0.19 * R0 ** 1.97
                  * (R0 / a) ** -0.58 * kappa_areal ** 0.78)
        tau_ds03 = (0.028 * Ip_MA ** 0.83 * b0 ** 0.07 * P_MW ** -0.55
                    * (0.5 * (nel + ne_vol) / 1e19) ** 0.49 * main_ion_mass ** 0.14
                    * R0 ** 2.11 * (R0 / a) ** -0.30 * geom["kappa"] ** 0.75)
    else:
        tau_98 = tau_ds03 = np.inf
    rec.update(tau_E_th=tau(W_th), tau_E=tau(W_th + W_fast), tau_98=tau_98,
               H_98=tau(W_th) / tau_98, tau_DS03=tau_ds03, W_th=W_th,
               W_fast=W_fast, P_net=P)
    return rec
