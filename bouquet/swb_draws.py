"""The swb solve method's draws (``solve_method="swb"``): every draw is one
``solve_with_bootstrap`` -- the baseline's solve B with resampled inputs.

:class:`SwbDraws` is the :class:`~bouquet.draw_methods.DrawMethod` that
:func:`~bouquet.TokaMaker_interface.generate_bouquet` calls; the baseline side
(solve A / solve B, the per-draw recipe, the sigma=0 check, the archive stamp)
is :mod:`bouquet.swb`.  docs/draw-methods.md compares the methods hook by hook.
"""
from __future__ import annotations

import numpy as np

from .draw_methods import DrawMethod
from .physics import ELEMENTARY_CHARGE
from .sampling import generate_perturbed_GPR, make_rng
from .TokaMaker_interface import _MAX_PRESSURE_ITER, draw_kinetics
from .utils import pchip_interp

#: GS iteration cap of every swb draw solve (Bouquet.generate's DrawSolveGuard
#: takes the larger of this and draw_solve_maxits)
SWB_DRAW_MAXITS = 100


def swb_fixed_pressure(ne_eq, ni_eq, ti_eq, p_fast_eq=None, z_fast_eq=None,
                       Z_imp=None):
    """SWB ``p_fixed`` on ``psi_N``: fast-ion plus single-impurity (carbon)
    thermal pressure, i.e. the solve pressure minus OFT's own
    ``eC(ne Te + ni Ti)``. Shared by the swb baseline and every swb draw.
    """
    p = np.zeros_like(np.asarray(ne_eq, dtype=float))
    if p_fast_eq is not None:
        p = p + np.asarray(p_fast_eq, dtype=float)
    if Z_imp:
        from .physics import impurity_pressure
        ne_th = (ne_eq if z_fast_eq is None else np.maximum(
            ne_eq - np.asarray(z_fast_eq, dtype=float), 0.0))
        p = p + impurity_pressure(ne_th, ni_eq, ti_eq, Z_imp)
    return p


def swb_draw(mygs, psi_N, pressure, ne, te, ni, ti,
             sigma_ne, sigma_te, sigma_ni, sigma_ti, sigma_jind, n_ls, t_ls,
             j_ls, Zeff, jind_seed, swb_recipe, *, coord, rng, p_thresh,
             max_pressure_iter=_MAX_PRESSURE_ITER, psi_N_kinetic=None,
             p_fast=None, z_fast=None, z2_fast=None, zeff_includes_fast=False,
             Z_imp=None, aux_sigmas=None, aux_baselines=None,
             aux_length_scales=None, ni_from_zeff=True, zeff_dne=None,
             isolate_edge_jBS=False):
    """One ``imas_baseline="swb"`` draw: the baseline's solve B with resampled
    inputs.

    Draws the kinetics (:func:`draw_kinetics`, <P> matched on ``mygs``'s
    current state) and one GPR redraw of the SWB inductive seed
    ``jind_seed`` (``sigma_jind`` in seed units), then calls
    ``swb_recipe(kin, j_seed)``, which resets ``mygs`` and runs the SWB
    exactly as the baseline's solve B did. All sigmas zero: the baseline
    inputs are passed through unchanged (no rng), so the draw is solve B.
    Returns the :func:`perturb_kinetic_equilibrium` tuple; with the sawtooth
    reset on, the diagnostics carry ``j_saw``, ``saw_rho_m``, ``saw_rho_out``,
    ``saw_n_dips`` (and ``saw_map_warn`` from the recipe's jphi_saw check).
    """
    rng = make_rng(rng)
    psi_kin = psi_N_kinetic if psi_N_kinetic is not None else psi_N

    def _kin_to_eq(arr):
        if psi_N_kinetic is None:
            return arr
        return pchip_interp(psi_kin, arr, psi_N)

    def _zero(s):
        return s is None or not np.any(np.asarray(s, dtype=float))

    _kin_sigma0 = (all(_zero(s) for s in (sigma_ne, sigma_te, sigma_ni, sigma_ti))
                   and all(_zero(s) for s in (aux_sigmas or {}).values()))
    if _kin_sigma0:
        ne_p, te_p, ni_p, ti_p = ne, te, ni, ti
        ne_eq, te_eq = _kin_to_eq(ne), _kin_to_eq(te)
        ni_eq, ti_eq = _kin_to_eq(ni), _kin_to_eq(ti)
        aux_out = {}
    else:
        _k = draw_kinetics(
            mygs, psi_N, pressure, ne, te, ni, ti,
            sigma_ne, sigma_te, sigma_ni, sigma_ti, n_ls, t_ls, Zeff,
            psi_kin=psi_kin, kin_to_eq=_kin_to_eq, coord=coord, rng=rng,
            p_thresh=p_thresh, max_pressure_iter=max_pressure_iter,
            p_fast=p_fast, z_fast=z_fast, z2_fast=z2_fast,
            zeff_includes_fast=zeff_includes_fast, Z_imp=Z_imp,
            aux_sigmas=aux_sigmas, aux_baselines=aux_baselines,
            aux_length_scales=aux_length_scales, ni_from_zeff=ni_from_zeff,
            zeff_dne=zeff_dne)
        ne_p, te_p, ni_p, ti_p = _k["ne"], _k["te"], _k["ni"], _k["ti"]
        ne_eq, te_eq, ni_eq, ti_eq = _k["ne_eq"], _k["te_eq"], _k["ni_eq"], _k["ti_eq"]
        Zeff, aux_out = _k["Zeff"], _k["aux"]

    jind_seed = np.asarray(jind_seed, dtype=float)
    j_seed, n_tries = jind_seed, 0
    if not _zero(sigma_jind):
        # resample until non-negative (20 tries), else keep the seed
        _j0 = jind_seed[0]
        for n_tries in range(1, 21):
            _c = generate_perturbed_GPR(
                psi_N, jind_seed / _j0,
                sigma_profile=np.asarray(sigma_jind, dtype=float) / _j0,
                length_scale=j_ls, n_samples=1, rng=rng, diag_plot=False) * _j0
            if np.all(_c >= 0.0):
                j_seed = _c
                break

    kin = dict(ne=ne_eq, te=te_eq, ni=ni_eq, ti=ti_eq, Zeff=Zeff,
               p_fixed=swb_fixed_pressure(
                   ne_eq, ni_eq, ti_eq,
                   None if p_fast is None else _kin_to_eq(np.asarray(p_fast, dtype=float)),
                   None if z_fast is None else _kin_to_eq(np.asarray(z_fast, dtype=float)),
                   Z_imp))
    res = swb_recipe(kin, j_seed)
    j_ind = np.asarray(res["j_inductive"], dtype=float)
    j_bs = np.asarray(res["isolated_j_BS"], dtype=float)
    alpha = float(np.dot(j_ind, j_seed) / np.dot(j_seed, j_seed))
    print(f"  [swb-draw] alpha={alpha:.5f}"
          + ("" if j_seed is jind_seed else f" (j_ind GPR, {n_tries} tries)")
          + (" [sigma=0 kinetics]" if _kin_sigma0 else ""))
    diagnostics = {
        "j0_scales": [], "Ip_scales": [], "iteration_l_is": [],
        "iteration_Ips": [],
        "j_inductive": j_ind,
        "j_BS": j_bs,
        "j_BS_edge": j_bs if isolate_edge_jBS else None,
        "proxy_bias_observed": None,
        "r2_ip_scale": None,
        "r2_f_ind": None,
        "aux": aux_out,
        "swb_alpha": alpha,
        "jind_resamples": int(n_tries),
        # OFT's SWB solves with pax = p[0] - p[-1]: the separatrix pressure
        # the written g-file adds back (generate_bouquet's p_sep_applied)
        "edge_pressure": {"p_sep_applied": float(
            ELEMENTARY_CHARGE * (ne_eq[-1] * te_eq[-1] + ni_eq[-1] * ti_eq[-1])
            + kin["p_fixed"][-1])},
    }
    if res.get("j_saw") is not None:      # swb_saw_q: the draw's own q reset
        diagnostics.update(j_saw=np.asarray(res["j_saw"], dtype=float),
                           saw_rho_m=float(res["saw_rho_m"]),
                           saw_n_dips=int(res["saw_n_dips"]))
        if res.get("saw_rho_out") is not None:
            diagnostics["saw_rho_out"] = float(res["saw_rho_out"])
        if res.get("saw_map_warn") is not None:
            diagnostics["saw_map_warn"] = bool(res["saw_map_warn"])
    return (ne_p, te_p, ni_p, ti_p, np.zeros_like(psi_N),
            np.asarray(res["total_j_phi"], dtype=float), diagnostics)


class SwbDraws(DrawMethod):
    """``solve_method="swb"``: every draw one SWB solve (:func:`swb_draw`),
    the baseline's solve B with resampled kinetics and a GPR redraw of the
    inductive seed.  No strong coil reg (the recipe installs solve B's reg
    toward solve A's coils), no SKIP_HOMOTOPY pin, no isoflux update, no
    homotopy: the coil drift is measured, not bounded."""

    name = "swb"
    bounded_coil_mode = False      # never installs coil bounds
    strong_coil_reg = False
    pin_bounds_on_skip = False
    iso_update = False
    homotopy = False

    def __init__(self, recipe, jind_seed, sigma_jind, *, stamp=None):
        #: the per-draw solve ``recipe(kin, j_seed)``: solve B's recipe
        self.recipe = recipe
        self.jind_seed = np.asarray(jind_seed, dtype=float)
        #: GPR sigma of the inductive seed, in seed units
        self.sigma_jind = np.asarray(sigma_jind, dtype=float)
        #: the attrs stamped on _baseline (bouquet.swb.build_swb_context)
        self.stamp = dict(stamp or {})

    def jphi_baseline(self, jphi_baseline):
        return False               # the prepared solve B IS the baseline

    def draw(self, mygs, rng, scale, count, *, coil_guard=None, bnd_diag=None,
             solve_guard=None, inputs=None, legacy=None):
        i = dict(inputs)
        return swb_draw(
            mygs, i["psi_N"], i["pressure"], i["ne"], i["te"], i["ni"],
            i["ti"], i["sigma_ne"], i["sigma_te"], i["sigma_ni"],
            i["sigma_ti"], self.sigma_jind, i["n_ls"], i["t_ls"], i["j_ls"],
            i["Zeff"], self.jind_seed, self.recipe, coord=i["coord"], rng=rng,
            p_thresh=i["p_thresh"], psi_N_kinetic=i["psi_N_kinetic"],
            p_fast=i["p_fast"], z_fast=i["z_fast"], z2_fast=i["z2_fast"],
            zeff_includes_fast=i["zeff_includes_fast"], Z_imp=i["Z_imp"],
            aux_sigmas=i["aux_sigmas"], aux_baselines=i["aux_baselines"],
            aux_length_scales=i["aux_length_scales"],
            ni_from_zeff=i["ni_from_zeff"], zeff_dne=i["zeff_dne"],
            isolate_edge_jBS=i["isolate_edge_jBS"])

    def announce_coil_reg(self):
        print("  [swb] coil reg left to the per-draw recipe; drift "
              "reference = the baseline (solve B) coils")

    def drift_without_homotopy(self, mygs, baseline_coils, coil_drift, *,
                               ip_aligned, skip_hard):
        """Drift measured, not enforced (no iso-update, no bounds, no
        re-solve: the draw stays solve B's setup)."""
        from .TokaMaker_interface import _coil_drift_pct
        cur, _ = mygs.get_coil_currents()
        return (_coil_drift_pct(cur, baseline_coils), -1, float("nan"),
                float("nan"))

    def store_draw(self, header, count, scan_key, diagnostics):
        from .utils import stamp_group_attrs
        stamp_group_attrs(header, scan_key, count, {
            "swb_alpha": diagnostics.get("swb_alpha"),
            "swb_jind_resamples": diagnostics.get("jind_resamples"),
            "swb_j_saw": diagnostics.get("j_saw"),
            "swb_saw_rho_m": diagnostics.get("saw_rho_m"),
            "swb_saw_rho_out": diagnostics.get("saw_rho_out"),
            "swb_saw_n_dips": diagnostics.get("saw_n_dips"),
            "swb_saw_map_warn": diagnostics.get("saw_map_warn")})

    # ---- Bouquet.generate -------------------------------------------------
    def scale_settings(self, jbs_range, bs_mult):
        if jbs_range is not None:
            print(f"[swb] jBS_scale_range={jbs_range} ignored: SWB re-solves "
                  "j_BS at scale 1 in every draw")
        return None, None

    def solve_maxits(self, maxits):
        return max(SWB_DRAW_MAXITS, maxits or 0)

    def achieved_jphi(self, store):
        return False               # SWB's own split is archived

    def draw_jbs_loop(self, settings):
        return None                # SWB's own self-consistency, never the Redl loop

    def store_baseline(self, header, scan_key, baseline):
        from .utils import stamp_group_attrs
        stamp_group_attrs(header, scan_key, None, self.stamp)

    @classmethod
    def verify_sigma0(cls, bq):
        if bq.baseline is None or bq.mygs is None:
            return None             # the legacy check raises the setup error
        return bq._verify_sigma0_swb()

    @classmethod
    def workflow_problems(cls, bq):
        from .config import swb_config_problems
        return [f'imas_baseline="swb": {m}'
                for m in swb_config_problems(bq.config)]
