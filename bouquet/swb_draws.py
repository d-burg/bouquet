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
#: GPR redraws of the inductive seed a draw may take to find a non-negative
#: one; past it the draw is REFUSED (:class:`SwbSeedRedrawRefused`), never
#: run on the unperturbed seed (review PR69 B3).
SWB_JIND_MAX_RESAMPLES = 20
#: The draw-rejection code of that refusal (generate_bouquet's record).
SWB_JIND_REJECTION = "swb_jind_redraw_refused"


class SwbSeedRedrawRefused(RuntimeError):
    """No non-negative GPR redraw of the swb inductive seed within
    :data:`SWB_JIND_MAX_RESAMPLES` tries: the draw is refused (and recorded
    as :data:`SWB_JIND_REJECTION`) instead of silently keeping the
    unperturbed seed, which would drop its j_inductive perturbation."""


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
        _ks = None          # nothing sampled
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
        _ks = _k.get("kinetic_sampler")

    jind_seed = np.asarray(jind_seed, dtype=float)
    j_seed, n_tries = jind_seed, 0
    if not _zero(sigma_jind):
        # resample until non-negative, at most SWB_JIND_MAX_RESAMPLES tries;
        # then the draw is REFUSED, never run on the unperturbed seed
        _j0 = jind_seed[0]
        if not (np.isfinite(_j0) and _j0 > 0.0):
            raise SwbSeedRedrawRefused(
                f"swb inductive seed has seed[0] = {_j0!r}: its GPR redraw "
                "is normalised by it and cannot be drawn")
        for n_tries in range(1, SWB_JIND_MAX_RESAMPLES + 1):
            _c = generate_perturbed_GPR(
                psi_N, jind_seed / _j0,
                sigma_profile=np.asarray(sigma_jind, dtype=float) / _j0,
                length_scale=j_ls, n_samples=1, rng=rng, diag_plot=False) * _j0
            if np.all(_c >= 0.0):
                j_seed = _c
                break
        else:
            _neg = int(np.sum(jind_seed < 0.0))
            raise SwbSeedRedrawRefused(
                f"no non-negative GPR redraw of the swb inductive seed in "
                f"{SWB_JIND_MAX_RESAMPLES} tries"
                + (f" (the seed itself is negative on {_neg} node(s))"
                   if _neg else "")
                + "; the draw is refused rather than run on the unperturbed "
                "seed")

    kin = dict(ne=ne_eq, te=te_eq, ni=ni_eq, ti=ti_eq, Zeff=Zeff,
               p_fixed=swb_fixed_pressure(
                   ne_eq, ni_eq, ti_eq,
                   None if p_fast is None else _kin_to_eq(np.asarray(p_fast, dtype=float)),
                   None if z_fast is None else _kin_to_eq(np.asarray(z_fast, dtype=float)),
                   Z_imp))
    res = swb_recipe(kin, j_seed)
    # the recipe returns bouquet's split (Bouquet._swb_split): j_BS the
    # field-aligned bootstrap, j_pressure the third bucket, SWB's own
    # inductive (alpha's) as swb_raw_j_inductive
    j_ind = np.asarray(res["j_inductive"], dtype=float)
    j_bs = np.asarray(res["isolated_j_BS"], dtype=float)
    j_raw = np.asarray(res.get("swb_raw_j_inductive", j_ind), dtype=float)
    alpha = float(np.dot(j_raw, j_seed) / np.dot(j_seed, j_seed))
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
        # the sampler version + clip counters (PR #56; None at sigma=0)
        "kinetic_sampler": _ks,
        "swb_alpha": alpha,
        "jind_resamples": int(n_tries),
        "j_pressure": (None if res.get("j_pressure") is None
                       else np.asarray(res["j_pressure"], dtype=float)),
        "swb_ip_rel_err": res.get("ip_rel_err"),
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


def _write_current_split(header, scan_key, count, j_pressure):
    """The swb split's third bucket on one archived group (a draw, or
    ``_baseline`` for ``count`` None): ``j_pressure`` and
    ``current_split_convention = "pressure_separate"`` (schema; owner
    decision D2).  Nothing when the draw carried none (a recipe that did not
    split)."""
    if j_pressure is None:
        return
    import h5py
    from .schema import write_current_split
    from .utils import _group_path, _scan_key
    bkey = _scan_key(scan_key)
    path = (_group_path(scan_key, count) if count is not None
            else (f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"))
    with h5py.File(f"{header}.h5", "a") as hf:
        write_current_split(hf[path], j_pressure)


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

    def __init__(self, recipe, jind_seed, sigma_jind, *, stamp=None,
                 j_pressure=None):
        #: the per-draw solve ``recipe(kin, j_seed)``: solve B's recipe
        self.recipe = recipe
        self.jind_seed = np.asarray(jind_seed, dtype=float)
        #: GPR sigma of the inductive seed, in seed units
        self.sigma_jind = np.asarray(sigma_jind, dtype=float)
        if np.any(self.sigma_jind) and not self.jind_seed[0] > 0.0:
            raise ValueError(
                f"swb: the inductive seed has seed[0] = {self.jind_seed[0]!r} "
                "with a non-zero sigma_jphi: its GPR redraw is normalised by "
                "it -- every draw would be refused")
        #: the baseline's (solve B's) j_pressure, archived on _baseline
        self.j_pressure = j_pressure
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

    def rejection_reason(self, exc, stage):
        if isinstance(exc, SwbSeedRedrawRefused):
            return SWB_JIND_REJECTION
        return super().rejection_reason(exc, stage)

    def store_draw(self, header, count, scan_key, diagnostics):
        from .utils import stamp_group_attrs
        _write_current_split(header, scan_key, count,
                             diagnostics.get("j_pressure"))
        stamp_group_attrs(header, scan_key, count, {
            "swb_alpha": diagnostics.get("swb_alpha"),
            "swb_ip_rel_err": diagnostics.get("swb_ip_rel_err"),
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
        _write_current_split(header, scan_key, None, self.j_pressure)
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
