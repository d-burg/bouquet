"""The swb solve method's baseline side (``solve_method="swb"``).

Solve A (the setup coil reg) finds the coils; solve B (the strong reg toward
A's coils) is the baseline, and every draw is solve B with resampled inputs
(:mod:`bouquet.swb_draws`).  :class:`SwbBaseline` holds the steps as methods
of :class:`bouquet.run.Bouquet` (it is one of its bases); :func:`build_swb_context`
builds the draw method ``generate()`` hands
:func:`~bouquet.TokaMaker_interface.generate_bouquet`.  docs/draw-methods.md
compares the methods.
"""
from __future__ import annotations

import numpy as np

from . import coords
from .utils import _shape_from_boundary

#: Default of GenerationConfig.swb_ip_tol (the swb solve's Ip acceptance):
#: kept at 5e-3 pending owner decision E6.
SWB_IP_TOL = 5e-3
#: An swb solve accepted with |Ip/Ip_target - 1| above this warns (the
#: engine's I_p acceptance; the method's own solver test measures <= 8e-6),
#: so how often the looser acceptance is used is visible (review PR69 B1).
SWB_IP_WARN = 1e-4
#: solve_with_bootstrap outputs of the sawtooth reset (GenerationConfig.swb_saw_q).
_SWB_SAW_KEYS = ("j_saw", "saw_rho_m", "saw_rho_out", "saw_n_dips")
#: swb_saw_q: with no reset (saw_n_dips 0) warn when max|j_saw - jphi_saw| exceeds
#: this fraction of max|j_phi| (an OFT mis-mapping jphi_saw; also frozen-saw reporting).
SWB_SAW_MAP_TOL = 1e-3


def _swb_taper_factor(j_fixed_out, jf_in):
    """OFT's edge-taper factor ``j_fixed / jphi_fixed``: 1 where the input is 0
    or the node is untapered (bit-identical there)."""
    jf_in = np.asarray(jf_in, dtype=float)
    f = np.where(np.abs(jf_in) > 0.0,
                 np.asarray(j_fixed_out, dtype=float) / np.where(jf_in != 0.0, jf_in, 1.0), 1.0)
    return np.where(np.abs(f - 1.0) > 1e-9, f, 1.0)


def _swb_saw_map_check(res, jphi_saw, jf_in=None):
    """``(dev_frac, warn)``: max|j_saw - f jphi_saw| / max|j_phi| of a solve that
    reports no reset (``saw_n_dips`` 0; else ``(None, False)``).  ``f`` the edge
    taper (from ``res["j_fixed"]`` / ``jf_in``), compared only inside the first
    tapered node and where ``f`` is known."""
    if res.get("j_saw") is None or int(res.get("saw_n_dips") or 0) != 0:
        return None, False
    j_saw = np.asarray(res["j_saw"], dtype=float)
    ref = np.asarray(jphi_saw, dtype=float)
    ok = np.ones(j_saw.size, dtype=bool)
    if jf_in is not None and res.get("j_fixed") is not None:
        f = _swb_taper_factor(res["j_fixed"], jf_in)
        ref = f * ref
        tap = np.flatnonzero(f != 1.0)
        ok = (np.arange(j_saw.size) < (tap[0] if tap.size else j_saw.size)) \
            | (np.asarray(jf_in, dtype=float) != 0.0)
    peak = float(np.max(np.abs(np.asarray(res["total_j_phi"], dtype=float)))) or 1.0
    dev = float(np.max(np.abs(j_saw - ref)[ok], initial=0.0)) / peak
    warn = dev > SWB_SAW_MAP_TOL
    if warn:
        print(f"WARN: swb_saw_q: no reset (saw_n_dips 0) but max|j_saw - jphi_saw| = "
              f"{dev:.2e} of max|j_phi| (> {SWB_SAW_MAP_TOL:g}): this toolkit may mis-map "
              "jphi_saw (or report a frozen saw)")
    return dev, warn


class SwbBaseline:
    """The swb baseline, its per-draw recipe and its sigma=0 check, as
    methods of :class:`bouquet.run.Bouquet`."""

    def _swb_baseline_kinetics(self):
        """SWB kinetic inputs on ``psi_N`` from the baseline, built exactly as
        a sigma=0 :func:`~bouquet.swb_draws.swb_draw` builds them."""
        from .utils import pchip_interp
        from .swb_draws import swb_fixed_pressure
        bl = self.baseline
        psi_N = np.asarray(bl.psi_N, dtype=float)
        pk = np.asarray(bl.psi_N_kinetic, dtype=float)
        k2e = lambda a: pchip_interp(pk, a, psi_N)
        ne, te, ni, ti = k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti)
        zf = getattr(bl, "z_fast", None)
        return dict(ne=ne, te=te, ni=ni, ti=ti, Zeff=self._zeff_eq(),
                    p_fixed=swb_fixed_pressure(
                        ne, ni, ti,
                        None if bl.p_fast is None else k2e(np.asarray(bl.p_fast, dtype=float)),
                        None if zf is None else k2e(np.asarray(zf, dtype=float)),
                        getattr(bl, "Z_imp", None)))

    def _zeff_eq(self):
        """Baseline Z_eff on ``psi_N``, clipped at 1 (the generate() input)."""
        from .utils import pchip_interp
        bl = self.baseline
        return np.clip(pchip_interp(bl.psi_N_kinetic, bl.Zeff,
                                    np.asarray(bl.psi_N, dtype=float)), 1.0, None)

    def _swb_source_split(self, psi_N):
        """Set ``bl.swb_seed_profile`` / ``swb_jphi_fixed`` from the source
        split as read; with ``swb_saw_q`` also ``swb_jphi_saw`` (=
        ``bl.j_sawteeth``), which ``swb_jphi_fixed`` then excludes.

        The reader's solve split carries the pressure-driven ``p'G`` in
        ``j_inductive`` (``Baseline.j_pressure`` records it).  A toolkit whose
        SWB returns TokaMaker jphi adds ``p'G`` itself (with its bootstrap),
        so the seed is ``j_inductive - j_pressure`` there; the upstream
        ``R_avg/F`` SWB adds none, and the seed keeps it."""
        from .physics import SWB_JBS_TOROIDAL, swb_jbs_convention
        bl = self.baseline
        j_ind = np.asarray(bl.j_inductive, dtype=float)
        j_fix = np.asarray(bl.j_phi, dtype=float) - j_ind - np.asarray(bl.j_BS, dtype=float)
        if swb_jbs_convention() == SWB_JBS_TOROIDAL:
            P = getattr(bl, "j_pressure", None)
            if P is None:
                raise RuntimeError(
                    'imas_baseline="swb": the source was read without its '
                    "pressure-driven current (the reader's ratio fallback: "
                    "no equilibrium geometry), and this toolkit's SWB adds "
                    "p'G itself -- the seed would count it twice")
            j_ind = j_ind - np.asarray(P, dtype=float)
        bl.swb_jphi_saw = None
        if self.config.generation.swb_saw_q is None:
            bl.swb_seed_profile, bl.swb_jphi_fixed = coords.swb_source_seed(
                psi_N, j_ind, j_fix)
            return
        j_st = getattr(bl, "j_sawteeth", None)
        j_st = np.zeros_like(j_fix) if j_st is None else np.asarray(j_st, dtype=float)
        bl.swb_seed_profile, bl.swb_jphi_fixed, bl.swb_jphi_saw = \
            coords.swb_source_seed(psi_N, j_ind, j_fix, j_st)

    def _swb_saw_kwargs(self):
        """``solve_with_bootstrap`` sawtooth-reset arguments; empty when
        ``swb_saw_q`` is None."""
        from .config import SWB_SAW_RULES
        gc, bl = self.config.generation, self.baseline
        if gc.swb_saw_q is None:
            return {}
        if getattr(bl, "swb_jphi_saw", None) is None:
            raise RuntimeError("swb_saw_q: no jphi_saw input (baseline not "
                               "prepared by _swb_imas_baseline)")
        return dict(jphi_saw=np.asarray(bl.swb_jphi_saw, dtype=float),
                    saw_q_s=float(gc.swb_saw_q), saw_dq=float(gc.swb_saw_dq),
                    saw_tol=float(gc.swb_saw_tol), saw_ramp=float(gc.swb_saw_ramp),
                    saw_rule=SWB_SAW_RULES[gc.swb_saw_rule])

    def _swb_solve(self, kin, j_seed, coil_reg_target=None):
        """The swb recipe: reset ``mygs`` -> coil reg -> ``init_psi`` from the
        slice LCFS -> one ``solve_with_bootstrap``.

        ``coil_reg_target`` None keeps the setup coil reg (solve A); a
        ``{coil: A-t}`` dict installs the strong reg toward it (solve B, the
        sigma=0 check and every draw). ``kin``: ``ne te ni ti Zeff p_fixed`` on
        ``psi_N``.  Returns the SWB result dict (its ``j_saw`` / ``saw_rho_m``
        / ``saw_n_dips`` only when ``swb_saw_q`` is set).
        """
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
        from .TokaMaker_interface import strong_coil_reg
        from .config import swb_bootstrap_kwargs
        bl, mygs, gc = self.baseline, self.mygs, self.config.generation
        psi_N = np.asarray(bl.psi_N, dtype=float)
        self._reset_solver_state()
        mygs.set_isoflux(self._iso[0], weights=self._iso[1])
        if coil_reg_target is not None:
            mygs.set_coil_reg(reg_terms=strong_coil_reg(
                mygs, coil_reg_target, gc.swb_coil_reg_weight, 1.0))
        mygs.init_psi(*_shape_from_boundary(self._boundary_RZ))
        self._seed_coil_init(mygs)
        saw_kw = self._swb_saw_kwargs()
        j_seed, jphi_fixed = np.asarray(j_seed, dtype=float), bl.swb_jphi_fixed
        res = solve_with_bootstrap(
            mygs, kin["ne"], kin["te"], kin["ni"], kin["ti"], kin["Zeff"],
            float(bl.Ip_target), j_seed,
            scale_jBS=1.0, isolate_edge_jBS=bool(gc.isolate_edge_jBS),
            diagnostic_plots=False, verbose=False,
            jphi_fixed=jphi_fixed, p_fixed=kin["p_fixed"],
            **saw_kw,
            **coords.swb_grid_kwargs(psi_N, getattr(bl, "coord", coords.PSI)),
            **swb_bootstrap_kwargs(gc))
        res = self._swb_split(res, solve_with_bootstrap, psi_N,
                              getattr(bl, "coord", coords.PSI))
        if saw_kw:
            # the taper factor is measured from j_fixed / jphi_fixed (1 where
            # untapered), whatever the setting: a toolkit may taper anyway
            res["saw_map_dev"], res["saw_map_warn"] = _swb_saw_map_check(
                res, bl.swb_jphi_saw, bl.swb_jphi_fixed)
        else:               # saw off: no saw outputs, whatever the toolkit returns
            res = {k: v for k, v in res.items() if k not in _SWB_SAW_KEYS}
        ip = abs(float(mygs.get_globals()[0]))
        err = ip / abs(float(bl.Ip_target)) - 1.0
        tol = float(getattr(gc, "swb_ip_tol", SWB_IP_TOL))
        if abs(err) > tol:
            raise RuntimeError(f"swb solve landed Ip {ip/1e6:.4f} MA, {100*err:+.2f}% "
                               f"off target (swb_ip_tol {tol:g})")
        if abs(err) > SWB_IP_WARN:
            import warnings
            warnings.warn(f"swb solve accepted at Ip {100*err:+.3f}% off target: "
                          f"above {SWB_IP_WARN:g} (the engine's acceptance), within "
                          f"swb_ip_tol={tol:g}", RuntimeWarning, stacklevel=2)
        res["ip_rel_err"] = float(err)
        return res

    def _swb_split(self, res, swb_fn, psi_N, coord):
        """SWB's raw split -> bouquet's: ``j_BS`` / ``isolated_j_BS`` the
        field-aligned ``kappa <j.B>`` (:func:`bouquet.physics.
        _swb_jbs_to_toroidal`, for the installed toolkit's own output
        convention), ``j_pressure`` the pressure-driven ``p'G`` of the solved
        equilibrium (the third bucket, owner decision D2; tapered with the
        edge taper when it is on), and ``j_inductive`` the rest, so
        ``total_j_phi = j_inductive + isolated_j_BS + j_pressure + fixed``
        exactly.  SWB's own arrays are kept as ``swb_raw_*``.  Evaluated on
        the equilibrium the SWB call left in ``mygs``, before anything else
        touches it."""
        from .physics import (SWB_JBS_TOROIDAL, _swb_jbs_to_toroidal,
                              edge_taper_weight, swb_jbs_convention,
                              swb_pressure_term)
        gc = self.config.generation
        conv = swb_jbs_convention()
        if conv == SWB_JBS_TOROIDAL and bool(gc.isolate_edge_jBS):
            raise RuntimeError(
                "isolate_edge_jBS with a toolkit whose solve_with_bootstrap "
                "returns TokaMaker jphi (kappa<j.B> + p'G): where p'G sits in "
                "its isolated spike is not defined, so the field-aligned "
                "bootstrap cannot be recovered; use isolate_edge_jBS=False")
        mygs = self.mygs
        x = np.asarray(psi_N, dtype=float)
        grid = (np.asarray(coords.psi_at(mygs, x, coord), dtype=float)
                if coords._swb_grid_arg() else None)
        P = swb_pressure_term(mygs, x.size, 1e-3, grid)
        if gc.swb_edge_taper_psi0 is not None:
            P = P * edge_taper_weight(coords.swb_grid(x) if grid is None
                                      else grid, gc.swb_edge_taper_psi0)
        raw_iso = np.asarray(res["isolated_j_BS"], dtype=float)
        raw_bs = np.asarray(res.get("j_BS", raw_iso), dtype=float)
        raw_ind = np.asarray(res["j_inductive"], dtype=float)
        fa_bs = _swb_jbs_to_toroidal(mygs, raw_bs, 1e-3, psi=grid,
                                     convention=conv)
        fa_iso = (fa_bs if np.array_equal(raw_iso, raw_bs) else
                  _swb_jbs_to_toroidal(mygs, raw_iso, 1e-3, psi=grid,
                                       convention=conv))
        res = dict(res)
        res.update(swb_raw_j_BS=raw_bs, swb_raw_isolated_j_BS=raw_iso,
                   swb_raw_j_inductive=raw_ind, j_BS=fa_bs,
                   isolated_j_BS=fa_iso, j_pressure=P,
                   j_inductive=raw_ind + (raw_iso - fa_iso - P),
                   swb_jbs_convention=conv)
        return res

    def _swb_state(self, res, j_seed, psi_pad=1e-3):
        """Record of one swb solve on ``mygs``: alpha, coils, LCFS, li, Ip
        (and j_saw, saw_rho_m, saw_rho_out, saw_n_dips and the jphi_saw map
        check with the sawtooth reset on)."""
        from .utils import safe_trace_surf
        mygs = self.mygs
        j_ind = np.asarray(res["j_inductive"], dtype=float)
        # alpha is SWB's own scale of the seed (its raw inductive)
        j_raw = np.asarray(res.get("swb_raw_j_inductive", j_ind), dtype=float)
        j_seed = np.asarray(j_seed, dtype=float)
        coils, _ = mygs.get_coil_currents()
        st = dict(
            alpha=float(np.dot(j_raw, j_seed) / np.dot(j_seed, j_seed)),
            coils={k: float(v) for k, v in coils.items()},
            lcfs=safe_trace_surf(mygs, 1.0 - psi_pad),
            li_3=float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")["l_i"]),
            Ip=float(mygs.get_globals()[0]),
            j_inductive=j_ind,
            j_BS=np.asarray(res["isolated_j_BS"], dtype=float),
            j_phi=np.asarray(res["total_j_phi"], dtype=float),
            j_fixed=(None if res.get("j_fixed") is None
                     else np.asarray(res["j_fixed"], dtype=float)),
            ip_rel_err=res.get("ip_rel_err"))
        if res.get("j_pressure") is not None:
            st["j_pressure"] = np.asarray(res["j_pressure"], dtype=float)
        if res.get("j_saw") is not None:
            st.update(j_saw=np.asarray(res["j_saw"], dtype=float),
                      saw_rho_m=float(res["saw_rho_m"]),
                      saw_n_dips=int(res["saw_n_dips"]),
                      saw_map_dev=res.get("saw_map_dev"),
                      saw_map_warn=bool(res.get("saw_map_warn", False)))
            if res.get("saw_rho_out") is not None:
                st["saw_rho_out"] = float(res["saw_rho_out"])
        return st

    def _swb_imas_baseline(self):
        """``imas_baseline="swb"``: the baseline is SWB's own equilibrium.

        Solve A (setup coil reg) finds the coils; solve B (strong reg toward
        A's coils) is the baseline, so that a draw rebuilt with the same reg
        reproduces it at sigma=0. The Fortran rescales the source inductive
        current alone (alpha) to Ip, holds NBI + RF + other fixed and re-solves
        the Redl bootstrap; the recorded split is SWB's raw output.  With
        ``swb_saw_q`` the sawteeth source is SWB's jphi_saw instead of part of
        jphi_fixed, and ``bl.j_saw`` (input + q reset) replaces it in j_phi.
        """
        bl = self.baseline
        psi_N = np.asarray(bl.psi_N, dtype=float)
        if not np.array_equal(coords.swb_grid(psi_N), psi_N):
            raise RuntimeError('imas_baseline="swb" needs SWB on the run grid')
        j_phi_src = np.asarray(bl.j_phi, dtype=float).copy()
        self._swb_source_split(psi_N)
        kin = self._swb_baseline_kinetics()

        st_a = self._swb_state(self._swb_solve(kin, bl.swb_seed_profile),
                               bl.swb_seed_profile)
        res = self._swb_solve(kin, bl.swb_seed_profile, coil_reg_target=st_a["coils"])
        st_b = self._swb_state(res, bl.swb_seed_profile)

        bl.coil_reg_target = st_a["coils"]
        bl.swb_baseline = st_b                 # what the sigma=0 check repeats
        bl.j_inductive = st_b["j_inductive"]
        bl.j_BS = st_b["j_BS"]
        bl.j_phi = st_b["j_phi"]
        # the third bucket (D2): j_phi = j_inductive + j_BS + j_pressure +
        # fixed; archived as such (SwbDraws.store_baseline)
        bl.j_pressure = st_b.get("j_pressure")
        if bl.j_pressure is not None:
            from .schema import SPLIT_PRESSURE_SEPARATE
            bl.current_split_convention = SPLIT_PRESSURE_SEPARATE
        bl.j_saw = st_b.get("j_saw")
        # taper_edge_jBS also tapers the fixed current: carry the same factor onto the
        # channels so j_phi = j_inductive + j_BS + j_NBI + j_RF + j_other still holds.
        # The factor is MEASURED (j_fixed / jphi_fixed; 1 where untapered), so the
        # split holds whatever the toolkit did; a taper with swb_edge_taper_psi0=None
        # (a toolkit whose own default is taper-on, review PR69 B2) is loud.
        jf_in = np.asarray(bl.swb_jphi_fixed, dtype=float)
        if st_b["j_fixed"] is not None:
            f = _swb_taper_factor(st_b["j_fixed"], jf_in)
            if np.any(f != 1.0) and self.config.generation.swb_edge_taper_psi0 is None:
                import warnings
                warnings.warn(
                    "swb: the toolkit TAPERED the edge current although "
                    "swb_edge_taper_psi0=None (off): its own default is "
                    "taper-on and it did not take taper_edge_jBS=False",
                    RuntimeWarning, stacklevel=2)
            for name in ("j_NBI", "j_RF", "j_other", "j_sawteeth"):
                if getattr(bl, name, None) is not None:
                    setattr(bl, name, f * np.asarray(getattr(bl, name), dtype=float))
        bl.ohm_scale, bl.bs_scale = st_b["alpha"], 1.0
        bl.bs_scale_profile = None
        bl.jBS_diff = bl.jphi_diff = None
        _dc = {k: st_b["coils"][k] - st_a["coils"].get(k, 0.0) for k in st_b["coils"]}
        _worst = max(_dc, key=lambda k: abs(_dc[k])) if _dc else None
        bl.ip_closure = dict(
            closure_channel="swb_internal",
            ohm_scale=st_b["alpha"], bs_scale=1.0,
            alpha_solve_A=st_a["alpha"], li_3_solve_A=st_a["li_3"],
            coil_B_minus_A_max=(float(_dc[_worst]) if _worst else 0.0),
            coil_B_minus_A_worst=_worst,
            closure_limited=not (0.5 <= st_b["alpha"] <= 2.0),
            ip_rel_err_solve_A=st_a.get("ip_rel_err"),
            ip_rel_err=st_b.get("ip_rel_err"),
            swb_ip_tol=float(getattr(self.config.generation, "swb_ip_tol",
                                     SWB_IP_TOL)),
            fuse_total_peak=float(np.max(np.abs(j_phi_src))))
        if bl.j_saw is not None:
            bl.ip_closure.update(saw_q_s=float(self.config.generation.swb_saw_q),
                                 saw_rho_m=st_b["saw_rho_m"],
                                 saw_rho_out=st_b.get("saw_rho_out"),
                                 saw_n_dips=st_b["saw_n_dips"],
                                 saw_rho_m_solve_A=st_a.get("saw_rho_m"),
                                 saw_map_dev=st_b.get("saw_map_dev"),
                                 saw_map_warn=bool(st_a.get("saw_map_warn")
                                                   or st_b.get("saw_map_warn")))
        print(f"[imas swb] solve A: alpha={st_a['alpha']:.5f} li_3={st_a['li_3']:.4f}; "
              f"solve B (strong reg toward A's coils): alpha={st_b['alpha']:.5f} "
              f"li_3={st_b['li_3']:.4f}; max coil B-A {bl.ip_closure['coil_B_minus_A_max']:+.1f} "
              f"A-t ({_worst})")
        if bl.j_saw is not None:
            print(f"[imas swb] saw reset q_s={bl.ip_closure['saw_q_s']}: rho_m "
                  f"{st_a['saw_rho_m']:.4f} (A) {st_b['saw_rho_m']:.4f} (B), "
                  f"n_dips {st_b['saw_n_dips']}, j_saw peak "
                  f"{np.max(np.abs(bl.j_saw))/1e6:.4f} MA/m^2 (input "
                  f"{np.max(np.abs(bl.swb_jphi_saw))/1e6:.4f})")
        # the pressure SWB solved with: the thermal part plus p_fixed
        from .physics import ELEMENTARY_CHARGE as _EC
        _pth_e = _EC * kin["ne"] * kin["te"]
        _pth_i = _EC * kin["ni"] * kin["ti"]
        _pfix = np.asarray(kin["p_fixed"], dtype=float)
        self._finish_imas_baseline(-1, ctx=dict(
            p_total=_pth_e + _pth_i + _pfix, psi_N=np.asarray(bl.psi_N, float),
            p_components={"electron_thermal": _pth_e, "ion_thermal": _pth_i,
                          "fixed": _pfix}))
        bl.li_metrics.update(imas_baseline="swb", jBS_baseline_mode="swb",
                             forward_solve_nl_its=None)

    def _verify_sigma0_swb(self):
        """``imas_baseline="swb"`` sigma=0 check: ONE draw through the swb draw
        route (:meth:`~bouquet.swb_draws.SwbDraws.draw`, every perturbation
        zero) must equal solve B -- split, alpha, li_3, coils and LCFS -- to
        the bit (``nthreads=1``). Leaves ``mygs`` on that draw.
        """
        bl = self.baseline
        ref = bl.swb_baseline
        seed = np.asarray(bl.swb_seed_profile, dtype=float)
        m, got = build_swb_context(self, {"sigma_jphi": np.zeros_like(seed)}), {}
        recipe = m.recipe

        def _recipe(kin, j_seed):
            got.update(res=recipe(kin, j_seed), j_seed=j_seed)
            return got["res"]
        m.recipe = _recipe
        m.draw(self.mygs, None, 1.0, 0, inputs=dict(
            dict.fromkeys(("pressure", "sigma_ne", "sigma_te", "sigma_ni",
                           "sigma_ti", "n_ls", "t_ls", "j_ls", "p_thresh",
                           "aux_sigmas", "aux_baselines", "aux_length_scales",
                           "zeff_dne", "z2_fast")),
            psi_N=np.asarray(bl.psi_N, dtype=float),
            psi_N_kinetic=np.asarray(bl.psi_N_kinetic, dtype=float),
            ne=bl.ne, te=bl.te, ni=bl.ni, ti=bl.ti, Zeff=self._zeff_eq(),
            coord=getattr(bl, "coord", coords.PSI), p_fast=bl.p_fast,
            z_fast=getattr(bl, "z_fast", None), Z_imp=getattr(bl, "Z_imp", None),
            zeff_includes_fast=bool(getattr(bl, "zeff_includes_fast", False)),
            ni_from_zeff=True,
            isolate_edge_jBS=bool(self.config.generation.isolate_edge_jBS)))
        st = self._swb_state(got["res"], got["j_seed"])
        psi_N = np.asarray(bl.psi_N, dtype=float)
        dev = {k: float(np.max(np.abs(st[k] - ref[k])))
               for k in ("j_inductive", "j_BS", "j_phi", "j_saw", "j_pressure")
               if k in ref}
        dev["alpha"] = abs(st["alpha"] - ref["alpha"])
        dev["li_3"] = abs(st["li_3"] - ref["li_3"])
        dev["coils"] = max((abs(st["coils"][k] - ref["coils"].get(k, np.nan))
                            for k in st["coils"]), default=0.0)
        la, lb = st["lcfs"], ref["lcfs"]
        dev["lcfs"] = (float(np.max(np.abs(np.asarray(la) - np.asarray(lb))))
                       if la is not None and lb is not None
                       and np.shape(la) == np.shape(lb) else float("nan"))
        for k in ("saw_rho_m", "saw_rho_out"):
            if k in ref:
                dev[k] = abs(st.get(k, np.nan) - ref[k])
        passed = all(v == 0.0 for v in dev.values())
        d_bs = np.abs(st["j_BS"] - ref["j_BS"])
        iw = int(np.argmax(d_bs))
        peak = float(np.max(np.abs(ref["j_BS"])))
        print(f"[sigma0-check swb] {'PASS' if passed else 'FAIL'} (sigma=0 draw vs "
              f"solve B): " + ", ".join(f"{k} {v:.3g}" for k, v in dev.items()))
        return dict(spike0=st["j_BS"], max_dev=float(d_bs[iw]),
                    rms_dev=float(np.sqrt(np.mean(d_bs ** 2))),
                    max_dev_frac=float(d_bs[iw]) / peak if peak else 0.0,
                    psi_worst=float(psi_N[iw]),
                    coord=getattr(bl, "coord", coords.PSI), passed=passed,
                    swb_dev=dev)


def build_swb_context(bq, env):
    """The :class:`~bouquet.swb_draws.SwbDraws` of a prepared swb baseline:
    each draw is solve B (``_swb_solve`` toward solve A's coils) with
    resampled inputs.  Puts ``mygs`` back on solve B first (generate_bouquet
    reads its drift / LCFS / warm-start references off ``mygs``)."""
    from functools import partial
    from .swb_draws import SwbDraws
    bl, gc = bq.baseline, bq.config.generation
    if bl.coil_reg_target is None or bl.swb_baseline is None:
        raise ValueError('imas_baseline="swb": the baseline was not prepared '
                         'by _swb_imas_baseline (no solve A coils)')
    bq._swb_solve(bq._swb_baseline_kinetics(), bl.swb_seed_profile,
                  coil_reg_target=bl.coil_reg_target)
    _ic = bl.ip_closure or {}
    from .physics import swb_conversion_record
    stamp = {
        "imas_baseline": "swb",
        # the swb-only acceptance and options this run used (D4, E6)
        "swb_ip_tol": float(gc.swb_ip_tol),
        "swb_ip_rel_err": _ic.get("ip_rel_err"),
        "swb_ip_rel_err_solve_A": _ic.get("ip_rel_err_solve_A"),
        "swb_edge_taper_psi0": ("off" if gc.swb_edge_taper_psi0 is None
                                else float(gc.swb_edge_taper_psi0)),
        **swb_conversion_record(),
        "swb_alpha": float(bl.ohm_scale),
        "swb_alpha_solve_A": _ic.get("alpha_solve_A"),
        "swb_li_3_solve_A": _ic.get("li_3_solve_A"),
        "coil_reg_target": bl.coil_reg_target,
        # sawtooth reset (None, so not written, when swb_saw_q is off)
        "swb_saw_q": gc.swb_saw_q,
        "swb_saw_rho_m": _ic.get("saw_rho_m"),
        "swb_saw_rho_out": _ic.get("saw_rho_out"),
        "swb_saw_n_dips": _ic.get("saw_n_dips"),
        "swb_saw_map_warn": _ic.get("saw_map_warn"),
        "swb_j_saw": bl.j_saw,
        "swb_jphi_saw": bl.swb_jphi_saw}
    return SwbDraws(
        partial(bq._swb_solve, coil_reg_target=bl.coil_reg_target),
        np.asarray(bl.swb_seed_profile, dtype=float),
        # sigma_jphi is on the solved inductive (alpha * seed)
        np.asarray(env["sigma_jphi"], dtype=float) / float(bl.ohm_scale),
        stamp=stamp, j_pressure=getattr(bl, "j_pressure", None))
