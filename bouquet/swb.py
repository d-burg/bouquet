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
_SWB_SAW_KEYS = ("jphi_saw_out", "saw_rho_m", "saw_rho_out", "saw_n_dips")
#: swb_saw_q: with no reset (saw_n_dips 0) warn when max|jphi_saw_out - jphi_saw| exceeds
#: this fraction of max|jphi_total| (an OFT mis-mapping jphi_saw; also frozen-saw reporting).
SWB_SAW_MAP_TOL = 1e-3


def swb_source_edge(x, seed, *others):
    """``(seed, others, record)``: the read-in SWB currents zeroed on the LCFS
    node (``x >= 1``, where the edge taper's weight is 0), then the seed's
    remaining negative nodes clipped to 0 (``swb_taper_source``)."""
    edge = coords.swb_grid(x) >= 1.0
    seed, *others = [np.where(edge, 0.0, np.asarray(a, dtype=float))
                     for a in (seed, *others)]
    neg = seed < 0.0
    rec = {"swb_source_edge_zeroed": int(edge.sum()),
           "swb_source_clipped": int(neg.sum())}
    if neg.any():
        rec["swb_source_clip_min"] = float(seed[neg].min())
        import warnings
        warnings.warn(f"swb_taper_source: inductive seed negative on "
                      f"{int(neg.sum())} node(s) inside the LCFS (min "
                      f"{seed[neg].min():.4g} A/m^2); clipped to 0")
        seed = np.where(neg, 0.0, seed)
    return seed, others, rec


def _swb_taper_factor(j_fixed_out, jf_in):
    """OFT's edge-taper factor ``j_fixed / jphi_fixed``: 1 where the input is 0
    or the node is untapered (bit-identical there)."""
    jf_in = np.asarray(jf_in, dtype=float)
    f = np.where(np.abs(jf_in) > 0.0,
                 np.asarray(j_fixed_out, dtype=float) / np.where(jf_in != 0.0, jf_in, 1.0), 1.0)
    return np.where(np.abs(f - 1.0) > 1e-9, f, 1.0)


def _swb_saw_map_check(res, jphi_saw, jf_in=None):
    """``(dev_frac, warn)``: max|jphi_saw_out - f jphi_saw| / max|jphi_total| of a solve that
    reports no reset (``saw_n_dips`` 0; else ``(None, False)``).  ``f`` the edge
    taper (from ``res["j_fixed"]`` / ``jf_in``), compared only inside the first
    tapered node and where ``f`` is known."""
    if res.get("jphi_saw_out") is None or int(res.get("saw_n_dips") or 0) != 0:
        return None, False
    jphi_saw_out = np.asarray(res["jphi_saw_out"], dtype=float)
    ref = np.asarray(jphi_saw, dtype=float)
    ok = np.ones(jphi_saw_out.size, dtype=bool)
    if jf_in is not None and res.get("j_fixed") is not None:
        f = _swb_taper_factor(res["j_fixed"], jf_in)
        ref = f * ref
        tap = np.flatnonzero(f != 1.0)
        ok = (np.arange(jphi_saw_out.size) < (tap[0] if tap.size else jphi_saw_out.size)) \
            | (np.asarray(jf_in, dtype=float) != 0.0)
    peak = float(np.max(np.abs(np.asarray(res["jphi_total"], dtype=float)))) or 1.0
    dev = float(np.max(np.abs(jphi_saw_out - ref)[ok], initial=0.0)) / peak
    warn = dev > SWB_SAW_MAP_TOL
    if warn:
        print(f"WARN: swb_saw_q: no reset (saw_n_dips 0) but max|jphi_saw_out - jphi_saw| = "
              f"{dev:.2e} of max|jphi_total| (> {SWB_SAW_MAP_TOL:g}): this toolkit may mis-map "
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
        """Set ``bl.swb_seed_profile`` / ``jphi_fixed`` from the source
        split as read; with ``swb_saw_q`` also ``jphi_saw_input`` (=
        ``bl.jphi_saw``), which ``jphi_fixed`` then excludes.

        The reader's solve split carries the pressure-driven ``p'G`` in
        ``jphi_ind`` (``Baseline.jphi_pd`` records it).  A toolkit whose
        SWB returns TokaMaker jphi adds ``p'G`` itself (with its bootstrap),
        so the seed is ``jphi_ind - jphi_pd`` there; the upstream
        ``R_avg/F`` SWB adds none, and the seed keeps it."""
        from .physics import SWB_JBS_TOROIDAL, swb_jbs_convention
        bl = self.baseline
        j_ind = np.asarray(bl.jphi_ind, dtype=float)
        j_fix = np.asarray(bl.jphi_total, dtype=float) - j_ind - np.asarray(bl.jphi_bs, dtype=float)
        if swb_jbs_convention() == SWB_JBS_TOROIDAL:
            P = getattr(bl, "jphi_pd", None)
            if P is None:
                raise RuntimeError(
                    'imas_baseline="swb": the source was read without its '
                    "pressure-driven current (the reader's ratio fallback: "
                    "no equilibrium geometry), and this toolkit's SWB adds "
                    "p'G itself -- the seed would count it twice")
            j_ind = j_ind - np.asarray(P, dtype=float)
        bl.jphi_saw_input = None
        if self.config.generation.swb_saw_q is None:
            out = coords.swb_source_seed(psi_N, j_ind, j_fix)
        else:
            j_st = getattr(bl, "jphi_saw", None)
            j_st = np.zeros_like(j_fix) if j_st is None else np.asarray(j_st, dtype=float)
            out = coords.swb_source_seed(psi_N, j_ind, j_fix, j_st)
        self._swb_source_edge = None
        if getattr(self.config.generation, "swb_taper_source", False):
            seed, rest, self._swb_source_edge = swb_source_edge(psi_N, *out)
            out = (seed, *rest)
        bl.swb_seed_profile, bl.jphi_fixed = out[:2]
        if len(out) == 3:
            bl.jphi_saw_input = out[2]

    def _swb_saw_kwargs(self):
        """``solve_with_bootstrap`` sawtooth-reset arguments; empty when
        ``swb_saw_q`` is None."""
        from .config import SWB_SAW_RULES
        gc, bl = self.config.generation, self.baseline
        if gc.swb_saw_q is None:
            return {}
        if getattr(bl, "jphi_saw_input", None) is None:
            raise RuntimeError("swb_saw_q: no jphi_saw input (baseline not "
                               "prepared by _swb_imas_baseline)")
        return dict(jphi_saw=np.asarray(bl.jphi_saw_input, dtype=float),
                    saw_q_s=float(gc.swb_saw_q), saw_dq=float(gc.swb_saw_dq),
                    saw_tol=float(gc.swb_saw_tol), saw_ramp=float(gc.swb_saw_ramp),
                    saw_rule=SWB_SAW_RULES[gc.swb_saw_rule])

    def _swb_solve(self, kin, j_seed, coil_reg_target=None):
        """The swb recipe: reset ``mygs`` -> coil reg -> ``init_psi`` from the
        slice LCFS -> one ``solve_with_bootstrap``.

        ``coil_reg_target`` None keeps the setup coil reg (solve A); a
        ``{coil: A-t}`` dict installs the strong reg toward it (solve B, the
        sigma=0 check and every draw). ``kin``: ``ne te ni ti Zeff p_fixed`` on
        ``psi_N``.  Returns the SWB result dict (its ``jphi_saw_out`` / ``saw_rho_m``
        / ``saw_n_dips`` only when ``swb_saw_q`` is set).
        """
        from .oft_bootstrap import solve_with_bootstrap
        from .TokaMaker_interface import strong_coil_reg
        from .config import swb_bootstrap_kwargs
        bl, mygs, gc = self.baseline, self.mygs, self.config.generation
        psi_N = np.asarray(bl.psi_N, dtype=float)
        self._reset_solver_state()
        if coil_reg_target is not None:
            mygs.set_coil_reg(reg_terms=strong_coil_reg(
                mygs, coil_reg_target, gc.swb_coil_reg_weight, 1.0))
        mygs.init_psi(*_shape_from_boundary(self._geom.boundary_RZ))
        self._seed_coil_init(mygs)
        saw_kw = self._swb_saw_kwargs()
        j_seed, jphi_fixed = np.asarray(j_seed, dtype=float), bl.jphi_fixed
        res = solve_with_bootstrap(
            mygs, kin["ne"], kin["te"], kin["ni"], kin["ti"], kin["Zeff"],
            float(bl.Ip_target), j_seed,
            scale_jbs=1.0, isolate_edge_jbs=bool(gc.isolate_edge_jbs),
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
                res, bl.jphi_saw_input, bl.jphi_fixed)
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
        """SWB's raw split -> bouquet's: ``jphi_bs`` / ``jphi_bs`` the
        field-aligned ``kappa <j.B>`` (:func:`bouquet.physics.
        _swb_jbs_to_toroidal`, for the installed toolkit's own output
        convention), ``jphi_pd`` the pressure-driven ``p'G`` of the solved
        equilibrium (the third bucket, owner decision D2; tapered with the
        edge taper when it is on), and ``jphi_ind`` the rest -- or, from a
        toolkit that reports p'G apart (``jphi_pd``), its own split as
        returned (:func:`bouquet.physics.swb_reported_split`) -- so
        ``jphi_total = jphi_ind + jphi_bs + jphi_pd + fixed``
        exactly.  SWB's own arrays are kept as ``swb_raw_*``.  Evaluated on
        the equilibrium the SWB call left in ``mygs``, before anything else
        touches it."""
        from .physics import (SWB_JBS_TOROIDAL, _swb_jbs_to_toroidal,
                              swb_conversion_record, swb_edge_taper,
                              swb_jbs_convention, swb_pressure_term,
                              swb_reported_split)
        gc = self.config.generation
        conv = swb_jbs_convention()
        reported = swb_reported_split(res) if conv == SWB_JBS_TOROIDAL else None
        if conv == SWB_JBS_TOROIDAL and reported is None and bool(gc.isolate_edge_jbs):
            raise RuntimeError(
                "isolate_edge_jbs with a toolkit whose solve_with_bootstrap "
                "returns TokaMaker jphi (kappa<j.B> + p'G) without reporting "
                "p'G apart (jphi_pd): where p'G sits in its isolated spike is "
                "not defined; use isolate_edge_jbs=False")
        mygs = self.mygs
        x = np.asarray(psi_N, dtype=float)
        grid = (np.asarray(coords.psi_at(mygs, x, coord), dtype=float)
                if coords._swb_grid_arg() else None)
        raw_iso = np.asarray(res["jphi_bs"], dtype=float)
        raw_bs = np.asarray(res.get("jphi_bs_raw", raw_iso), dtype=float)
        raw_ind = np.asarray(res["jphi_ind"], dtype=float)
        if reported is not None:        # the toolkit's own split is bouquet's
            fa_bs, fa_iso, P = reported
            j_ind = raw_ind
        else:
            P = swb_pressure_term(mygs, x.size, 1e-3, grid)
            # the edge taper multiplies every toroidal component SWB returns,
            # p'G included: the SAME weight tapers jphi_pd and is taken off
            # the bootstrap (review of #78, B3: an untapered p'G subtracted from
            # the tapered output left jphi_bs short by (1-w)P, moved into
            # jphi_ind)
            # (the call's own kwargs, swb_bootstrap_kwargs: psi0 from
            # swb_edge_taper_psi0, a taper_edge_shape from bootstrap_kwargs)
            w = None
            if gc.swb_edge_taper_psi0 is not None:
                w = swb_edge_taper(
                    dict(gc.bootstrap_kwargs, taper_edge_jphi=True,
                         taper_edge_psi0=float(gc.swb_edge_taper_psi0)),
                    coords.swb_grid(x) if grid is None else grid)
                P = P * w
            fa_bs = _swb_jbs_to_toroidal(mygs, raw_bs, 1e-3, psi=grid,
                                         convention=conv, taper=w)
            fa_iso = (fa_bs if np.array_equal(raw_iso, raw_bs) else
                      _swb_jbs_to_toroidal(mygs, raw_iso, 1e-3, psi=grid,
                                           convention=conv, taper=w))
            j_ind = raw_ind + (raw_iso - fa_iso - P)
        res = dict(res)
        res.update(swb_raw_jphi_bs_raw=raw_bs, swb_raw_jphi_bs=raw_iso,
                   swb_raw_jphi_ind=raw_ind, jphi_bs_raw=fa_bs,
                   jphi_bs=fa_iso, jphi_pd=P, jphi_ind=j_ind,
                   **swb_conversion_record(conv, reported is not None))
        return res

    def _swb_state(self, res, j_seed, psi_pad):
        """Record of one swb solve on ``mygs``: alpha, coils, LCFS, li, Ip
        (and jphi_saw_out, saw_rho_m, saw_rho_out, saw_n_dips and the jphi_saw map
        check with the sawtooth reset on)."""
        from .utils import safe_trace_surf
        mygs = self.mygs
        j_ind = np.asarray(res["jphi_ind"], dtype=float)
        # alpha is SWB's own scale of the seed (its raw inductive)
        j_raw = np.asarray(res.get("swb_raw_jphi_ind", j_ind), dtype=float)
        j_seed = np.asarray(j_seed, dtype=float)
        coils, _ = mygs.get_coil_currents()
        st = dict(
            alpha=float(np.dot(j_raw, j_seed) / np.dot(j_seed, j_seed)),
            coils={k: float(v) for k, v in coils.items()},
            lcfs=safe_trace_surf(mygs, 1.0 - psi_pad),
            li_3=float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")["l_i"]),
            Ip=float(mygs.get_globals()[0]),
            jphi_ind=j_ind,
            jphi_bs=np.asarray(res["jphi_bs"], dtype=float),
            jphi_total=np.asarray(res["jphi_total"], dtype=float),
            j_fixed=(None if res.get("j_fixed") is None
                     else np.asarray(res["j_fixed"], dtype=float)),
            ip_rel_err=res.get("ip_rel_err"))
        if res.get("jphi_pd") is not None:
            st["jphi_pd"] = np.asarray(res["jphi_pd"], dtype=float)
        if res.get("jphi_saw_out") is not None:
            st.update(jphi_saw_out=np.asarray(res["jphi_saw_out"], dtype=float),
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
        jphi_fixed, and ``bl.jphi_saw_out`` (input + q reset) replaces it in jphi_total.
        """
        bl = self.baseline
        psi_N = np.asarray(bl.psi_N, dtype=float)
        if not np.array_equal(coords.swb_grid(psi_N), psi_N):
            raise RuntimeError('imas_baseline="swb" needs SWB on the run grid')
        jphi_total_src = np.asarray(bl.jphi_total, dtype=float).copy()
        self._swb_source_split(psi_N)
        kin = self._swb_baseline_kinetics()

        pad = float(self.config.source.psi_pad)
        st_a = self._swb_state(self._swb_solve(kin, bl.swb_seed_profile),
                               bl.swb_seed_profile, pad)
        res = self._swb_solve(kin, bl.swb_seed_profile, coil_reg_target=st_a["coils"])
        st_b = self._swb_state(res, bl.swb_seed_profile, pad)

        bl.coil_reg_target = st_a["coils"]
        bl.swb_baseline = st_b                 # what the sigma=0 check repeats
        bl.jphi_ind = st_b["jphi_ind"]
        bl.jphi_bs = st_b["jphi_bs"]
        bl.jphi_total = st_b["jphi_total"]
        # the third bucket (D2): jphi_total = jphi_ind + jphi_bs + jphi_pd +
        # fixed; archived as such (SwbDraws.store_baseline)
        bl.jphi_pd = st_b.get("jphi_pd")
        if bl.jphi_pd is not None:
            from .schema import SPLIT_PRESSURE_SEPARATE
            bl.current_split_convention = SPLIT_PRESSURE_SEPARATE
        bl.jphi_saw_out = st_b.get("jphi_saw_out")
        # taper_edge_jphi also tapers the fixed current: carry the same factor onto the
        # channels so jphi_total = jphi_ind + jphi_bs + jphi_nbi + jphi_rf + jphi_other still holds.
        # The factor is MEASURED (j_fixed / jphi_fixed; 1 where untapered), so the
        # split holds whatever the toolkit did; a taper with swb_edge_taper_psi0=None
        # (a toolkit whose own default is taper-on, review PR69 B2) is loud.
        jf_in = np.asarray(bl.jphi_fixed, dtype=float)
        if st_b["j_fixed"] is not None:
            f = _swb_taper_factor(st_b["j_fixed"], jf_in)
            if np.any(f != 1.0) and self.config.generation.swb_edge_taper_psi0 is None:
                import warnings
                warnings.warn(
                    "swb: the toolkit TAPERED the edge current although "
                    "swb_edge_taper_psi0=None (off): its own default is "
                    "taper-on and it did not take taper_edge_jphi=False",
                    RuntimeWarning, stacklevel=2)
            for name in ("jphi_nbi", "jphi_rf", "jphi_other", "jphi_saw"):
                if getattr(bl, name, None) is not None:
                    setattr(bl, name, f * np.asarray(getattr(bl, name), dtype=float))
        bl.ohm_scale, bl.bs_scale = st_b["alpha"], 1.0
        bl.bs_scale_profile = None
        bl.jphi_bs_diff = bl.jphi_diff = None
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
            fuse_total_peak=float(np.max(np.abs(jphi_total_src))))
        if bl.jphi_saw_out is not None:
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
        if bl.jphi_saw_out is not None:
            print(f"[imas swb] saw reset q_s={bl.ip_closure['saw_q_s']}: rho_m "
                  f"{st_a['saw_rho_m']:.4f} (A) {st_b['saw_rho_m']:.4f} (B), "
                  f"n_dips {st_b['saw_n_dips']}, jphi_saw_out peak "
                  f"{np.max(np.abs(bl.jphi_saw_out))/1e6:.4f} MA/m^2 (input "
                  f"{np.max(np.abs(bl.jphi_saw_input))/1e6:.4f})")
        # the pressure SWB solved with: the thermal part plus p_fixed
        from .physics import ELEMENTARY_CHARGE as _EC
        _pth_e = _EC * kin["ne"] * kin["te"]
        _pth_i = _EC * kin["ni"] * kin["ti"]
        _pfix = np.asarray(kin["p_fixed"], dtype=float)
        self._finish_imas_baseline(-1, psi_pad=pad, ctx=dict(
            p_total=_pth_e + _pth_i + _pfix, psi_N=np.asarray(bl.psi_N, float),
            p_components={"electron_thermal": _pth_e, "ion_thermal": _pth_i,
                          "fixed": _pfix}))
        bl.li_metrics.update(imas_baseline="swb", jbs_baseline_mode="swb",
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
            isolate_edge_jbs=bool(self.config.generation.isolate_edge_jbs)))
        st = self._swb_state(got["res"], got["j_seed"], float(self.config.source.psi_pad))
        psi_N = np.asarray(bl.psi_N, dtype=float)
        dev = {k: float(np.max(np.abs(st[k] - ref[k])))
               for k in ("jphi_ind", "jphi_bs", "jphi_total", "jphi_saw_out", "jphi_pd")
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
        d_bs = np.abs(st["jphi_bs"] - ref["jphi_bs"])
        iw = int(np.argmax(d_bs))
        peak = float(np.max(np.abs(ref["jphi_bs"])))
        print(f"[sigma0-check swb] {'PASS' if passed else 'FAIL'} (sigma=0 draw vs "
              f"solve B): " + ", ".join(f"{k} {v:.3g}" for k, v in dev.items()))
        return dict(spike0=st["jphi_bs"], max_dev=float(d_bs[iw]),
                    rms_dev=float(np.sqrt(np.mean(d_bs ** 2))),
                    max_dev_frac=float(d_bs[iw]) / peak if peak else 0.0,
                    psi_worst=float(psi_N[iw]),
                    coord=getattr(bl, "coord", coords.PSI), passed=passed,
                    swb_dev=dev)


def swb_jind_sigma(sigma_jphi, jphi_total, seed):
    """The swb inductive redraw's sigma, in seed units: sigma_jphi's relative
    width (sigma_jphi / |jphi_total|) on |seed|, i.e. on the inductive current
    itself.  sigma_jphi in absolute units would be the total's width, which in
    a bootstrap-dominated pedestal exceeds the inductive current there, so no
    non-negative redraw exists (every draw refused)."""
    jt = np.abs(np.asarray(jphi_total, dtype=float))
    rel = np.divide(np.asarray(sigma_jphi, dtype=float), jt, out=np.zeros_like(jt), where=jt > 0)
    return rel * np.abs(np.asarray(seed, dtype=float))


def build_swb_context(bq, env):
    """The :class:`~bouquet.swb_draws.SwbDraws` of a prepared swb baseline:
    each draw is solve B (``_swb_solve`` toward solve A's coils) with
    resampled inputs.  Puts ``mygs`` back on solve B first (generate_bouquet
    reads its drift / LCFS / warm-start references off ``mygs``)."""
    from functools import partial
    from .filtering import resolve_inspec_bands
    from .swb_draws import SwbDraws
    bl, gc = bq.baseline, bq.config.generation
    if bl.coil_reg_target is None or bl.swb_baseline is None:
        raise ValueError('imas_baseline="swb": the baseline was not prepared '
                         'by _swb_imas_baseline (no solve A coils)')
    res_b = bq._swb_solve(bq._swb_baseline_kinetics(), bl.swb_seed_profile,
                          coil_reg_target=bl.coil_reg_target)
    _ic = bl.ip_closure or {}
    stamp = {
        "imas_baseline": "swb",
        # the swb-only acceptance and options this run used (D4, E6)
        "swb_ip_tol": float(gc.swb_ip_tol),
        "swb_ip_rel_err": _ic.get("ip_rel_err"),
        "swb_ip_rel_err_solve_A": _ic.get("ip_rel_err_solve_A"),
        "swb_edge_taper_psi0": ("off" if gc.swb_edge_taper_psi0 is None
                                else float(gc.swb_edge_taper_psi0)),
        "swb_taper_source": bool(getattr(gc, "swb_taper_source", False)),
        **(getattr(bq, "_swb_source_edge", None) or {}),
        **{k: res_b[k] for k in ("swb_jbs_convention", "swb_jbs_conversion")},
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
        "jphi_saw_out": bl.jphi_saw_out,
        "jphi_saw_input": bl.jphi_saw_input}
    return SwbDraws(
        partial(bq._swb_solve, coil_reg_target=bl.coil_reg_target),
        np.asarray(bl.swb_seed_profile, dtype=float),
        swb_jind_sigma(env["sigma_jphi"], bl.jphi_total, bl.swb_seed_profile),
        stamp=stamp, jphi_pd=getattr(bl, "jphi_pd", None),
        jind_clip=bool(getattr(gc, "swb_taper_source", False)),
        core_bands=resolve_inspec_bands(gc),
        core_ref=(np.asarray(bl.psi_N, dtype=float),
                  np.asarray(bl.jphi_total, dtype=float),
                  np.asarray(env["sigma_jphi"], dtype=float)))
