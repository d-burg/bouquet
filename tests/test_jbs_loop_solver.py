"""The self-consistent bootstrap loop -- live-solver half (``pytest -m solver``).

Runs on the synthetic D3D-like example only (``examples/D3D-like``: the OMAS
data dictionary for the IMAS path, the g-file + p-file for the reconstruction
path); no device data.  Every solver call lives in a subprocess probe (``OFT_env``
is a per-process singleton -- see ``tests/_harness.py``); each probe writes a
JSON of measurements and the tests below read it.

Plan tests covered here (the iteration-only ones -- (c), (d), (e) on synthetic
maps -- are in ``test_jbs_loop.py``; (d) and (e) also run live below):

(a) ``evaluate_jBS`` on a uniform grid equals ``solve_with_bootstrap``'s own
    FIRST-pass Redl evaluation on the same equilibrium, bit for bit (needs an
    OFT build whose ``solve_with_bootstrap`` takes ``psi_N=``; skipped with the
    reason otherwise), with the epsilon THAT build's SWB uses: ``<a>/<R>`` on
    OpenFUSIONToolkit main (``eps_definition="a_over_R"``), the geometric
    ``(R_max - R_min)/(2<R>)`` where SWB has ``use_sauter_eps`` on (the fork);
(a2) the two routes to the geometric epsilon -- ``sauter_fc(return_eps=True)``
    and ``get_fsa``'s ``R_min``/``R_max`` over the ``sauter_fc`` ``<R>`` --
    agree to rounding on a solved equilibrium (the comparison skips on builds
    without the fork option; the evaluation itself never does);
(b) the same physical profiles on a uniform and on a strongly non-uniform
    (rho-like) psi_N grid give the same j_BS on a live equilibrium, while the
    legacy uniform reading of the non-uniform arrays does not (defect A);
    plus the IMAS-type grid whose first intervals are finer than psi_pad;
(d) two different initial guesses (the anchor evaluation and the legacy SWB
    result) converge to the same baseline;
(e) a loop that cannot converge raises JBSNotConverged, and "flag" mode
    delivers the slice flagged;
(f) the sigma=0 draw loop reproduces the baseline under the loop (IMAS diff
    mode and the reconstruction path), and a sigma=0 route-R2 draw converges;
(g) diff mode at sigma=0 reproduces the source j_BS exactly.
(7f) a perturbed draw's anchor loop reaches the same fixed point from its own
    start (Redl at the anchor on the draw's own kinetics) and from the
    unperturbed baseline bootstrap (the counterfactual), and the per-draw
    block records the start.

Plus: the loop in every IMAS baseline mode/closure channel converges and
records its block; the structured closure's correctors are subsumed with their
bookkeeping intact; the MSE stage composes with the loop (converge without
MSE -> Jacobian -> chord steps with j_BS re-evaluated -> final Jacobian
refresh).
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_OMAS = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.3043
_files_ok = all(os.path.isfile(p) for p in (_OMAS, _GEQ, _PF, _MESH))


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")

#: The structured closure (soft preset, gated axis row) needed exactly the
#: approved default pass ceiling (8) on this synthetic case on the development
#: build -- its one-sided prior makes the closure map non-smooth, so an early
#: residual growth halves omega toward the floor.  Its integration tests below
#: run at the SHIPPED default ceiling (``GenerationConfig.jbs_max_passes``),
#: not at a raised test-only ceiling: a build on which that channel misses
#: the default by a pass is a finding about the default, and must show up
#: here as a failure rather than be absorbed by the test.


# ---------------------------------------------------------------------------
#  probe 1: the IMAS path
# ---------------------------------------------------------------------------
def _imas_probe(outdir, part):
    """``part``: "core" (the evaluator (a)/(b)/grid, diff mode (g), sigma=0
    (f)), "modes" (rescale, ohmic with both inits (d), sawtooth, (e)) or
    "structured" (the structured closure and its MSE stage).  Split so a
    failure in one part does not take the others' evidence with it."""
    import inspect
    import numpy as np
    import bouquet as bq
    from bouquet.baseline import resolve_baseline
    from bouquet.jbs_loop import (JBSNotConverged, profile_residuals,
                                  residual_weights, weighted_norm)
    from bouquet.physics import _accepts_kw, evaluate_jBS, geometric_eps
    from bouquet.TokaMaker_interface import _draw_jbs_composer
    from bouquet.utils import pchip_interp
    import OpenFUSIONToolkit.TokaMaker.bootstrap as B

    _harness.assert_bouquet_is_repo_local()
    out = {}
    b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                             header=os.path.join(outdir, "imas"), nthreads=1,
                             reconstruction_engine="legacy")
    b.setup_solver()
    g = b.config.generation
    src = resolve_baseline(b.config, None)
    src_jbs = np.asarray(src.j_BS, dtype=float)

    # ---- legacy baseline (flag OFF): the evaluator on its equilibrium ------
    if part != "core":
        return _imas_probe_loop(outdir, part, b, g, out)
    g.jbs_self_consistent = False            # the legacy path, explicitly
    bl = b.prepare_baseline()
    mygs = b.mygs
    psi = np.asarray(bl.psi_N, dtype=float)
    k2e = lambda a: pchip_interp(np.asarray(bl.psi_N_kinetic, float),
                                 np.asarray(a, float), psi)
    kin = (k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti),
           np.clip(k2e(bl.Zeff), 1.0, None))
    out["grid_uniform"] = bool(np.allclose(np.diff(psi), psi[1] - psi[0]))

    # (a2) the two routes to the geometric eps, on the solved surfaces
    pu = np.unique(np.clip(psi, 1e-3, 1.0 - 1e-3))
    _sau = mygs.sauter_fc(psi=pu.copy())
    _R_sau = np.asarray(_sau[2]["<R>"] if isinstance(_sau[2], dict)
                        else _sau[2][0], dtype=float)
    e_fsa, _r = geometric_eps(mygs, pu, sauter_fn=lambda **k: _sau,
                              R_avg=_R_sau)
    out["a2"] = dict(
        route_fsa=_r, R_avg_fsa_vs_sauter=float(np.max(np.abs(
            np.asarray(mygs.get_fsa(psi=pu.copy())["<R>"]) / _R_sau - 1.0))))
    if _accepts_kw(mygs.sauter_fc, "return_eps"):
        e_fork, r_fork = geometric_eps(mygs, pu)
        out["a2"].update(route_fork=r_fork, max_rel=float(np.max(
            np.abs(e_fsa / e_fork - 1.0))))
    else:
        out["a2"]["skip"] = ("this OFT build's sauter_fc has no return_eps "
                             "(the fork-only option), so only the get_fsa "
                             "route exists here")

    # (a) bit-level against SWB's first Redl evaluation on the same state,
    # with the epsilon this build's SWB itself uses
    _swb_par = inspect.signature(B.solve_with_bootstrap).parameters
    _use_sauter = _swb_par.get("use_sauter_eps")
    swb_eps = ("geometric" if _use_sauter is not None
               and bool(_use_sauter.default) else "a_over_R")
    out["a_eps_definition"] = swb_eps
    if "psi_N" in _swb_par:
        j_new, d_new = evaluate_jBS(mygs, psi, *kin, smooth_axis=False,
                                    eps_definition=swb_eps)
        cap = {}
        orig = B.redl_bootstrap

        class _Stop(Exception):
            pass

        def _spy(*a, **k):
            r = orig(*a, **k)
            cap["j"], cap["kw"] = r[0], k
            raise _Stop()

        B.redl_bootstrap = _spy
        try:
            py = ({"use_python_solve": True} if "use_python_solve" in
                  inspect.signature(B.solve_with_bootstrap).parameters else {})
            B.solve_with_bootstrap(mygs, *kin[:4], kin[4], bl.Ip_target,
                                   np.ones_like(psi), psi_N=psi,
                                   verbose=False, **py)   # the spy is on the Python Redl
        except _Stop:
            pass
        finally:
            B.redl_bootstrap = orig
        out["a"] = dict(
            bitwise=bool(np.array_equal(cap["j"], d_new["j_dot_B"])),
            inputs_bitwise={k: bool(np.array_equal(
                np.asarray(cap["kw"][k]), d_new[m])) for k, m in
                (("fT", "f_T"), ("q", "q"), ("eps", "eps"), ("R", "R_avg"),
                 ("I_psi", "F"))},
            max_abs=float(np.max(np.abs(cap["j"] - d_new["j_dot_B"]))))
    else:
        out["a"] = dict(skip="this OFT build's solve_with_bootstrap has no "
                             "psi_N argument, so its first pass cannot be "
                             "evaluated on the caller's grid")

    # (b) live: uniform vs rho-like non-uniform grid on the same equilibrium
    ju, _ = evaluate_jBS(mygs, psi, *kin, smooth_axis=False)
    rho = np.linspace(0.0, 1.0, 1025)
    xn = rho ** 2
    kin_n = tuple(pchip_interp(psi, np.asarray(a, float), xn) for a in kin)
    jn, dn = evaluate_jBS(mygs, xn, *kin_n, smooth_axis=False)
    jl, _ = evaluate_jBS(mygs, np.linspace(0.0, 1.0, xn.size), *kin_n,
                         smooth_axis=False)      # legacy uniform reading
    w, x, _k = residual_weights(mygs.copy_eq(), psi)
    sel = (psi > 0.02) & (psi < 0.99)
    ref = ju[sel]
    e_new = weighted_norm(np.interp(psi[sel], xn, jn) - ref, w[sel],
                          psi[sel]) / weighted_norm(ref, w[sel], psi[sel])
    e_old = weighted_norm(np.interp(psi[sel], xn, jl) - ref, w[sel],
                          psi[sel]) / weighted_norm(ref, w[sel], psi[sel])
    out["b"] = dict(e_new=float(e_new), e_old=float(e_old))
    # the IMAS-type grid: first intervals finer than psi_pad
    xi = (np.arange(101) / 100.0) ** 2
    xi[1:4] = [1.73e-4, 6.92e-4, 1.557e-3]
    xi = np.sort(xi)
    xi = xi[np.concatenate([[True], np.diff(xi) > 0])]
    kin_i = tuple(pchip_interp(psi, np.asarray(a, float), xi) for a in kin)
    ji, di = evaluate_jBS(mygs, xi, *kin_i)
    out["imas_grid"] = dict(finite=bool(np.all(np.isfinite(ji))),
                            n=int(xi.size),
                            n_inside_pad=int(np.sum(xi < 1e-3)),
                            n_geometry=int(di["n_geometry_surfaces"]),
                            psi_pad=float(di["psi_pad"]))

    # ---- (g) diff mode, loop ON --------------------------------------------
    g.jbs_self_consistent = True
    bl = b.prepare_baseline()
    rec = bl.li_metrics["jbs_loop"]
    peak = float(np.max(np.abs(src_jbs)))
    comp = _draw_jbs_composer(psi, *kin, 1e-3, bool(g.isolate_edge_jBS),
                              1.0, bool(g.floor_j_BS),
                              np.asarray(bl.jBS_diff, float), None, None)
    spike_on_base, _f, _d = comp(b.mygs.copy_eq())
    out["g"] = dict(
        converged=bool(rec["converged"]), n_passes=int(rec["n_passes"]),
        definition=rec.get("jBS_diff_definition"),
        split_identity=float(np.max(np.abs(
            np.asarray(bl.j_BS) + np.asarray(bl.jBS_diff) - src_jbs)) / peak),
        spike_on_baseline=float(np.max(np.abs(spike_on_base - src_jbs))
                                / peak))
    # ---- (f) sigma=0 invariant under the loop (diff baseline) --------------
    s0 = b.verify_sigma0_consistency()
    out["f_imas"] = {k: s0[k] for k in (
        "passed", "loop_converged", "r_j_vs_baseline", "r_I_vs_baseline",
        "dl_i_vs_baseline", "invariant")}
    out["f_imas"]["n_passes"] = int(s0["record"]["n_passes"])
    out["f_imas_draw_route"] = s0.get("draw_route")
    with open(os.path.join(outdir, "imas_core.json"), "w") as fh:
        json.dump(out, fh)


def _imas_probe_loop(outdir, part, b, g, out):
    import numpy as np
    from bouquet.jbs_loop import JBSNotConverged
    g.jbs_self_consistent = True

    # ---- the loop in every other baseline mode / channel -------------------
    def _run(tag, **kw):
        for k_, v_ in kw.items():
            setattr(g, k_, v_)
        blx = b.prepare_baseline()
        r = blx.li_metrics["jbs_loop"]
        ic = blx.ip_closure or {}
        out[tag] = dict(
            converged=bool(r["converged"]), n_passes=int(r["n_passes"]),
            r_j=r["r_j"], r_I=r["r_I"], dl_i=r["dl_i"], dq0=r["dq0"],
            init=r.get("init"), keys=sorted(r.keys()),
            closure_limited=bool(ic.get("closure_limited", False)),
            reasons=list(ic.get("closure_limited_reasons", ()) or ()),
            ip_closure_has_block=("jbs_loop" in ic),
            l_i=float(blx.l_i_target),
            j_BS=[float(v) for v in blx.j_BS],
            n_extra_solves=ic.get("n_extra_solves"),
            sawtooth_verdict=ic.get("sawtooth_verdict"),
            q0_residual=ic.get("q0_residual"),
            q0_solved_predictor=ic.get("q0_solved_predictor"),
            li_pred=ic.get("structured_li_achieved_predictor"),
            li_corr=ic.get("structured_li_achieved_corrected"),
            mse_status=ic.get("structured_mse_status"),
            mse_jbs=ic.get("structured_mse_jbs_loop"),
            mse_chi2_before=ic.get("structured_mse_chi2_before"),
            mse_chi2_after=ic.get("structured_mse_chi2_after"),
            mse_stage=r.get("mse_stage"))
        return blx

    if part == "structured":
        return _imas_probe_structured(outdir, b, g, out, _run)
    _run("rescale", jBS_baseline_mode="rescale")
    _run("ohmic_anchor", jBS_baseline_mode="ohmic",
         closure_channel="bootstrap", jbs_init="anchor")
    _run("ohmic_swb", jBS_baseline_mode="ohmic",
         closure_channel="bootstrap", jbs_init="swb")
    g.jbs_init = "anchor"
    _run("sawtooth", jBS_baseline_mode="ohmic",
         closure_channel="sawtooth_bootstrap")
    # (e) live: a loop that cannot converge (two passes, the first of which
    # starts from the anchor) raises; "flag" delivers it flagged
    g.closure_channel = "bootstrap"
    g.jbs_max_passes = 2
    try:
        b.prepare_baseline()
        out["e_raise"] = dict(raised=False)
    except JBSNotConverged as e:
        out["e_raise"] = dict(raised=True, n_passes=int(e.record["n_passes"]),
                              has_history=bool(e.record.get("r_j")))
    g.jbs_loop_on_fail = "flag"
    _run("e_flag")
    with open(os.path.join(outdir, "imas_modes.json"), "w") as fh:
        json.dump(out, fh)


def _imas_probe_structured(outdir, b, g, out, _run):
    import numpy as np
    g.jBS_baseline_mode = "ohmic"
    from bouquet.config import GenerationConfig
    # the shipped default ceiling (see the note at the top of the module)
    assert g.jbs_max_passes == GenerationConfig().jbs_max_passes
    _run("structured", closure_channel="structured",
         structured_li_target=None)
    # ---- the MSE composition: synthetic chords off the delivered field ----
    from bouquet.mse import mse_equilibrium_orientation, mse_field_at
    R0 = float(np.asarray(b.mygs.o_point, dtype=float)[0])
    R = np.linspace(R0 + 0.05, R0 + 0.55, 8)
    Z = np.zeros_like(R)
    Bf, _found = mse_field_at(b.mygs, R, Z)
    assert _found.all(), "a synthetic chord is off the mesh"
    tg = Bf[:, 2] / Bf[:, 1]
    # the chords come from the equilibrium's own field: state ITS orientation
    _or = mse_equilibrium_orientation(Bf, R, Z, b.mygs.o_point)
    g.mse_data = dict(R=list(R), Z=list(Z), tgamma=list(1.03 * tg),
                      sigma=[0.004] * 8, weight=[1.0] * 8,
                      A1=[1.0] * 8, A2=[1.0] * 8, A3=[0.0] * 8, A4=[0.0] * 8,
                      ip_sign=float(_or["ip"]), bt_sign=float(_or["bt"]))
    g.structured_mse_required = True
    _run("mse")
    with open(os.path.join(outdir, "imas_structured.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  probe 2: the reconstruction (geqdsk) path
# ---------------------------------------------------------------------------
def _recon_probe(outdir):
    import numpy as np
    import bouquet as bq
    from bouquet import perturb_kinetic_equilibrium
    from bouquet.jbs_loop import jbs_settings, profile_residuals, \
        residual_weights
    from bouquet.utils import pchip_interp

    _harness.assert_bouquet_is_repo_local()
    out = {}
    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "rec"), n_draws=1,
                               reconstruction_engine="legacy")
    g = b.config.generation
    g.jbs_self_consistent = True
    b.setup_solver()
    bl = b.prepare_baseline()
    rec = bl.reconstruction_metrics["jbs_loop"]
    out["recon"] = dict(converged=bool(rec["converged"]),
                        n_passes=int(rec["n_passes"]), r_j=rec["r_j"],
                        r_I=rec["r_I"], dl_i=rec["dl_i"],
                        post=rec.get("post_corrective"),
                        l_i_target=float(bl.l_i_target))
    s0 = b.verify_sigma0_consistency()
    out["f_recon"] = {k: s0[k] for k in (
        "passed", "loop_converged", "r_j_vs_baseline", "r_I_vs_baseline",
        "dl_i_vs_baseline")}
    out["f_recon_draw_route"] = s0.get("draw_route")

    # a sigma=0 route-R2 (Fix C) draw with the loop
    psi_N = np.asarray(bl.psi_N, dtype=float)
    psi_kin = np.asarray(bl.psi_N_kinetic, dtype=float)
    psi_pad = float(getattr(b.config.source, "psi_pad", 1e-3))
    k2e = lambda a: pchip_interp(psi_kin, np.asarray(a, float), psi_N)
    from bouquet.physics import ELEMENTARY_CHARGE as _EC   # loop-on draws
    pressure = _EC * (k2e(bl.ne) * k2e(bl.te)
                             + k2e(bl.ni) * k2e(bl.ti))
    Zeff_eq = np.clip(k2e(bl.Zeff), 1.0, None)
    zk, zj = np.zeros_like(psi_kin), np.zeros_like(psi_N)
    d = perturb_kinetic_equilibrium(
        b.mygs, psi_N, pressure, bl.ne, bl.te, bl.ni, bl.ti,
        np.asarray(bl.j_phi, dtype=float), zk, zk, zk, zk, zj,
        0.5, 0.4, 0.25, float(bl.Ip_target), float(bl.l_i_target), Zeff_eq,
        len(psi_N), input_jinductive=np.asarray(bl.j_inductive, float),
        l_i_tolerance=g.l_i_tolerance, psi_pad=psi_pad,
        constrain_sawteeth=False, recalculate_j_BS=True,
        isolate_edge_jBS=g.isolate_edge_jBS, floor_j_BS=g.floor_j_BS,
        scale_jBS=float(getattr(bl, "bs_scale", 1.0)),
        perturb_jind_in_anchor=True, accept_anchor_inband=False,
        psi_N_kinetic=psi_kin, p_thresh=0.05, rng=12345,
        jbs_loop=jbs_settings(g, draw=True))[6]
    w, x, _k = residual_weights(b.mygs.copy_eq(), psi_N, psi_pad)
    cmp_ = profile_residuals(np.asarray(d["j_BS"], float),
                             np.asarray(bl.j_BS, float), w, x,
                             float(bl.Ip_target))
    out["f_draw"] = dict(
        converged=bool(d["jbs_loop"]["converged"]),
        n_passes=int(d["jbs_loop"]["n_passes_total"]),
        r2_scale=float(d["r2_ip_scale"]), r2_f_ind=d.get("r2_f_ind"),
        r_j_vs_baseline=float(cmp_["r_j"]),
        r_I_vs_baseline=float(cmp_["r_I"]),
        li=float(b.mygs.get_stats(lcfs_pad=psi_pad,
                                  li_normalization="iter")["l_i"]),
        l_i_target=float(bl.l_i_target),
        li_realized_post_corrective=(
            None if (bl.recon or {}).get("li_realized_post_corrective")
            is None else float(bl.recon["li_realized_post_corrective"])),
        ctx_private="_jbs_ctx" in d,
        init_source=d["jbs_loop"].get("init_source"),
        loops=d["jbs_loop"].get("loops"))

    # ---- a perturbed draw's anchor loop from two starts (plan 7f) --------
    # The draw's own start (Redl at its state anchor on its OWN perturbed
    # kinetics) and the counterfactual (the unperturbed baseline bootstrap),
    # through the same composition and the same step the standard draw's
    # anchor loop uses (inductive held, bootstrap swapped, jphi-linterp solve
    # of the beta-relaxed total).  Deterministic kinetic perturbation.
    from bouquet.jbs_loop import run_jbs_loop
    from bouquet.TokaMaker_interface import _draw_jbs_composer
    from bouquet.utils import pchip_derivative
    ne_d = k2e(bl.ne) * 0.95
    te_d = k2e(bl.te) * 1.08
    ni_d = k2e(bl.ni) * 0.95
    ti_d = k2e(bl.ti) * 1.04
    from bouquet.physics import ELEMENTARY_CHARGE as _EC   # loop-on draws
    pres_d = _EC * (ne_d * te_d + ni_d * ti_d)
    j_ind = np.asarray(bl.j_inductive, dtype=float)
    j_phi_bl = np.asarray(bl.j_phi, dtype=float)
    j_fix = j_phi_bl - j_ind - np.asarray(bl.j_BS, dtype=float)
    compose = _draw_jbs_composer(
        psi_N, ne_d, te_d, ni_d, ti_d, Zeff_eq, psi_pad,
        bool(g.isolate_edge_jBS), float(getattr(bl, "bs_scale", 1.0)),
        bool(g.floor_j_BS), None, None, None)
    Ip = float(bl.Ip_target)
    mygs = b.mygs

    def _solve(j):
        _pr = mygs.psi_bounds[1] - mygs.psi_bounds[0]
        _pp = {"type": "linterp",
               "y": pchip_derivative(psi_N, pres_d) / _pr, "x": psi_N}
        _pp["y"][-1] = 0.0
        mygs.set_targets(Ip=Ip, pax=float(pres_d[0]))
        mygs.set_profiles(pp_prof=_pp, ffp_prof={
            "type": "jphi-linterp", "y": np.asarray(j, float), "x": psi_N})
        mygs.solve()

    def _meas():
        _snap = mygs.copy_eq()
        _w, _x, _k = residual_weights(_snap, psi_N, psi_pad)
        return dict(w=_w, x=_x, snap=_snap, li=float(mygs.get_stats(
            li_normalization="iter", lcfs_pad=psi_pad)["l_i"]))

    def _step(spk, k, relax=None):
        _j = j_ind + spk + j_fix
        _solve(_j if relax is None else relax(_j))
        return _meas()

    out["draw_init"] = {}
    for tag in ("baseline_jBS", "own_kinetics"):
        _solve(j_phi_bl)                   # the draw's state anchor
        li0 = _meas()["li"]
        jbs0 = (np.asarray(bl.j_BS, dtype=float) if tag == "baseline_jBS"
                else compose(mygs.copy_eq())[0])
        r = run_jbs_loop(jbs0, _step, lambda m: compose(m["snap"])[0],
                         jbs_settings(g, draw=True), Ip=abs(Ip),
                         meas0=dict(li=li0), gate_li=True,
                         label=f"draw init {tag}", init_source=tag,
                         raise_on_fail=False)
        out["draw_init"][tag] = dict(
            converged=bool(r["converged"]),
            n_passes=int(r["record"]["n_passes"]),
            r_j=r["record"]["r_j"], init_source=r["record"]["init_source"],
            j_BS=np.asarray(r["jbs_used"], float).tolist(),
            li=float(_meas()["li"]))

    # ---- B1: the post-homotopy re-solves go through the saturation guard --
    # The sigma=0 route-R2 draw's own rebuild context, its bootstrap nudged
    # 2 % off Redl so the stage must re-solve.  The guard's reference coils
    # sit 2 % below the coils mygs holds, against a 1 % bound: the FIRST
    # real re-solve must trip it (CoilSaturated), exactly as the homotopy's
    # own guard treats a pass on its bound.  The wide-bound control shows the
    # guard is called on every pass and passes a solve with headroom.
    from bouquet.TokaMaker_interface import (CoilSaturated,
                                             _make_coil_saturation_guard,
                                             _post_homotopy_jbs)
    from bouquet.jbs_loop import JBSNotConverged
    ctx = dict(d["_jbs_ctx"])
    ctx["spike_used"] = 1.02 * np.asarray(ctx["spike_used"], dtype=float)
    out["b1"] = {}
    for tag, lim in (("saturating", 0.01), ("headroom", 10.0)):
        _solve(j_phi_bl)
        cur, _ = mygs.get_coil_currents()
        base = {n: float(v) / 1.02 for n, v in cur.items()}
        guard = _make_coil_saturation_guard(mygs, base, ("F9A", "F9B"),
                                            lim, lim, stage=f"probe {tag}")
        try:
            _post_homotopy_jbs(mygs, ctx, jbs_settings(g, draw=True), psi_N,
                               psi_pad, float(bl.Ip_target), coil_guard=guard)
            outcome = "kept"
        except CoilSaturated:
            outcome = "CoilSaturated"
        except JBSNotConverged:
            outcome = "JBSNotConverged"
        out["b1"][tag] = dict(outcome=outcome, checks=list(guard.log))
    with open(os.path.join(outdir, "recon.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  probe 3: the legacy flag (jbs_self_consistent=False)
# ---------------------------------------------------------------------------
def _legacy_probe(outdir, part):
    """``jbs_self_consistent=False`` -- the flag that keeps the frozen-SWB
    bootstrap now that the loop is the default -- must take the legacy path
    everywhere: the loop kernel and the Redl evaluator are never entered,
    ``solve_with_bootstrap`` is, and nothing loop-shaped is recorded or
    archived.  The evaluator and the kernel are replaced by tripwires, so an
    accidental call fails the probe at the call site rather than being
    averaged into a number.  (The bit-level half of the A/B -- flag OFF
    equals the pre-loop tree array for array -- needs the pre-loop tree and is
    run out of tree on each change of the legacy path; the tripwires are what
    can live here.)  ``part``: "recon" (the geqdsk path) or "imas" -- two
    interpreters, because ``OFT_env`` is a per-process singleton."""
    import h5py
    import numpy as np
    import bouquet as bq
    import bouquet.jbs_loop as L
    import bouquet.physics as P
    import OpenFUSIONToolkit.TokaMaker.bootstrap as B

    _harness.assert_bouquet_is_repo_local()
    calls = {"evaluate_jBS": 0, "run_jbs_loop": 0, "solve_with_bootstrap": 0}

    def _tripwire(name):
        def _f(*a, **k):
            calls[name] += 1
            raise AssertionError(f"{name} entered with "
                                 "jbs_self_consistent=False")
        return _f

    P.evaluate_jBS = _tripwire("evaluate_jBS")
    L.run_jbs_loop = _tripwire("run_jbs_loop")
    _swb = B.solve_with_bootstrap

    import functools

    @functools.wraps(_swb)        # keeps the signature bouquet inspects
    def _count_swb(*a, **k):
        calls["solve_with_bootstrap"] += 1
        return _swb(*a, **k)

    B.solve_with_bootstrap = _count_swb

    def _archived_jbs_attrs(h5):
        found = []
        with h5py.File(h5, "r") as hf:
            def _v(name, obj):
                if isinstance(obj, h5py.Group):
                    found.extend(f"{name}:{k}" for k in obj.attrs
                                 if k.startswith("jbs_"))
            hf.visititems(_v)
        return found

    out = {}
    if part == "imas":
        return _legacy_probe_imas(outdir, calls, _archived_jbs_attrs, out)
    # ---- the reconstruction path: recon, sigma=0 check, one draw ----------
    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "leg_rec"),
                               n_draws=1, reconstruction_engine="legacy")
    b.config.generation.seed = 12345
    b.config.generation.jbs_self_consistent = False
    b.setup_solver()
    bl = b.prepare_baseline()
    n0 = calls["solve_with_bootstrap"]
    s0 = b.verify_sigma0_consistency()
    b.generate()
    out["recon"] = dict(
        swb_recon=n0, swb_total=calls["solve_with_bootstrap"],
        metrics_has_loop="jbs_loop" in (bl.reconstruction_metrics or {}),
        sigma0_has_loop=("record" in s0) or ("loop_converged" in s0),
        archived=_archived_jbs_attrs(b.config.output_header + ".h5"))
    out["calls"] = dict(calls)
    with open(os.path.join(outdir, "legacy_recon.json"), "w") as fh:
        json.dump(out, fh)


def _legacy_probe_imas(outdir, calls, _archived_jbs_attrs, out):
    import bouquet as bq
    # ---- the IMAS path: the default diff baseline + one draw, and the ohmic
    # structured closure (the channels the loop rewires most) ---------------
    bi = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                              header=os.path.join(outdir, "leg_imas"),
                              nthreads=1, reconstruction_engine="legacy")
    gi = bi.config.generation
    gi.seed = 12345
    gi.jbs_self_consistent = False
    bi.setup_solver()
    n1 = calls["solve_with_bootstrap"]
    bli = bi.prepare_baseline()
    n2 = calls["solve_with_bootstrap"]
    bi.generate()
    diff_rec = dict(li_has_loop="jbs_loop" in (bli.li_metrics or {}),
                    ip_has_loop="jbs_loop" in (bli.ip_closure or {}),
                    swb_baseline=n2 - n1,
                    archived=_archived_jbs_attrs(
                        bi.config.output_header + ".h5"))
    gi.jBS_baseline_mode = "ohmic"
    gi.closure_channel = "structured"
    gi.structured_li_target = None
    n3 = calls["solve_with_bootstrap"]
    bls = bi.prepare_baseline()
    out["imas"] = dict(
        diff=diff_rec,
        structured=dict(
            li_has_loop="jbs_loop" in (bls.li_metrics or {}),
            ip_has_loop="jbs_loop" in (bls.ip_closure or {}),
            swb_baseline=calls["solve_with_bootstrap"] - n3))
    out["calls"] = dict(calls)
    with open(os.path.join(outdir, "legacy_imas.json"), "w") as fh:
        json.dump(out, fh)


def _run_probe(tmp_path_factory, which, fname):
    work = tmp_path_factory.mktemp(which)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), which, str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"{which} probe failed (rc={proc.returncode}):\n"
                    f"{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}")
    with open(str(work / fname)) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def imas(tmp_path_factory):
    return _run_probe(tmp_path_factory, "imas_core", "imas_core.json")


@pytest.fixture(scope="module")
def imas_modes(tmp_path_factory):
    return _run_probe(tmp_path_factory, "imas_modes", "imas_modes.json")


@pytest.fixture(scope="module")
def imas_structured(tmp_path_factory):
    return _run_probe(tmp_path_factory, "imas_structured",
                      "imas_structured.json")


@pytest.fixture(scope="module")
def recon(tmp_path_factory):
    return _run_probe(tmp_path_factory, "recon", "recon.json")


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    r = _run_probe(tmp_path_factory, "legacy_recon", "legacy_recon.json")
    i = _run_probe(tmp_path_factory, "legacy_imas", "legacy_imas.json")
    calls = {k: r["calls"][k] + i["calls"][k] for k in r["calls"]}
    return dict(recon=r["recon"], imas=i["imas"], calls=calls)


_S = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3, tol_q0=2e-3)


def _rj(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


# ---------------------------------------------------------------------------
#  the evaluator
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
def test_a_evaluator_equals_swbs_first_pass_bit_for_bit(imas):
    a = imas["a"]
    if "skip" in a:
        pytest.skip(a["skip"])
    assert imas["grid_uniform"], "the synthetic dd grid should be uniform"
    assert all(a["inputs_bitwise"].values()), a["inputs_bitwise"]
    assert a["bitwise"], f"<j.B> differs from SWB's first pass by {a['max_abs']}"


@pytest.mark.solver
@solver_only
def test_a2_the_two_geometric_eps_routes_agree_to_rounding(imas):
    a2 = imas["a2"]
    assert a2["route_fsa"] == "get_fsa"
    if "skip" in a2:
        pytest.skip(a2["skip"])
    assert a2["route_fork"] == "sauter_fc(return_eps=True)"
    assert a2["max_rel"] <= 1e-12, a2


@pytest.mark.solver
@solver_only
def test_b_grid_independence_on_a_live_equilibrium(imas):
    b = imas["b"]
    assert b["e_new"] < 0.02, b
    assert b["e_old"] > 10.0 * b["e_new"], b


@pytest.mark.solver
@solver_only
def test_imas_type_grid_finer_than_psi_pad_is_handled(imas):
    ig = imas["imas_grid"]
    assert ig["finite"] and ig["psi_pad"] == 1e-3
    # the points inside the pad share one geometry sample; nothing merged
    assert ig["n_geometry"] == ig["n"] - ig["n_inside_pad"] + 1


# ---------------------------------------------------------------------------
#  the IMAS baseline under the loop
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
def test_g_diff_mode_sigma0_reproduces_the_source_bootstrap_exactly(imas):
    g = imas["g"]
    assert g["converged"] and g["n_passes"] == 0
    assert "pure model offset" in g["definition"]
    # the baseline split and the sigma=0 draw's first evaluation on the
    # delivered baseline both ARE the source j_BS, to rounding
    assert g["split_identity"] <= 1e-12, g
    assert g["spike_on_baseline"] <= 1e-12, g


@pytest.mark.solver
@solver_only
def test_f_sigma0_draw_loop_converges_back_to_the_imas_baseline(imas):
    f = imas["f_imas"]
    assert f["invariant"] == "jbs-loop"
    assert f["loop_converged"], f
    assert f["r_j_vs_baseline"] <= _S["rtol_j"], f
    assert f["r_I_vs_baseline"] <= _S["rtol_Ip"], f
    assert f["dl_i_vs_baseline"] <= _S["tol_li"], f
    assert f["passed"], f


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix, tag", [
    ("imas_modes", "rescale"), ("imas_modes", "ohmic_anchor"),
    ("imas_modes", "ohmic_swb"), ("imas_modes", "sawtooth"),
    ("imas_structured", "structured"), ("imas_structured", "mse")])
def test_every_mode_converges_and_records_its_block(request, fix, tag):
    r = request.getfixturevalue(fix)[tag]
    assert r["converged"], (tag, r["r_j"], r["r_I"], r["dl_i"], r["dq0"])
    for k in ("enabled", "init", "grid", "n_passes", "converged",
              "tolerances", "omega", "r_j", "r_I", "dl_i", "dq0", "I_BS",
              "jBS_peak_psiN", "jBS_peak", "wall_s", "evaluate_jBS_version",
              "oft_build"):
        assert k in r["keys"], (tag, k)
    # the last two passes meet every active criterion
    for i in (-1, -2):
        assert r["r_j"][i] <= _S["rtol_j"] and r["r_I"][i] <= _S["rtol_Ip"]
        assert r["dl_i"][i] is not None and r["dl_i"][i] <= _S["tol_li"]
    if tag != "rescale":
        assert r["ip_closure_has_block"]


@pytest.mark.solver
@solver_only
def test_d_the_fixed_point_does_not_depend_on_the_initial_guess(imas_modes):
    a, s = imas_modes["ohmic_anchor"], imas_modes["ohmic_swb"]
    assert a["init"] == "anchor" and s["init"] == "swb"
    # each delivered bootstrap is within rtol_j of its own fixed-point
    # residual; two converged runs of a contraction sit within a few rtol_j
    assert _rj(a["j_BS"], s["j_BS"]) <= 5 * _S["rtol_j"]
    assert abs(a["l_i"] - s["l_i"]) <= 2 * _S["tol_li"]


@pytest.mark.solver
@solver_only
def test_e_non_convergence_raises_and_flag_mode_flags_the_slice(imas_modes):
    e = imas_modes["e_raise"]
    assert e["raised"], e
    assert e["n_passes"] == 2 and e["has_history"]
    f = imas_modes["e_flag"]
    assert f["converged"] is False
    assert f["closure_limited"]
    assert any(str(x).startswith("j_BS loop: ") for x in f["reasons"])


@pytest.mark.solver
@solver_only
def test_axis_row_channels_subsume_the_corrector_with_bookkeeping(
        imas_modes, imas_structured):
    for r in (imas_modes["sawtooth"], imas_structured["structured"]):
        assert r["n_extra_solves"] == r["n_passes"] - 1
        assert "j_BS loop" in str(r["sawtooth_verdict"])
        assert r["q0_residual"] is not None
        assert r["q0_solved_predictor"] is not None
        # the q0 acceptance is still the unchanged q0_tol flag
        assert abs(r["q0_residual"]) <= 0.01 or any(
            "q0" in str(x) for x in r["reasons"])


@pytest.mark.solver
@solver_only
def test_mse_stage_composes_with_the_loop(imas_structured):
    m = imas_structured["mse"]
    assert m["mse_status"] == "applied" and m["mse_jbs"] is True
    st = m["mse_stage"]
    assert st["converged"], st
    assert "jacobian_refresh_rel_change" in st
    assert "objective_change_final_step" in st
    assert m["mse_chi2_after"] <= m["mse_chi2_before"]


# ---------------------------------------------------------------------------
#  the reconstruction path
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
def test_reconstruction_loop_converges(recon):
    r = recon["recon"]
    assert r["converged"], r
    assert r["post"] is not None


@pytest.mark.solver
@solver_only
def test_f_sigma0_invariant_on_the_reconstruction_path(recon):
    f = recon["f_recon"]
    assert f["loop_converged"] and f["passed"], f


@pytest.mark.solver
@solver_only
def test_f_sigma0_route_r2_draw_converges_near_the_baseline(recon):
    d = recon["f_draw"]
    assert d["converged"], d
    # the route-R2 sigma=0 invariant keeps its own, unchanged budget
    # (|s-1|*f_ind <= 3.86e-3, tests/test_seeded_reproducibility.py) with the
    # loop's bootstrap in place of the frozen SWB spike
    assert abs(d["r2_scale"] - 1.0) * float(d["r2_f_ind"]) <= 3.86e-3, d
    # ...and, at the loop's OWN tolerances, the unperturbed draw reproduces
    # the baseline: its bootstrap (r_j) and its l_i against the target the
    # draws are banded on.  No bar is widened for the draw route: a failure
    # here is a finding about the draw route (review M1), not a test to tune.
    assert d["r_j_vs_baseline"] <= _S["rtol_j"], d
    assert abs(d["li"] - d["l_i_target"]) <= _S["tol_li"], d
    # the private rebuild context travels on the diagnostics until
    # generate_bouquet pops it before archiving
    assert d["ctx_private"]
    # the draw's first loop started from Redl at its anchor on its own
    # kinetics, and the per-draw block says so
    assert d["init_source"].startswith(
        "evaluate_jBS on the draw's state-anchor equilibrium"), d
    assert d["loops"] and d["loops"][0]["init_source"] == d["init_source"]


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix, key", [("imas", "f_imas_draw_route"),
                                      ("recon", "f_recon_draw_route")])
def test_sigma0_check_reports_the_draw_route_beside_its_verdict(request, fix,
                                                               key):
    """The sigma=0 check carries the draw's own route at zero perturbation
    as a SEPARATE block.  RE-SCOPED BY THE OWNER'S DECISION (the
    zero-perturbation identity of the draws): its verdict used to gate
    nothing; it now decides the check's ``passed`` (asserted by the tests
    above and in tests/test_sigma0_identity_solver.py).  Only the presence
    and completeness of the measurement is asserted here."""
    blk = request.getfixturevalue(fix)[key]
    assert blk is not None and blk["routes"], blk
    assert "passed_draw_route" in blk
    assert blk["gates"].startswith("verify_sigma0_consistency's `passed`")
    for route, rr in blk["routes"].items():
        assert rr.get("error") is None, (route, rr)
        for k in ("loop_converged", "passes_used", "r_j", "r_I", "li_draw",
                  "dl_i_vs_l_i_target", "dl_i_vs_delivered",
                  "passed_draw_route"):
            assert k in rr, (route, k)


@pytest.mark.solver
@solver_only
def test_draw_fixed_point_does_not_depend_on_its_start(recon):
    """Plan 7f: a draw's loop starts from Redl at its anchor on its OWN
    perturbed kinetics.  Initialisation only: the same draw loop started from
    the unperturbed baseline bootstrap (the counterfactual) converges to the
    same fixed point; the own-kinetics start begins closer to it."""
    a = recon["draw_init"]["baseline_jBS"]
    b = recon["draw_init"]["own_kinetics"]
    assert a["converged"] and b["converged"], (a["r_j"], b["r_j"])
    assert a["init_source"] == "baseline_jBS"
    assert b["init_source"] == "own_kinetics"
    # the same bar as the baseline's init-independence test (d)
    assert _rj(a["j_BS"], b["j_BS"]) <= 5 * _S["rtol_j"]
    assert abs(a["li"] - b["li"]) <= 2 * _S["tol_li"]
    assert b["r_j"][0] < a["r_j"][0]


@pytest.mark.solver
@solver_only
def test_b1_post_homotopy_resolves_are_guarded_against_coil_saturation(
        recon):
    """B1 on the live solver: a post-homotopy re-solve that ends with the coils
    beyond 0.99 x the bound rejects the draw after its FIRST re-solve; with
    headroom every pass is checked and none is saturated."""
    b1 = recon["b1"]
    sat = b1["saturating"]
    assert sat["outcome"] == "CoilSaturated", sat
    assert len(sat["checks"]) == 1 and sat["checks"][0]["saturated"], sat
    assert sat["checks"][0]["label"].startswith("post-homotopy pass 1"), sat
    ok = b1["headroom"]
    assert ok["outcome"] != "CoilSaturated", ok
    assert ok["checks"] and not any(c["saturated"] for c in ok["checks"]), ok


# ---------------------------------------------------------------------------
#  the legacy flag
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
def test_the_legacy_flag_never_enters_the_loop(legacy):
    """A/B guard of the default flip: ``jbs_self_consistent=False`` runs the
    frozen-SWB bootstrap in the reconstruction, the sigma=0 check, the draws,
    the IMAS diff baseline and the structured closure, and neither the loop
    kernel nor the Redl evaluator is ever called."""
    c = legacy["calls"]
    assert c["evaluate_jBS"] == 0 and c["run_jbs_loop"] == 0, c
    r = legacy["recon"]
    assert r["swb_recon"] >= 1, r               # the recon's own SWB call
    assert r["swb_total"] > r["swb_recon"], r   # + the sigma=0 check / draw
    assert not r["metrics_has_loop"] and not r["sigma0_has_loop"], r
    assert r["archived"] == [], r["archived"]
    d, st = legacy["imas"]["diff"], legacy["imas"]["structured"]
    assert d["swb_baseline"] >= 1 and st["swb_baseline"] >= 1, legacy["imas"]
    for blk in (d, st):
        assert not blk["li_has_loop"] and not blk["ip_has_loop"], blk
    assert d["archived"] == [], d["archived"]


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    which, outdir = sys.argv[1], sys.argv[2]
    if which.startswith("imas_"):
        _imas_probe(outdir, which[len("imas_"):])
    elif which == "recon":
        _recon_probe(outdir)
    elif which.startswith("legacy_"):
        _legacy_probe(outdir, which[len("legacy_"):])
    else:
        raise SystemExit(f"unknown probe {which!r}")
