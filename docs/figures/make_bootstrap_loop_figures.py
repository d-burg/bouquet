"""Figures for the self-consistent bootstrap loop (docs/proposals/self-consistent-bootstrap-PR.md).

Everything here runs on the public synthetic example only: the D3D-like
g-file + p-file + mesh in ``examples/D3D-like`` and the configuration stored
in the git-tracked golden fixture (``tests/golden/D3Dlike_Hmode_golden_slim.h5``,
``scan/0/config_json``), read exactly the way
``tests/golden/regenerate_golden_run.py`` reads it.  No other data.  Only the
reconstruction (geqdsk) path is used -- single-slice solves, no draws, no IMAS.

Each figure's computation is its own subcommand and runs in its own process
(``OFT_env`` is a per-process singleton); results are cached as JSON in a work
directory and the ``plot`` subcommand draws the figures from the cache::

    python docs/figures/make_bootstrap_loop_figures.py recon --flag off
    python docs/figures/make_bootstrap_loop_figures.py recon --flag on     # figs 1-3
    python docs/figures/make_bootstrap_loop_figures.py init --start swb    # fig 4
    python docs/figures/make_bootstrap_loop_figures.py init --start x0.8
    python docs/figures/make_bootstrap_loop_figures.py init --start x1.2
    python docs/figures/make_bootstrap_loop_figures.py sigma0              # fig 5
    python docs/figures/make_bootstrap_loop_figures.py plot [--fig N]
    python docs/figures/make_bootstrap_loop_figures.py all                 # all of the above

``--work DIR`` (default: ``$TMPDIR/bouquet_jbs_loop_figures``) holds the
per-stage run directories and the JSON cache.  Every compute stage is one
thread (``OMP_NUM_THREADS=1``, and the stored config's ``solver.nthreads=1``)
and takes one to a few minutes.

Nothing in the library is changed or configured beyond the documented
``GenerationConfig`` fields (``jbs_self_consistent``, ``jbs_init``).  Two
figure-only instruments wrap the loop kernel for the duration of one stage:

* ``recon --flag on`` records the on-axis safety factor of each pass's
  equilibrium into the pass measurement, so the loop logs Delta q0.  On the
  reconstruction path Delta q0 is NOT a convergence criterion (``gate_q0`` is
  False there and stays False); the numbers are logged only.
* ``init --start x0.8 / x1.2`` multiplies the reconstruction loop's initial
  bootstrap iterate (the anchor evaluation) by the factor -- a deliberately
  wrong start, to show the fixed point does not depend on it.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIXTURE = os.path.join(_REPO, "tests", "golden", "D3Dlike_Hmode_golden_slim.h5")
_REGEN = os.path.join(_REPO, "tests", "golden", "regenerate_golden_run.py")
_DEFAULT_WORK = os.path.join(tempfile.gettempdir(), "bouquet_jbs_loop_figures")

#: label prefix of the reconstruction's main loop (bouquet.TokaMaker_interface)
_RECON_LOOP_LABEL = "geqdsk reconstruction (fit"

#: Wong (2011) colour-blind-safe palette
WONG = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9",
            green="#009E73", yellow="#F0E442", blue="#0072B2",
            vermillion="#D55E00", purple="#CC79A7")


# ---------------------------------------------------------------------------
#  set-up shared by every compute stage
# ---------------------------------------------------------------------------
def _regen_module():
    spec = importlib.util.spec_from_file_location("_regen_golden", _REGEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bouquet_for(work, tag, *, loop_on, init="anchor"):
    """A ``Bouquet`` on the synthetic reconstruction, configured from the
    golden fixture's stored config (the golden recipe's reader), with the
    self-consistent loop ON or OFF.  Returns ``(b, cfg)`` after
    ``setup_solver()``; the caller runs ``prepare_baseline()``."""
    sys.path.insert(0, _REPO)
    regen = _regen_module()
    import bouquet as bq
    got = os.path.abspath(bq.__file__)
    if not got.startswith(os.path.join(_REPO, "")):
        raise SystemExit(f"bouquet imported from {got}, not from {_REPO}")
    cfg = regen.load_stored_config(_FIXTURE)
    cfg.output_header = f"jbs_fig_{tag}"
    cfg.verbose = False
    cfg.generation.jbs_self_consistent = bool(loop_on)
    cfg.generation.jbs_init = str(init)
    run_dir = os.path.join(work, f"run_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    regen.link_example_inputs(cfg, run_dir)
    os.chdir(run_dir)
    b = bq.Bouquet(cfg)
    b.setup_solver()
    return b, cfg


def _kin_on_eq(bl):
    import numpy as np
    from bouquet.utils import pchip_interp
    psi = np.asarray(bl.psi_N, dtype=float)
    k2e = lambda a: pchip_interp(np.asarray(bl.psi_N_kinetic, float),  # noqa: E731
                                 np.asarray(a, float), psi)
    return (k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti),
            np.clip(k2e(bl.Zeff), 1.0, None))


def _q0(eq, psi_pad):
    import numpy as np
    return float(np.asarray(eq.get_q(psi=np.array([psi_pad, 0.5]))[1],
                            dtype=float)[0])


def _delivered(b, bl, cfg):
    """Profiles, integrals and scalars of the delivered baseline equilibrium.

    Currents are integrated with the loop's own residual measure
    (:func:`bouquet.jbs_loop.residual_weights` on the delivered equilibrium).
    The bootstrap FRACTION the figures quote is ``I_BS / I_phi`` -- both
    integrals of the delivered profiles in that one measure, so the measure's
    normalisation cancels; ``Ip`` (the reconstruction target) is kept too.
    """
    import numpy as np
    from bouquet.jbs_loop import residual_weights, _trap
    psi_pad = float(cfg.source.psi_pad)
    psi = np.asarray(bl.psi_N, dtype=float)
    snap = b.mygs.copy_eq()
    w, x, kind = residual_weights(snap, psi, psi_pad)
    Ip = abs(float(bl.Ip_target))
    I = lambda j: abs(_trap(w * np.asarray(j, float), x))  # noqa: E731
    st = b.mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
    return dict(
        psi_N=psi.tolist(), j_BS=np.asarray(bl.j_BS, float).tolist(),
        j_inductive=np.asarray(bl.j_inductive, float).tolist(),
        j_phi=np.asarray(bl.j_phi, float).tolist(),
        Ip=Ip, I_BS=I(bl.j_BS), I_ind=I(bl.j_inductive), I_phi=I(bl.j_phi),
        weights_kind=kind, l_i_target=float(bl.l_i_target),
        l_i_delivered=float(st["l_i"]), q0=_q0(snap, psi_pad),
        w=np.asarray(w, float).tolist())


def _dump(work, name, obj):
    from bouquet.jbs_loop import jsonable
    path = os.path.join(work, name)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh)
    print(f"wrote {path}", flush=True)


def _wrap_loop(transform_jbs0=None, log_q0_psi_pad=None, mygs=None):
    """Figure-only instrument around ``bouquet.jbs_loop.run_jbs_loop`` for the
    reconstruction's main loop: optionally scale its initial iterate, and/or
    add the pass equilibrium's q0 to each pass measurement (logged only;
    ``gate_q0`` is passed through unchanged)."""
    import numpy as np
    import bouquet.jbs_loop as L
    orig = L.run_jbs_loop
    calls = []

    def _run(jbs0, step, evaluate, settings, **kw):
        label = str(kw.get("label", ""))
        if not label.startswith(_RECON_LOOP_LABEL):
            return orig(jbs0, step, evaluate, settings, **kw)
        calls.append(label)
        if transform_jbs0 is not None:
            jbs0 = transform_jbs0(np.asarray(jbs0, dtype=float))
        if log_q0_psi_pad is not None:
            takes_relax = "relax" in inspect.signature(step).parameters

            def _add(meas):
                meas = dict(meas)
                meas["q0"] = _q0(meas["snap"], log_q0_psi_pad)
                return meas
            if takes_relax:
                def _step(jbs, k, relax=None):
                    return _add(step(jbs, k, relax=relax))
            else:
                def _step(jbs, k):
                    return _add(step(jbs, k))
            if mygs is not None:
                # q0 of E_0 (the solver holds the anchor when the loop starts)
                kw["meas0"] = dict(kw.get("meas0") or {},
                                   q0=_q0(mygs.copy_eq(), log_q0_psi_pad))
            return orig(jbs0, _step, evaluate, settings, **kw)
        return orig(jbs0, step, evaluate, settings, **kw)

    L.run_jbs_loop = _run
    return calls


# ---------------------------------------------------------------------------
#  compute stages
# ---------------------------------------------------------------------------
def stage_recon(work, flag):
    """Figure 1 (both flags), 2 and 3 (flag on)."""
    import numpy as np
    loop_on = flag == "on"
    t0 = time.time()
    b, cfg = _bouquet_for(work, f"recon_{flag}", loop_on=loop_on)
    psi_pad = float(cfg.source.psi_pad)
    calls = (_wrap_loop(log_q0_psi_pad=psi_pad, mygs=b.mygs) if loop_on
             else [])
    bl = b.prepare_baseline()
    out = dict(flag=flag, delivered=_delivered(b, bl, cfg),
               wall_s=time.time() - t0)
    if loop_on:
        if len(calls) != 1:
            raise RuntimeError(f"expected one reconstruction loop, saw {calls}")
        out["record"] = (bl.reconstruction_metrics or {}).get("jbs_loop")
        out["grid"] = _grid_regression(b, bl, psi_pad)
    _dump(work, f"recon_{flag}.json", out)


def _grid_regression(b, bl, psi_pad, n_nonuniform=513):
    """Defect A: the same physical profiles on the uniform psi_N grid and on
    a strongly non-uniform one (uniform in sqrt(psi_N)), evaluated on the
    delivered equilibrium with the true grid passed, and read the legacy way
    (the non-uniform arrays taken as if evenly sampled)."""
    import numpy as np
    from bouquet.jbs_loop import residual_weights, weighted_norm
    from bouquet.physics import evaluate_jBS
    from bouquet.utils import pchip_interp
    snap = b.mygs.copy_eq()
    psi = np.asarray(bl.psi_N, dtype=float)
    kin = _kin_on_eq(bl)
    ju, du = evaluate_jBS(snap, psi, *kin, psi_pad=psi_pad, smooth_axis=False)
    rho = np.linspace(0.0, 1.0, n_nonuniform)
    xn = rho ** 2
    kin_n = tuple(pchip_interp(psi, np.asarray(a, float), xn) for a in kin)
    jn, _ = evaluate_jBS(snap, xn, *kin_n, psi_pad=psi_pad, smooth_axis=False)
    jl, _ = evaluate_jBS(snap, np.linspace(0.0, 1.0, xn.size), *kin_n,
                         psi_pad=psi_pad, smooth_axis=False)
    w, x, _k = residual_weights(snap, psi, psi_pad)
    sel = (psi > 0.02) & (psi < 0.99)
    ref = ju[sel]
    jn_u = np.interp(psi, xn, jn)
    jl_u = np.interp(psi, xn, jl)
    nrm = weighted_norm(ref, w[sel], psi[sel])
    e_new = weighted_norm(jn_u[sel] - ref, w[sel], psi[sel]) / nrm
    e_old = weighted_norm(jl_u[sel] - ref, w[sel], psi[sel]) / nrm
    peak = float(np.max(np.abs(ju)))
    mid = (psi >= 0.3) & (psi <= 0.7)
    return dict(psi_uniform=psi.tolist(), j_uniform=ju.tolist(),
                psi_nonuniform=xn.tolist(), j_true_grid=jn.tolist(),
                j_legacy_reading=jl.tolist(), n_uniform=int(psi.size),
                n_nonuniform=int(xn.size), e_true_grid=float(e_new),
                e_legacy=float(e_old), peak=peak,
                max_err_true_grid_frac=float(np.max(np.abs(
                    jn_u[sel] - ref)) / peak),
                max_err_legacy_frac=float(np.max(np.abs(
                    jl_u[sel] - ref)) / peak),
                max_err_true_grid_mid_frac=float(np.max(np.abs(
                    jn_u[mid] - ju[mid])) / peak),
                max_err_legacy_mid_frac=float(np.max(np.abs(
                    jl_u[mid] - ju[mid])) / peak))


def stage_init(work, start):
    """Figure 4: the reconstruction loop from a different initial iterate."""
    t0 = time.time()
    if start == "swb":
        b, cfg = _bouquet_for(work, "init_swb", loop_on=True, init="swb")
        calls = _wrap_loop()
    else:
        fac = float(start.lstrip("x"))
        b, cfg = _bouquet_for(work, f"init_{start}", loop_on=True)
        calls = _wrap_loop(transform_jbs0=lambda j: fac * j)
    bl = b.prepare_baseline()
    if len(calls) != 1:
        raise RuntimeError(f"expected one reconstruction loop, saw {calls}")
    _dump(work, f"init_{start}.json",
          dict(start=start, delivered=_delivered(b, bl, cfg),
               record=(bl.reconstruction_metrics or {}).get("jbs_loop"),
               wall_s=time.time() - t0))


def stage_sigma0(work):
    """Figure 5: the sigma=0 draw loop against the loop-on baseline."""
    t0 = time.time()
    b, cfg = _bouquet_for(work, "sigma0", loop_on=True)
    bl = b.prepare_baseline()
    dl = _delivered(b, bl, cfg)
    s0 = b.verify_sigma0_consistency()
    # `passed` is the draws' zero-perturbation identity (the draw_route
    # block); the loop solved the baseline's way -- what this figure plots --
    # is `passed_baseline_way`
    keep = ("passed", "passed_baseline_way", "draw_route", "invariant",
            "loop_converged", "r_j_vs_baseline",
            "r_I_vs_baseline", "li_sigma0", "li_baseline",
            "li_baseline_reference", "dl_i_vs_baseline", "max_dev",
            "max_dev_frac", "psi_worst", "spike0", "record")
    _dump(work, "sigma0.json",
          dict(delivered=dl, sigma0={k: s0.get(k) for k in keep},
               tolerances=dict(rtol_j=cfg.generation.jbs_rtol_j,
                               rtol_Ip=cfg.generation.jbs_rtol_Ip,
                               tol_li=cfg.generation.jbs_tol_li),
               wall_s=time.time() - t0))


# ---------------------------------------------------------------------------
#  plotting
# ---------------------------------------------------------------------------
def _load(work, name):
    path = os.path.join(work, name)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} missing: run its compute stage first")
    with open(path) as fh:
        return json.load(fh)


def _save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(_HERE, f"{name}.{ext}")
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "text.usetex": False,
                         "mathtext.fontset": "dejavusans",
                         "font.size": 9, "axes.grid": True,
                         "grid.linestyle": ":", "grid.alpha": 0.6,
                         "lines.linewidth": 1.6, "legend.fontsize": 7.5,
                         "legend.framealpha": 0.9})
    return plt


def fig1(work):
    import numpy as np
    plt = _mpl()
    off = _load(work, "recon_off.json")["delivered"]
    on = _load(work, "recon_on.json")["delivered"]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    x0, x1 = np.asarray(off["psi_N"]), np.asarray(on["psi_N"])
    cols = (("j_BS", "$j_{BS}$", WONG["vermillion"]),
            ("j_inductive", "$j_{ind}$", WONG["blue"]),
            ("j_phi", "$j_{tor}$", WONG["black"]))
    for ax in axs:
        for key, lab, c in cols:
            ax.plot(x0, np.asarray(off[key]) / 1e6, color=c, ls="--", lw=1.3)
            ax.plot(x1, np.asarray(on[key]) / 1e6, color=c, ls="-")
        ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel(r"current density [MA m$^{-2}$]")
    axs[1].set_xlim(0.8, 1.0)
    axs[1].set_ylim(0.0, 0.8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, label=lab) for _k, lab, c in cols]
    h += [Line2D([], [], color="0.3", ls="--", lw=1.3,
                 label=(f"frozen SWB (legacy flag): "
                        f"$I_{{BS}}/I_p$ = {off['I_BS'] / off['I_phi']:.4f}")),
          Line2D([], [], color="0.3", ls="-",
                 label=(f"self-consistent Redl: "
                        f"$I_{{BS}}/I_p$ = {on['I_BS'] / on['I_phi']:.4f}"))]
    axs[0].legend(handles=h, loc="upper right")
    axs[1].text(0.02, 0.97, "edge / pedestal", transform=axs[1].transAxes,
                ha="left", va="top", fontsize=8)
    _save(fig, "jbs_loop_fig1_profiles")


def fig2(work):
    import numpy as np
    plt = _mpl()
    d = _load(work, "recon_on.json")
    r = d["record"]
    tol = r["tolerances"]
    pc = r.get("post_corrective") or {}
    pcp = pc.get("passes") or {}
    n_main = int(r["n_passes"])
    n_pc = int(pcp.get("n_passes", 0) or 0)
    k = np.arange(1, n_main + n_pc + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    series = (("r_j", r"$r_j$", "rtol_j", WONG["vermillion"], "o", True),
              ("r_I", r"$r_I$ (fraction of $I_p$)", "rtol_Ip", WONG["blue"],
               "s", True),
              ("dl_i", r"$\Delta l_i$", "tol_li", WONG["green"], "^", True),
              ("dq0", r"$\Delta q_0$ (logged, not a criterion on this path)",
               "tol_q0", WONG["purple"], "D", False))
    chk = pc.get("check") or {}
    for key, lab, tkey, c, m, gated in series:
        vals = list(r[key]) + list(pcp.get(key) or [None] * n_pc)
        v = np.array([np.nan if y is None else float(y) for y in vals])
        ax.semilogy(k[:n_main], v[:n_main], color=c, marker=m, ms=5,
                    label=lab)
        if n_pc:                    # separate segment: a new loop
            ax.semilogy(k[n_main:], v[n_main:], color=c, marker=m, ms=5)
        if chk.get(key) is not None:
            ax.semilogy([n_main + 0.5], [float(chk[key])], color=c,
                        marker=m, ms=7, mfc="white", ls="none")
        ax.axhline(float(tol[tkey]), color=c, lw=1.0,
                   ls=("--" if gated else ":"))
    ok = list(r.get("pass_ok") or []) + list(pcp.get("pass_ok") or [])
    for i, flag in enumerate(ok):
        ax.annotate("ok" if flag else "", (i + 1, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -10),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="0.3")
    if n_pc:
        ax.axvline(n_main + 0.5, color="0.5", lw=1.0)
        ax.text(n_main + 0.55, 0.30, "open symbols: residual check after the\n"
                "corrective iteration; then the post-\ncorrective passes", transform=ax.get_xaxis_transform(),
                fontsize=6.8, color="0.3", va="top")
    ax.set_xlabel("loop pass")
    ax.set_ylabel("residual")
    ax.set_xticks(k)
    note = (rf"$\omega$ = {tol['relax_start']:g} (bootstrap), "
            rf"$\beta$ = {tol['relax_current']:g} (solved current); "
            "omega never halved" + ("" if not r.get("omega_halved_at_pass")
                                    else " -- halved at "
                                    + str(r["omega_halved_at_pass"])) + "\n"
            "dashed: tolerance of a gated criterion; dotted: not gated; "
            f"converged = {tol['required_consecutive']} consecutive 'ok' passes")
    if pc:
        chk = pc.get("check") or {}
        note += (f"\npost-corrective check: r_j = {chk.get('r_j', float('nan')):.2e}, "
                 f"r_I = {chk.get('r_I', float('nan')):.2e} -> "
                 + ("accepted without further passes"
                    if pc.get("accepted_without_passes") else
                    f"{n_pc} further passes"))
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=6.6,
            va="bottom", ha="left",
            bbox=dict(fc="white", ec="0.8", alpha=0.9))
    ax.set_ylim(1e-9, 3e-2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False)
    _save(fig, "jbs_loop_fig2_residuals")


def fig3(work):
    import numpy as np
    plt = _mpl()
    g = _load(work, "recon_on.json")["grid"]
    xu, ju = np.asarray(g["psi_uniform"]), np.asarray(g["j_uniform"]) / 1e6
    xn = np.asarray(g["psi_nonuniform"])
    jn = np.asarray(g["j_true_grid"]) / 1e6
    jl = np.asarray(g["j_legacy_reading"]) / 1e6
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                            constrained_layout=True,
                            gridspec_kw=dict(height_ratios=(3, 2)))
    ax = axs[0]
    ax.plot(xu, ju, color=WONG["black"], lw=2.2,
            label=f"uniform $\\psi_N$ grid ({g['n_uniform']} pts), reference")
    ax.plot(xn, jn, color=WONG["skyblue"], ls="--",
            label=(f"non-uniform grid ({g['n_nonuniform']} pts, uniform in "
                   r"$\sqrt{\psi_N}$), grid passed"))
    ax.plot(xn, jl, color=WONG["vermillion"],
            label="same arrays read as evenly sampled (legacy assumption)")
    ax.set_ylabel(r"Redl $j_{BS}$ [MA m$^{-2}$]")
    ax.legend(loc="upper left")
    ax = axs[1]
    peak = g["peak"] / 1e6
    sel = (xu > 0.02) & (xu < 0.99)
    ax.plot(xu[sel], 100 * (np.interp(xu, xn, jn) - ju)[sel] / peak,
            color=WONG["skyblue"], ls="--",
            label=(f"grid passed: weighted error {100 * g['e_true_grid']:.2f} %"))
    ax.plot(xu[sel], 100 * (np.interp(xu, xn, jl) - ju)[sel] / peak,
            color=WONG["vermillion"],
            label=(f"legacy reading: weighted error {100 * g['e_legacy']:.1f} %"))
    ax.axvspan(0.3, 0.7, color="0.9", zorder=0)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel("error [% of peak $j_{BS}$]")
    ax.legend(loc="upper left")
    ax.text(0.5, 0.97, "mid-radius", transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color="0.35")
    _save(fig, "jbs_loop_fig3_grid")


_INIT_STYLE = (("anchor", "anchor evaluation (default)", WONG["black"], "-", "o"),
               ("swb", "legacy SWB profile", WONG["orange"], "--", "s"),
               ("x0.8", r"anchor $\times$ 0.8", WONG["blue"], "-.", "v"),
               ("x1.2", r"anchor $\times$ 1.2", WONG["vermillion"], ":", "^"))


def _init_runs(work):
    runs = {}
    on = _load(work, "recon_on.json")
    runs["anchor"] = dict(delivered=on["delivered"], record=on["record"])
    for tag in ("swb", "x0.8", "x1.2"):
        runs[tag] = _load(work, f"init_{tag}.json")
    return runs


def init_spread(work):
    """Pairwise distances of the delivered bootstraps (current-weighted, on
    the anchor run's weights) and the l_i spread."""
    import numpy as np
    from itertools import combinations
    sys.path.insert(0, _REPO)
    from bouquet.jbs_loop import weighted_norm
    runs = _init_runs(work)
    ref = runs["anchor"]["delivered"]
    x = np.asarray(ref["psi_N"])
    w = np.asarray(ref["w"])
    rj = {}
    for a, b in combinations(runs, 2):
        ja = np.asarray(runs[a]["delivered"]["j_BS"])
        jb = np.asarray(runs[b]["delivered"]["j_BS"])
        rj[f"{a}|{b}"] = weighted_norm(ja - jb, w, x) / weighted_norm(ja, w, x)
    li = {t: r["delivered"]["l_i_target"] for t, r in runs.items()}
    fbs = {t: r["delivered"]["I_BS"] / r["delivered"]["I_phi"]
           for t, r in runs.items()}
    npass = {t: (r["record"]["n_passes"],
                 ((r["record"].get("post_corrective") or {}).get("passes")
                  or {}).get("n_passes", 0)) for t, r in runs.items()}
    return dict(max_pairwise_rj=max(rj.values()), pairwise_rj=rj,
                l_i_target=li, l_i_spread=max(li.values()) - min(li.values()),
                I_BS_over_Ip=fbs, n_passes_main_post=npass)


def fig4(work):
    import numpy as np
    runs = _init_runs(work)
    sp = init_spread(work)            # imports bouquet: style set after it
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for tag, lab, c, ls, m in _INIT_STYLE:
        r = runs[tag]
        dl = r["delivered"]
        axs[0].plot(dl["psi_N"], np.asarray(dl["j_BS"]) / 1e6, color=c, ls=ls)
        rec = r["record"]
        hist = list(rec["r_j"])
        n_main = len(hist)
        pc = ((rec.get("post_corrective") or {}).get("passes") or {})
        hist += list(pc.get("r_j") or [])
        k = np.arange(1, len(hist) + 1)
        axs[1].semilogy(k[:n_main], hist[:n_main], color=c, ls=ls, marker=m,
                        ms=4, label=(f"{lab}: {n_main}"
                                     + (f" + {len(hist) - n_main}"
                                        if len(hist) > n_main else "")
                                     + " passes, $I_{BS}/I_p$ = "
                                     + f"{dl['I_BS'] / dl['I_phi']:.4f}"))
        if len(hist) > n_main:
            axs[1].semilogy(k[n_main:], hist[n_main:], color=c, ls=ls,
                            marker=m, ms=4, mfc="white")
    tol = runs["anchor"]["record"]["tolerances"]["rtol_j"]
    axs[1].axhline(tol, color="0.4", ls="--", lw=1.0)
    axs[1].text(0.6, tol, r" $r_j$ tolerance", va="bottom", fontsize=7,
                color="0.3")
    axs[0].set_xlabel(r"$\psi_N$")
    axs[0].set_ylabel(r"delivered $j_{BS}$ [MA m$^{-2}$]")
    ins = axs[0].inset_axes([0.36, 0.45, 0.42, 0.45])
    for tag, lab, c, ls, m in _INIT_STYLE:
        dl = runs[tag]["delivered"]
        ins.plot(dl["psi_N"], np.asarray(dl["j_BS"]) / 1e6, color=c, ls=ls)
    ins.set_xlim(0.9, 1.0)
    ins.tick_params(labelsize=6)
    ins.set_title(r"pedestal, $0.9 \leq \psi_N \leq 1$", fontsize=7)
    axs[0].text(0.02, 0.97, (f"max pairwise $r_j$ between\nthe delivered "
                             f"bootstraps:\n{sp['max_pairwise_rj']:.1e}\n"
                             f"$l_i$ spread: {sp['l_i_spread']:.1e}"),
                transform=axs[0].transAxes, fontsize=7, va="top")
    axs[1].set_xlabel("pass (filled: main loop; open: post-corrective loop)")
    axs[1].set_ylabel(r"$r_j$")
    axs[1].legend(loc="upper right", fontsize=7)
    _save(fig, "jbs_loop_fig4_init")


def fig5(work):
    import numpy as np
    plt = _mpl()
    d = _load(work, "sigma0.json")
    s = d["sigma0"]
    dl = d["delivered"]
    tol = d["tolerances"]
    x = np.asarray(dl["psi_N"])
    jb = np.asarray(dl["j_BS"]) / 1e6
    j0 = np.asarray(s["spike0"]) / 1e6
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                            constrained_layout=True,
                            gridspec_kw=dict(height_ratios=(3, 2)))
    axs[0].plot(x, jb, color=WONG["black"], lw=2.2,
                label="baseline (self-consistent reconstruction)")
    axs[0].plot(x, j0, color=WONG["orange"], ls="--",
                label=(f"$\\sigma=0$ draw loop "
                       f"({s['record']['n_passes']} passes, "
                       f"{'converged' if s['loop_converged'] else 'NOT converged'})"))
    axs[0].set_ylabel(r"$j_{BS}$ [MA m$^{-2}$]")
    axs[0].legend(loc="upper left")
    txt = (f"$r_j$ vs baseline = {s['r_j_vs_baseline']:.2e}  (tol {tol['rtol_j']:.0e})\n"
           f"$r_I$ vs baseline = {s['r_I_vs_baseline']:.2e}  (tol {tol['rtol_Ip']:.0e})\n"
           f"$l_i$: $\\sigma=0$ {s['li_sigma0']:.5f} vs delivered baseline "
           f"{s['li_baseline']:.5f}, $|\\Delta l_i|$ = "
           f"{s['dl_i_vs_baseline']:.1e} (tol {tol['tol_li']:.0e})\n"
           f"baseline's way: "
           f"{'PASS' if s.get('passed_baseline_way', s['passed']) else 'FAIL'}"
           f"; draw-route identity: {'PASS' if s['passed'] else 'FAIL'} "
           "(verify_sigma0_consistency)")
    axs[0].text(0.02, 0.45, txt, transform=axs[0].transAxes, fontsize=7,
                va="top", bbox=dict(fc="white", ec="0.8", alpha=0.9))
    peak = float(np.max(np.abs(jb)))
    axs[1].plot(x, 100 * (j0 - jb) / peak, color=WONG["orange"])
    axs[1].axhline(0.0, color="0.5", lw=0.8)
    axs[1].set_xlabel(r"$\psi_N$")
    axs[1].set_ylabel(r"$\sigma=0$ $-$ baseline [% of peak]")
    _save(fig, "jbs_loop_fig5_sigma0")


def summarize(work):
    out = {}
    try:
        off = _load(work, "recon_off.json")["delivered"]
        on = _load(work, "recon_on.json")
        out["fig1"] = dict(I_BS_over_Ip_frozen=off["I_BS"] / off["I_phi"],
                           I_BS_over_Ip_loop=on["delivered"]["I_BS"] / on["delivered"]["I_phi"],
                           I_BS_over_Ip_target_frozen=off["I_BS"] / off["Ip"],
                           I_BS_over_Ip_target_loop=on["delivered"]["I_BS"] / on["delivered"]["Ip"],
                           l_i_target_frozen=off["l_i_target"],
                           l_i_target_loop=on["delivered"]["l_i_target"],
                           q0_frozen=off["q0"], q0_loop=on["delivered"]["q0"])
        r = on["record"]
        out["fig2"] = {k: r[k] for k in ("n_passes", "converged", "r_j", "r_I",
                                         "dl_i", "dq0", "omega", "pass_ok")}
        out["fig2"]["post_corrective"] = r.get("post_corrective")
        out["fig3"] = {k: v for k, v in on["grid"].items()
                       if not isinstance(v, list)}
    except SystemExit as e:
        out["fig1-3"] = str(e)
    try:
        out["fig4"] = init_spread(work)
    except SystemExit as e:
        out["fig4"] = str(e)
    try:
        s = _load(work, "sigma0.json")["sigma0"]
        out["fig5"] = {k: v for k, v in s.items()
                       if k not in ("spike0", "record")}
        out["fig5"]["n_passes"] = s["record"]["n_passes"]
    except SystemExit as e:
        out["fig5"] = str(e)
    print(json.dumps(out, indent=1, default=str))


_FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=_DEFAULT_WORK,
                    help="run directories + JSON cache (default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("recon", help="reconstruction, loop off or on (figs 1-3)")
    p.add_argument("--flag", choices=("off", "on"), required=True)
    p = sub.add_parser("init", help="loop from another initial iterate (fig 4)")
    p.add_argument("--start", choices=("swb", "x0.8", "x1.2"), required=True)
    sub.add_parser("sigma0", help="sigma=0 draw loop vs baseline (fig 5)")
    p = sub.add_parser("plot", help="draw figures from the cache")
    p.add_argument("--fig", type=int, choices=sorted(_FIGS), action="append")
    sub.add_parser("summary", help="print the numbers the figures show")
    sub.add_parser("all", help="every compute stage (one process each), "
                               "then every figure")
    args = ap.parse_args(argv)
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    if args.cmd == "recon":
        stage_recon(work, args.flag)
    elif args.cmd == "init":
        stage_init(work, args.start)
    elif args.cmd == "sigma0":
        stage_sigma0(work)
    elif args.cmd == "plot":
        for n in (args.fig or sorted(_FIGS)):
            _FIGS[n](work)
    elif args.cmd == "summary":
        summarize(work)
    elif args.cmd == "all":
        me = os.path.abspath(__file__)
        for stage in (["recon", "--flag", "off"], ["recon", "--flag", "on"],
                      ["init", "--start", "swb"], ["init", "--start", "x0.8"],
                      ["init", "--start", "x1.2"], ["sigma0"]):
            subprocess.run([sys.executable, me, "--work", work] + stage,
                           check=True)
        for n in sorted(_FIGS):
            _FIGS[n](work)
        summarize(work)


if __name__ == "__main__":
    main()
