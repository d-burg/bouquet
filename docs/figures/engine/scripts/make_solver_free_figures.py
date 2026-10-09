#!/usr/bin/env python3
"""Solver-free review figures for the unified-engine PR stack.

Every figure is computed by calling the branch's OWN functions (the
repository's own code) on synthetic input only: the D3D-like
example shipped in examples/D3D-like (g-file + p-file + OMAS json), the test
suite's toy Grad-Shafranov stand-in (tests/_engine_toy.py), or the closed-form
two-state model of tests/test_jbs_loop.py.  No OpenFUSIONToolkit, no solver,
no experimental data.

usage: PYTHONPATH=<src>:<src>/tests python make_solver_free_figures.py <src> <outdir>
"""
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
EX = os.path.join(SRC, "examples", "D3D-like")
GEQ = os.path.join(EX, "D3Dlike_Hmode_baseline.geqdsk")
PF = os.path.join(EX, "D3Dlike_Hmode_baseline.peqdsk")
OMAS = os.path.join(EX, "D3Dlike_baseline_omas.json")
MESH = os.path.join(EX, "DIIID_mesh.h5")
TIME = 2.3043

W = dict(blue="#0072B2", verm="#D55E00", green="#009E73", pink="#CC79A7",
         orange="#E69F00", sky="#56B4E9", black="#000000")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":",
                     "lines.linewidth": 2, "savefig.dpi": 130,
                     "legend.fontsize": 8.5})
CAPTIONS = {}


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def save(fig, name, caption):
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    CAPTIONS[name] = caption
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. toy engine: pass histories (rows Ip+l_i, +q0, soft rows) and the MSE
#    stage under the two Jacobian schemes
# ---------------------------------------------------------------------------
def fig_toy_engine():
    os.chdir(os.path.join(SRC, "tests"))
    import _engine_toy as T
    from bouquet.engine import reconstruct
    from test_engine import _mse_data, LI0, Q0

    def run(ad, backend=None, **gc):
        b = backend if backend is not None else T.ToyGS()
        ad.read()
        eng, res, rec = quiet(reconstruct, ad, b, T.settings(**gc), label="toy")
        return res, rec

    st = T.settings()["loop"]
    cases = [
        ("rows Ip + l_i (hard)", T.ToyAdapter(li_target=LI0 * 1.01), {}, W["blue"]),
        ("rows Ip + l_i + q0 (hard)",
         T.ToyAdapter(li_target=LI0 * 1.01, q0_target=Q0 * 0.97),
         dict(engine_rows=["Ip", "l_i", "q0"]), W["verm"]),
        ("rows Ip + l_i (soft)", T.ToyAdapter(soft=True, li_target=LI0 * 1.01),
         {}, W["green"]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    summary = []
    for lab, ad, gc, col in cases:
        res, rec = run(ad, **gc)
        r = rec["phases"][0]["record"]
        n = np.arange(1, r["n_passes"] + 1)
        ax[0].semilogy(n, r["r_j"], "o-", color=col, label=lab, ms=4)
        ax[1].semilogy(n, np.abs(r["dl_i"]), "o-", color=col, label=lab, ms=4)
        summary.append((lab, r["n_passes"], rec["solves"]["total"],
                        rec["delivered"]["ok"]))
    ax[0].axhline(st["rtol_j"], color=W["black"], ls="--", lw=1.2,
                  label=r"$r_j$ tolerance")
    ax[1].axhline(st["tol_li"], color=W["black"], ls="--", lw=1.2,
                  label=r"$|\Delta l_i|$ tolerance")
    ax[0].set_xlabel("pass"); ax[0].set_ylabel(r"bootstrap residual $r_j$ [-]")
    ax[1].set_xlabel("pass"); ax[1].set_ylabel(r"$|\Delta l_i|$ per pass [-]")
    ax[0].legend(); ax[1].legend()
    md, ch = _mse_data()
    msum = []
    for scheme, col in (("fd_broyden", W["pink"]), ("fd_chord", W["blue"])):
        ad = T.ToyAdapter(soft=True, li_target=LI0 * 1.01,
                          mse=dict(chords=ch, er_terms="toy"))
        res, rec = run(ad, T.ToyGS(chords=ch), engine_rows=["Ip", "l_i", "mse"],
                       mse_data=md, engine_mse_jacobian=scheme)
        r = rec["phases"][1]["record"]
        n = np.arange(1, r["n_passes"] + 1)
        ax[2].semilogy(n, r["r_j"], "o-", color=col, ms=4,
                       label=f'{scheme}: {r["n_passes"]} MSE passes, '
                             f'{rec["solves"]["total"]} solves')
        msum.append((scheme, r["n_passes"], rec["solves"]["total"],
                     rec["delivered"]["checks"]["mse"]["ok"]))
    ax[2].axhline(st["rtol_j"], color=W["black"], ls="--", lw=1.2,
                  label=r"$r_j$ tolerance")
    ax[2].set_xlabel("MSE-stage pass"); ax[2].set_ylabel(r"bootstrap residual $r_j$ [-]")
    ax[2].legend()
    for a, t in zip(ax, "abc"):
        a.text(0.02, 0.04, f"({t})", transform=a.transAxes)
    fig.tight_layout()
    save(fig, "engine_toy_pass_histories",
         "Toy model. Per-pass residuals of the unified engine "
         "(bouquet.engine.reconstruct) driven by the test suite's "
         "toy Grad-Shafranov stand-in (tests/_engine_toy.py; no solver, no "
         "device data). (a) bootstrap residual r_j and (b) |Delta l_i| per "
         "pass for three row sets; (c) the MSE stage on the toy's synthetic "
         "pitch angles under the two Jacobian schemes (Broyden-updated vs "
         "the fixed finite-difference default 'fd_chord'). Dashed: the "
         "loop's shipped tolerances (unchanged). Summary: "
         + "; ".join(f"{a}: {b} passes, {c} solves, delivered ok={d}"
                     for a, b, c, d in summary)
         + "; " + "; ".join(f"{a}: {b} MSE passes, {c} solves, MSE check ok={d}"
                            for a, b, c, d in msum) + ".")


# ---------------------------------------------------------------------------
# 2. loop kernel: the closed-form two-state model, with and without the
#    current relaxation beta
# ---------------------------------------------------------------------------
def fig_kernel_two_state():
    os.chdir(os.path.join(SRC, "tests"))
    from test_jbs_loop import _two_state_problem, _GC, _IP
    from bouquet.jbs_loop import jbs_settings, run_jbs_loop
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    txt = []
    for beta, col in ((1.0, W["verm"]), (0.7, W["blue"])):
        step, ev, st, Lstar, Jstar = _two_state_problem()
        s = jbs_settings(_GC(jbs_relax_current=beta, jbs_max_passes=40))
        Ls = []

        def step_rec(jbs, k, relax=None, _s=step):
            out = _s(jbs, k, relax=relax)
            Ls.append(out["li"])
            return out
        o = quiet(run_jbs_loop, Jstar * 0.6, step_rec, ev, s, Ip=_IP,
                  meas0=dict(li=st["L"]), gate_li=True)
        r = o["record"]
        n = np.arange(1, r["n_passes"] + 1)
        ax[0].semilogy(n, r["r_j"], "o-", color=col, ms=4,
                       label=rf"$\beta$ = {beta}: {r['n_passes']} passes")
        ax[1].plot(np.arange(1, len(Ls) + 1), np.array(Ls) - Lstar, "o-",
                   color=col, ms=4, label=rf"$\beta$ = {beta}")
        txt.append(f"beta={beta}: {r['n_passes']} passes, converged="
                   f"{o['converged']}, |L-L*|={abs(st['L'] - Lstar):.1e}")
    ax[0].axhline(s["rtol_j"], color=W["black"], ls="--", lw=1.2, label=r"$r_j$ tolerance")
    ax[1].axhspan(-s["tol_li"], s["tol_li"], color=W["sky"], alpha=0.25,
                  label=r"$\pm$ l_i tolerance")
    ax[0].set_xlabel("pass"); ax[0].set_ylabel(r"bootstrap residual $r_j$ [-]")
    ax[1].set_xlabel("solve"); ax[1].set_ylabel(r"$L - L^*$ (geometry scalar) [-]")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout()
    save(fig, "loop_kernel_two_state_relaxation",
         "Toy model. The self-consistent bootstrap kernel "
         "(bouquet.jbs_loop.run_jbs_loop) on the closed-form two-state "
         "closure<->geometry problem of tests/test_jbs_loop.py (g = -0.55, "
         "eps = 0.2), started at 0.6 x the fixed point, without (beta = 1) "
         "and with (beta = 0.7, the default) the current relaxation. Both "
         "reach the closed-form fixed point L* to the loop's own tolerances; "
         "beta changes the path, not the answer. " + "; ".join(txt) + ".")


# ---------------------------------------------------------------------------
# 3. g-file adapter: identity I2 on the synthetic g-file's own surfaces
# ---------------------------------------------------------------------------
def gfile_contract():
    from bouquet.adapters import GFileAdapter
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    cfg = BouquetConfig(
        source=ReconstructionSource(geqdsk_path=GEQ, profiles_path=PF),
        solver=SolverConfig(mesh_path=MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    ad = GFileAdapter(cfg.source, cfg)
    c = quiet(ad.read)
    return ad, c


def fig_gfile_identity():
    from bouquet.adapters import gfile_parallel_current
    from bouquet.engine import compose
    ad, c = gfile_contract()
    jB, parts = gfile_parallel_current(ad.eqdsk)
    geom = dict(F=parts["F"], R_avg=parts["R_avg"], inv_R=parts["inv_R"],
                B2=parts["B2"], pprime=parts["pprime"])
    J, p = compose(geom, 0.6 * jB, 0.3 * jB, 0.1 * jB)
    x = np.asarray(c.psi_N)
    jin = parts["jphi_in"]
    sc = float(np.max(np.abs(jin)))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(x, jin / 1e6, color=W["black"], label=r"g-file $\langle j_\phi\rangle$")
    ax[0].plot(x, J / 1e6, "--", color=W["orange"], label=r"composed: $\kappa\langle j\cdot B\rangle$ + pressure term")
    ax[0].plot(x, p["kappa"] * jB / 1e6, color=W["blue"], lw=1.5,
               label=r"$\kappa\langle j\cdot B\rangle$ (field-aligned part)")
    ax[0].plot(x, p["pressure"] / 1e6, color=W["verm"], lw=1.5,
               label=r"$p'(\langle R\rangle - F^2\langle 1/R\rangle/\langle B^2\rangle)$")
    ax[0].set_xlabel(r"$\psi_N$"); ax[0].set_ylabel(r"current density [MA m$^{-2}$]")
    ax[0].legend(loc="upper center")
    ax[1].plot(x, (J - jin) / sc, color=W["blue"])
    ax[1].set_xlabel(r"$\psi_N$"); ax[1].set_ylabel(r"(composed $-$ g-file) / max [-]")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    frac = p["pressure"] / jin
    m = float(np.median(frac[x > 0.9]))
    save(fig, "engine_gfile_identity_I2",
         "Synthetic example shipped with the repository "
         "(examples/D3D-like g-file + p-file), computed by calling the "
         "branch's own adapters.gfile_parallel_current and engine.compose "
         "(no solver). The engine stores currents as parallel <j.B> and "
         "forms the solver's <j_phi> as kappa<j.B> + p'(<R> - F^2<1/R>/<B^2>) "
         "(identity I2). Split here as 0.6/0.3/0.1 of the parallel current "
         "across inductive/bootstrap/fixed: the composition reproduces the "
         f"g-file's own <j_phi> to max {float(np.max(np.abs(J - jin))) / sc:.1e} "
         f"of the peak; the pressure-driven term is a median {100 * m:.0f} % of "
         "<j_phi> at psi_N > 0.9.")


# ---------------------------------------------------------------------------
# 4. IDS adapter: the split of the synthetic OMAS source, residual vs j_ohmic
# ---------------------------------------------------------------------------
def ids_contract(path, **adkw):
    from bouquet.adapters import IdsAdapter
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=str(path), time=TIME),
        solver=SolverConfig(mesh_path=MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    bl = quiet(resolve_baseline, cfg, None)
    ad = IdsAdapter(cfg.source, cfg, bl, **adkw)
    return ad, quiet(ad.read), bl


def fig_ids_split():
    ad, c, bl = ids_contract(OMAS)
    _a, c_ohm, _b = ids_contract(OMAS, inductive="j_ohmic")
    with open(OMAS) as fh:
        dd = json.load(fh)
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    x = np.asarray(c.psi_N)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(x, c.jB_ind / B0 / 1e6, color=W["blue"],
               label=r"inductive = $j_{tot} - j_{BS} - \Sigma j_{driven}$ (default 'residual')")
    ax[0].plot(x, c_ohm.jB_ind / B0 / 1e6, "--", color=W["orange"],
               label=r"source $j_{ohmic}$ ('j_ohmic', explicit)")
    ax[0].plot(x, c.jB_fix / B0 / 1e6, color=W["green"],
               label=r"driven, held fixed ($\Sigma$ core_sources)")
    ax[0].set_xlabel(r"$\psi_N$"); ax[0].set_ylabel(r"$\langle j\cdot B\rangle/B_0$ [MA m$^{-2}$]")
    ax[0].legend(fontsize=7.5)
    sc = float(np.max(np.abs(c.jB_ind)))
    ax[1].plot(x, (c_ohm.jB_ind - c.jB_ind) / sc, color=W["verm"])
    ax[1].set_xlabel(r"$\psi_N$"); ax[1].set_ylabel(r"($j_{ohmic}$ $-$ residual) / max [-]")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    ic = c.provenance.get("inductive_consistency", {})
    keep = {k: ic[k] for k in ic if k in ("action", "net_fraction", "rms_fraction",
                                          "net", "rms", "tol")}
    save(fig, "engine_ids_inductive_split",
         "Synthetic example shipped with the repository (examples/D3D-like "
         "OMAS json, t = 2.3043 s of the synthetic source), computed by calling "
         "the branch's own baseline reader and adapters.IdsAdapter (no "
         "solver). Under the default engine_ids_inductive='residual' the "
         "inductive current is the parallel residual j_total - j_bootstrap - "
         "sum(driven); the source's j_ohmic is a stamped cross-check. On this "
         "self-consistent synthetic source the two agree to the plotted "
         f"difference; provenance stamp: {json.dumps(keep, default=str)}.")


# ---------------------------------------------------------------------------
# 5. reversed Ip / reversed B0: four orientations of the synthetic OMAS read
#    to one positive-frame contract
# ---------------------------------------------------------------------------
def fig_orientations():
    import tempfile
    sys.path.insert(0, os.path.join(SRC, "tests"))
    import _mirror_dd as mdd
    with open(OMAS) as fh:
        dd0 = json.load(fh)
    cols = [W["blue"], W["orange"], W["verm"], W["green"]]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ref = None
    diffs = []
    with tempfile.TemporaryDirectory() as td:
        for (s_ip, s_b0), col in zip(mdd.ORIENTATIONS, cols):
            dd = mdd.mirror_dd(dd0, s_ip, s_b0)
            p = os.path.join(td, f"dd_{mdd.tag(s_ip, s_b0)}.json")
            with open(p, "w") as fh:
                json.dump(dd, fh)
            _a, c, bl = ids_contract(p)
            cp = dd["core_profiles"]["profiles_1d"][2]
            rho = np.asarray(cp["grid"]["rho_tor_norm"])
            lab = f"Ip sign {s_ip:+.0f}, B0 sign {-s_b0:+.0f}"
            ax[0].plot(rho, np.asarray(cp["j_total"]) / 1e6, color=col, label=lab,
                       ls="-" if s_ip > 0 else "--")
            if ref is None:
                ref = c
            diffs.append(float(np.max(np.abs(c.jB_ind - ref.jB_ind))))
            ax[1].plot(c.psi_N, c.jB_ind / 1e6, color=col,
                       ls=["-", "--", ":", "-."][len(diffs) - 1], label=lab)
    ax[0].set_xlabel(r"$\rho_{tor,N}$"); ax[0].set_ylabel(r"source $j_{total}$ as stored [MA m$^{-2}$]")
    ax[1].set_xlabel(r"$\psi_N$"); ax[1].set_ylabel(r"read inductive $\langle j\cdot B\rangle$ [MA T m$^{-2}$]")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout()
    save(fig, "reversed_ip_four_orientations",
         "Synthetic example shipped with the repository (examples/D3D-like "
         "OMAS json) mirrored into all four (Ip, B0) orientations with the "
         "test suite's own helper (tests/_mirror_dd.py), then read through "
         "the branch's reader and IDS adapter (no solver). Left: the source "
         "current as stored (sign follows Ip). Right: the contract's inductive "
         "current in bouquet's positive-Ip frame -- the four curves coincide; "
         "max |difference| from the reference orientation = "
         + ", ".join(f"{d:.1e}" for d in diffs) + " (bit-identical when 0).")


# ---------------------------------------------------------------------------
# 6. edge pressure: what the solver is handed and what is reported, legacy
#    vs offset, on the synthetic g-file example; and a toy p_sep sweep
# ---------------------------------------------------------------------------
def fig_edge_pressure():
    from bouquet.edge_pressure import (EdgePressure, solver_pressure,
                                       separatrix_pressure_of)
    ad, c = gfile_contract()
    x = np.asarray(c.psi_N)
    p = np.asarray(c.pressure, dtype=float)
    vol = np.asarray(ad.eqdsk.geometry["vol"], dtype=float)
    leg = EdgePressure(edge_pprime_pin=True, separatrix_pressure="legacy")
    off = EdgePressure(edge_pprime_pin=True, separatrix_pressure="offset")
    psep = separatrix_pressure_of(p)

    def solver_shape(pr, edge):
        # the solver integrates P' from zero at the boundary and rescales to
        # the axis target: its pressure is pax * (p - p_sep)/(p0 - p_sep)
        ps = np.asarray(pr, float)
        pax = float(solver_pressure(ps, edge)[0])
        return pax * (ps - ps[-1]) / (ps[0] - ps[-1])

    def W(pr):
        return 1.5 * float(np.trapezoid(pr, vol))

    s_leg, s_off = solver_shape(p, leg), solver_shape(p, off)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(x, p / 1e3, color=W_["black"], label="input total pressure")
    ax[0].plot(x, s_leg / 1e3, "--", color=W_["verm"],
               label=r"solver, 'legacy' (axis target $p_{axis}$)")
    ax[0].plot(x, s_off / 1e3, color=W_["blue"], lw=1.5,
               label=r"solver, 'offset' (target $p_{axis}-p_{sep}$)")
    ax[0].plot(x, (s_off + psep) / 1e3, ":", color=W_["green"],
               label=r"'offset', reported ($+p_{sep}$)")
    ax[0].set_xlim(0.8, 1.0)
    ax[0].set_ylim(0, float(np.interp(0.8, x, p)) / 1e3 * 1.15)
    ax[0].set_xlabel(r"$\psi_N$"); ax[0].set_ylabel("pressure [kPa]")
    ax[0].legend(fontsize=7.5)
    # toy sweep: raise p_sep on the same shape, W of each frame / input W
    fr = np.linspace(0.0, 0.10, 41)
    rl, ro = [], []
    core = p - p[-1]
    for f in fr:
        pf = core / core[0] * p[0] * (1 - f) + f * p[0]
        wi = W(pf)
        rl.append(W(solver_shape(pf, leg)) / wi - 1)
        ro.append((W(solver_shape(pf, off)) + 1.5 * pf[-1] * vol[-1]) / wi - 1)
    ax[1].plot(100 * fr, 100 * np.array(rl), color=W_["verm"],
               label="'legacy': solver-frame W / input W - 1")
    ax[1].plot(100 * fr, 100 * np.array(ro), color=W_["blue"],
               label="'offset': reported W / input W - 1")
    ax[1].axvline(100 * psep / p[0], color=W_["black"], ls=":", lw=1.2,
                  label="the synthetic example's own $p_{sep}/p_{axis}$")
    ax[1].set_xlabel(r"$p_{sep}/p_{axis}$ [%]"); ax[1].set_ylabel(r"stored-energy error [%]")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "edge_pressure_separatrix_offset",
         "Synthetic example shipped with the repository (examples/D3D-like "
         "g-file + p-file; total pressure as handed to the solver, from the "
         "branch's GFileAdapter) and a toy sweep, using the branch's own "
         "bouquet.edge_pressure helpers; no solver. Left: the pressure the "
         "solver ends up with under separatrix_pressure='legacy' (P' inflated "
         "by p_axis/(p_axis - p_sep)) and 'offset' (the input's own P'; p_sep "
         "added back when reported). Right (toy: the example's shape with "
         "p_sep raised from 0 to 10 % of p_axis, volume from the g-file's "
         "traced surfaces): error of the stored energy in each frame vs the "
         f"input. The example's own p_sep/p_axis = {100 * psep / p[0]:.2f} %. "
         "Bookkeeping only -- the equilibrium response (beta_N, W from a solve) "
         "needs the solver figure listed in the plan.")


W_ = W


def main():
    import shutil
    for fn in (fig_toy_engine, fig_kernel_two_state, fig_gfile_identity,
               fig_ids_split, fig_orientations, fig_edge_pressure):
        try:
            fn()
        except Exception as e:  # report and continue; never fake a figure
            import traceback
            traceback.print_exc()
            CAPTIONS[fn.__name__] = f"NOT GENERATED: {type(e).__name__}: {e}"
    with open(os.path.join(OUT, "captions.json"), "w") as fh:
        json.dump(CAPTIONS, fh, indent=1)


if __name__ == "__main__":
    main()
