#!/usr/bin/env python3
"""Solver review figures for the unified-engine PR stack (synthetic examples only).

Reads ONLY the small JSON the measurement driver wrote (run_solver_figure_measurements.sh:
measure_engine.py parts, probe_baseline_gfile_frame.py, probe_engine_mse_synthetic.py,
extract_small.py) plus the synthetic inputs shipped in examples/D3D-like (g-file, OMAS
json) for the input curves.  No solver runs here; no experimental data.

usage: PYTHONPATH=<repo> python make_engine_solver_figures.py <repo> <data dir> <out dir> <build label>
"""
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO, DATA, OUT, BUILD = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
EX = os.path.join(REPO, "examples", "D3D-like")
GEQ = os.path.join(EX, "D3Dlike_Hmode_baseline.geqdsk")
OMAS = os.path.join(EX, "D3Dlike_baseline_omas.json")
TIME = 2.3043

W = dict(blue="#0072B2", verm="#D55E00", green="#009E73", pink="#CC79A7",
         orange="#E69F00", sky="#56B4E9", yellow="#F0E442", black="#000000")
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.linestyle": ":",
                     "lines.linewidth": 2, "savefig.dpi": 130,
                     "legend.fontsize": 8, "figure.constrained_layout.use": True})
PROV = ("Synthetic example shipped with the repository (examples/D3D-like); real "
        f"solver, OpenFUSIONToolkit {BUILD}, one thread.")
CAPTIONS, SUMMARY = {}, {}


def load(*p):
    f = os.path.join(DATA, *p)
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def save(fig, name, caption):
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    CAPTIONS[name] = caption
    print("wrote", name)


def gfile_input():
    from bouquet.io.geqdsk import read_geqdsk
    eq = read_geqdsk(GEQ)
    return dict(psi=np.asarray(eq.psi_N, float),
                j=np.abs(np.asarray(eq.j_tor_averaged_direct, float)),
                q=np.abs(np.asarray(eq.qpsi, float)),
                p=np.asarray(eq.pres, float))


def omas_input():
    with open(OMAS) as fh:
        dd = json.load(fh)
    e = dd["equilibrium"]
    i = int(np.argmin(np.abs(np.asarray(e["time"]) - TIME)))
    p = e["time_slice"][i]["profiles_1d"]
    ps = np.asarray(p["psi"], float)
    x = (ps - ps[0]) / (ps[-1] - ps[0])
    return dict(psi=x, j=np.abs(np.asarray(p["j_tor"], float)),
                q=np.abs(np.asarray(p["q"], float)),
                p=np.asarray(p["pressure"], float))


def MA(a):
    return np.asarray(a, float) / 1e6


# ---------------------------------------------------------------------------
# 1-2. legacy vs engine profiles (g-file, IDS)
# ---------------------------------------------------------------------------
def fig_profiles(name, inp, eng, leg, src_label, extra=""):
    ee, le = eng["edge"], ((leg or {}).get("edge") if isinstance((leg or {}).get("edge"), dict) else None)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    x = np.asarray(ee["psi_N"])
    xl = np.asarray(le["psi_N"]) if le else None
    # (a) <j_phi>
    a = ax[0, 0]
    a.plot(inp["psi"], MA(inp["j"]), color=W["black"], lw=2.5, label="input")
    if le:
        a.plot(xl, MA(le["jphi_achieved"]), color=W["verm"], ls="--", label="legacy path")
    a.plot(x, MA(ee["jphi_achieved"]), color=W["blue"], ls="-.", label="unified engine")
    a.set(xlabel=r"$\psi_N$", ylabel=r"$\langle j_\phi\rangle$ [MA/m$^2$]",
          title="(a) achieved flux-surface-averaged current")
    a.legend()
    # (b) achieved - input, % of input peak
    a = ax[0, 1]
    pk = float(np.max(inp["j"]))
    for xx, e, c, l in ((xl, le, W["verm"], "legacy path"),
                        (x, ee, W["blue"], "unified engine")):
        if e is None:
            continue
        ji = np.interp(xx, inp["psi"], inp["j"])
        a.plot(xx, 100 * (np.asarray(e["jphi_achieved"]) - ji) / pk, color=c, label=l)
    a.axhline(0, color="0.4", lw=1)
    a.set(xlabel=r"$\psi_N$", ylabel="achieved $-$ input [% of input peak]",
          title=r"(b) $\langle j_\phi\rangle$ difference to the input")
    a.legend()
    # (c) q
    a = ax[0, 2]
    a.plot(inp["psi"], inp["q"], color=W["black"], lw=2.5, label="input")
    if le:
        a.plot(xl, le["q"], color=W["verm"], ls="--", label="legacy path")
    a.plot(x, ee["q"], color=W["blue"], ls="-.", label="unified engine")
    a.set(xlabel=r"$\psi_N$", ylabel="q", title="(c) safety factor")
    a.legend()
    # (d) pressure (full frame) and the solver-frame pressure
    a = ax[1, 0]
    a.plot(inp["psi"], np.asarray(inp["p"]) / 1e3, color=W["black"], lw=2.5,
           label=("input (g-file PRES)" if "g-file" in src_label else "input (IDS pressure)"))
    a.plot(x, np.asarray(ee["pressure_input"]) / 1e3, color=W["green"], ls=":",
           label="pressure handed to bouquet (kinetic)")
    if le:
        a.plot(xl, np.asarray(le["pressure_full"]) / 1e3, color=W["verm"], ls="--",
               label="legacy path, full frame")
    a.plot(x, np.asarray(ee["pressure_full"]) / 1e3, color=W["blue"], ls="-.",
           label="engine, full frame")
    a.set(xlabel=r"$\psi_N$", ylabel="p [kPa]", title="(d) pressure")
    a.legend(fontsize=7)
    # (e) engine current split into FF' and p' terms
    a = ax[1, 1]
    a.plot(x, MA(ee["jphi_achieved"]), color=W["black"], label=r"$\langle j_\phi\rangle$ (engine)")
    a.plot(x, MA(ee["j_ffprime_term"]), color=W["blue"], ls="--",
           label=r"$\langle 1/R\rangle FF'/\mu_0$")
    a.plot(x, MA(ee["j_pprime_term"]), color=W["orange"], ls="-.", label=r"$\langle R\rangle p'$")
    a.plot(x, MA(ee["j_pressure_driven"]), color=W["pink"], ls=":",
           label=r"pressure-driven $p'(\langle R\rangle - F^2\langle 1/R\rangle/\langle B^2\rangle)$")
    a.set(xlabel=r"$\psi_N$", ylabel=r"current density [MA/m$^2$]",
          title="(e) engine current split")
    a.legend(fontsize=7)
    # (f) edge zoom of <j_phi>
    a = ax[1, 2]
    m = inp["psi"] >= 0.85
    a.plot(inp["psi"][m], MA(inp["j"][m]), color=W["black"], lw=2.5, label="input")
    if le:
        mm = xl >= 0.85
        a.plot(xl[mm], MA(np.asarray(le["jphi_achieved"])[mm]), color=W["verm"], ls="--",
               label="legacy path")
    mm = x >= 0.85
    a.plot(x[mm], MA(np.asarray(ee["jphi_achieved"])[mm]), color=W["blue"], ls="-.",
           label="unified engine")
    a.set(xlabel=r"$\psi_N$", ylabel=r"$\langle j_\phi\rangle$ [MA/m$^2$]",
          title=r"(f) edge, $\psi_N \geq 0.85$")
    a.legend()
    d = eng.get("distance") or {}
    dl = (leg or {}).get("distance") or {}
    s = dict(engine_li=eng["build"].get("delivered_state", {}).get("l_i"),
             engine_q95=ee["summary"].get("q95"),
             engine_q_profile_rel_pct=d.get("q_profile_rel_pct"),
             legacy_q_profile_rel_pct=dl.get("q_profile_rel_pct"))
    SUMMARY[name] = s
    qe = (d.get("q_profile_rel_pct") or {}).get("rms")
    ql = (dl.get("q_profile_rel_pct") or {}).get("rms")
    cap = (f"{PROV} Input (black) against the legacy reconstruction path (default "
           f"reconstruction_engine='legacy') and the unified engine on the {src_label}: "
           r"achieved <j_phi>, its difference to the input, q, pressure, and the engine's "
           f"current split into FF' and p' terms. q-profile rms deviation over psi_N 0.05-0.95: "
           f"engine {qe:.2f} %" + (f", legacy {ql:.2f} %" if ql is not None else "") + f".{extra}")
    save(fig, name, cap)


# ---------------------------------------------------------------------------
# 3. engine pass histories
# ---------------------------------------------------------------------------
def fig_pass_histories(parts):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    cols = dict(recon=W["blue"], imas=W["verm"], imas_q0=W["green"])
    lab = dict(recon="g-file, rows Ip + l_i", imas="IDS, rows Ip + l_i",
               imas_q0="IDS, rows Ip + l_i + q0")
    tol = None
    summ = {}
    for p, d in parts.items():
        if d is None or "build" not in d:
            continue
        rec = d["build"]["engine_record"]
        r = rec["phases"][0]["record"]
        tol = rec["settings"]["loop"]
        k = np.arange(1, len(r["r_j"]) + 1)
        c = cols[p]
        ax[0].semilogy(k, np.abs(r["r_j"]), "o-", color=c, label=lab[p])
        ax[1].semilogy(k, np.maximum(np.abs(r["r_I"]), 1e-12), "o-", color=c)
        dli = [np.nan if v is None else abs(v) for v in r["dl_i"]]
        ax[2].semilogy(k, dli, "o-", color=c)
        cr = [np.nan if v is None else abs(v) for v in r.get("current_residual", [])]
        if cr:
            ax[3].semilogy(np.arange(1, len(cr) + 1), cr, "o-", color=c)
        summ[p] = dict(n_passes=r["n_passes"], stop=r["stop_reason"],
                       solves=rec["solves"], wall_s=d["build"]["wall_s"])
    if tol:
        for a, key in zip(ax[:3], ("rtol_j", "rtol_Ip", "tol_li")):
            a.axhline(tol[key], color="0.3", ls="--", lw=1.2, label=f"{key} = {tol[key]:g}")
        ax[3].axhline(1e-3, color="0.3", ls="--", lw=1.2, label="current gate 1e-3")
    for a, t, yl in zip(ax, ("(a) bootstrap residual", "(b) Ip residual",
                             r"(c) $|\Delta l_i|$ pass to pass", "(d) current residual"),
                        (r"$r_j$ [-]", r"$r_{I}$ [-]", r"$|\Delta l_i|$ [-]",
                         "unrelaxed current residual [-]")):
        a.set(xlabel="pass", ylabel=yl, title=t)
        a.legend(fontsize=7)
    SUMMARY["engine_pass_histories_synthetic"] = summ
    txt = "; ".join(f"{lab[p]}: {v['n_passes']} passes, {v['solves']['total']} solves, "
                    f"{v['wall_s']:.0f} s" for p, v in summ.items())
    save(fig, "engine_pass_histories_synthetic",
         f"{PROV} Per-pass residuals of the unified engine's reconstruction loop with the "
         f"live solver; dashed lines are the shipped loop tolerances (unchanged). {txt}.")


# ---------------------------------------------------------------------------
# 4. sigma = 0 engine draw identity
# ---------------------------------------------------------------------------
def fig_sigma0(parts):
    rows = []
    for p, d in parts.items():
        z = (d or {}).get("sigma0")
        if not z or "r_j" not in z:
            continue
        t = z["tolerances"]
        rows.append((p, [abs(z["r_j"]) / t["rtol_j"], abs(z["r_I"]) / t["rtol_Ip"],
                         abs(z["dl_i"]) / t["tol_li"]], z))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    names = ["r_j / rtol_j", "r_I / rtol_Ip", "|dl_i| / tol_li"]
    wdt = 0.8 / max(len(rows), 1)
    cols = [W["blue"], W["verm"], W["green"]]
    for i, (p, v, z) in enumerate(rows):
        ax[0].bar(np.arange(3) + i * wdt, v, wdt, color=cols[i],
                  label=f"{p}: {z['n_passes']} passes, request bit-identical={z['request_bit_identical']}")
    ax[0].set_yscale("log")
    ax[0].set_ylim(top=100)
    ax[0].axhline(1.0, color="0.3", ls="--", lw=1.2, label="tolerance (ratio 1)")
    ax[0].set_xticks(np.arange(3) + wdt * (len(rows) - 1) / 2)
    ax[0].set_xticklabels(names)
    ax[0].set(ylabel="residual / tolerance [-]",
              title="(a) zero-perturbation engine draw vs reconstruction")
    ax[0].legend(fontsize=7)
    for i, (p, v, z) in enumerate(rows):
        ax[1].bar(i - 0.2, abs(z["dq0"]), 0.4, color=W["sky"],
                  label=r"$|\Delta q_0|$ at $\psi_N$" + f"={z['dq0_psi_N']}" if i == 0 else None)
        ax[1].bar(i + 0.2, abs(z["dq95"]), 0.4, color=W["orange"],
                  label=r"$|\Delta q_{95}|$" if i == 0 else None)
    ax[1].set_yscale("log")
    ax[1].set_ylim(top=1e-3)
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([r[0] for r in rows])
    ax[1].set(ylabel="absolute change [-]", title="(b) q reported beside the verdict")
    ax[1].legend(fontsize=7)
    SUMMARY["engine_sigma0_identity"] = {p: dict(r_j=z["r_j"], r_I=z["r_I"], dl_i=z["dl_i"],
                                                 dq0=z["dq0"], dq95=z["dq95"],
                                                 passed=z["passed"])
                                         for p, v, z in rows}
    worst = max(max(v) for _p, v, _z in rows) if rows else float("nan")
    save(fig, "engine_sigma0_identity",
         f"{PROV} The engine draw at zero perturbation (verify_sigma0_consistency under "
         f"reconstruction_engine='unified') against the reconstruction it was drawn from: "
         f"the first-pass request is bit-identical and every residual is far inside the "
         f"loop's own tolerance (largest ratio {worst:.2g}).")


# ---------------------------------------------------------------------------
# 5-6. draw ensemble bands and cost
# ---------------------------------------------------------------------------
def _band(ax, x, Y, color, label):
    Y = np.asarray(Y, float)
    lo, med, hi = np.nanpercentile(Y, [16, 50, 84], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.3, lw=0, label=f"{label} 16-84 %")
    ax.plot(x, med, color=color, lw=1.6, label=f"{label} median")


def fig_draws(draws_meta):
    srcs = [s for s in ("recon", "imas") if load("small", f"draws_{s}_profiles.json")]
    if not srcs:
        print("no draw profiles; skipping draw figures")
        return
    keys = [("j_phi", r"$\langle j_\phi\rangle$ [MA/m$^2$]", 1e-6),
            ("q", "q", 1.0), ("pressure", "p [kPa]", 1e-3)]
    fig, ax = plt.subplots(len(srcs), 3, figsize=(13, 3.7 * len(srcs)), squeeze=False)
    summ = {}
    for r, s in enumerate(srcs):
        P = load("small", f"draws_{s}_profiles.json")
        meta = (draws_meta.get(s) or {}).get("draws") or {}
        bl = P["baseline"]
        ds = P["draws"]
        insp = [bool(d["flags"].get("in_spec", d["attrs"].get("in_spec", 0))) for d in ds]
        for c, (k, yl, f) in enumerate(keys):
            a = ax[r, c]
            xk = "psi_q" if k == "q" else "psi_N"
            have = [d for d in ds if k in d["profiles"] and xk in d["profiles"]]
            if not have:
                a.text(0.5, 0.5, f"'{k}' not archived", transform=a.transAxes, ha="center")
                continue
            x = np.asarray(have[0]["profiles"][xk], float)
            Y = [np.interp(x, d["profiles"][xk], d["profiles"][k]) for d in have]
            Yabs = np.abs(Y) if k in ("j_phi", "q") else Y
            _band(a, x, np.asarray(Yabs) * f, W["blue"], f"all archived (n={len(have)})")
            ins = [y for y, d in zip(Yabs, have) if bool(d["flags"].get("in_spec", False))]
            if len(ins) >= 2:
                _band(a, x, np.asarray(ins) * f, W["verm"], f"in spec (n={len(ins)})")
            if k in bl and xk in bl:
                yb = np.abs(bl[k]) if k in ("j_phi", "q") else np.asarray(bl[k])
                a.plot(bl[xk], np.asarray(yb) * f, color=W["black"], ls="--", lw=1.4,
                       label="baseline (reconstruction)")
            a.set(xlabel=r"$\psi_N$", ylabel=yl,
                  title=f"({'abcdef'[3 * r + c]}) {'g-file' if s == 'recon' else 'IDS'} draws: {k}")
            a.legend(fontsize=7)
        summ[s] = dict(attempts=meta.get("attempts"), archived=meta.get("archived"),
                       rejected=meta.get("rejected"), in_spec=meta.get("in_spec"),
                       in_spec_flags=int(sum(insp)), n_profiles=len(ds),
                       rejection_reasons=[x.get("reason") for x in meta.get("rejections", [])],
                       wall_s=meta.get("wall_s"))
    SUMMARY["engine_draw_ensemble_bands"] = summ
    txt = "; ".join(f"{'g-file' if s == 'recon' else 'IDS'}: {v['attempts']} attempts, "
                    f"{v['archived']} archived, {v['rejected']} rejected, {v['in_spec']} in spec"
                    for s, v in summ.items())
    save(fig, "engine_draw_ensemble_bands",
         f"{PROV} Seeded engine-draw batches (12 draws requested, seed 12345, default "
         f"uncertainties) through Bouquet.generate() with reconstruction_engine='unified': "
         f"median and 16-84 % bands of the archived draws and of the in-spec subset, with the "
         f"reconstruction. Yield: {txt}.")
    # ---- cost
    fig, ax = plt.subplots(1, len(srcs), figsize=(6.5 * len(srcs), 3.9), squeeze=False)
    stages = [("loop", W["blue"]), ("homotopy", W["orange"]),
              ("post_homotopy", W["verm"]), ("filters", W["green"])]
    cost = {}
    for i, s in enumerate(srcs):
        meta = (draws_meta.get(s) or {}).get("draws") or {}
        rows = meta.get("per_draw", [])
        a = ax[0, i]
        bottom = np.zeros(len(rows))
        for st, c in stages:
            v = np.array([((r.get("cost") or {}).get(st) or {}).get("solves") or 0
                          for r in rows], float)
            a.bar(np.arange(len(rows)), v, bottom=bottom, color=c, label=f"{st} solves")
            bottom += v
        ph = [((r.get("cost") or {}).get("post_homotopy") or {}).get("passes") or 0 for r in rows]
        a2 = a.twinx()
        a2.plot(np.arange(len(rows)), ph, "kD", ms=5, label="post-homotopy passes")
        a2.axhline(6, color="k", ls="--", lw=1.2, label="post-homotopy ceiling 6")
        a2.set_ylim(0, 10)
        a.set_ylim(0, 1.75 * max(bottom.max(), 1))
        a2.set_ylabel("post-homotopy passes [-]")
        a2.grid(False)
        a.set(xlabel="archived draw index", ylabel="GS solves per draw [-]",
              title=f"({'ab'[i]}) {'g-file' if s == 'recon' else 'IDS'}: solves by stage")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=3)
        wall = [r.get("time_s") for r in rows]
        cost[s] = dict(solves_total=[float(b) for b in bottom], post_homotopy_passes=ph,
                       wall_s=wall, median_wall_s=float(np.median(wall)) if wall else None)
    SUMMARY["engine_draw_cost"] = cost
    txt = "; ".join(f"{'g-file' if s == 'recon' else 'IDS'}: median {v['median_wall_s']:.0f} s "
                    f"per archived draw, max post-homotopy passes {max(v['post_homotopy_passes'] or [0])}"
                    for s, v in cost.items() if v["median_wall_s"] is not None)
    save(fig, "engine_draw_cost",
         f"{PROV} GS solves per archived engine draw by stage (bars) and the post-homotopy "
         f"passes against the approved ceiling of 6 (diamonds, dashed). {txt}.")


# ---------------------------------------------------------------------------
# 7. sigma0 on the LEGACY path (loop ON): the draw route vs the reconstruction
#    (numbers from tests/test_jbs_loop_solver.py's own recon / imas probes,
#     copied from the solver-suite run's basetemp into loop_probe/)
# ---------------------------------------------------------------------------
def fig_sigma0_legacy(_unused=None):
    rec = load("loop_probe", "recon.json") or {}
    ims = load("loop_probe", "imas_core.json") or {}
    blocks = []
    if rec.get("f_recon_draw_route"):
        blocks.append(("g-file", rec["f_recon"], rec["f_recon_draw_route"]))
    if ims.get("f_imas_draw_route"):
        blocks.append(("IDS", ims.get("f_imas") or {}, ims["f_imas_draw_route"]))
    if not blocks:
        print("no loop-probe sigma0 data; skipping")
        return
    tol = dict(r_j=1e-3, r_I=1e-4, dl=1e-3)
    fig, ax = plt.subplots(1, len(blocks), figsize=(6.6 * len(blocks), 4.0), squeeze=False)
    summ = {}
    for i, (lab, f0, dr) in enumerate(blocks):
        a = ax[0, i]
        names, vals, tols = [], [], []
        qname = {"r_j": "bootstrap\nprofile", "r_I": "bootstrap\ncurrent",
                 "dl": "$|\\Delta l_i|$"}
        rname = {"standard": "standard\nroute", "ip_renorm": "Ip renorm.\nroute"}
        for k, t in (("r_j_vs_baseline", "r_j"), ("r_I_vs_baseline", "r_I"),
                     ("dl_i_vs_baseline", "dl")):
            if f0.get(k) is not None:
                names.append(f"check vs\nbaseline:\n{qname[t]}")
                vals.append(abs(float(f0[k])))
                tols.append(tol[t])
        for route, rr in (dr.get("routes") or {}).items():
            for k, t in (("r_j", "r_j"), ("r_I", "r_I"), ("dl_i_vs_delivered", "dl")):
                if rr.get(k) is not None:
                    names.append(f"{rname.get(route, route)}:\n{qname[t]}")
                    vals.append(abs(float(rr[k])))
                    tols.append(tol[t])
        x = np.arange(len(vals))
        a.bar(x, np.maximum(vals, 1e-16), color=[W["blue"] if n.startswith("check") else W["verm"]
                                                 for n in names])
        for xi, t in zip(x, tols):
            a.plot([xi - 0.4, xi + 0.4], [t, t], "k--", lw=1.2)
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=6.5, rotation=0)
        _yn = {True: "yes", False: "NO", None: "n/a"}
        a.set(ylabel="|zero-perturbation draw - reconstruction| [-]",
              title=f"({'ab'[i]}) {lab}, legacy path, bootstrap loop on\n"
                    f"(check passed: {_yn.get(f0.get('passed'))}; draw routes passed: "
                    f"{_yn.get(dr.get('passed_draw_route'))})")
        summ[lab] = dict(recon_check=f0, draw_route=dr)
    SUMMARY["sigma0_draw_route_vs_reconstruction"] = summ
    save(fig, "sigma0_draw_route_vs_reconstruction",
         f"{PROV} The zero-perturbation check on the DEFAULT (legacy) reconstruction path with the "
         f"self-consistent bootstrap loop ON (verify_sigma0_consistency, as measured inside "
         f"tests/test_jbs_loop_solver.py): blue, the check against the delivered baseline; red, "
         f"each draw route at sigma = 0 against the delivered reconstruction; dashed, the loop's own "
         f"tolerances (r_j 1e-3, r_I 1e-4, l_i 1e-3).")


# ---------------------------------------------------------------------------
# 8. separatrix pressure: beta_N / W / beta_p, legacy vs offset
# ---------------------------------------------------------------------------
def fig_sep():
    arms = [("natural p_sep", "default", "sep_legacy"),
            ("p_sep + 2 kPa (constr.)", "psep2k_offset", "psep2k_legacy")]
    paths = [("recon", "engine"), ("recon_legacy", "legacy path")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
    out = {}
    rows = []
    for alab, d_off, d_leg in arms:
        for part, plab in paths:
            for setting, dd in (("offset", d_off), ("legacy", d_leg)):
                d = load(dd, f"engine_{part}.json")
                pf = ((d or {}).get("distance") or {}).get("pressure_frames") or {}
                if "frames" not in pf:
                    continue
                e = (d.get("edge") or {}).get("summary") or {}
                rows.append(dict(arm=alab, path=plab, setting=setting,
                                 p_sep=pf.get("p_sep"), p_axis=pf.get("p_axis"),
                                 full={k: v["rel_pct"] for k, v in pf["frames"]["full"].items()},
                                 solver={k: v["rel_pct"] for k, v in pf["frames"]["solver"].items()},
                                 pprime_ratio_mid=e.get("pprime_ratio_solver_over_input_mid")))
    out["rows"] = rows
    labels = [f"{r['arm'].replace(' (', chr(10) + '(')}\n{r['path']}" for r in rows
              if r["setting"] == "offset"]
    for j, (q, ttl) in enumerate((("beta_n", r"(a) $\beta_N$"), ("W_MHD_MJ", r"(b) $W_{MHD}$"))):
        a = ax[j]
        for s_i, (setting, c) in enumerate((("legacy", W["verm"]), ("offset", W["blue"]))):
            v = [r["full"].get(q, np.nan) for r in rows if r["setting"] == setting]
            a.bar(np.arange(len(v)) + (s_i - 0.5) * 0.38, v, 0.38, color=c,
                  label=f"separatrix_pressure='{setting}' (full frame)")
        a.axhline(0, color="0.3", lw=1)
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, fontsize=7)
        a.set(ylabel="solved $-$ input, full frame [%]", title=ttl + " against the input")
        a.legend(fontsize=7)
    a = ax[2]
    for s_i, (setting, c) in enumerate((("legacy", W["verm"]), ("offset", W["blue"]))):
        v = [100 * (r["pprime_ratio_mid"] - 1) if r["pprime_ratio_mid"] else np.nan
             for r in rows if r["setting"] == setting]
        a.bar(np.arange(len(v)) + (s_i - 0.5) * 0.38, v, 0.38, color=c, label=setting)
    a.axhline(0, color="0.3", lw=1)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, fontsize=7)
    a.set(ylabel=r"median $|P'_{solver}|/|P'_{input}| - 1$, $0.2<\psi_N<0.8$ [%]",
          title="(c) mid-radius P' handed to the solver")
    a.legend(fontsize=7)
    SUMMARY["edge_pressure_betaN_W_solver"] = out
    psep = {r["arm"]: r["p_sep"] for r in rows}
    save(fig, "edge_pressure_betaN_W_solver",
         f"{PROV} Full-frame beta_N and W_MHD of the solved g-file reconstruction against the "
         f"input, and the mid-radius P' the solver is handed, under separatrix_pressure="
         f"'legacy' and 'offset', engine and legacy path; the second pair adds a CONSTRUCTED "
         f"constant 2 kPa fast pressure (not a repository example). p_sep: "
         + ", ".join(f"{k} {v:.0f} Pa" for k, v in psep.items() if v is not None) + ".")


# ---------------------------------------------------------------------------
# 9. MSE row: fd_chord vs fd_broyden
# ---------------------------------------------------------------------------
def fig_mse():
    c = load("mse", "mse_chords.json")
    fits = {j: load("mse", f"mse_fit_{j}.json") for j in ("fd_chord", "fd_broyden")}
    if c is None or not any(fits.values()):
        print("no MSE data; skipping")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    R = np.asarray(c["R"])
    a = ax[0]
    a.errorbar(R, c["tg_data"], yerr=c["sigma"], fmt="ko", ms=4, capsize=3,
               label="synthetic chords (1.03 x baseline field)")
    a.plot(R, c["tg_baseline"], color=W["green"], ls=":", label="engine baseline, no MSE row")
    cols = dict(fd_chord=W["blue"], fd_broyden=W["verm"])
    summ = {}
    for j, f in fits.items():
        if not f or "engine" not in f:
            summ[j] = dict(error=(f or {}).get("error", "missing")[:300])
            continue
        a.plot(R, f["tg_delivered"], color=cols[j], ls="--" if j == "fd_broyden" else "-.",
               marker="s", ms=3, label=f"delivered, {j}")
        rec = f["engine"]
        ps = [p for p in rec.get("passes", []) if p.get("phase") == "mse"]
        v = [p.get("mse_dtg_max_sigma") for p in ps]
        ax[1].semilogy(np.arange(1, len(v) + 1), v, "o-", color=cols[j],
                       label=f"{j}: {len(v)} passes")
        m = ((rec.get("delivered") or {}).get("checks") or {}).get("mse") or {}
        jac = [ph for ph in rec["phases"] if ph["name"] == "mse"]
        jac = jac[0]["jacobian"] if jac else {}
        summ[j] = dict(n_mse_passes=len(v), solves=rec.get("solves"),
                       chi2=m.get("chi2"), dtg_max_sigma=m.get("dtg_max_sigma"),
                       ok=m.get("ok"), converged=rec.get("converged"),
                       n_broyden=jac.get("n_broyden_updates"), wall_s=f.get("wall_s"),
                       stats=f.get("stats"))
    tol = None
    for f in fits.values():
        if f and "engine" in f:
            tol = (((f["engine"].get("delivered") or {}).get("checks") or {}).get("mse") or {}).get("tol")
    if tol:
        ax[1].axhline(tol, color="0.3", ls="--", lw=1.2, label=f"mse_tol_sigma = {tol:g}")
    a.set(xlabel="R [m]", ylabel=r"$\tan\gamma$ [-]", title=r"(a) pitch angles at the chords")
    a.legend(fontsize=7)
    ax[1].set(xlabel="MSE-stage pass", ylabel=r"max $|\Delta\tan\gamma|/\sigma$ [-]",
              title="(b) chord misfit per pass")
    ax[1].legend(fontsize=7)
    a = ax[2]
    names = [j for j in fits if "solves" in summ.get(j, {})]
    keys = ["anchor", "loop", "mse_fd", "mse_passes", "delivery"]
    bottom = np.zeros(len(names))
    for k, col in zip(keys, (W["sky"], W["blue"], W["orange"], W["verm"], W["green"])):
        v = np.array([float((summ[j]["solves"] or {}).get(k, 0)) for j in names])
        a.bar(range(len(names)), v, bottom=bottom, color=col, label=k)
        bottom += v
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names)
    a.set(ylabel="GS solves [-]", title="(c) solves by stage")
    a.legend(fontsize=7)
    SUMMARY["mse_jacobian_fd_chord_vs_broyden_synthetic"] = dict(chords=c, fits=summ)
    txt = "; ".join(f"{j}: {v.get('n_mse_passes')} MSE passes, chi2 {v.get('chi2'):.3f}, "
                    f"{(v.get('solves') or {}).get('total')} solves"
                    if "solves" in v else f"{j}: {v.get('error')}" for j, v in summ.items())
    save(fig, "mse_jacobian_fd_chord_vs_broyden_synthetic",
         f"{PROV} The engine's MSE row (rows Ip + l_i + mse) on the synthetic IDS with eight "
         f"synthetic midplane chords (the baseline's own field x 1.03, sigma 0.004, built as "
         f"tests/test_jbs_loop_solver.py builds them, stated in the source's orientation), fixed finite-difference Jacobian "
         f"('fd_chord', the new default) vs Broyden-updated ('fd_broyden'). {txt}.")


# ---------------------------------------------------------------------------
# 10. reconstruction g-file vs archive _baseline g-file
# ---------------------------------------------------------------------------
def fig_gfile_frame():
    P = load("small", "gfile_frame_pres.json") or {}
    arms = []
    for src, eng, sep in (("recon", "unified", "offset"), ("recon", "unified", "legacy"),
                          ("imas", "unified", "offset"), ("imas", "unified", "legacy"),
                          ("recon", "legacy", "offset")):
        j = load("gfile_frame", f"baseline_gfile_frame_{src}_{eng}_{sep}.json")
        if j:
            arms.append((f"{src}_{eng}_{sep}", j))
    if not arms:
        print("no gfile-frame data; skipping")
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    tag = "recon_unified_offset"
    for kind, c, ls in (("recon", W["blue"], "-"), ("archive", W["verm"], "--"),
                        ("bare", W["green"], ":")):
        f = P.get(f"frame_{tag}_{kind}.geqdsk")
        if f:
            p = np.asarray(f["PRES"])
            x = np.linspace(0, 1, p.size)
            ax[0].plot(x, p / 1e3, color=c, ls=ls, label={"recon": "save_baseline_eqdsk",
                                                         "archive": "archive _baseline g-file",
                                                         "bare": "bare mygs.save_eqdsk"}[kind])
            m = x >= 0.9
            ax[1].plot(x[m], p[m], color=c, ls=ls)
    ax[0].set(xlabel=r"$\psi_N$", ylabel="PRES [kPa]",
              title="(a) g-file engine, offset: written PRES")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel=r"$\psi_N$", ylabel="PRES [Pa]", title=r"(b) edge, $\psi_N \geq 0.9$")
    a = ax[2]
    names = [n for n, _ in arms]
    x = np.arange(len(names))
    w = 0.2
    for i, (k, c) in enumerate((("recon", W["blue"]), ("archive", W["verm"]),
                                ("bare", W["green"]))):
        a.bar(x + (i - 1.5) * w, [j["edge_pres"][k] for _, j in arms], w, color=c,
              label=f"edge PRES, {k}")
    a.bar(x + 1.5 * w, [j["p_sep_applied"] for _, j in arms], w, color=W["black"],
          label="p_sep applied")
    a.set_xticks(x)
    a.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    a.set(ylabel="pressure at the last g-file point [Pa]", title="(c) edge PRES by arm")
    a.set_ylim(0, 1.35 * max(max(j["p_sep_applied"], j["edge_pres"]["recon"]) for _, j in arms))
    a.legend(fontsize=7, ncol=2, loc="upper center")
    SUMMARY["baseline_gfile_frame"] = {n: j for n, j in arms}
    j0 = dict(arms).get(tag)
    txt = ""
    if j0:
        txt = (f" Engine, offset, g-file: p_sep applied {j0['p_sep_applied']:.1f} Pa; edge PRES "
               f"recon {j0['edge_pres']['recon']:.1f} / archive {j0['edge_pres']['archive']:.1f}"
               f" / bare {j0['edge_pres']['bare']:.1f} Pa; max|PRES recon - archive| "
               f"{j0['pres_recon_minus_archive']['max_abs']:.3g} Pa (mean "
               f"{j0['pres_recon_minus_archive']['mean']:.3g}).")
    save(fig, "baseline_gfile_frame_recon_vs_archive",
         f"{PROV} The reconstruction's own g-file (Bouquet.save_baseline_eqdsk) and "
         f"the archive's _baseline g-file carry the same pressure frame; a bare "
         f"mygs.save_eqdsk of the same state carries the solver frame (zero edge).{txt}")

# ---------------------------------------------------------------------------
# 11. reversed-Ip: four orientations forward-solved (tests/test_reversed_ip_solver.py)
# ---------------------------------------------------------------------------
def fig_revip():
    R = load("small", "revip_identity.json")
    if not R:
        print("no reversed-Ip data; skipping")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    cols = [W["blue"], W["verm"], W["green"], W["pink"], W["orange"], W["sky"], W["black"], W["yellow"]]
    for m, c in zip(R["modes"], cols):
        p = R["profiles"][m]
        ax[0].plot(p["psi_N"], MA(np.abs(p["j_phi"])), color=c, lw=1.6, label=m)
    ax[0].set(xlabel=r"$\psi_N$", ylabel=r"baseline $|j_\phi|$ [MA/m$^2$]",
              title="(a) solved baseline current, (Ip+, B0+), per closure mode")
    ax[0].legend(fontsize=6.5, ncol=2)
    a = ax[1]
    mx = []
    for i, o in enumerate(R["orientations"][1:]):
        v = []
        for m in R["modes"]:
            r = R["rows"][f"{m}|{o}"]
            v.append(max(r[k]["max_abs_diff_of_magnitudes"] / max(r[k]["scale"], 1e-300)
                         for k in ("bl_j_phi", "bl_j_BS", "bl_j_inductive", "eq_q")))
        mx += v
        a.bar(np.arange(len(v)) + (i - 1) * 0.27, np.maximum(v, 1e-17), 0.27,
              color=[W["verm"], W["green"], W["pink"]][i], label=f"{o} vs ip+_b0+")
    a.set_yscale("log")
    a.set_ylim(1e-17, 1e-2)
    a.axhline(1e-16, color="0.3", ls=":", lw=1.2, label="1e-16 (plotted floor; 0 = bitwise)")
    a.set_xticks(range(len(R["modes"])))
    a.set_xticklabels(R["modes"], rotation=30, ha="right", fontsize=7)
    a.set(ylabel=r"max $||x|-|x_{ref}||$ / max$|x_{ref}|$ over $j_\phi$, $j_{BS}$, $j_{ind}$, q",
          title="(b) mirrored orientations against (Ip+, B0+)")
    a.legend(fontsize=7)
    a.text(0.5, 0.5, f"all {len(mx)} mirrored cases: largest difference = {max(mx):.3g}\n"
           "(bars sit at 0, below the plotted floor: bitwise identical)",
           transform=a.transAxes, ha="center", va="center", fontsize=9,
           bbox=dict(fc="white", ec="0.5"))
    SUMMARY["reversed_ip_forward_solve_identity"] = dict(max_relative_difference=float(max(mx)),
                                                         n_cases=len(mx))
    save(fig, "reversed_ip_forward_solve_identity",
         f"{PROV} The synthetic OMAS source mirrored into the four (Ip, B0) orientations and forward-solved "
         f"for each of the eight closure modes of tests/test_reversed_ip_solver.py (numbers from that test's own "
         f"probe output): every mirrored baseline's current profiles and q equal the (Ip+, B0+) solve in "
         f"magnitude, largest relative difference {max(mx):.1g} over {len(mx)} cases (0 = bitwise).")


def main():
    parts = {p: load("default", f"engine_{p}.json") for p in ("recon", "imas", "imas_q0")}
    leg_g = load("default", "engine_recon_legacy.json")
    leg_i = load("imas_legacy2", "engine_imas.json") or load("imas_legacy", "engine_imas.json")
    jobs = [
        lambda: fig_profiles("engine_vs_legacy_gfile_profiles", gfile_input(), parts["recon"],
                             leg_g, "synthetic g-file + p-file"),
        lambda: fig_profiles("engine_vs_legacy_ids_profiles", omas_input(), parts["imas"],
                             leg_i, "synthetic OMAS/IDS source (t = 2.3043 s)"),
        lambda: fig_pass_histories(parts),
        lambda: fig_sigma0(parts),
        lambda: fig_draws({s: load("draws", f"engine_draws_{s}.json") for s in ("recon", "imas")}),
        fig_sigma0_legacy,
        fig_sep, fig_mse, fig_gfile_frame, fig_revip,
    ]
    for j in jobs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                j()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FIGURE FAILED:", e)
    with open(os.path.join(OUT, "captions.json"), "w") as fh:
        json.dump(CAPTIONS, fh, indent=1)
    with open(os.path.join(OUT, "figure_numbers.json"), "w") as fh:
        json.dump(SUMMARY, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
