#!/usr/bin/env python3
"""Draw-ensemble, draw-cost, sigma=0 true-route and reversed-Ip figures
(synthetic examples only), from the small JSON extract_draw_data.py wrote.  usage: make_draw_figures.py <data dir> <out dir> <bouquet sha> <build label>

* engine_draw_ensemble_bands  -- the seeded engine-draw batch at the SHIPPED
  NOTEBOOKS' inductive-shape sigma (0.05), bands of archived / in-spec draws;
  caption carries both yields (0.10 = UncertaintyConfig default, 0.05);
* engine_draw_cost            -- solves per draw by stage at 0.05;
* engine_sigma0_true_route    -- verify_sigma0_consistency, both engines x both
  synthetic sources, live numbers (residual / tolerance per stage);
* reversed_ip_identity_both_paths -- the legacy forward-solve mirrors (8 closure
  modes, test_reversed_ip_solver's own probe) and the engine arm (4 g-file
  mirrors, test_engine_reversed_ip_gfile_solver) against their stated bars.
Also writes yield_table.json / yield_table.md and captions.json.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, OUT, SHA, BUILD = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
W = dict(blue="#0072B2", verm="#D55E00", green="#009E73", pink="#CC79A7",
         orange="#E69F00", sky="#56B4E9", yellow="#F0E442", black="#000000")
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.linestyle": ":",
                     "lines.linewidth": 2, "savefig.dpi": 130,
                     "legend.fontsize": 8, "figure.constrained_layout.use": True})
PROV = ("Synthetic example shipped with the repository (examples/D3D-like); real "
        f"solver, OpenFUSIONToolkit {BUILD}, bouquet {SHA}, one thread.")
CAP, NUM = {}, {}
SRC = {"recon": "g-file", "imas": "IDS"}


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name, caption):
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    CAP[name] = caption
    print("wrote", name)


# ---------------------------------------------------------------------------
# yield table
# ---------------------------------------------------------------------------
def yields():
    rows = []
    for sg in ("0.10", "0.05"):
        for s in ("recon", "imas"):
            m = load(f"draws_{s}_{sg}_meta.json")
            if not m or "draws" not in m:
                continue
            dr = m["draws"]
            rej = dr.get("rejections") or []
            reasons = Counter(r.get("reason") for r in rej)
            caps = m.get("cap_events") or []
            # a cap rejection at the FIRST homotopy stage: generate_bouquet
            # announces a capped stage solve with outcome "rejected" exactly
            # when there is no earlier good stage to roll back to
            cap_first = [e for e in caps if e.get("stage") == "homotopy"
                         and e.get("outcome") == "rejected"]
            cap_rolled = [e for e in caps if e.get("outcome") == "rolled_back"]
            cap_other = [e for e in caps if e.get("stage") != "homotopy"]
            per = dr.get("per_draw") or []
            ph = [((p.get("post_hoc") or {}).get("reasons") or []) for p in per]
            arch_out = [p for p in per if not p.get("in_spec")]
            coil_out = sum(1 for p in per
                           if not (p.get("post_hoc") or {}).get("coil_in_spec", True))
            band_out = sum(1 for p in per
                           if not (p.get("post_hoc") or {}).get("in_band", True))
            rows.append(dict(
                sigma=sg, source=SRC[s], attempts=dr.get("attempts"),
                archived=dr.get("archived"), in_spec=dr.get("in_spec"),
                rejected=dr.get("rejected"), rejection_reasons=dict(reasons),
                cap_rejections_first_stage=len(cap_first),
                cap_rolled_back=len(cap_rolled),
                cap_events_other_stage=[dict(stage=e.get("stage"),
                                             outcome=e.get("outcome"))
                                        for e in cap_other],
                cap_seconds=[e.get("seconds") for e in caps],
                maxits=(caps[0].get("maxits") if caps else None),
                archived_out_of_spec=len(arch_out), coil_out=coil_out,
                band_out=band_out,
                post_hoc_reasons=Counter(x.split(":")[0] for r in ph for x in r),
                wall_s=dr.get("wall_s")))
    NUM["yield_table"] = rows
    L = ["| sigma_jphi | source | attempts | archived | in spec | rejected (by reason) | "
         "capped homotopy solves | archived out of spec (coil / l_i band) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rr = ", ".join(f"{k} {v}" for k, v in r["rejection_reasons"].items()) or "none"
        L.append(f"| {r['sigma']} | {r['source']} | {r['attempts']} | {r['archived']} | "
                 f"{r['in_spec']} | {r['rejected']}: {rr} | "
                 f"{r['cap_rejections_first_stage']} first-stage rejections at maxits "
                 f"{r['maxits']}, {r['cap_rolled_back']} rolled back | "
                 f"{r['archived_out_of_spec']} ({r['coil_out']} / {r['band_out']}) |")
    open(os.path.join(OUT, "yield_table.md"), "w").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.join(OUT, "yield_table.json"), "w"), indent=1,
              default=str)
    return rows


def _band(ax, x, Y, color, label):
    Y = np.asarray(Y, float)
    lo, med, hi = np.nanpercentile(Y, [16, 50, 84], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.3, lw=0, label=f"{label} 16-84 %")
    ax.plot(x, med, color=color, lw=1.6, label=f"{label} median")


def fig_draws(rows, sg="0.05"):
    srcs = [s for s in ("recon", "imas") if load(f"draws_{s}_{sg}_profiles.json")]
    if not srcs:
        print("no draw profiles")
        return
    keys = [("j_phi", r"$\langle j_\phi\rangle$ [MA/m$^2$]", 1e-6),
            ("q", "q", 1.0), ("pressure", "p [kPa]", 1e-3)]
    fig, ax = plt.subplots(len(srcs), 3, figsize=(13, 3.7 * len(srcs)), squeeze=False)
    for r, s in enumerate(srcs):
        P = load(f"draws_{s}_{sg}_profiles.json")
        bl, ds = P["baseline"], P["draws"]
        for c, (k, yl, f) in enumerate(keys):
            a = ax[r, c]
            xk = "psi_q" if k == "q" else "psi_N"
            have = [d for d in ds if k in d["profiles"] and xk in d["profiles"]]
            if not have:
                continue
            x = np.asarray(have[0]["profiles"][xk], float)
            Y = [np.interp(x, d["profiles"][xk], d["profiles"][k]) for d in have]
            Yabs = np.abs(Y) if k in ("j_phi", "q") else np.asarray(Y)
            _band(a, x, np.asarray(Yabs) * f, W["blue"], f"all archived (n={len(have)})")
            ins = [y for y, d in zip(Yabs, have) if bool(d["flags"].get("in_spec", False))]
            if len(ins) >= 2:
                _band(a, x, np.asarray(ins) * f, W["verm"], f"in spec (n={len(ins)})")
            elif len(ins) == 1:
                a.plot(x, np.asarray(ins[0]) * f, color=W["verm"], lw=1.4,
                       label="in spec (n=1)")
            if k in bl and xk in bl:
                yb = np.abs(bl[k]) if k in ("j_phi", "q") else np.asarray(bl[k])
                a.plot(bl[xk], np.asarray(yb) * f, color=W["black"], ls="--", lw=1.4,
                       label="reconstruction")
            a.set(xlabel=r"$\psi_N$", ylabel=yl,
                  title=f"({'abcdef'[3 * r + c]}) {SRC[s]} draws: {k}")
            a.legend(fontsize=7)
    y = {(r["sigma"], r["source"]): r for r in rows}

    def ytxt(sgx):
        return "; ".join(f"{src}: {y[(sgx, src)]['attempts']} attempts, "
                         f"{y[(sgx, src)]['archived']} archived, "
                         f"{y[(sgx, src)]['in_spec']} in spec"
                         for src in ("g-file", "IDS") if (sgx, src) in y)
    save(fig, "engine_draw_ensemble_bands",
         f"{PROV} Seeded engine-draw batches (12 draws requested, seed 12345) through "
         f"Bouquet.generate() with reconstruction_engine='unified', at the shipped notebooks' "
         f"inductive-shape sigma (UncertaintyConfig.jphi_scalar_sigma = 0.05, as set in "
         f"examples/D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb and "
         f"bouquet_D3Dlike_omas_example.ipynb; every other uncertainty at the default, which the "
         f"notebooks also use): median and 16-84 % bands of the archived draws and of the in-spec "
         f"subset, with the reconstruction. 'In spec' = generate()'s flag: coil drift within "
         f"+/-2 % AND l_i within the engine's +/-5 % band. Yield at 0.05: {ytxt('0.05')}. "
         f"Same batch at the UncertaintyConfig default 0.10: {ytxt('0.10')}.")


def fig_cost(sg="0.05"):
    """Solves per archived draw by stage (needs only the batches' meta files)."""
    srcs = [s for s in ("recon", "imas") if load(f"draws_{s}_{sg}_meta.json")]
    if not srcs:
        print("no draw meta")
        return
    fig, ax = plt.subplots(1, len(srcs), figsize=(6.5 * len(srcs), 3.9), squeeze=False)
    stages = [("loop", W["blue"]), ("homotopy", W["orange"]),
              ("post_homotopy", W["verm"]), ("filters", W["green"])]
    stage_label = {"loop": "bootstrap-loop solves", "homotopy": "homotopy solves",
                   "post_homotopy": "post-homotopy solves", "filters": "filter solves"}
    cost = {}
    for i, s in enumerate(srcs):
        m = load(f"draws_{s}_{sg}_meta.json")
        rws = (m.get("draws") or {}).get("per_draw", [])
        a = ax[0, i]
        bottom = np.zeros(len(rws))
        for st, c in stages:
            v = np.array([((r.get("cost") or {}).get(st) or {}).get("solves") or 0
                          for r in rws], float)
            a.bar(np.arange(len(rws)), v, bottom=bottom, color=c, label=stage_label[st])
            bottom += v
        ph = [((r.get("cost") or {}).get("post_homotopy") or {}).get("passes") or 0
              for r in rws]
        a2 = a.twinx()
        a2.plot(np.arange(len(rws)), ph, "kD", ms=5, label="post-homotopy passes")
        a2.axhline(6, color="k", ls="--", lw=1.2, label="post-homotopy ceiling 6")
        a2.set_ylim(0, 10)
        a.set_ylim(0, 1.75 * max(bottom.max() if bottom.size else 1, 1))
        a2.set_ylabel("post-homotopy passes [-]")
        a2.grid(False)
        a.set(xlabel="archived draw index", ylabel="Grad-Shafranov solves per draw [-]",
              title=f"({'ab'[i]}) {SRC[s]}: solves by stage")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=3)
        wall = [r.get("time_s") for r in rws if r.get("time_s") is not None]
        cost[SRC[s]] = dict(median_wall_s=float(np.median(wall)) if wall else None,
                            max_post_homotopy_passes=max(ph or [0]))
    NUM["engine_draw_cost"] = cost
    txt = "; ".join(f"{k}: median {v['median_wall_s']:.0f} s per archived draw, max "
                    f"post-homotopy passes {v['max_post_homotopy_passes']}"
                    for k, v in cost.items() if v["median_wall_s"] is not None)
    save(fig, "engine_draw_cost",
         f"{PROV} GS solves per archived engine draw by stage (bars) and the post-homotopy "
         f"passes against the approved ceiling of 6 (diamonds, dashed), batch at "
         f"jphi_scalar_sigma = 0.05 (the notebooks' setting). {txt}.")


def fig_sigma0():
    recs = {}
    for s in ("recon", "imas"):
        for e in ("unified", "legacy"):
            d = load(f"sigma0_{s}_{e}.json")
            if d and "record" in d:
                recs[(s, e)] = d
    stored = load("figure_numbers.json") if not recs else None
    if not recs and not (stored and "engine_sigma0_true_route" in stored):
        print("no sigma0 data")
        return
    if not recs:
        # re-plot from the numbers a previous run stored (figure_numbers.json):
        # the same values, so the same bars
        return _plot_sigma0(stored["engine_sigma0_true_route"])
    summ = {}
    for (s, e), d in recs.items():
        r = d["record"]
        if e == "unified":
            summ[f"{s}_engine"] = dict(passed=r.get("passed"), amplitude=r.get("amplitude"),
                                       r_j=r.get("r_j"), r_I=r.get("r_I"), dl_i=r.get("dl_i"),
                                       dq0=r.get("dq0"), dq95=r.get("dq95"),
                                       flux_range_rel=r.get("flux_range_rel"),
                                       request_bit_identical=r.get("request_bit_identical"),
                                       loop=(r.get("stages") or {}).get("loop"),
                                       t_check_s=d.get("t_check_s"))
        else:
            dr = r.get("draw_route") or {}
            routes = dr.get("routes") or {}
            summ[f"{s}_legacy"] = dict(passed=r.get("passed"),
                                       routes={k: {q: v.get(q) for q in (
                                           "r_j", "r_I", "dl_i_vs_l_i_target",
                                           "passed_draw_route")}
                                               for k, v in routes.items()
                                               if isinstance(v, dict)})
    return _plot_sigma0(summ)


#: the loop's own tolerances (GenerationConfig defaults: jbs_rtol_j,
#: jbs_rtol_Ip, jbs_tol_li), the bars every stage is judged at
_TOL = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3)
_ROUTE = {"standard": "standard", "ip_renorm": "Ip renorm."}


def _plot_sigma0(summ):
    """Panels (a) residual over tolerance per route and stage, (b) the engine's
    archived-state change, from the summary of the checks (plain-language labels)."""
    t = _TOL
    labels, vals = [], []
    for k, v in summ.items():
        s = k.split("_")[0]
        if k.endswith("_engine"):
            lp = v.get("loop") or {}
            for stg, z in (("loop", lp), ("archived", v)):
                if z.get("r_j") is None:
                    continue
                labels.append(f"{SRC[s]}\nengine\n{stg}")
                vals.append([abs(z["r_j"]) / t["rtol_j"], abs(z["r_I"]) / t["rtol_Ip"],
                             abs(z["dl_i"]) / t["tol_li"]])
        else:
            for rn, z in (v.get("routes") or {}).items():
                if z.get("r_j") is None:
                    continue
                labels.append(f"{SRC[s]}\nlegacy\n{_ROUTE.get(rn, rn)}")
                vals.append([abs(z["r_j"]) / t["rtol_j"], abs(z["r_I"]) / t["rtol_Ip"],
                             abs(z.get("dl_i_vs_l_i_target") or 0.0) / t["tol_li"]])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    names = ["bootstrap profile residual", "bootstrap current residual",
             r"|$\Delta l_i$|"]
    n = len(labels)
    wdt = 0.8 / 3
    for j in range(3):
        ax[0].bar(np.arange(n) + (j - 1) * wdt, [max(v[j], 1e-6) for v in vals], wdt,
                  color=[W["blue"], W["orange"], W["green"]][j], label=names[j])
    ax[0].axhline(1.0, color="0.3", ls="--", lw=1.2, label="tolerance (ratio 1)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 10)
    ax[0].set_xticks(range(n))
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set(ylabel="residual / its tolerance [-]",
              title="(a) zero-perturbation draw: every route and stage")
    ax[0].legend(fontsize=7, ncol=2)
    a = ax[1]
    eng = [(k, v) for k, v in summ.items() if k.endswith("_engine")]
    for i, (k, v) in enumerate(eng):
        for j, (q, c, ql) in enumerate((("dq0", W["sky"], r"|$\Delta q_0$|"),
                                        ("dq95", W["orange"], r"|$\Delta q_{95}$|"),
                                        ("flux_range_rel", W["pink"],
                                         "|relative change of the flux range|"))):
            x = v.get(q)
            a.bar(i + (j - 1) * 0.25, abs(x) if x is not None else 0.0, 0.25, color=c,
                  label=ql if i == 0 else None)
    a.set_yscale("log")
    a.set_xticks(range(len(eng)))
    a.set_xticklabels([f"{SRC[k.split('_')[0]]}, engine" for k, _ in eng])
    a.set(ylabel="absolute / relative change [-]",
          title="(b) engine, archived state against the reconstruction")
    a.legend(fontsize=7)
    NUM["engine_sigma0_true_route"] = summ
    worst = max(max(v) for v in vals)
    amp = {k: v.get("amplitude") for k, v in summ.items() if k.endswith("_engine")}
    save(fig, "engine_sigma0_true_route",
         f"{PROV} verify_sigma0_consistency on both synthetic sources and both engines, the "
         f"true route: under reconstruction_engine='unified' ONE draw through generate() with "
         f"every perturbation zero (loop stage and archived post-homotopy stage); under 'legacy' "
         f"each draw route of perturb_kinetic_equilibrium at sigma = 0. Bars: residual over the "
         f"loop's own tolerance (r_j 1e-3, r_I 1e-4, l_i 1e-3, unchanged); largest ratio "
         f"{worst:.2g}. All pass: "
         + ", ".join(f"{k.replace('_', ' ')} {v.get('passed')}" for k, v in summ.items())
         + f". The engine check restores all solver state and is bit-reproducible "
           f"(inductive amplitude {amp}).")


def fig_revip():
    R = load("revip_identity.json")
    E = load("revip_engine.json")
    if not (R or E):
        print("no revip data")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    mx = []
    if R:
        a = ax[0]
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
        a.axhline(1e-16, color="0.3", ls=":", lw=1.2, label="plotted floor (0 = bitwise)")
        a.set_xticks(range(len(R["modes"])))
        a.set_xticklabels(R["modes"], rotation=30, ha="right", fontsize=7)
        a.text(0.5, 0.5, f"all {len(mx)} mirrored cases: largest difference = {max(mx):.3g}\n"
               "(0 = bitwise identical; bars at 0 sit below the plotted floor)",
               transform=a.transAxes, ha="center", va="center", fontsize=9,
               bbox=dict(fc="white", ec="0.5"))
        a.set(ylabel="max rel. difference of magnitudes",
              title="(a) legacy path: synthetic IDS mirrors, 8 closure modes")
        a.legend(fontsize=7)
    if E:
        a = ax[1]
        ors = [o for o in E if not o.startswith("_") and o != "ip+1_bt+1"]
        ks = ("j_phi", "j_inductive", "j_BS", "l_i_rel")
        for j, k in enumerate(ks):
            v = [E[o].get(k, 0.0) for o in ors]
            xs = np.arange(len(ors)) + (j - 1.5) * 0.15
            a.plot(xs, np.maximum(v, 1e-16), "o", ms=8,
                   color=[W["blue"], W["green"], W["verm"], W["orange"]][j], label=k)
        for i, o in enumerate(ors):
            if all(E[o].get(k, 0.0) == 0.0 for k in ks):
                a.annotate("0 (bitwise)", (i, 1e-16), xytext=(0, 12),
                           textcoords="offset points", ha="center", fontsize=8)
        a.axhline(1e-4, color="k", ls="--", lw=1.2,
                  label="stated bar _IP_MIRROR_REL = 1e-4 (Ip mirrors)")
        a.axhline(1e-16, color="0.3", ls=":", lw=1.2, label="plotted floor (0 = bitwise)")
        a.set_yscale("log")
        a.set_ylim(1e-17, 1e-2)
        a.set_xticks(range(len(ors)))
        a.set_xticklabels(ors, fontsize=8)
        a.set(ylabel="max|x - x_ref| / max|x_ref|",
              title="(b) unified engine: g-file mirrors vs the unmirrored file")
        a.set_xlim(-0.6, len(ors) - 0.4)
        a.legend(fontsize=7, loc="center left")
        worst = max(max(E[o].get(k, 0.0) for k in ks) for o in ors)
        NUM["reversed_ip_engine"] = {o: {k: E[o].get(k) for k in ks + ("converged",
                                                                       "delivered_ok")}
                                     for o in ors}
    else:
        worst = None
    NUM["reversed_ip_legacy_max"] = max(mx) if mx else None
    save(fig, "reversed_ip_identity_both_paths",
         f"{PROV} Reversed plasma current / toroidal field. (a) legacy path: the synthetic OMAS "
         f"source mirrored into the four (Ip, B0) orientations and forward-solved for each of the "
         f"eight closure modes of tests/test_reversed_ip_solver.py: largest relative difference "
         f"{(max(mx) if mx else float('nan')):.1g} over {len(mx)} cases (0 = bitwise). (b) unified "
         f"engine: the synthetic g-file mirrored in its own COCOS and reconstructed "
         f"(tests/test_engine_reversed_ip_gfile_solver.py): the B_t mirror is bitwise; the Ip "
         f"mirrors differ at the reader's contour-tracing level, worst "
         f"{(worst if worst is not None else float('nan')):.3g} against the "
         f"stated bar 1e-4.")


rows = yields()
for f in (lambda: fig_draws(rows), fig_cost, fig_sigma0, fig_revip):
    try:
        f()
    except Exception:
        import traceback
        traceback.print_exc()
json.dump(CAP, open(os.path.join(OUT, "captions.json"), "w"), indent=1)
json.dump(NUM, open(os.path.join(OUT, "figure_numbers.json"), "w"), indent=1, default=str)
