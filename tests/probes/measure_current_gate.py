#!/usr/bin/env python
"""Measure the opt-in closure-half current gate (``jbs_gate_current_residual``)
with the live solver -- a MEASUREMENT, not a test (pytest does not collect
this file, and nothing here asserts).

For each synthetic example of the repository (``examples/D3D-like``: the
g-file + p-file reconstruction and the OMAS modelling source -- the input
files ``tests/test_jbs_loop_solver.py`` uses) and each closure channel the
solver tests exercise, the BASELINE is built with the gate OFF and ON, each
in its own interpreter (``OFT_env`` is a per-process singleton, and a fresh
process gives both settings the same start).  Then a small seeded draw batch
on the g-file example, gate OFF and ON.

Per case and setting it records: converged or not; passes used; the final
residuals (every one, the closure-half current residual included) and which
criterion was the last to be met; l_i, q0, q95, the bootstrap current
integral and Ip of the delivered equilibrium; and, per case, the DIFFERENCE
gate ON - gate OFF in l_i, q0, q95 and in the delivered current profile (max
and rms, relative).  Per draw: archived or rejected (with the reason), passes
used, and the same delivered quantities.

Usage::

    python tests/probes/measure_current_gate.py OUTDIR \\
        [--parts recon,imas_rescale,...] [--n-draws 6] [--seed 12345]
        [--jobs 1] [--child-timeout SECONDS]

Writes ``OUTDIR/current_gate_measurement.json`` (the summary),
``OUTDIR/parts/<part>__gate_<off|on>.json`` (one per child, written as it
finishes, so a partial run keeps what it measured) and
``OUTDIR/logs/<part>__gate_<off|on>.log`` (each child's full output).
Single-threaded (``nthreads=1`` and every BLAS/OpenMP thread count 1); no
network.  OpenFUSIONToolkit is found the way the solver tests find it
(``OFT_PYTHONPATH``, else the sibling checkout's ``build_release/python``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import _harness  # noqa: E402

_harness.ensure_repo_on_syspath()

_EXAMPLE = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like")
_OMAS = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.3043          # the OMAS example's time slice (as the solver tests)

#: part -> the GenerationConfig fields it sets (IMAS parts) -- the channels
#: tests/test_jbs_loop_solver.py drives through the loop.  "diff" is not a
#: part: its baseline runs no loop (the source total is pinned).
IMAS_PARTS = {
    "imas_rescale": dict(jBS_baseline_mode="rescale"),
    "imas_ohmic_bootstrap": dict(jBS_baseline_mode="ohmic",
                                 closure_channel="bootstrap",
                                 jbs_init="anchor"),
    "imas_ohmic_sawtooth_bootstrap": dict(
        jBS_baseline_mode="ohmic", closure_channel="sawtooth_bootstrap"),
    "imas_ohmic_structured": dict(jBS_baseline_mode="ohmic",
                                  closure_channel="structured",
                                  structured_li_target=None),
    "imas_ohmic_structured_mse": dict(jBS_baseline_mode="ohmic",
                                      closure_channel="structured",
                                      structured_li_target=None),
}
ALL_PARTS = ["recon"] + list(IMAS_PARTS)
_THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg")


# ===========================================================================
#  child side (runs the solver)
# ===========================================================================
def _oft_on_path():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    import OpenFUSIONToolkit  # noqa: F401  (fail loudly here, not later)


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _loop_summary(rec):
    """Compact, JSON-safe summary of one loop record."""
    from bouquet.jbs_loop import criteria_timeline, jsonable
    if not rec:
        return None
    rec = jsonable(dict(rec))
    keys = ("label", "converged", "n_passes", "stop_reason", "criteria",
            "tolerances", "final", "r_j", "r_I", "dl_i", "dq0", "li", "q0",
            "omega", "current_gap", "current_residual_unrelaxed",
            "current_residual", "current_residual_estimate",
            "current_residual_direct_over_estimate", "current_residual_ok",
            "current_gate", "last_criterion_met", "pass_ok", "wall_s",
            "relax_current", "current_relaxation")
    out = {k: rec.get(k) for k in keys if k in rec}
    try:
        out["timeline"] = criteria_timeline(rec)
    except Exception as e:               # never lose the record over this
        out["timeline"] = dict(error=f"{type(e).__name__}: {e}")
    return out


class _LoopLog:
    """Wraps ``bouquet.jbs_loop.run_jbs_loop`` (every caller imports it at
    call time) and logs every loop run -- converged, failed or raised --
    against the draw attempt that ran it."""

    def __init__(self):
        import bouquet.jbs_loop as JL
        self.JL = JL
        self.orig = JL.run_jbs_loop
        self.entries = []
        self.draw = None
        JL.run_jbs_loop = self._wrapped

    def _wrapped(self, *a, **k):
        t0 = time.perf_counter()
        rec, exc = None, None
        try:
            out = self.orig(*a, **k)
            rec = out.get("record")
            return out
        except self.JL.JBSNotConverged as e:
            rec, exc = e.record, f"{type(e).__name__}: {str(e)[:400]}"
            raise
        except Exception as e:
            exc = f"{type(e).__name__}: {str(e)[:400]}"
            raise
        finally:
            self.entries.append(dict(
                draw=self.draw, label=k.get("label"),
                wall_s=time.perf_counter() - t0, exception=exc,
                summary=_loop_summary(rec)))

    def for_draw(self, d):
        return [e for e in self.entries if e["draw"] == d]


def _jphi_achieved(mygs, psi_N, psi_pad, sign_ref):
    from bouquet.TokaMaker_interface import _achieved_jphi_fsa
    return _achieved_jphi_fsa(mygs, psi_N, psi_pad, sign_ref=sign_ref)


def _delivered(mygs, psi_N, psi_pad, j_BS, j_phi_ref):
    """l_i, q0, q95, Ip, the bootstrap current integral and the achieved
    current profile of the equilibrium ``mygs`` holds now."""
    import numpy as np
    from scipy.integrate import trapezoid
    from bouquet.jbs_loop import residual_weights
    psi_N = np.asarray(psi_N, dtype=float)
    st = mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
    out = dict(l_i=_f(st.get("l_i")), q0=_f(st.get("q_0")),
               q95=_f(st.get("q_95")), Ip=_f(st.get("Ip")))
    try:
        out["l_i_std"] = _f(mygs.get_stats(li_normalization="std",
                                           lcfs_pad=psi_pad)["l_i"])
    except Exception as e:
        out["l_i_std_error"] = f"{type(e).__name__}: {e}"
    j_ach = _jphi_achieved(mygs, psi_N, psi_pad, j_phi_ref)
    w, x, kind = residual_weights(mygs.copy_eq(), psi_N, psi_pad)
    out["weights_kind"] = kind
    if j_BS is not None:
        j_BS = np.asarray(j_BS, dtype=float)
        out["I_BS"] = _f(trapezoid(w * j_BS, x))
        out["j_BS"] = j_BS.tolist()
    out["I_phi_achieved"] = _f(trapezoid(w * j_ach, x))
    if out.get("I_BS") is not None and out.get("Ip"):
        out["I_BS_over_Ip"] = out["I_BS"] / abs(out["Ip"])
    out["psi_N"] = psi_N.tolist()
    out["w"] = np.asarray(w, dtype=float).tolist()
    out["j_phi_achieved"] = np.asarray(j_ach, dtype=float).tolist()
    return out


def _baseline_block(b, bl, psi_pad, t0):
    import numpy as np
    from bouquet.jbs_loop import jsonable
    rec = b._baseline_jbs_record()
    blk = dict(status="delivered", wall_s=time.perf_counter() - t0,
               converged=bool((rec or {}).get("converged")),
               n_passes=(rec or {}).get("n_passes"),
               loop=_loop_summary(rec), record=jsonable(rec))
    for sub in ("post_corrective", "mse_stage"):
        if isinstance((rec or {}).get(sub), dict):
            s = rec[sub]
            blk[sub] = dict(summary={k: jsonable(v) for k, v in s.items()
                                     if k != "passes"},
                            passes=_loop_summary(s.get("passes"))
                            if isinstance(s.get("passes"), dict) else None)
    blk["delivered"] = _delivered(b.mygs, bl.psi_N, psi_pad, bl.j_BS,
                                  np.asarray(bl.j_phi, dtype=float))
    blk["delivered"]["j_phi_requested"] = np.asarray(bl.j_phi,
                                                     float).tolist()
    blk["delivered"]["l_i_target"] = _f(getattr(bl, "l_i_target", None))
    blk["delivered"]["Ip_target"] = _f(getattr(bl, "Ip_target", None))
    return blk


def _failed_block(e, t0):
    from bouquet.jbs_loop import JBSNotConverged, jsonable
    blk = dict(status="raised", wall_s=time.perf_counter() - t0,
               exception=f"{type(e).__name__}: {str(e)[:2000]}",
               traceback=traceback.format_exc()[-4000:])
    if isinstance(e, JBSNotConverged):
        blk.update(converged=False, n_passes=e.record.get("n_passes"),
                   loop=_loop_summary(e.record),
                   record=jsonable(e.record))
    return blk


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _child(part, gate, outdir, n_draws, seed):
    _oft_on_path()
    import bouquet
    import bouquet as bq
    import numpy as np
    _harness.assert_bouquet_is_repo_local()
    print("bouquet imported from:", bouquet.__file__, flush=True)
    tag = f"{part}__gate_{'on' if gate else 'off'}"
    work = os.path.join(outdir, "work", tag)
    os.makedirs(work, exist_ok=True)
    out_path = os.path.join(outdir, "parts", tag + ".json")
    res = dict(part=part, gate=bool(gate), started=time.time())
    log = _LoopLog()
    t0 = time.perf_counter()

    if part == "recon":
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                                   header=os.path.join(work, "rec"),
                                   n_draws=1, reconstruction_engine="legacy")
        src = b.config.source
    else:
        b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                                 header=os.path.join(work, "imas"),
                                 nthreads=1, reconstruction_engine="legacy")
        src = b.config.source
    g = b.config.generation
    g.jbs_self_consistent = True
    g.jbs_gate_current_residual = False
    psi_pad = float(getattr(src, "psi_pad", 1e-3))
    b.setup_solver()
    res["settings"] = dict(
        jbs_max_passes=int(g.jbs_max_passes),
        jbs_max_passes_draw=int(g.jbs_max_passes_draw),
        jbs_rtol_j=float(g.jbs_rtol_j), jbs_rtol_Ip=float(g.jbs_rtol_Ip),
        jbs_tol_li=float(g.jbs_tol_li), jbs_tol_q0=float(g.jbs_tol_q0),
        jbs_relax=float(g.jbs_relax),
        jbs_relax_current=float(g.jbs_relax_current),
        jbs_loop_on_fail=str(g.jbs_loop_on_fail), psi_pad=psi_pad)

    if part == "imas_ohmic_structured_mse":
        # the chords: synthetic, off the field of the structured baseline
        # built with the gate OFF in BOTH processes, so ON and OFF fit the
        # same data; the MSE baseline below is then built with the setting
        from bouquet.mse import mse_equilibrium_orientation, mse_field_at
        for k_, v_ in IMAS_PARTS[part].items():
            setattr(g, k_, v_)
        t1 = time.perf_counter()
        try:
            b.prepare_baseline()
        except Exception as e:
            res["chord_reference"] = _failed_block(e, t1)
            res["baseline"] = dict(status="not run: the gate-OFF structured "
                                          "baseline the chords come from "
                                          "failed")
            _write(out_path, res)
            return
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
                          A1=[1.0] * 8, A2=[1.0] * 8, A3=[0.0] * 8,
                          A4=[0.0] * 8, ip_sign=float(_or["ip"]),
                          bt_sign=float(_or["bt"]))
        g.structured_mse_required = True
        res["chord_reference"] = dict(
            status="delivered (gate OFF)", tgamma=[float(v) for v in tg],
            wall_s=time.perf_counter() - t1)
        log.entries.clear()

    if part in IMAS_PARTS:
        for k_, v_ in IMAS_PARTS[part].items():
            setattr(g, k_, v_)
    g.jbs_gate_current_residual = bool(gate)
    t1 = time.perf_counter()
    try:
        bl = b.prepare_baseline()
        res["baseline"] = _baseline_block(b, bl, psi_pad, t1)
        ok = True
    except Exception as e:
        res["baseline"] = _failed_block(e, t1)
        ok = False
    res["baseline"]["loop_calls"] = list(log.entries)
    _write(out_path, res)

    if part == "recon" and n_draws > 0:
        if not ok:
            res["draws"] = dict(status="not run: the baseline failed")
            _write(out_path, res)
            return
        res["draws"] = _draw_batch(b, g, psi_pad, n_draws, seed, log, work)
        _write(out_path, res)
        try:
            with open(os.path.join(outdir, "logs", tag + "_generation.log"),
                      "w") as fh:
                fh.write(b.generation_log or "")
        except Exception:
            pass
    res["wall_s"] = time.perf_counter() - t0
    _write(out_path, res)


def _draw_batch(b, g, psi_pad, n_draws, seed, log, work):
    import numpy as np
    import bouquet.TokaMaker_interface as TMI
    from bouquet.jbs_loop import jsonable
    g.seed = int(seed)
    stored = {}
    t_start = {}
    orig_store = TMI.store_equilibrium

    def _fp(*arrs):
        h = hashlib.sha1()
        for a in arrs:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=float))
                     .tobytes())
        return h.hexdigest()[:16]

    def _store(header, count, eqdsk_filepath, psi_N, j_phi, j_BS,
               j_inductive, n_e, T_e, n_i, T_i, *a, **kw):
        try:
            d = _delivered(b.mygs, psi_N, psi_pad, j_BS, j_phi)
            d["j_phi_stored"] = np.asarray(j_phi, dtype=float).tolist()
            jl = kw.get("jbs_loop")
            stored[int(count)] = dict(
                delivered=d,
                sample_fingerprint=_fp(n_e, T_e, n_i, T_i),
                in_spec=kw.get("in_spec"),
                jbs_loop_block={k: jsonable(v) for k, v in (jl or {}).items()
                                if k != "final"},
                jbs_loop_final=_loop_summary((jl or {}).get("final")))
        except Exception as e:
            stored[int(count)] = dict(capture_error=f"{type(e).__name__}: "
                                                    f"{e}")
        return orig_store(header, count, eqdsk_filepath, psi_N, j_phi, j_BS,
                          j_inductive, n_e, T_e, n_i, T_i, *a, **kw)

    def _progress(count):
        log.draw = int(count)
        t_start[int(count)] = time.perf_counter()

    TMI.store_equilibrium = _store
    t0 = time.perf_counter()
    err = None
    try:
        b.generate(n=int(n_draws), progress_callback=_progress)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:2000]}"
    finally:
        TMI.store_equilibrium = orig_store
    rejections = {}
    for r in (getattr(b, "draw_rejections", None) or []):
        rejections.setdefault(int(r.get("draw", -1)), []).append(
            jsonable(r))
    starts = sorted(t_start)
    draws = []
    for i in range(int(n_draws)):
        calls = log.for_draw(i)
        passes = sum(int((c["summary"] or {}).get("n_passes") or 0)
                     for c in calls)
        ent = dict(draw=i, archived=i in stored,
                   rejections=rejections.get(i, []),
                   passes_used=passes, n_loop_calls=len(calls),
                   loop_calls=calls)
        if i in t_start:
            nxt = [t_start[j] for j in starts if j > i]
            ent["wall_s"] = (nxt[0] if nxt else time.perf_counter()) \
                - t_start[i]
        if i in stored:
            ent.update(stored[i])
            ent["outcome"] = "archived"
        elif rejections.get(i):
            ent["outcome"] = "rejected"
            ent["reason"] = "; ".join(f"{r.get('reason')} "
                                      f"({r.get('stage')})"
                                      for r in rejections[i])
        else:
            ent["outcome"] = ("not reached" if i not in t_start else
                              "neither archived nor rejection-logged")
        draws.append(ent)
    return dict(n_draws=int(n_draws), seed=int(seed), error=err,
                wall_s=time.perf_counter() - t0, draws=draws)


# ===========================================================================
#  parent side (no solver): run the children, then compare ON with OFF
# ===========================================================================
def _num(d, k):
    return None if not isinstance(d, dict) else _f(d.get(k))


def _profile_diff(on, off, w=None, x=None):
    import numpy as np
    a = np.asarray(on, dtype=float)
    b = np.asarray(off, dtype=float)
    if a.shape != b.shape or not a.size:
        return None
    d = a - b
    out = dict(max_abs=float(np.max(np.abs(d))),
               max_rel=float(np.max(np.abs(d)) / max(np.max(np.abs(b)),
                                                       1e-300)),
               rms_rel=float(np.sqrt(np.mean(d * d))
                             / max(np.sqrt(np.mean(b * b)), 1e-300)))
    if w is not None and x is not None:
        from bouquet.jbs_loop import weighted_norm
        wn = weighted_norm(b, w, x)
        out["weighted_rms_rel"] = (weighted_norm(d, w, x) / wn
                                   if wn > 0 else None)
    return out


def _delivered_diff(on, off):
    if not (isinstance(on, dict) and isinstance(off, dict)):
        return None
    out = {}
    for k in ("l_i", "l_i_std", "q0", "q95", "Ip", "I_BS", "I_BS_over_Ip",
              "I_phi_achieved"):
        a, b = _num(on, k), _num(off, k)
        out["d_" + k] = None if (a is None or b is None) else a - b
    w, x = off.get("w"), off.get("psi_N")
    for k in ("j_phi_achieved", "j_phi_requested", "j_BS", "j_phi_stored"):
        if k in on and k in off:
            out[k] = _profile_diff(on[k], off[k], w, x)
    return out


def _brief(side):
    """The headline numbers of one setting of one case."""
    if not isinstance(side, dict):
        return None
    bl = side.get("baseline") or {}
    lp = bl.get("loop") or {}
    d = bl.get("delivered") or {}
    return dict(status=bl.get("status"), converged=bl.get("converged"),
                n_passes=bl.get("n_passes"),
                stop_reason=lp.get("stop_reason"),
                final=lp.get("final"),
                last_criterion_met=(lp.get("timeline") or {}).get(
                    "last_met"),
                met_since_pass=(lp.get("timeline") or {}).get(
                    "met_since_pass"),
                l_i=d.get("l_i"), q0=d.get("q0"), q95=d.get("q95"),
                Ip=d.get("Ip"), I_BS=d.get("I_BS"),
                exception=bl.get("exception"), wall_s=bl.get("wall_s"))


def _compare(parts, outdir):
    summary = dict(cases={}, draws=None)
    for part in parts:
        sides = {}
        for gate in (False, True):
            p = os.path.join(outdir, "parts",
                             f"{part}__gate_{'on' if gate else 'off'}.json")
            sides[gate] = json.load(open(p)) if os.path.isfile(p) else None
        off, on = sides[False], sides[True]
        case = dict(gate_off=_brief(off), gate_on=_brief(on))
        bo = ((off or {}).get("baseline") or {}).get("delivered")
        bn = ((on or {}).get("baseline") or {}).get("delivered")
        case["on_minus_off"] = _delivered_diff(bn, bo)
        summary["cases"][part] = case
        if part == "recon":
            dr = {}
            for gate in (False, True):
                blk = (sides[gate] or {}).get("draws") or {}
                dr["gate_" + ("on" if gate else "off")] = [
                    dict(draw=e.get("draw"), outcome=e.get("outcome"),
                         reason=e.get("reason"),
                         passes_used=e.get("passes_used"),
                         wall_s=e.get("wall_s"),
                         sample_fingerprint=e.get("sample_fingerprint"),
                         jbs_converged=(e.get("jbs_loop_block") or {}).get(
                             "converged"),
                         final=(e.get("jbs_loop_final") or {}).get("final"),
                         last_criterion_met=((e.get("jbs_loop_final") or {})
                                             .get("timeline") or {})
                         .get("last_met"),
                         **{k: (e.get("delivered") or {}).get(k)
                            for k in ("l_i", "q0", "q95", "Ip", "I_BS")})
                    for e in (blk.get("draws") or [])]
            pairs = []
            offs = {e["draw"]: e for e in
                    ((sides[False] or {}).get("draws") or {}).get("draws")
                    or []}
            for e in (((sides[True] or {}).get("draws") or {}).get("draws")
                      or []):
                o = offs.get(e["draw"])
                if o is None:
                    continue
                pairs.append(dict(
                    draw=e["draw"],
                    same_sample=(e.get("sample_fingerprint") is not None and
                                 e.get("sample_fingerprint")
                                 == o.get("sample_fingerprint")),
                    outcome_off=o.get("outcome"), outcome_on=e.get("outcome"),
                    on_minus_off=_delivered_diff(e.get("delivered"),
                                                 o.get("delivered"))))
            dr["on_minus_off"] = pairs
            summary["draws"] = dr
    return summary


def _run_children(parts, outdir, n_draws, seed, jobs, timeout):
    env = _harness.subprocess_env(**_THREAD_ENV)
    queue = [(p, gate) for p in parts for gate in (False, True)]
    running, done = [], []
    while queue or running:
        while queue and len(running) < max(1, int(jobs)):
            part, gate = queue.pop(0)
            tag = f"{part}__gate_{'on' if gate else 'off'}"
            fh = open(os.path.join(outdir, "logs", tag + ".log"), "w")
            cmd = [sys.executable, os.path.abspath(__file__), outdir,
                   "--child", part, "on" if gate else "off",
                   "--n-draws", str(n_draws), "--seed", str(seed)]
            print(f"[measure] start {tag}", flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT)
            running.append((tag, proc, fh, time.time()))
        time.sleep(2.0)
        for item in list(running):
            tag, proc, fh, t0 = item
            rc = proc.poll()
            if rc is None and timeout and time.time() - t0 > timeout:
                proc.kill()
                rc = "killed (child timeout)"
                proc.wait()
            if rc is not None:
                fh.close()
                running.remove(item)
                done.append(dict(child=tag, returncode=rc,
                                 wall_s=time.time() - t0))
                print(f"[measure] done  {tag}: rc={rc} "
                      f"({time.time() - t0:.0f} s)", flush=True)
    return done


def _repo_commit():
    try:
        r = subprocess.run(["git", "-C", _harness.REPO_ROOT, "rev-parse",
                            "--short=12", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--parts", default=",".join(ALL_PARTS),
                    help="comma-separated subset of: " + ", ".join(ALL_PARTS))
    ap.add_argument("--n-draws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=1,
                    help="children run at once (each single-threaded)")
    ap.add_argument("--child-timeout", type=float, default=None,
                    help="seconds before a child is killed (default: none)")
    ap.add_argument("--child", nargs=2, metavar=("PART", "GATE"),
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        part, gate = a.child
        _child(part, gate == "on", os.path.abspath(a.outdir), a.n_draws,
               a.seed)
        return 0

    import bouquet
    _harness.assert_bouquet_is_repo_local()
    print("bouquet imported from:", bouquet.__file__, flush=True)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_PARTS]
    if bad:
        ap.error(f"unknown part(s) {bad}; choose from {ALL_PARTS}")
    outdir = os.path.abspath(a.outdir)
    for sub in ("parts", "logs", "work"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    t0 = time.time()
    children = _run_children(parts, outdir, a.n_draws, a.seed, a.jobs,
                             a.child_timeout)
    summary = _compare(parts, outdir)
    summary["meta"] = dict(
        repo_commit=_repo_commit(), parts=parts, n_draws=a.n_draws,
        seed=a.seed, jobs=a.jobs, children=children,
        wall_s=time.time() - t0,
        note=("gate = GenerationConfig.jbs_gate_current_residual; every "
              "other setting is the shipped default.  on_minus_off = gate "
              "ON minus gate OFF of the delivered equilibrium; profile "
              "differences are max|d|/max|off|, rms(d)/rms(off) and the "
              "loop's current-weighted norm on the OFF weights."))
    _write(os.path.join(outdir, "current_gate_measurement.json"), summary)
    print(f"[measure] wrote {os.path.join(outdir, 'current_gate_measurement.json')}"
          f" ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
