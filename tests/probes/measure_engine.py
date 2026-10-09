#!/usr/bin/env python
"""Measure the unified reconstruction engine with the live solver -- a
MEASUREMENT (pytest does not collect this file); ``tests/test_engine_solver.py``
runs it and asserts on its JSON.

For each synthetic example of the repository (``examples/D3D-like``: the
g-file + p-file, and the OMAS modelling source at t = 2.3043 s) the baseline
is built with ``reconstruction_engine="unified"``, each part in its own
interpreter (``OFT_env`` is a per-process singleton).  Per part it records:

* converged or not, passes per phase, the delivered-state checks (loop
  residuals, every row), the full engine record (per pass: coefficients,
  l_i, q, request - achieved, the uniform Ip factor);
* the GS solve count per stage and the wall time;
* the DISTANCE-TO-INPUT table, in the units of the three-state comparison
  report (g-file: l_i(3) matched and l_i(1) free, q on axis and at
  psi_N = 0.02, q95, the q-profile max/rms over psi_N 0.05-0.95, the
  achieved <j_phi> against the input core (psi_N < 0.8) / edge (>= 0.8)
  max/rms in % of the input's peak, beta_N, W_MHD, beta_p, the LCFS
  distance rms/max, requested - achieved core/edge; modelling source: the
  same against the IDS's own li_3, q profile, j_tor and boundary);
* the delivered state RE-SOLVED once from itself with its stored request:
  the change in l_i, q0, q95 and the achieved current.

Parts: ``recon``, ``imas`` (rows Ip + l_i), ``imas_q0`` (+ the q0 row, which
the source's sawtooth gate admits), ``recon_dc`` (the delivery correction
ON).  Every reconstruction part also runs the ENGINE DRAW at zero
perturbation (stage ``sigma0``: ``verify_sigma0_consistency`` under the
engine -- the request identity, ``r_j``/``r_I`` against the
reconstruction's bootstrap, ``dl_i``, ``dq0`` at its labelled radius,
``dq95``, the passes and solves).

Draw mode (Stage 3): part ``draws_recon`` (and ``draws_imas``) builds the
engine baseline and runs ``generate()`` with ``n_equils = --draws`` and
``seed = --seed`` (defaults 6 and 12345, the legacy batch it is compared
with), writing per draw: archived or rejected (with its
``DRAW_REJECTION_REASONS`` code), in spec, the loop's passes, the Ip
amplitude, l_i(3)/l_i(1)/beta_N/q0/q95, the flux range and their changes,
the post-hoc verdicts, and solves / passes / wall time by
stage (anchor, loop, homotopy, post_homotopy, filters, archive).  Usage::

    python tests/probes/measure_engine.py OUTDIR [--parts recon,imas,...]
    python tests/probes/measure_engine.py OUTDIR --draws 6 --seed 12345

Writes ``OUTDIR/engine_<part>.json`` per part (always, with the error in
place of the numbers when a stage raises) and ``OUTDIR/engine_measurement
.json`` (all parts).  Single-threaded (``nthreads=1``, every BLAS/OpenMP
count 1); no network.  OpenFUSIONToolkit is found as the solver tests find
it (``OFT_PYTHONPATH``, else the sibling checkout's
``build_release/python``).  ``BQ_ENGINE_PROBE_OUT=<dir>`` also copies every
part's JSON there.  ``BQ_ENGINE_PROBE_GC='<json object>'`` sets further
GenerationConfig fields on every part (after the part's own; recorded in
the part's ``settings``), e.g. ``'{"engine_draw_bootstrap_refresh": true}'``
to run the solver tests with a draw setting on -- no assertion changes.

The pressure handed to the solver (``bouquet.edge_pressure``).  The distance
table reports beta_N, beta_p and W_MHD BOTH ways (``pressure_frames``): the
solver's frame (its own pressure, zero at the boundary) against the input's
``p - p_edge`` quantities, and the full frame (``p_sep`` added back) against
the input's full-pressure quantities; a g-file's edge pressure is read from
its own ``PRES`` array.  Stage ``edge`` records the solved ``P'``, the
pressure, the achieved ``<j_phi>`` and its pressure-driven part on the
engine grid, the edge current over psi_N 0.95-1 and the pressure-driven
current at the boundary; stage ``solves`` every GS solve of the part
(iterations, seconds; opt-in with ``BQ_ENGINE_PROBE_SOLVELOG=1``, off the
solver is untouched).  Part ``recon_legacy`` is the g-file example on the
LEGACY path (``reconstruction_engine="legacy"``).
``BQ_ENGINE_PROBE_PSEP_ADD=<Pa>`` (g-file parts) is a CONSTRUCTED TEST, not
an example of the repository: a constant fast pressure of that many pascals
is added on the whole profile (``fixed_components.p_fast``), which leaves
the kinetics and the bootstrap untouched and gives the solve pressure a
separatrix value of that size; the input reference of the full frame is
then the g-file's pressure plus the same constant.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
_TIME = 2.3043

#: part -> (source, GenerationConfig fields set on top of the engine switch)
PARTS = {
    "recon": ("recon", dict()),
    "imas": ("imas", dict()),
    "imas_q0": ("imas", dict(engine_rows=["Ip", "l_i", "q0"])),
    "recon_dc": ("recon", dict(engine_delivery_correction=True)),
    # the two-scalar Ip + l_i closure: the named preset, and the q95
    # study's route to the same state (the settings' preset patched to the
    # constant two-scalar basis, rows Ip + l_i) for the identity check
    "recon_2s": ("recon", dict(engine_preset="two_scalar_li")),
    "recon_2s_patched": ("recon", dict(_patched_two_scalar=True)),
    # the g-file example on the LEGACY path (bouquet.edge_pressure study)
    "recon_legacy": ("recon", dict(_legacy=True)),
    # Stage 3: a seeded engine-draw batch (--draws / --seed)
    "draws_recon": ("recon", dict()),
    "draws_imas": ("imas", dict()),
}
#: the draw batch the legacy measurement used (6 draws, seed 12345)
DEFAULT_DRAWS, DEFAULT_SEED = 6, 12345


def oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
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


# ---------------------------------------------------------------------------
#  the child: one part in one interpreter
# ---------------------------------------------------------------------------
def _pct_stats(d, ref_peak, mask):
    import numpy as np
    v = 100.0 * np.asarray(d, float)[mask] / ref_peak
    return dict(max=float(np.max(np.abs(v))),
                rms=float(np.sqrt(np.mean(v ** 2))))


def _q_profile(mygs, psi):
    import numpy as np
    pq = np.ascontiguousarray(np.clip(np.asarray(psi, float), 1e-3,
                                      1.0 - 1e-3))
    return np.asarray(mygs.get_q(psi=pq)[1], dtype=float)


#: the grid every current comparison of the probe is made on
COMPARISON_GRID = ("linearly interpolated from the solver's uniform sampling "
                   "linspace(psi_pad, 1 - psi_pad, n) onto the psi_N of the "
                   "profile it is compared with")


def _registered(A_uniform, psi, psi_pad):
    from bouquet.TokaMaker_interface import register_corrective_output
    return register_corrective_output(A_uniform, psi, psi_pad)


def _grid_note(A_uniform, psi, psi_pad):
    """What the registration changed: the grid offset and the size of the
    index-for-index artefact it removes (% of the achieved peak)."""
    import numpy as np
    from bouquet.TokaMaker_interface import corrective_output_grid
    A_u = np.asarray(A_uniform, float)
    psi = np.asarray(psi, float)
    u = corrective_output_grid(A_u.size, psi_pad)
    d = A_u - _registered(A_u, psi, psi_pad)
    pk = float(np.max(np.abs(A_u))) or 1.0
    return dict(
        compared_on="the psi_N of the compared profile",
        sampled_on="linspace(psi_pad, 1 - psi_pad, n)", n=int(A_u.size),
        psi_pad=float(psi_pad),
        max_grid_offset=float(np.max(np.abs(u - psi))),
        index_for_index_artefact_pct_of_peak=dict(
            rms=float(100.0 * np.sqrt(np.mean(d ** 2)) / pk),
            max=float(100.0 * np.max(np.abs(d)) / pk)))


def _safe(fn):
    try:
        return fn()
    except Exception as e:                        # recorded, never fatal
        return dict(error=f"{type(e).__name__}: {e}")


def _psep_add():
    v = os.environ.get("BQ_ENGINE_PROBE_PSEP_ADD")
    return 0.0 if not v else float(v)


def _frames_gfile(eq, st_i, bl):
    """beta_N / beta_p / W_MHD both ways against the g-file's own numbers
    (:func:`bouquet.edge_pressure.pressure_frames` /
    ``input_pressure_frames``): the solver's frame against the input's
    ``p - p_edge``, the full frame against the input's full pressure.  A
    constructed constant pressure (``BQ_ENGINE_PROBE_PSEP_ADD``) is part of
    the input's FULL frame only (the input then is ``PRES + constant``)."""
    import numpy as np
    from bouquet.edge_pressure import input_pressure_frames, pressure_frames
    ep = getattr(bl, "edge_pressure", None) or {}
    fr = pressure_frames(st_i, float(ep.get("p_sep_applied", 0.0)))
    add = _psep_add()
    pres = np.asarray(eq.pres, float)
    V = float(eq.volume_integral(np.ones_like(pres))[-1])
    pvol = float(eq.volume_integral(pres)[-1])
    betas = {k: float(v) for k, v in eq.betas.items()
             if k in ("beta_n", "beta_p", "beta_t")}
    # betas are linear in int p dV: the constructed constant scales them
    f_add = (pvol + add * V) / pvol
    inp = input_pressure_frames(V, pvol + add * V, float(pres[-1]) + add,
                                betas={k: v * f_add
                                       for k, v in betas.items()})

    def _row(tok, ref):
        return dict(input=float(ref), engine=float(tok),
                    rel_pct=(100.0 * (float(tok) - float(ref)) / float(ref)
                             if ref else float("nan")))
    out = dict(p_sep=float(ep.get("p_sep", float("nan"))),
               p_sep_applied=float(ep.get("p_sep_applied", 0.0)),
               p_axis=float(ep.get("p_axis", float("nan"))),
               pax_target=float(ep.get("pax_target", float("nan"))),
               settings=dict(edge_pprime_pin=ep.get("edge_pprime_pin"),
                             separatrix_pressure=ep.get(
                                 "separatrix_pressure")),
               p_edge_input=float(pres[-1]) + add,
               constructed_constant_Pa=add, volume_engine=fr["volume"],
               volume_input=V, frames={})
    for k in ("solver", "full"):
        out["frames"][k] = dict(
            beta_n=_row(fr[k].get("beta_n", float("nan")),
                        inp[k].get("beta_n", float("nan"))),
            beta_p=_row(fr[k].get("beta_pol", float("nan")) / 100.0,
                        inp[k].get("beta_p", float("nan"))),
            W_MHD_MJ=_row(fr[k]["W_MHD"] / 1e6, inp[k]["W_MHD"] / 1e6))
    return out


def _distance_gfile(b, bl, psi_pad):
    """The three-state comparison report's table for the engine state."""
    import numpy as np
    from bouquet.engine import _lcfs_deviation_mm
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.TokaMaker_interface import _corrective_output_jphi
    mygs = b.mygs
    eq = read_geqdsk(b.config.source.geqdsk_path, cocos=b.config.source.cocos)
    psi = np.asarray(eq.psi_N, float)
    jin = np.abs(np.asarray(eq.j_tor_averaged_direct, float))
    peak = float(np.max(jin))
    A_u = np.asarray(_corrective_output_jphi(mygs, psi, psi_pad), float)
    # the achieved current is SAMPLED on the solver's uniform grid; every
    # comparison below is on the input's psi_N (see _registered)
    A = _registered(A_u, psi, psi_pad)
    R = np.asarray(bl.j_phi, float)
    st_i = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    st_s = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="std")
    qin = np.abs(np.asarray(eq.qpsi, float))
    qp = np.abs(_q_profile(mygs, psi))
    m = (psi >= 0.05) & (psi <= 0.95)
    rel = qp[m] / qin[m] - 1.0
    core, edge = psi < 0.8, psi >= 0.8
    bnd = np.column_stack([eq.boundary_R, eq.boundary_Z])
    rms, mx = _lcfs_deviation_mm(mygs, bnd)
    betas = eq.betas
    W_in = 1.5 * float(eq.volume_integral(eq.pres)[-1]) / 1e6
    from bouquet.physics import SOLVER_Q0_PSI_N
    q002 = float(np.abs(_q_profile(mygs, [SOLVER_Q0_PSI_N, 0.5]))[0])
    q_in_002 = float(np.interp(SOLVER_Q0_PSI_N, psi, qin))
    return dict(
        li3_matched=dict(input=float(eq.li["li(2)"]),
                         engine=float(st_i["l_i"]),
                         delta=float(st_i["l_i"]) - float(eq.li["li(2)"])),
        li1_free=dict(input=float(eq.li["li(1)_EFIT"]),
                      engine=float(st_s["l_i"]),
                      delta=float(st_s["l_i"]) - float(eq.li["li(1)_EFIT"])),
        q_axis=dict(input=float(qin[0]), engine=float(qp[0]),
                    delta=float(qp[0] - qin[0]), input_psi_N=0.0,
                    engine_psi_N=1e-3,
                    note=("NOT like radii: the input's qpsi[0] is on axis, "
                          "the engine's sample is the psi_N = 1e-3 clip "
                          "(kept for the comparison report's row)")),
        q_002=dict(input=q_in_002, engine=q002, delta=q002 - q_in_002,
                   psi_N=float(SOLVER_Q0_PSI_N),
                   code_q0=float(st_i.get("q_0", float("nan"))),
                   note=("like radii (physics.SOLVER_Q0_PSI_N, get_stats' "
                         "q_0 radius), as reconstruction_metrics' q0")),
        q95=dict(input=float(np.interp(0.95, psi, qin)),
                 engine=float(st_i["q_95"]),
                 delta=float(st_i["q_95"]) - float(np.interp(0.95, psi,
                                                             qin))),
        q_profile_rel_pct=dict(max=100.0 * float(np.max(np.abs(rel))),
                               rms=100.0 * float(np.sqrt(np.mean(rel ** 2)))),
        jphi_vs_input_pct_of_peak=dict(core=_pct_stats(A - jin, peak, core),
                                       edge=_pct_stats(A - jin, peak, edge),
                                       edge_argmax_psiN=float(psi[edge][
                                           int(np.argmax(np.abs(
                                               (A - jin)[edge])))])),
        requested_minus_achieved_pct_of_peak=dict(
            core=_pct_stats(R - A, peak, core),
            edge=_pct_stats(R - A, peak, edge)),
        beta_n=dict(input=float(betas.get("beta_n", float("nan"))),
                    engine=float(st_i.get("beta_n", float("nan")))),
        W_MHD_MJ=dict(input=W_in,
                      engine=float(st_i.get("W_MHD", float("nan"))) / 1e6),
        beta_p=dict(input=float(betas.get("beta_p", float("nan"))),
                    engine=float(st_i.get("beta_pol", float("nan"))) / 100.0),
        lcfs_mm=dict(rms=rms, max=mx),
        # beta / W both ways (the rows above are the solver's own stats)
        pressure_frames=_safe(lambda: _frames_gfile(eq, st_i, bl)),
        achieved_form="TokaMaker_interface._corrective_output_jphi (the "
                      "comparison report's 'achieved'), " + COMPARISON_GRID,
        comparison_grid=_grid_note(A_u, psi, psi_pad),
        input_form="|g-file j_tor_averaged_direct| (bouquet's reader)")


def _frames_ids(mygs, bl, p1, psn, psi, st_i, psi_pad):
    """W_MHD both ways against the source's own pressure (recorded with the
    error in place of the numbers when it cannot be formed)."""
    import numpy as np
    try:
        # W_MHD both ways.  The source carries no volume profile, so its
        # pressure is integrated on the SOLVED geometry (dV/dpsi of the engine
        # state): full = 1.5 int p_in dV, solver frame = 1.5 int (p_in -
        # p_in(1)) dV.
        from scipy.integrate import trapezoid
        from bouquet.edge_pressure import pressure_frames
        from bouquet.utils import fsa_current_geometry
        ep = getattr(bl, "edge_pressure", None) or {}
        fr = pressure_frames(st_i, float(ep.get("p_sep_applied", 0.0)))
        geo = fsa_current_geometry(mygs.copy_eq(), psi, psi_pad=psi_pad,
                                   want_pprime=False)
        dV = np.asarray(geo["dV_dpsi"], float) * float(geo["dpsi_dpsiN"])
        pin = np.interp(psi, psn, np.asarray(p1["pressure"], float))
        W_full = 1.5 * float(trapezoid(pin * dV, psi)) / 1e6
        W_sol = 1.5 * float(trapezoid((pin - pin[-1]) * dV, psi)) / 1e6

        def _row(tok, ref):
            return dict(input=float(ref), engine=float(tok),
                        rel_pct=100.0 * (float(tok) - float(ref)) / float(ref))
        frames = dict(
            p_sep=float(ep.get("p_sep", float("nan"))),
            p_sep_applied=float(ep.get("p_sep_applied", 0.0)),
            p_axis=float(ep.get("p_axis", float("nan"))),
            pax_target=float(ep.get("pax_target", float("nan"))),
            settings=dict(edge_pprime_pin=ep.get("edge_pprime_pin"),
                          separatrix_pressure=ep.get("separatrix_pressure")),
            p_edge_input=float(pin[-1]), p_axis_input=float(pin[0]),
            volume_engine=fr["volume"],
            frames=dict(
                solver=dict(W_MHD_MJ=_row(fr["solver"]["W_MHD"] / 1e6, W_sol),
                            beta_n=dict(engine=fr["solver"].get("beta_n"))),
                full=dict(W_MHD_MJ=_row(fr["full"]["W_MHD"] / 1e6, W_full),
                          beta_n=dict(engine=fr["full"].get("beta_n")))),
            note=("input W: the source pressure integrated on the solved "
                  "geometry (the source has no volume profile); the source "
                  "carries no beta"))
        return frames
    except Exception as e:                        # recorded, never fatal
        return dict(error=f"{type(e).__name__}: {e}")


def _distance_ids(b, bl, psi_pad):
    import numpy as np
    from bouquet.engine import _lcfs_deviation_mm
    from bouquet.io.imas import _nearest_index, read_imas_geometry
    from bouquet.TokaMaker_interface import _corrective_output_jphi
    mygs = b.mygs
    with open(b.config.source.ids_path) as fh:
        dd = json.load(fh)
    eq = dd["equilibrium"]
    # the SOURCE's slice time (never the synthetic example's constant: the
    # harness calls this on real dds); None -> the first slice, as the reader
    t_src = getattr(b.config.source, "time", None)
    t_use = float(eq["time"][0]) if t_src is None else float(t_src)
    ie = _nearest_index(eq["time"], t_use, "equilibrium")
    p1 = eq["time_slice"][ie]["profiles_1d"]
    gq = eq["time_slice"][ie]["global_quantities"]
    psq = np.asarray(p1["psi"], float)
    psn = (psq - psq[0]) / (psq[-1] - psq[0])
    psi = np.asarray(bl.psi_N, float)
    qin = np.abs(np.interp(psi, psn, np.asarray(p1["q"], float)))
    # the source's current in the frame the solve is in: the reader's own
    # orientation factor (a reversed-Ip source stores j_tor negative), else
    # the sign of the slice's own Ip
    sgn = getattr(bl, "source_current_sign", None)
    if sgn is None:
        ip = gq.get("ip")
        sgn = -1.0 if (ip is not None and float(ip) < 0.0) else 1.0
    sgn = float(sgn)
    jin = sgn * np.interp(psi, psn, np.asarray(p1["j_tor"], float))
    peak = float(np.max(np.abs(jin)))
    A_u = np.asarray(_corrective_output_jphi(mygs, psi, psi_pad), float)
    # SAMPLED on the solver's uniform grid, compared on the baseline's psi_N
    # (an IDS grid is not uniform: index for index this compared the current
    # at one radius with the source's at another)
    A = _registered(A_u, psi, psi_pad)
    R = np.asarray(bl.j_phi, float)
    st_i = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    qp = np.abs(_q_profile(mygs, psi))
    m = (psi >= 0.05) & (psi <= 0.95)
    rel = qp[m] / qin[m] - 1.0
    core, edge = psi < 0.8, psi >= 0.8
    _F0, bnd = read_imas_geometry(b.config.source)
    rms, mx = _lcfs_deviation_mm(mygs, bnd)
    frames = _frames_ids(mygs, bl, p1, psn, psi, st_i, psi_pad)
    return dict(
        pressure_frames=frames,
        slice=dict(time_requested=t_src, time_used=t_use, index=int(ie),
                   time_of_slice=float(eq["time"][ie])),
        li3=dict(input=float(gq["li_3"]), engine=float(st_i["l_i"]),
                 delta=float(st_i["l_i"]) - float(gq["li_3"])),
        q_axis=dict(input=float(qin[0]), engine=float(qp[0]),
                    delta=float(qp[0] - qin[0])),
        q95=dict(input=float(np.interp(0.95, psi, qin)),
                 engine=float(st_i["q_95"])),
        q_profile_rel_pct=dict(max=100.0 * float(np.max(np.abs(rel))),
                               rms=100.0 * float(np.sqrt(np.mean(rel ** 2)))),
        jphi_vs_equilibrium_jtor_pct_of_peak=dict(
            core=_pct_stats(A - jin, peak, core),
            edge=_pct_stats(A - jin, peak, edge),
            note=("the synthetic OMAS j_tor was written as a jphi-linterp "
                  "input (<j_phi>), so this is like for like on THIS file "
                  "only (jphi convention experiment, claim (d))")),
        requested_minus_achieved_pct_of_peak=dict(
            core=_pct_stats(R - A, peak, core),
            edge=_pct_stats(R - A, peak, edge)),
        lcfs_mm=dict(rms=rms, max=mx),
        source_current_sign=sgn,
        achieved_form="TokaMaker_interface._corrective_output_jphi, "
                      + COMPARISON_GRID,
        comparison_grid=_grid_note(A_u, psi, psi_pad))


def _resolve_self(b, bl, psi_pad):
    """Re-solve the delivered state's own stored request, once, from it."""
    import numpy as np
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend
    mygs = b.mygs
    st = bl.engine["state"]
    snap = mygs.copy_eq()
    psi = np.asarray(bl.psi_N, float)
    before = TokaMakerBackend(mygs, SimpleNamespace(
        psi_N=psi, pressure=np.asarray(st["pressure"], float),
        Ip=float(st["Ip"]), kinetics=_kin(bl, psi)), psi_pad=psi_pad,
        # the reconstruction's own edge-pressure settings
        edge_pressure=(bl.engine.get("settings") or {}).get("edge_pressure"))
    m0 = before.measure(final=True)
    before.solve(np.asarray(st["request"], float), n_passes=1)
    m1 = before.measure(final=True)
    mygs.replace_eq(source_eq=snap)
    a0, a1 = np.asarray(m0["achieved"]), np.asarray(m1["achieved"])
    return dict(dl_i=float(m1["li"] - m0["li"]),
                dq0=float(m1["q_row"] - m0["q_row"]),
                dq95=float(m1["stats"].get("q_95", float("nan"))
                           - m0["stats"].get("q_95", float("nan"))),
                dj_rel=float(np.linalg.norm(a1 - a0) / np.linalg.norm(a0)),
                dj_max_rel=float(np.max(np.abs(a1 - a0))
                                 / np.max(np.abs(a0))))


class _SolveLog:
    """Every GS solve of the part: iterations, seconds, ok.  ``mygs.solve``
    is wrapped on the instance (as ``DrawSolveGuard`` does), so the solves
    inside the solver's own helpers are seen too.  The solve is called with
    ``return_its=True`` to read its iteration count; the caller gets what
    it asked for."""

    def __init__(self, mygs, enabled=True):
        self.rows = []
        self.marks = {}
        self.enabled = bool(enabled)
        if not self.enabled:
            return
        orig = mygs.solve

        def solve(*a, **k):
            want = k.get("return_its", a[1] if len(a) > 1 else False)
            t0 = time.perf_counter()
            try:
                out = orig(*a[:1], **{**k, "return_its": True})
            except Exception as exc:
                self.rows.append(dict(ok=False, its=None,
                                      seconds=time.perf_counter() - t0,
                                      error=str(exc).strip()[:200]))
                raise
            self.rows.append(dict(ok=True, its=int(out[1]),
                                  seconds=time.perf_counter() - t0))
            return out if want else out[0]
        mygs.solve = solve

    def mark(self, name):
        self.marks[name] = len(self.rows)

    def summary(self, lo=0, hi=None):
        import numpy as np
        if not self.enabled:
            return None
        rows = self.rows[lo:hi]
        its = [r["its"] for r in rows if r["ok"]]
        return dict(
            n_solves=len(rows), n_failed=sum(not r["ok"] for r in rows),
            its_total=int(np.sum(its)) if its else 0,
            its_mean=float(np.mean(its)) if its else None,
            its_median=float(np.median(its)) if its else None,
            its_max=int(np.max(its)) if its else None,
            seconds_total=float(sum(r["seconds"] for r in rows)),
            its=[r["its"] for r in rows])


def _solve_pressure(bl):
    import numpy as np
    eng = getattr(bl, "engine", None)
    if eng:
        return np.asarray(eng["state"]["pressure"], float)
    return np.asarray(bl.recon["pres_tokamaker"], float)


def _edge(b, bl, psi_pad):
    """The solved P', pressure, achieved <j_phi> and its pressure-driven
    part on the baseline grid, with the edge numbers of the study."""
    import numpy as np
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend, pressure_term
    from bouquet.utils import pchip_derivative
    mygs = b.mygs
    psi = np.asarray(bl.psi_N, float)
    p_in = _solve_pressure(bl)
    ep = getattr(bl, "edge_pressure", None) or {}
    p_add = float(ep.get("p_sep_applied", 0.0))
    snap = mygs.copy_eq()
    try:
        be = TokaMakerBackend(mygs, SimpleNamespace(
            psi_N=psi, pressure=p_in, Ip=float(bl.Ip_target),
            kinetics=_kin(bl, psi)), psi_pad=psi_pad)
        m = be.measure(final=True)
    finally:
        mygs.replace_eq(source_eq=snap)
    g = m["geom"]
    pq = np.asarray(g["psi_q"], float)
    prof = mygs.get_profiles(psi=pq.copy())
    F, Fp = np.asarray(prof[1], float), np.asarray(prof[2], float)
    P, Pp = np.asarray(prof[3], float), np.asarray(prof[4], float)
    mu0 = 4.0e-7 * np.pi
    A = np.asarray(m["achieved"], float)
    sgn = 1.0 if float(np.median(A)) >= 0.0 else -1.0
    j_p = sgn * np.asarray(pressure_term(g), float)
    j_ppR = sgn * np.asarray(g["R_avg"], float) * Pp
    j_ffp = sgn * np.asarray(g["inv_R"], float) * F * Fp / mu0
    A = sgn * A
    dpsi = float(g["dpsi_dpsiN"])
    pp_in = pchip_derivative(psi, p_in) / dpsi
    em = psi >= 0.95

    def _at(y, x0):
        return float(np.interp(x0, psi, np.asarray(y, float)))
    q = np.asarray(mygs.get_q(psi=pq.copy())[1], float)
    return dict(
        psi_N=psi.tolist(), psi_sampled=pq.tolist(),
        pprime_solver=Pp.tolist(), pressure_solver=P.tolist(),
        pressure_full=(P + p_add).tolist(), pressure_input=p_in.tolist(),
        pprime_input_abs=np.abs(pp_in).tolist(),
        jphi_achieved=A.tolist(), j_pressure_driven=j_p.tolist(),
        j_pprime_term=j_ppR.tolist(), j_ffprime_term=j_ffp.tolist(),
        q=np.abs(q).tolist(),
        flux_range=dpsi,
        summary=dict(
            jphi_edge_mean_095_1=float(np.mean(A[em])),
            jphi_at_095=_at(A, 0.95), jphi_at_099=_at(A, 0.99),
            jphi_last_node=float(A[-1]),
            j_pressure_driven_last_node=float(j_p[-1]),
            j_pressure_driven_at_099=_at(j_p, 0.99),
            j_pprime_term_last_node=float(j_ppR[-1]),
            j_ffprime_term_last_node=float(j_ffp[-1]),
            pprime_solver_last_node=float(Pp[-1]),
            pprime_solver_at_099=_at(Pp, 0.99),
            pprime_input_last_node=float(pp_in[-1]),
            pprime_ratio_solver_over_input_mid=float(np.median(
                (np.abs(Pp) / np.abs(pp_in))[(psi > 0.2) & (psi < 0.8)])),
            pressure_solver_axis=float(P[0]),
            pressure_solver_last_node=float(P[-1]),
            pressure_full_last_node=float(P[-1] + p_add),
            pressure_input_last_node=float(p_in[-1]),
            last_node_psi_sampled=float(pq[-1]),
            l_i=float(m["li"]), q_row=float(m["q_row"]),
            q_row_psi_N=float(pq[0]),
            q0_stats=m["stats"].get("q_0"), q95=m["stats"].get("q_95"),
            note=("positive-current frame; the last node is sampled at "
                  "psi_N = 1 - psi_pad; j_pressure_driven = p'(<R> - "
                  "F^2<1/R>/<B^2>), j_pprime_term = <R> p', j_ffprime_term "
                  "= <1/R> FF'/mu0 (their sum is the achieved <j_phi>)")))


def _kin(bl, psi):
    import numpy as np
    from bouquet.utils import pchip_interp
    k = lambda a: pchip_interp(bl.psi_N_kinetic, a, psi)  # noqa: E731
    return dict(ne=k(bl.ne), te=k(bl.te), ni=k(bl.ni), ti=k(bl.ti),
                zeff=np.clip(k(bl.Zeff), 1.0, None))


def _sigma0(b):
    """The engine draw at zero perturbation (verify_sigma0_consistency)."""
    t0 = time.perf_counter()
    v = b.verify_sigma0_consistency()
    out = {k: v.get(k) for k in (
        "passed", "request_bit_identical", "request_max_abs_diff",
        "loop_converged", "n_passes", "r_j", "r_I", "dl_i", "dq0",
        "dq0_psi_N", "dq0_stats", "dq0_stats_psi_N", "dq95", "amplitude",
        "tolerances", "criterion")}
    rec = v.get("record") or {}
    out["cost"] = rec.get("cost")
    out["solves"] = rec.get("solves")
    out["wall_s"] = float(time.perf_counter() - t0)
    return out


def _draw_row(i, d):
    e = d.get("engine") or {}
    lp = e.get("loop") or {}
    return dict(
        index=int(i), archived=True, in_spec=bool(d.get("in_spec")),
        time_s=d.get("time"), loop_passes=lp.get("n_passes"),
        loop_converged=lp.get("converged"),
        loop_r_j=lp.get("r_j"), loop_r_I=lp.get("r_I"),
        loop_last_criterion_met=lp.get("last_criterion_met"),
        loop_bootstrap_refresh=lp.get("bootstrap_refresh"),
        amplitude=(e.get("amplitude") or {}).get("final"),
        delivered=e.get("delivered"), archived_state=e.get("archived"),
        reference=e.get("reference"), deltas=e.get("deltas"),
        post_hoc=e.get("post_hoc"),
        homotopy=e.get("homotopy"), post_homotopy=e.get("post_homotopy"),
        cost=e.get("cost"), inputs=e.get("inputs"),
        identity=e.get("identity"))


def _gfile_pressure_check(eq, psi_in, p_in, psi_pad):
    """Is a delivered g-file's pressure consistent with its own P' and with
    the pressure the solve was handed?  ``PRES`` against the inward
    integral of ``PPRIME`` from the file's last point (the gradient check),
    and against the input pressure at the edge and over the profile."""
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    pres = np.asarray(eq.pres, float)
    pp = np.asarray(eq.pprime, float)
    n = pres.size
    psi = np.linspace(float(eq.psi_axis), float(eq.psi_boundary), n)
    x = np.linspace(0.0, 1.0, n)
    # P(psi) = P(edge) - int_psi^edge P' dpsi, with the file's own sign of
    # P' against its psi (taken from the bulk, so either COCOS reads alike)
    cum = cumulative_trapezoid(pp, psi, initial=0.0)
    integ = cum - cum[-1]
    sgn = 1.0 if np.dot(integ, pres - pres[-1]) >= 0.0 else -1.0
    rebuilt = pres[-1] + sgn * integ
    pin = np.interp(x, np.asarray(psi_in, float), np.asarray(p_in, float))
    return dict(
        n=int(n), pres_axis=float(pres[0]), pres_edge=float(pres[-1]),
        pprime_edge=float(pp[-1]),
        input_axis=float(p_in[0]), input_edge=float(p_in[-1]),
        input_at_1_minus_pad=float(np.interp(1.0 - psi_pad, psi_in, p_in)),
        gradient_check_max_over_axis=float(
            np.max(np.abs(rebuilt - pres)) / abs(pres[0])),
        pres_minus_input_max_over_axis=float(
            np.max(np.abs(pres - pin)) / abs(p_in[0])),
        pres_minus_input_edge_over_axis=float(
            (pres[-1] - float(np.interp(1.0 - psi_pad, psi_in, p_in)))
            / abs(p_in[0])),
        note=("the file's psi_N grid is taken as uniform on [0, 1]; its "
              "last point is the 1 - psi_pad surface"))


def _delivered_gfiles(b, psi_pad):
    """The delivered g-files of the archive (the baseline re-save and every
    draw) checked with :func:`_gfile_pressure_check`, beside each one's
    archived edge-pressure record."""
    import h5py
    import numpy as np
    import bouquet as bq
    from bouquet.edge_pressure import load_record
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.schema import find_bytes_dataset
    from bouquet.utils import _baseline_group_path, read_eqdsk_from_bytes
    ar = bq.BouquetArchive(b)
    out = dict(draws=[])
    for sv in ar:
        with h5py.File(ar.path, "r") as hf:
            g = hf[_baseline_group_path(sv.scan_key)]
            nm = find_bytes_dataset(g, "eqdsk")
            raw = bytes(g[nm][()]) if nm is not None else None
        blp = sv.baseline
        if raw is not None:
            eq = read_eqdsk_from_bytes(raw, read_geqdsk)
            out["baseline"] = dict(
                record=load_record(ar.path, scan_key=sv.scan_key),
                **_gfile_pressure_check(eq, blp["psi_N"], blp["pressure"],
                                        psi_pad))
        for dv in sv.all:
            pr = dv.profiles
            out["draws"].append(dict(
                index=int(dv.count),
                record=load_record(ar.path, count=dv.count,
                                   scan_key=sv.scan_key),
                **_gfile_pressure_check(dv.equilibrium(), pr["psi_N"],
                                        pr["pressure"], psi_pad)))
    return out


def _draws(b, n, seed):
    """A seeded engine-draw batch through generate()."""
    g = b.config.generation
    g.n_equils = int(n)
    g.seed = int(seed)
    t0 = time.perf_counter()
    diags = b.generate() or []
    wall = float(time.perf_counter() - t0)
    rows = [_draw_row(i, d) for i, d in enumerate(diags)]
    rej = [dict(r) for r in (b.draw_rejections or [])]
    stages = ("anchor", "loop", "homotopy", "post_homotopy", "filters",
              "archive")
    tot = {s: dict(solves=0, passes=0, wall_s=0.0) for s in stages}
    for r in rows:
        for s in stages:
            c = (r.get("cost") or {}).get(s) or {}
            for k in ("solves", "passes", "wall_s"):
                tot[s][k] += c.get(k) or 0
    return dict(
        n_equils=int(n), seed=int(seed), wall_s=wall,
        attempts=len(rows) + len(rej), archived=len(rows),
        rejected=len(rej), in_spec=int(sum(r["in_spec"] for r in rows)),
        per_draw=rows, rejections=rej, stage_totals=tot,
        solve_failures=list(getattr(b, "solve_failures", []) or []),
        per_draw_wall_s=[r["time_s"] for r in rows],
        note=("per-draw wall time 'time_s' is generate_bouquet's per-draw "
              "clock (draw + homotopy + filters); 'cost' splits it by "
              "stage including the archive write"))


def child(part, outdir, draws=None, seed=None):
    import bouquet as bq
    _harness.assert_bouquet_is_repo_local()
    src, extra = PARTS[part]
    extra = dict(extra)
    _gc_env = os.environ.get("BQ_ENGINE_PROBE_GC")
    if _gc_env:
        extra.update(json.loads(_gc_env))
    legacy = bool(extra.pop("_legacy", False))
    out = dict(part=part, source=src, settings=dict(extra), stages={},
               engine=("legacy" if legacy else "unified"),
               constructed_constant_Pa=(_psep_add() if src == "recon"
                                        else 0.0))
    if extra.pop("_patched_two_scalar", False):
        # the q95 attribution study's monkeypatch, reproduced exactly: the
        # settings dict's preset -> the constant two-scalar basis (rows
        # stay Ip + l_i, both hard on the g-file); the config is unchanged
        from bouquet import engine as _E
        _orig_es = _E.engine_settings

        def _es(gc):
            s = dict(_orig_es(gc))
            s["preset"] = "sawtooth_two_scalar"
            return s
        _E.engine_settings = _es
    try:
        import OpenFUSIONToolkit as _oft
        out["oft_file"] = os.path.realpath(_oft.__file__)
    except Exception as e:                        # recorded, never fatal
        out["oft_file"] = f"unavailable: {e}"
    path = os.path.join(outdir, f"engine_{part}.json")

    def _stage(name, fn):
        try:
            out[name] = fn()
            out["stages"][name] = "ok"
        except Exception as e:                    # recorded, never lost
            out["stages"][name] = (f"{type(e).__name__}: {e}\n"
                                   + traceback.format_exc()[-3000:])

    try:
        _engine = "legacy" if legacy else "unified"
        if src == "recon":
            b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                       nthreads=1, n_draws=1,
                                       header=os.path.join(outdir, part),
                                       reconstruction_engine=_engine)
            psi_pad = float(b.config.source.psi_pad)
        else:
            b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME,
                                     n_draws=1, nthreads=1,
                                     header=os.path.join(outdir, part),
                                     reconstruction_engine=_engine)
            psi_pad = 1e-3
        g = b.config.generation
        for k, v in extra.items():
            setattr(g, k, v)
        if src == "recon" and _psep_add() != 0.0:
            # CONSTRUCTED TEST: a constant fast pressure on the whole
            # profile (kinetics and bootstrap untouched)
            import numpy as np
            fc = b.config.fixed_components
            fc.psi_N = np.linspace(0.0, 1.0, 129)
            fc.p_fast = np.full(129, _psep_add())
        b.setup_solver()
        # opt-in (BQ_ENGINE_PROBE_SOLVELOG=1): off, the solver is untouched
        slog = _SolveLog(b.mygs, enabled=(
            os.environ.get("BQ_ENGINE_PROBE_SOLVELOG", "0") == "1"))
        t0 = time.perf_counter()
        holder = {}

        def _build_legacy():
            bl = b.prepare_baseline()
            holder["bl"] = bl
            m = dict(bl.reconstruction_metrics or {})
            return dict(
                wall_s=float(time.perf_counter() - t0),
                converged=bool(m.get("converged")),
                verdict=m.get("verdict"),
                l_i_target=float(bl.l_i_target),
                metrics={k: v for k, v in m.items()
                         if isinstance(v, (int, float, str, bool, dict))},
                edge_pressure=getattr(bl, "edge_pressure", None))

        def _build():
            bl = b.prepare_baseline()
            holder["bl"] = bl
            rec = bl.engine
            return dict(
                wall_s=float(time.perf_counter() - t0),
                converged=bool(rec["converged"]),
                loop_converged=bool(rec["loop_converged"]),
                n_passes={ph["name"]: int(ph["record"]["n_passes"])
                          for ph in rec["phases"]},
                max_passes=int(rec["settings"]["loop"]["max_passes"]),
                solves=rec["solves"], delivered=rec["delivered"],
                l_i_target=float(bl.l_i_target),
                delivered_state={k: v for k, v in
                                 (bl.delivered_state or {}).items()
                                 if k != "j_phi_achieved"},
                engine_record=rec)
        _stage("build", _build_legacy if legacy else _build)
        slog.mark("build")
        out["solves_build"] = slog.summary(0, slog.marks["build"])
        bl = holder.get("bl")
        if bl is not None and part.startswith("draws_"):
            nd = DEFAULT_DRAWS if draws is None else int(draws)
            sd = DEFAULT_SEED if seed is None else int(seed)
            out["draw_mode"] = dict(draws=nd, seed=sd)
            _stage("draws", lambda: _draws(b, nd, sd))
            _stage("delivered_gfiles",
                   lambda: _delivered_gfiles(b, psi_pad))
        elif bl is not None:
            _stage("distance", lambda: (_distance_gfile if src == "recon"
                                        else _distance_ids)(b, bl, psi_pad))
            _stage("edge", lambda: _edge(b, bl, psi_pad))
            if not legacy:
                _stage("resolve_self",
                       lambda: _resolve_self(b, bl, psi_pad))
            slog.mark("pre_sigma0")
            _stage("sigma0", lambda: _sigma0(b))
            out["solves_sigma0"] = slog.summary(slog.marks["pre_sigma0"])
        out["solves_all"] = slog.summary()
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    finally:
        from bouquet.jbs_loop import jsonable
        with open(path, "w") as fh:
            json.dump(jsonable(out), fh, default=float)
        keep = os.environ.get("BQ_ENGINE_PROBE_OUT")
        if keep:
            os.makedirs(keep, exist_ok=True)
            shutil.copy(path, os.path.join(keep, os.path.basename(path)))
    return path


def run_part(part, outdir, timeout=None, draws=None, seed=None):
    """Run one part in its own interpreter; return its JSON (with ``_rc``)."""
    os.makedirs(outdir, exist_ok=True)
    extra = ([] if draws is None else ["--draws", str(int(draws))]) \
        + ([] if seed is None else ["--seed", str(int(seed))])
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), outdir, "--child", part]
        + extra,
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=timeout)
    p = os.path.join(outdir, f"engine_{part}.json")
    with open(os.path.join(outdir, f"engine_{part}.log"), "w") as fh:
        fh.write(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if not os.path.exists(p):
        return dict(part=part, stages={}, fatal=(
            f"no JSON (rc={proc.returncode}):\n{proc.stdout[-3000:]}\n"
            f"{proc.stderr[-3000:]}"), _rc=proc.returncode)
    with open(p) as fh:
        d = json.load(fh)
    d["_rc"] = proc.returncode
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-timeout", type=float, default=None)
    ap.add_argument("--draws", type=int, default=None,
                    help="draw mode: run the seeded engine-draw batch "
                         "(part draws_recon unless --parts names draws_*)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    oft_importable()
    if a.child:
        child(a.child, a.outdir, draws=a.draws, seed=a.seed)
        return 0
    parts = [p for p in a.parts.split(",") if p]
    default = a.parts == ",".join(PARTS)
    if a.draws is not None and default:
        parts = ["draws_recon"]        # --draws alone: the g-file batch
    elif default:
        parts = [p for p in parts if not p.startswith("draws_")]
    res = {}
    for part in parts:
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; known: {list(PARTS)}")
        res[part] = run_part(part, a.outdir, timeout=a.child_timeout,
                             draws=a.draws, seed=a.seed)
        print(f"[measure_engine] {part}: stages {res[part].get('stages')}",
              flush=True)
    with open(os.path.join(a.outdir, "engine_measurement.json"), "w") as fh:
        json.dump(res, fh, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
