#!/usr/bin/env python
"""Probe (live solver): the engine's MSE row on the repository's synthetic
OMAS example, fixed finite-difference Jacobian ("fd_chord") against the
Broyden-updated one ("fd_broyden").

Synthetic chords are built exactly as ``tests/test_jbs_loop_solver.py``
builds them for its MSE composition test: eight midplane chords from
R0 + 0.05 m to R0 + 0.55 m, tan(gamma) read off the engine baseline's own
field (no MSE row), scaled by 1.03, sigma 0.004, A1 = A2 = 1, A3 = A4 = 0,
E_r-corrected.  Orientation: the engine takes the SOURCE's declared
orientation and refuses a block that states the solved equilibrium's own
(positive-frame) signs when they disagree with the source (the synthetic IDS
declares B0 < 0) -- so the block states no signs (completed from the source)
and tan(gamma) is expressed in the source frame: tan(gamma)_source =
tan(gamma)_solver * sign(Ip_src * Ip_eq) * sign(B0_src * Bt_eq).  One interpreter
per step (``OFT_env`` is a per-process singleton)::

    python probe_engine_mse_synthetic.py OUTDIR --step chords
    python probe_engine_mse_synthetic.py OUTDIR --step fit --jac fd_chord
    python probe_engine_mse_synthetic.py OUTDIR --step fit --jac fd_broyden

Synthetic data only; single thread; no network.  Run with the repository
checkout on PYTHONPATH (and BQ_REPO pointing at it).
"""
import argparse
import json
import os
import sys
import time

#: the repository root: $BQ_REPO, else four levels up from this file
#: (docs/figures/engine/scripts/)
REPO = os.environ.get("BQ_REPO") or os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, os.path.join(REPO, "tests", "probes"))
import _harness  # noqa: E402
_harness.ensure_repo_on_syspath()
from measure_engine import _MESH, _OMAS, _TIME, oft_importable  # noqa: E402

FACTOR, SIGMA, N = 1.03, 0.004, 8


def _bouquet(header):
    """The engine configuration, selected IN the factory call: ``from_imas``
    then leaves its legacy-path workflow settings (``isolate_edge_jBS``,
    ``perturb_jind_in_anchor``, ``jBS_baseline_mode``) at their defaults.
    Building the legacy configuration and flipping the engine afterwards is
    refused by the engine's settings validation (those fields are never read
    by the engine and are refused when not at their defaults)."""
    import bouquet as bq
    return bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                                nthreads=1, header=header,
                                reconstruction_engine="unified")


def _configure_fit(b, c, jac):
    """The ``--step fit`` settings: the MSE row on the chords ``c`` (the
    ``--step chords`` output) with Jacobian scheme ``jac``."""
    g = b.config.generation
    g.engine_rows = ["Ip", "l_i", "mse"]
    g.engine_mse_jacobian = jac
    g.mse_data = dict(R=c["R"], Z=c["Z"], tgamma=c["tg_data"],
                      sigma=c["sigma"], weight=[1.0] * N, A1=[1.0] * N,
                      A2=[1.0] * N, A3=[0.0] * N, A4=[0.0] * N,
                      er_corrected=True)
    return b


def _tg_at(mygs, R, Z):
    import numpy as np
    from bouquet.mse import mse_field_at
    B, found = mse_field_at(mygs, np.asarray(R), np.asarray(Z))
    return (B[:, 2] / B[:, 1]).tolist(), np.asarray(found).tolist()


def _stats(b):
    st = b.mygs.get_stats(li_normalization="iter")
    return {k: float(st[k]) for k in ("l_i", "q_0", "q_95", "beta_n")
            if k in st}


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--step", choices=("chords", "fit"), required=True)
    ap.add_argument("--jac", choices=("fd_chord", "fd_broyden"),
                    default="fd_chord")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    assert oft_importable(), "OpenFUSIONToolkit is not importable"
    from bouquet.jbs_loop import jsonable
    cpath = os.path.join(a.outdir, "mse_chords.json")
    if a.step == "chords":
        from bouquet.mse import mse_equilibrium_orientation, mse_field_at
        b = _bouquet(os.path.join(a.outdir, "mse_base"))
        b.setup_solver()
        t0 = time.perf_counter()
        bl = b.prepare_baseline()
        R0 = float(np.asarray(b.mygs.o_point, dtype=float)[0])
        R = np.linspace(R0 + 0.05, R0 + 0.55, N)
        Z = np.zeros_like(R)
        Bf, found = mse_field_at(b.mygs, R, Z)
        assert np.all(found), "a synthetic chord is off the mesh"
        tg = Bf[:, 2] / Bf[:, 1]
        o = mse_equilibrium_orientation(Bf, R, Z, b.mygs.o_point)
        with open(_OMAS) as fh:
            dd = json.load(fh)
        eq = dd["equilibrium"]
        it = int(np.argmin(np.abs(np.asarray(eq["time"]) - _TIME)))
        ip_src = float(np.sign(eq["time_slice"][it]["global_quantities"]["ip"]))
        b0 = eq["vacuum_toroidal_field"]["b0"]
        b0_src = float(np.sign(b0[it] if isinstance(b0, list) else b0))
        flip = ip_src * float(o["ip"]) * b0_src * float(o["bt"])
        out = dict(R=R.tolist(), Z=Z.tolist(), tg_baseline=(flip * tg).tolist(),
                   tg_data=(flip * FACTOR * tg).tolist(), sigma=[SIGMA] * N,
                   factor=FACTOR, frame_flip=flip, ip_sign_eq=float(o["ip"]),
                   bt_sign_eq=float(o["bt"]), ip_sign_src=ip_src,
                   bt_sign_src=b0_src, R_axis=R0,
                   baseline_stats=_stats(b),
                   baseline_converged=bool(bl.engine["converged"]),
                   wall_s=time.perf_counter() - t0,
                   note=("synthetic chords: the engine baseline's own field "
                         "(no MSE row) scaled by 1.03"))
        with open(cpath, "w") as fh:
            json.dump(out, fh, indent=1)
        return
    with open(cpath) as fh:
        c = json.load(fh)
    b = _configure_fit(_bouquet(os.path.join(a.outdir, f"mse_{a.jac}")),
                       c, a.jac)
    b.setup_solver()
    t0 = time.perf_counter()
    out = dict(jac=a.jac)
    try:
        bl = b.prepare_baseline()
        out["wall_s"] = time.perf_counter() - t0
        out["engine"] = bl.engine
        tg, out["found"] = _tg_at(b.mygs, c["R"], c["Z"])
        out["tg_delivered"] = [c["frame_flip"] * v for v in tg]
        out["stats"] = _stats(b)
    except Exception as e:  # recorded, never lost
        import traceback
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        out["wall_s"] = time.perf_counter() - t0
    with open(os.path.join(a.outdir, f"mse_fit_{a.jac}.json"), "w") as fh:
        json.dump(jsonable(out), fh, default=float)


if __name__ == "__main__":
    main()
