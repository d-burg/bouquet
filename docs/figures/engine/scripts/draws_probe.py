"""Seeded engine-draw batch through Bouquet.generate() on a synthetic example,
at a chosen inductive-shape sigma (UncertaintyConfig.jphi_scalar_sigma).

usage: draws_probe.py recon|imas SIGMA OUTDIR [N_DRAWS SEED]
Everything else is the probe batch of tests/probes/measure_engine.py
(part draws_recon / draws_imas: reconstruction_engine="unified", nthreads=1,
UncertaintyConfig defaults apart from jphi_scalar_sigma).  The shipped
notebooks set jphi_scalar_sigma = 0.05 (their other uncertainty values equal
the defaults).  Writes OUTDIR/engine_draws_<src>.json (measure_engine's draw
block + the rejections in full + the engine-draw cap events) and the archive
OUTDIR/draws_<src>.h5.
"""
import json
import os
import sys
import time
import traceback

src, sigma, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 12345
#: the repository root: $BQ_REPO, else four levels up from this file
#: (docs/figures/engine/scripts/)
W = os.environ.get("BQ_REPO") or os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4))
EX = os.path.join(W, "examples", "D3D-like")
sys.path.insert(0, os.path.join(W, "tests"))
sys.path.insert(0, os.path.join(W, "tests", "probes"))
os.makedirs(outdir, exist_ok=True)
out = dict(source=src, jphi_scalar_sigma=sigma, n_equils=n, seed=seed)
t0 = time.time()
try:
    import bouquet as bq
    from bouquet.jbs_loop import jsonable
    import measure_engine as ME
    import OpenFUSIONToolkit as _oft
    hdr = os.path.join(outdir, f"draws_{src}")
    if src == "recon":
        b = bq.Bouquet.from_geqdsk(
            os.path.join(EX, "D3Dlike_Hmode_baseline.geqdsk"),
            profiles=os.path.join(EX, "D3Dlike_Hmode_baseline.peqdsk"),
            mesh=os.path.join(EX, "DIIID_mesh.h5"), nthreads=1, n_draws=1,
            header=hdr, reconstruction_engine="unified")
    else:
        b = bq.Bouquet.from_imas(
            os.path.join(EX, "D3Dlike_baseline_omas.json"),
            mesh=os.path.join(EX, "DIIID_mesh.h5"), time=2.3043, nthreads=1,
            n_draws=1, header=hdr, reconstruction_engine="unified")
    b.uncertainty.jphi_scalar_sigma = sigma
    out["uncertainty"] = {k: getattr(b.uncertainty, k) for k in (
        "ne_scalar_sigma", "te_scalar_sigma", "ni_scalar_sigma",
        "ti_scalar_sigma", "jphi_scalar_sigma", "zeff_scalar_sigma",
        "n_ls", "t_ls", "j_ls")}
    b.setup_solver()
    bl = b.prepare_baseline()
    out["t_baseline_s"] = time.time() - t0
    out["l_i_target"] = float(bl.l_i_target)
    out["draws"] = ME._draws(b, n, seed)
    out["draws"]["rejections"] = jsonable(
        [dict(r) for r in (b.draw_rejections or [])])
    out["cap_events"] = jsonable(
        [dict(e) for e in (getattr(b, "engine_draw_cap_events", None) or [])])
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()[-4000:]
out["wall_s"] = time.time() - t0
with open(os.path.join(outdir, f"engine_draws_{src}.json"), "w") as fh:
    json.dump(jsonable(out) if "jsonable" in dir() else out, fh, indent=1,
              default=str)
print("DONE", src, sigma, "error" in out, flush=True)
