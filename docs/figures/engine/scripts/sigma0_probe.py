"""verify_sigma0_consistency (the true draw route) on a synthetic example of the
repository, live solver, one thread.  usage: sigma0_probe.py recon|imas unified|legacy OUTDIR
Writes OUTDIR/sigma0_<src>_<engine>.json (read by extract_draw_data.py)."""
import json, os, sys, time, traceback
src, eng, outdir = sys.argv[1:4]
#: the repository root: $BQ_REPO, else four levels up from this file
#: (docs/figures/engine/scripts/)
W = os.environ.get("BQ_REPO") or os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4))
EX = os.path.join(W, "examples", "D3D-like")
os.makedirs(outdir, exist_ok=True)
tag = f"{src}_{eng}"
out = dict(source=src, engine=eng, stages={})
t0 = time.time()
try:
    import bouquet as bq
    from bouquet.jbs_loop import jsonable
    import OpenFUSIONToolkit as _oft
    hdr = os.path.join(outdir, f"s0_{tag}")
    if src == "recon":
        b = bq.Bouquet.from_geqdsk(os.path.join(EX, "D3Dlike_Hmode_baseline.geqdsk"),
                                   profiles=os.path.join(EX, "D3Dlike_Hmode_baseline.peqdsk"),
                                   mesh=os.path.join(EX, "DIIID_mesh.h5"), nthreads=1,
                                   n_draws=1, header=hdr, reconstruction_engine=eng)
    else:
        b = bq.Bouquet.from_imas(os.path.join(EX, "D3Dlike_baseline_omas.json"),
                                 mesh=os.path.join(EX, "DIIID_mesh.h5"), time=2.3043,
                                 nthreads=1, n_draws=1, header=hdr,
                                 reconstruction_engine=eng)
    b.setup_solver()
    bl = b.prepare_baseline()
    out["t_baseline_s"] = time.time() - t0
    out["l_i_target"] = float(bl.l_i_target)
    t1 = time.time()
    rec = b.verify_sigma0_consistency()
    out["t_check_s"] = time.time() - t1
    out["record"] = jsonable(rec)
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()[-4000:]
with open(os.path.join(outdir, f"sigma0_{tag}.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("DONE", tag, "error" in out, flush=True)
