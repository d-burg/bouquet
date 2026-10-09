"""Small JSON for the draw-ensemble, draw-cost, sigma=0 true-route and reversed-Ip
figures (run where the payload is; only small JSON is written).
usage: extract_draw_data.py S OUT_SMALL  (S: the directory run_draws_sigma0_revip.sh wrote)"""
import glob
import json
import os
import sys

import numpy as np

S, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def draws_profiles(h5, out):
    import bouquet as bq
    ar = bq.BouquetArchive(h5)
    sv = ar.scan(None) if len(ar.scan_keys) == 1 else ar.scan(ar.scan_keys[0])
    bl = sv.baseline
    res = dict(baseline={k: np.asarray(v, float).tolist()
                         for k, v in bl.items()
                         if isinstance(v, np.ndarray) and v.ndim == 1
                         and v.dtype.kind == "f" and v.size <= 2048},
               draws=[])
    try:
        import h5py
        from bouquet.schema import find_bytes_dataset
        from bouquet.utils import read_eqdsk_from_bytes
        from bouquet.io.geqdsk import read_geqdsk
        with h5py.File(ar.path, "r") as hf:
            g = hf["scan/0/_baseline"]
            raw = bytes(g[find_bytes_dataset(g, "eqdsk")][()])
        q = np.abs(np.asarray(read_eqdsk_from_bytes(raw, read_geqdsk).qpsi, float))
        res["baseline"]["q"] = q.tolist()
        res["baseline"]["psi_q"] = np.linspace(0.0, 1.0, q.size).tolist()
    except Exception as e:
        res["baseline_q_error"] = str(e)[:200]
    for dv in sv.all:
        pr = {k: np.asarray(v, float).tolist() for k, v in dv.profiles.items()
              if np.asarray(v).ndim == 1 and np.asarray(v).size <= 2048}
        try:
            eq = dv.equilibrium()
            q = np.abs(np.asarray(eq.qpsi, float))
            pr["q"] = q.tolist()
            pr["psi_q"] = np.linspace(0.0, 1.0, q.size).tolist()
        except Exception as e:
            pr["q_error"] = str(e)[:200]
        at = {k: _f(v) for k, v in dv.attrs.items()
              if np.ndim(v) == 0 and _f(v) is not None}
        res["draws"].append(dict(count=int(dv.count), profiles=pr, attrs=at,
                                 flags=dv.flags))
    with open(out, "w") as fh:
        json.dump(res, fh, default=float)


def slim_draw_meta(src, dst):
    """The draw block without the per-pass logs (kept: everything the yield
    table and the cost figure read)."""
    d = json.load(open(src))
    dr = d.get("draws") or {}
    for r in dr.get("per_draw", []):
        for k in ("delivered", "reference"):
            r.pop(k, None)
    json.dump(d, open(dst, "w"), default=str)


# ---- draws at both sigmas 
for sg in ("0.10", "0.05"):
    for s in ("recon", "imas"):
        base = os.path.join(S, f"draws_{sg}")
        j = os.path.join(base, f"engine_draws_{s}.json")
        if os.path.exists(j):
            slim_draw_meta(j, os.path.join(OUT, f"draws_{s}_{sg}_meta.json"))
        h = os.path.join(base, f"draws_{s}.h5")
        if os.path.exists(h):
            try:
                draws_profiles(h, os.path.join(OUT, f"draws_{s}_{sg}_profiles.json"))
            except Exception as e:
                print("profiles failed", s, sg, e)

# ---- sigma0 true route
for f in glob.glob(os.path.join(S, "sigma0", "sigma0_*.json")):
    d = json.load(open(f))
    json.dump(d, open(os.path.join(OUT, os.path.basename(f)), "w"), default=str)

# ---- reversed Ip: legacy (the test's own probe output) + the engine arm
rv = glob.glob(os.path.join(S, "suite", "tmp_revip", "*", "revip0"))
rv = rv or glob.glob(os.path.join(S, "suite", "tmp_revip", "revip0"))
if rv:
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "extract_revip.py"), rv[0],
                    os.path.join(OUT, "revip_identity.json")])
eng = {}
for f in glob.glob(os.path.join(S, "suite", "tmp_revip_engine", "**", "out_ip*_bt*.json"),
                   recursive=True):
    r = json.load(open(f))
    eng[os.path.basename(f)[4:-5]] = r
if eng:
    ref = eng.get("ip+1_bt+1")
    out = {}
    for o, r in sorted(eng.items()):
        row = dict(converged=r.get("converged"), delivered_ok=r.get("delivered_ok"),
                   l_i=r.get("l_i"), signs=r.get("signs"))
        if ref and "arrays" in r:
            for k, a in r["arrays"].items():
                b = np.asarray(ref["arrays"][k], float)
                a = np.asarray(a, float)
                row[k] = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) or 1.0))
            row["l_i_rel"] = abs(r["l_i"] - ref["l_i"]) / ref["l_i"]
        out[o] = row
    if ref:
        out["_profile"] = dict(j_phi=ref["arrays"]["j_phi"])
    json.dump(out, open(os.path.join(OUT, "revip_engine.json"), "w"), default=str)
print("extracted to", OUT)
