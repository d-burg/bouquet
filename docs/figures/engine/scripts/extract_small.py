#!/usr/bin/env python
"""Extract SMALL JSON from the measurement payload (run where the payload is):
draw-archive profiles (baseline + every draw, with flags and scalars) and
the PRES/PPRIME arrays of the g-file-frame probe's g-files.  Nothing else
leaves the shared home.

    python extract_small.py OUT
"""
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
SMALL = os.path.join(OUT, "small")
os.makedirs(SMALL, exist_ok=True)


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def draws(h5, tag):
    import bouquet as bq
    ar = bq.BouquetArchive(h5)
    sv = ar.scan(None) if len(ar.scan_keys) == 1 else ar.scan(ar.scan_keys[0])
    bl = sv.baseline
    out = dict(baseline={k: np.asarray(v, float).tolist()
                         for k, v in bl.items()
                         if isinstance(v, np.ndarray) and v.ndim == 1
                         and v.dtype.kind == "f" and v.size <= 2048},
               draws=[])
    try:          # the baseline g-file stored in the archive
        import h5py
        from bouquet.schema import find_bytes_dataset
        from bouquet.utils import read_eqdsk_from_bytes
        from bouquet.io.geqdsk import read_geqdsk
        with h5py.File(ar.path, "r") as hf:
            g = hf["scan/0/_baseline"]
            raw = bytes(g[find_bytes_dataset(g, "eqdsk")][()])
        q = np.abs(np.asarray(read_eqdsk_from_bytes(raw, read_geqdsk).qpsi, float))
        out["baseline"]["q"] = q.tolist()
        out["baseline"]["psi_q"] = np.linspace(0.0, 1.0, q.size).tolist()
    except Exception as e:
        out["baseline_q_error"] = str(e)[:200]
    for dv in sv.all:
        pr = {k: np.asarray(v, float).tolist() for k, v in dv.profiles.items()
              if np.asarray(v).ndim == 1 and np.asarray(v).size <= 2048}
        try:      # q is not archived as a profile: read it off the draw's own g-file
            eq = dv.equilibrium()
            q = np.abs(np.asarray(eq.qpsi, float))
            pr["q"] = q.tolist()
            pr["psi_q"] = np.linspace(0.0, 1.0, q.size).tolist()
        except Exception as e:
            pr["q_error"] = str(e)[:200]
        at = {k: _f(v) for k, v in dv.attrs.items()
              if np.ndim(v) == 0 and _f(v) is not None}
        out["draws"].append(dict(count=int(dv.count), profiles=pr, attrs=at,
                                 flags=dv.flags))
    with open(os.path.join(SMALL, f"draws_{tag}_profiles.json"), "w") as fh:
        json.dump(out, fh, default=float)


def frames():
    from bouquet.io.geqdsk import _read_geqdsk
    res = {}
    for f in sorted(glob.glob(os.path.join(OUT, "gfile_frame", "*.geqdsk"))):
        r = _read_geqdsk(f)
        res[os.path.basename(f)] = dict(PRES=np.asarray(r["PRES"]).tolist(),
                                        PPRIME=np.asarray(r["PPRIME"]).tolist())
    with open(os.path.join(SMALL, "gfile_frame_pres.json"), "w") as fh:
        json.dump(res, fh)


ONLY = sys.argv[2] if len(sys.argv) > 2 else None
for tag in ("recon", "imas"):
    h = os.path.join(OUT, "draws", f"draws_{tag}.h5")
    if os.path.exists(h):
        try:
            draws(h, tag)
        except Exception as e:
            print("draw extraction failed", tag, e)
if ONLY != "draws":
    frames()
