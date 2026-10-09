#!/usr/bin/env python
"""Small summary of tests/test_reversed_ip_solver.py's own probe output (its basetemp):
per mode and orientation, the largest difference of the solved baseline to the
(Ip+, B0+) orientation.  usage: python extract_revip.py <basetemp>/revip0 <out.json>"""
import json, os, sys
import numpy as np
R, OUT = sys.argv[1], sys.argv[2]
ORS = ["ip+_b0+", "ip+_b0-", "ip-_b0+", "ip-_b0-"]
modes = sorted(f[:-4] for f in os.listdir(os.path.join(R, ORS[0])) if f.endswith(".npz"))
res = dict(modes=modes, orientations=ORS, rows={}, profiles={})
for m in modes:
    ref = np.load(os.path.join(R, ORS[0], m + ".npz"))
    rj = json.load(open(os.path.join(R, ORS[0], m + ".json")))
    res["profiles"][m] = dict(psi_N=ref["bl_psi_N"].tolist(), j_phi=ref["bl_j_phi"].tolist(),
                              q=np.abs(ref["eq_q"]).tolist())
    for o in ORS:
        z = np.load(os.path.join(R, o, m + ".npz"))
        j = json.load(open(os.path.join(R, o, m + ".json")))
        row = dict(l_i_target=j.get("l_i_target"), Ip_target=j.get("Ip_target"),
                   source_current_sign=j.get("source_current_sign"),
                   source_b0_sign=j.get("source_b0_sign"))
        for k in ("bl_j_phi", "bl_j_BS", "bl_j_inductive", "eq_q", "eq_psi", "eq_globals"):
            a, b = np.asarray(z[k], float), np.asarray(ref[k], float)
            row[k] = dict(max_abs_diff=float(np.max(np.abs(a - b))),
                          max_abs_diff_of_magnitudes=float(np.max(np.abs(np.abs(a) - np.abs(b)))),
                          scale=float(np.max(np.abs(b))))
        row["dl_i_target"] = (None if row["l_i_target"] is None
                              else float(row["l_i_target"]) - float(rj.get("l_i_target")))
        res["rows"][f"{m}|{o}"] = row
with open(OUT, "w") as fh:
    json.dump(res, fh)
print("wrote", OUT, len(res["rows"]))
