#!/usr/bin/env python3
"""Solver-free: the boundary-metric selection on the test suite's synthetic flux map."""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(SRC, "tests"))
import test_lcfs_axis_selection as T  # the test's own flux map and contouring
from bouquet.utils import select_closed_lcfs

W = dict(blue="#0072B2", verm="#D55E00", green="#009E73", black="#000000", grey="#999999")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.linestyle": ":", "lines.linewidth": 2, "savefig.dpi": 130})
OUTSIDE = T.PSI_X * (1.0 + 2e-3)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
txt = []
for a, (lev, name) in zip(ax, ((T.INSIDE, "level just inside the saddle"), (OUTSIDE, "level just outside the saddle"))):
    segs = T._segments(lev)
    for s in segs:
        a.plot(s[:, 0], s[:, 1], color=W["grey"], lw=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = select_closed_lcfs(segs, context="figure")
        new = select_closed_lcfs(segs, context="figure", axis=T.AXIS)
    tb = T._true_boundary()
    a.plot(tb[:, 0], tb[:, 1], color=W["black"], lw=1, ls=":", label="analytic separatrix")
    if old is not None:
        a.plot(np.asarray(old)[:, 0], np.asarray(old)[:, 1], color=W["verm"], ls="--", label="old rule: longest closed segment")
    if new is not None:
        a.plot(np.asarray(new)[:, 0], np.asarray(new)[:, 1], color=W["blue"], label="new rule: curve around the axis")
    a.plot(*T.AXIS, "+", color=W["black"], ms=10, label="magnetic axis")
    a.set_aspect("equal"); a.set_xlabel("R [m]"); a.text(0.02, 0.02, name, transform=a.transAxes, fontsize=8.5)
    txt.append(f"{name}: old rule {'far loop' if (old is not None and not T._near_plasma(np.asarray(old))) else 'plasma boundary'}, "
               f"new rule {'plasma boundary' if (new is not None and T._near_plasma(np.asarray(new))) else 'NOT the boundary'}")
ax[0].set_ylabel("Z [m]"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "boundary_metric_axis_selection.png"))
cap_path = os.path.join(OUT, "captions.json")
caps = json.load(open(cap_path)) if os.path.exists(cap_path) else {}
caps["boundary_metric_axis_selection"] = (
    "Toy model (solver-free): the synthetic flux map of tests/test_lcfs_axis_selection.py -- a plasma well "
    "with an X-point, divertor legs and a far dip crossing the same flux level -- contoured as the measurement "
    "does, and the boundary chosen by bouquet.utils.select_closed_lcfs without (old rule) and with (new rule, "
    "the change) the magnetic axis. " + "; ".join(txt) + ".")
json.dump(caps, open(cap_path, "w"), indent=1)
print("\n".join(txt))
