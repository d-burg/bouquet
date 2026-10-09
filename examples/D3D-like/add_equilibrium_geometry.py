"""Give D3Dlike_baseline_omas.json the equilibrium geometry FUSE writes.

The reader converts IMAS currents to TokaMaker jphi with each equilibrium
slice's f, gm1/gm5/gm8/gm9 and dpressure_dpsi (docs/current-conventions.md).
This takes them from D3Dlike_Hmode_baseline.geqdsk (same shape) for all three
slices, puts psi in Wb (COCOS 11 = 2*pi * the g-file's COCOS 1), and rebuilds
core_profiles.j_total from j_tor exactly (A5, A7), so the dd is self-consistent
like a FUSE one.  j_tor (hence Ip), j_bootstrap and j_non_inductive are kept;
j_ohmic = j_total - j_non_inductive.  Idempotent.

    python add_equilibrium_geometry.py
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from bouquet.io.geqdsk import GEQDSKEquilibrium  # noqa: E402
from bouquet.physics import (jphi_tokamaker_pressure_term,  # noqa: E402
                             jphi_tokamaker_to_jpar, jtor_imas_to_jphi_tokamaker)

DD = os.path.join(HERE, "D3Dlike_baseline_omas.json")
GEQ = os.path.join(HERE, "D3Dlike_Hmode_baseline.geqdsk")


def main():
    with open(DD) as fh:
        dd = json.load(fh)
    eq, cps = dd["equilibrium"], dd["core_profiles"]["profiles_1d"]
    g = GEQDSKEquilibrium(GEQ, cocos=1, nlevels=257)
    avg, pn_g = g.averages, g.psi_N
    r0 = eq["vacuum_toroidal_field"]["r0"]
    for k, ts in enumerate(eq["time_slice"]):
        p1, cp = ts["profiles_1d"], cps[k]
        psi = np.asarray(p1["psi"], dtype=float)
        pn = (psi - psi[0]) / (psi[-1] - psi[0])
        psi11 = 2 * np.pi * (g.psi_axis + pn * (g.psi_boundary - g.psi_axis))
        at = lambda key: np.interp(pn, pn_g, avg[key])        # noqa: E731
        b0 = eq["vacuum_toroidal_field"]["b0"][k]
        F = at("F") * (r0 * b0 / g.fpol[-1])                   # vacuum F = r0*b0
        gm1 = at("1/R**2")
        p1.update(
            psi=psi11.tolist(),
            rho_tor_norm=list(cp["grid"]["rho_tor_norm"]),
            f=F.tolist(), gm1=gm1.tolist(), gm9=at("1/R").tolist(),
            gm8=at("R").tolist(), gm5=(F**2 * gm1 + at("Bp**2")).tolist(),
            dpressure_dpsi=CubicSpline(psi11, p1["pressure"]).derivative()(psi11).tolist())
        cp["grid"]["psi"] = psi11.tolist()
        geom = {"F": F, "avg_R": np.asarray(p1["gm8"]), "avg_inv_R": np.asarray(p1["gm9"]),
                "avg_inv_R2": gm1, "avg_B2": np.asarray(p1["gm5"]),
                "pprime": -2 * np.pi * np.asarray(p1["dpressure_dpsi"]), "B0": b0}
        jphi = jtor_imas_to_jphi_tokamaker(np.asarray(cp["j_tor"]), geom)
        j_total = jphi_tokamaker_to_jpar(jphi - jphi_tokamaker_pressure_term(geom), geom)
        cp["j_total"] = j_total.tolist()
        cp["j_ohmic"] = (j_total - np.asarray(cp["j_non_inductive"])).tolist()
        print(f"slice {k} t={ts['time']}: min j_ohmic {min(cp['j_ohmic']):.3e}, "
              f"jphi/j_tor-1 axis {jphi[0] / cp['j_tor'][0] - 1:+.2%}")
    with open(DD, "w") as fh:
        json.dump(dd, fh)


if __name__ == "__main__":
    main()
