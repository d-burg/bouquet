"""Equilibrium geometry for synthetic IMAS dds (tests only).

The reader converts IMAS currents with the equilibrium's own f, gm1/gm5/gm8/gm9
and dpressure_dpsi (docs/current-conventions.md), which FUSE always writes.
These are plausible shaped-plasma values obeying <R><1/R> >= 1,
<1/R^2> >= <1/R>^2 and <B^2> >= F^2<1/R^2>.
"""
import numpy as np

from bouquet.physics import (jpar_to_jphi_tokamaker, jphi_tokamaker_pressure_term,
                             jphi_tokamaker_to_jtor_imas)


def eq_geometry(psi, pressure, r0=1.7, b0=-2.0):
    """FUSE-named ``equilibrium.profiles_1d`` geometry on ``psi`` (COCOS 11)."""
    psi = np.asarray(psi, dtype=float)
    x = np.sqrt((psi - psi[0]) / (psi[-1] - psi[0]))         # rho_tor_norm
    F = r0 * b0 * (1.0 - 0.02 * x**2)
    inv_R = (1.0 + 0.03 * x**2) / r0
    inv_R2 = inv_R**2 * (1.0 + 0.08 * x**2)
    return {
        "rho_tor_norm": x.tolist(),
        "f": F.tolist(),
        "gm8": (r0 * (1.0 + 0.04 * x**2)).tolist(),            # <R>
        "gm9": inv_R.tolist(),                                  # <1/R>
        "gm1": inv_R2.tolist(),                                 # <1/R^2>
        "gm5": (F**2 * inv_R2 + 0.05 * x**2).tolist(),          # <B^2>
        "dpressure_dpsi": np.gradient(np.asarray(pressure, dtype=float),
                                      psi).tolist(),
    }


def current_geom(p1, b0):
    """:mod:`bouquet.physics` ``geom`` of FUSE-named profiles_1d fields."""
    a = {k: np.asarray(p1[k], dtype=float)
         for k in ("f", "gm1", "gm5", "gm8", "gm9", "dpressure_dpsi")}
    return {"F": a["f"], "avg_R": a["gm8"], "avg_inv_R": a["gm9"],
            "avg_inv_R2": a["gm1"], "avg_B2": a["gm5"],
            "pprime": -2 * np.pi * a["dpressure_dpsi"], "B0": b0}


def jtor_from_jtotal(j_total, p1, b0):
    """IMAS ``j_tor`` of ``j_total`` = <J.B>/B0 (A6), as FUSE writes it."""
    g = current_geom(p1, b0)
    return jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(j_total, g) + jphi_tokamaker_pressure_term(g), g)
