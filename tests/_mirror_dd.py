"""Orientation-mirrored copies of a synthetic IMAS/OMAS dd (test helper).

A tokamak equilibrium carries two independent orientations, the plasma current
Ip and the toroidal field B0.  A dd written for the same plasma in any of the
four combinations differs only in the SIGNS of its current profiles (Ip) and of
its vacuum field (B0) -- and, through their product, of its safety factor.
Nothing else (kinetics, pressure, geometry, rotation) is touched here: those
are what a real reversed-current dd shares with a normal one.

``mirror_dd(dd, s_ip, s_b0)`` returns a deep copy with

  * every current profile multiplied by ``s_ip``: ``equilibrium.time_slice[]``
    ``global_quantities.ip`` and ``profiles_1d.j_tor``; ``core_profiles``
    ``j_tor``/``j_total``/``j_ohmic``/``j_bootstrap``/``j_non_inductive``; and
    every ``core_sources`` ``j_parallel``;
  * ``equilibrium.vacuum_toroidal_field.b0`` multiplied by ``s_b0``;
  * ``equilibrium.time_slice[].profiles_1d.q`` multiplied by ``s_ip*s_b0``.

Every value is a pure negation or a copy, so a mirrored read that lands in the
same frame must reproduce the original read BIT FOR BIT (IEEE negation is
exact, and every operation the reader performs on a current is odd in it).

No proprietary data: the only dd these tests mirror is the synthetic D3D-like
example shipped in ``examples/D3D-like`` and dds built in-process.
"""
from __future__ import annotations

import copy
import json
import os

#: The four (s_ip, s_b0) combinations relative to the source dd.  The shipped
#: synthetic example has Ip > 0 and B0 < 0, so (-1, +1) is the reversed-current
#: case (Ip < 0 with the same B0) and (-1, -1) reverses both.
ORIENTATIONS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))

#: The shipped synthetic D3D-like dd and mesh (examples/, tracked in git).
EXAMPLE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "examples", "D3D-like"))
EXAMPLE_DD = os.path.join(EXAMPLE_DIR, "D3Dlike_baseline_omas.json")
EXAMPLE_MESH = os.path.join(EXAMPLE_DIR, "DIIID_mesh.h5")
#: The H-mode slice of the example (the one its notebooks run).
EXAMPLE_TIME = 2.3043

_CP_CURRENTS = ("j_tor", "j_total", "j_ohmic", "j_bootstrap", "j_non_inductive")


def _neg(seq, s):
    if isinstance(seq, list):
        return [_neg(v, s) for v in seq]
    return s * float(seq)


def mirror_dd(dd: dict, s_ip: float, s_b0: float) -> dict:
    """A copy of ``dd`` with its current orientation ``s_ip`` and field
    orientation ``s_b0`` applied (see the module docstring)."""
    s_ip, s_b0 = float(s_ip), float(s_b0)
    d = copy.deepcopy(dd)
    eq = d.get("equilibrium", {})
    vtf = eq.get("vacuum_toroidal_field")
    if vtf is not None and vtf.get("b0") is not None:
        vtf["b0"] = _neg(vtf["b0"], s_b0)
    for ts in eq.get("time_slice", []):
        gq = ts.get("global_quantities", {})
        if "ip" in gq:
            gq["ip"] = s_ip * float(gq["ip"])
        p1 = ts.get("profiles_1d", {})
        if "j_tor" in p1:
            p1["j_tor"] = _neg(p1["j_tor"], s_ip)
        if "q" in p1:
            p1["q"] = _neg(p1["q"], s_ip * s_b0)
    for c in d.get("core_profiles", {}).get("profiles_1d", []):
        for k in _CP_CURRENTS:
            if k in c:
                c[k] = _neg(c[k], s_ip)
    for src in d.get("core_sources", {}).get("source", []):
        for pr in src.get("profiles_1d", []):
            if "j_parallel" in pr:
                pr["j_parallel"] = _neg(pr["j_parallel"], s_ip)
    return d


def tag(s_ip, s_b0) -> str:
    """``"ip+_b0-"``-style label for an orientation relative to the source."""
    return f"ip{'+' if s_ip > 0 else '-'}_b0{'+' if s_b0 > 0 else '-'}"


def write_mirrors(dd: dict, outdir: str, stem: str = "dd") -> dict:
    """Write all four mirrors of ``dd`` to ``outdir``; ``{(s_ip, s_b0): path}``."""
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for s_ip, s_b0 in ORIENTATIONS:
        p = os.path.join(outdir, f"{stem}_{tag(s_ip, s_b0)}.json")
        with open(p, "w") as fh:
            json.dump(mirror_dd(dd, s_ip, s_b0), fh)
        out[(s_ip, s_b0)] = p
    return out
