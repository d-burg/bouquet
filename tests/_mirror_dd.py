"""Orientation-mirrored copies of a synthetic IMAS/OMAS dd (test helper).

A tokamak equilibrium carries two independent orientations, the plasma current
Ip and the toroidal field B0.  A dd written for the same plasma in any of the
four combinations -- in one fixed COCOS -- differs only in the SIGNS of the
quantities that carry one of those orientations.  Nothing else (kinetics,
pressure, geometry, rotation) is touched here: those are what a real
reversed-current dd shares with a normal one.

``mirror_dd(dd, s_ip, s_b0)`` returns a deep copy with every signed quantity
the dd carries transformed as a genuinely reversed source's would be (COCOS
held fixed; each field only where present):

  * odd in Ip -- multiplied by ``s_ip``:

    - ``equilibrium.time_slice[]``: ``global_quantities.ip``, ``psi_axis``,
      ``psi_boundary``; ``profiles_1d`` ``psi``, ``j_tor``, ``j_parallel``,
      ``dpressure_dpsi``, ``f_df_dpsi`` (FF' = F dF/dpsi: the psi derivative
      flips, F does not); ``profiles_2d[]`` ``psi``, ``j_tor``, ``j_parallel``,
      ``b_field_r``, ``b_field_z`` (the poloidal field); ``boundary.psi``;
    - ``core_profiles``: ``global_quantities`` ``ip``, ``current_*``,
      ``v_loop``; ``profiles_1d[]`` ``grid.psi``, ``grid.psi_magnetic_axis``,
      ``grid.psi_boundary`` and every current ``j_tor`` / ``j_total`` /
      ``j_ohmic`` / ``j_bootstrap`` / ``j_non_inductive``;
    - ``core_sources.source[]``: ``profiles_1d[]`` ``j_parallel``,
      ``current_parallel_inside``; ``global_quantities[]``
      ``current_parallel``;
    - ``pf_active.coil[].current.data`` -- the coils that make the mirrored
      poloidal field carry mirrored currents (``data_error_upper`` is a
      magnitude and is kept);

  * odd in B0 -- multiplied by ``s_b0``: ``equilibrium`` and
    ``core_profiles`` ``vacuum_toroidal_field.b0``, ``profiles_1d.f``, and
    ``profiles_2d[]`` ``b_field_tor`` / ``b_field_phi``;

  * odd in both -- multiplied by ``s_ip*s_b0``: every safety factor
    (``profiles_1d.q``, ``global_quantities`` ``q_axis`` / ``q_95`` /
    ``q_min.value``, ``core_profiles.profiles_1d[].q``).

Every value is a pure negation or a copy, so a mirrored read that lands in the
same frame must reproduce the original read BIT FOR BIT (IEEE negation is
exact, and every operation the reader performs on a current -- or on psi, whose
normalisation ``(psi - psi[0]) / (psi[-1] - psi[0])`` is even in it -- is odd
or even in it).

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
_CP_GLOBAL_IP = ("ip", "current_non_inductive", "current_bootstrap",
                 "current_ohmic", "v_loop")
_EQ_P1_IP = ("psi", "j_tor", "j_parallel", "dpressure_dpsi", "f_df_dpsi")
_EQ_P2_IP = ("psi", "j_tor", "j_parallel", "b_field_r", "b_field_z")
_EQ_P2_B0 = ("b_field_tor", "b_field_phi")


def _neg(seq, s):
    if isinstance(seq, list):
        return [_neg(v, s) for v in seq]
    return s * float(seq)


def _scale(d, keys, s):
    """Multiply ``d[k]`` (scalar or nested list) by ``s`` for each present key."""
    if not isinstance(d, dict):
        return
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = _neg(d[k], s)


def mirror_dd(dd: dict, s_ip: float, s_b0: float) -> dict:
    """A copy of ``dd`` with its current orientation ``s_ip`` and field
    orientation ``s_b0`` applied (see the module docstring)."""
    s_ip, s_b0 = float(s_ip), float(s_b0)
    s_q = s_ip * s_b0
    d = copy.deepcopy(dd)

    eq = d.get("equilibrium", {})
    _scale(eq.get("vacuum_toroidal_field"), ("b0",), s_b0)
    for ts in eq.get("time_slice", []):
        gq = ts.get("global_quantities", {})
        _scale(gq, ("ip", "psi_axis", "psi_boundary"), s_ip)
        _scale(gq, ("q_axis", "q_95"), s_q)
        _scale(gq.get("q_min"), ("value",), s_q)
        p1 = ts.get("profiles_1d", {})
        _scale(p1, _EQ_P1_IP, s_ip)
        _scale(p1, ("f",), s_b0)
        _scale(p1, ("q",), s_q)
        for p2 in ts.get("profiles_2d", []) or []:
            _scale(p2, _EQ_P2_IP, s_ip)
            _scale(p2, _EQ_P2_B0, s_b0)
        _scale(ts.get("boundary"), ("psi",), s_ip)

    cpi = d.get("core_profiles", {})
    _scale(cpi.get("vacuum_toroidal_field"), ("b0",), s_b0)
    _scale(cpi.get("global_quantities"), _CP_GLOBAL_IP, s_ip)
    for c in cpi.get("profiles_1d", []):
        _scale(c.get("grid"), ("psi", "psi_magnetic_axis", "psi_boundary"), s_ip)
        _scale(c, _CP_CURRENTS, s_ip)
        _scale(c, ("q",), s_q)

    for src in d.get("core_sources", {}).get("source", []):
        for pr in src.get("profiles_1d", []):
            _scale(pr, ("j_parallel", "current_parallel_inside"), s_ip)
        for g in src.get("global_quantities", []) or []:
            _scale(g, ("current_parallel",), s_ip)

    for coil in d.get("pf_active", {}).get("coil", []):
        _scale(coil.get("current"), ("data",), s_ip)
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
