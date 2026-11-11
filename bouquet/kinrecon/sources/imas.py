r"""IMAS provider: driven current from ``core_sources`` and fast ions from ``core_profiles``.

Reads the dense IDS JSON that OMAS and FUSE (IMASdd.jl) write, already loaded into a Python
dict, and returns :class:`.provider.NonInductiveSources`.  Machine-agnostic: only
data-dictionary paths are used.

Paths read (IMAS DD 3.x/4.x names):

=====================================================================  ==========================
``core_sources.source[:].identifier.{index,name}``                     source classification
``core_sources.source[:].profiles_1d[:].{time, grid.rho_tor_norm}``    slice and grid
``core_sources.source[:].profiles_1d[:].j_parallel``                   <J.B>/B0 [A/m^2] (C2)
``core_sources.vacuum_toroidal_field.{b0, r0}``, ``core_sources.time``  B0 (signed), R0
``core_profiles.profiles_1d[:].ion[:].{density_fast,
pressure_fast_parallel, pressure_fast_perpendicular, label, element}``   fast ions
``core_profiles.profiles_1d[:].electrons.{density_fast, pressure_fast_*}``  fast electrons
``core_profiles.global_quantities.ip`` or
``equilibrium.time_slice[:].global_quantities.ip``                    sign of Ip
=====================================================================  ==========================

Source classification (``identifier.index``, DD ``core_sources`` identifier list):
driven current is summed from ``nbi`` (2), ``ec`` (3), ``lh`` (4), ``ic`` (5), ``fusion``
(6), ``auxiliary`` (100) and the combined RF/NBI indices 101-107; ``total`` (1, would double
count), ``ohmic`` (7), ``bootstrap_current`` (13) and the FUSE time-derivative (409) and
sawtooth (701) entries are excluded (``kinrecon`` computes its own bootstrap and ohmic
current; report 26 B.4).  Any other source with a non-zero ``j_parallel`` is REFUSED unless
the caller classifies it in ``classify``.

Sign: the DD defines ``j_parallel = <J.B>/B0`` with the producer's signed B0, so the co-Ip
part of ``<J.B>`` is ``j_parallel |B0| sign(Ip)``; the result is stored relative to Ip.

Fast-ion pressure: the DD cannot say whether ``pressure_fast_parallel/perpendicular`` hold
one third of the true pressures (OMAS, IMAS.jl, FUSE and every OMFIT export) or the true
pressures (the TRANSP plasma-state export), so ``fast_pressure_components`` must be stated:
``"imas_one_third"`` or ``"true_pressures"`` (:mod:`.conventions`, eq. C8).

Caveat for FUSE ``dd_sim`` input (record in ``assumptions``): FUSE's NB current and p_fast were
computed on FUSE's own fitted profiles and equilibrium and carry the known biases of its
simple beam model (all energy components born at full energy; current fractions used as
power fractions; an unexplained 0.1 R factor in the NBCD normalization).
"""

import numpy as np

from .conventions import AmbiguousConventionError, array_sha256, p_fast_from_imas_components, p_fast_from_true
from .provider import NonInductiveSources, Provenance

__all__ = ["from_imas", "to_imas", "DRIVEN_SOURCE_INDICES", "EXCLUDED_SOURCE_INDICES"]

DRIVEN_SOURCE_INDICES = {2: "nbi", 3: "ec", 4: "lh", 5: "ic", 6: "fusion", 100: "auxiliary", 101: "ic_nbi",
                         102: "ic_fusion", 103: "ic_nbi_fusion", 104: "ec_lh", 105: "ec_ic", 106: "lh_ic",
                         107: "ec_lh_ic"}
EXCLUDED_SOURCE_INDICES = {1: "total", 7: "ohmic", 13: "bootstrap_current", 409: "time_derivative",
                           701: "sawteeth"}
_COMPONENTS = ("imas_one_third", "true_pressures")


def _slice_times(blocks, fallback):
    t = [b.get("time") for b in blocks]
    if all(isinstance(x, (int, float)) for x in t):
        return np.asarray(t, float)
    if fallback is not None and len(fallback) == len(blocks):
        return np.asarray(fallback, float)
    raise AmbiguousConventionError("profiles_1d slices carry no time and no matching IDS time vector")


def _select(times, time, time_window, max_time_offset):
    if time_window > 0:
        idx = np.where(np.abs(times - time) <= 0.5 * time_window + 1e-12)[0]
        if idx.size == 0:
            raise ValueError(f"no slice within the window {time} +/- {time_window / 2} s")
        return idx, 0.0
    if max_time_offset is None:
        raise AmbiguousConventionError("time_window = 0 needs max_time_offset (nearest-slice tolerance) [s]")
    i = int(np.argmin(np.abs(times - time)))
    off = float(abs(times[i] - time))
    if off > max_time_offset:
        raise ValueError(f"nearest slice is {off:.4g} s from the requested time (> max_time_offset)")
    return np.array([i]), off


def _arr(block, key):
    v = block.get(key)
    if v is None:
        return None
    a = np.asarray(v, float)
    return a if a.size else None


def _ip_sign(dd, time, ip_sign):
    if ip_sign is not None:
        if ip_sign not in (1, -1):
            raise AmbiguousConventionError("ip_sign must be +1 or -1")
        return float(ip_sign), "caller"
    gq = dd.get("core_profiles", {}).get("global_quantities", {})
    ip = gq.get("ip")
    t = dd.get("core_profiles", {}).get("time")
    if ip is not None and np.size(ip):
        ip = np.asarray(ip, float)
        val = ip[int(np.argmin(np.abs(np.asarray(t, float) - time)))] if t is not None and len(t) == ip.size else ip.flat[0]
        if val != 0:
            return float(np.sign(val)), "core_profiles.global_quantities.ip"
    for ts in dd.get("equilibrium", {}).get("time_slice", []):
        v = ts.get("global_quantities", {}).get("ip")
        if isinstance(v, (int, float)) and v != 0:
            return float(np.sign(v)), "equilibrium.time_slice[0].global_quantities.ip"
    raise AmbiguousConventionError("the sign of Ip is not in the IDS; pass ip_sign")


def _b0(dd, time, B0, R0):
    vtf = dd.get("core_sources", {}).get("vacuum_toroidal_field", {})
    b0 = vtf.get("b0")
    if B0 is not None:
        b0_signed, origin = float(B0), "caller"
    elif b0 is not None and np.size(b0):
        b0 = np.asarray(b0, float)
        t = dd.get("core_sources", {}).get("time")
        if b0.size > 1 and t is not None and len(t) == b0.size:
            b0_signed = float(np.interp(time, np.asarray(t, float), b0))
        else:
            b0_signed = float(b0.flat[0])
        origin = "core_sources.vacuum_toroidal_field.b0"
    else:
        raise AmbiguousConventionError("core_sources.vacuum_toroidal_field.b0 is absent; pass B0 (signed)")
    r0 = R0 if R0 is not None else vtf.get("r0")
    if r0 is None:
        raise AmbiguousConventionError("core_sources.vacuum_toroidal_field.r0 is absent; pass R0")
    if b0_signed == 0:
        raise ValueError("B0 is zero")
    return b0_signed, float(r0), origin


def from_imas(dd, time, rho, *, fast_pressure_components, time_window=0.0, max_time_offset=None,
              ip_sign=None, B0=None, R0=None, classify=None, include_fast_electrons=True,
              uncertainty=None, source_code=None, source_version=None, assumptions=None, limitations=None,
              inputs=None, negative_policy="refuse"):
    r"""Non-inductive sources at ``time`` on the rho_tor_norm grid ``rho`` from an IMAS JSON dict.

    Parameters
    ----------
    dd
        IDS dict (``json.load`` of an OMAS/FUSE JSON) holding ``core_sources`` and/or
        ``core_profiles``.
    time, time_window, max_time_offset
        Slice selection [s]: mean over slices within ``time +/- time_window/2``, or (window 0)
        the nearest slice, refused when farther than ``max_time_offset``.
    rho
        Output rho_tor_norm grid; every profile is interpolated linearly in rho from its own
        ``grid.rho_tor_norm`` (no extrapolation beyond the producer's grid: refused).
    fast_pressure_components
        ``"imas_one_third"`` or ``"true_pressures"`` (no default; see module docstring).
    ip_sign, B0, R0
        Overrides when the IDS lacks them (B0 SIGNED, as the producer used it).
    classify
        ``{index_or_name: "driven" | "exclude"}`` for sources outside the built-in lists.
    negative_policy
        ``"refuse"`` or ``"clip"`` for negative fast-ion pressure/density samples.
    """
    if fast_pressure_components not in _COMPONENTS:
        raise AmbiguousConventionError(f"fast_pressure_components must be one of {_COMPONENTS}")
    rho = np.asarray(rho, float)
    if rho.ndim != 1 or np.any(np.diff(rho) <= 0):
        raise ValueError("rho must be strictly increasing")
    classify = dict(classify or {})
    sip, ip_origin = _ip_sign(dd, time, ip_sign)
    b0_signed, r0, b0_origin = _b0(dd, time, B0, R0)
    B0abs = abs(b0_signed)

    def onto(block, key):
        y = _arr(block, key)
        if y is None:
            return None
        g = np.asarray(block["grid"]["rho_tor_norm"], float)
        if rho[0] < g[0] - 1e-9 or rho[-1] > g[-1] + 1e-9:
            raise ValueError(f"requested rho [{rho[0]}, {rho[-1]}] exceeds the producer's grid "
                             f"[{g[0]}, {g[-1]}] for {key}")
        return np.interp(rho, g, y)

    jdotb, used, skipped = {}, [], []
    offsets = []
    cs = dd.get("core_sources", {})
    for k, src in enumerate(cs.get("source", [])):
        ident = src.get("identifier", {})
        idx, name = ident.get("index"), ident.get("name") or f"source{k}"
        blocks = src.get("profiles_1d", [])
        if not blocks:
            continue
        verdict = classify.get(idx, classify.get(name))
        if verdict is None:
            if idx in DRIVEN_SOURCE_INDICES:
                verdict = "driven"
            elif idx in EXCLUDED_SOURCE_INDICES:
                verdict = "exclude"
        if verdict == "exclude":
            skipped.append(f"{name} (index {idx})")
            continue
        times = _slice_times(blocks, cs.get("time"))
        sel, off = _select(times, time, time_window, max_time_offset)
        profs = [onto(blocks[i], "j_parallel") for i in sel]
        if all(p is None for p in profs):
            continue
        prof = np.mean([p if p is not None else np.zeros_like(rho) for p in profs], axis=0)
        if verdict is None:
            if np.any(prof != 0):
                raise AmbiguousConventionError(f"source {name!r} (index {idx}) carries j_parallel but is not "
                                               f"classified; pass classify={{{idx!r}: 'driven' or 'exclude'}}")
            continue
        if verdict != "driven":
            raise ValueError(f"classify values must be 'driven' or 'exclude', got {verdict!r}")
        key = name if name not in jdotb else f"{name}#{k}"
        jdotb[key] = prof * B0abs * sip          # co-Ip <J.B> = j_par |B0| sign(Ip)
        used.append(f"{key} (index {idx})")
        offsets.append(off)

    p_fast, p_par, p_perp, n_fast = {}, {}, {}, {}
    clipped = {}
    cp = dd.get("core_profiles", {})
    if cp.get("profiles_1d"):
        blocks = cp["profiles_1d"]
        times = _slice_times(blocks, cp.get("time"))
        sel, off = _select(times, time, time_window, max_time_offset)
        offsets.append(off)
        species = []
        for b in (blocks[i] for i in sel):
            for j, ion in enumerate(b.get("ion", [])):
                el = (ion.get("element") or [{}])[0]
                label = ion.get("label") or f"ion{j}_A{el.get('a')}_Z{el.get('z_n')}"
                species.append((label, ("ion", j)))
            if include_fast_electrons:
                species.append(("electrons", ("electrons", None)))
        seen = []
        for label, (kind, j) in species:
            if (label, kind, j) in seen:
                continue
            seen.append((label, kind, j))
            pars, perps, dens = [], [], []
            for b in (blocks[i] for i in sel):
                blk = b["ion"][j] if kind == "ion" else b.get("electrons", {})
                blk = dict(blk, grid=b["grid"])
                pars.append(onto(blk, "pressure_fast_parallel"))
                perps.append(onto(blk, "pressure_fast_perpendicular"))
                dens.append(onto(blk, "density_fast"))
            def mean(lst):
                return None if all(x is None for x in lst) else np.mean([x if x is not None else 0 * rho for x in lst], 0)
            par, perp, den = mean(pars), mean(perps), mean(dens)
            if (par is None) != (perp is None):
                raise ValueError(f"{label}: only one of pressure_fast_parallel/perpendicular is present")
            for nm, a in (("p_par", par), ("p_perp", perp), ("n_fast", den)):
                if a is not None and np.any(a < 0):
                    if negative_policy != "clip":
                        raise ValueError(f"{label} {nm} has negative samples (negative_policy='clip' to zero them)")
                    clipped[f"{nm}[{label}]"] = int((a < 0).sum())
            if par is not None:
                par, perp = np.maximum(par, 0), np.maximum(perp, 0)
                if np.any(par != 0) or np.any(perp != 0):
                    if fast_pressure_components == "imas_one_third":
                        p_par[label], p_perp[label] = 3.0 * par, 3.0 * perp
                        p_fast[label] = p_fast_from_imas_components(par, perp)
                    else:
                        p_par[label], p_perp[label] = par, perp
                        p_fast[label] = p_fast_from_true(par, perp)
            if den is not None and np.any(den != 0):
                n_fast[label] = np.maximum(den, 0)

    if not (jdotb or p_fast or n_fast):
        raise ValueError("no driven current and no fast-ion profile found at this time")
    prov = Provenance(
        provider="imas", source_code=str(source_code or "unknown (not stated by caller)"),
        source_version=str(source_version or "unknown"),
        inputs=dict(inputs or {}, time=time, time_window=time_window, max_slice_offset=max(offsets or [0.0]),
                    sources_used=used, sources_excluded=skipped,
                    sha256=array_sha256(rho, *jdotb.values(), *p_fast.values(), *n_fast.values())),
        conventions={"j_parallel": "<J.B>/B0 with the producer's signed B0 (DD)", "B0_signed": b0_signed,
                     "B0_origin": b0_origin, "ip_sign": sip, "ip_sign_origin": ip_origin,
                     "fast_pressure_components": fast_pressure_components},
        settings={"classify": {str(k): v for k, v in classify.items()}, "include_fast_electrons": include_fast_electrons,
                  "negative_policy": negative_policy, "clipped": clipped, "interpolation": "linear in rho_tor_norm"},
        assumptions=list(assumptions or []) + ["driven current is a pure parallel current (conventions C3)"],
        limitations=list(limitations or []))
    return NonInductiveSources(rho=rho, time=time, time_window=time_window, B0=B0abs, R0=r0, jdotb=jdotb,
                               p_fast=p_fast, n_fast=n_fast, p_par=p_par, p_perp=p_perp,
                               uncertainty=list(uncertainty or []), provenance=prov)


def to_imas(sources, *, ip_sign, bt_sign, source_index=None, species_labels=None):
    r"""Minimal ``core_sources`` + ``core_profiles`` IDS dict holding ``sources`` (one slice).

    Inverse of :func:`from_imas` with ``fast_pressure_components="imas_one_third"``:
    ``j_parallel = <J.B>_rel sign(Ip) / |B0|`` and B0 is written signed (``bt_sign``); fast
    pressures are written as one-third components of the true pressures (or of the isotropic
    p_fast when no anisotropy is stored).  ``source_index`` maps source names to DD indices
    (default: names starting with ``nbi``/``ec`` get 2/3, anything else 100 = auxiliary).
    """
    if ip_sign not in (1, -1) or bt_sign not in (1, -1):
        raise AmbiguousConventionError("ip_sign and bt_sign must be +1 or -1")
    src_index = dict(source_index or {})
    rho = sources.rho.tolist()
    out_src = []
    for name, jb in sources.jdotb.items():
        idx = src_index.get(name, 2 if name.lower().startswith("nbi") else 3 if name.lower().startswith("ec") else 100)
        out_src.append({"identifier": {"index": idx, "name": name},
                        "profiles_1d": [{"time": sources.time, "grid": {"rho_tor_norm": rho},
                                         "j_parallel": (np.asarray(jb) * ip_sign / sources.B0).tolist()}]})
    ions = []
    for sp in sorted(set(sources.p_fast) | set(sources.n_fast)):
        if sp == "electrons":
            continue
        ion = {"label": sp}
        if sp in sources.p_fast:
            if sp in sources.p_par:
                par, perp = sources.p_par[sp] / 3.0, sources.p_perp[sp] / 3.0
            else:
                par = perp = sources.p_fast[sp] / 3.0
            ion["pressure_fast_parallel"] = np.asarray(par).tolist()
            ion["pressure_fast_perpendicular"] = np.asarray(perp).tolist()
        if sp in sources.n_fast:
            ion["density_fast"] = sources.n_fast[sp].tolist()
        ions.append(ion)
    return {
        "core_sources": {"time": [sources.time],
                         "vacuum_toroidal_field": {"b0": [bt_sign * sources.B0], "r0": sources.R0},
                         "source": out_src},
        "core_profiles": {"time": [sources.time], "global_quantities": {"ip": [float(ip_sign)]},
                          "profiles_1d": [{"time": sources.time, "grid": {"rho_tor_norm": rho}, "ion": ions}]},
    }
