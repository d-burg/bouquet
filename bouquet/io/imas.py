"""Reader for FUSE IMAS/OMAS data-dictionary files (``dd_sim.json``).

FUSE writes the IMAS data dictionary as a plain JSON dump, so this reads with the
stdlib ``json`` module -- no IMAS/OMAS/OMFIT install required. It returns a
fully-separated baseline (no GS reconstruction needed): the IDS already carries
j_ohmic, j_bootstrap and the driven currents split apart.

Field mapping (verified against a D3D FUSE run)::

    equilibrium.time_slice[t].global_quantities.ip            -> Ip_target
    equilibrium.time_slice[t].global_quantities.li_3          -> l_i_target
    core_profiles.profiles_1d[t].grid.psi                     -> normalised -> psi_N
    core_profiles.profiles_1d[t].j_total                      -> total <J.B>/B0
    core_profiles.profiles_1d[t].j_tor                        -> total IMAS j_tor
    core_profiles.profiles_1d[t].j_ohmic                      -> (unused: residual)
    core_profiles.profiles_1d[t].j_bootstrap                  -> bootstrap <J.B>/B0
    equilibrium.time_slice[k].profiles_1d.{f,gm1,gm5,gm8,gm9,dpressure_dpsi}
                                                              -> conversion geometry
    core_profiles.profiles_1d[t].electrons.{density_thermal,temperature}
    core_profiles.profiles_1d[t].ion[*].{density_thermal,temperature,element[].z_n}
    core_profiles.profiles_1d[t].{electrons,ion[*]}.pressure_fast_{perpendicular,parallel}
    core_sources.source[*].profiles_1d[t].j_parallel          -> j_NBI (beam), j_RF (EC/LH/IC),
                                                                 j_other (fusion, runaways, sawteeth)

Currents are converted exactly to TokaMaker ``jphi`` = <j_phi> (bouquet's
convention; :mod:`bouquet.physics` docstring, ``docs/current-conventions.md``)
with the geometry of the equilibrium slice FUSE paired with the core_profiles
slice: the total from IMAS ``j_tor`` (A5), the bootstrap as its field-aligned
part plus the pressure term p'G (A7), each driven source as its field-aligned
part.  The inductive component is the residual
``j_phi - j_BS - j_NBI - j_RF - j_other`` so the decomposition sums exactly.
Fast pressure is isotropized (see :func:`bouquet.physics.isotropize_fast_pressure`).

Current orientation: bouquet works in ONE positive-current frame.  The
TokaMaker anchor is always solved to ``|Ip|`` with ``F0 = |r0*b0|`` (see
:func:`read_imas_geometry`), so every bootstrap bouquet recomputes on it (the
legacy ``solve_with_bootstrap`` path and the self-consistent loop alike) comes
out positive.  A dd written for a reversed-current discharge (``ip < 0`` in its
own COCOS) carries NEGATIVE current profiles, and combining those with a
positive recomputed bootstrap adds the bootstrap AGAINST the inductive current.
:func:`read_imas_baseline` therefore multiplies EVERY current profile it reads
-- ``j_total``, ``j_tor``, ``j_ohmic``, ``j_bootstrap``, the beam
``j_parallel`` and the equilibrium ``j_tor`` behind ``jphi_diff`` -- by
``sign(equilibrium ip)``, and records the factor as
:attr:`~bouquet.baseline.Baseline.source_current_sign`.  For ``ip > 0`` the
factor is ``+1.0`` and the read is bit-identical to what it was.  Everything
that is not a current (kinetics, pressure, rotation, E_r, the dd's own q) is
read unchanged, and so is a user-supplied ``FixedComponentsConfig.j_NBI`` /
``j_RF``: those are defined in bouquet's positive-Ip frame (co-current
positive) on both source paths.  A dd whose net (area-weighted) ``core_profiles.j_tor`` -- or
the equilibrium ``j_tor`` the jphi anchor uses -- disagrees in sign with that
factor is REFUSED (ValueError): its currents and its ip were written in
different orientations and no single factor puts it in one frame.
``ImasSource.current_orientation = +1 / -1`` names the factor explicitly for
a user who knows the file's convention (default ``"auto"``); the factor's
origin is recorded as ``Baseline.source_current_sign_origin``.

Note: ``j_BS`` read here is the FUSE bootstrap baseline, but it is *overridden*
when ``GenerationConfig.recalculate_j_BS`` is True -- bouquet then recomputes
bootstrap per draw via TokaMaker ``solve_with_bootstrap``, whose output
:func:`bouquet.physics._swb_jbs_to_toroidal` converts to the field-aligned
``kappa <j.B>`` (the toolkit's own convention is identified, never assumed).

The pressure-driven ``p'G`` (A7) is its own bucket (owner decision D2,
2026-10-09): never in ``j_BS``; the read records it as ``Baseline.j_pressure``.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..physics import (fast_ion_density_equivalent, impurity_pressure,
                       impurity_charge_with_fast_ions,
                       isotropize_fast_pressure,
                       jpar_to_jphi_tokamaker,
                       jphi_tokamaker_pressure_term,
                       jphi_tokamaker_to_jpar,
                       jphi_tokamaker_to_jtor_imas,
                       jtor_imas_to_jphi_tokamaker,
                       main_ion_density_from_zeff, parallel_to_toroidal)
from ..schema import (SPLIT_PRESSURE_IN_BOOTSTRAP, SPLIT_PRESSURE_IN_INDUCTIVE,
                      SPLIT_PRESSURE_SEPARATE)

from ..physics import ELEMENTARY_CHARGE as _EC  # p = e * sum_s(n_s * T_s)

if TYPE_CHECKING:
    from ..config import ImasSource, FixedComponentsConfig
    from ..baseline import Baseline


@functools.lru_cache(maxsize=2)
def _cached_dd(ids_path: str, _mtime_ns: int, _size: int,
               _ino: int = 0) -> dict:
    """Parsed ``dd_sim.json`` (up to ~1 GB), cached per (real path, mtime,
    size, inode): a slice sweep reads one file once.

    READ-ONLY CONTRACT: the cache hands out the SAME parsed object to every
    caller (the legacy reader, the unified engine's ``IdsAdapter.read``,
    ``read_imas_geometry`` and the plotting readers).  No caller may write
    into it; anything that slices or edits a dd (``_slice_in_time`` in
    ``write_imas_draw``) must work on its own fresh ``json.load`` or a deep
    copy.  ``tests/test_imas_dd_cache.py`` pins this by digest.

    MEMORY: up to two parsed files (about 2x the file size each, so ~2 GB
    for a 1 GB dd) stay resident for the life of the process -- in a
    ``run_parallel`` pool, in every worker for the whole of ``generate()``.
    :func:`clear_dd_cache` releases them."""
    with open(ids_path, "rb") as fh:
        return json.loads(fh.read())


def _load_dd(ids_path) -> dict:
    """``dd_sim.json`` at ``ids_path``, via :func:`_cached_dd` (shared,
    read-only).  Keyed on the real path (``"dd.json"``, its absolute path
    and a ``Path`` share one entry), mtime_ns, size and inode (a file
    replaced by rename is re-read).  Limitation: a same-size rewrite within
    the filesystem's mtime granule, or one seen through a stale NFS
    attribute cache, is served from the cache -- call
    :func:`clear_dd_cache` after rewriting a dd in place."""
    path = os.path.realpath(os.fspath(ids_path))
    st = os.stat(path)
    return _cached_dd(path, st.st_mtime_ns, st.st_size, st.st_ino)


def clear_dd_cache() -> None:
    """Release every cached parsed dd (see :func:`_cached_dd`)."""
    _cached_dd.cache_clear()

# Core-source identifier index for neutral-beam current drive.
NBI_SOURCE_INDEX = 2          # neutral beam injection -> summed into j_NBI
# Core-source identifier index for the sawtooth model (IMAS core_sources
# identifier enumeration).  Its current is held fixed in j_other (share also in
# j_sawteeth), and it is read as a slice-level FLAG for the
# closure_channel="sawtooth_bootstrap" gate, which pins q0 only where sawteeth
# make q0 ~ 1 a physical fact rather than a model artefact.
SAWTOOTH_SOURCE_INDEX = 701

#: The time-match window [s] when NEITHER time base has a local step (a
#: single-time core_sources entry on a single-time core_profiles base, or a
#: single-time core_sources slice on a single-time core_profiles).
#: Owner-approved 2026-10-07, replacing a window of a few float ulp:
#: rounding-level mismatches of millisecond-stored times are MATCHES, with
#: their dt recorded; the off_before_record (entry's own time after the
#: slice) and refusal (entry's own time before the slice, carrying current)
#: rules apply only beyond it.  When either base has a local step the
#: half-step window is used and this floor plays no part.
IMAS_SINGLE_TIME_WINDOW_S = 1e-5

#: Key of the time-match windows a single-slice export (:func:`_slice_in_time`,
#: :func:`write_imas_draw`) records so a re-read applies the windows of the
#: read it came from.  Cut to one slice, core_profiles and core_sources each
#: hold one time and an entry only the own slices the rule consults, so the
#: half-step windows would collapse to :data:`IMAS_SINGLE_TIME_WINDOW_S` and
#: an entry matched at an offset own time would re-read as off, an offset
#: core_sources base as a refusal.  Stored in the SCHEMA-LEGAL
#: ``code.parameters`` string (a JSON object; :func:`_set_export_window`),
#: under this key, in two places:
#:
#: * ``core_sources.code.parameters`` -- the core_sources slice:
#:   ``{"core_profiles_time", "core_sources_time", "window", "window_basis"}``
#:   (:func:`core_sources_slice`'s window and its basis at the original read);
#: * ``core_sources.source[j].code.parameters`` -- one entry:
#:   ``{"core_profiles_time", "core_sources_time", "window_own",
#:   "window_core_profiles"}`` (:func:`_source_slice_at`'s two windows).
#:
#: Two records of the cut itself sit under the same key (read back only as
#: stated):
#:
#: * ``equilibrium.code.parameters`` -- the equilibrium slices kept and their
#:   roles (:func:`equilibrium_slices_read`: ``kept_times``,
#:   ``targets_time``, ``orientation_time``, ``current_pairing_times``;
#:   :func:`write_imas_draw`: the one slice of the draw and
#:   ``template_current_pairing_time``).  A record only: the reader's
#:   selection rules find the same slices among the kept ones;
#: * ``core_profiles.code.parameters`` -- ``{"core_profiles_time",
#:   "time_neighbours"}``, the core_profiles times adjacent to the kept
#:   slice, which the ida_hybrid time rule's local step uses
#:   (:func:`_cp_time_grid`; honoured only when the file's one time is
#:   ``core_profiles_time``).
#:
#: A template ``parameters`` string that is itself a JSON object keeps its
#: keys; any other non-empty one is kept verbatim under
#: :data:`IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY`.  The reader also accepts the
#: block as a direct key of the node (exports written before it moved).  It
#: honours a block only when its two times equal the slice times it is
#: reading (the exported slice itself); anywhere else it is ignored.
IMAS_EXPORT_TIME_WINDOW_KEY = "bouquet_time_window"
#: Where a template's non-JSON ``code.parameters`` text is kept when the
#: export window is added to it.
IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY = "template_parameters"


def _get_export_window(node):
    """The export window block of an IDS node (``core_sources`` or one of its
    ``source`` entries): from its ``code.parameters`` JSON string, else (an
    export written before the block moved there) the node's own key; None
    when absent or unreadable."""
    import json
    if not isinstance(node, dict):
        return None
    code = node.get("code")
    par = code.get("parameters") if isinstance(code, dict) else None
    if isinstance(par, str) and par.strip():
        try:
            obj = json.loads(par)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and isinstance(
                obj.get(IMAS_EXPORT_TIME_WINDOW_KEY), dict):
            return obj[IMAS_EXPORT_TIME_WINDOW_KEY]
    legacy = node.get(IMAS_EXPORT_TIME_WINDOW_KEY)
    return legacy if isinstance(legacy, dict) else None


def _set_export_window(node, meta, key=None):
    """Record the export window block *meta* in *node*'s schema-legal
    ``code.parameters`` string (JSON), keeping what the template had there
    (its JSON keys, or its text under
    :data:`IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY`).  ``key`` (default
    :data:`IMAS_EXPORT_TIME_WINDOW_KEY`) names the block."""
    import json
    key = IMAS_EXPORT_TIME_WINDOW_KEY if key is None else key
    code = node.get("code")
    if not isinstance(code, dict):
        code = node["code"] = {}
    par = code.get("parameters")
    obj = {}
    if isinstance(par, str) and par.strip():
        try:
            obj = json.loads(par)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            obj = {IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY: par}
    obj[key] = meta
    code["parameters"] = json.dumps(obj)
    node.pop(key, None)


def _drop_export_window(node):
    """Remove an export window block from *node* (both locations)."""
    import json
    node.pop(IMAS_EXPORT_TIME_WINDOW_KEY, None)
    code = node.get("code")
    par = code.get("parameters") if isinstance(code, dict) else None
    if isinstance(par, str) and par.strip():
        try:
            obj = json.loads(par)
        except ValueError:
            return
        if isinstance(obj, dict):
            obj.pop(IMAS_EXPORT_TIME_WINDOW_KEY, None)
            code["parameters"] = json.dumps(obj)


def _export_window(meta, t_cp, t_src):
    """The export window block *meta* when it describes the slice pairing
    (*t_cp*, *t_src*) exactly, else ``None``."""
    if not isinstance(meta, dict) or t_cp is None or t_src is None:
        return None
    try:
        if (float(meta["core_profiles_time"]) == float(t_cp)
                and float(meta["core_sources_time"]) == float(t_src)):
            return meta
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _local_step(grid, k, toward):
    """The local time-step of the sorted, distinct *grid* at node *k*, on
    the side of the time *toward* (the interval it lies in; at an end of the
    grid, the end interval).  ``None`` for a grid of fewer than two times."""
    if grid.size < 2:
        return None
    if toward >= grid[k]:
        return float(grid[k + 1] - grid[k] if k + 1 < grid.size
                     else grid[k] - grid[k - 1])
    return float(grid[k] - grid[k - 1] if k > 0 else grid[1] - grid[0])


def _entry_time_window(times, t_slice, base_times=None):
    """``(k, dt, half_step)`` for matching a core_sources entry to the slice
    time *t_slice* by its OWN per-slice *times*: *k* is the entry's slice
    NEAREST t_slice, ``dt`` its distance, and ``half_step`` the acceptance
    window -- HALF the local time-step of the entry's own time grid (the
    interval t_slice lies in, or the end interval past either end).  An
    entry with a single time uses the local step of *base_times* (the
    core_profiles time base) at the node nearest t_slice, on the entry's
    side.  When neither grid has a step (a single-time entry on a
    single-time base) the window is :data:`IMAS_SINGLE_TIME_WINDOW_S`
    (10 us, owner-approved 2026-10-07; it was a few float ulp).

    The rule (owner-approved 2026-10-06, replacing the 1e-6 s absolute
    match of 2026-10-05; the half-step value is recorded, to be confirmed):
    nearest own slice, accepted within half a local step, otherwise the
    caller REFUSES a driven entry carrying current near that time -- a
    driven current is never dropped to zero and never read at another time.
    Refinement (2026-10-06): an entry carrying no current on the slices
    BRACKETING the time (:func:`_entry_bracketing_slices`) is OFF there,
    not missing -- it contributes zero and is stamped, not refused."""
    tt = np.asarray(times, dtype=float)
    t_slice = float(t_slice)
    k = int(np.argmin(np.abs(tt - t_slice)))
    dt = abs(float(tt[k]) - t_slice)
    grid = np.unique(tt)
    if grid.size >= 2:
        kg = int(np.argmin(np.abs(grid - tt[k])))
        step = _local_step(grid, kg, t_slice)
    else:
        step = None
        if base_times is not None:
            bg = np.unique(np.asarray(base_times, dtype=float))
            if bg.size >= 2:
                kb = int(np.argmin(np.abs(bg - t_slice)))
                step = _local_step(bg, kb, float(tt[k]))
    if step is None:
        half = IMAS_SINGLE_TIME_WINDOW_S
    else:
        half = 0.5 * step
    return k, dt, half


def _entry_bracketing_slices(times, t_slice):
    """Indices of a core_sources entry's own slices that BRACKET the slice
    time *t_slice* on the entry's OWN time grid *times*: its nearest own
    slice at or before t_slice and its nearest own slice at or after it
    (every slice sharing that time, should the grid repeat one).  When
    t_slice lies outside the entry's time range only the nearest END slice
    exists, and only it is returned.

    Used when no own slice lies within half a local step of t_slice
    (:func:`_entry_time_window`): an entry whose ``j_parallel`` is absent or
    identically zero on every bracketing slice is OFF at that time (a model
    source idle there, e.g. one whose grid starts a step after the IDS time
    base), not a missing input -- it contributes zero.  If any bracketing
    slice carries current the caller still refuses (refinement of the
    half-step rule, 2026-10-06)."""
    tt = np.asarray(times, dtype=float)
    t_slice = float(t_slice)
    out = []
    below = tt <= t_slice
    if np.any(below):
        out.extend(np.flatnonzero(tt == tt[below].max()).tolist())
    above = tt >= t_slice
    if np.any(above):
        out.extend(np.flatnonzero(tt == tt[above].min()).tolist())
    return sorted(set(int(k) for k in out))


def _carries_current(q):
    """Whether one ``profiles_1d`` slice carries a non-zero (or non-finite)
    ``j_parallel``."""
    jp = q.get("j_parallel")
    return jp is not None and bool(np.any(np.asarray(jp, float) != 0.0))


def _entry_off_near(s, t_slice):
    """``None`` when the core_sources entry *s* carries current on a slice
    bracketing *t_slice* (or cannot be judged: no per-slice times), else
    the provenance reason it is OFF near that time
    (:func:`_entry_bracketing_slices`)."""
    pr = s.get("profiles_1d", [])
    times = [q.get("time") for q in pr]
    if t_slice is None or not pr or any(t is None for t in times):
        return None
    br = _entry_bracketing_slices(times, t_slice)
    if any(_carries_current(pr[k]) for k in br):
        return None
    at = ", ".join(f"{float(times[k]):.9g}" for k in br)
    return (f"off near the slice: no current on its bracketing slices at "
            f"{at} s (t = {float(t_slice):.9g} s; no own slice within half "
            "a time-step) -- a source idle at this time, contributing zero")


def _entry_time_why(t_slice, times, k, dt, half):
    """Why an entry has no slice within half a step of *t_slice*."""
    tt = np.asarray(times, dtype=float)
    return (f"no profiles_1d slice within half a time-step of t = "
            f"{t_slice:.9g} s (nearest own time {float(tt[k]):.9g} s, "
            f"|dt| = {dt:.3g} s > {half:.3g} s, half its local time-step; "
            f"its own times span {tt.min():.9g}-{tt.max():.9g} s)")


def _entry_time_refusal(who, idn, why):
    """The refusal text for a driven entry with no slice at this time."""
    return (f"{who}: core_sources {idn.get('name')!r} (index "
            f"{idn.get('index')}) carries a non-zero j_parallel but has "
            f"{why}, and carries current on its own slices bracketing that "
            "time.  Refusing rather than reading its current at another "
            "time or dropping it to zero (the half-step match rule, "
            "owner-approved 2026-10-06)")


def _half_local_step(times, t_at, toward):
    """HALF the local step of the time grid *times* at its node nearest
    *t_at*, on the side of *toward* (:func:`_local_step`), or ``None`` for a
    grid of fewer than two distinct times."""
    if times is None:
        return None
    grid = np.unique(np.asarray(times, dtype=float))
    if grid.size < 2:
        return None
    k = int(np.argmin(np.abs(grid - float(t_at))))
    return 0.5 * _local_step(grid, k, float(toward))


#: The time rule of the core_sources reads (owner decision 2026-10-06),
#: stamped on every record.
SOURCE_TIME_RULE = (
    "core_sources slice: nearest the core_profiles slice read, within half "
    "the local core_profiles time-step, else refused; entry: nearest own "
    "slice within half its own local step AND within half the local "
    "core_profiles step of the core_profiles slice time (never "
    "interpolated); before the entry's first own time: off "
    "(off_before_record); past its last own time: refused when that slice "
    "carries current; idle on its bracketing own slices: off; with no "
    "local step on either time base the window is 1e-05 s "
    "(IMAS_SINGLE_TIME_WINDOW_S)")


def _cp_window(cp_times, src_times, t_cp, toward):
    """``(half, basis)``: the core_profiles window at the core_profiles
    slice time *t_cp* -- half its local step on the side of *toward*; a
    single-time core_profiles uses the core_sources' own local step, and
    two single-time bases :data:`IMAS_SINGLE_TIME_WINDOW_S` (10 us,
    owner-approved 2026-10-07; it was a few float ulp)."""
    h = _half_local_step(cp_times, t_cp, toward)
    if h is not None:
        return h, "half the local core_profiles time-step"
    h = _half_local_step(src_times, t_cp, toward)
    if h is not None:
        return h, ("half the local core_sources time-step (core_profiles "
                   "has a single time)")
    return (IMAS_SINGLE_TIME_WINDOW_S,
            "the single-time floor IMAS_SINGLE_TIME_WINDOW_S (single-time "
            "core_profiles and core_sources)")


def core_sources_slice(src_ids, cp_times, ic, T=None, who="IMAS reader"):
    """``(isrc, t_src, record)``: the core_sources slice read with the
    core_profiles slice *ic* (owner decision 2026-10-06).

    The slice is the core_sources time NEAREST the core_profiles slice
    actually read (``cp_times[ic]``), and it must lie within HALF the local
    core_profiles time-step of it (:func:`_cp_window`), else ``ValueError``
    naming both times -- a single-time core_sources is no longer read at
    any requested time.  A core_sources with no time base is read by index
    (``ic``), as before.  The record carries both times, ``dt`` (core_sources
    minus core_profiles), the window and its basis, and the rule."""
    tb = src_ids.get("time")
    cpt = (None if cp_times is None or not len(cp_times)
           else np.asarray(cp_times, dtype=float))
    t_cp = None if cpt is None else float(cpt[min(ic, cpt.size - 1)])
    if not tb:
        return ic, None, dict(core_profiles_time=t_cp,
                              core_sources_time=None, dt=None, window=None,
                              rule="by index: core_sources carries no time "
                                   "base")
    tt = np.asarray(tb, dtype=float)
    if t_cp is None:
        isrc = _nearest_index(tt, T, "core_sources")
        return isrc, float(tt[isrc]), dict(
            core_profiles_time=None, core_sources_time=float(tt[isrc]),
            dt=None, window=None,
            rule="nearest the requested time (core_profiles has no time "
                 "base)")
    isrc = int(np.argmin(np.abs(tt - t_cp)))
    t_src = float(tt[isrc])
    dt = t_src - t_cp
    half, basis = _cp_window(cpt, tt, t_cp, t_src)
    _meta = _export_window(_get_export_window(src_ids), t_cp, t_src)
    if _meta is not None and cpt.size == 1 and tt.size == 1:
        # a single-slice export: the window of the read it came from
        half, basis = float(_meta["window"]), str(_meta["window_basis"])
    rec = dict(core_profiles_time=t_cp, core_sources_time=t_src, dt=dt,
               window=half, window_basis=basis, rule=SOURCE_TIME_RULE)
    if abs(dt) > half:
        raise ValueError(
            f"{who}: the core_sources slice nearest the core_profiles slice "
            f"read (t = {t_cp:.9g} s) is at t = {t_src:.9g} s: |dt| = "
            f"{abs(dt):.3g} s > {half:.3g} s, {basis} (core_sources times "
            f"span {tt.min():.9g}-{tt.max():.9g} s).  Refusing rather than "
            "reading the driven currents at another time (owner decision "
            "2026-10-06)")
    return isrc, t_src, rec


def _source_slice_at(s, isrc, t_slice, n_time, base_times=None, *,
                     t_cp=None, cp_half=None, rec=None):
    """``(profile, how)``: a ``core_sources`` entry's ``profiles_1d`` at the
    slice *isrc* (time *t_slice*), or ``(None, why)`` when the entry has no
    slice within half a local time-step of that time
    (:func:`_entry_time_window`; the CALLER decides: a beam entry is
    refused unless it is off -- :func:`_entry_off_before_record`,
    :func:`_entry_off_near`).  An entry carrying its own per-slice times is
    matched BY TIME to its nearest slice (a model's entry may start later
    than the IDS time base, so the list index is not the slice); one without
    them must have exactly the IDS's number of slices, or it cannot be
    aligned and is refused (``ValueError``) -- never its first slice taken
    in place of a missing one.  The rule of
    ``bouquet.adapters._ids_source_slice``.

    Owner decision 2026-10-06: the matched own slice must ALSO lie within
    *cp_half* (half the local core_profiles step) of the core_profiles
    slice time *t_cp* (default: *t_slice*), so a coarse own grid or a
    constant offset cannot pass on the entry's own step alone.  *rec*, a
    dict, receives the match: matched own time, ``dt`` (own minus
    core_profiles time), both windows, the bracketing own times, the
    entry's first / last own time and the status."""
    pr = s.get("profiles_1d", [])
    idn = s.get("identifier", {}) or {}
    if rec is None:
        rec = {}
    rec.update(name=idn.get("name"), index=idn.get("index"))
    if not pr:
        rec.update(status="no_profiles")
        return None, "no profiles_1d"
    times = [q.get("time") for q in pr]
    if t_slice is not None and all(t is not None for t in times):
        tt = np.asarray(times, dtype=float)
        t_ref = float(t_slice if t_cp is None else t_cp)
        k, dt, half = _entry_time_window(times, t_slice, base_times)
        _meta = _export_window(_get_export_window(s), t_ref, t_slice)
        if _meta is not None:
            # a single-slice export keeps only the own slices this rule
            # consults: the windows are those of the read it came from
            half = float(_meta["window_own"])
            if _meta.get("window_core_profiles") is not None:
                cp_half = float(_meta["window_core_profiles"])
        br = _entry_bracketing_slices(times, t_slice)
        rec.update(own_time_nearest=float(tt[k]),
                   dt=float(tt[k]) - t_ref, window_own=float(half),
                   window_core_profiles=(None if cp_half is None
                                         else float(cp_half)),
                   bracketing_own_times=[float(tt[j]) for j in br],
                   first_own_time=float(tt.min()),
                   last_own_time=float(tt.max()))
        if dt > half:
            rec.update(status="unmatched")
            return None, _entry_time_why(t_slice, times, k, dt, half)
        if cp_half is not None and abs(float(tt[k]) - t_ref) > cp_half:
            rec.update(status="unmatched")
            return None, (
                f"no profiles_1d slice within half the local core_profiles "
                f"time-step of t = {t_ref:.9g} s (nearest own time "
                f"{float(tt[k]):.9g} s, |dt| = "
                f"{abs(float(tt[k]) - t_ref):.3g} s > {cp_half:.3g} s, half "
                f"the local core_profiles step; within its own half-step "
                f"{half:.3g} s; its own times span {tt.min():.9g}-"
                f"{tt.max():.9g} s)")
        rec.update(status="matched", matched_time=float(tt[k]))
        return pr[k], "matched by time"
    if n_time is not None and len(pr) != n_time:
        raise ValueError(
            f"IMAS reader: core_sources {idn.get('name')!r} (index "
            f"{idn.get('index')}) has {len(pr)} profiles_1d slices for "
            f"{n_time} core_sources times and no per-slice time: it cannot be "
            "aligned with the slice read")
    if isrc >= len(pr):
        raise ValueError(
            f"IMAS reader: core_sources {idn.get('name')!r} (index "
            f"{idn.get('index')}) has no profiles_1d slice {isrc}")
    rec.update(status="matched", matched_time=t_slice, dt=(
        None if (t_slice is None or t_cp is None) else
        float(t_slice) - float(t_cp)), rule_entry="by index")
    return pr[isrc], "by index"


def _entry_off_before_record(s, t_slice):
    """The entry's FIRST own time when the slice time *t_slice* lies before
    it (the entry has no record before its first own sample: it is OFF at
    that time -- owner decision 2026-10-06), else ``None``.  Judged only on
    an entry carrying per-slice times."""
    pr = s.get("profiles_1d", [])
    times = [q.get("time") for q in pr]
    if t_slice is None or not pr or any(t is None for t in times):
        return None
    first = float(np.min(np.asarray(times, dtype=float)))
    return first if float(t_slice) < first else None


_OFF_BEFORE_ANNOUNCED = set()


def _announce_off_before(who, idn, first, t_slice, key=None):
    """Print and warn ONCE per run (per source file and entry) that a driven
    entry is off before its first own time."""
    import warnings
    tag = (key, idn.get("name"), idn.get("index"), float(first))
    if tag in _OFF_BEFORE_ANNOUNCED:
        return
    _OFF_BEFORE_ANNOUNCED.add(tag)
    msg = (f"{who}: core_sources {idn.get('name')!r} (index "
           f"{idn.get('index')}) has no record before its first own time "
           f"{float(first):.9g} s: it is OFF (zero) at t = "
           f"{float(t_slice):.9g} s and at every earlier slice, stamped "
           "off_before_record (owner decision 2026-10-06)")
    print(f"[imas] NOTE {msg}", flush=True)
    warnings.warn(msg, UserWarning, stacklevel=3)


def _nearest_index(time_array, t: Optional[float], what: str) -> int:
    """Index of the slice nearest ``t`` (seconds) in ``time_array``."""
    ta = np.asarray(time_array, dtype=float)
    if ta.size == 1:
        return 0
    if t is None:
        raise ValueError(
            f"{what} has {ta.size} time slices; set ImasSource.time (seconds). "
            f"Range [s]: [{ta.min():.4f}, {ta.max():.4f}]"
        )
    return int(np.argmin(np.abs(ta - t)))


def orientation_slice_index(dd: dict, t: Optional[float]) -> int:
    """Index of the ``equilibrium`` slice a dd's current orientation is read at.

    That is the equilibrium slice nearest the TIME of the ``core_profiles``
    slice nearest ``t`` -- the slice the currents are read from -- so the sign
    belongs to those currents even when the two IDSs have different time bases
    (with a common time base, as in FUSE output, it is simply the equilibrium
    slice nearest ``t``).  A dd without a ``core_profiles`` time base falls back
    to the equilibrium slice nearest ``t``.

    The one selection rule shared by :func:`read_imas_baseline` and
    :func:`bouquet.coil_targets.measured_from_pf_active`, so the factor the
    reader applies to the plasma currents and the one applied to the measured
    coil currents cannot drift apart.
    """
    eq = dd["equilibrium"]
    cp_t = np.atleast_1d(np.asarray(
        (dd.get("core_profiles") or {}).get("time", []), dtype=float))
    if cp_t.size:
        ic = _nearest_index(cp_t, t, "core_profiles")
        t = float(cp_t[min(ic, cp_t.size - 1)])
    return _nearest_index(eq["time"], t, "equilibrium")


def orientation_ip(dd: dict, t: Optional[float]) -> Optional[float]:
    """Signed ``equilibrium`` ip at :func:`orientation_slice_index`, or None.

    None when the dd carries no equilibrium ip there (no ``equilibrium``, no
    time slices, no ``global_quantities.ip``) or the slice cannot be chosen
    (several slices and ``t`` None).  ``sign`` of it is the ``"auto"``
    orientation factor (:func:`source_current_sign`).
    """
    eq = dd.get("equilibrium") or {}
    ts = eq.get("time_slice") or []
    if not ts:
        return None
    if "time" not in eq:
        eq = dict(eq, time=[0.0 if t is None else float(t)])
    try:
        i = orientation_slice_index(dict(dd, equilibrium=eq), t)
        return float(ts[min(i, len(ts) - 1)]["global_quantities"]["ip"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None


# ===========================================================================
#  Fast-pressure storage convention
#
#  ``pressure_fast_parallel`` / ``pressure_fast_perpendicular`` are written with
#  two incompatible meanings, and the scalar p_fast they reduce to differs by a
#  factor of THREE:
#
#    * IMAS.jl / FUSE store the PER-DEGREE-OF-FREEDOM pressure -- ``pressa/3``
#      in each field -- so the scalar is ``p_par + 2*p_perp``  ("sum").
#    * The IMAS data dictionary (and OMAS-written dds) define
#      ``pressure_fast_parallel`` as the FULL fast parallel pressure, so the
#      scalar is the trace ``(p_par + 2*p_perp)/3``            ("trace").
#
#  Neither convention is recorded in a dedicated dd field, so bouquet infers it
#  from the dd's recorded provenance (:func:`detect_p_fast_convention`) and says
#  loudly when it cannot.  An explicit ``p_fast_reduction`` always wins.
# ===========================================================================

#: Where a producer records who wrote an IDS.  Only these slots are inspected --
#: deliberately NOT every IDS, because sub-IDSes are routinely imported from
#: other codes (a FUSE dd carries an OMFIT-sourced ``nbi.ids_properties.comment``)
#: and would otherwise vote on a convention they do not set.
_PROVENANCE_IDS = ("dataset_description", "core_profiles", "equilibrium", "summary")
_PROVENANCE_SLOTS = (
    ("ids_properties", "comment"),
    ("ids_properties", "provider"),
    ("ids_properties", "source"),
    ("code", "name"),
    ("code", "description"),
    ("code", "repository"),
)

#: Top-level keys that only IMASdd.jl / FUSE write (they are not IMAS DD IDSes).
#: Verified present in FUSE ``dd_sim.json`` output.
_IMASJL_TOPLEVEL_MARKERS = (
    "global_time", "requirements", "build", "balance_of_plant",
    "solid_mechanics", "costing",
)

#: Producer name fragments (matched case-insensitively in the slots above).
_FUSE_PRODUCER_TOKENS = ("fuse", "imas.jl", "imasdd", "torreypines")
_DD_PRODUCER_TOKENS = ("omas", "omfit", "imaspy", "imas-python", "imas_core",
                       "access-layer", "access layer")

#: Explicit stamp a producer (or a hand-built fixture) can put in any of the
#: provenance slots above to state its convention outright, e.g.
#: ``"comment": "... p_fast_reduction=trace ..."``.
_EXPLICIT_STAMP_RE = (
    r"(?:p_fast_reduction|pressure_fast_convention|fast_pressure_convention)"
    r"\s*[:=]\s*([A-Za-z_-]+)"
)
#: Values the stamp may carry -> the reduction rule they select.
_STAMP_VALUES = {
    "sum": "sum", "per_dof": "sum", "per-dof": "sum",
    "per_degree_of_freedom": "sum",
    "trace": "trace", "total": "trace", "full": "trace",
    "mean": "mean", "perp": "perp",
}


def _provenance_strings(dd: dict):
    """``(location, text)`` for every recorded producer string in ``dd``.

    Only the slots in ``_PROVENANCE_IDS`` x ``_PROVENANCE_SLOTS`` are read.
    """
    out = []
    for ids in _PROVENANCE_IDS:
        node = dd.get(ids)
        if not isinstance(node, dict):
            continue
        for grp, key in _PROVENANCE_SLOTS:
            sub = node.get(grp)
            if not isinstance(sub, dict):
                continue
            val = sub.get(key)
            if isinstance(val, str) and val.strip():
                out.append((f"{ids}.{grp}.{key}", val.strip()))
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, str) and v.strip():
                        out.append((f"{ids}.{grp}.{key}", v.strip()))
    return out


def detect_p_fast_convention(dd: dict) -> dict:
    """Infer the fast-pressure storage convention of a loaded dd.

    Returns ``{"rule": <"sum"|"trace"|"mean"|"perp"|None>, "basis": str,
    "evidence": str}``.  ``rule`` is ``None`` when the dd records nothing that
    identifies its producer -- the caller then has to choose a fallback and warn.

    Precedence, most specific first:

    1. **explicit stamp** -- ``p_fast_reduction=<rule>`` (or
       ``pressure_fast_convention=<per_dof|total>``) in any provenance slot;
    2. **IMAS.jl structure** -- a top-level key only IMASdd.jl/FUSE writes
       (``global_time``, ``requirements``, ``build``, ``balance_of_plant``,
       ``solid_mechanics``, ``costing``).  This identifies the *writer of the
       file*, which is what sets the convention, so it outranks per-IDS producer
       strings: a FUSE dd legitimately carries IDSes imported from other codes.
    3. **producer strings** -- a FUSE/IMAS.jl name in a provenance slot selects
       ``"sum"``; an OMAS/OMFIT/IMASPy (data-dictionary) name selects ``"trace"``.
    """
    import re

    prov = _provenance_strings(dd)

    # 1. explicit stamp
    for where, text in prov:
        m = re.search(_EXPLICIT_STAMP_RE, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            if val in _STAMP_VALUES:
                return {"rule": _STAMP_VALUES[val], "basis": "explicit-stamp",
                        "evidence": f"{where} = {text!r}"}
            # A dd that states its own convention and is then misread is exactly
            # what the stamp exists to prevent -- never discard one silently.
            import warnings
            warnings.warn(
                f"{where} carries a fast-pressure convention stamp whose value "
                f"{m.group(1)!r} is not recognised; expected one of "
                + ", ".join(repr(v) for v in sorted(_STAMP_VALUES))
                + ". The stamp is ignored and the convention is inferred from the "
                "dd's structure/producer instead.",
                stacklevel=2,
            )

    # 2. IMAS.jl / FUSE structural markers
    found = [k for k in _IMASJL_TOPLEVEL_MARKERS if k in dd]
    if found:
        return {"rule": "sum", "basis": "imas.jl-structure",
                "evidence": ("top-level key(s) only IMASdd.jl/FUSE write: "
                             + ", ".join(sorted(found)))}

    # 3. producer strings
    for where, text in prov:
        low = text.lower()
        for tok in _FUSE_PRODUCER_TOKENS:
            if tok in low:
                return {"rule": "sum", "basis": "producer-string",
                        "evidence": f"{where} = {text!r} (matched {tok!r})"}
    for where, text in prov:
        low = text.lower()
        for tok in _DD_PRODUCER_TOKENS:
            if tok in low:
                return {"rule": "trace", "basis": "producer-string",
                        "evidence": f"{where} = {text!r} (matched {tok!r})"}

    return {"rule": None, "basis": "undetermined",
            "evidence": ("no IMASdd.jl top-level key and no producer recorded in "
                         "{dataset_description,core_profiles,equilibrium,summary}"
                         ".{ids_properties.{comment,provider,source},"
                         "code.{name,description,repository}}")}


#: Rule used when provenance is undeterminable.  "sum" is the majority case for
#: the dds bouquet reads (FUSE), so it is the safer bet -- but it is never
#: applied silently; see :func:`resolve_p_fast_reduction`.
P_FAST_UNDETERMINED_FALLBACK = "sum"


def warn_p_fast_undetermined(rule: str = P_FAST_UNDETERMINED_FALLBACK,
                             stacklevel: int = 2) -> None:
    """The factor-of-3 warning for a dd whose convention cannot be determined.

    Split out of :func:`resolve_p_fast_reduction` so a caller that resolves the
    rule eagerly can hold the warning back until it knows the choice moved a
    number -- see :func:`read_imas_baseline`.
    """
    import warnings
    warnings.warn(
        "p_fast_reduction='auto': this data dictionary records no producer, so the "
        "storage convention of pressure_fast_parallel/perpendicular cannot be "
        "determined. The two conventions differ by a FACTOR OF 3 in the fast-ion "
        "pressure (and hence in beta_N, W_MHD and p'):\n"
        "  'sum'   -- IMAS.jl/FUSE write the pressure PER DEGREE OF FREEDOM "
        "(pressa/3 in each field); scalar p_fast = p_par + 2*p_perp.\n"
        "  'trace' -- the IMAS data dictionary (and OMAS-written dds) define "
        "pressure_fast_parallel as the FULL parallel pressure; scalar "
        "p_fast = (p_par + 2*p_perp)/3.\n"
        f"Falling back to {rule!r} (the majority case for the dds bouquet reads). "
        "Set it explicitly to silence this: "
        "BouquetConfig.fixed_components.p_fast_reduction = 'sum' | 'trace' (or "
        "read_imas_baseline(..., p_fast_reduction=...)); alternatively stamp the dd "
        "itself, e.g. core_profiles.ids_properties.comment = "
        "'... p_fast_reduction=trace ...'.",
        stacklevel=stacklevel + 1,
    )


def resolve_p_fast_reduction(dd: dict, requested: str = "auto",
                             warn: bool = True) -> dict:
    """Resolve the fast-pressure reduction rule for ``dd``.

    ``requested`` is ``"auto"`` (inspect the dd's provenance) or one of the
    explicit rules, which always wins and is applied silently.

    Returns the metadata dict recorded on :attr:`bouquet.baseline.Baseline.p_fast_meta`::

        {"rule": str, "basis": str, "evidence": str, "requested": str,
         "warned": bool}

    When ``requested == "auto"`` and the dd's provenance is undeterminable the
    rule falls back to :data:`P_FAST_UNDETERMINED_FALLBACK` and a single
    ``UserWarning`` is raised naming both conventions and the factor-of-3 stake.
    ``warn=False`` resolves the metadata without raising it; the caller then owns
    the warning and can key on ``basis == "undetermined-fallback"``.
    """
    from ..physics import P_FAST_REDUCTIONS

    if requested != "auto":
        if requested not in P_FAST_REDUCTIONS:
            raise ValueError(
                f"unknown p_fast_reduction {requested!r}; expected 'auto' or one of "
                + ", ".join(repr(r) for r in P_FAST_REDUCTIONS))
        return {"rule": requested, "basis": "explicit-argument",
                "evidence": f"p_fast_reduction={requested!r} supplied by the caller",
                "requested": requested, "warned": False}

    det = detect_p_fast_convention(dd)
    if det["rule"] is not None:
        return {**det, "requested": "auto", "warned": False}

    rule = P_FAST_UNDETERMINED_FALLBACK
    if warn:
        warn_p_fast_undetermined(rule, stacklevel=2)
    return {"rule": rule, "basis": "undetermined-fallback",
            "evidence": det["evidence"], "requested": "auto",
            "warned": bool(warn)}


def _isotropic_fast_pressure(species: dict, method: str, n: int,
                             missing_parallel=None, label: str = ""):
    """Isotropized fast pressure for one species, or zeros if it carries none.

    When ``pressure_fast_parallel`` is absent the reader closes the tensor with
    ``p_par := p_perp``.  The closure is the same for every rule; what it MEANS
    is not:

      * ``"trace"`` / ``"mean"`` / ``"perp"`` (full directional pressures):
        ``p_par = p_perp`` is the isotropic fast-ion assumption and the scalar
        comes out as ``p_perp`` -- identical to reading a fully isotropic dd.
      * ``"sum"`` (per-degree-of-freedom fields): the same closure gives
        ``3*p_perp``.  That is the literal per-dof reading, but it is ambiguous --
        a writer that simply omitted an all-zero parallel field means
        ``2*p_perp``, and a dictionary-convention dd means ``p_perp``.  The
        caller is warned rather than left with a silent 3x.

    Species whose parallel field is missing while the perpendicular field is
    non-zero are appended to ``missing_parallel`` so the caller warns once.
    """
    if "pressure_fast_perpendicular" not in species:
        return np.zeros(n)
    p_perp = np.asarray(species["pressure_fast_perpendicular"], dtype=float)
    if "pressure_fast_parallel" in species:
        p_par = np.asarray(species["pressure_fast_parallel"], dtype=float)
    else:
        p_par = p_perp
        if missing_parallel is not None and float(np.max(np.abs(p_perp))) > 0.0:
            missing_parallel.append(label or species.get("label", "?"))
    return isotropize_fast_pressure(p_perp, p_par, method=method)


def _warn_missing_parallel(species_labels, rule: str):
    """Single warning for species that carry only the perpendicular fast field."""
    if not species_labels:
        return
    import warnings
    who = ", ".join(str(s) for s in species_labels)
    if rule == "sum":
        detail = (
            "Under p_fast_reduction='sum' the two fields are per-degree-of-freedom, "
            "so the closure p_par := p_perp yields 3*p_perp for these species. If "
            "the producer instead omitted an all-zero parallel field the correct "
            "scalar is 2*p_perp (1.5x less); if the dd follows the IMAS data "
            "dictionary it is p_perp (3x less). Supply pressure_fast_parallel, or "
            "set p_fast_reduction explicitly.")
    else:
        detail = (
            f"Under p_fast_reduction={rule!r} the two fields are the full "
            "directional pressures, so the closure p_par := p_perp is the isotropic "
            "fast-ion assumption and the scalar is p_perp.")
    warnings.warn(
        f"core_profiles species [{who}] carry pressure_fast_perpendicular but no "
        f"pressure_fast_parallel. {detail}", stacklevel=3)


def _phi_n_from_rho(rho, psi_N, where):
    """Φ_N of nodes at ψ_N ``psi_N`` from their ``rho_tor_norm``: ρ², normalised
    to [0, 1].

    Refuses a missing ``rho`` or the ``sqrt(psi_N)`` placeholder some writers
    store: neither says where the nodes sit in Φ_N.
    """
    if rho is None or np.shape(rho) != np.shape(psi_N):
        raise ValueError(f"coord='phi_n' needs {where} rho_tor_norm "
                         "on the same nodes as its psi")
    rho = np.asarray(rho, dtype=float)
    if not np.all(np.diff(rho) > 0):
        raise ValueError(f"coord='phi_n': {where} rho_tor_norm is not strictly increasing")
    if np.allclose(rho, np.sqrt(np.clip(psi_N, 0.0, None)), rtol=0, atol=1e-6):
        raise ValueError(f"coord='phi_n': {where} rho_tor_norm is the sqrt(psi_N) "
                         "placeholder, not a toroidal-flux coordinate")
    phi = rho ** 2
    return (phi - phi[0]) / (phi[-1] - phi[0])


def _dd_phi_n(cp, psi_N):
    """Φ_N of the core_profiles nodes (:func:`_phi_n_from_rho` of ``grid``)."""
    return _phi_n_from_rho(cp.get("grid", {}).get("rho_tor_norm"), psi_N,
                           "core_profiles grid")


def _override(arr, src_psi, dst_psi):
    """Resample a user-supplied fixed-component array onto the baseline grid."""
    arr = np.asarray(arr, dtype=float)
    if src_psi is None:
        if arr.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the baseline grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return arr
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), arr)


def source_current_sign(ip) -> float:
    """``+1.0`` / ``-1.0``: the factor that brings a dd's currents into bouquet's frame.

    bouquet solves every anchor to ``|Ip|`` with ``F0 = |r0*b0|``, i.e. in a
    positive-current frame, whatever orientation the source was written in.
    A source with ``ip < 0`` (a reversed-current discharge in the dd's own
    COCOS) has every current profile multiplied by ``-1`` on read; ``ip >= 0``
    (and a zero or non-finite ``ip``, which carries no orientation) returns
    ``+1.0`` and leaves the read untouched.
    """
    ip = float(ip)
    return -1.0 if (np.isfinite(ip) and ip < 0.0) else 1.0


#: Where the reader's current-orientation factor came from (recorded on the
#: Baseline as ``source_current_sign_origin``, in li_metrics / ip_closure and
#: on the archive's ``_baseline`` attrs).
ORIENTATION_ORIGIN_AUTO = "auto: sign(equilibrium ip)"
ORIENTATION_ORIGIN_OVERRIDE = "override: ImasSource.current_orientation"


def parse_current_orientation(setting, what="ImasSource.current_orientation"):
    """Validate ``ImasSource.current_orientation``: ``"auto"``, ``+1.0`` or ``-1.0``.

    (Also the ``current_orientation`` of
    :func:`bouquet.coil_targets.measured_from_pf_active`; *what* names the
    setting in the error message.)

    Accepts ``"auto"`` (any case), the numbers ``1`` / ``-1`` (int or float)
    and their string spellings (``"+1"``, ``"-1"``).  Anything else -- ``0``,
    ``True``, ``2``, ``"reversed"`` -- raises :class:`ValueError`: the setting
    names a sign, and a value that is not one is a typo, not a request.
    """
    v = None
    if isinstance(setting, str):
        s = setting.strip().lower()
        if s == "auto":
            return "auto"
        try:
            v = float(s)
        except ValueError:
            v = None
    elif not isinstance(setting, bool):
        try:
            v = float(setting)
        except (TypeError, ValueError):
            v = None
    if v not in (1.0, -1.0):
        raise ValueError(
            f"{what} must be 'auto' (default: "
            "sign(equilibrium ip), refusing a dd whose current profiles "
            "disagree with it), +1 or -1 (the factor that brings this dd's "
            f"currents into bouquet's positive-Ip frame); got {setting!r}")
    return v


def _area_measure(x_psiN, own=None, eq_p1=None, psiN_eq=None):
    """``(x, label)``: the best available cumulative-area coordinate on a grid.

    ``integral(j dx)`` over the returned ``x`` is the plasma current (or a
    positive multiple of it), so its SIGN is the sign of the net toroidal
    current -- which an unweighted ``integral(j dpsi_N)`` only approximates
    when ``j`` changes sign.  In order of preference:

      1. the grid's own ``area`` [m^2] (IMAS ``core_profiles.grid.area`` /
         ``equilibrium.profiles_1d.area``) -- exact;
      2. the equilibrium ``profiles_1d.area`` interpolated in psi_N -- exact
         up to interpolation;
      3. ``rho_tor_norm**2`` -- a proxy: the toroidal flux enclosed is ~ B0
         times the enclosed area, exact for a uniform toroidal field;
      4. ``psi_N`` -- no geometry at all (the previous heuristic).

    Every candidate is monotone in the enclosed area, so a single-signed
    profile -- the only kind a whole-profile orientation mismatch produces --
    integrates to the same sign under all four; they can differ only for a
    profile with a genuine sign reversal of comparable weight, and the label
    is quoted in any refusal so the user can judge that case.
    """
    n = np.asarray(x_psiN).size

    def _ok(a):
        if a is None:
            return None
        try:
            a = np.asarray(a, dtype=float)
        except (TypeError, ValueError):
            return None
        return a if (a.shape == (n,) and np.all(np.isfinite(a))
                     and np.ptp(a) > 0.0) else None

    own = own or {}
    a = _ok(own.get("area"))
    if a is not None:
        return a, "area-weighted (own grid area)"
    if eq_p1 is not None and psiN_eq is not None and "area" in eq_p1:
        try:
            ae = np.asarray(eq_p1["area"], dtype=float)
            o = np.argsort(psiN_eq)
            a = _ok(np.interp(x_psiN, np.asarray(psiN_eq)[o], ae[o]))
        except (TypeError, ValueError):
            a = None
        if a is not None:
            return a, "area-weighted (equilibrium profiles_1d.area)"
    r = _ok(own.get("rho_tor_norm"))
    if r is not None:
        return r ** 2, "rho_tor_norm^2-weighted (area proxy; no area on file)"
    return np.asarray(x_psiN, dtype=float), "psi_N-weighted (no geometry on file)"


def _net_current(j, x):
    """``integral(j dx)`` with ``x`` taken in ascending order (storage order,
    axis-first or boundary-first, must not flip the sign)."""
    from scipy.integrate import trapezoid
    j = np.asarray(j, dtype=float)
    x = np.asarray(x, dtype=float)
    o = np.argsort(x, kind="stable")
    return float(trapezoid(j[o], x[o]))


def _refuse_mixed_orientation(bad, ip, cur_sign, origin):
    """Raise for currents that disagree in sign with the chosen orientation.

    ``bad`` lists ``(quantity, raw_net, weighting)``: the quantity's net
    toroidal current AS STORED in the dd (before the factor) and how it was
    weighted.
    """
    lines = [f"  - {q}: net {v:+.4g} (sign {'+' if v > 0 else '-'}; {w})"
             for q, v, w in bad]
    if origin == ORIENTATION_ORIGIN_AUTO:
        how = (f"the factor is sign(equilibrium ip) = {cur_sign:+.0f} "
               "(ImasSource.current_orientation='auto')")
    else:
        how = (f"the factor is {cur_sign:+.0f}, set by "
               "ImasSource.current_orientation")
    raise ValueError(
        "IMAS current orientation: the dd's current profiles disagree in sign "
        f"with its plasma current. equilibrium ip = {float(ip):+.6g} A (sign "
        f"{'+' if float(ip) >= 0 else '-'}); {how}, but after multiplying by "
        "it these would integrate AGAINST the positive-Ip frame bouquet "
        "solves in:\n" + "\n".join(lines) + "\n"
        "Closing Ip on this would add the recomputed (positive) bootstrap "
        "against the inductive current. Typical cause: the IDSs were written "
        "in different COCOS/orientations. If you know this file's current "
        "convention, set ImasSource.current_orientation to the factor that "
        "makes its currents co-Ip (+1 keeps them as stored, -1 reverses "
        "them); if only the equilibrium j_tor disagrees, "
        "GenerationConfig.anchor_jtor_to_equilibrium=False stops it being "
        "used. Otherwise fix the file.")


#: equilibrium.profiles_1d fields the current conversions need (IMAS.jl names).
_FUSE_GEOM_FIELDS = ("rho_tor_norm", "f", "gm1", "gm5", "gm8", "gm9",
                     "dpressure_dpsi")
#: ``li_metrics["imas_current_conversion"]["method"]`` of an IMAS read: the
#: exact conversion on the paired equilibrium geometry (A5-A7), or the
#: per-surface ratio ``c = j_tor/j_total`` the reader falls back to (with a
#: warning) when that geometry is absent.
IMAS_CURRENT_EXACT = "exact (A5-A7, paired equilibrium geometry)"
IMAS_CURRENT_RATIO_FALLBACK = ("ratio c = j_tor/j_total (fallback: the "
                               "equilibrium geometry is absent)")


def _fsa_from_profiles_2d(ts, nlevels=257):
    """gm1, gm5, gm8, gm9 of an equilibrium slice on its profiles_1d psi, by
    flux-surface tracing of the rectangular ``profiles_2d.psi`` (COCOS 11).

    For producers that omit the averages; FUSE writes them.  Against FUSE's own
    (rt50, 65x65 psi): median 1e-4, <3e-3 inside psi_N 0.95, ~2 % at the
    separatrix; the converted currents agree to 2e-4 of their peak.
    """
    from .geqdsk import GEQDSKEquilibrium
    p1 = ts["profiles_1d"]
    p2 = next((p for p in ts.get("profiles_2d") or []
               if (p.get("grid_type") or {}).get("index") == 1 and p.get("psi")),
              None)
    if p2 is None:
        raise ValueError("no rectangular profiles_2d.psi to compute them from")
    if not p1.get("f"):
        raise ValueError("profiles_1d.f is needed to compute <B^2>")
    R = np.asarray(p2["grid"]["dim1"], dtype=float)
    Z = np.asarray(p2["grid"]["dim2"], dtype=float)
    for name, x in (("dim1", R), ("dim2", Z)):
        if not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-6, atol=0):
            raise ValueError(f"profiles_2d grid {name} is not uniform")
    # traced with psi rising from axis to edge: a mirrored slice traces the
    # same arrays (the averages do not depend on the psi sign)
    sg = -1.0 if float(p1["psi"][-1]) < float(p1["psi"][0]) else 1.0
    psi = sg * np.asarray(p1["psi"], dtype=float)
    pn = (psi - psi[0]) / (psi[-1] - psi[0])
    pn_u = np.linspace(0.0, 1.0, R.size)

    def on_u(key, s=1.0):               # geqdsk 1-D profiles: uniform psi_N, NW
        v = p1.get(key)
        return (s * np.interp(pn_u, pn, np.asarray(v, dtype=float)) if v
                else np.zeros(R.size))

    ax = ts["global_quantities"]["magnetic_axis"]
    ob = (ts.get("boundary") or {}).get("outline") or {}
    rb = np.asarray(ob.get("r", []), dtype=float)
    zb = np.asarray(ob.get("z", []), dtype=float)
    raw = dict(NW=R.size, NH=Z.size, RLEFT=R[0], RDIM=R[-1] - R[0],
               ZMID=0.5 * (Z[0] + Z[-1]), ZDIM=Z[-1] - Z[0],
               SIMAG=psi[0], SIBRY=psi[-1], RMAXIS=float(ax["r"]), ZMAXIS=float(ax["z"]),
               FPOL=on_u("f"), PRES=on_u("pressure"),
               PPRIME=on_u("dpressure_dpsi", sg), FFPRIM=on_u("f_df_dpsi", sg),
               QPSI=on_u("q"),
               PSIRZ=sg * np.asarray(p2["psi"], dtype=float).T,  # [R][Z] -> [Z][R]
               RBBBS=rb, ZBBBS=zb, RLIM=rb, ZLIM=zb,
               CURRENT=0.0, RCENTR=float(ax["r"]), BCENTR=0.0)
    avg = GEQDSKEquilibrium.from_raw(raw, cocos=11, nlevels=nlevels).averages
    pn_l = np.linspace(0.0, 1.0, nlevels)
    return {gm: np.interp(pn, pn_l, avg[key]) for gm, key in
            (("gm1", "1/R**2"), ("gm5", "Btot**2"), ("gm8", "R"), ("gm9", "1/R"))}


def _derive_geom_fields(ts, missing):
    """``missing`` derivable profiles_1d fields of an equilibrium slice:
    rho_tor_norm from phi (or q), dpressure_dpsi from pressure, gm's from
    profiles_2d (:func:`_fsa_from_profiles_2d`)."""
    from scipy.interpolate import CubicSpline
    p1 = ts["profiles_1d"]
    # splines in sg*psi, rising from axis to edge in either orientation
    sg = -1.0 if float(p1["psi"][-1]) < float(p1["psi"][0]) else 1.0
    psi = sg * np.asarray(p1["psi"], dtype=float)
    out = {}
    if "rho_tor_norm" in missing:
        if p1.get("phi"):
            phi = np.asarray(p1["phi"], dtype=float)
        elif p1.get("q"):
            phi = CubicSpline(psi, np.asarray(p1["q"], dtype=float)).antiderivative()(psi)
            phi = phi - phi[0]
        else:
            raise ValueError("rho_tor_norm needs profiles_1d phi or q")
        out["rho_tor_norm"] = np.sqrt(np.abs(phi / phi[-1]))
    if "dpressure_dpsi" in missing:
        if not p1.get("pressure"):
            raise ValueError("dpressure_dpsi needs profiles_1d pressure")
        out["dpressure_dpsi"] = sg * CubicSpline(
            psi, np.asarray(p1["pressure"], dtype=float)).derivative()(psi)
    if any(g in missing for g in ("gm1", "gm5", "gm8", "gm9")):
        out.update(_fsa_from_profiles_2d(ts))
    return out


def _slice_b0(eq, k):
    """Signed ``equilibrium.vacuum_toroidal_field.b0`` at slice ``k``."""
    b0 = eq["vacuum_toroidal_field"]["b0"]
    if isinstance(b0, list):
        return float(b0[min(k, len(b0) - 1)])
    return float(b0)


def _imasjl_cubic(x, y, xq):
    """IMAS.jl ``cubic_interp1d``: FastInterpolations cubic spline with the
    default ``CubicFit`` ends (end slopes from the cubic through the 4 end
    points), extended beyond the ends.  Matches FUSE's ``core_profiles.j_tor``
    to machine precision where a natural spline misses by ~5e-3 at the edge."""
    from scipy.interpolate import CubicSpline
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def slope(xs, ys, x0):
        return np.polyfit(xs - x0, ys, 3)[-2]

    bc = ((1, slope(x[:4], y[:4], x[0])), (1, slope(x[-4:], y[-4:], x[-1])))
    return CubicSpline(x, y, bc_type=bc)(np.asarray(xq, dtype=float))


def _fuse_current_geometry(eq, k, rho=None):
    """Current-convention ``geom`` (:mod:`bouquet.physics`) of FUSE equilibrium
    slice ``k``, on its own grid (``rho=None``) or interpolated in rho_tor_norm
    onto ``rho`` exactly as IMAS.jl ``JparB_2_JtoR`` does (:func:`_imasjl_cubic`).

    COCOS 11: p' = -2*pi*dpressure_dpsi; ``B0`` is the slice's signed b0, so
    ``geom`` takes IMAS <J.B>/B0 directly.  Fields a producer omitted are
    derived (:func:`_derive_geom_fields`).
    """
    ts = eq["time_slice"][k]
    p1 = ts["profiles_1d"]
    missing = [f for f in _FUSE_GEOM_FIELDS if not p1.get(f)]
    get = {f: np.asarray(p1[f], dtype=float)
           for f in _FUSE_GEOM_FIELDS if f not in missing}
    if missing:
        try:
            if "f" in missing:
                raise ValueError("f is not derivable")
            get.update(_derive_geom_fields(ts, missing))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise ValueError(
                f"equilibrium.time_slice[{k}].profiles_1d lacks {missing} "
                f"({exc}): the IMAS -> TokaMaker current conversion needs f and "
                "the flux-surface averages gm1=<1/R^2>, gm5=<B^2>, gm8=<R>, "
                "gm9=<1/R> (FUSE writes them), or a rectangular profiles_2d.psi "
                "with magnetic_axis to compute them from") from exc
    if rho is None:
        at = get
    else:
        x = get["rho_tor_norm"]
        o = np.argsort(x)
        at = {f: _imasjl_cubic(x[o], v[o], rho)
              for f, v in get.items() if f != "rho_tor_norm"}
    return {"F": at["f"], "avg_R": at["gm8"], "avg_inv_R": at["gm9"],
            "avg_inv_R2": at["gm1"], "avg_B2": at["gm5"],
            "pprime": -2.0 * np.pi * at["dpressure_dpsi"],
            "B0": _slice_b0(eq, k)}


def _jtor_from_jpar(j_par, geom):
    """IMAS ``j_tor`` of a total/bootstrap <J.B>/B0 (A6, IMAS.jl
    ``Jpar_2_Jtor(..., includes_bootstrap=true)``)."""
    return jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(j_par, geom)
        + jphi_tokamaker_pressure_term(geom), geom)


def paired_equilibrium_candidates(eq_times, t_cp):
    """Indices of the ``equilibrium`` slices :func:`_paired_current_geometry`
    considers for the core_profiles slice at ``t_cp`` [s]: the slice nearest
    ``t_cp``, then the last one strictly before it (FUSE's pairing on a
    time-dependent run) when that is another slice.  The one rule shared by
    the reader and the single-slice export cut (:func:`_slice_in_time`,
    which keeps exactly these slices so a cut re-reads identically)."""
    te = np.asarray(eq_times, dtype=float)
    k_near = int(np.argmin(np.abs(te - t_cp)))
    cands = [k_near]
    before = np.nonzero(te < t_cp - 1e-9 * max(1.0, abs(t_cp)))[0]
    if before.size and int(before[-1]) != k_near:
        cands.append(int(before[-1]))
    return cands


def _paired_current_geometry(eq, cp, t_cp, j_total=None, j_tor=None):
    """``(geom, meta)`` on the core_profiles grid from the equilibrium slice
    FUSE paired with the core_profiles slice at ``t_cp``.

    FUSE evaluates ``core_profiles.j_tor`` (IMAS.jl ``Jpar_2_Jtor``) against
    the equilibrium slice current at that time, which on a time-dependent run
    is the PREVIOUS slice (measured: t_cp - dt_eq on every slice of a
    FUSE D3D run).  Candidates are the nearest slice and the last one before
    ``t_cp``; the one whose geometry reproduces ``j_tor`` from ``j_total`` (A6)
    wins.  ``meta`` records the choice and its mismatch (median relative).
    """
    te = np.asarray(eq["time"], dtype=float)
    if not cp["grid"].get("rho_tor_norm"):
        raise ValueError("core_profiles grid lacks rho_tor_norm, the "
                         "coordinate the IMAS current conversion uses")
    rho = np.asarray(cp["grid"]["rho_tor_norm"], dtype=float)
    cands = paired_equilibrium_candidates(te, t_cp)
    best, err = None, None
    for k in cands:
        try:
            geom = _fuse_current_geometry(eq, k, rho)
        except ValueError as exc:
            err = exc
            continue
        mis = np.nan
        if j_total is not None and j_tor is not None:
            jt = np.asarray(j_tor, dtype=float)
            ok = np.abs(jt) > 1e-3 * np.max(np.abs(jt))
            if np.any(ok):
                mis = float(np.median(np.abs(
                    _jtor_from_jpar(j_total, geom)[ok] / jt[ok] - 1.0)))
        if best is None or (np.isfinite(mis) and not mis >= best[2]):
            best = (k, geom, mis)
    if best is None:
        raise err
    k, geom, mis = best
    return geom, {"index": k, "time": float(te[k]), "t_core_profiles": float(t_cp),
                  "jtor_mismatch": mis}


def current_frame(eq, cp, t_cp, cur_sign, ip_signed):
    """``(m, s, geom, meta)``: the frame the dd's currents are converted in.

    The conversions run in the frame of the dd's geometry (its signed F, B0
    and p'): ``m * current`` is in it, and ``s * converted`` is in bouquet's
    positive-current frame.  A consistent dd (and every ``"auto"`` read):
    ``m = 1``, ``s = cur_sign = sign(ip)``.  An orientation factor
    ``cur_sign`` that disagrees with sign(ip) leaves two readings -- ip alone
    stored reversed (``m = 1``, ``s = cur_sign``) or the currents alone
    (``m = -1``, ``s = sign(ip)``) -- and the one whose ``j_total``
    reproduces ``j_tor`` on the geometry (A6, :func:`_paired_current_geometry`)
    is taken.  ``geom, meta``: that geometry and its pairing record."""
    jt = np.asarray(cp["j_total"], dtype=float)
    jtor = np.asarray(cp["j_tor"], dtype=float)
    s_ip = source_current_sign(ip_signed)
    geom, meta = _paired_current_geometry(eq, cp, t_cp, jt, jtor)
    if cur_sign == s_ip:
        return 1.0, s_ip, geom, meta
    g2, meta2 = _paired_current_geometry(eq, cp, t_cp, -jt, -jtor)
    if meta2["jtor_mismatch"] < meta["jtor_mismatch"]:
        return -1.0, s_ip, g2, meta2                 # the currents reversed
    return 1.0, float(cur_sign), geom, meta          # ip reversed


def read_imas_geometry(source: "ImasSource"):
    """Return ``(F0, boundary_RZ)`` from a FUSE IDS for TokaMaker setup.

    ``F0 = |r0 * b0(t)|`` from ``equilibrium.vacuum_toroidal_field`` and the LCFS
    isoflux points from ``equilibrium.time_slice[t].boundary.outline``. Used by
    :meth:`Bouquet.setup_solver` when the source is an :class:`ImasSource`
    (replacing the g-file that the reconstruction path reads F0/boundary from).
    """
    dd = _load_dd(source.ids_path)
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], source.time, "equilibrium")
    vtf = eq["vacuum_toroidal_field"]
    r0 = float(vtf["r0"])
    b0 = vtf["b0"]
    b0v = float(b0[ie]) if isinstance(b0, list) else float(b0)
    F0 = abs(r0 * b0v)
    # Separatrix: an optional external g-file (if given) supplies the LCFS;
    # otherwise use the dd boundary outline. F0 stays from the dd vacuum field
    # either way (same shot).
    lcfs_g = getattr(source, "LCFS_geqdsk", None)
    if lcfs_g:
        from .geqdsk import read_geqdsk
        g = read_geqdsk(lcfs_g)
        boundary_RZ = np.column_stack([
            np.asarray(g.boundary_R, dtype=float),
            np.asarray(g.boundary_Z, dtype=float),
        ])
    else:
        out = eq["time_slice"][ie]["boundary"]["outline"]
        boundary_RZ = np.column_stack([
            np.asarray(out["r"], dtype=float), np.asarray(out["z"], dtype=float),
        ])
    return F0, boundary_RZ


def _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                    p_equilibrium, allow_incomplete,
                                    anchor_pressure=False, p_fast_meta=None):
    """Fail-fast when the IMAS source carries pressure the reconstruction omits.

    Hard fails (raise, or warn if ``allow_incomplete``) on DROPPED reconstructable
    components -- a thermal species or the fast channel:
      1. species  -- e*(ne*Te + ni*Ti) + impurity(nz*Ti) vs e*(ne*Te + Sum n_s*T_s)
      2. fast     -- any species carries pressure_fast_* but assembled p_fast ~ 0
    Plus a non-fatal backstop (warning only):
      3. reconstructed total vs equilibrium.pressure -- the equilibrium routinely
         carries more (anisotropic/beam + equilibrium-vs-transport gap) which the
         p_diff anchor absorbs; informational, not blocking.
    """
    problems = []
    # 1. species completeness (single-impurity model vs full Sum_s n_s T_s)
    full_thermal = _EC * ne * te
    for ion in cp["ion"]:
        full_thermal = full_thermal + _EC * (
            np.asarray(ion["density_thermal"], dtype=float)
            * np.asarray(ion["temperature"], dtype=float))
    recon_thermal = _EC * (ne * te + ni * ti) + p_imp
    denom = float(np.mean(np.abs(full_thermal))) or 1.0
    sp_gap = float(np.mean(np.abs(full_thermal - recon_thermal))) / denom
    if sp_gap > 0.02:
        problems.append(
            f"thermal species gap {sp_gap*100:.1f}% (>2%): single-impurity model "
            f"(Z_imp from Zeff, impurity at main-ion Ti) does not reproduce "
            f"Sum_s(n_s*T_s) over {len(cp['ion'])} ion species -- multiple "
            f"impurities or T_imp != Ti.")
    # 2. fast completeness
    avail_fast = 0.0
    for sp in [cp["electrons"]] + cp["ion"]:
        if "pressure_fast_perpendicular" in sp:
            avail_fast = max(avail_fast, float(np.max(np.abs(
                np.asarray(sp["pressure_fast_perpendicular"], dtype=float)))))
    if avail_fast > 0.0 and float(np.max(np.abs(p_fast))) <= 1e-3 * avail_fast:
        problems.append(
            f"fast pressure present in source (max {avail_fast/1e3:.1f} kPa) but "
            f"assembled p_fast ~ 0 -- fast-ion channel dropped.")
    # 3. backstop vs authoritative equilibrium pressure -- WARNING only, not a
    #    hard fail. The equilibrium.pressure routinely differs from the
    #    core_profiles thermal+fast reconstruction (anisotropic/beam pressure that
    #    is large early in a beam-heavy discharge, plus the
    #    equilibrium-vs-transport inconsistency -- the pressure analogue of the
    #    j_tor gap). When anchor_pressure_to_equilibrium is ON, p_diff absorbs the
    #    residual exactly and it rides fixed; when it is OFF (the default) nothing
    #    absorbs it and the gap propagates straight into the baseline pressure, so
    #    the message must not promise otherwise. Dropped *reconstructable*
    #    thermal/fast components are the hard fails (#1/#2 above).
    #
    #    This is also the one guard that can catch a wrong fast-pressure
    #    convention, whose signature is a ~3x (or ~1/3x) p_fast -- so report the
    #    SIGNED direction and name the rule in force.
    recon_total = recon_thermal + p_fast
    pe = float(np.mean(np.abs(p_equilibrium))) or 1.0
    signed = float(np.mean(recon_total - p_equilibrium)) / pe
    bk = float(np.mean(np.abs(recon_total - p_equilibrium))) / pe
    if bk > 0.05:
        import warnings
        direction = "above" if signed > 0 else "below"
        if anchor_pressure:
            remedy = ("anchor_pressure_to_equilibrium is ON, so the p_diff anchor "
                      "absorbs this exactly (the baseline still matches the dd "
                      "equilibrium.pressure) and the residual rides fixed -- it is "
                      "not perturbed in the UQ.")
        else:
            remedy = ("anchor_pressure_to_equilibrium is OFF (the default), so "
                      "p_diff is None and NOTHING absorbs this -- the gap "
                      "propagates into the baseline pressure and every downstream "
                      "beta_N / W_MHD / p'. Set "
                      "GenerationConfig.anchor_pressure_to_equilibrium=True to "
                      "anchor it.")
        meta = p_fast_meta or {}
        conv = ""
        if meta.get("rule") is not None:
            conv = (f" Fast-pressure reduction in force: {meta['rule']!r} "
                    f"(chosen by {meta.get('basis', '?')}). A wrong "
                    "pressure_fast_* convention shows up here as a ~3x / ~1/3x "
                    "p_fast -- check it before the anchor.")
        warnings.warn(
            f"reconstructed total pressure is {abs(signed)*100:.1f}% {direction} "
            f"dd equilibrium.pressure (mean |gap| {bk*100:.1f}%). {remedy}{conv}")
    if problems:
        msg = ("IMAS pressure completeness check failed:\n  - "
               + "\n  - ".join(problems)
               + "\nThe diff anchor still matches the baseline to FUSE, but the "
               "per-draw thermal perturbation omits the flagged component(s). "
               "Set GenerationConfig.allow_incomplete_pressure=True to bypass "
               "(backend tests only).")
        if allow_incomplete:
            import warnings
            warnings.warn(msg)
        else:
            raise ValueError(msg)


def _read_ida_omega(path, time_s, psi_N, place=None):
    """IDA toroidal rotation (omega_tor_12C6) resampled onto psi_N; None if absent.
    Mirrors read_ida's nearest-time selection (IDA time is ms; ``time_s`` is s)
    and, on the ensemble layout, its sample mean and shared radial grid.
    ``place`` (IDA-grid array -> run nodes) replaces the psi_N interpolation."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "omega_tor_12C6" not in f:
                return None
            it = np.asarray(f["time"][:], dtype=float).ravel()   # ms
            tms = (time_s * 1e3) if time_s is not None else float(it[0])
            j = int(np.argmin(np.abs(it - tms)))
            om = np.asarray(f["omega_tor_12C6"][j], dtype=float)
            if om.ndim == 2:            # ensemble: (n_samples, n_radial)
                om = om.mean(axis=0)
                ipsi = np.asarray(f["psi_n"][j], dtype=float)[0]
            else:
                ipsi = np.asarray(f["psi_n"][:], dtype=float)
            return place(om) if place is not None else np.interp(psi_N, ipsi, om)
    except Exception as e:
        import warnings
        warnings.warn(f"_read_ida_omega: {path!r} rotation not read "
                      f"({type(e).__name__}: {e}); omega left to FUSE")
        return None


#: psi_N points of the advisory IDA-vs-dd total-ni check: interior (an edge
#: ratio reports the edge model) and the axis (where a beam peaks).  The last
#: point also bounds the core :func:`_dd_zeff` classifies Z_eff on.
NI_FAST_GATE_PSI_N = (0.0, 0.2, 0.4, 0.6, 0.8)

#: Relative IDA-vs-dd TOTAL ni disagreement, at any of
#: :data:`NI_FAST_GATE_PSI_N`, above which the subtraction warns.  Advisory
#: only: the subtraction runs regardless.
NI_FAST_RTOL = 1e-2

#: rho at which the dd's psi_N(rho_tor_norm) is compared with the LCFS
#: g-file's (FUSE holds profiles fixed in rho while solving its own equilibrium).
PSI_RHO_GATE_RHO = (0.2, 0.4, 0.6, 0.8, 0.9)

#: Largest |psi_N_dd - psi_N_g| at :data:`PSI_RHO_GATE_RHO` before the reader
#: warns.  Advisory only.
PSI_RHO_DRIFT_TOL = 2e-2


def _psi_rho_drift(psi_N, rho, gfile):
    """The dd's psi_N(rho) against the g-file's, at :data:`PSI_RHO_GATE_RHO`.

    Returns a dict: ``gate`` (rho -> psi_N_dd - psi_N_g), ``max_abs``,
    ``rho_worst``, ``exceeds`` and a one-line ``evidence``; ``None`` without a
    usable rho grid.
    """
    from .geqdsk import read_geqdsk
    rho = np.asarray(rho, dtype=float)
    if rho.shape != np.shape(psi_N) or not np.all(np.diff(rho) > 0):
        return None
    if np.allclose(rho, np.sqrt(np.clip(psi_N, 0.0, None)), rtol=0, atol=1e-6):
        # a placeholder grid, not the dd's equilibrium: nothing to compare
        return {"gate": None, "max_abs": None, "rho_worst": None, "gfile": str(gfile),
                "exceeds": False, "placeholder": True,
                "evidence": "dd rho_tor_norm is the sqrt(psi_N) placeholder; not compared"}
    g = read_geqdsk(gfile)
    rho_g = np.asarray(g.rhovn, dtype=float)
    psi_g = np.linspace(0.0, 1.0, rho_g.size)
    pts = np.asarray(PSI_RHO_GATE_RHO, dtype=float)
    d = np.interp(pts, rho, psi_N) - np.interp(pts, rho_g, psi_g)
    iw = int(np.argmax(np.abs(d)))
    return {"gate": dict(zip(PSI_RHO_GATE_RHO, d.tolist())),
            "max_abs": float(abs(d[iw])), "rho_worst": float(pts[iw]),
            "gfile": str(gfile), "exceeds": float(abs(d[iw])) > PSI_RHO_DRIFT_TOL,
            "placeholder": False,
            "evidence": (
                f"dd psi_N(rho) differs from the g-file's by {d[iw]:+.3f} at "
                f"rho={pts[iw]:g} (psi_N {np.interp(pts[iw], rho, psi_N):.3f} "
                f"vs {np.interp(pts[iw], rho_g, psi_g):.3f}; "
                f"tol {PSI_RHO_DRIFT_TOL:g})")}


#: Relative core mismatch above which a stored Z_eff matches neither numerator.
ZEFF_CONVENTION_RTOL = 1e-2


def _dd_zeff(cp, zeff_th, z2_fast, ne, psi_N, record=None):
    """``(zeff, includes_fast)``: the Z_eff the dd's own bootstrap consumed.

    A stored ``cp1d.zeff`` may or may not include the fast ions in its
    numerator (IMAS's expression does; FUSE may have stored it before the
    beam), so it is classified against both on the core (``psi_N <= 0.8``).

    *record* (a dict, optional) is filled with how the convention was
    decided (owner item E1, archived as ``li_metrics["zeff_dd_provenance"]``):
    ``convention`` ("thermal+fast" / "thermal-only"), ``source`` ("stored
    cp.zeff" / "no stored zeff"), ``basis`` and, where both numerators were
    compared, the core misfits ``d_th`` / ``d_all`` and ``rtol``.
    """
    rec = {} if record is None else record
    if "zeff" not in cp:
        zeff_all = zeff_th + z2_fast / np.clip(ne, 1e-30, None)
        inc = bool(np.any(z2_fast))
        rec.update(convention="thermal+fast" if inc else "thermal-only",
                   source="no stored zeff",
                   basis=("recomputed from the dd's densities, fast ions "
                          "included in the numerator" if inc else
                          "recomputed from the dd's densities (no fast ions)"),
                   d_th=None, d_all=None)
        return zeff_all, inc
    stored = np.asarray(cp["zeff"], dtype=float)
    if not np.any(z2_fast):
        rec.update(convention="thermal-only", source="stored cp.zeff",
                   basis="no fast-ion population: the numerators coincide",
                   d_th=None, d_all=None)
        return stored, False
    zeff_all = zeff_th + z2_fast / np.clip(ne, 1e-30, None)
    core = np.asarray(psi_N, dtype=float) <= NI_FAST_GATE_PSI_N[-1]
    _s = np.abs(stored[core])
    d_th = float(np.max(np.abs(stored - zeff_th)[core] / _s))
    d_all = float(np.max(np.abs(stored - zeff_all)[core] / _s))
    includes = d_all < d_th
    rec.update(convention="thermal+fast" if includes else "thermal-only",
               source="stored cp.zeff",
               basis=("the closer of the two numerators on the core psi_N <= "
                      f"{NI_FAST_GATE_PSI_N[-1]:g}"),
               d_th=d_th, d_all=d_all, rtol=float(ZEFF_CONVENTION_RTOL),
               matches_neither=bool(min(d_th, d_all) > ZEFF_CONVENTION_RTOL))
    if min(d_th, d_all) > ZEFF_CONVENTION_RTOL:
        import warnings
        warnings.warn(
            f"core_profiles.zeff matches neither Z_eff numerator on the core "
            f"(thermal-only {d_th:.1e}, thermal+fast {d_all:.1e}); taking the "
            f"closer ({'thermal+fast' if includes else 'thermal-only'}). A zeff "
            f"stored before the dd's ne or ion densities were last changed "
            f"does this.")
    return stored, includes


def _subtract_fast_ni(psi_N, ni, ni_fuse_thermal, z_fast, z2_fast, impurity_Z):
    """Thermal ``(ni, meta)`` from a MEASURED (total) ni.

    Neither VB Z_eff nor CER carbon tells a beam ion from a thermal one, and
    everything downstream takes ni as thermal, so the subtraction of
    :func:`fast_ion_density_equivalent` is unconditional; ``sigma_ni`` keeps
    its absolute error (the removed density is a FUSE quantity with no IDA
    error).  Advisory cross-check: the IDA ni against the dd's total ni at
    :data:`NI_FAST_GATE_PSI_N`; beyond :data:`NI_FAST_RTOL` it warns.
    ``meta``: ``applied``, ``agrees``, ``gate`` (per-point deviations),
    ``evidence``.
    """
    ni = np.asarray(ni, dtype=float)
    ni_fast = fast_ion_density_equivalent(z_fast, z2_fast, impurity_Z)
    if not np.any(ni_fast):
        return ni, {"applied": False, "agrees": None, "mismatch": None,
                    "gate": None, "evidence": "dd carries no fast-ion population"}
    ni_total = np.asarray(ni_fuse_thermal, dtype=float) + ni_fast
    pts = np.asarray(NI_FAST_GATE_PSI_N, dtype=float)
    a = np.interp(pts, np.asarray(psi_N, dtype=float), ni)
    b = np.interp(pts, np.asarray(psi_N, dtype=float), ni_total)
    gate = mismatch = agrees = None
    if np.all(np.abs(b) > 0.0):
        gate = dict(zip(NI_FAST_GATE_PSI_N, (np.abs(a - b) / np.abs(b)).tolist()))
        mismatch = float(max(gate.values()))
        agrees = mismatch <= NI_FAST_RTOL
    ni_th = np.maximum(ni - ni_fast, 0.0)
    if agrees is None:
        evidence = ("dd main-ion density vanishes at a check point; "
                    "cross-check skipped")
    elif agrees:
        evidence = (f"IDA ni matches the dd TOTAL ni to {mismatch:.2e} over "
                    f"psi_N={NI_FAST_GATE_PSI_N}")
    else:
        worst = max(gate, key=gate.get)
        evidence = (f"IDA ni differs from the dd TOTAL ni by {mismatch:.2e} "
                    f"at psi_N={worst:g} (> {NI_FAST_RTOL:.0e}): the beam "
                    f"density comes from a plasma that is not the IDA one")
    return ni_th, {
        "applied": True, "agrees": agrees, "mismatch": mismatch, "gate": gate,
        "fast_fraction_peak": float(np.max(ni_fast / np.maximum(ni, 1e-30))),
        "evidence": evidence}


#: The time rule of the ida_hybrid IDA slice (#73 review; the source-time
#: rule of the core_sources reads applied to the IDA file's own time base),
#: stamped on every record.
IDA_TIME_RULE = (
    "IDA slice: nearest own slice to the requested time (ImasSource.ida_time, "
    "else the dd slice time), never interpolated, accepted within half the "
    "IDA file's local time-step (a single-slice IDA file: half the local "
    "core_profiles step when paired with the dd time, the 1e-05 s floor "
    "IMAS_SINGLE_TIME_WINDOW_S for an explicit ida_time); paired with the dd "
    "time (ida_time None) it must also lie within half the local "
    "core_profiles step of the core_profiles slice read; else refused; dt "
    "recorded")


def match_ida_slice(ida_times, t_req, *, what="IDA file"):
    """The IDA slice a SEPARATE IDA file (``UncertaintyConfig.ida_path``) is
    read at, by the IDA time rule of #73 (:data:`IDA_TIME_RULE`): the nearest
    own slice to *t_req* [s], never interpolated, accepted within half the
    file's local time-step (a single-slice file: the
    :data:`IMAS_SINGLE_TIME_WINDOW_S` floor -- there is no dd step to pair
    with), else REFUSED (``ValueError``).  Returns the match record
    (``rule``, requested / used time, ``dt``, the window and its basis, the
    number of slices).  *t_req* None: only a single-slice file is
    accepted (its one slice, ``dt`` None)."""
    tt = np.asarray(ida_times, dtype=float).ravel()
    if tt.size == 0:
        raise ValueError(f"{what}: the IDA file has an empty time base")
    rec = dict(rule=IDA_TIME_RULE, what=what, ida_time_requested=t_req,
               ida_n_times=int(tt.size))
    if t_req is None:
        if tt.size > 1:
            raise ValueError(
                f"{what}: the IDA file holds {tt.size} slices and no time was "
                "given (ImasSource.ida_time / the source's time)")
        rec.update(ida_time_used=float(tt[0]), dt=None, half_window=None,
                   window_basis="single-slice IDA file, no time requested")
        return rec
    t_req = float(t_req)
    k = int(np.argmin(np.abs(tt - t_req)))
    t_used = float(tt[k])
    dt = t_used - t_req
    grid = np.unique(tt)
    if grid.size >= 2:
        kg = int(np.argmin(np.abs(grid - t_used)))
        half = 0.5 * _local_step(grid, kg, t_req)
        basis = "half the IDA file's local time-step"
    else:
        half = IMAS_SINGLE_TIME_WINDOW_S
        basis = "single-slice IDA file: the floor IMAS_SINGLE_TIME_WINDOW_S"
    rec.update(ida_time_used=t_used, dt=dt, half_window=float(half),
               window_basis=basis)
    if abs(dt) > half:
        span = (f"{tt.min():.9g}-{tt.max():.9g} s" if tt.size > 1
                else f"{tt[0]:.9g} s")
        raise ValueError(
            f"{what}: no IDA slice within {basis} ({half:.3g} s) of the "
            f"requested time {t_req:.9g} s (nearest {t_used:.9g} s, |dt| = "
            f"{abs(dt):.3g} s; the file holds {span}).  Refusing rather than "
            "reading the sigmas at another time (never interpolated)")
    return rec


def _hybrid_timing(source, T, t_eq, t_cp, aux, tol=1e-6, *, cp_times=None,
                   ida_times=None):
    """ida_hybrid: the IDA slice time to read, matched by :data:`IDA_TIME_RULE`.

    Records the dd slice times on ``aux`` (``fuse_time_cp``,
    ``fuse_time_eq``) and the match on ``aux["ida_time_match"]`` (requested
    and used times, ``dt``, the window and its basis, the offset from the
    core_profiles slice), which the reader archives in ``li_metrics``.
    Warns when ``T`` is not a slice both core_profiles and equilibrium hold
    exactly (a FUSE macro step): j_bootstrap is then not from the slice asked
    for.  ``ida_times`` [s] defaults to the time base of
    ``source.ida_path``; ``cp_times`` is the dd's core_profiles time base.

    Raises ``ValueError`` when no IDA slice lies within the window -- the
    kinetics are never read at another time silently."""
    import warnings
    t_eq, t_cp = float(t_eq), float(t_cp)
    aux["fuse_time_cp"], aux["fuse_time_eq"] = t_cp, t_eq
    if T is not None and (abs(t_cp - T) > tol or abs(t_eq - t_cp) > tol):
        warnings.warn(f"ida_hybrid: time={T} s is not a dd macro step (core_profiles "
                      f"{t_cp} s, equilibrium {t_eq} s); its j_bootstrap was not computed "
                      f"on the IDA slice it is paired with")
    t_cfg = getattr(source, "ida_time", None)
    explicit = t_cfg is not None
    t_req = (float(t_cfg) if explicit else
             float(T) if T is not None else t_cp)
    if ida_times is None:
        from .ida import ida_time_base
        ida_times = ida_time_base(source.ida_path)
    tt = np.asarray(ida_times, dtype=float).ravel()
    if tt.size == 0:
        raise ValueError("ida_hybrid: the IDA file has an empty time base")
    k = int(np.argmin(np.abs(tt - t_req)))
    t_used = float(tt[k])
    dt = t_used - t_req
    grid = np.unique(tt)
    half_dd = _half_local_step(cp_times, t_cp, t_used)
    if grid.size >= 2:
        kg = int(np.argmin(np.abs(grid - t_used)))
        half = 0.5 * _local_step(grid, kg, t_req)
        basis = "half the IDA file's local time-step"
    elif not explicit and half_dd is not None:
        half = half_dd
        basis = ("single-slice IDA file: half the local core_profiles "
                 "time-step")
    else:
        half = IMAS_SINGLE_TIME_WINDOW_S
        basis = "single-slice IDA file: the floor IMAS_SINGLE_TIME_WINDOW_S"
    rec = dict(rule=IDA_TIME_RULE, ida_time_configured=t_cfg,
               ida_time_requested=t_req, ida_time_used=t_used, dt=dt,
               half_window=float(half), window_basis=basis,
               ida_n_times=int(tt.size), fuse_time_cp=t_cp, fuse_time_eq=t_eq,
               dd_offset=t_used - t_cp,
               dd_half_window=None if half_dd is None else float(half_dd),
               pairing_consistent=None, replayed_ida_time=None,
               pairing_table=None)
    aux["ida_time_match"] = rec
    span = f"{tt.min():.9g}-{tt.max():.9g} s" if tt.size > 1 else f"{tt[0]:.9g} s"
    if abs(dt) > half:
        raise ValueError(
            f"ida_hybrid: no IDA slice within {basis} ({half:.3g} s) of the "
            f"requested {'ida_time' if explicit else 'time'} {t_req:.9g} s "
            f"(nearest IDA slice {t_used:.9g} s, |dt| = {abs(dt):.3g} s; the "
            f"IDA file holds {span}).  Refusing rather than reading the "
            "kinetics at another time (never interpolated)")
    if not explicit:
        hd = half_dd if half_dd is not None else (
            half if grid.size >= 2 else IMAS_SINGLE_TIME_WINDOW_S)
        if abs(t_used - t_cp) > hd:
            raise ValueError(
                f"ida_hybrid: the IDA slice {t_used:.9g} s is "
                f"{abs(t_used - t_cp):.3g} s from the core_profiles slice "
                f"read ({t_cp:.9g} s), more than half its local time-step "
                f"({hd:.3g} s): the kinetics would not belong to the dd slice "
                "they are paired with.  Refusing; set ImasSource.ida_time to "
                "pair this dd slice with that IDA slice deliberately")
    elif half_dd is not None and abs(t_used - t_cp) > 2.0 * half_dd:
        warnings.warn(
            f"ida_hybrid: ida_time {t_used:.9g} s is "
            f"{abs(t_used - t_cp):.3g} s from the dd slice {t_cp:.9g} s, more "
            "than one local dd time-step: check the pairing (recorded as "
            "ida_time_match.dd_offset)")
    return t_used


def _check_replay_pairing(ids_path, aux, tol=1e-6, *, ida_path=None):
    """``aux['pairing_consistent']``: whether FUSE's own replay_pairing
    (``ida_provenance.json`` beside ``ids_path``, written by the external
    IDA_fuse tooling) says dd j_bootstrap at ``aux['fuse_time_cp']`` was
    computed on ``aux['ida_time_used']``; None when there is no table, when
    it is unreadable, or when it does not describe this run.  Read only,
    never derived, never raises.

    The table is trusted only when it is BOUND to this run: its
    ``ida_file`` must be this run's IDA file (*ida_path*; same real path, or
    the same file name when the table was written on another host), its
    ``sim_times`` (when present) must hold the dd slice time, and an
    optional ``dd_sha256`` must be the dd's own.  A directory match alone is
    not enough (a stale table from another run would otherwise answer).
    Rows lacking ``outcome`` (or other keys) are tolerated.  The binding
    and the verdict are written to ``aux['ida_time_match']`` too."""
    import json
    import warnings
    aux["pairing_consistent"] = None
    rec = aux.get("ida_time_match")
    table = dict(file="ida_provenance.json", status="absent", bound_by=None)
    if rec is not None:
        rec["pairing_table"] = table
    path = os.path.join(os.path.dirname(os.path.abspath(ids_path)), "ida_provenance.json")
    try:
        with open(path) as fh:
            prov = json.load(fh)
    except OSError:
        return
    except ValueError:
        table["status"] = "unreadable"
        return
    if not isinstance(prov, dict):
        table["status"] = "unreadable"
        return
    rows = prov.get("replay_pairing")
    if not rows:
        table["status"] = "no replay_pairing"
        return
    t_cp = aux.get("fuse_time_cp")
    # bind the table to this run: the IDA file, the dd slice, the dd itself
    why = None
    ida_file = prov.get("ida_file")
    if ida_file is None or ida_path is None:
        why = "the table names no ida_file" if ida_file is None else \
            "no IDA file to compare"
    elif os.path.realpath(str(ida_file)) == os.path.realpath(str(ida_path)):
        table["bound_by"] = "ida_file path"
    elif os.path.basename(str(ida_file)) == os.path.basename(str(ida_path)):
        table["bound_by"] = "ida_file name"
    else:
        why = (f"its ida_file {os.path.basename(str(ida_file))!r} is not this "
               f"run's {os.path.basename(str(ida_path))!r}")
    sim = prov.get("sim_times")
    if why is None and sim is not None and t_cp is not None:
        try:
            ok_t = any(abs(float(t) - t_cp) <= tol for t in sim)
        except (TypeError, ValueError):
            ok_t = False
        if not ok_t:
            why = f"its sim_times do not hold the dd slice {t_cp:.9g} s"
    if why is None and prov.get("dd_sha256"):
        if _sha256_file(ids_path) != str(prov["dd_sha256"]):
            why = "its dd_sha256 is not this dd's"
        else:
            table["bound_by"] += " + dd_sha256"
    if why is not None:
        table["status"] = f"not bound to this run: {why}"
        warnings.warn(f"ida_hybrid: {path} does not describe this run ({why}); "
                      "the replay pairing is not checked (pairing_consistent=None)")
        return
    good = []
    for r in rows:
        try:
            good.append((abs(float(r["t_sim"]) - t_cp), r))
        except (KeyError, TypeError, ValueError):
            continue
    if not good or t_cp is None:
        table["status"] = "no usable rows"
        return
    d, row = min(good, key=lambda x: x[0])
    if d > tol:
        table["status"] = "no row at the dd slice"
        return
    r_ida = row.get("ida_time")
    try:
        ok = r_ida is not None and abs(float(r_ida) - aux["ida_time_used"]) <= tol
    except (TypeError, ValueError):
        ok = False
    table["status"] = "checked"
    aux["pairing_consistent"] = ok
    aux["replayed_ida_time"] = r_ida
    if rec is not None:
        rec.update(pairing_consistent=ok, replayed_ida_time=r_ida,
                   replay_outcome=row.get("outcome"))
    if not ok:
        warnings.warn(f"ida_hybrid: dd j_bootstrap at {t_cp} s was computed on "
                      f"IDA {r_ida} ({row.get('outcome', 'outcome not recorded')}), "
                      f"not the IDA slice read ({aux['ida_time_used']} s)")


@functools.lru_cache(maxsize=4)
def _sha256_cached(path, _mtime_ns, _size):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path):
    p = os.path.realpath(path)
    st = os.stat(p)
    return _sha256_cached(p, st.st_mtime_ns, st.st_size)


def _merge_ida_kinetics(psi_N, ne_fuse, ni_fuse, Zeff_fuse, ida_path, time, impurity_Z,
                         ni_source="all", zeff_from_fuse=False,
                         z_fast=None, z2_fast=None, x_phi=None):
    """IDA-hybrid kinetics: replace FUSE ne/ni/Te/Ti/Zeff (+omega) with IDA fits,
    resampled onto the FUSE ``psi_N`` grid (psi_N == psi_N_kinetic).

    Zeff and ni both default to IDA: Zeff measured directly (``zeff_from_fuse=True``
    keeps the FUSE Zeff instead); ni via ``ni_source`` ("Zeff"/"CER"/"all", with
    Jacobian-propagated sigma_ni -- see :func:`bouquet.io.ida.read_ida`). The
    "Zeff"/"all" dilution always uses IDA's own Zeff regardless of
    ``zeff_from_fuse``.

    With the fast charge moments ``z_fast``/``z2_fast`` the IDA TOTAL ni is
    always converted to a THERMAL one; see :func:`_subtract_fast_ni`.

    Returns ``(ne, te, ti, ni, zeff, omega_or_None, sigma_ne, sigma_te, sigma_ni,
    sigma_ti, ida, ni_fast_meta)`` on ``psi_N``; ``ida`` is handed back so
    ``resolve_uncertainty`` reuses this slice instead of re-reading the file.
    ``x_phi`` (the run nodes' Φ_N) places the IDA fits by their Φ_N from the
    file's own q (:func:`bouquet.coords.phi_n_from_q`); the map
    ``(ida psi_N, ida Φ_N)`` is a 13th element (``None`` without ``x_phi``).
    """
    from .ida import read_ida
    ida = read_ida(ida_path, time=time, impurity_Z=impurity_Z, ni_source=ni_source)
    _ipsi = np.asarray(ida.psi_N, dtype=float)
    ida_map = None
    if x_phi is None:
        g = lambda a: np.interp(psi_N, _ipsi, np.asarray(a, dtype=float))
    else:
        if ida.q is None:
            raise ValueError(f"coord='phi_n': {ida_path!r} carries no q, so its "
                             "profiles cannot be placed in Phi_N")
        from ..coords import phi_n_from_q
        _in, _iphi = phi_n_from_q(_ipsi, ida.q, bracket=True)
        _n = int(np.count_nonzero(_ipsi <= 1.0))     # psi_map: inside the LCFS only
        ida_map = (_ipsi[_in][:_n], _iphi[:_n])
        g = lambda a: np.interp(x_phi, _iphi, np.asarray(a, dtype=float)[_in])
    ne, te, ti = g(ida.ne), g(ida.te), g(ida.ti)
    zeff = np.asarray(Zeff_fuse, dtype=float) if zeff_from_fuse else g(ida.Zeff)
    # read_ida's ni is main_ion_density_from_zeff of its (ne, Z_eff): rebuilt
    # from the interpolated pair so it stays quasineutral between IDA's nodes.
    ni = main_ion_density_from_zeff(
        ne, np.clip(g(ida.Zeff), 1.0, impurity_Z), impurity_Z)
    sigma_ne, sigma_te, sigma_ni, sigma_ti = (
        g(ida.sigma_ne), g(ida.sigma_te), g(ida.sigma_ni), g(ida.sigma_ti))
    # The ni_source ni is a TOTAL deuteron density; everything downstream
    # takes ni as thermal once the dd carries a beam.
    if z_fast is not None:
        ni, ni_fast_meta = _subtract_fast_ni(
            psi_N, ni, np.asarray(ni_fuse, dtype=float),
            z_fast, z2_fast, impurity_Z)
    else:
        ni_fast_meta = {"applied": False, "agrees": None, "mismatch": None,
                        "gate": None,
                        "evidence": "no fast-ion charge moments supplied"}
    omega = _read_ida_omega(ida_path, time, psi_N, place=None if x_phi is None else g)
    return (ne, te, ti, ni, zeff, omega, sigma_ne, sigma_te, sigma_ni,
            sigma_ti, ida, ni_fast_meta, ida_map)


def read_imas_baseline(
    source: "ImasSource",
    fixed: Optional["FixedComponentsConfig"] = None,
    p_fast_reduction: str = "auto",
    allow_incomplete_pressure: bool = False,
    anchor_jtor_to_equilibrium: bool = True,
    kinetic_source: str = "fuse",
    anchor_pressure_to_equilibrium: bool = False,
) -> "Baseline":
    """Read a FUSE ``dd_sim.json`` IDS and return a separated :class:`Baseline`.

    No Grad-Shafranov reconstruction is performed -- provenance is "imas".

    ``p_fast_reduction`` defaults to ``"auto"``: the fast-pressure storage
    convention is taken from the dd's own recorded provenance
    (:func:`resolve_p_fast_reduction`), with a loud warning if it cannot be
    determined.  An explicit ``"sum"`` / ``"trace"`` / ``"mean"`` / ``"perp"``
    always wins and is applied silently.  The rule that was used, and how it was
    chosen, are recorded on :attr:`Baseline.p_fast_meta`.

    Every driven ``core_sources`` current is held fixed, classified as the
    engine's IDS adapter does (:func:`bouquet.adapters._ids_driven_currents`):
    beams in ``j_NBI``, EC/LH/IC in ``j_RF``, fusion, runaways, sawteeth and
    unknown indices in ``j_other`` (its sawteeth share also in ``j_sawteeth``).
    Aggregate and bootstrap-like entries are never added.
    """
    from ..baseline import Baseline

    dd = _load_dd(source.ids_path)
    T = source.time

    # Which fast-pressure convention this dd was written in (factor of 3).
    # The metadata is resolved eagerly -- the reduction rule is needed to read the
    # fields and the completeness message quotes it -- but the loud undetermined
    # warning is held back until we know the choice actually moved a number: it
    # says nothing on a dd whose fast pressure is absent or identically zero, and
    # nothing on a run whose p_fast comes from FixedComponentsConfig instead.
    p_fast_meta = resolve_p_fast_reduction(dd, p_fast_reduction, warn=False)
    p_fast_rule = p_fast_meta["rule"]

    # --- targets from the equilibrium IDS ---
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], T, "equilibrium")
    gq = eq["time_slice"][ie]["global_quantities"]
    Ip_target = abs(float(gq["ip"]))
    ids_li_1 = float(gq["li_1"]) if "li_1" in gq else None
    ids_li_3 = float(gq["li_3"]) if "li_3" in gq else None
    # Provisional l_i_target = IDS li_3; the IMAS forward-solve (Bouquet) replaces
    # this with the TokaMaker-solved li_1 and records the IDS values for sanity.
    l_i_target = ids_li_3 if ids_li_3 is not None else 0.0

    # --- profiles + currents from core_profiles ---
    cp_ids = dd["core_profiles"]
    ic = _nearest_index(cp_ids["time"], T, "core_profiles")
    cp = cp_ids["profiles_1d"][ic]

    # --- current orientation ---------------------------------------------
    # bouquet's frame is positive-current (the anchor is solved to |Ip| with
    # F0 = |r0*b0|).  Every CURRENT profile read below is multiplied by this
    # factor so a reversed-current dd (ip < 0) lands in that frame instead of
    # being combined, with its own negative sign, with a bootstrap recomputed
    # on the positive anchor.  +1.0 (bit-identical read) for ip >= 0.
    # The sign is read at the equilibrium slice nearest the core_profiles
    # slice the currents come from (orientation_slice_index; the two IDSs
    # choose their slices independently; with a common time base, as in FUSE
    # output, this is the slice ``ie`` above).  ImasSource.current_orientation = +1/-1 replaces
    # it; either way the currents are checked against it further down.
    ie_s = orientation_slice_index(dd, T)
    ip_signed = float(eq["time_slice"][ie_s]["global_quantities"].get(
        "ip", gq["ip"]))
    _orient = parse_current_orientation(
        getattr(source, "current_orientation", "auto"))
    if _orient == "auto":
        cur_sign = source_current_sign(ip_signed)
        cur_origin = ORIENTATION_ORIGIN_AUTO
    else:
        cur_sign = float(_orient)
        cur_origin = ORIENTATION_ORIGIN_OVERRIDE
    # B0 orientation (recorded only), same slice.  A zero or unreadable b0
    # carries no orientation: None, not +1.
    _vtf = eq.get("vacuum_toroidal_field") or {}
    _b0 = _vtf.get("b0")
    try:
        _b0v = float(_b0[ie_s] if isinstance(_b0, list) else _b0)
        b0_sign = ((-1.0 if _b0v < 0.0 else 1.0)
                   if (np.isfinite(_b0v) and _b0v != 0.0) else None)
    except (TypeError, ValueError, IndexError):
        b0_sign = None

    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])   # 0 (axis) -> 1 (boundary)
    n = psi_N.size
    # Run grid: the same nodes, labelled in the run coordinate.  Every dd
    # profile below is read on the nodes' psi_N; a toroidal-flux run only
    # relabels them.  IDA fits are placed by their own Phi_N in such a run.
    from .. import coords as _coords
    coord = _coords.run_coord(getattr(source, "coord", _coords.PSI))
    x_run = psi_N if coord == _coords.PSI else _dd_phi_n(cp, psi_N)

    # The conversions below run in the frame of the dd's geometry: the
    # currents times _m, the results times s_ip (current_frame).  A dd whose
    # equilibrium lacks the averages the exact conversion needs (f, gm1, gm5,
    # gm8, gm9, dpressure_dpsi -- or a profiles_2d to trace them -- and
    # core_profiles rho_tor_norm) falls back, loudly and stamped, to the
    # per-surface ratio method every reader used before PR #64 (review
    # PR64 B7: refusing it was an undeclared breaking input change).
    cur_conv = dict(method=IMAS_CURRENT_EXACT)
    try:
        _m, s_ip, cur_geom, cur_meta = current_frame(
            eq, cp, float(cp_ids["time"][ic]), cur_sign, ip_signed)
    except ValueError as _geom_exc:
        cur_geom = None
        _m, s_ip = 1.0, float(cur_sign)
        cur_meta = {"index": ie, "time": float(eq["time"][ie]),
                    "t_core_profiles": float(cp_ids["time"][ic]),
                    "jtor_mismatch": float("nan")}
        cur_conv = dict(method=IMAS_CURRENT_RATIO_FALLBACK,
                        reason=str(_geom_exc))
        import warnings
        warnings.warn(
            "IMAS reader: the exact IMAS -> TokaMaker current conversion "
            f"(A5-A7) is unavailable ({_geom_exc}); FALLING BACK to the "
            "per-surface ratio c = j_tor/j_total of the pre-PR #64 reader "
            "(j_phi = core_profiles.j_tor, no p'G split).  Recorded as "
            "li_metrics['imas_current_conversion'].", stacklevel=2)
    j_total = _m * np.asarray(cp["j_total"], dtype=float)     # total <J.B>/B0
    j_tor = _m * np.asarray(cp["j_tor"], dtype=float)         # total IMAS j_tor
    j_boot = _m * np.asarray(cp["j_bootstrap"], dtype=float)  # <J.B>/B0 (inductive = residual)

    if cur_geom is not None:
        # Exact conversions to TokaMaker jphi on the geometry FUSE used for
        # this core_profiles slice (see _paired_current_geometry).  The
        # pressure-driven p'G is its own bucket (owner decision D2): never in
        # j_BS; recorded as Baseline.j_pressure.
        p_term = jphi_tokamaker_pressure_term(cur_geom)

        def to_jphi(j_par):
            return jpar_to_jphi_tokamaker(j_par, cur_geom)
    else:
        p_term = None

        def to_jphi(j_par):
            return parallel_to_toroidal(j_par, j_parallel_total=j_total,
                                        j_tor_total=j_tor)
    j_BS = to_jphi(j_boot)

    # --- driven currents: every core_sources entry, then convert ---
    # Each driven entry is read at the core_sources slice TIME, not at its list
    # index (owner-approved 2026-10-05, the sawteeth entry's rule below and
    # the engine IDS adapter's): an entry carrying its own per-slice times is
    # matched to its NEAREST own slice -- one that starts later than the IDS
    # time base was read one slice late, and a slice past its end from its
    # FIRST slice.  The match is accepted within HALF the entry's local
    # time-step (the core_profiles step for a single-time entry); otherwise
    # the read is REFUSED (owner-approved 2026-10-06: before, a 1e-6 s
    # absolute match dropped the beam to zero with a warning on any larger
    # mismatch) -- unless the entry carries no current on its own slices
    # bracketing the time: then it is off there, not missing (refinement of
    # 2026-10-06).  One without per-slice times must have the IDS's slice
    # count, or it cannot be aligned and is refused.
    # Owner decision 2026-10-06: the core_sources slice is the one nearest
    # the core_profiles slice READ and must lie within half the local
    # core_profiles step of it (core_sources_slice; a single-time
    # core_sources is no longer read at any time); an entry's matched own
    # slice must also lie within that half-step of the core_profiles time;
    # an entry whose own record starts AFTER the slice time is OFF there
    # (off_before_record, announced once); dt and the bracketing own times
    # are recorded (Baseline.source_time_match).
    src_ids = dd.get("core_sources", {})
    isrc, _src_t, _slice_rec = core_sources_slice(
        src_ids, cp_ids.get("time"), ic, T, who="IMAS reader")
    _src_tb = src_ids.get("time")
    _src_nt = None if not _src_tb else len(_src_tb)
    _t_cp = _slice_rec["core_profiles_time"]
    _cp_half = (None if (_t_cp is None or _src_t is None) else
                _cp_window(cp_ids.get("time"), _src_tb, _t_cp, _src_t)[0])
    source_time_match = dict(core_sources=_slice_rec, entries=[])
    # Every driven entry -- beams, EC/LH/IC, fusion, runaways, sawteeth,
    # unknown indices -- through the ONE source-time rule of the engine's IDS
    # adapter, with the arguments the engine passes (adapters.IdsAdapter.
    # read): the core_profiles slice time and half-step window, the off list,
    # the match records and the announcement key.  Refusals, off_idle /
    # off_before_record stamps and announcements are therefore identical on
    # the legacy reader and the engine (which runs this reader first).
    from ..adapters import EngineInputRefused, _ids_driven_currents
    _cpt = cp_ids.get("time")

    def _is_saw(s):
        return (s.get("identifier") or {}).get("index") == SAWTOOTH_SOURCE_INDEX
    hold_saw = bool(getattr(source, "hold_sawteeth", True))
    _held = (src_ids if hold_saw else dict(src_ids, source=[
        s for s in src_ids.get("source", []) if not _is_saw(s)]))
    _off = []
    _rule_kw = dict(t_cp=_t_cp, cp_half=_cp_half,
                    announce_key=str(source.ids_path), who="IMAS reader")
    _parts, _used, _ignored = _ids_driven_currents(
        _held, isrc, n, 1.0, _cpt, off=_off,
        matches=source_time_match["entries"], **_rule_kw)
    # the sawteeth share of j_other (already matched, refused or stamped
    # above: no records, no second announcement)
    _saw = (_ids_driven_currents(dict(src_ids, source=[
        s for s in src_ids.get("source", []) if _is_saw(s)]),
        isrc, n, 1.0, _cpt, **_rule_kw)[0]["other"]
        if hold_saw else np.zeros(n))
    source_time_match.update(driven_sources=_used, ignored_sources=_ignored,
                             off_sources=_off)
    source_time_match["sawteeth_hold"] = dict(
        held=hold_saw, setting="ImasSource.hold_sawteeth",
        how=("the sawteeth entry's j_parallel is held fixed in j_other "
             "(its share in j_sawteeth): FUSE's j_ohmic excludes it"
             if hold_saw else
             "opted out: the sawteeth entry is not held; its current stays "
             "in the residual j_inductive (the pre-#70 legacy split)"))
    j_NBI = s_ip * to_jphi(_m * _parts["nbi"])
    j_RF = s_ip * to_jphi(_m * _parts["rf"])
    j_other = s_ip * to_jphi(_m * _parts["other"])
    j_sawteeth = s_ip * to_jphi(_m * _saw)

    # --- sawtooth model presence/amplitude at this slice (gate input only) ----
    # Read here, from the same parsed dd as the currents above (the cached,
    # read-only _load_dd object), so no second pass over the file is needed.
    # "active" means the source EXISTS and carries a non-zero j_parallel at this
    # SLICE TIME: a declared-but-idle sawtooth source (all zeros before onset)
    # must NOT admit a ramp slice to the q0 pin.  The entry is read at the
    # core_sources slice TIME, not at the list index (owner-approved
    # 2026-10-05, the rule of the engine IDS adapter): a model's sawteeth
    # entry may start later than the IDS time base -- it was then read one
    # slice late, and at the last slice from its FIRST slice.
    sawtooth = {"source_index": SAWTOOTH_SOURCE_INDEX, "present": False,
                "j_par_max_abs": 0.0, "active": False, "q0_dd": None}
    for s in src_ids.get("source", []):
        if s.get("identifier", {}).get("index") == SAWTOOTH_SOURCE_INDEX:
            sawtooth["present"] = True
            pr = s.get("profiles_1d", [])
            if pr:
                # held (the default): its match is already recorded above
                _erec = {}
                if not hold_saw:
                    source_time_match["entries"].append(_erec)
                q_saw, how = _source_slice_at(s, isrc, _src_t, _src_nt,
                                              cp_ids.get("time"), t_cp=_t_cp,
                                              cp_half=_cp_half, rec=_erec)
                sawtooth["slice"] = how
                if q_saw is None:
                    _erec["reason"] = how
                    # no slice of the entry within half a step of this
                    # time: not active here (a gate FLAG, not a current --
                    # recorded in sawtooth["slice"], not refused)
                    continue
                jsaw = np.asarray(q_saw.get("j_parallel", []), dtype=float)
                if jsaw.size and np.any(np.isfinite(jsaw)):
                    sawtooth["j_par_max_abs"] = max(
                        sawtooth["j_par_max_abs"],
                        float(np.nanmax(np.abs(jsaw))))
    sawtooth["active"] = bool(sawtooth["present"]
                              and sawtooth["j_par_max_abs"] > 0.0)

    # --- kinetic profiles + Zeff + fast pressure ---
    el = cp["electrons"]
    ne = np.asarray(el["density_thermal"], dtype=float)
    te = np.asarray(el["temperature"], dtype=float)   # eV
    _no_par = []
    p_fast = _isotropic_fast_pressure(el, p_fast_rule, n, _no_par, "electrons")

    ni = None
    ti = None
    main_ion = None
    zeff_num = np.zeros(n)
    # The two charge moments of the fast population.  Quasineutrality weights
    # each fast species by Z_s, a measured Z_eff's numerator by Z_s^2, so both
    # are needed and neither implies the other (see physics.
    # fast_ion_density_equivalent).  No beam charge is assumed anywhere.
    z_fast = np.zeros(n)          # sum_s Z_s   n_s^fast  [m^-3]
    z2_fast = np.zeros(n)         # sum_s Z_s^2 n_s^fast  [m^-3]
    # pressure_fast_* and density_fast are independent fields: a dd can carry
    # one without the other, and a species with fast pressure but no fast
    # density gets the full p_fast treatment and ZERO dilution correction.
    fast_p_no_n = []
    for ion in cp["ion"]:
        Z = float(ion["element"][0]["z_n"])
        n_s = np.asarray(ion["density_thermal"], dtype=float)
        zeff_num += n_s * Z * Z
        if "density_fast" in ion:
            _nf = np.asarray(ion["density_fast"], dtype=float)
            z_fast += Z * _nf
            z2_fast += Z * Z * _nf
        p_fast_s = _isotropic_fast_pressure(
            ion, p_fast_rule, n, _no_par, str(ion.get("label", f"Z={Z:g}")))
        if np.any(p_fast_s) and not np.any(
                np.asarray(ion.get("density_fast", 0.0), dtype=float)):
            fast_p_no_n.append(f"Z={Z:g}")
        p_fast = p_fast + p_fast_s
        if Z == 1.0 and ni is None:        # main (hydrogenic) ion
            ni = n_s
            ti = np.asarray(ion["temperature"], dtype=float)
            main_ion = ion
    if ni is None:
        raise ValueError("no hydrogenic (Z=1) main ion found in core_profiles.ion")
    _warn_missing_parallel(_no_par, p_fast_rule)
    if fast_p_no_n:
        # Silence is the dangerous case here: the run looks exactly like an
        # ohmic one while Z_imp / nz / p_imp keep the whole fast-ion bias the
        # dilution correction exists to remove.
        import warnings
        warnings.warn(
            "core_profiles carries fast-ion PRESSURE but no density_fast for "
            f"ion species [{', '.join(fast_p_no_n)}]: p_fast is applied in "
            "full while the fast-ion dilution correction for those species is "
            "zero, so Z_imp / nz / p_imp retain the fast-ion bias. Fill "
            "core_profiles.ion[].density_fast to enable the correction.")
    # Thermal-only numerator over the full ne: the convention
    # impurity_charge_with_fast_ions inverts, built here from the dd's own
    # densities so it holds by construction.
    Zeff_th = zeff_num / ne
    # The dd's bootstrap Z_eff and its convention (see _dd_zeff).  Zeff is its
    # recomputation from the densities -- bit-identical to Zeff_th without a
    # beam -- and is what the baseline solve and the draws are handed.
    _zeff_dd_rec = {}
    zeff_dd, dd_zeff_includes_fast = _dd_zeff(cp, Zeff_th, z2_fast, ne, psi_N,
                                              record=_zeff_dd_rec)
    Zeff = (Zeff_th + z2_fast / np.clip(ne, 1e-30, None)
            if dd_zeff_includes_fast else Zeff_th)

    # --- auxiliary source-provided profiles for the switchboard ---------------
    # Read whatever this source carries (production FUSE files have rotation;
    # chi/E_r are typically absent and supplied via aux_baselines). All on the
    # core_profiles grid (== psi_N_kinetic for IMAS).
    aux = {"zeff": zeff_dd}
    if main_ion is not None and "rotation_frequency_tor" in main_ion:
        aux["omega_tor"] = np.asarray(main_ion["rotation_frequency_tor"], dtype=float)
    if "e_field" in cp and "radial" in cp["e_field"]:
        aux["e_r"] = np.asarray(cp["e_field"]["radial"], dtype=float)
    ctids = dd.get("core_transport")
    if ctids and ctids.get("model"):
        cpr = ctids["model"][0].get("profiles_1d", [])
        ict = (_nearest_index(ctids["time"], T, "core_transport")
               if ctids.get("time") else ic)
        if cpr:
            ctsl = cpr[ict if len(cpr) > ict else 0]
            ce = ctsl.get("electrons", {}).get("energy", {})
            if "d" in ce:
                aux["chi_e"] = np.asarray(ce["d"], dtype=float)
            if "d" in ctsl.get("total_ion_energy", {}):
                aux["chi_i"] = np.asarray(ctsl["total_ion_energy"]["d"], dtype=float)

    # --- IDA-hybrid: swap FUSE ne/Te/Ti/Zeff/omega for externally-fit IDA profiles ---
    # ni via source.ni_source; Zeff from IDA unless source.zeff_from_fuse. Done
    # before the pressure block so p_recon/Z_imp/p_imp use the IDA kinetics.
    use_ida = bool(kinetic_source == "ida_hybrid" and getattr(source, "ida_path", None))
    if getattr(source, "ida_time", None) is not None and not use_ida:
        # #73 review B6: outside ida_hybrid ida_time would move only the
        # IDA sigmas (resolve_uncertainty), not the kinetics -- refused
        raise ValueError(
            f"ImasSource.ida_time={source.ida_time!r} is set but the kinetics "
            f"do not come from an IDA file (kinetic_source={kinetic_source!r}, "
            f"ida_path={getattr(source, 'ida_path', None)!r}); it times only "
            "the ida_hybrid IDA slice -- unset it, or use "
            "kinetic_source='ida_hybrid' with an ida_path")
    zeff_includes_fast = dd_zeff_includes_fast
    # Geometry guard: does the dd place its profiles where the g-file does?
    _drift = None
    if getattr(source, "LCFS_geqdsk", None) and "rho_tor_norm" in cp["grid"]:
        _drift = _psi_rho_drift(psi_N, cp["grid"]["rho_tor_norm"], source.LCFS_geqdsk)
        aux["psi_rho_drift"] = _drift
        if _drift is not None and _drift["exceeds"]:
            import warnings
            warnings.warn(
                f"{_drift['evidence']}: the dd's equilibrium is not the g-file's, "
                "so its profiles and sources sit at shifted psi_N"
                + (f" against the IDA kinetics (placed by IDA "
                   f"{'psi_N' if coord == _coords.PSI else 'Phi_N'})"
                   if use_ida else ""))
    if use_ida:
        T_ida = _hybrid_timing(source, T, eq["time"][ie], cp_ids["time"][ic], aux,
                               cp_times=_cp_time_grid(cp_ids))
        (ne, te, ti, ni, Zeff, _omega,
         sigma_ne_ida, sigma_te_ida, sigma_ni_ida, sigma_ti_ida,
         _ida_read, _ni_fast_meta, _ida_map) = _merge_ida_kinetics(
            psi_N, ne, ni, Zeff, source.ida_path, T_ida,
            getattr(source, "impurity_Z", 6.0),
            ni_source=getattr(source, "ni_source", "all"),
            zeff_from_fuse=getattr(source, "zeff_from_fuse", False),
            z_fast=z_fast, z2_fast=z2_fast,
            x_phi=None if coord == _coords.PSI else x_run)
        if _ni_fast_meta["agrees"] is False and _drift is not None and _drift["exceeds"]:
            _ni_fast_meta["evidence"] += (
                f"; likely the psi_N(rho) drift ({_drift['evidence']})")
        aux["ni_fast_meta"] = _ni_fast_meta
        # Loud: the subtraction moves ni, and a failed cross-check means the
        # beam density belongs to a plasma that is not quite the IDA one.
        if _ni_fast_meta["applied"]:
            print(f"  [ni] thermal ni: subtracted the dd fast-ion equivalent "
                  f"(peak fast fraction "
                  f"{_ni_fast_meta['fast_fraction_peak']:.1%}); "
                  f"{_ni_fast_meta['evidence']}")
            if _ni_fast_meta["agrees"] is False:
                import warnings
                warnings.warn(f"ida_hybrid: {_ni_fast_meta['evidence']}; "
                              f"subtracted anyway")
        if _omega is not None:
            aux["omega_tor"] = _omega
        aux["zeff"] = Zeff   # keep the switchboard's zeff baseline consistent
        # IDA's Z_eff is MEASURED, so its numerator counts the fast ions;
        # zeff_from_fuse carries the dd's own convention over with its value.
        if not getattr(source, "zeff_from_fuse", False):
            zeff_includes_fast = True
        # Read once, shared: resolve_uncertainty reuses this instead of
        # opening the same file again (and possibly at another slice).
        aux["ida_profiles"] = (str(source.ida_path), _ida_read)
        aux["ida_time_used"] = float(_ida_read.time)
        if aux["ida_time_used"] != T_ida:       # the matched slice, exactly
            raise RuntimeError(
                f"ida_hybrid: read_ida returned the slice at "
                f"{aux['ida_time_used']!r} s, not the matched {T_ida!r} s")
        _check_replay_pairing(source.ids_path, aux,
                              ida_path=source.ida_path)
        aux["sigma_ne_ida"] = sigma_ne_ida
        aux["sigma_te_ida"] = sigma_te_ida
        aux["sigma_ni_ida"] = sigma_ni_ida
        aux["sigma_ti_ida"] = sigma_ti_ida

    # --- user overrides for fixed additive components ---
    if fixed is not None:
        # fixed.psi_N given on psi_N ("psi_n") goes through the dd's own map
        _fx = _coords.to_run_grid(fixed.psi_N, getattr(fixed, "coord", "run"),
                                  None if coord == _coords.PSI else (psi_N, x_run))
        if fixed.p_fast is not None:
            p_fast = _override(fixed.p_fast, _fx, x_run)
            p_fast_meta = {**p_fast_meta, "rule": None, "basis": "user-override",
                           "evidence": "FixedComponentsConfig.p_fast supplied; the "
                                       "dd fast-pressure fields were not read"}
        # A user-supplied fixed current is defined in bouquet's positive-Ip
        # frame (co-current positive) -- the frame the g-file path has always
        # taken it in (baseline._resolve_fixed) -- so it is NOT multiplied by
        # the dd's orientation factor: the same array means the same physics
        # on both source paths and for either orientation of the source.
        if fixed.j_NBI is not None:
            j_NBI = _override(fixed.j_NBI, _fx, x_run)
        if fixed.j_RF is not None:
            j_RF = _override(fixed.j_RF, _fx, x_run)
        if getattr(fixed, "j_other", None) is not None:
            j_other = _override(fixed.j_other, _fx, x_run)
            j_sawteeth = np.zeros_like(j_other)   # no longer a known part of it

    # The deferred factor-of-3 warning: the convention was undeterminable AND the
    # fast pressure it scales is non-zero AND it came from the dd (a user-supplied
    # p_fast has already rewritten the basis to "user-override").
    if (p_fast_meta["basis"] == "undetermined-fallback"
            and float(np.max(np.abs(np.asarray(p_fast, dtype=float)))) > 0.0):
        warn_p_fast_undetermined(p_fast_meta["rule"])
        p_fast_meta = {**p_fast_meta, "warned": True}

    # Authoritative total (IMAS j_tor -> TokaMaker jphi, A5); inductive absorbs
    # the residual so the decomposition sums exactly.  Every conversion above
    # runs in the dd's own orientation (its signed F, B0 and p'); the results
    # are then put into bouquet's positive-current frame by s_ip (A5 and A7
    # are odd in a whole-dd reversal, so this is exact).
    if cur_geom is not None:
        j_phi_dd = jtor_imas_to_jphi_tokamaker(j_tor, cur_geom)
        _pk = float(np.max(np.abs(j_phi_dd)))
        _closure = float(np.max(np.abs(to_jphi(j_total) + p_term
                                       - j_phi_dd))) / _pk
    else:                       # the ratio fallback: j_tor as TokaMaker jphi
        j_phi_dd = j_tor.copy()
        _pk = float(np.max(np.abs(j_phi_dd)))
        _closure = float("nan")
    _dconv = (j_phi_dd - j_tor) / _pk
    j_phi = s_ip * j_phi_dd
    j_BS = s_ip * j_BS
    # The solve split carries p'G in the residual inductive (the solver needs
    # the total; the pre-PR #64 convention).  j_pressure records it as its
    # own bucket: the archive writes j_inductive - j_pressure and j_pressure
    # (schema.write_current_split), the IDS exporter groups it with the
    # non-inductive currents.
    j_pressure = None if p_term is None else s_ip * p_term
    j_inductive = j_phi - j_BS - j_NBI - j_RF - j_other
    if cur_geom is not None:
        # which equilibrium slice the currents were converted on (the
        # pairing, _paired_current_geometry) and how well it reproduced
        # j_tor: a record, so a re-read (e.g. of a single-slice export) can
        # be checked to pair the same way
        cur_conv.update(equilibrium_time=float(cur_meta["time"]),
                        core_profiles_time=float(cur_meta["t_core_profiles"]),
                        jtor_mismatch=float(cur_meta["jtor_mismatch"]))
    cur_conv.update(
        pressure=("j_pressure (p'G) carried by j_inductive in the solve split"
                  if j_pressure is not None else
                  "no p'G split (ratio fallback): carried by every component"),
        current_split_convention=SPLIT_PRESSURE_IN_INDUCTIVE)
    if cur_geom is None:
        print("  [imas] currents -> TokaMaker jphi by the RATIO FALLBACK "
              "c = j_tor/j_total (no equilibrium geometry; j_phi = j_tor)")
    else:
        print(f"  [imas] currents -> TokaMaker jphi on equilibrium t="
              f"{cur_meta['time']:.4f} s (core_profiles t="
              f"{cur_meta['t_core_profiles']:.4f} s; j_tor reproduced to "
              f"{cur_meta['jtor_mismatch']:.1e}); j_total closure "
              f"{_closure:.1e} of peak; jphi - j_tor: axis {_dconv[0]:+.2%}, "
              f"max {_dconv[np.argmax(np.abs(_dconv))]:+.2%} of peak")
    if cur_geom is not None and not cur_meta["jtor_mismatch"] <= 1e-3:
        import warnings
        warnings.warn(
            f"core_profiles.j_tor is not reproduced from j_total by any "
            f"candidate equilibrium slice (best t={cur_meta['time']:.4f} s, "
            f"median mismatch {cur_meta['jtor_mismatch']:.1e}); the current "
            "split may carry a geometry/time-pairing error")
    if cur_sign < 0.0:
        _why = (f"source ip = {ip_signed / 1e6:+.4f} MA < 0 (reversed current "
                "in the dd's own COCOS)" if cur_origin == ORIENTATION_ORIGIN_AUTO
                else "ImasSource.current_orientation = -1 (override; source ip "
                     f"= {ip_signed / 1e6:+.4f} MA)")
        print(f"[imas] {_why}: every dd current profile "
              "(j_tor, j_bootstrap, NBI j_parallel, equilibrium j_tor; each "
              "converted in the dd's own frame) multiplied by -1 "
              "into bouquet's positive-current frame (Baseline."
              "source_current_sign = -1); user-supplied FixedComponentsConfig "
              "j_NBI/j_RF are already co-Ip positive and are not", flush=True)
    elif cur_origin == ORIENTATION_ORIGIN_OVERRIDE and ip_signed < 0.0:
        print(f"[imas] ImasSource.current_orientation = +1 (override): dd "
              f"currents kept as stored although source ip = "
              f"{ip_signed / 1e6:+.4f} MA < 0", flush=True)

    # --- pressure anchor ("diff" approach) + completeness validation ----------
    # The authoritative dd equilibrium pressure (GS-consistent total, incl.
    # thermal impurity + isotropized fast) interpolated onto psi_N. The solve
    # builds e*(ne*Te + ni*Ti) + impurity(nz*Ti) + p_fast; p_diff anchors that to
    # FUSE exactly (mirrors jBS_diff) so the per-draw kinetics perturb the thermal
    # while the carbon/fast/GS residual rides as a fixed offset. Z_imp is the
    # single effective impurity charge (one-Zeff single-impurity model).
    eqp1 = eq["time_slice"][ie]["profiles_1d"]
    psi_eq = np.asarray(eqp1["psi"], dtype=float)
    psiN_eq = (psi_eq - psi_eq[0]) / (psi_eq[-1] - psi_eq[0])
    _o = np.argsort(psiN_eq)
    # Nodes of equilibrium.profiles_1d in the run coordinate: its own Φ_N
    # (rho_tor_norm², else from its q) in a toroidal-flux run.
    x_eq, x_at = psiN_eq[_o], psi_N
    if coord != _coords.PSI:
        _rho = eqp1.get("rho_tor_norm")
        if _rho is None and "q" in eqp1:
            x_eq = _coords.phi_n_from_q(x_eq, np.asarray(eqp1["q"], dtype=float)[_o])[1]
        else:
            if _rho is not None and np.size(_rho) == _o.size:
                _rho = np.asarray(_rho, dtype=float)[_o]
            x_eq = _phi_n_from_rho(_rho, x_eq, "equilibrium profiles_1d")
        x_at = x_run
    p_equilibrium = np.interp(x_at, x_eq,
                              np.asarray(eqp1["pressure"], dtype=float)[_o])
    # The dd's OWN axis q -- taken at the SMALLEST psi_N (via the same ordering
    # the pressure uses), not blindly at index 0, since profiles_1d need not be
    # stored axis-first.  Comparison metric only; see Baseline.sawtooth.
    if "q" in eqp1:
        _qdd = np.asarray(eqp1["q"], dtype=float)[_o]
        if _qdd.size and np.isfinite(_qdd[0]):
            sawtooth["q0_dd"] = float(_qdd[0])
    # Only ne - sum_s Z_s n_s^fast is neutralised by THERMAL ions; charging
    # the fast-ion share to the impurity inflates nz.  The helper also
    # renormalizes Zeff (defined over the FULL ne above) onto the thermal
    # electrons -- without that the inversion recovers only half the bias
    # (see impurity_charge_with_fast_ions).  The Zeff consumed by the
    # bootstrap / forward solve deliberately stays the full-ne one.
    Z_imp, ne_th = impurity_charge_with_fast_ions(ne, ni, Zeff_th, z_fast)
    if use_ida:
        # ni was built at source.impurity_Z (read_ida): that IS the impurity charge
        Z_imp = float(getattr(source, "impurity_Z", 6.0))
    p_imp = impurity_pressure(ne_th, ni, ti, Z_imp)
    p_recon = _EC * (ne * te + ni * ti) + p_imp + p_fast
    # p_diff anchors the solve thermal pressure to the FUSE equilibrium.pressure.
    # Gated OFF by default: with IDA-hybrid kinetics we trust IDA's pressure and do
    # NOT force it back onto the FUSE total. When None, no anchor offset is added.
    p_diff = (p_equilibrium - p_recon) if anchor_pressure_to_equilibrium else None
    # The completeness guard compares the FUSE thermal/fast pressure against the FUSE
    # equilibrium.pressure; it is meaningless once ne/Te/Ti are swapped to IDA.
    if kinetic_source != "ida_hybrid":
        _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                        p_equilibrium, allow_incomplete_pressure,
                                        anchor_pressure=anchor_pressure_to_equilibrium,
                                        p_fast_meta=p_fast_meta)

    # --- total-current anchor: equilibrium.j_tor vs core_profiles.j_tor --------
    # core_profiles.j_tor (-> j_phi here) is the transport parallel-current sum
    # (QED-diffused ohmic + Sauter bootstrap + NBI) converted to toroidal; it
    # differs from the GS-consistent equilibrium.j_tor (which GPEC reads) from
    # ~q=2 outward (the equilibrium carries more pedestal current). jphi_diff
    # anchors the total to the equilibrium as a FIXED offset (mirrors jBS_diff):
    # added to the baseline + every draw while the SWB bootstrap / perturbed
    # j_inductive ride underneath. jphi_diff integrates to ~0 (both totals carry
    # the same Ip), so it redistributes rather than adds net current.
    jphi_diff = None
    if anchor_jtor_to_equilibrium:
        # IMAS j_tor -> TokaMaker jphi on the slice's own grid (exact, in the
        # dd's frame), then onto the run nodes, in the positive frame.  The
        # ratio fallback takes j_tor as it is (the pre-PR #64 anchor).
        eq_jphi = _m * np.asarray(eqp1["j_tor"], dtype=float)
        if cur_geom is not None:
            eq_jphi = jtor_imas_to_jphi_tokamaker(
                eq_jphi, _fuse_current_geometry(eq, ie))
        eq_jtor = s_ip * np.interp(x_at, x_eq, eq_jphi[_o])
        jphi_diff = eq_jtor - j_phi

    # --- orientation consistency: REFUSE a mixed-sign source ---------------
    # Every toroidal current bouquet USES must carry, after the factor, the
    # sign of the positive frame: core_profiles.j_tor (the authoritative total
    # j_phi) always, and the equilibrium j_tor when the jphi anchor uses it.
    # The test is on the NET current -- the area-weighted integral (see
    # _area_measure for the weighting actually available on the file and its
    # fallbacks), taken on each quantity's own grid -- so a profile with a
    # genuine local sign reversal (a current hole, a counter-current edge) is
    # not refused as long as its net current is co-Ip.  A dd with no area on
    # file falls back to a monotone proxy for it; every proxy classifies a
    # single-signed profile -- the only kind a frame mismatch produces --
    # exactly as the true area weighting does.  The factor is +-1, so the
    # check is done on the stored arrays: sign(raw) * cur_sign < 0.
    _checks = []
    _x, _w = _area_measure(psi_N, own=cp.get("grid"), eq_p1=eqp1,
                           psiN_eq=psiN_eq)
    _checks.append(("core_profiles.j_tor",
                    _net_current(np.asarray(cp["j_tor"], dtype=float), _x), _w))
    if anchor_jtor_to_equilibrium:
        _xe, _we = _area_measure(psiN_eq, own=eqp1)
        _checks.append(("equilibrium.profiles_1d.j_tor",
                        _net_current(np.asarray(eqp1["j_tor"], dtype=float),
                                     _xe), _we))
    _bad = [c for c in _checks if cur_sign * c[1] < 0.0]
    if _bad:
        _refuse_mixed_orientation(_bad, ip_signed, cur_sign, cur_origin)

    bl = Baseline(
        psi_N=x_run,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=x_run,
        coord=coord,
        # IDA sigmas follow the IDA fits: through the file's own map when the
        # fits were placed by it, else through the dd's.
        psi_map=(None if coord == _coords.PSI else
                 _ida_map if use_ida else (psi_N, x_run)),
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="imas",
        j_NBI=j_NBI,
        j_RF=j_RF,
        j_other=j_other,
        j_sawteeth=j_sawteeth,
        p_fast=p_fast,
        z_fast=(z_fast if np.any(z_fast) else None),
        z2_fast=(z2_fast if np.any(z_fast) else None),
        zeff_includes_fast=zeff_includes_fast,
        p_equilibrium=p_equilibrium,
        p_diff=p_diff,
        Z_imp=Z_imp,
        jphi_diff=jphi_diff,
        eqdsk_bytes=None,
        # The OMAS JSON is NOT an Osborne p-file; don't pass it as pfile_bytes
        # (generate_bouquet would try to parse it). Per-draw p-files are built
        # from the kinetic profiles.
        pfile_bytes=None,
        li_metrics={"ids_li_1": ids_li_1, "ids_li_3": ids_li_3,
                    "imas_current_conversion": cur_conv,
                    # ida_hybrid: which IDA slice was paired with this dd
                    # slice and how (archived with li_metrics on every
                    # route, like source_time_match)
                    **({"ida_time_match": dict(aux["ida_time_match"])}
                       if "ida_time_match" in aux else {}),
                    # owner item E1: how the dd's own Z_eff numerator
                    # convention was decided (_dd_zeff), and whether the
                    # baseline's zeff_includes_fast follows it or the
                    # measured IDA Z_eff
                    # PR #56 (owner item E2 stamp): how the IDA Z_eff /
                    # n_i were resolved (rung, weights, window, clamps)
                    **({"zeff_provenance": dict(
                        aux["ida_profiles"][1].zeff_provenance)}
                       if "ida_profiles" in aux and getattr(
                           aux["ida_profiles"][1], "zeff_provenance", None)
                       else {}),
                    "zeff_dd_provenance": dict(
                        _zeff_dd_rec,
                        baseline_zeff_includes_fast=bool(zeff_includes_fast),
                        baseline_zeff_from=(
                            "dd" if (not use_ida or getattr(
                                source, "zeff_from_fuse", False))
                            else "IDA (measured; numerator counts the fast "
                                 "ions)"))},
        aux=aux,
        p_fast_meta=p_fast_meta,
        sawtooth=sawtooth,
        source_current_sign=cur_sign,
        source_current_sign_origin=cur_origin,
        source_b0_sign=b0_sign,
        source_time_match=source_time_match,
    )
    # the third current bucket (owner decision D2), beside the solve split
    # (which carries it in j_inductive; li_metrics["imas_current_conversion"])
    bl.j_pressure = j_pressure
    # the in-memory split keeps p'G in the residual j_inductive (the archive
    # writer takes it off: TokaMaker_interface.generate_bouquet)
    bl.current_split_convention = SPLIT_PRESSURE_IN_INDUCTIVE
    return bl


# ===========================================================================
#  Perturbed-draw IMAS/OMAS write-back
#
#  Current fidelity: bouquet arrays are TokaMaker jphi; core_profiles j_tor
#  (IMAS convention) and the parallel split j_total / j_bootstrap / j_ohmic
#  (<J.B>/B0) are converted exactly (physics module docstring) with the draw's
#  OWN flux-surface geometry, captured from the live TokaMaker equilibrium at
#  generate time (physics.capture_equilibrium_fsa -> the eq_fsa archive block),
#  for fidelity="exact"/"auto".  fidelity="reconstruct" (archives without a
#  complete capture) applies the same formulas with the TEMPLATE's baseline
#  equilibrium geometry -- exact only when the draw's geometry matches it.
#
#  Remaining refinement (not a blocker): the EQUILIBRIUM IDS profiles_2d still
#  come from the archived 257^2 eqdsk (lossless to that grid, machine-precision
#  GS) rather than the live FE fields -- a direct OFT ODS export would upgrade
#  this.
# ===========================================================================
def _imas_b0(out, ie, ic):
    """Reference vacuum field B0 for the IMAS <j.B>/B0 normalisation.

    Tries core_profiles then equilibrium ``vacuum_toroidal_field.b0`` (nearest
    time index); falls back to 1.0 (raw <j.B>) if neither is present.
    """
    for ids_name, it in (("core_profiles", ic), ("equilibrium", ie)):
        vtf = out.get(ids_name, {}).get("vacuum_toroidal_field")
        if vtf and vtf.get("b0") is not None:
            b0 = np.asarray(vtf["b0"], dtype=float)
            if b0.size:
                return abs(float(b0[min(it, b0.size - 1)]))
    return 1.0


#: eq_fsa keys of the draw's own kappa, and of its exact IMAS j_tor (A5;
#: captures before avg_R/pprime lack the last two).
_EQ_FSA_KAPPA_KEYS = ("F", "avg_inv_R", "avg_B2")
_EQ_FSA_GEOM_KEYS = _EQ_FSA_KAPPA_KEYS + ("avg_R", "avg_inv_R2", "pprime")


def _positive_frame_geom(geom, s_I):
    """A dd-frame current ``geom`` (:func:`_fuse_current_geometry`) in bouquet's
    positive-current frame: ``|F|``, ``|B0|`` and ``p'`` times ``s_I``
    (dp/dpsi flips with psi, i.e. with Ip)."""
    g = dict(geom)
    g["F"] = np.abs(np.asarray(g["F"], dtype=float))
    g["B0"] = abs(float(g.get("B0", 1.0)))
    g["pprime"] = float(s_I) * np.asarray(g["pprime"], dtype=float)
    return g


def _eq_fsa_geom_on(eq_fsa, psiN_t, B0):
    """Interpolate a captured eq_fsa block onto the template psi grid -> geom
    for the current-convention helpers (:mod:`bouquet.physics`), with the
    :data:`_EQ_FSA_GEOM_KEYS` it carries; ``None`` if it lacks any of
    :data:`_EQ_FSA_KAPPA_KEYS`."""
    if any(eq_fsa.get(k) is None for k in _EQ_FSA_KAPPA_KEYS):
        return None
    src = np.asarray(eq_fsa["psi_N"], dtype=float)
    geom = {k: np.interp(psiN_t, src, np.asarray(eq_fsa[k], dtype=float))
            for k in _EQ_FSA_GEOM_KEYS if eq_fsa.get(k) is not None}
    geom["B0"] = float(B0)
    return geom


#: COCOS of the eqdsk bytes bouquet archives per draw (TokaMaker's
#: ``save_eqdsk``): psi decreasing outward for Ip > 0, per radian.
ARCHIVE_EQDSK_COCOS = 7

_P_TERM_CACHE = {}


def archived_pressure_term(eqdsk_bytes, psi_N):
    """The pressure-driven ``<j_phi>`` part ``p'(<R> - F^2<1/R>/<B^2>)``
    (:func:`bouquet.engine.pressure_term`) of an ARCHIVED draw eqdsk, in the
    archive's positive frame, interpolated onto *psi_N*.

    ``p'``, ``F``, ``<R>``, ``<1/R>``, ``<B^2>`` come from the eqdsk's own
    traced flux surfaces read as :data:`ARCHIVE_EQDSK_COCOS`
    (:func:`bouquet.adapters.gfile_parallel_current`, which refuses an
    eqdsk whose ``<j_phi>`` does not carry its own Ip's sign).  The archived
    ``j_BS`` carries this term; the IDS exporter subtracts it before
    converting to ``<j.B>``.  Cached per
    eqdsk content (the trace takes ~2 s)."""
    import hashlib
    from ..adapters import gfile_parallel_current
    from ..engine import pressure_term
    from .geqdsk import GEQDSKEquilibrium
    raw = bytes(eqdsk_bytes)
    key = hashlib.sha256(raw).hexdigest()
    hit = _P_TERM_CACHE.get(key)
    if hit is None:
        geq = GEQDSKEquilibrium.from_bytes(raw, cocos=ARCHIVE_EQDSK_COCOS)
        _jB, parts = gfile_parallel_current(geq)
        hit = (np.asarray(geq.psi_N, dtype=float), pressure_term(parts))
        if len(_P_TERM_CACHE) > 64:
            _P_TERM_CACHE.clear()
        _P_TERM_CACHE[key] = hit
    return np.interp(np.asarray(psi_N, dtype=float), hit[0], hit[1])


#: Signal fields cut with their time base (IMAS ``signal_flt_1d`` and kin:
#: ``data``, its errors and ``validity_timed`` share the signal's time axis).
_SIGNAL_TIMED_FIELDS = ("data", "data_error_upper", "data_error_lower",
                        "validity_timed")


def _is_time_tagged_aos(v):
    """An array of structures whose elements each carry a scalar ``time``
    (a dynamic AoS: ``equilibrium.time_slice``, ``core_profiles.profiles_1d``,
    ``core_sources.source[*].profiles_1d``, ...)."""
    return (isinstance(v, list) and len(v) > 0 and all(
        isinstance(e, dict) and isinstance(e.get("time"), (int, float))
        and not isinstance(e.get("time"), bool) for e in v))


def _cut_axis0(v, i, n):
    """``[v[i]]`` when *v* is a list of length *n* (> 1), else *v*; *i* may
    be a list of indices (kept in that order)."""
    if isinstance(v, list) and n > 1 and len(v) == n:
        return [v[k] for k in i] if isinstance(i, list) else [v[i]]
    return v


def _cut_dynamic(node, t, base_n=0, base_i=0):
    """Cut the dynamic parts of an IDS subtree in place, keyed on the IMAS
    structure, never on a list's length alone:

    * a time-tagged array of structures keeps its element nearest *t*
      (that element is recursed into for its own signals);
    * a signal -- a structure with ``data`` -- keeps its sample nearest *t*
      on its OWN ``time`` (a non-empty list), or, without one, on the IDS
      time base (*base_n* samples, index *base_i*: a homogeneous-time IDS);
      ``data``, its errors and ``validity_timed`` are cut, nothing else;
    * a summary-style ``{"value": [...]}`` on the IDS time base is cut;
    * anything else -- entry lists (``core_sources.source``,
      ``pf_active.coil``, ``nbi.unit``, ...), coil and limiter outlines,
      radial profiles -- is left whole."""
    if isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                _cut_dynamic(v, t, base_n, base_i)
        return
    if not isinstance(node, dict):
        return
    if "data" in node and isinstance(node.get("data"), list):
        own = node.get("time")
        if isinstance(own, list) and len(own) > 0:
            i = _nearest_index(own, t, "signal time")
            n = len(own)
            node["time"] = [own[i]]
        else:
            i, n = base_i, base_n
        for f in _SIGNAL_TIMED_FIELDS:
            if f in node:
                node[f] = _cut_axis0(node[f], i, n)
    if isinstance(node.get("value"), list):
        node["value"] = _cut_axis0(node["value"], base_i, base_n)
    for k, v in list(node.items()):
        if k in _SIGNAL_TIMED_FIELDS or k == "time":
            continue
        if _is_time_tagged_aos(v):
            kept = v[_nearest_index([e["time"] for e in v], t, k)]
            node[k] = [kept]
            _cut_dynamic(kept, t, 0, 0)
        elif isinstance(v, (dict, list)):
            _cut_dynamic(v, t, base_n, base_i)


def _cut_ids(ids, t, keep=None):
    """Cut one IDS to the time *t*: its ``time`` base, the homogeneous-time
    arrays on it (``vacuum_toroidal_field.b0``, ``code.output_flag``, every
    array under the IDS-level ``global_quantities``), then its dynamic
    parts (:func:`_cut_dynamic`).  Returns the kept index on its time base
    (``None`` without one).

    ``keep`` (times [s]): keep the slice nearest EACH of them instead (on
    the time base and in every time-tagged array of structures, in time
    order) -- the equilibrium slices the reader reads at one core_profiles
    slice (:func:`equilibrium_slices_read`)."""
    tb = ids.get("time")
    i, n = None, 0
    if isinstance(tb, list) and len(tb) > 0:
        n = len(tb)
        i = _nearest_index(tb, t, "time")
        idx = (i if keep is None else
               sorted({_nearest_index(tb, x, "time") for x in keep}))
        ids["time"] = _cut_axis0(tb, idx, n) if n > 1 else list(tb)
        vtf = ids.get("vacuum_toroidal_field")
        if isinstance(vtf, dict) and "b0" in vtf:
            vtf["b0"] = _cut_axis0(vtf["b0"], idx, n)
        code = ids.get("code")
        if isinstance(code, dict) and "output_flag" in code:
            code["output_flag"] = _cut_axis0(code["output_flag"], idx, n)
        gq = ids.get("global_quantities")
        if isinstance(gq, dict):
            _cut_leaves(gq, idx, n)
    for k, v in list(ids.items()):
        if k in ("time", "global_quantities"):
            continue
        if _is_time_tagged_aos(v):
            own = [e["time"] for e in v]
            if keep is None:
                kept = [v[_nearest_index(own, t, k)]]
            else:
                kept = [v[j] for j in sorted(
                    {_nearest_index(own, x, k) for x in keep})]
            ids[k] = kept
            for e in kept:
                _cut_dynamic(e, t, 0, 0)
        elif isinstance(v, (dict, list)):
            _cut_dynamic(v, t, n, 0 if i is None else (
                i if keep is None else idx))
    return i


def _cut_leaves(node, i, n):
    """Every list in *node* (a homogeneous-time ``global_quantities``)
    holding *n* samples keeps sample *i*."""
    for k, v in list(node.items()):
        if isinstance(v, dict):
            _cut_leaves(v, i, n)
        else:
            node[k] = _cut_axis0(v, i, n)


def _cut_core_sources(cs, cp_times, ic, t_cp):
    """Cut ``core_sources`` with the READER's rule, so the export re-reads
    as the archive it came from.

    The core_sources slice is the one :func:`core_sources_slice` reads with
    the core_profiles slice *ic* (refused as the reader refuses).  Each
    entry carrying per-slice times keeps the own slices the rule consults
    at that time -- those bracketing it (the nearest is one of them) -- and
    its first and last own slice (recorded in the match); one without them
    keeps the slice at the core_sources index.  The windows of this read
    (the core_sources slice window, each entry's own and core_profiles
    windows) are recorded under :data:`IMAS_EXPORT_TIME_WINDOW_KEY` in the
    schema-legal ``code.parameters`` JSON string: with
    one time on each base the re-read's half-step windows would otherwise
    collapse to :data:`IMAS_SINGLE_TIME_WINDOW_S`.  The entry list itself
    is never cut."""
    isrc, t_src, rec = core_sources_slice(cs, cp_times, ic, t_cp,
                                          who="IMAS export")
    tb = cs.get("time")
    n_time = len(tb) if tb else None
    cp_half = (None if (rec["core_profiles_time"] is None or t_src is None)
               else _cp_window(cp_times, tb, rec["core_profiles_time"],
                               t_src)[0])
    for s in cs.get("source", []) or []:
        if not isinstance(s, dict):
            continue
        pr = s.get("profiles_1d") or []
        times = [q.get("time") for q in pr if isinstance(q, dict)]
        if pr and t_src is not None and len(times) == len(pr) and all(
                isinstance(x, (int, float)) for x in times):
            erec = {}
            _source_slice_at(s, isrc, t_src, n_time, cp_times,
                             t_cp=rec["core_profiles_time"], cp_half=cp_half,
                             rec=erec)
            tt = np.asarray(times, dtype=float)
            keep = set(_entry_bracketing_slices(times, t_src))
            keep.update((int(np.argmin(tt)), int(np.argmax(tt))))
            s["profiles_1d"] = [pr[k] for k in sorted(keep)]
            _set_export_window(s, dict(
                core_profiles_time=rec["core_profiles_time"],
                core_sources_time=t_src,
                window_own=erec.get("window_own"),
                window_core_profiles=erec.get("window_core_profiles")))
        elif pr and (n_time is None or len(pr) == n_time) and isrc < len(pr):
            s["profiles_1d"] = [pr[isrc]]
        for k, v in list(s.items()):
            if k != "profiles_1d" and _is_time_tagged_aos(v):
                s[k] = [v[_nearest_index([e["time"] for e in v],
                                         t_src if t_src is not None else t_cp,
                                         k)]]
    if tb:
        n = len(tb)
        cs["time"] = [tb[isrc]]
        vtf = cs.get("vacuum_toroidal_field")
        if isinstance(vtf, dict) and "b0" in vtf:
            vtf["b0"] = _cut_axis0(vtf["b0"], isrc, n)
        code = cs.get("code")
        if isinstance(code, dict) and "output_flag" in code:
            code["output_flag"] = _cut_axis0(code["output_flag"], isrc, n)
        if t_src is not None and rec["core_profiles_time"] is not None:
            _set_export_window(cs, dict(
                core_profiles_time=rec["core_profiles_time"],
                core_sources_time=t_src, window=rec["window"],
                window_basis=rec["window_basis"]))


#: How the single-slice cut chose the equilibrium slices it kept (recorded
#: under :data:`IMAS_EXPORT_TIME_WINDOW_KEY` in ``equilibrium.code.parameters``).
EQUILIBRIUM_CUT_RULE = (
    "every equilibrium slice the reader reads at this core_profiles slice: "
    "the one nearest the requested time (targets: ip, l_i, pressure, q, "
    "boundary, F0), the one nearest the core_profiles time (current "
    "orientation and b0 sign) and the last one strictly before it (with the "
    "nearest, the candidates the core_profiles currents are paired with: "
    "paired_equilibrium_candidates)")


def equilibrium_slices_read(eq_times, t, t_cp):
    """The ``equilibrium`` slices :func:`read_imas_baseline` (and the
    engine's IDS adapter) read for the request time *t* [s] whose
    core_profiles slice is at *t_cp*: ``{"targets": i, "orientation": j,
    "current_pairing": [k, ...]}`` (indices on *eq_times*).

    ``targets`` is the slice nearest *t* (ip, l_i, pressure, q0, the jphi
    anchor, the boundary and F0; *t* None -> nearest *t_cp*), ``orientation``
    the slice nearest *t_cp* (:func:`orientation_slice_index`) and
    ``current_pairing`` the candidates the core_profiles currents are
    converted on (:func:`paired_equilibrium_candidates`)."""
    te = np.asarray(eq_times, dtype=float)
    k_cp = int(np.argmin(np.abs(te - t_cp)))
    i_t = (k_cp if t is None or te.size == 1 else
           int(np.argmin(np.abs(te - float(t)))))
    return {"targets": i_t, "orientation": k_cp,
            "current_pairing": paired_equilibrium_candidates(te, t_cp)}


def _record_cp_time_grid(cp, cpt, ic, t_cp):
    """Record, in ``core_profiles.code.parameters`` (under
    :data:`IMAS_EXPORT_TIME_WINDOW_KEY`), the core_profiles times adjacent
    to the kept slice: the only thing the reader takes from the slices a
    cut drops (the half local step of the ida_hybrid time rule,
    :func:`_hybrid_timing`; :func:`_cp_time_grid` reads it back)."""
    grid = np.unique(np.asarray(cpt, dtype=float))
    if grid.size < 2:
        return
    k = int(np.argmin(np.abs(grid - t_cp)))
    nb = grid[max(k - 1, 0):k + 2]
    _set_export_window(cp, dict(core_profiles_time=float(t_cp),
                                time_neighbours=[float(x) for x in nb]))


def _cp_time_grid(cp_ids):
    """The core_profiles time base the reader's local-step windows use: the
    IDS ``time``, or -- for a single-slice export that recorded them
    (:func:`_record_cp_time_grid`) and whose one time is the recorded slice
    -- the adjacent times of the dd it was cut from."""
    tb = cp_ids.get("time")
    if isinstance(tb, list) and len(tb) == 1:
        meta = _get_export_window(cp_ids)
        try:
            if (meta is not None
                    and float(meta["core_profiles_time"]) == float(tb[0])):
                return [float(x) for x in meta["time_neighbours"]]
        except (KeyError, TypeError, ValueError):
            pass
    return tb


def _slice_in_time(dd, t, equilibrium="read"):
    """Cut a dd in place to the ONE slice the reader reads at time *t* [s].

    The cut is taken at the core_profiles slice nearest *t* (the slice
    :func:`read_imas_baseline` reads, ``t_cp``), not at *t*: every IDS keeps
    its sample nearest ``t_cp`` on its own time base (``pf_active`` and
    other signals on their own ``time``), and ``core_sources`` is cut with
    the reader's own rule (:func:`_cut_core_sources`), recording the windows
    of the read so a re-read applies them.  What is cut is keyed on the IMAS
    structure (:func:`_cut_ids`, :func:`_cut_dynamic`): the IDS ``time``,
    time-tagged arrays of structures, signals, and the homogeneous-time
    arrays named there.  Lists of entries (sources, coils, beams, probes),
    static geometry (coil and limiter outlines) and radial profiles are
    never cut, whatever their length.  Returns ``t_cp``.

    ``equilibrium``: ``"read"`` (default) keeps EVERY equilibrium slice the
    reader reads at that core_profiles slice (:func:`equilibrium_slices_read`:
    the slice nearest *t*, the one nearest ``t_cp`` and the last one before
    ``t_cp``, which FUSE pairs the core_profiles currents with on a
    time-dependent run), so the cut re-reads identically; the kept times and
    their roles are recorded under :data:`IMAS_EXPORT_TIME_WINDOW_KEY` in
    ``equilibrium.code.parameters``.  ``"one"`` keeps only the slice nearest
    ``t_cp`` (:func:`write_imas_draw`: a draw is ONE equilibrium, and its
    currents are written on its own geometry).  core_profiles keeps the one
    slice; its adjacent times are recorded (:func:`_record_cp_time_grid`)."""
    if equilibrium not in ("read", "one"):
        raise ValueError(f"equilibrium must be 'read' or 'one', got "
                         f"{equilibrium!r}")
    cp = dd.get("core_profiles") if isinstance(dd.get("core_profiles"),
                                                dict) else {}
    cpt = cp.get("time")
    if isinstance(cpt, list) and len(cpt) > 0:
        ic = _nearest_index(cpt, t, "core_profiles")
        t_cp = float(cpt[ic])
    else:
        ic, t_cp = 0, (None if t is None else float(t))
    if t_cp is None:
        return None
    cs = dd.get("core_sources")
    if isinstance(cs, dict):
        _cut_core_sources(cs, cpt if cpt else None, ic, t_cp)
    if isinstance(cpt, list) and len(cpt) > 1:
        _record_cp_time_grid(cp, cpt, ic, t_cp)
    eq = dd.get("equilibrium")
    keep_eq, eq_rec = None, None
    if (equilibrium == "read" and isinstance(eq, dict)
            and isinstance(eq.get("time"), list) and len(eq["time"]) > 1):
        te = [float(x) for x in eq["time"]]
        roles = equilibrium_slices_read(te, t, t_cp)
        keep_eq = sorted({te[roles["targets"]], te[roles["orientation"]]}
                         | {te[k] for k in roles["current_pairing"]})
        eq_rec = dict(
            core_profiles_time=t_cp,
            requested_time=None if t is None else float(t),
            kept_times=keep_eq,
            targets_time=te[roles["targets"]],
            orientation_time=te[roles["orientation"]],
            current_pairing_times=[te[k] for k in roles["current_pairing"]],
            rule=EQUILIBRIUM_CUT_RULE)
    for name, ids in dd.items():
        if name == "core_sources" or not isinstance(ids, dict):
            continue
        _cut_ids(ids, t_cp, keep=keep_eq if name == "equilibrium" else None)
    if eq_rec is not None:
        _set_export_window(eq, eq_rec)
    return t_cp


def _signed_b0(out, ie, ic):
    """The template's own vacuum B0 (signed), equilibrium first; None if absent
    or zero.  The writer keeps ``vacuum_toroidal_field`` as it is, so this is
    the field orientation the exported F must agree with."""
    for ids_name, it in (("equilibrium", ie), ("core_profiles", ic)):
        vtf = out.get(ids_name, {}).get("vacuum_toroidal_field")
        if vtf and vtf.get("b0") is not None:
            b0 = np.atleast_1d(np.asarray(vtf["b0"], dtype=float))
            if b0.size:
                v = float(b0[min(it, b0.size - 1)])
                if np.isfinite(v) and v != 0.0:
                    return v
    return None


def _export_orientation(out, ie, ic, stamp):
    """``(s_I, s_B, s_q)``: the SOURCE orientation an export is restored to.

    bouquet's archive is in the positive-Ip frame (eqdsk ``CURRENT > 0``,
    ``F > 0``, q > 0; every current co-Ip positive).  The template is the
    source dd, in the source's own frame, and the writer keeps its fields that
    it does not overwrite (``core_sources``, ``pf_active``,
    ``vacuum_toroidal_field``, rotation, ...).  So every quantity the writer
    DOES overwrite is taken back to that frame:

      * ``s_I`` -- the Ip orientation: the archive's ``source_current_sign``
        (the factor the reader applied; it is its own inverse), or, for an
        archive with no stamp, ``sign(template ip)`` -- what the reader's
        ``"auto"`` rule gives;
      * ``s_B`` -- the sign of the template's own b0 (``+1`` if it has none);
      * ``s_q`` -- the template's own q sign convention (sign of its
        equilibrium q at this slice) where it carries q, else ``s_I * s_B``
        (IMAS COCOS 11 / 17, where q carries sign(Ip*B0)).

    Refuses a template whose ip or b0 sign contradicts the archive's stamp
    (with the reader's ``"auto"`` rule): that template is not the source this
    archive was generated from.  ``stamp`` is the ``_baseline`` attrs dict
    (possibly empty).
    """
    ts = out["equilibrium"]["time_slice"][ie]
    tpl_ip = ts.get("global_quantities", {}).get("ip")
    try:
        tpl_ip = float(tpl_ip)
        tpl_s = (source_current_sign(tpl_ip)
                 if np.isfinite(tpl_ip) and tpl_ip != 0.0 else None)
    except (TypeError, ValueError):
        tpl_s = None
    b0 = _signed_b0(out, ie, ic)
    s_B = 1.0 if b0 is None else (-1.0 if b0 < 0.0 else 1.0)

    if "source_current_sign" in stamp:
        s_I = float(stamp["source_current_sign"])
        origin = stamp.get("source_current_sign_origin")
        if isinstance(origin, bytes):
            origin = origin.decode()
        auto = origin in (None, ORIENTATION_ORIGIN_AUTO)
        if auto and tpl_s is not None and tpl_s != s_I:
            raise ValueError(
                "write_imas_draw: the archive was generated from a source with "
                f"source_current_sign = {s_I:+.0f} (sign of its ip), but this "
                f"template's ip = {tpl_ip:+.6g} A has the opposite sign. It is "
                "not the source dd this archive was built from; pass that dd as "
                "the template.")
        sb_stamp = stamp.get("source_b0_sign")
        if sb_stamp is not None and b0 is not None and float(sb_stamp) != s_B:
            raise ValueError(
                "write_imas_draw: the archive was generated from a source with "
                f"source_b0_sign = {float(sb_stamp):+.0f}, but this template's "
                f"b0 = {b0:+.6g} T has the opposite sign. It is not the source "
                "dd this archive was built from; pass that dd as the template.")
    else:
        s_I = 1.0 if tpl_s is None else tpl_s
        if s_I < 0.0:
            import warnings
            warnings.warn(
                "write_imas_draw: the archive carries no current-orientation "
                "stamp (_baseline source_current_sign) and the template has "
                "ip < 0. Assuming the archive is in bouquet's positive-Ip frame "
                "and restoring the template's orientation (x -1). An IMAS "
                "archive generated from a reversed-Ip source before the reader "
                "normalised currents is NOT in that frame (and is invalid: "
                "regenerate it).", stacklevel=3)

    s_q = None
    q_tpl = (ts.get("profiles_1d") or {}).get("q")
    if q_tpl is not None:
        try:
            qa = np.asarray(q_tpl, dtype=float)
            qm = float(np.nanmedian(qa)) if qa.size else float("nan")
            if np.isfinite(qm) and qm != 0.0:
                s_q = -1.0 if qm < 0.0 else 1.0
        except (TypeError, ValueError):
            s_q = None
    if s_q is None:
        s_q = s_I * s_B
    return float(s_I), float(s_B), float(s_q)


#: Key (in ``core_profiles.code.parameters``, JSON) of an exported draw's
#: ion-species record (:func:`_write_draw_ion_species`).
IMAS_EXPORT_SPECIES_KEY = "bouquet_species_model"

#: The species model every bouquet solve uses, which an exported draw
#: writes (:func:`_write_draw_ion_species`).
DRAW_SPECIES_MODEL = (
    "electrons (n_e, T_e); one hydrogenic main ion (thermal n_i, T_i); ONE "
    "effective impurity of charge Z_imp at the main-ion T_i with "
    "n_z = max(n_e - z_fast - n_i, 0) / Z_imp (physics.impurity_pressure on "
    "n_e - z_fast); the fast population (density_fast, pressure_fast_*) "
    "held fixed as the template's")


def _write_draw_ion_species(cp, ni_t, ti_t, ne_t, z_fast_t, Z_imp):
    """Write a draw's thermal ion species into the core_profiles slice *cp*
    (on the template grid) as the solve used them (:data:`DRAW_SPECIES_MODEL`),
    so the export is self-consistent: quasineutral, with the drawn Z_eff,
    and with the reader's single-impurity pressure equal to the species sum.

    The main ion is the first ``z_n == 1`` species.  With ``Z_imp`` (the
    archive's per-draw / baseline ``Z_imp``), the impurity is the template's
    non-hydrogenic species of that charge; failing one, its first other
    thermal species, relabelled to ``Z_imp`` (the solve's effective
    impurity; a species is added when the template has none).  Every other
    thermal species the solve did not carry (further impurities, a second
    hydrogenic species) gets zero thermal density.  Fast densities and
    pressures are untouched.  Without ``Z_imp`` (an archive that predates
    it, or a baseline with no dilution information) only the main ion is
    written and the template's impurities are kept, as before.  Returns the
    record written under :data:`IMAS_EXPORT_SPECIES_KEY`."""
    ions = cp.get("ion") or []
    main = next((ion for ion in ions
                 if float(ion["element"][0]["z_n"]) == 1.0), None)
    if main is None:
        raise ValueError("write_imas_draw: the template core_profiles has no "
                         "hydrogenic (z_n = 1) main ion to write the draw's "
                         "n_i into")
    main["density_thermal"] = np.asarray(ni_t, dtype=float).tolist()
    main["temperature"] = np.asarray(ti_t, dtype=float).tolist()
    rec = dict(model=DRAW_SPECIES_MODEL, main_ion=str(main.get("label")),
               Z_imp=None if not Z_imp else float(Z_imp), impurity=None,
               impurity_relabelled_from=None, zeroed=[])
    if not Z_imp:
        rec["impurity"] = ("template's kept: the archive carries no Z_imp "
                           "(no single-impurity model to write)")
        return rec
    Z_imp = float(Z_imp)
    ne_th = np.maximum(np.asarray(ne_t, dtype=float) - (
        0.0 if z_fast_t is None else np.asarray(z_fast_t, dtype=float)), 0.0)
    nz = np.clip((ne_th - np.asarray(ni_t, dtype=float)) / Z_imp, 0.0, None)

    def thermal(ion):
        d = ion.get("density_thermal")
        return d is not None and bool(np.any(np.asarray(d, dtype=float)))
    others = [ion for ion in ions if ion is not main
              and float(ion["element"][0]["z_n"]) != 1.0]
    imp = next((ion for ion in others if np.isclose(
        float(ion["element"][0]["z_n"]), Z_imp, rtol=1e-9, atol=0.0)), None)
    if imp is None:
        imp = next((ion for ion in others if thermal(ion)),
                   others[0] if others else None)
        if imp is None:
            imp = {"label": "impurity", "element": [{"z_n": Z_imp}]}
            ions.append(imp)
            cp["ion"] = ions
            rec["impurity_relabelled_from"] = "(added: no impurity species)"
        else:
            rec["impurity_relabelled_from"] = dict(
                label=imp.get("label"), z_n=float(imp["element"][0]["z_n"]))
            imp["element"][0]["z_n"] = Z_imp
            imp["label"] = f"impurity_Z{Z_imp:g}"
    imp["density_thermal"] = nz.tolist()
    imp["temperature"] = np.asarray(ti_t, dtype=float).tolist()
    rec["impurity"] = str(imp.get("label"))
    for ion in ions:
        if ion is main or ion is imp or not thermal(ion):
            continue
        ion["density_thermal"] = np.zeros_like(
            np.asarray(ion["density_thermal"], dtype=float)).tolist()
        rec["zeroed"].append(str(ion.get("label")))
    return rec


def write_imas_draw(h5path_or_header, draw_index, template_ids_path, out_path,
                    scan_key=None, time=None, fidelity="auto"):
    """Reconstruct a perturbed IMAS/OMAS IDS for one draw from the bouquet HDF5.

    Orientation: the exported file is in the SOURCE's frame throughout.  The
    archive is in bouquet's positive-Ip frame; the template (the source dd)
    is in the source's own, and the fields the writer keeps from it --
    ``core_sources`` (the beam ``j_parallel`` / ``current_parallel_inside``),
    ``pf_active`` coil currents, ``vacuum_toroidal_field.b0``,
    ``core_profiles.global_quantities``, rotation / ``E_r``, the
    ``core_profiles`` psi grid -- stay as they are.  Every quantity the writer
    overwrites is taken back to the source orientation to match them (see
    :func:`_export_orientation` for how ``s_I``, ``s_B``, ``s_q`` are found):

      * ``s_I`` (Ip): ``global_quantities`` ``ip`` / ``psi_axis`` /
        ``psi_boundary``, ``profiles_1d`` ``psi`` / ``dpressure_dpsi`` /
        ``f_df_dpsi``, ``profiles_2d`` ``psi``, and ``core_profiles`` ``j_tor``
        / ``j_total`` / ``j_ohmic`` / ``j_bootstrap``;
      * ``s_B`` (B0): ``profiles_1d.f``;
      * ``s_q``: ``profiles_1d.q``, ``q_axis``, ``q_95``;
      * even (unchanged): pressure, kinetics, l_i, betas, axis, boundary.

    So re-reading an export with :func:`read_imas_baseline` gives the same
    currents as re-reading the export of the un-mirrored source.  For a
    source with ``ip > 0`` and ``b0 > 0`` every factor is ``+1`` and the
    output is unchanged; for ``ip > 0``, ``b0 < 0`` the only change is that
    ``f`` (and, when the template carries no q, q) now takes b0's sign.
    The equilibrium psi / P' / FF' are the archived TokaMaker eqdsk's
    (COCOS 7, psi per radian) converted to COCOS 11 (``psi_11 = -2 pi
    psi_7``) before the orientation factor; ``profiles_1d`` also carries the
    eqdsk's own ``rho_tor_norm`` and, from a complete ``eq_fsa`` block, the
    draw's ``gm1/gm5/gm8/gm9``, and ``core_profiles.grid`` the draw's own psi
    and ``rho_tor_norm`` at the template's psi_N nodes -- so a reader converts
    the currents on the geometry they were written with (review PR64 B4: the
    COCOS-7 values came back ``-2 pi`` times the archived p').  The
    equilibrium ``profiles_1d`` written carries no ``j_tor``, so re-reading an
    export needs ``anchor_jtor_to_equilibrium=False``.

    Maps the draw's archived eqdsk to the ``equilibrium`` IDS
    (``profiles_1d`` / ``profiles_2d`` / ``global_quantities`` / ``boundary`` --
    lossless to the eqdsk grid, machine-precision GS) and the draw's ``.h5``
    kinetics/currents to ``core_profiles``.  The written IDS holds only that
    time slice (:func:`_slice_in_time`, review PR71): every IDS is cut at the
    ``core_profiles`` slice nearest ``time`` -- the slice the reader reads --
    keyed on the IMAS structure (the IDS ``time``, time-tagged arrays of
    structures, signals on their own or the homogeneous time base,
    ``vacuum_toroidal_field.b0``, ``code.output_flag``,
    ``global_quantities``), never on list length, keeping the template's
    structure; ``core_sources`` is cut by the reader's own source-time rule
    (each entry keeps its slices bracketing the time plus its first and last
    own slice), with the windows of that read recorded under
    :data:`IMAS_EXPORT_TIME_WINDOW_KEY` in the schema-legal
    ``code.parameters`` JSON string so a re-read matches the same entries
    exactly.  The equilibrium holds the ONE slice the draw is written into
    (the template's paired slice, which a pure cut keeps, is used only for
    ``fidelity="reconstruct"`` and recorded).  The template is this
    function's own fresh ``json.load``, never
    the shared parsed dd of :func:`_load_dd` (#72: the cut mutates it).

    bouquet's arrays are TokaMaker ``jphi``; they are written as IMAS
    ``j_tor`` (A5) and the parallel split ``j_total`` / ``j_ohmic`` /
    ``j_bootstrap`` as ``<J.B>/B0`` (A7; see the :mod:`bouquet.physics`
    docstring).  No exported parallel current carries the pressure-driven
    ``p'G`` (its ``<j.B>`` is zero): an engine draw's stored ``<j.B>`` parts
    (``jB_parallel/``) are written as they are; otherwise ``p'G`` comes off
    the bucket the archive keeps it in (``schema.read_current_split_convention``:
    its own ``j_pressure`` since owner decision D2, ``j_BS`` in a PR #64
    archive, the residual ``j_inductive`` in every earlier one).  ``j_tor``
    (A5 of the total) includes it; so, per FUSE (IMAS.jl ``Jpar_2_Jtor``,
    ``includes_bootstrap=true``), does the toroidal image of the
    non-inductive/bootstrap group, never the ohmic one.
    ``j_total = j_ohmic + j_bootstrap + driven``, and ``j_non_inductive``
    (when in the template) is ``j_total - j_ohmic``.  The geometry is set by
    ``fidelity``:

      * ``"exact"``       -- the draw's OWN captured flux-surface geometry
        (``eq_fsa`` block, from ``capture_live_eq=True`` at generate time).
        Raises if the block is absent or predates ``avg_R``/``pprime``.
      * ``"reconstruct"`` -- the template's baseline equilibrium geometry
        (exact only when the draw's flux geometry matches the baseline's).
      * ``"auto"`` (default) -- exact when a complete ``eq_fsa`` block is
        present; with an older block (no ``avg_R``/``pprime``) its own kappa
        and the template's geometry for ``j_tor`` (warns); else reconstruct.

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.
    draw_index : int
        Per-draw index within the scan group.
    template_ids_path : str
        The source IMAS/OMAS JSON, used as the structural template.
    out_path : str
        Where to write the perturbed IDS JSON.
    fidelity : {"auto", "exact", "reconstruct"}
        Parallel-current split fidelity (see above); default ``"auto"``.
    scan_key : str, float, or None
        Scan value selecting the ``scan/<scan_key>`` group.  The default
        ``None`` auto-resolves a single-scan archive (and is the flat layout
        for flat files); pass the explicit value only when the archive holds
        more than one scan.
    time : float, optional
        Time slice [s]; defaults to the template's nearest single slice.

    Returns
    -------
    str
        ``out_path``.
    """
    import json
    import copy
    import h5py
    from .geqdsk import read_geqdsk
    from ..utils import read_eqdsk_from_bytes, _group_path, _resolve_h5

    if fidelity not in ("auto", "exact", "reconstruct"):
        raise ValueError(
            f"fidelity must be 'auto'|'exact'|'reconstruct', got {fidelity!r}")

    # a FRESH parse: _slice_in_time cuts this object in place, so it must
    # never be the shared cached dd of _load_dd (#72 B5)
    with open(template_ids_path) as fh:
        out = json.load(fh)

    eq_ids = out["equilibrium"]
    ie = _nearest_index(eq_ids["time"], time, "equilibrium")
    cp_ids = out["core_profiles"]
    ic = _nearest_index(cp_ids["time"], time, "core_profiles")
    # Only the exported slice is written: every IDS is cut at the
    # core_profiles slice nearest the time, core_sources by the reader's rule
    # with its windows recorded (#71).  The cut first keeps every equilibrium
    # slice the reader reads there (the template's currents are paired with
    # one of them: _paired_current_geometry), so the template-geometry
    # conversion below uses the slice the template's currents belong to;
    # the equilibrium is then cut to the ONE slice the draw is written into
    # (a draw is one equilibrium, and its currents are written on its own
    # geometry, so a re-read pairs them with it).  This MUST stay before
    # ie = ic = 0: every index below addresses the one kept slice.
    t_cp = _slice_in_time(out, eq_ids["time"][ie] if time is None else time)
    cp = cp_ids["profiles_1d"][0]
    tmpl_geom, tmpl_pair = None, None
    if fidelity != "exact" and t_cp is not None:
        try:
            tmpl_geom, tmpl_pair = _paired_current_geometry(
                eq_ids, cp, t_cp, cp.get("j_total"), cp.get("j_tor"))
        except (KeyError, TypeError, ValueError):
            tmpl_geom = None
    _cut_ids(eq_ids, t_cp)
    _set_export_window(eq_ids, dict(
        core_profiles_time=t_cp, kept_times=[float(x) for x in eq_ids["time"]],
        rule=("write_imas_draw: the draw's own equilibrium, the one slice "
              "kept (nearest the core_profiles time); the exported currents "
              "are written on its geometry"),
        template_current_pairing_time=(None if tmpl_pair is None
                                       else tmpl_pair["time"])))
    ie = ic = 0

    h5 = _resolve_h5(h5path_or_header)
    if scan_key is None:
        # Convenience: a single-scan archive resolves unambiguously, so the
        # caller need not echo the generation scan_key.  Flat-layout files
        # (discover_scan_keys -> None) keep scan_key=None.
        from ..utils import discover_scan_keys
        keys = discover_scan_keys(h5)
        if keys:
            if len(keys) != 1:
                raise ValueError(
                    f"{h5} holds {len(keys)} scans {keys}; pass an explicit "
                    "scan_key to write_imas_draw().")
            scan_key = keys[0]
    gp = _group_path(scan_key, draw_index)
    with h5py.File(h5, "r") as hf:
        if gp not in hf:
            raise KeyError(f"draw {draw_index} (scan {scan_key}) not in {h5}")
        _bgp = (f"scan/{scan_key}/_baseline" if scan_key is not None
                else "_baseline")
        stamp = dict(hf[_bgp].attrs) if _bgp in hf else {}
        g = hf[gp]
        ne = np.asarray(g["n_e"][()]); te = np.asarray(g["T_e"][()])
        ni = np.asarray(g["n_i"][()]); ti = np.asarray(g["T_i"][()])
        pkin = np.asarray(g["psi_N_kinetic"][()]); peq = np.asarray(g["psi_N"][()])
        j_tor = np.asarray(g["j_phi"][()])
        j_ind = np.asarray(g["j_inductive"][()])
        j_bs = np.asarray(g["j_BS"][()])
        zeff = np.asarray(g["aux_zeff"][()]) if "aux_zeff" in g else None
        # the solve's species model (per draw, else the baseline's): the
        # effective impurity charge and the fast-ion charge density it was
        # assembled with (_write_draw_ion_species)
        Z_imp_d = g.attrs.get("Z_imp", stamp.get("Z_imp"))
        z_fast_d, zf_x = None, pkin
        if "z_fast" in g:
            z_fast_d = np.asarray(g["z_fast"][()], dtype=float)
        elif _bgp in hf and "z_fast" in hf[_bgp]:
            z_fast_d = np.asarray(hf[_bgp]["z_fast"][()], dtype=float)
            if "psi_N_kinetic" in hf[_bgp]:
                zf_x = np.asarray(hf[_bgp]["psi_N_kinetic"][()], dtype=float)
        li1 = float(g.attrs.get("l_i(1)", np.nan))
        li3 = float(g.attrs.get("l_i(3)", np.nan))
        from ..schema import (find_bytes_dataset, EQ_FSA_GROUP,
                              read_jB_parallel)
        eqk = find_bytes_dataset(g)
        if eqk is None:
            raise KeyError(f"draw {draw_index} has no archived eqdsk")
        eq_bytes = bytes(g[eqk][()])
        geq = read_eqdsk_from_bytes(eq_bytes, read_geqdsk)
        # the engine draw's stored PARALLEL parts (schema jB_parallel/)
        jB_par = read_jB_parallel(g)
        # where the archived split keeps p'G (schema; absent = pre-PR #64)
        from ..schema import read_current_split_convention, read_profile
        split_conv = read_current_split_convention(g, stamp)
        j_press = read_profile(g, "j_pressure")
        # optional captured live-equilibrium FSA block (exact conversion)
        eq_fsa = None
        if EQ_FSA_GROUP in g:
            eq_fsa = {k: np.asarray(g[EQ_FSA_GROUP][k][()], dtype=float)
                      for k in g[EQ_FSA_GROUP]}

    # --- source orientation to restore (see the docstring) -----------------
    s_I, s_B, s_q = _export_orientation(out, ie, ic, stamp)

    # Baseline (template) geometry for fidelity="reconstruct" (tmpl_geom):
    # read above, before the cut to one equilibrium slice and before the
    # slice is overwritten with the draw's eqdsk -- on the equilibrium slice
    # the template's own currents are paired with (as the reader converts
    # them), not merely the nearest one.

    # --- equilibrium IDS from the eqdsk (lossless to the eqdsk grid) ---------
    # The archived eqdsk is COCOS 7 (TokaMaker: psi per radian, decreasing
    # outward for Ip > 0); the IDS is COCOS 11 (psi per full turn, sigma_Bp
    # flipped): psi_11 = -2 pi psi_7, so d/dpsi_11 = d/dpsi_7 / (-2 pi).
    # Before 2026-10-09 the COCOS-7 values were written as they were, and a
    # re-read (p' = -2 pi dpressure_dpsi, COCOS 11) came back -2 pi times
    # the archived p' (review PR64 B4, confirmed by a round trip).
    c11 = -2.0 * np.pi
    ts = eq_ids["time_slice"][ie]
    psi1d = geq.psi_axis + geq.psi_N * (geq.psi_boundary - geq.psi_axis)
    q95 = float(np.interp(0.95, geq.psi_N, geq.qpsi))
    ts["profiles_1d"] = {
        "psi": (s_I * (c11 * psi1d)).tolist(),
        "q": (s_q * np.asarray(geq.qpsi, dtype=float)).tolist(),
        "pressure": geq.pres.tolist(),
        "f": (s_B * np.asarray(geq.fpol, dtype=float)).tolist(),
        "dpressure_dpsi": (s_I * np.asarray(geq.pprime, dtype=float)
                           / c11).tolist(),
        "f_df_dpsi": (s_I * np.asarray(geq.ffprim, dtype=float) / c11).tolist(),
        # the eqdsk's own normalised toroidal flux (from its q)
        "rho_tor_norm": np.asarray(geq.rhovn, dtype=float).tolist(),
    }
    # the draw's OWN flux-surface averages (eq_fsa), as FUSE writes them, so
    # a reader converts this IDS's currents on the geometry they were
    # exported with instead of re-tracing profiles_2d
    if eq_fsa is not None and all(eq_fsa.get(k) is not None for k in (
            "avg_R", "avg_inv_R", "avg_inv_R2", "avg_B2")):
        _src = np.asarray(eq_fsa["psi_N"], dtype=float)
        _pn = np.asarray(geq.psi_N, dtype=float)
        for _gm, _k in (("gm1", "avg_inv_R2"), ("gm5", "avg_B2"),
                        ("gm8", "avg_R"), ("gm9", "avg_inv_R")):
            ts["profiles_1d"][_gm] = np.interp(
                _pn, _src, np.asarray(eq_fsa[_k], dtype=float)).tolist()
    ts["profiles_2d"] = [{
        "grid_type": {"name": "rectangular", "index": 1},
        "grid": {"dim1": geq.R_grid.tolist(), "dim2": geq.Z_grid.tolist()},
        # psi_RZ is indexed [R][Z], matching IMAS dim1=R, dim2=Z
        "psi": (s_I * c11 * np.asarray(geq.psi_RZ, dtype=float)).tolist(),
    }]
    gq = dict(ts.get("global_quantities", {}))
    gq.update(
        ip=s_I * float(geq.Ip), psi_axis=s_I * c11 * float(geq.psi_axis),
        psi_boundary=s_I * c11 * float(geq.psi_boundary),
        magnetic_axis={"r": float(geq.R_mag), "z": float(geq.Z_mag)},
        q_axis=s_q * float(geq.qpsi[0]), q_95=s_q * q95,
        li_3=li3 if np.isfinite(li3) else geq.li.get("li(3)"),
        beta_normal=geq.betas.get("beta_n"), beta_pol=geq.betas.get("beta_p"),
        beta_tor=geq.betas.get("beta_t"),
    )
    if np.isfinite(li1):
        gq["li_1"] = li1
    ts["global_quantities"] = gq
    ts["boundary"] = {"outline": {"r": geq.boundary_R.tolist(),
                                  "z": geq.boundary_Z.tolist()}}

    # --- core_profiles from the draw (kinetics + currents) ------------------
    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
    # The draw's arrays are on the archive's run grid: a phi_n archive lands on
    # the template's own Phi_N nodes (grid.rho_tor_norm**2).
    # The draw's ψ_N at those nodes (its eqdsk's own Φ_N map) addresses its
    # flux-surface geometry and is written as grid.psi.
    from ..utils import profile_coord
    x_t = psiN_fsa = psiN_t
    if profile_coord(h5, scan_key) != "psi_n":
        x_t = _dd_phi_n(cp, psiN_t)
        phi_g = np.asarray(geq.rhovn, dtype=float) ** 2
        phi_g = (phi_g - phi_g[0]) / (phi_g[-1] - phi_g[0])
        psiN_fsa = np.interp(x_t, phi_g, np.asarray(geq.psi_N, dtype=float))
        cp["grid"]["psi"] = (s_I * c11 * (geq.psi_axis + psiN_fsa * (
            geq.psi_boundary - geq.psi_axis))).tolist()
    else:
        # the draw's own rho_tor_norm at the template's psi_N nodes (kept),
        # so the nodes and the equilibrium geometry a reader pairs them with
        # agree
        cp["grid"]["rho_tor_norm"] = np.interp(
            psiN_t, np.asarray(geq.psi_N, dtype=float),
            np.asarray(geq.rhovn, dtype=float)).tolist()

    def to_t(arr, src):     # interp draw array (on src grid) -> template grid
        return np.interp(x_t, src, arr)

    cp["electrons"]["density_thermal"] = to_t(ne, pkin).tolist()
    cp["electrons"]["temperature"] = to_t(te, pkin).tolist()
    # every thermal species the solve used, as it used them (main ion, the
    # one effective impurity on ne - z_fast at T_i); before, only the main
    # ion was written and the template's impurity stayed, so an export whose
    # n_i / Z_eff differ from the template's (an ida_hybrid draw) was not
    # quasineutral and the reader refused it (thermal species gap)
    species = _write_draw_ion_species(
        cp, to_t(ni, pkin), to_t(ti, pkin), to_t(ne, pkin),
        None if z_fast_d is None else to_t(z_fast_d, zf_x), Z_imp_d)
    _set_export_window(cp_ids, species, key=IMAS_EXPORT_SPECIES_KEY)
    if zeff is not None:
        cp["zeff"] = to_t(zeff, pkin).tolist()

    # Currents (TokaMaker jphi on the draw grid -> template grid -> IMAS),
    # converted in the archive's positive frame (|F|, |B0|, positive-frame p')
    # and written times s_I: the source-frame current.  kappa is the draw's
    # own (eq_fsa); so is the IMAS j_tor (A5) when the block carries avg_R
    # and pprime, else the template's (baseline) geometry is used.
    geom = None
    if fidelity in ("auto", "exact") and eq_fsa is not None:
        geom = _eq_fsa_geom_on(eq_fsa, psiN_fsa, _imas_b0(out, ie, ic))
    full = geom is not None and all(k in geom for k in _EQ_FSA_GEOM_KEYS)
    a5 = geom
    if not full:
        if fidelity == "exact":
            raise ValueError(
                f"fidelity='exact' requested but draw {draw_index} has no "
                "complete captured eq_fsa block (needs "
                f"{list(_EQ_FSA_GEOM_KEYS)}; generate with capture_live_eq="
                "True). Use fidelity='auto' to fall back to the template "
                "(baseline) geometry.")
        if eq_fsa is not None:
            import warnings
            warnings.warn(
                f"draw {draw_index}: eq_fsa block lacks "
                f"{[k for k in _EQ_FSA_GEOM_KEYS if eq_fsa.get(k) is None]}; "
                + ("j_tor" if geom is not None else "currents")
                + " converted with the template (baseline) geometry")
        if tmpl_geom is None:
            raise ValueError(
                f"draw {draw_index}: no complete eq_fsa block, and the "
                "template-geometry conversion needs the template equilibrium's "
                f"{list(_FUSE_GEOM_FIELDS)} (and core_profiles rho_tor_norm) "
                "at the exported slice")
        # the template's geometry is in the dd's own orientation: bring it
        # into the archive's positive frame first
        a5 = _positive_frame_geom(tmpl_geom, s_I)
        geom = a5 if geom is None else geom
    jphi_t = to_t(j_tor, peq)
    cp["j_tor"] = (s_I * jphi_tokamaker_to_jtor_imas(jphi_t, a5)).tolist()
    if fidelity in ("auto", "exact") and jB_par is not None:
        src = np.asarray(jB_par.get("psi_N", peq), dtype=float)
        b0 = _imas_b0(out, ie, ic)
        ohm, bs, drv = (to_t(np.asarray(jB_par[k], dtype=float), src) / b0
                        for k in ("jB_inductive", "jB_BS", "jB_NBI"))
        drv = drv + to_t(np.asarray(jB_par["jB_RF"], dtype=float), src) / b0
    else:
        jt_ind, jt_bs = to_t(j_ind, peq), to_t(j_bs, peq)
        # p'G has zero <j.B>: it comes off the bucket that carries it
        # (schema.read_current_split_convention) before the parallel
        # conversion -- the archived j_pressure when the split stores it,
        # else the archived eqdsk's own (what a reader of this IDS recovers)
        if split_conv == SPLIT_PRESSURE_SEPARATE and j_press is not None:
            P_t = to_t(j_press, peq)
        else:
            P_t = archived_pressure_term(eq_bytes, psiN_fsa)
        if split_conv == SPLIT_PRESSURE_SEPARATE:
            ohm = jphi_tokamaker_to_jpar(jt_ind, geom)
            bs = jphi_tokamaker_to_jpar(jt_bs, geom)
            drv = jphi_tokamaker_to_jpar(jphi_t - jt_ind - jt_bs - P_t, geom)
        elif split_conv == SPLIT_PRESSURE_IN_BOOTSTRAP:
            ohm = jphi_tokamaker_to_jpar(jt_ind, geom)
            bs = jphi_tokamaker_to_jpar(jt_bs - P_t, geom)
            drv = jphi_tokamaker_to_jpar(jphi_t - jt_ind - jt_bs, geom)
        else:                                   # pre-PR #64: in j_inductive
            ohm = jphi_tokamaker_to_jpar(jt_ind - P_t, geom)
            bs = jphi_tokamaker_to_jpar(jt_bs, geom)
            drv = jphi_tokamaker_to_jpar(jphi_t - jt_ind - jt_bs, geom)
    cp["j_ohmic"] = (s_I * ohm).tolist()
    cp["j_bootstrap"] = (s_I * bs).tolist()
    cp["j_total"] = (s_I * (ohm + bs + drv)).tolist()
    if "j_non_inductive" in cp:
        cp["j_non_inductive"] = (s_I * (bs + drv)).tolist()

    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out_path


def export_imas_drawset(h5path_or_header, template_ids_path, out_dir,
                        scan_key=None, time=None, selection="selected",
                        fidelity="auto"):
    """Write one perturbed IMAS/OMAS IDS per draw (see :func:`write_imas_draw`).

    Files are ``{out_dir}/{header}_draw{idx}.json``. ``selection`` is
    ``"selected"`` (in-spec only) or ``"all"``. ``fidelity`` is forwarded to
    :func:`write_imas_draw` (``"auto"`` -> exact per-draw conversion when the
    archive carries a complete ``eq_fsa`` block, else the template geometry).

    Operates on a single scan.  Pass ``scan_key`` to select it; the default
    ``scan_key=None`` is the flat layout, and is only unambiguous when the
    file holds exactly one scan (otherwise raises -- pass an explicit key).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.

    Returns
    -------
    list of str
        The written IDS paths.
    """
    import os
    from ..filtering import select_indices
    from ..utils import _resolve_h5

    os.makedirs(out_dir, exist_ok=True)
    h5 = _resolve_h5(h5path_or_header)
    base = os.path.basename(h5).replace(".h5", "")
    idxs = select_indices(h5, scan_key=scan_key, selection=selection)
    if isinstance(idxs, dict):
        # scan_key=None over a scan-layout file -> ambiguous unless single.
        if len(idxs) != 1:
            raise ValueError(
                f"{h5} holds {len(idxs)} scans {sorted(idxs)}; pass an "
                "explicit scan_key to export_imas_drawset().")
        scan_key, idxs = next(iter(idxs.items()))
    paths = []
    for i in idxs:
        out = os.path.join(out_dir, f"{base}_draw{i}.json")
        paths.append(write_imas_draw(h5, i, template_ids_path, out,
                                     scan_key=scan_key, time=time,
                                     fidelity=fidelity))
    return paths
