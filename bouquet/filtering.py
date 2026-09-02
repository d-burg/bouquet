"""Postprocessing filters for a bouquet HDF5 database.

These routines operate on the stored equilibria *after* a run, so the
user can see the distribution of in-spec / out-of-spec draws before
deciding what to cut.  They are **non-destructive**: each filter writes
per-draw boolean flags into the H5 (``passes_coil_filter``,
``passes_boundary_filter``) and a derived ``selected`` flag (the AND of
all applied filters), but never deletes data.  Re-running a filter with
different thresholds simply overwrites the flags -- no physics is re-run.

Typical workflow
----------------
>>> from bouquet import (filter_coil_currents, filter_boundaries,
...                      plot_bouquet, export_filtered)
>>> # 1. See the coil-current spec distribution and cut on it
>>> summ_c, fig_c = filter_coil_currents("run.h5")        # uses stored thresholds
>>> # 2. See the boundary-deviation distribution (diagnostic only first)
>>> summ_b, fig_b = filter_boundaries("run.h5")           # no threshold -> report only
>>> summ_b, fig_b = filter_boundaries("run.h5", rms_max_mm=5.0)   # then cut
>>> # 3. Re-plot keeping only the selected draws
>>> plot_bouquet("run.h5", selection="selected")
>>> # 4. (optional) materialise a pruned copy
>>> export_filtered("run.h5", "run_selected.h5")
"""
import os

import numpy as np
import h5py
from scipy.spatial import cKDTree as _cKDTree

from .schema import find_bytes_dataset
from .utils import (
    list_equilibrium_indices,
    discover_scan_keys,
    read_eqdsk_from_bytes,
    _scan_key,
    _group_path,
)
from .io import read_geqdsk

__all__ = [
    "filter_coil_currents",
    "filter_coil_chi2",
    "measured_coil_currents",
    "filter_boundaries",
    "read_filter_flags",
    "select_indices",
    "export_filtered",
]

_FILTER_FLAGS = ("passes_coil_filter", "passes_boundary_filter")


# --------------------------------------------------------------------------
#  internal helpers
# --------------------------------------------------------------------------
def _resolve(h5path_or_header):
    # Accept a Bouquet / BouquetArchive / header / path (F1) -- one resolver.
    from .utils import _resolve_h5
    return _resolve_h5(h5path_or_header)


def _require_scan_key(h5path, scan_key):
    """Raise a listing ``KeyError`` when an explicit ``scan_key`` is absent (F1).

    Silent-empty was the old failure mode: a mistyped ``scan_key`` returned no
    draws with no error. ``scan_key=None`` (all scans) is left untouched.
    """
    if scan_key is None:
        return
    svs = discover_scan_keys(h5path)
    if svs is None:                      # flat/legacy file: no scan keys to match
        return
    if _scan_key(scan_key) not in {_scan_key(s) for s in svs}:
        raise KeyError(
            f"scan_key {scan_key!r} not in {os.path.basename(h5path)}; "
            f"available: {svs}")


def _iter_scan_keys(h5path, scan_key):
    if scan_key is not None:
        return [scan_key]
    svs = discover_scan_keys(h5path)
    return svs if svs else [None]


def _finalize_scan_result(result, scan_key):
    """Intent-driven return shape, shared by the selection/filter accessors.

    An explicit ``scan_key`` returns that one scan's value (unwrapped); the
    default ``scan_key=None`` ("all scans") returns the ``{scan_key: value}``
    dict -- even for a single scan. So the shape is predictable from the CALL,
    not from how many scans the file happens to hold: name a key, get one;
    ask for all, get the collection. Applied uniformly across
    ``select_indices`` / ``read_filter_flags`` / the two filters.
    """
    if scan_key is not None:
        return next(iter(result.values()))
    return result


def _recompute_selected(grp):
    """``selected`` = AND of all filter flags present (True if none applied)."""
    flags = [bool(grp.attrs[a]) for a in _FILTER_FLAGS if a in grp.attrs]
    grp.attrs["selected"] = bool(all(flags)) if flags else True


def _write_filter_result(h5path, scan_key, results, flag_name):
    """Persist a {idx: bool} result under ``flag_name`` and refresh selected."""
    with h5py.File(h5path, "a") as hf:
        for idx, passed in results.items():
            gp = _group_path(scan_key, idx)
            if gp not in hf:
                continue
            grp = hf[gp]
            grp.attrs[flag_name] = bool(passed)
            _recompute_selected(grp)


def _baseline_boundary(hf, scan_key):
    """Recon baseline LCFS (recon_lcfs_ref preferred, else eqdsk boundary)."""
    bkey = _scan_key(scan_key)
    bl_path = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
    if bl_path not in hf:
        return None
    bl = hf[bl_path]
    if "recon_lcfs_ref" in bl:
        try:
            return np.asarray(bl["recon_lcfs_ref"][()], dtype=float)
        except Exception:
            pass
    eqk = find_bytes_dataset(bl)
    if eqk is None:
        return None
    try:
        eq = read_eqdsk_from_bytes(bytes(bl[eqk][()]), read_geqdsk)
        return np.column_stack([eq.boundary_R, eq.boundary_Z])
    except Exception:
        return None


def _boundary_devs(bl_boundary, grp):
    """(rms_mm, max_mm) of this draw's LCFS vs the baseline recon boundary.

    Uses the identical metric as ``plot_traces``: nearest-neighbour
    distance from each baseline point to the perturbed boundary
    (prefers the 10k-pt ``perturbed_lcfs_ref``, falls back to the
    eqdsk's coarse RBBBS/ZBBBS).
    """
    if bl_boundary is None:
        return (np.nan, np.nan)
    perturbed = None
    if "perturbed_lcfs_ref" in grp:
        try:
            perturbed = np.asarray(grp["perturbed_lcfs_ref"][()], dtype=float)
        except Exception:
            perturbed = None
    if perturbed is None:
        eqk = find_bytes_dataset(grp)
        if eqk is None:
            return (np.nan, np.nan)
        try:
            eq = read_eqdsk_from_bytes(bytes(grp[eqk][()]), read_geqdsk)
            perturbed = np.column_stack([eq.boundary_R, eq.boundary_Z])
        except Exception:
            return (np.nan, np.nan)
    try:
        tree = _cKDTree(perturbed)
        devs, _ = tree.query(bl_boundary)
        return (float(np.sqrt(np.mean(devs ** 2)) * 1e3),
                float(np.max(devs) * 1e3))
    except Exception:
        return (np.nan, np.nan)


# --------------------------------------------------------------------------
#  selection accessors
# --------------------------------------------------------------------------
def read_filter_flags(h5path_or_header, scan_key=None):
    """Per-draw filter flags: ``{idx: {flag: value, ...}}`` for stored draws.

    Each inner dict carries ``passes_coil_filter``, ``passes_boundary_filter``
    and ``selected`` when present (missing flags are omitted), plus the raw
    metrics ``max_F_drift_pct``, ``max_VSC_drift_pct``, ``in_spec``. An explicit
    ``scan_key`` returns the ``{idx: {...}}`` dict directly; the default
    ``scan_key=None`` returns ``{scan_key: {idx: {...}}}`` (shape follows the
    call, same convention as the other accessors).
    """
    h5path = _resolve(h5path_or_header)
    out = {}
    with h5py.File(h5path, "r") as hf:
        for sv in _iter_scan_keys(h5path, scan_key):
            sv_out = {}
            for i in list_equilibrium_indices(h5path, scan_key=sv):
                gp = _group_path(sv, i)
                if gp not in hf:
                    continue
                a = hf[gp].attrs
                rec = {}
                for k in (*_FILTER_FLAGS, "selected"):
                    if k in a:
                        rec[k] = bool(a[k])
                for k in ("max_F_drift_pct", "max_VSC_drift_pct", "in_spec"):
                    if k in a:
                        rec[k] = (float(a[k]) if k != "in_spec"
                                  else bool(a[k]))
                sv_out[i] = rec
            out[sv] = sv_out
    return _finalize_scan_result(out, scan_key)


def select_indices(h5path_or_header, scan_key=None, selection="selected"):
    """Stored draw indices filtered by ``selection``.

    ``selection`` is one of:

      - ``'all'``      : every stored draw
      - ``'selected'`` : draws with ``selected`` True; if NO filter has
        been applied yet (no ``selected`` attr anywhere), returns all
        draws (so plotting selected on an unfiltered file shows all).
      - ``'excluded'`` : the complement of ``'selected'`` among stored
        draws (empty on an unfiltered file).

    With an explicit ``scan_key`` returns a flat sorted list for that scan;
    with the default ``scan_key=None`` returns ``{scan_key: [idx, ...]}`` over
    all scans (shape follows the call, not the file's scan count).
    """
    if selection not in ("all", "selected", "excluded"):
        raise ValueError(
            f"selection must be 'all'|'selected'|'excluded', got {selection!r}")
    from .utils import _default_scan_key
    scan_key = _default_scan_key(h5path_or_header, scan_key)
    h5path = _resolve(h5path_or_header)
    _require_scan_key(h5path, scan_key)
    svs = _iter_scan_keys(h5path, scan_key)
    result = {}
    with h5py.File(h5path, "r") as hf:
        for sv in svs:
            idxs = list_equilibrium_indices(h5path, scan_key=sv)
            if selection == "all":
                result[sv] = idxs
                continue
            have_any = any("selected" in hf[_group_path(sv, i)].attrs
                           for i in idxs if _group_path(sv, i) in hf)
            if not have_any:
                # unfiltered: 'selected' -> all, 'excluded' -> none
                result[sv] = idxs if selection == "selected" else []
                continue
            sel = [i for i in idxs
                   if bool(hf[_group_path(sv, i)].attrs.get("selected", True))]
            if selection == "selected":
                result[sv] = sel
            else:
                result[sv] = [i for i in idxs if i not in set(sel)]
    return _finalize_scan_result(result, scan_key)


# --------------------------------------------------------------------------
#  the two filters
# --------------------------------------------------------------------------
def measured_coil_currents(dd_path, time_s):
    """``{name: (current_A, sigma_A)}`` from an IMAS ``pf_active`` at *time_s*.

    ``data_error_upper`` is read as the 1-sigma magnitude (FUSE writes the error
    itself there, not ``data + error``).
    """
    import json

    with open(dd_path) as fh:
        dd = json.load(fh)
    out = {}
    for c in dd.get("pf_active", {}).get("coil", []):
        name = c.get("name") or c.get("identifier")
        cur = c.get("current") or {}
        if not name or "data" not in cur or "time" not in cur:
            continue
        t = np.asarray(cur["time"], dtype=float)
        d = np.asarray(cur["data"], dtype=float)
        e = cur.get("data_error_upper")
        if e is None:
            continue
        e = np.abs(np.asarray(e, dtype=float))
        j = int(np.argmin(np.abs(t - float(time_s))))
        out[name] = (float(d[j]), float(e[j]))
    return out


def filter_coil_chi2(h5path_or_header, dd_path=None, scan_key=None,
                     chi2_max=4.0, apply=True, sigma=None, device=None,
                     shot=None, sigma_ref=None, z_max=5.0):
    """Measurement-referenced coil filter (see :mod:`bouquet.coil_spec`).

    Scores each draw by ``chi2/nu`` of its coil currents against the baseline,
    weighted by the measured per-coil precision from ``pf_active``, and cuts at
    *chi2_max*. Unlike :func:`filter_coil_currents` this covers EVERY coil the
    measurement carries (E-coils included) and weights each by its own
    precision rather than a flat fractional band.

    ``chi2_max=4.0`` is RMS |z| = 2 -- "within 2 sigma per coil on average" --
    and reproduces the legacy filter's acceptance rate on the DIII-D 174823
    ensemble (91%) while disagreeing on ~11% of individual draws.

    A draw with no usable coil scores NaN and FAILS: unjudgeable is not a pass.

    Per-coil sigma (see :func:`coil_spec.resolve_coil_sigma`): ``sigma`` explicit
    (``{"floor","fraction"}`` model, per-coil table, or callable), else the
    ``device`` model (named, or detected from the mesh coil names), else
    :class:`coil_spec.CoilSigmaUnavailable` is raised.  Needs no dd.

    ``sigma_ref`` (legacy, dd-referenced; requires ``dd_path``): ``"d3d"`` uses the
    digitizer table rescaled by |I_base|/|I_meas|; a dict ``{"F": A, "E": A}``
    a custom per-family table; ``"dd"`` the dd's own ``data_error_upper``.
    Raises :class:`ValueError` if given without ``dd_path``.

    A draw passes when ``chi2/nu <= chi2_max`` AND its worst single coil has
    ``|z| <= z_max`` (``None`` disables the guard): the pooled statistic alone
    lets one coil at 7 sigma hide behind seventeen quiet ones.

    The model used is stamped on the scan group as ``coil_sigma_model`` (JSON)
    and ``coil_filter`` = "chi2" when ``apply`` is True.
    """
    import json
    from .coil_spec import (coil_chi2, coil_sigma_in_base_units, with_sigma_ref,
                            resolve_coil_sigma)
    if sigma_ref is not None and dd_path is None:
        raise ValueError("filter_coil_chi2: sigma_ref is dd-referenced and needs dd_path")

    h5path = _resolve(h5path_or_header)
    summary = {}
    for sv in _iter_scan_keys(h5path, scan_key):
        with h5py.File(h5path, "r") as hf:
            grp = hf["scan"][str(sv)]
            if "_baseline" not in grp:
                continue
            bn = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in grp["_baseline"]["coil_names"][()]]
            baseline = dict(zip(bn, np.asarray(
                grp["_baseline"]["coil_currents"][()], dtype=float).tolist()))
            if sigma_ref is not None:
                meas = measured_coil_currents(dd_path, float(sv))
                if sigma_ref != "dd":
                    meas = with_sigma_ref(meas, None if sigma_ref == "d3d" else sigma_ref)
                sig, model = coil_sigma_in_base_units(baseline, meas), {"kind": "dd_referenced", "sigma_ref": str(sigma_ref)}
            else:
                sig, model = resolve_coil_sigma(baseline, sigma=sigma, device=device, shot=shot)
            rows = {}
            for key in sorted((k for k in grp if k.isdigit()), key=int):
                g = grp[key]
                if "coil_currents" not in g:
                    continue
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in g["coil_names"][()]]
                draw = dict(zip(names, np.asarray(
                    g["coil_currents"][()], dtype=float).tolist()))
                rows[int(key)] = coil_chi2(draw, baseline, sig)
        results = {i: bool(np.isfinite(r["chi2_nu"]) and r["chi2_nu"] <= chi2_max
                           and (z_max is None or r["max_abs_z"] <= z_max))
                   for i, r in rows.items()}
        if apply:
            _write_filter_result(h5path, sv, results, "passes_coil_filter")
            with h5py.File(h5path, "a") as hf:
                gp = f"scan/{sv}" if f"scan/{sv}" in hf else None
                if gp is not None:
                    hf[gp].attrs["coil_filter"] = "chi2"
                    hf[gp].attrs["coil_sigma_model"] = json.dumps(model)
        n_pass = sum(results.values())
        summary[sv] = {
            "n_total": len(results), "n_pass": n_pass,
            "n_fail": len(results) - n_pass, "chi2_max": float(chi2_max),
            "z_max": (None if z_max is None else float(z_max)), "sigma_model": model,
            "n_coils": (max((r["nu"] for r in rows.values()), default=0)),
            "draws": {i: {"chi2_nu": r["chi2_nu"], "max_abs_z": r["max_abs_z"],
                          "worst_coil": r["worst_coil"], "nu": r["nu"],
                          "passes": results[i]} for i, r in rows.items()},
        }
    return _finalize_scan_result(summary, scan_key)


def filter_coil_currents(h5path_or_header, scan_key=None,
                          F_max_pct=None, VSC_max_pct=None,
                          apply=True, plot=True):
    """Coil-current spec filter (the binding in-spec constraint).

    Compares each draw's stored ``max_F_drift_pct`` and
    ``max_VSC_drift_pct`` against thresholds.  Thresholds default to the
    per-draw ``inspec_F_max`` / ``inspec_VSC_max`` recorded at generation
    time (so the postprocess decision reproduces the in-loop one); pass
    ``F_max_pct`` / ``VSC_max_pct`` (in percent) to re-cut at a different
    tolerance without re-running any physics.

    Parameters
    ----------
    apply : bool
        When True (default) write ``passes_coil_filter`` + refresh
        ``selected`` in the H5.  When False, compute + plot only.
    plot : bool
        When True (default) return a distribution figure.

    Returns
    -------
    (summary, fig)
        An explicit ``scan_key`` gives a single ``{...counts, 'draws':
        {idx: {...}}}`` dict; the default ``scan_key=None`` gives
        ``{scan_key: {...}}`` across all scans (same convention as
        :func:`select_indices`). ``fig`` is the distribution figure (or None).
    """
    h5path = _resolve(h5path_or_header)
    summary = {}
    fig_payload = []
    for sv in _iter_scan_keys(h5path, scan_key):
        idxs = list_equilibrium_indices(h5path, scan_key=sv)
        rows = []
        with h5py.File(h5path, "r") as hf:
            for i in idxs:
                a = hf[_group_path(sv, i)].attrs
                F = float(a.get("max_F_drift_pct", np.nan))
                V = float(a.get("max_VSC_drift_pct", np.nan))
                fthr = (F_max_pct if F_max_pct is not None
                        else float(a.get("inspec_F_max", np.nan)) * 100.0)
                vthr = (VSC_max_pct if VSC_max_pct is not None
                        else float(a.get("inspec_VSC_max", np.nan)) * 100.0)
                rows.append((i, F, V, fthr, vthr))
        results = {}
        draws = {}
        for i, F, V, fthr, vthr in rows:
            ok_F = np.isfinite(F) and np.isfinite(fthr) and F <= fthr
            ok_V = np.isfinite(V) and np.isfinite(vthr) and V <= vthr
            passed = bool(ok_F and ok_V)
            results[i] = passed
            draws[i] = {"max_F_drift_pct": F, "max_VSC_drift_pct": V,
                        "F_max_pct": fthr, "VSC_max_pct": vthr,
                        "passes": passed}
        if apply:
            _write_filter_result(h5path, sv, results, "passes_coil_filter")
        n_pass = sum(results.values())
        summary[sv] = {"n_total": len(results), "n_pass": n_pass,
                       "n_fail": len(results) - n_pass, "draws": draws}
        fig_payload.append((sv, rows, results))
    fig = _plot_coil_filter(fig_payload) if plot else None
    return _finalize_scan_result(summary, scan_key), fig


def filter_boundaries(h5path_or_header, scan_key=None,
                       rms_max_mm=None, max_max_mm=None,
                       apply=True, plot=True):
    """LCFS boundary-deviation filter.

    Computes each draw's RMS and max LCFS deviation from the recon
    baseline (same metric as ``plot_traces``).  With **no threshold**
    given this is purely diagnostic -- it reports the distribution and
    does NOT write any flag, so you can read a sensible cut off the plot
    first.  Pass ``rms_max_mm`` and/or ``max_max_mm`` (millimetres) to
    actually cut; a draw passes when it satisfies every supplied bound.

    Returns ``(summary, fig)`` analogous to :func:`filter_coil_currents`.
    """
    h5path = _resolve(h5path_or_header)
    cutting = (rms_max_mm is not None) or (max_max_mm is not None)
    summary = {}
    fig_payload = []
    for sv in _iter_scan_keys(h5path, scan_key):
        idxs = list_equilibrium_indices(h5path, scan_key=sv)
        rows = []
        with h5py.File(h5path, "r") as hf:
            bl = _baseline_boundary(hf, sv)
            for i in idxs:
                rms, mx = _boundary_devs(bl, hf[_group_path(sv, i)])
                rows.append((i, rms, mx))
        results = {}
        draws = {}
        for i, rms, mx in rows:
            ok_rms = (rms_max_mm is None) or (np.isfinite(rms)
                                              and rms <= rms_max_mm)
            ok_max = (max_max_mm is None) or (np.isfinite(mx)
                                              and mx <= max_max_mm)
            passed = bool(ok_rms and ok_max)
            results[i] = passed
            draws[i] = {"rms_mm": rms, "max_mm": mx, "passes": passed}
        if apply and cutting:
            _write_filter_result(h5path, sv, results, "passes_boundary_filter")
        n_pass = sum(results.values())
        summary[sv] = {"n_total": len(results),
                       "n_pass": n_pass if cutting else len(results),
                       "n_fail": (len(results) - n_pass) if cutting else 0,
                       "cutting": cutting,
                       "rms_stats": _stats([r[1] for r in rows]),
                       "max_stats": _stats([r[2] for r in rows]),
                       "draws": draws}
        fig_payload.append((sv, rows, results, rms_max_mm, max_max_mm, cutting))
    fig = _plot_boundary_filter(fig_payload) if plot else None
    return _finalize_scan_result(summary, scan_key), fig


def _stats(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"median": np.nan, "p95": np.nan, "max": np.nan}
    return {"median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


# --------------------------------------------------------------------------
#  export
# --------------------------------------------------------------------------
def export_filtered(h5path_or_header, out_path, scan_key=None,
                    selection="selected", overwrite=False):
    """Write a pruned copy of the database keeping only ``selection`` draws.

    Baseline groups and all top-level metadata are preserved; only the
    excluded per-draw groups are removed.  Indices are NOT renumbered, so
    the surviving draws keep their original labels (gaps are expected).
    The source file is never modified.

    Returns the number of draws kept (summed over scan values).
    """
    import shutil
    src = _resolve(h5path_or_header)
    out = os.path.abspath(out_path)
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(
            f"{out} exists; pass overwrite=True to replace it")
    shutil.copy(src, out)
    kept = 0
    with h5py.File(out, "a") as hf:
        for sv in _iter_scan_keys(out, scan_key):
            keep = set(select_indices(out, scan_key=sv, selection=selection))
            for i in list_equilibrium_indices(out, scan_key=sv):
                if i in keep:
                    kept += 1
                else:
                    gp = _group_path(sv, i)
                    if gp in hf:
                        del hf[gp]
    return kept


# --------------------------------------------------------------------------
#  distribution plots
# --------------------------------------------------------------------------
def _bar_colors(passes):
    return ["#2ca02c" if p else "#d62728" for p in passes]


def _safe_legend(ax, **kw):
    """Add a legend only if there are labelled artists (avoids warnings)."""
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8, **kw)


def _plot_coil_filter(payload):
    import matplotlib.pyplot as plt
    n_sv = len(payload)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # Aggregate all draws across scan values for the histograms
    allF, allV, allP = [], [], []
    for sv, rows, results in payload:
        idx = [r[0] for r in rows]
        F = [r[1] for r in rows]
        V = [r[2] for r in rows]
        P = [results[r[0]] for r in rows]
        allF += F
        allV += V
        allP += P
        x = np.arange(len(idx))
        axes[0, 0].bar(x, F, color=_bar_colors(P), alpha=0.85)
        axes[1, 0].bar(x, V, color=_bar_colors(P), alpha=0.85)
        if rows:
            fthr = rows[0][3]
            vthr = rows[0][4]
            axes[0, 0].axhline(fthr, color="k", ls="--", lw=1.0,
                               label=f"F threshold {fthr:.2f}%")
            axes[1, 0].axhline(vthr, color="k", ls="--", lw=1.0,
                               label=f"VSC threshold {vthr:.2f}%")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([str(j) for j in idx], fontsize=7)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([str(j) for j in idx], fontsize=7)

    n_pass = int(np.sum(allP))
    n_tot = len(allP)
    axes[0, 0].set_title(f"Per-draw max F-coil drift  "
                         f"({n_pass}/{n_tot} in-spec)")
    axes[0, 0].set_ylabel("max F drift [%]")
    axes[0, 0].set_xlabel("stored draw index")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    axes[1, 0].set_title("Per-draw max VSC drift")
    axes[1, 0].set_ylabel("max VSC drift [%]")
    axes[1, 0].set_xlabel("stored draw index")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # Histograms (distribution view)
    _hist_split(axes[0, 1], allF, allP, "max F drift [%]",
                payload[0][1][0][3] if payload and payload[0][1] else None)
    _hist_split(axes[1, 1], allV, allP, "max VSC drift [%]",
                payload[0][1][0][4] if payload and payload[0][1] else None)
    axes[0, 1].set_title("Distribution (green=in-spec, red=out)")
    fig.suptitle("Coil-current spec filter", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _plot_boundary_filter(payload):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    allR, allM, allP = [], [], []
    for sv, rows, results, rms_thr, max_thr, cutting in payload:
        idx = [r[0] for r in rows]
        R = [r[1] for r in rows]
        M = [r[2] for r in rows]
        P = [results[r[0]] for r in rows]
        allR += R
        allM += M
        allP += P
        x = np.arange(len(idx))
        colR = _bar_colors(P) if cutting else "#1f77b4"
        colM = _bar_colors(P) if cutting else "#ff7f0e"
        axes[0, 0].bar(x, R, color=colR, alpha=0.85)
        axes[1, 0].bar(x, M, color=colM, alpha=0.85)
        if rms_thr is not None:
            axes[0, 0].axhline(rms_thr, color="k", ls="--", lw=1.0,
                               label=f"RMS threshold {rms_thr:.2f} mm")
        if max_thr is not None:
            axes[1, 0].axhline(max_thr, color="k", ls="--", lw=1.0,
                               label=f"max threshold {max_thr:.2f} mm")
        for ax in (axes[0, 0], axes[1, 0]):
            ax.set_xticks(x)
            ax.set_xticklabels([str(j) for j in idx], fontsize=7)

    cutting_any = any(p[5] for p in payload)
    title_tail = ""
    if cutting_any:
        n_pass = int(np.sum(allP))
        title_tail = f"  ({n_pass}/{len(allP)} pass)"
    axes[0, 0].set_title("Per-draw boundary RMS deviation" + title_tail)
    axes[0, 0].set_ylabel("RMS dev. [mm]")
    axes[0, 0].set_xlabel("stored draw index")
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    _safe_legend(axes[0, 0])
    axes[1, 0].set_title("Per-draw boundary max deviation")
    axes[1, 0].set_ylabel("max dev. [mm]")
    axes[1, 0].set_xlabel("stored draw index")
    axes[1, 0].grid(True, axis="y", alpha=0.3)
    _safe_legend(axes[1, 0])

    rms_thr0 = next((p[3] for p in payload if p[3] is not None), None)
    max_thr0 = next((p[4] for p in payload if p[4] is not None), None)
    _hist_split(axes[0, 1], allR, allP if cutting_any else None,
                "RMS dev. [mm]", rms_thr0)
    _hist_split(axes[1, 1], allM, allP if cutting_any else None,
                "max dev. [mm]", max_thr0)
    axes[0, 1].set_title("Distribution")
    fig.suptitle("Boundary-deviation filter", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _hist_split(ax, vals, passes, xlabel, threshold):
    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    if passes is None:
        ax.hist(vals[finite], bins=12, color="#4c72b0", alpha=0.85)
    else:
        passes = np.asarray(passes, dtype=bool)
        good = vals[finite & passes]
        bad = vals[finite & ~passes]
        bins = np.histogram_bin_edges(vals[finite], bins=12) \
            if finite.any() else 12
        if good.size:
            ax.hist(good, bins=bins, color="#2ca02c", alpha=0.8,
                    label="in-spec")
        if bad.size:
            ax.hist(bad, bins=bins, color="#d62728", alpha=0.8,
                    label="out-of-spec")
        if good.size or bad.size:
            ax.legend(fontsize=8)
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
