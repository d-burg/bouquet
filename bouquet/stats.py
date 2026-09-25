r"""Standard error bars from a bouquet archive: one recipe, one record.

A bouquet is an ensemble of perturbed equilibria. Any quantity computed per
draw -- by bouquet itself (:func:`draw_scalars`) or by an external code the
caller runs on each draw (:func:`draw_band`) -- gets its uncertainty band from
the same recipe, and every band comes back as a :class:`BandRecord` that says
exactly which draws went into it and why the others did not.

The recipe
----------
**Population.** Start from the draws the archive's stamped filters mark
``selected`` (the AND of the applied coil and boundary flags). An archive with
no ``coil_filter`` stamp would silently report every stored draw as selected,
so the default ``require_filter=True`` raises on it. The recipe is read-only:
it never re-filters. Then, per quantity, keep draws the evaluator reports with
status ``"ok"``, that are *regular* (see the pole rule), and whose value is
finite. Every removal is kept in ``dropped`` with its reason
(``not_selected``, ``user:<reason>``, ``status:<code>``,
``irregular:<label>``, ``non_finite``) and counted at each stage
(``n_requested`` -> ``n_attempted`` -> ``n_stored`` -> ``n_selected`` ->
``n_evaluated`` -> ``n_ok`` -> ``n_regular`` -> ``n_used``). Draws that failed
before they were archived are not in the draw list: ``n_attempted`` is read
from the archive when it was recorded and is ``None`` otherwise -- it is never
inferred from gaps in the draw indices.

**Statistic.** The median and the ``percentiles`` (default 16 and 84, numpy
``method="linear"``, i.e. Hyndman-Fan type 7), plus the min and max, over the
``n_used`` values. Not mean +/- sigma: quantities with a pole (Delta', growth
rates near marginality) are skewed and heavy-tailed, and a mean is dragged by
the one draw that sits near the pole. Mean and std are reported for
information, with ``skew_flag = |mean - median| > 0.25 std``; for a smooth,
near-symmetric global scalar (l_i, <P>, beta_N) with ``skew_flag`` False they
are fine to quote.

**Small n.** With linear interpolation the 16-84 band of ``n`` samples covers
about ``0.68 (n-1)/(n+1)`` of the distribution, not 0.68 (0.45 at n=5, 0.60 at
n=15), and the chance that the true 16th percentile lies below the sample
minimum is ``0.84**n``. Below ``min_n`` (default 15) the record is flagged
``below_floor`` -- shown, with an open marker, never suppressed. Below
``hard_min`` (default 5) no percentiles are returned (``no_band``): only the
median, min, max and the per-draw values.

**Pole rule** ``"majority_regular"``. The evaluator returns, per draw and per
quantity, a ``regular`` boolean whose meaning the caller defines (W_t > 0, the
rational surface exists, ...); bouquet supplies none. Irregular draws leave the
statistic with reason ``irregular:<label>`` and are counted, never deleted. If
half or fewer of the ok draws are regular (``regular_fraction <= 1/2``) the
record is ``gated``: the band is still computed and returned, but
``show=False`` and :func:`plot_band` draws the fraction instead of a band.
There is **no magnitude cut**: ``n_extreme`` counts values further than
10 MAD from the median (unscaled MAD) as a diagnostic, and they stay in. Hand
exclusions go only through ``exclude={draw: reason}`` and are recorded as
``user:<reason>``; the standard recipe passes none.

**Baseline overlay.** The same evaluator is applied to the scan's baseline
through a :class:`~bouquet.archive.BaselineView` (same byte and profile
accessors, so the same code path and grid). The baseline is overlaid, never
used as the centre: the record reports ``baseline_value``,
``baseline_quantile`` (fraction of used draws below it) and
``baseline_outside_range``. A baseline outside every draw has been the
signature of a draw-path bug before.

**Limitations** are recorded on every result. Only the perturbations bouquet
samples enter the band; anything the evaluator holds fixed (rotation,
transport coefficients, grid choices) makes the band conditional on it.

Archive fields that older archives do not carry (``n_requested``,
``n_attempted``, ``generation_mode``, the boundary thresholds, the per-scan
``bouquet_version``, the closure verdict) come back as ``None`` or
``"unrecorded"`` -- never guessed.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

__all__ = [
    "BandRecord", "BandTable", "UnfilteredArchiveError",
    "draw_band", "draw_bands", "draw_scalars", "plot_band",
]

UNRECORDED = "unrecorded"
POLE_RULES = ("majority_regular",)
_SELECTIONS = ("selected", "all", "excluded")
_BASELINE_KEYS = ("_baseline", "baseline")

LIMITATION_SAMPLED = (
    "Only the perturbations bouquet samples enter the band; any input the "
    "evaluator holds fixed (for example rotation, transport coefficients or "
    "grid choices) makes the band conditional on that input.")


class UnfilteredArchiveError(ValueError):
    """The scan carries no ``coil_filter`` stamp, so "selected" means "all"."""


# --------------------------------------------------------------------------
#  record + table
# --------------------------------------------------------------------------
_NAN = float("nan")


@dataclass(repr=False)
class BandRecord:
    """One scan key x one quantity: the band, its population and provenance.

    ``p16`` / ``p84`` hold the lower / upper of ``percentiles`` (named for the
    default ``(16, 84)``). ``status`` is ``"ok"``, ``"refused"`` (the scan
    group carries a ``refused_reason``) or ``"no_archive"`` (the archive or the
    key is absent).
    """

    scan_key: Optional[str]
    quantity: Optional[str]
    status: str = "ok"
    # ---- counts, stage by stage
    n_requested: Optional[int] = None
    n_requested_source: str = UNRECORDED
    n_attempted: Optional[int] = None
    n_stored: int = 0
    n_selected: int = 0
    n_evaluated: int = 0
    n_ok: int = 0
    n_regular: int = 0
    n_used: int = 0
    regular_fraction: float = _NAN
    # ---- statistic
    median: float = _NAN
    p16: float = _NAN
    p84: float = _NAN
    min: float = _NAN
    max: float = _NAN
    mean: float = _NAN
    std: float = _NAN
    skew_flag: Optional[bool] = None
    percentiles: tuple = (16, 84)
    percentile_method: str = "linear"
    # ---- flags
    below_floor: bool = True
    no_band: bool = True
    gated: bool = False
    show: bool = False
    n_extreme: int = 0
    # ---- per draw
    values: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)
    # ---- baseline overlay
    baseline_value: Optional[float] = None
    baseline_status: str = "not_requested"
    baseline_quantile: Optional[float] = None
    baseline_outside_range: Optional[bool] = None
    # ---- closure (per slice)
    closure_limited: Optional[bool] = None
    closure_limited_reasons: tuple = ()
    closure_channel: Optional[str] = None
    # ---- slice-level
    refused_reason: Optional[str] = None
    archive: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain JSON-able dict (tuples -> lists, numpy scalars -> Python)."""
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    def __repr__(self):
        head = f"<BandRecord {self.scan_key!r}/{self.quantity!r}"
        if self.status != "ok":
            why = f" ({self.refused_reason})" if self.refused_reason else ""
            return f"{head} {self.status}{why}>"
        flags = [n for n in ("below_floor", "no_band", "gated") if getattr(self, n)]
        if not self.show:
            flags.append("hidden")
        band = ("" if self.no_band else
                f" [{self.p16:.4g}, {self.p84:.4g}]")
        return (f"{head} median={self.median:.4g}{band} "
                f"n_used={self.n_used}/{self.n_stored} stored"
                f"{' ' + ','.join(flags) if flags else ''}>")


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k) if not isinstance(k, (int, str)) else k: _jsonable(x)
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# columns written flat to CSV; the rest are JSON-encoded
_JSON_COLS = ("percentiles", "values", "dropped", "closure_limited_reasons",
              "provenance")


class BandTable:
    """A list of :class:`BandRecord` (one per requested key x quantity)."""

    def __init__(self, records):
        self.records = list(records)

    def __repr__(self):
        return (f"<BandTable {len(self.records)} records, "
                f"quantities={self.quantities}>")

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def quantities(self) -> list:
        out = []
        for r in self.records:
            if r.quantity is not None and r.quantity not in out:
                out.append(r.quantity)
        return out

    def select(self, quantity) -> list:
        """Records for one quantity (plus key-level records with no quantity)."""
        return [r for r in self.records if r.quantity in (quantity, None)]

    def to_dataframe(self):
        """A ``pandas.DataFrame`` (one row per record); needs pandas."""
        import pandas as pd
        return pd.DataFrame([_flat_row(r) for r in self.records])

    def to_csv(self, path) -> str:
        """Write one row per record (no pandas). Floats keep full precision
        (``repr``); ``values`` / ``dropped`` / ``provenance`` are JSON."""
        names = [f.name for f in fields(BandRecord)]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=names)
            w.writeheader()
            for r in self.records:
                w.writerow(_flat_row(r))
        return str(path)

    @staticmethod
    def read_csv(path) -> list:
        """Read a :meth:`to_csv` file back as a list of dicts (values decoded)."""
        out = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out.append({k: _parse_cell(k, v) for k, v in row.items()})
        return out


def _flat_row(r: BandRecord) -> dict:
    d = r.to_dict()
    row = {}
    for k, v in d.items():
        if k in _JSON_COLS:
            row[k] = json.dumps(v)
        elif isinstance(v, float):
            row[k] = repr(v)
        elif v is None:
            row[k] = ""
        else:
            row[k] = v
    return row


_INT_COLS = {"n_requested", "n_attempted", "n_stored", "n_selected",
             "n_evaluated", "n_ok", "n_regular", "n_used", "n_extreme"}
_BOOL_COLS = {"skew_flag", "below_floor", "no_band", "gated", "show",
              "baseline_outside_range", "closure_limited"}
_FLOAT_COLS = {"regular_fraction", "median", "p16", "p84", "min", "max",
               "mean", "std", "baseline_value", "baseline_quantile"}


def _parse_cell(k, v):
    if k in _JSON_COLS:
        return json.loads(v)
    if v == "":
        return None
    if k in _INT_COLS:
        return int(v)
    if k in _FLOAT_COLS:
        return float(v)
    if k in _BOOL_COLS:
        return v == "True"
    return v


# --------------------------------------------------------------------------
#  archive context (read-only; optional fields -> None / "unrecorded")
# --------------------------------------------------------------------------
def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _as_archive(archive):
    from .archive import BouquetArchive
    return archive if isinstance(archive, BouquetArchive) else BouquetArchive(archive)


def _resolve_key(ar, scan_key):
    """The archive's own spelling of ``scan_key`` (KeyError if absent)."""
    from .utils import _scan_key
    keys = ar.scan_keys
    if keys == [None]:
        if scan_key is None:
            return None
        raise KeyError(f"flat archive has no scan keys; got {scan_key!r}")
    if scan_key is None:
        if len(keys) != 1:
            raise KeyError(f"{len(keys)} scans present {keys}; pass a scan_key")
        return keys[0]
    want = _scan_key(scan_key)
    for k in keys:
        if _scan_key(k) == want:
            return k
    raise KeyError(f"scan_key {scan_key!r} not in {keys}")


def _json_attr(v):
    v = _decode(v)
    if v is None:
        return None
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def _sigma_model_summary(model):
    if not isinstance(model, dict):
        return None
    acc = model.get("acceptance") or {}
    out = {"chi2_max": acc.get("chi2_max"), "z_max": acc.get("z_max"),
           "era": model.get("era")}
    for k in ("kind", "device"):
        if k in model:
            out[k] = model[k]
    if "source" in acc:
        out["acceptance_source"] = acc["source"]
    return out


def _scan_context(ar, key) -> dict:
    """Everything slice-level the record needs, read once."""
    import h5py
    from .utils import _scan_key, list_equilibrium_indices, load_baseline_profiles
    from . import __version__
    bkey = _scan_key(key)
    gp = f"scan/{bkey}" if bkey is not None else "/"
    with h5py.File(ar.path, "r") as hf:
        sattrs = {k: _decode(v) for k, v in hf[gp].attrs.items()}
        if bkey is None:                    # flat layout: root attrs are file-level
            sattrs.pop("bouquet_version", None)
        file_version = _decode(hf.attrs.get("bouquet_version"))
    ctx = {"attrs": sattrs, "refused_reason": sattrs.get("refused_reason")}
    if ctx["refused_reason"] is not None:
        ctx["refused_reason"] = str(ctx["refused_reason"])
        return ctx
    ctx["indices"] = list_equilibrium_indices(ar.path, scan_key=key)

    # ---- filters actually stamped
    cf = sattrs.get("coil_filter")
    ctx["coil_filter"] = str(cf) if cf is not None else None
    ctx["coil_sigma_model"] = _sigma_model_summary(
        _json_attr(sattrs.get("coil_sigma_model")))
    ctx["rms_max_mm"] = sattrs.get("boundary_rms_max_mm", UNRECORDED)
    ctx["max_max_mm"] = sattrs.get("boundary_max_max_mm", UNRECORDED)

    # ---- counts recorded at generation
    n_req, n_req_src, mode = None, UNRECORDED, sattrs.get("generation_mode")
    if sattrs.get("n_requested") is not None:
        n_req, n_req_src = int(sattrs["n_requested"]), "archive:n_requested"
    else:
        cfg, cfg_src = _load_scan_config(ar, key)
        if cfg is not None:
            gen = cfg.generation
            if getattr(gen, "n_inspec_target", None) is not None:
                n_req = int(gen.n_inspec_target)
                n_req_src = f"{cfg_src}:generation.n_inspec_target"
                mode = mode if mode is not None else "until_n (from config)"
            else:
                n_req = int(gen.n_equils)
                n_req_src = f"{cfg_src}:generation.n_equils"
                mode = mode if mode is not None else "fixed (from config)"
    ctx["n_requested"], ctx["n_requested_source"] = n_req, n_req_src
    ctx["generation_mode"] = mode if mode is not None else UNRECORDED
    na = sattrs.get("n_attempted")
    ctx["n_attempted"] = int(na) if na is not None else None
    ctx["bouquet_version"] = {
        "scan": str(sattrs["bouquet_version"]) if "bouquet_version" in sattrs
        else UNRECORDED,
        "file": str(file_version) if file_version is not None else UNRECORDED,
        "reader": __version__,
    }

    # ---- baseline / closure
    try:
        bl = load_baseline_profiles(ar.path, scan_key=key)
    except KeyError:
        bl = {}
    ctx["baseline"] = bl
    icl = bl.get("ip_closure") if isinstance(bl.get("ip_closure"), dict) else {}
    cl = bl.get("closure_limited")
    ctx["closure_limited"] = bool(_decode(cl)) if cl is not None else None
    ctx["closure_limited_reasons"] = tuple(icl.get("closure_limited_reasons") or ())
    ch = bl.get("closure_channel", icl.get("closure_channel"))
    ctx["closure_channel"] = str(_decode(ch)) if ch is not None else None
    scale = bl.get("l_i_scale")
    ctx["l_i_scale"] = str(_decode(scale)) if scale is not None else None
    return ctx


def _load_scan_config(ar, key):
    """``(BouquetConfig, source)`` for this slice, or ``(None, None)``.

    The per-scan copy is authoritative; a single-scan archive may also serve
    its root copy (``load_config``'s own rule). Nothing else is guessed.
    """
    from .utils import load_config
    try:
        return load_config(ar.path, scan_key=key), "config_json"
    except Exception:
        pass
    if len(ar.scan_keys) == 1:
        try:
            return load_config(ar.path, scan_key=None), "config_json(root)"
        except Exception:
            pass
    return None, None


def _per_draw_boundary(ar, key, draws):
    """``{draw: boundary_rms_mm}`` where recorded, else ``"unrecorded"``."""
    import h5py
    from .utils import _group_path
    out = {}
    with h5py.File(ar.path, "r") as hf:
        for d in draws:
            a = hf[_group_path(key, d)].attrs
            if "boundary_rms_mm" in a:
                out[int(d)] = float(a["boundary_rms_mm"])
    return out if out else UNRECORDED


# --------------------------------------------------------------------------
#  evaluation
# --------------------------------------------------------------------------
def _lookup_mapping(evaluate, d):
    if d in evaluate:
        return evaluate[d]
    if str(d) in evaluate:
        return evaluate[str(d)]
    return None


def _classify(entry, label_default):
    """``(reason_or_None, value, regular)`` for one draw's quantity entry."""
    if not isinstance(entry, Mapping):
        entry = {"value": entry}
    status = str(entry.get("status", "ok"))
    if status != "ok":
        return f"status:{status}", None, None
    regular = bool(entry.get("regular", True))
    try:
        value = float(entry.get("value"))
    except (TypeError, ValueError):
        value = _NAN
    if not regular:
        label = entry.get("label") or label_default
        return f"irregular:{label}", value, False
    if not math.isfinite(value):
        return "non_finite", value, True
    return None, value, True


def _stat_block(x, *, min_n, hard_min, percentiles, method):
    n = x.size
    out = {"n_used": int(n), "below_floor": n < min_n, "no_band": n < hard_min,
           "n_extreme": 0}
    if n == 0:
        return out
    med = float(np.median(x))
    out.update(median=med, min=float(x.min()), max=float(x.max()),
               mean=float(x.mean()))
    if n >= 2:
        sd = float(x.std(ddof=1))
        out["std"] = sd
        out["skew_flag"] = bool(abs(out["mean"] - med) > 0.25 * sd)
    if n >= hard_min:
        lo, hi = np.percentile(x, percentiles, method=method)
        out["p16"], out["p84"] = float(lo), float(hi)
    mad = float(np.median(np.abs(x - med)))
    out["n_extreme"] = int(np.sum(np.abs(x - med) > 10.0 * mad))
    return out


def _validate(selection, pole_rule, percentiles, min_n, hard_min):
    if selection not in _SELECTIONS:
        raise ValueError(f"selection must be one of {_SELECTIONS}, got {selection!r}")
    if pole_rule not in POLE_RULES:
        raise ValueError(f"pole_rule must be one of {POLE_RULES}, got {pole_rule!r}")
    lo, hi = percentiles
    if not (0 <= lo < hi <= 100):
        raise ValueError(f"percentiles must satisfy 0 <= lo < hi <= 100, got {percentiles!r}")
    if not (1 <= int(hard_min) <= int(min_n)):
        raise ValueError(f"need 1 <= hard_min <= min_n, got hard_min={hard_min}, min_n={min_n}")


def draw_band(archive, scan_key, evaluate, *, quantities=None,
              selection="selected", require_filter=True, min_n=15,
              hard_min=5, pole_rule="majority_regular",
              percentiles=(16, 84), method="linear", exclude=None,
              baseline=True, evaluator_meta=None) -> dict:
    """Across-draw uncertainty band for quantities computed per draw.

    ``evaluate`` is a callable taking a draw view (a
    :class:`~bouquet.archive.DrawView`, or the
    :class:`~bouquet.archive.BaselineView` when ``baseline=True``) and returning
    ``{quantity: {"value", "status", "regular"}}``, or a mapping
    ``{draw_count: that dict}`` of results computed elsewhere (sharded cluster
    runs, CSV ingest; the baseline under key ``"_baseline"``). It is called once
    per selected draw, never on an averaged equilibrium, and bouquet does not
    interpret the quantity. ``status`` defaults to ``"ok"`` and ``regular`` to
    True; a bare number stands for ``{"value": number}``. An optional
    ``"label"`` names the irregularity (else ``evaluator_meta["regular_label"]``).

    Population: draws flagged by ``selection`` (default ``"selected"``, the
    filters stamped on the archive -- an archive with no ``coil_filter`` stamp
    raises :class:`UnfilteredArchiveError` unless ``require_filter=False``),
    minus ``exclude={draw: reason}``; then, per quantity, status ``"ok"``, then
    ``regular``, then a finite value. Every removal is returned with its reason
    in ``dropped`` and counts are reported at each stage, including the
    requested count. Draws that failed before archiving are not visible in the
    draw list.

    Statistic: median and the ``percentiles`` (numpy ``method``), plus min and
    max. The baseline value is returned separately and is never the centre.
    Mean and std are informational; do not quote them for quantities with
    poles or skewed distributions.

    pole_rule ``"majority_regular"``: irregular draws are excluded and counted.
    If half or fewer of the ok draws are regular, the band is computed but
    marked ``gated`` / ``show=False``. No magnitude cut is applied.

    Small n: below ``min_n`` the record is flagged ``below_floor``. Below
    ``hard_min`` no percentiles are returned. With linear interpolation the
    16-84 band covers about 0.68(n-1)/(n+1) of the distribution, not 0.68.

    Limitations recorded on every result: only perturbations bouquet samples
    enter the band. Quantities the evaluator holds fixed (for example rotation
    or transport coefficients) make the band conditional on them.

    Returns ``{quantity: BandRecord}``. A scan carrying ``refused_reason``
    returns records with ``status="refused"`` (keyed ``None`` when
    ``quantities`` is not given). A missing key raises ``KeyError``.
    """
    _validate(selection, pole_rule, percentiles, min_n, hard_min)
    percentiles = tuple(percentiles)
    meta = dict(evaluator_meta or {})
    ar = _as_archive(archive)
    key = _resolve_key(ar, scan_key)
    ctx = _scan_context(ar, key)
    skey = None if key is None else str(key)

    if ctx["refused_reason"] is not None:
        qs = list(quantities) if quantities else [None]
        return {q: BandRecord(scan_key=skey, quantity=q, status="refused",
                              refused_reason=ctx["refused_reason"],
                              archive=ar.path, show=False)
                for q in qs}

    if require_filter and selection != "all" and ctx["coil_filter"] is None:
        raise UnfilteredArchiveError(
            f"scan {skey!r} carries no coil_filter stamp: no filter has been "
            "applied (or it predates the stamp), so selection="
            f"{selection!r} would silently report every stored draw. Run the "
            "filters first (Bouquet.filter / filter_coil_chi2 + "
            "filter_boundaries), or pass require_filter=False to band an "
            "unfiltered population knowingly.")

    from .archive import DrawView, BaselineView
    from .filtering import select_indices

    stored = list(ctx["indices"])
    sel = select_indices(ar.path, scan_key=key, selection=selection)
    if isinstance(sel, dict):               # flat layout: {None: [...]}
        sel = next(iter(sel.values()), [])
    sel = list(sel)
    sel_set = set(sel)
    excl = {int(d): str(r) for d, r in (exclude or {}).items()}
    unknown = sorted(set(excl) - set(stored))
    if unknown:
        raise KeyError(f"exclude names draws not stored in scan {skey!r}: {unknown}")

    common_dropped = []
    to_eval = []
    for d in stored:
        if d not in sel_set:
            common_dropped.append((d, "not_selected"))
        elif d in excl:
            common_dropped.append((d, f"user:{excl[d]}"))
        else:
            to_eval.append(d)

    is_map = isinstance(evaluate, Mapping)
    if not is_map and not callable(evaluate):
        raise TypeError("evaluate must be a callable(view) or a mapping {draw: result}")
    results = {}
    for d in to_eval:
        r = (_lookup_mapping(evaluate, d) if is_map
             else evaluate(DrawView(ar, key, d)))
        results[d] = r
    n_evaluated = sum(1 for r in results.values() if r is not None)

    base_res = None
    has_baseline = bool(ctx["baseline"])
    if baseline:
        if is_map:
            base_res = next((evaluate[k] for k in _BASELINE_KEYS if k in evaluate), None)
        elif has_baseline:
            base_res = evaluate(BaselineView(ar, key))

    if quantities is None:
        qs = []
        for r in results.values():
            for q in (r or {}):
                if q not in qs:
                    qs.append(q)
    else:
        qs = list(quantities)

    label_default = meta.get("regular_label") or "not_regular"
    draw_rms = _per_draw_boundary(ar, key, to_eval)
    boundary_flag_seen = _any_flag(ar, key, stored, "passes_boundary_filter")

    base_prov = {
        "selection": selection,
        "coil_filter": ctx["coil_filter"] if ctx["coil_filter"] is not None else "none",
        "coil_sigma_model": ctx["coil_sigma_model"],
        "boundary_filter_applied": boundary_flag_seen,
        "rms_max_mm": ctx["rms_max_mm"], "max_max_mm": ctx["max_max_mm"],
        "draw_boundary_rms_mm": draw_rms,
        "generation_mode": ctx["generation_mode"],
        "pole_rule": pole_rule, "regular_label": meta.get("regular_label"),
        "min_n": int(min_n), "hard_min": int(hard_min),
        "percentiles": list(percentiles), "method": method,
        "exclude": {int(k): v for k, v in excl.items()},
        "bouquet_version": ctx["bouquet_version"],
        "evaluator_meta": meta,
    }

    out = {}
    for q in qs:
        dropped = list(common_dropped)
        vals = {}
        n_ok = n_reg = 0
        for d in to_eval:
            r = results[d]
            if r is None:
                dropped.append((d, "status:not_evaluated"))
                continue
            if q not in r:
                dropped.append((d, "status:missing"))
                continue
            reason, value, regular = _classify(r[q], label_default)
            if reason is not None and reason.startswith("status:"):
                dropped.append((d, reason))
                continue
            n_ok += 1
            if not regular:
                dropped.append((d, reason))
                continue
            n_reg += 1
            if reason is not None:          # non_finite
                dropped.append((d, reason))
                continue
            vals[d] = value
        dropped.sort(key=lambda t: t[0])
        x = np.asarray([vals[d] for d in sorted(vals)], dtype=float)
        st = _stat_block(x, min_n=min_n, hard_min=hard_min,
                         percentiles=percentiles, method=method)
        frac = (n_reg / n_ok) if n_ok else _NAN
        gated = bool(n_ok and frac <= 0.5)

        rec = BandRecord(
            scan_key=skey, quantity=q, status="ok",
            n_requested=ctx["n_requested"],
            n_requested_source=ctx["n_requested_source"],
            n_attempted=ctx["n_attempted"], n_stored=len(stored),
            n_selected=len(sel), n_evaluated=n_evaluated, n_ok=n_ok,
            n_regular=n_reg, regular_fraction=frac,
            percentiles=percentiles, percentile_method=method,
            gated=gated, values={int(d): float(vals[d]) for d in sorted(vals)},
            dropped=[(int(d), r) for d, r in dropped],
            closure_limited=ctx["closure_limited"],
            closure_limited_reasons=ctx["closure_limited_reasons"],
            closure_channel=ctx["closure_channel"], archive=ar.path)
        for k, v in st.items():
            setattr(rec, k, v)
        rec.show = bool(rec.n_used > 0 and not gated)

        # ---- baseline overlay (never the centre)
        if baseline:
            rec.baseline_status, bv = _baseline_entry(base_res, q, label_default)
            if base_res is None and not has_baseline:
                rec.baseline_status = "no_baseline"
            if bv is not None:
                rec.baseline_value = bv
                if x.size:
                    rec.baseline_quantile = float(np.mean(x < bv))
                    rec.baseline_outside_range = bool(bv < x.min() or bv > x.max())

        prov = dict(base_prov)
        prov["limitations"] = _limitations(rec, ctx, meta, selection,
                                           require_filter, percentiles)
        rec.provenance = prov
        out[q] = rec
    return out


def _baseline_entry(base_res, q, label_default):
    """``(baseline_status, value_or_None)``; the value is kept whenever it is
    ok and finite (an irregular baseline is reported as such, value kept)."""
    if base_res is None:
        return "not_evaluated", None
    if q not in base_res:
        return "status:missing", None
    reason, value, regular = _classify(base_res[q], label_default)
    if reason is not None and reason.startswith("status:"):
        return reason, None
    if value is None or not math.isfinite(value):
        return (reason or "non_finite"), None
    return (reason or "ok"), float(value)


def _any_flag(ar, key, draws, flag):
    import h5py
    from .utils import _group_path
    with h5py.File(ar.path, "r") as hf:
        return any(flag in hf[_group_path(key, d)].attrs for d in draws)


def _limitations(rec, ctx, meta, selection, require_filter, percentiles):
    lim = [LIMITATION_SAMPLED]
    for extra in meta.get("limitations", ()) or ():
        lim.append(str(extra))
    if ctx["n_attempted"] is None:
        lim.append("n_attempted is not recorded on this archive: draws that "
                   "failed before archiving are invisible, so n_stored is not "
                   "the number of attempts.")
    if ctx["coil_filter"] is None and selection != "all":
        lim.append(f"No coil_filter stamp on the scan (require_filter="
                   f"{require_filter}): selection={selection!r} may be every "
                   "stored draw.")
    if ctx["rms_max_mm"] == UNRECORDED and ctx["max_max_mm"] == UNRECORDED:
        lim.append("The boundary acceptance threshold is not recorded on the "
                   "archive.")
    if ctx["closure_limited"]:
        lim.append("The baseline Ip closure is flagged closure_limited: do not "
                   "pool this slice with non-limited slices.")
    if rec.below_floor and rec.n_used >= 2:
        lo, hi = percentiles
        cov = (hi - lo) / 100.0 * (rec.n_used - 1) / (rec.n_used + 1)
        lim.append(f"n_used={rec.n_used} < min_n: with linear interpolation the "
                   f"p{lo:g}-p{hi:g} band covers about {cov:.2f} of the "
                   f"distribution, not {(hi - lo) / 100.0:.2f}.")
    return lim


# --------------------------------------------------------------------------
#  many keys
# --------------------------------------------------------------------------
def draw_bands(sources, evaluate, **kw) -> BandTable:
    """:func:`draw_band` over the REQUESTED ``(archive_or_path, scan_key)`` list.

    A requested key never becomes a silent gap: an absent archive or key yields
    a record with ``status="no_archive"``, a scan group carrying
    ``refused_reason`` one with ``status="refused"`` -- one per quantity seen
    elsewhere in the table (``quantity=None`` if none was).

    ``evaluate`` is a callable, or a mapping ``{scan_key: {draw: result}}``
    (key as ``str(scan_key)`` or an ``(archive_path, scan_key)`` tuple).
    ``exclude``, if given, is ``{scan_key: {draw: reason}}``. Other keywords go
    to :func:`draw_band` unchanged.
    """
    excl_all = kw.pop("exclude", None) or {}
    per_key = []                      # (skey, archive_ref, records | status)
    for ref, sk in sources:
        skey = None if sk is None else str(sk)
        ar_ref = getattr(ref, "path", ref)
        try:
            ar = _as_archive(ref)
            _resolve_key(ar, sk)
        except (FileNotFoundError, KeyError, OSError):
            per_key.append((skey, str(ar_ref), "no_archive"))
            continue
        ev = evaluate
        if isinstance(evaluate, Mapping):
            ev = evaluate.get((ar.path, skey), evaluate.get(skey, {}))
        recs = draw_band(ar, sk, ev, exclude=excl_all.get(skey), **kw)
        per_key.append((skey, ar.path, recs))

    seen_q = []
    for _, _, recs in per_key:
        if isinstance(recs, dict):
            for q in recs:
                if q is not None and q not in seen_q:
                    seen_q.append(q)
    qs = kw.get("quantities") or seen_q or [None]

    records = []
    for skey, ref, recs in per_key:
        if recs == "no_archive":
            records.extend(BandRecord(scan_key=skey, quantity=q,
                                      status="no_archive", archive=ref)
                           for q in qs)
            continue
        if any(r.status == "refused" for r in recs.values()):
            reason = next(iter(recs.values())).refused_reason
            records.extend(BandRecord(scan_key=skey, quantity=q, status="refused",
                                      refused_reason=reason, archive=ref)
                           for q in qs)
            continue
        records.extend(recs.values())
    return BandTable(records)


# --------------------------------------------------------------------------
#  bouquet's own scalars
# --------------------------------------------------------------------------
def _li_attr_for_scale(scale):
    """Per-draw attr on the archive's l_i estimator (``plotting`` convention)."""
    return "l_i(3)" if str(scale).startswith("iter") else "l_i(1)"


def _rational_rho(eq, m, n):
    r"""rho_tor at the OUTERMOST q = m/n crossing, or None when absent.

    rho_tor = sqrt(int_0^psi q dpsi_N / int_0^1 q dpsi_N) is the reader's own
    :attr:`GEQDSKEquilibrium.rhovn`; the crossing is linear in psi_N.
    """
    q = np.abs(np.asarray(eq.qpsi, dtype=float))
    psi = np.asarray(eq.psi_N, dtype=float)
    rho = np.asarray(eq.rhovn, dtype=float)
    dq = q - float(m) / float(n)
    idx = np.nonzero((dq[:-1] * dq[1:] <= 0) & ~((dq[:-1] == 0) & (dq[1:] == 0)))[0]
    if idx.size == 0:
        return None
    i = int(idx[-1])
    t = 0.0 if dq[i] == dq[i + 1] else dq[i] / (dq[i] - dq[i + 1])
    psi_s = psi[i] + t * (psi[i + 1] - psi[i])
    return float(np.interp(psi_s, psi, rho))


def draw_scalars(archive, scan_key=None, *, rational=((2, 1), (3, 1)),
                 q_source="auto", **kw) -> dict:
    """Bands for bouquet's own per-draw scalars -- no external code needed.

    Quantities (all through :func:`draw_band`, so the same population, floor,
    counts and provenance apply; ``**kw`` is passed on):

    * ``q0``, ``q95`` -- from the draw's ``eq_fsa/q`` (q at the innermost
      stored surface, and q interpolated at psi_N = 0.95), falling back to the
      g-file ``qpsi`` when no ``eq_fsa`` block was captured
      (``q_source="auto"``); ``"geqdsk"`` / ``"eq_fsa"`` force one source.
      The baseline has no ``eq_fsa``, so under ``"auto"`` its overlay comes
      from the g-file; the sources used are recorded in provenance and a
      mismatch is listed under limitations.
    * ``rho(q=m/n)`` for each ``(m, n)`` in ``rational`` -- rho_tor of the
      outermost q = m/n surface from the g-file (``rhovn``); a draw with no
      such surface is ``regular=False`` with label ``no_q=m/n_surface``.
    * ``l_i`` -- the archived per-draw attribute on the estimator named by the
      baseline's ``l_i_scale`` (``l_i(3)`` for ``"iter(li3)"``, ``l_i(1)``
      for ``"std(li1)"``, the documented default when the attr is absent).
      The baseline overlay is the recorded TokaMaker value on that estimator
      from ``li_metrics`` when present, else unrecorded. A g-file recompute of
      l_i is NOT used: it is a different estimator path.
    * ``beta_N``, ``<P> [kPa]`` -- from the g-file, exactly as
      :meth:`ScanView.spread` computes them.
    """
    if q_source not in ("auto", "geqdsk", "eq_fsa"):
        raise ValueError(f"q_source must be 'auto', 'geqdsk' or 'eq_fsa', got {q_source!r}")
    from .archive import _gfile_pressure_scalars
    from .utils import load_eq_fsa
    ar = _as_archive(archive)
    key = _resolve_key(ar, scan_key)

    # the l_i estimator is named by the archive, never assumed per draw
    try:
        from .utils import load_baseline_profiles
        bl = load_baseline_profiles(ar.path, scan_key=key)
    except KeyError:
        bl = {}
    scale_raw = bl.get("l_i_scale")
    scale = str(_decode(scale_raw)) if scale_raw is not None else "std(li1)"
    scale_src = "archive:_baseline.l_i_scale" if scale_raw is not None else \
        "default: std(li1) (no l_i_scale attr; pre-estimator-fix archive)"
    li_attr = _li_attr_for_scale(scale)
    li_meta_key = "tokamaker_li_3" if li_attr == "l_i(3)" else "tokamaker_li_1"
    li_base = (bl.get("li_metrics") or {}).get(li_meta_key)

    q_used = {"draws": set(), "baseline": None}
    labels = {(m, n): f"rho(q={m}/{n})" for m, n in rational}

    def _evaluate(view):
        out = {}
        try:
            eq = view.equilibrium()
        except KeyError:
            eq = None
        # ---- q0 / q95
        fsa = None
        if not view.is_baseline and q_source in ("auto", "eq_fsa"):
            fsa = load_eq_fsa(ar.path, view.count, scan_key=key)
        if fsa is not None and "q" in fsa and "psi_N" in fsa:
            qf, pf = np.abs(fsa["q"]), fsa["psi_N"]
            src = "eq_fsa"
        elif q_source == "eq_fsa" and not view.is_baseline:
            qf = pf = None
            src = None
        elif eq is not None:
            qf = np.abs(np.asarray(eq.qpsi, dtype=float))
            pf = np.asarray(eq.psi_N, dtype=float)
            src = "geqdsk"
        else:
            qf = pf = src = None
        if view.is_baseline:
            q_used["baseline"] = src
        elif src is not None:
            q_used["draws"].add(src)
        if qf is None:
            code = "no_eq_fsa" if q_source == "eq_fsa" else "no_eqdsk"
            out["q0"] = out["q95"] = {"value": _NAN, "status": code}
        else:
            out["q0"] = {"value": float(qf[0]), "status": "ok"}
            out["q95"] = {"value": float(np.interp(0.95, pf, qf)), "status": "ok"}
        # ---- rational surfaces, beta_N, <P>
        for (m, n), name in labels.items():
            if eq is None:
                out[name] = {"value": _NAN, "status": "no_eqdsk"}
                continue
            r = _rational_rho(eq, m, n)
            out[name] = ({"value": r, "status": "ok", "regular": True} if r is not None
                         else {"value": _NAN, "status": "ok", "regular": False,
                               "label": f"no_q={m}/{n}_surface"})
        if eq is None:
            out["beta_N"] = out["<P> [kPa]"] = {"value": _NAN, "status": "no_eqdsk"}
        else:
            p_avg, beta_n = _gfile_pressure_scalars(eq)
            out["beta_N"] = {"value": beta_n, "status": "ok"}
            out["<P> [kPa]"] = {"value": p_avg, "status": "ok"}
        # ---- l_i (archive attribute, estimator named by the archive)
        if view.is_baseline:
            out["l_i"] = ({"value": float(li_base), "status": "ok"} if li_base is not None
                          else {"value": _NAN, "status": UNRECORDED})
        else:
            a = view.attrs
            out["l_i"] = ({"value": float(a[li_attr]), "status": "ok"} if li_attr in a
                          else {"value": _NAN, "status": f"no_{li_attr}_attr"})
        return out

    meta = dict(kw.pop("evaluator_meta", None) or {})
    meta.setdefault("code", "bouquet.draw_scalars")
    meta.update(l_i_estimator=scale, l_i_estimator_source=scale_src,
                l_i_attr=li_attr, q_source=q_source,
                rho_definition="rho_tor = sqrt(int q dpsi_N / int_0^1 q dpsi_N), "
                               "outermost crossing, from the g-file qpsi")
    meta.setdefault("regular_label", "rational surface exists")
    recs = draw_band(ar, key, _evaluate, evaluator_meta=meta, **kw)
    used = {"draws": sorted(q_used["draws"]), "baseline": q_used["baseline"]}
    for q in ("q0", "q95"):
        r = recs.get(q)
        if r is None or r.status != "ok":
            continue
        r.provenance = dict(r.provenance, q_source_used=used)
        if used["baseline"] is not None and any(s != used["baseline"] for s in used["draws"]):
            r.provenance["limitations"] = list(r.provenance["limitations"]) + [
                f"The baseline {q} comes from {used['baseline']} while the draws "
                f"use {', '.join(used['draws'])}: the overlay is not like-for-like "
                "(pass q_source='geqdsk' for a single source)."]
    return recs


# --------------------------------------------------------------------------
#  plotting
# --------------------------------------------------------------------------
def plot_band(table, quantity, ax=None, x=None):
    """Median line + p16-p84 band for ``quantity`` across a :class:`BandTable`.

    Filled marker: n_used >= min_n. Hollow marker: ``below_floor``. No band
    where ``no_band``. No marker where ``show=False`` (gated, or no draws):
    the regular fraction (``n_regular/n_ok``) or the status is written there
    instead. The baseline is overlaid dashed, never used as the centre.
    ``x`` defaults to the scan keys as floats (their order when not numeric).
    Returns the axes.
    """
    import matplotlib.pyplot as plt
    if isinstance(table, Mapping):
        table = BandTable(table.values())
    recs = [r for r in table.records if r.quantity in (quantity, None)]
    if ax is None:
        _, ax = plt.subplots()
    if x is None:
        try:
            x = [float(r.scan_key) for r in recs]
        except (TypeError, ValueError):
            x = list(range(len(recs)))
    x = np.asarray(x, dtype=float)

    def col(name, keep):
        return np.array([getattr(r, name) if (k and getattr(r, name) is not None)
                         else np.nan for r, k in zip(recs, keep)], dtype=float)

    shown = [r.status == "ok" and r.show for r in recs]
    banded = [s and not r.no_band for s, r in zip(shown, recs)]
    med = col("median", shown)
    ax.fill_between(x, col("p16", banded), col("p84", banded), alpha=0.25,
                    color="C0", lw=0, label="p16-p84")
    ax.plot(x, med, "-", color="C0", lw=1.5, label="median")
    full = np.array([s and not r.below_floor for s, r in zip(shown, recs)])
    hollow = np.array([s and r.below_floor for s, r in zip(shown, recs)])
    ax.plot(x[full], med[full], "o", color="C0", ls="none")
    ax.plot(x[hollow], med[hollow], "o", mfc="none", color="C0", ls="none",
            label="n_used < min_n")
    base = np.array([np.nan if r.baseline_value is None else r.baseline_value
                     for r in recs], dtype=float)
    if np.isfinite(base).any():
        ax.plot(x, base, "--", color="k", lw=1.0, label="baseline")
    tr = ax.get_xaxis_transform()
    for xi, r, s in zip(x, recs, shown):
        if s:
            continue
        txt = (r.status if r.status != "ok"
               else f"{r.n_regular}/{r.n_ok}" if r.n_ok else "n=0")
        ax.text(xi, 0.03, txt, transform=tr, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(str(quantity))
    return ax
