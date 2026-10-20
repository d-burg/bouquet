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
finite. A value must be one number (a one-element array counts as one); an
array-valued quantity raises :class:`NonScalarQuantityError` rather than being
dropped as non-finite. Every removal is kept in ``dropped`` with its reason
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
statistic with reason ``irregular:<label>`` and are counted (``n_irregular``),
never deleted (their finite values are kept in ``irregular_values``); whenever
any are removed the band is conditional on the regular outcome, which a
limitation line and the :func:`plot_band` annotation state. ``regular`` must be
a real boolean (``"False"`` or NaN is refused, not coerced). If
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
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

__all__ = [
    "BandRecord", "BandTable", "UnfilteredArchiveError", "NonScalarQuantityError",
    "draw_band", "draw_bands", "draw_scalars", "plot_band",
    "BandSummary", "bands_summary", "plot_bands_summary",
    "SUMMARY_SCALARS", "SUMMARY_GRADIENTS",
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
    default ``(16, 84)``). ``status`` is ``"ok"``, ``"unreadable"`` (the
    archive file exists but could not be opened -- locked, being written or
    corrupt; the error is in the limitations), ``"empty"`` (the key was
    read but no draw survived to the statistic: ``n_used == 0``, no band, and
    ``empty_reason`` says at which stage the population ran out), ``"refused"``
    (the scan group carries a ``refused_reason``) or ``"no_archive"`` (the
    archive or the key is absent). A requested key or quantity is never
    dropped from a result: it comes back as one of these records.
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
    #: ok draws the pole rule removed (``n_ok - n_regular``); the band is
    #: CONDITIONAL on the regular outcome whenever this is > 0
    n_irregular: int = 0
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
    #: values further than 10 MAD from the median (diagnostic; no cut);
    #: None when the MAD is 0 and the count has no scale
    n_extreme: Optional[int] = 0
    # ---- per draw
    values: dict = field(default_factory=dict)
    #: finite values of the irregular draws, kept out of the statistic but
    #: not thrown away (``{draw: value}``)
    irregular_values: dict = field(default_factory=dict)
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
    #: why ``n_used == 0`` (status ``"empty"``): ``no_draws_stored``,
    #: ``all_draws_rejected`` (the stamped filters selected none),
    #: ``all_draws_excluded`` (``exclude`` removed the rest),
    #: ``none_evaluated``, ``no_ok_status``, ``no_regular_draws`` or
    #: ``no_finite_values``; None otherwise.
    empty_reason: Optional[str] = None
    refused_reason: Optional[str] = None
    archive: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain JSON-able dict (tuples -> lists, numpy scalars -> Python)."""
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    def __repr__(self):
        head = f"<BandRecord {self.scan_key!r}/{self.quantity!r}"
        if self.status != "ok":
            why = self.refused_reason or self.empty_reason
            why = f" ({why})" if why else ""
            return f"{head} {self.status}{why}>"
        flags = [n for n in ("below_floor", "no_band", "gated") if getattr(self, n)]
        if not self.show:
            flags.append("hidden")
        _xf = (self.provenance or {}).get("experimental_features")
        if isinstance(_xf, list) and _xf:
            flags.append("EXPERIMENTAL[" + ",".join(_xf) + "]")
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
_JSON_COLS = ("percentiles", "values", "irregular_values", "dropped",
              "closure_limited_reasons", "provenance")


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
             "n_evaluated", "n_ok", "n_regular", "n_used", "n_extreme",
             "n_irregular"}
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
        held = list_equilibrium_indices(ar.path, scan_key=key)
        if held:
            # the writers supersede a refusal when draws land, so this is an
            # inconsistent archive: refuse to pick one reading silently
            raise ValueError(
                f"scan {key!r} carries refused_reason={ctx['refused_reason']!r} "
                f"but holds {len(held)} draws; the archive is inconsistent "
                "(remove the attr or the draws deliberately)")
        return ctx
    ctx["indices"] = list_equilibrium_indices(ar.path, scan_key=key)

    # ---- filters actually stamped
    cf = sattrs.get("coil_filter")
    ctx["coil_filter"] = str(cf) if cf is not None else None
    ctx["coil_sigma_model"] = _sigma_model_summary(
        _json_attr(sattrs.get("coil_sigma_model")))
    ctx["rms_max_mm"] = sattrs.get("boundary_rms_max_mm", UNRECORDED)
    ctx["max_max_mm"] = sattrs.get("boundary_max_max_mm", UNRECORDED)
    # where that cut came from: "explicit" | "device:<name>" | "generic" |
    # "disabled" (the cut was switched off: no threshold, none applied)
    _src = sattrs.get("boundary_cut_source")
    ctx["boundary_cut_source"] = str(_src) if _src is not None else UNRECORDED
    if ctx["boundary_cut_source"] == "disabled":
        ctx["rms_max_mm"] = ctx["max_max_mm"] = None
    # the in-spec rule the band flags were judged under (owner decision
    # 2026-10-14: the magnetics, plus opt-in bands); UNRECORDED on an archive
    # that predates the record (its engine draws then carry the old
    # unconditional l_i band in their selection)
    _rule = sattrs.get("inspec_rule")
    ctx["inspec_rule"] = str(_rule) if _rule is not None else UNRECORDED
    ctx["inspec_bands"] = _json_attr(sattrs.get("inspec_bands_json"))
    # the cut an until-N loop counted its target against (generation stamp)
    _isrc = sattrs.get("inspec_cut_source")
    ctx["inloop_cut_source"] = str(_isrc) if _isrc is not None else UNRECORDED
    _irms = sattrs.get("inspec_rms_max_mm")
    ctx["inloop_rms_max_mm"] = (float(_irms) if _irms is not None
                                else (None if _isrc is not None else UNRECORDED))

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
    ctx["merge_partial"] = _json_attr(sattrs.get("merge_partial_json"))
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
    # the EXPERIMENTAL features the run enabled (bouquet.experimental;
    # _baseline attr experimental_features_json): a list ([] none), or
    # UNRECORDED on an archive that predates the record
    from .utils import load_experimental_features
    try:
        xf = load_experimental_features(ar.path, scan_key=key)
    except (KeyError, OSError, ValueError):
        xf = None
    ctx["experimental_features"] = UNRECORDED if xf is None else list(xf)
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


class NonScalarQuantityError(ValueError):
    """An evaluator returned an array (more than one element) for a quantity
    of the scalar recipe; it would otherwise be dropped as non-finite."""


def _scalar_value(v, where):
    """``float`` of a scalar value; a one-element array or list counts as a
    scalar, a longer (or empty) one raises :class:`NonScalarQuantityError`.
    Anything else that is not a number becomes NaN (-> ``non_finite``)."""
    if isinstance(v, (np.ndarray, list, tuple)):
        a = np.asarray(v)
        if a.size != 1:
            raise NonScalarQuantityError(
                f"{where}: value has shape {a.shape}; draw_band bands ONE "
                "number per draw and quantity. Return a scalar, or split a "
                "profile into one named quantity per point (e.g. "
                "'q@rho=0.5'), so each point gets its own counts and floors.")
        v = a.reshape(-1)[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NAN


def _strict_bool(v, where):
    """``regular`` must be a real boolean: ``bool("False")`` is True and
    ``bool(nan)`` is True, so anything else is refused rather than coerced."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    raise TypeError(f"{where}: 'regular' must be True/False (bool or numpy.bool_), "
                    f"got {v!r} of type {type(v).__name__}")


def _classify(entry, label_default, where="value"):
    """``(reason_or_None, value, regular)`` for one draw's quantity entry."""
    if not isinstance(entry, Mapping):
        entry = {"value": entry}
    status = str(entry.get("status", "ok"))
    if status != "ok":
        return f"status:{status}", None, None
    regular = _strict_bool(entry.get("regular", True), where)
    value = _scalar_value(entry.get("value"), where)
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
    # MAD = 0 (more than half the values identical) gives no scale: every
    # value off the median would count as "extreme", so the diagnostic is
    # reported as undefined (None) instead
    out["n_extreme"] = (int(np.sum(np.abs(x - med) > 10.0 * mad)) if mad > 0
                        else None)
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
              baseline=True, evaluator_meta=None, rescued="include") -> dict:
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
    minus ``exclude={draw: reason}``, minus -- with ``rescued="exclude"`` --
    the draws whose solve was rescued by the opt-in draw-solve rescue
    (stamped ``solve_recovered``; dropped as ``rescued:<recovered_by>``;
    the default ``"include"`` keeps them, lists them in the provenance's
    ``rescued_draws`` and warns with their count); then, per quantity,
    status ``"ok"``, then
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

    Returns ``{quantity: BandRecord}``, never empty. A quantity no draw
    survives for comes back with ``status="empty"``, ``n_used=0`` and an
    ``empty_reason``; when no quantity is known at all (``quantities`` not
    given and no draw evaluated) the single record is keyed ``None``. A scan
    carrying ``refused_reason`` returns records with ``status="refused"``
    (keyed ``None`` when ``quantities`` is not given). A missing key raises
    ``KeyError``.
    """
    _validate(selection, pole_rule, percentiles, min_n, hard_min)
    if rescued not in RESCUED_POLICIES:
        raise ValueError(f"rescued must be one of {RESCUED_POLICIES}, got "
                         f"{rescued!r}")
    percentiles = tuple(percentiles)
    meta = dict(evaluator_meta or {})
    ar = _as_archive(archive)
    key = _resolve_key(ar, scan_key)
    ctx = _scan_context(ar, key)
    skey = None if key is None else str(key)

    if ctx["refused_reason"] is not None:
        qs = list(quantities) if quantities else [None]
        return {q: _status_record(skey, q, "refused", ar.path,
                                  refused_reason=ctx["refused_reason"],
                                  selection=selection, pole_rule=pole_rule,
                                  min_n=min_n, hard_min=hard_min,
                                  percentiles=percentiles, method=method,
                                  evaluator_meta=meta)
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

    resc = rescued_draws(ar.path, key, stored)
    common_dropped = []
    to_eval = []
    for d in stored:
        if d not in sel_set:
            common_dropped.append((d, "not_selected"))
        elif d in excl:
            common_dropped.append((d, f"user:{excl[d]}"))
        elif rescued == "exclude" and d in resc:
            common_dropped.append((d, f"rescued:{resc[d]}"))
        else:
            to_eval.append(d)
    _xf = ctx.get("experimental_features")
    if isinstance(_xf, list) and _xf:
        import warnings
        from .experimental import ExperimentalFeatureWarning
        warnings.warn(
            f"draw_band: scan {skey!r} was generated with EXPERIMENTAL "
            f"feature(s) {_xf} (not validated on real data; see "
            "bouquet.experimental.REGISTRY); every record carries them in "
            "its provenance and limitations",
            ExperimentalFeatureWarning, stacklevel=2)
    _resc_in = sorted(d for d in to_eval if d in resc)
    if _resc_in:
        import warnings
        warnings.warn(
            f"draw_band: scan {skey!r} includes {len(_resc_in)} draw(s) whose "
            f"solve was RESCUED (draws {_resc_in}: a different urf, or "
            "accepted at a looser nl_tol than the solver's; see each draw's "
            "solve_* attrs); pass rescued='exclude' to band without them",
            UserWarning, stacklevel=2)

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
    if not qs:
        # nothing reached the evaluator (every draw rejected / excluded, none
        # stored) or it returned nothing: the key still gets a record, keyed
        # None, which draw_bands expands to the quantities seen elsewhere
        qs = [None]

    label_default = meta.get("regular_label") or "not_regular"
    draw_rms = _per_draw_boundary(ar, key, to_eval)
    boundary_flag_seen = _any_flag(ar, key, stored, "passes_boundary_filter")

    base_prov = {
        "selection": selection,
        "coil_filter": ctx["coil_filter"] if ctx["coil_filter"] is not None else "none",
        "coil_sigma_model": ctx["coil_sigma_model"],
        "boundary_filter_applied": boundary_flag_seen,
        "rms_max_mm": ctx["rms_max_mm"], "max_max_mm": ctx["max_max_mm"],
        "boundary_cut_source": ctx["boundary_cut_source"],
        "inloop_rms_max_mm": ctx["inloop_rms_max_mm"],
        "inloop_cut_source": ctx["inloop_cut_source"],
        "draw_boundary_rms_mm": draw_rms,
        "generation_mode": ctx["generation_mode"],
        "inspec_rule": ctx["inspec_rule"],
        "inspec_bands": ctx["inspec_bands"],
        "pole_rule": pole_rule, "regular_label": meta.get("regular_label"),
        "min_n": int(min_n), "hard_min": int(hard_min),
        "percentiles": list(percentiles), "method": method,
        "exclude": {int(k): v for k, v in excl.items()},
        "bouquet_version": ctx["bouquet_version"],
        "evaluator_meta": meta,
        "experimental_features": ctx["experimental_features"],
    }
    if resc or rescued != "include":
        # only where it says something: an archive without rescued draws
        # (the rescue is opt-in) keeps the record it had
        base_prov["rescued"] = rescued
        base_prov["rescued_draws"] = {int(d): r for d, r in resc.items()}

    out = {}
    for q in qs:
        dropped = list(common_dropped)
        vals = {}
        irr_vals = {}
        n_ok = n_reg = 0
        for d in to_eval:
            r = results[d]
            if r is None:
                dropped.append((d, "status:not_evaluated"))
                continue
            if q not in r:
                dropped.append((d, "status:missing"))
                continue
            reason, value, regular = _classify(
                r[q], label_default, where=f"scan {skey!r} draw {d} quantity {q!r}")
            if reason is not None and reason.startswith("status:"):
                dropped.append((d, reason))
                continue
            n_ok += 1
            if not regular:
                dropped.append((d, reason))
                if math.isfinite(value):
                    irr_vals[int(d)] = float(value)
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
            n_regular=n_reg, n_irregular=n_ok - n_reg, regular_fraction=frac,
            irregular_values=irr_vals,
            percentiles=percentiles, percentile_method=method,
            gated=gated, values={int(d): float(vals[d]) for d in sorted(vals)},
            dropped=[(int(d), r) for d, r in dropped],
            closure_limited=ctx["closure_limited"],
            closure_limited_reasons=ctx["closure_limited_reasons"],
            closure_channel=ctx["closure_channel"], archive=ar.path)
        for k, v in st.items():
            setattr(rec, k, v)
        rec.show = bool(rec.n_used > 0 and not gated)
        if rec.n_used == 0:
            rec.status = "empty"
            rec.empty_reason = _empty_reason(len(stored), len(sel), len(to_eval),
                                             n_evaluated, n_ok, n_reg)

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


def _empty_reason(n_stored, n_selected, n_to_eval, n_evaluated, n_ok, n_regular):
    """The first stage at which the population ran out (see BandRecord)."""
    if n_stored == 0:
        return "no_draws_stored"
    if n_selected == 0:
        return "all_draws_rejected"
    if n_to_eval == 0:
        return "all_draws_excluded"
    if n_evaluated == 0:
        return "none_evaluated"
    if n_ok == 0:
        return "no_ok_status"
    if n_regular == 0:
        return "no_regular_draws"
    return "no_finite_values"


def _baseline_entry(base_res, q, label_default):
    """``(baseline_status, value_or_None)``; the value is kept whenever it is
    ok and finite (an irregular baseline is reported as such, value kept)."""
    if base_res is None:
        return "not_evaluated", None
    if q not in base_res:
        return "status:missing", None
    reason, value, regular = _classify(base_res[q], label_default,
                                       where=f"baseline quantity {q!r}")
    if reason is not None and reason.startswith("status:"):
        return reason, None
    if value is None or not math.isfinite(value):
        return (reason or "non_finite"), None
    return (reason or "ok"), float(value)


#: What a band does with draws whose solve was RESCUED by the opt-in draw
#: solve rescue (``GenerationConfig.draw_solve_retry_urf`` /
#: ``draw_solve_loose_tol``; stamped ``solve_recovered`` on the draw):
#: ``"include"`` (default: kept, counted and warned) or ``"exclude"``
#: (dropped as ``rescued:<recovered_by>``).  Owner decision D5, 2026-10-09.
RESCUED_POLICIES = ("include", "exclude")


def rescued_draws(path, key, draws) -> dict:
    """``{draw: recovered_by}`` for the *draws* of scan *key* stamped
    ``solve_recovered`` (a rescued solve; see
    :meth:`bouquet.TokaMaker_interface.DrawSolveGuard.draw_stamp`)."""
    import h5py
    from .utils import _group_path
    out = {}
    with h5py.File(path, "r") as hf:
        for d in draws:
            gp = _group_path(key, d)
            if gp not in hf:
                continue
            a = hf[gp].attrs
            if bool(a.get("solve_recovered", False)):
                out[int(d)] = str(_decode(a.get("solve_recovered_by", "")))
    return out


def _any_flag(ar, key, draws, flag):
    import h5py
    from .utils import _group_path
    with h5py.File(ar.path, "r") as hf:
        return any(flag in hf[_group_path(key, d)].attrs for d in draws)


_RECIPE_DEFAULTS = dict(selection="selected", pole_rule="majority_regular",
                        min_n=15, hard_min=5, percentiles=(16, 84),
                        method="linear", evaluator_meta=None)


def _status_record(skey, quantity, status, archive, *, refused_reason=None,
                   detail=None, **kw) -> "BandRecord":
    """A ``refused`` / ``no_archive`` record that still honours the
    provenance contract: the recipe settings it was asked for, the reader
    version, and a limitation saying why nothing was evaluated -- so a
    consumer can tell an intentionally empty record from an unannotated one.
    """
    from . import __version__
    settings = {k: kw.get(k, v) for k, v in _RECIPE_DEFAULTS.items()}
    why = (f"status={status}"
           + (f": {refused_reason}" if refused_reason else "")
           + (f": {detail}" if detail else "")
           + " -- no draws were evaluated for this key")
    prov = {
        "selection": settings["selection"],
        "coil_filter": UNRECORDED, "coil_sigma_model": UNRECORDED,
        "boundary_filter_applied": None,
        "rms_max_mm": UNRECORDED, "max_max_mm": UNRECORDED,
        "boundary_cut_source": UNRECORDED, "draw_boundary_rms_mm": {},
        "inloop_rms_max_mm": UNRECORDED, "inloop_cut_source": UNRECORDED,
        "generation_mode": UNRECORDED,
        "inspec_rule": UNRECORDED, "inspec_bands": None,
        "pole_rule": settings["pole_rule"],
        "regular_label": (settings["evaluator_meta"] or {}).get("regular_label"),
        "min_n": int(settings["min_n"]), "hard_min": int(settings["hard_min"]),
        "percentiles": list(settings["percentiles"]), "method": settings["method"],
        "exclude": {},
        "bouquet_version": {"scan": UNRECORDED, "file": UNRECORDED,
                            "reader": __version__},
        "experimental_features": UNRECORDED,
        "evaluator_meta": dict(settings["evaluator_meta"] or {}),
        "limitations": [LIMITATION_SAMPLED, why],
    }
    return BandRecord(scan_key=skey, quantity=quantity, status=status,
                      refused_reason=refused_reason, archive=archive,
                      show=False, provenance=prov)


def _limitations(rec, ctx, meta, selection, require_filter, percentiles):
    lim = [LIMITATION_SAMPLED]
    for extra in meta.get("limitations", ()) or ():
        lim.append(str(extra))
    _xf = ctx.get("experimental_features")
    if isinstance(_xf, list) and _xf:
        lim.append(f"EXPERIMENTAL features were enabled for this run: {_xf} "
                   "-- not validated on real data (bouquet.experimental."
                   "REGISTRY lists each one's open validation items).")
    if ctx["n_attempted"] is None:
        lim.append("n_attempted is not recorded on this archive: draws that "
                   "failed before archiving are invisible, so n_stored is not "
                   "the number of attempts.")
    if ctx.get("merge_partial"):
        lim.append(f"PARTIAL parallel merge: workers "
                   f"{ctx['merge_partial'].get('missing_workers')} are missing, so "
                   "n_requested is the run's request, not what the merged shards "
                   "were asked for.")
    if ctx["coil_filter"] is None and selection != "all":
        lim.append(f"No coil_filter stamp on the scan (require_filter="
                   f"{require_filter}): selection={selection!r} may be every "
                   "stored draw.")
    if ctx.get("inspec_rule", UNRECORDED) == UNRECORDED and selection != "all":
        lim.append("The in-spec rule is not recorded on this archive (it "
                   "predates the 2026-10-14 rule): on unified-engine draws "
                   "the selection includes the old unconditional l_i band "
                   "(l_i_tolerance = 0.05), so an l_i band from it reflects "
                   "that tolerance. Re-filter under the magnetics rule with "
                   "bouquet.refilter_inspec(path) or Bouquet.filter().")
    elif str(ctx.get("inspec_rule", "")).startswith("magnetics+") \
            and selection != "all":
        lim.append(f"The selection includes opt-in band(s) (inspec_rule="
                   f"{ctx['inspec_rule']!r}): a band of a banded quantity "
                   "reflects that band's width as well as the ensemble.")
    if ctx["boundary_cut_source"] == "disabled":
        lim.append("No LCFS boundary cut was applied (boundary_cut_source="
                   "'disabled'): the population is the coil filter's alone.")
    elif ctx["rms_max_mm"] == UNRECORDED and ctx["max_max_mm"] == UNRECORDED:
        lim.append("The boundary acceptance threshold is not recorded on the "
                   "archive.")
    if (ctx["inloop_cut_source"] != UNRECORDED
            and ctx["boundary_cut_source"] != UNRECORDED
            and ctx["inloop_rms_max_mm"] != ctx["rms_max_mm"]):
        lim.append(f"The until-N loop counted its target against a boundary cut "
                   f"of {ctx['inloop_rms_max_mm']} mm ({ctx['inloop_cut_source']}) "
                   f"but the stamped filter cut is {ctx['rms_max_mm']} mm "
                   f"({ctx['boundary_cut_source']}): n_requested does not "
                   "describe this population.")
    if ctx["closure_limited"]:
        lim.append("The baseline Ip closure is flagged closure_limited: do not "
                   "pool this slice with non-limited slices.")
    if rec.n_used > 0 and rec.n_extreme is None:
        lim.append("The MAD of the used values is 0 (more than half are "
                   "identical): n_extreme is undefined and reported as None.")
    if rec.n_irregular > 0:
        lim.append(
            f"The band is CONDITIONAL on the regular outcome: {rec.n_irregular} "
            f"of {rec.n_ok} ok draws were irregular (regular_fraction="
            f"{rec.regular_fraction:.3f}) and are excluded from the statistic "
            "(their finite values are kept in irregular_values). Near a pole the "
            "irregular draws are the ones closest to it, so the band may be "
            "biased toward the regular side.")
    if rec.n_used == 0:
        lim.append(f"n_used=0 ({rec.empty_reason}): no draw reached the "
                   "statistic, so there is no median and no band for this key.")
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
    ``refused_reason`` one with ``status="refused"``, and a key where no draw
    reaches the statistic (every draw rejected by the filters, say) one with
    ``status="empty"`` and its ``empty_reason`` -- one per quantity seen
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
        if isinstance(ar_ref, (str, bytes, os.PathLike)) and not os.path.exists(ar_ref):
            per_key.append((skey, str(ar_ref), "no_archive"))
            continue
        try:
            ar = _as_archive(ref)
            _resolve_key(ar, sk)
        except (FileNotFoundError, KeyError):
            per_key.append((skey, str(ar_ref), "no_archive"))
            continue
        except OSError as exc:
            # the file EXISTS but cannot be read (locked, still being
            # written, corrupt): not the same fact as "no archive"
            per_key.append((skey, str(ar_ref), ("unreadable", str(exc)[:300])))
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
        if (isinstance(recs, dict) and list(recs) == [None]
                and recs[None].status == "empty" and qs != [None]):
            # an empty key (e.g. every draw rejected) reports every quantity
            # the table carries, with its counts -- never a silent gap
            records.extend(_requantify(recs[None], q) for q in qs)
            continue
        if recs == "no_archive":
            records.extend(_status_record(skey, q, "no_archive", ref, **kw)
                           for q in qs)
            continue
        if isinstance(recs, tuple) and recs[0] == "unreadable":
            records.extend(_status_record(skey, q, "unreadable", ref,
                                          detail=recs[1], **kw)
                           for q in qs)
            continue
        if any(r.status == "refused" for r in recs.values()):
            reason = next(iter(recs.values())).refused_reason
            records.extend(_status_record(skey, q, "refused", ref,
                                          refused_reason=reason, **kw)
                           for q in qs)
            continue
        records.extend(recs.values())
    return BandTable(records)


def _requantify(rec, quantity):
    """A copy of a quantity-less ``empty`` record for one named quantity."""
    import copy
    out = copy.deepcopy(rec)
    out.quantity = quantity
    return out


# --------------------------------------------------------------------------
#  bouquet's own scalars
# --------------------------------------------------------------------------
def _li_attr_for_scale(scale):
    """Per-draw attr on the archive's l_i estimator (``plotting`` convention)."""
    return "l_i(3)" if str(scale).startswith("iter") else "l_i(1)"


def _shafranov_entries(view) -> dict:
    """``{"beta_p": ..., "Lambda": ...}`` entries of one view (draw or
    baseline): the stored attrs, else -- for a draw -- formed from its
    edge-pressure record's full frame and ``l_i(3)``."""
    from .filtering import (BETA_P_ATTR, LAMBDA_ATTR, _full_beta_p,
                            shafranov_lambda)
    a = view.attrs
    bp, lam = a.get(BETA_P_ATTR), a.get(LAMBDA_ATTR)
    if (bp is None or lam is None) and not view.is_baseline:
        from .edge_pressure import EDGE_PRESSURE_ATTR
        raw = a.get(EDGE_PRESSURE_ATTR)
        if raw is not None:
            try:
                bp = _full_beta_p(json.loads(_decode(raw)))
            except (TypeError, ValueError):
                bp = None
        li3 = a.get("l_i(3)")
        lam = (shafranov_lambda(bp, li3)
               if bp is not None and li3 is not None else None)
    out = {}
    for name, v in (("beta_p", bp), ("Lambda", lam)):
        if v is None:
            out[name] = {"value": _NAN, "status": UNRECORDED
                         if view.is_baseline else f"no_{name}"}
        else:
            out[name] = {"value": float(v), "status": "ok"}
    return out


def _rational_rho(eq, m, n):
    r"""rho_tor at the OUTERMOST q = m/n crossing, or None when absent.

    rho_tor = sqrt(int_0^psi q dpsi_N / int_0^1 q dpsi_N) is the reader's own
    :attr:`GEQDSKEquilibrium.rhovn`; the crossing is linear in psi_N.
    """
    q = np.abs(np.asarray(eq.qpsi, dtype=float))
    psi = np.asarray(eq.psi_N, dtype=float)
    rho = np.asarray(eq.rhovn, dtype=float)
    psi_s = _outermost_crossing(psi, q, float(m) / float(n))
    if psi_s is None:
        return None
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
      from the g-file (q0 at the magnetic axis, not the innermost stored
      surface). The source of every used draw is recorded in provenance
      (``q_source_by_draw``); a statistic that mixes sources, or a baseline
      from a different source than the draws, is listed under limitations,
      and the latter also marks ``baseline_status`` ``"...:source_mismatch"``.
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
    * ``beta_p`` and ``Lambda`` = beta_p + l_i(3)/2
      (:data:`bouquet.filtering.LAMBDA_DEFINITION`: beta_p of the
      full-pressure frame, l_i(3) in TokaMaker's 'iter' normalisation) --
      the draw's stored ``beta_p`` / ``Lambda`` attrs, else formed from its
      edge-pressure record's full frame and its ``l_i(3)`` (an archive made
      before they were stored); the baseline's from its ``_baseline`` attrs
      (unified engine), else unrecorded.  Lambda is the scalar the
      magnetics constrain; it is REPORTED here and never filters (for a
      shaped plasma the coil test enforces the shape-weighted observable
      exactly).

    Every record's provenance carries ``axis_shift``: among the draws the
    statistic USED, how many carry the engine draws' axis record
    (``n_recorded``) and how many of them are displaced more than
    ``engine.AXIS_BRANCH_FRAC`` x a from the reconstruction's axis
    (``n_large``, ``large_draws``; a displacement, not a branch label), and
    ``axis_bimodality``: :func:`bouquet.engine.axis_bimodality` of those
    draws' axis positions, with the limitation "the ensemble is bimodal in
    the axis position: two equilibrium branches; the bands are mixtures"
    when it is.  Reported, never a filter: a band over both branches is the
    physics of the slice.
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

    q_used = {"draws": set(), "baseline": None, "by_draw": {}}
    axis_by_draw = {}
    from .schema import read_axis_entry
    labels = {(m, n): f"rho(q={m}/{n})" for m, n in rational}

    def _evaluate(view):
        out = {}
        if not view.is_baseline:
            import h5py
            with h5py.File(ar.path, "r") as _hf:
                _e = (read_axis_entry(_hf[view._gp]) if view._gp in _hf
                      else None)
            if _e is not None:
                axis_by_draw[int(view.count)] = _e
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
        elif q_source == "eq_fsa":
            # forced source, baseline included: a baseline without eq_fsa is
            # reported as no_eq_fsa rather than quietly overlaid from the
            # g-file on eq_fsa draw statistics
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
            q_used["by_draw"][int(view.count)] = src
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
        # ---- beta_p and Lambda (stored; never a filter)
        out.update(_shafranov_entries(view))
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
    from .filtering import LAMBDA_DEFINITION
    meta.update(l_i_estimator=scale, l_i_estimator_source=scale_src,
                l_i_attr=li_attr, q_source=q_source,
                lambda_definition=LAMBDA_DEFINITION,
                rho_definition="rho_tor = sqrt(int q dpsi_N / int_0^1 q dpsi_N), "
                               "outermost crossing, from the g-file qpsi")
    meta.setdefault("regular_label", "rational surface exists")
    # the fixed quantity list, so a slice with no usable draw still reports
    # every quantity (status="empty") instead of returning nothing
    kw.setdefault("quantities", ["q0", "q95", *labels.values(), "beta_N",
                                 "<P> [kPa]", "l_i", "beta_p", "Lambda"])
    recs = draw_band(ar, key, _evaluate, evaluator_meta=meta, **kw)
    used = {"draws": sorted(q_used["draws"]), "baseline": q_used["baseline"]}
    by_draw = dict(sorted(q_used["by_draw"].items()))
    _what = {"eq_fsa": "the innermost stored eq_fsa surface",
             "geqdsk": "the g-file magnetic axis"}
    for q in ("q0", "q95"):
        r = recs.get(q)
        if r is None or r.status != "ok":
            continue
        # which source each USED draw's value came from, so a mixed statistic
        # is identifiable draw by draw rather than only in aggregate
        src_used = {d: by_draw[d] for d in r.values if d in by_draw}
        r.provenance = dict(r.provenance, q_source_used=used,
                            q_source_by_draw=src_used)
        lim = list(r.provenance["limitations"])
        n_by = {}
        for v in src_used.values():
            n_by[v] = n_by.get(v, 0) + 1
        if len(n_by) > 1:
            lim.append(
                f"The {q} statistic MIXES sources: "
                + ", ".join(f"{n} draws from {k}" for k, n in sorted(n_by.items()))
                + (" (q0 from " + " vs ".join(_what[k] for k in sorted(n_by))
                   + ")" if q == "q0" else "")
                + "; see provenance q_source_by_draw, or pass q_source='geqdsk' "
                "for a single source.")
        if used["baseline"] is not None and any(s != used["baseline"] for s in n_by):
            # the overlay is not like-for-like: say so in the machine-readable
            # status too, so baseline_quantile / baseline_outside_range are
            # not read as a draw-path verdict
            r.baseline_status = f"{r.baseline_status}:source_mismatch"
            lim.append(
                f"The baseline {q} comes from {used['baseline']} while the draws "
                f"use {', '.join(sorted(n_by))}"
                + (f" ({_what[used['baseline']]} vs "
                   + ", ".join(_what[k] for k in sorted(n_by)) + ")"
                   if q == "q0" else "")
                + ": the overlay is not like-for-like, so baseline_quantile and "
                "baseline_outside_range compare different quantities (pass "
                "q_source='geqdsk' for a single source).")
        r.provenance["limitations"] = lim
    # the axis-displacement count and the bimodality of the axis positions
    # over each statistic's USED draws (reported, never a filter)
    from .engine import AXIS_BRANCH_FRAC, axis_bimodality
    for r in recs.values():
        used_d = [d for d in (getattr(r, "values", None) or {})
                  if d in axis_by_draw]
        large = sorted(int(d) for d in used_d
                       if axis_by_draw[d]["shift_large"])
        prov = dict(getattr(r, "provenance", None) or {})
        prov["axis_shift"] = dict(
            n_recorded=len(used_d), n_large=len(large),
            n_small=len(used_d) - len(large), large_draws=large,
            threshold_frac_a=float(AXIS_BRANCH_FRAC),
            rule=("large: the draw's archived magnetic axis more than "
                  "threshold_frac_a x a from the reconstruction's -- a "
                  "displacement, not a branch label; reported, never a "
                  "filter"))
        _a = next((axis_by_draw[d]["minor_radius_m"] for d in used_d
                   if axis_by_draw[d]["minor_radius_m"] is not None), None)
        bim = (axis_bimodality([axis_by_draw[d]["dZ_mm"] for d in used_d],
                               _a) if used_d else None)
        prov["axis_bimodality"] = bim
        if bim is not None and bim.get("axis_bimodal"):
            lo, up = bim["centres_dZ_mm"]
            nl, nu = bim["counts"]
            prov["limitations"] = list(prov.get("limitations") or []) + [
                "the ensemble is bimodal in the axis position: two "
                "equilibrium branches; the bands are mixtures "
                f"({nl} used draw(s) at axis dZ {lo:+.1f} mm, {nu} at "
                f"{up:+.1f} mm; {bim['version']}). Nothing is filtered on "
                "it."]
        r.provenance = prov
    return recs


# --------------------------------------------------------------------------
#  the standard error-bar summary (one call per archive slice)
# --------------------------------------------------------------------------
#: scalars :func:`bands_summary` reports: ``name -> (unit, definition)``
SUMMARY_SCALARS = {
    "beta_N": ("", "beta_N of the draw's g-file (as draw_scalars / ScanView.spread)"),
    "beta_p": ("", "the draw's stored beta_p (full-pressure frame)"),
    "Lambda": ("", "beta_p + l_i(3)/2 (bouquet.filtering.LAMBDA_DEFINITION)"),
    "l_i_3": ("", "the draw's archived l_i(3) (TokaMaker 'iter' normalisation)"),
    "l_i_1": ("", "the draw's archived l_i(1) (TokaMaker 'std' normalisation)"),
    "q0": ("", "q at the innermost stored surface of the draw's eq_fsa (else the g-file axis)"),
    "q95": ("", "q at psi_N = 0.95 of the draw's eq_fsa (else its g-file)"),
}
#: gradients :func:`bands_summary` reports: ``name -> (dataset, unit)``;
#: each is d(dataset)/d(psi_N) of the draw's own archived profile
SUMMARY_GRADIENTS = {
    "dp_dpsi": ("pressure", "Pa/psi_N"),
    "djphi_dpsi": ("jphi_total", "A m^-2/psi_N"),
}
GRADIENT_DEFINITION = (
    "d(profile)/d(psi_N) of the draw's own archived profile (pressure: the "
    "'pressure' dataset [Pa]; jphi_total: the 'jphi_total' dataset [A m^-2]), by "
    "second-order central differences (numpy.gradient) on the archived grid "
    "(mapped to psi_N through the draw's own q when the profile coordinate "
    "is Phi_N), linearly interpolated to the evaluation surface")


def _outermost_crossing(x, y, target):
    """``x`` at the OUTERMOST crossing ``y = target`` (linear), or None."""
    x = np.asarray(x, dtype=float)
    dy = np.asarray(y, dtype=float) - float(target)
    idx = np.nonzero((dy[:-1] * dy[1:] <= 0) & ~((dy[:-1] == 0) & (dy[1:] == 0)))[0]
    if idx.size == 0:
        return None
    i = int(idx[-1])
    t = 0.0 if dy[i] == dy[i + 1] else dy[i] / (dy[i] - dy[i + 1])
    return float(x[i] + t * (x[i + 1] - x[i]))


def _psi_from_phi(phi_grid, psi_q, q):
    """psi_N at the Phi_N values ``phi_grid``, from the draw's own q(psi_N)
    (Phi_N = int_0^psi q dpsi_N / int_0^1 q dpsi_N; q held flat to the axis
    and to psi_N = 1 where the q grid stops short of them)."""
    from scipy.integrate import cumulative_trapezoid
    psi_q = np.asarray(psi_q, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    xe = np.unique(np.concatenate([[0.0], psi_q[(psi_q > 0) & (psi_q < 1)], [1.0]]))
    phi = cumulative_trapezoid(np.interp(xe, psi_q, q), xe, initial=0.0)
    return np.interp(np.asarray(phi_grid, dtype=float), phi / phi[-1], xe)


@dataclass(repr=False)
class BandSummary:
    """What :func:`bands_summary` returns: one :class:`BandRecord` per
    quantity plus where the gradients were evaluated.

    ``records[q]`` holds the median (``median``), the band (``p16`` /
    ``p84``), the full range (``min`` / ``max``) and the count (``n_used``)
    over the in-spec draws, with the whole provenance of :func:`draw_band`.
    ``surface`` says how the evaluation surface was found: ``mode`` is
    ``"own_surface"`` (each draw at its own ``at`` surface),
    ``"fallback_psi_N"`` (no in-spec draw has one: every draw at the named
    psi_N) or ``"fixed_psi_N"`` (``at={"psi_N": x}``).
    """

    records: dict
    surface: dict
    scan_key: Optional[str]
    archive: Optional[str]
    n_selected: int
    units: dict = field(default_factory=dict)

    def __getitem__(self, quantity):
        return self.records[quantity]

    def __iter__(self):
        return iter(self.records)

    @property
    def population(self) -> str:
        """``"in-spec draws"`` for the default selection on a filtered
        archive; otherwise what the population actually is."""
        r = next(iter(self.records.values()), None)
        prov = (r.provenance if r is not None else {}) or {}
        sel = prov.get("selection", "selected")
        if sel == "selected" and prov.get("coil_filter", "none") != "none":
            return "in-spec draws"
        if sel == "selected":
            return "stored draws of an UNFILTERED archive"
        return f"{sel!r} draws"

    @property
    def where(self) -> str:
        """One phrase: where the gradients were evaluated."""
        s = self.surface
        if s.get("mode") == "own_surface":
            return f"each draw's own {s['rule']} surface"
        if s.get("mode") == "fallback_psi_N":
            return (f"psi_N = {s['fallback_psi_N']:g} (FALLBACK: no in-spec "
                    f"draw has a {s['rule']} surface)")
        if s.get("mode") == "fixed_psi_N":
            return f"psi_N = {s['psi_N']:g}"
        return "no gradients requested"

    def rows(self) -> list:
        """One plain dict per quantity (what :meth:`table` prints)."""
        out = []
        for q, r in self.records.items():
            flags = [n for n in ("below_floor", "no_band", "gated")
                     if getattr(r, n)]
            if r.status != "ok":
                flags.insert(0, r.status + (f"({r.empty_reason})"
                                            if r.empty_reason else ""))
            out.append(dict(
                quantity=q, unit=self.units.get(q, ""), median=r.median,
                p16=r.p16, p84=r.p84, min=r.min, max=r.max, n=r.n_used,
                n_selected=r.n_selected, n_irregular=r.n_irregular,
                kind="gradient" if q in SUMMARY_GRADIENTS else "scalar",
                flags=flags))
        return out

    def table(self) -> str:
        """The error-bar table as text (median, 16-84 % band, full range, n)."""
        lo, hi = (next(iter(self.records.values())).percentiles
                  if self.records else (16, 84))
        head = (f"Error bars over the {self.population} of scan "
                f"{self.scan_key!r}: n = {self.n_selected}  (median, "
                f"p{lo:g}-p{hi:g} band, full range, n used)")
        lines = [head]
        if self.population != "in-spec draws":
            lines.append(f"NOTE: the population is the {self.population}, "
                         "not an in-spec selection")
        if any(q in SUMMARY_GRADIENTS for q in self.records):
            s = self.surface
            g = f"gradients d/dpsi_N at {self.where}"
            if s.get("mode") == "own_surface":
                g += (f": located on {s['n_located']} of {s['n_considered']} "
                      f"draws ({s['n_no_surface']} without one, "
                      "counted and excluded)")
            lines.append(g)
        cols = ("quantity", "median", f"p{lo:g}", f"p{hi:g}", "min", "max", "n")
        w = max([len(c) for c in self.records] + [8])
        lines.append(f"{cols[0]:<{w}}  " + "  ".join(f"{c:>11}" for c in cols[1:6])
                     + f"  {cols[6]:>4}  unit / flags")
        for row in self.rows():
            vals = "  ".join(f"{row[k]:>11.4g}" for k in ("median", "p16", "p84",
                                                           "min", "max"))
            tail = row["unit"]
            if row["flags"]:
                tail = (tail + "  " if tail else "") + ",".join(row["flags"])
            lines.append(f"{row['quantity']:<{w}}  {vals}  {row['n']:>4}  {tail}")
        return "\n".join(lines)

    def __str__(self):
        return self.table()

    def __repr__(self):
        return (f"<BandSummary scan={self.scan_key!r} n_selected="
                f"{self.n_selected} quantities={list(self.records)} "
                f"gradients at: {self.where}>")

    def to_dict(self) -> dict:
        return {"scan_key": self.scan_key, "archive": self.archive,
                "n_selected": self.n_selected, "surface": _jsonable(self.surface),
                "units": dict(self.units),
                "records": {q: r.to_dict() for q, r in self.records.items()}}

    def plot(self, ax=None):
        """One band figure: see :func:`plot_bands_summary`."""
        return plot_bands_summary(self, ax=ax)


def bands_summary(archive, scan_key=None, *,
                  scalars=("beta_N", "l_i_3", "Lambda"),
                  gradients=("dp_dpsi", "djphi_dpsi"), at=None,
                  fallback_psi_N=0.5, **kw) -> BandSummary:
    r"""The standard error bars of one archive slice, in one call.

    Per quantity: the median, the 16-84 % band, the full range (min, max)
    and n over the in-spec draws -- the population and statistic of
    :func:`draw_band` (``**kw`` is passed on: ``selection``,
    ``require_filter``, ``min_n``, ``hard_min``, ``percentiles``,
    ``exclude``, ``rescued``, ...). Nothing is hand-rolled and nothing is
    filtered on.

    ``scalars``: names from :data:`SUMMARY_SCALARS` -- ``beta_N`` (the draw's
    g-file), ``beta_p`` and ``Lambda`` = beta_p + l_i(3)/2 (the stored attrs,
    else formed as :func:`draw_scalars` does), ``l_i_3`` / ``l_i_1`` (the
    archived attrs), ``q0`` / ``q95``.

    ``gradients``: names from :data:`SUMMARY_GRADIENTS` -- ``dp_dpsi`` and
    ``djphi_dpsi``, the derivative with respect to psi_N of the draw's own
    archived ``pressure`` [Pa] and ``jphi_total`` [A m^-2] profiles
    (:data:`GRADIENT_DEFINITION`), evaluated on each draw at its OWN
    surface:

    * ``at={"q": 2.0}`` (the default): the outermost q = 2 crossing of the
      draw's archived q profile (``eq_fsa/q``, else its g-file ``qpsi``; the
      source of every draw is recorded). A draw without a q = 2 surface is
      counted and excluded from the gradient bands (``regular=False``, label
      ``no_q=2_surface``; when half or fewer have one the record is ``gated``,
      the draw_band rule). When NO in-spec draw has one the gradients fall
      back to ``fallback_psi_N`` on every draw and the record says so
      (``surface["mode"] = "fallback_psi_N"``, a limitation line on every
      gradient record); ``fallback_psi_N=None`` disables the fallback (the
      gradient records then come back ``empty``).
    * ``at={"psi_N": x}``: every draw at psi_N = x.

    The baseline is overlaid on the same rule (its own surface, from its
    g-file), never the centre. Returns a :class:`BandSummary`; print it (or
    ``.table()``) for the table, ``.plot()`` for one band figure with n.
    """
    from .archive import DrawView, _gfile_pressure_scalars
    from .filtering import select_indices
    from .utils import load_baseline_profiles, load_eq_fsa
    at = {"q": 2.0} if at is None else dict(at)
    if len(at) != 1 or next(iter(at)) not in ("q", "psi_N"):
        raise ValueError(f"at must be {{'q': value}} or {{'psi_N': value}}, got {at!r}")
    at_key, at_val = next(iter(at.items()))
    at_val = float(at_val)
    bad_s = [s for s in scalars if s not in SUMMARY_SCALARS]
    bad_g = [g for g in gradients if g not in SUMMARY_GRADIENTS]
    if bad_s or bad_g:
        raise ValueError(
            f"unknown quantities {bad_s + bad_g}: scalars are "
            f"{sorted(SUMMARY_SCALARS)}, gradients {sorted(SUMMARY_GRADIENTS)}")
    if fallback_psi_N is not None and not (0.0 <= float(fallback_psi_N) <= 1.0):
        raise ValueError(f"fallback_psi_N must lie in [0, 1], got {fallback_psi_N!r}")
    ar = _as_archive(archive)
    key = _resolve_key(ar, scan_key)
    rule = f"q = {at_val:g}" if at_key == "q" else f"psi_N = {at_val:g}"
    label = f"no_q={at_val:g}_surface"

    try:
        bl = load_baseline_profiles(ar.path, scan_key=key)
    except KeyError:
        bl = {}
    # the baseline's l_i overlay: the recorded TokaMaker value (legacy
    # li_metrics), else the baseline's l_i_target on the estimator its
    # l_i_scale names (the unified engine's delivered l_i)
    _lm = bl.get("li_metrics") or {}
    _scale = str(_decode(bl.get("l_i_scale", "")))
    _tgt = bl.get("l_i_target")
    li_base = {"l_i_3": _lm.get("tokamaker_li_3"), "l_i_1": _lm.get("tokamaker_li_1")}
    li_base_src = {k: ("li_metrics" if v is not None else None) for k, v in li_base.items()}
    for k, pre in (("l_i_3", "iter"), ("l_i_1", "std")):
        if li_base[k] is None and _tgt is not None and _scale.startswith(pre):
            li_base[k] = float(_decode(_tgt))
            li_base_src[k] = f"_baseline.l_i_target ({_scale})"

    q_cache = {}

    def _q_of(view, eq_fn):
        """(psi_N, |q|, source) of the view's archived q profile."""
        ck = "_baseline" if view.is_baseline else int(view.count)
        if ck in q_cache:
            return q_cache[ck]
        out = (None, None, None)
        if not view.is_baseline:
            fsa = load_eq_fsa(ar.path, view.count, scan_key=key)
            if fsa is not None and "q" in fsa and "psi_N" in fsa:
                out = (np.asarray(fsa["psi_N"], float), np.abs(fsa["q"]), "eq_fsa")
        if out[0] is None:
            eq = eq_fn()
            if eq is not None:
                out = (np.asarray(eq.psi_N, float),
                       np.abs(np.asarray(eq.qpsi, float)), "geqdsk")
        q_cache[ck] = out
        return out

    # ---- pass 1: which in-spec draws have the surface (decides the fallback)
    selection = kw.get("selection", "selected")
    sel = select_indices(ar.path, scan_key=key, selection=selection)
    if isinstance(sel, dict):
        sel = next(iter(sel.values()), [])
    excl = {int(d) for d in (kw.get("exclude") or {})}
    considered = [int(d) for d in sel if int(d) not in excl]
    if kw.get("rescued") == "exclude":
        # the band's own population (draw_band drops them): a rescued draw's
        # surface must not keep the mode at own_surface
        _resc = rescued_draws(ar.path, key, considered)
        considered = [d for d in considered if d not in _resc]
    located, no_surface, no_q, q_src = {}, [], [], {}
    mode = "fixed_psi_N" if at_key == "psi_N" else "own_surface"
    if gradients and at_key == "q":
        for d in considered:
            v = DrawView(ar, key, d)

            def _eq(v=v):
                try:
                    return v.equilibrium()
                except KeyError:
                    return None
            psi, q, src = _q_of(v, _eq)
            q_src[d] = src
            if psi is None:
                no_q.append(d)            # no q profile at all: a status, not "no surface"
                continue
            x = _outermost_crossing(psi, q, at_val)
            if x is None:
                no_surface.append(d)
            else:
                located[d] = x
        if no_surface and not located and fallback_psi_N is not None:
            mode = "fallback_psi_N"
    surface = dict(rule=rule, mode=mode, target=at_val, coordinate=at_key,
                   crossing="outermost", fallback_psi_N=(
                       None if fallback_psi_N is None else float(fallback_psi_N)),
                   n_considered=len(considered), n_located=len(located),
                   n_no_surface=len(no_surface), no_surface_draws=sorted(no_surface),
                   n_no_q_profile=len(no_q), no_q_profile_draws=sorted(no_q),
                   psi_N_by_draw=dict(sorted(located.items())),
                   q_source_by_draw=dict(sorted(q_src.items())))
    if at_key == "psi_N":
        surface["psi_N"] = at_val
    # filled by the evaluator: where the baseline overlay was evaluated
    surface["baseline_q_source"] = None
    surface["baseline_psi_N"] = None

    def _surface_psi(view, eq_fn):
        if mode == "fixed_psi_N":
            return at_val, None
        if mode == "fallback_psi_N":
            return float(fallback_psi_N), None
        if not view.is_baseline and int(view.count) in located:
            return located[int(view.count)], None
        psi, q, _src = _q_of(view, eq_fn)
        x = None if psi is None else _outermost_crossing(psi, q, at_val)
        return x, (None if x is not None else
                   ("no_q_profile" if psi is None else "no_surface"))

    def _gradient(view, name, psi_s, eq_fn):
        from .schema import stored_name
        prof = view.profiles
        ds = stored_name(prof, SUMMARY_GRADIENTS[name][0])   # a pre-v4 archive: j_phi
        if ds not in prof or "psi_N" not in prof:
            return {"value": _NAN, "status": f"no_{ds}"}
        y = np.asarray(prof[ds], float)
        x = np.asarray(prof["psi_N"], float)
        if y.shape != x.shape:
            return {"value": _NAN, "status": f"{ds}_grid_mismatch"}
        coord = str(_decode(view.attrs.get("profile_coord", "psi_n")))
        if coord == "phi_n":
            psi, q, _src = _q_of(view, eq_fn)
            if psi is None:
                return {"value": _NAN, "status": "no_q_profile"}
            x = _psi_from_phi(x, psi, q)
        if x.size < 3 or not np.all(np.diff(x) > 0):
            return {"value": _NAN, "status": "non_monotone_grid"}
        g = np.gradient(y, x)
        return {"value": float(np.interp(psi_s, x, g)), "status": "ok",
                "regular": True}

    def _evaluate(view):
        eq_box = []

        def eq_fn():
            if not eq_box:
                try:
                    eq_box.append(view.equilibrium())
                except KeyError:
                    eq_box.append(None)
            return eq_box[0]
        out = {}
        a = view.attrs
        for s in scalars:
            if s == "beta_N":
                eq = eq_fn()
                out[s] = ({"value": _NAN, "status": "no_eqdsk"} if eq is None else
                          {"value": _gfile_pressure_scalars(eq)[1], "status": "ok"})
            elif s in ("beta_p", "Lambda"):
                out[s] = _shafranov_entries(view)[s]
            elif s in ("l_i_3", "l_i_1"):
                attr = "l_i(3)" if s == "l_i_3" else "l_i(1)"
                if view.is_baseline:
                    v = li_base[s]
                    out[s] = ({"value": float(v), "status": "ok"} if v is not None
                              else {"value": _NAN, "status": UNRECORDED})
                else:
                    out[s] = ({"value": float(a[attr]), "status": "ok"} if attr in a
                              else {"value": _NAN, "status": f"no_{attr}_attr"})
            else:                                   # q0 / q95
                psi, q, _src = _q_of(view, eq_fn)
                if psi is None:
                    out[s] = {"value": _NAN, "status": "no_q_profile"}
                else:
                    out[s] = {"value": float(q[0] if s == "q0" else
                                             np.interp(0.95, psi, q)), "status": "ok"}
        if gradients:
            psi_s, why = _surface_psi(view, eq_fn)
            if view.is_baseline:
                surface["baseline_psi_N"] = psi_s
                surface["baseline_q_source"] = _q_of(view, eq_fn)[2]
            for g in gradients:
                if psi_s is None:
                    out[g] = ({"value": _NAN, "status": "no_q_profile"}
                              if why == "no_q_profile" else
                              {"value": _NAN, "status": "ok", "regular": False,
                               "label": label})
                else:
                    out[g] = _gradient(view, g, psi_s, eq_fn)
        return out

    from .filtering import LAMBDA_DEFINITION
    meta = dict(kw.pop("evaluator_meta", None) or {})
    meta.setdefault("code", "bouquet.bands_summary")
    meta.update(scalar_definitions={s: SUMMARY_SCALARS[s][1] for s in scalars},
                gradient_definition=GRADIENT_DEFINITION if gradients else None,
                lambda_definition=LAMBDA_DEFINITION, surface_rule=rule,
                surface_mode=mode,
                baseline_l_i_source={k: li_base_src[k] for k in scalars
                                     if k in li_base_src})
    meta.setdefault("regular_label", f"the draw has a {rule} surface")
    kw["quantities"] = [*scalars, *gradients]
    recs = draw_band(ar, key, _evaluate, evaluator_meta=meta, **kw)
    for g in gradients:
        r = recs.get(g)
        if r is None:
            continue
        prov = dict(r.provenance or {})
        prov["surface"] = _jsonable(surface)
        lim = list(prov.get("limitations") or [])
        if mode == "fallback_psi_N":
            lim.append(
                f"No in-spec draw has a {rule} surface: {g} is evaluated at the "
                f"FALLBACK psi_N = {float(fallback_psi_N):g} on every draw, NOT "
                f"at {rule}.")
        elif mode == "own_surface" and no_surface:
            lim.append(
                f"{len(no_surface)} of the {len(considered)} in-spec draws have no "
                f"{rule} surface and are excluded from the {g} band (counted as "
                f"irregular: draws {sorted(no_surface)}).")
        prov["limitations"] = lim
        r.provenance = prov
    units = {s: SUMMARY_SCALARS[s][0] for s in scalars}
    units.update({g: SUMMARY_GRADIENTS[g][1] for g in gradients})
    n_sel = next(iter(recs.values())).n_selected if recs else 0
    return BandSummary(records=recs, surface=surface,
                       scan_key=None if key is None else str(key),
                       archive=ar.path, n_selected=int(n_sel), units=units)


# --------------------------------------------------------------------------
#  plotting
# --------------------------------------------------------------------------
def plot_band(table, quantity, ax=None, x=None):
    """Median line + p16-p84 band for ``quantity`` across a :class:`BandTable`.

    Filled marker: n_used >= min_n. Hollow marker: ``below_floor``. No band
    where ``no_band``. No marker where ``show=False`` (gated, or no draws):
    the regular fraction (``n_regular/n_ok``) or the status is written there
    instead. A shown band that excludes irregular draws is annotated with
    ``"<n_regular>/<n_ok> reg."``: it is conditional on the regular outcome. The baseline is overlaid dashed, never used as the centre.
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
            if r.n_irregular > 0:        # shown, but conditional on "regular"
                ax.text(xi, 0.03, f"{r.n_regular}/{r.n_ok} reg.", transform=tr,
                        ha="center", va="bottom", fontsize=7, color="C0")
            continue
        txt = ("n=0" if r.status == "empty"
               else r.status if r.status != "ok"
               else f"{r.n_regular}/{r.n_ok}" if r.n_ok else "n=0")
        ax.text(xi, 0.03, txt, transform=tr, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(str(quantity))
    return ax


def plot_bands_summary(summary, ax=None):
    """One band figure of a :class:`BandSummary`, n stated.

    One row per quantity, every value as its deviation from that quantity's
    median in % of |median|: the thick bar is the p16-p84 band, the thin line
    the full range (min, max), the dot the median (hollow when
    ``below_floor``), the cross the baseline overlay. The row label carries
    the median and n; the title the in-spec count and where the gradients
    were evaluated. A record with no band (or ``show=False``) is drawn
    without one and labelled with its status. Returns the axes.
    """
    import matplotlib.pyplot as plt
    rows = list(summary.records.items())
    made = ax is None
    if made:
        _, ax = plt.subplots(figsize=(8.0, 0.55 * len(rows) + 1.8))
    labels = []
    for i, (q, r) in enumerate(rows):
        y = len(rows) - 1 - i
        med = r.median
        unit = summary.units.get(q, "")
        if r.status != "ok" or not np.isfinite(med) or med == 0.0:
            labels.append(f"{q}: {r.status if r.status != 'ok' else 'no median'}"
                          f" (n={r.n_used})")
            continue
        scale = 100.0 / abs(med)

        def dev(v):
            return (v - med) * scale if v is not None and np.isfinite(v) else np.nan
        ax.plot([dev(r.min), dev(r.max)], [y, y], "-", color="0.45", lw=1.0)
        if r.show and not r.no_band:
            ax.plot([dev(r.p16), dev(r.p84)], [y, y], "-", color="C0", lw=6,
                    solid_capstyle="butt", alpha=0.8)
        ax.plot([0.0], [y], "o", color="C0",
                mfc="none" if r.below_floor else "C0", ms=6)
        if r.baseline_value is not None:
            ax.plot([dev(r.baseline_value)], [y], "x", color="k", ms=7)
        tag = "" if (r.show and not r.no_band) else "  [no band]"
        labels.append(f"{q} = {med:.3g} {unit} (n={r.n_used}){tag}".replace("  (", " ("))
    ax.set_yticks(range(len(rows) - 1, -1, -1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="0.7", lw=0.8, zorder=0)
    lo, hi = (rows[0][1].percentiles if rows else (16, 84))
    ax.set_xlabel(f"deviation from the median [% of |median|]\n"
                  f"bar: p{lo:g}-p{hi:g}, line: full range, dot: median, "
                  f"x: baseline", fontsize=8)
    title = f"{summary.population}: n = {summary.n_selected}"
    if any(q in SUMMARY_GRADIENTS for q in summary.records):
        title += f"\ngradients at {summary.where}"
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    if made:
        ax.figure.tight_layout()
    return ax
