"""High-level reader for a bouquet HDF5 archive.

The single authoritative home for archive-schema knowledge on the read side.
Downstream code should walk an archive through :class:`BouquetArchive` rather
than hand-rolling h5 tree traversal + ``.eqdsk``-suffix scanning + byte parsing
(the ~100-line pattern this replaces in the IDA_run debug scripts).

::

    ar = bq.BouquetArchive("run.h5")     # or a header stem, or a Bouquet
    ar.scan_keys                         # ['4400']  (flat legacy -> [None])
    sc = ar["4400"]                      # ScanView  (or ar.scan() when only one)
    sc.indices, sc.baseline              # gap-tolerant indices; baseline dict
    sc.selected, sc.excluded, sc.all     # lists of DrawView (per filter flags)
    d = sc[3]                            # DrawView (lazy)
    d.li1, d.li3, d.flags, d.attrs       # scalars + filter/spec metadata
    d.profiles                           # dict of named arrays (psi_N, j_phi, ...)
    d.equilibrium()                      # parsed GEQDSKEquilibrium (stored bytes)
    d.pfile()                            # parsed PFile (None if absent)
    d.extract("out/", formats=("geqdsk", "pfile"))

The functional readers (``load_equilibrium`` / ``select_indices`` / …) remain
the public API and are unchanged; this class is a convenience layer over the
same schema. Eqdsk/pfile datasets are located by the schema-v2 fixed names
(``eqdsk`` / ``pfile``) with a **suffix-scan fallback**, so pre-v2 archives
with ``{header}_{scan}_{count}.eqdsk`` names still yield their bytes (their
profile keys, however, remain v1 bracketed names -- a warning is emitted on
open).

Scalar metadata (``attrs`` and everything derived from it: ``li1``, ``li3``,
``selected``, ``flags``) is cached per view on first access -- a ``DrawView``
is a snapshot. After re-running a filter, get fresh views from the archive
(or call :meth:`DrawView.refresh`).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .schema import EQDSK_DS, PFILE_DS, find_bytes_dataset
from .utils import (
    _resolve_h5, _scan_key, _group_path,
    discover_scan_keys, list_equilibrium_indices, load_baseline_profiles,
    read_eqdsk_from_bytes,
)
from .filtering import select_indices, _FILTER_FLAGS


# Datasets that are not perturbed-profile arrays (excluded from DrawView.profiles).
_NON_PROFILE = {"config_json", EQDSK_DS, PFILE_DS}

# Attr keys surfaced by DrawView.flags (same record shape as read_filter_flags).
_FLAG_ATTRS = (*_FILTER_FLAGS, "selected")
_FLAG_METRICS = ("max_F_drift_pct", "max_VSC_drift_pct", "in_spec")


def _gfile_pressure_scalars(eq):
    r"""``(<P> [kPa], beta_N)`` of a parsed g-file equilibrium.

    :math:`\langle P\rangle = \int p\,dV / V` from the g-file's own pressure
    and traced geometry, and the reader's ``beta_n``. The one definition used
    by :meth:`ScanView.spread` and :func:`bouquet.stats.draw_scalars`.
    """
    vol = np.asarray(eq.geometry["vol"], dtype=float)
    p_avg_kpa = float(eq.volume_integral(np.asarray(eq.pres))[-1]) / float(vol[-1]) / 1e3
    return p_avg_kpa, float(eq.betas["beta_n"])


class DrawView:
    """Lazy view of one perturbed draw (``scan/<key>/<count>``)."""

    def __init__(self, archive: "BouquetArchive", scan_key, count: int):
        self._ar = archive
        self.scan_key = scan_key
        self.count = int(count)
        self._attrs_cache = None

    #: False for a perturbed draw; :class:`BaselineView` sets it True, so one
    #: evaluator can serve both (see :func:`bouquet.stats.draw_band`).
    is_baseline = False

    def __repr__(self):
        return f"<DrawView scan={self.scan_key!r} count={self.count}>"

    @property
    def _gp(self) -> str:
        """HDF5 group path of this view (overridden by :class:`BaselineView`)."""
        return _group_path(self.scan_key, self.count)

    def refresh(self) -> "DrawView":
        """Drop cached metadata (re-read after e.g. re-running a filter)."""
        self._attrs_cache = None
        return self

    # ---- scalars / metadata ------------------------------------------------
    @property
    def attrs(self) -> dict:
        """All group attrs (scalars + filter/spec metadata) as a plain dict.

        Read once and cached on the view -- ``d.li1; d.li3; d.selected`` costs
        one file open, not three, and looping ``.flags`` over N draws is O(N)
        instead of a whole-scan re-read per draw. Snapshot semantics: after
        re-filtering, take fresh views (or :meth:`refresh`).
        """
        if self._attrs_cache is None:
            import h5py
            with h5py.File(self._ar.path, "r") as hf:
                a = hf[self._gp].attrs
                self._attrs_cache = {
                    k: (a[k].item() if hasattr(a[k], "item") else a[k]) for k in a}
        return dict(self._attrs_cache)

    @property
    def li1(self) -> float:
        return float(self.attrs["l_i(1)"])

    @property
    def li3(self) -> float:
        return float(self.attrs["l_i(3)"])

    @property
    def flags(self) -> dict:
        """Filter/spec flags for this draw (``passes_*``, ``selected``, drifts).

        Same record shape as ``read_filter_flags(...)[count]``, but built from
        this draw's own attrs -- no whole-scan sweep per access.
        """
        a = self.attrs
        rec = {k: bool(a[k]) for k in _FLAG_ATTRS if k in a}
        for k in _FLAG_METRICS:
            if k in a:
                rec[k] = bool(a[k]) if k == "in_spec" else float(a[k])
        return rec

    @property
    def selected(self) -> bool:
        return bool(self.attrs.get("selected", True))

    # ---- profiles / bytes --------------------------------------------------
    @property
    def profiles(self) -> dict:
        """Named 1-D arrays stored for the draw (``psi_N``, ``j_phi [..]``, …)."""
        import h5py
        out = {}
        with h5py.File(self._ar.path, "r") as hf:
            grp = hf[self._gp]
            for name in grp:
                ds = grp[name]
                if not isinstance(ds, h5py.Dataset):
                    continue
                if name in _NON_PROFILE:
                    continue
                if ds.dtype == object or ds.dtype.kind in ("V", "S", "O", "U"):
                    continue                       # opaque bytes / strings
                out[name] = np.array(ds)
        return out

    def _read_bytes(self, *suffixes) -> dict:
        """Fetch the named byte blobs in ONE file open: {suffix: bytes|None}."""
        import h5py
        out = {}
        with h5py.File(self._ar.path, "r") as hf:
            grp = hf[self._gp]
            for suffix in suffixes:
                name = find_bytes_dataset(grp, suffix.lstrip("."))
                out[suffix] = bytes(grp[name][()]) if name is not None else None
        return out

    @property
    def eqdsk_bytes(self) -> Optional[bytes]:
        return self._read_bytes(".eqdsk")[".eqdsk"]

    @property
    def pfile_bytes(self) -> Optional[bytes]:
        """Profiles-source bytes for this draw.

        A text p-file source is stored per draw (rewritten with the draw's
        perturbed kinetics). A binary IDA ``.cdf`` source is stored ONCE in
        ``_baseline`` (it cannot be draw-perturbed, and per-draw copies bloated
        archives ~190 MB x n_draws with the new IDA-database files) -- so when
        the draw carries no blob, fall back to the scan's baseline copy.
        """
        b = self._read_bytes(".pfile")[".pfile"]
        if b is not None:
            return b
        import h5py
        with h5py.File(self._ar.path, "r") as hf:
            bl = f"scan/{self.scan_key}/_baseline"
            if bl in hf:
                name = find_bytes_dataset(hf[bl], "pfile")
                if name is not None:
                    return bytes(hf[bl][name][()])
        return None

    def equilibrium(self):
        """Parse the stored eqdsk bytes into a ``GEQDSKEquilibrium``."""
        from .io.geqdsk import read_geqdsk
        b = self.eqdsk_bytes
        if b is None:
            raise KeyError(f"no eqdsk stored for {self!r}")
        return read_eqdsk_from_bytes(b, read_geqdsk)

    def pfile(self):
        """Parse the stored p-file bytes into a ``PFile`` (``None`` if absent)."""
        b = self.pfile_bytes
        if b is None:
            return None
        from .io.pfile import PFile
        return PFile.from_bytes(b) if hasattr(PFile, "from_bytes") else None

    def coil_currents(self) -> dict:
        """Coil currents as ``{name: current_A}`` (empty if not stored)."""
        from .utils import _read_coil_names
        import h5py
        with h5py.File(self._ar.path, "r") as hf:
            grp = hf[self._gp]
            if "coil_currents" not in grp:
                return {}
            vals = np.asarray(grp["coil_currents"][()], dtype=float)
            names = _read_coil_names(grp)
        return dict(zip(names, vals.tolist())) if names else {}

    def profiles_doc(self) -> dict:
        """Portable, self-describing dict of this draw's full profile state.

        Source-agnostic (works for both geqdsk and IMAS draws): the 1-D
        profile arrays + their units, the scalar diagnostics, coil currents,
        and the captured live-equilibrium FSA block when present. Serialised by
        :meth:`extract` as ``*_profiles.json``.
        """
        from .schema import PROFILE_UNITS, EQ_FSA_UNITS
        from .utils import load_eq_fsa
        prof = self.profiles
        doc = {
            "scan_key": _scan_key(self.scan_key),
            "count": self.count,
            "profiles": {k: np.asarray(v).tolist() for k, v in prof.items()},
            "units": {k: PROFILE_UNITS.get(k, "") for k in prof},
            "scalars": self.attrs,          # li, Ip, drifts, in_spec, ... (JSON-safe)
            "coil_currents_A": self.coil_currents(),
        }
        fsa = load_eq_fsa(self._ar.path, self.count, scan_key=self.scan_key)
        if fsa is not None:
            doc["eq_fsa"] = {k: np.asarray(v).tolist() for k, v in fsa.items()}
            doc["eq_fsa_units"] = {k: EQ_FSA_UNITS.get(k, "") for k in fsa}
        return doc

    def extract(self, out_dir: str, formats=("geqdsk",)) -> dict:
        """Write per-draw files to ``out_dir``; return ``{format: path}``.

        Formats: ``"geqdsk"`` / ``"pfile"`` (raw stored bytes) and
        ``"profiles"`` (a self-describing JSON of profiles + scalars + coils +
        eq_fsa; see :meth:`profiles_doc`). Missing payloads are skipped.
        """
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{self._ar.header_basename}_{_scan_key(self.scan_key)}_{self.count}"
        paths = {}
        if "geqdsk" in formats or "pfile" in formats:
            blobs = self._read_bytes(".eqdsk", ".pfile")   # one file open for both
            if "geqdsk" in formats and blobs[".eqdsk"] is not None:
                p = os.path.join(out_dir, stem + ".geqdsk")
                with open(p, "wb") as fh:
                    fh.write(blobs[".eqdsk"])
                paths["geqdsk"] = os.path.abspath(p)
            if "pfile" in formats:
                # binary IDA sources live once in _baseline -> property fallback
                pf = blobs[".pfile"] if blobs[".pfile"] is not None else self.pfile_bytes
                if pf is not None:
                    p = os.path.join(out_dir, stem + ".peqdsk")
                    with open(p, "wb") as fh:
                        fh.write(pf)
                    paths["pfile"] = os.path.abspath(p)
        if "profiles" in formats:
            import json
            p = os.path.join(out_dir, stem + "_profiles.json")
            with open(p, "w") as fh:
                json.dump(self.profiles_doc(), fh)
            paths["profiles"] = os.path.abspath(p)
        return paths


class BaselineView(DrawView):
    """The scan's baseline (``scan/<key>/_baseline``) behind the draw accessors.

    Exposes the same byte and profile accessors as :class:`DrawView`
    (``attrs``, ``profiles``, ``eqdsk_bytes``, ``pfile_bytes``,
    ``equilibrium()``, ``pfile()``, ``coil_currents()``), so an evaluator
    written for draws runs unchanged on the baseline -- the same code path and
    grid for the overlay. ``count`` is ``None`` and ``is_baseline`` is True.
    The baseline carries no per-draw ``l_i(1)``/``l_i(3)`` attrs, so ``li1`` /
    ``li3`` raise ``KeyError`` here.
    """

    is_baseline = True

    def __init__(self, archive: "BouquetArchive", scan_key):
        self._ar = archive
        self.scan_key = scan_key
        self.count = None
        self._attrs_cache = None

    def __repr__(self):
        return f"<BaselineView scan={self.scan_key!r}>"

    @property
    def _gp(self) -> str:
        bkey = _scan_key(self.scan_key)
        return f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"

    @property
    def pfile_bytes(self) -> Optional[bytes]:
        return self._read_bytes(".pfile")[".pfile"]

    def profiles_doc(self) -> dict:          # per-draw document; not defined here
        raise NotImplementedError("profiles_doc is a per-draw document")

    def extract(self, out_dir: str, formats=("geqdsk",)) -> dict:
        raise NotImplementedError("extract the baseline via the draw bundle tools")


class ScanView:
    """View of one scan point (``scan/<key>/``) -- its baseline + draws."""

    def __init__(self, archive: "BouquetArchive", scan_key):
        self._ar = archive
        self.scan_key = scan_key

    def __repr__(self):
        return f"<ScanView scan={self.scan_key!r} ({len(self.indices)} draws)>"

    @property
    def indices(self) -> list:
        """Stored draw indices (gap-tolerant, sorted)."""
        return list_equilibrium_indices(self._ar.path, scan_key=self.scan_key)

    @property
    def baseline(self) -> dict:
        """Baseline profiles + sigmas dict for this scan point."""
        return load_baseline_profiles(self._ar.path, scan_key=self.scan_key)

    def baseline_view(self) -> BaselineView:
        """The baseline behind the :class:`DrawView` accessors (bytes, parse)."""
        return BaselineView(self._ar, self.scan_key)

    def _draws(self, selection: str) -> list:
        idx = select_indices(self._ar.path, scan_key=self.scan_key, selection=selection)
        return [DrawView(self._ar, self.scan_key, i) for i in idx]

    @property
    def all(self) -> list:
        return self._draws("all")

    @property
    def selected(self) -> list:
        return self._draws("selected")

    @property
    def excluded(self) -> list:
        return self._draws("excluded")

    def extract(self, out_dir: str, formats=("geqdsk", "profiles"),
                selection: str = "selected") -> dict:
        """Extract a bundle (geqdsk / pfile / profiles) for every draw in
        ``selection`` to ``out_dir``; return ``{draw_index: {format: path}}``.

        One call for "hand me the g-file + profiles for every in-spec draw":
        ``ar[key].extract("out/", formats=("geqdsk", "profiles"))``.
        """
        return {d.count: d.extract(out_dir, formats=formats)
                for d in self._draws(selection)}

    def spread(self, selection: str = "all", print_table: bool = True) -> dict:
        r"""Across-draw spread of the bouquet's global output scalars.

        For every draw in *selection* (``"all"`` / ``"selected"`` /
        ``"excluded"``) reports the mean, 1-sigma, relative sigma, and min-max
        range of the key global equilibrium scalars:

          * ``l_i(1)`` / ``l_i(3)`` -- internal inductance (stored per draw);
          * ``<P>``   -- volume-averaged pressure :math:`\int p\,dV / V` [kPa];
          * ``beta_N`` -- normalized beta.

        ``<P>`` and ``beta_N`` are read from each draw's g-file geometry (one
        flux-surface trace per draw, so this is O(seconds) for a full family),
        putting the pressure-quantity uncertainty right alongside the l_i
        variance in a single call. Returns ``{quantity: {n, mean, std, rel_std,
        min, max}}`` (``None`` for a quantity with no finite draws); prints a
        formatted table unless ``print_table=False``.

        This is a quick-look summary (mean/std, no counts, floor or
        provenance). For quotable error bars use
        :func:`bouquet.stats.draw_scalars` (``bq.draw_scalars``), which applies
        the standard recipe: filtered population, median with p16/p84,
        per-stage counts and a provenance record.
        """
        draws = self._draws(selection)
        cols = {"l_i(1)": [], "l_i(3)": [], "<P> [kPa]": [], "beta_N": []}
        for d in draws:
            cols["l_i(1)"].append(d.li1)
            cols["l_i(3)"].append(d.li3)
            p_avg_kpa, beta_n = _gfile_pressure_scalars(d.equilibrium())
            cols["<P> [kPa]"].append(p_avg_kpa)
            cols["beta_N"].append(beta_n)

        out = {}
        for name, vals in cols.items():
            x = np.asarray(vals, dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                out[name] = None
                continue
            m = float(x.mean())
            s = float(x.std(ddof=1)) if x.size > 1 else 0.0
            out[name] = {"n": int(x.size), "mean": m, "std": s,
                         "rel_std": (s / m if m else float("nan")),
                         "min": float(x.min()), "max": float(x.max())}

        if print_table:
            print(f"Bouquet output spread -- scan {self.scan_key!r}, "
                  f"selection={selection!r} ({len(draws)} draws)")
            print(f"  {'quantity':<11}{'mean':>10}{'1sigma':>10}{'σ/mean':>8}   range")
            for name, st in out.items():
                if st is None:
                    print(f"  {name:<11}  (no finite draws)")
                    continue
                print(f"  {name:<11}{st['mean']:>10.3f}{st['std']:>10.3f}"
                      f"{100 * st['rel_std']:>7.2f}%   [{st['min']:.3f}, {st['max']:.3f}]")
        return out

    def __getitem__(self, count: int) -> DrawView:
        if int(count) not in self.indices:
            raise KeyError(
                f"draw {count} not in scan {self.scan_key!r}; have {self.indices}")
        return DrawView(self._ar, self.scan_key, int(count))

    def __iter__(self):
        return iter(self.all)

    def __len__(self):
        return len(self.indices)


class BouquetArchive:
    """High-level reader for a ``{header}.h5`` bouquet archive."""

    def __init__(self, ref):
        # Accept a Bouquet, a header stem, or a full .h5 path.
        header = getattr(getattr(ref, "config", None), "output_header", None)
        ref_str = header if header is not None else str(ref)
        self.path = _resolve_h5(ref_str)
        self.header_basename = os.path.splitext(os.path.basename(self.path))[0]
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        # Friendly legacy signal: files written by a v2 bouquet carry a
        # schema_version file attr (stamped at creation). Without it this is a
        # pre-v2 (or hand-built) archive -- byte blobs still resolve via the
        # suffix scan, but profile datasets keep their v1 bracketed names.
        import h5py
        with h5py.File(self.path, "r") as hf:
            if "schema_version" not in hf.attrs:
                import warnings
                warnings.warn(
                    f"{self.path} carries no schema_version attr (pre-v2 or "
                    "hand-built archive). eqdsk/pfile bytes are still "
                    "readable, but profile keys may be the old bracketed "
                    "names (e.g. 'j_phi [A m^-2]'); regenerate the archive "
                    "with the current bouquet for full v2 support.")

    def __repr__(self):
        return f"<BouquetArchive {self.path!r} scans={self.scan_keys}>"

    @property
    def scan_keys(self) -> list:
        """Scan-point keys as a list (flat/legacy layout -> ``[None]``)."""
        keys = discover_scan_keys(self.path)
        return [None] if keys is None else list(keys)

    def scan(self, scan_key=None) -> ScanView:
        """A :class:`ScanView`; ``scan_key=None`` picks the only scan (or errors)."""
        keys = self.scan_keys
        if scan_key is None:
            if len(keys) != 1:
                raise KeyError(
                    f"{len(keys)} scans present {keys}; pass an explicit scan_key")
            return ScanView(self, keys[0])
        want = _scan_key(scan_key)
        for k in keys:
            if _scan_key(k) == want:
                return ScanView(self, k)
        raise KeyError(f"scan_key {scan_key!r} not in {keys}")

    def __getitem__(self, scan_key) -> ScanView:
        return self.scan(scan_key)

    def __iter__(self):
        return (self.scan(k) for k in self.scan_keys)

    @property
    def provenance(self) -> dict:
        """File-level provenance attrs + parsed config (Phase-1 fields)."""
        import h5py
        out = {}
        with h5py.File(self.path, "r") as hf:
            for k in ("schema_version", "bouquet_version", "created", "updated"):
                if k in hf.attrs:
                    v = hf.attrs[k]
                    out[k] = v.item() if hasattr(v, "item") else v
        try:
            from .utils import load_config
            out["config"] = load_config(self.path)
        except Exception:
            out["config"] = None
        return out
