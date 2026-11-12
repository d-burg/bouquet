"""Measured sawtooth crash tables (generic reader).

A crash table lists the crashes detected in one discharge: per crash its time [s], the time uncertainty
[s], the inversion radius (rho_tor,norm) with its uncertainty, and a confidence label; plus the time range
the detection covered (``record_start``, ``record_end`` [s]) and, optionally, sub-intervals in which the
detection could not see the core (``unobserved``, e.g. a diagnostic cutoff): a crash-free stretch there
means "not observed", not "no crash".

:func:`read_crash_table` reads a delimited text table with one header row naming the columns (lines
starting with ``#`` are comments; the comments are kept in the record).  The column names are mapped by
the caller (``columns``), with defaults for a common layout (``t_s``, ``t_err_s``, ``rho_inv``,
``rho_inv_err``, ``conf``); a missing optional column gives NaN / empty values, a missing time column is
refused.  ``min_confidence`` keeps only crashes whose label is in the given set (recorded).  The
coverage is stated by the caller (``record_start``, ``record_end``, ``unobserved``) -- a table's first
and last crash are NOT its coverage, and when the coverage is not stated the record says so.

The series driver (:mod:`.series`) uses a table through :meth:`CrashTable.times` and its
``record_start`` (the start of the time range over which crashes were detected, needed to decide whether
the record covers the history before a window).
"""

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CrashTable", "read_crash_table", "DEFAULT_COLUMNS"]

#: Default column names: role -> header name.
DEFAULT_COLUMNS = dict(time="t_s", time_err="t_err_s", rho_inv="rho_inv", rho_inv_err="rho_inv_err",
                       confidence="conf")


@dataclass
class CrashTable:
    """Measured crashes of one discharge and the coverage of the detection (module docstring)."""
    time: np.ndarray
    time_err: np.ndarray
    rho_inv: np.ndarray
    rho_inv_err: np.ndarray
    confidence: Sequence[str]
    record_start: Optional[float] = None
    record_end: Optional[float] = None
    unobserved: Tuple[Tuple[float, float], ...] = ()
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        n = self.time.size
        for k in ("time_err", "rho_inv", "rho_inv_err"):
            v = np.asarray(getattr(self, k), float)
            if v.shape != (n,):
                raise ValueError(f"crash table: {k} has shape {v.shape}, expected ({n},)")
            setattr(self, k, v)
        self.confidence = [str(c) for c in self.confidence]
        if len(self.confidence) != n:
            raise ValueError("crash table: one confidence label per crash")
        if n and (not np.all(np.isfinite(self.time)) or np.any(np.diff(self.time) <= 0)):
            raise ValueError("crash table: crash times must be finite and strictly increasing")
        if self.record_start is not None and self.record_end is not None and not self.record_end > self.record_start:
            raise ValueError("crash table: record_end must be after record_start")
        self.unobserved = tuple((float(a), float(b)) for a, b in self.unobserved)
        for a, b in self.unobserved:
            if not b > a:
                raise ValueError("crash table: every unobserved interval needs end > start")

    def times(self):
        """The crash times [s] as an array."""
        return self.time.copy()

    def covered(self, t0, t1):
        """Whether the detection covered ``[t0, t1]`` fully: inside ``[record_start, record_end]`` and not
        overlapping an ``unobserved`` interval.  None when the coverage was not stated."""
        if self.record_start is None or self.record_end is None:
            return None
        if t0 < self.record_start or t1 > self.record_end:
            return False
        return not any(a < t1 and b > t0 for a, b in self.unobserved)

    def as_dict(self):
        return dict(n=int(self.time.size), time=self.time.tolist(), time_err=self.time_err.tolist(),
                    rho_inv=self.rho_inv.tolist(), rho_inv_err=self.rho_inv_err.tolist(),
                    confidence=list(self.confidence), record_start=self.record_start, record_end=self.record_end,
                    unobserved=[list(u) for u in self.unobserved],
                    coverage_stated=self.record_start is not None and self.record_end is not None,
                    source=self.source, meta=dict(self.meta))


def _num(x):
    x = (x or "").strip()
    if not x:
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_crash_table(path_or_text, columns=None, min_confidence=None, record_start=None, record_end=None,
                     unobserved=(), delimiter=",", source=None):
    """Read a crash table (module docstring).

    ``path_or_text``: a file path or the table's text.  ``columns``: role -> header name overrides of
    :data:`DEFAULT_COLUMNS` (roles ``time``, ``time_err``, ``rho_inv``, ``rho_inv_err``, ``confidence``).
    ``min_confidence``: a collection of labels to keep (None keeps all).  ``record_start`` /
    ``record_end`` [s] and ``unobserved`` (``[(t0, t1), ...]`` [s]): the coverage of the detection, stated
    by the caller.  Rows are sorted by time; duplicate times are refused.  Returns a :class:`CrashTable`.
    """
    if isinstance(path_or_text, str) and os.path.isfile(path_or_text):
        with open(path_or_text) as fh:
            text = fh.read()
        src = source or os.path.basename(path_or_text)
    else:
        text = str(path_or_text)
        src = source or "text"
    cols = dict(DEFAULT_COLUMNS)
    unknown = set(columns or {}) - set(cols)
    if unknown:
        raise ValueError(f"crash table: unknown column roles {sorted(unknown)}; known: {sorted(cols)}")
    cols.update(columns or {})
    comments = [ln[1:].strip() for ln in text.splitlines() if ln.lstrip().startswith("#")]
    body = "\n".join(ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body), delimiter=delimiter))
    if rows and cols["time"] not in rows[0]:
        raise ValueError(f"crash table: time column {cols['time']!r} not in the header {sorted(rows[0])}")
    missing = [r for r in ("time_err", "rho_inv", "rho_inv_err", "confidence") if rows and cols[r] not in rows[0]]
    keep, dropped = [], 0
    for r in rows:
        conf = (r.get(cols["confidence"]) or "").strip()
        if min_confidence is not None and conf not in set(min_confidence):
            dropped += 1
            continue
        t = _num(r.get(cols["time"]))
        if not np.isfinite(t):
            raise ValueError(f"crash table: a row has no valid time ({r})")
        keep.append((t, _num(r.get(cols["time_err"])), _num(r.get(cols["rho_inv"])), _num(r.get(cols["rho_inv_err"])),
                     conf))
    keep.sort(key=lambda x: x[0])
    t = np.array([k[0] for k in keep], float)
    if t.size and np.any(np.diff(t) <= 0):
        raise ValueError("crash table: duplicate crash times")
    meta = dict(columns=cols, missing_optional_columns=missing, comments=comments,
                min_confidence=None if min_confidence is None else sorted(set(min_confidence)),
                n_rows=len(rows), n_dropped_by_confidence=dropped,
                coverage_note=("stated by the caller" if record_start is not None and record_end is not None
                               else "coverage not stated: a crash-free stretch cannot be read as 'no crash'"))
    return CrashTable(time=t, time_err=np.array([k[1] for k in keep], float),
                      rho_inv=np.array([k[2] for k in keep], float), rho_inv_err=np.array([k[3] for k in keep], float),
                      confidence=[k[4] for k in keep],
                      record_start=None if record_start is None else float(record_start),
                      record_end=None if record_end is None else float(record_end),
                      unobserved=tuple(unobserved or ()), source=src, meta=meta)
