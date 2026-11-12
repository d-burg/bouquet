# Vendored verbatim from the bouquet reader module ``bouquet/io/zeff_sources.py`` at commit
# f878d2bcda6b29c34fd2739f9957cce2a967ee3e (the checked Z_eff source rule).  Do not edit here; change the
# source module and re-vendor.  Imports only numpy and the standard library.
"""The checked-source rule for the effective charge of a kinetic profile file.

A kinetic profile file can carry two candidates for Z_eff:

* an **impurity-density source** (``IMPURITY``): a measured impurity density
  ``n_imp`` of charge ``Z`` (charge-exchange spectroscopy), from which
  ``Z_eff = 1 + Z (Z - 1) n_imp / n_e`` follows by single-impurity
  quasineutrality;
* a **line-integrated source** (``LINE``): a Z_eff profile supplied with the
  file and labeled as derived from a line-integrated measurement (visible
  bremsstrahlung).

Either can be unsafe: the supplied Z_eff can be a constrained measurement, a
prior-dominated default or a copy of the impurity-derived value; the impurity
density can be a placeholder on slices where no channel was available, or an
extrapolation where the channels do not cover the core.  This module tests
each candidate per time slice (tests I-1 ... I-4 and L-1 ... L-4 plus one
advisory), applies the decision table and returns a structured record of
everything it measured.  Nothing is chosen by fit quality, by agreement with
a reconstruction or by a cross-check.

Three states per candidate
--------------------------
* ``"qualified"``: every test was evaluated in full and none failed;
* ``"unverified"``: no evaluated test failed, but at least one test, or part
  of one, could not be evaluated (no measurement points for I-1, no raw
  analyses for I-4, no uncertainty for the sigma half of I-2 or the z half of
  I-4, ...).  Every such part is listed in the candidate's and the record's
  ``not_evaluated`` entries.  A missing fast-ion density is NOT such a part:
  I-3 is then evaluated without fast ions and that is recorded separately
  (``evaluated_without``), because with singly charged main and fast ions
  the impurity-derived Z_eff = 1 + Z (Z - 1) n_imp / n_e does not depend on
  the fast-ion density -- fast ions change the thermal main-ion density and
  pressure (they matter for n_i), not the choice of Z_eff source;
* ``"failed"``: at least one evaluated test failed (or the candidate is not
  usable at all, e.g. an impurity charge the file's density is not for).

Ranking (:func:`decide`): a qualified impurity source, else a qualified
line-integrated value, else an unverified impurity source (used, labeled
unverified and warned about), else refused unless an explicit, recorded
override names a source.  An unverified line-integrated value is never used
ahead of an unverified impurity source and is not used at all without an
explicit override (the conservative reading; docs/zeff-source-rule.md).

Self-contained: this module imports only numpy and the standard library (no
``bouquet`` import), so it can be vendored by another package.  The reader
glue that does not depend on a file format lives here too: the slice's
time-window unit inference (:func:`slice_window`), the rho_tor map from q
(:func:`rho_map_from_q`), the advisory's slice series
(:func:`advisory_core_series`), the validity screen restricted to the plasma
(:func:`validity_inside_boundary`) and the warning / refusal text.
:func:`bouquet.io.ida.read_ida` (``ni_source="checked"``) gathers the inputs
from an IDA-layout file and attaches the record to the returned profiles.
See docs/zeff-source-rule.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "IMPURITY", "LINE", "PASS", "FAIL", "NOT_EVALUATED", "NOT_APPLICABLE",
    "QUALIFIED", "UNVERIFIED", "FAILED", "ABSENT", "HAND_FIT", "AUTOMATIC",
    "OTHER", "ANALYSIS_KINDS", "SCHEMA_VERSION", "FAST_IONS_SUPPLIED",
    "FAST_IONS_DECLARED_NONE", "FAST_IONS_NOT_AVAILABLE",
    "COVERAGE_RHO_INNER_MAX", "COVERAGE_RHO_OUTER_MIN", "COVERAGE_MIN_POINTS",
    "COVERAGE_WINDOW_S", "PLACEHOLDER_REL_SPREAD_MIN",
    "PLACEHOLDER_REL_SIGMA_MAX", "NI_FRACTION_MIN", "NI_FRACTION_WARN",
    "POINTS_ABS_MEDIAN_Z_MAX", "POINTS_RATIO_MIN", "POINTS_RATIO_MAX",
    "POINTS_RHO_MAX", "POINTS_CORE_RHO_MAX", "LINE_REL_SIGMA_MAX",
    "COPY_REL_DIFF_MIN", "FLAT_PTP_MIN", "CORE_CV_ADVISORY",
    "TENSION_Z_MAX", "PSI_N_TEST_MAX", "PSI_N_PLASMA_MAX",
    "PSI_N_CORE_MEDIAN_MAX", "FILL_ABS_MIN", "THRESHOLDS",
    "MeasurementPoints", "ZeffSourceRefused", "ZeffSourceWarning",
    "impurity_tests", "unusable_impurity", "line_tests", "tension", "decide",
    "zeff_source_record", "series_source", "is_placeholder_slice",
    "core_median", "validity_inside_boundary", "slice_window",
    "rho_map_from_q", "advisory_core_series", "fit_points_from_arrays",
    "checked_warning", "refusal_message", "forced_route_failures",
]

#: Candidate names used in the record.
IMPURITY = "impurity_density"
LINE = "line_integrated"

#: Test verdicts.  ``NOT_APPLICABLE``: the test has no object (L-3 on a file
#: without an impurity density: there is nothing to be a copy of).
PASS, FAIL, NOT_EVALUATED = "pass", "fail", "not_evaluated"
NOT_APPLICABLE = "not_applicable"

#: Candidate states.
QUALIFIED, UNVERIFIED, FAILED, ABSENT = ("qualified", "unverified", "failed",
                                         "absent")

#: Raw-analysis kinds (``MeasurementPoints.kind``).  The reference analysis
#: of I-1 / I-4 is the one labeled HAND_FIT when present, else the one
#: labeled AUTOMATIC; OTHER analyses are evaluated and recorded only.
HAND_FIT, AUTOMATIC, OTHER = "hand_fit", "automatic", "other"
ANALYSIS_KINDS = (HAND_FIT, AUTOMATIC, OTHER)

#: Version of the record layout (:func:`zeff_source_record`).
SCHEMA_VERSION = 2

#: How I-3 treated fast ions (``tests["I-3"]["fast_ions"]``): a fast-ion
#: density was supplied; the caller declared there are none (evaluated with
#: fast ions = 0 by declaration); none was available (evaluated without fast
#: ions -- recorded, and NOT a reason for "unverified": see impurity_tests).
FAST_IONS_SUPPLIED = "supplied"
FAST_IONS_DECLARED_NONE = "none"
FAST_IONS_NOT_AVAILABLE = "not_available"

# ---------------------------------------------------------------------------
#  Thresholds.  Each one was set from its measured distribution over the
#  files of a validation set (12 discharges of one carbon-wall device): the
#  value sits in the gap between the slices it is meant to reject and the
#  slices it is meant to accept.  They are part of the public behavior of
#  ni_source="checked" -- a change of any value changes which source a file
#  gets -- and every record carries the values it used (THRESHOLDS).
# ---------------------------------------------------------------------------

#: I-1 coverage: the innermost measurement point must lie at rho_tor,norm <=
#: this value.
COVERAGE_RHO_INNER_MAX = 0.30
#: I-1 coverage: the outermost point inside the boundary must lie at
#: rho_tor,norm >= this value.
COVERAGE_RHO_OUTER_MIN = 0.80
#: I-1 coverage: at least this many points inside the boundary.
COVERAGE_MIN_POINTS = 8
#: I-1 / I-4: half-width [s] of the time window for supplied raw points, used
#: when the file states no window for the slice (+-25 ms, inclusive).
COVERAGE_WINDOW_S = 0.025
#: I-2 placeholder: the relative radial peak-to-peak of n_imp/n_e over
#: psi_N <= PSI_N_TEST_MAX must be at least this value.
PLACEHOLDER_REL_SPREAD_MIN = 1e-3
#: I-2 placeholder: the median relative uncertainty of n_imp over psi_N <=
#: PSI_N_TEST_MAX must be at most this value.
PLACEHOLDER_REL_SIGMA_MAX = 1.0
#: I-3 physical: n_i/n_e (fast ions included when supplied) must exceed this
#: value everywhere inside the boundary.
NI_FRACTION_MIN = 0.0
#: I-3 physical: a warning when the minimum of n_i/n_e is below this value.
NI_FRACTION_WARN = 0.1
#: I-4 consistency: |median z| of points against the profile at most this.
POINTS_ABS_MEDIAN_Z_MAX = 2.0
#: I-4 consistency: the median point/profile ratio must lie in
#: [POINTS_RATIO_MIN, POINTS_RATIO_MAX] (inclusive).
POINTS_RATIO_MIN = 2.0 / 3.0
POINTS_RATIO_MAX = 3.0 / 2.0
#: I-4 regions: rho_tor,norm <= POINTS_RHO_MAX ("all") and, separately,
#: rho_tor,norm < POINTS_CORE_RHO_MAX ("core").
POINTS_RHO_MAX = 0.9
POINTS_CORE_RHO_MAX = 0.5
#: L-2: the median relative uncertainty of the line-integrated Z_eff over
#: psi_N <= PSI_N_TEST_MAX must be at most this value.
LINE_REL_SIGMA_MAX = 0.25
#: L-3 not a copy: the maximum relative difference from the impurity-derived
#: Z_eff over psi_N <= PSI_N_TEST_MAX must EXCEED this value.
COPY_REL_DIFF_MIN = 1e-3
#: L-4 not flat: the radial peak-to-peak of the line-integrated Z_eff over
#: psi_N <= PSI_N_TEST_MAX must be at least this value.
FLAT_PTP_MIN = 0.02
#: Advisory: a warning when the coefficient of variation of the slice core
#: median (psi_N <= PSI_N_CORE_MEDIAN_MAX) over the file's non-placeholder
#: slices is below this value (a value that does not follow the discharge).
CORE_CV_ADVISORY = 0.05
#: Decision: both candidates qualify and |median z| of their difference over
#: psi_N <= PSI_N_TEST_MAX exceeds this value -> warn (the impurity source is
#: still used).
TENSION_Z_MAX = 2.0
#: Radial region [psi_N] of I-2, L-2, L-3, L-4 and the tension.
PSI_N_TEST_MAX = 0.9
#: Radial region [psi_N] of I-3 ("everywhere inside the boundary") and of
#: the validity screen the rule applies.
PSI_N_PLASMA_MAX = 1.0
#: Radial region [psi_N] of the slice core median (advisory).
PSI_N_CORE_MEDIAN_MAX = 0.5
#: Magnitude above which a value is a netCDF fill marker, not data (the same
#: value as the IDA reader's carbon screen).
FILL_ABS_MIN = 1e30

#: Every threshold, as recorded with each result.
THRESHOLDS = dict(
    coverage_rho_inner_max=COVERAGE_RHO_INNER_MAX,
    coverage_rho_outer_min=COVERAGE_RHO_OUTER_MIN,
    coverage_min_points=COVERAGE_MIN_POINTS,
    coverage_window_s=COVERAGE_WINDOW_S,
    placeholder_rel_spread_min=PLACEHOLDER_REL_SPREAD_MIN,
    placeholder_rel_sigma_max=PLACEHOLDER_REL_SIGMA_MAX,
    ni_fraction_min=NI_FRACTION_MIN,
    ni_fraction_warn=NI_FRACTION_WARN,
    points_abs_median_z_max=POINTS_ABS_MEDIAN_Z_MAX,
    points_ratio_min=POINTS_RATIO_MIN,
    points_ratio_max=POINTS_RATIO_MAX,
    points_rho_max=POINTS_RHO_MAX,
    points_core_rho_max=POINTS_CORE_RHO_MAX,
    line_rel_sigma_max=LINE_REL_SIGMA_MAX,
    copy_rel_diff_min=COPY_REL_DIFF_MIN,
    flat_ptp_min=FLAT_PTP_MIN,
    core_cv_advisory=CORE_CV_ADVISORY,
    tension_z_max=TENSION_Z_MAX,
    psi_n_test_max=PSI_N_TEST_MAX,
    psi_n_plasma_max=PSI_N_PLASMA_MAX,
    psi_n_core_median_max=PSI_N_CORE_MEDIAN_MAX,
)


class ZeffSourceRefused(ValueError):
    """Neither Z_eff candidate can be used (or a named one cannot be used).
    ``record`` holds the full rule record, failed tests included."""

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record


class ZeffSourceWarning(UserWarning):
    """A Z_eff source-rule finding: an unverified source in use, a tension
    above 2 sigma, an override, a strongly diluted plasma, an advisory, or a
    forced route whose source fails a test."""


@dataclass
class MeasurementPoints:
    """Impurity-density measurement points (one analysis), SI units.

    ``value`` and ``sigma`` in m^-3.  Positions as ``rho`` (rho_tor,norm) or
    ``psi_N`` (or both); the rule's coverage and consistency regions are in
    rho_tor,norm, so ``psi_N``-only points need a ``rho_map`` in
    :func:`impurity_tests`.  ``time`` [s] per point, or ``None`` for points
    already selected for the slice.  ``label`` names the analysis in the
    record.  ``kind`` labels a raw analysis: ``"hand_fit"``, ``"automatic"``
    or ``"other"`` -- the reference for I-1 / I-4 is the analysis labeled
    hand-fit when one is supplied, else the one labeled automatic; the order
    in which analyses are passed never decides.  Every analysis is recorded.
    """

    value: np.ndarray
    sigma: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    psi_N: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    label: str = "points"
    kind: str = OTHER

    def __post_init__(self):
        if self.kind not in ANALYSIS_KINDS:
            raise ValueError(f"MeasurementPoints.kind={self.kind!r} must be one "
                             f"of {ANALYSIS_KINDS}")

    @classmethod
    def coerce(cls, obj, label=None):
        if obj is None or isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            kw = {k: obj[k] for k in ("value", "sigma", "rho", "psi_N",
                                      "time", "label", "kind")
                  if k in obj}
            unknown = set(obj) - set(kw)
            if unknown:
                raise TypeError("measurement points: unknown key(s) "
                                f"{sorted(unknown)}")
            if label is not None and "label" not in kw:
                kw["label"] = str(label)
            return cls(**kw)
        raise TypeError(f"measurement points must be a MeasurementPoints or a "
                        f"dict, got {type(obj).__name__}")


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _f(x):
    """A JSON-safe float (None for None / non-finite)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _test(verdict, value=None, threshold=None, detail="", parts=(), **extra):
    """One test's record.  ``parts``: the parts of the test that could not
    be evaluated (strings); a test with a non-empty list leaves its
    candidate unverified unless the candidate fails."""
    out = dict(verdict=verdict, value=value, threshold=threshold,
               detail=str(detail), not_evaluated_parts=list(parts))
    out.update(extra)
    return out


def _arr(a):
    return None if a is None else np.asarray(a, dtype=float)


def _sigma_ok(s):
    """Nodes where an uncertainty is usable: finite and > 0 (a zero, a
    negative or a non-finite sigma counts as no uncertainty)."""
    return np.isfinite(s) & (s > 0)


def core_median(psi_N, profile):
    """The slice core median of *profile* (psi_N <= PSI_N_CORE_MEDIAN_MAX)."""
    psi_N, profile = _arr(psi_N), _arr(profile)
    m = (psi_N <= PSI_N_CORE_MEDIAN_MAX) & np.isfinite(profile)
    return float(np.median(profile[m])) if np.any(m) else float("nan")


def _rel_spread(psi_N, ne, n_imp):
    m = (psi_N <= PSI_N_TEST_MAX) & np.isfinite(n_imp) & np.isfinite(ne) \
        & (ne > 0)
    if not np.any(m):
        return None
    fc = n_imp[m] / ne[m]
    med = float(np.median(fc))
    if med == 0.0 or not np.isfinite(med):
        return None
    return float(np.ptp(fc) / abs(med))


def is_placeholder_slice(psi_N, ne, n_imp) -> bool:
    """Test I-2(a) alone: is the slice's n_imp/n_e radially flat (a
    placeholder)?  Used to exclude placeholder slices from the advisory."""
    s = _rel_spread(_arr(psi_N), _arr(ne), _arr(n_imp))
    return s is not None and s < PLACEHOLDER_REL_SPREAD_MIN


# ---------------------------------------------------------------------------
#  reader-independent glue (numpy only; used by bouquet.io.ida.read_ida)
# ---------------------------------------------------------------------------
def validity_inside_boundary(psi_N, ne, n_imp, sigma_n_imp=None,
                             sigma_ne=None):
    """The rule's validity screen: the reader's carbon-screen criteria
    (finite, positive, not a fill value; a finite non-negative sigma when
    supplied; a finite n_e sigma when supplied) applied INSIDE the boundary
    (psi_N <= PSI_N_PLASMA_MAX) only, where the rule's tests are defined.

    The arrays are on ``psi_N`` (last axis); a leading sample axis (the
    posterior samples of an ensemble file) is screened per sample.  Returns
    ``{"ok", "n_invalid_inside", "n_checked_inside", "n_invalid_outside",
    "n_checked_outside"}``: ``ok`` is False when any value inside the
    boundary is invalid; invalid values outside are counted and excluded.
    """
    psi = _arr(psi_N)
    ne, nc = _arr(ne), _arr(n_imp)
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(nc) & np.isfinite(ne) & (nc > 0) & (ne > 0)
              & (np.abs(nc) < FILL_ABS_MIN) & (np.abs(ne) < FILL_ABS_MIN))
        if sigma_n_imp is not None:
            s = _arr(sigma_n_imp)
            ok = ok & np.isfinite(s) & (s >= 0) & (np.abs(s) < FILL_ABS_MIN)
        if sigma_ne is not None:
            ok = ok & np.isfinite(_arr(sigma_ne))
    inside = psi <= PSI_N_PLASMA_MAX
    bad_in = ~ok[..., inside]
    bad_out = ~ok[..., ~inside]
    return dict(ok=not bool(np.any(bad_in)),
                n_invalid_inside=int(bad_in.sum()),
                n_checked_inside=int(bad_in.size),
                n_invalid_outside=int(bad_out.sum()),
                n_checked_outside=int(bad_out.size))


def slice_window(win_start, win_end, index, slice_time):
    """The slice's own time window [s] from per-slice window start / end
    arrays (``time_win_start`` / ``time_win_end`` of an IDA-layout file), or
    ``None``.  The arrays carry no unit; the unit (s or ms) is the one whose
    window contains ``slice_time`` [s]; a window that contains it in
    neither unit is not used."""
    if win_start is None or win_end is None:
        return None
    a = np.asarray(win_start, dtype=float).ravel()
    b = np.asarray(win_end, dtype=float).ravel()
    if a.size <= index or b.size <= index:
        return None
    for scale in (1.0, 1e-3):
        lo, hi = a[index] * scale, b[index] * scale
        if np.isfinite(lo) and np.isfinite(hi) and lo <= slice_time <= hi:
            return (float(lo), float(hi))
    return None


def rho_map_from_q(psi_N, q):
    """``(psi_N_inside, rho_tor_norm)`` on the nodes with psi_N <= 1, from
    the profile file's own q: Phi_N(psi_N) = int_0^psi_N q dpsi_N /
    int_0^1 q dpsi_N (trapezoid; q interpolated to psi_N = 1 when the grid
    does not hit it) and rho = sqrt(Phi_N).  Arithmetic identical to
    ``bouquet.coords.phi_n_from_q`` (without its bracket option).

    Raises ``ValueError`` naming the reason when the map cannot be built (a
    psi_N grid that does not rise from 0, a non-finite q inside the
    boundary)."""
    psi_N = np.asarray(psi_N, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    if q.shape != psi_N.shape:
        raise ValueError(f"q has shape {q.shape}, psi_N {psi_N.shape}")
    inside = psi_N <= 1.0
    x = psi_N[inside]
    if x.size < 2 or not np.all(np.diff(x) > 0) or abs(x[0]) > 1e-9:
        raise ValueError("psi_N must rise from 0")
    if not np.all(np.isfinite(q[inside])):
        raise ValueError("q is not finite inside the boundary")
    xe = x if x[-1] == 1.0 else np.append(x, 1.0)
    y = np.interp(xe, psi_N, q)
    # the cumulative trapezoid, term for term as scipy's cumulative_trapezoid
    phi = np.concatenate(([0.0], np.cumsum(np.diff(xe) * (y[1:] + y[:-1])
                                           / 2.0)))
    phi_n = phi[:x.size] / phi[-1]
    return x, np.sqrt(np.clip(phi_n, 0.0, None))


def advisory_core_series(psi_N, zeff_slices, ne_slices, n_imp_slices=None):
    """The advisory's input: the core median (psi_N <= 0.5) of the
    line-integrated Z_eff on every slice of a file, leaving out the slices
    whose impurity density is a placeholder (test I-2(a)).  Each argument is
    a sequence of per-slice profiles on ``psi_N`` (``n_imp_slices`` None: no
    impurity density, every slice counts)."""
    out = []
    for k, z_k in enumerate(zeff_slices):
        if n_imp_slices is not None and is_placeholder_slice(
                psi_N, ne_slices[k], n_imp_slices[k]):
            continue
        out.append(core_median(psi_N, z_k))
    return out


def fit_points_from_arrays(value, sigma, psi_N, label="file fit points"):
    """The points a profile was fitted to at one slice, from padded arrays
    (fill values and non-finite entries are padding and are dropped; an
    all-padding slice gives an empty set, which fails I-1)."""
    v, e, x = (np.asarray(a, dtype=float).ravel() for a in (value, sigma,
                                                            psi_N))
    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(v) & np.isfinite(x) & (np.abs(v) < FILL_ABS_MIN)
              & (np.abs(x) < FILL_ABS_MIN))
        e = np.where(np.isfinite(e) & (np.abs(e) < FILL_ABS_MIN), e, np.nan)
    return MeasurementPoints(value=v[ok], sigma=e[ok], psi_N=x[ok],
                             label=label)


# ---------------------------------------------------------------------------
#  points
# ---------------------------------------------------------------------------
def _positions(pts, rho_map):
    """(rho, psi_N) of each point; NaN where it cannot be placed or lies
    outside the boundary (rho > 1, psi_N > 1)."""
    n = np.size(pts.value)
    rho = None if pts.rho is None else np.asarray(pts.rho, dtype=float)
    psi = None if pts.psi_N is None else np.asarray(pts.psi_N, dtype=float)
    if rho_map is not None:
        mp, mr = (np.asarray(a, dtype=float) for a in rho_map)
        if rho is None and psi is not None:
            rho = np.where((psi >= 0) & (psi <= mp[-1]),
                           np.interp(psi, mp, mr), np.nan)
        if psi is None and rho is not None:
            psi = np.where((rho >= 0) & (rho <= mr[-1]),
                           np.interp(rho, mr, mp), np.nan)
    if rho is None:
        rho = np.full(n, np.nan)
    if psi is None:
        psi = np.full(n, np.nan)
    rho = np.where(rho <= 1.0, rho, np.nan)
    return rho, psi


def _in_window(pts, slice_time, slice_window):
    """Mask of the points inside the slice's window (all when the points
    carry no times) and the window used, as a description."""
    n = np.size(pts.value)
    if pts.time is None:
        return np.ones(n, dtype=bool), "points carry no times (taken as "\
            "selected for this slice)"
    t = np.asarray(pts.time, dtype=float)
    if slice_window is not None:
        a, b = float(slice_window[0]), float(slice_window[1])
        return (t >= a) & (t <= b), f"the slice's own window [{a:.4f}, "\
            f"{b:.4f}] s"
    if slice_time is None:
        raise ValueError("raw points with times need the slice time or window")
    hw = COVERAGE_WINDOW_S
    return (np.abs(t - float(slice_time)) <= hw), \
        f"+-{1e3 * hw:.0f} ms about {float(slice_time):.4f} s"


def _valid(pts):
    v = np.asarray(pts.value, dtype=float)
    return np.isfinite(v) & (v > 0)


def _coverage(rho):
    """(inner, outer, n_inside) over the points with a finite rho <= 1."""
    ok = np.isfinite(rho)
    if not np.any(ok):
        return None, None, 0
    return float(np.min(rho[ok])), float(np.max(rho[ok])), int(ok.sum())


def _coverage_test(rho, source):
    inner, outer, n = _coverage(rho)
    ok = (inner is not None and inner <= COVERAGE_RHO_INNER_MAX
          and outer >= COVERAGE_RHO_OUTER_MIN and n >= COVERAGE_MIN_POINTS)
    why = []
    if n < COVERAGE_MIN_POINTS:
        why.append(f"{n} point(s) inside the boundary (< "
                   f"{COVERAGE_MIN_POINTS})")
    if inner is not None and inner > COVERAGE_RHO_INNER_MAX:
        why.append(f"innermost point at rho {inner:.3f} (> "
                   f"{COVERAGE_RHO_INNER_MAX})")
    if outer is not None and outer < COVERAGE_RHO_OUTER_MIN:
        why.append(f"outermost point at rho {outer:.3f} (< "
                   f"{COVERAGE_RHO_OUTER_MIN})")
    return _test(PASS if ok else FAIL,
                 value=dict(rho_inner=_f(inner), rho_outer=_f(outer),
                            n_inside=int(n)),
                 threshold=dict(rho_inner_max=COVERAGE_RHO_INNER_MAX,
                                rho_outer_min=COVERAGE_RHO_OUTER_MIN,
                                min_points=COVERAGE_MIN_POINTS),
                 detail=("; ".join(why) if why else "covered"),
                 points=str(source))


def _consistency_region(rho, psi_pts, v_pts, s_pts, psi_N, n_imp, sigma,
                        region):
    if region == "all":
        m = np.isfinite(rho) & (rho <= POINTS_RHO_MAX)
    else:
        m = np.isfinite(rho) & (rho < POINTS_CORE_RHO_MAX)
    m = m & np.isfinite(psi_pts)
    if not np.any(m):
        return _test(NOT_EVALUATED, detail=f"no points in the {region} "
                     "region", parts=[f"the {region} region has no points"],
                     n=0, z_evaluated=False)
    prof = np.interp(psi_pts[m], psi_N, n_imp)
    good = np.isfinite(prof) & (prof > 0)
    if not np.any(good):
        return _test(FAIL, detail="profile not positive at the points", n=0,
                     z_evaluated=False)
    ratio = v_pts[m][good] / prof[good]
    med_ratio = float(np.median(ratio))
    med_z = None
    if s_pts is not None and sigma is not None:
        sp = np.interp(psi_pts[m], psi_N, sigma)[good]
        spt = s_pts[m][good]
        den = np.hypot(spt, sp)
        with np.errstate(invalid="ignore"):
            usable = np.isfinite(den) & (den > 0) & np.isfinite(spt) \
                & (spt >= 0) & np.isfinite(sp) & (sp >= 0)
        zz = np.divide(v_pts[m][good] - prof[good], den,
                       out=np.full(den.shape, np.nan), where=usable)
        if np.any(np.isfinite(zz)):
            med_z = float(np.nanmedian(zz))
    ratio_ok = POINTS_RATIO_MIN <= med_ratio <= POINTS_RATIO_MAX
    z_ok = None if med_z is None else abs(med_z) <= POINTS_ABS_MEDIAN_Z_MAX
    why, parts = [], []
    if not ratio_ok:
        why.append(f"median point/profile ratio {med_ratio:.3f} outside "
                   f"[{POINTS_RATIO_MIN:.4f}, {POINTS_RATIO_MAX:.4f}]")
    if z_ok is False:
        why.append(f"|median z| {abs(med_z):.3f} > {POINTS_ABS_MEDIAN_Z_MAX}")
    if z_ok is None:
        why.append("z not evaluated (no point or profile uncertainty)")
        parts.append(f"the z half in the {region} region (no point or "
                     "profile uncertainty)")
    verdict = PASS if (ratio_ok and z_ok is not False) else FAIL
    return _test(verdict,
                 value=dict(median_ratio=_f(med_ratio), median_z=_f(med_z)),
                 threshold=dict(ratio=[POINTS_RATIO_MIN, POINTS_RATIO_MAX],
                                abs_median_z_max=POINTS_ABS_MEDIAN_Z_MAX),
                 detail="; ".join(why) if why else "consistent", parts=parts,
                 n=int(np.count_nonzero(good)), z_evaluated=z_ok is not None)


def _reference(raw):
    """The reference raw analysis, chosen by label: the one labeled
    hand-fit when present, else the one labeled automatic.  Ambiguity (two
    of the deciding kind) or no analysis of either kind is an error: the
    caller labels the analyses, the order never decides."""
    for kind in (HAND_FIT, AUTOMATIC):
        sel = [p for p in raw if p.kind == kind]
        if len(sel) > 1:
            raise ValueError(
                f"{len(sel)} raw analyses are labeled kind={kind!r} "
                f"({', '.join(p.label for p in sel)}); label exactly one as "
                "the reference")
        if sel:
            return sel[0]
    raise ValueError(
        "raw analyses were supplied, but none is labeled "
        f"kind={HAND_FIT!r} or {AUTOMATIC!r}: the reference analysis of "
        "I-1 / I-4 is chosen by that label (hand-fit, else automatic), never "
        "by the order; label them")


def _finish(cand, gates):
    """The candidate's ``failed``, ``not_evaluated`` and ``state``.
    ``gates``: the qualification tests (the advisory is not one)."""
    tests = cand["tests"]
    failed = [k for k in gates if k in tests and tests[k]["verdict"] == FAIL]
    parts = []
    for k in gates:
        if k not in tests:
            continue
        for p in tests[k]["not_evaluated_parts"]:
            parts.append(f"{k}: {p}")
    cand["failed"] = failed
    cand["not_evaluated"] = parts
    cand["state"] = FAILED if failed else UNVERIFIED if parts else QUALIFIED
    cand["qualifies"] = cand["state"] == QUALIFIED
    return cand


# ---------------------------------------------------------------------------
#  the candidate tests
# ---------------------------------------------------------------------------
IMPURITY_TESTS = ("I-1", "I-2", "I-3", "I-4")
LINE_TESTS = ("L-1", "L-2", "L-3", "L-4")


def impurity_tests(psi_N, ne, n_imp, sigma_n_imp=None, *, impurity_Z=6.0,
                   n_fast=None, rho_map=None, fit_points=None,
                   raw_points=None, slice_time=None, slice_window=None,
                   validity=None):
    """Tests I-1 ... I-4 of an impurity-density candidate at one slice.

    Parameters
    ----------
    psi_N, ne, n_imp : array
        The profile grid and the electron and impurity densities [m^-3].
    sigma_n_imp : array, optional
        The impurity density's 1-sigma (I-2(b); the I-4 z statistic).  A
        node with a zero, negative or non-finite sigma counts as having no
        uncertainty.
    impurity_Z : float
        Impurity charge.
    n_fast : array or "none", optional
        Fast-ion charge density on ``psi_N`` [m^-3] (sum Z n_fast; singly
        charged ions: their density), included in I-3's quasineutrality;
        ``"none"``: the caller declares there are no fast ions (evaluated
        with fast ions = 0, recorded as a declaration).  Without it (None)
        I-3 is evaluated without fast ions, recorded on the I-3 entry
        (``evaluated_without_fast_ions``, ``fast_ions``) and in the
        candidate's ``evaluated_without`` list -- not as a part not
        evaluated, so it does not by itself leave the candidate unverified:
        with singly charged main and fast ions the impurity-derived Z_eff is
        independent of the fast-ion density, which changes the thermal n_i
        (and so the dilution warning), not the choice of source.
    rho_map : (psi_N, rho_tor_norm) arrays, optional
        Places points and regions in rho_tor,norm (I-1, I-4).
    fit_points : MeasurementPoints, optional
        The points the profile was fitted to at this slice (the file's own).
        Preferred for I-1 (coverage) and used for nothing else.  An empty
        set (the fit used no points) fails I-1.
    raw_points : sequence or dict of MeasurementPoints, optional
        Raw analyses with times; selected inside ``slice_window`` (the
        file's stated window, [s]) or within +-COVERAGE_WINDOW_S of
        ``slice_time``.  The reference analysis is the one labeled
        ``kind="hand_fit"``, else the one labeled ``"automatic"``; all are
        recorded.  I-4 is evaluated against the reference raw analysis
        only, never against the fit points; without raw analyses it is not
        evaluated.
    validity : dict, optional
        :func:`validity_inside_boundary` of the data (the reader passes the
        per-sample screen of an ensemble file); computed here from
        ``ne`` / ``n_imp`` / ``sigma_n_imp`` when omitted.  Invalid values
        inside the boundary fail I-3 as missing or bad data; invalid values
        outside are excluded and counted.

    Returns the candidate dict: ``present``, ``usable``, ``tests``,
    ``summary``, ``findings`` (warnings worth reporting when this source is
    used), ``failed``, ``not_evaluated`` and ``state``.
    """
    psi_N, ne, n_imp = _arr(psi_N), _arr(ne), _arr(n_imp)
    sigma = _arr(sigma_n_imp)
    Z = float(impurity_Z)
    findings, tests = [], {}
    fit = MeasurementPoints.coerce(fit_points, "fit points")
    raw = [MeasurementPoints.coerce(p, label=k) for k, p in (
        raw_points.items() if isinstance(raw_points, dict)
        else enumerate(raw_points or ()))]
    raw = [p for p in raw if p is not None]
    seen = set()
    for i, p in enumerate(raw):         # unique labels: the record's keys
        if p.label in seen:
            raw[i] = p = replace(p, label=f"{p.label} [{i}]")
        seen.add(p.label)
    ref = _reference(raw) if raw else None
    if validity is None:
        validity = validity_inside_boundary(psi_N, ne, n_imp, sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        zimp = 1.0 + Z * (Z - 1.0) * n_imp / np.where(ne > 0, ne, np.nan)
    m9 = psi_N <= PSI_N_TEST_MAX
    z9 = zimp[m9]
    z9 = z9[np.isfinite(z9)]
    summary = dict(
        zeff_median=_f(np.median(z9)) if z9.size else None,
        zeff_min=_f(np.min(z9)) if z9.size else None,
        zeff_max=_f(np.max(z9)) if z9.size else None,
        sigma_present=sigma is not None)

    # ---- I-1 coverage
    if fit is not None:
        ok = _valid(fit)
        rho, _ = _positions(fit, rho_map)
        if np.size(fit.value) and not np.any(np.isfinite(rho[ok])) and \
                fit.rho is None and rho_map is None:
            tests["I-1"] = _test(NOT_EVALUATED, detail=(
                "the fit points are on psi_N and no rho_tor map is available"),
                parts=["coverage (fit points on psi_N, no rho_tor map)"],
                points="fit points")
        else:
            tests["I-1"] = _coverage_test(np.where(ok, rho, np.nan),
                                          "fit points (the points the "
                                          "profile was fitted to)")
    elif raw:
        rec = {}
        for p in raw:
            w, wdesc = _in_window(p, slice_time, slice_window)
            rho, _ = _positions(p, rho_map)
            sel = w & _valid(p)
            if p.rho is None and rho_map is None:
                rec[p.label] = _test(NOT_EVALUATED, detail=(
                    "points on psi_N and no rho_tor map"),
                    parts=["coverage (points on psi_N, no rho_tor map)"],
                    points=p.label)
            else:
                rec[p.label] = _coverage_test(np.where(sel, rho, np.nan),
                                              f"{p.label}, {wdesc}")
        t = dict(rec[ref.label])
        t["reference_analysis"] = ref.label
        t["reference_kind"] = ref.kind
        t["all_analyses"] = rec
        tests["I-1"] = t
    else:
        tests["I-1"] = _test(NOT_EVALUATED, detail=(
            "no measurement points supplied"),
            parts=["coverage (no measurement points)"])

    # ---- I-2 placeholder
    spread = _rel_spread(psi_N, ne, n_imp)
    relsig, n_nosig = None, None
    if sigma is not None:
        base = m9 & np.isfinite(n_imp) & (n_imp > 0)
        mm = base & _sigma_ok(sigma)
        n_nosig = int(np.count_nonzero(base & ~_sigma_ok(sigma)))
        if np.any(mm):
            relsig = float(np.median(sigma[mm] / n_imp[mm]))
    a_ok = spread is not None and spread >= PLACEHOLDER_REL_SPREAD_MIN
    b_ok = None if relsig is None else relsig <= PLACEHOLDER_REL_SIGMA_MAX
    why, parts = [], []
    if not a_ok:
        why.append("n_imp/n_e radially flat (relative spread "
                   f"{spread if spread is not None else float('nan'):.3g} < "
                   f"{PLACEHOLDER_REL_SPREAD_MIN:g})")
    if b_ok is False:
        why.append(f"median relative sigma {relsig:.3g} > "
                   f"{PLACEHOLDER_REL_SIGMA_MAX:g}")
    if b_ok is None:
        why.append("relative-sigma half not evaluated (no usable impurity "
                   "density uncertainty)")
        parts.append("the relative-sigma half (no usable impurity-density "
                     "uncertainty)")
    tests["I-2"] = _test(
        PASS if (a_ok and b_ok is not False) else FAIL,
        value=dict(rel_spread=_f(spread), median_rel_sigma=_f(relsig),
                   n_nodes_without_sigma=n_nosig),
        threshold=dict(rel_spread_min=PLACEHOLDER_REL_SPREAD_MIN,
                       median_rel_sigma_max=PLACEHOLDER_REL_SIGMA_MAX),
        detail="; ".join(why) if why else "not a placeholder", parts=parts,
        sigma_evaluated=b_ok is not None)

    # ---- I-3 physical (inside the boundary)
    excluded = dict(n_invalid_outside=validity["n_invalid_outside"],
                    n_checked_outside=validity["n_checked_outside"],
                    note=(f"{validity['n_invalid_outside']} invalid value(s) "
                          "outside the boundary (psi_N > "
                          f"{PSI_N_PLASMA_MAX:g}) excluded"
                          if validity["n_invalid_outside"] else
                          "no invalid value outside the boundary"))
    mp = (psi_N <= PSI_N_PLASMA_MAX) & np.isfinite(ne) & (ne > 0)
    if isinstance(n_fast, str):
        if n_fast != FAST_IONS_DECLARED_NONE:
            raise ValueError(f"n_fast={n_fast!r}: an array, None (not "
                             f"available) or {FAST_IONS_DECLARED_NONE!r} "
                             "(declared: no fast ions)")
        nf, fast_basis = np.zeros_like(ne), FAST_IONS_DECLARED_NONE
        nf_note = "declared: no fast ions (fast ions = 0)"
    else:
        nf, fast_basis = _arr(n_fast), FAST_IONS_SUPPLIED
        nf_note = "fast-ion density supplied"
        if nf is not None and not np.all(np.isfinite(nf[mp])):
            nf, fast_basis = None, FAST_IONS_NOT_AVAILABLE
            nf_note = ("the fast-ion density is not finite inside the "
                       "boundary: evaluated without fast ions")
        elif nf is None:
            fast_basis = FAST_IONS_NOT_AVAILABLE
            nf_note = ("no fast-ion density available: evaluated without "
                       "fast ions")
    without = nf is None
    if not validity["ok"]:
        tests["I-3"] = _test(
            FAIL, value=dict(ni_over_ne_min=None, zeff_max=None,
                             n_invalid_inside=validity["n_invalid_inside"],
                             n_checked_inside=validity["n_checked_inside"]),
            threshold=dict(ni_over_ne_min=NI_FRACTION_MIN,
                           ni_over_ne_warn=NI_FRACTION_WARN, zeff_max=Z),
            detail=(f"the impurity density is invalid inside the boundary: "
                    f"{validity['n_invalid_inside']} of "
                    f"{validity['n_checked_inside']} value(s) missing, "
                    "non-finite, fill or non-positive (missing or bad "
                    "data, not a dilution result)"),
            cause="invalid_data", fast_ions_included=False,
            fast_ions=fast_basis, evaluated_without_fast_ions=without,
            fast_ion_note=nf_note, excluded=excluded, warn=False)
    elif not np.any(mp):
        tests["I-3"] = _test(NOT_EVALUATED, detail="no nodes inside the "
                             "boundary", parts=["no nodes inside the boundary"],
                             fast_ions_included=False, fast_ions=fast_basis,
                             evaluated_without_fast_ions=without,
                             fast_ion_note=nf_note, excluded=excluded,
                             warn=False)
    else:
        frac = 1.0 - Z * n_imp[mp] / ne[mp]
        if nf is not None:
            frac = frac - nf[mp] / ne[mp]
        fmin = float(np.min(frac))
        zmax = float(np.max(zimp[mp]))
        ok = fmin > NI_FRACTION_MIN and zmax <= Z
        warn = bool(ok and fmin < NI_FRACTION_WARN)
        tests["I-3"] = _test(
            PASS if ok else FAIL,
            value=dict(ni_over_ne_min=_f(fmin), zeff_max=_f(zmax)),
            threshold=dict(ni_over_ne_min=NI_FRACTION_MIN,
                           ni_over_ne_warn=NI_FRACTION_WARN, zeff_max=Z),
            detail=("physical" if ok else
                    f"n_i/n_e min {fmin:.4f} <= {NI_FRACTION_MIN:g}"
                    if fmin <= NI_FRACTION_MIN else
                    f"Z_eff max {zmax:.3f} > Z = {Z:g}"),
            cause=None if ok else "dilution",
            fast_ions_included=nf is not None, fast_ions=fast_basis,
            evaluated_without_fast_ions=without, fast_ion_note=nf_note,
            excluded=excluded, warn=warn)
        if warn:
            findings.append(f"n_i/n_e falls to {fmin:.3f} (< "
                            f"{NI_FRACTION_WARN:g}): a strongly diluted "
                            "plasma")

    # ---- I-4 consistency with supplied raw analyses
    if ref is None:
        tests["I-4"] = _test(NOT_EVALUATED, detail=(
            "no raw analyses supplied (I-4 is never evaluated against the "
            "points the profile was fitted to)"),
            parts=["no raw analyses supplied"])
    else:
        per = {}
        for p in raw:
            w, wdesc = _in_window(p, slice_time, slice_window)
            rho, psi_p = _positions(p, rho_map)
            rho = np.where(w & _valid(p), rho, np.nan)
            v = np.asarray(p.value, dtype=float)
            s = None if p.sigma is None else np.asarray(p.sigma, dtype=float)
            per[p.label] = dict(
                kind=p.kind, window=wdesc,
                all=_consistency_region(rho, psi_p, v, s, psi_N, n_imp,
                                        sigma, "all"),
                core=_consistency_region(rho, psi_p, v, s, psi_N, n_imp,
                                         sigma, "core"))
        r = per[ref.label]
        verdicts = [r["all"]["verdict"], r["core"]["verdict"]]
        if FAIL in verdicts:
            v = FAIL
        elif PASS in verdicts:
            v = PASS
        else:
            v = NOT_EVALUATED
        parts = [] if v == FAIL else (r["all"]["not_evaluated_parts"]
                                      + r["core"]["not_evaluated_parts"])
        tests["I-4"] = _test(
            v, value=dict(all=r["all"]["value"], core=r["core"]["value"]),
            threshold=dict(ratio=[POINTS_RATIO_MIN, POINTS_RATIO_MAX],
                           abs_median_z_max=POINTS_ABS_MEDIAN_Z_MAX,
                           rho_all_max=POINTS_RHO_MAX,
                           rho_core_max=POINTS_CORE_RHO_MAX),
            detail=f"all: {r['all']['detail']}; core: {r['core']['detail']}",
            parts=parts, regions=dict(all=r["all"], core=r["core"]),
            reference=f"raw analysis {ref.label!r} ({ref.kind})",
            reference_analysis=ref.label, reference_kind=ref.kind,
            all_analyses=per)
    cand = dict(present=True, usable=True, kind=IMPURITY, tests=tests,
                summary=summary, findings=findings,
                evaluated_without=(
                    [f"I-3: fast ions ({nf_note})"] if without else []))
    return _finish(cand, IMPURITY_TESTS)


def unusable_impurity(reason):
    """The candidate record of an impurity density the file carries but the
    rule cannot use (e.g. ``impurity_Z`` is not the charge the density is
    for): present, not usable, failed, with the reason."""
    return dict(present=True, usable=False, kind=IMPURITY, tests={},
                summary={}, findings=[], failed=[], not_evaluated=[],
                state=FAILED, qualifies=False, reason=str(reason))


def line_tests(psi_N, zeff, sigma_zeff=None, *, zeff_impurity=None,
               core_median_series=None):
    """Tests L-1 ... L-4 and the advisory of a line-integrated Z_eff at one
    slice.

    ``sigma_zeff``: a node with a zero, negative or non-finite sigma counts
    as having no uncertainty (an all-zero array fails L-1).
    ``zeff_impurity`` is the impurity-derived Z_eff on the same grid, when
    the file carries an impurity density (L-3; ``"not_applicable"`` without
    it: there is nothing to be a copy of).  ``core_median_series`` is the
    slice core median (psi_N <= 0.5) over the file's non-placeholder slices
    (the advisory, :func:`advisory_core_series`; not evaluated with fewer
    than two; it never disqualifies).
    """
    psi_N, zeff = _arr(psi_N), _arr(zeff)
    sig = _arr(sigma_zeff)
    m9 = (psi_N <= PSI_N_TEST_MAX) & np.isfinite(zeff)
    tests, findings = {}, []
    summary = dict(
        zeff_median=_f(np.median(zeff[m9])) if np.any(m9) else None,
        zeff_min=_f(np.min(zeff[m9])) if np.any(m9) else None,
        zeff_max=_f(np.max(zeff[m9])) if np.any(m9) else None,
        sigma_present=sig is not None)
    n9 = int(np.count_nonzero(m9))
    sok = None if sig is None else (m9 & _sigma_ok(sig))
    n_sig = 0 if sok is None else int(np.count_nonzero(sok))
    has_sig = n_sig > 0
    tests["L-1"] = _test(
        PASS if has_sig else FAIL,
        value=dict(n_nodes_with_sigma=n_sig, n_nodes=n9),
        threshold="required (finite and > 0)",
        detail=("carries its own uncertainty" if has_sig else
                "no Z_eff uncertainty in the file" if sig is None else
                "the Z_eff uncertainty is zero, negative or non-finite on "
                "every node (counted as none)"))
    if has_sig:
        mm = sok & (zeff > 0)
        rs = float(np.median(sig[mm] / zeff[mm])) if np.any(mm) else None
        ok = rs is not None and rs <= LINE_REL_SIGMA_MAX
        tests["L-2"] = _test(
            PASS if ok else FAIL, value=_f(rs), threshold=LINE_REL_SIGMA_MAX,
            detail=("constrained" if ok else
                    f"median relative sigma "
                    f"{rs if rs is not None else float('nan'):.3f} "
                    f"> {LINE_REL_SIGMA_MAX}"))
    else:
        tests["L-2"] = _test(NOT_EVALUATED, threshold=LINE_REL_SIGMA_MAX,
                             detail="no uncertainty (L-1 fails)",
                             parts=["no uncertainty (L-1 fails)"])
    zi = _arr(zeff_impurity)
    if zi is None:
        tests["L-3"] = _test(NOT_APPLICABLE, threshold=COPY_REL_DIFF_MIN,
                             detail="the file carries no impurity density "
                             "(nothing to be a copy of)")
    else:
        with np.errstate(invalid="ignore"):
            mm = m9 & np.isfinite(zi) & (zi > 0)
        if not np.any(mm):
            tests["L-3"] = _test(NOT_EVALUATED, threshold=COPY_REL_DIFF_MIN,
                                 detail="no valid impurity-derived Z_eff "
                                 f"over psi_N <= {PSI_N_TEST_MAX}",
                                 parts=["no valid impurity-derived Z_eff to "
                                        "compare with"])
        else:
            d = float(np.max(np.abs(zeff[mm] - zi[mm]) / zi[mm]))
            ok = d > COPY_REL_DIFF_MIN
            tests["L-3"] = _test(
                PASS if ok else FAIL, value=_f(d), threshold=COPY_REL_DIFF_MIN,
                detail=("not a copy" if ok else
                        "a copy of the impurity-derived Z_eff (max relative "
                        f"difference {d:.2g} <= {COPY_REL_DIFF_MIN:g})"))
    ptp = float(np.ptp(zeff[m9])) if np.any(m9) else None
    tests["L-4"] = _test(
        PASS if (ptp is not None and ptp >= FLAT_PTP_MIN) else FAIL,
        value=_f(ptp), threshold=FLAT_PTP_MIN,
        detail=("not flat" if (ptp is not None and ptp >= FLAT_PTP_MIN) else
                f"radially flat (peak-to-peak "
                f"{ptp if ptp is not None else float('nan'):.4f} < "
                f"{FLAT_PTP_MIN})"))
    ser = None if core_median_series is None else \
        np.asarray(core_median_series, dtype=float)
    ser = None if ser is None else ser[np.isfinite(ser)]
    if ser is None or ser.size < 2:
        tests["advisory"] = _test(NOT_EVALUATED, threshold=CORE_CV_ADVISORY,
                                  detail="fewer than two non-placeholder "
                                  "slices")
    else:
        cv = float(np.std(ser) / np.mean(ser))
        low = cv < CORE_CV_ADVISORY
        tests["advisory"] = _test(
            "warn" if low else PASS, value=_f(cv), threshold=CORE_CV_ADVISORY,
            detail=(f"core median varies by CV {cv:.3f} (< "
                    f"{CORE_CV_ADVISORY}) over {ser.size} slices: a value "
                    "that does not follow the discharge" if low else
                    f"core median CV {cv:.3f} over {ser.size} slices"),
            n_slices=int(ser.size))
        if low:
            findings.append("advisory: " + tests["advisory"]["detail"])
    cand = dict(present=True, usable=True, kind=LINE, tests=tests,
                summary=summary, findings=findings)
    return _finish(cand, LINE_TESTS)


def tension(psi_N, zeff_imp, sigma_zeff_imp, zeff_line, sigma_zeff_line):
    """The disagreement of the two candidates over psi_N <= PSI_N_TEST_MAX:
    z = (Z_imp - Z_line) / sqrt(sigma_imp^2 + sigma_line^2).  ``None`` when
    either candidate or either uncertainty is missing."""
    if any(a is None for a in (zeff_imp, sigma_zeff_imp, zeff_line,
                               sigma_zeff_line)):
        return None
    psi_N = _arr(psi_N)
    zi, si, zl, sl = (_arr(a) for a in (zeff_imp, sigma_zeff_imp, zeff_line,
                                        sigma_zeff_line))
    den = np.hypot(si, sl)
    m = ((psi_N <= PSI_N_TEST_MAX) & np.isfinite(zi) & np.isfinite(zl)
         & np.isfinite(den) & (den > 0))
    if not np.any(m):
        return None
    z = (zi[m] - zl[m]) / den[m]
    med = float(np.median(z))
    return dict(median_z=_f(med), max_abs_z=_f(np.max(np.abs(z))),
                median_ratio=_f(np.median(zi[m] / zl[m])),
                threshold=TENSION_Z_MAX,
                exceeds=bool(abs(med) > TENSION_Z_MAX),
                n_nodes=int(np.count_nonzero(m)))


# ---------------------------------------------------------------------------
#  the decision
# ---------------------------------------------------------------------------
_SHORT = {IMPURITY: "impurity source", LINE: "line-integrated Z_eff"}


def _state(c):
    return ABSENT if c is None else c["state"]


def _failure_text(c):
    """``"I-3 (detail); L-4 (detail)"`` for a failed candidate."""
    if c.get("reason") and not c.get("usable", True):
        return f"not usable ({c['reason']})"
    return "; ".join(f"{k} ({c['tests'][k]['detail']})" for k in c["failed"])


def _describe(name, c):
    """A short phrase for a candidate's state."""
    s = _state(c)
    if s == ABSENT:
        return ("no impurity density in the file" if name == IMPURITY
                else "no line-integrated Z_eff in the file")
    if s == QUALIFIED:
        return f"the {_SHORT[name]} is qualified"
    if s == FAILED:
        return f"the {_SHORT[name]} fails {_failure_text(c)}"
    return (f"the {_SHORT[name]} is unverified (not evaluated: "
            + "; ".join(c["not_evaluated"]) + ")")


def decide(impurity, line, *, tension_rec=None, override=None):
    """Apply the ranking.

    ``impurity`` / ``line``: the dicts of :func:`impurity_tests` (or
    :func:`unusable_impurity`) / :func:`line_tests`, or ``None`` (absent).
    Ranking: a qualified impurity source; else a qualified line-integrated
    value; else an unverified impurity source (labeled, warned); else
    refuse.  An unverified line-integrated value is used only by an
    explicit override.  ``override``: ``IMPURITY`` or ``LINE`` -- an
    explicit, recorded choice honored whatever the tests say, except that
    it is refused for a candidate the file does not carry or cannot use,
    and for an impurity density that fails I-3 (an override cannot make an
    invalid density or n_i <= 0 usable).  Returns the decision dict; the
    caller refuses when ``decision["source"]`` is None.
    """
    si, sl = _state(impurity), _state(line)
    warns = []
    if si == QUALIFIED:
        src, rank = IMPURITY, 1
        reason = (f"{_describe(IMPURITY, impurity)}: it is used; "
                  f"{_describe(LINE, line)}")
        if sl == QUALIFIED and tension_rec is not None \
                and tension_rec["exceeds"]:
            warns.append(
                "the two qualified Z_eff candidates disagree by |median z| = "
                f"{abs(tension_rec['median_z']):.2f} > {TENSION_Z_MAX:g} sigma "
                f"over psi_N <= {PSI_N_TEST_MAX}; the impurity source is used")
    elif sl == QUALIFIED:
        src, rank = LINE, 2
        reason = (f"{_describe(LINE, line)}: it is used; "
                  f"{_describe(IMPURITY, impurity)}")
    elif si == UNVERIFIED:
        src, rank = IMPURITY, 3
        reason = (f"{_describe(IMPURITY, impurity)}; no qualified "
                  f"alternative ({_describe(LINE, line)}"
                  + ("; an unverified line-integrated value is not used "
                     "without an override" if sl == UNVERIFIED else "")
                  + "): the unverified impurity source is used")
    else:
        src, rank = None, 4
        reason = (f"{_describe(IMPURITY, impurity)}; {_describe(LINE, line)}"
                  + ("; an unverified line-integrated value is not used "
                     "without an override" if sl == UNVERIFIED else ""))
    row = f"impurity {si}, line-integrated {sl}"
    out = dict(rule_source=src, rule_source_state=(None if src is None else
                                                   (si if src == IMPURITY
                                                    else sl)),
               states={IMPURITY: si, LINE: sl}, rank=rank, row=row,
               reason=reason, source=src,
               source_state=(None if src is None else
                             (si if src == IMPURITY else sl)),
               override=None, refused=src is None, warnings=warns)
    if override is not None:
        if override not in (IMPURITY, LINE):
            raise ValueError(f"override must be {IMPURITY!r} or {LINE!r}, got "
                             f"{override!r}")
        cand = impurity if override == IMPURITY else line
        if cand is None:
            raise ZeffSourceRefused(
                f"the override names the {override} source, but the file "
                "does not carry it")
        if not cand.get("usable", True):
            raise ZeffSourceRefused(
                f"the override names the {override} source, but it is not "
                f"usable: {cand.get('reason')}")
        if override == IMPURITY and "I-3" in cand["failed"]:
            raise ZeffSourceRefused(
                "the override names the impurity source, but it fails I-3 ("
                f"{cand['tests']['I-3']['detail']}); an override cannot make "
                "an invalid impurity density or n_i <= 0 usable")
        named_state = cand["state"]
        out["source"], out["refused"] = override, False
        out["source_state"] = named_state
        out["override"] = dict(
            source=override, rule_source=src, named_state=named_state,
            named_qualifies=named_state == QUALIFIED,
            named_failed_tests=list(cand["failed"]),
            named_not_evaluated=list(cand["not_evaluated"]))
        if named_state == FAILED:
            warns.append(f"override: the {_SHORT[override]} is used although "
                         f"it fails {', '.join(cand['failed'])}")
        elif named_state == UNVERIFIED:
            warns.append(f"override: the {_SHORT[override]} is used although "
                         "it is unverified")
        elif src is not None and src != override:
            warns.append(f"override: the {_SHORT[override]} is used where the "
                         f"rule would use the {_SHORT[src]}")
    return out


def zeff_source_record(impurity, line, *, sigma_zeff_imp=None, psi_N=None,
                       zeff_imp=None, zeff_line=None, sigma_zeff_line=None,
                       override=None, time=None, forced=None,
                       crosschecks=None):
    """The full record: candidates (with every test and state), tension,
    decision, override, thresholds, the list of every part that could not
    be evaluated, the list of what was evaluated under a stated
    simplification (``evaluated_without``: I-3 without fast ions when no
    fast-ion density was available), the findings and the warnings for the
    decided source.

    ``forced``: on an explicit route that takes a source regardless of the
    rule (``ni_source="CER"`` / ``"Zeff"`` / ``"all"``), a dict naming the
    route and its candidate(s); the decision then reports what the rule
    WOULD have decided.  ``crosschecks``: optional cross-checks (raw channel
    comparison, forward-modeled bremsstrahlung, radiated power, ...),
    recorded beside the decision and never used to select a source.
    """
    ten = tension(psi_N, zeff_imp, sigma_zeff_imp, zeff_line,
                  sigma_zeff_line) if psi_N is not None else None
    dec = decide(impurity, line, tension_rec=ten, override=override)
    warns = list(dec.pop("warnings"))
    not_eval, without = [], []
    for name, c in ((IMPURITY, impurity), (LINE, line)):
        if c is not None:
            not_eval.extend(f"{name} {p}" for p in c["not_evaluated"])
            without.extend(f"{name} {p}"
                           for p in c.get("evaluated_without", ()))
    used = dec["source"]
    if used is not None and dec["source_state"] == UNVERIFIED \
            and dec["override"] is None:
        c = impurity if used == IMPURITY else line
        warns.insert(0, "not evaluated: " + "; ".join(c["not_evaluated"]))
    if used is not None:
        c = impurity if used == IMPURITY else line
        warns.extend(c.get("findings", ()))
    return dict(
        rule="zeff_source_rule", schema_version=SCHEMA_VERSION,
        time=_f(time),
        candidates={IMPURITY: impurity if impurity is not None
                    else dict(present=False, state=ABSENT),
                    LINE: line if line is not None
                    else dict(present=False, state=ABSENT)},
        tension=ten, decision=dec, override=dec.get("override"),
        forced=forced, not_evaluated=not_eval, evaluated_without=without,
        warnings=warns,
        crosschecks=None if crosschecks is None else dict(crosschecks),
        thresholds=dict(THRESHOLDS))


# ---------------------------------------------------------------------------
#  messages
# ---------------------------------------------------------------------------
def checked_warning(record):
    """The one warning of a ``"checked"`` read, or ``None``: the decided
    source and the findings that concern it (unverified, tension, override,
    dilution, advisory), short."""
    if not record["warnings"]:
        return None
    dec = record["decision"]
    state = dec["source_state"]
    return (f"Z_eff from the {_SHORT[dec['source']]}, "
            + (state.upper() if state == UNVERIFIED else state) + ": "
            + "; ".join(record["warnings"]))


def refusal_message(record):
    """Why a ``"checked"`` read is refused (the decision's reason)."""
    return record["decision"]["reason"]


def forced_route_failures(record, used):
    """On a forced route: one phrase per candidate in ``used`` (the
    candidate names the route actually takes) that FAILS a recorded test.
    An unverified candidate is not reported (recorded only)."""
    out = []
    for name in used:
        c = record["candidates"].get(name) or {}
        if c.get("present") and c.get("state") == FAILED:
            out.append(f"the {_SHORT[name]} it takes fails "
                       f"{_failure_text(c)}")
    return out


# ---------------------------------------------------------------------------
#  time series
# ---------------------------------------------------------------------------
def series_source(records: Sequence[dict], source: Optional[str] = None):
    """One source for a whole time series, named by the user; failing and
    unverified slices flagged.

    ``records``: one rule record per slice (``IDAProfiles.zeff_rule`` or
    :func:`zeff_source_record`).  ``source`` (``IMPURITY`` or ``LINE``) is
    the user's name for the series' source; it is used on every slice, and
    the slices on which it fails its tests, or is unverified, are flagged
    (never switched).  Without a name the series is refused, with the
    per-slice outcomes listed so the user can name one: the rule never picks
    a series source itself, even when every slice agrees.

    Returns ``{"source", "named", "times", "per_slice", "failing_slices",
    "unverified_slices"}``.
    """
    per = []
    for r in records:
        c = r["candidates"]
        d = r["decision"]
        per.append(dict(time=r.get("time"), rule_source=d["rule_source"],
                        impurity_state=c[IMPURITY].get("state"),
                        line_state=c[LINE].get("state"),
                        impurity_failed=c[IMPURITY].get("failed"),
                        line_failed=c[LINE].get("failed")))
    if source is None:
        desc = "; ".join(
            f"{p['time']}: rule -> {p['rule_source']} (impurity "
            f"{p['impurity_state']}, line-integrated {p['line_state']})"
            for p in per)
        raise ZeffSourceRefused(
            "a time series needs one Z_eff source named by the user; name it "
            f"({IMPURITY!r} or {LINE!r}).  Per-slice outcomes of the rule: "
            f"{desc}.  The series is never switched between sources, and "
            "independent per-slice reads do not enforce this")
    if source not in (IMPURITY, LINE):
        raise ValueError(f"source must be {IMPURITY!r} or {LINE!r}")
    key = "impurity" if source == IMPURITY else "line"
    for p in per:
        if p[f"{key}_state"] in (None, ABSENT):
            raise ZeffSourceRefused(
                f"the {source} source is absent at slice {p['time']}")
    failing = [p["time"] for p in per if p[f"{key}_state"] == FAILED]
    unver = [p["time"] for p in per if p[f"{key}_state"] == UNVERIFIED]
    return dict(source=source, named=True,
                times=[p["time"] for p in per], per_slice=per,
                failing_slices=failing, unverified_slices=unver)
