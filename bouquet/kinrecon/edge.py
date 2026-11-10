"""Explicit edge P' taper, nonlinear-residual limit-cycle detector, and the ``auto`` policy.

Why this exists
---------------
A pressure-gradient input P'(psi_N) that is still large at the separatrix gives a
pressure-driven toroidal current density that steps from a finite value to zero at the
last closed flux surface.  On a discrete mesh the set of plasma nodes changes as the
boundary moves, and with such a step the Picard (fixed-point) iteration of a free-boundary
Grad-Shafranov solve can settle into a sustained cycle between neighboring node sets
instead of contracting.  The mechanism is inferred, not shown; the observation is that
zeroing P' over the outermost layer turns a period-3 residual cycle into monotone
convergence at unchanged solver settings, while slow solves are unaffected by it.

Ramping P' to zero over the outermost layer is an explicit, logged modeling choice about
the separatrix boundary condition -- never a change of tolerance, under-relaxation or
iteration count, and never silent.  Project decisions (stated in neutral words):

* the current-diffusion domain ends at psi_N = 0.995 (0.99 on coarse 2-D grids, never
  beyond 0.999); the region outside is a lumped outer layer.  The taper is confined to that
  layer: it has its own knot at the taper start, which must lie at or outside the domain
  edge, and P' ramps linearly to zero at psi_N = 1;
* setting ``off`` / ``auto`` / ``on`` (default ``auto``).  ``auto`` applies the taper only
  after the untapered solve was detected to be in a limit cycle, never to a solve that
  converges without it, and never to a solve that fails for another reason;
* the taper is applied uniformly within a case: if the reference solve of a case needed
  it, every later solve of that case (every draw) gets the same taper;
* every use is recorded (flag, taper width, change in pedestal and axis pressure, fraction
  of I_p and of stored energy affected), and "converged" and "converged with edge taper"
  are distinct statuses.

This module imports only numpy and the standard library and never imports a solver: the
policy helper :func:`solve_with_edge_taper` drives a caller-supplied solve callable.

Conventions
-----------
* ``psi_n`` is normalized poloidal flux, 0 on axis and 1 at the boundary; knots must start
  at exactly 0 and end at exactly 1 (TokaMaker's piecewise-linear profile convention).
* ``pprime`` is dp/dpsi in the caller's units, interpreted as the piecewise-linear
  interpolant of the knots (``linterp`` profiles).  All integrals below are exact for that
  interpolant (trapezoid rule on the knots, including the inserted one).
* ``psi_scale`` = dpsi/dpsi_N = psi_boundary - psi_axis in the same flux units as
  ``pprime``, so that dp/dpsi_N = ``pprime * psi_scale``.  The I_p bookkeeping uses
  dI = P' dV / (2 pi), which holds for psi in Wb/rad (TokaMaker; COCOS-1 g-files): pass
  ``psi_scale`` and ``pprime`` per radian.
* **Pressure convention: edge-pinned.**  The separatrix pressure p(psi_N = 1) is held fixed
  and the interior P' is kept bit-for-bit, so removing P' inside the layer shifts the
  pressure *everywhere inside the taper start by the same constant*
  ``dp_inside = psi_scale * int_start^1 (P'_orig - P'_taper) dpsi_N``
  (negative for a pressure that falls outward).  The alternative -- holding the axis
  pressure (or a solver's axis-to-edge pressure target) fixed -- would rescale the interior
  P' and so would not leave it untouched; it is reported only as ``pax_ratio``: a solver
  that sets the pressure scale from an axis-to-edge target must multiply that target by
  ``pax_ratio`` to reproduce the edge-pinned convention (interior P' unchanged).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

#: Diffusion-domain edge (project decision): default, coarse-grid value, upper limit.
DOMAIN_EDGE_DEFAULT = 0.995
DOMAIN_EDGE_COARSE = 0.99
DOMAIN_EDGE_MAX = 0.999
#: Default taper start = the default domain edge.
TAPER_START_DEFAULT = DOMAIN_EDGE_DEFAULT

#: Statuses reported by :func:`solve_with_edge_taper`.
STATUS_CONVERGED = "converged"
STATUS_CONVERGED_TAPER = "converged_with_edge_taper"
STATUS_FAILED = "failed"


# ============================================================== the taper itself
@dataclass(frozen=True)
class EdgeTaperSpec:
    """Where the taper starts and which domain edge it was checked against."""
    start: float = TAPER_START_DEFAULT
    domain_edge: float = DOMAIN_EDGE_DEFAULT

    def __post_init__(self):
        _check_start(self.start, self.domain_edge)

    def as_dict(self):
        return dict(start=float(self.start), width=float(1.0 - self.start), domain_edge=float(self.domain_edge))


def _check_start(start, domain_edge):
    if not (DOMAIN_EDGE_COARSE <= domain_edge <= DOMAIN_EDGE_MAX):
        raise ValueError(f"domain_edge {domain_edge!r} outside [{DOMAIN_EDGE_COARSE}, {DOMAIN_EDGE_MAX}] "
                         "(the diffusion domain ends at 0.995, 0.99 on coarse grids, never beyond 0.999)")
    if not (domain_edge <= start < 1.0):
        raise ValueError(f"taper start {start!r} must lie at or outside the domain edge {domain_edge!r} "
                         "and below psi_N = 1")


@dataclass
class TaperedProfile:
    """Result of :func:`apply_edge_taper`: the new knots and the bookkeeping record."""
    psi_n: np.ndarray
    pprime: np.ndarray
    pressure: Optional[np.ndarray]
    record: Dict[str, Any]


def _trap(y, x):
    """Exact integral of the piecewise-linear interpolant of (x, y)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _cum_from_edge(y, x):
    """int_{x_i}^{1} y dx at every knot (exact for the piecewise-linear interpolant)."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.zeros_like(x)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def apply_edge_taper(psi_n, pprime, pressure=None, *, start=None, domain_edge=None, psi_scale=None,
                     psi_pedestal_top=None, volume=None, ip=None):
    """Ramp P' linearly to zero between ``start`` and psi_N = 1; leave the interior untouched.

    Parameters
    ----------
    psi_n, pprime : array (n,)
        Knots (strictly increasing, ``psi_n[0] == 0``, ``psi_n[-1] == 1``) and P' there.
    pressure : array (n,), optional
        p on the same knots.  Returned shifted by the edge-pinned convention (module
        docstring); used to infer ``psi_scale`` when that is not given.
    start : float, optional
        Taper start in psi_N.  ``None`` means :data:`TAPER_START_DEFAULT` and the record
        says so.  Must satisfy ``domain_edge <= start < 1``.
    domain_edge : float, optional
        Diffusion-domain edge the start is checked against (``None``:
        :data:`DOMAIN_EDGE_DEFAULT`, recorded); allowed range 0.99--0.999.
    psi_scale : float, optional
        dpsi/dpsi_N (module docstring).  Needed for absolute pressure, I_p and energy numbers.
    psi_pedestal_top : float, optional
        psi_N of the pedestal top (<= start), for the pedestal-top change.  This module
        does not guess it; when absent the record says so (the absolute change is the same
        at every psi_N <= start in the edge-pinned convention).
    volume : array (n,), optional
        Enclosed volume V(psi_N) on the input knots (V[0] = 0), for the stored-energy and
        I_p bookkeeping; linearly interpolated at the inserted knot.
    ip : float, optional
        Plasma current, for the I_p fraction.

    Returns
    -------
    TaperedProfile
        ``psi_n`` = the input knots plus the taper-start knot (if it is not already a knot);
        ``pprime`` = the input values bit-for-bit at every knot below ``start``, the
        interpolated input value at ``start``, the linear ramp beyond, exactly 0 at 1;
        ``pressure`` (edge-pinned; ``None`` if not given); ``record`` (always complete:
        quantities that cannot be computed are ``None`` with the reason in ``notes``).
    """
    notes: List[str] = []
    start_source = "given"
    if start is None:
        start, start_source = TAPER_START_DEFAULT, "default (taper start = default domain edge)"
    edge_source = "given"
    if domain_edge is None:
        domain_edge, edge_source = DOMAIN_EDGE_DEFAULT, "default"
    start, domain_edge = float(start), float(domain_edge)
    _check_start(start, domain_edge)

    x = np.asarray(psi_n, dtype=float)
    y = np.asarray(pprime, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("psi_n and pprime must be 1-D arrays of the same length >= 2")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("psi_n and pprime must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("psi_n must be strictly increasing")
    if x[0] != 0.0 or x[-1] != 1.0:
        raise ValueError("psi_n must start at exactly 0 and end at exactly 1")
    p = None
    if pressure is not None:
        p = np.asarray(pressure, dtype=float)
        if p.shape != x.shape or not np.all(np.isfinite(p)):
            raise ValueError("pressure must be finite and on the psi_n knots")

    k = int(np.searchsorted(x, start, side="left"))     # knots 0..k-1 lie strictly inside start
    inserted = not (k < x.size and x[k] == start)
    if inserted:
        ys = y[k - 1] + (y[k] - y[k - 1]) * (start - x[k - 1]) / (x[k] - x[k - 1])
        xl = x[k:]                                       # layer knots beyond start (incl. 1)
        yl_orig = y[k:]
    else:
        ys = y[k]
        xl = x[k + 1:]
        yl_orig = y[k + 1:]
    ramp = ys * (1.0 - xl) / (1.0 - start)
    ramp[-1] = 0.0
    x_new = np.concatenate((x[:k], [start], xl))
    y_new = np.concatenate((y[:k], [ys], ramp))
    y_ref = np.concatenate((y[:k], [ys], yl_orig))       # input interpolant on the new knots

    i_orig = _trap(y_ref, x_new)
    i_tap = _trap(y_new, x_new)
    layer_removed = _trap((y_ref - y_new)[k:], x_new[k:])   # int_start^1 (P'_orig - P'_taper)
    ymax = float(np.max(np.abs(y))) if y.size else 0.0
    rec: Dict[str, Any] = dict(
        applied=True, convention="edge_pinned",
        start=start, start_source=start_source, width=1.0 - start,
        domain_edge=domain_edge, domain_edge_source=edge_source,
        knot_inserted=bool(inserted), n_knots_in=int(x.size), n_knots_out=int(x_new.size),
        n_layer_knots_replaced=int(xl.size), interior_knots_untouched=int(k),
        pprime_at_start=float(ys), pprime_edge_input=float(y[-1]),
        edge_over_max_abs_input=(float(abs(y[-1]) / ymax) if ymax > 0 else None),
        integral_pprime_dpsin_input=i_orig, integral_pprime_dpsin_tapered=i_tap,
        integral_removed_in_layer=layer_removed,
        pax_ratio=(i_tap / i_orig if i_orig != 0.0 else None),
        removed_fraction_of_axis_to_edge_drop=(layer_removed / i_orig if i_orig != 0.0 else None),
        psi_scale=None, psi_scale_source=None, pressure_consistency_rel=None,
        dp_inside=None, p_axis_before=None, dp_axis_rel=None, p_start_before=None, dp_start_rel=None,
        psi_pedestal_top=None, p_pedestal_top_before=None, dp_pedestal_top=None, dp_pedestal_top_rel=None,
        ip_removed=None, ip_fraction=None, stored_energy_before=None, stored_energy_after=None,
        stored_energy_change=None, stored_energy_rel=None, notes=notes)

    s = None
    if psi_scale is not None:
        s = float(psi_scale)
        rec.update(psi_scale=s, psi_scale_source="given")
    elif p is not None and i_orig != 0.0:
        s = float((p[-1] - p[0]) / i_orig)
        rec.update(psi_scale=s, psi_scale_source="inferred from pressure: (p(1) - p(0)) / int P' dpsi_N")
    else:
        notes.append("no psi_scale and no pressure: absolute pressure, I_p and energy changes not computed")

    p_new = None
    if s is not None:
        dp = s * layer_removed
        rec["dp_inside"] = dp
        c_orig = _cum_from_edge(y_ref, x_new)              # int_x^1 P'_orig on the new knots
        c_tap = _cum_from_edge(y_new, x_new)
        if p is not None:
            p_edge = float(p[-1])
            model = p_edge - s * _cum_from_edge(y, x)
            span = float(np.max(np.abs(p))) or 1.0
            rec["pressure_consistency_rel"] = float(np.max(np.abs(model - p)) / span)
            # input pressure on the new knots: the given values at every input knot; at an
            # inserted start knot, the value implied by the input P' integrated from the edge
            p_start_before = (p_edge - s * c_orig[k]) if inserted else float(p[k])
            p_ref = np.concatenate((p[:k], [p_start_before], p[k:] if inserted else p[k + 1:]))
            # edge-pinned: p_new(x) = p_ref(x) + psi_scale * int_x^1 (P'_orig - P'_taper)
            # (= p_ref + dp_inside at every knot <= start; exactly p_ref at psi_N = 1)
            p_new = p_ref + s * (c_orig - c_tap)
            p_new[:k] = p[:k] + dp
            p_new[-1] = p_edge
            rec.update(p_axis_before=float(p[0]), dp_axis_rel=(dp / p[0] if p[0] != 0 else None),
                       p_start_before=float(p_start_before),
                       dp_start_rel=(dp / p_start_before if p_start_before != 0 else None))
            if psi_pedestal_top is not None:
                xp = float(psi_pedestal_top)
                if not (0.0 <= xp <= start):
                    raise ValueError("psi_pedestal_top must lie in [0, start]")
                pp_top = float(np.interp(xp, x, p))
                rec.update(psi_pedestal_top=xp, p_pedestal_top_before=pp_top, dp_pedestal_top=dp,
                           dp_pedestal_top_rel=(dp / pp_top if pp_top != 0 else None))
            else:
                notes.append("pedestal-top location not supplied: dp_inside applies at every psi_N <= start "
                             "(edge-pinned convention); relative change given on axis and at the taper start")
        else:
            notes.append("no pressure given: dp_inside computed from psi_scale; relative changes need pressure")
        if volume is not None:
            v = np.asarray(volume, dtype=float)
            if v.shape != x.shape or not np.all(np.isfinite(v)):
                raise ValueError("volume must be finite and on the psi_n knots")
            v_new = np.interp(x_new, x, v)
            dv = np.diff(v_new[k:])
            dpp = (y_ref - y_new)[k:]
            ipr = s / (2.0 * np.pi) * float(np.sum(0.5 * (dpp[1:] + dpp[:-1]) * dv))
            rec["ip_removed"] = ipr
            if ip is not None and ip != 0:
                rec["ip_fraction"] = abs(ipr) / abs(float(ip))
            else:
                notes.append("ip not supplied: I_p fraction not computed")
            if p is not None:
                w0 = 1.5 * _trap(p_ref, v_new)
                w1 = 1.5 * _trap(p_new, v_new)
                rec.update(stored_energy_before=w0, stored_energy_after=w1, stored_energy_change=w1 - w0,
                           stored_energy_rel=((w1 - w0) / w0 if w0 != 0 else None))
            else:
                notes.append("no pressure: stored energy not computed")
        else:
            notes.append("volume not supplied: I_p and stored-energy fractions not computed")
    return TaperedProfile(psi_n=x_new, pprime=y_new, pressure=p_new, record=rec)


def no_taper_record(reason):
    """The record to log when no taper is applied (so the field is never silently absent)."""
    return dict(applied=False, reason=str(reason))


# ============================================================== setting
class EdgeTaperMode(str, Enum):
    """``off``: never taper.  ``auto`` (default): taper only after a detected limit cycle.
    ``on``: always taper."""
    OFF = "off"
    AUTO = "auto"
    ON = "on"


@dataclass(frozen=True)
class EdgeTaperSetting:
    """Mode plus the taper geometry used when the taper is applied."""
    mode: EdgeTaperMode = EdgeTaperMode.AUTO
    spec: EdgeTaperSpec = field(default_factory=EdgeTaperSpec)

    def __post_init__(self):
        object.__setattr__(self, "mode", EdgeTaperMode(self.mode))

    def as_dict(self):
        return dict(mode=self.mode.value, **self.spec.as_dict())


# ============================================================== limit-cycle detector
#: Parameters of :func:`classify_residual_history` (all in decades of the residual norm).
DETECTOR_DEFAULTS = dict(periods=(2, 3, 4, 5, 6), n_periods=3, min_amplitude=0.02, periodicity_frac=0.2,
                         amplitude_ratio_min=0.75, drift_frac=0.1, drift_per_iteration_max=0.005,
                         trend_window=12, fast_rate=0.05)


def classify_residual_history(residuals, tol=None, **criterion):
    """Classify a nonlinear-residual history (one norm per iteration).

    Work in l_k = log10(r_k).  A **limit cycle of period p** (p = 2..6, smallest first) is
    declared on the last ``n_periods * p`` iterations (needs ``n_periods * p + 1`` values)
    split into ``n_periods`` consecutive periods P_1..P_m, when ALL of:

    C0  amplitude: A = mean over periods of (max - min of l in the period) >= ``min_amplitude``
        (0.02 decades, about 5 %).  Smaller wobble is not distinguishable from stagnation,
        so it is reported as ``stalled``, never as a cycle.
    C1  periodicity: max_k |l_k - l_{k-p}| over the window <= ``periodicity_frac`` * A (0.2).
        A steady trend of slope s gives |l_k - l_{k-p}| = p|s| while A = (p-1)|s|, so a
        monotone solve, however slow, fails C1 by construction.
    C2  oscillation: every period contains at least one rise and one fall of l (the step
        into each period counts).
    C3  no amplitude decay: A(P_m) / A(P_1) >= ``amplitude_ratio_min`` (0.75).  A damped
        oscillation fails here.
    C4  no net progress: the per-period drift of the period means, D = (mean P_m - mean P_1)
        / (m - 1), satisfies |D| <= ``drift_frac`` * A (0.1) and |D| / p <=
        ``drift_per_iteration_max`` (0.005 decades per iteration).  At that rate, closing
        even one decade takes more than 200 iterations, i.e. the iteration is not
        contracting on any practical budget; a faster drift means an oscillating but
        contracting (or diverging) solve, which is not a cycle.

    Reasoning: the failure the taper addresses is a sustained, repeating, non-decaying
    oscillation, against which more iterations do nothing.  C1 + C2 require repetition with
    real ups and downs, C3 + C4 require that it is not dying out or drifting.  A slow solve
    -- monotone, or oscillating while still making net progress -- fails C1, C3 or C4 and is
    classified by its trend instead.

    Otherwise, with slope s of a least-squares line through the last ``trend_window`` (12)
    values of l and the fraction f of falling steps there:

    * ``converged``             -- ``tol`` given and r_last < tol;
    * ``monotone_contraction``  -- s <= -``fast_rate`` (0.05 decades/iteration, a contraction
      factor <= 0.89 per iteration) and f >= 0.9 (ran out of iterations while contracting fast);
    * ``decaying_oscillation``  -- s <= -``fast_rate`` and f < 0.9;
    * ``slow_contraction``      -- -``fast_rate`` < s <= -``drift_per_iteration_max``;
    * ``stalled``               -- |s| < ``drift_per_iteration_max`` and no cycle;
    * ``diverging``             -- s >= ``drift_per_iteration_max``, or a non-finite residual;
    * ``insufficient_history``  -- fewer than ``n_periods`` * 2 + 1 values (7 by default: the
      shortest cycle test needs that many).

    The trend fit uses up to the last 12 values.  The detector judges only the residual: a
    solve whose state alternates (e.g. the magnetic axis jumping between two locations) while
    its residual stays flat to within ``min_amplitude`` is ``stalled``, not ``limit_cycle``.

    Returns a dict with ``verdict``, ``period`` (cycle only), the measured quantities and
    the criterion used.
    """
    crit = dict(DETECTOR_DEFAULTS)
    unknown = set(criterion) - set(crit)
    if unknown:
        raise TypeError(f"unknown detector parameters {sorted(unknown)}")
    crit.update(criterion)
    r = np.asarray(residuals, dtype=float).ravel()
    out: Dict[str, Any] = dict(verdict=None, period=None, n=int(r.size), criterion=_jsonable(crit),
                               tol=None if tol is None else float(tol))
    if r.size and not np.all(np.isfinite(r)):
        out.update(verdict="diverging", reason="non-finite residual")
        return out
    if r.size and np.any(r <= 0.0):
        out.update(verdict="converged" if tol is not None and r[-1] < tol else "insufficient_history",
                   reason="non-positive residual (exact zero)")
        return out
    if r.size:
        out.update(first=float(r[0]), last=float(r[-1]), min=float(np.min(r)))
    if tol is not None and r.size and r[-1] < tol:
        out["verdict"] = "converged"
        return out
    m = int(crit["n_periods"])
    if r.size < m * min(crit["periods"]) + 1:
        out["verdict"] = "insufficient_history"
        return out
    lg = np.log10(r)
    tested = {}
    for p in crit["periods"]:
        if lg.size < m * p + 1:
            break
        w = lg[-m * p:]
        d = np.diff(lg[-m * p - 1:])                      # step into each window value
        per = w.reshape(m, p)
        amps = per.max(axis=1) - per.min(axis=1)
        A = float(np.mean(amps))
        lag = float(np.max(np.abs(w[p:] - w[:-p])))
        osc = all(np.any(d[i * p:(i + 1) * p] > 0) and np.any(d[i * p:(i + 1) * p] < 0) for i in range(m))
        amp_ratio = float(amps[-1] / amps[0]) if amps[0] > 0 else (np.inf if amps[-1] > 0 else 0.0)
        means = per.mean(axis=1)
        drift = float((means[-1] - means[0]) / (m - 1))
        t = dict(amplitude=A, periodicity_ratio=(lag / A if A > 0 else None), oscillating=bool(osc),
                 amplitude_ratio=amp_ratio, drift_per_period=drift)
        ok = (A >= crit["min_amplitude"] and lag <= crit["periodicity_frac"] * A and osc
              and amp_ratio >= crit["amplitude_ratio_min"] and abs(drift) <= crit["drift_frac"] * A
              and abs(drift) / p <= crit["drift_per_iteration_max"])
        t["passes"] = bool(ok)
        tested[p] = t
        if ok:
            out.update(verdict="limit_cycle", period=int(p), amplitude_decades=A,
                       periodicity_ratio=t["periodicity_ratio"], amplitude_ratio=amp_ratio,
                       drift_per_period=drift, tail=[float(v) for v in r[-m * p:]])
            out["cycle_tests"] = _jsonable(tested)
            return out
    out["cycle_tests"] = _jsonable(tested)
    nw = int(min(crit["trend_window"], lg.size))
    tw = lg[-nw:]
    s = float(np.polyfit(np.arange(nw, dtype=float), tw, 1)[0])
    falls = float(np.mean(np.diff(tw) < 0))
    out.update(trend_decades_per_iteration=s, falling_fraction=falls)
    if tol is not None and s < 0:
        out["projected_iterations_to_tol"] = float((lg[-1] - np.log10(tol)) / (-s))
    if s <= -crit["fast_rate"]:
        out["verdict"] = "monotone_contraction" if falls >= 0.9 else "decaying_oscillation"
    elif s <= -crit["drift_per_iteration_max"]:
        out["verdict"] = "slow_contraction"
    elif s < crit["drift_per_iteration_max"]:
        out["verdict"] = "stalled"
    else:
        out["verdict"] = "diverging"
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def summarize_history(residuals, n_tail=10):
    """Small JSON-safe summary of a residual history (for ledgers)."""
    r = np.asarray(residuals, dtype=float).ravel()
    if not r.size:
        return dict(n=0)
    t = r[-n_tail:]
    return dict(n=int(r.size), first=float(r[0]), min=float(np.min(r)), last=float(r[-1]),
                tail_min=float(np.min(t)), tail_max=float(np.max(t)),
                n_increases=int(np.sum(np.diff(r) > 0)), tail=[float(v) for v in t])


# ============================================================== auto policy
@dataclass
class SolveAttempt:
    """What a solve callable returns: the solver's own convergence flag, the residual
    history (one norm per iteration; may be empty if unavailable), and any payload."""
    converged: bool
    residuals: Any = ()
    payload: Any = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyOutcome:
    """``record`` is JSON-safe (ledger); ``payload`` is that of the attempt whose state is final."""
    record: Dict[str, Any]
    payload: Any = None


@dataclass
class EdgeTaperCase:
    """Case-level record that keeps the taper uniform within a case.

    The first solve run through :func:`solve_with_edge_taper` with this case is the case's
    reference.  Its outcome locks the case: if the reference converged with the taper,
    every later solve receives exactly that taper (no untapered attempt is made); if the
    reference converged without it, later solves run untapered and a later limit cycle is
    reported as ``failed`` rather than tapered, so that one case never mixes tapered and
    untapered equilibria.  A failed reference leaves the case undecided.
    """
    case_id: str
    setting: EdgeTaperSetting = field(default_factory=EdgeTaperSetting)
    locked: Optional[bool] = None            # None undecided, True tapered, False untapered
    spec: Optional[EdgeTaperSpec] = None
    decided_by: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def taper_for_later_solves(self):
        """The taper every later solve of this case must use (``None`` = untapered or undecided)."""
        return self.spec if self.locked else None

    def as_dict(self):
        return dict(case_id=self.case_id, setting=self.setting.as_dict(), locked=self.locked,
                    taper=self.spec.as_dict() if self.spec else None, decided_by=self.decided_by,
                    solves=list(self.history))


def _attempt(solve, taper, label, tol, detector_kw):
    a = solve(taper)
    if isinstance(a, Mapping):
        a = SolveAttempt(converged=bool(a.get("converged")), residuals=a.get("residuals", ()),
                         payload=a.get("payload"), info=dict(a.get("info", {})))
    res = np.asarray(a.residuals if a.residuals is not None else (), dtype=float).ravel()
    # the detector is consulted only for a solve the solver did not flag as converged; ``tol``
    # only feeds its projected-iterations estimate (verdict 'converged' cannot occur here
    # unless the history disagrees with the solver's flag, which is then visible in the record)
    det = classify_residual_history(res, tol=tol, **detector_kw) if not a.converged else None
    rec = dict(label=label, taper=None if taper is None else taper.as_dict(), converged=bool(a.converged),
               history=summarize_history(res), detector=det, info=_jsonable(a.info))
    return a, rec


def solve_with_edge_taper(solve: Callable[[Optional[EdgeTaperSpec]], Any], setting=None, *, case=None,
                          label="solve", tol=None, detector_kw=None):
    """Run ``solve`` under the edge-taper policy.

    ``solve(taper)`` is called with ``None`` (untapered) or an :class:`EdgeTaperSpec`; it must
    build its profiles with :func:`apply_edge_taper` when given a spec, solve at the solver's
    own settings, and return a :class:`SolveAttempt` (or a mapping with ``converged``,
    ``residuals``, ``payload``, ``info``).  ``tol`` is recorded only; convergence is the
    solver's own flag.

    * ``off``:  one untapered solve -> ``converged`` or ``failed``.
    * ``on``:   one tapered solve -> ``converged_with_edge_taper`` or ``failed``.
    * ``auto``: untapered solve; converged -> ``converged``.  Not converged and the detector
      says ``limit_cycle`` -> tapered re-solve -> ``converged_with_edge_taper`` or ``failed``
      (both histories kept).  Any other verdict -> ``failed``, taper NOT applied.
    * With ``case`` locked by its reference, the lock overrides: tapered case -> one tapered
      solve; untapered case -> one untapered solve, a limit cycle is reported, not tapered.
    """
    if setting is not None and not isinstance(setting, EdgeTaperSetting):
        setting = EdgeTaperSetting(EdgeTaperMode(setting))
    if case is not None:
        if setting is not None and setting != case.setting:
            raise ValueError("setting differs from the case's setting; a case uses one setting")
        setting = case.setting
    setting = setting or EdgeTaperSetting()
    dkw = dict(detector_kw or {})
    mode = setting.mode
    rec: Dict[str, Any] = dict(label=label, setting=setting.as_dict(), mode=mode.value, attempts=[],
                               taper_applied=False, status=None, reason=None, case_lock=None,
                               tol_recorded=None if tol is None else float(tol))
    final = None
    if case is not None and case.locked is not None:
        rec["case_lock"] = dict(case_id=case.case_id, locked=case.locked, decided_by=case.decided_by)
        taper = case.spec if case.locked else None
        a, ar = _attempt(solve, taper, label + (":tapered(case lock)" if taper else ":untapered(case lock)"),
                         tol, dkw)
        rec["attempts"].append(ar)
        final = a
        rec["taper_applied"] = taper is not None
        if a.converged:
            rec["status"] = STATUS_CONVERGED_TAPER if taper else STATUS_CONVERGED
            rec["reason"] = "case locked by its reference solve"
        else:
            rec["status"] = STATUS_FAILED
            v = (ar["detector"] or {}).get("verdict")
            rec["reason"] = (f"case locked {'tapered' if taper else 'untapered'} by its reference; solve not "
                             f"converged (detector: {v}); the lock is not changed by a later solve")
    elif mode is EdgeTaperMode.ON:
        a, ar = _attempt(solve, setting.spec, label + ":tapered", tol, dkw)
        rec["attempts"].append(ar)
        final, rec["taper_applied"] = a, True
        rec["status"] = STATUS_CONVERGED_TAPER if a.converged else STATUS_FAILED
        rec["reason"] = "mode on: taper always applied"
    else:
        a, ar = _attempt(solve, None, label + ":untapered", tol, dkw)
        rec["attempts"].append(ar)
        final = a
        if a.converged:
            rec["status"], rec["reason"] = STATUS_CONVERGED, "converged without the taper"
        elif mode is EdgeTaperMode.OFF:
            rec["status"], rec["reason"] = STATUS_FAILED, "mode off: not converged, no taper"
        else:
            v = ar["detector"]["verdict"]
            if v == "limit_cycle":
                b, br = _attempt(solve, setting.spec, label + ":tapered", tol, dkw)
                rec["attempts"].append(br)
                final, rec["taper_applied"] = b, True
                rec["status"] = STATUS_CONVERGED_TAPER if b.converged else STATUS_FAILED
                rec["reason"] = (f"untapered solve in a limit cycle (period {ar['detector']['period']}); "
                                 + ("tapered re-solve converged" if b.converged else
                                    "tapered re-solve did not converge either (both histories kept)"))
            else:
                rec["status"] = STATUS_FAILED
                rec["reason"] = f"not converged and no limit cycle (detector: {v}); taper not applied"
    rec["detector_verdict_untapered"] = next((x["detector"]["verdict"] for x in rec["attempts"]
                                              if x["taper"] is None and x["detector"]), None)
    if case is not None:
        if case.locked is None and rec["status"] != STATUS_FAILED:
            case.locked = bool(rec["taper_applied"])
            case.spec = setting.spec if rec["taper_applied"] else None
            case.decided_by = label
            rec["case_lock"] = dict(case_id=case.case_id, locked=case.locked, decided_by=label, set_now=True)
        case.history.append(dict(label=label, status=rec["status"], taper_applied=rec["taper_applied"]))
    return PolicyOutcome(record=_jsonable(rec), payload=final.payload if final is not None else None)
