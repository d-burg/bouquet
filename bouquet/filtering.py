"""Postprocessing filters for a bouquet HDF5 database.

These routines operate on the stored equilibria *after* a run, so the
user can see the distribution of in-spec / out-of-spec draws before
deciding what to cut.  They are **non-destructive**: each filter writes
per-draw boolean flags into the H5 (``passes_coil_filter``,
``passes_boundary_filter``; the opt-in bands ``passes_l_i_band`` /
``passes_beta_p_band`` / ``passes_q0_gate`` and their AND
``passes_draw_band``) and a derived ``selected`` flag (the AND of all
applied filters), but never deletes data.

**The in-spec rule** (owner decision 2026-10-14): a draw is in spec when it
passes the configured coil test and the boundary cut -- the magnetics.  The
l_i band (``GenerationConfig.l_i_tolerance`` / ``l_i_sigma``) and the beta_p
band (``beta_p_target`` / ``beta_p_sigma``) are opt-in, for a source that
states a measurement uncertainty of them; the rule in force is stamped on
the scan (``inspec_rule``) and :func:`refilter_inspec` re-judges an archive
under another one.  Re-running a filter with
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
import warnings

import numpy as np
import h5py
from scipy.spatial import cKDTree as _cKDTree

from .schema import IFILE_DS, find_bytes_dataset
from .utils import (
    list_equilibrium_indices,
    discover_scan_keys,
    read_eqdsk_from_bytes,
    _scan_key,
    _group_path,
    _read_coil_names,
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
    "boundary_deviation_mm",
    "passes_coil_spec",
    "passes_coil_chi2",
    "passes_boundary_spec",
    "passes_all_filters",
    "make_coil_predicate",
    "until_n_delivered",
    "InspecBands",
    "resolve_inspec_bands",
    "refilter_inspec",
    "core_current_metrics",
    "read_inspec_rule",
    "read_inspec_record",
    "in_spec_fields",
    "shafranov_lambda",
    "LAMBDA_DEFINITION",
]

#: Per-draw pass flags ANDed into ``selected``.  ``passes_draw_band`` is
#: the AND of the OPT-IN bands of the in-spec rule (:class:`InspecBands`:
#: an l_i band, a beta_p band, the constrain_sawteeth q0 gate; all off by
#: default, owner decision 2026-10-14), written at archive time on draws
#: made by the unified engine (True when no band is on) and by
#: :func:`refilter_inspec`; a legacy draw never carries it from generation,
#: so its ``selected`` is exactly the coil + boundary verdict as before.
_FILTER_FLAGS = ("passes_coil_filter", "passes_boundary_filter",
                 "passes_draw_band")
#: The opt-in bands' own per-draw verdicts (their AND is
#: ``passes_draw_band``): reported by :func:`read_filter_flags`, not ANDed
#: into ``selected`` a second time.
_BAND_FLAGS = ("passes_l_i_band", "passes_beta_p_band", "passes_q0_gate",
               "passes_jphi_band", "passes_core_gradient")


class _Polyline:
    """A closed polyline's segments ``a[i] -> a[i+1]``, and a KD-tree of samples along them at most
    ``s`` apart: the larger of the median and the mean segment length, so a few long segments among
    dense points add few samples."""

    def __init__(self, contour):
        a = np.asarray(contour, dtype=float)
        ab = np.roll(a, -1, axis=0) - a
        seg_len = np.linalg.norm(ab, axis=1)
        # rows ax, ay, abx, aby, 1/|ab|^2: one gather per row in distance()
        self.segments = np.vstack([a.T, ab.T, 1.0 / np.maximum(seg_len ** 2, 1e-300)])
        total = seg_len.sum()
        self.s = max(np.median(seg_len[seg_len > 0]), total / len(a)) if total > 0 else 1.0
        n_sub = np.maximum(np.ceil(seg_len / self.s).astype(int), 1)
        self.owner = np.repeat(np.arange(len(a)), n_sub)    # the segment each sample lies on
        frac = (np.arange(len(self.owner)) - np.repeat(np.cumsum(n_sub) - n_sub, n_sub) + 0.5) / n_sub[self.owner]
        self.tree = _cKDTree(a[self.owner] + frac[:, None] * ab[self.owner])

    def distance(self, p, j):
        """Distance from points ``p`` (m, 2) to segments ``j`` (m,) or (m, k)."""
        px, py = (p[:, c].reshape((-1,) + (1,) * (j.ndim - 1)) for c in (0, 1))
        ax, ay, bx, by, inv = self.segments[:, j]
        dx, dy = px - ax, py - ay
        t = np.clip((dx * bx + dy * by) * inv, 0.0, 1.0)
        return np.hypot(dx - t * bx, dy - t * by)


def contour_distance(pts, contour, k=8):
    """Distance from each of ``pts`` to the closed polyline through ``contour``, exactly.

    Measured to the segments, not the vertices: OFT's cut-cell ``trace_surf`` spaces its points ~5 mm
    apart, which a nearest-vertex distance would read as boundary deviation.  Each point is measured to
    the segments of its ``k`` nearest samples (:class:`_Polyline`, spacing ``s``).  Any other segment
    has every sample at least the k-th sample's distance ``r_k`` away, so it is at least ``r_k - s/2``
    away; the few points nearer than that bound are re-measured against every segment with a sample
    within their distance + ``s/2``."""
    pts = np.asarray(pts, dtype=float)
    line = _Polyline(contour)
    k = min(k, line.tree.n)
    r, idx = line.tree.query(pts, k=k)
    r, idx = np.reshape(r, (len(pts), k)), np.reshape(idx, (len(pts), k))
    d = line.distance(pts, line.owner[idx]).min(axis=1)
    unsure = np.flatnonzero(d > r[:, -1] - 0.5 * line.s) if k < line.tree.n else []
    if len(unsure):
        near = line.tree.query_ball_point(pts[unsure], (d[unsure] + 0.5 * line.s) * (1 + 1e-9))
        n_near = np.fromiter(map(len, near), dtype=int, count=len(unsure))
        dn = line.distance(pts[np.repeat(unsure, n_near)], line.owner[np.concatenate(near).astype(int)])
        d[unsure] = np.minimum.reduceat(dn, np.cumsum(n_near) - n_near)
    return d


# --------------------------------------------------------------------------
#  the selection predicate -- ONE definition, two call sites
# --------------------------------------------------------------------------
#  These three are the whole of what "in spec" means by default (the
#  magnetics: the coil test and the boundary cut; owner decision 2026-10-14
#  -- the opt-in bands of InspecBands, below, are the only additions). They
#  are used BOTH by the postprocess filters below AND, in-loop, by
#  generate_bouquet's until-N-in-spec stopping rule
#  (GenerationConfig.n_inspec_target).
#
#  Keeping one definition is not tidiness -- it is the guarantee that a run
#  which stops after counting N in-spec draws still shows N selected draws
#  once .filter() runs over the archive. Two independent implementations
#  would drift: the in-loop [bnd-diag] print, for instance, queries the
#  BASELINE tree with the PERTURBED points, which is the opposite direction
#  from boundary_deviation_mm below and gives a different max (and a
#  slightly different RMS) on the same pair of contours.
def boundary_deviation_mm(baseline_boundary, perturbed_boundary):
    """``(rms_mm, max_mm)`` of a perturbed LCFS vs the baseline contour.

    Distance **from each baseline point to the perturbed contour**
    (:func:`contour_distance`) -- the direction ``plot_traces`` and the
    boundary filter both report. Returns ``(nan, nan)`` when either contour is missing or too
    short to build a tree from (fewer than 2 points), which the callers
    treat as "no verdict" rather than as a pass.
    """
    if baseline_boundary is None or perturbed_boundary is None:
        return (np.nan, np.nan)
    bl = np.asarray(baseline_boundary, dtype=float)
    pt = np.asarray(perturbed_boundary, dtype=float)
    if bl.ndim != 2 or pt.ndim != 2 or len(bl) < 2 or len(pt) < 2:
        return (np.nan, np.nan)
    try:
        devs = contour_distance(bl, pt)
        return (float(np.sqrt(np.mean(devs ** 2)) * 1e3),
                float(np.max(devs) * 1e3))
    except Exception:
        return (np.nan, np.nan)


def passes_coil_spec(F_pct, V_pct, F_max_pct, V_max_pct):
    """Coil-current spec test, in PERCENT on both sides.

    A non-finite drift or a non-finite threshold fails: an unmeasured draw is
    not silently in spec.
    """
    ok_F = (np.isfinite(F_pct) and np.isfinite(F_max_pct)
            and F_pct <= F_max_pct)
    ok_V = (np.isfinite(V_pct) and np.isfinite(V_max_pct)
            and V_pct <= V_max_pct)
    return bool(ok_F and ok_V)


def passes_coil_chi2(chi2_nu, max_abs_z, chi2_max, z_max=None):
    """Measurement-referenced coil test: ``chi2/nu <= chi2_max`` AND worst
    single-coil ``|z| <= z_max`` (``z_max=None`` disables the second guard).

    The chi2 sibling of :func:`passes_coil_spec`, and the SAME expression
    :func:`filter_coil_chi2` cuts on -- it is called there, not re-written, so
    the until-N loop and the postprocess filter cannot drift apart. An
    unjudgeable draw scores NaN and FAILS: unjudgeable is not a pass.
    """
    ok = bool(np.isfinite(chi2_nu) and np.isfinite(chi2_max)
              and chi2_nu <= chi2_max)
    if ok and z_max is not None:
        ok = bool(np.isfinite(max_abs_z) and max_abs_z <= z_max)
    return ok


def passes_boundary_spec(rms_mm, max_mm, rms_max_mm=None, max_max_mm=None):
    """LCFS-deviation test against whichever bounds are supplied.

    A bound left at ``None`` is not applied (that channel passes), so with
    both ``None`` this is the report-only mode of :func:`filter_boundaries`
    and every draw passes. A supplied bound against a non-finite deviation
    fails, for the same reason as in :func:`passes_coil_spec`.
    """
    ok_rms = (rms_max_mm is None) or (np.isfinite(rms_mm)
                                      and rms_mm <= rms_max_mm)
    ok_max = (max_max_mm is None) or (np.isfinite(max_mm)
                                      and max_mm <= max_max_mm)
    return bool(ok_rms and ok_max)


def passes_all_filters(F_pct=np.nan, V_pct=np.nan, baseline_boundary=None,
                       perturbed_boundary=None,
                       F_max_pct=np.nan, V_max_pct=np.nan,
                       rms_max_mm=None, max_max_mm=None, coil_ok=None):
    """The full ``selected`` verdict for one draw, from raw quantities.

    Composes the coil verdict and :func:`passes_boundary_spec` over
    :func:`boundary_deviation_mm` -- the same AND that ``_recompute_selected``
    forms from the two stored flags. This is what ``generate_bouquet``'s
    until-N loop counts with, so that a run stopping at N in-spec draws
    yields N ``selected`` draws once the postprocess filters run.

    ``coil_ok`` is the verdict of the CONFIGURED coil filter, already formed
    against the same thresholds :meth:`Bouquet.filter` cuts on (in practice by
    a predicate from :func:`make_coil_predicate`, chi2 or legacy). Left at
    ``None`` the LEGACY percent test is applied to ``F_pct``/``V_pct`` here --
    which is what ``.filter()`` applies when ``filtering.coil_filter`` is
    ``'legacy'``, and what direct callers of the old signature get.

    Returns ``(passed, rms_mm, max_mm, reasons)``, where ``reasons`` is the
    tuple of failing channels (empty on a pass) -- ``('coil',)``,
    ``('boundary',)`` or both -- for the per-draw log line.
    """
    rms_mm, max_mm = boundary_deviation_mm(baseline_boundary,
                                           perturbed_boundary)
    ok_coil = (passes_coil_spec(F_pct, V_pct, F_max_pct, V_max_pct)
               if coil_ok is None else bool(coil_ok))
    ok_bnd = passes_boundary_spec(rms_mm, max_mm, rms_max_mm, max_max_mm)
    reasons = tuple(([] if ok_coil else ["coil"])
                    + ([] if ok_bnd else ["boundary"]))
    return bool(ok_coil and ok_bnd), rms_mm, max_mm, reasons


# --------------------------------------------------------------------------
#  the CONFIGURED coil predicate -- one builder, two call sites
# --------------------------------------------------------------------------
#  Bouquet.filter() cuts the archive with filter_coil_chi2 (or the legacy
#  rule); generate_bouquet's until-N loop has to reach the SAME verdict per
#  draw, in flight, from the coil-current vectors it already holds. This
#  builder is the single place that decides which rule applies, with which
#  sigma and which acceptance numbers, so the loop cannot cut at numbers the
#  postprocess disagrees with.
def _coil_fallback_message(exc, inspec_F_max, inspec_VSC_max):
    """The ONE wording for 'the chi2 coil filter could not be resolved'.

    Shared by :meth:`Bouquet.filter` and :func:`make_coil_predicate` so the
    loop falls back exactly when, and says exactly what, the postprocess does.
    """
    return (
        "COIL FILTER FALLBACK: chi2 coil filter disabled -- " + str(exc) +
        f" Using the legacy rule (|dI/I| <= {inspec_F_max:.0%} F-coils, "
        f"{inspec_VSC_max:.0%} VSC), which is NOT a measurement-referenced "
        "criterion: it is a flat fractional band, so it is many sigma on a "
        "high-current coil and a fraction of one on a low-current coil, and it "
        "rejects a large and state-dependent share of an L-mode ensemble.")


def make_coil_predicate(coil_filter="chi2", baseline_currents=None,
                        inspec_F_max=0.02, inspec_VSC_max=0.02,
                        sigma=None, device=None, era=None,
                        chi2_max=None, z_max=None):
    """Build the coil half of the until-N stopping rule.

    Returns ``(predicate, kind, model)``.  ``predicate(diagnostics,
    draw_currents)`` -> ``(ok, info)``, where *info* carries the per-draw
    numbers worth logging/archiving (``chi2_nu`` / ``max_abs_z`` /
    ``coil_nu`` / ``worst_coil`` on the chi2 path, the drift percentages on
    the legacy one).  ``kind`` is ``"chi2"``, ``"legacy"`` or
    ``"legacy(fallback)"``; ``model`` is the resolved sigma/acceptance
    provenance record (``None`` on the legacy path).

    Both paths mirror :meth:`Bouquet.filter` exactly:

    * ``coil_filter="chi2"`` resolves the per-coil sigma ONCE for the run
      (:func:`coil_spec.resolve_coil_sigma`, with the same ``sigma`` /
      ``device`` / ``era`` the postprocess is given) and takes the acceptance
      thresholds from :func:`coil_spec.resolve_coil_acceptance` -- the same
      function :func:`filter_coil_chi2` calls, so ``chi2_max`` / ``z_max``
      and the sigma floors are identical on both sides;
    * a :class:`coil_spec.CoilSigmaUnavailable` (unregistered mesh, an
      archive/baseline with no coil names) falls back to the legacy predicate
      LOUDLY, with the same warning text ``.filter()`` emits, because that is
      precisely when ``.filter()`` falls back;
    * a draw with no coil currents is unjudgeable and FAILS -- never skipped,
      never a pass.

    ``inspec_F_max`` / ``inspec_VSC_max`` are FRACTIONS (as in
    :class:`FilterConfig`); the percent conversion happens here.
    """
    import warnings
    from .coil_spec import (CoilSigmaUnavailable, coil_chi2,
                            resolve_coil_acceptance, resolve_coil_sigma)

    def _legacy(diagnostics, draw_currents=None):
        F = float((diagnostics or {}).get("max_F_drift_pct", np.nan))
        V = float((diagnostics or {}).get("max_VSC_drift_pct", np.nan))
        ok = passes_coil_spec(F, V, float(inspec_F_max) * 100.0,
                              float(inspec_VSC_max) * 100.0)
        return ok, {"max_F_drift_pct": F, "max_VSC_drift_pct": V}

    if coil_filter == "legacy":
        return _legacy, "legacy", None
    if coil_filter != "chi2":
        raise ValueError("coil_filter must be 'chi2' or 'legacy'")

    try:
        if not baseline_currents:
            raise CoilSigmaUnavailable(
                "the baseline carries no coil currents (coil storage disabled, "
                "or a solver state without coils), so no draw can be judged "
                "against it.")
        base = {str(k): float(v) for k, v in baseline_currents.items()}
        sig, model = resolve_coil_sigma(base, sigma=sigma, device=device,
                                        era=era)
        cm, zm, acc_src, cal_nu = resolve_coil_acceptance(
            model, chi2_max=chi2_max, z_max=z_max)
    except CoilSigmaUnavailable as exc:
        warnings.warn(_coil_fallback_message(exc, inspec_F_max, inspec_VSC_max),
                      stacklevel=2)
        return _legacy, "legacy(fallback)", None

    def _chi2(diagnostics, draw_currents=None):
        if not draw_currents:
            r = {"chi2_nu": float("nan"), "max_abs_z": float("nan"), "nu": 0,
                 "worst_coil": None}
        else:
            r = coil_chi2({str(k): float(v) for k, v in draw_currents.items()},
                          base, sig)
        ok = passes_coil_chi2(r["chi2_nu"], r["max_abs_z"], cm, zm)
        return ok, {"chi2_nu": float(r["chi2_nu"]),
                    "max_abs_z": float(r["max_abs_z"]),
                    "coil_nu": int(r["nu"]), "worst_coil": r["worst_coil"]}

    model = dict(model, acceptance={"chi2_max": cm, "z_max": zm,
                                    "source": acc_src,
                                    "calibrated_nu": cal_nu,
                                    "nu_sigma": len(sig)})
    return _chi2, "chi2", model


def until_n_delivered(diagnostics_list):
    """In-spec draws delivered by an until-N run, from its diagnostics.

    Counts the per-draw ``until_n_inspec`` verdicts ``generate_bouquet``
    stores at accounting time.  This is how ``Bouquet.generate`` re-derives
    the outcome OUTSIDE the quiet-mode output capture -- the in-loop prints
    and the cap-missed warning land in ``generation_log`` on the default
    path, which is not a place a failure signal may live alone.
    """
    return sum(bool(d.get("until_n_inspec", False))
               for d in (diagnostics_list or []))


# --------------------------------------------------------------------------
#  the in-spec rule: the magnetics, plus OPT-IN measured bands
# --------------------------------------------------------------------------
#  Owner decision, 2026-10-14.  The external magnetics constrain I_p, the
#  boundary and the Shafranov combination beta_p + l_i/2; bouquet applies
#  that constraint through the free-boundary draw solve, the coil chi-squared
#  test and the boundary-deviation cut.  l_i(3) alone is an equilibrium
#  inference, not a magnetics measurement, so the l_i band that used to be
#  applied to every engine draw (l_i_tolerance = 0.05) is no longer part of
#  the default: a draw is IN SPEC when it passes the configured coil test and
#  the boundary cut (and was not rejected before archiving).  The l_i band
#  and a beta_p band are opt-in, for a source that STATES a measurement
#  uncertainty of them (an MSE-constrained fit, a diamagnetic loop).
#
#  Lambda = beta_p + l_i(3)/2 is computed and stored for every draw and for
#  the reconstruction, and reported in the band tables; it never filters.

#: Scan-group attribute: the in-spec rule the archive's band flags were
#: judged under (:meth:`InspecBands.rule`): ``"magnetics"`` (the default),
#: ``"magnetics+li"``, ``"magnetics+beta_p"``, ``"magnetics+li+beta_p"``,
#: with ``"+q0"`` when the ``constrain_sawteeth`` gate is on.
INSPEC_RULE_ATTR = "inspec_rule"
#: Scan-group attribute: the JSON record of the band settings
#: (:meth:`InspecBands.record`), so a later re-filter knows what was applied.
INSPEC_BANDS_ATTR = "inspec_bands_json"
#: Key of the :data:`INSPEC_BANDS_ATTR` record (and of the baseline engine
#: record's ``draws``): the q0 gate as ``generate_bouquet`` APPLIED it --
#: ``configured`` (``constrain_sawteeth``), ``applied``, the reconstruction's
#: ``q0`` and the ``reason`` when it was switched off (a sawtoothing
#: reconstruction, q0 < 1: every draw would fail the gate).  The stamped
#: rule is the applied one (no ``"+q0"`` when it was switched off).
Q0_GATE_RECORD_KEY = "q0_gate"
#: Per-draw attribute saying what the archived ``in_spec`` is.
#: :data:`IN_SPEC_RULE_SOURCE`: the in-spec rule's verdict -- the configured
#: coil test (chi2 by default) AND the boundary cut AND the opt-in bands, the
#: verdict until-N counts and ``selected`` reproduces.  ABSENT on an archive
#: written before it (2026-10-15), whose ``in_spec`` is the legacy +/-2 %
#: per-coil drift verdict (AND the engine's opt-in bands; no boundary cut):
#: readers report it as :data:`IN_SPEC_DRIFT_SOURCE` (:func:`in_spec_fields`).
IN_SPEC_SOURCE_ATTR = "in_spec_source"
IN_SPEC_RULE_SOURCE = "inspec_rule"
IN_SPEC_DRIFT_SOURCE = ("coil_drift: the archive predates the rule verdict; "
                        "its in_spec is the legacy +/-2 % drift test")
#: Per-draw attribute: the legacy +/-2 % per-coil drift verdict
#: (``passes_coil_spec`` on the bare F / VSC-channel drifts), kept beside the
#: rule's ``in_spec``; never a filter.
COIL_DRIFT_IN_SPEC_ATTR = "coil_drift_in_spec"
#: Per-draw opt-in band verdicts (written only while that band is on).
L_I_BAND_FLAG = "passes_l_i_band"
BETA_P_BAND_FLAG = "passes_beta_p_band"
Q0_GATE_FLAG = "passes_q0_gate"
JPHI_BAND_FLAG = "passes_jphi_band"
CORE_GRADIENT_FLAG = "passes_core_gradient"
#: Per-draw stored numbers of the core-current checks (:func:`core_current_metrics`).
JPHI_BAND_Z_ATTR = "jphi_band_z"
CORE_GRADIENT_RATIO_ATTR = "core_gradient_ratio"
#: Per-draw (and ``_baseline``) stored scalars: beta_p and Lambda.
BETA_P_ATTR = "beta_p"
LAMBDA_ATTR = "Lambda"
#: Scan-group attribute naming the Lambda definition used.
LAMBDA_DEFINITION_ATTR = "lambda_definition"

#: The exact definition of the stored Lambda (also docs/physics-notes.md).
LAMBDA_DEFINITION = (
    "Lambda = beta_p + l_i(3)/2. beta_p = 2 mu0 <p>_V / B_pa^2 with B_pa = "
    "mu0 I_p / L_p (L_p the poloidal circumference of the LCFS) and <p>_V "
    "the volume average of the FULL pressure (thermal + fast, the "
    "separatrix pressure added back: the full-pressure frame of "
    "bouquet.edge_pressure; TokaMaker beta_pol / 100). l_i(3) = 2 int B_p^2 "
    "dV / (mu0^2 I_p^2 R_0) with R_0 the magnetic-axis major radius "
    "(TokaMaker li_normalization='iter'). No '-1' (the large-aspect-ratio "
    "Shafranov-shift convention beta_p + l_i/2 - 1 is not used). For a "
    "shaped plasma the magnetics' true observable is a shape-weighted "
    "combination of the current and pressure moments, which the coil "
    "chi-squared test against the free-boundary solve enforces exactly; "
    "Lambda is the scalar summary, reported, never filtered on.")

#: The l_i tolerance the LEGACY routes keep using to CONSTRUCT their draws
#: (rejection sampling of the GPR inductive inside the draw, and the legacy
#: reconstruction's post-corrective report) when ``l_i_tolerance`` is unset:
#: the old default, so those routes are unchanged.  It is not a filter.
LEGACY_L_I_CONSTRUCTION_TOLERANCE = 0.05
#: Default half-width of a measurement-derived band, in units of the stated
#: 1-sigma (``GenerationConfig.inspec_band_nsigma``).
DEFAULT_BAND_NSIGMA = 2.0


def shafranov_lambda(beta_p, l_i):
    """``beta_p + l_i/2`` (:data:`LAMBDA_DEFINITION`); NaN when either is
    missing or non-finite."""
    try:
        b, li = float(beta_p), float(l_i)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(b) and np.isfinite(li)):
        return float("nan")
    return b + 0.5 * li


def l_i_construction_tolerance(value):
    """The l_i tolerance a LEGACY route constructs its draws (and reports its
    reconstruction) with: ``value`` (``GenerationConfig.l_i_tolerance``) when
    set, else :data:`LEGACY_L_I_CONSTRUCTION_TOLERANCE`.  Accepts the config
    itself."""
    if hasattr(value, "l_i_tolerance"):
        value = value.l_i_tolerance
    return (LEGACY_L_I_CONSTRUCTION_TOLERANCE if value is None
            else float(value))


def _positive(name, v, allow_none=True):
    import math
    import numbers
    if v is None and allow_none:
        return None
    if (isinstance(v, bool) or not isinstance(v, numbers.Real)
            or not math.isfinite(float(v)) or not float(v) > 0.0):
        raise ValueError(f"{name}={v!r} must be a finite number > 0"
                         + (" (or None = off)" if allow_none else ""))
    return float(v)


def pedestal_top(x, jphi_ref):
    """The run-grid coordinate inside which the core-current checks act: the
    minimum of the baseline *jphi_ref* between ``x = 0.5`` and its maximum on
    ``x >= 0.5`` (the edge bootstrap peak); ``x = 0.5`` without an edge peak."""
    x, j = np.asarray(x, dtype=float), np.asarray(jphi_ref, dtype=float)
    idx = np.flatnonzero(x >= 0.5)
    seg = idx[idx <= idx[np.argmax(j[idx])]]
    return float(x[seg[np.argmin(j[seg])]])


def core_current_metrics(x, jphi, jphi_ref, sigma_ref):
    """``(z, gradient_ratio, x_ped)`` of a draw's *jphi* inside the pedestal
    top (:func:`pedestal_top` of the baseline *jphi_ref*): the largest
    ``|jphi - jphi_ref| / sigma_ref`` there, and the largest ``|d jphi /
    d sqrt(x)|`` there over the baseline's.  NaN when not computable."""
    x = np.asarray(x, dtype=float)
    j, jr = np.asarray(jphi, dtype=float), np.asarray(jphi_ref, dtype=float)
    sg = np.asarray(sigma_ref, dtype=float)
    xp = pedestal_top(x, jr)
    core = x <= xp
    ok = core & (sg > 0)
    z = float(np.max(np.abs(j - jr)[ok] / sg[ok])) if ok.any() else float("nan")
    rho = np.sqrt(np.clip(x, 0.0, None))
    g_ref = np.max(np.abs(np.gradient(jr, rho)[core]))
    ratio = (float(np.max(np.abs(np.gradient(j, rho)[core])) / g_ref)
             if g_ref > 0 else float("nan"))
    return z, ratio, xp


class InspecBands:
    """The opt-in bands of the in-spec rule (everything beyond the
    magnetics).  Built by :func:`resolve_inspec_bands` from a
    :class:`~bouquet.config.GenerationConfig`; every band off by default.

    * ``l_i_tolerance`` -- a FRACTIONAL l_i(3) band ``|l_i - l_i*| <=
      l_i_tolerance * l_i*`` (``l_i*`` the reconstruction's l_i), the
      historical knob, applied only when set explicitly;
    * ``l_i_sigma`` -- the stated 1-sigma of l_i(3) (absolute, l_i(3)
      units) of the source's MSE / diamagnetic fit: the band is ``|l_i -
      l_i*| <= nsigma * l_i_sigma``; exclusive with ``l_i_tolerance``;
    * ``beta_p_target`` / ``beta_p_sigma`` -- a diamagnetic beta_p and its
      1-sigma: ``|beta_p - beta_p_target| <= nsigma * beta_p_sigma``, beta_p
      as in :data:`LAMBDA_DEFINITION`; both or neither;
    * ``nsigma`` -- the half-width of a sigma band, in sigma;
    * ``constrain_sawteeth`` -- the q0 >= 1 gate (``GenerationConfig.
      constrain_sawteeth``), carried here because it is judged with them;
    * ``jphi_band_nsigma`` -- the draw's ``jphi_total`` inside the pedestal
      top within this many of the baseline's ``sigma_jphi`` of the baseline's
      (:func:`core_current_metrics`);
    * ``core_gradient_ratio`` -- the draw's steepest ``d jphi_total / d
      sqrt(x)`` inside the pedestal top at most this times the baseline's.

    A draw whose banded quantity is missing or non-finite FAILS that band.
    """

    __slots__ = ("l_i_tolerance", "l_i_sigma", "beta_p_target",
                 "beta_p_sigma", "nsigma", "constrain_sawteeth",
                 "jphi_band_nsigma", "core_gradient_ratio")

    def __init__(self, l_i_tolerance=None, l_i_sigma=None, beta_p_target=None,
                 beta_p_sigma=None, nsigma=DEFAULT_BAND_NSIGMA,
                 constrain_sawteeth=False, jphi_band_nsigma=None,
                 core_gradient_ratio=None):
        import math
        import numbers
        self.l_i_tolerance = _positive("l_i_tolerance", l_i_tolerance)
        self.l_i_sigma = _positive("l_i_sigma", l_i_sigma)
        if self.l_i_tolerance is not None and self.l_i_sigma is not None:
            raise ValueError(
                "l_i_tolerance and l_i_sigma are two spellings of the l_i "
                "band (a fraction, or a stated 1-sigma); set one, not both")
        if (beta_p_target is None) != (beta_p_sigma is None):
            raise ValueError(
                "the beta_p band needs both beta_p_target and beta_p_sigma "
                "(a diamagnetic measurement and its 1-sigma), or neither")
        if beta_p_target is not None:
            if (isinstance(beta_p_target, bool)
                    or not isinstance(beta_p_target, numbers.Real)
                    or not math.isfinite(float(beta_p_target))):
                raise ValueError(f"beta_p_target={beta_p_target!r} must be a "
                                 "finite number")
            beta_p_target = float(beta_p_target)
        self.beta_p_target = beta_p_target
        self.beta_p_sigma = _positive("beta_p_sigma", beta_p_sigma)
        self.nsigma = _positive("inspec_band_nsigma", nsigma, allow_none=False)
        self.constrain_sawteeth = bool(constrain_sawteeth)
        self.jphi_band_nsigma = _positive("jphi_band_nsigma", jphi_band_nsigma)
        self.core_gradient_ratio = _positive("core_gradient_ratio",
                                             core_gradient_ratio)

    # ---- what is on ---------------------------------------------------
    @property
    def l_i_on(self) -> bool:
        return self.l_i_tolerance is not None or self.l_i_sigma is not None

    @property
    def beta_p_on(self) -> bool:
        return self.beta_p_target is not None

    @property
    def core_on(self) -> bool:
        """A core-current check is on (judged on the draw's jphi_total)."""
        return (self.jphi_band_nsigma is not None
                or self.core_gradient_ratio is not None)

    @property
    def any_on(self) -> bool:
        return (self.l_i_on or self.beta_p_on or self.constrain_sawteeth
                or self.core_on)

    def rule(self) -> str:
        """The in-spec rule's name (:data:`INSPEC_RULE_ATTR`)."""
        return ("magnetics" + ("+li" if self.l_i_on else "")
                + ("+beta_p" if self.beta_p_on else "")
                + ("+q0" if self.constrain_sawteeth else "")
                + ("+jphi" if self.jphi_band_nsigma is not None else "")
                + ("+jgrad" if self.core_gradient_ratio is not None else ""))

    def record(self) -> dict:
        """JSON-able record of the settings (:data:`INSPEC_BANDS_ATTR`)."""
        return dict(rule=self.rule(), l_i_tolerance=self.l_i_tolerance,
                    l_i_sigma=self.l_i_sigma,
                    beta_p_target=self.beta_p_target,
                    beta_p_sigma=self.beta_p_sigma, nsigma=self.nsigma,
                    constrain_sawteeth=self.constrain_sawteeth,
                    jphi_band_nsigma=self.jphi_band_nsigma,
                    core_gradient_ratio=self.core_gradient_ratio)

    @classmethod
    def from_record(cls, rec):
        rec = dict(rec or {})
        return cls(l_i_tolerance=rec.get("l_i_tolerance"),
                   l_i_sigma=rec.get("l_i_sigma"),
                   beta_p_target=rec.get("beta_p_target"),
                   beta_p_sigma=rec.get("beta_p_sigma"),
                   nsigma=rec.get("nsigma", DEFAULT_BAND_NSIGMA),
                   constrain_sawteeth=rec.get("constrain_sawteeth", False),
                   jphi_band_nsigma=rec.get("jphi_band_nsigma"),
                   core_gradient_ratio=rec.get("core_gradient_ratio"))

    def same_as(self, other) -> bool:
        return other is not None and self.record() == other.record()

    def __eq__(self, other):
        return isinstance(other, InspecBands) and self.same_as(other)

    def __hash__(self):
        return hash(tuple(sorted(self.record().items())))

    def __repr__(self):
        return f"<InspecBands {self.rule()} {self.record()}>"

    # ---- the verdict --------------------------------------------------
    def verdicts(self, *, l_i=None, l_i_reference=None, beta_p=None,
                 q0=None, q0_psi_N=None, jphi_z=None,
                 gradient_ratio=None) -> dict:
        """The opt-in bands' verdicts for one draw.

        Returns ``l_i_in_band`` / ``beta_p_in_band`` / ``q0_ok`` (``None``
        for a band that is off), the numbers judged (``l_i_3``,
        ``l_i_reference``, ``l_i_rel``, ``l_i_half_width``, ``beta_p``,
        ``beta_p_z``, ``q0``), ``Lambda`` (reported, never judged),
        ``in_band`` (the AND of the bands that are on; True with none on),
        ``reasons`` and ``rule``.  The core-current checks judge *jphi_z* and
        *gradient_ratio* (:func:`core_current_metrics`): ``jphi_in_band`` /
        ``core_gradient_ok``."""
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
        li, li_ref, bp = _f(l_i), _f(l_i_reference), _f(beta_p)
        rel = (abs(li - li_ref) / li_ref
               if np.isfinite(li) and np.isfinite(li_ref) and li_ref
               else float("nan"))
        out = dict(rule=self.rule(), l_i_3=li, l_i_reference=li_ref,
                   l_i_rel=float(rel), beta_p=bp,
                   Lambda=shafranov_lambda(bp, li),
                   l_i_tolerance=self.l_i_tolerance, l_i_sigma=self.l_i_sigma,
                   beta_p_target=self.beta_p_target,
                   beta_p_sigma=self.beta_p_sigma, nsigma=self.nsigma,
                   constrain_sawteeth=self.constrain_sawteeth,
                   l_i_in_band=None, beta_p_in_band=None, q0_ok=None,
                   l_i_half_width=None, beta_p_z=None, q0=_f(q0),
                   jphi_band_nsigma=self.jphi_band_nsigma,
                   core_gradient_ratio=self.core_gradient_ratio,
                   jphi_z=_f(jphi_z), gradient_ratio=_f(gradient_ratio),
                   jphi_in_band=None, core_gradient_ok=None)
        reasons = []
        if self.l_i_on:
            if self.l_i_tolerance is not None:
                hw = self.l_i_tolerance * abs(li_ref)
                what = f"l_i_tolerance {self.l_i_tolerance:g} (fraction)"
            else:
                hw = self.nsigma * self.l_i_sigma
                what = f"{self.nsigma:g} x l_i_sigma {self.l_i_sigma:g}"
            dev = abs(li - li_ref)
            ok = bool(np.isfinite(dev) and np.isfinite(hw) and dev <= hw)
            out["l_i_half_width"] = float(hw)
            out["l_i_in_band"] = ok
            if not ok:
                reasons.append(f"l_i band: |l_i - l_i*| = {dev:.4g} > {hw:.4g} "
                               f"({what}; l_i = {li:.4f}, l_i* = {li_ref:.4f})")
        if self.beta_p_on:
            z = (bp - self.beta_p_target) / self.beta_p_sigma
            ok = bool(np.isfinite(z) and abs(z) <= self.nsigma)
            out["beta_p_z"] = float(z)
            out["beta_p_in_band"] = ok
            if not ok:
                reasons.append(f"beta_p band: beta_p = {bp:.4g} is {z:+.2f} "
                               f"sigma from {self.beta_p_target:g} (|z| <= "
                               f"{self.nsigma:g})")
        if self.constrain_sawteeth:
            q = out["q0"]
            ok = bool(np.isfinite(q) and q >= 1.0)
            out["q0_ok"] = ok
            if not ok:
                reasons.append(f"constrain_sawteeth: q0 = {q:.4f} < 1"
                               + ("" if q0_psi_N is None
                                  else f" at psi_N {float(q0_psi_N):g}"))
        if self.jphi_band_nsigma is not None:
            z = out["jphi_z"]
            ok = bool(np.isfinite(z) and z <= self.jphi_band_nsigma)
            out["jphi_in_band"] = ok
            if not ok:
                reasons.append(f"jphi band: |jphi - jphi*| reaches {z:.2f} "
                               f"sigma_jphi inside the pedestal top (<= "
                               f"{self.jphi_band_nsigma:g})")
        if self.core_gradient_ratio is not None:
            r = out["gradient_ratio"]
            ok = bool(np.isfinite(r) and r <= self.core_gradient_ratio)
            out["core_gradient_ok"] = ok
            if not ok:
                reasons.append(f"core gradient: steepest d jphi/d sqrt(x) "
                               f"inside the pedestal top is {r:.2f} x the "
                               f"baseline's (<= {self.core_gradient_ratio:g})")
        out["in_band"] = bool(all(v for v in (out["l_i_in_band"],
                                              out["beta_p_in_band"],
                                              out["q0_ok"], out["jphi_in_band"],
                                              out["core_gradient_ok"])
                                  if v is not None))
        out["reasons"] = reasons
        return out

    def flags(self, verdicts) -> dict:
        """The per-draw flag attrs of *verdicts* (only the bands that are
        on) plus ``passes_draw_band`` (their AND)."""
        out = {}
        if verdicts.get("l_i_in_band") is not None:
            out[L_I_BAND_FLAG] = bool(verdicts["l_i_in_band"])
        if verdicts.get("beta_p_in_band") is not None:
            out[BETA_P_BAND_FLAG] = bool(verdicts["beta_p_in_band"])
        if verdicts.get("q0_ok") is not None:
            out[Q0_GATE_FLAG] = bool(verdicts["q0_ok"])
        if verdicts.get("jphi_in_band") is not None:
            out[JPHI_BAND_FLAG] = bool(verdicts["jphi_in_band"])
        if verdicts.get("core_gradient_ok") is not None:
            out[CORE_GRADIENT_FLAG] = bool(verdicts["core_gradient_ok"])
        out["passes_draw_band"] = bool(verdicts["in_band"])
        return out


def resolve_inspec_bands(gc) -> InspecBands:
    """The :class:`InspecBands` of a :class:`~bouquet.config.GenerationConfig`
    (``l_i_tolerance``, ``l_i_sigma``, ``beta_p_target``, ``beta_p_sigma``,
    ``inspec_band_nsigma``, ``constrain_sawteeth``, ``jphi_band_nsigma``,
    ``core_gradient_ratio``).  Validates; a field a stand-in config does not
    carry is off."""
    return InspecBands(
        l_i_tolerance=getattr(gc, "l_i_tolerance", None),
        l_i_sigma=getattr(gc, "l_i_sigma", None),
        beta_p_target=getattr(gc, "beta_p_target", None),
        beta_p_sigma=getattr(gc, "beta_p_sigma", None),
        nsigma=getattr(gc, "inspec_band_nsigma", DEFAULT_BAND_NSIGMA),
        constrain_sawteeth=bool(getattr(gc, "constrain_sawteeth", False)),
        jphi_band_nsigma=getattr(gc, "jphi_band_nsigma", None),
        core_gradient_ratio=getattr(gc, "core_gradient_ratio", None))


def in_spec_fields(attrs) -> dict:
    """``in_spec``, ``coil_drift_in_spec`` and ``in_spec_source`` of one
    archived draw's attrs (those present).

    An archive that stamps :data:`IN_SPEC_SOURCE_ATTR` carries the rule's
    verdict in ``in_spec`` and the drift verdict in
    :data:`COIL_DRIFT_IN_SPEC_ATTR`.  An older one carries only ``in_spec``,
    which IS the drift verdict: it is reported under both names, with
    ``in_spec_source`` = :data:`IN_SPEC_DRIFT_SOURCE` saying so -- select
    on ``selected`` (the filters' verdict) on such an archive."""
    out = {}
    src = attrs.get(IN_SPEC_SOURCE_ATTR) if IN_SPEC_SOURCE_ATTR in attrs \
        else None
    if isinstance(src, bytes):
        src = src.decode()
    if src is not None:
        if "in_spec" in attrs:
            out["in_spec"] = bool(attrs["in_spec"])
        if COIL_DRIFT_IN_SPEC_ATTR in attrs:
            out[COIL_DRIFT_IN_SPEC_ATTR] = bool(attrs[COIL_DRIFT_IN_SPEC_ATTR])
        out["in_spec_source"] = str(src)
    elif "in_spec" in attrs:
        out["in_spec"] = bool(attrs["in_spec"])
        out[COIL_DRIFT_IN_SPEC_ATTR] = bool(attrs["in_spec"])
        out["in_spec_source"] = IN_SPEC_DRIFT_SOURCE
    return out


def q0_gate_switched_off(q0_gate) -> bool:
    """Whether a :data:`Q0_GATE_RECORD_KEY` record says the configured q0
    gate was switched off for the scan."""
    return bool(isinstance(q0_gate, dict) and q0_gate.get("configured")
                and not q0_gate.get("applied"))


def apply_q0_gate_record(bands, extra, q0_gate):
    """``(bands, extra)`` of the rule AS APPLIED: with the q0 gate switched
    off by ``generate_bouquet`` (*q0_gate*, :data:`Q0_GATE_RECORD_KEY`) the
    bands lose ``constrain_sawteeth`` (and a legacy route's
    ``constrain_sawteeth_construction`` is False); the record rides in
    *extra* either way when one was made."""
    extra = dict(extra or {})
    if not isinstance(q0_gate, dict):
        return bands, extra
    extra[Q0_GATE_RECORD_KEY] = dict(q0_gate)
    if q0_gate_switched_off(q0_gate):
        bands = InspecBands.from_record(dict(bands.record(),
                                             constrain_sawteeth=False))
        if "constrain_sawteeth_construction" in extra:
            extra["constrain_sawteeth_construction"] = False
    return bands, extra


def read_inspec_record(h5path_or_header, scan_key=None):
    """The scan's stamped :data:`INSPEC_BANDS_ATTR` record as a dict (the
    bands plus the extra keys stamped beside them), or ``None``."""
    import json
    h5path = _resolve(h5path_or_header)
    with h5py.File(h5path, "r") as hf:
        gp = _scan_group_path(scan_key)
        if gp not in hf:
            return None
        rec = hf[gp].attrs.get(INSPEC_BANDS_ATTR)
    if rec is None:
        return None
    try:
        out = json.loads(rec.decode() if isinstance(rec, bytes) else str(rec))
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _scan_group_path(scan_key):
    bkey = _scan_key(scan_key)
    return f"scan/{bkey}" if bkey is not None else "/"


def read_inspec_rule(h5path_or_header, scan_key=None):
    """``(rule, InspecBands or None)`` stamped on a scan, or ``(None,
    None)`` on an archive that predates the record (its engine draws' band
    flags were then judged under the old unconditional l_i band)."""
    import json
    h5path = _resolve(h5path_or_header)
    with h5py.File(h5path, "r") as hf:
        gp = _scan_group_path(scan_key)
        if gp not in hf:
            return None, None
        a = hf[gp].attrs
        rule = a.get(INSPEC_RULE_ATTR)
        rec = a.get(INSPEC_BANDS_ATTR)
    if rule is None:
        return None, None
    rule = rule.decode() if isinstance(rule, bytes) else str(rule)
    bands = None
    if rec is not None:
        try:
            bands = InspecBands.from_record(json.loads(
                rec.decode() if isinstance(rec, bytes) else str(rec)))
        except (TypeError, ValueError):
            bands = None
    return rule, bands


def stamp_inspec_rule(h5path_or_header, scan_key, bands, extra=None):
    """Stamp the in-spec rule and its band settings on the scan group
    (:data:`INSPEC_RULE_ATTR`, :data:`INSPEC_BANDS_ATTR`) and the Lambda
    definition (:data:`LAMBDA_DEFINITION_ATTR`).  *extra* is merged into the
    JSON record (e.g. a legacy route's draw-construction tolerance).

    A :data:`Q0_GATE_RECORD_KEY` record already on the scan that says the
    q0 gate was switched off is kept when *bands* carry no q0 gate and
    *extra* brings none (a re-filter under the stamped rule must not lose
    why the configured gate is absent)."""
    import json
    h5path = _resolve(h5path_or_header)
    rec = dict(bands.record(), **(extra or {}))
    with h5py.File(h5path, "a") as hf:
        gp = _scan_group_path(scan_key)
        if gp not in hf:
            return
        a = hf[gp].attrs
        if Q0_GATE_RECORD_KEY not in rec and not bands.constrain_sawteeth:
            old = a.get(INSPEC_BANDS_ATTR)
            try:
                old = (json.loads(old.decode() if isinstance(old, bytes)
                                  else str(old)) if old is not None else None)
            except (TypeError, ValueError):
                old = None
            if (isinstance(old, dict)
                    and q0_gate_switched_off(old.get(Q0_GATE_RECORD_KEY))):
                rec[Q0_GATE_RECORD_KEY] = old[Q0_GATE_RECORD_KEY]
        a[INSPEC_RULE_ATTR] = bands.rule()
        a[INSPEC_BANDS_ATTR] = json.dumps(rec)
        a[LAMBDA_DEFINITION_ATTR] = LAMBDA_DEFINITION


def _full_beta_p(edge_record):
    """beta_p (fraction) of an archived edge-pressure record's FULL frame,
    or NaN."""
    try:
        return float(edge_record["frames"]["full"]["beta_pol"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _engine_post_hoc(grp):
    try:
        from .engine import read_engine_json
        rec = read_engine_json(grp)
    except Exception:
        return None
    return (rec or {}).get("post_hoc") if isinstance(rec, dict) else None


def _draw_scalars_for_rule(grp, l_i_reference):
    """``(l_i, l_i_reference, beta_p, q0, q0_psi_N, source)`` of one
    archived draw: the engine's own post-hoc record when present (the very
    numbers its in-flight verdict judged), else the archived ``l_i(3)`` /
    ``beta_p`` attrs (beta_p from the edge-pressure record's full frame when
    the attr is absent) against the baseline's ``l_i_target``."""
    import json
    from .edge_pressure import EDGE_PRESSURE_ATTR
    a = grp.attrs
    ph = _engine_post_hoc(grp) or {}
    li = ph.get("l_i_3", a.get("l_i(3)", np.nan))
    ref = ph.get("l_i_reference", l_i_reference)
    bp = ph.get("beta_p", a.get(BETA_P_ATTR, None))
    if bp is None:
        raw = a.get(EDGE_PRESSURE_ATTR)
        try:
            bp = (_full_beta_p(json.loads(raw.decode() if isinstance(raw, bytes)
                                          else str(raw)))
                  if raw is not None else np.nan)
        except (TypeError, ValueError):
            bp = np.nan
    src = ENGINE_RECORD_SOURCE if ph else "archived attrs"
    return (li, ref, bp, ph.get("q0", np.nan), ph.get("q0_psi_N"), src)


ENGINE_RECORD_SOURCE = "engine post_hoc record"


_JPHI = "jphi_total"


def _core_reference(hf, bl_path):
    """``(x, jphi_ref, sigma_ref)`` of a scan's baseline for the core-current
    checks; None when the baseline lacks one of them."""
    from .schema import stored_name
    if bl_path not in hf:
        return None
    g = hf[bl_path]
    names = ("psi_N", stored_name(g, _JPHI), "sigma_jphi")
    if not all(n in g for n in names):
        return None
    return tuple(np.asarray(g[n][()], dtype=float) for n in names)


def refilter_inspec(h5path_or_header, bands=None, scan_key=None, *,
                    verbose=True):
    """Re-judge the opt-in bands of an archive under *bands* (an
    :class:`InspecBands`, a GenerationConfig, or ``None`` = the default
    rule, the magnetics alone), from the stored per-draw numbers; no physics
    is re-run.

    For every archived draw: the band flags that are on are (re)written
    (:data:`L_I_BAND_FLAG`, :data:`BETA_P_BAND_FLAG`, :data:`Q0_GATE_FLAG`,
    :data:`JPHI_BAND_FLAG`, :data:`CORE_GRADIENT_FLAG`),
    those that are off are REMOVED, ``passes_draw_band`` is the AND of the
    ones on (written whenever a band is on or the draw already carried the
    flag -- every engine draw; removed from another draw when no band is
    on), ``beta_p`` / ``Lambda`` are stamped where
    missing, and ``selected`` is recomputed with the coil and boundary flags
    already there (run the coil / boundary filters, e.g.
    :meth:`bouquet.run.Bouquet.filter`, to re-judge those).  The scan is
    stamped with the rule (:func:`stamp_inspec_rule`).

    The numbers judged: an engine draw's own post-hoc record (``l_i_3``
    against its ``l_i_reference``, ``beta_p``, ``q0``) where present, else
    the archived ``l_i(3)`` against the baseline's ``l_i_target`` and the
    archived ``beta_p`` (else the edge-pressure record's full-frame
    beta_pol / 100).  The q0 gate is judged on engine draws only (a legacy
    draw records no q0 at the gate's radius; its flag is left absent).  The
    core-current checks are recomputed from the archived ``jphi_total`` and
    the baseline's ``jphi_total`` / ``sigma_jphi`` (:func:`core_current_metrics`).

    This is how an archive made before the 2026-10-14 rule is re-filtered:
    its engine draws carry ``passes_draw_band`` from the old unconditional
    l_i band, which ``refilter_inspec(path)`` resets to the magnetics rule.
    The in-loop ``until_n_inspec`` stamps are NOT rewritten: they record what
    the loop counted when it ran.

    Returns ``{scan_key: {"rule", "n_draws", "n_in_band", "n_selected",
    "draws": {idx: verdicts}}}`` (unwrapped for an explicit ``scan_key``).
    """
    if bands is None:
        bands = InspecBands()
    elif not isinstance(bands, InspecBands):
        bands = resolve_inspec_bands(bands)
    h5path = _resolve(h5path_or_header)
    _require_scan_key(h5path, scan_key)
    no_q0 = InspecBands.from_record(dict(bands.record(),
                                         constrain_sawteeth=False))
    out = {}
    for sv in _iter_scan_keys(h5path, scan_key):
        bkey = _scan_key(sv)
        bl_path = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
        rows = {}
        idxs = list_equilibrium_indices(h5path, scan_key=sv)
        with h5py.File(h5path, "a") as hf:
            ref = (float(hf[bl_path].attrs["l_i_target"])
                   if bl_path in hf and "l_i_target" in hf[bl_path].attrs
                   else np.nan)
            core_ref = _core_reference(hf, bl_path) if bands.core_on else None
            for i in idxs:
                gp = _group_path(sv, i)
                if gp not in hf:
                    continue
                g = hf[gp]
                li, li_ref, bp, q0, q0_psi, src = _draw_scalars_for_rule(g, ref)
                b = (bands if src == ENGINE_RECORD_SOURCE
                     or not bands.constrain_sawteeth else no_q0)
                jz = gr = None
                if core_ref is not None and _JPHI in g:
                    from .schema import stored_name
                    jz, gr, _ = core_current_metrics(
                        core_ref[0], g[stored_name(g, _JPHI)][()], *core_ref[1:])
                    g.attrs[JPHI_BAND_Z_ATTR] = float(jz)
                    g.attrs[CORE_GRADIENT_RATIO_ATTR] = float(gr)
                v = b.verdicts(l_i=li, l_i_reference=li_ref, beta_p=bp, q0=q0,
                               q0_psi_N=q0_psi, jphi_z=jz, gradient_ratio=gr)
                v["rule"], v["source"] = bands.rule(), src
                fl = b.flags(v)
                for name in (L_I_BAND_FLAG, BETA_P_BAND_FLAG, Q0_GATE_FLAG,
                             JPHI_BAND_FLAG, CORE_GRADIENT_FLAG):
                    if name in fl:
                        g.attrs[name] = fl[name]
                    elif name in g.attrs:
                        del g.attrs[name]
                # passes_draw_band: written while a band is on; with none on
                # it stays (True) on an engine draw, which always carries it,
                # and is removed from any other draw, where it would be an
                # earlier re-filter's
                if b.any_on or (src == ENGINE_RECORD_SOURCE
                                and "passes_draw_band" in g.attrs):
                    g.attrs["passes_draw_band"] = fl["passes_draw_band"]
                elif "passes_draw_band" in g.attrs:
                    del g.attrs["passes_draw_band"]
                if BETA_P_ATTR not in g.attrs and np.isfinite(v["beta_p"]):
                    g.attrs[BETA_P_ATTR] = float(v["beta_p"])
                if LAMBDA_ATTR not in g.attrs and np.isfinite(v["Lambda"]):
                    g.attrs[LAMBDA_ATTR] = float(v["Lambda"])
                _recompute_selected(g)
                rows[i] = v
        stamp_inspec_rule(h5path, sv, bands)
        sel = select_indices(h5path, scan_key=sv, selection="selected")
        n_in = sum(1 for v in rows.values() if v["in_band"])
        out[sv] = dict(rule=bands.rule(), n_draws=len(rows), n_in_band=n_in,
                       n_selected=len(sel), draws=rows)
        if verbose:
            print(f"[in-spec rule] scan {sv!r}: {bands.rule()} -- "
                  + (f"{n_in}/{len(rows)} draws inside the opt-in bands"
                     if bands.any_on else "no opt-in band on")
                  + f"; {len(sel)} of {len(rows)} selected")
    return _finalize_scan_result(out, scan_key)


def stamp_shafranov(h5path_or_header, scan_key=None):
    """Stamp ``beta_p`` and ``Lambda`` (:data:`LAMBDA_DEFINITION`) on every
    archived draw of the scan that lacks them, from its ``l_i(3)`` and its
    edge-pressure record's full frame, and the Lambda definition on the
    scan.  Returns the number of draws stamped.  (Engine draws stamp their
    own at archive time, from the state their band verdict judged.)"""
    h5path = _resolve(h5path_or_header)
    n = 0
    svs = _iter_scan_keys(h5path, scan_key)
    idx = {sv: list_equilibrium_indices(h5path, scan_key=sv) for sv in svs}
    with h5py.File(h5path, "a") as hf:
        for sv in svs:
            gs = _scan_group_path(sv)
            if gs in hf:
                hf[gs].attrs[LAMBDA_DEFINITION_ATTR] = LAMBDA_DEFINITION
            for i in idx[sv]:
                gp = _group_path(sv, i)
                if gp not in hf:
                    continue
                g = hf[gp]
                if LAMBDA_ATTR in g.attrs and BETA_P_ATTR in g.attrs:
                    continue
                li, _, bp, _, _, _ = _draw_scalars_for_rule(g, np.nan)
                lam = shafranov_lambda(bp, li)
                if np.isfinite(lam):
                    g.attrs[BETA_P_ATTR] = float(bp)
                    g.attrs[LAMBDA_ATTR] = float(lam)
                    n += 1
    return n


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
    return boundary_deviation_mm(bl_boundary, perturbed)


# --------------------------------------------------------------------------
#  selection accessors
# --------------------------------------------------------------------------
def read_filter_flags(h5path_or_header, scan_key=None):
    """Per-draw filter flags: ``{idx: {flag: value, ...}}`` for stored draws.

    Each inner dict carries ``passes_coil_filter``, ``passes_boundary_filter``,
    ``passes_draw_band`` (and the opt-in bands' own ``passes_l_i_band`` /
    ``passes_beta_p_band`` / ``passes_q0_gate``) and ``selected`` when
    present (missing flags are omitted), plus the raw metrics
    ``max_F_drift_pct``, ``max_VSC_drift_pct``, ``beta_p`` and ``Lambda``,
    and ``in_spec`` / ``coil_drift_in_spec`` / ``in_spec_source``
    (:func:`in_spec_fields`: the rule's verdict and the legacy drift one; on
    an archive that predates the rule verdict ``in_spec`` is the drift one,
    and ``in_spec_source`` says so). An explicit
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
                for k in (*_FILTER_FLAGS, *_BAND_FLAGS, "selected"):
                    if k in a:
                        rec[k] = bool(a[k])
                for k in ("max_F_drift_pct", "max_VSC_drift_pct",
                          "boundary_rms_mm", "boundary_max_mm", "beta_p",
                          "Lambda"):
                    if k in a:
                        rec[k] = float(a[k])
                rec.update(in_spec_fields(a))
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
#: The nearest ``pf_active`` sample may sit this far [s] from the requested time
#: before :func:`measured_coil_currents` warns that it is extrapolating.
PF_ACTIVE_TIME_TOL_S = 0.05


def measured_coil_currents(dd_path, time_s, time_tol_s=PF_ACTIVE_TIME_TOL_S):
    """``{name: (current_A, sigma_A)}`` from an IMAS ``pf_active`` at *time_s*.

    ``data_error_upper`` is read as the 1-sigma magnitude (FUSE writes the error
    itself there, not ``data + error``).

    *time_s* is a time in SECONDS on the coil time base -- not a scan key. The
    nearest sample is taken; when it is further than *time_tol_s* from the
    request (e.g. a time outside the coil time base, which would otherwise
    return the endpoint silently) a warning is emitted naming the offset.
    ``time_tol_s=None`` disables the check.
    """
    import json
    import warnings

    with open(dd_path) as fh:
        dd = json.load(fh)
    time_s = float(time_s)
    out, worst = {}, (0.0, None)
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
        j = int(np.argmin(np.abs(t - time_s)))
        dt = float(abs(t[j] - time_s))
        if dt > worst[0]:
            worst = (dt, name)
        out[name] = (float(d[j]), float(e[j]))
    if time_tol_s is not None and worst[1] is not None and worst[0] > float(time_tol_s):
        warnings.warn(
            f"measured_coil_currents: nearest pf_active sample for {worst[1]!r} is "
            f"{worst[0]:.4g} s from the requested t = {time_s:.4g} s (tolerance "
            f"{float(time_tol_s):.4g} s). The requested time is probably outside the "
            "coil time base; the endpoint sample is being used.", stacklevel=2)
    return out


def filter_coil_chi2(h5path_or_header, dd_path=None, scan_key=None,
                     chi2_max=None, apply=True, sigma=None, device=None,
                     era=None, sigma_ref=None, z_max=None, time_s=None):
    """Measurement-referenced coil filter (see :mod:`bouquet.coil_spec`).

    Scores each draw by ``chi2/nu`` of its coil currents against the baseline,
    weighted by the measured per-coil precision from ``pf_active``, and cuts at
    *chi2_max*. Unlike :func:`filter_coil_currents` this covers EVERY coil the
    measurement carries (E-coils included) and weights each by its own
    precision rather than a flat fractional band.

    The cut is NOT ``chi2_max=4.0``: that signature default is ``None``, which
    resolves to the device's calibrated acceptance (DIII-D: 6.1 / 6.3) or, with
    no device calibration, to the generic 4.0 / 5.0 rule of thumb.  The applied
    numbers are reported in the summary and stamped on the scan group.

    The acceptance thresholds are NOT dof-based: they are the chosen quantile of
    what real machine states score under the same sigma model (see
    :attr:`DeviceSpec.acceptance_provenance`), so they are tied to the coil set
    they were calibrated over.  The number of coils actually judged (``nu``) and
    the number the thresholds were calibrated at (``calibrated_nu``) are both
    recorded in the stamped model, and a mismatch warns -- ``max|z|`` in
    particular is an order statistic whose quantile moves with the coil count.

    A draw with no usable coil scores NaN and FAILS: unjudgeable is not a pass.
    A draw carrying no ``coil_currents``/``coil_names`` at all is likewise
    recorded as a FAILURE (and counted), never skipped.

    Per-coil sigma (see :func:`coil_spec.resolve_coil_sigma`): ``sigma`` explicit
    (``{"floor","fraction"}`` model, per-coil table, or callable), else the
    ``device`` model (named, or detected from the mesh coil names), else
    :class:`coil_spec.CoilSigmaUnavailable` is raised.  Needs no dd.

    ``era`` names the device tolerance era (see :func:`devices.era_labels`) when
    the device's floor is era-dependent.  It is never inferred from a file name:
    pass it explicitly (``FilterConfig.coil_daq_era``, or via
    :func:`devices.era_for_pulse` from an explicit pulse number).  ``None``
    selects the device's strictest-documented default era and warns.

    ``sigma_ref`` (legacy, dd-referenced; requires ``dd_path`` and ``time_s``):
    ``"d3d"`` uses the digitizer table rescaled by |I_base|/|I_meas|; a dict
    ``{"F": A, "E": A}`` a custom per-family table; ``"dd"`` the dd's own
    ``data_error_upper``.  Raises :class:`ValueError` if given without them.
    ``time_s`` is a time in seconds on the coil time base -- the scan key is a
    label, not a time, and is never used as one.

    A draw passes when ``chi2/nu <= chi2_max`` AND its worst single coil has
    ``|z| <= z_max``: the pooled statistic alone lets one coil at 7 sigma hide
    behind seventeen quiet ones.  Both default (``None``) to the device's
    empirically calibrated acceptance (:attr:`DeviceSpec.acceptance`) when the
    sigma came from a device model, else to :data:`devices.GENERIC_ACCEPTANCE`.
    Pass ``z_max=False`` to disable the guard.

    The model used is stamped on the scan group as ``coil_sigma_model`` (JSON)
    and ``coil_filter`` = "chi2" when ``apply`` is True.
    """
    import json
    import warnings
    from .coil_spec import (CoilSigmaUnavailable, coil_chi2,
                            coil_sigma_in_base_units, with_sigma_ref,
                            resolve_coil_acceptance, resolve_coil_sigma)
    if sigma_ref is not None and dd_path is None:
        raise ValueError("filter_coil_chi2: sigma_ref is dd-referenced and needs dd_path")
    if sigma_ref is not None and time_s is None:
        raise ValueError("filter_coil_chi2: sigma_ref is dd-referenced and needs an "
                         "explicit time_s [s] on the coil time base (the scan key is "
                         "a label, not a time)")

    h5path = _resolve(h5path_or_header)
    summary = {}
    for sv in _iter_scan_keys(h5path, scan_key):
        bkey = _scan_key(sv)
        idxs = list_equilibrium_indices(h5path, scan_key=sv)
        no_coil_data = []
        with h5py.File(h5path, "r") as hf:
            # flat/legacy archives have no ``scan/`` group -- same idiom as
            # ``_baseline_boundary``: the draws hang off the file root.
            grp = hf[f"scan/{bkey}"] if bkey is not None else hf
            if "_baseline" not in grp:
                # same class of failure as the v1 baseline below -- there is no
                # machine state to judge the draws against -- so it takes the
                # same loud exit.  Skipping it silently left the summary empty,
                # and an explicit ``scan_key`` then turned that into a bare
                # ``StopIteration`` out of ``_finalize_scan_result``.
                where = f"scan/{bkey}" if bkey is not None else "the archive root"
                raise CoilSigmaUnavailable(
                    f"{where} carries no _baseline group, so no draw can be judged "
                    "against the reconstructed machine state. Re-run the generation "
                    "(the baseline is written at the start of a scan), or use the "
                    "legacy coil filter (filtering.coil_filter = 'legacy').")
            bl = grp["_baseline"]
            bn = _read_coil_names(bl)          # schema-v2 dataset; [] on a v1 archive
            if not bn or "coil_currents" not in bl:
                raise CoilSigmaUnavailable(
                    "the archive baseline carries no coil_names/coil_currents (schema v1 "
                    "or a run stored without coil data), so no draw can be judged against "
                    "it. Re-run the generation with coil storage, or use the legacy "
                    "coil filter (filtering.coil_filter = 'legacy').")
            baseline = dict(zip(bn, np.asarray(
                bl["coil_currents"][()], dtype=float).tolist()))
            if sigma_ref is not None:
                meas = measured_coil_currents(dd_path, float(time_s))
                if sigma_ref != "dd":
                    meas = with_sigma_ref(meas, None if sigma_ref == "d3d" else sigma_ref)
                sig, model = coil_sigma_in_base_units(baseline, meas), {"kind": "dd_referenced", "sigma_ref": str(sigma_ref), "time_s": float(time_s)}
            else:
                sig, model = resolve_coil_sigma(baseline, sigma=sigma, device=device, era=era)
            # acceptance: explicit > device calibration (when sigma is the device model) >
            # generic. Resolved by the shared helper, which the until-N loop also calls --
            # the loop must cut at exactly these numbers.
            from .devices import get_device
            cm, zm, acc_src, cal_nu = resolve_coil_acceptance(
                model, chi2_max=chi2_max, z_max=z_max)
            model = dict(model, acceptance={"chi2_max": cm, "z_max": zm, "source": acc_src,
                                            "calibrated_nu": cal_nu, "nu_sigma": len(sig)})
            rows = {}
            for i in idxs:
                gp = _group_path(sv, i)
                if gp not in hf:
                    continue
                g = hf[gp]
                names = _read_coil_names(g)
                if "coil_currents" not in g or not names:
                    # unjudgeable: recorded as NaN so it FAILS below and is counted,
                    # rather than silently skipped (which left it `selected`).
                    rows[i] = {"chi2_nu": float("nan"), "chi2": float("nan"), "nu": 0,
                               "max_abs_z": float("nan"), "worst_coil": None, "z": {}}
                    no_coil_data.append(i)
                    continue
                draw = dict(zip(names, np.asarray(
                    g["coil_currents"][()], dtype=float).tolist()))
                rows[i] = coil_chi2(draw, baseline, sig)
        # the SAME predicate generate_bouquet's until-N loop stops on
        results = {i: passes_coil_chi2(r["chi2_nu"], r["max_abs_z"], cm, zm)
                   for i, r in rows.items()}
        if no_coil_data:
            warnings.warn(
                f"coil filter: {len(no_coil_data)} of {len(rows)} draws carry no coil "
                f"currents (indices {no_coil_data[:8]}{'...' if len(no_coil_data) > 8 else ''}) "
                "and were marked NOT selected -- unjudgeable is not a pass.", stacklevel=2)
        nu_used = max((r["nu"] for r in rows.values()), default=0)
        if cal_nu is not None and nu_used and int(nu_used) != int(cal_nu):
            warnings.warn(
                f"coil filter: acceptance chi2/nu <= {cm} and max|z| <= {zm} were calibrated "
                f"over {int(cal_nu)} coils but are being applied over {int(nu_used)}. These are "
                "empirical quantiles, not dof-based bounds: max|z| is an order statistic whose "
                "quantile rises with the coil count, so the true false-rejection rate is ABOVE "
                f"the nominal {100 * (1 - get_device(model['device']).acceptance.get('quantile', float('nan'))):.0f}%. "
                "Thresholds are unchanged; recalibrate them over the judged coil set, or "
                "restrict the metric to the calibrated set, before relying on the quantile.",
                stacklevel=2)
        model["acceptance"]["nu_used"] = int(nu_used)
        if apply:
            _write_filter_result(h5path, sv, results, "passes_coil_filter")
            with h5py.File(h5path, "a") as hf:
                gp = f"scan/{bkey}" if bkey is not None else "/"
                if gp in hf:
                    hf[gp].attrs["coil_filter"] = "chi2"
                    hf[gp].attrs["coil_sigma_model"] = json.dumps(model)
        n_pass = sum(results.values())
        summary[sv] = {
            "n_total": len(results), "n_pass": n_pass,
            "n_fail": len(results) - n_pass, "chi2_max": cm,
            "z_max": zm, "sigma_model": model,
            "n_coils": nu_used, "n_no_coil_data": len(no_coil_data),
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
        NOTE: with ``apply=True`` this REPLACES whatever coil verdict is
        already stored -- including the chi2 verdict that
        :meth:`bouquet.run.Bouquet.filter` writes by default -- and warns
        when it does.  To read the stored verdict use
        :func:`read_filter_flags`; to re-cut hypothetically use
        ``apply=False``.
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
            passed = passes_coil_spec(F, V, fthr, vthr)
            results[i] = passed
            draws[i] = {"max_F_drift_pct": F, "max_VSC_drift_pct": V,
                        "F_max_pct": fthr, "VSC_max_pct": vthr,
                        "passes": passed}
        if apply:
            # Refuse to overwrite a chi2 verdict SILENTLY.  Since the chi2
            # test became the default coil filter, the documented
            # "run.filter(); filter_coil_currents(HEADER, ...)" sequence
            # replaced the chi2 selection with this legacy band on every
            # notebook that used it (default apply=True), and nothing said
            # so.  The overwrite still happens when asked for -- this is a
            # legitimate re-cut -- but it is announced, and the scan-level
            # provenance attr is relabelled so the archive stays honest.
            _bkey = _scan_key(sv)
            _gp = f"scan/{_bkey}" if _bkey is not None else "/"
            with h5py.File(h5path, "a") as hf:
                if _gp in hf:
                    _prev = hf[_gp].attrs.get("coil_filter", None)
                    _prev = (_prev.decode() if isinstance(_prev, bytes)
                             else _prev)
                    if _prev == "chi2":
                        warnings.warn(
                            "filter_coil_currents(apply=True): overwriting the "
                            "chi2 coil-filter verdict on this scan with the "
                            "legacy +/-% band. If you only wanted the counts, "
                            "read them back with read_filter_flags() or pass "
                            "apply=False; the archive's coil_filter attr is now "
                            "'legacy'.", stacklevel=2)
                    hf[_gp].attrs["coil_filter"] = "legacy"
            _write_filter_result(h5path, sv, results, "passes_coil_filter")
        n_pass = sum(results.values())
        summary[sv] = {"n_total": len(results), "n_pass": n_pass,
                       "n_fail": len(results) - n_pass, "draws": draws}
        fig_payload.append((sv, rows, results))
    fig = _plot_coil_filter(fig_payload) if plot else None
    return _finalize_scan_result(summary, scan_key), fig


def filter_boundaries(h5path_or_header, scan_key=None,
                       rms_max_mm=None, max_max_mm=None,
                       apply=True, plot=True, cut_source="explicit"):
    """LCFS boundary-deviation filter.

    Computes each draw's RMS and max LCFS deviation from the recon
    baseline (same metric as ``plot_traces``).  With **no threshold**
    given this is purely diagnostic -- it reports the distribution and
    does NOT write any flag, so you can read a sensible cut off the plot
    first.  Pass ``rms_max_mm`` and/or ``max_max_mm`` (millimetres) to
    actually cut; a draw passes when it satisfies every supplied bound.

    Returns ``(summary, fig)`` analogous to :func:`filter_coil_currents`.

    With ``apply=True`` and a cut, the cut itself is recorded on the scan
    group (``boundary_rms_max_mm`` / ``boundary_max_max_mm`` /
    ``boundary_cut_source``, the last being *cut_source*: ``"explicit"``,
    ``"device:<name>"`` or ``"generic"``) and each draw's metric on the draw
    (``boundary_rms_mm`` / ``boundary_max_mm``), so the population a later
    statistic is built on is readable from the archive alone.

    ``cut_source="disabled"`` (no bound given) records that the boundary cut
    was switched OFF: with ``apply=True`` it removes any earlier
    ``passes_boundary_filter`` flag (so ``selected`` is the coil verdict
    alone), drops the stamped thresholds, writes each draw's metric and
    stamps ``boundary_cut_source="disabled"``.
    """
    h5path = _resolve(h5path_or_header)
    cutting = (rms_max_mm is not None) or (max_max_mm is not None)
    disabling = str(cut_source) == "disabled"
    if disabling and cutting:
        raise ValueError("cut_source='disabled' with a boundary bound given")
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
            passed = passes_boundary_spec(rms, mx, rms_max_mm, max_max_mm)
            results[i] = passed
            draws[i] = {"rms_mm": rms, "max_mm": mx, "passes": passed}
        if apply and disabling:
            with h5py.File(h5path, "a") as hf:
                for i, rms, mx in rows:
                    gp = _group_path(sv, i)
                    if gp in hf:
                        g = hf[gp]
                        g.attrs.pop("passes_boundary_filter", None)
                        _recompute_selected(g)
        if apply and (cutting or disabling):
            if cutting:
                _write_filter_result(h5path, sv, results, "passes_boundary_filter")
            _bkey = _scan_key(sv)
            _gp = f"scan/{_bkey}" if _bkey is not None else "/"
            with h5py.File(h5path, "a") as hf:
                if _gp in hf:
                    a = hf[_gp].attrs
                    for _name, _val in (("boundary_rms_max_mm", rms_max_mm),
                                        ("boundary_max_max_mm", max_max_mm)):
                        if _val is None:
                            a.pop(_name, None)
                        else:
                            a[_name] = float(_val)
                    a["boundary_cut_source"] = str(cut_source)
                for i, rms, mx in rows:
                    gp = _group_path(sv, i)
                    if gp in hf:
                        hf[gp].attrs["boundary_rms_mm"] = float(rms)
                        hf[gp].attrs["boundary_max_mm"] = float(mx)
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
    excluded per-draw groups are left out.  Indices are NOT renumbered, so
    the surviving draws keep their original labels (gaps are expected).
    The copy holds only what it keeps (a deleted HDF5 group would leave its
    space in the file).  The source file is never modified.

    Returns the number of draws kept (summed over scan values).
    """
    src = _resolve(h5path_or_header)
    out = os.path.abspath(out_path)
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(
            f"{out} exists; pass overwrite=True to replace it")
    kept, drop = 0, set()
    for sv in _iter_scan_keys(src, scan_key):
        keep = set(select_indices(src, scan_key=sv, selection=selection))
        for i in list_equilibrium_indices(src, scan_key=sv):
            if i in keep:
                kept += 1
            else:
                drop.add("/" + _group_path(sv, i).strip("/"))
    above = {p.rsplit("/", k)[0] for p in drop for k in range(1, p.count("/"))}

    def _copy(g_src, g_dst):
        g_dst.attrs.update(g_src.attrs)
        for name, obj in g_src.items():
            if obj.name in drop:
                continue
            if obj.name in above:
                _copy(obj, g_dst.create_group(name))
            else:
                g_src.copy(obj, g_dst, name=name)

    with h5py.File(src, "r") as hs, h5py.File(out, "w") as hd:
        _copy(hs, hd)
    return kept


def drop_file_blobs(h5path_or_header, names=(IFILE_DS,)):
    """Rewrite the archive without the byte blobs ``names`` (default the
    i-files, once they are used; also ``EQDSK_DS``, the g-files:
    ``GenerationConfig.store_tokamaker_state`` keeps the state both can be
    rebuilt from, ``DrawView.rebuild_ifile`` / ``rebuild_eqdsk``).
    Everything else is copied unchanged; the file is replaced atomically (a
    deleted HDF5 dataset would leave its space in the file).  Returns the
    number of blobs dropped."""
    src = _resolve(h5path_or_header)
    tmp = src + ".dropping"
    n = 0

    def _copy(g_src, g_dst):
        nonlocal n
        g_dst.attrs.update(g_src.attrs)
        for name, obj in g_src.items():
            if isinstance(obj, h5py.Group):
                _copy(obj, g_dst.create_group(name))
            elif name in names:
                n += 1
            else:
                g_src.copy(obj, g_dst, name=name)

    with h5py.File(src, "r") as hs, h5py.File(tmp, "w") as hd:
        _copy(hs, hd)
    os.replace(tmp, src)
    return n


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
