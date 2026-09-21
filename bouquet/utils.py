"""
HDF5 archive helpers and eqdsk I/O utilities for perturbed equilibria.
"""

import contextlib
import os
import sys
import tempfile
import warnings

import h5py
import numpy as np

from .schema import write_profile

#: The single l_i estimator the whole reconstruction/draw chain runs on.
#:
#: ``"iter(li3)"`` == TokaMaker ``li_normalization='iter'``
#: == ``2 * int(Bp^2 dV) / ((mu0 * Ip)^2 * R_axis)``
#: == the g-file reader's ``li["li(2)"]`` key (whose name is historical and
#: misleading -- it is not Jackson's li(2)).
#:
#: This is the ONLY estimator bouquet and TokaMaker agree on (0.17% across the
#: DIII-D 169510 beta-scan g-files); the li(1)/EFIT pair differs by +3.3%
#: because TokaMaker projects the padded surface onto the true separatrix
#: before summing perimeter.  Targeting one and measuring the other is
#: issue #20.  Written into every archive's ``_baseline`` group as
#: ``l_i_scale`` so a reader never has to guess.
LI_SCALE = "iter(li3)"


class OpenLCFSContourWarning(UserWarning):
    """No closed contour was found at the LCFS flux level.

    Emitted by :func:`select_closed_lcfs` when every candidate segment is
    open, so the caller is falling back to the longest one and any
    boundary metric derived from it is unreliable.
    """


#: Endpoint gap, as a fraction of a segment's own bounding-box diagonal,
#: under which :func:`select_closed_lcfs` calls that segment closed.
#:
#: matplotlib repeats the first vertex exactly when a contour closes, so a
#: genuinely closed LCFS scores 0.0 and any positive threshold would do.
#: The bar is kept this tight so that an OPEN branch cannot sneak through:
#: to be misread as closed it would have to return to within 0.1% of its own
#: extent, at which point it is closed in every sense that matters here.
_LCFS_CLOSURE_RTOL = 1e-3


def select_closed_lcfs(segs, context=""):
    r"""Pick the longest **closed** contour segment from *segs*.

    On a diverted equilibrium the :math:`\psi = \psi_\mathrm{LCFS}` level set
    contains the open separatrix branch running down to the divertor as well
    as the closed LCFS.  The open branch spans the full vessel height, so it
    frequently carries *more* points than the closed boundary -- and a plain
    ``max(segs, key=len)`` then silently returns the wrong curve.  Measured on
    a diverted lower-single-null case, that mis-selection reported a boundary
    RMS of 891.86 mm (open branch, n=1070) where the true closed-LCFS value is
    2.06 mm (n=969), which flipped the acceptance verdict PASS -> CHECK.  A
    second case selected the closed branch only because it happened to be
    longer, so the defect is general rather than case-specific (issue #33).

    Closedness is tested directly -- first vertex coincident with last, within
    :data:`_LCFS_CLOSURE_RTOL` of the segment's own bounding-box diagonal --
    rather than inferred from length.

    Parameters
    ----------
    segs : sequence of ndarray, shape (N, 2)
        Candidate contour segments, as returned by matplotlib ``allsegs``.
    context : str, optional
        Caller label, used only in the fallback warning message.

    Returns
    -------
    ndarray, shape (N, 2) or None
        The longest closed segment; the longest segment overall (with an
        :class:`OpenLCFSContourWarning`) if none closes; ``None`` when no
        candidate survives -- either *segs* was empty, or every entry had
        4 or fewer vertices and was discarded as degenerate.  Callers must
        treat ``None`` as "no usable contour" and fall back accordingly.
    """
    segs = [np.asarray(s, dtype=float) for s in segs if len(s) > 4]
    if not segs:
        return None

    closed = []
    for s in segs:
        span = np.ptp(s, axis=0)
        diag = float(np.hypot(*span))
        gap = float(np.hypot(*(s[0] - s[-1])))
        if diag > 0.0 and gap <= _LCFS_CLOSURE_RTOL * diag:
            closed.append(s)

    if closed:
        return max(closed, key=len)

    warnings.warn(
        f"no CLOSED contour at the LCFS flux level"
        f"{' in ' + context if context else ''} -- falling back to the "
        f"longest of {len(segs)} open segment(s).  Any boundary RMS/max "
        f"derived from it is unreliable (issue #33).",
        OpenLCFSContourWarning,
        stacklevel=2,
    )
    return max(segs, key=len)


class DerivativeSanityWarning(UserWarning):
    """Data-sanity warning category for :func:`pchip_derivative`.

    Raised (as a warning, or as ``ValueError`` when ``strict=True``) for
    conditions that are not fatal to computing a derivative but are strong
    signals of upstream data problems: a corrupted/duplicated ``psi_N``
    grid, an unsorted grid, or a spike in the fitted derivative that looks
    like an axis-mapping artifact rather than real pedestal/edge physics.
    """


@contextlib.contextmanager
def capture_native_output(enabled=True):
    """Capture *all* stdout/stderr — both Python-level and OS-level.

    Two layers are needed to fully silence a TokaMaker solve:

    * :func:`contextlib.redirect_stdout`/``redirect_stderr`` catch Python
      ``print()`` (the homotopy / boundary / li-match diagnostics), which under
      a Jupyter kernel go straight to the cell via ``sys.stdout``, bypassing the
      file descriptors.
    * an ``os.dup2`` file-descriptor redirect catches output from the compiled
      extension (the Fortran/C solver: ``DLSODE``, ``gs_get_qprof`` warnings)
      that bypasses Python's ``sys.stdout``.

    Yields a dict whose ``"text"`` key holds the combined captured output once
    the block exits (also populated if the block raises). ``enabled=False`` is a
    no-op pass-through, so callers can gate on a ``verbose`` flag.
    """
    import io

    holder = {"text": ""}
    if not enabled:
        yield holder
        return
    sys.stdout.flush()
    sys.stderr.flush()
    tmp = tempfile.TemporaryFile(mode="w+b")
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    py_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(py_buf), contextlib.redirect_stderr(py_buf):
            yield holder
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        holder["text"] = py_buf.getvalue() + tmp.read().decode("utf-8", "replace")
        tmp.close()


# ====================================================================
#  Internal helpers
# ====================================================================

def safe_trace_surf(mygs, psi):
    r'''Snapshot/restore-wrapped trace_surf.

    The Fortran-side `trace_surf` has been observed to perturb subsequent
    `get_stats` / boundary measurements via mutation of state hanging off
    the `gs_equil` struct (mesh-cell caches, tracer step state, etc.).
    Pre-OFT-PR-248 this was hard to undo cleanly; with PR #248 we now have
    `mygs.copy_eq()` / `mygs.replace_eq()` which atomically swap the
    `gs_equil` pointer.  This wrapper snapshots the equilibrium before
    the trace, copies the returned LCFS into an owned numpy array, then
    restores the original equilibrium.

    Whatever state lives outside `gs_equil` (eg. module-level
    `active_tracer` in `tracing_2d`) is *not* restored — but that state
    is reset at the start of each `trace_surf` call anyway, so the only
    risk surface is mid-trace interaction with concurrent
    `get_stats`-like calls, which bouquet does not do.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ (`copy_eq` /
        `replace_eq` available).
    psi : float
        Normalized psi value to trace (eg. ``1.0 - psi_pad``).

    Returns
    -------
    numpy.ndarray or None
        ``(N, 2)`` array of (R, Z) points along the traced surface,
        owned by the caller (a copy, decoupled from any internal
        TokaMaker buffers).  Returns ``None`` if the trace fails.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        # legacy OFT build: fall through to bare trace_surf (no protection)
        result = mygs.trace_surf(psi)
        return None if result is None else np.asarray(result).copy()
    saved = mygs.copy_eq()
    try:
        result = mygs.trace_surf(psi)
        if result is not None:
            result = np.asarray(result).copy()
        return result
    finally:
        mygs.replace_eq(source_eq=saved)


def safe_save_eqdsk(mygs, filename, **kwargs):
    r'''Snapshot/restore-wrapped save_eqdsk.

    `mygs.save_eqdsk` internally re-runs the q-profile tracer (which
    sets `active_tracer` state and may mutate cached `<R>` / `<1/R>`
    geometry on the `gs_equil` struct) and has been empirically
    observed to shift `mygs.get_globals()[0]` (Ip integral) by ~0.5-
    0.8% when called against a converged equilibrium.  This wrapper
    snapshots the equilibrium via `mygs.copy_eq()` before the save,
    then restores via `mygs.replace_eq()` after -- so the .geqdsk
    file is written from the unmodified state AND downstream
    diagnostics see the same state recon converged to.

    On legacy OFT builds without `copy_eq` / `replace_eq` (pre-PR
    #248), falls through to a bare `mygs.save_eqdsk(...)` with no
    protection.

    Parameters
    ----------
    mygs : OpenFUSIONToolkit.TokaMaker.TokaMaker
        Active TokaMaker instance.  Requires PR #248+ for the
        protected snapshot/restore path.
    filename : str
        Same as `mygs.save_eqdsk` filename arg.
    **kwargs
        Passed through to `mygs.save_eqdsk(...)`.
    '''
    if not hasattr(mygs, 'copy_eq') or not hasattr(mygs, 'replace_eq'):
        return mygs.save_eqdsk(filename, **kwargs)
    saved = mygs.copy_eq()
    try:
        return mygs.save_eqdsk(filename, **kwargs)
    finally:
        mygs.replace_eq(source_eq=saved)


def pchip_derivative(x, y, x_eval=None, strict=False):
    r'''Analytic derivative dy/dx via PCHIP on the native (x, y) grid.

    Replaces the ``np.gradient(y) / np.gradient(x)`` central-difference
    pattern used throughout bouquet to build P'(psi_N) for TokaMaker's
    ``pp_prof``. A :class:`scipy.interpolate.PchipInterpolator` is fit to
    the *native* (unresampled) grid and differentiated analytically, which:

    * avoids the smearing a resample-then-``np.gradient`` pipeline
      introduces at sharp features (e.g. an H-mode pedestal), and
    * is shape-preserving / no-overshoot by construction -- monotone input
      data can never yield a derivative of the "wrong" sign, unlike an
      interpolating cubic spline (:class:`scipy.interpolate.CubicSpline`),
      which can ring on noisy data. On smooth (e.g. GPR) profiles PCHIP and
      CubicSpline derivatives are numerically indistinguishable, so this
      guranteed-safe behavior at sharp/noisy features is a pure win.

    Two tiers of data-sanity checks guard against silently producing a
    plausible-looking but wrong P' from corrupted input:

    **Tier 1 -- hard errors** (always ``raise ValueError``, regardless of
    *strict*):

    1. Non-finite (``NaN``/``inf``) values in *x* or *y*.
    2. Fewer than 2 unique *x* points after de-duplication (a degenerate
       grid cannot support a derivative -- this must not silently return
       zeros).
    3. Non-finite values in the computed derivative.
    4. Shape-guarantee tripwire: PCHIP's monotonicity invariant guarantees
       that monotone input data can never yield a derivative of the wrong
       sign. If the de-duplicated, sorted *y* is monotone (within
       ``rtol=1e-12*max|y|``) but the fitted derivative disagrees in sign
       beyond ``1e-12*max|secant slope|``, that invariant has been broken
       -- almost certainly by scrambled ``(x, y)`` pairing rather than a
       genuine numerical edge case -- so this raises rather than warns.

    **Tier 2 -- aggressive warnings** (:class:`DerivativeSanityWarning` by
    default; promoted to ``ValueError`` when ``strict=True``):

    5. Duplicate *x* points were dropped (count + first few duplicated
       locations reported) -- a duplicated ``psi_N`` grid is a suspected
       upstream cause of stepwise ``j_BS`` artifacts, so this is surfaced
       loudly rather than silently fixed.
    6. Input *x* was not sorted (ascending sort was applied).
    7. More than 5% of points were dropped as duplicates ("grid likely
       corrupted").
    8. Spike detector: the fitted derivative's peak magnitude exceeds
       ``50 *`` the median secant-slope magnitude (guarded on
       ``median > 0``) -- flagged as a possible unphysical feature (e.g.
       an axis-mapping artifact) rather than real edge/pedestal physics,
       with the spike's *x* location and the ratio reported.

    Parameters
    ----------
    x : array_like
        1-D abscissa (e.g. ``psi_N``). Need not be strictly increasing or
        unique on input -- see the duplicate-handling note below.
    y : array_like
        1-D ordinate (e.g. pressure), same length as *x*.
    x_eval : array_like or None, optional
        Points at which to evaluate the derivative. Defaults to *x* itself
        (i.e. the derivative is returned on the native grid).
    strict : bool, optional
        When ``True``, every Tier-2 condition that would normally emit a
        :class:`DerivativeSanityWarning` instead raises ``ValueError``.
        Default ``False`` (warn only); bouquet's call sites keep the
        default so a merely-suspicious profile doesn't abort a solve.

    Returns
    -------
    numpy.ndarray
        ``dy/dx`` evaluated at *x_eval* (or at the original, full-length
        *x* when *x_eval* is ``None`` -- so the returned array matches the
        input grid's shape even when duplicate points were dropped for
        fitting; a repeated abscissa simply gets the same derivative value
        at each of its occurrences).

    Notes
    -----
    ``PchipInterpolator`` requires strictly increasing abscissas.
    Duplicated or non-monotonic ``psi_N`` grid points are a suspected
    cause of stepwise ``j_BS`` artifacts downstream, so this guard is
    load-bearing, not cosmetic: repeated ``x`` values are collapsed via
    ``np.unique`` (keeping the first occurrence of each distinct value,
    then sorting), silently making the interpolant well-posed rather than
    raising on real-world profile data that occasionally carries a
    repeated grid point -- but see Tier 2 items 5-7 above: "silently"
    well-posed does not mean "silently" as far as the user is concerned.
    '''
    from scipy.interpolate import PchipInterpolator

    def _flag(msg):
        """Tier-2 report: warn (default) or raise (``strict=True``).

        ``stacklevel=3`` walks past this closure and ``pchip_derivative``
        itself so the warning is attributed to the code that called
        ``pchip_derivative``, not to a line inside this helper.
        """
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, DerivativeSanityWarning, stacklevel=3)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---- Tier 1.1: non-finite inputs -----------------------------------
    if not np.all(np.isfinite(x)):
        n_bad = int(np.count_nonzero(~np.isfinite(x)))
        raise ValueError(
            f"pchip_derivative: x contains {n_bad} non-finite value(s) "
            "(NaN/inf) -- cannot fit a PCHIP interpolant.")
    if not np.all(np.isfinite(y)):
        n_bad = int(np.count_nonzero(~np.isfinite(y)))
        raise ValueError(
            f"pchip_derivative: y contains {n_bad} non-finite value(s) "
            "(NaN/inf) -- cannot fit a PCHIP interpolant.")

    # ---- Tier 2.6: unsorted input ---------------------------------------
    if x.size >= 2 and np.any(np.diff(x) < 0):
        _flag(
            "pchip_derivative: input x is not sorted ascending; an "
            "ascending sort was applied before fitting. If this x grid "
            "(e.g. psi_N) is expected to be monotonic, check upstream "
            "for a scrambled profile.")

    # ---- de-duplicate + sort -------------------------------------------
    x_unique, idx, counts = np.unique(x, return_index=True, return_counts=True)
    y_unique = y[idx]
    n_total = x.size
    n_dropped = n_total - x_unique.size

    # ---- Tier 2.5 / 2.7: duplicate x points dropped ---------------------
    if n_dropped > 0:
        dup_locs = x_unique[counts > 1]
        loc_str = ", ".join(f"{v:.6g}" for v in dup_locs[:5])
        if dup_locs.size > 5:
            loc_str += f", ... ({dup_locs.size} total distinct dup locations)"
        frac_dropped = n_dropped / n_total
        if frac_dropped > 0.05:
            _flag(
                f"pchip_derivative: {n_dropped}/{n_total} points "
                f"({100*frac_dropped:.1f}%) dropped as duplicate x values "
                f"(at x = {loc_str}) -- grid likely corrupted. A duplicated "
                "psi_N grid is a suspected cause of stepwise j_BS "
                "artifacts; check the upstream profile/grid construction.")
        else:
            _flag(
                f"pchip_derivative: dropped {n_dropped} duplicate x "
                f"point(s) (at x = {loc_str}) before fitting. A duplicated "
                "psi_N grid is a suspected cause of stepwise j_BS "
                "artifacts; check the upstream profile/grid construction.")

    # ---- Tier 1.2: degenerate grid ---------------------------------------
    if x_unique.size < 2:
        raise ValueError(
            f"pchip_derivative: only {x_unique.size} unique x point(s) "
            "after de-duplication -- at least 2 are required to fit a "
            "derivative; refusing to silently return zeros.")

    interp = PchipInterpolator(x_unique, y_unique)
    deriv_unique = interp.derivative()(x_unique)   # for the sanity checks below
    x_eval = x if x_eval is None else np.asarray(x_eval, dtype=float)
    result = interp.derivative()(x_eval)

    # ---- Tier 1.3: non-finite output -------------------------------------
    if not np.all(np.isfinite(result)):
        n_bad = int(np.count_nonzero(~np.isfinite(result)))
        raise ValueError(
            f"pchip_derivative: computed derivative contains {n_bad} "
            "non-finite value(s) -- check the input profile for "
            "near-degenerate spacing or extreme dynamic range.")

    # ---- secant slopes on the native (unique, sorted) grid ---------------
    dx = np.diff(x_unique)
    dy = np.diff(y_unique)
    secants = dy / dx
    abs_secants = np.abs(secants)

    # ---- Tier 1.4: shape-guarantee tripwire ------------------------------
    max_y = np.max(np.abs(y_unique)) if y_unique.size else 0.0
    y_tol = 1e-12 * max_y
    max_secant = np.max(abs_secants) if abs_secants.size else 0.0
    slope_tol = 1e-12 * max_secant
    non_increasing = np.all(dy <= y_tol)
    non_decreasing = np.all(dy >= -y_tol)
    if non_increasing and not non_decreasing:
        bad = deriv_unique > slope_tol
        if np.any(bad):
            raise ValueError(
                "pchip_derivative: internal invariant violated -- y is "
                "monotone non-increasing but the PCHIP derivative has "
                f"{int(np.count_nonzero(bad))} positive-sign entrie(s) "
                "beyond tolerance. PCHIP's shape-preservation guarantee "
                "should make this impossible; this likely indicates "
                "scrambled (x, y) pairing rather than a numerical edge "
                "case.")
    elif non_decreasing and not non_increasing:
        bad = deriv_unique < -slope_tol
        if np.any(bad):
            raise ValueError(
                "pchip_derivative: internal invariant violated -- y is "
                "monotone non-decreasing but the PCHIP derivative has "
                f"{int(np.count_nonzero(bad))} negative-sign entrie(s) "
                "beyond tolerance. PCHIP's shape-preservation guarantee "
                "should make this impossible; this likely indicates "
                "scrambled (x, y) pairing rather than a numerical edge "
                "case.")

    # ---- Tier 2.8: spike detector -----------------------------------------
    median_secant = np.median(abs_secants) if abs_secants.size else 0.0
    if median_secant > 0:
        peak_idx = int(np.argmax(np.abs(deriv_unique)))
        peak_val = float(np.abs(deriv_unique[peak_idx]))
        ratio = peak_val / median_secant
        if ratio > 50.0:
            _flag(
                f"pchip_derivative: derivative peak |dy/dx|={peak_val:.4g} "
                f"at x={x_unique[peak_idx]:.6g} is {ratio:.1f}x the median "
                f"secant-slope magnitude ({median_secant:.4g}) -- possible "
                "unphysical feature in input data (e.g. an axis mapping "
                "artifact) rather than real profile physics.")

    return result


# =====================================================================
#  Flux-surface-averaged plasma-current integral
# =====================================================================
#
# ``TokaMaker.compute_flux_integral`` is NOT ``int_plasma f dA``.  Measured on
# the synthetic D3D-like example (see ``tests/test_fsa_current_integral.py``):
#
#   * it integrates over the whole ``reg == 1`` (limiter) region, and the
#     flux-function interpolator returns the profile's EDGE value everywhere
#     outside the LCFS (``gs_prof_interp_apply`` CASE(4) returns 0 -- the LCFS
#     end of the internal psi coordinate -- off the plasma, and ``gs_flux_int``
#     then evaluates the profile there).  ``compute_flux_integral(1.0)`` is
#     therefore 2.83853 m^2, the LIMITER-region area, against a true plasma
#     cross-section of 1.79005 m^2;
#   * so for a profile with a finite edge value the excess area is charged at
#     ``f(psi_N=1)``.  On the archived total that is
#     ``1.36e5 A/m^2 * 1.05 m^2 = 1.43e5 A``, i.e. +11.9 % of I_p -- almost the
#     whole of the +12.9 % "representation bias" 7dc254b calibrated away.
#
# The measure below never uses the mesh integral.  It is the textbook
# axisymmetric current integral,
#
#     I_p = int j_phi dA = int dpsi (V'/2pi) <j_phi/R>,
#
# with ``V' = dV/dpsi`` and the flux-surface averages taken from ``get_q``.
#
# CONVENTION.  Two readings of a bouquet current array are in play and they
# differ by O(1 %) on a DIII-D-like case:
#
#   ``"fsa"``           J = <j_phi/R>/<1/R>, the FSA toroidal density
#                       => I_p = int dpsi (V'/2pi) <1/R> J
#   ``"jphi-linterp"``  J = <R> P' + <1/R> FF'/mu0, TokaMaker's own
#                       ``jphi-linterp`` profile variable (``jphi_update`` in
#                       ``grad_shaf_prof_phys.F90``), i.e. j_phi evaluated at
#                       the surface's mean major radius
#                       => FF'/mu0 = (J - <R> P')/<1/R> and hence
#                          I_p = int dpsi (V'/2pi) [ J <1/R^2>/<1/R>
#                                                    + P' (1 - <R><1/R^2>/<1/R>) ]
#
# bouquet's arrays are ``jphi-linterp`` values -- that is the variable they are
# handed to ``set_profiles`` as, and the variable the reconstruction fits.  On
# the D3D-like anchor the ``"jphi-linterp"`` reading recovers the true I_p of
# the archived total to +0.072 %, against +0.927 % for ``"fsa"``, and it agrees
# to 0.011 % with the solver's own internal renormalisation factor (measured
# independently as ``<w J_eq>/<w J_archived>``).  Both readings integrate the
# equilibrium's OWN profile to its true I_p to better than 0.01 %, which is the
# validation that the V'/<1/R> plumbing and the psi_N -> psi Jacobian are right.

_FSA_PSI_PAD = 1.0e-3
_FSA_MU0 = 4.0e-7 * np.pi


def fsa_current_geometry(eq, psi_N, psi_pad=_FSA_PSI_PAD, want_pprime=True):
    r"""Per-surface FSA geometry for :func:`Ip_fsa_integral`, from ``get_q``.

    *eq* is anything exposing TokaMaker's ``get_q`` / ``get_profiles`` /
    ``psi_bounds`` -- a live ``TokaMaker`` or a ``copy_eq()`` snapshot
    (``TokaMaker_equilibrium``); both were verified to return bit-identical
    ``ravgs`` for the same state, and evaluating them on a snapshot does not
    perturb the live solver.

    Two traps this function exists to close:

    * **exact endpoints silently collapse ``get_q``.**  ``get_q(psi=...)`` with
      ``psi_N`` containing 0.0 (the magnetic axis) returns the AXIS values on
      *every* surface -- ``<R>`` constant to 2e-15, ``dV/dPsi`` constant --
      with no error and no warning; the surface tracer starts from the axis and
      never leaves it.  The sampling grid is therefore clipped to
      ``[psi_pad, 1 - psi_pad]`` and the collapse is asserted against.
    * **``dV/dPsi`` is per DIMENSIONAL psi.**  ``int dV/dPsi dpsi`` recovers
      ``get_stats()['vol']`` to -0.25 % (edge/axis truncation) while the
      ``dpsi_N`` reading is out by +291 %.  The Jacobian is
      ``|psi_bounds[1] - psi_bounds[0]|``, applied here once.

    Returns a dict with ``psi_N``, ``R_avg``, ``inv_R``, ``inv_R2`` (``None``
    on OFT builds whose ``ravgs`` is the legacy 3-entry positional array),
    ``dV_dpsi`` (magnitude), ``dpsi_dpsiN``, ``pprime`` (``None`` if not
    requested) and ``dA_dpsiN = (V'/2pi) <1/R> |dpsi/dpsi_N|``.
    """
    from .physics import q_ravg

    psi_N = np.asarray(psi_N, dtype=float)
    psi_q = np.ascontiguousarray(
        np.clip(psi_N, float(psi_pad), 1.0 - float(psi_pad)), dtype=float)
    ravgs = eq.get_q(psi=psi_q)[2]
    R_avg = np.asarray(q_ravg(ravgs, "<R>"), dtype=float)
    inv_R = np.asarray(q_ravg(ravgs, "<1/R>"), dtype=float)
    dV_dpsi = np.abs(np.asarray(q_ravg(ravgs, "dV/dPsi"), dtype=float))
    inv_R2 = None
    if isinstance(ravgs, dict) and "<1/R^2>" in ravgs:
        inv_R2 = np.asarray(ravgs["<1/R^2>"], dtype=float)

    if R_avg.size > 1 and np.ptp(R_avg) <= 1.0e-9 * float(np.mean(R_avg)):
        raise RuntimeError(
            "fsa_current_geometry: get_q returned a CONSTANT <R> across all "
            f"{R_avg.size} surfaces ({float(R_avg[0]):.6f} m) -- the surface "
            "tracer collapsed onto the magnetic axis.  This is the silent "
            "failure mode triggered by sampling psi_N = 0 exactly; raise "
            "psi_pad.")
    if not (np.all(np.isfinite(R_avg)) and np.all(np.isfinite(inv_R))
            and np.all(np.isfinite(dV_dpsi))):
        raise RuntimeError("fsa_current_geometry: get_q returned non-finite "
                           "flux-surface averages")

    bounds = np.asarray(eq.psi_bounds, dtype=float)
    dpsi_dpsiN = abs(float(bounds[1]) - float(bounds[0]))
    if not np.isfinite(dpsi_dpsiN) or dpsi_dpsiN <= 0.0:
        raise RuntimeError(f"fsa_current_geometry: bad psi_bounds {bounds}")

    if inv_R2 is not None and not np.all(np.isfinite(inv_R2)):
        raise RuntimeError("fsa_current_geometry: get_q returned non-finite "
                           "<1/R^2> averages")

    pprime = None
    if want_pprime:
        prof = eq.get_profiles(psi=psi_q.copy())
        pprime = np.asarray(prof[4], dtype=float)
        if not np.all(np.isfinite(pprime)):
            raise RuntimeError("fsa_current_geometry: get_profiles returned "
                               "non-finite P' -- a NaN here would poison the "
                               "affine c term and pass every downstream "
                               "threshold gate silently")

    return {
        "psi_N": psi_N,
        "psi_q": psi_q,
        "R_avg": R_avg,
        "inv_R": inv_R,
        "inv_R2": inv_R2,
        "dV_dpsi": dV_dpsi,
        "dpsi_dpsiN": dpsi_dpsiN,
        "pprime": pprime,
        "dA_dpsiN": dV_dpsi / (2.0 * np.pi) * inv_R * dpsi_dpsiN,
    }


def Ip_fsa_weights(geom, convention="jphi-linterp", pprime_sign=1.0):
    r"""``(w, c)`` such that ``I_p[J] = trapezoid(w * J, psi_N) + c``.

    The measure is affine, not linear: in the ``jphi-linterp`` convention the
    ``P'`` term of the Grad-Shafranov source is independent of *J* and lands in
    the constant *c* (it is -3.3 % of I_p on the D3D-like anchor -- large
    enough that dropping it costs +3.4 %).  In the ``fsa`` convention ``c`` is
    exactly 0.

    ``pprime_sign`` flips ``get_profiles``' ``P'`` if the case's flux/current
    sign convention needs it; :class:`~bouquet.TokaMaker_interface._AnchorIpRenorm`
    determines it from the anchor rather than assuming.
    """
    g = geom["dV_dpsi"] / (2.0 * np.pi) * geom["dpsi_dpsiN"]
    if convention == "fsa":
        return g * geom["inv_R"], 0.0
    if convention != "jphi-linterp":
        raise ValueError(f"unknown convention {convention!r} "
                         "(want 'jphi-linterp' or 'fsa')")
    inv_R2 = geom["inv_R2"]
    if inv_R2 is None:
        raise ValueError(
            "the 'jphi-linterp' measure needs <1/R^2>, which this OFT build's "
            "get_q does not return (legacy positional ravgs).  Use "
            "convention='fsa' on that build.")
    pprime = geom["pprime"]
    if pprime is None:
        raise ValueError("the 'jphi-linterp' measure needs P'; rebuild the "
                         "geometry with want_pprime=True")
    ratio = inv_R2 / geom["inv_R"]
    w = g * ratio
    c = float(_trapezoid(g * float(pprime_sign) * pprime
                         * (1.0 - geom["R_avg"] * ratio), geom["psi_N"]))
    return w, c


def Ip_fsa_affine_profile(geom, convention="jphi-linterp", pprime_sign=1.0):
    r"""The affine ``P'`` term of :func:`Ip_fsa_weights`, CUMULATIVELY.

    :func:`Ip_fsa_weights` returns the affine term as the single number ``c``
    -- its contribution to the TOTAL plasma current.  The l_i closure needs the
    same term as a function of radius, because it works with the ENCLOSED
    current

    .. math:: I(\psi_N) = \int_0^{\psi_N} w_{\rm lin} J \,{\rm d}\psi_N'
              + c(\psi_N),

    so this returns ``c(psi_N)`` on ``geom["psi_N"]``, zero at the first sample
    and equal to :func:`Ip_fsa_weights`' ``c`` (to rounding) at the last.  The
    integrand is character-for-character the one ``Ip_fsa_weights`` integrates;
    that function is deliberately NOT refactored to call this one, so the
    scalar ``c`` every existing closure depends on keeps its exact
    floating-point value.  ``convention="fsa"`` has no affine term and returns
    zeros.
    """
    from scipy.integrate import cumulative_trapezoid

    psi = np.asarray(geom["psi_N"], dtype=float)
    if convention == "fsa":
        return np.zeros_like(psi)
    if convention != "jphi-linterp":
        raise ValueError(f"unknown convention {convention!r} "
                         "(want 'jphi-linterp' or 'fsa')")
    inv_R2 = geom["inv_R2"]
    if inv_R2 is None:
        raise ValueError(
            "the 'jphi-linterp' measure needs <1/R^2>, which this OFT build's "
            "get_q does not return (legacy positional ravgs).  Use "
            "convention='fsa' on that build.")
    pprime = geom["pprime"]
    if pprime is None:
        raise ValueError("the 'jphi-linterp' measure needs P'; rebuild the "
                         "geometry with want_pprime=True")
    g = geom["dV_dpsi"] / (2.0 * np.pi) * geom["dpsi_dpsiN"]
    ratio = inv_R2 / geom["inv_R"]
    return cumulative_trapezoid(
        g * float(pprime_sign) * pprime * (1.0 - geom["R_avg"] * ratio),
        psi, initial=0.0)


def _trapezoid(y, x):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(y, dtype=float),
                           np.asarray(x, dtype=float)))


def closure_sign_convention(ip_ind, ip_bs, ip_fix, c_affine, Ip_abs):
    """Sign-consistent ``(target, affine term)`` pair for :func:`close_ip`.

    ``close_ip`` solves an AFFINE identity, so the P'-term constant has to
    carry the same current-direction convention as the target and the linear
    parts -- that is the contract
    ``test_negative_current_convention_closes_too`` states, by negating all
    three together.

    The production caller cannot read that convention off ``c_affine``: the
    anchor equilibrium is always solved to ``abs(Ip)``, so ``c_affine``
    comes back in the POSITIVE orientation whatever the data does, while the
    ``ip_*`` linear parts carry the data's own sign.  Pairing them means
    taking the sign from the linear total (NOT from an affine integral,
    whose ``c`` could out-vote the linear part on a low-current slice) and
    applying it to the target AND the constant.

    Without this, negative-current data closes against ``+c`` instead of
    ``-c`` and the scales come out wrong by ``2c`` -- ~6 % of Ip at the
    measured c/Ip ~ 3 % -- and the post-closure self-check cannot catch it,
    because it reuses the same unpaired ``c`` and is satisfied identically.

    Returns ``(sgn, Ip_target_signed, c_signed)``.  For the ordinary
    positive convention ``sgn`` is ``+1`` and both values pass through
    unchanged.
    """
    _lin_total = float(ip_ind) + float(ip_bs) + float(ip_fix)
    if not np.isfinite(_lin_total):
        raise RuntimeError(
            "closure_sign_convention: the linear component total is "
            "non-finite; the current-direction convention cannot be read "
            "from it")
    sgn = float(np.sign(_lin_total) or 1.0)
    return sgn, sgn * abs(float(Ip_abs)), sgn * float(c_affine)


def close_ip(channel, Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
             scale_bounds=(0.2, 5.0)):
    """Channel scales closing the affine Ip measure on a hybrid baseline.

    Solves ``s_ohm*ip_ind + s_bs*ip_bs + ip_fix + c = Ip`` for the one free
    scale, where ``ip_*`` are the LINEAR parts (``trapezoid(w * j)``) of each
    component under :func:`Ip_fsa_weights` and ``c`` is the affine P' term
    carried exactly once.  ``channel`` picks which component absorbs the
    deficit: ``"bootstrap"`` keeps ``s_ohm = 1`` and rescales the bootstrap;
    ``"ohmic"`` keeps ``s_bs = 1`` and rescales the inductive.  Returns
    ``(ohm_scale, bs_scale)``.

    ``Ip_target_signed`` and ``c_affine`` must carry the SAME
    current-direction convention as the ``ip_*`` linear parts; see
    :func:`closure_sign_convention`, which is how the production caller
    pairs them.

    Refuses (``RuntimeError``) when the target is non-finite or its
    magnitude is zero (there is nothing to close onto, and the relative
    zero-divisor guards below would degenerate), when the rescaled
    component's linear part is ~0 relative to the target (the closure would
    be a division by noise), or when the resulting scale falls outside
    ``scale_bounds`` -- the hybrid components then simply do not add up to
    Ip and hiding that behind a rescale would be a lie.  The bounds are
    STRICTLY exclusive: a scale of exactly ``lo`` or ``hi`` is refused.
    Unknown channels raise ``ValueError``.
    """
    if not np.isfinite(float(Ip_target_signed)):
        raise RuntimeError(
            f"close_ip: Ip target is non-finite ({Ip_target_signed!r}); "
            "the measure or the baseline carries bad values, refusing to "
            "close on it")
    Ip_t = abs(float(Ip_target_signed))
    if Ip_t <= 0.0:
        raise RuntimeError(
            "close_ip: |Ip target| is zero; there is no current to close "
            "the hybrid components onto")
    if not (np.isfinite(float(c_affine)) and np.isfinite(float(ip_ind))
            and np.isfinite(float(ip_bs)) and np.isfinite(float(ip_fix))):
        raise RuntimeError(
            "close_ip: a non-finite affine term or component integral "
            "(c_affine/ip_ind/ip_bs/ip_fix); refusing to close on it")
    deficit = float(Ip_target_signed) - float(c_affine) - float(ip_fix)
    if channel == "bootstrap":
        if abs(ip_bs) < 1e-6 * Ip_t:
            raise RuntimeError("close_ip/bootstrap channel: j_BS integrates "
                               "to ~0; cannot close Ip on it")
        scales = (1.0, (deficit - float(ip_ind)) / float(ip_bs))
    elif channel == "ohmic":
        if abs(ip_ind) < 1e-6 * Ip_t:
            raise RuntimeError("close_ip/ohmic channel: j_inductive "
                               "integrates to ~0; cannot close Ip on it")
        scales = ((deficit - float(ip_bs)) / float(ip_ind), 1.0)
    else:
        raise ValueError(f"unknown closure_channel {channel!r} "
                         "(expected 'ohmic' or 'bootstrap')")
    lo, hi = scale_bounds
    for name, s in zip(("ohm_scale", "bs_scale"), scales):
        if not (lo < s < hi):
            # The test is strict on both ends, so the message says so: the
            # accepted set is the OPEN interval, and a scale of exactly lo or
            # hi is refused.
            raise RuntimeError(
                f"close_ip/{channel} channel: {name} {s:.3f} is outside "
                f"({lo:g}, {hi:g}) -- the hybrid components do not add up "
                "to Ip; refusing to hide that behind a rescale")
    return scales


def unrenormalise_q0(q0_anchor, j_achieved0, j_requested0):
    r"""The anchor's q0 mapped back onto the source's OWN current.

    ``solve_jphi`` hands TokaMaker a jphi-linterp *shape* and TokaMaker
    renormalises it to ``Ip_target``.  When the source's total does not itself
    carry Ip (FUSE's ``core_profiles`` total reads -3.89 % on the reference
    validation slice) the anchor equilibrium therefore sits at a current the
    source never claimed, and its ``q0`` with it.  Undo that to the same first
    order the whole predictor runs on -- ``q0 ~ 1/j_phi(0)`` at frozen
    geometry:

    .. math:: q_{0,\mathrm{target}} = q_{0,\mathrm{anchor}}
              \; j_{\mathrm{achieved}}(0) / j_{\mathrm{requested}}(0)

    where *j_achieved0* is the anchor's GS-reconstructed own axis current
    (:func:`eq_jphi_profile`, which round-trips to its achieved Ip) and
    *j_requested0* the source total at the same psi_pad-clipped axis sample.

    A named function rather than three inline characters because this is the
    one place the Ip-deficit artefact is deliberately NOT propagated into the
    current split: every other channel absorbs that deficit into a single
    scale and leaves the shape alone, and this keeps the q0 channel consistent
    with them.  Raises ``RuntimeError`` on a zero or non-finite requested axis
    current, where the ratio is meaningless.
    """
    q0_anchor = float(q0_anchor)
    j_a, j_r = float(j_achieved0), float(j_requested0)
    if not (np.isfinite(q0_anchor) and np.isfinite(j_a) and np.isfinite(j_r)):
        raise RuntimeError("unrenormalise_q0: non-finite input "
                           f"(q0_anchor={q0_anchor!r}, j_achieved0={j_a!r}, "
                           f"j_requested0={j_r!r})")
    if j_r == 0.0:
        raise RuntimeError("unrenormalise_q0: the source total has zero axis "
                           "current density; q0 cannot be un-renormalised "
                           "onto it")
    return q0_anchor * (j_a / j_r)


def q0_gate_admits(sawtooth_active, q0_dd, q0_target, q0_gate):
    """``(admitted, basis)`` for the ``"sawtooth_bootstrap"`` gate.

    The q0 pin is well-founded where sawteeth justify it: admitted when the
    source's sawtooth model is ACTIVE at the slice, or when the source's OWN
    axis safety factor ``|q0_dd|`` is at/below ``q0_gate``.  The comparison is
    on the source's ``q0_dd`` -- the physically clamped value -- and not on
    ``q0_target`` (the TokaMaker-estimator mapping of it): the estimator
    reads systematically lower, and gating on it admitted idle-sawtooth
    ramp slices whose own ``q0_dd`` (1.10-1.19) sat above the threshold.
    ``q0_target`` is used only when the source carries no axis q at all,
    and the returned ``basis`` says which was used.  Magnitudes throughout:
    q carries a COCOS sign and a negative value would pass ``<= gate``
    trivially.
    """
    if sawtooth_active:
        return True, "sawtooth active"
    if q0_dd is not None and np.isfinite(q0_dd):
        return bool(abs(float(q0_dd)) <= float(q0_gate)), "|q0_dd|"
    if q0_target is not None and np.isfinite(q0_target):
        return bool(abs(float(q0_target)) <= float(q0_gate)), \
            "|q0_target| (source carries no axis q)"
    return False, "no q0 available"


#: post-closure round-trip budget [% of the gate's reference].  It is a
#: MACHINE-PRECISION budget on the assembly of the closed hybrid, not a
#: physics acceptance -- see :func:`ip_roundtrip_gate`.
IP_ROUNDTRIP_TOL_PCT = 0.05


def ip_roundtrip_gate(ip_closed, Ip_measured, posterior=None, sigma_Ip=None,
                      tol_pct=IP_ROUNDTRIP_TOL_PCT):
    r"""Does the ASSEMBLED hybrid carry the current its closure said it would?

    The ohmic-mode closure returns multiplier profiles (or scalars); the caller
    multiplies them onto the components and adds them up.  This gate checks
    THAT assembly -- a wrong component, a double-counted affine term, a
    misapplied sign -- by re-integrating the assembled profile and comparing it
    with what the closure itself produced.  *tol_pct* is the budget for that
    arithmetic and has never been a statement about the data.

    **The reference.**  On every HARD channel the closure's own Ip *is* the
    measurement (Ip is imposed exactly), so *posterior* is ``None``, the
    reference is ``Ip_measured`` and this is the gate it always was.  On the
    SOFT structured channel Ip is a Gaussian MEASUREMENT with ``sigma_Ip``, so
    the posterior mode deliberately lands a little off it (0.02-0.1 % at
    ``sigma_Ip`` = 0.5 % of Ip).  Pass that posterior and the algebra is
    checked against it; the distance from the measurement is *reported*
    (``measured_residual_pct``, ``residual_sigma_Ip``) and, past 1 sigma_Ip,
    flagged by :func:`closure_health` -- never refused, never retried.

    Comparing a posterior against the measurement with this budget refused ~40
    of 335 campaign slices for doing exactly what the soft channel is for
    (2026-09-16).  The fix is the reference; *tol_pct* is untouched.

    ``measured_residual_pct`` is the POSTERIOR's distance from the measurement
    (what the closure chose); ``residual_sigma_Ip`` is the ACHIEVED one -- the
    delivered hybrid's own Ip against the measurement, in sigma units.  They
    differ only by the round-trip itself.

    :raises RuntimeError: round trip outside *tol_pct* of the reference, or
        not finite.  ``abs(nan) > tol`` is ``False``, so a NaN in the assembled
        profile -- exactly the class of defect this gate exists to catch --
        used to pass it silently and be recorded as ``err_pct = nan``.  A gate
        that cannot read its input has not been passed.
    """
    ip_closed = abs(float(ip_closed))
    Ip_meas = abs(float(Ip_measured))
    if not np.isfinite(ip_closed):
        raise RuntimeError(
            f"ohmic mode: the closed hybrid integrates to {ip_closed} -- the "
            "assembled profile is not finite, refusing")
    if not (np.isfinite(Ip_meas) and Ip_meas > 0.0):
        raise RuntimeError(
            f"ohmic mode: Ip_measured is {Ip_measured!r} -- the round-trip "
            "gate has no usable reference, refusing")
    ref, ref_name = Ip_meas, "Ip_target"
    if posterior is not None and np.isfinite(float(posterior)) \
            and abs(float(posterior)) > 0.0:
        ref, ref_name = abs(float(posterior)), "the closure's own posterior Ip"
    err_pct = 100.0 * (ip_closed - ref) / ref
    # Non-finite is a refusal, not a pass: abs(nan) > tol is False, so the
    # profile this gate exists to catch would otherwise sail straight through
    # it (the local check this function replaces already refused it).
    if not np.isfinite(err_pct) or abs(err_pct) > float(tol_pct):
        raise RuntimeError(
            f"ohmic mode: closed hybrid integrates to {err_pct:+.3f}% "
            f"of {ref_name} after closure -- algebra error, refusing")
    return dict(
        err_pct=float(err_pct), reference=float(ref), reference_name=ref_name,
        measured_residual_pct=100.0 * (ref - Ip_meas) / Ip_meas,
        residual_sigma_Ip=(None if not sigma_Ip
                           else (ip_closed - Ip_meas) / float(sigma_Ip)))


#: closure-health reason text for a soft-channel Ip that ended up further than
#: 1 sigma_Ip from the measurement.  Named so the predictor-stage flag and the
#: corrector's refreshed one can be recognised and replaced rather than
#: accumulated (see :meth:`bouquet.run.Bouquet._close_ip_structured_corrector`).
SOFT_IP_FLAG_PREFIX = "soft Ip beyond 1 sigma_Ip"


def closure_health(ohm_scale, bs_scale, Ip_target_signed, c_affine,
                   ip_ind, ip_bs, ip_fix,
                   mismatch_max_pct=10.0, bs_scale_min=0.5,
                   soft_ip_residual_sigma=None):
    """Per-slice closure-health record for every ohmic-mode channel.

    ``Ip_target_signed`` and ``c_affine`` must carry the SAME
    current-direction convention as the ``ip_*`` linear parts -- the pair
    :func:`closure_sign_convention` returns -- or the raw mismatch below
    is off by ``2c`` on negative-current data, exactly as in
    :func:`close_ip`.

    Ip conservation only says the hybrid components' INTEGRAL is off; a
    single rescale cannot say where.  So record how far the raw (unscaled)
    components miss Ip, the unscaled and closed bootstrap fractions, and flag
    the slice **closure-limited** when the reconciliation asked of one scale
    is large: raw mismatch beyond ``mismatch_max_pct`` of Ip, or the bootstrap
    scaled below ``bs_scale_min``.  A refusal (scale outside (0.2, 5), or a
    singular q0 system) is closure-limited by construction and raises before
    this is reached.  Downstream consumers (Delta' pipelines) should treat
    closure-limited slices as unvalidated regardless of channel -- that is
    the honest boundary of what two scale factors can do.

    *soft_ip_residual_sigma* is the SOFT structured channel's ACHIEVED Ip
    residual against the MEASUREMENT, in units of ``sigma_Ip`` (``None`` on
    every other channel, where Ip is imposed exactly and there is nothing to
    say).  The soft channel's posterior Ip differs from the measurement by
    design, so a small offset is the channel working; beyond 1 sigma_Ip the
    slice is FLAGGED here -- never retried and never refused, exactly like a
    missed hard l_i row.

    **Non-finite inputs are flagged, not ignored.**  Every test here is of the
    form ``abs(x) > threshold``, which is ``False`` for a NaN, so a NaN scale
    or a NaN component integral used to give ``closure_limited = False`` with
    ``f_BS_closed = nan`` and an empty reason tuple -- a clean verdict on an
    unreadable slice.  A quantity this record cannot read is now its own named
    reason.  Flag, not refusal: the caller decides what to do with a slice it
    could not measure, and every other verdict in this record is a flag too.
    """
    Ip_t = abs(float(Ip_target_signed))
    raw = float(ip_ind) + float(ip_bs) + float(ip_fix) + float(c_affine)
    mismatch_pct = 100.0 * (abs(raw) - Ip_t) / Ip_t
    f_bs_unscaled = abs(float(ip_bs)) / Ip_t
    f_bs_closed = abs(float(bs_scale) * float(ip_bs)) / Ip_t
    reasons = []
    _nonfinite = [nm for nm, v in (("ohm_scale", ohm_scale),
                                   ("bs_scale", bs_scale),
                                   ("Ip_target", Ip_target_signed),
                                   ("c_affine", c_affine),
                                   ("ip_ind", ip_ind), ("ip_bs", ip_bs),
                                   ("ip_fix", ip_fix))
                  if not np.isfinite(float(v))]
    if _nonfinite or not np.isfinite(mismatch_pct):
        reasons.append("closure health is unreadable: non-finite "
                       + ", ".join(_nonfinite or ["Ip mismatch"]))
    if abs(mismatch_pct) > float(mismatch_max_pct):
        reasons.append(f"raw components miss Ip by {mismatch_pct:+.1f}% "
                       f"(> {float(mismatch_max_pct):g}%)")
    if float(bs_scale) < float(bs_scale_min):
        reasons.append(f"bs_scale {float(bs_scale):.3f} < {float(bs_scale_min):g}")
    if (soft_ip_residual_sigma is not None
            and np.isfinite(float(soft_ip_residual_sigma))
            and abs(float(soft_ip_residual_sigma)) > 1.0):
        reasons.append(SOFT_IP_FLAG_PREFIX
                       + f" (z_Ip = {float(soft_ip_residual_sigma):+.2f})")
    return dict(
        raw_components_ip_mismatch_pct=float(mismatch_pct),
        f_BS_unscaled=float(f_bs_unscaled),
        f_BS_closed=float(f_bs_closed),
        closure_limited=bool(reasons),
        closure_limited_reasons=tuple(reasons),
        closure_limited_thresholds=dict(mismatch_max_pct=float(mismatch_max_pct),
                                        bs_scale_min=float(bs_scale_min)),
    )


def warn_deprecated_channel(channel):
    """DeprecationWarning for ``closure_channel="ohmic"`` -- diagnostic only.

    Kept selectable for bracketing studies, but never for production or for
    anything that feeds a stability code: rescaling j_inductive alone hollows
    the core, lifts q0 well above the source's (1.3-2.6 measured), loses the
    q=1 surface on most sawtoothing slices, and produces implausible Delta'.
    Use ``"sawtooth_bootstrap"`` (sawtoothing discharges) or ``"bootstrap"``.
    """
    if str(channel) == "ohmic":
        import warnings
        msg = ("closure_channel='ohmic' is DEPRECATED (diagnostic bracket "
               "only): rescaling j_inductive alone hollows the core and lifts "
               "q0 far above the source's, drops the q=1 surface and gives "
               "implausible Delta'. Use 'sawtooth_bootstrap' (sawtoothing "
               "discharges) or 'bootstrap'. Do not feed its output to GPEC.")
        print(f"[imas SWB-split:ohmic] WARNING: {msg}", flush=True)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)


def close_ip_q0(Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
                j_ind0, j_bs0, j_fix0, j_ref0,
                scale_bounds=(0.2, 5.0), det_rtol=1e-6):
    r"""Both hybrid scales from Ip **and** an on-axis-current (q0) constraint.

    The ``"sawtooth_bootstrap"`` channel's predictor.  Two unknowns
    (``s_ohm``, ``s_bs``), two targets, one 2x2 linear system and **no GS
    solve**:

    .. code-block:: text

        s_ohm*j_ind0 + s_bs*j_bs0 = j_ref0 - j_fix0      (axis current -> q0)
        s_ohm*ip_ind + s_bs*ip_bs = Ip_signed - c - ip_fix   (exact Ip, affine)

    The first row is the q0 constraint *linearised*: at frozen anchor geometry
    (kappa0, B0, R0 fixed with the snapshot) the on-axis safety factor is
    ``q0 = 2 B0 (1 + kappa0^2) / (2 kappa0 mu0 R0 j0)``, i.e. ``q0 ~ 1/j0``, so
    matching q0 to the reference equilibrium is -- to first order -- matching
    its on-axis toroidal current density ``j_ref0``.  All the ``j*0`` are the
    profile values at the SAME psi_N sample, the psi_pad-clipped axis of
    :func:`fsa_current_geometry` (never psi_N = 0 exactly; see that docstring's
    collapse trap).  The second row is untouched from :func:`close_ip`: the
    LINEAR parts of the affine FSA measure with the P' term ``c`` carried once,
    so Ip stays exact by construction wherever this succeeds.

    Degenerate by design where the recomputed bootstrap has no core content:
    ``j_bs0 ~ 0`` makes row 1 pin ``s_ohm = (j_ref0 - j_fix0)/j_ind0`` (~1 when
    the source's own axis current is reproduced) and row 2 hand the whole Ip
    deficit to ``s_bs`` -- the channel reduces to ``"bootstrap"`` at zero cost.
    That is not the singular case; the singular case is the two rows becoming
    proportional, which is refused against a RELATIVE determinant floor
    (``det_rtol``) rather than an absolute one, because ``ip_*`` are amps and
    ``j*0`` are A/m^2 and no absolute epsilon is meaningful across both.

    Raises ``RuntimeError`` on a singular system, on non-finite inputs, or when
    either scale leaves ``scale_bounds`` -- the three sources then simply do not
    admit a common (Ip, q0) solution, which is a finding to report, not
    something to hide behind a rescale.  The bounds are STRICTLY exclusive, as
    in :func:`close_ip`: a scale of exactly ``lo`` or ``hi`` is refused.
    Returns ``(ohm_scale, bs_scale)``.
    """
    vals = (Ip_target_signed, c_affine, ip_ind, ip_bs, ip_fix,
            j_ind0, j_bs0, j_fix0, j_ref0)
    if not all(np.isfinite(float(v)) for v in vals):
        raise RuntimeError("close_ip_q0: non-finite input "
                           f"{tuple(float(v) for v in vals)}")
    b_axis = float(j_ref0) - float(j_fix0)
    b_ip = float(Ip_target_signed) - float(c_affine) - float(ip_fix)
    det = float(j_ind0) * float(ip_bs) - float(j_bs0) * float(ip_ind)
    floor = det_rtol * max(abs(float(j_ind0)), abs(float(j_bs0))) \
        * max(abs(float(ip_ind)), abs(float(ip_bs)))
    if abs(det) <= floor:
        raise RuntimeError(
            f"close_ip_q0: singular 2x2 (det {det:.4e} <= relative floor "
            f"{floor:.4e}) -- the inductive and bootstrap components are "
            "proportional in (axis current, Ip); Ip and q0 cannot both be "
            "imposed on this split")
    ohm_scale = (b_axis * float(ip_bs) - b_ip * float(j_bs0)) / det
    bs_scale = (float(j_ind0) * b_ip - float(ip_ind) * b_axis) / det
    lo, hi = scale_bounds
    for name, s in (("ohm_scale", ohm_scale), ("bs_scale", bs_scale)):
        if not (lo < s < hi):
            raise RuntimeError(
                f"close_ip_q0: {name} {s:.3f} is outside ({lo:g}, {hi:g}) -- "
                "no (Ip, q0)-consistent split exists within the scale bounds; "
                "refusing to hide that behind a rescale")
    return ohm_scale, bs_scale


# ── closure_channel="structured": a minimal-norm, radially-structured closure ──
#
# The scalar channels ("bootstrap", "ohmic", "sawtooth_bootstrap") all answer
# the same question -- "by what ONE number do I multiply a component so the
# hybrid integrates to Ip?" -- and the MSE arbiter showed that question has no
# single right answer: on some discharges the measured pitch angles want LESS
# bootstrap in the CORE (s_ohm > 1 on the Ip-closed line), while on others they
# want MORE bootstrap in the PEDESTAL (s_ohm < 1).  A scalar cannot be both.
# So make the multiplier a smooth PROFILE and pick, among the profiles that
# close Ip exactly, the one that deviates least from "trust the sources"
# (s == 1) under a weighting that encodes where each source is trusted.
#
# Everything below is linear algebra on the SAME affine measure the scalar
# channels use, so the cost is one small dense solve and zero GS solves.

#: Shipped default basis: four peak-normalised Gaussians spanning core to
#: pedestal.  Peak-normalised (not area-normalised) so a coefficient reads
#: directly as "how much the multiplier moves away from 1 near this radius".
STRUCTURED_BASIS_DEFAULT = dict(kind="gaussian",
                                centres=(0.15, 0.45, 0.75, 0.95),
                                widths=(0.20, 0.20, 0.20, 0.20))

#: Shipped default trust weights (the physics prior).  Large W penalises
#: deviation from 1 -> "this source is trusted at this radius".  The inductive
#: current is the one FUSE actually diffuses with a transport model and is most
#: trustworthy on axis; the bootstrap is a local formula that is at its best in
#: the steep-gradient pedestal, and at its worst near the axis (where the
#: Sauter collisionality expansion and the vanishing trapped fraction make it
#: small and uncertain).  Hence the ladders run in opposite directions.
STRUCTURED_WEIGHTS_PHYSICS = dict(name="physics-prior",
                                  ind=(100.0, 10.0, 3.0, 1.0),
                                  bs=(1.0, 3.0, 10.0, 100.0))

def structured_default_weights(K):
    """The default trust weights for a basis of *K* functions.

    :data:`STRUCTURED_WEIGHTS_PHYSICS` is a ladder over the RADII of the
    shipped 4-Gaussian basis; it has no meaning on a basis of a different
    length, and pairing it with one raised
    ``"K basis functions but 4/4 ind/bs weights"`` -- including for the
    documented ``structured_basis={"kind": "constant"}`` one-liner, which is
    how the channel is meant to be collapsed back onto a single scalar pair.
    For any other K the default is therefore UNIFORM (no prior), and the
    recorded ``name`` says so, so a reader of the archive can never mistake it
    for the physics ladder.  Supplying ``structured_weights`` explicitly is
    unaffected.
    """
    K = int(K)
    n = len(STRUCTURED_WEIGHTS_PHYSICS["ind"])
    if K == n:
        return dict(STRUCTURED_WEIGHTS_PHYSICS)
    return dict(name=f"uniform (K={K}; the physics prior is a ladder over the "
                     f"{n}-function default basis's radii and does not apply "
                     "to this basis)",
                ind=(1.0,) * K, bs=(1.0,) * K)


#: The ONE documented alternative, for a sensitivity: no prior at all, every
#: coefficient penalised equally.  The difference between the two answers is
#: the part of the result that the physics prior -- not the data -- is holding
#: up, and it is meant to be reported, not hidden.
STRUCTURED_WEIGHTS_UNIFORM = dict(name="uniform",
                                  ind=(1.0, 1.0, 1.0, 1.0),
                                  bs=(1.0, 1.0, 1.0, 1.0))


# ── named one-switch configurations of the structured closure ───────────────
#
# A preset is a PRIOR plus a claim about the data.  It contains no acceptance
# criterion, no convergence threshold and no bound, it cannot change what
# "closed" means, and it is always opt-in: the shipped closure_channel default
# stays "bootstrap".  Spelled in SIGMAS (the natural unit of a prior width),
# with the trust weights derived as W = sigma^-2 so everything downstream --
# and the campaign CSV's weights_ind/weights_bs -- reads on the usual scale.

#: Named structured-closure presets, each a dict of prior widths plus the
#: data-statement switches that go with them.  See
#: :func:`structured_preset_settings`.
#:
#: ``"li_soft_onesided"`` is the configuration the l_i work converged on, and
#: is the one documented recommendation.  Its ladders, on the shipped
#: four-Gaussian basis (psi_N 0.15 / 0.45 / 0.75 / 0.95):
#:
#: * ``sigma_bs = (0.50, 0.30, 0.15, 0.10)`` -- the bootstrap is a local
#:   formula, tight at the PEDESTAL where Redl/Sauter is validated against
#:   drift-kinetic (NEO) calculations, loose in the CORE where the trapped
#:   fraction vanishes, the collisionality expansion is at its worst and the
#:   bootstrap current is negligible anyway, so a wide multiplier there costs
#:   almost no current.
#: * ``sigma_ind = (0.10, 0.40, 0.40, 0.40)`` -- the DOWN side.  The inductive
#:   core is pinned tight because the on-axis current row already determines
#:   it wherever the sawtooth gate admits one; mid-radius and mantle are left
#:   four times looser, which is where the closure is allowed to work.
#: * ``sigma_ind_up = (0.10, 0.10, 0.10, 0.40)`` -- the UP side of the
#:   one-sided prior: mid-radius inductive current may FALL freely and is
#:   resisted from rising.  EMPIRICALLY MOTIVATED (see
#:   :func:`close_ip_structured`), with the mechanism as support, not a
#:   derivation.
#: * ``structured_soft=True`` with ``sigma_Ip = 0.5 %`` of Ip -- the magnetics'
#:   own accuracy -- and, when the caller supplies an l_i target,
#:   ``sigma_li = 0.04``.
STRUCTURED_PRESETS = {
    "li_soft_onesided": dict(
        sigma_ind=(0.10, 0.40, 0.40, 0.40),
        sigma_ind_up=(0.10, 0.10, 0.10, 0.40),
        sigma_bs=(0.50, 0.30, 0.15, 0.10),
        structured_soft=True,
        structured_ip_sigma_frac=0.005,
        structured_li_sigma=0.04,
    ),
}

#: The preset the structured channel uses when the caller names none.
#: ``closure_channel="structured"`` with ``structured_preset=None`` resolves to
#: this preset, because the raw shipped fields (the symmetric physics ladder on
#: the hard solver) are the configuration the l_i study SUPERSEDED, and a
#: default that nobody is expected to want is a trap, not a default.  The
#: channel itself stays opt-in: nothing here changes ``closure_channel``, whose
#: shipped default is still ``"bootstrap"``.
#:
#: SCOPE, honestly stated: these sigmas were set from a study on ONE device
#: with ONE integrated-modelling source for the inductive current.  They are
#: PRIORS in RELATIVE units -- fractions of the component profiles themselves,
#: on normalised flux -- not device constants, which is why they transfer at
#: all; but they are a starting point elsewhere, not a validated setting.  On
#: another device or another source, run it and read the recorded
#: closure-health flags (the 0.2 < s < 5 scale bounds, |s_bs - 1| > 0.5, the q0
#: miss, the l_i z-score) before trusting the answer, and consider
#: :data:`STRUCTURED_WEIGHTS_UNIFORM` as the no-prior sensitivity.
STRUCTURED_PRESET_DEFAULT = "li_soft_onesided"

#: The ``structured_preset`` spelling that DECLINES the default preset and
#: leaves every structured field at its own shipped default (the symmetric
#: physics ladder, hard solver, Ip imposed exactly) -- the behaviour a bare
#: ``closure_channel="structured"`` had before :data:`STRUCTURED_PRESET_DEFAULT`
#: became the default.  It is spelled as a NAME rather than as ``None`` because
#: ``None`` now means "the caller expressed no preference", and the two have to
#: be distinguishable or the old default is unreachable.
STRUCTURED_PRESET_NONE = "none"


def structured_preset_settings(name):
    """Resolved :class:`~bouquet.config.GenerationConfig` settings for a preset.

    Returns a fresh dict keyed by the config field names the preset fills --
    ``structured_weights`` (W = sigma^-2 of the down-side inductive and the
    bootstrap ladders, carrying the preset's name), ``structured_sigma_ind_up``,
    ``structured_soft``, ``structured_ip_sigma_frac`` and
    ``structured_li_sigma``.  An unknown name is REFUSED, never silently
    ignored: a preset that does not exist is a typo, and a typo that quietly
    left the shipped prior in place would be invisible in the record.

    :data:`STRUCTURED_PRESET_NONE` (``"none"``) is accepted and returns an
    EMPTY dict: it is the opt-out spelling, i.e. "fill nothing, leave every
    structured field at its shipped default".  It is handled here, rather than
    only in the config, so that the same one refusal covers every spelling.

    The caller decides what to do with the settings; ``GenerationConfig``
    applies them only to fields still holding their dataclass default VALUE --
    which a field explicitly set to that same value also does, so such a field
    is overridden (see :meth:`bouquet.config.GenerationConfig.__post_init__`,
    which warns with the list of fields it filled) -- and it drops
    ``structured_li_sigma`` when there is no ``structured_li_target`` (a sigma
    on a measurement that was not supplied would be recorded as if an l_i row
    existed).
    """
    key = str(name)
    if key == STRUCTURED_PRESET_NONE:
        return {}
    if key not in STRUCTURED_PRESETS:
        raise ValueError(
            f"unknown structured_preset {key!r}; known presets: "
            + ", ".join(sorted(STRUCTURED_PRESETS))
            + f" (or {STRUCTURED_PRESET_NONE!r} to decline the default preset "
              "and keep every structured field at its shipped default)")
    spec = STRUCTURED_PRESETS[key]
    return {
        "structured_weights": dict(
            name=key,
            ind=tuple(float(v) for v in _weights_from_sigma(spec["sigma_ind"])),
            bs=tuple(float(v) for v in _weights_from_sigma(spec["sigma_bs"])),
        ),
        "structured_sigma_ind_up": [float(v) for v in spec["sigma_ind_up"]],
        "structured_soft": bool(spec["structured_soft"]),
        "structured_ip_sigma_frac": float(spec["structured_ip_sigma_frac"]),
        "structured_li_sigma": float(spec["structured_li_sigma"]),
    }


def structured_basis_eval(spec, psi):
    """``(K, len(psi))`` basis matrix for :func:`close_ip_structured`.

    *spec* is a dict: ``kind="gaussian"`` with ``centres`` and ``widths``
    (peak-normalised Gaussians, ``exp(-0.5 ((psi-mu)/w)^2)``), or
    ``kind="constant"`` -- a single function identically 1, which is what makes
    the channel reduce EXACTLY to the scalar channels (see the module tests).
    ``None`` selects :data:`STRUCTURED_BASIS_DEFAULT`.

    The basis deliberately does not have to be orthogonal or well-conditioned:
    with at most two equality constraints on 2K unknowns the problem is
    massively under-determined and the Tikhonov objective -- not the basis --
    is what selects the answer.
    """
    spec = dict(STRUCTURED_BASIS_DEFAULT if spec is None else spec)
    psi = np.asarray(psi, dtype=float)
    kind = str(spec.get("kind", "gaussian"))
    if kind == "constant":
        return np.ones((1, psi.size), dtype=float)
    if kind != "gaussian":
        raise ValueError(f"unknown structured basis kind {kind!r} "
                         "(expected 'gaussian' or 'constant')")
    mu = np.asarray(spec["centres"], dtype=float)
    wd = np.asarray(spec["widths"], dtype=float)
    # A genuinely SCALAR width broadcasts ("one width for all centres"); a
    # one-element LIST does not -- that is a length mismatch the caller should
    # hear about, not a shape numpy quietly repairs.
    if wd.ndim == 0:
        wd = np.full(mu.shape, float(wd))
    if mu.ndim != 1 or mu.size < 1 or wd.shape != mu.shape:
        raise ValueError("structured basis: 'centres' and 'widths' must be "
                         "1-D and the same length (or 'widths' a scalar)")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(wd))
            and np.all(wd > 0.0)):
        raise ValueError("structured basis: non-finite centre, or a width "
                         "that is not strictly positive")
    return np.exp(-0.5 * ((psi[None, :] - mu[:, None]) / wd[:, None]) ** 2)


# ── l_i as a second global measurement on the structured closure ────────────
#
# Ip alone under-determines the closure: it is ONE number against 2K
# coefficients, and the campaign showed that what the multiplier profiles do
# between the axis and the pedestal is exactly what Ip cannot see.  l_i is the
# second global number a magnetics reconstruction actually reports, and it is
# sensitive to precisely the radial redistribution Ip is blind to.
#
# The discrete form is EXACT, not a proxy.  With psi the per-radian poloidal
# flux (B_p = |grad psi| / R, which is TokaMaker's psi), Ampere's law on a flux
# surface reads
#
#     mu0 I(psi) = \oint (|grad psi| / R) dl
#                = (1/2pi) \oint (|grad psi|^2 / R^2) (2 pi R dl / |grad psi|)
#                = (V' / 2pi) <|grad psi|^2 / R^2>,
#
# and <B_p^2> IS <|grad psi|^2 / R^2>, so
#
#     <B_p^2>(psi) = 2 pi mu0 I(psi) / V'(psi)
#     =>  \int B_p^2 dV = \int <B_p^2> V' dpsi = 2 pi mu0 \int I(psi) dpsi.
#
# The flux-surface geometry cancels identically: the poloidal field energy of
# ANY equilibrium is fixed by its enclosed-current profile alone.  So with
# S = \int I dpsi (dimensional psi), TokaMaker's own two normalisations
# (OpenFUSIONToolkit TokaMaker _core.get_stats; Jackson et al 2008 eqs 1-2) are
#
#     l_i(1) = (Bp_vol/vol) / (mu0 Ip / dl)^2 = 2 pi dl^2 S / (mu0 vol Ip^2)
#     l_i(3) = 2 Bp_vol / ((mu0 Ip)^2 R_axis) = 4 pi S / (mu0 Ip^2 R_axis)
#
# with `dl` the LCFS poloidal perimeter, `vol` the plasma volume and `R_axis`
# the magnetic axis major radius (o_point[0] -- NOT R_geo; that is what
# get_stats uses).  Both are (prefactor) x S / Ip^2, i.e. LINEAR in the
# enclosed-current profile over the square of the total, which is what makes
# the constraint cheap: S is linear in the structured coefficients at frozen
# anchor geometry, exactly as Ip is.
#
# Measured on the synthetic D3D-like anchor (tests/test_li_closure.py, solver
# marker), on the equilibrium's own GS current profile: the identity reproduces
# get_stats' own Bp_vol to +0.0208 % and, fed get_stats' OWN dl, its l_i(1) to
# +0.0215 %; l_i(3) (which carries no perimeter) to +0.0215 %.
#
# WHICH PERIMETER, though, is not a detail -- l_i(1) ~ L^2.  TokaMaker's
# get_stats takes `dl` from get_q, and that reading sits ABOVE the perimeter of
# the traced 1 - psi_pad contour (the surface save_eqdsk writes as
# RBBBS/ZBBBS): +0.8 % on this anchor, +1.6 % on real diverted DIII-D
# equilibria, i.e. +1.7...+3.2 % of l_i(1), or 0.013...0.032 absolute.  An
# EFIT a-file LI is l_i(1) normalised through the Ampere-law mean field
# mu0 Ip / L_LCFS (OMFIT's transcription of the a-file spec), so a target taken
# from a reconstruction lives on the CONTOUR perimeter, and closing on
# get_stats' dl would charge the closure with current peaking it did not do.
# li_closure_geometry therefore measures the contour itself (lcfs_perimeter),
# and li_achieved reads the solve back the same way.  Calibrated against an
# independent out-of-tree g-file l_i estimator -- one estimator run on both
# equilibria -- on the anchor's saved g-file: closure l_i(1) +0.00038, achieved
# readback +0.00020, l_i(3) +0.0013 / +0.0012, against +0.01337 for
# get_stats('std').

#: The l_i normalisations this closure can target, spelled as TokaMaker's
#: ``get_stats(li_normalization=...)`` spells them.  ``"li_1"`` is the
#: EFIT-like one (volume-averaged B_p^2 over the LCFS-perimeter mean).
LI_KINDS = ("li_1", "li_3")

#: Log-gain exponent of the structured closure's l_i corrector row update.
#:
#: The corrector moves the l_i constraint ROW and measures what the re-solved
#: equilibrium delivers.  The row is satisfied exactly in the coefficient
#: algebra (with Ip pinned, l_i is linear in the coefficients), so the whole
#: row-to-achieved gap is the closure model's frozen-anchor geometry -- and the
#: piece of that geometry which moves is ``dpsi_dpsiN = |psi_axis - psi_bnd|``,
#: which :func:`li_closure_geometry` freezes at the anchor.
#:
#: ``l_i`` therefore enters the achieved equilibrium TWICE: once through the
#: prescribed enclosed-current shape integral the closure controls, and once
#: more through the flux range that responds to it.  With
#: ``Delta_psi ~ l_i**q`` the achieved value follows the row as
#: ``achieved ~ row**p`` with ``p = 1/(1 - q)``; ``q = 1/2``
#: (``Delta_psi ~ sqrt(l_i)``) gives ``p = 2``, i.e. the correct row update is
#: the SQUARE ROOT of the ratio the proportional model would have applied.
#:
#: Measured, out of tree, on a closure-cloud campaign of 133 corrected hard
#: slices (2026-09-14):
#: ``li_achieved / li_model = Delta_psi_solved / Delta_psi_anchor``
#: to ~1 % slice by slice, the empirical ``d(achieved)/d(row)`` is 2.14 against
#: the 0.96 the proportional model assumed, and the log-space exponent is
#: ``p = 2.17`` (median, IQR [2.13, 2.34], per-shot medians 2.05-2.42) --
#: i.e. ``Delta_psi ~ l_i**0.54``.  The value used here is the parameter-free
#: ``p = 2``, NOT the campaign-calibrated 2.17: 2.17 is a constant fitted to
#: one device's ohmic slices and is deliberately not adopted.  Slices whose own
#: ``p`` departs from 2 are what the optional second (log-secant) corrector
#: step is for -- it measures ``p`` per slice instead of assuming it.
LI_GAIN_EXPONENT = 2.0


def li_corrector_row(li_target, li_achieved_value, row=None,
                     exponent=LI_GAIN_EXPONENT):
    r"""Next l_i constraint row from one measured ``(row, achieved)`` pair.

    ``row' = row * (li_target / li_achieved) ** (1 / exponent)``

    The multiplicative form keeps the correction scale-free (l_i is positive);
    the exponent is the log-gain ``d ln(achieved) / d ln(row)`` of the achieved
    equilibrium -- see :data:`LI_GAIN_EXPONENT` for why it is 2 and not 1.  The
    superseded proportional update was this with ``exponent = 1``, which
    assumed a gain of ~1 where the measured gain is ~2 and therefore overshot
    by a little over a factor of two on every slice.

    *row* defaults to *li_target*, which is the hard channel's predictor row
    (the predictor imposes ``row = target`` exactly), so the first corrector
    step is ``T * (T / A) ** (1 / p)``.  Passing the previous row makes the
    same call serve a second step.

    Raises ``ValueError`` on a non-positive or non-finite argument -- an l_i,
    a row and a log-gain are all strictly positive, and a silent NaN row would
    be imposed by the closure as a constraint.
    """
    T = float(li_target)
    A = float(li_achieved_value)
    r = T if row is None else float(row)
    p = float(exponent)
    if not all(np.isfinite(v) for v in (T, A, r, p)):
        raise ValueError("li_corrector_row: non-finite argument "
                         f"(target={T!r}, achieved={A!r}, row={r!r}, "
                         f"exponent={p!r})")
    if T <= 0.0 or A <= 0.0 or r <= 0.0:
        raise ValueError("li_corrector_row needs positive l_i target, achieved "
                         f"and row (got {T!r}, {A!r}, {r!r})")
    if p <= 0.0:
        raise ValueError(f"li_corrector_row needs a positive exponent, got {p!r}")
    return float(r * (T / A) ** (1.0 / p))


#: Bounds the per-slice log-secant exponent must land in to be USED.  Not a
#: tolerance and not a fit: a sanity gate on a ratio of two logarithms, whose
#: job is to refuse a degenerate secant (a step too small to measure, or a
#: non-monotone pair) rather than impose a wild row.  The campaign's measured
#: spread is 1.83-2.57 (hard) and ~1.1-1.4 (soft), well inside.
LI_GAIN_EXPONENT_BOUNDS = (0.25, 8.0)


def li_gain_exponent_secant(row_prev, li_prev, row, li_now,
                            bounds=LI_GAIN_EXPONENT_BOUNDS, min_step=1.0e-6):
    r"""Per-slice log-secant exponent, or ``None`` when it is not measurable.

    ``p = ln(li_now / li_prev) / ln(row / row_prev)``

    After one corrector step there are TWO measured ``(row, achieved)`` pairs
    on the same slice, so the log-gain no longer has to be assumed: it can be
    read off.  This is a secant step in log space -- no fitted constant
    anywhere, in the same family as taking another iteration -- and it is what
    the optional second corrector step uses in place of
    :data:`LI_GAIN_EXPONENT`.

    Returns ``None`` (rather than raising) when the pair cannot support a
    secant: a non-positive or non-finite value, a step in the row too small to
    measure (``|ln(row/row_prev)| < min_step``), or an exponent outside
    *bounds*.  The caller then falls back to the parameter-free exponent.
    """
    try:
        r0, a0, r1, a1 = (float(row_prev), float(li_prev),
                          float(row), float(li_now))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (r0, a0, r1, a1)):
        return None
    if min(r0, a0, r1, a1) <= 0.0:
        return None
    dln_row = float(np.log(r1 / r0))
    if abs(dln_row) < float(min_step):
        return None
    p = float(np.log(a1 / a0) / dln_row)
    if not np.isfinite(p) or not (float(bounds[0]) <= p <= float(bounds[1])):
        return None
    return p


def lcfs_perimeter(eq, psi_pad=_FSA_PSI_PAD):
    r"""``(L, source)`` -- the LCFS poloidal perimeter [m], and where it came from.

    ``l_i(1)`` is proportional to ``L^2``, so which surface's perimeter is used
    is not a detail: a 1.6 % difference in ``L`` is a 3.2 % difference in
    ``l_i(1)``, which is several times any tolerance in this module.

    TokaMaker's ``get_stats`` takes its ``dl`` from ``get_q``, and on the
    synthetic D3D-like anchor that reading is **+0.83 % above** the perimeter of
    the traced ``1 - psi_pad`` contour -- the same contour ``save_eqdsk`` writes
    into a g-file's ``RBBBS``/``ZBBBS``, i.e. the surface an EFIT-convention
    l_i(1) estimator measures on.  (On real diverted DIII-D equilibria the
    calibration study measured +1.6 %.)  Left alone that makes the closure's
    l_i(1) sit ~0.013-0.032 above any g-file-based estimator, which would be
    read as current peaking that is not there.

    So the default here is the CONTOUR's own polygon perimeter, traced with
    :func:`safe_trace_surf` at ``1 - psi_pad`` and closed.  Measured on the
    anchor (tests/test_li_closure.py, solver marker), that choice puts the
    closure's l_i(1) within **0.0004** of ``lib/li_from_geqdsk.py`` run on the
    saved g-file, against ``+0.0134`` for the ``get_stats`` reading.

    Falls back to the ``get_stats`` ``dl`` when the trace fails, and SAYS so in
    the returned source string -- never silently.
    """
    pts = None
    try:
        pts = safe_trace_surf(eq, 1.0 - float(psi_pad))
    except Exception:                                # pragma: no cover
        pts = None
    pts = None if pts is None else np.asarray(pts, dtype=float)
    if pts is not None and pts.ndim == 2 and pts.shape[0] >= 4 \
            and np.all(np.isfinite(pts)):
        rc = np.append(pts[:, 0], pts[0, 0])
        zc = np.append(pts[:, 1], pts[0, 1])
        L = float(np.hypot(np.diff(rc), np.diff(zc)).sum())
        if np.isfinite(L) and L > 0.0:
            return L, (f"traced LCFS contour at psi_N = 1 - {float(psi_pad):g} "
                       f"({pts.shape[0]} points), closed polygon length -- the "
                       "same surface save_eqdsk writes as RBBBS/ZBBBS")
    dl = eq.get_q(np.r_[1.0 - float(psi_pad), 0.95, 0.02],
                  compute_geo=True)[3]
    return (float(np.ravel(np.asarray(dl, dtype=float))[-1]),
            "FALLBACK: get_stats' own get_q 'dl' (the LCFS trace failed); "
            "this reads ~1 % HIGH against a g-file perimeter, so l_i(1) from "
            "it reads ~2 % high")


def li_achieved(eq, li_kind="li_1", psi_pad=_FSA_PSI_PAD, perimeter=None):
    r"""The SOLVED equilibrium's own l_i, in the closure's convention.

    Built from TokaMaker's exact volume integrals (``get_globals``: ``Ip``,
    ``vol``, ``Bp_vol = int B_p^2 dV``) rather than from the closure's model,
    so it is an independent reading of what the solve actually produced -- but
    normalised the SAME way the target is, which ``get_stats`` is not:

    * ``li_1`` = ``(Bp_vol/vol) / (mu0 |Ip| / L)^2`` with *L* from
      :func:`lcfs_perimeter`, i.e. ``get_stats(li_normalization='std')`` with
      its ``dl`` replaced by the LCFS contour's own perimeter.  Comparing a
      target that came from a g-file estimator against ``get_stats('std')``
      would charge the closure with the perimeter offset documented there.
    * ``li_3`` = ``2 Bp_vol / ((mu0 Ip)^2 R_axis)`` -- identical to
      ``get_stats(li_normalization='iter')``, which the calibration study found
      already agrees with a g-file l_i(3) to ~0.001.

    Returns ``(li, info)``; *info* records ``perimeter``, its source, and the
    raw globals, so a recorded l_i can always be re-derived.
    """
    Ip, _centroid, vol, _pvol, _dflux, _tflux, Bp_vol = eq.get_globals()
    Ip, vol, Bp_vol = float(Ip), float(vol), float(Bp_vol)
    if not (np.isfinite(Ip) and Ip != 0.0 and np.isfinite(vol) and vol > 0.0
            and np.isfinite(Bp_vol)):
        raise RuntimeError("li_achieved: unusable globals "
                           f"(Ip={Ip!r}, vol={vol!r}, Bp_vol={Bp_vol!r})")
    kind = str(li_kind)
    info = dict(Ip=Ip, vol=vol, Bp_vol=Bp_vol, li_kind=kind,
                perimeter=None, perimeter_source=None, R_axis=None)
    if kind == "li_1":
        if perimeter is None:
            L, src = lcfs_perimeter(eq, psi_pad=psi_pad)
        else:
            L, src = float(perimeter), "caller-supplied"
        info.update(perimeter=L, perimeter_source=src)
        return float((Bp_vol / vol) / (_FSA_MU0 * abs(Ip) / L) ** 2), info
    if kind == "li_3":
        R_axis = float(np.asarray(eq.o_point, dtype=float)[0])
        info["R_axis"] = R_axis
        return float(2.0 * Bp_vol / ((_FSA_MU0 * Ip) ** 2 * R_axis)), info
    raise ValueError(f"li_achieved: unknown li_kind {kind!r} "
                     f"(expected one of {LI_KINDS})")


def li_closure_geometry(eq, geom, convention="jphi-linterp", pprime_sign=1.0,
                        psi_pad=_FSA_PSI_PAD, axis_pad=0.02, perimeter=None):
    r"""The three scalars (plus the affine profile) the l_i constraint needs.

    Read off the SAME frozen anchor the Ip measure is built on -- *eq* is a
    live ``TokaMaker`` or a ``copy_eq()`` snapshot, *geom* the matching
    :func:`fsa_current_geometry` dict.  Nothing here is a fit or a calibration:
    every number is the one ``get_stats`` itself uses, taken from the same call
    with the same padding, so ``li_value`` on the anchor's own current profile
    reproduces ``get_stats(li_normalization=...)['l_i']``.

    Returns a dict with

    ``psi_N``        the geometry's grid (the enclosed-current grid)
    ``dpsi_dpsiN``   |psi_bounds[1] - psi_bounds[0]|, the S Jacobian
    ``vol``          plasma volume [m^3]      (``get_globals()[2]``)
    ``perimeter``    LCFS poloidal perimeter [m] -- :func:`lcfs_perimeter`, the
                     traced ``1 - psi_pad`` contour's own polygon length, NOT
                     ``get_stats``' ``dl``; read that function's docstring
                     before changing it, the difference is ~2 % of l_i(1)
    ``perimeter_get_stats_dl`` what ``get_stats`` would have used, and
                     ``perimeter_ratio_dl_over_L`` the ratio, so the offset is
                     recorded rather than argued about
    ``R_axis``       magnetic-axis major radius [m] (``o_point[0]``)
    ``affine_cum``   the cumulative ``P'`` current, :func:`Ip_fsa_affine_profile`

    ``psi_pad`` must match the ``lcfs_pad`` the rest of the slice uses
    (bouquet passes ``source.psi_pad``); a different padding is a different
    LCFS and therefore a different perimeter.  ``perimeter`` overrides the
    measurement entirely, for a caller that has one from elsewhere.
    """
    psi = np.asarray(geom["psi_N"], dtype=float)
    vol = float(eq.get_globals()[2])
    if perimeter is None:
        perimeter, perimeter_source = lcfs_perimeter(eq, psi_pad=psi_pad)
    else:
        perimeter, perimeter_source = float(perimeter), "caller-supplied"
    # what get_stats itself would have used, recorded for the comparison
    dl = eq.get_q(np.r_[1.0 - float(psi_pad), 0.95, float(axis_pad)],
                  compute_geo=True)[3]
    dl = float(np.ravel(np.asarray(dl, dtype=float))[-1])
    R_axis = float(np.asarray(eq.o_point, dtype=float)[0])
    if not (np.isfinite(vol) and vol > 0.0):
        raise RuntimeError(f"li_closure_geometry: bad plasma volume {vol!r}")
    if not (np.isfinite(perimeter) and perimeter > 0.0):
        raise RuntimeError("li_closure_geometry: bad LCFS perimeter "
                           f"{perimeter!r}")
    if not (np.isfinite(R_axis) and R_axis > 0.0):
        raise RuntimeError(f"li_closure_geometry: bad axis radius {R_axis!r}")
    return dict(
        psi_N=psi,
        dpsi_dpsiN=float(geom["dpsi_dpsiN"]),
        vol=vol, perimeter=perimeter, R_axis=R_axis,
        perimeter_source=perimeter_source,
        perimeter_get_stats_dl=dl,
        perimeter_ratio_dl_over_L=(dl / perimeter if perimeter else None),
        affine_cum=Ip_fsa_affine_profile(geom, convention=convention,
                                         pprime_sign=pprime_sign),
        psi_pad=float(psi_pad),
    )


def li_prefactor(li_geom, li_kind="li_1"):
    r"""``G`` in ``l_i = G * S * sgn / Ip^2`` for the requested normalisation.

    ``li_1`` -> ``2 pi dl^2 / (mu0 vol)``; ``li_3`` -> ``4 pi / (mu0 R_axis)``.
    See the module comment above for the derivation.
    """
    kind = str(li_kind)
    if kind == "li_1":
        return (2.0 * np.pi * float(li_geom["perimeter"]) ** 2
                / (_FSA_MU0 * float(li_geom["vol"])))
    if kind == "li_3":
        return 4.0 * np.pi / (_FSA_MU0 * float(li_geom["R_axis"]))
    raise ValueError(f"unknown li_kind {kind!r} (expected one of {LI_KINDS})")


def li_value(psi_N, I_enclosed, li_geom, li_kind="li_1", Ip=None):
    r"""``l_i`` of an enclosed-current profile, on the anchor's geometry.

    *I_enclosed* is ``I(psi_N)`` [A] (zero on axis, the total plasma current at
    the last sample); *Ip* defaults to that last sample.  The sign convention
    is carried through with ``sgn = sign(Ip)`` so a COCOS-negative current
    returns a POSITIVE l_i rather than a negative one.
    """
    psi = np.asarray(psi_N, dtype=float)
    I = np.asarray(I_enclosed, dtype=float)
    Ip_v = float(I[-1]) if Ip is None else float(Ip)
    if Ip_v == 0.0 or not np.isfinite(Ip_v):
        raise RuntimeError(f"li_value: unusable total current {Ip_v!r}")
    S = float(li_geom["dpsi_dpsiN"]) * _trapezoid(I, psi)
    sgn = float(np.sign(Ip_v))
    return float(li_prefactor(li_geom, li_kind) * S * sgn / Ip_v ** 2)


def structured_li_model(psi_N, w_lin, Phi, j_ind, j_bs, j_fix, li_geom,
                        Ip_target_signed, li_kind="li_1"):
    r"""The l_i response of the structured closure, as coefficient algebra.

    Everything the l_i constraint (hard or soft) needs, evaluated ONCE at the
    frozen anchor geometry.  With ``x = (a, b)`` the 2K structured coefficients,

    .. code-block:: text

        Ip(x) = Ip0 + Ip_ind . a + Ip_bs . b          (exactly close_ip's row)
        S(x)  = S0  + S_ind  . a + S_bs  . b          (the psi-integral of I)
        li(x) = G S(x) sgn / Ip(x)^2

    -- ``S`` linear because the enclosed current is a cumulative integral of a
    linear functional of the multipliers, and the 2 pi mu0 identity turns the
    poloidal field energy into ``S`` with no geometry left in it.  ``li`` is
    therefore a RATIO of a linear form to the square of another: linear when Ip
    is pinned (the hard channel), mildly nonlinear when Ip is soft.

    Returns the model dict consumed by :func:`structured_li_of` and
    :func:`structured_li_gradient`.
    """
    from scipy.integrate import cumulative_trapezoid

    psi = np.asarray(psi_N, dtype=float)
    w_lin = np.asarray(w_lin, dtype=float)
    aff = np.asarray(li_geom["affine_cum"], dtype=float)
    if aff.shape != psi.shape:
        raise ValueError("structured_li_model: li_geom['affine_cum'] is on a "
                         f"different grid ({aff.shape} vs {psi.shape}) -- the "
                         "l_i geometry and the Ip measure must share one grid")
    dpsi = float(li_geom["dpsi_dpsiN"])
    _cum = lambda y: cumulative_trapezoid(w_lin * np.asarray(y, float), psi,
                                          initial=0.0)
    _S = lambda I: dpsi * _trapezoid(I, psi)

    I0 = _cum(np.asarray(j_ind, float) + np.asarray(j_bs, float)
              + np.asarray(j_fix, float)) + aff
    K = int(np.asarray(Phi).shape[0])
    S_ind = np.array([_S(_cum(Phi[k] * np.asarray(j_ind, float)))
                      for k in range(K)], dtype=float)
    S_bs = np.array([_S(_cum(Phi[k] * np.asarray(j_bs, float)))
                     for k in range(K)], dtype=float)
    sgn = float(np.sign(float(Ip_target_signed)) or 1.0)
    return dict(
        li_kind=str(li_kind), G=float(li_prefactor(li_geom, li_kind)),
        S0=float(_S(I0)), S_ind=S_ind, S_bs=S_bs,
        Ip0=float(I0[-1]),
        Ip_ind=np.array([float(_cum(Phi[k] * np.asarray(j_ind, float))[-1])
                         for k in range(K)], dtype=float),
        Ip_bs=np.array([float(_cum(Phi[k] * np.asarray(j_bs, float))[-1])
                        for k in range(K)], dtype=float),
        sgn=sgn, basis_K=K, I_enclosed_0=I0,
    )


def structured_li_of(model, x=None):
    """``(li, Ip)`` of the structured coefficient vector *x* (default 0)."""
    K = int(model["basis_K"])
    x = np.zeros(2 * K, dtype=float) if x is None \
        else np.asarray(x, dtype=float)
    a, b = x[:K], x[K:]
    Ip = float(model["Ip0"] + model["Ip_ind"] @ a + model["Ip_bs"] @ b)
    S = float(model["S0"] + model["S_ind"] @ a + model["S_bs"] @ b)
    if Ip == 0.0 or not np.isfinite(Ip):
        raise RuntimeError("structured_li_of: the closed hybrid carries zero "
                           "total current; l_i is undefined there")
    return float(model["G"] * S * model["sgn"] / Ip ** 2), Ip


def structured_li_gradient(model, x=None):
    r"""``d li / d x`` at *x* (default the anchor, ``x = 0`` i.e. ``s == 1``).

    .. math:: \nabla l_i = \frac{G\,{\rm sgn}}{I_p^2}
              \left( \nabla S - \frac{2 S}{I_p} \nabla I_p \right)

    -- the second term is the Ip response, and vanishes identically wherever Ip
    is pinned (the hard channel's Ip row), which is why the hard constraint row
    is exact rather than approximate in Ip.
    """
    K = int(model["basis_K"])
    x = np.zeros(2 * K, dtype=float) if x is None \
        else np.asarray(x, dtype=float)
    a, b = x[:K], x[K:]
    Ip = float(model["Ip0"] + model["Ip_ind"] @ a + model["Ip_bs"] @ b)
    S = float(model["S0"] + model["S_ind"] @ a + model["S_bs"] @ b)
    gS = np.concatenate([model["S_ind"], model["S_bs"]])
    gIp = np.concatenate([model["Ip_ind"], model["Ip_bs"]])
    return (model["G"] * model["sgn"] / Ip ** 2) * (gS - (2.0 * S / Ip) * gIp)


def _structured_li_row(psi, w_lin, Phi, j_ind, j_bs, j_fix, li_geom,
                       Ip_target_signed, li_target, li_kind, who,
                       Ip_pinned=None):
    """``(model, rhs, li_anchor, gradient)`` for the l_i constraint row.

    Shared by the hard and the soft solver so there is exactly one place that
    decides what "the l_i constraint" means.

    The row is the linearisation of ``l_i(x)`` about ``s == 1``, taken **along
    the Ip-closed manifold** when *Ip_pinned* is given (which is always, in
    :func:`close_ip_structured`, whose Ip row is an exact equality).  That
    matters and is not a detail: ``l_i = G S(x) sgn / Ip(x)^2`` and ``S`` is
    linear, so with ``Ip`` pinned at its target the whole expression is
    LINEAR in *x* and the "linearised" row is the exact one --

    .. code-block:: text

        grad_S . x = li_target * Ip_target^2 / (G sgn) - S0.

    Linearising about the ANCHOR's own ``Ip0`` instead leaves the
    ``(Ip0/Ip_target)^2`` mismatch in the row, which on a 4 % Ip deficit costs
    ~0.3 % of l_i -- an error the one-step corrector would then be spending
    its single extra solve on, for nothing.  ``Ip_pinned=None`` (the soft
    solver's Ip-as-a-measurement case) falls back to the full gradient at
    ``x = 0``, where the Ip response term is real and must be kept.
    """
    if li_geom is None:
        raise ValueError(f"{who}: li_target was given without li_geom -- build "
                         "it once from the anchor with li_closure_geometry()")
    if not np.isfinite(float(li_target)):
        raise RuntimeError(f"{who}: non-finite li_target {li_target!r}")
    if str(li_kind) not in LI_KINDS:
        raise ValueError(f"{who}: unknown li_kind {li_kind!r} "
                         f"(expected one of {LI_KINDS})")
    model = structured_li_model(psi, w_lin, Phi, j_ind, j_bs, j_fix, li_geom,
                                Ip_target_signed, li_kind=li_kind)
    li0, _ip0 = structured_li_of(model)
    if Ip_pinned is None:
        grad = structured_li_gradient(model)
        rhs = float(li_target) - float(li0)
    else:
        Ip_p = float(Ip_pinned)
        if Ip_p == 0.0 or not np.isfinite(Ip_p):
            raise RuntimeError(f"{who}: unusable pinned Ip {Ip_pinned!r}")
        k = model["G"] * model["sgn"] / Ip_p ** 2
        grad = k * np.concatenate([model["S_ind"], model["S_bs"]])
        rhs = float(li_target) - k * float(model["S0"])
    if not (np.isfinite(li0) and np.isfinite(rhs)
            and np.all(np.isfinite(grad))):
        raise RuntimeError(f"{who}: the l_i model is non-finite on this anchor "
                           f"(l_i(s==1) = {li0!r})")
    return model, float(rhs), float(li0), grad


def _li_record(li_model, li_anchor, li_target, li_grad, x, Ip_pinned=None):
    """The l_i block every structured solver puts in its result dict.

    ``li_predicted`` is the EXACT model value at the solution (not the
    linearised one), so the gap between it and ``li_target`` is the
    linearisation error the row itself carries -- zero when Ip is pinned and
    the model is exactly linear, and worth reading when it is not.
    """
    if li_model is None:
        return dict(li_kind=None, li_target=None, li_anchor=None,
                    li_anchor_at_target_Ip=None, li_predicted=None,
                    li_predictor_residual=None, li_gradient=None)
    li_pred, _ip = structured_li_of(li_model, x)
    proj = None
    if Ip_pinned not in (None, 0.0):
        proj = float(li_model["G"] * li_model["sgn"] * li_model["S0"]
                     / float(Ip_pinned) ** 2)
    return dict(
        li_kind=str(li_model["li_kind"]),
        li_target=float(li_target),
        li_anchor=float(li_anchor),
        li_anchor_at_target_Ip=proj,
        li_predicted=float(li_pred),
        li_predictor_residual=float(li_pred) - float(li_target),
        li_gradient=np.asarray(li_grad, dtype=float),
    )


#: sign-iteration cap for the ONE-SIDED inductive prior (see
#: :func:`_one_sided_sign_iterate`).  Not a tolerance: the iteration either
#: settles on a sign pattern exactly or is refused.
SIGN_ITER_MAX = 8


def _one_sided_sign_iterate(solve_for, K, who, max_iter=SIGN_ITER_MAX):
    r"""Solve an asymmetric-Tikhonov structured closure by SIGN ITERATION.

    The one-sided prior (see :func:`close_ip_structured`) penalises an
    inductive coefficient ``a_k`` with ``1/sigma_down_k^2`` while it is
    negative and ``1/sigma_up_k^2`` while it is positive.  The penalty

    .. math:: f(a) = a^2/\sigma_{\rm down}^2\ (a<0), \qquad
              f(a) = a^2/\sigma_{\rm up}^2\ (a>0)

    is continuous at 0 with ``f'(0^-) = f'(0^+) = 0`` and a non-decreasing
    derivative, so it is CONVEX for any pair of positive sigmas.  The PRIOR
    term is therefore convex whichever channel calls this.

    **Where the exactness claim holds, and where it does not.**

    * :func:`close_ip_structured` (hard Ip), and
      :func:`close_ip_structured_soft` with a HARD Ip (``Ip_sigma=None``):
      Ip is imposed exactly, so ``l_i`` is LINEAR in the coefficients and the
      whole objective is convex with a unique minimiser.  Then fixing a sign
      pattern gives an ordinary quadratic whose minimiser the existing solver
      returns in closed form; if the returned coefficients' signs AGREE with
      the assumed pattern, the point is stationary for the true piecewise
      objective, hence its global minimum.  A stable pattern is the exact
      answer, not an approximation to it, and because the minimiser is unique
      every stable pattern carries the same coefficients -- the starting
      pattern can change how many solves it takes, never what comes out.
    * :func:`close_ip_structured_soft` with a FINITE ``Ip_sigma`` -- the soft
      channel's headline use: the ``l_i`` measurement term is
      ``(G S(x) sgn / Ip(x)^2 - li_target)^2 / sigma^2``, a linear form over
      the SQUARE of another linear form, which is not convex.  Neither
      uniqueness nor "every stable pattern carries the same coefficients" is
      guaranteed there, and Gauss-Newton itself only finds a local stationary
      point.  What a stable pattern gives is a stationary point of the true
      piecewise objective, not a certified global minimum.

      Empirically benign at shipped settings: on the reference fixture, with
      sigma_Ip = 0.5 % of Ip and sigma_li = 0.03, a 40-restart
      Nelder-Mead + BFGS multistart found the same minimum this solver
      returned (objective agreeing to 10 digits, coefficients to 2.6e-8).
      That is evidence, not a proof, and an A/B study on this channel should
      say so.

    *solve_for* takes a length-*K* boolean pattern (True = "this coefficient
    is on the up side") and returns a tuple whose FIRST entry is the full
    ``(2K,)`` coefficient vector.  The iteration starts from the all-down
    pattern, costs zero GS solves (each step is the same linear algebra the
    symmetric channel does once), and is capped at *max_iter*.  ``a_k == 0``
    counts as DOWN, which is the only convention that leaves the symmetric
    case (``sigma_up == sigma_down``) settling on the first solve.

    **Refusals** (``RuntimeError``, never a quiet clamp or a best-effort
    return): a pattern that has already been visited comes back (a cycle --
    the minimiser is being chased between two faces and no fixed point is
    reached), or the cap is hit without the pattern settling.  A caller that
    sees either is being told the one-sided prior is ill-posed on this slice,
    which is a finding to report.
    """
    _fmt = lambda p: "".join("+" if v else "-" for v in p)
    pattern = (False,) * int(K)
    seen = [pattern]
    for it in range(1, int(max_iter) + 1):
        out = solve_for(pattern)
        a = np.asarray(out[0], dtype=float)[:int(K)]
        new = tuple(bool(v) for v in (a > 0.0))
        if new == pattern:
            return out, pattern, it
        if new in seen:
            raise RuntimeError(
                f"{who}: the one-sided inductive prior's sign iteration is "
                f"CYCLING -- pattern {_fmt(new)} was already visited "
                f"(history {' -> '.join(_fmt(p) for p in seen)} -> "
                f"{_fmt(new)}) after {it} solve(s).  No sign pattern is "
                "self-consistent, so the piecewise-quadratic minimiser cannot "
                "be reached by this iteration; refusing to return whichever "
                "face the loop happened to stop on")
        seen.append(new)
        pattern = new
    raise RuntimeError(
        f"{who}: the one-sided inductive prior's sign iteration did not "
        f"settle in {int(max_iter)} solves (history "
        f"{' -> '.join(_fmt(p) for p in seen)}); refusing to return an "
        "unconverged sign pattern")


def _one_sided_ladder_check(sig_up, sig_down, who):
    """The up-side ladder may set WIDTHS, never which coefficients exist.

    ``sigma = 0`` pins a coefficient to zero and ``sigma = inf`` leaves it
    unpenalised; both change the set of unknowns (and, in the soft solver, the
    hard-constraint null space too).  Those decisions belong to the ONE prior
    ladder, not to whichever side a coefficient happens to be on -- otherwise
    the problem's dimension would flip with the sign during the iteration and
    the fixed point would not mean anything.  So the up ladder must pin and
    un-penalise in exactly the same places as the down ladder.
    """
    up = np.asarray(sig_up, dtype=float)
    dn = np.asarray(sig_down, dtype=float)
    bad = ((up <= 0.0) != (dn <= 0.0)) | (np.isfinite(up) != np.isfinite(dn))
    if np.any(bad):
        k = int(np.argmax(bad))
        raise ValueError(
            f"{who}: the one-sided up-side sigma ladder disagrees with the "
            f"down-side ladder about basis {k} (up {up[k]!r}, down "
            f"{dn[k]!r}) -- sigma 0 (pinned) and sigma inf (unpenalised) "
            "decide WHICH coefficients exist and must match on both sides; "
            "the one-sided ladder may only set the width of the up-side "
            "penalty")


def close_ip_structured(psi_N, w_lin, c_affine, Ip_target_signed,
                        j_ind, j_bs, j_fix, basis=None, weights=None,
                        axis=None, scale_bounds=(0.2, 5.0), cond_rtol=1e-6,
                        li_target=None, li_kind="li_1", li_geom=None,
                        sigma_ind_up=None):
    r"""Minimal-norm radial multiplier profiles closing Ip (and optionally q0).

    The ``closure_channel="structured"`` algebra.  Unknowns are two smooth
    multiplier PROFILES on a small basis :math:`\phi_k`,

    .. math::

        s_{\rm ind}(\psi) = 1 + \sum_k a_k \phi_k(\psi), \qquad
        s_{\rm bs}(\psi)  = 1 + \sum_k b_k \phi_k(\psi),

    and the closed hybrid is
    ``s_ind*j_ind + s_bs*j_BS + j_fix``.  Among all coefficient vectors that
    satisfy the constraints exactly, the one returned minimises

    .. math:: \sum_k W^{\rm ind}_k a_k^2 + \sum_k W^{\rm bs}_k b_k^2,

    i.e. it is the closure that departs LEAST from "trust the sources"
    (:math:`s \equiv 1`) in the trust-weighted norm.  Two constraints, both
    LINEAR in the coefficients and both already in use by the scalar channels:

    * **Ip, exactly, in the affine FSA measure.**  ``I_p[J] = trapezoid(w_lin
      J, psi_N) + c`` is affine and linear in *J*, so
      ``sum_k a_k lin(phi_k j_ind) + sum_k b_k lin(phi_k j_BS) = deficit`` with
      ``deficit = sgn*Ip - c - lin(j_ind) - lin(j_BS) - lin(j_fix)``.  The
      affine ``c`` is carried ONCE, exactly as in :func:`close_ip`.
    * **the on-axis current (q0), when *axis* is given.**  Same row as
      :func:`close_ip_q0`: ``s_ind(0) j_ind0 + s_bs(0) j_bs0 + j_fix0 =
      j_ref0``, from ``q0 ~ 1/j_phi(0)`` at frozen anchor geometry.  Pass
      ``axis=None`` (the sawtooth gate rejected the slice) and only Ip is
      imposed.
    * **the internal inductance, when *li_target* is given** (with the matching
      *li_geom* from :func:`li_closure_geometry`).  ``l_i`` is
      ``G S(x) sgn / Ip(x)^2`` with ``S`` the psi-integral of the enclosed
      current (see :func:`structured_li_model` and the module comment above);
      the row imposed is its LINEARISATION about ``s == 1``,
      ``grad l_i(0) . x = li_target - l_i(0)``.  Because the Ip row pins
      ``Ip(x)`` exactly, the only approximation in that row is the frozen
      anchor geometry -- the same approximation the q0 row makes, corrected the
      same way (ONE post-solve step; see run.py's structured corrector).  This
      is the SECOND global measurement the closure needs: Ip is one number
      against 2K coefficients and is blind to radial redistribution, which is
      exactly what l_i sees.

    Solved in closed form through the KKT/bordered system

    .. code-block:: text

        [ 2W   C^T ] [x]   [0]
        [ C    0   ] [l] = [d]

    with one ``numpy.linalg.solve`` and **zero GS solves**, like the q0
    predictor.  The rows of ``C`` and the entries of ``W`` are normalised
    first (each row by its own inf-norm, ``W`` by its maximum) -- both are
    exact reparametrisations that leave the minimiser untouched, and they are
    what makes the singularity test below scale-free: the raw system mixes
    amps (the Ip row) with A/m^2 (the axis row) and no absolute epsilon is
    meaningful across both.  Same reasoning as ``close_ip_q0``'s ``det_rtol``.

    **Weights.**  ``weights`` is ``{"ind": (K,), "bs": (K,)}`` (default
    :data:`STRUCTURED_WEIGHTS_PHYSICS`); ``numpy.inf`` hard-pins that
    coefficient to 0 by removing it from the unknowns, which is how the
    equivalence with the scalar channels is made EXACT rather than asymptotic.
    Weights must be strictly positive.

    **One-sided (asymmetric) inductive prior.**  ``sigma_ind_up`` is an
    OPTIONAL second ladder for the inductive coefficients only.  ``None`` (the
    default) is the symmetric prior above, unchanged and taking exactly one
    solve.  When it is given, ``a_k`` is penalised with the ``weights``
    ladder's own width while it is negative and with ``sigma_ind_up[k]`` while
    it is positive -- i.e. ``sigma_down = W_ind^{-1/2}`` is the DOWN side and
    ``sigma_ind_up`` the UP side, so a tight ``sigma_ind_up`` lets a
    multiplier fall freely and resists it rising.  The objective becomes
    piecewise-quadratic but stays convex, and the sign-iterated KKT is its
    EXACT minimiser once the sign pattern is stable; see
    :func:`_one_sided_sign_iterate` for the argument, the cap and the
    cycling refusal.  Still zero GS solves.

    *Why one-sided, and why on the inductive multiplier.*  This prior is
    EMPIRICALLY MOTIVATED, with a supporting mechanism -- not derived from
    one.  The empirical part: over the closure cloud, the l_i-informed
    channels win on the over-shoot slices and LOSE on the under-shoot ones,
    and the MSE chords are what says so -- they penalise inductive current
    ADDED at mid-radius (measured out of tree; see the closure-cloud study
    write-ups kept with the campaign data).
    The mechanism that makes that asymmetry expected rather than a fluke: the
    transport model's edge T_e collapses relative to the reference kinetics
    (ratios of 2 to 40), the edge resistivity therefore runs high, current
    diffuses inward faster than it should, and its mid-radius inductive current
    is
    accordingly more likely OVER- than under-estimated.  A prior that lets
    mid-radius ``s_ind`` fall freely while resisting a rise encodes exactly
    that.

    *Circularity, stated plainly.*  The asymmetry was tuned on the same MSE
    chords that are then used to judge it.  The mechanism above is what keeps
    it from being pure curve-fitting, but it is NOT an independent
    confirmation, and any result obtained with this prior has to be reported
    with that caveat attached.

    **Refusals** (``RuntimeError``, never a quiet clamp): a non-finite input;
    constraint rows that are DEGENERATE against the relative floor
    ``cond_rtol`` (the row-normalised constraint matrix's smallest singular
    value below ``cond_rtol`` x its largest); a constraint row that is
    identically zero (the constraint cannot be imposed on this split at all);
    or a resulting ``s_ind``/``s_bs`` that leaves ``scale_bounds`` ANYWHERE on
    the grid.  A structured closure that has to drive a multiplier to 0.1 or 8
    somewhere is telling you the three sources do not add up, which is a
    finding to report.

    The degeneracy test is on the CONSTRAINT rows only, exactly as
    :func:`close_ip_structured_soft` tests its hard rows.  It is not asked of
    the bordered KKT matrix, whose smallest singular value is bounded by the
    PRIOR's dynamic range: that version refused any ``W_max/W_min`` beyond
    ~``1/cond_rtol``, i.e. a sigma ratio of a few hundred, and told the user
    their constraints were degenerate when they were independent.  A tight
    ``sigma_ind_up`` -- the whole point of the asymmetric prior -- is now an
    ordinary request: the weighted system is solved in the substitution
    ``y = W^(1/2) x``, which is the same minimiser in a scaling that does not
    carry the prior.  ``cond_rtol`` itself is unchanged.

    Returns a dict carrying ``s_ind``/``s_bs`` (on *psi_N*), the coefficients,
    the effective scalar equivalents (see below), the constraint residuals and
    the structure numbers.

    **Effective scalar equivalents.**  ``ohm_scale_eff = lin(s_ind j_ind) /
    lin(j_ind)`` is the ONE number that would move the same current in the Ip
    measure, so by construction ``ohm_scale_eff*ip_ind + bs_scale_eff*ip_bs +
    ip_fix + c = Ip`` -- the structured solution satisfies the SAME scalar
    closure equation its effective scalars do, which is exactly what makes
    :func:`closure_health` applicable unchanged.  ``structure_ind/bs`` is the
    current-weighted RMS of ``s`` about that mean: 0 means the answer really
    was a scalar, and a large value is the part of the closure no scalar
    channel can represent.
    """
    from scipy.integrate import trapezoid

    psi = np.asarray(psi_N, dtype=float)
    w_lin = np.asarray(w_lin, dtype=float)
    j_ind = np.asarray(j_ind, dtype=float)
    j_bs = np.asarray(j_bs, dtype=float)
    j_fix = np.asarray(j_fix, dtype=float)
    if not (psi.shape == w_lin.shape == j_ind.shape == j_bs.shape
            == j_fix.shape):
        raise ValueError("close_ip_structured: psi_N, w_lin and the three "
                         "current components must share one grid, got "
                         f"{psi.shape}, {w_lin.shape}, {j_ind.shape}, "
                         f"{j_bs.shape}, {j_fix.shape}")
    for nm, arr in (("w_lin", w_lin), ("j_ind", j_ind), ("j_bs", j_bs),
                    ("j_fix", j_fix)):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"close_ip_structured: non-finite {nm}")
    if not (np.isfinite(c_affine) and np.isfinite(Ip_target_signed)):
        raise RuntimeError("close_ip_structured: non-finite c_affine or "
                           "Ip_target_signed")

    basis_spec = dict(STRUCTURED_BASIS_DEFAULT if basis is None else basis)
    Phi = structured_basis_eval(basis_spec, psi)             # (K, N)
    K = Phi.shape[0]
    wspec = dict(structured_default_weights(K) if weights is None else weights)
    W_ind = np.atleast_1d(np.asarray(wspec["ind"], dtype=float)).astype(float)
    W_bs = np.atleast_1d(np.asarray(wspec["bs"], dtype=float)).astype(float)
    if W_ind.size == 1 and K > 1:
        W_ind = np.full(K, float(W_ind[0]))
    if W_bs.size == 1 and K > 1:
        W_bs = np.full(K, float(W_bs[0]))
    if W_ind.size != K or W_bs.size != K:
        raise ValueError(f"close_ip_structured: {K} basis functions but "
                         f"{W_ind.size}/{W_bs.size} ind/bs weights")
    if np.any(np.isnan(W_ind)) or np.any(np.isnan(W_bs)) \
            or np.any(W_ind <= 0.0) or np.any(W_bs <= 0.0):
        raise ValueError("close_ip_structured: trust weights must be strictly "
                         "positive (numpy.inf allowed: hard-pins that "
                         "coefficient to 0)")
    sig_ind_up = W_ind_up = None
    if sigma_ind_up is not None:
        sig_ind_up = _sigma_ladder(sigma_ind_up, K, None,
                                   "close_ip_structured", "ind_up")
        if np.any(np.isinf(sig_ind_up)):
            raise ValueError(
                "close_ip_structured: sigma_ind_up = inf would mean a trust "
                "weight of 0, which this solver refuses on the symmetric "
                "ladder too (the coefficient would be unpenalised and the "
                "minimal-norm statement empty); use the soft solver if an "
                "unpenalised up side is really wanted")
        _one_sided_ladder_check(sig_ind_up, sigma_from_weights(
            {"ind": W_ind, "bs": W_bs}, K)["ind"], "close_ip_structured")
        W_ind_up = _weights_from_sigma(sig_ind_up)

    _lin = lambda y: float(trapezoid(w_lin * np.asarray(y, float), psi))
    ip_ind, ip_bs, ip_fix = _lin(j_ind), _lin(j_bs), _lin(j_fix)
    Ip_t = abs(float(Ip_target_signed))
    deficit = (float(Ip_target_signed) - float(c_affine)
               - ip_ind - ip_bs - ip_fix)
    A = np.array([_lin(Phi[k] * j_ind) for k in range(K)], dtype=float)
    B = np.array([_lin(Phi[k] * j_bs) for k in range(K)], dtype=float)

    rows = [np.concatenate([A, B])]
    rhs = [deficit]
    names = ["Ip"]
    phi0 = None
    if axis is not None:
        j_ind0 = float(axis["j_ind0"]); j_bs0 = float(axis["j_bs0"])
        j_fix0 = float(axis["j_fix0"]); j_ref0 = float(axis["j_ref0"])
        psi0 = float(axis["psi"])
        if not all(np.isfinite(v) for v in
                   (j_ind0, j_bs0, j_fix0, j_ref0, psi0)):
            raise RuntimeError("close_ip_structured: non-finite axis row "
                               f"{(psi0, j_ind0, j_bs0, j_fix0, j_ref0)}")
        phi0 = structured_basis_eval(basis_spec,
                                     np.array([psi0], dtype=float))[:, 0]
        rows.append(np.concatenate([phi0 * j_ind0, phi0 * j_bs0]))
        rhs.append(j_ref0 - j_ind0 - j_bs0 - j_fix0)
        names.append("axis current (q0)")
    li_model = li_row = None
    li_anchor = li_grad = None
    if li_target is not None:
        li_model, li_row, li_anchor, li_grad = _structured_li_row(
            psi, w_lin, Phi, j_ind, j_bs, j_fix, li_geom, Ip_target_signed,
            li_target, li_kind, "close_ip_structured",
            Ip_pinned=Ip_target_signed)
        rows.append(li_grad)
        rhs.append(li_row)
        names.append(f"l_i ({li_model['li_kind']}, on the Ip-closed manifold)")
    C_full = np.asarray(rows, dtype=float)                  # (m, 2K)
    d = np.asarray(rhs, dtype=float)                        # (m,)

    W_full = np.concatenate([W_ind, W_bs])
    free = np.isfinite(W_full)
    if not free.any():
        raise RuntimeError("close_ip_structured: every trust weight is "
                           "infinite -- no coefficient is free to move and "
                           "the closure cannot be imposed at all")
    C = C_full[:, free]
    Wf = W_full[free]

    row_norm = np.max(np.abs(C), axis=1)
    for i, rn in enumerate(row_norm):
        if not (rn > 0.0):
            raise RuntimeError(
                f"close_ip_structured: the '{names[i]}' constraint row is "
                "identically zero on the free coefficients -- the basis (or "
                "the trust weights pinning it) cannot move this constraint; "
                "refusing to pretend it was imposed")
    Cn = C / row_norm[:, None]
    dn = d / row_norm

    # ---- are the CONSTRAINT rows independent on the free coefficients? -----
    # This is the question the refusal below asks, and the only one it may
    # answer.  It is a property of the rows and the basis ALONE -- the prior
    # cannot make two independent constraints degenerate, and it must not be
    # able to trip this test.  Same test, same relative floor, same
    # `cond_rtol` value the soft solver uses on its hard rows.
    #
    # It used to be asked of the whole bordered matrix
    # ``[[2 diag(W/max W), Cn'], [Cn, 0]]``, whose smallest singular value is
    # bounded by the PRIOR's dynamic range: any W_max/W_min beyond ~1/cond_rtol
    # was reported as "the constraint rows are degenerate", which was both
    # false and the wrong diagnosis.  A tight one-sided `sigma_ind_up` -- the
    # setting the asymmetric prior exists for -- hit it at a sigma ratio of a
    # few hundred.
    # The test is on the RANK against the row count, not on the smallest
    # returned singular value: `svd` returns only min(m, n) of them, so with
    # MORE constraint rows than free coefficients (m > n -- e.g. the
    # `{"kind": "constant"}` basis, K = 1, carrying Ip + an axis row + an l_i
    # row) every returned value can be comfortably nonzero while the system is
    # unsatisfiable.  `_kkt` would then hand back the LEAST-SQUARES answer and
    # claim a hard KKT solve, with Ip no longer exact -- the one property this
    # channel is built on.  `cond_rtol` and its meaning are unchanged; this is
    # the same form the soft solver uses on its hard rows.
    sv_c = np.linalg.svd(Cn, compute_uv=False)
    _m = Cn.shape[0]
    _rank = int(np.sum(sv_c > float(cond_rtol) * sv_c[0]))
    if not np.all(np.isfinite(sv_c)) or _rank < _m:
        raise RuntimeError(
            f"close_ip_structured: singular KKT system (rank {_rank} of the "
            f"{_m} constraint row(s) against the relative floor "
            f"{float(cond_rtol):g}; smallest returned singular value "
            f"{sv_c[-1]:.4e} after normalisation) -- the {_m} constraint "
            f"row(s) are degenerate on this basis, or there are more of them "
            f"than the {int(np.count_nonzero(free))} free coefficient(s) can "
            "carry; Ip and the axis current cannot both be imposed on this "
            "split")

    def _kkt(Wf_now):
        """Minimal-norm solve for ONE fixed set of trust weights.

        ``min x' W x  s.t.  Cn x = dn``, solved in the substitution
        ``y = W^(1/2) x``, where the objective is the plain Euclidean norm and
        the prior's dynamic range has been divided out exactly:

        .. code-block:: text

            minimise ||y||  subject to  (Cn W^(-1/2)) y = dn
            y = pinv(Cn W^(-1/2)) dn                   (minimum-norm)
            x = W^(-1/2) y

        For full-row-rank ``Cn`` this is algebraically the bordered KKT
        solution ``x = W^-1 Cn' (Cn W^-1 Cn')^-1 dn`` -- the same minimiser,
        obtained in a scaling whose conditioning does not carry the prior.  A
        weight ratio of 1e12 is then an ordinary request, not a refusal.
        """
        Wn = np.asarray(Wf_now, dtype=float) / float(np.max(Wf_now))
        scal = 1.0 / np.sqrt(Wn)                 # W^(-1/2), all finite here
        Aw = Cn * scal[None, :]
        U, sv, Vt = np.linalg.svd(Aw, full_matrices=False)
        if not np.all(np.isfinite(sv)) or sv[-1] <= 0.0:
            raise RuntimeError(
                "close_ip_structured: the weighted constraint matrix "
                f"Cn W^(-1/2) is rank deficient (singular values {sv}) -- the "
                "free coefficients cannot carry these constraints")
        y = Vt.T @ ((U.T @ dn) / sv)
        x_now = np.zeros(2 * K, dtype=float)
        x_now[free] = scal * y
        return x_now, sv

    if W_ind_up is None:
        x, sv = _kkt(Wf)
        sign_pattern, n_sign_iter = None, 0
    else:
        # The scale-bound and finiteness checks below are deliberately OUTSIDE
        # the iteration: an intermediate sign pattern is not an answer, and
        # refusing on one would refuse a slice whose actual minimiser is
        # perfectly in bounds.
        def _solve_for(pat):
            W_now = np.concatenate(
                [np.where(np.asarray(pat, dtype=bool), W_ind_up, W_ind), W_bs])
            return _kkt(W_now[free])

        (x, sv), sign_pattern, n_sign_iter = _one_sided_sign_iterate(
            _solve_for, K, "close_ip_structured")
    a, b = x[:K], x[K:]

    s_ind = 1.0 + a @ Phi
    s_bs = 1.0 + b @ Phi
    lo, hi = scale_bounds
    probe = {"s_ind": s_ind, "s_bs": s_bs}
    if phi0 is not None:
        probe["s_ind"] = np.concatenate([s_ind, [1.0 + float(a @ phi0)]])
        probe["s_bs"] = np.concatenate([s_bs, [1.0 + float(b @ phi0)]])
    for nm, sp in probe.items():
        if not np.all(np.isfinite(sp)):
            raise RuntimeError(f"close_ip_structured: non-finite {nm}(psi)")
        if not (np.all(sp > lo) and np.all(sp < hi)):
            i = int(np.argmax(np.abs(sp - 1.0)))
            raise RuntimeError(
                f"close_ip_structured: {nm}(psi) reaches {float(sp[i]):.3f}, "
                f"outside {lo:g} < {nm} < {hi:g} (the bounds are STRICT, as "
                f"in close_ip and close_ip_q0) -- no minimal-norm radial "
                "closure exists within the scale bounds; the components do "
                "not add up "
                "to Ip and refusing to hide that behind a multiplier profile")

    # Effective scalar equivalents: the ONE number carrying the same current in
    # the Ip measure, so closure_health applies to them unchanged (see the
    # docstring).  A component with ~no current in the measure has no such
    # number; fall back to the plain grid mean and SAY so rather than dividing
    # by noise (the same relative floor close_ip refuses on).
    def _eff(s, j, ip):
        if abs(ip) < 1e-6 * Ip_t:
            return float(np.mean(s)), "unweighted grid mean (component " \
                "carries ~0 Ip in the measure)"
        return _lin(s * j) / ip, "Ip-weighted mean lin(s*j)/lin(j)"

    ohm_eff, ohm_eff_basis = _eff(s_ind, j_ind, ip_ind)
    bs_eff, bs_eff_basis = _eff(s_bs, j_bs, ip_bs)

    def _struct(s, j, mean):
        u = np.abs(w_lin * j)
        den = float(trapezoid(u, psi))
        if den <= 0.0:
            return float("nan")
        return float(np.sqrt(max(
            float(trapezoid(u * (s - mean) ** 2, psi)) / den, 0.0)))

    ip_hybrid = _lin(s_ind * j_ind + s_bs * j_bs + j_fix) + float(c_affine)
    ip_resid = ip_hybrid - float(Ip_target_signed)
    axis_resid = None
    if phi0 is not None:
        axis_resid = float((1.0 + a @ phi0) * j_ind0
                           + (1.0 + b @ phi0) * j_bs0 + j_fix0 - j_ref0)
    li_rec = _li_record(li_model, li_anchor, li_target, li_grad, x,
                        Ip_pinned=Ip_target_signed)

    return dict(
        s_ind=s_ind, s_bs=s_bs, a=a, b=b, basis=basis_spec,
        basis_K=int(K), basis_phi0=(None if phi0 is None else phi0),
        weights_ind=W_ind, weights_bs=W_bs,
        weights_ind_up=W_ind_up,
        sigma_ind_up=sig_ind_up,
        one_sided_ind=bool(W_ind_up is not None),
        sign_pattern=(None if sign_pattern is None
                      else tuple(bool(v) for v in sign_pattern)),
        n_sign_iter=int(n_sign_iter),
        weights_name=str(wspec.get("name", "custom")),
        constraints=tuple(names),
        Ip_lin_ind=ip_ind, Ip_lin_bs=ip_bs, Ip_lin_fix=ip_fix,
        deficit=float(deficit),
        A_row=A, B_row=B,
        ohm_scale_eff=float(ohm_eff), bs_scale_eff=float(bs_eff),
        ohm_scale_eff_basis=ohm_eff_basis, bs_scale_eff_basis=bs_eff_basis,
        structure_ind=_struct(s_ind, j_ind, ohm_eff),
        structure_bs=_struct(s_bs, j_bs, bs_eff),
        Ip_hybrid=float(ip_hybrid),
        ip_residual=float(ip_resid),
        ip_residual_pct=100.0 * float(ip_resid) / Ip_t,
        axis_residual=axis_resid,
        # The system actually inverted: Cn W^(-1/2).  Its conditioning carries
        # the prior's dynamic range by construction and is a diagnostic, never
        # an acceptance -- what IS tested is `constraint_cond` below, the
        # prior-independent independence of the constraint rows.
        kkt_singular_values=sv,
        kkt_cond=float(sv[0] / sv[-1]),
        constraint_singular_values=sv_c,
        constraint_cond=float(sv_c[0] / sv_c[-1]),
        solver="hard-KKT",
        **li_rec,
    )


def sigma_from_weights(weights=None, K=None):
    r"""``{"ind": sigma, "bs": sigma}`` from a trust-weight dict, ``sigma = W^-1/2``.

    The bridge between the hard channel's Tikhonov weights and the soft
    channel's prior widths, so the two solvers are given the SAME prior and a
    hard/soft comparison is a comparison of the likelihood, not of two
    different priors.  ``W = inf`` (hard-pinned to 0) maps to ``sigma = 0``,
    which the soft solver also treats as a pin; ``W -> 0`` maps to
    ``sigma = inf``, an unpenalised coefficient.
    """
    wspec = dict(weights if weights is not None
                 else structured_default_weights(
                     len(STRUCTURED_WEIGHTS_PHYSICS["ind"]) if K is None
                     else K))
    out = {"name": str(wspec.get("name", "custom"))}
    for key in ("ind", "bs"):
        W = np.atleast_1d(np.asarray(wspec[key], dtype=float)).astype(float)
        if K is not None and W.size == 1 and int(K) > 1:
            W = np.full(int(K), float(W[0]))
        if np.any(np.isnan(W)) or np.any(W < 0.0):
            raise ValueError("sigma_from_weights: trust weights must be "
                             "non-negative and not NaN")
        with np.errstate(divide="ignore"):
            out[key] = np.where(np.isinf(W), 0.0,
                                np.where(W > 0.0, 1.0 / np.sqrt(W), np.inf))
    return out


def _weights_from_sigma(sigma):
    """``W = sigma^-2``, the inverse of :func:`sigma_from_weights`.

    Recorded so a soft run's ``weights_ind``/``weights_bs`` read on the same
    scale as a hard run's, which is what lets the two be compared row by row
    in the campaign CSV.  ``sigma = 0`` (pinned) -> ``inf``; ``sigma = inf``
    (unpenalised) -> ``0``.
    """
    s = np.asarray(sigma, dtype=float)
    out = np.zeros_like(s)
    pinned = s <= 0.0
    free = np.isfinite(s) & ~pinned
    out[pinned] = np.inf
    out[free] = 1.0 / s[free] ** 2
    return out


def _sigma_ladder(spec, K, K_default, who, name):
    """A validated (K,) sigma ladder, with the scalar broadcast rule."""
    if spec is None:
        spec = K_default
    s = np.atleast_1d(np.asarray(spec, dtype=float)).astype(float)
    if s.size == 1 and K > 1:
        s = np.full(K, float(s[0]))
    if s.size != K:
        raise ValueError(f"{who}: {K} basis functions but {s.size} {name} "
                         "sigmas")
    if np.any(np.isnan(s)) or np.any(s < 0.0):
        raise ValueError(f"{who}: {name} sigmas must be non-negative and not "
                         "NaN (0 hard-pins that coefficient to 0, inf leaves "
                         "it unpenalised)")
    return s


def close_ip_structured_soft(psi_N, w_lin, c_affine, Ip_target_signed,
                             Ip_sigma, j_ind, j_bs, j_fix,
                             basis=None, sigma_ind=None, sigma_bs=None,
                             li_target=None, li_sigma=None, li_kind="li_1",
                             li_geom=None, axis=None, axis_sigma=None,
                             scale_bounds=(0.2, 5.0), rtol=1e-10,
                             max_iter=100, cond_rtol=1e-6,
                             sigma_ind_up=None):
    r"""The POSTERIOR-MODE structured closure: Ip and l_i as measurements.

    :func:`close_ip_structured` treats Ip (and the axis current, and l_i) as
    things that are *true*: it imposes them exactly and lets the trust norm
    pick among the profiles that satisfy them.  That is the right statement
    when the targets carry no error bar.  They do.  Ip is known to the
    magnetics' own accuracy, and a reconstruction's l_i is a fitted quantity
    with a real uncertainty and a definition offset.  Imposing an uncertain
    number exactly spends the whole prior budget matching noise.

    So minimise, over the same 2K coefficients ``x = (a, b)``,

    .. math::

        \Phi(x) = \sum_k \frac{a_k^2}{\sigma^{\rm ind}_k{}^2}
                + \sum_k \frac{b_k^2}{\sigma^{\rm bs}_k{}^2}
                + \frac{(I_p[x] - I_p^{\rm t})^2}{\sigma_{I_p}^2}
                + \frac{(l_i[x] - l_i^{\rm t})^2}{\sigma_{l_i}^2}
                + \frac{(j_0[x] - j_0^{\rm ref})^2}{\sigma_{\rm axis}^2},

    the negative log posterior of a Gaussian prior on the multiplier
    deviations against Gaussian measurements of Ip, l_i and (optionally) the
    on-axis current.  The prior widths are ``sigma = W^-1/2`` of the hard
    channel's trust weights (:func:`sigma_from_weights`), so hard and soft are
    the same physics prior under two different statements about the data.

    **Which terms are present is decided by the sigmas, and only by them:**

    * ``Ip_sigma=None`` -> Ip is imposed EXACTLY (a hard equality constraint,
      eliminated on the constraint null space, not a large weight).  So is
      ``axis_sigma=None`` when an *axis* row is given.  Both are exact, not
      asymptotic, exactly like the ``W=inf`` pins in the hard channel.
    * ``li_target=None`` drops the l_i term; ``li_sigma=None`` WITH an
      ``li_target`` is refused -- a hard l_i belongs in
      :func:`close_ip_structured`, whose KKT solves it in closed form.
    * ``sigma=0`` for a basis coefficient pins it to 0 and removes it from the
      unknowns (the ``W=inf`` case); ``sigma=inf`` leaves it unpenalised.

    **One-sided (asymmetric) inductive prior.**  ``sigma_ind_up`` is an
    OPTIONAL second ladder for the inductive coefficients.  ``None`` (the
    default) is the symmetric prior, unchanged and taking exactly one
    Gauss-Newton solve.  When it is given, ``a_k`` is drawn from a prior of
    width ``sigma_ind[k]`` while it is negative and ``sigma_ind_up[k]`` while
    it is positive -- a half-Gaussian pair, so a tight ``sigma_ind_up`` lets a
    multiplier fall freely and resists it rising.  The negative log posterior
    stays convex in the PRIOR term and the answer is found by SIGN ITERATION
    around the Gauss-Newton solve (:func:`_one_sided_sign_iterate`, capped at
    :data:`SIGN_ITER_MAX` and refused loudly if it cycles).  A stable pattern
    is the EXACT minimiser when Ip is hard (``Ip_sigma=None``), where ``l_i``
    is linear in the coefficients and the objective is convex; with a FINITE
    ``Ip_sigma`` the l_i term is not convex, so a stable pattern is a
    stationary point of the true piecewise objective rather than a certified
    global one.  Benign at shipped sigmas (a 40-restart multistart agreed with
    this solver to 2.6e-8), but see :func:`_one_sided_sign_iterate` for the
    statement in full -- it is the one place the two are spelled out
    together.  The up ladder must pin (``sigma = 0``) and un-penalise
    (``sigma = inf``) in exactly the same places as the down ladder, so the
    set of unknowns and the hard-constraint null space never change with the
    sign.  Still no GS solves.

    *Why one-sided, and why on the inductive multiplier.*  EMPIRICALLY
    MOTIVATED with a supporting mechanism, not derived from one -- the full
    statement, including the circularity caveat (it was tuned on the same MSE
    chords that judge it), is in :func:`close_ip_structured`'s docstring.  In
    brief: FUSE's edge T_e collapses relative to the reference kinetics
    (ratios 2 to 40), so the edge resistivity runs high, current diffuses inward
    faster than it should, and FUSE's mid-radius inductive current is more
    likely OVER- than under-estimated.  The prior therefore lets mid-radius
    ``s_ind`` fall freely and resists it rising.

    **Method.**  Eight unknowns (2K), **no GS solves**.  Every term but l_i is
    linear in *x*; l_i is ``G S(x) sgn / Ip(x)^2`` -- linear over the square of
    a linear form, so it is exactly linear whenever Ip is hard and mildly
    nonlinear when it is not.  Gauss-Newton on the residual vector, each step
    from a least-squares solve of the Jacobian (never the normal equations),
    with a Levenberg damping fallback: if a full step does not reduce the sum
    of squares, lambda is raised (x10, starting from a 1e-8 floor) until it
    does or the step is abandoned.  **What bounds that search is the try
    count, not a lambda ceiling**: 16 tries from the 1e-8 floor reach 1e7, so
    the ``trial_lam > 1e12`` guard in the loop is a belt-and-braces bound that
    the budget always reaches first.  Acceptance is ``F_new <= F (1 + 1e-14)``
    -- a hair of slack so a step that is downhill in exact arithmetic but flat
    to rounding still counts, which makes it a NON-INCREASE test rather than a
    strict descent test; the ``rtol`` check on the next iteration then
    declares convergence.  Converged when the relative change in the objective
    falls below *rtol* (1e-10) or the step is below the same relative floor
    on *x*;
    a run that hits *max_iter* without either is a ``RuntimeError``, never a
    quietly-returned half-solution.

    **Refusals** (``RuntimeError``): non-finite input, a degenerate hard
    constraint system (against ``cond_rtol``), non-convergence, and -- as in
    the hard channel -- an ``s_ind``/``s_bs`` that leaves *scale_bounds*
    anywhere on the grid.

    Returns the same dict shape as :func:`close_ip_structured` (so every
    downstream recorder and :func:`closure_health` work unchanged) plus the
    soft-specific block: ``sigma_*``, ``residual_sigma_*`` (each measurement
    residual in units of its own sigma), ``objective``, ``n_iter``,
    ``lm_lambda`` and ``prior_chi2``.

    **``Ip_hybrid`` is a POSTERIOR, not the target.**  With a finite
    *Ip_sigma* the returned closure integrates to an Ip that differs from
    ``Ip_target_signed`` by design -- typically 0.02-0.1 % at ``sigma_Ip`` =
    0.5 % of Ip -- and ``ip_residual`` / ``residual_sigma_Ip`` say by how much.
    Callers that check the round trip of an ASSEMBLED hybrid must compare it
    against ``Ip_hybrid``, not against the measurement: the first is an algebra
    check, the second is a statement about the data that this channel exists
    to relax.
    """
    from scipy.integrate import trapezoid

    psi = np.asarray(psi_N, dtype=float)
    w_lin = np.asarray(w_lin, dtype=float)
    j_ind = np.asarray(j_ind, dtype=float)
    j_bs = np.asarray(j_bs, dtype=float)
    j_fix = np.asarray(j_fix, dtype=float)
    if not (psi.shape == w_lin.shape == j_ind.shape == j_bs.shape
            == j_fix.shape):
        raise ValueError("close_ip_structured_soft: psi_N, w_lin and the "
                         "three current components must share one grid, got "
                         f"{psi.shape}, {w_lin.shape}, {j_ind.shape}, "
                         f"{j_bs.shape}, {j_fix.shape}")
    for nm, arr in (("w_lin", w_lin), ("j_ind", j_ind), ("j_bs", j_bs),
                    ("j_fix", j_fix)):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"close_ip_structured_soft: non-finite {nm}")
    if not (np.isfinite(c_affine) and np.isfinite(Ip_target_signed)):
        raise RuntimeError("close_ip_structured_soft: non-finite c_affine or "
                           "Ip_target_signed")

    basis_spec = dict(STRUCTURED_BASIS_DEFAULT if basis is None else basis)
    Phi = structured_basis_eval(basis_spec, psi)             # (K, N)
    K = Phi.shape[0]
    _dflt = sigma_from_weights(None, K)
    sig_ind = _sigma_ladder(sigma_ind, K, _dflt["ind"],
                            "close_ip_structured_soft", "ind")
    sig_bs = _sigma_ladder(sigma_bs, K, _dflt["bs"],
                           "close_ip_structured_soft", "bs")
    sig_ind_up = None
    if sigma_ind_up is not None:
        sig_ind_up = _sigma_ladder(sigma_ind_up, K, None,
                                   "close_ip_structured_soft", "ind_up")
        _one_sided_ladder_check(sig_ind_up, sig_ind,
                                "close_ip_structured_soft")

    _lin = lambda y: float(trapezoid(w_lin * np.asarray(y, float), psi))
    ip_ind, ip_bs, ip_fix = _lin(j_ind), _lin(j_bs), _lin(j_fix)
    Ip_t = abs(float(Ip_target_signed))
    Ip0 = float(c_affine) + ip_ind + ip_bs + ip_fix
    A = np.array([_lin(Phi[k] * j_ind) for k in range(K)], dtype=float)
    B = np.array([_lin(Phi[k] * j_bs) for k in range(K)], dtype=float)
    ip_row = np.concatenate([A, B])

    phi0 = None
    axis_row = None
    if axis is not None:
        j_ind0 = float(axis["j_ind0"]); j_bs0 = float(axis["j_bs0"])
        j_fix0 = float(axis["j_fix0"]); j_ref0 = float(axis["j_ref0"])
        psi0 = float(axis["psi"])
        if not all(np.isfinite(v) for v in
                   (j_ind0, j_bs0, j_fix0, j_ref0, psi0)):
            raise RuntimeError("close_ip_structured_soft: non-finite axis row "
                               f"{(psi0, j_ind0, j_bs0, j_fix0, j_ref0)}")
        phi0 = structured_basis_eval(basis_spec,
                                     np.array([psi0], dtype=float))[:, 0]
        axis_row = np.concatenate([phi0 * j_ind0, phi0 * j_bs0])
        axis0 = j_ind0 + j_bs0 + j_fix0

    li_model = li_anchor = li_grad0 = None
    if li_target is not None:
        if li_sigma is None:
            raise ValueError(
                "close_ip_structured_soft: li_target given with li_sigma=None. "
                "A HARD l_i is close_ip_structured(li_target=...), which "
                "solves it in closed form; this solver wants a finite sigma.")
        if not (np.isfinite(float(li_sigma)) and float(li_sigma) > 0.0):
            raise ValueError("close_ip_structured_soft: li_sigma must be "
                             f"finite and positive, got {li_sigma!r}")
        li_model, _rhs, li_anchor, li_grad0 = _structured_li_row(
            psi, w_lin, Phi, j_ind, j_bs, j_fix, li_geom, Ip_target_signed,
            li_target, li_kind, "close_ip_structured_soft")
    for nm, s in (("Ip_sigma", Ip_sigma), ("axis_sigma", axis_sigma)):
        if s is not None and not (np.isfinite(float(s)) and float(s) > 0.0):
            raise ValueError(f"close_ip_structured_soft: {nm} must be None "
                             f"(hard) or finite and positive, got {s!r}")

    # ---- unknowns: pinned (sigma == 0) coefficients leave the problem -------
    sig_full = np.concatenate([sig_ind, sig_bs])
    free = sig_full > 0.0
    if not free.any():
        raise RuntimeError("close_ip_structured_soft: every prior sigma is 0 "
                           "-- no coefficient is free to move and no "
                           "measurement can be fitted at all")
    n = int(free.sum())
    sig_f = sig_full[free]
    prior_active = np.isfinite(sig_f)                # sigma = inf: no row

    # ---- hard equality rows, eliminated on their null space -----------------
    hard_rows, hard_rhs, hard_names = [], [], []
    if Ip_sigma is None:
        hard_rows.append(ip_row)
        hard_rhs.append(float(Ip_target_signed) - Ip0)
        hard_names.append("Ip")
    if axis_row is not None and axis_sigma is None:
        hard_rows.append(axis_row)
        hard_rhs.append(j_ref0 - axis0)
        hard_names.append("axis current (q0)")
    if hard_rows:
        E = np.asarray(hard_rows, dtype=float)[:, free]
        f = np.asarray(hard_rhs, dtype=float)
        rn = np.max(np.abs(E), axis=1)
        for i, v in enumerate(rn):
            if not (v > 0.0):
                raise RuntimeError(
                    f"close_ip_structured_soft: the hard '{hard_names[i]}' "
                    "row is identically zero on the free coefficients -- the "
                    "basis (or the prior pins) cannot move it; refusing to "
                    "pretend it was imposed")
        En, fn = E / rn[:, None], f / rn
        U, sv_h, Vt = np.linalg.svd(En, full_matrices=True)
        rank = int(np.sum(sv_h > float(cond_rtol) * sv_h[0]))
        if rank < En.shape[0]:
            raise RuntimeError(
                f"close_ip_structured_soft: the {En.shape[0]} hard constraint "
                f"row(s) are degenerate (rank {rank} against the relative "
                f"floor {cond_rtol:g}) -- they cannot all be imposed on this "
                "basis")
        x_p = Vt[:rank].T @ ((U[:, :rank].T @ fn) / sv_h[:rank])
        N = Vt[rank:].T if rank < n else np.zeros((n, 0), dtype=float)
    else:
        x_p = np.zeros(n, dtype=float)
        N = np.eye(n, dtype=float)
    if N.shape[1] == 0 and hard_rows:
        # fully determined by the hard rows alone: nothing left to fit
        pass

    def _full(xf):
        x = np.zeros(2 * K, dtype=float)
        x[free] = xf
        return x

    def _gauss_newton(sig_f):
        """One posterior-mode solve for ONE fixed inductive prior ladder.

        Factored out so the one-sided prior can re-run it under a different
        sign pattern (:func:`_one_sided_sign_iterate`).  ``sig_f`` is the
        free-coefficient sigma vector actually in force; every other input
        (the unknown set, the hard-constraint particular solution ``x_p`` and
        null space ``N``, ``prior_active``) is fixed, which is exactly what
        :func:`_one_sided_ladder_check` guarantees.
        """
        def _resid_jac(z):
            xf = x_p + N @ z
            x = _full(xf)
            r, J = [], []
            # prior
            for i in np.nonzero(prior_active)[0]:
                e = np.zeros(n, dtype=float); e[i] = 1.0 / sig_f[i]
                r.append(xf[i] / sig_f[i]); J.append(e)
            if Ip_sigma is not None:
                r.append((Ip0 + ip_row @ x - float(Ip_target_signed))
                         / float(Ip_sigma))
                J.append(ip_row[free] / float(Ip_sigma))
            if axis_row is not None and axis_sigma is not None:
                r.append((axis0 + axis_row @ x - j_ref0) / float(axis_sigma))
                J.append(axis_row[free] / float(axis_sigma))
            if li_model is not None:
                li_x, _ipx = structured_li_of(li_model, x)
                r.append((li_x - float(li_target)) / float(li_sigma))
                J.append(structured_li_gradient(li_model, x)[free]
                         / float(li_sigma))
            r = np.asarray(r, dtype=float)
            Jx = np.asarray(J, dtype=float).reshape(r.size, n)
            return r, Jx @ N

        # ---- Gauss-Newton with a Levenberg damping fallback ---------------------
        nz = N.shape[1]
        z = np.zeros(nz, dtype=float)
        r, J = _resid_jac(z)
        F = float(r @ r)
        lam = 0.0
        n_iter = 0
        converged = nz == 0            # nothing free to fit: the hard rows decide
        for n_iter in range(1, int(max_iter) + 1):
            if converged:
                n_iter -= 1
                break
            step = None
            trial_lam = lam
            for _ in range(16):
                if trial_lam <= 0.0:
                    Ja, ra = J, -r
                else:
                    Ja = np.vstack([J, np.sqrt(trial_lam) * np.eye(nz)])
                    ra = np.concatenate([-r, np.zeros(nz)])
                d = np.linalg.lstsq(Ja, ra, rcond=None)[0]
                if not np.all(np.isfinite(d)):
                    trial_lam = max(10.0 * trial_lam, 1.0e-8)
                    continue
                r_new, J_new = _resid_jac(z + d)
                F_new = float(r_new @ r_new)
                # a hair of slack so a step that is downhill in exact arithmetic
                # but flat to rounding still counts as accepted
                if np.isfinite(F_new) and F_new <= F * (1.0 + 1.0e-14):
                    step = (d, r_new, J_new, F_new)
                    break
                trial_lam = max(10.0 * trial_lam, 1.0e-8)
                # Belt and braces: the 16-try budget above is what actually
                # bounds this search (1e-8 x 10^15 = 1e7), so this never
                # fires.  Kept so raising the budget cannot run lambda away.
                if trial_lam > 1.0e12:
                    break
            if step is None:
                # Damping cannot find a downhill step: either we are already at
                # the minimum to machine precision, or the model is degenerate.
                # Tell those apart with a SCALED gradient test -- an absolute one
                # is meaningless here, because a tight sigma_Ip puts entries of
                # order 1/sigma_Ip in J and the raw gradient inherits that scale.
                gnorm = float(np.max(np.abs(J.T @ r)))
                floor = float(rtol) * max(float(np.max(np.abs(J))), 1.0) \
                    * max(float(np.sqrt(F)), 1.0)
                if gnorm <= floor:
                    converged = True
                    break
                raise RuntimeError(
                    "close_ip_structured_soft: Levenberg damping could not find a "
                    f"descent step at objective {F:.6e} (scaled gradient "
                    f"{gnorm:.3e} > floor {floor:.3e}) -- the measurement rows and "
                    "the prior are inconsistent on this basis")
            d, r, J, F_new = step
            lam = 0.0 if trial_lam == 0.0 else max(trial_lam / 10.0, 1.0e-12)
            dF = abs(F - F_new)
            z = z + d
            stepped = float(np.max(np.abs(d))) <= float(rtol) * (
                1.0 + float(np.max(np.abs(z))))
            F, F_prev = F_new, F
            if dF <= float(rtol) * max(F, 1.0e-300) or stepped:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"close_ip_structured_soft: Gauss-Newton did not converge in "
                f"{max_iter} iterations (objective {F:.6e}); refusing to return a "
                "half-solved posterior mode")

        return z, F, n_iter, lam

    if sig_ind_up is None:
        z, F, n_iter, lam = _gauss_newton(sig_f)
        sig_f_used = sig_f
        sign_pattern, n_sign_iter = None, 0
    else:
        def _solve_for(pat):
            s_now = np.concatenate(
                [np.where(np.asarray(pat, dtype=bool), sig_ind_up, sig_ind),
                 sig_bs])[free]
            z_n, F_n, ni_n, lam_n = _gauss_newton(s_now)
            return (_full(x_p + N @ z_n), z_n, F_n, ni_n, lam_n, s_now)

        _out, sign_pattern, n_sign_iter = _one_sided_sign_iterate(
            _solve_for, K, "close_ip_structured_soft")
        _x, z, F, n_iter, lam, sig_f_used = _out

    x = _full(x_p + N @ z)
    a, b = x[:K], x[K:]
    s_ind = 1.0 + a @ Phi
    s_bs = 1.0 + b @ Phi
    lo, hi = scale_bounds
    probe = {"s_ind": s_ind, "s_bs": s_bs}
    if phi0 is not None:
        probe["s_ind"] = np.concatenate([s_ind, [1.0 + float(a @ phi0)]])
        probe["s_bs"] = np.concatenate([s_bs, [1.0 + float(b @ phi0)]])
    for nm, sp in probe.items():
        if not np.all(np.isfinite(sp)):
            raise RuntimeError(f"close_ip_structured_soft: non-finite {nm}(psi)")
        if not (np.all(sp > lo) and np.all(sp < hi)):
            i = int(np.argmax(np.abs(sp - 1.0)))
            raise RuntimeError(
                f"close_ip_structured_soft: {nm}(psi) reaches "
                f"{float(sp[i]):.3f}, outside {lo:g} < {nm} < {hi:g} (STRICT, "
                f"as everywhere in this module) -- the "
                "posterior mode has to drive a multiplier off the scale "
                "bounds to reconcile the measurements with the prior; that is "
                "a finding, not something to clamp")

    def _eff(s, j, ip):
        if abs(ip) < 1e-6 * Ip_t:
            return float(np.mean(s)), "unweighted grid mean (component " \
                "carries ~0 Ip in the measure)"
        return _lin(s * j) / ip, "Ip-weighted mean lin(s*j)/lin(j)"

    ohm_eff, ohm_eff_basis = _eff(s_ind, j_ind, ip_ind)
    bs_eff, bs_eff_basis = _eff(s_bs, j_bs, ip_bs)

    def _struct(s, j, mean):
        u = np.abs(w_lin * j)
        den = float(trapezoid(u, psi))
        if den <= 0.0:
            return float("nan")
        return float(np.sqrt(max(
            float(trapezoid(u * (s - mean) ** 2, psi)) / den, 0.0)))

    ip_hybrid = _lin(s_ind * j_ind + s_bs * j_bs + j_fix) + float(c_affine)
    ip_resid = ip_hybrid - float(Ip_target_signed)
    axis_resid = None
    if phi0 is not None:
        axis_resid = float((1.0 + a @ phi0) * j_ind0
                           + (1.0 + b @ phi0) * j_bs0 + j_fix0 - j_ref0)
    li_rec = _li_record(li_model, li_anchor, li_target, li_grad0, x)
    names = (["Ip" + ("" if Ip_sigma is None else " (soft)")]
             + ([] if axis_row is None else
                ["axis current (q0)" + ("" if axis_sigma is None
                                        else " (soft)")])
             + ([] if li_model is None else
                [f"l_i ({li_model['li_kind']}, soft)"]))
    prior_chi2 = float(np.sum(
        (x[free][prior_active] / sig_f_used[prior_active]) ** 2))

    return dict(
        s_ind=s_ind, s_bs=s_bs, a=a, b=b, basis=basis_spec,
        basis_K=int(K), basis_phi0=(None if phi0 is None else phi0),
        weights_ind=_weights_from_sigma(sig_ind),
        weights_bs=_weights_from_sigma(sig_bs),
        weights_name="sigma ladder (soft)",
        sigma_ind=sig_ind, sigma_bs=sig_bs,
        sigma_ind_up=sig_ind_up,
        one_sided_ind=bool(sig_ind_up is not None),
        sign_pattern=(None if sign_pattern is None
                      else tuple(bool(v) for v in sign_pattern)),
        n_sign_iter=int(n_sign_iter),
        weights_ind_up=(None if sig_ind_up is None
                        else _weights_from_sigma(sig_ind_up)),
        sigma_Ip=(None if Ip_sigma is None else float(Ip_sigma)),
        sigma_li=(None if li_sigma is None else float(li_sigma)),
        sigma_axis=(None if axis_sigma is None else float(axis_sigma)),
        constraints=tuple(names),
        Ip_lin_ind=ip_ind, Ip_lin_bs=ip_bs, Ip_lin_fix=ip_fix,
        deficit=float(Ip_target_signed) - Ip0,
        A_row=A, B_row=B,
        ohm_scale_eff=float(ohm_eff), bs_scale_eff=float(bs_eff),
        ohm_scale_eff_basis=ohm_eff_basis, bs_scale_eff_basis=bs_eff_basis,
        structure_ind=_struct(s_ind, j_ind, ohm_eff),
        structure_bs=_struct(s_bs, j_bs, bs_eff),
        Ip_hybrid=float(ip_hybrid),
        ip_residual=float(ip_resid),
        ip_residual_pct=100.0 * float(ip_resid) / Ip_t,
        residual_sigma_Ip=(None if Ip_sigma is None
                           else float(ip_resid) / float(Ip_sigma)),
        axis_residual=axis_resid,
        residual_sigma_axis=(None if (axis_resid is None or axis_sigma is None)
                             else float(axis_resid) / float(axis_sigma)),
        residual_sigma_li=(None if li_model is None else
                           float(li_rec["li_predictor_residual"])
                           / float(li_sigma)),
        objective=float(F), prior_chi2=prior_chi2,
        n_iter=int(n_iter), lm_lambda=float(lam),
        kkt_singular_values=None, kkt_cond=None,
        constraint_singular_values=None, constraint_cond=None,
        solver="soft-GaussNewton",
        **li_rec,
    )


def Ip_fsa_integral(eq, psi_N, j_profile, convention="jphi-linterp",
                    psi_pad=_FSA_PSI_PAD, pprime_sign=1.0, geom=None):
    r"""Plasma current [A] carried by a bouquet current profile.

    The physically-exact axisymmetric current integral, evaluated on *eq*'s
    flux-surface geometry -- see the module comment above for the two profile
    conventions and the measured accuracy of each.  Pass a *geom* from
    :func:`fsa_current_geometry` to reuse a frozen snapshot's geometry (and to
    avoid re-tracing every surface on each call).
    """
    if geom is None:
        geom = fsa_current_geometry(eq, psi_N, psi_pad=psi_pad,
                                    want_pprime=(convention == "jphi-linterp"))
    w, c = Ip_fsa_weights(geom, convention=convention, pprime_sign=pprime_sign)
    return _trapezoid(w * np.asarray(j_profile, dtype=float),
                      geom["psi_N"]) + c


def eq_jphi_profile(geom, convention="jphi-linterp", eq=None, psi_N=None,
                    pprime_sign=1.0):
    r"""The equilibrium's OWN current profile on ``geom``'s grid, in *convention*.

    Reconstructed from the Grad-Shafranov source functions -- ``jphi-linterp``
    gives ``<R> P' + <1/R> FF'/mu0``, ``fsa`` gives
    ``(P' + <1/R^2> FF'/mu0)/<1/R>`` -- so that
    ``Ip_fsa_integral(..., this profile) == the equilibrium's true I_p``.  That
    round trip is the self-consistency check the measure is validated by
    (measured: +0.0055 % on the D3D-like anchor).

    Needs ``F`` and ``F'`` as well as ``P'``, so it takes *eq* and re-reads
    ``get_profiles`` unless *geom* already carries ``FFp``.
    """
    FFp = geom.get("FFp")
    if FFp is None:
        if eq is None:
            raise ValueError("eq_jphi_profile needs eq (or geom['FFp'])")
        prof = eq.get_profiles(psi=geom["psi_q"].copy())
        FFp = np.asarray(prof[1], dtype=float) * np.asarray(prof[2], dtype=float)
    FFp = float(pprime_sign) * np.asarray(FFp, dtype=float) / _FSA_MU0
    pprime = float(pprime_sign) * np.asarray(geom["pprime"], dtype=float)
    if convention == "jphi-linterp":
        return geom["R_avg"] * pprime + geom["inv_R"] * FFp
    if convention == "fsa":
        if geom["inv_R2"] is None:
            raise ValueError("the 'fsa' equilibrium profile needs <1/R^2>")
        return (pprime + geom["inv_R2"] * FFp) / geom["inv_R"]
    raise ValueError(f"unknown convention {convention!r}")


def Ip_flux_integral_vs_target(alpha, mygs, jtor_prof, spike_profile, psi_N, Ip_target):
    r'''! Compute difference between integrated a*j_tor+j_spike profile and Ip_target

    @param alpha Scaling factor to solve for
    @param jtor_prof Input j_inductive profile
    @param spike_profile Isolated j_bootstrap spike (a Gaussian), 0.0 everywhere else
    @param my_psi_N Local psi_N grid
    @param my_Ip_target Ip target
    '''
    prof = alpha*jtor_prof + spike_profile
    Ip_computed = mygs.flux_integral(psi_N, prof)
    return Ip_computed - Ip_target

def Hmode_profiles(edge=0.08, ped=0.4, core=2.5, rgrid=201, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    r'''! This function generates H-mode density and temperature profiles evenly spaced in your favorite 
    radial coordinate. Copied from https://omfit.io/_modules/omfit_classes/utils_fusion.html

    @param edge Separatrix height (float)
    @param ped Pedestal height (float)
    @param core On-axis profile height (float)
    @param rgrid Number of radial grid points (int)
    @param expin Inner core exponent for H-mode pedestal profile (float)
    @param expout Outer core exponent for H-mode pedestal profile (float)
    @param widthp Width of pedestal (float)
    @param xphalf Position of tanh (float, optional)
    @result H-mode profile array over radial grid
    '''

    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    xpsi = np.linspace(0, 1, rgrid)
    ones = np.ones(rgrid)

    val = 0.5 * a_t * (1.0 - np.tanh((xpsi - xphalf) / w_E1) - pconst) + edge * ones

    xtoped = xpsi / xped
    for i in range(0, rgrid):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout

    return val

def _scan_key(scan_key):
    """Convert a scan-value label (float, int, or str) to an HDF5-safe string.

    Returns ``None`` when *scan_key* is ``None`` (flat layout).
    """
    if scan_key is None:
        return None
    return str(scan_key)


def _resolve_h5(h5path_or_header):
    """Resolve an archive reference to an absolute ``.h5`` path.

    Accepts (duck-typed, so no import cycle):
      * a ``BouquetArchive`` (uses its ``.path``);
      * a ``Bouquet`` (uses ``config.output_header``);
      * a full path ending in ``.h5`` (returned unchanged);
      * a bare *header* stem (``<header>.h5`` resolved to absolute).

    The single place archive references are normalized, so every reader accepts
    a run object, an archive, a header, or a path interchangeably (F1).
    """
    p = getattr(h5path_or_header, "path", None)          # BouquetArchive
    if isinstance(p, str) and p.endswith(".h5"):
        return p
    cfg = getattr(h5path_or_header, "config", None)      # Bouquet
    ref = getattr(cfg, "output_header", None) if cfg is not None else h5path_or_header
    s = str(ref)
    return s if s.endswith(".h5") else os.path.abspath(f"{s}.h5")


def _default_scan_key(ref, scan_key):
    """Fill a missing ``scan_key`` from a ``Bouquet`` reference's config."""
    if scan_key is None:
        cfg = getattr(ref, "config", None)
        if cfg is not None:
            return getattr(cfg.generation, "scan_key", None)
    return scan_key


def _group_path(scan_key, count):
    """Return the internal HDF5 group path for a given entry."""
    bkey = _scan_key(scan_key)
    if bkey is not None:
        return f"scan/{bkey}/{int(count)}"
    return str(int(count))


# ====================================================================
#  Database lifecycle
# ====================================================================
def initialize_equilibrium_database(header):
    """
    Create (or open) the top-level HDF5 database file on disk.

    Stamps ``schema_version`` / ``bouquet_version`` / ``created`` at creation
    (the single chokepoint every archive passes through), so the ABSENCE of
    ``schema_version`` reliably identifies a pre-v2 legacy file to readers.
    ``config_json`` provenance is added separately by :func:`write_provenance`.

    Parameters
    ----------
    header : str
        Base name for the database.  File will be ``<header>.h5``.

    Returns
    -------
    db_path : str
        Absolute path to the HDF5 file.
    """
    import datetime
    from . import __version__
    db_path = os.path.abspath(f"{header}.h5")
    with h5py.File(db_path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = datetime.datetime.now().isoformat(
                timespec="seconds")
    return db_path


# HDF5 archive schema version -- imported from the schema module (now v2:
# bare dataset names + units attrs, fixed eqdsk/pfile names, unified coils).
from .schema import SCHEMA_VERSION


def write_provenance(h5path_or_header, config=None, scan_key=None):
    """Stamp provenance onto an archive: schema/version/timestamp + config JSON.

    File-level attrs ``schema_version``, ``bouquet_version``, ``created`` (set
    once) and ``updated``; and, when a :class:`~bouquet.BouquetConfig` is given,
    a ``config_json`` dataset at the file root mirrored into the
    ``scan/<key>/`` group when ``scan_key`` is provided (each slice can carry its
    own config). The per-scan copies are AUTHORITATIVE; the root copy is only
    the most recent write and is overwritten by each slice of a multi-slice
    sweep -- :func:`load_config` therefore disambiguates by scan and never
    silently serves a stale root copy. Called by ``Bouquet.generate`` /
    ``run_shard`` / ``merge_archives``; safe to call repeatedly.
    """
    import datetime
    from . import __version__
    path = _resolve_h5(h5path_or_header)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with h5py.File(path, "a") as hf:
        hf.attrs["schema_version"] = int(SCHEMA_VERSION)
        hf.attrs["bouquet_version"] = str(__version__)
        if "created" not in hf.attrs:
            hf.attrs["created"] = now
        hf.attrs["updated"] = now
        if config is not None:
            cj = config.to_json()
            targets = [hf]
            bkey = _scan_key(scan_key)
            if bkey is not None:
                targets.append(hf.require_group(f"scan/{bkey}"))
            for grp in targets:
                if "config_json" in grp:
                    del grp["config_json"]
                grp.create_dataset("config_json", data=cj)


def load_config(h5path_or_header, scan_key=None):
    """Reconstruct the :class:`~bouquet.BouquetConfig` stored in an archive.

    The per-scan ``config_json`` copies are authoritative (the root copy is
    just the most recent write). With an explicit ``scan_key``, that scan's
    copy is read. With ``scan_key=None``: a single stored per-scan config is
    served directly, but a MULTI-scan archive raises and lists the keys --
    each slice of a sweep carried its own config, so "the" config is
    ambiguous and the (last-writer-wins) root copy would silently describe
    only the final slice. Raises a clear error on pre-provenance archives.
    """
    from .config import BouquetConfig
    path = _resolve_h5(h5path_or_header)
    with h5py.File(path, "r") as hf:
        node = None
        bkey = _scan_key(scan_key)
        scan_cfgs = ([k for k in hf["scan"] if f"scan/{k}/config_json" in hf]
                     if "scan" in hf else [])
        if bkey is not None:
            if f"scan/{bkey}/config_json" in hf:
                node = hf[f"scan/{bkey}/config_json"]
        elif len(scan_cfgs) == 1:
            node = hf[f"scan/{scan_cfgs[0]}/config_json"]
        elif len(scan_cfgs) > 1:
            raise KeyError(
                f"{path} holds per-scan configs for {len(scan_cfgs)} scan keys "
                f"({scan_cfgs}); pass scan_key=<one of these> -- each slice of "
                "a sweep carries its own config, so there is no single file "
                "config to return.")
        elif "config_json" in hf:
            node = hf["config_json"]
        if node is None:
            raise KeyError(
                f"{path} has no stored config for scan_key={scan_key!r} "
                f"(schema_version={hf.attrs.get('schema_version', 'absent')}; "
                f"per-scan configs present: {scan_cfgs or 'none'}). Only files "
                "written by a provenance-aware Bouquet carry a config.")
        raw = node[()]
    return BouquetConfig.from_json(raw.decode() if isinstance(raw, bytes) else str(raw))


# ====================================================================
#  Per-equilibrium storage
# ====================================================================
_PROFILE_KEYS = [
    "psi_N",
    "j_phi",
    "j_BS",
    "j_BS,edge",
    "j_inductive",
    "n_e",
    "T_e",
    "n_i",
    "T_i",
    "w_ExB",
]


def _read_coil_names(grp):
    """Coil-name list from the ``coil_names`` string dataset (schema v2; F15)."""
    if "coil_names" not in grp:
        return []
    return [n.decode() if isinstance(n, bytes) else str(n)
            for n in grp["coil_names"][()]]


def store_equilibrium(
    header,
    count,
    eqdsk_filepath,
    psi_N,
    j_phi,
    j_BS,
    j_inductive,
    n_e,
    T_e,
    n_i,
    T_i,
    w_ExB,
    li1,
    li3,
    scan_key=None,
    pressure=None,
    pressure_thermal=None,
    j_BS_edge=None,
    pfile_bytes=None,
    Zeff=None,
    z_fast=None,
    Z_imp=None,
    coil_currents=None,
    psi_N_kinetic=None,
    homotopy_pass=None,
    homotopy_F_lim=None,
    homotopy_VSC_lim=None,
    max_F_drift_pct=None,
    max_VSC_drift_pct=None,
    in_spec=None,
    inspec_F_max=None,
    inspec_VSC_max=None,
    perturbed_lcfs_ref=None,
    l_i_target_used=None,
    l_i_uncertainty=None,
    x_points=None,
    diverted=None,
    aux=None,
    eq_fsa=None,
):
    """
    Write one perturbed equilibrium into the HDF5 database.

    Parameters
    ----------
    header : str
        Base name (same string passed to ``initialize_equilibrium_database``).
    count : int
        Perturbation index (typically 0 -- N-1).
    eqdsk_filepath : str
        Path to the ``.geqdsk`` / ``.eqdsk`` file.  Read as raw bytes so
        the Fortran-namelist formatting is preserved exactly.
    psi_N, j_phi, j_BS, j_inductive,
    n_e, T_e, n_i, T_i, w_ExB : array_like, 1-D
        Profile arrays.
    li1 : float
        Internal inductance l_i(1).
    li3 : float
        Internal inductance l_i(3).
    scan_key : str, float, int, or None
        Scan-point label.  When provided, an extra ``scan/{label}/``
        group layer is inserted.  ``None`` gives the flat layout.
    pressure : array_like or None
        1-D total pressure.
    j_BS_edge : array_like or None
        1-D isolated edge bootstrap current [A m^-2].
    pfile_bytes : bytes or None
        Raw p-file content to store alongside the g-file bytes.
    Zeff : array_like or None
        1-D effective charge profile (dimensionless).
    coil_currents : dict or None
        Coil currents {name: current_A} from TokaMaker.
    """
    db_path = os.path.abspath(f"{header}.h5")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found.  "
            f"Call initialize_equilibrium_database('{header}') first."
        )

    with open(eqdsk_filepath, "rb") as fh:
        eqdsk_bytes = fh.read()

    grp_path = _group_path(scan_key, count)

    with h5py.File(db_path, "a") as hf:
        # clean slate if this entry already exists
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        # ---- raw eqdsk (opaque binary -- bit-perfect; schema-v2 fixed
        # name, the group path carries the coordinates) --------------------
        from .schema import EQDSK_DS
        grp.create_dataset(EQDSK_DS, data=np.void(eqdsk_bytes))

        # ---- 1-D profiles -----------------------------------------------
        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "j_phi", j_phi)
        write_profile(grp, "j_BS", j_BS)
        write_profile(grp, "j_inductive", j_inductive)

        if j_BS_edge is not None:
            write_profile(grp, "j_BS,edge", j_BS_edge)
        write_profile(grp, "n_e", n_e)
        write_profile(grp, "T_e", T_e)
        write_profile(grp, "n_i", n_i)
        write_profile(grp, "T_i", T_i)
        write_profile(grp, "w_ExB", w_ExB)

        # ---- optional: kinetic profile grid (when different from psi_N) ----
        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        if pressure is not None:
            write_profile(grp, "pressure", pressure)
        # Thermal (main-ion + electron) pressure alongside the total stored above,
        # so plots can show what the kinetic profiles contribute vs the impurity +
        # fast-ion pressure the GS solve actually used (total - thermal).
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)

        # ---- optional: auxiliary perturbed profiles (the switchboard) ----
        # Stored on psi_N_kinetic. Named "aux_<name>" (e.g. aux_omega_tor,
        # aux_chi_e, aux_zeff) for downstream (transport) consumers.
        if aux:
            for _name, _arr in aux.items():
                grp.create_dataset(f"aux_{_name}",
                                   data=np.asarray(_arr, dtype=np.float64))

        # ---- scalars (group attributes) ----------------------------------
        grp.attrs["l_i(1)"] = float(li1)
        grp.attrs["l_i(3)"] = float(li3)
        grp.attrs["count"]  = int(count)
        if scan_key is not None:
            grp.attrs["scan_key"] = scan_key

        # ---- optional: p-file bytes ----------------------------------------
        # Per-draw pfile blobs are only stored for TEXT p-files (rewritten with
        # this draw's perturbed kinetics -- genuinely draw-specific data). A
        # binary IDA .cdf source cannot be draw-perturbed and would only be
        # duplicated verbatim into every draw (~190 MB x n_draws with the new
        # IDA-database files); it lives ONCE in _baseline and per-draw readers
        # (BouquetArchive.DrawView.pfile_bytes, load_equilibrium) fall back.
        from .schema import is_binary_profile_source
        if pfile_bytes is not None and not is_binary_profile_source(pfile_bytes):
            pf_ds = "pfile"
            grp.create_dataset(pf_ds, data=np.void(pfile_bytes))

        # ---- optional: Zeff profile ----------------------------------------
        if Zeff is not None:
            write_profile(grp, "Zeff", Zeff)

        # ---- optional: fast-ion charge density ------------------------------
        # On psi_N_kinetic, like the kinetics.  Fixed across draws, but written
        # per draw so a reader holding one entry can reproduce the thermal
        # electron density ne - z_fast the solve used; without it any consumer
        # re-deriving the impurity from (ne, ni) charges the beam to carbon.
        if z_fast is not None:
            write_profile(grp, "z_fast", z_fast)
        if Z_imp:
            grp.attrs["Z_imp"] = float(Z_imp)

        # ---- optional: coil currents ---------------------------------------
        if coil_currents is not None:
            import json
            names = list(coil_currents.keys())
            values = np.array([coil_currents[n] for n in names], dtype=np.float64)
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))

        # ---- homotopy / in-spec metadata (per-draw) -----------------------
        if homotopy_pass is not None:
            grp.attrs["homotopy_pass"] = int(homotopy_pass)
        if homotopy_F_lim is not None:
            grp.attrs["homotopy_F_lim"] = float(homotopy_F_lim)
        if homotopy_VSC_lim is not None:
            grp.attrs["homotopy_VSC_lim"] = float(homotopy_VSC_lim)
        if max_F_drift_pct is not None:
            grp.attrs["max_F_drift_pct"] = float(max_F_drift_pct)
        if max_VSC_drift_pct is not None:
            grp.attrs["max_VSC_drift_pct"] = float(max_VSC_drift_pct)
        if in_spec is not None:
            grp.attrs["in_spec"] = bool(in_spec)
        if inspec_F_max is not None:
            grp.attrs["inspec_F_max"] = float(inspec_F_max)
        if inspec_VSC_max is not None:
            grp.attrs["inspec_VSC_max"] = float(inspec_VSC_max)
        # The l_i_target this draw actually converged to.  Equal to the
        # bouquet's l_i_target when l_i_uncertainty=0; otherwise a per-
        # draw sample from N(l_i_target, l_i_uncertainty * l_i_target).
        # Stored so post-run analysis can verify the sampled
        # distribution + diagnose any draw whose realised l_i diverged
        # from its (intentionally perturbed) target.
        if l_i_target_used is not None:
            grp.attrs["l_i_target_used"] = float(l_i_target_used)
        if l_i_uncertainty is not None:
            grp.attrs["l_i_uncertainty"] = float(l_i_uncertainty)

        # ---- High-resolution LCFS reference for this draw.
        # Captured by perturb_kinetic_equilibrium via safe_trace_surf at
        # the same mygs state save_eqdsk was called from -- so this and
        # the baseline.recon_lcfs_ref are method-consistent (both 10k-pt
        # trace_surf at psi = 1 - psi_pad).  Eliminates the ~4 mm
        # sampling-noise floor that the 100-pt eqdsk RBBBS introduces
        # when used as the perturbed-side boundary in plot_traces.
        if perturbed_lcfs_ref is not None:
            grp.create_dataset(
                "perturbed_lcfs_ref",
                data=np.asarray(perturbed_lcfs_ref, dtype=np.float64),
            )

        # ---- TokaMaker-computed X-point(s) for this draw ------------------
        # The poloidal-field nulls returned by mygs.get_xpoints(), captured
        # at the SAME solver state the eqdsk was saved from.  This is the
        # authoritative X-point location (a true B_p=0 saddle), used by
        # plot_boundary_point_traces in place of any geometric corner guess.
        # Shape (N, 2) = [[R, Z], ...]; the active (boundary-defining) null
        # is the last row when ``diverted`` is True.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

        # ---- Live-equilibrium FSA block (optional subgroup) --------------
        # Captured from the converged TokaMaker equilibrium at the same state
        # the eqdsk was saved from, so this draw's own flux geometry enables
        # an exact toroidal<->parallel conversion at IMAS export
        # (physics.capture_equilibrium_fsa -> physics.toroidal_to_parallel).
        if eq_fsa:
            from .schema import EQ_FSA_GROUP, EQ_FSA_UNITS
            fsa_grp = grp.create_group(EQ_FSA_GROUP)
            for _name, _arr in eq_fsa.items():
                if _arr is None:
                    continue
                ds = fsa_grp.create_dataset(
                    _name, data=np.asarray(_arr, dtype=np.float64))
                _u = EQ_FSA_UNITS.get(_name)
                if _u:
                    ds.attrs["units"] = _u


def load_eq_fsa(header, count, scan_key=None):
    """Load one draw's live-equilibrium FSA block, or ``None`` if not captured.

    Returns a dict of 1-D arrays (``psi_N``, ``F``, ``avg_inv_R``,
    ``avg_inv_R2`` (present only when the exact quadrature succeeded),
    ``avg_B2``, ``q``, ``dV_dpsi``, ``f_trap``, ``B_avg``) -- the geometry
    :func:`bouquet.physics.toroidal_to_parallel` needs for an exact IMAS
    write-back. ``None`` for archives written without live capture (fall back
    to the baseline-ratio reconstruction).
    """
    from .schema import EQ_FSA_GROUP
    h5path = _resolve_h5(header)
    gp = _group_path(scan_key, count)
    with h5py.File(h5path, "r") as hf:
        if gp not in hf or EQ_FSA_GROUP not in hf[gp]:
            return None
        g = hf[gp][EQ_FSA_GROUP]
        return {k: np.array(g[k], dtype=float) for k in g}


def load_equilibrium(header, count, scan_key=None, eqdsk_out_dir=None):
    """
    Retrieve one equilibrium entry from the HDF5 database.

    Parameters
    ----------
    header : str
        Base name of the database.
    count : int
        Perturbation index.
    scan_key : str, float, int, or None
        Scan-point label (must match what was used at write time).
    eqdsk_out_dir : str or None, optional
        If given, the raw eqdsk is written to a file in this directory.

    Returns
    -------
    result : dict
        Keys: ``"eqdsk_filepath"``, ``"eqdsk_bytes"``,
        the 1-D array names, ``"l_i(1)"``, ``"l_i(3)"``,
        and optionally ``"pressure"``, ``"Zeff"``,
        ``"coil_currents"``, ``"pfile_bytes"``.
    """
    from .schema import EQDSK_DS

    db_path  = os.path.abspath(f"{header}.h5")
    grp_path = _group_path(scan_key, count)

    result = {}

    with h5py.File(db_path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {db_path}"
            )
        grp = hf[grp_path]

        # ---- eqdsk raw bytes (schema-v2 fixed name) ----------------------
        if EQDSK_DS not in grp:
            legacy = [k for k in grp if k.endswith(".eqdsk")]
            raise KeyError(
                f"no '{EQDSK_DS}' dataset in {grp_path} of {db_path}"
                + (f" -- found legacy-named {legacy}: this is a pre-v2 "
                   "archive; regenerate it with the current bouquet (or "
                   "read it via BouquetArchive, whose suffix scan still "
                   "resolves legacy eqdsk names)." if legacy else "."))
        eqdsk_bytes = bytes(grp[EQDSK_DS][()])
        result["eqdsk_bytes"] = eqdsk_bytes

        if eqdsk_out_dir is not None:
            os.makedirs(eqdsk_out_dir, exist_ok=True)
            # coordinate-carrying FILENAME (the fixed dataset name would
            # make every extracted draw clobber the same 'eqdsk' file)
            bkey = _scan_key(scan_key)
            stem = os.path.basename(header) + (f"_{bkey}" if bkey else "")
            out_path = os.path.join(eqdsk_out_dir, f"{stem}_{int(count)}.eqdsk")
            with open(out_path, "wb") as fh:
                fh.write(eqdsk_bytes)
            result["eqdsk_filepath"] = os.path.abspath(out_path)
        else:
            result["eqdsk_filepath"] = None

        # ---- 1-D arrays ------------------------------------------------
        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        # ---- scalars ----------------------------------------------------
        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

        # ---- optional: Zeff -----------------------------------------------
        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])
        if "z_fast" in grp:
            result["z_fast"] = np.array(grp["z_fast"])
        if "Z_imp" in grp.attrs:
            result["Z_imp"] = float(grp.attrs["Z_imp"])

        # ---- optional: p-file bytes ----------------------------------------
        # Text p-file sources are stored per draw (draw-perturbed); binary IDA
        # .cdf sources live ONCE in _baseline -- fall back there when the draw
        # carries no blob (see store_equilibrium / is_binary_profile_source).
        pf_ds = "pfile"
        if pf_ds in grp:
            result["pfile_bytes"] = bytes(grp[pf_ds][()])
        else:
            _bl = grp.parent.get("_baseline") if grp.parent is not None else None
            if _bl is not None and pf_ds in _bl:
                result["pfile_bytes"] = bytes(_bl[pf_ds][()])

        # ---- optional: coil currents ---------------------------------------
        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
            result["coil_currents"] = dict(zip(names, values))

    return result


# ====================================================================
#  Baseline (input) profile storage
# ====================================================================
def store_baseline_profiles(
    header,
    psi_N,
    ne,
    te,
    ni,
    ti,
    pressure,
    j_phi,
    sigma_ne,
    sigma_te,
    sigma_ni,
    sigma_ti,
    sigma_jphi,
    Ip_target,
    l_i_target,
    scan_key=None,
    l_i_scale=LI_SCALE,
    pressure_thermal=None,
    z_fast=None,
    Z_imp=None,
    eqdsk_bytes=None,
    pfile_bytes=None,
    psi_N_kinetic=None,
    coil_currents=None,
    coil_names=None,
    recon_lcfs_ref=None,
    x_points=None,
    diverted=None,
    aux_baselines=None,
    aux_sigmas=None,
    j_BS=None,
    j_inductive=None,
    source_kind=None,
):
    """
    Store the input (baseline) profiles and their uncertainties.

    For hierarchical layout (*scan_key* is not ``None``), these are
    stored in ``scan/{label}/_baseline/``.  For flat layout they go
    in ``_baseline/``.

    Parameters
    ----------
    eqdsk_bytes : bytes or None
        Raw baseline geqdsk file content.  Stored so that
        ``plot_geqdsk_bouquet`` can distinguish the true baseline
        from perturbed equilibria.
    pfile_bytes : bytes or None
        Raw baseline p-file content.

    This data is written once per scan-point and is required by the
    plotting GUI to be fully self-contained.
    """
    db_path = os.path.abspath(f"{header}.h5")
    bkey = _scan_key(scan_key)

    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    with h5py.File(db_path, "a") as hf:
        if grp_path in hf:
            del hf[grp_path]

        grp = hf.create_group(grp_path)

        write_profile(grp, "psi_N", psi_N)
        write_profile(grp, "n_e", ne)
        write_profile(grp, "T_e", te)
        write_profile(grp, "n_i", ni)
        write_profile(grp, "T_i", ti)
        write_profile(grp, "pressure", pressure)
        if pressure_thermal is not None:
            write_profile(grp, "pressure_thermal", pressure_thermal)
        # See store_equilibrium: needed to recover ne - z_fast downstream.
        if z_fast is not None:
            write_profile(grp, "z_fast", z_fast)
        if Z_imp:
            grp.attrs["Z_imp"] = float(Z_imp)
        write_profile(grp, "j_phi", j_phi)
        if j_BS is not None:
            write_profile(grp, "j_BS", j_BS)
        if j_inductive is not None:
            write_profile(grp, "j_inductive", j_inductive)
        write_profile(grp, "sigma_ne", sigma_ne)
        write_profile(grp, "sigma_te", sigma_te)
        write_profile(grp, "sigma_ni", sigma_ni)
        write_profile(grp, "sigma_ti", sigma_ti)
        write_profile(grp, "sigma_jphi", sigma_jphi)

        if psi_N_kinetic is not None:
            write_profile(grp, "psi_N_kinetic", psi_N_kinetic)

        # ---- auxiliary-profile baselines + sigmas (on the kinetic
        # grid), so plot_transport_profiles can draw input +/- sigma bands
        if aux_baselines:
            for _n, _a in aux_baselines.items():
                grp.create_dataset(f"aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))
        if aux_sigmas:
            for _n, _a in aux_sigmas.items():
                grp.create_dataset(f"sigma_aux_{_n}",
                                   data=np.asarray(_a, dtype=np.float64))

        grp.attrs["Ip_target"]  = float(Ip_target)
        grp.attrs["l_i_target"] = float(l_i_target)
        # Which l_i estimator `l_i_target` is on, so the archive is
        # self-describing and a reader can pick the matching per-draw attr
        # (`l_i(3)` for "iter(li3)", `l_i(1)` for the legacy "std(li1)").
        # Archives written before issue #20 lack this attr entirely --
        # readers must default to "std(li1)".
        grp.attrs["l_i_scale"] = str(l_i_scale)
        # Provenance marker for robust path detection in plotting (independent of
        # the source-decoupled aux switchboard): "imas" or "geqdsk".
        if source_kind is not None:
            grp.attrs["source_kind"] = str(source_kind)

        if eqdsk_bytes is not None:
            grp.create_dataset("eqdsk", data=np.void(eqdsk_bytes))
        if pfile_bytes is not None:
            grp.create_dataset("pfile", data=np.void(pfile_bytes))

        # ---- Recon-LCFS reference for downstream boundary-deviation
        # measurements.  Captured by the caller via mygs.trace_surf() at
        # the SAME mygs state where the baseline.eqdsk was saved, giving
        # a method-consistent reference (~10000 points) for plot_traces
        # and the per-stage bnd-diag inside generate_bouquet.  Using this
        # instead of the eqdsk's 100-pt boundary eliminates ~2-3 mm of
        # save_eqdsk sampling noise (see Probe 2 in save_eqdsk_probe.py).
        if recon_lcfs_ref is not None:
            grp.create_dataset(
                "recon_lcfs_ref",
                data=np.asarray(recon_lcfs_ref, dtype=np.float64),
            )

        # ---- TokaMaker-computed X-point(s) for the recon baseline --------
        # Same provenance as the per-draw ``x_points`` (mygs.get_xpoints()
        # at the recon-converged state).  plot_boundary_point_traces uses
        # this to decide whether the top/bottom traces are tracking a true
        # X-point and to anchor the deviation reference.
        if x_points is not None:
            xp_arr = np.asarray(x_points, dtype=np.float64).reshape(-1, 2)
            if xp_arr.size:
                grp.create_dataset("x_points", data=xp_arr)
        if diverted is not None:
            grp.attrs["diverted"] = bool(diverted)

        # Recon's converged coil currents (the perturbation reference).
        # Saved alongside profiles so post-processors can compute
        # absolute coil drift per draw without re-running recon.
        if coil_currents is not None:
            if coil_names is not None:
                names = list(coil_names)
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            elif isinstance(coil_currents, dict):
                names = list(coil_currents.keys())
                values = np.array([float(coil_currents[n]) for n in names],
                                  dtype=np.float64)
            else:
                names = [f"coil_{i}" for i in range(len(coil_currents))]
                values = np.asarray(coil_currents, dtype=np.float64)
            grp.create_dataset("coil_currents", data=values)
            grp.create_dataset("coil_names",
                               data=np.array(names, dtype=h5py.string_dtype()))


# ====================================================================
#  Introspection helpers (used by GUI and notebook API)
# ====================================================================
def discover_scan_keys(h5path_or_header):
    """
    Discover all scan values in an HDF5 equilibrium database.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- either a full ``.h5`` path or a bare header
        stem (``<header>.h5`` is resolved).

    Returns
    -------
    scan_keys : list[str] or None
        Sorted list of scan-value keys, or ``None`` if the file uses
        the flat layout (no ``scan/`` group).
    """
    h5path = _resolve_h5(h5path_or_header)
    with h5py.File(h5path, "r") as hf:
        if "scan" not in hf:
            return None
        keys = list(hf["scan"].keys())

    # Sort numerically when all keys look like numbers, otherwise
    # fall back to lexicographic order.
    try:
        return sorted(keys, key=float)
    except (ValueError, TypeError):
        return sorted(keys)


def count_equilibria(h5path_or_header, scan_key=None):
    """
    Count the number of perturbed equilibria stored for a scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    n : int
    """
    return len(list_equilibrium_indices(h5path_or_header, scan_key=scan_key))


def list_equilibrium_indices(h5path_or_header, scan_key=None):
    """Return the sorted list of integer draw indices actually stored.

    Band-rejected / failed draws leave GAPS in the index sequence (e.g.
    ``[0, 1, 2, 3, 4, 5, 7, ...]`` with draw 6 missing), so callers must
    iterate these indices rather than ``range(count_equilibria(...))`` --
    the latter assumes a contiguous ``0..n-1`` and KeyErrors on the gap.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        Scan-value key.  ``None`` for the flat layout.

    Returns
    -------
    list of int
        Sorted stored draw indices.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    with h5py.File(h5path, "r") as hf:
        parent = hf[f"scan/{bkey}"] if bkey is not None else hf
        return sorted(
            int(k) for k in parent.keys()
            if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit()
        )


def load_baseline_profiles(h5path_or_header, scan_key=None):
    """
    Load the baseline profiles and uncertainties for a given scan value.

    Parameters
    ----------
    h5path_or_header : str
        Archive reference -- full ``.h5`` path or bare header stem.
    scan_key : str, float, or None
        ``None`` for flat-layout files.

    Returns
    -------
    result : dict
        All stored baseline arrays and scalar attributes.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    if bkey is not None:
        grp_path = f"scan/{bkey}/_baseline"
    else:
        grp_path = "_baseline"

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Baseline group '{grp_path}' not found in {h5path}.  "
                f"Was store_baseline_profiles() called?"
            )
        grp = hf[grp_path]
        for key in grp.keys():
            result[key] = np.array(grp[key])
        for attr in grp.attrs:
            result[attr] = grp.attrs[attr]

    return result


def load_equilibrium_by_path(h5path_or_header, count, scan_key=None):
    """
    Load one perturbed equilibrium (profiles + scalars only).

    Like :func:`load_equilibrium`, but addressed by *scan_key* and does
    **not** extract the raw eqdsk bytes (use :func:`load_equilibrium` if
    you need those).  Accepts either a full ``.h5`` path or a bare header
    stem for *h5path_or_header*.
    """
    h5path = _resolve_h5(h5path_or_header)
    bkey = _scan_key(scan_key)
    if bkey is not None:
        grp_path = f"scan/{bkey}/{int(count)}"
    else:
        grp_path = str(int(count))

    result = {}
    with h5py.File(h5path, "r") as hf:
        if grp_path not in hf:
            raise KeyError(
                f"Group '{grp_path}' not found in {h5path}"
            )
        grp = hf[grp_path]

        for key in _PROFILE_KEYS:
            if key in grp:
                result[key] = np.array(grp[key])

        if "pressure" in grp:
            result["pressure"] = np.array(grp["pressure"])
        if "pressure_thermal" in grp:
            result["pressure_thermal"] = np.array(grp["pressure_thermal"])

        if "psi_N_kinetic" in grp:
            result["psi_N_kinetic"] = np.array(grp["psi_N_kinetic"])

        if "Zeff" in grp:
            result["Zeff"] = np.array(grp["Zeff"])
        if "z_fast" in grp:
            result["z_fast"] = np.array(grp["z_fast"])
        if "Z_imp" in grp.attrs:
            result["Z_imp"] = float(grp.attrs["Z_imp"])

        # auxiliary ("switchboard") perturbed profiles -- aux_zeff, aux_omega_tor,
        # aux_chi_e, ... on psi_N_kinetic -- so per-draw plots (draw_zeff, the aux
        # figure) can overlay the draws, not just the baseline.
        for _k in grp.keys():
            if str(_k).startswith("aux_"):
                result[str(_k)] = np.array(grp[_k])

        if "coil_currents" in grp:
            import json
            values = np.array(grp["coil_currents"])
            names = _read_coil_names(grp)
            result["coil_currents"] = dict(zip(names, values))

        result["l_i(1)"] = float(grp.attrs["l_i(1)"])
        result["l_i(3)"] = float(grp.attrs["l_i(3)"])

    return result


# ====================================================================
#  kinetic-profile regridding helper
# ====================================================================
def pchip_interp(x_src, y_src, x_out):
    r'''Shape-preserving (PCHIP) regrid of a profile onto a new grid.

    The single shared kin->eq regridding used everywhere a kinetic profile
    (measured on the IDA/kinetic ``psi_N`` grid) is resampled onto the
    equilibrium grid. Replaces the piecewise-LINEAR ``np.interp`` /
    ``interp1d(kind='linear')`` pattern: linear regrid leaves a slope kink
    at every source knot, so ANY subsequent derivative (including OFT's
    PCHIP-derivative bootstrap) is a staircase with plateaus between knots
    -- visible as stepped j_BS between the kinetic knots (verified on a
    weak-pedestal case: |d2 j_BS| correlates 0.99 with |d2 dTe/dpsi| of the
    linear regrid, identical in np.gradient and PCHIP OFT builds; direct
    PCHIP regrid cuts the step energy ~7x).

    PCHIP passes through the same source knot values (no alteration of the
    measured points), is monotone/shape-preserving between them (no spline
    ringing at a pedestal), and is C1 -- so downstream gradients are
    smooth. Duplicate source-x points are dropped (first occurrence kept,
    matching :func:`pchip_derivative`); outside the source range the
    endpoint values are held constant (the fill_value=(y[0], y[-1])
    convention of the interp1d calls this replaces).

    All regrid sites must use this helper (or none): mixing linear and
    PCHIP regrids between the recon and draw paths would break the
    sigma=0 consistency that verify_sigma0_consistency() guards.
    '''
    from scipy.interpolate import PchipInterpolator

    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_out = np.asarray(x_out, dtype=float)
    x_u, idx = np.unique(x_src, return_index=True)
    y_u = y_src[idx]
    if x_u.size < 2:
        raise ValueError("pchip_interp: fewer than 2 unique source points")
    out = PchipInterpolator(x_u, y_u, extrapolate=False)(x_out)
    # hold endpoint values outside the source range (no poly extrapolation)
    out = np.where(x_out < x_u[0], y_u[0], out)
    out = np.where(x_out > x_u[-1], y_u[-1], out)
    return out


# ====================================================================
#  LCFS shape helper
# ====================================================================
def _shape_from_boundary(boundary_RZ):
    """LCFS shape params (R0, Z0, a, kappa, delta) from boundary (R,Z) points.

    Lives here rather than in ``bouquet.run`` because both the orchestrator and
    ``TokaMaker_interface`` need it, and importing it from ``run`` into
    ``TokaMaker_interface`` at runtime would close a dependency cycle (``run``
    already imports from ``TokaMaker_interface``).  ``bouquet.run`` re-exports
    it for callers that use its original location.
    """
    rz = np.asarray(boundary_RZ, dtype=float)
    R, Z = rz[:, 0], rz[:, 1]
    Rmax, Rmin, Zmax, Zmin = R.max(), R.min(), Z.max(), Z.min()
    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    kappa = (Zmax - Zmin) / (Rmax - Rmin)
    R_upper = R[int(np.argmax(Z))]
    R_lower = R[int(np.argmin(Z))]
    delta = 0.5 * ((R0 - R_upper) + (R0 - R_lower)) / a
    return R0, Z0, a, kappa, delta


# ====================================================================
#  eqdsk byte-stream helper
# ====================================================================
def read_eqdsk_from_bytes(raw_bytes, reader_func):
    """
    Call an existing eqdsk reader that expects a filename,
    but feed it in-memory bytes instead of a file on disk.
    """
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".eqdsk",
        delete=False,
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        result = reader_func(tmp_path)
    finally:
        os.remove(tmp_path)

    return result
