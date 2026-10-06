"""Physics reductions shared by the baseline resolvers.

Two conversions are needed to bring heterogeneous inputs into bouquet's internal
conventions:

  * :func:`isotropize_fast_pressure` -- collapse an anisotropic (gyrotropic)
    fast-ion pressure to a single scalar, since TokaMaker solves a scalar-pressure
    Grad-Shafranov equation.

  * :func:`parallel_to_toroidal` -- convert a flux-surface-averaged *parallel*
    current density <j.B>/B0 (the IMAS / neoclassical convention for j_ohmic,
    j_bootstrap, and the driven currents) to the *toroidal* current density
    <j_phi/R>/<1/R>. bouquet stores every current component as toroidal j_phi.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Map the legacy positional index of ``TokaMaker.get_q``'s ``ravgs`` array to the
# key used by newer OFT builds, which return ``ravgs`` as a dict.
_GET_Q_RAVG_INDEX = {"<R>": 0, "<1/R>": 1, "dV/dPsi": 2}


def q_ravg(ravgs, which: str):
    """Extract a flux-surface average from the ``ravgs`` element of ``TokaMaker.get_q``.

    Supports both the legacy positional array ``[<R>, <1/R>, dV/dPsi]`` and the
    newer dict form ``{'<R>', '<1/R>', '<1/R^2>', 'dV/dPsi'}`` returned by recent
    OpenFUSIONToolkit releases. ``which`` is one of ``'<R>'``, ``'<1/R>'``,
    ``'dV/dPsi'`` (note the dict inserted ``'<1/R^2>'``, so the positional index
    of ``dV/dPsi`` is unchanged at 2 in the legacy array).
    """
    if isinstance(ravgs, dict):
        return ravgs[which]
    return ravgs[_GET_Q_RAVG_INDEX[which]]


#: Reduction rules :func:`isotropize_fast_pressure` accepts.  ``"sum"`` is the
#: per-degree-of-freedom rule; the other three read the two fields as the full
#: perpendicular/parallel pressures of a gyrotropic tensor.
P_FAST_REDUCTIONS = ("trace", "mean", "perp", "sum")

#: Which storage convention each rule assumes for
#: ``pressure_fast_{parallel,perpendicular}``.  This is the factor-of-3 axis:
#: "per_dof" fields hold a third of the fast pressure each, "total" fields hold
#: the full directional pressures.
P_FAST_CONVENTION_OF_RULE = {
    "sum": "per_dof",
    "trace": "total",
    "mean": "total",
    "perp": "total",
}


def isotropize_fast_pressure(p_perp, p_par, method: str):
    """Reduce anisotropic fast-ion pressure to a scalar for the scalar-p GS solve.

    ``method`` is REQUIRED and has no default: the two families below differ by
    a factor of three on the same input, so there is no value that is safe to
    assume on a caller's behalf.  Readers that have a data dictionary to look at
    should call :func:`bouquet.io.imas.resolve_p_fast_reduction`, which picks the
    rule from the dd's recorded provenance.

    **Fields stored as the FULL directional pressures** (the IMAS/OMAS data
    dictionary reading of ``pressure_fast_parallel`` -- "fast (non-thermal)
    parallel pressure").  For a gyrotropic tensor
    ``P = p_par b b + p_perp (I - b b)``:

        method="trace"  ->  (2 * p_perp + p_par) / 3
        method="mean"   ->  (p_perp + p_par) / 2
        method="perp"   ->  p_perp

    ``"trace"`` is the textbook scalar pressure p = tr(P)/3 of a gyrotropic
    distribution and it preserves the fast-ion energy density
    (w = (1/2)(p_par + 2 p_perp) = (3/2) p_scalar), consistent with how
    kinetic-EFIT constrains the total stored pressure
    (p_tot = p_e + p_i + p_Z + p_fast). Use ``"perp"`` only if matching the
    diamagnetic magnetic response specifically; the rigorous alternative is a
    modified anisotropic Grad-Shafranov solve (out of scope for a scalar solver).

    **Fields stored PER DEGREE OF FREEDOM** (IMAS.jl / FUSE):

        method="sum"    ->  p_par + 2 * p_perp

    IMAS.jl's ``pressure`` expression is ``pressure_thermal +
    pressure_fast_parallel + 2*pressure_fast_perpendicular``, and its
    ``physics/fast.jl`` writes ``pressa/3`` into *each* of
    ``pressure_fast_parallel`` and ``pressure_fast_perpendicular``.  So on an
    IMAS.jl-written dd the two fields carry a third of the fast pressure each and
    ``"sum"`` recovers ``pressa``; ``"trace"`` would return one third of it.
    Measured on a set of beam-heated tokamak discharges reconstructed through
    FUSE, that shortfall was 8-35 % of the total pressure and closed to <2 % with
    ``"sum"``.  Conversely, ``"sum"`` on a dictionary-convention dd over-counts
    the fast pressure by exactly 3x.

    Note that ``"sum"`` is not a tensor reduction: on isotropic input it returns
    ``3*p``, not ``p``, because the input is a third of the pressure per degree of
    freedom rather than a directional pressure.  The invariant "every reduction
    is the identity on isotropic input" holds for the three ``"total"``-convention
    rules only.

    Inputs are per-species arrays on a common grid; the caller sums species.

    References
    ----------
    - Scalar pressure as tr(P)/3 of a gyrotropic tensor (standard kinetic theory).
    - Anisotropic Grad-Shafranov treatment: arXiv:1301.4714; J. Plasma Phys.,
      "Analysis of the isotropic and anisotropic Grad-Shafranov equation".
    - Kinetic-EFIT total-pressure constraint p_tot = p_e + p_i + p_Z + p_fast.
    - IMAS.jl (ProjectTorreyPines): ``src/expressions/dynamic.jl`` (the
      ``pressure`` expression) and ``src/physics/fast.jl`` (``pressa/3`` into
      each directional field).
    """
    p_perp = np.asarray(p_perp, dtype=float)
    p_par = np.asarray(p_par, dtype=float)
    if p_perp.shape != p_par.shape:
        raise ValueError(
            f"p_perp and p_par must have the same shape; got {p_perp.shape} vs {p_par.shape}"
        )
    if method == "trace":
        return (2.0 * p_perp + p_par) / 3.0
    if method == "mean":
        return (p_perp + p_par) / 2.0
    if method == "perp":
        return p_perp
    if method == "sum":
        return p_par + 2.0 * p_perp
    raise ValueError(
        f"unknown p_fast reduction method {method!r}; expected 'trace', 'mean', 'perp', or 'sum'"
    )


def parallel_to_toroidal(
    j_parallel,
    *,
    j_parallel_total=None,
    j_tor_total=None,
    geom: Optional[dict] = None,
):
    """Convert FSA parallel current density <j.B>/B0 to toroidal <j_phi/R>/<1/R>.

    bouquet stores all current components (inductive/ohmic, bootstrap, NBI, RF)
    as toroidal j_phi. IMAS/neoclassical outputs are parallel; this applies the
    per-flux-surface geometric conversion. The correction is typically modest but
    grows toward the edge / at low aspect ratio.

    This is needed in TWO places, not just on IMAS input:
      * IMAS-input bootstrap/ohmic/driven currents (use the *ratio* method).
      * bouquet's OWN recomputed bootstrap -- TokaMaker ``solve_with_bootstrap``
        returns a *parallel* j_BS, which must be converted here before it
        replaces the baseline j_BS (when ``recalculate_j_BS`` is True). Use the
        *analytic* method with the TokaMaker equilibrium's FSA metrics.

    Two methods:

    * **ratio** (preferred, used for FUSE input) -- when the source provides both
      the total parallel current and the total toroidal current (FUSE
      ``core_profiles`` carries ``j_total`` *and* ``j_tor``), form the per-surface
      factor ``c(psi) = j_tor_total / j_parallel_total`` and apply it to the
      component. Exact to the geometric mapping shared by field-aligned
      components; self-consistent with the source equilibrium.

    * **analytic** -- compute from equilibrium FSA metrics in ``geom`` when
      totals are unavailable (the reconstruction path, and bouquet's own
      per-draw ``solve_with_bootstrap`` output). Models the component as
      field-aligned, ``j = lambda(psi) B`` with ``lambda = <j.B>/<B^2>``
      (the standard treatment of the neoclassical banana-plateau /
      driven currents; the Pfirsch-Schlueter return current, which has
      ``<j_PS.B> = 0``, is by construction not part of the component), so

          j_tor = <j_phi/R>/<1/R> = lambda * F * <1/R^2> / <1/R>
                = <j.B> * F * <1/R^2> / (<B^2> <1/R>)
                = <j.B> / (F <1/R>) * [<B_phi^2>/<B^2>]

      using ``<B_phi^2> = F^2 <1/R^2>`` (exact, since ``B_phi = F/R``).
      ``geom`` keys:

        ``F``          flux function ``R*B_phi`` [T m]
        ``avg_inv_R``  ``<1/R>`` [1/m]
        ``avg_B2``     ``<B^2>`` [T^2]
        ``avg_inv_R2`` ``<1/R^2>`` [1/m^2], OPTIONAL -- when absent the
                       bracket ``<B_phi^2>/<B^2>`` is taken as 1,
                       neglecting ``<B_p^2>/<B^2> ~ (eps/q)^2`` (sub-1%%
                       at a DIII-D edge); the retained ``1/(F<1/R>)``
                       projection carries the O(eps^2) geometry.
        ``B0``         normalisation of the input, OPTIONAL (default 1):
                       pass the IMAS ``vacuum_toroidal_field`` B0 when
                       ``j_parallel`` is the IMAS convention ``<j.B>/B0``;
                       leave at 1 when passing raw ``<j.B>`` [T A/m^2].

    Pass either (``j_parallel_total``, ``j_tor_total``) for the ratio method or
    ``geom`` for the analytic method.
    """
    j_parallel = np.asarray(j_parallel, dtype=float)

    if j_parallel_total is not None and j_tor_total is not None:
        j_parallel_total = np.asarray(j_parallel_total, dtype=float)
        j_tor_total = np.asarray(j_tor_total, dtype=float)
        # Per-surface geometric factor c(psi) = j_tor_total / j_parallel_total,
        # shared by all field-aligned components. Guard the on-axis / low-current
        # surfaces where the total parallel current passes through zero: there the
        # ratio is ill-defined, so fall back to the nearest well-defined factor.
        eps = 1e-12 * np.nanmax(np.abs(j_parallel_total)) if j_parallel_total.size else 0.0
        good = np.abs(j_parallel_total) > eps
        if not np.any(good):
            raise ValueError("j_parallel_total is ~0 everywhere; cannot form ratio")
        c = np.ones_like(j_parallel_total)
        c[good] = j_tor_total[good] / j_parallel_total[good]
        if not np.all(good):
            # nearest-neighbour fill for the masked (near-zero) surfaces
            idx = np.arange(c.size)
            c[~good] = np.interp(idx[~good], idx[good], c[good])
        return j_parallel * c

    if geom is not None:
        try:
            F = np.asarray(geom["F"], dtype=float)
            avg_inv_R = np.asarray(geom["avg_inv_R"], dtype=float)
            avg_B2 = np.asarray(geom["avg_B2"], dtype=float)
        except KeyError as missing:
            raise ValueError(
                f"geom is missing required key {missing} "
                "(need 'F', 'avg_inv_R', 'avg_B2'; optional 'avg_inv_R2', 'B0')"
            ) from None
        j_dot_B = j_parallel * float(geom.get("B0", 1.0))
        # field-aligned component: j_tor = <j.B> F <1/R^2> / (<B^2> <1/R>);
        # F^2 <1/R^2> == <B_phi^2>, ~= <B^2> when <1/R^2> is unavailable
        # (neglects <B_p^2>/<B^2> ~ (eps/q)^2).
        if "avg_inv_R2" in geom and geom["avg_inv_R2"] is not None:
            bphi2_over_B2 = F**2 * np.asarray(geom["avg_inv_R2"], dtype=float) / avg_B2
        else:
            bphi2_over_B2 = 1.0
        return j_dot_B * bphi2_over_B2 / (F * avg_inv_R)

    raise ValueError(
        "provide either (j_parallel_total, j_tor_total) for the ratio method "
        "or geom for the analytic method"
    )


def toroidal_to_parallel(j_tor, *, geom: dict):
    """Inverse of :func:`parallel_to_toroidal` (analytic method).

    Convert bouquet's toroidal current density ``<j_phi/R>/<1/R>`` back to the
    IMAS flux-surface-averaged parallel current ``<j.B>/B0``. This is the
    write-back direction: bouquet stores every current component (ohmic,
    bootstrap, driven) as toroidal ``j_phi``; IMAS ``core_profiles.j_ohmic /
    j_bootstrap / j_total`` are parallel ``<j.B>/B0`` (EUROfusion/IMAS
    convention, with ``j_phi == <J^phi>/<1/R>``). Uses the draw's OWN
    flux-surface-averaged (FSA) geometry, so it is exact per surface -- unlike
    the interim baseline-ratio reconstruction it replaces.

    Physics (verified against Wesson 4th ed. sec 4.4 -- FSA
    ``<A> = oint (A/B_p) dl / oint dl/B_p``, and the field-aligned /
    Pfirsch-Schlueter decomposition with ``<j_PS.B> = 0``). For a field-aligned
    component ``j = lambda(psi) B`` with ``lambda = <j.B>/<B^2>`` and
    ``B_phi = F/R``:

        j_tor = <j_phi/R>/<1/R> = <j.B> F <1/R^2> / (<B^2> <1/R>)

    so the inverse is

        <j.B> = j_tor <B^2> <1/R> / (F <1/R^2>)
              = j_tor F <1/R> [<B^2>/<B_phi^2>]      (<B_phi^2> = F^2 <1/R^2>)

    and the IMAS parallel current is ``<j.B>/B0``.

    ``geom`` keys (all per-surface arrays unless noted):

        ``F``          flux function ``R*B_phi`` [T m]
        ``avg_inv_R``  ``<1/R>`` [1/m]
        ``avg_B2``     ``<B^2>`` [T^2]
        ``avg_inv_R2`` ``<1/R^2>`` [1/m^2], OPTIONAL -- when absent the exact
                       ``<B_phi^2> = F^2 <1/R^2>`` is unavailable and the
                       bracket ``<B^2>/<B_phi^2>`` is taken as 1, neglecting
                       ``<B_p^2>/<B^2> ~ (eps/q)^2`` (~<1%% at a DIII-D edge).
                       Provide it (from the captured live equilibrium) for a
                       machine-exact conversion.
        ``B0``         output normalisation (default 1): pass the IMAS
                       ``vacuum_toroidal_field`` B0 to return ``<j.B>/B0``;
                       leave at 1 to return raw ``<j.B>`` [T A/m^2].

    Round-trips with :func:`parallel_to_toroidal` (analytic) to machine
    precision when the same ``geom`` (including ``avg_inv_R2``) is used.
    """
    j_tor = np.asarray(j_tor, dtype=float)
    try:
        F = np.asarray(geom["F"], dtype=float)
        avg_inv_R = np.asarray(geom["avg_inv_R"], dtype=float)
        avg_B2 = np.asarray(geom["avg_B2"], dtype=float)
    except KeyError as missing:
        raise ValueError(
            f"geom is missing required key {missing} "
            "(need 'F', 'avg_inv_R', 'avg_B2'; optional 'avg_inv_R2', 'B0')"
        ) from None
    if "avg_inv_R2" in geom and geom["avg_inv_R2"] is not None:
        # <B^2>/<B_phi^2>, exact (B_phi = F/R -> <B_phi^2> = F^2 <1/R^2>)
        B2_over_Bphi2 = avg_B2 / (F**2 * np.asarray(geom["avg_inv_R2"], dtype=float))
    else:
        B2_over_Bphi2 = 1.0
    j_dot_B = j_tor * F * avg_inv_R * B2_over_Bphi2       # raw <j.B> [T A/m^2]
    return j_dot_B / float(geom.get("B0", 1.0))           # IMAS <j.B>/B0


def _fsa_over_contour(R, Z, Bp, field):
    """Flux-surface average <field> = oint(field/Bp)dl / oint(dl/Bp) over one
    closed (R,Z) contour with poloidal field magnitude ``Bp`` at each vertex.

    Segment-midpoint quadrature: each arc element ``dl`` carries the midpoint
    value of the integrand. The ``1/Bp`` weight's overall scale cancels in the
    ratio, so only the *shape* of ``Bp`` around the surface matters.
    """
    R = np.asarray(R, float); Z = np.asarray(Z, float); Bp = np.asarray(Bp, float)
    Rc = np.append(R, R[0]); Zc = np.append(Z, Z[0]); Bpc = np.append(Bp, Bp[0])
    dl = np.hypot(np.diff(Rc), np.diff(Zc))
    w = dl / (0.5 * (Bpc[:-1] + Bpc[1:]))            # dl/Bp at segment midpoints
    fmid = 0.5 * (np.append(field, field[0])[:-1] + np.append(field, field[0])[1:])
    return float(np.sum(fmid * w) / np.sum(w))


def _capture_exact_inv_R2(mygs, psi_hat, safe_trace):
    """Exact ``<1/R^2>`` per surface by FSA quadrature over traced contours.

    Traces each flux surface (``safe_trace(mygs, psi)`` -> (N,2) R,Z),
    evaluates the poloidal field ``Bp = hypot(B_R, B_Z)`` there, and forms the
    proper FSA. Also recomputes ``<1/R>`` and ``<B^2>`` the SAME way and returns
    them so the caller can self-validate against ``sauter_fc`` (a wrong B-field
    component order or a bad trace shows up as a ``<1/R>`` mismatch). Returns
    ``(inv_R2, inv_R_check, B2_check)`` on ``psi_hat``.
    """
    inv_R2 = np.empty(psi_hat.size)
    inv_R_chk = np.empty(psi_hat.size)
    B2_chk = np.empty(psi_hat.size)
    for k, ps in enumerate(psi_hat):
        rz = np.asarray(safe_trace(mygs, float(ps)), dtype=float)
        # Create the field interpolator AFTER the trace: safe_trace_surf swaps
        # the equilibrium pointer (copy_eq/replace_eq), which invalidates any
        # interpolator made earlier -> a stale eval() segfaults. Fresh each
        # surface (cheap) keeps it bound to the current (restored) equilibrium.
        Beval = mygs.get_field_eval("B")             # eval([R,Z]) -> [B_R,B_t,B_Z]
        R, Z = rz[:, 0], rz[:, 1]
        Bvec = np.array([Beval.eval(p[:2]) for p in rz], dtype=float)
        Bp = np.hypot(Bvec[:, 0], Bvec[:, 2])        # poloidal = (R,Z) comps
        B2 = Bvec[:, 0] ** 2 + Bvec[:, 1] ** 2 + Bvec[:, 2] ** 2
        inv_R2[k] = _fsa_over_contour(R, Z, Bp, 1.0 / R ** 2)
        inv_R_chk[k] = _fsa_over_contour(R, Z, Bp, 1.0 / R)
        B2_chk[k] = _fsa_over_contour(R, Z, Bp, B2)
    return inv_R2, inv_R_chk, B2_chk


def _native_fsa_inv_R2(geo_sauter, bfield, F, avg_inv_R, rtol=0.02):
    """Extract <1/R^2> from an EXTENDED ``sauter_fc`` output, or ``None``.

    Forward-compat for OpenFUSIONToolkit#312 / hansenc's branch, which adds
    <1/R^2> (and <B_phi^2>) to ``sauter_fc``. Until then ``sauter_fc`` returns
    geo ``[<R>, <1/R>, <a>]`` and bfield ``[<|B|>, <|B|^2>]``; the branch is
    expected to append the new averages to those sub-arrays. When it lands this
    reads them directly and we skip the trace quadrature entirely.

    Assumed layout: geo -> ``[<R>, <1/R>, <a>, <1/R^2>]`` (extra row = <1/R^2>),
    bfield -> ``[<|B|>, <|B|^2>, <B_phi^2>]``. That is a GUESS at the exact
    index, so the read is only trusted when it passes physical checks -- Jensen
    ``<1/R^2> >= <1/R>^2`` and, when <B_phi^2> is also present, the exact
    identity ``<B_phi^2> = F^2 <1/R^2>``. A wrong index fails these and the
    caller falls back to the (verified) quadrature. Confirm/simplify the indices
    against the merged PR signature.
    """
    geo = np.asarray(geo_sauter, dtype=float)
    if geo.ndim != 2 or geo.shape[0] < 4:
        return None                                  # no extra average present
    cand = geo[3]
    inv_R = np.asarray(avg_inv_R, dtype=float)
    if not np.all(np.isfinite(cand)) or np.any(cand < inv_R ** 2 - 1e-9):
        return None                                  # fails Jensen -> wrong read
    bf = np.asarray(bfield, dtype=float)
    if bf.ndim == 2 and bf.shape[0] >= 3:            # cross-check via the identity
        if not np.allclose(np.asarray(F, dtype=float) ** 2 * cand, bf[2], rtol=rtol):
            return None
    return cand


def capture_equilibrium_fsa(mygs, npsi: int = 257, psi_pad: float = 1e-3,
                            exact_inv_R2: bool = True, inv_R2_npsi=None,
                            inv_R2_check_rtol: float = 0.02):
    """Snapshot the live TokaMaker equilibrium's FSA metrics for exact export.

    Called at generate time on the converged ``mygs`` (right where the eqdsk is
    saved) so a perturbed draw's OWN flux-surface geometry travels with it into
    the archive, enabling an **exact** per-draw toroidal->parallel conversion at
    IDS write-back (:func:`toroidal_to_parallel`) instead of the interim
    baseline-ratio reconstruction.

    Returns a dict of 1-D arrays on a uniform ``psi_N`` grid of ``npsi`` points
    (default 257 to match the archived eqdsk; do not go below ~129 -- the edge
    bootstrap needs it):

        ``psi_N``       normalised poloidal flux, [npsi]
        ``F``           R*B_phi flux function [T m]           (get_profiles)
        ``avg_inv_R``   <1/R> [1/m]                           (sauter_fc)
        ``avg_inv_R2``  <1/R^2> [1/m^2]                       (exact quadrature)
        ``avg_B2``      <B^2> [T^2]                           (sauter_fc)
        ``q``           safety factor                         (get_q)
        ``dV_dpsi``     dV/dpsi                               (get_q, geo)
        ``f_trap``      trapped fraction f_c                  (sauter_fc)
        ``B_avg``       <|B|> [T]                             (sauter_fc)

    ``exact_inv_R2`` (default True) computes ``<1/R^2>`` -- which TokaMaker does
    not expose -- by flux-surface quadrature over traced contours
    (:func:`_capture_exact_inv_R2`), making :func:`toroidal_to_parallel`
    machine-exact instead of relying on ``<B_phi^2> ~= <B^2>`` (the
    ``<B_p^2>/<B^2> ~ (eps/q)^2 ~<1%%`` bracket). By default it is traced on the
    FULL ``npsi`` grid -- same resolution as every other metric, most accurate
    at the edge where the surfaces bunch up and the bootstrap peaks; the trace
    is cheap (a few ms/surface, ~2 s at npsi=257). ``inv_R2_npsi`` (default
    ``None`` = ``npsi``) can be set smaller to trace ``<1/R^2>`` coarsely and
    spline it onto ``psi_N`` -- only worth it for very large bouquets. The
    quadrature is **self-validated** each call: its independently-recomputed
    ``<1/R>`` must agree with ``sauter_fc`` to ``inv_R2_check_rtol`` (default
    2%), else ``avg_inv_R2`` is dropped (with a warning) and the conversion
    falls back to the ``<1%`` bracket -- never silently wrong.

    Set ``exact_inv_R2=False`` to skip the ``<1/R^2>`` surface traces entirely
    (bracket fallback) if the capture cost is ever material.

    Pure extraction (no re-solve); ``mygs`` is passed in so this module stays
    OFT-import-free / headless-safe.
    """
    import warnings
    psi_hat = np.linspace(psi_pad, 1.0 - psi_pad, int(npsi))

    # F(psi) = R*B_phi from the G-S source profiles
    _, F, _Fp, _P, _Pp = mygs.get_profiles(psi=psi_hat)

    # <1/R>, <B^2>, f_c from the Sauter flux-surface coefficients. OFT actually
    # returns (psi_hat, f_c, r_avgs, [<|B|>,<|B|^2>]) -- a leading psi grid the
    # docstring omits, matching get_q/get_profiles. Take the last three fields
    # so this is robust to the psi being present or absent.  r_avgs is a DICT
    # ({'<R>','<1/R>','<a>'}) on current OFT and a positional list
    # ([<R>,<1/R>,<a>]) on legacy builds -- q_ravg handles both.  (The legacy
    # positional read np.asarray(geo)[1] on a dict is 0-d -> IndexError, which
    # the per-draw capture guard swallowed as a WARN: every archive built on a
    # dict-era OFT silently lost its eq_fsa blocks until this was fixed.)
    f_c, geo_sauter, bfield = mygs.sauter_fc(psi=psi_hat)[-3:]
    avg_inv_R = q_ravg(geo_sauter, "<1/R>")
    if isinstance(bfield, dict):
        B_avg = bfield["<|B|>"]
        avg_B2 = bfield["<|B|^2>"]
    else:
        B_avg = np.asarray(bfield)[0]               # [<|B|>, <|B|^2>]
        avg_B2 = np.asarray(bfield)[1]

    # q and dV/dpsi (metadata; dV/dpsi also useful for volume integrals)
    _, q, geo_q, *_rest = mygs.get_q(psi=psi_hat, compute_geo=True)
    dV_dpsi = q_ravg(geo_q, "dV/dPsi")

    out = {
        "psi_N": psi_hat,
        "F": np.asarray(F, dtype=float),
        "avg_inv_R": np.asarray(avg_inv_R, dtype=float),
        "avg_B2": np.asarray(avg_B2, dtype=float),
        "q": np.asarray(q, dtype=float),
        "dV_dpsi": np.asarray(dV_dpsi, dtype=float),
        "f_trap": np.asarray(f_c, dtype=float),
        "B_avg": np.asarray(B_avg, dtype=float),
    }

    if exact_inv_R2:
        # Fast path 1: current OFT (OpenFUSIONToolkit#313) exposes <1/R^2>
        # directly in get_q's ravgs dict -- exact, no tracing needed.  Jensen
        # (<1/R^2> >= <1/R>^2) guards against a mislabeled field.
        if isinstance(geo_q, dict) and geo_q.get("<1/R^2>") is not None:
            _cand = np.asarray(geo_q["<1/R^2>"], dtype=float)
            if (np.all(np.isfinite(_cand))
                    and not np.any(_cand < out["avg_inv_R"] ** 2 - 1e-9)):
                out["avg_inv_R2"] = _cand
                return out
        # Fast path 2: legacy extended-array layout guess (OFT#312 era) --
        # validated, so a build without it falls through to the quadrature.
        native = _native_fsa_inv_R2(geo_sauter, bfield, out["F"], out["avg_inv_R"])
        if native is not None:
            out["avg_inv_R2"] = native
            return out

        from .utils import safe_trace_surf
        try:
            # trace <1/R^2> on the full psi_N grid by default (same resolution
            # as every other metric, no interpolation) -- cheap (~a few
            # ms/surface) and most accurate at the edge where flux surfaces
            # bunch up and the bootstrap peaks. inv_R2_npsi < npsi opts into a
            # coarser trace + spline for the rare large-bouquet cost case.
            n_ir2 = int(inv_R2_npsi) if inv_R2_npsi else psi_hat.size
            if n_ir2 >= psi_hat.size:
                grid = psi_hat
            else:
                grid = np.linspace(psi_pad, 1.0 - psi_pad, n_ir2)
            inv_R2_g, inv_R_chk, _B2_chk = _capture_exact_inv_R2(
                mygs, grid, safe_trace_surf)
            # self-consistency: my <1/R> quadrature vs sauter_fc's <1/R>
            ref = np.interp(grid, psi_hat, out["avg_inv_R"])
            rel = np.nanmax(np.abs(inv_R_chk - ref) / np.abs(ref))
            if not np.isfinite(rel) or rel > inv_R2_check_rtol:
                raise ValueError(
                    f"<1/R> FSA quadrature disagrees with sauter_fc by "
                    f"{rel:.1%} (> {inv_R2_check_rtol:.0%}) -- trace/B_p "
                    "capture unreliable")
            if grid is psi_hat:
                out["avg_inv_R2"] = inv_R2_g
            else:                                   # coarse -> full psi_N grid
                from scipy.interpolate import InterpolatedUnivariateSpline
                out["avg_inv_R2"] = InterpolatedUnivariateSpline(
                    grid, inv_R2_g, k=3)(psi_hat)
        except Exception as exc:                    # pragma: no cover - live-only
            warnings.warn(
                f"exact <1/R^2> capture failed ({exc}); IDS export will use the "
                "<B_phi^2>~=<B^2> bracket (~<1% at the edge). Set "
                "exact_inv_R2=False to silence.")
    return out


def effective_impurity_charge(ne, ni, zeff, min_dilution=1e-3):
    """Effective single-impurity charge Z_imp from a baseline (ne, ni, Zeff).

    Quasineutrality with one impurity species (main ion Z=1) gives, per flux
    surface::

        Z_imp = 1 + ne (Zeff - 1) / (ne - ni)

    Returns the median over surfaces with meaningful dilution
    (``(ne - ni)/ne > min_dilution`` and ``Zeff > 1``), which is robust to
    edge noise and to the axis where dilution can vanish. Returns ``None``
    when the baseline carries no dilution information at all (``ni ~= ne``
    everywhere, e.g. the IDA ``ni = ne`` workflow) -- in that case a Zeff
    draw cannot be mapped onto a main-ion density.
    """
    ne = np.asarray(ne, dtype=float)
    ni = np.asarray(ni, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dil = (ne - ni) / ne
        z = 1.0 + ne * (zeff - 1.0) / (ne - ni)
    ok = np.isfinite(z) & (dil > min_dilution) & (zeff > 1.0)
    if not np.any(ok):
        return None
    return float(np.median(z[ok]))


def impurity_charge_with_fast_ions(ne, ni, zeff, z_fast=None):
    """``(Z_imp, ne_th)`` when fast ions carry part of the neutralization.

    With a beam population, quasineutrality reads
    ``ne = ni + Z_imp*nz + z_fast`` -- only ``ne_th = ne - z_fast`` is
    neutralized by THERMAL ions.  A ``zeff`` normalized to the FULL ``ne``
    (the IMAS convention: thermal-species numerator over total electron
    density) must therefore be renormalized to ``zeff * ne / ne_th`` before
    the single-impurity inversion; passing the thermal ``ne`` with the
    full-``ne`` ``zeff`` recovers only half the bias (measured: raw 1.95,
    half-applied 3.19, true 6.00 for C6 at 25 % fast fraction).  Surfaces
    where ``z_fast >= ne`` get a non-finite renormalized zeff and are
    excluded by :func:`effective_impurity_charge`'s own validity mask.

    ``z_fast`` is optional.  ``None`` or an all-zero profile short-circuits
    to the plain :func:`effective_impurity_charge` inversion on the full
    ``ne``, so a source with no fast ions reproduces the pre-fast-ion result
    BIT-FOR-BIT.  The general branch would not: ``zeff * ne / ne`` is not an
    exact identity in floating point (~8 % of realistic values differ, at
    ~1 ulp), which is far below any physics scale but enough to move an
    archive that the repo's regeneration contract says must be reproducible.

    ASSUMPTION (load-bearing, not verified here): the source's ``zeff`` is
    normalized to the FULL ``ne`` with only THERMAL species in its numerator
    -- i.e. ``zeff = sum_thermal(n_s Z_s^2) / ne``.  That is what the
    ``zeff * ne / ne_th`` renormalization assumes and what the local
    fallback numerator in :mod:`bouquet.io.imas` builds.  If a producer's
    ``zeff`` already carries the fast-ion contribution in its numerator, the
    fast-ion charge is counted twice and ``Z_imp`` comes out too high.  The
    convention of any given producer has not been confirmed against a real
    data file; treat a source whose documented convention differs as out of
    scope for this helper.
    """
    ne = np.asarray(ne, dtype=float)
    if z_fast is None:
        return effective_impurity_charge(ne, ni, zeff), ne
    z_fast = np.asarray(z_fast, dtype=float)
    if not np.any(z_fast):
        return effective_impurity_charge(ne, ni, zeff), ne
    ne_th = np.maximum(ne - z_fast, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeff_th = np.where(ne_th > 0.0,
                           np.asarray(zeff, dtype=float) * ne / ne_th,
                           np.nan)
    return effective_impurity_charge(ne_th, ni, zeff_th), ne_th


def main_ion_density_from_zeff(ne, zeff, Z_imp, z_fast=None):
    """Main-ion density from (ne, Zeff) under single-impurity quasineutrality.

    ::

        ni  = ne (Z_imp - Zeff) / (Z_imp - 1)
        nz  = (ne - ni) / Z_imp            (the implied impurity density)

    For ``1 <= Zeff <= Z_imp`` this guarantees ``0 <= ni <= ne`` and
    ``nz >= 0`` -- the consistent (ne, ni, Zeff, nz) set that the independent
    per-channel draws cannot provide. Returns ``ni``.

    With a fast-ion charge profile ``z_fast`` the thermal quasineutrality is
    ``ni + Z_imp nz = ne - z_fast`` while ``zeff`` keeps the full-``ne``
    normalization, giving

    ::

        ni = (Z_imp (ne - z_fast) - Zeff ne) / (Z_imp - 1)

    which reduces to the plain form at ``z_fast = 0``.  The corresponding
    physical bounds on a full-``ne`` Zeff are
    ``ne_th/ne <= Zeff <= Z_imp ne_th/ne`` (both reduce to the familiar
    ``[1, Z_imp]`` without fast ions).
    """
    ne = np.asarray(ne, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    Z_imp = float(Z_imp)
    if not Z_imp > 1.0:
        raise ValueError(f"Z_imp must exceed 1 (got {Z_imp})")
    if z_fast is None:
        return ne * (Z_imp - zeff) / (Z_imp - 1.0)
    ne_th = np.maximum(ne - np.asarray(z_fast, dtype=float), 0.0)
    return (Z_imp * ne_th - zeff * ne) / (Z_imp - 1.0)


# Elementary charge [C] -- thermal pressure p = e * sum_s(n_s * T_s) with n in
# m^-3 and T in eV.
_EC = 1.602176634e-19


def impurity_pressure(ne, ni, ti, Z_imp):
    """Thermal pressure of the (single, effective) impurity species [Pa].

    One-Zeff single-impurity model: the impurity density follows from the SAME
    ``(ne, ni, Z_imp)`` set that derives the main ion, ``nz = (ne - ni)/Z_imp``,
    assumed thermalized at the main-ion ``ti``. This is the carbon (impurity)
    pressure term that single-ion ``e*(ne*Te + ni*Ti)`` omits. Returns zeros if
    ``Z_imp`` is falsy/None (no measured dilution -> no impurity to add), so the
    same call is safe on impurity-free sources.
    """
    if not Z_imp:
        return np.zeros_like(np.asarray(ni, dtype=float))
    nz = np.clip((np.asarray(ne, dtype=float) - np.asarray(ni, dtype=float))
                 / float(Z_imp), 0.0, None)
    return _EC * nz * np.asarray(ti, dtype=float)


def fast_pressure_residual(psi_N, ne, te, ni, ti, Z_imp, psi_N_gfile, p_gfile):
    """Fast-ion pressure inferred as the g-file total minus the thermal pressure.

    The reconstruction path takes thermal kinetics from the IDA ``.cdf`` (no fast
    channel), but the g-file's ``equilibrium.pressure`` is the *total* (thermal +
    impurity + fast) pressure that constrained the original EFIT. The difference
    is the fast (beam) pressure plus any small GS / numerical residual::

        p_fast(psi) = max( p_gfile(psi)
                           - e (ne Te + ni Ti)        # main e + i thermal
                           - p_impurity(ne, ni, Ti)   # carbon thermal
                         , 0 )

    ``p_gfile`` (on its own ``psi_N_gfile`` grid) is linearly interpolated onto
    the kinetic ``psi_N`` grid before differencing, so the result is ready to feed
    to ``FixedComponentsConfig.p_fast`` (with ``psi_N``). It is clipped at zero to
    stay physical. Subtracting :func:`impurity_pressure` here avoids double-counting
    the carbon term, which bouquet re-adds per draw. An explicit TRANSP/ONETWO
    ``p_fast`` should be preferred when available; this residual is the
    no-extra-data fallback for NBI-heated shots.

    Returns ``p_fast`` [Pa] on ``psi_N``.
    """
    psi_N = np.asarray(psi_N, dtype=float)
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te, dtype=float)
    ni = np.asarray(ni, dtype=float)
    ti = np.asarray(ti, dtype=float)

    p_gfile_kin = np.interp(psi_N, np.asarray(psi_N_gfile, dtype=float),
                            np.asarray(p_gfile, dtype=float))
    p_thermal = _EC * (ne * te + ni * ti)
    p_imp = impurity_pressure(ne, ni, ti, Z_imp)
    return np.clip(p_gfile_kin - p_thermal - p_imp, 0.0, None)


def infer_fast_pressure(psi_N, ne, te, ni, ti, Z_imp, psi_N_gfile, p_gfile,
                        axis_psi_N: float = 0.15):
    """Validated fast-ion pressure from the g-file/thermal residual.

    Wraps :func:`fast_pressure_residual` with a physical-validity gate. The
    residual ``p_gfile - p_thermal - p_imp`` is only a meaningful fast pressure
    when the g-file is a *kinetic*-EFIT whose total pressure was constrained to
    ``p_thermal + p_fast``; for a magnetics/standard EFIT the total can fall
    *below* the kinetic thermal pressure near the axis, and the clipped residual
    is then an artifact (zero core, spurious mid-radius bump) rather than a real
    beam profile -- a true NBI fast pressure peaks *at* the axis.

    Gate: a real core-peaked fast profile must be non-negative on axis, so if the
    (unclipped) residual is negative on average over the near-axis region
    (``psi_N <= axis_psi_N``) -- thermal already exceeds the g-file total there --
    the g-file is not usable for fast-pressure extraction, and this returns
    **zeros** (thermal-only, matching the reference behaviour) with a diagnostic
    message. Otherwise it returns the clipped residual.

    Returns ``(p_fast, info)`` where ``info`` is a dict with ``valid`` (bool),
    ``axis_residual_Pa``, ``peak_Pa``, ``peak_psi_N``, and a human-readable
    ``message`` the caller can print.
    """
    psi_N = np.asarray(psi_N, dtype=float)
    ne = np.asarray(ne, dtype=float); te = np.asarray(te, dtype=float)
    ni = np.asarray(ni, dtype=float); ti = np.asarray(ti, dtype=float)

    p_gfile_kin = np.interp(psi_N, np.asarray(psi_N_gfile, dtype=float),
                            np.asarray(p_gfile, dtype=float))
    raw = p_gfile_kin - _EC * (ne * te + ni * ti) - impurity_pressure(ne, ni, ti, Z_imp)

    axis = psi_N <= axis_psi_N
    axis_resid = float(np.mean(raw[axis])) if axis.any() else float(raw[0])
    valid = axis_resid >= 0.0

    if valid:
        p_fast = np.clip(raw, 0.0, None)
        ipk = int(np.argmax(p_fast))
        info = dict(valid=True,
                    axis_residual_Pa=axis_resid,
                    peak_Pa=float(p_fast[ipk]),
                    peak_psi_N=float(psi_N[ipk]),
                    message=(f"[p_fast] kinetic-EFIT residual OK: peak "
                             f"{p_fast[ipk]:.0f} Pa at psi_N={psi_N[ipk]:.2f}"))
    else:
        p_fast = np.zeros_like(psi_N)
        info = dict(valid=False,
                    axis_residual_Pa=axis_resid,
                    peak_Pa=0.0,
                    peak_psi_N=float("nan"),
                    message=(f"[p_fast] g-file total < thermal near the axis "
                             f"(mean near-axis residual {axis_resid:.0f} Pa < 0): "
                             f"not a kinetic-EFIT -> p_fast=0 (thermal-only). Supply "
                             f"a kinetic-EFIT g-file or a TRANSP/ONETWO p_fast to "
                             f"include fast pressure."))
    return p_fast, info


def radial_field_from_impurity_force_balance(
        psi_N, n_imp, t_imp, omega_tor, v_pol,
        Bpol, Rmaj, dpsiN_dR, B_phi, Z_imp=6.0,
        sigma_n_imp=None, sigma_t_imp=None,
        sigma_omega_tor=None, sigma_v_pol=None):
    r"""Radial electric field from the measured impurity radial force balance.

    The steady-state radial force balance of the measured impurity species
    (charge ``e_a = Z_C e``), with inertia and viscosity neglected, is

    .. math::

        E_r = \frac{1}{Z_C e\, n_C}\frac{dp_C}{dR}
              \;-\; v_\theta B_\phi \;+\; v_\phi B_\theta ,

    with :math:`p_C = e\,n_C T_C`, :math:`v_\phi = \omega_{tor} R`, and
    :math:`v_\theta` the measured poloidal velocity. This is the standard CER
    relation (Wesson, *Tokamaks* 4th ed., momentum eq. 2.23.2 and the radial
    force-balance passage; Burrell, *Phys. Plasmas* 4, 1499 (1997)). All three
    terms are of comparable size at low rotation. Any measured impurity species
    works (``Z_imp`` its charge); on DIII-D that is usually carbon from CER.

    Inputs are SI on the common ``psi_N`` grid: ``n_imp`` m^-3, ``t_imp``
    eV, ``omega_tor`` rad/s, ``v_pol`` m/s, ``Bpol``/``B_phi`` T, ``Rmaj`` m,
    ``dpsiN_dR`` (= d psi_N / dR) 1/m. Signs are taken from the inputs as given
    (the caller must supply ``B_phi``/``Bpol``/``v_pol``/``omega_tor`` in a
    consistent convention); inspect the returned component terms to check.

    ``dp_C/dR`` is formed as ``d p_C/d psi_N * dpsiN_dR``. If the measured
    ``sigma_*`` envelopes are supplied, a 1-sigma ``sigma`` on E_r is propagated
    (diamagnetic term via the fractional (n_C, T_C) errors -- approximate through
    the gradient -- plus the exact rotation terms), suitable as a switchboard
    ``aux_sigmas['e_r']``.

    Returns ``(E_r, info)`` with ``info`` holding ``diamagnetic``, ``toroidal``
    (:math:`v_\phi B_\theta`), ``poloidal`` (:math:`-v_\theta B_\phi`), and
    ``sigma`` (zeros if no errors given), all [V/m].
    """
    psi_N = np.asarray(psi_N, dtype=float)
    n_C = np.asarray(n_imp, dtype=float)
    t_C = np.asarray(t_imp, dtype=float)
    Z_C = float(Z_imp)

    p_C = _EC * n_C * t_C                                   # Pa
    dpC_dR = np.gradient(p_C, psi_N) * np.asarray(dpsiN_dR, dtype=float)   # Pa/m
    with np.errstate(divide="ignore", invalid="ignore"):
        diamag = np.where(n_C > 0, dpC_dR / (Z_C * _EC * n_C), 0.0)   # V/m
    v_phi = np.asarray(omega_tor, dtype=float) * np.asarray(Rmaj, dtype=float)
    toroidal = v_phi * np.asarray(Bpol, dtype=float)       # v_phi B_theta
    poloidal = -np.asarray(v_pol, dtype=float) * np.asarray(B_phi, dtype=float)  # -v_theta B_phi
    E_r = diamag + toroidal + poloidal

    sigma = np.zeros_like(E_r)
    if any(s is not None for s in (sigma_n_imp, sigma_t_imp,
                                   sigma_omega_tor, sigma_v_pol)):
        def _frac(sig, val):
            if sig is None:
                return np.zeros_like(val)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(np.abs(val) > 0, np.asarray(sig, float) / np.abs(val), 0.0)
        # diamagnetic: approximate its relative error by the (n_C, T_C) fractional
        # errors added in quadrature (the gradient's relative error ~ profile's).
        rel_dia = np.sqrt(_frac(sigma_n_imp, n_C) ** 2 + _frac(sigma_t_imp, t_C) ** 2)
        s_dia = np.abs(diamag) * rel_dia
        s_tor = (np.abs(np.asarray(Rmaj, float) * np.asarray(Bpol, float))
                 * np.asarray(sigma_omega_tor, float)) if sigma_omega_tor is not None \
            else np.zeros_like(E_r)
        s_pol = (np.abs(np.asarray(B_phi, float)) * np.asarray(sigma_v_pol, float)) \
            if sigma_v_pol is not None else np.zeros_like(E_r)
        sigma = np.sqrt(s_dia ** 2 + s_tor ** 2 + s_pol ** 2)

    info = dict(diamagnetic=diamag, toroidal=toroidal, poloidal=poloidal, sigma=sigma)
    return E_r, info


def radial_field_from_cer(psi_N, n_carbon, t_carbon, omega_tor, v_pol,
                          Bpol, Rmaj, dpsiN_dR, B_phi, Z_C=6.0,
                          sigma_n_carbon=None, sigma_t_carbon=None,
                          sigma_omega_tor=None, sigma_v_pol=None):
    """Deprecated diagnostic-specific alias.

    Kept so existing scripts keep running; the physics is generic impurity force
    balance, so prefer :func:`radial_field_from_impurity_force_balance`, which
    names its inputs for the species rather than for one machine's diagnostic.
    """
    return radial_field_from_impurity_force_balance(
        psi_N, n_carbon, t_carbon, omega_tor, v_pol, Bpol, Rmaj, dpsiN_dR,
        B_phi, Z_imp=Z_C, sigma_n_imp=sigma_n_carbon, sigma_t_imp=sigma_t_carbon,
        sigma_omega_tor=sigma_omega_tor, sigma_v_pol=sigma_v_pol)



# =====================================================================
#  Core-pressure hollowness: a report-only health record
# =====================================================================
#
# The record DESCRIBES the core shape of a pressure profile.  It says whether,
# and by how much, the pressure rises above its value at the innermost grid
# node inside a stated core window.  It does not say why: a hollow core can be
# physical (off-axis heating, an off-axis fast-ion population, impurity
# accumulation) or can come from how the inputs were fitted or composed.
# Nothing in bouquet reads it back: no profile, solve, filter decision,
# in-spec count or until-N count depends on it.
#
# The recorded NUMBERS are the result.  ``is_hollow`` is a convenience boolean
# derived from them by a reporting threshold; that threshold is a choice about
# when to call a rise worth mentioning, not an acceptance criterion.

#: Outer edge of the core window, in psi_N.  The maximum is searched, and the
#: positive-gradient intervals are counted, over nodes with psi_N <= this.
CORE_PSI_N = 0.5

#: Reporting threshold on ``rise_frac`` for the ``is_hollow`` boolean.
#: ``is_hollow`` is True when the core maximum exceeds the innermost-node
#: value by MORE than 1 % of that value.  Chosen an order of magnitude above
#: 1e-3-level grid-scale noise and interpolation wiggles, so a single noisy
#: node does not set it.  It gates nothing -- it is a reporting choice, not an
#: acceptance criterion -- and the numbers it is derived from are always
#: recorded next to it, so a reader can apply any other threshold.
CORE_HOLLOW_RISE_FRAC = 0.01

#: The innermost grid node stands in for the magnetic axis.  If it lies
#: further out than this in psi_N, the record is "not evaluated": a rise that
#: happens inside the first node would be invisible, so a "not hollow" answer
#: could be false.
CORE_HOLLOW_MAX_REF_PSI_N = 0.05

#: Minimum number of grid nodes inside the core window for the record to be
#: evaluated.
CORE_HOLLOW_MIN_CORE_NODES = 3

#: The components summed into the THERMAL-only profile (all thermal species:
#: electrons, main ions, the impurity).  Fast-ion pressure and any anchor
#: offset are excluded.  Note this differs from the archived
#: ``pressure_thermal`` dataset, which holds electrons + main ions only.
THERMAL_PRESSURE_COMPONENTS = ("electron_thermal", "ion_thermal", "impurity")


class CorePressureHollowWarning(UserWarning):
    """Report-only notice from :func:`core_pressure_hollow_record`.

    Emitted when a total pressure profile rises above its innermost-node value
    by more than the reporting threshold inside the core window.  It changes
    nothing: the profile is used exactly as it was built, and nothing is
    filtered or rejected.
    """


def _hollow_not_evaluated(reason, **extra):
    rec = {
        "evaluated": False,
        "reason": str(reason),
        "is_hollow": None,
        "rise_frac": None,
        "p_ref": None,
        "psi_N_ref": None,
        "p_core_max": None,
        "psi_N_of_max": None,
        "rise_extent": None,
        "positive_gradient_extent": None,
        "positive_gradient_psi_N_max": None,
        "n_core_nodes": None,
        "component_shares": None,
    }
    rec.update(extra)
    return rec


def core_pressure_health(psi_N, pressure, components=None,
                         psi_N_core=CORE_PSI_N,
                         rise_threshold=CORE_HOLLOW_RISE_FRAC,
                         max_ref_psi_N=CORE_HOLLOW_MAX_REF_PSI_N):
    r"""Measure how far a pressure profile rises above its axis value in the core.

    Descriptive only.  The function never modifies its inputs, never raises on
    bad data (it returns a "not evaluated" record with a reason instead) and
    its result is read by nothing else in bouquet.

    Definition
    ----------
    The innermost grid node stands in for the axis: ``p_ref = p[i0]`` at
    ``psi_N_ref = psi_N[i0]`` (``i0`` is the node with the smallest psi_N; a
    descending grid is read in reverse, which reorders nothing in the caller's
    arrays).  Over the core window ``psi_N <= psi_N_core``:

    * ``rise_frac = (p_core_max - p_ref) / p_ref`` -- how far the core
      maximum exceeds the axis value, as a fraction of the axis value.
      Zero when the axis node is the core maximum.
    * ``psi_N_of_max`` -- where that maximum sits; ``rise_extent =
      psi_N_of_max - psi_N_ref`` is the radial distance over which the
      pressure climbs to it.
    * ``positive_gradient_extent`` -- the summed psi_N width of the core
      intervals (both nodes inside the window) whose secant slope
      ``Delta p / Delta psi_N`` is positive, and
      ``positive_gradient_psi_N_max`` the outer edge of the outermost one.
      This counts every positive secant, including ones produced by
      grid-scale noise; ``rise_frac`` and ``rise_extent`` are the
      noise-robust pair.
    * ``component_shares`` (when *components* is given) -- for each named
      additive component, ``sum_W Delta p_c / sum_W Delta p``, where ``W``
      is the set of core intervals on which the TOTAL secant is positive.
      The shares add to 1 when the components add to *pressure*.  They
      describe how the rise is made up; they do not attribute a cause.

    ``is_hollow = rise_frac > rise_threshold``.  This boolean is a reporting
    convenience: the threshold (default :data:`CORE_HOLLOW_RISE_FRAC`, 1 %)
    is a choice about what to call a rise, NOT an acceptance criterion, and
    nothing is gated, filtered or modified on it.

    Not evaluated (``evaluated`` False, ``is_hollow`` None, ``reason`` set)
    when: the arrays are not matching 1-D arrays of at least 3 points; any
    grid or pressure value is non-finite; the grid is not strictly monotone;
    fewer than :data:`CORE_HOLLOW_MIN_CORE_NODES` nodes lie in the core
    window; the innermost node lies beyond ``max_ref_psi_N``; or ``p_ref`` is
    not positive.  A record that could not be evaluated never reads as
    "not hollow".

    Parameters
    ----------
    psi_N : array_like
        Normalized poloidal flux of the pressure samples.
    pressure : array_like
        Pressure [Pa] on *psi_N*.
    components : dict, optional
        Named additive contributions to *pressure* on the same grid.
    psi_N_core : float, optional
        Outer edge of the core window (default :data:`CORE_PSI_N`).
    rise_threshold : float, optional
        Reporting threshold for ``is_hollow`` (default 0.01).
    max_ref_psi_N : float, optional
        Furthest psi_N the innermost node may sit at and still stand for the
        axis (default :data:`CORE_HOLLOW_MAX_REF_PSI_N`).

    Returns
    -------
    dict
        JSON-serializable record (plain Python floats, ints, bools, None).
    """
    try:
        x = np.array(psi_N, dtype=float, copy=True)
        p = np.array(pressure, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        return _hollow_not_evaluated(f"inputs are not numeric arrays ({exc})")
    if x.ndim != 1 or p.shape != x.shape or x.size < 3:
        return _hollow_not_evaluated(
            f"need matching 1-D arrays of at least 3 points (got psi_N "
            f"{x.shape}, pressure {p.shape})")
    n_bad = int(np.count_nonzero(~np.isfinite(x)) + np.count_nonzero(~np.isfinite(p)))
    if n_bad:
        return _hollow_not_evaluated(
            f"{n_bad} non-finite value(s) in psi_N or pressure")
    dx = np.diff(x)
    if np.all(dx < 0.0):
        order = slice(None, None, -1)
    elif np.all(dx > 0.0):
        order = slice(None)
    else:
        return _hollow_not_evaluated("psi_N is not strictly monotone")
    x = x[order]
    p = p[order]

    psi_N_ref = float(x[0])
    if psi_N_ref > float(max_ref_psi_N):
        return _hollow_not_evaluated(
            f"innermost node at psi_N={psi_N_ref:.4g} is beyond "
            f"{float(max_ref_psi_N):g}, so it cannot stand for the axis value",
            psi_N_ref=psi_N_ref)
    core = x <= float(psi_N_core)
    n_core = int(np.count_nonzero(core))
    if n_core < CORE_HOLLOW_MIN_CORE_NODES:
        return _hollow_not_evaluated(
            f"only {n_core} node(s) inside psi_N <= {float(psi_N_core):g} "
            f"(need {CORE_HOLLOW_MIN_CORE_NODES})",
            psi_N_ref=psi_N_ref, n_core_nodes=n_core)
    p_ref = float(p[0])
    if not p_ref > 0.0:
        return _hollow_not_evaluated(
            f"pressure at the innermost node is {p_ref:.6g}, not positive; "
            f"a relative rise is undefined",
            psi_N_ref=psi_N_ref, p_ref=p_ref, n_core_nodes=n_core)

    k_max = int(np.argmax(p[core]))          # first occurrence on ties
    p_core_max = float(p[core][k_max])
    psi_N_of_max = float(x[core][k_max])
    rise_frac = (p_core_max - p_ref) / p_ref

    dp = np.diff(p)
    in_core = core[:-1] & core[1:]
    positive = in_core & (dp > 0.0)          # dx > 0 after reordering
    pos_extent = float(np.sum(np.diff(x)[positive])) if np.any(positive) else 0.0
    pos_edge = float(np.max(x[1:][positive])) if np.any(positive) else None

    shares = None
    if components:
        total_rise = float(np.sum(dp[positive])) if np.any(positive) else 0.0
        shares = {}
        for name, arr in components.items():
            try:
                c = np.array(arr, dtype=float, copy=True)
            except (TypeError, ValueError):
                shares[str(name)] = None
                continue
            if c.shape != p.shape or not np.all(np.isfinite(c)) or total_rise <= 0.0:
                shares[str(name)] = None
                continue
            c = c[order]
            shares[str(name)] = float(np.sum(np.diff(c)[positive]) / total_rise)

    return {
        "evaluated": True,
        "reason": None,
        "is_hollow": bool(rise_frac > float(rise_threshold)),
        "rise_frac": float(rise_frac),
        "p_ref": p_ref,
        "psi_N_ref": psi_N_ref,
        "p_core_max": p_core_max,
        "psi_N_of_max": psi_N_of_max,
        "rise_extent": float(psi_N_of_max - psi_N_ref),
        "positive_gradient_extent": pos_extent,
        "positive_gradient_psi_N_max": pos_edge,
        "n_core_nodes": n_core,
        "component_shares": shares,
    }


def core_pressure_hollow_definition(psi_N_core=CORE_PSI_N,
                                    rise_threshold=CORE_HOLLOW_RISE_FRAC,
                                    max_ref_psi_N=CORE_HOLLOW_MAX_REF_PSI_N):
    """The definition block stored with every ``core_pressure_hollow`` record."""
    return {
        "measure": ("rise_frac = (max p over psi_N <= psi_N_core - p_ref) / "
                    "p_ref, p_ref = pressure at the innermost grid node"),
        "psi_N_core": float(psi_N_core),
        "rise_threshold": float(rise_threshold),
        "max_ref_psi_N": float(max_ref_psi_N),
        "min_core_nodes": int(CORE_HOLLOW_MIN_CORE_NODES),
        "thermal_components": list(THERMAL_PRESSURE_COMPONENTS),
        "kind": ("report-only: describes the observed profile and gates "
                 "nothing; rise_threshold only sets the is_hollow "
                 "convenience boolean"),
    }


def core_pressure_hollow_record(psi_N, input_total, input_components=None,
                                achieved_total=None, achieved_psi_N=None,
                                achieved_reason=None, warn=True, **kw):
    """Assemble the per-slice ``core_pressure_hollow`` health record.

    Evaluates :func:`core_pressure_health` on

    * ``input.total`` -- the pressure handed to the solver;
    * ``input.thermal`` -- the sum of the :data:`THERMAL_PRESSURE_COMPONENTS`
      present in *input_components* (electrons + main ions + impurity; fast
      ions and any anchor offset excluded), when a component breakdown with
      at least the electron and main-ion terms is supplied;
    * ``achieved.total`` -- the pressure the converged equilibrium carries
      (*achieved_total* on *achieved_psi_N*, default *psi_N*);
    * ``achieved.thermal`` -- always "not evaluated": the converged
      equilibrium carries one total pressure, so its thermal part cannot be
      separated.

    Report-only: the inputs are not modified and nothing downstream reads the
    record.  With *warn* True a :class:`CorePressureHollowWarning` is emitted
    once if an evaluated TOTAL profile has ``is_hollow`` True.  Extra keyword
    arguments go to :func:`core_pressure_health`.
    """
    import warnings

    rec = {"definition": core_pressure_hollow_definition(
        **{k: v for k, v in kw.items()
           if k in ("psi_N_core", "rise_threshold", "max_ref_psi_N")})}

    comps = dict(input_components or {})
    total = core_pressure_health(psi_N, input_total, components=comps or None, **kw)
    total["composition"] = sorted(comps) if comps else None
    th_keys = [k for k in THERMAL_PRESSURE_COMPONENTS if k in comps]
    if "electron_thermal" in th_keys and "ion_thermal" in th_keys:
        try:
            th = sum(np.asarray(comps[k], dtype=float) for k in th_keys)
            thermal = core_pressure_health(
                psi_N, th, components={k: comps[k] for k in th_keys}, **kw)
        except Exception as exc:              # never let a report break a run
            thermal = _hollow_not_evaluated(f"thermal sum failed ({exc})")
        thermal["composition"] = th_keys
    else:
        thermal = _hollow_not_evaluated(
            "no electron + main-ion component breakdown supplied")
    rec["input"] = {"total": total, "thermal": thermal}

    if achieved_total is not None:
        ach = core_pressure_health(
            psi_N if achieved_psi_N is None else achieved_psi_N,
            achieved_total, **kw)
    else:
        ach = _hollow_not_evaluated(
            achieved_reason or "no converged equilibrium pressure supplied")
    rec["achieved"] = {
        "total": ach,
        "thermal": _hollow_not_evaluated(
            "the converged equilibrium carries one total pressure; its "
            "thermal part is not separable"),
    }

    msgs = []
    for where, r in (("input", total), ("achieved", ach)):
        if r.get("is_hollow"):
            msgs.append(
                f"{where} total pressure rises {100.0 * r['rise_frac']:.2f}% "
                f"above its innermost-node value, to a maximum at psi_N="
                f"{r['psi_N_of_max']:.3g}")
    rec["warned"] = bool(warn and msgs)
    if rec["warned"]:
        warnings.warn(
            "core_pressure_hollow: " + "; ".join(msgs)
            + f" (reporting threshold {100.0 * rec['definition']['rise_threshold']:g}%"
            " inside psi_N <= "
            f"{rec['definition']['psi_N_core']:g}). Report-only: nothing was "
            "modified, filtered or rejected.",
            CorePressureHollowWarning, stacklevel=2)
    return rec
