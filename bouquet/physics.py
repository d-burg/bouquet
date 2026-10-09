"""Physics reductions shared by the baseline resolvers.

  * :func:`isotropize_fast_pressure` -- collapse an anisotropic (gyrotropic)
    fast-ion pressure to a single scalar, since TokaMaker solves a scalar-pressure
    Grad-Shafranov equation.

  * :func:`parallel_to_toroidal` -- convert a flux-surface-averaged *parallel*
    current density <j.B>/B0 (the IMAS / neoclassical convention for j_ohmic,
    j_bootstrap, and the driven currents) to the *toroidal* current density
    bouquet stores and the solver consumes: the plain flux-surface average
    <j_phi> (TokaMaker's ``jphi-linterp``).  The field-aligned conversion is
    ONE factor in the whole package, :func:`field_aligned_conversion`
    (``kappa = F<1/R>/<B^2>``), shared with the unified engine.
  * :func:`jtor_imas_to_jphi_tokamaker` / :func:`jphi_tokamaker_to_jtor_imas`
    -- IMAS ``j_tor`` (``<j_phi/R>/<1/R>``) <-> ``<j_phi>`` exactly (A5 of
    ``docs/current-conventions.md``); :func:`jphi_tokamaker_pressure_term` is
    the pressure-driven ``p'(<R> - F^2<1/R>/<B^2>)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import coords as _coords


#: Elementary charge [C] (CODATA 2018, exact): THE eV -> J factor of every
#: thermal pressure ``p = e * sum_s(n_s * T_s)`` (n in m^-3, T in eV) and of
#: the Redl drive -- the same constant OFT's ``solve_with_bootstrap`` uses.
#: Every site in bouquet reads it from here.
ELEMENTARY_CHARGE = 1.602176634e-19

#: FROZEN LEGACY value (1.6022e-19, 1.46e-5 relative above the exact one)
#: that the legacy (``jbs_self_consistent=False``) draw pressure and g-file
#: reconstruction pressure have always used.  It is kept ONLY so the frozen
#: legacy path stays bit for bit; nothing else may use it.  Select with
#: :func:`thermal_pressure_charge`.
ELEMENTARY_CHARGE_LEGACY = 1.6022e-19


#: The normalised poloidal flux at which the solver's reported ``q0`` lives:
#: TokaMaker's ``get_stats()['q_0']`` is ``q`` at its ``axis_pad`` default,
#: psi_N = 0.02 (the first traced surface), NOT the magnetic axis.  A g-file's
#: ``qpsi[0]`` (and an IDS ``q[0]``) is the axis value, psi_N = 0.  Compare
#: like with like: evaluate the input at this radius, or say which radius a
#: number is at (``q0_psi_N`` in the records).
SOLVER_Q0_PSI_N = 0.02


def thermal_pressure_charge(jbs_loop=None):
    """The eV -> J factor of the thermal pressure for a run.

    :data:`ELEMENTARY_CHARGE` whenever the self-consistent bootstrap loop is
    on (``jbs_loop`` a settings dict with ``enabled``, or ``True``) -- the
    value the modelling-source forward solve and the impurity / Redl terms
    use, so the reconstruction and its draws share one pressure --
    and the frozen :data:`ELEMENTARY_CHARGE_LEGACY` on the legacy path.
    """
    if isinstance(jbs_loop, dict):
        on = bool(jbs_loop.get("enabled"))
    else:
        on = bool(jbs_loop)
    return ELEMENTARY_CHARGE if on else ELEMENTARY_CHARGE_LEGACY


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


#: ``geom`` keys of the current-convention helpers (per-surface arrays):
#: ``F`` = R B_phi [T m], ``avg_R`` = <R> [m], ``avg_inv_R`` = <1/R> [1/m],
#: ``avg_inv_R2`` = <1/R^2> [1/m^2], ``avg_B2`` = <B^2> [T^2], ``pprime`` =
#: dp/dpsi in the sign/normalisation of the J being converted (TokaMaker psi
#: per radian; on IMAS COCOS-11 data use -2*pi*dpressure_dpsi), and optional
#: scalar ``B0``: when present the parallel side is the IMAS <J.B>/B0
#: (multiplied by B0 on input, divided on output); absent = raw <J.B>.
CURRENT_GEOM_KEYS = ("F", "avg_R", "avg_inv_R", "avg_inv_R2", "avg_B2", "pprime")


def _geom(geom, *keys):
    """Float arrays for ``keys`` from ``geom``; ValueError naming any missing."""
    missing = [k for k in keys if geom is None or geom.get(k) is None]
    if missing:
        raise ValueError(f"geom is missing {missing} (current-convention "
                         f"helpers take {list(CURRENT_GEOM_KEYS)} + optional 'B0')")
    return [np.asarray(geom[k], dtype=float) for k in keys]


def field_aligned_conversion(F, avg_inv_R, avg_B2):
    """``kappa = F <1/R> / <B^2>``: THE ``<j.B>`` -> ``<j_phi>`` factor.

    For a field-aligned current component ``j = lambda(psi) B`` (``lambda =
    <j.B>/<B^2>``; the neoclassical banana-plateau bootstrap and the driven
    currents -- the Pfirsch-Schlueter return current has ``<j_PS.B> = 0`` and
    is not part of the component) with ``B_phi = F/R``::

        <j_phi> = lambda F <1/R> = <j.B> F <1/R> / <B^2> = kappa <j.B>

    ``<j_phi>`` is the plain flux-surface average the solver consumes
    (OpenFUSIONToolkit's ``jphi-linterp`` ``jphi_update``), and with the
    pressure-driven part the composition identity of the unified engine

        <j_phi> = kappa <j.B>  +  p' (<R> - F^2 <1/R> / <B^2>)

    is exact.  Every conversion in bouquet -- :func:`parallel_to_toroidal`,
    :func:`toroidal_to_parallel` (its exact inverse), :func:`evaluate_jBS`
    and :func:`bouquet.engine.conversion_factor`, and the SWB output after
    :func:`_swb_jbs_to_toroidal` -- is this one factor
    (2026-10-06; see :func:`parallel_to_toroidal` for what moved).
    """
    return (np.asarray(F, dtype=float) * np.asarray(avg_inv_R, dtype=float)
            / np.asarray(avg_B2, dtype=float))


_INV_R2_REFUSED = (
    "geom carries 'avg_inv_R2': bouquet's <j.B> <-> <j_phi> conversion is "
    "the one field-aligned factor kappa = F<1/R>/<B^2> "
    "(physics.field_aligned_conversion) since 2026-10-06; the former "
    "<1/R^2> branch (the IMAS <j_phi/R>/<1/R> convention, "
    "<j.B>F<1/R^2>/(<B^2><1/R>)) is gone.  Drop 'avg_inv_R2' from geom "
    "(or set it to None).")


def _analytic_geom(geom):
    try:
        F = np.asarray(geom["F"], dtype=float)
        avg_inv_R = np.asarray(geom["avg_inv_R"], dtype=float)
        avg_B2 = np.asarray(geom["avg_B2"], dtype=float)
    except KeyError as missing:
        raise ValueError(
            f"geom is missing required key {missing} "
            "(need 'F', 'avg_inv_R', 'avg_B2'; optional 'B0')"
        ) from None
    if geom.get("avg_inv_R2") is not None:
        raise ValueError(_INV_R2_REFUSED)
    return F, avg_inv_R, avg_B2


def parallel_to_toroidal(
    j_parallel,
    *,
    j_parallel_total=None,
    j_tor_total=None,
    geom: Optional[dict] = None,
):
    """Convert FSA parallel current density ``<j.B>/B0`` to toroidal ``<j_phi>``.

    bouquet stores all current components (inductive/ohmic, bootstrap, NBI, RF)
    as the toroidal current the solver consumes, the plain flux-surface
    average ``<j_phi>``.  IMAS/neoclassical outputs are parallel; this applies
    the per-flux-surface geometric conversion.


    * **ratio** (preferred, used for FUSE input) -- when the source provides both
      the total parallel current and the total toroidal current (FUSE
      ``core_profiles`` carries ``j_total`` *and* ``j_tor``), form the per-surface
      factor ``c(psi) = j_tor_total / j_parallel_total`` and apply it to the
      component. Self-consistent with the source equilibrium.

    * **analytic** -- the field-aligned factor of
      :func:`field_aligned_conversion`, from equilibrium FSA metrics in
      ``geom`` (the reconstruction path, bouquet's own Redl bootstrap)::

          <j_phi> = kappa <j.B>,   kappa = F <1/R> / <B^2>

      ``geom`` keys: ``F`` (``R B_phi`` [T m]), ``avg_inv_R`` (``<1/R>``
      [1/m]), ``avg_B2`` (``<B^2>`` [T^2]) and, optionally, ``B0`` (the
      normalisation of the input: pass the IMAS ``vacuum_toroidal_field`` B0
      when ``j_parallel`` is ``<j.B>/B0``; default 1 = raw ``<j.B>``
      [T A/m^2]).  A ``geom`` carrying ``avg_inv_R2`` is REFUSED (the former
      ``<1/R^2>`` branch is gone; see below).

    **Declared default physics change (2026-10-06, owner-approved).**  Until
    then the package held three conventions: the legacy bootstrap sites
    (:func:`evaluate_jBS`, ``TokaMaker_interface._swb_jbs_to_toroidal``)
    called this function without ``<1/R^2>`` -> ``<j.B>/(F<1/R>)``; the IDS
    exporter passed ``<1/R^2>`` -> ``<j.B>F<1/R^2>/(<B^2><1/R>)`` (the IMAS
    ``<j_phi/R>/<1/R>`` convention); the unified engine used kappa.  The
    legacy factor exceeds kappa by

        <B^2>/(F^2 <1/R>^2) = [<B^2>/<B_phi^2>] x [<1/R^2>/<1/R>^2]

    (``<B_phi^2> = F^2 <1/R^2>``): the bracket (the poloidal-field content,
    ~1.4 %) times the Jensen ratio (~5.2 %) -- +6.8 % at psi_N ~ 0.97 on the
    synthetic D3D-like example (+1.0 % at 0.1, +4.3 % at 0.5, +6.4 % at
    0.9; measured 2026-10-06).  The legacy bootstrap at the pedestal therefore drops by that
    fraction; the engine path is unchanged (it already used kappa).

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
        F, avg_inv_R, avg_B2 = _analytic_geom(geom)
        j_dot_B = j_parallel * float(geom.get("B0", 1.0))
        return j_dot_B * field_aligned_conversion(F, avg_inv_R, avg_B2)

    raise ValueError(
        "provide either (j_parallel_total, j_tor_total) for the ratio method "
        "or geom for the analytic method"
    )


def toroidal_to_parallel(j_tor, *, geom: dict):
    """The exact inverse of :func:`parallel_to_toroidal` (analytic method).

    Convert bouquet's toroidal current density ``<j_phi>`` back to the IMAS
    flux-surface-averaged parallel current ``<j.B>/B0`` (the write-back
    direction: IMAS ``core_profiles.j_ohmic / j_bootstrap / j_total`` are
    parallel)::

        <j.B> = <j_phi> / kappa,   kappa = F <1/R> / <B^2>

    (:func:`field_aligned_conversion`), so a component converted in by
    :func:`parallel_to_toroidal` round-trips to machine precision on the same
    ``geom``.  Uses the draw's OWN flux-surface geometry when given it.

    ``geom`` keys: ``F``, ``avg_inv_R``, ``avg_B2`` and optionally ``B0``
    (output normalisation: pass the IMAS ``vacuum_toroidal_field`` B0 to
    return ``<j.B>/B0``; default 1 = raw ``<j.B>`` [T A/m^2]).  A ``geom``
    carrying ``avg_inv_R2`` is REFUSED: the former ``<1/R^2>`` branch (the
    inverse of the IMAS ``<j_phi/R>/<1/R>`` convention) is gone since
    2026-10-06 -- see :func:`parallel_to_toroidal`.
    """
    j_tor = np.asarray(j_tor, dtype=float)
    F, avg_inv_R, avg_B2 = _analytic_geom(geom)
    j_dot_B = j_tor / field_aligned_conversion(F, avg_inv_R, avg_B2)
    return j_dot_B / float(geom.get("B0", 1.0))           # IMAS <j.B>/B0


def jpar_to_jphi_tokamaker(j_dot_B, geom):
    """:func:`parallel_to_toroidal` on a current-convention ``geom``, which
    may carry ``avg_inv_R2`` (for A5)."""
    return parallel_to_toroidal(j_dot_B, geom={**geom, "avg_inv_R2": None})


def jphi_tokamaker_pressure_term(geom):
    """Pressure-driven ``<j_phi>``: p'(<R> - F^2<1/R>/<B^2>) (A7)."""
    F, R, inv_R, B2, pp = _geom(geom, "F", "avg_R", "avg_inv_R", "avg_B2",
                                "pprime")
    return pp * (R - F * F * inv_R / B2)


def jphi_tokamaker_to_jpar(jphi, geom):
    """:func:`toroidal_to_parallel` on a current-convention ``geom`` (the
    field-aligned part only; subtract :func:`jphi_tokamaker_pressure_term`
    from a total first)."""
    return toroidal_to_parallel(jphi, geom={**geom, "avg_inv_R2": None})


def jtor_imas_to_jphi_tokamaker(j_tor, geom):
    """IMAS ``j_tor`` -> TokaMaker ``jphi`` (A5):
    J_TM = <R>p' + <1/R>(J_IMAS<1/R> - p')/<1/R^2>."""
    R, inv_R, inv_R2, pp = _geom(geom, "avg_R", "avg_inv_R", "avg_inv_R2",
                                 "pprime")
    J = np.asarray(j_tor, dtype=float)
    return R * pp + inv_R * (J * inv_R - pp) / inv_R2


def jphi_tokamaker_to_jtor_imas(jphi, geom):
    """TokaMaker ``jphi`` -> IMAS ``j_tor`` (A5, inverse):
    J_IMAS = [p' + <1/R^2>(J_TM - <R>p')/<1/R>]/<1/R>."""
    R, inv_R, inv_R2, pp = _geom(geom, "avg_R", "avg_inv_R", "avg_inv_R2",
                                 "pprime")
    J = np.asarray(jphi, dtype=float)
    return (pp + inv_R2 * (J - R * pp) / inv_R) / inv_R


# ---------------------------------------------------------------------------
#  OpenFUSIONToolkit's solve_with_bootstrap (SWB) output convention
# ---------------------------------------------------------------------------
#: SWB's ``j_BS`` is the Redl ``<j.B>`` projected as ``<j.B> R_avg/F`` (the
#: upstream Python ``solve_with_bootstrap``: ``j_BS_final = j_BS_neo *
#: (R_avg / f)``).  :func:`_swb_jbs_to_toroidal` converts it with kappa.
SWB_JBS_RAVG_OVER_F = "R_avg/F"
#: SWB's ``j_BS`` is already TokaMaker jphi by (A7), ``kappa <j.B> + p'G``:
#: the toolkit generation bouquet's toroidal-flux support needs
#: (``solve_with_bootstrap(x, coord)``; its PR names OFT 1eda3aa as the
#: change).  :func:`_swb_jbs_to_toroidal` takes ``p'G`` off it.
SWB_JBS_TOROIDAL = "kappa<j.B>+p'G"
SWB_JBS_CONVENTIONS = (SWB_JBS_RAVG_OVER_F, SWB_JBS_TOROIDAL)
#: The upstream projection line that identifies :data:`SWB_JBS_RAVG_OVER_F`
#: in the installed toolkit's Python ``solve_with_bootstrap``.
_SWB_RAVG_OVER_F_MARKER = "j_BS_neo * (R_avg / f)"
_SWB_CONVENTION_CACHE = {}


class SwbConventionUnknown(RuntimeError):
    """The installed toolkit's ``solve_with_bootstrap`` returns its bootstrap
    in a convention bouquet cannot identify: no SWB-derived current is used,
    rather than one converted on a guess."""


def swb_jbs_convention(solve_with_bootstrap=None, params=None) -> str:
    """The convention of ``solve_with_bootstrap``'s ``j_BS`` / ``isolated_j_BS``
    on the INSTALLED toolkit (one of :data:`SWB_JBS_CONVENTIONS`); raises
    :class:`SwbConventionUnknown` otherwise.

    A capability check, from the function's arguments (``params``; default:
    :func:`bouquet.coords._swb_params`, the one probe of the toolkit's SWB
    signature, or the signature of the ``solve_with_bootstrap`` given):

    * :data:`SWB_JBS_TOROIDAL` when it takes the grid ``x`` -- the toolkit
      generation whose ``solve_with_bootstrap`` returns TokaMaker jphi
      (``docs/current-conventions.md``, "Where these are used"); upstream
      OFT's grid argument is ``psi_N``;
    * otherwise :data:`SWB_JBS_RAVG_OVER_F` when the Python implementation
      the call runs carries the upstream projection ``j_BS_neo * (R_avg / f)``
      and has no ``use_python_solve`` switch routing the call elsewhere;
    * anything else (a Fortran-routed solve, a ``<j.B>/<|B|>`` projection, an
      unreadable source) is REFUSED: never a silent ``R_avg/F`` path.

    ``solve_with_bootstrap`` defaults to the installed toolkit's.
    """
    import inspect
    if params is None:
        if solve_with_bootstrap is None:
            from . import coords as _c
            params = _c._swb_params()
        else:
            try:
                params = frozenset(
                    inspect.signature(solve_with_bootstrap).parameters)
            except (TypeError, ValueError):
                params = frozenset()
    if "x" in params:
        return SWB_JBS_TOROIDAL
    if "use_python_solve" not in params:
        if solve_with_bootstrap is None:
            try:
                from OpenFUSIONToolkit.TokaMaker.bootstrap import (
                    solve_with_bootstrap)
            except Exception:
                solve_with_bootstrap = None
        hit = _SWB_CONVENTION_CACHE.get(id(solve_with_bootstrap))
        if hit is not None and hit[0] is solve_with_bootstrap:
            src = hit[1]
        else:
            try:
                src = inspect.getsource(solve_with_bootstrap)
            except (OSError, TypeError):
                src = ""
            _SWB_CONVENTION_CACHE[id(solve_with_bootstrap)] = (
                solve_with_bootstrap, src)
        if _SWB_RAVG_OVER_F_MARKER in src:
            return SWB_JBS_RAVG_OVER_F
    raise SwbConventionUnknown(
        "the installed OpenFUSIONToolkit's solve_with_bootstrap returns its "
        "bootstrap in a convention bouquet does not recognise (neither the "
        "upstream <j.B> R_avg/F projection nor the TokaMaker-jphi output of "
        "the toolkit with solve_with_bootstrap(x=...)); bouquet will not "
        "convert it on a guess.  Install one of those toolkits, or use a path "
        "that does not run solve_with_bootstrap.")


def _swb_surfaces(n, psi_pad, psi):
    """``(kw, inv)``: the keywords addressing SWB's own geometry surfaces and
    the map back to the ``n`` profile nodes.  ``psi`` None: OFT's uniform
    ``npsi``/``psi_pad`` grid (the call before grids were passed, bit for
    bit); otherwise the clipped nodes ``clip(psi, psi_pad, 1 - psi_pad)`` SWB
    evaluates its geometry on, each distinct surface once."""
    if psi is None:
        return dict(npsi=int(n), psi_pad=psi_pad), None
    psi = np.asarray(psi, dtype=float)
    if psi.shape != (n,):
        raise ValueError(f"psi has shape {psi.shape}, expected ({n},) to "
                         "match the SWB profile")
    u, inv = np.unique(np.clip(psi, psi_pad, 1.0 - psi_pad),
                       return_inverse=True)
    return dict(psi=np.ascontiguousarray(u, dtype=float)), inv


def _swb_geometry(mygs, n, psi_pad, psi):
    """``F``, ``F'``, ``p'``, ``<R>``, ``<1/R>`` (``get_q``, as SWB reads
    them) and ``<B^2>`` (``sauter_fc``) on SWB's surfaces."""
    kw, inv = _swb_surfaces(n, psi_pad, psi)
    _, F, Fp, _, pp = mygs.get_profiles(**kw)
    _, _, ravgs, _, _, _ = mygs.get_q(**kw)
    _sfc = getattr(mygs, "sauter_fc", None) or getattr(mygs, "calc_sauter_fc")
    B2 = _sauter_avg(_sfc(**kw)[-1], "<|B|^2>", _SAUTER_MODB_INDEX)
    out = dict(F=F, Fp=Fp, pp=pp, R=q_ravg(ravgs, "<R>"),
               inv_R=q_ravg(ravgs, "<1/R>"), B2=B2)
    out = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    if inv is not None:
        out = {k: v[inv] for k, v in out.items()}
    return out


def swb_pressure_term(mygs, n, psi_pad=1e-3, psi=None):
    """The pressure-driven ``<j_phi>`` part ``p'(<R> - F^2<1/R>/<B^2>)`` (A7)
    of the equilibrium *mygs* holds, on SWB's surfaces (``n`` nodes; ``psi``
    as for :func:`_swb_jbs_to_toroidal`): the third current bucket
    ``j_pressure`` of an SWB-derived split.

    ``p'`` is signed so the equilibrium's own TokaMaker jphi
    ``<R>p' + <1/R>FF'/mu0`` is positive, the rule of
    :func:`capture_equilibrium_fsa` (bouquet's positive-current frame)."""
    g = _swb_geometry(mygs, n, psi_pad, psi)
    jphi_eq = g["R"] * g["pp"] + g["inv_R"] * g["F"] * g["Fp"] / (4.0e-7
                                                                  * np.pi)
    sign = 1.0 if float(np.sum(jphi_eq)) >= 0.0 else -1.0
    return sign * g["pp"] * (g["R"] - g["F"] ** 2 * g["inv_R"] / g["B2"])


def _swb_jbs_to_toroidal(mygs, j_bs_swb, psi_pad, psi=None, convention=None):
    """``solve_with_bootstrap``'s ``j_BS`` (or ``isolated_j_BS``) as the
    field-aligned toroidal bootstrap ``kappa <j.B>`` bouquet stores
    (:func:`field_aligned_conversion`), whatever the installed toolkit
    returns (:func:`swb_jbs_convention`):

    * :data:`SWB_JBS_RAVG_OVER_F` -- SWB projected the Redl ``<j.B>`` by
      ``R_avg/F`` (its own ``# to-do: project j_BS_parallel to j_phi more
      accurately?``).  Undone and converted with kappa: net factor
      ``F^2 <1/R> / (<R> <B^2>)``, with ``<R>`` and ``<1/R>`` from ``get_q``
      -- the SAME quantities SWB used, so the undo is exact -- and ``<B^2>``
      from ``sauter_fc``.  Left unconverted it is +7 % at psi_N 0.5 and
      +12-13 % at the pedestal on the synthetic D3D-like example.
    * :data:`SWB_JBS_TOROIDAL` -- SWB returned ``kappa <j.B> + p'G`` (A7);
      ``p'G`` (:func:`swb_pressure_term`) comes off: the pressure-driven
      current is the third bucket ``j_pressure``, never part of ``j_BS``.

    Evaluated on the equilibrium the SWB call just left in ``mygs``: call it
    IMMEDIATELY after ``solve_with_bootstrap``, before any further solve.
    ``psi``: the grid passed to SWB (``x`` / ``psi_N``), None for a call that
    passed none (OFT's uniform grid).  ``convention`` defaults to the
    installed toolkit's.  :func:`swb_conversion_record` is the stamp.
    """
    j = np.asarray(j_bs_swb, dtype=float)
    conv = swb_jbs_convention() if convention is None else convention
    if conv == SWB_JBS_TOROIDAL:
        return j - swb_pressure_term(mygs, j.size, psi_pad, psi)
    if conv != SWB_JBS_RAVG_OVER_F:
        raise ValueError(f"convention must be one of {SWB_JBS_CONVENTIONS}, "
                         f"got {conv!r}")
    g = _swb_geometry(mygs, j.size, psi_pad, psi)
    j_dot_B = j * g["F"] / g["R"]           # undo SWB's R_avg/F projection
    return parallel_to_toroidal(
        j_dot_B, geom={"F": g["F"], "avg_inv_R": g["inv_R"],
                       "avg_B2": g["B2"]})


def swb_conversion_record(convention=None) -> dict:
    """The archive stamp of an SWB-derived bootstrap: the toolkit's output
    convention and what :func:`_swb_jbs_to_toroidal` applied to it."""
    conv = swb_jbs_convention() if convention is None else convention
    applied = {SWB_JBS_RAVG_OVER_F: "kappa F/<R> (undo R_avg/F, then kappa)",
               SWB_JBS_TOROIDAL: "minus p'G (p'G is j_pressure)"}
    return {"swb_jbs_convention": conv,
            "swb_jbs_conversion": applied[conv]}


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
    the archive, enabling per-draw conversions at IDS write-back
    (:func:`toroidal_to_parallel`, and the IMAS ``j_tor`` of A5, from ``F``,
    ``<1/R>``, ``<1/R^2>``, ``<B^2>`` and ``<R>``) instead of the interim
    baseline-ratio reconstruction.

    Returns a dict of 1-D arrays on a uniform ``psi_N`` grid of ``npsi`` points
    (default 257 to match the archived eqdsk; do not go below ~129 -- the edge
    bootstrap needs it):

        ``psi_N``       normalised poloidal flux, [npsi]
        ``F``           R*B_phi flux function [T m]           (get_profiles)
        ``pprime``      p' [Pa/Wb], signed so ``jphi_eq`` > 0   (get_profiles)
        ``jphi_eq``     own TokaMaker jphi <R>p' + <1/R>FF'/mu0  [A/m^2]
        ``avg_R``       <R> [m]                               (get_q)
        ``avg_inv_R``   <1/R> [1/m]                           (sauter_fc)
        ``avg_inv_R2``  <1/R^2> [1/m^2]                       (exact quadrature)
        ``avg_B2``      <B^2> [T^2]                           (sauter_fc)
        ``q``           safety factor                         (get_q)
        ``dV_dpsi``     dV/dpsi                               (get_q, geo)
        ``f_trap``      trapped fraction f_c                  (sauter_fc)
        ``B_avg``       <|B|> [T]                             (sauter_fc)

    ``exact_inv_R2`` (default True) also records ``<1/R^2>`` (exposed by
    current OFT builds' ``get_q``; otherwise by flux-surface quadrature over
    traced contours, :func:`_capture_exact_inv_R2`).  It does not enter the
    field-aligned factor (:func:`field_aligned_conversion`); the IMAS
    ``j_tor`` written at IDS export (A5) needs it. By default it is traced on the
    FULL ``npsi`` grid -- same resolution as every other metric, most accurate
    at the edge where the surfaces bunch up and the bootstrap peaks; the trace
    is cheap (a few ms/surface, ~2 s at npsi=257). ``inv_R2_npsi`` (default
    ``None`` = ``npsi``) can be set smaller to trace ``<1/R^2>`` coarsely and
    spline it onto ``psi_N`` -- only worth it for very large bouquets. The
    quadrature is **self-validated** each call: its independently-recomputed
    ``<1/R>`` must agree with ``sauter_fc`` to ``inv_R2_check_rtol`` (default
    2%), else ``avg_inv_R2`` is dropped (with a warning) -- never silently
    wrong.

    Set ``exact_inv_R2=False`` to skip the ``<1/R^2>`` surface traces entirely
    if the capture cost is ever material.

    Pure extraction (no re-solve); ``mygs`` is passed in so this module stays
    OFT-import-free / headless-safe.
    """
    import warnings
    psi_hat = np.linspace(psi_pad, 1.0 - psi_pad, int(npsi))

    # F(psi) = R*B_phi and p' from the G-S source profiles
    _, F, Fp, _P, Pp = mygs.get_profiles(psi=psi_hat)

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

    # get_profiles' p' sign follows the case's flux convention: sign it so the
    # equilibrium's own TokaMaker jphi is positive, like bouquet's arrays (the
    # same rule as utils.eq_jphi_profile's pprime_sign).
    F = np.asarray(F, dtype=float)
    avg_R = np.asarray(q_ravg(geo_q, "<R>"), dtype=float)
    jphi_eq = (avg_R * np.asarray(Pp, dtype=float) + np.asarray(avg_inv_R, float)
               * F * np.asarray(Fp, dtype=float) / (4.0e-7 * np.pi))
    sign = 1.0 if float(np.sum(jphi_eq)) >= 0.0 else -1.0

    out = {
        "psi_N": psi_hat,
        "F": F,
        "pprime": sign * np.asarray(Pp, dtype=float),
        "jphi_eq": sign * jphi_eq,
        "avg_R": avg_R,
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
                f"exact <1/R^2> capture failed ({exc}); this draw cannot be "
                "IDS-exported exactly. Set exact_inv_R2=False to silence.")
    return out


#: OpenFUSIONToolkit's ``taper_edge_shape`` codes (bootstrap ``boot_ops``)
EDGE_TAPER_SHAPES = {1: "cos^2 (Hann)", 2: "quintic smoothstep",
                     3: "cubic power"}


def edge_taper_weight(psi_N, psi0=0.999, shape=2):
    """The factor SWB's edge taper (``taper_edge_jBS``) multiplies every
    toroidal current component by: 1 for ``psi_N < psi0``, falling to 0 at
    ``psi_N = 1`` with *shape* (1 cos^2, 2 quintic smoothstep, 3 cubic).
    A port of OFT ``grad_shaf_bootstrap.F90:apply_edge_taper`` (standard
    convention); no taper when ``1 - psi0 < 1e-6``."""
    psi = np.asarray(psi_N, dtype=float)
    w = np.ones_like(psi)
    psi0 = float(psi0)
    span = 1.0 - psi0
    if span < 1.0e-6:
        return w
    m = psi >= psi0
    t = np.clip((psi[m] - psi0) / span, 0.0, 1.0)
    shape = int(shape)
    if shape == 1:
        w[m] = np.cos(0.5 * np.pi * t) ** 2
    elif shape == 2:
        w[m] = 1.0 - t ** 3 * (6.0 * t ** 2 - 15.0 * t + 10.0)
    elif shape == 3:
        w[m] = (1.0 - t) ** 3
    else:
        raise ValueError(f"edge_taper_weight: unknown shape {shape!r} "
                         f"(one of {sorted(EDGE_TAPER_SHAPES)})")
    return w


#: The inverse aspect ratio entering the Redl collisionalities
#: (``nu_e*, nu_i* ~ eps^-3/2``), by name.  ``"geometric"`` (default since
#: ``evaluate_jBS/4``): ``eps = (R_max - R_min) / (2 <R>)`` per flux surface,
#: the half-width of the surface over its flux-surface-averaged major radius
#: -- the definition the Sauter (1999) / Redl (2021) fits are written in
#: (owner decision E4, 2026-10-09).  ``"a_over_R"``: ``eps = <a>/<R>``, the
#: dl/B_p-weighted mean distance from the magnetic axis over ``<R>`` (OFT
#: ``sauter_fc``'s ``<a>``), the definition of ``evaluate_jBS/1..3`` and of
#: OpenFUSIONToolkit main's own ``solve_with_bootstrap``; kept as an explicit
#: opt-in for A/B comparison.
EPS_DEFINITIONS = {
    "geometric": "(R_max - R_min)/(2<R>)",
    "a_over_R": "<a>/<R>",
}
#: The default of :func:`evaluate_jBS`'s ``eps_definition``.
EPS_DEFINITION_DEFAULT = "geometric"


def check_eps_definition(eps_definition):
    """Validate an ``eps_definition`` name; returns it."""
    if eps_definition not in EPS_DEFINITIONS:
        raise ValueError(
            f"eps_definition={eps_definition!r} is not one of "
            f"{sorted(EPS_DEFINITIONS)} "
            f"({'; '.join(f'{k}: {v}' for k, v in EPS_DEFINITIONS.items())})")
    return eps_definition


def evaluate_jbs_version(eps_definition=EPS_DEFINITION_DEFAULT):
    """The :data:`EVALUATE_JBS_VERSION` string for an ``eps_definition``
    (the opt-in ``"a_over_R"`` is named in the tag, so a record says which
    epsilon produced its bootstrap)."""
    eps_definition = check_eps_definition(eps_definition)
    eps_txt = ("geometric eps = (R_max-R_min)/(2<R>)"
               if eps_definition == "geometric" else
               "OPT-IN eps = <a>/<R> (the /3 definition)")
    return ("evaluate_jBS/4 (Redl 2021 jboot1, NRL/Zavg lnLambda, Koh nu_i*, "
            f"{eps_txt}, psi_N-native, kappa = F<1/R>/<B^2> toroidal "
            "conversion; p'G separate as j_pressure)")


#: Version tag of :func:`evaluate_jBS`, recorded with every loop record so an
#: archive states which evaluator produced its bootstrap.
#: ``/2`` (2026-10-06): the toroidal output is ``kappa <j.B>``, ``kappa =
#: F<1/R>/<B^2>`` (was ``<j.B>/(F<1/R>)`` in ``/1``); ``/3`` (PR #64, never
#: on main): plus ``p'G`` inside the bootstrap, with the Redl ``eps`` still
#: ``<a>/<R>`` on every OFT build that ran (PR #60's ``(R_max - R_min)/(2<R>)``
#: required a fork-only ``sauter_fc(return_eps=True)`` and raised elsewhere).
#: ``/4`` is ONE new convention carrying two owner decisions together:
#:
#: * D2 (PR #64): the toroidal output is ``kappa <j.B>`` only, as in ``/2``,
#:   and ``p'G`` is returned beside it as ``diag["j_pressure"]`` (the third
#:   current bucket) -- a new tag, so that a ``/3`` record (p'G inside j_BS)
#:   is never read as this convention (:func:`bouquet.schema.
#:   read_current_split_convention` keys on the ``/3`` prefix only);
#: * E4 (PR #60): the geometric ``eps = (R_max - R_min)/(2<R>)`` by default on
#:   every build -- from ``sauter_fc(return_eps=True)`` where the installed
#:   OFT has it, else from ``get_fsa``'s ``R_min``/``R_max`` over the sauter
#:   ``<R>`` (the two agree to rounding) -- with ``<a>/<R>`` as the opt-in
#:   ``eps_definition="a_over_R"``, whose records carry
#:   :func:`evaluate_jbs_version` of it (still ``evaluate_jBS/4``, still p'G
#:   separate; only the eps text differs).  ``nu_e*`` and ``nu_i*`` scale as
#:   ``eps^-3/2``, so the default bootstrap moves wherever the two epsilons
#:   differ (most in the pedestal; docs/CHANGES_SUMMARY.md).
EVALUATE_JBS_VERSION = evaluate_jbs_version(EPS_DEFINITION_DEFAULT)

#: Positional layout of ``sauter_fc``'s geometry block on OFT builds that
#: return it as a ``(3, n)`` array (builds after OpenFUSIONToolkit#313 return
#: a dict with these keys instead).
_SAUTER_RAVG_INDEX = {"<R>": 0, "<1/R>": 1, "<a>": 2}
#: ... and of its ``[<|B|>, <|B|^2>]`` block.
_SAUTER_MODB_INDEX = {"<|B|>": 0, "<|B|^2>": 1}

#: Elementary charge [C] -- the eV -> J factor of the Redl drive, the same
#: constant OFT's ``solve_with_bootstrap`` uses (:data:`ELEMENTARY_CHARGE`).
_EC = ELEMENTARY_CHARGE


class JBSEvaluationError(ValueError):
    """:func:`evaluate_jBS` refused an input or a flux-surface geometry on
    which the Redl bootstrap is undefined.

    Raised instead of returning ``j_BS = 0`` there.  ``quantity`` names what
    is wrong, ``psi_N`` / ``index`` the first grid node where it is, and
    ``n_bad`` how many nodes are affected.  A ``ValueError``, so callers that
    already treat a malformed evaluator input as a ``ValueError`` see it.
    """

    def __init__(self, message, *, quantity=None, psi_N=None, index=None,
                 n_bad=None):
        super().__init__(message)
        self.quantity = quantity
        self.psi_N = psi_N
        self.index = index
        self.n_bad = n_bad


def _first_bad(bad, psi_N, arr, quantity, rule, what="input"):
    """Raise :class:`JBSEvaluationError` at the first True of *bad*."""
    bad = np.asarray(bad, dtype=bool)
    if not np.any(bad):
        return
    i = int(np.argmax(bad))
    raise JBSEvaluationError(
        f"evaluate_jBS: {what} {quantity} must be {rule}; got "
        f"{float(np.asarray(arr, dtype=float)[i])!r} at psi_N="
        f"{float(psi_N[i]):.6g} (grid index {i}; {int(bad.sum())} of "
        f"{bad.size} node(s) affected).  The Redl bootstrap is undefined "
        "there -- refusing to return a silently zeroed j_BS.",
        quantity=quantity, psi_N=float(psi_N[i]), index=i,
        n_bad=int(bad.sum()))


def _sauter_avg(block, which, index):
    """Read one flux-surface average off a ``sauter_fc`` output block, dict or
    positional array (both OFT layouts are in production)."""
    if isinstance(block, dict):
        return np.asarray(block[which], dtype=float)
    return np.asarray(block, dtype=float)[index[which]]


def _accepts_kw(fn, name):
    """True when callable *fn* takes keyword *name* (explicitly or through
    ``**kwargs``); False when its signature cannot be read."""
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def geometric_eps(mygs, psi_u, sauter_fn=None, R_avg=None):
    """``(eps, route)``: ``(R_max - R_min)/(2<R>)`` on the surfaces *psi_u*.

    ``route`` is ``"sauter_fc(return_eps=True)"`` when the installed OFT's
    ``sauter_fc`` returns it (a 5-tuple: the fork's Fortran,
    ``(rmax_surf - rmin_surf)/(2 r_avgs(:,1))`` with ``r_avgs(:,1)`` the
    ``sauter_fc`` ``<R>``), else ``"get_fsa"``: the per-surface ``R_min`` and
    ``R_max`` of OFT's ``get_fsa`` (OFT >= v26.6) over *R_avg* -- pass the
    ``sauter_fc`` ``<R>`` of the same surfaces to divide by exactly the
    fork's denominator (``get_fsa``'s own ``<R>`` comes from a separate
    trace: 2.9e-6 relative apart on the synthetic D3D-like example); without
    it ``get_fsa``'s ``<R>`` is used.  Never raises for a missing fork
    option; raises :class:`RuntimeError` only when the build offers neither
    (pre-v26.6), naming ``eps_definition="a_over_R"``.  A failed trace's zero
    row gives ``eps = 0`` (refused by the caller)."""
    psi_u = np.ascontiguousarray(psi_u, dtype=float)
    if sauter_fn is None:
        sauter_fn = getattr(mygs, "sauter_fc", None) or getattr(
            mygs, "calc_sauter_fc")
    if _accepts_kw(sauter_fn, "return_eps"):
        out = sauter_fn(psi=psi_u.copy(), return_eps=True)
        if len(out) == 5:
            return np.asarray(out[4], dtype=float), "sauter_fc(return_eps=True)"
    get_fsa = getattr(mygs, "get_fsa", None)
    if get_fsa is None:
        raise RuntimeError(
            "evaluate_jBS: the geometric eps = (R_max - R_min)/(2<R>) needs "
            "OpenFUSIONToolkit's get_fsa (v26.6 or newer) or "
            "sauter_fc(return_eps=True); this build has neither.  Upgrade "
            "OpenFUSIONToolkit, or pass eps_definition='a_over_R' for the "
            "<a>/<R> epsilon of evaluate_jBS/3")
    f = get_fsa(psi=psi_u.copy())
    r_min = np.asarray(f["R_min"], dtype=float)
    r_max = np.asarray(f["R_max"], dtype=float)
    r_avg = np.asarray(f["<R>"] if R_avg is None else R_avg, dtype=float)
    eps = np.divide(r_max - r_min, 2.0 * r_avg, out=np.zeros_like(r_avg),
                    where=r_avg > 0.0)
    return eps, "get_fsa"


def evaluate_jBS(mygs, psi_N, ne, te, ni, ti, zeff, *, psi_pad=1e-3,
                 isolate_edge=False, smooth_axis=True, coord="psi_n",
                 eps_definition=None):
    r"""Redl bootstrap current on the CURRENT equilibrium, on the caller's grid.

    A faithful port of the inner physics of OpenFUSIONToolkit's
    ``solve_with_bootstrap`` (SWB) -- NRL electron / ``Zavg`` ion Coulomb
    logarithms, Koh multi-species ion collisionality, the electron
    collisionality, ``redl_bootstrap(formula_form='jboot1',
    use_sign_q=True)`` -- evaluated ONCE on whatever equilibrium ``mygs``
    holds.  No Grad-Shafranov solve happens inside: the self-consistent loop
    (:mod:`bouquet.jbs_loop`) owns the solves, this function only reads.

    It differs from SWB's inner evaluation in exactly three ways, each of
    which is the reason it exists:

    1. **Geometry on the caller's grid.**  ``F``, ``f_T = 1 - f_c``,
       ``eps`` (``eps_definition``), ``q`` and ``<R>`` are sampled at
       ``psi_eval = clip(psi_N, psi_pad, 1 - psi_pad)`` -- the caller's own
       surfaces -- not on a uniform grid of the same length.
    2. **Gradients on the true grid.**  ``d/dpsi = numpy.gradient(y, psi_N,
       edge_order=2) / (psi_bounds[1] - psi_bounds[0])`` with the CURRENT
       equilibrium's flux range; a non-uniform ``psi_N`` (e.g. a grid uniform
       in rho_tor) is differentiated as what it is.
    3. **Direct toroidal conversion.**  Redl returns the FSA parallel
       ``<j_BS.B>``; it is converted with the package's ONE field-aligned
       factor (:func:`field_aligned_conversion`; ``F``, ``<1/R>`` and
       ``<B^2>`` from the SAME surfaces), never through SWB's ``R_avg/F``
       projection and its undo::

           <j_phi>_BS = kappa <j.B>,   kappa = F <1/R> / <B^2>

       -- the field-aligned share of the plain flux-surface average the
       solver consumes (OpenFUSIONToolkit's ``jphi-linterp``).  The
       pressure-driven ``p'(<R> - F^2<1/R>/<B^2>)`` of (A7) is NOT in it: it
       is returned beside it as ``diag["j_pressure"]``, the third current
       bucket (owner decision D2, 2026-10-09), so the bootstrap multiplier,
       the per-draw jitter, ``floor_j_BS``, ``DIFF_BS`` and the loop's
       residual norm act on the bootstrap alone.  IMAS export groups it with
       the non-inductive currents (FUSE ``includes_bootstrap=true``).

    **Refusals, never a silent zero.**  The Redl expressions are undefined
    for non-physical input and on a surface the tracer failed on; the
    historical code mapped the resulting NaN to ``j_BS = 0`` at that node
    (and a non-positive temperature also corrupts the neighbours' gradients).
    Instead :class:`JBSEvaluationError` (a ``ValueError``) is raised, naming
    the quantity and the first ``psi_N`` where it happens, for

    * inputs: ``ne``, ``ni``, ``te``, ``ti`` not strictly positive, or
      ``zeff < 1``, at ANY node (including the axis and the separatrix);
    * geometry, at ANY node: a non-finite average, a non-positive ``<R>``,
      ``<1/R>``, ``<a>``, ``<B^2>`` or ``dV/dpsi``, ``F = 0``, ``q = 0``, or a
      trapped fraction ``f_T >= 1`` -- the signature of the all-zero row a
      failed flux-surface trace returns; and ``f_T <= 0`` on any surface
      other than the clipped axis surface;
    * a non-finite Redl ``<j.B>`` or toroidal ``j_BS`` at any node that is
      not an END node (below).

    **End nodes.**  Nodes whose geometry is the CLIPPED axis or separatrix
    surface -- ``psi_N <= psi_pad`` or ``psi_N >= 1 - psi_pad``, identified by
    coordinate, not by the value computed there -- are where the geometry is
    singular by construction (``f_T, eps -> 0`` at the axis, the separatrix
    limit at the edge).  There, and only there, the historical treatment is
    kept exactly: a non-finite value is mapped by ``numpy.nan_to_num(...,
    nan=0.0)``.  How many values that touched is recorded in
    ``diag["n_nonfinite_zeroed_at_ends"]``.  For every accepted input the
    returned profile is bit-identical to the pre-refusal evaluator.

    **Grids whose first intervals are finer than ``psi_pad``.**  A grid such
    as ``[0, 1.7e-4, 6.9e-4, 1.6e-3, ...]`` puts several surfaces inside
    ``psi_pad``, where the flux-surface tracer cannot resolve geometry.
    ``psi_pad`` is NOT changed and no surface is merged: every point keeps its
    own profile values and its own gradient on the true grid, and only the
    GEOMETRY lookup of the points inside the pad is taken at ``psi_pad`` (the
    innermost surface the tracer resolves).  Geometry is evaluated once per
    distinct clipped value and mapped back, so the solver is never handed a
    repeated surface.  The same holds at the separatrix side.

    Parameters
    ----------
    mygs : TokaMaker or TokaMaker_equilibrium
        Anything exposing ``get_profiles(psi=)``, ``sauter_fc(psi=)`` (a
        ``copy_eq()`` snapshot names it ``calc_sauter_fc``), ``get_q(psi=)``
        and ``psi_bounds`` -- a live solver or a snapshot.  Only these primitives are used, so the
        function runs on every OFT build bouquet supports (they exist with
        and without the ``psi_N=`` argument of ``solve_with_bootstrap``).
    psi_N : array_like
        Strictly increasing run grid in [0, 1] the kinetic profiles live on
        (ψ_N, or Φ_N with ``coord="phi_n"``).
    ne, te, ni, ti : array_like
        Densities [m^-3] and temperatures [eV] on ``psi_N``.
    zeff : array_like or float
        Effective charge on ``psi_N`` (a scalar is broadcast).
    psi_pad : float
        Geometry clip at the axis and the separatrix (SWB's own default).
    isolate_edge : bool
        Return the isolated edge spike (OFT ``analyze_bootstrap_edge_spike``
        ``masked_spike``, applied in SWB's own projection so the shelf/mask
        detection is identical) instead of the full profile.
    smooth_axis : bool
        Apply :func:`bouquet.TokaMaker_interface.smooth_jbs_transition` (the
        shared innermost-surface repair every SWB-derived profile receives).
        ``False`` returns the raw profile (the delta-composition mode needs it).
    coord : str
        The run coordinate.  ``"phi_n"``: the nodes are mapped to ψ_N on
        *mygs*'s own toroidal-flux map (``get_torflux_map``) first, and
        everything below -- geometry, gradients, end nodes, the edge spike,
        ``I_BS`` -- works on those ψ_N, exactly as OFT's Fortran bootstrap
        (``grad_shaf_bootstrap.F90``: kinetic values at the mapped nodes,
        gradients numerical in ψ).  The profile is returned on the same nodes.
    eps_definition : {"geometric", "a_over_R"}, optional
        The inverse aspect ratio of the Redl collisionalities (``nu* ~
        eps^-3/2``; :data:`EPS_DEFINITIONS`).  ``None`` = the default
        :data:`EPS_DEFINITION_DEFAULT` = ``"geometric"``, ``(R_max -
        R_min)/(2<R>)`` -- from ``sauter_fc(return_eps=True)`` when the
        installed OFT has it, else from ``get_fsa`` (:func:`geometric_eps`;
        never raises for the missing fork option).  ``"a_over_R"``: ``<a>/<R>``
        (``evaluate_jBS/3``), the explicit opt-in for A/B.  The definition, its
        route and the version tag are in ``diag``.

    Returns
    -------
    (j_BS_tor, diag)
        ``j_BS_tor`` -- toroidal FSA current density ``<j_phi>`` =
        ``kappa <j.B>`` [A/m^2] on ``psi_N`` (isolated/smoothed as
        requested).  ``diag`` --
        ``psi_eval``, ``f_T``, ``nu_e_star``, ``nu_i_star``, ``q``, ``eps``,
        ``R_avg``, ``F``, ``avg_inv_R``, ``avg_B2``, ``dpsi`` (signed flux
        range), ``j_dot_B`` (Redl ``<j.B>``), ``j_tor_full_raw`` (full
        profile, unsmoothed), ``j_tor_raw`` (selected profile before
        smoothing), ``I_BS`` (signed FSA integral of ``j_BS_tor`` [A]),
        ``j_pressure`` (= ``p_term``: the pressure-driven ``p'G`` on the
        same nodes, positive-frame ``p'``, never part of ``j_BS_tor``),
        ``eps_definition``, ``eps_formula``, ``eps_route``, ``version``.
    """
    eps_definition = check_eps_definition(
        EPS_DEFINITION_DEFAULT if eps_definition is None else eps_definition)
    psi_N = np.asarray(psi_N, dtype=float)
    n = psi_N.size
    if psi_N.ndim != 1 or n < 3:
        raise ValueError("evaluate_jBS: psi_N must be 1-D with >= 3 points")
    if not np.all(np.isfinite(psi_N)):
        raise ValueError("evaluate_jBS: psi_N contains non-finite values")
    if np.any(np.diff(psi_N) <= 0.0):
        raise ValueError("evaluate_jBS: psi_N must be strictly increasing")
    if psi_N[0] < 0.0 or psi_N[-1] > 1.0:
        raise ValueError(
            f"evaluate_jBS: psi_N must lie in [0, 1] (got {psi_N[0]}, "
            f"{psi_N[-1]})")
    psi_pad = float(psi_pad)
    if not (0.0 < psi_pad < 0.5):
        raise ValueError(f"evaluate_jBS: psi_pad must be in (0, 0.5), got "
                         f"{psi_pad!r}")
    x_run = psi_N
    if _coords.check_coord(coord) == _coords.PHI:
        psi_N = np.clip(np.asarray(_coords.psi_at(mygs, psi_N, coord),
                                   dtype=float), 0.0, 1.0)
        if np.any(np.diff(psi_N) <= 0.0) or not np.all(np.isfinite(psi_N)):
            raise ValueError("evaluate_jBS: the toroidal-flux map of this "
                             "equilibrium is not strictly increasing on the "
                             "run grid")

    def _prof(a, name):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            a = np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(f"evaluate_jBS: {name} has shape {a.shape}, "
                             f"expected ({n},) to match psi_N")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"evaluate_jBS: {name} contains non-finite "
                             "values")
        return a

    ne = _prof(ne, "ne")
    te = _prof(te, "te")
    ni = _prof(ni, "ni")
    ti = _prof(ti, "ti")
    zeff = _prof(zeff, "zeff")
    # the physical domain of the Redl inputs, at EVERY node
    for _nm, _a, _unit in (("ne", ne, "m^-3"), ("ni", ni, "m^-3"),
                           ("te", te, "eV"), ("ti", ti, "eV")):
        _first_bad(~(_a > 0.0), psi_N, _a, _nm,
                   f"strictly positive [{_unit}] on every node")
    _first_bad(~(zeff >= 1.0), psi_N, zeff, "zeff", ">= 1 on every node")
    # OpenFUSIONToolkit only after the input checks: the refusals above are
    # the evaluation's own input domain and hold without OFT importable
    # (bouquet.engine_draws.check_draw_kinetics mirrors them; the fast CI
    # suite runs without OFT).
    import OpenFUSIONToolkit.TokaMaker.bootstrap as _oft_bs

    # ---- geometry on the caller's surfaces (distinct clipped values only) ---
    psi_eval = np.clip(psi_N, psi_pad, 1.0 - psi_pad)
    psi_u, inv = np.unique(psi_eval, return_inverse=True)
    psi_u = np.ascontiguousarray(psi_u, dtype=float)
    _, F_u, Fp_u, _, pp_u = mygs.get_profiles(psi=psi_u.copy())
    # a live TokaMaker exposes sauter_fc; a copy_eq() snapshot
    # (TokaMaker_equilibrium) exposes the same routine as calc_sauter_fc
    _sfc = getattr(mygs, "sauter_fc", None)
    if _sfc is None:
        _sfc = getattr(mygs, "calc_sauter_fc")
    # the plain call (every OFT build): f_c, [<R>, <1/R>, <a>], [<|B|>, <|B|^2>]
    fc_u, r_sau, modb = _sfc(psi=psi_u.copy())[-3:]
    if eps_definition == "geometric":
        eps_u, eps_route = geometric_eps(
            mygs, psi_u, _sfc,
            R_avg=_sauter_avg(r_sau, "<R>", _SAUTER_RAVG_INDEX))
    else:   # the opt-in <a>/<R> of evaluate_jBS/3
        # (a failed trace's zero row makes this 0/0; it is refused below)
        with np.errstate(divide="ignore", invalid="ignore"):
            eps_u = (_sauter_avg(r_sau, "<a>", _SAUTER_RAVG_INDEX)
                     / _sauter_avg(r_sau, "<R>", _SAUTER_RAVG_INDEX))
        eps_route = "sauter_fc <a>/<R>"
    _, q_u, ravgs_q, *_rest = mygs.get_q(psi=psi_u.copy())
    F = np.asarray(F_u, dtype=float)[inv]
    f_T = (1.0 - np.asarray(fc_u, dtype=float))[inv]
    eps = np.asarray(eps_u, dtype=float)[inv]
    avg_inv_R = _sauter_avg(r_sau, "<1/R>", _SAUTER_RAVG_INDEX)[inv]
    avg_B2 = _sauter_avg(modb, "<|B|^2>", _SAUTER_MODB_INDEX)[inv]
    q = np.asarray(q_u, dtype=float)[inv]
    R_avg = np.asarray(q_ravg(ravgs_q, "<R>"), dtype=float)[inv]
    inv_R_q = np.asarray(q_ravg(ravgs_q, "<1/R>"), dtype=float)[inv]
    dV_dpsi = np.abs(np.asarray(q_ravg(ravgs_q, "dV/dPsi"), dtype=float))[inv]

    # ---- geometry validity, at EVERY node (a failed trace is a zero row) ---
    # END nodes: geometry sampled on the clipped axis / separatrix surface,
    # identified by COORDINATE (psi_eval at the clip), never by the value
    end = (psi_eval <= psi_pad) | (psi_eval >= 1.0 - psi_pad)
    axis_end = psi_eval <= psi_pad
    _a_sau = _sauter_avg(r_sau, "<a>", _SAUTER_RAVG_INDEX)[inv]
    _R_sau = _sauter_avg(r_sau, "<R>", _SAUTER_RAVG_INDEX)[inv]
    with np.errstate(invalid="ignore"):
        for _nm, _a in (("F", F), ("f_T = 1 - f_c", f_T), ("eps", eps),
                        ("<a>", _a_sau),
                        ("<R> (sauter_fc)", _R_sau),
                        ("<1/R> (sauter_fc)", avg_inv_R), ("<B^2>", avg_B2),
                        ("q", q), ("<R> (get_q)", R_avg),
                        ("<1/R> (get_q)", inv_R_q), ("dV/dpsi", dV_dpsi)):
            _first_bad(~np.isfinite(_a), psi_N, _a, _nm, "finite",
                       what="flux-surface average")
        for _nm, _a in (("eps", eps), ("<a>", _a_sau),
                        ("<R> (sauter_fc)", _R_sau),
                        ("<1/R> (sauter_fc)", avg_inv_R), ("<B^2>", avg_B2),
                        ("<R> (get_q)", R_avg), ("<1/R> (get_q)", inv_R_q),
                        ("dV/dpsi", dV_dpsi)):
            _first_bad(~(_a > 0.0), psi_N, _a, _nm,
                       "positive (zero is the row a failed flux-surface "
                       "trace returns)", what="flux-surface average")
        _first_bad(F == 0.0, psi_N, F, "F", "non-zero (failed trace?)",
                   what="flux function")
        _first_bad(q == 0.0, psi_N, q, "q", "non-zero (failed trace?)",
                   what="safety factor")
        _first_bad(~(f_T < 1.0), psi_N, f_T, "f_T = 1 - f_c",
                   "< 1 (f_c = 0 is the row a failed trace returns)",
                   what="trapped fraction")
        _first_bad(~(f_T > 0.0) & ~axis_end, psi_N, f_T, "f_T = 1 - f_c",
                   "> 0 away from the clipped axis surface",
                   what="trapped fraction")

    # ---- gradients on the TRUE grid, current flux range ---------------------
    bounds = np.asarray(mygs.psi_bounds, dtype=float)
    psi_range = float(bounds[1] - bounds[0])
    if psi_range == 0.0 or not np.isfinite(psi_range):
        raise ValueError(f"evaluate_jBS: degenerate psi_bounds {bounds}")

    def _d(y):
        return np.gradient(y, psi_N, edge_order=2) / psi_range

    dn_e = _d(ne)
    dT_e = _d(te)
    dn_i = _d(ni)
    dT_i = _d(ti)

    # ---- collisionality (verbatim SWB physics) ------------------------------
    ln_le, ln_lii = _oft_bs.calculate_ln_lambda(
        te, ti, ne, ni, zeff,
        electron_lnLambda_model="NRL", ion_lnLambda_model="Zavg")
    Zdom = 1.0                         # deuterium main ion
    Zavg = ne / ni
    Zion = (Zdom ** 2 * Zavg * zeff) ** 0.25
    nu_i_star = (4.90e-18 * np.abs(q) * R_avg * ni
                 * Zion ** 4 * ln_lii / (ti ** 2 * eps ** 1.5))
    nu_e_star = (6.921e-18 * np.abs(q) * R_avg * ne
                 * zeff * ln_le / (te ** 2 * eps ** 1.5))

    j_dot_B, _coeffs = _oft_bs.redl_bootstrap(
        psi_N=psi_N, Te=te, Ti=ti, ne=ne, ni=ni,
        pe=_EC * (ne * te), pi=_EC * (ni * ti),
        Zeff=zeff, R=R_avg, q=q, eps=eps, fT=f_T, I_psi=F,
        dT_e_dpsi=dT_e, dT_i_dpsi=dT_i,
        dn_e_dpsi=dn_e, dn_i_dpsi=dn_i,
        ln_lambda_e=ln_le, ln_lambda_ii=ln_lii,
        nu_e_star_override=nu_e_star, nu_i_star_override=nu_i_star,
        use_legacy_L34=False, use_sign_q=True, formula_form="jboot1")
    n_zeroed = [0]

    def _ends_only(y, what):
        """The historical ``nan_to_num(y, nan=0.0)`` at END nodes only; a
        non-finite value anywhere else is refused."""
        y = np.asarray(y, dtype=float)
        bad = ~np.isfinite(y)
        _first_bad(bad & ~end, psi_N, y, what,
                   "finite away from the clipped axis/separatrix nodes",
                   what="Redl result")
        n_zeroed[0] += int(np.count_nonzero(bad & end))
        return np.nan_to_num(y, nan=0.0)

    j_dot_B = _ends_only(j_dot_B, "<j_BS.B>")

    # the toroidal bootstrap is the field-aligned kappa <j.B> ONLY: the
    # pressure-driven p'G is its own bucket, j_pressure (owner decision D2,
    # 2026-10-09), returned beside it and never scaled, jittered, floored or
    # differenced with the bootstrap.  p' signed so the equilibrium's own
    # jphi is positive (capture_equilibrium_fsa's rule).
    _pp = np.asarray(pp_u, dtype=float)[inv]
    _jeq = R_avg * _pp + inv_R_q * F * np.asarray(Fp_u, dtype=float)[inv] / (
        4.0e-7 * np.pi)
    _pp_sign = 1.0 if float(np.sum(_jeq)) >= 0.0 else -1.0
    geom = {"F": F, "avg_inv_R": avg_inv_R, "avg_B2": avg_B2,
            "avg_R": R_avg, "pprime": _pp_sign * _pp}
    p_term = jphi_tokamaker_pressure_term(geom)
    j_tor_full = _ends_only(parallel_to_toroidal(j_dot_B, geom=geom),
                            "toroidal j_BS")

    if isolate_edge:
        # SWB isolates the spike on its OWN projection <j.B> R_avg/F (the
        # shelf/mask detection is value-dependent), and bouquet then converts
        # the masked spike by the per-surface factor.  Same order here.
        swb_proj = j_dot_B * (R_avg / F)
        res = _oft_bs.analyze_bootstrap_edge_spike(psi_N, swb_proj)
        masked = np.asarray(res["masked_spike"], dtype=float)
        j_tor_sel = _ends_only(parallel_to_toroidal(
            masked * F / R_avg, geom=geom), "isolated toroidal j_BS")
    else:
        j_tor_sel = j_tor_full

    if smooth_axis:
        from .TokaMaker_interface import smooth_jbs_transition
        j_out = smooth_jbs_transition(j_tor_sel)
    else:
        j_out = np.asarray(j_tor_sel, dtype=float).copy()

    # FSA current of the delivered profile ('fsa' measure: V'/2pi <1/R>).
    w_fsa = dV_dpsi / (2.0 * np.pi) * inv_R_q * abs(psi_range)
    from scipy.integrate import trapezoid as _trap
    I_BS = float(_trap(w_fsa * j_out, psi_N))

    diag = dict(
        psi_eval=psi_eval, psi_pad=psi_pad,
        n_geometry_surfaces=int(psi_u.size),
        f_T=f_T, nu_e_star=nu_e_star, nu_i_star=nu_i_star, q=q, eps=eps,
        R_avg=R_avg, F=F, avg_inv_R=avg_inv_R, avg_B2=avg_B2,
        pprime=geom["pprime"], p_term=p_term, j_pressure=p_term,
        ln_lambda_e=np.asarray(ln_le, dtype=float),
        ln_lambda_ii=np.asarray(ln_lii, dtype=float),
        dpsi=psi_range, j_dot_B=j_dot_B,
        j_tor_full_raw=j_tor_full, j_tor_raw=np.asarray(j_tor_sel, float),
        I_BS=I_BS, isolate_edge=bool(isolate_edge),
        n_nonfinite_zeroed_at_ends=int(n_zeroed[0]),
        smooth_axis=bool(smooth_axis),
        eps_definition=eps_definition,
        eps_formula=EPS_DEFINITIONS[eps_definition], eps_route=eps_route,
        version=evaluate_jbs_version(eps_definition),
        coord=str(coord), x=np.asarray(x_run, dtype=float), psi_N=psi_N,
    )
    return j_out, diag


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

    CONVENTION: ``zeff`` must have only THERMAL species in its numerator,
    over the FULL ``ne`` (``sum_thermal(n_s Z_s^2) / ne``) -- what the
    ``zeff * ne / ne_th`` renormalization assumes.  A thermal+fast numerator
    (IMAS's own ``zeff`` expression, and a MEASURED Z_eff) counts the fast
    charge twice and ``Z_imp`` comes out too high, so :mod:`bouquet.io.imas`
    passes its own thermal-numerator recomputation, never the dd's ``zeff``.
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


def fast_ion_density_equivalent(z_fast, z2_fast, Z_imp):
    """The fast-ion part of a main-ion density derived from a MEASURED Z_eff [m^-3].

    Each fast species enters quasineutrality with weight ``Z_s`` and the Z_eff
    numerator with ``Z_s^2``, so::

        sum_s n_s^fast Z_s (Z_imp - Z_s) / (Z_imp - 1)
            == (Z_imp z_fast - z2_fast) / (Z_imp - 1)

    with ``z_fast = sum_s Z_s n_s^fast`` and ``z2_fast = sum_s Z_s^2 n_s^fast``
    (no beam charge assumed).  A hydrogenic beam gives the fast density; a
    species at ``Z_imp`` gives zero.
    """
    Z_imp = float(Z_imp)
    if not Z_imp > 1.0:
        raise ValueError(f"Z_imp must exceed 1 (got {Z_imp})")
    return (Z_imp * np.asarray(z_fast, dtype=float)
            - np.asarray(z2_fast, dtype=float)) / (Z_imp - 1.0)


def _require_z2(z2_fast, who):
    if z2_fast is None:
        raise ValueError(
            f"{who}: zeff_includes_fast=True needs z2_fast (= sum_s Z_s^2 "
            f"n_s^fast) as well as z_fast. A measured Z_eff weights each fast "
            f"species by Z_s^2 while quasineutrality weights it by Z_s, so the "
            f"two moments are independent and the beam charge cannot be "
            f"inferred from z_fast alone. Pass Baseline.z2_fast.")
    return z2_fast


def main_ion_density_from_zeff(ne, zeff, Z_imp, z_fast=None, z2_fast=None,
                               zeff_includes_fast=False):
    """Main-ion density from (ne, Zeff) under single-impurity quasineutrality.

    ::

        ni  = ne (Z_imp - Zeff) / (Z_imp - 1)
        nz  = (ne - ni) / Z_imp            (the implied impurity density)

    For ``1 <= Zeff <= Z_imp`` this guarantees ``0 <= ni <= ne`` and
    ``nz >= 0`` -- the consistent (ne, ni, Zeff, nz) set that the independent
    per-channel draws cannot provide. Returns ``ni``.

    With a fast-ion population the result is the THERMAL main-ion density, and
    which formula gives it depends on what is in ``zeff``'s numerator.  The
    fast population enters through its charge moments ``z_fast`` (= sum_s Z_s
    n_s^fast) and ``z2_fast`` (= sum_s Z_s^2 n_s^fast); see
    :func:`fast_ion_density_equivalent`.  Both branches reduce to the plain
    form when the fast population is absent.

    ``zeff_includes_fast=False`` -- THERMAL numerator over the full ``ne``
    (``zeff = sum_thermal(n_s Z_s^2)/ne``).  Only the charge matters here::

        ni = (Z_imp (ne - z_fast) - Zeff ne) / (Z_imp - 1)

    ``zeff_includes_fast=True`` -- ALL ions in the numerator, fast included.
    This is IMAS's ``zeff`` expression (``ion.density`` = thermal + fast), and
    what a MEASURED Z_eff is: VB bremsstrahlung counts a beam ion by
    its own Z_s exactly like a thermal one, and a CER Z_eff built as
    ``1 + Z(Z-1) nC/ne`` inherits the same normalisation.  Then ``z2_fast`` is
    REQUIRED, because the numerator weights the beam by ``Z_s^2``::

        ni = ne (Z_imp - Zeff)/(Z_imp - 1) - (Z_imp z_fast - z2_fast)/(Z_imp - 1)

    The two conventions differ by ``z2_fast/(Z_imp - 1)``; :func:`zeff_bounds`
    gives each one's validity window.
    """
    ne = np.asarray(ne, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    Z_imp = float(Z_imp)
    if not Z_imp > 1.0:
        raise ValueError(f"Z_imp must exceed 1 (got {Z_imp})")
    if z_fast is None:
        return ne * (Z_imp - zeff) / (Z_imp - 1.0)
    if zeff_includes_fast:
        _require_z2(z2_fast, "main_ion_density_from_zeff")
        return (ne * (Z_imp - zeff) / (Z_imp - 1.0)
                - fast_ion_density_equivalent(z_fast, z2_fast, Z_imp))
    ne_th = np.maximum(ne - np.asarray(z_fast, dtype=float), 0.0)
    return (Z_imp * ne_th - zeff * ne) / (Z_imp - 1.0)


def zeff_bounds(ne, Z_imp, z_fast=None, z2_fast=None,
                zeff_includes_fast=False):
    """``(lo, hi)`` on Z_eff: the window where ni >= 0 and nz >= 0.

    A Z_eff draw is clipped to it.  Without fast ions it is ``[1, Z_imp]``;
    with them it follows the convention of :func:`main_ion_density_from_zeff`::

        zeff_includes_fast=False
            [ne_th/ne,                     Z_imp ne_th/ne]
        zeff_includes_fast=True
            [1 + (z2_fast - z_fast)/ne,    Z_imp - (Z_imp z_fast - z2_fast)/ne]

    Both reduce to ``[1, Z_imp]`` when the fast population is absent, and the
    ``True`` window reduces to ``[1, Z_imp - (Z_imp - 1) z_fast/ne]`` for a
    hydrogenic beam.  Returns scalars when ``z_fast`` is None and arrays
    otherwise; ``hi`` is not clamped above ``lo``, so a surface whose fast
    population overwhelms ``ne`` yields an empty window the caller can detect.
    """
    Z_imp = float(Z_imp)
    if z_fast is None:
        return 1.0, Z_imp
    ne = np.asarray(ne, dtype=float)
    _ne = np.clip(ne, 1e-30, None)
    zf = np.asarray(z_fast, dtype=float)
    if zeff_includes_fast:
        z2 = np.asarray(_require_z2(z2_fast, "zeff_bounds"), dtype=float)
        return 1.0 + (z2 - zf) / _ne, Z_imp - (Z_imp * zf - z2) / _ne
    f = np.clip(zf / _ne, 0.0, 1.0)
    return 1.0 - f, Z_imp * (1.0 - f)


# Elementary charge [C] -- thermal pressure p = e * sum_s(n_s * T_s) with n in
# m^-3 and T in eV (:data:`ELEMENTARY_CHARGE`).
_EC = ELEMENTARY_CHARGE


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
