"""Reader for FUSE IMAS/OMAS data-dictionary files (``dd_sim.json``).

FUSE writes the IMAS data dictionary as a plain JSON dump, so this reads with the
stdlib ``json`` module -- no IMAS/OMAS/OMFIT install required. It returns a
fully-separated baseline (no GS reconstruction needed): the IDS already carries
j_ohmic, j_bootstrap and the driven currents split apart.

Field mapping (verified against a D3D FUSE run)::

    equilibrium.time_slice[t].global_quantities.ip            -> Ip_target
    equilibrium.time_slice[t].global_quantities.li_3          -> l_i_target
    core_profiles.profiles_1d[t].grid.psi                     -> normalised -> psi_N
    core_profiles.profiles_1d[t].j_total                      -> total PARALLEL current
    core_profiles.profiles_1d[t].j_tor                        -> total TOROIDAL current
    core_profiles.profiles_1d[t].j_ohmic                      -> inductive (parallel)
    core_profiles.profiles_1d[t].j_bootstrap                  -> bootstrap (parallel)
    core_profiles.profiles_1d[t].electrons.{density_thermal,temperature}
    core_profiles.profiles_1d[t].ion[*].{density_thermal,temperature,element[].z_n}
    core_profiles.profiles_1d[t].{electrons,ion[*]}.pressure_fast_{perpendicular,parallel}
    core_sources.source[*].profiles_1d[t].j_parallel          -> beam-source j_NBI only

Currents are converted parallel->toroidal (see :func:`bouquet.physics.parallel_to_toroidal`)
via the per-surface factor c = j_tor/j_total, and fast pressure is isotropized
(see :func:`bouquet.physics.isotropize_fast_pressure`). The total j_phi is set to
the authoritative toroidal ``j_tor`` and the inductive component is taken as the
residual ``j_phi - j_BS - j_NBI - j_RF`` so the decomposition sums exactly and Ip
is preserved.

Note: ``j_BS`` read here is the FUSE bootstrap baseline, but it is *overridden*
when ``GenerationConfig.recalculate_j_BS`` is True -- bouquet then recomputes
bootstrap per draw via TokaMaker ``solve_with_bootstrap`` (whose output is also
parallel and must be converted to toroidal; see ``parallel_to_toroidal``).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ..physics import (fast_ion_density_equivalent, impurity_pressure,
                       impurity_charge_with_fast_ions,
                       isotropize_fast_pressure,
                       parallel_to_toroidal)

# Elementary charge [C]: thermal pressure p = e * sum_s(n_s * T_s).
_EC = 1.602176634e-19

if TYPE_CHECKING:
    from ..config import ImasSource, FixedComponentsConfig
    from ..baseline import Baseline

# Core-source identifier index for neutral-beam current drive.
NBI_SOURCE_INDEX = 2          # neutral beam injection -> summed into j_NBI
# Core-source identifier index for the sawtooth model (IMAS core_sources
# identifier enumeration).  NOT summed into any current here: it is read only as
# a slice-level FLAG -- "is the source's sawtooth model doing anything at this
# time?" -- for the closure_channel="sawtooth_bootstrap" gate, which pins q0
# only where sawteeth make q0 ~ 1 a physical fact rather than a model artefact.
SAWTOOTH_SOURCE_INDEX = 701
# NOTE: j_RF is NOT computed internally (RF is the least-common input). It is
# left as zeros and accepted as a user-supplied array via
# FixedComponentsConfig.j_RF. See the "revisit RF" flag in the project notes
# if/when internal EC/IC/LH summation is wanted.


def _nearest_index(time_array, t: Optional[float], what: str) -> int:
    """Index of the slice nearest ``t`` (seconds) in ``time_array``."""
    ta = np.asarray(time_array, dtype=float)
    if ta.size == 1:
        return 0
    if t is None:
        raise ValueError(
            f"{what} has {ta.size} time slices; set ImasSource.time (seconds). "
            f"Range [s]: [{ta.min():.4f}, {ta.max():.4f}]"
        )
    return int(np.argmin(np.abs(ta - t)))


# ===========================================================================
#  Fast-pressure storage convention
#
#  ``pressure_fast_parallel`` / ``pressure_fast_perpendicular`` are written with
#  two incompatible meanings, and the scalar p_fast they reduce to differs by a
#  factor of THREE:
#
#    * IMAS.jl / FUSE store the PER-DEGREE-OF-FREEDOM pressure -- ``pressa/3``
#      in each field -- so the scalar is ``p_par + 2*p_perp``  ("sum").
#    * The IMAS data dictionary (and OMAS-written dds) define
#      ``pressure_fast_parallel`` as the FULL fast parallel pressure, so the
#      scalar is the trace ``(p_par + 2*p_perp)/3``            ("trace").
#
#  Neither convention is recorded in a dedicated dd field, so bouquet infers it
#  from the dd's recorded provenance (:func:`detect_p_fast_convention`) and says
#  loudly when it cannot.  An explicit ``p_fast_reduction`` always wins.
# ===========================================================================

#: Where a producer records who wrote an IDS.  Only these slots are inspected --
#: deliberately NOT every IDS, because sub-IDSes are routinely imported from
#: other codes (a FUSE dd carries an OMFIT-sourced ``nbi.ids_properties.comment``)
#: and would otherwise vote on a convention they do not set.
_PROVENANCE_IDS = ("dataset_description", "core_profiles", "equilibrium", "summary")
_PROVENANCE_SLOTS = (
    ("ids_properties", "comment"),
    ("ids_properties", "provider"),
    ("ids_properties", "source"),
    ("code", "name"),
    ("code", "description"),
    ("code", "repository"),
)

#: Top-level keys that only IMASdd.jl / FUSE write (they are not IMAS DD IDSes).
#: Verified present in FUSE ``dd_sim.json`` output.
_IMASJL_TOPLEVEL_MARKERS = (
    "global_time", "requirements", "build", "balance_of_plant",
    "solid_mechanics", "costing",
)

#: Producer name fragments (matched case-insensitively in the slots above).
_FUSE_PRODUCER_TOKENS = ("fuse", "imas.jl", "imasdd", "torreypines")
_DD_PRODUCER_TOKENS = ("omas", "omfit", "imaspy", "imas-python", "imas_core",
                       "access-layer", "access layer")

#: Explicit stamp a producer (or a hand-built fixture) can put in any of the
#: provenance slots above to state its convention outright, e.g.
#: ``"comment": "... p_fast_reduction=trace ..."``.
_EXPLICIT_STAMP_RE = (
    r"(?:p_fast_reduction|pressure_fast_convention|fast_pressure_convention)"
    r"\s*[:=]\s*([A-Za-z_-]+)"
)
#: Values the stamp may carry -> the reduction rule they select.
_STAMP_VALUES = {
    "sum": "sum", "per_dof": "sum", "per-dof": "sum",
    "per_degree_of_freedom": "sum",
    "trace": "trace", "total": "trace", "full": "trace",
    "mean": "mean", "perp": "perp",
}


def _provenance_strings(dd: dict):
    """``(location, text)`` for every recorded producer string in ``dd``.

    Only the slots in ``_PROVENANCE_IDS`` x ``_PROVENANCE_SLOTS`` are read.
    """
    out = []
    for ids in _PROVENANCE_IDS:
        node = dd.get(ids)
        if not isinstance(node, dict):
            continue
        for grp, key in _PROVENANCE_SLOTS:
            sub = node.get(grp)
            if not isinstance(sub, dict):
                continue
            val = sub.get(key)
            if isinstance(val, str) and val.strip():
                out.append((f"{ids}.{grp}.{key}", val.strip()))
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, str) and v.strip():
                        out.append((f"{ids}.{grp}.{key}", v.strip()))
    return out


def detect_p_fast_convention(dd: dict) -> dict:
    """Infer the fast-pressure storage convention of a loaded dd.

    Returns ``{"rule": <"sum"|"trace"|"mean"|"perp"|None>, "basis": str,
    "evidence": str}``.  ``rule`` is ``None`` when the dd records nothing that
    identifies its producer -- the caller then has to choose a fallback and warn.

    Precedence, most specific first:

    1. **explicit stamp** -- ``p_fast_reduction=<rule>`` (or
       ``pressure_fast_convention=<per_dof|total>``) in any provenance slot;
    2. **IMAS.jl structure** -- a top-level key only IMASdd.jl/FUSE writes
       (``global_time``, ``requirements``, ``build``, ``balance_of_plant``,
       ``solid_mechanics``, ``costing``).  This identifies the *writer of the
       file*, which is what sets the convention, so it outranks per-IDS producer
       strings: a FUSE dd legitimately carries IDSes imported from other codes.
    3. **producer strings** -- a FUSE/IMAS.jl name in a provenance slot selects
       ``"sum"``; an OMAS/OMFIT/IMASPy (data-dictionary) name selects ``"trace"``.
    """
    import re

    prov = _provenance_strings(dd)

    # 1. explicit stamp
    for where, text in prov:
        m = re.search(_EXPLICIT_STAMP_RE, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            if val in _STAMP_VALUES:
                return {"rule": _STAMP_VALUES[val], "basis": "explicit-stamp",
                        "evidence": f"{where} = {text!r}"}
            # A dd that states its own convention and is then misread is exactly
            # what the stamp exists to prevent -- never discard one silently.
            import warnings
            warnings.warn(
                f"{where} carries a fast-pressure convention stamp whose value "
                f"{m.group(1)!r} is not recognised; expected one of "
                + ", ".join(repr(v) for v in sorted(_STAMP_VALUES))
                + ". The stamp is ignored and the convention is inferred from the "
                "dd's structure/producer instead.",
                stacklevel=2,
            )

    # 2. IMAS.jl / FUSE structural markers
    found = [k for k in _IMASJL_TOPLEVEL_MARKERS if k in dd]
    if found:
        return {"rule": "sum", "basis": "imas.jl-structure",
                "evidence": ("top-level key(s) only IMASdd.jl/FUSE write: "
                             + ", ".join(sorted(found)))}

    # 3. producer strings
    for where, text in prov:
        low = text.lower()
        for tok in _FUSE_PRODUCER_TOKENS:
            if tok in low:
                return {"rule": "sum", "basis": "producer-string",
                        "evidence": f"{where} = {text!r} (matched {tok!r})"}
    for where, text in prov:
        low = text.lower()
        for tok in _DD_PRODUCER_TOKENS:
            if tok in low:
                return {"rule": "trace", "basis": "producer-string",
                        "evidence": f"{where} = {text!r} (matched {tok!r})"}

    return {"rule": None, "basis": "undetermined",
            "evidence": ("no IMASdd.jl top-level key and no producer recorded in "
                         "{dataset_description,core_profiles,equilibrium,summary}"
                         ".{ids_properties.{comment,provider,source},"
                         "code.{name,description,repository}}")}


#: Rule used when provenance is undeterminable.  "sum" is the majority case for
#: the dds bouquet reads (FUSE), so it is the safer bet -- but it is never
#: applied silently; see :func:`resolve_p_fast_reduction`.
P_FAST_UNDETERMINED_FALLBACK = "sum"


def warn_p_fast_undetermined(rule: str = P_FAST_UNDETERMINED_FALLBACK,
                             stacklevel: int = 2) -> None:
    """The factor-of-3 warning for a dd whose convention cannot be determined.

    Split out of :func:`resolve_p_fast_reduction` so a caller that resolves the
    rule eagerly can hold the warning back until it knows the choice moved a
    number -- see :func:`read_imas_baseline`.
    """
    import warnings
    warnings.warn(
        "p_fast_reduction='auto': this data dictionary records no producer, so the "
        "storage convention of pressure_fast_parallel/perpendicular cannot be "
        "determined. The two conventions differ by a FACTOR OF 3 in the fast-ion "
        "pressure (and hence in beta_N, W_MHD and p'):\n"
        "  'sum'   -- IMAS.jl/FUSE write the pressure PER DEGREE OF FREEDOM "
        "(pressa/3 in each field); scalar p_fast = p_par + 2*p_perp.\n"
        "  'trace' -- the IMAS data dictionary (and OMAS-written dds) define "
        "pressure_fast_parallel as the FULL parallel pressure; scalar "
        "p_fast = (p_par + 2*p_perp)/3.\n"
        f"Falling back to {rule!r} (the majority case for the dds bouquet reads). "
        "Set it explicitly to silence this: "
        "BouquetConfig.fixed_components.p_fast_reduction = 'sum' | 'trace' (or "
        "read_imas_baseline(..., p_fast_reduction=...)); alternatively stamp the dd "
        "itself, e.g. core_profiles.ids_properties.comment = "
        "'... p_fast_reduction=trace ...'.",
        stacklevel=stacklevel + 1,
    )


def resolve_p_fast_reduction(dd: dict, requested: str = "auto",
                             warn: bool = True) -> dict:
    """Resolve the fast-pressure reduction rule for ``dd``.

    ``requested`` is ``"auto"`` (inspect the dd's provenance) or one of the
    explicit rules, which always wins and is applied silently.

    Returns the metadata dict recorded on :attr:`bouquet.baseline.Baseline.p_fast_meta`::

        {"rule": str, "basis": str, "evidence": str, "requested": str,
         "warned": bool}

    When ``requested == "auto"`` and the dd's provenance is undeterminable the
    rule falls back to :data:`P_FAST_UNDETERMINED_FALLBACK` and a single
    ``UserWarning`` is raised naming both conventions and the factor-of-3 stake.
    ``warn=False`` resolves the metadata without raising it; the caller then owns
    the warning and can key on ``basis == "undetermined-fallback"``.
    """
    from ..physics import P_FAST_REDUCTIONS

    if requested != "auto":
        if requested not in P_FAST_REDUCTIONS:
            raise ValueError(
                f"unknown p_fast_reduction {requested!r}; expected 'auto' or one of "
                + ", ".join(repr(r) for r in P_FAST_REDUCTIONS))
        return {"rule": requested, "basis": "explicit-argument",
                "evidence": f"p_fast_reduction={requested!r} supplied by the caller",
                "requested": requested, "warned": False}

    det = detect_p_fast_convention(dd)
    if det["rule"] is not None:
        return {**det, "requested": "auto", "warned": False}

    rule = P_FAST_UNDETERMINED_FALLBACK
    if warn:
        warn_p_fast_undetermined(rule, stacklevel=2)
    return {"rule": rule, "basis": "undetermined-fallback",
            "evidence": det["evidence"], "requested": "auto",
            "warned": bool(warn)}


def _isotropic_fast_pressure(species: dict, method: str, n: int,
                             missing_parallel=None, label: str = ""):
    """Isotropized fast pressure for one species, or zeros if it carries none.

    When ``pressure_fast_parallel`` is absent the reader closes the tensor with
    ``p_par := p_perp``.  The closure is the same for every rule; what it MEANS
    is not:

      * ``"trace"`` / ``"mean"`` / ``"perp"`` (full directional pressures):
        ``p_par = p_perp`` is the isotropic fast-ion assumption and the scalar
        comes out as ``p_perp`` -- identical to reading a fully isotropic dd.
      * ``"sum"`` (per-degree-of-freedom fields): the same closure gives
        ``3*p_perp``.  That is the literal per-dof reading, but it is ambiguous --
        a writer that simply omitted an all-zero parallel field means
        ``2*p_perp``, and a dictionary-convention dd means ``p_perp``.  The
        caller is warned rather than left with a silent 3x.

    Species whose parallel field is missing while the perpendicular field is
    non-zero are appended to ``missing_parallel`` so the caller warns once.
    """
    if "pressure_fast_perpendicular" not in species:
        return np.zeros(n)
    p_perp = np.asarray(species["pressure_fast_perpendicular"], dtype=float)
    if "pressure_fast_parallel" in species:
        p_par = np.asarray(species["pressure_fast_parallel"], dtype=float)
    else:
        p_par = p_perp
        if missing_parallel is not None and float(np.max(np.abs(p_perp))) > 0.0:
            missing_parallel.append(label or species.get("label", "?"))
    return isotropize_fast_pressure(p_perp, p_par, method=method)


def _warn_missing_parallel(species_labels, rule: str):
    """Single warning for species that carry only the perpendicular fast field."""
    if not species_labels:
        return
    import warnings
    who = ", ".join(str(s) for s in species_labels)
    if rule == "sum":
        detail = (
            "Under p_fast_reduction='sum' the two fields are per-degree-of-freedom, "
            "so the closure p_par := p_perp yields 3*p_perp for these species. If "
            "the producer instead omitted an all-zero parallel field the correct "
            "scalar is 2*p_perp (1.5x less); if the dd follows the IMAS data "
            "dictionary it is p_perp (3x less). Supply pressure_fast_parallel, or "
            "set p_fast_reduction explicitly.")
    else:
        detail = (
            f"Under p_fast_reduction={rule!r} the two fields are the full "
            "directional pressures, so the closure p_par := p_perp is the isotropic "
            "fast-ion assumption and the scalar is p_perp.")
    warnings.warn(
        f"core_profiles species [{who}] carry pressure_fast_perpendicular but no "
        f"pressure_fast_parallel. {detail}", stacklevel=3)


def _override(arr, src_psi, dst_psi):
    """Resample a user-supplied fixed-component array onto the baseline grid."""
    arr = np.asarray(arr, dtype=float)
    if src_psi is None:
        if arr.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the baseline grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return arr
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), arr)


def read_imas_geometry(source: "ImasSource"):
    """Return ``(F0, boundary_RZ)`` from a FUSE IDS for TokaMaker setup.

    ``F0 = |r0 * b0(t)|`` from ``equilibrium.vacuum_toroidal_field`` and the LCFS
    isoflux points from ``equilibrium.time_slice[t].boundary.outline``. Used by
    :meth:`Bouquet.setup_solver` when the source is an :class:`ImasSource`
    (replacing the g-file that the reconstruction path reads F0/boundary from).
    """
    import json

    with open(source.ids_path) as fh:
        dd = json.load(fh)
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], source.time, "equilibrium")
    vtf = eq["vacuum_toroidal_field"]
    r0 = float(vtf["r0"])
    b0 = vtf["b0"]
    b0v = float(b0[ie]) if isinstance(b0, list) else float(b0)
    F0 = abs(r0 * b0v)
    # Separatrix: an optional external g-file (if given) supplies the LCFS;
    # otherwise use the dd boundary outline. F0 stays from the dd vacuum field
    # either way (same shot).
    lcfs_g = getattr(source, "LCFS_geqdsk", None)
    if lcfs_g:
        from .geqdsk import read_geqdsk
        g = read_geqdsk(lcfs_g)
        boundary_RZ = np.column_stack([
            np.asarray(g.boundary_R, dtype=float),
            np.asarray(g.boundary_Z, dtype=float),
        ])
    else:
        out = eq["time_slice"][ie]["boundary"]["outline"]
        boundary_RZ = np.column_stack([
            np.asarray(out["r"], dtype=float), np.asarray(out["z"], dtype=float),
        ])
    return F0, boundary_RZ


def _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                    p_equilibrium, allow_incomplete,
                                    anchor_pressure=False, p_fast_meta=None):
    """Fail-fast when the IMAS source carries pressure the reconstruction omits.

    Hard fails (raise, or warn if ``allow_incomplete``) on DROPPED reconstructable
    components -- a thermal species or the fast channel:
      1. species  -- e*(ne*Te + ni*Ti) + impurity(nz*Ti) vs e*(ne*Te + Sum n_s*T_s)
      2. fast     -- any species carries pressure_fast_* but assembled p_fast ~ 0
    Plus a non-fatal backstop (warning only):
      3. reconstructed total vs equilibrium.pressure -- the equilibrium routinely
         carries more (anisotropic/beam + equilibrium-vs-transport gap) which the
         p_diff anchor absorbs; informational, not blocking.
    """
    problems = []
    # 1. species completeness (single-impurity model vs full Sum_s n_s T_s)
    full_thermal = _EC * ne * te
    for ion in cp["ion"]:
        full_thermal = full_thermal + _EC * (
            np.asarray(ion["density_thermal"], dtype=float)
            * np.asarray(ion["temperature"], dtype=float))
    recon_thermal = _EC * (ne * te + ni * ti) + p_imp
    denom = float(np.mean(np.abs(full_thermal))) or 1.0
    sp_gap = float(np.mean(np.abs(full_thermal - recon_thermal))) / denom
    if sp_gap > 0.02:
        problems.append(
            f"thermal species gap {sp_gap*100:.1f}% (>2%): single-impurity model "
            f"(Z_imp from Zeff, impurity at main-ion Ti) does not reproduce "
            f"Sum_s(n_s*T_s) over {len(cp['ion'])} ion species -- multiple "
            f"impurities or T_imp != Ti.")
    # 2. fast completeness
    avail_fast = 0.0
    for sp in [cp["electrons"]] + cp["ion"]:
        if "pressure_fast_perpendicular" in sp:
            avail_fast = max(avail_fast, float(np.max(np.abs(
                np.asarray(sp["pressure_fast_perpendicular"], dtype=float)))))
    if avail_fast > 0.0 and float(np.max(np.abs(p_fast))) <= 1e-3 * avail_fast:
        problems.append(
            f"fast pressure present in source (max {avail_fast/1e3:.1f} kPa) but "
            f"assembled p_fast ~ 0 -- fast-ion channel dropped.")
    # 3. backstop vs authoritative equilibrium pressure -- WARNING only, not a
    #    hard fail. The equilibrium.pressure routinely differs from the
    #    core_profiles thermal+fast reconstruction (anisotropic/beam pressure that
    #    is large early in a beam-heavy discharge, plus the
    #    equilibrium-vs-transport inconsistency -- the pressure analogue of the
    #    j_tor gap). When anchor_pressure_to_equilibrium is ON, p_diff absorbs the
    #    residual exactly and it rides fixed; when it is OFF (the default) nothing
    #    absorbs it and the gap propagates straight into the baseline pressure, so
    #    the message must not promise otherwise. Dropped *reconstructable*
    #    thermal/fast components are the hard fails (#1/#2 above).
    #
    #    This is also the one guard that can catch a wrong fast-pressure
    #    convention, whose signature is a ~3x (or ~1/3x) p_fast -- so report the
    #    SIGNED direction and name the rule in force.
    recon_total = recon_thermal + p_fast
    pe = float(np.mean(np.abs(p_equilibrium))) or 1.0
    signed = float(np.mean(recon_total - p_equilibrium)) / pe
    bk = float(np.mean(np.abs(recon_total - p_equilibrium))) / pe
    if bk > 0.05:
        import warnings
        direction = "above" if signed > 0 else "below"
        if anchor_pressure:
            remedy = ("anchor_pressure_to_equilibrium is ON, so the p_diff anchor "
                      "absorbs this exactly (the baseline still matches the dd "
                      "equilibrium.pressure) and the residual rides fixed -- it is "
                      "not perturbed in the UQ.")
        else:
            remedy = ("anchor_pressure_to_equilibrium is OFF (the default), so "
                      "p_diff is None and NOTHING absorbs this -- the gap "
                      "propagates into the baseline pressure and every downstream "
                      "beta_N / W_MHD / p'. Set "
                      "GenerationConfig.anchor_pressure_to_equilibrium=True to "
                      "anchor it.")
        meta = p_fast_meta or {}
        conv = ""
        if meta.get("rule") is not None:
            conv = (f" Fast-pressure reduction in force: {meta['rule']!r} "
                    f"(chosen by {meta.get('basis', '?')}). A wrong "
                    "pressure_fast_* convention shows up here as a ~3x / ~1/3x "
                    "p_fast -- check it before the anchor.")
        warnings.warn(
            f"reconstructed total pressure is {abs(signed)*100:.1f}% {direction} "
            f"dd equilibrium.pressure (mean |gap| {bk*100:.1f}%). {remedy}{conv}")
    if problems:
        msg = ("IMAS pressure completeness check failed:\n  - "
               + "\n  - ".join(problems)
               + "\nThe diff anchor still matches the baseline to FUSE, but the "
               "per-draw thermal perturbation omits the flagged component(s). "
               "Set GenerationConfig.allow_incomplete_pressure=True to bypass "
               "(backend tests only).")
        if allow_incomplete:
            import warnings
            warnings.warn(msg)
        else:
            raise ValueError(msg)


def _read_ida_omega(path, time_s, psi_N):
    """IDA toroidal rotation (omega_tor_12C6) resampled onto psi_N; None if absent.
    Mirrors read_ida's nearest-time selection (IDA time is ms; ``time_s`` is s)."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "omega_tor_12C6" not in f:
                return None
            it = np.asarray(f["time"], dtype=float)            # ms
            tms = (time_s * 1e3) if time_s is not None else float(it[0])
            j = int(np.argmin(np.abs(it - tms)))
            ipsi = np.asarray(f["psi_n"], dtype=float)
            om = np.asarray(f["omega_tor_12C6"], dtype=float)[j]
            return np.interp(psi_N, ipsi, om)
    except Exception:
        return None


#: Where the IDA-vs-dd ni cross-check is evaluated.  Five interior points: the
#: comparison is a POINTWISE relative one, so it must be taken where ni is a
#: real number.  Past ~0.8 both profiles roll off towards the separatrix and a
#: ratio there reports the edge model, not whether the two ni are the same
#: quantity.  The axis is included because that is where a beam population is
#: most peaked.  Also the core window :func:`_dd_zeff` classifies Z_eff on.
NI_FAST_GATE_PSI_N = (0.0, 0.2, 0.4, 0.6, 0.8)

#: Relative IDA-vs-dd TOTAL ni disagreement, at any of
#: :data:`NI_FAST_GATE_PSI_N`, above which the subtraction warns.  Advisory
#: only: the subtraction runs regardless.
NI_FAST_RTOL = 1e-2

#: psi_N(rho_tor_norm) of the dd's core_profiles grid vs the LCFS g-file's, at
#: these rho: FUSE holds replayed profiles fixed in rho while it solves its own
#: equilibrium, so anything placed by psi_N from elsewhere (an IDA fit on EFIT
#: psi_N) lands at a shifted radius against the dd's profiles and sources.
PSI_RHO_GATE_RHO = (0.2, 0.4, 0.6, 0.8, 0.9)

#: Largest |psi_N_dd - psi_N_g| at :data:`PSI_RHO_GATE_RHO` before the reader
#: warns.  Advisory only.
PSI_RHO_DRIFT_TOL = 2e-2


def _psi_rho_drift(psi_N, rho, gfile):
    """The dd's psi_N(rho) against the g-file's, at :data:`PSI_RHO_GATE_RHO`.

    Returns a dict: ``gate`` (rho -> psi_N_dd - psi_N_g), ``max_abs``,
    ``rho_worst``, ``exceeds`` and a one-line ``evidence``; ``None`` without a
    usable rho grid.
    """
    from .geqdsk import read_geqdsk
    rho = np.asarray(rho, dtype=float)
    if rho.shape != np.shape(psi_N) or not np.all(np.diff(rho) > 0):
        return None
    if np.allclose(rho, np.sqrt(np.clip(psi_N, 0.0, None)), rtol=0, atol=1e-6):
        # a placeholder grid, not the dd's equilibrium: nothing to compare
        return {"gate": None, "max_abs": None, "rho_worst": None, "gfile": str(gfile),
                "exceeds": False, "placeholder": True,
                "evidence": "dd rho_tor_norm is the sqrt(psi_N) placeholder; not compared"}
    g = read_geqdsk(gfile)
    rho_g = np.asarray(g.rhovn, dtype=float)
    psi_g = np.linspace(0.0, 1.0, rho_g.size)
    pts = np.asarray(PSI_RHO_GATE_RHO, dtype=float)
    d = np.interp(pts, rho, psi_N) - np.interp(pts, rho_g, psi_g)
    iw = int(np.argmax(np.abs(d)))
    out = {"gate": dict(zip(PSI_RHO_GATE_RHO, d.tolist())),
           "max_abs": float(abs(d[iw])), "rho_worst": float(pts[iw]),
           "gfile": str(gfile)}
    out["exceeds"] = out["max_abs"] > PSI_RHO_DRIFT_TOL
    out["placeholder"] = False
    out["evidence"] = (
        f"dd psi_N(rho) differs from the g-file's by {d[iw]:+.3f} at "
        f"rho={pts[iw]:g} (psi_N {np.interp(pts[iw], rho, psi_N):.3f} vs "
        f"{np.interp(pts[iw], rho_g, psi_g):.3f}; tol {PSI_RHO_DRIFT_TOL:g})")
    return out


#: Relative core mismatch above which a stored Z_eff matches neither numerator.
ZEFF_CONVENTION_RTOL = 1e-2


def _dd_zeff(cp, zeff_th, z2_fast, ne, psi_N):
    """``(zeff, includes_fast)``: the Z_eff the dd's own bootstrap consumed.

    IMAS resolves ``cp1d.zeff`` as stored data if present, else the expression
    ``IMAS.zeff`` = ``sum_s n_s Z_s^2 / ne`` over ``ion.density`` -- thermal
    PLUS fast.  FUSE's ``fast_particles_profiles!`` carves the beam out of
    ``density_thermal`` holding the total fixed, so a zeff stored before the
    beam arrived also has the fast ions in its numerator.  Neither is
    guaranteed, so a stored zeff is classified against both numerators on the
    core (``psi_N <= 0.8``, where carbon is fully stripped and IMAS's ``avgZ``
    equals ``z_n``).
    """
    if "zeff" not in cp:
        zeff_all = zeff_th + z2_fast / np.clip(ne, 1e-30, None)
        return zeff_all, bool(np.any(z2_fast))
    stored = np.asarray(cp["zeff"], dtype=float)
    if not np.any(z2_fast):
        return stored, False
    zeff_all = zeff_th + z2_fast / np.clip(ne, 1e-30, None)
    core = np.asarray(psi_N, dtype=float) <= NI_FAST_GATE_PSI_N[-1]
    _s = np.abs(stored[core])
    d_th = float(np.max(np.abs(stored - zeff_th)[core] / _s))
    d_all = float(np.max(np.abs(stored - zeff_all)[core] / _s))
    includes = d_all < d_th
    if min(d_th, d_all) > ZEFF_CONVENTION_RTOL:
        import warnings
        warnings.warn(
            f"core_profiles.zeff matches neither Z_eff numerator on the core "
            f"(thermal-only {d_th:.1e}, thermal+fast {d_all:.1e}); taking the "
            f"closer ({'thermal+fast' if includes else 'thermal-only'}). A zeff "
            f"stored before the dd's ne or ion densities were last changed "
            f"does this.")
    return stored, includes


def _subtract_fast_ni(psi_N, ni, sigma_ni, ni_fuse_thermal, z_fast, z2_fast,
                      impurity_Z):
    """Thermal ``(ni, sigma_ni)`` from a MEASURED ni, plus a provenance dict.

    A measured ni is a TOTAL deuteron density: neither the VB Z_eff nor the
    CER carbon can tell a beam ion from a thermal one.  Everything downstream
    of the reader (bootstrap, impurity pressure, archive, p-file ``nb``) takes
    ni as THERMAL whenever the dd carries a fast population, so the
    subtraction is unconditional; leaving ni total would mis-label it.

    The density removed is :func:`fast_ion_density_equivalent` of the charge
    moments ``z_fast``/``z2_fast``, so no beam charge is assumed.

    ``sigma_ni`` is returned unchanged, keeping the ABSOLUTE error: the fast
    density removed is a FUSE quantity carrying no IDA error, so subtracting
    it leaves the measurement's error as it was.  This is also the spread of
    an ni derived per draw from the drawn (ne, Z_eff).

    Cross-check (advisory): the IDA ni against the dd's TOTAL ni at each of
    :data:`NI_FAST_GATE_PSI_N`.  They agree when FUSE was built from the same
    IDA fits.  A disagreement beyond :data:`NI_FAST_RTOL` warns -- the beam
    density then comes from a plasma that is not quite the IDA one -- but
    does not stop the subtraction.

    Returns ``(ni, sigma_ni, meta)``; ``meta["applied"]`` says whether it
    fired, ``meta["agrees"]`` the cross-check verdict, ``meta["gate"]`` the
    per-point relative deviations, ``meta["evidence"]`` a one-line reason.
    """
    ni = np.asarray(ni, dtype=float)
    # What the MEASURED ni actually carries of the fast population: each fast
    # species enters it weighted Z_s(Z_imp - Z_s)/(Z_imp - 1), not as a bare
    # particle density.  Equals the fast density for a hydrogenic beam and
    # zero for a fast species at the impurity charge.
    ni_fast = fast_ion_density_equivalent(z_fast, z2_fast, impurity_Z)
    if not np.any(ni_fast):
        return ni, sigma_ni, {"applied": False, "agrees": None,
                              "mismatch": None, "gate": None,
                              "evidence": "dd carries no fast-ion population"}
    ni_total = np.asarray(ni_fuse_thermal, dtype=float) + ni_fast
    pts = np.asarray(NI_FAST_GATE_PSI_N, dtype=float)
    a = np.interp(pts, np.asarray(psi_N, dtype=float), ni)
    b = np.interp(pts, np.asarray(psi_N, dtype=float), ni_total)
    gate = mismatch = agrees = None
    if np.all(np.abs(b) > 0.0):
        gate = dict(zip(NI_FAST_GATE_PSI_N, (np.abs(a - b) / np.abs(b)).tolist()))
        mismatch = float(max(gate.values()))
        agrees = mismatch <= NI_FAST_RTOL
    ni_th = np.maximum(ni - ni_fast, 0.0)
    if agrees is None:
        evidence = ("dd main-ion density vanishes at a check point; "
                    "cross-check skipped")
    elif agrees:
        evidence = (f"IDA ni matches the dd TOTAL ni to {mismatch:.2e} over "
                    f"psi_N={NI_FAST_GATE_PSI_N}")
    else:
        worst = max(gate, key=gate.get)
        evidence = (f"IDA ni differs from the dd TOTAL ni by {mismatch:.2e} "
                    f"at psi_N={worst:g} (> {NI_FAST_RTOL:.0e}): the beam "
                    f"density comes from a plasma that is not the IDA one")
    return ni_th, sigma_ni, {
        "applied": True, "agrees": agrees, "mismatch": mismatch, "gate": gate,
        "fast_fraction_peak": float(np.max(ni_fast / np.maximum(ni, 1e-30))),
        "evidence": evidence}


def _merge_ida_kinetics(psi_N, ne_fuse, ni_fuse, Zeff_fuse, ida_path, time, impurity_Z,
                         ni_source="all", zeff_from_fuse=False,
                         z_fast=None, z2_fast=None):
    """IDA-hybrid kinetics: replace FUSE ne/ni/Te/Ti/Zeff (+omega) with IDA fits,
    resampled onto the FUSE ``psi_N`` grid (psi_N == psi_N_kinetic).

    Zeff and ni both default to IDA: Zeff measured directly (``zeff_from_fuse=True``
    keeps the FUSE Zeff instead); ni via ``ni_source`` ("Zeff"/"CER"/"all", with
    Jacobian-propagated sigma_ni -- see :func:`bouquet.io.ida.read_ida`). The
    "Zeff"/"all" dilution always uses IDA's own Zeff regardless of
    ``zeff_from_fuse``.

    With the fast charge moments ``z_fast``/``z2_fast`` the IDA TOTAL ni is
    always converted to a THERMAL one; see :func:`_subtract_fast_ni`.

    Returns ``(ne, te, ti, ni, zeff, omega_or_None, sigma_ne, sigma_te, sigma_ni,
    sigma_ti, ida, ni_fast_meta)`` on ``psi_N`` -- the ``IDAProfiles`` is the read
    itself, handed back so ``resolve_uncertainty`` can reuse it instead of
    re-reading the file (see ``Baseline.aux['ida_profiles']``). Re-reading
    risked a DIFFERENT slice: this path resolves ``time`` against the IMAS
    slice, while the envelope path only had the requested ``source.time``.
    """
    from .ida import read_ida
    ida = read_ida(ida_path, time=time, impurity_Z=impurity_Z, ni_source=ni_source)
    _ipsi = np.asarray(ida.psi_N, dtype=float)
    g = lambda a: np.interp(psi_N, _ipsi, np.asarray(a, dtype=float))
    ne, ni, te, ti = g(ida.ne), g(ida.ni), g(ida.te), g(ida.ti)
    zeff = np.asarray(Zeff_fuse, dtype=float) if zeff_from_fuse else g(ida.Zeff)
    sigma_ne, sigma_te, sigma_ni, sigma_ti = (
        g(ida.sigma_ne), g(ida.sigma_te), g(ida.sigma_ni), g(ida.sigma_ti))
    # The ni_source ni is a TOTAL deuteron density; everything downstream
    # takes ni as thermal once the dd carries a beam.
    if z_fast is not None:
        ni, sigma_ni, ni_fast_meta = _subtract_fast_ni(
            psi_N, ni, sigma_ni, np.asarray(ni_fuse, dtype=float),
            z_fast, z2_fast, impurity_Z)
    else:
        ni_fast_meta = {"applied": False, "agrees": None, "mismatch": None,
                        "gate": None,
                        "evidence": "no fast-ion charge moments supplied"}
    omega = _read_ida_omega(ida_path, time, psi_N)
    return (ne, te, ti, ni, zeff, omega, sigma_ne, sigma_te, sigma_ni,
            sigma_ti, ida, ni_fast_meta)


def read_imas_baseline(
    source: "ImasSource",
    fixed: Optional["FixedComponentsConfig"] = None,
    p_fast_reduction: str = "auto",
    allow_incomplete_pressure: bool = False,
    anchor_jtor_to_equilibrium: bool = True,
    kinetic_source: str = "fuse",
    anchor_pressure_to_equilibrium: bool = False,
) -> "Baseline":
    """Read a FUSE ``dd_sim.json`` IDS and return a separated :class:`Baseline`.

    No Grad-Shafranov reconstruction is performed -- provenance is "imas".

    ``p_fast_reduction`` defaults to ``"auto"``: the fast-pressure storage
    convention is taken from the dd's own recorded provenance
    (:func:`resolve_p_fast_reduction`), with a loud warning if it cannot be
    determined.  An explicit ``"sum"`` / ``"trace"`` / ``"mean"`` / ``"perp"``
    always wins and is applied silently.  The rule that was used, and how it was
    chosen, are recorded on :attr:`Baseline.p_fast_meta`.
    """
    import json
    from ..baseline import Baseline

    with open(source.ids_path, "rb") as fh:
        raw_bytes = fh.read()
    dd = json.loads(raw_bytes)
    T = source.time

    # Which fast-pressure convention this dd was written in (factor of 3).
    # The metadata is resolved eagerly -- the reduction rule is needed to read the
    # fields and the completeness message quotes it -- but the loud undetermined
    # warning is held back until we know the choice actually moved a number: it
    # says nothing on a dd whose fast pressure is absent or identically zero, and
    # nothing on a run whose p_fast comes from FixedComponentsConfig instead.
    p_fast_meta = resolve_p_fast_reduction(dd, p_fast_reduction, warn=False)
    p_fast_rule = p_fast_meta["rule"]

    # --- targets from the equilibrium IDS ---
    eq = dd["equilibrium"]
    ie = _nearest_index(eq["time"], T, "equilibrium")
    gq = eq["time_slice"][ie]["global_quantities"]
    Ip_target = abs(float(gq["ip"]))
    ids_li_1 = float(gq["li_1"]) if "li_1" in gq else None
    ids_li_3 = float(gq["li_3"]) if "li_3" in gq else None
    # Provisional l_i_target = IDS li_3; the IMAS forward-solve (Bouquet) replaces
    # this with the TokaMaker-solved li_1 and records the IDS values for sanity.
    l_i_target = ids_li_3 if ids_li_3 is not None else 0.0

    # --- profiles + currents from core_profiles ---
    cp_ids = dd["core_profiles"]
    ic = _nearest_index(cp_ids["time"], T, "core_profiles")
    cp = cp_ids["profiles_1d"][ic]

    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psi_N = (psi - psi[0]) / (psi[-1] - psi[0])   # 0 (axis) -> 1 (boundary)
    n = psi_N.size

    j_total = np.asarray(cp["j_total"], dtype=float)   # total parallel
    j_tor = np.asarray(cp["j_tor"], dtype=float)       # total toroidal (authoritative)
    j_ohmic = np.asarray(cp["j_ohmic"], dtype=float)   # parallel (unused: inductive = residual)
    j_boot = np.asarray(cp["j_bootstrap"], dtype=float)  # parallel

    def to_toroidal(j_par):
        return parallel_to_toroidal(j_par, j_parallel_total=j_total, j_tor_total=j_tor)

    j_BS = to_toroidal(j_boot)

    # --- NBI: sum beam-source parallel currents, then convert ---
    src_ids = dd.get("core_sources", {})
    isrc = _nearest_index(src_ids["time"], T, "core_sources") if src_ids.get("time") else ic
    jnbi_par = np.zeros(n)
    for s in src_ids.get("source", []):
        if s.get("identifier", {}).get("index") == NBI_SOURCE_INDEX:
            pr = s.get("profiles_1d", [])
            if pr:
                idx = isrc if len(pr) > isrc else 0
                jnbi_par = jnbi_par + np.asarray(pr[idx]["j_parallel"], dtype=float)
    j_NBI = to_toroidal(jnbi_par)
    j_RF = np.zeros(n)   # never computed internally; user-supplied only

    # --- sawtooth model presence/amplitude at this slice (gate input only) ----
    # Read here because the dd (100s of MB) is not retained past this function.
    # "active" means the source EXISTS and carries a non-zero j_parallel at this
    # time index: a declared-but-idle sawtooth source (all zeros before onset)
    # must NOT admit a ramp slice to the q0 pin.
    sawtooth = {"source_index": SAWTOOTH_SOURCE_INDEX, "present": False,
                "j_par_max_abs": 0.0, "active": False, "q0_dd": None}
    for s in src_ids.get("source", []):
        if s.get("identifier", {}).get("index") == SAWTOOTH_SOURCE_INDEX:
            sawtooth["present"] = True
            pr = s.get("profiles_1d", [])
            if pr:
                jsaw = np.asarray(pr[isrc if len(pr) > isrc else 0]
                                  .get("j_parallel", []), dtype=float)
                if jsaw.size and np.any(np.isfinite(jsaw)):
                    sawtooth["j_par_max_abs"] = max(
                        sawtooth["j_par_max_abs"],
                        float(np.nanmax(np.abs(jsaw))))
    sawtooth["active"] = bool(sawtooth["present"]
                              and sawtooth["j_par_max_abs"] > 0.0)

    # --- kinetic profiles + Zeff + fast pressure ---
    el = cp["electrons"]
    ne = np.asarray(el["density_thermal"], dtype=float)
    te = np.asarray(el["temperature"], dtype=float)   # eV
    _no_par = []
    p_fast = _isotropic_fast_pressure(el, p_fast_rule, n, _no_par, "electrons")

    ni = None
    ti = None
    main_ion = None
    zeff_num = np.zeros(n)
    # The two charge moments of the fast population.  Quasineutrality weights
    # each fast species by Z_s, a measured Z_eff's numerator by Z_s^2, so both
    # are needed and neither implies the other (see physics.
    # fast_ion_density_equivalent).  No beam charge is assumed anywhere.
    z_fast = np.zeros(n)          # sum_s Z_s   n_s^fast  [m^-3]
    z2_fast = np.zeros(n)         # sum_s Z_s^2 n_s^fast  [m^-3]
    # pressure_fast_* and density_fast are independent fields: a dd can carry
    # one without the other, and a species with fast pressure but no fast
    # density gets the full p_fast treatment and ZERO dilution correction.
    fast_p_no_n = []
    for ion in cp["ion"]:
        Z = float(ion["element"][0]["z_n"])
        n_s = np.asarray(ion["density_thermal"], dtype=float)
        zeff_num += n_s * Z * Z
        if "density_fast" in ion:
            _nf = np.asarray(ion["density_fast"], dtype=float)
            z_fast += Z * _nf
            z2_fast += Z * Z * _nf
        p_fast_s = _isotropic_fast_pressure(
            ion, p_fast_rule, n, _no_par, str(ion.get("label", f"Z={Z:g}")))
        if np.any(p_fast_s) and not np.any(
                np.asarray(ion.get("density_fast", 0.0), dtype=float)):
            fast_p_no_n.append(f"Z={Z:g}")
        p_fast = p_fast + p_fast_s
        if Z == 1.0 and ni is None:        # main (hydrogenic) ion
            ni = n_s
            ti = np.asarray(ion["temperature"], dtype=float)
            main_ion = ion
    if ni is None:
        raise ValueError("no hydrogenic (Z=1) main ion found in core_profiles.ion")
    _warn_missing_parallel(_no_par, p_fast_rule)
    if fast_p_no_n:
        # Silence is the dangerous case here: the run looks exactly like an
        # ohmic one while Z_imp / nz / p_imp keep the whole fast-ion bias the
        # dilution correction exists to remove.
        import warnings
        warnings.warn(
            "core_profiles carries fast-ion PRESSURE but no density_fast for "
            f"ion species [{', '.join(fast_p_no_n)}]: p_fast is applied in "
            "full while the fast-ion dilution correction for those species is "
            "zero, so Z_imp / nz / p_imp retain the fast-ion bias. Fill "
            "core_profiles.ion[].density_fast to enable the correction.")
    # Thermal-only numerator over the full ne: the convention
    # impurity_charge_with_fast_ions inverts, built here from the dd's own
    # densities so it holds by construction.
    Zeff_th = zeff_num / ne
    # The dd's bootstrap Z_eff and its convention (see _dd_zeff).  Zeff is its
    # recomputation from the densities -- bit-identical to Zeff_th without a
    # beam -- and is what the baseline solve and the draws are handed.
    zeff_dd, dd_zeff_includes_fast = _dd_zeff(cp, Zeff_th, z2_fast, ne, psi_N)
    Zeff = (Zeff_th + z2_fast / np.clip(ne, 1e-30, None)
            if dd_zeff_includes_fast else Zeff_th)

    # --- auxiliary source-provided profiles for the switchboard ---------------
    # Read whatever this source carries (production FUSE files have rotation;
    # chi/E_r are typically absent and supplied via aux_baselines). All on the
    # core_profiles grid (== psi_N_kinetic for IMAS).
    aux = {"zeff": zeff_dd}
    if main_ion is not None and "rotation_frequency_tor" in main_ion:
        aux["omega_tor"] = np.asarray(main_ion["rotation_frequency_tor"], dtype=float)
    if "e_field" in cp and "radial" in cp["e_field"]:
        aux["e_r"] = np.asarray(cp["e_field"]["radial"], dtype=float)
    ctids = dd.get("core_transport")
    if ctids and ctids.get("model"):
        cpr = ctids["model"][0].get("profiles_1d", [])
        ict = (_nearest_index(ctids["time"], T, "core_transport")
               if ctids.get("time") else ic)
        if cpr:
            ctsl = cpr[ict if len(cpr) > ict else 0]
            ce = ctsl.get("electrons", {}).get("energy", {})
            if "d" in ce:
                aux["chi_e"] = np.asarray(ce["d"], dtype=float)
            if "d" in ctsl.get("total_ion_energy", {}):
                aux["chi_i"] = np.asarray(ctsl["total_ion_energy"]["d"], dtype=float)

    # --- IDA-hybrid: swap FUSE ne/Te/Ti/Zeff/omega for externally-fit IDA profiles ---
    # ni via source.ni_source; Zeff from IDA unless source.zeff_from_fuse. Done
    # before the pressure block so p_recon/Z_imp/p_imp use the IDA kinetics. IDA
    # sigmas land in aux as sigma_*_ida (informational -- resolve_uncertainty still
    # needs UncertaintyConfig.ida_path for the actual generation envelope).
    use_ida = bool(kinetic_source == "ida_hybrid" and getattr(source, "ida_path", None))
    zeff_includes_fast = dd_zeff_includes_fast
    # Geometry guard: does the dd place its profiles where the g-file does?
    _drift = None
    if getattr(source, "LCFS_geqdsk", None) and "rho_tor_norm" in cp["grid"]:
        _drift = _psi_rho_drift(psi_N, cp["grid"]["rho_tor_norm"], source.LCFS_geqdsk)
        aux["psi_rho_drift"] = _drift
        if _drift is not None and _drift["exceeds"]:
            import warnings
            warnings.warn(
                f"{_drift['evidence']}: the dd's equilibrium is not the g-file's, "
                "so its profiles and sources sit at shifted psi_N"
                + (" against the IDA kinetics (placed by IDA psi_N)" if use_ida else ""))
    if use_ida:
        (ne, te, ti, ni, Zeff, _omega,
         sigma_ne_ida, sigma_te_ida, sigma_ni_ida, sigma_ti_ida,
         _ida_read, _ni_fast_meta) = _merge_ida_kinetics(
            psi_N, ne, ni, Zeff, source.ida_path, T,
            getattr(source, "impurity_Z", 6.0),
            ni_source=getattr(source, "ni_source", "all"),
            zeff_from_fuse=getattr(source, "zeff_from_fuse", False),
            z_fast=z_fast, z2_fast=z2_fast)
        if _ni_fast_meta["agrees"] is False and _drift is not None and _drift["exceeds"]:
            _ni_fast_meta["evidence"] += (
                f"; likely the psi_N(rho) drift ({_drift['evidence']})")
        aux["ni_fast_meta"] = _ni_fast_meta
        # Loud: the subtraction moves ni, and a failed cross-check means the
        # beam density belongs to a plasma that is not quite the IDA one.
        if _ni_fast_meta["applied"]:
            print(f"  [ni] thermal ni: subtracted the dd fast-ion equivalent "
                  f"(peak fast fraction "
                  f"{_ni_fast_meta['fast_fraction_peak']:.1%}); "
                  f"{_ni_fast_meta['evidence']}")
            if _ni_fast_meta["agrees"] is False:
                import warnings
                warnings.warn(f"ida_hybrid: {_ni_fast_meta['evidence']}; "
                              f"subtracted anyway")
        if _omega is not None:
            aux["omega_tor"] = _omega
        aux["zeff"] = Zeff   # keep the switchboard's zeff baseline consistent
        # IDA's Z_eff is MEASURED, so its numerator counts the fast ions;
        # zeff_from_fuse carries the dd's own convention over with its value.
        if not getattr(source, "zeff_from_fuse", False):
            zeff_includes_fast = True
        # Read once, shared: resolve_uncertainty reuses this instead of
        # opening the same file again (and possibly at another slice).
        aux["ida_profiles"] = (str(source.ida_path), _ida_read)
        aux["sigma_ne_ida"] = sigma_ne_ida
        aux["sigma_te_ida"] = sigma_te_ida
        aux["sigma_ni_ida"] = sigma_ni_ida
        aux["sigma_ti_ida"] = sigma_ti_ida

    # --- user overrides for fixed additive components ---
    if fixed is not None:
        if fixed.p_fast is not None:
            p_fast = _override(fixed.p_fast, fixed.psi_N, psi_N)
            p_fast_meta = {**p_fast_meta, "rule": None, "basis": "user-override",
                           "evidence": "FixedComponentsConfig.p_fast supplied; the "
                                       "dd fast-pressure fields were not read"}
        if fixed.j_NBI is not None:
            j_NBI = _override(fixed.j_NBI, fixed.psi_N, psi_N)
        if fixed.j_RF is not None:
            j_RF = _override(fixed.j_RF, fixed.psi_N, psi_N)

    # The deferred factor-of-3 warning: the convention was undeterminable AND the
    # fast pressure it scales is non-zero AND it came from the dd (a user-supplied
    # p_fast has already rewritten the basis to "user-override").
    if (p_fast_meta["basis"] == "undetermined-fallback"
            and float(np.max(np.abs(np.asarray(p_fast, dtype=float)))) > 0.0):
        warn_p_fast_undetermined(p_fast_meta["rule"])
        p_fast_meta = {**p_fast_meta, "warned": True}

    # Authoritative toroidal total; inductive absorbs the residual so the
    # decomposition sums exactly and Ip is preserved.
    j_phi = j_tor.copy()
    j_inductive = j_phi - j_BS - j_NBI - j_RF

    # --- pressure anchor ("diff" approach) + completeness validation ----------
    # The authoritative dd equilibrium pressure (GS-consistent total, incl.
    # thermal impurity + isotropized fast) interpolated onto psi_N. The solve
    # builds e*(ne*Te + ni*Ti) + impurity(nz*Ti) + p_fast; p_diff anchors that to
    # FUSE exactly (mirrors jBS_diff) so the per-draw kinetics perturb the thermal
    # while the carbon/fast/GS residual rides as a fixed offset. Z_imp is the
    # single effective impurity charge (one-Zeff single-impurity model).
    eqp1 = eq["time_slice"][ie]["profiles_1d"]
    psi_eq = np.asarray(eqp1["psi"], dtype=float)
    psiN_eq = (psi_eq - psi_eq[0]) / (psi_eq[-1] - psi_eq[0])
    _o = np.argsort(psiN_eq)
    p_equilibrium = np.interp(psi_N, psiN_eq[_o],
                              np.asarray(eqp1["pressure"], dtype=float)[_o])
    # The dd's OWN axis q -- taken at the SMALLEST psi_N (via the same ordering
    # the pressure uses), not blindly at index 0, since profiles_1d need not be
    # stored axis-first.  Comparison metric only; see Baseline.sawtooth.
    if "q" in eqp1:
        _qdd = np.asarray(eqp1["q"], dtype=float)[_o]
        if _qdd.size and np.isfinite(_qdd[0]):
            sawtooth["q0_dd"] = float(_qdd[0])
    # Only ne - sum_s Z_s n_s^fast is neutralised by THERMAL ions; charging
    # the fast-ion share to the impurity inflates nz.  The helper also
    # renormalizes Zeff (defined over the FULL ne above) onto the thermal
    # electrons -- without that the inversion recovers only half the bias
    # (see impurity_charge_with_fast_ions).  The Zeff consumed by the
    # bootstrap / forward solve deliberately stays the full-ne one.
    # (Zeff_th: the helper's thermal-numerator convention.  On the ida path
    # only ne_th is used, which does not depend on Z_eff.)
    _Z_inverted, ne_th = impurity_charge_with_fast_ions(
        ne, ni, Zeff if use_ida else Zeff_th, z_fast)
    # With IDA-hybrid kinetics, ni was built (read_ida)
    # under single-impurity quasineutrality at charge source.impurity_Z, so that IS
    # the impurity charge; the inversion is then only needed for ne_th.
    Z_imp = (float(getattr(source, "impurity_Z", 6.0)) if use_ida
             else _Z_inverted)
    p_imp = impurity_pressure(ne_th, ni, ti, Z_imp)
    p_recon = _EC * (ne * te + ni * ti) + p_imp + p_fast
    # p_diff anchors the solve thermal pressure to the FUSE equilibrium.pressure.
    # Gated OFF by default: with IDA-hybrid kinetics we trust IDA's pressure and do
    # NOT force it back onto the FUSE total. When None, no anchor offset is added.
    p_diff = (p_equilibrium - p_recon) if anchor_pressure_to_equilibrium else None
    # The completeness guard compares the FUSE thermal/fast pressure against the FUSE
    # equilibrium.pressure; it is meaningless once ne/Te/Ti are swapped to IDA.
    if kinetic_source != "ida_hybrid":
        _validate_pressure_completeness(cp, ne, te, ni, ti, p_fast, p_imp,
                                        p_equilibrium, allow_incomplete_pressure,
                                        anchor_pressure=anchor_pressure_to_equilibrium,
                                        p_fast_meta=p_fast_meta)

    # --- total-current anchor: equilibrium.j_tor vs core_profiles.j_tor --------
    # core_profiles.j_tor (== j_phi here) is the transport parallel-current sum
    # (QED-diffused ohmic + Sauter bootstrap + NBI) converted to toroidal; it
    # differs from the GS-consistent equilibrium.j_tor (which GPEC reads) from
    # ~q=2 outward (the equilibrium carries more pedestal current). jphi_diff
    # anchors the total to the equilibrium as a FIXED offset (mirrors jBS_diff):
    # added to the baseline + every draw while the SWB bootstrap / perturbed
    # j_inductive ride underneath. jphi_diff integrates to ~0 (both totals carry
    # the same Ip), so it redistributes rather than adds net current.
    jphi_diff = None
    if anchor_jtor_to_equilibrium:
        eq_jtor = np.interp(psi_N, psiN_eq[_o],
                            np.asarray(eqp1["j_tor"], dtype=float)[_o])
        jphi_diff = eq_jtor - j_phi

    return Baseline(
        psi_N=psi_N,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="imas",
        j_NBI=j_NBI,
        j_RF=j_RF,
        p_fast=p_fast,
        z_fast=(z_fast if np.any(z_fast) else None),
        z2_fast=(z2_fast if np.any(z_fast) else None),
        zeff_includes_fast=zeff_includes_fast,
        p_equilibrium=p_equilibrium,
        p_diff=p_diff,
        Z_imp=Z_imp,
        jphi_diff=jphi_diff,
        eqdsk_bytes=None,
        # The OMAS JSON is NOT an Osborne p-file; don't pass it as pfile_bytes
        # (generate_bouquet would try to parse it). Per-draw p-files are built
        # from the kinetic profiles.
        pfile_bytes=None,
        li_metrics={"ids_li_1": ids_li_1, "ids_li_3": ids_li_3},
        aux=aux,
        p_fast_meta=p_fast_meta,
        sawtooth=sawtooth,
    )


# ===========================================================================
#  Perturbed-draw IMAS/OMAS write-back
#
#  Current-split fidelity: the parallel split (j_ohmic/j_bootstrap/j_total) is
#  now EXACT per draw (fidelity="exact"/"auto") -- it uses the draw's OWN
#  flux-surface geometry, captured from the live TokaMaker equilibrium at
#  generate time (physics.capture_equilibrium_fsa -> the eq_fsa archive block)
#  and applied via physics.toroidal_to_parallel. The legacy baseline-ratio
#  c(psi)=j_tor/j_total (fidelity="reconstruct") is kept as a fallback for
#  archives written without capture. j_tor is exact either way.
#
#  Remaining refinements (not blockers): (1) the EQUILIBRIUM IDS profiles_2d
#  still come from the archived 257^2 eqdsk (lossless to that grid,
#  machine-precision GS) rather than the live FE fields -- a direct OFT ODS
#  export would upgrade this; (2) exact <1/R^2> is computed by flux-surface
#  quadrature since TokaMaker does not yet expose it
#  (OpenFUSIONToolkit/OpenFUSIONToolkit#312) -- when it does, read it directly.
# ===========================================================================
def _imas_b0(out, ie, ic):
    """Reference vacuum field B0 for the IMAS <j.B>/B0 normalisation.

    Tries core_profiles then equilibrium ``vacuum_toroidal_field.b0`` (nearest
    time index); falls back to 1.0 (raw <j.B>) if neither is present.
    """
    for ids_name, it in (("core_profiles", ic), ("equilibrium", ie)):
        vtf = out.get(ids_name, {}).get("vacuum_toroidal_field")
        if vtf and vtf.get("b0") is not None:
            b0 = np.asarray(vtf["b0"], dtype=float)
            if b0.size:
                return abs(float(b0[min(it, b0.size - 1)]))
    return 1.0


def _eq_fsa_geom_on(eq_fsa, psiN_t, B0):
    """Interpolate a captured eq_fsa block onto the template psi grid -> geom
    dict for :func:`bouquet.physics.toroidal_to_parallel`. ``None`` if the
    block lacks the required FSA metrics (caller then reconstructs)."""
    try:
        src = np.asarray(eq_fsa["psi_N"], dtype=float)
        F = np.interp(psiN_t, src, np.asarray(eq_fsa["F"], dtype=float))
        avg_inv_R = np.interp(psiN_t, src, np.asarray(eq_fsa["avg_inv_R"], dtype=float))
        avg_B2 = np.interp(psiN_t, src, np.asarray(eq_fsa["avg_B2"], dtype=float))
    except (KeyError, TypeError):
        return None
    geom = {"F": F, "avg_inv_R": avg_inv_R, "avg_B2": avg_B2, "B0": float(B0)}
    if eq_fsa.get("avg_inv_R2") is not None:     # exact bracket when captured
        geom["avg_inv_R2"] = np.interp(
            psiN_t, src, np.asarray(eq_fsa["avg_inv_R2"], dtype=float))
    return geom


def write_imas_draw(h5path_or_header, draw_index, template_ids_path, out_path,
                    scan_key=None, time=None, fidelity="auto"):
    """Reconstruct a perturbed IMAS/OMAS IDS for one draw from the bouquet HDF5.

    Maps the draw's archived eqdsk to the ``equilibrium`` IDS
    (``profiles_1d`` / ``profiles_2d`` / ``global_quantities`` / ``boundary`` --
    lossless to the eqdsk grid, machine-precision GS) and the draw's ``.h5``
    kinetics/currents to ``core_profiles``. ``j_tor`` is exact.

    The parallel split (``j_total`` / ``j_ohmic`` / ``j_bootstrap`` =
    IMAS ``<j.B>/B0``) fidelity is set by ``fidelity``:

      * ``"exact"``       -- convert each toroidal component with the draw's OWN
        captured flux-surface geometry (``eq_fsa`` block, from
        ``capture_live_eq=True`` at generate time) via
        :func:`bouquet.physics.toroidal_to_parallel`. Raises if the block is
        absent.
      * ``"reconstruct"`` -- the interim baseline ratio ``c = j_tor/j_total``
        from the template (exact only when the draw's flux geometry matches the
        baseline's).
      * ``"auto"`` (default) -- exact when the ``eq_fsa`` block is present,
        else reconstruct.

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.
    draw_index : int
        Per-draw index within the scan group.
    template_ids_path : str
        The source IMAS/OMAS JSON, used as the structural template.
    out_path : str
        Where to write the perturbed IDS JSON.
    fidelity : {"auto", "exact", "reconstruct"}
        Parallel-current split fidelity (see above); default ``"auto"``.
    scan_key : str, float, or None
        Scan value selecting the ``scan/<scan_key>`` group.  The default
        ``None`` auto-resolves a single-scan archive (and is the flat layout
        for flat files); pass the explicit value only when the archive holds
        more than one scan.
    time : float, optional
        Time slice [s]; defaults to the template's nearest single slice.

    Returns
    -------
    str
        ``out_path``.
    """
    import json
    import copy
    import h5py
    from .geqdsk import read_geqdsk
    from ..utils import read_eqdsk_from_bytes, _group_path, _resolve_h5

    if fidelity not in ("auto", "exact", "reconstruct"):
        raise ValueError(
            f"fidelity must be 'auto'|'exact'|'reconstruct', got {fidelity!r}")

    with open(template_ids_path) as fh:
        out = json.load(fh)

    eq_ids = out["equilibrium"]
    ie = _nearest_index(eq_ids["time"], time, "equilibrium")
    cp_ids = out["core_profiles"]
    ic = _nearest_index(cp_ids["time"], time, "core_profiles")

    h5 = _resolve_h5(h5path_or_header)
    if scan_key is None:
        # Convenience: a single-scan archive resolves unambiguously, so the
        # caller need not echo the generation scan_key.  Flat-layout files
        # (discover_scan_keys -> None) keep scan_key=None.
        from ..utils import discover_scan_keys
        keys = discover_scan_keys(h5)
        if keys:
            if len(keys) != 1:
                raise ValueError(
                    f"{h5} holds {len(keys)} scans {keys}; pass an explicit "
                    "scan_key to write_imas_draw().")
            scan_key = keys[0]
    gp = _group_path(scan_key, draw_index)
    with h5py.File(h5, "r") as hf:
        if gp not in hf:
            raise KeyError(f"draw {draw_index} (scan {scan_key}) not in {h5}")
        g = hf[gp]
        ne = np.asarray(g["n_e"][()]); te = np.asarray(g["T_e"][()])
        ni = np.asarray(g["n_i"][()]); ti = np.asarray(g["T_i"][()])
        pkin = np.asarray(g["psi_N_kinetic"][()]); peq = np.asarray(g["psi_N"][()])
        j_tor = np.asarray(g["j_phi"][()])
        j_ind = np.asarray(g["j_inductive"][()])
        j_bs = np.asarray(g["j_BS"][()])
        zeff = np.asarray(g["aux_zeff"][()]) if "aux_zeff" in g else None
        li1 = float(g.attrs.get("l_i(1)", np.nan))
        li3 = float(g.attrs.get("l_i(3)", np.nan))
        from ..schema import find_bytes_dataset, EQ_FSA_GROUP
        eqk = find_bytes_dataset(g)
        if eqk is None:
            raise KeyError(f"draw {draw_index} has no archived eqdsk")
        geq = read_eqdsk_from_bytes(bytes(g[eqk][()]), read_geqdsk)
        # optional captured live-equilibrium FSA block (exact conversion)
        eq_fsa = None
        if EQ_FSA_GROUP in g:
            eq_fsa = {k: np.asarray(g[EQ_FSA_GROUP][k][()], dtype=float)
                      for k in g[EQ_FSA_GROUP]}

    # --- equilibrium IDS from the eqdsk (lossless to the eqdsk grid) ---------
    ts = eq_ids["time_slice"][ie]
    psi1d = geq.psi_axis + geq.psi_N * (geq.psi_boundary - geq.psi_axis)
    q95 = float(np.interp(0.95, geq.psi_N, geq.qpsi))
    ts["profiles_1d"] = {
        "psi": psi1d.tolist(),
        "q": geq.qpsi.tolist(),
        "pressure": geq.pres.tolist(),
        "f": geq.fpol.tolist(),
        "dpressure_dpsi": geq.pprime.tolist(),
        "f_df_dpsi": geq.ffprim.tolist(),
    }
    ts["profiles_2d"] = [{
        "grid_type": {"name": "rectangular", "index": 1},
        "grid": {"dim1": geq.R_grid.tolist(), "dim2": geq.Z_grid.tolist()},
        # psi_RZ is indexed [R][Z], matching IMAS dim1=R, dim2=Z
        "psi": geq.psi_RZ.tolist(),
    }]
    gq = dict(ts.get("global_quantities", {}))
    gq.update(
        ip=float(geq.Ip), psi_axis=float(geq.psi_axis),
        psi_boundary=float(geq.psi_boundary),
        magnetic_axis={"r": float(geq.R_mag), "z": float(geq.Z_mag)},
        q_axis=float(geq.qpsi[0]), q_95=q95,
        li_3=li3 if np.isfinite(li3) else geq.li.get("li(3)"),
        beta_normal=geq.betas.get("beta_n"), beta_pol=geq.betas.get("beta_p"),
        beta_tor=geq.betas.get("beta_t"),
    )
    if np.isfinite(li1):
        gq["li_1"] = li1
    ts["global_quantities"] = gq
    ts["boundary"] = {"outline": {"r": geq.boundary_R.tolist(),
                                  "z": geq.boundary_Z.tolist()}}

    # --- core_profiles from the draw (kinetics + currents) ------------------
    cp = cp_ids["profiles_1d"][ic]
    psi = np.asarray(cp["grid"]["psi"], dtype=float)
    psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])

    def to_t(arr, src):     # interp draw array (on src grid) -> template psi grid
        return np.interp(psiN_t, src, arr)

    cp["electrons"]["density_thermal"] = to_t(ne, pkin).tolist()
    cp["electrons"]["temperature"] = to_t(te, pkin).tolist()
    for ion in cp["ion"]:
        if float(ion["element"][0]["z_n"]) == 1.0:
            ion["density_thermal"] = to_t(ni, pkin).tolist()
            ion["temperature"] = to_t(ti, pkin).tolist()
            break
    if zeff is not None:
        cp["zeff"] = to_t(zeff, pkin).tolist()

    # j_tor is exact (bouquet stores toroidal current directly).
    jt_t = to_t(j_tor, peq)
    cp["j_tor"] = jt_t.tolist()

    # Parallel split (j_total / j_ohmic / j_bootstrap = IMAS <j.B>/B0). Two
    # fidelities (`fidelity` arg): EXACT uses the draw's own captured
    # flux-surface geometry (eq_fsa) via physics.toroidal_to_parallel;
    # RECONSTRUCT falls back to the interim baseline ratio c=j_tor/j_total from
    # the template (exact only when the draw's flux geometry matches baseline).
    if "j_total" in cp and "j_tor" in cp:
        use_exact = False
        if fidelity in ("auto", "exact") and eq_fsa is not None:
            geom = _eq_fsa_geom_on(eq_fsa, psiN_t, _imas_b0(out, ie, ic))
            if geom is not None:
                from ..physics import toroidal_to_parallel
                cp["j_total"] = toroidal_to_parallel(jt_t, geom=geom).tolist()
                cp["j_ohmic"] = toroidal_to_parallel(to_t(j_ind, peq), geom=geom).tolist()
                cp["j_bootstrap"] = toroidal_to_parallel(to_t(j_bs, peq), geom=geom).tolist()
                use_exact = True
        if fidelity == "exact" and not use_exact:
            raise ValueError(
                f"fidelity='exact' requested but draw {draw_index} has no "
                "captured eq_fsa block (generate with capture_live_eq=True). "
                "Use fidelity='auto' to fall back to the baseline-ratio "
                "reconstruction.")
        if not use_exact:                      # baseline-ratio reconstruction
            base_jtot = np.asarray(cp["j_total"], dtype=float)
            base_jtor = np.asarray(cp["j_tor"], dtype=float)
            eps = 1e-9 * np.nanmax(np.abs(base_jtot)) if base_jtot.size else 0.0
            good = np.abs(base_jtot) > eps
            c = np.ones_like(base_jtot)
            c[good] = base_jtor[good] / base_jtot[good]
            if not np.all(good):
                idx = np.arange(c.size)
                c[~good] = np.interp(idx[~good], idx[good], c[good])
            with np.errstate(divide="ignore", invalid="ignore"):
                cp["j_total"] = (jt_t / c).tolist()
                cp["j_ohmic"] = (to_t(j_ind, peq) / c).tolist()
                cp["j_bootstrap"] = (to_t(j_bs, peq) / c).tolist()

    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return out_path


def export_imas_drawset(h5path_or_header, template_ids_path, out_dir,
                        scan_key=None, time=None, selection="selected",
                        fidelity="auto"):
    """Write one perturbed IMAS/OMAS IDS per draw (see :func:`write_imas_draw`).

    Files are ``{out_dir}/{header}_draw{idx}.json``. ``selection`` is
    ``"selected"`` (in-spec only) or ``"all"``. ``fidelity`` is forwarded to
    :func:`write_imas_draw` (``"auto"`` -> exact per-draw conversion when the
    archive carries the captured ``eq_fsa`` block, else baseline-ratio).

    Operates on a single scan.  Pass ``scan_key`` to select it; the default
    ``scan_key=None`` is the flat layout, and is only unambiguous when the
    file holds exactly one scan (otherwise raises -- pass an explicit key).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive reference -- ``.h5`` path or bare header stem.

    Returns
    -------
    list of str
        The written IDS paths.
    """
    import os
    from ..filtering import select_indices
    from ..utils import _resolve_h5

    os.makedirs(out_dir, exist_ok=True)
    h5 = _resolve_h5(h5path_or_header)
    base = os.path.basename(h5).replace(".h5", "")
    idxs = select_indices(h5, scan_key=scan_key, selection=selection)
    if isinstance(idxs, dict):
        # scan_key=None over a scan-layout file -> ambiguous unless single.
        if len(idxs) != 1:
            raise ValueError(
                f"{h5} holds {len(idxs)} scans {sorted(idxs)}; pass an "
                "explicit scan_key to export_imas_drawset().")
        scan_key, idxs = next(iter(idxs.items()))
    paths = []
    for i in idxs:
        out = os.path.join(out_dir, f"{base}_draw{i}.json")
        paths.append(write_imas_draw(h5, i, template_ids_path, out,
                                     scan_key=scan_key, time=time,
                                     fidelity=fidelity))
    return paths
