"""Registry of the experimental features: opt-in, unvalidated, and stamped.

A feature listed here is reachable only by an explicit configuration value.
It has not been validated on real data, so it is never a default.  When a
run enables one:

* :meth:`bouquet.Bouquet.prepare_baseline` emits one
  :class:`ExperimentalFeatureWarning` per feature, naming it and its open
  validation items;
* the baseline record carries the list (``Baseline.experimental_features``)
  and the archive carries it as the ``_baseline`` attr
  ``experimental_features_json`` on every solve method (engine, legacy, swb),
  next to ``engine_resolved_defaults_json``;
* :func:`bouquet.stats.draw_band` and :class:`bouquet.archive.ScanView`
  print it with their summaries.

``REGISTRY`` maps a feature key to a record:

``description``
    What the feature does, in one paragraph.
``enabled_by``
    The configuration values that enable it.
``introduced``
    The pull request(s) and release that added it.
``validation_todo``
    What has to be shown before it may become a default.
``status``
    ``"experimental"``.

:func:`experimental_features_enabled` returns the keys a configuration
enables.  The docs section "Experimental features and their validation
status" (docs/workflows.md) lists the same keys; a test holds the two in
step.
"""
from __future__ import annotations

import warnings

__all__ = ["REGISTRY", "ExperimentalFeatureWarning",
           "experimental_features_enabled", "warn_experimental_features",
           "docs_markdown", "DOCS_BEGIN", "DOCS_END",
           "STANDARD_NI_SOURCE", "IDA_ION_ROUTES", "NI_SOURCES",
           "AUTO_NI_SOURCE", "MEASURED_NI_SOURCE", "IMAS_NI_SOURCES",
           "GRADUATED", "resolved_ni_source", "resolved_ni_subtract_fast",
           "ida_ion_route_on", "fuse_zeff_fast_ions_on",
           "ida_ni_beam_subtraction_on", "kinetic_clips_on",
           "ida_fuse_timing_on", "graduated_markdown", "GRADUATED_BEGIN",
           "GRADUATED_END"]


class ExperimentalFeatureWarning(UserWarning):
    """A run enables a feature listed in :data:`REGISTRY`."""


#: The pre-#56 Z_eff / n_i route: the file's VB Z_eff on a g-file + IDA run
#: (ReconstructionSource's default); on ida_hybrid, the dd's Z_eff with n_i
#: from it and the IDA n_e (an explicit choice there since 2026-10).
STANDARD_NI_SOURCE = "standard"
#: The measured IDA routes (PR #56): VB Z_eff, CER carbon, or their mean.
IDA_ION_ROUTES = ("Zeff", "CER", "all")
#: Every ``ni_source`` a :class:`~bouquet.config.ReconstructionSource` takes.
NI_SOURCES = (STANDARD_NI_SOURCE,) + IDA_ION_ROUTES
#: :class:`~bouquet.config.ImasSource`'s default: resolved per run
#: (:func:`resolved_ni_source`).
AUTO_NI_SOURCE = "auto"
#: What ``"auto"`` resolves to when an IDA file supplies the kinetics
#: (``kinetic_source="ida_hybrid"``): the measured route -- the VB+CER mean
#: where both routes have an envelope, else whichever does (owner decision
#: 2026-10: ONE thermal-kinetics source per run).
MEASURED_NI_SOURCE = "all"
#: Every ``ni_source`` an :class:`~bouquet.config.ImasSource` takes.
IMAS_NI_SOURCES = (AUTO_NI_SOURCE,) + NI_SOURCES


REGISTRY = {
    "fuse_zeff_fast_ions": dict(
        description=(
            "On the FUSE (IMAS) path, classify the dd's stored Z_eff against "
            "the thermal-only and the thermal+fast numerators (io.imas."
            "_dd_zeff) and, where it counts the fast ions (or when no Z_eff "
            "is stored and the dd carries a beam), hand the bootstrap "
            "Z_eff_th + z2_fast/n_e with Baseline.zeff_includes_fast=True.  "
            "Default: the thermal-only Z_eff recomputed from the dd's "
            "thermal ion densities."),
        enabled_by=["ImasSource.zeff_fast_ions=True"],
        introduced="PR #56, 1.4.0",
        validation_todo=[
            "on >= 5 real beam-heated dd's: the classification against the "
            "dd's own j_bootstrap (which numerator reproduces it)",
            "I_BS and l_i sensitivity to the fast-ion term on the same dd's",
        ],
        status="experimental"),
    "swb_solve_method": dict(
        description=(
            "solve_with_bootstrap as the baseline and every draw on the IMAS "
            "path (bouquet.swb): solve A at the setup coil regularisation, "
            "solve B regularised toward A's coils, the draws solve B with "
            "resampled kinetics; I_p accepted within swb_ip_tol; the edge "
            "taper (swb_edge_taper_psi0) opt-in."),
        enabled_by=["GenerationConfig.solve_method='swb'",
                    "GenerationConfig.imas_baseline='swb'"],
        introduced="PR #64 / #69 / #70, 1.4.0",
        validation_todo=[
            "real-data arms on an OpenFUSIONToolkit build whose "
            "solve_with_bootstrap takes x / jphi_fixed / p_fixed, on dd's "
            "from a current IDA_fuse (so far: 6 of 8 IDA-hybrid "
            "reconstructions converged on the PR author's build; dd vintage "
            "unrecorded)",
            "swb_ip_tol statistics: the distribution of each solve's "
            "|I_p/I_p,target - 1| (swb_ip_rel_err) over real draws, against "
            "the 5e-3 acceptance",
            "taper off (swb_edge_taper_psi0=None) verified on real slices: "
            "the Picard 2-cycle the taper was added for does not occur, or "
            "is caught",
        ],
        status="experimental"),
    "bootstrap_convergence_override": dict(
        description=(
            "Accept the bootstrap_kwargs keys that change a convergence "
            "criterion of the toolkit's internal bootstrap solve "
            "(BOOTSTRAP_CONVERGENCE_KWARGS: djbs_tol, saw_relax).  Legacy "
            "and swb paths only; the unified engine refuses it."),
        enabled_by=["GenerationConfig.bootstrap_convergence_override=True"],
        introduced="PR #60, 1.4.0",
        validation_todo=[
            "for each key, the converged jphi_bs, l_i and q0 against the "
            "toolkit's default criterion on >= 5 real slices; a value that "
            "changes a converged result beyond the default criterion's own "
            "residual is a loosened criterion and must be flagged",
            "the effective values stamped per run are read back by "
            "load_config and shown with the archive summary",
        ],
        status="experimental"),
    "ida_fuse_timing": dict(
        description=(
            "Read the IDA slice (ida_hybrid's ImasSource.ida_path, or a "
            "separate uncertainty.ida_path) at the file's nearest own slice "
            "to the requested time AT ANY DISTANCE, as IDA_fuse pairs its "
            "own dd and IDA slices (io.imas.IDA_FUSE_TIME_RULE, the #73 "
            "timing before review).  It removes the IDA time-window "
            "acceptance: an IDA slice far from the dd slice is read without "
            "a refusal (the match, dt included, is still archived in "
            "li_metrics['ida_time_match']).  Default 'window': the nearest "
            "slice within half the IDA file's local time-step, else refused "
            "(io.imas.IDA_TIME_RULE)."),
        enabled_by=["ImasSource.ida_timing='ida_fuse' with an ida_path or "
                    "uncertainty.ida_path"],
        introduced="#79, 1.4.0",
        validation_todo=[
            "on >= 5 real IDA_fuse runs: the |dt| distribution of the "
            "slices it accepts beyond the window rule's half-step, and the "
            "kinetic profiles of each against the in-window slice",
            "I_BS, l_i(3), q0 and beta_N of a run timed by 'ida_fuse' "
            "against the same slice timed by 'window', where both accept "
            "it, and wherever 'window' refuses: show the accepted slice is "
            "the one FUSE's replay pairing used (ida_provenance.json)",
        ],
        status="experimental"),
}


#: Features that LEFT the registry: defaults now, on validation evidence, with
#: the open items of their registry record kept as KNOWN LIMITATIONS (docs/
#: workflows.md, "Graduated features and their known limitations").  Never
#: warned about; the route a run took is stamped on the baseline record
#: (``li_metrics["kinetics_route"]``, ``IDAProfiles.zeff_provenance``).
GRADUATED = {
    "ida_ion_route": dict(
        description=(
            "Z_eff and n_i from the IDA file's measured routes: the "
            "visible-bremsstrahlung Z_eff ('Zeff'), the CER carbon density "
            "('CER'), or the clamped equal-weight mean of the two ('all'), "
            "with the route disagreement folded into sigma_ni / sigma_Zeff, "
            "the 'IDA-resolved' Z_eff envelope and the PR #56 auto rule of "
            "UncertaintyConfig.ni_from_zeff.  The DEFAULT on ida_hybrid "
            "(ImasSource.ni_source='auto' -> 'all'; ONE thermal-kinetics "
            "source per run); on a g-file + IDA run an explicit "
            "ReconstructionSource.ni_source."),
        graduated=("2026-10 (owner decision): default on ida_hybrid; left "
                   "the experimental registry"),
        evidence=[
            "the sigma=0 draw reproduced the reconstruction on a reversed-Ip "
            "near-double-null IDA-hybrid slice with the measured route, "
            "where the dd-Z_eff route missed the archived-stage gates "
            "(before the engine draws held the reconstruction's isoflux)",
            "on real IDA-hybrid slices the dd's Z_eff differed from the IDA "
            "file's own by factors 0.78 and 3.9 (median over psi_N <= 0.9), "
            "with no signal of the disagreement on the dd route",
        ],
        known_limitations=[
            "the IDA-derived n_i has not been compared against the dd's "
            "total n_i on >= 5 real slices (agreement within the combined "
            "1-sigma envelope, or each disagreement explained)",
            "Z_eff on axis: the VB+CER mean has not been compared against "
            "the CER-only and VB-only routes on the same slices with the "
            "route tension (zeff_route_chi) reported",
            "the I_BS, l_i(3), q0 and beta_N sensitivity to the route has "
            "not been measured on >= 5 real H-mode slices against the dd "
            "route and the measurement uncertainties",
            "old IDA vintages without Zeff_err ('all' falls to CER alone): "
            "not shown that the CER-only Z_eff is no worse than VB on those "
            "files",
        ]),
    "kinetic_sampler_clips": dict(
        description=(
            "The PR #56 clips in the kinetic sampler: a drawn Z_eff floored "
            "at 1 even where physics.zeff_bounds allows less (thermal-"
            "numerator Z_eff with a beam), a derived n_i held in "
            "[0, n_e - z_fast], an independent n_i capped at n_e - z_fast "
            "when Z_imp is declared, and a passive Z_eff aux draw clipped to "
            "the same window.  Every clip that fires is counted per draw "
            "(KineticDraw.clips).  On (UncertaintyConfig.kinetic_clips=None, "
            "auto) with the measured IDA ni_source route in play -- the "
            "IDA-hybrid DEFAULT -- the beam subtraction or "
            "fuse_zeff_fast_ions; off otherwise; an explicit bool wins."),
        graduated=("2026-10-14 (owner decision): on with the measured route, "
                   "default included; left the experimental registry with it"),
        evidence=[
            "the clips change no draw unless they bind, and every clip that "
            "fires is counted per draw and archived",
            "they ride with the measured IDA route (graduated 2026-10 on the "
            "sigma=0 evidence of a reversed-Ip near-double-null slice)",
        ],
known_limitations=[
            "per-clip counts on a real ensemble (KineticDraw.clips): the "
            "fraction of draws and nodes each clip moves",
            "the bias each clip introduces in the drawn Z_eff, n_i and I_BS "
            "distributions against the unclipped draws, same seed",
        ],),
    "ida_ni_beam_subtraction": dict(
        description=(
            "ida_hybrid on a measured route: subtract the dd's fast-ion "
            "density equivalent from the IDA (total) n_i to give a thermal "
            "n_i (io.imas._subtract_fast_ni), so the fast ions are counted "
            "once -- in the dd's p_fast, the one declared cross-source "
            "input.  The IDA n_i and the dd's total n_i are compared at "
            "psi_N 0, 0.2, ..., 0.8 and a disagreement above 1 % warns.  "
            "The DEFAULT there (ImasSource.ni_subtract_fast=None follows "
            "include_pfast=True)."),
        graduated=("2026-10 (owner decision): default with the measured "
                   "route and include_pfast=True (the #79 rule: the fast "
                   "ions in p_fast with the subtraction, or both off)"),
        evidence=[
            "the sigma=0 draw reproduced the reconstruction on a reversed-Ip "
            "near-double-null IDA-hybrid slice with the measured route and "
            "the subtraction",
        ],
        known_limitations=[
            "the IDA n_i has not been compared against the dd's total n_i on "
            ">= 5 real beam slices (the gate the subtraction prints); the "
            "subtracted density is trustworthy only where that disagreement "
            "is within the measurement uncertainty",
            "the thermal n_i floor at 0 (counted, ni_fast_meta.n_floored) "
            "is not shown not to bind on real slices",
        ]),
}


# ---------------------------------------------------------------------------
#  predicates: one definition, used by the code paths AND by the registry
# ---------------------------------------------------------------------------
def _ida_in_play(source, uncertainty=None) -> bool:
    """An IDA ``.cdf`` reaches the run: the reconstruction source's own
    profiles file, ``ImasSource.ida_path`` or ``uncertainty.ida_path``."""
    from .config import ImasSource, ReconstructionSource
    if uncertainty is not None and getattr(uncertainty, "ida_path", None):
        return True
    if isinstance(source, ReconstructionSource):
        return str(getattr(source, "profiles_path", "") or "").endswith(".cdf")
    if isinstance(source, ImasSource):
        return bool(getattr(source, "ida_path", None))
    return False


def resolved_ni_source(source, kinetic_source=None) -> str:
    """The Z_eff / n_i route a run takes.  An explicit ``ni_source`` is
    honoured as given; ``"auto"`` (:class:`~bouquet.config.ImasSource`'s
    default) is the measured route :data:`MEASURED_NI_SOURCE` when an IDA file
    supplies the kinetics (``kinetic_source="ida_hybrid"`` with an
    ``ida_path``), else ``"standard"`` (FUSE kinetics: the dd's Z_eff)."""
    v = getattr(source, "ni_source", STANDARD_NI_SOURCE)
    if v != AUTO_NI_SOURCE:
        return v
    from .config import ImasSource
    if (isinstance(source, ImasSource) and kinetic_source == "ida_hybrid"
            and getattr(source, "ida_path", None)):
        return MEASURED_NI_SOURCE
    return STANDARD_NI_SOURCE


def resolved_ni_subtract_fast(source, kinetic_source=None) -> bool:
    """Whether the dd's beam density is subtracted from the IDA (total) n_i.
    An explicit ``ImasSource.ni_subtract_fast`` is honoured; ``None`` (the
    default) follows ``include_pfast`` on the ida_hybrid measured route --
    the fast ions in ``p_fast`` with the subtraction, or both off (the #79
    rule) -- and is False elsewhere (no IDA-derived n_i)."""
    from .config import ImasSource
    if not isinstance(source, ImasSource):
        return False
    v = getattr(source, "ni_subtract_fast", None)
    if v is not None:
        return bool(v)
    return bool(kinetic_source == "ida_hybrid"
                and getattr(source, "ida_path", None)
                and resolved_ni_source(source, kinetic_source)
                in IDA_ION_ROUTES
                and bool(getattr(source, "include_pfast", True)))


def ida_ion_route_on(source, uncertainty=None, kinetic_source=None) -> bool:
    """The resolved ``ni_source`` (:func:`resolved_ni_source`) is a measured
    IDA route and an IDA file is in play."""
    return (resolved_ni_source(source, kinetic_source) in IDA_ION_ROUTES
            and _ida_in_play(source, uncertainty))


def _explicit_ida_ion_route(source, uncertainty=None) -> bool:
    """``ni_source`` was SET to a measured route (not resolved from
    ``"auto"``) and an IDA file is in play: the PR #56 opt-in the sampler
    clips' auto rule follows, unchanged."""
    return (getattr(source, "ni_source", STANDARD_NI_SOURCE) in IDA_ION_ROUTES
            and _ida_in_play(source, uncertainty))


def fuse_zeff_fast_ions_on(source) -> bool:
    """:data:`REGISTRY` ``"fuse_zeff_fast_ions"``."""
    from .config import ImasSource
    return isinstance(source, ImasSource) and bool(
        getattr(source, "zeff_fast_ions", False))


def ida_ni_beam_subtraction_on(source, kinetic_source=None) -> bool:
    """:func:`resolved_ni_subtract_fast` (:data:`GRADUATED`
    ``"ida_ni_beam_subtraction"``)."""
    return resolved_ni_subtract_fast(source, kinetic_source)


def kinetic_clips_on(config) -> bool:
    """``UncertaintyConfig.kinetic_clips`` resolved: an explicit bool wins;
    ``None`` (auto) is on whenever a kinetic route that can draw outside the
    physical window is in play -- the measured IDA ``ni_source`` route
    (resolved, so the IDA-hybrid DEFAULT counts; owner decision 2026-10-14),
    the beam subtraction (resolved), or ``fuse_zeff_fast_ions``.  Every clip
    that fires is counted per draw, so the default changes nothing unless a
    clip binds, and then the archive says so."""
    unc = getattr(config, "uncertainty", None)
    v = getattr(unc, "kinetic_clips", None)
    if v is not None:
        return bool(v)
    src = getattr(config, "source", None)
    ks = getattr(getattr(config, "generation", None), "kinetic_source", None)
    return (ida_ion_route_on(src, unc, ks) or fuse_zeff_fast_ions_on(src)
            or ida_ni_beam_subtraction_on(src, ks))


def ida_fuse_timing_on(source, uncertainty=None) -> bool:
    """:data:`REGISTRY` ``"ida_fuse_timing"``: ``ImasSource.ida_timing=
    "ida_fuse"`` with an IDA file in play (without one it times nothing)."""
    from .config import ImasSource
    return (isinstance(source, ImasSource)
            and getattr(source, "ida_timing", "window") == "ida_fuse"
            and _ida_in_play(source, uncertainty))


def _swb_on(config) -> bool:
    gc = getattr(config, "generation", None)
    if gc is None:
        return False
    if getattr(gc, "solve_method", None) == "swb":
        return True
    return getattr(gc, "imas_baseline", None) == "swb"


_PREDICATES = {
    "fuse_zeff_fast_ions": lambda c: fuse_zeff_fast_ions_on(
        getattr(c, "source", None)),
    "swb_solve_method": _swb_on,
    "bootstrap_convergence_override": lambda c: bool(getattr(
        getattr(c, "generation", None), "bootstrap_convergence_override",
        False)),
    "ida_fuse_timing": lambda c: ida_fuse_timing_on(
        getattr(c, "source", None), getattr(c, "uncertainty", None)),
}
assert set(_PREDICATES) == set(REGISTRY)
assert not set(GRADUATED) & set(REGISTRY)


def experimental_features_enabled(config) -> list:
    """The :data:`REGISTRY` keys *config* enables, in registry order."""
    return [k for k in REGISTRY if _PREDICATES[k](config)]


def warn_experimental_features(config, stacklevel=2) -> list:
    """One :class:`ExperimentalFeatureWarning` per enabled feature, naming it
    and its validation items; returns :func:`experimental_features_enabled`."""
    on = experimental_features_enabled(config)
    for k in on:
        r = REGISTRY[k]
        warnings.warn(
            f"EXPERIMENTAL feature {k!r} is enabled "
            f"({'; '.join(r['enabled_by'])}; introduced {r['introduced']}). "
            "It is not validated on real data; open validation items: "
            + " | ".join(r["validation_todo"])
            + f".  See bouquet.experimental.REGISTRY[{k!r}].",
            ExperimentalFeatureWarning, stacklevel=stacklevel + 1)
    return on


#: Markers around the generated block in docs/workflows.md.
DOCS_BEGIN = "<!-- experimental-registry:begin (generated by bouquet.experimental.docs_markdown) -->"
DOCS_END = "<!-- experimental-registry:end -->"


def docs_markdown() -> str:
    """The docs block (docs/workflows.md, "Experimental features and their
    validation status") generated from :data:`REGISTRY`, between
    :data:`DOCS_BEGIN` and :data:`DOCS_END`.  A test holds the docs to it."""
    out = [DOCS_BEGIN, ""]
    for k, r in REGISTRY.items():
        out.append(f"#### `{k}` ({r['status']}; introduced {r['introduced']})")
        out.append("")
        out.append(r["description"])
        out.append("")
        out.append("Enabled by: " + "; ".join(f"`{e}`" for e in r["enabled_by"]))
        out.append("")
        out.append("Validation to do:")
        out.append("")
        out.extend(f"- {t}" for t in r["validation_todo"])
        out.append("")
    out.append(DOCS_END)
    return "\n".join(out)


#: Markers around the generated "known limitations" block in docs/workflows.md.
GRADUATED_BEGIN = ("<!-- graduated-features:begin (generated by "
                   "bouquet.experimental.graduated_markdown) -->")
GRADUATED_END = "<!-- graduated-features:end -->"


def graduated_markdown() -> str:
    """The docs block (docs/workflows.md, "Graduated features and their
    known limitations") generated from :data:`GRADUATED`, between
    :data:`GRADUATED_BEGIN` and :data:`GRADUATED_END`.  A test holds the
    docs to it."""
    out = [GRADUATED_BEGIN, ""]
    for k, r in GRADUATED.items():
        out.append(f"#### `{k}` (graduated {r['graduated']})")
        out.append("")
        out.append(r["description"])
        out.append("")
        out.append("Evidence:")
        out.append("")
        out.extend(f"- {t}" for t in r["evidence"])
        out.append("")
        out.append("Known limitations (open validation items, kept):")
        out.append("")
        out.extend(f"- {t}" for t in r["known_limitations"])
        out.append("")
    out.append(GRADUATED_END)
    return "\n".join(out)
