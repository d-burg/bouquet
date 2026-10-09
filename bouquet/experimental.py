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
           "STANDARD_NI_SOURCE", "IDA_ION_ROUTES", "NI_SOURCES",
           "ida_ion_route_on", "fuse_zeff_fast_ions_on",
           "ida_ni_beam_subtraction_on", "kinetic_clips_on"]


class ExperimentalFeatureWarning(UserWarning):
    """A run enables a feature listed in :data:`REGISTRY`."""


#: The established Z_eff / n_i route (the default, as before PR #56).
STANDARD_NI_SOURCE = "standard"
#: The PR #56 routes (experimental): VB Z_eff, CER carbon, or their mean.
IDA_ION_ROUTES = ("Zeff", "CER", "all")
#: Every accepted ``ni_source``.
NI_SOURCES = (STANDARD_NI_SOURCE,) + IDA_ION_ROUTES


REGISTRY = {
    "ida_ion_route": dict(
        description=(
            "Z_eff and n_i from the PR #56 IDA ladder: the visible-"
            "bremsstrahlung Z_eff ('Zeff'), the CER carbon density ('CER'), "
            "or the clamped mean of the two ('all'), with the route "
            "disagreement folded into sigma_ni / sigma_Zeff, the "
            "'IDA-resolved' Z_eff envelope and the PR #56 auto rule of "
            "UncertaintyConfig.ni_from_zeff.  On ida_hybrid it also takes "
            "Z_eff and n_i from the IDA file instead of the dd (unless "
            "ImasSource.zeff_from_fuse).  Default 'standard': Z_eff is the "
            "file's VB value (the dd's on ida_hybrid), n_i follows from it "
            "by quasineutrality with the n_e-fraction sigma, and the Z_eff "
            "envelope comes from the carbon > VB > scalar ladder."),
        enabled_by=["ReconstructionSource.ni_source in ('Zeff', 'CER', 'all') "
                    "with an IDA .cdf profiles_path or uncertainty.ida_path",
                    "ImasSource.ni_source in ('Zeff', 'CER', 'all') with an "
                    "ida_path or uncertainty.ida_path"],
        introduced="PR #56, 1.4.0",
        validation_todo=[
            "compare the IDA-derived n_i against the dd's total n_i on >= 5 "
            "real slices; agree within the combined 1-sigma envelope, or "
            "explain each disagreement",
            "Z_eff on axis: the VB+CER mean against the CER-only and the "
            "VB-only routes on the same slices, with the route tension "
            "(zeff_route_chi) reported",
            "I_BS, l_i(3), q0 and beta_N sensitivity to the route on >= 5 "
            "real H-mode slices, against the standard route and the "
            "measurement uncertainties",
            "old IDA vintages without Zeff_err ('all' falls to CER alone): "
            "show the CER-only Z_eff is no worse than VB on those files",
        ],
        status="experimental"),
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
    "ida_ni_beam_subtraction": dict(
        description=(
            "ida_hybrid with an experimental ni_source: subtract the dd's "
            "fast-ion density equivalent from the IDA (total) n_i to give a "
            "thermal n_i (io.imas._subtract_fast_ni).  The IDA n_i and the "
            "dd's total n_i are compared at psi_N 0, 0.2, ..., 0.8 and a "
            "disagreement above 1 % warns.  Default: no subtraction."),
        enabled_by=["ImasSource.ni_subtract_fast=True (with "
                    "kinetic_source='ida_hybrid' and an experimental "
                    "ni_source)"],
        introduced="PR #56, 1.4.0",
        validation_todo=[
            "the IDA n_i against the dd's total n_i on >= 5 real beam slices "
            "(the gate the subtraction prints); the disagreement must be "
            "within the measurement uncertainty before the subtracted "
            "density can be trusted",
            "the thermal n_i floor at 0 must not bind on real slices",
        ],
        status="experimental"),
    "kinetic_sampler_clips": dict(
        description=(
            "The PR #56 clips in the kinetic sampler: a drawn Z_eff floored "
            "at 1 even where physics.zeff_bounds allows less (thermal-"
            "numerator Z_eff with a beam), a derived n_i held in "
            "[0, n_e - z_fast], an independent n_i capped at n_e - z_fast "
            "when Z_imp is declared, and a passive Z_eff aux draw clipped to "
            "the same window.  Every clip that fires is counted per draw.  "
            "Default: only the Z_eff window physics.zeff_bounds gives (the "
            "bound in place before PR #56)."),
        enabled_by=["UncertaintyConfig.kinetic_clips=True",
                    "UncertaintyConfig.kinetic_clips=None (auto) with "
                    "ida_ion_route, fuse_zeff_fast_ions or "
                    "ida_ni_beam_subtraction enabled"],
        introduced="PR #56, 1.4.0",
        validation_todo=[
            "per-clip counts on a real ensemble (KineticDraw.clips): the "
            "fraction of draws and nodes each clip moves",
            "the bias each clip introduces in the drawn Z_eff, n_i and I_BS "
            "distributions against the unclipped draws, same seed",
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
            "a real-data arm on an OpenFUSIONToolkit build whose "
            "solve_with_bootstrap takes x / jphi_fixed / p_fixed (every "
            "real-data arm so far was refused at prepare_baseline)",
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
            "(BOOTSTRAP_CONVERGENCE_KWARGS: djBS_tol, saw_relax).  Legacy "
            "and swb paths only; the unified engine refuses it."),
        enabled_by=["GenerationConfig.bootstrap_convergence_override=True"],
        introduced="PR #60, 1.4.0",
        validation_todo=[
            "for each key, the converged j_BS, l_i and q0 against the "
            "toolkit's default criterion on >= 5 real slices; a value that "
            "changes a converged result beyond the default criterion's own "
            "residual is a loosened criterion and must be flagged",
            "the effective values stamped per run are read back by "
            "load_config and shown with the archive summary",
        ],
        status="experimental"),
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


def ida_ion_route_on(source, uncertainty=None) -> bool:
    """``ni_source`` names a PR #56 route and an IDA file is in play
    (:data:`REGISTRY` ``"ida_ion_route"``)."""
    return (getattr(source, "ni_source", STANDARD_NI_SOURCE) in IDA_ION_ROUTES
            and _ida_in_play(source, uncertainty))


def fuse_zeff_fast_ions_on(source) -> bool:
    """:data:`REGISTRY` ``"fuse_zeff_fast_ions"``."""
    from .config import ImasSource
    return isinstance(source, ImasSource) and bool(
        getattr(source, "zeff_fast_ions", False))


def ida_ni_beam_subtraction_on(source) -> bool:
    """:data:`REGISTRY` ``"ida_ni_beam_subtraction"``."""
    from .config import ImasSource
    return isinstance(source, ImasSource) and bool(
        getattr(source, "ni_subtract_fast", False))


def kinetic_clips_on(config) -> bool:
    """``UncertaintyConfig.kinetic_clips`` resolved: an explicit bool wins;
    ``None`` (auto) follows the PR #56 kinetic route -- on when any of
    ``ida_ion_route``, ``fuse_zeff_fast_ions`` or
    ``ida_ni_beam_subtraction`` is enabled."""
    unc = getattr(config, "uncertainty", None)
    v = getattr(unc, "kinetic_clips", None)
    if v is not None:
        return bool(v)
    src = getattr(config, "source", None)
    return (ida_ion_route_on(src, unc) or fuse_zeff_fast_ions_on(src)
            or ida_ni_beam_subtraction_on(src))


def _swb_on(config) -> bool:
    gc = getattr(config, "generation", None)
    if gc is None:
        return False
    if getattr(gc, "solve_method", None) == "swb":
        return True
    return getattr(gc, "imas_baseline", None) == "swb"


_PREDICATES = {
    "ida_ion_route": lambda c: ida_ion_route_on(
        getattr(c, "source", None), getattr(c, "uncertainty", None)),
    "fuse_zeff_fast_ions": lambda c: fuse_zeff_fast_ions_on(
        getattr(c, "source", None)),
    "ida_ni_beam_subtraction": lambda c: ida_ni_beam_subtraction_on(
        getattr(c, "source", None)),
    "kinetic_sampler_clips": kinetic_clips_on,
    "swb_solve_method": _swb_on,
    "bootstrap_convergence_override": lambda c: bool(getattr(
        getattr(c, "generation", None), "bootstrap_convergence_override",
        False)),
}
assert set(_PREDICATES) == set(REGISTRY)


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
