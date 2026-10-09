"""Typed configuration for the Bouquet orchestrator.

The configuration is split along two independent axes so the same generation
machinery serves multiple input pipelines:

  * **Baseline source** -- where the baseline current density (j_phi / j_ohmic /
    j_BS), the targets (Ip, l_i) and the baseline kinetic profiles come from.
    Either reconstruct them from a g-file (:class:`ReconstructionSource`) or read
    them pre-separated from a FUSE IMAS/OMAS IDS (:class:`ImasSource`).

  * **Uncertainty envelope** (:class:`UncertaintyConfig`) -- the kinetic sigma
    profiles and the j_phi sigma, plus the GPR correlation lengths that define
    how profiles are perturbed. Orthogonal to the baseline source.

Pass a fully-populated :class:`BouquetConfig` to ``bq.Bouquet(config)``.

Using dataclasses (rather than a raw dict) buys validation, IDE autocomplete,
and a documented home for every knob -- a typo fails immediately in
``__post_init__`` instead of deep inside a GS solve.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def require_integer_count(value, name):
    """``int(value)`` for an optional count, refusing bool and non-integral input.

    ``int()`` accepts both silently -- ``7.9`` becomes 7 and ``True`` becomes 1 --
    and these counts decide how many draws a run makes and when it stops, so a
    truncated one is a quietly different run. Shared by
    :meth:`BouquetConfig.__post_init__` and the re-check in ``Bouquet.generate``
    (the documented notebook idiom mutates the fields after construction).
    """
    if value is None:
        return None
    if isinstance(value, bool) or float(value) != int(value):
        raise ValueError(
            f"{name}={value!r} must be an integer count (or None); bool and "
            "fractional values are refused rather than truncated")
    return int(value)


# ---------------------------------------------------------------------------
# Solver (common to every baseline source -- perturbed draws are always solved
# with TokaMaker, regardless of where the baseline came from)
# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """TokaMaker setup -- everything needed to stand up ``mygs``.

    ``cond_dict`` / ``coil_dict`` normally come straight from the mesh file and
    need no adjustment; they are intentionally *not* surfaced here. Pass
    ``region_overrides`` only in the special cases where you must tweak them.
    """

    mesh_path: str
    # Default nthreads=1: GS solves are then bit-reproducible. nthreads>1 uses
    # OpenMP whose reduction order is non-deterministic -- which the LCFS-normalised
    # li_1 amplifies to ~±1% run-to-run, AND (at stiff slices, e.g. beam-heavy
    # equilibria with a sharp pedestal current) can occasionally tip the GS DLSODE
    # solve into a non-convergence loop on the unlucky thread schedule (a hard hang).
    # nthreads=1 removes both at ~no speed cost here (the solve barely thread-scales);
    # for throughput, run multiple single-threaded PROCESSES in parallel, one per
    # slice, each BLAS-pinned (VECLIB_MAXIMUM_THREADS=1 etc.), NOT nthreads>1.
    nthreads: int = 1
    order: int = 3
    F0: Optional[float] = None                       # vacuum R*Bt; default from g-file/IDS
    isoflux_pts: Optional["np.ndarray"] = None       # (N, 2) boundary constraints
    isoflux_weights: Optional["np.ndarray"] = None   # (N,)
    # Optional saddle (X-point) constraints: (N, 2) points where B_pol is driven
    # to zero (TokaMaker set_saddle_constraints). Pins the separatrix X-point --
    # without it a diverted forward solve typically ROUNDS the boundary corner
    # by a few cm (isoflux alone does not localize the null). Opt-in: default
    # None changes nothing. Weight ~ isoflux scale to start (tune per case).
    saddle_targets: Optional["np.ndarray"] = None    # (N, 2) X-point pins
    saddle_weights: Optional["np.ndarray"] = None    # (N,); default 1.0 each
    coil_vsc: dict = field(default_factory=lambda: {"F9A": 1.0, "F9B": -1.0})
    # Coil-current regularisation terms, each
    #   {"coils": {name: coeff}, "target": float, "weight": float}
    # Empty (default) => every coil pulled toward ZERO at unit weight, the
    # historical behaviour. Populate to pin coils to measured currents; see
    # bouquet.coil_targets.coil_reg_from_measured. Targets are in the SOLVE
    # (positive-Ip) frame: measured_from_pf_active multiplies the measured
    # circuit currents by the source's orientation factor and records it on
    # each term ("source_current_sign"); a term whose recorded factor disagrees
    # with the IMAS baseline's source_current_sign is refused. Applied by
    # Bouquet._apply_coil_reg at BOTH setup_solver and _reset_solver_state --
    # the reset runs immediately before the IMAS baseline solve, so anything
    # installed only at setup is discarded.
    # When populated, these targets are also what the draw path's WEAK
    # exploratory regularisation aims at (same targets, historical weight 1.0)
    # instead of zero -- otherwise the exploration is pulled along the very coil
    # null space the targets exist to remove. Empty => that path is unchanged.
    coil_reg: list = field(default_factory=list)
    # Initial coil currents {name: A-t} for the IMAS baseline inverse solve --
    # seeds the iteration in a chosen basin; does NOT constrain the answer.
    # Applied by Bouquet._seed_coil_init, after init_psi. Unset => coils start
    # where init_psi left them (historical behaviour).
    # KNOWN NO-OP on the shipped path: the inverse solver re-solves every coil
    # current at each Picard step, so the seed is discarded before it can change
    # the converged baseline. It is retained as the single named hook for basin
    # selection if a forward-mode or warm-started baseline is ever added. To move
    # the baseline's coils use coil_reg (see bouquet.coil_targets), not this.
    coil_init: Optional[dict] = None
    region_overrides: Optional[dict] = None          # special-case cond/coil dict edits

    def __post_init__(self):
        """Validate the coil settings here, not inside ``setup_solver``.

        A malformed ``coil_reg`` entry used to die on ``set(t["coils"])`` with a
        bare ``KeyError``/``TypeError`` naming neither the entry nor the field,
        after the mesh had been loaded; ``coil_init`` was only type-checked when
        ``_seed_coil_init`` ran. Both mistakes are config typos and both checks
        are free, so they happen at construction.
        """
        if self.coil_reg is None:
            self.coil_reg = []
        if not isinstance(self.coil_reg, (list, tuple)):
            raise TypeError(
                "solver.coil_reg must be a list of "
                "{'coils': {name: coeff}, 'target': float, 'weight': float} terms, got "
                f"{type(self.coil_reg).__name__}")
        for i, term in enumerate(self.coil_reg):
            where = f"solver.coil_reg[{i}]"
            if not isinstance(term, dict):
                raise TypeError(f"{where} must be a dict, got {type(term).__name__}")
            if "coils" not in term:
                raise ValueError(
                    f"{where} has no 'coils' key; every term names the coils it "
                    "constrains as {name: coefficient}")
            if not isinstance(term["coils"], dict) or not term["coils"]:
                raise TypeError(
                    f"{where}['coils'] must be a non-empty {{name: coefficient}} dict, "
                    f"got {type(term['coils']).__name__}")
            for key in ("target", "weight"):
                if key in term:
                    try:
                        float(term[key])
                    except (TypeError, ValueError):
                        raise TypeError(
                            f"{where}[{key!r}] must be a number, got "
                            f"{type(term[key]).__name__}") from None
        if self.coil_init is not None and not hasattr(self.coil_init, "items"):
            raise TypeError(
                "solver.coil_init must be a {coil_name: current_A_turns} mapping, got "
                f"{type(self.coil_init).__name__}")


# ---------------------------------------------------------------------------
# Baseline sources (discriminated union via `BouquetConfig.source`)
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionSource:
    """Baseline obtained by reconstructing a GS equilibrium from a g-file.

    Runs :func:`reconstruct_equilibrium`: fits a smooth inductive profile and
    matches l_i(1) via secant iteration, producing the separated
    j_phi / j_inductive / j_BS that generation needs.

    Baseline kinetic profiles come from ``profiles_path``. The reader dispatches
    on file type -- an IDA ``.cdf`` is used if given (and is then also the
    default sigma source for :class:`UncertaintyConfig`), otherwise a p-file.
    No mixing by default: one file supplies the full profile set. Individual
    profiles can still be replaced via ``profile_overrides`` (e.g.
    ``{"ti": my_ti_array}``) on the kinetic psi_N grid.

    ``impurity_Z`` is the **machine impurity charge** used to map between Z_eff
    and the main-ion density under single-impurity quasineutrality. It is the
    controlling input for the IDA path (which measures ne + Z_eff and derives
    ni = ne (Z_imp - Zeff)/(Z_imp - 1)); the default 6.0 is **carbon**, correct
    for DIII-D and other carbon-wall machines. **Set it for your device** --
    e.g. tungsten ~ 74 (use the effective radiating charge if W is not fully
    stripped), beryllium 4, neon 10. For a p-file this is informational: the
    Osborne ``N Z A`` footer carries the species directly and is authoritative.

    ``ni_source`` picks the IDA main-ion density route: ``"Zeff"``
    (single-impurity quasineutrality), ``"CER"`` (``ni = max(ne - Z_imp n_C, 0)``
    from the measured ``n_12C6``), or ``"all"`` (default, the mean of the two).
    The two routes are independent measurements and may disagree. Ignored for p-files.
    """

    geqdsk_path: str
    profiles_path: str                 # IDA .cdf OR p-file (auto-detected by extension)
    cocos: int = 1
    time: Optional[float] = None       # IDA time slice [s] (multi-time .cdf files)
    impurity_Z: float = 6.0            # effective impurity charge (carbon); set per machine
    ni_source: str = "all"             # IDA path: "Zeff" | "CER" (n_12C6) | "all" (mean)
    profile_overrides: dict = field(default_factory=dict)  # name -> array, manual override
    # reconstruction knobs
    psi_pad: float = 1e-3
    n_k: int = 5                       # inductive-spline knots
    psi_bridge: float = 0.99          # Hermite edge-bridge location
    rescale_j_BS: bool = False
    shelf_psi_N: float = 0.0
    # Radial coordinate of the run (bouquet.coords): "psi_n", "phi_n" (profiles,
    # envelopes and solver inputs on normalised toroidal flux, mapped at read
    # with the g-file's own q), or "rho_tor" (read as phi_n = rho_tor**2).
    coord: str = "psi_n"
    # guess_jinductive is derived from the g-file j_phi when None

    def __post_init__(self):
        from .coords import run_coord
        run_coord(self.coord)


@dataclass
class ImasSource:
    """Baseline read directly from a FUSE IMAS/OMAS IDS -- no reconstruction.

    The IDS already carries j_ohmic and j_BS separated (j_phi = j_ohmic + j_BS),
    along with the equilibrium (Ip, l_i) and core_profiles (ne/Te/ni/Ti/Zeff),
    so the reconstruction step is skipped entirely. This is expected to become
    the primary input pipeline.
    """

    ids_path: str                      # IMAS/OMAS file (FUSE output)
    time: Optional[float] = None       # time slice [s]; None -> single/first slice
    # IDA-lite slice [s] for ida_hybrid; None -> `time`. `time` then picks only the dd slices
    # (equilibrium, core_profiles, core_sources). FUSE (replay_first) computed dd j_bootstrap(t)
    # on ONE IDA slice, recorded in ida_provenance.json "replay_pairing"; pass that slice here
    # and the macro step t as `time` to keep IDA kinetics and FUSE currents consistent.
    ida_time: Optional[float] = None
    # --- IDA-hybrid kinetics (GenerationConfig.kinetic_source = "ida_hybrid") ---
    # When set, the baseline ne/Te/Ti/ni/Z_eff/omega_tor come from this IDA
    # .cdf (externally fit, smoother across time than FUSE's per-slice fits),
    # resampled onto the FUSE core_profiles psi_N grid; currents, equilibrium,
    # p_fast and anchors stay FUSE.  ni via ni_source; zeff_from_fuse=True keeps
    # FUSE's Z_eff (consistent with FUSE's j_ohmic).  The sigma envelopes come
    # from the same file (resolve_uncertainty).
    ida_path: Optional[str] = None
    impurity_Z: float = 6.0            # machine impurity charge (carbon); ni dilution
    ni_source: str = "all"             # IDA ni route for ida_hybrid: "Zeff" | "CER" | "all"
    zeff_from_fuse: bool = False       # ida_hybrid: keep FUSE Z_eff instead of IDA's
    # OPTIONAL. A gEQDSK whose LCFS replaces the dd boundary outline as the
    # isoflux separatrix target. Leave None to use the source's own boundary.
    # Supply one when you have a more accurate separatrix for the slice than the
    # dd carries -- typically a magnetics-only equilibrium reconstruction, which
    # fits the boundary to the external magnetics without kinetic assumptions.
    # One g-file per slice; the driver picks the nearest time.
    LCFS_geqdsk: Optional[str] = None
    # Current orientation of this dd: the factor that brings EVERY current
    # profile it carries (core_profiles j_*, core_sources j_parallel,
    # equilibrium j_tor) into bouquet's positive-Ip frame.
    #   "auto" (default) -> sign(equilibrium ip) at the slice of the currents,
    #       and the read is REFUSED (ValueError) when the area-weighted
    #       integral of core_profiles.j_tor, or of the equilibrium j_tor that
    #       is used, disagrees with that sign -- a dd whose current profiles
    #       and plasma current were written in different orientations (e.g. an
    #       IDS conversion that mixed COCOS between IDSs).
    #   +1 / -1 -> use this factor instead of sign(ip).  For a user who KNOWS
    #       the file's current convention.  The normalised currents must still
    #       integrate positive; a factor that leaves them negative is refused
    #       the same way.  (A dd whose equilibrium and core_profiles currents
    #       disagree with EACH OTHER has no single factor: set
    #       GenerationConfig.anchor_jtor_to_equilibrium=False so the
    #       equilibrium j_tor is not used, or fix the file.)
    # The factor used and where it came from are recorded on the Baseline
    # (source_current_sign / source_current_sign_origin), in li_metrics and
    # ip_closure, and on the archive's _baseline attrs.
    current_orientation: Union[str, float] = "auto"
    # Radial coordinate of the run (bouquet.coords): "psi_n", "phi_n" (the
    # dd's core_profiles grid.rho_tor_norm**2), or "rho_tor" (same run).
    coord: str = "psi_n"

    def __post_init__(self):
        from .coords import run_coord
        run_coord(self.coord)


BaselineSource = Union[ReconstructionSource, ImasSource]


# ---------------------------------------------------------------------------
# Fixed additive components (NEVER perturbed by GPR draws)
# ---------------------------------------------------------------------------
@dataclass
class FixedComponentsConfig:
    """Externally-driven current + fast-ion pressure, held fixed across draws.

    These are summed into the baseline *and* every perturbed equilibrium,
    untouched by the GPR perturbation::

        j_phi_total = j_inductive + j_BS + j_NBI + j_RF + j_other
        p_total     = p_thermal(perturbed) + p_fast

    Any component may simply be handed in as a 1-D array over ``psi_N`` -- this is
    a first-class input path for *both* sources, not just a fallback. Supplying an
    array always overrides whatever a source would otherwise derive.

    Provenance / defaults:
      * ``p_fast`` -- :class:`ImasSource` builds it from per-species
        ``pressure_fast_{perpendicular,parallel}`` (reduced via
        ``p_fast_reduction``); :class:`ReconstructionSource` defaults to zero.
        Either way an explicit array here wins (e.g. from TRANSP/ONETWO).
      * ``j_NBI`` -- :class:`ImasSource` sums beam-source ``j_parallel``;
        :class:`ReconstructionSource` defaults to zero. Explicit array wins.
      * ``j_RF`` -- :class:`ImasSource` sums EC/LH/IC ``j_parallel``;
        :class:`ReconstructionSource` defaults to zero. Explicit array wins.
      * ``j_other`` -- :class:`ImasSource` sums fusion, runaways, sawteeth and
        unknown-index ``j_parallel``; zero elsewhere. Explicit array wins.

    All arrays are on ``psi_N`` (kinetic grid, in the run coordinate --
    Φ_N in a ``coord="phi_n"`` run, unless ``coord="psi_n"``), SI units,
    toroidal current convention for j_*. ``None`` -> zeros.

    Current orientation: ``j_NBI`` / ``j_RF`` are given in bouquet's
    POSITIVE-Ip frame -- co-current drive is positive, counter-current drive
    negative -- whatever the orientation of the source.  They are used exactly
    as given on both source paths: the IMAS reader does NOT multiply them by
    the dd's orientation factor (``Baseline.source_current_sign``) the way it
    multiplies the dd's own currents, and the reconstruction path never
    re-signs them either.  So for a reversed-Ip discharge a co-current beam is
    still a POSITIVE array here.
    """

    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure
    j_NBI: Optional["np.ndarray"] = None    # beam-driven TOROIDAL current density [A/m^2], co-Ip > 0
    j_RF: Optional["np.ndarray"] = None     # RF-driven TOROIDAL current density [A/m^2], co-Ip > 0
    j_other: Optional["np.ndarray"] = None  # other fixed driven TOROIDAL current [A/m^2]
    psi_N: Optional["np.ndarray"] = None    # grid for the above (if arrays given)
    # Coordinate of ``psi_N``: "run" (the run's), or "psi_n" (mapped to the
    # run coordinate through the source equilibrium's psi_N -> Phi_N map).
    coord: str = "run"

    # How to collapse anisotropic fast-ion pressure (p_perp, p_par) to the scalar
    # p_fast that a scalar-pressure GS solver needs. See
    # bouquet.physics.isotropize_fast_pressure and
    # bouquet.io.imas.resolve_p_fast_reduction.
    #
    # The dd field pair is written with two incompatible meanings whose scalars
    # differ by a FACTOR OF 3, and no dd field records which one is in use:
    #   "sum"   -> p_par + 2*p_perp       for IMAS.jl/FUSE, which store the fields
    #                                     PER DEGREE OF FREEDOM (pressa/3 each)
    #   "trace" -> (2*p_perp + p_par)/3   tr(P)/3, for the IMAS data-dictionary
    #                                     reading (full directional pressures) --
    #                                     what OMAS-written dds carry
    #   "mean"  -> (p_perp + p_par)/2
    #   "perp"  -> p_perp                 (diamagnetic-dominant)
    #   "auto"  -> [DEFAULT] pick "sum" or "trace" from the dd's own recorded
    #              provenance; if that cannot be determined, fall back to "sum"
    #              with a loud one-time warning. An explicit rule always wins and
    #              is applied silently. The rule used and the grounds for it are
    #              recorded on Baseline.p_fast_meta.
    p_fast_reduction: str = "auto"

    def __post_init__(self):
        from .coords import INPUT_COORDS
        if self.coord not in INPUT_COORDS:
            raise ValueError(f"fixed_components.coord must be one of "
                             f"{INPUT_COORDS}, got {self.coord!r}")


# ---------------------------------------------------------------------------
# Uncertainty envelope (orthogonal to the baseline source)
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyConfig:
    """Kinetic sigma profiles + j_phi sigma + GPR correlation lengths.

    Kinetic sigmas are sourced from an IDA ``.cdf`` (``ida_path``). Two modes
    match the operational notebook:

      * ``"ensemble"``  -- reduce the (n_samples, n_radial) posterior to a band
        via ``sigma_method`` (``"percentile"`` -> (p84 - p16)/2, or ``"std"``).
      * ``"direct"``    -- read the ``*_err`` datasets (n_e_err / T_e_err /
        T_12C6_err) directly.

    Kinetic sigma precedence (per channel, highest first)
    -----------------------------------------------------
    Each of ``ne``/``te``/``ni``/``ti`` resolves INDEPENDENTLY, and a winning
    source SHADOWS the ones below it -- it does not combine with them::

        1. sigma_profiles[chan]        explicit absolute array, kinetic grid
        2. an IDA .cdf                 ida_path, OR -- and this is the easy one
                                       to miss -- ReconstructionSource.
                                       profiles_path when it ends in '.cdf',
                                       which baseline.resolve_uncertainty
                                       adopts automatically
        3. <chan>_scalar_sigma         flat fraction x |baseline profile|

    **The scalars do nothing when an IDA file is in play.** Setting
    ``ne_scalar_sigma = te_scalar_sigma = ... = 0.0`` to get a deterministic
    (sigma=0) run is therefore a NO-OP against an IDA source: the resolved
    sigmas stay at the full operational envelope and every "deterministic"
    point is a full-sigma draw. ``resolve_uncertainty`` logs the winning
    source per channel and raises a ``UserWarning`` when a scalar you moved
    off its default is being ignored, but the only setting that actually wins
    is an explicit profile::

        n_kin = len(baseline.psi_N_kinetic)
        unc.sigma_profiles = {ch: np.zeros(n_kin)
                              for ch in ('ne', 'te', 'ni', 'ti')}

    ``sigma_jphi`` and the aux channels have no ``.cdf`` branch, so their
    scalars always apply.
    """

    # Kinetic sigma resolution, in priority order per channel (ne/te/ni/ti):
    #   1. an explicit psi_N-dependent profile in `sigma_profiles` (absolute
    #      sigma on the kinetic grid), e.g. from `synthetic_ida_sigma` or a real
    #      diagnostic envelope;
    #   2. an IDA `.cdf` (`ida_path`, or the reconstruction source's own .cdf) --
    #      its `*_err` datasets;
    #   3. a flat fractional fallback, `<chan>_scalar_sigma` * baseline profile.
    ida_path: Optional[str] = None
    sigma_mode: str = "auto"               # "auto" (by dim) | "direct" (*_err) | "ensemble"
    sigma_method: str = "percentile"       # "percentile" | "std"  (ensemble only)

    # flat fractional kinetic sigma envelopes (per channel). ni/Ti default wider
    # than ne/Te -- ion density and temperature are harder to diagnose.
    ne_scalar_sigma: float = 0.05
    te_scalar_sigma: float = 0.05
    ni_scalar_sigma: float = 0.10
    ti_scalar_sigma: float = 0.10
    # explicit psi_N-dependent ABSOLUTE sigma profiles (on the kinetic grid),
    # keyed by 'ne'/'te'/'ni'/'ti'. Present channels override the scalar/IDA path.
    sigma_profiles: dict = field(default_factory=dict)

    # Log one line per channel naming which of the three sources above actually
    # won, and the resolved peak. The precedence is silent by construction and a
    # silent win can invert the meaning of a run, so this defaults ON.
    log_sigma_sources: bool = True

    # j_phi uncertainty: flat fractional envelope on |j_phi_baseline|
    jphi_scalar_sigma: float = 0.10

    # Z_eff uncertainty: flat fractional envelope that ENABLES the zeff channel
    # by default for every source (the physically-consistent density scheme --
    # see the aux block below). Each draw perturbs Z_eff within this band and
    # DERIVES the main-ion density ni from (ne, Z_eff) via quasineutrality, so
    # ni/nz/Z_eff stay mutually consistent and ni_scalar_sigma is unused.
    # Real Z_eff (visible bremsstrahlung / CER) is ~10-20% uncertain; the 0.05
    # default is deliberately conservative -- widen it for a realistic envelope.
    # Set 0.0 to disable (Z_eff held at baseline, ni drawn independently).
    # An explicit aux_sigmas['zeff'] always overrides this.
    zeff_scalar_sigma: float = 0.05
    # Where the Z_eff envelope's MAGNITUDE comes from (the channel is enabled
    # by zeff_scalar_sigma > 0 either way):
    #   "auto"     -- highest-fidelity tier the file supports:
    #                 carbon-propagated dilution sigma (n_12C6_err / the
    #                 dilution posterior; 1.9-5.8 % of Zeff in-core on the
    #                 demo shots, sane in the SOL) > the file's VB-measured
    #                 sigma_Zeff (Zeff_err / sample spread; 8-9 % core but
    #                 44-130 % SOL, grand means to ~90 % on some shots) >
    #                 the scalar.  The Zeff-primary scheme perturbs Zeff to
    #                 move the dilution ni = ne - Z nC, and CER carbon IS
    #                 that dilution's direct measurement, hence the order.
    #   "carbon"   -- require the carbon-propagated tier; loud fallback.
    #   "measured" -- require the VB-measured envelope; loud fallback.
    #   "scalar"   -- always the flat zeff_scalar_sigma fraction (the only
    #                 behaviour before 1.4.0).
    # Only the reconstruction/IDA path is eligible for the measured tiers,
    # and only when the sigma .cdf IS the source's own profiles file: on the
    # IMAS/ida_hybrid path the Z_eff baseline is FUSE's, and pairing a FUSE
    # baseline (or a p-file one, or a different .cdf vintage named via
    # ida_path) with an IDA-measured envelope would mix channels.  That file
    # test compares RESOLVED paths (expanduser + realpath, samefile when both
    # exist), so a relative-vs-absolute, '~'-prefixed, trailing-slash or
    # symlinked spelling of the same file stays eligible.
    # NO step down this ladder is silent: each one emits a single warning
    # naming the tier chosen, the tier skipped and why (source ineligible /
    # missing dataset / invalid data), and the same record is returned as
    # resolve_uncertainty()'s "zeff_sigma_tier" metadata.
    zeff_sigma_source: str = "auto"

    # With the zeff channel active: True derives ni per draw from the drawn
    # (ne, Zeff) (sigma_ni unused); False draws ni from its own sigma_ni.
    # None = auto: True for the flat ni_scalar_sigma fallback or one IDA
    # resolution (sigma_ni kept via IDAProfiles.zeff_dne), False for any other
    # real ni envelope.
    ni_from_zeff: Optional[bool] = None

    # GPR correlation length scales (psi_N units) -- define the perturbation
    n_ls: float = 0.5                      # density
    t_ls: float = 0.4                      # temperature
    j_ls: float = 0.25                     # current density

    # --- switchboard: auxiliary perturbed profiles (source-decoupled) --------
    # Supplying a sigma profile (on psi_N_kinetic) ENABLES perturbing a named
    # profile. Baselines are auto-filled by the source where available
    # (omega_tor from the IDS, zeff computed); otherwise supply them in
    # aux_baselines (required for chi_e/chi_i and E_r, which production FUSE
    # files lack, and for the reconstruction/geqdsk path). A warning is emitted
    # if a sigma is given for a baseline that is all-zero or absent.
    # Recognised names: 'zeff' (ACTIVE -- re-enters the per-draw SWB bootstrap),
    # 'omega_tor', 'e_r', 'chi_e', 'chi_i' (passive: perturbed + stored, not GS
    # inputs). Works identically for ImasSource and ReconstructionSource.
    aux_sigmas: dict = field(default_factory=dict)         # {name: sigma(psi_N)}
    aux_baselines: dict = field(default_factory=dict)      # {name: baseline(psi_N)}
    aux_length_scales: dict = field(default_factory=dict)  # {name: GPR length} (default 0.4)


#: GenerationConfig.swb_saw_rule -> OFT boot_ops saw_rule.
SWB_SAW_RULES = {"fuse": 1, "local": 2}

# ---------------------------------------------------------------------------
# Generation + filtering
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Perturbed-bouquet sampling over the uncertainty neighborhood."""

    n_equils: int = 20
    # --- until-N-in-spec (optional) ----------------------------------------
    # None (default): draw exactly n_equils times, whatever the yield -- the
    # historical behaviour, and bit-identical to it (the sampler's draw stream
    # is untouched when this is off).
    #
    # An int: keep drawing until this many draws pass BOTH postprocess filters
    # (the CONFIGURED coil filter + LCFS deviation, at filtering.coil_filter
    # with its sigma/era/acceptance settings, and filtering.rms_max_mm), then
    # stop. n_equils becomes the initial allocation rather than the total, so a
    # shot with a 40% yield spends ~2.5x the solves that a 100%-yield shot does
    # for the same delivered ensemble.
    #
    # The verdict is computed by the SAME predicate the postprocess filters use
    # (bouquet.filtering.passes_all_filters over a coil predicate from
    # make_coil_predicate -- passes_coil_chi2 on the default chi2 filter,
    # passes_coil_spec on the legacy band -- and passes_boundary_spec /
    # boundary_deviation_mm), so the count this loop stops on is the count
    # .filter() then marks 'selected'. Draws that fail are still archived --
    # nothing is discarded, the run just doesn't stop until N have passed.
    #
    # SERIAL ONLY: run_parallel/SLURM shards cannot see each other's yield, so
    # the shard runner rejects a config with this set (bouquet.parallel).
    n_inspec_target: Optional[int] = None
    # Hard ceiling on total draw ATTEMPTS when n_inspec_target is set -- the
    # backstop against a configuration whose yield is ~0 solving forever.
    # None -> 5 * n_inspec_target (and never below n_equils). Reaching the cap
    # is a loud, non-fatal outcome: the run returns what it got and says so.
    max_total_draws: Optional[int] = None
    # The run's ONE random seed. It is consumed into a single
    # numpy.random.Generator (sampling.make_rng) that is threaded explicitly
    # into every draw site -- the GPR kinetic/aux/j_phi draws, the per-draw
    # scale_jBS sample and the per-draw l_i target sample -- so two runs with
    # the same seed, inputs and solver produce bitwise-identical archives
    # ON ONE MACHINE. Across machines the draws agree to ~1e-9, not bitwise:
    # the GP kernel is factorised by a fixed-order Cholesky, which leaves a
    # LAPACK build no discrete choices (see GPRProfilePerturber
    # .generate_profiles), so only rounding differs between builds.  The eigh
    # factorisation this replaced left basis freedom to the build; the same
    # seed differed by 1.3% between the macOS and Linux CI builds.
    # None (default) draws from fresh OS entropy: deliberately not regenerable.
    seed: Optional[int] = None
    # Label for this bouquet within the HDF5 file: draws are stored under
    # scan/<scan_key>/. Use it to keep several bouquets in one file under a
    # meaningful key (a time in ms, a beta value, ...). Default 0.
    scan_key: float = 0
    l_i_tolerance: float = 0.05            # l_i acceptance band (fraction of target)
    constrain_sawteeth: bool = False
    # When True, recompute bootstrap each draw via TokaMaker solve_with_bootstrap
    # (whose output is already TokaMaker jphi; physics module docstring),
    # overriding the baseline/FUSE j_BS. When False, keep the baseline j_BS.
    recalculate_j_BS: bool = True
    # Treat j_phi as ONE profile: no inductive/bootstrap decomposition anywhere.
    # The baseline total is taken straight from the source (the g-file's j_tor on
    # the reconstruction path, equilibrium.j_tor on the IMAS path), collapsed to
    # j_inductive = j_phi with j_BS = j_NBI = j_RF = 0, and the GPR perturbs that
    # total directly. Implies recalculate_j_BS=False, so no Sauter/Redl
    # solve_with_bootstrap call runs per draw.
    #
    # For L-mode, where the bootstrap is negligible and the Sauter edge-spike
    # machinery (classification, shelf, edge isolation) has nothing real to act
    # on and can misfire. NOT for H-mode: it discards the physical bootstrap and
    # its per-draw response, so the ensemble no longer carries bootstrap UQ.
    #
    # The sigma=0 guard is a no-op here (there is no split to reproduce) and
    # verify_sigma0_consistency() reports that instead of running a solve.
    #
    # SIGMA MEANING CHANGES: jphi_scalar_sigma is applied to whatever is being
    # perturbed, which here is the TOTAL rather than the inductive component. On
    # a baseline with a real bootstrap the same sigma is therefore a materially
    # larger absolute current perturbation, and the draws drift the boundary and
    # coils further (measured on an H-mode test case: 0/4 draws in spec at
    # sigma=0.05, vs 4/5-5/5 with the split). Expect to re-tune jphi_scalar_sigma
    # (and possibly the in-spec cuts) for this mode rather than inheriting the
    # split-mode values.
    #
    # SCOPE: this removes the per-draw Sauter calls (the N-times cost). On the
    # g-file path the bootstrap is still computed ONCE during reconstruction,
    # and that is not removable by simply skipping the solve: fit_inductive_profile
    # matches l_i by scaling the inductive against a FIXED j_BS, and with j_BS = 0
    # the cylindrical l_i proxy is scale-invariant (scaling j scales B_p, so the
    # proxy's numerator and denominator both go as k^2). The residual then never
    # brackets, _solve_ind_scale silently falls back to ind_scale = 1.0, and the
    # unmatched profile makes the downstream GS solve abort (OFT error stop 255).
    # Verified 2026-08-03. Making the skip work needs a different l_i strategy for
    # this mode, not just an early return.
    single_profile_jphi: bool = False
    jBS_scale_range: tuple = (0.99, 1.01)  # bootstrap multiplicative spread
    # Delta composition of the per-draw bootstrap (alternative to sharing the
    # near-axis smoothing): each draw's spike is built as
    #   baseline_j_BS + (SWB(perturbed) - SWB(sigma=0)),
    # both SWB terms RAW (no smoothing of perturbed profiles).  Any common-mode
    # evaluation artifact (e.g. the collapsed innermost-surface point) cancels
    # exactly, and the per-draw Sauter response passes through unfiltered.
    # Costs one extra solve_with_bootstrap call per run (the sigma=0 reference,
    # computed in the same pre-draw anchor context).  False (default) keeps the
    # shared-smoothing treatment (smooth_jbs_transition on every spike).
    jbs_delta_mode: bool = False
    # Bootstrap profile mode in solve_with_bootstrap (legacy paths). True
    # isolates the edge spike, yielding a clean positive bootstrap; False uses
    # the full SWB profile, which for FUSE equilibria carries an unphysical
    # inner negative lobe (must then be floored, leaving kinks -- see
    # baseline_jphi_caseA plots).  None (default): resolved per engine at
    # prepare_baseline() (bouquet.engine.ENGINE_DEPENDENT_DEFAULTS): False
    # under reconstruction_engine="legacy" (the validated full-profile
    # decomposition, both input types), True under "unified" (which never
    # reads it).  An explicit value is kept; one contradicting the engine's
    # validated value warns (and is refused under "unified").
    isolate_edge_jBS: Optional[bool] = None
    # How the SWB bootstrap is reconciled with the FUSE baseline on the IMAS
    # path (see run._forward_solve_imas_baseline):
    #   "diff"    : keep FUSE total; add fixed correction diff = FUSE_jBS - SWB
    #               to baseline and every draw (anchors to FUSE; risks edge
    #               misalignment when the perturbed pedestal moves).
    #   "rescale" : keep FUSE ohmic; rescale SWB by a single factor so l_i
    #               matches the source (fully self-consistent bootstrap).
    #   "ohmic"   : HYBRID. SWB bootstrap (from the kinetic source, e.g. IDA)
    #               and FUSE NBI/RF taken as-is; Ip closed on the channel
    #               named by ``closure_channel`` (default "bootstrap": rescale
    #               j_BS, factor recorded as Baseline.bs_scale; the
    #               deprecated "ohmic" channel rescales FUSE j_ohmic instead,
    #               Baseline.ohm_scale; "sawtooth_bootstrap" and "structured"
    #               are documented on ``closure_channel``). The jphi_diff
    #               equilibrium anchor is NOT applied. Use when the kinetic
    #               source has a materially different pedestal than FUSE --
    #               "diff" would erase that current change.  BASELINE-ONLY
    #               for now: generate() refuses this mode unless
    #               workflow="custom" (the draw-path sigma=0 reproduction of
    #               an ohmic-closed baseline is unverified).
    #   ``closure_channel`` is read ONLY in this mode (and only with
    #   recalculate_j_BS=True); on "diff"/"rescale" a non-default channel is
    #   refused rather than silently ignored.
    jBS_baseline_mode: str = "diff"
    #: Which channel absorbs the Ip closure in jBS_baseline_mode="ohmic":
    #: "bootstrap" (default) keeps j_inductive exactly as the source diffused it
    #: and rescales j_BS; "ohmic" is DEPRECATED (diagnostic bracket only; emits a
    #: DeprecationWarning): rescaling j_inductive alone hollows the core, lifts
    #: q0 far above the source's, loses the q=1 surface on most sawtoothing
    #: slices and yields implausible Delta' -- never feed it to a stability code.
    #: RECOMMENDED for sawtoothing discharges: "sawtooth_bootstrap" (below) --
    #: it is never worse than "bootstrap" (it degenerates to it wherever the
    #: recomputed bootstrap has no core content, e.g. the early ramp), holds q0
    #: within ~0.01 of the source's through flattop and ramp-down where
    #: "bootstrap" drifts by several times that, reproduces the q=1 surface far
    #: more often, and costs at most one extra solve.  The code default stays
    #: "bootstrap" until a mid-radius constraint exists for the high-beta_p
    #: regime (both channels flag closure_limited there -- see ip_closure).
    #: Every channel records a closure-health block in Baseline.ip_closure
    #: (raw_components_ip_mismatch_pct, f_BS_unscaled/closed, closure_limited +
    #: reasons): a slice is closure-limited when the raw components miss Ip by
    #: >10 % or the bootstrap is scaled below 0.5 -- treat its current split, and
    #: any Delta' built on it, as unvalidated regardless of channel.
    #: "sawtooth_bootstrap" is "bootstrap" plus a q0 constraint: on a sawtoothing
    #: flattop q0 ~ 1 is a robust physical fact, and the plain bootstrap channel
    #: has nothing holding q0 in place (a large s_bs down-scale removes the CORE
    #: share of the bootstrap too).  Both scales are then determined -- Ip exactly
    #: (the affine FSA measure) and q0 to first order (the on-axis current
    #: density, at frozen anchor geometry) -- by a 2x2 linear solve that costs no
    #: extra GS solve, with at most ONE Newton correction after the closed-hybrid
    #: solve.  Where the recomputed bootstrap has negligible core content it
    #: reduces to "bootstrap" exactly.
    #: "structured" replaces the scalar entirely: the multiplier on EACH source
    #: becomes a smooth radial PROFILE, s_ind(psi) and s_bs(psi), on a small
    #: basis (structured_basis below), and among all profiles that close Ip
    #: exactly it returns the one that departs least from "trust the sources"
    #: (s == 1) in a trust-weighted norm (structured_weights).  Motivation: the
    #: MSE arbiter found the true correction is radially structured AND
    #: shot-dependent -- core bootstrap EXCESS on one discharge, pedestal
    #: bootstrap DEFICIT on another -- which no single number can represent.
    #: Same constraints as the channels above (Ip exactly in the affine FSA
    #: measure; the on-axis current row whenever the sawtooth gate admits, same
    #: gate as "sawtooth_bootstrap"), solved in closed form through a small KKT
    #: system, so it costs ZERO extra GS solves like the q0 predictor, with the
    #: same at-most-ONE post-solve q0 correction.  It records effective scalar
    #: equivalents (the Ip-weighted mean of each multiplier), which satisfy the
    #: scalar closure equation exactly, so closure_health still applies.
    #: Refuses when either multiplier leaves 0.2 < s < 5 (STRICT, as in
    #: close_ip/close_ip_q0: a multiplier landing exactly on a bound is
    #: refused) anywhere on the grid or
    #: the KKT system is singular against a relative floor.
    #: "structured" also takes a SECOND global measurement, l_i
    #: (structured_li_target below), in either of two forms: HARD (imposed
    #: exactly, one more KKT row) or SOFT (structured_soft=True: Ip and l_i
    #: become Gaussian measurements with structured_ip_sigma / structured_li_sigma
    #: and the answer is the posterior mode, 8 unknowns, Gauss-Newton, still no
    #: GS solves).  Ip alone is ONE number against 2K coefficients and cannot
    #: see radial redistribution; l_i can.  Both forms take at most ONE
    #: post-solve correction, shared with the q0 corrector.
    closure_channel: str = "bootstrap"
    #: closure_channel="structured": a NAMED one-switch configuration, resolved
    #: at construction by ``utils.structured_preset_settings``.
    #:
    #: ``None`` (the field default) means "the caller expressed no preference".
    #: With ``closure_channel="structured"`` that resolves to the DEFAULT preset
    #: ``utils.STRUCTURED_PRESET_DEFAULT`` = ``"li_soft_onesided"``: the raw
    #: shipped fields are the configuration the l_i study superseded, so a bare
    #: structured channel now gets the validated one.  ``"none"``
    #: (``utils.STRUCTURED_PRESET_NONE``) DECLINES it and reproduces those raw
    #: fields exactly -- the symmetric physics ladder on the hard solver, Ip
    #: imposed exactly -- which is what a bare structured channel did before.
    #: With any other ``closure_channel`` nothing is applied unless a preset is
    #: named explicitly (naming one still fills the fields, and still does not
    #: switch the channel on).
    #:
    #: The default application is DECLINED, with a warning and no fills, when
    #: ``structured_basis`` is set: the preset's ladders are widths at the
    #: shipped basis's radii and have no meaning on another basis (the same
    #: reason ``utils.structured_default_weights`` falls back to uniform).  A
    #: preset NAMED explicitly is applied regardless -- the caller asked.
    #:
    #: The sigmas are PRIORS in relative units (fractions of the component
    #: profiles, on normalised flux), set from a study on one device with one
    #: integrated-modelling source for the inductive current.  They are not
    #: device constants; elsewhere they are a starting point, and the recorded
    #: closure-health flags are what says whether they held.
    #:
    #: ``"li_soft_onesided"`` is the recommended candidate configuration and
    #: fills, in sigma terms, ``sigma_bs = (0.50, 0.30, 0.15, 0.10)``,
    #: ``sigma_ind`` (down) ``= (0.10, 0.40, 0.40, 0.40)``, ``sigma_ind_up =
    #: (0.10, 0.10, 0.10, 0.40)`` -- recorded as ``structured_weights``
    #: (W = sigma^-2) and ``structured_sigma_ind_up`` -- together with
    #: ``structured_soft=True``, ``structured_ip_sigma_frac=0.005`` and, ONLY
    #: when ``structured_li_target`` is also set, ``structured_li_sigma=0.04``.
    #: A preset with no l_i target simply omits the l_i term.
    #:
    #: EXPLICIT SETTINGS ALWAYS WIN: a preset fills only the fields still at
    #: their dataclass default, so anything the caller set survives untouched.
    #: The one asymmetry worth knowing is ``structured_soft``, whose default is
    #: ``False`` and is therefore indistinguishable from an explicit ``False``
    #: -- a caller who wants these ladders on the HARD solver should set the
    #: sigma fields directly instead of naming the preset.  An unknown preset
    #: name is refused at construction, never ignored.
    #:
    #: A preset is a PRIOR plus a claim about the data.  It contains no
    #: acceptance criterion, convergence threshold or bound, and it does not
    #: change ``closure_channel``, which stays ``"bootstrap"`` -- the structured
    #: channel and every preset of it are opt-in.
    structured_preset: Optional[str] = None
    #: RECORDED, not set: which preset is in force after resolution (``None``
    #: when none is), written by :func:`resolve_structured_preset`.
    structured_preset_in_force: Optional[str] = field(init=False, default=None)
    #: RECORDED, not set: HOW that preset got there -- ``"explicit"`` (named),
    #: ``"default"`` (the structured channel's default preset), ``"opt-out"``
    #: (``structured_preset="none"``), ``"default-declined-custom-basis"``, or
    #: ``"unset"`` (no preset named and the channel is not structured).
    structured_preset_source: str = field(init=False, default="unset")
    #: RECORDED, not set: the config fields the preset actually filled.
    structured_preset_fields: list = field(init=False, default_factory=list)
    #: closure_channel="structured" basis, as a dict.  ``None`` selects the
    #: shipped default ``utils.STRUCTURED_BASIS_DEFAULT`` -- four peak-normalised
    #: Gaussians at psi_N = 0.15/0.45/0.75/0.95 with width 0.2, spanning core to
    #: pedestal.  Override with e.g.
    #: ``{"kind": "gaussian", "centres": [...], "widths": [...]}``;
    #: ``{"kind": "constant"}`` collapses the channel back onto a single scalar
    #: pair (that is how the tests prove it CONTAINS close_ip / close_ip_q0),
    #: and it works on its own: with ``structured_weights`` left at ``None``
    #: the default prior is derived for the basis ACTUALLY given, so a basis
    #: whose length is not the default 4 gets a uniform ladder (no prior
    #: without ``mse_data``; a sigma = 1 prior with it -- see
    #: ``structured_weights``), named as such in the record.  The physics prior below is a ladder over
    #: the DEFAULT basis's radii and has no meaning on any other basis.
    #: Peak-normalised, so a coefficient reads as "how far the multiplier moves
    #: from 1 near this radius"; the basis need not be orthogonal, since with at
    #: most two constraints on 2K unknowns the trust norm -- not the basis --
    #: selects the answer.
    structured_basis: Optional[dict] = None
    #: closure_channel="structured" trust weights, as a dict
    #: ``{"ind": (K,), "bs": (K,)}``.  ``None`` selects the shipped physics
    #: prior ``utils.STRUCTURED_WEIGHTS_PHYSICS`` (ind 100/10/3/1, bs 1/3/10/100):
    #: large W penalises deviation from 1, i.e. "trust this source here".
    #:
    #: The quantity to reason about is the WIDTH: ``W = 1/sigma^2``, with
    #: ``utils.sigma_from_weights`` the bridge, so the hard and soft solvers are
    #: handed the same prior.  ``sigma = 0`` (``W = inf``) hard-pins a
    #: coefficient to 0; ``sigma = inf`` (``W = 0``) leaves it unpenalised.  The
    #: two ladders run in OPPOSITE directions for different reasons: the
    #: inductive core is tight because the on-axis current row already pins it
    #: wherever the sawtooth gate admits one (mid-radius and mantle are left
    #: looser -- that is where the closure is meant to work), while the
    #: bootstrap is tight at the PEDESTAL, where Redl/Sauter is validated
    #: against drift-kinetic (NEO) calculations, and loose in the CORE, where
    #: the trapped fraction vanishes, the collisionality expansion is at its
    #: worst and the bootstrap current is negligible anyway.  ``utils.STRUCTURED_WEIGHTS_UNIFORM``
    #: (all 1) is the ONE documented alternative and exists to be run as a
    #: sensitivity: the difference between the two answers is the part of the
    #: result the prior -- not the data -- is holding up.  ``numpy.inf`` hard-pins
    #: a coefficient to 0.  These are a PRIOR, not a tolerance: they change
    #: which exactly-Ip-closing profile is chosen, never what "closed" means.
    #:
    #: **Scale.**  Without ``mse_data`` only the RATIOS of the weights matter
    #: (the hard closure is invariant under ``W -> c W``).  With ``mse_data``
    #: they are ABSOLUTE: ``W = sigma^-2`` in peak-normalised coefficient units
    #: trades against the chords' chi^2, so multiplying every weight by 100
    #: tightens the prior tenfold in sigma and moves the answer, and the
    #: uniform ladder is a sigma = 1 prior, not "no prior".  Write the weights
    #: as the widths you mean when MSE is on; the ladder in force is recorded
    #: as ``structured_mse_prior_sigma_*``.
    structured_weights: Optional[dict] = None
    #: closure_channel="structured": the SECOND global measurement.  Ip is one
    #: number against 2K coefficients and is blind to radial redistribution --
    #: which is exactly what the campaign found the closure gets wrong.  l_i is
    #: the other global number a magnetics reconstruction reports, and it sees
    #: precisely that.  ``None`` (default) = no l_i constraint, i.e. the channel
    #: behaves exactly as before.  Set it to the reconstruction's l_i for the
    #: slice.  The value is a PLAIN INPUT: bouquet does not fetch it, does not
    #: know which code produced it, and applies no definition offset -- whatever
    #: cross-code offset the caller's l_i carries (a magnetics reconstruction's
    #: l_i is li_1-like) must already be in this number.
    structured_li_target: Optional[float] = None
    #: closure_channel="structured": 1-sigma uncertainty on ``structured_li_target``
    #: [dimensionless l_i].  Used ONLY by the soft solver
    #: (``structured_soft=True``); the hard solver imposes the target exactly
    #: and ignores it.  Required when ``structured_soft`` and a target are both
    #: set -- a soft channel with no error bar is a hard channel with extra
    #: steps, and bouquet refuses to guess one.
    structured_li_sigma: Optional[float] = None
    #: closure_channel="structured": which l_i normalisation the target is in,
    #: spelled as TokaMaker's ``get_stats(li_normalization=...)``: ``"li_1"``
    #: (the EFIT-like one: volume-averaged B_p^2 over the LCFS-perimeter mean
    #: field) or ``"li_3"`` (ITER: 2 Bp_vol / (mu0 Ip)^2 R_axis).  Both are exact
    #: discrete forms of the SAME poloidal-field-energy identity, differing only
    #: by a geometric prefactor -- see utils' l_i module comment.
    structured_li_kind: str = "li_1"
    #: closure_channel="structured": 1-sigma uncertainty on Ip [A].  ``None``
    #: (default) = Ip is imposed EXACTLY, as every channel has always done.  A
    #: finite value turns Ip into a measurement in the soft solver (typical
    #: choice: 0.5 % of Ip, the magnetics' own accuracy).  Ignored by the hard
    #: solver, which cannot do anything but close Ip exactly.
    #:
    #: With a finite sigma the closed hybrid integrates to a POSTERIOR Ip a
    #: little away from the measurement -- that is the channel working, not an
    #: error, and the post-closure round-trip gate therefore checks the
    #: assembly against that posterior (``structured_ip_posterior``), not
    #: against Ip_target.  The distance from the measurement is recorded as
    #: ``structured_ip_measured_residual_pct`` and, in sigma units, as
    #: ``structured_residual_sigma_Ip``; beyond 1 sigma_Ip the slice picks up a
    #: closure-health flag ("soft Ip beyond 1 sigma_Ip").  It is never refused
    #: and never retried on that basis.
    structured_ip_sigma: Optional[float] = None
    #: closure_channel="structured": sigma_Ip as a FRACTION of |Ip_target|, for
    #: callers who do not know Ip when they build the config (a campaign runner
    #: reading a slice table, say).  Resolved to amps inside the closure, where
    #: Ip_target is known.  Mutually exclusive with ``structured_ip_sigma``;
    #: setting both is refused rather than silently resolved one way.
    structured_ip_sigma_frac: Optional[float] = None
    #: closure_channel="structured": use the POSTERIOR-MODE solver
    #: (``utils.close_ip_structured_soft``) instead of the hard KKT one.  The
    #: hard solver says "Ip and l_i are true"; the soft one says "they are
    #: measurements with error bars" and minimises the trust-weighted prior plus
    #: the squared z-scores.  The prior is the SAME (sigma = W^-1/2 of
    #: structured_weights, via utils.sigma_from_weights), so a hard/soft pair
    #: differs only in what is claimed about the data.  The on-axis-current row,
    #: where the sawtooth gate admits it, stays HARD in both -- the q0 pin is a
    #: topological statement, not a measurement with a sigma.
    structured_soft: bool = False
    #: closure_channel="structured": the UP side of a ONE-SIDED (asymmetric
    #: Tikhonov) prior on the INDUCTIVE multipliers, as a list of K sigmas.
    #: ``None`` (default) = the symmetric prior, byte-identical to every run
    #: made before this field existed.  When set, basis coefficient ``a_k`` is
    #: penalised with the ordinary ladder's width (``structured_weights``
    #: ind, as sigma = W^-1/2) while it is NEGATIVE and with this list's
    #: ``sigma_ind_up[k]`` while it is POSITIVE.  A tight up-side sigma
    #: therefore lets ``s_ind`` FALL freely at that radius while resisting a
    #: rise.  Applies to both solvers (hard KKT and soft posterior mode),
    #: solved by sign iteration at zero extra GS solves; a sign pattern that
    #: cycles is refused loudly, never returned.
    #:
    #: EMPIRICALLY MOTIVATED, with a supporting mechanism -- not derived from
    #: one.  Empirically, over the closure cloud the l_i-informed channels win
    #: on the over-shoot slices and LOSE on the under-shoot ones, because the
    #: MSE chords penalise inductive current ADDED at mid-radius.  The
    #: mechanism that makes that asymmetry expected: the transport model's edge
    #: T_e collapses relative to the reference kinetics (ratios 2-40), so the
    #: edge resistivity runs high, current diffuses inward faster than it
    #: should, and the source's mid-radius inductive current is more likely
    #: OVER- than under-estimated.
    #: The asymmetry was tuned on the same MSE chords that then judge it, and
    #: any result obtained with it carries that circularity caveat.
    #:
    #: This is a PRIOR, not a tolerance: it changes which closure is chosen,
    #: never what "closed" means.  ``sigma = 0`` (pinned) and ``sigma = inf``
    #: (unpenalised) decide which coefficients exist and must therefore match
    #: the symmetric ladder entry for entry; a mismatch is refused.
    structured_sigma_ind_up: Optional[list] = None
    #: closure_channel="structured": absolute l_i acceptance for the post-solve
    #: corrector -- the l_i analogue of q0_tol.  The predictor is EXACT in the
    #: coefficient algebra but runs on the FROZEN anchor geometry; the solved
    #: equilibrium's own l_i is the first time the real value is knowable.  If
    #: it lands further than this from the target, ONE analytic step is taken and
    #: its result accepted whatever it gives.  There is no iteration loop -- the
    #: cost ceiling (at most one extra GS solve per slice, SHARED with the q0
    #: corrector) is the point.  This is an acceptance threshold, not a
    #: convergence tolerance: changing it does not change what "closed" means,
    #: only how often the single correction gets spent.
    structured_li_tol: float = 0.005
    #: closure_channel="structured": how many l_i corrector steps may be spent
    #: on a slice.  DEFAULT 1 -- the shipped cost ceiling of one extra GS solve
    #: per slice, shared with the q0 corrector.
    #:
    #: Raising it to 2 enables a CONDITIONAL second step, taken only on slices
    #: whose corrected l_i still misses ``structured_li_tol``.  The first step
    #: uses the parameter-free log-gain ``utils.LI_GAIN_EXPONENT = 2``; after it
    #: there are TWO measured (row, achieved) pairs on this slice, so the second
    #: step reads the slice's OWN log-gain off them
    #: (``utils.li_gain_exponent_secant``) instead of assuming one.  That is a
    #: secant iteration with no fitted constant -- it changes the cost, never
    #: the acceptance criterion (``structured_li_tol`` is untouched).  On the
    #: campaign that motivated the gain law it would fire on ~18 % of hard
    #: slices, i.e. ~+0.18 solves/slice.
    structured_li_max_corrector_steps: int = 1
    #: closure_channel="structured": measured MSE pitch angles, as a THIRD
    #: measurement.  A plain dict in the schema of :mod:`bouquet.mse` --
    #: per-chord ``R``, ``Z``, ``tgamma``, ``sigma``, ``weight`` (the fit
    #: weight, folded into ``sigma_eff = sigma/sqrt(weight)``), ``A1``..``A4``,
    #: optionally ``A5`` + ``Er`` (the E_r term is then carried by the forward
    #: model) or ``er_corrected=True`` (``tgamma`` already E_r-corrected
    #: upstream; the forward model then carries no E_r term), and the REQUIRED
    #: scalars ``ip_sign`` / ``bt_sign`` (+1 or -1: the directions of Ip and
    #: B_t in the A-coefficients' right-handed (R, phi, Z) frame -- the field
    #: orientation is a stated convention, never fitted; see
    #: ``bouquet.mse``).  With neither ``Er`` nor ``er_corrected=True`` the
    #: model takes E_R = 0, which in a rotating plasma BIASES the fit: to
    #: first order it reads B_Z + (A5/A1) E_R as B_Z at every chord, a
    #: systematic (not random) shift of the fitted current profile's shape
    #: that the closure absorbs into s_ind/s_bs -- warned and recorded
    #: (``structured_mse_er_neglected``, ``structured_mse_er_terms``).  A6
    #: (E_Z) is never used; a non-zero A7 (the denominator E_R coefficient)
    #: with an applied E_r is refused.  ``None`` (the
    #: default) adds nothing and leaves the structured closure exactly as it
    #: was.  When given, ``chi2_MSE = sum_k ((tan_gamma_pred - tgamma) /
    #: sigma_eff)^2`` joins the structured objective; tan(gamma) is linearised
    #: in the coefficients by finite differences on SOLVED equilibria (one GS
    #: solve per free coefficient) and the closure re-solved
    #: ``structured_mse_steps`` time(s) -- see
    #: ``utils.structured_mse_outer``.  With MSE on, the structured trust
    #: weights are an ABSOLUTE ``sigma^-2`` prior (see ``structured_weights``:
    #: their overall scale now matters, and a uniform ladder is sigma = 1).
    #: On any configuration that never runs the structured closure (another
    #: channel, a g-file source, jBS_baseline_mode != "ohmic",
    #: recalculate_j_BS off) a supplied block -- or any non-default
    #: ``structured_mse_*`` knob -- is REFUSED at prepare_baseline() rather
    #: than silently ignored; ``workflow='custom'`` downgrades that to a WARN.
    mse_data: Optional[dict] = None
    #: closure_channel="structured": REFUSE (raise) when ``mse_data`` is absent
    #: or unusable (fewer than ``structured_mse_min_chords`` weighted finite
    #: chords, a malformed block, a double-counted E_r), or when the
    #: MSE-constrained closure cannot be delivered.  Default False keeps the
    #: channel's previous behaviour: no block -> no MSE term; a block that is
    #: unusable -> no MSE term, WARNED and recorded (``structured_mse_status``)
    #: -- never silently.  Setting it on a configuration that never runs the
    #: structured closure is refused, and ``workflow='custom'`` does NOT
    #: downgrade that (a required constraint cannot be waived).
    structured_mse_required: bool = False
    #: closure_channel="structured" + ``mse_data``: forward-difference step of
    #: the tan(gamma) Jacobian, in coefficient units.  A numerical-
    #: differentiation step, not a tolerance; the linearisation residual it
    #: leaves is measured and recorded on every slice.
    structured_mse_fd_step: float = 0.02
    #: closure_channel="structured" + ``mse_data``: chord-method steps (closure
    #: re-solve + equilibrium solve) after the Jacobian.  1 = one re-solve;
    #: 2 refreshes the linearisation offset from the first step's solve and
    #: re-solves once more (same Jacobian).  Cost only -- no acceptance moves.
    structured_mse_steps: int = 1
    #: closure_channel="structured" + ``mse_data``: an OPTIONAL systematic
    #: uncertainty on tan(gamma), added in quadrature to every chord's
    #: ``sigma_eff``.  Default 0.0 (the stated uncertainties are used as
    #: they are); it is a statement about the data, recorded with the result.
    structured_mse_sigma_sys: float = 0.0
    #: closure_channel="structured" + ``mse_data``: the fewest usable chords
    #: the block must carry (``bouquet.mse.MSE_MIN_CHORDS``, the same floor
    #: the scalar MSE arbiter refuses below).
    structured_mse_min_chords: int = 4
    #: closure_channel="sawtooth_bootstrap" gate: the q0 pin is only well-founded
    #: where sawteeth justify it.  Admitted when the source's sawtooth model is
    #: active at the slice (core_sources identifier index 701 carrying non-zero
    #: j_parallel) OR the source's OWN axis |q0_dd| is at/below this value;
    #: otherwise the slice falls back to the plain "bootstrap" channel with a
    #: printed note (never silently -- on reversed shear / early ramp the
    #: source's own q0 is model-dependent and pinning to it is not obviously
    #: better than bootstrap).  The comparison is on |q0_dd|, not on q0_target:
    #: the TokaMaker-estimator target reads systematically lower and gating on
    #: it admitted idle-sawtooth ramp slices; q0_target is used only when the
    #: source carries no axis q (recorded as q0_gate_basis).
    q0_gate: float = 1.1
    #: Absolute q0 acceptance for closure_channel="sawtooth_bootstrap".  The
    #: predictor is first-order (q0 ~ 1/j_phi(0) at frozen geometry); if the
    #: solved q0 lands further than this from q0_ref, ONE analytic Newton step
    #: along the Ip-closed manifold is taken and its result accepted whatever it
    #: gives.  There is no iteration loop -- the cost ceiling is the point.
    #: Under the self-consistent loop the same band is the acceptance flag on
    #: the delivered equilibrium and, with jbs_loop_q0_corrector=True, also a
    #: convergence criterion of the loop (the value itself is unchanged).
    q0_tol: float = 0.01
    # Fix B: when the recon-anchor's equilibrium l_i is already within the band,
    # accept the anchor and skip find_optimal_scale + the corrective iteration
    # (which otherwise overshoot l_i and drift degenerate coils off baseline).
    accept_anchor_inband: bool = False
    # Fix C: GPR-perturb the inductive in the recon-anchor itself (then accept).
    # This is an ALTERNATIVE to the standard flagship l_i loop, which ALREADY
    # GPR-perturbs j_inductive (rejection sampling, TokaMaker_interface.py ~1842);
    # BOTH consume sigma_jphi/j_ls and BOTH satisfy the "perturb all profiles incl.
    # j_inductive" rule. Difference: Fix C accepts the perturbed-in-anchor draw and
    # skips find_optimal_scale + the corrective (which can homogenize draws).
    # Default False = the standard flagship loop -- the mechanism geqdsks used
    # successfully (high draw completion). Set True for FUSE/IMAS diff mode (avoids
    # the matching-loop homogenization: full draw yield on the reference IMAS case);
    # but it DROPS draws on stiff high-l_i geqdsks via band-conditioning rejection, so it is
    # NOT a good geqdsk default. FUSE runs opt in explicitly. (Only the PIN_JPHI /
    # DIFF_BS-at-sigma=0 diagnostic modes actually freeze j_ind -- those are the
    # backend-test exceptions to the rule.)
    # None (default): resolved per engine at prepare_baseline()
    # (bouquet.engine.ENGINE_DEPENDENT_DEFAULTS): under
    # reconstruction_engine="legacy" False for a g-file source and True for
    # an IDS source (diff+C), under "unified" False (never read).  An
    # explicit value is kept; one contradicting the engine's validated value
    # warns (and is refused under "unified").
    perturb_jind_in_anchor: Optional[bool] = None
    # Escape hatch for the per-path workflow guard (run._validate_workflow):
    # from_imas/from_geqdsk auto-apply their validated workflow (IMAS=diff+C,
    # geqdsk=standard l_i loop) and generate() raises on a known-bad combo
    # (geqdsk+Fix C, IMAS without Fix C, or jphi_scalar_sigma<=0 which freezes
    # j_inductive). Set True ONLY for deliberate backend tests / experiments to
    # bypass the guard (it will warn instead of raise).
    allow_unsafe_workflow: bool = False
    # Named workflow preset -- a clearer facade over the per-path flag combos the
    # guard enforces (F4). "auto" (default): resolve per source type at
    # generate() (what from_geqdsk/from_imas hardcode -- geqdsk = standard l_i
    # loop, imas = diff+C). "geqdsk-standard" / "imas-diff-c" name those two
    # explicitly (and assert the source matches). "custom": leave the flags as-is
    # and downgrade the guard to a warning -- subsumes allow_unsafe_workflow,
    # which stays as a deprecated alias (True is treated as "custom").
    workflow: str = "auto"
    # Fail-fast (IMAS path) if the source carries more thermal pressure than the
    # reconstructed total used by the solve -- a dropped impurity species or a
    # fast-ion channel left out. io.imas.read_imas_baseline raises naming the
    # missing component + magnitude; the diff anchor then absorbs only the small
    # residual. Set True ONLY for backend tests to bypass (warns instead).
    allow_incomplete_pressure: bool = False
    # Anchor the total j_phi to equilibrium.profiles_1d.j_tor (the GS-consistent
    # current GPEC reads, carrying the pedestal current) instead of
    # core_profiles.j_tor (the transport parallel-current sum, which differs from
    # ~q=2 outward). Adds a FIXED jphi_diff = equilibrium.j_tor - core_profiles.j_tor
    # to the baseline AND every draw (mirroring jBS_diff/p_diff); the SWB bootstrap
    # + perturbed j_ind ride underneath so the edge current still carries Sauter UQ.
    # IMAS path only (geqdsk reads its total from the g-file). Default True.
    anchor_jtor_to_equilibrium: bool = True
    # Source of the baseline kinetic profiles on the IMAS path:
    #   "fuse"       -> FUSE core_profiles ne/Te/Ti (default; original behaviour)
    #   "ida_hybrid" -> ne/Te/Ti/ni/Z_eff/omega_tor from ImasSource.ida_path at
    #                   ImasSource.ida_time (resampled onto the FUSE grid; Z_eff is
    #                   IDA's unless ImasSource.zeff_from_fuse); currents/equilibrium/
    #                   p_fast/anchors stay FUSE, at ImasSource.time.
    kinetic_source: str = "fuse"
    # Anchor the solve thermal pressure to equilibrium.pressure via the fixed
    # p_diff = equilibrium.pressure - p_reconstructed offset. With FUSE kinetics
    # this re-adds the carbon/fast/GS residual the recon omits. With IDA-hybrid
    # kinetics it would force the (trusted) IDA pressure back onto the FUSE total,
    # which we do NOT want -- so default False. When False, p_diff is None and the
    # solve pressure is e(ne*Te+ni*Ti) + impurity + p_fast with no equilibrium anchor.
    anchor_pressure_to_equilibrium: bool = False
    # Corrective j_phi iteration on the IMAS baseline forward solve. The single
    # jphi-linterp solve imposes the requested j_phi(psi_N) with pre-solve
    # geometry, so the ACHIEVED FSA j_phi drifts from the request once psi
    # converges (measured +3.2-3.4% over psi_N 0.2-0.9 on the reference DIII-D
    # case, appearing as l_i +3.4% and q(psi_N) -5% vs the equilibrium IDS).
    # True re-uses the recon path's _corrective_jphi_iteration to drive the
    # achieved profile onto the anchor target. Opt-in while being validated.
    imas_corrective_jphi: bool = False
    # Floor the SWB bootstrap at 0 (drop unphysical negative excursions) before
    # it enters j_phi, in both the baseline and every draw. Default False: only
    # needed for the isolate_edge_jBS=False full-profile mode (which carries an
    # inner negative lobe). With isolate_edge_jBS=True the spike is
    # already ~clean, so flooring is redundant -- and it REGRESSED a stiff
    # high-l_i case (clipping its isolate-edge spike drove yield to 0).
    floor_j_BS: bool = False
    # Keyword options forwarded to every solve_with_bootstrap call (keys
    # validated in __post_init__).  The unified engine reads only the
    # edge-taper keys (taper off by default).  Replaces swb_iterations.
    bootstrap_kwargs: dict = field(default_factory=dict)
    # GS iteration cap for generate()'s draw loop (TokaMaker_interface.
    # DrawSolveGuard).  None (default) keeps the solver's own setup cap, so
    # nothing changes unless it is set; a solve that hits a cap still fails
    # the draw exactly as before (no re-solve at another tolerance).  Every
    # draw solve that raises is recorded either way: per draw in
    # diagnostics['solve_failures'], on Bouquet.solve_failures, and in one
    # printed "[draw-solves]" line.
    draw_solve_maxits: Optional[int] = None
    # SWB inputs on the IMAS path (baseline split, draws, sigma=0 check):
    # "source" seeds SWB with the source's j_inductive and holds the rest of
    # its current (NBI + RF + other) fixed via jphi_fixed; "generic" uses the
    # (1 - s^1.5)^1.5 seed and no fixed current.  g-file paths: "generic".
    swb_seed: str = "source"
    # The solve method, one of SOLVE_METHODS: "legacy", "swb"
    # (solve_with_bootstrap is the baseline and every draw; IMAS sources) or
    # "engine" (the unified engine).  None: derived from imas_baseline /
    # reconstruction_engine; set, it sets them (resolve_solve_method).
    solve_method: Optional[str] = None
    # IMAS baseline + draws: "closure" (legacy) or "swb": solve A at the setup
    # coil reg, solve B with the strong reg toward A's coils is the baseline,
    # and every draw is solve B with resampled kinetics and inductive seed
    # (bouquet.swb).
    imas_baseline: str = "closure"
    # imas_baseline="swb": weight of solve B's (and every draw's) coil reg toward
    # solve A's coils (#VSC toward 0 at 1.0); 1e4 can send SWB to a wrong
    # equilibrium.
    swb_coil_reg_weight: float = 1.0e3
    # Taper j_phi to 0 from this psi_N to the LCFS in every SWB solve (OFT
    # taper_edge_jBS; None: off): finite edge current next to a near-degenerate
    # second null can leave the Picard on a 2-cycle.
    swb_edge_taper_psi0: Optional[float] = 0.999
    # imas_baseline="swb" sawtooth q reset inside SWB (OFT saw_q_s; None = off):
    # Baseline.j_sawteeth becomes SWB's jphi_saw input.  swb_saw_dq / _tol /
    # _ramp map onto OFT's saw_dq / saw_tol / saw_ramp, swb_saw_rule onto
    # saw_rule ("local": each dip below swb_saw_q; "fuse": axis to the mixing
    # radius).  saw_relax goes via bootstrap_kwargs.
    swb_saw_q: Optional[float] = None
    swb_saw_dq: float = 0.03
    swb_saw_tol: float = 1.0e-4
    swb_saw_ramp: float = 0.01
    swb_saw_rule: str = "local"
    # --- self-consistent bootstrap loop (bouquet.jbs_loop) -------------------
    # True (default): j_BS is re-evaluated (physics.evaluate_jBS: Redl on the
    # caller's own psi_N grid and the CURRENT equilibrium's geometry) inside a
    # relaxed outer loop closure <-> GS solve <-> Redl, in every path that
    # builds a j_phi containing a bootstrap (the IMAS baseline in every
    # jBS_baseline_mode and closure channel incl. the structured/MSE closures,
    # every draw incl. Fix C and the standard l_i loop, the sigma=0 check and
    # the geqdsk reconstruction), until the residuals below hold on TWO
    # CONSECUTIVE passes.  See docs/physics-notes.md, "Self-consistent
    # bootstrap".
    # False: the LEGACY frozen bootstrap -- computed once
    # (solve_with_bootstrap on its own auxiliary equilibrium) and then only
    # rescaled; byte-identical to every run made before these fields existed.
    # Required with single_profile_jphi=True or recalculate_j_BS=False (no
    # bootstrap to iterate; the combination is refused, never silently
    # downgraded).  A stored config that predates the field (an old archive's
    # config_json) loads with False -- the behaviour it was produced with; see
    # BouquetConfig.from_dict.
    jbs_self_consistent: bool = True
    # Initial guess of the loop: "anchor" = evaluate_jBS on the anchor
    # equilibrium (the source's own total current and full pressure); "swb" =
    # the legacy solve_with_bootstrap result (A/B only).  The fixed point does
    # not depend on it.  (Only a seed of the Redl loop: not solve_method="swb".)
    jbs_init: str = "anchor"
    # Convergence tolerances (all active ones must hold on two consecutive
    # passes):
    #   jbs_rtol_j  -- current-weighted L2 residual of the j_BS profile
    #   jbs_rtol_Ip -- residual of the bootstrap current integral, over Ip
    #   jbs_tol_li  -- absolute change of the solved l_i between passes
    #   jbs_tol_q0  -- absolute change of the solved q0 between passes (only
    #                  where an axis row / q0 target is active)
    jbs_rtol_j: float = 1.0e-3
    jbs_rtol_Ip: float = 1.0e-4
    jbs_tol_li: float = 1.0e-3
    jbs_tol_q0: float = 2.0e-3
    # Pass ceilings (limits, not tolerances: convergence is still every active
    # criterion on two consecutive passes).  jbs_max_passes: the baseline /
    # reconstruction loop.  jbs_max_passes_draw: EACH loop of a draw (its
    # anchor loop, every l_i-match candidate's Gauss-Seidel coupling, every
    # Fix C resample).  jbs_max_passes_post_homotopy: the passes a draw may
    # take at the tight coil stage when Redl on the post-homotopy equilibrium
    # misses its bootstrap (the check itself is not a pass; with the
    # two-consecutive rule a stage whose first pass misses needs >= 3).
    # The post-homotopy ceiling is 6 (was 4): an owner-approved change of a
    # pass ceiling, on measurement -- no draw of the convergence study
    # needed more than 5, and every draw the ceiling of 4 rejected converged
    # on the next pass; 6 is the measured need plus one pass.  No tolerance
    # and no criterion moved; the stage does not exist with
    # jbs_self_consistent=False, so that path is untouched.
    # The reconstruction ceiling is 12 (was 8): owner-approved 2026-10-07 on
    # measurement.  With r_j / r_I measured against the bootstrap the pass
    # actually solved (not the iterate) a healthy q0-pinned loop needed nine
    # passes (pass 8 met every criterion, pass 7 missed r_I by 17 %, so the
    # two-consecutive rule needed one more).  A ceiling is a limit, not a
    # tolerance: a run that converges in fewer passes is unchanged; one that
    # needs more now gets them instead of failing.  Matches the draw ceiling.
    jbs_max_passes: int = 12
    jbs_max_passes_draw: int = 12
    jbs_max_passes_post_homotopy: int = 6
    # Under-relaxation omega of the bootstrap: j_BS <- (1-omega) j_BS + omega
    # Redl.  Held fixed, and halved (floor jbs_loop.JBS_RELAX_FLOOR = 0.25)
    # only on SUSTAINED growth of r_j: growth on jbs_relax_halve_on
    # consecutive passes (1 = halve on every growth).  A single growth event
    # is the forced response of the oscillating closure <-> geometry mode, not
    # divergence.  Three growing passes at the floor abort the loop.
    # Relaxation changes the path, not the fixed point.
    jbs_relax: float = 0.7
    jbs_relax_halve_on: int = 3
    # Under-relaxation beta of the SOLVED current: from the second pass on the
    # equilibrium is solved with (1-beta) x the previous pass's solved j_phi +
    # beta x the closure's new j_phi.  Each pass closes on the previous
    # equilibrium's geometry, so closure and geometry form an oscillating mode
    # (l_i swings back by a fraction g ~ -0.5 per pass) that omega does not act
    # on; beta ~ 1/(1-g) damps it.  1 = no relaxation.  Path only: at the fixed
    # point the solved and the closure current agree, and the loop record
    # carries beta and the per-pass gap ||j_solved - j_closure|| / ||j_closure||
    # (recorded, not gated).  Applied where a pass solves one assembled j_phi:
    # the IMAS baseline loop (every jBS_baseline_mode / closure channel), the
    # sigma=0 check and the draws' anchor loops (Fix C, standard anchor,
    # post-homotopy).  NOT applied (the record says so) in the standard
    # draw's l_i-match coupling and the geqdsk reconstruction, whose passes
    # re-run multi-solve fits, nor in the MSE chord steps, whose
    # linearisation is centred on the closure's own current.
    jbs_relax_current: float = 0.7
    # OPT-IN, under evaluation (default False = the loop exactly as without
    # this field).  True ADDS one convergence criterion, on the same two
    # consecutive passes as the others: the closure-half residual
    # ||jc_k - js_k-1||_w / ||jc_k||_w -- the current the previous pass
    # SOLVED against the current this pass's closure composes on the
    # equilibrium that solve produced (with the bootstrap evaluated on it),
    # computed directly from the two arrays -- must be <= jbs_rtol_j (same
    # current-weighted norm, same tolerance; nothing is relaxed and the pass
    # ceilings are unchanged).  It is measured one pass late, so pass 1 cannot
    # count.  Steps that do not take the relaxer report their solved current
    # (meas["j_solved"]); a step that reports neither stops the loop at once
    # under the failure policy.  It bounds the RESIDUAL (the delivered
    # current's consistency), not the distance to the fixed point on a slow
    # monotone mode.  Not applied by the MSE chord stage or the post-homotopy
    # / post-corrective acceptance checks, which have their own tests.
    jbs_gate_current_residual: bool = False
    # Non-convergence: "raise" (default) -> jbs_loop.JBSNotConverged carrying
    # the residual history; "flag" -> keep the last iterate, record
    # jbs_converged=False plus a closure_limited reason (drivers then exclude
    # the slice from verdicts).  A draw whose loop does not converge is a
    # FAILED draw in either mode.
    jbs_loop_on_fail: str = "raise"
    # The on-axis safety-factor pin under the loop (closure_channel=
    # "sawtooth_bootstrap", or "structured" with the sawtooth gate admitting
    # the axis row; the IMAS baseline in jBS_baseline_mode="ohmic").
    # False (default): record-only, bit for bit the behaviour before the
    # field existed -- every pass closes with the axis row held at the
    # anchor's requested axis current, no Newton step is taken, and the q0
    # residual |q0 - q0_target| on the delivered equilibrium is only recorded
    # and flagged against q0_tol.  True: the pin ACTS inside the loop -- the
    # axis row is moved once per pass from the q0 MEASURED on that pass's
    # solved equilibrium (j_ref0 <- j0_solved * q0 / q0_target, the legacy
    # structured corrector's update applied per pass; jbs_loop.AxisRowPin),
    # and convergence ADDITIONALLY requires |q0 - q0_target| <= q0_tol (the
    # unchanged q0_tol) on two consecutive passes, next to the jbs_tol_q0
    # step criterion and the bootstrap criteria; the MSE chord stage keeps
    # the pin acting.  So the delivered equilibrium -- the last one solved,
    # whose bootstrap was the last evaluated -- meets the loop criteria AND
    # the q0 target.  A joint iteration that does not converge within the
    # unchanged pass ceiling fails exactly as the loop fails (raise, or flag
    # with jbs_loop_on_fail="flag"), with the q0 residuals in the record.
    # Records: ip_closure["jbs_loop"]["q0_pin"] (per pass: axis row, solved
    # axis current, q0, residual, residual / q0_tol, next row) and, in
    # ip_closure, q0_pin_acted / q0_pin_n_row_updates / q0_residual_over_tol.
    # The l_i row is NOT covered (it stays held at its target either way).
    # No effect with jbs_self_consistent=False (the legacy path's own
    # corrector already takes its Newton step) or on a channel/slice without
    # an axis row.
    jbs_loop_q0_corrector: bool = False
    # --- the unified reconstruction engine (bouquet.engine; docs/engine.md) --
    # "unified" (the DEFAULT since 2026-10-06, owner decision): ONE
    # reconstruction loop for both input types -- parallel current
    # components composed on the latest solved geometry, the structured
    # closure with rows + discrepancies, one GS solve per pass, the delivery
    # solve checked on every row.  The engine builds the baseline AND runs
    # the draws (generate(), verify_sigma0_consistency();
    # bouquet.engine_draws, docs/engine.md).  "legacy" (opt-in; the default
    # until 2026-10-06): prepare_baseline() runs the existing
    # reconstruction / IMAS baseline paths and the legacy draws.  A stored
    # config without the field (it predates the engine) loads as "legacy",
    # with a warning (BouquetConfig.from_dict).  The engine_* fields below
    # configure "unified" only and are REFUSED when changed with "legacy"
    # (they would do nothing); the legacy-path fields the engine never reads
    # are refused under "unified" (bouquet.engine.ENGINE_UNREAD_LEGACY_FIELDS).
    reconstruction_engine: str = "unified"
    # "structured" (4-Gaussian basis, li_soft_onesided priors; the default),
    # "structured_uniform" (the same basis and rows under the documented
    # uniform ladder utils.STRUCTURED_WEIGHTS_UNIFORM: the prior-sensitivity
    # run), "bootstrap_scalar" (s_bs constant; rows Ip),
    # "sawtooth_two_scalar" (s_ind, s_bs constants; rows Ip, q0) or
    # "two_scalar_li" (s_ind, s_bs constants; rows Ip, l_i: the legacy
    # secant's two-scalar l_i family as a named closure).
    engine_preset: str = "structured"
    # The measurement rows: "Ip" (always), "l_i", "q0" (on-axis safety
    # factor; active only where the sawtooth gate admits it), "mse" (needs
    # mse_data with E_r-CORRECTED pitch angles, er_corrected=True).
    engine_rows: tuple = ("Ip", "l_i")
    # Per-pass delivery correction (request minus achieved, one Newton step
    # per pass, part of the state a draw inherits).  Default off; the default
    # is to be decided after the solver's jphi-linterp defect is fixed.
    engine_delivery_correction: bool = False
    # How the MSE Jacobian is formed: "fd_chord" (finite differences once
    # at convergence, held fixed -- the legacy chord stage's treatment) or
    # "fd_broyden" (the same finite differences, then Broyden updates every
    # pass; the design note's recommendation).  Either way the linearisation
    # offset is refreshed from every solve.  Default "fd_broyden" ->
    # "fd_chord" 2026-10-02 (owner-approved): on the second MSE pass the
    # coefficients barely move while tan-gamma still changes with the
    # relaxing bootstrap and geometry, so the rank-one update shifted the
    # Jacobian by 17-32 % and the loop spent 4-5 passes recovering; the
    # fixed Jacobian converged every MSE case (Broyden 8 of 10), was never
    # slower, agreed within |dl_i| <= 5e-4 and sat 3-5 % from a fresh
    # Jacobian against 16.5 %.
    engine_mse_jacobian: str = "fd_chord"
    # The chord chi^2 / N of a delivered MSE fit (unified engine; the raw
    # E_r-corrected chords with the stage's own weights) above which the fit
    # is FLAGGED "mse_chi2_per_chord_high" (closure-limited reason, warning,
    # engine record "mse").  A FLAG ONLY, never an acceptance criterion: the
    # fit is delivered either way.  10.0 is owner-approved 2026-10-07 (flag
    # only; introduced 2026-10-07 with the review's chi^2 / N record); the
    # delivered chi^2 above the no-MSE reconstruction's is flagged
    # separately ("mse_worse_than_without") whatever this value.  A finite
    # number > 0; refused non-default under reconstruction_engine="legacy".
    mse_chi2n_flag: float = 10.0
    # Under-relaxation r of the l_i row's discrepancy update between passes
    # (the reconstruction only; draws carry no l_i row):
    #   d_k = (1 - r w) d_k-1 + r w [l_i(E_k+1) - l_i_model(js_k; G_k+1)],
    # w the loop's bootstrap omega (the first update, from d = 0, takes
    # w = 1).  0 < r <= 1.  1.0 (default) is the update before the setting
    # existed, bit for bit.  r < 1 shrinks the row's per-pass gain by r: a
    # solver-side remedy for a row that overshoots (a period-2 oscillation);
    # it changes the PATH only -- no tolerance, criterion, ceiling or target
    # moves, and the fixed point (d = the measured discrepancy) is the same.
    # The q0 row (AxisRowPin) and the MSE chords (Broyden) are not affected.
    engine_li_row_relaxation: float = 1.0
    # How the IDS adapter forms the inductive current (IDS sources only;
    # bouquet.adapters.IdsAdapter's ``inductive``).  "residual" (default,
    # owner decision 2026-10-02): the inductive current IS the parallel
    # residual j_total - j_bootstrap - sum(driven) by definition; the
    # source's j_ohmic is a cross-check, compared and stamped (no threshold,
    # no warning); a source without j_total / j_bootstrap is refused.
    # Evidence: on self-consistent sources residual and j_ohmic are
    # indistinguishable (|dl_i| <= 1.8e-3); on locally inconsistent ones the
    # residual is closer to the source's own <j_phi>.  Explicit options:
    # "j_ohmic" (the source's, warning when it misses the residual by more
    # than the adapter's 2 % tolerance) and "auto" (j_ohmic, or the residual
    # with a warning when j_ohmic is absent or misses by more than 2 %).
    # The consistency numbers are stamped in the contract's provenance
    # whichever is used.  A non-default value is refused with a g-file
    # source (it would have no effect there).
    engine_ids_inductive: str = "residual"
    # The normalisation radius of the IDS l_i row's TARGET (the source's
    # equilibrium global_quantities.li_3).  The IMAS data dictionary gives
    # li_3 no normalisation radius: IMAS.jl (and so FUSE) writes it with the
    # geometric radius R_geo = (R_out + R_in)/2 of the boundary, TokaMaker's
    # measurement (utils.li_achieved) uses the magnetic axis R_axis.
    # "auto" (default): recompute li_3 from the source's own equilibrium
    # (profiles_1d gm2, dpsi_drho_tor, dvolume_dpsi; the axis; r_outboard /
    # r_inboard) with each radius and take the one reproducing the stored
    # value within adapters.LI3_RADIUS_MATCH_TOL (0.5 %); neither -> REFUSED,
    # naming both; a source that cannot be recomputed -> the target is used
    # unrescaled (as before the setting), printed, recorded "undetermined".
    # A stored IMAS unified config without the field loads as "axis" (what
    # it ran with), with a warning.
    # "geometric" / "axis": stated.  The target is rescaled to the
    # measurement's radius by R_src / R_axis; choice, ratios and factor are
    # recorded (contract rows/provenance, Baseline.li_metrics).  Refused
    # non-default with a g-file source (no effect there).
    imas_li3_radius: str = "auto"
    # --- the draws on the engine (Stage 3; bouquet.engine_draws) ------------
    # A draw holds the reconstruction's coefficients x* and closes ONLY the
    # Ip row (a scalar amplitude on the inductive, in the exact measure).
    # True additionally keeps the reconstruction's q0 row acting in every
    # draw (bouquet.jbs_loop.AxisRowPin on a second scalar, the bootstrap
    # amplitude) -- for sawtoothing discharges; needs "q0" in engine_rows.
    engine_draw_q0_row: bool = False
    # The coil homotopy stage after a draw's loop (then the existing
    # post-homotopy bootstrap check with its saturation guard).  True is
    # what generate() does today; False skips the homotopy and measures the
    # coil drift of the loop's own delivered draw.
    engine_draw_homotopy: bool = True
    # After a draw's FIRST loop solve, re-evaluate the anchor's Redl
    # increment on that solved geometry (the draw's kinetics and pressure)
    # and restart the loop's bootstrap from it instead of blending toward
    # the start computed on the reconstruction's geometry.  Changes the
    # loop's PATH only (no criterion, tolerance or ceiling; zero extra
    # solves).  False (default) is the behaviour before the setting existed.
    engine_draw_bootstrap_refresh: bool = False
    # The Grad-Shafranov iteration cap on EVERY solve inside an engine draw:
    # the loop (its anchor and passes), each coil-homotopy stage, a
    # homotopy rollback re-solve and the post-homotopy bootstrap passes (and
    # the zero-perturbation draw of verify_sigma0_consistency).  None keeps
    # the solver's own cap.  A capped HOMOTOPY STAGE is a failed stage like
    # any other: it rolls back to the last good (looser) stage, and the draw
    # is rejected only when there is none (homotopy_maxits); a capped loop
    # solve, rollback re-solve or post-homotopy pass rejects the draw
    # (perturb_failed / anchor_solve_failed, homotopy_maxits,
    # post_homotopy_maxits).  Every capped solve is recorded (stage,
    # iterations, seconds, outcome).  A solve that converges below the cap
    # is untouched.  The engine refuses draw_solve_maxits (the legacy draws'
    # cap); the legacy path never reads this field.
    engine_draw_solve_maxits: Optional[int] = 100
    # --- the pressure handed to the GS solver (bouquet.edge_pressure) --------
    # Both settings act on EVERY solve of the package (legacy paths, the
    # unified engine, the draws, the zero-perturbation checks).
    # edge_pprime_pin defaults to the behaviour before it existed;
    # separatrix_pressure does NOT (default "offset" since 2026-10-02, an
    # owner-approved physics change; "legacy" is the behaviour before).  The
    # solver's resulting uniform P' rescale is recorded as p_scale
    # (edge_pressure.P_SCALE_DEFINITION) on every delivered state.
    # edge_pprime_pin: True sets the last node of P' (psi_N = 1) to zero, so
    # P' ramps to zero across the final grid interval; False keeps the
    # profile's own derivative there.  False moves the edge current between
    # the P' and FF' terms and un-zeroes the pressure-driven current at the
    # boundary (a PHYSICS change: edge current and q95 move).  Measured on
    # the synthetic g-file example (unified engine): pressure-driven current
    # at the boundary 0.006 -> 0.018 MA/m^2, <j_phi> at the last node 0.076
    # -> 0.115 MA/m^2 with the FF' term changing sign there, q95 +0.002;
    # l_i, the core and the iteration counts unchanged.
    edge_pprime_pin: bool = True
    # separatrix_pressure: "legacy" passes the full axis pressure as the
    # solver's target (the solver's pressure is zero at the boundary, so a
    # non-zero input p_sep inflates P' everywhere by p_axis/(p_axis-p_sep)
    # and beta / W_MHD are those of a different profile).  "offset" passes
    # p_axis - p_sep and adds p_sep back wherever pressure, beta or W_MHD is
    # reported or delivered (records carry both frames; written g-files
    # carry the full pressure).  A PHYSICS change when p_sep != 0: P' moves
    # by the factor (p_axis - p_sep)/p_axis.  Default "legacy" -> "offset"
    # 2026-10-02 (owner-approved physics change: on real cases it narrowed the
    # full-frame beta_N / W_MHD gap to the input, l_i / q / cost unchanged);
    # "legacy" restores the pre-change numbers.
    separatrix_pressure: str = "offset"
    # Coil handling (homotopy-based). The inverse solve drifts coils within
    # coil_drift, stepped through homotopy_passes = list of (F_tol, VSC_tol)
    # stages that tighten loose->tight (each warm-starts the next). A single
    # direct solve at the tight target is usually infeasible -- the schedule is
    # what makes ±1% coils reachable.
    coil_drift: float = 0.01
    # Optional HARD inequality bounds at +/- coil_drift_hard_factor*coil_drift,
    # enforced in every solve (not just the soft homotopy). None = soft only.
    # Backstop for degenerate-coil drift (e.g. F9B) at the cost of possible
    # boundary error when the reshaped equilibrium genuinely wants the coil off.
    coil_drift_hard_factor: float = None
    homotopy_passes: list = field(
        default_factory=lambda: [(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]
    )
    diagnostic_plots: bool = False

    # Live-equilibrium capture for exact IMAS/OMAS export. When True (default),
    # each draw's converged TokaMaker flux-surface-average metrics are snapshot
    # into the archive (scan/<key>/<draw>/eq_fsa/), so IDS write-back converts
    # currents exactly per draw instead of with the template (baseline)
    # geometry. Cheap; set False to skip.
    capture_live_eq: bool = True
    # FSA grid for the captured block (matches the 257^2 eqdsk; >=129).
    capture_npsi: int = 257
    # Record <1/R^2> in the captured block (from get_q when the OFT build
    # exposes it, else by flux-surface quadrature): the exact IMAS j_tor
    # conversion (A5) at IDS export needs it (a draw without it exports with
    # the template geometry).  The quadrature adds ~65 surface traces/draw;
    # set False to skip that cost.
    capture_exact_inv_R2: bool = True
    # Also archive an OFT i-file (save_ifile: R,Z on flux surfaces, F, p, q, FF', p')
    # for GPEC eq_type='ldp_i' next to the eqdsk; needs an OFT with GPECf_interface.
    write_ifile: bool = False
    ifile_npsi: int = 129
    ifile_ntheta: int = 257

    #: Set from the ``swb_saw_*`` fields, never from ``bootstrap_kwargs``.
    _SAW_RESERVED = frozenset(
        "jphi_saw saw_q_s saw_dq saw_tol saw_ramp saw_rule".split())

    def __post_init__(self):
        """Validate ``bootstrap_kwargs``, then resolve ``structured_preset``
        into the individual structured fields.

        Thin wrapper over :func:`resolve_structured_preset`, which carries the
        rules (and is called again at the closure's own entry point, where it
        is a no-op for a config that was built with its channel already set).

        Applied ONLY to fields still HOLDING their dataclass default VALUE.

        **The limitation this cannot see past:** a plain dataclass field that
        was explicitly set to its own default value is indistinguishable from
        one that was never set, so such a field IS overridden by the preset.
        ``GenerationConfig(structured_preset="li_soft_onesided",
        structured_soft=False)`` comes back with ``structured_soft=True``.
        Every field a preset fills is therefore named in a warning, so the
        override is visible in the log rather than only in the archive; a run
        that wants a preset's priors with one field held against it should set
        that field AFTER construction.

        ``structured_li_sigma`` is filled only when a ``structured_li_target``
        was supplied -- a preset with no l_i target simply omits the l_i term
        rather than recording a sigma for a measurement that does not exist.
        An unknown preset name raises here, before any GS solve.

        Nothing else is touched: in particular ``closure_channel`` keeps its
        shipped ``"bootstrap"`` default, so naming a preset never silently
        switches the channel on -- ``structured_preset=None`` resolves to the
        DEFAULT preset only when the channel is already ``"structured"``.
        """
        if self.swb_seed not in ("source", "generic"):
            raise ValueError(f"swb_seed={self.swb_seed!r} not in ('source', 'generic')")
        if self.imas_baseline not in ("closure", "swb"):
            raise ValueError(
                f"imas_baseline={self.imas_baseline!r} not in ('closure', 'swb')")
        if self.swb_saw_q is not None and not float(self.swb_saw_q) > 0.0:
            raise ValueError(f"swb_saw_q={self.swb_saw_q!r}: must be > 0 (None = off)")
        for name in ("swb_saw_dq", "swb_saw_tol"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name}={getattr(self, name)!r} must be > 0")
        if not float(self.swb_saw_ramp) >= 0.0:
            raise ValueError(f"swb_saw_ramp={self.swb_saw_ramp!r} must be >= 0")
        if self.swb_saw_rule not in SWB_SAW_RULES:
            raise ValueError(f"swb_saw_rule={self.swb_saw_rule!r} not in {tuple(SWB_SAW_RULES)}")
        resolve_solve_method(self)
        validate_bootstrap_kwargs(
            self.bootstrap_kwargs,
            _BOOTSTRAP_RESERVED | self._SAW_RESERVED
            | ({"jphi_fixed"} if self.swb_seed == "source" else set())
            | ({"p_fixed"} if self.imas_baseline == "swb" else set()))
        resolve_structured_preset(self, stacklevel=4)
        validate_structured_mse_settings(self)
        from .edge_pressure import validate_edge_pressure_settings
        validate_edge_pressure_settings(self.edge_pprime_pin,
                                        self.separatrix_pressure)
        _m = self.draw_solve_maxits
        import numbers
        if _m is not None and (isinstance(_m, bool) or not isinstance(
                _m, numbers.Integral) or _m < 1):
            raise ValueError(f"draw_solve_maxits={_m!r} must be an integer "
                             ">= 1, or None for the solver's own cap")


def validate_structured_mse_settings(gc) -> None:
    """Refuse invalid MSE settings at CONFIG time, before any GS solve.

    Called from :meth:`GenerationConfig.__post_init__` and again at the
    structured closure's entry (so a field changed after construction is
    caught too).  Raises a plain ``ValueError`` -- deliberately NOT
    :class:`bouquet.mse.MSEDataUnusable`, which the non-required path turns
    into a per-slice "not applied" -- because a bad setting is the caller's
    configuration error, not a property of one slice's data:

    * ``structured_mse_steps`` an integer >= 1;
    * ``structured_mse_fd_step`` a finite number > 0;
    * ``structured_mse_sigma_sys`` a finite number >= 0;
    * ``structured_mse_min_chords`` an integer >= 1;
    * ``structured_mse_required`` a bool;
    * ``mse_data`` ``None`` or a dict with no key outside the schema of
      :mod:`bouquet.mse` (the values are judged per slice, where a block that
      is unusable is refused or recorded as not applied);
    * ``mse_data`` together with ``imas_corrective_jphi=True``: the MSE
      Jacobian's finite-difference probes are plain solves, while the
      predictor it is differenced against is solve + corrective iteration,
      so every column would carry that difference divided by the step.
    """
    import math
    import numbers

    def _int_ge1(name):
        v = getattr(gc, name, None)
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) \
                or int(v) < 1:
            raise ValueError(f"generation.{name} must be an integer >= 1, "
                             f"got {v!r}")

    def _real(name, lo, strict):
        v = getattr(gc, name, None)
        ok = (not isinstance(v, bool) and isinstance(v, numbers.Real)
              and math.isfinite(float(v))
              and (float(v) > lo if strict else float(v) >= lo))
        if not ok:
            raise ValueError(f"generation.{name} must be a finite number "
                             f"{'>' if strict else '>='} {lo:g}, got {v!r}")

    _int_ge1("structured_mse_steps")
    _int_ge1("structured_mse_min_chords")
    _real("structured_mse_fd_step", 0.0, strict=True)
    _real("structured_mse_sigma_sys", 0.0, strict=False)
    req = getattr(gc, "structured_mse_required", False)
    if not isinstance(req, bool):
        raise ValueError("generation.structured_mse_required must be a bool, "
                         f"got {req!r}")
    md = getattr(gc, "mse_data", None)
    if md is None:
        return
    if not isinstance(md, dict):
        raise ValueError("generation.mse_data must be None or a dict in the "
                         f"schema of bouquet.mse, got {type(md).__name__}")
    from .mse import (MSE_OPTIONAL_KEYS, MSE_ORIENTATION_KEYS,
                      MSE_REQUIRED_KEYS, mse_block_unknown_keys)
    unknown = mse_block_unknown_keys(md)
    if unknown:
        raise ValueError(
            "generation.mse_data carries unknown key(s) " + ", ".join(unknown)
            + " -- refused rather than ignored (known: "
            + ", ".join(MSE_REQUIRED_KEYS + MSE_ORIENTATION_KEYS
                        + MSE_OPTIONAL_KEYS) + ")")
    if bool(getattr(gc, "imas_corrective_jphi", False)):
        raise ValueError(
            "generation.mse_data with imas_corrective_jphi=True is refused: "
            "the MSE Jacobian differences plain solves against a predictor "
            "solved WITH the corrective iteration, which biases every column")


#: Arguments the call sites set themselves; ``bootstrap_kwargs`` may not
#: shadow them (duplicate keyword, or a silent override of a per-draw value).
_BOOTSTRAP_RESERVED = frozenset(
    "mygs ne Te ni Ti Zeff Ip_target inductive_jphi scale_jBS "
    "isolate_edge_jBS verbose diagnostic_plots psi_pad psi_N x coord "
    "ffp_prof ne_prof te_prof ni_prof ti_prof".split()
)


@functools.lru_cache(maxsize=None)
def _bootstrap_kwarg_names():
    """Every keyword ``bootstrap_kwargs`` can reach, introspected from the
    toolkit's bootstrap entry points.  ``None`` when OpenFUSIONToolkit is not
    importable, which skips the unknown-key check.  Cached per process.
    """
    try:
        import inspect

        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap

        # Only the entry points this toolkit has: on one without the
        # internal solve, the accepted set is solve_with_bootstrap's own
        # arguments, which is what it accepts there.
        names = set()
        for fn in (solve_with_bootstrap,
                   getattr(TokaMaker, "solve_bootstrap", None),
                   getattr(TokaMaker, "set_boot_ops", None)):
            if fn is None:
                continue
            names |= {prm.name for prm in inspect.signature(fn).parameters.values()
                      if prm.kind in (prm.POSITIONAL_OR_KEYWORD, prm.KEYWORD_ONLY)}
        return frozenset(names) - {"self"}
    except Exception:
        # No OFT (unit tests, a docs build): cannot introspect, so do not
        # guess; a wrong key then surfaces at the call.
        return None


#: The internal solve methods (GenerationConfig.solve_method).
SOLVE_METHODS = ("legacy", "swb", "engine")


def resolve_solve_method(gc) -> str:
    """The solve method of *gc*, with ``imas_baseline`` /
    ``reconstruction_engine`` brought in line with it (in place).

    ``solve_method=None`` is derived from those two older fields; an explicit
    value sets them, and refuses a contradicting ``imas_baseline="swb"``.
    Idempotent; called at construction and again by
    :class:`bouquet.run.Bouquet` before it builds or draws (a config may have
    been edited in between)."""
    sm = getattr(gc, "solve_method", None)
    swb = str(getattr(gc, "imas_baseline", "closure")) == "swb"
    if sm is None and not swb:
        eng = str(getattr(gc, "reconstruction_engine", "legacy")) == "unified"
        return "engine" if eng else "legacy"
    if sm is None:
        sm = "swb"
    elif sm not in SOLVE_METHODS:
        raise ValueError(f"generation.solve_method={sm!r} must be one of "
                         f"{SOLVE_METHODS}")
    # an imas_baseline="swb" this function wrote itself follows a later
    # solve_method (reconstruction_engine defaults to "unified": never a
    # contradiction)
    elif (swb and sm != "swb" and getattr(gc, "_solve_method_written", None)
          != (gc.imas_baseline, gc.reconstruction_engine)):
        raise ValueError(f"generation.solve_method={sm!r} contradicts "
                         'imas_baseline="swb"; set solve_method alone')
    gc.imas_baseline = "swb" if sm == "swb" else (
        "closure" if gc.imas_baseline == "swb" else gc.imas_baseline)
    gc.reconstruction_engine = "unified" if sm == "engine" else "legacy"
    gc._solve_method_written = (gc.imas_baseline, gc.reconstruction_engine)
    return sm


def swb_config_problems(config):
    """Settings ``imas_baseline="swb"`` cannot honour, as messages (empty: OK).

    The Fortran SWB's inductive scale alpha IS the Ip closure, so every
    bouquet closure / anchor / delta mode is refused rather than combined.
    """
    import os
    gc, sc = config.generation, config.solver
    p = []
    if not isinstance(config.source, ImasSource):
        p.append("needs an ImasSource")
    if gc.kinetic_source != "ida_hybrid":
        p.append(f"kinetic_source={gc.kinetic_source!r} (only 'ida_hybrid' for now)")
    if gc.swb_seed != "source":
        p.append("swb_seed must be 'source' (the source split is the SWB input)")
    for name in ("single_profile_jphi", "imas_corrective_jphi", "jbs_delta_mode",
                 "anchor_pressure_to_equilibrium"):
        if getattr(gc, name, False):
            p.append(f"{name}=True")
    if not gc.recalculate_j_BS:
        p.append("recalculate_j_BS=False (SWB re-solves j_BS by construction)")
    if str(gc.closure_channel) != "bootstrap":
        p.append(f"closure_channel={gc.closure_channel!r} is never read: SWB's alpha "
                 "(ohmic channel) is the closure; leave it at 'bootstrap'")
    if gc.coil_drift_hard_factor is not None:
        p.append("coil_drift_hard_factor: drift is measured, not bounded")
    if int(sc.nthreads) != 1:
        p.append(f"nthreads={sc.nthreads} (sigma=0 exactness needs 1)")
    if gc.bootstrap_kwargs.get("use_python_solve"):
        p.append("bootstrap_kwargs use_python_solve (needs the Fortran SWB)")
    # OFT solve_bootstrap pins P'(psi_N=1)=0 and targets pax=p[0]-p[-1]:
    # the engine defaults; anything else would be silently ignored
    if not getattr(gc, "edge_pprime_pin", True):
        p.append("edge_pprime_pin=False (OFT's SWB always pins P' at the edge)")
    if str(getattr(gc, "separatrix_pressure", "offset")) != "offset":
        p.append(f"separatrix_pressure={gc.separatrix_pressure!r} (OFT's SWB "
                 "always solves with pax = p_axis - p_sep)")
    if {"taper_edge_jBS", "taper_edge_psi0"} & set(gc.bootstrap_kwargs):
        p.append("bootstrap_kwargs taper_edge_*: set swb_edge_taper_psi0 instead")
    t = gc.swb_edge_taper_psi0
    if t is not None and not 0.0 < float(t) < 1.0:
        p.append(f"swb_edge_taper_psi0={t!r} must be in (0, 1) or None")
    for env in ("DIFF_BS", "PIN_JPHI"):
        if os.environ.get(env, "0") == "1":
            p.append(f"{env}=1")
    from .coords import _swb_grid_arg, _swb_params
    if not _swb_grid_arg() or not {"jphi_fixed", "p_fixed"} <= _swb_params():
        p.append("this OpenFUSIONToolkit's solve_with_bootstrap lacks x/jphi_fixed/p_fixed")
    if gc.swb_saw_q is not None and "jphi_saw" not in _swb_params():
        p.append("swb_saw_q: this OpenFUSIONToolkit's solve_with_bootstrap lacks jphi_saw")
    return p


def swb_bootstrap_kwargs(gc):
    """``bootstrap_kwargs`` for an ``imas_baseline="swb"`` solve: the user's, plus the
    edge taper from ``swb_edge_taper_psi0``."""
    kw = dict(gc.bootstrap_kwargs)
    if gc.swb_edge_taper_psi0 is not None:
        kw.update(taper_edge_jBS=True, taper_edge_psi0=float(gc.swb_edge_taper_psi0))
    return kw


def validate_bootstrap_kwargs(bootstrap_kwargs, reserved=_BOOTSTRAP_RESERVED,
                              known=None):
    """Refuse a ``bootstrap_kwargs`` key that would not survive the call chain.

    Validated at config time: the call sites end in ``**kwargs``, so a wrong
    key would otherwise fail every draw inside the draw loop's ``except``.

    Parameters
    ----------
    bootstrap_kwargs : dict
        The keys to check.
    reserved : set of str
        Names the call sites pass themselves.
    known : set of str, optional
        The accepted keyword names; defaults to :func:`_bootstrap_kwarg_names`
        (``None`` from it skips the unknown-key check).  Passed explicitly by
        the tests, which run without OpenFUSIONToolkit.
    """
    keys = set(bootstrap_kwargs)

    bad = sorted(reserved & keys)
    if bad:
        raise ValueError(
            f"bootstrap_kwargs may not set {bad}: passed explicitly at call sites.")

    if "swb_iterations" in keys:
        raise ValueError(
            "bootstrap_kwargs: 'swb_iterations' is now 'iterations'.")

    if known is None:
        known = _bootstrap_kwarg_names()
    if known is None:
        return
    unknown = sorted(keys - set(known))
    if unknown:
        raise ValueError(
            f"bootstrap_kwargs has no such solve_with_bootstrap option(s): "
            f"{unknown}. Accepted: {sorted(set(known) - set(reserved))}.")


def resolve_structured_preset(gc, warn: bool = True, stacklevel: int = 3):
    """Resolve ``gc.structured_preset`` onto *gc*, in place.  Idempotent.

    The one place the preset rules live, called both from
    :meth:`GenerationConfig.__post_init__` and from the structured closure's
    own entry point (so a config whose ``closure_channel`` was set AFTER
    construction is not silently left on the superseded raw fields).  It works
    on any object carrying the ``GenerationConfig`` attribute names, and reads
    every default from ``GenerationConfig.__dataclass_fields__``.

    The rules, in order:

    * ``structured_preset=None`` and ``closure_channel != "structured"`` --
      nothing is applied, exactly as before this function existed.
    * ``structured_preset=None`` and ``closure_channel == "structured"`` -- the
      DEFAULT preset (``utils.STRUCTURED_PRESET_DEFAULT``) is applied, and the
      warning says BY DEFAULT and how to decline it.  Declined outright, with a
      warning and no fills, when ``structured_basis`` is set: the preset's
      ladders are widths at the shipped basis's radii and mean nothing on
      another basis.  Declined for ``structured_ip_sigma_frac`` alone when the
      caller set an absolute ``structured_ip_sigma``, because the two are
      mutually exclusive downstream and a DEFAULT may not turn a configuration
      that ran yesterday into a refusal.  (A preset NAMED explicitly still
      fills the frac and lets the closure refuse the clash: there the caller
      asked for the preset, so the ambiguity is theirs to resolve.)
    * ``structured_preset="none"`` (``utils.STRUCTURED_PRESET_NONE``) -- the
      opt-out: nothing is applied, every structured field keeps its shipped
      default.  This reproduces, exactly, what a bare structured channel did
      before the default preset existed.
    * any other name -- applied as it always was, whatever the channel.

    In every case a preset fills ONLY fields still holding their dataclass
    default VALUE, which a field explicitly set to that same value also does
    (the limitation :meth:`GenerationConfig.__post_init__` documents); the
    fields it filled are named in a ``UserWarning`` and recorded.

    Returns, and writes onto *gc* as ``structured_preset_in_force`` /
    ``structured_preset_source`` / ``structured_preset_fields``,
    ``{"name", "source", "fields"}``: which preset is in force, whether it was
    chosen ``"explicit"``-ly or by ``"default"`` (or ``"opt-out"`` /
    ``"default-declined-custom-basis"`` / ``"unset"``), and the fields it
    filled.  The closure record copies these, so the archive says not just
    which prior was used but who chose it.
    """
    import warnings

    from .utils import (STRUCTURED_PRESET_DEFAULT, STRUCTURED_PRESET_NONE,
                        structured_preset_settings)

    specs = GenerationConfig.__dataclass_fields__

    def _get(name):
        return getattr(gc, name, specs[name].default)

    def _record(name, source, applied):
        applied = sorted(applied)
        if (getattr(gc, "structured_preset_in_force", None) == name
                and getattr(gc, "structured_preset_source", None) == source):
            # An idempotent re-resolution fills nothing; keep the field list
            # written by the first pass rather than blanking the record.
            applied = sorted(set(applied)
                             | set(getattr(gc, "structured_preset_fields", [])
                                   or []))
        gc.structured_preset_in_force = name
        gc.structured_preset_source = source
        gc.structured_preset_fields = list(applied)
        return dict(name=name, source=source, fields=list(applied))

    name = _get("structured_preset")
    by_default = False
    if name is None:
        if str(_get("closure_channel")) != "structured":
            return _record(None, "unset", ())
        name, by_default = STRUCTURED_PRESET_DEFAULT, True
        if _get("structured_basis") is not None:
            if warn:
                warnings.warn(
                    "closure_channel='structured' with an explicit "
                    "structured_basis and no structured_preset: the default "
                    f"preset {name!r} is a ladder of prior WIDTHS at the "
                    "shipped basis's radii and has no meaning on another "
                    "basis, so it is NOT applied -- every structured field "
                    "keeps its own default (the uniform, no-prior ladder for a "
                    "basis of a different length).  Name the preset explicitly "
                    "to apply it anyway, or set structured_weights / "
                    "structured_sigma_ind_up for this basis.",
                    stacklevel=stacklevel)
            return _record(None, "default-declined-custom-basis", ())

    key = str(name)
    filled = structured_preset_settings(key)     # refuses an unknown name
    if not filled:                               # the "none" opt-out
        return _record(None, "opt-out", ())
    if _get("structured_li_target") is None:
        filled.pop("structured_li_sigma", None)
    if by_default and _get("structured_ip_sigma") is not None:
        filled.pop("structured_ip_sigma_frac", None)

    applied = []
    for field_name, value in filled.items():
        if _get(field_name) == specs[field_name].default:
            setattr(gc, field_name, value)
            applied.append(field_name)
    if applied and warn:
        if by_default:
            warnings.warn(
                "closure_channel='structured' with no structured_preset: the "
                f"validated preset {key!r} is applied BY DEFAULT and filled "
                + ", ".join(sorted(applied))
                + " -- a field explicitly set to its own default value is "
                  "indistinguishable from an unset one and is overridden "
                  "here; set it after construction to hold it against the "
                  "preset, or pass structured_preset='none' to decline the "
                  "default and keep every structured field as shipped",
                stacklevel=stacklevel)
        else:
            warnings.warn(
                f"structured_preset={key!r} filled "
                + ", ".join(sorted(applied))
                + " -- a field explicitly set to its own default value is "
                  "indistinguishable from an unset one and is overridden "
                  "here; set it after construction to hold it against the "
                  "preset", stacklevel=stacklevel)
    return _record(key, "default" if by_default else "explicit", applied)


#: the two word settings of ``FilterConfig.rms_max_mm`` (besides a number / None)
BOUNDARY_CUT_WORDS = ("auto", "off")


def check_boundary_cut_setting(value, name="rms_max_mm"):
    """Validate a boundary-cut setting: ``"auto"``, ``"off"``, ``None`` (=
    ``"off"``) or a finite real number (not a bool). Returns it unchanged."""
    import math
    import numbers
    if value is None:
        return value
    if isinstance(value, str):
        if value not in BOUNDARY_CUT_WORDS:
            raise ValueError(f"{name} must be 'auto', 'off', None or a number of "
                             f"mm, got {value!r}")
        return value
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be 'auto', 'off', None or a finite number "
                         f"of mm, got {value!r}")
    return value


@dataclass
class FilterConfig:
    """Postprocessing selection of the machine-realizable subset."""

    #: LCFS boundary-deviation cut [mm rms] applied by ``Bouquet.filter()`` and
    #: by the until-N in-loop verdict (the same number, by construction).
    #:
    #: * ``"auto"`` (default) resolves to the DEVICE's calibrated cut
    #:   (:attr:`bouquet.devices.DeviceSpec.boundary_rms_max_mm`; 8.5 mm on
    #:   DIII-D from its boundary-UQ study) or, with no device calibration, to
    #:   the generic 5.0 mm (:data:`bouquet.devices.GENERIC_BOUNDARY_RMS_MM`);
    #: * a number is an explicit cut and always wins;
    #: * ``"off"`` disables the boundary cut (loop and filter alike);
    #: * ``None`` is the historical spelling of ``"off"`` and keeps meaning
    #:   "no boundary cut", so configs written that way behave as they did.
    #:
    #: The resolved value and its source (``"explicit"``, ``"device:<name>"``,
    #: ``"generic"`` or ``"disabled"``) are printed once and stamped on the
    #: archive (``boundary_rms_max_mm`` / ``boundary_cut_source``), so a band
    #: built later can say which cut defined its population.
    rms_max_mm: Union[float, str, None] = "auto"
    # Coil filter used by Bouquet.filter():
    #   "chi2"   -> measurement-referenced chi2/nu <= chi2_max, with the per-coil
    #               sigma resolved from `coil_sigma` below (default: the device's
    #               own tolerance model; needs no dd)                   [DEFAULT]
    #   "legacy" -> +/-inspec_F_max on F-coils, +/-inspec_VSC_max on the VSC pair
    coil_filter: str = "chi2"
    # Acceptance: None -> the device's empirically calibrated thresholds (DIII-D:
    # chi2/nu <= 6.1 and worst-coil |z| <= 6.3, the 95th percentile of what real
    # machine states score), or the generic 4 / 5 when no device calibration
    # applies. Give numbers to override; z_max=False disables the guard.
    chi2_max: Optional[float] = None
    z_max: Optional[Any] = None
    # Per-coil tolerance for the chi2 filter. None -> the device model (device named
    # in BouquetConfig.device or detected from the mesh coil names); if neither is
    # available Bouquet.filter() falls back LOUDLY to the legacy rule.
    #   {"floor": A-t, "fraction": f} -> sigma_i = hypot(floor, fraction*|I_i|)
    #   {coil_name: sigma_A-t, ...}   -> per-coil table
    #   callable(baseline) -> {coil: sigma}
    #   "<model name>"                -> a named model of the device (e.g. "rms_incl_offset")
    coil_sigma: Optional[Any] = None
    # Acquisition era whose tolerance floor applies (bouquet.devices.era_labels;
    # DIII-D: "pre2014" or "modern").  The era chooses the sigma FLOOR, so it is an
    # acceptance criterion and must be stated, never guessed: nothing infers it from
    # a run header, mesh name or file path.  None -> the era is taken from an explicit
    # source.pulse/source.shot through the device's era bands, where the pulse number
    # is a DATE PROXY for the acquisition upgrade (so the boundary is approximate);
    # failing that, the device's default band, which carries the tightest floor.
    # Bouquet.filter() prints the era it resolved, the floor it buys and which of
    # those three routes it came from, once per call.
    coil_daq_era: Optional[str] = None
    inspec_F_max: float = 0.02      # +/-2% coil-current spec (DIII-D); legacy only
    inspec_VSC_max: float = 0.02

    def __post_init__(self):
        check_boundary_cut_setting(self.rms_max_mm, "filtering.rms_max_mm")
        if self.coil_filter not in ("chi2", "legacy"):
            raise ValueError("filtering.coil_filter must be 'chi2' or 'legacy'")
        if self.coil_daq_era is not None and not isinstance(self.coil_daq_era, str):
            raise ValueError("filtering.coil_daq_era must be None or an era label string")
        cs = self.coil_sigma
        if cs is not None and not callable(cs) and not isinstance(cs, (dict, str)):
            raise ValueError("filtering.coil_sigma must be None, a {'floor','fraction'} dict, "
                             "a {coil: sigma} dict, or a callable(baseline)")
        if isinstance(cs, dict) and {"floor", "fraction"} <= set(cs):
            if float(cs["floor"]) < 0 or not (0 <= float(cs["fraction"]) < 1):
                raise ValueError("filtering.coil_sigma: floor must be >= 0 A-t and 0 <= fraction < 1")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class BouquetConfig:
    """Top-level configuration. Pass to ``bq.Bouquet(config)``."""

    source: BaselineSource
    solver: SolverConfig
    output_header: str                              # HDF5 written to f"{header}.h5"
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    fixed_components: FixedComponentsConfig = field(default_factory=FixedComponentsConfig)
    # Device name (bouquet.devices.DEVICES). Optional: detected from the mesh coil
    # names when they match a registered device exactly; required only when the
    # chi2 coil filter needs a device tolerance model and no filtering.coil_sigma
    # is given -- Bouquet.filter() then falls back loudly to the legacy rule.
    device: Optional[str] = None
    # When False (default) the verbose TokaMaker solver chatter emitted during
    # baseline reconstruction (DLSODE / gs_get_qprof / li-match iteration) is
    # captured to baseline.reconstruction_log and only a curated quality summary
    # is printed. Set True to stream the full solver output for debugging.
    verbose: bool = False

    def __post_init__(self):
        """Validate cross-field invariants early (before any GS solve)."""
        if not self.output_header:
            raise ValueError("output_header must be a non-empty string")

        src = self.source
        if isinstance(src, ReconstructionSource):
            if not src.geqdsk_path or not src.profiles_path:
                raise ValueError(
                    "ReconstructionSource requires geqdsk_path and profiles_path"
                )
        elif isinstance(src, ImasSource):
            if not src.ids_path:
                raise ValueError("ImasSource requires ids_path")
            # fail here, not after the dd (100s of MB) has been read
            from .io.imas import parse_current_orientation
            parse_current_orientation(src.current_orientation)
        else:
            raise TypeError(
                "source must be a ReconstructionSource or ImasSource, got "
                f"{type(src).__name__}"
            )

        from .coords import PHI, run_coord
        if (run_coord(src.coord) == PHI
                and self.generation.bootstrap_kwargs.get("use_python_solve")):
            raise ValueError("coord='phi_n' needs the internal bootstrap "
                             "solve: drop use_python_solve from bootstrap_kwargs")

        if self.fixed_components.p_fast_reduction not in (
                "auto", "trace", "mean", "perp", "sum"):
            raise ValueError(
                "fixed_components.p_fast_reduction must be 'auto', 'trace', 'mean', "
                "'perp', or 'sum'"
            )
        if self.device is not None:
            # fail here, not after a whole ensemble has been solved: the device is
            # only consulted by Bouquet.filter(), at the very end of a run.
            from .devices import DEVICES, era_labels, get_device
            if self.device not in DEVICES:
                raise ValueError(
                    f"unknown device {self.device!r}; registered: {sorted(DEVICES)}")
            era = self.filtering.coil_daq_era
            if era is not None:
                known = era_labels(get_device(self.device))
                if era not in known:
                    raise ValueError(
                        f"filtering.coil_daq_era {era!r} is not an era of device "
                        f"{self.device!r}; available: {sorted(known)}")
        if self.uncertainty.sigma_mode not in ("auto", "direct", "ensemble"):
            raise ValueError("uncertainty.sigma_mode must be 'auto', 'direct', or 'ensemble'")
        if self.uncertainty.sigma_method not in ("percentile", "std"):
            raise ValueError(
                "uncertainty.sigma_method must be 'percentile' or 'std'")
        # Caught here rather than only in resolve_zeff_envelope, which runs
        # inside Bouquet.generate() -- i.e. after prepare_baseline() has
        # already paid for the baseline GS solve.
        if self.uncertainty.zeff_sigma_source not in (
                "auto", "carbon", "measured", "scalar"):
            raise ValueError(
                "uncertainty.zeff_sigma_source must be 'auto', 'carbon', "
                "'measured', or 'scalar'")
        if self.generation.n_equils < 1:
            raise ValueError("generation.n_equils must be >= 1")
        _tgt = require_integer_count(
            self.generation.n_inspec_target, "generation.n_inspec_target")
        _cap = require_integer_count(
            self.generation.max_total_draws, "generation.max_total_draws")
        if _tgt is not None:
            if _tgt < 1:
                raise ValueError(
                    "generation.n_inspec_target must be >= 1 (or None to draw "
                    "exactly n_equils)")
            if _cap is not None and _cap < _tgt:
                raise ValueError(
                    f"generation.max_total_draws ({_cap}) is below "
                    f"n_inspec_target ({_tgt}): the cap would stop the "
                    f"run before the target could ever be met")
        elif _cap is not None:
            raise ValueError(
                "generation.max_total_draws only applies with "
                "n_inspec_target set; without a target the run draws exactly "
                "n_equils")
        if self.generation.workflow not in (
                "auto", "geqdsk-standard", "imas-diff-c", "custom"):
            raise ValueError(
                "generation.workflow must be 'auto', 'geqdsk-standard', "
                "'imas-diff-c', or 'custom'")
        # the self-consistent bootstrap loop's settings (values only; the
        # workflow-level refusals live in Bouquet._validate_workflow)
        from .jbs_loop import (deprecated_jbs_settings_warning,
                               validate_jbs_settings)
        validate_jbs_settings(self.generation)
        # the pressure handed to the solver (bouquet.edge_pressure)
        from .edge_pressure import resolve_edge_pressure
        resolve_edge_pressure(self.generation)
        # the unified reconstruction engine's settings (refused by name;
        # engine_* fields changed under the legacy engine are refused too)
        from .engine import validate_engine_settings
        validate_engine_settings(self.generation)
        # bootstrap_kwargs the loop's Redl does not read: loud, not silent
        deprecated_jbs_settings_warning(self.generation, stacklevel=3)

    # ── serialization (h5 provenance, per-shot templating, SLURM bundles) ──
    def to_dict(self) -> dict:
        """Plain-Python, JSON-round-trippable snapshot of the whole config.

        ndarray fields (isoflux, ``sigma_profiles``, ``aux_*``, fixed
        components, ``profile_overrides``) are encoded as ``{"__ndarray__":
        [...]}``; the baseline source records a ``"source_type"`` discriminator
        (``"reconstruction"`` | ``"imas"``). Reverse with :meth:`from_dict`.
        """
        d = _encode(self)
        d["source"]["source_type"] = (
            "reconstruction" if isinstance(self.source, ReconstructionSource) else "imas")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BouquetConfig":
        """Rebuild a :class:`BouquetConfig` from :meth:`to_dict` output."""
        d = _decode(d)
        srcd = dict(d["source"])
        stype = srcd.pop("source_type", None)
        if stype is None:                       # infer if the discriminator is absent
            stype = "reconstruction" if "geqdsk_path" in srcd else "imas"
        SrcCls = ReconstructionSource if stype == "reconstruction" else ImasSource
        gend = _checked_generation_keys(dict(d.get("generation", {})))
        if "jbs_self_consistent" not in gend:
            # A stored config written before the self-consistent bootstrap
            # existed was produced by the frozen-bootstrap path: rebuild it on
            # that path (the default flipped to True afterwards), so replaying
            # an old archive's config_json reproduces what it recorded.
            # to_dict() always writes the field, so a current config never
            # takes this branch.  (A misspelt field no longer lands here: an
            # unknown generation key is refused above.)
            import warnings
            warnings.warn(
                "config has no generation.jbs_self_consistent (it predates "
                "the self-consistent bootstrap loop): loading it with "
                "jbs_self_consistent=False, the LEGACY frozen-bootstrap "
                "behaviour it was produced with, so it reproduces its old "
                "results.  To run the self-consistent bootstrap loop "
                "instead, opt in explicitly: add "
                '"jbs_self_consistent": true to the "generation" section of '
                "the dict/JSON, or set cfg.generation.jbs_self_consistent = "
                "True after loading.", UserWarning, stacklevel=2)
            gend["jbs_self_consistent"] = False
        if "separatrix_pressure" not in gend:
            # The same for the separatrix-pressure setting, whose default
            # moved "legacy" -> "offset" (2026-10-02): a stored config that
            # predates the setting was produced with the full axis pressure
            # as the solver's target, so it is rebuilt with "legacy" and
            # replays what it recorded.  to_dict() always writes the field.
            import warnings
            warnings.warn(
                "config has no generation.separatrix_pressure (it predates "
                "the setting): loading it with separatrix_pressure='legacy', "
                "the behaviour it was produced with, so it reproduces its old "
                "results.  The current default is 'offset' (p_sep removed "
                "from the solver's axis target and added back in every "
                "reported pressure, beta and W_MHD); to use it, add "
                '"separatrix_pressure": "offset" to the "generation" section '
                "or set cfg.generation.separatrix_pressure = 'offset' after "
                "loading.", UserWarning, stacklevel=2)
            gend["separatrix_pressure"] = "legacy"
        if "reconstruction_engine" not in gend:
            # The same for the reconstruction engine, whose default moved
            # "legacy" -> "unified" (2026-10-06): a stored config that
            # predates the field (introduced 2026-09-29 at "legacy") was
            # produced by the legacy paths, so it is rebuilt on them and
            # replays what it recorded.  to_dict() always writes the field.
            import warnings
            warnings.warn(
                "config has no generation.reconstruction_engine (it predates "
                "the unified engine): loading it with "
                "reconstruction_engine='legacy', the reconstruction and draw "
                "paths it was produced with, so it reproduces its old "
                "results.  The current default is 'unified'; to use it, add "
                '"reconstruction_engine": "unified" to the "generation" '
                "section (legacy-only settings must then be at their "
                "defaults) or rebuild the config with Bouquet.from_geqdsk / "
                "from_imas.", UserWarning, stacklevel=2)
            gend["reconstruction_engine"] = "legacy"
        _stored_config_compat(gend)
        if (SrcCls is ImasSource
                and gend.get("reconstruction_engine") == "unified"
                and "imas_li3_radius" not in gend):
            # An IMAS unified config stored before the li_3-radius setting
            # (2026-10-06) ran its l_i row on the source's li_3 UNRESCALED
            # -- what "axis" does -- so it is loaded that way and replays
            # what it recorded.  (A g-file config is not concerned: the
            # setting has no effect there and must stay at its default.)
            import warnings
            warnings.warn(
                "stored IMAS unified config has no generation."
                "imas_li3_radius (it predates the setting): loading it with "
                "'axis' -- the l_i row's target was the source's li_3 "
                "unrescaled -- so it reproduces what it recorded; today's "
                "default is 'auto'", UserWarning, stacklevel=2)
            gend["imas_li3_radius"] = "axis"
        return cls(
            source=_build(SrcCls, srcd),
            solver=_build(SolverConfig, d["solver"]),
            output_header=d["output_header"],
            uncertainty=_build(UncertaintyConfig, d.get("uncertainty", {})),
            generation=_build(GenerationConfig, gend),
            filtering=_build(FilterConfig, d.get("filtering", {})),
            fixed_components=_build(FixedComponentsConfig, d.get("fixed_components", {})),
            verbose=bool(d.get("verbose", False)),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """The config as a JSON string (see :meth:`to_dict`)."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "BouquetConfig":
        """Rebuild a config from a JSON string produced by :meth:`to_json`."""
        import json
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Serialization helpers (used by BouquetConfig.to_dict / from_dict)
# ---------------------------------------------------------------------------
import dataclasses as _dc


def _encode(v):
    """Recursively convert a config value to a JSON-round-trippable form."""
    import numpy as np
    if isinstance(v, np.ndarray):
        return {"__ndarray__": v.tolist()}
    if _dc.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _encode(getattr(v, f.name)) for f in _dc.fields(v)}
    if isinstance(v, dict):
        return {k: _encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    return v


def _decode(v):
    """Reverse :func:`_encode`: restore ndarrays; recurse dicts/lists."""
    import numpy as np
    if isinstance(v, dict):
        if set(v.keys()) == {"__ndarray__"}:
            return np.asarray(v["__ndarray__"], dtype=float)
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


#: GenerationConfig fields that existed once and were removed.  An old
#: stored config may still carry them; they are dropped WITH a warning (they
#: have no effect on the current code), never mistaken for a typo.
_RETIRED_GENERATION_KEYS = ("coil_drift_threshold_A", "lock_coils",
                            "lock_coils_weight", "window_coord", "seed_coord",
                            "engine_split_pressure")


#: Every earlier DEFAULT of an engine-only field, with the dates it was the
#: default.  to_dict() writes every field, so a config stored while one of
#: these was the default carries it explicitly; under
#: ``reconstruction_engine="legacy"`` an engine field has no effect, so
#: :func:`_stored_config_compat` loads such a value as today's default
#: instead of refusing the config (validate_engine_settings refuses a
#: non-default engine field under "legacy").
ENGINE_FIELD_HISTORICAL_DEFAULTS = {
    "engine_mse_jacobian": (("fd_broyden", "2026-09-29", "2026-10-02"),),
    "engine_ids_inductive": (("auto", "2026-10-02", "2026-10-02"),),
}

#: GenerationConfig fields whose DEFAULT changed after they were introduced:
#: field -> (value a stored config that PREDATES the field ran with, or
#: ``None`` when that is not knowable from the config alone; what it was).
#: Consulted for EVERY stored config (2026-10-06; before, only for a stored
#: ``reconstruction_engine="unified"`` config, so the loud entries never
#: fired for a config bouquet itself stored): the engine-only entries
#: (:data:`_ENGINE_ONLY_PRE_INTRODUCTION`) for unified configs only, the
#: loop's own (:data:`_LOOP_PRE_INTRODUCTION`) for any config that ran the
#: self-consistent loop, the rest for every config.
#: ``jbs_self_consistent``, ``separatrix_pressure`` and
#: ``reconstruction_engine`` itself (default ``"legacy"`` -> ``"unified"``
#: on 2026-10-06; a config without it predates the engine and loads as
#: ``"legacy"``) have their own back-fills in :meth:`BouquetConfig.from_dict`.
FIELD_PRE_INTRODUCTION = {
    # introduced 2026-10-02 at "auto" -- the IDS adapter's own default
    # before the setting existed -- then "residual" (owner decision)
    "engine_ids_inductive": (
        "auto", "the IDS adapter's inductive default before the setting "
                "existed (2026-10-02)"),
    # introduced with reconstruction_engine itself (2026-09-29): a unified
    # config without it was not written by to_dict()
    "engine_mse_jacobian": (None, "a unified config without it was not "
                                  "written by to_dict()"),
    # the post-homotopy pass ceiling was the constant 2 before the field
    # (introduced 2026-09-27 at 4, then 6 on 2026-10-01)
    "jbs_max_passes_post_homotopy": (
        2, "jbs_loop.JBS_POST_HOMOTOPY_PASSES before the field "
           "(2026-09-25 to 2026-09-27)"),
    # the loop's relaxation before the two fields (introduced 2026-09-27 at
    # 0.7 / 3): no relaxation of the solved current, and omega halved on
    # EVERY growth of r_j (jbs_loop tolerances_record's pre-field values)
    "jbs_relax_current": (
        1.0, "no relaxation of the solved current before the field "
             "(2026-09-25 to 2026-09-27)"),
    "jbs_relax_halve_on": (
        1, "omega halved on every growth of r_j before the field "
           "(2026-09-25 to 2026-09-27)"),
    # older changed defaults (2026-06): not knowable from the config alone
    "l_i_tolerance": (None, "default 0.01 -> 0.05 on 2026-06-04"),
    "jBS_scale_range": (None, "default None -> (0.99, 1.01) on 2026-06-04"),
    "homotopy_passes": (None, "default None -> three passes on 2026-06-04"),
    "floor_j_BS": (None, "default True -> False on 2026-06-24"),
    "jbs_max_passes_draw": (None, "default 6 -> 12 on 2026-09-27"),
}


#: :data:`FIELD_PRE_INTRODUCTION` entries that only exist on the unified
#: engine (a legacy config must keep them at their defaults --
#: ``validate_engine_settings`` refuses otherwise).
_ENGINE_ONLY_PRE_INTRODUCTION = ("engine_ids_inductive", "engine_mse_jacobian")

#: :data:`FIELD_PRE_INTRODUCTION` entries of the self-consistent loop: back-
#: filled only for a config that ran the loop (``jbs_self_consistent``); on
#: one that did not they had no effect.
_LOOP_PRE_INTRODUCTION = ("jbs_max_passes_post_homotopy", "jbs_relax_current",
                          "jbs_relax_halve_on", "jbs_max_passes_draw")


#: Legacy-path fields the factories set whatever the engine until
#: 2026-10-05, which the unified engine never read and now refuses when not
#: at their defaults (:data:`bouquet.engine.ENGINE_UNREAD_LEGACY_FIELDS`).
#: Since 2026-10-07 both default to ``None`` and are resolved per engine at
#: ``prepare_baseline()`` (:data:`bouquet.engine.ENGINE_DEPENDENT_DEFAULTS`).
#: Kept for reference; the stored-load rule (c) of
#: :func:`_stored_config_compat` now covers EVERY entry of
#: ``ENGINE_UNREAD_LEGACY_FIELDS`` (2026-10-06), of which these are two.
STORED_UNIFIED_UNREAD_FIELDS = ("isolate_edge_jBS", "perturb_jind_in_anchor")


def _stored_unified_unread_fields():
    """Every legacy-path field the unified engine never reads (and refuses
    when set on a NEW config): :data:`bouquet.engine.
    ENGINE_UNREAD_LEGACY_FIELDS`, in its order.  A stored unified config
    carrying one at a non-default value loads at the default (rule (c) of
    :func:`_stored_config_compat`)."""
    from .engine import ENGINE_UNREAD_LEGACY_FIELDS
    return tuple(ENGINE_UNREAD_LEGACY_FIELDS)


def _stored_config_compat(gend: dict) -> None:
    """Load a stored ``generation`` dict as it was produced (in place).

    (a) Under ``reconstruction_engine="legacy"`` (or absent) an engine field
    holding one of its HISTORICAL defaults
    (:data:`ENGINE_FIELD_HISTORICAL_DEFAULTS`) -- the value to_dict() wrote
    while it was the default -- is loaded as today's default, with a
    warning: it had no effect on that path, and refusing it would make the
    config unloadable.

    (b) A stored ``"unified"`` config that LACKS a field whose default has
    changed since (:data:`FIELD_PRE_INTRODUCTION`) is loaded with the value
    it was produced with where that is knowable, with a warning; where it is
    not, today's default is used with a LOUD warning naming the field.
    ``engine_draw_solve_maxits`` missing: before it existed the engine draws
    were capped by ``draw_solve_maxits``, so that value moves over (and
    ``draw_solve_maxits``, which the engine now refuses, is cleared).

    (c) A stored ``"unified"`` config carrying a non-default value of ANY
    legacy-path field the engine never reads
    (:data:`bouquet.engine.ENGINE_UNREAD_LEGACY_FIELDS`, 21 fields; or
    ``homotopy_passes`` with ``engine_draw_homotopy=False``) is loaded at
    the default, with a warning naming the field: the engine ignored the
    value, so the default reproduces what the stored config actually ran.
    Refused since 2026-10-04 (3779b51) for a NEW config; until 2026-10-06
    only ``isolate_edge_jBS`` / ``perturb_jind_in_anchor`` had this
    stored-load rule (6e052d0), and a stored config carrying one of the
    other 20 could not be loaded."""
    import warnings
    from .engine import ENGINE_FIELD_DEFAULTS
    eng = gend.get("reconstruction_engine", "legacy")
    if gend.get("solve_method") == "engine":
        eng = "unified"
    if eng == "legacy":
        for name, hist in ENGINE_FIELD_HISTORICAL_DEFAULTS.items():
            if name not in gend:
                continue
            for val, since, until in hist:
                if gend[name] == val:
                    warnings.warn(
                        f"stored config: generation.{name}={val!r} was its "
                        f"default from {since} to {until}; it has no effect "
                        "with reconstruction_engine='legacy' and is loaded "
                        "as today's default "
                        f"{ENGINE_FIELD_DEFAULTS[name]!r}", UserWarning,
                        stacklevel=3)
                    gend[name] = ENGINE_FIELD_DEFAULTS[name]
                    break
    if gend.get("jbs_self_consistent") and \
            "jbs_max_passes_post_homotopy" not in gend:
        val, why = FIELD_PRE_INTRODUCTION["jbs_max_passes_post_homotopy"]
        warnings.warn(
            "stored config has no generation.jbs_max_passes_post_homotopy "
            f"(it predates the field): loading it with {val}, {why}, so it "
            "reproduces what it recorded; today's default is "
            f"{_field_default('jbs_max_passes_post_homotopy')!r}",
            UserWarning, stacklevel=3)
        gend["jbs_max_passes_post_homotopy"] = val
    _relax = [n for n in ("jbs_relax_current", "jbs_relax_halve_on")
              if n not in gend]
    if gend.get("jbs_self_consistent") and _relax:
        # a loop config stored 2026-09-25..27 ran with no current
        # relaxation and omega halved on every growth (1.0 / 1), not
        # today's 0.7 / 3: loaded with what it ran, ONE warning
        parts = []
        for n in _relax:
            val, why = FIELD_PRE_INTRODUCTION[n]
            gend[n] = val
            parts.append(f"{n}={val!r} ({why}; today's default "
                         f"{_field_default(n)!r})")
        warnings.warn(
            "stored loop config has no generation."
            + " / generation.".join(_relax)
            + " (it predates the field"
            + ("s" if len(_relax) > 1 else "") + "): loading it with "
            + "; ".join(parts) + " -- the values it ran with, so it "
            "reproduces what it recorded", UserWarning, stacklevel=3)
    # the remaining pre-introduction entries, for EVERY stored config (the
    # engine-only ones below, under "unified"): before 2026-10-06 this loop
    # ran for unified configs only, so the loud entries never fired for a
    # config bouquet itself stored
    _pre_introduction_backfill(
        gend, [n for n in FIELD_PRE_INTRODUCTION
               if n not in _ENGINE_ONLY_PRE_INTRODUCTION
               and n not in ("jbs_max_passes_post_homotopy",
                             "jbs_relax_current", "jbs_relax_halve_on")
               and (n not in _LOOP_PRE_INTRODUCTION
                    or gend.get("jbs_self_consistent") or eng == "unified")],
        "unified" if eng == "unified" else "")
    if eng != "unified":
        return
    # (c) every legacy-path field the engine never reads: the engine
    # ignored a stored non-default value (the factories set
    # isolate_edge_jBS / perturb_jind_in_anchor for the legacy path
    # whatever the engine until 2026-10-05; the other 20 were accepted and
    # ignored until 3779b51), and refuses one on a NEW config since -- a
    # stored unified config carrying one is loaded at the default (the run
    # it recorded is the same), with a warning
    from .engine import (ENGINE_DEPENDENT_DEFAULTS, _same_value,
                         engine_validated_value)
    for name in _stored_unified_unread_fields():
        if name not in gend:
            continue
        d = _field_default(name)
        if name in ENGINE_DEPENDENT_DEFAULTS:
            # stored before 2026-10-07 as the engine's value (then the
            # dataclass default), or unset since: loaded unchanged
            d = engine_validated_value(name, "unified", "reconstruction")
            same = gend[name] is None or _same_value(gend[name], d)
        else:
            same = (gend[name] is None) if d is None else \
                _same_value(gend[name], d)
        if not same:
            warnings.warn(
                f"stored unified config: generation.{name}={gend[name]!r} "
                "(a legacy-path setting) was never read by the unified "
                f"engine; it is loaded as its default {d!r}, which the "
                "engine now requires -- the stored run is unchanged",
                UserWarning, stacklevel=3)
            gend[name] = d
    if gend.get("engine_draw_homotopy") is False and \
            "homotopy_passes" in gend:
        d = _field_default("homotopy_passes")
        if not _same_value(gend["homotopy_passes"], d):
            warnings.warn(
                "stored unified config: generation.homotopy_passes="
                f"{gend['homotopy_passes']!r} with engine_draw_homotopy="
                "False was never read by the unified engine (no homotopy "
                f"runs in an engine draw); it is loaded as its default "
                f"{d!r}, which the engine now requires -- the stored run "
                "is unchanged", UserWarning, stacklevel=3)
            gend["homotopy_passes"] = d
    if "engine_draw_solve_maxits" not in gend:
        cap = gend.get("draw_solve_maxits")
        warnings.warn(
            "stored unified config has no generation.engine_draw_solve_maxits "
            "(it predates the field, 2026-09-30): the engine draws were then "
            f"capped by draw_solve_maxits={cap!r}, so it is loaded as "
            f"engine_draw_solve_maxits={cap!r} (draw_solve_maxits cleared: "
            "the engine refuses it now); today's default is "
            f"{ENGINE_FIELD_DEFAULTS['engine_draw_solve_maxits']}",
            UserWarning, stacklevel=3)
        gend["engine_draw_solve_maxits"] = cap
        gend["draw_solve_maxits"] = None
    _pre_introduction_backfill(gend, list(_ENGINE_ONLY_PRE_INTRODUCTION),
                               "unified")


def _pre_introduction_backfill(gend, names, kind):
    """Back-fill each of *names* (:data:`FIELD_PRE_INTRODUCTION` entries)
    that the stored generation dict *gend* lacks: with the value it ran
    with where that is knowable (a warning), else today's default with a
    LOUD warning naming the field.  *kind* ("unified" or "") only words the
    message."""
    import warnings
    what = "stored unified config" if kind == "unified" else "stored config"
    for name in names:
        if name in gend:
            continue
        val, why = FIELD_PRE_INTRODUCTION[name]
        if val is not None:
            warnings.warn(
                f"{what} has no generation.{name} (it "
                f"predates the field): loading it with {val!r}, {why}, so it "
                "reproduces what it recorded; today's default is "
                f"{_field_default(name)!r}", UserWarning, stacklevel=4)
            gend[name] = val
        else:
            warnings.warn(
                f"{what.upper()} LACKS generation.{name}, whose "
                f"default has changed ({why}): the value it was produced "
                "with is NOT knowable from the config, so today's default "
                f"{_field_default(name)!r} is used -- results may differ "
                f"from the stored run; set generation.{name} explicitly",
                UserWarning, stacklevel=4)


def _field_default(name):
    f = GenerationConfig.__dataclass_fields__[name]
    return (f.default if f.default is not _dc.MISSING
            else f.default_factory())


def _checked_generation_keys(gend: dict) -> dict:
    """Refuse an unknown ``generation`` key in a config dict.

    A misspelt field (``jbs_self_consistant``) used to be dropped silently by
    :func:`_build`, leaving the default in force -- for the loop switch, with
    a warning that wrongly said the config predates the loop.  Every key must
    now be a :class:`GenerationConfig` field (``init=False`` recorded fields
    included) or one of :data:`_RETIRED_GENERATION_KEYS` (dropped, with a
    warning).  The refusal names the key and the nearest valid one.
    """
    import difflib
    import warnings
    names = {f.name for f in _dc.fields(GenerationConfig)}
    if "swb_iterations" in gend:
        # retired for bootstrap_kwargs; every stored config carries it (to_dict
        # writes all fields), so the default is dropped silently and any
        # other value is carried over as bootstrap_kwargs["iterations"] --
        # except under the unified engine, which never read it (dropped)
        from .jbs_loop import SWB_ITERATIONS_DEFAULT
        gend = dict(gend)
        v = gend.pop("swb_iterations")
        unified = gend.get("reconstruction_engine") == "unified"
        if v not in (None, SWB_ITERATIONS_DEFAULT) and unified:
            warnings.warn(
                f"stored unified config: generation.swb_iterations={v!r} "
                "(a retired legacy-path setting) was never read by the "
                "unified engine; it is dropped -- the stored run is "
                "unchanged", UserWarning, stacklevel=3)
        elif v not in (None, SWB_ITERATIONS_DEFAULT):
            bk = dict(gend.get("bootstrap_kwargs") or {})
            bk.setdefault("iterations", int(v))
            gend["bootstrap_kwargs"] = bk
            warnings.warn(
                f"config generation.swb_iterations={v!r} (retired) loaded as "
                f"bootstrap_kwargs={{'iterations': {bk['iterations']}}}.",
                UserWarning, stacklevel=3)
    retired = [k for k in gend if k in _RETIRED_GENERATION_KEYS]
    unknown = sorted(k for k in gend
                     if k not in names and k not in _RETIRED_GENERATION_KEYS)
    if unknown:
        parts = []
        for k in unknown:
            near = difflib.get_close_matches(str(k), sorted(names), n=1,
                                             cutoff=0.0)
            parts.append(f"{k!r} (nearest valid key: {near[0]!r})" if near
                         else repr(k))
        raise ValueError(
            "config generation section has unknown key(s): "
            + ", ".join(parts) + ".  Refusing to drop them silently (a "
            "misspelt field would leave its default in force); fix the "
            "spelling or remove the key.")
    if retired:
        warnings.warn(
            "config generation section carries retired field(s) "
            f"{retired}: they no longer exist and are ignored.",
            UserWarning, stacklevel=3)
        gend = {k: v for k, v in gend.items() if k not in retired}
    return gend


def _build(cls, d):
    """Instantiate dataclass ``cls`` from decoded dict ``d`` (unknown keys dropped).

    ``init=False`` fields (the recorded preset provenance) are dropped too:
    they are outputs of ``__post_init__``, not constructor arguments, and are
    recomputed identically on the rebuilt config.
    """
    names = {f.name for f in _dc.fields(cls) if f.init}
    return cls(**{k: v for k, v in d.items() if k in names})
