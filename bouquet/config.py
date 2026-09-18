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
    # bouquet.coil_targets.coil_reg_from_measured. Applied by
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
    """

    geqdsk_path: str
    profiles_path: str                 # IDA .cdf OR p-file (auto-detected by extension)
    cocos: int = 1
    time: Optional[float] = None       # IDA time slice [s] (multi-time .cdf files)
    impurity_Z: float = 6.0            # effective impurity charge (carbon); set per machine
    profile_overrides: dict = field(default_factory=dict)  # name -> array, manual override
    # reconstruction knobs
    psi_pad: float = 1e-3
    n_k: int = 5                       # inductive-spline knots
    psi_bridge: float = 0.99          # Hermite edge-bridge location
    rescale_j_BS: bool = False
    shelf_psi_N: float = 0.0
    # guess_jinductive is derived from the g-file j_phi when None


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
    # --- IDA-hybrid kinetics (GenerationConfig.kinetic_source = "ida_hybrid") ---
    # When set, the baseline ne/Te/Ti/omega_tor are taken from this IDA .cdf
    # (externally fit, smoother across time than FUSE's per-slice profile fits),
    # resampled onto the FUSE core_profiles psi_N grid. Z_eff / Z_imp / the ni
    # dilution stay FUSE, for consistency with FUSE's own resistive diffusion
    # (which consumed FUSE's Z_eff to produce j_ohmic).  NOTE the old blanket
    # rationale "IDA's Z_eff is internally inconsistent with its own carbon
    # density" is SHOT-DEPENDENT, not general: measured Zeff(VB) vs
    # 1+Z(Z-1)nC/ne core-median deviations are -1.7 % / +4.7 % / +11.3 % on
    # three DIII-D demo shots, i.e. mostly within the file's own measured
    # sigma_Zeff (~8-9 %); read_ida now prints this cross-check per file
    # (Callahan 2019 JINST 14 C10002 is the agreement pedigree when C6+
    # dominates). Everything else (currents,
    # equilibrium, p_fast, anchors) stays FUSE. Also wire it to
    # UncertaintyConfig.ida_path so the sigma envelopes come from the same IDA.
    ida_path: Optional[str] = None
    impurity_Z: float = 6.0            # machine impurity charge (carbon); ni dilution
    # OPTIONAL. A gEQDSK whose LCFS replaces the dd boundary outline as the
    # isoflux separatrix target. Leave None to use the source's own boundary.
    # Supply one when you have a more accurate separatrix for the slice than the
    # dd carries -- typically a magnetics-only equilibrium reconstruction, which
    # fits the boundary to the external magnetics without kinetic assumptions.
    # One g-file per slice; the driver picks the nearest time.
    LCFS_geqdsk: Optional[str] = None


BaselineSource = Union[ReconstructionSource, ImasSource]


# ---------------------------------------------------------------------------
# Fixed additive components (NEVER perturbed by GPR draws)
# ---------------------------------------------------------------------------
@dataclass
class FixedComponentsConfig:
    """Externally-driven current + fast-ion pressure, held fixed across draws.

    These are summed into the baseline *and* every perturbed equilibrium,
    untouched by the GPR perturbation::

        j_phi_total = j_inductive + j_BS + j_NBI + j_RF
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
      * ``j_RF`` -- **never computed internally** (RF is the least-common input).
        Always zeros unless the user supplies an array here.

    All arrays are on ``psi_N`` (kinetic grid), SI units, toroidal current
    convention for j_*. ``None`` -> zeros.
    """

    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure
    j_NBI: Optional["np.ndarray"] = None    # beam-driven TOROIDAL current density [A/m^2]
    j_RF: Optional["np.ndarray"] = None     # RF-driven TOROIDAL current density [A/m^2]
    psi_N: Optional["np.ndarray"] = None    # grid for the above (if arrays given)

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
    sigma_ni_from_ne: bool = True          # IDA path only: sigma_ni = sigma_ne

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
    #   "scalar"   -- always the flat zeff_scalar_sigma fraction (pre-1.3.2
    #                 behaviour).
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
    # and convert its parallel output to toroidal (see physics.parallel_to_toroidal),
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
    # Bootstrap profile mode in solve_with_bootstrap. True (default) isolates the
    # edge spike, yielding a clean positive bootstrap; False uses the full SWB
    # profile, which for FUSE equilibria carries an unphysical inner negative
    # lobe (must then be floored, leaving kinks -- see baseline_jphi_caseA plots).
    isolate_edge_jBS: bool = True
    # How the SWB bootstrap is reconciled with the FUSE baseline on the IMAS
    # path (see run._forward_solve_imas_baseline):
    #   "diff"    : keep FUSE total; add fixed correction diff = FUSE_jBS - SWB
    #               to baseline and every draw (anchors to FUSE; risks edge
    #               misalignment when the perturbed pedestal moves).
    #   "rescale" : keep FUSE ohmic; rescale SWB by a single factor so l_i
    #               matches the source (fully self-consistent bootstrap).
    #   "ohmic"   : HYBRID. SWB bootstrap (from the kinetic source, e.g. IDA)
    #               and FUSE NBI/RF taken as-is; Ip closed by rescaling FUSE
    #               j_ohmic only (factor recorded as Baseline.ohm_scale). The
    #               jphi_diff equilibrium anchor is NOT applied. Use when the
    #               kinetic source has a materially different pedestal than
    #               FUSE -- "diff" would erase that current change.
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
    #: whose length is not the default 4 gets a uniform (no-prior) ladder,
    #: named as such in the record.  The physics prior below is a ladder over
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
    perturb_jind_in_anchor: bool = False
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
    #   "ida_hybrid" -> ne/Te/Ti/omega_tor from ImasSource.ida_path (resampled onto
    #                   the FUSE psi_N grid); Z_eff/Z_imp/ni-dilution stay FUSE;
    #                   currents/equilibrium/p_fast/anchors stay FUSE.
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
    # inner negative lobe). With the default isolate_edge_jBS=True the spike is
    # already ~clean, so flooring is redundant -- and it REGRESSED a stiff
    # high-l_i case (clipping its isolate-edge spike drove yield to 0).
    floor_j_BS: bool = False
    # solve_with_bootstrap H-mode self-consistency iterations per draw (default
    # 3); lowering to 2 trades a little accuracy for speed on large bouquets.
    swb_iterations: int = 3
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
    # into the archive (scan/<key>/<draw>/eq_fsa/), so IDS write-back does an
    # exact per-draw toroidal->parallel current conversion instead of the
    # interim baseline-ratio reconstruction. Cheap; set False to skip.
    capture_live_eq: bool = True
    # FSA grid for the captured block (matches the 257^2 eqdsk; >=129).
    capture_npsi: int = 257
    # Compute exact <1/R^2> by flux-surface quadrature (TokaMaker does not
    # expose it) so the conversion is machine-exact rather than using the
    # <B_phi^2>~=<B^2> bracket (~<1%). Adds ~65 surface traces/draw; set False
    # to skip that cost (self-validated + graceful fallback either way).
    capture_exact_inv_R2: bool = True

    def __post_init__(self):
        """Resolve ``structured_preset`` into the individual structured fields.

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
        resolve_structured_preset(self, stacklevel=4)


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


@dataclass
class FilterConfig:
    """Postprocessing selection of the machine-realizable subset."""

    rms_max_mm: float = 5.0
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
        else:
            raise TypeError(
                "source must be a ReconstructionSource or ImasSource, got "
                f"{type(src).__name__}"
            )

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
        return cls(
            source=_build(SrcCls, srcd),
            solver=_build(SolverConfig, d["solver"]),
            output_header=d["output_header"],
            uncertainty=_build(UncertaintyConfig, d.get("uncertainty", {})),
            generation=_build(GenerationConfig, d.get("generation", {})),
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


def _build(cls, d):
    """Instantiate dataclass ``cls`` from decoded dict ``d`` (unknown keys dropped).

    ``init=False`` fields (the recorded preset provenance) are dropped too:
    they are outputs of ``__post_init__``, not constructor arguments, and are
    recomputed identically on the rebuilt config.
    """
    names = {f.name for f in _dc.fields(cls) if f.init}
    return cls(**{k: v for k, v in d.items() if k in names})
