"""The :class:`Baseline` -- the common product every source resolves to.

``generate()`` consumes a :class:`Baseline` and never depends on reconstruction
directly. Both :class:`~bouquet.config.ReconstructionSource` and
:class:`~bouquet.config.ImasSource` resolve to this same structure, so adding a
new input pipeline means adding a resolver, not touching generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .config import BaselineSource, SolverConfig


@dataclass
class Baseline:
    """Separated baseline currents + targets + kinetic profiles.

    The currents are split into ohmic + bootstrap + driven (j_phi =
    j_inductive + j_BS + j_NBI + j_RF) regardless of provenance: the
    reconstruction source *produces* the split, the IMAS source *reads* it
    pre-separated.

    CURRENT CONVENTION: every current component here is TokaMaker ``jphi`` =
    <j_phi> [A/m^2] (``docs/current-conventions.md``). IMAS inputs (``j_tor``
    and parallel <j.B>/B0) are converted exactly on read
    (:mod:`bouquet.physics` docstring), so downstream code never mixes
    conventions.
    """

    # --- required fields (no defaults) ---------------------------------
    # current-density grid + separated currents
    psi_N: "np.ndarray"
    j_phi: "np.ndarray"            # total [A/m^2] = j_inductive + j_BS + j_NBI + j_RF + j_other
    j_inductive: "np.ndarray"     # ohmic part [A/m^2]   (perturbed via l_i matching)
    j_BS: "np.ndarray"            # bootstrap part [A/m^2] (recomputed per draw)

    # baseline kinetic profiles (SI: m^-3, eV) on `psi_N_kinetic`
    psi_N_kinetic: "np.ndarray"
    ne: "np.ndarray"
    te: "np.ndarray"
    ni: "np.ndarray"
    ti: "np.ndarray"
    Zeff: "np.ndarray"

    # targets held fixed across all perturbed draws
    Ip_target: float
    l_i_target: float

    provenance: str               # "reconstruction" | "imas"

    # --- optional / defaulted fields -----------------------------------
    # Which l_i estimator `l_i_target` (and every downstream l_i comparison --
    # the per-draw acceptance band, the archive `l_i_target` attr, plot_traces)
    # is expressed on.  "iter(li3)" == TokaMaker li_normalization='iter'
    # == 2*int(Bp^2 dV)/((mu0 Ip)^2 R_axis).  Carried explicitly so an archive
    # is self-describing and a future scale change cannot pass silently
    # (issue #20).  Single source of truth: bouquet.utils.LI_SCALE.
    l_i_scale: str = "iter(li3)"    # == utils.LI_SCALE
    # Fixed additive components -- summed into EVERY draw, never GPR-perturbed.
    # None is treated as zeros. See FixedComponentsConfig for the contract:
    #   j_phi_total = j_inductive + j_BS + j_NBI + j_RF
    #   p_total     = p_thermal(perturbed) + p_fast
    j_NBI: Optional["np.ndarray"] = None    # beam-driven current [A/m^2]
    j_RF: Optional["np.ndarray"] = None     # RF-driven current [A/m^2]
    p_fast: Optional["np.ndarray"] = None   # fast/beam pressure

    # How p_fast was reduced from the source's anisotropic fields, and how that
    # rule was chosen (IMAS path only; None on the g-file path, where the p-file
    # supplies p_fast directly). Keys: "rule" (the reduction applied, or None if
    # the user supplied p_fast outright), "basis" (explicit-argument /
    # explicit-stamp / imas.jl-structure / producer-string /
    # undetermined-fallback / user-override), "evidence" (the dd field and text
    # the decision rests on), "requested" (what the caller asked for) and
    # "warned". The two dd storage conventions differ by a FACTOR OF THREE, so
    # the decision is recorded with the baseline rather than re-inferred later.
    # See bouquet.io.imas.resolve_p_fast_reduction.
    p_fast_meta: Optional[dict] = None

    # bootstrap amplitude factor applied when the j_BS/j_inductive split is
    # rebuilt against SWB and calibrated to the measured l_i (IMAS path's
    # _forward_solve_imas_baseline, mirroring the g-file fit_inductive_profile).
    # 1.0 means no rebuild was done (raw source split kept).
    bs_scale: float = 1.0
    # 'ohmic' mode only: factor applied to FUSE j_inductive so the hybrid
    # (s*j_ohm + SWB_jBS + j_fixed) integrates to Ip_target. 1.0 otherwise.
    ohm_scale: float = 1.0
    # 'ohmic' mode bookkeeping: proxy-Ip of each component, the proxy's own
    # error on the FUSE total, and the jphi_diff anchor that was NOT applied.
    ip_closure: Optional[dict] = None
    # IMAS path only: the two slice-level facts closure_channel=
    # "sawtooth_bootstrap" gates on, read ONCE at load time because the reader
    # does not retain the (100s of MB) dd -- the source's sawtooth model
    # amplitude at this slice (core_sources source with identifier index 701)
    # and the dd's OWN on-axis q from equilibrium.profiles_1d.q.  The latter is
    # recorded for comparison only: the closure's reference is TokaMaker's q0
    # for the source total re-solved on the anchor, not the dd's own estimator
    # (issue #20 -- never compare two estimators of the same name).
    # Keys: source_index, present, j_par_max_abs, active, q0_dd, and -- when
    # the dd has a sawteeth entry with profiles -- slice (how it was read at
    # this slice: "matched by time", "by index", or why it has no slice at
    # this time, in which case it is not active here).
    sawtooth: Optional[dict] = None

    # Case-B ("diff") fixed bootstrap correction profile [A/m^2] = FUSE_jBS - SWB,
    # added to the baseline AND every draw's j_phi so the total anchors to the
    # FUSE bootstrap while the SWB delta tracks per-draw kinetics. None => not in
    # diff mode (Case-A "rescale" or no SWB rebuild).
    jBS_diff: Optional["np.ndarray"] = None

    # IMAS pressure anchor ("diff" approach, mirroring jBS_diff). p_equilibrium
    # is the authoritative dd equilibrium.pressure on psi_N; p_diff =
    # p_equilibrium - reconstructed baseline pressure (e*(ne*Te + ni*Ti) +
    # impurity + p_fast). p_diff is added to the baseline AND every draw so the
    # solve pressure anchors to FUSE exactly while the reconstructed thermal
    # delta tracks per-draw kinetics. Z_imp is the single effective impurity
    # charge (one-Zeff single-impurity model) used to recover nz = (ne-ni)/Z_imp
    # for the impurity (carbon) pressure term. All None on the geqdsk path
    # (p-file pressure is already the total).
    p_equilibrium: Optional["np.ndarray"] = None
    p_diff: Optional["np.ndarray"] = None
    Z_imp: Optional[float] = None
    # Fast-ion charge density sum_s Z_s n_s^fast [m^-3] on the KINETIC grid
    # (IMAS path; None elsewhere).  Fixed across draws, like p_fast: the
    # thermal impurity math (Z_imp above, nz, p_imp, the Zeff-primary ni
    # derivation) must run on ne - z_fast, while the Zeff consumed by the
    # bootstrap stays on the full ne.
    z_fast: Optional[np.ndarray] = None       # sum_s Z_s   n_s^fast

    # IMAS total-current anchor: jphi_diff = equilibrium.profiles_1d.j_tor
    # (the GS-consistent current GPEC reads, with the pedestal current) minus the
    # reconstructed core_profiles.j_tor total. A FIXED offset added to the baseline
    # AND every draw so the total anchors to the equilibrium while the SWB bootstrap
    # + perturbed j_inductive ride underneath. None on the geqdsk path / when
    # GenerationConfig.anchor_jtor_to_equilibrium is False.
    jphi_diff: Optional["np.ndarray"] = None

    # raw bytes preserved for archival into the HDF5 (optional)
    eqdsk_bytes: Optional[bytes] = None
    pfile_bytes: Optional[bytes] = None

    # full reconstruction diagnostics, when provenance == "reconstruction"
    recon: Optional[dict] = None

    # l_i sanity metrics (IMAS path): IDS-reported li_1/li_3 vs the
    # TokaMaker-solved li_1/li_3 from the forward-solve. l_i_target is set to
    # the TokaMaker li_1; the IDS values are kept here for comparison/plots.
    li_metrics: Optional[dict] = None

    # auxiliary source-provided profiles available to the perturbation
    # switchboard -- rotation ('omega_tor', 'e_r'), transport ('chi_e',
    # 'chi_i') and impurity ('zeff') channels --
    # on psi_N_kinetic. Populated by the reader where the source has them; the
    # user may add/override via UncertaintyConfig.aux_baselines.
    aux: Optional[dict] = None

    # curated reconstruction quality metrics (reconstruction path only): a flat
    # dict consumed by Bouquet's reconstruction summary -- Ip/l_i target-vs-
    # achieved, boundary RMS/max, q0/q95, shape, and a pass/fail verdict.
    reconstruction_metrics: Optional[dict] = None
    # full captured solver chatter from the reconstruction (when verbose=False),
    # kept available for debugging without cluttering the notebook output.
    reconstruction_log: Optional[str] = None

    # Appended LAST on purpose: Baseline is a public, positionally
    # constructible dataclass, so a new field must not shift the slots of
    # the pre-existing ones (aux, reconstruction_metrics, ...).
    # Core-pressure hollowness health record (see
    # physics.core_pressure_hollow_record): how far the core pressure rises
    # above its innermost-node value, and over what radial extent, measured on
    # the INPUT pressure (total, and thermal species only) and on the ACHIEVED
    # pressure of the converged equilibrium.  Descriptive and report-only:
    # nothing reads it back, so no profile, solve, filter decision, in-spec or
    # until-N count depends on it.  Also carried inside ``li_metrics`` so it
    # reaches the archive.  None when the source path did not evaluate it.
    core_pressure_hollow: Optional[dict] = None

    # Appended AFTER every pre-existing field on purpose (Baseline is a
    # public, positionally constructible dataclass; a new field must not
    # shift the slots of the older ones).
    # Current orientation of the SOURCE, and what the reader did about it.
    # bouquet works in one positive-current frame: every anchor is solved to
    # |Ip| with F0 = |R*B|, so every bootstrap it recomputes is positive.  The
    # IMAS reader multiplies every current profile it reads by
    # ``source_current_sign`` (= sign of the dd's equilibrium ip, +1.0 for
    # ip >= 0, unless ``ImasSource.current_orientation`` names the factor
    # explicitly) so all currents on this Baseline are in that frame -- see
    # :func:`bouquet.io.imas.read_imas_baseline`.  +1.0 on the reconstruction
    # path, whose split is produced by a TokaMaker fit in the same frame.
    # ``source_current_sign_origin`` says where the factor came from (IMAS
    # path: "auto: sign(equilibrium ip)" or "override: ImasSource.
    # current_orientation"; None elsewhere).
    # ``source_b0_sign`` is the sign of the source's vacuum B0 (IMAS path;
    # None elsewhere, and None for a zero or unreadable b0) -- recorded only:
    # nothing in the reader flips on it, and F0 = |r0*b0| was already
    # orientation-free.  Together they are what a consumer needs to map a
    # delivered (positive-frame) equilibrium back onto the experiment's
    # orientation.
    source_current_sign: float = 1.0
    source_current_sign_origin: Optional[str] = None
    source_b0_sign: Optional[float] = None

    # closure_channel="structured" with mse_data only: the MSE term's
    # PER-CHORD arrays (chords used, measured tan(gamma), sigma_eff, residuals
    # in sigma before/after/delivered, predicted tan(gamma), the n x 2K
    # Jacobian, excluded chords with reasons).  Kept out of ip_closure -- which
    # is archived as one size-capped JSON attribute -- and archived as
    # datasets under _baseline/structured_mse.  None otherwise.
    mse_record: Optional[dict] = None

    # ---- the ONE reconstruction state (jbs_self_consistent=True only) -------
    # With the self-consistent loop on, the current split above (j_phi,
    # j_inductive, j_BS, jBS_diff, the fixed parts) is the jphi-linterp
    # REQUEST of the reconstruction's final equilibrium F, in the form every
    # draw consumes it: one jphi-linterp solve of j_phi (+ jphi_diff) is F;
    # it is normalised to Ip_target in the 'exact' FSA current measure on F;
    # j_BS (+ jBS_diff) is the draws' own sigma=0 bootstrap composition on F;
    # j_inductive is the residual.  ``jphi_request_offset`` is that request
    # minus F's ACHIEVED current (in the corrective iteration's form, also
    # Ip-normalised): the standard draw route targets achieved currents, so
    # it perturbs ``j_inductive - jphi_request_offset`` and starts its
    # corrective iteration from target + offset -- F's own request at zero
    # perturbation.  ``delivered_state`` records F: l_i (== l_i_target), q0,
    # q95, the normalisation factors, F's achieved FSA current
    # (``j_phi_achieved``) and how the state was reached.  Both None with the
    # loop off (the legacy split, bit for bit).
    jphi_request_offset: Optional["np.ndarray"] = None
    delivered_state: Optional[dict] = None

    # the unified reconstruction engine's full record
    # (GenerationConfig.reconstruction_engine="unified" only; None on the
    # legacy paths): contract, settings, convergence constants and their
    # origins, per-pass log, delivery checks, the state a draw inherits and
    # the solve counts.  See bouquet.engine / docs/engine.md.
    engine: Optional[dict] = None

    # The pressure handed to the solver (bouquet.edge_pressure): the two
    # settings (edge_pprime_pin, separatrix_pressure), p_sep (the solve
    # pressure at psi_N = 1), the offset applied, the axis target, and both
    # frames of beta / W_MHD of the delivered equilibrium (solver: the
    # solver's own pressure, zero at the boundary; full: with p_sep added
    # back).
    edge_pressure: Optional[dict] = None

    # The coil least-squares mode the solver ran this baseline in
    # (bouquet.solver_state.coil_solve_mode): "bounded" -- OpenFUSIONToolkit's
    # bounded (BVLS) coil solve, entered once at Bouquet.setup_solver before
    # the reconstruction, so the draws use the same coil solver -- or
    # "unknown" for a solver bouquet did not set up.  Set by
    # Bouquet.prepare_baseline on both paths.
    coil_solve_mode: Optional[str] = None

    # How the engine-dependent generation settings were resolved for this
    # baseline (bouquet.engine.resolve_engine_defaults, at
    # Bouquet.prepare_baseline): field -> {"value", "origin"}, origin
    # "resolved from engine=<x>" or "explicit" (+ "engine_validated" when an
    # explicit value contradicts the engine's).  Archived as the _baseline
    # attr engine_resolved_defaults_json.
    engine_resolved_defaults: Optional[dict] = None

    # IMAS sources: how the core_sources slice and each beam / sawteeth entry
    # were matched to the core_profiles slice read (owner decision
    # 2026-10-06, io.imas.core_sources_slice / _source_slice_at): both
    # times, dt, the windows and their basis, the bracketing own times, and
    # each entry's status ("matched", "off_before_record" with its first own
    # time, "off_idle", "zero").  None on the g-file paths.
    source_time_match: Optional[dict] = None

    # Appended to keep the positional slots above.
    z2_fast: Optional[np.ndarray] = None      # sum_s Z_s^2 n_s^fast (kinetic grid)
    # Z_eff's numerator counts the fast ions: True for a measured Z_eff
    # (ida_hybrid); read off the dd otherwise (io.imas._dd_zeff).  Only
    # matters with z_fast; see physics.zeff_bounds.
    zeff_includes_fast: bool = False
    # Coordinate of psi_N / psi_N_kinetic (bouquet.coords): "psi_n" or
    # "phi_n".  Every profile and envelope of the run is on it.
    coord: str = "psi_n"
    # (psi_N, x) at the source's nodes: the io-time map from a psi_N-tabulated
    # input (an IDA sigma) to the run grid.  None in a psi_n run.
    psi_map: Optional[tuple] = None
    # Fields appended so every earlier field keeps its positional slot.
    # Other fixed driven current [A/m^2] (IMAS: fusion, runaways, sawteeth,
    # unknown core_sources indices); j_phi carries it.
    j_other: Optional["np.ndarray"] = None
    # The sawteeth share of j_other [A/m^2] (core_sources 701; IMAS path only).
    # With GenerationConfig.swb_saw_q it is SWB's jphi_saw input, and
    # j_phi = j_inductive + j_BS + j_NBI + j_RF + (j_other - j_sawteeth) + j_saw.
    j_sawteeth: Optional["np.ndarray"] = None
    # The structured closure's bootstrap multiplier PROFILE s_bs(psi) on psi_N
    # (bl.j_BS = s_bs * SWB(scale 1)); None when the multiplier is the scalar
    # bs_scale.  generate() hands it to the draws, which apply it after SWB.
    bs_scale_profile: Optional["np.ndarray"] = None
    # swb_seed="source": the SWB inputs of the baseline split (inductive seed,
    # jphi_fixed) on SWB's grid, reused unchanged by the draws and the sigma=0
    # check.  None => generic seed.
    swb_seed_profile: Optional["np.ndarray"] = None
    swb_jphi_fixed: Optional["np.ndarray"] = None
    # swb_saw_q set: SWB's jphi_saw input (j_sawteeth on SWB's grid), which
    # swb_jphi_fixed then excludes.  None => saw off.
    swb_jphi_saw: Optional["np.ndarray"] = None
    # swb_saw_q set: solve B's j_saw output (jphi_saw + the q reset current),
    # in place of j_sawteeth in j_phi.  None => saw off.
    j_saw: Optional["np.ndarray"] = None
    # imas_baseline="swb": solve A's coil currents {name: A-t}, the target of the
    # strong reg of solve B, the sigma=0 check and every draw.
    coil_reg_target: Optional[dict] = None
    # imas_baseline="swb": solve B's record (alpha, coils, lcfs, li_3, Ip, split),
    # the reference the sigma=0 check compares against.
    swb_baseline: Optional[dict] = None

    def __repr__(self):
        # concise summary -- the default dataclass repr dumps every numpy array,
        # which floods a notebook when `reconstruct()`/`prepare_baseline()` is the
        # last expression in a cell.
        import numpy as np
        ng = len(np.atleast_1d(self.psi_N))
        nk = len(np.atleast_1d(self.psi_N_kinetic))
        aux = list((self.aux or {}).keys())
        return (f"Baseline(provenance={self.provenance!r}, "
                f"Ip={self.Ip_target/1e6:.3f} MA, l_i={self.l_i_target:.3f}, "
                f"grids: {ng} eq / {nk} kinetic"
                + (f", aux={aux}" if aux else "") + ")")


def resolve_baseline(config: "BouquetConfig", mygs=None) -> Baseline:
    """Dispatch on ``config.source`` and return a populated :class:`Baseline`.

    * :class:`ImasSource` -> delegate to
      :func:`bouquet.io.imas.read_imas_baseline`: take j_ohmic / j_BS / Ip / l_i
      and core_profiles directly (no GS reconstruction), convert parallel->
      toroidal, isotropize p_fast. ``mygs`` is unused.
    * :class:`ReconstructionSource` -> read g-file + profiles (IDA .cdf or
      p-file), run :func:`reconstruct_equilibrium` on ``mygs``, package the
      fitted split currents (converted to toroidal) plus any user-supplied
      fixed components. (Implemented in phase 2 step 3.)

    Both paths return a :class:`Baseline` with every current as toroidal j_phi
    and the fixed additive components (j_NBI, j_RF, p_fast) attached.

    Implemented as a free function so sources stay declarative (plain config)
    and the resolution logic lives in one place.
    """
    from .config import ImasSource, ReconstructionSource

    source = config.source

    if isinstance(source, ImasSource):
        from .io.imas import read_imas_baseline
        return read_imas_baseline(
            source,
            fixed=config.fixed_components,
            p_fast_reduction=config.fixed_components.p_fast_reduction,
            allow_incomplete_pressure=config.generation.allow_incomplete_pressure,
            anchor_jtor_to_equilibrium=config.generation.anchor_jtor_to_equilibrium,
            kinetic_source=config.generation.kinetic_source,
            anchor_pressure_to_equilibrium=config.generation.anchor_pressure_to_equilibrium,
        )

    if isinstance(source, ReconstructionSource):
        return _resolve_reconstruction(source, config, mygs)

    raise TypeError(f"unknown baseline source type: {type(source).__name__}")


def _norm_path(p) -> str:
    """Normalise a path spelling for identity comparison.

    ``expandvars`` + ``expanduser`` + ``abspath`` + ``realpath``: resolves
    ``~``, environment variables, relative prefixes, ``.``/``..`` segments,
    trailing separators and symlinks (including symlinked parent
    directories).  ``realpath`` is defined for paths that do NOT exist -- it
    normalises the part it can and leaves the rest -- so this doubles as the
    tolerant fallback for a file that has not been written yet.  Returns
    ``""`` for an empty/None input.
    """
    import os

    s = str(p or "")
    if not s:
        return ""
    return os.path.realpath(os.path.abspath(os.path.expanduser(
        os.path.expandvars(s))))


def _same_path(a, b) -> bool:
    """True when two path spellings denote the SAME FILE.

    **Never compare raw path strings when deciding sigma eligibility.**
    ``a == b`` on the raw strings answers "different file" for every one of:
    relative vs absolute, a ``~`` prefix, a trailing slash or ``./``
    segment, and a symlink (or a symlinked parent directory).  In the first
    version of the Z_eff gate each of those silently disabled both measured
    tiers and dropped the run to the ASSUMED scalar envelope -- the same
    failure class this feature's own end-to-end A/B originally caught.

    Uses ``os.path.samefile`` (inode identity, so hardlinks and bind mounts
    also compare equal) when both paths exist, and normalised-absolute
    equality otherwise.  A genuinely different file is refused by both
    tests.
    """
    import os

    if not a or not b:
        return False
    na, nb = _norm_path(a), _norm_path(b)
    if na == nb:
        return True
    try:
        return os.path.samefile(na, nb)
    except OSError:          # one or both do not exist -> not the same file
        return False


def zeff_sigma_eligibility(source, ida_path):
    """May a measured Z_eff envelope be paired with this baseline?

    Returns ``(eligible, reason)``.  ``reason`` is ``""`` when eligible and a
    human-readable explanation otherwise, so a refusal can be WARNED about
    and RECORDED instead of silently costing the run its measured tiers.

    The measured tiers are absolute sigmas from one IDA ``.cdf`` and pair
    only with a Z_eff baseline from that same file: the source's own
    ``profiles_path`` (:class:`ReconstructionSource`) or, on the
    EXPERIMENTAL ``ni_source`` routes only, ``ida_path``
    (:class:`ImasSource`; ``zeff_from_fuse=True`` stays eligible, the
    envelope carried absolute).  Refusals: no IDA file configured; an
    :class:`ImasSource` with ``ni_source="standard"`` (its Z_eff is the
    dd's); the source declares no ``.cdf``; different files (compared
    resolved, by :func:`_same_path`).
    """
    import os

    from .config import ImasSource, ReconstructionSource

    if ida_path is None:
        return False, "no IDA sigma file is configured (no ladder applies)"
    ida_name = os.path.basename(_norm_path(ida_path)) or str(ida_path)

    from .experimental import IDA_ION_ROUTES
    if isinstance(source, ReconstructionSource):
        own, own_kind = str(getattr(source, "profiles_path", "") or ""), "profiles file"
    elif (isinstance(source, ImasSource)
          and getattr(source, "ni_source", "standard") not in IDA_ION_ROUTES):
        # ni_source="standard" (as before PR #56): the IMAS/ida_hybrid Z_eff
        # is the dd's, never the IDA one
        return False, (
            "the Z_eff baseline comes from the IMAS/FUSE source rather than "
            f"from {ida_name}; pairing a FUSE Z_eff with an IDA-measured "
            "envelope would mix channels")
    elif isinstance(source, ImasSource):
        own, own_kind = str(getattr(source, "ida_path", "") or ""), "ida_path"
    else:
        return False, (
            f"source type {type(source).__name__} declares no IDA file, so "
            f"its Z_eff baseline did not come from {ida_name}")
    own_norm = _norm_path(own)
    if not own_norm.endswith(".cdf"):
        return False, (
            f"the source's own {own_kind} is not an IDA .cdf "
            f"({os.path.basename(own_norm) or '<unset>'}), so its Z_eff "
            f"baseline did not come from {ida_name}")
    if not _same_path(ida_path, own):
        return False, (
            f"the sigma file ({ida_name}) is a genuinely different file "
            f"from the source's own {own_kind} "
            f"({os.path.basename(own_norm)}) -- compared after expanduser + "
            f"realpath, so this is a real mismatch, not a path spelling")
    return True, ""


def resolve_zeff_envelope(zeff_sigma_source, zeff_scalar_sigma, base_zeff,
                          zeff_is_ida, measured_sigma, measured_source,
                          carbon_sigma=None, carbon_source="none",
                          ineligible_reason="", ida_in_play=False,
                          ladder="standard"):
    """The Z_eff envelope ladder: highest-fidelity tier available per file.

    Returns ``(sigma_array, label, meta)``.

    ``ladder="standard"`` (the default; the source's ``ni_source`` is
    ``"standard"``) -- tiers, in fidelity order:

    1. **carbon-propagated** (``sigma_Zeff_carbon``: n_12C6_err on the direct
       layout, the dilution's own posterior on the ensemble layout).  The
       Zeff-primary scheme perturbs Zeff precisely to move the DILUTION
       ``ni = ne - Z nC``, and CER carbon density is that dilution's direct
       measurement; drawing Zeff with this sigma IS error propagation
       through ``ni = ne - Z nC``.
    2. **VB-measured** (``sigma_Zeff``: the file's Zeff_err / Zeff sample
       spread).  Conservative -- the visible-bremsstrahlung inversion's own
       error, which carries n_e^2 sqrt(T_e) propagation, calibration and
       mantle-subtraction systematics.
    3. the flat ``zeff_scalar_sigma`` fraction of ``|Z_eff|``.

    ``zeff_sigma_source`` picks "auto" (carbon > VB > scalar), "carbon",
    "measured" (VB), "scalar".

    ``ladder="resolved"`` (EXPERIMENTAL, an experimental ``ni_source``;
    ``bouquet.experimental.REGISTRY["ida_ion_route"]``) -- tiers:

    1. **IDA-resolved** (``sigma_Zeff``): the envelope
       :func:`bouquet.io.ida.read_ida` resolved by walking
       ``VB+CER > CER > VB`` over what the file supports, with the route
       disagreement folded in.  ``measured_source`` names the rung it landed
       on, so the label reads e.g. "measured IDA (VB+CER)".  It is the same
       resolution ``ni`` came from, so the sampler cannot get an ``ni`` its
       own ``Z_eff`` fails to reproduce.
    2. the flat ``zeff_scalar_sigma`` fraction of ``|Z_eff|`` -- the
       pre-1.3.2 behaviour, the "scalar" setting, and the loud fallback.

    ``zeff_sigma_source="carbon"`` overrides the resolution with the bare
    carbon-propagated array (``sigma_Zeff_carbon``) for an A/B against the
    combined envelope; it is a single-route override, so the ``ni`` in play
    was NOT necessarily derived from it.

    Both measured tiers are eligible only when the Z_eff baseline itself is
    the IDA one (``zeff_is_ida``, decided by
    :func:`zeff_sigma_eligibility`); a baseline not built from that IDA file
    must not be paired with an IDA envelope.  ``zeff_sigma_source`` picks:
    "auto" (IDA-resolved > scalar), "carbon" (single-route override),
    "measured" (IDA-resolved, warns on fallback), "scalar".

    **No fallback down this ladder is silent.**  Every skipped tier is
    recorded in ``meta["skipped"]`` as ``{"tier", "reason"}`` -- and the
    reason distinguishes an ineligible source (path mismatch / wrong source
    type) from a missing dataset from invalid data -- and a SINGLE
    ``UserWarning`` names the tier chosen, the tiers skipped and why.  That
    warning fires whenever an IDA file was in play at all (so a path
    mismatch cannot pass unnoticed) or whenever a forced ``"carbon"`` /
    ``"measured"`` could not be honoured.  The one case with no warning is
    the one with no ladder: no IDA file configured, or ``"scalar"`` asked
    for explicitly -- both still recorded in ``meta``.

    A chosen envelope whose median fraction of |Zeff| exceeds 25 % draws a
    separate report-only warning naming the alternatives.
    """
    import warnings

    import numpy as np

    if zeff_sigma_source not in ("auto", "carbon", "measured", "scalar"):
        raise ValueError(
            f"zeff_sigma_source={zeff_sigma_source!r} is not one of "
            "'auto', 'carbon', 'measured', 'scalar'")
    if ladder not in ("standard", "resolved"):
        raise ValueError(f"ladder={ladder!r} is not 'standard' or 'resolved'")
    _resolved = ladder == "resolved"
    _measured_tier = "IDA-resolved" if _resolved else "VB-measured"
    base = np.abs(np.asarray(base_zeff, dtype=float))
    scalar_env = float(zeff_scalar_sigma) * base
    scalar_label = f"scalar zeff_scalar_sigma={float(zeff_scalar_sigma):g}"

    skipped = []                      # [(tier, reason)], in ladder order

    def _usable(m, tier):
        """Validated tier array, or None with the reason recorded."""
        if m is None:
            return None
        a = np.asarray(m, dtype=float)
        if a.shape != base.shape:
            skipped.append((tier, f"invalid data: shape {a.shape} does not "
                                  f"match the Z_eff baseline {base.shape}"))
            return None
        if not np.all(np.isfinite(a)):
            skipped.append((tier, "invalid data: non-finite entries"))
            return None
        if np.any(a < 0.0):
            # These are 1-sigma MAGNITUDES.  A negative entry is not a wide
            # band, it is corrupt data, and it propagates a sign into the
            # draw scales -- so `all finite and any > 0` was too weak: it
            # admitted an array with negative entries as long as one entry
            # was positive.
            skipped.append((tier, f"invalid data: {int(np.sum(a < 0.0))} of "
                                  f"{a.size} entries are negative (these are "
                                  f"1-sigma magnitudes)"))
            return None
        if not np.any(a > 0.0):
            skipped.append((tier, "invalid data: all-zero"))
            return None
        return a

    env = label = provenance = None
    tier = None
    if not zeff_is_ida:
        _why = ineligible_reason or (
            "the Z_eff baseline does not come from the IDA file supplying "
            "the sigmas")
        _attempted = {"auto": (("IDA-resolved",) if _resolved else
                               ("carbon-propagated", "VB-measured")),
                      "carbon": ("carbon-propagated", _measured_tier),
                      "measured": (_measured_tier,),
                      "scalar": ()}[zeff_sigma_source]
        for _t in _attempted:
            skipped.append((_t, f"source ineligible: {_why}"))
    else:
        # standard: carbon > VB > scalar.  resolved (experimental): "auto"
        # takes the reader's resolved envelope -- read_ida has already
        # walked VB+CER > CER > VB (see bouquet.io.ida), and it is the route
        # ni was derived from -- and "carbon" stays as an explicit
        # single-route override.
        if (zeff_sigma_source == "carbon"
                or (zeff_sigma_source == "auto" and not _resolved)):
            c = _usable(carbon_sigma, "carbon-propagated")
            if c is not None:
                env, tier = c, "carbon-propagated"
                provenance = str(carbon_source)
                label = f"carbon-propagated ({carbon_source})"
            elif carbon_sigma is None:
                skipped.append(("carbon-propagated",
                                "missing dataset: this file provides no "
                                "n_12C6 uncertainty"))
        if env is None and zeff_sigma_source in ("auto", "carbon", "measured"):
            m = _usable(measured_sigma, _measured_tier)
            if m is not None:
                env, tier = m, _measured_tier
                provenance = str(measured_source)
                label = f"measured IDA ({measured_source})"
                # A rung the reader could not reach is still a fallback, and
                # no fallback down this ladder is silent: read_ida only
                # prints its notice, which a batch run loses.
                if _resolved and str(measured_source) != "VB+CER":
                    skipped.append(
                        ("VB+CER", "missing dataset: this file supports only "
                                   f"the {measured_source} route, so the "
                                   "envelope carries no cross-route check"))
            elif measured_sigma is None:
                skipped.append((_measured_tier,
                                "missing dataset: this IDA file carries no "
                                "usable Z_eff envelope on either route"
                                if _resolved else
                                "missing dataset: this IDA file carries no "
                                "Zeff uncertainty (older direct vintage)"))

    if env is None:
        env, tier = scalar_env, "scalar"
        provenance = f"zeff_scalar_sigma={float(zeff_scalar_sigma):g}"
        label = scalar_label + ("" if zeff_sigma_source == "scalar"
                                else " (FALLBACK -- an ASSUMED width)")

    # ONE warning, naming the tier chosen, the tiers skipped and why.  It
    # fires whenever an IDA file was actually in play (so a path mismatch or
    # a missing dataset can never drop the run to the assumed scalar
    # unnoticed) and whenever a forced measured mode could not be honoured.
    warned = bool(skipped) and (bool(ida_in_play)
                                or zeff_sigma_source in ("carbon", "measured"))
    if warned:
        warnings.warn(
            "resolve_zeff_envelope: Z_eff envelope resolved to the "
            f"{tier.upper()} tier ({label}) with "
            f"zeff_sigma_source={zeff_sigma_source!r}; "
            + "; ".join(f"{_t} skipped -- {_r}" for _t, _r in skipped)
            + ("."
               if tier != "scalar" else
               ".  The scalar envelope is an ASSUMED width, not a measured "
               "one: the resulting ni bands are not error propagation "
               "through ni = ne - Z nC."),
            stacklevel=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        _frac = float(np.median(env / np.clip(base, 1e-3, None)))
    if _frac > 0.25:
        warnings.warn(
            f"resolve_zeff_envelope: the chosen Z_eff envelope "
            f"({label}) has median fraction {100 * _frac:.0f} % of |Zeff| -- "
            f"implausibly large for a 1-sigma dilution uncertainty (VB "
            f"Zeff_err is known to blow up in the SOL and on some shots; "
            f"grand means near 90 % have been observed).  Report-only: "
            f"consider zeff_sigma_source='carbon' (direct dilution "
            f"measurement) or 'scalar'.", stacklevel=2)

    meta = {
        "requested": str(zeff_sigma_source),
        "tier": tier,
        "label": label,
        "provenance": provenance,
        "eligible": bool(zeff_is_ida),
        "ineligible_reason": str(ineligible_reason or ""),
        "skipped": [{"tier": _t, "reason": _r} for _t, _r in skipped],
        "fell_back": bool(skipped),
        "warned": bool(warned),
        "median_fraction_of_zeff": _frac,
    }
    return env, label, meta



#: Fraction of the kinetic grid's psi_N extent over which an input sigma
#: must exceed the profile it perturbs for :func:`sigma_exceeds_profile` to
#: report the channel.  A report threshold only: nothing is clipped,
#: nothing about the sampling changes.
SIGMA_EXCEEDS_PROFILE_MIN_FRACTION = 0.05


def sigma_exceeds_profile(psi_N, profiles, sigmas,
                          min_fraction=SIGMA_EXCEEDS_PROFILE_MIN_FRACTION):
    """Channels whose 1-sigma envelope exceeds the profile itself over at
    least *min_fraction* of the radius (REPORT ONLY).

    A Gaussian draw of a positive profile with ``sigma > value`` is negative
    with probability > 16 % at that radius; such a draw is non-physical and
    is rejected (``kinetics_nonphysical``), so a batch on such an input
    yields little and its statistics say nothing about the solve.

    *profiles* / *sigmas*: ``{"ne" | "te" | "ni" | "ti": array}`` on
    *psi_N*.  The radial fraction is measured in ``psi_N`` (trapezoid
    weights), not in node count, so a grid dense near the axis is not
    over-counted.  Returns a list of ``dict(channel, fraction, psi_N_range,
    max_ratio, max_ratio_psi_N, min_fraction)``, empty when nothing
    qualifies.
    """
    import numpy as np
    x = np.asarray(psi_N, dtype=float)
    if x.ndim != 1 or x.size < 2:
        return []
    w = np.gradient(x)
    w = np.abs(w) / float(np.sum(np.abs(w)))
    out = []
    for ch in ("ne", "te", "ni", "ti"):
        if ch not in profiles or ch not in sigmas:
            continue
        v = np.abs(np.asarray(profiles[ch], dtype=float))
        s = np.asarray(sigmas[ch], dtype=float)
        if v.shape != x.shape or s.shape != x.shape:
            continue
        over = np.isfinite(s) & np.isfinite(v) & (s > v)
        frac = float(np.sum(w[over]))
        if not over.any() or frac < float(min_fraction):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(v > 0.0, s / v, np.inf)
        i = int(np.argmax(np.where(over, ratio, -np.inf)))
        out.append(dict(channel=ch, fraction=frac,
                        psi_N_range=[float(x[over][0]), float(x[over][-1])],
                        max_ratio=(float(ratio[i]) if np.isfinite(ratio[i])
                                   else None),
                        max_ratio_psi_N=float(x[i]),
                        min_fraction=float(min_fraction)))
    return out


def sigma_exceeds_profile_line(records) -> str:
    """The one-line report of :func:`sigma_exceeds_profile` records."""
    parts = []
    for r in records:
        mr = ("inf" if r["max_ratio"] is None else f"{r['max_ratio']:.2g}")
        parts.append(
            f"sigma_{r['channel']} > {r['channel']} over "
            f"{100.0 * r['fraction']:.0f}% of psi_N "
            f"({r['psi_N_range'][0]:.2f}-{r['psi_N_range'][1]:.2f}; max "
            f"sigma/value {mr} at psi_N={r['max_ratio_psi_N']:.2f})")
    return ("[sigma-check] WARNING: " + "; ".join(parts) + " -- Gaussian "
            "draws go non-positive there and are rejected "
            "(kinetics_nonphysical); expect a low yield.  Report only: "
            "nothing is clipped and the sampling is unchanged (threshold: "
            f"{100.0 * records[0]['min_fraction']:.0f}% of the radius).")


def resolve_uncertainty(config, baseline) -> dict:
    """Resolve the perturbation envelope for :func:`generate_bouquet`.

    Returns kinetic sigmas (on ``baseline.psi_N_kinetic``), ``sigma_jphi`` (a
    fractional envelope on ``|baseline.j_phi|``, on ``baseline.psi_N``), and the
    GPR correlation length scales.

    Kinetic sigma source precedence:
      1. ``UncertaintyConfig.ida_path`` if set;
      2. else the reconstruction source's own IDA ``.cdf`` (sigmas read once);
      3. else a per-channel flat fraction ``<chan>_scalar_sigma * |profile|``.
    An explicit ``UncertaintyConfig.sigma_profiles[chan]`` array overrides all.

    The resolved source is LOGGED, one line per channel (disable with
    ``UncertaintyConfig.log_sigma_sources=False``), and a ``UserWarning`` is
    raised when a ``<chan>_scalar_sigma`` was moved off its default but loses
    the precedence to an IDA file -- most importantly the case where a user
    zeroes the scalars intending a sigma=0 run and silently gets the full
    operational IDA envelope instead.  See :class:`UncertaintyConfig` for the
    full contract and the way to actually force zero.
    """
    import os
    import warnings

    import numpy as np
    from .config import ImasSource, ReconstructionSource, UncertaintyConfig

    # Read the defaults off the dataclass so "the user changed this" can never
    # drift from the declared defaults.
    _SCALAR_SIGMA_DEFAULTS = {
        _c: float(getattr(UncertaintyConfig, f"{_c}_scalar_sigma"))
        for _c in ("ne", "te", "ni", "ti")
    }

    unc = config.uncertainty
    src = config.source
    psi_kin = np.asarray(baseline.psi_N_kinetic, dtype=float)

    # The PR #56 Z_eff / n_i route (EXPERIMENTAL; bouquet.experimental.
    # REGISTRY["ida_ion_route"]) or the standard one.
    from .experimental import STANDARD_NI_SOURCE, ida_ion_route_on
    _ion_route = ida_ion_route_on(src, unc)
    # IDA file for the sigmas: unc.ida_path, else the source's own .cdf.
    ida_path = unc.ida_path
    if ida_path is None:
        if isinstance(src, ReconstructionSource) \
                and src.profiles_path.endswith(".cdf"):
            ida_path = src.profiles_path
        elif isinstance(src, ImasSource) and _ion_route:
            # EXPERIMENTAL route only: the IMAS source's own IDA file; under
            # ni_source="standard" only unc.ida_path supplies IDA sigmas
            # (from_imas sets it), as before PR #56
            ida_path = getattr(src, "ida_path", None)

    # IDA arrays (read once) available as a fallback below
    ida_sig = None
    _sig_match = None       # a separate sigma file's IDA slice match (#73)
    _ida_zeff_sigma, _ida_zeff_source = None, "none"
    _ida_zeff_carbon, _ida_zeff_carbon_source = None, "none"
    if ida_path is not None:
        from .io.ida import read_ida
        # Reuse the ida_hybrid read (the slice the kinetics came from).
        _shared = (baseline.aux or {}).get("ida_profiles")
        ida = (_shared[1] if _shared is not None
               and _same_path(_shared[0], ida_path) else None)
        if ida is None:
            _t = getattr(src, "ida_time", None)
            _t_req = getattr(src, "time", None) if _t is None else _t
            # the source's OWN kinetics file is read at the slice its
            # kinetics were (the reconstruction loader's nearest slice); a
            # SEPARATE sigma file (UncertaintyConfig.ida_path) by the IDA
            # time rule of #73 -- half-step window, dt recorded, refused
            # outside (review #73 B2/B6 residual)
            _own = (src.profiles_path
                    if isinstance(src, ReconstructionSource) else
                    getattr(src, "ida_path", None))
            if not (_own and _same_path(_own, ida_path)):
                from .io.ida import ida_time_base
                from .io.imas import match_ida_slice
                _sig_match = match_ida_slice(
                    ida_time_base(ida_path), _t_req,
                    what="uncertainty.ida_path")
                _t_req = _sig_match["ida_time_used"]
            ida = read_ida(
                ida_path, time=_t_req,
                sigma_mode=unc.sigma_mode, sigma_method=unc.sigma_method,
                ni_source=(getattr(src, "ni_source", STANDARD_NI_SOURCE)
                           if _ion_route else STANDARD_NI_SOURCE),
                # the carbon tier's Z(Z-1) propagation is quadratically
                # Z-sensitive; the kinetics loader already passes this, and
                # omitting it here silently pinned the sigma math to carbon
                impurity_Z=float(getattr(src, "impurity_Z", 6.0)),
            )
            if _sig_match is not None and abs(
                    float(ida.time) - _sig_match["ida_time_used"]) > 1e-9:
                raise RuntimeError(
                    f"uncertainty.ida_path: read_ida returned the slice at "
                    f"{float(ida.time)!r} s, not the matched "
                    f"{_sig_match['ida_time_used']!r} s")

        _ida_x, _ida_in = np.asarray(ida.psi_N, dtype=float), slice(None)
        if baseline.psi_map is not None:
            # psi_N -> run coordinate (inside the LCFS): through the source's
            # own map, or a file other than the source's by its own q.
            _map_src = (src.profiles_path if isinstance(src, ReconstructionSource)
                        and src.profiles_path.endswith(".cdf") else
                        _shared[0] if _shared is not None else None)
            if (getattr(ida, "q", None) is not None
                    and not (_map_src and _same_path(_map_src, ida_path))):
                from .coords import phi_n_from_q
                _ida_in, _ida_x = phi_n_from_q(_ida_x, ida.q, bracket=True)
            else:
                _ida_in = _ida_x <= 1.0
                _ida_x = np.interp(_ida_x[_ida_in], *baseline.psi_map)

        def _to_kin(arr):
            return np.interp(psi_kin, _ida_x, np.asarray(arr, dtype=float)[_ida_in])

        ida_sig = {"ne": _to_kin(ida.sigma_ne), "te": _to_kin(ida.sigma_te),
                   "ni": _to_kin(ida.sigma_ni), "ti": _to_kin(ida.sigma_ti)}
        # The ida_hybrid read's sigma_ni, placed on the baseline grid with
        # the baseline ni itself.
        _sni = (baseline.aux or {}).get("sigma_ni_ida")
        if (_ion_route and _shared is not None and ida is _shared[1]
                and _sni is not None and np.shape(_sni) == psi_kin.shape):
            ida_sig["ni"] = np.asarray(_sni, dtype=float)
        if getattr(ida, "sigma_Zeff", None) is not None:
            _ida_zeff_sigma = _to_kin(ida.sigma_Zeff)
            _ida_zeff_source = str(getattr(ida, "sigma_Zeff_source", "?"))
        if getattr(ida, "sigma_Zeff_carbon", None) is not None:
            _ida_zeff_carbon = _to_kin(ida.sigma_Zeff_carbon)
            _ida_zeff_carbon_source = str(
                getattr(ida, "sigma_Zeff_carbon_source", "?"))

    # Per-channel resolution: explicit profile > IDA > flat scalar fraction.
    _profiles = unc.sigma_profiles or {}
    _scalars = {"ne": unc.ne_scalar_sigma, "te": unc.te_scalar_sigma,
                "ni": unc.ni_scalar_sigma, "ti": unc.ti_scalar_sigma}
    _baseprof = {"ne": baseline.ne, "te": baseline.te,
                 "ni": baseline.ni, "ti": baseline.ti}
    out = {}
    # a separate sigma file's IDA slice and how it was matched (#73 rule):
    # on the envelope and in the baseline record (archived as li_metrics)
    if _sig_match is not None:
        out["ida_sigma_time_match"] = dict(_sig_match)
        baseline.li_metrics = dict(getattr(baseline, "li_metrics", None)
                                   or {}, ida_sigma_time_match=dict(
                                       _sig_match))
    _won = {}       # channel -> human-readable winning source (for the log)
    _shadowed = []  # channels whose deliberately-set scalar lost to the IDA
    for _ch in ("ne", "te", "ni", "ti"):
        if _ch in _profiles and _profiles[_ch] is not None:
            _arr = np.asarray(_profiles[_ch], dtype=float)
            if _arr.shape != psi_kin.shape:
                raise ValueError(
                    f"sigma_profiles['{_ch}'] has shape {_arr.shape}; expected "
                    f"the kinetic grid {psi_kin.shape} (psi_N_kinetic)")
            out[f"sigma_{_ch}"] = _arr
            _won[_ch] = "explicit sigma_profiles"
        elif ida_sig is not None:
            out[f"sigma_{_ch}"] = ida_sig[_ch]
            _won[_ch] = f"IDA {os.path.basename(ida_path)}"
            # A scalar the user moved off its default is being SILENTLY
            # ignored. Zeroing them is the classic footgun: it reads like
            # "no perturbation" but leaves the full operational IDA envelope
            # in place, so a run intended as deterministic is a full-sigma
            # ensemble (the 2026-08 beta-scan case).
            if float(_scalars[_ch]) != _SCALAR_SIGMA_DEFAULTS[_ch]:
                _shadowed.append(
                    f"{_ch}_scalar_sigma={float(_scalars[_ch]):g}")
        else:
            out[f"sigma_{_ch}"] = float(_scalars[_ch]) * np.abs(
                np.asarray(_baseprof[_ch], dtype=float))
            _won[_ch] = f"scalar {float(_scalars[_ch]):g} x |baseline|"

    # REPORT ONLY: a kinetic sigma larger than its own profile over a stated
    # fraction of the radius (draws through zero are rejected, never clipped)
    try:
        out["sigma_exceeds_profile"] = sigma_exceeds_profile(
            psi_kin, _baseprof, {_c: out[f"sigma_{_c}"]
                                 for _c in ("ne", "te", "ni", "ti")})
    except Exception:           # a report must never fail the resolution
        out["sigma_exceeds_profile"] = []
    if out["sigma_exceeds_profile"]:
        print("  " + sigma_exceeds_profile_line(out["sigma_exceeds_profile"]),
              flush=True)

    out["sigma_jphi"] = unc.jphi_scalar_sigma * np.abs(np.asarray(baseline.j_phi, dtype=float))
    _won["jphi"] = f"scalar {float(unc.jphi_scalar_sigma):g} x |j_phi|"
    out["n_ls"], out["t_ls"], out["j_ls"] = unc.n_ls, unc.t_ls, unc.j_ls

    # --- precedence audit trail ---------------------------------------------
    # One line per channel naming the winner and the resolved magnitude. The
    # precedence is silent by construction (a winning source simply shadows the
    # others), and a silent win can invert the meaning of a whole run, so it is
    # worth the four lines. Turn off with UncertaintyConfig.log_sigma_sources.
    if bool(getattr(unc, "log_sigma_sources", True)):
        for _ch in ("ne", "te", "ni", "ti", "jphi"):
            _s = np.asarray(out[f"sigma_{_ch}"], dtype=float)
            _pk = float(np.max(np.abs(_s))) if _s.size else 0.0
            print(f"  [sigma-source] sigma_{_ch:<4s} <- {_won[_ch]:<28s} "
                  f"(peak {_pk:.4g}{', ALL ZERO' if _pk == 0.0 else ''})")

    if _shadowed:
        warnings.warn(
            f"resolve_uncertainty: {', '.join(_shadowed)} was set but IGNORED "
            f"-- an IDA source ({os.path.basename(ida_path)}) is active and "
            f"wins the kinetic sigma precedence "
            f"(sigma_profiles > IDA .cdf > <chan>_scalar_sigma). The resolved "
            f"sigmas are the full IDA envelope, NOT your scalars. To force a "
            f"specific envelope (e.g. zero, for a deterministic run) pass "
            f"explicit arrays instead:  "
            f"unc.sigma_profiles = {{ch: np.zeros_like(baseline.psi_N_kinetic) "
            f"for ch in ('ne','te','ni','ti')}}  -- or clear "
            f"UncertaintyConfig.ida_path / use a non-.cdf profiles_path.",
            stacklevel=2,
        )

    # --- switchboard: resolve the auxiliary perturbed profiles ---------------
    # A sigma entry enables a profile. Baseline = manual (aux_baselines) over
    # source-provided (baseline.aux). Warn + skip if the baseline is absent or
    # all-zero (the user supplied a sigma for nothing).
    src_aux = dict(baseline.aux or {})
    man_base = dict(unc.aux_baselines or {})

    # Z_eff channel is enabled by default for EVERY source (the consistent
    # density scheme): unless the user set an explicit aux_sigmas['zeff'], the
    # envelope is the IDA Zeff_err when an IDA is in play, else a flat fractional
    # zeff_scalar_sigma * Z_eff_baseline. zeff_scalar_sigma still GATES the
    # channel either way (0.0 -> disabled). The baseline Z_eff is source-provided
    # (baseline.aux['zeff'] for IMAS, else baseline.Zeff), on the kinetic grid.
    user_sigmas = dict(unc.aux_sigmas or {})
    # Provenance of the Z_eff envelope, always present: None when the ladder
    # never ran (an explicit aux_sigmas['zeff'], or the channel disabled),
    # else the full tier record -- chosen tier, skipped tiers and why.
    out["zeff_sigma_tier"] = None
    if "zeff" not in user_sigmas and float(getattr(unc, "zeff_scalar_sigma", 0.0)) > 0:
        base_zeff = src_aux.get("zeff")
        if base_zeff is None:
            base_zeff = np.asarray(baseline.Zeff, dtype=float)
        if "zeff" not in man_base:
            man_base["zeff"] = np.asarray(base_zeff, dtype=float)
        # Measured-tier eligibility is decided by the SOURCE TYPE and by FILE
        # IDENTITY, not by whether baseline.aux carries a 'zeff' entry: the
        # reconstruction path also populates aux['zeff'] (it IS the IDA
        # Zeff, stored for the aux plots), so testing the aux dict wrongly
        # disqualified every recon-path run -- caught by an end-to-end A/B,
        # where the 'auto' arm silently resolved to the scalar.  The file
        # test lives in zeff_sigma_eligibility() and compares RESOLVED paths
        # (expanduser + realpath, samefile when both exist): a raw string
        # comparison re-introduces exactly the same silent drop for a
        # relative-vs-absolute, '~'-prefixed, trailing-slash or symlinked
        # spelling of the very same file.  It returns the REASON as well as
        # the verdict so the fallback can be warned about and recorded.
        _zeff_baseline_is_ida, _zeff_inelig = zeff_sigma_eligibility(
            src, ida_path)
        _z_env, _z_label, _z_meta = resolve_zeff_envelope(
            getattr(unc, "zeff_sigma_source", "auto"),
            unc.zeff_scalar_sigma,
            man_base["zeff"],
            zeff_is_ida=_zeff_baseline_is_ida,
            measured_sigma=_ida_zeff_sigma,
            measured_source=_ida_zeff_source,
            carbon_sigma=_ida_zeff_carbon,
            carbon_source=_ida_zeff_carbon_source,
            ineligible_reason=_zeff_inelig,
            ida_in_play=ida_path is not None,
            ladder="resolved" if _ion_route else "standard",
        )
        user_sigmas["zeff"] = _z_env
        out["zeff_sigma_tier"] = _z_meta
        if getattr(unc, "log_sigma_sources", True):
            print(f"[sigma] zeff envelope <- {_z_label}")
            for _sk in _z_meta["skipped"]:
                print(f"[sigma]   {_sk['tier']} tier skipped: "
                      f"{_sk['reason']}")

    # --- who draws ni when the zeff channel is active ------------------------
    # Standard route: always derived (as before PR #56).  EXPERIMENTAL route
    # (PR #56 rule): derived per draw from the drawn (ne, Zeff) when ni and
    # Z_eff are one IDA resolution (zeff_dne keeps sigma_ni) or ni has only
    # the scalar fallback; any other real ni envelope stays its own channel.
    # unc.ni_from_zeff wins either way.
    _ida_pair = bool(
        _ion_route and ida_sig is not None and _won["ni"].startswith("IDA")
        and (out["zeff_sigma_tier"] or {}).get("tier") == "IDA-resolved"
        and "zeff" not in (unc.aux_baselines or {})
        and not getattr(src, "zeff_from_fuse", False)
        and (isinstance(src, ReconstructionSource)
             or "ida_profiles" in (baseline.aux or {}))
        and getattr(ida, "zeff_dne", None) is not None)
    _nfz = getattr(unc, "ni_from_zeff", None)
    out["ni_from_zeff"] = (
        bool(_nfz) if _nfz is not None else
        bool(_won["ni"].startswith("scalar") or _ida_pair) if _ion_route
        else True)
    # the PR #56 sampler clips (EXPERIMENTAL; kinetic_sampler_clips): the
    # explicit setting, else on only with a PR #56 kinetic feature
    from .experimental import kinetic_clips_on
    out["kinetic_clips"] = bool(kinetic_clips_on(config))
    out["zeff_dne"] = (_to_kin(ida.zeff_dne)
                       if _ida_pair and out["ni_from_zeff"] else None)
    if bool(getattr(unc, "log_sigma_sources", True)) and "zeff" in user_sigmas:
        print("  [sigma-source] ni per draw   <- "
              + ("derived from the drawn (ne, Z_eff)"
                 + (" (one IDA resolution)" if out["zeff_dne"] is not None else "")
                 if out["ni_from_zeff"] else "its own sigma_ni (independent of Z_eff)"))

    resolved_sigma, resolved_base = {}, {}
    for name, sig in user_sigmas.items():
        base = man_base.get(name, src_aux.get(name))
        if base is None or not np.any(np.asarray(base, dtype=float)):
            warnings.warn(
                f"aux_sigmas['{name}'] supplied but its baseline is "
                f"{'absent' if base is None else 'all-zero'} in this source; "
                f"perturbation of '{name}' is skipped. Provide it via "
                f"UncertaintyConfig.aux_baselines."
            )
            continue
        resolved_sigma[name] = np.asarray(sig, dtype=float)
        resolved_base[name] = np.asarray(base, dtype=float)
    out["aux_sigmas"] = resolved_sigma
    out["aux_baselines"] = resolved_base
    out["aux_length_scales"] = dict(unc.aux_length_scales or {})
    return out


def floor_inductive_split(j_inductive, j_BS, psi_N=None, warn_frac=1e-4,
                          coord="psi_n"):
    """Enforce the ``j_inductive >= 0`` component convention on a (j_ind, j_BS)
    split, absorbing any negative sliver into ``j_BS`` so the pair still sums
    exactly to the same total.

    A strong pedestal can push the achieved total BELOW the full-Sauter
    bootstrap locally, leaving a small negative inductive residual. That is
    unphysical in this convention and -- fed to the GPR sampler as its mean --
    makes essentially every current draw go negative and be rejected. Returns
    ``(j_inductive_floored, j_BS_adjusted)`` (copies; inputs untouched) and
    prints a one-line note when the correction is non-trivial.  ``psi_N`` is
    the grid, in ``coord`` (labels the note).
    """
    import numpy as np

    j_ind = np.asarray(j_inductive, dtype=float)
    jbs = np.asarray(j_BS, dtype=float)
    if not np.any(j_ind < 0.0):
        return j_ind, jbs
    j_ind_f = np.maximum(j_ind, 0.0)
    deficit = j_ind - j_ind_f                    # <= 0 where floored
    jbs_f = jbs + deficit                        # absorb -> sum unchanged
    scale = float(np.max(np.abs(j_ind))) or 1.0
    worst = float(-deficit.min())
    if worst / scale > warn_frac:
        n = int(np.sum(deficit < 0.0))
        where = ""
        if psi_N is not None:
            pn = np.asarray(psi_N, dtype=float)
            sel = pn[deficit < 0.0]
            lab = "Phi_N" if coord == "phi_n" else "psi_N"
            where = f" over {lab} [{sel.min():.3f}, {sel.max():.3f}]"
        print(f"  [baseline] floored negative j_inductive ({n} pts{where}, "
              f"worst {worst/1e6:.4f} MA/m^2, {100*worst/scale:.2f}% of peak) "
              f"-- deficit absorbed into j_BS (split still sums to j_phi)")
    return j_ind_f, jbs_f


def _resolve_fixed(comp, src_psi, dst_psi):
    """Fixed additive component onto ``dst_psi`` (zeros if not supplied)."""
    import numpy as np

    if comp is None:
        return np.zeros_like(dst_psi)
    comp = np.asarray(comp, dtype=float)
    if src_psi is None:
        if comp.shape != dst_psi.shape:
            raise ValueError(
                "fixed-component array length does not match the target grid; "
                "provide FixedComponentsConfig.psi_N for resampling"
            )
        return comp
    return np.interp(dst_psi, np.asarray(src_psi, dtype=float), comp)


def _load_kinetic_profiles(source) -> dict:
    """Kinetic profiles (SI) from the reconstruction source's ``profiles_path``.

    Dispatches on file type: an IDA ``.cdf`` (ni = ne, quasi-neutrality) or an
    Osborne p-file (real ni via quasineutrality + Zeff from the ion mix). Returns
    a dict with ``psi_N`` (native kinetic grid), ``ne``/``te``/``ni``/``ti``
    (m^-3, eV), ``Zeff`` (clipped >= 1), and ``raw_bytes`` for archival.
    """
    import numpy as np

    path = source.profiles_path
    if path.endswith(".cdf"):
        from .io.ida import read_ida
        ida = read_ida(path, time=source.time, impurity_Z=source.impurity_Z,
                       ni_source=getattr(source, "ni_source", "standard"))
        return dict(
            psi_N=np.asarray(ida.psi_N, dtype=float),
            ne=np.asarray(ida.ne, dtype=float),
            te=np.asarray(ida.te, dtype=float),
            ni=np.asarray(ida.ni, dtype=float),
            ti=np.asarray(ida.ti, dtype=float),
            Zeff=np.clip(np.asarray(ida.Zeff, dtype=float), 1.0, None),
            raw_bytes=ida.raw_bytes,
            q=None if ida.q is None else np.asarray(ida.q, dtype=float),
            # how IDA's Z_eff / n_i were resolved (rung, weights, window,
            # clamps; PR #56) -- archived with the baseline record
            zeff_provenance=getattr(ida, "zeff_provenance", None),
        )

    # Osborne p-file: ne/ni in 1e20 m^-3, Te/Ti in keV -> SI.
    from .io.pfile import read_pfile
    with open(path, "rb") as fh:
        raw = fh.read()
    pf = read_pfile(path)
    if pf.ion_species is None:
        raise ValueError(
            "p-file carries no ion species in its footer; cannot compute "
            "quasineutrality / Zeff"
        )
    pf.compute_quasineutrality()
    psi_pf, Zeff = pf.compute_zeff()
    return dict(
        psi_N=np.asarray(psi_pf, dtype=float),
        ne=np.asarray(pf.ne, dtype=float) * 1e20,
        te=np.asarray(pf.te, dtype=float) * 1e3,
        ni=np.asarray(pf.ni, dtype=float) * 1e20,
        ti=np.asarray(pf.ti, dtype=float) * 1e3,
        Zeff=np.clip(np.asarray(Zeff, dtype=float), 1.0, None),
        raw_bytes=raw,
    )


def _resolve_reconstruction(source, config, mygs) -> Baseline:
    """Reconstruction-source baseline: GS reconstruct on a live ``mygs``.

    Mirrors the operational notebook: read g-file + IDA profiles, interpolate
    onto the g-file's nodes (in the run coordinate), run :func:`reconstruct_equilibrium`, and package
    the (toroidal) fitted currents. The reconstructed total ``j_phi_fit`` already
    contains all driven current, so fixed components (j_NBI / j_RF) default to
    zero and only re-partition the inductive part if the user supplies them;
    ``p_fast`` (absent from thermal IDA profiles) likewise defaults to zero.
    """
    import numpy as np

    from .io.geqdsk import read_geqdsk
    from .TokaMaker_interface import reconstruct_equilibrium

    if mygs is None:
        raise ValueError(
            "ReconstructionSource requires a live TokaMaker solver; call "
            "setup_solver() before prepare_baseline()"
        )
    if source.profile_overrides:
        raise NotImplementedError("profile_overrides is not yet applied")

    with open(source.geqdsk_path, "rb") as fh:
        eqdsk_bytes = fh.read()
    eqdsk = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
    psi_N = np.asarray(eqdsk.psi_N, dtype=float)

    kin = _load_kinetic_profiles(source)
    psi_N_kin = kin["psi_N"]

    # Run grids (x_run: the g-file's nodes; x_kin: the kinetic nodes) in the
    # run coordinate (coords.gfile_run_grids); psi_N / psi_N_kin stay the
    # sources' ψ_N.
    from . import coords
    coord = coords.run_coord(getattr(source, "coord", coords.PSI))
    x_run, x_kin, _in, kin = coords.gfile_run_grids(
        eqdsk, kin, source.profiles_path, coord)
    psi_map = None if _in is None else (np.asarray(psi_N_kin)[_in], x_kin)

    # kinetic profiles (native SI) regridded onto the equilibrium nodes.
    # Shape-preserving PCHIP (single shared helper): a linear regrid leaves a
    # slope kink at every kinetic knot, which the Sauter bootstrap inherits
    # as a stepped j_BS (see utils.pchip_interp).
    from .utils import pchip_interp

    def to_eq(arr):
        return pchip_interp(x_kin, arr, x_run)

    ne_eq, te_eq, ni_eq, ti_eq = to_eq(kin["ne"]), to_eq(kin["te"]), to_eq(kin["ni"]), to_eq(kin["ti"])
    Zeff_eq = np.clip(to_eq(kin["Zeff"]), 1.0, None)

    # isoflux from the g-file boundary (matches the recon-stage notebook weight)
    iso_pts = np.column_stack([eqdsk.boundary_R, eqdsk.boundary_Z])
    iso_w = np.ones(len(iso_pts)) * 200.0
    mygs.set_isoflux(iso_pts, weights=iso_w)

    # Seed shape in ψ_N (the g-file's nodes).
    guess_jinductive = coords.swb_seed(x_run, psi_N)
    _recon_coord = ({} if coord == coords.PSI else
                    dict(coord=coord, x=x_run))

    # Fixed (non-perturbed) pressure components must be resolved BEFORE the
    # reconstruction, not after it: the reconstruction's GS pressure has to be
    # the same pressure every draw solves, otherwise l_i_target is measured on
    # a lower-pressure equilibrium than the draws it targets (see the pressure
    # block in reconstruct_equilibrium).
    #
    # Grid: p_fast is resolved onto the KINETIC grid first and then mapped to
    # the equilibrium grid with `to_eq` -- deliberately the same two-step path
    # the draws take (baseline resolves onto x_kin, then
    # perturb_kinetic_equilibrium applies `_kin_to_eq`, which is the identical
    # pchip_interp).  Resolving fc.psi_N -> x_run in one hop would be a
    # slightly different array and would reintroduce the very inconsistency
    # this is fixing.  `p_fast_kin` is also what the returned Baseline.p_fast
    # field carries (kinetic grid), which downstream depends on -- unchanged.
    #
    # When fc.p_fast is unset, pass None rather than the zeros _resolve_fixed
    # returns, so the default-off path does not even enter the new branch and
    # is provably a no-op (not merely "adds 0.0").
    fc = config.fixed_components
    # fc.psi_N in the run coordinate (a psi_n input via the g-file's map).
    fc_x = coords.to_run_grid(fc.psi_N, getattr(fc, "coord", coords.RUN),
                              None if coord == coords.PSI else (psi_N, x_run))
    p_fast_kin = _resolve_fixed(fc.p_fast, fc_x, x_kin)
    p_fast_eq = to_eq(p_fast_kin) if fc.p_fast is not None else None
    # Z_imp is plumbed for symmetry with the draw path, but is INERT here today:
    # FixedComponentsConfig (config.py) carries no Z_imp field at all -- Z_imp is
    # an *output* field of the Baseline dataclass, populated only on the IMAS
    # path. getattr keeps this a no-op now and makes it activate automatically
    # if the config ever gains the field, rather than silently diverging again.
    Z_imp_recon = getattr(fc, "Z_imp", None)

    # Capture the verbose solver chatter (DLSODE / gs_get_qprof / li-match) unless
    # the user asked for it; the curated summary is printed by Bouquet.reconstruct.
    from .utils import capture_native_output
    verbose = bool(getattr(config, "verbose", False))
    # Self-consistent bootstrap (GenerationConfig.jbs_self_consistent): the
    # loop wraps fit + l_i match + corrective; None -> the legacy SWB path.
    from .jbs_loop import jbs_settings as _jbs_settings
    _jbs = _jbs_settings(config.generation)
    _jbs_kw = {"jbs_loop": _jbs} if _jbs["enabled"] else {}
    from .edge_pressure import resolve_edge_pressure
    _edge = resolve_edge_pressure(config.generation)
    with capture_native_output(enabled=not verbose) as _cap:
        result = reconstruct_equilibrium(
            mygs, eqdsk,
            ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
            iso_pts, iso_w, source.psi_pad,
            guess_jinductive=guess_jinductive,
            n_k=source.n_k,
            psi_bridge=source.psi_bridge,
            rescale_j_BS=source.rescale_j_BS,
            shelf_psi_N=source.shelf_psi_N,
            initialize_psi=True,
            p_fast=p_fast_eq,
            Z_imp=Z_imp_recon,
            l_i_tolerance=float(config.generation.l_i_tolerance),
            **_recon_coord,
            edge_pressure=_edge,
            **_jbs_kw,
            bootstrap_kwargs=config.generation.bootstrap_kwargs,
        )
        # get_stats traces the q-profile and can emit gs_get_qprof warnings, so
        # keep these inside the capture too.
        Ip_target = abs(float(eqdsk.Ip))
        # l_i scale: 'iter' == li(3) == 2*int(Bp^2 dV)/((mu0 Ip)^2 R_axis).
        # This is the estimator reconstruct_equilibrium targets (the g-file's
        # `li(2)` key, which is numerically the same functional) and the ONLY
        # one the two codes agree on (0.17%).  The whole downstream chain --
        # per-draw acceptance band, archive attrs, plots -- is on this scale;
        # see issue #20.  DO NOT mix with 'std'/li(1) numbers.
        #
        # WHICH l_i (issue #25).  This used to be a fresh get_stats read of
        # whatever state the reconstruction happened to END on -- i.e. AFTER
        # step 7's corrective iteration, which runs up to 8 further GS solves
        # with a stopping rule that knows nothing about l_i.  So the number the
        # whole ensemble is banded around was not the value step 6 matched; it
        # was that value plus however far step 7 drifted (measured +0.50 to
        # +0.76 % across the beta-scan family, against a 1 % per-draw band).
        #
        # `result['li_final']` IS the step-6 matched value: the one the secant
        # loop drove onto the g-file's li(2), and therefore the only one with a
        # defensible provenance as a target.  The post-corrective read is kept
        # as a separate archived diagnostic rather than dropped, so this change
        # ADDS provenance -- `l_i_realized_post_corrective` says where the
        # reconstruction actually finished, and its distance from the target is
        # reported loudly by reconstruct_equilibrium when it exceeds the band.
        #
        # `l_i_target` INTENTIONALLY DIFFERS FROM THE PRE-#27 ARCHIVED VALUE by
        # exactly the step-7 corrective drift -- that IS the change described
        # above, not a side effect of it.  Worked example, main @ 4ad4894 on
        # the synthetic golden: the old target (the post-corrective read) was
        # 0.656074, the new one (step-6 matched) is 0.653840, and
        # 0.653840 * 1.003416 = 0.656074 -- the +0.342 % recorded in
        # `li_corrective_drift_pct` and nothing else.  Still far in-band
        # (+/-5.00 %, `li_corrective_out_of_band` False).  This is also the
        # direct cause of the sigma=0 R2 l_i residual reading -0.083 % rather
        # than the pre-#27 -0.457 %: route R2 skips the corrective iteration,
        # so the old target was charging it for a drift it never applies (see
        # `_LI_REL` in tests/test_seeded_reproducibility.py, and issue #28).
        l_i_target = float(result["li_final"])
        # THE NOTE BELOW IS ABOUT THIS LINE ONLY -- a bit-neutral refactor, not
        # a claim about `l_i_target` (which moves by design, see above).
        # Consume the value reconstruct_equilibrium already measured at step 7b
        # rather than re-reading get_stats here.  The re-read was redundant --
        # same solver state, same lcfs_pad (source.psi_pad is exactly what was
        # passed in above), same li_normalization='iter' -- and it cost a
        # second q-profile trace plus its gs_get_qprof warnings.  It also had
        # the WEAKER provenance of the two: it reported whatever state the
        # solver happens to be in at THIS line, which is "post step 7" only for
        # as long as nothing in between touches the equilibrium.  Consuming the
        # returned field pins the number to the step it is named after.
        # Verified bit-identical to the old re-read on the synthetic golden
        # fixture (delta exactly 0.0), so no archived value moves.  RE-VERIFIED
        # on main @ 4ad4894 for issue #28, capturing both at this exact point
        # in the control flow: old re-read and consumed value are both
        # 0.6560742094407394, delta exactly 0.0.  Nothing between step 7b and
        # here mutates the equilibrium, so the "only for as long as" caveat
        # above still holds and is still worth keeping.
        l_i_realized_post_corrective = float(
            result["li_realized_post_corrective"])
        recon_metrics = _reconstruction_metrics(
            mygs, eqdsk, result, source, l_i_target,
            l_i_realized_post_corrective=l_i_realized_post_corrective,
            edge_pressure=_edge)
        if result.get("jbs_loop") is not None:
            from .jbs_loop import jsonable as _jsonable
            recon_metrics = dict(recon_metrics or {})
            recon_metrics["jbs_loop"] = _jsonable(result["jbs_loop"])
        # the +/-50 % bootstrap prior on the legacy g-file path: the inductive
        # fit's bootstrap scale (1.0 unless rescale_j_BS) -- flagged when
        # |s_bs - 1| > 0.5 (a closure failure, never clamped), recorded
        from .utils import bootstrap_prior_record, merge_closure_flags
        recon_metrics = dict(recon_metrics or {})
        recon_metrics["closure_health"] = bootstrap_prior_record(
            float(result.get("bs_scale_fit", 1.0)),
            "fit_inductive_profile's bootstrap scale (rescale_j_BS; 1.0 "
            "otherwise)", "g-file reconstruction")
        merge_closure_flags(recon_metrics, recon_metrics["closure_health"])

    j_phi = np.asarray(result["j_phi_fit"], dtype=float)
    j_BS = np.asarray(result["j_BS_used"], dtype=float)

    j_NBI = _resolve_fixed(fc.j_NBI, fc_x, x_run)
    j_RF = _resolve_fixed(fc.j_RF, fc_x, x_run)
    j_other = _resolve_fixed(getattr(fc, "j_other", None), fc_x, x_run)
    _request_offset = None
    _delivered = None
    if _jbs["enabled"] and result.get("request_jphi") is not None:
        # The self-consistent loop: store the ONE reconstruction state in
        # the form the draws consume it (see Baseline.jphi_request_offset).
        _log2 = None
        with capture_native_output(enabled=not verbose) as _cap2:
            j_phi, j_inductive, j_BS, _request_offset, _delivered = \
                _deliver_reconstruction_state(
                    mygs, config, source, result, x_run, ne_eq, te_eq, ni_eq,
                    ti_eq, Zeff_eq, Ip_target, l_i_target, j_NBI,
                    j_RF + j_other,     # every fixed part (j_other: unified)
                    recon_metrics, coord=coord)
        _log2 = _cap2["text"] or None
        if _log2:
            _cap["text"] = (_cap["text"] or "") + _log2
        if _delivered["n_floored_inductive"]:
            # visible (outside the capture): a floored point is one where a
            # zero-perturbation draw cannot reproduce the reconstruction
            print(f"  [delivered state] {_delivered['n_floored_inductive']} "
                  "point(s) of the request inductive were floored at zero: "
                  "a zero-perturbation draw cannot reproduce the "
                  "reconstruction there (it composes the Redl bootstrap, not "
                  "the floored remainder)", flush=True)
    else:
        j_inductive = j_phi - j_BS - j_NBI - j_RF - j_other   # == j_inductive_fit when all 0
        # Physical component convention: the inductive current is >= 0. On shots
        # with a strong pedestal the achieved total can dip BELOW the full-Sauter
        # bootstrap there, leaving a small negative residual (~1% of the core) --
        # which, fed to the GPR sampler as its mean, makes essentially every draw
        # go negative and be rejected (observed on a strong-pedestal case:
        # 0/500 candidates survived).
        # Floor the inductive at zero and absorb the deficit into j_BS so the
        # split still sums exactly to j_phi.
        j_inductive, j_BS = floor_inductive_split(j_inductive, j_BS, x_run,
                                                  coord=coord)

    # Resolved above (before the reconstruction, which now consumes it).
    # Unchanged contract: the returned field is on the KINETIC grid.
    p_fast = p_fast_kin

    # Report-only core-pressure hollowness record built inside the
    # reconstruction (see physics.core_pressure_hollow_record).
    _cph_recon = (result.get("quality") or {}).get("core_pressure_hollow")

    return Baseline(
        psi_N=x_run,
        j_phi=j_phi,
        j_inductive=j_inductive,
        j_BS=j_BS,
        psi_N_kinetic=x_kin,
        coord=coord,
        psi_map=psi_map,
        ne=kin["ne"],
        te=kin["te"],
        ni=kin["ni"],
        ti=kin["ti"],
        Zeff=kin["Zeff"],
        Ip_target=Ip_target,
        l_i_target=l_i_target,
        provenance="reconstruction",
        j_NBI=j_NBI,
        j_RF=j_RF,
        j_other=j_other,
        p_fast=p_fast,
        # SAME source as the value handed to the reconstruction above, so the
        # two paths activate together or not at all.  The draws read
        # `Baseline.Z_imp` (run.py hands it to generate_bouquet, and the
        # forward / sigma=0 solves read it directly); the recon reads
        # `Z_imp_recon`.  Leaving this at its None default while feeding
        # `Z_imp_recon` to the recon meant that the day FixedComponentsConfig
        # gains a Z_imp field, the reconstruction would start adding impurity
        # pressure while the draws still did not -- silently recreating the
        # exact recon-vs-draw pressure inconsistency this plumbing exists to
        # remove, through the very getattr that was meant to guard it.
        # Today both are None on this path, so this is a no-op.
        Z_imp=Z_imp_recon,
        eqdsk_bytes=eqdsk_bytes,
        pfile_bytes=kin["raw_bytes"],
        # expose the baseline Z_eff (kinetic grid) as a switchboard channel,
        # mirroring the IMAS path -- so the zeff channel resolves its baseline
        # for both the default auto-injection and an explicit aux_sigmas['zeff']
        aux={"zeff": np.asarray(kin["Zeff"], dtype=float)},
        recon=result,
        reconstruction_metrics=recon_metrics,
        reconstruction_log=_cap["text"] or None,
        # Lifted out of the reconstruction's own quality block so both source
        # paths expose the record under one name, and carried in li_metrics so
        # store_baseline_profiles archives it (report-only; see
        # physics.core_pressure_hollow_record).
        core_pressure_hollow=_cph_recon,
        li_metrics=({k: v for k, v in (
            ("core_pressure_hollow", _cph_recon),
            # PR #56 (owner item E2 stamp): how the IDA Z_eff / n_i were
            # resolved; archived as _baseline li_metrics_json
            ("zeff_provenance", kin.get("zeff_provenance")),
            # review PR64 B1: how SWB's bootstrap was converted (absent on
            # the self-consistent loop, which runs no SWB)
            ("swb_conversion", result.get("swb_conversion"))) if v}
            or None),
        jphi_request_offset=_request_offset,
        delivered_state=_delivered,
        edge_pressure=(recon_metrics or {}).get("edge_pressure"),
    )


def _deliver_reconstruction_state(mygs, config, source, result, psi_N, ne_eq,
                                  te_eq, ni_eq, ti_eq, Zeff_eq, Ip_target,
                                  l_i_target, j_NBI, j_RF, recon_metrics,
                                  coord="psi_n"):
    """The reconstruction path's ONE state, stored in the draws' form
    (``jbs_self_consistent=True`` only).

    ``mygs`` holds the reconstruction's final equilibrium F (the l_i
    re-matched state; ``result["request_jphi"]`` is the jphi-linterp input one
    solve of which is F).  The bootstrap is the draws' own sigma=0 composition
    on F (:func:`~bouquet.TokaMaker_interface._draw_jbs_composer` with the
    generation's ``isolate_edge_jBS`` / ``floor_j_BS``, scale 1 -- this
    path's ``bs_scale``), the request is normalised to ``Ip_target`` in the
    'exact' measure on F, and the inductive is the residual, floored at zero
    by the usual convention (``floor_inductive_split``; a floored point is
    one where a zero-perturbation draw cannot reproduce F, and is counted).
    Returns ``(j_phi, j_inductive, j_BS, request_offset, delivered_state)``.
    """
    import numpy as np
    from .TokaMaker_interface import (DELIVERED_SPLIT_CONVENTION,
                                      _achieved_jphi_fsa,
                                      _deliver_request_split,
                                      _draw_jbs_composer, _request_offset)
    gc = config.generation
    psi_pad = float(source.psi_pad)
    comp = _draw_jbs_composer(psi_N, ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
                              psi_pad, bool(gc.isolate_edge_jBS), 1.0,
                              bool(gc.floor_j_BS), None, None, None,
                              coord=coord)
    j_bs0 = np.asarray(comp(mygs.copy_eq())[0], dtype=float)
    fixed = np.asarray(j_NBI, dtype=float) + np.asarray(j_RF, dtype=float)
    dv = _deliver_request_split(mygs, psi_N, psi_pad, Ip_target,
                                result["request_jphi"], j_bs0, fixed,
                                label="recon delivered state", coord=coord)
    j_ind, j_BS = floor_inductive_split(dv["j_inductive"], j_bs0, psi_N,
                                        coord=coord)
    n_floored = int(np.sum(np.asarray(dv["j_inductive"]) < 0.0))
    j_phi = j_ind + j_BS + fixed          # == dv["request"] (floor: sum kept)
    offset, n_fl_t = _request_offset(j_ind, dv["achieved"], j_bs0, fixed)
    from .physics import SOLVER_Q0_PSI_N
    m = recon_metrics or {}
    state = dict(
        convention=DELIVERED_SPLIT_CONVENTION,
        path="reconstruction",
        l_i=float(l_i_target), l_i_scale="iter(li3)",
        q0=float(m.get("q0", float("nan"))),
        q0_psi_N=float(m.get("q0_psi_N", SOLVER_Q0_PSI_N)),
        q95=float(m.get("q95", float("nan"))),
        Ip_target=float(Ip_target),
        edge_pressure=m.get("edge_pressure"),
        request_normalisation=float(dv["kappa"]),
        achieved_normalisation=float(dv["kappa_achieved"]),
        n_floored_inductive=n_floored,
        n_floored_target_inductive=n_fl_t,
        li_corrective_state=result.get("li_corrective_state"),
        li_step6_matched=result.get("li_step6_matched"),
        li_input=float((result.get("eqdsk_li") or {}).get(
            "li(2)", float("nan"))),
        j_phi_achieved=_achieved_jphi_fsa(mygs, psi_N, psi_pad,
                                          sign_ref=j_phi, coord=coord),
        how=("step-7 corrective iteration, then the l_i re-match of its "
             "landed request (every loop pass ends there); one jphi-linterp "
             "solve of j_phi reproduces it"))
    return j_phi, j_ind, j_BS, offset, state


def _reconstruction_metrics(mygs, eqdsk, result, source, l_i_achieved,
                            l_i_realized_post_corrective=None,
                            edge_pressure=None) -> dict:
    """Curate a TokaMaker-vs-EFIT reconstruction-fidelity dict for the summary.

    Each global scalar that isn't ~0 by construction (Ip, l_i, q0/q95, beta,
    shape, separatrix current, stored energy) is reported as a % error against
    the g-file's own EFIT value. Geometric residuals that *should* be ~0
    (boundary match, axis offset, j_phi profile RMS) stay absolute. A simple
    pass/fail verdict is applied to the reconstruction targets.
    """
    import numpy as np

    def pct(tok, ref):
        ref = float(ref)
        if not np.isfinite(ref) or abs(ref) < 1e-12:
            return float("nan")
        return float(100.0 * (float(tok) - ref) / abs(ref))

    from .physics import SOLVER_Q0_PSI_N as _Q0_PSI_N
    q = dict(result.get("quality") or {})
    # Global scalars other than l_i are normalization-independent; take them
    # from the 'iter' call so there is exactly one get_stats scale in play.
    stats = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="iter")
    # The FREE (untargeted) li(1) pair, for the cross-estimator report below.
    stats_std = mygs.get_stats(lcfs_pad=source.psi_pad, li_normalization="std")

    # --- TokaMaker (reconstructed) values ---
    Ip_tok = float(result.get("Ip_tokamaker", float("nan")))
    li_tok = float(l_i_achieved)
    q0_tok = float(stats.get("q_0", float("nan")))
    q95_tok = float(stats.get("q_95", float("nan")))
    betan_tok = float(stats.get("beta_n", float("nan")))
    betap_tok = float(stats.get("beta_pol", float("nan"))) / 100.0  # x100 -> standard
    kappa_tok = float(stats.get("kappa", float("nan")))
    delta_tok = float(stats.get("delta", float("nan")))
    W_tok = float(stats.get("W_MHD", float("nan"))) / 1e6           # MJ
    # Both pressure frames (bouquet.edge_pressure).  The solver's pressure is
    # zero at the boundary; under separatrix_pressure="offset" the solve
    # pressure's p_sep was removed from the axis target and is added back
    # here, so the REPORTED beta / W_MHD are the full-pressure ones.  With
    # nothing added back the two frames are the solver's own numbers.
    from .edge_pressure import (archive_record, input_pressure_frames,
                                resolve_edge_pressure)
    _edge = resolve_edge_pressure(edge_pressure)
    from .edge_pressure import solver_p_scale
    _edge_rec = archive_record(_edge, result.get("pres_tokamaker"),
                               stats=stats, p_scale=solver_p_scale(mygs))
    _fr = _edge_rec.get("frames")
    if _fr is not None and _fr["p_sep"] != 0.0:
        betan_tok = float(_fr["full"].get("beta_n", float("nan")))
        betap_tok = float(_fr["full"].get("beta_pol", float("nan"))) / 100.0
        W_tok = float(_fr["full"]["W_MHD"]) / 1e6
    o_point = getattr(mygs, "o_point", [float("nan"), float("nan")])
    # separatrix current evaluated just inside the LCFS (psi_N=0.99), where the
    # edge current is better-defined than the near-singular psi_N=1 point.
    _psiN_eq = np.asarray(eqdsk.psi_N, dtype=float)
    _jsep_psiN = 0.99
    jphi = np.asarray(result.get("j_phi_fit"), dtype=float)
    jsep_tok = float(np.interp(_jsep_psiN, _psiN_eq, jphi)) / 1e6   # MA/m^2

    # --- EFIT (g-file) reference values; tolerate a tracing failure ---
    efit = {}
    try:
        efit["Ip"] = abs(float(eqdsk.Ip))
        # `li(2)` is the g-file-side li(3)/'iter' functional -- the estimator
        # the reconstruction actually targets (issue #20).  `li(1)` is kept
        # alongside it for the free cross-estimator pair.
        efit["li"] = float(eqdsk.li.get("li(2)", float("nan")))
        efit["li1"] = float(eqdsk.li.get("li(1)_EFIT", float("nan")))
        qpsi = np.asarray(eqdsk.qpsi, dtype=float)
        psiN = np.asarray(eqdsk.psi_N, dtype=float)
        # like for like: the solver's q0 (get_stats 'q_0') is q at psi_N =
        # SOLVER_Q0_PSI_N, so the g-file is read at the SAME radius; its axis
        # value (psi_N = 0) is kept under its own name
        efit["q0"] = float(np.interp(_Q0_PSI_N, psiN, qpsi))
        efit["q0_axis"] = float(qpsi[0])
        efit["q95"] = float(np.interp(0.95, psiN, qpsi))
        betas = eqdsk.betas
        efit["beta_n"] = float(betas.get("beta_n", float("nan")))
        efit["beta_p"] = float(betas.get("beta_p", float("nan")))
        geo = eqdsk.geometry
        efit["kappa"] = float(np.asarray(geo["kappa"])[-1])
        efit["delta"] = float(np.asarray(geo["delta"])[-1])
        efit["R_mag"] = float(eqdsk.R_mag)
        efit["Z_mag"] = float(eqdsk.Z_mag)
        efit["W_MHD"] = 1.5 * float(eqdsk.volume_integral(eqdsk.pres)[-1]) / 1e6
        jtor_efit = np.asarray(result.get("eqdsk_jtor"), dtype=float)
        efit["j_sep"] = float(np.interp(_jsep_psiN, _psiN_eq, jtor_efit)) / 1e6
    except Exception as exc:  # tracing/format issue -> % errors fall back to nan
        import warnings
        warnings.warn(f"EFIT reference metrics unavailable: {exc}")

    g = lambda k: float(efit.get(k, float("nan")))

    # Like for like: the solver frame against the input's (p - p_edge)
    # quantities, the full frame against the input's full-pressure ones.
    _like = None
    try:
        _pres_in = np.asarray(eqdsk.pres, dtype=float)
        _in = input_pressure_frames(
            float(eqdsk.volume_integral(np.ones_like(_pres_in))[-1]),
            float(eqdsk.volume_integral(_pres_in)[-1]), float(_pres_in[-1]),
            betas=dict(beta_n=g("beta_n"), beta_p=g("beta_p")))
        if _fr is not None:
            def _row(tok, ref):
                return dict(tokamaker=float(tok), input=float(ref),
                            err_pct=pct(tok, ref))
            _like = dict(p_edge_input=_in["p_sep"], frames={
                _k: dict(
                    beta_n=_row(_fr[_k].get("beta_n", float("nan")),
                                _in[_k]["beta_n"]),
                    beta_p=_row(_fr[_k].get("beta_pol", float("nan")) / 100.0,
                                _in[_k]["beta_p"]),
                    W_MHD_MJ=_row(_fr[_k]["W_MHD"] / 1e6,
                                  _in[_k]["W_MHD"] / 1e6))
                for _k in ("solver", "full")},
                note=("solver: the solver's own pressure (zero at psi_N = 1) "
                      "against the input's p - p_edge; full: with p_sep "
                      "added back against the input's full pressure"))
    except Exception as exc:
        import warnings
        warnings.warn(f"like-for-like pressure frames unavailable: {exc}")

    bnd_rms = float(q.get("boundary_rms_mm", float("nan")))
    bnd_max = float(q.get("boundary_max_dev_mm", float("nan")))
    axis_off_mm = float(np.hypot(o_point[0] - g("R_mag"),
                                 o_point[1] - g("Z_mag")) * 1e3)

    # Convergence of the final accepted GS state. reconstruct_equilibrium's
    # li-match only ever keeps successful solves (a failed solve restores the
    # last good psi and is retried), and an unrecoverable solver failure
    # raises before reaching here -- so the honest residual check is that the
    # final state produced finite global quantities.
    converged = bool(np.isfinite(Ip_tok) and np.isfinite(li_tok)
                     and np.isfinite(q95_tok))

    # verdict on the reconstruction *targets*: converged, Ip within 0.5%,
    # boundary RMS < 5 mm, l_i within 5%
    ip_err = pct(Ip_tok, g("Ip"))
    li_err = pct(li_tok, g("li"))
    li1_tok = float(stats_std.get("l_i", float("nan")))
    li1_cross_err = pct(li1_tok, g("li1"))
    ok = bool(
        converged
        and np.isfinite(ip_err) and abs(ip_err) < 0.5
        and np.isfinite(bnd_rms) and bnd_rms < 5.0
        and np.isfinite(li_err) and abs(li_err) < 5.0
    )
    return {
        "converged": converged,
        "verdict": "PASS" if ok else "CHECK",
        # fidelity: TokaMaker value, EFIT reference, % error
        "Ip_MA": Ip_tok / 1e6, "Ip_efit_MA": g("Ip") / 1e6, "Ip_err_pct": ip_err,
        # --- l_i, on the estimator the loop targets ('iter' == li(3)) -------
        # `li_err_pct` is the MATCHED pair: the secant loop drives these two
        # numbers together, so it is ~0 by construction and cannot detect an
        # estimator mismatch.  That is precisely how issue #20 stayed hidden.
        #
        # SINCE #25 IT IS VACUOUS OUTRIGHT, not merely weak.  `li` is now
        # `l_i_target` == `result['li_final']`, the step-6 MATCHED value; so
        # `li_err_pct` compares the secant's converged output against the very
        # target it converged onto, and reports nothing but that loop's own
        # tolerance.  Before #25 it was read post-step-7, so it at least
        # carried the corrective iteration's drift.  It is kept for continuity
        # of the summary table and because a NON-tiny value would mean the
        # secant did not converge -- but it is not a fidelity measure.
        # For "where did the reconstruction actually finish", read
        # `li_realized_post_corrective` / `li_corrective_drift_pct` below; for
        # a genuinely free estimator comparison, read the li(1) pair further
        # down, which nothing drives together.
        "li": li_tok, "li_efit": g("li"), "li_err_pct": li_err,
        "li_scale": "iter(li3)",
        # --- step-6 matched vs step-7 realized (issue #25) ------------------
        # `li` above (== l_i_target) is the MATCHED value.  These say where the
        # corrective iteration actually left the equilibrium, and whether that
        # is further from the target than a draw is allowed to be.  Archived,
        # never enforced: a visible condition, not an acceptance criterion.
        "li_realized_post_corrective": (
            float("nan") if l_i_realized_post_corrective is None
            else float(l_i_realized_post_corrective)),
        "li_corrective_drift_pct": float(
            q.get("li3_corrective_drift_pct", float("nan"))),
        "li_corrective_band_pct": float(
            q.get("li3_corrective_band_pct", float("nan"))),
        "li_corrective_out_of_band": bool(
            q.get("li3_corrective_out_of_band", False)),
        # --- cross-estimator drift monitor (de-circularization, issue #20) --
        # The FREE pair: TokaMaker's 'std'/li(1) vs the g-file's li(1)_EFIT.
        # Nothing drives these together, so a nonzero residual is a genuine
        # measurement of convention/geometry drift between the two codes
        # (historically ~+3.3% on DIII-D geqdsks, from TokaMaker's projected
        # separatrix perimeter).  If this ever collapses to ~0 or changes
        # sharply, an estimator moved -- investigate before trusting the run.
        "li1_cross": li1_tok, "li1_cross_efit": g("li1"),
        "li1_cross_err_pct": li1_cross_err,
        # q0 at LIKE radii: TokaMaker's reported q0 and the g-file's q, both
        # at psi_N = q0_psi_N (0.02, the solver's first traced surface).  The
        # g-file's own axis value (psi_N = 0) and the solver-vs-axis error --
        # what "q0_err_pct" used to report -- are kept under honest names.
        "q0": q0_tok, "q0_psi_N": _Q0_PSI_N,
        "q0_efit": g("q0"), "q0_err_pct": pct(q0_tok, g("q0")),
        "q0_efit_axis": g("q0_axis"),
        "q0_err_pct_vs_axis": pct(q0_tok, g("q0_axis")),
        "q95": q95_tok, "q95_efit": g("q95"), "q95_err_pct": pct(q95_tok, g("q95")),
        "beta_n": betan_tok, "beta_n_efit": g("beta_n"),
        "beta_n_err_pct": pct(betan_tok, g("beta_n")),
        "beta_p": betap_tok, "beta_p_efit": g("beta_p"),
        "beta_p_err_pct": pct(betap_tok, g("beta_p")),
        "kappa": kappa_tok, "kappa_efit": g("kappa"),
        "kappa_err_pct": pct(kappa_tok, g("kappa")),
        "delta": delta_tok, "delta_efit": g("delta"),
        "delta_err_pct": pct(delta_tok, g("delta")),
        "j_sep_MA": jsep_tok, "j_sep_efit_MA": g("j_sep"),
        "j_sep_err_pct": pct(jsep_tok, g("j_sep")),
        "W_MHD_MJ": W_tok, "W_MHD_efit_MJ": g("W_MHD"),
        "W_MHD_err_pct": pct(W_tok, g("W_MHD")),
        # the edge-pressure settings, p_sep, both frames of the delivered
        # equilibrium, and the like-for-like comparison with the input
        "edge_pressure": _edge_rec,
        "pressure_like_for_like": _like,
        # zero-ideal residuals (absolute)
        "boundary_rms_mm": bnd_rms, "boundary_max_mm": bnd_max,
        "axis_offset_mm": axis_off_mm,
        "jphi_core_rms_MA": float(q.get("jphi_core_rms", float("nan"))) / 1e6,
        "jphi_edge_rms_MA": float(q.get("jphi_edge_rms", float("nan"))) / 1e6,
        # fit_inductive_profile's l_i-proxy amplitude search fell back to 1.0
        # (no bracket): recorded with reason and bracket, never silent
        "ind_scale_fallback": bool(q.get("ind_scale_fallback", False)),
        "ind_scale_fallback_n": int(q.get("ind_scale_fallback_n", 0) or 0),
        "ind_scale_fallback_records": [
            dict(r) for r in (q.get("ind_scale_fallback_records") or ())],
    }
