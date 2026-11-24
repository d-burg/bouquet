r"""Time-dependent current evolution through a series of reconstructed time slices.

The ohmic current is not assumed relaxed: the poloidal-flux state of :mod:`.current.diffusion`
(psi on the domain's rho_tor,norm cells) is carried from slice to slice and advanced with

.. code-block:: text

    a dpsi/dt|_rho = d/drho (k psi') - S + a w psi'           (diffusion module, eqs. 6-7)

with, on each interval ``[t_{k-1}, t_k]``,

* geometry ``a, k, V', <1/R^2>, F, Phi_b`` interpolated linearly in time between the equilibria of
  slices ``k-1`` and ``k`` (:class:`.current.diffusion.GeometrySchedule`; ``w = rho dPhi_b/dt /
  (2 Phi_b)`` from the change of the domain's toroidal flux);
* conductivity (Redl 2021 from the kinetic profiles AT EACH SLICE, on that slice's geometry),
  bootstrap (Redl 2021) and driven current (provider) at both ends, linear in time;
* the plasma current ``Ip(t)`` linear between the slices' values (which values: "Plasma current in a
  series" below), applied at the last closed surface through the lumped outer layer (its conductance, source
  and ``|F|`` at the last closed surface linear in time between the two slices' values, so that ``Ip/F`` at
  both ends of an interval is the slice's own; ``layer_fpol="interval_mean"`` holds ``|F|`` at the mean of
  the two values, the earlier treatment, kept for comparison);
* the vacuum field: every slice is solved with its own ``F0 = R0 B0`` (:attr:`SliceData.F0`, the
  measured value at that time; when it is not given the previous slice's value is kept and the
  slice record says so), so ``F`` at the last closed surface differs between the two ends of an
  interval (recorded as ``fpol_lcfs_used`` and ``fpol_lcfs_change``);
* the core treatment.  Default ``sawtooth="q_floor"``: the TIME-AVERAGED state, with the resistive q floor
  (:class:`.current.sawtooth.ResistiveFloor`, the resistivity step of the stationary-sawtooth treatment FUSE runs;
  ``q_floor`` is :attr:`.current_model.CurrentModelSettings.q_floor`, default 1): on every diffusion step the
  resistivity inside the SINGLE-SLICE flattening region of the step's model sources at the time of the step (the
  region a single-slice fit would use for those sources, :func:`.current.sawtooth.resistive_floor_steady`: from the
  axis to the smallest edge whose steady state has q >= floor, where the steady q is below the floor on axis; none
  where it is below the floor off axis only or nowhere) is set to its value at the region's edge (cut cells
  overlap-weighted), so the core current relaxes toward the single-slice floored state of the same sources; the
  current stays continuous and Ip is the interval's Ip condition.  q is NOT held at the floor: in a transient it
  can sit below or above it; a series that stands still in time ends on the single-slice state.  The region does
  not depend on the evolving state (which is read only for the refusal and the record), so it does not switch on and
  off from step to step (FUSE's rule, from the previous step's q, does; module docstring of
  :mod:`.current.sawtooth`).  A state whose q at the domain edge is below the floor is refused.  No sawtooth
  detection, crash time or amplitude is used.  The region is decided once per step, at the step's start time.  The
  first slice starts from the same treatment as a single-slice fit (``core_relaxation="q_floor"``, its steady-state
  form).
  An interval's in-window current corrections (below) are NOT floored: their response over the interval (the advance
  with them among the sources minus the advance without them, both with the plain resistivity; the flux diffusion
  is linear in its sources) is added to the model end state, so the floor acts on the model state first and the
  corrections after it, as in a single-slice fit.  From the next interval on that state is the model state: where a
  correction left q below the floor at the slice, the floor acts on it in the next interval (the flattened
  resistivity relaxes that part of the correction away over the resistive time), so a sub-floor core is not carried
  unchanged from one interval to the next.  ``CurrentModelSettings.q_floor_form="sharp"`` selects the earlier clamp
  (:func:`.current.sawtooth.floor_q` after every step; opt-in, comparison).  Every slice records its core treatment
  (``core``: ``core_relaxation``, ``q_floor`` and its form, whether the floor acted during the interval, the
  flattening radius at its end, every crossing of the floor by the model end state's q, inner and outer, and the
  minimum of the scored q; the floor's effect over the interval against the same interval advanced without it),
  ``core_vs_floor`` (:func:`.current_model.core_vs_floor_record`) and ``floor_effect``
  (:func:`.current_model.floor_effect_record`), both records only.
  The reconnection operator (:mod:`.current.sawtooth`, the identity wherever q >= 1) is an explicit opt-in:
  ``sawtooth="crash_times"`` applies it at MEASURED crash times (``crash_table``, a
  :class:`.crash_table.CrashTable`, or ``crash_times``; a table or times given with another mode are
  refused); whenever the evolved q falls below 1 on a slice with no crash acting, the slice record carries
  ``q_below_one`` with its kind (:func:`_sawtooth_warning`), and it is a ``sawtooth_warning`` listed in the
  series' ``warnings`` unless the record shows sawtoothing that covers it (q < 1 within the measured crash
  spacing since the last recorded crash is normal between crashes; a recorded crash that did not act, an
  uncovered interval, or a covered crash-free stretch longer than any recorded crash interval -- a model-data
  inconsistency -- are warnings).  ``sawtooth="off"``: no core treatment at all (q may fall below 1; recorded
  as ``q_below_one`` of kind ``no_record`` with a warning).  The periodic train is available only as
  the explicit option ``sawtooth="periodic"`` (recorded):
  ``sawtooth="crash_times"`` uses measured crash times; ``sawtooth="periodic"`` places crashes every
  ``1/nu`` with ``nu`` from the two parameters ``(f, ln Lambda)``; ``ln Lambda`` is fitted at the
  first slice when the initial state is a first-slice fit (``"mse_fit"``, ``"magnetics_fit"``; a free parameter of the first-slice fit by
  default, as in single-slice fits) and that value sets the train, otherwise it is
  ``sawtooth_log_strength`` (default: the fit module's prior mean).  Phase of the periodic train
  (stated, not measured): the first slice's state is the CYCLE AVERAGE of the periodic orbit (initial
  ``"relaxed"`` and the first-slice fits), and the first crash of the train is one period AFTER the first
  slice time (``t_0 + 1/nu``, ``t_0 + 2/nu``, ...; ``sawtooth_first_crash_phase`` moves it to
  ``t_0 + phase/nu`` for sensitivity studies).  With measured crash times the first slice's state
  is the window average of the measured-crash slice run of the current model over the window
  ``[t_0 - (t_1 - t_0), t_0]`` (the first interval's length; the run starts on the periodic orbit at
  the measured mean interval ``history`` before the window, so the crash record must cover it; a record
  that covers the history and holds no crash in it starts the series crash-free, recorded; a record that did
  not observe the whole history is refused whatever the number of crashes recorded in it, and so are bare
  ``crash_times`` without a table, which state no coverage), and
  the later crashes are the measured ones; ``"off"``: none.  Every evolved slice is self-consistent: the state it
  scores is the state it carries forward, the instantaneous end state of its interval at the slice time, after
  every crash in the interval and (filter) with the fit's corrections (``crash_slice_state="instantaneous"``,
  default; recorded per slice in ``scored_vs_carried_state``).  A crash just before a slice time therefore shows
  in that slice in full.  ``crash_slice_state="window_average"`` is the earlier treatment, kept as a comparison
  option only: the slice scores the crash-free end state plus the sawtooth effect averaged over the interval
  (``psi = psi_end^(no crash) + <psi^(crashes)>_window - <psi^(no crash)>_window``) while it carries the
  instantaneous state, so on a slice whose interval holds an active crash the scored and the carried states
  differ (``same_state = False``).  With a periodic train whose period is longer than the interval the scored
  state depends on the phase (stated).

Plasma current in a series (:data:`IP_CONVENTION`, recorded in the result and per slice).  A fit treats
the Ip target as a parameter (``ip_rel``, :mod:`.fit`).  The relative offset ``d = (Ip_fitted -
Ip_measured)/Ip_measured`` of the latest fitted Ip is carried forward as a slowly varying nuisance: each
interval is driven from the carried state's current to the next slice's measured Ip times ``(1 + d)``, a
prediction slice is solved to that target, and only an ABSORBED filter fit changes ``d`` (the carried state
is then rescaled once, by ``(1 + d_new)/(1 + d_old)``, never again for the same offset).  A constant true offset
of Ip therefore gives the evolution with the offset current all along, and the first interval after a
first-slice fit has no spurious loop-voltage step.  Every slice record carries the interval's Ip end values
(``interval.ip_start_A``, ``ip_end_A``, ``ip_offset_used``) and the carried ``ip_carried_A`` /
``ip_offset_carried``.

Splitting and its order.  Within an interval the end-of-step equilibrium (slice ``k``) is not known
in advance: each pass of the slice's pass-consistency loop (:func:`.current_model.consistent_solve`
with ``state_fn``) re-advances the diffusion from the CARRIED state at ``t_{k-1}`` over the whole
interval with the geometry, conductivity, sources and outer layer of the current pass's slice-``k``
equilibrium at ``t_k`` (and slice ``k-1``'s at ``t_{k-1}``), assembles ``<j_phi>`` from the advanced
state on the slice-``k`` geometry and solves the anchored equilibrium; the loop ends when the
profiles, the state scalars and the q check are consistent (the module's criteria).  This is an
implicit (Picard) coupling of the diffusion and the equilibrium within each step: the coefficients
seen by the TR-BDF2 stages are linear interpolants between consistent end states, so the coupled
step is second order in ``Delta t`` in the converged limit (the diffusion integrator itself is
second order; the outer-layer ``|F|`` freeze is first order in its small variation).  The state
carried to the next interval is the state the slice scored: after the loop the interval is re-advanced on the
equilibrium of the accepted pass (the loop's last state-function call is made on the equilibrium of the
following solve, which is not the scored one).

Initial state (first slice):

* ``"mse_fit"``: the first slice's current is fitted to magnetics and E_r-corrected MSE by
  :func:`.fit.reconstruct` with the current-correction Gaussians on top of the relaxed model shape
  (the first slice's core treatment: the q-floor state by default, an opt-in sawtooth state otherwise);
  the fitted state (whose ohmic
  part then carries the corrections, i.e. is NOT relaxed) starts the evolution, and the
  corrections are not applied again;
* ``"magnetics_fit"``: the same first-slice fit with MSE out of its fit groups (magnetics, coil currents and
  plasma current only), for a discharge with no usable MSE; the same acceptance standard and labels apply.
  On nine discharges whose MSE was withheld and only scored, a series started this way stayed close to the
  series fitted to MSE, where a series started from the relaxed state did not;
* ``"relaxed"``: the model state of the first slice under its core treatment (the q-floor state by default);
* ``"from_profile"``: a supplied current profile ``<J.B>(rho_tor,norm)`` (e.g. from a prior
  equilibrium; controlled tests), held as a profile through the first slice's pass loop, the state
  being the one with that current on the converged geometry.  By default the profile is used as
  given (its integral sets the current inside the domain edge, and the outer layer carries the rest
  of Ip, which can be negative when the profile was made on other surfaces);
  ``initial_profile_domain_fraction = f`` scales it on every pass so that the current inside the
  domain edge is ``f Ip`` (e.g. the producing equilibrium's own enclosed-current fraction at that
  surface; the scale factor is recorded in the state's ``meta`` and in every slice record as
  ``initial_profile_scale``).  ``f`` lies in (0, 1] by default; a fraction slightly above 1 (a
  reference whose edge current is slightly negative) is accepted only with an explicit
  ``initial_profile_domain_fraction_tolerance``, and the excess ``f - 1`` is recorded.  (Holding a psi(rho) or q(rho) state
  FIXED while the geometry iterates was tried and did not converge on the public fixture: the passes
  drift by ~4e-4 in l_i per pass without and oscillate with Anderson mixing; not offered.)
* ``"auto"``: ``"mse_fit"`` when the first slice has an MSE group declared E_r-corrected; else
  ``"magnetics_fit"`` when it has a flux-loop or a poloidal-field-probe group; else ``"relaxed"`` (the
  choice is recorded as ``initial_auto_choice``).  A magnetics-only first-slice fit that does not end
  converged refuses the series like any first-slice fit; ``initial="relaxed"`` asks for the relaxed
  state explicitly.

A first-slice fit must end ``"converged"`` or ``"converged_within_noise"`` to start a series (the
standard the filter applies to its fits; :data:`ACCEPTED_FIT_STATUSES`; a within-noise fit stopped at
the optimum to within the measured noise and carries its label into the first slice's record,
``fit.status`` and ``start_label``): any other status -- stalled, max_iter, a failed Jacobian, failed
trials, a failure -- refuses the series with the reason
(``status="refused_initial_fit_not_converged"``), unless ``initial_fit_nonconverged="proceed_labeled"``
was set, in which case the series starts from that fit's center and the start is labeled in the result
(``initial_fit_nonconverged_start``) and in the first slice's fit record (``start_label``).

Fit interaction: (i) default -- the initial state is fitted once; later slices are evolved and
SCORED with no fitting (a prediction test); (ii) ``filter=True`` -- at every later slice the
correction amplitudes are re-fitted with the evolved state as the prior mean (corrections
N(0, sigma) on top of it); with ``filter_times`` given, only at the slices whose time lies within
``filter_time_tol`` of one of those times (anchor fits; every other slice is evolved and scored
without fitting, and its record says ``filter: None``).  A filter fit with status ``"converged"`` or
``"converged_within_noise"`` is absorbed (the label is kept in the slice's ``filter.status``); any other fit
status (stalled, max_iter, failed) leaves the slice at its evolved, unfitted equilibrium and state, which are
scored and carried; the record says ``absorbed: False`` with the fit's status.

How the corrections enter the evolution (``filter_correction="in_window"``, default).  The corrections are
a non-inductive current SOURCE ``sum_k A_k g_k`` (Gaussians on rho_tor,norm, amplitudes fractions of Ip,
:func:`.current_model.correction_shapes`, on each end's geometry) of the flux-diffusion equation over the
interval that ends at the slice: the diffusion over that interval decides their response (a source acting
for a time short against the resistive time changes the current near its own location with a local return;
only after many resistive times does the response approach the zero-net-current steady form,
:func:`.current_model.correction_response`, with its return distributed like the conductivity).  The net
current is unchanged by construction: the interval is driven at the same Ip boundary value with or without
the source, so the ohmic current returns what the source adds.  A filter fit's parameters ARE the source
amplitudes ``a_k`` over the interval that ends at its slice (the prior N(0, sigma) on each); the fit's response
to them is the in-window response over that interval: the fit evaluates every trial state by advancing the
interval with that source, so the state a fit's center scored and the state carried on from the slice are the
same state (re-advanced on the accepted center's equilibrium with the center's parameters; the fit's other
solves -- Jacobian columns, trials -- also advance the state, on their perturbed equilibria, and none of those
is carried).  The source acts over its own interval ONLY: it is not carried into later intervals (no hold), so
every later interval starts with no correction source and only the response the source left in the carried
state, which then evolves under the model's own sources.  A prediction slice and a non-absorbed fit apply no
source.  The first slice (and every single-slice fit) has no preceding interval: its corrections are sources
of the steady state / periodic orbit (the steady treatment) and start the series inside its state; they are
not a source of the first interval (with ``CurrentModelSettings.corr_return="local"`` those steady-form
corrections are the zero-net local-return shapes; the in-window source always uses the plain Gaussians).  Every
slice records what its corrections did to the core (``correction_core_effect``: the first slice's fit record; an
absorbed filter fit's slice evaluated in the form it used; none on a prediction slice).
``filter_correction="steady_response"`` is the earlier treatment, kept as a comparison option: an absorbed
fit adds the corrections' zero-net-current steady response to the carried state and nothing is held.

``filter_correction="carried_outside_core"`` (NON-DEFAULT option): an absorbed fit changes the CARRIED state
outside a core radius ``r_c`` only.  Each plain Gaussian ``g_k`` is split by a continuous taper ``w`` (0 inside
``r_c``, rising as ``sin^2`` to 1 at ``r_c + carry_taper_width``; :func:`.current_model.carry_taper`): the core
part ``(1 - w) g_k`` is a source over the slice's interval exactly as in the in-window form (not carried), and
the outer part becomes ``h_k = N_k w [G(c_k, s) - lam_k G(c_k, r s)]``, a change of the state's current density
that carries zero net current, is zero inside ``r_c`` and returns its current in the correction's own flank
outside ``r_c`` (:func:`.current_model.carried_outside_core_parts`; ``r`` the existing
``CurrentModelSettings.corr_return_width_ratio``).  The fit's parameters are the same amplitudes with the same
prior; every trial state is the interval advanced with the core parts as a source plus ``sum_k a_k h_k``
(:func:`.current_model.carried_update`), so the state a fit's center scored and the state carried on are the same
object, as in the in-window form.  A prediction slice and a non-absorbed fit apply nothing; the first slice is
unchanged.  The core radius ``r_c`` (``carry_core_radius``, stated by the caller; no default): a number (a fixed
rho_tor,norm), or ``"floor_region"`` -- from the MODEL state, the outer edge of the q floor's region at the slice
(the region of the step's model sources at the slice time, from the slice's prediction before its fit:
``core["rho_floor"]``), times ``carry_core_radius_scale`` (1; other values are diagnostics).  Where the floor has
no region at a slice (its model q at or above the floor on axis), the region of the latest slice that had one is
held (the first slice's, from its own floor record); a series in which the floor never had a region (a discharge
with no q = 1 surface in the model, or the floor off) has ``r_c = 0``: the carried change then reaches in to the
taper next to the axis.  Every slice records the radius used and where it came from (``carry_core_radius``).

Records of the current corrections (every form; records only, nothing reads them): each fit's corrections in units
of their prior sigma (``correction_size``: per function and root-sum-square); every slice's ACCUMULATED absorbed
correction per function (``accumulated_correction``: the sum of the absorbed filter amplitudes since the first slice
in units of the prior sigma, the first slice's own corrections excluded, with a plain flag where a magnitude exceeds
:data:`ACCUMULATED_CORRECTION_REPORT_LEVEL` -- a REPORTING level for the reader, not an acceptance criterion; it
steers nothing).  Steady-state input check (``steady_state_check``, ON by default, a RECORD that steers nothing; it
costs a pass loop per slice): the slice's unfitted model steady state (the single-slice core treatment, no
corrections, the slice's own inputs) scored on the MSE chords by radius (:data:`MSE_RADIAL_CLASSES_M`,
:func:`.fit.steady_state_check`) next to the slice's own result; the solver is restored afterwards and the series
continues from the slice's own result.  The first slice's record is its fit's (a first-slice-fit start; without a
fit its own result is that steady state).  Slice solve counts exclude the check's solves, which its record states;
the series total includes them.

The vessel/eddy-current contribution through the previous slice's flux is not modeled
(``vessel_eddy=False``; ``True`` is refused).

Numerical defaults (documented; each a setting): the carried state lives on ``n_cells = 100``
uniform rho_tor,norm cells (:class:`.current_model.CurrentModelSettings`); the diffusion is advanced
with TR-BDF2 in steps of at most ``dt_max = 1 ms``; the model profiles are evaluated on a physics grid
of 201 points uniform in rho_tor,norm plus the outer-layer samples; the pass loop of each slice is
Anderson-accelerated with depth 3 and falls back to the plain input when a mixed input does not
solve; the diffusion and the equilibrium are coupled by Picard iteration within each step (holding
psi(rho) or q(rho) fixed while the geometry iterates was tried and did not converge); the psi_N <->
rho_tor,norm map is interpolated in rho^2 with a bisection inverse (PCHIP in rho was less accurate
near the axis).
"""

import copy
import dataclasses
import functools
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import current_model as _cm
from . import fit as _fit
from . import solver_rows as _rows
from .current import sawtooth as _saw
from .current.diffusion import CurrentDiffusion, CurrentState, GeometrySchedule, IpBoundary, OuterLayer
from .score import chi2_report, flux_offset_record, score

__all__ = ["SliceData", "SeriesSettings", "run_series", "INITIAL_MODES", "SAWTOOTH_MODES"]

INITIAL_MODES = ("auto", "mse_fit", "magnetics_fit", "relaxed", "from_profile")
#: The initial modes that start a series from a first-slice fit (``"magnetics_fit"``: MSE out of the fit groups).
FIT_START_MODES = ("mse_fit", "magnetics_fit")
#: The measurement groups whose presence lets ``initial="auto"`` choose ``"magnetics_fit"`` without usable MSE.
MAGNETIC_GROUPS = ("flux_loop", "bpol_probe")
#: Core treatments of a series (module docstring): ``"q_floor"`` (default), and the opt-ins ``"crash_times"``,
#: ``"periodic"`` and ``"off"``.
SAWTOOTH_MODES = ("q_floor", "crash_times", "off", "periodic")
#: The first slice's ``CurrentModelSettings.core_relaxation`` under each series mode (``"crash_times"``: replaced
#: by ``"crash_times"`` when the covered history holds measured crashes, :func:`_measured_crash_start`).
FIRST_SLICE_CORE = dict(q_floor="q_floor", crash_times="none", off="none", periodic="periodic_orbit")
#: What a series does when its first-slice fit (:data:`FIT_START_MODES`) did not end ``"converged"``.
INITIAL_FIT_POLICIES = ("refuse", "proceed_labeled")
#: The plasma-current convention of a series (module docstring, "Plasma current in a series"; recorded).
IP_CONVENTION = ("Ip(t) driving each interval: linear from the carried state's current to the next slice's measured "
                 "Ip x (1 + d), d = the relative offset (fitted - measured)/measured of the last fitted Ip, carried "
                 "forward (0 before any Ip fit); a prediction slice is solved to that same target; an absorbed filter "
                 "fit sets d to its own value and rescales the carried state once by (1 + d_new)/(1 + d_old)")
#: Fit statuses a filter absorbs and a series may start from (each carries its label into the records); stalled,
#: max_iter, failed-Jacobian, failed-trial and failed fits are not absorbed and do not start a series.
ACCEPTED_FIT_STATUSES = ("converged", "converged_within_noise")
#: How the lumped outer layer's ``|F|`` at the last closed surface varies over an interval.
LAYER_FPOL_MODES = ("linear", "interval_mean")
#: Which state an evolved slice scores when a sawtooth crash acted in its interval (module docstring):
#: ``"instantaneous"`` (default) -- the instantaneous end state, the state carried forward; ``"window_average"`` --
#: the earlier treatment (crash-free end state plus the interval's averaged sawtooth effect), comparison only.
CRASH_SLICE_STATES = ("instantaneous", "window_average")
#: How an absorbed filter fit's current corrections enter the evolution (module docstring, "Fit interaction"):
#: ``"in_window"`` (default) -- a non-inductive source over the interval that ends at the slice only (not carried
#: into later intervals); ``"steady_response"`` -- the earlier treatment (the zero-net-current
#: steady response added to the state), kept for comparison.
FILTER_CORRECTION_FORMS = ("in_window", "steady_response", "carried_outside_core")
#: What each form does (recorded in every series result).
FILTER_CORRECTION_RULES = {
    "in_window": ("current corrections are a non-inductive source over the interval that ends at the slice (the flux "
                  "diffusion of that interval gives their response); a filter fit's parameters are the source "
                  "amplitudes over that interval; the source is not carried into later intervals (no hold); the "
                  "first slice's corrections are in its starting state (steady form) and are not a source of the "
                  "first interval"),
    "steady_response": ("COMPARISON OPTION (the earlier treatment): an absorbed fit adds the corrections' zero-net-"
                        "current steady response to the carried state; nothing is held as a source"),
    "carried_outside_core": ("NON-DEFAULT OPTION: each correction is split at the core radius r_c by a continuous "
                             "taper; inside r_c it is a source over the slice's interval as in the in-window form (not "
                             "carried); outside r_c an absorbed fit changes the carried state by a zero-net current "
                             "change that is zero inside r_c (its return in the correction's own flank outside r_c); "
                             "the fit's trial states use the same form, so the scored state is the carried state")}
#: Definitions of the core radius of ``filter_correction="carried_outside_core"`` (``carry_core_radius``): a number (a
#: fixed rho_tor,norm) or ``"floor_region"`` (module docstring).
CARRY_CORE_RADIUS_RULES = ("floor_region",)
#: REPORTING level of the accumulated absorbed correction [prior sigmas of each function] (``accumulated_correction``):
#: a plain flag for the reader where the magnitude of a function's accumulated amplitude exceeds it.  Not an
#: acceptance criterion; nothing reads it and it steers nothing.
ACCUMULATED_CORRECTION_REPORT_LEVEL = 3.0
#: Radial classes of the MSE chords in the ``steady_state_check`` record (:data:`.fit.MSE_RADIAL_CLASSES_M`).
MSE_RADIAL_CLASSES_M = _fit.MSE_RADIAL_CLASSES_M


@dataclass
class SliceData:
    """Inputs of one time slice: ``time`` [s], measurement set (with ``ip``), reference boundary
    (anchor), kinetic inputs, sources and the vacuum ``F0`` AT THAT TIME."""
    time: float
    measurements: Any
    reference_boundary: np.ndarray
    kinetics: Any
    sources: Any
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    F0: Optional[float] = None       # vacuum R0 B_phi [T m] at this time (sign as the discharge); None: keep


@dataclass
class SeriesSettings:
    """Every setting of the series driver (recorded)."""
    initial: str = "auto"
    initial_profile: Optional[Any] = None                # (rho_tor_norm, <J.B> [A T/m^2]) for "from_profile"
    initial_profile_domain_fraction: Optional[float] = None   # scale it to carry f Ip inside the domain edge
    initial_profile_domain_fraction_tolerance: float = 0.0    # f may exceed 1 by up to this (excess recorded)
    initial_fit: Optional[_fit.FitSettings] = None       # None: free ("log_strength", "a_corr") with periodic
    #                                                      sawteeth, ("a_corr",) otherwise
    initial_fit_nonconverged: str = "refuse"             # INITIAL_FIT_POLICIES
    filter: bool = False
    filter_fit: _fit.FitSettings = field(default_factory=lambda: _fit.FitSettings(free=("a_corr",)))
    filter_times: Optional[Sequence[float]] = None       # with filter=True: fit only at these slice times [s]
    filter_time_tol: float = 1.0e-6                      # [s] match tolerance of filter_times
    filter_correction: str = "in_window"                 # FILTER_CORRECTION_FORMS ("steady_response": comparison)
    carry_core_radius: Optional[Any] = None              # "carried_outside_core" only: rho_tor,norm or "floor_region"
    carry_core_radius_scale: float = 1.0                 # "floor_region" only: multiplier (other than 1: diagnostic)
    carry_taper_width: float = 0.1                       # "carried_outside_core": taper width in rho_tor,norm
    steady_state_check: bool = True                      # record (steers nothing): unfitted steady state on MSE
    dt_max: float = 1.0e-3                               # diffusion step [s]
    scheme: str = "trbdf2"
    sawtooth: str = "q_floor"                            # SAWTOOTH_MODES; the others only as explicit options
    sawtooth_f: float = _fit.PRIOR_F
    sawtooth_log_strength: Optional[float] = None        # None: the prior mean of the fit module
    sawtooth_first_crash_phase: float = 1.0              # periodic train: first crash at t0 + phase/nu (0, 1]
    crash_times: Optional[Sequence[float]] = None
    crash_table: Optional[Any] = None                    # a crash_table.CrashTable (times and detection coverage)
    crash_record_start: Optional[float] = None           # [s] start of the crash detection (default: the table's)
    layer_fpol: str = "linear"                           # LAYER_FPOL_MODES
    crash_slice_state: str = "instantaneous"             # CRASH_SLICE_STATES ("window_average": comparison only)
    vessel_eddy: bool = False
    zeff_series_source: Optional[str] = None  # the series' ONE Z_eff source kind for the checked-source rule
    #                                            ("impurity_density" or "line_integrated"; see check_zeff_series)
    zeff_source_schedule: Optional[Sequence[Dict[str, Any]]] = None  # NON-DEFAULT: declared Z_eff source intervals
    #                                            (start, end, candidate, source); checked within each interval
    keep_going_after_failure: bool = False   # True: a failed slice is recorded and skipped; the next interval
    #                                          starts from the last consistent state (recorded per slice)
    # Moving anchor target in a series (only with FitSettings.anchor_follow on in the first-slice fit or the filter):
    anchor_carry_retest: bool = True         # a carried target is used on a slice only if its filter fit has a lower
    #                                          fitted chi^2 than the filter fit on the slice's own reference boundary
    anchor_follow_total: Optional[int] = None   # at most this many kept moves along one carried chain (None: no bound)

    def __post_init__(self):
        if self.initial not in INITIAL_MODES:
            raise ValueError(f"initial must be one of {INITIAL_MODES}")
        if self.sawtooth not in SAWTOOTH_MODES:
            raise ValueError(f"sawtooth must be one of {SAWTOOTH_MODES}")
        self.sawtooth_requested = self.sawtooth
        if (self.crash_table is not None or self.crash_times is not None) and self.sawtooth != "crash_times":
            raise ValueError(f"a crash table or crash times are given but sawtooth is {self.sawtooth!r}: measured "
                             "crash times are an explicit opt-in (sawtooth='crash_times')")
        if self.crash_table is not None:
            if self.crash_times is not None:
                raise ValueError("give crash_table or crash_times, not both")
            self.crash_times = tuple(float(t) for t in self.crash_table.times())
            if self.crash_record_start is None:
                self.crash_record_start = self.crash_table.record_start
        if self.sawtooth == "crash_times" and not self.crash_times:
            raise ValueError("sawtooth='crash_times' needs crash_times")
        if self.vessel_eddy:
            raise NotImplementedError("the vessel/eddy-current contribution is not modeled (vessel_eddy=False)")
        tol = float(self.initial_profile_domain_fraction_tolerance)
        if not 0.0 <= tol < 1.0:
            raise ValueError("initial_profile_domain_fraction_tolerance must lie in [0, 1)")
        if self.initial_profile_domain_fraction is not None and not (
                0.0 < float(self.initial_profile_domain_fraction) <= 1.0 + tol):
            raise ValueError("initial_profile_domain_fraction must lie in (0, 1 + "
                             "initial_profile_domain_fraction_tolerance] (default tolerance 0: (0, 1]); an enclosed "
                             "fraction above 1 means a negative current outside the domain edge")
        if self.initial_fit_nonconverged not in INITIAL_FIT_POLICIES:
            raise ValueError(f"initial_fit_nonconverged must be one of {INITIAL_FIT_POLICIES}")
        if not 0.0 < float(self.sawtooth_first_crash_phase) <= 1.0:
            raise ValueError("sawtooth_first_crash_phase must lie in (0, 1]")
        if self.layer_fpol not in LAYER_FPOL_MODES:
            raise ValueError(f"layer_fpol must be one of {LAYER_FPOL_MODES}")
        if self.crash_slice_state not in CRASH_SLICE_STATES:
            raise ValueError(f"crash_slice_state must be one of {CRASH_SLICE_STATES}")
        if self.filter_correction not in FILTER_CORRECTION_FORMS:
            raise ValueError(f"filter_correction must be one of {FILTER_CORRECTION_FORMS}")
        if self.filter_correction == "carried_outside_core":
            rc = self.carry_core_radius
            if rc is None:
                raise ValueError("filter_correction='carried_outside_core' needs carry_core_radius stated: a "
                                 f"rho_tor,norm in [0, 1) or one of {CARRY_CORE_RADIUS_RULES}")
            if isinstance(rc, str):
                if rc not in CARRY_CORE_RADIUS_RULES:
                    raise ValueError(f"carry_core_radius must be a number or one of {CARRY_CORE_RADIUS_RULES}")
            elif not 0.0 <= float(rc) < 1.0:
                raise ValueError("a fixed carry_core_radius must lie in [0, 1) (rho_tor,norm)")
            if not (np.isfinite(float(self.carry_core_radius_scale)) and float(self.carry_core_radius_scale) > 0.0):
                raise ValueError("carry_core_radius_scale must be finite and > 0")
            if float(self.carry_core_radius_scale) != 1.0 and rc != "floor_region":
                raise ValueError("carry_core_radius_scale applies to carry_core_radius='floor_region' only")
            if not (np.isfinite(float(self.carry_taper_width)) and float(self.carry_taper_width) > 0.0):
                raise ValueError("carry_taper_width must be finite and > 0")
            if self.sawtooth not in ("q_floor", "off"):
                raise ValueError("filter_correction='carried_outside_core' is built for sawtooth='q_floor' (default) "
                                 "and 'off' only")
        elif self.carry_core_radius is not None or float(self.carry_core_radius_scale) != 1.0:
            raise ValueError("carry_core_radius and carry_core_radius_scale are read only by "
                             "filter_correction='carried_outside_core'")
        if self.filter_times is not None:
            if not self.filter:
                raise ValueError("filter_times select the slices of the filter: they need filter=True")
            self.filter_times = tuple(float(t) for t in self.filter_times)
            if not self.filter_times:
                raise ValueError("filter_times is empty (give None to fit at every slice)")
        if self.initial_fit is None:
            free = ("log_strength", "a_corr") if self.sawtooth == "periodic" else ("a_corr",)
            self.initial_fit = _fit.FitSettings(free=free)
        if self.sawtooth != "periodic" and "log_strength" in self.initial_fit.free:
            raise ValueError("log_strength is a free parameter of the first-slice fit only with periodic sawteeth "
                             "(with measured crash times the rate is measured; with sawteeth off it has no effect)")

    def log_strength(self):
        return (_fit.PARAMETERS["log_strength"]["prior"][0] if self.sawtooth_log_strength is None
                else float(self.sawtooth_log_strength))

    def filter_at(self, t):
        """True when the filter fits the slice at time ``t`` [s] (every slice without ``filter_times``)."""
        if not self.filter:
            return False
        if self.filter_times is None:
            return True
        return any(abs(float(t) - ft) <= self.filter_time_tol for ft in self.filter_times)

    def as_dict(self):
        d = {k: v for k, v in asdict(self).items() if k not in ("initial_profile", "crash_table")}
        d["initial_profile"] = None if self.initial_profile is None else "supplied (rho_tor_norm, <J.B>)"
        d["crash_table"] = None if self.crash_table is None else self.crash_table.as_dict()
        d["sawtooth_requested"] = self.sawtooth_requested
        d["sawtooth_resolved"] = self.sawtooth
        d["sawtooth_rule"] = {
            "q_floor": ("time-averaged core: the q floor (CurrentModelSettings.q_floor, q_floor_form) on every "
                        "diffusion step -- resistive: the resistivity inside the outermost face where the step's "
                        "starting q is below the floor set to its value there -- the in-window corrections added "
                        "after the floor"),
            "crash_times": "EXPLICIT OPTION: operator at the measured crash times only (identity where q >= 1)",
            "off": "EXPLICIT OPTION: no core treatment (a warning is recorded on every slice whose evolved q falls "
                   "below 1)",
            "periodic": "EXPLICIT OPTION: prescribed periodic crash train (not the default)"}[self.sawtooth]
        d["first_slice_core_relaxation"] = FIRST_SLICE_CORE[self.sawtooth]
        if self.sawtooth not in ("periodic", "crash_times"):
            # read only by the crash operator of the opt-in modes: recorded as not used, not as its opt-in default
            d["sawtooth_f"] = None
            d["settings_not_used"] = ["sawtooth_f", "sawtooth_log_strength", "sawtooth_first_crash_phase"]
        d["sawtooth_log_strength_used"] = self.log_strength() if self.sawtooth == "periodic" else None
        return d


def _period(settings, log_strength=None):
    lam = float(np.exp(settings.log_strength() if log_strength is None else log_strength))
    return 1.0 / float(_saw.rate_from_strength(lam, settings.sawtooth_f))


def _crash_times(settings, t0, t_end, log_strength=None):
    if settings.sawtooth in ("off", "q_floor"):
        return np.zeros(0)
    if settings.sawtooth == "crash_times":
        return np.asarray(settings.crash_times, float)
    tau = _period(settings, log_strength)
    n = int(np.ceil((t_end - t0) / tau)) + 2
    return t0 + tau * (float(settings.sawtooth_first_crash_phase) + np.arange(0, n - 1))


def _layer_integrals(lag, sigma_ge, jni_ge):
    _, sl = _cm._domain_arrays(lag, sigma_ge, log=True)
    _, jl = _cm._domain_arrays(lag, jni_ge)
    L = lag["layer"]
    return float(L.integral(L.conductance_density(sl))), float(L.integral(L.source_density(jl))), float(L.fpol_lcfs)


def _jni(lag, kin, src, params):
    ph, ge = lag["phys"], lag["phys"]["psi_n"]
    p = _cm.full_params(params)
    jcd = p["s_cd"] * src.jdotb_at(ge, b2=ph["b2"], fpol=ph["F"], inv_r2=ph["inv_r2"], rho=ph["rho"])
    return p["s_bs"] * lag["jbs"] + jcd


def interval_solver(lag0, lag1, jni0, jni1, t0, t1, ip0, ip1, layer_fpol="linear"):
    """Time-dependent :class:`CurrentDiffusion` on ``[t0, t1]`` between two slices' lagged inputs
    (module docstring).  ``layer_fpol``: the outer layer's ``|F|`` at the last closed surface linear in
    time between the two slices' values (``"linear"``, default; consistent with the domain's ``F`` at
    both ends) or held at their mean (``"interval_mean"``, the earlier treatment)."""
    g0, g1 = lag0["geom"], lag1["geom"]
    if not np.array_equal(g0.rho, g1.rho):
        g1 = g1.resample(g0.rho)
    geos = [copy.copy(g0), copy.copy(g1)]
    for g in geos:
        g.dphib_dt = 0.0
    sched = GeometrySchedule([t0, t1], geos)
    rho = g0.rho
    s0, _ = _cm._domain_arrays(lag0, lag0["sigma"], log=True)
    s1, _ = _cm._domain_arrays(lag1, lag1["sigma"], log=True)
    j0, _ = _cm._domain_arrays(lag0, jni0)
    j1, _ = _cm._domain_arrays(lag1, jni1)
    a0, b0, f0 = _layer_integrals(lag0, lag0["sigma"], jni0)
    a1, b1, f1 = _layer_integrals(lag1, lag1["sigma"], jni1)
    c0, c1 = float(lag0["psi_n_edge"]), float(lag1["psi_n_edge"])
    if abs(c0 - c1) > 1e-12:
        raise ValueError(f"domain edge changed between slices ({c0} -> {c1}): not supported")
    if layer_fpol == "linear":
        flay = ((t0, t1), (f0, f1))
    elif layer_fpol == "interval_mean":
        flay = 0.5 * (f0 + f1)
    else:
        raise ValueError(f"layer_fpol must be one of {LAYER_FPOL_MODES}")
    layer = OuterLayer(c1, flay, ((t0, t1), (a0, a1)), ((t0, t1), (b0, b1)))
    sol = CurrentDiffusion(sched, ((t0, t1), rho, np.vstack((s0, s1))), IpBoundary(((t0, t1), (ip0, ip1))),
                           j_ni=((t0, t1), rho, np.vstack((j0, j1))), outer_layer=layer)
    return sol, dict(fpol_lcfs_used=(0.5 * (f0 + f1) if layer_fpol == "interval_mean" else [f0, f1]),
                     layer_fpol=layer_fpol, fpol_lcfs_change=f1 - f0, dphib_dt=(g1.phi_b - g0.phi_b) / (t1 - t0),
                     layer_A=(a0, a1), layer_B=(b0, b1))


def advance_interval(sol, state, t1, settings, crash_times, dt):
    """Advance ``state`` to ``t1`` applying the crash operator at ``crash_times`` (identity where
    q >= 1).  Returns ``(end_state, events, trajectory_states)``."""
    traj = _saw.Trajectory(state.time, t1)
    ct = np.asarray(crash_times, float)
    ct = ct[(ct > state.time) & (ct <= t1)]
    sch = _saw.CrashSchedule(ct, settings.sawtooth_f if settings.sawtooth != "off" else 1.0, trajectory=traj)
    end = _saw.advance_with_crashes(sol, state, t1, sch, dt, scheme=settings.scheme)
    return end, sch.events, traj.states


def advance_q_floor(sol, state, t1, settings, dt, q_floor, sol_all=None, form="resistive", total_current="off",
                    transition="step"):
    """Advance ``state`` to ``t1`` with the q floor (the default core treatment, module docstring); ``sol`` carries the
    MODEL sources only.

    ``form="resistive"`` (default): every step uses the resistivity of :class:`.current.sawtooth.ResistiveFloor` --
    inside the single-slice flattening region of ``sol``'s sources at the step's start time
    (:meth:`.current.sawtooth.ResistiveFloor.region_info`) the resistivity set to its value at the region's edge --
    so the current stays continuous and Ip is the interval's Ip condition; q is not held at the floor but relaxed
    toward the single-slice floored state of the same sources.  ``form="sharp"`` (opt-in): the earlier clamp
    :func:`.current.sawtooth.floor_q` after every step.

    ``sol_all`` (optional): the interval's solver with the current corrections added to the sources; their response
    over the interval (``sol_all`` minus ``sol``, both advanced from ``state`` in the same steps with the plain
    resistivity) is added to the model end state (the corrections after the floor).  Where the floor does not act the
    end state is the advance with the corrections among the sources, to round-off.  From the next interval on that
    end state is the model state, so where a correction left q below the floor there, the floor acts on it in the
    next interval (resistive: the flattened resistivity relaxes it over the resistive time; sharp: the first step
    clamps it away).

    The region is decided once per step, so the series state depends on the step ``dt`` (``SeriesSettings.dt_max``)
    at first order.

    Returns ``(end_state, core_record)``: the form, ``q_floor``, the number of steps and of steps on which the floor
    acted, and at the end of the interval the region at the end time (``rho_floor`` and ``rho_floor_inner``, the
    domain's rho; None where the floor does not act; ``floor_branch``; resistive also ``state_branch``, the branch the
    end state's own q would give, a record) with every crossing of the floor by the model end
    state's q (``rho_floor_crossings``, inner and outer); resistive also the outer edge used on the last step and the
    largest over the interval, sharp the last step's floor record and the largest change of the enclosed current
    made by the floor (fraction of Ip).  RECORD only, both forms: the model end state's axis q and minimum q against
    the same interval advanced from ``state`` without the floor (plain resistivity, model sources), and the change of
    the boundary loop voltage (relative), of ``li3`` (NaN in a series: the interval solver's outer layer carries no
    reference surfaces) and of the domain's ``li3_domain`` (always available), and the fraction of Ip moved from
    inside the end state's outer
    edge (resistive: the region at the end time; sharp: the outermost upward crossing of the end state's q) to outside
    it (0 where there is no region), made in a guard (a failure is recorded with ``floor_effect_computed`` False and
    never ends the series); resistive also the floor's branch on every step (``branch_steps``, count per branch) and
    the smallest distance of the plain steady state's axis q from the floor over the steps.

    Non-default options of the resistive form (:mod:`.current.sawtooth`, "Non-default options"; off by default):
    ``total_current="flat_at_level"`` -- every step also changes the sources inside the step's region so that the
    single-slice state of the step's sources has q flat at the level there (the diffusion relaxes toward it);
    ``transition="blend_margin"`` -- the step's region is that of the blended transition."""
    if form not in _saw.Q_FLOOR_FORMS:
        raise ValueError(f"form must be one of {_saw.Q_FLOOR_FORMS}")
    acc = dict(n_steps=0, n_acted=0, d_i_max=0.0, last=None, regions=[])
    sol_f = sol
    if form == "sharp":
        def hook(st, t, solver):
            st2, rec = _saw.floor_q(st, solver, q_floor)
            acc["n_steps"] += 1
            if rec["acted"]:
                acc["n_acted"] += 1
                acc["d_i_max"] = max(acc["d_i_max"], float(rec["d_enclosed_current_max_over_ip"]))
            acc["last"] = rec
            return st2

        end = sol.advance(state, t1, dt=dt, scheme=settings.scheme, post_step=hook)
    else:
        mod = (_saw.ResistiveFloor(sol, q_floor) if total_current == "off" and transition == "step" else
               _saw.ResistiveFloor(sol, q_floor, total_current=total_current, transition=transition))
        sol_f = mod.with_solver()
        acc["regions"].append(mod.region_info(state))

        def hook(st, t, solver):
            # the region of each step's END state is the one the next step uses (the last one: the end of the interval)
            acc["n_steps"] += 1
            acc["regions"].append(mod.region_info(st))
            return st

        end = sol_f.advance(state, t1, dt=dt, scheme=settings.scheme, post_step=hook)
        used = acc["regions"][:-1]
        acc["n_acted"] = sum(1 for r in used if r["k"] > 0)
    q_end = _saw.face_q(end, sol.geometry_at(end.time).phi_b)
    model_end_crossings = _saw.floor_crossings(end.rho_faces, q_end, q_floor)     # the model end state's q
    # the same interval advanced without the floor: the corrections' response needs it when they are given; otherwise
    # it is made for the record only, inside the record's guard (a record must never end a series)
    e_mod = sol.advance(state, t1, dt=dt, scheme=settings.scheme) if sol_all is not None else None
    try:
        effect = _interval_floor_effect(sol, sol_f, e_mod, end, q_end, q_floor,
                                        r_out=acc["regions"][-1]["rho_inv"] if form != "sharp" else "crossing",
                                        advance_plain=lambda: sol.advance(state, t1, dt=dt, scheme=settings.scheme))
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series (the guard is also inside)
        effect = dict(floor_effect_computed=False, floor_effect_reason=f"{type(exc).__name__}: {exc}"[:300])
    if sol_all is not None:
        e_all = sol_all.advance(state, t1, dt=dt, scheme=settings.scheme)
        end = CurrentState(end.rho_faces, end.psi + (e_all.psi - e_mod.psi),
                           end.psi_boundary + (e_all.psi_boundary - e_mod.psi_boundary), end.time, dict(end.meta))
    common = dict(core_relaxation="q_floor", q_floor_form=form, q_floor=float(q_floor), n_steps=acc["n_steps"],
                  n_steps_floor_acted=acc["n_acted"], acted_in_interval=bool(acc["n_acted"]), dt=float(dt),
                  corrections="in-window response added after the floor" if sol_all is not None else None)
    if form == "sharp":
        core = dict(acc["last"] or {}, **common, d_enclosed_current_max_over_ip_interval=acc["d_i_max"])
    else:
        radii = [r["rho_inv"] for r in acc["regions"][:-1] if r["rho_inv"] is not None]
        r_end = acc["regions"][-1]
        if total_current != "off" or transition != "step":
            common.update(total_current=total_current, floor_transition=transition)
        core = dict(common, acted=bool(acc["n_acted"]), rho_floor=r_end["rho_inv"],
                    rho_floor_inner=r_end["rho_inner"], floor_branch=r_end["branch"], floor_rule=mod.rule,
                    state_branch=r_end.get("state_branch"),
                    n_cells_flattened=int(r_end["k"]), rho_floor_crossings=model_end_crossings,
                    rho_floor_last_step=acc["regions"][-2]["rho_inv"] if len(acc["regions"]) > 1 else None,
                    rho_floor_max_interval=max(radii) if radii else None,
                    q_min_model_end=float(np.nanmin(q_end)))
        try:                  # the branch of every step: a record only (never ends a series)
            core.update(_step_branch_record(acc["regions"], q_floor))
        except Exception as exc:  # noqa: BLE001 -- a record must never end a series
            core.update(branch_steps=None, branch_steps_reason=f"{type(exc).__name__}: {exc}"[:300])
    core.update(effect)
    return end, core


def _step_branch_record(regions, q_floor):
    """RECORD of the floor's branch on the diffusion steps of one interval (the regions the steps used, and the one at
    the end time): the count per branch (``branch_steps``) and the smallest distance of the plain steady state's axis
    q from the floor (``q_axis_model_steady_min_abs_dev_floor``).  Never raises; nothing reads it."""
    try:
        steps = {}
        for r in regions[:-1]:
            steps[str(r.get("branch"))] = steps.get(str(r.get("branch")), 0) + 1
        devs = [abs(float(r["q_axis_model_steady"]) - float(q_floor)) for r in regions
                if r.get("q_axis_model_steady") is not None and np.isfinite(r["q_axis_model_steady"])]
        return dict(branch_steps=steps, q_axis_model_steady_min_abs_dev_floor=min(devs) if devs else None)
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        return dict(branch_steps=None, branch_steps_reason=f"{type(exc).__name__}: {exc}"[:300])


#: The keys of the interval's floor-effect record (:func:`_interval_floor_effect`): only these are taken into the
#: interval's core record, so a record can never overwrite a key of the floor's region or branch.
INTERVAL_FLOOR_EFFECT_KEYS = ("q_axis_model_end", "q_axis_no_floor_end", "q_min_no_floor_end", "v_loop_plain",
                              "v_loop_floor", "d_v_loop_rel", "li3_plain", "li3_floor", "d_li3", "li3_domain_plain",
                              "li3_domain_floor", "d_li3_domain", "current_moved_outside_over_ip",
                              "floor_effect_reference")


#: The reference of the interval's floor-effect record (also where that record failed).
INTERVAL_FLOOR_EFFECT_REFERENCE = ("the same interval advanced from the same carried state without the floor "
                                   "(model sources, before the corrections)")


def _interval_floor_effect(sol, sol_f, plain_end, end, q_end, q_floor, r_out="crossing", advance_plain=None):
    """RECORD of the floor's effect over one interval: the model end state (``end``, with the floor; ``sol_f`` the
    solver that made it) against the same interval advanced without the floor (``plain_end``, ``sol``; when it is
    None it is made here by ``advance_plain()``, inside this record's guard); the current moved is counted across
    ``r_out`` (the region's edge, None for no region; ``"crossing"``: the outermost upward crossing of the end state's
    q).  Never raises (a record must not end a series): its own keys only (:data:`INTERVAL_FLOOR_EFFECT_KEYS`) with
    ``floor_effect_computed=True``, or, if anything in it fails, NaN values with ``floor_effect_computed=False`` and
    the reason (``floor_effect_reason``)."""
    try:
        if plain_end is None:
            plain_end = advance_plain()
        out = _interval_floor_effect_values(sol, sol_f, plain_end, end, q_end, q_floor, r_out)
        out = {k: out[k] for k in INTERVAL_FLOOR_EFFECT_KEYS}
        out["floor_effect_computed"] = True
        return out
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        out = dict.fromkeys(INTERVAL_FLOOR_EFFECT_KEYS, float("nan"))
        out.update(floor_effect_reference=INTERVAL_FLOOR_EFFECT_REFERENCE, floor_effect_computed=False,
                   floor_effect_reason=f"{type(exc).__name__}: {exc}"[:300])
        return out


def _interval_floor_effect_values(sol, sol_f, plain_end, end, q_end, q_floor, r_out):
    """The values of :func:`_interval_floor_effect` (unguarded)."""
    p0, p1 = sol.profiles(plain_end), sol_f.profiles(end)
    q0 = np.asarray(p0["q"], float)
    if isinstance(r_out, str):
        r_out = _saw.floor_region(end.rho_faces, q_end, q_floor)["rho_outer"]
    ip = float(p0["ip"])
    moved = 0.0
    if r_out is not None and ip:
        rf = np.asarray(end.rho_faces, float)
        moved = float((np.interp(r_out, rf, p0["I_enclosed"]) - np.interp(r_out, rf, p1["I_enclosed"])) / ip)
    v0, v1 = float(p0["v_loop_boundary"]), float(p1["v_loop_boundary"])
    return dict(q_axis_model_end=float(q_end[0]), q_axis_no_floor_end=float(q0[0]),
                q_min_no_floor_end=float(np.nanmin(q0)),
                v_loop_plain=v0, v_loop_floor=v1, d_v_loop_rel=(v1 / v0 - 1.0) if v0 else float("nan"),
                li3_plain=float(p0["li3"]), li3_floor=float(p1["li3"]), d_li3=float(p1["li3"] - p0["li3"]),
                li3_domain_plain=float(p0["li3_domain"]), li3_domain_floor=float(p1["li3_domain"]),
                d_li3_domain=float(p1["li3_domain"] - p0["li3_domain"]),
                current_moved_outside_over_ip=moved,
                floor_effect_reference=INTERVAL_FLOOR_EFFECT_REFERENCE)


def _branch_log_open():
    """Open a branch log for one slice (:func:`.current_model.open_branch_log`); ``(None, None)`` if that fails (a
    record must never end a series)."""
    try:
        return _cm.open_branch_log()
    except Exception:  # noqa: BLE001 -- a record must never end a series
        return None, None


def _branch_log_close(token):
    try:
        if token is not None:
            _cm.close_branch_log(token)
    except Exception:  # noqa: BLE001 -- a record must never end a series
        pass


def _branch_record(log, res, nested=None):
    """:func:`.current_model.branch_flip_record` of one slice's assemblies (a record only; never raises).  ``nested``:
    the ``floor_branch_record`` of a fit run inside the slice (its filter fit), whose Jacobians are the slice's: their
    count, the mixed count and the mixed list are taken from it (``jacobians_from``); without one the slice computed
    no Jacobian (0)."""
    try:
        if log is None:
            return dict(computed=False, reason="no branch log")
        out = _cm.branch_flip_record(log, None, ((res.assembly or {}).get("core") if res is not None else None))
        if isinstance(nested, dict) and nested.get("n_jacobians") is not None:
            out.update(n_jacobians=int(nested["n_jacobians"]),
                       n_jacobians_mixed_branches=int(nested.get("n_jacobians_mixed_branches") or 0),
                       jacobians_mixed_branches=list(nested.get("jacobians_mixed_branches") or []),
                       jacobians_mixed_branches_total=int(nested.get("jacobians_mixed_branches_total") or 0),
                       jacobians_mixed_branches_truncated=bool(nested.get("jacobians_mixed_branches_truncated")),
                       jacobians_from="the slice's filter fit (its floor_branch_record)")
        else:
            out["jacobians_from"] = "no fit in this slice: no Jacobian computed"
        return out
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _window_state(history, t_a, t_b, rf):
    """Trapezoid time average of psi over [t_a, t_b] from stored states (crash jumps stored as two states
    at one time)."""
    ts = np.array([s.time for s in history])
    use = [i for i, s in enumerate(history) if t_a - 1e-12 <= s.time <= t_b + 1e-12]
    before = [i for i in range(len(history)) if ts[i] < t_a]
    idx = ([before[-1]] if before else []) + use
    ts, P = ts[idx], np.array([history[i].psi for i in idx])
    pb = np.array([history[i].psi_boundary for i in idx])
    if ts[0] > t_a + 1e-12:
        raise ValueError("window starts before the recorded history")
    g = np.column_stack((P, pb))
    avg = _saw._pl_integral(ts, g, t_a, t_b) / (t_b - t_a)
    return CurrentState(rf, avg[:-1], float(avg[-1]), t_b, dict(window_average=[t_a, t_b]))


def _slice_ip(sl):
    if sl.measurements is None:
        return abs(float(sl.meta["ip"]))
    return abs(float(sl.measurements.groups["ip"].measured[0]))


def _anchor_offset(fm):
    """The moved anchor target's displacement from the slice's reference boundary
    (:meth:`.forward.ForwardModel.anchor_offset`), or ``None`` when nothing was moved (always, unless
    ``FitSettings.anchor_follow`` is on in the first-slice fit or the filter fit).  Never raises."""
    try:
        get = getattr(fm, "anchor_offset", None)
        return None if get is None else get()
    except Exception:  # noqa: BLE001 -- carrying nothing is the default behavior
        return None


def _carry_anchor(fm, offset):
    """Moving anchor target in a series: the previous slice's displacement of the anchor target from its reference
    boundary, as a function of the poloidal angle about the magnetic axis, is applied to this slice's reference
    boundary before its solves (:meth:`.forward.ForwardModel.apply_anchor_offset`).  ``None`` when nothing was moved
    (the default: the target is the slice's reference boundary).  Returns the record for the slice."""
    if offset is None:
        return None
    try:
        return fm.apply_anchor_offset(offset)
    except Exception as exc:  # noqa: BLE001 -- the slice then runs on its reference boundary, and the record says so
        return dict(carried=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _slice_record(fm, res, sl, label, extra=None):
    """Distilled record of one solved slice (no measured values)."""
    ms = sl.measurements
    tab = rep = None
    summ = None
    if ms is not None and res.converged:
        groups = [g for g in ("flux_loop", "bpol_probe", "ip", "pf_current", "mse") if g in ms.groups]
        tab = score(fm.gs, ms, groups=groups)
        # headline scores without the coils the source left free (reported apart), and the degrees of freedom with
        # the coil solve's effective parameter count (forward.ForwardModel.inner_dof) on these channels
        roles = {n: v.get("role") for n, v in (getattr(fm, "rows_record", {}) or {}).get("coils", {})
                 .get("per_coil", {}).items()}
        summ = _fit.headline_summary(tab, ms, roles)
        excl = {}
        if "pf_current" in tab.model:
            grp = ms.groups["pf_current"]
            names = [str(n) for n in grp.geometry.get("coil_names", grp.labels)]
            excl["pf_current"] = np.array([roles.get(n, "free") == "free" for n in names], dtype=bool)
        inner = fm.inner_dof_data(groups) if hasattr(fm, "inner_dof_data") else 0.0
        # the parameters fitted to these channels: the free parameters of the fit whose center the slice scores (the
        # first-slice fit, or an absorbed filter fit), none on an evolved, unfitted slice
        filt, fit0 = (extra or {}).get("filter") or {}, (extra or {}).get("fit") or {}
        n_fit = (len(filt.get("free") or ()) if filt.get("absorbed") else len(fit0.get("free") or ()))
        rep = chi2_report(tab, n_params=n_fit, inner_dof=inner, exclude=excl)
    asm = res.assembly
    st = res.solve.stats if res.solve is not None else {}
    lag = res.lag
    out = dict(label=label, time=sl.time, status=res.status, passes=res.passes, wall_s=res.wall_s,
               chi2=summ,
               chi2_report=rep, stats=st, q_check=res.record.get("q_check"), reason=res.record.get("reason"),
               pass_trace=[{k: p.get(k) for k in ("k", "dj", "q_max_rel", "solve_status", "its", "anderson_mixed",
                                                  "anderson_fallback", "anderson_restart", "d_axis_m")}
                           for p in res.record.get("passes", [])],
               q_axis_traced=None if lag is None else lag["phys"].get("q_axis"),
               q0_axis=((res.record.get("q_check") or {}).get("axis") or {}).get("q0_eq"),
               fractions=asm["fractions"] if asm else None, currents_A=asm["currents_A"] if asm else None,
               core=asm.get("core") if asm else None,
               v_loop=asm["v_loop"] if asm else None, sawtooth=asm["sawtooth"] if asm else None,
               q_model=dict(rho=np.asarray(asm["q_model"]["rho"]).tolist(), q=np.asarray(asm["q_model"]["q"]).tolist())
               if asm else None,
               profiles_rho={k: np.asarray(v).tolist() for k, v in asm["profiles_rho"].items()} if asm else None,
               boundary=fm.boundary_distance() if res.converged else None)
    out["ip_fit"] = _fit.ip_record(tab, ms, (asm or {}).get("params") or {}, bool((extra or {}).get("ip_fitted")))
    out["flux_offset"] = None if tab is None else flux_offset_record(tab)
    out["coil_residuals"] = _fit.coil_residual_record(tab, ms, {
        n: v.get("role") for n, v in (getattr(fm, "rows_record", {}) or {}).get("coils", {}).get("per_coil", {}).items()})
    st_meta = dict(getattr((asm or {}).get("state"), "meta", None) or {})
    if "profile_scale" in st_meta:
        out["initial_profile_scale"] = dict(profile_scale=st_meta["profile_scale"],
                                            domain_fraction=st_meta.get("domain_fraction"),
                                            domain_fraction_excess=st_meta.get("domain_fraction_excess"))
    if lag is not None and res.converged:
        out["state_psi"] = dict(rho_faces=asm["state"].rho_faces.tolist(), psi=asm["state"].psi.tolist(),
                                psi_boundary=asm["state"].psi_boundary)
    if lag is not None:
        out["q_eq"] = dict(rho=np.asarray(lag["phys"]["rho"]).tolist(), q=np.asarray(lag["phys"]["q"]).tolist())
    out["ip_convention"] = "carried fitted offset (series.IP_CONVENTION)"
    out.update(extra or {})
    try:      # a record only: nothing reads it
        out["core_vs_floor"] = _cm.core_vs_floor_record(out.get("core"), out.get("correction_core_effect"))
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        out["core_vs_floor"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
    try:      # a record only: nothing reads it
        out["floor_effect"] = _cm.floor_effect_record(out.get("core"), out.get("q_check"),
                                                      None if lag is None else lag["phys"].get("q"),
                                                      out.get("q_axis_traced"), (st or {}).get("q_0"))
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        out["floor_effect"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
    return out


@_rows.keep_constraints()        # the solver's constraint set kept between score calls (solver_rows; no value changes)
def run_series(fm, slices: List[SliceData], cfg, settings=None, start_psi=None, initial_profiles=None,
               callback=None, speed=None):
    """Evolve the current through ``slices`` (module docstring).

    ``fm``: :class:`.forward.ForwardModel` configured for ``slices[0]`` (its rows are rebuilt for
    every slice with :meth:`~.forward.ForwardModel.reconfigure`; the edge-taper case is kept, so the
    taper stays uniform within the series); ``cfg``: :class:`.current_model.CurrentModelSettings`
    (``n_cells`` sets the carried state's grid); ``start_psi``/``initial_profiles``: first-slice
    starting state of the solver (as :func:`.fit.reconstruct`).  ``callback(record)`` is called after
    every slice.  Returns a dict with ``slices`` (one record each), ``settings``, ``solver`` (the solver build,
    the cut-cell setting in effect and the edge-taper setting, :meth:`.forward.ForwardModel.solver_record`) and
    the totals.  ``speed``: :class:`.speed.SpeedSettings` (parallel Jacobian columns of the fits, opt-in; the memo
    of pure evaluations, the indexed point location and reused statistics, on by default -- not given:
    ``SpeedSettings()``; ``SpeedSettings.all_off()`` turns every option off; changes no result; recorded as ``speed`` when
    given without workers -- with workers, every Jacobian's ``parallel`` record holds it).  The options in effect
    are recorded as ``speed_in_effect`` in every series record, also when ``speed`` is not given
    (:func:`.speed.in_effect`).
    """
    from . import speed as _speed
    eff = _speed.effective(speed)                     # the default SpeedSettings() when none is given
    _speed.validate(eff, fm)
    with _speed.speed_scope(eff):                     # each in-process option set as in eff, also when off
        effect = _speed.in_effect(speed)
        out = _run_series(fm, slices, cfg, settings=settings, start_psi=start_psi, initial_profiles=initial_profiles,
                          callback=callback, speed=speed)
    if isinstance(out, dict):
        out["speed_in_effect"] = effect                    # the options in effect, in every series record
    if speed is not None and not speed.parallel and isinstance(out, dict):
        out["speed"] = dict(settings=speed.as_dict())      # the settings as given
    return out


def _speed_kw(speed):
    """``speed`` as a keyword only when it is given (the calls without it are the calls as before)."""
    return {} if speed is None else dict(speed=speed)


def _run_series(fm, slices, cfg, settings=None, start_psi=None, initial_profiles=None, callback=None, speed=None):
    """:func:`run_series` inside its speed scope."""
    s = settings or SeriesSettings()
    t_all = time.perf_counter()
    n_solves0 = fm.n_solves
    its0 = len(getattr(fm, "iteration_log", None) or [])
    times = np.array([float(x.time) for x in slices])
    if np.any(np.diff(times) <= 0):
        raise ValueError("slice times must increase strictly")
    out = dict(settings=s.as_dict(), model_settings=cfg.as_dict(), slices=[], times=times.tolist(),
               ip_convention=IP_CONVENTION, solver=fm.solver_record())
    out["zeff_series"] = check_zeff_series(slices, s.zeff_series_source, s.zeff_source_schedule)
    zeff_flags = set(out["zeff_series"].get("failing_slices") or []) | set(out["zeff_series"].get("unverified_slices")
                                                                         or [])
    sl0 = slices[0]
    mode = s.initial
    if mode == "auto":
        groups0 = {} if sl0.measurements is None else sl0.measurements.groups
        mse = groups0.get("mse")
        if mse is not None and bool(mse.meta.get("er_corrected", False)):
            mode = "mse_fit"
        elif any(g in groups0 for g in MAGNETIC_GROUPS):
            mode = "magnetics_fit"
        else:
            mode = "relaxed"
        out["initial_auto_choice"] = mode
    case = fm.case
    ls_used = s.log_strength() if s.sawtooth == "periodic" else None
    ls_source = "series setting (sawtooth_log_strength, or the fit module's prior mean)" if ls_used is not None else None
    p_relaxed = dict(log_strength=ls_used, f=s.sawtooth_f) if s.sawtooth == "periodic" else dict(log_strength=None)
    # the first slice's core treatment follows the series' mode (recorded; FIRST_SLICE_CORE)
    cfg0 = dataclasses.replace(cfg, core_relaxation=FIRST_SLICE_CORE[s.sawtooth])
    out["first_slice_core_relaxation"] = cfg0.core_relaxation
    if s.sawtooth == "crash_times":
        cfg0, p_relaxed, out["first_slice_crash_start"] = _measured_crash_start(s, cfg0, times)
        out["first_slice_core_relaxation"] = cfg0.core_relaxation
    # ---------------------------------------------------------------- slice 0
    t0 = time.perf_counter()
    n0 = fm.n_solves
    fitrec = None
    f0rec0 = _apply_f0(fm, sl0)
    if start_psi is not None:
        fm.set_psi(start_psi)
    anchor_moves0 = 0
    if mode in FIT_START_MODES:
        fit0 = s.initial_fit
        if mode == "magnetics_fit":
            fit0 = dataclasses.replace(fit0, fit_groups=tuple(g for g in fit0.fit_groups if g != "mse"))
            out["initial_fit_groups"] = list(fit0.fit_groups)
        fr = _fit.reconstruct(fm, sl0.kinetics, sl0.sources, cfg0, fit0, params0=p_relaxed,
                              initial_profiles=initial_profiles, label=f"{sl0.label}:{mode}",
                              steady_state_check=s.steady_state_check, **_speed_kw(speed))
        ssc0 = fr.record.get("steady_state_check")
        anchor_moves0 = int((fr.record.get("anchor_follow") or {}).get("n_moves_kept") or 0)
        if fr.record.get("center") is None:
            c0 = fr.record.get("center0") or {}
            its = fr.record.get("iterations", [])
            out["initial_fit_failure"] = dict(
                status=fr.status, center0_status=c0.get("status"), center0_reason=c0.get("reason"),
                n_iterations=len(its), last_jacobian=(its[-1].get("jacobian") if its else None),
                last_trials=[{k: t.get(k) for k in ("lam", "status", "chi2")} for t in (its[-1].get("trials", []) if its else [])])
            out["slices"].append(dict(label=sl0.label, time=sl0.time, status=f"initial_{fr.status}",
                                      fit_status=fr.status, n_solves=fm.n_solves - n0))
            out.update(status="failed_initial", n_solves=fm.n_solves - n_solves0,
                       wall_s=round(time.perf_counter() - t_all, 2))
            _note_iterations(out, fm, its0)
            return out
        res0 = fr.record["center"]["res"]
        fm.restore(fr.record["center"]["snap"])
        fm.probe_correction = fr.record["center"]["probe_correction"].copy()
        start_note = None
        if fr.status == "converged_within_noise":
            start_note = ("the first-slice fit ended 'converged_within_noise' (stopped at the optimum to within the "
                          "measured noise); the series starts from its center with that label")
        elif fr.status not in ACCEPTED_FIT_STATUSES:
            # the same standard as the filter: a series does not start from a fit that is not converged
            # unless the caller asked for it explicitly, and then the start is labeled
            start_note = (f"the first-slice fit ended '{fr.status}', not converged; the series starts from its "
                          f"center by explicit option (initial_fit_nonconverged='proceed_labeled')")
            if s.initial_fit_nonconverged == "refuse":
                msg = (f"first-slice fit ended '{fr.status}', not converged: the series is not started from it "
                       "(set initial_fit_nonconverged='proceed_labeled' to start from its center, labeled)")
                out["slices"].append(dict(label=sl0.label, time=sl0.time, status="initial_fit_not_converged",
                                          fit_status=fr.status, stopping_summary=fr.record.get("stopping_summary"),
                                          n_solves=fm.n_solves - n0, reason=msg))
                out.update(status="refused_initial_fit_not_converged", reason=msg, n_solves=fm.n_solves - n_solves0,
                           wall_s=round(time.perf_counter() - t_all, 2))
                _note_iterations(out, fm, its0)
                return out
            out["initial_fit_nonconverged_start"] = dict(status=fr.status, policy=s.initial_fit_nonconverged,
                                                         note=start_note)
        if "log_strength" in fr.free:
            ls_used = float(fr.params["log_strength"])
            ls_source = f"fitted at the first slice (status {fr.status})"
        fitrec = dict(status=fr.status, start_label=start_note, free=list(fr.free), theta=fr.theta.tolist(), sigma=fr.sigma.tolist(),
                      prior={n: fr.record["specs"].get(n, {}).get("prior") for n in fr.free}, chi2=fr.chi2,
                      n_solves=fr.record["n_solves"], wall_s=fr.record["wall_s"],
                      iterations=len(fr.record["iterations"]), stopping=fr.record.get("final_stopping_tests"),
                      stopping_summary=fr.record.get("stopping_summary"),
                      noise=fr.record.get("noise"), column_flags=fr.record.get("covariance_column_flags"),
                      ip_fit=fr.record.get("ip_fit"),
                      correction_size=_correction_size(fr.params, cfg0, fit0))
        cce0 = fr.record.get("correction_core_effect")
        branch0 = fr.record.get("floor_branch_record")
        params0 = fr.params
    else:
        state_fn0 = None
        if mode == "from_profile":
            def state_fn0(lag):
                return _given_state(lag, s, cfg, _slice_ip(sl0))
        blog0, btok0 = _branch_log_open()
        try:
            res0 = _cm.consistent_solve(fm, sl0.kinetics, sl0.sources, p_relaxed, cfg0, label=f"{sl0.label}:{mode}",
                                        initial_profiles=initial_profiles, state_fn=state_fn0)
        finally:
            _branch_log_close(btok0)
        branch0 = _branch_record(blog0, res0)
        params0 = p_relaxed
        cce0 = dict(computed=False, reason=f"no fit at the first slice (initial mode '{mode}'): no corrections")
        ssc0 = dict(computed=False, reason=f"no fit at the first slice (initial mode '{mode}'): its own result is the "
                                           "unfitted model steady state")
    # the first slice's steady-state check is its fit's; its solves are stated there, not in the slice's count
    n_chk0 = int((ssc0 or {}).get("n_solves") or 0)
    ip_c0 = (float(res0.assembly["ip"]) if res0.converged and res0.assembly and res0.assembly.get("ip")
             else _slice_ip(sl0))
    rec0 = _slice_record(fm, res0, sl0, sl0.label, dict(initial_mode=mode, fit=fitrec,
                                                        n_solves=fm.n_solves - n0 - n_chk0,
                                                        ip_carried_A=ip_c0,
                                                        ip_offset_carried=ip_c0 / _slice_ip(sl0) - 1.0,
                                                        **({"zeff_flag": "the series' Z_eff source fails its tests or "
                                                            "is unverified at this slice"}
                                                           if float(sl0.time) in zeff_flags else {}),
                                                        wall_total_s=round(time.perf_counter() - t0, 2),
                                                        vacuum_F0=f0rec0, correction_core_effect=cce0,
                                                        floor_branch_record=branch0,
                                                        ip_fitted=bool(fitrec and "ip_rel" in fitrec["free"])))
    n_corr = len(cfg.corr_centers_rho)
    sig_filter = _fit.correction_prior_sigmas(cfg, s.filter_fit)
    acc_corr = np.zeros(n_corr)            # accumulated absorbed filter amplitudes [prior sigmas] (a record only)
    n_absorbed = 0
    rec0["accumulated_correction"] = _accumulated_record(acc_corr, 0, s, sig_filter)
    if s.steady_state_check:
        rec0["steady_state_check"] = ssc0
    w0 =_sawtooth_warning(s, res0, float(times[0]), float(times[0]), [])
    if w0 is not None:
        rec0["q_below_one"] = w0
        if w0["warn"]:
            rec0["sawtooth_warning"] = w0
            out.setdefault("warnings", []).append(dict(slice=0, time=float(times[0]), sawtooth=w0["text"]))
    out["slices"].append(rec0)
    if callback:
        _rows.release_all()                          # the callback may use the solver: no constraint set kept
        callback(rec0)
    if res0.converged and mode in ("relaxed",) + FIT_START_MODES:
        out["relaxed_start_guard"] = _relaxed_start_guard(res0, sl0, mode)
    if not res0.converged:
        out.update(status="failed_initial", n_solves=fm.n_solves - n_solves0, wall_s=round(time.perf_counter() - t_all, 2))
        _note_iterations(out, fm, its0)
        return out
    crash = _crash_times(s, times[0], times[-1], ls_used)
    out["crash_times_used"] = crash.tolist() if s.sawtooth != "periodic" else dict(
        period_s=_period(s, ls_used), phase=("first slice: cycle average of the periodic orbit; first crash "
                                             f"{s.sawtooth_first_crash_phase:g} period after the first slice time"),
        first_crash_phase=float(s.sawtooth_first_crash_phase), f=s.sawtooth_f,
        log_strength=ls_used, log_strength_source=ls_source)
    st0 = res0.assembly["state"]
    carried = CurrentState(st0.rho_faces, st0.psi, st0.psi_boundary, times[0], dict(initial_mode=mode, ip_A=ip_c0))
    out["slices"][-1]["carried_state"] = _state_record(carried)
    lag_prev = res0.lag
    jni_prev = _jni(lag_prev, sl0.kinetics, sl0.sources, params0)
    # current corrections in the evolution (module docstring, "Fit interaction"): a filter fit's amplitudes are a
    # source over the interval ending at its slice only (the first slice's corrections are in its state)
    in_window = s.filter_correction == "in_window"
    hybrid = s.filter_correction == "carried_outside_core"
    out["filter_correction"] = dict(form=s.filter_correction, rule=FILTER_CORRECTION_RULES[s.filter_correction])
    rc_held = None
    if hybrid:
        # the core radius rule's held value starts from the first slice's own floor region (module docstring)
        core0 = (res0.assembly or {}).get("core") or {}
        rc_held = float(core0["rho_floor"]) if core0.get("rho_floor") is not None else None
        out["filter_correction"].update(carry_core_radius=s.carry_core_radius,
                                        carry_core_radius_scale=float(s.carry_core_radius_scale),
                                        carry_taper_width=float(s.carry_taper_width),
                                        first_slice_floor_region=rc_held)
    # Ip convention (module docstring, "Plasma current in a series"): the carried state's current is the first
    # slice's solved target, and the relative offset of the fitted from the measured Ip (0 without an Ip fit) is
    # carried forward: every later interval is driven from the carried current to the NEXT measured Ip times
    # (1 + carried offset); a filter fit that is absorbed updates the offset (one rescale by the change of the offset)
    ip_carried = ip_c0
    ip_off = float(ip_carried / _slice_ip(sl0) - 1.0)
    snap_prev = fm.snapshot()
    pc_prev = fm.probe_correction.copy()
    off_prev = _anchor_offset(fm)       # a moved anchor target (FitSettings.anchor_follow) is carried slice to slice
    anchor_box = dict(moves=anchor_moves0, carried=False)      # kept moves along the carried chain
    status = "completed"
    for k in range(1, len(slices)):
        sl = slices[k]
        t1, tp = times[k], carried.time      # the interval starts at the carried (last consistent) state
        tk0 = time.perf_counter()
        nk = fm.n_solves
        fm.restore(snap_prev)
        f0rec = _apply_f0(fm, sl)
        anchor_rec = None
        if sl.measurements is None:     # generation mode (synthetic truth series)
            fm.reconfigure(None, sl.reference_boundary, case_id=case.case_id, ip=_slice_ip(sl),
                           generation_coil_reg=sl.meta["generation_coil_reg"])
        else:
            pf = sl.measurements.groups.get("pf_current")
            roles = sl.meta.get("coil_roles") or (pf.meta.get("coil_roles") if pf is not None else None)
            if not roles:      # no roles stated for this slice: the previous slice's roles are kept (recorded)
                roles = {n: v["role"] for n, v in fm.rows_record.get("coils", {}).get("per_coil", {}).items()}
            fm.reconfigure(sl.measurements, sl.reference_boundary, coil_roles=roles, case_id=case.case_id,
                           **_reference_points_kw(sl))
            fm.probe_correction = pc_prev.copy()
            anchor_rec = _carry_anchor(fm, off_prev)
        fm.case = case
        ip0, ip1 = ip_carried, _slice_ip(sl) * (1.0 + ip_off)
        holder = {}
        p_pred = dict(log_strength=None, f=s.sawtooth_f,
                      s_bs=params0.get("s_bs", 1.0) if isinstance(params0, dict) else 1.0, ip_rel=ip_off)
        shapes_prev = {}
        rc_box = dict(r=None)      # carried_outside_core: this slice's core radius (set from its prediction)

        def state_fn(lag, params=None, sl=sl, t1=t1, tp=tp, ip0=ip0, ip1=ip1, carried=carried, lag_prev=lag_prev,
                     jni_prev=jni_prev, holder=holder, shapes_prev=shapes_prev, rc_box=rc_box):
            jni1 = _jni(lag, sl.kinetics, sl.sources, p_pred)
            j0, j1, amps, inc = jni_prev, jni1, None, None
            c0 = c1 = None
            parts1 = None
            if in_window:
                # the corrections as a non-inductive source over this interval only: the parameters' amplitudes (a
                # fit's free a_corr; zero for a prediction), on each end's geometry; nothing from earlier intervals
                # (the plain Gaussians, form "in_window", whatever corr_return says: module docstring)
                amps = (tuple(float(v) for v in _cm.full_params(params)["a_corr"]) if params is not None
                        else (0.0,) * n_corr)
                if any(a != 0.0 for a in amps):
                    if "g" not in shapes_prev:
                        shapes_prev["g"] = _cm.correction_shapes(lag_prev, cfg, ip0, form="in_window")
                    g1 = _cm.correction_shapes(lag, cfg, ip1, form="in_window")
                    c0 = sum(a * g for a, g in zip(amps, shapes_prev["g"]))
                    c1 = sum(a * g for a, g in zip(amps, g1))
                    if s.sawtooth != "q_floor":
                        j0, j1 = jni_prev + c0, jni1 + c1
            elif hybrid:
                # NON-DEFAULT form (module docstring): the core parts (1 - w) g_k a source over this interval only;
                # the outer parts a zero-net change of the state added at the end (carried on)
                amps = (tuple(float(v) for v in _cm.full_params(params)["a_corr"]) if params is not None
                        else (0.0,) * n_corr)
                if any(a != 0.0 for a in amps):
                    rcv = rc_box["r"]
                    if rcv is None:
                        raise RuntimeError("carried_outside_core: the slice's core radius is not set")
                    if "parts" not in shapes_prev:
                        shapes_prev["parts"] = _cm.carried_outside_core_parts(lag_prev, cfg, ip0, rcv,
                                                                              s.carry_taper_width, carried.rho_faces)
                    parts1 = _cm.carried_outside_core_parts(lag, cfg, ip1, rcv, s.carry_taper_width,
                                                            carried.rho_faces)
                    c0 = sum(a * g for a, g in zip(amps, shapes_prev["parts"]["core"]))
                    c1 = sum(a * g for a, g in zip(amps, parts1["core"]))
                    if s.sawtooth != "q_floor":
                        j0, j1 = jni_prev + c0, jni1 + c1
            sol, info = interval_solver(lag_prev, lag, j0, j1, tp, t1, ip0, ip1, s.layer_fpol)
            core = None
            if s.sawtooth == "q_floor":
                # the floor acts on the model state after every step; the corrections' in-window response (the
                # advance with them among the sources minus the one without) is added after it (module docstring)
                sol_a = (interval_solver(lag_prev, lag, jni_prev + c0, jni1 + c1, tp, t1, ip0, ip1, s.layer_fpol)[0]
                         if c0 is not None else None)
                opt = {k: v for k, v in (("total_current", getattr(cfg, "q_floor_total_current", "off")),
                                         ("transition", getattr(cfg, "q_floor_transition", "step")))
                       if v not in ("off", "step")}
                end, core = advance_q_floor(sol, carried, t1, s, s.dt_max, cfg.q_floor, sol_all=sol_a,
                                            form=cfg.q_floor_form, **opt)
                events, states = [], []
            else:
                end, events, states = advance_interval(sol, carried, t1, s, crash, s.dt_max)
            upd = None
            if parts1 is not None:
                # the carried outer parts: a zero-net change of the state, zero inside r_c (current_model.carried_update)
                dps, dpb, upd = _cm.carried_update(parts1, amps)
                end = CurrentState(end.rho_faces, end.psi + dps, end.psi_boundary + dpb, end.time, dict(end.meta))
                upd.update(r_core=float(rc_box["r"]), taper_width=float(s.carry_taper_width),
                           lam=[None if v is None else float(v) for v in parts1["lam"]])
            holder.clear()
            end.meta["ip_A"] = float(ip1)
            end.meta["current_scale"] = None          # this interval made no rescale
            end.meta["core"] = core if core is not None else dict(core_relaxation=f"series mode {s.sawtooth!r}")
            if in_window or hybrid:
                end.meta.update(correction_source=amps)
            if parts1 is not None:
                end.meta.update(carried_outside_core=dict(r_core=float(rc_box["r"]),
                                                          taper_width=float(s.carry_taper_width)))
            holder.update(end=end, events=events, states=states, info=info, jni1=jni1, window_average=False,
                          amps=amps)
            if upd is not None:
                holder["carried_update"] = upd
            use = end
            if events and s.crash_slice_state == "window_average":
                # COMPARISON OPTION (the earlier treatment): the sawtooth effect over the interval -- the interval
                # average WITH the crashes minus the same average WITHOUT them -- added to the crash-free end state;
                # the instantaneous end state is still the one carried, so the slice is not self-consistent
                end0, _, states0 = advance_interval(sol, carried, t1, s, np.zeros(0), s.dt_max)
                aw = _window_state(states, tp, t1, end.rho_faces)
                a0 = _window_state(states0, tp, t1, end.rho_faces)
                meta = dict(sawtooth_window=[tp, t1], ip_A=float(ip1))
                if in_window:
                    meta.update(correction_source=amps)
                use = CurrentState(end.rho_faces, end0.psi + aw.psi - a0.psi,
                                   end0.psi_boundary + aw.psi_boundary - a0.psi_boundary, t1, meta)
                holder["window_average"] = bool(any(e.record.active for e in events))
            return use

        state_fn.corrections_in_window = in_window or hybrid
        state_fn.side_state = holder            # rewritten on every call (read by the parallel columns, .speed)
        blog, btok = _branch_log_open()          # the floor's branch at every assembly of this slice (a record)
        rc_rec = None
        try:
            res = _cm.consistent_solve(fm, sl.kinetics, sl.sources, p_pred, cfg, label=f"{sl.label}:evolve",
                                       state_fn=state_fn)
            if res.converged:
                # the loop's last state-function call was made on the equilibrium of the solve AFTER the accepted
                # pass; re-advance on the accepted pass's equilibrium, so that the state carried on is the state
                # scored
                _cm.call_state_fn(state_fn, res.lag, p_pred)
                if hybrid:
                    # the slice's core radius, from its prediction (model state), before any fit (module docstring)
                    rc_box["r"], rc_held, rc_rec = _core_radius(s, (res.assembly or {}).get("core"), rc_held)
            filt = None
            if res.converged and s.filter_at(t1):
                anchor_box["carried"] = bool(anchor_rec and anchor_rec.get("carried"))
                filt, res = _filter_carried(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1, anchor_box,
                                            **_speed_kw(speed))
            cce = _interval_correction_effect(sl, cfg, s, state_fn, res, filt)
        finally:
            _branch_log_close(btok)
        crash_rec = [dict(time=e.time, active=bool(e.record.active), rho_mix=getattr(e.record, "rho_mix", None),
                          dpsi_max=getattr(e.record, "dpsi_max", None)) for e in holder.get("events", [])]
        extra = dict(n_solves=fm.n_solves - nk, interval=dict(holder.get("info", {}), start_s=float(tp), end_s=float(t1),
                                                              ip_start_A=float(ip0), ip_end_A=float(ip1),
                                                              ip_offset_used=ip_off,
                                                              correction_source=(list(holder["amps"])
                                                                                 if holder.get("amps") is not None
                                                                                 else None)),
                     ip_fitted=bool(filt and filt.get("absorbed") and "ip_rel" in (filt.get("free") or [])),
                     crashes=crash_rec if res.converged else [],
                     window_average=holder.get("window_average", False), filter=filt, vacuum_F0=f0rec,
                     correction_core_effect=cce, wall_total_s=round(time.perf_counter() - tk0, 2),
                     floor_branch_record=_branch_record(blog, res, (filt or {}).get("floor_branch_record")))
        if hybrid:
            extra["carry_core_radius"] = rc_rec
        if res.converged and filt and filt.get("absorbed"):
            acc_corr = acc_corr + _prior_units(filt.get("a_corr"), sig_filter)
            n_absorbed += 1
        extra["accumulated_correction"] = _accumulated_record(acc_corr, n_absorbed, s, sig_filter)
        if not res.converged and crash_rec:
            # the crashes of the failed slice's last attempt were never applied to a carried state
            extra["crashes_of_failed_attempt"] = crash_rec
            extra["crashes_note"] = ("slice not consistent: the crashes of its last attempt were not applied; the next "
                                     "interval starts from the last consistent state")
        if res.converged:
            extra["carried_state"] = _state_record(holder["end"])
            sc, ca = res.assembly["state"], holder["end"]
            same = bool(np.array_equal(np.asarray(sc.psi), np.asarray(ca.psi)) and sc.psi_boundary == ca.psi_boundary)
            extra["scored_vs_carried_state"] = dict(
                zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(ca, sc)), same_state=same,
                crash_slice_state=s.crash_slice_state,
                note=("the scored state is the carried end state (identical arrays)" if same else
                      "COMPARISON OPTION crash_slice_state='window_average' with an active crash in the interval: the "
                      "scored state is the crash-free end state plus the interval's sawtooth effect; the carried "
                      "state is the instantaneous end state" if holder.get("window_average") else
                      "the scored and the carried states differ"))
        # what the next interval starts from (taken before the callback, which may use the solver)
        nxt = dict(carried=holder.get("end"), jni1=holder.get("jni1"), ip=float(holder.get("ip_end", ip1)),
                   ip_off=float(holder.get("ip_offset_end", ip_off)))
        if res.converged:
            extra.update(ip_carried_A=nxt["ip"], ip_offset_carried=nxt["ip_off"])
        else:
            extra.update(ip_carried_A=None, ip_offset_carried=None,
                         ip_carry_note="slice not consistent: the previous carried state, current and offset are kept")
        if float(sl.time) in zeff_flags:
            extra["zeff_flag"] = "the series' Z_eff source fails its tests or is unverified at this slice"
        rec = _slice_record(fm, res, sl, sl.label, extra)
        if anchor_rec is not None:
            rec["anchor_target"] = anchor_rec
        if s.steady_state_check and res.converged:
            # a record (module docstring); the solver is restored to the slice's own result afterwards
            rec["steady_state_check"] = _steady_state_check(fm, sl, cfg, s, p_pred)
        w = _sawtooth_warning(s, res, float(tp), float(t1), holder.get("events", []))
        if w is not None:
            rec["q_below_one"] = w
            if w["warn"]:
                rec["sawtooth_warning"] = w
                out.setdefault("warnings", []).append(dict(slice=k, time=float(t1), sawtooth=w["text"]))
        out["slices"].append(rec)
        if callback:
            _rows.release_all()                      # the callback may use the solver: no constraint set kept
            callback(rec)
        if not res.converged:
            status = f"failed_at_slice_{k}"
            if not s.keep_going_after_failure:
                break
            continue
        carried = nxt["carried"]
        ip_carried, ip_off = nxt["ip"], nxt["ip_off"]
        lag_prev = res.lag
        jni_prev = nxt["jni1"]
        snap_prev = fm.snapshot()
        pc_prev = fm.probe_correction.copy()
        off_prev = _anchor_offset(fm)
    out["boundary_displacement"] = boundary_displacement_history(out["slices"])
    out.update(status=status, n_solves=fm.n_solves - n_solves0, wall_s=round(time.perf_counter() - t_all, 2))
    _note_iterations(out, fm, its0)
    return out


def _note_iterations(out, fm, start):
    """``out["solver_iterations"]``: every solve's iteration count over the series and the iteration limit in effect
    (:meth:`.forward.ForwardModel.iteration_record` from index ``start``); a record only, never raises, absent for a
    forward model without the log."""
    get_its = getattr(fm, "iteration_record", None)
    if get_its is None:
        return
    try:
        out["solver_iterations"] = get_its(start=start)
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        out["solver_iterations"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def check_zeff_series(slices, source=None, schedule=None):
    """One Z_eff source for the whole series (the checked-source rule's series provision): every slice's
    kinetic inputs must use the same source, never switched between slices.

    * kinetics without a Z_eff record (e.g. :class:`.current_model.KineticProfiles`): nothing to check (recorded);
    * slices whose kinetic inputs name different Z_eff candidates are refused (``ValueError``);
    * when the slices were assembled with the checked-source rule (``record["zeff_rule"]``), the series needs
      its source named (``source``: ``"impurity_density"`` or ``"line_integrated"``,
      :func:`.zeff_sources.series_source`; without a name it is refused with the per-slice outcomes), every
      slice must have used that source (assemble them with ``zeff_rule={"override": ...}`` when the rule would
      pick another), and the slices on which it fails its tests or is unverified are flagged
      (``failing_slices`` / ``unverified_slices``, slice times), never switched.

    ``schedule`` (NON-DEFAULT; e.g. named sources by phase of a discharge): a list of declared intervals
    ``{"start": t0, "end": t1, "candidate": name, "source": rule source or None}`` [s] (``start <= t < end``),
    declared up front.  Every slice must lie in exactly one interval and use that interval's candidate (anything
    else is refused); the check above -- including the checked-source rule with the interval's ``source`` and
    its failure / unverified flags -- is applied WITHIN each interval.  The declared list is returned
    (``schedule``) with the per-interval results; the flags are the union over the intervals.
    """
    if schedule is not None:
        decl = [dict(start=float(d["start"]), end=float(d["end"]), candidate=str(d["candidate"]),
                     source=d.get("source")) for d in schedule]
        if any(not d["end"] > d["start"] for d in decl):
            raise ValueError("Z_eff schedule: every interval needs end > start")
        srt = sorted(decl, key=lambda d: d["start"])
        over = [(a, b) for a, b in zip(srt[:-1], srt[1:]) if b["start"] < a["end"]]
        if over:
            raise ValueError("Z_eff schedule: intervals overlap ("
                             + "; ".join(f"[{a['start']}, {a['end']}) and [{b['start']}, {b['end']})" for a, b in over)
                             + "): every slice must lie in exactly one declared interval")
        out = dict(applied=True, schedule=decl, intervals=[], failing_slices=[], unverified_slices=[],
                   note="NON-DEFAULT: Z_eff sources declared by interval; the one-source check applied within each")
        for d in decl:
            sel = [sl for sl in slices if d["start"] <= float(sl.time) < d["end"]]
            if not sel:
                out["intervals"].append(dict(d, n_slices=0))
                continue
            names = {(getattr(sl.kinetics, "record", None) or {}).get("zeff_source") for sl in sel}
            if names != {d["candidate"]}:
                raise ValueError(f"Z_eff schedule: the slices in [{d['start']}, {d['end']}) use "
                                 f"{sorted(map(str, names))}, declared {d['candidate']!r}")
            r = check_zeff_series(sel, d["source"])
            out["intervals"].append(dict(d, n_slices=len(sel), result=r))
            out["failing_slices"] += list(r.get("failing_slices") or [])
            out["unverified_slices"] += list(r.get("unverified_slices") or [])
        outside = [float(sl.time) for sl in slices
                   if not any(d["start"] <= float(sl.time) < d["end"] for d in decl)]
        if outside:
            raise ValueError(f"Z_eff schedule: slices at {outside} lie in no declared interval")
        return out
    from . import zeff_sources as _zr
    recs = [dict(getattr(sl.kinetics, "record", None) or {}) for sl in slices]
    names = [r.get("zeff_source") for r in recs]
    if not any(n is not None for n in names):
        return dict(applied=False, note="the kinetic inputs carry no Z_eff record: one-source check not applicable")
    if None in names:
        raise ValueError("Z_eff: some slices carry a Z_eff source record and some do not; one source per series")
    if len(set(names)) > 1:
        raise ValueError(f"Z_eff source switches within the series ({sorted(set(names))}): a series takes one "
                         "named source (or declare the intervals with zeff_source_schedule)")
    rules = [r.get("zeff_rule") for r in recs]
    out = dict(applied=True, candidate=names[0], failing_slices=[], unverified_slices=[])
    if not any(rr is not None for rr in rules):
        out["note"] = "every slice uses the same named candidate (no checked-source rule)"
        return out
    if None in rules:
        raise ValueError("Z_eff: the checked-source rule was applied on some slices only; one source per series")
    rr = [dict(r, time=float(sl.time)) for r, sl in zip(rules, slices)]
    res = _zr.series_source(rr, source)
    used = [r["decision"]["source"] for r in rules]
    if any(u != source for u in used):
        bad = [float(sl.time) for u, sl in zip(used, slices) if u != source]
        raise ValueError(f"Z_eff: the series source is {source!r} but the slices at {bad} used another source; "
                         "assemble them with zeff_rule={'override': <the named candidate>}")
    out.update(source=source, failing_slices=list(res["failing_slices"]),
               unverified_slices=list(res["unverified_slices"]), per_slice=res["per_slice"])
    return out


def _sawtooth_warning(s, res, t0, t1, events):
    """The q < 1 record of the opt-in modes ``"crash_times"`` and ``"off"`` (module docstring), or None when the
    slice's solved q stays >= 1, a crash acted in the interval, or the mode is ``"q_floor"`` or ``"periodic"``.
    ``warn`` says whether it is a WARNING (listed in the series' ``warnings``); ``kind``:

    * ``no_record`` (warning): no crash record, no crashes applied (``sawtooth="off"``);
    * ``not_covered`` (warning): the record's detection did not cover the interval (or since the last recorded
      crash), or its coverage was not stated;
    * ``crash_recorded_not_active`` (warning): a crash was recorded in the interval but the operator acted as the
      identity (the evolved q >= 1 at its time) and q < 1 at the slice -- the model's q fell below 1 later than
      the measured crash;
    * ``covered_no_crash`` (warning, a model-data inconsistency): the record covers the time since the last
      recorded crash (or since its start, when no crash was recorded yet) and that crash-free stretch is longer
      than the longest interval between consecutive recorded crashes over covered time (or fewer than two
      crashes are recorded, so no crash spacing is measured);
    * ``between_crashes`` (no warning): q < 1 within the measured crash spacing since the last recorded crash --
      normal between sawtooth crashes.

    The record holds ``q_min``, the gap since the last recorded crash and the longest recorded interval used."""
    if not res.converged or res.lag is None or s.sawtooth in ("periodic", "q_floor"):
        # q_floor: the model q is held at the floor; a q below it can come only from the fitted corrections and is
        # recorded in the slice's core record (q_min_model_minus_floor), not classified here
        return None
    q = np.abs(np.asarray(res.lag["phys"]["q"], float))
    qmin = float(np.min(q[np.isfinite(q)])) if np.any(np.isfinite(q)) else None
    if qmin is None or qmin >= 1.0 or any(bool(e.record.active) for e in events):
        return None
    rec = dict(q_min=qmin, interval=[t0, t1])
    tab = s.crash_table
    if s.sawtooth == "off":
        kind, why = "no_record", "no crash record: no crashes applied (sawtooth off)"
    elif events:
        kind = "crash_recorded_not_active"
        why = (f"a crash was recorded in this interval (t = {', '.join(f'{e.time:.4f}' for e in events)} s) but the "
               "operator acted as the identity (the evolved q >= 1 at its time): the model's q fell below 1 later "
               "than the measured crash")
    else:
        cov = tab.covered(t0, t1) if tab is not None else None
        ct = np.asarray(s.crash_times if s.crash_times is not None else (), float)
        last = ct[ct <= t1]
        t_ref = float(last[-1]) if last.size else (tab.record_start if tab is not None else None)
        cov_gap = (tab.covered(t_ref, t1) if (tab is not None and t_ref is not None) else None)
        longest = None
        if tab is not None and ct.size >= 2:
            iv = [float(b - a) for a, b in zip(ct[:-1], ct[1:]) if tab.covered(float(a), float(b)) is True]
            longest = max(iv) if iv else None
        gap = None if t_ref is None else float(t1 - t_ref)
        rec.update(last_recorded_crash=(float(last[-1]) if last.size else None), gap_since_last_crash_s=gap,
                   longest_recorded_interval_s=longest)
        if cov is not True or cov_gap is not True:
            kind = "not_covered"
            why = ("the crash record's detection did not cover this interval (or the time since the last recorded "
                   "crash): no crashes applied" if (cov is False or cov_gap is False) else
                   "no measured crash in this interval and the record's coverage was not stated")
        elif not last.size:
            kind = "covered_no_crash"
            why = ("the crash record covers the time since its start and shows no crash at all: model-data "
                   "inconsistency (the evolved q < 1 where no sawtooth was observed)")
        elif longest is None:
            kind = "covered_no_crash"
            why = (f"no crash recorded for {gap:.4f} s and fewer than two crashes over covered time (no measured "
                   "crash spacing): model-data inconsistency (the evolved q < 1 where no sawtooth was observed)")
        elif gap > longest:
            kind = "covered_no_crash"
            why = (f"no crash recorded for {gap:.4f} s, longer than the longest recorded interval between crashes "
                   f"({longest:.4f} s): model-data inconsistency (the evolved q < 1 where no sawtooth was observed)")
        else:
            kind = "between_crashes"
            why = (f"q < 1 {gap:.4f} s after the last recorded crash, within the longest recorded interval between "
                   f"crashes ({longest:.4f} s): normal between sawtooth crashes (no warning)")
    warn = kind != "between_crashes"
    text = f"evolved q_min = {qmin:.3f} < 1 at t = {t1:.4f} s; {why}"
    if warn:
        import warnings
        warnings.warn(text, RuntimeWarning, stacklevel=3)
    rec.update(text=text, kind=kind, warn=warn)
    return rec


#: Relaxed-start applicability guard (warning only): reference q_min and non-inductive fraction limits.
RELAXED_GUARD_Q_MIN = 1.2
RELAXED_GUARD_F_NI = 0.5


def _relaxed_start_guard(res0, sl0, mode):
    """Warn when the first slice's relaxed-current start is applied outside its range: reference q_min >
    :data:`RELAXED_GUARD_Q_MIN` or non-inductive (bootstrap + driven) current fraction >
    :data:`RELAXED_GUARD_F_NI`.  The reference q_min is ``sl0.meta["reference_q_min"]`` when given (e.g. the
    reference equilibrium's), else the first slice's solved equilibrium's q on rho_tor,norm <= 0.95.  A
    warning only (recorded); nothing is refused."""
    import warnings
    ph = res0.lag["phys"] if res0.lag is not None else None
    qmin, qsrc = sl0.meta.get("reference_q_min"), "slice meta reference_q_min"
    if qmin is None and ph is not None:
        m = np.asarray(ph["rho"]) <= 0.95
        qmin, qsrc = float(np.min(np.asarray(ph["q"])[m])), "first slice's solved equilibrium (rho_tor,norm <= 0.95)"
    fr = (res0.assembly or {}).get("fractions") or {}
    fni = float(fr.get("bootstrap", 0.0) + fr.get("driven", 0.0)) if fr else None
    reasons = []
    if qmin is not None and qmin > RELAXED_GUARD_Q_MIN:
        reasons.append(f"reference q_min = {qmin:.3g} > {RELAXED_GUARD_Q_MIN}")
    if fni is not None and fni > RELAXED_GUARD_F_NI:
        reasons.append(f"non-inductive fraction = {fni:.3g} > {RELAXED_GUARD_F_NI}")
    rec = dict(mode=mode, q_min=qmin, q_min_source=qsrc, f_ni=fni, q_min_limit=RELAXED_GUARD_Q_MIN,
               f_ni_limit=RELAXED_GUARD_F_NI, warned=bool(reasons), reasons=reasons)
    if reasons:
        warnings.warn(f"the first slice's start rests on a relaxed ohmic current ({mode}) outside its range of "
                      f"applicability: {'; '.join(reasons)}", RuntimeWarning, stacklevel=3)
    return rec


def _measured_crash_start(s, cfg, times):
    """First-slice settings with measured crash times: the slice-run state of the current model
    (:func:`.current.sawtooth.simulate_slice`: periodic start at the measured mean interval, the measured
    crashes from ``history`` before the window on) averaged over the window ``[t_0 - (t_1 - t_0), t_0]``
    -- the same window convention as every later slice (the interval since the previous slice).  The
    strength is the one of the measured mean interval at ``sawtooth_f`` (it sets only the orbit's step).
    Returns ``(cfg0, params, record)``; a crash record that does not cover the history raises, and so do bare
    ``crash_times`` (no table: the detection coverage is not stated)."""
    import dataclasses
    if len(times) < 2:
        raise ValueError("sawtooth='crash_times' needs at least two slices (the first window is the first interval)")
    if s.crash_table is None:
        # bare crash times state no detection coverage, so the history before the first slice was not shown to be
        # observed: refused like a table whose coverage is not stated
        raise ValueError("measured crash times cannot start the series: crash times given without a crash table state "
                         "no detection coverage ('no crash recorded' is not 'no crash'); give crash_table= a "
                         "crash_table.CrashTable with its record_start / record_end (and unobserved intervals)")
    window = (float(times[0] - (times[1] - times[0])), float(times[0]))
    try:
        sched = _saw.schedule_for_slice(s.crash_times, window, history=cfg.history, record_start=s.crash_record_start)
    except ValueError as exc:
        raise ValueError(f"measured crash times cannot start the series at its first slice: {exc}") from exc
    n_hist = int(np.count_nonzero((np.asarray(s.crash_times, float) > sched.t_start)
                                  & (np.asarray(s.crash_times, float) <= window[1])))
    # the record must have OBSERVED the whole history, whatever the number of crashes recorded in it: 'no crash
    # recorded' is not 'no crash', and 'fewer crashes recorded' is not 'fewer crashes' (the starting crash rate is
    # set from the recorded ones)
    if s.crash_table is not None and s.crash_table.covered(sched.t_start, window[1]) is not True:
        raise ValueError(f"measured crash times cannot start the series: {n_hist} crash(es) recorded in the history "
                         f"[{sched.t_start}, {window[1]}], but the crash record did not observe all of it (unobserved "
                         "interval or coverage not stated): 'no crash recorded' is not 'no crash', and a crash count "
                         "over unobserved time is not a measured crash rate")
    if n_hist == 0:
        # no crash RECORDED in the covered history
        # the record covers the history (checked above) and holds no crash in it: the first slice is crash-free
        rec = dict(window=list(window), history_s=cfg.history, record_start=s.crash_record_start, n_crashes_history=0,
                   note="no measured crash in the covered history before the first slice: the first slice is the "
                        "crash-free (steady) model state; the measured crashes act from then on")
        return cfg, dict(log_strength=None, f=s.sawtooth_f), rec
    if sched.mean_interval is None:
        raise ValueError("measured crash times cannot start the series: one crash in the history "
                         f"[{sched.t_start}, {window[1]}] (two are needed to set the starting orbit)")
    ls = float(np.log(_saw.crash_strength(s.sawtooth_f, 1.0 / sched.mean_interval)))
    cfg0 = dataclasses.replace(cfg, core_relaxation="crash_times", crash_times=tuple(float(x) for x in s.crash_times),
                               window=window, crash_record_start=s.crash_record_start)
    rec = dict(window=list(window), history_s=cfg.history, mean_interval_s=sched.mean_interval,
               record_start=s.crash_record_start,
               n_crashes_history=sched.n_crashes_history, n_crashes_window=sched.n_crashes_window,
               last_crash_before_history=sched.last_crash_before_start, f=s.sawtooth_f, log_strength=ls,
               note="first slice: window average of the measured-crash slice run over the first interval's length "
                    "before the first slice time; the series carries that state")
    return cfg0, dict(log_strength=ls, f=s.sawtooth_f), rec


def _apply_f0(fm, sl):
    """Set the slice's vacuum ``F0`` on the forward model (kept when the slice gives none)."""
    if sl.F0 is None:
        return dict(F0=float(fm.F0), source="not given for this slice: the previous value is kept")
    fm.set_F0(float(sl.F0))
    return dict(F0=float(fm.F0), source="slice")


def _state_record(st):
    m = st.meta or {}
    return dict(time=float(st.time), rho_faces=np.asarray(st.rho_faces).tolist(), psi=np.asarray(st.psi).tolist(),
                psi_boundary=float(st.psi_boundary), ip_A=m.get("ip_A"), current_scale=m.get("current_scale"))


def _psi_shape_diff(a, b):
    """``max |(a - b) - (a - b)[0]| / max |b - b[0]|`` on the cells (the gauge-free size of a state
    difference relative to the state's psi span) and the boundary-value difference [Wb/rad]."""
    d = np.asarray(a.psi, float) - np.asarray(b.psi, float)
    span = float(np.max(np.abs(np.asarray(b.psi, float) - float(b.psi[0])))) or 1.0
    return float(np.max(np.abs(d - d[0])) / span), float(a.psi_boundary - b.psi_boundary)


def _follow_settings(s, box):
    """The filter-fit settings of this slice under the bound on kept moves along a carried chain
    (``SeriesSettings.anchor_follow_total``): the moving target's cap is the smaller of the filter fit's own and what
    the bound leaves; with nothing left the fit runs without further moves."""
    ff = s.filter_fit
    if not ff.anchor_follow or s.anchor_follow_total is None:
        return ff
    left = int(s.anchor_follow_total) - int(box.get("moves", 0))
    if left <= 0:
        return dataclasses.replace(ff, anchor_follow=False)
    return dataclasses.replace(ff, anchor_follow_max_steps=min(int(ff.anchor_follow_max_steps), left))


def _kept_moves(filt):
    return int(((filt or {}).get("anchor_follow") or {}).get("n_moves_kept") or 0)


def _filter_carried(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1, box, speed=None):
    """The filter fit of a slice with the moving anchor target (:func:`_filter_slice` otherwise, unchanged).

    Where the slice's anchor target was carried from the previous slice and ``SeriesSettings.anchor_carry_retest`` is
    on, two filter fits are made from the same evolved state: one on the slice's own reference boundary (no moves) and
    one on the carried target (with the further moves the settings allow).  The carried one is used only if it is
    absorbed and its fitted chi^2 (``chi2["data"]``) is lower than the reference one's (or the reference one is not
    absorbed); otherwise the reference fit is the slice's answer, the target is the reference boundary again, and
    nothing is carried on.  The comparison is recorded (``filter["anchor_retest"]``)."""
    kw = _speed_kw(speed)
    if not (box.get("carried") and s.anchor_carry_retest and hasattr(fm, "set_anchor_target")):
        filt, out = _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1, filter_fit=_follow_settings(s, box),
                                  **kw)
        box["moves"] = int(box.get("moves", 0)) + _kept_moves(filt) if _anchor_offset(fm) is not None else 0
        return filt, out
    pre = dict(snap=fm.snapshot(), pc=fm.probe_correction.copy(), holder=dict(holder), target=fm.anchor_target())

    def chi(f):
        return float((f.get("chi2") or {}).get("data")) if f.get("absorbed") else None

    fm.set_anchor_target(fm.reference_boundary)
    filt_r, res_r = _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1,
                                  filter_fit=dataclasses.replace(s.filter_fit, anchor_follow=False), **kw)
    ref = dict(snap=fm.snapshot(), pc=fm.probe_correction.copy(), holder=dict(holder))
    fm.restore(pre["snap"])
    fm.probe_correction = pre["pc"].copy()
    holder.clear()
    holder.update(pre["holder"])
    fm.set_anchor_target(pre["target"])
    filt_c, res_c = _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1,
                                  filter_fit=_follow_settings(s, box), **kw)
    c_r, c_c = chi(filt_r), chi(filt_c)
    use_carried = c_c is not None and (c_r is None or c_c < c_r)
    rec = dict(chi2_reference=c_r, chi2_carried=c_c, status_reference=filt_r.get("status"),
               status_carried=filt_c.get("status"), kept="carried" if use_carried else "reference",
               n_solves_reference=filt_r.get("n_solves"), n_solves_carried=filt_c.get("n_solves"),
               rule="the carried target is used only if its filter fit is absorbed with a lower fitted chi^2 than "
                    "the filter fit on the slice's own reference boundary")
    if use_carried:
        box["moves"] = int(box.get("moves", 0)) + _kept_moves(filt_c)
        filt_c["anchor_retest"] = rec
        return filt_c, res_c
    fm.set_anchor_target(fm.reference_boundary)
    fm.restore(ref["snap"])
    fm.probe_correction = ref["pc"].copy()
    holder.clear()
    holder.update(ref["holder"])
    box["moves"] = 0
    filt_r["anchor_retest"] = rec
    return filt_r, res_r


def _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1, speed=None, filter_fit=None):
    """Filter fit of one evolved slice (module docstring).  Returns ``(record, result)``; on return
    the solver holds the slice's answer and ``holder`` the state to carry.  ``filter_fit``: the fit settings in place
    of ``s.filter_fit`` (the moving anchor target's per-slice settings, :func:`_filter_carried`)."""
    pred = dict(res=res, holder=dict(holder), snap=fm.snapshot(), pc=fm.probe_correction.copy())
    fr = _fit.reconstruct(fm, sl.kinetics, sl.sources, cfg, filter_fit or s.filter_fit, params0=p_pred,
                          start=pred["snap"], label=f"{sl.label}:filter", state_fn=state_fn, **_speed_kw(speed))
    stale = holder.get("end")       # advanced by the LAST solve of the fit (a column, a trial or the center)
    c = fr.record.get("center")
    filt = dict(status=fr.status, absorbed=False, free=list(fr.free), theta=fr.theta.tolist(), sigma=fr.sigma.tolist(),
                ip_fit=fr.record.get("ip_fit"), flux_offset=(fr.chi2 or {}).get("flux_offset"),
                prior={n: fr.record["specs"].get(n, {}).get("prior") for n in fr.free}, chi2=fr.chi2,
                n_solves=fr.record["n_solves"], wall_s=fr.record["wall_s"],
                stopping=fr.record.get("final_stopping_tests"), stopping_summary=fr.record.get("stopping_summary"),
                floor_branch_record=fr.record.get("floor_branch_record"),
                anchor_follow=fr.record.get("anchor_follow"),      # None unless FitSettings.anchor_follow is on
                a_corr=[float(v) for v in _cm.full_params(fr.params)["a_corr"]],
                correction_size=_correction_size(fr.params, cfg, s.filter_fit))
    if c is None or fr.status not in ACCEPTED_FIT_STATUSES:
        # not absorbed (stalled, max_iter, failed Jacobian or trials, failures): the slice stays at its evolved,
        # unfitted equilibrium and state
        fm.restore(pred["snap"])
        fm.probe_correction = pred["pc"].copy()
        holder.clear()
        holder.update(pred["holder"])
        filt["note"] = (f"filter fit not converged ({fr.status}): its corrections are not absorbed; the evolved, "
                        "unfitted equilibrium is scored and its state carried")
        return filt, pred["res"]
    res = c["res"]
    fm.restore(c["snap"])
    fm.probe_correction = c["probe_correction"].copy()
    # re-advance the carried state on the accepted center's own equilibrium (the geometry its assembly used);
    # with the Ip target fitted, the state is brought to the fitted Ip ONCE, by the ratio of the fitted target to
    # the current it was advanced to (as in the fit, fit._scaled_state_fn), and the fitted offset is carried on
    ip_meas = _slice_ip(sl)
    off_new = float(fr.params.get("ip_rel", 0.0) or 0.0)
    ip_fit = ip_meas * (1.0 + off_new)
    ac = fr.params["a_corr"]
    k_ip = ip_fit / float(ip1)
    asm_st = res.assembly["state"]
    if s.filter_correction in ("in_window", "carried_outside_core"):
        # the corrections acted as a source over the interval (carried_outside_core: inside r_c; outside it their
        # zero-net change of the state is part of the same state function): re-advancing with the center's parameters
        # (the same function the fit evaluated, fit._scaled_state_fn) gives the state the center scored; that end
        # state is carried, and the source stops at this slice (not carried into the next interval)
        use_c = _fit._scaled_state_fn(state_fn, fr.params, ip_meas)(res.lag)
        if k_ip != 1.0:
            holder["end"] = _cm.scale_state(holder["end"], k_ip)
        holder["ip_end"] = ip_fit
        holder["ip_offset_end"] = off_new
        e = holder["end"]
        holder["end"] = CurrentState(e.rho_faces, e.psi, e.psi_boundary, e.time,
                                     dict(e.meta, filter_corrections_absorbed=list(ac)))
        hyb = s.filter_correction == "carried_outside_core"
        filt.update(absorbed=True,
                    carried_from=(("state re-advanced on the accepted center's equilibrium with the corrections' core "
                                   "parts as a source over the interval and their outer parts as a zero-net change of "
                                   "the state outside r_c (carried_outside_core); the source stops at the slice")
                                  if hyb else
                                  ("state re-advanced on the accepted center's equilibrium with the corrections as a "
                                   "source over the interval (in-window); the source stops at the slice")),
                    correction_source=list(holder.get("amps") or ()),
                    center_state_check=dict(zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(use_c, asm_st))),
                    last_solve_state_vs_center=(dict(zip(("max_dpsi_rel", "dpsi_boundary"),
                                                         _psi_shape_diff(stale, holder["end"])))
                                                if stale is not None else None))
        if hyb:
            filt["carried_update"] = holder.get("carried_update")
        return filt, res
    use_c = state_fn(res.lag)
    if k_ip != 1.0:
        use_c = _cm.scale_state(use_c, k_ip)
        holder["end"] = _cm.scale_state(holder["end"], k_ip)
    holder["ip_end"] = ip_fit
    holder["ip_offset_end"] = off_new
    e = holder["end"]
    dpsi, dpb = np.zeros_like(e.psi), 0.0
    if any(a != 0.0 for a in ac):
        shapes = _cm.correction_shapes(res.lag, cfg, ip_fit)
        dpsi, dpb = _cm.correction_response(res.lag, shapes, ac, e.rho_faces)
    holder["end"] = CurrentState(e.rho_faces, e.psi + dpsi, e.psi_boundary + dpb, e.time,
                                 dict(e.meta, filter_corrections_absorbed=list(ac)))
    # checks (recorded): the re-advanced, corrected state is the center's own assembly state, and the size
    # of the difference to the state of the fit's last solve (which is not carried)
    chk = CurrentState(e.rho_faces, use_c.psi + dpsi, use_c.psi_boundary + dpb, e.time)
    filt.update(absorbed=True,
                carried_from="COMPARISON OPTION: state re-advanced on the accepted center's equilibrium, plus the "
                             "corrections' steady response",
                center_state_check=dict(zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(chk, asm_st))),
                last_solve_state_vs_center=(dict(zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(stale, e)))
                                            if stale is not None else None))
    return filt, res


def _interval_correction_effect(sl, cfg, s, state_fn, res, filt):
    """The corrections' effect on the core at an evolved slice (:func:`.current_model.correction_core_effect`; a
    record only), in the form the slice used: for an absorbed filter fit, the slice's state re-made with the
    correction amplitudes set to zero -- in-window: the interval re-advanced from the same carried state with no
    correction source, on the accepted center's geometry and Ip target; steady comparison option: the same
    re-advanced state without the corrections' steady response -- and assembled on the same geometry.  Nothing the
    series carries is touched (a separate state holder).  A prediction slice or a fit that was not absorbed applies
    no correction in its interval; the response of EARLIER corrections carried in the state is not separated here
    (that would need the history re-run without them)."""
    if not res.converged:
        return dict(computed=False, reason="slice not consistent")
    if not (filt and filt.get("absorbed")):
        return dict(computed=False, reason=("no correction source in this interval (" +
                                            ("prediction slice" if filt is None else "filter fit not absorbed") +
                                            "); the response of earlier corrections carried in the state is not "
                                            "separated (that needs the history re-run without them)"))
    try:
        asm = res.assembly
        p0 = _cm.zero_corrections(asm["params"])
        fn0 = functools.partial(state_fn, holder={})
        fn0.corrections_in_window = bool(getattr(state_fn, "corrections_in_window", False))
        st0 = _fit._scaled_state_fn(fn0, p0, _slice_ip(sl))(res.lag)
        asm0 = _cm.assemble(res.lag, sl.kinetics, sl.sources, p0, cfg, asm["ip"], state=st0)
        how = (("carried_outside_core form: the interval re-advanced from the same carried state with no correction "
                "source and no carried change, on the accepted center's geometry and Ip target, and assembled there; "
                "no new equilibrium solve")
               if s.filter_correction == "carried_outside_core" else
               ("in-window form: the interval re-advanced from the same carried state with no correction source, on "
                "the accepted center's geometry and Ip target, and assembled there; no new equilibrium solve")
               if fn0.corrections_in_window else
               ("steady comparison option: the re-advanced state of the accepted center without the corrections' "
                "steady response, assembled on its geometry; no new equilibrium solve"))
        return _cm.correction_core_effect(res.lag, asm, asm0, cfg, _fit.correction_prior_sigmas(cfg, s.filter_fit),
                                          how)
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _core_radius(s, core, held):
    """The core radius ``r_c`` of one slice in the form ``filter_correction="carried_outside_core"`` (module docstring)
    and the held floor region for the next slice.  ``core``: the slice's core record from its prediction (model state,
    before any fit); ``held``: the floor region of the latest slice that had one (None: none yet).  Returns
    ``(r_c, held_new, record)``; the rule reads the model state only, never the data."""
    if not isinstance(s.carry_core_radius, str):
        r = float(s.carry_core_radius)
        return r, held, dict(r_core=r, rule="fixed rho_tor,norm (carry_core_radius)")
    rho_f = (core or {}).get("rho_floor")
    scale = float(s.carry_core_radius_scale)
    if rho_f is not None:
        held = float(rho_f)
        src = "the floor's region at this slice (its prediction)"
    elif held is not None:
        src = "no floor region at this slice: the region of the latest slice that had one, held"
    else:
        src = "the floor has had no region in this series: r_c = 0"
    r = scale * held if held is not None else 0.0
    return r, held, dict(r_core=r, rule="floor_region", scale=scale, rho_floor_slice=rho_f, rho_floor_used=held,
                         source=src)


def _prior_units(amps, sigmas):
    """Amplitudes divided by their prior sigmas (NaN where a prior sigma is missing or not positive)."""
    a = np.asarray([float(v) for v in (amps or [])], float)
    sg = np.asarray([np.nan if v is None or not float(v) > 0 else float(v) for v in sigmas], float)
    if a.size != sg.size:
        return np.full(sg.size, np.nan)
    return a / sg


def _correction_size(params, cfg, fit_settings):
    """RECORD: one fit's current corrections in units of their prior sigma -- per function and root-sum-square."""
    try:
        sig = _fit.correction_prior_sigmas(cfg, fit_settings)
        r = _prior_units(_cm.full_params(params)["a_corr"], sig)
        return dict(ratio_to_prior_sigma=[float(v) for v in r], rss_prior_sigma=float(np.sqrt(np.sum(r ** 2))),
                    prior_sigma=[None if v is None else float(v) for v in sig],
                    centers_rho=[float(c) for c in cfg.corr_centers_rho])
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _accumulated_record(acc, n_absorbed, s, sigmas):
    """RECORD: the accumulated absorbed filter correction per function [prior sigmas of the filter fit] since the first
    slice (its own corrections excluded) and the reader's flag at :data:`ACCUMULATED_CORRECTION_REPORT_LEVEL`."""
    lvl = ACCUMULATED_CORRECTION_REPORT_LEVEL
    acc = np.asarray(acc, float)
    flags = [f"accumulated correction of function {k} is {v:+.2f} prior sigmas (reporting level {lvl:g})"
             for k, v in enumerate(acc) if abs(v) > lvl]
    meaning = {"in_window": "sum of the absorbed source amplitudes (each acted over its own interval only)",
               "steady_response": "sum of the absorbed steady responses added to the carried state",
               "carried_outside_core": ("sum of the absorbed amplitudes (their outer parts were added to the carried "
                                        "state, their core parts acted over their own interval only)")}
    return dict(per_function_prior_sigma=[float(v) for v in acc], n_absorbed=int(n_absorbed),
                prior_sigma=[None if v is None else float(v) for v in sigmas], report_level_prior_sigma=lvl,
                flagged=bool(flags), flags=flags, form=s.filter_correction,
                meaning=meaning.get(s.filter_correction),
                note=("a record for the reader: the reporting level is not an acceptance criterion and steers nothing; "
                      "the first slice's corrections start the series inside its state and are not counted"))


_mse_by_radius = _fit.mse_by_radius


def _steady_state_check(fm, sl, cfg, s, params):
    """RECORD (``steady_state_check``, on by default; steers nothing): :func:`.fit.steady_state_check` of the slice
    with the series' first-slice core treatment, the slice's own inputs and Ip target.  Never raises."""
    try:
        ms = sl.measurements
        if ms is None or "mse" not in ms.groups:
            return dict(computed=False, reason="no MSE group at this slice")
        if s.sawtooth not in ("q_floor", "off"):
            return dict(computed=False, reason=f"not built for sawtooth={s.sawtooth!r}")
        cfg_ss = dataclasses.replace(cfg, core_relaxation=FIRST_SLICE_CORE[s.sawtooth])
        return _fit.steady_state_check(fm, sl.kinetics, sl.sources, cfg_ss, params, ms, f"{sl.label}:steady_check")
    except Exception as exc:  # noqa: BLE001 -- a record must never end a series
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _reference_points_kw(sl):
    """``reference_points`` of a slice (``sl.meta["reference_points"]``) for :meth:`.forward.ForwardModel.reconfigure`
    (records only); empty when the slice gives none."""
    rp = (sl.meta or {}).get("reference_points")
    return {} if rp is None else dict(reference_points=rp)


#: Boundary-displacement quantities summarized over a series (mm; :meth:`.forward.ForwardModel.boundary_displacement`).
BOUNDARY_SUMMARY_KEYS = ("rms", "max", "mean", "outboard_midplane", "inboard_midplane")


def boundary_displacement_history(slice_records):
    """Per-slice history of the boundary displacement from the anchor boundary (mm) and its summary statistics over
    the slices that have one (median, mean, max |.|, n; also for each X-point and strike point by name)."""
    hist = []
    for r in slice_records:
        d = (r.get("boundary") or {}).get("displacement_mm") if isinstance(r, dict) else None
        if not d or "error" in d:
            continue
        h = dict(time=r.get("time"), **{k: d.get(k) for k in BOUNDARY_SUMMARY_KEYS})
        for x in d.get("x_points") or []:
            h[f"x_point_{x['position']}"] = x.get("distance")
        for name, v in (d.get("strike_points") or {}).items():
            h[f"strike_{name}"] = v.get("distance")
        hist.append(h)
    keys = sorted({k for h in hist for k in h if k != "time"})
    summ = {}
    for k in keys:
        v = np.array([h[k] for h in hist if h.get(k) is not None and np.isfinite(h[k])], float)
        if v.size:
            summ[k] = dict(median=float(np.median(v)), mean=float(np.mean(v)), max_abs=float(np.max(np.abs(v))),
                           n=int(v.size))
    return dict(units="mm", history=hist, summary=summ,
                note="displacement of the achieved boundary from the anchor (reference) boundary per consistent slice")


def _given_state(lag, s, cfg, ip):
    """State of a supplied <J.B>(rho_tor,norm) on the current pass's geometry (the profile is held, the
    state follows the geometry; the profile is evaluated inside its table only, edge-clipped)."""
    if s.initial_profile is None:
        raise ValueError("initial='from_profile' needs initial_profile = (rho_tor_norm, <J.B>)")
    from scipy.interpolate import PchipInterpolator
    r, j = (np.asarray(v, float) for v in s.initial_profile)
    rq = np.clip(lag["phys"]["rho"], r[0], r[-1])
    jb = PchipInterpolator(r, j)(rq)
    st = _cm.state_from_jdotb(lag, jb, ip, cfg.n_cells)
    if s.initial_profile_domain_fraction is None:
        return st
    scale = float(s.initial_profile_domain_fraction) * abs(float(ip)) / st.meta["enclosed_current_domain_edge"]
    st = _cm.state_from_jdotb(lag, scale * jb, ip, cfg.n_cells)
    f = float(s.initial_profile_domain_fraction)
    st.meta.update(profile_scale=scale, domain_fraction=f, domain_fraction_excess=max(0.0, f - 1.0),
                   domain_fraction_tolerance=float(s.initial_profile_domain_fraction_tolerance))
    return st
