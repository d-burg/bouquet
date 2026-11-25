r"""The kinetic reconstruction: a MAP fit over a small set of physical parameters.

Model: :mod:`.current_model` (ohmic + bootstrap + driven current with the time-averaged q-floor core state by
default, thermal + fast-ion pressure) iterated to consistency with the anchored forward solve
(:mod:`.forward`).  Data: the residuals of :func:`.score.score` for the fit groups

* ``flux_loop`` and ``bpol_probe``: every channel the measurement set marks valid (for the EFIT
  k-file route: the EFIT-weighted channels), scored in the measurement's own form (flux loops
  relative to the reference loop when the set says so; probes length-averaged when lengths exist);
* ``pf_current``: only coils whose measurement is a constraint (role ``"fitted"``; held coils are
  exact by construction, free coils carry no measurement constraint -- EFIT semantics);
* ``ip``;
* ``mse``: only when the group is declared E_r-corrected (``meta['er_corrected']``) -- MSE whose
  E_r status is not established is not a fit constraint;

plus one prior row ``(theta_j - mu_j)/sigma_j`` per free parameter with a Gaussian prior, and -- when the
flux loops are scored with a fitted additive offset that carries a reference prior row
(:func:`.score.score`, ``flux_offset_prior``) -- that prior row ``(c - c0)/s0`` as one more datum, with the
offset counted as one more parameter in the degrees of freedom (the offset is solved in closed form inside
the score, so the objective is the full chi^2 minimized over it, the same footing as
:func:`.score.chi2_report`).  Thomson is never a weight.  The objective is ``chi^2 = sum r^2`` (diagonal Gaussian
likelihood + Gaussian priors); the MAP point is found by Levenberg-Marquardt with a forward-difference Jacobian.

Parameters (:data:`PARAMETERS`; default free set: the ``a_corr`` current corrections and the pressure factor
``s_press``; ``log_strength`` and ``f`` are parameters of the opt-in sawtooth core treatments only, ``CurrentModelSettings.core_relaxation`` =
``"periodic_orbit"`` or ``"crash_times"``):

=================  ==========================================  ======================================
name               meaning                                     default prior / bounds
=================  ==========================================  ======================================
``log_strength``   ln of the sawtooth strength                 N(ln Lambda_0, 1.0); Lambda_0 at f = 0.37
                   ``Lambda = -ln(1 - 0.999 f) nu`` [1/s]      and a 60 ms period; bounds ln(1e-2)..ln(1e4)
``f``              reconnection fraction at fixed strength     no Gaussian prior (uniform 0.1-1)
``s_pfast``        fast-ion pressure scale                     N(1, 0.20); bounds 0.2..3
``s_cd``           driven-current scale                        N(1, 0.25); bounds 0..3
``s_temp``         temperature factor: T_e and T_i both        N(1, 0.10); bounds 0.5..2.  Not free by
                   multiplied by it; thermal pressure,         default.  The fast-ion sources are the
                   bootstrap and conductivity follow           caller's and do not follow it.
``s_press``        pressure factor: the pressure of the force  N(1, 0.10); bounds 0.5..2.  FREE by default
                   balance = s_press x the kinetic pressure;   in a single-slice fit.  The kinetic profiles,
                   recorded with every result                  bootstrap and conductivity do NOT follow it.
``s_bs``           bootstrap multiplier                        N(1, 0.5); bounds 0..3
``a_corr_k``       current-correction amplitude k (fraction    N(0, 0.02) (fraction of Ip); +-0.2
                   of Ip; Gaussians on rho_tor,norm,
                   ``current_model.correction_shapes``)
``ip_rel``         plasma-current target relative to the       no prior row (its datum is the Ip
                   measured Ip: target = Ip_meas (1 + ip_rel)  measurement row); +-0.2
=================  ==========================================  ======================================

The plasma current is fitted, not pinned (``FitSettings.ip_target = "fitted"``, default): whenever the
Ip measurement row is one of the fit groups, ``ip_rel`` joins the parameters (also when it is not named
in ``free``), so the forward solve's Ip target is a fit parameter whose datum is the Ip row with its own
sigma -- one datum, one parameter, no prior row.  This is the least invasive form for the solver, which
takes Ip as a target and renormalizes the current profile to it; on the time-dependent path the supplied
state's current density is scaled by the same factor.  ``ip_target = "measured"`` holds the target at
the measured value (the earlier behavior: the Ip noise then enters as an unmodeled input error).  Every
result records the measured Ip, the fitted Ip (the solved equilibrium's) and ``(measured -
fitted)/sigma`` (``record["ip_fit"]``).

Every result records its core treatment (``record["core"]``, :func:`.current_model.core_record`: the
``core_relaxation``, ``q_floor`` and its form, whether the floor acted, the floor radius and the floor crossings in
rho_tor,norm) and, as a record for the reader with no threshold, whether the data pull the core below the floor
(``record["core_vs_floor"]``, :func:`.current_model.core_vs_floor_record`: the fitted model q at the axis and its
minimum against the floor, and the signed change the corrections made to the core current and the axis q) and what the
floor did (``record["floor_effect"]``, :func:`.current_model.floor_effect_record`: the achieved q0 and q_min of the
solved equilibrium, the flattening region, and the change of the loop voltage and of l_i and the fraction of Ip moved
outside the region against the same model without the floor); nothing reads them.

Every result also records what its current corrections did to the core (``record["correction_core_effect"]``,
:func:`.current_model.correction_core_effect`; a record only): each amplitude against its prior sigma, and the
change of the current inside rho_tor,norm 0.3, of the axis current density and of q at the axis and at rho 0.2
between the final center and the same center with the amplitudes set to zero (the current model re-evaluated on the
center's geometry).  Where the corrections' current returns in the steady form is a model setting
(:attr:`.current_model.CurrentModelSettings.corr_return`).

Every result with a final center also records, as records that steer nothing, the solver's vertical-stability coil
current and each of its coils' own current and total (``record["coil_currents"]``,
:meth:`.forward.ForwardModel.coil_current_record`), and how far the fitted surfaces sit from those of the equilibrium
that produced the kinetic profiles in outboard-midplane major radius (``record["radial_coordinate"]``,
:meth:`.forward.ForwardModel.midplane_radius_record`; "not supplied" unless the kinetic inputs carry that
equilibrium's table).

A single-slice fit (no supplied state) also records the steady-state input check (``record["steady_state_check"]``,
:func:`steady_state_check`; ON by default, ``reconstruct(..., steady_state_check=False)`` switches it off): after the
fit, the slice's unfitted model steady state (the fit's starting parameters with the corrections zero, the fit's own
core treatment) is solved and scored on the MSE chords by radius next to the fit's own result, and the solver is
restored to the state the fit left.  A RECORD that steers nothing: the fit and its own counts (``n_solves``, solver
iterations) are complete before it runs; its solves are counted on the forward model and stated in its record.

Every prior, bound, step rule and stopping criterion is an explicit setting recorded in the
result (they are decisions for the user; the defaults are this package's proposals).

The strength (opt-in sawtooth treatments only; no default path uses it) is defined with the cap of
:mod:`.current.sawtooth`: ``Lambda = -ln(1 - 0.999 f) nu``.  Its prior center, used only when the strength is a
free parameter of an opt-in treatment, is the strength of f = 0.37 and a 60 ms period, so at fixed f it implies
that period.  A prior that implies no period is available as an option,
:data:`LOG_STRENGTH_PRIOR_BOUNDS_BROAD` (``FitSettings(priors={"log_strength":
LOG_STRENGTH_PRIOR_BOUNDS_BROAD})``): a Gaussian in ``ln Lambda`` derived from the parameter's
bounds alone -- centered on the middle of ``[ln 1e-2, ln 1e4]`` with the bounds at +-2 sigma -- which
keeps the normal equations regular when the strength has no effect (q >= 1 everywhere) while
carrying no information beyond the bounds.  It is not the default.

Algorithm (Levenberg-Marquardt, H. B. Nielsen's damping update with the diagonal scaling of
D. W. Marquardt, J. SIAM 11, 431 (1963)):

1. center = consistent solve at the initial parameters (from the supplied starting state); the
   channel set of the fit is fixed there (a later state that cannot model one of these channels is
   a failed trial, never a silently smaller chi^2);
2. Jacobian columns by forward differences, each a FULL CONSISTENT solve warm from the center (no
   lagged inputs); the step of parameter j starts at ``step0_j`` and is enlarged x``step_growth``
   (up to ``max_step_growths`` times) until ``|dr| >= step_noise_factor * |noise|``, where ``noise``
   is the residual change of a consistent re-solve of the first center's parameters from another
   state (measured once, with its chi^2 noise ``noise_chi2``); a column still below that is flagged
   "no information"; a column whose solve fails is retried once with the opposite step
   (``column_retry_opposite``), otherwise the fit stops as ``jacobian_failed``;
3. stopping tests at the center, with that fresh Jacobian (:func:`stopping_tests`): converged when
   (a) the UNDAMPED Gauss-Newton step is below ``step_tol_sigma`` posterior sigmas in every
   parameter, or (b) the Newton decrement ``g^T A^-1 g`` (the predicted chi^2 decrease of that step)
   is below ``max(grad_tol_chi2, noise_factor * noise_chi2)``, or (c) the last ``chi2_rel_window``
   accepted steps each decreased chi^2 by less than ``chi2_rel_tol`` (relative) while the GN step is
   below 1 sigma.  A single damped step is never convergence: the tests are always made at a new
   center with a new Jacobian.  When a test holds, the undamped Gauss-Newton step it computed is
   taken as a last trial (``final_gn_step``) and kept if chi^2 does not rise by more than the noise;
4. LM step ``(J^T J + lambda diag(J^T J)) delta = -J^T r`` shrunk as a whole to the trust caps
   ``max_step`` and clipped to the bounds (each correction amplitude by default at most 0.04 of Ip per
   step; a step rule only, no stopping test reads it); when the cap binds and the step does not change with
   lambda, lambda is raised without a solve; the trial is a FULL consistent solve warm from the
   center; a failed trial (any solve not converged or not verified, a model failure, a q-consistency
   failure, a changed channel set) is a rejected step (lambda x ``lm_up``);
5. a trial is accepted only if it lowers chi^2 by more than ``noise_factor * noise_chi2``; when no
   trial does up to ``lm_lambda_max`` the fit stops as ``stalled_no_significant_descent`` (NOT
   converged; the stopping-test values are recorded);
6. final: the accepted center is consistent and verified by construction; the Laplace covariance
   ``(J_a^T J_a)^-1`` (data and prior rows) uses a fresh consistent-solve Jacobian at that center.

Lagged step columns (``jacobian_steps="lagged"``, an option; default ``"consistent"``).  The
optimizer's STEPS may use lagged columns: each a single forward solve with the geometry-dependent
inputs (geometry, conductivity, bootstrap, the supplied state of the time-dependent path) held at
the center's own values (:func:`.current_model.lagged_solve` on the geometry the center's assembly
used), differenced against a lagged solve at the center's parameters; the step-size rule and the
opposite-step retry are those of the consistent columns.  The definition of convergence is
unchanged: the stopping tests are ALWAYS made on consistent-solve columns.  With lagged steps, the
tests on the lagged columns only decide when the consistent columns are computed -- when they
hold, and when no lagged step descends (before a stall is declared) -- and a consistent-column
Jacobian computed for that purpose is then also used for that iteration's step.  The covariance is
always from consistent columns, also after ``failed_trials``.

Parallel columns (``reconstruct(..., speed=SpeedSettings(column_workers=N))``, an option; off by default; Linux
only).  The columns of every Jacobian are evaluated in ``N`` processes at once (:mod:`.speed`); the result is bit for
bit the serial one (same columns, same decisions in the same order, same records apart from wall times and the
Jacobian's ``parallel`` entry, which holds the speed settings, the time bound and every serial re-run).  A latency tool: one fit's wall time drops, its core-seconds rise.
A worker column that ends without a result, exceeds its time bound, or cannot be started is computed here serially.
The in-process speed options (the memo of pure evaluations, the indexed point location, reused statistics) are on
by default and change no value; ``speed=SpeedSettings.all_off()`` turns them off (:mod:`.speed`).

Statuses: ``converged``, ``converged_within_noise``, ``stalled_no_significant_descent``, ``max_iter``,
``failed_trials``, ``jacobian_failed``, ``failed_initial_consistency``, ``no_free_parameters`` (and
``<status>_no_final_jacobian`` when the covariance Jacobian failed).  ``converged_within_noise`` is a
fit that stopped through a stopping test like ``converged`` but whose stop rests on the measured
noise: test (b) held only through its noise-scaled floor (the Newton decrement is above
``grad_tol_chi2`` and at or below ``noise_factor * noise_chi2``), or the final Gauss-Newton step was
kept although it raised chi^2 (by less than the noise).  The stopping criteria are the same for both
labels; the label says which kind of stop it was (:func:`convergence_label`).  The number of forward
solves, wall time, every trial (predicted and realized chi^2 change, gain ratio) are recorded, and
``stopping_summary`` (:func:`stopping_summary`) states which stopping test ended the fit, the value
of test (b)'s floor and whether its noise-scaled branch set it, and whether the final Gauss-Newton
step was kept with a chi^2 increase (and by how much).

Edge taper: the taper is off unless the forward model's settings ask for it
(:class:`.edge.EdgeTaperSetting`; a workaround for solver builds without the cut-cell quadrature at
the boundary).  The whole-reconstruction tapered restart after a limit cycle in the first center is a
NON-DEFAULT option (``taper_restart=False`` by default) that acts only with the taper in the named
mode ``auto``; without it such a center is a failure.

Robustness options (NON-DEFAULT; off by default, and with them off the fit is the one described above; the
pass-loop option is :class:`.current_model.CurrentModelSettings`'s EXPERIMENTAL ``pass_relaxation``):

* ``noise_retries = 1`` (0 by default; 0 or 1): the noise re-solve of step 2 is a single consistent solve; when it
  fails (it did not converge or verify), the default fit stops as ``jacobian_failed``.  With the option it is
  repeated once, from the state of a consistent solve at the first column's parameter moved by
  ``NOISE_RETRY_STEPS[0]`` (-1) times that column's own signed step (the opposite step, as ``column_retry_opposite``
  does for a column); the solves and the verification criterion are the same.  A successful retry is a NEW noise
  draw from another starting state, not a repeat of the failed one (one draw of the noise can differ several-fold
  between starting states); ``record["noise"]["method"]`` then says so, and every attempt is recorded
  (``record["noise"]["retries"]``).  It acts only when the first measurement failed.
* ``stall_diagnostics = True``: a RECORD ONLY.  When a stall is about to be declared, it records the stopping tests
  on central-difference columns of that center (each forward column averaged with a backward column at the
  opposite step), the noise of a consistent re-solve of that center from the first backward column's state, and
  the stopping tests on the forward columns read with that noise.  The stall stays a stall; no value recorded
  here is read by the fit.  Cost: one consistent solve per parameter plus one noise re-solve, at a stall only.

``record["robustness"]`` (present only when an option is set) says which options were in effect and what they did.
"""

import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Sequence

import numpy as np

from . import current_model as _cm
from . import pressure as _pressure
from . import solver_rows as _rows
from . import speed as _speed
from .current import sawtooth as _saw
from .score import chi2_summary, flux_offset_record, score
from .sources.priors import DRIVEN_CURRENT_SCALE_SIGMA, FAST_ION_PRESSURE_SCALE_SIGMA

#: Reference crash period of the strength prior of the OPT-IN sawtooth treatments (proposal: median 60 ms) [s].
PRIOR_PERIOD_S = 0.060
#: Reference reconnection fraction (Bateman et al. 2006 hint) used by that prior center (opt-in treatments only).
PRIOR_F = 0.37


def _lam0():
    return float(math.log(_saw.crash_strength(PRIOR_F, 1.0 / PRIOR_PERIOD_S)))


#: Bounds of ``ln Lambda`` (the strength in 1/s).
LOG_STRENGTH_BOUNDS = (math.log(1e-2), math.log(1e4))
#: Option: strength prior derived from the bounds alone (module docstring): center at the middle of the
#: bounds in ln Lambda, sigma = a quarter of their width (bounds at +-2 sigma).
LOG_STRENGTH_PRIOR_BOUNDS_BROAD = (0.5 * (LOG_STRENGTH_BOUNDS[0] + LOG_STRENGTH_BOUNDS[1]),
                                   0.25 * (LOG_STRENGTH_BOUNDS[1] - LOG_STRENGTH_BOUNDS[0]))

#: Column kinds the optimizer's steps may use (:attr:`FitSettings.jacobian_steps`).
JACOBIAN_STEP_KINDS = ("consistent", "lagged")
#: Prior width of the temperature factor ``s_temp`` (relative; a stated value, of the size of the systematic
#: uncertainty of the temperature profiles).
TEMPERATURE_FACTOR_SIGMA = 0.10
#: Prior width of the pressure factor ``s_press`` (relative; a stated value, of the size of the combined systematic
#: uncertainty of the kinetic pressure: density and temperature calibration, dilution, fast ions).
#: PROVISIONAL, pending more data (as is the flag range, :data:`.current_model.PRESSURE_FACTOR_FLAG_RANGE`).
PRESSURE_FACTOR_SIGMA = 0.10


#: Parameter specifications: prior (mean, sigma; sigma None = no prior row), bounds, initial step.
PARAMETERS = {
    "log_strength": dict(prior=(_lam0(), 1.0), bounds=LOG_STRENGTH_BOUNDS, step0=0.1,
                         unit="ln(1/s)", note="sawtooth strength along -ln(1-0.999f) nu (MSE-constrained)"),
    "f": dict(prior=(None, None), bounds=(0.1, 1.0), step0=0.05, unit="1",
              note="reconnection fraction at fixed strength (uniform prior 0.1-1: bounds only)"),
    "s_pfast": dict(prior=(1.0, FAST_ION_PRESSURE_SCALE_SIGMA), bounds=(0.2, 3.0), step0=0.05, unit="1",
                    note="fast-ion pressure scale (prior width: sources.priors)"),
    "s_temp": dict(prior=(1.0, TEMPERATURE_FACTOR_SIGMA), bounds=(0.5, 2.0), step0=0.02, unit="1",
                   note="temperature factor on T_e and T_i (thermal pressure, bootstrap and conductivity follow; "
                        "the fast-ion sources do not); a measure of the mismatch between the kinetic profiles and "
                        "the magnetic measurements, to be reported with every result"),
    "s_press": dict(prior=(1.0, PRESSURE_FACTOR_SIGMA), bounds=(0.5, 2.0), step0=0.02, unit="1",
                    note="pressure factor: the pressure of the force balance = s_press x the kinetic pressure "
                         "(thermal + scaled fast-ion); the kinetic profiles, the bootstrap current and the "
                         "conductivity are NOT scaled, so the equilibrium pressure and the kinetic pressure differ "
                         "by this factor; free by default in a single-slice fit; recorded with every result "
                         "(pressure_factor); the prior width is provisional, pending more data"),
    "s_cd": dict(prior=(1.0, DRIVEN_CURRENT_SCALE_SIGMA), bounds=(0.0, 3.0), step0=0.05, unit="1",
                 note="driven-current scale (prior width: sources.priors)"),
    "s_bs": dict(prior=(1.0, 0.5), bounds=(0.0, 3.0), step0=0.05, unit="1", note="bootstrap multiplier"),
    "a_corr": dict(prior=(0.0, 0.02), bounds=(-0.2, 0.2), step0=0.005, unit="fraction of Ip",
                   note="model-discrepancy current amplitudes (one per center)"),
    "ip_rel": dict(prior=(None, None), bounds=(-0.2, 0.2), step0=0.002, unit="fraction of the measured Ip",
                   note="plasma-current target relative to the measured Ip; no prior row: the Ip measurement row "
                        "(its own sigma) is its datum"),
}

#: How the plasma-current target of the forward solve is treated by a fit (:attr:`FitSettings.ip_target`).
IP_TARGETS = ("fitted", "measured")

#: Starting state of the noise re-solve's retry (:attr:`FitSettings.noise_retries`, an option): the retry starts from
#: the state of a consistent solve at the first Jacobian column's parameter moved by this multiple of that column's
#: own signed step (the first try used +1; -1 is the opposite step, as ``column_retry_opposite`` takes for a column).
NOISE_RETRY_STEPS = (-1.0,)

#: Radial classes of the MSE chords in the ``steady_state_check`` record: edges of ``|R_chord - R_axis|`` [m] (core
#: below the first, mid between, outer above the second); a reporting convention of that record only.
MSE_RADIAL_CLASSES_M = (0.15, 0.35)
STEADY_STATE_CHECK_NOTE = ("a record for the reader: the slice's model steady state with no fit, scored on the same "
                           "chords as the slice's own result; nothing reads it")



@dataclass
class FitSettings:
    """Free parameters, priors and the algorithm's settings (all recorded; all decisions)."""
    # default: the current corrections and the pressure factor (the series' fits name their own free sets)
    free: Sequence[str] = ("a_corr", "s_press")
    priors: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (mean, sigma)
    bounds: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (lo, hi)
    max_iter: int = 12
    lm_lambda0: float = 1.0e-2
    lm_up: float = 10.0
    lm_down: float = 0.3
    lm_lambda_min: float = 1.0e-6
    lm_lambda_max: float = 1.0e6
    step_tol_sigma: float = 0.1          # (a) undamped GN step below this many posterior sigmas
    grad_tol_chi2: float = 1.0e-2        # (b) Newton decrement floor (chi^2 units)
    noise_factor: float = 1.0            # acceptance / (b): multiples of the measured chi^2 noise
    chi2_rel_tol: float = 1.0e-4         # (c) relative chi^2 decrease per accepted step
    chi2_rel_window: int = 2             # (c) number of consecutive accepted steps
    step_noise_factor: float = 20.0
    step_growth: float = 4.0
    max_step_growths: int = 3
    max_failed_trials: int = 4
    fit_groups: Sequence[str] = ("flux_loop", "bpol_probe", "pf_current", "ip", "mse")
    # trust caps of one LM step (step 4 of the algorithm); a_corr: fraction of Ip per correction amplitude per step
    max_step: Dict[str, float] = field(default_factory=lambda: dict(log_strength=0.5, f=0.1, s_pfast=0.1, s_cd=0.1,
                                                                    s_bs=0.2, a_corr=0.04, ip_rel=0.02,
                                                                    s_temp=0.05, s_press=0.05))
    fitted_coil_role: str = "fitted"
    column_retry_opposite: bool = True   # a failed column solve is retried once with the opposite step
    final_gn_step: bool = True           # take the undamped GN step of the converged test (accepted within noise)
    jacobian_steps: str = "consistent"   # "consistent" or "lagged": the columns used for the optimizer's STEPS
    #                                      (stopping tests and covariance always use consistent columns)
    taper_restart: bool = False          # NON-DEFAULT: restart the whole reconstruction tapered (module docstring)
    ip_target: str = "fitted"            # "fitted": ip_rel joins the parameters (Ip row its datum); "measured":
    #                                      the Ip target is held at the measured value (the earlier behavior)
    # NON-DEFAULT robustness options (module docstring, "Robustness options"):
    noise_retries: int = 0               # 1: a failed noise re-solve is repeated once from a stated other state
    stall_diagnostics: bool = False      # True: at a stall, RECORD central-column tests and the noise at the center
    # NON-DEFAULT moving anchor target (:func:`reconstruct`, "anchor_follow"; single-slice fits only):
    anchor_follow: bool = False          # True: after the fit, move the anchor target toward the solved boundary and
    #                                      refit, while the fitted groups' chi^2 falls (the weight never changes)
    anchor_follow_fraction: float = 0.5  # fraction of the way to the solved boundary per move
    anchor_follow_max_steps: int = 8     # cost bound: at most this many moves

    def __post_init__(self):
        if self.jacobian_steps not in JACOBIAN_STEP_KINDS:
            raise ValueError(f"jacobian_steps must be one of {JACOBIAN_STEP_KINDS}")
        if isinstance(self.noise_retries, bool) or int(self.noise_retries) != self.noise_retries \
                or not 0 <= self.noise_retries <= len(NOISE_RETRY_STEPS):
            raise ValueError(f"noise_retries must be a whole number from 0 to {len(NOISE_RETRY_STEPS)}")
        if not isinstance(self.stall_diagnostics, bool):
            raise ValueError("stall_diagnostics must be True or False")
        if self.ip_target not in IP_TARGETS:
            raise ValueError(f"ip_target must be one of {IP_TARGETS}")
        if not isinstance(self.anchor_follow, bool):
            raise ValueError("anchor_follow must be True or False")
        if not 0.0 < float(self.anchor_follow_fraction) <= 1.0:
            raise ValueError("anchor_follow_fraction must be in (0, 1]")
        if isinstance(self.anchor_follow_max_steps, bool) or int(self.anchor_follow_max_steps) != \
                self.anchor_follow_max_steps or self.anchor_follow_max_steps < 1:
            raise ValueError("anchor_follow_max_steps must be a whole number >= 1")
        if self.ip_target == "measured" and "ip_rel" in self.free:
            raise ValueError("ip_rel is free but ip_target='measured' holds the Ip target: choose one")

    def as_dict(self):
        return asdict(self)


def _expand(free, cfg):
    """Free parameter names expanded (``a_corr`` -> one entry per correction center)."""
    out = []
    for name in free:
        if name == "a_corr":
            out += [f"a_corr_{k}" for k in range(len(cfg.corr_centers_rho))]
        elif name in PARAMETERS:
            out.append(name)
        else:
            raise ValueError(f"unknown parameter {name!r}; known: {sorted(PARAMETERS)}")
    return out


def _spec(name, settings):
    base = PARAMETERS["a_corr" if name.startswith("a_corr_") else name]
    prior = settings.priors.get(name, base["prior"])
    bounds = settings.bounds.get(name, base["bounds"])
    return dict(prior=tuple(prior), bounds=tuple(bounds), step0=base["step0"], unit=base["unit"], note=base["note"])


def _to_params(theta, names, base):
    p = _cm.full_params(base)
    a = list(p["a_corr"])
    for n, v in zip(names, theta):
        if n.startswith("a_corr_"):
            a[int(n.split("_")[-1])] = float(v)
        else:
            p[n] = float(v)
    p["a_corr"] = tuple(a)
    return p


def _from_params(p, names):
    p = _cm.full_params(p)
    out = []
    for n in names:
        out.append(p["a_corr"][int(n.split("_")[-1])] if n.startswith("a_corr_") else p[n])
    return np.array(out, dtype=float)


# ============================================================== residuals
def fit_groups(ms, settings):
    """The groups that enter the fit and why others do not."""
    use, why = [], {}
    for g in settings.fit_groups:
        if g not in ms.groups:
            why[g] = "absent from the measurement set"
            continue
        if g == "mse" and not bool(ms.groups[g].meta.get("er_corrected", False)):
            why[g] = "MSE not declared E_r-corrected: not a fit constraint"
            continue
        use.append(g)
    return use, why


def residual_vector(fm, ms, groups, settings, coil_roles):
    """Normalized residuals of the fit groups on the solver's current state, plus the score table."""
    tab = score(fm.gs, ms, groups=groups)
    parts, index = [], []
    for g in groups:
        r = np.asarray(tab.residual(g), dtype=float)
        if g == "pf_current":
            grp = ms.groups[g]
            used = tab.used(g)
            names = [str(n) for n in grp.geometry.get("coil_names", grp.labels)]
            lab = [names[i] for i in np.nonzero(used)[0]]
            keep = np.array([coil_roles.get(n) == settings.fitted_coil_role for n in lab], dtype=bool)
            r = r[keep]
            lab = [lab_i for lab_i, k in zip(lab, keep) if k]
        else:
            grp = ms.groups[g]
            lab = [grp.labels[i] for i in np.nonzero(tab.used(g))[0]]
        parts.append(r)
        index += [(g, s) for s in lab]
    off = flux_offset_record(tab) if "flux_loop" in groups else None
    if off is not None and off.get("prior") is not None:
        # the fitted loop offset's reference prior row (module docstring): one more datum of the objective
        c0, s0 = (float(v) for v in off["prior"])
        parts.append(np.array([(float(off["offset"]) - c0) / s0]))
        index.append(("flux_offset_prior", "offset"))
    return (np.concatenate(parts) if parts else np.zeros(0)), index, tab


def _offset_params(tab, groups):
    """Number of fitted loop-offset parameters in the fit groups' score table (0 or 1; module docstring)."""
    off = flux_offset_record(tab) if "flux_loop" in groups else None
    return 0 if off is None else int(off["n_extra_params"])


def _prior_rows(theta, names, settings):
    rows, lab = [], []
    for n, v in zip(names, theta):
        mu, sig = _spec(n, settings)["prior"]
        if sig is not None:
            rows.append((v - mu) / sig)
            lab.append(("prior", n))
    return np.array(rows, dtype=float), lab


def _prior_jac(names, settings):
    J = []
    for j, n in enumerate(names):
        mu, sig = _spec(n, settings)["prior"]
        if sig is not None:
            row = np.zeros(len(names))
            row[j] = 1.0 / sig
            J.append(row)
    return np.array(J).reshape(-1, len(names))


def constraints_table(tab, ms, groups):
    """``equilibrium.time_slice[:].constraints``-compatible table of one score table: per group a list
    of ``{source, measured, measured_error_upper, reconstructed, weight, chi_squared}`` over the used
    channels (IMAS DD names; MSE in the group's own quantity)."""
    names = {"mse": "mse_polarisation_angle"}
    out = {}
    for g in groups:
        grp = ms.groups[g]
        use = tab.used(g)
        model = np.asarray(tab.model[g], dtype=float)
        rows = []
        for i in np.nonzero(use)[0]:
            r = grp.weight[i] * (model[i] - grp.measured[i]) / grp.sigma[i]
            rows.append(dict(source=str(grp.labels[i]), measured=float(grp.measured[i]),
                             measured_error_upper=float(grp.sigma[i]), reconstructed=float(model[i]),
                             weight=float(grp.weight[i]), chi_squared=float(r * r)))
        out[names.get(g, g)] = rows
    return out


# ============================================================== the fit
@dataclass
class FitResult:
    status: str
    params: Dict[str, Any]
    free: Sequence[str]
    theta: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    chi2: Dict[str, Any]
    record: Dict[str, Any]


def _consistent(fm, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
    return _cm.consistent_solve(fm, kin, src, params, cfg, start=start, z0=z0, label=label,
                                initial_profiles=initial_profiles, state_fn=state_fn)


def _same_index(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def lm_step(A, g, lam, cap):
    """Damped Gauss-Newton step ``(A + lam diag(A)) delta = -g`` shrunk as a whole to the trust caps.
    Returns ``(delta, capped)``."""
    Ad = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
    delta = -np.linalg.solve(Ad, g)
    scale = float(np.max(np.abs(delta) / cap)) if np.any(delta) else 0.0
    if scale > 1.0:
        return delta / scale, True
    return delta, False


def stopping_tests(A, g, settings, noise_chi2, history):
    """The stated stopping criteria at a center with a FRESH Jacobian (module docstring).

    ``A = J_a^T J_a`` (data and prior rows), ``g = J_a^T r``.  Returns a dict with the undamped
    Gauss-Newton step in posterior sigmas, the Newton decrement ``g^T A^-1 g`` (the predicted chi^2
    decrease of that step), the last relative chi^2 decreases, and which criterion (if any) holds."""
    try:
        Ainv = np.linalg.inv(A)
        dgn = -Ainv @ g
        post = np.sqrt(np.maximum(np.diag(Ainv), 0.0))
        step_sig = np.abs(dgn) / np.where(post > 0, post, np.inf)
        dec = float(g @ Ainv @ g)
    except np.linalg.LinAlgError:
        return dict(converged=False, reason="J^T J singular")
    grad_floor = max(settings.grad_tol_chi2, settings.noise_factor * noise_chi2)
    rel = [h for h in history[-settings.chi2_rel_window:]]
    out = dict(gn_step_sigma=step_sig.tolist(), newton_decrement=dec, grad_floor=grad_floor,
               grad_floor_fixed=float(settings.grad_tol_chi2), grad_floor_noise=float(settings.noise_factor * noise_chi2),
               grad_floor_noise_scaled=bool(settings.noise_factor * noise_chi2 > settings.grad_tol_chi2),
               last_rel_decreases=rel, noise_chi2=noise_chi2, test=None)
    if np.all(step_sig < settings.step_tol_sigma):
        out.update(converged=True, test="a",
                   criterion=f"undamped Gauss-Newton step < {settings.step_tol_sigma} posterior sigma in every parameter")
    elif dec <= grad_floor:
        out.update(converged=True, test="b",
                   criterion=f"Newton decrement g^T A^-1 g = {dec:.3g} <= {grad_floor:.3g} "
                             f"(max(grad_tol_chi2, noise_factor x chi^2 noise))")
    elif (len(rel) >= settings.chi2_rel_window and all(r < settings.chi2_rel_tol for r in rel)
          and np.all(step_sig < 1.0)):
        out.update(converged=True, test="c",
                   criterion=f"{settings.chi2_rel_window} accepted steps each decreased chi^2 by "
                             f"< {settings.chi2_rel_tol:g} (relative) and the GN step is < 1 sigma")
    else:
        out.update(converged=False)
    return out


def stopping_summary(status, stop, log):
    """Which stopping test ended a fit and how (recorded in every result as ``stopping_summary``).

    ``test``: ``"a"`` (undamped Gauss-Newton step), ``"b"`` (Newton decrement) or ``"c"`` (small
    relative decreases) when a test held, else ``None``; ``grad_floor`` of test (b) and whether its
    noise-scaled branch set it (``grad_floor_noise_scaled``: ``noise_factor x chi^2 noise`` above
    ``grad_tol_chi2``); ``final_gn_step``: whether the undamped Gauss-Newton step of the test was
    taken and kept, and the chi^2 change it made (``kept_with_chi2_increase`` and ``chi2_increase``
    when it was kept although chi^2 rose, within the measured noise).  Values only; no rule is
    applied here."""
    st = stop or {}
    fin = (log[-1].get("final_gn_step") if log else None)
    fgn = dict(taken=fin is not None)
    if fin is not None:
        dchi = fin.get("chi2_change")
        kept = bool(fin.get("accepted"))
        fgn.update(accepted=kept, status=fin.get("status"), chi2_change=dchi,
                   kept_with_chi2_increase=bool(kept and dchi is not None and dchi > 0),
                   chi2_increase=(float(dchi) if (kept and dchi is not None and dchi > 0) else 0.0))
    within = []
    if (st.get("converged") and st.get("test") == "b" and st.get("grad_floor_noise_scaled")
            and st.get("newton_decrement") is not None and st.get("grad_floor_fixed") is not None
            and st["newton_decrement"] > st["grad_floor_fixed"]):
        within.append("test (b) held only through its noise-scaled floor")
    if fgn.get("kept_with_chi2_increase"):
        within.append("final Gauss-Newton step kept with a chi^2 increase")
    return dict(status=status, within_noise_reasons=within, test=st.get("test") if st.get("converged") else None,
                criterion=st.get("criterion") if st.get("converged") else None,
                grad_floor=st.get("grad_floor"), grad_floor_fixed=st.get("grad_floor_fixed"),
                grad_floor_noise=st.get("grad_floor_noise"), grad_floor_noise_scaled=st.get("grad_floor_noise_scaled"),
                newton_decrement=st.get("newton_decrement"), gn_step_sigma=st.get("gn_step_sigma"),
                final_gn_step=fgn)


def convergence_label(stop, final_gn=None):
    """Status of a fit that stopped through a stopping test (module docstring): ``"converged"``, or
    ``"converged_within_noise"`` when the stop rests on the measured noise -- test (b) held only through
    its noise-scaled floor (decrement above ``grad_floor_fixed``), or the final Gauss-Newton step
    (``final_gn``: its record) was kept although chi^2 rose.  No criterion is applied here beyond the
    stopping test's own values."""
    st = stop or {}
    via_noise_floor = bool(st.get("converged") and st.get("test") == "b" and st.get("grad_floor_noise_scaled")
                           and st.get("newton_decrement") is not None and st.get("grad_floor_fixed") is not None
                           and st["newton_decrement"] > st["grad_floor_fixed"])
    fin = final_gn or {}
    kept_up = bool(fin.get("accepted") and fin.get("chi2_change") is not None and fin["chi2_change"] > 0)
    return "converged_within_noise" if (via_noise_floor or kept_up) else "converged"


@_rows.keep_constraints()        # the solver's constraint set kept between score calls (solver_rows; no value changes)
def reconstruct(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None, label="fit",
                state_fn=None, speed=None, steady_state_check=True):
    """MAP reconstruction (module docstring).

    ``fm``: :class:`.forward.ForwardModel` in reconstruction mode (with the measurement set);
    ``kin``/``src``: kinetic inputs (:class:`.kinetics.KineticInputs` or
    :class:`.current_model.KineticProfiles`) and :class:`.current_model.SourceProfiles`;
    ``cfg``: :class:`.current_model.CurrentModelSettings`; ``params0``: starting parameters
    (default: the prior means of the free parameters and :data:`current_model.DEFAULT_PARAMS`
    otherwise); ``start``: starting state (snapshot or psi on nodes); ``initial_profiles``:
    ``(x, jphi, p)`` for the first solve when ``start`` is only a guess; ``state_fn``: a supplied
    current state per pass (time-dependent path; the corrections then act on it); ``speed``:
    :class:`.speed.SpeedSettings` (changes no result; not given: ``SpeedSettings()``, the in-process options on and
    one process; parallel Jacobian columns are opt-in; ``SpeedSettings.all_off()`` turns every option off; set for
    the duration of the fit; recorded as ``speed`` when given without workers -- with workers, every Jacobian's
    ``parallel`` record holds it).  The options in effect are
    recorded as ``record["speed_in_effect"]`` in every fit record, also when ``speed`` is not given and when a fit
    ends before its first Jacobian (:func:`.speed.in_effect`).

    A branch log (:func:`.current_model.open_branch_log`) is open while the fit runs: the q floor's branch at every
    assembly is recorded (``record["floor_branch_record"]``, :func:`.current_model.branch_flip_record`), a record
    only; where a log was already open (a filter fit inside a series slice), this fit's entries are appended to it as
    well, marked ``nested_fit``.  With ``cfg.q_floor_branch_freeze="first_assembly"`` (a diagnostic option, off by
    default) the q floor's branch of the fit's first assembly is used for every later assembly of the fit.

    ``steady_state_check`` (default True; single-slice fits only, i.e. without ``state_fn``): the record
    ``record["steady_state_check"]`` (module docstring, :func:`steady_state_check`), made after the fit and its branch
    log are complete; it steers nothing.

    ``settings.anchor_follow`` (NON-DEFAULT; single-slice fits only): the moving anchor target.  The fit above first,
    with the reference boundary as the anchor target.  Then, at most ``anchor_follow_max_steps`` times: every point of
    the anchor target moves ``anchor_follow_fraction`` of the way to the solved boundary
    (:meth:`.forward.ForwardModel.move_anchor_target`; the anchor weight never changes) and the fit is run again from
    the converged state and parameters.  A move is KEPT only if the refit converges and the chi^2 of the fitted groups
    (``chi2["data"]``) is lower than before; at the first move that is not kept the target and the state are put back
    and the last kept fit is the result.  The reference boundary then acts as a starting guess and an implicit prior
    whose strength is the number of moves; every step is in ``record["anchor_follow"]``.  With ``state_fn`` (a filter
    fit of a series slice) the same rule applies to the filter fit; the series driver carries the moved target to the
    next slice (:mod:`.series`, "Moving anchor target").
    """
    st = settings or FitSettings()
    kw = dict(label=label, state_fn=state_fn, speed=speed)
    if not st.anchor_follow:
        return _reconstruct_checked(fm, kin, src, cfg, settings=settings, params0=params0, start=start,
                                    initial_profiles=initial_profiles, steady_state_check=steady_state_check, **kw)
    base = replace(st, anchor_follow=False)
    return _follow_anchor(fm, kin, src, cfg, base, st, params0, start, initial_profiles, label, speed,
                          steady_state_check, state_fn)


#: Fit statuses that count as converged for the moving anchor target.
CONVERGED_STATUSES = ("converged", "converged_within_noise")


def _follow_anchor(fm, kin, src, cfg, base, st, params0, start, initial_profiles, label, speed, check,
                   state_fn=None):
    """The moving anchor target of :func:`reconstruct` (``FitSettings.anchor_follow``)."""
    t0, n0 = time.perf_counter(), fm.n_solves
    box = {}
    fr = _reconstruct_checked(fm, kin, src, cfg, settings=base, params0=params0, start=start,
                              initial_profiles=initial_profiles, label=label, speed=speed, steady_state_check=False,
                              start_box_out=box, state_fn=state_fn)

    def data_chi2(r):
        try:
            return float(r.chi2["data"])
        except Exception:  # noqa: BLE001
            return float("nan")

    steps = [dict(step=0, status=fr.status, chi2_data=data_chi2(fr), kept=True)]
    rec = dict(applied=True, fraction=float(st.anchor_follow_fraction), max_steps=int(st.anchor_follow_max_steps),
               rule="a move is kept only if the refit converges and the fitted groups' chi^2 (chi2['data']) is lower "
                    "than before; the anchor weight never changes; the reference boundary is the first target",
               steps=steps)
    stop = None
    for k in range(1, int(st.anchor_follow_max_steps) + 1):
        if fr.status not in CONVERGED_STATUSES:
            stop = f"the fit before move {k} is not converged ({fr.status})"
            break
        c_prev = fr.record.get("center") if isinstance(getattr(fr, "record", None), dict) else None
        if isinstance(c_prev, dict) and c_prev.get("snap") is not None:
            # the kept fit's accepted center is the state to move from and to put back (the solver may hold the state
            # of the fit's last solve, a column or a trial)
            fm.restore(c_prev["snap"])
            if c_prev.get("probe_correction") is not None:
                fm.probe_correction = np.array(c_prev["probe_correction"], float)
        keep_target, keep_state = fm.anchor_target(), fm.snapshot()
        keep_pc = np.array(getattr(fm, "probe_correction", np.zeros(0)), float)
        move = fm.move_anchor_target(st.anchor_follow_fraction)
        nbox = {}
        new = _reconstruct_checked(fm, kin, src, cfg, settings=base, params0=dict(fr.params), start=keep_state,
                                   label=f"{label}:anchor{k}", speed=speed, steady_state_check=False,
                                   start_box_out=nbox, state_fn=state_fn)
        c_old, c_new = data_chi2(fr), data_chi2(new)
        kept = bool(new.status in CONVERGED_STATUSES and c_new < c_old)
        steps.append(dict(step=k, status=new.status, chi2_data=c_new, kept=kept, move=move))
        if not kept:
            fm.set_anchor_target(keep_target)
            fm.restore(keep_state)
            fm.probe_correction = keep_pc
            stop = (f"move {k} not kept: refit {new.status}" if new.status not in CONVERGED_STATUSES else
                    f"move {k} not kept: chi^2 of the fitted groups did not fall ({c_old:.6g} -> {c_new:.6g})")
            break
        fr, box = new, nbox
    else:
        stop = f"the cap of {int(st.anchor_follow_max_steps)} moves"
    rec.update(stop=stop, n_moves_kept=sum(1 for x in steps[1:] if x["kept"]), n_solves_total=fm.n_solves - n0,
               wall_s_total=round(time.perf_counter() - t0, 2))
    if isinstance(getattr(fr, "record", None), dict):
        fr.record["anchor_follow"] = rec
        if state_fn is None:
            with _speed.speed_scope(_speed.effective(speed)):
                fr.record["steady_state_check"] = _fit_steady_state_check(fm, kin, src, cfg, fr, box, check, label)
    return fr


def _reconstruct_checked(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None,
                         label="fit", state_fn=None, speed=None, steady_state_check=True, start_box_out=None):
    """One fit of :func:`reconstruct` with its branch log and steady-state record (``start_box_out`` receives the
    starting parameters)."""
    token = blog = parent = None
    start_box = {} if start_box_out is None else start_box_out
    try:                          # the branch log: a record only, never raises
        parent = _cm.current_branch_log()
        blog, token = _cm.open_branch_log()
    except Exception:  # noqa: BLE001 -- a record must never end a fit
        pass
    # the frozen-branch diagnostic (option, off by default): the first assembly's branch for the whole fit
    ftok = _cm.open_branch_freeze()[1] if getattr(cfg, "q_floor_branch_freeze", "off") != "off" else None
    try:
        eff = _speed.effective(speed)                 # the default SpeedSettings() when none is given
        _speed.validate(eff, fm)
        with _speed.speed_scope(eff):                 # each in-process option set as in eff, also when off
            out = _reconstruct(fm, kin, src, cfg, settings=settings, params0=params0, start=start,
                               initial_profiles=initial_profiles, label=label, state_fn=state_fn, speed=speed,
                               start_box=start_box)
    finally:
        if ftok is not None:
            _cm.close_branch_freeze(ftok)
        if token is not None:
            try:
                _cm.close_branch_log(token)
                if parent is not None and blog is not None:
                    # a fit inside another log (a series slice's filter fit): its assemblies belong to that log too
                    parent.extend(dict(e, nested_fit=str(label)) for e in blog if isinstance(e, dict))
            except Exception:  # noqa: BLE001 -- a record must never end a fit
                pass
    if state_fn is None and isinstance(getattr(out, "record", None), dict):
        # the steady-state input check: a record made after the fit is complete (module docstring)
        with _speed.speed_scope(eff):
            out.record["steady_state_check"] = _fit_steady_state_check(fm, kin, src, cfg, out, start_box,
                                                                       steady_state_check, label)
    return out


def _fit_steady_state_check(fm, kin, src, cfg, fr, start_box, on, label):
    """``record["steady_state_check"]`` of a single-slice fit (:func:`steady_state_check`); never raises."""
    if not on:
        return dict(computed=False, reason="switched off (steady_state_check=False)")
    if fr.record.get("center") is None:
        return dict(computed=False, reason=f"no consistent final center (fit status {fr.status!r})")
    if cfg.core_relaxation in _cm.SAWTOOTH_RELAXATIONS or "params" not in start_box:
        return dict(computed=False, reason=f"not built for core_relaxation={cfg.core_relaxation!r}")
    return steady_state_check(fm, kin, src, cfg, start_box["params"], fm.ms, f"{label}:steady_check")


def _note_jacobian(jacobians, log, i0, c, lab):
    """Record (never raises) the floor branches of one Jacobian's assemblies (``log[i0:]``) and of its center ``c``."""
    try:
        if log is None or i0 is None:
            return
        center_branch = (((c.get("res").assembly or {}).get("core") or {}).get("floor_branch")
                         if c.get("res") is not None else None)
        br = sorted({str(e.get("branch")) for e in log[i0:]} | {str(center_branch)})
        jacobians.append(dict(label=lab, center_branch=center_branch, branches=br, n_assemblies=len(log) - i0,
                              mixed=len(br) > 1))
    except Exception:  # noqa: BLE001 -- a record must never end a fit
        pass


def _reconstruct(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None, label="fit",
                 state_fn=None, speed=None, start_box=None):
    """:func:`reconstruct` with its branch log open (``start_box`` receives the starting parameters)."""
    t_all = time.perf_counter()
    speed_given = speed
    speed = _speed.check_speed(speed, fm)
    branch_jacobians = []
    settings = settings or FitSettings()
    saw_free = [n for n in settings.free if n in ("log_strength", "f")]
    if saw_free and state_fn is None and cfg.core_relaxation not in _cm.SAWTOOTH_RELAXATIONS:
        raise ValueError(f"{saw_free} are parameters of the opt-in sawtooth core treatments "
                         f"{_cm.SAWTOOTH_RELAXATIONS}; core_relaxation is {cfg.core_relaxation!r}")
    names = _expand(settings.free, cfg)
    ms = fm.ms
    coil_roles = {n: v["role"] for n, v in fm.rows_record["coils"]["per_coil"].items()}
    groups, excluded = fit_groups(ms, settings)
    ip_note = _ip_parameter(names, groups, ms, settings)
    given = dict(params0 or {})
    base = _cm.full_params(given)
    th = np.zeros(len(names))
    for j, n in enumerate(names):
        mu = _spec(n, settings)["prior"][0]
        if n.startswith("a_corr_"):
            th[j] = base["a_corr"][int(n.split("_")[-1])] if "a_corr" in given else (mu or 0.0)
        elif n in given and given[n] is not None:
            th[j] = float(given[n])
        elif mu is not None:
            th[j] = mu
        elif base.get(n) is not None:
            th[j] = float(base[n])
        else:
            raise ValueError(f"no starting value for {n}")
    lo = np.array([_spec(n, settings)["bounds"][0] for n in names])
    hi = np.array([_spec(n, settings)["bounds"][1] for n in names])
    th = np.clip(th, lo, hi)
    if start_box is not None:
        start_box["params"] = _to_params(th, names, base)
    cap = np.array([settings.max_step.get("a_corr" if n.startswith("a_corr_") else n,
                                          FitSettings().max_step["a_corr" if n.startswith("a_corr_") else n])
                    for n in names])
    n0 = fm.n_solves
    i0 = len(getattr(fm, "iteration_log", None) or [])
    log = []
    rec = dict(label=label, free=names, specs={n: _spec(n, settings) for n in names}, settings=settings.as_dict(),
               model_settings=cfg.as_dict(), fit_groups=groups, excluded_groups=excluded, iterations=log,
               forward=None, kinetics_source=getattr(kin, "source", ""), sources_source=src.source,
               sources_convention=dict(src.convention),
               kinetics_provenance=dict(getattr(kin, "provenance", {}) or {}),
               kinetics_record=dict(getattr(kin, "record", {}) or {}), ip_target=ip_note,
               speed_in_effect=_speed.in_effect(speed_given))
    if speed_given is not None and not speed_given.parallel:
        # the speed settings when they were given without workers (the default record is unchanged; with workers every
        # Jacobian's ``parallel`` record holds them)
        rec["speed"] = dict(settings=speed_given.as_dict())

    def finish(status, center, J=None, extra=None, jinfo=None):
        rec["status"] = status
        rec["stopping_summary"] = stopping_summary(status, rec.get("final_stopping_tests") or (extra or {}).get(
            "final_stopping_tests") or (log[-1].get("stopping_tests") if log else None), log)
        rec["n_solves"] = fm.n_solves - n0
        rec["wall_s"] = round(time.perf_counter() - t_all, 2)
        rec["forward"] = fm.settings_record()
        get_its = getattr(fm, "iteration_record", None)
        if get_its is not None:
            try:      # every solve's iteration count and the limit in effect: a record only, nothing reads it
                rec["solver_iterations"] = get_its(start=i0)
            except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
                rec["solver_iterations"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        if extra:
            rec.update(extra)
        p_rlx = getattr(cfg, "pass_relaxation", None)
        if settings.noise_retries or settings.stall_diagnostics or p_rlx is not None:
            # NON-DEFAULT robustness options in effect and what they did (a record only; absent when all are off)
            sd = [e["stall_diagnostics"] for e in log if e.get("stall_diagnostics")]
            rec["robustness"] = dict(
                noise_retries=int(settings.noise_retries), stall_diagnostics=bool(settings.stall_diagnostics),
                pass_relaxation=p_rlx, pass_relaxation_note=(_cm.PASS_RELAXATION_NOTE if p_rlx is not None else None),
                noise_retried=bool((rec.get("noise") or {}).get("retries")),
                stall_diagnostics_made=[dict(
                    central_test=(d.get("stopping_tests_central") or {}).get("test"),
                    noise_chi2_at_center=(d.get("noise_at_center") or {}).get("chi2"),
                    test_with_noise_at_center=(d.get("stopping_tests_with_noise_at_center") or {}).get("test"))
                    for d in sd])
        rec["ip_fit"] = ip_record(center["all_tab"] if center is not None else None, ms,
                                  center["params"] if center is not None else _to_params(th, names, base),
                                  "ip_rel" in names)
        rec["coil_residuals"] = coil_residual_record(center["all_tab"] if center is not None else None, ms, coil_roles)
        rec["correction_core_effect"] = _correction_effect_record(center, kin, src, cfg, settings, state_fn)
        # the core treatment of the final center: core_relaxation, q_floor, whether the floor acted, the floor radius
        rec["core"] = ((center["res"].assembly or {}).get("core") if center is not None and center["res"].assembly
                       else dict(core_relaxation=cfg.core_relaxation, q_floor=float(cfg.q_floor),
                                 note="no consistent center"))
        try:      # whether the data pull the core below the floor: a record only, nothing reads it
            rec["core_vs_floor"] = _cm.core_vs_floor_record(rec["core"], rec["correction_core_effect"])
        except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
            rec["core_vs_floor"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        # what the q floor did (achieved q0 / q_min, region, loop voltage, l_i, current moved): a record only
        q_chk = q_eq = q_tr = q_s0 = None
        try:
            res_c = center["res"] if center is not None else None
            q_chk = (getattr(res_c, "record", None) or {}).get("q_check")
            lag_c = getattr(res_c, "lag", None)
            q_eq = None if lag_c is None else lag_c["phys"].get("q")
            q_tr = None if lag_c is None else lag_c["phys"].get("q_axis")
            q_s0 = (getattr(getattr(res_c, "solve", None), "stats", None) or {}).get("q_0")
        except Exception:  # noqa: BLE001 -- the achieved q is then not recorded
            pass
        try:
            rec["floor_effect"] = _cm.floor_effect_record(rec["core"], q_chk, q_eq, q_tr, q_s0)
        except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
            rec["floor_effect"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        # whether the floor's branch changed between the assemblies of this fit: a record only, nothing reads it
        try:
            rec["floor_branch_record"] = _cm.branch_flip_record(_cm.current_branch_log(), branch_jacobians,
                                                                rec["core"])
        except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
            rec["floor_branch_record"] = dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        cov = np.full((len(names), len(names)), np.nan)
        sig = np.full(len(names), np.nan)
        chi = {}
        if center is not None:
            chi = center["chi2"]
            if J is not None and len(names):
                Ja = np.vstack((J, _prior_jac(names, settings)))
                try:
                    cov = np.linalg.inv(Ja.T @ Ja)
                    sig = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    rec["covariance_note"] = "J^T J singular: covariance not available"
                if jinfo is not None:
                    info_flags = {}
                    for col in jinfo.get("columns", []):
                        if col.get("noise_limited"):
                            info_flags[col["name"]] = "no information: column at the noise floor after the step growths"
                    rec["covariance_column_flags"] = info_flags
        par = _to_params(center["theta"] if center is not None else th, names, base)
        if state_fn is not None or cfg.core_relaxation not in _cm.SAWTOOTH_RELAXATIONS:
            # the sawtooth parameters are not read on this path (the default core treatment or a supplied state):
            # recorded as not used rather than as their opt-in defaults
            par.update(log_strength=None, f=None)
            rec["params_not_used"] = ["log_strength", "f"]
        return FitResult(status=status, params=par,
                         free=names, theta=(center["theta"] if center is not None else th).copy(), sigma=sig,
                         covariance=cov, chi2=chi, record=rec)

    def evaluate(theta, start_state, z0, lab, init_prof=None, ref_index=None):
        p = _to_params(theta, names, base)
        res = _consistent(fm, kin, src, p, cfg, start_state, z0, lab, init_prof,
                          _scaled_state_fn(state_fn, p, getattr(fm, "ip_measured", None)))
        if speed is not None and speed.parallel:
            _speed.note_solve_wall(getattr(res, "wall_s", None))       # the time bound's scale (no value read)
        if not res.converged:
            return None, res
        r, idx, tab = residual_vector(fm, ms, groups, settings, coil_roles)
        if ref_index is not None and not _same_index(idx, ref_index):
            res = _cm.ConsistentResult(False, "channel_set_changed", res.passes, res.assembly, res.lag, res.solve,
                                       dict(res.record, reason="a fit channel became unmodelable (the channel set "
                                                               "is fixed at the first center)"), res.wall_s)
            return None, res
        rp, lp = _prior_rows(theta, names, settings)
        full = np.concatenate((r, rp))
        all_groups = [g for g in ms.groups if g in ("flux_loop", "bpol_probe", "ip", "pf_current", "mse")]
        all_tab = score(fm.gs, ms, groups=all_groups)
        summ = headline_summary(all_tab, ms, coil_roles)
        summ_offset = flux_offset_record(all_tab)
        fit_chi = {g: float(np.sum(r[[i for i, (gg, _) in enumerate(idx) if gg == g]] ** 2)) for g in groups}
        n_off = _offset_params(tab, groups)
        off_rows = [i for i, (gg, _) in enumerate(idx) if gg == "flux_offset_prior"]
        if off_rows:
            fit_chi["flux_offset_prior"] = float(np.sum(r[off_rows] ** 2))
        inner = _inner_dof_record(fm, groups)
        dof = float(r.size - len(names) - n_off - inner["data"])
        c = dict(theta=np.array(theta, float), params=p, res=res, r=full, r_data=r, index=idx, prior_index=lp,
                 chi2=dict(objective=float(full @ full), data=float(r @ r), prior=float(rp @ rp), fit_groups=fit_chi,
                           n_data=int(r.size), n_params=len(names) + n_off, n_params_outer=len(names),
                           n_params_flux_offset=n_off, n_data_flux_offset_prior=len(off_rows),
                           inner_dof=inner["data"], inner_dof_record=inner, dof=dof,
                           dof_outer_only=int(r.size) - len(names) - n_off,
                           chi2_per_dof=(float(r @ r) / dof if dof > 0 else float("nan")),
                           dof_rule="N_data (with the loop offset's prior row when present) - n_params (outer, with "
                                    "the fitted loop offset) - inner_dof (trace of the coil solve's hat matrix "
                                    "over the data rows of the fit groups)",
                           scored_all=summ,
                           scored_all_dof=dict(n_channels=summ["total"]["n"], n_params=len(names),
                                               inner_dof=_inner_dof_record(fm, all_groups)["data"],
                                               dof=summ["total"]["n"] - len(names)
                                               - _inner_dof_record(fm, all_groups)["data"],
                                               note="chi2/n in scored_all is descriptive (sigma are "
                                                    "conventions); the n_params were fitted to the fit "
                                                    "groups, a subset of these channels; free coils are "
                                                    "reported apart (pf_current_free) and not in the total"),
                           flux_offset=summ_offset),
                 tab=tab, all_tab=all_tab, snap=fm.snapshot(), probe_correction=fm.probe_correction.copy(),
                 z0=res.assembly["orbit_z"] if res.assembly else None)
        return c, res

    if isinstance(start, np.ndarray):
        fm.set_psi(start)
        start = None
    if start is not None:
        fm.restore(start)
    snap_init = fm.snapshot()
    pc_init = fm.probe_correction.copy()
    center, res0 = evaluate(th, None, None, f"{label}:center0", initial_profiles)
    rec["center0"] = dict(status=res0.status, passes=res0.passes, wall_s=res0.wall_s,
                          passes_record=res0.record.get("passes"), reason=res0.record.get("reason"),
                          q_check=res0.record.get("q_check"))
    if (center is None and res0.status == "limit_cycle_untapered_lock" and settings.taper_restart
            and fm.case.setting.mode.value == "auto" and not fm.taper_restarts):
        # NON-DEFAULT policy option (taper_restart=True): a limit cycle in an untapered solve of this
        # reconstruction before it had a consistent state restarts the WHOLE reconstruction tapered
        fm.lock_tapered(f"limit cycle in {res0.record.get('passes', [{}])[-1].get('k')} pass of {label}:center0 "
                        "(untapered); reconstruction restarted with the edge taper (taper_restart=True)")
        fm.restore(snap_init)
        fm.probe_correction = pc_init.copy()
        center, res0 = evaluate(th, None, None, f"{label}:center0:tapered", initial_profiles)
        rec["center0_restart_tapered"] = dict(status=res0.status, passes=res0.passes, wall_s=res0.wall_s,
                                              passes_record=res0.record.get("passes"),
                                              reason=res0.record.get("reason"))
    if center is None:
        return finish("failed_initial_consistency", None)
    ref_index = center["index"]

    def restore_center(c):
        fm.restore(c["snap"])
        fm.probe_correction = c["probe_correction"].copy()

    if not names:
        restore_center(center)
        extra = _final_extra(center, fm, ms, kin)
        out = finish("no_free_parameters", center, None, extra)
        out.record["center"] = center
        return out

    noise = dict(chi2=None, r=None, norm=None)

    def column_solve(c, theta, lab):
        """Consistent solve at theta warm from the center; returns the data residuals or None."""
        restore_center(c)
        trial, tres = evaluate(theta, c["snap"], c["z0"], lab, ref_index=ref_index)
        state_after = fm.snapshot()
        restore_center(c)
        if trial is None:
            return None, tres.status, None
        return trial["r_data"], "ok", (state_after, trial)

    def lagged_column_solve(c, theta, lab):
        """ONE forward solve at theta with the center's geometry-dependent inputs (and, on the
        time-dependent path, the state supplied on the center's geometry); data residuals or None."""
        restore_center(c)
        p = _to_params(theta, names, base)
        lag = c["res"].lag
        kw = {}
        if state_fn is not None:
            kw["state"] = _scaled_state_fn(state_fn, p, getattr(fm, "ip_measured", None))(lag)
        sres, _, why = _cm.lagged_solve(fm, lag, kin, src, p, cfg, z0=c["z0"], label=lab, **kw)
        if sres is None or not sres.converged:
            restore_center(c)
            return None, (why if sres is None else sres.status), None
        r, idx, _ = residual_vector(fm, ms, groups, settings, coil_roles)
        state_after = fm.snapshot()
        restore_center(c)
        if not _same_index(idx, ref_index):
            return None, "channel_set_changed", None
        return r, "ok", (state_after, None)

    def measure_noise(c, from_state, lab=None):
        """chi^2 and residual noise of a consistent re-solve: the center's parameters re-solved from a
        different (nearby) starting state."""
        trial, tres = evaluate(c["theta"], from_state, c["z0"], lab or f"{label}:noise", ref_index=ref_index)
        restore_center(c)
        if trial is None:
            return dict(status=tres.status)
        dr = trial["r_data"] - c["r_data"]
        return dict(status="ok", chi2=abs(trial["chi2"]["objective"] - c["chi2"]["objective"]), r=dr,
                    norm=float(np.linalg.norm(dr)))

    def noise_retries(c, j, hs, m_first):
        """NON-DEFAULT option (``noise_retries=1``): the failed noise re-solve is repeated once, from the state of a
        consistent solve at parameter ``j`` moved by ``NOISE_RETRY_STEPS[0] * hs`` (``hs`` = the first column's
        signed step), with the same solves and the same verification; a success is the noise measurement -- a new
        draw, recorded as such.  Every attempt is recorded under ``record["noise"]["retries"]``."""
        tries = []
        m = m_first
        for n, s in enumerate(NOISE_RETRY_STEPS[:int(settings.noise_retries)], 1):
            tr = c["theta"].copy()
            tr[j] += s * hs
            row = dict(retry=n, step_multiple=s, parameter=names[j], step=float(s * hs))
            if not lo[j] <= tr[j] <= hi[j]:
                tries.append(dict(row, status="starting state outside the parameter bounds: skipped"))
                continue
            _, why_r, es_r = column_solve(c, tr, f"{label}:noise_state_retry{n}")
            if es_r is None:
                tries.append(dict(row, status=f"starting-state solve failed: {why_r}"))
                continue
            m = measure_noise(c, es_r[0], f"{label}:noise_retry{n}")
            tries.append(dict(row, status=m["status"], chi2=m.get("chi2"), norm=m.get("norm")))
            if m["status"] == "ok":
                break
        first_method = rec["noise"].get("method")
        rec["noise"].update(status=m["status"], chi2=m.get("chi2"), norm=m.get("norm"))
        ok_try = next((t for t in tries if t.get("status") == "ok"), None)
        if ok_try is not None:
            rec["noise"]["method"] = (
                f"RETRY (noise_retries): consistent re-solve of the first center's parameters from the state of a "
                f"consistent solve at {ok_try['parameter']} moved by {ok_try['step_multiple']:g} x the first column's "
                f"signed step ({ok_try['step']:.6g}); a NEW noise draw from another starting state, not a repeat of "
                f"the failed one")
        rec["noise"]["retries"] = dict(
            first_status=m_first["status"], first_method=first_method, attempts=tries,
            setting=int(settings.noise_retries), measured_by_retry=ok_try is not None,
            rule="NON-DEFAULT option noise_retries: a failed noise re-solve repeated once from the state of a "
                 "consistent solve at the first column's parameter moved by NOISE_RETRY_STEPS x its signed step; same "
                 "solves, same verification criterion; the retry's measurement is a new draw")
        return m

    def jacobian(c, lab, lagged=False):
        log = i0 = None
        try:                      # the floor branches of this Jacobian's assemblies: a record only, never raises
            log = _cm.current_branch_log()
            i0 = len(log) if log is not None else None
        except Exception:  # noqa: BLE001 -- a record must never end a fit
            pass
        out = _jacobian(c, lab, lagged)
        try:
            _note_jacobian(branch_jacobians, log, i0, c, lab)
        except Exception:  # noqa: BLE001 -- a record must never end a fit
            pass
        return out

    def one_column(c, j, n, lab, lagged, solve_col, r_ref, J, cols, noise_wait=None, noise_sent=None):
        """The serial loop body of column ``j`` (step, opposite-step retry, step growth against the noise).  Writes
        ``J[:, j]``; returns ``(info, early)`` where ``early`` is the Jacobian's return value when the noise
        measurement failed.  ``noise_wait`` (a worker process, :mod:`.speed`) receives the noise measured in this
        process's first column instead of measuring it; ``noise_sent`` passes a newly measured noise on."""
        h = _spec(n, settings)["step0"]
        info = dict(name=n, attempts=[])
        for gi in range(settings.max_step_growths + 1):
            tj = c["theta"].copy()
            direction = 1.0 if tj[j] + h <= hi[j] else -1.0
            tj[j] += direction * h
            rj, why, extra_state = solve_col(c, tj, f"{lab}:col{j}")
            if rj is None and settings.column_retry_opposite and tj[j] - 2 * direction * h >= lo[j] \
                    and tj[j] - 2 * direction * h <= hi[j]:
                # the column solve failed: one retry with the opposite step (a backward difference); the
                # failure is recorded
                info["attempts"].append(dict(h=direction * h, status=why, retried_opposite=True))
                direction = -direction
                tj = c["theta"].copy()
                tj[j] += direction * h
                rj, why, extra_state = solve_col(c, tj, f"{lab}:col{j}:opp")
            if rj is None:
                info["attempts"].append(dict(h=direction * h, status=why))
                break
            if noise["norm"] is None and noise_wait is not None:
                noise.update(noise_wait())
            if noise["norm"] is None:
                # the same measurement in both column modes: the starting state is that of a CONSISTENT solve
                # at the first column's step (in lagged mode that solve is made for this purpose; a lagged
                # column's state would make the noise -- and every threshold built on it -- depend on the mode)
                from_state, why_n = (extra_state[0] if not lagged else None), None
                if lagged:
                    _, why_n, es_c = column_solve(c, tj, f"{lab}:noise_state")
                    from_state = None if es_c is None else es_c[0]
                m = (measure_noise(c, from_state) if from_state is not None
                     else dict(status=f"noise starting state failed: {why_n}"))
                rec["noise"] = dict(status=m["status"], chi2=m.get("chi2"), norm=m.get("norm"),
                                    method="consistent re-solve of the first center's parameters from the "
                                           "state of a consistent solve at its first Jacobian column's step"
                                           + (" (made for the noise measurement: the columns are lagged)"
                                              if lagged else ""))
                if m["status"] != "ok" and settings.noise_retries > 0:
                    m = noise_retries(c, j, direction * h, m)
                if m["status"] != "ok":
                    return info, (None, dict(reason=f"noise measurement failed: {m['status']}", columns=cols))
                noise.update(chi2=m["chi2"], r=m["r"], norm=m["norm"])
                if noise_sent is not None:
                    noise_sent(dict(noise))
            dr = rj - r_ref
            ok = np.linalg.norm(dr) >= settings.step_noise_factor * noise["norm"]
            info["attempts"].append(dict(h=direction * h, norm_dr=float(np.linalg.norm(dr)), ok=bool(ok)))
            J[:, j] = dr / (direction * h)
            if ok:
                break
            h *= settings.step_growth
        info["final_h"] = info["attempts"][-1].get("h")
        info["noise_limited"] = not info["attempts"][-1].get("ok", False)
        return info, None

    def _jacobian(c, lab, lagged=False):
        solve_col = lagged_column_solve if lagged else column_solve
        r_ref = c["r_data"]
        if lagged:
            # reference for the differences: the lagged solve at the center's own parameters
            r_ref, why, _ = lagged_column_solve(c, c["theta"].copy(), f"{lab}:lagref")
            if r_ref is None:
                return None, dict(reason=f"lagged reference solve failed: {why}", columns=[], column_kind="lagged")
        J = np.zeros((c["r_data"].size, len(names)))
        cols = []
        par_rec = None
        done = 0
        if speed is not None and speed.parallel and len(names) > 1:
            par_rec, done, early = _parallel_columns(c, lab, lagged, solve_col, r_ref, J, cols)
            if early is not None:
                return early
        for j, n in enumerate(names):
            if j < done:
                continue
            info, early = one_column(c, j, n, lab, lagged, solve_col, r_ref, J, cols)
            if early is not None:
                return early
            cols.append(info)
            if par_rec is not None:
                par_rec["columns_here"].append(j)
            if info["attempts"][-1].get("status"):
                return None, dict(reason=f"column {n} failed: {info['attempts'][-1]['status']}", columns=cols,
                                  column_kind="lagged" if lagged else "consistent")
        out = dict(noise_norm=noise["norm"], noise_chi2=noise["chi2"],
                   noise_source="consistent re-solve (measured once)", columns=cols,
                   column_kind=("lagged single solves (geometry-dependent inputs of the center)" if lagged
                                else "consistent solves (no lagged inputs)"))
        if par_rec is not None:
            out["parallel"] = par_rec
        return J, out

    def _parallel_columns(c, lab, lagged, solve_col, r_ref, J, cols):
        """The columns in contiguous blocks, the first (smallest) block here and the others in forked workers
        (:mod:`.speed`).  Fills ``J`` and ``cols`` in column order and returns ``(record, number of leading columns
        done, early return or None)``; the caller runs the remaining columns serially (all from the first column
        whose worker result cannot be used)."""
        t_par = time.perf_counter()
        log = _cm.current_branch_log()
        side = getattr(state_fn, "side_state", None)
        n_proc = min(int(speed.column_workers), len(names))
        sizes = [len(b) for b in np.array_split(np.arange(len(names)), n_proc)][::-1]   # the smallest block here
        starts = np.concatenate(([0], np.cumsum(sizes)))
        blocks = [list(range(int(starts[i]), int(starts[i + 1]))) for i in range(n_proc)]
        owner = {j: w for w, blk in enumerate(blocks[1:]) for j in blk}
        # the most consistent solves one column can make: every step growth with an opposite-step retry
        n_col = (settings.max_step_growths + 1) * (2 if settings.column_retry_opposite else 1)
        prec = dict(column_workers=int(speed.column_workers), workers=n_proc - 1, blocks=blocks, columns_here=[],
                    columns_from_workers=[], serial_reruns=[], speed=speed.as_dict(),
                    time_bound=dict(rule="max(column_time_floor_s, column_time_factor * n_col * the longest consistent "
                                         "solve made so far in this process and its workers)",
                                    n_col=int(n_col), n_col_rule="(max_step_growths + 1), times 2 with "
                                                                 "column_retry_opposite"),
                    time_bound_s={}, timeouts=[], worker_start_failure=None)
        workers = []

        def work(ch, cols_w):
            def wait():
                msg = ch.recv()
                if msg is None:                     # the parent no longer needs this column
                    os._exit(0)
                return msg
            Jw = np.zeros_like(J)
            for j in cols_w:
                mark = _speed.effects_mark(fm, log, side)
                info, _ = one_column(c, j, names[j], lab, lagged, solve_col, r_ref, Jw, [], noise_wait=wait)
                ch.send(("column", j, info, Jw[:, j].copy(), _speed.effects_since(mark, fm, log, side),
                         _speed.longest_solve()))
                if info["attempts"][-1].get("status"):
                    return                          # the serial loop stops at a failed column

        def broadcast(nz):
            for w in workers:
                w.send(dict(chi2=nz["chi2"], r=nz["r"], norm=nz["norm"]))

        def failed(j, info):
            return (None, dict(reason=f"column {names[j]} failed: {info['attempts'][-1]['status']}", columns=cols,
                               column_kind="lagged" if lagged else "consistent"))

        def rerun_from(j, reason):
            prec["serial_reruns"].append(dict(column=j, reason=reason))
            return prec, j, None

        try:
            try:
                for blk in blocks[1:]:
                    workers.append(_speed.ForkWorker(lambda ch, blk=blk: work(ch, blk),
                                                     oft_env=getattr(fm, "env", None)))
            except Exception as exc:  # noqa: BLE001 -- an interrupt (KeyboardInterrupt, SystemExit) is not caught
                # the workers cannot be made (fork, pipe or temporary directory refused: a process or memory limit;
                # or any other error while they are made): every column of this Jacobian is computed here serially
                for w in workers:
                    w.close()
                workers.clear()
                prec["worker_start_failure"] = f"{type(exc).__name__}: {exc}"[:600]
                return prec, 0, None
            for j in blocks[0]:                     # this process's block: the serial loop body
                info, early = one_column(c, j, names[j], lab, lagged, solve_col, r_ref, J, cols,
                                         noise_sent=broadcast if noise["norm"] is None else None)
                if early is not None:
                    return prec, j + 1, early
                cols.append(info)
                prec["columns_here"].append(j)
                if info["attempts"][-1].get("status"):
                    return prec, j + 1, failed(j, info)
            for j in range(len(blocks[0]), len(names)):
                # the time bound counts from now: the worker began this column no later than now (its earlier
                # column's result has been read, or this is its first column), so it is killed only after running
                # the column for at least the bound
                bound = _speed.column_time_bound(speed, n_col)
                prec["time_bound_s"][str(j)] = round(bound, 3)
                t_wait = time.perf_counter()
                msg = workers[owner[j]].recv(timeout=bound)
                if msg is _speed.TIMED_OUT:
                    # no result within the bound (a hung worker): killed by its process id, and this column and every
                    # later one are re-run here serially, as for a worker that died
                    workers[owner[j]].close()
                    prec["timeouts"].append(dict(column=j, bound_s=round(bound, 3),
                                                 waited_s=round(time.perf_counter() - t_wait, 3)))
                    return rerun_from(j, f"no result within the time bound ({bound:.1f} s)")
                if not (isinstance(msg, tuple) and msg[0] == "column" and msg[1] == j) or msg[4]["case_changed"]:
                    # a worker that raised, died or changed the edge-taper case: this column and every later one
                    # are re-run here serially (an exception or failure then surfaces as in the serial loop)
                    return rerun_from(j, ("edge-taper case changed" if isinstance(msg, tuple) and msg[0] == "column"
                                          else (msg[1][-600:] if isinstance(msg, tuple)
                                                else "worker ended without a result")))
                _, _, info, col, eff, t_long = msg
                _speed.note_solve_wall(t_long)
                _speed.apply_effects(eff, fm, log, side)
                J[:, j] = col
                cols.append(info)
                prec["columns_from_workers"].append(j)
                if info["attempts"][-1].get("status"):
                    return prec, j + 1, failed(j, info)
            return prec, len(names), None
        finally:
            for w in workers:
                w.close()
            prec["wall_s"] = round(time.perf_counter() - t_par, 3)

    def stall_diagnostics(c, Jf, jf_info, lab):
        """NON-DEFAULT diagnostic (``stall_diagnostics=True``), RECORD ONLY -- nothing here steers the fit, the stall
        stays a stall.  At the center ``c`` where no trial descended, with its consistent forward-difference Jacobian
        ``Jf``: (1) central-difference columns (each forward column averaged with a backward one, a consistent solve
        at the opposite of that column's own final signed step) and the stopping tests on them; (2) the noise of a
        consistent re-solve of this center from the first backward column's state, and the stopping tests on the
        forward columns read with that noise instead of the first measurement."""
        Jc = np.array(Jf, dtype=float, copy=True)
        cols, first_state, ok = [], None, True
        for j, n in enumerate(names):
            col = (jf_info.get("columns") or [{}] * len(names))[j]
            hs = col.get("final_h")
            row = dict(name=n, forward_step=hs)
            if hs is None or not lo[j] <= c["theta"][j] - hs <= hi[j]:
                cols.append(dict(row, central=False, reason="no forward step or opposite step outside the bounds"))
                continue
            tj = c["theta"].copy()
            tj[j] -= hs
            rj, why, es = column_solve(c, tj, f"{lab}:col{j}:back")
            if rj is None:
                cols.append(dict(row, central=False, reason=f"backward solve failed: {why}"))
                ok = False
                break
            if first_state is None:
                first_state = es[0]
            jb = (rj - c["r_data"]) / (-hs)
            Jc[:, j] = 0.5 * (Jf[:, j] + jb)
            nf = float(np.linalg.norm(Jf[:, j]))
            cols.append(dict(row, central=True, rel_change=(float(np.linalg.norm(Jc[:, j] - Jf[:, j])) / nf
                                                             if nf > 0 else None)))
        out = dict(note="record only (stall_diagnostics): nothing here steers the fit", central_columns=cols,
                   column_kind="central differences of consistent solves: the forward columns of this center averaged "
                               "with backward columns at the opposite step")
        if ok:
            A_c, g_c = normal_eq(Jc)
            out["stopping_tests_central"] = stopping_tests(A_c, g_c, settings, noise["chi2"] or 0.0, rel_hist)
        if first_state is not None:
            m = measure_noise(c, first_state, f"{lab}:noise_record")
            out["noise_at_center"] = dict(status=m["status"], chi2=m.get("chi2"), norm=m.get("norm"),
                                          chi2_first_measurement=noise["chi2"], norm_first_measurement=noise["norm"])
            if m["status"] == "ok":
                A_f, g_f = normal_eq(Jf)
                out["stopping_tests_with_noise_at_center"] = stopping_tests(A_f, g_f, settings, m["chi2"], rel_hist)
        return out

    lam = settings.lm_lambda0
    fails = 0
    J = None
    jinfo = None
    J_consistent = False
    status = "max_iter"
    rel_hist = []
    stop = None
    lagged_steps = settings.jacobian_steps == "lagged"

    def normal_eq(Jm):
        Ja = np.vstack((Jm, _prior_jac(names, settings)))
        return Ja.T @ Ja, Ja.T @ center["r"]

    for it in range(settings.max_iter):
        entry = dict(it=it, theta=center["theta"].tolist(), chi2=center["chi2"]["objective"])
        need_consistent = not lagged_steps
        if lagged_steps:
            stop = None                         # only consistent-column tests are recorded as stopping tests
            Jl, jl_info = jacobian(center, f"{label}:it{it}:lag", lagged=True)
            entry["jacobian_lagged"] = jl_info
            if Jl is None:
                need_consistent = True          # a failed lagged column: this iteration uses consistent columns
            else:
                A, g = normal_eq(Jl)
                stop_l = stopping_tests(A, g, settings, noise["chi2"] or 0.0, rel_hist)
                entry["stopping_tests_lagged"] = stop_l
                if stop_l.get("converged"):
                    need_consistent = True      # convergence is decided on consistent columns only
                else:
                    J, jinfo, J_consistent = Jl, jl_info, False
        converged_now = False
        if need_consistent:
            J, jinfo = jacobian(center, f"{label}:it{it}")
            J_consistent = True
            entry["jacobian"] = jinfo
            if J is None:
                entry["status"] = "jacobian_failed"
                log.append(entry)
                return finish("jacobian_failed", center, None)
            A, g = normal_eq(J)
            stop = stopping_tests(A, g, settings, noise["chi2"] or 0.0, rel_hist)
            entry["stopping_tests"] = stop
            converged_now = bool(stop.get("converged"))
        else:
            entry["jacobian"] = jinfo
        for attempt in range(2):
            if converged_now:
                break
            accepted = False
            last_delta = None
            while not accepted:
                try:
                    delta, capped = lm_step(A, g, lam, cap)
                except np.linalg.LinAlgError:
                    lam *= settings.lm_up
                    continue
                tt = np.clip(center["theta"] + delta, lo, hi)
                step = tt - center["theta"]
                if last_delta is not None and np.allclose(step, last_delta, rtol=1e-6, atol=1e-12):
                    # the trust cap binds and the step did not change: raise lambda without a solve
                    lam *= settings.lm_up
                    if lam > settings.lm_lambda_max:
                        break
                    continue
                last_delta = step
                pred = float(-(g @ step) * 2.0 - step @ A @ step)       # predicted chi^2 decrease (quadratic model)
                trial, tres = evaluate(tt, center["snap"], center["z0"], f"{label}:it{it}:trial",
                                       ref_index=ref_index)
                tr = dict(lam=lam, delta=step.tolist(), capped=bool(capped), predicted_dchi2=pred,
                          status=(tres.status if trial is None else "consistent"), passes=tres.passes,
                          chi2=(None if trial is None else trial["chi2"]["objective"]), wall_s=tres.wall_s,
                          columns="consistent" if J_consistent else "lagged")
                entry.setdefault("trials", []).append(tr)
                if trial is None:
                    fails += 1
                    lam *= settings.lm_up
                    restore_center(center)
                    if fails >= settings.max_failed_trials:
                        log.append(entry)
                        if not J_consistent:        # the covariance is always from consistent columns
                            J, jinfo = jacobian(center, f"{label}:it{it}:cov")
                        return finish("failed_trials", center, J, jinfo=jinfo)
                    continue
                dchi = center["chi2"]["objective"] - trial["chi2"]["objective"]
                tr["realized_dchi2"] = dchi
                tr["gain_ratio"] = dchi / pred if pred > 0 else None
                if dchi > settings.noise_factor * (noise["chi2"] or 0.0) and dchi > 0:
                    rel_hist.append(dchi / max(center["chi2"]["objective"], 1e-300))
                    center = trial
                    lam = max(lam * settings.lm_down, settings.lm_lambda_min)
                    accepted = True
                    entry.update(accepted=True, dchi2=dchi)
                else:
                    lam *= settings.lm_up
                    restore_center(center)
                    if lam > settings.lm_lambda_max:
                        break
            if not accepted and J_consistent and settings.stall_diagnostics and "stall_diagnostics" not in entry:
                # NON-DEFAULT diagnostic, record only: the stall is about to be declared; nothing here changes that
                entry["stall_diagnostics"] = stall_diagnostics(center, J, jinfo, f"{label}:it{it}:stall_diag")
            if accepted or J_consistent:
                break
            # no lagged step descended: before a stall is declared, the consistent columns decide (stopping
            # tests), and when they do not hold, the steps are retried with them
            J, jinfo = jacobian(center, f"{label}:it{it}:stall_check")
            J_consistent = True
            entry["jacobian_stall_check"] = jinfo
            if J is None:
                entry["status"] = "jacobian_failed"
                log.append(entry)
                return finish("jacobian_failed", center, None)
            A, g = normal_eq(J)
            stop = stopping_tests(A, g, settings, noise["chi2"] or 0.0, rel_hist)
            entry["stopping_tests"] = stop
            converged_now = bool(stop.get("converged"))
            lam = settings.lm_lambda0
        if converged_now:
            entry["stop_rule"] = stop["criterion"]
            # final undamped Gauss-Newton step (the stopping test computed it): taken when it does not
            # raise chi^2 by more than the measured noise; the covariance is then evaluated at the new center
            try:
                dgn = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                dgn = np.zeros_like(g)
            tt = np.clip(center["theta"] + dgn, lo, hi)
            if settings.final_gn_step and np.any(tt != center["theta"]):
                trial, tres = evaluate(tt, center["snap"], center["z0"], f"{label}:it{it}:final_gn",
                                       ref_index=ref_index)
                fin = dict(delta=(tt - center["theta"]).tolist(), status=(tres.status if trial is None else "consistent"),
                           chi2=None if trial is None else trial["chi2"]["objective"],
                           chi2_center=center["chi2"]["objective"],
                           chi2_change=(None if trial is None else
                                        trial["chi2"]["objective"] - center["chi2"]["objective"]))
                if trial is not None and trial["chi2"]["objective"] <= center["chi2"]["objective"] + \
                        settings.noise_factor * (noise["chi2"] or 0.0):
                    center = trial
                    fin["accepted"] = True
                else:
                    fin["accepted"] = False
                    restore_center(center)
                entry["final_gn_step"] = fin
            status = convergence_label(stop, entry.get("final_gn_step"))
            log.append(entry)
            break
        if not accepted:
            entry["note"] = (f"no trial decreased chi^2 by more than the measured noise up to lambda = "
                             f"{settings.lm_lambda_max:g}; stopping criteria NOT met")
            log.append(entry)
            status = "stalled_no_significant_descent"
            break
        log.append(entry)
    restore_center(center)
    # Laplace covariance from a FRESH Jacobian of consistent solves at the final, verified center (the
    # convergence test above already computed one at this center when it stopped there)
    if status in ("converged", "converged_within_noise") and J is not None and J_consistent and log \
            and log[-1].get("stopping_tests") is not None \
            and log[-1]["theta"] == center["theta"].tolist():
        Jf, jf_info = J, dict(jinfo, reused="Jacobian computed at the final center for the stopping test")
    else:
        Jf, jf_info = jacobian(center, f"{label}:final")
    restore_center(center)
    extra = _final_extra(center, fm, ms, kin)
    extra["final_jacobian"] = jf_info
    extra["final_stopping_tests"] = stop
    out = finish(status if Jf is not None else status + "_no_final_jacobian", center, Jf, extra, jinfo=jf_info)
    out.record["center"] = center
    return out


def _ip_parameter(names, groups, ms, settings):
    """Add ``ip_rel`` to the parameter list when the Ip target is fitted (``settings.ip_target``); returns the
    record of what was done.  The parameter has no prior row: its datum is the Ip measurement row of the fit
    groups (one datum, one parameter).  Without an Ip row in the fit groups the target stays at the measured
    value (recorded)."""
    if settings.ip_target == "measured":
        return dict(mode="measured", note="Ip target held at the measured value (ip_target='measured')")
    if "ip" not in groups:
        if "ip_rel" in names:
            raise ValueError("ip_rel is free but the fit groups have no Ip measurement row to constrain it")
        return dict(mode="measured", note="no Ip measurement row in the fit groups: the Ip target is held at the "
                                          "measured value")
    if "ip_rel" not in names:
        names.append("ip_rel")
    g = ms.groups["ip"]
    return dict(mode="fitted", parameter="ip_rel",
                rule="Ip target = measured Ip x (1 + ip_rel); ip_rel has no prior row, its datum is the Ip "
                     "measurement row with its own sigma; on the time-dependent path the supplied state's current "
                     "density is scaled by the same factor (the solver renormalizes to its Ip target alike)",
                sigma_model=g.meta.get("sigma_model_id", g.meta.get("sigma_model")))


def correction_prior_sigmas(cfg, settings):
    """The prior sigma of each current-correction amplitude under ``settings`` (None where there is no prior row)."""
    return [_spec(f"a_corr_{k}", settings)["prior"][1] for k in range(len(cfg.corr_centers_rho))]


def _correction_effect_record(center, kin, src, cfg, settings, state_fn):
    """``record["correction_core_effect"]`` (:func:`.current_model.correction_core_effect`; a record only): for a
    single-slice fit, the steady re-evaluation at the final center; on the time-dependent path (``state_fn``) the
    series records it per slice in the form the slice used, so the fit leaves it to the series."""
    if center is None or center["res"].assembly is None:
        return dict(computed=False, reason="no consistent center")
    if state_fn is not None:
        return dict(computed=False, reason="time-dependent path: recorded by the series in the slice record "
                                           "(filter.correction_core_effect)")
    try:
        return _cm.steady_correction_effect(center["res"].lag, center["res"].assembly, kin, src, cfg,
                                            correction_prior_sigmas(cfg, settings), z0=center["z0"])
    except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def _scaled_state_fn(state_fn, params, base_ip=None):
    """``state_fn`` with the supplied state scaled to the forward solve's Ip target ``|base_ip| (1 + ip_rel)``.
    The state's own current is its ``meta["ip_A"]`` (the Ip boundary value it was advanced to; set by the series
    driver), so a state that already carries the target is NOT rescaled; a state without that record is taken to
    carry ``base_ip``.  A state function that applies the current corrections over its interval (attribute
    ``corrections_in_window``, :func:`.current_model.call_state_fn`) receives ``params``.  Unchanged when there is no
    state function."""
    if state_fn is None:
        return None
    k = 1.0 + float(_cm.full_params(params).get("ip_rel", 0.0))

    def fn(lag, _params=None):
        st = _cm.call_state_fn(state_fn, lag, params)
        ip_s = (st.meta or {}).get("ip_A")
        f = (abs(float(base_ip)) * k / abs(float(ip_s))) if (ip_s and base_ip is not None) else k
        return st if f == 1.0 else _cm.scale_state(st, f, ip_A=None if ip_s is None or base_ip is None
                                                    else abs(float(base_ip)) * k)
    fn.corrections_in_window = bool(getattr(state_fn, "corrections_in_window", False))
    return fn


def _inner_dof_record(fm, groups):
    """The forward model's inner (coil) degrees of freedom on the data rows of ``groups`` (``fm.inner_dof``), or
    zero with the reason when the forward model does not provide them."""
    get = getattr(fm, "inner_dof", None)
    if get is None:
        return dict(data=0.0, note="forward model provides no inner degrees of freedom")
    d = get()
    if d is None:
        return dict(data=0.0, note="no data rows in the forward solve (generation mode)")
    return dict(data=float(fm.inner_dof_data(groups)), by_kind=d["by_kind"], n_rows=d["n_rows"],
                n_unknowns=d["n_unknowns"], rank=d["rank"], condition=d["condition"], method=d["method"])


def headline_summary(tab, ms, coil_roles):
    """Per-group ``{chi2, n, chi2_per_n}`` of a score table and their ``total``, with the coils the source left
    free (role ``"free"``: no measurement constraint in the solve) taken OUT of ``pf_current`` and of the total
    and reported apart as ``pf_current_free`` (with the coil names): their residuals are a cross-check, not part
    of the headline all-channels score."""
    out = {k: dict(chi2=v[0], n=v[1], chi2_per_n=v[2]) for k, v in chi2_summary(tab).items()}
    if "pf_current" not in tab.model or "pf_current" not in ms.groups:
        return out
    grp = ms.groups["pf_current"]
    used = tab.used("pf_current")
    names = [str(n) for n in grp.geometry.get("coil_names", grp.labels)]
    lab = [names[i] for i in np.nonzero(used)[0]]
    r = np.asarray(tab.residual("pf_current"), float)
    free = np.array([coil_roles.get(n, "free") == "free" for n in lab], dtype=bool)
    c_free, n_free = float(np.sum(r[free] ** 2)), int(free.sum())
    out["pf_current_free"] = dict(chi2=c_free, n=n_free, chi2_per_n=(c_free / n_free if n_free else float("nan")),
                                  coils=[n for n, f in zip(lab, free) if f],
                                  note="coils without a measurement constraint in the solve (free); not in the total")
    pc = out["pf_current"]
    c_pf, n_pf = pc["chi2"] - c_free, pc["n"] - n_free
    out["pf_current"] = dict(chi2=c_pf, n=n_pf, chi2_per_n=(c_pf / n_pf if n_pf else float("nan")))
    t = out["total"]
    c_t, n_t = t["chi2"] - c_free, t["n"] - n_free
    out["total"] = dict(chi2=c_t, n=n_t, chi2_per_n=(c_t / n_t if n_t else float("nan")))
    return out


#: REPORTING level of a fitted coil's current residual [sigma] (:func:`coil_residual_record`): a plain flag for the
#: reader where the magnitude of a coil's normalized residual reaches it.  Not an acceptance criterion; nothing reads
#: it and it steers nothing.
COIL_RESIDUAL_REPORT_LEVEL = 3.0


def coil_residual_record(tab, ms, coil_roles):
    """How far the solve left the coils that carry a measurement constraint from their measured currents.

    The coil currents are not parameters of the fit: the forward solve's least squares chooses them against the
    magnetics, the measured currents and the boundary anchor rows together (:mod:`.forward`), so a coil that
    ends well away from its measurement says those rows disagree.  Per coil with a role other than ``"free"``:
    the normalized residual ``w (model - measured) / sigma``; over them the rms, the largest magnitude with its
    coil, and the coils at or above :data:`COIL_RESIDUAL_REPORT_LEVEL`.  A record only: nothing reads it."""
    if tab is None or ms is None or "pf_current" not in getattr(tab, "model", {}) or "pf_current" not in ms.groups:
        return dict(available=False, note="no scored pf_current group")
    grp = ms.groups["pf_current"]
    used = tab.used("pf_current")
    names = [str(n) for n in grp.geometry.get("coil_names", grp.labels)]
    lab = [names[i] for i in np.nonzero(used)[0]]
    r = np.asarray(tab.residual("pf_current"), float)
    keep = np.array([coil_roles.get(n, "free") != "free" for n in lab], dtype=bool)
    lab, r = [n for n, k in zip(lab, keep) if k], r[keep]
    if not r.size:
        return dict(available=False, note="no coil with a measurement constraint in the solve")
    i = int(np.argmax(np.abs(r)))
    lvl = COIL_RESIDUAL_REPORT_LEVEL
    flags = [dict(coil=n, residual_sigma=float(v)) for n, v in zip(lab, r) if abs(v) >= lvl]
    return dict(available=True, n=int(r.size), rms_sigma=float(np.sqrt(np.mean(r ** 2))),
                max_abs_sigma=float(abs(r[i])), max_coil=lab[i], max_residual_sigma=float(r[i]),
                residual_sigma={n: float(v) for n, v in zip(lab, r)},
                report_level_sigma=lvl, flags=flags, flagged=bool(flags),
                sign="model - measured",
                note="a record for the reader: the reporting level is not an acceptance criterion and steers nothing")


def ip_record(tab, ms, params, fitted):
    """Measured, fitted (the solved equilibrium's Ip) and ``(measured - fitted)/sigma`` of the plasma current,
    with the target rule (``fitted``: ``ip_rel`` was a fit parameter)."""
    g = ms.groups.get("ip") if ms is not None else None
    if g is None:
        return dict(available=False, note="no Ip group in the measurement set")
    meas, sig = float(np.asarray(g.measured, float)[0]), float(np.asarray(g.sigma, float)[0])
    model = None
    if tab is not None and "ip" in getattr(tab, "model", {}):
        mv = np.asarray(tab.model["ip"], float)
        model = float(mv[0]) if mv.size and np.isfinite(mv[0]) else None
    rel = float(_cm.full_params(params).get("ip_rel", 0.0))
    return dict(available=True, measured_A=meas, sigma_A=sig, fitted_A=model, ip_rel=rel, ip_target_fitted=bool(fitted),
                pull_measured_minus_fitted_sigma=(None if model is None or not sig > 0 else (meas - model) / sig))


def _final_extra(center, fm, ms, kin=None):
    asm = center["res"].assembly
    return dict(final_center=dict(passes=center["res"].passes, passes_record=center["res"].record.get("passes"),
                                  stats=center["res"].solve.stats, solve_status=center["res"].solve.status,
                                  verification=center["res"].solve.record["policy"].get("verification"),
                                  q_check=center["res"].record.get("q_check")),
                assembly=dict({k: v for k, v in asm.items() if k not in ("orbit_z", "state")}),
                constraints=constraints_table(center["all_tab"], ms, list(center["all_tab"].model)),
                chi2_detail=center["chi2"], boundary=fm.boundary_distance(),
                coil_currents=_coil_currents(fm, center["res"].solve),
                radial_coordinate=(_rc := _radial_coordinate(fm, kin)),
                separatrix_te=separatrix_te_record(_rc, kin),
                pressure_frames=pressure_frames_record(center["res"].solve, fm),
                pressure_factor=asm.get("pressure_factor"),
                excluded_channels={k: v for k, v in center["all_tab"].excluded.items()},
                score_notes=list(center["all_tab"].notes))


def _coil_currents(fm, solve):
    """The forward model's coil-current record of the present state (:meth:`.forward.ForwardModel.
    coil_current_record`: the VSC current and each VSC coil's own current and total); a record only."""
    get = getattr(fm, "coil_current_record", None)
    if get is None:
        return dict(computed=False, reason="the forward model has no coil-current record")
    return get(solve)


def _radial_coordinate(fm, kin):
    """The radial-coordinate record of the present state (:meth:`.forward.ForwardModel.midplane_radius_record`)
    against the producing equilibrium's midplane table carried by the kinetic inputs, when supplied; a record only."""
    get = getattr(fm, "midplane_radius_record", None)
    if get is None:
        return dict(computed=False, reason="the forward model has no radial-coordinate record")
    return get(getattr(kin, "midplane_radius", None))


def pressure_frames_record(solve, fm):
    """RECORD that steers nothing: the stored energy, the betas and the axis pressure of a solved state in BOTH
    pressure frames (:func:`.pressure.pressure_frames`).

    The solver's own numbers (``stats``) count the pressure above its boundary value only; with the default
    ``separatrix_pressure="offset"`` the kinetic pressure at psi_N = 1 (``p_sep``) is added back in the full frame
    (W_MHD + 1.5 p_sep V).  A comparison with another code's stored energy, or with a diamagnetic measurement, must use
    the full frame.  Under ``"legacy"`` no offset was applied and both frames are the solver's.  ``computed`` is False,
    with the reason, when the solve carries no statistics or no boundary pressure."""
    stats = getattr(solve, "stats", None) or {}
    p_sep = (getattr(solve, "record", None) or {}).get("p_sep")
    mode = getattr(getattr(fm, "settings", None), "separatrix_pressure", None)
    if p_sep is None or "vol" not in stats or "W_MHD" not in stats:
        return dict(computed=False, reason="the solve carries no boundary pressure, volume or stored energy")
    applied = 0.0 if mode == "legacy" else float(p_sep)
    try:
        out = _pressure.pressure_frames(stats, applied)
    except ValueError as exc:
        return dict(computed=False, reason=str(exc)[:200])
    return dict(out, computed=True, separatrix_pressure=mode, p_sep_kinetic=float(p_sep))


def separatrix_te_record(radial, kin):
    """RECORD that steers nothing: the electron temperature of the kinetic profiles at the reconstruction's own
    separatrix IN REAL SPACE, on the outboard midplane.

    The kinetic profiles are applied on the reconstruction's flux coordinate as the producing equilibrium mapped them;
    where the two equilibria differ at the edge, the measured edge sits elsewhere in real space than the
    reconstruction places it.  From the radial-coordinate record (``radial``, :func:`_radial_coordinate`: the
    outboard-midplane major radius of our separatrix) and the producing equilibrium's midplane table carried by the
    kinetic inputs (major radius against its psi_N, with its rho_tor,norm when supplied): the table's coordinate at
    our separatrix radius, and T_e and n_e of the kinetic inputs there, beside their values at the table's own edge.
    Where our separatrix lies outside the table's last point the profile is not extrapolated and the record says so.
    Never raises."""
    try:
        tab = getattr(kin, "midplane_radius", None)
        if tab is None or not isinstance(radial, dict) or not radial.get("computed"):
            return dict(computed=False, reason="no midplane table or no radial-coordinate record")
        coord = getattr(kin, "coordinate", None)
        xt = tab.get("rho_tor_norm") if coord == "rho_tor_norm" else tab.get("psi_n")
        if xt is None:
            return dict(computed=False, reason=f"the midplane table carries no {coord} (kinetic inputs are on {coord})")
        tp, tr, xt = (np.asarray(v, float) for v in (tab["psi_n"], tab["r_m"], xt))
        r_sep = float(radial["edge_r_ours_m"])
        x_hi = float(np.asarray(kin.x, float)[-1])

        def at(x):
            x = float(min(max(x, float(kin.x[0])), x_hi))
            return float(kin._f("te")(x)), float(kin._f("ne")(x))

        te_e, ne_e = at(float(np.interp(min(1.0, tp[-1]), tp, xt)))
        out = dict(computed=True, r_separatrix_ours_m=r_sep, r_table_edge_m=float(np.interp(min(1.0, tp[-1]), tp, tr)),
                   coordinate=coord, te_table_edge_eV=te_e, ne_table_edge_m3=ne_e, source=tab.get("source"),
                   note="record only: T_e of the kinetic inputs where OUR separatrix sits on the producing "
                        "equilibrium's outboard midplane; nothing reads it")
        out["separatrix_minus_table_edge_mm"] = 1e3 * (r_sep - out["r_table_edge_m"])
        if r_sep > tr[-1]:
            out.update(inside_table=False, te_eV=None, ne_m3=None,
                       reason="our separatrix lies outside the table's last point: no extrapolation")
            return out
        x_at = float(np.interp(r_sep, tr, xt))
        te, ne = at(x_at)
        out.update(inside_table=True, psi_n_table_at_our_separatrix=float(np.interp(r_sep, tr, tp)),
                   coordinate_at_our_separatrix=x_at, te_eV=te, ne_m3=ne,
                   te_over_table_edge=(te / te_e if te_e > 0 else None))
        return out
    except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])


def mse_by_radius(fm, ms):
    """chi^2/N of the MSE chords of the solver's present equilibrium by ``|R - R_axis|`` class
    (:data:`MSE_RADIAL_CLASSES_M`)."""
    tab = score(fm.gs, ms, groups=["mse"])
    g = ms.groups["mse"]
    use = np.asarray(tab.used("mse"), bool)
    res = np.asarray(tab.residual("mse"), float)
    ra = float(np.asarray(fm.gs.o_point, float)[0])
    d = np.abs(np.asarray(g.geometry["r"], float)[use] - ra)
    e1, e2 = MSE_RADIAL_CLASSES_M
    out = dict(r_axis_m=ra, class_edges_m=list(MSE_RADIAL_CLASSES_M))
    for name, m in (("core", d < e1), ("mid", (d >= e1) & (d <= e2)), ("outer", d > e2), ("all", np.ones_like(d, bool))):
        out[name] = dict(n=int(np.count_nonzero(m)), chi2_per_n=float(np.mean(res[m] ** 2)) if np.any(m) else None)
    return out


def steady_state_check(fm, kin, src, cfg, params, ms, label):
    """RECORD that steers nothing (``steady_state_check``; on by default in single-slice fits and series): the
    slice's unfitted model steady state -- the consistent solve of ``params`` with the corrections zero, the core
    treatment of ``cfg``, the slice's own inputs and Ip target -- scored on the MSE chords by radius
    (:func:`mse_by_radius`), next to the score of the solver's present state (the slice's own result).  The solver's
    state and probe correction are restored afterwards to the state found on entry.  Its solves are counted on the
    forward model as every solve is and stated here (``n_solves``); callers take their own counts before it.  Never
    raises."""
    try:
        if ms is None or "mse" not in ms.groups:
            return dict(computed=False, reason="no MSE group at this slice")
        own = mse_by_radius(fm, ms)
        snap, pc = fm.snapshot(), fm.probe_correction.copy()
        n0, t0 = fm.n_solves, time.perf_counter()
        p = dict(params, a_corr=(0.0,) * len(cfg.corr_centers_rho), log_strength=None)
        try:
            r = _cm.consistent_solve(fm, kin, src, p, cfg, label=label)
            ss = mse_by_radius(fm, ms) if r.converged else None
            stats = (r.solve.stats if r.solve is not None else {}) or {}
            st = dict(status=r.status, l_i=stats.get("l_i"), q_0=stats.get("q_0"), steady_state=ss)
        finally:
            fm.restore(snap)
            fm.probe_correction = pc
        return dict(computed=True, own=own, **st, n_solves=fm.n_solves - n0, wall_s=round(time.perf_counter() - t0, 2),
                    note=STEADY_STATE_CHECK_NOTE)
    except Exception as exc:  # noqa: BLE001 -- a record must never end a fit or a series
        return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])
