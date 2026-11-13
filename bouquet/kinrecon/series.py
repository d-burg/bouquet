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
* sawteeth (:mod:`.current.sawtooth`) applied by the reconnection operator, which is the identity
  wherever q >= 1, so there is no sawtooth effect at all on a state with q >= 1 everywhere.  Default
  ``sawtooth="auto"``: NO prescribed crash train -- the operator acts only at MEASURED crash times
  (``crash_table``, a :class:`.crash_table.CrashTable`, or ``crash_times``), i.e. ``"crash_times"``
  when a table or times are given, and ``"off"`` (no crashes) otherwise; whenever the evolved q falls
  below 1 on a slice with no crashes applied (no table, an interval the table's detection did not
  cover, or a covered interval with no observed crash -- a model-data inconsistency), the slice record carries
  a ``sawtooth_warning`` and the series lists it in ``warnings``
  (the choice made is recorded as ``sawtooth_resolved``).  The periodic train is available only as
  the explicit option ``sawtooth="periodic"`` (recorded):
  ``sawtooth="crash_times"`` uses measured crash times; ``sawtooth="periodic"`` places crashes every
  ``1/nu`` with ``nu`` from the two parameters ``(f, ln Lambda)``; ``ln Lambda`` is fitted at the
  first slice when the initial state is ``"mse_fit"`` (a free parameter of the first-slice fit by
  default, as in single-slice fits) and that value sets the train, otherwise it is
  ``sawtooth_log_strength`` (default: the fit module's prior mean).  Phase of the periodic train
  (stated, not measured): the first slice's state is the CYCLE AVERAGE of the periodic orbit (initial
  ``"relaxed"`` and ``"mse_fit"``), and the first crash of the train is one period AFTER the first
  slice time (``t_0 + 1/nu``, ``t_0 + 2/nu``, ...; ``sawtooth_first_crash_phase`` moves it to
  ``t_0 + phase/nu`` for sensitivity studies).  With measured crash times the first slice's state
  is the window average of the measured-crash slice run of the current model over the window
  ``[t_0 - (t_1 - t_0), t_0]`` (the first interval's length; the run starts on the periodic orbit at
  the measured mean interval ``history`` before the window, so the crash record must cover it; a record
  that covers the history and holds no crash in it starts the series crash-free, recorded), and
  the later crashes are the measured ones; ``"off"``: none.  A slice reports the crash-free end state plus the sawtooth effect averaged
  over the interval since the previous slice (the measurement window): ``psi = psi_end^(no crash) +
  <psi^(crashes)>_window - <psi^(no crash)>_window``.  This is the instantaneous state when the operator
  acted as the identity, is continuous in the state (a switch between averaged and instantaneous states
  made the pass loop oscillate; measured), and carries no smooth time lag.  With a period longer than
  the interval the window does not span a whole period (phase dependence remains; stated).

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
carried to the next interval is the instantaneous end state of the last pass.

Initial state (first slice):

* ``"mse_fit"``: the first slice's current is fitted to magnetics and E_r-corrected MSE by
  :func:`.fit.reconstruct` with the current-correction Gaussians on top of the relaxed model shape
  (steady state, or the sawtooth periodic state when it is active); the fitted state (whose ohmic
  part then carries the corrections, i.e. is NOT relaxed) starts the evolution, and the
  corrections are not applied again;
* ``"relaxed"``: the steady (or periodic) model state of the first slice;
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
* ``"auto"``: ``"mse_fit"`` when the first slice has an MSE group declared E_r-corrected, else
  ``"relaxed"`` (recorded).

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
``"converged_within_noise"`` is absorbed (the label is kept in the slice's ``filter.status``): the state is
re-advanced on the ACCEPTED center's own equilibrium (the fit's other solves -- Jacobian columns,
trials -- also advance the state, on their perturbed equilibria, and none of those is carried),
the corrections are added to it as their zero-net-current steady response, and that state is
carried forward and the center is the slice's answer.  Any other fit status (stalled, max_iter,
failed) leaves the slice at its evolved, unfitted equilibrium and state, which are scored and
carried; the record says ``absorbed: False`` with the fit's status.

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
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import current_model as _cm
from . import fit as _fit
from .current import sawtooth as _saw
from .current.diffusion import CurrentDiffusion, CurrentState, GeometrySchedule, IpBoundary, OuterLayer
from .score import chi2_report, chi2_summary, flux_offset_record, score

__all__ = ["SliceData", "SeriesSettings", "run_series", "INITIAL_MODES", "SAWTOOTH_MODES"]

INITIAL_MODES = ("auto", "mse_fit", "relaxed", "from_profile")
SAWTOOTH_MODES = ("auto", "crash_times", "off", "periodic")
#: What a series does when its first-slice fit (``initial="mse_fit"``) did not end ``"converged"``.
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
    dt_max: float = 1.0e-3                               # diffusion step [s]
    scheme: str = "trbdf2"
    sawtooth: str = "auto"                               # SAWTOOTH_MODES; "periodic" only as an explicit option
    sawtooth_f: float = _fit.PRIOR_F
    sawtooth_log_strength: Optional[float] = None        # None: the prior mean of the fit module
    sawtooth_first_crash_phase: float = 1.0              # periodic train: first crash at t0 + phase/nu (0, 1]
    crash_times: Optional[Sequence[float]] = None
    crash_table: Optional[Any] = None                    # a crash_table.CrashTable (times and detection coverage)
    crash_record_start: Optional[float] = None           # [s] start of the crash detection (default: the table's)
    layer_fpol: str = "linear"                           # LAYER_FPOL_MODES
    vessel_eddy: bool = False
    zeff_series_source: Optional[str] = None  # the series' ONE Z_eff source kind for the checked-source rule
    #                                            ("impurity_density" or "line_integrated"; see check_zeff_series)
    zeff_source_schedule: Optional[Sequence[Dict[str, Any]]] = None  # NON-DEFAULT: declared Z_eff source intervals
    #                                            (start, end, candidate, source); checked within each interval
    keep_going_after_failure: bool = False   # True: a failed slice is recorded and skipped; the next interval
    #                                          starts from the last consistent state (recorded per slice)

    def __post_init__(self):
        if self.initial not in INITIAL_MODES:
            raise ValueError(f"initial must be one of {INITIAL_MODES}")
        if self.sawtooth not in SAWTOOTH_MODES:
            raise ValueError(f"sawtooth must be one of {SAWTOOTH_MODES}")
        self.sawtooth_requested = self.sawtooth
        if self.crash_table is not None:
            if self.crash_times is not None:
                raise ValueError("give crash_table or crash_times, not both")
            self.crash_times = tuple(float(t) for t in self.crash_table.times())
            if self.crash_record_start is None:
                self.crash_record_start = self.crash_table.record_start
        if self.sawtooth == "auto":
            self.sawtooth = "crash_times" if self.crash_times else "off"
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
            "crash_times": "operator at the measured crash times only (identity where q >= 1)",
            "off": "no crashes (a warning is recorded on every slice whose evolved q falls below 1)",
            "periodic": "EXPLICIT OPTION: prescribed periodic crash train (not the default)"}[self.sawtooth]
        d["sawtooth_log_strength_used"] = self.log_strength() if self.sawtooth == "periodic" else None
        return d


def _period(settings, log_strength=None):
    lam = float(np.exp(settings.log_strength() if log_strength is None else log_strength))
    return 1.0 / float(_saw.rate_from_strength(lam, settings.sawtooth_f))


def _crash_times(settings, t0, t_end, log_strength=None):
    if settings.sawtooth == "off":
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


def _slice_record(fm, res, sl, label, extra=None):
    """Distilled record of one solved slice (no measured values)."""
    ms = sl.measurements
    tab = rep = None
    if ms is not None and res.converged:
        groups = [g for g in ("flux_loop", "bpol_probe", "ip", "pf_current", "mse") if g in ms.groups]
        tab = score(fm.gs, ms, groups=groups)
        rep = chi2_report(tab)
    asm = res.assembly
    st = res.solve.stats if res.solve is not None else {}
    lag = res.lag
    out = dict(label=label, time=sl.time, status=res.status, passes=res.passes, wall_s=res.wall_s,
               chi2=None if tab is None else {k: dict(chi2=v[0], n=v[1], chi2_per_n=v[2])
                                              for k, v in chi2_summary(tab).items()},
               chi2_report=rep, stats=st, q_check=res.record.get("q_check"), reason=res.record.get("reason"),
               pass_trace=[{k: p.get(k) for k in ("k", "dj", "q_max_rel", "solve_status", "its", "anderson_mixed",
                                                  "anderson_fallback", "d_axis_m")} for p in res.record.get("passes", [])],
               q_axis_traced=None if lag is None else lag["phys"].get("q_axis"),
               q0_axis=((res.record.get("q_check") or {}).get("axis") or {}).get("q0_eq"),
               fractions=asm["fractions"] if asm else None, currents_A=asm["currents_A"] if asm else None,
               v_loop=asm["v_loop"] if asm else None, sawtooth=asm["sawtooth"] if asm else None,
               q_model=dict(rho=np.asarray(asm["q_model"]["rho"]).tolist(), q=np.asarray(asm["q_model"]["q"]).tolist())
               if asm else None,
               profiles_rho={k: np.asarray(v).tolist() for k, v in asm["profiles_rho"].items()} if asm else None,
               boundary=fm.boundary_distance() if res.converged else None)
    out["ip_fit"] = _fit.ip_record(tab, ms, (asm or {}).get("params") or {}, bool((extra or {}).get("ip_fitted")))
    out["flux_offset"] = None if tab is None else flux_offset_record(tab)
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
    return out


def run_series(fm, slices: List[SliceData], cfg, settings=None, start_psi=None, initial_profiles=None,
               callback=None):
    """Evolve the current through ``slices`` (module docstring).

    ``fm``: :class:`.forward.ForwardModel` configured for ``slices[0]`` (its rows are rebuilt for
    every slice with :meth:`~.forward.ForwardModel.reconfigure`; the edge-taper case is kept, so the
    taper stays uniform within the series); ``cfg``: :class:`.current_model.CurrentModelSettings`
    (``n_cells`` sets the carried state's grid); ``start_psi``/``initial_profiles``: first-slice
    starting state of the solver (as :func:`.fit.reconstruct`).  ``callback(record)`` is called after
    every slice.  Returns a dict with ``slices`` (one record each), ``settings`` and the totals.
    """
    s = settings or SeriesSettings()
    t_all = time.perf_counter()
    n_solves0 = fm.n_solves
    times = np.array([float(x.time) for x in slices])
    if np.any(np.diff(times) <= 0):
        raise ValueError("slice times must increase strictly")
    out = dict(settings=s.as_dict(), model_settings=cfg.as_dict(), slices=[], times=times.tolist(),
               ip_convention=IP_CONVENTION)
    out["zeff_series"] = check_zeff_series(slices, s.zeff_series_source, s.zeff_source_schedule)
    zeff_flags = set(out["zeff_series"].get("failing_slices") or []) | set(out["zeff_series"].get("unverified_slices")
                                                                         or [])
    sl0 = slices[0]
    mode = s.initial
    if mode == "auto":
        mse = None if sl0.measurements is None else sl0.measurements.groups.get("mse")
        mode = "mse_fit" if (mse is not None and bool(mse.meta.get("er_corrected", False))) else "relaxed"
        out["initial_auto_choice"] = mode
    case = fm.case
    ls_used = s.log_strength() if s.sawtooth == "periodic" else None
    ls_source = "series setting (sawtooth_log_strength, or the fit module's prior mean)" if ls_used is not None else None
    p_relaxed = dict(log_strength=ls_used, f=s.sawtooth_f)
    cfg0 = cfg
    if s.sawtooth == "crash_times":
        cfg0, p_relaxed, out["first_slice_crash_start"] = _measured_crash_start(s, cfg, times)
    # ---------------------------------------------------------------- slice 0
    t0 = time.perf_counter()
    n0 = fm.n_solves
    fitrec = None
    f0rec0 = _apply_f0(fm, sl0)
    if start_psi is not None:
        fm.set_psi(start_psi)
    if mode == "mse_fit":
        fr = _fit.reconstruct(fm, sl0.kinetics, sl0.sources, cfg0, s.initial_fit, params0=p_relaxed,
                              initial_profiles=initial_profiles, label=f"{sl0.label}:mse_fit")
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
                      ip_fit=fr.record.get("ip_fit"))
        params0 = fr.params
    else:
        state_fn0 = None
        if mode == "from_profile":
            def state_fn0(lag):
                return _given_state(lag, s, cfg, _slice_ip(sl0))
        res0 = _cm.consistent_solve(fm, sl0.kinetics, sl0.sources, p_relaxed, cfg0, label=f"{sl0.label}:{mode}",
                                    initial_profiles=initial_profiles, state_fn=state_fn0)
        params0 = p_relaxed
    ip_c0 = (float(res0.assembly["ip"]) if res0.converged and res0.assembly and res0.assembly.get("ip")
             else _slice_ip(sl0))
    rec0 = _slice_record(fm, res0, sl0, sl0.label, dict(initial_mode=mode, fit=fitrec, n_solves=fm.n_solves - n0,
                                                        ip_carried_A=ip_c0,
                                                        ip_offset_carried=ip_c0 / _slice_ip(sl0) - 1.0,
                                                        **({"zeff_flag": "the series' Z_eff source fails its tests or "
                                                            "is unverified at this slice"}
                                                           if float(sl0.time) in zeff_flags else {}),
                                                        wall_total_s=round(time.perf_counter() - t0, 2),
                                                        vacuum_F0=f0rec0,
                                                        ip_fitted=bool(fitrec and "ip_rel" in fitrec["free"])))
    w0 = _sawtooth_warning(s, res0, float(times[0]), float(times[0]), [])
    if w0 is not None:
        rec0["sawtooth_warning"] = w0
        out.setdefault("warnings", []).append(dict(slice=0, time=float(times[0]), sawtooth=w0["text"]))
    out["slices"].append(rec0)
    if callback:
        callback(rec0)
    if res0.converged and mode in ("relaxed", "mse_fit"):
        out["relaxed_start_guard"] = _relaxed_start_guard(res0, sl0, mode)
    if not res0.converged:
        out.update(status="failed_initial", n_solves=fm.n_solves - n_solves0, wall_s=round(time.perf_counter() - t_all, 2))
        return out
    crash = _crash_times(s, times[0], times[-1], ls_used)
    out["crash_times_used"] = crash.tolist() if s.sawtooth != "periodic" else dict(
        period_s=_period(s, ls_used), phase=("first slice: cycle average of the periodic orbit; first crash "
                                             f"{s.sawtooth_first_crash_phase:g} period after the first slice time"),
        first_crash_phase=float(s.sawtooth_first_crash_phase), f=s.sawtooth_f,
        log_strength=ls_used, log_strength_source=ls_source)
    st0 = res0.assembly["state"]
    carried = CurrentState(st0.rho_faces, st0.psi, st0.psi_boundary, times[0], dict(initial_mode=mode))
    out["slices"][-1]["carried_state"] = _state_record(carried)
    lag_prev = res0.lag
    jni_prev = _jni(lag_prev, sl0.kinetics, sl0.sources, params0)
    # Ip convention (module docstring, "Plasma current in a series"): the carried state's current is the first
    # slice's solved target, and the relative offset of the fitted from the measured Ip (0 without an Ip fit) is
    # carried forward: every later interval is driven from the carried current to the NEXT measured Ip times
    # (1 + carried offset); a filter fit that is absorbed updates the offset (one rescale by the change of the offset)
    ip_carried = ip_c0
    ip_off = float(ip_carried / _slice_ip(sl0) - 1.0)
    snap_prev = fm.snapshot()
    pc_prev = fm.probe_correction.copy()
    status = "completed"
    for k in range(1, len(slices)):
        sl = slices[k]
        t1, tp = times[k], carried.time      # the interval starts at the carried (last consistent) state
        tk0 = time.perf_counter()
        nk = fm.n_solves
        fm.restore(snap_prev)
        f0rec = _apply_f0(fm, sl)
        if sl.measurements is None:     # generation mode (synthetic truth series)
            fm.reconfigure(None, sl.reference_boundary, case_id=case.case_id, ip=_slice_ip(sl),
                           generation_coil_reg=sl.meta["generation_coil_reg"])
        else:
            pf = sl.measurements.groups.get("pf_current")
            roles = sl.meta.get("coil_roles") or (pf.meta.get("coil_roles") if pf is not None else None)
            if not roles:      # no roles stated for this slice: the previous slice's roles are kept (recorded)
                roles = {n: v["role"] for n, v in fm.rows_record.get("coils", {}).get("per_coil", {}).items()}
            fm.reconfigure(sl.measurements, sl.reference_boundary, coil_roles=roles, case_id=case.case_id)
            fm.probe_correction = pc_prev.copy()
        fm.case = case
        ip0, ip1 = ip_carried, _slice_ip(sl) * (1.0 + ip_off)
        holder = {}
        p_pred = dict(log_strength=None, f=s.sawtooth_f,
                      s_bs=params0.get("s_bs", 1.0) if isinstance(params0, dict) else 1.0, ip_rel=ip_off)

        def state_fn(lag, sl=sl, t1=t1, tp=tp, ip0=ip0, ip1=ip1, carried=carried, lag_prev=lag_prev,
                     jni_prev=jni_prev, holder=holder):
            jni1 = _jni(lag, sl.kinetics, sl.sources, p_pred)
            sol, info = interval_solver(lag_prev, lag, jni_prev, jni1, tp, t1, ip0, ip1, s.layer_fpol)
            end, events, states = advance_interval(sol, carried, t1, s, crash, s.dt_max)
            holder.clear()
            end.meta["ip_A"] = float(ip1)
            holder.update(end=end, events=events, states=states, info=info, jni1=jni1, window_average=False)
            use = end
            if events:
                # sawtooth effect over the interval (the measurement window): the interval average WITH the
                # crashes minus the same average WITHOUT them, added to the crash-free end state.  Identical to the
                # instantaneous state when the operator acted as the identity (q >= 1), continuous in the state
                # (the operator is), and free of the smooth time lag a plain window average would add.
                end0, _, states0 = advance_interval(sol, carried, t1, s, np.zeros(0), s.dt_max)
                aw = _window_state(states, tp, t1, end.rho_faces)
                a0 = _window_state(states0, tp, t1, end.rho_faces)
                use = CurrentState(end.rho_faces, end0.psi + aw.psi - a0.psi,
                                   end0.psi_boundary + aw.psi_boundary - a0.psi_boundary, t1,
                                   dict(sawtooth_window=[tp, t1], ip_A=float(ip1)))
                holder["window_average"] = bool(any(e.record.active for e in events))
            return use

        res = _cm.consistent_solve(fm, sl.kinetics, sl.sources, p_pred, cfg, label=f"{sl.label}:evolve",
                                   state_fn=state_fn)
        filt = None
        if res.converged and s.filter_at(t1):
            filt, res = _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1)
        extra = dict(n_solves=fm.n_solves - nk, interval=dict(holder.get("info", {}), start_s=float(tp), end_s=float(t1),
                                                              ip_start_A=float(ip0), ip_end_A=float(ip1),
                                                              ip_offset_used=ip_off),
                     ip_fitted=bool(filt and filt.get("absorbed") and "ip_rel" in (filt.get("free") or [])),
                     crashes=[dict(time=e.time, active=bool(e.record.active),
                                   rho_mix=getattr(e.record, "rho_mix", None),
                                   dpsi_max=getattr(e.record, "dpsi_max", None)) for e in holder.get("events", [])],
                     window_average=holder.get("window_average", False), filter=filt, vacuum_F0=f0rec,
                     wall_total_s=round(time.perf_counter() - tk0, 2))
        if res.converged:
            extra["carried_state"] = _state_record(holder["end"])
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
        w = _sawtooth_warning(s, res, float(tp), float(t1), holder.get("events", []))
        if w is not None:
            rec["sawtooth_warning"] = w
            out.setdefault("warnings", []).append(dict(slice=k, time=float(t1), sawtooth=w["text"]))
        out["slices"].append(rec)
        if callback:
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
    out.update(status=status, n_solves=fm.n_solves - n_solves0, wall_s=round(time.perf_counter() - t_all, 2))
    return out


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
    """The recorded warning of the default sawtooth handling (module docstring): the slice's solved q falls below
    1 and no crash was applied -- because there is no crash record (``sawtooth="off"``, kind ``no_record``), the
    record's detection did not cover the interval or its coverage was not stated (``not_covered``), or the record
    covers the interval and shows no crash (``covered_no_crash``: a model-data inconsistency).  None otherwise.
    A record only."""
    if not res.converged or res.lag is None or s.sawtooth == "periodic":
        return None
    q = np.abs(np.asarray(res.lag["phys"]["q"], float))
    qmin = float(np.min(q[np.isfinite(q)])) if np.any(np.isfinite(q)) else None
    if qmin is None or qmin >= 1.0 or any(bool(e.record.active) for e in events):
        return None
    if s.sawtooth == "off":
        why = "no crash record: no crashes applied (sawtooth off)"
    else:
        cov = s.crash_table.covered(t0, t1) if s.crash_table is not None else None
        if cov is True:
            why = ("the crash record covers this interval and shows no crash: model-data inconsistency (the evolved "
                   "q < 1 where no sawtooth was observed)")
        else:
            why = ("the crash record's detection did not cover this interval: no crashes applied" if cov is False else
                   "no measured crash in this interval and the record's coverage was not stated")
    import warnings
    text = f"evolved q_min = {qmin:.3f} < 1 at t = {t1:.4f} s; {why}"
    warnings.warn(text, RuntimeWarning, stacklevel=3)
    kind = ("no_record" if s.sawtooth == "off" else
            ("covered_no_crash" if (s.crash_table is not None and s.crash_table.covered(t0, t1) is True)
             else "not_covered"))
    return dict(q_min=qmin, interval=[t0, t1], text=text, kind=kind)


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
    Returns ``(cfg0, params, record)``; a crash record that does not cover the history raises."""
    import dataclasses
    if len(times) < 2:
        raise ValueError("sawtooth='crash_times' needs at least two slices (the first window is the first interval)")
    window = (float(times[0] - (times[1] - times[0])), float(times[0]))
    try:
        sched = _saw.schedule_for_slice(s.crash_times, window, history=cfg.history, record_start=s.crash_record_start)
    except ValueError as exc:
        raise ValueError(f"measured crash times cannot start the series at its first slice: {exc}") from exc
    n_hist = int(np.count_nonzero((np.asarray(s.crash_times, float) > sched.t_start)
                                  & (np.asarray(s.crash_times, float) <= window[1])))
    if n_hist == 0:
        # no crash RECORDED in the history: crash-free only where the record says the core was observed
        if s.crash_table is not None and s.crash_table.covered(sched.t_start, window[1]) is not True:
            raise ValueError(f"measured crash times cannot start the series: no crash recorded in the history "
                             f"[{sched.t_start}, {window[1]}], but the crash record did not observe all of it "
                             "(unobserved interval or coverage not stated): 'no crash recorded' is not 'no crash'")
        # the record covers the history (checked above) and holds no crash in it: the first slice is crash-free
        rec = dict(window=list(window), history_s=cfg.history, record_start=s.crash_record_start, n_crashes_history=0,
                   note="no measured crash in the covered history before the first slice: the first slice is the "
                        "crash-free (steady) model state; the measured crashes act from then on")
        return cfg, dict(log_strength=None, f=s.sawtooth_f), rec
    if sched.mean_interval is None:
        raise ValueError("measured crash times cannot start the series: one crash in the history "
                         f"[{sched.t_start}, {window[1]}] (two are needed to set the starting orbit)")
    ls = float(np.log(_saw.crash_strength(s.sawtooth_f, 1.0 / sched.mean_interval)))
    cfg0 = dataclasses.replace(cfg, crash_times=tuple(float(x) for x in s.crash_times), window=window,
                               crash_record_start=s.crash_record_start)
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


def _filter_slice(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1):
    """Filter fit of one evolved slice (module docstring).  Returns ``(record, result)``; on return
    the solver holds the slice's answer and ``holder`` the state to carry."""
    pred = dict(res=res, holder=dict(holder), snap=fm.snapshot(), pc=fm.probe_correction.copy())
    fr = _fit.reconstruct(fm, sl.kinetics, sl.sources, cfg, s.filter_fit, params0=p_pred,
                          start=pred["snap"], label=f"{sl.label}:filter", state_fn=state_fn)
    stale = holder.get("end")       # advanced by the LAST solve of the fit (a column, a trial or the center)
    c = fr.record.get("center")
    filt = dict(status=fr.status, absorbed=False, free=list(fr.free), theta=fr.theta.tolist(), sigma=fr.sigma.tolist(),
                ip_fit=fr.record.get("ip_fit"), flux_offset=(fr.chi2 or {}).get("flux_offset"),
                prior={n: fr.record["specs"].get(n, {}).get("prior") for n in fr.free}, chi2=fr.chi2,
                n_solves=fr.record["n_solves"], wall_s=fr.record["wall_s"],
                stopping=fr.record.get("final_stopping_tests"), stopping_summary=fr.record.get("stopping_summary"))
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
    use_c = state_fn(res.lag)
    ac = fr.params["a_corr"]
    k_ip = ip_fit / float(ip1)
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
    asm_st = res.assembly["state"]
    chk = CurrentState(e.rho_faces, use_c.psi + dpsi, use_c.psi_boundary + dpb, e.time)
    filt.update(absorbed=True,
                carried_from="state re-advanced on the accepted center's equilibrium, plus the corrections' steady "
                             "response",
                center_state_check=dict(zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(chk, asm_st))),
                last_solve_state_vs_center=(dict(zip(("max_dpsi_rel", "dpsi_boundary"), _psi_shape_diff(stale, e)))
                                            if stale is not None else None))
    return filt, res


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
