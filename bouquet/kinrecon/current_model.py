r"""Current-profile model of the kinetic reconstruction and its pass-consistency loop.

Physics (all in Python, iterated with the geometry):

.. code-block:: text

    <J.B>      = <J.B>_ohm + s_bs <J.B>_BS + s_cd <J.B>_CD (+ sum_k a_k g_k)                      (1)
    <J.B>_ohm  = sigma_neo <E.B>,   <E.B> from the steady state of the flux-diffusion equation with
                 I(LCFS) = Ip, held at the q floor (or an opt-in sawtooth state)                   (2)
    <J_phi>    = kappa <J.B> - (dp/dpsi) (<R> - F^2 <1/R> / <B^2>),  kappa = F <1/R> / <B^2>        (3)
    p          = p_thermal(n_e, T_e, n_i, T_i, n_imp) + s_pf p_fast                                (4)

* (2): :class:`.current.diffusion.CurrentDiffusion` in steady-state mode on the geometry of the
  CURRENT equilibrium (:func:`.current.geometry.from_tokamaker`), domain to psi_N = 0.995 with the
  lumped outer layer; ``sigma_neo`` from :func:`.current.conductivity.neoclassical_conductivity`
  (Redl 2021 by default) on the kinetic profiles.  The core treatment is
  :attr:`CurrentModelSettings.core_relaxation` (:data:`CORE_RELAXATIONS`):

  - ``"q_floor"`` (default): the time-averaged state.  The steady state of the MODEL sources
    (bootstrap and driven current) with the RESISTIVE q floor (``q_floor`` default 1, one documented
    setting; :func:`.current.sawtooth.resistive_floor_steady`, the resistivity step of the
    stationary-sawtooth treatment FUSE runs; FUSE also flattens the non-inductive current in a separate
    sawtooth source actor, which this package does not): where that steady state's q is below the
    floor ON AXIS, the resistivity inside a flattening region from the axis is set to its value at the
    region's outer edge, the outer edge being the smallest whose steady state has q >= floor everywhere,
    found to round-off (cut cells overlap-weighted, so the state is continuous in it and converges with
    the grid); where the model q is below the floor off axis only (reversed shear) the floor does not
    act and the record says so (the branches depend on the model state's q only).  The current
    stays continuous (no jump at the edges), Ip is the steady state's own, q_min ends AT the floor and
    q0 above it (the non-inductive current inside is left hollow) -- or q_min below the floor where
    even the plain state's outermost crossing does not reach it (recorded as ``q_floor_met``).  A
    state whose q at the domain edge is below the floor is refused.  The current corrections are added
    AFTER the floor, as their steady response with the plain resistivity (the steady state with them
    among the sources minus the one without), so the floor never cancels them and the fit may pull q
    below the floor where the data demand it; where the floor does not act the state is the steady
    state with the corrections among the sources, to the bit.  No sawtooth parameter, crash time or
    amplitude is used.  What the floor did is recorded (``core``; :func:`floor_effect_record`).
    ``q_floor_form="sharp"`` (opt-in, comparison) is the earlier clamp ``q -> max(q, q_floor)``
    (:func:`.current.sawtooth.floor_q`), which makes ``<J.B>`` jump at the floor radius;
  - ``"periodic_orbit"`` (opt-in): the periodic orbit of "diffuse for one period, reconnect with
    fraction f" (:func:`.current.sawtooth.solve_periodic_orbit`) replaces the steady state and its
    cycle average is used (the window average over an unknown phase, averaged over phases, equals
    the cycle average); its parameters are the strength ``Lambda = -ln(1 - 0.999 f) nu`` (nu the mean
    crash rate) and ``f``, which must then be given;
  - ``"crash_times"`` (opt-in): with measured crash times the time-dependent slice run
    (:func:`.current.sawtooth.simulate_slice`) averaged over the measurement window;
  - ``"none"``: the plain steady state, q unconstrained.

  Every assembly records the treatment, ``q_floor``, its form, whether the floor acted, the
  flattening (floor) radius and every crossing of the floor by the model q, inner and outer
  (``assembly["core"]``, :func:`core_record`).
* ``<J.B>_BS``: Redl 2021, the solver's own evaluation ``TokaMaker.eval_bootstrap`` on the physics grid
  (:mod:`.current.bootstrap`), multiplier ``s_bs = 1`` by default.
* ``<J.B>_CD`` and ``p_fast`` are supplied as arrays (:class:`SourceProfiles`) with optional
  scale factors; this module never computes them.
* (3): the Grad-Shafranov identity for the plain surface average <J_phi> that TokaMaker's
  ``jphi-linterp`` profile takes (derived from ``J_phi = -R p' - F F'/(mu0 R)``
  and ``<J.B> = -F p' - F' <B^2>/mu0``; the same expression as ``CurrentDiffusion.profiles``).
  In the outer layer (psi_N > 0.995) ``<J.B>_ohm = sigma (V_layer / 2 pi) F <1/R^2>`` with the
  layer's uniform loop voltage.
* Frames: psi_N is the equilibrium's own (0 on axis, 1 on the LCFS); the "positive frame" of the
  diffusion module (``|F|``, psi increasing outward, co-current positive).  Kinetic profiles and
  sources are given on psi_N in the full pressure frame (``p(1) = p_sep``).

Coordinates (:mod:`.coordinates`): every model profile lives on rho_tor,norm of the CURRENT
equilibrium.  The physics grid is ``n_model_rho`` points uniform in rho (mapped to psi_N through
the current equilibrium) plus the outer-layer samples; kinetic inputs and sources are evaluated
at those surfaces by the coordinate they carry (:class:`.kinetics.KineticInputs`,
``SourceProfiles.coordinate``).  The current-correction Gaussians are on rho_tor,norm.  TokaMaker
receives psi_N knots from :func:`.coordinates.solver_knots` (default: uniform in rho through the
current pass's equilibrium, plus the domain-edge rule's outer-layer knots).  Default: the outer
layer (psi_N > 0.995) is resolved for the solver by the domain-edge knot and two interior knots in
it; four layer knots gave the same reliability and cost on the public fixture, and the flat
solver stall seen on some real cases is not lifted by them (the stall gate of :mod:`.edge` is
unchanged).

Pass-consistency loop (:func:`consistent_solve`): evaluate (1)-(4) on the current equilibrium,
solve (:class:`.forward.ForwardModel`), repeat, until

* ``max |j_k - j_{k-1}| / max |j_k| < pass_tol`` on a FIXED comparison grid of ``n_cmp`` points
  uniform in rho (the knots move between passes, so they are not the comparison grid),
* the axis moves < 0.5 mm and l_i, q0, Ip change < 1e-3 (relative) between successive solves,
* the lagged probe correction of the forward solve changes by < ``pass_tol`` probe sigmas,
* the q-consistency check passes (:func:`.coordinates.q_consistency`: the solved equilibrium's q
  against the diffusion model's q on rho; ``q_check.required``),
* every solve converged AND verified.

The passes may be Anderson-accelerated (``pass_acceleration="anderson"``, default; depth 3) on the
comparison-grid profile: the input of the next solve is a mixture of the recent inputs and
assembled profiles (the mixing correction is interpolated onto the knots); the fixed point and the
criterion are unchanged, and a mixed input that does not solve falls back to the plain one.  Safeguard:
when the input of the previous pass was a mixture and the max-norm of the pass residual (assembled
minus input profile on the comparison grid) grew against the pass before, the history is restarted and
that pass takes the plain input (recorded as ``anderson_restart`` in the pass record); growth after a
plain input does not restart (where the plain passes themselves diverge, the mixing is what converges);
the residual and the convergence measure are the same.  Every pass record carries ``residual_cos``, the
cosine between this pass's residual and the previous one on the comparison grid (a record only; near -1
on every pass is the signature of an oscillating pass map).

The q-consistency check of pass k compares the equilibrium solved at pass k-1 with the model q of the assembly
made at pass k-1 (``asm``); when that pass solved a mixed (Anderson) or under-relaxed input rather than the
assembled profile, the solved state belongs to that input, not exactly to that assembly (the difference is the
mixing or damping correction, of the size of the pass residual).

Robustness option of the loop (NON-DEFAULT, EXPERIMENTAL; off by default, and with it off the loop is the one
described above).  It changes how the next input is chosen, never the criterion or the budget:

* ``pass_relaxation = beta`` (0 < beta < 1): every pass is under-relaxed from the first -- a plain step
  takes ``x + beta (G(x) - x)`` and an Anderson step the mixing form with the same factor,
  ``x + beta f - (dX + beta dF) gamma`` (the mixing-parameter form of Anderson acceleration, H. F. Walker and
  P. Ni, SIAM J. Numer. Anal. 49, 1715 (2011)), where ``f = G(x) - x`` is the pass residual on the comparison grid.
  **This option does NOT pass the package's factor-of-two test** (:data:`PASS_RELAXATION_NOTE`): with 0.5 and 0.75
  it closed loops that the default does not, but with 0.25 it lost fits that converge by default (heavy damping
  slows the non-oscillating modes until other loops exhaust the same pass budget), and it moves every converged
  fit within its resolution.  The loop record says so (``robustness.pass_relaxation_note``).

An under-relaxed pass leaves the fixed point unchanged (``G(x) = x``) and the convergence measure ``dj`` is the
same pass residual, now of the damped input that was solved, so closure is not loosened.  The loop's state tests
(axis, l_i, q0, Ip between successive solves), however, then compare successive DAMPED solves, whose changes are
about ``beta`` times those of undamped passes.

A supplied current state (``state_fn``; the time-dependent path of :mod:`.series`) replaces the
steady state / periodic orbit: the profiles are then those of that state on the current geometry.
The corrections then act as a non-inductive SOURCE over the interval that ends at the slice: a state
function marked ``corrections_in_window`` (:func:`call_state_fn`) receives the parameters and advances
the flux-diffusion equation with ``sum_k a_k g_k`` among its sources, so the scored state already holds
their response over that interval (the state records the source amplitudes, ``meta["correction_source"]``).
A state function without that mark gets the earlier treatment, the zero-net-current STEADY response
``sum_k a_k psi_k`` (:func:`correction_response`, the response after infinitely many resistive times)
added to its state; :mod:`.series` keeps that form only as a comparison option.

Where a steady-form correction's current returns (:attr:`CurrentModelSettings.corr_return`,
:data:`CORRECTION_RETURNS`).  Default ``"conductivity"``: each correction ``a_k g_k`` is a Gaussian
non-inductive source carrying ``a_k Ip``; at fixed Ip the steady ohmic current gives that current back with a
uniform loop voltage, i.e. distributed like ``sigma_neo F <1/R^2>`` -- mostly on the axis.  On a relaxed state the
axis current then moves by about ``-sum_k a_k`` (relative) whatever the corrections' centers, so every correction
is also a lever on the axis current.  Option ``"local"``: each correction is the Gaussian minus a co-centered
Gaussian ``corr_return_width_ratio`` times wider, scaled to carry zero net current (:func:`correction_shapes` gives
the equations).  A zero-net source leaves the steady loop voltage, hence the ohmic current, unchanged, so the steady
change of ``<J.B>`` is the shape itself: a correction centered at rho 0.2 still changes the core (it is a core
basis function by position), while one at rho 0.5 or 0.8 changes the axis ``<J.B>`` by at most
``exp(-c^2/(2 (r w)^2))`` of its own peak (4.4 % at 0.5, 0.03 % at 0.8 with the default widths) -- its negative
flank does reach inside rho 0.3 for a center at 0.5.  With the sawtooth periodic state the reconnection can still
move current in the q < 1 region when a correction moves the q = 1 surface.  The in-window source of the
time-dependent path always uses the plain Gaussians.  Every fit and series slice records what its corrections did
to the core (:func:`correction_core_effect`).

The series' non-default form ``filter_correction="carried_outside_core"`` (:mod:`.series`) splits each plain
Gaussian at a core radius ``r_c`` with a continuous taper ``w`` (:func:`carry_taper`): the core part ``(1 - w) g_k``
is an in-window source, and the outer part becomes a zero-net change of the carried state that is zero inside
``r_c`` (:func:`carried_outside_core_parts`, :func:`carried_update`).

:func:`lagged_solve` (geometry-dependent inputs lagged from a center) is kept for callers that
want it; the fit (:mod:`.fit`) now takes its Jacobian columns from consistent solves.
"""

import contextvars
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .current import sawtooth as _saw
from .current.bootstrap import EC, solver_jdotb
from .current.conductivity import neoclassical_conductivity
from .current.diffusion import CurrentDiffusion, CurrentState, IpBoundary, OuterLayer
from .current.diffusion import _boundary_gradient_weights as _diff_bgw
from .current.geometry import from_tokamaker
from ._numpy_compat import trapezoid as _trapezoid
from .coordinates import KnotSettings, QCheckSettings, q_consistency, rho_map_from_geometry, solver_knots
from .pressure import pressure_gradient

#: The previous default knots (129 uniform in psi_N), kept by name: ``knot_settings=KnotSettings(
#: rule="psin_uniform")`` reproduces them up to the domain-edge rule; ``knots=DEFAULT_KNOTS``
#: reproduces them exactly (fixed explicit knots, no edge rule).
DEFAULT_KNOTS = np.linspace(0.0, 1.0, 129)
#: psi_N at which the solved equilibrium's axis q is taken for the q-consistency check (traced q limit).
Q_AXIS_PSIN = 1.0e-6
#: Where the net current of a current correction returns in its STEADY form (single-slice fits, the first slice of
#: a series, the series' steady comparison option; :attr:`CurrentModelSettings.corr_return`, module docstring):
#: ``"conductivity"`` (default) -- each correction is a Gaussian non-inductive source and the ohmic current of the
#: steady state returns its net current with a uniform loop voltage, i.e. distributed like the conductivity (mostly
#: on the axis); ``"local"`` -- each correction is the Gaussian minus a wider co-centered Gaussian scaled to zero net
#: current, so the steady ohmic current is unchanged and the return sits in the correction's own flanks.
CORRECTION_RETURNS = ("conductivity", "local")
#: rho_tor,norm inside which the corrections' change of the enclosed current is reported
#: (:func:`correction_core_effect`).
CORRECTION_CORE_RHO = 0.3
#: rho_tor,norm at which the corrections' change of q is reported beside the axis value.
CORRECTION_Q_RHO = 0.2
#: REPORTING level (not an acceptance criterion; nothing is rejected or changed by it): a result whose corrections
#: change the axis <J.B> by more than this fraction of its uncorrected value carries a plain flag.
CORRECTION_AXIS_REPORT_LEVEL = 0.05
#: Core treatments of the current model (:attr:`CurrentModelSettings.core_relaxation`, module docstring):
#: ``"q_floor"`` (default; time-averaged, q held at ``q_floor``), the opt-ins ``"periodic_orbit"`` and
#: ``"crash_times"`` (they need the sawtooth parameters ``log_strength`` and ``f``), and ``"none"``.
CORE_RELAXATIONS = ("q_floor", "periodic_orbit", "crash_times", "none")

#: What the record says about the EXPERIMENTAL ``pass_relaxation`` option (module docstring).
PASS_RELAXATION_NOTE = ("EXPERIMENTAL: does NOT pass the package's factor-of-two test -- at 0.25 it lost fits that "
                        "converge by default (slow modes exhaust the pass budget); 0.5 and 0.75 did not")
#: Core treatments that use the sawtooth parameters ``log_strength`` and ``f``.
SAWTOOTH_RELAXATIONS = ("periodic_orbit", "crash_times")


class ModelFailure(RuntimeError):
    """A model evaluation that cannot produce a profile (e.g. a periodic orbit that does not converge)."""


# ============================================================== inputs
@dataclass
class KineticProfiles:
    """Thermal kinetic profiles on psi_N (the equilibrium's own normalization), full frame.

    ``ne``, ``ni``, ``nimp`` [m^-3], ``te``, ``ti`` [eV], ``zeff``.  ``nimp`` (impurity ion density,
    optional) enters the thermal pressure ``p = e (n_e T_e + (n_i + n_imp) T_i)``; ``p_thermal``
    [Pa], when given, replaces that formula (e.g. a source's own total thermal pressure).
    ``source`` names where the profiles came from.
    """
    psi_n: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    zeff: np.ndarray
    nimp: Optional[np.ndarray] = None
    p_thermal: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self):
        for k in ("psi_n", "ne", "te", "ni", "ti", "zeff", "nimp", "p_thermal"):
            v = getattr(self, k)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if v.shape != np.shape(self.psi_n) or not np.all(np.isfinite(v)):
                    raise ValueError(f"kinetic profile '{k}' must be finite and on psi_n")
                setattr(self, k, v)
        if np.any(np.diff(self.psi_n) <= 0):
            raise ValueError("psi_n must increase strictly")
        if self.psi_n[0] > 1e-12 or self.psi_n[-1] < 1.0:
            raise ValueError("kinetic profiles must cover psi_N in [0, 1] (no extrapolation is done; a first "
                             "sample within 1e-12 of 0 counts as the axis)")
        for k in ("ne", "te", "ni", "ti", "zeff"):
            if np.any(getattr(self, k)[self.psi_n <= 1.0] <= 0):
                raise ValueError(f"'{k}' must be positive inside the LCFS")

    def at(self, x):
        """All profiles at psi_N = x (PCHIP; x within the given range)."""
        x = np.asarray(x, dtype=float)
        out = {}
        for k in ("ne", "te", "ni", "ti", "zeff", "nimp", "p_thermal"):
            v = getattr(self, k)
            if v is not None:
                out[k] = PchipInterpolator(self.psi_n, v)(x)
        return out

    def thermal_pressure(self, x):
        d = self.at(x)
        if "p_thermal" in d:
            return d["p_thermal"]
        ni = d["ni"] + (d["nimp"] if "nimp" in d else 0.0)
        return EC * (d["ne"] * d["te"] + ni * d["ti"])

    #: These profiles are tabulated on the equilibrium's own psi_N (the previous container); the
    #: rho_tor,norm container with quasineutrality and provenance is :class:`.kinetics.KineticInputs`.
    coordinate = "psi_n"

    def profiles_at(self, psi_n=None, rho=None):
        """Common interface with :class:`.kinetics.KineticInputs`: evaluated at ``psi_n``."""
        if psi_n is None:
            raise ValueError("KineticProfiles are on psi_N: pass psi_n")
        return self.at(psi_n)

    def thermal_pressure_at(self, psi_n=None, rho=None):
        if psi_n is None:
            raise ValueError("KineticProfiles are on psi_N: pass psi_n")
        return self.thermal_pressure(psi_n)


@dataclass
class SourceProfiles:
    """Non-inductive sources supplied as arrays on psi_N (the provider computes them).

    ``jdotb_cd`` [A T/m^2]: flux-surface-averaged driven current density <j_CD . B>, signed relative
    to Ip (positive = co-current).  Alternatively ``jtor_r0_cd`` [A/m^2] with ``r0_cd`` [m]: the
    transport-code form ``<J_phi R0/R>`` (``I = int J dV / (2 pi R0)``), converted on the
    CURRENT geometry for a parallel driven current ``J = (<J.B>/<B^2>) B``:
    ``<J.B> = J <B^2> / (F R0 <1/R^2>)``.  ``p_fast`` [Pa]: fast-ion pressure as an isotropic-equivalent
    scalar ``(p_par + 2 p_perp)/3`` (what enters the Grad-Shafranov P' in the isotropic model),
    full frame.  Either may be None (zero).  Use :meth:`from_j_parallel` for a
    ``j_parallel = <j.B>/B0`` input (IMAS ``core_profiles...j_parallel`` convention).
    ``convention`` records units/frames as received; ``source`` names the provider.
    """
    psi_n: np.ndarray
    jdotb_cd: Optional[np.ndarray] = None
    p_fast: Optional[np.ndarray] = None
    convention: Dict[str, str] = field(default_factory=dict)
    source: str = ""
    jtor_r0_cd: Optional[np.ndarray] = None
    r0_cd: Optional[float] = None
    #: coordinate of the ``psi_n`` array: ``"psi_n"`` (the equilibrium's own psi_N, previous behavior)
    #: or ``"rho_tor_norm"`` (transport-code arrays; placed by rho on every equilibrium)
    coordinate: str = "psi_n"

    def __post_init__(self):
        if self.coordinate not in ("psi_n", "rho_tor_norm"):
            raise ValueError("coordinate must be 'psi_n' or 'rho_tor_norm'")
        self.psi_n = np.asarray(self.psi_n, dtype=float)
        if np.any(np.diff(self.psi_n) <= 0):
            raise ValueError("psi_n must increase strictly")
        if self.jdotb_cd is not None and self.jtor_r0_cd is not None:
            raise ValueError("give jdotb_cd OR jtor_r0_cd, not both")
        if self.jtor_r0_cd is not None and not (self.r0_cd and self.r0_cd > 0):
            raise ValueError("jtor_r0_cd needs r0_cd > 0")
        for k in ("jdotb_cd", "p_fast", "jtor_r0_cd"):
            v = getattr(self, k)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if v.shape != self.psi_n.shape or not np.all(np.isfinite(v)):
                    raise ValueError(f"'{k}' must be finite and on psi_n")
                setattr(self, k, v)
        if self.p_fast is not None and np.any(self.p_fast < 0):
            raise ValueError("p_fast must be >= 0")

    @classmethod
    def from_j_parallel(cls, psi_n, j_parallel, B0, p_fast=None, source="", convention=None):
        """``<j.B>_CD = j_parallel B0`` (IMAS ``j_parallel = <j.B>/B0``)."""
        c = dict(convention or {})
        c["jdotb_cd"] = f"j_parallel [A/m^2] x B0 = {float(B0):g} T (IMAS j_parallel = <j.B>/B0)"
        return cls(psi_n, None if j_parallel is None else np.asarray(j_parallel, float) * float(B0), p_fast,
                   c, source)

    def _at(self, name, x, rho=None):
        v = getattr(self, name)
        if v is None:
            return np.zeros_like(np.asarray(x, dtype=float))
        if self.coordinate == "rho_tor_norm":
            if rho is None:
                raise ValueError(f"'{name}' is tabulated on rho_tor_norm: pass rho")
            x = rho
        x = np.asarray(x, dtype=float)
        if np.any(x < self.psi_n[0] - 1e-12) or np.any(x > self.psi_n[-1] + 1e-12):
            raise ValueError(f"'{name}' does not cover {self.coordinate} in [{x.min():g}, {x.max():g}]")
        return PchipInterpolator(self.psi_n, v)(np.clip(x, self.psi_n[0], self.psi_n[-1]))

    def jdotb_at(self, x, b2=None, fpol=None, inv_r2=None, rho=None):
        """<J_CD.B> at psi_N = x (or at rho for a ``rho_tor_norm`` table); for the ``jtor_r0_cd`` form the
        current geometry's <B^2>, |F>, <1/R^2> at x are required (class docstring)."""
        if self.jtor_r0_cd is not None:
            if b2 is None or fpol is None or inv_r2 is None:
                raise ValueError("jtor_r0_cd needs <B^2>, F and <1/R^2> on x")
            return self._at("jtor_r0_cd", x, rho) * np.asarray(b2) / (np.asarray(fpol) * float(self.r0_cd)
                                                                         * np.asarray(inv_r2))
        return self._at("jdotb_cd", x, rho)

    def p_fast_at(self, x, rho=None):
        return self._at("p_fast", x, rho)


@dataclass
class CurrentModelSettings:
    """Settings of the current model and of the pass-consistency loop.  ``r0`` (reference major
    radius for the geometry, e.g. the g-file RCENTR) is device data supplied by the caller."""
    r0: float = None
    pass_tol: float = 1.0e-5                 # proposed (a tightening of the earlier 1e-3)
    max_passes: int = 15
    state_axis_m: float = 5.0e-4             # consistency: axis change between successive solves
    state_rel: float = 1.0e-3                # consistency: l_i, q0, Ip change between successive solves
    n_cells: int = 100
    nrho: int = 201
    npsi: int = 200
    n_layer: int = 40
    psi_n_edge: Any = "auto"                 # domain edge rule: 0.995
    knots: Optional[Sequence[float]] = None  # explicit FIXED psi_N knots (overrides knot_settings)
    knot_settings: KnotSettings = field(default_factory=KnotSettings)
    q_check: QCheckSettings = field(default_factory=QCheckSettings)
    n_model_rho: int = 201                   # physics grid: points uniform in rho_tor,norm (+ layer samples)
    n_cmp: int = 129                         # pass-consistency comparison grid: points uniform in rho
    fsa_pad_axis: float = 1.0e-3
    fsa_pad_edge: float = 5.0e-5
    conductivity_model: str = "redl2021"
    ln_lambda_model: str = "sauter"          # for sigma_neo (conductivity module default)
    ion_collisionality: str = "spike"        # for the bootstrap (current.bootstrap; the solver's ion_charge_model 1)
    bootstrap_ln_lambda_model: str = "sauter"   # bootstrap Coulomb logarithms: "sauter" (S99 18d-e) or "nrl" (earlier)
    pass_acceleration: str = "anderson"      # "anderson" (depth anderson_depth) or "none"
    anderson_depth: int = 3
    # NON-DEFAULT robustness options of the pass loop (module docstring, "Robustness options"); None = off
    pass_relaxation: Optional[float] = None         # EXPERIMENTAL under-relaxation of every pass (0 < beta < 1);
    #                                                 does NOT pass the factor-of-two test (PASS_RELAXATION_NOTE)
    core_relaxation: str = "q_floor"         # CORE_RELAXATIONS: the core treatment (module docstring)
    q_floor: float = _saw.DEFAULT_Q_FLOOR    # floor of the "q_floor" treatment (FUSE ActorQED qmin_desired default)
    q_floor_form: str = "resistive"          # Q_FLOOR_FORMS: "resistive" (default) or "sharp" (opt-in, comparison)
    # non-default options of the resistive floor (current.sawtooth module docstring, "Non-default options"):
    q_floor_total_current: str = "off"       # TOTAL_CURRENT_FORMS: "off" (default), "flat_at_level",
    #                                          or "flat_at_level_own_region" (its region from its own states)
    q_floor_transition: str = "step"         # FLOOR_TRANSITIONS: "step" (default) or "blend_margin"
    q_floor_branch_freeze: str = "off"       # BRANCH_FREEZES: "off" (default) or "first_assembly" (diagnostic)
    sawtooth_nsteps: int = 40
    sawtooth_sheet_width: float = _saw.DEFAULT_SHEET_WIDTH
    crash_times: Optional[Sequence[float]] = None   # measured crash times [s] ("crash_times" treatment)
    window: Optional[Sequence[float]] = None        # measurement window [t0, t1] [s] (with crash times)
    crash_record_start: Optional[float] = None      # [s] start of the crash detection (default: first crash)
    history: float = _saw.DEFAULT_HISTORY
    corr_centers_rho: Sequence[float] = (0.2, 0.5, 0.8)   # current-correction Gaussians, centers in rho_tor,norm
    corr_width_rho: float = 0.1                           # their width (standard deviation) in rho_tor,norm
    corr_return: str = "conductivity"     # CORRECTION_RETURNS: how a steady-form correction's net current returns
    corr_return_width_ratio: float = 2.0  # "local": width of the compensating Gaussian / corr_width_rho

    def __post_init__(self):
        if self.core_relaxation not in CORE_RELAXATIONS:
            raise ValueError(f"core_relaxation must be one of {CORE_RELAXATIONS}")
        qf = float(self.q_floor)
        if not (np.isfinite(qf) and qf > 0.0):
            raise ValueError("q_floor must be finite and > 0")
        if self.q_floor_form not in _saw.Q_FLOOR_FORMS:
            raise ValueError(f"q_floor_form must be one of {_saw.Q_FLOOR_FORMS}")
        if self.q_floor_branch_freeze not in _saw.BRANCH_FREEZES:
            raise ValueError(f"q_floor_branch_freeze must be one of {_saw.BRANCH_FREEZES}")
        if self.q_floor_total_current != "off" or self.q_floor_transition != "step" or \
                self.q_floor_branch_freeze != "off":
            if self.q_floor_form != "resistive":
                raise ValueError("the non-default options of the q floor need q_floor_form='resistive'")
            _saw._check_options("continuous", self.q_floor_total_current, self.q_floor_transition,
                                "axis" if self.q_floor_branch_freeze != "off" else None)
        v = self.pass_relaxation
        if v is not None and (isinstance(v, bool) or not (np.isfinite(float(v)) and 0.0 < float(v) < 1.0)):
            raise ValueError("pass_relaxation must be None (off) or a number strictly between 0 and 1")
        if self.crash_times is not None and self.core_relaxation != "crash_times":
            raise ValueError(f"crash_times are given but core_relaxation is {self.core_relaxation!r}: measured crash "
                             "times are an opt-in treatment (core_relaxation='crash_times')")

    def as_dict(self):
        d = asdict(self)
        if self.knots is None:
            d["knots"] = dict(generated_per_pass=True, **self.knot_settings.as_dict())
        else:
            d["knots"] = dict(rule="explicit fixed psi_N knots", n=len(self.knots),
                              min_spacing=float(np.min(np.diff(self.knots))))
        return d


#: Model parameters and their defaults (the fit decides which are free).
DEFAULT_PARAMS = dict(log_strength=None, f=0.37, s_bs=1.0, s_cd=1.0, s_pfast=1.0, a_corr=(0.0, 0.0, 0.0),
                      ip_rel=0.0)


def full_params(params):
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    p["a_corr"] = tuple(float(v) for v in p["a_corr"])
    p["ip_rel"] = float(p["ip_rel"] if p.get("ip_rel") is not None else 0.0)
    return p


def ip_target(base_ip, params):
    """The forward solve's Ip target [A]: ``|base_ip| (1 + ip_rel)`` (``ip_rel`` = 0: the measured value)."""
    return abs(float(base_ip)) * (1.0 + full_params(params)["ip_rel"])


def scale_state(state, factor, ip_A=None):
    """A :class:`.current.diffusion.CurrentState` with its current density scaled uniformly by ``factor``: psi
    differences scaled about the boundary value, which is kept (only differences of psi are physical).  The
    state's recorded current ``meta["ip_A"]`` is scaled too (or set to ``ip_A`` when given)."""
    f = float(factor)
    pb = float(state.psi_boundary)
    meta = dict(state.meta or {}, current_scale=f)
    if ip_A is not None:
        meta["ip_A"] = float(ip_A)
    elif meta.get("ip_A") is not None:
        meta["ip_A"] = float(meta["ip_A"]) * f
    return CurrentState(state.rho_faces, pb + f * (np.asarray(state.psi, float) - pb), pb, state.time, meta)


def call_state_fn(state_fn, lag, params):
    """The supplied state of a time-dependent pass: ``state_fn(lag, params)`` for a state function that applies the
    current corrections over its interval itself (attribute ``corrections_in_window = True``: the corrections are a
    non-inductive source of the flux diffusion over the interval, so the state depends on the parameters), else
    ``state_fn(lag)``."""
    if getattr(state_fn, "corrections_in_window", False):
        return state_fn(lag, params)
    return state_fn(lag)


# ============================================================== geometry / physics on the current state
def _psin_to_rho(xs, rs):
    """psi_N -> rho map from the geometry's samples (axis (0, 0) prepended), and its inverse."""
    x = np.concatenate(([0.0], xs))
    r = np.concatenate(([0.0], rs))
    fwd = PchipInterpolator(x, r, extrapolate=True)
    inv = PchipInterpolator(r, x, extrapolate=True)
    return fwd, inv


def evaluate_geometry(gs, kin, cfg):
    """Geometry-dependent ("lagged") inputs on the current equilibrium: the diffusion geometry and
    outer layer, the psi_N <-> rho_tor,norm map, flux-surface averages, conductivity and the raw
    bootstrap on the physics grid (``n_model_rho`` points uniform in rho plus the layer samples)."""
    t0 = time.perf_counter()
    geom = from_tokamaker(gs, cfg.r0, npsi=cfg.npsi, psi_n_edge=cfg.psi_n_edge, nrho=cfg.nrho,
                          with_trapped_fraction=False, n_layer=cfg.n_layer)
    layer = geom.meta["outer_layer_geometry"]
    c = float(geom.meta["psi_n_edge"])
    xs, rs = np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"])
    to_rho, to_psin = _psin_to_rho(xs, rs)
    rmap = rho_map_from_geometry(geom, layer)
    if cfg.knots is not None:
        knots = np.asarray(cfg.knots, dtype=float)
        if knots[0] != 0.0 or knots[-1] != 1.0:
            raise ValueError("knots must run from exactly 0 to exactly 1")
    grid = np.unique(np.concatenate((rmap.psin_of_rho(np.linspace(0.0, 1.0, int(cfg.n_model_rho))), layer.psi_n)))
    ge = np.clip(grid, cfg.fsa_pad_axis, 1.0 - cfg.fsa_pad_edge)
    ge = np.unique(ge)
    # drop near-duplicates created by the clipping (gap < 1e-9 in psi_N)
    ge = ge[np.concatenate(([True], np.diff(ge) > 1e-9))]
    rho_ge = rmap.rho_of_psin(ge)
    fsa = gs.get_fsa(psi=ge)
    _, fc, _, _, eps = gs.sauter_fc(psi=ge, return_eps=True)
    F = np.abs(np.asarray(fsa["F"], float))
    g1 = np.asarray(fsa["<1/R^2>"], float)
    bp2 = np.asarray(fsa["<Bp^2>"], float)
    phys = dict(psi_n=ge, rho=rho_ge, q=np.abs(np.asarray(fsa["q"], float)), F=F,
                inv_r=np.asarray(fsa["<1/R>"], float), r_avg=np.asarray(fsa["<R>"], float), inv_r2=g1,
                b2=bp2 + F ** 2 * g1, r_geo=0.5 * (np.asarray(fsa["R_min"], float) + np.asarray(fsa["R_max"], float)),
                eps=np.asarray(eps, float), ft=1.0 - np.asarray(fc, float))
    dpsi = abs(float(fsa["psi_boundary"]) - float(fsa["psi_axis"]))
    # axis value of the equilibrium's own q profile (limit psi_N -> 0 of the traced q).  TokaMaker's
    # get_stats()['q_0'] is q at psi_N = 0.02 by its definition, not an axis value; this trace is a raw
    # diagnostic (unreliable near the axis); the q check uses extrapolated axis values (coordinates.axis_q)
    try:
        phys["q_axis"] = float(abs(np.asarray(gs.get_q(psi=np.array([Q_AXIS_PSIN]))[1], float)[0]))
    except Exception:  # noqa: BLE001
        phys["q_axis"] = float("nan")
    k = kin.profiles_at(psi_n=ge, rho=rho_ge)
    sig, sparts = neoclassical_conductivity(k["ne"], k["te"], k["zeff"], phys["ft"], phys["q"], phys["r_geo"],
                                            phys["eps"], model=cfg.conductivity_model,
                                            ln_lambda_model=cfg.ln_lambda_model)
    jbs, bsettings = solver_jdotb(gs, ge, k["te"], k["ti"], k["ne"], k["ni"], k["zeff"], fsa["F"],
                                  nimp=k.get("nimp"), ion_collisionality=cfg.ion_collisionality,
                                  ln_lambda_model=cfg.bootstrap_ln_lambda_model)
    return dict(geom=geom, layer=layer, psi_n_edge=c, to_rho=to_rho, to_psin=to_psin, rho_map=rmap, phys=phys,
                dpsi=dpsi, sigma=np.asarray(sig, float), jbs=np.asarray(jbs, float), kin=k,
                kinetics_coordinate=getattr(kin, "coordinate", "psi_n"),
                bootstrap_settings=bsettings, wall_s=round(time.perf_counter() - t0, 4),
                fsa_psi_axis=float(fsa["psi_axis"]), fsa_psi_boundary=float(fsa["psi_boundary"]))


def _interp(x, y, xq, log=False):
    f = PchipInterpolator(x, np.log(y) if log else y, extrapolate=True)
    v = f(np.asarray(xq, float))
    return np.exp(v) if log else v


def correction_shapes(lag, cfg, ip, form="steady"):
    """Unit current-correction shapes ``g_k`` on the physics grid (``<J.B>`` [A T/m^2] per unit
    amplitude): Gaussians in rho_tor,norm, centers ``corr_centers_rho``, width ``corr_width_rho``,
    each normalized so that it carries ``I = Ip`` (``I = F_lcfs int <J.B>/(2 pi F^2) dV`` over the
    domain and the outer layer).  An amplitude ``a_k`` is therefore a fraction of Ip.

    ``form``: ``"steady"`` (the corrections are sources of a steady state / periodic orbit) or ``"in_window"``
    (a source over one interval of the time-dependent path).  With ``cfg.corr_return = "local"`` the STEADY
    shapes are zero-net (:data:`CORRECTION_RETURNS`, :func:`local_return_shape`):

    .. code-block:: text

        g_k = Ip / (F_lcfs I_1) [G(rho; c_k, w) - (I_1 / I_r) G(rho; c_k, r w)]
        G(rho; c, s) = exp(-(rho - c)^2 / (2 s^2)),   I_s = int G(rho; c_k, s) dV / (2 pi F^2)  (domain + layer)

    with ``w = corr_width_rho`` and ``r = corr_return_width_ratio``: the first term is the default shape (the
    amplitude stays the fraction of Ip it carries) and the second returns that current in the correction's own
    flanks.  The in-window shapes are always the plain Gaussians (over a short interval the flux diffusion
    already returns their current near their location)."""
    if cfg.corr_return not in CORRECTION_RETURNS:
        raise ValueError(f"corr_return must be one of {CORRECTION_RETURNS}")
    if form not in ("steady", "in_window"):
        raise ValueError("form must be 'steady' or 'in_window'")
    geom, layer = lag["geom"], lag["layer"]
    rmap = lag["rho_map"]
    ge, rho_ge = lag["phys"]["psi_n"], lag["phys"]["rho"]
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    rho_std_g = geom.rho * rmap.rho_edge
    rho_l = rmap.rho_of_psin(layer.psi_n)
    local = cfg.corr_return == "local" and form == "steady"
    if local and not float(cfg.corr_return_width_ratio) > 1.0:
        raise ValueError("corr_return_width_ratio must be > 1")
    out = []
    for rc0 in cfg.corr_centers_rho:
        w = float(cfg.corr_width_rho)
        dom = _trapezoid(gc["dvol_drho"] * np.exp(-0.5 * ((rho_std_g - rc0) / w) ** 2)
                           / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
        lay = layer.integral(layer.source_density(np.exp(-0.5 * ((rho_l - rc0) / w) ** 2)))
        norm = abs(float(ip)) / (layer.fpol_lcfs * (dom + lay))
        if not local:
            out.append(np.exp(-0.5 * ((rho_ge - rc0) / w) ** 2) * norm)
            continue
        wr = float(cfg.corr_return_width_ratio) * w
        dom_r = _trapezoid(gc["dvol_drho"] * np.exp(-0.5 * ((rho_std_g - rc0) / wr) ** 2)
                             / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
        lay_r = layer.integral(layer.source_density(np.exp(-0.5 * ((rho_l - rc0) / wr) ** 2)))
        out.append(local_return_shape(rho_ge, rc0, w, wr, (dom + lay) / (dom_r + lay_r)) * norm)
    return out


def local_return_shape(rho, center, width, return_width, lam):
    """``G(rho; center, width) - lam G(rho; center, return_width)`` (unnormalized local-return correction shape,
    :func:`correction_shapes`); ``lam = I_1 / I_r`` makes it carry zero net current.  Since ``return_width >
    width``, ``G(rho; c, return_width) >= G(rho; c, width)`` everywhere, so ``I_r >= I_1`` and ``0 < lam <= 1``; at
    the axis ``|shape(0)| <= exp(-center^2 / (2 return_width^2))`` (the larger of the two terms there)."""
    rho = np.asarray(rho, float)
    return np.exp(-0.5 * ((rho - center) / width) ** 2) - lam * np.exp(-0.5 * ((rho - center) / return_width) ** 2)


def carry_taper(rho, r_core, width):
    """Weight of the carried part of a correction in the series form ``filter_correction="carried_outside_core"``:

    .. code-block:: text

        w(rho) = 0                                    rho <= r_c
        w(rho) = sin^2(pi (rho - r_c) / (2 Delta))    r_c < rho < r_c + Delta
        w(rho) = 1                                    rho >= r_c + Delta

    on rho_tor,norm, ``r_c = r_core``, ``Delta = width`` (> 0).  ``w`` and its slope are continuous, so neither the
    carried part ``w g`` nor the in-window part ``(1 - w) g`` of a correction jumps anywhere."""
    if not float(width) > 0.0:
        raise ValueError("the taper width must be > 0")
    x = np.clip((np.asarray(rho, float) - float(r_core)) / float(width), 0.0, 1.0)
    return np.sin(0.5 * np.pi * x) ** 2


def _gauss_current_measure(geom, layer, rmap, center, width, r_core=None, taper_width=None):
    """``int G(rho; center, width) [w(rho)] dV / (2 pi F^2)`` over the domain (trapezoid on the geometry's grid) and
    the outer layer: the integral :func:`correction_shapes` normalizes a Gaussian with (``w`` the taper of
    :func:`carry_taper` when ``r_core`` is given)."""
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    rg = geom.rho * rmap.rho_edge
    rl = rmap.rho_of_psin(layer.psi_n)
    fg = np.exp(-0.5 * ((rg - center) / width) ** 2)
    fl = np.exp(-0.5 * ((rl - center) / width) ** 2)
    if r_core is not None:
        fg, fl = fg * carry_taper(rg, r_core, taper_width), fl * carry_taper(rl, r_core, taper_width)
    dom = _trapezoid(gc["dvol_drho"] * fg / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
    return float(dom + layer.integral(layer.source_density(fl)))


def carried_outside_core_parts(lag, cfg, ip, r_core, taper_width, rho_faces):
    r"""The two parts of each current correction in the series form ``filter_correction="carried_outside_core"``
    (:mod:`.series`), for the core radius ``r_c = r_core`` and the taper width ``Delta = taper_width`` (rho_tor,norm;
    :func:`carry_taper`, ``w``):

    .. code-block:: text

        g_k      = N_k G(rho; c_k, s)                         the plain Gaussian (form "in_window"), carrying a_k Ip
        core_k   = (1 - w) g_k                                an in-window source over the slice's interval
        h_k      = N_k w [G(rho; c_k, s) - lam_k G(rho; c_k, r s)]     the CARRIED part
        lam_k    = M[w G(c_k, s)] / M[w G(c_k, r s)]

    with ``s = corr_width_rho``, ``r = corr_return_width_ratio`` (the existing settings of the local return) and
    ``M[f] = sum_cells V' f / (2 pi F^2) h + (outer-layer source integral of f)``, the measure in which the diffusion
    state counts current (its cell sources and the lumped layer).  So ``h_k`` carries zero net current in the state's
    own measure (to round-off), its return sits in the correction's own flank OUTSIDE ``r_c``, and ``h_k = 0`` for
    ``rho <= r_c``.  ``N_k`` is the plain Gaussian's normalization (:func:`correction_shapes`), so ``a_k`` keeps its
    meaning (the fraction of Ip the plain Gaussian carries) and the prior on it is unchanged.  Where ``w G(c_k, r s)``
    has no current (``r_c`` beyond the domain) ``h_k = 0``.

    Returns a dict: ``core`` and ``outer`` (lists of arrays on the physics grid ``lag["phys"]["rho"]``), ``outer_cells``
    (the carried parts' cell sources ``V' h_k / (2 pi F^2) h`` on ``rho_faces``, the diffusion's ``s_c``),
    ``outer_layer_B`` (their outer-layer source integrals), ``lam``, the solver coefficients used for the state
    (``k_f``, ``rc``) and the settings."""
    geom, layer, rmap = lag["geom"], lag["layer"], lag["rho_map"]
    rho_ge = np.asarray(lag["phys"]["rho"], float)
    w_ge = carry_taper(rho_ge, r_core, taper_width)
    plain = correction_shapes(lag, cfg, ip, form="in_window")
    rf = np.asarray(rho_faces, float)
    rc = 0.5 * (rf[1:] + rf[:-1])
    h = np.diff(rf)
    c = build_solver(lag, np.zeros_like(np.asarray(lag["phys"]["psi_n"], float)), 0.0).coefficients(0.0, rf)
    vfac = c.geo_c["dvol_drho"] / (2.0 * np.pi * c.geo_c["fpol"] ** 2) * h
    rc_std = rc * rmap.rho_edge
    rl = rmap.rho_of_psin(layer.psi_n)
    w_c, w_l = carry_taper(rc_std, r_core, taper_width), carry_taper(rl, r_core, taper_width)
    s, r = float(cfg.corr_width_rho), float(cfg.corr_return_width_ratio)
    if not r > 1.0:
        raise ValueError("corr_return_width_ratio must be > 1")
    core, outer, cells, lay_b, lams = [], [], [], [], []
    for c0, g in zip(cfg.corr_centers_rho, plain):
        c0 = float(c0)
        norm = abs(float(ip)) / (layer.fpol_lcfs * _gauss_current_measure(geom, layer, rmap, c0, s))
        core.append((1.0 - w_ge) * g)

        def gauss(x, wid):
            return np.exp(-0.5 * ((np.asarray(x, float) - c0) / wid) ** 2)

        m1 = float(np.sum(vfac * w_c * gauss(rc_std, s)) + layer.integral(layer.source_density(w_l * gauss(rl, s))))
        m2 = float(np.sum(vfac * w_c * gauss(rc_std, r * s))
                   + layer.integral(layer.source_density(w_l * gauss(rl, r * s))))
        if not (m1 > 0.0 and m2 > 0.0):
            lams.append(None)
            outer.append(np.zeros_like(rho_ge))
            cells.append(np.zeros_like(rc))
            lay_b.append(0.0)
            continue
        lam = m1 / m2
        lams.append(lam)
        outer.append(norm * w_ge * (gauss(rho_ge, s) - lam * gauss(rho_ge, r * s)))
        cells.append(norm * vfac * w_c * (gauss(rc_std, s) - lam * gauss(rc_std, r * s)))
        lay_b.append(float(norm * layer.integral(layer.source_density(w_l * (gauss(rl, s) - lam * gauss(rl, r * s))))))
    return dict(core=core, outer=outer, outer_cells=cells, outer_layer_B=lay_b, lam=lams, k_f=np.asarray(c.k_f),
                F_f=np.asarray(c.F_f), rc=rc, rho_faces=rf, rho_edge=float(rmap.rho_edge), r_core=float(r_core),
                taper_width=float(taper_width), ip_A=abs(float(ip)), fpol_lcfs=float(layer.fpol_lcfs))


def carried_update(parts, amplitudes):
    """The change of the carried diffusion state made by an absorbed fit in the series form
    ``filter_correction="carried_outside_core"``: the state whose current density is ``sum_k a_k h_k`` (the carried
    parts of :func:`carried_outside_core_parts`) on the cells, with no ohmic part (zero net current, so the steady
    loop voltage is zero).  Built as :func:`state_from_jdotb` builds a state (``I/F`` at the faces is the cumulative
    cell source, ``psi' = (I/F) / k``), so the enclosed current at every face inside ``r_c`` is exactly zero.
    Returns ``(dpsi, dpsi_boundary, record)``: the change to add to a state on the same faces (the boundary value is
    kept: 0) and a record of the update's net current over the domain and the layer, its current at the domain edge,
    and the largest enclosed current it makes at a face inside ``r_c`` (fractions of Ip; the first and the last are
    zero by construction, to round-off)."""
    from .current.diffusion import _psi_boundary_from_flux
    a = [float(v) for v in amplitudes]
    rc, k_f, rf = parts["rc"], parts["k_f"], parts["rho_faces"]
    n = rc.size
    s_c = np.zeros(n)
    lay_b = 0.0
    for ak, sk, bk in zip(a, parts["outer_cells"], parts["outer_layer_B"]):
        if ak != 0.0:
            s_c = s_c + ak * sk
            lay_b += ak * bk
    gam = np.cumsum(s_c)                                   # I/F at faces 1..n (the ohmic rate is zero)
    dpsi = gam[:n - 1] * np.diff(rc) / k_f[1:n]
    psi = np.concatenate(([0.0], np.cumsum(dpsi)))
    psi = psi - _psi_boundary_from_flux(psi, rc, gam[-1] / k_f[-1])
    ipv, fl, ff = parts["ip_A"], parts["fpol_lcfs"], parts["F_f"][1:]
    inside = rf[1:] <= parts["r_core"] / parts["rho_edge"]          # faces inside r_c (domain rho)
    rec = dict(net_current_over_ip=float(fl * (gam[-1] + lay_b) / ipv) if ipv else float("nan"),
               domain_edge_current_over_ip=float(ff[-1] * gam[-1] / ipv) if ipv else float("nan"),
               current_inside_r_core_over_ip=(float(np.max(np.abs(ff[inside] * gam[inside]))) / ipv
                                              if (ipv and np.any(inside)) else 0.0))
    return psi, 0.0, rec


def _domain_arrays(lag, values_ge, log=False):
    """Values given on the physics grid -> (domain rho grid values, layer sample values)."""
    ge = lag["phys"]["psi_n"]
    geom, layer = lag["geom"], lag["layer"]
    x_g = lag["to_psin"](geom.rho)
    x_g[0] = 0.0
    vg = _interp(ge, values_ge, np.clip(x_g, ge[0], None), log=log)
    vl = _interp(ge, values_ge, layer.psi_n, log=log)
    return vg, vl


def build_solver(lag, jni_ge, ip, sigma=None):
    """Steady (static-geometry) :class:`.current.diffusion.CurrentDiffusion` on the lagged geometry with
    the conductivity of ``lag`` (or ``sigma`` on the physics grid), ``<J_NI.B>`` = ``jni_ge`` on the
    physics grid, Ip boundary ``ip`` and the lumped outer layer."""
    sig_ge = lag["sigma"] if sigma is None else sigma
    sig_g, sig_l = _domain_arrays(lag, sig_ge, log=True)
    jni_g, jni_l = _domain_arrays(lag, jni_ge)
    rho_g = lag["geom"].rho
    lx = lag["layer"].psi_n
    return CurrentDiffusion(lag["geom"], (rho_g, sig_g), IpBoundary(abs(float(ip))), j_ni=(rho_g, jni_g),
                            outer_layer=OuterLayer.from_profiles(lag["layer"], (lx, sig_l), (lx, jni_l)))


def correction_response(lag, shapes, amplitudes, rho_faces):
    """Zero-net-current steady response of the corrections: ``sum_k a_k psi_k`` where ``psi_k`` is the
    steady state with ``<J_NI.B> = g_k`` and Ip = 0 (the ohmic current compensates with a uniform
    E_par).  Returns cell values and the boundary value to add to a state's psi.  This is the response
    of a source that has acted for infinitely many resistive times (the limit of the in-window
    response of :mod:`.series`); on the time-dependent path it is used only by the comparison option."""
    dpsi = np.zeros(len(rho_faces) - 1)
    dpb = 0.0
    for a, g in zip(amplitudes, shapes):
        if a == 0.0:
            continue
        st = build_solver(lag, g, 0.0).steady_state(rho_faces)
        dpsi = dpsi + a * st.psi
        dpb += a * st.psi_boundary
    return dpsi, dpb


def state_from_jdotb(lag, jdotb_ge, ip, n_cells):
    """Diffusion state whose flux-surface-averaged current is ``jdotb_ge`` (on the physics grid of
    ``lag``): ``I/F = int V' <J.B> / (2 pi F^2) drho``, ``psi' = (I/F) / k`` on the domain cells
    (diffusion module, eqs. 3-4); psi = 0 on the first cell."""
    sol = build_solver(lag, jdotb_ge, ip)
    rf = CurrentState.uniform_faces(n_cells)
    rc = 0.5 * (rf[1:] + rf[:-1])
    n = rf.size - 1
    c = sol.coefficients(0.0, rf)
    gam = np.cumsum(c.s_c)
    dpsi = gam[:n - 1] * np.diff(rc) / c.k_f[1:n]
    psi = np.concatenate(([0.0], np.cumsum(dpsi)))
    wb, w1, w2 = _diff_bgw(rc)
    psi_b = float((gam[-1] / c.k_f[-1] - w1 * psi[-1] - w2 * psi[-2]) / wb)
    return CurrentState(rf, psi, psi_b, 0.0, dict(from_jdotb=True, enclosed_current_domain_edge=float(gam[-1] * c.F_f[-1])))


def assemble(lag, kin, src, params, cfg, ip, z0=None, state=None):
    """Profiles for one forward solve from the lagged geometry inputs ``lag``
    (:func:`evaluate_geometry`) and the parameters.

    The corrections are non-inductive sources ``sum_k a_k g_k`` of the diffusion solver.  Without
    ``state`` they enter the core treatment of ``cfg.core_relaxation`` (the ohmic part keeps Ip): with the
    default ``"q_floor"`` their steady response is added to the floored model state (module docstring), otherwise
    they are sources of the steady state / periodic orbit.  With ``state``
    (a :class:`.current.diffusion.CurrentState` on ``n_cells`` cells of the domain rho, the
    time-dependent path) that state replaces the steady state / orbit.  A state that carries
    ``meta["correction_source"]`` was advanced with the corrections as an in-window source
    (module docstring): nothing is added, and the non-inductive profile uses those source amplitudes.
    Otherwise (earlier treatment, comparison option) the corrections are added as their
    zero-net-current steady response (:func:`correction_response`).

    Returns a dict with the knots ``x`` (psi_N) and their ``rho_knots``, ``jphi`` [A/m^2] and the
    full-frame pressure ``p`` [Pa] on them, ``jphi_cmp`` on the fixed comparison grid ``rho_cmp``, the
    parts of <J.B> and the current fractions, the model q on rho (``q_model``), the sawtooth record,
    the core-treatment record (``core``, :func:`core_record`), the diffusion ``state`` used, and ``orbit_z``
    (continuation for the next orbit)."""
    t0 = time.perf_counter()
    p = full_params(params)
    geom, layer, c = lag["geom"], lag["layer"], lag["psi_n_edge"]
    rmap = lag["rho_map"]
    ph, ge = lag["phys"], lag["phys"]["psi_n"]
    rho_ge = ph["rho"]
    ipv = abs(float(ip))
    # sources on the physics grid (placed by the coordinate they carry)
    jcd = p["s_cd"] * src.jdotb_at(ge, b2=ph["b2"], fpol=ph["F"], inv_r2=ph["inv_r2"], rho=rho_ge)
    jbs = p["s_bs"] * lag["jbs"]
    jcorr = np.zeros_like(ge)
    corr_shapes = []
    shapes = None
    # the corrections' source amplitudes at this slice: the parameters, or -- for a supplied state that already
    # carries them as an in-window source (time-dependent path, ``meta["correction_source"]``) -- the source
    # amplitudes that state was advanced with (the parameters' amplitudes over the interval ending at this slice)
    src_amp = None if state is None else (state.meta or {}).get("correction_source")
    amps = tuple(float(v) for v in (p["a_corr"] if src_amp is None else src_amp))
    if len(p["a_corr"]) != len(cfg.corr_centers_rho) and any(a != 0.0 for a in p["a_corr"]):
        raise ValueError("a_corr needs one amplitude per correction center")
    if any(a != 0.0 for a in amps) or any(a != 0.0 for a in p["a_corr"]):
        if len(amps) != len(cfg.corr_centers_rho):
            raise ValueError("a_corr needs one amplitude per correction center")
        shapes = correction_shapes(lag, cfg, ipv, form="steady" if src_amp is None else "in_window")
        coc = None if (state is None or src_amp is None) else (state.meta or {}).get("carried_outside_core")
        if coc is not None:
            # series form "carried_outside_core": the slice's corrections are the in-window core parts plus the
            # carried outer parts (carried_outside_core_parts); both are already in the supplied state
            prt = carried_outside_core_parts(lag, cfg, ipv, coc["r_core"], coc["taper_width"], state.rho_faces)
            shapes = [cp + op for cp, op in zip(prt["core"], prt["outer"])]
        for a, sa, g, rc0 in zip(p["a_corr"], amps, shapes, cfg.corr_centers_rho):
            jcorr = jcorr + sa * g
            corr_shapes.append(dict(center_rho_tor=rc0, width_rho_tor=cfg.corr_width_rho, amplitude_fraction_of_ip=a,
                                    source_amplitude_fraction_of_ip=sa))
    jni_ge = jbs + jcd + jcorr
    sol = build_solver(lag, jni_ge, ipv)
    lx = layer.psi_n
    sig_l = _interp(ge, lag["sigma"], lx, log=True)
    rf = CurrentState.uniform_faces(cfg.n_cells)
    saw = dict(active=False)
    orbit_z = None
    corr_form = None
    if state is not None:
        if state.psi.size != cfg.n_cells:
            raise ValueError("the supplied state must be on n_cells cells")
        rf = state.rho_faces
        st_use = state
        if src_amp is not None:
            # in-window form: the corrections acted as a non-inductive source over the interval that ends at this
            # slice; the flux diffusion of that interval already put their response into the state
            corr_form = "in-window source over the interval ending at this slice (in the supplied state)"
            if (state.meta or {}).get("carried_outside_core") is not None:
                corr_form = ("carried outside the core: in-window source inside r_c, zero-net carried change of the "
                             "state outside it (in the supplied state)")
        elif any(a != 0.0 for a in p["a_corr"]):
            # steady form (the earlier treatment, kept for comparison): the zero-net-current steady response
            dpsi, dpb = correction_response(lag, shapes, p["a_corr"], rf)
            st_use = CurrentState(rf, state.psi + dpsi, state.psi_boundary + dpb, state.time, dict(state.meta))
            corr_form = "zero-net-current steady response added to the supplied state"
        state = st_use
        saw.update(mode="supplied state (time-dependent path)")
        floor_rec = dict((state.meta or {}).get("core") or dict(core_relaxation="supplied state (no core record)"))
    elif cfg.core_relaxation in SAWTOOTH_RELAXATIONS:
        if p["log_strength"] is None:
            raise ValueError(f"core_relaxation={cfg.core_relaxation!r} needs the sawtooth parameter log_strength")
        lam = float(np.exp(p["log_strength"]))
        rate = float(_saw.rate_from_strength(lam, p["f"]))
        period = 1.0 / rate
        if cfg.core_relaxation == "crash_times":
            if cfg.crash_times is None:
                raise ValueError("core_relaxation='crash_times' needs crash_times")
            if cfg.window is None:
                raise ValueError("crash_times need the measurement window")
            sched = _saw.schedule_for_slice(cfg.crash_times, cfg.window, history=cfg.history,
                                            record_start=cfg.crash_record_start)
            sim = _saw.simulate_slice(sol, rf, sched, p["f"], dt=min(sched.mean_interval, period) / cfg.sawtooth_nsteps,
                                      nsteps_orbit=cfg.sawtooth_nsteps, sheet_width=cfg.sawtooth_sheet_width,
                                      orbit_z0=z0)
            orb = sim["orbit"]
            if not orb.converged:
                raise ModelFailure(f"starting periodic orbit not converged (residual {orb.residual:.2e})")
            tr = sim["trajectory"]
            mpsi = tr.window_average(lambda s: s.psi, *cfg.window)
            mpb = float(tr.window_average(lambda s: np.array([s.psi_boundary]), *cfg.window)[0])
            state = CurrentState(rf, mpsi, mpb, cfg.window[1], dict(window_average=True))
            saw.update(mode="measured crash times", n_crashes_window=sched.n_crashes_window,
                       n_crashes_history=sched.n_crashes_history, mean_interval_s=sched.mean_interval)
            orbit_z = orb.z
        else:
            orb = _saw.solve_periodic_orbit(sol, rf, p["f"], period, nsteps=cfg.sawtooth_nsteps,
                                            sheet_width=cfg.sawtooth_sheet_width, z0=z0)
            if not orb.converged:
                raise ModelFailure(f"periodic orbit not converged (residual {orb.residual:.2e}, stepping "
                                   f"{orb.residual_stepping:.2e}) at f = {p['f']}, period = {period:.4g} s")
            state = orb.mean_state()
            orbit_z = orb.z
            saw.update(mode="periodic orbit, cycle average (phase unknown)")
        cr = orb.crash
        saw.update(active=bool(cr.active), strength_per_s=lam, log_strength=float(p["log_strength"]), f=float(p["f"]),
                   rate_per_s=rate, period_s=period, q0_pre=float(cr.q0_pre) if cr.active else None,
                   q0_post=float(cr.q0_post) if cr.active else None,
                   rho_mix=float(cr.rho_mix) if cr.active else None, topology=cr.topology,
                   orbit_residual=float(orb.residual), orbit_residual_stepping=float(orb.residual_stepping),
                   n_newton=int(orb.n_newton), n_plain_cycles=int(orb.n_plain_cycles),
                   orbit_wall_s=round(orb.wall_time, 4))
        floor_rec = dict(core_relaxation=cfg.core_relaxation)
    else:
        if p["log_strength"] is not None:
            raise ValueError(f"log_strength is a parameter of the opt-in sawtooth treatments {SAWTOOTH_RELAXATIONS}; "
                             f"core_relaxation is {cfg.core_relaxation!r}")
        if cfg.core_relaxation == "q_floor":
            # the time-averaged state: the MODEL's steady state (bootstrap + driven current) with the q floor FIRST,
            # then the corrections' steady response -- the steady state with them among the sources minus the one
            # without, plain resistivity -- added, so the floor never cancels a correction; where the floor does not
            # act this is the steady state with the corrections among the sources, to the bit
            sol_m = build_solver(lag, jbs + jcd, ipv)
            st_m = sol_m.steady_state(rf)
            if cfg.q_floor_form == "sharp":
                st_f, floor_rec = _saw.floor_q(st_m, sol_m, cfg.q_floor)
            elif (cfg.q_floor_total_current == "off" and cfg.q_floor_transition == "step" and
                  cfg.q_floor_branch_freeze == "off"):
                st_f, floor_rec, _ = _saw.resistive_floor_steady(sol_m, rf, cfg.q_floor)
            else:
                st_f, floor_rec = _floor_with_options(sol_m, rf, cfg)
            state = st_f
            if any(a != 0.0 for a in p["a_corr"]):
                st_c = sol.steady_state(rf)
                state = CurrentState(rf, st_c.psi + (st_f.psi - st_m.psi),
                                     st_c.psi_boundary + (st_f.psi_boundary - st_m.psi_boundary), st_c.time,
                                     dict(st_c.meta, **({"q_floor": st_f.meta["q_floor"]} if "q_floor" in st_f.meta
                                                        else {})))
            saw.update(mode=f"q floor, {cfg.q_floor_form} (time-averaged state; the corrections added after the floor)")
        else:
            state = sol.steady_state(rf)
            floor_rec = dict(core_relaxation="none")
            saw.update(mode="none (steady state, q unconstrained)")
    saw_params_used = str(saw.get("mode", "")).startswith(("periodic orbit", "measured crash times"))
    prof = sol.profiles(state)
    rate_layer = prof["v_loop_boundary"] / (2.0 * np.pi)
    # knots of this pass (from this pass's equilibrium) and the fixed comparison grid
    if cfg.knots is not None:
        xk = np.asarray(cfg.knots, dtype=float)
        krec = dict(rule="explicit fixed psi_N knots", n=int(xk.size))
    else:
        xk, krec = solver_knots(rmap, cfg.knot_settings)
    rho_cmp = np.linspace(0.0, 1.0, int(cfg.n_cmp))
    x_cmp = rmap.psin_of_rho(rho_cmp)
    x_cmp[0], x_cmp[-1] = 0.0, 1.0
    xs_c = lag["to_psin"](prof["rho_cells"])
    F_ge, kap_ge = ph["F"], ph["F"] * ph["inv_r"] / ph["b2"]

    def jphi_at(xq):
        xq = np.asarray(xq, dtype=float)
        dom = xq <= c
        jb = np.empty_like(xq)
        jb_ni = np.empty_like(xq)
        jb[dom] = _interp(xs_c, prof["jdotb"], xq[dom])
        xqc = np.clip(xq, ge[0], ge[-1])
        jb_ni[:] = _interp(ge, jni_ge, xqc)
        lay = ~dom
        F_k = _interp(ge, F_ge, xqc)
        g1_k = _interp(ge, ph["inv_r2"], xqc)
        jlay_ohm = _interp(ge, lag["sigma"], xqc[lay], log=True) * rate_layer * F_k[lay] * g1_k[lay]
        jb[lay] = jlay_ohm + jb_ni[lay]
        kap = _interp(ge, kap_ge, xqc)
        rq = rmap.rho_of_psin(xq)
        pq = kin.thermal_pressure_at(psi_n=xq, rho=rq) + p["s_pfast"] * src.p_fast_at(xq, rho=rq)
        return jb, jb_ni, kap, F_k, pq, xqc, None

    jb, jb_ni, kap, F_k, pk, xkc, _ = jphi_at(xk)
    dp_dpsin = pressure_gradient(xk, pk)
    dp_dpsi = dp_dpsin / lag["dpsi"]
    jpd = -dp_dpsi * (_interp(ge, ph["r_avg"], xkc) - F_k * kap)
    jphi = kap * jb + jpd
    if not np.all(np.isfinite(jphi)):
        raise ModelFailure("assembled <j_phi> is not finite")
    # comparison grid: the same <j_phi> evaluated on fixed rho points (pressure gradient from the knots)
    jb_c, _, kap_c, F_c, _, xcc, _ = jphi_at(x_cmp)
    dpc = PchipInterpolator(xk, dp_dpsi)(x_cmp)
    jphi_cmp = kap_c * jb_c - dpc * (_interp(ge, ph["r_avg"], xcc) - F_c * kap_c)
    # current fractions: I_x = F_lcfs * int <J.B>_x / (2 pi F^2) dV (domain cells + layer)
    gc = geom.evaluate(prof["rho_cells"], ["dvol_drho", "fpol"])
    h = np.diff(rf)
    w = gc["dvol_drho"] / (2 * np.pi * gc["fpol"] ** 2) * h
    xc = np.clip(xs_c, ge[0], None)
    parts_c = dict(bootstrap=_interp(ge, jbs, xc), driven=_interp(ge, jcd, xc), correction=_interp(ge, jcorr, xc))
    parts_c["ohmic"] = prof["jdotb"] - sum(parts_c.values())
    parts_l = dict(bootstrap=_interp(ge, jbs, lx), driven=_interp(ge, jcd, lx), correction=_interp(ge, jcorr, lx),
                   ohmic=sig_l * rate_layer * layer.fpol * layer.avg_inv_r2)
    cur = {kk: float(layer.fpol_lcfs * (np.sum(w * parts_c[kk]) + layer.integral(layer.source_density(parts_l[kk]))))
           for kk in parts_c}
    tot = sum(cur.values())
    rho_faces_std = prof["rho_faces"] * rmap.rho_edge
    core = core_record(floor_rec, cfg, prof["q"], rho_faces_std, rmap.rho_edge)
    out = dict(x=xk, rho_knots=rmap.rho_of_psin(xk), jphi=jphi, p=pk, jdotb=jb, jdotb_ni=jb_ni, kappa=kap,
               jphi_pressure_term=jpd, rho_cmp=rho_cmp, jphi_cmp=jphi_cmp, knots=krec,
               currents_A=cur, current_sum_A=tot, ip=ipv,
               fractions={kk: v / ipv for kk, v in cur.items()},
               v_loop=float(prof["v_loop_boundary"]), q_diffusion=dict(rho=prof["rho_faces"], q=prof["q"]),
               q_model=dict(rho=rho_faces_std, q=prof["q"]),
               profiles_rho=dict(rho=prof["rho_cells"] * rmap.rho_edge, jdotb=prof["jdotb"],
                                 **{f"jdotb_{kk}": v for kk, v in parts_c.items()}),
               li3_diffusion=float(prof["li3"]), outer_layer_current_A=float(prof["outer_layer_current"]),
               sawtooth=saw, core=core, orbit_z=orbit_z, corrections=corr_shapes, state=state,
               # the sawtooth parameters are read only by the opt-in treatments without a supplied state; elsewhere
               # they are recorded as not used rather than as their opt-in defaults
               params=(p if saw_params_used else dict(p, log_strength=None, f=None)),
               params_not_used=None if saw_params_used else ["log_strength", "f"],
               correction_form=(corr_form if state is not None and corr_form is not None else
                                ("source of the steady state / periodic orbit" if corr_shapes else None)),
               p_sep=float(pk[-1]), pax=float(pk[0] - pk[-1]), wall_s=round(time.perf_counter() - t0, 4))
    try:                         # the floor's branch, to the open branch log: a record only
        _log_assembly(core)
    except Exception:  # noqa: BLE001 -- a record must never end a fit or a series
        pass
    return out


def _floor_with_options(sol_m, rf, cfg):
    """The resistive floor with its non-default options (:class:`CurrentModelSettings` ``q_floor_total_current``,
    ``q_floor_transition``, ``q_floor_branch_freeze``).  The frozen branch: inside a fit (an open freeze,
    :func:`open_branch_freeze`), the first assembly's branch is taken for every later assembly of that fit; outside a
    fit nothing is frozen (recorded).  The frozen branch is the state of the open freeze (the holder), filled by the
    floor itself (``branch_holder``), never read from a record.  Limits (module docstring of :mod:`.current.sawtooth`,
    "Non-default options"): a frozen "no action" branch (``"off_axis"``, ``"none"``) switches the floor off for the
    rest of the fit, even where the plain axis q falls below the floor; only the branch is frozen, not the radius."""
    kw = dict(total_current=cfg.q_floor_total_current, transition=cfg.q_floor_transition)
    hold = _BRANCH_FREEZE.get() if cfg.q_floor_branch_freeze != "off" else None
    used = {}
    if hold is not None and hold.get("branch") is not None:
        kw["branch"] = hold["branch"]
    st_f, rec, _ = _saw.resistive_floor_steady(sol_m, rf, cfg.q_floor, branch_holder=used, **kw)
    if cfg.q_floor_branch_freeze != "off":
        if hold is None:
            rec["branch_freeze"] = "no open freeze (not inside a fit): the plain state's branch"
        elif hold.get("branch") is None:
            hold["branch"] = used.get("branch")
            rec["branch_freeze"] = "first assembly of the fit: its branch is frozen for the rest of the fit"
        else:
            rec["branch_freeze"] = "branch frozen at the fit's first assembly"
    return st_f, rec


#: The frozen branch of the q floor within one fit (diagnostic option ``q_floor_branch_freeze="first_assembly"``):
#: a dict holding the first assembly's branch, opened by :func:`.fit.reconstruct` per fit, or None.
_BRANCH_FREEZE = contextvars.ContextVar("kinrecon_floor_branch_freeze", default=None)


def open_branch_freeze():
    """Open a branch freeze for one fit; returns ``(holder, token)``; close it with :func:`close_branch_freeze`."""
    hold = dict(branch=None)
    return hold, _BRANCH_FREEZE.set(hold)


def close_branch_freeze(token):
    _BRANCH_FREEZE.reset(token)


def core_record(floor_rec, cfg, q_model, rho_tor, rho_edge):
    """The core-treatment record of an assembly (``assembly["core"]``; a record only): the treatment, ``q_floor``,
    its form, whether the floor acted, the floor (flattening) radius in rho_tor,norm (``rho_floor``;
    ``rho_floor_domain`` on the diffusion domain's rho), every crossing of the floor by the model q before the floor
    (``rho_floor_crossings``, inner and outer), q_min before the floor (from
    :func:`.current.sawtooth.resistive_floor_steady` or :func:`.current.sawtooth.floor_q`), and the minimum and the
    axis value of the scored model q (after the corrections) with their differences from the floor (negative where the
    corrections pull q below it)."""
    rec = dict(floor_rec or {})
    rec.setdefault("core_relaxation", cfg.core_relaxation)
    rec.setdefault("q_floor", float(cfg.q_floor) if rec["core_relaxation"] == "q_floor" else None)
    if rec["core_relaxation"] == "q_floor":
        rec.setdefault("q_floor_form", cfg.q_floor_form)
    if "rho_floor" in rec and "rho_floor_domain" not in rec:
        # the floor works on the diffusion domain's rho (0..1 at the domain edge): rho_tor,norm = rho * rho_edge
        rec["rho_floor_domain"] = rec["rho_floor"]
        if rec.get("rho_floor_inner") is not None:
            rec["rho_floor_inner_domain"] = rec["rho_floor_inner"]
        for k in ("rho_floor", "rho_floor_inner", "rho_q_min_before", "rho_q_min_after"):
            if rec.get(k) is not None:
                rec[k] = float(rec[k]) * float(rho_edge)
        if rec.get("rho_floor_crossings"):
            rec["rho_floor_crossings"] = [dict(c, rho=float(c["rho"]) * float(rho_edge), rho_domain=float(c["rho"]))
                                          for c in rec["rho_floor_crossings"]]
    q = np.asarray(q_model, float)
    ok = np.isfinite(q)
    if np.any(ok):
        j = int(np.argmin(np.where(ok, q, np.inf)))
        rec.update(q_min_model=float(q[j]), rho_q_min_model=float(np.asarray(rho_tor, float)[j]),
                   q_axis_model=float(q[0]))
        if rec.get("q_floor") is not None:
            rec["q_min_model_minus_floor"] = float(q[j] - rec["q_floor"])
            rec["q_axis_model_minus_floor"] = float(q[0] - rec["q_floor"])
    return rec


#: Definition of the extrapolated axis q of the solved equilibrium recorded by :func:`floor_effect_record`
#: (``q_axis_extrapolated_eq``).  It is NOT an achieved (read-back) axis value: it cannot see structure inside the
#: window's inner edge, and the same estimator applied to the model q is off its own axis value by
#: ``q_axis_extrapolation_offset_model``.
Q_AXIS_EXTRAPOLATED_EQ_DEFINITION = (
    "regular-singular extrapolation q0 + q2 rho^2 + q4 rho^4 of the solved equilibrium's traced q, least squares on "
    "rho_tor,norm in the q check's axis window (coordinates.axis_q; the q check's axis.q0_eq): the axis value the "
    "surfaces in that window imply under that smooth form, not a trace of the innermost surface and not the "
    "achieved axis q")
#: Definition of the read-back of the core level (``q_axis_model_readback``): the model's own axis q of the
#: consistent state.
Q_AXIS_MODEL_READBACK_DEFINITION = (
    "the scored model q at the axis face of the consistent state (after the corrections; core_record's q_axis_model): "
    "the axis q of the current profile the solver was given; valid only where the q consistency check of that state "
    "passed (q_axis_model_readback_valid)")


def floor_effect_record(core, q_check=None, q_equilibrium=None, q_traced_innermost=None, q_solver_q0=None):
    """What the q floor did in one result -- a RECORD for the reader (nothing reads it; it cannot change a result).

    From the core record (:func:`core_record`; for a series slice the interval's record of
    :func:`.series.advance_q_floor`): the form, the branch (:data:`.current.sawtooth.FLOOR_BRANCHES`), whether it acted
    and met the floor, the outer and inner edges of the flattening region (rho_tor,norm), the change of the loop
    voltage (relative) and of l_i (``li3``, the full plasma, NaN in a series; ``li3_domain``, the diffusion domain)
    against the same model without the floor and the fraction of Ip the floor
    moved from inside its outer edge to outside it, with whether that record was computed (``effect_computed``; NaN
    values and ``effect_reason`` where it failed).  The reference: for a single slice the steady state of the same
    model sources (before the corrections) with the plain resistivity; for a series slice the same interval advanced
    from the same carried state without the floor (the floor's effect over this interval, not its history); a
    reference that is absent or None reads as the single-slice text.

    The READ-BACK of the core level: ``q_axis_model_readback``, the model's own axis q of the consistent state
    (:data:`Q_AXIS_MODEL_READBACK_DEFINITION`), with ``q_axis_model_readback_valid`` True only where the q consistency
    check of that state passed (``q_check["passed"] is True``; False, with the reason, otherwise).

    The EXTRAPOLATED axis q of the solved equilibrium: ``q_axis_extrapolated_eq``
    (:data:`Q_AXIS_EXTRAPOLATED_EQ_DEFINITION`: the q check's extrapolation ``q_check["axis"]["q0_eq"]``, with its
    window, degree and fit rms) -- the axis value the surfaces in the window imply under a smooth form, NOT an achieved
    axis value -- and the minimum of the equilibrium's traced q (``q_min_traced_eq``, from ``q_equilibrium``), next to
    the scored model q's axis value and minimum (after the corrections).  Named cross-checks (``q_axis_cross_checks``):
    the trace of the innermost surface (``q_traced_innermost``; the equilibrium's q at psi_N = 1e-6 -- a correct trace
    of the discrete field, which is invalid where the field's discrete minimum lies on or next to a mesh element edge,
    where the innermost contours of a kinked minimum give q falling without limit; elsewhere it can be flat and
    correct), the solver's ``q_0`` (``q_solver_q0``; q at psi_N = 0.02 by the solver's definition, not an axis value),
    the same extrapolation applied to the model q (``q_model_axis_extrapolated``) and the model q's own axis face value
    (``q_model_axis``).  ``q_axis_spread``: the largest minus the smallest of the extrapolated value, the innermost
    trace and the solver's ``q_0``; ``q_axis_extrapolation_offset_model``: the extrapolation of the model q minus the
    model's own axis value -- where the axis value is known, how far the estimator is from it."""
    core = core or {}
    ax = ((q_check or {}).get("axis") or {})
    qe = None if q_equilibrium is None else np.asarray(q_equilibrium, float)
    q_min_eq = float(np.min(qe[np.isfinite(qe)])) if qe is not None and np.any(np.isfinite(qe)) else None

    def _f(v):
        try:
            v = abs(float(v))
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    q_ext = _f(ax.get("q0_eq"))
    q_tr, q_s0 = _f(q_traced_innermost), _f(q_solver_q0)
    q_mx, q_ma = _f(ax.get("q0_model")), _f(core.get("q_axis_model"))
    trio = [v for v in (q_ext, q_tr, q_s0) if v is not None]
    passed = (q_check or {}).get("passed")
    rb_valid = passed is True and q_ma is not None
    rb_reason = (None if rb_valid else ("no q check" if q_check is None else
                                        "no model axis q" if q_ma is None else
                                        f"q check not passed ({(q_check or {}).get('max_rel', (q_check or {}).get('reason'))})"))
    out = dict(q_floor=core.get("q_floor"), q_floor_form=core.get("q_floor_form"),
               floor_rule=core.get("floor_rule"), floor_branch=core.get("floor_branch"),
               acted=core.get("acted"), q_floor_met=core.get("q_floor_met"),
               rho_floor=core.get("rho_floor"), rho_floor_inner=core.get("rho_floor_inner"),
               q_axis_model_readback=q_ma, q_axis_model_readback_valid=bool(rb_valid),
               q_axis_model_readback_invalid_reason=rb_reason,
               q_axis_model_readback_definition=Q_AXIS_MODEL_READBACK_DEFINITION,
               q_axis_extrapolated_eq=ax.get("q0_eq"),
               q_axis_extrapolated_eq_definition=Q_AXIS_EXTRAPOLATED_EQ_DEFINITION,
               q_axis_extrapolated_eq_window=ax.get("window"), q_axis_extrapolated_eq_degree=ax.get("degree"),
               q_axis_extrapolated_eq_fit_rms_rel=ax.get("fit_rms_rel_eq"), q_min_traced_eq=q_min_eq,
               q_axis_cross_checks=dict(q_traced_innermost=q_tr, q_solver_q0=q_s0, q_model_axis_extrapolated=q_mx,
                                        q_model_axis=q_ma),
               q_axis_spread=(max(trio) - min(trio)) if len(trio) >= 2 else None,
               q_axis_extrapolation_offset_model=(q_mx - q_ma) if q_mx is not None and q_ma is not None else None,
               q_axis_model=core.get("q_axis_model"), q_min_model=core.get("q_min_model"),
               d_v_loop_rel=core.get("d_v_loop_rel"), d_li3=core.get("d_li3"), d_li3_domain=core.get("d_li3_domain"),
               current_moved_outside_over_ip=core.get("current_moved_outside_over_ip"),
               effect_computed=core.get("floor_effect_computed"), effect_reason=core.get("floor_effect_reason"),
               reference=(core.get("floor_effect_reference") or
                          "steady state of the same model sources without the floor (before the corrections)"),
               note="record only: nothing reads it")
    return out


def core_vs_floor_record(core, correction_effect):
    """Whether the data pull the core below the q floor -- a RECORD for the reader, with no threshold and no verdict
    (nothing reads it: it cannot change a result).  From quantities the result already has: the scored model q at the
    axis and its minimum against the floor (``core``, :func:`core_record`; after the fitted corrections) and the
    signed change the corrections made to the current inside rho_tor,norm :data:`CORRECTION_CORE_RHO` and to the axis
    q (``correction_effect``, :func:`correction_core_effect`).  Negative q differences and a positive change of the
    core current mean the fit pulled the core below the floor."""
    core = core or {}
    cce = correction_effect or {}
    return dict(q_floor=core.get("q_floor"), floor_acted=core.get("acted"), q_floor_met=core.get("q_floor_met"),
                q_axis_model_minus_floor=core.get("q_axis_model_minus_floor"),
                q_min_model_minus_floor=core.get("q_min_model_minus_floor"),
                rho_q_min_model=core.get("rho_q_min_model"),
                core_rho=cce.get("core_rho"), d_current_inside_core_over_ip=cce.get("d_current_inside_core_over_ip"),
                d_q_axis_model=cce.get("d_q_axis_model"), correction_effect_computed=cce.get("computed"),
                note="record only: no threshold, nothing reads it")


# ============================================================== the floor's branch per assembly (recording only)
#: The open log of the q floor's branch at every assembly (a list), or None.  :func:`.fit.reconstruct` opens one per
#: fit and :func:`.series.run_series` one per slice (:func:`open_branch_log`); :func:`assemble` appends one entry per
#: assembly to the log that is open.  A RECORD only (:func:`branch_flip_record`): nothing reads it.
_ASSEMBLY_LOG = contextvars.ContextVar("kinrecon_floor_branch_log", default=None)


def open_branch_log():
    """Open a branch log (:data:`_ASSEMBLY_LOG`); returns ``(log, token)``; close it with :func:`close_branch_log`."""
    log = []
    return log, _ASSEMBLY_LOG.set(log)


def close_branch_log(token):
    """Close the branch log opened with ``token`` (the log open before it is open again)."""
    _ASSEMBLY_LOG.reset(token)


def current_branch_log():
    """The open branch log, or None."""
    return _ASSEMBLY_LOG.get()


def _log_assembly(core):
    """Append the floor branch of one assembly to the open branch log; never raises (a record must never end a fit
    or a series).  Entry: the branch (:data:`.current.sawtooth.FLOOR_BRANCHES`, None for another core treatment), the
    floor and the plain model state's axis q (single slice); for a series interval also the branch of every diffusion
    step and the smallest distance of the plain axis q from the floor over the steps."""
    try:
        log = _ASSEMBLY_LOG.get()
        if log is None:
            return
        c = core or {}
        e = dict(branch=c.get("floor_branch"), q_floor=c.get("q_floor"), q_axis_plain=c.get("q_axis_before"))
        if c.get("branch_steps") is not None:
            e.update(branch_steps=dict(c["branch_steps"]),
                     q_axis_plain_min_abs_dev=c.get("q_axis_model_steady_min_abs_dev_floor"))
        log.append(e)
    except Exception:  # noqa: BLE001 -- a record must never end a fit or a series
        pass


def branch_flip_record(log, jacobians=None, final_core=None):
    """Whether the q floor's branch (:func:`.current.sawtooth.floor_region`, decided on the plain model state on every
    assembly) changed inside one result -- a RECORD for the reader (nothing reads it; it cannot change a result).

    ``log``: the entries of :func:`_log_assembly` in call order (every pass of every consistent solve, every Jacobian
    column and trial, every lagged solve); ``jacobians``: per Jacobian the branches of its assemblies and of its center
    (``dict(label, branches, mixed)``); ``final_core``: the result's core record.  Returned: the number of assemblies,
    the count per branch, the number of branch changes in call order, whether any Jacobian had assemblies on different
    branches (and which: at most the first 20 listed, with the total count and whether the list is truncated), the smallest distance of the plain axis q from the floor seen, the final branch; for a series
    slice also the count of diffusion steps per branch over every pass (``diffusion_steps_per_branch``).  A fit run
    inside an open log (a series slice's filter fit) appends its assemblies to that log (:func:`.fit.reconstruct`,
    ``nested_fit``), so in a series slice ``n_assemblies``, ``assemblies_per_branch`` and
    ``diffusion_steps_per_branch`` count the filter fit's assemblies and interval advances as well as the slice's own
    consistent solve.  ``jacobians`` is not given for a slice log (its Jacobians are the filter fit's own;
    :func:`.series.run_series` adds their counts from the filter fit's record)."""
    ents = [e for e in (log or []) if isinstance(e, dict)]
    br = [e.get("branch") for e in ents]
    counts, steps, devs = {}, {}, []
    for b in br:
        counts[str(b)] = counts.get(str(b), 0) + 1
    for e in ents:
        for k, v in (e.get("branch_steps") or {}).items():
            steps[str(k)] = steps.get(str(k), 0) + int(v)
        qf, qa = e.get("q_floor"), e.get("q_axis_plain")
        if qf is not None and qa is not None and np.isfinite(qa):
            devs.append(abs(float(qa) - float(qf)))
        if e.get("q_axis_plain_min_abs_dev") is not None:
            devs.append(float(e["q_axis_plain_min_abs_dev"]))
    jac = [j for j in (jacobians or []) if isinstance(j, dict)]
    mixed = [j for j in jac if j.get("mixed")]
    out = dict(n_assemblies=len(ents), assemblies_per_branch=counts,
               n_branch_changes_in_call_order=sum(1 for a, b in zip(br[:-1], br[1:]) if a != b),
               branch_changed=bool(len(counts) > 1 or len(steps) > 1),
               n_jacobians=len(jac), n_jacobians_mixed_branches=len(mixed), jacobians_mixed_branches=mixed[:20],
               jacobians_mixed_branches_total=len(mixed), jacobians_mixed_branches_truncated=len(mixed) > 20,
               min_abs_plain_q_axis_minus_floor=min(devs) if devs else None,
               final_branch=(final_core or {}).get("floor_branch"),
               note="record only: nothing reads it")
    if steps:
        out["diffusion_steps_per_branch"] = steps
    return out


# ============================================================== the corrections' effect on the core (recording only)
def zero_corrections(params):
    """``params`` (full) with every current-correction amplitude set to zero."""
    p = full_params(params)
    p["a_corr"] = tuple(0.0 for _ in p["a_corr"])
    return p


def _enclosed_current(lag, asm, rho):
    """Current [A] inside rho_tor,norm = ``rho`` of an assembly's diffusion state: ``F_lcfs int <J.B>/(2 pi F^2) dV``
    over the state's cells (the definition of the assembly's current fractions), linear between cell faces."""
    rf = np.asarray(asm["state"].rho_faces, float)
    rc = np.asarray(asm["profiles_rho"]["rho"], float) / lag["rho_map"].rho_edge
    gc = lag["geom"].evaluate(rc, ["dvol_drho", "fpol"])
    dens = gc["dvol_drho"] / (2 * np.pi * gc["fpol"] ** 2) * np.diff(rf) * np.asarray(asm["profiles_rho"]["jdotb"], float)
    cum = lag["layer"].fpol_lcfs * np.concatenate(([0.0], np.cumsum(dens)))
    return float(np.interp(rho, rf * lag["rho_map"].rho_edge, cum))


def correction_core_effect(lag, asm, asm0, cfg, prior_sigmas, how):
    """What the current corrections do to the core of one result (a RECORD; nothing is changed or rejected).

    ``asm`` is the result's assembly (with its corrections) and ``asm0`` the assembly of the same state with the
    correction amplitudes set to zero, both on the geometry ``lag`` (no new equilibrium solve: the geometry is held);
    ``how`` says how ``asm0`` was made.  Recorded: each correction's amplitude (fraction of Ip), its prior sigma and
    their ratio; the change of the current inside rho_tor,norm = :data:`CORRECTION_CORE_RHO` (as a fraction of Ip
    and of the uncorrected value); the change of the axis current density -- ``<J.B>`` of the innermost cell of the
    diffusion state and the solver input ``<j_phi>`` at psi_N = 0, as fractions of their uncorrected values; the
    change of the model q (diffusion q on the held geometry) at the axis face and at rho_tor,norm =
    :data:`CORRECTION_Q_RHO`.  ``flags`` lists, in plain words, every amplitude above its prior sigma and an axis
    ``<J.B>`` change above :data:`CORRECTION_AXIS_REPORT_LEVEL` of its uncorrected value -- a REPORTING level, not an
    acceptance criterion."""
    amps = [float(a) for a in full_params(asm["params"])["a_corr"]]
    rc = CORRECTION_CORE_RHO
    ip = float(asm["ip"])
    i1, i0 = _enclosed_current(lag, asm, rc), _enclosed_current(lag, asm0, rc)
    jb1, jb0 = (float(np.asarray(a["profiles_rho"]["jdotb"], float)[0]) for a in (asm, asm0))
    jp1, jp0 = (float(np.asarray(a.get("jphi_assembled", a["jphi"]), float)[0]) for a in (asm, asm0))
    q1, q0 = (np.asarray(a["q_model"]["q"], float) for a in (asm, asm0))
    r1, r0 = (np.asarray(a["q_model"]["rho"], float) for a in (asm, asm0))
    qr1, qr0 = float(np.interp(CORRECTION_Q_RHO, r1, q1)), float(np.interp(CORRECTION_Q_RHO, r0, q0))
    corr = []
    flags = []
    for c, a, s in zip(cfg.corr_centers_rho, amps, prior_sigmas):
        ratio = (abs(a) / float(s)) if (s is not None and float(s) > 0) else None
        corr.append(dict(center_rho=float(c), amplitude_fraction_of_ip=a,
                         prior_sigma=None if s is None else float(s), ratio_to_prior_sigma=ratio))
        if ratio is not None and ratio > 1.0:
            flags.append(f"correction at rho {c:g}: amplitude {a:+.4f} of Ip is {ratio:.2f} prior sigmas")
    d_axis = (jb1 - jb0) / jb0 if jb0 != 0.0 else float("nan")
    if not abs(d_axis) <= CORRECTION_AXIS_REPORT_LEVEL:
        flags.append(f"the corrections change the axis <J.B> by {100 * d_axis:+.1f} % (reporting level "
                     f"{100 * CORRECTION_AXIS_REPORT_LEVEL:g} %)")
    return dict(computed=True, how=how, basis=dict(corr_return=cfg.corr_return,
                                                   corr_return_width_ratio=float(cfg.corr_return_width_ratio),
                                                   corr_width_rho=float(cfg.corr_width_rho)),
                corrections=corr, ip_A=ip,
                core_rho=rc, current_inside_core_A=dict(corrected=i1, uncorrected=i0),
                d_current_inside_core_over_ip=(i1 - i0) / ip,
                d_current_inside_core_over_uncorrected=((i1 - i0) / i0 if i0 != 0.0 else float("nan")),
                axis_jdotb=dict(corrected=jb1, uncorrected=jb0, rho=float(np.asarray(asm["profiles_rho"]["rho"])[0])),
                d_axis_jdotb_rel=d_axis,
                axis_jphi=dict(corrected=jp1, uncorrected=jp0),
                d_axis_jphi_rel=((jp1 - jp0) / jp0 if jp0 != 0.0 else float("nan")),
                q_axis_model=dict(corrected=float(q1[0]), uncorrected=float(q0[0])), d_q_axis_model=float(q1[0] - q0[0]),
                q_rho=CORRECTION_Q_RHO, q_at_rho=dict(corrected=qr1, uncorrected=qr0), d_q_at_rho=qr1 - qr0,
                reporting_level_axis=CORRECTION_AXIS_REPORT_LEVEL, flags=flags, flagged=bool(flags),
                note="recording only: the reporting level is not an acceptance criterion")


def steady_correction_effect(lag, asm, kin, src, cfg, prior_sigmas, z0=None):
    """:func:`correction_core_effect` of a single-slice result (corrections in the steady state / periodic orbit):
    the current model re-evaluated on the result's geometry with the amplitudes set to zero (the core treatment the
    result used: the q-floor state, the steady state, or the opt-in periodic orbit or measured-crash run from the
    orbit continuation ``z0``).  A
    re-evaluation that fails is recorded with its reason."""
    how = ("steady re-evaluation: the current model (the core treatment the fit used: the q-floor state, the plain "
           "steady state or an opt-in sawtooth state) re-evaluated on the fitted equilibrium's geometry with the "
           "correction amplitudes set to zero; no new equilibrium solve")
    p = full_params(asm["params"])
    if not any(a != 0.0 for a in p["a_corr"]):
        return correction_core_effect(lag, asm, asm, cfg, prior_sigmas, how + " (all amplitudes are zero)")
    try:
        asm0 = assemble(lag, kin, src, zero_corrections(p), cfg, asm["ip"], z0=z0)
    except (ModelFailure, ValueError, RuntimeError) as exc:
        return dict(computed=False, how=how, reason=f"{type(exc).__name__}: {exc}"[:300],
                    amplitudes=list(p["a_corr"]), prior_sigmas=list(prior_sigmas))
    return correction_core_effect(lag, asm, asm0, cfg, prior_sigmas, how)


# ============================================================== pass-consistency loop
@dataclass
class ConsistentResult:
    converged: bool
    status: str
    passes: int
    assembly: Optional[Dict[str, Any]]
    lag: Optional[Dict[str, Any]]
    solve: Any
    record: Dict[str, Any]
    wall_s: float


def _state_change(a, b):
    try:
        ax = float(np.hypot(a["R_axis"] - b["R_axis"], a["Z_axis"] - b["Z_axis"]))
        rel = {k: abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in ("l_i", "q_0", "Ip")}
    except (KeyError, TypeError):
        return np.inf, {"missing": True}
    return ax, rel


def q_check(lag, asm, stats, cfg):
    """q consistency of the solved equilibrium (``lag``: its traced q on the physics grid) against the
    model q of the assembly that produced it (:func:`.coordinates.q_consistency`).  The two sides'
    extrapolated axis values are always computed and recorded (``axis``), but they enter the pass/fail
    verdict only with ``QCheckSettings(test_axis=True)``; the default (``test_axis=False``) tests the
    interior only, and ``axis["tested"]`` says which.  ``q0_eq_axis`` (the traced q at psi_N =
    :data:`Q_AXIS_PSIN`) and ``q0_model_axis`` (the model's innermost face value) are the raw near-axis
    values, kept as diagnostics only: neither is an axis estimate the test uses."""
    ph = lag["phys"]
    out = q_consistency(ph["rho"], ph["q"], asm["q_model"]["rho"], asm["q_model"]["q"], cfg.q_check)
    out["q0_eq_axis"] = ph.get("q_axis", np.nan)
    out["q0_model_axis"] = float(np.asarray(asm["q_model"]["q"])[0])
    out["q0_axis_rel"] = abs(out["q0_eq_axis"] - out["q0_model_axis"]) / out["q0_model_axis"]
    return out


def _knot_input(asm, x_in_cmp):
    """TokaMaker input on the knots for a (possibly Anderson-mixed) comparison-grid profile: the
    assembled knot values plus the mixing correction interpolated in rho."""
    d = np.asarray(x_in_cmp, float) - asm["jphi_cmp"]
    if not np.any(d):
        return asm["jphi"]
    return asm["jphi"] + PchipInterpolator(asm["rho_cmp"], d)(np.clip(asm["rho_knots"], 0.0, 1.0))


def consistent_solve(fm, kin, src, params, cfg, ip=None, start=None, z0=None, label="consistent",
                     initial_profiles=None, state_fn=None):
    """Iterate model evaluation and forward solve to full consistency (module docstring).

    ``fm`` is a :class:`.forward.ForwardModel`; ``start`` an optional snapshot (or psi array) to
    start from (default: the solver's current state); ``z0`` an orbit continuation vector.
    ``initial_profiles`` = ``(x, jphi, p)``: for the FIRST solve of a case, whose starting psi is
    only an initial guess (no solved equilibrium to evaluate the model on yet), one forward solve
    with these profiles (e.g. the reference equilibrium's <j_phi> and the kinetic pressure) is made
    before the loop ("pass -1"; recorded); it is a starting state, not part of the answer.
    ``state_fn(lag) -> CurrentState`` supplies the current state on each pass's geometry (the
    time-dependent path); without it the steady state / periodic orbit is used.
    Failures (a model failure, a solve that does not converge or verify, no consistency within
    ``max_passes``, or a q-consistency failure once the profiles have converged) are returned with
    ``converged=False`` and the reason; the solver is then left in its last state (the caller
    restores a snapshot)."""
    t0 = time.perf_counter()
    ipv = ip_target(fm.ip_measured if ip is None else ip, params)
    if start is not None:
        if isinstance(start, np.ndarray):
            fm.set_psi(start)
        else:
            fm.restore(start)
    passes = []
    prev_j = prev_stats = None
    last_probe_change = None
    asm = lag = res = None
    used = None
    if initial_profiles is not None:
        x0, j0, p0 = (np.asarray(v, float) for v in initial_profiles)
        r0 = fm.solve((x0, j0), (x0, p0), ipv, init="keep", label=f"{label}:initial", lock_case=False)
        passes.append(dict(k=-1, initial_profiles=True, solve_status=r0.status,
                           its=r0.record["attempts"][-1].get("its"), solve_wall_s=r0.wall_s))
        if not r0.converged:
            return ConsistentResult(False, "initial_solve_failed", 0, None, None, r0,
                                    dict(passes=passes, reason=r0.status), round(time.perf_counter() - t0, 3))
        try:
            fm.update_probe_correction()
        except RuntimeError as exc:
            return ConsistentResult(False, "probe_correction_failed", 0, None, None, r0,
                                    dict(passes=passes, reason=str(exc)), round(time.perf_counter() - t0, 3))
    if cfg.pass_acceleration not in ("anderson", "none"):
        raise ValueError("pass_acceleration must be 'anderson' or 'none'")
    hx, hf = [], []          # Anderson histories: inputs and residuals G(x) - x on the comparison grid
    x_prev = None
    prev_mixed = False       # whether the input of the previous pass was an Anderson mixture
    qrec = None
    # robustness option (module docstring; off by default): the under-relaxation factor in effect (None = none)
    beta = None if getattr(cfg, "pass_relaxation", None) is None else float(cfg.pass_relaxation)
    last_f = None            # the previous pass residual (for residual_cos)

    def _opts():
        """The loop record of the robustness option (nothing when it is off)."""
        if beta is None:
            return {}
        return dict(robustness=dict(pass_relaxation=beta, pass_relaxation_note=PASS_RELAXATION_NOTE))
    for k in range(int(cfg.max_passes)):
        try:
            lag_k = evaluate_geometry(fm.gs, kin, cfg)
            st_k = call_state_fn(state_fn, lag_k, params) if state_fn is not None else None
            asm_k = assemble(lag_k, kin, src, params, cfg, ipv, z0=z0, state=st_k)
        except (ModelFailure, ValueError, RuntimeError) as exc:
            return ConsistentResult(False, "model_failure", k, asm, lag, res,
                                    dict(passes=passes, reason=f"{type(exc).__name__}: {exc}", q_check=qrec, **_opts()),
                                    round(time.perf_counter() - t0, 3))
        z0 = asm_k["orbit_z"]
        rec = dict(k=k, geometry_wall_s=lag_k["wall_s"], assembly_wall_s=asm_k["wall_s"],
                   fractions=asm_k["fractions"], sawtooth_active=asm_k["sawtooth"].get("active"),
                   n_knots=asm_k["knots"].get("n"))
        if asm is not None and res is not None:
            qrec = q_check(lag_k, asm, res.stats, cfg)
            rec.update(q_max_rel=qrec.get("max_rel"), q_passed=qrec.get("passed"))
        if prev_j is not None:
            dj = float(np.max(np.abs(asm_k["jphi_cmp"] - prev_j)) / np.max(np.abs(asm_k["jphi_cmp"])))
            ax, rel = _state_change(res.stats, prev_stats) if prev_stats is not None else (np.inf, {})
            rec.update(dj=dj, d_axis_m=ax, d_rel=rel, d_probe_sigma=last_probe_change)
            ok_state = prev_stats is not None and ax < cfg.state_axis_m and all(v < cfg.state_rel for v in rel.values())
            ok_probe = last_probe_change is None or last_probe_change < cfg.pass_tol
            if dj < cfg.pass_tol and ok_state and ok_probe:
                passes.append(rec)
                crit = dict(pass_tol=cfg.pass_tol, axis_m=cfg.state_axis_m, rel=cfg.state_rel,
                            q_check=cfg.q_check.as_dict(), comparison_grid=f"{cfg.n_cmp} points uniform in rho_tor")
                q_ok = bool(qrec and qrec.get("passed"))
                if cfg.q_check.required and not q_ok:
                    return ConsistentResult(False, "q_inconsistent", k, used, lag, res,
                                            dict(passes=passes, criterion=crit, q_check=qrec, **_opts(),
                                                 reason=f"profiles consistent but q check failed: "
                                                        f"{(qrec or {}).get('max_rel', (qrec or {}).get('reason'))}"),
                                            round(time.perf_counter() - t0, 3))
                return ConsistentResult(True, "consistent", k, used, lag, res,
                                        dict(passes=passes, criterion=crit, q_check=qrec,
                                             assembly_on_final_state=dict(dj=dj), **_opts()),
                                        round(time.perf_counter() - t0, 3))
        prev_stats = None if res is None else res.stats
        x_in, mixed = asm_k["jphi_cmp"], False
        relaxed = False          # robustness options: the input is an under-relaxed (damped) step
        if x_prev is not None:
            f_k = asm_k["jphi_cmp"] - x_prev
            if last_f is not None:
                nrm = float(np.linalg.norm(f_k) * np.linalg.norm(last_f))
                rec["residual_cos"] = float(f_k @ last_f) / nrm if nrm > 0 else None
            last_f = f_k
            hx.append(x_prev)
            hf.append(f_k)
            del hx[:-(cfg.anderson_depth + 1)], hf[:-(cfg.anderson_depth + 1)]
            if (cfg.pass_acceleration == "anderson" and prev_mixed and len(hf) >= 2
                    and np.max(np.abs(hf[-1])) > np.max(np.abs(hf[-2]))):
                # safeguard: the Anderson-mixed input of the previous pass made the residual grow -> restart the
                # history; this pass takes the plain input
                del hx[:-1], hf[:-1]
                rec["anderson_restart"] = True
            if cfg.pass_acceleration == "anderson" and len(hf) >= 2:
                dF = np.column_stack([hf[i + 1] - hf[i] for i in range(len(hf) - 1)])
                dX = np.column_stack([hx[i + 1] - hx[i] for i in range(len(hx) - 1)])
                gam = np.linalg.lstsq(dF, hf[-1], rcond=None)[0]
                if beta is None:
                    cand = hx[-1] + hf[-1] - (dX + dF) @ gam
                else:
                    cand = hx[-1] + beta * hf[-1] - (dX + beta * dF) @ gam
                if np.all(np.isfinite(cand)):
                    x_in, mixed = cand, True
            if beta is not None and not mixed:
                x_in, relaxed = x_prev + beta * hf[-1], True
            if beta is not None:
                rec["relaxation"] = beta
        moved = mixed or relaxed     # the input is not the assembled profile
        snap_k = fm.snapshot() if moved else None
        jk = _knot_input(asm_k, x_in) if moved else asm_k["jphi"]
        res = fm.solve((asm_k["x"], jk), (asm_k["x"], asm_k["p"]), ipv, init="keep", label=f"{label}:pass{k}")
        if moved and not res.converged:
            rec["anderson_fallback" if mixed else "relaxation_fallback"] = res.status
            fm.restore(snap_k)
            x_in, mixed, relaxed, moved = asm_k["jphi_cmp"], False, False, False
            jk = asm_k["jphi"]
            hx.clear()
            hf.clear()
            res = fm.solve((asm_k["x"], jk), (asm_k["x"], asm_k["p"]), ipv, init="keep",
                           label=f"{label}:pass{k}:plain")
        rec.update(solve_status=res.status, its=res.record["attempts"][-1].get("its"), solve_wall_s=res.wall_s,
                   anderson_mixed=mixed)
        if relaxed:
            rec["relaxed_plain_step"] = True
        passes.append(rec)
        if not res.converged:
            st = ("limit_cycle_untapered_lock" if res.status == "failed_limit_cycle_untapered_lock"
                  or rec.get("anderson_fallback") == "failed_limit_cycle_untapered_lock"
                  or rec.get("relaxation_fallback") == "failed_limit_cycle_untapered_lock" else "solve_failed")
            return ConsistentResult(False, st, k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=res.status, q_check=qrec, **_opts()),
                                    round(time.perf_counter() - t0, 3))
        try:
            last_probe_change = fm.update_probe_correction()
        except RuntimeError as exc:
            return ConsistentResult(False, "probe_correction_failed", k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=str(exc), **_opts()), round(time.perf_counter() - t0, 3))
        prev_j = x_in
        x_prev = x_in
        prev_mixed = mixed
        used = dict(asm_k, jphi=jk, jphi_assembled=asm_k["jphi"], jphi_cmp_used=x_in) if moved else asm_k
        lag, asm = lag_k, asm_k
    return ConsistentResult(False, "max_passes", int(cfg.max_passes), used, lag, res,
                            dict(passes=passes, reason=f"not consistent within {cfg.max_passes} passes", q_check=qrec,
                                 **_opts()),
                            round(time.perf_counter() - t0, 3))


def lagged_solve(fm, lag, kin, src, params, cfg, ip=None, start=None, z0=None, label="column", state=None):
    """One forward solve with the geometry-dependent inputs lagged from ``lag``.  Returns
    ``(SolveResult or None, assembly or None, reason)``."""
    ipv = ip_target(fm.ip_measured if ip is None else ip, params)
    if start is not None:
        fm.restore(start)
    try:
        asm = assemble(lag, kin, src, params, cfg, ipv, z0=z0, state=state)
    except (ModelFailure, ValueError, RuntimeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    res = fm.solve((asm["x"], asm["jphi"]), (asm["x"], asm["p"]), ipv, init="keep", label=label)
    return res, asm, res.status
