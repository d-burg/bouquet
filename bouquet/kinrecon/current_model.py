r"""Current-profile model of the kinetic reconstruction and its pass-consistency loop.

Physics (all in Python, iterated with the geometry):

.. code-block:: text

    <J.B>      = <J.B>_ohm + s_bs <J.B>_BS + s_cd <J.B>_CD (+ sum_k a_k g_k)                      (1)
    <J.B>_ohm  = sigma_neo <E.B>,   <E.B> from the steady state (or the sawtooth periodic state)
                 of the flux-diffusion equation with I(LCFS) = Ip                                   (2)
    <J_phi>    = kappa <J.B> - (dp/dpsi) (<R> - F^2 <1/R> / <B^2>),  kappa = F <1/R> / <B^2>        (3)
    p          = p_thermal(n_e, T_e, n_i, T_i, n_imp) + s_pf p_fast                                (4)

* (2): :class:`.current.diffusion.CurrentDiffusion` in steady-state mode on the geometry of the
  CURRENT equilibrium (:func:`.current.geometry.from_tokamaker`), domain to psi_N = 0.995 with the
  lumped outer layer; ``sigma_neo`` from :func:`.current.conductivity.neoclassical_conductivity`
  (Redl 2021 by default) on the kinetic profiles.  With the sawtooth model on, the periodic orbit
  of "diffuse for one period, reconnect with fraction f" (:func:`.current.sawtooth.solve_periodic_orbit`)
  replaces the steady state and its cycle average is used (the window average over an unknown
  phase, averaged over phases, equals the cycle average); with measured crash times the
  time-dependent slice run (:func:`.current.sawtooth.simulate_slice`) averaged over the
  measurement window is used instead.  The sawtooth parameters are the strength
  ``Lambda = -ln(1 - 0.999 f) nu`` (nu the mean crash rate) and ``f``.
* ``<J.B>_BS``: Redl 2021 (:mod:`.current.bootstrap`), multiplier ``s_bs = 1`` by default.
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
current pass's equilibrium, plus the domain-edge rule's outer-layer knots).

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
criterion are unchanged, and a mixed input that does not solve falls back to the plain one.

A supplied current state (``state_fn``; the time-dependent path of :mod:`.series`) replaces the
steady state / periodic orbit: the profiles are then those of that state on the current geometry,
and the corrections act as the zero-net-current steady response ``sum_k a_k psi_k`` added to it.

:func:`lagged_solve` (geometry-dependent inputs lagged from a center) is kept for callers that
want it; the fit (:mod:`.fit`) now takes its Jacobian columns from consistent solves.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .current import sawtooth as _saw
from .current.bootstrap import EC, redl_jdotb
from .current.conductivity import neoclassical_conductivity
from .current.diffusion import CurrentDiffusion, CurrentState, IpBoundary, OuterLayer
from .current.diffusion import _boundary_gradient_weights as _diff_bgw
from .current.geometry import from_tokamaker
from .coordinates import KnotSettings, QCheckSettings, q_consistency, rho_map_from_geometry, solver_knots
from .pressure import pressure_gradient

#: The previous default knots (129 uniform in psi_N), kept by name: ``knot_settings=KnotSettings(
#: rule="psin_uniform")`` reproduces them up to the domain-edge rule; ``knots=DEFAULT_KNOTS``
#: reproduces them exactly (fixed explicit knots, no edge rule).
DEFAULT_KNOTS = np.linspace(0.0, 1.0, 129)
#: psi_N at which the solved equilibrium's axis q is taken for the q-consistency check (traced q limit).
Q_AXIS_PSIN = 1.0e-6


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
    transport-code form ``<J_phi R0/R>`` (ONETWO; ``I = int J dV / (2 pi R0)``), converted on the
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
    ion_collisionality: str = "spike"        # for the bootstrap (current.bootstrap)
    pass_acceleration: str = "anderson"      # "anderson" (depth anderson_depth) or "none"
    anderson_depth: int = 3
    sawtooth: bool = True
    sawtooth_nsteps: int = 40
    sawtooth_sheet_width: float = _saw.DEFAULT_SHEET_WIDTH
    crash_times: Optional[Sequence[float]] = None   # measured crash times [s] (None: periodic orbit)
    window: Optional[Sequence[float]] = None        # measurement window [t0, t1] [s] (with crash times)
    history: float = _saw.DEFAULT_HISTORY
    corr_centers_rho: Sequence[float] = (0.2, 0.5, 0.8)   # current-correction Gaussians, centers in rho_tor,norm
    corr_width_rho: float = 0.1                           # their width (standard deviation) in rho_tor,norm

    def as_dict(self):
        d = asdict(self)
        if self.knots is None:
            d["knots"] = dict(generated_per_pass=True, **self.knot_settings.as_dict())
        else:
            d["knots"] = dict(rule="explicit fixed psi_N knots", n=len(self.knots),
                              min_spacing=float(np.min(np.diff(self.knots))))
        return d


#: Model parameters and their defaults (the fit decides which are free).
DEFAULT_PARAMS = dict(log_strength=None, f=0.37, s_bs=1.0, s_cd=1.0, s_pfast=1.0, a_corr=(0.0, 0.0, 0.0))


def full_params(params):
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    p["a_corr"] = tuple(float(v) for v in p["a_corr"])
    return p


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
    # get_stats()['q_0'] is a different estimate (5-11 % above this limit on the public fixture)
    try:
        phys["q_axis"] = float(abs(np.asarray(gs.get_q(psi=np.array([Q_AXIS_PSIN]))[1], float)[0]))
    except Exception:  # noqa: BLE001
        phys["q_axis"] = float("nan")
    k = kin.profiles_at(psi_n=ge, rho=rho_ge)
    sig, sparts = neoclassical_conductivity(k["ne"], k["te"], k["zeff"], phys["ft"], phys["q"], phys["r_geo"],
                                            phys["eps"], model=cfg.conductivity_model,
                                            ln_lambda_model=cfg.ln_lambda_model)
    jbs, bparts = redl_jdotb(ge, k["te"], k["ti"], k["ne"], k["ni"], k["zeff"], phys["q"], phys["r_geo"],
                             phys["eps"], phys["ft"], F, dpsi, ion_collisionality=cfg.ion_collisionality)
    return dict(geom=geom, layer=layer, psi_n_edge=c, to_rho=to_rho, to_psin=to_psin, rho_map=rmap, phys=phys,
                dpsi=dpsi, sigma=np.asarray(sig, float), jbs=np.asarray(jbs, float), kin=k,
                kinetics_coordinate=getattr(kin, "coordinate", "psi_n"),
                bootstrap_settings=bparts["settings"], wall_s=round(time.perf_counter() - t0, 4),
                fsa_psi_axis=float(fsa["psi_axis"]), fsa_psi_boundary=float(fsa["psi_boundary"]))


def _interp(x, y, xq, log=False):
    f = PchipInterpolator(x, np.log(y) if log else y, extrapolate=True)
    v = f(np.asarray(xq, float))
    return np.exp(v) if log else v


def correction_shapes(lag, cfg, ip):
    """Unit current-correction shapes ``g_k`` on the physics grid (``<J.B>`` [A T/m^2] per unit
    amplitude): Gaussians in rho_tor,norm, centers ``corr_centers_rho``, width ``corr_width_rho``,
    each normalized so that it carries ``I = Ip`` (``I = F_lcfs int <J.B>/(2 pi F^2) dV`` over the
    domain and the outer layer).  An amplitude ``a_k`` is therefore a fraction of Ip."""
    geom, layer = lag["geom"], lag["layer"]
    rmap = lag["rho_map"]
    ge, rho_ge = lag["phys"]["psi_n"], lag["phys"]["rho"]
    gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
    rho_std_g = geom.rho * rmap.rho_edge
    rho_l = rmap.rho_of_psin(layer.psi_n)
    out = []
    for rc0 in cfg.corr_centers_rho:
        w = float(cfg.corr_width_rho)
        dom = np.trapezoid(gc["dvol_drho"] * np.exp(-0.5 * ((rho_std_g - rc0) / w) ** 2)
                           / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
        lay = layer.integral(layer.source_density(np.exp(-0.5 * ((rho_l - rc0) / w) ** 2)))
        norm = abs(float(ip)) / (layer.fpol_lcfs * (dom + lay))
        out.append(np.exp(-0.5 * ((rho_ge - rc0) / w) ** 2) * norm)
    return out


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
    E_par).  Returns cell values and the boundary value to add to a state's psi."""
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
    ``state`` they enter the steady state / periodic orbit (the ohmic part keeps Ip).  With ``state``
    (a :class:`.current.diffusion.CurrentState` on ``n_cells`` cells of the domain rho, the
    time-dependent path) that state replaces the steady state / orbit and the corrections are added
    to it as their zero-net-current steady response (:func:`correction_response`), which by
    linearity is what they add to a steady state.

    Returns a dict with the knots ``x`` (psi_N) and their ``rho_knots``, ``jphi`` [A/m^2] and the
    full-frame pressure ``p`` [Pa] on them, ``jphi_cmp`` on the fixed comparison grid ``rho_cmp``, the
    parts of <J.B> and the current fractions, the model q on rho (``q_model``), the sawtooth record,
    the diffusion ``state`` used, and ``orbit_z`` (continuation for the next orbit)."""
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
    if any(a != 0.0 for a in p["a_corr"]):
        if len(p["a_corr"]) != len(cfg.corr_centers_rho):
            raise ValueError("a_corr needs one amplitude per correction center")
        shapes = correction_shapes(lag, cfg, ipv)
        for a, g, rc0 in zip(p["a_corr"], shapes, cfg.corr_centers_rho):
            jcorr = jcorr + a * g
            corr_shapes.append(dict(center_rho_tor=rc0, width_rho_tor=cfg.corr_width_rho, amplitude_fraction_of_ip=a))
    jni_ge = jbs + jcd + jcorr
    sol = build_solver(lag, jni_ge, ipv)
    lx = layer.psi_n
    sig_l = _interp(ge, lag["sigma"], lx, log=True)
    rf = CurrentState.uniform_faces(cfg.n_cells)
    saw = dict(active=False)
    orbit_z = None
    if state is not None:
        if state.psi.size != cfg.n_cells:
            raise ValueError("the supplied state must be on n_cells cells")
        rf = state.rho_faces
        st_use = state
        if shapes is not None:
            dpsi, dpb = correction_response(lag, shapes, p["a_corr"], rf)
            st_use = CurrentState(rf, state.psi + dpsi, state.psi_boundary + dpb, state.time, dict(state.meta))
        state = st_use
        saw.update(mode="supplied state (time-dependent path)")
    elif cfg.sawtooth and p["log_strength"] is not None:
        lam = float(np.exp(p["log_strength"]))
        rate = float(_saw.rate_from_strength(lam, p["f"]))
        period = 1.0 / rate
        if cfg.crash_times is not None:
            if cfg.window is None:
                raise ValueError("crash_times need the measurement window")
            sched = _saw.schedule_for_slice(cfg.crash_times, cfg.window, history=cfg.history)
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
    else:
        state = sol.steady_state(rf)
        saw.update(mode="off (steady state)" if not cfg.sawtooth else "off (log_strength None)")
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
    out = dict(x=xk, rho_knots=rmap.rho_of_psin(xk), jphi=jphi, p=pk, jdotb=jb, jdotb_ni=jb_ni, kappa=kap,
               jphi_pressure_term=jpd, rho_cmp=rho_cmp, jphi_cmp=jphi_cmp, knots=krec,
               currents_A=cur, current_sum_A=tot, ip=ipv,
               fractions={kk: v / ipv for kk, v in cur.items()},
               v_loop=float(prof["v_loop_boundary"]), q_diffusion=dict(rho=prof["rho_faces"], q=prof["q"]),
               q_model=dict(rho=rho_faces_std, q=prof["q"]),
               profiles_rho=dict(rho=prof["rho_cells"] * rmap.rho_edge, jdotb=prof["jdotb"],
                                 **{f"jdotb_{kk}": v for kk, v in parts_c.items()}),
               li3_diffusion=float(prof["li3"]), outer_layer_current_A=float(prof["outer_layer_current"]),
               sawtooth=saw, orbit_z=orbit_z, corrections=corr_shapes, params=p, state=state,
               p_sep=float(pk[-1]), pax=float(pk[0] - pk[-1]), wall_s=round(time.perf_counter() - t0, 4))
    return out


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
    model q of the assembly that produced it (:func:`.coordinates.q_consistency`); the two axis values
    (the equilibrium's traced q at psi_N = :data:`Q_AXIS_PSIN`, the model's extrapolated q0) are
    reported, not tested."""
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
    ipv = abs(fm.ip_measured if ip is None else float(ip))
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
    qrec = None
    for k in range(int(cfg.max_passes)):
        try:
            lag_k = evaluate_geometry(fm.gs, kin, cfg)
            st_k = state_fn(lag_k) if state_fn is not None else None
            asm_k = assemble(lag_k, kin, src, params, cfg, ipv, z0=z0, state=st_k)
        except (ModelFailure, ValueError, RuntimeError) as exc:
            return ConsistentResult(False, "model_failure", k, asm, lag, res,
                                    dict(passes=passes, reason=f"{type(exc).__name__}: {exc}", q_check=qrec),
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
                                            dict(passes=passes, criterion=crit, q_check=qrec,
                                                 reason=f"profiles consistent but q check failed: "
                                                        f"{(qrec or {}).get('max_rel', (qrec or {}).get('reason'))}"),
                                            round(time.perf_counter() - t0, 3))
                return ConsistentResult(True, "consistent", k, used, lag, res,
                                        dict(passes=passes, criterion=crit, q_check=qrec,
                                             assembly_on_final_state=dict(dj=dj)),
                                        round(time.perf_counter() - t0, 3))
        prev_stats = None if res is None else res.stats
        x_in, mixed = asm_k["jphi_cmp"], False
        if x_prev is not None:
            hx.append(x_prev)
            hf.append(asm_k["jphi_cmp"] - x_prev)
            del hx[:-(cfg.anderson_depth + 1)], hf[:-(cfg.anderson_depth + 1)]
            if cfg.pass_acceleration == "anderson" and len(hf) >= 2:
                dF = np.column_stack([hf[i + 1] - hf[i] for i in range(len(hf) - 1)])
                dX = np.column_stack([hx[i + 1] - hx[i] for i in range(len(hx) - 1)])
                gam = np.linalg.lstsq(dF, hf[-1], rcond=None)[0]
                cand = hx[-1] + hf[-1] - (dX + dF) @ gam
                if np.all(np.isfinite(cand)):
                    x_in, mixed = cand, True
        snap_k = fm.snapshot() if mixed else None
        jk = _knot_input(asm_k, x_in) if mixed else asm_k["jphi"]
        res = fm.solve((asm_k["x"], jk), (asm_k["x"], asm_k["p"]), ipv, init="keep", label=f"{label}:pass{k}")
        if mixed and not res.converged:
            rec["anderson_fallback"] = res.status
            fm.restore(snap_k)
            x_in, mixed = asm_k["jphi_cmp"], False
            jk = asm_k["jphi"]
            hx.clear()
            hf.clear()
            res = fm.solve((asm_k["x"], jk), (asm_k["x"], asm_k["p"]), ipv, init="keep",
                           label=f"{label}:pass{k}:plain")
        rec.update(solve_status=res.status, its=res.record["attempts"][-1].get("its"), solve_wall_s=res.wall_s,
                   anderson_mixed=mixed)
        passes.append(rec)
        if not res.converged:
            st = ("limit_cycle_untapered_lock" if res.status == "failed_limit_cycle_untapered_lock"
                  or rec.get("anderson_fallback") == "failed_limit_cycle_untapered_lock" else "solve_failed")
            return ConsistentResult(False, st, k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=res.status, q_check=qrec),
                                    round(time.perf_counter() - t0, 3))
        try:
            last_probe_change = fm.update_probe_correction()
        except RuntimeError as exc:
            return ConsistentResult(False, "probe_correction_failed", k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=str(exc)), round(time.perf_counter() - t0, 3))
        prev_j = x_in
        x_prev = x_in
        used = dict(asm_k, jphi=jk, jphi_assembled=asm_k["jphi"], jphi_cmp_used=x_in) if mixed else asm_k
        lag, asm = lag_k, asm_k
    return ConsistentResult(False, "max_passes", int(cfg.max_passes), used, lag, res,
                            dict(passes=passes, reason=f"not consistent within {cfg.max_passes} passes", q_check=qrec),
                            round(time.perf_counter() - t0, 3))


def lagged_solve(fm, lag, kin, src, params, cfg, ip=None, start=None, z0=None, label="column", state=None):
    """One forward solve with the geometry-dependent inputs lagged from ``lag``.  Returns
    ``(SolveResult or None, assembly or None, reason)``."""
    ipv = abs(fm.ip_measured if ip is None else float(ip))
    if start is not None:
        fm.restore(start)
    try:
        asm = assemble(lag, kin, src, params, cfg, ipv, z0=z0, state=state)
    except (ModelFailure, ValueError, RuntimeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    res = fm.solve((asm["x"], asm["jphi"]), (asm["x"], asm["p"]), ipv, init="keep", label=label)
    return res, asm, res.status
