r"""Current-profile model of the kinetic reconstruction and its pass-consistency loop.

Physics (all in Python, iterated with the geometry; project decisions D12, D15, R08, R16, R29):

.. code-block:: text

    <J.B>      = <J.B>_ohm + s_bs <J.B>_BS + s_cd <J.B>_CD (+ sum_k a_k g_k)                      (1)
    <J.B>_ohm  = sigma_neo <E.B>,   <E.B> from the steady state (or the sawtooth periodic state)
                 of the flux-diffusion equation with I(LCFS) = Ip                                   (2)
    <J_phi>    = kappa <J.B> - (dp/dpsi) (<R> - F^2 <1/R> / <B^2>),  kappa = F <1/R> / <B^2>        (3)
    p          = p_thermal(n_e, T_e, n_i, T_i, n_imp) + s_pf p_fast                                (4)

* (2): :class:`.current.diffusion.CurrentDiffusion` in steady-state mode on the geometry of the
  CURRENT equilibrium (:func:`.current.geometry.from_tokamaker`), domain to psi_N = 0.995 with the
  lumped outer layer (R29); ``sigma_neo`` from :func:`.current.conductivity.neoclassical_conductivity`
  (Redl 2021 by default) on the kinetic profiles.  With the sawtooth model on, the periodic orbit
  of "diffuse for one period, reconnect with fraction f" (:func:`.current.sawtooth.solve_periodic_orbit`)
  replaces the steady state and its cycle average is used (the window average over an unknown
  phase, averaged over phases, equals the cycle average); with measured crash times the
  time-dependent slice run (:func:`.current.sawtooth.simulate_slice`) averaged over the
  measurement window is used instead.  The sawtooth parameters are the strength
  ``Lambda = -ln(1 - 0.999 f) nu`` (nu the mean crash rate) and ``f``.
* ``<J.B>_BS``: Redl 2021 (:mod:`.current.bootstrap`), multiplier ``s_bs = 1`` by default (D15).
* ``<J.B>_CD`` and ``p_fast`` are supplied as arrays (:class:`SourceProfiles`) with optional
  scale factors; this module never computes them.
* (3): the Grad-Shafranov identity for the plain surface average <J_phi> that TokaMaker's
  ``jphi-linterp`` profile takes (derived in the P0c report from ``J_phi = -R p' - F F'/(mu0 R)``
  and ``<J.B> = -F p' - F' <B^2>/mu0``; the same expression as ``CurrentDiffusion.profiles``).
  In the outer layer (psi_N > 0.995) ``<J.B>_ohm = sigma (V_layer / 2 pi) F <1/R^2>`` with the
  layer's uniform loop voltage.
* Frames: psi_N is the equilibrium's own (0 on axis, 1 on the LCFS); the "positive frame" of the
  diffusion module (``|F|``, psi increasing outward, co-current positive).  Kinetic profiles and
  sources are given on psi_N in the full pressure frame (``p(1) = p_sep``).

Pass-consistency loop (:func:`consistent_solve`): evaluate (1)-(4) on the current equilibrium,
solve (:class:`.forward.ForwardModel`), repeat, until

* ``max |j_k - j_{k-1}| / max |j_k| < pass_tol`` (default 1e-5, R08 proposal; 1e-3 was the spike
  value),
* the axis moves < 0.5 mm and l_i, q0, Ip change < 1e-3 (relative) between successive solves,
* the lagged probe correction of the forward solve changes by < ``pass_tol`` probe sigmas,
* every solve converged AND verified.

The passes may be Anderson-accelerated (``pass_acceleration="anderson"``, default; depth 3): the
input of the next solve is a mixture of the recent inputs and assembled profiles; the fixed point
and the criterion are unchanged, and a mixed input that does not solve falls back to the plain
one.

Within a Jacobian column the geometry-dependent inputs (geometry, conductivity, bootstrap, flux
surface averages) are lagged from the center state (:func:`lagged_solve`; R16); accepted states
and the final answer are always iterated to full consistency.
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
from .current.geometry import from_tokamaker
from .pressure import pressure_gradient

#: Default knots handed to TokaMaker: 129 uniform in psi_N (the validation suite's default for
#: ``jphi-linterp``).  A grid densified over the pedestal (101 uniform + 41 on [0.9, 1]) made the
#: solves erratic on a public synthetic fixture (convergence depended on the 3rd digit of a model
#: parameter; residual floor just below nl_tol), the uniform grid did not (P1 report).
DEFAULT_KNOTS = np.linspace(0.0, 1.0, 129)


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

    def __post_init__(self):
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

    def _at(self, name, x):
        v = getattr(self, name)
        if v is None:
            return np.zeros_like(np.asarray(x, dtype=float))
        x = np.asarray(x, dtype=float)
        if np.any(x < self.psi_n[0] - 1e-12) or np.any(x > self.psi_n[-1] + 1e-12):
            raise ValueError(f"'{name}' does not cover psi_N in [{x.min():g}, {x.max():g}]")
        return PchipInterpolator(self.psi_n, v)(x)

    def jdotb_at(self, x, b2=None, fpol=None, inv_r2=None):
        """<J_CD.B> at x; for the ``jtor_r0_cd`` form the current geometry's <B^2>, |F>, <1/R^2> at x
        are required (class docstring)."""
        if self.jtor_r0_cd is not None:
            if b2 is None or fpol is None or inv_r2 is None:
                raise ValueError("jtor_r0_cd needs <B^2>, F and <1/R^2> on x")
            return self._at("jtor_r0_cd", x) * np.asarray(b2) / (np.asarray(fpol) * float(self.r0_cd)
                                                                    * np.asarray(inv_r2))
        return self._at("jdotb_cd", x)

    def p_fast_at(self, x):
        return self._at("p_fast", x)


@dataclass
class CurrentModelSettings:
    """Settings of the current model and of the pass-consistency loop.  ``r0`` (reference major
    radius for the geometry, e.g. the g-file RCENTR) is device data supplied by the caller."""
    r0: float = None
    pass_tol: float = 1.0e-5                 # R08 proposal (1e-3 was the spike value)
    max_passes: int = 15
    state_axis_m: float = 5.0e-4             # consistency: axis change between successive solves
    state_rel: float = 1.0e-3                # consistency: l_i, q0, Ip change between successive solves
    n_cells: int = 100
    nrho: int = 201
    npsi: int = 200
    n_layer: int = 40
    psi_n_edge: Any = "auto"                 # R29: 0.995
    knots: Sequence[float] = field(default_factory=lambda: DEFAULT_KNOTS.copy())
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
    corr_centers_rho: Sequence[float] = (0.2, 0.5, 0.8)   # model-discrepancy Gaussians (opt-in)
    corr_width_rho: float = 0.1

    def as_dict(self):
        d = asdict(self)
        d["knots"] = dict(n=len(self.knots), min_spacing=float(np.min(np.diff(self.knots))))
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
    outer layer, flux-surface averages, conductivity and the raw bootstrap on a physics grid."""
    t0 = time.perf_counter()
    geom = from_tokamaker(gs, cfg.r0, npsi=cfg.npsi, psi_n_edge=cfg.psi_n_edge, nrho=cfg.nrho,
                          with_trapped_fraction=False, n_layer=cfg.n_layer)
    layer = geom.meta["outer_layer_geometry"]
    c = float(geom.meta["psi_n_edge"])
    xs, rs = np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"])
    to_rho, to_psin = _psin_to_rho(xs, rs)
    knots = np.asarray(cfg.knots, dtype=float)
    if knots[0] != 0.0 or knots[-1] != 1.0:
        raise ValueError("knots must run from exactly 0 to exactly 1")
    grid = np.unique(np.concatenate((knots, layer.psi_n)))
    ge = grid.copy()
    ge = np.clip(ge, cfg.fsa_pad_axis, 1.0 - cfg.fsa_pad_edge)
    ge = np.unique(ge)
    fsa = gs.get_fsa(psi=ge)
    _, fc, _, _, eps = gs.sauter_fc(psi=ge, return_eps=True)
    F = np.abs(np.asarray(fsa["F"], float))
    g1 = np.asarray(fsa["<1/R^2>"], float)
    bp2 = np.asarray(fsa["<Bp^2>"], float)
    phys = dict(psi_n=ge, q=np.abs(np.asarray(fsa["q"], float)), F=F, inv_r=np.asarray(fsa["<1/R>"], float),
                r_avg=np.asarray(fsa["<R>"], float), inv_r2=g1, b2=bp2 + F ** 2 * g1,
                r_geo=0.5 * (np.asarray(fsa["R_min"], float) + np.asarray(fsa["R_max"], float)),
                eps=np.asarray(eps, float), ft=1.0 - np.asarray(fc, float))
    dpsi = abs(float(fsa["psi_boundary"]) - float(fsa["psi_axis"]))
    k = kin.at(ge)
    sig, sparts = neoclassical_conductivity(k["ne"], k["te"], k["zeff"], phys["ft"], phys["q"], phys["r_geo"],
                                            phys["eps"], model=cfg.conductivity_model,
                                            ln_lambda_model=cfg.ln_lambda_model)
    jbs, bparts = redl_jdotb(ge, k["te"], k["ti"], k["ne"], k["ni"], k["zeff"], phys["q"], phys["r_geo"],
                             phys["eps"], phys["ft"], F, dpsi, ion_collisionality=cfg.ion_collisionality)
    return dict(geom=geom, layer=layer, psi_n_edge=c, to_rho=to_rho, to_psin=to_psin, phys=phys, dpsi=dpsi,
                sigma=np.asarray(sig, float), jbs=np.asarray(jbs, float), kin=k,
                bootstrap_settings=bparts["settings"], wall_s=round(time.perf_counter() - t0, 4),
                fsa_psi_axis=float(fsa["psi_axis"]), fsa_psi_boundary=float(fsa["psi_boundary"]))


def _interp(x, y, xq, log=False):
    f = PchipInterpolator(x, np.log(y) if log else y, extrapolate=True)
    v = f(np.asarray(xq, float))
    return np.exp(v) if log else v


def assemble(lag, kin, src, params, cfg, ip, z0=None):
    """Profiles for one forward solve from the lagged geometry inputs ``lag``
    (:func:`evaluate_geometry`) and the parameters.  Returns a dict with the knots ``x``, the
    current ``jphi`` [A/m^2] and full-frame pressure ``p`` [Pa] on them, the parts of <J.B> and the
    current fractions, the sawtooth record, and ``orbit_z`` (continuation for the next orbit)."""
    t0 = time.perf_counter()
    p = full_params(params)
    geom, layer, c = lag["geom"], lag["layer"], lag["psi_n_edge"]
    ph, ge = lag["phys"], lag["phys"]["psi_n"]
    ipv = abs(float(ip))
    # sources on the physics grid
    jcd = p["s_cd"] * src.jdotb_at(ge, b2=ph["b2"], fpol=ph["F"], inv_r2=ph["inv_r2"])
    jbs = p["s_bs"] * lag["jbs"]
    rho_ge = np.where(ge <= c, lag["to_rho"](np.minimum(ge, c)), np.nan)
    jcorr = np.zeros_like(ge)
    corr_shapes = []
    if any(a != 0.0 for a in p["a_corr"]):
        if len(p["a_corr"]) != len(cfg.corr_centers_rho):
            raise ValueError("a_corr needs one amplitude per correction center")
        # unit-current shapes: g_k(rho) Gaussian, normalized so that it carries I = Ip (I/F integral
        # over the domain times F at the LCFS), before the ohmic part adjusts to keep Ip fixed
        gc = geom.evaluate(geom.rho, ["dvol_drho", "fpol"])
        for a, rc0 in zip(p["a_corr"], cfg.corr_centers_rho):
            shape_rho = np.exp(-0.5 * ((geom.rho - rc0) / cfg.corr_width_rho) ** 2)
            gam = np.trapezoid(gc["dvol_drho"] * shape_rho / (2 * np.pi * gc["fpol"] ** 2), geom.rho)
            norm = ipv / (layer.fpol_lcfs * gam)
            g = np.where(np.isfinite(rho_ge), np.exp(-0.5 * ((np.nan_to_num(rho_ge) - rc0) / cfg.corr_width_rho) ** 2),
                         0.0) * norm
            jcorr = jcorr + a * g
            corr_shapes.append(dict(center_rho=rc0, width_rho=cfg.corr_width_rho, amplitude_fraction_of_ip=a))
    jni_ge = jbs + jcd + jcorr
    # domain arrays on the geometry's rho grid
    rho_g = geom.rho
    x_g = lag["to_psin"](rho_g)
    x_g[0] = 0.0
    sig_g = _interp(ge, lag["sigma"], np.clip(x_g, ge[0], None), log=True)
    jni_g = _interp(ge, jni_ge, np.clip(x_g, ge[0], None))
    lx = layer.psi_n
    sig_l = _interp(ge, lag["sigma"], lx, log=True)
    jni_l = _interp(ge, jni_ge, lx)
    sol = CurrentDiffusion(geom, (rho_g, sig_g), IpBoundary(ipv), j_ni=(rho_g, jni_g),
                           outer_layer=OuterLayer.from_profiles(layer, (lx, sig_l), (lx, jni_l)))
    rf = CurrentState.uniform_faces(cfg.n_cells)
    saw = dict(active=False)
    orbit_z = None
    if cfg.sawtooth and p["log_strength"] is not None:
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
    # <J.B> and its parts on the knots
    xk = np.asarray(cfg.knots, dtype=float)
    xs_c = lag["to_psin"](prof["rho_cells"])
    dom = xk <= c
    jb = np.empty_like(xk)
    jb_ni = np.empty_like(xk)
    jb[dom] = _interp(xs_c, prof["jdotb"], xk[dom])
    jb_ni[dom] = _interp(ge, jni_ge, np.clip(xk[dom], ge[0], None))
    lay = ~dom
    xkc = np.clip(xk, ge[0], ge[-1])
    F_k = _interp(ge, ph["F"], xkc)
    g1_k = _interp(ge, ph["inv_r2"], xkc)
    jb_ni[lay] = _interp(ge, jni_ge, xkc[lay])
    jb[lay] = _interp(ge, lag["sigma"], xkc[lay], log=True) * rate_layer * F_k[lay] * g1_k[lay] + jb_ni[lay]
    kap = F_k * _interp(ge, ph["inv_r"], xkc) / _interp(ge, ph["b2"], xkc)
    pk = kin.thermal_pressure(xk) + p["s_pfast"] * src.p_fast_at(xk)
    dp_dpsin = pressure_gradient(xk, pk)
    dp_dpsi = dp_dpsin / lag["dpsi"]
    jpd = -dp_dpsi * (_interp(ge, ph["r_avg"], xkc) - F_k * kap)
    jphi = kap * jb + jpd
    if not np.all(np.isfinite(jphi)):
        raise ModelFailure("assembled <j_phi> is not finite")
    # current fractions: I_x = F_lcfs * int <J.B>_x / (2 pi F^2) dV (domain cells + layer), eq. (6)/(8)
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
    out = dict(x=xk, jphi=jphi, p=pk, jdotb=jb, jdotb_ni=jb_ni, kappa=kap, jphi_pressure_term=jpd,
               currents_A=cur, current_sum_A=tot, ip=ipv,
               fractions={kk: v / ipv for kk, v in cur.items()},
               v_loop=float(prof["v_loop_boundary"]), q_diffusion=dict(rho=prof["rho_faces"], q=prof["q"]),
               li3_diffusion=float(prof["li3"]), outer_layer_current_A=float(prof["outer_layer_current"]),
               sawtooth=saw, orbit_z=orbit_z, corrections=corr_shapes, params=p,
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


def consistent_solve(fm, kin, src, params, cfg, ip=None, start=None, z0=None, label="consistent",
                     initial_profiles=None):
    """Iterate model evaluation and forward solve to full consistency (module docstring).

    ``fm`` is a :class:`.forward.ForwardModel`; ``start`` an optional snapshot (or psi array) to
    start from (default: the solver's current state); ``z0`` an orbit continuation vector.
    ``initial_profiles`` = ``(x, jphi, p)``: for the FIRST solve of a case, whose starting psi is
    only an initial guess (no solved equilibrium to evaluate the model on yet), one forward solve
    with these profiles (e.g. the reference equilibrium's <j_phi> and the kinetic pressure) is made
    before the loop ("pass -1"; recorded); it is a starting state, not part of the answer.
    Failures (a model failure, a solve that does not converge or verify, or no consistency within
    ``max_passes``) are returned with ``converged=False`` and the reason; the solver is then left in
    its last state (the caller restores a snapshot)."""
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
    hx, hf = [], []          # Anderson histories: inputs and residuals G(x) - x of the j fixed point
    x_prev = None
    for k in range(int(cfg.max_passes)):
        try:
            lag_k = evaluate_geometry(fm.gs, kin, cfg)
            asm_k = assemble(lag_k, kin, src, params, cfg, ipv, z0=z0)
        except (ModelFailure, ValueError, RuntimeError) as exc:
            return ConsistentResult(False, "model_failure", k, asm, lag, res,
                                    dict(passes=passes, reason=f"{type(exc).__name__}: {exc}"),
                                    round(time.perf_counter() - t0, 3))
        z0 = asm_k["orbit_z"]
        rec = dict(k=k, geometry_wall_s=lag_k["wall_s"], assembly_wall_s=asm_k["wall_s"],
                   fractions=asm_k["fractions"], sawtooth_active=asm_k["sawtooth"].get("active"))
        if prev_j is not None:
            dj = float(np.max(np.abs(asm_k["jphi"] - prev_j)) / np.max(np.abs(asm_k["jphi"])))
            ax, rel = _state_change(res.stats, prev_stats) if prev_stats is not None else (np.inf, {})
            rec.update(dj=dj, d_axis_m=ax, d_rel=rel, d_probe_sigma=last_probe_change)
            ok_state = prev_stats is not None and ax < cfg.state_axis_m and all(v < cfg.state_rel for v in rel.values())
            ok_probe = last_probe_change is None or last_probe_change < cfg.pass_tol
            if dj < cfg.pass_tol and ok_state and ok_probe:
                passes.append(rec)
                return ConsistentResult(True, "consistent", k, used, lag, res,
                                        dict(passes=passes, criterion=dict(pass_tol=cfg.pass_tol,
                                                                           axis_m=cfg.state_axis_m,
                                                                           rel=cfg.state_rel),
                                             assembly_on_final_state=dict(dj=dj)),
                                        round(time.perf_counter() - t0, 3))
        prev_stats = None if res is None else res.stats
        # next input of the fixed point j -> G(j): plain (G(j) itself) or Anderson-mixed (type II,
        # H. F. Walker and P. Ni, SIAM J. Numer. Anal. 49, 1715 (2011)); the fixed point and the
        # consistency criterion (|G(j) - j| on the assembled profile) are the same either way
        x_in, mixed = asm_k["jphi"], False
        if x_prev is not None:
            hx.append(x_prev)
            hf.append(asm_k["jphi"] - x_prev)
            del hx[:-(cfg.anderson_depth + 1)], hf[:-(cfg.anderson_depth + 1)]
            if cfg.pass_acceleration == "anderson" and len(hf) >= 2:
                dF = np.column_stack([hf[i + 1] - hf[i] for i in range(len(hf) - 1)])
                dX = np.column_stack([hx[i + 1] - hx[i] for i in range(len(hx) - 1)])
                gam = np.linalg.lstsq(dF, hf[-1], rcond=None)[0]
                cand = hx[-1] + hf[-1] - (dX + dF) @ gam
                if np.all(np.isfinite(cand)):
                    x_in, mixed = cand, True
        snap_k = fm.snapshot() if mixed else None
        res = fm.solve((asm_k["x"], x_in), (asm_k["x"], asm_k["p"]), ipv, init="keep", label=f"{label}:pass{k}")
        if mixed and not res.converged:
            # safeguard: the mixed input did not solve -> plain input from the same state
            rec["anderson_fallback"] = res.status
            fm.restore(snap_k)
            x_in, mixed = asm_k["jphi"], False
            hx.clear()
            hf.clear()
            res = fm.solve((asm_k["x"], x_in), (asm_k["x"], asm_k["p"]), ipv, init="keep",
                           label=f"{label}:pass{k}:plain")
        rec.update(solve_status=res.status, its=res.record["attempts"][-1].get("its"), solve_wall_s=res.wall_s,
                   anderson_mixed=mixed)
        passes.append(rec)
        if not res.converged:
            st = ("limit_cycle_untapered_lock" if res.status == "failed_limit_cycle_untapered_lock"
                  or rec.get("anderson_fallback") == "failed_limit_cycle_untapered_lock" else "solve_failed")
            return ConsistentResult(False, st, k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=res.status), round(time.perf_counter() - t0, 3))
        try:
            last_probe_change = fm.update_probe_correction()
        except RuntimeError as exc:
            return ConsistentResult(False, "probe_correction_failed", k, asm_k, lag_k, res,
                                    dict(passes=passes, reason=str(exc)), round(time.perf_counter() - t0, 3))
        prev_j = x_in
        x_prev = x_in
        used = dict(asm_k, jphi=x_in, jphi_assembled=asm_k["jphi"]) if mixed else asm_k
        lag, asm = lag_k, asm_k
    return ConsistentResult(False, "max_passes", int(cfg.max_passes), used, lag, res,
                            dict(passes=passes, reason=f"not consistent within {cfg.max_passes} passes"),
                            round(time.perf_counter() - t0, 3))


def lagged_solve(fm, lag, kin, src, params, cfg, ip=None, start=None, z0=None, label="column"):
    """One forward solve with the geometry-dependent inputs lagged from ``lag`` (a Jacobian column,
    R16).  Returns ``(SolveResult or None, assembly or None, reason)``."""
    ipv = abs(fm.ip_measured if ip is None else float(ip))
    if start is not None:
        fm.restore(start)
    try:
        asm = assemble(lag, kin, src, params, cfg, ipv, z0=z0)
    except (ModelFailure, ValueError, RuntimeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    res = fm.solve((asm["x"], asm["jphi"]), (asm["x"], asm["p"]), ipv, init="keep", label=label)
    return res, asm, res.status
