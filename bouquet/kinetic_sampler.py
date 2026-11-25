"""One kinetic draw for every solve path (legacy, swb, unified engine).

Solver-free: the caller passes the thermal-pressure integral of a draw
(:class:`KineticDraw` -> float) and gets the drawn profiles on the kinetic
grid.  RNG order: ``ne``, ``Te``, then ``Z_eff`` (deriving ``ni``) or ``ni``,
then ``Ti`` -- redrawn together until the pressure integral is within
``p_thresh`` -- then each auxiliary channel.  In ``"gibbs_level"`` mode each
profile is ``base + (sample - mean) * b0`` (and in ``"stiff_inward"`` mode is
integrated inward in log-space with analytical drift debiasing), so a zero
sigma returns the base exactly on every mode.
A drawn Z_eff is clipped to :func:`bouquet.physics.zeff_bounds`; a derived
``ni`` is an increment on the baseline ``ni``.  The PR #56 clips -- the
Z_eff floor at 1, ``ni`` held inside ``[0, ne - z_fast]``, the passive Z_eff
aux clip -- are switched by ``KineticBase.clips`` (False here, for a
direct caller); a run resolves them from ``UncertaintyConfig.kinetic_clips``,
ON by default on every route since 2026-10-17
(:func:`bouquet.experimental.kinetic_clips_on`; an explicit False wins).
Every clip
that moves a node is counted on the draw (:attr:`KineticDraw.clips`,
:meth:`KineticDraw.record`); the version is :data:`KINETIC_SAMPLER_VERSION`.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

#: drawn Z_eff floor of the PR #56 clips (``KineticBase.clips``)
ZEFF_DRAW_MIN = 1.0

#: Version of the kinetic draw, to be recorded with every draw.
#: ``/1`` (6116d5f): engine and legacy each had their own sampler; the legacy
#: Z_eff route derived ``ni`` ABSOLUTELY, ``ni_of(ne_d, Z_eff_d, Z_imp)``,
#: which at sigma = 0 does not return the baseline ``ni`` whenever the
#: baseline is not single-impurity quasineutral at the median ``Z_imp``
#: (multi-species / beam p-files, IDA) -- a sigma = 0 violation.
#: ``/2`` (PR #56): one sampler for every path; a derived ``ni`` is an
#: INCREMENT on the baseline, ``ni + ni_of(ne_d, Z_eff_d) - ni_of(ne, Z_eff)``
#: (exactly the baseline at sigma = 0); each profile is ``base + (sample -
#: mean) b0``; a Z_eff draw is held in :func:`bouquet.physics.zeff_bounds`
#: floored at :data:`ZEFF_DRAW_MIN`, including a passive Z_eff aux draw; a
#: derived ``ni`` is held in ``[0, ne - z_fast]`` and an independent one
#: below ``ne - z_fast`` when ``Z_imp`` is declared.  Legacy draws with the
#: Z_eff channel on therefore differ from ``/1`` (seed-for-seed); engine
#: draws differ only where a clip binds.  Every clip that fires is counted
#: per draw (:attr:`KineticDraw.clips`).  (Since 1.4.0 the engine IMAS
#: draws declare ``Baseline.Z_imp``, the thermal inversion, instead of the
#: inversion of the dd's Z_eff; the record names the Z_imp used and its
#: origin.)
#: ``/3`` (1.4.0): ``/2`` with the PR #56 clips (the floor at 1, the ``ni``
#: floor and ceiling, the passive Z_eff aux clip) EXPERIMENTAL and opt-in
#: (:attr:`KineticBase.clips`); by default a Z_eff draw is held in
#: :func:`bouquet.physics.zeff_bounds` alone (the window before PR #56) and
#: ``ni`` is not clipped.  With ``clips=True`` a draw equals ``/2``'s.  The
#: record says which (``clips_enabled``).
#: ``/3.1`` (2026-10-17): the draw arithmetic of ``/3`` for a given
#: ``clips`` flag, seed for seed; only the run-level default changed: a run
#: resolves the clips ON on every route (``UncertaintyConfig.kinetic_clips``
#: None = auto; an explicit False wins), so the ``/3`` string's "clips
#: opt-in" no longer described the archived draws.  The minor number marks
#: archives written under the new default; ``clips_enabled`` on each record
#: remains the per-draw truth.
#: ``/3.2`` (2026-11-21): adds optional physics-informed sampling modes
#: (``gradient_policy="physics_bounds"``, ``sampling_mode="stiff_inward"``,
#: ``chi_consistency="power_balance" | "surrogate"``, and fuel-gated DT
#: ``tritium_fraction`` alpha-heating feedback), recorded under
#: ``physics_sampling`` when active; legacy default draw arithmetic is
#: seed-for-seed identical to ``/3.1``.
KINETIC_SAMPLER_VERSION = (
    "kinetic_sampler/3.2 (shared sampler; derived ni = increment on the "
    "baseline ni; Z_eff window = zeff_bounds; PR #56 clips on by default "
    "(auto), explicit False off; physics_bounds / stiff_inward / "
    "power_balance chi extensions; clips counted per draw)")

#: The names of :attr:`KineticDraw.clips` (each the number of kinetic-grid
#: nodes the clip moved, on the ACCEPTED draw).
CLIP_COUNTERS = (
    # Z_eff primary draw: lifted to the floor 1 where zeff_bounds itself
    # would have allowed a lower value (the PR #56 floor, owner rule E3)
    "zeff_floor_1",
    # Z_eff primary draw: held at zeff_bounds' own lower / upper edge
    "zeff_window_lo", "zeff_window_hi",
    # passive Z_eff aux draw (ni not derived from it): the same window
    "aux_zeff_floor_1", "aux_zeff_window_lo", "aux_zeff_window_hi",
    # ni: the thermal main-ion density floored at 0 / capped at ne - z_fast
    "ni_floor_0", "ni_ceiling",
)


def dilution_Z_imp(ne, ni, aux_sigmas, aux_baselines, Z_imp=None,
                   z_fast=None) -> Optional[float]:
    """Impurity charge of an active Z_eff channel: the declared *Z_imp*, else
    inverted from (ne, ni, Z_eff).  None without a channel or a dilution."""
    if not (aux_sigmas and "zeff" in aux_sigmas
            and (aux_baselines or {}).get("zeff") is not None):
        return None
    if Z_imp:
        return float(Z_imp)
    from .physics import impurity_charge_with_fast_ions
    ne = np.asarray(ne, dtype=float)
    Z, _ = impurity_charge_with_fast_ions(
        ne, ni, np.asarray(aux_baselines["zeff"], dtype=float),
        np.zeros_like(ne) if z_fast is None else z_fast)
    return Z


@dataclass
class KineticBase:
    """The unperturbed kinetics and their uncertainties, on the kinetic grid."""
    psi_kin: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    sigma_ne: np.ndarray
    sigma_te: np.ndarray
    sigma_ni: np.ndarray
    sigma_ti: np.ndarray
    n_ls: object
    t_ls: object
    aux_sigmas: Optional[dict] = None
    aux_baselines: Optional[dict] = None
    aux_length_scales: Optional[dict] = None
    Z_imp: Optional[float] = None
    z_fast: Optional[np.ndarray] = None
    z2_fast: Optional[np.ndarray] = None
    zeff_includes_fast: bool = False
    ni_from_zeff: bool = True
    zeff_dne: Optional[np.ndarray] = None
    #: the PR #56 clips (the Z_eff floor at 1, ni in [0, ne - z_fast], the
    #: passive Z_eff aux clip); False keeps only the zeff_bounds window.
    #: False here for a direct caller; a run passes the resolved
    #: UncertaintyConfig.kinetic_clips (on by default on every route)
    clips: bool = False
    #: Optional per-species correlation length overrides (default None -> n_ls / t_ls)
    ne_ls: object = None
    te_ls: object = None
    ni_ls: object = None
    ti_ls: object = None
    #: Regional gradient & positivity policy: 'legacy' | 'positivity_only' | 'physics_bounds'
    gradient_policy: str = "legacy"
    #: Profile sampler architecture: 'gibbs_level' | 'stiff_inward'
    sampling_mode: str = "gibbs_level"
    #: Thermal diffusivity consistency rule: 'passive_gp' | 'power_balance' | 'surrogate'
    chi_consistency: str = "passive_gp"
    #: Scale burning-plasma alpha-heating fraction by Bosch-Hale DT reactivity in power_balance chi
    alpha_heating_feedback: bool = False
    #: Stiffness / geometry metadata for physics_bounds, stiff_inward, and surrogate chi
    psi_ped_top: float = 0.92
    has_pedestal: bool = True
    tritium_fraction: float = 0.0
    coord: str = "psi_n"
    r_norm_kin: Optional[np.ndarray] = None
    a_over_R0: float = 0.33
    dkappa_allow: Optional[dict] = None
    kappa_bounds: Optional[dict] = None
    miller_kin: object = None
    omega_tor_rad_s: Optional[np.ndarray] = None
    tglfnn_model: str = "sat2_em_d3d_azf-1_withnegD"
    stiff_gradient_ls: float = 0.30

    def __post_init__(self):
        for k in ("psi_kin", "ne", "te", "ni", "ti", "sigma_ne", "sigma_te",
                  "sigma_ni", "sigma_ti"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=float))
        for k in ("z_fast", "z2_fast", "zeff_dne", "r_norm_kin", "omega_tor_rad_s"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, np.asarray(v, dtype=float))

    @property
    def zeff_channel(self) -> bool:
        """A Z_eff aux channel with a sigma and a baseline."""
        return (bool(self.aux_sigmas) and "zeff" in self.aux_sigmas
                and (self.aux_baselines or {}).get("zeff") is not None)

    def resolved_Z_imp(self) -> Optional[float]:
        """The impurity charge of the Z_eff channel (None: no dilution)."""
        return dilution_Z_imp(self.ne, self.ni, self.aux_sigmas,
                              self.aux_baselines, self.Z_imp, self.z_fast)

    def zeff_window(self, ne, Z_imp):
        """``(lo, hi)`` a Z_eff draw is clipped to on *ne* (``hi`` None: open):
        :meth:`zeff_bounds_raw`, floored at :data:`ZEFF_DRAW_MIN` only with
        the PR #56 :attr:`clips`."""
        lo, hi = self.zeff_bounds_raw(ne, Z_imp)
        if not self.clips:
            return lo, hi
        return np.maximum(lo, ZEFF_DRAW_MIN), hi

    def zeff_bounds_raw(self, ne, Z_imp):
        """:func:`bouquet.physics.zeff_bounds` before the floor at 1 (the
        window the physics alone allows; ``(1, None)`` without ``Z_imp``)."""
        if Z_imp is None:
            return ZEFF_DRAW_MIN, None
        from .physics import zeff_bounds
        return zeff_bounds(ne, Z_imp, self.z_fast, self.z2_fast,
                           self.zeff_includes_fast)

    def ni_ceiling(self, ne):
        """``ne - z_fast`` (>= 0): the most main ions the thermal electrons
        neutralise."""
        zf = 0.0 if self.z_fast is None else self.z_fast
        return np.maximum(ne - zf, 0.0)


@dataclass
class KineticDraw:
    """One draw on the kinetic grid (``zeff``: the drawn Z_eff, or None)."""
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    zeff: Optional[np.ndarray] = None
    aux: dict = field(default_factory=dict)
    zeff_primary: bool = False
    iterations: int = 0
    p_err_pct: float = 0.0
    #: nodes each clip moved on this draw (:data:`CLIP_COUNTERS`); all zero
    #: when nothing was clipped
    clips: dict = field(default_factory=lambda: dict.fromkeys(
        CLIP_COUNTERS, 0))
    #: whether the PR #56 clips were on (KineticBase.clips)
    clips_enabled: bool = False
    #: the impurity charge the Z_eff channel used (None: no channel or no
    #: dilution) and where it came from: ``"declared"`` (the caller's
    #: ``KineticBase.Z_imp``, e.g. ``Baseline.Z_imp``) or ``"inverted"``
    #: (the sampler inverted it from the baseline ne, ni and Z_eff)
    Z_imp: Optional[float] = None
    Z_imp_origin: Optional[str] = None
    physics_sampling: Optional[dict] = None

    @property
    def native(self) -> dict:
        return dict(ne=self.ne, te=self.te, ni=self.ni, ti=self.ti)

    @property
    def clipped(self) -> bool:
        """True when any clip moved any node of this draw."""
        return any(int(v) > 0 for v in self.clips.values())

    def record(self) -> dict:
        """JSON-safe per-draw stamp: the sampler version, the pressure match,
        the clip counters and the resolved ``Z_imp`` with its origin (for the
        draw's archive record)."""
        rec = dict(version=KINETIC_SAMPLER_VERSION,
                   clips_enabled=bool(self.clips_enabled),
                   Z_imp=(None if self.Z_imp is None
                          else float(self.Z_imp)),
                   Z_imp_origin=self.Z_imp_origin,
                   zeff_primary=bool(self.zeff_primary),
                   iterations=int(self.iterations),
                   p_err_pct=float(self.p_err_pct),
                   clipped=bool(self.clipped),
                   clips={k: int(v) for k, v in self.clips.items()})
        if self.physics_sampling is not None:
            rec["physics_sampling"] = dict(self.physics_sampling)
        return rec


def compute_power_balance_chi(
    psi_N: np.ndarray,
    chi_base: np.ndarray,
    n_base: np.ndarray,
    t_base: np.ndarray,
    n_draw: np.ndarray,
    t_draw: np.ndarray,
    *,
    species: str = "e",
    q_source_mult: Optional[np.ndarray] = None,
    alpha_heating_feedback: bool = False,
    tritium_fraction: float = 0.0,
    ni_base: Optional[np.ndarray] = None,
    ti_base: Optional[np.ndarray] = None,
    ni_draw: Optional[np.ndarray] = None,
    ti_draw: Optional[np.ndarray] = None,
    te_base: Optional[np.ndarray] = None,
    te_draw: Optional[np.ndarray] = None,
    volume_m3: Optional[np.ndarray] = None,
    surface_m2: Optional[np.ndarray] = None,
    r_min_m: Optional[np.ndarray] = None,
    ratio_bounds: tuple = (0.1, 10.0),
) -> np.ndarray:
    """Compute power-balance-consistent thermal diffusivity ``chi_s(psi_N)``.

    Enforces conductive flux consistency ``Q_s = -n_s * chi_s * dT_s/dpsi_N``:

        chi_draw = chi_base * (n_base / n_draw) * (G[T_base] / G[T_draw]) * M_Q

    with:
    1. Exact zero-perturbation identity (returns ``chi_base.copy()`` when
       ``n_draw == n_base``, ``t_draw == t_base``, and ``q_source_mult == 1``);
    2. Smooth per-profile axis & hollow-core gradient regularization
       ``G[T] = sqrt((dT/dpsi_N)^2 + g_reg^2)`` with ``g_reg = 0.10 * span(T)``,
       guaranteeing exact ``1 / alpha`` scaling under ``T_draw = alpha * T_base``
       even at a flat/parabolic magnetic axis and ``C^inf`` positivity across
       radiatively cooled hollow cores (``dT/dpsi_N > 0``);
    3. Fuel-composition-gated burning-plasma DT alpha-heating feedback
       (identically zero when ``tritium_fraction <= 0``, i.e. pure Deuterium;
       otherwise ``n_D n_T = f_T (1 - f_T) n_fuel^2 <sigma*v>_DT E_alpha`` with
       Stix electron slowing-down fraction ``f_{alpha,e}(T_e) = 1 / (1 +
       (T_e / 33 keV)^{3/2})`` and volume-integrated burn fraction
       ``P_{alpha,s} / max(P_{tot,s}, P_{alpha,s})``).

    Notes
    -----
    **Open physics investigation (SLAYER rational-surface tearing UQ):**
    In ``SLAYER``'s 4-field two-fluid drift-tearing layer model (Cole &
    Fitzpatrick 2006; Fitzpatrick 2022), kinetic draws shift rational-surface
    tearing stability and error-field penetration primarily through the electron
    diamagnetic frequency ``omega_{*e} ~ -(1 / (e B n_e)) dp_e/dr``, parallel
    Spitzer resistivity ``eta_|| ~ Z_eff T_e^{-3/2}``, and perpendicular layer
    closures (``chi_e``, ``D_perp``, and anomalous ion viscosity
    ``mu_perp ~ chi_i``).  Because ``bouquet`` archives represent stationary
    force- and energy-balanced equilibria at fixed auxiliary heating power,
    ``chi_consistency='power_balance'`` is the energetically self-consistent
    closure (steeper ``|-dT_e/dpsi_N|`` implies lower realized ``chi_e``,
    ``r < 0``), whereas ``chi_consistency='surrogate'`` passes
    ``q_source_mult = Q_TGLFNN(draw) / Q_TGLFNN(base)`` (``r > 0`` above
    critical gradient, corresponding to uncompensated 3x-15x heating-power
    excursions or transient unrelaxed states).  Use ``'power_balance'`` as the
    primary steady-state closure and ``'surrogate'`` as an upper sensitivity
    bracket.
    """
    psi = np.asarray(psi_N, dtype=float)
    chi0 = np.asarray(chi_base, dtype=float)
    n0 = np.asarray(n_base, dtype=float)
    t0 = np.asarray(t_base, dtype=float)
    nd = np.asarray(n_draw, dtype=float)
    td = np.asarray(t_draw, dtype=float)

    q_mult = (np.ones_like(psi) if q_source_mult is None
              else np.maximum(np.asarray(q_source_mult, dtype=float), 0.1))

    f_trit = float(np.clip(tritium_fraction, 0.0, 1.0))
    has_dt_alpha = bool(
        alpha_heating_feedback
        and 0.0 < f_trit < 1.0
        and ni_base is not None
        and ti_base is not None
        and ni_draw is not None
        and ti_draw is not None
    )

    if (np.array_equal(nd, n0) and np.array_equal(td, t0)
            and np.allclose(q_mult, 1.0, atol=0.0)
            and (not has_dt_alpha or (
                np.array_equal(ni_draw, ni_base)
                and np.array_equal(ti_draw, ti_base)
                and (te_draw is None or te_base is None
                     or np.array_equal(te_draw, te_base))))):
        return chi0.copy()

    # Smooth per-profile gradient regularization (C^inf across hollow cores and
    # exact 1/alpha scaling at psi_N = 0 under T_draw = alpha * T_base)
    g0 = -np.gradient(t0, psi)
    gd = -np.gradient(td, psi)
    span0 = float(np.max(t0) - np.min(t0))
    spand = float(np.max(td) - np.min(td))
    g_reg_0 = 0.10 * (
        span0 if span0 > 1.0e-12 else max(float(np.max(np.abs(t0))), 1.0e-12)
    )
    g_reg_d = 0.10 * (
        spand if spand > 1.0e-12 else max(float(np.max(np.abs(td))), 1.0e-12)
    )
    g0_eff = np.sqrt(g0 ** 2 + g_reg_0 ** 2)
    gd_eff = np.sqrt(gd ** 2 + g_reg_d ** 2)

    alpha_mult = np.ones_like(psi)
    if has_dt_alpha:
        from scipy import integrate
        from .physics import ELEMENTARY_CHARGE
        from .stiffness import bosch_hale_dt_reactivity

        vol = (
            20.0 * (np.maximum(psi, 0.0) ** 1.5)
            if volume_m3 is None
            else np.maximum(np.asarray(volume_m3, dtype=float), 0.0)
        )
        fuel_pair = f_trit * (1.0 - f_trit)
        e_alpha_J = 3.52e6 * ELEMENTARY_CHARGE
        p_a0 = (
            fuel_pair
            * (np.maximum(np.asarray(ni_base, dtype=float), 0.0) ** 2)
            * bosch_hale_dt_reactivity(np.asarray(ti_base, dtype=float))
            * e_alpha_J
        )
        p_ad = (
            fuel_pair
            * (np.maximum(np.asarray(ni_draw, dtype=float), 0.0) ** 2)
            * bosch_hale_dt_reactivity(np.asarray(ti_draw, dtype=float))
            * e_alpha_J
        )

        # Stix / Sivukhin alpha slowing-down electron energy fraction (E_crit ~ 33 keV)
        te0_arr = (
            np.asarray(te_base, dtype=float)
            if te_base is not None
            else (t0 if species == "e" else np.asarray(ti_base, dtype=float))
        )
        ted_arr = (
            np.asarray(te_draw, dtype=float)
            if te_draw is not None
            else (td if species == "e" else np.asarray(ti_draw, dtype=float))
        )
        f_ae_0 = 1.0 / (1.0 + (np.maximum(te0_arr, 1.0) / 33.0e3) ** 1.5)
        f_ae_d = 1.0 / (1.0 + (np.maximum(ted_arr, 1.0) / 33.0e3) ** 1.5)
        if species == "e":
            p_as_0 = f_ae_0 * p_a0
            p_as_d = f_ae_d * p_ad
        else:
            p_as_0 = (1.0 - f_ae_0) * p_a0
            p_as_d = (1.0 - f_ae_d) * p_ad

        P_as_0 = integrate.cumulative_trapezoid(p_as_0, vol, initial=0.0)
        P_as_d = integrate.cumulative_trapezoid(p_as_d, vol, initial=0.0)

        # Local baseline conductive power P_{cond,s,0}(psi_N) = -n_{s,0} e chi_{s,0} dT_{s,0}/dr S
        if surface_m2 is not None and r_min_m is not None:
            r_arr = np.asarray(r_min_m, dtype=float)
            s_arr = np.maximum(np.asarray(surface_m2, dtype=float), 0.0)
            a_eff = max(float(r_arr[-1] - r_arr[0]), 1.0e-3)
            dt_dr = np.sqrt((-np.gradient(t0, r_arr)) ** 2 + (g_reg_0 / a_eff) ** 2)
            P_cond_0 = (
                np.maximum(n0, 0.0)
                * ELEMENTARY_CHARGE
                * np.maximum(chi0, 0.0)
                * dt_dr
                * s_arr
            )
        else:
            v_tot = max(float(vol[-1]) if vol.size else 20.0, 1.0)
            a_est = max(
                (v_tot / (2.0 * (np.pi ** 2) * 3.0 * 1.7)) ** (1.0 / 3.0), 0.1
            )
            r_est = a_est * np.sqrt(np.clip(psi, 0.0, 1.0))
            dv_dr = np.maximum(np.gradient(vol, np.maximum(r_est, 1.0e-6)), 0.0)
            dr_dpsi = np.maximum(np.gradient(r_est, psi), 1.0e-4)
            dt_dr = np.sqrt((g0 / dr_dpsi) ** 2 + (g_reg_0 / a_est) ** 2)
            P_cond_0 = (
                np.maximum(n0, 0.0)
                * ELEMENTARY_CHARGE
                * np.maximum(chi0, 0.0)
                * dt_dr
                * dv_dr
            )

        P_tot_0 = np.maximum(P_cond_0, P_as_0)
        f_burn_s = np.where(P_tot_0 > 1.0e-6, P_as_0 / np.maximum(P_tot_0, 1.0e-6), 0.0)
        ratio_p = np.where(P_as_0 > 1.0e-6, P_as_d / np.maximum(P_as_0, 1.0e-6), 1.0)
        if psi.size > 1:
            f_burn_s[0] = f_burn_s[1]
            ratio_p[0] = (
                float(p_as_d[0] / p_as_0[0])
                if p_as_0[0] > 1.0e-12
                else float(ratio_p[1])
            )
        alpha_mult = (1.0 - f_burn_s) + f_burn_s * np.clip(ratio_p, 0.2, 5.0)

    n_ratio = np.maximum(n0, 1e15) / np.maximum(nd, 1e15)
    raw_factor = n_ratio * (g0_eff / gd_eff) * q_mult * alpha_mult
    lo_r, hi_r = float(ratio_bounds[0]), float(ratio_bounds[1])
    return chi0 * np.clip(raw_factor, lo_r, hi_r)


def _clip_zeff(z, lo, hi):
    return np.clip(z, lo, None if hi is None else hi * (1.0 - 1e-9))


def _count_zeff_clips(raw, lo_bounds, lo, hi, prefix=""):
    """Nodes of a raw Z_eff draw the window moves: lifted to the floor 1
    where ``zeff_bounds``' own edge *lo_bounds* lies below it, held at
    ``zeff_bounds``' lower edge, or at its upper edge *hi*."""
    raw = np.asarray(raw, dtype=float)
    lo = np.broadcast_to(np.asarray(lo, dtype=float), raw.shape)
    lo_b = np.broadcast_to(np.asarray(lo_bounds, dtype=float), raw.shape)
    below = raw < lo
    # lifted by the floor (the PR #56 clips) above zeff_bounds' edge
    floor1 = below & (lo_b < ZEFF_DRAW_MIN) & (lo > lo_b)
    out = {prefix + "zeff_floor_1": int(np.count_nonzero(floor1)),
           prefix + "zeff_window_lo": int(np.count_nonzero(below & ~floor1)),
           prefix + "zeff_window_hi": 0}
    if hi is not None:
        out[prefix + "zeff_window_hi"] = int(np.count_nonzero(
            raw > np.asarray(hi, dtype=float) * (1.0 - 1e-9)))
    return out


def sample_kinetics(base: KineticBase, rng, thermal_integral: Callable,
                    target: float, *, p_thresh: float = 0.05,
                    max_pressure_iter: Optional[int] = None,
                    on_zeff_without_dilution: Optional[Callable] = None
                    ) -> KineticDraw:
    """Draw kinetics whose thermal pressure integral matches *target*.

    *thermal_integral(draw)* returns the flux-surface integral of the draw's
    thermal pressure on the caller's equilibrium (*target*: the same for the
    base).  *p_thresh* is a FRACTION; the error is mean ``|target - p| /
    target`` in percent.  Raises RuntimeError after *max_pressure_iter*
    tries.
    """
    from .sampling import (_MAX_PRESSURE_ITER, _draw_monotonic_perturbation,
                           draw_bounded_perturbation, generate_perturbed_GPR)
    if max_pressure_iter is None:
        max_pressure_iter = _MAX_PRESSURE_ITER
    b = base
    x = b.psi_kin
    aux_sigmas = b.aux_sigmas or {}
    aux_baselines = b.aux_baselines or {}
    aux_ls = b.aux_length_scales or {}

    if b.sampling_mode not in ("gibbs_level", "stiff_inward"):
        raise ValueError(
            f"Unknown sampling_mode={b.sampling_mode!r}; expected "
            "'gibbs_level' or 'stiff_inward'"
        )
    if b.chi_consistency not in ("passive_gp", "power_balance", "surrogate"):
        raise ValueError(
            f"Unknown chi_consistency={b.chi_consistency!r}; expected "
            "'passive_gp', 'power_balance', or 'surrogate'"
        )

    _ls_ne = b.n_ls if b.ne_ls is None else b.ne_ls
    _ls_te = b.t_ls if b.te_ls is None else b.te_ls
    _ls_ni = b.n_ls if b.ni_ls is None else b.ni_ls
    _ls_ti = b.t_ls if b.ti_ls is None else b.ti_ls

    def _mono(prof, sig, ls):
        p0 = float(prof[0]) if float(prof[0]) > 0.0 else float(np.max(np.abs(prof)))
        if not (p0 > 0.0):
            p0 = 1.0
        mn = prof / p0
        smp = _draw_monotonic_perturbation(x, mn, sig / p0, ls, rng=rng)
        return prof + (smp - mn) * p0

    def _draw_primary(chan, prof, sig, ls):
        if b.sampling_mode == "gibbs_level" and b.gradient_policy == "legacy":
            return _mono(prof, sig, ls)
        if b.sampling_mode == "stiff_inward":
            from .stiff_sampler import sample_stiff_inward_channel
            dk_map = b.dkappa_allow or {}
            dk = dk_map.get(chan)
            if dk is None:
                dk = np.full_like(x, 1.5 if chan in ("te", "ti") else 1.2)
            ell_arr = (np.full_like(x, float(ls))
                       if np.ndim(ls) == 0 else np.asarray(ls, dtype=float))
            return sample_stiff_inward_channel(
                x, prof, sig, dk, ell_arr,
                r_norm=b.r_norm_kin,
                a_over_R0=float(b.a_over_R0),
                psi_ped_top=float(b.psi_ped_top),
                has_pedestal=bool(b.has_pedestal),
                ell_kappa=float(b.stiff_gradient_ls),
                enforce_ped_monotonic=(b.gradient_policy != "positivity_only"
                                       and bool(b.has_pedestal)
                                       and chan in ("te", "ti")),
                rng=rng,
            )
        p0 = float(np.max(np.abs(prof)))
        if not (p0 > 0):
            p0 = 1.0
        mn = prof / p0
        smp = draw_bounded_perturbation(
            x, mn, sig / p0, ls,
            policy=b.gradient_policy,
            channel=chan,
            psi_ped_top=float(b.psi_ped_top),
            has_pedestal=bool(b.has_pedestal),
            coord=str(b.coord),
            r_norm=b.r_norm_kin,
            a_over_R0=float(b.a_over_R0),
            kappa_bounds=(b.kappa_bounds or {}).get(chan),
            rng=rng,
        )
        return prof + (smp - mn) * p0

    def _gpr(prof, sig, ls):
        p0 = float(np.max(np.abs(prof)))
        if not (p0 > 0):
            p0 = 1.0
        mn = prof / p0
        smp = np.atleast_1d(np.asarray(np.squeeze(generate_perturbed_GPR(
            x, mn, sig / p0, length_scale=ls, n_samples=1, rng=rng)),
            dtype=float))
        return prof + (smp - mn) * p0

    Z_imp = b.resolved_Z_imp()
    want = b.zeff_channel and bool(b.ni_from_zeff)
    active = want and Z_imp is not None
    if want and Z_imp is None and on_zeff_without_dilution is not None:
        on_zeff_without_dilution()
    if active:
        from .physics import main_ion_density_from_zeff
        zb = np.asarray(aux_baselines["zeff"], dtype=float)
        zs = np.asarray(aux_sigmas["zeff"], dtype=float)
        if b.zeff_dne is not None:
            zs = np.sqrt(np.maximum(zs ** 2 - (b.zeff_dne * b.sigma_ne) ** 2,
                                    0.0))

        def _ni_of(ne_, z_):
            return main_ion_density_from_zeff(
                ne_, z_, Z_imp, z_fast=b.z_fast, z2_fast=b.z2_fast,
                zeff_includes_fast=b.zeff_includes_fast)
        _zb_lo, _zb_hi = b.zeff_window(b.ne, Z_imp)
        ni_of_base = _ni_of(b.ne, _clip_zeff(zb, _zb_lo, _zb_hi))

    thr = float(p_thresh) * 100.0
    p_err, n_iter = np.inf, 0
    zeff_draw = None
    clips = dict.fromkeys(CLIP_COUNTERS, 0)
    while p_err > thr:
        n_iter += 1
        if n_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} "
                f"iterations (last error {p_err:.2f}% vs threshold "
                f"{thr:.2f}%)")
        ne_d = _draw_primary("ne", b.ne, b.sigma_ne, _ls_ne)
        te_d = _draw_primary("te", b.te, b.sigma_te, _ls_te)
        clips = dict.fromkeys(CLIP_COUNTERS, 0)     # this candidate's
        if active:
            zeff_draw = _gpr(zb, zs, aux_ls.get("zeff", 0.4))
            if b.zeff_dne is not None:
                zeff_draw = zeff_draw + b.zeff_dne * (ne_d - b.ne)
            _lo, _hi = b.zeff_window(ne_d, Z_imp)
            clips.update(_count_zeff_clips(
                zeff_draw, b.zeff_bounds_raw(ne_d, Z_imp)[0], _lo, _hi))
            zeff_draw = _clip_zeff(zeff_draw, _lo, _hi)
            ni_d = b.ni + (_ni_of(ne_d, zeff_draw) - ni_of_base)
            if b.clips:
                _cap = b.ni_ceiling(ne_d)
                clips["ni_floor_0"] = int(np.count_nonzero(ni_d < 0.0))
                clips["ni_ceiling"] = int(np.count_nonzero(ni_d > _cap))
                ni_d = np.clip(ni_d, 0.0, _cap)
        else:
            ni_d = _draw_primary("ni", b.ni, b.sigma_ni, _ls_ni)
            if b.clips and b.Z_imp:
                _cap = b.ni_ceiling(ne_d)
                clips["ni_ceiling"] = int(np.count_nonzero(ni_d > _cap))
                ni_d = np.minimum(ni_d, _cap)
        ti_d = _draw_primary("ti", b.ti, b.sigma_ti, _ls_ti)
        draw = KineticDraw(ne=ne_d, te=te_d, ni=ni_d, ti=ti_d)
        p = float(thermal_integral(draw))
        p_err = float(np.mean(np.abs(target - p) / target) * 100.0)

    aux_out = {}
    for name, es in aux_sigmas.items():
        if name == "zeff" and zeff_draw is not None:
            continue
        if name in ("chi_e", "chi_i") and b.chi_consistency != "passive_gp":
            continue
        eb = aux_baselines.get(name)
        if eb is None:
            continue
        ep = _gpr(np.asarray(eb, dtype=float), np.asarray(es, dtype=float),
                  aux_ls.get(name, 0.4))
        if name == "zeff" and b.clips:
            _lo, _hi = b.zeff_window(ne_d, Z_imp)
            clips.update(_count_zeff_clips(
                ep, b.zeff_bounds_raw(ne_d, Z_imp)[0], _lo, _hi,
                prefix="aux_"))
            ep = _clip_zeff(ep, _lo, _hi)
        aux_out[name] = ep
    if zeff_draw is not None:
        aux_out["zeff"] = zeff_draw

    if b.chi_consistency in ("power_balance", "surrogate"):
        vol_kin = getattr(b.miller_kin, "volume_m3", None)
        surf_kin = getattr(b.miller_kin, "surface_m2", None)
        rmin_kin = getattr(b.miller_kin, "r_min_m", None)
        f_trit = float(np.clip(b.tritium_fraction, 0.0, 1.0))
        a_main = 2.0 * (1.0 - f_trit) + 3.0 * f_trit

        z0 = np.asarray(aux_baselines.get("zeff", np.full_like(x, 1.5)), dtype=float)
        zd = np.asarray(aux_out.get("zeff", z0), dtype=float)
        w0 = (b.omega_tor_rad_s if b.omega_tor_rad_s is not None
              else aux_baselines.get("omega_tor"))
        w0_arr = None if w0 is None else np.asarray(w0, dtype=float)
        wd = aux_out.get("omega_tor", w0_arr)
        wd_arr = None if wd is None else np.asarray(wd, dtype=float)

        _state_unchanged = (
            np.array_equal(ne_d, b.ne)
            and np.array_equal(te_d, b.te)
            and np.array_equal(ni_d, b.ni)
            and np.array_equal(ti_d, b.ti)
            and np.array_equal(zd, z0)
            and ((w0_arr is None and wd_arr is None)
                 or (w0_arr is not None and wd_arr is not None
                     and np.array_equal(wd_arr, w0_arr)))
        )

        surr_base = surr_draw = None
        mg_c = b.miller_kin
        if (b.chi_consistency == "surrogate"
                and not _state_unchanged
                and any(aux_baselines.get(cn) is not None for cn in ("chi_e", "chi_i"))):
            from .geometry.miller import miller_from_baseline
            from .surrogates.tglfnn import evaluate_tglfnn_stiffness
            if mg_c is None:
                from types import SimpleNamespace
                mg_c = miller_from_baseline(SimpleNamespace(
                    psi_N_kinetic=x, Ip_target=1.0e6, l_i_target=1.0, aux=None
                ))
            _psi_surr = (
                mg_c.psi_N
                if (getattr(mg_c, "psi_N", None) is not None
                    and np.shape(mg_c.psi_N) == x.shape)
                else x
            )
            zc = float(Z_imp or 6.0)
            surr_base = evaluate_tglfnn_stiffness(
                mg_c, _psi_surr, b.ne, b.te, b.ni, b.ti, z0,
                Z_imp=zc, A_main=a_main, z_fast_m3=b.z_fast,
                omega_tor_rad_s=w0_arr, model_spec=b.tglfnn_model,
            )
            surr_draw = evaluate_tglfnn_stiffness(
                mg_c, _psi_surr, ne_d, te_d, ni_d, ti_d, zd,
                Z_imp=zc, A_main=a_main, z_fast_m3=b.z_fast,
                omega_tor_rad_s=wd_arr, model_spec=b.tglfnn_model,
            )

        for cname, sp, nb, tb, nd, td in (
            ("chi_e", "e", b.ne, b.te, ne_d, te_d),
            ("chi_i", "i", b.ni, b.ti, ni_d, ti_d),
        ):
            eb = aux_baselines.get(cname)
            if eb is None:
                continue
            eb_arr = np.asarray(eb, dtype=float)
            q_mult = None
            if cname in aux_sigmas:
                es_arr = np.asarray(aux_sigmas[cname], dtype=float)
                rel_sig = np.clip(es_arr / np.maximum(np.abs(eb_arr), 1e-12), 0.0, 1.0)
                q_mult = _gpr(np.ones_like(x), rel_sig, aux_ls.get(cname, 0.5))
            if b.chi_consistency == "power_balance":
                aux_out[cname] = compute_power_balance_chi(
                    x, eb_arr, nb, tb, nd, td,
                    species=sp,
                    q_source_mult=q_mult,
                    alpha_heating_feedback=bool(b.alpha_heating_feedback),
                    tritium_fraction=f_trit,
                    ni_base=b.ni, ti_base=b.ti, ni_draw=ni_d, ti_draw=ti_d,
                    te_base=b.te, te_draw=te_d,
                    volume_m3=vol_kin,
                    surface_m2=surf_kin,
                    r_min_m=rmin_kin,
                )
            else:
                if _state_unchanged and (q_mult is None or np.allclose(q_mult, 1.0, atol=0.0)):
                    aux_out[cname] = eb_arr.copy()
                else:
                    qm = np.ones_like(x) if q_mult is None else np.maximum(q_mult, 0.1)
                    if _state_unchanged:
                        q_tot_mult = qm
                    else:
                        from .geometry.miller import gyrobohm_flux_scales
                        q0_w = surr_base.Qe_W_m2 if sp == "e" else surr_base.Qi_W_m2
                        qd_w = surr_draw.Qe_W_m2 if sp == "e" else surr_draw.Qi_W_m2
                        gb0 = gyrobohm_flux_scales(mg_c, b.ne, b.te)
                        q_floor = np.maximum(0.02 * gb0["Q_gB_imas_W_m2"], 1.0)
                        q_tot_mult = ((qd_w + q_floor) / (q0_w + q_floor)) * qm
                    aux_out[cname] = compute_power_balance_chi(
                        x, eb_arr, nb, tb, nd, td,
                        species=sp,
                        q_source_mult=q_tot_mult,
                        alpha_heating_feedback=False,
                        tritium_fraction=f_trit,
                        volume_m3=vol_kin,
                        surface_m2=surf_kin,
                        r_min_m=rmin_kin,
                        ratio_bounds=(0.1, 15.0),
                    )

    draw.aux = aux_out
    draw.zeff = aux_out.get("zeff")
    draw.zeff_primary = zeff_draw is not None
    draw.iterations = int(n_iter)
    draw.p_err_pct = float(p_err)
    draw.clips = clips
    draw.clips_enabled = bool(b.clips)
    draw.Z_imp = None if Z_imp is None else float(Z_imp)
    draw.Z_imp_origin = (None if Z_imp is None
                         else "declared" if b.Z_imp else "inverted")
    if (b.sampling_mode != "gibbs_level" or b.gradient_policy != "legacy"
            or b.chi_consistency != "passive_gp"):
        draw.physics_sampling = {
            "sampling_mode": str(b.sampling_mode),
            "gradient_policy": str(b.gradient_policy),
            "chi_consistency": str(b.chi_consistency),
            "alpha_heating_feedback": bool(b.alpha_heating_feedback),
            "tritium_fraction": float(b.tritium_fraction),
            "has_pedestal": bool(b.has_pedestal),
        }
    if draw.clipped:
        print("  [kinetic_sampler] clip(s) fired on this draw (nodes moved): "
              + ", ".join(f"{k}={v}" for k, v in clips.items() if v))
    return draw
