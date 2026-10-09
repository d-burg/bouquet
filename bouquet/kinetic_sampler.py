"""One kinetic draw for every solve path (legacy, swb, unified engine).

Solver-free: the caller passes the thermal-pressure integral of a draw
(:class:`KineticDraw` -> float) and gets the drawn profiles on the kinetic
grid.  RNG order: ``ne``, ``Te``, then ``Z_eff`` (deriving ``ni``) or ``ni``,
then ``Ti`` -- redrawn together until the pressure integral is within
``p_thresh`` -- then each auxiliary channel.  Every profile is
``base + (sample - mean) * b0``, so a zero sigma returns the base exactly.
A drawn Z_eff is clipped to :func:`bouquet.physics.zeff_bounds`; a derived
``ni`` is an increment on the baseline ``ni``.  The PR #56 clips -- the
Z_eff floor at 1, ``ni`` held inside ``[0, ne - z_fast]``, the passive Z_eff
aux clip -- are EXPERIMENTAL and opt-in (``KineticBase.clips``;
``bouquet.experimental.REGISTRY["kinetic_sampler_clips"]``).  Every clip
that moves a node is counted on the draw (:attr:`KineticDraw.clips`,
:meth:`KineticDraw.record`); the version is :data:`KINETIC_SAMPLER_VERSION`.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

#: drawn Z_eff floor of the EXPERIMENTAL clips (``KineticBase.clips``)
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
#: per draw (:attr:`KineticDraw.clips`).
#: ``/3`` (1.4.0): ``/2`` with the PR #56 clips (the floor at 1, the ``ni``
#: floor and ceiling, the passive Z_eff aux clip) EXPERIMENTAL and opt-in
#: (:attr:`KineticBase.clips`); by default a Z_eff draw is held in
#: :func:`bouquet.physics.zeff_bounds` alone (the window before PR #56) and
#: ``ni`` is not clipped.  With ``clips=True`` a draw equals ``/2``'s.  The
#: record says which (``clips_enabled``).
KINETIC_SAMPLER_VERSION = (
    "kinetic_sampler/3 (shared sampler; derived ni = increment on the "
    "baseline ni; Z_eff window = zeff_bounds; PR #56 clips opt-in; clips "
    "counted per draw)")

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
    #: EXPERIMENTAL PR #56 clips (the Z_eff floor at 1, ni in
    #: [0, ne - z_fast], the passive Z_eff aux clip); False keeps only the
    #: zeff_bounds window
    clips: bool = False

    def __post_init__(self):
        for k in ("psi_kin", "ne", "te", "ni", "ti", "sigma_ne", "sigma_te",
                  "sigma_ni", "sigma_ti"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=float))
        for k in ("z_fast", "z2_fast", "zeff_dne"):
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
        the experimental :attr:`clips`."""
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
    #: whether the experimental PR #56 clips were on (KineticBase.clips)
    clips_enabled: bool = False

    @property
    def native(self) -> dict:
        return dict(ne=self.ne, te=self.te, ni=self.ni, ti=self.ti)

    @property
    def clipped(self) -> bool:
        """True when any clip moved any node of this draw."""
        return any(int(v) > 0 for v in self.clips.values())

    def record(self) -> dict:
        """JSON-safe per-draw stamp: the sampler version, the pressure match
        and the clip counters (for the draw's archive record)."""
        return dict(version=KINETIC_SAMPLER_VERSION,
                    clips_enabled=bool(self.clips_enabled),
                    zeff_primary=bool(self.zeff_primary),
                    iterations=int(self.iterations),
                    p_err_pct=float(self.p_err_pct),
                    clipped=bool(self.clipped),
                    clips={k: int(v) for k, v in self.clips.items()})


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
    # lifted by the floor (the experimental clips) above zeff_bounds' edge
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
                           generate_perturbed_GPR)
    if max_pressure_iter is None:
        max_pressure_iter = _MAX_PRESSURE_ITER
    b = base
    x = b.psi_kin
    aux_sigmas = b.aux_sigmas or {}
    aux_baselines = b.aux_baselines or {}
    aux_ls = b.aux_length_scales or {}

    def _mono(prof, sig, ls):
        mn = prof / prof[0]
        smp = _draw_monotonic_perturbation(x, mn, sig / prof[0], ls, rng=rng)
        return prof + (smp - mn) * prof[0]

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
        ne_d = _mono(b.ne, b.sigma_ne, b.n_ls)
        te_d = _mono(b.te, b.sigma_te, b.t_ls)
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
            ni_d = _mono(b.ni, b.sigma_ni, b.n_ls)
            if b.clips and b.Z_imp:
                _cap = b.ni_ceiling(ne_d)
                clips["ni_ceiling"] = int(np.count_nonzero(ni_d > _cap))
                ni_d = np.minimum(ni_d, _cap)
        ti_d = _mono(b.ti, b.sigma_ti, b.t_ls)
        draw = KineticDraw(ne=ne_d, te=te_d, ni=ni_d, ti=ti_d)
        p = float(thermal_integral(draw))
        p_err = float(np.mean(np.abs(target - p) / target) * 100.0)

    aux_out = {}
    for name, es in aux_sigmas.items():
        if name == "zeff" and zeff_draw is not None:
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
    draw.aux = aux_out
    draw.zeff = aux_out.get("zeff")
    draw.zeff_primary = zeff_draw is not None
    draw.iterations = int(n_iter)
    draw.p_err_pct = float(p_err)
    draw.clips = clips
    draw.clips_enabled = bool(b.clips)
    if draw.clipped:
        print("  [kinetic_sampler] clip(s) fired on this draw (nodes moved): "
              + ", ".join(f"{k}={v}" for k, v in clips.items() if v))
    return draw
