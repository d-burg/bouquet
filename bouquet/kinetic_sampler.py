"""One kinetic draw for every solve path (legacy, swb, unified engine).

Solver-free: the caller passes the thermal-pressure integral of a draw
(:class:`KineticDraw` -> float) and gets the drawn profiles on the kinetic
grid.  RNG order: ``ne``, ``Te``, then ``Z_eff`` (deriving ``ni``) or ``ni``,
then ``Ti`` -- redrawn together until the pressure integral is within
``p_thresh`` -- then each auxiliary channel.  Every profile is
``base + (sample - mean) * b0``, so a zero sigma returns the base exactly.
A drawn Z_eff is clipped to :func:`bouquet.physics.zeff_bounds` (floor 1); a
derived ``ni`` is an increment on the baseline ``ni``; a drawn ``ni`` is held
inside ``[0, ne - z_fast]`` when ``Z_imp`` is declared.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

#: drawn Z_eff floor (a Z_eff draw is never below this)
ZEFF_DRAW_MIN = 1.0


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
        """``(lo, hi)`` a Z_eff draw is clipped to on *ne* (``hi`` None: open)."""
        if Z_imp is None:
            return ZEFF_DRAW_MIN, None
        from .physics import zeff_bounds
        lo, hi = zeff_bounds(ne, Z_imp, self.z_fast, self.z2_fast,
                             self.zeff_includes_fast)
        return np.maximum(lo, ZEFF_DRAW_MIN), hi

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

    @property
    def native(self) -> dict:
        return dict(ne=self.ne, te=self.te, ni=self.ni, ti=self.ti)


def _clip_zeff(z, lo, hi):
    return np.clip(z, lo, None if hi is None else hi * (1.0 - 1e-9))


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
    while p_err > thr:
        n_iter += 1
        if n_iter > max_pressure_iter:
            raise RuntimeError(
                f"Pressure match not found within {max_pressure_iter} "
                f"iterations (last error {p_err:.2f}% vs threshold "
                f"{thr:.2f}%)")
        ne_d = _mono(b.ne, b.sigma_ne, b.n_ls)
        te_d = _mono(b.te, b.sigma_te, b.t_ls)
        if active:
            zeff_draw = _gpr(zb, zs, aux_ls.get("zeff", 0.4))
            if b.zeff_dne is not None:
                zeff_draw = zeff_draw + b.zeff_dne * (ne_d - b.ne)
            zeff_draw = _clip_zeff(zeff_draw, *b.zeff_window(ne_d, Z_imp))
            ni_d = b.ni + (_ni_of(ne_d, zeff_draw) - ni_of_base)
            ni_d = np.clip(ni_d, 0.0, b.ni_ceiling(ne_d))
        else:
            ni_d = _mono(b.ni, b.sigma_ni, b.n_ls)
            if b.Z_imp:
                ni_d = np.minimum(ni_d, b.ni_ceiling(ne_d))
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
        if name == "zeff":
            ep = _clip_zeff(ep, *b.zeff_window(ne_d, Z_imp))
        aux_out[name] = ep
    if zeff_draw is not None:
        aux_out["zeff"] = zeff_draw
    draw.aux = aux_out
    draw.zeff = aux_out.get("zeff")
    draw.zeff_primary = zeff_draw is not None
    draw.iterations = int(n_iter)
    draw.p_err_pct = float(p_err)
    return draw
