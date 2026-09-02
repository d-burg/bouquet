"""Measurement-referenced coil-drift metric.

The legacy in-spec test is ``max_i |dI_i / I_i| <= tol`` over F-coils only,
compared against the reconstructed baseline. Three problems:

* the measured precision is ABSOLUTE and roughly constant per coil family
  (DIII-D: ~7 A on F-coils, ~69 A on E-coils), so a flat fractional tolerance is
  ~17 sigma on a high-current coil and ~1.7 sigma on a low-current one;
* ``max`` discards a draw for one marginal coil while ignoring every other;
* E-coils are excluded, yet they carry the largest deviations in practice.

This module scores a draw by how far its coil currents sit from the baseline in
units of the MEASURED precision::

    z_i     = (I_i^draw - I_i^base) / sigma_i^base
    chi2/nu = mean_i z_i^2

``sigma_i^base = sigma_i^meas * |I_i^base| / |I_i^meas|`` rescales the measured
FRACTIONAL precision onto the baseline current, so

    z_i = (fractional drift from baseline) / (measured fractional precision)

i.e. "how many measurement precisions is this draw's drift". No turns table is
needed and the metric is invariant under any per-coil rescaling shared by
baseline and draw.

The assumption is that the measured fractional precision is a fair yardstick for
the solver's current -- NOT that ``|I^base|/|I^meas|`` is a fixed turns factor.
It is not: measured across 26 slices of DIII-D 174823 that ratio is stable only
for F6A/F6B/F7A/F7B (54-56, 8-12% spread) and varies 59-295% for 16 of 24 coils,
because the free-boundary solve fits the BOUNDARY and lands on one of many coil
sets consistent with it. Since sigma^base is formed per slice from that slice's
own baseline and measurement, the metric stays self-consistent regardless.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["coil_sigma_in_base_units", "coil_sigma_fixed", "coil_chi2",
           "SIGMA_REF_D3D_A", "with_sigma_ref",
           "EFIT_RESIDUAL_FLOOR_AT", "EFIT_RESIDUAL_FRACTION", "coil_sigma_efit_residual",
           "coil_sigma_floor_fraction", "resolve_coil_sigma", "CoilSigmaUnavailable"]

#: Below this measured current [A] the fractional precision is meaningless and
#: the coil is dropped from the metric rather than allowed to dominate it.
MIN_ABS_MEASURED_A = 50.0

# Fixed per-family measured sigma [A per turn], DIII-D 2017+ DAQ.  The OMAS
# d3d mapping writes pf_active data_error_upper as 10 digitizer LSB, which is
# ~8x larger on pre-2017 hardware (150000: 55 A F / 551 A E), so a chi2 cut
# referenced to the dd value changes meaning across DAQ epochs.  This table is
# what chi2_max=4.0 was calibrated against.
SIGMA_REF_D3D_A = {"F": 7.0, "E": 69.0}

# Machine tolerance from EFIT itself: rms of (calculated - measured) F-coil
# current over the flat-top, 72 coil-shots on DIII-D 150000/171317/173982/
# 174823 with FWTFC=0 (coils floating against the magnetics), fitted as
# sigma^2 = floor^2 + (fraction*|I|)^2 in ampere-turns.  Mostly a floor with a
# mild current dependence (log-log slope 0.36).  E-coils were not in the fit
# (EFIT holds them fixed); the same model is applied to them in their own
# baseline units as a stated assumption.
EFIT_RESIDUAL_FLOOR_AT = 1050.0
EFIT_RESIDUAL_FRACTION = 0.0088


def coil_sigma_floor_fraction(baseline, floor, fraction):
    """Per-coil sigma [baseline units] = hypot(floor, fraction*|I_base|).

    The intuitive two-number tolerance model ("about X A-t, plus Y % of the
    coil current").  Needs only the baseline currents; every coil gets a sigma.
    """
    return {n: float(np.hypot(float(floor), float(fraction) * abs(float(i))))
            for n, i in baseline.items()}


def coil_sigma_efit_residual(baseline, floor=EFIT_RESIDUAL_FLOOR_AT,
                             fraction=EFIT_RESIDUAL_FRACTION):
    """DIII-D EFIT-residual instance of :func:`coil_sigma_floor_fraction`."""
    return coil_sigma_floor_fraction(baseline, floor, fraction)


class CoilSigmaUnavailable(RuntimeError):
    """No per-coil tolerance could be resolved for this archive/device."""


def resolve_coil_sigma(baseline, sigma=None, device=None):
    """Resolve the per-coil sigma for a chi2 coil filter.  Returns (sigma, model)
    where *model* is a JSON-able provenance record.

    Resolution order (first that applies):
      1. ``sigma`` given explicitly --
         * ``{"floor": A-t, "fraction": f}``  -> floor+fraction model
         * ``{coil_name: sigma, ...}``       -> per-coil table (coils absent from
           the table are NOT judged; must cover >= 1 baseline coil)
         * callable(baseline) -> {coil: sigma}
      2. ``device`` (a :class:`DeviceSpec`, a device name, or None -> detected
         from the baseline coil names): the device's floor+fraction model.
      3. otherwise :class:`CoilSigmaUnavailable` -- the caller decides the
         fallback (Bouquet.filter falls back LOUDLY to the legacy rule).
    """
    from .devices import DeviceSpec, resolve_device
    if sigma is not None:
        if callable(sigma):
            out = {str(k): float(v) for k, v in sigma(baseline).items() if k in baseline}
            return out, {"kind": "callable", "n_coils": len(out)}
        if isinstance(sigma, dict) and {"floor", "fraction"} <= set(sigma):
            fl, fr = float(sigma["floor"]), float(sigma["fraction"])
            return coil_sigma_floor_fraction(baseline, fl, fr), {"kind": "floor_fraction", "floor": fl, "fraction": fr}
        if isinstance(sigma, dict):
            out = {str(k): float(v) for k, v in sigma.items() if k in baseline and float(v) > 0}
            if not out:
                raise CoilSigmaUnavailable("per-coil sigma table names none of the baseline coils")
            return out, {"kind": "per_coil", "n_coils": len(out)}
        raise TypeError("sigma must be {'floor','fraction'}, {coil: sigma}, or callable(baseline)")
    spec = device if isinstance(device, DeviceSpec) else resolve_device(device, baseline.keys())
    if spec is None:
        raise CoilSigmaUnavailable(
            "no coil-current tolerance available: the mesh coil names match no registered "
            f"device ({sorted(baseline)[:6]}...). Set BouquetConfig.device, or give "
            "filtering.coil_sigma = {'floor': <A-t>, 'fraction': <f>} (or a per-coil table).")
    return (coil_sigma_floor_fraction(baseline, spec.sigma_floor, spec.sigma_fraction),
            {"kind": "device", "device": spec.name, "floor": spec.sigma_floor,
             "fraction": spec.sigma_fraction, "provenance": spec.sigma_provenance})


def with_sigma_ref(measured, table=None):
    """Replace each coil's measured sigma by the per-family value in *table*
    (keyed by the coil name's first letter); coils with no family entry keep
    their own sigma.  ``table=None`` -> ``SIGMA_REF_D3D_A``."""
    table = SIGMA_REF_D3D_A if table is None else table
    return {n: (i, float(table.get(n[:1], s))) for n, (i, s) in measured.items()}


def coil_sigma_in_base_units(
    baseline: Dict[str, float],
    measured: Dict[str, Tuple[float, float]],
    min_abs_measured: float = MIN_ABS_MEASURED_A,
) -> Dict[str, float]:
    """Per-coil sigma expressed in the baseline's current units.

    Parameters
    ----------
    baseline : {name: current}
        Reconstructed baseline coil currents (any consistent unit).
    measured : {name: (current_A, sigma_A)}
        Measured current and its 1-sigma uncertainty, from ``pf_active``.
    min_abs_measured : float
        Coils whose |measured current| falls below this are omitted.

    Returns
    -------
    {name: sigma} for the coils common to both and above the floor.
    """
    out = {}
    for name, i_base in baseline.items():
        if name not in measured:
            continue
        i_meas, sigma = measured[name]
        if not np.isfinite(i_meas) or not np.isfinite(sigma) or sigma <= 0.0:
            continue
        if abs(i_meas) < min_abs_measured:
            continue
        out[name] = float(sigma) * abs(float(i_base)) / abs(float(i_meas))
    return out


def coil_chi2(
    draw: Dict[str, float],
    baseline: Dict[str, float],
    sigma: Dict[str, float],
    coils: Optional[Sequence[str]] = None,
) -> dict:
    """Reduced chi^2 of a draw's coil currents against the baseline.

    Returns ``{'chi2_nu', 'chi2', 'nu', 'max_abs_z', 'worst_coil', 'z'}``.
    ``nu`` is 0 (and ``chi2_nu`` NaN) when no coil is usable -- callers must
    treat that as "cannot judge", never as a pass.
    """
    names = [c for c in (coils if coils is not None else sigma)
             if c in draw and c in baseline and c in sigma and sigma[c] > 0.0]
    if not names:
        return {"chi2_nu": float("nan"), "chi2": float("nan"), "nu": 0,
                "max_abs_z": float("nan"), "worst_coil": None, "z": {}}
    z = {c: (float(draw[c]) - float(baseline[c])) / float(sigma[c]) for c in names}
    zv = np.array(list(z.values()), dtype=float)
    worst = names[int(np.argmax(np.abs(zv)))]
    return {"chi2_nu": float(np.mean(zv ** 2)), "chi2": float(np.sum(zv ** 2)),
            "nu": len(names), "max_abs_z": float(np.max(np.abs(zv))),
            "worst_coil": worst, "z": z}


def coil_sigma_fixed(samples, min_abs_measured=1000.0,
                     fallback_abs_measured=MIN_ABS_MEASURED_A):
    """Per-coil sigma held FIXED across a discharge, in baseline units.

    ``coil_sigma_in_base_units`` rescales by the INSTANTANEOUS
    ``|I^base|/|I^meas|``. When a coil's measured current passes near zero that
    ratio explodes and the coil is handed an absurdly loose tolerance -- on
    DIII-D 174823, F4B gets 9.7x and ECOILA/ECOILB ~8x their typical sigma on
    one slice, purely because a different current momentarily crossed zero.
    The reconstruction's own current is no less determined at those instants.

    Here the conversion factor is a robust median over the discharge, taken only
    from slices where the coil is carrying enough current for the ratio to mean
    anything, so sigma is constant in time:

        sigma_i = median_t(sigma_i^meas) * median_t(|I_i^base| / |I_i^meas|)

    Typical values are unchanged (within a few percent on 174823); only the
    outliers go away.

    Parameters
    ----------
    samples : {name: sequence of (i_measured, i_baseline, sigma_measured)}
    min_abs_measured : float
        Slices below this |measured current| are excluded from the ratio.
    fallback_abs_measured : float
        Relaxed floor used when fewer than 3 slices clear ``min_abs_measured``.

    Returns
    -------
    ({name: sigma}, {name: conversion_factor})
    """
    sigma, factor = {}, {}
    for name, rows in samples.items():
        arr = np.asarray([r for r in rows], dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        i_meas, i_base, sig_meas = arr[:, 0], arr[:, 1], arr[:, 2]
        good = np.abs(i_meas) > min_abs_measured
        if int(good.sum()) < 3:
            good = np.abs(i_meas) > fallback_abs_measured
        if int(good.sum()) < 1:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(i_base[good] / i_meas[good])
        ratio = ratio[np.isfinite(ratio)]
        s_m = sig_meas[np.isfinite(sig_meas) & (sig_meas > 0)]
        if not ratio.size or not s_m.size:
            continue
        factor[name] = float(np.median(ratio))
        sigma[name] = float(np.median(s_m)) * factor[name]
    return sigma, factor
