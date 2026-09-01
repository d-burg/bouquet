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

``sigma_i^base = sigma_i^meas * |I_i^base| / |I_i^meas|`` carries the measured
fractional precision into whatever units the solver reports. That ratio is the
per-coil turns factor, so no turns table is needed, and the metric is invariant
under any per-coil rescaling shared by baseline and draw.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["coil_sigma_in_base_units", "coil_chi2"]

#: Below this measured current [A] the fractional precision is meaningless and
#: the coil is dropped from the metric rather than allowed to dominate it.
MIN_ABS_MEASURED_A = 50.0


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
