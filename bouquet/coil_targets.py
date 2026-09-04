"""Coil-current regularisation targets from measured currents.

bouquet's default regularisation pulls every coil toward ZERO at unit weight,
leaving the isoflux boundary as the only opposing constraint. The solve
therefore lands on one of many coil sets consistent with the boundary rather
than the one the machine actually ran: on DIII-D 174823 the resulting baseline
sits a median 15.4 % (max 178 %) from the measured currents, and the E-coil
circuits trade current among themselves nearly freely.

This module builds a ``SolverConfig.coil_reg`` spec that pins each coil to its
measured current instead, mirroring the shipped OFT DIII-D example.
"""

from typing import Dict, List, Optional

import numpy as np

__all__ = ["TURNFC_D3D", "coil_reg_from_measured", "measured_from_pf_active"]

#: DIII-D F-coil turns, EFIT ``dprobe.dat_20200406``::
#:
#:     TURNFC= 5*58.0, 2*55.0, 58.0, 55.0
#:            5*58.0, 2*55.0, 58.0, 55.0
#:
#: TokaMaker coil currents are ampere-turns, so a measured circuit current is
#: converted with these. E-coils are deliberately absent: the shipped D3D mesh
#: carries their turns itself (``coil_dict`` nturns sums to 61.0 for ECOILA and
#: ECOILB), which is the ``/61.0`` divisor in the OFT DIII-D example.
# Turn counts now live in the device registry (bouquet.devices); this name is kept
# as an alias so existing callers and configs keep working.
from .devices import get_device as _get_device
TURNFC_D3D = dict(_get_device("DIII-D").turns)


def measured_from_pf_active(dd_path: str, time_s: float) -> Dict[str, float]:
    """``{name: circuit_current_A}`` from an IMAS ``pf_active`` at *time_s*."""
    import json

    with open(dd_path) as fh:
        dd = json.load(fh)
    out = {}
    for c in dd.get("pf_active", {}).get("coil", []):
        name = c.get("name") or c.get("identifier")
        cur = c.get("current") or {}
        if not name or "data" not in cur or "time" not in cur:
            continue
        t = np.asarray(cur["time"], dtype=float)
        d = np.asarray(cur["data"], dtype=float)
        out[name] = float(d[int(np.argmin(np.abs(t - float(time_s))))])
    return out


def coil_reg_from_measured(measured: Dict[str, float],
                           weights: Optional[Dict[str, float]] = None,
                           default_weight: float = 1.0,
                           turns: Optional[Dict[str, float]] = None) -> List[dict]:
    """``SolverConfig.coil_reg`` spec pinning each coil to its measured current.

    Parameters
    ----------
    measured : {name: current}
        Measured circuit currents [A].
    weights : {name: weight}, optional
        Per-coil weights; coils absent fall back to *default_weight*.
    turns : {name: turns}, optional
        Circuit-amps -> solver-units conversion; default the DIII-D device
        registry's turns (:data:`TURNFC_D3D` is an alias of it).
        A coil with no entry converts at 1.0.
    """
    turns = TURNFC_D3D if turns is None else turns
    weights = weights or {}
    spec = []
    for name, i_circuit in measured.items():
        if not np.isfinite(i_circuit):
            continue
        spec.append({
            "coils": {name: 1.0},
            "target": float(i_circuit) * float(turns.get(name, 1.0)),
            "weight": float(weights.get(name, default_weight)),
        })
    return spec
