"""Device registry: everything that is machine-specific lives here, keyed by device name.

A TokaMaker mesh carries no device identifier (only ``coil_dict``/``cond_dict``),
so a device is either named in ``BouquetConfig.device`` or detected from the
exact set of coil-set names in the mesh.  Detection is all-or-nothing: an exact
signature match is full confidence, anything else is "unknown".
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

__all__ = ["DeviceSpec", "DEVICES", "detect_device", "resolve_device", "get_device"]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    coil_signature: frozenset                 # exact coil-set names in the mesh
    # coil-current tolerance model sigma_i = hypot(floor, fraction*|I_i|), baseline units
    sigma_floor: float                        # [A-t]
    sigma_fraction: float
    sigma_provenance: str
    # measured-current conventions (used by the coil-target and dd-referenced paths)
    turns: Dict[str, float] = field(default_factory=dict)       # measured A -> mesh A-t
    coil_family: Callable[[str], str] = lambda n: n[:1]         # for per-family tables
    digitizer_sigma: Dict[str, float] = field(default_factory=dict)  # per family, A (measured)
    vsc_pair: Tuple[str, ...] = ()


_D3D_F = [f"F{i}{s}" for i in range(1, 10) for s in "AB"]
DEVICES: Dict[str, DeviceSpec] = {
    "DIII-D": DeviceSpec(
        name="DIII-D",
        coil_signature=frozenset(_D3D_F + ["ECOILA", "ECOILB"]),
        sigma_floor=1050.0, sigma_fraction=0.0088,
        sigma_provenance=("rms of EFIT calculated-minus-measured F-coil current over the "
                          "flat-top, 72 coil-shots (150000/171317/173982/174823, FWTFC=0), "
                          "fitted as hypot(floor, fraction*|I|); 2026-09-02"),
        turns={**{f"F{i}{s}": 58.0 for i in (1, 2, 3, 4, 5, 8) for s in "AB"},
               **{f"F{i}{s}": 55.0 for i in (6, 7, 9) for s in "AB"}},
        digitizer_sigma={"F": 7.0, "E": 69.0},
        vsc_pair=("F9A", "F9B"),
    ),
}


def detect_device(coil_names) -> Optional[str]:
    """Device whose coil signature EXACTLY matches *coil_names*, else None."""
    names = frozenset(str(n) for n in coil_names)
    for spec in DEVICES.values():
        if names == spec.coil_signature:
            return spec.name
    return None


def get_device(name: str) -> DeviceSpec:
    try:
        return DEVICES[name]
    except KeyError:
        raise KeyError(f"unknown device {name!r}; registered: {sorted(DEVICES)}") from None


def resolve_device(device: Optional[str], coil_names=None) -> Optional[DeviceSpec]:
    """Explicit name wins; otherwise detect from the coil names; None if neither."""
    if device is not None:
        return get_device(device)
    if coil_names is not None:
        d = detect_device(coil_names)
        if d is not None:
            return DEVICES[d]
    return None
