"""Device registry: everything that is machine-specific lives here, keyed by device name.

A TokaMaker mesh carries no device identifier (only ``coil_dict``/``cond_dict``),
so a device is either named in ``BouquetConfig.device`` or detected from the
exact set of coil-set names in the mesh.  Detection is all-or-nothing: an exact
signature match is full confidence, anything else is "unknown".
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

__all__ = ["DeviceSpec", "DEVICES", "detect_device", "resolve_device", "get_device", "tolerance_for"]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    coil_signature: frozenset                 # exact coil-set names in the mesh
    # coil-current tolerance model sigma_i = hypot(floor, fraction*|I_i|), baseline units.
    # This is the RANDOM part of the reconstruction's coil-current residual (per-shot
    # systematic offsets are already absorbed by the baseline fit, so a draw about the
    # baseline must not re-explore them).  The floor may depend on the DAQ era: give
    # sigma_floor_by_shot as ((shot_lo, shot_hi, floor), ...) and sigma_floor is the
    # default when the shot is unknown.
    sigma_floor: float                        # [A-t]
    sigma_fraction: float
    sigma_provenance: str
    sigma_floor_by_shot: Tuple[Tuple[float, float, float], ...] = ()
    # alternative models a user may select by name via filtering.coil_sigma="<name>"
    sigma_models: Dict[str, Tuple[float, float]] = field(default_factory=dict)
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
        sigma_floor=325.0, sigma_fraction=0.0035,
        sigma_floor_by_shot=((0, 165000, 825.0), (165000, float("inf"), 325.0)),
        sigma_models={"random": (325.0, 0.0035), "random_pre2014": (825.0, 0.0030),
                      "rms_incl_offset": (1050.0, 0.0088)},
        sigma_provenance=("offset-removed std of EFIT calculated-minus-measured F-coil current "
                          "over the flat-top (FWTFC=0), per-shot floor+fraction fits over 497 "
                          "DIII-D shots (105-shot CTM set + 392-shot IBS set): fraction ~0.3% "
                          "in every era; floor ~800 A-t for shots < ~165000 and ~300 A-t after "
                          "(DAQ era). 'rms_incl_offset' (1050 + 0.88%, 4 shots) also includes "
                          "the per-shot reported-current bias; 2026-09-02"),
        turns={**{f"F{i}{s}": 58.0 for i in (1, 2, 3, 4, 5, 8) for s in "AB"},
               **{f"F{i}{s}": 55.0 for i in (6, 7, 9) for s in "AB"}},
        digitizer_sigma={"F": 7.0, "E": 69.0},
        vsc_pair=("F9A", "F9B"),
    ),
}


def tolerance_for(spec: DeviceSpec, shot=None, model: Optional[str] = None):
    """(floor, fraction) for *spec*: a named alternative model, else the era floor for
    *shot* (default floor when the shot is unknown)."""
    if model is not None:
        try:
            return spec.sigma_models[model]
        except KeyError:
            raise KeyError(f"device {spec.name!r} has no sigma model {model!r}; "
                           f"available: {sorted(spec.sigma_models)}") from None
    floor = spec.sigma_floor
    if shot is not None:
        for lo, hi, fl in spec.sigma_floor_by_shot:
            if lo <= float(shot) < hi:
                floor = fl
                break
    return floor, spec.sigma_fraction


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
