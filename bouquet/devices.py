"""Device registry: everything that is machine-specific lives here, keyed by device name.

A TokaMaker mesh carries no device identifier (only ``coil_dict``/``cond_dict``),
so a device is either named in ``BouquetConfig.device`` or detected from the
exact set of coil-set names in the mesh.  Detection is all-or-nothing: an exact
signature match is full confidence, anything else is "unknown".

Coil-current tolerance model -- how it was derived and what it assumes
----------------------------------------------------------------------
The chi2 coil filter compares each draw's coil currents with the baseline's,
coil by coil, in units of a per-coil sigma.  For DIII-D that sigma comes from
the reconstruction code's own coil-current residual: over the flat-top of each
shot, r(t) = (EFIT calculated coil current) - (measured coil current), in
ampere-turns, with the coil-current fit weights at zero so the coils float
against the magnetics.  Two scalars are formed per coil per shot:

* ``rms`` = sqrt(mean r^2) -- includes the shot's mean offset;
* ``std`` = scatter of r(t) about its own mean -- the offset removed.

The offset is a signed, per-shot, per-coil bias (typically 1-3 kA-t) between
the reported current and what the magnetics want.  The TokaMaker baseline fit
carries its own such offset, so draws about the baseline must not re-explore
it: the DEFAULT model (``"random"``) is fitted to ``std``.  ``"rms_incl_offset"``
is fitted to ``rms`` and is the looser "everything the plasma might have seen"
option.  Both are floor+fraction fits, sigma^2 = floor^2 + (fraction*|I|)^2,
across coils and shots (log-space least squares); per-coil floors are the
per-shot std with the universal fraction removed in quadrature, median over
shots per DAQ era.

What has been checked (2026-09-03, 2940 coil-shots, 296k slices): r(t) about
its mean is symmetric and Gaussian in the core (skew +0.1, |z|>2 fraction
4.4 % vs 4.55 %), so a symmetric sigma is defensible; but the far tail is
heavy (|z|>5 occurs ~750x the Gaussian rate), so the ``z_max = 5`` guard is
empirically a ~3.5-sigma cut.  About 63 % of the variance behind ``std`` is
slow coherent drift of the discrepancy within the flat-top, not
slice-to-slice scatter -- the F6A/F6B/F9A "floors" are drift, not noise.
``std`` is kept deliberately: a reconstruction is one slice and the drift is
present at that slice.  The 18 F-coil residuals are not independent (median
|pair correlation| 0.37, n_eff ~ 4), so chi2/nu is not an 18-dof statistic;
the max-|z| guard carries most of the discrimination.

Not yet verified: the baseline's own offset relative to the measured current
vs EFIT's on the same shots (in progress).
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
    # additional exact signatures for other meshes of the same device (e.g. finer
    # meshes that split a coil into separately-driven circuits)
    alt_signatures: Tuple[frozenset, ...] = ()
    sigma_floor_by_shot: Tuple[Tuple[float, float, float, str], ...] = ()   # (lo, hi, floor, era label)
    # per-coil floors by era label (coils absent from the table use the era floor);
    # clipped below at sigma_floor_min[era] so a coil near zero current keeps a floor
    sigma_floor_by_coil: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sigma_floor_min: Dict[str, float] = field(default_factory=dict)
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
        # xia_v1 mesh: the E-coil split into the six EFIT E circuits
        alt_signatures=(frozenset(_D3D_F + ["ECOILA", "ECOILB", "E567UP", "E567DN", "E89UP", "E89DN"]),),
        sigma_floor=325.0, sigma_fraction=0.0035,
        sigma_floor_by_shot=((0, 165000, 825.0, "pre2014"), (165000, float("inf"), 325.0, "modern")),
        # per-coil random floor with the 0.35 percent fraction removed in quadrature, median
        # over shots (408 pre-2014 / 63 modern); F6A/F6B and F9A carry the largest scatter
        sigma_floor_by_coil={"pre2014": {"F1A": 1170, "F2A": 1040, "F3A": 910, "F4A": 1060, "F5A": 740, "F6A": 650, "F7A": 250, "F8A": 420, "F9A": 570, "F1B": 1470, "F2B": 780, "F3B": 1000, "F4B": 1170, "F5B": 1070, "F6B": 1120, "F7B": 0, "F8B": 240, "F9B": 370},
                             "modern": {"F1A": 0, "F2A": 0, "F3A": 160, "F4A": 0, "F5A": 0, "F6A": 840, "F7A": 0, "F8A": 230, "F9A": 580, "F1B": 0, "F2B": 0, "F3B": 0, "F4B": 0, "F5B": 0, "F6B": 780, "F7B": 0, "F8B": 0, "F9B": 180}},
        sigma_floor_min={"pre2014": 250.0, "modern": 100.0},
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
    """(floor, fraction, floor_by_coil, era) for *spec*.

    A named alternative model gives (floor, fraction, {}, model).  Otherwise the
    era is chosen from *shot* (the default era -- the last band -- when the shot
    is unknown) and the per-coil floor table for that era is returned, clipped
    below at ``sigma_floor_min[era]``; coils absent from the table get the era
    floor.
    """
    if model is not None:
        try:
            fl, fr = spec.sigma_models[model]
        except KeyError:
            raise KeyError(f"device {spec.name!r} has no sigma model {model!r}; "
                           f"available: {sorted(spec.sigma_models)}") from None
        return fl, fr, {}, model
    floor, era = spec.sigma_floor, None
    bands = spec.sigma_floor_by_shot
    if bands:
        lo, hi, fl, era = bands[-1]; floor = fl            # default: the latest era
        if shot is not None:
            for lo, hi, fl, lab in bands:
                if lo <= float(shot) < hi:
                    floor, era = fl, lab
                    break
    fmin = spec.sigma_floor_min.get(era, 0.0) if era else 0.0
    by_coil = {c: max(float(v), fmin) for c, v in spec.sigma_floor_by_coil.get(era, {}).items()} if era else {}
    return floor, spec.sigma_fraction, by_coil, era


def detect_device(coil_names) -> Optional[str]:
    """Device whose coil signature EXACTLY matches *coil_names*, else None."""
    names = frozenset(str(n) for n in coil_names)
    for spec in DEVICES.values():
        if names == spec.coil_signature or names in spec.alt_signatures:
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
