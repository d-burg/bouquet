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
the reported current and what the magnetics want.  It is a property of the
reconstruction's null-space choice, not of the machine: a different code
lands on a different offset.  The DEFAULT model (``"random"``) is therefore
fitted to ``std`` -- the part of the residual that is stable from shot to shot
and independent of which fit produced the baseline -- and answers "how far
can this coil plausibly move for a plausible change of the plasma state".
``"rms_incl_offset"`` is fitted to ``rms`` and is the looser option.  NOTE:
this is NOT because the baseline "absorbs" EFIT's offset -- see below.  Both are floor+fraction fits, sigma^2 = floor^2 + (fraction*|I|)^2,
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

Scope of the calibration -- READ BEFORE TRUSTING THE QUANTILES
-------------------------------------------------------------
Both the sigma model and the acceptance quantiles were measured over the
**18 F-coils only**, because those are the coils the reconstruction lets float
against the magnetics.  The filter, however, assigns a sigma to EVERY baseline
coil: the E-coils ride on the F-coil-derived era floor plus the same fractional
term, which is an EXTRAPOLATION (the reconstruction holds the E-coils fixed, so
they contribute no residual to fit).  Two consequences, neither of which is
absorbed by the numbers below:

* ``chi2/nu`` pools calibrated and extrapolated terms, so the pooled statistic
  is not the random variable the quantile was measured on;
* ``max|z|`` is an ORDER STATISTIC.  Its 95th percentile depends on how many
  coils the maximum runs over, and the shipped mesh judges 20 (18 F + 2 E)
  while the finer signature judges 24.  Applying an 18-coil quantile to a
  20- or 24-coil maximum gives a false-rejection rate ABOVE the nominal 5 %.

``DeviceSpec.acceptance["calibrated_nu"]`` records the coil count the
thresholds were measured at; the filter records the count it actually used and
warns when they differ.  The thresholds themselves are left at their calibrated
values -- moving an acceptance threshold is a decision for the operator, not a
side effect of a mesh choice.

One further transfer assumption, recorded rather than fixed: the calibration
population is (reconstruction-calculated minus measured) coil current, while
the filter measures (draw minus baseline).  These are different random
variables; the adopted sigma is used as a yardstick for both.

Baseline vs measured currents (checked 2026-09-03, 42 slices x 18 coils): an
UNREGULARISED TokaMaker baseline (``coil_reg`` empty) sits a median 11.6 kA-t
from the measured currents, ~10x EFIT's own offset, essentially uncorrelated
with it (r^2 = 0.05, same sign 68 %), and 25-30 adopted sigma away -- 86 % of
its coils would fail this filter's own guard.  The excursion is low-rank
(3 SVD modes, F4/F5/F8 and the F9A-F9B direction), i.e. the coil null space.
Draws about such a baseline are judged for plausibility RELATIVE TO IT; the
baseline's own distance from the magnetics is invisible to this filter and
must be handled upstream, by regularising the baseline toward the measured
currents (``SolverConfig.coil_reg``, see ``coil_targets``).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

__all__ = ["DeviceSpec", "DEVICES", "detect_device", "resolve_device", "get_device",
           "tolerance_for", "era_labels", "era_for_pulse", "GENERIC_ACCEPTANCE"]


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    coil_signature: frozenset                 # exact coil-set names in the mesh
    # coil-current tolerance model sigma_i = hypot(floor, fraction*|I_i|), baseline units.
    # This is the RANDOM (offset-removed) part of the reconstruction's coil-current
    # residual.  It is NOT that the baseline "absorbs" the reconstruction's per-pulse
    # systematic offset -- an unregularised baseline demonstrably does not (it sits
    # 25-30 sigma away along the coil null space; see the module docstring).  The
    # offset is excluded because it is a property of whichever fit produced the
    # baseline, so it is not a fair yardstick for how far a DRAW about that baseline
    # may move; regularising the baseline toward the measured currents
    # (``SolverConfig.coil_reg``) is the fix for the offset itself.
    # The floor may depend on the acquisition era: give sigma_floor_by_era as
    # ((pulse_lo, pulse_hi, floor, era_label), ...) and sigma_floor is the default
    # when the era is unknown.
    sigma_floor: float                        # [A-t]
    sigma_fraction: float
    sigma_provenance: str
    # additional exact signatures for other meshes of the same device (e.g. finer
    # meshes that split a coil into separately-driven circuits)
    alt_signatures: Tuple[frozenset, ...] = ()
    # Era bands, ordered; the LAST band is the default when the era is unknown.
    # The pulse bounds are era BOUNDARIES (the pulse index at which the coil-current
    # acquisition hardware changed), not discharges of interest -- use
    # ``era_for_pulse`` to map an explicitly-known pulse onto a label.
    sigma_floor_by_era: Tuple[Tuple[float, float, float, str], ...] = ()
    # per-coil floors by era label (coils absent from the table use the era floor);
    # clipped below at sigma_floor_min[era] so a coil near zero current keeps a floor
    sigma_floor_by_coil: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sigma_floor_min: Dict[str, float] = field(default_factory=dict)
    # alternative models a user may select by name via filtering.coil_sigma="<name>"
    sigma_models: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # acceptance thresholds calibrated on the machine's own residual distribution:
    # the chosen quantile of chi2/nu and worst-coil |z| that real flat-top slices
    # score under this device's sigma model. None -> the generic defaults.
    # "calibrated_nu" is the number of coils the quantiles were measured over;
    # max|z| is an order statistic, so applying them to a different coil count
    # changes their meaning (the filter records both counts and warns).
    acceptance: Dict[str, float] = field(default_factory=dict)   # {"chi2_max","z_max","quantile","calibrated_nu"}
    acceptance_provenance: str = ""
    # measured-current conventions (used by the coil-target and dd-referenced paths)
    turns: Dict[str, float] = field(default_factory=dict)       # measured A -> mesh A-t
    coil_family: Callable[[str], str] = lambda n: n[:1]         # for per-family tables
    digitizer_sigma: Dict[str, float] = field(default_factory=dict)  # per family, A (measured)
    vsc_pair: Tuple[str, ...] = ()


# Generic acceptance when no device calibration is available (a Gaussian-ish
# "RMS 2 sigma per coil, no coil beyond 5" rule; DIII-D's empirical values are looser).
GENERIC_ACCEPTANCE = {"chi2_max": 4.0, "z_max": 5.0}

_D3D_F = [f"F{i}{s}" for i in range(1, 10) for s in "AB"]

#: Era boundary for the DIII-D coil-current tolerance model: the pulse index at
#: which the coil-current acquisition (digitiser/DAQ) was upgraded and the
#: residual floor dropped by ~2.5x.  This is a BOUNDARY between two hardware
#: eras, not a discharge -- it is a number only because pulse index is the only
#: monotone clock the archive carries.  Use :func:`era_for_pulse` to map an
#: explicitly-known pulse onto an era label; nothing infers it from a file name.
#:
#: The pulse index is a DATE PROXY and only that: the upgrade happened on a date,
#: and date -> pulse index is what turns it into a number here.  The boundary is
#: therefore APPROXIMATE -- a pulse within a commissioning period either side of
#: it may carry the other era's acquisition -- and it is a step standing in for a
#: changeover.  What the era buys is the sigma FLOOR of the coil chi2 filter:
#: 825 A-t below the boundary (``"pre2014"``), 325 A-t at or above it
#: (``"modern"``), refined by the per-coil tables in ``sigma_floor_by_coil``.
#: That is an acceptance criterion, so :meth:`bouquet.run.Bouquet.filter` prints
#: the era it resolved and where it came from, once per call; set
#: ``filtering.coil_daq_era`` to an era label to state it explicitly and bypass
#: the proxy entirely.
D3D_DAQ_UPGRADE_PULSE = 165000

DEVICES: Dict[str, DeviceSpec] = {
    "DIII-D": DeviceSpec(
        name="DIII-D",
        coil_signature=frozenset(_D3D_F + ["ECOILA", "ECOILB"]),
        # xia_v1 mesh: the E-coil split into the six EFIT E circuits
        alt_signatures=(frozenset(_D3D_F + ["ECOILA", "ECOILB", "E567UP", "E567DN", "E89UP", "E89DN"]),),
        sigma_floor=325.0, sigma_fraction=0.0035,
        sigma_floor_by_era=((0, D3D_DAQ_UPGRADE_PULSE, 825.0, "pre2014"),
                            (D3D_DAQ_UPGRADE_PULSE, float("inf"), 325.0, "modern")),
        # per-coil random floor with the 0.35 percent fraction removed in quadrature, median
        # over pulses (408 pre-upgrade / 63 post-upgrade); F6A/F6B and F9A carry the largest scatter
        sigma_floor_by_coil={"pre2014": {"F1A": 1170, "F2A": 1040, "F3A": 910, "F4A": 1060, "F5A": 740, "F6A": 650, "F7A": 250, "F8A": 420, "F9A": 570, "F1B": 1470, "F2B": 780, "F3B": 1000, "F4B": 1170, "F5B": 1070, "F6B": 1120, "F7B": 0, "F8B": 240, "F9B": 370},
                             "modern": {"F1A": 0, "F2A": 0, "F3A": 160, "F4A": 0, "F5A": 0, "F6A": 840, "F7A": 0, "F8A": 230, "F9A": 580, "F1B": 0, "F2B": 0, "F3B": 0, "F4B": 0, "F5B": 0, "F6B": 780, "F7B": 0, "F8B": 0, "F9B": 180}},
        # Lower clip on the per-coil floor, adopted rather than fitted: several
        # coils fit a per-coil floor of exactly 0, which would leave their sigma
        # as the fractional term alone and make a quiet coil arbitrarily hard to
        # satisfy. The clip is ~1/3 of the era floor in both eras. It only makes
        # sigma LARGER for the affected coils, so it is a documented softening of
        # the per-coil table toward the era floor, not a change to the acceptance
        # thresholds; revisit it if the per-coil floors are ever refitted.
        sigma_floor_min={"pre2014": 250.0, "modern": 100.0},
        sigma_models={"random": (325.0, 0.0035), "random_pre2014": (825.0, 0.0030),
                      "rms_incl_offset": (1050.0, 0.0088)},
        sigma_provenance=("offset-removed std of reconstruction calculated-minus-measured "
                          "F-coil current over the flat-top (coil fit weights zeroed), "
                          "per-pulse floor+fraction fits over 497 DIII-D pulses drawn from two "
                          "flat-top survey sets: fraction ~0.3% in every era; floor ~800 A-t "
                          "before the coil-current DAQ upgrade and ~300 A-t after. "
                          "'rms_incl_offset' (1050 + 0.88%, 4 pulses) also includes the "
                          "per-pulse reported-current bias. Calibrated on the 18 F-coils only; "
                          "E-coils are carried on the same model as a stated extrapolation. "
                          "2026-09-02"),
        turns={**{f"F{i}{s}": 58.0 for i in (1, 2, 3, 4, 5, 8) for s in "AB"},
               **{f"F{i}{s}": 55.0 for i in (6, 7, 9) for s in "AB"}},
        digitizer_sigma={"F": 7.0, "E": 69.0},
        vsc_pair=("F9A", "F9B"),
        acceptance={"chi2_max": 6.1, "z_max": 6.3, "quantile": 0.95, "calibrated_nu": 18},
        acceptance_provenance=("95th percentile of chi2/nu and worst-coil |z| scored by real DIII-D "
                               "flat-top slices (r(t)-mean per coil over the adopted per-coil era "
                               "sigma; 16573 slices from the larger of the two flat-top survey sets "
                               "-- the second set was NOT pooled into the quantile): a 5% "
                               "false-rejection rate on real machine states by construction. "
                               "The empirical distribution is "
                               "not a chi2 of any dof (q50 0.78, q95 6.1, q99 51; the tail is slow "
                               "drift), so dof-based calibration is not used. Measured over the 18 "
                               "F-coils ('calibrated_nu'); max|z| is an order statistic, so these "
                               "quantiles are tied to that coil count -- see the module docstring. "
                               "2026-09-04"),
    ),
}


def era_labels(spec: DeviceSpec) -> Tuple[str, ...]:
    """The tolerance-era labels *spec* defines, in order (() if it has none)."""
    return tuple(lab for _lo, _hi, _fl, lab in spec.sigma_floor_by_era)


def era_for_pulse(spec: DeviceSpec, pulse) -> Optional[str]:
    """Era label whose pulse band contains *pulse*, else None.

    The ONLY sanctioned way to turn a pulse number into an era.  *pulse* must be
    a number the caller actually knows (an explicit source field or config
    setting) -- never digits scraped out of a file name or run header, which is
    how a mesh resolution or a date used to buy a 2.5x looser tolerance floor.

    The band bounds are DATE PROXIES: an acquisition upgrade happens on a date
    and the pulse index is merely the monotone clock the archive carries, so a
    boundary is approximate and a pulse close to one may belong to the other
    era.  The era chooses the sigma floor of the coil chi2 filter (DIII-D:
    825 A-t ``"pre2014"`` / 325 A-t ``"modern"``), i.e. an acceptance criterion,
    so the caller is expected to state which era it resolved and how
    (:meth:`bouquet.run.Bouquet.filter` prints exactly that, once per call).  A
    user who knows better sets ``filtering.coil_daq_era``, which wins over this
    mapping.
    """
    if pulse is None:
        return None
    try:
        p = float(pulse)
    except (TypeError, ValueError):
        return None
    for lo, hi, _fl, lab in spec.sigma_floor_by_era:
        if lo <= p < hi:
            return lab
    return None


def tolerance_for(spec: DeviceSpec, era: Optional[str] = None,
                  model: Optional[str] = None):
    """(floor, fraction, floor_by_coil, era) for *spec*.

    A named alternative model gives (floor, fraction, {}, model).  Otherwise the
    per-coil floor table for *era* is returned, clipped below at
    ``sigma_floor_min[era]``; coils absent from the table get the era floor.

    *era* must be one of :func:`era_labels`.  ``None`` means "era unknown" and
    selects the device's default band (the last one, which is the most recent
    and carries the TIGHTEST floor -- an unknown era must never buy a looser
    tolerance); callers are expected to say so out loud.
    """
    if model is not None:
        try:
            fl, fr = spec.sigma_models[model]
        except KeyError:
            raise KeyError(f"device {spec.name!r} has no sigma model {model!r}; "
                           f"available: {sorted(spec.sigma_models)}") from None
        return fl, fr, {}, model
    floor, resolved = spec.sigma_floor, None
    bands = spec.sigma_floor_by_era
    if bands:
        if era is None:
            _lo, _hi, floor, resolved = bands[-1]        # default: the latest era
        else:
            known = era_labels(spec)
            if era not in known:
                raise KeyError(f"device {spec.name!r} has no tolerance era {era!r}; "
                               f"available: {sorted(known)}")
            for _lo, _hi, fl, lab in bands:
                if lab == era:
                    floor, resolved = fl, lab
                    break
    fmin = spec.sigma_floor_min.get(resolved, 0.0) if resolved else 0.0
    by_coil = ({c: max(float(v), fmin)
                for c, v in spec.sigma_floor_by_coil.get(resolved, {}).items()}
               if resolved else {})
    return floor, spec.sigma_fraction, by_coil, resolved


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
