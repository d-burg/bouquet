"""The stored-energy pressure scale ``s_W`` of the unified engine.

The engine's Grad-Shafranov pressure is assembled from the kinetics,
``p_kin = e (n_e T_e + n_i T_i) + p_imp + p_fast``; the input's own pressure
(the g-file's ``PRES``, the IDS's ``equilibrium.profiles_1d.pressure``) never
enters the solve.  When a component is missing from the kinetics -- most
often the fast-ion pressure of an IDA-only run -- the reconstruction and every
draw sit low in stored energy.  This module measures the gap as one number
and, by default, closes it (owner decision 2026-11-24)::

    s_W = W_input / W_kinetic,    W = 1.5 * integral(p dV)

both stored energies of the FULL pressure (separatrix pressure included) on
the INPUT's own flux-surface volumes (g-file: the reader's
``volume_integral``; IDS: its own ``profiles_1d.volume``), computed before the
first solve (no extra solve).  ``s`` multiplies the TOTAL pressure handed to
Grad-Shafranov, so ``P'`` and the archived ``j_pressure`` follow; the kinetic
profiles, the Redl bootstrap, Z_eff and n_i are NOT scaled, and the unscaled
parts stay recorded (``EngineContract.pressure_parts``).  Draws carry the
reconstruction's ``s`` fixed; their pressure match stays on the unscaled
thermal integral.

``GenerationConfig.pressure_scale``: ``"match_W"`` (``s = s_W``; the
unified-engine default), ``"none"`` (``s = 1``, ``s_W`` still reported; the
1.4.1 pressure level) or a positive float (that ``s``, ``s_W`` still reported).
Under the legacy route and the experimental swb solve method the feature is
not applied (an explicit non-default value is refused;
:data:`bouquet.engine.ENGINE_DEPENDENT_DEFAULTS`).

A scalar fixes the LEVEL of the pressure, not its SHAPE: the record carries
the shape residual after scaling (relative RMS of ``s p_kin - p_in`` on the
input's nodes), and the report-only pressure-fidelity check of the g-file
route is computed on the scaled pressure, so it reports only what a scalar
cannot fix.
"""
from __future__ import annotations

import numbers

import numpy as np

#: The record's version.
PRESSURE_SCALE_VERSION = "pressure_scale/1"

#: The two named values of ``GenerationConfig.pressure_scale`` (a positive
#: float is the third form: a fixed ``s``).
PRESSURE_SCALE_MODES = ("match_W", "none")

#: The band outside which ``s_W`` WARNS (one UserWarning where the
#: reconstruction summary is printed), never refuses: ``s_W > 1.2`` or
#: ``s_W < 1/1.2``.  Owner decision 2026-11-24.  Measured on 17 real cases
#: (no solves): IDA-only runs without a fast-ion pressure have
#: ``W_input / W_kinetic`` = 1.18-1.72 (typically 1.24).
PRESSURE_SCALE_WARN_BAND = 1.2

#: How ``W`` is defined (stamped in every record).
STORED_ENERGY_DEFINITION = (
    "W = 1.5 * integral(p dV) of the FULL pressure (separatrix pressure "
    "included) on the input's own flux-surface volumes; W_input from the "
    "input's pressure, W_kinetic from the assembled kinetic pressure "
    "(thermal + impurity + fast) interpolated onto the same nodes")

#: The shape residual's definition (stamped in every record).
SHAPE_RESIDUAL_DEFINITION = (
    "relative RMS of s * p_kin - p_in over the input's pressure nodes: "
    "sqrt(mean((s p_kin - p_in)^2)) / sqrt(mean(p_in^2))")

#: Cause texts of the warning (:func:`pressure_scale_cause`).
CAUSE_NO_FAST = (
    "no fast-ion pressure was supplied; s is standing in for it as a uniform "
    "scale (fast-ion pressure is core-peaked) -- supply "
    "fixed_components.p_fast")
CAUSE_HIGH = ("a pressure component may be missing, or the input's pressure "
              "disagrees with the kinetics")
CAUSE_LOW = ("the kinetics exceed the input's pressure -- e.g. a fast-ion "
             "pressure counted twice (a doubled pb), a units error, or a "
             "wrong fast-ion reduction convention")


def check_pressure_scale_setting(value, who="generation.pressure_scale"):
    """Refuse a malformed ``pressure_scale`` value, by name: one of
    :data:`PRESSURE_SCALE_MODES`, a finite float > 0 (not a bool), or
    ``None`` (unset: resolved per engine)."""
    if value is None:
        return
    if isinstance(value, str):
        if value not in PRESSURE_SCALE_MODES:
            raise ValueError(f"{who}={value!r} must be one of "
                             f"{PRESSURE_SCALE_MODES}, a float > 0, or None "
                             "(resolved per engine: 'match_W' under the "
                             "unified engine)")
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            or not np.isfinite(float(value)) or not float(value) > 0.0):
        raise ValueError(f"{who}={value!r} must be one of "
                         f"{PRESSURE_SCALE_MODES}, a finite float > 0, or "
                         "None (resolved per engine)")


def resolve_pressure_scale_setting(value):
    """``(mode, fixed)`` of a ``pressure_scale`` value: ``("match_W",
    None)``, ``("none", None)`` or ``("fixed", s)``.  ``None`` (unset; the
    engine resolves it at ``prepare_baseline()``) reads as ``"match_W"``, the
    unified engine's default."""
    check_pressure_scale_setting(value)
    if value is None:
        return "match_W", None
    if isinstance(value, str):
        return value, None
    return "fixed", float(value)


def stored_energy(p, volume):
    """``1.5 * integral(p dV)`` [J] by the trapezoid rule over the
    cumulative volume *volume* [m^3] (axis first)."""
    p = np.asarray(p, dtype=float)
    v = np.asarray(volume, dtype=float)
    return float(1.5 * np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(v)))


def shape_residual(s, p_kin, p_in):
    """:data:`SHAPE_RESIDUAL_DEFINITION` (``nan`` for a zero input)."""
    p_kin = np.asarray(p_kin, dtype=float)
    p_in = np.asarray(p_in, dtype=float)
    den = float(np.sqrt(np.mean(p_in ** 2)))
    if not den > 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((float(s) * p_kin - p_in) ** 2)) / den)


def pressure_scale_cause(s_W, p_fast_supplied):
    """The likely cause named by the warning of an out-of-band *s_W*
    (``None`` inside the band)."""
    if not np.isfinite(s_W):
        return None
    if s_W > PRESSURE_SCALE_WARN_BAND:
        return CAUSE_HIGH if p_fast_supplied else CAUSE_NO_FAST
    if s_W < 1.0 / PRESSURE_SCALE_WARN_BAND:
        return CAUSE_LOW
    return None


def pressure_scale_record(setting, *, W_input, W_parts, p_kin_on_input,
                          p_input, p_fast_supplied, p_fast_source=None,
                          input_basis="", volume_source="",
                          unavailable=None):
    """The ``pressure_scale`` record and the ``s`` it applies.

    *setting*: the resolved ``GenerationConfig.pressure_scale``.
    *W_input*: the input's stored energy [J]; *W_parts*: the kinetic
    pressure's, by part (``thermal``, ``impurity``, ``fast``), on the same
    volumes.  *p_kin_on_input* / *p_input*: the kinetic and the input's
    pressure on the input's nodes (the shape residual).  *unavailable*: the
    reason ``s_W`` could not be computed (the IDS carries no volume), which
    leaves ``s = 1`` under ``"match_W"`` (a fixed float still applies) and
    is warned.  Never raises on the numbers."""
    mode, fixed = resolve_pressure_scale_setting(setting)
    W_parts = {k: float(v) for k, v in (W_parts or {}).items()}
    W_kin = float(sum(W_parts.values())) if W_parts else float("nan")
    s_W = float("nan")
    if unavailable is None:
        if not (np.isfinite(W_input) and np.isfinite(W_kin) and W_kin > 0.0
                and W_input > 0.0):
            unavailable = (f"non-positive or non-finite stored energy "
                           f"(W_input {W_input!r}, W_kinetic {W_kin!r})")
        else:
            s_W = float(W_input) / W_kin
    if mode == "match_W":
        s = s_W if unavailable is None else 1.0
    elif mode == "none":
        s = 1.0
    else:
        s = float(fixed)
    status = "unavailable" if unavailable is not None else "computed"
    rec = dict(
        version=PRESSURE_SCALE_VERSION,
        setting=(setting if isinstance(setting, str) or setting is None
                 else float(setting)),
        mode=mode, s=float(s), s_W=s_W, applied=bool(s != 1.0),
        status=status, unavailable_reason=unavailable,
        W_input_J=(float(W_input) if W_input is not None
                   else float("nan")),
        W_kinetic_J=W_kin, W_kinetic_parts_J=W_parts,
        W_scaled_J=(float(s) * W_kin if np.isfinite(W_kin)
                    else float("nan")),
        shape_residual=float("nan"), shape_residual_unscaled=float("nan"),
        shape_residual_at_s_W=float("nan"),
        warn_band=[1.0 / PRESSURE_SCALE_WARN_BAND, PRESSURE_SCALE_WARN_BAND],
        warn_flag=False, cause=None,
        p_fast_supplied=bool(p_fast_supplied), p_fast_source=p_fast_source,
        input_pressure=str(input_basis), volume_source=str(volume_source),
        definition=STORED_ENERGY_DEFINITION,
        shape_residual_definition=SHAPE_RESIDUAL_DEFINITION,
        scaled=("the total Grad-Shafranov pressure (thermal + impurity + "
                "fast) and with it P' and j_pressure; the kinetics, the "
                "Redl bootstrap, Z_eff and n_i are not scaled"),
        draws=("every draw carries this s fixed; its pressure match stays "
               "on the unscaled thermal integral"),
        owner_decision="2026-11-24")
    if p_kin_on_input is not None and p_input is not None:
        rec["shape_residual"] = shape_residual(s, p_kin_on_input, p_input)
        rec["shape_residual_unscaled"] = shape_residual(1.0, p_kin_on_input,
                                                        p_input)
        if np.isfinite(s_W):
            rec["shape_residual_at_s_W"] = shape_residual(
                s_W, p_kin_on_input, p_input)
    cause = pressure_scale_cause(s_W, p_fast_supplied)
    rec["warn_flag"] = bool(cause is not None)
    rec["cause"] = cause
    return rec


def warn_pressure_scale(record, stacklevel=2):
    """The ONE UserWarning of a ``pressure_scale`` record: ``s_W`` outside
    :data:`PRESSURE_SCALE_WARN_BAND` (its likely cause named), or ``s_W``
    unavailable; nothing otherwise.  Issued where the reconstruction summary
    is printed (as :func:`bouquet.baseline.warn_pressure_fidelity`).  Never a
    refusal."""
    import warnings
    if not isinstance(record, dict):
        return
    if record.get("status") == "unavailable":
        warnings.warn(
            f"pressure scale: s_W could not be computed "
            f"({record.get('unavailable_reason')}); the run uses s = "
            f"{record.get('s', 1.0):g} (pressure_scale="
            f"{record.get('setting')!r})", UserWarning,
            stacklevel=stacklevel + 1)
        return
    if not record.get("warn_flag"):
        return
    lo, hi = record.get("warn_band") or (1.0 / PRESSURE_SCALE_WARN_BAND,
                                         PRESSURE_SCALE_WARN_BAND)
    warnings.warn(
        f"pressure scale: s_W = W_input / W_kinetic = {record['s_W']:.3f} "
        f"(W_input {record['W_input_J'] / 1e6:.4g} MJ, W_kinetic "
        f"{record['W_kinetic_J'] / 1e6:.4g} MJ) is outside "
        f"[{lo:.3f}, {hi:.3f}]; the run uses s = {record['s']:.3f} "
        f"(pressure_scale={record.get('setting')!r}): {record['cause']}",
        UserWarning, stacklevel=stacklevel + 1)


def summary_line(record) -> str:
    """The one-line summary printed with the reconstruction summary."""
    if not isinstance(record, dict):
        return ""
    if record.get("status") == "unavailable":
        return (f"s = {record.get('s', 1.0):.4f} (mode {record.get('mode')}); "
                f"s_W unavailable: {record.get('unavailable_reason')}")
    return (f"s_W {record['s_W']:.4f} (W_input "
            f"{record['W_input_J'] / 1e6:.4f} MJ, W_kinetic "
            f"{record['W_kinetic_J'] / 1e6:.4f} MJ); applied s "
            f"{record['s']:.4f} (mode {record['mode']}); shape residual "
            f"{record['shape_residual']:.3f} (unscaled "
            f"{record['shape_residual_unscaled']:.3f})"
            + ("  ⚠" if record.get("warn_flag") else ""))


def scale_of(record) -> float:
    """``s`` of a record, 1.0 for ``None`` (an archive or a contract written
    before the feature)."""
    if not isinstance(record, dict):
        return 1.0
    try:
        s = float(record.get("s", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return s if np.isfinite(s) else 1.0
