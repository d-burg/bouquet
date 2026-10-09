"""The pressure handed to the Grad-Shafranov solver: ONE place.

Every solve of this package hands the solver a ``P'`` profile
(``{"type": "linterp", ...}``) and an axis-pressure target (``pax``).  The
solver builds the pressure by integrating ``P'`` INWARD from the plasma
boundary starting at ZERO, then rescales ``P'`` so the axis value equals
``pax``.  Two consequences, each behind one setting here:

``edge_pprime_pin`` (default ``True``: the behaviour before the setting)
    ``True`` sets the LAST node of ``P'`` (``psi_N = 1``) to zero, so ``P'``
    ramps linearly to zero across the final grid interval.  ``False`` leaves
    the profile's own derivative there (``P'`` then jumps to zero outside
    the plasma, which the solver's piecewise-linear flux function
    represents).  What changes with ``False``: the pressure-driven part of
    the edge current ``P' (<R> - F^2 <1/R>/<B^2>)`` is no longer forced to
    zero at the boundary, so for the same requested total ``<j_phi>`` the
    split between the ``P'`` and ``FF'`` terms in the last interval moves
    (``FF'`` carries less there), and the edge current and ``q95`` follow.
    Measured on the synthetic g-file example (unified engine; see
    docs/CHANGES_SUMMARY.md): the pressure-driven current at the boundary
    goes from 0.006 to 0.018 MA/m^2, ``<j_phi>`` at the last node from 0.076
    to 0.115 MA/m^2 with the ``FF'`` term changing sign there, and ``q95``
    rises by 0.002; ``l_i``, the core and the iteration counts do not move.

``separatrix_pressure`` (default ``"offset"`` since 2026-10-02; ``"legacy"`` is
the behaviour before the setting)
    ``"legacy"`` passes the FULL axis pressure as the target.  When the
    input pressure is not zero at ``psi_N = 1`` (``p_sep``), the solver's
    pressure -- which is zero there by construction -- reaches that target
    only by inflating ``P'`` everywhere by ``p_axis / (p_axis - p_sep)``;
    the reported ``beta`` and ``W_MHD`` are then those of a different
    pressure profile (``p_axis (p - p_sep) / (p_axis - p_sep)``).
    ``"offset"`` passes ``p_axis - p_sep`` as the target, so ``P'`` is the
    input's own, and ``p_sep`` is added back wherever pressure, ``beta`` or
    stored energy is REPORTED or DELIVERED (:func:`pressure_frames`, the
    ``lcfs_pressure`` of EVERY g-file bouquet writes: the archive's
    ``_baseline`` and each draw in ``generate()``, and the reconstruction's
    own through ``Bouquet.save_baseline_eqdsk`` /
    :func:`save_full_pressure_eqdsk`; a bare ``mygs.save_eqdsk`` writes the
    solver frame).  ``p_sep`` is the TOTAL pressure
    handed to the solver (thermal + impurity + fast, exactly the array the
    solve is built from) at its last node, :func:`separatrix_pressure_of`.

Where the model stops: a pressure that is ``p_sep`` just inside the boundary
and zero just outside is not physical.  The real separatrix pressure
continues into the scrape-off layer, which a vacuum-outside free-boundary
equilibrium cannot represent.  Only ``P'`` enters the Grad-Shafranov
equation, so the equilibrium inside the boundary is the one the input's
``P'`` asks for; the constant ``p_sep`` is bookkeeping for readers of the
pressure, not a force.

The DEFAULT and the PRE-CHANGE settings are two different things.  The
pre-change settings (:data:`PRE_CHANGE_EDGE_PRESSURE`: pin on, ``"legacy"``)
are the behaviour of the scattered ``pp["y"][-1] = 0.0`` / ``pax = p[0]``
sites before this module existed: at them every array this module returns
is bit for bit what those sites returned (the frozen-copy tests).  The
defaults (:data:`EDGE_PRESSURE_DEFAULTS`: pin on, ``"offset"``) differ from
them in the separatrix setting only -- an owner-approved PHYSICS change of
2026-10-02: on real g-file and IDS cases ``"offset"`` brought the full-frame
``beta_N`` / ``W_MHD`` closer to the input on every comparable g-file case
(by 1.4-8 points) and by ~0.5 points on IDS slices, converged on every case,
at the same cost, with ``l_i``, ``q`` and the current distances unchanged.
With ``p_sep = 0`` the two hand the solver the same arrays;
``separatrix_pressure="legacy"`` restores the pre-change numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ``GenerationConfig.separatrix_pressure`` values.
SEPARATRIX_PRESSURE_CHOICES = ("legacy", "offset")
#: The two settings BEFORE they existed: what every inline site did.  The
#: frozen-copy tests prove the legacy paths bit for bit at THESE settings.
PRE_CHANGE_EDGE_PRESSURE = {"edge_pprime_pin": True,
                            "separatrix_pressure": "legacy"}
#: The two settings' defaults.  ``separatrix_pressure`` moved from
#: ``"legacy"`` to ``"offset"`` on 2026-10-02 (owner-approved physics change;
#: see the module docstring); ``edge_pprime_pin`` is the pre-change value.
EDGE_PRESSURE_DEFAULTS = {"edge_pprime_pin": True,
                          "separatrix_pressure": "offset"}
#: What the two frames of :func:`pressure_frames` are.
FRAME_NOTE = (
    "solver frame: the solver's own pressure (zero at psi_N = 1), what the "
    "equilibrium responds to; full frame: the solver's pressure + p_sep "
    "(W_MHD + 1.5 p_sep V; every beta scaled by (int p dV + p_sep V) / "
    "int p dV, V and int p dV from the solved equilibrium)")

_MU0 = 4.0e-7 * np.pi


def validate_edge_pressure_settings(edge_pprime_pin, separatrix_pressure):
    """Refuse a malformed value of either setting, by name."""
    if not isinstance(edge_pprime_pin, (bool, np.bool_)):
        raise ValueError("generation.edge_pprime_pin must be a bool, got "
                         f"{edge_pprime_pin!r}")
    if (not isinstance(separatrix_pressure, str)
            or separatrix_pressure not in SEPARATRIX_PRESSURE_CHOICES):
        raise ValueError("generation.separatrix_pressure must be one of "
                         f"{SEPARATRIX_PRESSURE_CHOICES}, got "
                         f"{separatrix_pressure!r}")


@dataclass(frozen=True)
class EdgePressure:
    """The two settings, validated (see the module docstring).  The field
    defaults are :data:`EDGE_PRESSURE_DEFAULTS`; :meth:`pre_change` gives
    :data:`PRE_CHANGE_EDGE_PRESSURE`."""
    edge_pprime_pin: bool = True
    separatrix_pressure: str = "offset"

    def __post_init__(self):
        validate_edge_pressure_settings(self.edge_pprime_pin,
                                        self.separatrix_pressure)

    @property
    def offset(self) -> bool:
        """``separatrix_pressure == "offset"``."""
        return self.separatrix_pressure == "offset"

    @classmethod
    def pre_change(cls) -> "EdgePressure":
        """The settings before they existed (:data:`PRE_CHANGE_EDGE_PRESSURE`:
        pin on, ``"legacy"``) -- NOT the defaults."""
        return cls(**PRE_CHANGE_EDGE_PRESSURE)

    @property
    def is_default(self) -> bool:
        """The settings are :data:`EDGE_PRESSURE_DEFAULTS`."""
        return self.record() == EDGE_PRESSURE_DEFAULTS

    @property
    def is_pre_change(self) -> bool:
        """The settings are :data:`PRE_CHANGE_EDGE_PRESSURE` (pin on,
        ``"legacy"``): every array is the pre-change code's, bit for bit."""
        return self.record() == PRE_CHANGE_EDGE_PRESSURE

    def record(self) -> dict:
        """The settings as a JSON-able dict."""
        return dict(edge_pprime_pin=bool(self.edge_pprime_pin),
                    separatrix_pressure=str(self.separatrix_pressure))

    # the helper, as methods (the module functions below are the same)
    def pprime(self, psi_N, pressure, psi_range):
        return solver_pprime(psi_N, pressure, psi_range, self)

    def pax(self, pressure) -> float:
        return solver_pax(pressure, self)

    def p_offset(self, pressure) -> float:
        return applied_offset(pressure, self)


def resolve_edge_pressure(obj=None) -> EdgePressure:
    """An :class:`EdgePressure` from ``None`` (the defaults), an
    :class:`EdgePressure`, a dict with the two keys, or anything carrying
    the two attributes (a ``GenerationConfig``).  Validates loudly."""
    if obj is None:
        return EdgePressure()
    if isinstance(obj, EdgePressure):
        return obj
    d = EDGE_PRESSURE_DEFAULTS
    if isinstance(obj, dict):
        unknown = set(obj) - set(d)
        if unknown:
            raise ValueError("edge-pressure settings: unknown key(s) "
                             f"{sorted(unknown)} (known: {sorted(d)})")
        return EdgePressure(
            edge_pprime_pin=obj.get("edge_pprime_pin", d["edge_pprime_pin"]),
            separatrix_pressure=obj.get("separatrix_pressure",
                                        d["separatrix_pressure"]))
    return EdgePressure(
        edge_pprime_pin=getattr(obj, "edge_pprime_pin", d["edge_pprime_pin"]),
        separatrix_pressure=getattr(obj, "separatrix_pressure",
                                    d["separatrix_pressure"]))


# ---------------------------------------------------------------------------
#  what the solver is handed
# ---------------------------------------------------------------------------
def pressure_gradient(psi_N, pressure):
    """``d p / d psi_N`` on *psi_N* (:func:`bouquet.utils.pchip_derivative`):
    the derivative every ``P'`` of this package is built from, and the one
    the engine's first-pass pressure-driven term of a draw is shifted by."""
    from .utils import pchip_derivative
    return pchip_derivative(psi_N, pressure)


def solver_pprime(psi_N, pressure, psi_range, edge=None):
    """The ``P'`` node values handed to the solver: ``d p / d psi_N`` over
    the flux range, with the last node zeroed when ``edge_pprime_pin``."""
    edge = resolve_edge_pressure(edge)
    y = pressure_gradient(psi_N, pressure) / psi_range
    if edge.edge_pprime_pin:
        y[-1] = 0.0
    return y


def solver_pp_profile(psi_N, pressure, psi_range, edge=None, coord="psi_n"):
    """The ``pp_prof`` dict of :func:`solver_pprime` (``x`` is *psi_N*
    itself, as every site passed it), tagged for the run coordinate
    (:func:`bouquet.coords.tag_prof`: in a Phi_N run *psi_N* is the Phi_N
    grid and ``y`` is dp/dPhi_N over the flux range, which TokaMaker maps)."""
    from .coords import tag_prof
    return tag_prof({"type": "linterp",
                     "y": solver_pprime(psi_N, pressure, psi_range, edge),
                     "x": psi_N}, coord)


def separatrix_pressure_of(pressure) -> float:
    """``p_sep``: the pressure handed to the solver at its last node
    (``psi_N = 1``; total: thermal + impurity + fast, whatever the solve's
    pressure array is built from)."""
    return float(np.asarray(pressure, dtype=float)[-1])


class NegativeSeparatrixPressure(ValueError):
    """``separatrix_pressure="offset"`` met a NEGATIVE pressure at
    ``psi_N = 1``: unphysical input, refused (owner rule: failures are
    loud).  Reaches the BASELINE as well as the draws: ``prepare_baseline``
    re-raises it naming the baseline (the input's own pressure at the
    separatrix is negative); a legacy draw is rejected as ``perturb_failed``
    carrying this message."""


def applied_offset(pressure, edge=None) -> float:
    """The pressure removed from the axis target and added back at
    reporting / delivery: ``p_sep`` under ``"offset"``, exactly ``0.0``
    under ``"legacy"``."""
    edge = resolve_edge_pressure(edge)
    if not edge.offset:
        return 0.0
    p_sep = separatrix_pressure_of(pressure)
    if not np.isfinite(p_sep):
        raise ValueError("separatrix_pressure='offset': the pressure at "
                         f"psi_N = 1 is not finite ({p_sep!r})")
    if p_sep < 0.0:
        # a negative separatrix pressure is not physical (e.g. a legacy
        # draw's perturbed edge n_e or T_e below zero): offsetting by it
        # would RAISE the axis target and write a negative boundary PRES
        raise NegativeSeparatrixPressure(
            "separatrix_pressure='offset': the pressure at psi_N = 1 is "
            f"negative ({p_sep!r} Pa); a negative separatrix pressure is not "
            "physical input, so the equilibrium it belongs to is REFUSED -- "
            "the baseline in prepare_baseline() (the input's own pressure), "
            "or a draw (a legacy draw is rejected as perturb_failed).  "
            "separatrix_pressure='legacy' never reads the edge value")
    return p_sep


def solver_pax(pressure, edge=None) -> float:
    """The axis-pressure target: ``p[0]`` (``"legacy"``) or
    ``p[0] - p_sep`` (``"offset"``; refused unless positive)."""
    edge = resolve_edge_pressure(edge)
    p0 = float(pressure[0])
    if not edge.offset:
        return p0
    pax = p0 - applied_offset(pressure, edge)
    if not (np.isfinite(pax) and pax > 0.0):
        raise ValueError(
            "separatrix_pressure='offset': the axis target p_axis - p_sep = "
            f"{pax!r} Pa is not positive (p_axis = {p0!r}, p_sep = "
            f"{separatrix_pressure_of(pressure)!r})")
    return pax


def solver_pressure(pressure, edge=None):
    """The pressure array whose FIRST element is the axis target, for the
    solver-side routines that take a pressure and read ``pressure[0]`` as
    ``pax``: *pressure* itself (``"legacy"``; the same object) or
    ``pressure - p_sep`` (``"offset"``)."""
    edge = resolve_edge_pressure(edge)
    if not edge.offset:
        return pressure
    solver_pax(pressure, edge)          # the same refusal
    return np.asarray(pressure, dtype=float) - applied_offset(pressure, edge)


# ---------------------------------------------------------------------------
#  what is reported
# ---------------------------------------------------------------------------
_PVOL_KEYS = ("W_MHD", "beta_pol", "beta_tor", "beta_n")


def pressure_frames(stats, p_sep) -> dict:
    """Both frames of the pressure-integral quantities of a solved
    equilibrium.

    *stats* is the solver's ``get_stats()`` dict (``vol``, ``W_MHD = 1.5 int
    p dV``, ``beta_pol``, ``beta_tor``, ``beta_n``, ``P_ax`` -- all built
    from the solver's own pressure, zero at the boundary).  *p_sep* [Pa] is
    the constant added back.  Returns::

        {"p_sep": p_sep, "volume": V, "p_sep_volume": p_sep V,
         "factor": 1 + p_sep V / int p dV,
         "solver": {W_MHD, beta_pol, beta_tor, beta_n, P_ax},
         "full":   {W_MHD + 1.5 p_sep V, beta_* factor, P_ax + p_sep}}

    ``W_MHD`` and every ``beta`` of the solver are linear in ``int p dV``
    with the same geometry, current and field, so the full-frame value is
    the solver's times ``factor`` -- exactly ``+ 1.5 p_sep V`` for the
    energy and ``+ 2 mu0 p_sep / <B_ref^2>`` for each beta.  With ``p_sep =
    0`` both frames ARE the solver's numbers (``factor`` is exactly 1).
    Keys the solver did not report are omitted.
    """
    p_sep = float(p_sep)
    V = float(stats["vol"])
    W = float(stats["W_MHD"])
    pvol = W / 1.5
    if p_sep == 0.0:
        factor = 1.0
    else:
        if not (np.isfinite(pvol) and pvol > 0.0):
            raise ValueError("pressure_frames: the solver's int p dV = "
                             f"{pvol!r} is not positive; the full-frame "
                             "betas cannot be formed")
        factor = 1.0 + p_sep * V / pvol
    solver = {k: float(stats[k]) for k in _PVOL_KEYS if k in stats}
    full = {k: (v if p_sep == 0.0 else
                (v + 1.5 * p_sep * V if k == "W_MHD" else v * factor))
            for k, v in solver.items()}
    if "P_ax" in stats:
        solver["P_ax"] = float(stats["P_ax"])
        full["P_ax"] = float(stats["P_ax"]) + p_sep
    return dict(p_sep=p_sep, volume=V, p_sep_volume=p_sep * V,
                factor=float(factor), solver=solver, full=full,
                note=FRAME_NOTE)


def input_pressure_frames(volume, pvol, p_edge, betas=None) -> dict:
    """The same two frames for an INPUT equilibrium, from its own numbers:
    *volume* [m^3], *pvol* ``= int p dV`` of its FULL pressure, *p_edge*
    its pressure at ``psi_N = 1`` and (optionally) its ``betas`` (any
    mapping of beta names to values built from the full pressure).  The
    solver-frame quantities are those of ``p - p_edge``."""
    volume, pvol, p_edge = float(volume), float(pvol), float(p_edge)
    pv_s = pvol - p_edge * volume
    ratio = pv_s / pvol if pvol != 0.0 else float("nan")
    full = dict(W_MHD=1.5 * pvol)
    solver = dict(W_MHD=1.5 * pv_s)
    for k, v in (betas or {}).items():
        full[k] = float(v)
        solver[k] = float(v) * ratio
    return dict(p_sep=p_edge, volume=volume, p_sep_volume=p_edge * volume,
                factor=(1.0 / ratio if ratio not in (0.0,) and
                        np.isfinite(ratio) else float("nan")),
                solver=solver, full=full)


def describe(edge=None, pressure=None) -> dict:
    """The record block: the settings, ``p_sep`` of *pressure* (the input
    value, whatever the setting), the offset applied and the axis target."""
    edge = resolve_edge_pressure(edge)
    out = edge.record()
    if pressure is not None:
        out["p_sep"] = separatrix_pressure_of(pressure)
        out["p_axis"] = float(pressure[0])
        out["p_sep_applied"] = applied_offset(pressure, edge)
        out["pax_target"] = solver_pax(pressure, edge)
    return out


# ---------------------------------------------------------------------------
#  delivery (written g-files) and the archive record
# ---------------------------------------------------------------------------
#: JSON attribute carrying the edge-pressure record of an archived group
#: (``_baseline`` and every draw): the two settings, ``p_sep``, the offset
#: applied, the axis target and -- for a draw -- both pressure frames.
EDGE_PRESSURE_ATTR = "edge_pressure_json"


def lcfs_kwargs(p_sep) -> dict:
    """The extra ``save_eqdsk`` keyword that makes a written g-file carry
    the FULL pressure: ``{"lcfs_pressure": p_sep}`` (the solver adds the
    constant to ``PRES``; ``PPRIME`` is unchanged, so ``PRES`` still
    differentiates to ``PPRIME``), or ``{}`` when nothing is added back --
    the call is then exactly the one made before the setting existed."""
    p_sep = float(p_sep)
    return {} if p_sep == 0.0 else {"lcfs_pressure": p_sep}


def delivered_p_sep(record) -> float:
    """The separatrix pressure a written g-file of a delivered equilibrium
    carries: ``p_sep_applied`` of its edge-pressure record (a
    ``Baseline.edge_pressure``, an engine record's ``edge_pressure``, an
    archived ``edge_pressure_json``) -- that equilibrium's own ``p_sep``
    under ``"offset"``, exactly ``0.0`` under ``"legacy"``.  Refuses a
    missing record rather than guess a frame."""
    if not isinstance(record, dict) or record.get("p_sep_applied") is None:
        raise ValueError(
            "no edge-pressure record with 'p_sep_applied': the pressure "
            "frame of a written g-file cannot be decided (got "
            f"{record!r})")
    p = float(record["p_sep_applied"])
    if not np.isfinite(p):
        raise ValueError(f"edge-pressure record: p_sep_applied = {p!r} is "
                         "not finite")
    return p


def save_full_pressure_eqdsk(mygs, filename, p_sep, **kwargs):
    """Write *mygs*'s current equilibrium as a g-file carrying the FULL
    pressure: ``save_eqdsk(filename, **kwargs, **lcfs_kwargs(p_sep))``
    through :func:`bouquet.utils.safe_save_eqdsk` (the solver state is
    snapshotted and restored around the write).  ``PRES`` is the solver's
    pressure plus *p_sep*; ``PPRIME`` is unchanged.  With ``p_sep = 0`` the
    call is exactly a bare save.  A bare ``mygs.save_eqdsk`` writes the
    SOLVER frame (``PRES`` zero at ``psi_N = 1``) instead.  Refuses an
    explicit ``lcfs_pressure`` in *kwargs* (it would replace or double the
    offset)."""
    if "lcfs_pressure" in kwargs:
        raise ValueError("save_full_pressure_eqdsk: pass the separatrix "
                         "pressure as p_sep, not lcfs_pressure")
    from .utils import safe_save_eqdsk
    return safe_save_eqdsk(mygs, filename, **kwargs, **lcfs_kwargs(p_sep))


#: What ``p_scale`` in an edge-pressure record is.
P_SCALE_DEFINITION = (
    "the solver's uniform P' rescale of the solved equilibrium "
    "(OpenFUSIONToolkit TokaMaker p_scale = pax / P(psi_axis), P the "
    "integral of the P' profile handed to the solver from the boundary): "
    "1 when the handed P' integrates to the axis target; the edge P' pin "
    "(edge_pprime_pin), the flux range the profile was built with and, "
    "under separatrix_pressure='legacy', p_sep each move it off 1 (on a "
    "pedestal ~1.02-1.03, the size of the separatrix correction); None "
    "when the solver did not report it")


def solver_p_scale(mygs):
    """The solver's uniform ``P'`` rescale of its CURRENT equilibrium
    (:data:`P_SCALE_DEFINITION`): OpenFUSIONToolkit's ``mygs.p_scale``, or
    ``None`` for a solver object without it (a test stand-in) or a value
    that is not a finite number.  A read only."""
    try:
        v = float(getattr(mygs, "p_scale"))
    except (AttributeError, TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def archive_record(edge=None, pressure=None, stats=None, p_sep_applied=None,
                   p_scale=None):
    """The record stored under :data:`EDGE_PRESSURE_ATTR`.

    :func:`describe` of *pressure* (when given), ``p_sep_applied``
    overridden by an explicit value (an engine draw reports its own),
    ``frames`` = :func:`pressure_frames` of *stats* (``None`` without
    stats, or when they cannot be formed -- the reason is recorded), and
    ``p_scale`` (:data:`P_SCALE_DEFINITION`; ``None`` when not known)."""
    edge = resolve_edge_pressure(edge)
    out = describe(edge, pressure)
    out["p_scale"] = None if p_scale is None else float(p_scale)
    if p_sep_applied is not None:
        out["p_sep_applied"] = float(p_sep_applied)
    out["frames"] = None
    if stats is not None:
        try:
            out["frames"] = pressure_frames(stats,
                                            out.get("p_sep_applied", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            out["frames_error"] = f"{type(exc).__name__}: {exc}"
            if float(out.get("p_sep_applied", 0.0) or 0.0) != 0.0:
                # the reported beta / W_MHD then stay in the SOLVER frame
                # although p_sep was removed from the axis target: loud
                import warnings
                warnings.warn(
                    "edge pressure: the full-pressure frame could not be "
                    f"formed ({out['frames_error']}); the reported beta / "
                    "W_MHD of this solve are in the SOLVER frame (p_sep = "
                    f"{out['p_sep_applied']:g} Pa NOT added back)",
                    RuntimeWarning, stacklevel=2)
    return out


def store_record(header, record, scan_key=None, count=None) -> None:
    """Write *record* as the JSON attribute :data:`EDGE_PRESSURE_ATTR` on
    the archive's ``_baseline`` group (``count=None``) or on draw *count*.
    No-op when the group does not exist or *record* is ``None``."""
    if record is None:
        return
    import json
    import h5py
    from .jbs_loop import jsonable
    from .utils import _baseline_group_path, _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "a") as hf:
        gp = (_baseline_group_path(scan_key) if count is None
              else _group_path(scan_key, count))
        if gp in hf:
            hf[gp].attrs[EDGE_PRESSURE_ATTR] = json.dumps(
                jsonable(record), allow_nan=True)


def load_record(header, count=None, scan_key=None):
    """The record :func:`store_record` wrote, or ``None``."""
    import json
    import h5py
    from .utils import _baseline_group_path, _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "r") as hf:
        gp = (_baseline_group_path(scan_key) if count is None
              else _group_path(scan_key, count))
        if gp not in hf or EDGE_PRESSURE_ATTR not in hf[gp].attrs:
            return None
        return json.loads(hf[gp].attrs[EDGE_PRESSURE_ATTR])
