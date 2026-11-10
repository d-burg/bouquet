"""Separatrix pressure and the two pressure frames: ONE place in ``kinrecon``.

TokaMaker builds the plasma pressure by integrating P' inward from the boundary starting
at ZERO, then rescales P' uniformly so that the axis value equals its axis-pressure target
(``pax``).  A kinetic pressure is generally NOT zero at psi_N = 1 (``p_sep``), so:

* ``separatrix_pressure="offset"`` (default): the target is ``p_axis - p_sep``, which the
  input's own P' integrates to, so P' is the input's (solver rescale ~1, up to the
  discretization of the integral).  ``p_sep`` is added back wherever pressure, beta or
  stored energy is reported or written.
* ``separatrix_pressure="legacy"``: the target is the full ``p_axis``; the solver then
  inflates P' everywhere by ``p_axis / (p_axis - p_sep)`` and the equilibrium is that of
  ``p_axis (p - p_sep) / (p_axis - p_sep)``.  Kept only to reproduce that behavior; its
  inflation factor is always recorded.

The two frames (names and semantics aligned with bouquet's own edge-pressure convention,
re-implemented here because ``kinrecon`` imports nothing from the rest of bouquet):

* **solver frame** -- the solver's own pressure, zero at psi_N = 1 (``p - p_sep``); what
  the equilibrium responds to and what TokaMaker's ``get_stats`` reports;
* **full frame** -- the solver's pressure + ``p_sep``.  With V the plasma volume and
  int p dV the solver's own (``vol``, ``W_MHD / 1.5``)::

      W_MHD(full)  = W_MHD(solver) + 1.5 p_sep V
      beta_X(full) = beta_X(solver) * (int p dV + p_sep V) / int p dV      X = pol, tor, n
      P_ax(full)   = P_ax(solver) + p_sep

  (every beta is 2 mu0 <p> / B_ref^2 with one reference field, so a constant adds
  2 mu0 p_sep / B_ref^2).

Where the model stops: a pressure that is ``p_sep`` just inside the boundary and zero just
outside is not physical; the separatrix pressure continues into the scrape-off layer, which
a vacuum-outside free-boundary equilibrium cannot represent.  Only P' enters the
Grad-Shafranov equation, so the constant ``p_sep`` exerts no force: it is bookkeeping for
readers of the pressure.

What ``kinrecon`` scores does not depend on the frame: magnetics and MSE follow from the
solved fields; Thomson compares n_e and T_e (not pressure) mapped through psi_N; no
pressure or beta constraint exists in :mod:`.score`.  Any future pressure-like constraint
must compare in the full frame (:func:`pressure_frames`, :func:`full_pressure`).

Edge taper: the taper (:mod:`.edge`) acts on P' only and holds p(psi_N = 1) fixed, so
``p_sep`` is unchanged by it.  With the ``edge_pinned`` / ``last_node`` variants the
full-frame interior pressure drops by one constant and the solver target becomes
``(p_axis - p_sep) * pax_ratio``; with ``pressure_conserving`` it does not
(:func:`solver_inputs_with_taper`).

Delivery: :func:`save_full_pressure_eqdsk` writes a g-file whose ``PRES`` is the full
pressure (TokaMaker's ``lcfs_pressure``; ``PPRIME`` unchanged, so ``PRES`` still
differentiates to it); :func:`equilibrium_profiles_1d` gives the IMAS
``equilibrium.time_slice[:].profiles_1d`` pressure block in the full frame with ``p_sep``
recorded, and :func:`write_profiles_1d_json` / :func:`read_profiles_1d_json` round-trip it.
"""

import json

import numpy as np

#: ``separatrix_pressure`` values (bouquet's names).
SEPARATRIX_PRESSURE_CHOICES = ("legacy", "offset")
SEPARATRIX_PRESSURE_DEFAULT = "offset"
FRAME_NOTE = (
    "solver frame: the solver's own pressure (zero at psi_N = 1), what the equilibrium "
    "responds to; full frame: the solver's pressure + p_sep (W_MHD + 1.5 p_sep V; every beta "
    "scaled by (int p dV + p_sep V) / int p dV, V and int p dV from the solved equilibrium)")
_PVOL_KEYS = ("W_MHD", "beta_pol", "beta_tor", "beta_n")


class NegativeSeparatrixPressure(ValueError):
    """A negative pressure at psi_N = 1 under ``"offset"``: unphysical input, refused."""


def _mode(separatrix_pressure):
    if separatrix_pressure not in SEPARATRIX_PRESSURE_CHOICES:
        raise ValueError(f"separatrix_pressure must be one of {SEPARATRIX_PRESSURE_CHOICES}, "
                         f"got {separatrix_pressure!r}")
    return separatrix_pressure


def separatrix_pressure_of(pressure):
    """``p_sep``: the TOTAL pressure array's value at its last node (psi_N = 1)."""
    return float(np.asarray(pressure, dtype=float)[-1])


def applied_offset(pressure, separatrix_pressure=SEPARATRIX_PRESSURE_DEFAULT):
    """The pressure removed from the axis target and added back when reporting/writing:
    ``p_sep`` under ``"offset"`` (refused if negative or not finite), exactly 0 under ``"legacy"``."""
    if _mode(separatrix_pressure) == "legacy":
        return 0.0
    p_sep = separatrix_pressure_of(pressure)
    if not np.isfinite(p_sep):
        raise ValueError(f"the pressure at psi_N = 1 is not finite ({p_sep!r})")
    if p_sep < 0.0:
        raise NegativeSeparatrixPressure(f"the pressure at psi_N = 1 is negative ({p_sep!r}); a negative "
                                         "separatrix pressure is not physical input")
    return p_sep


def solver_pax(pressure, separatrix_pressure=SEPARATRIX_PRESSURE_DEFAULT):
    """The axis-pressure target: ``p[0] - p_sep`` (``"offset"``; refused unless > 0) or ``p[0]``."""
    p0 = float(np.asarray(pressure, dtype=float)[0])
    pax = p0 - applied_offset(pressure, separatrix_pressure)
    if not (np.isfinite(pax) and pax > 0.0):
        raise ValueError(f"axis target p_axis - p_sep = {pax!r} is not positive")
    return pax


def pressure_gradient(psi_n, pressure):
    """dp/dpsi_N by PCHIP on the native grid (shape-preserving), as bouquet builds P'."""
    from scipy.interpolate import PchipInterpolator
    x = np.asarray(psi_n, dtype=float)
    return PchipInterpolator(x, np.asarray(pressure, dtype=float)).derivative()(x)


def solver_inputs(psi_n, pressure, *, pprime=None, psi_scale=None,
                  separatrix_pressure=SEPARATRIX_PRESSURE_DEFAULT):
    """Solver-frame inputs from a FULL-frame (kinetic) pressure profile.

    ``pprime`` (dp/dpsi on the knots) is used as given and never modified; without it it
    is built as dp/dpsi_N / ``psi_scale`` (PCHIP).  Returns a dict with ``pprime``, ``pax``
    (``p_axis - p_sep`` under ``"offset"``), ``p_sep``, ``p_sep_applied``, ``p_axis``,
    ``frame`` notes and ``legacy_pprime_inflation`` = p_axis / (p_axis - p_sep): the factor
    by which a full-``p_axis`` target would have inflated P' (1 when p_sep = 0).  When
    ``psi_scale`` is known, ``implied_solver_rescale`` = pax / (-psi_scale int P' dpsi_N),
    the uniform rescale the solver will apply (1 up to the discretization of the integral).
    """
    x = np.asarray(psi_n, dtype=float)
    p = np.asarray(pressure, dtype=float)
    if p.shape != x.shape:
        raise ValueError("pressure must be on the psi_n knots")
    if pprime is None:
        if psi_scale is None:
            raise ValueError("give pprime, or psi_scale to build it from the pressure")
        pp = pressure_gradient(x, p) / float(psi_scale)
        pp_source = "dp/dpsi_N (PCHIP) / psi_scale"
    else:
        pp = np.asarray(pprime, dtype=float)
        pp_source = "given (unchanged)"
    p_sep = separatrix_pressure_of(p)
    pax = solver_pax(p, separatrix_pressure)
    p0 = float(p[0])
    out = dict(pprime=pp, pprime_source=pp_source, pax=pax, p_axis=p0, p_sep=p_sep,
               p_sep_over_p_axis=(p_sep / p0 if p0 else None),
               p_sep_applied=applied_offset(p, separatrix_pressure), separatrix_pressure=separatrix_pressure,
               legacy_pprime_inflation=(p0 / (p0 - p_sep) if p0 != p_sep else None),
               frames=dict(pax="solver", p_axis="full", p_sep="full", pprime="frame-independent"),
               implied_solver_rescale=None)
    if psi_scale is not None:
        integ = float(np.sum(0.5 * (pp[1:] + pp[:-1]) * np.diff(x))) * float(psi_scale)
        if integ != 0.0:
            out["implied_solver_rescale"] = pax / (-integ)
    return out


def solver_inputs_with_taper(psi_n, pressure, pprime, psi_scale, spec,
                             separatrix_pressure=SEPARATRIX_PRESSURE_DEFAULT, **taper_kw):
    """:func:`solver_inputs` combined with an edge taper (:class:`.edge.EdgeTaperSpec`).

    The taper acts on P' and holds p(1) = p_sep fixed.  The solver target is
    ``pax_offset * pax_ratio`` so that the interior P' handed to the solver is the input's:
    for ``edge_pinned`` / ``last_node`` that lowers the full-frame interior pressure by the
    record's ``dp_inside``; for ``pressure_conserving`` ``pax_ratio`` is 1 to round-off.
    Returns (x, pprime, pax, record); the record carries ``p_sep``, the taper record (its
    pressures in the full frame) and the frame of every pressure-like number."""
    from .edge import apply_edge_taper
    base = solver_inputs(psi_n, pressure, pprime=pprime, separatrix_pressure=separatrix_pressure)
    t = apply_edge_taper(psi_n, pprime, pressure, psi_scale=psi_scale, **spec.kwargs(), **taper_kw)
    pax = base["pax"] * t.record["pax_ratio"]
    rec = dict(separatrix_pressure=separatrix_pressure, p_sep=base["p_sep"], p_sep_applied=base["p_sep_applied"],
               p_axis_full=base["p_axis"], pax_untapered=base["pax"], pax=pax, taper_variant=spec.variant,
               taper=t.record, p_sep_after_taper=(float(t.pressure[-1]) if t.pressure is not None else None),
               frames=dict(pax="solver", pax_untapered="solver", p_axis_full="full", p_sep="full",
                           taper_pressures="full (the input pressure's frame; p(1) = p_sep held fixed)"))
    return t.psi_n, t.pprime, pax, rec


def full_pressure(solver_pressure, p_sep):
    """Full-frame pressure from the solver's (zero at psi_N = 1)."""
    return np.asarray(solver_pressure, dtype=float) + float(p_sep)


def pressure_frames(stats, p_sep):
    """Both frames of the pressure-integral quantities of a solved equilibrium.

    ``stats`` is TokaMaker's ``get_stats()`` dict (``vol``, ``W_MHD``, ``beta_pol``,
    ``beta_tor``, ``beta_n``, ``P_ax``; built from the solver's pressure).  Returns
    ``{"p_sep", "volume", "p_sep_volume", "factor", "solver": {...}, "full": {...}, "note"}``
    (module docstring for the formulas).  With p_sep = 0 both frames are the solver's
    numbers and ``factor`` is exactly 1.  Keys the solver did not report are omitted."""
    p_sep = float(p_sep)
    V = float(stats["vol"])
    W = float(stats["W_MHD"])
    pvol = W / 1.5
    if p_sep == 0.0:
        factor = 1.0
    else:
        if not (np.isfinite(pvol) and pvol > 0.0):
            raise ValueError(f"the solver's int p dV = {pvol!r} is not positive; full-frame betas cannot be formed")
        factor = 1.0 + p_sep * V / pvol
    solver = {k: float(stats[k]) for k in _PVOL_KEYS if k in stats}
    full = {k: (v if p_sep == 0.0 else (v + 1.5 * p_sep * V if k == "W_MHD" else v * factor))
            for k, v in solver.items()}
    if "P_ax" in stats:
        solver["P_ax"] = float(stats["P_ax"])
        full["P_ax"] = float(stats["P_ax"]) + p_sep
    return dict(p_sep=p_sep, volume=V, p_sep_volume=p_sep * V, factor=float(factor), solver=solver, full=full,
                note=FRAME_NOTE)


# ============================================================== delivery
def lcfs_kwargs(p_sep):
    """``{"lcfs_pressure": p_sep}`` for TokaMaker's ``save_eqdsk``/``save_ifile``, or ``{}`` when 0."""
    p_sep = float(p_sep)
    return {} if p_sep == 0.0 else {"lcfs_pressure": p_sep}


def save_full_pressure_eqdsk(eq, filename, p_sep, **kwargs):
    """Write ``eq``'s equilibrium as a g-file carrying the FULL pressure: ``PRES`` = the
    solver's pressure + ``p_sep``; ``PPRIME`` unchanged.  Refuses an explicit
    ``lcfs_pressure`` (it would replace or double the offset).  Returns the record to keep
    next to the file."""
    if "lcfs_pressure" in kwargs:
        raise ValueError("pass the separatrix pressure as p_sep, not lcfs_pressure")
    eq.save_eqdsk(filename, **kwargs, **lcfs_kwargs(p_sep))
    return dict(file=str(filename), p_sep=float(p_sep), pressure_frame="full",
                note="PRES = solver pressure + p_sep (TokaMaker lcfs_pressure); PPRIME unchanged")


def equilibrium_profiles_1d(psi_n, solver_pressure, dpressure_dpsi, p_sep, psi_axis, psi_boundary,
                            dpsi_to_ids=None):
    """IMAS ``equilibrium.time_slice[:].profiles_1d`` pressure block in the FULL frame.

    ``solver_pressure`` [Pa] and ``dpressure_dpsi`` (dp/dpsi in the caller's flux units,
    psi_axis/psi_boundary in the same units) on ``psi_n``.  ``dpsi_to_ids`` converts the
    caller's flux to the IDS flux, psi_IDS = dpsi_to_ids * psi_caller; it is REQUIRED
    to be stated (for TokaMaker's per-radian flux into a COCOS-11 IDS bouquet uses -2 pi;
    that factor is taken from bouquet's convention, not re-derived here).  Returns
    ``psi`` (IDS flux), ``pressure`` (full frame), ``dpressure_dpsi`` (per IDS flux), and
    the record (``p_sep``, frame, conversion)."""
    if dpsi_to_ids is None:
        raise ValueError("state dpsi_to_ids (psi_IDS = dpsi_to_ids * psi_caller); no COCOS is assumed")
    x = np.asarray(psi_n, dtype=float)
    c = float(dpsi_to_ids)
    psi = c * (psi_axis + x * (psi_boundary - psi_axis))
    return dict(psi=psi, pressure=full_pressure(solver_pressure, p_sep),
                dpressure_dpsi=np.asarray(dpressure_dpsi, dtype=float) / c,
                meta=dict(p_sep=float(p_sep), pressure_frame="full", dpsi_to_ids=c,
                          note="pressure = solver pressure + p_sep; dpressure_dpsi unchanged except for the flux "
                               "unit; pressure(psi_N = 1) = p_sep"))


def check_consistency(psi, pressure, dpressure_dpsi):
    """max |p - (p(1) - int_psi^psi_b p' dpsi)| / max|p|, trapezoid on the given knots: how
    well a written pressure and its derivative agree (0 for an exact piecewise-linear p')."""
    psi = np.asarray(psi, dtype=float)
    p = np.asarray(pressure, dtype=float)
    d = np.asarray(dpressure_dpsi, dtype=float)
    seg = 0.5 * (d[1:] + d[:-1]) * np.diff(psi)
    cum = np.zeros_like(p)
    cum[:-1] = np.cumsum(seg[::-1])[::-1]
    return float(np.max(np.abs(p - (p[-1] - cum))) / (np.max(np.abs(p)) or 1.0))


def write_profiles_1d_json(path, block):
    """Write the block of :func:`equilibrium_profiles_1d` as IDS-named JSON
    (``equilibrium.time_slice[0].profiles_1d.{psi,pressure,dpressure_dpsi}`` + ``meta``)."""
    d = {"equilibrium": {"time_slice": [{"profiles_1d": {k: [float(v) for v in block[k]]
                                                          for k in ("psi", "pressure", "dpressure_dpsi")}}]},
         "kinrecon_pressure_meta": block["meta"]}
    with open(path, "w") as f:
        json.dump(d, f)
        f.write("\n")


def read_profiles_1d_json(path):
    """Read :func:`write_profiles_1d_json` output back (arrays + ``meta``)."""
    with open(path) as f:
        d = json.load(f)
    pr = d["equilibrium"]["time_slice"][0]["profiles_1d"]
    out = {k: np.asarray(pr[k], dtype=float) for k in ("psi", "pressure", "dpressure_dpsi")}
    out["meta"] = d["kinrecon_pressure_meta"]
    return out
