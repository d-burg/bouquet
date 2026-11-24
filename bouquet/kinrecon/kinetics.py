r"""Kinetic inputs of the current model, assembled with provenance for every profile.

What is assembled
-----------------
From measured or fitted ``n_e``, ``T_e``, ``T_i``, a CHOSEN effective charge ``Z_eff`` and the
fast-ion (beam-ion) density ``n_fast`` of a source provider, with one impurity species of charge
``Z`` (e.g. carbon, Z = 6) and singly charged main and fast ions:

.. code-block:: text

    quasineutrality   n_e        = n_i + Z n_imp + n_fast                                 (K1)
    effective charge  Z_eff n_e  = n_i + Z^2 n_imp + n_fast                               (K2)
    =>                n_imp      = n_e (Z_eff - 1) / (Z (Z - 1))                          (K3)
                      n_i        = n_e - n_fast - Z n_imp                                 (K4)
    thermal pressure  p_th       = e (n_e T_e + (n_i + n_imp) T_i)                        (K5)

(K3) does not involve ``n_fast`` (a singly charged fast ion adds the same amount to both sides of
(K1) and (K2)); (K4) removes the fast ions from the thermal ion density, so they are not counted
once as thermal ions at ``T_i`` and again in the fast-ion pressure.  The impurity is taken at
``T_i`` (stated).  A Z_eff source given as an impurity DENSITY (e.g. charge-exchange carbon) is
turned into a Z_eff candidate with (K2)-(K1): ``Z_eff = 1 + Z (Z - 1) n_imp / n_e``.

Z_eff source: there is no default.  Every candidate the caller has (e.g. a visible-bremsstrahlung
Z_eff and a CER impurity density) is recorded with the pairwise disagreement, and the caller
either names the one used (``zeff_source=<name>``, the candidate is taken as given) or asks for the
checked-source rule (``zeff_source="checked"``); a run that omits it is refused.  Nothing is
clipped: ``Z_eff < 1``, ``Z_eff > Z`` or a non-positive ``n_i`` inside the plasma is refused with
the location.

Checked source (``zeff_source="checked"``): the rule of :mod:`.zeff_sources` (a verbatim copy of
the bouquet reader's rule module) tests each candidate per slice -- an impurity density: I-1
coverage of its measurement points, I-2 not a placeholder, I-3 positive main-ion density with the
fast ions in quasineutrality, I-4 consistency with supplied raw analyses; a line-integrated Z_eff:
L-1 own uncertainty, L-2 median relative uncertainty <= 0.25, L-3 not a copy of the
impurity-derived value, L-4 not radially flat -- and decides: a qualified impurity source, else a
qualified line-integrated value, else an unverified impurity source (used, labeled and warned),
else refuse.  A test that cannot be evaluated (no points, no uncertainty, no raw analyses) leaves
its candidate "unverified", never "qualified"; a missing fast-ion density does not (I-3 is then
recorded as evaluated without fast ions).  A candidate named explicitly on the checked route
(``zeff_rule={"override": <name>}``) is used only if it does not fail its tests, unless the caller
also accepts that explicitly (``"accept_failed": True``, recorded and warned).  The full rule record
(every candidate, test value, verdict, threshold and the decision) is stored in
``KineticInputs.record["zeff_rule"]`` and summarized in the provenance.

Ion-temperature coverage (``ti_coverage``, required): the caller passes the ion-temperature measurement points of
the slice (the analysis the ``T_i`` profile was fitted to) and states the core criterion (``core_psi_n_max``: a psi_N,
or ``None`` for none; there is no default, the value depends on the device's diagnostic coverage).  A slice with no
valid point inside the plasma boundary (or, criterion applied, none at psi_N <= ``core_psi_n_max``) has an ion
temperature that is not measured there: it is refused (:class:`TiCoverageRefused`, with the check's record) unless the
caller names an assumption for it (``"assumption": "equal_to_electron"``: ``T_i = T_e`` over the whole table, with a
relative uncertainty the caller states; or ``"assumption": "ratio_to_electron"``: ``T_i = r T_e`` with the ratio
``r`` the caller supplies on the table (``assumption_ratio``; e.g. the measured ``T_i/T_e`` of the neighboring covered
slices interpolated in time), its relative uncertainty (a number or an array on the table) and a description of where
the ratio comes from (``assumption_detail``, recorded as given)).  An applied assumption is recorded (``record["ti_coverage"]``: assumed, the
rule, the radial range, the stated uncertainty), named in the provenance and warned about; it is never applied
silently, and a named assumption on a covered slice is recorded and not applied.

Coordinate (the working coordinate is rho_tor,norm):

* profiles that carry rho_tor,norm are placed by it (``coordinate="rho_tor_norm"``);
* profiles that carry only psi_N of ANOTHER equilibrium are mapped through that equilibrium's own
  ``psi_N -> rho_tor,norm`` table when the caller supplies it (``psi_n_to_rho=(psi_n, rho)``,
  recorded), and kept on psi_N otherwise (``coordinate="psi_n"``: then they are evaluated at the
  CURRENT equilibrium's psi_N in every pass, which is recorded as "mapped through ours").

The result evaluates on any equilibrium through :meth:`KineticInputs.profiles_at` with that
equilibrium's ``(psi_N, rho)`` pair; :mod:`.current_model` calls it on every pass.
"""

import json
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import zeff_sources as _zr

EC = 1.602176634e-19

__all__ = ["KineticInputs", "assemble_kinetic_inputs", "zeff_from_impurity_density", "COORDINATES",
           "CHECKED", "ZEFF_RULE_OPTIONS", "FAST_IONS_NONE", "ti_coverage_check", "TiCoverageRefused",
           "TiCoverageWarning", "TI_COVERAGE_KEYS", "TI_ASSUMPTIONS", "TI_EQUAL_TO_ELECTRON", "TI_RATIO_TO_ELECTRON", "TI_COVERED",
           "TI_NOT_COVERED"]

#: Coordinates a kinetic table can carry.
COORDINATES = ("rho_tor_norm", "psi_n")
#: The ``zeff_source`` value that applies the checked-source rule (module docstring).
CHECKED = "checked"
#: ``n_fast`` value that declares the absence of fast ions (:func:`assemble_kinetic_inputs`; the zero density is
#: then a stated input, recorded, never a silent default).
FAST_IONS_NONE = "none"
#: Keys of the ``zeff_rule`` options (all optional): ``psi_n`` (psi_N of every ``x``, needed for a
#: ``rho_tor_norm`` table: the rule's regions are in psi_N), ``slice_time`` [s], ``slice_window``
#: ``(t0, t1)`` [s], ``override`` (a candidate name), ``accept_failed`` (bool), ``fast_ions``
#: (``"none"``: the caller declares there are no fast ions, as ``n_fast="none"`` does; otherwise the ``n_fast``
#: given is used).
ZEFF_RULE_OPTIONS = ("psi_n", "slice_time", "slice_window", "override", "accept_failed", "fast_ions")
#: Keys a candidate may carry when given as a dict (``value`` and ``kind`` required).
_CANDIDATE_KEYS = ("kind", "value", "sigma", "fit_points", "raw_points", "provenance", "core_median_series")
#: Keys of the ``ti_coverage`` argument of :func:`assemble_kinetic_inputs`: ``points`` and ``core_psi_n_max`` are
#: required (``core_psi_n_max`` may be ``None``, stated), ``slice_window`` (required for points that carry times),
#: ``assumption`` and ``assumption_sigma_rel`` optional; ``assumption_ratio`` and ``assumption_detail`` are required
#: with, and allowed only with, the assumption ``"ratio_to_electron"``.
TI_COVERAGE_KEYS = ("points", "core_psi_n_max", "slice_window", "assumption", "assumption_sigma_rel",
                    "assumption_ratio", "assumption_detail")
#: The assumptions a caller may name for an ion temperature that is not measured inside the boundary: ``T_i = T_e``,
#: and ``T_i = r T_e`` with a ratio ``r`` the caller supplies on the table and describes.
TI_EQUAL_TO_ELECTRON = "equal_to_electron"
TI_RATIO_TO_ELECTRON = "ratio_to_electron"
TI_ASSUMPTIONS = (TI_EQUAL_TO_ELECTRON, TI_RATIO_TO_ELECTRON)
#: Verdicts of :func:`ti_coverage_check`.
TI_COVERED, TI_NOT_COVERED = "covered", "not_covered"


class TiCoverageRefused(ValueError):
    """The ion temperature is not measured inside the plasma boundary on this slice and no assumption was named
    (:func:`assemble_kinetic_inputs`).  ``record`` holds the coverage check's record."""

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record


class TiCoverageWarning(UserWarning):
    """An assumed ion temperature is in use (the slice's ion temperature is not measured inside the boundary)."""


def ti_coverage_check(points, *, core_psi_n_max, slice_window=None):
    """Whether the ion temperature is measured inside the plasma boundary on one slice; returns the check's record.

    Parameters
    ----------
    points : :class:`.zeff_sources.MeasurementPoints` or its dict
        The ion-temperature measurement points of the analysis the ``T_i`` profile was fitted to: ``value`` [eV],
        ``psi_N`` (of the producing equilibrium; required), optional ``sigma`` and ``time`` [s].  Padding
        (non-finite or fill-magnitude entries) is dropped.  A valid point has a finite value > 0, a finite psi_N and,
        when an uncertainty is given for it, a finite uncertainty > 0; it is inside the boundary at psi_N <= 1.
    core_psi_n_max : float or None
        Required, no default: the innermost valid point inside the boundary must lie at psi_N <= this value
        (0 < value <= 1); ``None`` states that no core criterion is applied (recorded).
    slice_window : (t0, t1) [s], optional
        Required when the points carry times: only the points inside it (inclusive) count.

    Covered: at least one valid point inside the boundary and, the criterion applied, the innermost of them at
    psi_N <= ``core_psi_n_max``.
    """
    pts = _zr.MeasurementPoints.coerce(points, label="ion-temperature points")
    if pts is None or pts.psi_N is None:
        raise ValueError("ti_coverage points need 'value' and 'psi_N' (psi_N of the producing equilibrium)")
    if core_psi_n_max is not None:
        core_psi_n_max = float(core_psi_n_max)
        if not 0.0 < core_psi_n_max <= 1.0:
            raise ValueError("core_psi_n_max must lie in (0, 1], or be None (no core criterion, stated)")
    big = _zr.FILL_ABS_MIN
    v = np.asarray(pts.value, dtype=float).ravel()
    p = np.asarray(pts.psi_N, dtype=float).ravel()
    if p.shape != v.shape:
        raise ValueError("ti_coverage points: 'value' and 'psi_N' must have the same length")
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(v) & (np.abs(v) < big) & np.isfinite(p) & (np.abs(p) < big)
        n = int(ok.sum())
        valid = ok & (v > 0)
        if pts.sigma is not None:
            s = np.asarray(pts.sigma, dtype=float).ravel()
            if s.shape != v.shape:
                raise ValueError("ti_coverage points: 'sigma' must have the length of 'value'")
            valid &= np.isfinite(s) & (s > 0) & (np.abs(s) < big)
    window = "points carry no times (taken as selected for this slice)"
    if pts.time is not None:
        if slice_window is None:
            raise ValueError("ti_coverage points carry times: pass ti_coverage['slice_window'] = (t0, t1) [s]")
        a, b = float(slice_window[0]), float(slice_window[1])
        t = np.asarray(pts.time, dtype=float).ravel()
        valid &= (t >= a) & (t <= b)
        window = f"the slice's window [{a:.4f}, {b:.4f}] s"
    inside = valid & (p <= 1.0)
    ni = int(inside.sum())
    inner = float(p[inside].min()) if ni else None
    outer = float(p[inside].max()) if ni else None
    why = []
    if ni == 0:
        why.append(f"no valid point inside the boundary ({n} point(s) given, {int(valid.sum())} valid)")
    elif core_psi_n_max is not None and inner > core_psi_n_max:
        why.append(f"innermost valid point at psi_N {inner:.3f} (> core_psi_n_max {core_psi_n_max:g})")
    return dict(verdict=TI_NOT_COVERED if why else TI_COVERED, points=str(pts.label), n_points=n,
                n_valid=int(valid.sum()), n_valid_inside=ni, psi_n_inner=inner, psi_n_outer=outer,
                core_psi_n_max=core_psi_n_max,
                core_criterion=("applied" if core_psi_n_max is not None else
                                "not applied (core_psi_n_max=None, stated by the caller)"),
                window=window, reason="; ".join(why) if why else "covered")


def _ti_coverage(x, coordinate, out, opts):
    """The coverage check on the slice and the refusal or the named assumption (module docstring).  Replaces
    ``out["ti"]`` when an assumption is applied; returns the record."""
    if not isinstance(opts, dict):
        raise ValueError("ti_coverage must be a dict (TI_COVERAGE_KEYS)")
    unknown = set(opts) - set(TI_COVERAGE_KEYS)
    if unknown or "points" not in opts or "core_psi_n_max" not in opts:
        raise ValueError(f"ti_coverage needs 'points' and 'core_psi_n_max' (a psi_N, or None stated) and may carry "
                         f"only {TI_COVERAGE_KEYS} (got {sorted(opts)})")
    rule, sig = opts.get("assumption"), opts.get("assumption_sigma_rel")
    if rule is not None and rule not in TI_ASSUMPTIONS:
        raise ValueError(f"ti_coverage assumption must be one of {TI_ASSUMPTIONS} (or omitted: refuse)")
    ratio, detail = opts.get("assumption_ratio"), opts.get("assumption_detail")
    by_ratio = rule == TI_RATIO_TO_ELECTRON
    if sig is not None:
        sig = np.asarray(sig, dtype=float)
        if sig.ndim and not (by_ratio and sig.shape == x.shape):
            raise ValueError("ti_coverage: 'assumption_sigma_rel' must be one number (with the assumption "
                             f"{TI_RATIO_TO_ELECTRON!r} it may be an array on the table)")
    if rule is not None and not (sig is not None and np.all(np.isfinite(sig)) and np.all(sig > 0)):
        raise ValueError("ti_coverage: an assumption needs 'assumption_sigma_rel', the relative 1-sigma uncertainty "
                         "of the assumed T_i (> 0), stated by the caller")
    if rule is None and sig is not None:
        raise ValueError("ti_coverage: 'assumption_sigma_rel' is given without an assumption")
    if not by_ratio and (ratio is not None or detail is not None):
        raise ValueError(f"ti_coverage: 'assumption_ratio' and 'assumption_detail' belong to the assumption "
                         f"{TI_RATIO_TO_ELECTRON!r} only")
    if by_ratio:
        if ratio is None or not isinstance(detail, dict) or not detail:
            raise ValueError(f"ti_coverage: the assumption {TI_RATIO_TO_ELECTRON!r} needs 'assumption_ratio' (T_i/T_e "
                             "on the table) and 'assumption_detail' (a dict saying where the ratio comes from, "
                             "recorded as given)")
        ratio = np.asarray(ratio, dtype=float)
        if ratio.shape != x.shape or not (np.all(np.isfinite(ratio)) and np.all(ratio > 0)):
            raise ValueError("ti_coverage: 'assumption_ratio' must be finite and > 0 at every point of the table")
        try:
            detail = json.loads(json.dumps(detail))
        except (TypeError, ValueError) as err:
            raise ValueError("ti_coverage: 'assumption_detail' must hold plain numbers, strings, lists and "
                             "dicts") from err
    rec = ti_coverage_check(opts["points"], core_psi_n_max=opts["core_psi_n_max"],
                            slice_window=opts.get("slice_window"))
    rec["assumption_named"] = rule
    if rec["verdict"] == TI_COVERED:
        rec["ti"] = "measured"
        if rule is not None:
            rec["note"] = f"assumption {rule!r} named but not applied: the ion temperature is measured"
        return rec
    if rule is None:
        raise TiCoverageRefused(
            "ion temperature not measured inside the plasma boundary: " + rec["reason"] + "; refused (pass "
            f"ti_coverage={{..., 'assumption': {TI_EQUAL_TO_ELECTRON!r}, 'assumption_sigma_rel': <stated>}} to fit "
            f"it with T_i = T_e, or the assumption {TI_RATIO_TO_ELECTRON!r} with a supplied T_i/T_e, recorded)", rec)
    rec["ti"] = "assumed"
    note = "stated by the caller and recorded; the kinetic inputs carry no uncertainties"
    if by_ratio:
        out["ti"] = ratio * out["te"]
        statement = "T_i = r T_e, r supplied by the caller"
        sig_rec = float(sig) if not sig.ndim else [float(v) for v in sig]
        sig_txt = f"{float(sig):g}" if not sig.ndim else f"{float(np.median(sig)):g} (median over the table)"
        rec["assumption"] = dict(rule=rule, statement=statement, sigma_rel=sig_rec, sigma_note=note,
                                 ratio=[float(v) for v in ratio], detail=detail, coordinate=coordinate,
                                 range=[float(x[0]), float(x[-1])], extent="the whole table")
    else:
        out["ti"] = out["te"].copy()
        statement, sig_txt = "T_i = T_e", f"{float(sig):g}"
        rec["assumption"] = dict(rule=rule, statement=statement, sigma_rel=float(sig), sigma_note=note,
                                 coordinate=coordinate, range=[float(x[0]), float(x[-1])], extent="the whole table")
    rec["assumption"]["sigma_text"] = sig_txt
    warnings.warn(f"ion temperature ASSUMED ({statement}, rule {rule!r}, stated relative 1-sigma {sig_txt}) over "
                  f"{coordinate} [{x[0]:g}, {x[-1]:g}]: not measured inside the boundary ({rec['reason']})",
                  TiCoverageWarning, stacklevel=3)
    return rec


def zeff_from_impurity_density(ne, nimp, z_imp):
    """``Z_eff = 1 + Z (Z - 1) n_imp / n_e`` (module docstring, (K1)-(K2) with singly charged main and
    fast ions)."""
    ne, nimp = np.asarray(ne, float), np.asarray(nimp, float)
    return 1.0 + float(z_imp) * (float(z_imp) - 1.0) * nimp / ne


@dataclass
class KineticInputs:
    """Kinetic profiles on one coordinate with provenance (build with :func:`assemble_kinetic_inputs`).

    Arrays on ``x`` (``coordinate`` = ``"rho_tor_norm"`` or ``"psi_n"``): ``ne``, ``ni``, ``nimp``,
    ``n_fast`` [m^-3], ``te``, ``ti`` [eV], ``zeff``.  ``provenance`` names the source of every
    profile; ``record`` holds the Z_eff candidates, the choice, the coordinate mapping and checks.
    """
    coordinate: str
    x: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ti: np.ndarray
    zeff: np.ndarray
    ni: np.ndarray
    nimp: np.ndarray
    n_fast: np.ndarray
    z_imp: float
    provenance: Dict[str, str] = field(default_factory=dict)
    record: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    #: The producing equilibrium's outboard-midplane major radius against its psi_N (``psi_n``, ``r_m`` [m],
    #: ``source``), when the caller supplied it (:func:`assemble_kinetic_inputs`); read only by the fit's
    #: radial-coordinate record, never by the model.
    midplane_radius: Optional[Dict[str, Any]] = None

    def _f(self, name):
        cache = self.__dict__.setdefault("_interp", {})
        if name not in cache:
            cache[name] = PchipInterpolator(self.x, getattr(self, name), extrapolate=False)
        return cache[name]

    def _coord(self, psi_n, rho):
        q = rho if self.coordinate == "rho_tor_norm" else psi_n
        if q is None:
            raise ValueError(f"kinetic inputs are on {self.coordinate}: pass it")
        q = np.asarray(q, dtype=float)
        if np.any(q < self.x[0] - 1e-12) or np.any(q > self.x[-1] + 1e-12):
            raise ValueError(f"kinetic inputs cover {self.coordinate} [{self.x[0]:g}, {self.x[-1]:g}]; requested "
                             f"[{q.min():g}, {q.max():g}] (no extrapolation)")
        return np.clip(q, self.x[0], self.x[-1])

    def profiles_at(self, psi_n=None, rho=None):
        """Every profile at the given surfaces of an equilibrium (``psi_N`` and/or ``rho_tor,norm``; the
        one matching ``coordinate`` is used).  Returns ``ne, te, ni, ti, zeff, nimp, n_fast``."""
        q = self._coord(psi_n, rho)
        return {k: self._f(k)(q) for k in ("ne", "te", "ni", "ti", "zeff", "nimp", "n_fast")}

    def with_temperature_factor(self, s):
        """These inputs with BOTH temperatures multiplied by ``s`` (``T_e -> s T_e``, ``T_i -> s T_i``; densities and
        ``Z_eff`` untouched): the kinetic inputs a fit sees at the temperature factor ``s`` (``s_temp`` of
        :mod:`.fit`).  Everything computed from them (thermal pressure, bootstrap current, conductivity) follows.
        ``s = 1`` returns this object itself; the copy's record and provenance name the factor."""
        s = float(s)
        if s == 1.0:
            return self
        if not (np.isfinite(s) and s > 0):
            raise ValueError("the temperature factor must be finite and > 0")
        cache = self.__dict__.setdefault("_scaled", {})
        if s not in cache:
            if len(cache) > 64:
                cache.clear()
            prov = dict(self.provenance)
            for k in ("te", "ti"):
                prov[k] = f"{prov.get(k, 'caller')}; MULTIPLIED by the fitted temperature factor {s:.6g}"
            cache[s] = replace(self, te=self.te * s, ti=self.ti * s, provenance=prov,
                               record=dict(self.record, temperature_factor=s))
        return cache[s]

    def thermal_pressure_at(self, psi_n=None, rho=None):
        """(K5): ``e (n_e T_e + (n_i + n_imp) T_i)`` [Pa], full frame."""
        d = self.profiles_at(psi_n, rho)
        return EC * (d["ne"] * d["te"] + (d["ni"] + d["nimp"]) * d["ti"])


#: Profiles moved together by the separatrix-temperature alignment: the electron profiles and, with the electron
#: density, the ion and impurity densities and Z_eff (so that quasi-neutrality and Z_eff hold as given at every
#: point); T_i and the fast-ion density are not moved.
ALIGNED_PROFILES = ("ne", "te", "ni", "nimp", "zeff")


def te_position_in_table(kin, te_target):
    """Where the kinetic inputs' T_e has the value ``te_target`` [eV] on their producing equilibrium's outboard
    midplane: ``(r_m, te_used, reached)``.  The OUTERMOST crossing is taken.  Where T_e never falls to the target
    inside the profile (its last point is hotter), the last point is used and ``te_used`` is its T_e (``reached``
    False): nothing is extrapolated."""
    tab = kin.midplane_radius
    xt = np.asarray(tab["rho_tor_norm"] if kin.coordinate == "rho_tor_norm" else tab["psi_n"], float)
    tr = np.asarray(tab["r_m"], float)
    x = np.asarray(kin.x, float)
    te = np.asarray(kin.te, float)
    m = x <= min(float(xt[-1]), float(x[-1])) + 1e-12
    x, te = x[m], te[m]
    if te[-1] > te_target:
        return float(np.interp(x[-1], xt, tr)), float(te[-1]), False
    i = int(np.max(np.where(te > te_target)[0])) if np.any(te > te_target) else 0
    x_t = x[i] + (te[i] - te_target) / (te[i] - te[i + 1]) * (x[i + 1] - x[i]) if i + 1 < len(x) and te[i] > te[i + 1] \
        else x[i]
    return float(np.interp(x_t, xt, tr)), float(te_target), True


def shifted_in_major_radius(kin, r_of_x, delta_m, names=ALIGNED_PROFILES):
    """Kinetic inputs with the profiles ``names`` moved rigidly in outboard-midplane major radius: at the
    reconstruction's surface of coordinate ``kin.x[i]`` and midplane radius ``r_of_x[i]`` [m] the profile takes the
    value the ORIGINAL has at the radius ``r_of_x[i] + delta_m`` of the producing equilibrium (through the midplane
    table the kinetic inputs carry).  Radii beyond the table's ends take the end values (no extrapolation).  The
    other profiles, the grid and the records are unchanged; ``record["separatrix_alignment"]`` states the move."""
    tab = kin.midplane_radius
    if tab is None:
        raise ValueError("the kinetic inputs carry no midplane table")
    xt = tab.get("rho_tor_norm") if kin.coordinate == "rho_tor_norm" else tab.get("psi_n")
    if xt is None:
        raise ValueError(f"the midplane table carries no {kin.coordinate}")
    xt, tr = np.asarray(xt, float), np.asarray(tab["r_m"], float)
    r_of_x = np.asarray(r_of_x, float)
    if r_of_x.shape != np.shape(kin.x) or np.any(np.diff(r_of_x) < 0):
        raise ValueError("r_of_x must give a non-decreasing major radius for every point of kin.x")
    xs = np.clip(np.interp(r_of_x + float(delta_m), tr, xt), float(kin.x[0]), float(kin.x[-1]))
    new = {n: np.asarray(kin._f(n)(xs), float) for n in names}
    rec = dict(kin.record)
    rec["separatrix_alignment"] = dict(delta_mm=1e3 * float(delta_m), profiles=list(names),
                                       rule="f(x) = original f at the producing equilibrium's midplane radius "
                                            "r_ours(x) + delta; end values beyond the table")
    return replace(kin, record=rec, **new)


def assemble_kinetic_inputs(x, ne, te, ti, *, zeff_candidates, zeff_source, z_imp, coordinate="rho_tor_norm",
                            n_fast=None, n_fast_source=None, psi_n_to_rho=None,
                            mapping_source=None, provenance=None, ni_reference=None, source="",
                            disagreement_range=(0.0, 0.9), zeff_rule=None, midplane_radius=None, ti_coverage=None):
    """Build :class:`KineticInputs` (module docstring).

    Parameters
    ----------
    x : array
        Radial coordinate of every input array: rho_tor,norm or psi_N (``coordinate``), strictly
        increasing, covering [0, 1] (no extrapolation is done).
    ne, te, ti : arrays on x
        [m^-3], [eV], [eV].
    zeff_candidates : dict
        ``name -> ("zeff", array)`` or ``name -> ("impurity_density", array)``, or ``name -> dict``
        with ``kind`` and ``value`` and, for the checked route, optionally ``sigma`` (1-sigma on x),
        ``fit_points`` (the points the profile was fitted to: a :class:`.zeff_sources.MeasurementPoints`
        or its dict), ``raw_points`` (raw analyses with times, for I-1 / I-4), ``provenance`` (str,
        recorded) and, for a line-integrated value, ``core_median_series`` (the advisory); at least
        one entry.  The checked route takes at most one candidate of each kind.
    zeff_source : str
        The candidate used, or ``"checked"`` for the checked-source rule (required: there is no
        default).
    zeff_rule : dict, optional
        Options of the checked route (:data:`ZEFF_RULE_OPTIONS`); refused with a named source.
    z_imp : float
        Charge of the single impurity species (> 1).
    n_fast : array on x, or ``"none"``
        Fast-ion density [m^-3] from the source provider, or :data:`FAST_IONS_NONE` (``"none"``) to declare that
        there are no fast ions (the zero density is then a stated input, recorded).  Required: ``None`` is
        refused (a missing fast-ion density is never taken as zero silently).
    n_fast_source : str, optional
        Name of the fast-ion source, recorded (default: what was declared).
    psi_n_to_rho : (psi_n, rho), optional
        With ``coordinate="psi_n"``: the producing equilibrium's own psi_N -> rho_tor,norm table; the
        inputs are then placed by rho (``mapping_source`` names that equilibrium).
    ni_reference : array on x, optional
        A main-ion density from the producer (e.g. a transport code's), compared, never used.
    midplane_radius : (psi_n, r_m), (psi_n, r_m, source) or (psi_n, r_m, source, rho_tor_norm), optional
        The producing equilibrium's outboard-midplane major radius [m] against its psi_N (both strictly
        increasing, finite).  Kept on the result for the fit's radial-coordinate record (how far the fitted
        surfaces sit from the producer's in real space); it does not enter the model.  The record says
        "not supplied" without it.
    ti_coverage : dict
        Required (``None`` is refused): the ion-temperature coverage check of the slice (module docstring;
        :data:`TI_COVERAGE_KEYS`, :func:`ti_coverage_check`).  ``points``: the ion-temperature measurement points;
        ``core_psi_n_max``: a psi_N or ``None`` (stated); ``slice_window``: for points with times; ``assumption``
        (:data:`TI_ASSUMPTIONS`) with ``assumption_sigma_rel`` (and, for ``"ratio_to_electron"``, ``assumption_ratio``
        and ``assumption_detail``): used only when the slice is not covered.  A slice that
        is not covered, with no assumption named, raises :class:`TiCoverageRefused`.  Recorded in
        ``record["ti_coverage"]``.
    """
    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}")
    if not zeff_candidates:
        raise ValueError("give at least one Z_eff candidate")
    if CHECKED in zeff_candidates:
        raise ValueError(f"{CHECKED!r} is reserved for the checked-source rule; rename that candidate")
    checked = zeff_source == CHECKED
    if zeff_source is None or not (checked or zeff_source in zeff_candidates):
        raise ValueError(f"zeff_source must name one of the candidates {sorted(zeff_candidates)} or be "
                         f"{CHECKED!r}: there is no default Z_eff source")
    if zeff_rule is not None and not checked:
        raise ValueError(f"zeff_rule options apply only with zeff_source={CHECKED!r}")
    z = float(z_imp)
    if not z > 1.0:
        raise ValueError("z_imp must be > 1")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or np.any(np.diff(x) <= 0):
        raise ValueError("x must increase strictly")
    arrs = dict(ne=ne, te=te, ti=ti)
    out = {}
    for k, v in arrs.items():
        v = np.asarray(v, dtype=float)
        if v.shape != x.shape or not np.all(np.isfinite(v)):
            raise ValueError(f"'{k}' must be finite and on x")
        out[k] = v
    inside = x <= 1.0
    for k in ("ne", "te", "ti"):
        if np.any(out[k][inside] <= 0):
            raise ValueError(f"'{k}' must be positive inside the plasma")
    if n_fast is None:
        raise ValueError("n_fast is required: the fast-ion density on x, or n_fast='none' to declare that there are "
                         "no fast ions (a missing fast-ion density is never taken as zero silently)")
    declared_none = isinstance(n_fast, str)
    if declared_none and n_fast != FAST_IONS_NONE:
        raise ValueError(f"n_fast must be an array on x or {FAST_IONS_NONE!r} (declared: no fast ions)")
    nf = np.zeros_like(x) if declared_none else np.asarray(n_fast, dtype=float)
    if nf.shape != x.shape or not np.all(np.isfinite(nf)) or np.any(nf < 0):
        raise ValueError("n_fast must be finite, >= 0 and on x")
    if n_fast_source is None:
        n_fast_source = "declared by the caller: no fast ions" if declared_none else "supplied (source not named)"
    if ti_coverage is None:
        raise ValueError("ti_coverage is required: the ion-temperature measurement points of the slice and the core "
                         "criterion, dict(points=..., core_psi_n_max=<psi_N or None>) (an ion temperature is never "
                         "taken as measured without the check)")
    ti_rec = _ti_coverage(x, coordinate, out, ti_coverage)
    rec = dict(z_imp=z, n_fast_source=n_fast_source, zeff_source=zeff_source, candidates={}, ti_coverage=ti_rec)
    cand = {}
    specs = {}
    for name, spec in zeff_candidates.items():
        if isinstance(spec, dict):
            unknown = set(spec) - set(_CANDIDATE_KEYS)
            if unknown or "kind" not in spec or "value" not in spec:
                raise ValueError(f"Z_eff candidate {name!r}: a dict needs 'kind' and 'value' and may carry only "
                                 f"{_CANDIDATE_KEYS} (got {sorted(spec)})")
            spec = dict(spec)
        else:
            kind_, arr_ = spec
            spec = dict(kind=kind_, value=arr_)
        specs[name] = spec
        kind, arr = spec["kind"], spec["value"]
        arr = np.asarray(arr, dtype=float)
        if arr.shape != x.shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"Z_eff candidate {name!r} must be finite and on x")
        if kind == "zeff":
            zc = arr
        elif kind == "impurity_density":
            if np.any(arr[inside] < 0):
                raise ValueError(f"impurity density {name!r} is negative inside the plasma")
            zc = zeff_from_impurity_density(out["ne"], arr, z)
        else:
            raise ValueError(f"candidate {name!r}: kind must be 'zeff' or 'impurity_density'")
        cand[name] = zc
        sel = inside & (x >= disagreement_range[0]) & (x <= disagreement_range[1])
        rec["candidates"][name] = dict(kind=kind, min=float(zc[inside].min()), max=float(zc[inside].max()),
                                       median=float(np.median(zc[sel])) if np.any(sel) else None)
        if spec.get("provenance") is not None:
            rec["candidates"][name]["provenance"] = str(spec["provenance"])
    names = sorted(cand)
    pairs = {}
    sel = inside & (x >= disagreement_range[0]) & (x <= disagreement_range[1])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (cand[a] - cand[b]) / cand[b]
            pairs[f"{a} vs {b}"] = dict(median_rel=float(np.median(d[sel])), max_abs_rel=float(np.max(np.abs(d[sel]))),
                                        range=list(disagreement_range), coordinate=coordinate)
    rec["pairwise_disagreement"] = pairs
    if checked:
        zeff_source = _checked_source(x, coordinate, out["ne"], nf, declared_none, z, specs, cand, psi_n_to_rho,
                                      dict(zeff_rule or {}), rec)
        rec["zeff_source"] = zeff_source
    zeff = cand[zeff_source]
    if np.any(zeff[inside] < 1.0):
        i = int(np.argmin(np.where(inside, zeff, np.inf)))
        raise ValueError(f"chosen Z_eff ({zeff_source}) < 1 at {coordinate} = {x[i]:.4g} ({zeff[i]:.4g}): refused, "
                         "nothing is clipped")
    if np.any(zeff[inside] > z):
        i = int(np.argmax(np.where(inside, zeff, -np.inf)))
        raise ValueError(f"chosen Z_eff ({zeff_source}) > Z = {z:g} at {coordinate} = {x[i]:.4g}: impossible with "
                         "one impurity of charge Z")
    nimp = out["ne"] * (zeff - 1.0) / (z * (z - 1.0))
    ni = out["ne"] - nf - z * nimp
    if np.any(ni[inside] <= 0):
        i = int(np.argmin(np.where(inside, ni, np.inf)))
        raise ValueError(f"main-ion density from quasineutrality is <= 0 at {coordinate} = {x[i]:.4g} "
                         f"(n_e {out['ne'][i]:.3e}, n_fast {nf[i]:.3e}, Z n_imp {z * nimp[i]:.3e}): inputs inconsistent")
    rec["fast_ion_fraction_of_ne"] = dict(axis=float(nf[0] / out["ne"][0]), max=float(np.max(nf[inside] / out["ne"][inside])))
    if ni_reference is not None:
        nr = np.asarray(ni_reference, dtype=float)
        rec["ni_vs_reference"] = dict(median_rel=float(np.median((ni[sel] - nr[sel]) / nr[sel])),
                                      max_abs_rel=float(np.max(np.abs((ni[sel] - nr[sel]) / nr[sel]))))
    prov = dict(provenance or {})
    prov.setdefault("ne", "caller")
    prov.setdefault("te", "caller")
    prov.setdefault("ti", "caller")
    if ti_rec["ti"] == "assumed":
        a = ti_rec["assumption"]
        prov["ti"] = (f"ASSUMED: {a['statement']} (rule {a['rule']!r}, stated relative 1-sigma {a['sigma_text']}) over "
                      f"{a['coordinate']} [{a['range'][0]:g}, {a['range'][1]:g}]; not measured: {ti_rec['reason']}; "
                      f"the supplied T_i ({prov['ti']}) is not used")
    prov["zeff"] = f"candidate {zeff_source!r}"
    if checked:
        d = rec["zeff_rule"]["decision"]
        prov["zeff"] += (f" chosen by the checked-source rule ({d['source']}, {d['source_state']}"
                         + (", explicit override" if d.get("override") else "") + f"; {d['row']})")
    elif specs[zeff_source].get("provenance") is not None:
        prov["zeff"] += f" ({specs[zeff_source]['provenance']}; named by the caller, not tested)"
    else:
        prov["zeff"] += " (named by the caller, not tested)"
    prov["nimp"] = f"(K3) from n_e and Z_eff ({zeff_source}), Z = {z:g}"
    prov["ni"] = "(K4) quasineutrality with fast ions"
    prov["n_fast"] = n_fast_source
    prov["pressure"] = "(K5) e (n_e T_e + (n_i + n_imp) T_i), impurity at T_i"
    xs, coord = x, coordinate
    if coordinate == "psi_n":
        if psi_n_to_rho is not None:
            tp, tr = (np.asarray(v, dtype=float) for v in psi_n_to_rho)
            if np.any(np.diff(tp) <= 0) or np.any(np.diff(tr) <= 0):
                raise ValueError("psi_n_to_rho must increase strictly in both coordinates")
            if x[0] < tp[0] - 1e-12 or min(x[-1], 1.0) > tp[-1] + 1e-12:
                raise ValueError("psi_n_to_rho does not cover the inputs' psi_N range inside the plasma")
            keep = x <= min(1.0, tp[-1])
            xs = PchipInterpolator(tp, tr, extrapolate=False)(x[keep])
            xs[0] = max(xs[0], 0.0)
            for k in list(out):
                out[k] = out[k][keep]
            zeff, nimp, ni, nf = zeff[keep], nimp[keep], ni[keep], nf[keep]
            if np.any(np.diff(xs) <= 0):
                raise ValueError("mapped rho is not strictly increasing")
            coord = "rho_tor_norm"
            rec["mapping"] = f"psi_N of {mapping_source or 'the producing equilibrium'} -> rho_tor,norm through " \
                             "that equilibrium's own table"
        else:
            rec["mapping"] = "psi_N kept: evaluated at the reconstruction's own psi_N in every pass (mapped through ours)"
    else:
        rec["mapping"] = "native rho_tor,norm (placed by rho)"
    if xs[0] > 1e-9 or xs[-1] < 1.0 - 1e-9:
        raise ValueError(f"kinetic inputs must cover [0, 1] in {coord} after mapping (got [{xs[0]:.4g}, {xs[-1]:.4g}]); "
                         "no extrapolation is done")
    mid = None
    if midplane_radius is not None:
        mp, mr = (np.asarray(v, dtype=float) for v in midplane_radius[:2])
        if mp.ndim != 1 or mp.shape != mr.shape or mp.size < 2 or not (np.all(np.isfinite(mp))
                                                                       and np.all(np.isfinite(mr))):
            raise ValueError("midplane_radius must be two finite 1-D arrays of the same length (psi_n, r_m)")
        if np.any(np.diff(mp) <= 0) or np.any(np.diff(mr) <= 0) or mr[0] <= 0:
            raise ValueError("midplane_radius: psi_n and r_m must increase strictly (outboard midplane, r_m > 0)")
        mid = dict(psi_n=mp, r_m=mr, source=str(midplane_radius[2]) if len(midplane_radius) > 2 else
                   (mapping_source or "the producing equilibrium"))
        # the table's own rho_tor,norm (a fourth element, or from psi_n_to_rho): read only by the fit's record of
        # T_e at the reconstruction's separatrix in real space (fit.separatrix_te_record), never by the model
        mrho = None
        if len(midplane_radius) > 3 and midplane_radius[3] is not None:
            mrho = np.asarray(midplane_radius[3], dtype=float)
            if mrho.shape != mp.shape or not np.all(np.isfinite(mrho)) or np.any(np.diff(mrho) <= 0):
                raise ValueError("midplane_radius: rho_tor_norm must be finite, strictly increasing and as long as psi_n")
        elif psi_n_to_rho is not None:
            tp_, tr_ = (np.asarray(v, dtype=float) for v in psi_n_to_rho)
            if mp[0] >= tp_[0] - 1e-12 and mp[-1] <= tp_[-1] + 1e-12:
                mrho = np.interp(mp, tp_, tr_)
        if mrho is not None:
            mid["rho_tor_norm"] = mrho
    rec["midplane_radius"] = ("not supplied" if mid is None else
                              f"supplied: {mid['source']}, {mp.size} points, psi_N {mp[0]:.4g} to {mp[-1]:.4g}")
    return KineticInputs(coordinate=coord, x=xs, ne=out["ne"], te=out["te"], ti=out["ti"], zeff=zeff, ni=ni,
                         nimp=nimp, n_fast=nf, z_imp=z, provenance=prov, record=rec, source=source,
                         midplane_radius=mid)


def _checked_source(x, coordinate, ne, n_fast, n_fast_none, z, specs, cand, psi_n_to_rho, opts, rec):
    """The checked-source rule on the candidates (module docstring); returns the chosen candidate's
    name and stores the rule record in ``rec["zeff_rule"]``.  Refusals raise
    :class:`.zeff_sources.ZeffSourceRefused` with the record attached."""
    unknown = set(opts) - set(ZEFF_RULE_OPTIONS)
    if unknown:
        raise ValueError(f"unknown zeff_rule option(s) {sorted(unknown)}; known: {ZEFF_RULE_OPTIONS}")
    by_kind = {}
    for name, spec in specs.items():
        by_kind.setdefault(spec["kind"], []).append(name)
    for kind, names in by_kind.items():
        if len(names) > 1:
            raise ValueError(f"the checked route takes at most one candidate of each kind; {kind!r}: {names}")
    imp_name = (by_kind.get("impurity_density") or [None])[0]
    line_name = (by_kind.get("zeff") or [None])[0]
    # the rule's regions are in psi_N; its point positions in rho_tor,norm
    if coordinate == "psi_n":
        psi = np.asarray(x, dtype=float)
        if opts.get("psi_n") is not None:
            raise ValueError("zeff_rule 'psi_n' is for a rho_tor_norm table; this table is on psi_N already")
        rho_map = None if psi_n_to_rho is None else tuple(np.asarray(v, dtype=float) for v in psi_n_to_rho)
    else:
        if opts.get("psi_n") is None:
            raise ValueError("the checked route on a rho_tor_norm table needs zeff_rule['psi_n'] (psi_N of every x, "
                             "from the producing equilibrium): the rule's test regions are in psi_N")
        psi = np.asarray(opts["psi_n"], dtype=float)
        if psi.shape != np.shape(x) or np.any(np.diff(psi) <= 0):
            raise ValueError("zeff_rule['psi_n'] must be on x and increase strictly")
        inside = psi <= 1.0
        rho_map = (psi[inside], np.asarray(x, dtype=float)[inside])
    fast = opts.get("fast_ions")
    if fast not in (None, "none"):
        raise ValueError("zeff_rule['fast_ions'] must be 'none' (declared absent) or omitted")
    if fast == "none" or n_fast_none:
        if np.any(n_fast > 0):
            raise ValueError("zeff_rule declares no fast ions but a non-zero n_fast was given")
        nf_rule = "none"
    else:
        nf_rule = n_fast
    st, sw = opts.get("slice_time"), opts.get("slice_window")
    imp = line = None
    sig_zimp = zimp = None
    if imp_name is not None:
        sp = specs[imp_name]
        nimp = np.asarray(sp["value"], dtype=float)
        sig = None if sp.get("sigma") is None else np.asarray(sp["sigma"], dtype=float)
        imp = _zr.impurity_tests(psi, ne, nimp, sig, impurity_Z=z, n_fast=nf_rule, rho_map=rho_map,
                                 fit_points=sp.get("fit_points"), raw_points=sp.get("raw_points"), slice_time=st,
                                 slice_window=sw)
        zimp = cand[imp_name]
        if sig is not None:
            sig_zimp = z * (z - 1.0) * sig / ne
    if line_name is not None:
        sp = specs[line_name]
        line = _zr.line_tests(psi, cand[line_name], sp.get("sigma"), zeff_impurity=zimp,
                              core_median_series=sp.get("core_median_series"))
    names = {_zr.IMPURITY: imp_name, _zr.LINE: line_name}
    override = opts.get("override")
    ov_kind = None
    if override is not None:
        if override not in specs:
            raise ValueError(f"zeff_rule override {override!r} is not one of the candidates {sorted(specs)}")
        ov_kind = _zr.IMPURITY if override == imp_name else _zr.LINE
    record = _zr.zeff_source_record(imp, line, sigma_zeff_imp=sig_zimp, psi_N=psi, zeff_imp=zimp,
                                    zeff_line=None if line_name is None else cand[line_name],
                                    sigma_zeff_line=None if line_name is None else specs[line_name].get("sigma"),
                                    override=ov_kind, time=st)
    record["candidate_names"] = dict(names)
    record["accept_failed"] = bool(opts.get("accept_failed", False))
    rec["zeff_rule"] = record
    dec = record["decision"]
    if dec["source"] is None:
        raise _zr.ZeffSourceRefused("Z_eff source refused by the checked-source rule: " + _zr.refusal_message(record),
                                    record)
    if dec.get("override") and dec["source_state"] == _zr.FAILED and not record["accept_failed"]:
        raise _zr.ZeffSourceRefused(
            f"the named Z_eff candidate {override!r} fails "
            + ", ".join(dec["override"]["named_failed_tests"])
            + "; refused (pass zeff_rule={'override': ..., 'accept_failed': True} to use it anyway, recorded)", record)
    msg = _zr.checked_warning(record)
    if msg:
        warnings.warn(msg, _zr.ZeffSourceWarning, stacklevel=3)
    return names[dec["source"]]
