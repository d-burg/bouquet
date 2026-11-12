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

Coordinate (the working coordinate is rho_tor,norm):

* profiles that carry rho_tor,norm are placed by it (``coordinate="rho_tor_norm"``);
* profiles that carry only psi_N of ANOTHER equilibrium are mapped through that equilibrium's own
  ``psi_N -> rho_tor,norm`` table when the caller supplies it (``psi_n_to_rho=(psi_n, rho)``,
  recorded), and kept on psi_N otherwise (``coordinate="psi_n"``: then they are evaluated at the
  CURRENT equilibrium's psi_N in every pass, which is recorded as "mapped through ours").

The result evaluates on any equilibrium through :meth:`KineticInputs.profiles_at` with that
equilibrium's ``(psi_N, rho)`` pair; :mod:`.current_model` calls it on every pass.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import zeff_sources as _zr

EC = 1.602176634e-19

__all__ = ["KineticInputs", "assemble_kinetic_inputs", "zeff_from_impurity_density", "COORDINATES",
           "CHECKED", "ZEFF_RULE_OPTIONS"]

#: Coordinates a kinetic table can carry.
COORDINATES = ("rho_tor_norm", "psi_n")
#: The ``zeff_source`` value that applies the checked-source rule (module docstring).
CHECKED = "checked"
#: Keys of the ``zeff_rule`` options (all optional): ``psi_n`` (psi_N of every ``x``, needed for a
#: ``rho_tor_norm`` table: the rule's regions are in psi_N), ``slice_time`` [s], ``slice_window``
#: ``(t0, t1)`` [s], ``override`` (a candidate name), ``accept_failed`` (bool), ``fast_ions``
#: (``"none"``: the caller declares there are no fast ions; otherwise the ``n_fast`` given is used,
#: or I-3 is evaluated without fast ions when ``n_fast`` is None).
ZEFF_RULE_OPTIONS = ("psi_n", "slice_time", "slice_window", "override", "accept_failed", "fast_ions")
#: Keys a candidate may carry when given as a dict (``value`` and ``kind`` required).
_CANDIDATE_KEYS = ("kind", "value", "sigma", "fit_points", "raw_points", "provenance", "core_median_series")


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

    def thermal_pressure_at(self, psi_n=None, rho=None):
        """(K5): ``e (n_e T_e + (n_i + n_imp) T_i)`` [Pa], full frame."""
        d = self.profiles_at(psi_n, rho)
        return EC * (d["ne"] * d["te"] + (d["ni"] + d["nimp"]) * d["ti"])


def assemble_kinetic_inputs(x, ne, te, ti, *, zeff_candidates, zeff_source, z_imp, coordinate="rho_tor_norm",
                            n_fast=None, n_fast_source="none (no fast ions given)", psi_n_to_rho=None,
                            mapping_source=None, provenance=None, ni_reference=None, source="",
                            disagreement_range=(0.0, 0.9), zeff_rule=None):
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
    n_fast : array on x, optional
        Fast-ion density [m^-3] from the source provider (zero when None, recorded).
    psi_n_to_rho : (psi_n, rho), optional
        With ``coordinate="psi_n"``: the producing equilibrium's own psi_N -> rho_tor,norm table; the
        inputs are then placed by rho (``mapping_source`` names that equilibrium).
    ni_reference : array on x, optional
        A main-ion density from the producer (e.g. a transport code's), compared, never used.
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
    nf = np.zeros_like(x) if n_fast is None else np.asarray(n_fast, dtype=float)
    if nf.shape != x.shape or not np.all(np.isfinite(nf)) or np.any(nf < 0):
        raise ValueError("n_fast must be finite, >= 0 and on x")
    rec = dict(z_imp=z, n_fast_source=n_fast_source, zeff_source=zeff_source, candidates={})
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
        zeff_source = _checked_source(x, coordinate, out["ne"], nf, n_fast is None, z, specs, cand, psi_n_to_rho,
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
    return KineticInputs(coordinate=coord, x=xs, ne=out["ne"], te=out["te"], ti=out["ti"], zeff=zeff, ni=ni,
                         nimp=nimp, n_fast=nf, z_imp=z, provenance=prov, record=rec, source=source)


def _checked_source(x, coordinate, ne, n_fast, n_fast_missing, z, specs, cand, psi_n_to_rho, opts, rec):
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
    if fast == "none":
        if not n_fast_missing and np.any(n_fast > 0):
            raise ValueError("zeff_rule declares no fast ions but a non-zero n_fast was given")
        nf_rule = "none"
    else:
        nf_rule = None if n_fast_missing else n_fast
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
