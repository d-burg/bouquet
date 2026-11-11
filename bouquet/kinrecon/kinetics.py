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
Z_eff and a CER impurity density) is recorded with the pairwise disagreement, and the caller names
the one used (``zeff_source``); a run that omits it is refused.  Nothing is clipped: ``Z_eff < 1``,
``Z_eff > Z`` or a non-positive ``n_i`` inside the plasma is refused with the location.

Coordinate (the working coordinate is rho_tor,norm):

* profiles that carry rho_tor,norm are placed by it (``coordinate="rho_tor_norm"``);
* profiles that carry only psi_N of ANOTHER equilibrium are mapped through that equilibrium's own
  ``psi_N -> rho_tor,norm`` table when the caller supplies it (``psi_n_to_rho=(psi_n, rho)``,
  recorded), and kept on psi_N otherwise (``coordinate="psi_n"``: then they are evaluated at the
  CURRENT equilibrium's psi_N in every pass, which is recorded as "mapped through ours").

The result evaluates on any equilibrium through :meth:`KineticInputs.profiles_at` with that
equilibrium's ``(psi_N, rho)`` pair; :mod:`.current_model` calls it on every pass.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

EC = 1.602176634e-19

__all__ = ["KineticInputs", "assemble_kinetic_inputs", "zeff_from_impurity_density", "COORDINATES"]

#: Coordinates a kinetic table can carry.
COORDINATES = ("rho_tor_norm", "psi_n")


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
                            disagreement_range=(0.0, 0.9)):
    """Build :class:`KineticInputs` (module docstring).

    Parameters
    ----------
    x : array
        Radial coordinate of every input array: rho_tor,norm or psi_N (``coordinate``), strictly
        increasing, covering [0, 1] (no extrapolation is done).
    ne, te, ti : arrays on x
        [m^-3], [eV], [eV].
    zeff_candidates : dict
        ``name -> ("zeff", array)`` or ``name -> ("impurity_density", array)``; at least one entry.
    zeff_source : str
        The candidate used (required: there is no default).
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
    if zeff_source is None or zeff_source not in zeff_candidates:
        raise ValueError(f"zeff_source must name one of the candidates {sorted(zeff_candidates)}: there is no "
                         "default Z_eff source")
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
    for name, spec in zeff_candidates.items():
        kind, arr = spec
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
    names = sorted(cand)
    pairs = {}
    sel = inside & (x >= disagreement_range[0]) & (x <= disagreement_range[1])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (cand[a] - cand[b]) / cand[b]
            pairs[f"{a} vs {b}"] = dict(median_rel=float(np.median(d[sel])), max_abs_rel=float(np.max(np.abs(d[sel]))),
                                        range=list(disagreement_range), coordinate=coordinate)
    rec["pairwise_disagreement"] = pairs
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
