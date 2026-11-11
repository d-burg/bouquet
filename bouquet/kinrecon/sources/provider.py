r"""The provider output container :class:`NonInductiveSources`, its uncertainty and provenance records.

A *provider* is anything that returns a :class:`NonInductiveSources` for one time slice (or
window): :func:`.arrays.from_arrays` (any transport code's arrays with declared
conventions), :func:`.imas.from_imas` (IMAS ``core_sources`` / ``core_profiles``),
:class:`.beam.SimpleBeamModel` and :class:`.ec.SimpleECModel` (built-in zeroth-order
models).  Every field is a plain numpy array on the stated grid, in the frame stated in
:mod:`.conventions`:

=====================  ========================================================  ===========
field                  meaning                                                   units
=====================  ========================================================  ===========
``rho``                rho_tor_norm grid (C1), strictly increasing               --
``jdotb[name]``        <J_CD . B> per source, signed relative to Ip              A T/m^2
``B0``, ``R0``         reference vacuum field magnitude and its major radius     T, m
``p_fast[species]``    isotropic-equivalent fast-ion pressure (C8)               Pa
``p_par[species]``,    TRUE directional pressures (optional; only when the       Pa
``p_perp[species]``    producer computed or stored the anisotropy)
``n_fast[species]``    fast-ion density                                          m^-3
``heating[name]``      (q_e, q_i) power densities (optional)                     W/m^3
``uncertainty``        list of :class:`ScalePrior` / :class:`ShapePrior`         --
``provenance``         :class:`Provenance`                                       --
=====================  ========================================================  ===========

``j_parallel`` (= <J.B>/B0, eq. C2) and the toroidal frames (eqs. C3-C7, need geometry) are
computed on request, never stored twice.

No silent defaults: the constructor refuses a missing grid, mismatched lengths, non-finite
values, a non-positive B0, or a pressure entry whose true components disagree with its
isotropic-equivalent value.  ``uncertainty`` may be left empty, but then
:meth:`NonInductiveSources.completeness` reports it and
:meth:`NonInductiveSources.require_complete` raises -- the default prior widths are a
user decision (project decision D-B3), so nothing is filled in on the caller's behalf.
"""

import datetime
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conventions import (CURRENT_FRAME, PRESSURE_CONVENTION, RADIAL_COORDINATE, jpar_from_jdotb, p_fast_from_true,
                          psi_n_to_rho, toroidal_frames)

__all__ = ["ScalePrior", "ShapePrior", "Provenance", "NonInductiveSources", "SOURCES_VERSION"]

#: Version of the ``kinrecon.sources`` layer recorded in every provenance record.
SOURCES_VERSION = "0.1.0"

_QUANTITIES = ("jdotb", "p_fast", "n_fast", "heating")


@dataclass(frozen=True)
class ScalePrior:
    """Multiplicative scale-factor prior on one quantity (log-normal unless stated).

    ``quantity`` is ``"jdotb"``, ``"p_fast"``, ``"n_fast"`` or ``"heating"``; ``target`` names
    the source or species it multiplies (``"*"`` = all).  The draw is
    ``s = center * exp(sigma_log * z)``, ``z ~ N(0, 1)``, truncated to ``s <= upper`` when
    ``upper`` is given (one-sided priors, e.g. anomalous fast-ion loss).  ``region`` optionally
    restricts it to ``(rho_min, rho_max)``.  ``basis`` must say what the width rests on and
    ``status`` whether it is ``"decided"`` (user) or ``"proposed"``.
    """

    quantity: str
    target: str
    center: float
    sigma_log: float
    basis: str
    status: str
    upper: Optional[float] = None
    region: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"ScalePrior.quantity must be one of {_QUANTITIES}")
        if not (self.center > 0 and self.sigma_log >= 0):
            raise ValueError("ScalePrior needs center > 0 and sigma_log >= 0")
        if self.status not in ("decided", "proposed"):
            raise ValueError("ScalePrior.status must be 'decided' or 'proposed'")
        if not self.basis:
            raise ValueError("ScalePrior.basis must state what the width rests on")


@dataclass(frozen=True)
class ShapePrior:
    """Prior on a location or shape parameter of a profile.

    ``kind`` is ``"shift"`` (rigid radial shift delta-rho of the profile, sigma in rho units),
    ``"width"`` (log-normal scale of the profile width about its centroid, sigma_log) or
    ``"flatten"`` (anomalous-diffusion flattening that conserves the volume integral).
    """

    quantity: str
    target: str
    kind: str
    sigma: float
    basis: str
    status: str

    def __post_init__(self):
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"ShapePrior.quantity must be one of {_QUANTITIES}")
        if self.kind not in ("shift", "width", "flatten"):
            raise ValueError("ShapePrior.kind must be 'shift', 'width' or 'flatten'")
        if self.sigma < 0 or self.status not in ("decided", "proposed") or not self.basis:
            raise ValueError("ShapePrior needs sigma >= 0, a status and a basis")


@dataclass
class Provenance:
    """Where the numbers came from.  Every field is required except ``notes``.

    ``provider`` -- the kinrecon provider (``"arrays"``, ``"imas"``, ``"simple_beam"``, ...);
    ``source_code`` / ``source_version`` -- the producing code (e.g. ``"ONETWO"`` /
    ``"5.9.7"``, ``"kinrecon.sources.beam"`` / :data:`SOURCES_VERSION`); ``inputs`` -- what was
    read (paths, hashes, time window); ``conventions`` -- the declared units/frames/pressure
    form of the inputs; ``settings`` -- every model setting, defaults included;
    ``assumptions`` and ``limitations`` -- plain-language lists.
    """

    provider: str
    source_code: str
    source_version: str
    inputs: Dict[str, object]
    conventions: Dict[str, object]
    settings: Dict[str, object]
    assumptions: List[str]
    limitations: List[str]
    notes: str = ""
    created_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    sources_version: str = SOURCES_VERSION

    def __post_init__(self):
        for name in ("provider", "source_code", "source_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Provenance.{name} must be a non-empty string")
        for name in ("inputs", "conventions", "settings"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Provenance.{name} must be a dict")
        for name in ("assumptions", "limitations"):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"Provenance.{name} must be a list")

    def to_dict(self):
        return {k: getattr(self, k) for k in ("provider", "source_code", "source_version", "inputs", "conventions",
                                              "settings", "assumptions", "limitations", "notes", "created_utc",
                                              "sources_version")}


def _arr(x, n, name):
    a = np.array(x, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


@dataclass
class NonInductiveSources:
    """Driven current, fast-ion pressure/density and their uncertainty on one rho_tor_norm grid.

    See the module docstring for fields and units.  ``time`` is the slice time [s] and
    ``time_window`` the averaging window [s] (0 for a single slice).
    """

    rho: np.ndarray
    time: float
    time_window: float
    B0: float
    R0: float
    jdotb: Dict[str, np.ndarray]
    p_fast: Dict[str, np.ndarray]
    n_fast: Dict[str, np.ndarray]
    provenance: Provenance
    p_par: Dict[str, np.ndarray] = field(default_factory=dict)
    p_perp: Dict[str, np.ndarray] = field(default_factory=dict)
    heating: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    uncertainty: List[object] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    radial_coordinate = RADIAL_COORDINATE
    current_frame = CURRENT_FRAME
    pressure_convention = PRESSURE_CONVENTION

    def __post_init__(self):
        rho = np.array(self.rho, dtype=float)
        if rho.ndim != 1 or rho.size < 2 or np.any(np.diff(rho) <= 0) or rho[0] < 0 or rho[-1] > 1.0 + 1e-12:
            raise ValueError("rho must be a strictly increasing rho_tor_norm grid within [0, 1]")
        self.rho = rho
        n = rho.size
        for name in ("B0", "R0"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a positive magnitude")
            setattr(self, name, v)
        self.time = float(self.time)
        self.time_window = float(self.time_window)
        if self.time_window < 0:
            raise ValueError("time_window must be >= 0")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance record")
        self.jdotb = {k: _arr(v, n, f"jdotb[{k!r}]") for k, v in self.jdotb.items()}
        self.p_fast = {k: _arr(v, n, f"p_fast[{k!r}]") for k, v in self.p_fast.items()}
        self.n_fast = {k: _arr(v, n, f"n_fast[{k!r}]") for k, v in self.n_fast.items()}
        self.p_par = {k: _arr(v, n, f"p_par[{k!r}]") for k, v in self.p_par.items()}
        self.p_perp = {k: _arr(v, n, f"p_perp[{k!r}]") for k, v in self.p_perp.items()}
        if set(self.p_par) != set(self.p_perp):
            raise ValueError("p_par and p_perp must name the same species")
        for k in self.p_par:
            if k not in self.p_fast:
                raise ValueError(f"true components given for {k!r} without its p_fast")
            iso = p_fast_from_true(self.p_par[k], self.p_perp[k])
            scale = max(np.max(np.abs(self.p_fast[k])), 1e-30)
            if np.max(np.abs(iso - self.p_fast[k])) > 1e-9 * scale + 1e-12:
                raise ValueError(f"p_fast[{k!r}] disagrees with (P_par + 2 P_perp)/3 of its true components")
        for k, v in self.p_fast.items():
            if np.any(v < 0):
                raise ValueError(f"p_fast[{k!r}] is negative somewhere (refused; clip upstream and say so)")
        for k, v in self.n_fast.items():
            if np.any(v < 0):
                raise ValueError(f"n_fast[{k!r}] is negative somewhere")
        self.heating = {k: (_arr(qe, n, f"heating[{k!r}].q_e"), _arr(qi, n, f"heating[{k!r}].q_i"))
                        for k, (qe, qi) in self.heating.items()}

    # ------------------------------------------------------------------ totals and frames
    @property
    def jdotb_total(self):
        """Sum of <J_CD.B> over sources [A T/m^2]."""
        return sum(self.jdotb.values(), np.zeros_like(self.rho))

    @property
    def p_fast_total(self):
        """Sum of the isotropic-equivalent fast-ion pressure over species [Pa] (C8)."""
        return sum(self.p_fast.values(), np.zeros_like(self.rho))

    @property
    def n_fast_total(self):
        """Sum of fast-ion densities over species [m^-3]."""
        return sum(self.n_fast.values(), np.zeros_like(self.rho))

    def j_parallel(self, name=None):
        """IMAS ``j_parallel`` = <J.B>/B0 [A/m^2] of one source or the total (C2)."""
        return jpar_from_jdotb(self.jdotb_total if name is None else self.jdotb[name], self.B0)

    def toroidal(self, geometry, name=None):
        """Toroidal-current frames (C3-C7) of one source or the total; needs a FluxSurfaceGeometry."""
        return toroidal_frames(self.jdotb_total if name is None else self.jdotb[name], self.rho, geometry)

    def on_psi_n(self, psi_n, psi_n_table, rho_table, which="p_fast_total"):
        """A total profile resampled onto ``psi_n`` through the caller's psi_N <-> rho table.

        ``which`` is ``"p_fast_total"``, ``"jdotb_total"`` or ``"n_fast_total"``.  Points outside the
        provider's grid are NaN (no extrapolation).
        """
        rho_q = psi_n_to_rho(psi_n, psi_n_table, rho_table)
        y = getattr(self, which)
        out = np.interp(rho_q, self.rho, y, left=np.nan, right=np.nan)
        out[~np.isfinite(rho_q)] = np.nan
        return out

    # ------------------------------------------------------------------ completeness
    def completeness(self):
        """List of plain-language gaps (empty when every quantity has a prior and provenance is full)."""
        gaps = []
        if not self.uncertainty:
            gaps.append("no uncertainty descriptors (prior widths are a user decision, D-B3)")
        else:
            covered = {(u.quantity, u.target) for u in self.uncertainty if isinstance(u, ScalePrior)}
            for q, d in (("jdotb", self.jdotb), ("p_fast", self.p_fast), ("n_fast", self.n_fast)):
                for k in d:
                    if (q, k) not in covered and (q, "*") not in covered:
                        gaps.append(f"no scale prior for {q}[{k!r}]")
        if not self.provenance.assumptions:
            gaps.append("provenance lists no assumptions")
        if not self.provenance.limitations:
            gaps.append("provenance lists no limitations")
        return gaps

    def require_complete(self):
        """Raise ``ValueError`` listing every gap of :meth:`completeness`."""
        gaps = self.completeness()
        if gaps:
            raise ValueError("incomplete NonInductiveSources: " + "; ".join(gaps))
        return self

    # ------------------------------------------------------------------ serialization
    def to_dict(self):
        """JSON-ready dict (arrays as lists); :meth:`from_dict` inverts it."""
        def pri(u):
            d = dict(u.__dict__)
            d["type"] = type(u).__name__
            return d
        return {
            "radial_coordinate": RADIAL_COORDINATE, "current_frame": CURRENT_FRAME,
            "pressure_convention": PRESSURE_CONVENTION,
            "rho": self.rho.tolist(), "time": self.time, "time_window": self.time_window,
            "B0": self.B0, "R0": self.R0,
            "jdotb": {k: v.tolist() for k, v in self.jdotb.items()},
            "p_fast": {k: v.tolist() for k, v in self.p_fast.items()},
            "n_fast": {k: v.tolist() for k, v in self.n_fast.items()},
            "p_par": {k: v.tolist() for k, v in self.p_par.items()},
            "p_perp": {k: v.tolist() for k, v in self.p_perp.items()},
            "heating": {k: [qe.tolist(), qi.tolist()] for k, (qe, qi) in self.heating.items()},
            "uncertainty": [pri(u) for u in self.uncertainty],
            "provenance": self.provenance.to_dict(), "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        for key, want in (("radial_coordinate", RADIAL_COORDINATE), ("current_frame", CURRENT_FRAME),
                          ("pressure_convention", PRESSURE_CONVENTION)):
            if d.get(key) != want:
                raise ValueError(f"serialized sources declare {key}={d.get(key)!r}; this version reads {want!r}")
        unc = []
        for u in d.get("uncertainty", []):
            u = dict(u)
            typ = u.pop("type")
            if u.get("region") is not None:
                u["region"] = tuple(u["region"])
            unc.append({"ScalePrior": ScalePrior, "ShapePrior": ShapePrior}[typ](**u))
        prov = dict(d["provenance"])
        return cls(rho=d["rho"], time=d["time"], time_window=d["time_window"], B0=d["B0"], R0=d["R0"],
                   jdotb=d["jdotb"], p_fast=d["p_fast"], n_fast=d["n_fast"], p_par=d.get("p_par", {}),
                   p_perp=d.get("p_perp", {}), heating={k: tuple(v) for k, v in d.get("heating", {}).items()},
                   uncertainty=unc, provenance=Provenance(**prov), meta=d.get("meta", {}))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
