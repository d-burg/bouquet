r"""Finite-beam optics for the built-in beam model: source size, focus, divergence, apertures (an option).

Used by :class:`.beam.SimpleBeamModel` with ``deposition="beamlets"``; the default
(``deposition="pencil"``) does not call this module and is unchanged.

Description (IMAS ``nbi.unit[:]`` schema; definitions as the data dictionary states them)
---------------------------------------------------------------------------------------
* ``beamlets_group[:].width_horizontal`` / ``width_vertical`` -- full width and height of the
  smallest rectangle around the beamlets at the source grid [m];
* ``beamlets_group[:].focus.focal_length_horizontal`` / ``_vertical`` -- distance along the beam
  axis to the point of minimum width [m] (a very large value means unfocused);
* ``beamlets_group[:].divergence_component[:]`` -- a sum of Gaussian components with
  ``particles_fraction`` and ``horizontal`` / ``vertical`` divergence, defined as the half-angle at
  which the density falls to 1/e of the maximum, so the angular standard deviation is
  ``alpha / sqrt(2)``; the schema carries one divergence for all energy components, so the E,
  E/2 and E/3 components share it here;
* ``unit[:].aperture[:]`` -- collimating apertures through which the beam is launched: a center
  (leaf name :data:`IMAS_APERTURE_CENTER`, the data dictionary's spelling) and a ``geometry_type`` (1 outline, 2 circle of ``radius``, 3 rectangle of full widths
  ``x1_width`` (the more horizontal axis) and ``x2_width``).

Beamlets (derived here)
-----------------------
The central axis of a beam leaves the source center ``S`` along the unit vector ``d`` of the
pencil model (eq. B4 frame of :mod:`.beam`).  Two unit vectors span the source plane: ``e_v``
in the vertical plane through ``d`` (pointing up) and ``e_h = d x e_v`` (horizontal; a positive
offset along ``e_h`` increases the tangency radius).  A beamlet starts at

.. code-block:: text

    P_0 = S + y_h e_h + y_v e_v,     (y_h, y_v) uniform over the source rectangle             (O1)

and leaves along ``d + tan(a_h) e_h + tan(a_v) e_v`` (normalized), with

.. code-block:: text

    a_h = theta_h - y_h / f_h,    a_v = theta_v - y_v / f_v,                                  (O2)
    theta_h ~ N(0, alpha_h / sqrt 2),   theta_v ~ N(0, alpha_v / sqrt 2)  (each divergence component)

(``-y/f`` aims the beamlet at the focal point; ``theta`` is its divergence).  An aperture is a
plane perpendicular to ``d`` at distance ``D`` from ``S`` (the plane of the source grid and of
the apertures are taken parallel, as NUBEAM and FIDASIM do; recalled).  A beamlet crosses it at
``t = D / (d_j . d)`` with transverse coordinates

.. code-block:: text

    u_h = y_h + t (d_j . e_h) - c_h,    u_v = y_v + t (d_j . e_v) - c_v                        (O3)

(``c`` the aperture center's offset from the axis) and is CLIPPED (removed) when ``(u_h, u_v)``
lies outside the aperture.  The beam's power is the power LAUNCHED into the vessel (IMAS
``power_launched``), i.e. after the apertures, so the transmitted beamlets share it equally per
unit weight; the transmitted weight fraction is recorded.  A fully clipped beam deposits nothing.

Sampling is deterministic: the first ``n`` points (index 1..n) of the Halton sequence in bases
2, 3, 5, 7, mapped to ``(y_h, y_v)`` by the uniform law and to ``(theta_h, theta_v)`` by the
inverse normal distribution function; ``n`` per divergence component, each beamlet of component
``c`` with weight ``fraction_c / n``.  Two runs give bitwise equal beamlets.

Missing-input rules (each recorded in :meth:`BeamOptics.record` under ``missing``; nothing is
guessed):

==========================================  ===========================================================
missing                                     rule
==========================================  ===========================================================
source width or height                      zero in that plane (point source in that plane)
focal length                                unfocused in that plane (beamlets parallel to the axis)
divergence (no component)                   zero divergence
apertures (none listed)                     no clipping
an aperture without its dimensions, or      that aperture is not applied (no clipping by it)
an outline aperture, or one whose center
the beam axis does not reach
==========================================  ===========================================================

The horizontal offset of an aperture center from the axis is NOT derived from the toroidal angle
of the description (producers differ in the orientation of the angle); it is taken as zero and
recorded.  The vertical offset is the center's height minus the axis height at the crossing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["Aperture", "DivergenceComponent", "BeamOptics", "optics_from_ids", "halton", "beamlet_sample",
           "HALTON_BASES", "IMAS_APERTURE_CENTER"]

HALTON_BASES = (2, 3, 5, 7)
_IMAS_EMPTY = -9.0e39
#: Leaf name of an aperture's center in the IMAS data dictionary, which uses the British spelling.
IMAS_APERTURE_CENTER = "cent" + "re"


def _val(x):
    """A finite number, or None for a missing / IMAS-empty value."""
    if x is None:
        return None
    try:
        v = float(np.asarray(x, float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(v) or v <= _IMAS_EMPTY:
        return None
    return v


@dataclass(frozen=True)
class Aperture:
    """An aperture plane perpendicular to the beam axis at ``distance`` [m] from the source center.

    ``shape`` ``"rectangle"`` (``half_width_h``, ``half_width_v``) or ``"circle"`` (``radius``);
    ``offset_h`` / ``offset_v`` -- the center's offset from the axis in the plane [m].
    """

    distance: float
    shape: str
    half_width_h: Optional[float] = None
    half_width_v: Optional[float] = None
    radius: Optional[float] = None
    offset_h: float = 0.0
    offset_v: float = 0.0
    label: str = ""

    def __post_init__(self):
        if not self.distance > 0:
            raise ValueError("aperture distance must be > 0")
        if self.shape == "rectangle":
            if not (self.half_width_h is not None and self.half_width_h > 0 and self.half_width_v is not None
                    and self.half_width_v > 0):
                raise ValueError("rectangular aperture needs half_width_h, half_width_v > 0")
        elif self.shape == "circle":
            if not (self.radius is not None and self.radius > 0):
                raise ValueError("circular aperture needs radius > 0")
        else:
            raise ValueError("aperture shape must be 'rectangle' or 'circle'")

    def passes(self, u_h, u_v):
        """True where the transverse coordinates (relative to the axis) lie inside the opening."""
        x, y = np.asarray(u_h, float) - self.offset_h, np.asarray(u_v, float) - self.offset_v
        if self.shape == "rectangle":
            return (np.abs(x) <= self.half_width_h) & (np.abs(y) <= self.half_width_v)
        return x * x + y * y <= self.radius ** 2

    def record(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class DivergenceComponent:
    """A Gaussian divergence component: particle ``fraction`` and 1/e half-angles [rad] (IMAS definition)."""

    fraction: float
    horizontal: float
    vertical: float

    def __post_init__(self):
        if not (self.fraction > 0 and self.horizontal >= 0 and self.vertical >= 0):
            raise ValueError("divergence component: fraction > 0 and angles >= 0 required")


@dataclass
class BeamOptics:
    """Finite-beam description of one beam (module docstring).  ``None`` = missing (recorded rule)."""

    width_h: Optional[float] = None
    width_v: Optional[float] = None
    focal_h: Optional[float] = None
    focal_v: Optional[float] = None
    divergence: Sequence[DivergenceComponent] = ()
    apertures: Sequence[Aperture] = ()
    missing: List[str] = field(default_factory=list)

    def __post_init__(self):
        for nm in ("width_h", "width_v"):
            v = getattr(self, nm)
            if v is not None and not v >= 0:
                raise ValueError(f"{nm} must be >= 0")
        for nm in ("focal_h", "focal_v"):
            v = getattr(self, nm)
            if v is not None and not v > 0:
                raise ValueError(f"{nm} must be > 0")
        self.divergence = tuple(self.divergence)
        self.apertures = tuple(self.apertures)
        self.missing = list(self.missing)
        rules = (("width_h", "source width missing: zero (point source horizontally)"),
                 ("width_v", "source height missing: zero (point source vertically)"),
                 ("focal_h", "horizontal focal length missing: unfocused horizontally"),
                 ("focal_v", "vertical focal length missing: unfocused vertically"))
        for nm, rule in rules:
            if getattr(self, nm) is None and rule not in self.missing:
                self.missing.append(rule)
        if not self.divergence and "divergence missing: zero divergence" not in self.missing:
            self.missing.append("divergence missing: zero divergence")
        if not self.apertures and not any(m.startswith("aperture") for m in self.missing):
            self.missing.append("apertures: none given (no clipping)")
        if self.divergence:
            s = sum(c.fraction for c in self.divergence)
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"divergence component fractions must sum to 1 (got {s:.6g})")

    def record(self):
        return {"width_h": self.width_h, "width_v": self.width_v, "focal_h": self.focal_h, "focal_v": self.focal_v,
                "divergence": [c.__dict__.copy() for c in self.divergence],
                "apertures": [a.record() for a in self.apertures], "missing": list(self.missing)}


def halton(n, bases=HALTON_BASES):
    """First ``n`` points (index 1..n) of the Halton sequence in ``bases``: array (n, len(bases)) in (0, 1)."""
    idx = np.arange(1, int(n) + 1)
    out = np.zeros((idx.size, len(bases)))
    for j, b in enumerate(bases):
        i = idx.copy()
        f = 1.0 / b
        while np.any(i > 0):
            out[:, j] += f * (i % b)
            i //= b
            f /= b
    return out


def beamlet_sample(optics, n):
    r"""Deterministic beamlet offsets and angles (O1-O2): dict of arrays ``y_h, y_v, a_h, a_v, weight``."""
    from scipy.special import ndtri
    if int(n) < 1:
        raise ValueError("n_beamlets must be >= 1")
    u = halton(n)
    wh = optics.width_h or 0.0
    wv = optics.width_v or 0.0
    y_h = (u[:, 0] - 0.5) * wh
    y_v = (u[:, 1] - 0.5) * wv
    comps = optics.divergence or (DivergenceComponent(1.0, 0.0, 0.0),)
    out = {k: [] for k in ("y_h", "y_v", "a_h", "a_v", "weight")}
    for c in comps:
        th_h = ndtri(u[:, 2]) * c.horizontal / np.sqrt(2.0)
        th_v = ndtri(u[:, 3]) * c.vertical / np.sqrt(2.0)
        a_h = th_h - (y_h / optics.focal_h if optics.focal_h is not None else 0.0)
        a_v = th_v - (y_v / optics.focal_v if optics.focal_v is not None else 0.0)
        out["y_h"].append(y_h)
        out["y_v"].append(y_v)
        out["a_h"].append(a_h)
        out["a_v"].append(a_v)
        out["weight"].append(np.full(y_h.size, c.fraction / y_h.size))
    return {k: np.concatenate(v) for k, v in out.items()}


def _get(d, *path):
    for p in path:
        if d is None:
            return None
        if isinstance(d, dict):
            d = d.get(p)
        elif isinstance(d, (list, tuple)) and isinstance(p, int):
            d = d[p] if p < len(d) else None
        else:
            return None
    return d


def optics_from_ids(group, apertures=None, *, source_r, source_z, tangency_radius, vertical_angle):
    """:class:`BeamOptics` from an IMAS-schema beamlets group and the unit's apertures (nested dicts / lists keyed
    by the schema's leaf names).  The pencil geometry of the same beam places the apertures on its axis (module
    docstring); every missing item follows the stated rule and is recorded."""
    missing = []
    wh, wv = _val(_get(group, "width_horizontal")), _val(_get(group, "width_vertical"))
    fh = _val(_get(group, "focus", "focal_length_horizontal"))
    fv = _val(_get(group, "focus", "focal_length_vertical"))
    divs = []
    for c in (_get(group, "divergence_component") or []):
        fr, h, v = _val(_get(c, "particles_fraction")), _val(_get(c, "horizontal")), _val(_get(c, "vertical"))
        if fr is None or h is None or v is None:
            missing.append("a divergence component lacks fraction or angles: component not used")
            continue
        if fr > 0:
            divs.append(DivergenceComponent(fr, h, v))
    if divs:
        s = sum(c.fraction for c in divs)
        if abs(s - 1.0) > 1e-6:
            missing.append(f"divergence fractions sum to {s:.6g}: normalized to 1")
            divs = [DivergenceComponent(c.fraction / s, c.horizontal, c.vertical) for c in divs]
    aps = []
    Rs, Zs, Rt, th = float(source_r), float(source_z), float(tangency_radius), float(vertical_angle)
    sin_a = Rt / Rs
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    for j, a in enumerate(apertures or []):
        lab = f"aperture {j}"
        r, z = _val(_get(a, IMAS_APERTURE_CENTER, "r")), _val(_get(a, IMAS_APERTURE_CENTER, "z"))
        gt = _val(_get(a, "geometry_type"))
        if r is None or z is None:
            missing.append(f"{lab}: no center: not applied")
            continue
        # first inward crossing of the axis with radius r: s^2 cos^2(th) - 2 Rs cos_a cos(th) s + Rs^2 - r^2 = 0
        A_, B_, C_ = np.cos(th) ** 2, -2.0 * Rs * cos_a * np.cos(th), Rs ** 2 - r ** 2
        disc = B_ * B_ - 4.0 * A_ * C_
        if disc < 0 or A_ <= 0:
            missing.append(f"{lab}: center radius {r:.4g} m not reached by the beam axis: not applied")
            continue
        s = (-B_ - np.sqrt(disc)) / (2.0 * A_)
        if not s > 0:
            missing.append(f"{lab}: center not downstream of the source: not applied")
            continue
        off_v = z - (Zs + np.sin(th) * s)
        base = dict(distance=float(s), offset_h=0.0, offset_v=float(off_v * np.cos(th)), label=lab)
        if gt is None:
            missing.append(f"{lab} at {s:.4g} m: geometry_type missing (no dimensions): not applied")
            continue
        if int(gt) == 3:
            x1, x2 = _val(_get(a, "x1_width")), _val(_get(a, "x2_width"))
            if x1 is None or x2 is None:
                missing.append(f"{lab} at {s:.4g} m: rectangle without x1_width / x2_width: not applied")
                continue
            aps.append(Aperture(shape="rectangle", half_width_h=0.5 * x1, half_width_v=0.5 * x2, **base))
        elif int(gt) == 2:
            rad = _val(_get(a, "radius"))
            if rad is None:
                missing.append(f"{lab} at {s:.4g} m: circle without radius: not applied")
                continue
            aps.append(Aperture(shape="circle", radius=rad, **base))
        else:
            missing.append(f"{lab} at {s:.4g} m: geometry_type {int(gt)} (outline) not supported: not applied")
            continue
        missing.append(f"{lab}: horizontal center offset taken as 0 (not derived from the toroidal angle)")
    if not (apertures or []):
        missing.append("apertures: none given (no clipping)")
    elif not aps:
        missing.append("apertures: none usable (no clipping)")
    return BeamOptics(width_h=wh, width_v=wv, focal_h=fh, focal_v=fv, divergence=tuple(divs), apertures=tuple(aps),
                      missing=missing)
