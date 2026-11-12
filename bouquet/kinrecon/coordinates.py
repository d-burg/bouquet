r"""Radial coordinates of the current model: rho_tor,norm maps, solver knots and the q-consistency check.

Coordinates
-----------
Three radial labels appear in the reconstruction:

* ``psi_N`` -- normalized poloidal flux of the CURRENT equilibrium (0 on axis, 1 on the last closed
  surface); the coordinate in which TokaMaker takes its profiles (``jphi-linterp`` and ``linterp``
  knots);
* ``rho`` -- the standard normalized toroidal-flux radius ``rho_tor,norm = sqrt(Phi / Phi_LCFS)``
  of the same equilibrium (the working coordinate of the current model; transport codes and
  kinetic profile fits tabulate on it);
* ``rho_dom`` -- the toroidal-flux radius of the diffusion DOMAIN, normalized to the toroidal flux
  inside the domain edge ``psi_N = c`` (:mod:`.current.geometry`), ``rho_dom = sqrt(Phi / Phi(c))``.
  ``rho = rho_dom sqrt(Phi(c) / Phi_LCFS)`` inside the domain.

The toroidal flux comes from the geometry, never from q:

.. code-block:: text

    dPhi = F <1/R^2> dV / (2 pi)                                   (Lyons eq. 13)
    Phi(psi_N <= c) = Phi(c) rho_dom^2                             (domain geometry)
    Phi(psi_N > c)  = Phi(c) + int_c^psi_N F <1/R^2> / (2 pi) dV  (outer-layer volume tabulation)
    Phi_LCFS        = Phi(c) + int_c^1 F <1/R^2> / (2 pi) dV

The layer integral uses :class:`.current.geometry.OuterLayerGeometry` (trapezoid in enclosed volume
plus its logarithmic X-point tail); the integrand ``F <1/R^2>`` is regular through a separatrix
(the logarithmic divergence of q is carried by ``dV/dpsi``, which the volume tabulation absorbs),
so ``Phi_LCFS`` is finite on a diverted boundary.  :class:`RhoMap` holds the resulting
``psi_N <-> rho`` table of one equilibrium (PCHIP both ways in ``psi_N`` and ``rho^2``, the axis point
``(0, 0)`` included).

Knots
-----
The knots are SAMPLING points of the modeled ``<j_phi>`` and ``p`` handed to TokaMaker, never fit
parameters.  :func:`solver_knots` builds them for one pass from that pass's :class:`RhoMap`:

* ``"rho_tor_uniform"`` (default): ``n_knots`` points uniform in rho on [0, 1], mapped to psi_N
  through the current equilibrium (so they pack toward the axis, where ``psi_N ~ rho^2``, and
  toward the edge, where ``d psi_N / d rho -> 0``); they move between passes;
* ``"sqrt_psin_uniform"`` (tested fallback): uniform in ``sqrt(psi_N)`` on [0, c], fixed;
* ``"psin_uniform"`` (the previous default, kept by name): uniform in psi_N on [0, 1], fixed.

Every rule then applies the domain-edge rule: the domain edge ``psi_N = c`` is a knot, the outer
layer ``(c, 1)`` carries ``n_layer_knots`` interior knots uniform in psi_N, and ``psi_N = 1`` is
the last knot (the lumped outer layer's current is a profile on its own samples; a single knot
there leaves the steep edge current represented by one linear segment).  Knots of the core rule
that fall inside the layer, or closer than ``min_gap`` to the edge knot or to another knot, are
dropped (recorded).

q consistency
-------------
:func:`q_consistency` compares the solver's q(rho) with the diffusion model's q(rho) on the model
grid.  Criterion (a setting; :data:`Q_CHECK_DEFAULT`): ``max |q_eq - q_model| / q_model <= tol_rel``
(1e-2) over ``rho in [rho_min, rho_max]`` = [0.05, 0.95] by default.  ``rho_min=None`` uses the
innermost surface of the equilibrium table.  The pass-consistency loop of :mod:`.current_model`
requires it in addition to the profile-change criterion.

Axis.  Neither side's innermost values are usable as the axis q: the solver's traced q on the
innermost surfaces (rho below ~0.05) scatters by up to ~2 % around a smooth profile (contour
tracing of small surfaces), a trace at psi_N -> 0 is worse (up to ~27 % off), and the model's q on its
innermost faces deviates from a smooth profile by a similar amount (finite differences next to the
axis).  TokaMaker's ``get_stats()['q_0']`` is q at psi_N = 0.02 by its definition, not an axis
value.  The axis value of each side is therefore the regular-singular extrapolation of its own q
from surfaces where it is smooth: q is an even, regular function of rho at the magnetic axis, so

.. code-block:: text

    q(rho) = q0 + q2 rho^2 + q4 rho^4 + ...

is fitted by least squares (``axis_degree`` terms in rho^2 after q0) on ``rho in axis_window``
(default [0.05, 0.15], where both sides deviate from a smooth fit by ~1e-3, measured) and ``q0`` is
the axis value; the same construction is applied to the equilibrium and to the model.  The axis
test (``test_axis=True``) applies the SAME criterion as the interior, ``|q0_eq - q0_model| /
q0_model <= tol_rel``, and is then part of ``passed``.  By default the axis values and their
difference are computed and recorded on every check but NOT tested: on a peaked public test
equilibrium (q0 ~ 0.47, no sawteeth) the equilibrium's trace is rough inside the window (rms 0.4 %
of q0 about the fit) and the extrapolated axis values differ by 1.08 %, so whether the test measures
the profiles or the trace there is not settled; whether to test the axis is a decision for the
user.  Because the window starts where the interior range
starts, the axis test checks that the two profiles extrapolate to the same axis value; a
disagreement confined to rho < 0.05 is outside what either side resolves.  The fit's rms deviation
(relative to q0) is recorded for both sides (``axis``).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["RhoMap", "rho_map_from_geometry", "solver_knots", "KNOT_RULES", "KnotSettings", "QCheckSettings",
           "Q_CHECK_DEFAULT", "q_consistency", "axis_q"]

#: Knot rules understood by :func:`solver_knots`.
KNOT_RULES = ("rho_tor_uniform", "sqrt_psin_uniform", "psin_uniform")


@dataclass
class RhoMap:
    """``psi_N <-> rho_tor,norm`` table of one equilibrium (module docstring).

    ``psi_n`` and ``rho`` are strictly increasing from (0, 0) to (1, 1); ``psi_n_edge`` is the domain
    edge ``c`` and ``rho_edge`` its rho; ``phi_edge`` / ``phi_lcfs`` the toroidal fluxes [Wb] inside
    ``c`` and inside the last closed surface.  ``rho_dom = rho / rho_edge`` inside the domain.
    """
    psi_n: np.ndarray
    rho: np.ndarray
    psi_n_edge: float
    phi_edge: float
    phi_lcfs: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.psi_n = np.asarray(self.psi_n, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        if self.psi_n.shape != self.rho.shape or self.psi_n.size < 4:
            raise ValueError("psi_n and rho must be matching arrays of at least 4 points")
        if np.any(np.diff(self.psi_n) <= 0) or np.any(np.diff(self.rho) <= 0):
            raise ValueError("the psi_N <-> rho table must increase strictly in both coordinates")
        if abs(self.psi_n[0]) > 1e-14 or abs(self.rho[0]) > 1e-14 or abs(self.psi_n[-1] - 1.0) > 1e-12 \
                or abs(self.rho[-1] - 1.0) > 1e-12:
            raise ValueError("the table must run from (0, 0) to (1, 1)")
        if not (self.phi_lcfs >= self.phi_edge > 0):
            raise ValueError("need Phi_LCFS >= Phi(c) > 0")
        # interpolated in the toroidal-flux fraction s = rho^2, which is smooth in psi_N at the axis
        # (ds/dpsi_N = q / q_avg); rho itself goes like sqrt(psi_N) there and PCHIP in rho would not
        # represent the first interval
        s2 = self.rho ** 2
        self._fwd = PchipInterpolator(self.psi_n, s2, extrapolate=False)
        self._inv = PchipInterpolator(s2, self.psi_n, extrapolate=False)   # starting guess only

    @property
    def rho_edge(self):
        """rho_tor,norm of the domain edge, ``sqrt(Phi(c) / Phi_LCFS)``."""
        return float(np.sqrt(self.phi_edge / self.phi_lcfs))

    def rho_of_psin(self, x):
        """rho_tor,norm at psi_N = x (x in [0, 1]; outside raises)."""
        x = np.asarray(x, dtype=float)
        if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
            raise ValueError("psi_N outside [0, 1]")
        return np.sqrt(np.clip(self._fwd(np.clip(x, 0.0, 1.0)), 0.0, 1.0))

    def psin_of_rho(self, r):
        """psi_N at rho_tor,norm = r (r in [0, 1]; outside raises)."""
        r = np.asarray(r, dtype=float)
        if np.any(r < -1e-12) or np.any(r > 1.0 + 1e-12):
            raise ValueError("rho outside [0, 1]")
        # the exact inverse of rho_of_psin (bisection on the monotone forward interpolant, bracketed
        # around the inverse interpolant's guess), so the two maps round-trip to round-off
        t = np.clip(r, 0.0, 1.0) ** 2
        lo = np.zeros_like(t)
        hi = np.ones_like(t)
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f = self._fwd(mid)
            below = f < t
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out = 0.5 * (lo + hi)
        return np.where(t <= 0.0, 0.0, np.where(t >= 1.0, 1.0, out))

    def rho_dom_of_rho(self, r):
        """Domain radius ``rho / rho_edge`` (meaningful for rho <= rho_edge)."""
        return np.asarray(r, dtype=float) / self.rho_edge

    def as_dict(self):
        return dict(psi_n_edge=self.psi_n_edge, rho_edge=self.rho_edge, phi_edge=self.phi_edge,
                    phi_lcfs=self.phi_lcfs, n=int(self.psi_n.size), meta=dict(self.meta))


def rho_map_from_geometry(geom, layer):
    """:class:`RhoMap` of an equilibrium from its diffusion geometry and outer-layer tabulation.

    ``geom`` is the :class:`.current.geometry.FluxSurfaceGeometry` of :func:`.current.geometry.from_tokamaker`
    (``meta['psi_n_samples']``, ``meta['rho_samples']``, ``phi_b`` = Phi(c)) and ``layer`` its
    :class:`.current.geometry.OuterLayerGeometry` (module docstring).
    """
    xs = np.asarray(geom.meta["psi_n_samples"], dtype=float)
    rs = np.asarray(geom.meta["rho_samples"], dtype=float)
    c = float(geom.meta["psi_n_edge"])
    if abs(xs[-1] - c) > 1e-12:
        raise ValueError("the geometry's last sample must be the domain edge")
    phi_c = float(geom.phi_b)
    dens = np.asarray(layer.fpol, float) * np.asarray(layer.avg_inv_r2, float) / (2.0 * np.pi)
    cum = layer.cumulative(dens)
    phi_l = phi_c + layer.integral(dens)
    xl = np.asarray(layer.psi_n, dtype=float)
    if abs(xl[0] - c) > 1e-12:
        raise ValueError("the outer layer must start at the domain edge")
    rho_dom_scale = np.sqrt(phi_c / phi_l)
    x = np.concatenate(([0.0], xs, xl[1:], [1.0]))
    r = np.concatenate(([0.0], rs * rho_dom_scale, np.sqrt((phi_c + cum[1:]) / phi_l), [1.0]))
    if xl[-1] >= 1.0:            # a layer tabulated to psi_N = 1 itself: its last sample is the boundary
        x, r = x[:-1], r[:-1]
        x[-1], r[-1] = 1.0, 1.0
    meta = dict(method="Phi from F <1/R^2> dV/(2 pi): domain geometry + outer-layer volume tabulation",
                layer_tail=layer.meta.get("tail_method"), layer_flux_fraction=float(1.0 - phi_c / phi_l))
    return RhoMap(x, r, c, phi_c, phi_l, meta)


@dataclass
class KnotSettings:
    """How the solver knots are generated (module docstring).  All are recorded settings."""
    rule: str = "rho_tor_uniform"
    n_knots: int = 129                # core rule points (uniform in the rule's coordinate, 0 .. 1 or 0 .. c)
    n_layer_knots: int = 2            # interior knots in the outer layer (c, 1), uniform in psi_N
    min_gap: float = 1.0e-6           # psi_N: knots closer than this to another knot are dropped

    def __post_init__(self):
        if self.rule not in KNOT_RULES:
            raise ValueError(f"knot rule must be one of {KNOT_RULES}")
        if self.n_knots < 8:
            raise ValueError("n_knots must be >= 8")
        if self.n_layer_knots < 0:
            raise ValueError("n_layer_knots must be >= 0")

    def as_dict(self):
        return asdict(self)


def solver_knots(rho_map, settings=None):
    """psi_N knots for one forward solve (module docstring).  Returns ``(psi_n_knots, record)``.

    ``rho_map`` is the :class:`RhoMap` of the CURRENT equilibrium (needed for ``rho_tor_uniform``; for
    the fixed rules only its ``psi_n_edge`` is used).  The knots run strictly from exactly 0 to exactly 1.
    """
    s = settings or KnotSettings()
    c = float(rho_map.psi_n_edge)
    if s.rule == "rho_tor_uniform":
        core = rho_map.psin_of_rho(np.linspace(0.0, 1.0, int(s.n_knots)))
        core[0], core[-1] = 0.0, 1.0
    elif s.rule == "sqrt_psin_uniform":
        core = np.linspace(0.0, np.sqrt(c), int(s.n_knots)) ** 2
        core[-1] = c
    else:
        core = np.linspace(0.0, 1.0, int(s.n_knots))
    core = np.asarray(core, dtype=float)
    inner = core[core < c - s.min_gap]
    dropped_layer = int(np.count_nonzero((core > c + s.min_gap) & (core < 1.0 - s.min_gap)))
    layer = c + (1.0 - c) * np.arange(1, s.n_layer_knots + 1) / (s.n_layer_knots + 1)
    x = np.concatenate((inner, [c], layer, [1.0]))
    x = np.unique(x)
    gaps = np.diff(x)
    close = np.nonzero(gaps < s.min_gap)[0]
    dropped_close = 0
    if close.size:
        drop = set(int(i) + 1 for i in close if int(i) + 1 < x.size - 1)
        dropped_close = len(drop)
        x = np.array([v for i, v in enumerate(x) if i not in drop])
    if x[0] != 0.0 or x[-1] != 1.0 or np.any(np.diff(x) <= 0):
        raise RuntimeError("knot generation produced an invalid set")
    rho_k = rho_map.rho_of_psin(x)
    rec = dict(settings=s.as_dict(), n=int(x.size), psi_n_edge=c, n_layer=int(np.count_nonzero(x > c)),
               dropped_core_points_in_layer=dropped_layer, dropped_close_points=dropped_close,
               min_spacing_psin=float(np.min(np.diff(x))), rho_first_interior=float(rho_k[1]),
               max_spacing_rho=float(np.max(np.diff(rho_k))))
    return x, rec


@dataclass
class QCheckSettings:
    """q-consistency criterion (module docstring).  ``tol_rel`` is a decision (recorded).

    ``required=True``: the pass-consistency loop declares consistency only when the check passes
    (``status="q_inconsistent"`` otherwise, once the profile criterion is met)."""
    tol_rel: float = 1.0e-2
    rho_min: Optional[float] = 0.05   # None: the innermost surface of the equilibrium table
    rho_max: float = 0.95
    n_rho: int = 39
    required: bool = True
    test_axis: bool = False                   # True: the extrapolated axis values (module docstring) at tol_rel
    axis_window: Sequence[float] = (0.05, 0.15)
    axis_degree: int = 2                      # terms in rho^2 after q0

    def as_dict(self):
        return asdict(self)


Q_CHECK_DEFAULT = QCheckSettings()


def q_consistency(rho_eq, q_eq, rho_model, q_model, settings=None):
    """Compare q of the solved equilibrium with the diffusion model's q on the model grid.

    ``(rho_eq, q_eq)``: the solver's q on rho_tor,norm (positive values); ``(rho_model, q_model)``: the
    model's q on rho_tor,norm.  Both are interpolated (PCHIP) onto ``n_rho`` points uniform in
    ``[rho_min, rho_max]`` (inside both tables; ``rho_min=None``: the larger of the two tables' first
    radii).  Returns a dict with
    ``passed``, ``max_rel``, where it occurs, the axis values and their test (``axis``,
    :func:`axis_q`), and the criterion.
    """
    s = settings or Q_CHECK_DEFAULT
    re, qe = np.asarray(rho_eq, float), np.abs(np.asarray(q_eq, float))
    rm, qm = np.asarray(rho_model, float), np.abs(np.asarray(q_model, float))
    for a, b, what in ((re, qe, "equilibrium"), (rm, qm, "model")):
        ok = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(ok) < 4:
            return dict(passed=False, reason=f"{what} q table has fewer than 4 finite points",
                        criterion=s.as_dict())
    oke = np.isfinite(re) & np.isfinite(qe)
    okm = np.isfinite(rm) & np.isfinite(qm)
    re, qe, rm, qm = re[oke], qe[oke], rm[okm], qm[okm]
    lo = float(max(re[0], rm[0])) if s.rho_min is None else float(s.rho_min)
    hi = float(s.rho_max)
    if lo < max(re[0], rm[0]) - 1e-9 or hi > min(re[-1], rm[-1]) + 1e-9:
        return dict(passed=False, reason=f"q tables do not cover rho [{lo}, {hi}] "
                                         f"(eq [{re[0]:.3g}, {re[-1]:.3g}], model [{rm[0]:.3g}, {rm[-1]:.3g}])",
                    criterion=s.as_dict())
    r = np.linspace(lo, hi, int(s.n_rho))
    a = PchipInterpolator(re, qe)(r)
    b = PchipInterpolator(rm, qm)(r)
    rel = np.abs(a - b) / np.abs(b)
    i = int(np.argmax(rel))
    interior_ok = bool(rel[i] <= s.tol_rel)
    ax = dict(tested=bool(s.test_axis))
    try:
        ae = axis_q(re, qe, s.axis_window, s.axis_degree)
        am = axis_q(rm, qm, s.axis_window, s.axis_degree)
        arel = abs(ae["q0"] - am["q0"]) / abs(am["q0"])
        ax.update(q0_eq=ae["q0"], q0_model=am["q0"], rel=float(arel), passed=bool(arel <= s.tol_rel),
                  fit_rms_rel_eq=ae["rms_rel"], fit_rms_rel_model=am["rms_rel"], n_eq=ae["n"], n_model=am["n"],
                  window=list(s.axis_window), degree=int(s.axis_degree))
    except ValueError as exc:
        ax.update(passed=False, reason=str(exc))
    passed = interior_ok and (ax["passed"] or not s.test_axis)
    return dict(passed=passed, interior_passed=interior_ok, max_rel=float(rel[i]), rho_at_max=float(r[i]),
                q_eq_at_max=float(a[i]), q_model_at_max=float(b[i]), q0_eq=float(a[0]), q0_model=float(b[0]),
                axis=ax, criterion=s.as_dict(), rel_profile=dict(rho=r.tolist(), rel=rel.tolist()))


def axis_q(rho, q, window=(0.05, 0.15), degree=2):
    """Axis value of a q profile by regular-singular extrapolation (module docstring): least squares
    ``q = q0 + sum_{j=1..degree} c_j rho^(2j)`` on ``rho in window``.  Returns ``dict(q0, rms_rel, n)``;
    fewer than ``degree + 3`` points in the window raises ``ValueError``."""
    r, qq = np.asarray(rho, float), np.abs(np.asarray(q, float))
    m = np.isfinite(r) & np.isfinite(qq) & (r >= float(window[0]) - 1e-12) & (r <= float(window[1]) + 1e-12)
    n = int(np.count_nonzero(m))
    if n < int(degree) + 3:
        raise ValueError(f"axis q: {n} points in rho {list(window)} (need >= {int(degree) + 3})")
    V = np.vander(r[m] ** 2, int(degree) + 1, increasing=True)
    c, *_ = np.linalg.lstsq(V, qq[m], rcond=None)
    res = qq[m] - V @ c
    return dict(q0=float(c[0]), rms_rel=float(np.sqrt(np.mean(res ** 2)) / abs(c[0])), n=n)
