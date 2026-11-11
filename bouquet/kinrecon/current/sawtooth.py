r"""Sawtooth model for the 1-D poloidal-flux state: helical-flux reconnection.

This module acts on the state of :mod:`.diffusion` (``psi`` on a fixed
``rho = rho_tor_norm`` finite-volume grid, positive frame) and uses only its
public interface (:class:`~.diffusion.CurrentDiffusion`,
:class:`~.diffusion.CurrentState`, the ``post_step`` hook of
:meth:`~.diffusion.CurrentDiffusion.advance`).

Coordinates
-----------
With ``s = rho^2 = Phi/Phi_b`` (toroidal flux fraction), ``c = Phi_b / (2 pi)``
and ``psi`` the poloidal flux per radian, the safety factor of the diffusion
module, ``q = Phi_b rho / (pi dpsi/drho)``, becomes

.. code-block:: text

    dpsi/ds = c iota,   iota = 1/q                                                    (S1)

and the helical flux with respect to the q = 1 resonance is

.. code-block:: text

    chi(s) = psi(s) - c s,      dchi/ds = c (iota - 1)                                (S2)

(B. B. Kadomtsev, "Disruptive instability in tokamaks", Sov. J. Plasma Phys.
1, 389 (1975); the same function, "poloidal flux minus the poloidal flux if
q = 1 everywhere", defines the NTCC module KDSAW, D. McCune, PPPL,
https://w3.pppl.gov/NTCC/KDSAW/DOC).  ``chi`` rises where q < 1 and falls
where q > 1; its local maxima/minima are the q = 1 surfaces.

Reconnection operator (full, f = 1)
-----------------------------------
Kadomtsev pairing: each helical-flux level ``L`` between ``chi(0)`` and the
maximum ``chi(s_1)`` is taken by one surface ``s_-(L) < s_1`` and one surface
``s_+(L) > s_1``; they reconnect into one surface enclosing the toroidal flux
between them,

.. code-block:: text

    s_new(L) = s_+(L) - s_-(L),   chi_new = L,   chi(s_mix) = chi(0)                  (K1)

(Kadomtsev 1975; KDSAW: paired surfaces "are mapped to regions of equal
toroidal flux", mixing boundary ``Psi_1(x_mix) = Psi_1(0)``).  Outside
``s_mix`` nothing changes.  (K1) says that the toroidal-flux measure of every
super-level set ``{chi > L}`` inside ``[0, s_mix]`` is preserved and that
``chi_new`` is non-increasing, i.e. ``chi_new`` is the *decreasing
rearrangement* of ``chi`` on ``[0, s_mix]``:

.. code-block:: text

    mu(L) = | { s in D : chi(s) > L } |,      chi_new(s) = mu^{-1}(s - s_lo)          (K2)

(rearrangement: E. H. Lieb and M. Loss, *Analysis*, 2nd ed., AMS (2001),
ch. 3; non-expansivity ``||f* - g*|| <= ||f - g||`` in every L^p norm, their
Theorem 3.5).  For a unimodal ``chi`` (one q = 1 surface, q0 < 1) (K2) is
identical to (K1): ``mu(L) = s_+(L) - s_-(L)``.

Generalization to non-monotonic q (this module's construction, not taken
from the literature): (K2) is applied as written to any ``chi``, on the domain
``D = [s_lo, s_mix]`` defined by the innermost and outermost q = 1 surfaces:

* ``s_*`` = outermost local maximum of ``chi`` (outermost q = 1 surface with
  q < 1 inside it); ``m_* = min chi`` over ``[0, s_*]`` (the axis value, or
  the innermost local minimum = innermost q = 1 surface when q0 > 1);
* ``s_mix`` = the root of ``chi = m_*`` beyond ``s_*`` (as in (K1));
* ``s_lo`` = 0, except when q0 > 1 and the axis value exceeds every value
  that can be reconnected: then the decreasing axis segment above
  ``M = max chi`` over ``[s_a, s_mix]`` is left untouched and ``s_lo`` is the
  root of ``chi = M`` on it.

Properties: identity when q >= 1 everywhere (no local maximum); equal to
(K1) in the monotonic case; continuous in the state through the change of
topology (q0 crossing 1 with an off-axis q < 1 region, a second q < 1 region
appearing elsewhere), because the rearrangement is non-expansive and
``m_*``, ``s_mix``, ``s_lo`` are continuous; ``q_new >= 1`` on ``D``.
A q-floor reset (as OFT's ``saw_reset_1d``) is a different operator;
switching to it when q becomes non-monotonic would be discontinuous, which
is why the rearrangement is used for every topology.

Current sheets.  ``chi_new`` is continuous but its slope jumps where ``D``
ends and at the new position of every interior critical level.  With
``S(L) = sum over crossings chi(s) = L of 1/|chi'(s)|`` (``mu' = -S``), the
one-sided slopes are ``chi_new' = -1/S`` (0 where a crossing is a q = 1
surface).  At ``s_mix``, in the monotonic case,

.. code-block:: text

    J_s = chi'(s_mix+) - chi_new'(s_mix-) = -c a^2 / (b - a),
    a = iota(s_mix) - 1 < 0,  b = iota(0) - 1 > 0                                     (K3)

(derived from (K1)-(K2); post-crash ``1/q_new - 1 = a b/(b - a)``, which
agrees with KDSAW's statement that Kadomtsev "produces one current sheet at
x_mix").  ``J_s < 0`` is a reverse (counter-current) sheet.

Sheet smoothing (TRANSP ``XSWIDQ``: A. Y. Pankin et al., "TRANSP integrated
modeling code for interpretive and predictive analysis of tokamak plasmas",
Comput. Phys. Commun. 312, 109611 (2025), sec. VIII.A, default 0.05 in
``sqrt(Phi_N)``).  Each slope jump ``J_rho = 2 rho_k J_s`` of ``psi`` at
``rho_k`` is spread with

.. code-block:: text

    psi_K,s = psi_K + J_rho G(rho - rho_k; w_eff),
    G(x; w) = int_{-w}^{x} (S_w(x') - H(x')) dx',   S_w = 6u^5 - 15u^4 + 10u^3,
    u = (x/w + 1)/2,   w_eff = w rho_k / sqrt(rho_k^2 + w^2)                          (K4)

(C2 smoothstep: K. Perlin, "Improving noise", ACM Trans. Graph. 21, 681
(2002)).  ``G = 0`` for ``|x| >= w`` (``S_w`` is antisymmetric about 1/2), so
``psi`` and the enclosed current are unchanged outside ``|rho - rho_k| < w_eff``;
the slope jump is cancelled exactly, and the operator is C1 in the data when
a cell center crosses ``rho_k`` (the derivative jump of ``psi_K`` there is
``J_rho d rho_k``, cancelled by ``-J_rho G'`` jumping by ``J_rho``).
``w_eff`` keeps the zone at ``rho > 0`` near threshold.

Partial reconnection
--------------------
.. code-block:: text

    psi_post = psi + f (psi_K,s - psi)     at fixed rho (fixed toroidal flux)       (K5)

(TRANSP ``XSWID2 = 1 - f``, Pankin et al. 2025 sec. VIII.A, "a model that
'averages'" no and full current mixing).  Conserves Phi (fixed grid), psi and
the enclosed current outside the changed zone, psi at the boundary and Ip;
it does **not** conserve helical flux for 0 < f < 1 (phenomenological).

Smoothness.  As ``q0 -> 1-`` (monotonic case) ``s_mix ~ (1 - q0)``, the change
of ``iota`` is O(1 - q0) over a measure O(1 - q0), so ``|psi_post - psi| =
O((1 - q0)^2)``: the operator meets the identity with zero slope and has no
switch.  ``q0`` itself (a derivative functional) is only C0 there:
``iota_post(0) = (1 - f) iota(0) + f``.

Periodic state ("single-slice mode")
------------------------------------
Fixed point of the Poincare map ``z -> G K_f(D_tau(z))``: diffusion over one
period ``tau`` (``nsteps`` equal steps of the solver), then reconnection; ``G``
removes the offset of the last cell (the flux grows uniformly).  ``D_tau`` is
affine for frozen inputs; it is built from the solver's own ``step`` and
checked against ``advance``.  Solved by Newton's method with a Jacobian
``K'(y) M`` from central differences along the dominant left singular
vectors of ``M`` restricted to the cells the operator reads (smooth
directions) and backtracking.  Where Newton's linear model fails -- the
fixed point can sit next to the slope break of the map at q0 = 1, e.g. for
fast, full reconnection -- the iteration continues with Anderson-accelerated
cycles of the physical map (H. F. Walker and P. Ni, "Anderson acceleration
for fixed-point iterations", SIAM J. Numer. Anal. 49, 1715 (2011)).
Convergence measure (decided criterion 1e-9, never loosened):

.. code-block:: text

    r = || G K_f(M z + c) - z ||_inf / ( psi(rho_mix) - psi(0) )                       (P1)

for the pre-crash state ``y = M z + c`` (``psi(rho_mix) - psi(0)`` from the
same interpolant as the operator; when the operator is inactive the
denominator is ``psi(last cell) - psi(0)``).  Both the matrix form and the
real stepping (``advance`` then ``reconnect``) must meet it.

Time-dependent mode
-------------------
Crash times from data are applied by :class:`CrashSchedule` through
``advance(..., post_step=...)``; :func:`advance_with_crashes` splits the
interval so that a step ends exactly on every crash time.  No trigger model is
used anywhere in the fit path; :func:`q1_shear_diagnostic` is a separate,
optional report.

Parameterization
----------------
Two parameters, ``f`` and the mean crash rate ``nu``.  The combination
``Lambda = -ln(1 - f_cap f) nu`` (``f_cap = 0.999`` keeps f = 1 finite) is
what near-axis MSE constrains; :func:`move_along_strength` and
:func:`move_across_strength` move along and across it.

Window averages
---------------
Diagnostics are averaged, not the current: piecewise-linear interpolation in
time between stored states (trapezoid rule), with the pre- and post-crash
states stored at the same time so a crash is a jump; exact integration over
any window.
"""

import math
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .diffusion import CurrentDiffusion, CurrentState

__all__ = [
    "CYCLE_TOL",
    "DEFAULT_SHEET_WIDTH",
    "DEFAULT_HISTORY",
    "F_CAP",
    "ReconnectionRecord",
    "reconnect",
    "reconnect_psi",
    "PeriodicOrbit",
    "solve_periodic_orbit",
    "CrashEvent",
    "Trajectory",
    "CrashSchedule",
    "advance_with_crashes",
    "SliceSchedule",
    "schedule_for_slice",
    "simulate_slice",
    "crash_strength",
    "rate_from_strength",
    "to_strength_coordinates",
    "from_strength_coordinates",
    "move_along_strength",
    "move_across_strength",
    "q1_shear_diagnostic",
]

CYCLE_TOL = 1e-9            # decided periodic-orbit criterion (P1); never loosened
DEFAULT_SHEET_WIDTH = 0.05  # TRANSP XSWIDQ default, in rho = sqrt(Phi_N)
DEFAULT_HISTORY = 1.5       # s of crash history before a window (core memory 0.5-1 s measured)
F_CAP = 0.999               # cap in -ln(1 - F_CAP f) so that f = 1 is finite
N_SAMPLES = 1024            # samples per monotone piece (fixed relative positions)
N_LEVELS = 2048             # helical-flux levels of the rearranged profile (fixed relative positions)
_HALF_PI = 0.5 * np.pi


# ====================================================================== interpolant of chi(s)
def _fd_slopes(x, y):
    """Three-point (non-uniform) derivative estimates at the nodes; one-sided at the ends.  Linear in ``y``.

    Standard second-order Lagrange differences (e.g. B. Fornberg, Math. Comp. 51, 699 (1988)).
    """
    n = x.size
    m = np.empty(n)
    h0 = x[1:-1] - x[:-2]
    h1 = x[2:] - x[1:-1]
    m[1:-1] = (-h1 / (h0 * (h0 + h1))) * y[:-2] + ((h1 - h0) / (h0 * h1)) * y[1:-1] + (h0 / (h1 * (h0 + h1))) * y[2:]
    a, b = x[1] - x[0], x[2] - x[0]
    m[0] = -(a + b) / (a * b) * y[0] + b / (a * (b - a)) * y[1] - a / (b * (b - a)) * y[2]
    a, b = x[-2] - x[-1], x[-3] - x[-1]
    m[-1] = -(a + b) / (a * b) * y[-1] + b / (a * (b - a)) * y[-2] - a / (b * (b - a)) * y[-3]
    return m


def _axis_weights(s3):
    """Lagrange weights of the quadratic through (s3[k], psi_k) evaluated at s = 0 (psi is smooth in s = rho^2)."""
    s0, s1, s2 = s3
    return np.array([s1 * s2 / ((s0 - s1) * (s0 - s2)), s0 * s2 / ((s1 - s0) * (s1 - s2)),
                     s0 * s1 / ((s2 - s0) * (s2 - s1))])


class _Hermite:
    """C1 piecewise cubic Hermite interpolant with given node slopes (linear in the data).

    Basis: h00 = 2t^3 - 3t^2 + 1, h10 = t^3 - 2t^2 + t, h01 = -2t^3 + 3t^2, h11 = t^3 - t^2
    (standard cubic Hermite form, e.g. C. de Boor, *A Practical Guide to Splines*, Springer (1978)).
    """

    def __init__(self, x, y, m):
        self.x, self.y, self.m = x, y, m
        self.h = np.diff(x)

    def _seg(self, xe):
        i = np.clip(np.searchsorted(self.x, xe, side="right") - 1, 0, self.x.size - 2)
        return i, (xe - self.x[i]) / self.h[i]

    def __call__(self, xe):
        xe = np.asarray(xe, dtype=float)
        i, t = self._seg(xe)
        h = self.h[i]
        t2 = t * t
        t3 = t2 * t
        return ((2 * t3 - 3 * t2 + 1) * self.y[i] + (t3 - 2 * t2 + t) * self.m[i] * h
                + (-2 * t3 + 3 * t2) * self.y[i + 1] + (t3 - t2) * self.m[i + 1] * h)

    def deriv(self, xe):
        xe = np.asarray(xe, dtype=float)
        i, t = self._seg(xe)
        h = self.h[i]
        t2 = t * t
        return ((6 * t2 - 6 * t) * self.y[i] + (3 * t2 - 4 * t + 1) * self.m[i] * h
                + (-6 * t2 + 6 * t) * self.y[i + 1] + (3 * t2 - 2 * t) * self.m[i + 1] * h) / h

    def deriv2(self, xe):
        xe = np.asarray(xe, dtype=float)
        i, t = self._seg(xe)
        h = self.h[i]
        return ((12 * t - 6) * self.y[i] + (6 * t - 4) * self.m[i] * h
                + (-12 * t + 6) * self.y[i + 1] + (6 * t - 2) * self.m[i + 1] * h) / h ** 2

    def critical_points(self):
        """Sign changes of the derivative (simple roots of the per-segment quadratic, and exact node zeros).

        Returns positions (sorted) and kinds (+1 local maximum, -1 local minimum).  Tangencies (double roots)
        are not sign changes and are not returned.  Positions are continuous in the data.
        """
        y0, y1 = self.y[:-1], self.y[1:]
        m0, m1 = self.m[:-1] * self.h, self.m[1:] * self.h
        A = 6 * y0 + 3 * m0 - 6 * y1 + 3 * m1
        B = -6 * y0 - 4 * m0 + 6 * y1 - 2 * m1
        C = m0
        pos, kind = [], []
        scale = np.abs(B) + np.abs(C)
        lin = np.abs(A) <= 1e-12 * scale
        # linear derivative
        il = np.nonzero(lin & (B != 0))[0]
        t = -C[il] / B[il]
        ok = (t > 0) & (t < 1)
        for k, tt in zip(il[ok], t[ok]):
            pos.append(self.x[k] + tt * self.h[k])
            kind.append(1 if B[k] < 0 else -1)
        # quadratic derivative
        iq = np.nonzero(~lin)[0]
        Aq, Bq, Cq = A[iq], B[iq], C[iq]
        disc = Bq * Bq - 4 * Aq * Cq
        has = disc > 0
        iq, Aq, Bq, Cq, disc = iq[has], Aq[has], Bq[has], Cq[has], disc[has]
        sq = np.sqrt(disc)
        qq = -0.5 * (Bq + np.where(Bq >= 0, sq, -sq))
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = qq / Aq
            r2 = np.where(qq != 0, Cq / qq, np.nan)
        for r in (r1, r2):
            ok = (r > 0) & (r < 1)
            for k, a_, b_, tt in zip(iq[ok], Aq[ok], Bq[ok], r[ok]):
                slope = 2 * a_ * tt + b_
                if slope != 0:
                    pos.append(self.x[k] + tt * self.h[k])
                    kind.append(1 if slope < 0 else -1)
        # exact zeros of the node slope at interior nodes
        for i in np.nonzero(self.m[1:-1] == 0)[0] + 1:
            left = -np.sign(2 * A[i - 1] + B[i - 1])    # sign of chi' just left of node i
            right = np.sign(B[i])                        # sign of chi' just right of node i
            if left * right < 0:
                pos.append(self.x[i])
                kind.append(1 if left > 0 else -1)
        if not pos:
            return np.zeros(0), np.zeros(0, dtype=int)
        pos = np.asarray(pos)
        kind = np.asarray(kind, dtype=int)
        o = np.argsort(pos, kind="stable")
        return pos[o], kind[o]


def _smoothstep_G(x, w):
    """G(x; w) of (K4): integral of (C2 smoothstep - Heaviside); zero for |x| >= w."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    if w <= 0:
        return out
    inz = np.abs(x) < w
    u = (x[inz] / w + 1.0) / 2.0
    out[inz] = 2.0 * w * (u ** 6 - 3 * u ** 5 + 2.5 * u ** 4) - np.where(x[inz] > 0, x[inz], 0.0)
    return out


def _w_eff(rho_k, w):
    """Smooth cap of the sheet half-width, w rho/sqrt(rho^2 + w^2) (K4)."""
    return 0.0 if w <= 0 else w * rho_k / math.sqrt(rho_k * rho_k + w * w)


def _strictly_increasing(x, y):
    """Drop samples whose abscissa does not exceed every earlier one (round-off ties); keep both end samples."""
    prev = np.maximum.accumulate(np.concatenate(([-np.inf], x[:-1])))
    keep = x > prev
    keep[0] = True
    if not keep[-1]:
        last = np.nonzero(keep)[0][-1]
        keep[last] = last == 0
        keep[-1] = True
    return x[keep], y[keep]


# ====================================================================== monotone pieces and their inverses
class _Piece:
    """A monotone piece [a, b] of chi with a smooth inverse in the chart theta = arcsin(sqrt((L - lo)/(hi - lo))).

    Near a critical end (q = 1 surface) chi - L_c ~ (s - s_c)^2, so s(theta) is regular there; near a
    regular end s(theta) has zero slope.  Samples sit at fixed relative positions in s.
    """

    def __init__(self, H, a, b, la, lb, crit_a, crit_b, n_samples):
        self.a, self.b, self.la, self.lb = a, b, la, lb
        self.crit_a, self.crit_b = crit_a, crit_b
        self.inc = lb > la
        self.lo, self.hi = (la, lb) if self.inc else (lb, la)
        self.width = b - a
        self.span = self.hi - self.lo
        self.flat = not (self.span > 0 and self.width > 0)
        if self.flat:
            return
        v = np.linspace(0.0, 1.0, n_samples)
        s = a + self.width * v
        L = H(s)
        L[0], L[-1] = la, lb
        th = self.theta(L)
        if not self.inc:
            th, s = th[::-1], s[::-1]
        th, s = _strictly_increasing(th, s)
        self._inv = PchipInterpolator(th, s, extrapolate=False)

    def theta(self, L):
        u = np.clip((np.asarray(L, dtype=float) - self.lo) / self.span, 0.0, 1.0)
        return np.arcsin(np.sqrt(u))

    def crossing(self, L):
        """s in the piece with chi(s) = L (L clipped to the piece's range)."""
        return self._inv(self.theta(L))

    def measure(self, L):
        """|{s in piece : chi(s) > L}|."""
        L = np.asarray(L, dtype=float)
        if self.flat:
            return np.where(L < self.lo, self.width, 0.0)
        sc = self.crossing(L)
        return (self.b - sc) if self.inc else (sc - self.a)


# ====================================================================== record
@dataclass
class ReconnectionRecord:
    """What one application of the operator did.  Radii are rho = rho_tor_norm; q from the operator's interpolant.

    ``topology``: ``'none'`` (q >= 1 everywhere), ``'monotonic'`` (q0 < 1, one q = 1 surface),
    ``'reversed'`` (q0 >= 1 with an off-axis q < 1 region), ``'multiple'`` (more than one q < 1 region).
    Conservation residuals (expected exactly 0): ``psi_outside_max_change`` (cells outside the changed zone),
    ``current_outside_max_rel_change`` (relative change of dpsi/drho, i.e. of the enclosed current, on faces
    between unchanged cells), ``psi_boundary_change``.  Toroidal flux is conserved by construction (fixed grid).
    """

    active: bool = False
    topology: str = "none"
    f: float = 1.0
    sheet_width: float = DEFAULT_SHEET_WIDTH
    time: Optional[float] = None
    rho_q1: tuple = ()
    rho_q1_outer: float = 0.0
    rho_q1_inner: float = 0.0
    rho_mix: float = 0.0
    rho_lo: float = 0.0
    zone_inner: float = 0.0
    zone_outer: float = 0.0
    q0_pre: float = float("nan")
    q0_post: float = float("nan")
    dpsi_max: float = 0.0
    dpsi_axis: float = 0.0
    depth: float = float("nan")
    n_changed_cells: int = 0
    sheets: list = field(default_factory=list)
    psi_outside_max_change: float = 0.0
    current_outside_max_rel_change: float = 0.0
    psi_boundary_change: float = 0.0

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ====================================================================== the operator
def _core(psi, rc, cfac, f, w, n_samples, n_levels):
    """Reconnection (K1)-(K5) on cell values.  Returns (psi_post, info dict)."""
    s_cells = rc * rc
    lw = _axis_weights(s_cells[:3])
    psi_axis = float(lw @ psi[:3])
    s_nodes = np.concatenate(([0.0], s_cells))
    chi = np.concatenate(([psi_axis], psi)) - cfac * s_nodes
    m = _fd_slopes(s_nodes, chi)
    H = _Hermite(s_nodes, chi, m)
    crit_s, crit_k = H.critical_points()
    info = dict(active=False, topology="none", psi_axis=psi_axis, chi_axis_slope=float(m[0]),
                q0_pre=float(cfac / (cfac + m[0])) if cfac + m[0] != 0 else float("inf"),
                rho_q1=tuple(np.sqrt(crit_s).tolist()), depth=float(psi[-1] - psi_axis))
    maxima = crit_s[crit_k > 0]
    if maxima.size == 0:
        if m[0] > 0:
            raise ValueError("q < 1 from the axis out to the last cell: no mixing region exists")
        return psi.copy(), info
    s_star = float(maxima[-1])
    chi_max = H(maxima)
    i_top = int(np.argmax(chi_max))
    s_top, M = float(maxima[i_top]), float(chi_max[i_top])
    minima = crit_s[(crit_k < 0) & (crit_s < s_star)]
    cand = np.concatenate(([chi[0]], H(minima)))
    j = int(np.argmin(cand))
    m_star = float(cand[j])
    min_at_axis = j == 0
    s_min = 0.0 if min_at_axis else float(minima[j - 1])
    # outer edge of the mixing region: chi decreasing beyond s_star, root of chi = m_star
    idx = np.nonzero((s_nodes > s_star) & (chi <= m_star))[0]
    if idx.size == 0:
        raise ValueError("the mixing region reaches the last cell")
    k = int(idx[0])
    lo_b = max(float(s_nodes[k - 1]), s_star)
    if chi[k] == m_star:
        s_mix = float(s_nodes[k])
    else:
        s_mix = brentq(lambda x: float(H(x)) - m_star, lo_b, float(s_nodes[k]), xtol=1e-16, rtol=1e-15,
                       maxiter=200)
    # inner edge: untouched decreasing axis segment when q0 > 1 and chi(0) exceeds everything reconnected
    s_lo = 0.0
    if m[0] < 0 and M < chi[0]:
        c1 = float(crit_s[0])
        s_lo = brentq(lambda x: float(H(x)) - M, 0.0, c1, xtol=1e-16, rtol=1e-15, maxiter=200)
    n_max = int(maxima.size)
    topology = "monotonic" if (m[0] > 0 and n_max == 1) else ("reversed" if n_max == 1 else "multiple")
    # monotone pieces of chi on D = [s_lo, s_mix]
    inner = (crit_s > s_lo) & (crit_s < s_mix)
    cs, ck = crit_s[inner], crit_k[inner]
    cl = H(cs)
    bps = np.concatenate(([s_lo], cs, [s_mix]))
    lev = np.concatenate(([M if s_lo > 0 else chi[0]], cl, [m_star]))
    is_crit = np.concatenate(([False], np.ones(cs.size, dtype=bool), [False]))
    pieces = [_Piece(H, bps[i], bps[i + 1], lev[i], lev[i + 1], is_crit[i], is_crit[i + 1], n_samples)
              for i in range(bps.size - 1)]

    def mu(L):
        tot = np.zeros(np.shape(L))
        for p in pieces:
            tot = tot + p.measure(L)
        return tot

    # rearranged profile on D, levels at fixed relative positions in a chart regular at the top level
    th = np.linspace(_HALF_PI, 0.0, n_levels)
    Lg = m_star + (M - m_star) * np.sin(th) ** 2
    sg = s_lo + mu(Lg)
    sg[0], sg[-1] = s_lo, s_mix
    Lg[0], Lg[-1] = M, m_star
    sg, Lg = _strictly_increasing(sg, Lg)
    P = PchipInterpolator(sg, Lg, extrapolate=False)
    sel = (s_cells > s_lo) & (s_cells < s_mix)
    psi_K = psi.copy()
    psi_K[sel] = P(s_cells[sel]) + cfac * s_cells[sel]
    # current sheets (slope jumps of chi_new), positions in s and jumps in d/ds
    sheets = []
    inv = (1.0 / m[0] if m[0] > 0 else np.inf) if min_at_axis else np.inf
    d_mix = float(H.deriv(s_mix))
    inv = inv + 1.0 / abs(d_mix) if d_mix != 0 else np.inf
    slope_left = -1.0 / inv
    sheets.append((s_mix, d_mix - slope_left))
    if s_lo > 0:
        sheets.append((s_lo, -float(H.deriv(s_lo))))
    # interior critical levels (all at once): position mu(L_p), jump +-1/S_reg from the regular crossings
    idx_int = [i for i in range(1, bps.size - 1)
               if not (bps[i] == s_top or ((not min_at_axis) and bps[i] == s_min))]
    if idx_int:
        ii = np.array(idx_int)
        Lp = lev[ii]
        S_reg = np.zeros(ii.size)
        for jj, pc in enumerate(pieces):
            if pc.flat:
                continue
            use = (pc.lo < Lp) & (Lp < pc.hi) & (ii != jj) & (ii - 1 != jj)
            if np.any(use):
                d = np.abs(H.deriv(pc.crossing(Lp[use])))
                S_reg[use] += np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        pos = s_lo + mu(Lp)
        for k, i in enumerate(ii):
            if S_reg[k] > 0:
                sheets.append((float(pos[k]), (1.0 if ck[i - 1] > 0 else -1.0) / S_reg[k]))
    sheet_out = []
    psi_Ks = psi_K
    for s_k, J_s in sheets:
        rho_k = math.sqrt(s_k)
        J_rho = 2.0 * rho_k * J_s
        we = _w_eff(rho_k, w)
        if we > 0:
            psi_Ks = psi_Ks + J_rho * _smoothstep_G(rc - rho_k, we)
        sheet_out.append(dict(rho=rho_k, jump_dpsi_drho=J_rho, half_width=we))
    rho_mix, rho_lo = math.sqrt(s_mix), math.sqrt(s_lo)
    zone_outer = rho_mix + _w_eff(rho_mix, w)
    zone_inner = (rho_lo - _w_eff(rho_lo, w)) if s_lo > 0 else 0.0
    if zone_outer >= rc[-2]:
        raise ValueError("the reconnection zone reaches the last two cells")
    post = psi + f * (psi_Ks - psi)
    info.update(active=True, topology=topology, rho_q1_outer=math.sqrt(s_star), rho_q1_inner=math.sqrt(crit_s[0]),
                rho_mix=rho_mix, rho_lo=rho_lo, zone_inner=zone_inner, zone_outer=zone_outer, sheets=sheet_out,
                depth=float(m_star + cfac * s_mix - chi[0]), s_mix=s_mix, s_lo=s_lo)
    return post, info


def _check_f(f, sheet_width):
    f = float(f)
    if not 0.0 <= f <= 1.0:
        raise ValueError("reconnection fraction f must lie in [0, 1]")
    if sheet_width < 0:
        raise ValueError("sheet_width must be >= 0")
    return f


def reconnect_psi(psi, rho_faces, phi_b, f=1.0, sheet_width=DEFAULT_SHEET_WIDTH, n_samples=N_SAMPLES,
                  n_levels=N_LEVELS, record=True):
    r"""Apply the reconnection operator (K1)-(K5) to cell values ``psi`` [Wb/rad].

    ``rho_faces`` are the diffusion grid's faces, ``phi_b`` the boundary toroidal flux [Wb].  The operator is
    metric-free (it needs only psi(rho) and Phi_b).  Returns ``(psi_post, ReconnectionRecord or None)``.
    The boundary value of psi is not needed: the two outermost cells are never changed (an error is raised
    if the zone reaches them).
    """
    f = _check_f(f, sheet_width)
    psi = np.asarray(psi, dtype=float)
    rf = np.asarray(rho_faces, dtype=float)
    rc = 0.5 * (rf[1:] + rf[:-1])
    if psi.size != rc.size or psi.size < 6:
        raise ValueError("psi must have one value per cell (at least 6 cells)")
    if not np.all(np.isfinite(psi)):
        raise ValueError("psi contains non-finite values")
    cfac = float(phi_b) / (2.0 * np.pi)
    post, info = _core(psi, rc, cfac, f, float(sheet_width), int(n_samples), int(n_levels))
    if not record:
        return post, None
    rec = ReconnectionRecord(f=f, sheet_width=float(sheet_width), **{k: info[k] for k in (
        "active", "topology", "rho_q1", "q0_pre", "depth")})
    if info["active"]:
        for k in ("rho_q1_outer", "rho_q1_inner", "rho_mix", "rho_lo", "zone_inner", "zone_outer", "sheets"):
            setattr(rec, k, info[k])
    d = post - psi
    rec.dpsi_max = float(np.max(np.abs(d)))
    lw = _axis_weights((rc[:3]) ** 2)
    rec.dpsi_axis = float(lw @ d[:3])
    changed = d != 0
    rec.n_changed_cells = int(np.count_nonzero(changed))
    outside = (rc >= rec.zone_outer) | (rc <= rec.zone_inner) if info["active"] else np.ones(rc.size, bool)
    rec.psi_outside_max_change = float(np.max(np.abs(d[outside]))) if np.any(outside) else 0.0
    # enclosed current on faces between two unchanged cells: I = F k dpsi/drho, so dI/I = d(dpsi)/dpsi
    gp = np.diff(psi) / np.diff(rc)
    gq = np.diff(post) / np.diff(rc)
    both = outside[1:] & outside[:-1]
    rec.current_outside_max_rel_change = float(np.max(np.abs(gq[both] - gp[both]) / np.abs(gp[both]))) \
        if np.any(both) else 0.0
    # q0 after, from the same interpolant
    s_nodes = np.concatenate(([0.0], rc * rc))
    chi_p = np.concatenate(([float(lw @ post[:3])], post)) - cfac * s_nodes
    m0 = _fd_slopes(s_nodes[:3], chi_p[:3])[0]
    rec.q0_post = float(cfac / (cfac + m0))
    return post, rec


def reconnect(state, phi_b, f=1.0, sheet_width=DEFAULT_SHEET_WIDTH, n_samples=N_SAMPLES, n_levels=N_LEVELS):
    """Apply the reconnection operator to a :class:`CurrentState`; returns ``(new_state, ReconnectionRecord)``.

    ``psi_boundary``, ``time`` and ``meta`` are carried over unchanged (the boundary flux and Ip are
    conserved).
    """
    post, rec = reconnect_psi(state.psi, state.rho_faces, phi_b, f, sheet_width, n_samples, n_levels)
    rec.time = state.time
    out = CurrentState(state.rho_faces, post, state.psi_boundary, state.time, dict(state.meta))
    rec.psi_boundary_change = float(out.psi_boundary - state.psi_boundary)
    return out, rec


# ====================================================================== parameterization
def crash_strength(f, rate, f_cap=F_CAP):
    """Strength Lambda = -ln(1 - f_cap f) * rate [1/s]: the combination that near-axis MSE constrains.

    The -ln(1 - f) form collapsed the cycle-mean q0 best among the forms tried (kinrecon development checks);
    ``f_cap`` < 1 keeps f = 1 finite.
    """
    return -np.log1p(-f_cap * np.asarray(f, dtype=float)) * np.asarray(rate, dtype=float)


def rate_from_strength(strength, f, f_cap=F_CAP):
    """Mean crash rate [1/s] that gives ``strength`` at reconnection fraction ``f`` (inverse of crash_strength)."""
    g = -np.log1p(-f_cap * np.asarray(f, dtype=float))
    if np.any(g <= 0):
        raise ValueError("f must be > 0")
    return np.asarray(strength, dtype=float) / g


def to_strength_coordinates(f, rate, f_cap=F_CAP):
    """(f, rate) -> (ln Lambda, f): the first coordinate moves along the constrained direction."""
    return np.log(crash_strength(f, rate, f_cap)), np.asarray(f, dtype=float)


def from_strength_coordinates(log_strength, f, f_cap=F_CAP):
    """(ln Lambda, f) -> (f, rate)."""
    return np.asarray(f, dtype=float), rate_from_strength(np.exp(log_strength), f, f_cap)


def move_along_strength(f, rate, dlog_strength):
    """Change the strength by the factor exp(dlog_strength) at fixed f (moves the rate)."""
    return float(f), float(rate * math.exp(dlog_strength))


def move_across_strength(f, rate, df, f_cap=F_CAP):
    """Change f by ``df`` at fixed strength (the rate compensates): the poorly constrained direction."""
    lam = float(crash_strength(f, rate, f_cap))
    f2 = float(f) + df
    if not 0.0 < f2 <= 1.0:
        raise ValueError("f + df must lie in (0, 1]")
    return f2, float(rate_from_strength(lam, f2, f_cap))


# ====================================================================== piecewise-linear time integrals
def _pl_integral(t, g, a, b):
    """Integral over [a, b] of the piecewise-linear interpolant of samples (t_i, g_i).

    ``t`` non-decreasing; equal consecutive times represent a jump (zero-length interval).  ``g`` has shape
    (m, ...).  Requires t[0] <= a <= b <= t[-1].
    """
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    if a < t[0] - 1e-15 * max(1.0, abs(t[0])) or b > t[-1] + 1e-15 * max(1.0, abs(t[-1])):
        raise ValueError("window outside the recorded interval")
    t0, t1 = t[:-1], t[1:]
    dt = t1 - t0
    lo = np.clip(np.maximum(a, t0), None, t1)
    hi = np.clip(np.minimum(b, t1), t0, None)
    use = (dt > 0) & (hi > lo)
    if not np.any(use):
        return np.zeros(g.shape[1:])
    t0, dt, lo, hi = t0[use], dt[use], lo[use], hi[use]
    g0, g1 = g[:-1][use], g[1:][use]
    shape = (-1,) + (1,) * (g.ndim - 1)
    wl = ((lo - t0) / dt).reshape(shape)
    wh = ((hi - t0) / dt).reshape(shape)
    glo = g0 + (g1 - g0) * wl
    ghi = g0 + (g1 - g0) * wh
    return np.sum(0.5 * (glo + ghi) * (hi - lo).reshape(shape), axis=0)


def _evaluate(fn, states):
    return np.array([np.asarray(fn(s), dtype=float) for s in states])


# ====================================================================== periodic orbit
@dataclass
class PeriodicOrbit:
    """Periodic state of "diffuse for one period, reconnect".

    ``states[k]`` (k = 0..nsteps) are the states at phase k/nsteps after a crash, from real stepping of the
    solver: ``states[0]`` is just after the crash, ``states[-1]`` just before the next one.  Times run from
    ``t_ref`` to ``t_ref + period``.  ``residual`` is (P1) for the matrix map, ``residual_stepping`` for the
    real stepping; ``converged`` requires both below ``tol``.
    """

    f: float
    period: float
    nsteps: int
    sheet_width: float
    states: List[CurrentState]
    residual: float
    residual_stepping: float
    residual_history: List[float]
    converged: bool
    tol: float
    n_newton: int
    n_plain_cycles: int
    crash: ReconnectionRecord
    wall_time: float
    solver: Optional[CurrentDiffusion] = None
    z: Optional[np.ndarray] = None

    @property
    def pre_crash(self):
        return self.states[-1]

    @property
    def post_crash(self):
        return self.states[0]

    @property
    def times(self):
        return np.array([s.time for s in self.states])

    def phase_state(self, phase):
        """State at phase in [0, 1] after the crash.  Grid phases are returned as stored; other phases are
        advanced from the post-crash state with steps no longer than the orbit's step."""
        phase = float(phase)
        if not 0.0 <= phase <= 1.0:
            raise ValueError("phase must lie in [0, 1]")
        k = phase * self.nsteps
        if abs(k - round(k)) < 1e-12:
            return self.states[int(round(k))].copy()
        t0 = self.states[0].time
        return self.solver.advance(self.states[0], t0 + phase * self.period, dt=self.period / self.nsteps)

    def mean_state(self):
        """Trapezoid time average of psi over the cycle (a state whose diagnostics are first-order averages)."""
        w = np.ones(self.nsteps + 1)
        w[0] = w[-1] = 0.5
        w /= self.nsteps
        psi = w @ np.array([s.psi for s in self.states])
        pb = float(w @ np.array([s.psi_boundary for s in self.states]))
        return CurrentState(self.states[0].rho_faces, psi, pb, self.states[0].time, dict(cycle_average=True))

    def cycle_average(self, fn: Callable):
        """Time average over one period of a diagnostic ``fn(state)`` (trapezoid on the stored states)."""
        g = _evaluate(fn, self.states)
        return _pl_integral(self.times - self.times[0], g, 0.0, self.period) / self.period

    def _periodic_integral(self, g, a, b):
        tk = self.times - self.times[0]
        tau = self.period
        total = np.zeros(g.shape[1:])
        c = math.floor(a / tau)
        t = a
        while t < b - 1e-15 * tau:
            lo = t - c * tau
            hi = min(b - c * tau, tau)
            total = total + _pl_integral(tk, g, max(lo, 0.0), hi)
            c += 1
            t = c * tau
        return total

    def window_average(self, fn: Callable, window: float, phase_start: float = 0.0, values=None):
        """Average of ``fn(state)`` over a window of length ``window`` [s] starting ``phase_start`` periods after
        a crash, on the periodic extension of the orbit (crashes at every multiple of the period)."""
        if window <= 0:
            raise ValueError("window must be positive")
        g = _evaluate(fn, self.states) if values is None else np.asarray(values, dtype=float)
        a = float(phase_start) * self.period
        return self._periodic_integral(g, a, a + window) / window

    def window_average_over_phases(self, fn: Callable, window: float, n_phases: Optional[int] = None):
        """Window averages for window starts at every stored phase (k/nsteps, k < nsteps): returns
        dict(phases, values, mean, min, max).  The mean over uniformly spaced starts equals the cycle average."""
        g = _evaluate(fn, self.states)
        n = self.nsteps if n_phases is None else int(n_phases)
        ph = np.arange(n) / n
        vals = np.array([self.window_average(None, window, p, values=g) for p in ph])
        return dict(phases=ph, values=vals, mean=vals.mean(axis=0), min=vals.min(axis=0), max=vals.max(axis=0))


def _gauge(x):
    return x - x[-1]


def _step_matrices(solver, rf, dt, t_ref, scheme):
    """Affine one-step map y1 = S y0 + s0 of the solver (Ip boundary), built from the solver's own step."""
    n = rf.size - 1
    base = solver.step(CurrentState(rf, np.zeros(n), 0.0, t_ref), dt, scheme=scheme).psi
    S = np.empty((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        S[:, j] = solver.step(CurrentState(rf, e, 0.0, t_ref), dt, scheme=scheme).psi - base
        e[j] = 0.0
    return S, base


def _affine_power(S, s0, k):
    """(S^k, sum_{i<k} S^i s0) by binary doubling."""
    n = S.shape[0]
    R, r = np.eye(n), np.zeros(n)
    P, p = S.copy(), s0.copy()
    while k:
        if k & 1:
            R, r = P @ R, P @ r + p
        k >>= 1
        if k:
            P, p = P @ P, P @ p + p
    return R, r


def _flat_q1_state(solver, rf, t_ref, phi_b):
    """Relaxed (steady) state with q floored at 1: the default initial guess for the orbit."""
    st = solver.steady_state(rf, t=t_ref)
    q = solver.profiles(st)["q"]
    qf = np.maximum(q, 1.0)
    rc = 0.5 * (rf[1:] + rf[:-1])
    dp = phi_b * rf / (np.pi * qf)
    psi = np.concatenate(([0.0], np.cumsum(dp[1:-1] * np.diff(rc))))
    return _gauge(psi)


def _anderson(resid, z, nrm, tol, max_evals, hist, verbose, depth=5):
    """Anderson acceleration (type II, depth ``depth``) of the physical map z -> z + r(z).  An accelerated step
    is accepted only if it lowers the residual measure below the current one; otherwise the history is
    cleared and a plain step (one physical cycle) is taken, which is always accepted.  Returns
    (z, residual, n_evals, y, info) for the best iterate.  The stopping test is the unchanged (P1)."""
    r, y, info, nrm = resid(z)
    best = (nrm, z, y, info)
    Gs, Fs = [], []
    n_ev = 1
    while n_ev < max_evals and best[0] >= tol:
        g = _gauge(z + r)
        Gs.append(g)
        Fs.append(r)
        if len(Gs) > depth + 1:
            Gs.pop(0)
            Fs.pop(0)
        step = None
        if len(Gs) >= 2:
            dF = np.diff(np.array(Fs), axis=0).T
            dG = np.diff(np.array(Gs), axis=0).T
            gam = np.linalg.lstsq(dF, r, rcond=None)[0]
            za = _gauge(g - dG @ gam)
            try:
                ra, ya, infa, na = resid(za)
                n_ev += 1
                if na < nrm:
                    step = (za, ra, ya, infa, na)
            except ValueError:
                n_ev += 1
        if step is None:
            Gs, Fs = [g], [r]
            rg, yg, infg, ng = resid(g)
            n_ev += 1
            step = (g, rg, yg, infg, ng)
        z, r, y, info, nrm = step
        if nrm < best[0]:
            best = (nrm, z, y, info)
        hist.append(nrm)
        if verbose and n_ev % 100 < 2:
            print(f"  anderson {n_ev}: residual {nrm:.3e} best {best[0]:.3e}")
    return best[1], best[0], n_ev, best[2], best[3]


def solve_periodic_orbit(solver: CurrentDiffusion, rho_faces, f, period, nsteps=40,
                         sheet_width=DEFAULT_SHEET_WIDTH, z0=None, t_ref=0.0, tol=CYCLE_TOL, max_iter=40,
                         max_map_evals=5000, scheme="trbdf2", n_samples=N_SAMPLES, n_levels=N_LEVELS,
                         verbose=False):
    """Periodic orbit of "diffuse for ``period`` in ``nsteps`` steps, then reconnect with fraction ``f``".

    Inputs (geometry, conductivity, non-inductive current, Ip) are frozen: the one-step map is built at
    ``t_ref`` and checked against :meth:`CurrentDiffusion.advance` over one period (a ``ValueError`` is
    raised if they differ, i.e. if the inputs are not stationary).  Requires the Ip boundary condition.
    Newton's method is tried first (``max_iter`` iterations); if its line search fails or the cap is reached,
    the iteration continues with the Anderson-accelerated physical map (at most ``max_map_evals``
    evaluations; ``n_plain_cycles`` counts them).
    ``z0`` (cell values; any offset) warm-starts the Newton iteration (continuation).  ``tol`` is the
    criterion (P1); it may be made stricter but not looser than :data:`CYCLE_TOL`.  Non-convergence is
    reported (``converged=False`` and a warning), never hidden.
    """
    t_start = _time.perf_counter()
    if tol > CYCLE_TOL:
        raise ValueError(f"tol {tol:g} is looser than the decided cycle criterion {CYCLE_TOL:g}")
    if not solver.ip_mode:
        raise NotImplementedError("the periodic orbit is implemented for the Ip boundary condition")
    f = _check_f(f, sheet_width)
    rf = np.asarray(rho_faces, dtype=float)
    rc = 0.5 * (rf[1:] + rf[:-1])
    n = rc.size
    nsteps = int(nsteps)
    phi_b = solver.geometry_at(t_ref).phi_b
    cfac = phi_b / (2.0 * np.pi)
    S, s0 = _step_matrices(solver, rf, period / nsteps, t_ref, scheme)
    M, c = _affine_power(S, s0, nsteps)
    # stationarity / consistency check of the matrix form against the solver
    x = solver.steady_state(rf, t=t_ref).psi
    ref = solver.advance(CurrentState(rf, x, 0.0, t_ref), t_ref + period, nsteps=nsteps, scheme=scheme).psi
    mism = np.max(np.abs(M @ x + c - ref)) / max(np.max(np.abs(ref - ref[-1])), 1e-300)
    if mism > 1e-10:
        raise ValueError(f"one-period map differs from advance by {mism:.2e}: inputs are not stationary")

    def K(y):
        return _core(y, rc, cfac, f, sheet_width, n_samples, n_levels)

    def resid(z):
        y = M @ z + c
        post, info = K(y)
        r = _gauge(post) - z
        return r, y, info, float(np.max(np.abs(r)) / info["depth"])

    def jacobian(y, info):
        """G (K'(y) M) with K' along the dominant left singular vectors of M's rows that K reads."""
        if info["active"]:
            mprime = int(min(n, np.searchsorted(rc, info["zone_outer"]) + 4))
        else:
            mprime = 4
        U, sv, Vt = np.linalg.svd(M[:mprime, :], full_matrices=False)
        keep = sv > 1e-13 * sv[0]
        DM = np.zeros_like(M)
        for kk in np.nonzero(keep)[0]:
            u = np.zeros(n)
            u[:mprime] = U[:, kk]
            h = 1e-6 * info["depth"] / np.max(np.abs(u))
            try:
                d = (K(y + h * u)[0] - K(y - h * u)[0]) / (2.0 * h) - u
            except ValueError:
                continue
            DM += np.outer(d, sv[kk] * Vt[kk])
        J = M + DM
        return J - J[-1:, :]

    z = _gauge(_flat_q1_state(solver, rf, t_ref, phi_b) if z0 is None else np.asarray(z0, dtype=float))
    r, y, info, nrm = resid(z)
    hist = [nrm]
    n_plain = 0
    it = 0
    eye = np.eye(n)
    while it < max_iter and nrm >= tol:
        it += 1
        J = jacobian(y, info)
        dz = np.linalg.solve(J - eye, -r)
        lam = 1.0
        accepted = False
        for _ in range(11):                 # lam down to 2^-10; a shorter step means the linear model failed
            zt = _gauge(z + lam * dz)
            try:
                rt, yt, it_info, nt = resid(zt)
            except ValueError:
                nt = np.inf
            if nt < (1.0 - 1e-4 * lam) * nrm or nt < tol:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            break
        z, r, y, info, nrm = zt, rt, yt, it_info, nt
        hist.append(nrm)
        if verbose:
            print(f"  newton {it}: lam {lam:.3g} residual {nrm:.3e} rho_mix {info.get('rho_mix', 0.0):.4f}")
    if nrm >= tol:
        # Newton's linear model failed (the fixed point lies next to a kink of the map, e.g. the q0 = 1 onset)
        # or the iteration cap was reached: Anderson-accelerated iteration of the physical map itself
        # (H. F. Walker and P. Ni, SIAM J. Numer. Anal. 49, 1715 (2011)), with restarts to plain cycles.
        z, nrm, n_plain, y, info = _anderson(resid, z, nrm, tol, max_map_evals, hist, verbose)
    # the orbit by real stepping from the post-crash state
    st = CurrentState(rf, z, 0.0, t_ref)
    pre = solver.advance(st, t_ref + period, nsteps=nsteps, scheme=scheme)
    post, rec = reconnect(pre, phi_b, f, sheet_width, n_samples, n_levels)
    z_step = _gauge(post.psi)
    res_step = float(np.max(np.abs(z_step - z)) / rec.depth)
    states = [st]
    s = st
    for kk in range(nsteps):
        s = solver.step(s, period / nsteps, scheme=scheme)
        s.time = t_ref + (kk + 1) * period / nsteps if kk < nsteps - 1 else t_ref + period
        states.append(s)
    # psi_boundary of the post-crash state: the crash leaves the boundary flux unchanged
    states[0] = CurrentState(rf, z, pre.psi_boundary - pre.psi[-1], t_ref, dict(phase=0.0))
    converged = bool(nrm < tol and res_step < tol)
    if not converged:
        warnings.warn(f"periodic orbit not converged: residual {nrm:.3e} (stepping {res_step:.3e}) "
                      f"> {tol:g} for f = {f}, period = {period}", RuntimeWarning)
    return PeriodicOrbit(f=f, period=float(period), nsteps=nsteps, sheet_width=float(sheet_width), states=states,
                         residual=float(nrm), residual_stepping=res_step, residual_history=hist,
                         converged=converged, tol=tol, n_newton=it, n_plain_cycles=n_plain, crash=rec,
                         wall_time=_time.perf_counter() - t_start, solver=solver, z=z)


# ====================================================================== time-dependent mode
@dataclass
class CrashEvent:
    time: float
    record: ReconnectionRecord


class Trajectory:
    """States recorded in [t_start, t_end] (copies), with crash jumps stored as two states at the same time."""

    def __init__(self, t_start=-np.inf, t_end=np.inf):
        self.t_start, self.t_end = float(t_start), float(t_end)
        self.states: List[CurrentState] = []

    def add(self, state):
        if self.t_start <= state.time <= self.t_end:
            self.states.append(state.copy())

    @property
    def times(self):
        return np.array([s.time for s in self.states])

    def values(self, fn: Callable):
        return self.times, _evaluate(fn, self.states)

    def window_average(self, fn: Callable, t_a, t_b, values=None):
        """Average of ``fn(state)`` over [t_a, t_b] (piecewise linear in time between stored states)."""
        if not t_b > t_a:
            raise ValueError("t_b must exceed t_a")
        g = _evaluate(fn, self.states) if values is None else np.asarray(values, dtype=float)
        return _pl_integral(self.times, g, float(t_a), float(t_b)) / (float(t_b) - float(t_a))


class CrashSchedule:
    """``post_step`` hook that applies the reconnection operator at given crash times (from data).

    Use with :func:`advance_with_crashes`, which makes a step end exactly on every crash time.  If a step
    ends past a pending crash time without landing on it, a ``RuntimeError`` is raised (a crash would
    otherwise be skipped silently).  Crashes at exactly the start time of a run are treated as already
    applied.  ``events`` lists every applied crash with its :class:`ReconnectionRecord`.
    """

    def __init__(self, crash_times: Sequence[float], f=1.0, sheet_width=DEFAULT_SHEET_WIDTH,
                 trajectory: Optional[Trajectory] = None, n_samples=N_SAMPLES, n_levels=N_LEVELS):
        self.times = np.unique(np.asarray(crash_times, dtype=float))
        self.f = _check_f(f, sheet_width)
        self.sheet_width = float(sheet_width)
        self.trajectory = trajectory
        self.n_samples, self.n_levels = n_samples, n_levels
        self.events: List[CrashEvent] = []
        self._next = 0

    def reset(self, t0):
        self._next = int(np.searchsorted(self.times, t0, side="right"))

    def pending(self, t0, t1):
        """Crash times in (t0, t1]."""
        return self.times[(self.times > t0) & (self.times <= t1)]

    def __call__(self, state, t, solver):
        if self._next < self.times.size and self.times[self._next] < t:
            raise RuntimeError(f"crash time {self.times[self._next]!r} skipped (step ended at {t!r}); "
                               "use advance_with_crashes so that steps end on crash times")
        if self.trajectory is not None:
            self.trajectory.add(state)
        if self._next < self.times.size and self.times[self._next] == t:
            phi_b = solver.geometry_at(t).phi_b
            state, rec = reconnect(state, phi_b, self.f, self.sheet_width, self.n_samples, self.n_levels)
            self.events.append(CrashEvent(time=float(t), record=rec))
            self._next += 1
            if self.trajectory is not None:
                self.trajectory.add(state)
        return state


def advance_with_crashes(solver: CurrentDiffusion, state: CurrentState, t_end, schedule: CrashSchedule, dt,
                         scheme="trbdf2", theta=0.5, stops=()):
    """Advance to ``t_end`` with steps no longer than ``dt``, applying ``schedule`` at its crash times.

    The interval is split at every crash time in (state.time, t_end] and at the optional ``stops`` (e.g. the
    edges of a measurement window, so that the trajectory holds a state exactly there); each segment is
    advanced in equal steps (step sizes differ slightly between segments), and the last step of a segment
    lands exactly on its end, where the schedule (the ``post_step`` hook) applies the operator if it is a crash
    time.
    """
    schedule.reset(state.time)
    if schedule.trajectory is not None:
        schedule.trajectory.add(state)
    t_end = float(t_end)
    extra = [float(x) for x in stops if state.time < float(x) < t_end]
    bounds = sorted(set(list(schedule.pending(state.time, t_end)) + extra + [t_end]))
    s = state
    for b in bounds:
        if b > s.time:
            s = solver.advance(s, b, dt=dt, scheme=scheme, theta=theta, post_step=schedule)
    return s


@dataclass
class SliceSchedule:
    """Crash schedule for one measurement window with a stated history before it."""

    window: tuple
    history: float
    t_start: float
    crash_times: np.ndarray
    mean_interval: Optional[float]
    last_crash_before_start: Optional[float]
    n_crashes_history: int
    n_crashes_window: int


def schedule_for_slice(crash_times, window, history=DEFAULT_HISTORY, record_start=None):
    """Build the crash schedule for a window ``(t_a, t_b)`` from measured crash times.

    The run starts at ``t_start = t_a - history``.  ``record_start`` is the start of the time range over
    which crashes were detected; by default the first detected crash.  A ``ValueError`` is raised if the
    record does not cover the history (``record_start > t_start``).  The mean interval uses all crashes in
    [t_start, t_b] (None if fewer than two).
    """
    t_a, t_b = (float(window[0]), float(window[1]))
    if not t_b > t_a:
        raise ValueError("window must have t_b > t_a")
    if history < 0:
        raise ValueError("history must be >= 0")
    ct = np.unique(np.asarray(crash_times, dtype=float))
    if ct.size == 0:
        raise ValueError("no crash times: use the periodic orbit (single-slice mode)")
    t_start = t_a - float(history)
    rs = float(ct[0]) if record_start is None else float(record_start)
    if rs > t_start:
        raise ValueError(f"crash record starts at {rs} but the history requires coverage from {t_start} "
                         f"({history} s before the window)")
    use = ct[(ct > t_start) & (ct <= t_b)]
    before = ct[ct <= t_start]
    span = ct[(ct >= t_start) & (ct <= t_b)]
    mean = float(np.mean(np.diff(span))) if span.size >= 2 else None
    return SliceSchedule(window=(t_a, t_b), history=float(history), t_start=t_start, crash_times=use,
                         mean_interval=mean, last_crash_before_start=float(before[-1]) if before.size else None,
                         n_crashes_history=int(np.count_nonzero(use <= t_a)),
                         n_crashes_window=int(np.count_nonzero((use > t_a) & (use <= t_b))))


def simulate_slice(solver, rho_faces, sched: SliceSchedule, f, dt, nsteps_orbit=40,
                   sheet_width=DEFAULT_SHEET_WIDTH, scheme="trbdf2", orbit_z0=None):
    """Time-dependent run for one window: start from the periodic orbit at the schedule's mean interval
    (phase from the last crash before the start when known, else just after a crash), apply the measured
    crashes, record the window.  Inputs are those of ``solver`` (frozen inputs give the periodic start).
    Returns dict(trajectory, events, orbit, initial_phase, state)."""
    if sched.mean_interval is None:
        raise ValueError("need at least two crashes to set the starting orbit")
    tau = sched.mean_interval
    orbit = solve_periodic_orbit(solver, rho_faces, f, tau, nsteps=nsteps_orbit, sheet_width=sheet_width,
                                 z0=orbit_z0, scheme=scheme)
    phase = 0.0
    if sched.last_crash_before_start is not None:
        phase = min((sched.t_start - sched.last_crash_before_start) / tau, 1.0)
    s0 = orbit.phase_state(phase)
    s0 = CurrentState(s0.rho_faces, s0.psi, s0.psi_boundary, sched.t_start, dict(s0.meta))
    traj = Trajectory(*sched.window)
    sch = CrashSchedule(sched.crash_times, f, sheet_width, trajectory=traj)
    s = advance_with_crashes(solver, s0, sched.window[1], sch, dt, scheme=scheme, stops=(sched.window[0],))
    return dict(trajectory=traj, events=sch.events, orbit=orbit, initial_phase=phase, state=s)


# ====================================================================== trigger diagnostic (never in the fit)
def q1_shear_diagnostic(state, phi_b, minimum_radius=0.05, s_critical=0.1):
    """Optional report, not used by any model function: outermost q = 1 radius and magnetic shear there.

    Shear s = (rho/q) dq/drho = -2 rho^2 chi''(s1)/c at q = 1 (from (S1)-(S2)).  ``trigger`` is the TORAX
    ``simple`` criterion rho_q1 > minimum_radius and s(rho_q1) > s_critical (google-deepmind/torax,
    ``torax/_src/mhd/sawtooth/simple_trigger.py``, defaults 0.05 and 0.1).  A boolean of the state; it must
    never enter a residual or a likelihood.
    """
    rf = state.rho_faces
    rc = 0.5 * (rf[1:] + rf[:-1])
    cfac = phi_b / (2.0 * np.pi)
    s_cells = rc * rc
    lw = _axis_weights(s_cells[:3])
    s_nodes = np.concatenate(([0.0], s_cells))
    chi = np.concatenate(([float(lw @ state.psi[:3])], state.psi)) - cfac * s_nodes
    H = _Hermite(s_nodes, chi, _fd_slopes(s_nodes, chi))
    cs, ck = H.critical_points()
    mx = cs[ck > 0]
    if mx.size == 0:
        return dict(rho_q1=0.0, shear=0.0, trigger=False)
    s1 = float(mx[-1])
    rho1 = math.sqrt(s1)
    shear = float(-2.0 * s1 * H.deriv2(s1) / cfac)
    return dict(rho_q1=rho1, shear=shear, trigger=bool(rho1 > minimum_radius and shear > s_critical))
