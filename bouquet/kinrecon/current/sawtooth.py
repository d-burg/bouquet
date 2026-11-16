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
ends, at the new position of every interior critical level, and -- when the
minimum of ``chi`` on ``[0, s_*]`` is interior and ``D`` starts at the axis --
at the new position ``s_lo + mu(chi(0))`` of the axis level, where the axis
crossing (slope ``chi'(0) != 0``) ends.  With
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
the enclosed current outside the changed zone (``|rho - rho_k| >= w_eff``
around every sheet, so out to ``rho_mix + w_eff``, not only ``rho_mix``), psi
at the boundary and Ip.  Helical flux (the toroidal-flux measure of every
level set of ``chi``) is conserved only for ``f = 1`` and ``w = 0``: it is
not conserved for 0 < f < 1 (phenomenological), and the sheet smoothing (K4)
changes it near every sheet (by 1-3 % of ``s_mix`` at ``w = 0.05`` for
q0 = 0.7-0.9 in the independent review's check).

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

    r = || G K_f(M z + c) - z ||_inf / | psi(rho_mix) - psi(0) |                     (PO)

for the pre-crash state ``y = M z + c`` (``psi(rho_mix) - psi(0)`` from the
same interpolant as the operator; when the operator is inactive the
denominator is ``|psi(last cell) - psi(0)|``).  The denominator is positive
in the positive frame, which is enforced: Ip <= 0, Phi_b <= 0 or a state with
``psi(last cell) <= psi(0)`` raises :class:`~.diffusion.PositiveFrameError`
(a negative denominator would let any negative residual pass).  Both the matrix form and the
real stepping (``advance`` then ``reconnect``) must meet it.

Time-averaged core (the default core treatment of the package)
--------------------------------------------------------------
The default holds the minimum of the core q at a floor (``q_floor``, default :data:`DEFAULT_Q_FLOOR` = 1) through the
RESISTIVITY, with the stationary-sawtooth model that FUSE attributes to Jardin:

.. code-block:: text

    eta(rho) -> eta(rho_inv)   inside the flattening region,   unchanged outside                (F1)

i.e. the resistivity inside the region is set to its own value at the region's outer edge ``rho_inv``.  Inside, the
ohmic current then follows the (flat) resistivity instead of the peaked one, so the core current is flatter and q
higher; the resistivity is continuous across the edges (its slope breaks), so ``<J.B>`` has no jump there, and Ip is
untouched (the diffusion keeps its Ip condition).

What FUSE does, and what this module takes from it.  FUSE's sawtooth treatment is TWO steps that its compound
actors (``ActorStationaryPlasma``, ``ActorDynamicPlasma``) run together: (1) its current-diffusion actor
(``ActorQED``, ``src/actors/current/qed_actor.jl``, parameter ``qmin_desired``, default 1.0) flattens the
RESISTIVITY inside the inversion radius as in (F1) (``QED.eta_JBni_sawteeth`` in ``ext/IMAS_Ext.jl`` of QED.jl);
(2) a separate sawtooth source actor (``ActorSawteethSource``, ``src/actors/hcd/sawteeth_source_actor.jl``) writes
a ``sawteeth`` source that FLATTENS THE NON-INDUCTIVE CURRENT inside the same radius (its parameters: the
flattening factor ``flat_factor``, default 0.5; a smoothing width ``min(rho/4, 0.05)``; ``period``, infinite by
default), and ``ActorQED`` reads that source back as part of the non-inductive current.  (FUSE and QED.jl are public,
ProjectTorreyPines on GitHub; the rule is re-implemented here, no code is taken from there.)  THIS MODULE IMPLEMENTS
STEP (1), THE RESISTIVITY, ALONE; the non-inductive (mostly bootstrap) current is left as the sources give it.  What
that implies for q: the bootstrap current inside the radius is usually hollow, so the total current dips toward the
axis, q_min sits near the outer edge of the region (at the floor, where the floor is met) and the axis q0 lies ABOVE
the floor; with FUSE's step (2) as well, q would be nearly flat at the floor inside the radius.  Read a q0 above the
floor as a property of this one-step model, not as a measurement of the core.

The flattening region is decided by the MODEL's q, never by classifying data: the q of the steady state of the
model sources WITHOUT the floor (:func:`floor_region`; the face q of :func:`face_q`, the axis value extrapolated),
for a single slice and, at the time of each diffusion step, in a series (below):

* (a) q below the floor ON AXIS: the region is ``[0, rho_inv]`` (FUSE's rule, with a continuous edge, below);
* (b) q at or above the floor on axis and below it OFF AXIS only (reversed shear): the floor does NOT act; the state
  is left as it is and the record says so (``floor_branch = "off_axis"``, ``acted = False``, ``q_floor_met =
  False``, the crossings of the sub-floor region kept).  Reason: a resistivity floor can only take OHMIC current out
  of the region it flattens.  Flattening from the axis out past an off-axis minimum (FUSE's rule in this case) sets
  the whole core's resistivity to its value outside the minimum; the core's current is then mostly the ohmic one,
  it falls by about that resistivity ratio (times the rise of the loop voltage) and q0 rises by about the same
  factor, several times the floor where the minimum is carried by non-inductive current (q0 1.26 -> 4.5 on a
  constructed case).  Flattening only the annulus between the crossings leaves such a minimum where it is (the
  current inside it is non-inductive), lowers q0 through the higher loop voltage (below the floor on the constructed
  cases) and drives the axis q toward zero where the annulus reaches close to the axis (the core inside the annulus
  keeps its own, higher conductivity).  The stationary-sawtooth model is a model of a core with q below the floor on
  axis, which an off-axis minimum is not;
* (c) q at or above the floor everywhere: nothing changes.

Limitations (not removed by these rules; recorded, not hidden): (i) in case (a), where the current that holds q
below the floor off axis is largely non-inductive (a strong off-axis source), the region must still reach past that
minimum and q0 rises far above the floor by the mechanism of (b) (about 4.5 on a constructed case with q0 = 0.96
and an off-axis minimum of 0.88); the record shows it (``q_axis_after``, ``eta_ratio_inv_to_axis``).  FUSE's second
step (the non-inductive flattening, not implemented here) is what keeps such a core near the floor there.  (ii) At
the boundary between (a) and (b) -- the plain q0 crossing the floor while an off-axis minimum lies below it -- the
state is not continuous: on the (a) side the region reaches past the minimum, on the (b) side nothing acts; the jump
is of the size of what the floor must lift there (about 0.02 in q0 and q_min where the bootstrap current leaves a
shallow dip next to the axis; the full blow-up of (i) where a strong off-axis source carries the minimum).

``rho_inv`` is continuous, not a grid face (rule ``"continuous"``, the default):

* steady state (:func:`resistive_floor_steady`, the single-slice state of :mod:`..current_model`): ``rho_inv`` is the
  smallest outer edge whose steady state has q >= floor on every face, searched from the axis out to the plain
  state's outermost upward crossing of the floor: the faces in between are scanned outward to bracket
  the first such edge and the edge is then bisected to round-off (:data:`RADIUS_ROUND_OFF`) inside the bracket, so
  where the floor is met q_min equals it to round-off.  If even the outermost crossing does not meet it, that
  crossing is kept (as FUSE keeps its outermost node) and recorded (``q_floor_met = False``);
* time-dependent (:class:`ResistiveFloor` as the ``resistivity_modifier`` of a diffusion step, :mod:`..series`): on
  every step the region is the SINGLE-SLICE region of the step's model sources at the time of the step (the steady
  rule above applied to the diffusion's own sources, conductivity and Ip at that time); the evolving state is read
  only for the refusal at the domain edge and the record.  The diffusion then relaxes the core toward the
  single-slice floored state of the same sources: q is NOT held at the floor in a transient, and a series that
  stands still in time ends on the single-slice state.  This is NOT FUSE's time-dependent rule, which takes the
  radius from the previous step's q (``rule="grid"`` keeps that form).  Reason: the floored state sits at the floor
  by construction, so a region read from its own q switches on and off from step to step -- off whenever q is at or
  above the floor everywhere (the region then jumps from the whole core to nothing), and, with branch (b) read from
  the state, whenever its axis q moves above the floor while the minimum at the region's edge is below it.  Measured
  on a clean synthetic series (11 slices, 50 ms apart, flat Ip and a +2 %/slice ramp, steps of 1 and 0.5 ms): the
  floor acted on 47 of 50 steps of an interval and 1-5 slices per series were lost to the pass limit; with branch
  (b) taken from the model's steady state instead, the floor acted on nearly every step but the pass loop failed on
  most slices (the region's edge, the state's own crossing, moves with the geometry of each pass); with this rule
  every slice converged (2-3 passes at flat Ip, 7 on the ramp) and the steps of 1 and 0.5 ms agreed in the traced
  q0 to about 1e-8 (flat) and 1e-7 (ramp).  Cost: one steady search (tens of steady states) per diffusion step,
  kept small by the solver reusing the geometry it evaluated at a time;
* the resistivity enters the diffusion per cell (the cell integral of the conductivity, midpoint rule); a cell cut
  by the region's edge takes the OVERLAP-WEIGHTED conductivity ``w sigma(rho_inv) + (1 - w) sigma_cell``
  (``w`` the fraction of the cell inside the region; 1 or 0 exactly for a cell wholly inside or outside), so the
  state is continuous in ``rho_inv`` and converges with the grid instead of stepping with it.

``rule="grid"`` (an argument of :func:`resistive_floor_steady` and :class:`ResistiveFloor` only, for comparison) is
FUSE's own grid rule as built earlier: the radius snapped to a grid face (the first face whose steady state meets the
floor; in time, from the state at the start of the step, the outermost face below it) and the region always starting
at the axis.

This is not the reconnection model of this module and needs no sawtooth parameter, crash time or amplitude.  It is a
time-averaged model of a sawtoothing core, not a clamp.  A state whose q is below the floor at the domain edge is
refused (:class:`QFloorRefused`; FUSE has no such refusal).  Further differences from FUSE's actor (none adds a
number): the grid is this diffusion grid (``n_cells`` uniform cells) and the resistivity is set per cell, where FUSE
sets it on the nodes of its core-profile grid and interpolates log(eta) between them; in time the region is decided
on every diffusion step from the sources (above), where FUSE decides it from the state's q once per actor step (a
step of ``Nt`` sub-steps) when it solves for Ip, and on every sub-step in its loop-voltage-controller mode.

Record only (nothing reads it): the floor's branch, both edges, q0 and q_min before and after, and its effect against
the same model without the floor -- the loop voltage, l_i (``li3``) and the fraction of Ip moved from inside
``rho_inv`` to outside it.

:func:`floor_q` (opt-in, ``q_floor_form="sharp"``) is the earlier construction: q clamped to ``max(q, q_floor)`` on
the faces, which makes ``<J.B>`` jump at the floor radius (q is continuous there, its slope breaks); it is kept for
comparison and for the periodic orbit's starting guess.  :mod:`..current_model` applies the default to the steady
model state of a single slice and :mod:`..series` to every diffusion step; the periodic orbit, the measured-crash runs
and the crash tables below are opt-ins.

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
:func:`move_across_strength` move along and across it.  The strength is
DEFINED with the cap: every strength this package fits, records or takes as
input is ``-ln(1 - 0.999 f) nu``, not ``-ln(1 - f) nu`` (the two differ by
~0.1 % at f = 0.37 and are finite vs infinite at f = 1).

Numerical defaults (documented; each a module constant or an argument):
sheet width 0.05 in rho (:data:`DEFAULT_SHEET_WIDTH`), 1,024 samples per
monotone piece and 2,048 helical-flux levels (:data:`N_SAMPLES`,
:data:`N_LEVELS`), 1.5 s of crash history before a measurement window
(:data:`DEFAULT_HISTORY`), the periodic-orbit criterion 1e-9
(:data:`CYCLE_TOL`).

Window averages
---------------
Diagnostics are averaged, not the current: piecewise-linear interpolation in
time between stored states (trapezoid rule), with the pre- and post-crash
states stored at the same time so a crash is a jump; exact integration over
any window.
"""

import copy
import math
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .diffusion import CurrentDiffusion, CurrentState, PositiveFrameError
from .diffusion import _boundary_gradient_weights

__all__ = [
    "CYCLE_TOL",
    "DEFAULT_Q_FLOOR",
    "QFloorRefused",
    "Q_FLOOR_FORMS",
    "floor_q",
    "floor_crossings",
    "floor_region",
    "FLOOR_BRANCHES",
    "RESISTIVE_FLOOR_RULES",
    "RADIUS_ROUND_OFF",
    "face_q",
    "ResistiveFloor",
    "resistive_floor_steady",
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

CYCLE_TOL = 1e-9            # decided periodic-orbit criterion (PO); never loosened
DEFAULT_SHEET_WIDTH = 0.05  # TRANSP XSWIDQ default, in rho = sqrt(Phi_N)
DEFAULT_HISTORY = 1.5       # s of crash history before a window (core memory 0.5-1 s measured)
F_CAP = 0.999               # cap in -ln(1 - F_CAP f) so that f = 1 is finite
N_SAMPLES = 1024            # samples per monotone piece (fixed relative positions)
N_LEVELS = 2048             # helical-flux levels of the rearranged profile (fixed relative positions)
_HALF_PI = 0.5 * np.pi


#: Default floor of the time-averaged q-floor core treatment (one documented setting).  The value is FUSE's
#: ``ActorQED`` default ``qmin_desired = 1.0`` (``src/actors/current/qed_actor.jl``), which FUSE's compound actors
#: also pass to its separate sawtooth source actor (``ActorSawteethSource``: non-inductive current flattened inside the
#: same radius; ``flat_factor`` 0.5, width ``min(rho/4, 0.05)``, ``period`` infinite by default).  This package applies
#: the floor through the resistivity alone (module docstring), so q_min sits at the floor and q0 above it.
DEFAULT_Q_FLOOR = 1.0
#: Branches of the resistive floor, decided by the model state's q (:func:`floor_region`): ``"axis"`` (q below the
#: floor on axis: the region from the axis), ``"off_axis"`` (q below it off axis only: no change, recorded),
#: ``"none"``.
FLOOR_BRANCHES = ("axis", "off_axis", "none")
#: Rules for the flattening radius of the resistive floor: ``"continuous"`` (default; the radius to round-off, the cut
#: cells overlap-weighted, the region by :func:`floor_region`) and ``"grid"`` (FUSE's grid rule: the radius snapped
#: to a face, the region from the axis; comparison only).
RESISTIVE_FLOOR_RULES = ("continuous", "grid")
#: Bisection stop of the steady-state flattening radius, in rho (0..1): round-off, not a tolerance.
RADIUS_ROUND_OFF = 1e-14
_RADIUS_BISECTIONS = 80     # loop bound only: the bracket is at most one cell wide, 2^-80 of it is far below round-off
#: Forms of the q floor: ``"resistive"`` (default; the resistivity inside the flattening radius set to its value
#: there, :class:`ResistiveFloor`, :func:`resistive_floor_steady`) and ``"sharp"`` (opt-in; q clamped,
#: :func:`floor_q`).
Q_FLOOR_FORMS = ("resistive", "sharp")


class QFloorRefused(ValueError):
    """The q floor cannot act without changing the current at the domain edge: q below the floor at the domain edge
    (both forms), or on the last interior face, which the boundary gradient uses (:func:`floor_q` only)."""


def _check_q_floor(q_floor):
    qf = float(q_floor)
    if not (np.isfinite(qf) and qf > 0.0):
        raise ValueError("q_floor must be finite and > 0")
    return qf


def floor_crossings(rho_faces, q, q_floor):
    """Every radius where ``q`` crosses ``q_floor`` between two faces (linear in q between them), innermost first,
    each with its direction: ``"up"`` where q rises through the floor going outward (the outer edge of a region below
    the floor), ``"down"`` where it falls through it (the inner edge of an off-axis region, reversed shear)."""
    rf = np.asarray(rho_faces, float)
    d = np.asarray(q, float) - float(q_floor)
    out = []
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or (a < 0.0) == (b < 0.0):
            continue
        r = float(rf[i] + (-a) * (rf[i + 1] - rf[i]) / (b - a)) if b != a else float(rf[i])
        out.append(dict(rho=r, direction="up" if a < 0.0 else "down"))
    return out


def face_q(state, phi_b):
    """q on the faces of ``state`` from its own psi, with the formulas of
    :meth:`.diffusion.CurrentDiffusion.profiles` (``q = Phi_b rho / (pi psi')``; interior faces two-point, the domain
    edge through ``psi_boundary`` and the two outer cells, the axis value extrapolated as ``q0 + c rho^2``), without
    evaluating any coefficient, so it can be used inside a ``resistivity_modifier``.  ``phi_b``: the boundary toroidal
    flux [Wb] at the state's time."""
    rf = np.asarray(state.rho_faces, float)
    rc = 0.5 * (rf[1:] + rf[:-1])
    n = rc.size
    dpsi = np.zeros(n + 1)
    dpsi[1:n] = np.diff(state.psi) / np.diff(rc)
    wb, w1, w2 = _boundary_gradient_weights(rc)
    dpsi[n] = wb * state.psi_boundary + w1 * state.psi[-1] + w2 * state.psi[-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = float(phi_b) * rf / (np.pi * dpsi)
    q[0] = q[1] - (q[2] - q[1]) * rf[1] ** 2 / (rf[2] ** 2 - rf[1] ** 2)
    return q


def floor_region(rho_faces, q, q_floor):
    """The region the resistive floor flattens, from a state's face q (module docstring, branches (a)-(c)).

    Returns ``dict(branch, rho_inner, rho_outer, crossings)``: ``branch`` one of :data:`FLOOR_BRANCHES` -- ``"axis"``
    where q is below the floor on the axis face (the extrapolated axis value): the floor acts on ``[0, rho_outer]``;
    ``"off_axis"`` where q is at or above it on axis and below it on some other face: the floor does not act, and
    ``[rho_inner, rho_outer]`` (``rho_inner`` the innermost downward crossing) is the extent of the sub-floor region,
    for the record; ``"none"`` where q >= floor on every face (both edges None).  ``rho_outer`` is the outermost upward
    crossing.  Crossings are linear in q between faces
    (:func:`floor_crossings`, also returned).  A q below the floor at the last face is refused
    (:class:`QFloorRefused`).  A face q that is not finite counts as below the floor."""
    qf = _check_q_floor(q_floor)
    rf = np.asarray(rho_faces, float)
    q = np.asarray(q, float)
    if not q[-1] >= qf:
        raise QFloorRefused(f"q = {q[-1]:.4g} at the domain edge is below the floor q_floor = {qf:g}: the core "
                            "cannot be held at the floor")
    cr = floor_crossings(rf, q, qf)
    below = ~(q >= qf)
    if not np.any(below):
        return dict(branch="none", rho_inner=None, rho_outer=None, crossings=cr)
    ups = [c["rho"] for c in cr if c["direction"] == "up"]
    if below[0]:
        return dict(branch="axis", rho_inner=0.0, rho_outer=float(ups[-1]), crossings=cr)
    downs = [c["rho"] for c in cr if c["direction"] == "down"]
    return dict(branch="off_axis", rho_inner=float(downs[0]), rho_outer=float(ups[-1]), crossings=cr)


def _faces_from_cells(rho_cells):
    """Faces of a grid from its cell centers (the centers are face midpoints; the first face is the axis)."""
    rc = np.asarray(rho_cells, float)
    f = np.empty(rc.size + 1)
    f[0] = 0.0
    for i in range(rc.size):
        f[i + 1] = 2.0 * rc[i] - f[i]
    return f


def _overlap_weights(faces, r_in, r_out):
    """Fraction of each cell ``[f_i, f_i+1]`` inside ``[r_in, r_out]``: exactly 1 for a cell wholly inside, exactly 0
    for a cell wholly outside, the overlap over the cell width for a cut cell."""
    f = np.asarray(faces, float)
    lo, hi = f[:-1], f[1:]
    w = np.clip(np.minimum(hi, r_out) - np.maximum(lo, r_in), 0.0, None) / (hi - lo)
    w[(lo >= r_in) & (hi <= r_out)] = 1.0
    w[(hi <= r_in) | (lo >= r_out)] = 0.0
    return w


class ResistiveFloor:
    r"""Resistivity modifier of the resistive q floor (module docstring, eq. F1): inside the flattening region the
    conductivity is the solver's conductivity AT the region's outer edge ``rho_inv``; a cell cut by an edge of the
    region takes the overlap-weighted conductivity (``rule="continuous"``).

    For :class:`~.diffusion.CurrentDiffusion` (``resistivity_modifier``; :meth:`with_solver` attaches it to a copy
    of a solver).  The region is either fixed (``rho_inv=...`` with ``rho_inner``, default 0: the steady state,
    :func:`resistive_floor_steady`) or, with ``rho_inv=None``, decided per diffusion step at the time of the state
    the diffusion passes in (the state at the start of the step; :meth:`region_info`): ``rule="continuous"``, the
    single-slice region of the solver's sources at that time (:func:`resistive_floor_steady`; no change in its
    ``"off_axis"`` and ``"none"`` branches); ``rule="grid"`` (comparison only), FUSE's rule from the state's own q --
    the region from the axis to the outermost FACE where q is below the floor, no change where q >= floor on every
    face.  A state with q below the floor at the domain edge is refused (:class:`QFloorRefused`).  The outer layer is
    never touched (the diffusion's ``resistivity_modifier`` acts on the domain cells only).  ``rho_faces``: the grid's
    faces where no state is passed (the steady state); otherwise the state's own, and without either they are rebuilt
    from the cell centers.

    ``last`` holds the region of the latest state examined (:meth:`region_info`).
    """

    def __init__(self, solver, q_floor=DEFAULT_Q_FLOOR, rho_inv=None, rho_inner=0.0, rho_faces=None,
                 rule="continuous"):
        if rule not in RESISTIVE_FLOOR_RULES:
            raise ValueError(f"rule must be one of {RESISTIVE_FLOOR_RULES}")
        self.solver = solver
        self.q_floor = _check_q_floor(q_floor)
        self.rho_inv = None if rho_inv is None else float(rho_inv)
        self.rho_inner = float(rho_inner)
        self.rho_faces = None if rho_faces is None else np.asarray(rho_faces, float)
        self.rule = rule
        self._seen = None
        self.last = None

    def region_info(self, state):
        """The region at the time of ``state`` under the time-dependent rule: ``dict(k, rho_inv, rho_inner, branch,
        q, ...)`` with ``k`` the number of cells whose resistivity is changed (cut cells included; 0 when none, and
        under the grid rule also when only the extrapolated axis value is below the floor), ``rho_inv`` the outer edge
        (None when ``k`` is 0), ``rho_inner`` the inner edge (0) and ``branch`` one of :data:`FLOOR_BRANCHES`.

        Continuous rule (default): the region is the SINGLE-SLICE region of the solver's sources at the state's time
        -- :func:`resistive_floor_steady` of ``self.solver`` (the sources without the floor) at ``t = state.time`` on
        the state's faces -- so ``branch`` and ``rho_inv`` are those of the steady rule; the state itself is read only
        for the refusal at the domain edge and for the record (``state_branch``, its own :func:`floor_region` branch).
        Also recorded: ``q_axis_model_steady`` and ``q_min_model_steady`` (the plain steady state's axis q and
        minimum), ``q_floor_met_steady`` (the floored steady state meets the floor) and ``n_steady_states`` (the steady
        states the search formed).  Grid rule (comparison): from the state's own q, FUSE's rule as built earlier."""
        if self._seen is not None and self._seen[0] is state:
            return self._seen[1]
        q = face_q(state, self.solver.geometry_at(state.time).phi_b)
        rf = np.asarray(state.rho_faces, float)
        if self.rule == "grid":
            if not q[-1] >= self.q_floor:
                raise QFloorRefused(f"q = {q[-1]:.4g} at the domain edge is below the floor q_floor = "
                                    f"{self.q_floor:g}: the core cannot be held at the floor")
            below = np.flatnonzero(~(q >= self.q_floor))
            k = int(below[-1]) if below.size else 0
            out = dict(k=k, rho_inv=float(rf[k]) if k > 0 else None, rho_inner=0.0,
                       branch="axis" if k > 0 else "none", q=q)
        else:
            # the state's own q: the refusal at the domain edge and its branch, for the record only
            reg = floor_region(rf, q, self.q_floor)
            # the region: the single-slice region of the step's sources at this time (module docstring)
            _, rs, _ = resistive_floor_steady(self.solver, rf, self.q_floor, t=state.time)
            common = dict(state_branch=reg["branch"], q_axis_model_steady=rs["q_axis_before"],
                          q_min_model_steady=rs["q_min_before"], q_floor_met_steady=rs["q_floor_met"],
                          n_steady_states=rs["n_steady_states"], q=q)
            if rs["acted"]:
                w = _overlap_weights(rf, 0.0, rs["rho_floor"])
                out = dict(k=int(np.count_nonzero(w > 0.0)), rho_inv=float(rs["rho_floor"]), rho_inner=0.0,
                           branch=rs["floor_branch"], **common)
            else:
                out = dict(k=0, rho_inv=None, rho_inner=None, branch=rs["floor_branch"], **common)
        self._seen = (state, out)
        self.last = out
        return out

    def region(self, state):
        """``(k, rho_inv)`` of :meth:`region_info`."""
        info = self.region_info(state)
        return info["k"], info["rho_inv"]

    def __call__(self, t, rho_cells, eta, state):
        if self.rho_inv is not None:
            r_in, r_out = self.rho_inner, self.rho_inv
        else:
            if state is None:
                raise ValueError("the time-dependent q floor needs the state at the start of the step")
            info = self.region_info(state)
            if info["rho_inv"] is None:
                return eta
            r_in, r_out = info["rho_inner"], info["rho_inv"]
        rc = np.asarray(rho_cells, float)
        faces = (np.asarray(state.rho_faces, float) if state is not None else
                 (self.rho_faces if self.rho_faces is not None else _faces_from_cells(rc)))
        w = _overlap_weights(faces, r_in, r_out)
        if not np.any(w > 0.0):
            return eta
        eta = np.array(eta, dtype=float)
        s_r = float(np.asarray(self.solver.sigma_fn(t, np.array([r_out])), float).ravel()[0])
        full = w == 1.0
        cut = (w > 0.0) & ~full
        eta[full] = 1.0 / s_r
        eta[cut] = 1.0 / (w[cut] * s_r + (1.0 - w[cut]) / eta[cut])
        return eta

    def with_solver(self):
        """A shallow copy of the solver with this modifier as its ``resistivity_modifier`` (the solver itself is
        not changed); refused when the solver already has a modifier."""
        if self.solver.resistivity_modifier is not None:
            raise ValueError("the solver already has a resistivity modifier")
        s = copy.copy(self.solver)
        s.resistivity_modifier = self
        return s


def _min_face_q(q):
    q = np.asarray(q, float)
    return float(np.min(np.where(np.isfinite(q), q, -np.inf)))


def _floor_effect(solver, st0, sol_f, st_f, r_out):
    """Record of what the floor did against the same sources without it (``st0``, plain resistivity): the loop
    voltage and l_i (``li3``) of both, and the fraction of Ip moved from inside ``r_out`` to outside it (the enclosed
    current at ``r_out``, linear between faces, plain minus floored, over Ip)."""
    p0, p1 = solver.profiles(st0), sol_f.profiles(st_f)
    rf = np.asarray(st0.rho_faces, float)
    ip = float(p0["ip"])
    i0 = float(np.interp(r_out, rf, p0["I_enclosed"]))
    i1 = float(np.interp(r_out, rf, p1["I_enclosed"]))
    v0, v1 = float(st0.meta.get("v_loop", p0["v_loop_boundary"])), float(st_f.meta.get("v_loop", p1["v_loop_boundary"]))
    return dict(v_loop_plain=v0, v_loop_floor=v1, d_v_loop_rel=(v1 / v0 - 1.0) if v0 else float("nan"),
                li3_plain=float(p0["li3"]), li3_floor=float(p1["li3"]), d_li3=float(p1["li3"] - p0["li3"]),
                current_moved_outside_over_ip=(i0 - i1) / ip if ip else float("nan"),
                d_enclosed_current_max_over_ip=(float(np.max(np.abs(np.asarray(p1["I_enclosed"]) -
                                                                    np.asarray(p0["I_enclosed"])))) / ip
                                                if ip else float("nan")))


def resistive_floor_steady(solver, rho_faces, q_floor=DEFAULT_Q_FLOOR, t=0.0, psi_boundary=0.0, rule="continuous"):
    r"""Steady state of ``solver`` with the resistive q floor (module docstring, eq. F1 and the steady-state rule).

    The plain steady state first.  The branch is the one of :func:`floor_region` (``rule="continuous"``, default):
    where the plain q is below ``q_floor`` on axis (:func:`face_q`, the extrapolated axis value), the region starts at
    the axis and its outer edge ``rho_inv`` is the smallest one whose steady state (:class:`ResistiveFloor` with that
    fixed region) has q >= ``q_floor`` on every face, searched from the axis out to the plain state's outermost
    upward crossing; where it is below the floor off axis only, or nowhere, the plain state is returned (recorded as
    ``floor_branch`` ``"off_axis"`` with ``q_floor_met = False``, or ``"none"``).  The search: the faces between the
    axis and that crossing are scanned outward, the first one that meets the floor brackets the edge with the point
    before it, and the edge is bisected inside the bracket to :data:`RADIUS_ROUND_OFF`; the upper end of the final
    bracket (which meets the floor) is taken.  If not even the outermost crossing meets the floor, that crossing is
    taken (as FUSE's actor keeps its outermost node), recorded as ``q_floor_met = False``.  ``rule="grid"``
    (comparison only): FUSE's grid rule as built earlier -- the region from the axis, its edge scanned over the faces
    ``k = 1 ... k_max`` (``k_max`` the outermost face where the plain state's q is below the floor), the first face
    whose steady state meets the floor, else ``k_max``.  Ip is the solver's Ip condition throughout.  Refused
    (:class:`QFloorRefused`): q below the floor at the domain edge.

    Returns ``(state, record, solver_used)``: the state (the plain steady state itself where nothing is flattened),
    ``solver_used`` the solver whose conductivity made it (``solver`` itself, or a copy carrying the modifier).
    ``record``: ``core_relaxation``, ``q_floor_form``, ``q_floor``, ``floor_rule``, ``floor_branch``
    (:data:`FLOOR_BRANCHES`), ``acted`` (the resistivity was changed), ``q_min_before`` and its ``rho``,
    ``q_axis_before``, ``q_edge``, every crossing of the floor by the plain state's q (``rho_floor_crossings``: inner
    and outer edges of each region below the floor), ``rho_floor`` (the outer edge ``rho_inv``; the domain's rho,
    0..1), ``rho_floor_inner`` (the inner edge), ``n_cells_flattened`` (the number of cells whose resistivity was
    changed, cut cells included; 0 when nothing acted) and ``n_cells_cut``, the resistivity ratio
    ``eta(rho_inv)/eta(axis cell)`` of the plain profile, ``q_min_after`` and its rho, ``q_axis_after``,
    ``q_floor_met``, the number of steady states formed, the flux moved on the axis cell, and -- a RECORD of the
    floor's effect against the plain steady state of the same sources -- the loop voltage and ``li3`` of both and their
    changes (``d_v_loop_rel``, ``d_li3``), the fraction of Ip moved from inside ``rho_inv`` to outside it
    (``current_moved_outside_over_ip``) and the largest change of the enclosed current (fraction of Ip).
    """
    if rule not in RESISTIVE_FLOOR_RULES:
        raise ValueError(f"rule must be one of {RESISTIVE_FLOOR_RULES}")
    qf = _check_q_floor(q_floor)
    rf = np.asarray(rho_faces, dtype=float)
    st0 = solver.steady_state(rf, t=t, psi_boundary=psi_boundary)
    phi_b = solver.geometry_at(t).phi_b
    q0 = face_q(st0, phi_b)
    jmin = int(np.argmin(np.where(np.isfinite(q0), q0, np.inf)))
    rec = dict(core_relaxation="q_floor", q_floor_form="resistive", q_floor=qf, floor_rule=rule, floor_branch="none",
               q_min_before=float(q0[jmin]), rho_q_min_before=float(rf[jmin]), q_axis_before=float(q0[0]),
               q_edge=float(q0[-1]), rho_floor_crossings=floor_crossings(rf, q0, qf), acted=False,
               n_cells_flattened=0, n_cells_cut=0, rho_floor=None, rho_floor_inner=None, eta_ratio_inv_to_axis=None,
               q_min_after=float(q0[jmin]), rho_q_min_after=float(rf[jmin]), q_axis_after=float(q0[0]),
               q_floor_met=bool(np.all(q0 >= qf)), n_steady_states=1, d_enclosed_current_max_over_ip=0.0,
               d_psi_axis_cell=0.0, d_v_loop_rel=0.0, d_li3=0.0, current_moved_outside_over_ip=0.0)
    if not q0[-1] >= qf:
        raise QFloorRefused(f"q = {q0[-1]:.4g} at the domain edge is below the floor q_floor = {qf:g}: the core "
                            "cannot be held at the floor")
    if rule == "grid":
        below = np.flatnonzero(~(q0 >= qf))
        k_max = int(below[-1]) if below.size else 0
        if k_max < 1:
            return st0, rec, solver
        st = sol = q = None
        for k in range(1, k_max + 1):
            sol = ResistiveFloor(solver, qf, rho_inv=rf[k], rho_faces=rf).with_solver()
            st = sol.steady_state(rf, t=t, psi_boundary=psi_boundary)
            rec["n_steady_states"] += 1
            q = face_q(st, phi_b)
            if np.all(q >= qf):
                break
        r_in, r_use, branch = 0.0, float(rf[k]), "axis"
    else:
        reg = floor_region(rf, q0, qf)
        if reg["branch"] == "none":
            return st0, rec, solver
        if reg["branch"] == "off_axis":
            # below the floor off axis only: the floor does not act (module docstring, case (b)); recorded
            rec.update(floor_branch="off_axis", q_floor_met=False)
            return st0, rec, solver
        r_in, r_max, branch = 0.0, reg["rho_outer"], "axis"

        def ev(r):
            s_ = ResistiveFloor(solver, qf, rho_inv=r, rho_inner=r_in, rho_faces=rf).with_solver()
            st_ = s_.steady_state(rf, t=t, psi_boundary=psi_boundary)
            rec["n_steady_states"] += 1
            q_ = face_q(st_, phi_b)
            return _min_face_q(q_) - qf, (st_, s_, q_)

        lo, hi, best = r_in, None, None
        for r in [float(x) for x in rf if r_in < x < r_max] + [r_max]:
            fr, cand = ev(r)
            best = cand
            if fr >= 0.0:
                hi = r
                break
            lo = r
        if hi is None:
            r_use = r_max                    # not even the outermost crossing meets the floor: keep it (as FUSE)
        else:
            for _ in range(_RADIUS_BISECTIONS):
                mid = 0.5 * (lo + hi)
                if hi - lo <= RADIUS_ROUND_OFF or not lo < mid < hi:
                    break
                fm, cand = ev(mid)
                if fm >= 0.0:
                    hi, best = mid, cand
                else:
                    lo = mid
            r_use = hi
        st, sol, q = best
    rc = 0.5 * (rf[1:] + rf[:-1])
    w = _overlap_weights(rf, r_in, r_use)
    sig = np.asarray(solver.sigma_fn(t, np.array([rc[0], r_use])), float)
    j1 = int(np.argmin(np.where(np.isfinite(q), q, np.inf)))
    rec.update(acted=True, floor_branch=branch, n_cells_flattened=int(np.count_nonzero(w > 0.0)),
               n_cells_cut=int(np.count_nonzero((w > 0.0) & (w < 1.0))), rho_floor=float(r_use),
               rho_floor_inner=float(r_in), eta_ratio_inv_to_axis=float(sig[0] / sig[1]), q_min_after=float(q[j1]),
               rho_q_min_after=float(rf[j1]), q_axis_after=float(q[0]), q_floor_met=bool(np.all(q >= qf)),
               d_psi_axis_cell=float(st.psi[0] - st0.psi[0]))
    rec.update(_floor_effect(solver, st0, sol, st, r_use))
    st.meta = dict(st.meta or {}, q_floor=dict(q_floor=qf, form="resistive", rule=rule, branch=branch, acted=True,
                                               rho_floor=float(r_use), rho_floor_inner=float(r_in),
                                               q_floor_met=rec["q_floor_met"]))
    return st, rec, sol


def floor_q(state, solver, q_floor=DEFAULT_Q_FLOOR):
    r"""Sharp q floor (opt-in ``q_floor_form="sharp"``; the default is the resistive floor, module docstring): q
    clamped to ``q_floor`` wherever the state's own q is below it.

    On every interior face where ``q = Phi_b rho / (pi psi')`` (positive frame,
    :meth:`.diffusion.CurrentDiffusion.profiles`) is below ``q_floor``, ``psi'`` is replaced by
    ``Phi_b rho / (pi q_floor)``; ``psi`` is rebuilt from the outermost cell inward, so the outermost cell value and
    ``psi_boundary`` (the boundary gauge) are kept.  The enclosed current ``I = k psi' F`` changes only on the floored
    faces: the current inside a floored region is flattened and its total returns at the outer face of the region, so
    ``Ip`` is conserved without a separate correction (``q I`` is a function of rho alone on fixed surfaces).
    ``q_new = max(q, q_floor)`` on every interior face, so q is continuous across the floor radius (the slope breaks
    there) and ``<J.B>`` is NOT: it jumps at the floor radius (the flat-q core against the unflattened shear outside;
    a drop of about a third of the local value on the synthetic test fixture).  It is the identity on a state with
    ``q >= q_floor`` on every interior face.

    Refused (:class:`QFloorRefused`): q below the floor at the domain edge, or on the last interior face (the second-
    order boundary gradient uses the two outer cells, so flooring there would change the edge current).

    Returns ``(state, record)`` (the input state itself when nothing changes).  ``record``: ``acted`` (the state
    changed), ``q_floor``, ``q_min_before`` and its ``rho``, ``q_edge``, the number of floored faces
    (``n_faces_floored``; 0 when nothing changed) and of faces below the floor before (``n_faces_below_floor``, which
    counts faces at the floor to round-off in a state floored before), every crossing of the floor by the state's q
    (``rho_floor_crossings``: inner and outer edges of each region, reversed shear included), ``rho_floor``
    (the outer crossing ``q = q_floor`` of the outermost floored region, interpolated linearly in q between the last
    floored face and the next face; the domain's rho, 0..1), the largest change of the enclosed current as a fraction
    of ``Ip`` and the flux moved on the axis cell [Wb/rad].
    """
    qf = _check_q_floor(q_floor)
    rf = np.asarray(state.rho_faces, dtype=float)
    rc = state.rho_cells
    n = rc.size
    prof = solver.profiles(state)
    q = np.asarray(prof["q"], dtype=float)
    i_f = np.asarray(prof["I_enclosed"], dtype=float)
    ip = float(prof["ip"])
    phi_b = solver.geometry_at(state.time).phi_b
    dpsi = np.zeros(n + 1)
    dpsi[1:n] = np.diff(state.psi) / np.diff(rc)
    inner = np.arange(1, n)
    below = ~(q[inner] >= qf)
    jmin = int(np.argmin(np.where(np.isfinite(q), q, np.inf)))
    rec = dict(core_relaxation="q_floor", q_floor_form="sharp", q_floor=qf, q_min_before=float(q[jmin]),
               rho_q_min_before=float(rf[jmin]), q_edge=float(q[n]), acted=bool(np.any(below)),
               n_faces_floored=int(np.count_nonzero(below)), n_faces_below_floor=int(np.count_nonzero(below)),
               rho_floor_crossings=floor_crossings(rf, q, qf), rho_floor=None, d_enclosed_current_max_over_ip=0.0,
               d_psi_axis_cell=0.0)
    if not q[n] >= qf or not q[n - 1] >= qf:
        raise QFloorRefused(f"q = {q[n]:.4g} at the domain edge and {q[n - 1]:.4g} on the last interior face: below "
                            f"the floor q_floor = {qf:g} there, so the core cannot be floored at fixed edge current")
    if not rec["acted"]:
        return state, rec
    fl = inner[below]
    new = dpsi.copy()
    new[fl] = phi_b * rf[fl] / (np.pi * qf)
    psi = np.empty(n)
    psi[-1] = state.psi[-1]
    psi[:-1] = state.psi[-1] - np.cumsum((new[1:n] * np.diff(rc))[::-1])[::-1]
    if np.array_equal(psi, state.psi):
        # faces already at the floor to round-off (e.g. a state floored before): nothing changes
        rec.update(acted=False, n_faces_floored=0)
        return state, rec
    k = int(fl[-1])
    q0_, q1_ = q[k], q[k + 1]
    rho1 = float(rf[k] + (qf - q0_) * (rf[k + 1] - rf[k]) / (q1_ - q0_)) if q1_ != q0_ else float(rf[k])
    with np.errstate(divide="ignore", invalid="ignore"):
        di = np.where(dpsi[fl] != 0.0, i_f[fl] * (new[fl] / dpsi[fl] - 1.0), np.nan)
    rec.update(rho_floor=rho1, d_psi_axis_cell=float(psi[0] - state.psi[0]),
               d_enclosed_current_max_over_ip=(float(np.nanmax(np.abs(di))) / ip if ip else float("nan")))
    meta = dict(state.meta or {})
    meta["q_floor"] = dict(q_floor=qf, form="sharp", acted=True, rho_floor=rho1)
    return CurrentState(rf, psi, state.psi_boundary, state.time, meta), rec


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
    Locality residuals: ``psi_outside_max_change`` (cells outside the changed zone),
    ``current_outside_max_rel_change`` (relative change of dpsi/drho, i.e. of the enclosed current, on faces
    between unchanged cells), ``psi_boundary_change``.  They are exactly 0 by construction (the operator copies
    those cells), so they check the zone bookkeeping, not conservation by the rearrangement itself; toroidal flux
    is conserved by construction (fixed grid).  Helical flux is not recorded here (see (K5)).
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
def _check_positive_frame(cfac, depth):
    """Refuse a state outside the positive frame (Phi_b > 0 and psi rising from the axis to the last cell).

    The operator, its record and the cycle criterion (PO) assume ``psi' > 0`` overall (Ip > 0, module
    docstring of :mod:`.diffusion`).  With Ip < 0 the helical flux ``chi = psi - c s`` falls everywhere, the
    operator would be silently inactive, and a residual divided by a negative depth would pass any test.
    """
    if not (np.isfinite(cfac) and cfac > 0):
        raise PositiveFrameError(f"boundary toroidal flux must be positive (Phi_b/2pi = {cfac!r}): map the "
                                 "discharge onto the positive frame first")
    if not (np.isfinite(depth) and depth > 0):
        raise PositiveFrameError(f"psi(last cell) - psi(axis) = {depth!r} <= 0: the state is not in the positive "
                                 "frame (Ip > 0, psi' > 0); map the discharge onto it first (reverse the signs of Ip "
                                 "and psi), or the current is reversed over most of the plasma")


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
    _check_positive_frame(cfac, float(psi[-1] - psi_axis))
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
    # the axis-level sheet: when the minimum of chi on [0, s_*] is interior and D starts at the axis, the axis
    # crossing (slope m0 = chi'(0) != 0) ends at the level L0 = chi(0); chi_new has a slope jump at the position
    # s_lo + mu(L0) of that level (S(L) changes by 1/|m0| there).  Levels above a position lie to its left
    # (chi_new decreasing), so for m0 > 0 (rising axis piece: crossings exist above L0) the left slope includes
    # the axis crossing; for m0 < 0 (falling axis piece: crossings below L0) the right slope does.
    if s_lo == 0.0 and not min_at_axis and m[0] != 0.0 and m_star < chi[0] < M:
        L0 = float(chi[0])
        S_reg0 = 0.0
        for jj, pc in enumerate(pieces):
            if pc.flat or jj == 0:
                continue
            if pc.lo < L0 < pc.hi:
                d0 = abs(float(H.deriv(pc.crossing(np.array([L0]))[0])))
                S_reg0 += 1.0 / d0 if d0 > 0 else np.inf
        if np.isfinite(S_reg0) and S_reg0 > 0:
            with_axis, without = -1.0 / (S_reg0 + 1.0 / abs(m[0])), -1.0 / S_reg0
            jump = (without - with_axis) if m[0] > 0 else (with_axis - without)
            sheets.append((float(s_lo + mu(np.array([L0]))[0]), jump))
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
    ``t_ref`` to ``t_ref + period``.  ``residual`` is (PO) for the matrix map, ``residual_stepping`` for the
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


def _anderson(resid, z, nrm, tol, max_evals, hist, verbose, depth=5):
    """Anderson acceleration (type II, depth ``depth``) of the physical map z -> z + r(z).  An accelerated step
    is accepted only if it lowers the residual measure below the current one; otherwise the history is
    cleared and a plain step (one physical cycle) is taken, which is always accepted.  Returns
    (z, residual, n_evals, y, info) for the best iterate.  The stopping test is the unchanged (PO)."""
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
            n_ev += 1
            try:
                rg, yg, infg, ng = resid(g)
            except ValueError as exc:
                # the physical map itself failed (e.g. the zone reaches the last two cells): stop and report
                # the best iterate as not converged instead of aborting the solve
                if verbose:
                    print(f"  anderson {n_ev}: plain step failed ({exc}); stopping")
                break
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
    criterion (PO); it may be made stricter but not looser than :data:`CYCLE_TOL`.  Non-convergence is
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
    ip_ref = float(solver.boundary(t_ref))
    if not ip_ref > 0:
        raise PositiveFrameError(f"Ip = {ip_ref!r} at t_ref: the periodic orbit needs Ip > 0 (positive frame)")
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
        return r, y, info, float(np.max(np.abs(r)) / abs(info["depth"]))

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
            h = 1e-6 * abs(info["depth"]) / np.max(np.abs(u))
            try:
                d = (K(y + h * u)[0] - K(y - h * u)[0]) / (2.0 * h) - u
            except ValueError:
                continue
            DM += np.outer(d, sv[kk] * Vt[kk])
        J = M + DM
        return J - J[-1:, :]

    # default initial guess: the relaxed (steady) state with q floored at 1
    z = _gauge(floor_q(solver.steady_state(rf, t=t_ref), solver, 1.0)[0].psi if z0 is None
               else np.asarray(z0, dtype=float))
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
    res_step = float(np.max(np.abs(z_step - z)) / abs(rec.depth))
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
    applied.  ``events`` lists every crash applied in the CURRENT run with its
    :class:`ReconnectionRecord`: :meth:`reset` (called by :func:`advance_with_crashes` at the start of
    every run) clears it, so a schedule reused for another run reports only that run's crashes.
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
        """Start a run at ``t0``: crashes after ``t0`` are pending; the event list is cleared."""
        self._next = int(np.searchsorted(self.times, t0, side="right"))
        self.events = []

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
    The starting phase is capped at 1 (the pre-crash state) when the last crash lies more than one mean
    interval before the start; the cap is reported (``initial_phase_capped``, ``initial_phase_raw``).
    Returns dict(trajectory, events, orbit, initial_phase, state, initial_phase_capped, initial_phase_raw)."""
    if sched.mean_interval is None:
        raise ValueError("need at least two crashes to set the starting orbit")
    tau = sched.mean_interval
    orbit = solve_periodic_orbit(solver, rho_faces, f, tau, nsteps=nsteps_orbit, sheet_width=sheet_width,
                                 z0=orbit_z0, scheme=scheme)
    phase, phase_raw = 0.0, None
    if sched.last_crash_before_start is not None:
        phase_raw = (sched.t_start - sched.last_crash_before_start) / tau
        phase = min(phase_raw, 1.0)
    s0 = orbit.phase_state(phase)
    s0 = CurrentState(s0.rho_faces, s0.psi, s0.psi_boundary, sched.t_start, dict(s0.meta))
    traj = Trajectory(*sched.window)
    sch = CrashSchedule(sched.crash_times, f, sheet_width, trajectory=traj)
    s = advance_with_crashes(solver, s0, sched.window[1], sch, dt, scheme=scheme, stops=(sched.window[0],))
    return dict(trajectory=traj, events=sch.events, orbit=orbit, initial_phase=phase, state=s,
                initial_phase_capped=bool(phase_raw is not None and phase_raw > 1.0), initial_phase_raw=phase_raw)


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
