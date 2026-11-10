r"""1-D flux-surface-averaged poloidal-flux (current) diffusion.

Formulation
-----------
Field ``B = grad(phi) x grad(psi) + F grad(phi)`` with ``psi`` the poloidal flux
per radian [Wb/rad] (Lyons eq. 1).  In flux coordinates that keep the
toroidal flux Phi fixed, Faraday's law and a parallel Ohm's law
``E + u x B = eta_par (J - J_NI)`` give (Lyons eqs. 5, 23, 25; Jardin 2010,
surface-averaged poloidal-flux equation)

.. code-block:: text

    dpsi/dt |_Phi = <B . R> / <B . grad phi> = eta_par (<J.B> - <J_NI.B>) / (F <1/R^2>)    (1)
    V_loop(rho)   = 2 pi dpsi/dt |_Phi                                                      (2)

Ampere's law averaged over a surface (Lyons eq. 28 with ``Phi' iota / 2pi =
dpsi/drho``) gives, with ``V' = dV/drho`` and ``g2 = <|grad rho|^2/R^2>``,

.. code-block:: text

    I(rho)  = V' g2 psi' / (2 pi mu0)                 enclosed toroidal current          (3)
    <J.B>   = (2 pi F^2 / V') d/drho ( I / F )                                           (4)
    <J_phi/R> = (2 pi / V') dI/drho                                                       (5)

Substituting (3)-(4) in (1) gives the conservative diffusion equation solved
here, on ``rho = rho_tor_norm`` (``' = d/drho``):

.. code-block:: text

    a dpsi/dt|_Phi = d/drho ( k psi' ) - S                                               (6)
    a = V' <1/R^2> sigma_par / (2 pi F),   k = V' g2 / (2 pi mu0 F) = I / (F psi'),
    S = V' <J_NI.B> / (2 pi F^2)

``k psi' = I/F`` is the flux of (6), so the Ip boundary condition is a flux
(Neumann) condition and the total current is conserved by construction.  If
the boundary toroidal flux Phi_b changes in time, the state is held on fixed
``rho`` and (Felici et al. 2011, RAPTOR form)

.. code-block:: text

    dpsi/dt |_rho = dpsi/dt |_Phi + (rho dPhi_b/dt / (2 Phi_b)) psi'                     (7)

Boundary conditions: ``k psi' = 0`` on axis (``I(0) = 0``; also ``V'(0) = 0``);
at ``rho = 1`` either ``I(1) = Ip(t)`` or ``dpsi_b/dt|_Phi = V_loop(t) / 2pi``.

Steady state ("fully relaxed"): ``dpsi/dt|_Phi = V/2pi`` uniform in rho.  Then
(6) integrates to ``I/F = int_0^rho (a V/2pi + S) drho'``; ``V`` follows from
``I(1) = Ip`` (or ``Ip`` from a given ``V``).  This is the discrete
fixed point of the time-dependent scheme (it is integrated exactly by every
one-step method used here), so steady state equals the long-time limit to
round-off.

Sign convention ("positive frame"): ``F > 0`` and positive ``Ip`` gives
``psi' > 0``, ``q > 0`` and positive co-current ``<J.B>``.  All current inputs
(``<J_NI.B>``) are signed relative to the plasma current, and ``V_loop > 0``
drives co-current.  A discharge with Ip anti-parallel to B_phi maps onto
this frame by the parity symmetry of (1)-(7); reverse signs on output if a
COCOS-signed result is needed.

Units: SI, ``psi`` in Wb/rad, ``mu0 = 4 pi 1e-7``.

Discretization (choice justified in the P0c report)
---------------------------------------------------
Cell-centered conservative finite volumes on faces ``rho_f`` (uniform by
default, any strictly increasing set from 0 to 1): the unknowns are cell
values ``psi_c`` (plus ``psi_b`` at rho = 1 under the loop-voltage condition).
Face fluxes are ``k_f (psi_{c+1} - psi_c)/(rho_{c+1} - rho_c)``; cell
integrals of ``a`` and ``S`` use the midpoint rule.  Second order in space,
including the axis (the cylindrical singularity is handled by ``k(0) = 0``).
At ``rho = 1`` the gradient is the second-order one-sided formula through
``psi_b`` and the two outer cell values (``(8, -9, 1)/(3h)`` on uniform cells),
so ``Ip`` under the loop-voltage condition, and ``psi_b`` under the Ip
condition, are second order as well.
Time: TR-BDF2 (L-stable, second order; default), or the theta method
(``theta = 1`` backward Euler, ``theta = 0.5`` Crank-Nicolson as in QED.jl).
Crank-Nicolson is not L-stable: with a resistive edge (large eta, fine grid)
and a step much longer than the edge diffusion time it leaves undamped
oscillations in ``<J.B>`` at the edge (measured in the P0c benchmark), which
is why TR-BDF2 is the default.
Every implicit stage is one tridiagonal solve; coefficients are re-evaluated
at each stage time, so geometry, conductivity, non-inductive current and the
boundary value may all vary within a step.

Sawtooth hook (no model is implemented here)
--------------------------------------------
* ``post_step(state, t, solver) -> state`` is applied after every step, e.g.
  a crash operator acting on ``state.psi``;
* ``resistivity_modifier(t, rho, eta, state) -> eta`` modifies the resistivity
  used within the step; it is evaluated with the state at the start of the
  step for every stage, so each step stays linear.

Credits
-------
Formulation after B. C. Lyons, "Neoclassical current diffusion solver"
(``equations_readme.pdf`` in QED.jl, General Atomics, Apache-2.0) and S. C.
Jardin, *Computational Methods in Plasma Physics* (CRC Press, 2010); the
moving-boundary term (7) after F. Felici et al., Nucl. Fusion 51, 083052
(2011).  The equations were re-derived (see the P0c report) and the code is
written independently; QED.jl is used only as a numerical benchmark.  The
conductivity model is in :mod:`.conductivity` (Sauter 1999, Redl 2021).
TR-BDF2: R. E. Bank et al., IEEE Trans. CAD 4, 436 (1985); M. E. Hosea and
L. F. Shampine, Appl. Numer. Math. 20, 21 (1996).
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded

from .geometry import MU0, FluxSurfaceGeometry

__all__ = [
    "CurrentState",
    "IpBoundary",
    "VloopBoundary",
    "GeometrySchedule",
    "CurrentDiffusion",
    "as_profile_function",
    "as_time_function",
]

_TRBDF2_GAMMA = 2.0 - np.sqrt(2.0)


# ====================================================================== inputs
def as_time_function(spec):
    """Scalar time function from a constant, a callable ``f(t)`` or ``(times, values)`` (linear)."""
    if callable(spec):
        return spec
    if np.isscalar(spec):
        v = float(spec)
        return lambda t: v
    times, values = spec
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must increase strictly")
    return lambda t: float(np.interp(t, times, values))


def as_profile_function(spec, log=False):
    """Radial profile ``f(t, rho)`` from several input forms.

    * ``None`` -> zero; a scalar -> constant;
    * a callable ``f(t, rho)`` -> used as is;
    * ``(rho, values)`` -> PCHIP in rho, constant in time;
    * ``(times, rho, values[n_t, n_rho])`` -> PCHIP in rho, linear in time.

    ``log=True`` interpolates ``log(values)`` (for conductivity, which varies
    by orders of magnitude near the edge), as QED.jl does for resistivity.
    """
    if spec is None:
        return lambda t, rho: np.zeros_like(np.asarray(rho, dtype=float))
    if callable(spec):
        return spec
    if np.isscalar(spec):
        v = float(spec)
        return lambda t, rho: np.full_like(np.asarray(rho, dtype=float), v)
    fwd = np.log if log else (lambda x: x)
    inv = np.exp if log else (lambda x: x)
    if len(spec) == 2:
        rho_in, vals = (np.asarray(x, dtype=float) for x in spec)
        p = PchipInterpolator(rho_in, fwd(vals), extrapolate=True)
        return lambda t, rho: inv(p(rho))
    if len(spec) == 3:
        times, rho_in, vals = (np.asarray(x, dtype=float) for x in spec)
        if vals.shape != (times.size, rho_in.size):
            raise ValueError("values must have shape (n_times, n_rho)")
        ps = [PchipInterpolator(rho_in, fwd(v), extrapolate=True) for v in vals]

        def f(t, rho):
            if t <= times[0]:
                return inv(ps[0](rho))
            if t >= times[-1]:
                return inv(ps[-1](rho))
            k = int(np.searchsorted(times, t)) - 1
            w = (t - times[k]) / (times[k + 1] - times[k])
            return inv((1.0 - w) * ps[k](rho) + w * ps[k + 1](rho))

        return f
    raise ValueError("unrecognized profile specification")


@dataclass
class IpBoundary:
    """Total plasma current boundary condition I(rho = 1) = Ip(t) [A] (positive frame)."""

    ip: object

    def __post_init__(self):
        self._f = as_time_function(self.ip)

    def __call__(self, t):
        return self._f(t)


@dataclass
class VloopBoundary:
    """Edge loop-voltage boundary condition 2 pi dpsi_b/dt|_Phi = V(t) [V] (positive drives co-current)."""

    vloop: object

    def __post_init__(self):
        self._f = as_time_function(self.vloop)

    def __call__(self, t):
        return self._f(t)


class GeometrySchedule:
    """Time-dependent geometry: piecewise-linear interpolation between snapshots.

    Each snapshot is resampled to the first snapshot's rho grid; profiles and
    scalars are interpolated linearly in time and ``dphib_dt`` is the slope of
    Phi_b on the bracketing interval (overriding the snapshots' own value).
    Outside the time range the end snapshots are held (``dphib_dt = 0``).
    """

    def __init__(self, times, geometries):
        self.times = np.asarray(times, dtype=float)
        if self.times.size != len(geometries) or np.any(np.diff(self.times) <= 0):
            raise ValueError("need one geometry per strictly increasing time")
        rho = geometries[0].rho
        self.geoms = [g if np.array_equal(g.rho, rho) else g.resample(rho) for g in geometries]

    def __call__(self, t):
        ts, gs = self.times, self.geoms
        if t < ts[0] or len(gs) == 1:
            return _with_dphib(gs[0], 0.0)
        if t > ts[-1]:
            return _with_dphib(gs[-1], 0.0)
        # closed intervals: at a snapshot time the slope of the adjacent interval is used
        # (at ts[-1] the last interval's), so a step ending exactly on a snapshot sees dPhi_b/dt
        k = min(max(int(np.searchsorted(ts, t, side="right")) - 1, 0), len(ts) - 2)
        w = (t - ts[k]) / (ts[k + 1] - ts[k])
        g0, g1 = gs[k], gs[k + 1]
        d0, d1 = g0.to_dict(), g1.to_dict()
        out = {}
        for key, v0 in d0.items():
            v1 = d1[key]
            if key == "meta":
                out[key] = dict(v0)
            elif v0 is None or v1 is None:
                out[key] = None
            elif key == "rho":
                out[key] = v0
            else:
                out[key] = (1.0 - w) * np.asarray(v0, dtype=float) + w * np.asarray(v1, dtype=float)
        out["dphib_dt"] = (g1.phi_b - g0.phi_b) / (ts[k + 1] - ts[k])
        for key in ("phi_b", "B0", "R0"):
            out[key] = float(out[key])
        return FluxSurfaceGeometry.from_dict(out)


def _with_dphib(g, val):
    d = g.to_dict()
    d["dphib_dt"] = val
    return FluxSurfaceGeometry.from_dict(d)


def _boundary_gradient_weights(rc):
    """Weights (w_b, w_1, w_2) of the second-order one-sided derivative at rho = 1.

    psi'(1) = w_b psi_b + w_1 psi_{n-1} + w_2 psi_{n-2}, from the parabola through
    (1, psi_b), (rc[-1], psi_{n-1}), (rc[-2], psi_{n-2}); with d1 = 1 - rc[-1], d2 = 1 - rc[-2]:
    w_b = (d1 + d2)/(d1 d2), w_1 = -d2/(d1 (d2 - d1)), w_2 = d1/(d2 (d2 - d1)).  Uniform cells of
    width h: (8, -9, 1)/(3h).
    """
    d1, d2 = 1.0 - rc[-1], 1.0 - rc[-2]
    return (d1 + d2) / (d1 * d2), -d2 / (d1 * (d2 - d1)), d1 / (d2 * (d2 - d1))


def _psi_boundary_from_flux(psi, rc, dpsi_b):
    """psi_b such that the one-sided derivative at rho = 1 equals ``dpsi_b``."""
    wb, w1, w2 = _boundary_gradient_weights(rc)
    return float((dpsi_b - w1 * psi[-1] - w2 * psi[-2]) / wb)


# ====================================================================== state
@dataclass
class CurrentState:
    """Persistent state of the diffusion: psi on a fixed rho_tor_norm finite-volume grid.

    ``rho_faces`` (n+1): cell faces, 0 ... 1.  ``psi`` (n): cell values of the
    poloidal flux per radian [Wb/rad] at the cell midpoints.  ``psi_boundary``:
    psi at rho = 1.  ``time`` [s].  Only differences of psi are physical; the
    offset is carried so that ``psi(t)`` histories (and hence V_loop) are
    meaningful across steps and slices.
    """

    rho_faces: np.ndarray
    psi: np.ndarray
    psi_boundary: float
    time: float = 0.0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.rho_faces = np.array(self.rho_faces, dtype=float)
        self.psi = np.array(self.psi, dtype=float)
        f = self.rho_faces
        if abs(f[0]) > 1e-14 or abs(f[-1] - 1.0) > 1e-14 or np.any(np.diff(f) <= 0):
            raise ValueError("rho_faces must increase strictly from 0 to 1")
        if self.psi.size != f.size - 1:
            raise ValueError("psi must have one value per cell (len(rho_faces) - 1)")
        if not np.all(np.isfinite(self.psi)):
            raise ValueError("psi contains non-finite values")
        self.psi_boundary = float(self.psi_boundary)
        self.time = float(self.time)

    @property
    def rho_cells(self):
        return 0.5 * (self.rho_faces[1:] + self.rho_faces[:-1])

    def copy(self):
        return CurrentState(self.rho_faces.copy(), self.psi.copy(), self.psi_boundary, self.time, dict(self.meta))

    # -------------------------------------------------------------- serialization
    def to_dict(self):
        """JSON-friendly dict (lists of floats; meta must be JSON-serializable)."""
        return dict(rho_faces=self.rho_faces.tolist(), psi=self.psi.tolist(), psi_boundary=self.psi_boundary,
                    time=self.time, meta=dict(self.meta), format="bouquet.kinrecon.current.CurrentState/1")

    @classmethod
    def from_dict(cls, d):
        return cls(rho_faces=d["rho_faces"], psi=d["psi"], psi_boundary=d["psi_boundary"], time=d["time"],
                   meta=dict(d.get("meta", {})))

    def save(self, path):
        """Write to ``.json``, ``.npz`` or ``.h5``/``.hdf5`` (chosen by suffix); round-trips exactly."""
        path = str(path)
        if path.endswith(".json"):
            with open(path, "w") as fh:
                json.dump(self.to_dict(), fh)
        elif path.endswith(".npz"):
            np.savez(path, rho_faces=self.rho_faces, psi=self.psi, psi_boundary=self.psi_boundary, time=self.time,
                     meta=json.dumps(self.meta))
        elif path.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(path, "w") as fh:
                fh["rho_faces"] = self.rho_faces
                fh["psi"] = self.psi
                fh.attrs["psi_boundary"] = self.psi_boundary
                fh.attrs["time"] = self.time
                fh.attrs["meta"] = json.dumps(self.meta)
                fh.attrs["format"] = "bouquet.kinrecon.current.CurrentState/1"
        else:
            raise ValueError("unknown suffix; use .json, .npz or .h5")

    @classmethod
    def load(cls, path):
        path = str(path)
        if path.endswith(".json"):
            with open(path) as fh:
                return cls.from_dict(json.load(fh))
        if path.endswith(".npz"):
            with np.load(path) as z:
                return cls(z["rho_faces"], z["psi"], float(z["psi_boundary"]), float(z["time"]),
                           json.loads(str(z["meta"])))
        if path.endswith((".h5", ".hdf5")):
            import h5py
            with h5py.File(path, "r") as fh:
                return cls(fh["rho_faces"][()], fh["psi"][()], float(fh.attrs["psi_boundary"]),
                           float(fh.attrs["time"]), json.loads(fh.attrs["meta"]))
        raise ValueError("unknown suffix; use .json, .npz or .h5")

    # -------------------------------------------------------------- constructors
    @staticmethod
    def uniform_faces(n_cells):
        return np.linspace(0.0, 1.0, int(n_cells) + 1)

    @classmethod
    def from_psi(cls, rho, psi, rho_faces, time=0.0):
        """Sample a psi(rho) profile [Wb/rad] (PCHIP) onto the cells and the boundary."""
        p = PchipInterpolator(np.asarray(rho, dtype=float), np.asarray(psi, dtype=float), extrapolate=True)
        rf = np.asarray(rho_faces, dtype=float)
        rc = 0.5 * (rf[1:] + rf[:-1])
        return cls(rf, p(rc), float(p(1.0)), time)

    @classmethod
    def from_q(cls, geometry, rho, q, rho_faces, psi_axis=0.0, time=0.0):
        r"""State from a safety-factor profile (positive frame).

        ``psi' = Phi_b rho / (pi q)`` (from ``q = dPhi/drho / (2 pi psi')`` and
        ``Phi = Phi_b rho^2``), integrated from ``psi(0) = psi_axis`` with the
        PCHIP antiderivative of ``rho / q``.
        """
        rho = np.asarray(rho, dtype=float)
        integrand = geometry.phi_b * rho / (np.pi * np.asarray(q, dtype=float))
        anti = PchipInterpolator(rho, integrand, extrapolate=True).antiderivative()
        rf = np.asarray(rho_faces, dtype=float)
        rc = 0.5 * (rf[1:] + rf[:-1])
        return cls(rf, psi_axis + anti(rc) - anti(0.0), psi_axis + float(anti(1.0) - anti(0.0)), time)


# ====================================================================== solver
@dataclass
class _Coeffs:
    t: float
    k_f: np.ndarray      # face conductance k = V' g2 / (2 pi mu0 F)            (n+1)
    F_f: np.ndarray      # F at faces                                         (n+1)
    m_c: np.ndarray      # cell mass  int a drho, a = V' <1/R^2> sigma /(2 pi F)  (n)
    s_c: np.ndarray      # cell source int V' <J_NI.B> /(2 pi F^2) drho          (n)
    w_c: np.ndarray      # advection speed rho dPhi_b/dt /(2 Phi_b) at cells    (n)
    w_b: float           # same at rho = 1
    sigma_c: np.ndarray
    jni_c: np.ndarray
    geo_c: Dict[str, np.ndarray]
    geo: FluxSurfaceGeometry
    bc_value: float


class CurrentDiffusion:
    r"""Solver for equation (6) of the module docstring.

    Parameters
    ----------
    geometry
        :class:`FluxSurfaceGeometry` (static), :class:`GeometrySchedule`, or a
        callable ``t -> FluxSurfaceGeometry``.
    conductivity
        Parallel conductivity sigma_par(t, rho) [1/(Ohm m)] in any form
        accepted by :func:`as_profile_function` (arrays are interpolated in
        log space).  See :mod:`.conductivity` for Spitzer/Redl values.
    boundary
        :class:`IpBoundary` or :class:`VloopBoundary`.
    j_ni
        ``<J_NI . B>`` (t, rho) [A T / m^2], signed relative to Ip
        (bootstrap + driven currents).  Default zero.
    resistivity_modifier
        Optional ``(t, rho_cells, eta, state) -> eta`` (sawtooth hook; see
        module docstring).
    """

    def __init__(self, geometry, conductivity, boundary, j_ni=None, resistivity_modifier=None):
        if isinstance(geometry, FluxSurfaceGeometry):
            self._geo_fn = None
            self._geo_static = geometry
        elif callable(geometry):
            self._geo_fn = geometry
            self._geo_static = None
        else:
            raise TypeError("geometry must be a FluxSurfaceGeometry or a callable t -> FluxSurfaceGeometry")
        self.sigma_fn = as_profile_function(conductivity, log=True)
        self.jni_fn = as_profile_function(j_ni)
        if not isinstance(boundary, (IpBoundary, VloopBoundary)):
            raise TypeError("boundary must be IpBoundary or VloopBoundary")
        self.boundary = boundary
        self.resistivity_modifier = resistivity_modifier
        self._static_cache = {}

    @property
    def ip_mode(self):
        return isinstance(self.boundary, IpBoundary)

    def geometry_at(self, t):
        return self._geo_static if self._geo_fn is None else self._geo_fn(t)

    # -------------------------------------------------------------- coefficients
    def coefficients(self, t, rho_faces, state=None):
        rf = np.asarray(rho_faces, dtype=float)
        rc = 0.5 * (rf[1:] + rf[:-1])
        h = np.diff(rf)
        key = rf.tobytes()
        if self._geo_fn is None and key in self._static_cache:
            geo, gf, gc = self._static_cache[key]
        else:
            geo = self.geometry_at(t)
            gf = geo.evaluate(rf, ["dvol_drho", "avg_grad_rho2_r2", "fpol"])
            gf["dvol_drho"][0] = 0.0
            gc = geo.evaluate(rc)
            if self._geo_fn is None:
                self._static_cache[key] = (geo, gf, gc)
        sigma = np.asarray(self.sigma_fn(t, rc), dtype=float)
        if self.resistivity_modifier is not None:
            sigma = 1.0 / np.asarray(self.resistivity_modifier(t, rc, 1.0 / sigma, state), dtype=float)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("conductivity must be positive and finite")
        jni = np.asarray(self.jni_fn(t, rc), dtype=float)
        Fc, Vc = gc["fpol"], gc["dvol_drho"]
        k_f = gf["dvol_drho"] * gf["avg_grad_rho2_r2"] / (2.0 * np.pi * MU0 * gf["fpol"])
        m_c = Vc * gc["avg_inv_r2"] * sigma / (2.0 * np.pi * Fc) * h
        s_c = Vc * jni / (2.0 * np.pi * Fc ** 2) * h
        wfac = geo.dphib_dt / (2.0 * geo.phi_b)
        return _Coeffs(t=t, k_f=k_f, F_f=gf["fpol"], m_c=m_c, s_c=s_c, w_c=wfac * rc, w_b=wfac, sigma_c=sigma,
                       jni_c=jni, geo_c=gc, geo=geo, bc_value=self.boundary(t))

    # -------------------------------------------------------------- linear operator
    def _operator(self, c, rf):
        """Tridiagonal A (banded, shape (3, m)) and vector b with dy/dt = A y + b."""
        n = rf.size - 1
        rc = 0.5 * (rf[1:] + rf[:-1])
        ip_mode = self.ip_mode
        m = n if ip_mode else n + 1
        lower = np.zeros(m)   # A[i, i-1]
        diag = np.zeros(m)
        upper = np.zeros(m)   # A[i, i+1]
        b = np.zeros(m)
        d = np.diff(rc)                    # center distances (n-1)
        cond = c.k_f[1:n] / d              # interior face conductances
        # diffusion, interior faces
        diag[:n - 1] -= cond / c.m_c[:n - 1]
        upper[:n - 1] += cond / c.m_c[:n - 1]
        diag[1:n] -= cond / c.m_c[1:n]
        lower[1:n] += cond / c.m_c[1:n]
        b[:n] -= c.s_c / c.m_c
        delta = 1.0 - rc[-1]
        wb, w1, w2 = _boundary_gradient_weights(rc)
        if ip_mode:
            gam_b = c.bc_value / c.F_f[-1]          # I/F at the boundary face
            b[n - 1] += gam_b / c.m_c[n - 1]
        else:
            # boundary flux k_b psi'(1), psi'(1) from the second-order one-sided formula
            kb = c.k_f[-1] / c.m_c[n - 1]
            upper[n - 1] += kb * wb
            diag[n - 1] += kb * w1
            lower[n - 1] += kb * w2
            b[n] += c.bc_value / (2.0 * np.pi)
        # moving-boundary advection, eq. (7): + w psi'
        if np.any(c.w_c != 0.0):
            w = c.w_c
            # axis cell: mirror ghost psi(-rho_0) = psi_0
            g0 = w[0] / (rc[1] + rc[0])
            diag[0] -= g0
            upper[0] += g0
            for i in range(1, n - 1):
                gi = w[i] / (rc[i + 1] - rc[i - 1])
                upper[i] += gi
                lower[i] -= gi
            i = n - 1
            gi = w[i] / (1.0 - rc[i - 1])
            lower[i] -= gi
            if ip_mode:   # psi_b = (Gamma_b / k_b - w1 psi_{n-1} - w2 psi_{n-2}) / wb
                diag[i] -= gi * w1 / wb
                lower[i] -= gi * w2 / wb
                b[i] += gi * gam_b / (c.k_f[-1] * wb)
            else:
                upper[i] += gi
                # boundary row: two-point psi'(1) keeps the system tridiagonal (first order in this term only)
                gb = c.w_b / delta
                diag[n] += gb
                lower[n] -= gb
        ab = np.zeros((3, m))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        return ab, b

    @staticmethod
    def _matvec(ab, y):
        out = ab[1] * y
        out[:-1] += ab[0, 1:] * y[1:]
        out[1:] += ab[2, :-1] * y[:-1]
        return out

    @staticmethod
    def _solve(alpha, ab, rhs):
        """Solve (I - alpha A) x = rhs."""
        lhs = -alpha * ab
        lhs[1] += 1.0
        return solve_banded((1, 1), lhs, rhs)

    def _pack(self, state):
        return state.psi.copy() if self.ip_mode else np.concatenate((state.psi, [state.psi_boundary]))

    def _unpack(self, y, rf, t, c_end, meta):
        n = rf.size - 1
        if self.ip_mode:
            psi = y.copy()
            psi_b = _psi_boundary_from_flux(psi, 0.5 * (rf[1:] + rf[:-1]),
                                            c_end.bc_value / c_end.F_f[-1] / c_end.k_f[-1])
        else:
            psi, psi_b = y[:n].copy(), float(y[n])
        return CurrentState(rf, psi, psi_b, t, meta)

    # -------------------------------------------------------------- time stepping
    def step(self, state, dt, scheme="trbdf2", theta=0.5):
        """Advance ``state`` by one step ``dt``; returns a new state."""
        rf = state.rho_faces
        t0 = state.time
        y0 = self._pack(state)
        c0 = self.coefficients(t0, rf, state)
        A0, b0 = self._operator(c0, rf)
        f0 = self._matvec(A0, y0) + b0
        if scheme == "theta":
            c1 = self.coefficients(t0 + dt, rf, state)
            A1, b1 = self._operator(c1, rf)
            rhs = y0 + dt * ((1.0 - theta) * f0 + theta * b1)
            y1 = self._solve(theta * dt, A1, rhs)
        elif scheme == "trbdf2":
            g = _TRBDF2_GAMMA
            cg = self.coefficients(t0 + g * dt, rf, state)
            Ag, bg = self._operator(cg, rf)
            rhs = y0 + 0.5 * g * dt * (f0 + bg)
            ys = self._solve(0.5 * g * dt, Ag, rhs)
            c1 = self.coefficients(t0 + dt, rf, state)
            A1, b1 = self._operator(c1, rf)
            kap = (1.0 - g) / (2.0 - g)
            rhs = ys / (g * (2.0 - g)) - (1.0 - g) ** 2 / (g * (2.0 - g)) * y0 + kap * dt * b1
            y1 = self._solve(kap * dt, A1, rhs)
        else:
            raise ValueError("scheme must be 'trbdf2' or 'theta'")
        return self._unpack(y1, rf, t0 + dt, c1, dict(state.meta))

    def advance(self, state, t_end, dt=None, nsteps=None, scheme="trbdf2", theta=0.5,
                post_step: Optional[Callable] = None):
        """Advance from ``state.time`` to ``t_end`` in equal steps.

        Give either ``nsteps`` or a maximum step ``dt`` (the interval is split
        into ``ceil((t_end - t0)/dt)`` equal steps, so ``t_end`` is hit
        exactly).  ``post_step(state, t, solver) -> state`` is applied after
        each step (sawtooth hook).  Returns the final state; the input is not
        modified.
        """
        span = float(t_end) - state.time
        if span < 0:
            raise ValueError("t_end is before the state's time")
        if span == 0:
            return state.copy()
        if nsteps is None:
            if dt is None:
                raise ValueError("give dt or nsteps")
            nsteps = int(np.ceil(span / dt - 1e-12))
        h = span / nsteps
        t0 = state.time
        s = state
        for k in range(nsteps):
            s = self.step(s, h, scheme=scheme, theta=theta)
            # time from the step index, not by accumulation; the last step lands on t_end exactly
            s.time = float(t_end) if k == nsteps - 1 else t0 + (k + 1) * h
            if post_step is not None:
                s = post_step(s, s.time, self)
        return s

    # -------------------------------------------------------------- steady state
    def steady_state(self, rho_faces, t=0.0, psi_boundary=0.0):
        r"""Fully relaxed state: ``dpsi/dt|_Phi = V/2pi`` uniform (module docstring).

        Under :class:`IpBoundary`, ``V = 2 pi (Ip/F_b - sum S) / sum m``; under
        :class:`VloopBoundary`, ``V`` is the boundary value and Ip follows.
        ``psi_boundary`` fixes the arbitrary offset.  The returned state's
        ``meta['v_loop']`` holds V [V] and ``meta['ip']`` the total current.
        The moving-boundary term (7) does not enter (relaxation is defined in
        the fixed-Phi frame).
        """
        rf = np.asarray(rho_faces, dtype=float)
        rc = 0.5 * (rf[1:] + rf[:-1])
        n = rf.size - 1
        c = self.coefficients(t, rf)
        if self.ip_mode:
            ip = c.bc_value
            rate = (ip / c.F_f[-1] - c.s_c.sum()) / c.m_c.sum()     # dpsi/dt|_Phi
        else:
            rate = c.bc_value / (2.0 * np.pi)
        gam = np.cumsum(c.m_c * rate + c.s_c)                      # I/F at faces 1..n
        if not self.ip_mode:
            ip = gam[-1] * c.F_f[-1]
        dpsi = gam[:n - 1] * np.diff(rc) / c.k_f[1:n]
        psi = np.concatenate(([0.0], np.cumsum(dpsi)))
        psi_b_rel = _psi_boundary_from_flux(psi, rc, gam[-1] / c.k_f[-1])
        psi += psi_boundary - psi_b_rel
        return CurrentState(rf, psi, psi_boundary, t, dict(v_loop=2.0 * np.pi * rate, ip=float(ip)))

    # -------------------------------------------------------------- outputs
    def profiles(self, state, dpressure_drho=None):
        r"""Derived profiles of ``state`` at ``state.time``.

        Face quantities (``rho_faces``): ``psi``, ``dpsi_drho``, ``I_enclosed``
        [A] (eq. 3), ``q`` and ``iota = 1/q`` (positive frame; axis value
        extrapolated as ``q0 + c rho^2``).

        Cell quantities (``rho_cells``): ``jdotb`` = <J.B> (eq. 4),
        ``jdotb_ni``, ``jdotb_ohmic`` = <J.B> - <J_NI.B> [A T/m^2],
        ``j_par`` = <J.B>/B0 [A/m^2] (IMAS ``j_parallel`` convention),
        ``jt_over_r`` = <J_phi/R> (eq. 5) [A/m^3], ``j_tor`` =
        <J_phi/R>/<1/R> [A/m^2] (IMAS ``j_tor``; needs ``avg_inv_r``),
        ``v_loop`` = 2 pi eta <J_ohm.B>/(F <1/R^2>) [V] (eqs. 1-2),
        ``conductivity``.  With ``dpressure_drho`` (callable rho -> dp/drho
        [Pa], or an array on the cells) and ``avg_r``, ``avg_b2`` in the
        geometry also ``jphi_avg`` = <J_phi> (plain surface average, the
        TokaMaker ``jphi`` input), from the Grad-Shafranov identity

        .. code-block:: text

            <J_phi> = (F <1/R> / <B^2>) <J.B> - (dp/dpsi) (<R> - F^2 <1/R> / <B^2>)

        (derived in the P0c report from J_phi = -R p' - F F'/(mu0 R) and
        <J.B> = -F p' - F' <B^2>/mu0 for the field of eq. 1, p' = dp/dpsi).

        Scalars: ``ip`` [A], ``v_loop_boundary`` [V], ``w_pol`` = int B_p^2/(2 mu0) dV [J],
        ``li3`` = 2 int B_p^2 dV / (mu0^2 Ip^2 R0).
        """
        rf = state.rho_faces
        rc = state.rho_cells
        n = rc.size
        h = np.diff(rf)
        c = self.coefficients(state.time, rf, state)
        geo = c.geo
        gc = c.geo_c
        # face gradients and fluxes
        dpsi_f = np.zeros(n + 1)
        dpsi_f[1:n] = np.diff(state.psi) / np.diff(rc)
        wb, w1, w2 = _boundary_gradient_weights(rc)
        dpsi_f[n] = wb * state.psi_boundary + w1 * state.psi[-1] + w2 * state.psi[-2]
        gam_f = c.k_f * dpsi_f                       # I/F
        if self.ip_mode:
            gam_f[n] = c.bc_value / c.F_f[-1]
            dpsi_f[n] = gam_f[n] / c.k_f[-1]
        I_f = gam_f * c.F_f
        ip = float(I_f[-1])
        Vc = gc["dvol_drho"]
        jdotb = 2.0 * np.pi * gc["fpol"] ** 2 * np.diff(gam_f) / (Vc * h)
        jt_r = 2.0 * np.pi * np.diff(I_f) / (Vc * h)
        j_ohm = jdotb - c.jni_c
        v_loop = 2.0 * np.pi * j_ohm / (c.sigma_c * gc["fpol"] * gc["avg_inv_r2"])
        v_b = v_loop[-1] + (v_loop[-1] - v_loop[-2]) * (1.0 - rc[-1]) / (rc[-1] - rc[-2])
        if not self.ip_mode:
            v_b = c.bc_value
        # psi on faces
        psi_f = np.empty(n + 1)
        psi_f[1:n] = state.psi[:-1] + (state.psi[1:] - state.psi[:-1]) * (rf[1:n] - rc[:-1]) / np.diff(rc)
        psi_f[0] = state.psi[0] - (state.psi[1] - state.psi[0]) * rc[0] ** 2 / (rc[1] ** 2 - rc[0] ** 2)
        psi_f[n] = state.psi_boundary
        with np.errstate(divide="ignore", invalid="ignore"):
            q_f = geo.phi_b * rf / (np.pi * dpsi_f)
        q_f[0] = q_f[1] - (q_f[2] - q_f[1]) * rf[1] ** 2 / (rf[2] ** 2 - rf[1] ** 2)
        # poloidal field energy: int B_p^2 dV = int psi'^2 g2 V' drho = 2 pi mu0 int I psi' drho
        bp2_int = 2.0 * np.pi * MU0 * np.trapezoid(I_f * dpsi_f, rf)
        out = dict(time=state.time, rho_faces=rf.copy(), rho_cells=rc.copy(), psi=psi_f, dpsi_drho=dpsi_f,
                   I_enclosed=I_f, q=q_f, iota=1.0 / q_f, jdotb=jdotb, jdotb_ni=c.jni_c.copy(), jdotb_ohmic=j_ohm,
                   j_par=jdotb / geo.B0, jt_over_r=jt_r, v_loop=v_loop, conductivity=c.sigma_c.copy(), ip=ip,
                   v_loop_boundary=float(v_b), w_pol=bp2_int / (2.0 * MU0),
                   li3=2.0 * bp2_int / (MU0 ** 2 * ip ** 2 * geo.R0) if ip != 0 else np.nan)
        if "avg_inv_r" in gc:
            out["j_tor"] = jt_r / gc["avg_inv_r"]
        if dpressure_drho is not None and "avg_r" in gc and "avg_b2" in gc:
            dp = dpressure_drho(rc) if callable(dpressure_drho) else np.asarray(dpressure_drho, dtype=float)
            dpsi_c = 0.5 * (dpsi_f[1:] + dpsi_f[:-1])
            dpsi_c[0] = dpsi_f[1] * rc[0] / rf[1]     # psi' ~ rho near the axis
            kap = gc["fpol"] * gc["avg_inv_r"] / gc["avg_b2"]
            out["jphi_avg"] = kap * jdotb - (dp / dpsi_c) * (gc["avg_r"] - gc["fpol"] * kap)
        return out
