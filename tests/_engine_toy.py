"""A toy Grad-Shafranov stand-in for the unified engine's fast tests.

Not a test module (no ``test_`` prefix).  It implements the engine's backend
interface (``solve`` / ``measure`` / ``snapshot`` / ``restore`` /
``n_solves``) with a deterministic map from the requested ``jphi-linterp``
current to a "solved" state, modelled on the mocks the loop tests use but
with the couplings the engine has to handle:

* **Ip is imposed** exactly (like TokaMaker): the delivered current is the
  request rescaled by one uniform factor ``c``, in the affine FSA measure;
* **a delivery defect**: the solver delivers ``c * R * (1 - delta(psi))``
  with a localised edge bump ``delta`` (the ``jphi-linterp`` defect's
  footprint); ``defect=0`` switches it off;
* **the flux range responds to l_i**: ``dpsi = D0^2 Q / li_ref`` with
  ``l_i = dpsi Q`` (``Q`` the shape factor of the enclosed current), so the
  delivered l_i goes as the square of the frozen-geometry model's -- the
  measured log-gain of 2 (``utils.LI_GAIN_EXPONENT``); ``gain=False`` holds
  the flux range fixed;
* **the Redl bootstrap follows the geometry** (``~ 1/dpsi``: gradients per
  unit flux);
* **q at the row radius** ``~ 1 / j_phi(axis)`` of the delivered current;
* **pitch angles**: ``B_Z / B_phi`` at a few chords from the delivered
  enclosed current.

Every flux-surface quantity is analytic and the l_i and Ip measures are the
package's own (:func:`bouquet.utils.Ip_fsa_weights`, ``li_value``), so the
engine's model and the toy's measurement agree exactly on a frozen geometry
and differ only through the couplings above.

Synthetic numbers only (a generic D-shaped plasma; no device data).
"""
from __future__ import annotations

import numpy as np

from bouquet.adapters import EngineContract, validate_contract

MU0 = 4.0e-7 * np.pi
R0, A_MIN, B0, VTOT, KAPPA = 1.70, 0.60, 2.0, 20.0, 1.7
PSI = np.linspace(0.0, 1.0, 101)
PAD = 1e-3


def _gauss(x, mu, w):
    return np.exp(-0.5 * ((x - mu) / w) ** 2)


def base_inductive(psi=PSI):
    """A peaked inductive <j.B> profile [T A/m^2]."""
    return B0 * (1.2e6 * (1.0 - psi ** 2) ** 1.5 + 1.0e5 * (1.0 - psi))


def pressure(psi=PSI):
    return 6.0e4 * (1.0 - psi ** 2) ** 1.5 + 1.2e4 * (1.0 - psi ** 8)


class ToyGS:
    """The backend (see the module docstring)."""

    def __init__(self, psi=PSI, *, Ip=1.2e6, defect=0.02, gain=True,
                 dpsi0=0.30, li_ref=0.80, q_scale=1.5e6, chords=None,
                 nan_redl_at=None, redl_scale=1.0):
        self.psi = np.asarray(psi, dtype=float)
        self.Ip = float(Ip)
        self.defect = float(defect)
        self.gain = bool(gain)
        self.dpsi0 = float(dpsi0)
        self.li_ref = float(li_ref)
        self.q_scale = float(q_scale)
        self.chords = chords
        self.nan_redl_at = nan_redl_at
        self.redl_scale = float(redl_scale)
        self.n_solves = 0
        self.D0sq = None                      # calibrated on the first solve
        self.state = None
        self.history = []
        # a draw's inputs (set_inputs): the pressure the geometry's p' comes
        # from and the kinetics the Redl bootstrap scales with; the defaults
        # are the toy's own (bit for bit what the toy did before draws)
        self.p = pressure(self.psi)
        self.kin = None

    # ---- geometry ----------------------------------------------------------
    def geometry(self, dpsi):
        x = self.psi
        eps = (A_MIN / R0) * np.sqrt(np.clip(x, PAD, 1 - PAD))
        R_avg = R0 * (1.0 + 0.25 * eps ** 2)
        inv_R = (1.0 / R0) * (1.0 + 0.5 * eps ** 2)
        inv_R2 = (1.0 / R0 ** 2) * (1.0 + 1.5 * eps ** 2)
        F = np.full_like(x, R0 * B0)
        Bp2 = (0.12 * B0 * eps) ** 2
        B2 = F ** 2 * inv_R2 + Bp2
        dp = np.gradient(self.p, x)
        return dict(psi_N=x.copy(), psi_q=np.clip(x, PAD, 1 - PAD),
                    R_avg=R_avg, inv_R=inv_R, inv_R2=inv_R2,
                    dV_dpsi=np.full_like(x, VTOT / dpsi),
                    dpsi_dpsiN=float(dpsi), pprime=-dp / dpsi, F=F, B2=B2)

    def li_geom(self, geom):
        from bouquet.utils import Ip_fsa_affine_profile
        per = 2.0 * np.pi * A_MIN * np.sqrt((1.0 + KAPPA ** 2) / 2.0)
        return dict(psi_N=geom["psi_N"], dpsi_dpsiN=geom["dpsi_dpsiN"],
                    vol=VTOT, perimeter=per, R_axis=R0,
                    affine_cum=Ip_fsa_affine_profile(geom), psi_pad=PAD)

    def _delta(self):
        return self.defect * _gauss(self.psi, 0.97, 0.015)

    def _deliver(self, R, dpsi):
        from scipy.integrate import cumulative_trapezoid, trapezoid
        from bouquet.utils import Ip_fsa_weights, li_value
        g = self.geometry(dpsi)
        w, caff = Ip_fsa_weights(g)
        shape = R * (1.0 - self._delta())
        c = (self.Ip - caff) / float(trapezoid(w * shape, self.psi))
        A = c * shape
        I = cumulative_trapezoid(w * A, self.psi, initial=0.0) \
            + Ip_fsa_affine_profile_cached(g)
        lg = self.li_geom(g)
        li = float(li_value(self.psi, I, lg, "li_3"))
        return g, A, I, li, c

    def solve(self, request, n_passes=1):
        R = np.asarray(request, dtype=float)
        if not np.all(np.isfinite(R)):
            raise RuntimeError("toy: non-finite request")
        self.n_solves += int(n_passes)
        dpsi = self.dpsi0 if self.state is None else self.state["dpsi"]
        if self.D0sq is None:
            _g, _A, _I, li, _c = self._deliver(R, self.dpsi0)
            # calibrate so this first request sits at dpsi0 (li = dpsi0 Q)
            self.D0sq = self.dpsi0 * self.li_ref / (li / self.dpsi0)
        for _ in range(200):
            g, A, I, li, c = self._deliver(R, dpsi)
            if not self.gain:
                break
            new = self.D0sq * (li / dpsi) / self.li_ref
            if abs(new - dpsi) <= 1e-15 * dpsi:
                dpsi = new
                break
            dpsi = new
        g, A, I, li, c = self._deliver(R, dpsi)
        self.state = dict(R=R.copy(), dpsi=float(dpsi), A=A, I=I, li=li,
                          c=float(c), geom=g)
        self.history.append(dict(li=li, dpsi=float(dpsi), c=float(c)))

    def snapshot(self):
        return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self.state.items()}

    def restore(self, eq):
        self.state = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in eq.items()}

    # ---- a draw's inputs ----------------------------------------------------
    def set_inputs(self, pressure=None, kinetics=None):
        """The engine backend's draw hook: *pressure* feeds the geometry's
        p' (and beta), *kinetics* scales the Redl bootstrap pointwise by
        ``ne Te / (ne Te)_toy`` (exactly 1 for the toy's own kinetics)."""
        self.p = (globals()["pressure"](self.psi) if pressure is None
                  else np.asarray(pressure, dtype=float).copy())
        self.kin = None if kinetics is None else dict(kinetics)

    def _kin_factor(self, kin):
        if kin is None:
            return 1.0
        return (np.asarray(kin["ne"], float) * np.asarray(kin["te"], float)
                / (np.full(self.psi.size, 5e19) * np.full(self.psi.size,
                                                          1e3)))

    def flux_integral(self, psi_N, profile):
        """A volume integral on the current state (uniform dV/dpsi)."""
        from scipy.integrate import trapezoid
        dpsi = self.dpsi0 if self.state is None else self.state["dpsi"]
        g = self.geometry(dpsi)
        return float(trapezoid(g["dV_dpsi"] * dpsi
                               * np.asarray(profile, float),
                               np.asarray(psi_N, float)))

    def redl(self, kinetics=None):
        """Redl on the CURRENT state with *kinetics* (the engine backend
        interface; default: the backend's own)."""
        dpsi = self.dpsi0 if self.state is None else self.state["dpsi"]
        return self._redl_profile(dpsi, self.kin if kinetics is None
                                  else kinetics)

    # ---- measurements --------------------------------------------------------
    def _redl_profile(self, dpsi, kin=None):
        x = self.psi
        j = B0 * (2.4e5 * _gauss(x, 0.94, 0.03) + 6.0e4 * x * (1.0 - x)) \
            * (self.dpsi0 / dpsi) * self.redl_scale * self._kin_factor(kin)
        if self.nan_redl_at is not None and self.n_solves >= self.nan_redl_at:
            j = j.copy()
            j[50] = np.nan
        return j

    def q_at(self, st, psi0):
        j0 = float(np.interp(psi0, self.psi, st["A"]))
        return self.q_scale / j0

    def field_at_chords(self, st):
        """``(B, found)`` at the chords, as :func:`bouquet.mse.mse_field_at`:
        chords on the outboard midplane at ``R = R0 + a sqrt(psi_N)``; a
        chord beyond the LCFS (``psi_N > 1``) is off the toy's "mesh" and is
        reported (NaN, ``found=False``), never read.  Right-handed
        ``(R, phi, Z)``: a current along +phi has ``B_Z < 0`` outboard."""
        ch = self.chords
        r = np.asarray(ch["R"], dtype=float) - R0
        psi = (r / A_MIN) ** 2
        found = (r > 0.0) & (psi <= 1.0)
        I = np.interp(np.clip(psi, 0.0, 1.0), self.psi, st["I"])
        Bz = np.where(found, -MU0 * I / (2.0 * np.pi * np.where(found, r, 1.0)),
                      np.nan)
        Bphi = np.where(found, R0 * B0 / (R0 + r), np.nan)
        return np.column_stack([np.where(found, 0.0, np.nan), Bphi, Bz]), found

    def measure(self, want_chords=False, final=False):
        st = self.state
        g = dict(st["geom"])
        g["li_geom"] = self.li_geom(g)
        out = dict(geom=g, redl=self._redl_profile(st["dpsi"], self.kin),
                   li=st["li"],
                   q_row=self.q_at(st, float(g["psi_q"][0])), Ip=self.Ip,
                   achieved=st["A"].copy())
        if want_chords and self.chords is not None:
            out["B_chords"], out["chords_found"] = self.field_at_chords(st)
            out["axis"] = (R0, 0.0)
        if final:
            from scipy.integrate import trapezoid
            j95 = float(np.interp(0.95, self.psi, st["A"]))
            pav = float(trapezoid(self.p, self.psi))
            out["stats"] = dict(
                q_0=out["q_row"], l_i=st["li"],
                q_95=(self.q_scale / j95 if j95 > 0 else float("nan")),
                beta_n=100.0 * (2.0 * MU0 * pav / B0 ** 2) * A_MIN * B0
                / (self.Ip / 1e6))
            out["li_1"] = st["li"]
        return out


_AFF_CACHE = {}


def Ip_fsa_affine_profile_cached(g):
    from bouquet.utils import Ip_fsa_affine_profile
    return Ip_fsa_affine_profile(g)


def toy_chords(n=8, off_mesh=0):
    """Chord geometry for the toy (tan(gamma) = B_Z / B_phi); the last
    *off_mesh* chords sit outside the LCFS."""
    psi = np.linspace(0.08, 0.80, n)
    if off_mesh:
        psi[-off_mesh:] = 1.3
    r = A_MIN * np.sqrt(psi)
    return dict(R=R0 + r, Z=np.zeros(n), A1=np.ones(n),
                A2=np.ones(n), A3=np.zeros(n), A4=np.zeros(n))


class ToyAdapter:
    """An adapter producing a contract on the toy grid (g-file-like HARD rows
    or IDS-like SOFT rows)."""

    kind = "toy"

    def __init__(self, *, soft=False, li_target=None, q0_target=None,
                 mse=None, Ip=1.2e6, jB_ind=None):
        self.soft = bool(soft)
        self.li_target = li_target
        self.q0_target = q0_target
        self.mse = mse
        self.Ip = float(Ip)
        self.jB_ind = jB_ind
        self._c = None

    def read(self):
        psi = PSI
        n = psi.size
        kin = dict(ne=np.full(n, 5e19), te=np.full(n, 1e3),
                   ni=np.full(n, 4.5e19), ti=np.full(n, 1e3),
                   zeff=np.full(n, 1.8))
        rows = dict(
            Ip=dict(target=self.Ip, hard=not self.soft,
                    sigma=(0.005 * self.Ip if self.soft else None)),
            l_i=(None if self.li_target is None else dict(
                target=float(self.li_target), kind="li_3", hard=not self.soft,
                tol=(1e-3 if not self.soft else None),
                sigma=(0.04 if self.soft else None))),
            q0=(None if self.q0_target is None else dict(
                target=float(self.q0_target), psi=PAD, admitted=True,
                gate_basis="toy")),
            mse=self.mse)
        jind = base_inductive(psi) if self.jB_ind is None else self.jB_ind
        self._c = EngineContract(
            kind="toy", psi_N=psi, kinetics=kin,
            kinetics_native=dict(psi_N=psi, **kin),
            pressure=pressure(psi),
            pressure_parts=dict(thermal=pressure(psi), impurity=0 * psi,
                                fast=0 * psi),
            jB_ind=np.asarray(jind, float),
            jB_fix=B0 * 3.0e4 * _gauss(psi, 0.3, 0.1),
            jB_fix_parts=dict(nbi=B0 * 3.0e4 * _gauss(psi, 0.3, 0.1),
                              rf=0 * psi),
            boundary=np.column_stack([R0 + A_MIN * np.cos(np.linspace(0, 6.2, 40)),
                                      KAPPA * A_MIN * np.sin(np.linspace(0, 6.2, 40))]),
            Ip=self.Ip, rows=rows,
            signs=dict(current_sign=1.0, b0_sign=None, frame="positive"),
            anchor_request=1.1e6 * (1.0 - psi ** 2) ** 1.2 + 1.0e5)
        validate_contract(self._c)
        return self._c

    def finalize(self, jB_bs_anchor, anchor_geom):
        self._c.jB_bs_anchor = np.asarray(jB_bs_anchor, float).copy()
        validate_contract(self._c)
        return self._c


def settings(**gc_kw):
    """Engine settings from a GenerationConfig (unified engine)."""
    from bouquet.config import GenerationConfig
    from bouquet.engine import engine_settings
    kw = dict(reconstruction_engine="unified")
    kw.update(gc_kw)
    return engine_settings(GenerationConfig(**kw))
