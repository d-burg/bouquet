"""A TokaMaker stand-in over the toy engine backend, for the fast end-to-end
tests of the engine draws through ``generate_bouquet`` (not a test module).

``FakeTokaMaker`` wraps a :class:`_engine_toy.ToyGS`: the engine draw's
backend IS that toy (``bouquet.engine_draws.tokamaker_backend`` is
monkeypatched to return it), so a ``solve()`` issued by ``generate_bouquet``
itself (the baseline re-solve, the homotopy passes) re-solves the toy's
last request -- the toy has no coils, so every coil stays at its baseline
value and every homotopy pass is feasible.  Geometry-only reads (traces,
x-points, stats) return fixed synthetic numbers derived from the toy state.
``save_eqdsk`` writes the repository's synthetic example g-file (the archive
needs real g-file bytes; its contents are not the toy's equilibrium).

Synthetic numbers only; no device data.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

import _engine_toy as T

_HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_GEQDSK = os.path.join(_HERE, os.pardir, "examples", "D3D-like",
                              "D3Dlike_Hmode_baseline.geqdsk")


class _Settings:
    maxits = 80


class FakeTokaMaker:
    """The live-solver surface ``generate_bouquet`` touches, over a toy."""

    def __init__(self, toy, *, coils=None, lcfs_scale=1.0):
        self.toy = toy
        self.coil_sets = dict(coils or {"F1A": 0, "F2A": 1, "F9A": 2,
                                        "F9B": 3})
        self._coils = {"F1A": 2.0e4, "F2A": -1.5e4, "F9A": 3.0e3,
                       "F9B": -3.0e3}
        self.settings = _Settings()
        self.calls = []
        self._ffp = None
        self.lcfs_scale = float(lcfs_scale)
        self.bounds = None
        self.reg = None

    # ---- state ------------------------------------------------------------
    def copy_eq(self):
        return self.toy.snapshot()

    def replace_eq(self, source_eq=None):
        self.toy.restore(source_eq)

    @property
    def psi_bounds(self):
        dpsi = (self.toy.dpsi0 if self.toy.state is None
                else self.toy.state["dpsi"])
        return [-float(dpsi), 0.0]

    @property
    def o_point(self):
        return [T.R0, 0.0]

    def update_settings(self):
        self.calls.append(("update_settings", self.settings.maxits))

    # ---- profiles / solve --------------------------------------------------
    def set_targets(self, Ip=None, pax=None, **kw):
        self.calls.append(("set_targets", Ip, pax))

    def set_profiles(self, pp_prof=None, ffp_prof=None, **kw):
        self._ffp = None if ffp_prof is None else np.asarray(ffp_prof["y"],
                                                             float)

    def solve(self, *a, **k):
        self.calls.append(("solve",))
        req = (self._ffp if self._ffp is not None
               else self.toy.state["R"])
        self.toy.solve(req, n_passes=1)
        self._ffp = None

    def init_psi(self, *a):
        self.calls.append(("init_psi",))

    def get_psi(self, normalized=True):
        return np.zeros(16)

    def set_psi(self, psi, update_bounds=False):
        pass

    def set_isoflux(self, pts, weights=None):
        self.calls.append(("set_isoflux", len(pts)))

    # ---- coils ---------------------------------------------------------
    def get_coil_currents(self):
        return dict(self._coils), None

    def set_coil_currents(self, c):
        pass

    def set_coil_bounds(self, b):
        self.bounds = b

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        return (tuple(coffs), float(target), float(weight))

    def set_coil_reg(self, reg_terms=None, **kw):
        self.reg = reg_terms

    def set_coil_vsc(self, *a, **k):
        pass

    # ---- reads -----------------------------------------------------------
    def get_globals(self):
        return [self.toy.Ip, 0.0, 0.0]

    def get_q(self, psi=None, npsi=None, psi_pad=1e-3, **kw):
        st = self.toy.state
        x = (np.linspace(psi_pad, 1 - psi_pad, int(npsi or 50))
             if psi is None else np.asarray(psi, float))
        A = st["A"]
        q = self.toy.q_scale / np.maximum(np.interp(x, self.toy.psi, A),
                                          1.0)
        return x, q, None, None, None, None

    def get_stats(self, lcfs_pad=None, li_normalization="std", **kw):
        st = self.toy.state
        q = self.get_q(psi=[0.02, 0.95])[1]
        return dict(l_i=float(st["li"]), Ip=float(self.toy.Ip),
                    q_0=float(q[0]), q_95=float(q[1]), beta_n=1.5,
                    beta_pol=60.0, W_MHD=5.0e5, kappa=1.7, delta=0.3)

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        x = (np.linspace(0.0, 1.0, int(npsi or 50)) if psi is None
             else np.asarray(psi, float))
        n = len(x)
        return (x, np.full(n, T.R0 * T.B0), np.full(n, -0.1), np.zeros(n),
                np.zeros(n))

    def trace_surf(self, psi):
        th = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
        a = T.A_MIN * self.lcfs_scale * (1.0 + 0.001 * (
            self.toy.state["li"] if self.toy.state else 0.0))
        return np.column_stack([T.R0 + a * np.cos(th),
                                T.KAPPA * a * np.sin(th)])

    def get_xpoints(self):
        return None, False

    def flux_integral(self, psi_N, profile):
        return self.toy.flux_integral(psi_N, profile)

    def save_eqdsk(self, filename, **kw):
        shutil.copyfile(EXAMPLE_GEQDSK, filename)
