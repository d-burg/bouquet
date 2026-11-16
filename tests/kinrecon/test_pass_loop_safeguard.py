"""The pass-consistency loop's Anderson safeguard (``current_model.consistent_solve``), on a mocked model (no OFT).

The equilibrium solve and the model evaluation are replaced by a fixed map ``G`` of the comparison-grid profile
(the "solve" stores its input; the next "assembly" returns ``G`` of it), so the loop is a fixed-point iteration whose
residuals are known.  Checked: the history is restarted -- and that pass takes the plain input -- exactly on the
passes whose residual max-norm grew against the previous pass's; the loop still stops by the unchanged criterion
(``max |j_k - j_{k-1}| / max |j_k| < pass_tol`` with the other consistency tests) at the map's fixed point; plain
passes never restart.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM

N = 24
RHO = np.linspace(0.0, 1.0, N)
XSTAR = 1.0 + np.cos(2.0 * RHO)
# a contraction with rotating, unequal modes and a non-smooth part: plain passes converge, Anderson's affine model is
# wrong near the kink, so residuals grow on some passes
_ROT = np.zeros((N, N))
for _i in range(0, N - 1, 2):
    c, s = np.cos(1.0), np.sin(1.0)
    _ROT[_i:_i + 2, _i:_i + 2] = 0.8 * np.array([[c, -s], [s, c]])
_ROT[N - 1, N - 1] = -0.7


def G(x):
    d = x - XSTAR
    return XSTAR + _ROT @ d + 0.3 * np.abs(d) * np.sign(RHO - 0.5)


class FakeFM:
    def __init__(self):
        self.state = XSTAR + 0.3 * np.sin(7.0 * RHO)
        self.inputs = []
        self.gs = None
        self.n_solves = 0

    def solve(self, j, p, ip, init="keep", label="", lock_case=True):
        self.state = np.asarray(j[1], float).copy()
        self.inputs.append(self.state.copy())
        self.n_solves += 1
        return SimpleNamespace(converged=True, status="converged", record={"attempts": [{"its": 1}]}, wall_s=0.0,
                               stats=dict(R_axis=1.7, Z_axis=0.0, l_i=1.0, q_0=1.0, Ip=1.0))

    def snapshot(self):
        return self.state.copy()

    def restore(self, snap):
        self.state = snap.copy()

    def update_probe_correction(self):
        return 0.0


def _install(monkeypatch, fm):
    monkeypatch.setattr(CM, "evaluate_geometry", lambda gs, kin, cfg: dict(wall_s=0.0))

    def assemble(lag, kin, src, params, cfg, ip, z0=None, state=None):
        out = G(fm.state)
        return dict(orbit_z=None, wall_s=0.0, fractions={}, sawtooth={}, knots={"n": N}, jphi_cmp=out, jphi=out,
                    x=RHO, rho_knots=RHO, rho_cmp=RHO, p=np.zeros(N))

    monkeypatch.setattr(CM, "assemble", assemble)
    monkeypatch.setattr(CM, "q_check", lambda lag, asm, stats, cfg: dict(passed=True, max_rel=0.0))


@pytest.mark.parametrize("accel", ["anderson", "none"])
def test_history_restarts_exactly_when_the_residual_grows(monkeypatch, accel):
    fm = FakeFM()
    _install(monkeypatch, fm)
    cfg = CM.CurrentModelSettings(r0=1.0, max_passes=200, pass_acceleration=accel)
    res = CM.consistent_solve(fm, None, None, {}, cfg, ip=1.0)
    assert res.converged and res.status == "consistent"
    # the unchanged criterion stopped it, at the map's fixed point
    last = res.record["passes"][-1]
    assert last["dj"] < cfg.pass_tol
    np.testing.assert_allclose(fm.state, XSTAR, rtol=0, atol=1e-3)
    # residual of pass k: the assembled profile minus the input of the previous solve
    recs = res.record["passes"]
    xs = fm.inputs
    restarts = [bool(r.get("anderson_restart")) for r in recs]
    if accel == "none":
        assert not any(restarts)
        return
    r = [np.max(np.abs(G(xs[k - 1]) - xs[k - 1])) for k in range(1, len(xs))]       # residual seen at pass k
    for k in range(2, len(xs)):
        grew = r[k - 1] > r[k - 2]
        assert restarts[k] == bool(grew), (k, r[k - 2], r[k - 1])
        if grew:
            assert recs[k]["anderson_mixed"] is False
    assert any(restarts) and any(r_.get("anderson_mixed") for r_ in recs)
