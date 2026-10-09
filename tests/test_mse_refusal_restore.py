"""MSE chord stage under the self-consistent loop: the refusal fallback (M2).

With ``structured_mse_required=False`` a refused MSE stage must deliver the
PRE-MSE loop result.  Every chord step's ``refresh`` re-closes on the latest
geometry and overwrites ``bl.j_phi`` / ``bl.j_BS`` / ``bl.ip_closure``, so the
fallback has to restore the state captured BEFORE the first step, re-solve the
current the loop actually solved, and check that re-solve against the pre-MSE
residuals -- ``converged`` is the result of that check, never assumed.

Mocked solver and mocked MSE algebra (no OFT): the stage runs one chord step,
refreshes (mutating the baseline), and the closure refuses on the second step.

Synthetic inputs only; no device data.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.jbs_loop import JBSNotConverged, jbs_settings

_X = np.linspace(0.0, 1.0, 41)
_W = 1.0 + 0.5 * _X
_IP = 1.0e6
_NCH = 4


class _GC:
    def __init__(self, **kw):
        self.jbs_self_consistent = True
        self.jbs_init = "anchor"
        self.jbs_rtol_j = 1e-3
        self.jbs_rtol_Ip = 1e-4
        self.jbs_tol_li = 1e-3
        self.jbs_tol_q0 = 2e-3
        self.jbs_max_passes = 8
        self.jbs_max_passes_draw = 12
        self.jbs_max_passes_post_homotopy = 4
        self.jbs_relax = 0.7
        self.jbs_loop_on_fail = "raise"
        for k, v in kw.items():
            setattr(self, k, v)


def _jbs0():
    return 3.0e5 * np.exp(-0.5 * ((_X - 0.93) / 0.03) ** 2) + 1.0e5 * (1 - _X)


class _Snap:
    def __init__(self, j):
        self.j = np.array(j, dtype=float, copy=True)


class _Eq:
    """The equilibrium IS the last solved current (a deterministic mock):
    Redl on it returns the pre-MSE bootstrap iff it is the pre-MSE solve."""

    def __init__(self, j0, drift=0.0):
        self.j = np.array(j0, dtype=float, copy=True)
        self.solved = []
        self.replaced = []
        self.drift = float(drift)       # a re-solve landing elsewhere

    def copy_eq(self):
        return _Snap(self.j)

    def replace_eq(self, source_eq=None):
        self.replaced.append(source_eq)
        self.j = np.array(source_eq.j, dtype=float, copy=True)

    def solve_jphi(self, j):
        self.solved.append(np.array(j, dtype=float, copy=True))
        self.j = np.array(j, dtype=float, copy=True) * (1.0 + self.drift)
        return 7


def _chords():
    """The mocked chord block: the keys the stage reads (positions, input
    indices and the STATED orientation ip_sign/bt_sign)."""
    return dict(sigma_eff=np.ones(_NCH), n_active=_NCH,
                R=np.linspace(1.9, 2.2, _NCH), Z=np.zeros(_NCH),
                index=np.arange(_NCH), ip_sign=1.0, bt_sign=1.0,
                min_chords=_NCH, er_applied=False)


def _field_at(R, Z):
    """``field_at(R, Z) -> (B, found)`` as bouquet.mse.mse_field_at: every
    chord on the mesh (the tan(gamma) algebra is mocked)."""
    n = np.size(R)
    return np.zeros((n, 3)), np.ones(n, dtype=bool)


# The stage runs ONE zero-Jacobian dry run of the closure (the free-dimension
# check) before its first chord step; with n_ok_resolves=2 the first chord
# step's closure is still the one that succeeds and the second the one that
# refuses, as this module's scenario needs.
def _patch_mse(monkeypatch, n_ok_resolves=2):
    import bouquet.mse as M
    import bouquet.utils as U
    calls = {"resolve": 0}
    monkeypatch.setattr(U, "structured_basis_eval",
                        lambda spec, psi: np.ones((1, np.size(psi))))
    # the equilibrium's own directions (the mocked field carries none); the
    # stated orientation then comes from the block, as in production
    monkeypatch.setattr(M, "mse_equilibrium_orientation",
                        lambda B, R, Z, axis: dict(ip=1.0, bt=1.0,
                                                   n_ip_agree=_NCH))
    monkeypatch.setattr(M, "mse_tan_gamma",
                        lambda B, ch, sp=1.0, st=1.0: np.zeros(_NCH))
    monkeypatch.setattr(M, "mse_chi2",
                        lambda tg, ch: (0.0, np.zeros(_NCH)))
    monkeypatch.setattr(U, "structured_mse_jacobian",
                        lambda f, x, tg, free, step=0.02: np.zeros((_NCH, 2)))
    monkeypatch.setattr(U, "structured_mse_linear_model",
                        lambda *a, **k: None)
    monkeypatch.setattr(U, "structured_objective_no_mse", lambda out: 0.0)

    def _close(*a, **k):
        calls["resolve"] += 1
        if calls["resolve"] > n_ok_resolves:
            raise RuntimeError("structured closure: multipliers out of bounds")
        return dict(a=[0.0], b=[0.0], mse_chi2_model=0.0,
                    mse_objective_model=0.0)

    monkeypatch.setattr(U, "close_ip_structured", _close)
    return calls


def _setup(monkeypatch, drift_on_restore=0.0, on_fail="raise"):
    _patch_mse(monkeypatch)
    jbs0 = _jbs0()
    j_ind = 2.0e6 * (1.0 - _X) ** 2
    j_pre = j_ind + jbs0                       # the closure's assembly
    j_solved = 0.98 * j_ind + jbs0             # what the last pass SOLVED
    bl = SimpleNamespace(
        j_phi=j_pre.copy(), j_BS=jbs0.copy(), j_inductive=j_ind.copy(),
        jBS_diff=None, jphi_diff=None, ohm_scale=1.0, bs_scale=1.0,
        ip_closure=dict(closure_limited=False,
                        closure_limited_reasons=("pre-MSE reason",),
                        marker="pre-MSE"))
    eq = _Eq(j_solved)
    state = dict(mse=_chords(),
                 mse_required=False, soft=False, psi_geom=_X, basis=None,
                 free=np.array([True, True]), x_pred=[0.0, 0.0],
                 F_pred=0.0, j_ind=j_ind, j_BS_swb=jbs0,
                 j_fixed=np.zeros_like(_X), w_lin=_W, c_signed=0.0,
                 Ip_signed=_IP, weights=None)
    caller = dict(restored=0)

    def refresh(jbs, k):
        # what _refresh_structured does: a NEW closure on the latest geometry
        bl.j_phi = 1.3 * j_pre
        bl.j_BS = 1.3 * jbs0
        bl.ohm_scale = 1.7
        bl.ip_closure = dict(closure_limited=True,
                             closure_limited_reasons=("refresh reason",),
                             marker="mid-chord")
        return dict(state, j_ind=1.3 * j_ind)

    def evaluate(snap):
        # Redl: the pre-MSE bootstrap exactly on the pre-MSE equilibrium,
        # 5 % off anywhere else (every chord step fails r_j)
        same = np.allclose(snap.j, j_solved, rtol=0, atol=1e-6)
        return jbs0 if same else 1.05 * jbs0

    def measure(snap):
        return dict(li=0.8 + 1e-9 * float(np.sum(snap.j)) / 1e6, q0=None)

    def restore():
        caller["restored"] += 1

    s = jbs_settings(_GC(jbs_loop_on_fail=on_fail))

    def run(pre_mse=True):
        from bouquet.run import Bouquet
        eq.drift = 0.0
        pm = (dict(j_solved=j_solved, restore=restore,
                   final=dict(r_j=1e-4, r_I=1e-5)) if pre_mse else None)

        def _solve(j):
            if eq.solved and drift_on_restore:
                eq.drift = drift_on_restore      # the restore re-solve drifts
            return eq.solve_jphi(j)

        return Bouquet._structured_mse_jbs_stage(
            state, bl, eq, _solve, jbs0, refresh, evaluate, measure,
            lambda snap: (_W, _X), s, _IP, gate_q0=False,
            field_at=_field_at, pre_mse=pm)

    return dict(bl=bl, eq=eq, jbs0=jbs0, j_pre=j_pre, j_solved=j_solved,
                caller=caller, run=run)


def test_refusal_restores_the_pre_mse_state_and_verifies_it(monkeypatch):
    t = _setup(monkeypatch)
    nl, st, srec = t["run"]()
    bl, eq = t["bl"], t["eq"]
    # the refresh DID overwrite the baseline mid-chord (the bug's premise) ...
    assert len(eq.solved) >= 2
    # ... and the fallback put back the pre-MSE state, exactly
    np.testing.assert_array_equal(bl.j_phi, t["j_pre"])
    np.testing.assert_array_equal(bl.j_BS, t["jbs0"])
    assert bl.ohm_scale == 1.0
    assert bl.ip_closure["marker"] == "pre-MSE"
    # reasons: the restored record's own + the refusal, not the refresh's
    rs = bl.ip_closure["closure_limited_reasons"]
    assert "pre-MSE reason" in rs and "refresh reason" not in rs
    assert any("stage refused" in r for r in rs)
    assert bl.ip_closure["closure_limited"] is True
    # the caller's own closure state was restored too
    assert t["caller"]["restored"] == 1
    # the equilibrium was put back and the SOLVED pre-MSE current re-solved
    assert eq.replaced and isinstance(eq.replaced[-1], _Snap)
    np.testing.assert_array_equal(eq.solved[-1], t["j_solved"])
    # converged is the verification's verdict, with its numbers
    assert srec["restored_pre_mse"] is True
    chk = srec["restore_check"]
    assert chk["ok"] is True and srec["converged"] is True
    assert chk["r_j"] <= 1e-3 and chk["r_I"] <= 1e-4 and chk["dl_i"] <= 1e-3
    assert chk["pre_mse_final"] == dict(r_j=1e-4, r_I=1e-5)
    assert "holds the loop tolerances" in srec["stop_reason"]
    assert srec["final"]["r_j"] == chk["r_j"]


def test_a_restore_that_does_not_reproduce_is_recorded_not_converged(
        monkeypatch):
    """The restored re-solve lands 1 % away: the residual check fails at the
    loop's own tolerances, so the stage says so -- "flag" delivers the slice
    flagged with converged=False and the numbers; "raise" raises."""
    t = _setup(monkeypatch, drift_on_restore=0.01, on_fail="flag")
    nl, st, srec = t["run"]()
    assert srec["converged"] is False
    assert srec["restore_check"]["ok"] is False
    assert "does NOT reproduce" in srec["stop_reason"]
    assert srec["restore_check"]["r_j"] > 1e-3
    assert "fail_message" in srec
    t2 = _setup(monkeypatch, drift_on_restore=0.01, on_fail="raise")
    with pytest.raises(JBSNotConverged):
        t2["run"]()


def test_without_the_callers_snapshot_the_closure_current_is_resolved(
        monkeypatch):
    """No ``pre_mse`` (a direct caller): the stage still restores its own
    snapshot and re-solves the restored ``bl.j_phi`` -- and the check then
    reports honestly that this is not the solved pre-MSE state."""
    t = _setup(monkeypatch, on_fail="flag")
    nl, st, srec = t["run"](pre_mse=False)
    np.testing.assert_array_equal(t["eq"].solved[-1], t["j_pre"])
    assert srec["restore_check"]["solved"].startswith("bl.j_phi")
    assert srec["converged"] is False
