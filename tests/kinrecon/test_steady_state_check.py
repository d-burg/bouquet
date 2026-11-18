"""The steady-state input check (``steady_state_check``) on a mocked forward model (no OFT).

It is ON by default in single-slice fits (:func:`bouquet.kinrecon.fit.reconstruct`) and in series, and it is a record
that steers nothing.  Checked here:
(a) the defaults (series setting and the fit's keyword) are on;
(b) the check zeroes the corrections, uses the given core treatment, scores the present state BEFORE its solve and
    the steady state after it, and restores the solver state and the probe correction afterwards, also when its solve
    fails or raises;
(c) a single-slice fit records it after the fit is complete: the fit's own ``n_solves`` is unchanged, the forward
    model's counter is not reset (it grows by exactly the check's solves, which the record states), and the solver
    is left in the state the fit left;
(d) switched off, without a consistent center, under a sawtooth core treatment, without an MSE group, or on the
    time-dependent path (a supplied state) it makes no solve;
(e) the series wrapper uses the series' first-slice core treatment and refuses the sawtooth modes it is not built
    for.
The bitwise on-versus-off identity of every fitted number on the live solver is in
``test_steady_state_check_solver.py``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM
from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon import series as SE
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet


class FakeFM:
    """State = a label; a solve sets it to the solve's label and changes the probe correction."""

    def __init__(self, mse=True):
        self.state = "fit-final"
        self.n_solves = 0
        self.probe_correction = np.array([1.0, 2.0])
        groups = {"flux_loop": ConstraintGroup("flux_loop", ["a"], np.zeros(1), np.ones(1), np.ones(1, bool))}
        if mse:
            groups["mse"] = ConstraintGroup("mse", ["m0", "m1"], np.zeros(2), np.ones(2), np.ones(2, bool))
        self.ms = MeasurementSet("toy", 0.0, 0.0, groups)

    def snapshot(self):
        return dict(state=self.state)

    def restore(self, snap):
        self.state = snap["state"]


def _install(monkeypatch, ok=True, raises=False, n=4):
    seen = []

    def consistent(fm, kin, src, params, cfg, label="", **kw):
        seen.append(dict(params=dict(params), core=cfg.core_relaxation, label=label, state_before=fm.state, kw=kw))
        if raises:
            raise RuntimeError("injected")
        fm.n_solves += n
        fm.state = label
        fm.probe_correction = np.array([9.0, 9.0])
        solve = SimpleNamespace(stats={"l_i": 0.8, "q_0": 1.1})
        return CM.ConsistentResult(ok, "consistent" if ok else "max_passes", 3, {} if ok else None, {}, solve, {},
                                   0.0)

    monkeypatch.setattr(FIT._cm, "consistent_solve", consistent)
    monkeypatch.setattr(FIT, "mse_by_radius", lambda fm, ms: dict(state=fm.state))
    return seen


def test_defaults_are_on():
    import inspect
    assert SE.SeriesSettings().steady_state_check is True
    assert inspect.signature(FIT.reconstruct).parameters["steady_state_check"].default is True
    assert SE.MSE_RADIAL_CLASSES_M == FIT.MSE_RADIAL_CLASSES_M


def test_check_zeroes_corrections_scores_both_states_and_restores(monkeypatch):
    seen = _install(monkeypatch)
    fm = FakeFM()
    cfg = CM.CurrentModelSettings(r0=1.0)
    p = dict(CM.DEFAULT_PARAMS, a_corr=(0.01, -0.02, 0.03), s_bs=1.3)
    rec = FIT.steady_state_check(fm, None, None, cfg, p, fm.ms, "x:steady_check")
    assert rec["computed"] is True and rec["status"] == "consistent"
    assert rec["own"] == dict(state="fit-final") and rec["steady_state"] == dict(state="x:steady_check")
    assert rec["l_i"] == 0.8 and rec["q_0"] == 1.1 and rec["n_solves"] == 4 and fm.n_solves == 4
    assert seen[0]["params"]["a_corr"] == (0.0, 0.0, 0.0) and seen[0]["params"]["s_bs"] == 1.3
    assert seen[0]["params"]["log_strength"] is None and seen[0]["core"] == "q_floor"
    assert seen[0]["state_before"] == "fit-final" and "start" not in seen[0]["kw"]
    assert fm.state == "fit-final" and np.array_equal(fm.probe_correction, [1.0, 2.0])
    assert "nothing reads it" in rec["note"]


@pytest.mark.parametrize("mode", ["not_converged", "raises"])
def test_check_restores_after_a_failed_or_raising_solve(monkeypatch, mode):
    _install(monkeypatch, ok=False, raises=(mode == "raises"))
    fm = FakeFM()
    rec = FIT.steady_state_check(fm, None, None, CM.CurrentModelSettings(r0=1.0), dict(CM.DEFAULT_PARAMS), fm.ms, "x")
    assert fm.state == "fit-final" and np.array_equal(fm.probe_correction, [1.0, 2.0])
    if mode == "raises":
        assert rec["computed"] is False and "RuntimeError: injected" in rec["reason"]
    else:
        assert rec["computed"] is True and rec["status"] == "max_passes" and rec["steady_state"] is None


def test_no_mse_group_makes_no_solve(monkeypatch):
    seen = _install(monkeypatch)
    fm = FakeFM(mse=False)
    rec = FIT.steady_state_check(fm, None, None, CM.CurrentModelSettings(r0=1.0), dict(CM.DEFAULT_PARAMS), fm.ms, "x")
    assert rec == dict(computed=False, reason="no MSE group at this slice") and not seen and fm.n_solves == 0


def _fake_fit(monkeypatch, center=True, n_fit=7):
    """``fit._reconstruct`` replaced: it makes ``n_fit`` solves, fills the starting parameters and leaves the solver in
    the state ``"fit-final"``."""
    def body(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None, label="fit",
             state_fn=None, speed=None, start_box=None):
        fm.n_solves += n_fit
        fm.state = "fit-final"
        if start_box is not None:
            start_box["params"] = dict(CM.DEFAULT_PARAMS, a_corr=(0.1, 0.1, 0.1))
        rec = dict(n_solves=n_fit, center=dict(theta=[0.1]) if center else None)
        return FIT.FitResult(status="converged" if center else "failed_initial_consistency", params={}, free=[],
                             theta=np.zeros(0), sigma=np.zeros(0), covariance=np.zeros((0, 0)), chi2={}, record=rec)

    monkeypatch.setattr(FIT, "_reconstruct", body)
    monkeypatch.setattr(FIT._speed, "validate", lambda eff, fm: None)


def test_single_slice_fit_records_the_check_after_the_fit(monkeypatch):
    seen = _install(monkeypatch)
    _fake_fit(monkeypatch)
    fm = FakeFM()
    out = FIT.reconstruct(fm, None, None, CM.CurrentModelSettings(r0=1.0), label="fx")
    chk = out.record["steady_state_check"]
    assert chk["computed"] is True and chk["n_solves"] == 4
    assert out.record["n_solves"] == 7                        # the fit's own count
    assert fm.n_solves == 7 + 4                               # the forward model's counter is never reset
    assert fm.state == "fit-final" and np.array_equal(fm.probe_correction, [1.0, 2.0])
    assert seen[0]["label"] == "fx:steady_check" and seen[0]["params"]["a_corr"] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("case", ["off", "no_center", "sawtooth_core", "state_fn"])
def test_single_slice_cases_without_a_solve(monkeypatch, case):
    seen = _install(monkeypatch)
    _fake_fit(monkeypatch, center=(case != "no_center"))
    fm = FakeFM()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit" if case == "sawtooth_core" else "q_floor")
    kw = dict(steady_state_check=False) if case == "off" else {}
    if case == "state_fn":
        kw["state_fn"] = lambda lag: None
    out = FIT.reconstruct(fm, None, None, cfg, **kw)
    assert not seen and fm.n_solves == 7
    if case == "state_fn":
        assert "steady_state_check" not in out.record           # the series records it per slice
        return
    chk = out.record["steady_state_check"]
    assert chk["computed"] is False
    assert {"off": "switched off", "no_center": "no consistent final center",
            "sawtooth_core": "not built for core_relaxation"}[case] in chk["reason"]


def test_series_wrapper_uses_the_first_slice_core_and_refuses_sawtooth_modes(monkeypatch):
    seen = _install(monkeypatch)
    fm = FakeFM()
    sl = SimpleNamespace(measurements=fm.ms, kinetics=None, sources=None, label="s3")
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="none")
    rec = SE._steady_state_check(fm, sl, cfg, SE.SeriesSettings(), dict(CM.DEFAULT_PARAMS))
    assert rec["computed"] is True and seen[0]["core"] == SE.FIRST_SLICE_CORE["q_floor"]
    assert seen[0]["label"] == "s3:steady_check" and fm.state == "fit-final"
    rec = SE._steady_state_check(fm, sl, cfg, SE.SeriesSettings(sawtooth="periodic"), dict(CM.DEFAULT_PARAMS))
    assert rec["computed"] is False and "not built for sawtooth" in rec["reason"] and len(seen) == 1
