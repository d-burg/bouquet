"""``bouquet.kinrecon.fit`` machinery on a mocked forward model (no OFT).

The forward model is replaced by a linear one, ``r(theta) = A (theta - theta_true) + b``, so the
MAP point and the Laplace covariance are known in closed form; failure injection checks that a
failed trial is a rejected step, never an accepted state.  The live-solver closure tests are in
``test_fit_solver.py``.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM
from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet

A = np.array([[2.0], [-1.0], [0.5], [3.0]])
B = np.array([0.3, -0.2, 0.1, 0.05])
TRUE = 2.0


class FakeFM:
    def __init__(self):
        self.theta_state = None
        self.n_solves = 0
        self.probe_correction = np.zeros(0)
        self.gs = None
        self.ms = MeasurementSet("toy", 0.0, 0.0, {"flux_loop": ConstraintGroup(
            "flux_loop", ["a", "b", "c", "d"], np.zeros(4), np.ones(4), np.ones(4, bool))})
        self.rows_record = {"coils": {"per_coil": {}}}

    def snapshot(self):
        return dict(theta=self.theta_state)

    def restore(self, snap):
        self.theta_state = snap["theta"]

    def settings_record(self):
        return {"settings": {"mock": True}, "n_solves": self.n_solves}

    def boundary_distance(self):
        return {"rms_m": 0.0}


def _install(monkeypatch, fm, fail_calls=()):
    calls = {"consistent": 0}

    def consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None):
        calls["consistent"] += 1
        fm_.n_solves += 3
        ok = calls["consistent"] not in fail_calls
        if ok:
            fm_.theta_state = params["log_strength"]
        solve = SimpleNamespace(stats={"q_0": 1.0}, status="converged" if ok else "failed",
                                record={"policy": {"verification": {"verdict": "PASS"}}})
        return CM.ConsistentResult(ok, "consistent" if ok else "solve_failed", 3,
                                   {"orbit_z": None, "fractions": {}} if ok else None, {"lag": True}, solve,
                                   {"passes": [], "reason": None if ok else "injected"}, 0.01)

    def lagged(fm_, lag, kin, src, params, cfg, ip=None, start=None, z0=None, label=""):
        fm_.n_solves += 1
        fm_.theta_state = params["log_strength"]
        return SimpleNamespace(converged=True, status="converged"), {}, "converged"

    def residual_vector(fm_, ms, groups, settings, roles):
        r = A[:, 0] * (fm_.theta_state - TRUE) + B
        return r, [("flux_loop", s) for s in "abcd"], SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent)
    monkeypatch.setattr(FIT._cm, "lagged_solve", lagged)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    monkeypatch.setattr(FIT, "score", lambda *a, **k: SimpleNamespace(model={}, excluded={}, notes=[]))
    monkeypatch.setattr(FIT, "chi2_summary", lambda tab: {"total": (0.0, 0, 0.0)})
    monkeypatch.setattr(FIT, "constraints_table", lambda *a, **k: {})
    return calls


def _expected(prior_sigma):
    mu = FIT.PARAMETERS["log_strength"]["prior"][0]
    Ja = np.vstack((A, [[1.0 / prior_sigma]]))
    ra0 = np.concatenate((B - A[:, 0] * TRUE, [-mu / prior_sigma]))       # r = Ja theta + ra0
    th = -np.linalg.lstsq(Ja, ra0, rcond=None)[0]
    cov = np.linalg.inv(Ja.T @ Ja)
    return th, cov


def _kin():
    return SimpleNamespace(source="toy"), SimpleNamespace(source="toy", convention={})


def test_lm_reaches_the_linear_map_point_and_laplace_covariance(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm)
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0)
    out = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(), params0=dict(log_strength=0.5, f=0.37))
    th, cov = _expected(FIT.PARAMETERS["log_strength"]["prior"][1])
    assert out.status.startswith("converged"), out.status
    # forward differences of a linear model are exact; LM stops once a step is below
    # step_tol_sigma (0.1) posterior sigmas, so the answer is the MAP point to that accuracy
    tol = FIT.FitSettings().step_tol_sigma * np.sqrt(cov[0, 0])
    assert abs(out.theta[0] - th[0]) < tol
    assert out.covariance[0, 0] == pytest.approx(cov[0, 0], rel=1e-9)
    assert out.sigma[0] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-9)
    assert out.record["n_solves"] == fm.n_solves


def test_failed_trial_is_a_rejected_step(monkeypatch):
    fm = FakeFM()
    calls = _install(monkeypatch, fm, fail_calls=(2,))       # call 1 = center, call 2 = first trial
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5, f=0.37))
    trials = out.record["iterations"][0]["trials"]
    assert trials[0]["status"] == "solve_failed" and trials[0]["chi2"] is None
    assert trials[1]["status"] == "consistent"
    assert trials[1]["lam"] == pytest.approx(10.0 * trials[0]["lam"])     # damping raised after the failure
    th, cov = _expected(FIT.PARAMETERS["log_strength"]["prior"][1])
    assert abs(out.theta[0] - th[0]) < FIT.FitSettings().step_tol_sigma * np.sqrt(cov[0, 0])
    assert calls["consistent"] >= 3


def test_failed_center_and_too_many_failures(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm, fail_calls=(1,))
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5))
    assert out.status == "failed_initial_consistency" and np.all(np.isnan(out.sigma))
    fm = FakeFM()
    _install(monkeypatch, fm, fail_calls=(2, 3, 4, 5))
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5))
    assert out.status == "failed_trials"
    assert out.theta[0] == 0.5            # never moved: no failed trial was accepted


def test_settings_record_is_complete(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm)
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5))
    rec = out.record
    for key in ("free", "specs", "settings", "model_settings", "fit_groups", "excluded_groups", "iterations",
                "forward", "n_solves", "wall_s", "status", "final_jacobian", "center0", "constraints",
                "chi2_detail", "kinetics_source", "sources_source", "sources_convention"):
        assert key in rec, key
    for key in ("max_iter", "lm_lambda0", "chi2_rel_tol", "chi2_abs_tol", "step_tol_sigma", "step_noise_factor",
                "fit_groups", "free", "priors", "bounds"):
        assert key in rec["settings"], key
    spec = rec["specs"]["log_strength"]
    assert spec["prior"] == FIT.PARAMETERS["log_strength"]["prior"] and len(spec["bounds"]) == 2
    for key in ("pass_tol", "max_passes", "conductivity_model", "sawtooth", "pass_acceleration", "knots"):
        assert key in rec["model_settings"], key
    # MSE that is not declared E_r-corrected never enters the fit
    ms = MeasurementSet("toy", 0, 0, {"mse": ConstraintGroup("mse", ["m"], [0.1], [0.01], [True],
                                                              meta={"er_corrected": False})})
    use, why = FIT.fit_groups(ms, FIT.FitSettings())
    assert use == [] and "E_r" in why["mse"]


def test_parameter_expansion_and_unknown():
    cfg = CM.CurrentModelSettings(r0=1.0)
    assert FIT._expand(("log_strength", "a_corr"), cfg) == ["log_strength", "a_corr_0", "a_corr_1", "a_corr_2"]
    with pytest.raises(ValueError):
        FIT._expand(("nope",), cfg)
    p = FIT._to_params([1.0, 0.01], ["log_strength", "a_corr_1"], {})
    assert p["log_strength"] == 1.0 and p["a_corr"] == (0.0, 0.01, 0.0)
