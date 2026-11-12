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


def _install(monkeypatch, fm, fail_calls=(), fail_trials=()):
    """``fail_calls``: consistent-solve call numbers that fail; ``fail_trials``: LM trial numbers
    (1-based, counting only solves labeled as trials) that fail."""
    calls = {"consistent": 0, "trial": 0}

    def consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        calls["consistent"] += 1
        fm_.n_solves += 3
        ok = calls["consistent"] not in fail_calls
        if ":trial" in label:
            calls["trial"] += 1
            ok = ok and calls["trial"] not in fail_trials
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
    assert out.status == "converged", out.status
    # forward differences of a linear model are exact, and the stopping tests are made on the
    # UNDAMPED Gauss-Newton step at a fresh center: the MAP point is reached to round-off
    assert abs(out.theta[0] - th[0]) < 1e-6
    assert out.covariance[0, 0] == pytest.approx(cov[0, 0], rel=1e-9)
    assert out.sigma[0] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-9)
    assert out.record["n_solves"] == fm.n_solves


def test_failed_trial_is_a_rejected_step(monkeypatch):
    fm = FakeFM()
    calls = _install(monkeypatch, fm, fail_trials=(1,))      # the first LM trial fails
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5, f=0.37))
    trials = out.record["iterations"][0]["trials"]
    assert trials[0]["status"] == "solve_failed" and trials[0]["chi2"] is None
    assert trials[1]["status"] == "consistent"
    # damping raised after the failure: x10 for the rejection, then x10 twice more without a solve because
    # the capped step did not change with lambda (deterministic on this problem: exactly x1000)
    assert trials[0]["capped"]
    assert trials[1]["lam"] == pytest.approx(1000.0 * trials[0]["lam"], rel=1e-12)
    th, cov = _expected(FIT.PARAMETERS["log_strength"]["prior"][1])
    assert abs(out.theta[0] - th[0]) < 1e-6
    assert calls["consistent"] >= 3


def test_failed_center_and_too_many_failures(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm, fail_calls=(1,))
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5))
    assert out.status == "failed_initial_consistency" and np.all(np.isnan(out.sigma))
    fm = FakeFM()
    _install(monkeypatch, fm, fail_trials=(1, 2, 3, 4))
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
    for key in ("max_iter", "lm_lambda0", "chi2_rel_tol", "grad_tol_chi2", "step_tol_sigma", "step_noise_factor",
                "noise_factor", "lm_lambda_max", "taper_restart", "fit_groups", "free", "priors", "bounds"):
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


def test_stopping_summary_records_which_test_fired(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm)
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), FIT.FitSettings(),
                          params0=dict(log_strength=0.5, f=0.37))
    sm = out.record["stopping_summary"]
    assert out.status == "converged" and sm["test"] == "a"
    assert sm["grad_floor"] == FIT.FitSettings().grad_tol_chi2 and sm["grad_floor_noise_scaled"] is False
    assert sm["final_gn_step"]["taken"] and sm["final_gn_step"]["kept_with_chi2_increase"] is False
    # test (b) through its noise-scaled branch: GN step 0.2 sigma (test (a) fails), decrement 0.04, floor
    # max(0.01, 0.05) = 0.05 set by the noise
    st = FIT.FitSettings()
    r = FIT.stopping_tests(np.array([[1.0]]), np.array([-0.2]), st, 0.05, [])
    assert r["converged"] and r["test"] == "b" and r["grad_floor"] == 0.05 and r["grad_floor_noise_scaled"] is True
    r0 = FIT.stopping_tests(np.array([[1.0]]), np.array([-0.2]), st, 0.0, [])
    assert not r0["converged"] and r0["test"] is None and r0["grad_floor_noise_scaled"] is False
    # a final Gauss-Newton step kept although chi^2 rose (within the noise) is stated with the increase
    log = [dict(final_gn_step=dict(accepted=True, status="consistent", chi2_change=0.003))]
    s2 = FIT.stopping_summary("converged", r, log)
    assert s2["test"] == "b" and s2["final_gn_step"]["kept_with_chi2_increase"] is True
    assert s2["final_gn_step"]["chi2_increase"] == pytest.approx(0.003)
    s3 = FIT.stopping_summary("stalled_no_significant_descent", r0, [dict()])
    assert s3["test"] is None and s3["final_gn_step"] == dict(taken=False)


def test_parameter_expansion_and_unknown():
    cfg = CM.CurrentModelSettings(r0=1.0)
    assert FIT._expand(("log_strength", "a_corr"), cfg) == ["log_strength", "a_corr_0", "a_corr_1", "a_corr_2"]
    with pytest.raises(ValueError):
        FIT._expand(("nope",), cfg)
    p = FIT._to_params([1.0, 0.01], ["log_strength", "a_corr_1"], {})
    assert p["log_strength"] == 1.0 and p["a_corr"] == (0.0, 0.01, 0.0)


def test_convergence_label_separates_stops_resting_on_the_noise():
    """``converged_within_noise``: test (b) held only through its noise-scaled floor, or the final
    Gauss-Newton step was kept with a chi^2 increase; the stopping tests themselves are unchanged."""
    st = FIT.FitSettings()
    # (b) through the noise floor: decrement 0.04 > grad_tol_chi2 0.01, floor max(0.01, 0.05)
    rb = FIT.stopping_tests(np.array([[1.0]]), np.array([-0.2]), st, 0.05, [])
    assert rb["converged"] and rb["test"] == "b"
    assert FIT.convergence_label(rb) == "converged_within_noise"
    # a decrement at or below the fixed floor implies every GN step component <= 0.1 posterior sigma
    # (Cauchy-Schwarz: delta_j^2 / (A^-1)_jj <= delta^T A delta), so test (a) holds first and the stop is plain
    rf = FIT.stopping_tests(np.array([[1.0]]), np.array([-0.09]), st, 0.05, [])
    assert rf["converged"] and rf["test"] == "a"
    assert FIT.convergence_label(rf) == "converged"
    # a record with test (b) and the decrement below the fixed floor (not reachable through stopping_tests,
    # see above) is labeled plain as well
    assert FIT.convergence_label(dict(rb, newton_decrement=0.005)) == "converged"
    # (a), with a final GN step kept downhill / uphill
    ra = FIT.stopping_tests(np.array([[1.0]]), np.array([-0.05]), st, 0.05, [])
    assert FIT.convergence_label(ra, dict(accepted=True, chi2_change=-1e-4)) == "converged"
    assert FIT.convergence_label(ra, dict(accepted=True, chi2_change=+1e-4)) == "converged_within_noise"
    assert FIT.convergence_label(ra, dict(accepted=False, chi2_change=+1.0)) == "converged"
    sm = FIT.stopping_summary("converged_within_noise", rb, [dict()])
    assert sm["within_noise_reasons"] == ["test (b) held only through its noise-scaled floor"]


def _install_lagged_bias(monkeypatch, fm, scale):
    """As :func:`_install`, but a LAGGED solve's residuals respond to theta with ``scale`` times the true
    sensitivity (a lagged column is only an approximation of the consistent one)."""
    calls = _install(monkeypatch, fm)
    lagged_mode = {"on": False}

    def lagged(fm_, lag, kin, src, params, cfg, ip=None, start=None, z0=None, label="", state=None):
        fm_.n_solves += 1
        fm_.theta_state = params["log_strength"]
        lagged_mode["on"] = True
        return SimpleNamespace(converged=True, status="converged"), {}, "converged"

    base_rv = FIT.residual_vector

    def residual_vector(fm_, ms, groups, settings, roles):
        r, idx, tab = base_rv(fm_, ms, groups, settings, roles)
        if lagged_mode["on"]:
            lagged_mode["on"] = False
            r = B + scale * A[:, 0] * (fm_.theta_state - TRUE)
        return r, idx, tab

    monkeypatch.setattr(FIT._cm, "lagged_solve", lagged)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    return calls


@pytest.mark.parametrize("scale", [1.0, 0.6])
def test_lagged_step_columns_keep_the_convergence_definition(monkeypatch, scale):
    """Lagged columns drive the steps; the stopping tests and the covariance use consistent columns, so
    the fit ends at the same MAP point with the same covariance as the consistent route even when the
    lagged columns are off by 40 %."""
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0)
    fm0 = FakeFM()
    _install(monkeypatch, fm0)
    ref = FIT.reconstruct(fm0, kin, src, cfg, FIT.FitSettings(), params0=dict(log_strength=0.5, f=0.37))
    fm = FakeFM()
    _install_lagged_bias(monkeypatch, fm, scale)
    out = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(jacobian_steps="lagged"),
                          params0=dict(log_strength=0.5, f=0.37))
    th, cov = _expected(FIT.PARAMETERS["log_strength"]["prior"][1])
    assert out.status == ref.status == "converged"
    assert abs(out.theta[0] - th[0]) < 1e-6
    assert out.covariance[0, 0] == pytest.approx(cov[0, 0], rel=1e-9)
    last = out.record["iterations"][-1]
    assert last["stopping_tests"]["converged"] and last["jacobian"]["column_kind"].startswith("consistent")
    assert any("jacobian_lagged" in e for e in out.record["iterations"])
    assert all(t["columns"] in ("lagged", "consistent") for e in out.record["iterations"] for t in e.get("trials", []))
    assert out.record["final_jacobian"]["column_kind"].startswith("consistent")
    with pytest.raises(ValueError):
        FIT.FitSettings(jacobian_steps="bogus")
