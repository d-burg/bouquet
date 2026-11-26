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

#: The mocked model is linear in ``log_strength``: the fits free it as under the opt-in periodic-orbit core treatment
#: (``core_relaxation="periodic_orbit"``, whose free set was the package default before the q floor).
SAW_FREE = ("log_strength", "a_corr")
A = np.array([[2.0], [-1.0], [0.5], [3.0]])
B = np.array([0.3, -0.2, 0.1, 0.05])
TRUE = 2.0


def _fs(**kw):
    """Fit settings for the tests of the fit machinery: the moving anchor target (``FitSettings.anchor_follow``, on by
    default) is named off, since these tests are not about it.  The anchor-target tests below name their own."""
    return FIT.FitSettings(anchor_follow=False, **kw)


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
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    out = FIT.reconstruct(fm, kin, src, cfg, _fs(free=SAW_FREE), params0=dict(log_strength=0.5, f=0.37))
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
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE),
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
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE),
                          params0=dict(log_strength=0.5))
    assert out.status == "failed_initial_consistency" and np.all(np.isnan(out.sigma))
    fm = FakeFM()
    _install(monkeypatch, fm, fail_trials=(1, 2, 3, 4))
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE),
                          params0=dict(log_strength=0.5))
    assert out.status == "failed_trials"
    assert out.theta[0] == 0.5            # never moved: no failed trial was accepted


def test_settings_record_is_complete(monkeypatch):
    fm = FakeFM()
    _install(monkeypatch, fm)
    kin, src = _kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE),
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
    for key in ("pass_tol", "max_passes", "conductivity_model", "core_relaxation", "q_floor", "pass_acceleration",
                "knots"):
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
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE),
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
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    assert FIT._expand(("log_strength", "a_corr"), cfg) == ["log_strength", "a_corr_0", "a_corr_1", "a_corr_2"]
    with pytest.raises(ValueError):
        FIT._expand(("nope",), cfg)
    p = FIT._to_params([1.0, 0.01], ["log_strength", "a_corr_1"], {})
    assert p["log_strength"] == 1.0 and p["a_corr"] == (0.0, 0.01, 0.0)


def test_default_step_cap_of_the_correction_amplitudes():
    """The default trust cap of one LM step in each correction amplitude is 0.04 of Ip; the other caps are as before,
    and a step larger than the cap is shrunk as a whole to it."""
    st = FIT.FitSettings()
    assert st.max_step["a_corr"] == 0.04
    assert st.max_step == dict(log_strength=0.5, f=0.1, s_pfast=0.1, s_cd=0.1, s_bs=0.2, a_corr=0.04, ip_rel=0.02,
                               s_temp=0.05, s_press=0.05)
    assert st.as_dict()["max_step"]["a_corr"] == 0.04
    cap = np.array([0.04, 0.04])
    delta, capped = FIT.lm_step(np.eye(2), np.array([-0.1, 0.02]), 0.0, cap)
    assert capped and np.allclose(delta, [0.04, -0.008], rtol=0, atol=1e-15)
    delta, capped = FIT.lm_step(np.eye(2), np.array([-0.03, 0.02]), 0.0, cap)
    assert not capped and np.allclose(delta, [0.03, -0.02], rtol=0, atol=1e-15)


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
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    fm0 = FakeFM()
    _install(monkeypatch, fm0)
    ref = FIT.reconstruct(fm0, kin, src, cfg, _fs(free=SAW_FREE), params0=dict(log_strength=0.5, f=0.37))
    fm = FakeFM()
    _install_lagged_bias(monkeypatch, fm, scale)
    out = FIT.reconstruct(fm, kin, src, cfg, _fs(free=SAW_FREE, jacobian_steps="lagged"),
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


# ============================================================== plasma current fitted, not pinned
SIG_IP = 0.01          # sigma of the Ip row as a fraction of Ip
Z_IP = 1.7             # the "measured" Ip lies Z_IP sigma above the model's Ip at ip_rel = 0
C_IP = np.array([0.4, -0.3, 0.2, 0.6])     # data-row response to ip_rel (per unit ip_rel / SIG_IP)


class FakeFMIp(FakeFM):
    def __init__(self):
        super().__init__()
        self.ms.groups["ip"] = ConstraintGroup("ip", ["ip"], np.array([1.0]), np.array([SIG_IP]), np.array([True]))
        self.ip_state = 0.0


def _install_ip(monkeypatch, fm):
    _install(monkeypatch, fm)

    def consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        fm_.n_solves += 3
        fm_.theta_state = params["log_strength"]
        fm_.ip_state = CM.full_params(params)["ip_rel"]
        solve = SimpleNamespace(stats={"q_0": 1.0}, status="converged",
                                record={"policy": {"verification": {"verdict": "PASS"}}})
        return CM.ConsistentResult(True, "consistent", 3, {"orbit_z": None, "fractions": {}}, {"lag": True}, solve,
                                   {"passes": [], "reason": None}, 0.01)

    def residual_vector(fm_, ms, groups, settings, roles):
        x = fm_.ip_state / SIG_IP
        r = A[:, 0] * (fm_.theta_state - TRUE) + B + C_IP * x
        rows, idx = [r], [("flux_loop", s) for s in "abcd"]
        if "ip" in groups:
            rows.append(np.array([x - Z_IP]))
            idx.append(("ip", "ip"))
        return np.concatenate(rows), idx, SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)


def test_ip_target_is_a_fit_parameter_with_its_measurement_row(monkeypatch):
    """The Ip row is the datum of ``ip_rel`` (no prior row); the fit reaches the linear MAP point of the joint
    problem, and ``ip_target='measured'`` keeps the earlier pinned target."""
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    fm = FakeFMIp()
    _install_ip(monkeypatch, fm)
    out = FIT.reconstruct(fm, kin, src, cfg, _fs(free=("log_strength",)), params0=dict(log_strength=0.5))
    assert out.free == ["log_strength", "ip_rel"] and out.record["ip_target"]["mode"] == "fitted"
    assert out.status == "converged" and out.record["specs"]["ip_rel"]["prior"] == (None, None)
    mu, sp = FIT.PARAMETERS["log_strength"]["prior"]
    # linear model in (theta, x = ip_rel/SIG_IP): rows A theta + C x + (B - A TRUE), x - Z_IP, (theta - mu)/sp
    J = np.vstack((np.column_stack((A[:, 0], C_IP)), [[0.0, 1.0]], [[1.0 / sp, 0.0]]))
    r0 = np.concatenate((B - A[:, 0] * TRUE, [-Z_IP], [-mu / sp]))
    th = -np.linalg.lstsq(J, r0, rcond=None)[0]
    assert out.theta[0] == pytest.approx(th[0], abs=1e-6)
    assert out.theta[1] == pytest.approx(th[1] * SIG_IP, abs=1e-8)
    assert out.chi2["n_params"] == 2
    fm2 = FakeFMIp()
    _install_ip(monkeypatch, fm2)
    pinned = FIT.reconstruct(fm2, kin, src, cfg, _fs(free=("log_strength",), ip_target="measured"),
                             params0=dict(log_strength=0.5))
    assert pinned.free == ["log_strength"] and pinned.record["ip_target"]["mode"] == "measured"
    with pytest.raises(ValueError, match="ip_target"):
        FIT.FitSettings(free=("ip_rel",), ip_target="measured")
    with pytest.raises(ValueError, match="ip_target"):
        FIT.FitSettings(ip_target="pinned")


def test_ip_record_and_target():
    ms = MeasurementSet("toy", 0, 0, {"ip": ConstraintGroup("ip", ["ip"], [1.0e6], [3.0e4], [True])})
    tab = SimpleNamespace(model={"ip": np.array([0.97e6])})
    rec = FIT.ip_record(tab, ms, dict(ip_rel=-0.03), True)
    assert rec["pull_measured_minus_fitted_sigma"] == pytest.approx(1.0, rel=1e-12)
    assert rec["fitted_A"] == 0.97e6 and rec["ip_target_fitted"] and rec["ip_rel"] == -0.03
    assert CM.ip_target(-1.0e6, dict(ip_rel=-0.03)) == pytest.approx(0.97e6, rel=1e-15)
    assert FIT.ip_record(None, MeasurementSet("toy", 0, 0, {}), {}, False)["available"] is False


def test_coil_residual_record_is_a_record_of_the_constrained_coils():
    from bouquet.kinrecon.interfaces import ResidualTable
    names = ["A", "B", "C", "D"]
    g = ConstraintGroup("pf_current", names, [1.0e5, -2.0e5, 5.0e4, 1.0e3], [1.0e3, 2.0e3, 1.0e3, 1.0e2],
                        [True, True, True, True], geometry={"coil_names": names})
    ms = MeasurementSet("toy", 0, 0, {"pf_current": g})
    tab = ResidualTable(ms, model={"pf_current": np.array([1.0e5 - 3.5e3, -2.0e5 + 2.0e3, 5.0e4, 9.0e3])})
    roles = dict(A="fitted", B="fitted", C="held", D="free")
    rec = FIT.coil_residual_record(tab, ms, roles)
    assert rec["available"] and rec["n"] == 3 and "D" not in rec["residual_sigma"]       # the free coil is left out
    assert rec["residual_sigma"]["A"] == pytest.approx(-3.5) and rec["residual_sigma"]["B"] == pytest.approx(1.0)
    assert rec["max_coil"] == "A" and rec["max_abs_sigma"] == pytest.approx(3.5)
    assert rec["rms_sigma"] == pytest.approx(np.sqrt((3.5 ** 2 + 1.0 + 0.0) / 3.0))
    assert rec["flagged"] and [f["coil"] for f in rec["flags"]] == ["A"]
    assert rec["report_level_sigma"] == FIT.COIL_RESIDUAL_REPORT_LEVEL == 3.0
    quiet = FIT.coil_residual_record(ResidualTable(ms, model={"pf_current": np.asarray(g.measured, float)}), ms, roles)
    assert not quiet["flagged"] and quiet["max_abs_sigma"] == 0.0
    assert FIT.coil_residual_record(None, ms, roles)["available"] is False
    assert FIT.coil_residual_record(tab, ms, dict.fromkeys(names, "free"))["available"] is False


def test_noise_is_the_same_measurement_with_lagged_columns(monkeypatch):
    """The noise measurement re-solves the center from a CONSISTENT state in both column modes: here a
    re-solve from a lagged column's state would carry an offset (the mocked lagged state is not a
    consistent one), so the lagged mode would otherwise measure a larger noise."""
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    out = {}
    for mode in ("consistent", "lagged"):
        fm = FakeFM()
        _install(monkeypatch, fm)

        def consistent(fm_, kin_, src_, params, cfg_, start, z0, label, initial_profiles=None, state_fn=None):
            fm_.n_solves += 3
            lagged_start = isinstance(start, dict) and start.get("kind") == "lagged"
            fm_.theta_state = params["log_strength"] + (0.05 if lagged_start else 0.0)
            fm_.kind = "consistent"
            solve = SimpleNamespace(stats={"q_0": 1.0}, status="converged",
                                    record={"policy": {"verification": {"verdict": "PASS"}}})
            return CM.ConsistentResult(True, "consistent", 3, {"orbit_z": None, "fractions": {}}, {"lag": True},
                                       solve, {"passes": [], "reason": None}, 0.01)

        def lagged(fm_, lag, kin_, src_, params, cfg_, ip=None, start=None, z0=None, label="", state=None):
            fm_.n_solves += 1
            fm_.theta_state = params["log_strength"]
            fm_.kind = "lagged"
            return SimpleNamespace(converged=True, status="converged"), {}, "converged"

        fm.kind = "consistent"
        fm.snapshot = (lambda f=fm: dict(theta=f.theta_state, kind=f.kind))
        monkeypatch.setattr(FIT, "_consistent", consistent)
        monkeypatch.setattr(FIT._cm, "lagged_solve", lagged)
        res = FIT.reconstruct(fm, kin, src, cfg, _fs(free=SAW_FREE, jacobian_steps=mode),
                              params0=dict(log_strength=0.5))
        out[mode] = res.record["noise"]
    assert out["consistent"]["status"] == out["lagged"]["status"] == "ok"
    assert out["lagged"]["chi2"] == out["consistent"]["chi2"] == 0.0
    assert out["lagged"]["norm"] == out["consistent"]["norm"]
    assert "consistent solve" in out["lagged"]["method"]


def test_core_vs_floor_is_a_record_that_cannot_change_a_result(monkeypatch):
    """``record["core_vs_floor"]`` (whether the data pull the core below the q floor) is made once, after the result,
    and nothing reads it: a recorder that raises or returns anything leaves the fit's status, parameters,
    covariance, chi^2 and solve count unchanged."""
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    outs, calls = {}, []
    real = CM.core_vs_floor_record

    def counting(core, cce):
        calls.append(1)
        return real(core, cce)

    def raising(core, cce):
        calls.append(1)
        raise RuntimeError("injected")

    def garbage(core, cce):
        calls.append(1)
        return dict(q_axis_model_minus_floor=-99.0, d_current_inside_core_over_ip=99.0, verdict="below")

    for name, fn in (("real", counting), ("raising", raising), ("garbage", garbage)):
        fm = FakeFM()
        _install(monkeypatch, fm)
        monkeypatch.setattr(CM, "core_vs_floor_record", fn)
        n0 = len(calls)
        outs[name] = FIT.reconstruct(fm, kin, src, cfg, _fs(free=SAW_FREE),
                                     params0=dict(log_strength=0.5, f=0.37))
        assert len(calls) == n0 + 1                       # made once, at the end
    ref = outs["real"]
    assert ref.record["core_vs_floor"]["note"].startswith("record only")
    assert outs["raising"].record["core_vs_floor"]["computed"] is False
    assert outs["garbage"].record["core_vs_floor"]["verdict"] == "below"
    for name in ("raising", "garbage"):
        o = outs[name]
        assert o.status == ref.status and o.record["n_solves"] == ref.record["n_solves"]
        np.testing.assert_array_equal(o.theta, ref.theta)
        np.testing.assert_array_equal(o.covariance, ref.covariance)
        np.testing.assert_equal(o.chi2, ref.chi2)
        assert o.params == ref.params
    # the record carries the floor and correction quantities as they are, with no threshold or verdict
    core = dict(q_floor=1.0, acted=True, q_floor_met=True, q_axis_model_minus_floor=-0.03, q_min_model_minus_floor=-0.04,
                rho_q_min_model=0.1)
    cce = dict(computed=True, core_rho=0.3, d_current_inside_core_over_ip=0.01, d_q_axis_model=-0.05)
    r = real(core, cce)
    assert r["q_axis_model_minus_floor"] == -0.03 and r["d_current_inside_core_over_ip"] == 0.01
    assert r["d_q_axis_model"] == -0.05 and not any(isinstance(v, bool) and k not in ("floor_acted", "q_floor_met",
                                                                                         "correction_effect_computed")
                                                    for k, v in r.items())


def test_default_fit_records_the_sawtooth_parameters_as_not_used(monkeypatch):
    """On a path that does not read them (the default core treatment), the sawtooth parameters are recorded as not
    used rather than as their opt-in defaults; under the opt-in orbit they are kept."""
    kin, src = _kin()
    fm = FakeFM()
    _install(monkeypatch, fm)
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          _fs(free=SAW_FREE), params0=dict(log_strength=0.5, f=0.37))
    assert out.params["f"] == 0.37 and "params_not_used" not in out.record
    # the default treatment (q floor; the sawtooth parameters cannot be freed there): recorded as None
    with pytest.raises(ValueError, match="opt-in sawtooth"):
        FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), _fs(free=SAW_FREE))
    fm = FakeFM()
    _install(monkeypatch, fm)

    def consistent(fm_, kin_, src_, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        fm_.n_solves += 1
        fm_.theta_state = TRUE + params["a_corr"][0]          # the mocked model reads the first correction instead
        solve = SimpleNamespace(stats={"q_0": 1.0}, status="converged",
                                record={"policy": {"verification": {"verdict": "PASS"}}})
        return CM.ConsistentResult(True, "consistent", 3, {"orbit_z": None, "fractions": {}}, {"lag": True}, solve,
                                   {"passes": [], "reason": None}, 0.01)

    monkeypatch.setattr(FIT, "_consistent", consistent)
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), _fs())
    assert out.params["f"] is None and out.params["log_strength"] is None
    assert out.record["params_not_used"] == ["log_strength", "f"]


class FakeFMAnchor(FakeFM):
    """The mocked forward model with a movable anchor target: the misfit offset is scaled by ``scale[n_moves]``."""

    def __init__(self, scale):
        super().__init__()
        self.scale = list(scale)
        self.target = 0                      # the number of moves the target now stands at
        self.moves = 0                       # every move_anchor_target call

    def snapshot(self):
        return dict(theta=self.theta_state)

    def anchor_target(self):
        return self.target

    def set_anchor_target(self, target):
        self.target = target

    def move_anchor_target(self, fraction):
        self.target += 1
        self.moves += 1
        return dict(fraction=fraction, move_rms_mm=1.0)


def _install_anchor(monkeypatch, fm, fail_at_target=None):
    calls = _install(monkeypatch, fm)
    consistent = FIT._consistent

    def consistent_t(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        out = consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=initial_profiles,
                         state_fn=state_fn)
        if fail_at_target is not None and fm_.target == fail_at_target:
            return CM.ConsistentResult(False, "solve_failed", 3, None, {"lag": True}, out.solve,
                                       {"passes": [], "reason": "injected"}, 0.01)
        return out

    def residual_vector(fm_, ms, groups, settings, roles):
        r = A[:, 0] * (fm_.theta_state - TRUE) + B * fm_.scale[fm_.target]
        return r, [("flux_loop", s) for s in "abcd"], SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent_t)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    return calls


def _anchor_fit(fm, **kw):
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    return FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=SAW_FREE, **kw),
                           params0=dict(log_strength=0.5, f=0.37))


def test_anchor_follow_is_on_by_default_and_checked():
    s = FIT.FitSettings()
    assert s.anchor_follow is True and s.anchor_follow_fraction == 0.5 and s.anchor_follow_max_steps == 8
    assert s.as_dict()["anchor_follow"] is True
    for bad in (dict(anchor_follow=1), dict(anchor_follow_fraction=0.0), dict(anchor_follow_fraction=1.5),
                dict(anchor_follow_max_steps=0), dict(anchor_follow_max_steps=2.5)):
        with pytest.raises(ValueError):
            FIT.FitSettings(**bad)


def test_fit_with_anchor_follow_off_never_moves_the_anchor_target(monkeypatch):
    fm = FakeFMAnchor([1.0, 0.5])
    _install_anchor(monkeypatch, fm)
    out = _anchor_fit(fm, anchor_follow=False)
    assert out.status == "converged" and fm.moves == 0 and fm.target == 0
    assert "anchor_follow" not in out.record


def test_default_fit_moves_the_anchor_target(monkeypatch):
    """With the default settings the fit follows: the first move lowers the fitted chi^2 and is kept, the second
    raises it and is not kept."""
    fm = FakeFMAnchor([1.0, 0.5, 0.7])
    _install_anchor(monkeypatch, fm)
    out = _anchor_fit(fm)
    rec = out.record["anchor_follow"]
    assert rec["applied"] and rec["max_steps"] == 8 and rec["fraction"] == 0.5
    assert rec["n_moves_kept"] == 1 and fm.moves == 2 and fm.target == 1
    assert [s["kept"] for s in rec["steps"]] == [True, True, False] and "did not fall" in rec["stop"]


def test_anchor_follow_keeps_moves_while_the_fitted_chi2_falls(monkeypatch):
    fm = FakeFMAnchor([1.0, 0.5, 0.25, 0.6, 0.1])        # falls twice, then rises: the third move is not kept
    _install_anchor(monkeypatch, fm)
    ref = FakeFMAnchor([0.25])
    _install_anchor(monkeypatch, ref)
    want = _anchor_fit(ref, anchor_follow=False)
    _install_anchor(monkeypatch, fm)
    out = _anchor_fit(fm, anchor_follow=True)
    rec = out.record["anchor_follow"]
    assert rec["applied"] and rec["n_moves_kept"] == 2 and fm.moves == 3
    assert fm.target == 2                                   # the target put back to the last kept move
    assert [s["kept"] for s in rec["steps"]] == [True, True, True, False]
    c = [s["chi2_data"] for s in rec["steps"]]
    assert c[1] < c[0] and c[2] < c[1] and c[3] > c[2]
    assert "did not fall" in rec["stop"]
    # the result is the last kept fit: the fit of the twice-moved target, and the state is its state
    assert out.status == "converged" and out.chi2["data"] == pytest.approx(c[2], rel=1e-12)
    assert out.theta[0] == pytest.approx(want.theta[0], abs=1e-6)
    assert fm.theta_state == pytest.approx(out.theta[0], abs=1e-6)
    assert rec["n_solves_total"] == fm.n_solves


def test_anchor_follow_puts_the_target_back_when_a_refit_fails(monkeypatch):
    fm = FakeFMAnchor([1.0, 0.5, 0.25])
    _install_anchor(monkeypatch, fm, fail_at_target=2)     # every solve on the twice-moved target fails
    out = _anchor_fit(fm, anchor_follow=True)
    rec = out.record["anchor_follow"]
    assert rec["n_moves_kept"] == 1 and fm.target == 1 and out.status == "converged"
    assert rec["steps"][-1]["kept"] is False and "refit" in rec["stop"]


def test_anchor_follow_stops_at_the_cap_and_is_not_applied_in_a_series(monkeypatch):
    fm = FakeFMAnchor([1.0, 0.8, 0.6, 0.4, 0.2])
    _install_anchor(monkeypatch, fm)
    out = _anchor_fit(fm, anchor_follow=True, anchor_follow_max_steps=2)
    assert out.record["anchor_follow"]["n_moves_kept"] == 2 and "cap" in out.record["anchor_follow"]["stop"]
    assert fm.target == 2


def test_anchor_follow_also_follows_with_a_supplied_state(monkeypatch):
    """A filter fit of a series slice (``state_fn`` given) follows by the same rule; no steady-state record is made."""
    fm = FakeFMAnchor([1.0, 0.5, 0.7])
    _install_anchor(monkeypatch, fm)
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    out = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=SAW_FREE, anchor_follow=True),
                          params0=dict(log_strength=0.5, f=0.37), state_fn=lambda lag, params=None: None)
    rec = out.record["anchor_follow"]
    assert rec["applied"] and rec["n_moves_kept"] == 1 and fm.target == 1
    assert "steady_state_check" not in out.record


def test_anchor_offset_is_carried_onto_another_polygon_by_angle():
    """``ForwardModel.anchor_offset`` / ``apply_anchor_offset`` (no solver): the displacement of a moved target is
    carried onto another slice's reference polygon along the rays from the magnetic axis."""
    from bouquet.kinrecon.forward import ForwardModel

    def model(ref):
        fm = ForwardModel.__new__(ForwardModel)
        fm.reference_boundary = ref
        fm._anchor_polygon = ref.copy()
        fm._iso = dict(points=ref[::2].copy(), weight=3.0)
        fm.settings = SimpleNamespace(isoflux_stride=2)
        fm.gs = SimpleNamespace(o_point=(1.72, 0.02))
        return fm

    ax = np.array([1.72, 0.02])
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    ref = np.c_[1.7 + 0.6 * np.cos(th), 1.0 * np.sin(th)]
    fm = model(ref)
    assert fm.anchor_offset() is None                      # nothing moved: nothing to carry
    d = ref - ax
    rr = np.hypot(d[:, 0], d[:, 1])
    dr = -0.02 + 0.01 * np.cos(np.arctan2(d[:, 1], d[:, 0]))
    fm.set_anchor_target(ax + (rr + dr)[:, None] * d / rr[:, None])
    off = fm.anchor_offset()
    assert np.allclose(off["dr"], dr, atol=1e-14)
    assert fm._iso["weight"] == 3.0 and len(fm._iso["points"]) == 40 and np.array_equal(fm.reference_boundary, ref)
    th2 = np.linspace(0.1, 2 * np.pi + 0.1, 61, endpoint=False)
    fm2 = model(np.c_[1.71 + 0.61 * np.cos(th2), 1.02 * np.sin(th2)])
    rec = fm2.apply_anchor_offset(off)
    d2 = fm2.reference_boundary - ax
    t2 = fm2.anchor_target() - ax
    want = -0.02 + 0.01 * np.cos(np.arctan2(d2[:, 1], d2[:, 0]))
    assert np.max(np.abs(np.hypot(t2[:, 0], t2[:, 1]) - np.hypot(d2[:, 0], d2[:, 1]) - want)) < 5e-5
    assert np.allclose(t2[:, 0] * d2[:, 1] - t2[:, 1] * d2[:, 0], 0.0, atol=1e-12)     # on the same rays
    assert rec["carried"] and rec["mean_mm"] == pytest.approx(-20.0, abs=0.5)
    with pytest.raises(ValueError):
        fm2.set_anchor_target(ref)                          # another polygon's shape is refused


# ------------------------------------------------------------------ final Jacobian of the returned fit only (option)
class FakeFMAnchorCase(FakeFMAnchor):
    """:class:`FakeFMAnchor` with an edge-taper case (``locked``: as the forward model's case after its first solve)."""

    def __init__(self, scale, locked=False):
        super().__init__(scale)
        self.case = SimpleNamespace(locked=locked)


def _install_final_fail(monkeypatch, fm, final_fail_targets=()):
    """:func:`_install_anchor` with every column solve of a final Jacobian failing at the given targets."""
    calls = _install_anchor(monkeypatch, fm)
    consistent = FIT._consistent

    def consistent_f(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        out = consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=initial_profiles,
                         state_fn=state_fn)
        if ":final:col" in label and fm_.target in final_fail_targets:
            return CM.ConsistentResult(False, "solve_failed", 3, None, {"lag": True}, out.solve,
                                       {"passes": [], "reason": "injected"}, 0.01)
        return out

    monkeypatch.setattr(FIT, "_consistent", consistent_f)
    return calls


def _pair(monkeypatch, scale, locked=False, final_fail_targets=(), **kw):
    """The same moving-target fit with the option off and on: (result, model) for each."""
    out = []
    for on in (False, True):
        fm = FakeFMAnchorCase(scale, locked=locked)
        _install_final_fail(monkeypatch, fm, final_fail_targets)
        out.append((_anchor_fit(fm, anchor_follow=True, anchor_follow_defer_final_jacobian=on, **kw), fm))
    return out


def _same_returned_fit(a, b):
    (ra, fa), (rb, fb) = a, b
    assert ra.status == rb.status
    assert ra.params == rb.params and ra.free == rb.free
    assert np.array_equal(ra.theta, rb.theta)
    assert np.array_equal(ra.sigma, rb.sigma, equal_nan=True)
    assert np.array_equal(ra.covariance, rb.covariance, equal_nan=True)
    assert repr(ra.chi2) == repr(rb.chi2)                          # repr: NaN entries compare equal
    assert fa.target == fb.target and fa.theta_state == fb.theta_state
    ka, kb = ra.record["anchor_follow"], rb.record["anchor_follow"]
    assert repr(ka["steps"]) == repr(kb["steps"]) and ka["stop"] == kb["stop"] and ka["n_moves_kept"] == kb["n_moves_kept"]
    assert repr(ra.record["final_jacobian"]) == repr(rb.record["final_jacobian"])
    assert ra.record["n_solves"] == rb.record["n_solves"]          # the returned fit's own solves
    for key in ("stopping_summary", "final_stopping_tests", "iterations", "settings", "covariance_column_flags"):
        assert repr(ra.record.get(key)) == repr(rb.record.get(key)), key


def test_deferred_final_jacobian_is_off_by_default_checked_and_not_recorded_when_off():
    s = FIT.FitSettings()
    assert s.anchor_follow_defer_final_jacobian is False
    assert "anchor_follow_defer_final_jacobian" not in s.as_dict()          # the default record as before
    assert FIT.FitSettings(anchor_follow_defer_final_jacobian=True).as_dict()["anchor_follow_defer_final_jacobian"]
    with pytest.raises(ValueError):
        FIT.FitSettings(anchor_follow_defer_final_jacobian=1)


@pytest.mark.parametrize("scale,kw", [([1.0, 0.5, 0.25, 0.6, 0.1], {}),            # a move not kept
                                      ([1.0, 0.8, 0.6, 0.4, 0.2], dict(anchor_follow_max_steps=2)),   # the cap
                                      ([1.0, 1.5], {})])                             # the first move not kept
def test_deferred_final_jacobian_returns_the_same_fit_with_fewer_solves(monkeypatch, scale, kw):
    off, on = _pair(monkeypatch, scale, **kw)
    _same_returned_fit(off, on)
    d = on[0].record["anchor_follow"]["final_jacobian_deferred"]
    assert d["on"] and not d["backtracked"] and "final_jacobian_deferred" not in off[0].record["anchor_follow"]
    n_kept = off[0].record["anchor_follow"]["n_moves_kept"]
    assert on[1].n_solves < off[1].n_solves if n_kept else on[1].n_solves <= off[1].n_solves


def test_deferred_final_jacobian_failing_for_the_returned_fit_gives_the_default_answer(monkeypatch):
    """The returned fit's final Jacobian fails: by default that move is not kept, and so it is with the option (the
    fit before it is completed and returned in its place)."""
    off, on = _pair(monkeypatch, [1.0, 0.5, 0.25, 0.6], final_fail_targets=(2,))
    assert off[0].record["anchor_follow"]["n_moves_kept"] == 1 and off[1].target == 1
    assert "converged_no_final_jacobian" in off[0].record["anchor_follow"]["stop"]
    _same_returned_fit(off, on)
    assert on[0].record["anchor_follow"]["final_jacobian_deferred"]["backtracked"][0]["step"] == 2


def test_deferred_final_jacobian_failing_for_the_returned_first_fit_gives_the_default_answer(monkeypatch):
    """The first fit is returned (its first move is not kept) and its final Jacobian fails: the default makes no move
    after it; the option made one, and gives the default's answer and steps."""
    off, on = _pair(monkeypatch, [1.0, 1.5], final_fail_targets=(0,))
    assert off[0].status == "converged_no_final_jacobian" and off[1].moves == 0 and on[1].moves == 1
    _same_returned_fit(off, on)
    assert on[1].target == 0


@pytest.mark.parametrize("fail_at", [0, 1])
def test_deferred_final_jacobian_states_where_it_can_differ(monkeypatch, fail_at):
    """The one stated difference: a final Jacobian of an EARLIER kept fit fails while the returned one's succeeds.
    The default stops at that move; the option never computes that Jacobian and goes on."""
    off, on = _pair(monkeypatch, [1.0, 0.5, 0.25, 0.6], final_fail_targets=(fail_at,))
    assert off[1].target == max(fail_at - 1, 0) and on[1].target == 2
    assert "earlier kept fit" in on[0].record["anchor_follow"]["final_jacobian_deferred"]["rule"]


def test_deferred_final_jacobian_waits_for_the_edge_taper_case_lock(monkeypatch):
    """An undecided edge-taper case (``locked`` None): no fit is deferred, so a skipped Jacobian can never decide the
    case; the solves are those of the default."""
    off, on = _pair(monkeypatch, [1.0, 0.5, 0.25, 0.6], locked=None)
    _same_returned_fit(off, on)
    assert on[1].n_solves == off[1].n_solves
    assert all("note" in c for c in on[0].record["anchor_follow"]["final_jacobian_deferred"]["completed"])


def test_deferred_final_jacobian_is_refused_with_a_supplied_state(monkeypatch):
    fm = FakeFMAnchorCase([1.0, 0.5])
    _install_anchor(monkeypatch, fm)
    kin, src = _kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    with pytest.raises(ValueError):
        FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=SAW_FREE, anchor_follow=True,
                                                           anchor_follow_defer_final_jacobian=True),
                        params0=dict(log_strength=0.5, f=0.37), state_fn=lambda lag, params=None: None)


def test_boundary_crossings_all_points_at_once_equal_one_ray_at_a_time(monkeypatch):
    """``ForwardModel.boundary_crossings`` (no solver; an analytic psi_N): bit for bit the per-ray scan and bisection."""
    from bouquet.kinrecon import forward as FW
    ax = np.array([1.70, 0.03])

    def psi_n_at(gs, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        x, z = (p[:, 0] - ax[0]) / 0.62, (p[:, 1] - ax[1]) / 1.05
        v = x * x + z * z + 0.07 * x ** 3
        v[(p[:, 0] < 1.0) & (p[:, 1] > 0.9)] = np.nan               # a patch off the mesh
        return v, None

    monkeypatch.setattr(FW._diag, "psi_n_at", psi_n_at)
    fm = FW.ForwardModel.__new__(FW.ForwardModel)
    fm.gs = SimpleNamespace(o_point=tuple(ax))
    th = np.linspace(0, 2 * np.pi, 37, endpoint=False)
    pts = np.c_[ax[0] + 0.6 * np.cos(th), ax[1] + 1.0 * np.sin(th)]
    pts = np.vstack((pts, [ax[0] + 0.01, ax[1]], [ax[0] + 5.0, ax[1]], [0.95, 1.2]))   # inside, far outside, near the off-mesh patch

    def one_ray(points, span=(0.6, 1.4), n_scan=161, n_bisect=40):
        p = np.asarray(points, float)
        d = p - ax
        sg = np.linspace(span[0], span[1], int(n_scan))
        out = np.full_like(p, np.nan)
        for i in range(len(p)):
            pn = np.asarray(psi_n_at(None, ax + sg[:, None] * d[i])[0], float)
            k = np.where(pn >= 1.0)[0]
            if not len(k) or k[0] == 0:
                continue
            lo, hi = sg[k[0] - 1], sg[k[0]]
            for _ in range(int(n_bisect)):
                mid = 0.5 * (lo + hi)
                if float(np.asarray(psi_n_at(None, (ax + mid * d[i])[None, :])[0])[0]) >= 1.0:
                    hi = mid
                else:
                    lo = mid
            out[i] = ax + 0.5 * (lo + hi) * d[i]
        return out

    got, want = fm.boundary_crossings(pts), one_ray(pts)
    assert np.isfinite(want[:37, 0]).all() and np.isnan(want[-3:-1, 0]).all()
    assert np.array_equal(got, want, equal_nan=True)
    assert fm.boundary_crossings(np.zeros((0, 2))).shape == (0, 2)


def _fit_with_failed_trials(monkeypatch, fail_trials, **kw):
    fm = FakeFM()
    _install(monkeypatch, fm, fail_trials=fail_trials)
    kin, src = _kin()
    return FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                           FIT.FitSettings(free=SAW_FREE, **kw), params0=dict(log_strength=0.5))


def test_failed_trials_counted_over_the_fit_by_default(monkeypatch):
    # three failed trials in the first iteration, one in the second: the default budget (4 over the fit) ends the fit
    out = _fit_with_failed_trials(monkeypatch, (1, 2, 3, 5))
    assert out.status == "failed_trials"
    ft = out.record["failed_trials"]
    assert ft["counting"] == "over_fit" and ft["max_failed_trials"] == 4
    assert ft["per_iteration"][:2] == [3, 1] and sum(ft["per_iteration"]) == 4
