"""The NON-DEFAULT robustness options of the pass loop and of the fit, on mocked models (no OFT).

Pass loop (``current_model.consistent_solve``; the mock of ``test_pass_loop_safeguard.py``: the solve stores its
input, the next assembly returns ``G`` of it, so the loop is a fixed-point iteration with a known map):

* defaults: the option is off, the loop record carries no ``robustness`` entry and no pass is relaxed, and
  ``residual_cos`` (a record only) is the cosine of consecutive pass residuals;
* a synthetic PLATEAU (an oscillating mode with eigenvalue -0.93: the plain loop closes far too slowly for the
  budget): the EXPERIMENTAL under-relaxation closes it within the unchanged budget; the fixed point is the same, and
  the record carries the note that the option does not pass the factor-of-two test;
* the Anderson step under relaxation is the mixing-parameter form ``x + beta f - (dX + beta dF) gamma`` (checked
  against an independent evaluation on a map with unequal modes), and the plain relaxed step ``x + beta f``;
* the returned assembly of a relaxed pass names the input actually solved;
* an under-relaxed input that does not solve falls back to the plain input (as a mixed one does).

Fit (``fit.reconstruct`` on the linear mock of ``test_fit_unit.py``):

* ``noise_retries`` (0 or 1): a failed noise re-solve ends the fit by default (``jacobian_failed``); with the retry
  the noise is measured from the stated other starting state (the state of a consistent solve at the first column's
  parameter moved by MINUS that column's step) and the fit converges; the retry's method is recorded as a new draw;
  no retry is made when the first measurement succeeded; a retry whose starting state lies outside the parameter
  bounds is skipped without a solve;
* ``stall_diagnostics=True`` (record only): on a quadratic model whose forward-difference columns are biased and
  whose trials are all rejected (a stall), the fit is the default's stall in every value; the record says that the
  stopping tests on central-difference columns would hold there (test a) and carries the noise re-measured at the
  center; on a LINEAR model the central columns equal the forward ones.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM
from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet

N = 24
RHO = np.linspace(0.0, 1.0, N)
XSTAR = 1.0 + np.cos(2.0 * RHO)
MODE = np.sin(3.0 * RHO)


class LoopFM:
    def __init__(self, start=None, fail_moved=False):
        self.state = XSTAR + 0.3 * MODE if start is None else start.copy()
        self.inputs = []
        self.gs = None
        self.n_solves = 0
        self.fail_moved = fail_moved
        self.last_assembled = None

    def solve(self, j, p, ip, init="keep", label="", lock_case=True):
        x = np.asarray(j[1], float).copy()
        self.n_solves += 1
        if self.fail_moved and self.last_assembled is not None and not np.array_equal(x, self.last_assembled):
            return SimpleNamespace(converged=False, status="failed", record={"attempts": [{"its": 1}]}, wall_s=0.0,
                                   stats={})
        self.state = x
        self.inputs.append(x.copy())
        return SimpleNamespace(converged=True, status="converged", record={"attempts": [{"its": 1}]}, wall_s=0.0,
                               stats=dict(R_axis=1.7, Z_axis=0.0, l_i=1.0, q_0=1.0, Ip=1.0))

    def snapshot(self):
        return self.state.copy()

    def restore(self, snap):
        self.state = snap.copy()

    def update_probe_correction(self):
        return 0.0


def _install_loop(monkeypatch, fm, G):
    monkeypatch.setattr(CM, "evaluate_geometry", lambda gs, kin, cfg: dict(wall_s=0.0))

    def assemble(lag, kin, src, params, cfg, ip, z0=None, state=None):
        out = G(fm.state)
        fm.last_assembled = out.copy()
        return dict(orbit_z=None, wall_s=0.0, fractions={}, sawtooth={}, knots={"n": N}, jphi_cmp=out, jphi=out,
                    x=RHO, rho_knots=RHO, rho_cmp=RHO, p=np.zeros(N))

    monkeypatch.setattr(CM, "assemble", assemble)
    monkeypatch.setattr(CM, "q_check", lambda lag, asm, stats, cfg: dict(passed=True, max_rel=0.0))


def plateau(x):
    """An oscillating mode with eigenvalue -0.93: plain passes close, but far too slowly for the budget."""
    return XSTAR - 0.93 * (x - XSTAR)


LAMS = np.linspace(-0.9, 0.6, N)


def unequal(x):
    """A linear map with unequal eigenvalues (-0.9 .. 0.6, one per point): a one-difference Anderson step is not
    exact on it, so the plain and the mixing-parameter forms of the step differ."""
    return XSTAR + LAMS * (x - XSTAR)


def test_defaults_are_off_and_validated():
    cfg = CM.CurrentModelSettings(r0=1.0)
    assert cfg.pass_relaxation is None
    assert not hasattr(cfg, "pass_cycle_relaxation") and not hasattr(cfg, "pass_cycle_window")
    fs = FIT.FitSettings()
    assert fs.noise_retries == 0 and fs.stall_diagnostics is False
    assert FIT.NOISE_RETRY_STEPS == (-1.0,)
    for bad in (0.0, 1.0, -0.5, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            CM.CurrentModelSettings(r0=1.0, pass_relaxation=bad)
    for bad in (-1, 1.5, True, 2):
        with pytest.raises(ValueError):
            FIT.FitSettings(noise_retries=bad)
    assert FIT.FitSettings(noise_retries=1).noise_retries == 1
    with pytest.raises(ValueError):
        FIT.FitSettings(stall_diagnostics="central")
    assert "factor-of-two" in CM.PASS_RELAXATION_NOTE and "NOT" in CM.PASS_RELAXATION_NOTE


@pytest.mark.parametrize("accel", ["anderson", "none"])
def test_default_loop_records_the_residual_cosine_and_no_option(monkeypatch, accel):
    fm = LoopFM()
    _install_loop(monkeypatch, fm, plateau)
    res = CM.consistent_solve(fm, None, None, {}, CM.CurrentModelSettings(r0=1.0, pass_acceleration=accel), ip=1.0)
    assert "robustness" not in res.record
    recs = res.record["passes"]
    assert not any(r.get("relaxation") or r.get("relaxed_plain_step") for r in recs)
    xs = fm.inputs
    resid =[plateau(xs[k - 1]) - xs[k - 1] for k in range(1, len(xs))]   # residual seen at pass k
    for k in range(2, len(xs)):
        a, b = resid[k - 1], resid[k - 2]
        assert recs[k]["residual_cos"] == pytest.approx(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
                                                        rel=1e-9, abs=1e-12)


def test_relaxed_steps_have_the_stated_form(monkeypatch):
    """Pass 1 is the plain relaxed step x + beta f; pass 2 (two history entries, no restart possible yet) is the
    mixing-parameter Anderson step x + beta f - (dX + beta dF) gamma, gamma = lstsq(dF, f)."""
    beta = 0.5
    fm = LoopFM()
    _install_loop(monkeypatch, fm, unequal)
    cfg = CM.CurrentModelSettings(r0=1.0, pass_relaxation=beta, max_passes=4)
    res = CM.consistent_solve(fm, None, None, {}, cfg, ip=1.0)
    x0, x1, x2 = fm.inputs[0], fm.inputs[1], fm.inputs[2]
    f0, f1 = unequal(x0) - x0, unequal(x1) - x1
    np.testing.assert_allclose(x1, x0 + beta * f0, rtol=0, atol=1e-13)
    dF, dX = (f1 - f0)[:, None], (x1 - x0)[:, None]
    gam = np.linalg.lstsq(dF, f1, rcond=None)[0]
    want = x1 + beta * f1 - (dX + beta * dF) @ gam
    np.testing.assert_allclose(x2, want, rtol=0, atol=1e-12)
    # the undamped form differs on this map by far more than the tolerance (so the check above has teeth)
    plain = x1 + f1 - (dX + dF) @ gam
    assert np.max(np.abs(plain - want)) > 1e-6
    recs = res.record["passes"]
    assert recs[1].get("relaxed_plain_step") and recs[2].get("anderson_mixed") and recs[2].get("relaxation") == beta
    assert res.record["robustness"] == dict(pass_relaxation=beta, pass_relaxation_note=CM.PASS_RELAXATION_NOTE)


def test_returned_assembly_names_the_relaxed_input(monkeypatch):
    """A loop that closes after relaxed passes returns the assembly with the input actually solved."""
    fm = LoopFM()
    _install_loop(monkeypatch, fm, plateau)
    cfg = CM.CurrentModelSettings(r0=1.0, pass_acceleration="none", pass_relaxation=0.5)
    res = CM.consistent_solve(fm, None, None, {}, cfg, ip=1.0)
    assert res.converged
    used = res.assembly
    assert "jphi_cmp_used" in used and np.array_equal(used["jphi"], fm.inputs[-1])
    assert not np.array_equal(used["jphi"], used["jphi_assembled"])


@pytest.mark.parametrize("accel", ["anderson", "none"])
def test_plateau_closes_with_under_relaxation(monkeypatch, accel):
    out = {}
    for opt in (None, 0.5):
        fm = LoopFM()
        _install_loop(monkeypatch, fm, plateau)
        cfg = CM.CurrentModelSettings(r0=1.0, pass_acceleration=accel, pass_relaxation=opt)
        out[opt] = (CM.consistent_solve(fm, None, None, {}, cfg, ip=1.0), fm, cfg)
    res0 = out[None][0]
    if accel == "none":
        # plain passes shrink the residual by only 0.93 per pass: not within the budget of 15
        assert res0.status == "max_passes"
        dj = [r["dj"] for r in res0.record["passes"] if r.get("dj") is not None]
        assert dj[-1] > 10 * out[None][2].pass_tol
    res, fm, cfg = out[0.5]
    assert res.converged and res.status == "consistent" and res.passes <= cfg.max_passes
    assert res.record["passes"][-1]["dj"] < cfg.pass_tol
    np.testing.assert_allclose(fm.state, XSTAR, rtol=0, atol=1e-4)
    assert res.record["robustness"]["pass_relaxation"] == 0.5
    assert all(r.get("relaxation") == 0.5 for r in res.record["passes"][1:-1])


def test_relaxed_input_that_fails_falls_back_to_the_plain_input(monkeypatch):
    fm = LoopFM(fail_moved=True)
    _install_loop(monkeypatch, fm, plateau)
    cfg = CM.CurrentModelSettings(r0=1.0, pass_acceleration="none", pass_relaxation=0.5, max_passes=4)
    res = CM.consistent_solve(fm, None, None, {}, cfg, ip=1.0)
    recs = res.record["passes"]
    assert recs[1]["relaxation_fallback"] == "failed" and not recs[1].get("relaxed_plain_step")
    # every solved input is the assembled (plain) profile
    assert all(np.array_equal(x, plateau(xp)) for xp, x in zip(fm.inputs[:-1], fm.inputs[1:]))


# ---------------------------------------------------------------------------------------------- fit options
A = np.array([2.0, -1.0, 0.5, 3.0])
B = np.array([0.3, -0.2, 0.1, 0.05])
TRUE = 2.0


class FitFM:
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


def _install_fit(monkeypatch, fm, resid, fail=None, reject_trials=False):
    """``resid(theta)``: the data residuals; ``fail(label)``: whether that consistent solve fails;
    ``reject_trials``: every LM trial returns a chi^2 far above the center's (a stall)."""
    labels = []
    calls = []

    def consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        labels.append(label)
        calls.append(dict(label=label, theta=params["log_strength"],
                          start=(None if start is None else (start.get("theta") or (None,))[0])))
        fm_.n_solves += 3
        ok = not (fail is not None and fail(label))
        if ok:
            fm_.theta_state = (params["log_strength"], ":trial" in label and reject_trials)
        solve = SimpleNamespace(stats={"q_0": 1.0}, status="converged" if ok else "failed_verification",
                                record={"policy": {"verification": {"verdict": "PASS" if ok else "FAIL"}}})
        return CM.ConsistentResult(ok, "consistent" if ok else "solve_failed", 3,
                                   {"orbit_z": None, "fractions": {}} if ok else None, {"lag": True}, solve,
                                   {"passes": [], "reason": None if ok else "injected"}, 0.01)

    def residual_vector(fm_, ms, groups, settings, roles):
        th, penal = fm_.theta_state
        r = resid(th) + (10.0 if penal else 0.0)
        return r, [("flux_loop", s) for s in "abcd"], SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    monkeypatch.setattr(FIT, "score", lambda *a, **k: SimpleNamespace(model={}, excluded={}, notes=[]))
    monkeypatch.setattr(FIT, "chi2_summary", lambda tab: {"total": (0.0, 0, 0.0)})
    monkeypatch.setattr(FIT, "constraints_table", lambda *a, **k: {})
    _CALLS[:] = [calls]           # every consistent solve of the last installed mock: label, parameter, start state
    return labels


_CALLS = []


def _calls():
    return _CALLS[0]


def _fs(**kw):
    """Fit settings for these robustness tests: the moving anchor target (``FitSettings.anchor_follow``, on by default)
    is named off, since these tests are not about it."""
    return FIT.FitSettings(anchor_follow=False, **kw)


def _fit(fm, fs, theta0):
    kin, src = SimpleNamespace(source="toy"), SimpleNamespace(source="toy", convention={})
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    return FIT.reconstruct(fm, kin, src, cfg, fs, params0=dict(log_strength=theta0, f=0.37))


def test_noise_retry_from_the_stated_state(monkeypatch):
    lin = lambda th: A * (th - TRUE) + B  # noqa: E731
    first = lambda lab: lab.endswith(":noise")  # noqa: E731 -- only the first noise re-solve fails
    fm = FitFM()
    _install_fit(monkeypatch, fm, lin, fail=first)
    out = _fit(fm, _fs(free=("log_strength", "a_corr")), 0.5)
    assert out.status == "jacobian_failed"
    assert out.record["noise"]["status"] == "solve_failed" and "retries" not in out.record["noise"]
    assert "robustness" not in out.record
    fm = FitFM()
    labels = _install_fit(monkeypatch, fm, lin, fail=first)
    out = _fit(fm, _fs(free=("log_strength", "a_corr"), noise_retries=1), 0.5)
    assert out.status in ("converged", "converged_within_noise"), out.status
    rt = out.record["noise"]["retries"]
    assert rt["first_status"] == "solve_failed" and rt["setting"] == 1
    a0 = rt["attempts"][0]
    assert a0["status"] == "ok" and a0["step_multiple"] == -1.0 and a0["parameter"] == "log_strength"
    h = FIT.PARAMETERS["log_strength"]["step0"]
    assert a0["step"] == pytest.approx(-h)
    assert out.record["noise"]["status"] == "ok"
    assert any(lab.endswith(":noise_state_retry1") for lab in labels)
    assert any(lab.endswith(":noise_retry1") for lab in labels)
    assert out.record["robustness"]["noise_retried"] is True
    # the starting state is that of a consistent solve at the center's parameter MINUS the first column's step,
    # and the retry re-solves the center's parameters FROM that state (not from the center)
    calls = _calls()
    c0 = next(c for c in calls if c["label"].endswith(":center0"))
    st = next(c for c in calls if c["label"].endswith(":noise_state_retry1"))
    re = next(c for c in calls if c["label"].endswith(":noise_retry1"))
    assert st["theta"] == pytest.approx(c0["theta"] - h, abs=1e-15)
    assert re["theta"] == pytest.approx(c0["theta"], abs=1e-15)
    assert re["start"] == pytest.approx(st["theta"], abs=1e-15) and re["start"] != pytest.approx(c0["theta"])
    # the record says the noise came from the retry, as a new draw
    assert rt["measured_by_retry"] is True and rt["first_method"].startswith("consistent re-solve")
    assert out.record["noise"]["method"].startswith("RETRY") and "NEW noise draw" in out.record["noise"]["method"]


def test_no_retry_when_the_first_measurement_succeeds(monkeypatch):
    lin = lambda th: A * (th - TRUE) + B  # noqa: E731
    out = {}
    for n in (0, 1):
        fm = FitFM()
        labels = _install_fit(monkeypatch, fm, lin)
        out[n] = (_fit(fm, _fs(free=("log_strength", "a_corr"), noise_retries=n), 0.5), labels)
    (o0, l0), (o1, l1) = out[0], out[1]
    assert not any("retry" in lab for lab in l1) and l1 == l0
    assert "retries" not in o1.record["noise"] and o1.record["noise"] == o0.record["noise"]
    assert np.array_equal(o0.theta, o1.theta) and o1.record["robustness"]["noise_retried"] is False


def test_retry_outside_the_bounds_is_skipped(monkeypatch):
    lin = lambda th: A * (th - TRUE) + B  # noqa: E731
    first = lambda lab: lab.endswith(":noise")  # noqa: E731
    lo = FIT.PARAMETERS["log_strength"]["bounds"][0]
    h = FIT.PARAMETERS["log_strength"]["step0"]
    fm = FitFM()
    labels = _install_fit(monkeypatch, fm, lin, fail=first)
    out = _fit(fm, _fs(free=("log_strength", "a_corr"), noise_retries=1), lo + 0.5 * h)
    assert out.status == "jacobian_failed"
    att = out.record["noise"]["retries"]["attempts"]
    assert len(att) == 1 and att[0]["status"].startswith("starting state outside the parameter bounds")
    assert not any("noise_state_retry" in lab or "noise_retry" in lab for lab in labels)
    assert out.record["noise"]["retries"]["measured_by_retry"] is False


Q = np.array([4.0, 3.0, -2.0, 5.0])
B2 = np.array([3.0, -2.0, 1.0, 2.0])


def _quadratic(th):
    d = th - TRUE
    return A * d + B2 + Q * d * d


def _map_point(prior_mu, prior_sigma):
    """The minimizer of sum r^2 + ((theta - mu)/sigma)^2 for the quadratic model (Newton on the exact gradient)."""
    th = TRUE
    for _ in range(200):
        d = th - TRUE
        r = _quadratic(th)
        J = A + 2.0 * Q * d
        g = J @ r + (th - prior_mu) / prior_sigma ** 2
        H = J @ J + r @ (2.0 * Q) + 1.0 / prior_sigma ** 2
        th -= g / H
    return th


def test_stall_diagnostics_record_and_never_steer(monkeypatch):
    """At a stall on biased forward columns, the diagnostic records that the central-column test (a) would hold, and
    the fit is still the default's stall: same status, same parameters, same chi^2."""
    mu, sig = FIT.PARAMETERS["log_strength"]["prior"]
    th0 = _map_point(mu, sig)
    out, labels = {}, {}
    for opt in (False, True):
        fm = FitFM()
        labels[opt] = _install_fit(monkeypatch, fm, _quadratic, reject_trials=True)
        out[opt] = _fit(fm, _fs(free=("log_strength",), stall_diagnostics=opt), th0)
    d, c = out[False], out[True]
    assert d.status == c.status == "stalled_no_significant_descent"
    assert np.array_equal(d.theta, c.theta) and d.chi2["objective"] == c.chi2["objective"]
    st = d.record["iterations"][0]["stopping_tests"]
    assert st["converged"] is False and st["gn_step_sigma"][0] > FIT.FitSettings().step_tol_sigma
    assert "stall_diagnostics" not in d.record["iterations"][0] and "robustness" not in d.record
    sd = c.record["iterations"][0]["stall_diagnostics"]
    # the forward tests are kept as they were; the central-column test is only recorded
    assert c.record["iterations"][0]["stopping_tests"] == st
    assert sd["stopping_tests_central"]["test"] == "a"
    assert sd["stopping_tests_central"]["gn_step_sigma"][0] < FIT.FitSettings().step_tol_sigma
    assert sd["central_columns"][0]["central"] is True and sd["noise_at_center"]["status"] == "ok"
    assert "stopping_tests_with_noise_at_center" in sd
    assert c.record["robustness"]["stall_diagnostics_made"][0]["central_test"] == "a"
    # the extra solves are the diagnostic's own: one backward column and one noise re-solve, labeled as such
    extra = [lab for lab in labels[True] if ":stall_diag" in lab]
    assert len(extra) == 2 and len(labels[True]) == len(labels[False]) + 2


def test_stall_diagnostics_on_a_linear_model(monkeypatch):
    lin = lambda th: A * (th - TRUE) + B  # noqa: E731
    fm = FitFM()
    _install_fit(monkeypatch, fm, lin, reject_trials=True)
    c = _fit(fm, _fs(free=("log_strength",), stall_diagnostics=True), 0.5)
    assert c.status == "stalled_no_significant_descent"
    sd = c.record["iterations"][0]["stall_diagnostics"]
    # forward differences of a linear model are exact: the central columns equal them
    assert sd["central_columns"][0]["rel_change"] == pytest.approx(0.0, abs=1e-12)
    assert sd["stopping_tests_central"]["converged"] is False
