"""Three NON-DEFAULT options, without OFT: the starting-solve retry ladder (``FitSettings.start_retry``), the solver's
under-relaxation factor (``ForwardSettings.solver_urf``) and the flagged psi_N placement of kinetic inputs whose
psi_N -> rho_tor,norm table does not increase (``assemble_kinetic_inputs(..., rho_map_fallback="psi_n")``).

For each: the default is unchanged (off, and absent from the default records), the option is checked, and what it
did is recorded.  The fit tests use a mocked forward model (the consistent solve replaced by a stand-in that fails
or succeeds by the solve's label), so they test the ladder's logic, not a solver.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM
from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon import forward as FW
from bouquet.kinrecon import kinetics as KI
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet

SAW_FREE = ("log_strength", "a_corr")
A = np.array([[2.0], [-1.0], [0.5], [3.0]])
B = np.array([0.3, -0.2, 0.1, 0.05])
TRUE = 2.0
X0 = np.linspace(0.0, 1.0, 11)
J0 = 1.0e6 * (1.0 - X0 ** 2)
P_KIN = 4.0e4 * (1.0 - X0 ** 2) ** 2           # the first attempt's pressure
P_ANCHOR = 2.5e4 * (1.0 - X0 ** 2)             # the anchor's pressure (another profile)


class FakeFM:
    def __init__(self):
        self.theta_state = None
        self.n_solves = 0
        self.probe_correction = np.zeros(0)
        self.gs = None
        self.ms = MeasurementSet("toy", 0.0, 0.0, {"flux_loop": ConstraintGroup(
            "flux_loop", ["a", "b", "c", "d"], np.zeros(4), np.ones(4), np.ones(4, bool))})
        self.rows_record = {"coils": {"per_coil": {}}}
        self.restores = 0

    def snapshot(self):
        return dict(theta=self.theta_state)

    def restore(self, snap):
        self.restores += 1
        self.theta_state = snap["theta"]

    def settings_record(self):
        return {"settings": {"mock": True}, "n_solves": self.n_solves}

    def boundary_distance(self):
        return {"rms_m": 0.0}


def _install(monkeypatch, fail_labels=()):
    """The consistent solve fails when its label ends with one of ``fail_labels``; every first-solve input is kept."""
    seen = []

    def consistent(fm_, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
        fm_.n_solves += 3
        ok = not any(label.endswith(s) for s in fail_labels)
        if ":center0" in label:
            seen.append(dict(label=label, initial_profiles=initial_profiles, params=dict(params)))
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
    return seen


def _fit(monkeypatch, fail_labels=(), start_retry=False, inputs=None, initial=(X0, J0, P_KIN)):
    fm = FakeFM()
    seen = _install(monkeypatch, fail_labels)
    kin, src = SimpleNamespace(source="toy"), SimpleNamespace(source="toy", convention={})
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    fs = FIT.FitSettings(anchor_follow=False, free=SAW_FREE, start_retry=start_retry)
    out = FIT.reconstruct(fm, kin, src, cfg, fs, params0=dict(log_strength=0.5, f=0.37), initial_profiles=initial,
                          start_retry_inputs=inputs, steady_state_check=False)
    return out, seen, fm


# ------------------------------------------------------------------ the retry ladder: settings
def test_start_retry_is_on_by_default_and_recorded():
    s = FIT.FitSettings()
    assert s.start_retry is True
    assert s.as_dict()["start_retry"] is True
    assert FIT.FitSettings(start_retry=False).as_dict()["start_retry"] is False
    assert FIT.FitSettings(start_retry=True).as_dict()["start_retry"] is True
    with pytest.raises(ValueError, match="start_retry"):
        FIT.FitSettings(start_retry=1)


def test_start_retry_inputs_are_checked():
    assert FIT._check_start_retry_inputs(None) is None
    with pytest.raises(ValueError, match="unknown"):
        FIT._check_start_retry_inputs(dict(anchor=1))
    with pytest.raises(ValueError, match="anchor_pressure"):
        FIT._check_start_retry_inputs(dict(anchor_pressure=([0.0, 0.5, 0.4], [1.0, 2.0, 3.0])))
    with pytest.raises(ValueError, match="params"):
        FIT._check_start_retry_inputs(dict(neighbor=dict(label="n")))
    with pytest.raises(ValueError, match="unknown"):
        FIT._check_start_retry_inputs(dict(neighbor=dict(params={}, state=1)))
    with pytest.raises(ValueError, match="initial_profiles"):
        FIT._check_start_retry_inputs(dict(neighbor=dict(params={}, initial_profiles=(X0, J0))))


# ------------------------------------------------------------------ the retry ladder: behavior
def test_off_a_failed_start_fails_as_before_with_no_retry(monkeypatch):
    out, seen, _ = _fit(monkeypatch, fail_labels=(":center0",),
                        inputs=dict(anchor_pressure=(X0, P_ANCHOR), neighbor=dict(params=dict(log_strength=1.0))))
    assert out.status == "failed_initial_consistency"
    assert "start_retry" not in out.record
    assert [s["label"] for s in seen] == ["fit:center0"]


def test_on_with_a_good_start_is_the_default_fit(monkeypatch):
    off, _, _ = _fit(monkeypatch)
    on, seen, _ = _fit(monkeypatch, start_retry=True, inputs=dict(anchor_pressure=(X0, P_ANCHOR)))
    assert "start_retry" not in on.record and len(seen) == 1
    assert on.status == off.status and np.array_equal(on.theta, off.theta)
    assert np.array_equal(on.covariance, off.covariance)


def test_anchor_pressure_rung_keeps_the_current_and_replaces_only_the_pressure(monkeypatch):
    out, seen, fm = _fit(monkeypatch, fail_labels=(":center0",), start_retry=True,
                         inputs=dict(anchor_pressure=(np.linspace(0, 1, 51), 2.5e4 * (1 - np.linspace(0, 1, 51) ** 2)),
                                     neighbor=dict(params=dict(log_strength=1.0))))
    assert out.status == "converged"
    assert [s["label"] for s in seen] == ["fit:center0", "fit:center0:retry_anchor_pressure"]
    x, j, p = seen[1]["initial_profiles"]
    assert np.array_equal(x, X0) and np.array_equal(j, J0)
    np.testing.assert_allclose(p, P_ANCHOR, rtol=1e-12)
    assert seen[1]["params"]["log_strength"] == seen[0]["params"]["log_strength"]
    sr = out.record["start_retry"]
    assert sr["recovered_by"] == "anchor_pressure" and sr["first_attempt"]["status"] == "solve_failed"
    assert [r["rung"] for r in sr["rungs"]] == ["anchor_pressure"] and sr["rungs"][0]["tried"] is True
    assert sr["rungs"][0]["p_axis_retry"] == pytest.approx(P_ANCHOR[0])
    assert "unchanged" in sr["criterion"]
    assert out.record["settings"]["start_retry"] is True


def test_neighbor_rung_takes_the_neighbor_parameters_and_profiles(monkeypatch):
    jn, pn = 0.9 * J0, 0.8 * P_KIN
    out, seen, _ = _fit(monkeypatch, fail_labels=(":center0", ":retry_anchor_pressure"), start_retry=True,
                        inputs=dict(anchor_pressure=(X0, P_ANCHOR),
                                    neighbor=dict(params=dict(log_strength=1.7, s_bs=2.0), label="slice t-1",
                                                  initial_profiles=(X0, jn, pn))))
    assert out.status == "converged"
    assert [s["label"] for s in seen][-1] == "fit:center0:retry_neighbor"
    assert seen[-1]["params"]["log_strength"] == pytest.approx(1.7)
    assert np.array_equal(seen[-1]["initial_profiles"][1], jn) and np.array_equal(seen[-1]["initial_profiles"][2], pn)
    sr = out.record["start_retry"]
    assert sr["recovered_by"] == "neighbor"
    nb = sr["rungs"][1]
    assert nb["tried"] and nb["neighbor"] == "slice t-1" and nb["params_taken"] == {"log_strength": 1.7}
    assert sr["rungs"][0]["tried"] and sr["rungs"][0]["status"] == "solve_failed"


def test_rungs_are_skipped_with_reasons_and_the_fit_still_fails(monkeypatch):
    # the first attempt already started from the anchor's pressure; no neighbor supplied
    out, seen, _ = _fit(monkeypatch, fail_labels=(":center0",), start_retry=True,
                        inputs=dict(anchor_pressure=(X0, P_KIN)))
    assert out.status == "failed_initial_consistency" and len(seen) == 1
    sr = out.record["start_retry"]
    assert sr["recovered_by"] is None
    assert [(r["rung"], r["tried"]) for r in sr["rungs"]] == [("anchor_pressure", False), ("neighbor", False)]
    assert "already" in sr["rungs"][0]["reason"] and "no neighboring" in sr["rungs"][1]["reason"]
    # no inputs at all
    out2, _, _ = _fit(monkeypatch, fail_labels=(":center0",), start_retry=True)
    assert "no anchor pressure" in out2.record["start_retry"]["rungs"][0]["reason"]
    # neighbor rung clips its parameters to the bounds and fails when its solve fails
    hi = FIT.PARAMETERS["log_strength"]["bounds"][1]
    out3, seen3, _ = _fit(monkeypatch, fail_labels=(":center0", ":retry_neighbor"), start_retry=True,
                          inputs=dict(neighbor=dict(params=dict(log_strength=hi + 10.0))))
    assert out3.status == "failed_initial_consistency"
    assert seen3[-1]["params"]["log_strength"] == pytest.approx(hi)
    assert out3.record["start_retry"]["rungs"][1]["status"] == "solve_failed"


# ------------------------------------------------------------------ solver under-relaxation factor
def test_solver_urf_default_is_none_and_absent_from_the_default_record():
    s = FW.ForwardSettings()
    assert s.solver_urf is None and "solver_urf" not in s.as_dict()
    assert FW.ForwardSettings(solver_urf=0.5).as_dict()["solver_urf"] == 0.5


@pytest.mark.parametrize("value, expected", [(None, None), (0.0, 0.0), (0.5, 0.5), (0.9, 0.9), (np.float64(0.3), 0.3)])
def test_check_solver_urf_accepts(value, expected):
    assert FW.check_solver_urf(value) == expected


@pytest.mark.parametrize("value", [True, "0.5", 1.0, -0.1, float("nan"), 2, [0.5]])
def test_check_solver_urf_refuses(value):
    with pytest.raises(ValueError, match="solver_urf"):
        FW.check_solver_urf(value)


class _FakeGS:
    def __init__(self):
        self.settings = SimpleNamespace(nl_tol=1e-8, urf=0.2, maxits=40)
        self.sent = []

    def update_settings(self):
        self.sent.append(dict(vars(self.settings)))


def _bare_fm(maxits=None, urf=None):
    fm = object.__new__(FW.ForwardModel)
    fm.gs = _FakeGS()
    fm.defaults = fm._read_settings()
    fm.iteration_limit = fm._configure_iteration_limit(maxits)
    fm.relaxation = fm._configure_relaxation(urf)
    return fm


def test_configure_relaxation_sets_only_urf_and_records_it():
    fm = _bare_fm()
    assert fm.relaxation["setting"] is None and fm.gs.sent == [] and fm.expected_settings == fm.defaults
    assert fm._relaxation_record() == {}
    fm = _bare_fm(urf=0.6)
    assert fm.gs.settings.urf == 0.6 and fm.gs.settings.nl_tol == 1e-8 and fm.gs.settings.maxits == 40
    assert fm.expected_settings == dict(nl_tol=1e-8, urf=0.6, maxits=40)
    assert fm.relaxation["setting"] == 0.6 and fm.relaxation["oft_default"] == 0.2
    assert fm._relaxation_record()["relaxation"]["now"] == 0.6
    fm = _bare_fm(maxits=100, urf=0.5)
    assert fm.expected_settings == dict(nl_tol=1e-8, urf=0.5, maxits=100)


# ------------------------------------------------------------------ flagged psi_N placement of the kinetic inputs
TI_COV = dict(points=dict(value=[2.0e3, 1.0e3], psi_N=[0.1, 0.6]), core_psi_n_max=None)
XK = np.linspace(0.0, 1.0, 51)
NE = 5e19 * (1.1 - 0.6 * XK ** 2)
TE = 3e3 * (1.05 - XK ** 2) + 50
TI = 2.5e3 * (1.05 - XK ** 2) + 60
TP = np.linspace(0.0, 1.0, 101)
TR_PLATEAU = np.sqrt(TP)
TR_PLATEAU[95:100] = TR_PLATEAU[95]          # rho constant over psi_N 0.95 .. 0.99, then the endpoint (1, 1)


def _kin(tab, **kw):
    return KI.assemble_kinetic_inputs(XK, NE, TE, TI, zeff_candidates={"z": ("zeff", np.full_like(XK, 1.5))},
                                      zeff_source="z", z_imp=6.0, coordinate="psi_n", psi_n_to_rho=tab,
                                      mapping_source="the profile fit's equilibrium", n_fast="none",
                                      ti_coverage=TI_COV, **kw)


def test_rho_plateau_is_placed_by_own_psi_n_by_default_and_refused_with_none():
    k = _kin((TP, TR_PLATEAU))
    assert k.coordinate == "psi_n" and k.record["mapping_fallback"]["setting"] == "psi_n"
    with pytest.raises(ValueError, match="increase strictly"):
        _kin((TP, TR_PLATEAU), rho_map_fallback=None)


def test_rho_map_fallback_places_by_own_psi_n_and_records_the_points():
    k = _kin((TP, TR_PLATEAU), rho_map_fallback="psi_n")
    assert k.coordinate == "psi_n" and np.array_equal(k.x, XK)
    assert np.array_equal(k.ne, NE) and np.array_equal(k.te, TE)          # the data are not changed
    assert "FLAGGED FALLBACK" in k.record["mapping"]
    mf = k.record["mapping_fallback"]
    assert mf["applied"] and mf["flagged"] and mf["setting"] == "psi_n"
    assert mf["non_increasing_intervals"]["index"] == [95, 96, 97, 98]
    assert mf["first_non_increasing_psi_n"] == pytest.approx(0.95)
    pl = mf["placement"]
    assert pl["by_reconstruction_psi_n"] == 51 and pl["by_table_rho"] == 0
    assert pl["n_would_be_by_table_rho"] == int(np.sum(XK < 0.95))
    assert pl["n_would_be_by_table_rho"] + pl["n_in_or_beyond_non_increasing"] == 51
    np.testing.assert_allclose(k.profiles_at(psi_n=0.36)["ne"], np.interp(0.36, XK, NE), rtol=2e-3)


def test_rho_map_fallback_leaves_an_increasing_table_alone():
    tab = (TP, np.sqrt(TP))
    a, b = _kin(tab), _kin(tab, rho_map_fallback="psi_n")
    assert a.coordinate == b.coordinate == "rho_tor_norm" and np.array_equal(a.x, b.x)
    assert "mapping_fallback" not in a.record and "mapping_fallback" not in b.record


def test_rho_map_fallback_is_checked():
    with pytest.raises(ValueError, match="rho_map_fallback must be one of"):
        _kin((TP, TR_PLATEAU), rho_map_fallback="rho")
    tp_bad = TP.copy()
    tp_bad[50] = tp_bad[49]
    with pytest.raises(ValueError, match="increase strictly"):           # psi_N not increasing: refused anyway
        _kin((tp_bad, np.sqrt(TP)), rho_map_fallback="psi_n")
    # without a psi_N table the fallback has nothing to act on: no effect, no record
    k = KI.assemble_kinetic_inputs(XK, NE, TE, TI, zeff_candidates={"z": ("zeff", np.full_like(XK, 1.5))},
                                   zeff_source="z", z_imp=6.0, coordinate="rho_tor_norm", n_fast="none",
                                   ti_coverage=TI_COV, rho_map_fallback="psi_n")
    assert "mapping_fallback" not in k.record


# ------------------------------------------------------------------ the ladder's record survives the moving target
class _AnchorFM:
    def __init__(self):
        self.n_solves, self.target, self.state, self.probe_correction = 0, 0, 0, np.zeros(0)

    def anchor_target(self):
        return self.target

    def set_anchor_target(self, t):
        self.target = t

    def move_anchor_target(self, fraction):
        self.target += 1
        return dict(moved=True)

    def snapshot(self):
        return dict(s=self.state)

    def restore(self, snap):
        self.state = snap["s"]


def test_start_retry_record_is_carried_to_a_returned_refit(monkeypatch):
    calls = []

    def fake_checked(fm, kin, src, cfg, settings=None, label="fit", start_retry_inputs=None, **kw):
        calls.append(label)
        rec = dict(center=None)
        if len(calls) == 1:
            rec["start_retry"] = dict(recovered_by="anchor_pressure", rungs=[])
        chi = 10.0 - len(calls) if len(calls) <= 2 else 99.0       # move 1 kept, move 2 not
        return SimpleNamespace(status="converged", chi2=dict(data=chi), params={}, record=rec)

    monkeypatch.setattr(FIT, "_reconstruct_checked", fake_checked)
    st = FIT.FitSettings(start_retry=True, anchor_follow_defer_final_jacobian=False)
    base = FIT.replace(st, anchor_follow=False, anchor_follow_defer_final_jacobian=False)
    fr = FIT._follow_anchor(_AnchorFM(), None, None, None, base, st, None, None, None, "fit", None, False,
                            start_retry_inputs=dict(anchor_pressure=(X0, P_ANCHOR)))
    assert calls == ["fit", "fit:anchor1", "fit:anchor2"]
    assert fr.record["anchor_follow"]["n_moves_kept"] == 1
    assert fr.record["start_retry"]["recovered_by"] == "anchor_pressure"
    assert "first fit" in fr.record["start_retry"]["carried_from"]
