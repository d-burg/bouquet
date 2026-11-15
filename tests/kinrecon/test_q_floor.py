"""The time-averaged q floor (``current.sawtooth.floor_q``), the default core treatment of single-slice fits and series.

Synthetic geometry as in ``test_coordinates_kinetics_series.py`` and the mocked series of ``test_series_driver.py``
(no OFT).  The model's steady state at the fixture's Ip keeps q > 1.6 everywhere; at 2.2 times that Ip it has q < 1
inside rho ~0.4 with q ~2.7 at the domain edge, so the floor q = 1 acts there.

Checked: the operator (Ip conserved, q >= floor after, the identity where q >= floor, refusal when q at the domain
edge is below the floor, continuity of q at the floor radius, idempotence); the default treatment of a single slice
(the floor on the MODEL state, the corrections added after it, bit-identical to the plain steady state where the floor
does not act); a series under the default floors every step with the same operator and setting and records it per
slice; the opt-in sawtooth treatments stay reachable and are refused where they are not selected.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
import test_correction_local_return as LR  # noqa: E402
import test_series_driver as SD  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import CurrentState  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

IP_LOW_Q = 2.2 * GEO.IP           # q < 1 in the core of the model's steady state on the fixture geometry
ROUND_OFF = 1e-12                 # relative: q rebuilt from psi differences on the cells (round trip q -> psi -> q)


def _steady(ip, n=100):
    lag = GEO._lag()
    sol = CM.build_solver(lag, lag["jbs"], ip)
    return sol, sol.steady_state(CurrentState.uniform_faces(n))


def test_floor_q_conserves_ip_and_holds_q_at_the_floor():
    sol, st = _steady(IP_LOW_Q)
    p0 = sol.profiles(st)
    assert p0["q"][1:-1].min() < 1.0 < p0["q"][-1]
    st1, rec = SAW.floor_q(st, sol, 1.0)
    p1 = sol.profiles(st1)
    assert rec["acted"] and rec["q_min_before"] < 1.0 and 0.0 < rec["rho_floor"] < 1.0
    # Ip and the boundary gauge unchanged; the enclosed current outside the floored region unchanged to the bit
    assert p1["ip"] == p0["ip"] and st1.psi_boundary == st.psi_boundary and st1.psi[-1] == st.psi[-1]
    outside = p0["q"] >= 1.0
    outside[0] = False
    np.testing.assert_array_equal(p1["I_enclosed"][outside], p0["I_enclosed"][outside])
    # q >= floor on every face after (to the round trip), and q = max(q_before, floor) on the interior faces
    assert p1["q"].min() >= 1.0 - ROUND_OFF
    np.testing.assert_allclose(p1["q"][1:-1], np.maximum(p0["q"][1:-1], 1.0), rtol=ROUND_OFF)
    # the current inside the floored region is flattened (less on the axis), its total returns at the outer face
    assert p1["jdotb"][0] < p0["jdotb"][0] and rec["d_enclosed_current_max_over_ip"] > 0.0
    assert st1.meta["q_floor"]["rho_floor"] == rec["rho_floor"]


def test_floor_q_is_continuous_at_the_floor_radius_and_idempotent():
    sol, st = _steady(IP_LOW_Q)
    q0 = sol.profiles(st)["q"]
    st1, rec = SAW.floor_q(st, sol, 1.0)
    q1 = sol.profiles(st1)["q"]
    rf = st.rho_faces
    k = int(np.searchsorted(rf, rec["rho_floor"])) - 1        # last face inside the floor radius
    assert rf[k] <= rec["rho_floor"] < rf[k + 1] and q0[k] < 1.0 <= q0[k + 1]
    # continuous: the step of q across the floor radius is no larger than the state's own step there
    assert abs(q1[k + 1] - q1[k]) <= abs(q0[k + 1] - q0[k]) * (1 + ROUND_OFF)
    assert np.max(np.abs(np.diff(q1[1:-1]))) <= np.max(np.abs(np.diff(q0[1:-1]))) * (1 + ROUND_OFF)
    # the slope breaks there: flat inside, rising outside
    assert np.allclose(q1[1:k + 1], 1.0, rtol=ROUND_OFF) and q1[k + 2] > q1[k + 1]
    # a floored state is left as it is (the identity, the same object)
    st2, rec2 = SAW.floor_q(st1, sol, 1.0)
    assert st2 is st1 and rec2["acted"] is False


def test_floor_q_is_the_identity_where_q_is_above_the_floor():
    sol, st = _steady(GEO.IP)
    assert sol.profiles(st)["q"].min() > 1.0
    st1, rec = SAW.floor_q(st, sol, 1.0)
    assert st1 is st and rec["acted"] is False and rec["n_faces_floored"] == 0 and rec["rho_floor"] is None


def test_floor_q_refuses_a_state_below_the_floor_at_the_domain_edge():
    sol, st = _steady(IP_LOW_Q)
    q = sol.profiles(st)["q"]
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.floor_q(st, sol, 1.01 * q[-1])
    with pytest.raises(SAW.QFloorRefused, match="last interior face"):
        SAW.floor_q(st, sol, 0.5 * (q[-2] + q[-1]) if q[-2] > q[-1] else 1.0001 * q[-2])
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="q_floor"):
            SAW.floor_q(st, sol, bad)
    with pytest.raises(ValueError, match="q_floor"):
        CM.CurrentModelSettings(r0=GEO.R0, q_floor=0.0)


def test_single_slice_default_floors_the_model_state_then_adds_the_corrections():
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    assert cfg.core_relaxation == "q_floor" and cfg.q_floor == 1.0 and FIT.FitSettings().free == ("a_corr",)
    rf = CurrentState.uniform_faces(cfg.n_cells)
    # (1) the floor acts: the uncorrected state is floor_q of the model's steady state
    a0 = CM.assemble(lag, kin, src, {}, cfg, IP_LOW_Q)
    sol = CM.build_solver(lag, lag["jbs"], IP_LOW_Q)
    want, frec = SAW.floor_q(sol.steady_state(rf), sol, 1.0)
    np.testing.assert_array_equal(a0["state"].psi, want.psi)
    core = a0["core"]
    assert core["core_relaxation"] == "q_floor" and core["q_floor"] == 1.0 and core["acted"]
    assert core["rho_floor"] == pytest.approx(frec["rho_floor"] * lag["rho_map"].rho_edge, rel=1e-15)
    assert core["rho_floor_domain"] == frec["rho_floor"] and core["q_min_model"] >= 1.0 - ROUND_OFF
    # (2) the corrections are added AFTER the floor: their steady response is not floored away
    amps = (-0.03, 0.0, 0.0)             # a core correction that lowers q further
    a1 = CM.assemble(lag, kin, src, dict(a_corr=amps), cfg, IP_LOW_Q)
    shapes = CM.correction_shapes(lag, cfg, IP_LOW_Q)
    sol_c = CM.build_solver(lag, lag["jbs"] + sum(a * g for a, g in zip(amps, shapes)), IP_LOW_Q)
    resp = sol_c.steady_state(rf).psi - sol.steady_state(rf).psi
    np.testing.assert_allclose(a1["state"].psi - a0["state"].psi, resp, rtol=1e-12, atol=1e-12 * np.ptp(resp))
    assert a1["core"]["q_min_model"] != pytest.approx(a0["core"]["q_min_model"], rel=1e-6)
    # (3) where the floor does not act the default is the plain steady state with the corrections, to the bit
    amps = (0.03, 0.02, 0.01)
    none = CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="none")
    for a in ((0.0, 0.0, 0.0), amps):
        d = CM.assemble(lag, kin, src, dict(a_corr=a), cfg, GEO.IP)
        n = CM.assemble(lag, kin, src, dict(a_corr=a), none, GEO.IP)
        assert not d["core"]["acted"]
        np.testing.assert_array_equal(d["state"].psi, n["state"].psi)
        np.testing.assert_array_equal(d["jphi"], n["jphi"])


def test_the_q_floor_setting_moves_the_floor():
    lag, kin, src = LR._fixture_inputs()
    out = {}
    for qf in (0.95, 1.0, 1.05):
        a = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, q_floor=qf), IP_LOW_Q)
        out[qf] = a["core"]
        assert a["core"]["q_floor"] == qf and a["core"]["q_min_model"] == pytest.approx(qf, rel=ROUND_OFF)
    assert out[0.95]["rho_floor"] < out[1.0]["rho_floor"] < out[1.05]["rho_floor"]


def test_opt_in_sawtooth_treatments_are_reachable_and_refused_elsewhere():
    lag, kin, src = LR._fixture_inputs()
    ls = float(np.log(SAW.crash_strength(0.37, 1.0 / 0.05)))
    orb = CM.assemble(lag, kin, src, dict(log_strength=ls, f=0.37),
                      CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="periodic_orbit"), IP_LOW_Q)
    assert orb["sawtooth"]["mode"].startswith("periodic orbit") and orb["sawtooth"]["active"]
    assert orb["core"]["core_relaxation"] == "periodic_orbit" and orb["core"]["q_floor"] is None
    crash = CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="crash_times", crash_times=(-0.2, -0.15, -0.1, -0.05),
                                    window=(-0.04, 0.0), crash_record_start=-2.0, history=0.3)
    m = CM.assemble(lag, kin, src, dict(log_strength=ls, f=0.37), crash, IP_LOW_Q)
    assert m["sawtooth"]["mode"] == "measured crash times" and m["core"]["core_relaxation"] == "crash_times"
    # the sawtooth parameters and crash times are refused where their treatment is not selected
    with pytest.raises(ValueError, match="opt-in sawtooth"):
        CM.assemble(lag, kin, src, dict(log_strength=ls), CM.CurrentModelSettings(r0=GEO.R0), IP_LOW_Q)
    with pytest.raises(ValueError, match="needs the sawtooth parameter"):
        CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="periodic_orbit"), IP_LOW_Q)
    with pytest.raises(ValueError, match="explicit|opt-in"):
        CM.CurrentModelSettings(r0=GEO.R0, crash_times=(0.1,))
    with pytest.raises(ValueError, match="core_relaxation must be one of"):
        CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="sawtooth")
    with pytest.raises(ValueError, match="opt-in sawtooth core treatments"):
        FIT.reconstruct(SimpleNamespace(ms=None), kin, src, CM.CurrentModelSettings(r0=GEO.R0),
                        FIT.FitSettings(free=("log_strength", "a_corr")))


def test_series_default_floors_every_step_with_the_single_slice_operator(monkeypatch):
    """Both modes use the same treatment by default: the first slice is solved with core_relaxation 'q_floor' and the
    series' q_floor, and every interval floors the evolved state after each step with the same operator."""
    lag = SD._lag()
    ip = IP_LOW_Q
    fm = SD.FakeFM(dict(pred=lag))
    st0 = SD._steady(lag, ip)
    seen = []

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        seen.append((cfg.core_relaxation, cfg.q_floor, state_fn is None))
        fm_.n_solves += 1
        if state_fn is None:
            return SD._result(lag, st0)
        st = state_fn(lag)
        res = SD._result(lag, st)
        res.assembly["core"] = dict(st.meta["core"])      # what current_model.assemble records for a supplied state
        return res

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), cfg, SE.SeriesSettings(initial="relaxed"))
    assert out["status"] == "completed" and out["first_slice_core_relaxation"] == "q_floor"
    assert seen[0] == ("q_floor", 1.0, True) and all(s[1] == 1.0 for s in seen)
    # the starting state has q < 1; every later state is held at the floor, and the slices record it
    sol = CM.build_solver(lag, lag["jbs"], ip)
    assert sol.profiles(st0)["q"][1:-1].min() < 1.0
    for rec in out["slices"][1:]:
        st = CurrentState(rec["carried_state"]["rho_faces"], rec["carried_state"]["psi"],
                          rec["carried_state"]["psi_boundary"])
        assert sol.profiles(st)["q"].min() >= 1.0 - ROUND_OFF
        assert rec["core"]["core_relaxation"] == "q_floor" and rec["core"]["q_floor"] == 1.0
        assert rec["core"]["n_steps"] == 20 and "rho_floor" in rec["core"]
    assert out["slices"][1]["core"]["acted_in_interval"] and out["slices"][1]["core"]["n_steps_floor_acted"] >= 1
    # one step of the series' advance equals one plain step followed by the operator
    s = SE.SeriesSettings()
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 1e-3, ip, ip)
    st_in = CurrentState(st0.rho_faces, st0.psi, st0.psi_boundary, 0.0)
    end, core = SE.advance_q_floor(isol, st_in, 1e-3, s, 1e-3, 1.0)
    ref, _ = SAW.floor_q(isol.step(st_in, 1e-3), isol, 1.0)
    np.testing.assert_array_equal(end.psi, ref.psi)
    assert core["n_steps"] == 1 and core["core_relaxation"] == "q_floor"


def test_series_floor_does_not_act_on_the_in_window_corrections(monkeypatch):
    """Where the floor does not act, the q-floor series is the series without any core treatment (to round-off), with
    and without an in-window correction source; the corrections' response is added after the floor."""
    lag = SD._lag()
    sol_pred = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, GEO.IP, GEO.IP)[0]
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, GEO.IP)))
    assert sol_pred.profiles(st_in)["q"].min() > 1.0
    shapes = CM.correction_shapes(lag, CM.CurrentModelSettings(r0=GEO.R0), GEO.IP, form="in_window")
    c = sum(a * g for a, g in zip((0.02, -0.01, 0.01), shapes))
    sol_all = SE.interval_solver(lag, lag, lag["jbs"] + c, lag["jbs"] + c, 0.0, 0.02, GEO.IP, GEO.IP)[0]
    s = SE.SeriesSettings()
    end, core = SE.advance_q_floor(sol_pred, st_in, 0.02, s, 1e-3, 1.0, sol_all=sol_all)
    plain = sol_all.advance(st_in, 0.02, dt=1e-3)
    np.testing.assert_allclose(end.psi, plain.psi, rtol=0, atol=1e-12 * np.ptp(plain.psi))
    assert core["n_steps_floor_acted"] == 0 and core["corrections"].startswith("in-window")
