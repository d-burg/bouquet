"""The resistive q floor with a CONTINUOUS flattening radius (the default rule of ``current.sawtooth``), its branches
on the model state's q, and the records of what the floor did.

Synthetic geometry as in ``test_q_floor.py`` (no OFT).  Checked:

* the radius is found to round-off, so where the floor is met q_min equals it (not one face's step above it), and the
  radius lies between faces;
* a cell cut by the region's edge takes the overlap-weighted conductivity, cells wholly inside the value at the edge,
  cells outside their own; the state is continuous when the edge crosses a face;
* the radius and q0 converge with the grid; the radius and q0 move continuously and monotonically with the level;
* the three branches: (a) q below the floor on axis -- the region from the axis; (b) below it off axis only -- no
  change, recorded; (c) nowhere -- no change; the time-dependent rule takes on every step the single-slice region of
  the step's sources (the same branches), not a region read from the evolving state;
* the records (loop voltage, l_i, current moved outside the radius, achieved q) are what they say and cannot change a
  fit or a series.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
import test_fit_unit as FU  # noqa: E402
import test_series_driver as SD  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import CurrentState  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

IP_MET = 1.86 * GEO.IP            # the floor 1 acts and is met on the fixture (q0 of the plain state ~0.8)
IP_LOW_Q = 2.2 * GEO.IP           # q < 1 out to rho ~0.56 (as in test_q_floor.py)
ULP2 = 4.5e-16                    # two ulps: the diffusion's 1/(1/sigma) round trip through its resistivity hook
CUT_RTOL = 1e-14                  # the cut cell's weighted conductivity through the same round trip (a few ulps)
Q_LAND = 1e-9                     # q_min above the floor where met: the radius is bisected to 1e-14 in rho


def _solver(ip, jni=None):
    lag = GEO._lag()
    return CM.build_solver(lag, lag["jbs"] if jni is None else jni, ip)


def _fq(sol, st):
    return SAW.face_q(st, sol.geometry_at(st.time).phi_b)


def _reversed_shear():
    """Off-axis co-current source: q0 above the floor, an off-axis minimum below it (as in test_q_floor.py)."""
    lag = GEO._lag()
    rho = lag["phys"]["rho"]
    jni = lag["jbs"] + 8.0 * np.max(np.abs(lag["jbs"])) * np.exp(-0.5 * ((rho - 0.5) / 0.08) ** 2)
    return CM.build_solver(lag, jni, 2.3 * GEO.IP)


def test_continuous_radius_lands_q_min_on_the_floor_between_faces():
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_rule"] == "continuous" and rec["floor_branch"] == "axis" and rec["acted"]
    assert rec["q_floor_met"] and 0.0 <= rec["q_min_after"] - 1.0 <= Q_LAND
    assert np.min(_fq(s2, st)) == rec["q_min_after"] and rec["rho_floor_inner"] == 0.0
    r = rec["rho_floor"]
    assert not np.any(rf == r) and 0.0 < r < 1.0 and rec["n_cells_cut"] == 1
    assert rec["n_cells_flattened"] == int(np.count_nonzero(rf[:-1] < r))
    # Ip is the steady state's own; the boundary gauge is kept
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=1e-12)
    assert st.psi_boundary == st0.psi_boundary
    # the grid rule (FUSE's, comparison) snaps the radius to a face and lands above the floor by up to one face's step
    _, rec_g, _ = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    assert rec_g["rho_floor"] in rf and rec_g["q_min_after"] >= rec["q_min_after"]
    with pytest.raises(ValueError, match="rule"):
        SAW.resistive_floor_steady(sol, rf, 1.0, rule="face")


def test_cut_cell_takes_the_overlap_weighted_conductivity_and_the_state_is_continuous_in_the_radius():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    rc = 0.5 * (rf[1:] + rf[:-1])
    k = 30
    r = rf[k] + 0.3 * (rf[k + 1] - rf[k])            # cuts cell k, 30 % inside
    s2 = SAW.ResistiveFloor(sol, 1.0, rho_inv=r, rho_faces=rf).with_solver()
    sig = s2.coefficients(0.0, rf).sigma_c
    sig0 = sol.coefficients(0.0, rf).sigma_c
    s_r = float(sol.sigma_fn(0.0, np.array([r]))[0])
    np.testing.assert_allclose(sig[:k], s_r, rtol=ULP2, atol=0)                       # wholly inside
    w = (r - rf[k]) / (rf[k + 1] - rf[k])             # 0.3 to round-off
    np.testing.assert_allclose(sig[k], w * s_r + (1.0 - w) * sig0[k], rtol=CUT_RTOL, atol=0)  # cut cell
    np.testing.assert_allclose(sig[k + 1:], sig0[k + 1:], rtol=ULP2, atol=0)           # wholly outside
    assert rc[k] > r                                   # the cut cell's center is outside: weighted, not left alone
    # at a face the weighting is the cell rule exactly (cells inside the face take the value there, the rest their own)
    s_f = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k], rho_faces=rf).with_solver()
    sig_f = s_f.coefficients(0.0, rf).sigma_c
    np.testing.assert_allclose(sig_f[:k], float(sol.sigma_fn(0.0, np.array([rf[k]]))[0]), rtol=ULP2, atol=0)
    np.testing.assert_allclose(sig_f[k:], sig0[k:], rtol=ULP2, atol=0)
    # the state is continuous when the edge crosses a face
    d = 1e-9
    lo = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k] - d, rho_faces=rf).with_solver().steady_state(rf)
    hi = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k] + d, rho_faces=rf).with_solver().steady_state(rf)
    assert np.max(np.abs(hi.psi - lo.psi)) <= 1e-6 * np.ptp(lo.psi)


def test_radius_and_q0_converge_with_the_grid():
    sol = _solver(IP_MET)
    out = {n: SAW.resistive_floor_steady(sol, CurrentState.uniform_faces(n), 1.0)[1] for n in (50, 100, 200, 400)}
    r = {n: out[n]["rho_floor"] for n in out}
    assert abs(r[50] - r[100]) >= abs(r[100] - r[200]) >= abs(r[200] - r[400])
    assert abs(out[100]["q_axis_after"] - out[400]["q_axis_after"]) <= 1e-3
    assert all(out[n]["q_floor_met"] and 0.0 <= out[n]["q_min_after"] - 1.0 <= Q_LAND for n in out)


def test_radius_and_q0_move_continuously_with_the_level():
    """No steps: the radius moves by less than half a cell between levels 0.0025 apart (the grid rule moves by a whole
    face or not at all), and both the radius and q0 rise strictly with the level."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    levels = np.linspace(0.95, 1.05, 41)
    recs = [SAW.resistive_floor_steady(sol, rf, float(lv))[1] for lv in levels]
    r = np.array([x["rho_floor"] for x in recs])
    q0 = np.array([x["q_axis_after"] for x in recs])
    assert np.all(np.diff(r) > 0) and np.all(np.diff(q0) > 0)
    assert np.max(np.diff(r)) <= 0.5 * (rf[1] - rf[0])
    for lv, x in zip(levels, recs):
        assert x["q_floor_met"] and 0.0 <= x["q_min_after"] - lv <= Q_LAND


def test_floor_region_branches_on_the_model_q():
    rf = np.linspace(0.0, 1.0, 6)
    a = SAW.floor_region(rf, np.array([0.8, 0.9, 1.1, 1.3, 1.6, 2.0]), 1.0)
    assert a["branch"] == "axis" and a["rho_inner"] == 0.0 and a["rho_outer"] == pytest.approx(0.3, rel=1e-12)
    b = SAW.floor_region(rf, np.array([1.2, 0.95, 0.9, 1.1, 1.6, 2.0]), 1.0)
    assert b["branch"] == "off_axis" and b["rho_inner"] == pytest.approx(0.2 * 0.2 / 0.25 + 0.0, rel=1e-12)
    assert b["rho_outer"] == pytest.approx(0.4 + 0.2 * 0.1 / 0.2, rel=1e-12)
    c = SAW.floor_region(rf, np.array([1.2, 1.1, 1.05, 1.1, 1.6, 2.0]), 1.0)
    assert c["branch"] == "none" and c["rho_inner"] is None and c["rho_outer"] is None
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.floor_region(rf, np.array([0.8, 0.9, 1.1, 1.3, 1.6, 0.9]), 1.0)
    # a q that is not finite counts as below the floor
    assert SAW.floor_region(rf, np.array([np.nan, 0.9, 1.1, 1.3, 1.6, 2.0]), 1.0)["branch"] == "axis"


def test_branch_a_axis_below_the_floor_flattens_from_the_axis():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    assert _fq(sol, st0)[0] < 1.0
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_branch"] == "axis" and rec["acted"] and rec["rho_floor_inner"] == 0.0
    sig = s2.profiles(st)["conductivity"]
    s_r = float(sol.sigma_fn(0.0, np.array([rec["rho_floor"]]))[0])
    np.testing.assert_allclose(sig[rf[1:] <= rec["rho_floor"]], s_r, rtol=ULP2, atol=0)
    # the time-dependent rule takes the single-slice region of the sources; on this fixture the floor is not met, so
    # that region ends at the plain state's outermost upward crossing (the steady rule's fallback)
    mod = SAW.ResistiveFloor(sol, 1.0)
    info = mod.region_info(st0)
    up = [c["rho"] for c in SAW.floor_crossings(rf, _fq(sol, st0), 1.0) if c["direction"] == "up"][-1]
    assert info["branch"] == "axis" and info["rho_inner"] == 0.0 and info["rho_inv"] == up
    assert mod.region(st0) == (info["k"], up) and info["k"] == int(np.count_nonzero(rf[:-1] < up))


def test_branch_b_off_axis_minimum_only_leaves_the_state_unchanged_and_records_it():
    sol = _reversed_shear()
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    q = _fq(sol, st0)
    assert q[0] > 1.0 and q.min() < 1.0 < q[-1]
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert s2 is sol and rec["floor_branch"] == "off_axis" and rec["acted"] is False
    assert rec["q_floor_met"] is False and rec["rho_floor"] is None and rec["n_cells_flattened"] == 0
    np.testing.assert_array_equal(st.psi, st0.psi)
    assert [c["direction"] for c in rec["rho_floor_crossings"]] == ["down", "up"]
    assert rec["q_axis_after"] == rec["q_axis_before"] == q[0] and rec["q_min_after"] == q.min()
    # FUSE's rule (from the axis, here the grid rule kept for comparison) raises q0 on the same state: what branch (b)
    # avoids
    _, rec_g, _ = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    assert rec_g["acted"] and rec_g["q_axis_after"] > rec_g["q_axis_before"]
    # time-dependent: no change on a step from this state
    mod = SAW.ResistiveFloor(sol, 1.0)
    info = mod.region_info(st0)
    assert info["branch"] == "off_axis" and info["k"] == 0 and info["rho_inv"] is None
    ref = sol.step(st0, 1e-3).psi
    np.testing.assert_allclose(mod.with_solver().step(st0, 1e-3).psi, ref, rtol=0, atol=1e-12 * np.ptp(ref))


def test_time_dependent_region_is_the_single_slice_region_of_the_sources():
    """On a diffusion step the region is the single-slice region of the solver's sources at the step's time, whatever
    the state: here a floored state whose own q0 is just above the floor and whose minimum is just below it, off axis
    (its own q would say branch (b), no change, and switch the whole region off for that step), and the plain state
    (its own q would put the edge at its outermost crossing) both get the single-slice region; a step with it is a
    step with that region fixed.  Where the sources' steady state is in branch (b) or (c) nothing changes, whatever
    the state (the reversed-shear and q-above-floor cases are in the branch tests)."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st_ss, rec, _ = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_branch"] == "axis" and rec["acted"]
    r = rec["rho_floor"] - 0.5 * (rf[1] - rf[0])        # half a cell short of the radius that meets the floor
    st = SAW.ResistiveFloor(sol, 1.0, rho_inv=r, rho_faces=rf).with_solver().steady_state(rf)
    q = _fq(sol, st)
    assert q[0] >= 1.0 > q.min() and int(np.argmin(q)) > 0
    st0 = sol.steady_state(rf)
    for s_ in (st, st0, st_ss):
        mod = SAW.ResistiveFloor(sol, 1.0)
        info = mod.region_info(s_)
        assert info["branch"] == "axis" and info["rho_inner"] == 0.0 and info["rho_inv"] == rec["rho_floor"]
        assert info["k"] == rec["n_cells_flattened"] and info["q_axis_model_steady"] == rec["q_axis_before"]
        fixed = SAW.ResistiveFloor(sol, 1.0, rho_inv=rec["rho_floor"], rho_faces=rf).with_solver().step(s_, 1e-3)
        np.testing.assert_array_equal(mod.with_solver().step(s_, 1e-3).psi, fixed.psi)
    assert SAW.ResistiveFloor(sol, 1.0).region_info(st)["state_branch"] == "off_axis"
    assert SAW.ResistiveFloor(sol, 1.0).region_info(st0)["state_branch"] == "axis"
    # the floored steady state is a fixed point of the time-dependent rule: a step from it leaves it in place
    st1 = SAW.ResistiveFloor(sol, 1.0).with_solver().step(st_ss, 1e-3)
    np.testing.assert_allclose(st1.psi - st1.psi_boundary, st_ss.psi - st_ss.psi_boundary, rtol=0,
                               atol=1e-9 * np.ptp(st_ss.psi))


def test_time_dependent_geometry_is_evaluated_once_per_time_and_the_reuse_changes_no_number():
    """The time-dependent rule forms tens of steady states at each step's time; the solver keeps the geometry it
    evaluated at a time (a cache) instead of evaluating it again.  Every coefficient from a solver that reuses it is
    bitwise the coefficient of a solver that has never seen that time."""
    lag0, lag1 = SD._lag(q0=1.1), SD._lag(q0=1.4)
    args = (lag0, lag1, lag0["jbs"] + 1.0, 2.0 * lag1["jbs"] + 1.0, 0.0, 0.05, IP_LOW_Q, 1.1 * IP_LOW_Q)
    sol = SE.interval_solver(*args)[0]
    rf = CurrentState.uniform_faces(100)
    times = (0.0, 0.0125, 0.05, 0.0125, 0.0, 0.03, 0.0125)
    for t in times:
        c = sol.coefficients(t, rf)
        ref = SE.interval_solver(*args)[0].coefficients(t, rf)
        for name in ("k_f", "F_f", "m_c", "s_c", "w_c", "sigma_c", "jni_c"):
            np.testing.assert_array_equal(getattr(c, name), getattr(ref, name))
        assert (c.w_b, c.bc_value, c.layer_A, c.layer_B, c.F_lcfs) == (ref.w_b, ref.bc_value, ref.layer_A,
                                                                       ref.layer_B, ref.F_lcfs)
    assert len(sol._time_cache) == len(set(times))
    # the geometry differs between the ends (so the check above is not trivial)
    assert not np.array_equal(sol.coefficients(0.0, rf).k_f, sol.coefficients(0.05, rf).k_f)
    # a steady search and a step with the floor give the same numbers on a fresh solver
    _, rec, _ = SAW.resistive_floor_steady(sol, rf, 1.0, t=0.0125)
    _, rec2, _ = SAW.resistive_floor_steady(SE.interval_solver(*args)[0], rf, 1.0, t=0.0125)
    assert rec["rho_floor"] == rec2["rho_floor"] and rec["q_axis_after"] == rec2["q_axis_after"]


def test_branch_c_q_above_the_floor_everywhere_changes_nothing():
    sol = _solver(GEO.IP)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert s2 is sol and rec["floor_branch"] == "none" and rec["acted"] is False and rec["q_floor_met"]
    np.testing.assert_array_equal(st.psi, st0.psi)
    assert rec["d_v_loop_rel"] == 0.0 and rec["d_li3"] == 0.0 and rec["current_moved_outside_over_ip"] == 0.0
    info = SAW.ResistiveFloor(sol, 1.0).region_info(st0)
    assert info["branch"] == "none" and info["k"] == 0 and info["rho_inv"] is None


def test_the_floor_records_its_effect_against_the_same_model_without_it():
    """Lowering the conductivity at fixed Ip raises the loop voltage (Ip = sum of sigma V and the non-inductive
    current); the current taken from inside the radius returns outside it; the broader current lowers l_i.  The
    recorded values are the ones recomputed from the two states."""
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    p0, p1 = sol.profiles(st0), s2.profiles(st)
    assert rec["d_v_loop_rel"] > 0.0 and rec["current_moved_outside_over_ip"] > 0.0 and rec["d_li3"] < 0.0
    assert rec["d_v_loop_rel"] == pytest.approx(st.meta["v_loop"] / st0.meta["v_loop"] - 1.0, rel=1e-12)
    assert rec["d_li3"] == pytest.approx(p1["li3"] - p0["li3"], rel=1e-12)
    moved = (np.interp(rec["rho_floor"], rf, p0["I_enclosed"]) - np.interp(rec["rho_floor"], rf, p1["I_enclosed"]))
    assert rec["current_moved_outside_over_ip"] == pytest.approx(moved / p0["ip"], rel=1e-12)
    assert rec["q_axis_after"] == pytest.approx(_fq(s2, st)[0], rel=1e-15)
    # the assembly's core record carries them, the floor-effect record reads them, with the achieved q of a solve
    core = dict(rec, q_axis_model=1.02, q_min_model=0.99)
    q_check = dict(axis=dict(q0_eq=1.03))
    fe = CM.floor_effect_record(core, q_check, np.array([1.03, 1.0, 0.98, 1.5, np.nan]))
    assert fe["q_axis_achieved"] == 1.03 and fe["q_min_achieved"] == 0.98 and fe["floor_branch"] == "axis"
    assert fe["d_v_loop_rel"] == rec["d_v_loop_rel"] and fe["current_moved_outside_over_ip"] == rec[
        "current_moved_outside_over_ip"] and fe["note"].startswith("record only")
    assert CM.floor_effect_record(None)["q_axis_achieved"] is None


def test_floor_effect_is_a_record_that_cannot_change_a_fit(monkeypatch):
    kin, src = FU._kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    outs, calls = {}, []
    real = CM.floor_effect_record

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    def raising(*a, **k):
        calls.append(1)
        raise RuntimeError("injected")

    def garbage(*a, **k):
        calls.append(1)
        return dict(q_axis_achieved=-99.0, verdict="below")

    for name, fn in (("real", counting), ("raising", raising), ("garbage", garbage)):
        fm = FU.FakeFM()
        FU._install(monkeypatch, fm)
        monkeypatch.setattr(CM, "floor_effect_record", fn)
        n0 = len(calls)
        outs[name] = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=FU.SAW_FREE),
                                     params0=dict(log_strength=0.5, f=0.37))
        assert len(calls) == n0 + 1                       # made once, at the end
    ref = outs["real"]
    assert ref.record["floor_effect"]["note"].startswith("record only")
    assert outs["raising"].record["floor_effect"]["computed"] is False
    for name in ("raising", "garbage"):
        o = outs[name]
        assert o.status == ref.status and o.record["n_solves"] == ref.record["n_solves"]
        np.testing.assert_array_equal(o.theta, ref.theta)
        np.testing.assert_array_equal(o.covariance, ref.covariance)
        np.testing.assert_equal(o.chi2, ref.chi2)
        assert o.params == ref.params


def test_series_records_the_floor_effect_per_slice_and_the_record_cannot_change_the_series(monkeypatch):
    lag = SD._lag()
    ip = IP_LOW_Q
    st0 = SD._steady(lag, ip)

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 1
        if state_fn is None:
            return SD._result(lag, st0)
        st = state_fn(lag)
        res = SD._result(lag, st)
        res.assembly["core"] = dict(st.meta["core"])
        return res

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
    runs = {}
    real = CM.floor_effect_record

    def raising(*a, **k):
        raise RuntimeError("injected")

    for name, fn in (("real", real), ("raising", raising)):
        monkeypatch.setattr(SE._cm, "floor_effect_record", fn)
        runs[name] = SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(3) * 0.02, np.full(3, ip)), cfg,
                                   SE.SeriesSettings(initial="relaxed"))
    ref, bad = runs["real"], runs["raising"]
    assert ref["status"] == bad["status"] == "completed"
    for a, b in zip(ref["slices"][1:], bad["slices"][1:]):
        np.testing.assert_array_equal(a["carried_state"]["psi"], b["carried_state"]["psi"])
        assert a["status"] == b["status"] and b["floor_effect"]["computed"] is False
        c = a["core"]
        assert c["floor_branch"] in SAW.FLOOR_BRANCHES and "d_v_loop_rel" in c and "d_li3" in c
        assert "current_moved_outside_over_ip" in c and c["floor_effect_reference"].startswith("the same interval")
        fe = a["floor_effect"]
        assert fe["note"].startswith("record only")
        np.testing.assert_equal(fe["d_li3"], c["d_li3"])     # NaN on this mocked geometry (no layer for l_i)
    # where the floor acted in the interval the loop voltage rose against the same interval advanced without it
    c1 = ref["slices"][1]["core"]
    assert c1["acted_in_interval"] and c1["d_v_loop_rel"] > 0.0
    # where it does not act the advance equals the plain one to round-off and the effect is zero to round-off
    sol_pred = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, GEO.IP, GEO.IP)[0]
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, GEO.IP)))
    end, core = SE.advance_q_floor(sol_pred, st_in, 0.02, SE.SeriesSettings(), 1e-3, 1.0)
    assert core["n_steps_floor_acted"] == 0 and core["floor_branch"] == "none"
    assert abs(core["d_v_loop_rel"]) <= 1e-12 and core["current_moved_outside_over_ip"] == 0.0
    # l_i needs the outer layer's geometry, which this mocked geometry lacks (NaN on both sides)
    assert (np.isnan(core["li3_plain"]) and np.isnan(core["d_li3"])) or abs(core["d_li3"]) <= 1e-12
