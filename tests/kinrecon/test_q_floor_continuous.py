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
  fit or a series -- on the default q-floor path too, where the floor's effect record is made inside every assembly
  and every diffusion step (a record that raises or returns anything leaves theta, covariance, chi^2, solve count
  and carried states bitwise unchanged);
* the default rule's own properties (axis current lowered, face q = profiles q, meta = record, Ip after a
  time-dependent step, the not-met fallback, q_min ON the floor where met in an assembled state and in a fit);
* the record of the floor's branch at every assembly (count per branch, Jacobians with columns on different branches,
  the smallest distance of the plain axis q from the floor) shows a flip at the branch boundary and is record-only;
* the extrapolated axis q of the equilibrium is recorded under its own name (not as an achieved axis q), the other
  axis values named cross-checks with their spread; the model's axis q is read back, valid only where the q check
  passed; a fit inside an open branch log (a filter fit) appends its assemblies to it.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
from bouquet.kinrecon import coordinates as CO  # noqa: E402
import test_correction_local_return as LR  # noqa: E402
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
    assert rec["q_floor_met"] is False and info["q_floor_met_steady"] is False    # why rho_inv == up holds here
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


def test_the_domain_li_change_is_recorded_where_the_full_plasma_li_is_not_available():
    """The full-plasma l_i needs the outer layer's reference surfaces, which a series interval solver does not carry
    (NaN there); the record also carries the diffusion domain's l_i change, which is always available, in a single
    slice and in a series interval, recomputed from the same two states."""
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    want = s2.profiles(st)["li3_domain"] - sol.profiles(st0)["li3_domain"]
    assert np.isfinite(rec["d_li3_domain"]) and rec["d_li3_domain"] == pytest.approx(want, rel=1e-12)
    assert rec["d_li3_domain"] < 0.0                    # the broader current lowers l_i
    lag = SD._lag()
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, IP_LOW_Q, IP_LOW_Q)
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, IP_LOW_Q)))
    end, core = SE.advance_q_floor(isol, st_in, 0.02, SE.SeriesSettings(), 1e-3, 1.0)
    plain = isol.advance(st_in, 0.02, dt=1e-3, scheme=SE.SeriesSettings().scheme)
    mod_sol = SAW.ResistiveFloor(isol, 1.0).with_solver()
    want_s = mod_sol.profiles(end)["li3_domain"] - isol.profiles(plain)["li3_domain"]
    assert core["n_steps_floor_acted"] > 0 and np.isfinite(core["d_li3_domain"])
    assert core["d_li3_domain"] == pytest.approx(want_s, rel=1e-12, abs=1e-15)
    assert CM.floor_effect_record(core)["d_li3_domain"] == core["d_li3_domain"]


# ====================================================================== the default rule's own properties
@pytest.mark.parametrize("ip", [IP_MET, IP_LOW_Q])
def test_default_rule_lowers_the_axis_current_and_its_face_q_and_meta_are_consistent(ip):
    """On the default (continuous) rule, where it acts (met: IP_MET; not met: IP_LOW_Q): the core current on the axis
    is lowered and q raised; the operator's face q is the profiles' q; the state's meta says what the record says."""
    sol = _solver(ip)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    p0, p1 = sol.profiles(st0), s2.profiles(st)
    assert rec["floor_rule"] == "continuous" and rec["acted"] and rec["floor_branch"] == "axis"
    assert p1["jdotb"][0] < p0["jdotb"][0] and rec["q_min_after"] > rec["q_min_before"]
    np.testing.assert_allclose(_fq(s2, st), p1["q"], rtol=1e-12)
    m = st.meta["q_floor"]
    assert m["rho_floor"] == rec["rho_floor"] and m["rho_floor_inner"] == rec["rho_floor_inner"] == 0.0
    assert m["form"] == "resistive" and m["rule"] == "continuous" and m["branch"] == "axis" and m["acted"] is True
    assert m["q_floor"] == 1.0 and m["q_floor_met"] is rec["q_floor_met"] is (ip == IP_MET)
    assert rec["floor_effect_computed"] is True


@pytest.mark.parametrize("ip", [IP_MET, IP_LOW_Q])
def test_default_time_dependent_step_conserves_ip(ip):
    sol = _solver(ip)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    mod = SAW.ResistiveFloor(sol, 1.0)
    s2 = mod.with_solver()
    assert mod.rule == "continuous" and mod.region_info(st0)["k"] > 0
    st1 = s2.step(st0, 1e-3)
    assert s2.profiles(st1)["ip"] == pytest.approx(ip, rel=1e-12)


def test_default_rule_not_met_keeps_the_outermost_crossing_with_q_min_below_the_floor():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    q0 = _fq(sol, st0)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    up = [c["rho"] for c in SAW.floor_crossings(rf, q0, 1.0) if c["direction"] == "up"][-1]
    assert rec["q_floor_met"] is False and rec["q_min_after"] < 1.0 and np.min(_fq(s2, st)) < 1.0
    assert rec["rho_floor"] == up and rec["floor_branch"] == "axis"


def test_assembled_default_state_lands_q_min_on_the_floor_where_met():
    """The single-slice default (assembly) on the fixture: where the floor is met the floored model state's q_min IS
    the floor (to the bisection's round-off), and the scored model q (no corrections) carries the same minimum."""
    lag, kin, src = LR._fixture_inputs()
    for qf in (0.95, 1.0):                 # the plain model axis q here is ~0.905
        a = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, q_floor=qf), IP_MET)
        c = a["core"]
        assert c["acted"] and c["q_floor_met"] is True and c["floor_branch"] == "axis"
        assert 0.0 <= c["q_min_after"] - qf <= Q_LAND
        assert c["q_min_model"] == pytest.approx(c["q_min_after"], rel=1e-12)


# ====================================================================== records cannot change a default-path fit
CENTERS = np.array([0.15, 0.3, 0.45, 0.6])


def _default_fit(monkeypatch, ip=IP_MET, q_floor=1.0, free=("a_corr",), target_shift=0.03):
    """A fit on the DEFAULT core treatment (q floor, resistive, continuous) whose every evaluation is a real
    assembly on the fixture (no forward solve): the residuals are the scored model q at four radii against a target
    (the default model's q plus a shift); Jacobian columns, trials and the noise solve all assemble."""
    lag, kin, src = LR._fixture_inputs()
    kin = FU.SimpleNamespace(**vars(kin), source="toy")
    src = FU.SimpleNamespace(**vars(src), source="toy", convention={})
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor=q_floor)
    a0 = CM.assemble(lag, kin, src, {}, cfg, ip)
    target = np.interp(CENTERS, a0["q_model"]["rho"], a0["q_model"]["q"]) + target_shift
    fm = FU.FakeFM()
    FU._install(monkeypatch, fm)

    def _eval(fm_, params):
        asm = CM.assemble(lag, kin, src, params, cfg, ip)
        fm_.r = (np.interp(CENTERS, asm["q_model"]["rho"], asm["q_model"]["q"]) - target) / 0.02
        return asm

    def consistent(fm_, kin_, src_, params, cfg_, start, z0, label, initial_profiles=None, state_fn=None):
        fm_.n_solves += 1
        asm = _eval(fm_, params)
        solve = FU.SimpleNamespace(stats={"q_0": 1.0}, status="converged", converged=True,
                                   record={"policy": {"verification": {"verdict": "PASS"}}})
        return CM.ConsistentResult(True, "consistent", 1, asm, lag, solve, {"passes": [], "reason": None}, 0.01)

    def lagged(fm_, lag_, kin_, src_, params, cfg_, ip=None, start=None, z0=None, label="", state=None):
        fm_.n_solves += 1
        asm = _eval(fm_, params)
        return FU.SimpleNamespace(converged=True, status="converged"), asm, "converged"

    def residual_vector(fm_, ms, groups, settings, roles):
        return fm_.r.copy(), [("flux_loop", s_) for s_ in "abcd"], FU.SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent)
    monkeypatch.setattr(FIT._cm, "lagged_solve", lagged)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    return FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=free))


def _assert_same_fit(o, ref):
    assert o.status == ref.status and o.record["n_solves"] == ref.record["n_solves"]
    np.testing.assert_array_equal(o.theta, ref.theta)
    np.testing.assert_array_equal(o.covariance, ref.covariance)
    np.testing.assert_equal(o.chi2, ref.chi2)
    assert o.params == ref.params


def test_floor_records_cannot_change_a_default_path_fit(monkeypatch):
    """The default q-floor path: the floor's effect record made inside every assembly (``_floor_effect``) and the
    end-of-fit record (``floor_effect_record``) raising, or returning anything, leave the fit's status, parameters,
    covariance, chi^2 and solve count bitwise unchanged; a failed record says so."""
    ref = _default_fit(monkeypatch)
    assert ref.record["core"]["floor_branch"] == "axis" and ref.record["core"]["acted"]
    assert ref.record["core"]["floor_effect_computed"] is True
    # the fit's own center lands q_min on the floor (the floored model state, before the corrections)
    assert ref.record["core"]["q_floor_met"] is True and 0.0 <= ref.record["core"]["q_min_after"] - 1.0 <= Q_LAND
    assert len(ref.theta) == 3 and np.any(ref.theta != 0.0)

    def raising(*a, **k):
        raise RuntimeError("injected")

    junk = dict.fromkeys(SAW.FLOOR_EFFECT_KEYS, -99.0)
    for name, fe, fer in (("effect raising", raising, None),
                          ("effect garbage", lambda *a, **k: dict(acted=False, rho_floor=None, q_floor_met=False), None),
                          ("effect junk values", lambda *a, **k: dict(junk, acted=False, rho_floor=0.9), None),
                          ("effect not a dict", lambda *a, **k: 7, None),
                          ("end record raising", None, raising),
                          ("end record garbage", None, lambda *a, **k: dict(q_axis_achieved=-99.0)),
                          ("both raising", raising, raising)):
        with monkeypatch.context() as mp:
            if fe is not None:
                mp.setattr(SAW, "_floor_effect", fe)
            if fer is not None:
                mp.setattr(CM, "floor_effect_record", fer)
            o = _default_fit(mp)
        _assert_same_fit(o, ref)
        c = o.record["core"]
        assert c["acted"] and c["rho_floor"] == ref.record["core"]["rho_floor"], name
        if fe is raising or name in ("effect garbage", "effect not a dict"):
            assert c["floor_effect_computed"] is False and np.isnan(c["d_v_loop_rel"]) and c["floor_effect_reason"]
        if fe is not None and name == "effect junk values":
            assert c["d_v_loop_rel"] == -99.0          # junk values stay in the record's own keys only
        if fer is raising:
            assert o.record["floor_effect"]["computed"] is False


def test_series_floor_records_cannot_change_the_carried_states(monkeypatch):
    """A default series (mocked solves, real interval advances with the floor on every step): the floor's effect
    record inside every step's region search (``_floor_effect``) and the interval's record (the unfloored advance and
    its profiles) raising or returning anything leave every carried state bitwise unchanged."""
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

    def raising(*a, **k):
        raise RuntimeError("injected")

    def run(mp):
        mp.setattr(SE._cm, "consistent_solve", consistent)
        cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
        return SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(3) * 0.02, np.full(3, ip)), cfg,
                             SE.SeriesSettings(initial="relaxed"))

    with monkeypatch.context() as mp:
        ref = run(mp)
    assert ref["status"] == "completed" and ref["slices"][1]["core"]["floor_effect_computed"] is True
    for name, patches in (("step effect raising", [(SAW, "_floor_effect", raising)]),
                          ("step effect garbage", [(SAW, "_floor_effect", lambda *a, **k: dict(acted=False))]),
                          ("interval values raising", [(SE, "_interval_floor_effect_values", raising)]),
                          ("interval values garbage",
                           [(SE, "_interval_floor_effect_values", lambda *a, **k: dict(floor_branch="none"))]),
                          ("step branch record raising", [(SE, "_step_branch_record", raising)]),
                          ("all", [(SAW, "_floor_effect", raising), (SE, "_interval_floor_effect_values", raising),
                                   (SE, "_step_branch_record", raising)])):
        with monkeypatch.context() as mp:
            for mod, attr, fn in patches:
                mp.setattr(mod, attr, fn)
            out = run(mp)
        assert out["status"] == ref["status"], name
        for a, b in zip(ref["slices"][1:], out["slices"][1:]):
            np.testing.assert_array_equal(a["carried_state"]["psi"], b["carried_state"]["psi"])
            assert a["carried_state"]["psi_boundary"] == b["carried_state"]["psi_boundary"]
            assert a["status"] == b["status"] and a["core"]["rho_floor"] == b["core"]["rho_floor"], name
        if "interval values" in name or name == "all":
            assert out["slices"][1]["core"]["floor_effect_computed"] is False


# ====================================================================== branch-flip record (record only)
def test_branch_flip_record_counts_branches_jacobians_and_the_distance_from_the_floor():
    log = [dict(branch="axis", q_floor=1.0, q_axis_plain=0.998), dict(branch="axis", q_floor=1.0, q_axis_plain=0.999),
           dict(branch="off_axis", q_floor=1.0, q_axis_plain=1.0004), dict(branch="axis", q_floor=1.0, q_axis_plain=0.9)]
    jac = [dict(label="it0", branches=["axis", "off_axis"], mixed=True), dict(label="it1", branches=["axis"],
                                                                                mixed=False)]
    r = CM.branch_flip_record(log, jac, dict(floor_branch="axis"))
    assert r["n_assemblies"] == 4 and r["assemblies_per_branch"] == {"axis": 3, "off_axis": 1}
    assert r["n_branch_changes_in_call_order"] == 2 and r["branch_changed"] is True
    assert r["n_jacobians"] == 2 and r["n_jacobians_mixed_branches"] == 1 and r["jacobians_mixed_branches"][0][
        "label"] == "it0"
    assert r["min_abs_plain_q_axis_minus_floor"] == pytest.approx(4e-4, rel=1e-9) and r["final_branch"] == "axis"
    assert r["note"].startswith("record only")
    one = CM.branch_flip_record(log[:2], [], dict(floor_branch="axis"))
    assert one["branch_changed"] is False and one["n_branch_changes_in_call_order"] == 0
    # a series interval entry: the diffusion steps per branch
    s = CM.branch_flip_record([dict(branch="axis", q_floor=1.0, q_axis_plain=None,
                                    branch_steps={"axis": 12, "off_axis": 8}, q_axis_plain_min_abs_dev=2e-3)])
    assert s["branch_changed"] is True and s["diffusion_steps_per_branch"] == {"axis": 12, "off_axis": 8}
    assert s["min_abs_plain_q_axis_minus_floor"] == 2e-3


def _boundary_level(ip=IP_MET, ds=0.05):
    """A floor level between the plain model axis q at the default bootstrap scale and at the scale one Jacobian
    step above it, so that the center and its s_bs column sit on different branches."""
    lag, _, _ = LR._fixture_inputs()
    rf = CurrentState.uniform_faces(100)
    q = [float(_fq(s_, s_.steady_state(rf))[0]) for s_ in (CM.build_solver(lag, f * lag["jbs"], ip)
                                                          for f in (1.0, 1.0 + ds))]
    return 0.5 * (q[0] + q[1]), abs(q[1] - q[0])


def test_branch_flip_is_recorded_on_a_fit_at_the_branch_boundary_and_the_record_is_record_only(monkeypatch):
    """A default-path fit (as above, the bootstrap scale free) whose floor level lies between the plain axis q at
    the center and at its s_bs column: the record shows the flip (a Jacobian with columns on different branches, the
    smallest distance of the plain axis q from the level); the record raising leaves the fit bitwise unchanged."""
    level, dq = _boundary_level()
    assert dq > 0.0
    ref = _default_fit(monkeypatch, q_floor=level, free=("s_bs", "a_corr"))
    r = ref.record["floor_branch_record"]
    assert r["note"].startswith("record only") and r["n_assemblies"] > 0
    assert r["branch_changed"] is True and r["n_jacobians_mixed_branches"] >= 1
    assert r["min_abs_plain_q_axis_minus_floor"] <= 0.5 * dq * (1.0 + 1e-9)
    assert r["final_branch"] == ref.record["core"]["floor_branch"]

    def raising(*a, **k):
        raise RuntimeError("injected")

    for patches in ([(CM, "_log_assembly", raising)], [(CM, "branch_flip_record", raising)],
                    [(FIT, "_note_jacobian", raising)], [(CM, "branch_flip_record", lambda *a, **k: dict(x=1))],
                    [(CM, "_log_assembly", raising), (CM, "branch_flip_record", raising),
                     (FIT, "_note_jacobian", raising), (CM, "open_branch_log", raising)]):
        with monkeypatch.context() as mp:
            for mod, attr, fn in patches:
                mp.setattr(mod, attr, fn)
            o = _default_fit(mp, q_floor=level, free=("s_bs", "a_corr"))
        _assert_same_fit(o, ref)
    # away from the boundary nothing flips
    far = _default_fit(monkeypatch)
    assert far.record["floor_branch_record"]["branch_changed"] is False
    assert far.record["floor_branch_record"]["n_jacobians_mixed_branches"] == 0


def test_series_interval_records_the_branch_of_every_step():
    """An interval whose plain model axis q crosses the floor (an Ip ramp): the steps' branches are counted per
    branch, the record shows the change, and the smallest distance of the plain axis q from the floor is recorded."""
    lag = SD._lag_ni()
    rf = CurrentState.uniform_faces(100)

    def q0_plain(ip_):
        s_ = CM.build_solver(lag, lag["jbs"], ip_)
        return float(_fq(s_, s_.steady_state(rf))[0])
    lo, hi = GEO.IP, 3.0 * GEO.IP                 # q0 falls with Ip: bracket the Ip where the plain q0 is 1
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q0_plain(mid) > 1.0 else (lo, mid)
    ip1 = 0.5 * (lo + hi)
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, 0.97 * ip1, 1.03 * ip1)
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, 0.97 * ip1)))
    end, core = SE.advance_q_floor(isol, st_in, 0.02, SE.SeriesSettings(), 1e-3, 1.0)
    steps = core["branch_steps"]
    assert sum(steps.values()) == core["n_steps"] and len(steps) >= 2 and "axis" in steps
    assert core["q_axis_model_steady_min_abs_dev_floor"] < abs(q0_plain(0.97 * ip1) - 1.0)
    r = CM.branch_flip_record([dict(branch=core["floor_branch"], q_floor=1.0, q_axis_plain=None,
                                    branch_steps=steps,
                                    q_axis_plain_min_abs_dev=core["q_axis_model_steady_min_abs_dev_floor"])])
    assert r["branch_changed"] is True and r["diffusion_steps_per_branch"] == steps


def _synthetic_axis_values():
    """Axis values for the record tests, all from the synthetic fixture (no discharge): the floored model state's own
    axis face q, its extrapolation from the q check's window, and round values for the equilibrium-side inputs."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    q = _fq(s2, st)
    q_ax = float(q[0])
    q_ext_model = float(CO.axis_q(rf, q, (0.05, 0.15), 2)["q0"])
    return q_ax, q_ext_model


def test_floor_effect_record_names_the_extrapolated_axis_q_and_keeps_the_others_as_cross_checks():
    """The q check's extrapolation of the solved equilibrium is recorded under its own name (not as an achieved axis
    q), with its definition; the trace of the innermost surface, the solver's q_0 (q at psi_N = 0.02) and the model's
    values are named cross-checks with their spread and the estimator's offset on the model, where the model's own
    axis value is known.  The old key names carry the same values (aliases).  Numbers from the synthetic fixture."""
    q_ax, q_ext_model = _synthetic_axis_values()
    q_ext_eq, q_trace, q_s0 = 1.1, 0.5, 1.25
    core = dict(q_floor=1.0, floor_branch="axis", acted=True, q_axis_model=q_ax, q_min_model=q_ax,
                floor_effect_computed=False, floor_effect_reason="RuntimeError: injected")
    q_check = dict(axis=dict(q0_eq=q_ext_eq, q0_model=q_ext_model, fit_rms_rel_eq=0.01, window=[0.05, 0.15], degree=2))
    fe = CM.floor_effect_record(core, q_check, np.array([1.1, 1.05, 1.3]), q_trace, -q_s0)
    assert fe["q_axis_achieved"] == q_ext_eq and fe["q_axis_achieved_definition"] == CM.Q_AXIS_ACHIEVED_DEFINITION
    assert fe["q_axis_achieved_window"] == [0.05, 0.15] and fe["q_axis_achieved_degree"] == 2
    cc = fe["q_axis_cross_checks"]
    assert cc == dict(q_traced_innermost=q_trace, q_solver_q0=q_s0, q_model_axis_extrapolated=q_ext_model,
                      q_model_axis=q_ax)
    assert fe["q_axis_spread"] == pytest.approx(q_s0 - q_trace, rel=1e-12)
    assert fe["q_axis_extrapolation_offset_model"] == pytest.approx(q_ext_model - q_ax, rel=1e-12)
    assert fe["effect_computed"] is False and fe["effect_reason"].startswith("RuntimeError")
    none = CM.floor_effect_record(core)
    assert none["q_axis_achieved"] is None and none["q_axis_spread"] is None
    # the new names (the record no longer calls the extrapolation achieved; the old names are aliases)
    assert fe["q_axis_extrapolated_eq"] == q_ext_eq and fe["q_min_traced_eq"] == 1.05
    assert fe["q_axis_extrapolated_eq_definition"] == CM.Q_AXIS_EXTRAPOLATED_EQ_DEFINITION
    assert "not the achieved axis q" in CM.Q_AXIS_EXTRAPOLATED_EQ_DEFINITION
    for old, new in CM.DEPRECATED_FLOOR_EFFECT_ALIASES.items():
        assert fe[old] == fe[new] and old in fe["deprecated_keys"]
    assert "achieved" not in "".join(CM.DEPRECATED_FLOOR_EFFECT_ALIASES.values())
    assert "edge" in CM.floor_effect_record.__doc__ and "not resolved there" not in CM.floor_effect_record.__doc__


def test_floor_effect_record_reads_back_the_model_axis_q_only_where_the_q_check_passed():
    """The read-back of the core level is the model's own axis q of the consistent state; it is marked valid only
    where the q consistency check passed, and invalid (with the reason) without a check, with a failed one, or without
    a model axis q.  Record only."""
    q_ax, _ = _synthetic_axis_values()
    core = dict(q_floor=1.0, q_axis_model=q_ax)
    ok = CM.floor_effect_record(core, dict(passed=True, max_rel=0.004, axis={}))
    assert ok["q_axis_model_readback"] == q_ax and ok["q_axis_model_readback_valid"] is True
    assert ok["q_axis_model_readback_invalid_reason"] is None
    assert ok["q_axis_model_readback_definition"] == CM.Q_AXIS_MODEL_READBACK_DEFINITION
    bad = CM.floor_effect_record(core, dict(passed=False, max_rel=0.02, axis={}))
    assert bad["q_axis_model_readback"] == q_ax and bad["q_axis_model_readback_valid"] is False
    assert "not passed" in bad["q_axis_model_readback_invalid_reason"]
    none = CM.floor_effect_record(core)
    assert none["q_axis_model_readback_valid"] is False and none["q_axis_model_readback_invalid_reason"] == "no q check"
    nomodel = CM.floor_effect_record(dict(q_floor=1.0), dict(passed=True, axis={}))
    assert nomodel["q_axis_model_readback"] is None and nomodel["q_axis_model_readback_valid"] is False
    # a truthy value that is not True does not count as passed
    odd = CM.floor_effect_record(core, dict(passed=1, axis={}))
    assert odd["q_axis_model_readback_valid"] is False


def test_branch_flip_record_states_the_total_and_the_truncation_of_mixed_jacobians():
    """The list of Jacobians with columns on different branches keeps the first 20; the record states the total and
    whether the list was truncated."""
    log = [dict(branch="axis", q_floor=1.0, q_axis_plain=0.99)]
    many = [dict(label=f"it{i}", branches=["axis", "off_axis"], mixed=True) for i in range(25)]
    r = CM.branch_flip_record(log, many + [dict(label="x", branches=["axis"], mixed=False)])
    assert r["n_jacobians_mixed_branches"] == 25 and r["jacobians_mixed_branches_total"] == 25
    assert len(r["jacobians_mixed_branches"]) == 20 and r["jacobians_mixed_branches_truncated"] is True
    few = CM.branch_flip_record(log, many[:2])
    assert few["jacobians_mixed_branches_total"] == 2 and few["jacobians_mixed_branches_truncated"] is False


def test_failed_interval_record_keeps_its_reference_text():
    """A failed interval floor-effect record carries the interval's reference text (not None), and the slice's
    floor-effect record reads a reference that is None as the default text."""
    def raising():
        raise RuntimeError("injected")
    out = SE._interval_floor_effect(None, None, None, None, None, 1.0, advance_plain=raising)
    assert out["floor_effect_computed"] is False and out["floor_effect_reason"].startswith("RuntimeError")
    assert out["floor_effect_reference"] == SE.INTERVAL_FLOOR_EFFECT_REFERENCE
    fe = CM.floor_effect_record(dict(out, q_floor=1.0))
    assert fe["reference"] == SE.INTERVAL_FLOOR_EFFECT_REFERENCE
    fe0 = CM.floor_effect_record(dict(q_floor=1.0, floor_effect_reference=None))
    assert fe0["reference"].startswith("steady state of the same model sources")


def test_a_fit_inside_an_open_branch_log_appends_its_assemblies_to_it(monkeypatch):
    """A fit run while a branch log is open (a series slice's filter fit) appends its assemblies to that log, marked
    with the fit's label; its own record and its result are unchanged."""
    ref = _default_fit(monkeypatch)
    parent, tok = CM.open_branch_log()
    try:
        parent.append(dict(branch="none", q_floor=1.0, q_axis_plain=None))
        o = _default_fit(monkeypatch)
    finally:
        CM.close_branch_log(tok)
    _assert_same_fit(o, ref)
    assert o.record["floor_branch_record"] == ref.record["floor_branch_record"]
    # the helper's own assembly before the fit is logged directly; the fit's assemblies carry the fit's label
    nested = [e for e in parent if "nested_fit" in e]
    assert len(nested) == ref.record["floor_branch_record"]["n_assemblies"] > 0
    assert all(e["nested_fit"] == "fit" for e in nested) and len(parent) == len(nested) + 2
    assert "nested_fit" not in parent[0] and parent[0]["branch"] == "none"
    assert CM.current_branch_log() is None


def test_series_slice_branch_record_includes_the_filter_fits_assemblies(monkeypatch):
    """In a series slice the slice's branch log is open while its filter fit runs: the filter fit's assemblies are in
    the slice's record (``nested_fit`` entries), not only the assemblies of the slice's own solves."""
    seen = {}

    def fake_filter(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1):
        log = CM.current_branch_log()
        log.append(dict(branch="axis", q_floor=1.0, q_axis_plain=0.97, nested_fit="filter"))
        seen["n"] = len(log)
        return None, res

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
    monkeypatch.setattr(SE, "_filter_slice", fake_filter)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
    out = SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(2) * 0.02, np.full(2, ip)), cfg,
                        SE.SeriesSettings(initial="relaxed", filter=True))
    r = out["slices"][1]["floor_branch_record"]
    assert seen["n"] >= 1 and r["n_assemblies"] == seen["n"]
