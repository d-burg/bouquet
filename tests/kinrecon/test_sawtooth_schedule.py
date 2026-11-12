"""Time-dependent sawtooth mode, slice schedules, window averages and parameter helpers (synthetic inputs).

Bounds: crash times are hit exactly (bitwise equality of the event times with the schedule); window averages of
piecewise-linear test functions are exact up to round-off (1e-13 relative); the parameter helpers are exact
inverses up to round-off (1e-12 relative).
"""
import numpy as np
import pytest

from bouquet.kinrecon.current import CurrentDiffusion, CurrentState, FluxSurfaceGeometry, IpBoundary
from bouquet.kinrecon.current.sawtooth import (DEFAULT_HISTORY, CrashSchedule, Trajectory, advance_with_crashes,
                                               crash_strength, from_strength_coordinates, move_across_strength,
                                               move_along_strength, q1_shear_diagnostic, rate_from_strength,
                                               reconnect, schedule_for_slice, simulate_slice,
                                               to_strength_coordinates)

RF = np.linspace(0.0, 1.0, 201)


def make_solver(ip=0.8e6):
    geo = FluxSurfaceGeometry.cylinder(0.6, 2.0, 2.0, n=401)
    rho = geo.rho
    sig = 1.0e8 * (0.033 + 0.967 * (1 - rho ** 2) ** 1.5) ** 1.5
    return CurrentDiffusion(geo, (rho, sig), IpBoundary(ip))


def test_schedule_lands_exactly_on_crash_times():
    sv = make_solver()
    st = sv.steady_state(RF)
    crashes = [0.0123456789, 0.0400000001, 0.07, 0.0917]               # deliberately off any uniform step grid
    traj = Trajectory()
    sch = CrashSchedule(crashes, f=0.5, trajectory=traj)
    out = advance_with_crashes(sv, st, 0.1, sch, dt=1e-3)
    assert out.time == 0.1
    assert [e.time for e in sch.events] == crashes                    # bitwise
    assert all(e.record.active for e in sch.events)
    t = traj.times
    for tc in crashes:                                                # pre and post states stored at the crash time
        assert np.count_nonzero(t == tc) == 2
    assert np.all(np.diff(t) >= 0) and np.all(np.diff(t)[np.diff(t) > 0] <= 1e-3 + 1e-15)
    # the stored post-crash state is the operator applied to the stored pre-crash state
    k = int(np.nonzero(t == crashes[1])[0][0])
    post, _ = reconnect(traj.states[k], sv.geometry_at(0.0).phi_b, 0.5)
    assert np.array_equal(post.psi, traj.states[k + 1].psi)


def test_skipped_crash_raises():
    sv = make_solver()
    sch = CrashSchedule([0.0105], f=1.0)
    sch.reset(0.0)
    with pytest.raises(RuntimeError):
        sv.advance(sv.steady_state(RF), 0.02, dt=1e-3, post_step=sch)  # plain advance steps over 0.0105


def test_reused_schedule_reports_only_the_current_runs_crashes():
    """A schedule reused for a second run from the same start applies the same crashes again and reports
    only those (it used to accumulate the first run's events as well)."""
    sv = make_solver()
    st = sv.steady_state(RF)
    sch = CrashSchedule([0.005, 0.012], f=0.5)
    a = advance_with_crashes(sv, st, 0.02, sch, dt=1e-3)
    first = [e.time for e in sch.events]
    b = advance_with_crashes(sv, st, 0.02, sch, dt=1e-3)
    assert first == [0.005, 0.012] and [e.time for e in sch.events] == first
    assert np.array_equal(a.psi, b.psi)
    advance_with_crashes(sv, st, 0.008, sch, dt=1e-3)
    assert [e.time for e in sch.events] == [0.005]


def test_crash_at_start_time_is_not_reapplied():
    sv = make_solver()
    st = sv.steady_state(RF, t=0.01)
    sch = CrashSchedule([0.01, 0.015], f=1.0)
    advance_with_crashes(sv, st, 0.02, sch, dt=1e-3)
    assert [e.time for e in sch.events] == [0.015]


def test_trajectory_window_average_exact():
    """Piecewise-linear function with a jump: exact integral by hand."""
    rf = RF
    tr = Trajectory()
    pts = [(0.0, 1.0), (1.0, 3.0), (1.0, 0.0), (2.0, 2.0), (4.0, 2.0)]   # jump at t = 1
    for t, v in pts:
        tr.add(CurrentState(rf, np.full(rf.size - 1, v), 0.0, t))
    g = lambda s: s.psi[0]                                                  # noqa: E731
    assert tr.window_average(g, 0.0, 4.0) == pytest.approx((2.0 + 1.0 + 4.0) / 4.0, rel=1e-13)
    # [0.5, 1.5]: 0.5*(2+3)/2 + 0.5*(0+1)/2 = 1.5
    assert tr.window_average(g, 0.5, 1.5) == pytest.approx(1.5, rel=1e-13)
    with pytest.raises(ValueError):
        tr.window_average(g, -1.0, 1.0)


def test_schedule_for_slice():
    ct = np.arange(0.0, 3.0, 0.05) + 0.003
    s = schedule_for_slice(ct, (2.0, 2.1))
    assert s.history == DEFAULT_HISTORY == 1.5
    assert s.t_start == pytest.approx(0.5)
    assert np.all((s.crash_times > s.t_start) & (s.crash_times <= 2.1))
    assert s.n_crashes_window == 2 and s.n_crashes_history == 30
    assert s.mean_interval == pytest.approx(0.05, rel=1e-9)
    assert s.last_crash_before_start == pytest.approx(0.453)
    with pytest.raises(ValueError):                                   # record shorter than the history
        schedule_for_slice(ct[ct > 1.0], (2.0, 2.1))
    s2 = schedule_for_slice(ct[ct > 1.0], (2.0, 2.1), history=0.9)
    assert s2.t_start == pytest.approx(1.1)
    s3 = schedule_for_slice(ct[ct > 1.0], (2.0, 2.1), record_start=0.4)  # detection covered 0.4 s onward
    assert s3.t_start == pytest.approx(0.5)


def test_simulate_slice_runs_and_records_window():
    sv = make_solver()
    ct = np.arange(0.0, 0.6, 0.02) + 0.001
    sched = schedule_for_slice(ct, (0.5, 0.55), history=0.3)
    out = simulate_slice(sv, RF, sched, f=0.37, dt=1e-3, nsteps_orbit=20)
    tr = out["trajectory"]
    assert tr.times[0] == 0.5 and tr.times[-1] == 0.55
    assert [e.time for e in out["events"]] == list(sched.crash_times)
    q0 = tr.window_average(lambda s: sv.profiles(s)["q"][0], 0.5, 0.55)
    assert 0.5 < q0 < 1.05


def test_strength_helpers():
    f, rate = 0.37, 20.0
    lam = crash_strength(f, rate)
    assert lam == pytest.approx(-np.log(1 - 0.999 * f) * rate, rel=1e-14)
    assert rate_from_strength(lam, f) == pytest.approx(rate, rel=1e-12)
    u, v = to_strength_coordinates(f, rate)
    f2, r2 = from_strength_coordinates(u, v)
    assert f2 == f and r2 == pytest.approx(rate, rel=1e-12)
    fa, ra = move_along_strength(f, rate, 0.1)
    assert fa == f and crash_strength(fa, ra) == pytest.approx(lam * np.exp(0.1), rel=1e-12)
    fc, rc_ = move_across_strength(f, rate, 0.2)
    assert fc == pytest.approx(0.57) and crash_strength(fc, rc_) == pytest.approx(lam, rel=1e-12)
    assert np.isfinite(crash_strength(1.0, rate))                     # f = 1 is finite through f_cap
    with pytest.raises(ValueError):
        move_across_strength(f, rate, -0.5)


def test_trigger_diagnostic_is_separate():
    sv = make_solver()
    st = sv.steady_state(RF)
    d = q1_shear_diagnostic(st, sv.geometry_at(0.0).phi_b)
    assert set(d) == {"rho_q1", "shear", "trigger"} and d["rho_q1"] > 0 and d["shear"] > 0
    # the operator's own q = 1 radius agrees with the diagnostic's
    _, rec = reconnect(st, sv.geometry_at(0.0).phi_b)
    assert d["rho_q1"] == pytest.approx(rec.rho_q1_outer, abs=1e-12)
