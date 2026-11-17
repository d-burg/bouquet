"""The opt-in speed settings (``bouquet.kinrecon.speed``): validation, the default path never entering them, and the
worker-process mechanics (results, errors and a dead worker) without the solver.

The worker tests fork small processes that do no computation; they need ``os.fork``."""

import os
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import speed as SP

_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def test_default_is_serial_and_check_returns_none():
    assert SP.SpeedSettings().column_workers == 1
    assert not SP.SpeedSettings().active
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=1))
    assert SP.check_speed(None, fm) is None
    assert SP.check_speed(SP.SpeedSettings(), fm) is None
    assert SP.check_speed(SP.SpeedSettings(column_workers=4), fm).column_workers == 4


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_invalid_worker_counts_are_refused(bad):
    with pytest.raises(ValueError):
        SP.SpeedSettings(column_workers=bad)


def test_threaded_solver_is_refused():
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=2))
    with pytest.raises(ValueError, match="nthreads"):
        SP.check_speed(SP.SpeedSettings(column_workers=2), fm)


def test_wrong_type_is_refused():
    with pytest.raises(TypeError):
        SP.check_speed(dict(column_workers=2), SimpleNamespace(settings=SimpleNamespace(nthreads=1)))


@_FORK
def test_worker_sends_results_in_order_and_receives_messages():
    def work(ch):
        ch.send(("column", 1, [1.0, 2.0]))
        msg = ch.recv()
        ch.send(("column", 2, msg["x"] * 2))

    w = SP.ForkWorker(work)
    try:
        assert w.recv() == ("column", 1, [1.0, 2.0])
        assert w.send(dict(x=21))
        assert w.recv() == ("column", 2, 42)
        assert w.recv() is None                   # the worker has ended
    finally:
        w.close()


@_FORK
def test_worker_exception_is_reported_not_raised_here():
    def work(ch):
        raise RuntimeError("boom in the worker")

    w = SP.ForkWorker(work)
    try:
        msg = w.recv()
        assert msg[0] == "error" and "boom in the worker" in msg[1]
    finally:
        w.close()


@_FORK
def test_dead_worker_gives_none_and_close_reaps_a_running_one():
    w = SP.ForkWorker(lambda ch: os.kill(os.getpid(), signal.SIGKILL))
    try:
        assert w.recv() is None
    finally:
        w.close()
    w2 = SP.ForkWorker(lambda ch: ch.recv())      # blocks until killed
    w2.close()
    with pytest.raises(ProcessLookupError):
        os.kill(w2.pid, 0)


def test_effects_round_trip_on_a_stand_in_model():
    case = SimpleNamespace(locked=False, spec=None, decided_by="center", history=[1])
    fm = SimpleNamespace(n_solves=3, history=["a"], case=case, taper_restarts=[])
    log, side = [dict(branch="none")], dict(end="e0")
    mark = SP.effects_mark(fm, log, side)
    fm.n_solves += 4
    fm.history.append("b")
    case.history.append(2)
    log.append(dict(branch="axis"))
    side.clear()
    side.update(end="e1")
    eff = SP.effects_since(mark, fm, log, side)
    assert eff["n_solves"] == 4 and eff["history"] == ["b"] and eff["case_history"] == [2]
    assert eff["log"] == [dict(branch="axis")] and eff["side"] == dict(end="e1") and not eff["case_changed"]
    fm2 = SimpleNamespace(n_solves=3, history=["a"], case=SimpleNamespace(history=[1]), taper_restarts=[])
    log2, side2 = [dict(branch="none")], dict(end="e0")
    SP.apply_effects(eff, fm2, log2, side2)
    assert (fm2.n_solves, fm2.history, fm2.case.history, log2, side2) == (7, ["a", "b"], [1, 2], log, dict(end="e1"))
    # unchanged side state is not sent; a changed case lock is flagged
    mark = SP.effects_mark(fm, log, side)
    case.locked = True
    eff = SP.effects_since(mark, fm, log, side)
    assert eff["side"] is None and eff["case_changed"]


def test_options_validation_and_threads():
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=2))
    sp = SP.SpeedSettings(memo=True)
    assert sp.active and not sp.parallel
    assert SP.check_speed(sp, fm) is sp                  # threads matter only for forked workers
    with pytest.raises(ValueError):
        SP.SpeedSettings(memo=1)
    a = SP.SpeedSettings.all_on(4)
    assert (a.column_workers, a.memo, a.mesh_index, a.reuse_stats) == (4, True, True, True)


def test_scope_sets_and_restores_the_process_options():
    from bouquet.kinrecon import diagnostics as D
    from bouquet.kinrecon import forward as FW
    from bouquet.kinrecon.current import diffusion as DF
    before = (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"])
    assert before == (False, False, False)
    with SP.speed_scope(SP.SpeedSettings.all_on(1)):
        assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"]) == (True, True, True)
        with SP.speed_scope(SP.SpeedSettings(memo=True)):
            assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"]) == (True, False)
        assert D.MESH_INDEX["on"]
    with pytest.raises(RuntimeError):
        with SP.speed_scope(SP.SpeedSettings(memo=True)):
            raise RuntimeError("leaves the scope")
    assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"]) == before


@pytest.mark.parametrize("form", ["2", "3"])
def test_profile_memo_returns_the_same_values_and_fresh_arrays(form):
    from bouquet.kinrecon.current import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    spec = (rho, vals) if form == "2" else ((0.0, 0.02), rho, np.vstack((vals, 2 * vals)))
    f = DF.as_profile_function(spec, log=True)
    pts = np.linspace(0.05, 0.95, 37)
    times = [0.0, 0.007, 0.013, 0.02]
    off = [f(t, pts) for t in times]
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        on1 = [f(t, pts) for t in times]
        on1[0][:] = -1.0                                 # a caller's change must not reach the stored value
        on2 = [f(t, pts) for t in times]
        with pytest.raises(ValueError):
            f(0.01, np.array([1.5]))                    # the range check still applies
    assert all(np.array_equal(a, b) for a, b in zip(off, on2))
    assert all(np.array_equal(a, b) for a, b in zip(off[1:], on1[1:]))
