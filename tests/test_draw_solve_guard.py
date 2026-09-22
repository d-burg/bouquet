"""DrawSolveGuard: the draw-loop maxits cap, the recovery of solves that hit it,
and the failed-solve record (solver-free)."""
import inspect
from types import SimpleNamespace

import pytest

from bouquet.TokaMaker_interface import (DRAW_SOLVE_LOOSE_TOL, DRAW_SOLVE_MAXITS,
                                         DRAW_SOLVE_RETRY_URF, DrawSolveGuard,
                                         generate_bouquet)

MAXITS = 'Error in solve: Exceeded "maxits"      '


class FakeGS:
    """TokaMaker stand-in.  Solves numbered in ``stuck`` sit in the period-2
    cycle: at the default urf 0.2 and nl_tol 1e-6 they exceed maxits.
    ``escape`` says what gets them out: "urf" (any other urf), "loose"
    (nl_tol >= 2e-5), "never", or "error" (a non-maxits failure on retry)."""

    def __init__(self, stuck=(), escape="urf"):
        self.settings = SimpleNamespace(maxits=800, urf=0.2, nl_tol=1e-6)
        self.pushed = []
        self.stuck = set(stuck)
        self.escape = escape
        self.calls = 0          # solves the caller asked for
        self.attempts = []      # every solver run, with the settings it saw

    def update_settings(self):
        self.pushed.append((self.settings.maxits, self.settings.urf, self.settings.nl_tol))

    def solve(self, return_its=False):
        st = self.settings
        retry = bool(self.attempts) and self.attempts[-1][0] == self.calls \
            and self.attempts[-1][3] == "fail"
        if not retry:
            self.calls += 1
        n = self.calls
        ok = n not in self.stuck or (
            (self.escape == "urf" and st.urf != 0.2)
            or (self.escape == "loose" and st.nl_tol >= 2e-5))
        if not ok and retry and self.escape == "error":
            self.attempts.append((n, st.urf, st.nl_tol, "error"))
            raise ValueError("Error in solve: lost the axis")
        self.attempts.append((n, st.urf, st.nl_tol, "ok" if ok else "fail"))
        if not ok:
            raise ValueError(MAXITS)
        return (None, 12) if return_its else None


def _call_site(gs):
    return gs.solve()


# ---- the cap ----------------------------------------------------------------
def test_cap_applied_inside_and_restored_after():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100):
        assert gs.settings.maxits == 100
        gs.solve()
    assert gs.settings.maxits == 800
    assert [p[0] for p in gs.pushed] == [100, 800]
    assert "solve" not in vars(gs), "the instance wrapper must be removed"


def test_restored_when_the_block_raises():
    gs = FakeGS()
    with pytest.raises(RuntimeError):
        with DrawSolveGuard(gs, 50):
            raise RuntimeError("boom")
    assert gs.settings.maxits == 800 and "solve" not in vars(gs)


def test_clean_summary_and_bad_arguments():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100) as g:
        gs.solve()
    assert g.summary() == "[draw-solves] 1 solves, none failed (maxits 100, its max 12)"
    for kw in ({"maxits": 0}, {"retry_urf": (0.0,)}, {"retry_urf": (1.5,)},
               {"loose_tol": 0.0}):
        with pytest.raises(ValueError):
            DrawSolveGuard(gs, **{"maxits": 100, **kw})


def test_defaults_match_the_config():
    from bouquet.config import GenerationConfig
    g = GenerationConfig()
    assert DRAW_SOLVE_MAXITS % 2 == 0   # same phase of the period-2 cycle as 800
    assert g.draw_solve_maxits == DRAW_SOLVE_MAXITS
    assert tuple(g.draw_solve_retry_urf) == DRAW_SOLVE_RETRY_URF
    assert g.draw_solve_loose_tol == DRAW_SOLVE_LOOSE_TOL


# ---- recovery ---------------------------------------------------------------
def test_a_different_urf_saves_the_solve_first():
    gs = FakeGS(stuck={1}, escape="urf")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        g.begin_draw(0)
        assert _call_site(gs) is None           # returned, not raised
    assert [a[1:] for a in gs.attempts] == [(0.2, 1e-6, "fail"), (0.1, 1e-6, "ok")]
    assert g.records[0]["recovered_by"] == "urf=0.1"
    assert gs.settings.urf == 0.2 and gs.settings.nl_tol == 1e-6


def test_the_loose_tolerance_is_the_last_resort():
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        gs.solve()
    # both urfs tried at the strict tolerance, then urf 0.2 at the loose one
    assert [a[1:] for a in gs.attempts] == [
        (0.2, 1e-6, "fail"), (0.1, 1e-6, "fail"), (0.3, 1e-6, "fail"), (0.2, 2e-5, "ok")]
    assert g.records[0]["recovered_by"] == "nl_tol=2e-05"
    assert gs.settings.nl_tol == 1e-6


def test_an_unrecoverable_solve_still_raises_with_its_first_error():
    gs = FakeGS(stuck={1}, escape="never")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.attempts) == 4 and g.records[0]["recovered_by"] is None
    assert gs.settings.urf == 0.2 and gs.settings.nl_tol == 1e-6


def test_a_different_failure_during_recovery_stops_it():
    gs = FakeGS(stuck={1}, escape="error")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.attempts) == 2 and g.records[0]["recovered_by"] is None


def test_only_maxits_failures_are_retried():
    class Boom(FakeGS):
        def solve(self, return_its=False):
            self.attempts.append(1)
            raise ValueError("Error in solve: lost the axis")
    gs = Boom()
    with DrawSolveGuard(gs, 100) as g:
        with pytest.raises(ValueError, match="axis"):
            gs.solve()
    assert gs.attempts == [1] and g.records[0]["recovered_by"] is None


def test_recovery_can_be_switched_off():
    gs = FakeGS(stuck={1}, escape="urf")
    with DrawSolveGuard(gs, 100, retry_urf=(), loose_tol=None):
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.attempts) == 1


def test_by_default_a_stuck_solve_goes_straight_to_the_loose_tolerance():
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs) as g:
        assert gs.settings.maxits == DRAW_SOLVE_MAXITS
        gs.solve()
    assert [a[1:] for a in gs.attempts] == [(0.2, 1e-6, "fail"), (0.2, 2e-5, "ok")]
    assert g.records[0]["recovered_by"] == "nl_tol=2e-05"


def test_iterations_are_counted_without_changing_what_the_caller_gets():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100) as g:
        assert gs.solve() is None
        assert gs.solve(return_its=True) == (None, 12)
    assert g.its == [12, 12]


# ---- the record ---------------------------------------------------------------
def test_records_per_draw_and_the_summary():
    gs = FakeGS(stuck={2, 3, 4}, escape="never")
    gs.escape_by_call = {2: "urf", 3: "loose"}
    orig = FakeGS.solve

    def solve(self, return_its=False):
        n = self.calls if (self.attempts and self.attempts[-1][3] == "fail") else self.calls + 1
        self.escape = self.escape_by_call.get(n, "never")
        return orig(self, return_its)
    gs.solve = solve.__get__(gs)
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        g.begin_draw(0)
        _call_site(gs)
        _call_site(gs)                           # saved by urf
        g.begin_draw(1)
        _call_site(gs)                           # saved by the loose tolerance
        with pytest.raises(ValueError):
            _call_site(gs)                       # lost
    assert g.n_solves == 4
    assert [(r["draw"], r["recovered_by"]) for r in g.records] == [
        (0, "urf=0.1"), (1, "nl_tol=2e-05"), (1, None)]
    r = g.failures(1)[1]
    assert r["site"] == "?"   # no bouquet frame on a test-only stack
    assert r["error"] == 'ValueError: Error in solve: Exceeded "maxits"'
    assert r["seconds"] >= 0.0 and r["retry_seconds"] >= 0.0
    assert g.failures(2) == []
    s = g.summary()
    assert s.startswith("[draw-solves] 3/4 solves failed (3 exceeded maxits 100, its max 12)")
    assert "recovered 2 (" in s and "1 by urf=0.1" in s and "1 by nl_tol=2e-05" in s
    assert s.endswith("lost 1, draws [1]: ? x1")


# ---- wiring -------------------------------------------------------------------
def test_generate_bouquet_threads_the_guard_through_the_draw_loop():
    sig = inspect.signature(generate_bouquet)
    assert sig.parameters["solve_guard"].default is None
    src = inspect.getsource(generate_bouquet)
    assert "solve_guard.begin_draw(count)" in src
    assert "diagnostics['solve_failures']" in src


def test_bouquet_generate_enters_the_guard_with_the_config():
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet.generate)
    assert "DrawSolveGuard(self.mygs, gc.draw_solve_maxits," in src
    assert "retry_urf=gc.draw_solve_retry_urf" in src
    assert "loose_tol=gc.draw_solve_loose_tol" in src
    assert "solve_guard=_solve_guard" in src
    assert "print(_solve_guard.summary())" in src
