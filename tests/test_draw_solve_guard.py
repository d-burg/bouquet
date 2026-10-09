"""DrawSolveGuard: the optional draw-loop GS iteration cap and the
failed-solve record (solver-free).

Only the cap and its record are here: a solve that hits the cap fails the draw
exactly as before -- there is no re-solve at a looser tolerance or another
under-relaxation.  The default (``draw_solve_maxits=None``) keeps the
solver's own cap and passes every call through untouched.
"""
import inspect
from types import SimpleNamespace

import pytest

from bouquet.TokaMaker_interface import DrawSolveGuard, generate_bouquet

MAXITS = 'Error in solve: Exceeded "maxits"      '


class FakeGS:
    """TokaMaker stand-in.  Solves numbered in ``stuck`` exceed maxits;
    ``lost`` ones fail for another reason."""

    def __init__(self, stuck=(), lost=()):
        self.settings = SimpleNamespace(maxits=800, urf=0.2, nl_tol=1e-6)
        self.pushed = []
        self.stuck = set(stuck)
        self.lost = set(lost)
        self.calls = []

    def update_settings(self):
        self.pushed.append((self.settings.maxits, self.settings.urf,
                            self.settings.nl_tol))

    def solve(self, *a, **k):
        self.calls.append((a, dict(k), self.settings.maxits,
                           self.settings.nl_tol))
        n = len(self.calls)
        if n in self.stuck:
            raise ValueError(MAXITS)
        if n in self.lost:
            raise ValueError("Error in solve: lost the axis")
        return (None, 12) if k.get("return_its") else None


def _call_site(gs, **k):
    return gs.solve(**k)


# ---- the default: the solver's cap, calls untouched ---------------------------
def test_the_default_changes_nothing():
    from bouquet.config import GenerationConfig
    assert GenerationConfig().draw_solve_maxits is None
    gs = FakeGS()
    with DrawSolveGuard(gs) as g:
        assert gs.settings.maxits == 800
        assert gs.solve() is None
        gs.solve(True, foo=1)
    # no settings pushed, the caller's arguments passed through as given
    assert gs.pushed == []
    assert gs.calls[0][:2] == ((), {}) and gs.calls[1][:2] == ((True,),
                                                               {"foo": 1})
    assert g.its == [] and "solve" not in vars(gs)
    assert g.summary() == ("[draw-solves] 2 solves, none failed (maxits "
                           "solver default)")


def test_a_failure_is_recorded_and_re_raised_unchanged():
    gs = FakeGS(stuck={2}, lost={3})
    with DrawSolveGuard(gs) as g:
        g.begin_draw(0)
        _call_site(gs)
        with pytest.raises(ValueError, match="maxits"):
            _call_site(gs)
        g.begin_draw(1)
        with pytest.raises(ValueError, match="axis"):
            _call_site(gs)
    # nothing was re-solved: one solver run per call
    assert len(gs.calls) == 3
    assert [(r["draw"], r["exceeded_maxits"]) for r in g.records] == [
        (0, True), (1, False)]
    r = g.failures(0)[0]
    assert r["error"] == 'ValueError: Error in solve: Exceeded "maxits"'
    assert r["site"] == "?" and r["seconds"] >= 0.0 and r["maxits"] is None
    assert g.failures(2) == []
    s = g.summary()
    assert s.startswith("[draw-solves] 2/3 solves failed (1 exceeded maxits "
                        "solver default)")
    assert "draws [0, 1]" in s and s.endswith(": ? x2")


# ---- the cap, when set --------------------------------------------------------
def test_cap_applied_inside_and_restored_after():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100) as g:
        assert gs.settings.maxits == 100
        gs.solve()
        assert gs.solve(return_its=True) == (None, 12)
    assert gs.settings.maxits == 800
    assert [p[0] for p in gs.pushed] == [100, 800]
    # the tolerance is never touched
    assert all(c[3] == 1e-6 for c in gs.calls) and gs.settings.nl_tol == 1e-6
    assert g.its == [12, 12] and "solve" not in vars(gs)
    assert g.summary() == ("[draw-solves] 2 solves, none failed (maxits 100, "
                           "its max 12)")


def test_a_capped_solve_fails_the_draw_as_before_no_rescue():
    gs = FakeGS(stuck={1})
    with DrawSolveGuard(gs, 50) as g:
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.calls) == 1                  # no retry of any kind
    assert g.records[0]["exceeded_maxits"] and g.records[0]["maxits"] == 50
    assert not any(k in g.records[0] for k in ("recovered_by",
                                               "retry_seconds"))


def test_restored_when_the_block_raises():
    gs = FakeGS()
    with pytest.raises(RuntimeError):
        with DrawSolveGuard(gs, 50):
            raise RuntimeError("boom")
    assert gs.settings.maxits == 800 and "solve" not in vars(gs)


def test_an_instance_level_solve_is_put_back():
    gs = FakeGS()
    own = gs.solve
    gs.solve = own                              # an instance attribute
    with DrawSolveGuard(gs):
        assert gs.solve is not own
    assert gs.solve is own


def test_no_solver_is_a_no_op():
    with DrawSolveGuard(None, 50) as g:
        pass
    assert g.n_solves == 0


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "50"])
def test_bad_caps_are_refused(bad):
    from bouquet.config import GenerationConfig
    with pytest.raises((ValueError, TypeError)):
        DrawSolveGuard(FakeGS(), bad)
    with pytest.raises(ValueError):
        GenerationConfig(draw_solve_maxits=bad)


def test_the_rescue_settings_are_not_ported():
    """The looser-tolerance / other-urf re-solve is NOT approved: neither
    its config fields nor its constants exist."""
    import bouquet.TokaMaker_interface as TI
    from bouquet.config import GenerationConfig
    g = GenerationConfig()
    assert not hasattr(g, "draw_solve_loose_tol")
    assert not hasattr(g, "draw_solve_retry_urf")
    assert not hasattr(TI, "DRAW_SOLVE_LOOSE_TOL")
    assert "nl_tol" not in inspect.getsource(DrawSolveGuard)


# ---- wiring ---------------------------------------------------------------------
def test_generate_bouquet_threads_the_guard_through_the_draw_loop():
    sig = inspect.signature(generate_bouquet)
    assert sig.parameters["solve_guard"].default is None
    src = inspect.getsource(generate_bouquet)
    assert "solve_guard.begin_draw(count)" in src
    assert "diagnostics['solve_failures']" in src


def test_bouquet_generate_enters_the_guard_with_the_config():
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet.generate)
    # the draw method's cap: legacy the config's, swb a higher one, engine none
    assert "_m.solve_maxits(gc.draw_solve_maxits)" in src
    from bouquet.draw_methods import DrawMethod
    from bouquet.swb_draws import SWB_DRAW_MAXITS, SwbDraws
    assert DrawMethod().solve_maxits(50) == 50
    assert SwbDraws(None, [1.0], [0.0]).solve_maxits(None) == SWB_DRAW_MAXITS
    assert "solve_guard=_solve_guard" in src
    assert "print(_solve_guard.summary())" in src
