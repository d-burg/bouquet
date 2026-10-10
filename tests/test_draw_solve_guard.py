"""DrawSolveGuard: the legacy draws' GS iteration cap, the OPT-IN rescue of a
capped draw solve, and the failed-solve record (solver-free).

Owner decision D5 (2026-10-09) on #75: the rescue is opt-in -- (1) off by
default, (2) every rescued draw archives ``recovered_by``, the residual it
reached and the ``nl_tol`` it was accepted at, (3) ``draw_band`` /
``merge_archives`` can exclude rescued draws, (4) this file asserts that
contract (it replaces the guard that recorded the rescue as not approved);
the cap and the rescue act on DRAW solves only, never on the cold baseline
re-solve or the sigma=0 reference solve before the draw loop.
"""
import inspect
import json
import os
import sys
import traceback
import warnings
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


def _call_site(gs, **k):
    return gs.solve(**k)


# ---- the default and the pass-through (restored from 6116d5f) ----------------
def test_the_default_changes_nothing():
    """No cap and no rescue: a pure pass-through -- no settings pushed, the
    caller's arguments passed as given, nothing counted, the wrapper gone."""
    gs = FakeGS()
    calls = []
    orig = gs.solve
    gs.solve = lambda *a, **k: calls.append((a, k)) or orig(**k)
    own = gs.solve
    with DrawSolveGuard(gs, None) as g:
        g.begin_draw(0)
        assert gs.settings.maxits == 800
        assert gs.solve() is None
        gs.solve(return_its=True)
    assert gs.pushed == [] and calls == [((), {}), ((), {"return_its": True})]
    assert g.its == [] and gs.solve is own
    assert g.summary() == ("[draw-solves] 2 solves, none failed (maxits "
                           "solver default)")


def test_an_instance_level_solve_is_put_back():
    gs = FakeGS()
    own = gs.solve
    gs.solve = own                              # an instance attribute
    with DrawSolveGuard(gs):
        assert gs.solve is not own
    assert gs.solve is own


def test_no_solver_is_a_no_op():
    with DrawSolveGuard(None, 50) as g:
        g.begin_draw(0)
    assert g.n_solves == 0


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "50"])
def test_bad_caps_are_refused(bad):
    from bouquet.config import GenerationConfig
    with pytest.raises((ValueError, TypeError)):
        DrawSolveGuard(FakeGS(), bad)
    with pytest.raises(ValueError):
        GenerationConfig(draw_solve_maxits=bad)


@pytest.mark.parametrize("kw", [
    dict(draw_solve_retry_urf=(1.5,)), dict(draw_solve_retry_urf=(0.0,)),
    dict(draw_solve_retry_urf=("0.1",)), dict(draw_solve_retry_urf=(True,)),
    dict(draw_solve_retry_urf="0.1"),
    dict(draw_solve_loose_tol=0.0), dict(draw_solve_loose_tol=-1e-5),
    dict(draw_solve_loose_tol=float("nan")), dict(draw_solve_loose_tol="2e-5"),
    dict(draw_solve_loose_tol=True)])
def test_bad_rescue_settings_are_refused_at_config_time(kw):
    """#75 review B7: refused when the config is built, not minutes later."""
    from bouquet.config import GenerationConfig
    with pytest.raises(ValueError):
        GenerationConfig(reconstruction_engine="legacy", **kw)


# ---- the cap: draws only ------------------------------------------------------
def test_the_cap_applies_from_the_first_draw_and_is_lifted_after():
    gs = FakeGS()
    with DrawSolveGuard(gs, 100) as g:
        assert gs.settings.maxits == 800        # before the draws: setup cap
        gs.solve()
        g.begin_draw(0)
        assert gs.settings.maxits == 100
        assert gs.solve(return_its=True) == (None, 12)
        g.begin_draw(1)
        gs.solve()
    assert gs.settings.maxits == 800
    assert [p[0] for p in gs.pushed] == [100, 800]
    # the tolerance is never touched by the cap
    assert all(a[2] == 1e-6 for a in gs.attempts) and gs.settings.nl_tol == 1e-6
    assert g.its == [12, 12] and "solve" not in vars(gs)
    assert g.summary() == ("[draw-solves] 3 solves, none failed (maxits 100, "
                           "its max 12)")


def test_restored_when_the_block_raises():
    gs = FakeGS()
    with pytest.raises(RuntimeError):
        with DrawSolveGuard(gs, 50) as g:
            g.begin_draw(0)
            raise RuntimeError("boom")
    assert gs.settings.maxits == 800 and "solve" not in vars(gs)


def test_a_pre_draw_solve_is_recorded_but_never_rescued():
    """The cold baseline re-solve / sigma=0 reference solve: a failure there
    is recorded (draw None, no cap) and re-raised as is -- no retry."""
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1,), loose_tol=2e-5) as g:
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.attempts) == 1
    r = g.records[0]
    assert r["draw"] is None and r["maxits"] is None
    assert r["recovered_by"] is None and "attempts" not in r


# ---- the rescue: opt-in, draws only, stamped ----------------------------------
def test_the_rescue_is_opt_in_and_stamped():
    """The contract that replaces test_the_rescue_settings_are_not_ported
    (owner decision D5): (1) off by default everywhere; with it on, (2) a
    rescued draw carries recovered_by, the tolerance it was accepted at and
    the residual bounds, a clean draw says it was not rescued, and with it
    off nothing is stamped at all."""
    from bouquet.config import GenerationConfig
    g0 = GenerationConfig()
    assert tuple(g0.draw_solve_retry_urf) == () == DRAW_SOLVE_RETRY_URF
    assert g0.draw_solve_loose_tol is None and DRAW_SOLVE_LOOSE_TOL is None
    assert g0.draw_solve_maxits == "auto"
    # off: a stuck draw solve fails its draw, one attempt, no stamp
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs) as g:
        g.begin_draw(0)
        assert gs.settings.maxits == DRAW_SOLVE_MAXITS
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert [a[1:] for a in gs.attempts] == [(0.2, 1e-6, "fail")]
    assert not g.rescue_enabled and g.draw_stamp(0) is None
    # on: rescued at the loose tolerance and stamped so
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs, 100, loose_tol=2e-5) as g:
        g.begin_draw(0)
        gs.solve()
        g.begin_draw(1)
        gs.solve()
    st = g.draw_stamp(0)
    assert st["solve_recovered"] is True
    assert st["solve_recovered_by"] == "nl_tol=2e-05"
    assert st["solve_nl_tol_accepted"] == 2e-5 == st["solve_residual_upper"]
    assert st["solve_residual_lower"] == 1e-6 == st["solve_strict_nl_tol"]
    assert st["solve_rescue_its"] == 12
    assert g.draw_stamp(1) == {"solve_recovered": False}
    assert gs.settings.nl_tol == 1e-6           # restored after the rescue


def test_a_different_urf_saves_the_solve_first():
    gs = FakeGS(stuck={1}, escape="urf")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        g.begin_draw(0)
        assert _call_site(gs) is None           # returned, not raised
    assert [a[1:] for a in gs.attempts] == [(0.2, 1e-6, "fail"), (0.1, 1e-6, "ok")]
    r = g.records[0]
    assert r["recovered_by"] == "urf=0.1"
    # a urf rescue meets the solver's own criterion
    assert r["nl_tol_accepted"] == 1e-6 and r["residual_lower"] is None
    assert r["attempts"] == [dict(step="urf=0.1", its=12, error=None)]
    assert gs.settings.urf == 0.2 and gs.settings.nl_tol == 1e-6
    assert g.draw_stamp(0)["solve_nl_tol_accepted"] == 1e-6


def test_the_loose_tolerance_is_the_last_resort():
    gs = FakeGS(stuck={1}, escape="loose")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3), loose_tol=2e-5) as g:
        g.begin_draw(0)
        gs.solve()
    # both urfs tried at the strict tolerance, then urf 0.2 at the loose one
    assert [a[1:] for a in gs.attempts] == [
        (0.2, 1e-6, "fail"), (0.1, 1e-6, "fail"), (0.3, 1e-6, "fail"), (0.2, 2e-5, "ok")]
    assert g.records[0]["recovered_by"] == "nl_tol=2e-05"
    assert [a["step"] for a in g.records[0]["attempts"]] == [
        "urf=0.1", "urf=0.3", "nl_tol=2e-05"]
    assert gs.settings.nl_tol == 1e-6


def test_an_unrecoverable_solve_still_raises_with_its_first_error():
    gs = FakeGS(stuck={1}, escape="never")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3), loose_tol=2e-5) as g:
        g.begin_draw(0)
        with pytest.raises(ValueError, match="maxits"):
            gs.solve()
    assert len(gs.attempts) == 4 and g.records[0]["recovered_by"] is None
    assert gs.settings.urf == 0.2 and gs.settings.nl_tol == 1e-6
    assert g.draw_stamp(0) == {"solve_recovered": False}


def test_a_different_failure_during_recovery_surfaces():
    """#75 review B9: not hidden behind the original cap error."""
    gs = FakeGS(stuck={1}, escape="error")
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3)) as g:
        g.begin_draw(0)
        with pytest.raises(ValueError, match="axis") as ei:
            gs.solve()
    assert "maxits" in str(ei.value.__cause__)
    r = g.records[0]
    assert len(gs.attempts) == 2 and r["recovered_by"] is None
    assert r["retry_error"] == "ValueError: Error in solve: lost the axis"


def test_only_maxits_failures_are_retried():
    class Boom(FakeGS):
        def solve(self, return_its=False):
            self.attempts.append(1)
            raise ValueError("Error in solve: lost the axis")
    gs = Boom()
    with DrawSolveGuard(gs, 100, loose_tol=2e-5) as g:
        g.begin_draw(0)
        with pytest.raises(ValueError, match="axis"):
            gs.solve()
    assert gs.attempts == [1] and g.records[0]["recovered_by"] is None


def test_records_per_draw_and_the_summary():
    gs = FakeGS(stuck={2, 3, 4}, escape="never")
    gs.escape_by_call = {2: "urf", 3: "loose"}
    orig = FakeGS.solve

    def solve(self, return_its=False):
        n = self.calls if (self.attempts and self.attempts[-1][3] == "fail") else self.calls + 1
        self.escape = self.escape_by_call.get(n, "never")
        return orig(self, return_its)
    gs.solve = solve.__get__(gs)
    with DrawSolveGuard(gs, 100, retry_urf=(0.1, 0.3), loose_tol=2e-5) as g:
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


# ---- generate_bouquet: the pre-draw solves keep the setup cap -----------------
def _solve_line(fn_src, after):
    """0-based index (within the function source) of the first
    ``mygs.solve()`` at or after the line holding *after*."""
    lines = fn_src.splitlines()
    i = next(k for k, ln in enumerate(lines) if after in ln)
    return next(k for k in range(i, len(lines))
                if lines[k].strip() == "mygs.solve()")


def test_generate_keeps_the_setup_cap_for_the_baseline_and_sigma0_solves(
        tmp_path, monkeypatch):
    """#75 review B3: the cold jphi-linterp baseline re-solve and the sigma=0
    (jBS-delta) reference anchor solve run before the first draw -- at the
    solver's setup cap and never rescued; the draw solves run capped."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_draw_rejections as R
    import bouquet.TokaMaker_interface as TI
    lines, first = inspect.getsourcelines(TI.generate_bouquet)
    src = "".join(lines)
    want = {first + _solve_line(src, "set_profiles(pp_prof=_pp_b"):
            "baseline re-solve",
            first + _solve_line(src, "set_profiles(pp_prof=_cache_pp"):
            "sigma=0 reference anchor"}

    class GS(R._FakeGS):
        def __init__(self):
            super().__init__()
            self.settings = SimpleNamespace(maxits=800, urf=0.2, nl_tol=1e-6)
            self.log = []

        def update_settings(self):
            pass

        def solve(self, return_its=False):
            fr = [f for f in traceback.extract_stack()
                  if f.filename == TI.__file__
                  and f.name == "generate_bouquet"]
            self.log.append((self.settings.maxits,
                             fr[-1].lineno if fr else None))
            return (None, 10) if return_its else None

    gs = GS()
    monkeypatch.setattr(R, "_FakeGS", lambda: gs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with DrawSolveGuard(gs, 100, retry_urf=(0.1,), loose_tol=2e-5) as g:
            R._run(tmp_path, monkeypatch, ["ok", "ok"], solve_guard=g,
                   jbs_delta_mode=True)
    pre = {ln: m for m, ln in gs.log if ln in want}
    assert set(pre) == set(want), (gs.log, want)      # both solves ran ...
    assert set(pre.values()) == {800}                 # ... uncapped
    drawn = [m for m, ln in gs.log if ln not in want]
    assert drawn and set(drawn) == {100}              # the draws: capped
    assert gs.settings.maxits == 800                  # lifted after


def test_generate_bouquet_threads_the_guard_through_the_draw_loop():
    sig = inspect.signature(generate_bouquet)
    assert sig.parameters["solve_guard"].default is None
    src = inspect.getsource(generate_bouquet)
    assert "solve_guard.begin_draw(count)" in src
    assert "diagnostics['solve_failures']" in src
    assert "solve_guard.draw_stamp(count)" in src


def test_the_draw_methods_own_cap_and_rescue():
    """legacy: the config's; swb: its higher cap, the SAME rescue (stamped
    like a legacy draw); the engine: neither."""
    from bouquet.draw_methods import DrawMethod
    from bouquet.engine_draws import GenerateEngineDraws
    from bouquet.swb_draws import SWB_DRAW_MAXITS, SwbDraws
    assert DrawMethod().solve_maxits(50) == 50
    assert SwbDraws(None, [1.0], [0.0]).solve_maxits(50) == SWB_DRAW_MAXITS
    assert SwbDraws(None, [1.0], [0.0]).solve_maxits(None) == SWB_DRAW_MAXITS
    assert DrawMethod().solve_retry_urf((0.5,)) == (0.5,)
    assert DrawMethod().solve_loose_tol(1e-6) == 1e-6
    assert SwbDraws(None, [1.0], [0.0]).solve_loose_tol(2e-5) == 2e-5
    e = GenerateEngineDraws.__new__(GenerateEngineDraws)
    assert e.solve_maxits(50) is None
    assert e.solve_retry_urf((0.5,)) == () and e.solve_loose_tol(2e-5) is None


# ---- engine-dependent default, engine refusal, stored configs ----------------
def _cfg(eng, **g):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
        solver=SolverConfig(mesh_path="m"), output_header="h",
        generation=GenerationConfig(reconstruction_engine=eng, **g))


def test_the_cap_is_resolved_per_engine_and_recorded():
    from bouquet.engine import resolve_engine_defaults
    c = _cfg("legacy")
    rec = resolve_engine_defaults(c)
    assert c.generation.draw_solve_maxits == 100 == DRAW_SOLVE_MAXITS
    assert rec["draw_solve_maxits"] == {"value": 100,
                                        "origin": "resolved from engine=legacy"}
    c = _cfg("unified")
    rec = resolve_engine_defaults(c)
    assert c.generation.draw_solve_maxits is None
    assert rec["draw_solve_maxits"]["origin"] == "resolved from engine=unified"
    # switching after a resolution re-resolves (never taken for explicit)
    c.generation.reconstruction_engine = "legacy"
    resolve_engine_defaults(c)
    assert c.generation.draw_solve_maxits == 100
    # an explicit legacy None (the setup cap) is KEPT
    c = _cfg("legacy", draw_solve_maxits=None)
    with pytest.warns(UserWarning, match="draw_solve_maxits=None"):
        rec = resolve_engine_defaults(c)
    assert c.generation.draw_solve_maxits is None
    assert rec["draw_solve_maxits"]["origin"] == "explicit"


@pytest.mark.parametrize("kw", [dict(draw_solve_maxits=100),
                                dict(draw_solve_retry_urf=(0.1,)),
                                dict(draw_solve_loose_tol=2e-5)])
def test_the_engine_refuses_the_legacy_cap_and_the_rescue(kw):
    """#75 review B4/B5: refused by the standard unread-settings rule."""
    from bouquet.engine import validate_engine_settings
    with pytest.raises(ValueError, match="never reads"):
        _cfg("unified", **kw)
    c = _cfg("legacy", **kw)                    # legacy: read, accepted
    c.generation.reconstruction_engine = "unified"
    with pytest.raises(ValueError, match="never reads"):
        validate_engine_settings(c.generation)


def test_the_engine_accepts_the_cap_unset():
    _cfg("unified")
    _cfg("unified", draw_solve_maxits=None)
    _cfg("unified", draw_solve_maxits="auto")


def test_a_stored_config_without_the_field_replays_uncapped():
    """#75 review B8: before #66 the draws ran under the setup cap; a stored
    config without the field (or with null) replays so, not at 100."""
    from bouquet.config import BouquetConfig
    d = _cfg("legacy").to_dict()
    assert d["generation"]["draw_solve_maxits"] == "auto"
    assert BouquetConfig.from_dict(json.loads(json.dumps(d))) \
        .generation.draw_solve_maxits == "auto"           # round trip
    d2 = json.loads(json.dumps(d))
    del d2["generation"]["draw_solve_maxits"]
    with pytest.warns(UserWarning, match="predates the field"):
        g = BouquetConfig.from_dict(d2).generation
    assert g.draw_solve_maxits is None
    d3 = json.loads(json.dumps(d))
    d3["generation"]["draw_solve_maxits"] = None
    assert BouquetConfig.from_dict(d3).generation.draw_solve_maxits is None
