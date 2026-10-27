"""The homotopy branch guard, a diagnostic for bistable equilibria (#87, opt-in
``GenerationConfig.guard_bistability``): the detector and the continuation
on synthetic iterates, the stage's objective at s, the guard on a stand-in
solver, the setting's validation and the per-draw / per-scan stamps on the
toy engine.  The live-solver half is
``tests/test_homotopy_guard_solver.py``."""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bouquet import homotopy_guard as H  # noqa: E402

#: a reconstruction-like minor radius [m]; the threshold is 1 % of it (6 mm)
A = 0.6
R0 = 1.7


# ---------------------------------------------------------------------------
#  the detector
# ---------------------------------------------------------------------------
def test_the_threshold_is_the_branch_discriminator():
    from bouquet.engine import AXIS_BRANCH_FRAC, MSE_FD_BRANCH_AXIS_FRAC
    assert H.GUARD_AXIS_FRAC == AXIS_BRANCH_FRAC == MSE_FD_BRANCH_AXIS_FRAC
    # three halvings: 1/2, 1/4, 1/8
    assert H.GUARD_FIRST_STEP / 4 == H.GUARD_MIN_STEP == 0.125


@pytest.mark.parametrize("dz_mm, fires", [(0.3, False), (5.9, False),
                                          (6.1, True), (17.0, True)])
def test_the_detector_at_the_threshold(dz_mm, fires):
    d = H.step_displacement_frac((R0, 0.0), (R0, 1e-3 * dz_mm), A)
    assert d == pytest.approx(1e-3 * dz_mm / A)
    assert H.jumped(d) is fires


def test_an_unknown_axis_or_radius_never_fires():
    assert H.step_displacement_frac(None, (R0, 0.0), A) is None
    assert H.step_displacement_frac((R0, 0.0), None, A) is None
    assert H.step_displacement_frac((R0, 0.0), (R0, 0.1), None) is None
    assert H.step_displacement_frac((R0, 0.0), (R0, 0.1), 0.0) is None
    assert H.jumped(None) is False


# ---------------------------------------------------------------------------
#  the continuation on synthetic iterates
# ---------------------------------------------------------------------------
class _Branches:
    """Synthetic solves of a stage parametrised by s: branch A moves the
    axis ``slope_mm`` per unit s and exists for ``s <= s_fold``; a step
    longer than ``basin`` (or beyond the fold) lands on branch B at
    ``z_b_mm``; a state on B stays on B.  ``fail_at``: a solve to that s
    raises."""

    def __init__(self, slope_mm=1.0, basin=0.3, s_fold=2.0, z_b_mm=-17.0,
                 fail_at=None):
        self.slope, self.basin, self.fold = slope_mm, basin, s_fold
        self.zb, self.fail_at = z_b_mm, fail_at
        self.calls = []

    def solve_from(self, state, s):
        self.calls.append((state["s"], s))
        if self.fail_at is not None and abs(s - self.fail_at) < 1e-12:
            raise RuntimeError("maxits")
        on_b = state["branch"] == "B"
        if on_b or s > self.fold or (s - state["s"]) > self.basin + 1e-12:
            st = dict(s=s, branch="B", z=1e-3 * self.zb)
        else:
            st = dict(s=s, branch="A", z=1e-3 * self.slope * s)
        return st, (R0, st["z"])


_START = dict(s=0.0, branch="A", z=0.0)


def _full_step(m):
    st, ax = m.solve_from(dict(_START), 1.0)
    m.calls.clear()
    return st, ax


def test_branch_hopping_is_held_on_the_branch():
    m = _Branches()
    _, jump_ax = _full_step(m)
    assert H.jumped(H.step_displacement_frac((R0, 0.0), jump_ax, A))
    w = H.continue_in_halves(m.solve_from, dict(_START), (R0, 0.0), A)
    assert w["outcome"] == "completed" and w["s_reached"] == 1.0
    assert w["state"]["branch"] == "A"
    # 1/2 rejected, then 1/4 steps accepted to the end
    assert [round(x["s"], 3) for x in w["steps"]] == [0.5, 0.25, 0.5, 0.75,
                                                      1.0]
    assert [x["accepted"] for x in w["steps"]] == [False, True, True, True,
                                                   True]
    act, sep = H.classify(w, jump_ax, A)
    assert act == "held" and sep == pytest.approx(18e-3 / A)


def test_a_continuous_response_is_legitimate():
    """A large but continuous response (20 mm over the stage): every
    sub-step moves in proportion to the step, the walk ends where the
    jump landed."""
    m = _Branches(slope_mm=20.0, basin=10.0)
    _, jump_ax = _full_step(m)
    w = H.continue_in_halves(m.solve_from, dict(_START), (R0, 0.0), A)
    assert w["outcome"] == "completed"
    assert w["n_solves"] == 5      # 1/2 (10 mm) rejected, 4 x 1/4 (5 mm)
    act, sep = H.classify(w, jump_ax, A)
    assert act == "legitimate" and sep == pytest.approx(0.0, abs=1e-12)


def test_a_jump_that_persists_at_an_eighth_is_a_crossing():
    """Branch A ends at s = 0.6 (a fold): the walk reaches 0.5 and every
    step beyond, down to 1/8, jumps."""
    m = _Branches(s_fold=0.6)
    _, jump_ax = _full_step(m)
    w = H.continue_in_halves(m.solve_from, dict(_START), (R0, 0.0), A)
    assert w["outcome"] == "persistent" and w["s_reached"] == 0.5
    rej = [x for x in w["steps"] if not x["accepted"]]
    assert [round(x["s"] - x["s_from"], 4) for x in rej][-1] == 0.125
    assert H.classify(w, jump_ax, A) == ("crossed", None)
    # bounded: at most log2(1/2 / 1/8) + 1/(1/8) solves
    assert w["n_solves"] <= 2 + 8 + 1


def test_a_substep_that_does_not_solve_is_backed_off():
    """A sub-step that raises is retried at half the step, like a jump:
    here s = 0.5 jumps and s = 0.25 fails, s = 0.125 solves, and the walk
    goes on in eighths (the step never grows again)."""
    class _Once(_Branches):
        def solve_from(self, state, s):
            if state["s"] == 0.0 and abs(s - 0.25) < 1e-12:
                raise RuntimeError("maxits")
            return super().solve_from(state, s)
    m = _Once()
    w = H.continue_in_halves(m.solve_from, dict(_START), (R0, 0.0), A)
    assert w["outcome"] == "completed" and w["state"]["branch"] == "A"
    assert "maxits" in w["steps"][1]["error"]
    assert [round(x["s"], 3) for x in w["steps"][:3]] == [0.5, 0.25, 0.125]


def test_a_substep_that_never_solves_is_failed():
    class _Never(_Branches):
        def solve_from(self, state, s):
            if s - state["s"] < 0.6:
                raise RuntimeError("maxits")
            return super().solve_from(state, s)
    w = H.continue_in_halves(_Never().solve_from, dict(_START), (R0, 0.0), A)
    assert w["outcome"] == "failed" and w["s_reached"] == 0.0
    assert all("maxits" in x["error"] for x in w["steps"])
    assert [round(x["s"], 3) for x in w["steps"]] == [0.5, 0.25, 0.125]
    assert H.classify(w, (R0, -0.017), A) == ("failed", None)


def test_the_summary_action_order():
    assert H.summary_action([]) == "none"
    assert H.summary_action(["none", "legitimate"]) == "legitimate"
    assert H.summary_action(["held", "legitimate"]) == "held"
    assert H.summary_action(["held", "crossed"]) == "crossed"
    assert H.summary_action(["failed", "held"]) == "failed"


# ---------------------------------------------------------------------------
#  the stage's objective at s
# ---------------------------------------------------------------------------
def _term(coffs, target=0.0, weight=1.0):
    import collections
    T = collections.namedtuple("coil_reg_term", ["coffs", "target", "weight"])
    return T(dict(coffs), float(target), float(weight))


def test_the_regularisation_walks_the_coil_targets():
    strong = [_term({"F1A": 1.0}, 2.1e4, 1.0e4),
              _term({"F9A": 1.0, "F9B": -1.0}, 0.0, 1.0e4),
              _term({"#VSC": 1.0}, 0.0, 1.0)]
    cur = {"F1A": 2.0e4, "F9A": 3.0e3, "F9B": -2.0e3}
    assert H.term_value({"F9A": 1.0, "F9B": -1.0}, cur) == 5.0e3
    assert H.term_value({"#VSC": 1.0}, cur) is None
    at0 = H.blend_reg(_term, strong, cur, 0.0)
    assert [t.target for t in at0] == [2.0e4, 5.0e3, 0.0]
    half = H.blend_reg(_term, strong, cur, 0.5)
    assert [t.target for t in half] == pytest.approx([2.05e4, 2.5e3, 0.0])
    # weights never change; s = 1 is exactly the strong terms
    assert [t.weight for t in half] == [1.0e4, 1.0e4, 1.0]
    at1 = H.blend_reg(_term, strong, cur, 1.0)
    assert [(t.coffs, t.target, t.weight) for t in at1] == \
        [(t.coffs, t.target, t.weight) for t in strong]
    # plain 3-tuples (the stand-in solver's terms) are read too
    tup = H.blend_reg(lambda c, target, weight: (c, target, weight),
                      [(("F1A",), 1.0, 1e4)], {"F1A": 3.0}, 0.5)
    assert tup == [(("F1A",), 2.0, 1e4)]


def test_the_bounds_end_exactly_on_the_stage_and_start_containing():
    target = {"F1A": [9.0, 11.0], "#VSC": [-2.0, 2.0]}
    prev = H.containing_bounds(target, {"F1A": 13.0, "#VSC": 5.0})
    assert prev == {"F1A": [9.0, 13.0], "#VSC": [-5.0, 5.0]}
    # a value inside leaves the interval alone
    assert H.containing_bounds(target, {"F1A": 10.0})["F1A"] == [9.0, 11.0]
    assert H.blend_bounds(prev, target, 1.0) == target
    assert H.blend_bounds(prev, target, 0.5) == {"F1A": [9.0, 12.0],
                                                 "#VSC": [-3.5, 3.5]}
    # the previous stage's interval, linearly
    p2 = {"F1A": [5.0, 15.0]}
    assert H.blend_bounds(p2, target, 0.25)["F1A"] == [6.0, 14.0]
    # a channel missing from the previous stage takes the target's
    assert H.blend_bounds(p2, target, 0.25)["#VSC"] == [-2.0, 2.0]


# ---------------------------------------------------------------------------
#  the guard on a stand-in solver
# ---------------------------------------------------------------------------
class _Eq:
    def __init__(self, s, branch, z):
        self.s, self.branch, self.z = s, branch, z


class _GS:
    """A solver whose stage parameter is read off the installed F1A bound:
    previous interval [0, 10] (s = 0) to the stage's [0, 0] (s = 1)."""

    def __init__(self, model):
        self.m = model
        self.eq = _Eq(0.0, "A", 0.0)
        self.bounds = None
        self.reg = None
        self.n_solve = self.n_copy = self.n_replace = 0

    @property
    def o_point(self):
        return [R0, self.eq.z]

    def copy_eq(self):
        self.n_copy += 1
        return _Eq(self.eq.s, self.eq.branch, self.eq.z)

    def replace_eq(self, source_eq=None):
        self.n_replace += 1
        self.eq = _Eq(source_eq.s, source_eq.branch, source_eq.z)

    def set_coil_bounds(self, b):
        self.bounds = b

    def set_coil_reg(self, reg_terms=None):
        self.reg = reg_terms

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        return _term(coffs, target, weight)

    def get_coil_currents(self):
        return {"F1A": 5.0}, None

    def solve(self):
        self.n_solve += 1
        s = 1.0 - float(self.bounds["F1A"][1]) / 10.0
        st, _ = self.m.solve_from(dict(s=self.eq.s, branch=self.eq.branch,
                                       z=self.eq.z), s)
        self.eq = _Eq(st["s"], st["branch"], st["z"])


_STAGE = {"F1A": [0.0, 0.0]}
_PREV = {"F1A": [0.0, 10.0]}


def _guard():
    return H.HomotopyBranchGuard(
        reference_axis=dict(R=R0, Z=0.0, minor_radius_m=A))


def _stage(model):
    gs = _GS(model)
    gs.set_coil_bounds(_STAGE)
    g = _guard()
    st = g.solve_stage(gs, 1, "homotopy pass 2/3", _STAGE, prev_bounds=_PREV)
    return gs, g, st


def test_no_jump_reads_only():
    gs, g, st = _stage(_Branches(slope_mm=1.0, basin=10.0))
    assert (gs.n_solve, gs.n_copy, gs.n_replace) == (1, 1, 0)
    assert st["fired"] is False and st["action"] == "none"
    assert g.record["fired"] is False and g.record["path"] == "continuous"
    assert gs.bounds == _STAGE
    assert gs.eq.z == pytest.approx(1e-3)


def test_branch_hopping_leaves_the_draw_on_its_branch():
    gs, g, st = _stage(_Branches())
    assert st["fired"] is True and st["action"] == "held"
    assert gs.eq.branch == "A" and gs.eq.z == pytest.approx(1e-3)
    assert gs.bounds == _STAGE            # the stage's own objective
    assert st["n_extra_solves"] == 5 and gs.n_solve == 6
    assert g.record["action"] == "held" and g.record["path"] == "continuous"
    assert st["final_step_displacement_frac_a"] == pytest.approx(1e-3 / A)


@pytest.mark.parametrize("model, action, path", [
    (_Branches(slope_mm=20.0, basin=10.0), "legitimate", "continuous"),
    (_Branches(s_fold=0.6), "crossed", "crossed"),
    (_Branches(fail_at=0.125, basin=0.0), "failed", "continuous")])
def test_otherwise_the_jumped_state_is_kept(model, action, path):
    gs, g, st = _stage(model)
    assert st["fired"] is True and st["action"] == action
    assert g.record["path"] == path
    # exactly the unguarded stage's (jumped) state
    z_jump = (20e-3 if action == "legitimate" else -17e-3)
    assert gs.eq.z == pytest.approx(z_jump)
    assert gs.bounds == _STAGE


def test_the_first_stage_blends_the_regularisation_and_records_the_entry():
    strong = [_term({"F1A": 1.0}, 5.0, 1e4)]
    gs = _GS(_Branches())
    gs.set_coil_bounds(_STAGE)
    g = H.HomotopyBranchGuard(
        reference_axis=dict(R=R0, Z=0.0, minor_radius_m=A),
        strong_reg_terms=strong)
    st = g.solve_stage(gs, 0, "homotopy pass 1/3", _STAGE,
                       prev_bounds=_PREV, first=True)
    assert st["reg_blended"] is True and st["action"] == "held"
    # the stage ends on exactly the strong terms
    assert [(t.target, t.weight) for t in gs.reg] == [(5.0, 1e4)]
    assert g.record["entry_axis"]["displacement_mm"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  review of #102: the wall budget (F1) and failures inside the guard (F2)
# ---------------------------------------------------------------------------
class _FakeClock:
    """``perf_counter`` the solver double advances (as in
    tests/test_draw_wall_budget.py); everything else is the real module."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        import time
        return getattr(time, name)


class _TimedGS(_GS):
    """The stand-in solver; every solve takes *dt* of the fake clock."""

    class settings:
        maxits = 100

    def __init__(self, model, clock, dt):
        super().__init__(model)
        self.clock, self.dt = clock, dt

    def solve(self):
        self.clock.now += self.dt
        super().solve()

    def update_settings(self):
        pass


def _budgeted(monkeypatch, model, budget_s, dt, elapsed=0.0):
    import bouquet.TokaMaker_interface as TI
    clock = _FakeClock()
    monkeypatch.setattr(TI, "time", clock)
    gs = _TimedGS(model, clock, dt)
    gs.set_coil_bounds(_STAGE)
    sg = TI.DrawSolveGuard(gs, maxits=None, wall_budget_s=budget_s)
    sg.__enter__()
    sg.begin_draw(0)
    clock.now += elapsed
    g = H.HomotopyBranchGuard(
        reference_axis=dict(R=R0, Z=0.0, minor_radius_m=A), budget=sg)
    return gs, sg, g, clock


def test_a_firing_guard_does_not_consume_the_wall_budget(monkeypatch):
    """F1: the stage's own solve is charged (0.3 s); the continuation's five
    sub-steps (1.5 s more on the clock) are not, so a 1.0 s budget is still
    unspent and the next solve starts."""
    gs, sg, g, clock = _budgeted(monkeypatch, _Branches(), 1.0, 0.3)
    st = g.solve_stage(gs, 1, "homotopy pass 2/3", _STAGE, prev_bounds=_PREV)
    assert st["action"] == "held" and st["n_extra_solves"] == 5
    assert gs.n_solve == 6 and clock.now == pytest.approx(1.8)
    assert not sg.check_wall_budget() and 0 not in sg.over_budget
    gs.solve()                                    # charged: 0.6 s of 1.0
    assert not sg.check_wall_budget()
    sg.__exit__(None, None, None)


def test_a_spent_budget_runs_no_continuation(monkeypatch):
    """F1: when the stage's own solve spends the budget, the guard does not
    walk; the draw goes on as unguarded (the next solve is refused by the
    budget, exactly as without the guard)."""
    from bouquet.TokaMaker_interface import DrawWallBudgetExceeded
    gs, sg, g, clock = _budgeted(monkeypatch, _Branches(), 1.0, 0.3,
                                 elapsed=0.9)
    st = g.solve_stage(gs, 1, "homotopy pass 2/3", _STAGE, prev_bounds=_PREV)
    assert st["action"] == "failed" and "wall budget" in st["reason"]
    assert gs.n_solve == 1 and gs.eq.branch == "B"   # the jumped state
    with pytest.raises(DrawWallBudgetExceeded):
        gs.solve()
    sg.__exit__(None, None, None)


def test_a_budget_overrun_inside_the_walk_propagates():
    """F1: the draw's wall-budget exception is never stamped as a sub-step
    that did not solve."""
    from bouquet.TokaMaker_interface import DrawWallBudgetExceeded

    def solve_from(state, s):
        raise DrawWallBudgetExceeded("draw 0: budget spent")
    with pytest.raises(DrawWallBudgetExceeded):
        H.continue_in_halves(solve_from, dict(_START), (R0, 0.0), A)


def test_a_programming_error_in_the_walk_propagates():
    def solve_from(state, s):
        raise KeyError("a bug")
    with pytest.raises(KeyError):
        H.continue_in_halves(solve_from, dict(_START), (R0, 0.0), A)


class _FailingGS(_GS):
    """The stand-in solver whose n-th ``replace_eq`` / ``copy_eq`` raises
    the toolkit's plain ``Exception``."""

    def __init__(self, model, *, fail_replace=None, fail_copy=None,
                 exc=Exception):
        super().__init__(model)
        self.fail_replace, self.fail_copy, self.exc = (fail_replace,
                                                       fail_copy, exc)

    def copy_eq(self):
        if self.fail_copy is not None and self.n_copy + 1 == self.fail_copy:
            self.n_copy += 1
            raise self.exc("copy failed")
        return super().copy_eq()

    def replace_eq(self, source_eq=None):
        if (self.fail_replace is not None
                and self.n_replace + 1 == self.fail_replace):
            self.n_replace += 1
            raise self.exc("replace failed")
        super().replace_eq(source_eq=source_eq)


@pytest.mark.parametrize("kw, why", [
    (dict(fail_copy=2), "jumped state could not be copied"),
    (dict(fail_replace=1), "starting state could not be re-installed")])
def test_a_toolkit_failure_inside_the_guard_keeps_the_unguarded_state(kw,
                                                                       why):
    """F2: the stage solved; a failing toolkit call inside the guard leaves
    the draw on the jumped (unguarded) state under the stage's own bounds,
    stamped "failed" with the reason -- never a stage failure."""
    gs = _FailingGS(_Branches(), **kw)
    gs.set_coil_bounds(_STAGE)
    g = _guard()
    st = g.solve_stage(gs, 1, "homotopy pass 2/3", _STAGE, prev_bounds=_PREV)
    assert st["action"] == "failed" and why in st["reason"]
    assert gs.eq.branch == "B" and gs.eq.z == pytest.approx(-17e-3)
    assert gs.bounds == _STAGE and g.record["action"] == "failed"


def test_a_programming_error_inside_the_guard_propagates():
    gs = _FailingGS(_Branches(), fail_copy=2, exc=KeyError)
    gs.set_coil_bounds(_STAGE)
    with pytest.raises(KeyError):
        _guard().solve_stage(gs, 1, "homotopy pass 2/3", _STAGE,
                             prev_bounds=_PREV)


def test_an_unrestorable_jumped_state_fails_the_stage():
    """The one case the guard cannot undo: the walk ran and the jumped
    state cannot be put back, so the stage fails under the homotopy's own
    rule (a RuntimeError), stamped with the restore error."""
    gs = _FailingGS(_Branches(s_fold=0.6), fail_replace=7)  # 1 + 5 sub-steps, then the restore
    gs.set_coil_bounds(_STAGE)
    g = _guard()
    with pytest.raises(RuntimeError, match="could not be restored"):
        g.solve_stage(gs, 1, "homotopy pass 2/3", _STAGE, prev_bounds=_PREV)
    st = g.record["stages"][-1]
    assert st["action"] == "failed" and "replace failed" in st["restore_error"]


def test_counts_and_text():
    recs = [dict(armed=True, fired=False, action="none"),
            dict(armed=True, fired=True, action="held", n_extra_solves=5),
            dict(armed=True, fired=True, action="legitimate",
                 n_extra_solves=4),
            H.unarmed_record("no reference"), None]
    c = H.guard_counts(recs)
    assert (c["n_records"], c["n_armed"], c["n_fired"]) == (4, 3, 2)
    assert c["by_action"]["held"] == 1 and c["n_extra_solves"] == 9
    t = H.guard_text(c)
    assert "fired on 2 of 3 armed" in t and "held 1" in t
    assert H.guard_text(H.guard_counts([H.unarmed_record("x")])) == ""


# ---------------------------------------------------------------------------
#  the setting
# ---------------------------------------------------------------------------
def test_the_setting_is_opt_in_and_validated():
    from bouquet.config import GenerationConfig
    from bouquet.engine import ENGINE_FIELD_DEFAULTS, validate_engine_settings
    g = GenerationConfig(reconstruction_engine="unified")
    assert g.guard_bistability is False
    assert ENGINE_FIELD_DEFAULTS["guard_bistability"] is False
    validate_engine_settings(g)
    g.guard_bistability = True
    validate_engine_settings(g)
    g.guard_bistability = "yes"
    with pytest.raises(ValueError, match="guard_bistability must be a bool"):
        validate_engine_settings(g)
    g.guard_bistability = True
    g.engine_draw_homotopy = False
    g.homotopy_passes = GenerationConfig().homotopy_passes
    with pytest.raises(ValueError, match="engine_draw_homotopy=False"):
        validate_engine_settings(g)
    gl = GenerationConfig(reconstruction_engine="legacy")
    validate_engine_settings(gl)
    gl.guard_bistability = True
    with pytest.raises(ValueError, match="guard_bistability"):
        validate_engine_settings(gl)


def test_it_is_a_registered_experimental_feature():
    from types import SimpleNamespace
    from bouquet.config import GenerationConfig
    from bouquet.experimental import REGISTRY, experimental_features_enabled
    assert "bistability_guard" in REGISTRY
    c = SimpleNamespace(generation=GenerationConfig(), source=None,
                        uncertainty=None)
    assert "bistability_guard" not in experimental_features_enabled(c)
    c.generation.guard_bistability = True
    assert "bistability_guard" in experimental_features_enabled(c)


def test_the_legacy_method_has_no_guard():
    from bouquet.draw_methods import DrawMethod
    assert DrawMethod().homotopy_guard(object()) is None


# ---------------------------------------------------------------------------
#  end to end on the toy engine: the stamps (the stand-in never jumps)
# ---------------------------------------------------------------------------
import _engine_toy as T  # noqa: E402
import test_engine_draws as TD  # noqa: E402
from test_engine_draws import toy_bouquet_solver  # noqa: E402,F401


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _prepared(tmp_path, guard):
    b = TD._bq(tmp_path)
    b.config.generation.guard_bistability = guard
    b.setup_solver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quiet(b.prepare_baseline)
    return b


def _archive(b):
    import h5py
    from bouquet.utils import _group_path
    h, sk = b.config.output_header, b.config.generation.scan_key
    out = []
    with h5py.File(h + ".h5", "r") as hf:
        for i in range(2):
            g = hf[_group_path(sk, i)]
            out.append(({k: g[k][()] for k in g.keys()
                         if hasattr(g[k], "shape") and k != "eqdsk"},
                        dict(g.attrs)))
    return out


def test_the_stamps_on_and_none_off(tmp_path, toy_bouquet_solver):
    import json
    import h5py
    from bouquet.engine import load_baseline_engine
    from bouquet.engine_draws import read_draw_engine
    from bouquet.filtering import _scan_group_path
    from bouquet.schema import (GUARD_ACTION_ATTR, GUARD_ARMED_ATTR,
                                GUARD_ARMED_SCAN_ATTR, GUARD_COUNTS_SCAN_ATTR,
                                GUARD_FIRED_ATTR, GUARD_FIRED_SCAN_ATTR)
    runs = {}
    for guard in (False, True):
        d = tmp_path / str(guard)
        d.mkdir()
        b = _prepared(d, guard)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diags = b.generate()
        assert len(diags) == 2
        runs[guard] = (b, diags, buf.getvalue())
    b, diags, txt = runs[True]
    h, sk = b.config.output_header, b.config.generation.scan_key
    for i, dg in enumerate(diags):
        gr = dg["engine"]["bistability_guard"]
        assert gr["armed"] is True and gr["fired"] is False
        assert gr["action"] == "none" and gr["path"] == "continuous"
        assert gr["version"] == H.GUARD_VERSION
        assert len(gr["stages"]) >= 1          # the homotopy ran, guarded
        assert [s["stage"] for s in gr["stages"]] == list(
            range(1, len(gr["stages"]) + 1))
        assert gr["entry_axis"]["displacement_mm"] == pytest.approx(0.0)
        assert read_draw_engine(h, i, scan_key=sk)["bistability_guard"][
            "armed"] is True
    with h5py.File(h + ".h5", "r") as hf:
        s = hf[_scan_group_path(sk)].attrs
        assert int(s[GUARD_ARMED_SCAN_ATTR]) == 2
        assert int(s[GUARD_FIRED_SCAN_ATTR]) == 0
        assert json.loads(s[GUARD_COUNTS_SCAN_ATTR])["by_action"]["none"] == 2
    for _, attrs in _archive(b):
        assert bool(attrs[GUARD_ARMED_ATTR]) is True
        assert bool(attrs[GUARD_FIRED_ATTR]) is False
        assert attrs[GUARD_ACTION_ATTR] in ("none", b"none")
    assert "bistability guard: fired on 0 of 2 armed draw(s)" in txt
    assert b.engine_bistability_guard["n_armed"] == 2
    assert load_baseline_engine(h, scan_key=sk)["draws"][
        "bistability_guard"]["on"] is True
    # off: nothing is written, and the archived draws are the same numbers
    b0, diags0, txt0 = runs[False]
    h0 = b0.config.output_header
    assert "bistability guard" not in txt0
    assert not hasattr(b0, "engine_bistability_guard")
    assert all("bistability_guard" not in dg["engine"] for dg in diags0)
    with h5py.File(h0 + ".h5", "r") as hf:
        s = hf[_scan_group_path(sk)].attrs
        assert GUARD_ARMED_SCAN_ATTR not in s and GUARD_COUNTS_SCAN_ATTR \
            not in s
    assert "bistability_guard" not in load_baseline_engine(
        h0, scan_key=sk)["draws"]
    for (ds_on, at_on), (ds_off, at_off) in zip(_archive(b), _archive(b0)):
        assert GUARD_ARMED_ATTR not in at_off
        assert set(ds_on) == set(ds_off)
        for k in ds_off:
            np.testing.assert_array_equal(ds_on[k], ds_off[k], err_msg=k)


def test_the_sigma0_check_keeps_the_ensembles_record_and_an_off_run_clears_it(
        tmp_path, toy_bouquet_solver):
    """F5: verify_sigma0_consistency() after an ensemble leaves the
    ensemble's guard record in place; a later guard-off generate() leaves
    none from the earlier run."""
    b = _prepared(tmp_path, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quiet(b.generate)
        rec = dict(b.engine_bistability_guard)
        assert rec["n_armed"] == 2
        _quiet(b.verify_sigma0_consistency)
        assert b.engine_bistability_guard["n_armed"] == 2
        assert b.engine_bistability_guard["records"] == rec["records"]
        b.config.generation.guard_bistability = False
        _quiet(b.generate)
    assert not hasattr(b, "engine_bistability_guard")
