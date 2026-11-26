"""Two in-process speed changes that repeat no evaluation and change no value (no OFT; stand-in models).

(1) Geometry hand-off of the pass-consistency loop (``current_model.consistent_solve``): a converged solve keeps the
    geometry its final pass evaluated on the converged state; a solve that starts from exactly the paired snapshot
    uses a fresh copy of it for pass 0 instead of evaluating again.  Checked: the hit (one evaluation fewer, pass 0
    sees the geometry of the restored state, a copy that shares no object with the stored one); every case that must
    evaluate instead (another snapshot object with the same content, a snapshot of another state, other kinetic
    inputs, another settings object, the same settings object with a changed value, the memo off, starting profiles,
    no start); a failed solve hands nothing.
(2) One ``get_globals`` per ``forward._stats`` (``reuse_stats``): the second call inside one ``_stats`` returns the
    first call's values; the memo never outlives the call (a changed state is evaluated again), is never shared
    between solver objects, is removed also when ``get_stats`` raises, and is off with ``reuse_stats`` off.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import current_model as CM
from bouquet.kinrecon import forward as FW
from bouquet.kinrecon import speed as SP

N = 12
RHO = np.linspace(0.0, 1.0, N)


class FakeFM:
    """A forward model whose 'solve' moves the state halfway to the profile it is given."""

    def __init__(self, x0=0.0):
        self.state = np.full(N, float(x0))
        self.gs = SimpleNamespace(fm=self)
        self.n_solves = 0

    def solve(self, j, p, ip, init="keep", label="", lock_case=True):
        self.state = 0.5 * (self.state + np.asarray(j[1], float))
        self.n_solves += 1
        return SimpleNamespace(converged=True, status="converged", record={"attempts": [{"its": 1}]}, wall_s=0.0,
                               stats=dict(R_axis=1.7, Z_axis=0.0, l_i=1.0, q_0=1.0, Ip=1.0))

    def snapshot(self):
        return SimpleNamespace(state=self.state.copy())

    def restore(self, snap):
        self.state = snap.state.copy()

    def set_psi(self, psi):
        self.state = np.asarray(psi, float).copy()

    def update_probe_correction(self):
        return 0.0


class Recorder:
    """Stand-in geometry evaluation (a function of the solver state and the kinetic inputs) and assembly (the
    fixed point is the kinetic 'target'); records every evaluation and the geometry each assembly saw."""

    def __init__(self):
        self.evals = []
        self.seen = []

    def install(self, monkeypatch):
        def evaluate_geometry(gs, kin, cfg):
            lag = dict(state=gs.fm.state.copy(), kin_target=float(kin.target), wall_s=0.0, nested=dict(v=[1.0]))
            self.evals.append(lag)
            return lag

        def assemble(lag, kin, src, params, cfg, ip, z0=None, state=None):
            self.seen.append(lag)
            out = np.full(N, float(kin.target)) + 0.25 * (lag["state"] - float(kin.target))
            lag["nested"]["v"].append(2.0)          # an assembly that changes what it was given must not leak
            return dict(orbit_z=None, wall_s=0.0, fractions={}, sawtooth={}, knots={"n": N}, jphi_cmp=out, jphi=out,
                        x=RHO, p=np.zeros(N))

        monkeypatch.setattr(CM, "evaluate_geometry", evaluate_geometry)
        monkeypatch.setattr(CM, "assemble", assemble)
        monkeypatch.setattr(CM, "q_check", lambda lag, asm, stats, cfg: dict(passed=True, max_rel=0.0))


def _kin(target=1.0):
    return SimpleNamespace(target=target)


def _cfg(**kw):
    return CM.CurrentModelSettings(r0=1.0, max_passes=200, pass_acceleration="none", **kw)


def _center(fm, kin, cfg):
    res = CM.consistent_solve(fm, kin, None, {}, cfg, ip=1.0)
    assert res.converged and res.final_geometry is not None
    snap = fm.snapshot()
    return res, snap, CM.final_geometry_of(res, snap)


def test_handoff_skips_one_evaluation_and_gives_a_fresh_copy(monkeypatch):
    rec = Recorder()
    rec.install(monkeypatch)
    fm, kin, cfg = FakeFM(0.0), _kin(1.0), _cfg()
    with SP.speed_scope(SP.SpeedSettings()):
        res, snap, geo = _center(fm, kin, cfg)
        stored = geo["lag"]
        stored_v = list(stored["nested"]["v"])
        fm.state = fm.state + 7.0                     # the solver moved elsewhere in between
        n0 = len(rec.evals)
        res2 = CM.consistent_solve(fm, kin, None, {}, cfg, ip=1.0, start=snap, start_geometry=geo)
    assert res2.converged
    n_evals = len(rec.evals) - n0
    assert n_evals == len(res2.record["passes"]) - 1      # every pass but the first evaluated
    first = rec.seen[-len(res2.record["passes"])]
    assert np.array_equal(first["state"], snap.state)     # the geometry of the restored state
    assert first is not stored and first["nested"] is not stored["nested"]
    assert stored["nested"]["v"] == stored_v              # the stored geometry is untouched by the assembly
    # the same solve without the hand-off: the same answer, one evaluation more
    fm.state = fm.state + 7.0
    with SP.speed_scope(SP.SpeedSettings()):
        n1 = len(rec.evals)
        res3 = CM.consistent_solve(fm, kin, None, {}, cfg, ip=1.0, start=snap)
    assert len(rec.evals) - n1 == n_evals + 1
    assert np.array_equal(res3.assembly["jphi"], res2.assembly["jphi"])
    assert len(res3.record["passes"]) == len(res2.record["passes"])


def _case(name, fm, kin, cfg, snap, geo):
    """(start, start_geometry, kin, cfg, initial_profiles) of one case that must evaluate pass 0."""
    other = SimpleNamespace(state=snap.state.copy())
    moved = SimpleNamespace(state=snap.state + 3.0)
    cfg_other = _cfg()
    if name == "equal_content_other_object":
        return other, geo, kin, cfg, None
    if name == "snapshot_of_another_state":
        return moved, geo, kin, cfg, None
    if name == "other_kinetic_inputs":
        return snap, geo, _kin(1.0), cfg, None
    if name == "other_settings_object":
        return snap, geo, kin, cfg_other, None
    if name == "settings_changed_in_place":
        cfg.pass_tol = cfg.pass_tol * 0.5
        return snap, geo, kin, cfg, None
    if name == "initial_profiles":
        return snap, geo, kin, cfg, (RHO, np.ones(N), np.zeros(N))
    if name == "no_start":
        return None, geo, kin, cfg, None
    raise KeyError(name)


@pytest.mark.parametrize("name", ["equal_content_other_object", "snapshot_of_another_state", "other_kinetic_inputs",
                                  "other_settings_object", "settings_changed_in_place", "initial_profiles",
                                  "no_start", "memo_off"])
def test_handoff_is_used_only_for_its_own_snapshot_inputs_and_settings(monkeypatch, name):
    rec = Recorder()
    rec.install(monkeypatch)
    fm, kin, cfg = FakeFM(0.0), _kin(1.0), _cfg()
    with SP.speed_scope(SP.SpeedSettings()):
        _, snap, geo = _center(fm, kin, cfg)
    if name == "memo_off":
        start, g, k, c, init, scope = snap, geo, kin, cfg, None, SP.SpeedSettings.all_off()
    else:
        (start, g, k, c, init), scope = _case(name, fm, kin, cfg, snap, geo), SP.SpeedSettings()
    fm.state = fm.state + 5.0
    state_at_start = (start.state.copy() if start is not None else fm.state.copy())
    with SP.speed_scope(scope):
        n0, s0 = len(rec.evals), len(rec.seen)
        res = CM.consistent_solve(fm, k, None, {}, c, ip=1.0, start=start, start_geometry=g, initial_profiles=init)
    assert res.converged
    # every pass evaluated its own geometry, and pass 0 saw the state it started from (or the starting solve's)
    assert len(rec.evals) - n0 == len(res.record["passes"]) - (1 if init is not None else 0)
    first = rec.seen[s0]
    assert first is rec.evals[n0]
    if init is None:
        assert np.array_equal(first["state"], state_at_start)
        assert first["kin_target"] == float(k.target)


def test_failed_solve_hands_nothing(monkeypatch):
    rec = Recorder()
    rec.install(monkeypatch)
    fm, kin = FakeFM(0.0), _kin(1.0)
    cfg = CM.CurrentModelSettings(r0=1.0, max_passes=2, pass_acceleration="none")
    with SP.speed_scope(SP.SpeedSettings()):
        res = CM.consistent_solve(fm, kin, None, {}, cfg, ip=1.0)
    assert not res.converged and res.final_geometry is None
    assert CM.final_geometry_of(res, fm.snapshot()) is None
    assert CM.final_geometry_of(SimpleNamespace(), fm.snapshot()) is None


# ------------------------------------------------------------------ one get_globals per _stats
class FakeEquil:
    """The equilibrium object: get_stats calls its own get_globals, as TokaMaker's does."""

    def __init__(self, owner):
        self.owner = owner
        self.calls = 0
        self.fail_after_globals = False

    def get_globals(self):
        self.calls += 1
        v = float(self.owner.value)
        return v, np.array([v, -v]), 2.0 * v, 3.0, 0.0, 0.0, 4.0 * v

    def get_stats(self):
        ip, centroid, vol, _pv, _df, _tf, bpv = self.get_globals()
        if self.fail_after_globals:
            raise RuntimeError("trace failed")
        return dict(Ip=ip, l_i=bpv / vol, q_0=1.0, q_95=3.0, vol=vol)


class FakeGS:
    def __init__(self, value):
        self.value = value
        self._tMaker_equil = FakeEquil(self)
        self.o_point = np.array([1.7, 0.0])

    def get_stats(self):
        return self._tMaker_equil.get_stats()

    def get_globals(self):
        return self._tMaker_equil.get_globals()

    def get_q(self, psi):
        return None, np.array([2.5])


def _stats_under(reuse, gs):
    with SP.speed_scope(SP.SpeedSettings(reuse_stats=reuse)):
        return FW._stats(gs)


def test_one_get_globals_per_stats_same_values():
    gs_on, gs_off = FakeGS(1.25), FakeGS(1.25)
    a = _stats_under(True, gs_on)
    b = _stats_under(False, gs_off)
    assert a == b and FW.VERIFY_LI in a
    assert gs_on._tMaker_equil.calls == 1 and gs_off._tMaker_equil.calls == 2
    assert "get_globals" not in vars(gs_on._tMaker_equil)        # removed on exit


def test_get_globals_memo_never_outlives_the_call_or_crosses_solvers():
    gs = FakeGS(1.0)
    first = _stats_under(True, gs)
    gs.value = 2.0                                              # the state changed between two _stats calls
    second = _stats_under(True, gs)
    assert second["Ip"] == 2.0 and second != first
    assert second == _stats_under(False, FakeGS(2.0))
    assert gs._tMaker_equil.calls == 2
    g1, g2 = FakeGS(1.0), FakeGS(3.0)
    assert _stats_under(True, g1)["Ip"] == 1.0 and _stats_under(True, g2)["Ip"] == 3.0


def test_get_globals_memo_removed_when_get_stats_raises():
    gs = FakeGS(1.5)
    gs._tMaker_equil.fail_after_globals = True
    a = _stats_under(True, gs)
    assert "stats_error" in a and "get_globals" not in vars(gs._tMaker_equil)
    b = _stats_under(False, _failing(1.5))
    assert a == b


def _failing(v):
    g = FakeGS(v)
    g._tMaker_equil.fail_after_globals = True
    return g


def test_stand_in_without_equilibrium_object_is_unchanged():
    gs = SimpleNamespace(get_stats=lambda: dict(Ip=1.0), get_globals=lambda: (1.0, None, 1.0, 0, 0, 0, 1.0),
                         get_q=lambda psi: (None, np.array([2.0])), o_point=np.array([1.0, 0.0]))
    assert _stats_under(True, gs) == _stats_under(False, gs)
