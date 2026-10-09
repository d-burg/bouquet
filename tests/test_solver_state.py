"""``bouquet.solver_state``: every piece of solver state a guarded call
changes is put back, and the one-way coil-bound mode is entered ONCE, at
solver setup (:func:`enter_bounded_coil_mode`, from ``Bouquet.setup_solver``),
so two guarded calls -- and the reconstruction before them -- start from bit
for bit the same coil-solve mode.

The stand-in below models WHERE TokaMaker keeps its state (the equilibrium
object that ``copy_eq`` / ``replace_eq`` move; the device -- coil bounds with
OpenFUSIONToolkit's one-way bounded mode, VSC gains, Vcoils, pushed settings;
Python attributes on the object), and its ``solve()`` result depends on ALL
of it, so any piece left behind by a guarded call shows up in the next
call's result.  The live counterpart is
``tests/test_seeded_reproducibility.py::test_sigma0_is_bit_reproducible_unified``
(solver-marked).

Solver-free; no data.
"""
import copy
import hashlib
import json

import pytest

from bouquet.solver_state import (BOUQUET_SOLVER_ATTRS, COIL_SOLVE_BOUNDED,
                                  SolverState, coil_solve_mode,
                                  enter_bounded_coil_mode,
                                  preserved_solver_state)


class _Settings:
    def __init__(self):
        self.maxits = 800
        self.nl_tol = 1e-6
        self.urf = 0.2


class ObservableSolver:
    """Equilibrium object + device + Python attributes, all observable."""

    BIG = 1.0e98

    def __init__(self):
        self.eq = dict(psi=[0.0, 1.0, 2.0], coils={"F1A": 1.0, "F2A": -2.0},
                       reg=[("F1A", 0.0, 1.0)], targets=dict(Ip=1.0e6,
                                                             pax=1.0e4),
                       isoflux=[(1.0, 0.0), (2.0, 0.5)], profiles="recon")
        # device
        self.bounds = None              # None: never set (unbounded solve)
        self.settings = _Settings()
        self.device_settings = dict(vars(self.settings))
        self._virtual_coils = {"#VSC": {"id": 2, "facs": {"F9A": 1.0}}}
        self.vsc_device = {"F9A": 1.0}
        self._vcoils = {}
        self.log = []

    # ---- the equilibrium object -------------------------------------------
    def copy_eq(self):
        return copy.deepcopy(self.eq)

    def replace_eq(self, source_eq=None):
        self.eq = copy.deepcopy(source_eq)

    # ---- setters (equilibrium object) -------------------------------------
    def set_coil_reg(self, reg_terms=None):
        self.eq["reg"] = list(reg_terms)

    def set_targets(self, **kw):
        self.eq["targets"] = dict(kw)

    def set_isoflux(self, pts, weights=None):
        self.eq["isoflux"] = list(pts)

    def set_profiles(self, tag):
        self.eq["profiles"] = tag

    # ---- setters (device) ---------------------------------------------------
    def set_coil_bounds(self, b=None):
        # OpenFUSIONToolkit: the first call allocates the bounds (+/-1e98)
        # and switches the coil solve to bounded least squares for good
        self.log.append(("set_coil_bounds", copy.deepcopy(b)))
        cur = dict(self.bounds or {})
        if b is None:
            cur = {k: (-self.BIG, self.BIG) for k in self.eq["coils"]}
        else:
            for k, v in b.items():
                cur[k] = tuple(v)
            for k in self.eq["coils"]:
                cur.setdefault(k, (-self.BIG, self.BIG))
        self.bounds = cur

    def update_settings(self):
        self.device_settings = dict(vars(self.settings))

    def set_coil_vsc(self, gains):
        self.vsc_device = dict(gains)
        self._virtual_coils["#VSC"]["facs"] = dict(gains)

    def set_vcoils(self, res):
        self._vcoils = dict(res)

    # ---- the observable state and a result that depends on all of it -------
    def state(self):
        d = self.__dict__
        return dict(
            eq=copy.deepcopy(self.eq),
            bounded=self.bounds is not None, bounds=copy.deepcopy(self.bounds),
            settings_py=dict(vars(self.settings)),
            settings_dev=dict(self.device_settings),
            vsc=copy.deepcopy(self._virtual_coils), vsc_dev=dict(
                self.vsc_device), vcoils=dict(self._vcoils),
            attrs={k: copy.deepcopy(d[k]) for k in BOUQUET_SOLVER_ATTRS
                   if k in d})

    def solve(self):
        blob = json.dumps(self.state(), sort_keys=True, default=repr)
        return hashlib.sha1(blob.encode()).hexdigest()


def _set_up():
    """A stand-in as ``Bouquet.setup_solver`` leaves it: the bounded coil
    mode entered."""
    m = ObservableSolver()
    enter_bounded_coil_mode(m)
    return m


def _generate_like(m):
    """What a generate() does to the solver: solves from the state it is
    handed (the result), then leaves its own regularisation, targets,
    isoflux, profiles, settings, stashes and the bounded mode behind."""
    result = m.solve()
    m.set_coil_reg([("F1A", 7.0, 1e4), ("#VSC", 0.0, 1.0)])
    m._strong_coil_reg = [("F1A", 7.0, 1e4)]
    m.set_coil_bounds({"F1A": (0.0, 2.0)})        # homotopy pin ...
    m.set_coil_bounds(None)                       # ... released: +/-1e98
    m.set_targets(Ip=1.1e6)
    m.set_isoflux([(9.0, 9.0)])
    m.set_profiles("draw")
    m.eq["psi"] = [5.0, 6.0, 7.0]
    m.eq["coils"] = {"F1A": 3.0, "F2A": -1.0}
    m.settings.maxits = 100
    m.update_settings()
    m.settings.maxits = 800                       # Python value only
    m.set_coil_vsc({"F9A": 2.0})
    m.set_vcoils({"F1A": 1e-3})
    return result


def test_a_guarded_call_puts_every_piece_back():
    m = _set_up()
    st = SolverState.capture(m)
    captured = m.state()
    _generate_like(m)
    assert m.state() != captured
    st.restore()
    assert m.state() == captured


def test_setup_enters_the_one_way_bounded_mode_once_without_a_constraint():
    m = ObservableSolver()
    assert m.bounds is None and coil_solve_mode(m) == "unknown"
    assert enter_bounded_coil_mode(m) == COIL_SOLVE_BOUNDED
    assert m.bounds is not None
    assert all(v == (-m.BIG, m.BIG) for v in m.bounds.values())
    assert coil_solve_mode(m) == COIL_SOLVE_BOUNDED
    n = len(m.log)
    enter_bounded_coil_mode(m)                    # idempotent: no call
    assert len(m.log) == n


def test_the_capture_does_not_touch_the_coil_bounds():
    """The mode is entered at setup, not per guarded call."""
    m = _set_up()
    n = len(m.log)
    SolverState.capture(m)
    assert len(m.log) == n


def test_the_mode_marker_survives_a_restore():
    """The mode is one-way, so no restore may drop its record."""
    m = _set_up()
    with preserved_solver_state(m):
        _generate_like(m)
    assert coil_solve_mode(m) == COIL_SOLVE_BOUNDED


def test_recorded_hard_bounds_are_reinstalled_as_recorded():
    m = _set_up()
    m._coil_drift_bounds = {"F1A": [0.5, 1.5]}
    m.set_coil_bounds(m._coil_drift_bounds)
    with preserved_solver_state(m):
        captured = m.state()
        _generate_like(m)
        del m._coil_drift_bounds
    assert m.state() == captured
    assert m.bounds["F1A"] == (0.5, 1.5)
    assert m._coil_drift_bounds == {"F1A": [0.5, 1.5]}


def test_two_guarded_calls_are_bit_identical():
    """The engine sigma=0 check's contract: a second check starts from the
    state the first did.  With the bounded mode entered at setup every call
    is identical; on a solver that skipped it the first call is the odd one
    out (it alone solves unbounded) -- that is the leak the live solver
    showed (3.8e-7 in a_ind), and why setup enters the mode."""
    m = _set_up()
    runs = []
    for _ in range(3):
        with preserved_solver_state(m):
            runs.append(_generate_like(m))
    assert runs[0] == runs[1] == runs[2]

    m = ObservableSolver()
    runs = []
    for _ in range(3):
        with preserved_solver_state(m):
            runs.append(_generate_like(m))
    assert runs[0] != runs[1] and runs[1] == runs[2]


def test_an_equilibrium_only_restore_leaves_the_rest_behind():
    """What the check restored before (psi via replace_eq + the isoflux):
    the device and the stashes stay as generate() left them."""
    m = ObservableSolver()
    snap = m.copy_eq()
    before = m.state()
    _generate_like(m)
    m.replace_eq(source_eq=snap)
    after = m.state()
    assert after["eq"] == before["eq"]
    leaked = {k for k in before if before[k] != after[k]}
    assert leaked == {"bounded", "bounds", "settings_dev", "vsc", "vsc_dev",
                      "vcoils", "attrs"}


def test_an_exception_inside_the_guard_still_restores():
    m = _set_up()
    with pytest.raises(RuntimeError):
        with preserved_solver_state(m):
            captured = m.state()
            _generate_like(m)
            raise RuntimeError("solve failed")
    assert m.state() == captured


def test_an_absent_stash_stays_absent_and_a_present_one_is_put_back():
    m = _set_up()
    m._weak_coil_reg = [("F1A", 0.0, 1.0)]
    with preserved_solver_state(m):
        _generate_like(m)
        m._weak_coil_reg = "changed"
    assert not hasattr(m, "_strong_coil_reg")
    assert m._weak_coil_reg == [("F1A", 0.0, 1.0)]


def test_setup_solver_enters_the_bounded_mode_once_before_any_solve(
        monkeypatch, tmp_path):
    """The real ``Bouquet.setup_solver`` (OpenFUSIONToolkit replaced by a
    recording stand-in): the coil solve is put in the bounded mode ONCE,
    after the VSC and the coil regularisation are installed (the bounds
    array is sized by the virtual coils) and before the clean-equilibrium
    snapshot and any solve, with no constraint (``None`` = +/-1e98); a second
    ``setup_solver`` is a no-op."""
    import os
    import sys
    import types

    import bouquet as bq

    calls = []

    class _S:
        maxits = 0
        pm = True

    class RecordingTokaMaker:
        def __init__(self, env):
            self.settings = _S()
            self.coil_sets = {"F1A": {"id": 0, "net_turns": 1.0},
                              "F2A": {"id": 1, "net_turns": 1.0}}

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)

            def _rec(*a, **k):
                calls.append(name if name != "set_coil_bounds"
                             else ("set_coil_bounds", a[0] if a else None))
                if name == "coil_reg_term":
                    return (a, k)
                if name == "copy_eq":
                    return {}
            return _rec

    oft = types.ModuleType("OpenFUSIONToolkit")
    oft.OFT_env = lambda nthreads=1: object()
    tm = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    tm.TokaMaker = RecordingTokaMaker
    me = types.ModuleType("OpenFUSIONToolkit.TokaMaker.meshing")
    me.load_gs_mesh = lambda path: (None, None, None, {}, {})
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", oft)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tm)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.meshing",
                        me)
    ex = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                      "examples", "D3D-like")
    b = bq.Bouquet.from_geqdsk(
        os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1,
        header=str(tmp_path / "bq"))
    b.setup_solver()
    names = [c if isinstance(c, str) else c[0] for c in calls]
    assert calls.count(("set_coil_bounds", None)) == 1
    assert names.count("set_coil_bounds") == 1
    i = names.index("set_coil_bounds")
    assert names.index("set_coil_vsc") < i
    assert names.index("set_coil_reg") < i
    assert i < names.index("copy_eq")
    assert "solve" not in names
    assert coil_solve_mode(b.mygs) == COIL_SOLVE_BOUNDED
    n = len(calls)
    b.setup_solver()
    assert len(calls) == n
