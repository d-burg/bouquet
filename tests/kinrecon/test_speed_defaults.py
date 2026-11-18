"""The in-process speed options as the default, without the solver: table inputs copied when their functions are
made, the coefficient memo only on inputs known to be pure, the options in effect in every fit and series record,
and the default itself (``bouquet.kinrecon.speed``)."""

import numpy as np
import pytest

from bouquet.kinrecon import speed as SP


# ---------------------------------------------------------------------------------------------------- 1. table copies
@pytest.mark.parametrize("form", ["2", "3"])
@pytest.mark.parametrize("memo", [False, True])
def test_profile_table_changed_in_place_afterwards_changes_no_value(form, memo):
    from bouquet.kinrecon.current import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    times = np.array([0.0, 0.02])
    table = np.vstack((vals, 2 * vals))
    spec = (rho, vals) if form == "2" else (times, rho, table)
    pts = np.linspace(0.05, 0.95, 37)
    with SP.speed_scope(SP.SpeedSettings(memo=memo, mesh_index=False, reuse_stats=False)):
        f = DF.as_profile_function(spec, log=True)
        before = [f(t, pts) for t in (0.004, 0.011)]
        # the caller changes every table array in place after the function was made
        times[1] = 2.0
        rho[:] = rho[::-1] * 0.5
        vals *= 3.0
        table *= 5.0
        after = [f(t, pts) for t in (0.004, 0.011)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_time_table_changed_in_place_afterwards_changes_no_value():
    from bouquet.kinrecon.current import diffusion as DF
    times, values = np.array([0.0, 0.01, 0.02]), np.array([1.0e6, 1.2e6, 1.5e6])
    for f in (DF.as_time_function((times, values)), DF.IpBoundary((times, values))):
        before = [f(t) for t in (0.004, 0.013)]
        saved = times.copy(), values.copy()
        times[1], values[:] = 0.019, 7.0
        assert [f(t) for t in (0.004, 0.013)] == before
        times[:], values[:] = saved


# ---------------------------------------------------------------------------------------------------- 2. pure inputs
def _cyl():
    from bouquet.kinrecon.current.geometry import FluxSurfaceGeometry
    return FluxSurfaceGeometry.cylinder(0.6, 1.7, 2.0, n=41)


def test_known_pure_inputs():
    from bouquet.kinrecon.current import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    assert DF.known_pure(None)
    assert DF.known_pure(DF.as_profile_function(None)) and DF.known_pure(DF.as_profile_function(2.0))
    assert DF.known_pure(DF.as_profile_function((rho, rho + 1.0)))
    assert DF.known_pure(DF.as_profile_function(((0.0, 1.0), rho, np.vstack((rho, rho)))))
    assert not DF.known_pure(DF.as_profile_function(lambda t, r: r))
    assert DF.known_pure(DF.IpBoundary(1.0e6)) and DF.known_pure(DF.IpBoundary(((0.0, 1.0), (1.0e6, 2.0e6))))
    assert DF.known_pure(DF.VloopBoundary(0.5)) and DF.known_pure(DF.VloopBoundary(((0.0, 1.0), (0.5, 0.6)), True))
    assert not DF.known_pure(DF.IpBoundary(lambda t: 1.0e6)) and not DF.known_pure(DF.VloopBoundary(lambda t: 0.5))
    assert DF.known_pure(DF.OuterLayer(0.95, 2.0, ((0.0, 1.0), (1.0, 2.0)), 0.0))
    assert DF.known_pure(DF.OuterLayer(0.95, ((0.0, 1.0), (2.0, 2.1)), 1.0, 0.5, extrapolate=True))
    assert not DF.known_pure(DF.OuterLayer(0.95, 2.0, lambda t: 1.0, 0.0))
    assert not DF.known_pure(DF.OuterLayer(0.95, 2.0, 1.0, lambda t: 0.0))
    assert not DF.known_pure(DF.OuterLayer(0.95, lambda t: 2.0, 1.0, 0.0))
    assert not DF.known_pure(lambda t: 1.0) and not DF.known_pure(object())


def _solver(which, state):
    """A static-geometry solver whose ``which`` input ('jni', 'boundary', 'layer' or 'none': all tables) is a
    caller's callable that reads the mutable ``state``."""
    from bouquet.kinrecon.current import diffusion as DF
    g = _cyl()
    rho = g.rho
    sig = ((0.0, 0.02), rho, np.vstack((3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5,
                                        3.3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5)))
    jni = ((0.0, 0.02), rho, np.vstack((2e5 * rho, 3e5 * rho)))
    bnd = DF.IpBoundary(((0.0, 0.02), (1.5e6, 1.6e6)))
    layer = DF.OuterLayer(0.95, 2.0, ((0.0, 0.02), (1.0e3, 1.1e3)), ((0.0, 0.02), (10.0, 20.0)))
    if which == "jni":
        jni = lambda t, r: state["k"] * 2e5 * np.asarray(r)                       # noqa: E731
    elif which == "boundary":
        bnd = DF.IpBoundary(lambda t: state["k"] * 1.5e6)
    elif which == "layer":
        layer = DF.OuterLayer(0.95, 2.0, lambda t: state["k"] * 1.0e3, 10.0)
    return DF.CurrentDiffusion(g, sig, bnd, j_ni=jni, outer_layer=layer)


def _values(sol, rf, t):
    c = sol.coefficients(t, rf)
    return [c.jni_c.copy(), c.s_c.copy(), c.k_f.copy(), np.array([c.bc_value, c.layer_A, c.layer_B])]


@pytest.mark.parametrize("which", ["jni", "boundary", "layer"])
def test_coefficient_memo_stores_nothing_for_a_callers_callable(which):
    from bouquet.kinrecon.current import diffusion as DF
    rf = DF.CurrentState.uniform_faces(24)
    state = dict(k=1.0)
    sol = _solver(which, state)
    with SP.speed_scope(SP.SpeedSettings(memo=True, mesh_index=False, reuse_stats=False)):
        first = _values(sol, rf, 0.007)
        state["k"] = 1.7                                        # the callable's state changes between evaluations
        second = _values(sol, rf, 0.007)
    assert len(sol._base_cache) == 0
    ref_state = dict(k=1.7)
    with SP.speed_scope(SP.SpeedSettings(memo=False, mesh_index=False, reuse_stats=False)):
        ref = _values(_solver(which, ref_state), rf, 0.007)
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))    # the input did change
    assert all(np.array_equal(a, b) for a, b in zip(second, ref))


def test_coefficient_memo_stores_table_inputs_and_changes_no_value():
    from bouquet.kinrecon.current import diffusion as DF
    rf = DF.CurrentState.uniform_faces(24)
    times = [0.0, 0.007, 0.02, 0.007]
    with SP.speed_scope(SP.SpeedSettings(memo=False, mesh_index=False, reuse_stats=False)):
        off_sol = _solver("none", {})
        off = [v for t in times for v in _values(off_sol, rf, t)]
    assert len(off_sol._base_cache) == 0
    on_sol = _solver("none", {})
    with SP.speed_scope(SP.SpeedSettings(memo=True, mesh_index=False, reuse_stats=False)):
        on = [v for t in times for v in _values(on_sol, rf, t)]
    assert len(on_sol._base_cache) == 3                         # three distinct times stored, the fourth a hit
    assert all(np.array_equal(a, b) for a, b in zip(off, on))


# ---------------------------------------------------------------------------------------------------- 3. the record
def _effect(memo, mesh_index, reuse_stats, workers=1, factor=3.0, floor=60.0):
    return dict(memo=memo, mesh_index=mesh_index, reuse_stats=reuse_stats,
                workers=dict(column_workers=workers, column_time_factor=factor, column_time_floor_s=floor))


def test_series_record_holds_the_options_in_effect(monkeypatch):
    import sys
    from types import SimpleNamespace
    from bouquet.kinrecon import series as SE
    monkeypatch.setattr(SE, "_run_series", lambda *a, **k: dict(status="completed"))
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=1))
    assert SE.run_series(fm, [], None)["speed_in_effect"] == _effect(True, True, True)     # the default
    assert SE.run_series(fm, [], None, speed=SP.SpeedSettings.all_off())["speed_in_effect"] == \
        _effect(False, False, False)
    memo = SP.SpeedSettings(memo=True, mesh_index=True, reuse_stats=False)
    assert SE.run_series(fm, [], None, speed=memo)["speed_in_effect"] == _effect(True, True, False)
    if sys.platform.startswith("linux"):
        par = SP.SpeedSettings(column_workers=2, memo=True, mesh_index=False, reuse_stats=True, column_time_factor=2.0)
        assert SE.run_series(fm, [], None, speed=par)["speed_in_effect"] == _effect(True, False, True, 2, 2.0)


# ---------------------------------------------------------------------------------------------------- 4. the default
def _switches():
    from bouquet.kinrecon import diagnostics as D
    from bouquet.kinrecon import forward as FW
    from bouquet.kinrecon.current import diffusion as DF
    return (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"])


def test_default_settings_are_the_in_process_options_without_workers():
    d = SP.SpeedSettings()
    assert (d.column_workers, d.memo, d.mesh_index, d.reuse_stats) == (1, True, True, True)
    assert d.active and not d.parallel
    o = SP.SpeedSettings.all_off()
    assert (o.column_workers, o.memo, o.mesh_index, o.reuse_stats) == (1, False, False, False)
    assert not o.active
    assert SP.effective(None) == d and SP.effective(o) is o
    with pytest.raises(TypeError):
        SP.effective("all")
    assert _switches() == (False, False, False)                # outside a scope the process switches are off


@pytest.mark.parametrize("entry", ["fit", "series"])
def test_entry_points_run_with_the_default_and_a_given_setting_replaces_it(entry, monkeypatch):
    from types import SimpleNamespace
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon import series as SE
    seen = []

    def body(*a, **k):
        seen.append((_switches(), k.get("speed")))
        return dict(status="completed")

    monkeypatch.setattr(FIT, "_reconstruct", body)
    monkeypatch.setattr(SE, "_run_series", body)
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=1))

    def call(**kw):
        if entry == "fit":
            FIT.reconstruct(fm, None, None, None, **kw)
        else:
            SE.run_series(fm, [], None, **kw)
        return seen[-1]

    assert call() == ((True, True, True), None)                 # not given: the in-process options on
    off = SP.SpeedSettings.all_off()
    assert call(speed=off) == ((False, False, False), off)
    one = SP.SpeedSettings(memo=False, mesh_index=True, reuse_stats=False)
    assert call(speed=one) == ((False, True, False), one)
    with SP.speed_scope(SP.SpeedSettings()):                   # an outer scope with every option on
        assert call(speed=off)[0] == (False, False, False)     # all_off turns them off inside the call
        assert _switches() == (True, True, True)               # and the outer scope's values come back
    assert _switches() == (False, False, False)
