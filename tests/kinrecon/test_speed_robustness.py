"""Robustness of the opt-in speed settings (``bouquet.kinrecon.speed``) without the solver: the time bound of a worker
column, the thread check on the solver's own environment, the refusal off Linux, a worker that cannot be made, the
workers' temporary directories, the mesh caches keyed on the mesh itself, C-stdio text across a fork, and the memo's
cached values (a mutated hit, a time-dependent input on a static geometry, a solver copy with other inputs).

The worker tests fork small processes that do no computation; they need ``os.fork``."""

import copy
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as D
from bouquet.kinrecon import speed as SP

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def _fm(nthreads=1, env=None):
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=nthreads))
    if env is not None:
        fm.env = env
    return fm


# ---------------------------------------------------------------------------------------------------- settings
def test_time_bound_settings_and_rule(monkeypatch):
    s = SP.SpeedSettings()
    assert (s.column_time_factor, s.column_time_floor_s) == (3.0, 60.0)
    assert s.as_dict()["column_time_factor"] == 3.0 and s.as_dict()["column_time_floor_s"] == 60.0
    for bad in (dict(column_time_factor=0.0), dict(column_time_factor=-1.0), dict(column_time_factor=float("nan")),
                dict(column_time_factor=True), dict(column_time_floor_s=-1.0), dict(column_time_floor_s=float("inf")),
                dict(column_time_floor_s="60")):
        with pytest.raises(ValueError):
            SP.SpeedSettings(**bad)
    monkeypatch.setitem(SP._SOLVE_WALL, "longest", 0.0)
    sp = SP.SpeedSettings(column_workers=2, column_time_factor=3.0, column_time_floor_s=60.0)
    assert SP.column_time_bound(sp, 8) == 60.0                     # nothing measured yet: the floor
    for w in (2.0, None, float("nan"), "x", 13.0, 5.0):
        SP.note_solve_wall(w)
    assert SP.longest_solve() == 13.0
    assert SP.column_time_bound(sp, 8) == 3.0 * 8 * 13.0
    assert SP.column_time_bound(SP.SpeedSettings(column_time_factor=0.01, column_time_floor_s=0.0), 8) == \
        pytest.approx(0.01 * 8 * 13.0)


# ---------------------------------------------------------------------------------------------------- threads
def test_threads_are_read_from_the_solver_environment_not_the_settings(monkeypatch):
    par = SP.SpeedSettings(column_workers=2)
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    # an environment with 2 threads while ForwardSettings.nthreads stays 1: refused (a forked worker could hang)
    with pytest.raises(ValueError, match="nthreads"):
        SP.check_speed(par, _fm(1, SimpleNamespace(nthreads=2)))
    # the settings' count is not what the solver uses when an environment is given
    assert SP.check_speed(par, _fm(2, SimpleNamespace(nthreads=1))) is par
    # the OpenMP runtime's count in effect now is read too (set_num_threads after the environment was made)
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: 4)
    with pytest.raises(ValueError, match="omp_get_max_threads"):
        SP.check_speed(par, _fm(1, SimpleNamespace(nthreads=1)))
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: 1)
    assert SP.check_speed(par, _fm(1, SimpleNamespace(nthreads=1))) is par
    # an environment left at the runtime default with no runtime to ask: the count is unknown, refused
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    with pytest.raises(ValueError):
        SP.check_speed(par, _fm(1, SimpleNamespace(nthreads=-1)))
    # threads matter only for forked workers
    memo = SP.SpeedSettings(memo=True)
    assert SP.check_speed(memo, _fm(1, SimpleNamespace(nthreads=4))) is memo


def test_openmp_query_loads_nothing():
    n = SP._openmp_max_threads()
    assert n is None or (isinstance(n, int) and n >= 1)


# ---------------------------------------------------------------------------------------------------- platform
def test_parallel_columns_are_refused_off_linux(monkeypatch):
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon import series as SE
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    monkeypatch.setattr(sys, "platform", "darwin")
    par = SP.SpeedSettings(column_workers=2)
    with pytest.raises(RuntimeError, match="Linux"):
        SP.validate(par, _fm())
    with pytest.raises(RuntimeError, match="Linux"):
        FIT.reconstruct(_fm(), None, None, None, FIT.FitSettings(anchor_follow=False),
                        speed=par)                                    # refused before anything is solved
    with pytest.raises(RuntimeError, match="Linux"):
        SE.run_series(_fm(), [], None, speed=par)
    # the in-process options stay available on every platform
    memo = SP.SpeedSettings(memo=True, mesh_index=True, reuse_stats=True)
    assert SP.validate(memo, _fm()) is memo
    assert SP.validate(None, _fm()) is None
    monkeypatch.setattr(sys, "platform", "linux")
    assert SP.validate(par, _fm()) is par


# ---------------------------------------------------------------------------------------------------- workers
@_FORK
def test_a_silent_worker_times_out_and_close_kills_it():
    w = SP.ForkWorker(lambda ch: ch.recv())                   # waits for a message that never comes
    try:
        t0 = time.perf_counter()
        assert w.recv(timeout=0.3) is SP.TIMED_OUT
        assert 0.25 <= time.perf_counter() - t0 < 10.0
    finally:
        w.close()
    with pytest.raises(ProcessLookupError):
        os.kill(w.pid, 0)


@_FORK
def test_a_result_within_the_bound_is_read():
    def work(ch):
        time.sleep(0.2)
        ch.send(("column", 0, "late but in time"))

    w = SP.ForkWorker(work)
    try:
        assert w.recv(timeout=30.0) == ("column", 0, "late but in time")
        assert w.recv(timeout=30.0) is None                   # ended
    finally:
        w.close()


def _open_fds():
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    return None


def test_a_refused_fork_raises_oserror_and_leaves_nothing(monkeypatch, tmp_path):
    def no_fork():
        raise BlockingIOError(11, "Resource temporarily unavailable (refused for the test)")

    monkeypatch.setattr(os, "fork", no_fork)
    env = SimpleNamespace(tempdir=str(tmp_path / "oft_1"), pid=1)
    n0 = _open_fds()
    with pytest.raises(OSError):
        SP.ForkWorker(lambda ch: None, oft_env=env)
    assert not [x for x in os.listdir(tmp_path) if x.startswith("oft_worker_")]
    assert _open_fds() == n0
    assert env.tempdir == str(tmp_path / "oft_1")


# ---------------------------------------------------------------------------------------------------- temp dirs
def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_stale_worker_directories_of_dead_processes_are_removed_and_nothing_else(tmp_path):
    tag, dead = SP._host_tag(), _dead_pid()
    names = dict(
        stale=f"oft_worker_{tag}_{dead}_ab12cd_3",                     # this host, this user, dead process: removed
        live=f"oft_worker_{tag}_{os.getpid()}_ab12cd_3",               # this process: kept
        other_host=f"oft_worker_{'0' * 8 if tag != '0' * 8 else '1' * 8}_{dead}_ab12cd_3",
        old_style="oft_worker_ab12cd_3",                               # no process id in the name: kept
        other="oft_123",                                               # not made by the workers: kept
    )
    for n in names.values():
        os.mkdir(tmp_path / n)
        (tmp_path / n / "f.dat").write_text("x")
    as_file = f"oft_worker_{tag}_{dead}_file_1"                         # the name, but a file: kept
    (tmp_path / as_file).write_text("x")
    removed = SP.remove_stale_worker_dirs(str(tmp_path))
    assert removed == [names["stale"]]
    left = set(os.listdir(tmp_path))
    assert left == (set(names.values()) - {names["stale"]}) | {as_file}
    assert SP.remove_stale_worker_dirs(str(tmp_path / "missing")) == []


@_FORK
def test_worker_directory_is_named_after_this_process_and_removed(tmp_path, monkeypatch):
    tag, dead = SP._host_tag(), _dead_pid()
    stale = tmp_path / f"oft_worker_{tag}_{dead}_zz_9"
    stale.mkdir()
    monkeypatch.setattr(SP, "_CLEANED", set())
    env = SimpleNamespace(tempdir=str(tmp_path / f"oft_{os.getpid()}"), pid=os.getpid())

    def work(ch):
        ch.send(os.path.basename(env.tempdir))                  # the worker's own directory

    w = SP.ForkWorker(work, oft_env=env)
    try:
        name = w.recv(timeout=30.0)
        assert SP._WORKER_DIR.match(name) and name.startswith(f"oft_worker_{tag}_{os.getpid()}_")
        assert (tmp_path / name).is_dir()
        assert not stale.exists()                               # removed when the first worker was made here
    finally:
        w.close()
    assert not (tmp_path / name).exists()
    assert env.tempdir == str(tmp_path / f"oft_{os.getpid()}")   # the parent's environment is unchanged


@_FORK
def test_c_stdio_text_is_flushed_before_the_fork_and_written_once(tmp_path):
    """Text buffered in C stdio before a fork would be copied into the worker and written twice if the worker's C
    library flushed it; the parent flushes before every fork."""
    out = tmp_path / "out.txt"
    script = textwrap.dedent("""
        import ctypes, sys
        from bouquet.kinrecon import speed as SP
        libc = ctypes.CDLL(None)
        libc.printf(b"PARENT-TEXT\\n")                       # stays in the C buffer (stdout is a file)
        w = SP.ForkWorker(lambda ch: (libc.fflush(None), ch.send("done")))
        assert w.recv(timeout=60.0) == "done"
        w.close()
        libc.fflush(None)
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, os.environ.get("PYTHONPATH", "")) if p))
    with open(out, "w") as fh:
        proc = subprocess.run([sys.executable, "-c", script], stdout=fh, stderr=subprocess.PIPE, text=True, env=env,
                              cwd=str(tmp_path), timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.read_text().count("PARENT-TEXT") == 1


# ---------------------------------------------------------------------------------------------------- mesh caches
class _Mesh:
    def __init__(self, xy, tri):
        self.r = np.column_stack((xy, np.zeros(len(xy))))
        self.lc = tri
        self.lim_contour = None


def _mirror_meshes():
    """Two different meshes with equal coordinate sums, equal x-weighted sums and equal shapes: a grid symmetric in z
    with a notch cut from its upper half, and its mirror image."""
    xs, zs = np.arange(0, 9), np.arange(-4, 5)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    xy = np.column_stack((X.ravel(), Z.ravel())).astype(float)
    tri = []
    for i in range(xs.size - 1):
        for j in range(zs.size - 1):
            if 3 <= i <= 4 and zs[j] >= 1:
                continue
            a, b, c, d = i * zs.size + j, (i + 1) * zs.size + j, (i + 1) * zs.size + j + 1, i * zs.size + j + 1
            tri += [(a, b, c), (a, c, d)]
    tri = np.array(tri, dtype=np.int64)
    xy2 = xy.copy()
    xy2[:, 1] *= -1.0
    return _Mesh(xy, tri), _Mesh(xy2, tri), xy2, tri


@pytest.mark.parametrize("index", [False, True])
def test_mesh_caches_tell_two_meshes_with_equal_sums_apart(index, monkeypatch):
    m1, m2, xy2, tri = _mirror_meshes()
    pA = np.array([[1.5, 0.5], [6.5, -2.5]])
    pB = np.array([[4.0, 2.0], [4.0, -2.0]])                    # in the notch of m1 / in the notch of m2
    ref = D._on_mesh_compute_scan(xy2, tri, pB)
    assert ref.tolist() == [True, False]
    monkeypatch.setitem(D.MESH_INDEX, "on", index)
    for first in (pA, pB):                                      # the index cache (pA), the point-set cache (pB)
        D._ON_MESH_CACHE.clear()
        D._MESH_INDEX.clear()
        D.on_mesh(m1, first)
        assert D.on_mesh(m2, pB).tolist() == ref.tolist()
    D._ON_MESH_CACHE.clear()
    D._MESH_INDEX.clear()


# ---------------------------------------------------------------------------------------------------- the memo
@pytest.mark.parametrize("form", ["2", "3"])
def test_profile_memo_hit_is_a_fresh_array(form):
    from bouquet.kinrecon.current import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    spec = (rho, vals) if form == "2" else ((0.0, 0.02), rho, np.vstack((vals, 2 * vals)))
    f = DF.as_profile_function(spec, log=True)
    pts = np.linspace(0.05, 0.95, 37)
    off = f(0.007, pts)
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        f(0.007, pts)                                           # stored
        hit = f(0.007, pts)                                     # a hit
        hit[:] = -1.0                                           # a caller's change of a hit must not reach the memo
        again = f(0.007, pts)
    assert np.array_equal(again, off)
    assert not np.shares_memory(hit, again)


def _static_solver():
    """A static geometry with a time-dependent conductivity, <J_NI.B> and Ip boundary: the geometry object is the
    same at every time, so only the time tells the coefficients of two times apart."""
    from bouquet.kinrecon.current import diffusion as DF
    from bouquet.kinrecon.current.geometry import FluxSurfaceGeometry
    g = FluxSurfaceGeometry.cylinder(0.6, 1.7, 2.0, n=41)
    rho = g.rho
    sig = np.vstack((3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5, 3.3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5))
    jni = np.vstack((2e5 * rho * np.exp(-4 * (rho - 0.7) ** 2), 3.1e5 * rho * np.exp(-4 * (rho - 0.6) ** 2)))
    return DF.CurrentDiffusion(g, ((0.0, 0.02), rho, sig), DF.IpBoundary(((0.0, 0.02), (1.5e6, 1.62e6))),
                               j_ni=((0.0, 0.02), rho, jni)), rho


def _coeff_values(sol, rf, times):
    out = []
    for t in times:
        c = sol.coefficients(t, rf)
        out += [c.jni_c.copy(), c.s_c.copy(), c.m_c.copy(), c.k_f.copy(), np.array([float(c.bc_value)])]
    return out


def test_memo_on_a_static_geometry_keeps_time_dependent_inputs_apart():
    from bouquet.kinrecon.current import diffusion as DF
    times = [0.0, 0.007, 0.013, 0.02, 0.007, 0.0]
    sol, _ = _static_solver()
    rf = DF.CurrentState.uniform_faces(24)
    off = _coeff_values(sol, rf, times)
    st0 = sol.steady_state(rf, t=0.0)
    end_off = sol.advance(st0, 0.02, dt=0.001)
    sol_on, _ = _static_solver()
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        on = _coeff_values(sol_on, rf, times)
        end_on = sol_on.advance(sol_on.steady_state(rf, t=0.0), 0.02, dt=0.001)
        end_on2 = sol_on.advance(sol_on.steady_state(rf, t=0.0), 0.02, dt=0.001)   # the second pass hits
    assert not np.array_equal(off[0], off[5])                   # the inputs do depend on time
    assert len(off) == len(on) and all(np.array_equal(a, b) for a, b in zip(off, on))
    for end in (end_on, end_on2):
        assert np.array_equal(end_off.psi, end.psi) and end_off.psi_boundary == end.psi_boundary


def test_memo_on_a_solver_copy_with_other_inputs_uses_those_inputs():
    from bouquet.kinrecon.current import diffusion as DF
    sol, rho = _static_solver()
    rf = DF.CurrentState.uniform_faces(24)
    other_jni = DF.as_profile_function(((0.0, 0.02), rho, np.vstack((1e5 * rho, 1.5e5 * rho))))
    other_bnd = DF.IpBoundary(((0.0, 0.02), (1.1e6, 1.2e6)))

    def copies(s):
        a = copy.copy(s)                                        # a copy shares the solver's caches
        a.jni_fn = other_jni
        b = copy.copy(s)
        b.boundary = other_bnd
        return a, b

    ref_a, ref_b = copies(sol)
    ref = (_coeff_values(ref_a, rf, [0.007]), _coeff_values(ref_b, rf, [0.007]))
    sol_on, _ = _static_solver()
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        base = _coeff_values(sol_on, rf, [0.007])               # stored for the original inputs
        a, b = copies(sol_on)
        got = (_coeff_values(a, rf, [0.007]), _coeff_values(b, rf, [0.007]))
    assert not np.array_equal(base[0], ref[0][0])               # the copy's <J_NI.B> differs from the original's
    assert not np.array_equal(base[4], ref[1][4])               # the copy's Ip boundary value differs too
    for r, g in zip(ref, got):
        assert len(r) == len(g) and all(np.array_equal(x, y) for x, y in zip(r, g))
