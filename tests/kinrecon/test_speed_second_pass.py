"""Second robustness pass of the speed settings, without the solver: the mesh key computed once per mesh and never
stale, a stale-directory cleanup that cannot raise, worker pipes numbered 1024 and above, a time bound over a whole
message, the speed settings in a series record without workers, and the boot in the host tag.

The worker tests fork small processes that do no computation; they need ``os.fork``."""

import os
import resource
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as D
from bouquet.kinrecon import speed as SP

_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


class _Mesh:
    def __init__(self, xy, tri):
        self.r = np.column_stack((xy, np.zeros(len(xy))))
        self.lc = tri
        self.lim_contour = None


def _notched_grid():
    """A grid symmetric in z with a notch cut from its upper half (its mirror image has equal coordinate sums)."""
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
    return xy, np.array(tri, dtype=np.int64)


_PB = np.array([[4.0, 2.0], [4.0, -2.0]])                       # in the upper notch / in the lower half


def _fresh(monkeypatch, index):
    monkeypatch.setitem(D.MESH_INDEX, "on", index)
    monkeypatch.setattr(D, "_ON_MESH_CACHE", {})
    monkeypatch.setattr(D, "_MESH_INDEX", {})
    monkeypatch.setattr(D, "_MESH_KEYS", {})


# ---------------------------------------------------------------------------------------------------- 1. mesh key
def test_mesh_key_is_computed_once_per_mesh(monkeypatch):
    _fresh(monkeypatch, False)
    calls = []
    real = D._mesh_key
    monkeypatch.setattr(D, "_mesh_key", lambda xy, tri: (calls.append(1), real(xy, tri))[1])
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    rng = np.random.default_rng(3)
    for _ in range(6):
        pts = rng.random((5, 2)) * [8.0, 8.0] - [0.0, 4.0]
        assert np.array_equal(D.on_mesh(m, pts), D._on_mesh_compute_scan(xy, tri, pts))
    assert len(calls) == 1


@pytest.mark.parametrize("index", [False, True])
def test_mesh_key_follows_a_mesh_changed_in_place_or_replaced(index, monkeypatch):
    _fresh(monkeypatch, index)
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    assert D.on_mesh(m, _PB).tolist() == [False, True]
    m.r[:, 1] *= -1.0                                           # the same arrays, changed in place: the mirror mesh
    assert D.on_mesh(m, _PB).tolist() == [True, False]
    m.lc[:] = m.lc[::-1].copy()                                 # triangles reordered in place: same answer, new key
    assert D.on_mesh(m, _PB).tolist() == [True, False]
    m.r = np.column_stack((xy, np.zeros(len(xy))))              # replaced by a new array: the original mesh
    assert D.on_mesh(m, _PB).tolist() == [False, True]
    keys = {k[2] for k in D._ON_MESH_CACHE}
    assert len(keys) == 2                                       # the original and the mirror points
    del m
    m2 = _Mesh(*_notched_grid())                                # new objects (perhaps at a freed id): no stale key
    m2.r[:, 1] *= -1.0
    assert D.on_mesh(m2, _PB).tolist() == [True, False]


def test_mesh_key_of_equals_the_digest_key():
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    assert D._mesh_key_of(m.r, m.lc) == D._mesh_key(m.r[:, :2], m.lc)
    assert D._mesh_key_of(m.r, m.lc) == D._mesh_key(np.ascontiguousarray(xy), tri)


# ---------------------------------------------------------------------------------------------------- 2. cleanup
def test_stale_cleanup_cannot_raise_on_planted_names(tmp_path):
    tag = SP._host_tag()
    planted = [f"oft_worker_{tag}_99999999999999999999_x",       # os.kill raises OverflowError
               f"oft_worker_{tag}_{2 ** 63}_x",
               f"oft_worker_{tag}_12345_x\n"]                    # a trailing newline is not the pattern
    made = []
    for n in planted:
        try:
            os.mkdir(tmp_path / n)
            made.append(n)
        except OSError:
            pass
    assert made
    assert SP.remove_stale_worker_dirs(str(tmp_path)) == []
    assert sorted(os.listdir(tmp_path)) == sorted(made)


@_FORK
def test_a_worker_is_made_beside_a_planted_name(tmp_path, monkeypatch):
    tag = SP._host_tag()
    os.mkdir(tmp_path / f"oft_worker_{tag}_99999999999999999999_x")
    monkeypatch.setattr(SP, "_CLEANED", set())
    env = SimpleNamespace(tempdir=str(tmp_path / f"oft_{os.getpid()}"), pid=os.getpid())
    w = SP.ForkWorker(lambda ch: ch.send("ok"), oft_env=env)
    try:
        assert w.recv(timeout=30.0) == "ok"
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------------- 3. fds >= 1024
@_FORK
def test_worker_pipes_numbered_1024_and_above():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 1400
    if hard != resource.RLIM_INFINITY and hard < want:
        pytest.skip(f"the hard limit on open files ({hard}) is below {want}")
    held = []
    try:
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        null = os.open(os.devnull, os.O_RDONLY)
        held.append(null)
        while True:
            fd = os.dup(null)
            held.append(fd)
            if fd >= 1030:
                break

        def work(ch):
            ch.send("from a high pipe")
            ch.recv()                                           # then silent

        w = SP.ForkWorker(work)
        try:
            assert w._r >= 1024 and w._w >= 1024
            assert w.recv(timeout=30.0) == "from a high pipe"
            assert w.recv(timeout=0.2) is SP.TIMED_OUT
        finally:
            w.close()
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


# ---------------------------------------------------------------------------------------------------- 4. half a message
@_FORK
def test_time_bound_covers_a_message_that_stopped_half_way():
    def work(ch):
        os.write(ch._w, struct.pack("<Q", 100000) + b"x" * 1000)     # a message begun ...
        ch.recv()                                                      # ... and never finished

    w = SP.ForkWorker(work)
    try:
        t0 = time.perf_counter()
        assert w.recv(timeout=0.5) is SP.TIMED_OUT
        assert time.perf_counter() - t0 < 20.0
    finally:
        w.close()
    with pytest.raises(ProcessLookupError):
        os.kill(w.pid, 0)


@_FORK
def test_a_whole_message_sent_in_pieces_within_the_bound_is_read():
    def work(ch):
        import pickle
        data = pickle.dumps(("column", 0, list(range(1000))))
        msg = struct.pack("<Q", len(data)) + data
        for i in range(0, len(msg), 512):
            os.write(ch._w, msg[i:i + 512])
            time.sleep(0.01)

    w = SP.ForkWorker(work)
    try:
        assert w.recv(timeout=30.0) == ("column", 0, list(range(1000)))
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------------- 5. series record
def test_series_records_the_settings_only_when_given_without_workers(monkeypatch):
    from bouquet.kinrecon import series as SE
    monkeypatch.setattr(SE, "_run_series", lambda *a, **k: dict(status="completed"))
    monkeypatch.setattr(SP, "_openmp_max_threads", lambda: None)
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=1))
    assert "speed" not in SE.run_series(fm, [], None)
    memo = SP.SpeedSettings(memo=True, mesh_index=True)
    assert SE.run_series(fm, [], None, speed=memo)["speed"] == dict(settings=memo.as_dict())
    off = SP.SpeedSettings()
    assert SE.run_series(fm, [], None, speed=off)["speed"] == dict(settings=off.as_dict())
    if sys.platform.startswith("linux"):
        assert "speed" not in SE.run_series(fm, [], None, speed=SP.SpeedSettings(column_workers=2))


# ---------------------------------------------------------------------------------------------------- 6. boot id
def test_host_tag_includes_the_boot_and_falls_back_to_the_host_name(monkeypatch):
    tag = SP._host_tag()
    assert len(tag) == 8 and all(c in "0123456789abcdef" for c in tag)
    monkeypatch.setattr(SP, "_boot_id", lambda: "boot-a")
    a = SP._host_tag()
    monkeypatch.setattr(SP, "_boot_id", lambda: "boot-b")
    b = SP._host_tag()
    monkeypatch.setattr(SP, "_boot_id", lambda: "")
    c = SP._host_tag()
    assert len({a, b, c}) == 3 and SP._WORKER_DIR.fullmatch(f"oft_worker_{c}_123_x")


def test_boot_id_is_read_or_empty(monkeypatch):
    bid = SP._boot_id()
    assert isinstance(bid, str)
    if os.path.exists("/proc/sys/kernel/random/boot_id"):
        assert len(bid) >= 32

    def no_open(*a, **k):
        raise PermissionError("not readable")

    monkeypatch.setattr(SP, "open", no_open, raising=False)     # the module's own name lookup finds it first
    assert SP._boot_id() == ""
