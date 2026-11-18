r"""Opt-in speed settings of the reconstruction.  Off by default; none of them changes a result.

Parallel Jacobian columns (:attr:`SpeedSettings.column_workers` > 1).  The columns of one Jacobian
(:func:`.fit.reconstruct`, module docstring step 2) are independent consistent solves warm from the same center: each
starts by restoring the center's snapshot and its probe correction.  With ``column_workers = W`` the first column (and
the one-off noise re-solve, which starts from that column's end state) runs in this process, and the other columns run
in ``W - 1`` worker processes made by ``os.fork()`` at the moment the Jacobian is formed.  A forked worker holds an
exact copy of this process's memory at that moment -- the solver object with its mesh and factored operator, the
center snapshot, the probe correction, the edge-taper case, the kinetic inputs, the current-model settings and, on the
time-dependent path, the state function with everything it closes over -- so a column's solve in a worker starts from
exactly what the serial loop starts from, with no setup cost and nothing rebuilt from arrays.  Each worker runs the
serial loop body of its columns (the same function the serial loop calls: step, opposite-step retry, step growth
against the measured noise) and sends back each column's values and the side effects the serial loop would have left
in this process; they are applied here in column order:

* the forward model's solve count, its residual-history list, its per-solve iteration log
  (:meth:`.forward.ForwardModel.iteration_record`) and the edge-taper case's history list;
* the entries of the open floor-branch log (:func:`.current_model.current_branch_log`);
* the state function's side state (``state_fn.side_state``, a dict the function rewrites on every call, as the
  series driver's holder): the content left by the serially last column that changed it;
* the debugging capture list :data:`.forward.CAPTURE_LOG` when it is set.

The solver's iteration limit (:attr:`.forward.ForwardSettings.solver_maxits`) is part of the solver state a worker
inherits at the fork, so a worker's solves run under the same limit as the serial loop's.

The serial order of decisions is kept: the columns are taken in index order, the Jacobian stops at the first column
that fails (the results of later columns are discarded, with their solve counts and records, as the serial loop would
never have run them), and the noise measured in the first column is passed to the workers before they test their step
sizes against it.  A column whose worker raised, died, could not send its result, or changed the edge-taper case's
lock is re-run here serially, and so is every later column: an exception or a failure then surfaces exactly as in the
serial loop.  Workers are killed when the Jacobian ends, also on an exception (and, on Linux, when this process dies).

Robustness of the workers (none of these changes a value; each only decides where a column is computed):

* a worker column whose result has not arrived whole within the time bound (:func:`column_time_bound`; the bound
  covers the whole message, also one that stopped half way) is killed and re-run here serially, with every later
  column, exactly as a worker that died;
* when the workers cannot be made (``fork``, ``pipe`` or the temporary directory fails, as under a process or memory
  limit, or any other error while they are made; an interrupt is not caught), every column of that Jacobian is
  computed here serially;
* each re-run is recorded in the Jacobian's ``parallel`` record (``serial_reruns``, ``timeouts``,
  ``worker_start_failure``, ``time_bound_s``).

Bitwise identity with the serial loop requires one solver thread per process: parallel columns are refused when the
solver's OpenMP thread count (:func:`solver_threads`: the OFT environment the forward model actually uses and the
OpenMP runtime's own setting, not only ``ForwardSettings.nthreads``) is not 1, both because OpenMP is not safe across
``fork()`` -- a worker forked from a process whose OpenMP runtime runs a thread team can wait forever -- and because
threaded solves are not reproducible.  The count is checked once, when a fit or a series starts (:func:`validate`); a
count raised later (for example in a series callback) is not caught there, and the time bound then ends a resulting
hang.  The total number of cores used is ``column_workers`` (one thread each).  Each worker writes the solver's
temporary files into its own directory, named after this host, its boot and this process; directories left by a
process that was killed (the directory of a dead process of this user on this host and boot) are removed when the next
workers are made.  Parallel columns are refused off Linux (:func:`validate`): ``fork`` without ``exec`` is supported there,
while macOS system libraries are not safe across ``fork()`` (which is why Python itself does not fork there by default).

Parallel columns are a latency tool, not a throughput tool: they shorten the wall time of one fit, but the columns are
solved in more processes than the serial loop keeps busy, so a fit costs more core-seconds than the serial fit (about
1.6-1.9 times the allocated core-seconds with four processes on the measured cases).  For many independent fits, the
in-process options (``memo``, ``mesh_index``, ``reuse_stats``) in one process per fit cost the fewest core-seconds.

Non-physical numbers of this module (none changes a value; they change only where and when work is done): the time
bound's ``column_time_factor`` and ``column_time_floor_s`` (:class:`SpeedSettings`), and the 4000 and 600 character
truncations of a worker's traceback in the record.
"""

import contextlib
import ctypes
import hashlib
import os
import pickle
import re
import select
import shutil
import signal
import socket
import stat
import struct
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["SpeedSettings", "check_speed", "column_time_bound", "solver_threads", "speed_scope", "validate"]


@dataclass(frozen=True)
class SpeedSettings:
    """Speed options of :func:`.fit.reconstruct` and :func:`.series.run_series` (module docstring).  Every option is
    off by default, and none changes a value.

    ``column_workers``: processes that evaluate the columns of a Jacobian at once, this one included (1: serial, the
    default behavior).  Choose at most the number of cores available to the run.  Linux only.  A latency tool: it
    shortens one fit's wall time and costs more core-seconds than the serial fit (module docstring).
    ``memo``: the memo of pure evaluations in the current model (:func:`.current.diffusion.set_speed_memo`).
    ``mesh_index``: the bucket-indexed exact point-in-triangle test (:func:`.diagnostics.on_mesh`).
    ``reuse_stats``: a forward solve reuses the equilibrium statistics of its latest attempt instead of evaluating
    them again on the same state (:meth:`.forward.ForwardModel.solve`).
    ``column_time_factor`` and ``column_time_floor_s``: the time bound of a worker column (used only with
    ``column_workers > 1``; :func:`column_time_bound`).  A worker column that has sent no result after
    ``max(column_time_floor_s, column_time_factor * n_col * t_solve)`` seconds is killed and re-run in this process
    with the serial code, where ``t_solve`` is the longest consistent solve this process (and its workers) has made so
    far and ``n_col`` the most consistent solves one column can make (every step growth with an opposite-step retry:
    ``2 * (max_step_growths + 1)``, 8 by default).  Non-physical numbers that change no value, only the wall time:
    a bound that fires too early re-runs columns serially (the result is the same bit for bit), one that fires too
    late only lets a hung worker wait longer.  The defaults (factor 3, floor 60 s) allow a column 24 times the
    longest solve; a column usually makes one solve, and the longest solve is often the first, cold one, so this is
    24 to about 100 times a typical column on the measured cases -- room for a machine loaded several times over,
    while a hang still ends after minutes, not never."""
    column_workers: int = 1
    memo: bool = False
    mesh_index: bool = False
    reuse_stats: bool = False
    column_time_factor: float = 3.0
    column_time_floor_s: float = 60.0

    def __post_init__(self):
        if isinstance(self.column_workers, bool) or not isinstance(self.column_workers, (int, np.integer)) \
                or int(self.column_workers) < 1:
            raise ValueError("column_workers must be an integer >= 1")
        for name in ("memo", "mesh_index", "reuse_stats"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")
        for name in ("column_time_factor", "column_time_floor_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v) \
                    or v < 0 or (name == "column_time_factor" and v == 0):
                raise ValueError(f"{name} must be a finite number ({'> 0' if name == 'column_time_factor' else '>= 0'})")

    @classmethod
    def all_on(cls, column_workers):
        """Every option on, with ``column_workers`` processes for the columns."""
        return cls(column_workers=column_workers, memo=True, mesh_index=True, reuse_stats=True)

    @property
    def parallel(self):
        return int(self.column_workers) > 1

    @property
    def active(self):
        return self.parallel or self.memo or self.mesh_index or self.reuse_stats

    def as_dict(self):
        return asdict(self)


def _openmp_max_threads():
    """``omp_get_max_threads()`` of an OpenMP runtime already loaded into this process (the threads the next parallel
    region of the calling thread would use), or None when none is loaded (only an already loaded library is asked;
    nothing is loaded here)."""
    mode = getattr(os, "RTLD_NOLOAD", None)
    if mode is None:
        return None
    for name in ("libgomp.so.1", "libomp.so", "libomp.so.5", "libiomp5.so"):
        try:
            lib = ctypes.CDLL(name, mode=mode | getattr(os, "RTLD_LAZY", 1))
            fn = lib.omp_get_max_threads
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = []
        return int(fn())
    return None


def solver_threads(fm):
    """The OpenMP thread count the forward model's solver runs with, and where it was read.

    The forward model's OFT environment (``fm.env``; when one is given to :class:`.forward.ForwardModel`,
    ``ForwardSettings.nthreads`` is not used) holds the count it set at its start, and the loaded OpenMP runtime
    holds the count in effect now (``OFT_env.set_num_threads`` changes it later); the larger of the two is returned.
    Without an environment (a stand-in model) ``ForwardSettings.nthreads`` is the count.  Returns ``(count or None
    when it cannot be established, description)``."""
    env = getattr(fm, "env", None)
    if env is None:
        nt = getattr(getattr(fm, "settings", None), "nthreads", 1)
        return int(nt), f"ForwardSettings.nthreads = {nt} (the forward model has no solver environment)"
    found = {}
    nt_env = getattr(env, "nthreads", None)
    if nt_env is not None and int(nt_env) > 0:            # <= 0: the runtime's default, read from the runtime
        found["the solver environment's nthreads"] = int(nt_env)
    rt = _openmp_max_threads()
    if rt is not None:
        found["the OpenMP runtime's omp_get_max_threads()"] = rt
    if not found:
        return None, (f"the solver environment's nthreads is {nt_env!r} and no OpenMP runtime could be asked")
    return max(found.values()), "; ".join(f"{k} = {v}" for k, v in found.items())


def check_speed(speed, fm):
    """``speed`` when it asks for anything (validated against the forward model), else None."""
    if speed is None:
        return None
    if not isinstance(speed, SpeedSettings):
        raise TypeError("speed must be a SpeedSettings")
    if not speed.active:
        return None
    if speed.parallel:
        if not hasattr(os, "fork"):
            raise RuntimeError("parallel Jacobian columns need os.fork (not available on this platform)")
        nt, where = solver_threads(fm)
        if nt != 1:
            raise ValueError(f"parallel Jacobian columns need one solver thread per process (ForwardSettings.nthreads"
                             f" = 1 and an OFT environment made with nthreads=1); found {where}: OpenMP is not safe "
                             "across fork() (a forked worker can wait forever) and threaded solves are not "
                             "reproducible")
    return speed


def validate(speed, fm):
    """:func:`check_speed`, and parallel columns refused off Linux.  Called by :func:`.fit.reconstruct` and
    :func:`.series.run_series` when they start, before anything is solved (the thread count is read at that moment
    only)."""
    if isinstance(speed, SpeedSettings) and speed.parallel and not sys.platform.startswith("linux"):
        raise RuntimeError(f"parallel Jacobian columns (column_workers > 1) are refused on {sys.platform!r}: the "
                           "workers are made by fork() without exec, which is supported and tested on Linux only "
                           "(macOS system libraries are not safe across fork()). Use column_workers=1; the other "
                           "speed options work on every platform")
    return check_speed(speed, fm)


def in_effect(speed):
    """The speed options in effect, as written into every fit and series record (``speed_in_effect``): ``memo``,
    ``mesh_index`` and ``reuse_stats`` read from the process-wide switches at the moment of the call (inside the
    entry point's :func:`speed_scope`, so what the run actually uses), and under ``workers`` the worker settings of
    ``speed`` (``column_workers`` and the time bound; ``column_workers = 1``: one process).  A record only."""
    from . import diagnostics as _diag
    from . import forward as _fw
    from .current import diffusion as _dif
    sp = speed if isinstance(speed, SpeedSettings) else SpeedSettings()
    return dict(memo=bool(_dif._SPEED_MEMO["on"]), mesh_index=bool(_diag.MESH_INDEX["on"]),
                reuse_stats=bool(_fw.SPEED_OPTIONS["reuse_stats"]),
                workers=dict(column_workers=int(sp.column_workers), column_time_factor=float(sp.column_time_factor),
                             column_time_floor_s=float(sp.column_time_floor_s)))


# ============================================================== the time bound of a worker column
#: The longest consistent solve (wall seconds) made so far in this process and reported by its workers.
_SOLVE_WALL = dict(longest=0.0)


def note_solve_wall(seconds):
    """Record the wall time of a consistent solve (only the longest is kept)."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if np.isfinite(s) and s > _SOLVE_WALL["longest"]:
        _SOLVE_WALL["longest"] = s


def longest_solve():
    """The longest consistent solve noted in this process so far [s]."""
    return float(_SOLVE_WALL["longest"])


def column_time_bound(speed, n_col):
    """Wall seconds a worker column may run before it is killed and re-run here:
    ``max(column_time_floor_s, column_time_factor * n_col * longest consistent solve so far)``
    (:class:`SpeedSettings`; ``n_col``: the most consistent solves one column can make)."""
    return max(float(speed.column_time_floor_s), float(speed.column_time_factor) * int(n_col) * longest_solve())


@contextlib.contextmanager
def speed_scope(speed):
    """Turn the process-wide options of ``speed`` (``memo``, ``mesh_index``, ``reuse_stats``) on for the duration of
    the block and restore the previous values afterwards (:func:`.fit.reconstruct` and :func:`.series.run_series` do
    this themselves; wrap the forward model's construction too, so that its row set-up uses ``mesh_index``)."""
    if speed is None:
        yield
        return
    from . import diagnostics as _diag
    from . import forward as _fw
    from .current import diffusion as _dif
    prev = (_dif.set_speed_memo(speed.memo), _diag.MESH_INDEX["on"], _fw.SPEED_OPTIONS["reuse_stats"])
    _diag.MESH_INDEX["on"] = bool(speed.mesh_index)
    _fw.SPEED_OPTIONS["reuse_stats"] = bool(speed.reuse_stats)
    try:
        yield
    finally:
        _dif.set_speed_memo(prev[0])
        _diag.MESH_INDEX["on"] = prev[1]
        _fw.SPEED_OPTIONS["reuse_stats"] = prev[2]


# ============================================================== framed pickles over pipes
def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _wait_readable(fd, deadline):
    """True when ``fd`` can be read (data or end of stream) before ``deadline`` (a ``time.monotonic()`` value; None:
    wait as long as it takes), False when the deadline passes first.  ``poll``, not ``select``: it accepts descriptors
    of any number (``select`` refuses 1024 and above)."""
    if deadline is None:
        return True
    p = select.poll()
    p.register(fd, select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR)
    while True:
        left = deadline - time.monotonic()
        if left <= 0.0:
            return bool(p.poll(0))
        if p.poll(int(min(left, 3600.0) * 1000.0) + 1):
            return True


def _read_exact(fd, n, deadline=None):
    """``n`` bytes from ``fd``; None at the end of the stream; :data:`TIMED_OUT` when ``deadline`` passes first (also
    in the middle of a message)."""
    buf = bytearray()
    while len(buf) < n:
        if not _wait_readable(fd, deadline):
            return TIMED_OUT
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _send(fd, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _write_all(fd, struct.pack("<Q", len(data)) + data)


#: What :meth:`ForkWorker.recv` returns when a message has not arrived whole within its time bound.
TIMED_OUT = ("timed_out",)


def _recv(fd, timeout=None):
    """The next framed message; None at the end of the stream; :data:`TIMED_OUT` when the whole message has not
    arrived within ``timeout`` seconds (None: wait as long as it takes).  The bound covers the whole read, so a sender
    that stops in the middle of a message cannot block the reader."""
    deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
    head = _read_exact(fd, 8, deadline)
    if head is None or head is TIMED_OUT:
        return head
    data = _read_exact(fd, struct.unpack("<Q", head)[0], deadline)
    if data is None or data is TIMED_OUT:
        return data
    return pickle.loads(data)


#: Parent-side pipe ends of the workers alive in this process (a new worker closes them: its own pipes only).
_OPEN_FDS = set()


def _die_with_parent():
    """Linux: the worker gets SIGKILL when the process that forked it ends (no orphans)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))       # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 -- best effort; the parent also kills its workers
        pass


class _Channel:
    """The worker's side: ``send`` a result to the parent, ``recv`` a message from it (None when it is gone)."""

    def __init__(self, r, w):
        self._r, self._w = r, w

    def send(self, obj):
        _send(self._w, obj)

    def recv(self):
        return _recv(self._r)


def _flush_c_stdio():
    """Flush every C stdio stream of this process (``fflush(NULL)``), so that text buffered before a fork is not
    copied into the worker and written twice."""
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:  # noqa: BLE001 -- best effort (no C library handle on this platform)
        pass


# ============================================================== the workers' temporary directories
#: Name of a worker's temporary directory: ``oft_worker_<host tag>_<pid of the process that made it>_<random>``.
_WORKER_DIR = re.compile(r"^oft_worker_([0-9a-f]{8})_([0-9]+)_[A-Za-z0-9_]+$")
_CLEANED = set()


def _boot_id():
    """This boot of this machine (Linux ``/proc/sys/kernel/random/boot_id``), or "" where it cannot be read."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "r") as fh:
            return fh.read(64).strip()
    except Exception:  # noqa: BLE001 -- not Linux, or not readable: the host name alone
        return ""


def _host_tag():
    """Eight hex digits naming this host and this boot of it: a temporary directory on a shared file system is seen
    by several hosts, whose process ids are unrelated, and after a reboot old process ids mean nothing.  Hosts that
    share a name (containers, ``localhost``) differ in their boot id.  Where the boot id cannot be read, the host
    name alone."""
    return hashlib.sha1((socket.gethostname() + "\0" + _boot_id()).encode("utf-8", "replace")).hexdigest()[:8]


def _pid_alive(pid):
    """False only when no process ``pid`` exists on this host; any other answer (alive, another user's, a number no
    process can have, an error) counts as alive, so nothing is removed on a doubt."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001 -- PermissionError (another user's), OverflowError (no such pid can exist), ...
        return True
    return True


def remove_stale_worker_dirs(base):
    """Remove, in ``base``, the worker directories (:data:`_WORKER_DIR`) made on this host by a process of this user
    that no longer runs (left when that process was killed: SIGTERM, SIGKILL, a scheduler's time limit).  Nothing else
    is touched: the exact name, a directory (not a link), this user as owner, this host's tag and a dead process id
    are all required.  Returns the names removed; never raises."""
    removed = []
    try:
        tag, uid, me = _host_tag(), os.getuid(), os.getpid()
        names = os.listdir(base)
    except Exception:  # noqa: BLE001 -- cleanup is best effort
        return removed
    for name in names:
        try:
            m = _WORKER_DIR.fullmatch(name) if isinstance(name, str) else None
            if m is None or m.group(1) != tag:
                continue
            pid = int(m.group(2))
            if pid == me or pid <= 1 or _pid_alive(pid):
                continue
            path = os.path.join(base, name)
            st = os.lstat(path)
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid:
                continue
            shutil.rmtree(path)
            removed.append(name)
        except Exception:  # noqa: BLE001
            continue
    return removed


class ForkWorker:
    """One worker process running ``work(channel)`` on a copy of this process (``os.fork``).  The worker never
    returns into the caller's code: it leaves with ``os._exit`` (no finalizers, no exit handlers).  An exception in
    ``work`` is sent as ``("error", traceback text)``.

    ``oft_env``: the process's OFT environment; the worker then writes the solver's temporary files (the profile
    tables every solve passes to the solver through files) into a directory of its own, made here and removed by
    :meth:`close` -- with the parent's directory the processes would overwrite each other's files.  The directory is
    made beside the environment's own one and named after this process (:data:`_WORKER_DIR`); the first worker made
    in a directory removes the stale ones there (:func:`remove_stale_worker_dirs`).

    An ``OSError`` while the worker is made (temporary directory, pipes, ``fork``) is raised after everything made
    so far is released (no directory, no open pipe left)."""

    def __init__(self, work, oft_env=None):
        self._tmpdir = None
        fds = []
        try:
            if oft_env is not None:
                base = getattr(oft_env, "tempdir", None)
                where = os.path.dirname(base) if base else tempfile.gettempdir()
                if where not in _CLEANED:
                    _CLEANED.add(where)
                    remove_stale_worker_dirs(where)
                self._tmpdir = tempfile.mkdtemp(prefix=f"oft_worker_{_host_tag()}_{os.getpid()}_", dir=where)
            r_res, w_res = os.pipe()
            fds += [r_res, w_res]
            r_cmd, w_cmd = os.pipe()
            fds += [r_cmd, w_cmd]
            sys.stdout.flush()
            sys.stderr.flush()
            _flush_c_stdio()
            pid = os.fork()
        except BaseException:
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if self._tmpdir is not None:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            raise
        if pid == 0:                                       # ---- worker
            code = 0
            try:
                _die_with_parent()
                for fd in list(_OPEN_FDS):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                os.close(r_res)
                os.close(w_cmd)
                if oft_env is not None:
                    oft_env.tempdir = self._tmpdir
                    oft_env.pid = os.getpid()
                ch = _Channel(r_cmd, w_res)
                try:
                    work(ch)
                except BaseException:  # noqa: BLE001 -- reported to the parent, which re-runs the work serially
                    code = 1
                    try:
                        ch.send(("error", traceback.format_exc()[-4000:]))
                    except BaseException:  # noqa: BLE001
                        pass
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except BaseException:  # noqa: BLE001
                    pass
            finally:
                os._exit(code)
        os.close(w_res)
        os.close(r_cmd)
        self.pid, self._r, self._w = pid, r_res, w_cmd
        _OPEN_FDS.update((r_res, w_cmd))
        self._closed = False

    def send(self, obj):
        """Message to the worker; False when it is gone."""
        try:
            _send(self._w, obj)
            return True
        except (BrokenPipeError, OSError):
            return False

    def recv(self, timeout=None):
        """Next result of the worker, None when it ended without sending one, or :data:`TIMED_OUT` when none began
        within ``timeout`` seconds (None: wait as long as it takes)."""
        try:
            return _recv(self._r, timeout)
        except (OSError, EOFError, pickle.UnpicklingError):
            return None

    def close(self):
        """Kill the worker (whatever it is doing) and reap it."""
        if self._closed:
            return
        self._closed = True
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for fd in (self._r, self._w):
            _OPEN_FDS.discard(fd)
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


# ============================================================== side effects of a column
def effects_mark(fm, log, side):
    """What a column's side effects are measured against (taken in the worker before the column)."""
    from . import forward as _fw
    case = fm.case
    its = getattr(fm, "iteration_log", None)
    return dict(n_solves=fm.n_solves, history=len(fm.history), case_history=len(case.history),
                iteration_log=None if its is None else len(its),
                case_lock=(case.locked, case.spec, case.decided_by), taper_restarts=len(fm.taper_restarts),
                log=None if log is None else len(log),
                capture=None if _fw.CAPTURE_LOG is None else len(_fw.CAPTURE_LOG),
                side=None if side is None else dict(side))


def effects_since(mark, fm, log, side):
    """The side effects of the column run since ``mark`` (sent to the parent with the column)."""
    from . import forward as _fw
    case = fm.case
    lock = (case.locked, case.spec, case.decided_by)
    side_changed = False
    if side is not None:
        before = mark["side"]
        side_changed = set(before) != set(side) or any(side[k] is not before[k] for k in side)
    its = getattr(fm, "iteration_log", None)
    return dict(n_solves=fm.n_solves - mark["n_solves"], history=list(fm.history[mark["history"]:]),
                case_history=list(case.history[mark["case_history"]:]),
                iteration_log=(None if its is None or mark.get("iteration_log") is None
                               else [dict(e) for e in its[mark["iteration_log"]:]]),
                case_changed=bool(lock[0] != mark["case_lock"][0] or lock[1] is not mark["case_lock"][1]
                                  or lock[2] != mark["case_lock"][2]
                                  or len(fm.taper_restarts) != mark["taper_restarts"]),
                log=None if log is None or mark["log"] is None else list(log[mark["log"]:]),
                capture=(None if _fw.CAPTURE_LOG is None or mark["capture"] is None
                         else list(_fw.CAPTURE_LOG[mark["capture"]:])),
                side=dict(side) if side_changed else None)


def apply_effects(eff, fm, log, side):
    """Apply a worker column's side effects in this process, as the serial loop would have left them."""
    from . import forward as _fw
    fm.n_solves += int(eff["n_solves"])
    fm.history.extend(eff["history"])
    fm.case.history.extend(eff["case_history"])
    its = getattr(fm, "iteration_log", None)
    if its is not None and eff.get("iteration_log"):
        its.extend(eff["iteration_log"])       # the worker's solves' iteration counts, in column order
    if log is not None and eff["log"]:
        log.extend(eff["log"])
    if _fw.CAPTURE_LOG is not None and eff["capture"]:
        _fw.CAPTURE_LOG.extend(eff["capture"])
    if side is not None and eff["side"] is not None:
        side.clear()
        side.update(eff["side"])
