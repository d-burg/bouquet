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

* the forward model's solve count, its residual-history list and the edge-taper case's history list;
* the entries of the open floor-branch log (:func:`.current_model.current_branch_log`);
* the state function's side state (``state_fn.side_state``, a dict the function rewrites on every call, as the
  series driver's holder): the content left by the serially last column that changed it;
* the debugging capture list :data:`.forward.CAPTURE_LOG` when it is set.

The serial order of decisions is kept: the columns are taken in index order, the Jacobian stops at the first column
that fails (the results of later columns are discarded, with their solve counts and records, as the serial loop would
never have run them), and the noise measured in the first column is passed to the workers before they test their step
sizes against it.  A column whose worker raised, died, could not send its result, or changed the edge-taper case's
lock is re-run here serially, and so is every later column: an exception or a failure then surfaces exactly as in the
serial loop.  Workers are killed when the Jacobian ends, also on an exception (and, on Linux, when this process dies).

Bitwise identity with the serial loop requires one solver thread per process (``ForwardSettings.nthreads = 1``,
the default): more threads are refused, both because OpenMP is not safe across ``fork()`` and because threaded
solves are not reproducible.  The total number of cores used is ``column_workers`` (one thread each).  Each worker
writes the solver's temporary files into its own directory.  Only platforms with ``os.fork`` are supported (Linux is
the tested one).
"""

import os
import pickle
import shutil
import signal
import struct
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["SpeedSettings", "check_speed"]


@dataclass(frozen=True)
class SpeedSettings:
    """Speed options of :func:`.fit.reconstruct` and :func:`.series.run_series` (module docstring).

    ``column_workers``: processes that evaluate the columns of a Jacobian at once, this one included (1: serial, the
    default behavior).  Choose at most the number of cores available to the run."""
    column_workers: int = 1

    def __post_init__(self):
        if isinstance(self.column_workers, bool) or not isinstance(self.column_workers, (int, np.integer)) \
                or int(self.column_workers) < 1:
            raise ValueError("column_workers must be an integer >= 1")

    @property
    def active(self):
        return int(self.column_workers) > 1

    def as_dict(self):
        return asdict(self)


def check_speed(speed, fm):
    """``speed`` when it asks for anything (validated against the forward model), else None."""
    if speed is None:
        return None
    if not isinstance(speed, SpeedSettings):
        raise TypeError("speed must be a SpeedSettings")
    if not speed.active:
        return None
    if not hasattr(os, "fork"):
        raise RuntimeError("parallel Jacobian columns need os.fork (not available on this platform)")
    nt = getattr(getattr(fm, "settings", None), "nthreads", 1)
    if nt != 1:
        raise ValueError(f"parallel Jacobian columns need ForwardSettings.nthreads = 1 (got {nt}): OpenMP is not safe "
                         "across fork() and threaded solves are not reproducible")
    return speed


# ============================================================== framed pickles over pipes
def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _read_exact(fd, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _send(fd, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _write_all(fd, struct.pack("<Q", len(data)) + data)


def _recv(fd):
    head = _read_exact(fd, 8)
    if head is None:
        return None
    data = _read_exact(fd, struct.unpack("<Q", head)[0])
    return None if data is None else pickle.loads(data)


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


class ForkWorker:
    """One worker process running ``work(channel)`` on a copy of this process (``os.fork``).  The worker never
    returns into the caller's code: it leaves with ``os._exit`` (no finalizers, no exit handlers).  An exception in
    ``work`` is sent as ``("error", traceback text)``.

    ``oft_env``: the process's OFT environment; the worker then writes the solver's temporary files (the profile
    tables every solve passes to the solver through files) into a directory of its own, made here and removed by
    :meth:`close` -- with the parent's directory the processes would overwrite each other's files."""

    def __init__(self, work, oft_env=None):
        self._tmpdir = None
        if oft_env is not None:
            base = getattr(oft_env, "tempdir", None)
            self._tmpdir = tempfile.mkdtemp(prefix="oft_worker_", dir=os.path.dirname(base) if base else None)
        r_res, w_res = os.pipe()
        r_cmd, w_cmd = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
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

    def recv(self):
        """Next result of the worker, or None when it ended without sending one."""
        try:
            return _recv(self._r)
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
    return dict(n_solves=fm.n_solves, history=len(fm.history), case_history=len(case.history),
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
    return dict(n_solves=fm.n_solves - mark["n_solves"], history=list(fm.history[mark["history"]:]),
                case_history=list(case.history[mark["case_history"]:]),
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
    if log is not None and eff["log"]:
        log.extend(eff["log"])
    if _fw.CAPTURE_LOG is not None and eff["capture"]:
        _fw.CAPTURE_LOG.extend(eff["capture"])
    if side is not None and eff["side"] is not None:
        side.clear()
        side.update(eff["side"])
