"""One thread per process for every parallel path: TokaMaker (``nthreads``) and Python's numeric stack.

TokaMaker at ``nthreads>1`` reduces in a non-deterministic order (about 1 % jitter in li_1) and can hang
DLSODE on stiff slices, and BLAS/OpenMP pools sized to the node oversubscribe it once there is one solver
per core. Parallelism is across processes only.
"""
from __future__ import annotations

import contextlib
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

_PIN_KEYS = THREAD_VARS + ("HDF5_USE_FILE_LOCKING",)

#: the stack of active pins (innermost last): :func:`unpin_threads` undoes the last one
_active: list = []


def _threadpoolctl():
    """threadpoolctl, or an ImportError that says what needs it and how to get it."""
    try:
        import threadpoolctl
    except ImportError as exc:          # pragma: no cover - depends on the environment
        raise ImportError(
            "bouquet's parallel paths (parallel_generate / run_shard, parallel_cases, bouquet.units) pin "
            "every thread pool to one thread with threadpoolctl, which is a declared dependency of bouquet "
            "but is not installed in this interpreter: `python -m pip install threadpoolctl`") from exc
    return threadpoolctl


def worker_env(env=None) -> dict:
    """The environment of a worker process: every thread pool at 1, and HDF5 file locking off (shared
    inputs are only read, and filesystems without flock, CFS among them, refuse the lock).  Updates
    ``env`` (default ``os.environ``) in place and returns it."""
    env = os.environ if env is None else env
    env.update({v: "1" for v in THREAD_VARS})
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    return env


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@contextlib.contextmanager
def worker_environment():
    """:func:`worker_env` on ``os.environ`` for the duration (spawned children inherit it), then restored."""
    saved = {k: os.environ.get(k) for k in _PIN_KEYS}
    worker_env()
    try:
        yield
    finally:
        _restore_env(saved)


class ThreadPin:
    """One thread per pool in THIS process, reversibly (:func:`pin_threads`).

    Pinning sets the :func:`worker_env` variables (they bind libraries loaded from now on) and limits the
    pools already loaded to one thread (threadpoolctl).  :meth:`unpin` -- or leaving the ``with`` block --
    restores both: the environment variables to what they were (unset ones unset again) and every pool to
    its previous thread count.  A spawned worker or a CLI process may simply stay pinned; an in-process
    caller (a test calling ``run_shard``) must not leave the interpreter pinned.

    ``report`` is :func:`thread_report` right after pinning: the pinned thread count of every loaded pool,
    for the case record."""

    def __init__(self):
        tpc = _threadpoolctl()
        self._saved_env = {k: os.environ.get(k) for k in _PIN_KEYS}
        worker_env()
        self._limits = tpc.threadpool_limits(limits=1)
        self.pinned = True
        self.report = thread_report()
        _active.append(self)

    def unpin(self) -> None:
        """Restore the environment variables and the pools' thread counts (idempotent)."""
        if not self.pinned:
            return
        self.pinned = False
        try:
            self._limits.restore_original_limits()
        finally:
            _restore_env(self._saved_env)
            if self in _active:
                _active.remove(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unpin()
        return False


def pin_threads() -> ThreadPin:
    """Hold this process to one thread (see :class:`ThreadPin`): the environment for libraries loaded from
    now on, and threadpoolctl for the pools already loaded (the environment binds only at load time).

    Returns the pin: ``with pin_threads(): ...`` or ``p = pin_threads(); ...; p.unpin()`` restores the
    previous state; :func:`unpin_threads` undoes the most recent pin."""
    return ThreadPin()


def unpin_threads() -> bool:
    """Undo the most recent :func:`pin_threads` still active; ``False`` when none is."""
    if not _active:
        return False
    _active[-1].unpin()
    return True


def pinned() -> bool:
    """Whether a :func:`pin_threads` is active in this process."""
    return bool(_active)


def thread_report() -> list:
    """``[{"library", "num_threads"}]`` for every thread pool loaded in this process (threadpoolctl; a
    pool it cannot see, e.g. Apple Accelerate, is not listed -- the environment variables hold it)."""
    return [{"library": p.get("internal_api"), "num_threads": p.get("num_threads")}
            for p in _threadpoolctl().threadpool_info()]


def require_single_thread(n, what):
    if int(n) != 1:
        raise ValueError(f"{what}={n}: parallel runs are one thread per process (TokaMaker at nthreads>1 is "
                         "not reproducible and can hang); use more workers instead")
