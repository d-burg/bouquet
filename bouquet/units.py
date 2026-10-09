"""Case-level parallel runs: many bouquets (one per baseline) claimed from a shared-filesystem queue.

    python -m bouquet.units add    --queue Q units.json     # [{"id": ..., "config": BouquetConfig.to_dict()}]
    python -m bouquet.units work   --queue Q --out-dir D --workers N   # one launcher per node
    python -m bouquet.units status --queue Q
    python -m bouquet.units stop|drain|undrain --queue Q
    python -m bouquet.units retry-failed --queue Q --out-dir D [--signature S]

A unit is one bouquet on one baseline (``bouquet.Bouquet(config)``). Workers (``workqueue``) claim units
across processes, nodes and jobs; each unit runs in a fresh single-threaded process (:mod:`bouquet.threads`)
in its worker's directory, on its worker's copy of the mesh: ``setup_solver -> prepare_baseline ->
generate -> filter``. The archive is built in a scratch dir and moved to ``<out_dir>/<id>.h5``; the unit
record ``<out_dir>/<id>.json`` (status ``done`` or ``failed``) is written last, so a reader never sees a
partial archive. :func:`make_units` builds units from one config and per-unit overrides (e.g. a time
sweep), :func:`units_from_cases` from the cases :func:`bouquet.parallel.parallel_cases` takes (which runs
them on one node instead). The seed of each unit is derived from the config seed and its id (from the
case, :func:`bouquet.parallel.case_seed`, for :func:`units_from_cases`). Input paths may be relative to
the directory units are made in; queue and output paths to the directory ``work`` starts in.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback

from . import workqueue as wq
from .threads import worker_env

__all__ = ["make_units", "units_from_cases", "add_units", "run_unit", "unit_seed", "archive_counts", "work",
           "record_queue_failures"]


def unit_seed(seed, uid):
    """Seed of unit ``uid``: independent of which worker runs it, different across units."""
    return int(hashlib.sha256(f"{seed}_{uid}".encode()).hexdigest()[:8], 16)


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"no config field {dotted!r}")
    d[keys[-1]] = value


def make_units(config, overrides, ids):
    """Units from one :class:`~bouquet.config.BouquetConfig` and per-unit overrides.

    ``overrides``: one dict per unit of dotted config fields, e.g.
    ``{"source.time": 2.4, "source.ida_time": 2.413, "source.LCFS_geqdsk": g, "generation.scan_key": 2413}``.
    ``ids``: unit ids (``[a-zA-Z0-9_.-]``). A unit's seed is :func:`unit_seed` of the config seed unless its
    overrides set ``generation.seed``. Each unit's config is validated here (``BouquetConfig.from_dict``)
    and stored with absolute input paths.
    """
    from .config import BouquetConfig, with_absolute_paths
    base = config.to_dict()
    units = []
    for uid, over in zip(ids, overrides, strict=True):
        d = copy.deepcopy(base)
        for k, v in over.items():
            _set(d, k, v)
        if "generation.seed" not in over:
            d["generation"]["seed"] = unit_seed(d["generation"].get("seed", 0), uid)
        units.append(dict(id=uid, config=with_absolute_paths(BouquetConfig.from_dict(d)).to_dict()))
    return units


def units_from_cases(config, cases, ids=None):
    """One unit per :class:`~bouquet.config.CaseSpec` (or a ``ParallelSource``, expanded) on the run-level
    ``config``, as for :func:`bouquet.parallel.parallel_cases`.  ``ids`` default to
    ``<index>_<group>_<scan_key>``."""
    import re

    from .config import _encode_source
    from .parallel import case_seed, check_cases
    cases = list(cases.expand() if hasattr(cases, "expand") else cases)
    check_cases(cases, config)
    if ids is None:
        ids = [re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{i:04d}_{c.group}_{c.scan_key}") for i, c in enumerate(cases)]
    return [dict(u, group=c.group) for u, c in zip(make_units(config, [
        {"source": _encode_source(c.source), "generation.scan_key": c.scan_key,
         "generation.seed": case_seed(config.generation.seed, c)} for c in cases], ids), cases)]


def add_units(queue, units):
    """Append units to the queue at ``queue``; returns how many were new."""
    return wq.Queue(queue).add_units(units)


def archive_counts(h5):
    """(all draw counts, selected counts, profile_coord) of an archive."""
    import h5py
    counts, sel, coord = [], [], None
    with h5py.File(h5, "r") as f:
        for key in f["scan"]:
            sc = f["scan"][key]
            for c in sc:
                if c.isdigit():
                    counts.append(int(c))
                    if bool(sc[c].attrs.get("selected", False)):
                        sel.append(int(c))
                    coord = coord or str(sc[c].attrs.get("profile_coord", "psi_n"))
    return sorted(counts), sorted(sel), coord


def write_record(out_dir, uid, status, **info):
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(unit_id=uid, status=status, archive=None, host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID", ""), error=None)
    rec.update(info)
    wq.write_json(os.path.join(out_dir, f"{uid}.json"), rec)
    return rec


def run_unit(queue, uid, out_dir, work_root=None, keep_work=False, mesh=None):
    """Run one unit in this process (one thread, see :mod:`bouquet.threads`); returns its record.  ``mesh``:
    a worker-local copy of the mesh to use.  Raises on failure (no record written)."""
    from .threads import pin_threads
    with pin_threads():          # undone on return: an in-process caller is not left pinned
        return _run_unit_pinned(queue, uid, out_dir, work_root=work_root, keep_work=keep_work, mesh=mesh)


def _run_unit_pinned(queue, uid, out_dir, work_root=None, keep_work=False, mesh=None):
    from . import __version__
    from .config import BouquetConfig, CaseSpec
    from .parallel import check_cases, release_dd_cache, run_case_on
    from .paths import add_oft_to_path
    from .run import Bouquet
    add_oft_to_path()
    unit = wq.Queue(queue).units()[uid]
    d = unit["config"]
    work = os.path.join(work_root or os.path.join(out_dir, "_work"), f"{uid}.{socket.gethostname()}.{os.getpid()}")
    os.makedirs(work)
    cfg = BouquetConfig.from_dict(d)
    if mesh:
        cfg.solver.mesh_path = mesh
    case = CaseSpec(source=cfg.source, header=os.path.join(work, uid), scan_key=cfg.generation.scan_key,
                    group=unit.get("group"))
    check_cases([case], cfg)
    t0 = time.time()
    # one unit per process: nothing reuses the parsed dd after the baseline
    res = run_case_on(Bouquet(cfg).setup_solver(), case, case.header,
                      hooks={"after_baseline": release_dd_cache})
    counts, sel, coord = archive_counts(res["path"])
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"{uid}.h5.tmp.{socket.gethostname()}.{os.getpid()}")
    shutil.copyfile(res["path"], tmp)
    os.replace(tmp, os.path.join(out_dir, f"{uid}.h5"))
    rec = write_record(out_dir, uid, "done", archive=f"{uid}.h5", group=case.group, scan_key=case.scan_key,
                       n_draws=len(counts), n_selected=len(sel), all_counts=counts, selected_counts=sel,
                       profile_coord=coord, seed=cfg.generation.seed, bouquet_version=__version__,
                       config_hash=hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12],
                       F0=res["F0"], threads=res["threads"], started=t0, finished=time.time(),
                       wall_s=round(time.time() - t0, 1))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return rec


class _Runner:
    """``make_run`` for :func:`workqueue.launch`: each unit as a ``run-unit`` child process."""

    def __init__(self, queue, out_dir, work_root, timeout, min_left, keep_work):
        # absolute: each child runs in its worker's directory
        self.queue, self.out_dir = os.path.abspath(queue), os.path.abspath(out_dir)
        self.work_root = os.path.abspath(work_root) if work_root else None
        self.timeout, self.min_left, self.keep_work = timeout, min_left, keep_work
        self.deadline = wq.job_deadline(0.0)
        # the child runs in its worker's directory: it imports this bouquet through PYTHONPATH
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env = worker_env(dict(os.environ))
        self.env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_root, os.environ.get("PYTHONPATH")) if p)

    def __call__(self, q, worker):
        # each worker: its own cwd (TokaMaker scratch files) and its own copy of the mesh, made once
        wdir = os.path.join(self.work_root or os.path.join(self.out_dir, "_work"),
                            f"{worker}.{socket.gethostname()}")
        os.makedirs(wdir, exist_ok=True)
        meshes = {}

        def local_mesh(src):
            if src not in meshes:       # named by its source path: two meshes may share a basename
                tag = hashlib.sha256(os.path.abspath(src).encode()).hexdigest()[:8]
                meshes[src] = os.path.join(wdir, f"{tag}_{os.path.basename(src)}")
                shutil.copy2(src, meshes[src])
            return meshes[src]

        def run(uid, unit):
            if self.deadline and self.deadline - time.time() < self.min_left:
                raise wq.Requeue("too little walltime left")
            log = os.path.join(self.out_dir, "logs", f"{uid}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "a") as fh:
                fh.write(f"\n=== {time.ctime()} {worker} attempt {q.n_attempts(uid)}\n")
            res = os.path.join(q.root, "results", f"{uid}.{worker}.{q.n_attempts(uid)}.json")
            argv = [sys.executable, "-m", "bouquet.units", "run-unit", "--queue", self.queue, "--unit", uid,
                    "--out-dir", self.out_dir, "--result", res,
                    "--mesh", local_mesh(unit["config"]["solver"]["mesh_path"])] \
                + (["--work-root", self.work_root] if self.work_root else []) + (["--keep-work"] if self.keep_work else [])
            try:
                return wq.run_child(argv, log, timeout=self.timeout, result_path=res, env=self.env, cwd=wdir)
            except Exception as e:
                rec = wq.read_json(os.path.join(self.out_dir, f"{uid}.json")) or {}
                if q.n_failures(uid) + 1 >= q.MAX_FAILURES and rec.get("status") != "done":
                    write_record(self.out_dir, uid, "failed",
                                 error=dict(signature=wq.error_signature(e), message=str(e)[:2000], log=log))
                raise
        return run


def record_queue_failures(queue, out_dir):
    """Write the ``failed`` record of every unit the queue failed without running it to the end (a crash,
    preemption or timeout on every attempt), so a reader waiting for ``<out_dir>/<id>.json`` gets one."""
    q = wq.Queue(queue)
    for uid in q.done_ids():
        o = q.outcome(uid) or {}
        if o.get("status") == "failed" and \
                (wq.read_json(os.path.join(out_dir, f"{uid}.json")) or {}).get("status") not in ("done", "failed"):
            write_record(out_dir, uid, "failed", error=dict(signature=o.get("reason", "unknown"), message=None,
                                                            log=os.path.join(out_dir, "logs", f"{uid}.log")))


def work(queue, out_dir, workers, work_root=None, timeout=None, min_left=0.0, margin=300.0, idle_exit=600.0,
         keep_work=False):
    """Run ``workers`` worker loops on this node until the queue is drained, then write the records of
    units the queue failed (:func:`record_queue_failures`); returns the workers' exit reasons."""
    queue, out_dir = os.path.abspath(queue), os.path.abspath(out_dir)
    reasons = wq.launch(queue, workers, _Runner(queue, out_dir, work_root, timeout, min_left, keep_work),
                        deadline=wq.job_deadline(margin), idle_exit_s=idle_exit)
    record_queue_failures(queue, out_dir)
    return reasons


def _cmd_run_unit(a):
    res = a.result or os.path.join(wq.Queue(a.queue).root, "results", f"{a.unit}.json")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    try:
        rec = run_unit(a.queue, a.unit, a.out_dir, a.work_root, a.keep_work, a.mesh)
        wq.write_json(res, dict(ok=True, n_selected=rec["n_selected"]))
    except BaseException as e:
        traceback.print_exc()
        wq.write_json(res, dict(error=dict(type=type(e).__name__, signature=wq.error_signature(e))))
        sys.exit(1)


def _cmd_status(a):
    s = wq.Queue(a.queue).status()
    print(f"units {s['total']}: done {s['done']}, failed {s['failed']}, running {s['running']}, "
          f"stale {s['stale']}, open {s['open']}" + (" [STOP]" if s["stopped"] else "")
          + (" [DRAIN]" if s["draining"] else ""))
    for sig, v in sorted(s["failures"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {v['n']:4d} x {sig}  (e.g. {v['units'][0]})")


def _cmd_retry_failed(a):
    q = wq.Queue(a.queue)
    n = 0
    for uid in q.done_ids():
        o = q.outcome(uid) or {}
        if o.get("status") == "failed" and a.signature in json.dumps(o):
            q.reopen(uid)
            if os.path.exists(rec := os.path.join(a.out_dir, f"{uid}.json")):
                os.remove(rec)          # the stale failed record: readers would take it as final
            n += 1
    print(f"retry-failed: {n} unit(s) reopened")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bouquet.units", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.add_argument("--queue", required=True)
        p.set_defaults(fn=fn)
        return p

    def cmd_add(a):
        with open(a.units) as fh:
            print(f"add: {add_units(a.queue, json.load(fh))} new unit(s)")

    add("add", cmd_add).add_argument("units", help="JSON list of {id, config}")
    p = add("work", lambda a: print(work(a.queue, a.out_dir, a.workers, a.work_root, a.timeout, a.min_left,
                                         a.margin, a.idle_exit, a.keep_work)))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--work-root", default=None)
    p.add_argument("--timeout", type=float, default=None, help="s per unit")
    p.add_argument("--min-left", type=float, default=0.0, help="s of walltime needed to start a unit")
    p.add_argument("--margin", type=float, default=300.0, help="s before the job end to stop claiming")
    p.add_argument("--idle-exit", type=float, default=600.0)
    p.add_argument("--keep-work", action="store_true")
    p = add("run-unit", _cmd_run_unit)
    p.add_argument("--unit", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--result", default=None, help="result JSON for the parent worker")
    p.add_argument("--mesh", default=None, help="worker-local mesh copy")
    p.add_argument("--work-root", default=None)
    p.add_argument("--keep-work", action="store_true")
    add("status", _cmd_status)
    for name in ("stop", "drain"):
        add(name, lambda a: wq.Queue(a.queue).set_flag(a.cmd.upper(), "user"))
    add("undrain", lambda a: os.remove(wq.Queue(a.queue).path("DRAIN")))
    p = add("retry-failed", _cmd_retry_failed)
    p.add_argument("--signature", default="")
    p.add_argument("--out-dir", required=True, help="the reopened units' failed records are removed here")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
