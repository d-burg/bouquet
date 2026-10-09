"""Process-parallel bouquets: the draws of one bouquet (:func:`parallel_generate`) or many bouquets on
one node (:func:`parallel_cases`).

``OFT_env`` is a per-process singleton, so a second TokaMaker cannot live in the
same interpreter -- parallelism is across **processes**, each with its own
solver. Draws are embarrassingly parallel, and the baseline forward-solve is
**bit-identical across processes at ``nthreads=1``** (verified: a fresh process
reproduces li_1/li_3/Ip/psi to 0), so every worker simply runs the ordinary
serial path (``setup_solver -> prepare_baseline -> generate``) on its shard of
``n_equils`` and the per-worker archives are concatenated.

Two launchers, one ``run_shard`` entry point:

* **laptop / single node** -- :func:`parallel_generate` drives a
  ``ProcessPoolExecutor`` (spawn), one single-threaded worker per core
  (:mod:`bouquet.threads`).
* **cluster** -- :func:`emit_slurm_script` writes a SLURM job-array + dependent
  merge job that call ``python -m bouquet.parallel`` on the same ``run_shard``.

Note: parallel draws are NOT bit-identical to a serial run of the same seed --
each worker's ``GenerationConfig.seed`` (and hence the single
``numpy.random.Generator`` its ``generate_bouquet`` builds via
``sampling.make_rng``) is derived from ``(seed, worker_id, scan_key)`` via
``np.random.SeedSequence`` (see :func:`_derive_seed`), so the union is a
statistically-equivalent *different* draw set. The baseline is identical.
Folding the ``scan_key`` into the derivation decorrelates a multi-slice sweep
run with one ``seed``: draw *i* of slice A and draw *i* of slice B no longer
share a perturbation stream.

The parallel run IS reproducible as a whole: the derivation is a pure function
of ``(seed, worker_id, scan_key)``, so re-running the same ``seed`` over the
same ``n_workers`` regenerates every shard's draws bitwise. Changing
``n_workers`` re-partitions the draws and therefore changes the ensemble --
record it alongside the seed.

**Shared until-N.**  ``generation.n_inspec_target`` is honoured on both
launchers through a *shared yield ledger*: every worker records each draw that
passes the in-loop filters, and checks the pooled count at the top of every
attempt, stopping once the run's ONE target is met -- a cooperative stop at the
attempt boundary, never a kill, so every archived draw is complete.  The
laptop pool shares a ``multiprocessing.Manager`` counter; the SLURM array
shares an append-only file on the job's filesystem.  Up to one extra in-spec
draw per worker can land after the threshold (a draw in flight is finished,
not discarded), so the merged archive holds *at least* the target.  The
attempt cap ``max_total_draws`` is split across workers the same way
``n_equils`` is, so a zero-yield configuration still terminates.

A shared-stop run is reproducible in a different sense from a fixed-``n``
one: each draw is still a pure function of ``(seed, worker_id, scan_key,
index)``, but WHICH draws exist depends on worker timing.  The merged archive
therefore carries a manifest (scan-group attr ``parallel_manifest_json``) with
every worker's attempt count; replaying those counts as fixed per-worker
allocations regenerates the archive exactly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

from .threads import pin_threads, require_single_thread, worker_env, worker_environment

__all__ = [
    "run_shard",
    "merge_archives",
    "parallel_generate",
    "emit_slurm_script",
    "apply_filters_after_merge",
    "SharedYieldLedger",
    "ManagerYieldLedger",
    "FileYieldLedger",
    "shared_until_n_budget",
    "parallel_cases",
    "run_case",
    "run_case_on",
    "case_seed",
    "merge_cases",
    "check_cases",
]


def _shard_size(total, n_workers, worker_id):
    """Draws assigned to *worker_id* when *total* is split over *n_workers*."""
    base, rem = divmod(int(total), int(n_workers))
    return base + (1 if worker_id < rem else 0)


def _derive_seed(seed_base, worker_id, scan_key):
    """Per-shard seed from ``(seed_base, worker_id, scan_key)``.

    The returned int becomes the shard's ``GenerationConfig.seed``, which
    ``generate_bouquet`` turns into that shard's one
    ``numpy.random.Generator`` (``sampling.make_rng``). Shards are therefore
    *independently* seeded but *deterministically derived* from the master
    seed: same triple -> same shard, always.

    ``SeedSequence`` scrambles the entropy tuple into a well-separated 32-bit
    seed, replacing the old ``seed_base + worker_id`` scheme, which (a) gave
    adjacent-integer seeds across workers and (b) reused the *same* streams
    for every slice of a timeseries swept with one ``seed_base`` --
    correlating draw *i* across time slices. The ``scan_key`` enters through
    its canonical string form (``utils._scan_key``), so ``2000`` and
    ``"2000"`` derive the same stream (mirroring the archive layout).
    """
    import numpy as np
    from .utils import _scan_key

    entropy = [int(seed_base), int(worker_id)]
    bkey = _scan_key(scan_key)
    if bkey is not None:
        entropy.append(int.from_bytes(
            hashlib.sha256(bkey.encode()).digest()[:8], "little"))
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _get_num_cpus(use_logical=True):
    """Worker count: the CPUs this process may use (its affinity mask, so a 4-CPU allocation on a
    128-core node gives 4), capped by the scheduler's CPU count (SLURM, PBS, LSF, SGE).
    ``use_logical=False``: one worker per physical core (Linux sysfs), to keep SMT siblings idle."""
    try:
        affinity = os.sched_getaffinity(0)
        n_logical = len(affinity)
    except AttributeError:                      # macOS / Windows
        affinity, n_logical = None, os.cpu_count() or 1
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "LSB_DJOB_NUMPROC", "NSLOTS"):
        if os.environ.get(var):
            n_logical = min(n_logical, int(os.environ[var]))
            break
    if use_logical or affinity is None:
        return n_logical
    cores = set()
    for cpu in affinity:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id") as f:
                pkg = f.read().strip()
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id") as f:
                cores.add((pkg, f.read().strip()))
        except OSError:
            pass
    return min(len(cores), n_logical) if cores else n_logical


# --------------------------------------------------------------------------
#  worker: generate one shard
# --------------------------------------------------------------------------
class SharedYieldLedger:
    """Pooled in-spec count shared by every worker of ONE until-N run.

    ``record()`` is called by a worker for each draw that passes the in-loop
    filters; ``count()`` returns the pooled total; ``reached(target)`` is the
    worker's stop test.  Backends differ only in where the count lives.
    """

    def record(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def reached(self, target) -> bool:
        return self.count() >= int(target)

    def reset(self) -> None:
        """Start a fresh run (no-op where the backend is created fresh)."""


class ManagerYieldLedger(SharedYieldLedger):
    """Laptop pool: a ``multiprocessing.Manager`` counter behind a lock.

    Both proxies pickle into spawned workers exactly as the progress queue
    does; the parent owns the manager for the life of the pool.
    """

    def __init__(self, manager):
        self._v = manager.Value("i", 0)
        self._lock = manager.Lock()

    def record(self) -> None:
        with self._lock:
            self._v.value += 1

    def count(self) -> int:
        return int(self._v.value)

    def reset(self) -> None:
        with self._lock:
            self._v.value = 0


class FileYieldLedger(SharedYieldLedger):
    """SLURM array: one line appended per in-spec draw to a shared file.

    Each ``record()`` is a single ``O_APPEND`` write of two bytes, which
    stays atomic across the array's tasks on a shared filesystem; ``count()``
    is the line count.  On NFS the visible count may lag other nodes' appends
    by seconds -- that lag only delays the stop, i.e. adds overshoot, which
    the merged archive keeps anyway.  ``submit.sh`` truncates the file before
    the array starts so a previous run's ledger can never pre-satisfy the
    target.
    """

    def __init__(self, path):
        self.path = os.path.abspath(str(path))

    def record(self) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, b"1\n")
        finally:
            os.close(fd)

    _RECORD_BYTES = 2                     # b"1\n"

    def count(self) -> int:
        # every record() appends exactly _RECORD_BYTES, so the count is the
        # file size: O(1) per call, where a re-read would be O(N) per attempt
        # and O(N^2) over a run on a shared filesystem
        try:
            return int(os.stat(self.path).st_size) // self._RECORD_BYTES
        except FileNotFoundError:
            return 0

    def reset(self) -> None:
        with open(self.path, "wb"):
            pass


def ledger_path_for(out_header) -> str:
    """The SLURM ledger file that belongs to ``{out_header}.h5``."""
    return os.path.abspath(f"{out_header}_inspec.ledger")


def shared_until_n_budget(n_inspec_target, max_total_draws, n_equils_total,
                          n_workers, worker_id) -> dict:
    """Per-worker attempt budget for a shared until-N run.

    The run's attempt cap (``max_total_draws``, default
    ``max(n_equils_total, 5 * target)`` exactly as the serial rule) is split
    over workers with :func:`_shard_size`, so the shares sum to the cap and a
    zero-yield configuration still terminates.  The worker's LOCAL
    ``n_inspec_target`` is ``min(target, share)``: a worker can never see more
    in-spec draws than attempts, so this keeps the serial validation
    (``max_total_draws >= n_inspec_target``) satisfied while the real stop is
    the shared ledger.  Returns ``dict(total_cap, cap, local_target)``.
    """
    # the same integer-count validation Bouquet.generate() applies: a float
    # or bool target is refused, never truncated
    from .config import require_integer_count
    require_integer_count(n_inspec_target, "generation.n_inspec_target")
    require_integer_count(max_total_draws, "generation.max_total_draws")
    tgt = int(n_inspec_target)
    if tgt < 1:
        raise ValueError("n_inspec_target must be >= 1")
    total_cap = (int(max_total_draws) if max_total_draws is not None
                 else max(int(n_equils_total), 5 * tgt))
    if total_cap < tgt:
        raise ValueError(
            f"max_total_draws={total_cap} is below n_inspec_target={tgt}; "
            "the shared target could never be met")
    share = _shard_size(total_cap, n_workers, worker_id)
    return dict(total_cap=total_cap, cap=int(share),
                local_target=min(tgt, int(share)))


def run_shard(config, worker_id, n_workers, *, n_equils_total, seed_base,
              out_header, scan_key, threads_per_worker, verbose=False,
              progress_q=None, ledger=None):
    """Generate worker *worker_id*'s shard of draws in THIS process.

    Builds its own TokaMaker (own ``OFT_env``), forward-solves the baseline, and
    writes its draws to ``{out_header}_w{worker_id}.h5``. Returns a metadata dict
    (shard path, count, baseline ``li``/``Ip``) consumed by the merge and the
    cross-worker baseline check. A worker assigned zero draws is a no-op.

    ``verbose=False`` (default) captures the worker's native solver/mesh chatter
    (otherwise N workers x every slice floods the parent's stdout); set True to
    stream it for debugging.
    """
    tgt = getattr(config.generation, "n_inspec_target", None)
    budget = None
    if tgt is not None:
        if ledger is None:
            raise ValueError(
                f"generation.n_inspec_target={tgt} on run_shard needs a shared "
                "ledger (parallel_generate / the SLURM CLI supply one): N "
                "workers each chasing the target alone would deliver "
                "N*target draws.")
        budget = shared_until_n_budget(
            tgt, getattr(config.generation, "max_total_draws", None),
            n_equils_total, n_workers, worker_id)
        n = budget["cap"]
    else:
        n = _shard_size(n_equils_total, n_workers, worker_id)
    if n == 0:
        return dict(worker_id=worker_id, path=None, n=0, n_attempts=0,
                    n_inspec=0, li_target=None, Ip_target=None)

    # Silence the worker's entire stdout/stderr at the fd level BEFORE importing
    # OFT, which caches fd 1 at init -- a later redirect leaks the banner + N x
    # the mesh/solver chatter into the parent (notebook cell). The shard result
    # returns through the pool and exceptions propagate as pickled objects, so
    # nothing needed for results or debugging is lost. verbose=True streams it.
    _saved = None
    _pin = None
    if not verbose:
        _dn = os.open(os.devnull, os.O_WRONLY)
        _saved = (os.dup(1), os.dup(2))
        os.dup2(_dn, 1)
        os.dup2(_dn, 2)
        os.close(_dn)
    try:
        # one thread for the shard, undone on return: a spawned worker loses
        # nothing, and an in-process caller (tests) is not left pinned
        _pin = pin_threads()
        import bouquet as bq
        cfg = copy.deepcopy(config)
        cfg.solver.nthreads = 1
        cfg.generation.n_equils = int(n)
        if budget is not None:
            # This worker's share of the run's attempt cap, and a local
            # target the serial validation accepts; the shared ledger is
            # the stop that matters (see shared_until_n_budget).
            cfg.generation.max_total_draws = int(budget["cap"])
            cfg.generation.n_inspec_target = int(budget["local_target"])
            print(f"[until-N] worker {worker_id}: shared target {int(tgt)} "
                  f"across {n_workers} workers; this worker's attempt cap "
                  f"{budget['cap']} of {budget['total_cap']} total.")
        # Independent, slice-decorrelated, deterministically-derived stream:
        # generate_bouquet consumes this into the shard's single Generator
        # (see _derive_seed and sampling.make_rng), so re-running the same
        # (seed_base, n_workers, scan_key) regenerates the shard bitwise.
        cfg.generation.seed = _derive_seed(seed_base, worker_id, scan_key)
        cfg.generation.scan_key = scan_key
        cfg.output_header = f"{out_header}_w{worker_id}"

        # Fresh shard: the archive writer opens the h5 in append mode and only
        # deletes the draw groups it rewrites, so a leftover shard from a
        # previous run (crash before cleanup, cancelled SLURM array, or a
        # re-run with fewer draws per worker) would keep its stale
        # higher-index draws -- and merge_archives copies every draw group it
        # finds. Remove any pre-existing file so the shard holds ONLY this
        # run's draws.
        _shard_h5 = os.path.abspath(f"{cfg.output_header}.h5")
        if os.path.exists(_shard_h5):
            os.remove(_shard_h5)

        # per-draw progress -> parent aggregate bar (one tick per draw
        # attempt); the same tick counts this worker's attempts for the
        # manifest, which is what makes a shared-stop run replayable.
        _attempts = [0]

        def cb(_count, _q=progress_q, _w=worker_id):
            _attempts[0] = int(_count) + 1
            if _q is not None:
                try:
                    _q.put(_w)
                except Exception:
                    pass

        _inspec = [0]
        on_inspec = stop_check = None
        if budget is not None:
            def on_inspec(_l=ledger):
                _inspec[0] += 1
                _l.record()

            def stop_check(_l=ledger, _t=int(tgt)):
                return _l.reached(_t)

        b = bq.Bouquet(cfg)
        b.setup_solver()
        b.prepare_baseline()
        # the baseline holds what it needs: release the parsed dd(s) the
        # read cached (#72 B3), so a pool worker's resident memory is what
        # it was before the shared cache (a later read re-parses)
        from .io.imas import clear_dd_cache
        clear_dd_cache()
        b.generate(progress_callback=cb, on_inspec=on_inspec,
                   stop_check=stop_check)
        rec = dict(worker_id=int(worker_id), path=f"{cfg.output_header}.h5",
                   n=int(n), n_attempts=int(_attempts[0]),
                   n_inspec=int(_inspec[0]),
                   seed=int(cfg.generation.seed),
                   li_target=float(b.baseline.l_i_target),
                   Ip_target=float(b.baseline.Ip_target),
                   threads=list(_pin.report))
        if budget is not None:
            rec.update(shared_target=int(tgt), local_target=budget["local_target"],
                       attempt_cap=budget["cap"], total_cap=budget["total_cap"])
            # the LCFS bound this worker's in-spec count was taken against,
            # as generate() stamped it on the shard (the value the loop used)
            from .utils import read_generation_provenance
            try:
                _gp = read_generation_provenance(rec["path"], scan_key=scan_key)
            except OSError:
                _gp = {}
            rec.update({k: _gp.get(k) for k in _INLOOP_CUT_KEYS})
        # Stamp the worker record on the shard so the merge (either launcher)
        # can build the run manifest from the shards alone.
        _write_worker_record(rec["path"], scan_key, rec)
        return rec
    finally:
        if _pin is not None:
            _pin.unpin()
        if _saved is not None:
            os.dup2(_saved[0], 1)
            os.dup2(_saved[1], 2)
            os.close(_saved[0])
            os.close(_saved[1])


#: the until-N loop's boundary cut, as stamped on each shard's scan group
_INLOOP_CUT_KEYS = ("inspec_rms_max_mm", "inspec_max_max_mm", "inspec_cut_source")


def _attr_value(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if hasattr(v, "item") else v


def _write_worker_record(shard_path, scan_key, rec):
    """Attr ``parallel_worker_json`` on the shard's scan group (or root)."""
    import h5py
    from .utils import _scan_key
    bkey = _scan_key(scan_key)
    gp = f"scan/{bkey}" if bkey is not None else "/"
    try:
        with h5py.File(shard_path, "a") as hf:
            if gp not in hf:
                hf.require_group(gp)
            hf[gp].attrs["parallel_worker_json"] = json.dumps(
                {k: v for k, v in rec.items() if k != "path"})
    except OSError as exc:            # a missing shard is the caller's problem
        print(f"WARN: could not stamp worker record on {shard_path}: {exc}")


def _read_worker_record(src, base_path):
    parent = src[base_path] if base_path else src
    raw = parent.attrs.get("parallel_worker_json", None)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
#  merge per-worker shards into one archive
# --------------------------------------------------------------------------
def merge_archives(shard_paths, out_header, scan_key=None, *, cleanup=False,
                   baseline_match_rtol=1e-6, config=None, missing_workers=None,
                   rescued="include"):
    """Concatenate per-worker shard archives into ``{out_header}.h5``.

    Draw groups are renumbered to a contiguous running index; under schema v2
    the raw bytes keep their fixed ``eqdsk`` / ``pfile`` dataset names, so no
    rename is needed (the group path carries the coordinates). The
    ``_baseline`` group is copied once (the baseline is identical across
    workers). Returns ``(out_path, n_draws)``.

    Pass the RUN-level ``config`` (the original :class:`~bouquet.BouquetConfig`,
    not a per-worker mutation) to stamp ``config_json`` provenance onto the
    merged archive -- the shards' own provenance records per-worker configs
    (derived seed, shard n_equils, ``_w{i}`` header) and is not copied, and
    the shards themselves are deleted under ``cleanup=True``, so without this
    the deliverable file carries no config record.

    Every shard's stored baseline (``_baseline`` attrs ``l_i_target`` /
    ``Ip_target``) is verified against the first shard's before anything is
    copied; a drift beyond ``baseline_match_rtol`` raises. This is the merge's
    own guard -- unlike the pre-merge check in :func:`parallel_generate` it
    also covers the SLURM CLI path, where drifted baselines (heterogeneous
    nodes, a stray ``nthreads>1``) would otherwise merge silently, mixing
    draws accepted against different l_i targets. Shards whose baselines differ
    in ``profile_coord`` (ψ_N vs Φ_N grids) also raise. A listed shard that does not
    exist on disk raises (missing workers must be handled by the caller, not
    dropped silently).

    ``missing_workers`` (a list of worker ids the caller knowingly left out,
    e.g. the CLI's ``--allow-missing``) marks the result as a PARTIAL merge:
    the manifest and the scan group carry ``merge_partial_json`` and
    ``n_requested_source`` says so, since the run-level request was not
    delivered by the shards merged. Shards whose worker records disagree on
    the generation mode or the shared target, or that counted against
    different boundary cuts, are refused before anything is written.

    ``rescued`` (owner decision D5): draws whose solve was rescued by the
    opt-in draw-solve rescue (stamped ``solve_recovered``) are merged by
    default (``"include"``, counted and warned); ``"exclude"`` leaves them
    out.  Either way, when any shard holds one, the scan group records
    ``merge_rescued_json`` (the policy, the count and each rescued draw's
    shard, index and ``recovered_by``).
    """
    if rescued not in ("include", "exclude"):
        raise ValueError(f"rescued must be 'include' or 'exclude', got "
                         f"{rescued!r}")
    import warnings
    import h5py
    from .utils import (initialize_equilibrium_database, _scan_key,
                        _group_path, group_coord)

    bkey = _scan_key(scan_key)
    base_path = f"scan/{bkey}" if bkey is not None else None
    bl_dst = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"

    # ---- pre-pass: shard existence + cross-shard baseline consistency ----
    def _baseline_targets(src):
        parent = src[base_path] if base_path else src
        if "_baseline" not in parent:
            return None
        a = parent["_baseline"].attrs
        if "l_i_target" not in a or "Ip_target" not in a:
            return None
        return float(a["l_i_target"]), float(a["Ip_target"])

    def _baseline_coord(src):
        parent = src[base_path] if base_path else src
        return (group_coord(parent["_baseline"]) if "_baseline" in parent
                else None)

    def _inloop_cut(src):
        parent = src[base_path] if base_path else src
        a = parent.attrs
        return tuple(_attr_value(a[k]) if k in a else None for k in _INLOOP_CUT_KEYS)

    def _mode(src):
        rec = _read_worker_record(src, base_path)
        if rec is None:
            return None
        return (("until_n", rec.get("shared_target"), rec.get("total_cap"))
                if "shared_target" in rec else ("fixed", None, None))

    targets, shard_coords = [], {}
    cuts = {}
    modes = {}
    for sp in shard_paths:
        if sp is None:
            continue
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"shard archive not found: {sp}. Merging a partial set "
                "silently shrinks the bouquet -- re-run the missing worker, "
                "or drop the path explicitly from shard_paths.")
        with h5py.File(sp, "r") as src:
            targets.append((sp, _baseline_targets(src)))
            c = _baseline_coord(src)
            if c is not None:
                shard_coords[sp] = c
            cuts[sp] = _inloop_cut(src)
            modes[sp] = _mode(src)
    if len(set(shard_coords.values())) > 1:
        raise RuntimeError(
            f"shards differ in profile_coord: {shard_coords}. "
            "Nothing was merged.")
    _mode_set = {m for m in modes.values() if m is not None}
    if len(_mode_set) > 1:
        raise RuntimeError(
            "shards come from different generation set-ups (mode, shared "
            "target, total cap): "
            + "; ".join(f"{os.path.basename(sp)}: {m}" for sp, m in modes.items())
            + ". Nothing was merged.")
    # every shard's until-N count must have been taken against ONE boundary
    # cut (a device detected on one node and not on another would mix them)
    _distinct = {c for c in cuts.values() if c != (None, None, None)}
    if len(_distinct) > 1:
        raise RuntimeError(
            "shards counted their in-spec draws against different boundary "
            "cuts (inspec_rms_max_mm, inspec_max_max_mm, inspec_cut_source): "
            + "; ".join(f"{os.path.basename(sp)}: {c}" for sp, c in cuts.items())
            + ". Nothing was merged -- set filtering.rms_max_mm (or "
            "config.device) explicitly and re-run.")
    present = [(sp, t) for sp, t in targets if t is not None]
    unchecked = [sp for sp, t in targets if t is None]
    if unchecked and present:
        warnings.warn(
            "shard(s) without stored baseline targets (l_i_target/Ip_target "
            f"attrs) skipped the baseline consistency check: {unchecked}")
    if len(present) >= 2:
        sp0, (li0, ip0) = present[0]
        for sp, (li, ip) in present[1:]:
            if (abs(li - li0) > baseline_match_rtol * abs(li0) or
                    abs(ip - ip0) > baseline_match_rtol * abs(ip0)):
                raise RuntimeError(
                    f"shard baseline drifted: {sp} has l_i={li:.8f}, "
                    f"Ip={ip:.6e} vs {sp0} l_i={li0:.8f}, Ip={ip0:.6e} "
                    f"(rtol={baseline_match_rtol:g}). Workers did not "
                    "converge to the same baseline -- check nthreads=1 "
                    "(threads_per_worker=1), identical source/config, and "
                    "on SLURM that all array tasks ran on the same node "
                    "architecture. Nothing was merged.")

    out_path = os.path.abspath(f"{out_header}.h5")
    if os.path.exists(out_path):
        os.remove(out_path)                  # fresh archive (init opens append)
    initialize_equilibrium_database(out_header)

    offset = 0
    workers = []
    _resc = []                  # rescued draws met in the shards
    n_unrecorded = 0            # shards with no worker record (draws still merged)
    with h5py.File(out_path, "a") as out:
        if bkey is not None and base_path not in out:
            out.create_group(base_path)
        for sp in shard_paths:
            if sp is None:
                continue
            with h5py.File(sp, "r") as src:
                parent = src[base_path] if base_path else src
                if "_baseline" in parent and bl_dst not in out:
                    out.copy(parent["_baseline"], bl_dst)
                _wrec = _read_worker_record(src, base_path)
                if _wrec is not None:
                    _wrec["first_index"] = offset      # where its draws land
                    # the shard's own generation provenance (attempt outcomes)
                    _pa = (src[base_path] if base_path else src).attrs
                    _raw = _pa.get("attempt_outcomes_json", None)
                    if _raw is not None:
                        try:
                            _wrec["attempt_outcomes"] = json.loads(
                                _raw.decode() if isinstance(_raw, bytes) else str(_raw))
                        except (ValueError, TypeError):
                            pass
                    for _k in ("n_attempted", "bouquet_version", *_INLOOP_CUT_KEYS):
                        if _k in _pa:
                            _wrec[_k] = _attr_value(_pa[_k])
                    workers.append(_wrec)
                else:
                    n_unrecorded += 1
                idxs = sorted(
                    int(k) for k in parent.keys()
                    if k not in ("_baseline", "scan")
                    and str(k).lstrip("-").isdigit()
                )
                for i in idxs:
                    _ga = parent[str(i)].attrs
                    if bool(_ga.get("solve_recovered", False)):
                        _rb = _ga.get("solve_recovered_by", "")
                        _resc.append(dict(
                            shard=os.path.basename(sp), index=int(i),
                            recovered_by=(_rb.decode() if isinstance(_rb, bytes)
                                          else str(_rb)),
                            merged_as=(None if rescued == "exclude"
                                       else int(offset))))
                        if rescued == "exclude":
                            continue
                    dst = _group_path(scan_key, offset)
                    out.copy(parent[str(i)], dst)
                    g = out[dst]
                    g.attrs["count"] = offset
                    # Schema v2: draws store fixed `eqdsk`/`pfile` names, so the
                    # copy needs no rename (F11) -- the group path carries the
                    # coordinate. load_equilibrium resolves by the fixed name.
                    offset += 1

    if _resc:
        with h5py.File(out_path, "a") as out:
            gp = base_path if base_path else "/"
            out[gp].attrs["merge_rescued_json"] = json.dumps(dict(
                policy=rescued, n_rescued=len(_resc), draws=_resc))
        if rescued == "include":
            warnings.warn(
                f"merge_archives: {len(_resc)} merged draw(s) had their solve "
                "RESCUED (solve_recovered; a different urf or a looser "
                "nl_tol) -- recorded in merge_rescued_json; pass "
                "rescued='exclude' to leave them out", UserWarning,
                stacklevel=2)
    # Run-level provenance on the merged archive (schema/version/timestamp
    # always; config_json when the caller supplied the run config).
    from .utils import write_provenance
    write_provenance(out_header, config=config, scan_key=scan_key)

    # Run manifest: one record per worker (attempts, in-spec, seed, caps).
    # For a shared until-N run this is the replay key -- re-running with each
    # worker's n_attempts as a fixed allocation regenerates the archive.
    partial = (dict(missing_workers=sorted(int(w) for w in missing_workers),
                    merged_workers=sorted(int(w["worker_id"]) for w in workers))
               if missing_workers else None)
    if workers:
        shared = [w for w in workers if "shared_target" in w]
        manifest = dict(
            n_workers=len(workers), n_draws=int(offset), workers=workers,
            n_shards_without_record=int(n_unrecorded), partial=partial,
            until_n=(dict(target=shared[0]["shared_target"],
                          total_cap=shared[0].get("total_cap"),
                          n_inspec_recorded=sum(int(w.get("n_inspec", 0))
                                                for w in shared),
                          # the one boundary cut every shard counted against
                          # (only when the shards recorded it)
                          **({k: v for k, v in zip(_INLOOP_CUT_KEYS, next(iter(_distinct)))}
                             if _distinct else {}))
                     if shared else None))
        with h5py.File(out_path, "a") as out:
            gp = base_path if base_path else "/"
            out[gp].attrs["parallel_manifest_json"] = json.dumps(manifest)
        # run-level generation provenance, aggregated over the shards (the
        # same record a serial run stamps; attempts summed, outcomes keyed by
        # worker since shard attempt indices overlap)
        from .utils import stamp_generation_provenance
        _n_att = [w.get("n_attempted") for w in workers]
        _vers = sorted({str(w["bouquet_version"]) for w in workers
                        if w.get("bouquet_version") is not None})
        stamp_generation_provenance(
            out_header, scan_key=scan_key,
            n_requested=(int(shared[0]["shared_target"]) if shared
                         else (int(config.generation.n_equils) if config is not None
                               else None)),
            n_requested_source=(("n_inspec_target" if shared else "n_equils")
                                + (" (PARTIAL merge: workers "
                                   f"{partial['missing_workers']} missing)"
                                   if partial else "")),
            generation_mode=("until_n" if shared else "fixed"),
            # attempts are known only if EVERY merged shard recorded them
            n_attempted=(int(sum(int(a) for a in _n_att))
                         if n_unrecorded == 0 and all(a is not None for a in _n_att)
                         else None),
            merge_partial_json=partial,
            n_stored=int(offset),
            attempt_outcomes_json={str(w["worker_id"]): w.get("attempt_outcomes")
                                   for w in workers},
            bouquet_version=(",".join(_vers) if _vers else None),
            **dict(zip(_INLOOP_CUT_KEYS, next(iter(_distinct), (None, None, None)))),
        )
        if shared:
            _u = manifest["until_n"]
            print(f"[until-N] merged {offset} draws from {len(workers)} "
                  f"workers; {_u['n_inspec_recorded']} in-spec recorded "
                  f"against a shared target of {_u['target']}.")

    if partial and not workers:            # no worker records, still partial
        from .utils import stamp_generation_provenance
        stamp_generation_provenance(out_header, scan_key=scan_key,
                                    merge_partial_json=partial)

    if cleanup:
        for sp in shard_paths:
            if sp and os.path.exists(sp):
                os.remove(sp)
    return out_path, offset


def apply_filters_after_merge(config):
    """Run ``Bouquet(config).filter()`` on the merged archive.

    The parallel path used to leave the merged archive UNFILTERED: the only
    coil verdict on it was the in-loop legacy band, so ``selection="selected"``
    silently meant something different from a serial run.  This applies the
    configured filters (chi2 coil filter by default, plus the boundary cut)
    exactly as the serial ``run.filter()`` does.  Needs no solver.
    """
    from .run import Bouquet
    return Bouquet(copy.deepcopy(config)).filter()


# --------------------------------------------------------------------------
#  orchestration
# --------------------------------------------------------------------------
def parallel_generate(config, *, n_workers=None, threads_per_worker=1, seed=0,
                      backend="laptop", baseline_match_rtol=1e-6, cleanup=True,
                      verbose=False, progress=True, slurm=None,
                      apply_filters=True):
    """Fan ``config.generation.n_equils`` draws across worker processes, merge.

    ``backend="laptop"`` runs a ``ProcessPoolExecutor`` (spawn) now;
    ``backend="slurm"`` writes a job-array + merge script via
    :func:`emit_slurm_script` (pass options as the ``slurm`` dict) and returns
    their paths without running.

    ``n_workers=None`` defaults to the physical cores this process may use
    (one single-threaded worker per core; :func:`_get_num_cpus`).
    ``threads_per_worker`` must be 1 (:mod:`bouquet.threads`).

    The cross-worker **baseline check**: every worker reports its forward-solved
    baseline ``l_i``/``Ip``; if any drifts beyond ``baseline_match_rtol`` the run
    raises (a worker did not converge to the shared baseline -- e.g. a stray
    ``nthreads>1`` or a mismatched source). Returns a summary dict.
    """
    n_total = int(config.generation.n_equils)
    tgt = getattr(config.generation, "n_inspec_target", None)
    if tgt is not None:
        # shared until-N: the pool's total is the run's attempt cap
        n_total = shared_until_n_budget(
            tgt, getattr(config.generation, "max_total_draws", None),
            n_total, 1, 0)["total_cap"]
    scan_key = config.generation.scan_key
    out_header = config.output_header
    require_single_thread(threads_per_worker, "threads_per_worker")
    if n_workers is None:
        n_workers = _get_num_cpus(use_logical=False)
    nw = max(1, min(int(n_workers), n_total))

    if backend == "slurm":
        return emit_slurm_script(
            config, n_workers=nw, seed=seed,
            threads_per_worker=threads_per_worker, **(slurm or {}))

    if backend != "laptop":
        raise ValueError(f"backend must be 'laptop' or 'slurm', got {backend!r}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    results = [None] * nw

    # Optional live progress: workers post one item per draw attempt to a shared
    # queue; a daemon thread drains it into a single aggregate bar (per-worker
    # tqdm can't surface across processes, and worker stderr is suppressed).
    import threading
    import queue as _queue
    mgr = ctx.Manager() if (progress or tgt is not None) else None
    pq = mgr.Queue() if progress else None
    ledger = ManagerYieldLedger(mgr) if tgt is not None else None
    bar = None
    drain_stop = threading.Event()
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=n_total, desc=f"draws ({nw}w x{threads_per_worker}t)",
                       unit="draw")
        except ImportError:
            bar = None

    def _drain():
        per_worker = {}
        while not drain_stop.is_set() or not pq.empty():
            try:
                w = pq.get(timeout=0.2)
            except (_queue.Empty, OSError):
                continue
            per_worker[w] = per_worker.get(w, 0) + 1
            if bar is not None:
                bar.update(1)
                bar.set_postfix_str(
                    " ".join(f"w{k}:{v}" for k, v in sorted(per_worker.items())))

    drain_thread = None
    if progress:
        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

    try:
        with worker_environment(), ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            futs = {
                ex.submit(run_shard, config, i, nw,
                          n_equils_total=n_total, seed_base=seed,
                          out_header=out_header, scan_key=scan_key,
                          threads_per_worker=threads_per_worker, verbose=verbose,
                          progress_q=pq, ledger=ledger): i
                for i in range(nw)
            }
            for fut in as_completed(futs):
                r = fut.result()
                results[r["worker_id"]] = r
    finally:
        drain_stop.set()
        if drain_thread is not None:
            drain_thread.join(timeout=2.0)
        if bar is not None:
            bar.close()
        if mgr is not None:
            mgr.shutdown()

    # cross-worker baseline check (the one-line guard)
    solved = [r for r in results if r and r["li_target"] is not None]
    if not solved:
        raise RuntimeError("no worker produced a baseline")
    li0, ip0 = solved[0]["li_target"], solved[0]["Ip_target"]
    for r in solved:
        if (abs(r["li_target"] - li0) > baseline_match_rtol * abs(li0) or
                abs(r["Ip_target"] - ip0) > baseline_match_rtol * abs(ip0)):
            raise RuntimeError(
                f"worker {r['worker_id']} baseline drifted from worker "
                f"{solved[0]['worker_id']}: li {r['li_target']:.8f} vs "
                f"{li0:.8f}, Ip {r['Ip_target']:.6e} vs {ip0:.6e}. Workers did "
                "not converge to the same baseline -- check nthreads=1 and that "
                "every worker used the identical source/config.")

    paths = [r["path"] for r in results if r and r["path"]]
    out_path, n_merged = merge_archives(paths, out_header, scan_key=scan_key,
                                        cleanup=cleanup, config=config)
    summary = dict(out_path=out_path, n_draws=n_merged, n_workers=nw,
                   threads_per_worker=threads_per_worker,
                   li_target=li0, Ip_target=ip0,
                   n_attempts=sum(int(r.get("n_attempts", 0)) for r in results if r),
                   n_inspec_recorded=sum(int(r.get("n_inspec", 0)) for r in results if r))
    if tgt is not None:
        summary["until_n"] = dict(target=int(tgt), total_cap=n_total,
                                  reached=summary["n_inspec_recorded"] >= int(tgt))
        if not summary["until_n"]["reached"]:
            import warnings
            warnings.warn(
                f"shared until-N did not reach its target: "
                f"{summary['n_inspec_recorded']}/{int(tgt)} in-spec draws "
                f"after the pooled attempt cap of {n_total}. The archive "
                "holds every attempt; raise max_total_draws, loosen the "
                "filter thresholds deliberately, or treat the low yield as a "
                "finding about this equilibrium.", RuntimeWarning, stacklevel=2)
    if apply_filters:
        summary["filter"] = apply_filters_after_merge(config)
    return summary


# --------------------------------------------------------------------------
#  cluster: emit a SLURM job-array + dependent merge
# --------------------------------------------------------------------------
def emit_slurm_script(config, *, n_workers, seed, threads_per_worker,
                      out_dir=".", job_name="bouquet", partition=None,
                      time_limit="02:00:00", mem_per_task="16G",
                      python="python", setup=None, apply_filters=True):
    """Write a SLURM job-array (one shard per task) + a dependent merge job.

    Serialises the run into ``{job_name}_bundle.json`` (config via
    ``BouquetConfig.to_dict``) and writes two sbatch
    scripts plus a ``{job_name}_submit.sh`` that chains them with
    ``--dependency=afterany`` (the merge validates shards itself: it aborts
    loudly on missing workers or a drifted baseline, so it is safe -- and
    more informative -- to let it run after a partial array instead of
    leaving it pending forever behind ``afterok``). Nothing is launched.
    Returns the written paths.

    ``setup`` is an optional list of shell lines inserted before the payload
    in BOTH sbatch scripts -- environment activation the compute node needs,
    e.g. ``["module load conda", "conda activate bouquet",
    "export OFT_PYTHONPATH=/path/to/OFT/python"]``. Without OFT importable,
    every shard dies at ``import OpenFUSIONToolkit``.

    Launch with ``{job_name}_submit.sh`` -- it ``cd``'s to its own directory
    first, so it works from any CWD. (The sbatch scripts reference the bundle
    by basename and therefore assume the submission CWD is ``out_dir``;
    shard/merged ``.h5`` outputs land there too unless
    ``config.output_header`` is an absolute path.)
    """
    require_single_thread(threads_per_worker, "threads_per_worker")
    os.makedirs(out_dir, exist_ok=True)
    tgt = getattr(config.generation, "n_inspec_target", None)
    if tgt is not None:      # validate the pooled budget now, not on the node
        shared_until_n_budget(tgt, getattr(config.generation, "max_total_draws", None),
                              int(config.generation.n_equils), 1, 0)
    # Config JSON bundle (not pickle): portable across package/Python versions,
    # human-inspectable, and the same serialization the h5 provenance uses (F25).
    import json
    bundle = dict(
        config=config.to_dict(), n_workers=int(n_workers), seed=int(seed),
        threads_per_worker=int(threads_per_worker),
        n_equils_total=int(config.generation.n_equils),
        scan_key=config.generation.scan_key,
        out_header=config.output_header,
        # shared until-N: the array tasks pool their in-spec count in this
        # file (see FileYieldLedger); None when the run is a fixed-n one.
        n_inspec_target=(int(tgt) if tgt is not None else None),
        max_total_draws=getattr(config.generation, "max_total_draws", None),
        ledger=(ledger_path_for(config.output_header) if tgt is not None
                else None),
        # whether the merge job runs the configured filters on the merged
        # archive (the serial run.filter() equivalent); the merge CLI's
        # --no-filter flag also switches it off
        apply_filters=bool(apply_filters),
        _shard_note="solver.nthreads is overwritten to 1 by the shard runner",
    )
    bname = f"{job_name}_bundle.json"
    bpath = os.path.join(out_dir, bname)
    with open(bpath, "w") as fh:
        json.dump(bundle, fh, indent=2)

    part = f"#SBATCH --partition={partition}\n" if partition else ""
    # No setup lines given: emit the commented hint instead of nothing --
    # without OFT importable every shard dies at `import OpenFUSIONToolkit`,
    # and a user copying the sbatch never saw this docstring.  (The hint
    # shipped in the committed example scripts for months and was lost when
    # they were regenerated; Copilot review on PR #39.)
    if setup:
        extra = "".join(f"{line}\n" for line in setup)
    else:
        extra = ("# compute-node environment (adjust for your cluster), e.g.:\n"
                 "# module load conda && conda activate bouquet\n"
                 "# export OFT_PYTHONPATH=/path/to/OpenFUSIONToolkit"
                 "/build_release/python\n")
    env = "".join(f"export {k}={v}\n" for k, v in worker_env({}).items())

    # The bundle is referenced by BASENAME: sbatch tasks run in the submission
    # CWD, and submit.sh cd's to this directory first -- so the pair works
    # from any CWD without baking a machine-specific absolute path into the
    # scripts.
    array = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}\n"
        f"#SBATCH --array=0-{n_workers - 1}\n"
        f"#SBATCH --cpus-per-task={threads_per_worker}\n"
        f"#SBATCH --time={time_limit}\n#SBATCH --mem={mem_per_task}\n{part}"
        f"{extra}{env}"
        f"{python} -m bouquet.parallel shard {bname} $SLURM_ARRAY_TASK_ID\n"
    )
    merge = (
        f"#!/bin/bash\n#SBATCH --job-name={job_name}_merge\n"
        f"#SBATCH --cpus-per-task=1\n#SBATCH --time=00:20:00\n"
        f"#SBATCH --mem={mem_per_task}\n{part}{extra}"
        f"{python} -m bouquet.parallel merge {bname}\n"
    )
    submit = (
        "#!/bin/bash\n"
        "# run from anywhere: everything below is relative to this script\n"
        'cd "$(dirname "$0")"\n'
        "# chain array + merge; afterany (not afterok) so the merge still\n"
        "# runs -- and reports exactly which shards are missing -- after a\n"
        "# partial array, instead of pending forever.\n"
        + (f"# shared until-N: start from an EMPTY ledger so a previous run's\n"
           f"# count can never pre-satisfy the target\n"
           f": > \"{bundle['ledger']}\"\n" if tgt is not None else "")
        + f"aid=$(sbatch --parsable {job_name}_array.sbatch)\n"
        f"sbatch --dependency=afterany:$aid {job_name}_merge.sbatch\n"
    )

    apath = os.path.join(out_dir, f"{job_name}_array.sbatch")
    mpath = os.path.join(out_dir, f"{job_name}_merge.sbatch")
    spath = os.path.join(out_dir, f"{job_name}_submit.sh")
    for p, txt in ((apath, array), (mpath, merge), (spath, submit)):
        with open(p, "w") as fh:
            fh.write(txt)
    os.chmod(spath, 0o755)
    return dict(bundle=bpath, array=apath, merge=mpath, submit=spath)


# --------------------------------------------------------------------------
#  CLI used by the SLURM scripts:  python -m bouquet.parallel {shard|merge} ...
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  tier 2: many bouquets on one node, one standing solver per worker
# --------------------------------------------------------------------------
#  A case is a CaseSpec swapped onto a worker's Bouquet (set_case), so every case runs the ordinary
#  prepare_baseline -> generate -> filter.  parallel_generate splits the draws of ONE bouquet; never call
#  it inside a case worker.  Across jobs and nodes, bouquet.units queues the same cases.

_worker_state: dict = {}


class _IndexMap:
    """Picklable ``idx -> case`` map, saved as ``map_object.pkl`` (completion order is not run order)."""

    def __init__(self, flat_list):
        self.flat_list = list(flat_list)

    def __call__(self, idx):
        return self.flat_list[idx]

    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)


def check_cases(cases, config):
    """Refuse a sweep that cannot run: a parallel config at ``nthreads != 1``, or a case source the solve
    method does not take (swb runs on IDS sources only)."""
    from .config import ImasSource, resolve_solve_method
    require_single_thread(config.solver.nthreads, "solver.nthreads")
    if resolve_solve_method(config.generation) == "swb":
        bad = [i for i, c in enumerate(cases) if not isinstance(c.source, ImasSource)]
        if bad:
            raise ValueError(f"solve_method='swb' takes IDS sources only; cases {bad} are not ImasSource")


def release_dd_cache(bouquet, case):
    """``after_baseline`` hook: release the parsed dd(s) the baseline read cached (``io.imas._cached_dd``,
    up to two ~GB files).  For a sweep over different dd files on few workers; a slice sweep over one
    dd would then re-parse it per case."""
    from .io.imas import clear_dd_cache
    clear_dd_cache()


def run_case_on(bouquet, case, header, hooks=None) -> dict:
    """One case on a standing solver: ``set_case`` -> ``before_baseline`` -> ``prepare_baseline`` ->
    ``after_baseline`` -> ``generate`` -> ``filter``.  Not ``Bouquet.run()``: its export would write a
    second archive per case.  Returns the case record."""
    from .threads import pinned, thread_report
    hooks = hooks or {}
    bouquet.set_case(case)
    bouquet.output_header = header
    if hooks.get("before_baseline"):
        hooks["before_baseline"](bouquet, case)
    bouquet.prepare_baseline()
    if hooks.get("after_baseline"):
        hooks["after_baseline"](bouquet, case)
    bouquet.generate()
    bouquet.filter()
    bl = bouquet.baseline
    return dict(path=os.path.abspath(f"{header}.h5"), group=case.group, scan_key=case.scan_key,
                n_all=len(bouquet.selected_indices("all")), n_sel=len(bouquet.selected_indices("selected")),
                l_i_target=float(bl.l_i_target), Ip_target=float(bl.Ip_target), F0=float(bouquet._geom.F0),
                threads=thread_report(), threads_pinned=pinned())


def _init_case_worker(worker_id_queue, master_working_dir, config_dict, init_status_queue, hooks=None,
                      verbose=False):
    """Pool initialiser: one thread, own directory and log, own mesh copy, one ``setup_solver``.

    Reports ``(worker_id, traceback)`` on failure, which the parent raises once the pool breaks."""
    import shutil
    import sys
    import traceback
    worker_id = -1
    try:
        pin_threads()
        worker_id = worker_id_queue.get(timeout=60)     # 0..n-1: the executor never replaces a worker
        master_working_dir = os.path.abspath(master_working_dir)
        working_dir = os.path.join(master_working_dir, f"worker_{worker_id}")
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)               # TokaMaker writes scratch files in the cwd
        log_path = os.path.join(master_working_dir, f"worker_{worker_id}.log")
        if not verbose:                     # fd level: OFT writes to fd 1/2 directly
            fh = open(log_path, "w", buffering=1)
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            sys.stdout = sys.stderr = fh
        from .config import BouquetConfig
        from .paths import add_oft_to_path
        from .run import Bouquet
        add_oft_to_path()
        config = BouquetConfig.from_dict(config_dict)
        # one mesh file opened by many OFT processes at once has deadlocked a serial HDF5 build
        local_mesh = os.path.join(working_dir, os.path.basename(config.solver.mesh_path))
        shutil.copy2(config.solver.mesh_path, local_mesh)
        config.solver.mesh_path = local_mesh
        _worker_state.update(worker_id=worker_id, working_dir=working_dir, log_path=log_path,
                             bouquet=Bouquet(config).setup_solver(), hooks=dict(hooks or {}))
    except Exception:
        tb = traceback.format_exc()
        print(f"[worker {worker_id}] init failed:\n{tb}", flush=True)
        try:
            init_status_queue.put((worker_id, tb))
        except Exception:
            pass
        raise


def case_seed(seed, case) -> int:
    """Seed of one case of a sweep, from the run seed and the case's ``group``/``scan_key``: independent
    of the worker, the run order and the tier (:func:`parallel_cases`, ``units.units_from_cases``)."""
    from .units import unit_seed
    return unit_seed(seed, f"{case.group}/{case.scan_key}")


def run_case(run_args):
    """Pool task: one case on this worker's solver, at its own seed.  Never raises: a failed case returns
    its traceback, so one bad slice cannot stop a sweep."""
    import traceback
    idx, case, case_dir, seed = run_args
    header = os.path.join(case_dir, f"{os.path.basename(case.header)}_idx{idx}")
    try:
        _worker_state["bouquet"].config.generation.seed = seed
        rec = run_case_on(_worker_state["bouquet"], case, header, _worker_state.get("hooks"))
        return idx, True, None, dict(rec, idx=idx, worker_id=_worker_state["worker_id"])
    except Exception:
        tb = traceback.format_exc()
        print(f"[worker {_worker_state.get('worker_id')} | case {idx}] failed:\n{tb}", flush=True)
        return idx, False, tb, None


def _worker_config(config, case) -> dict:
    """The run-level config with a case's source, so a worker's ``setup_solver`` points at a real baseline
    (the run-level source may be a placeholder)."""
    from .config import _encode_source
    return dict(config.to_dict(), source=_encode_source(case.source))


def _picklable_hooks(**hooks) -> dict:
    import pickle
    out = {}
    for name, fn in hooks.items():
        if fn is None:
            continue
        if not callable(fn):
            raise TypeError(f"{name} must be callable, got {type(fn).__name__}")
        try:
            pickle.loads(pickle.dumps(fn))
        except Exception as exc:
            raise TypeError(f"{name} is not picklable ({exc}); workers are spawned, so a hook must be a "
                            "module-level function (not a lambda, closure or bound method)") from exc
        out[name] = fn
    return out


def parallel_cases(source, config, master_working_dir, *, use_logical_cpus=True, n_cpus_override=None,
                   verbose=False, merge=True, group_by="group", cleanup=False, before_baseline=None,
                   after_baseline=None):
    """Run many independent bouquets in parallel on one node, one single-threaded solver per worker.

    Call it under ``if __name__ == "__main__":`` in a script: workers are spawned (OFT's Fortran is not
    fork-safe) and re-import the main module.  Input paths may be relative to the launch directory.
    Each case runs at :func:`case_seed` of ``config.generation.seed``.  A worker that dies (a crash or
    an out-of-memory kill) ends the sweep: finished cases are kept and merged, the unfinished ones are
    reported in ``errors``.

    Parameters
    ----------
    source : ParallelSource or list of CaseSpec
        The cases (a ``ParallelSource`` is expanded).
    config : BouquetConfig
        The run-level config (solver, uncertainty, generation); each case supplies the source.
    master_working_dir : str
        Holds ``worker_N/`` (scratch, mesh copy), ``worker_N.log``, ``cases/*_idxN.h5``,
        ``map_object.pkl``, ``errors.pkl`` and the merged archives.
    use_logical_cpus : bool
        One worker per logical CPU (default) or per physical core.  Every worker is single-threaded.
    n_cpus_override : int, optional
        The worker count, bypassing detection.
    verbose : bool
        Stream worker output to the terminal instead of ``worker_N.log``.
    merge, group_by, cleanup
        :func:`merge_cases` on the successful cases.
    before_baseline, after_baseline : callable, optional
        ``f(bouquet, case)`` either side of ``prepare_baseline()`` on the worker: per-case setup a
        ``CaseSpec`` cannot express (solver targets before; anything on the baseline's grid after, e.g.
        ``uncertainty.aux_baselines``).  Module-level functions only (they are pickled).

    Returns
    -------
    dict
        ``n_runs, n_success, cases, results, errors, merged, map_object_path``.
    """
    import dataclasses
    import multiprocessing
    import pickle
    import queue
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    from .config import absolute_source, with_absolute_paths

    config = with_absolute_paths(config)        # workers run in their own directories
    cases = [dataclasses.replace(c, source=absolute_source(c.source))
             for c in (source.expand() if hasattr(source, "expand") else source)]
    check_cases(cases, config)
    hooks = _picklable_hooks(before_baseline=before_baseline, after_baseline=after_baseline)
    master_working_dir = os.path.abspath(master_working_dir)
    case_dir = os.path.join(master_working_dir, "cases")
    os.makedirs(case_dir, exist_ok=True)
    n_runs = len(cases)
    if n_runs == 0:
        return dict(n_runs=0, n_success=0, cases=[], results={}, errors={}, merged={}, map_object_path=None)

    n_cpus = int(n_cpus_override) if n_cpus_override is not None else _get_num_cpus(use_logical_cpus)
    n_workers = max(1, min(n_cpus, n_runs))
    print(f"[parallel_cases] {n_runs} case(s) on {n_workers} worker(s)", flush=True)
    map_object_path = os.path.join(master_working_dir, "map_object.pkl")
    with open(map_object_path, "wb") as fh:
        pickle.dump(_IndexMap(cases), fh)

    ctx = multiprocessing.get_context("spawn")
    init_status_queue, worker_id_queue = ctx.Queue(), ctx.Queue()
    for w in range(n_workers):
        worker_id_queue.put(w)
    seed = config.generation.seed
    errors, results = {}, {}
    with worker_environment():
        ex = ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_case_worker,
                                 initargs=(worker_id_queue, master_working_dir, _worker_config(config, cases[0]),
                                           init_status_queue, hooks, bool(verbose)))
        try:
            futs = {ex.submit(run_case, (i, c, case_dir, case_seed(seed, c))): i for i, c in enumerate(cases)}
            for fut in as_completed(futs):
                idx, ok, tb, rec = fut.result()
                c = cases[idx]
                if ok:
                    results[idx] = rec
                    print(f"[parallel_cases] case {idx} ({c.group}/{c.scan_key}) done: "
                          f"{rec['n_sel']}/{rec['n_all']} in spec", flush=True)
                else:
                    errors[idx] = tb
                    print(f"[parallel_cases] case {idx} ({c.group}/{c.scan_key}) failed:\n{tb}", flush=True)
        except BrokenProcessPool:
            failed = []
            while True:
                try:
                    failed.append(init_status_queue.get(timeout=1.0))
                except queue.Empty:
                    break
            if failed:
                raise RuntimeError(f"{len(failed)} worker(s) failed to initialise:\n"
                                   + "\n".join(f"  worker {w}:\n{t}" for w, t in failed)) from None
            lost = sorted(set(range(n_runs)) - set(results) - set(errors))
            for i in lost:
                errors[i] = "not finished: a worker process died (crash or out-of-memory kill) during the sweep"
            print(f"[parallel_cases] a worker died; {len(lost)} case(s) not finished: {lost}", flush=True)
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    if errors:
        with open(os.path.join(master_working_dir, "errors.pkl"), "wb") as fh:
            pickle.dump(errors, fh)
    merged = {}
    if merge and results:
        merged = merge_cases([results[i] for i in sorted(results)],
                             os.path.join(master_working_dir, os.path.basename(config.output_header)),
                             group_by=group_by, cleanup=cleanup)
    print(f"[parallel_cases] {n_runs - len(errors)}/{n_runs} case(s) succeeded", flush=True)
    return dict(n_runs=n_runs, n_success=n_runs - len(errors), cases=cases, results=results, errors=errors,
                merged=merged, map_object_path=map_object_path)


def merge_cases(results, out_header, *, group_by="group", cleanup=False):
    """Merge per-case archives into ``{out_header}_{group}.h5`` per case group (``group_by=None``: one
    ``{out_header}.h5``).  Each case's ``scan/<scan_key>`` group is copied whole, with its own
    ``config_json`` and ``_baseline``.  A duplicate scan_key in a group, a missing archive or a missing
    scan group raises: a partial merge would silently shrink the sweep.  Returns ``{group: path}``."""
    import h5py
    from .utils import _scan_key, initialize_equilibrium_database, write_provenance

    buckets = {}
    for r in results:
        if r and r.get("path"):
            buckets.setdefault(r.get("group") if group_by == "group" else None, []).append(r)
    merged = {}
    for gname, rows in buckets.items():
        seen = {}
        for r in rows:
            k = _scan_key(r["scan_key"])
            if k in seen:
                raise ValueError(f"duplicate scan_key {r['scan_key']!r} in group {gname!r}: cases {seen[k]} "
                                 f"and {r['idx']} would overwrite each other; give each case its own scan_key")
            seen[k] = r["idx"]
        stem = out_header if gname is None else f"{out_header}_{gname}"
        out_path = os.path.abspath(f"{stem}.h5")
        if os.path.exists(out_path):
            os.remove(out_path)
        initialize_equilibrium_database(stem)
        with h5py.File(out_path, "a") as dst:
            for r in sorted(rows, key=lambda x: x["idx"]):
                if not os.path.exists(r["path"]):
                    raise FileNotFoundError(f"case archive not found: {r['path']} (group {gname!r}, scan_key "
                                            f"{r['scan_key']!r}); re-run the case or drop it from `results`")
                grp = f"scan/{_scan_key(r['scan_key'])}"
                with h5py.File(r["path"], "r") as src:
                    if grp not in src:
                        raise KeyError(f"{r['path']} has no '{grp}' group: the case ran with a different "
                                       f"scan_key than its CaseSpec ({r['scan_key']!r})")
                    dst.copy(src[grp], grp)
        write_provenance(stem)
        merged[gname] = out_path
        if cleanup:
            for r in rows:
                if os.path.exists(r["path"]):
                    os.remove(r["path"])
    return merged


def _cli(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit("usage: python -m bouquet.parallel {shard <bundle> "
                         "<worker_id> | merge <bundle> [--allow-missing]}")
    cmd, bpath = argv[0], argv[1]
    import json
    from .config import BouquetConfig
    with open(bpath) as fh:
        b = json.load(fh)
    b["config"] = BouquetConfig.from_dict(b["config"])
    if cmd == "shard":
        # verbose=True: SLURM already isolates each task's output in its own
        # slurm-*.out, so the notebook flood rationale for fd-suppression does
        # not apply -- and an empty log is useless when a shard dies.
        ledger = (FileYieldLedger(b["ledger"]) if b.get("ledger") else None)
        run_shard(b["config"], int(argv[2]), b["n_workers"],
                  n_equils_total=b["n_equils_total"], seed_base=b["seed"],
                  out_header=b["out_header"], scan_key=b["scan_key"],
                  threads_per_worker=b["threads_per_worker"], verbose=True,
                  ledger=ledger)
    elif cmd == "merge":
        allow_missing = "--allow-missing" in argv[2:]
        # workers assigned zero draws legitimately produce no shard file;
        # only count the ones that were expected to write one.
        _tot = (shared_until_n_budget(b["n_inspec_target"], b.get("max_total_draws"),
                                      b["n_equils_total"], 1, 0)["total_cap"]
                if b.get("n_inspec_target") is not None else b["n_equils_total"])
        expected = [i for i in range(b["n_workers"])
                    if _shard_size(_tot, b["n_workers"], i) > 0]
        paths = {i: f"{b['out_header']}_w{i}.h5" for i in expected}
        missing = sorted(i for i, p in paths.items() if not os.path.exists(p))
        if missing:
            msg = (f"missing shard archive(s) for worker(s) {missing}: "
                   f"expected {len(expected)}, found "
                   f"{len(expected) - len(missing)}")
            if not allow_missing:
                raise SystemExit(
                    "merge aborted: " + msg + ". Re-run those array indices "
                    "(sbatch --array=" + ",".join(map(str, missing)) +
                    " <array.sbatch>), or pass --allow-missing to merge the "
                    "partial set.")
            print(f"WARN: {msg} -- merging the partial set")
        shard_list = [p for i, p in sorted(paths.items()) if i not in missing]
        out_path, n = merge_archives(shard_list, b["out_header"],
                                     scan_key=b["scan_key"], cleanup=True,
                                     config=b["config"],
                                     missing_workers=missing or None)
        print(f"merged {n} draws -> {out_path}")
        if b.get("ledger"):
            _led = FileYieldLedger(b["ledger"])
            _cnt, _tgt = _led.count(), int(b["n_inspec_target"])
            print(f"[until-N] ledger: {_cnt} in-spec recorded against a "
                  f"shared target of {_tgt}"
                  + ("" if _cnt >= _tgt else " -- TARGET NOT REACHED "
                     "(pooled attempt cap exhausted or shards missing)"))
            try:
                os.remove(b["ledger"])
            except OSError:
                pass
        # the merged archive is unfiltered until this runs (see
        # apply_filters_after_merge); a serial run.filter() equivalent
        if "--no-filter" not in argv[2:] and b.get("apply_filters", True):
            apply_filters_after_merge(b["config"])
        else:
            print("merged archive left UNFILTERED (apply_filters off): run "
                  "Bouquet(config).filter() before selecting draws")
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    _cli()
