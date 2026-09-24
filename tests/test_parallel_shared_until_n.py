"""Shared until-N on the parallel launchers (solve-free).

The draw loop's two hooks (``on_inspec`` / ``stop_check``) are exercised
through a fake ``Bouquet`` that replays the loop's contract, so these tests
cover the ledger, the per-worker budget, ``run_shard``'s wiring, the merge
manifest and the SLURM merge CLI without a solver.  The real loop's use of
the hooks is a structural check on ``generate_bouquet``'s source.
"""
import inspect
import json
import multiprocessing as mp
import os

import h5py
import numpy as np
import pytest

from bouquet.config import (BouquetConfig, ImasSource, SolverConfig,
                            FilterConfig)
from bouquet.parallel import (FileYieldLedger, ManagerYieldLedger,
                              shared_until_n_budget, run_shard, merge_archives,
                              apply_filters_after_merge, _shard_size, _cli)


# ---------------------------------------------------------------------------
# ledgers
# ---------------------------------------------------------------------------

def _append_many(path, n):
    led = FileYieldLedger(path)
    for _ in range(n):
        led.record()


def test_file_ledger_counts_concurrent_appends(tmp_path):
    path = str(tmp_path / "x.ledger")
    led = FileYieldLedger(path)
    assert led.count() == 0                       # missing file reads as 0
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(path, 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert led.count() == 100
    assert led.reached(100) and not led.reached(101)
    led.reset()
    assert led.count() == 0


def test_manager_ledger_round_trip():
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        led = ManagerYieldLedger(mgr)
        for _ in range(3):
            led.record()
        assert led.count() == 3 and led.reached(3) and not led.reached(4)
        led.reset()
        assert led.count() == 0


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

def test_budget_shares_sum_to_the_cap_and_local_target_is_bounded():
    for nw in (1, 2, 3, 8):
        shares = [shared_until_n_budget(15, None, 20, nw, w) for w in range(nw)]
        assert all(b["total_cap"] == 75 for b in shares)      # 5 x target
        assert sum(b["cap"] for b in shares) == 75
        assert all(b["local_target"] == min(15, b["cap"]) for b in shares)
        assert all(b["local_target"] <= b["cap"] for b in shares)


def test_budget_default_never_below_n_equils_and_explicit_cap_wins():
    assert shared_until_n_budget(3, None, 40, 1, 0)["total_cap"] == 40
    assert shared_until_n_budget(3, 12, 40, 1, 0)["total_cap"] == 12
    with pytest.raises(ValueError, match="could never be met"):
        shared_until_n_budget(10, 4, 4, 1, 0)


# ---------------------------------------------------------------------------
# run_shard wiring, via a fake Bouquet that replays the loop's contract
# ---------------------------------------------------------------------------

class _FakeBaseline:
    l_i_target = 0.9
    Ip_target = 1.0e6


class _FakeBouquet:
    """Replays generate_bouquet's until-N contract: stop_check at the top of
    every attempt, one archived draw per attempt, on_inspec on the passing
    ones, local break at the (local) target."""
    inspec_pattern = None            # per test: list of bools, cycled

    def __init__(self, cfg):
        self.config = cfg
        self.baseline = _FakeBaseline()

    def setup_solver(self):
        pass

    def prepare_baseline(self):
        pass

    def generate(self, progress_callback=None, on_inspec=None, stop_check=None):
        gc = self.config.generation
        pat = list(self.inspec_pattern or [True])
        path = f"{self.config.output_header}.h5"
        with h5py.File(path, "w") as hf:
            g = hf.create_group(f"scan/{gc.scan_key}/_baseline")
            g.attrs["l_i_target"] = 0.9
            g.attrs["Ip_target"] = 1.0e6
        local = 0
        for count in range(int(gc.max_total_draws)):
            if stop_check is not None and stop_check():
                break
            if progress_callback is not None:
                progress_callback(count)
            ok = pat[count % len(pat)]
            with h5py.File(path, "a") as hf:
                d = hf.create_group(f"scan/{gc.scan_key}/{count}")
                d.attrs["count"] = count
                d.attrs["max_F_drift_pct"] = 0.5 if ok else 5.0
                d.attrs["max_VSC_drift_pct"] = 0.5
                d.attrs["inspec_F_max"] = 0.02
                d.attrs["inspec_VSC_max"] = 0.02
                d.attrs["in_spec"] = bool(ok)
            if ok:
                local += 1
                if on_inspec is not None:
                    on_inspec()
                if local >= int(gc.n_inspec_target):
                    break
        return []


def _cfg(tmp_path, **gen):
    cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                        solver=SolverConfig(mesh_path="unused.h5"),
                        output_header=str(tmp_path / "run"),
                        filtering=FilterConfig(coil_filter="legacy"))
    cfg.generation.scan_key = 0
    cfg.generation.n_equils = 8
    for k, v in gen.items():
        setattr(cfg.generation, k, v)
    return cfg


@pytest.fixture
def fake_bouquet(monkeypatch):
    import bouquet
    monkeypatch.setattr(bouquet, "Bouquet", _FakeBouquet)
    _FakeBouquet.inspec_pattern = None
    return _FakeBouquet


def _shard(cfg, w, nw, ledger, tmp_path):
    return run_shard(cfg, w, nw, n_equils_total=cfg.generation.n_equils,
                     seed_base=1, out_header=str(tmp_path / "run"),
                     scan_key=0, threads_per_worker=1, verbose=True,
                     ledger=ledger)


def test_worker_stops_at_the_shared_target_and_records_it(fake_bouquet, tmp_path):
    led = FileYieldLedger(tmp_path / "l")
    cfg = _cfg(tmp_path, n_inspec_target=4, max_total_draws=20)
    rec = _shard(cfg, 0, 1, led, tmp_path)
    assert rec["n_inspec"] == 4 and rec["n_attempts"] == 4
    assert led.count() == 4
    with h5py.File(rec["path"], "r") as hf:
        w = json.loads(hf["scan/0"].attrs["parallel_worker_json"])
    assert w["shared_target"] == 4 and w["n_attempts"] == 4


def test_worker_yields_to_a_ledger_already_at_target(fake_bouquet, tmp_path):
    led = FileYieldLedger(tmp_path / "l")
    for _ in range(4):
        led.record()                               # other workers got there
    cfg = _cfg(tmp_path, n_inspec_target=4, max_total_draws=20)
    rec = _shard(cfg, 0, 1, led, tmp_path)
    assert rec["n_attempts"] == 0 and rec["n_inspec"] == 0
    assert led.count() == 4                        # nothing added


def test_two_workers_share_one_target_and_the_merge_manifests_it(fake_bouquet, tmp_path):
    led = FileYieldLedger(tmp_path / "l")
    cfg = _cfg(tmp_path, n_inspec_target=5, max_total_draws=10)
    # worker 0 (cap 5, every draw in-spec) fills the target ...
    r0 = _shard(cfg, 0, 2, led, tmp_path)
    assert r0["attempt_cap"] == 5 and r0["local_target"] == 5
    assert r0["n_inspec"] == 5 and led.count() == 5
    # ... so worker 1 exits at its first attempt boundary
    r1 = _shard(cfg, 1, 2, led, tmp_path)
    assert r1["n_attempts"] == 0
    out, n = merge_archives([r0["path"], r1["path"]], str(tmp_path / "run"),
                            scan_key=0, cleanup=True, config=cfg)
    assert n == 5
    with h5py.File(out, "r") as hf:
        m = json.loads(hf["scan/0"].attrs["parallel_manifest_json"])
    assert m["n_workers"] == 2 and m["n_draws"] == 5
    assert m["until_n"] == {"target": 5, "total_cap": 10, "n_inspec_recorded": 5}
    assert [w["n_attempts"] for w in m["workers"]] == [5, 0]
    assert m["workers"][0]["first_index"] == 0


def test_overshoot_is_kept_not_discarded(fake_bouquet, tmp_path):
    """A draw in flight when the pooled count crosses the target is archived:
    the stop is at the attempt boundary, so the merged archive holds AT LEAST
    the target, never fewer."""
    led = FileYieldLedger(tmp_path / "l")
    fake_bouquet.inspec_pattern = [True, False]    # 50 % yield
    cfg = _cfg(tmp_path, n_inspec_target=3, max_total_draws=12)
    r0 = _shard(cfg, 0, 2, led, tmp_path)          # cap 6: 3 in-spec in 5 attempts
    assert r0["n_inspec"] == 3 and r0["n_attempts"] == 5
    r1 = _shard(cfg, 1, 2, led, tmp_path)
    assert r1["n_attempts"] == 0
    out, n = merge_archives([r0["path"], r1["path"]], str(tmp_path / "run"),
                            scan_key=0, cleanup=True, config=cfg)
    assert n == 5                                  # all 5 attempts archived


def test_pooled_cap_exhausted_is_reported_not_hidden(fake_bouquet, tmp_path):
    led = FileYieldLedger(tmp_path / "l")
    fake_bouquet.inspec_pattern = [False]          # zero yield
    cfg = _cfg(tmp_path, n_inspec_target=3, max_total_draws=4)
    r0 = _shard(cfg, 0, 2, led, tmp_path)
    r1 = _shard(cfg, 1, 2, led, tmp_path)
    assert r0["n_attempts"] + r1["n_attempts"] == 4     # the whole cap, then stop
    assert led.count() == 0


def test_post_merge_filters_mark_selected_like_a_serial_run(fake_bouquet, tmp_path):
    led = FileYieldLedger(tmp_path / "l")
    fake_bouquet.inspec_pattern = [True, False]
    cfg = _cfg(tmp_path, n_inspec_target=2, max_total_draws=8)
    r0 = _shard(cfg, 0, 1, led, tmp_path)          # 2 in-spec in 3 attempts
    merge_archives([r0["path"]], str(tmp_path / "run"), scan_key=0,
                   cleanup=True, config=cfg)
    from bouquet.filtering import read_filter_flags
    before = read_filter_flags(str(tmp_path / "run.h5"), scan_key=0)
    assert all("passes_coil_filter" not in f for f in before.values())   # unfiltered
    summ = apply_filters_after_merge(cfg)
    assert summ["coil_filter_used"] == "legacy"
    after = read_filter_flags(str(tmp_path / "run.h5"), scan_key=0)
    assert [after[i]["passes_coil_filter"] for i in sorted(after)] == [True, False, True]


# ---------------------------------------------------------------------------
# SLURM merge CLI: manifest, ledger report + removal, filters
# ---------------------------------------------------------------------------

def test_cli_merge_reports_the_ledger_and_removes_it(fake_bouquet, tmp_path, capsys):
    led_path = str(tmp_path / "run_inspec.ledger")
    led = FileYieldLedger(led_path)
    cfg = _cfg(tmp_path, n_inspec_target=3, max_total_draws=6)
    r0 = _shard(cfg, 0, 2, led, tmp_path)
    r1 = _shard(cfg, 1, 2, led, tmp_path)
    bundle = dict(config=cfg.to_dict(), n_workers=2, seed=1, threads_per_worker=1,
                  n_equils_total=8, scan_key=0, out_header=str(tmp_path / "run"),
                  n_inspec_target=3, max_total_draws=6, ledger=led_path)
    bpath = str(tmp_path / "b.json")
    json.dump(bundle, open(bpath, "w"))
    _cli(["merge", bpath])
    out = capsys.readouterr().out
    assert "3 in-spec recorded against a shared target of 3" in out
    assert not os.path.exists(led_path)
    from bouquet.filtering import read_filter_flags
    flags = read_filter_flags(str(tmp_path / "run.h5"), scan_key=0)
    assert all("passes_coil_filter" in f for f in flags.values())      # filtered


# ---------------------------------------------------------------------------
# the real loop honours the hooks (structural)
# ---------------------------------------------------------------------------

def test_generate_bouquet_consults_the_hooks_where_it_should():
    from bouquet.TokaMaker_interface import generate_bouquet
    src = inspect.getsource(generate_bouquet)
    body = src.split("for count in eq_iter:", 1)[1]
    # stop_check is the FIRST thing in an attempt, before the progress tick
    assert body.index("stop_check()") < body.index("progress_callback(count)")
    # on_inspec fires right after the local count is incremented
    inc = body.index("_n_inspec_seen += 1")
    assert 0 < body.index("on_inspec()") - inc < 400
    # and a shared stop is not reported as a missed target
    assert "not _stopped_by_shared" in src
