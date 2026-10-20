"""Many bouquets: parallel sources, set_case, run_case, merge_cases, the worker count and the
single-thread policy.  No OFT: the per-case body runs on a stub solver; the pool's startup failure
runs for real (it fails before the solver); the rest of the pool is in tests/test_parallel_cases_solver.py."""

import json
import os
import pickle

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq
from bouquet.config import (BouquetConfig, SolverConfig, ReconstructionSource,
                            ImasSource, CaseSpec, ParallelSource,
                            GeqdskProfilePairs, IdaTimeslices, ImasTimeslices,
                            absolute_source, with_absolute_paths)
from bouquet.parallel import (_get_num_cpus, _IndexMap, case_seed, check_cases, merge_cases, run_case,
                              parallel_cases, parallel_generate, _worker_state)
from bouquet.schema import SCHEMA_VERSION
from bouquet.utils import initialize_equilibrium_database


def _ida_file(path, times_ms, n_rad=8):
    """Minimal IDA-shaped .cdf: (n_time, n_radial) profiles + a time axis [ms]."""
    with h5py.File(path, "w") as f:
        f.create_dataset("n_e", data=np.ones((len(times_ms), n_rad)))
        f.create_dataset("time", data=np.asarray(times_ms, dtype=float))
    return str(path)


# ---------------------------------------------------------------------------
# parallel sources: raw inputs -> flat list of atomic cases
# ---------------------------------------------------------------------------
class TestGeqdskProfilePairs:
    def test_one_case_per_pair(self):
        src = GeqdskProfilePairs(header="sweep",
                                 pairs=[("/d/g1.geqdsk", "/d/p1.peqdsk"),
                                        ("/d/g2.geqdsk", "/d/p2.peqdsk")])
        cases = src.expand()
        assert len(cases) == len(src) == 2
        assert [c.group for c in cases] == ["g1", "g2"]
        assert [c.source.geqdsk_path for c in cases] == ["/d/g1.geqdsk", "/d/g2.geqdsk"]
        # each pair is its own group -> the merge leaves one archive per pair
        assert len({c.group for c in cases}) == 2

    def test_source_kwargs_reach_every_case(self):
        cases = GeqdskProfilePairs(header="s", pairs=[("g", "p")],
                                   source_kwargs={"cocos": 3, "impurity_Z": 74.0}
                                   ).expand()
        assert cases[0].source.cocos == 3
        assert cases[0].source.impurity_Z == 74.0

    def test_empty_is_empty(self):
        assert GeqdskProfilePairs(header="s").expand() == []


class TestIdaTimeslices:
    def test_one_case_per_slice_grouped_by_file(self, tmp_path):
        a = _ida_file(tmp_path / "IDA_A.cdf", [2000.0, 2500.0, 3000.0])
        b = _ida_file(tmp_path / "IDA_B.cdf", [4000.0, 4500.0])
        cases = IdaTimeslices(
            header="sweep",
            inputs=[(a, [f"ga{i}.geqdsk" for i in range(3)]),
                    (b, [f"gb{i}.geqdsk" for i in range(2)])]).expand()

        assert len(cases) == 5
        # the non-atomic recast: one .cdf -> N cases, all sharing a group
        assert [c.group for c in cases] == ["IDA_A"] * 3 + ["IDA_B"] * 2
        # scan_key in ms (the run_slices convention), source.time in seconds
        assert [c.scan_key for c in cases] == [2000, 2500, 3000, 4000, 4500]
        assert [c.source.time for c in cases] == [2.0, 2.5, 3.0, 4.0, 4.5]
        # one geqdsk per slice, in time order; profiles point at the .cdf
        assert [os.path.basename(c.source.geqdsk_path) for c in cases[:3]] == \
            ["ga0.geqdsk", "ga1.geqdsk", "ga2.geqdsk"]
        assert all(c.source.profiles_path.endswith(".cdf") for c in cases)

    def test_geqdsk_count_mismatch_raises(self, tmp_path):
        a = _ida_file(tmp_path / "IDA_A.cdf", [1000.0, 2000.0, 3000.0])
        with pytest.raises(ValueError, match="one g-file per slice"):
            IdaTimeslices(header="s", inputs=[(a, ["only_one.geqdsk"])]).expand()

    def test_inconsistent_time_axis_raises(self, tmp_path):
        p = tmp_path / "bad.cdf"
        with h5py.File(p, "w") as f:
            f.create_dataset("n_e", data=np.ones((3, 8)))
            f.create_dataset("time", data=np.array([1.0, 2.0]))    # 2 != 3
        with pytest.raises(ValueError, match="inconsistent"):
            IdaTimeslices(header="s", inputs=[(str(p), list("abc"))]).expand()

    def test_no_time_axis_falls_back_to_index(self, tmp_path):
        p = tmp_path / "notime.cdf"
        with h5py.File(p, "w") as f:
            f.create_dataset("n_e", data=np.ones((2, 8)))
        cases = IdaTimeslices(header="s",
                              inputs=[(str(p), ["g0", "g1"])]).expand()
        assert [c.scan_key for c in cases] == [0, 1]
        assert [c.source.time for c in cases] == [None, None]


class TestCaseSerialization:
    """Cases cross a process boundary, so they must pickle AND round-trip JSON."""

    def test_casespec_roundtrip_and_pickle(self):
        c = CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p",
                                                 cocos=2),
                     header="h", scan_key=2500, group="A")
        assert CaseSpec.from_dict(c.to_dict()) == c
        assert pickle.loads(pickle.dumps(c)) == c
        # the source discriminator survives, so the union member is restored
        assert isinstance(CaseSpec.from_dict(c.to_dict()).source, ReconstructionSource)

    def test_group_defaults_to_header_basename(self):
        assert CaseSpec(source=ReconstructionSource(geqdsk_path="g",
                                                    profiles_path="p"),
                        header="/out/run_A").group == "run_A"

    def test_empty_header_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                     header="")

    def test_parallel_source_roundtrip_is_tagged(self, tmp_path):
        src = GeqdskProfilePairs(header="s", pairs=[("g", "p")], source_kwargs={"cocos": 2})
        back = ParallelSource.from_dict(src.to_dict())
        assert isinstance(back, GeqdskProfilePairs)
        assert back.expand() == src.expand()

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="unknown parallel_source_type"):
            ParallelSource.from_dict({"parallel_source_type": "Nope", "header": "h"})

    def test_index_map_roundtrips(self):
        cases = GeqdskProfilePairs(header="s",
                                   pairs=[("g1", "p1"), ("g2", "p2")]).expand()
        m = pickle.loads(pickle.dumps(_IndexMap(cases)))
        assert len(m) == 2 and m(1) == cases[1] and list(m) == cases


# ---------------------------------------------------------------------------
# Bouquet.set_case -- swap a whole case onto a standing solver
# ---------------------------------------------------------------------------
class TestSetCase:
    def _bouquet(self):
        cfg = BouquetConfig(
            source=ReconstructionSource(geqdsk_path="g0", profiles_path="p0"),
            solver=SolverConfig(mesh_path="/m.h5"), output_header="base")
        return bq.Bouquet(cfg)

    def test_swaps_source_header_and_scan_key(self):
        b = self._bouquet()
        case = CaseSpec(source=ReconstructionSource(geqdsk_path="g9", profiles_path="p9"),
                        header="run_9", scan_key=4400, group="G")
        b.set_case(case)
        assert b.config.source.geqdsk_path == "g9"
        assert b.output_header == "run_9"
        assert b.config.generation.scan_key == 4400

    def test_clears_cached_state_so_the_next_case_re_solves(self):
        b = self._bouquet()
        b.baseline = "stale"
        b._resolved_uncertainty = "stale"
        b.diagnostics = ["stale"]
        b._selection = {"stale": 1}
        b.set_case(source=ReconstructionSource(geqdsk_path="g1", profiles_path="p1"))
        assert (b.baseline, b._resolved_uncertainty, b.diagnostics,
                b._selection) == (None, None, None, None)

    def test_rejects_casespec_mixed_with_keywords(self):
        b = self._bouquet()
        case = CaseSpec(source=b.config.source, header="h")
        with pytest.raises(TypeError, match="not both"):
            b.set_case(case, header="other")

    def _up(self, b):                   # pretend the solver is up
        from bouquet.config import resolve_solve_method
        b.mygs = object()
        b._solver_key = ("/m.h5", resolve_solve_method(b.config.generation), 1)
        return b

    @pytest.mark.parametrize("change", ["mesh", "nthreads", "method"])
    def test_a_per_solver_setting_change_after_setup_solver_raises(self, change):
        b = self._up(self._bouquet())
        if change == "mesh":
            b.config.solver.mesh_path = "/different.h5"
        elif change == "nthreads":
            b.config.solver.nthreads = 2
        else:
            b.config.generation.solve_method = "legacy"
        with pytest.raises(ValueError, match="fixed per solver"):
            b.set_case(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"))

    def test_set_slice_still_works_through_set_case(self):
        b = self._bouquet()
        b.baseline = "stale"
        b.set_slice(time=3.3, header="h2")
        assert b.config.source.time == 3.3
        assert b.output_header == "h2"
        assert b.baseline is None


# ---------------------------------------------------------------------------
# ImasTimeslices -- the TIME sweep over one IDS
# ---------------------------------------------------------------------------
class TestImasTimeslices:
    def _source(self, **kw):
        kw.setdefault("times", [3.163, 3.263, 3.363])
        return ImasTimeslices(header="sweep", ids_path="/data/dd_sim.json",
                              ida_path="/data/IDA_synthetic.cdf", **kw)

    def test_one_case_per_time_labelled_in_ms(self):
        cases = self._source().expand()
        assert [c.scan_key for c in cases] == [3163, 3263, 3363]
        # times stay in SECONDS on the source (what read_ida takes)
        assert [c.source.time for c in cases] == [3.163, 3.263, 3.363]
        assert all(isinstance(c.source, ImasSource) for c in cases)
        assert all(c.source.ids_path == "/data/dd_sim.json" for c in cases)
        assert all(c.source.ida_path == "/data/IDA_synthetic.cdf" for c in cases)

    def test_whole_sweep_is_one_group_so_the_merge_writes_one_archive(self):
        cases = self._source().expand()
        assert {c.group for c in cases} == {"dd_sim"}       # the IDS stem
        assert {c.header for c in cases} == {"sweep_dd_sim"}

    def test_group_override(self):
        assert {c.group for c in self._source(group="run1").expand()} == {"run1"}

    def test_one_shared_gfile_is_broadcast(self):
        cases = self._source(LCFS_geqdsk="/data/g000001.00000").expand()
        assert [c.source.LCFS_geqdsk for c in cases] == ["/data/g000001.00000"] * 3

    def test_one_gfile_per_slice_is_paired_in_order(self):
        gs = ["/g.03160", "/g.03260", "/g.03360"]
        cases = self._source(LCFS_geqdsk=gs).expand()
        assert [c.source.LCFS_geqdsk for c in cases] == gs

    def test_gfile_count_mismatch_raises(self):
        # a silent mis-pairing would put the wrong separatrix on every slice
        with pytest.raises(ValueError, match="give one per time"):
            self._source(LCFS_geqdsk=["/g.03160", "/g.03260"]).expand()

    def test_source_kwargs_reach_every_case(self):
        cases = self._source(source_kwargs={"impurity_Z": 10.0}).expand()
        assert all(c.source.impurity_Z == 10.0 for c in cases)

    def test_times_closer_than_a_millisecond_are_rejected(self):
        # both would label as scan/3263/ and silently overwrite in the merge
        with pytest.raises(ValueError, match="both key as 3263 ms"):
            self._source(times=[3.2631, 3.2634]).expand()

    def test_empty_sweep_raises(self):
        with pytest.raises(ValueError, match="times is empty"):
            self._source(times=[]).expand()

    def test_missing_ids_path_raises(self):
        with pytest.raises(ValueError, match="ids_path must be set"):
            ImasTimeslices(header="sweep", times=[1.0]).expand()

    def test_len_is_the_case_count(self):
        assert len(self._source()) == 3

    def test_round_trips_through_the_tagged_registry(self):
        src = self._source(LCFS_geqdsk="/data/g", source_kwargs={"impurity_Z": 6.0})
        back = ParallelSource.from_dict(src.to_dict())
        assert isinstance(back, ImasTimeslices)
        assert [c.to_dict() for c in back.expand()] == \
               [c.to_dict() for c in src.expand()]


# ---------------------------------------------------------------------------
# run_case -- one case on a worker's standing solver (stubbed)
# ---------------------------------------------------------------------------
class _StubBouquet:
    """Stands in for the worker's Bouquet: records the calls run_case makes."""

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.calls = []
        self.output_header = None
        self.case = None
        self.baseline = type("BL", (), {"l_i_target": 1.25, "Ip_target": 1.1e6})()
        self._geom = type("G", (), {"F0": 1.7})()
        self.config = type("C", (), {"generation": type("G", (), {"seed": 0})()})()

    def _step(self, name):
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"boom in {name}")

    def set_case(self, case):
        self.case = case
        self._step("set_case")
        return self

    def prepare_baseline(self):
        self._step("prepare_baseline")

    def generate(self):
        self._step("generate")

    def filter(self):
        self._step("filter")

    def selected_indices(self, selection="selected"):
        return [0, 1, 2] if selection == "all" else [0, 1]


@pytest.fixture
def stub_worker():
    """Install a stub Bouquet in the module-level worker state, then restore."""
    saved = dict(_worker_state)

    def _install(**kw):
        hooks = kw.pop("hooks", {})
        stub = _StubBouquet(**kw)
        _worker_state.clear()
        _worker_state.update(worker_id=0, nthreads=1, bouquet=stub,
                             working_dir=os.getcwd(), hooks=dict(hooks))
        return stub

    yield _install
    _worker_state.clear()
    _worker_state.update(saved)


class TestRunCase:
    def _case(self):
        return CaseSpec(
            source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
            header="sweep_A", scan_key=2500, group="A")

    def test_runs_the_stages_and_reports_the_case(self, stub_worker, tmp_path):
        from bouquet.threads import pin_threads
        stub = stub_worker()
        with pin_threads():          # what _init_case_worker does in a worker
            idx, ok, err, out = run_case((7, self._case(), str(tmp_path), 11))

        assert ok and err is None
        # no export(): the merge produces the deliverable, so a per-case
        # {header}_selected.h5 would be pure duplication
        assert stub.calls == ["set_case", "prepare_baseline", "generate", "filter"]
        # archive is absolute, in the shared case dir, disambiguated by index
        assert out["path"] == str(tmp_path / "sweep_A_idx7.h5")
        assert stub.output_header == str(tmp_path / "sweep_A_idx7")
        # the grouping the merge needs travels with the result
        assert (out["idx"], out["group"], out["scan_key"]) == (7, "A", 2500)
        assert (out["n_all"], out["n_sel"]) == (3, 2)
        assert out["l_i_target"] == 1.25 and out["Ip_target"] == 1.1e6 and out["F0"] == 1.7
        assert stub.config.generation.seed == 11        # the case's own seed, set before the case runs
        # the pinned thread counts travel with the case record
        assert out["threads_pinned"] is True
        assert all(p["num_threads"] == 1 for p in out["threads"])

    def test_failure_comes_back_as_data_not_an_exception(self, stub_worker, tmp_path):
        # one bad slice must not take down a sweep of hundreds
        stub_worker(fail_at="generate")
        idx, ok, err, out = run_case((3, self._case(), str(tmp_path), 11))
        assert (idx, ok, out) == (3, False, None)
        assert "boom in generate" in err and "Traceback" in err

    def test_hooks_bracket_the_baseline_solve(self, stub_worker, tmp_path):
        # before_baseline sets SOLVER targets (consumed by prepare_baseline);
        # after_baseline needs the resolved baseline's grid. Order is the point.
        seen = []

        def before(bouquet, case):
            bouquet.calls.append("hook:before")
            seen.append(("before", case.scan_key))

        def after(bouquet, case):
            bouquet.calls.append("hook:after")
            seen.append(("after", case.scan_key))

        stub = stub_worker(hooks={"before_baseline": before,
                                  "after_baseline": after})
        idx, ok, err, out = run_case((0, self._case(), str(tmp_path), 11))

        assert ok, err
        assert stub.calls == ["set_case", "hook:before", "prepare_baseline",
                              "hook:after", "generate", "filter"]
        assert seen == [("before", 2500), ("after", 2500)]

    def test_a_raising_hook_fails_only_its_own_case(self, stub_worker, tmp_path):
        def boom(bouquet, case):
            raise RuntimeError("bad switchboard")

        stub_worker(hooks={"after_baseline": boom})
        idx, ok, err, out = run_case((5, self._case(), str(tmp_path), 11))
        assert (ok, out) == (False, None)
        assert "bad switchboard" in err

    def test_no_hooks_configured_is_the_default_path(self, stub_worker, tmp_path):
        stub = stub_worker()
        _, ok, err, _ = run_case((0, self._case(), str(tmp_path), 11))
        assert ok, err
        assert stub.calls == ["set_case", "prepare_baseline", "generate", "filter"]


class TestHookValidation:
    """parallel_cases rejects un-spawnable hooks before standing up the pool."""

    def _cfg(self):
        return BouquetConfig(
            source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
            solver=SolverConfig(mesh_path="mesh.h5"), output_header="sweep")

    def _cases(self):
        return [CaseSpec(source=ReconstructionSource(geqdsk_path="g",
                                                     profiles_path="p"),
                         header="sweep_A", scan_key=1, group="A")]

    def test_a_lambda_is_rejected_with_a_pointed_message(self, tmp_path):
        # a closure only fails once the pool is up, as an opaque init error
        with pytest.raises(TypeError, match="module-level function"):
            parallel_cases(self._cases(), self._cfg(), str(tmp_path),
                           n_cpus_override=1,
                           after_baseline=lambda b, c: None)

    def test_a_non_callable_is_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="before_baseline must be callable"):
            parallel_cases(self._cases(), self._cfg(), str(tmp_path),
                           n_cpus_override=1, before_baseline="not a function")


class TestPoolStartup:
    def test_a_worker_that_cannot_start_fails_the_run_with_its_traceback(self, tmp_path):
        # the mesh copy fails before any solver: a real spawned pool, no OFT
        cfg = BouquetConfig(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                            solver=SolverConfig(mesh_path=str(tmp_path / "no_mesh.h5")), output_header="sweep")
        cases = [CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                          header="sweep", scan_key=k, group="A") for k in (1, 2, 3)]
        with pytest.raises(RuntimeError, match=r"failed to initialise(.|\n)*no_mesh\.h5"):
            parallel_cases(cases, cfg, str(tmp_path / "work"), n_cpus_override=2)


class TestRelativePaths:
    """Workers run in their own directories, so inputs are made absolute in the parent."""

    def test_source_and_config_inputs_become_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = absolute_source(ImasSource(ids_path="dd.json", time=1.0, ida_path="ida.cdf", LCFS_geqdsk="g"))
        assert (src.ids_path, src.ida_path, src.LCFS_geqdsk) == tuple(
            str(tmp_path / f) for f in ("dd.json", "ida.cdf", "g"))
        cfg = BouquetConfig(source=ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf"),
                            solver=SolverConfig(mesh_path="mesh.h5"), output_header="sweep")
        out = with_absolute_paths(cfg)
        assert (out.solver.mesh_path, out.source.geqdsk_path, out.source.profiles_path) == tuple(
            str(tmp_path / f) for f in ("mesh.h5", "g", "p.cdf"))
        assert cfg.solver.mesh_path == "mesh.h5"           # a copy: the caller's config is untouched

    def test_unset_optional_inputs_stay_unset(self):
        src = absolute_source(ImasSource(ids_path="/d/dd.json", time=1.0))
        assert src.ida_path is None and src.LCFS_geqdsk is None


class TestCaseSeed:
    def _case(self, key, group="A"):
        return CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                        header="sweep", scan_key=key, group=group)

    def test_each_case_gets_its_own_reproducible_seed(self):
        seeds = {case_seed(7, self._case(k, g)) for k in (1, 2) for g in ("A", "B")}
        assert len(seeds) == 4
        assert case_seed(7, self._case(1)) == case_seed(7, self._case(1)) != case_seed(8, self._case(1))

    def test_units_from_cases_gives_each_case_the_same_seed(self):
        from bouquet import units
        cfg = BouquetConfig(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                            solver=SolverConfig(mesh_path="mesh.h5"), output_header="sweep")
        cases = [self._case(1), self._case(2)]
        us = units.units_from_cases(cfg, cases)
        # the default seed (None): one entropy base for the sweep, stamped on every unit
        assert cfg.generation.seed is None
        assert {(u["seed_base"], u["seed_rule"]) for u in us} == {(us[0]["seed_base"], "entropy")}
        assert [u["config"]["generation"]["seed"] for u in us] == [case_seed(us[0]["seed_base"], c)
                                                                   for c in cases]
        # ... so two sweeps at the default seed are not identical (review of #78, A5)
        assert [u["config"]["generation"]["seed"] for u in units.units_from_cases(cfg, cases)] != \
            [u["config"]["generation"]["seed"] for u in us]
        cfg.generation.seed = 7
        us = units.units_from_cases(cfg, cases)
        assert [u["config"]["generation"]["seed"] for u in us] == [case_seed(7, c) for c in cases]
        assert {(u["seed_base"], u["seed_rule"]) for u in us} == {(7, "config")}

    def test_parallel_cases_derives_case_seeds_from_one_stamped_base(self):
        """parallel_cases draws the base once in the parent (sweep_seed_base: entropy for seed=None) and
        stamps seed, seed_base and seed_rule on each case record (the pool itself is not run here)."""
        import inspect

        from bouquet.parallel import parallel_cases
        src = inspect.getsource(parallel_cases)
        assert "seed_base, seed_rule = sweep_seed_base(config.generation.seed)" in src
        assert "case_seed(seed_base, c)" in src and "case_seed(seed," not in src.replace(
            "case_seed(seed_base", "")
        assert "seed_base=seed_base" in src and "seed_rule=seed_rule" in src


# ---------------------------------------------------------------------------
# merge_cases -- reassemble the sweep along the INPUT structure
# ---------------------------------------------------------------------------
def _case_archive(dirpath, idx, group, scan_key, n_draws=2):
    """A per-case archive shaped like one Bouquet.generate() wrote it."""
    stem = os.path.join(str(dirpath), f"case_{idx}")
    initialize_equilibrium_database(stem)
    with h5py.File(stem + ".h5", "a") as f:
        g = f.require_group(f"scan/{scan_key}")
        g.create_dataset("config_json", data='{"case": %d}' % idx)
        g.create_group("_baseline").attrs["l_i_target"] = 1.0 + idx
        for i in range(n_draws):
            d = g.create_group(str(i))
            d.attrs["count"] = i
            d.create_dataset("jphi_total", data=np.arange(4.0) + i)
    return dict(idx=idx, path=os.path.abspath(stem + ".h5"), group=group,
                scan_key=scan_key, n_all=n_draws, n_sel=n_draws)


class TestMergeCases:
    def test_one_archive_per_input_group(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "IDA_A", 2000),
                   _case_archive(tmp_path, 1, "IDA_A", 2500),
                   _case_archive(tmp_path, 2, "IDA_B", 4000)]
        merged = merge_cases(results, str(tmp_path / "sweep"))

        assert set(merged) == {"IDA_A", "IDA_B"}
        assert merged["IDA_A"].endswith("sweep_IDA_A.h5")
        with h5py.File(merged["IDA_A"], "r") as f:
            assert sorted(f["scan"].keys(), key=int) == ["2000", "2500"]
            # each case's own provenance + baseline ride along inside its group
            assert f["scan/2500/_baseline"].attrs["l_i_target"] == 2.0
            assert f["scan/2500/config_json"][()] == b'{"case": 1}'
            assert sorted(k for k in f["scan/2000"] if k.isdigit()) == ["0", "1"]
            assert f.attrs["schema_version"] == SCHEMA_VERSION
        with h5py.File(merged["IDA_B"], "r") as f:
            assert list(f["scan"].keys()) == ["4000"]

    def test_the_merged_archive_carries_the_sweeps_seed_stamp(self, tmp_path):
        from bouquet.units import sweep_seed_base
        base, rule = sweep_seed_base(None)
        results = [dict(_case_archive(tmp_path, i, "A", k), seed=case_seed(base, self._c(k)),
                        seed_base=base, seed_rule=rule) for i, k in ((0, 2000), (1, 2500))]
        merged = merge_cases(results, str(tmp_path / "sweep"))
        with h5py.File(merged["A"], "r") as f:
            assert f.attrs["seed_rule"] == "entropy" and int(f.attrs["seed_base"]) == base
            rows = json.loads(f.attrs["merge_case_seeds_json"])
        assert [r["seed"] for r in rows] == [results[0]["seed"], results[1]["seed"]]
        assert {r["seed_rule"] for r in rows} == {"entropy"}
        # records without a stamp (a direct merge of older records) write none
        merged = merge_cases([_case_archive(tmp_path, 2, "B", 3000)], str(tmp_path / "old"))
        with h5py.File(merged["B"], "r") as f:
            assert "seed_rule" not in f.attrs and "merge_case_seeds_json" not in f.attrs

    @staticmethod
    def _c(key):
        return CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
                        header="sweep", scan_key=key, group="A")

    def test_merged_archive_is_readable_by_the_archive_api(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 1, "A", 2500)]
        merged = merge_cases(results, str(tmp_path / "sweep"))
        arch = bq.BouquetArchive(merged["A"][:-3])
        assert arch.scan_keys == ["2000", "2500"]
        assert arch.scan(2000).indices == [0, 1]

    def test_group_by_none_gives_one_combined_archive(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 1, "B", 4000)]
        merged = merge_cases(results, str(tmp_path / "all"), group_by=None)
        assert list(merged) == [None]
        with h5py.File(merged[None], "r") as f:
            assert sorted(f["scan"].keys(), key=int) == ["2000", "4000"]

    def test_duplicate_scan_key_in_a_group_raises(self, tmp_path):
        # would silently overwrite one case with another: the sweep would look
        # complete while holding fewer bouquets than it ran
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 1, "A", 2000)]
        with pytest.raises(ValueError, match="duplicate scan_key"):
            merge_cases(results, str(tmp_path / "sweep"))

    def test_missing_case_archive_raises(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000)]
        results[0]["path"] = str(tmp_path / "vanished.h5")
        with pytest.raises(FileNotFoundError, match="case archive not found"):
            merge_cases(results, str(tmp_path / "sweep"))

    def test_wrong_scan_key_raises(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000)]
        results[0]["scan_key"] = 9999
        with pytest.raises(KeyError, match="different scan_key"):
            merge_cases(results, str(tmp_path / "sweep"))

    def test_cleanup_removes_the_per_case_archives(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000)]
        merge_cases(results, str(tmp_path / "sweep"), cleanup=True)
        assert not os.path.exists(results[0]["path"])

    def test_rerun_replaces_rather_than_appends(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 1, "A", 2500)]
        merge_cases(results, str(tmp_path / "sweep"))
        merged = merge_cases(results[:1], str(tmp_path / "sweep"), overwrite=True)
        with h5py.File(merged["A"], "r") as f:      # stale 2500 must be gone
            assert list(f["scan"].keys()) == ["2000"]

    def test_an_existing_merge_is_not_replaced_unless_asked(self, tmp_path):
        # a re-run that lost cases must not silently replace a complete merge
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 1, "A", 2500)]
        merged = merge_cases(results, str(tmp_path / "sweep"))
        with pytest.raises(FileExistsError, match="overwrite=True"):
            merge_cases(results[:1], str(tmp_path / "sweep"))
        with h5py.File(merged["A"], "r") as f:      # untouched
            assert sorted(f["scan"].keys(), key=int) == ["2000", "2500"]
            assert f.attrs["merge_partial"] == False  # noqa: E712
            assert f.attrs["merge_expected_cases"] == 2

    def test_a_failed_case_is_recorded_and_marks_a_partial_merge(self, tmp_path):
        import json
        results = [_case_archive(tmp_path, 0, "A", 2000),
                   _case_archive(tmp_path, 2, "B", 4000)]
        failed = [dict(idx=1, group="A", scan_key=2500,
                       error="Traceback (most recent call last):\n  ...\nRuntimeError: synthetic failure"),
                  dict(idx=3, group="C", scan_key=6000, error="ValueError: also synthetic")]
        with pytest.warns(RuntimeWarning, match="PARTIAL merge"):
            merged = merge_cases(results, str(tmp_path / "sweep"), failed=failed)
        # a group whose only case failed still gets its archive: never only in errors.pkl
        assert set(merged) == {"A", "B", "C"}
        with h5py.File(merged["A"], "r") as f:
            assert sorted(f["scan"].keys(), key=int) == ["2000", "2500"]
            assert f["scan/2500"].attrs["refused_reason"] == \
                "parallel case 1 failed: RuntimeError: synthetic failure"
            assert not any(k.isdigit() for k in f["scan/2500"])
            assert f.attrs["merge_partial"] == True  # noqa: E712
            assert f.attrs["merge_expected_cases"] == 2
            part = json.loads(f.attrs["merge_partial_json"])
            assert part == dict(expected_cases=2, merged_cases=[0], failed_cases=[1])
            assert json.loads(f.attrs["merge_failed_cases_json"])[0]["scan_key"] == 2500
        with h5py.File(merged["B"], "r") as f:      # complete groups are not marked
            assert f.attrs["merge_partial"] == False  # noqa: E712
            assert "merge_partial_json" not in f.attrs
        with h5py.File(merged["C"], "r") as f:
            assert f.attrs["merge_partial"] == True and list(f["scan"].keys()) == ["6000"]  # noqa: E712
        # a series reader reports the failed case, not a gap
        ev = {"_baseline": {"q": {"value": 1.0, "status": "ok", "regular": True}}}
        assert bq.draw_band(merged["A"], "2500", ev)[None].status == "refused"

    def test_a_failed_case_cannot_hide_a_duplicate_scan_key(self, tmp_path):
        results = [_case_archive(tmp_path, 0, "A", 2000)]
        with pytest.raises(ValueError, match="duplicate scan_key"):
            merge_cases(results, str(tmp_path / "sweep"),
                        failed=[dict(idx=1, group="A", scan_key=2000, error="x")])


# ---------------------------------------------------------------------------
# worker/thread budget
# ---------------------------------------------------------------------------
_SCHED = ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "LSB_DJOB_NUMPROC", "NSLOTS")


class TestGetNumCpus:
    def test_counts_the_affinity_mask(self, monkeypatch):
        for v in _SCHED:
            monkeypatch.delenv(v, raising=False)
        if hasattr(os, "sched_getaffinity"):
            assert _get_num_cpus() == len(os.sched_getaffinity(0))
        else:                       # macOS / Windows: no affinity mask, the logical count
            assert _get_num_cpus() == (os.cpu_count() or 1)

    def test_physical_cores_never_exceed_logical(self, monkeypatch):
        for v in _SCHED:
            monkeypatch.delenv(v, raising=False)
        assert 1 <= _get_num_cpus(use_logical=False) <= _get_num_cpus()

    @pytest.mark.parametrize("var", _SCHED)
    def test_scheduler_allocation_caps_the_worker_count(self, monkeypatch, var):
        for v in _SCHED:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(var, "1")
        assert _get_num_cpus() == 1 and _get_num_cpus(use_logical=False) == 1

    def test_cap_never_inflates_beyond_the_affinity_mask(self, monkeypatch):
        for v in _SCHED:
            monkeypatch.delenv(v, raising=False)
        avail = _get_num_cpus()
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "100000")
        assert _get_num_cpus() == avail


# ---------------------------------------------------------------------------
# one thread per process
# ---------------------------------------------------------------------------
class TestSingleThread:
    def _cfg(self, **solver):
        return BouquetConfig(source=ImasSource(ids_path="d.json", time=1.0),
                             solver=SolverConfig(mesh_path="/m.h5", **solver), output_header="h")

    def test_a_parallel_config_above_one_thread_is_refused(self):
        with pytest.raises(ValueError, match="one thread per process"):
            check_cases([], self._cfg(nthreads=2))

    def test_parallel_generate_refuses_threads_per_worker_above_one(self):
        with pytest.raises(ValueError, match="one thread per process"):
            parallel_generate(self._cfg(), threads_per_worker=2)

    def test_swb_takes_ids_cases_only(self):
        cfg = self._cfg()
        cfg.generation.solve_method = "swb"
        g = CaseSpec(source=ReconstructionSource(geqdsk_path="g", profiles_path="p"), header="h")
        with pytest.raises(ValueError, match="IDS sources only"):
            check_cases([g], cfg)
        check_cases([CaseSpec(source=cfg.source, header="h")], cfg)

    def test_a_pin_is_scoped_and_restores_the_process(self, monkeypatch):
        """pin_threads() is reversible: the environment variables (set ones to
        their value, unset ones unset) and every loaded pool's thread count
        come back, so an in-process caller does not leave the session pinned."""
        from threadpoolctl import threadpool_info
        from bouquet.threads import THREAD_VARS, pin_threads, pinned, thread_report
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        for v in THREAD_VARS[1:] + ("HDF5_USE_FILE_LOCKING",):
            monkeypatch.delenv(v, raising=False)
        before_env = {k: os.environ.get(k) for k in THREAD_VARS + ("HDF5_USE_FILE_LOCKING",)}
        before_pools = [(p["filepath"], p["num_threads"]) for p in threadpool_info()]
        assert not pinned()
        with pin_threads() as pin:
            assert pinned()
            assert all(os.environ[v] == "1" for v in THREAD_VARS)
            assert os.environ["HDF5_USE_FILE_LOCKING"] == "FALSE"
            assert all(p["num_threads"] == 1 for p in thread_report())
            assert pin.report == thread_report()
        assert not pinned()
        assert {k: os.environ.get(k) for k in before_env} == before_env
        assert [(p["filepath"], p["num_threads"]) for p in threadpool_info()] == before_pools

    def test_unpin_threads_undoes_the_most_recent_pin(self, monkeypatch):
        from bouquet.threads import pin_threads, pinned, unpin_threads
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        outer = pin_threads()
        inner = pin_threads()
        assert unpin_threads() and not inner.pinned and outer.pinned and pinned()
        outer.unpin()
        outer.unpin()                 # idempotent
        assert not pinned() and not unpin_threads()
        assert "OPENBLAS_NUM_THREADS" not in os.environ

    def test_the_worker_environment(self):
        from bouquet.threads import THREAD_VARS, worker_env
        env = worker_env({})
        assert all(env[v] == "1" for v in THREAD_VARS) and env["HDF5_USE_FILE_LOCKING"] == "FALSE"

    def test_a_pinned_process_runs_every_loaded_pool_at_one_thread(self):
        """A fresh interpreter (this module as a script, importing THIS tree), with no thread variables
        set, loads its pools and pins: every pool threadpoolctl can see runs at one thread."""
        import json
        import subprocess
        import sys

        import _harness
        env = _harness.subprocess_env()
        for k in [k for k in env if k.endswith("_NUM_THREADS")]:
            del env[k]
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--thread-probe"],
                             capture_output=True, text=True, env=env, check=True)
        pools = json.loads(out.stdout.strip().splitlines()[-1])
        if not pools:
            pytest.skip("no thread pool threadpoolctl can see in this interpreter (e.g. Apple Accelerate "
                        "BLAS and no OpenFUSIONToolkit): nothing to observe")
        assert all(p["num_threads"] == 1 for p in pools)


def _thread_probe():
    """Child of test_a_pinned_process_runs_every_loaded_pool_at_one_thread: load the numeric stack (and
    OpenFUSIONToolkit's OpenMP runtime when it is importable), pin, print the pools."""
    import json

    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    try:
        from bouquet.paths import add_oft_to_path
        add_oft_to_path()
        import OpenFUSIONToolkit.TokaMaker  # noqa: F401
    except Exception:
        pass
    from bouquet.threads import pin_threads, thread_report
    pin_threads()
    print(json.dumps(thread_report()))


if __name__ == "__main__":
    import sys

    import _harness
    _harness.assert_bouquet_is_repo_local()
    if "--thread-probe" in sys.argv:
        _thread_probe()
