"""bouquet.units: unit building, records, failures through the queue (solver-free), and two real units."""
import gzip
import json
import os
import shutil

import h5py
import pytest

from bouquet import units, workqueue as wq
from bouquet.config import BouquetConfig, ImasSource, SolverConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))


def _config(ids_path=_DD):
    return BouquetConfig(source=ImasSource(ids_path=ids_path, time=1.2), solver=SolverConfig(mesh_path=_MESH),
                         output_header="x")


def test_make_units():
    cfg = _config()
    us = units.make_units(cfg, [{"source.time": 1.0, "generation.scan_key": 1000},
                                {"source.time": 1.1, "generation.scan_key": 1100}], ["a_1000", "a_1100"])
    assert [u["id"] for u in us] == ["a_1000", "a_1100"]
    c = [BouquetConfig.from_dict(u["config"]) for u in us]
    assert (c[0].source.time, c[1].generation.scan_key) == (1.0, 1100)
    assert c[0].generation.seed == units.unit_seed(cfg.generation.seed, "a_1000") != c[1].generation.seed
    json.dumps(us)
    with pytest.raises(KeyError):
        units.make_units(cfg, [{"source.no_such_field": 1}], ["b"])


def test_make_units_stores_absolute_inputs(tmp_path, monkeypatch):
    # a unit runs in its worker's directory, not where it was made
    monkeypatch.chdir(tmp_path)
    cfg = BouquetConfig(source=ImasSource(ids_path="dd.json", time=1.0, LCFS_geqdsk="g"),
                        solver=SolverConfig(mesh_path="mesh.h5"), output_header="x")
    c = units.make_units(cfg, [{"generation.seed": 5}], ["a"])[0]["config"]
    assert (c["solver"]["mesh_path"], c["source"]["ids_path"], c["source"]["LCFS_geqdsk"]) == tuple(
        str(tmp_path / f) for f in ("mesh.h5", "dd.json", "g"))
    assert c["generation"]["seed"] == 5                  # an explicit seed is kept, not re-derived


def test_a_unit_the_queue_failed_gets_a_record_and_retry_reopens_it(tmp_path):
    # crashed or preempted on every attempt: the queue fails it without running it to the end
    q, out = str(tmp_path / "q"), str(tmp_path / "out")
    units.add_units(q, units.make_units(_config(), [{}], ["u"]))
    qq = wq.Queue(q)
    for i in range(qq.MAX_ATTEMPTS):
        assert qq.next_unit(wq.Cursor(), f"w{i}") == "u"
        qq.expire("claims", "u")
    assert qq.next_unit(wq.Cursor(), "x") is None
    units.record_queue_failures(q, out)
    rec = json.load(open(os.path.join(out, "u.json")))
    assert rec["status"] == "failed" and rec["error"]["signature"] == "max_attempts"
    units.main(["retry-failed", "--queue", q, "--out-dir", out])
    assert not os.path.exists(os.path.join(out, "u.json"))
    assert qq.next_unit(wq.Cursor(), "y") == "u"         # a fresh attempt budget


def test_archive_counts(tmp_path):
    h5 = tmp_path / "a.h5"
    with h5py.File(h5, "w") as f:
        for c, sel in ((0, True), (1, False), (2, True)):
            g = f.create_group(f"scan/1000/{c}")
            g.attrs["selected"] = sel
            g.attrs["profile_coord"] = "phi_n"
        f.create_group("scan/1000/_baseline")
    assert units.archive_counts(str(h5)) == ([0, 1, 2], [0, 2], "phi_n")


def test_failing_units_get_failed_records(tmp_path):
    q = str(tmp_path / "q")
    wq.Queue(q)
    wq.write_json(os.path.join(q, "config.json"), dict(MAX_CONSECUTIVE_FAILURES=10, STALE_S=5))
    bad = units.make_units(_config(ids_path=str(tmp_path / "missing.json")), [{}, {}], ["u1", "u2"])
    assert units.add_units(q, bad) == 2
    out = str(tmp_path / "out")
    reasons = units.work(q, out, workers=2, idle_exit=0)
    assert set(reasons.values()) <= {"drained", "idle"}
    for uid in ("u1", "u2"):
        rec = json.load(open(os.path.join(out, f"{uid}.json")))
        assert rec["status"] == "failed" and rec["archive"] is None and rec["error"]["signature"]
        assert not os.path.exists(os.path.join(out, f"{uid}.h5"))
    s = wq.Queue(q).status()
    assert (s["failed"], s["open"]) == (2, 0)


def _oft_ready():
    try:
        from test_phi_solver_draws import _oft_phi_ready
        return _oft_phi_ready()
    except Exception:
        return False


@pytest.mark.solver
@pytest.mark.skipif(not (os.path.isfile(_MESH) and _oft_ready()), reason="needs OFT")
def test_two_units_end_to_end(tmp_path):
    dd = str(tmp_path / "dd.json")
    with gzip.open(_DD, "rb") as fi, open(dd, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    import bouquet as bq
    run = bq.Bouquet.from_imas(dd, mesh=_MESH, time=1.2, n_draws=1, ida_path=_IDA, LCFS_geqdsk=_GEQ,
                               impurity_Z=6.0, solve_method="swb")
    run.config.generation.seed = 7              # unseeded units stay unseeded
    q, out = str(tmp_path / "q"), str(tmp_path / "out")
    units.add_units(q, units.make_units(run.config, [{"generation.scan_key": 1200}] * 2, ["s_a", "s_b"]))
    units.work(q, out, workers=2, idle_exit=0)
    seeds = set()
    for uid in ("s_a", "s_b"):
        rec = json.load(open(os.path.join(out, f"{uid}.json")))
        assert rec["status"] == "done" and rec["n_draws"] == 1 and os.path.isfile(os.path.join(out, rec["archive"]))
        assert rec["threads"] and all(p["num_threads"] == 1 for p in rec["threads"])
        seeds.add(rec["seed"])
        assert bq.BouquetArchive(os.path.join(out, rec["archive"])).scan_keys
    assert len(seeds) == 2
    # each worker ran its units in its own directory, on its own copy of the mesh
    wdirs = [d for d in os.listdir(os.path.join(out, "_work")) if any(
        f.endswith("_" + os.path.basename(_MESH)) for f in os.listdir(os.path.join(out, "_work", d)))]
    assert wdirs


def test_run_unit_releases_the_dd_cache_after_the_baseline(tmp_path, monkeypatch):
    # one unit per process: nothing needs the parsed dd once the baseline is read
    import bouquet.parallel as P
    from bouquet.run import Bouquet
    q = str(tmp_path / "q")
    units.add_units(q, units.make_units(_config(), [{}], ["u"]))
    seen = {}

    def fake_run_case_on(bouquet, case, header, hooks=None):
        seen["hooks"] = hooks
        raise RuntimeError("stop")
    monkeypatch.setattr(P, "run_case_on", fake_run_case_on)
    monkeypatch.setattr(Bouquet, "setup_solver", lambda self: self)
    # run_unit's process-wide setup (thread pins, OFT on sys.path) must not leak into this process
    monkeypatch.setattr("bouquet.threads.pin_threads", lambda: None)
    monkeypatch.setattr("bouquet.paths.add_oft_to_path", lambda: None)
    with pytest.raises(RuntimeError, match="stop"):
        units.run_unit(q, "u", str(tmp_path / "out"))
    assert seen["hooks"] == {"after_baseline": P.release_dd_cache}
