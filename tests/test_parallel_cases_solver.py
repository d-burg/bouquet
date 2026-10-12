"""``parallel_cases`` end to end (swb, synthetic IMAS slice): 3 cases on 2 workers, given relative input
paths, equal the same cases run one per fresh process at their case seeds, bit for bit.  The inputs are
read-only, so nothing writes beside them; each worker has its own directory and mesh copy; every case ran
single-threaded.  A worker that dies mid-sweep ends it without hanging."""
import gzip
import os
import shutil
import stat
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

from test_phi_solver_draws import _oft_phi_ready

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))
_TIME = 1.2

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not (os.path.isfile(_MESH) and _oft_phi_ready()), reason="needs OFT"),
]


def _inputs(d):
    """Read-only copies of the synthetic slice."""
    os.makedirs(d)
    with gzip.open(os.path.join(_DATA, "dd_synthetic.json.gz"), "rb") as fi, \
            open(os.path.join(d, "dd.json"), "wb") as fo:
        shutil.copyfileobj(fi, fo)
    for f in ("diiid_profs_synthetic.cdf", "g_synthetic.geqdsk"):
        shutil.copy(os.path.join(_DATA, f), d)
    shutil.copy(_MESH, d)
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), stat.S_IRUSR | stat.S_IRGRP)
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)
    return {k: os.path.join(d, f) for k, f in (("dd", "dd.json"), ("ida", "diiid_profs_synthetic.cdf"),
                                                ("geq", "g_synthetic.geqdsk"),
                                                ("mesh", os.path.basename(_MESH)))}


def _sweep(inp, header):
    import bouquet as bq
    cfg = bq.Bouquet.from_imas(inp["dd"], mesh=inp["mesh"], time=_TIME, n_draws=1,
                               ida_path=inp["ida"],
                               impurity_Z=6.0, header=header, solve_method="swb").config
    cfg.generation.seed = 7
    src = cfg.source
    cases = [bq.CaseSpec(source=bq.ImasSource(**{**vars(src), "LCFS_geqdsk": g}), header="sweep",
                         scan_key=k, group="g")
             for k, g in ((1, inp["geq"]), (2, None), (3, inp["geq"]))]
    return cfg, cases


def _serial(inp, work, i):
    """Case ``i`` alone, in this (fresh) process, at its case seed."""
    import warnings
    from bouquet.parallel import case_seed, run_case_on
    from bouquet.run import Bouquet
    from bouquet.threads import pin_threads
    pin_threads()
    os.chdir(work)
    cfg, cases = _sweep(inp, os.path.join(work, "x"))
    cfg.generation.seed = case_seed(cfg.generation.seed, cases[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the per-case seed parallel_cases derives (case_seed(seed, "<group>/<scan_key>"))
        run_case_on(Bouquet(cfg).setup_solver(), cases[i], os.path.join(work, f"serial{i}"))


def _draw(h5, key):
    import h5py
    with h5py.File(h5, "r") as f:
        g = f[f"scan/{key}/0"]
        return {k: np.asarray(g[k]) for k in g if isinstance(g[k], h5py.Dataset) and g[k].dtype.kind == "f"}


def _die(bouquet, case):
    if case.scan_key == 2:
        os._exit(1)                                      # as a segfault or an OOM kill would


def test_parallel_cases_equals_one_fresh_process_per_case(tmp_path, monkeypatch):
    import bouquet as bq
    inp = _inputs(str(tmp_path / "inputs"))
    env = _harness.subprocess_env(MPLBACKEND="Agg")
    serial = [subprocess.Popen([sys.executable, os.path.abspath(__file__), str(tmp_path), str(i)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
    monkeypatch.chdir(tmp_path)                          # the sweep gets paths relative to here
    cfg, cases = _sweep({k: os.path.relpath(v) for k, v in inp.items()}, str(tmp_path / "work" / "sweep"))
    out = bq.parallel_cases(cases, cfg, "work", n_cpus_override=2)
    for p in serial:
        _, err = p.communicate()
        assert p.returncode == 0, err[-4000:]
    assert out["n_success"] == 3, out["errors"]
    for w in (0, 1):
        assert os.path.isfile(tmp_path / "work" / f"worker_{w}" / os.path.basename(_MESH))
    assert out["cases"][0].source.ids_path == inp["dd"]
    for i, rec in out["results"].items():
        assert rec["threads"] and all(p["num_threads"] == 1 for p in rec["threads"])
        a, b = _draw(out["merged"]["g"], cases[i].scan_key), _draw(str(tmp_path / f"serial{i}.h5"),
                                                                   cases[i].scan_key)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=f"case {i}: {k}")


def test_a_worker_death_ends_the_sweep_and_keeps_the_finished_cases(tmp_path):
    import bouquet as bq
    inp = _inputs(str(tmp_path / "inputs"))
    cfg, cases = _sweep(inp, str(tmp_path / "work" / "sweep"))
    out = bq.parallel_cases(cases, cfg, str(tmp_path / "work"), n_cpus_override=1, before_baseline=_die)
    assert set(out["errors"]) >= {1} and all("worker process died" in out["errors"][i] for i in out["errors"])
    assert set(out["results"]) | set(out["errors"]) == {0, 1, 2}
    assert set(out["merged"]) == ({"g"} if out["results"] else set())


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    t = sys.argv[1]
    _serial({k: os.path.join(t, "inputs", f) for k, f in (
        ("dd", "dd.json"), ("ida", "diiid_profs_synthetic.cdf"), ("geq", "g_synthetic.geqdsk"),
        ("mesh", os.path.basename(_MESH)))}, t, int(sys.argv[2]))
