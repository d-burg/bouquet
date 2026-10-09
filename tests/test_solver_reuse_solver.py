"""One solver reused across baselines gives exactly what a fresh solver gives (swb and the engine).

Two cases on the synthetic IMAS slice (tests/data) with different geometry: A has the g-file LCFS and the
IDS F0; B has the IDS boundary outline and F0 x 1.1.  One process runs A -> B -> A on one solver
(``set_case``); fresh processes run A and B alone.  The baseline (psi, coil currents, F at the boundary)
and one seeded draw must match bit for bit (one thread per process).  The swb draw runs under the strong
coil reg toward solve A's coils, installed after each per-solve reset, so the per-solve reset to the
case's own F0 and isoflux is covered too.  Every run is a subprocess (``OFT_env`` is one per process).
"""
import gzip
import json
import os
import shutil
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

from test_phi_solver_draws import _oft_phi_ready

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))
_TIME = 1.2
_METHODS = ("swb", "engine")

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not (all(os.path.isfile(p) for p in (_DD, _IDA, _GEQ, _MESH)) and _oft_phi_ready()),
                       reason="needs OFT"),
]


def _run_cases(work, method, labels):
    """Run ``labels`` (A/B) in order on ONE solver; write each case's results to ``<label><i>.npz``."""
    import h5py
    import warnings
    import bouquet as bq
    from bouquet.io.imas import read_imas_geometry
    ddp = os.path.join(work, "dd.json")
    with gzip.open(_DD, "rb") as fi, open(ddp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    b = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, ida_time=_IDA_TIME, n_draws=1, ida_path=_IDA, impurity_Z=6.0,
                             header=os.path.join(work, "run"), solve_method=method)
    b.config.generation.seed = 42
    src_a = b.config.source
    src_a.LCFS_geqdsk = _GEQ
    src_b = bq.ImasSource(**{**vars(src_a), "LCFS_geqdsk": None})
    f0_b = 1.1 * read_imas_geometry(src_b)[0]
    b.setup_solver()
    for i, lab in enumerate(labels):
        b.set_case(source=src_a if lab == "A" else src_b, header=os.path.join(work, f"{lab}{i}"),
                   scan_key=lab)
        b.config.solver.F0 = None if lab == "A" else f0_b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.prepare_baseline()
            psi = np.asarray(b.mygs.get_psi(normalized=False), float)
            coils = np.asarray(list(b.mygs.get_coil_currents()[0].values()), float)
            f = np.asarray(b.mygs.get_profiles(npsi=5)[1], float)
            f_edge = f[np.argmin(np.abs(f - b._geom.F0))]     # F at the boundary end
            b.generate()
        with h5py.File(f"{b.output_header}.h5", "r") as hf:
            draw = hf[f"scan/{lab}/0"]
            arrays = {f"draw_{k}": np.asarray(draw[k]) for k in draw if isinstance(draw[k], h5py.Dataset)
                      and draw[k].dtype.kind == "f"}
        np.savez(os.path.join(work, f"{lab}{i}.npz"), psi=psi, coils=coils, f_edge=f_edge,
                 F0=b._geom.F0, **arrays)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    env = _harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    jobs = {(m, seq): str(tmp_path_factory.mktemp(f"reuse_{m}_{seq}"))
            for m in _METHODS for seq in ("ABA", "A", "B")}
    procs = {k: subprocess.Popen([sys.executable, os.path.abspath(__file__), w, k[0], k[1]], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k, w in jobs.items()}
    errs = {}
    for k, pr in procs.items():
        _, err = pr.communicate()
        if pr.returncode:
            errs[k] = err[-4000:]
    return jobs, errs


def _load(jobs, errs, key, name):
    if key in errs:
        pytest.fail(f"{key} failed:\n{errs[key]}")
    return dict(np.load(os.path.join(jobs[key], f"{name}.npz")))


def _same(x, y):
    assert x.keys() == y.keys()
    for k in x:
        np.testing.assert_array_equal(x[k], y[k], err_msg=k)


@pytest.mark.parametrize("method", _METHODS)
def test_a_reused_solver_matches_a_fresh_one(runs, method):
    jobs, errs = runs
    fresh_a, fresh_b = _load(jobs, errs, (method, "A"), "A0"), _load(jobs, errs, (method, "B"), "B0")
    reuse = [_load(jobs, errs, (method, "ABA"), n) for n in ("A0", "B1", "A2")]
    assert fresh_b["F0"] == pytest.approx(1.1 * fresh_a["F0"])
    assert fresh_b["f_edge"] == pytest.approx(fresh_b["F0"], rel=1e-3)      # F0 reached the solver
    assert not np.array_equal(fresh_a["psi"], fresh_b["psi"])               # the cases differ
    _same(reuse[0], fresh_a)
    _same(reuse[1], fresh_b)
    _same(reuse[2], fresh_a)


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    _run_cases(sys.argv[1], sys.argv[2], list(sys.argv[3]))
