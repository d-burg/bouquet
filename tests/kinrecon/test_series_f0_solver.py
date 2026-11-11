"""Live-solver test (``pytest -m solver``): a series slice is solved with its own vacuum ``F0``.

Public synthetic fixture (as ``test_series_solver.py``), sawtooth model off, generation mode (no
measurements).  Two slices 20 ms apart; the second slice's ``F0 = R0 B0`` is 0.5 % larger.  With
the per-slice value, ``F`` at the last closed surface of the second slice's equilibrium is that
slice's ``|F0|``, so the interval's recorded ``fpol_lcfs_change`` equals ``|F0_1| - |F0_0|``
(bound: within 2 % of that 0.5 % change, set before the first run); without a per-slice value the
first slice's ``F0`` is kept and the change is exactly zero.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_fit_solver as _base  # noqa: E402

DF = 0.005

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def child(out):
    import dataclasses
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import series as SE
    fm, kin, src, cfg, init, ip = _base._setup()
    cfg = dataclasses.replace(cfg, sawtooth=False)
    F00 = float(fm.F0)
    rec = dict(F0_0=F00)
    tr0 = CM.consistent_solve(fm, kin, src, dict(log_strength=None), cfg, label="f0truth0", initial_profiles=init)
    rec["truth0"] = tr0.status
    if not tr0.converged:
        json.dump(_base._jsonable(rec), open(out, "w"))
        return
    snap0 = fm.snapshot()
    bnd = np.asarray(fm.reference_boundary, float)

    def run(F0s, tag):
        fm.restore(snap0)
        fm.set_F0(F00)
        sl = [SE.SliceData(time=t, measurements=None, reference_boundary=bnd, kinetics=kin, sources=src,
                           label=f"{tag}{k}", meta=dict(ip=ip, generation_coil_reg=1e-2), F0=F0s[k])
              for k, t in enumerate((0.0, 0.02))]
        r = SE.run_series(fm, sl, cfg, SE.SeriesSettings(initial="relaxed", sawtooth="off"))
        return dict(status=r["status"], slices=[dict(status=x["status"], interval=x.get("interval"),
                                                     vacuum_F0=x.get("vacuum_F0")) for x in r["slices"]],
                    F0_after=float(fm.F0))

    rec["per_slice"] = run([F00, F00 * (1.0 + DF)], "f0var")
    rec["frozen"] = run([None, None], "f0frz")
    with open(out, "w") as fh:
        json.dump(_base._jsonable(rec), fh, indent=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_series_f0")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_each_slice_uses_its_own_F0(result):
    assert result["truth0"] == "consistent", result
    r = result["per_slice"]
    assert r["status"] == "completed", r
    F00 = abs(result["F0_0"])
    expected = F00 * DF
    s1 = r["slices"][1]
    assert s1["vacuum_F0"]["F0"] == pytest.approx(result["F0_0"] * (1.0 + DF), rel=1e-15)
    change = s1["interval"]["fpol_lcfs_change"]
    assert abs(change - expected) <= 0.02 * expected, (change, expected)


def test_without_per_slice_F0_the_first_value_is_kept(result):
    r = result["frozen"]
    assert r["status"] == "completed", r
    assert r["slices"][1]["vacuum_F0"]["source"].startswith("not given")
    assert r["slices"][1]["interval"]["fpol_lcfs_change"] == 0.0


if __name__ == "__main__":
    child(sys.argv[1])
