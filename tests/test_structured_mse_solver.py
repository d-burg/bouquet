"""MSE field reads on a LIVE TokaMaker equilibrium (``pytest -m solver``).

The solver-free tests use an interpolator double that reproduces the real
one's failure mode (an off-mesh point returns the previous point's buffer).
These check the same guards against the real interpolator:

* an off-mesh chord placed BETWEEN two on-mesh chords is reported
  ``found=False`` with a NaN row -- not the previous chord's field -- and the
  on-mesh chords read exactly what a fresh single-point read gives;
* the equilibrium's own current/field directions are readable off its field
  about ``mygs.o_point``, with every outboard-midplane chord agreeing.

Every solver call runs in a subprocess (``OFT_env`` is a per-process
singleton; see ``tests/_harness.py``).  Synthetic D3D-like example only.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


def _probe(outdir):
    import bouquet as bq
    from bouquet.mse import mse_equilibrium_orientation, mse_field_at

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "mse"), n_draws=1,
                               reconstruction_engine="legacy")
    b.setup_solver()
    b.prepare_baseline()
    mygs = b.mygs
    Ra, Za = (float(v) for v in mygs.o_point)
    # outboard-midplane chords, with one far off the mesh in the middle
    R_on = Ra + np.array([0.15, 0.30, 0.45])
    R = np.array([R_on[0], 10.0, R_on[1], R_on[2]])
    Z = np.full(R.size, Za)
    B, found = mse_field_at(mygs, R, Z)
    single = [mse_field_at(mygs, [r], [Za])[0][0] for r in R_on]
    o = mse_equilibrium_orientation(B[found], R[found], Z[found], (Ra, Za))
    out = dict(found=[bool(f) for f in found],
               B=[[None if not np.isfinite(x) else float(x) for x in row]
                  for row in B],
               single=[[float(x) for x in row] for row in single],
               orientation=dict(ip=o["ip"], bt=o["bt"],
                                n_ip_agree=o["n_ip_agree"],
                                n_ip_used=o["n_ip_used"]),
               Ip_globals=float(mygs.get_globals()[0]),
               axis=[Ra, Za])
    with open(os.path.join(outdir, "mse.json"), "w") as fh:
        json.dump(out, fh)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    work = tmp_path_factory.mktemp("mse")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"MSE probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    with open(str(work / "mse.json")) as fh:
        return json.load(fh)


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like mesh/baseline; skipped when unavailable")


@pytest.mark.solver
@solver_only
def test_off_mesh_chord_is_not_given_the_previous_chords_field(measured):
    assert measured["found"] == [True, False, True, True]
    assert measured["B"][1] == [None, None, None]
    on = [measured["B"][i] for i in (0, 2, 3)]
    np.testing.assert_allclose(np.asarray(on), np.asarray(
        measured["single"]), rtol=1e-12, atol=0.0)


@pytest.mark.solver
@solver_only
def test_the_equilibrium_orientation_is_readable(measured):
    o = measured["orientation"]
    assert o["ip"] in (1.0, -1.0) and o["bt"] in (1.0, -1.0)
    assert o["n_ip_agree"] == o["n_ip_used"] == 3


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1])
