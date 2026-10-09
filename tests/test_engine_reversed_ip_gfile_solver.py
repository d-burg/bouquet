"""The unified engine on a reversed-Ip / reversed-B_t g-file -- LIVE SOLVER
half (``pytest -m solver``; NOT run in the fast suite or in CI).

STATUS: written 2026-10-04, first run 2026-10-05 (7 passed); the bar for the
Ip-reversed mirrors (``_IP_MIRROR_REL``) is stated below with its measured
provenance (owner-confirmed 2026-10-05).

The synthetic example g-file is mirrored into all four (Ip, B_t)
orientations in its own COCOS (``test_engine_reversed_ip_gfile.mirror_raw``)
and each mirror is reconstructed by ``prepare_baseline`` with
``reconstruction_engine="unified"`` in its own interpreter (``OFT_env`` is a
per-process singleton), single-threaded:

* every orientation converges and delivers every row
  (``Baseline.engine["delivered"]["ok"]``);
* the B_t mirror hands the solver IDENTICAL inputs (the adapter reads it bit
  for bit, the solver's ``F0`` is ``|R B_t|``), so its delivered state must
  equal the normal file's BIT FOR BIT;
* the Ip mirrors hand it inputs that differ at the reader's contour-tracing
  level (~5e-6 relative, the fast half pins that), so they are compared at a
  stated relative bar;
* the engine record stamps the file's own orientation (``current_sign``,
  ``b0_sign``) and the delivered currents are positive (the positive frame;
  delivering in the source orientation is a separate pending change).

Synthetic inputs only.
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
sys.path.insert(0, _HERE)
import test_engine_reversed_ip_gfile as F  # noqa: E402

#: Stated bar for the Ip-reversed mirrors (owner-confirmed 2026-10-05):
#: their INPUTS differ by <= 1e-5 relative (contour tracing of the negated
#: psi_RZ); the delivered arrays are allowed 10x that.
#: Provenance -- measured on the first run (2026-10-05, bouquet ee69a60, the
#: fixed OpenFUSIONToolkit build at fork commit 7da4f18, one thread), as
#: max|delta| / max|reference| per array against the unmirrored file, worst
#: over the two Ip mirrors (Ip and Ip+B_t, identical): j_BS 5.34e-5,
#: j_phi 1.97e-5, j_inductive 7.35e-6, j_NBI / j_RF 0; l_i 1.3e-7 relative.
#: The B_t mirror is bitwise (asserted separately).  Margin on the worst
#: array (j_BS): 1e-4 / 5.34e-5 = 1.9x.
_IP_MIRROR_REL = 1e-4

_ARRAYS = ("j_phi", "j_inductive", "j_BS", "j_NBI", "j_RF")


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
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


_files_ok = all(os.path.isfile(p) for p in (F._GEQ, F._PF, F._MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")


def child(geqdsk, out):
    """One mirror, one interpreter: the engine baseline, written as JSON."""
    import bouquet as bq
    from bouquet.jbs_loop import jsonable
    _harness.assert_bouquet_is_repo_local()
    rec = dict(geqdsk=os.path.basename(geqdsk))
    try:
        b = bq.Bouquet.from_geqdsk(geqdsk, profiles=F._PF, mesh=F._MESH,
                                   nthreads=1, n_draws=1,
                                   header=os.path.splitext(out)[0],
                                   reconstruction_engine="unified")
        b.setup_solver()
        bl = b.prepare_baseline()
        e = bl.engine
        rec.update(converged=bool(e["converged"]),
                   delivered_ok=bool(e["delivered"]["ok"]),
                   misses=e["delivered"]["misses"],
                   l_i=float(e["delivered"]["checks"]["l_i"]["delivered"]),
                   signs=e["contract"]["signs"],
                   Ip_target=float(bl.Ip_target),
                   l_i_target=float(bl.l_i_target),
                   arrays={k: np.asarray(getattr(bl, k), float).tolist()
                           for k in _ARRAYS if getattr(bl, k) is not None})
    except Exception as exc:                       # recorded, never lost
        import traceback
        rec["error"] = f"{type(exc).__name__}: {exc}\n" \
            + traceback.format_exc()[-3000:]
    with open(out, "w") as fh:
        json.dump(jsonable(rec), fh)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    work = tmp_path_factory.mktemp("engine_mirrors")
    paths = F.write_mirrors(work)
    procs, outs = {}, {}
    for o, p in paths.items():
        outs[o] = str(work / f"out_ip{o[0]:+d}_bt{o[1]:+d}.json")
        procs[o] = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--child", p,
             outs[o]],
            env=_harness.subprocess_env(OMP_NUM_THREADS="1",
                                        MKL_NUM_THREADS="1",
                                        OPENBLAS_NUM_THREADS="1",
                                        MPLBACKEND="Agg"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    res = {}
    for o, pr in procs.items():
        log, _ = pr.communicate()
        with open(outs[o] + ".log", "w") as fh:
            fh.write(log)
        with open(outs[o]) as fh:
            res[o] = json.load(fh)
    return res


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("o", F.ORIENTATIONS, ids=F._IDS)
def test_every_orientation_converges_and_delivers(solved, o):
    r = solved[o]
    assert "error" not in r, r.get("error")
    assert r["converged"] and r["delivered_ok"], r["misses"]
    assert float(np.trapezoid(r["arrays"]["j_phi"])) > 0.0  # positive frame


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("o", F.ORIENTATIONS[1:], ids=F._IDS[1:])
def test_the_mirror_delivers_the_normal_files_state(solved, o):
    r, r0 = solved[o], solved[(1, 1)]
    assert "error" not in r and "error" not in r0
    assert r["signs"]["current_sign"] == o[0] * r0["signs"]["current_sign"]
    assert r["signs"]["b0_sign"] == o[1] * r0["signs"]["b0_sign"]
    assert r["Ip_target"] == r0["Ip_target"]
    for k in r0["arrays"]:
        a, b = np.asarray(r["arrays"][k]), np.asarray(r0["arrays"][k])
        if o[0] > 0:
            np.testing.assert_array_equal(a, b, err_msg=k)
        else:
            rel = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) or 1.0))
            assert rel <= _IP_MIRROR_REL, (k, rel)
    if o[0] > 0:
        assert r["l_i"] == r0["l_i"]
    else:
        assert abs(r["l_i"] - r0["l_i"]) <= _IP_MIRROR_REL * r0["l_i"]


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1], sys.argv[i + 2])
