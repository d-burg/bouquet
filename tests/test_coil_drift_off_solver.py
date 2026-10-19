"""Coil handling off (``GenerationConfig.coil_drift = None``), issue #95 --
the LIVE-SOLVER half (``pytest -m solver``).

The synthetic g-file + IDA-layout profile pair (the g-file golden's inputs)
is reconstructed on the unified engine and one seeded draw is generated with
coil handling off.  Before the fix every such run raised
``UnboundLocalError`` before archiving anything.  The draw must be archived,
with its geqdsk, and stamped as a draw whose coil drifts were not measured
(``homotopy_pass`` -1, NaN drifts, the legacy drift verdict
``coil_drift_in_spec`` False).  Its ``in_spec`` is the in-spec rule's verdict
under the CONFIGURED coil filter (chi2 by default: the draw's coil currents
against the measured per-coil sigma, which a coil-off draw has like any
other), so it is not asserted to be False: ``Bouquet.filter()`` must
reproduce it, and only under the legacy filter is it False by construction.
The fast half is ``tests/test_coil_drift_off.py``.

One interpreter (``OFT_env`` is a per-process singleton), one thread.
Synthetic inputs only.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest  # noqa: E402

_DATA = os.path.join(_harness.REPO_ROOT, "tests", "data")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_PROF = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_MESH = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like",
                     "DIIID_mesh.h5")


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


_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PROF, _MESH))
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not (_files_ok and _oft_importable()),
                       reason="needs OFT + the synthetic fixtures"),
]


def child(out):
    """Reconstruct, then generate one seeded draw with coil handling off;
    every number written as JSON, an error recorded in its place."""
    import h5py
    import numpy as np
    import bouquet as bq
    from bouquet.jbs_loop import jsonable
    _harness.assert_bouquet_is_repo_local()
    rec = {}
    try:
        header = os.path.splitext(out)[0]
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PROF, mesh=_MESH,
                                   nthreads=1, n_draws=1, header=header,
                                   reconstruction_engine="unified")
        gc = b.config.generation
        gc.seed = 12345
        gc.n_inspec_target = None
        gc.coil_drift = None
        b.setup_solver()
        b.prepare_baseline()
        diags = b.generate()
        rec["n_diags"] = len(diags)
        rec["rejections"] = [dict(reason=r.get("reason"),
                                  stage=r.get("stage"))
                             for r in (getattr(b, "draw_rejections", None)
                                       or [])]
        with h5py.File(header + ".h5", "r") as hf:
            grp = (hf["scan/" + sorted(hf["scan"].keys())[0]]
                   if "scan" in hf else hf)
            draws = sorted(k for k in grp.keys() if k.isdigit())
            rec["draws"] = draws
            rec["n_stored"] = int(grp.attrs.get("n_stored", -1))
            if draws:
                g = grp[draws[0]]
                a = g.attrs
                rec["has_eqdsk"] = "eqdsk" in g
                rec["has_coil_currents"] = "coil_currents" in g
                rec["stamps"] = dict(
                    homotopy_pass=int(a["homotopy_pass"]),
                    max_F_drift_pct=float(a["max_F_drift_pct"]),
                    max_VSC_drift_pct=float(a["max_VSC_drift_pct"]),
                    coil_drift_in_spec=bool(a["coil_drift_in_spec"]),
                    in_spec=bool(a["in_spec"]))
                rec["l_i"] = float(a.get("l_i(3)", np.nan))
        rec["coil_filter"] = str(diags[0].get("in_spec_coil_filter"))
        # the configured filter, re-applied by Bouquet.filter(): it must
        # reproduce the archived verdict
        from bouquet.filtering import read_filter_flags
        b.filter()
        flags = read_filter_flags(header + ".h5")     # {scan: {idx: {}}}
        f0 = next(iter(flags.values()))[int(rec["draws"][0])]
        rec["filter_selected"] = (None if "selected" not in f0
                                  else bool(f0["selected"]))
        rec["filter_in_spec"] = (None if "in_spec" not in f0
                                 else bool(f0["in_spec"]))
    except Exception as exc:                       # recorded, never lost
        import traceback
        rec["error"] = f"{type(exc).__name__}: {exc}\n" \
            + traceback.format_exc()[-3000:]
    with open(out, "w") as fh:
        json.dump(jsonable(rec), fh)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    work = tmp_path_factory.mktemp("coil_off")
    out = str(work / "coil_off.json")
    pr = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", out],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out + ".log", "w") as fh:
        fh.write(pr.stdout)
    assert os.path.isfile(out), pr.stdout[-4000:]
    with open(out) as fh:
        return json.load(fh)


def test_coil_off_archives_a_draw(solved):
    assert "error" not in solved, solved.get("error")
    assert solved["n_diags"] == 1, solved
    assert solved["draws"] == ["0"], solved
    assert solved["n_stored"] == 1
    assert solved["has_eqdsk"] and solved["has_coil_currents"]


def test_coil_off_draw_is_stamped_unmeasured(solved):
    assert "error" not in solved, solved.get("error")
    s = solved["stamps"]
    assert s["homotopy_pass"] == -1
    assert s["max_F_drift_pct"] != s["max_F_drift_pct"]       # NaN
    assert s["max_VSC_drift_pct"] != s["max_VSC_drift_pct"]   # NaN
    assert s["coil_drift_in_spec"] is False
    # in_spec: the configured coil filter's verdict, reproduced by filter()
    assert solved["coil_filter"] in ("chi2", "legacy"), solved
    assert solved["filter_in_spec"] is s["in_spec"], solved
    assert solved["filter_selected"] is s["in_spec"], solved
    if solved["coil_filter"] == "legacy":
        assert s["in_spec"] is False


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1])
