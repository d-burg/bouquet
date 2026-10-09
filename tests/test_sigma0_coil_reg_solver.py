"""The engine sigma=0 check with a CONFIGURED coil regularisation -- LIVE
SOLVER half (``pytest -m solver``; NOT run in the fast suite or in CI).

STATUS: written 2026-10-06; not yet run (to be run on the cluster).

The fast half (``tests/test_sigma0_coil_reg_and_gates.py``) pins, on a toy
stand-in, that every engine draw's loop installs the reconstruction's own
coil-regularisation term list.  Here the synthetic g-file example is
reconstructed on the live solver with measured-coil targets configured the
way the repository's coil-target tests build them
(:func:`bouquet.coil_targets.coil_reg_from_measured` with
``device="DIII-D"``: each coil pinned at a measured current at the default
reference weight ``W0`` = 100), the measured currents being the coil
currents of the same example's reconstruction without targets (so the
targets are physical and self-consistent).  The configured reconstruction
then runs (after ``Bouquet._reset_solver_state``, which installs the
configured terms, as a slice sweep does) and ``verify_sigma0_consistency``
must pass its widened gate: the request bit-identical, the loop converged,
``r_j`` / ``r_I`` / ``|dl_i|`` / ``|dq0|`` at the loop tolerances on both
stages, and the archived coil drift and LCFS deviation within the hard coil
bound and the in-spec boundary cut -- with the draw solving under the
reconstruction's configured term list (``coil_reg.source``).

One interpreter (``OFT_env`` is a per-process singleton), single-threaded.
Synthetic inputs only.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest  # noqa: E402

_EX = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")


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


_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")


def child(out):
    """Reconstruct without, then with, measured-coil targets; the sigma=0
    check of the configured one.  Everything written as JSON."""
    import bouquet as bq
    from bouquet.coil_targets import TURNFC_D3D, coil_reg_from_measured
    from bouquet.jbs_loop import jsonable
    _harness.assert_bouquet_is_repo_local()
    rec = {}
    try:
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                   nthreads=1, n_draws=1,
                                   header=os.path.splitext(out)[0],
                                   reconstruction_engine="unified")
        b.setup_solver()
        b.prepare_baseline()
        rec["unconfigured_source"] = b._engine_run["coil_reg"]["record"][
            "source"]
        cur, _ = b.mygs.get_coil_currents()
        # the circuit currents a pf_active read would hand over (solver
        # units / the device's turns), pinned at W0 = 100
        meas = {str(n): float(v) / float(TURNFC_D3D.get(n, 1.0))
                for n, v in cur.items()}
        spec = coil_reg_from_measured(meas, device="DIII-D")
        rec["n_terms_configured"] = len(spec)
        b.config.solver.coil_reg = spec
        b._reset_solver_state()
        bl = b.prepare_baseline()
        rec["configured"] = dict(
            converged=bool(bl.engine["converged"]),
            delivered_ok=bool(bl.engine["delivered"]["ok"]),
            coil_reg=b._engine_run["coil_reg"]["record"])
        v = b.verify_sigma0_consistency()
        rec["sigma0"] = dict(
            passed=bool(v["passed"]), gates=v.get("gates"),
            coil_reg=v.get("coil_reg"), rejection=v.get("rejection"),
            request_bit_identical=v.get("request_bit_identical"),
            loop_converged=v.get("loop_converged"),
            archived_geometry=v["stages"].get("archived_geometry"))
    except Exception as exc:                       # recorded, never lost
        import traceback
        rec["error"] = f"{type(exc).__name__}: {exc}\n" \
            + traceback.format_exc()[-3000:]
    with open(out, "w") as fh:
        json.dump(jsonable(rec), fh)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    work = tmp_path_factory.mktemp("sigma0_coil_reg")
    out = str(work / "sigma0_coil_reg.json")
    pr = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", out],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out + ".log", "w") as fh:
        fh.write(pr.stdout)
    with open(out) as fh:
        return json.load(fh)


@pytest.mark.solver
@solver_only
def test_the_configured_reconstruction_records_its_terms(solved):
    assert "error" not in solved, solved.get("error")
    assert solved["unconfigured_source"] == "default"
    c = solved["configured"]
    assert c["converged"] and c["delivered_ok"]
    assert c["coil_reg"]["source"] == "configured"
    assert any(t["weight"] == 100.0 for t in c["coil_reg"]["terms"])


@pytest.mark.solver
@solver_only
def test_the_sigma0_draw_passes_the_widened_gate_under_coil_reg(solved):
    assert "error" not in solved, solved.get("error")
    s = solved["sigma0"]
    assert s["rejection"] is None, s["rejection"]
    assert s["coil_reg"]["source"] == "reconstruction (configured)"
    assert s["coil_reg"]["installed"] is True
    assert s["request_bit_identical"] and s["loop_converged"]
    for stage in ("loop", "archived", "archived_geometry"):
        for name, g in s["gates"][stage].items():
            assert g["passed"] is not False, (stage, name, g)
    assert s["passed"] is True


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1])
