"""Live-solver test of the DEFAULT core treatment (``pytest -m solver``): the resistive q floor of
``current_model.CurrentModelSettings()`` and the default fit ``fit.FitSettings()`` on the public synthetic fixture.

Fixture as ``test_fit_solver.py`` (its truth: the sawtoothing L-mode slice made with the opt-in periodic orbit, and
its noisy synthetic measurements).  The default model is NOT the truth model, so nothing is asserted about the
parameters: (a) the relaxed consistent solve under the default settings (q floor 1, resistive form, the default pass
budget and acceleration) reaches ``"consistent"``, the floor acts on this case (q0 below 1 in the plain steady state),
its q check passes, and Ip is the target; (b) one default fit stops by a stopping test.  No setting, tolerance or pass
budget is changed for this test.

``OFT_env`` is a per-process singleton, so the live work runs in one child interpreter that writes JSON.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_fit_solver as B  # noqa: E402

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(B._OFT is None or not all(os.path.isfile(p) for p in (B._MESH, B._GEQ, B._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def child(out):
    import math
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.current import sawtooth as SAW
    rec = {}
    fm, kin, src, cfg_orb, init, ip = B._setup()
    tr = CM.consistent_solve(fm, kin, src, dict(log_strength=float(math.log(SAW.crash_strength(B._F_TRUE, B._RATE_TRUE))),
                                                f=B._F_TRUE), cfg_orb, label="truth", initial_profiles=init)
    rec["truth"] = dict(status=tr.status)
    if not tr.converged:
        json.dump(B._jsonable(rec), open(out, "w"))
        return
    snap = fm.snapshot()
    gs = fm.gs
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    noisy = B._measurements(gs, B._sensors(gs), roles, ip, np.random.default_rng(B._SEED))
    fm.reconfigure(noisy, np.asarray(gs.trace_surf(0.999), float), roles, case_id="noisy")
    cfg = CM.CurrentModelSettings(r0=cfg_orb.r0)
    rec["cfg"] = dict(core_relaxation=cfg.core_relaxation, q_floor=cfg.q_floor, q_floor_form=cfg.q_floor_form,
                      max_passes=cfg.max_passes, pass_tol=cfg.pass_tol, pass_acceleration=cfg.pass_acceleration)
    r0 = CM.consistent_solve(fm, kin, src, {}, cfg, start=snap, label="default")
    rec["relaxed"] = dict(status=r0.status, passes=r0.passes, reason=r0.record.get("reason"),
                          core=(r0.assembly or {}).get("core"), q_check=r0.record.get("q_check"),
                          ip=r0.solve.stats.get("Ip") if r0.converged else None, ip_target=float(fm.ip_measured),
                          dj=[p.get("dj") for p in r0.record.get("passes", [])])
    fm.restore(snap)
    fr = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(), start=snap, label="default_fit")
    rec["fit"] = dict(status=fr.status, stopping_summary=fr.record.get("stopping_summary"),
                      core=fr.record.get("core"), core_vs_floor=fr.record.get("core_vs_floor"),
                      params=fr.params, n_solves=fr.record.get("n_solves"))
    with open(out, "w") as fh:
        json.dump(B._jsonable(rec), fh, indent=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_default_core")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (B._REPO, B._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=5400)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_default_relaxed_solve_is_consistent(result):
    assert result["truth"]["status"] == "consistent", result["truth"]
    assert result["cfg"] == dict(core_relaxation="q_floor", q_floor=1.0, q_floor_form="resistive", max_passes=15,
                                 pass_tol=1e-5, pass_acceleration="anderson")
    r = result["relaxed"]
    assert r["status"] == "consistent", r
    core = r["core"]
    assert core["core_relaxation"] == "q_floor" and core["q_floor_form"] == "resistive"
    assert core["acted"] and core["q_min_before"] < 1.0 and core["n_cells_flattened"] >= 1
    assert r["q_check"]["passed"] is True
    assert r["ip"] == pytest.approx(r["ip_target"], rel=1e-3)


def test_default_fit_stops_by_a_stopping_test(result):
    f = result["fit"]
    assert f["status"] in ("converged", "converged_within_noise"), f["status"]
    assert f["core"]["core_relaxation"] == "q_floor" and "q_axis_model_minus_floor" in f["core_vs_floor"]
    assert f["params"]["f"] is None and f["params"]["log_strength"] is None


if __name__ == "__main__":
    child(sys.argv[1])
