"""Live-solver test of the NON-DEFAULT robustness options (``pytest -m solver``) on the public synthetic fixture.

Fixture as ``test_default_core_live.py`` (the truth of ``test_fit_solver.py`` and its noisy synthetic measurements).
From the same starting state, the default consistent solve (q floor, default budget and acceleration) and the same
solve with the EXPERIMENTAL ``pass_relaxation=0.5`` both reach ``"consistent"`` by the unchanged criterion, and
they reach the same state to within the loop's own state tests (axis within ``state_axis_m``, l_i, q0 and Ip within
``state_rel``); the option is recorded only where set, with its factor-of-two note.  One fit with
``pass_relaxation=0.5`` and the fit options ``noise_retries=1`` and ``stall_diagnostics=True`` stops by a stopping
test and records the options.  No setting, tolerance or budget of the package is changed by this test.

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

ARMS = (("default", {}), ("relaxation", dict(pass_relaxation=0.5)))


def child(out):
    import dataclasses
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
    base = CM.CurrentModelSettings(r0=cfg_orb.r0)
    rec["criteria"] = dict(state_axis_m=base.state_axis_m, state_rel=base.state_rel, pass_tol=base.pass_tol,
                           max_passes=base.max_passes)
    pc0 = fm.probe_correction.copy()
    for name, kw in ARMS:
        fm.restore(snap)
        fm.probe_correction = pc0.copy()
        cfg = dataclasses.replace(base, **kw)
        r = CM.consistent_solve(fm, kin, src, {}, cfg, start=snap, label=f"loop_{name}")
        rec[name] = dict(status=r.status, passes=r.passes, robustness=r.record.get("robustness"),
                         relaxation=[p.get("relaxation") for p in r.record.get("passes", [])],
                         residual_cos=[p.get("residual_cos") for p in r.record.get("passes", [])],
                         stats=({k: r.solve.stats.get(k) for k in ("R_axis", "Z_axis", "l_i", "q_0", "Ip")}
                                if r.converged else None))
    fm.restore(snap)
    fm.probe_correction = pc0.copy()
    fs = FIT.FitSettings(noise_retries=1, stall_diagnostics=True)
    fr = FIT.reconstruct(fm, kin, src, dataclasses.replace(base, pass_relaxation=0.5), fs, start=snap,
                         label="options_fit")
    rec["fit"] = dict(status=fr.status, stopping_summary=fr.record.get("stopping_summary"),
                      robustness=fr.record.get("robustness"), settings=fr.record.get("settings"),
                      model_settings=fr.record.get("model_settings"))
    with open(out, "w") as fh:
        json.dump(B._jsonable(rec), fh, indent=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_robustness")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (B._REPO, B._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=5400)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_loop_options_reach_the_default_state(result):
    assert result["truth"]["status"] == "consistent", result["truth"]
    crit = result["criteria"]
    assert crit == dict(state_axis_m=5.0e-4, state_rel=1.0e-3, pass_tol=1.0e-5, max_passes=15)
    d = result["default"]
    assert d["status"] == "consistent", d
    assert d["robustness"] is None and not any(d["relaxation"])
    for name in ("relaxation",):
        r = result[name]
        assert r["status"] == "consistent", (name, r)
        assert r["robustness"] is not None
        ax = float(np.hypot(r["stats"]["R_axis"] - d["stats"]["R_axis"], r["stats"]["Z_axis"] - d["stats"]["Z_axis"]))
        assert ax < crit["state_axis_m"], (name, ax)
        for k in ("l_i", "q_0", "Ip"):
            assert abs(r["stats"][k] - d["stats"][k]) / abs(d["stats"][k]) < crit["state_rel"], (name, k)
    assert result["relaxation"]["robustness"]["pass_relaxation"] == 0.5
    assert all(v == 0.5 for v in result["relaxation"]["relaxation"][1:-1])
    assert "factor-of-two" in result["relaxation"]["robustness"]["pass_relaxation_note"]


def test_fit_with_options_stops_by_a_stopping_test(result):
    f = result["fit"]
    assert f["status"] in ("converged", "converged_within_noise"), f["status"]
    assert f["stopping_summary"]["test"] in ("a", "b", "c")
    rb = f["robustness"]
    assert rb["noise_retries"] == 1 and rb["stall_diagnostics"] is True and rb["pass_relaxation"] == 0.5
    assert f["settings"]["noise_retries"] == 1 and f["model_settings"]["pass_relaxation"] == 0.5


if __name__ == "__main__":
    child(sys.argv[1])
