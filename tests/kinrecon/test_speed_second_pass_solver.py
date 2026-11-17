"""Live-solver tests (``pytest -m solver``) of the second robustness pass of the speed settings.  Public synthetic
fixture only (as ``test_fit_solver.py``); each run is a separate process.

Checked, every saved value compared bit for bit with the serial fit (wall times, the Jacobians' ``parallel`` records
and the fit's ``speed`` record excepted):
(a) the in-process options without workers (memo, indexed point location, reused statistics) give the serial fit,
    and the fit record holds the speed settings (``record["speed"]``), which the default record does not;
(b) with a directory name planted beside the solver's temporary directory that names a process id no process can
    have, and with the second worker of every Jacobian failing with an error that is not an ``OSError``, the parallel
    fit falls back to serial columns, ends equal to the serial fit and records the failure.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_fit_solver as _base  # noqa: E402
from test_speed_parallel_solver import _parallel_records, differences  # noqa: E402

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parallel columns are Linux only"),
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def _fixture_fit(mode):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon import speed as SP
    from bouquet.kinrecon.current import sawtooth as SAW
    fm, kin, src, cfg, init, ip = _base._setup()
    ls_true = float(np.log(SAW.crash_strength(_base._F_TRUE, _base._RATE_TRUE)))
    tr = CM.consistent_solve(fm, kin, src, dict(log_strength=ls_true, f=_base._F_TRUE), cfg, label="truth",
                             initial_profiles=init)
    assert tr.converged, tr.status
    snap = fm.snapshot()
    gs = fm.gs
    geo = _base._sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    ms = _base._measurements(gs, geo, roles, ip, np.random.default_rng(_base._SEED))
    fm.reconfigure(ms, np.asarray(gs.trace_surf(0.999), float), roles, case_id="fit")
    speed = None
    planted = None
    if mode == "inproc":
        speed = SP.SpeedSettings(memo=True, mesh_index=True, reuse_stats=True)
    elif mode == "setupfail":
        base = os.path.dirname(fm.env.tempdir)
        planted = os.path.join(base, f"oft_worker_{SP._host_tag()}_99999999999999999999_x")
        os.mkdir(planted)
        real_fork, calls = os.fork, [0]

        def fork_second_fails():
            calls[0] += 1
            if calls[0] % 2 == 0:
                raise RuntimeError("worker set-up failed for the test (not an OSError)")
            return real_fork()

        os.fork = fork_second_fails
        speed = SP.SpeedSettings.all_on(3)
    fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(), start=snap,
                         label="fx", speed=speed)
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    out = dict(mode=mode, status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance, chi2=fr.chi2,
               params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves,
               history=fm.history, case_history=fm.case.history)
    if planted is not None:
        out["planted_kept"] = os.path.isdir(planted)
        os.rmdir(planted)
    return out


def child(mode, out):
    with open(out, "w") as fh:
        json.dump(_base._jsonable(_fixture_fit(mode)), fh)


def _run(tmp_path_factory, mode):
    d = tmp_path_factory.mktemp(f"speed2_{mode}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               TMPDIR=str(d))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), mode, out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    return {m: _run(tmp_path_factory, m) for m in ("serial", "inproc", "setupfail")}


def _without(r, *keys):
    out = dict(r)
    out["record"] = {k: v for k, v in r["record"].items() if k not in keys}
    return {k: v for k, v in out.items() if k not in ("planted_kept",)}


def test_in_process_options_are_recorded_and_change_nothing(fits):
    from bouquet.kinrecon.speed import SpeedSettings
    s, p = fits["serial"], fits["inproc"]
    assert "speed" not in s["record"]
    assert p["record"]["speed"] == dict(settings=SpeedSettings(memo=True, mesh_index=True,
                                                               reuse_stats=True).as_dict())
    assert not _parallel_records(p["record"])
    assert differences(s, _without(p, "speed")) == []


def test_any_worker_setup_error_falls_back_to_serial(fits):
    s, p = fits["serial"], fits["setupfail"]
    assert p["planted_kept"] is True
    assert "speed" not in p["record"]                       # with workers the parallel records hold the settings
    assert differences(s, _without(p)) == []
    par = _parallel_records(p["record"])
    assert par and all("RuntimeError" in (r["worker_start_failure"] or "") for r in par)
    assert all(r["columns_here"] == [0, 1, 2, 3] and r["columns_from_workers"] == [] for r in par)


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2])
