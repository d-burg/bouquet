"""Live-solver tests (``pytest -m solver``) of the speed defaults.  Public synthetic fixture only (as
``test_fit_solver.py``); each run is a separate process.

Checked:
(a) the default fit (no ``speed`` argument: the in-process options on) equals the fit with every option off
    (``SpeedSettings.all_off()``) bit for bit, apart from wall times and the records of the settings;
(b) both records hold the options in effect (``speed_in_effect``); only the fit given settings holds ``speed``;
(c) a fit with column workers whose first center fails (injected) ends before any Jacobian and still records the
    options in effect, workers included.
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

WORKERS = 3


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
    kw = {}
    if mode == "off":
        kw["speed"] = SP.SpeedSettings.all_off()
    elif mode == "nojac":
        kw["speed"] = SP.SpeedSettings.all_on(WORKERS)
        real = CM.consistent_solve

        def failing_center(*a, **k):
            if str(k.get("label", "")).endswith(":center0"):
                return CM.ConsistentResult(False, "solve_failed", 0, None, None, None,
                                           dict(passes=[], reason="injected failure of the first center"), 0.0)
            return real(*a, **k)

        CM.consistent_solve = failing_center
    fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(anchor_follow=False), start=snap,
                         label="fx", **kw)
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    return dict(mode=mode, status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance, chi2=fr.chi2,
                params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves,
                history=fm.history, case_history=fm.case.history)


def child(mode, out):
    with open(out, "w") as fh:
        json.dump(_base._jsonable(_fixture_fit(mode)), fh)


def _run(tmp_path_factory, mode):
    d = tmp_path_factory.mktemp(f"speeddef_{mode}")
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
    return {m: _run(tmp_path_factory, m) for m in ("default", "off", "nojac")}


def _without(r, *keys):
    out = dict(r)
    out["record"] = {k: v for k, v in r["record"].items() if k not in keys}
    return {k: v for k, v in out.items() if k != "mode"}


def _effect(on, workers=1):
    return dict(memo=on, mesh_index=on, reuse_stats=on,
                workers=dict(column_workers=workers, column_time_factor=3.0, column_time_floor_s=60.0))


def test_default_fit_equals_the_fit_with_every_option_off(fits):
    d, o = fits["default"], fits["off"]
    assert d["status"] in ("converged", "converged_within_noise"), d["status"]
    assert differences(_without(d, "speed_in_effect"), _without(o, "speed_in_effect", "speed")) == []


def test_fit_records_hold_the_options_in_effect(fits):
    from bouquet.kinrecon.speed import SpeedSettings
    d, o = fits["default"]["record"], fits["off"]["record"]
    assert d["speed_in_effect"] == _effect(True) and "speed" not in d
    assert o["speed_in_effect"] == _effect(False) and o["speed"] == dict(settings=SpeedSettings.all_off().as_dict())


def test_parallel_fit_that_forms_no_jacobian_records_the_options(fits):
    n = fits["nojac"]
    assert n["status"] == "failed_initial_consistency", n["status"]
    assert not _parallel_records(n["record"]) and "speed" not in n["record"]
    assert n["record"]["speed_in_effect"] == _effect(True, WORKERS)


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2])
