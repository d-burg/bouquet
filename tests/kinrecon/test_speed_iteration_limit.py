"""The parallel Jacobian columns carry the solver iteration limit and the per-solve iteration records exactly as the
serial loop does (``ForwardSettings.solver_maxits``, ``ForwardModel.iteration_record``).

No solver: a worker column's iteration-log entries travel with its side effects and are applied in column order.
Solver (``pytest -m solver``, public fixture, each run a separate process): a fit with ``solver_maxits = 80`` and
three column processes equals the serial fit with the same limit bit for bit, including the fit's
``solver_iterations`` record, whose entries cover every solve -- the workers' included."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import speed as SP

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
MAXITS = 80


def test_iteration_log_travels_with_a_worker_column():
    case = SimpleNamespace(locked=False, spec=None, decided_by="center", history=[])
    fm = SimpleNamespace(n_solves=2, history=[], case=case, taper_restarts=[],
                         iteration_log=[dict(label="center", its=7, kind="solve", stopped_at_limit=False)])
    mark = SP.effects_mark(fm, None, None)
    fm.n_solves += 2
    fm.iteration_log += [dict(label="col1", its=4, kind="solve", stopped_at_limit=False),
                         dict(label="col1:verify", its=1, kind="verify", stopped_at_limit=False)]
    eff = SP.effects_since(mark, fm, None, None)
    assert [e["label"] for e in eff["iteration_log"]] == ["col1", "col1:verify"]
    parent = SimpleNamespace(n_solves=2, history=[], case=SimpleNamespace(history=[]), taper_restarts=[],
                             iteration_log=[dict(label="center", its=7, kind="solve", stopped_at_limit=False)])
    SP.apply_effects(eff, parent, None, None)
    assert parent.iteration_log == fm.iteration_log and parent.n_solves == fm.n_solves
    # a stand-in model without the log (an older forward model) still works
    old = SimpleNamespace(n_solves=0, history=[], case=SimpleNamespace(locked=False, spec=None, decided_by=None,
                                                                         history=[]), taper_restarts=[])
    m = SP.effects_mark(old, None, None)
    e = SP.effects_since(m, old, None, None)
    assert e["iteration_log"] is None
    SP.apply_effects(e, old, None, None)
    assert not hasattr(old, "iteration_log")


# ---------------------------------------------------------------------------------------------------- solver
def _fit(workers, out):
    sys.path.insert(0, _HERE)
    import test_fit_solver as base
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.current import sawtooth as SAW
    fm, kin, src, cfg, init, ip = base._setup(solver_maxits=MAXITS)
    ls_true = float(np.log(SAW.crash_strength(base._F_TRUE, base._RATE_TRUE)))
    tr = CM.consistent_solve(fm, kin, src, dict(log_strength=ls_true, f=base._F_TRUE), cfg, label="truth",
                             initial_profiles=init)
    assert tr.converged, tr.status
    snap = fm.snapshot()
    gs = fm.gs
    geo = base._sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    ms = base._measurements(gs, geo, roles, ip, np.random.default_rng(base._SEED))
    fm.reconfigure(ms, np.asarray(gs.trace_surf(0.999), float), roles, case_id="fit")
    speed = SP.SpeedSettings(column_workers=workers) if workers > 1 else None
    fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(anchor_follow=False), start=snap,
                         label="fx", speed=speed)
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    res = dict(workers=workers, status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance,
               chi2=fr.chi2, params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves,
               history=fm.history, iteration_log=fm.iteration_log, limit_now=int(gs.settings.maxits))
    with open(out, "w") as fh:
        json.dump(base._jsonable(res), fh)


def _run(tmp_path_factory, workers):
    sys.path.insert(0, _HERE)
    import test_fit_solver as base
    d = tmp_path_factory.mktemp(f"speed_maxits_{workers}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               TMPDIR=str(d))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), str(workers), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def _solver_available():
    sys.path.insert(0, _HERE)
    try:
        import test_fit_solver as base
    except Exception:  # noqa: BLE001
        return False
    return base._OFT is not None and all(os.path.isfile(p) for p in (base._MESH, base._GEQ, base._IDA))


@pytest.mark.solver
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parallel columns are Linux only")
@pytest.mark.skipif(not _solver_available(), reason="needs OpenFUSIONToolkit and the public synthetic fixtures")
def test_parallel_fit_with_an_iteration_limit_is_bitwise_serial(tmp_path_factory):
    sys.path.insert(0, _HERE)
    from test_speed_parallel_solver import _parallel_records, differences
    s, p = _run(tmp_path_factory, 1), _run(tmp_path_factory, 3)
    assert differences(s, p) == []
    its = p["record"]["solver_iterations"]
    assert its["limit"]["setting"] == MAXITS and its["limit"]["in_effect"] == MAXITS and p["limit_now"] == MAXITS
    assert its["n_solves"] == p["record"]["n_solves"] == len(its["its"])     # every solve, the workers' included
    par = _parallel_records(p["record"])
    assert par and any(r["columns_from_workers"] for r in par)
    assert all(i is None or i <= MAXITS for i in its["its"])


if __name__ == "__main__":
    _fit(int(sys.argv[1]), sys.argv[2])
