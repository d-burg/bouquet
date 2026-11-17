"""Live-solver tests (``pytest -m solver``) of the parallel Jacobian columns (``bouquet.kinrecon.speed``).

Public synthetic fixture only (as ``test_fit_solver.py``).  Each run is a separate process; the parallel runs use
three column processes in contiguous blocks (this one with column 0 and the noise re-solve, one worker with column 1,
one with columns 2 and 3), so a worker holding several columns is exercised.

Checked, with every saved value compared bit for bit (wall times and the parallel run's own record excepted):
(a) a single-slice fit (default free set: three correction amplitudes and the Ip target, noisy data, default core):
    serial and parallel results are identical, including the iteration log, the noise, the solve count, the
    forward model's residual histories and the final psi;
(b) a column whose consistent solve FAILS (injected, in every process) ends both fits as ``jacobian_failed`` with
    identical records;
(c) a column whose solve RAISES (injected) raises the same exception from both fits;
(d) a worker process that DIES in a column (injected in workers only) leaves the parallel fit identical to the serial
    one: the column is re-run serially and the re-run is recorded;
(e) a short series with filter fits (three slices of the time-dependent path's synthetic truth, noisy data, the
    in-window corrections): serial and parallel series records are identical, including the record of the state
    left by each fit's last solve, which the parallel path takes from the worker that made it.
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

WORKERS = 3
_IGNORE = {"wall", "parallel"}

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork"),
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


# ---------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------
def _inject(mode, label_tail):
    """Wrap the consistent solve so that the column ``label_tail`` of the first Jacobian fails, raises or (in a
    worker only) kills its process."""
    from bouquet.kinrecon import current_model as CM
    real = CM.consistent_solve
    parent = os.getpid()

    def wrapped(*a, **kw):
        lab = kw.get("label", "")
        if lab.endswith(label_tail) or lab.endswith(label_tail + ":opp"):
            if mode == "fail":
                return CM.ConsistentResult(False, "solve_failed", 0, None, None, None,
                                           dict(passes=[], reason="injected column failure"), 0.0)
            if mode == "raise":
                raise RuntimeError("injected exception in a column solve")
            if mode == "crash" and os.getpid() != parent:
                os._exit(3)
        return real(*a, **kw)

    CM.consistent_solve = wrapped


def _fixture_fit(workers, mode):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.current import sawtooth as SAW
    from bouquet.kinrecon.speed import SpeedSettings
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
    if mode != "none":
        _inject(mode, "fx:it0:col2")
    speed = SpeedSettings(column_workers=workers) if workers > 1 else None
    out = dict(workers=workers, mode=mode)
    try:
        fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(), start=snap,
                             label="fx", speed=speed)
    except Exception as exc:  # noqa: BLE001 -- the exception is the result of mode "raise"
        out["exception"] = f"{type(exc).__name__}: {exc}"
        return out
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    out.update(status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance, chi2=fr.chi2,
               params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves,
               history=fm.history, case_history=fm.case.history)
    return out


def _series(workers):
    import dataclasses
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import series as SE
    from bouquet.kinrecon.speed import SpeedSettings
    import test_series_solver as TS
    fm, kin, src, cfg, init, ip = _base._setup()
    cfg = dataclasses.replace(cfg, core_relaxation="none")
    tr0 = CM.consistent_solve(fm, kin, src, dict(log_strength=None, a_corr=TS.A_TRUE), cfg, label="truth0",
                              initial_profiles=init)
    assert tr0.converged, tr0.status
    pr = tr0.assembly["profiles_rho"]
    j_true = (np.asarray(pr["rho"]), np.asarray(pr["jdotb"]))
    snap0 = fm.snapshot()
    geo = _base._sensors(fm.gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted"))
             for n in fm.gs.coil_sets}
    times = TS.TIMES[:3]
    bnd = np.asarray(fm.reference_boundary, float)
    gen = [SE.SliceData(time=t, measurements=None, reference_boundary=bnd, kinetics=kin, sources=src,
                        label=f"truth{k}", meta=dict(ip=ip * (1 + TS.RAMP * k), generation_coil_reg=1e-2))
           for k, t in enumerate(times)]
    meas, refs = [], []
    rng = np.random.default_rng(TS._SEED)

    def cb(r):
        meas.append(_base._measurements(fm.gs, geo, roles, ip * (1 + TS.RAMP * len(meas)), rng))
        refs.append(np.asarray(fm.gs.trace_surf(0.999), float))

    truth = SE.run_series(fm, gen, cfg, SE.SeriesSettings(initial="from_profile", initial_profile=j_true,
                                                          sawtooth="off"), callback=cb)
    assert truth["status"] == "completed", truth["status"]
    fm.restore(snap0)
    fm.reconfigure(meas[0], refs[0], roles, case_id="filtered")
    sl = [SE.SliceData(time=t, measurements=meas[k], reference_boundary=refs[k], kinetics=kin, sources=src,
                       label=f"f{k}") for k, t in enumerate(times)]
    st = SE.SeriesSettings(initial="from_profile", initial_profile=j_true, sawtooth="off", filter=True)
    r = SE.run_series(fm, sl, cfg, st, speed=SpeedSettings(column_workers=workers) if workers > 1 else None)
    return dict(workers=workers, series=r, psi=np.asarray(fm.gs.get_psi(False)), n_solves=fm.n_solves,
                history=fm.history)


def child(kind, workers, mode, out):
    res = _series(workers) if kind == "series" else _fixture_fit(workers, mode)
    with open(out, "w") as fh:
        json.dump(_base._jsonable(res), fh)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------
def _run(tmp_path_factory, kind, workers, mode="none"):
    d = tmp_path_factory.mktemp(f"speed_{kind}_{workers}_{mode}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), kind, str(workers), mode, out], cwd=str(d),
                          env=env, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def differences(a, b, path=""):
    """Every leaf that differs between two results (bitwise for numbers), ignoring wall times and the parallel
    run's own record."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in _IGNORE or k.endswith("wall_s") or k == "workers":
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in one result only")
            else:
                out += differences(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{i}]")
    elif not (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)):
        out.append(f"{path}: {str(a)[:60]} != {str(b)[:60]}")
    return out


def _parallel_records(rec):
    """Every ``parallel`` record of a fit's Jacobians."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "parallel" in o and isinstance(o["parallel"], dict):
                found.append(o["parallel"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rec)
    return found


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    return {(w, m): _run(tmp_path_factory, "fit", w, m)
            for w, m in [(1, "none"), (WORKERS, "none"), (1, "fail"), (WORKERS, "fail"), (1, "raise"),
                         (WORKERS, "raise"), (WORKERS, "crash")]}


def test_parallel_fit_is_bitwise_serial(fits):
    s, p = fits[(1, "none")], fits[(WORKERS, "none")]
    assert s["status"] in ("converged", "converged_within_noise"), s["status"]
    assert differences(s, p) == []
    par = _parallel_records(p["record"])
    assert par and all(r["workers"] == WORKERS - 1 for r in par)
    assert all(r["blocks"] == [[0], [1], [2, 3]] for r in par)
    assert all(sorted(r["columns_here"] + r["columns_from_workers"]) == [0, 1, 2, 3] for r in par)
    assert all(r["serial_reruns"] == [] for r in par)
    assert not _parallel_records(s["record"])


def test_failed_column_surfaces_as_serial(fits):
    s, p = fits[(1, "fail")], fits[(WORKERS, "fail")]
    assert s["status"] == "jacobian_failed"
    assert differences(s, p) == []


def test_raising_column_surfaces_as_serial(fits):
    s, p = fits[(1, "raise")], fits[(WORKERS, "raise")]
    assert s["exception"] == "RuntimeError: injected exception in a column solve"
    assert p["exception"] == s["exception"]


def test_dead_worker_is_rerun_serially(fits):
    s, p = fits[(1, "none")], fits[(WORKERS, "crash")]
    assert differences(s, p) == []
    reruns = [x for r in _parallel_records(p["record"]) for x in r["serial_reruns"]]
    assert reruns and reruns[0]["column"] == 2


def test_parallel_series_is_bitwise_serial(tmp_path_factory):
    s, p = _run(tmp_path_factory, "series", 1), _run(tmp_path_factory, "series", WORKERS)
    assert s["series"]["status"] == "completed", s["series"]["status"]
    fits_s = [x["filter"] for x in s["series"]["slices"] if x.get("filter")]
    assert fits_s and all(f.get("absorbed") for f in fits_s)
    assert differences(s, p) == []


if __name__ == "__main__":
    child(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
