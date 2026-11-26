"""Live-solver tests (``pytest -m solver``) of the robustness of the parallel Jacobian columns and of the reused
statistics (``bouquet.kinrecon.speed``).  Public synthetic fixture only (as ``test_fit_solver.py``); each run is a
separate process.

Checked, with every saved value compared bit for bit with the serial fit (wall times and the Jacobians' ``parallel``
records excepted):
(a) a worker that HANGS in a column (injected in workers only) is killed at its time bound and its column, with every
    later one, is re-run serially; the fit ends, equal to the serial one, with the time-outs recorded and no worker
    process or worker directory left;
(b) when the second worker of every Jacobian cannot be FORKED (an injected ``BlockingIOError``, as a process limit
    gives), the first worker is killed and every column is computed serially; the fit ends, equal to the serial one;
(c) an OFT environment made with two OpenMP threads while ``ForwardSettings.nthreads`` is 1 is refused at once (no
    worker is forked: such a worker can wait forever), and so is an environment whose thread count was raised after
    it was made (``set_num_threads``);
(d) a forward solve that makes TWO attempts (a tapered attempt, then the policy's own) gives the same result and
    record with ``reuse_stats`` on and off: the reused statistics are those of the LAST attempt.
"""
import json
import os
import subprocess
import sys
import time

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

WORKERS = 3
HANG_COLUMN = "fx:it0:col2"            # column 2 of the first Jacobian: the first column of the second worker's block

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parallel columns are Linux only"),
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


# ---------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------
def _children():
    out = []
    me = os.getpid()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as fh:
                    if int(fh.read().rsplit(")", 1)[1].split()[1]) == me:
                        out.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return out


def _fixture_fit(mode):
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
    speed = None
    if mode == "hang":
        real = CM.consistent_solve
        parent = os.getpid()

        def wrapped(*a, **kw):
            lab = kw.get("label", "")
            if os.getpid() != parent and (lab.endswith(HANG_COLUMN) or lab.endswith(HANG_COLUMN + ":opp")):
                time.sleep(10 ** 7)                       # a worker that never answers
            return real(*a, **kw)

        CM.consistent_solve = wrapped
        # a short bound, so that the test does not wait long (it may also stop healthy columns: they are then
        # re-run serially too, which changes no value)
        speed = SpeedSettings(column_workers=WORKERS, column_time_factor=0.05, column_time_floor_s=0.0)
    elif mode == "forkfail":
        real_fork, calls = os.fork, [0]

        def fork_second_refused():
            calls[0] += 1
            if calls[0] % 2 == 0:
                raise BlockingIOError(11, "Resource temporarily unavailable (refused for the test)")
            return real_fork()

        os.fork = fork_second_refused
        speed = SpeedSettings.all_on(WORKERS)
    out = dict(mode=mode)
    fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(anchor_follow=False, free=("a_corr",)), start=snap,
                         label="fx", speed=speed)
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    out.update(status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance, chi2=fr.chi2,
               params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves,
               history=fm.history, case_history=fm.case.history)
    time.sleep(0.5)
    out["children_left"] = _children()
    base = os.path.dirname(fm.env.tempdir) if getattr(fm.env, "tempdir", None) else None
    out["worker_dirs_left"] = sorted(x for x in os.listdir(base) if x.startswith("oft_worker_")) if base else []
    return out


def _threads(kind):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.speed import SpeedSettings, check_speed, solver_threads
    if kind == "env2":
        import OpenFUSIONToolkit as OFT
        init = OFT.OFT_env.__init__

        def init2(self, *a, **k):                          # an environment made with 2 threads by the caller
            k["nthreads"] = 2
            init(self, *a, **k)

        OFT.OFT_env.__init__ = init2
    fm, kin, src, cfg, init_prof, ip = _base._setup()
    out = dict(kind=kind, settings_nthreads=fm.settings.nthreads, env_nthreads=getattr(fm.env, "nthreads", None),
               threads=solver_threads(fm)[0])
    par = SpeedSettings(column_workers=2)
    if kind == "env2":
        n0 = fm.n_solves
        t0 = time.perf_counter()
        try:
            FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(anchor_follow=False, free=("a_corr",)), label="fx",
                            speed=par)
            out["refused"] = None
        except ValueError as exc:
            out["refused"] = str(exc)
        out.update(seconds=time.perf_counter() - t0, solves=fm.n_solves - n0, children_left=_children())
    else:
        out["accepted_at_one"] = check_speed(par, fm) is par
        fm.env.set_num_threads(3)
        out["threads_after_raise"] = solver_threads(fm)[0]
        try:
            check_speed(par, fm)
            out["refused"] = None
        except ValueError as exc:
            out["refused"] = str(exc)
        fm.env.set_num_threads(1)
        out["accepted_again"] = check_speed(par, fm) is par
    return out


def _two_attempts(reuse):
    """A consistent solve whose every forward solve makes a tapered attempt before the policy's own attempts."""
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import edge as E
    from bouquet.kinrecon.current import sawtooth as SAW
    from bouquet.kinrecon.speed import SpeedSettings, speed_scope
    fm, kin, src, cfg, init, ip = _base._setup()
    real = E.solve_with_edge_taper
    spec = E.EdgeTaperSetting().spec

    def two(solve, *a, **kw):
        solve(spec)                                        # an earlier attempt on another state (tapered)
        return real(solve, *a, **kw)

    E.solve_with_edge_taper = two
    ls_true = float(np.log(SAW.crash_strength(_base._F_TRUE, _base._RATE_TRUE)))
    with speed_scope(SpeedSettings(reuse_stats=reuse)):
        res = CM.consistent_solve(fm, kin, src, dict(log_strength=ls_true, f=_base._F_TRUE), cfg, label="two",
                                  initial_profiles=init)
    return dict(status=res.status, converged=res.converged, passes=res.passes, record=res.record,
                stats=getattr(res.solve, "stats", None), solve_record=getattr(res.solve, "record", None),
                psi=np.asarray(fm.gs.get_psi(False)), n_solves=fm.n_solves, history=fm.history)


def child(kind, arg, out):
    if kind == "fit":
        res = _fixture_fit(arg)
    elif kind == "threads":
        res = _threads(arg)
    else:
        res = _two_attempts(arg == "on")
    with open(out, "w") as fh:
        json.dump(_base._jsonable(res), fh)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------
def _run(tmp_path_factory, kind, arg, timeout=7200):
    d = tmp_path_factory.mktemp(f"speedrob_{kind}_{arg}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               TMPDIR=str(d))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), kind, arg, out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    return {m: _run(tmp_path_factory, "fit", m) for m in ("serial", "hang", "forkfail")}


def _comparable(r):
    return {k: v for k, v in r.items() if k not in ("children_left", "worker_dirs_left")}


def test_hung_worker_is_killed_at_its_bound_and_rerun_serially(fits):
    s, p = fits["serial"], fits["hang"]
    assert s["status"] in ("converged", "converged_within_noise"), s["status"]
    assert differences(_comparable(s), _comparable(p)) == []
    par = _parallel_records(p["record"])
    first = par[0]
    assert first["timeouts"]                                     # the hung column (or an earlier one) timed out
    assert 2 in first["columns_here"] and 2 not in first["columns_from_workers"]
    assert all(t["waited_s"] >= t["bound_s"] for r in par for t in r["timeouts"])
    assert all(r["serial_reruns"][0]["column"] == r["timeouts"][0]["column"] for r in par if r["timeouts"])
    assert all(r["speed"]["column_time_factor"] == 0.05 and r["time_bound"]["n_col"] == 8 for r in par)
    assert all({str(j) for j in r["columns_from_workers"]} <= set(r["time_bound_s"]) for r in par)
    assert p["children_left"] == [] and p["worker_dirs_left"] == []


def test_refused_fork_falls_back_to_serial(fits):
    s, p = fits["serial"], fits["forkfail"]
    assert differences(_comparable(s), _comparable(p)) == []
    par = _parallel_records(p["record"])
    assert par and all(r["worker_start_failure"] and "BlockingIOError" in r["worker_start_failure"] for r in par)
    assert all(r["columns_here"] == [0, 1, 2, 3] and r["columns_from_workers"] == [] for r in par)
    assert p["children_left"] == [] and p["worker_dirs_left"] == []


def test_default_record_has_no_speed_entries(fits):
    s = fits["serial"]
    assert "speed" not in s["record"] and not _parallel_records(s["record"])
    assert s["children_left"] == []


def test_a_threaded_solver_environment_is_refused_before_any_solve(tmp_path_factory):
    r = _run(tmp_path_factory, "threads", "env2", timeout=1800)
    assert (r["settings_nthreads"], r["env_nthreads"], r["threads"]) == (1, 2, 2)
    assert r["refused"] and "= 2" in r["refused"]
    assert r["solves"] == 0 and r["children_left"] == []
    q = _run(tmp_path_factory, "threads", "env1", timeout=1800)
    assert q["threads"] == 1 and q["accepted_at_one"] and q["accepted_again"]
    assert q["threads_after_raise"] == 3 and q["refused"] and "= 3" in q["refused"]


def test_reused_stats_are_those_of_the_last_attempt(tmp_path_factory):
    off, on = _run(tmp_path_factory, "stats", "off"), _run(tmp_path_factory, "stats", "on")
    assert off["converged"], off["status"]
    assert any(h["label"].endswith(":tapered") for h in off["history"])     # the extra attempts were made
    assert differences(off, on) == []


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2], sys.argv[3])
