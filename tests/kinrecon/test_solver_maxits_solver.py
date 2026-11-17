"""Live-solver checks of the iteration-limit setting :attr:`bouquet.kinrecon.forward.ForwardSettings.solver_maxits`
(``pytest -m solver``).

Public synthetic fixture only (as ``test_fit_solver.py``): one generation-mode forward solve of the g-file's own
current and pressure from the g-file's psi, in three child interpreters (one OFT environment and one TokaMaker
object per process): the default settings, ``solver_maxits=80`` and ``solver_maxits=3``.  Checked:

(a) default: nothing is set -- OFT's ``nl_tol``, ``urf`` and ``maxits`` read back unchanged before and after the
    solve, the record says ``setting: None``, every solve's ``G7`` is PASS, and the iteration record has one entry
    per solve counted in ``n_solves`` (the warm verification re-solve included), none above OFT's limit;
(b) a raised limit changes nothing for a solve that converges below OFT's limit: ``maxits`` reads back 80,
    ``nl_tol`` and ``urf`` are OFT's, ``G7`` PASS, and the solve takes the same number of iterations and gives a
    bit-identical equilibrium (psi on the nodes, coil currents, stats) to the default;
(c) a limit below what the solve needs stops it there: the solve ends with OFT's iteration-limit error after
    exactly that many iterations, which the iteration record flags (``stopped_at_limit``); ``nl_tol`` and ``urf``
    unchanged, ``G7`` PASS against OFT's defaults with ``maxits`` replaced; and the limit stays in force: two
    further solves right after it, from the same start, each stop after exactly that many iterations too.
The settings values compared here are the TokaMaker wrapper's Python-side settings (the solver has no getter); that
the value reaches the solver is shown by behavior, (c).
"""
import hashlib
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

KINDS = dict(default=None, raised=80, low=3)

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None
                       or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, float)).tobytes()).hexdigest()


def child(out, kind):
    from bouquet.kinrecon import forward as FW
    kw = {} if KINDS[kind] is None else dict(solver_maxits=KINDS[kind])
    fm, kin, src, cfg, init, ip = _base._setup(**kw)
    rec = dict(kind=kind, defaults=fm.defaults, settings_before=fm._read_settings(),
               solver_maxits=fm.settings.as_dict()["solver_maxits"],
               limit=fm.solver_record()["iteration_limit"],
               expected=fm.settings_record()["oft_settings_expected"])
    xj, jj, pg = init
    snap0 = fm.snapshot()
    r = fm.solve((xj, jj), (xj, pg), ip=ip, init=snap0, label="control")
    att = r.record["attempts"]
    rec.update(status=r.status, converged=r.converged, its=[a.get("its") for a in att],
               errors=[a.get("error") for a in att], G7=[a.get("G7") for a in att],
               settings_at_solve=[a.get("settings_at_solve") for a in att],
               settings_after=fm._read_settings(), n_solves=fm.n_solves, iterations=fm.iteration_record(),
               build=FW.solver_build()["liboftpy_sha256"])
    if r.converged:
        rec.update(psi=_sha(fm.gs.get_psi(False)),
                   coils=_sha([v for _, v in sorted(dict(fm.gs.get_coil_currents()[0]).items())]),
                   stats=r.stats)
    if kind == "low":
        # two more solves of the same case from the same start, right after the first: each must stop at the limit
        # too (the setting must stay in force in the solver, not only in the record)
        rep = []
        for k in (1, 2):
            rk = fm.solve((xj, jj), (xj, pg), ip=ip, init=snap0, label=f"repeat{k}")
            rep.append(dict(status=rk.status, converged=rk.converged,
                            its=[a.get("its") for a in rk.record["attempts"]],
                            errors=[a.get("error") for a in rk.record["attempts"]]))
        rec.update(repeats=rep, iterations_after=fm.iteration_record(), n_solves_after=fm.n_solves,
                   settings_after_repeats=fm._read_settings())
    with open(out, "w") as fh:
        json.dump(_base._jsonable(rec), fh, indent=1)


def _run(tmp_path_factory, kind):
    d = tmp_path_factory.mktemp(f"kinrecon_maxits_{kind}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out, kind], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return {k: _run(tmp_path_factory, k) for k in KINDS}


def test_default_sets_nothing_and_logs_every_solve(results):
    d = results["default"]
    oft = d["defaults"]
    assert d["solver_maxits"] is None
    assert d["settings_before"] == d["settings_after"] == oft == d["expected"]
    assert d["limit"]["setting"] is None and d["limit"]["in_effect"] == d["limit"]["now"] == oft["maxits"]
    assert d["converged"], d["status"]
    assert all(g == "PASS" for g in d["G7"]) and all(s == oft for s in d["settings_at_solve"])
    it = d["iterations"]
    assert it["n_solves"] == d["n_solves"] == len(d["its"]) + 1           # the attempt(s) and the verification
    assert it["kind"] == ["solve"] * len(d["its"]) + ["verify"]
    assert it["its"][:len(d["its"])] == d["its"] and it["n_unknown"] == 0
    assert it["n_over_oft_default"] == 0 and it["n_stopped_at_limit"] == 0
    assert 3 < d["its"][-1] < oft["maxits"]        # the case (c) relies on: more than 3, fewer than OFT's limit


def test_raised_limit_changes_nothing_below_the_default_limit(results):
    d, r = results["default"], results["raised"]
    oft = d["defaults"]
    assert r["defaults"] == oft and r["build"] == d["build"]
    assert r["solver_maxits"] == 80
    assert r["settings_after"] == dict(oft, maxits=80) == r["expected"]
    assert r["settings_after"]["nl_tol"] == oft["nl_tol"] and r["settings_after"]["urf"] == oft["urf"]
    assert r["limit"] == dict(r["limit"], setting=80, in_effect=80, oft_default=oft["maxits"], now=80)
    assert all(g == "PASS" for g in r["G7"])
    assert r["converged"] and r["status"] == d["status"]
    assert r["its"] == d["its"]
    assert r["psi"] == d["psi"] and r["coils"] == d["coils"] and r["stats"] == d["stats"]
    assert r["iterations"]["its"] == d["iterations"]["its"]


def test_low_limit_stops_the_solve_there(results):
    lo, d = results["low"], results["default"]
    oft = d["defaults"]
    assert lo["settings_after"] == dict(oft, maxits=3) == lo["expected"]
    assert all(g == "PASS" for g in lo["G7"])
    assert not lo["converged"]
    assert lo["errors"][0] is not None and "maxits" in lo["errors"][0].lower()
    it = lo["iterations"]
    first = it["over_oft_default"][0]
    assert first["stopped_at_limit"] and first["its"] == 3 and first["kind"] == "solve"
    assert it["n_stopped_at_limit"] >= 1 and it["n_solves"] == lo["n_solves"]
    assert it["limit"]["setting"] == 3 and it["limit"]["oft_default"] == oft["maxits"]


def test_low_limit_holds_for_consecutive_solves(results):
    """The limit stays in force in the solver after a solve: the next two solves of the same case, from the same
    start, also end with the limit error after exactly 3 iterations (a solver that fell back to OFT's own limit
    after the first solve would converge them, or take more than 3), and every logged attempt stopped there."""
    lo = results["low"]
    assert len(lo["repeats"]) == 2
    for rp in lo["repeats"]:
        assert not rp["converged"], rp
        assert rp["its"] and all(i is None for i in rp["its"])           # a failed solve returns no count
        assert all(e is not None and "maxits" in e.lower() for e in rp["errors"]), rp["errors"]
    it = lo["iterations_after"]
    assert it["n_solves"] == lo["n_solves_after"] >= lo["n_solves"] + 2
    solves = [(i, k) for i, k in zip(it["its"], it["kind"])]
    assert all(k == "solve" and i == 3 for i, k in solves), solves      # no verification: none converged
    assert it["n_stopped_at_limit"] == len(solves) >= 3
    assert lo["settings_after_repeats"] == dict(lo["defaults"], maxits=3)


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2])
