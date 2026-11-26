"""Live-solver tests (``pytest -m solver``) of the two in-process speed changes of ``test_speed_handoff.py``, on the
public synthetic fixture of ``test_fit_solver.py`` (one child process).

Checked:
(a) the geometry hand-off rests on this: the geometry evaluated on a state restored from a snapshot equals, bit for
    bit and object by object, the geometry the final pass of the consistent solve evaluated on the live state the
    snapshot was taken from (after the solver was moved to another state in between);
(b) a consistent solve from that snapshot with the hand-off equals the same solve without it (passes, assembly,
    statistics, final psi), with one geometry evaluation fewer;
(c) ``forward._stats`` with one ``get_globals`` equals ``_stats`` with two on the same state.
The fit-level identity (the default fit equal to the fit with every option off) is ``test_speed_defaults_solver.py``.
"""
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_fit_solver as _base  # noqa: E402

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def same(a, b, path="", out=None, seen=None):
    """Exact structural comparison (arrays bit for bit, NaN equal to NaN; objects by their attributes; functions by
    code and closure contents); timing keys (names ending in ``wall_s``) are skipped.  Returns the list of differing
    paths."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if len(out) > 20:
        return out
    key = (id(a), id(b))
    if key in seen:
        return out
    seen.add(key)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(map(str, set(a) ^ set(b)))}")
        for k in set(a) & set(b):
            if not str(k).endswith("wall_s"):
                same(a[k], b[k], f"{path}/{k}", out, seen)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", out, seen)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")):
            out.append(f"{path}: arrays differ")
    elif isinstance(a, types.FunctionType) and isinstance(b, types.FunctionType):
        if a.__code__ is not b.__code__:
            out.append(f"{path}: other function")
        else:
            ca = [c.cell_contents for c in (a.__closure__ or ())]
            cb = [c.cell_contents for c in (b.__closure__ or ())]
            same(ca, cb, f"{path}<closure>", out, seen)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (a != a and b != b)):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif hasattr(a, "__dict__") and type(a) is type(b) and not isinstance(a, type):
        same(vars(a), vars(b), f"{path}.", out, seen)
    elif a is not b and a != b:
        out.append(f"{path}: {str(a)[:60]} vs {str(b)[:60]}")
    return out


def _child():
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import forward as FW
    from bouquet.kinrecon import speed as SP
    from bouquet.kinrecon.current import sawtooth as SAW
    fm, kin, src, cfg, init, ip = _base._setup()
    ls_true = float(np.log(SAW.crash_strength(_base._F_TRUE, _base._RATE_TRUE)))
    p1 = dict(log_strength=ls_true, f=_base._F_TRUE)
    p2 = dict(log_strength=ls_true, f=_base._F_TRUE * 1.05)
    out = {}
    with SP.speed_scope(SP.SpeedSettings()):
        tr = CM.consistent_solve(fm, kin, src, p1, cfg, label="truth", initial_profiles=init)
        assert tr.converged, tr.status
        res = CM.consistent_solve(fm, kin, src, p2, cfg, label="center")
        assert res.converged, res.status
        live = res.final_geometry["lag"]
        snap = fm.snapshot()
        geo = CM.final_geometry_of(res, snap)
        # move the solver to another state, then restore the snapshot and evaluate the geometry again
        other = CM.consistent_solve(fm, kin, src, p1, cfg, label="elsewhere")
        assert other.converged, other.status
        fm.restore(snap)
        fresh = CM.evaluate_geometry(fm.gs, kin, cfg)
        out["geometry_differences"] = same(live, fresh)
        # (c) _stats on the restored state, one get_globals and two
        s_on = FW._stats(fm.gs)
    with SP.speed_scope(SP.SpeedSettings(reuse_stats=False)):
        s_off = FW._stats(fm.gs)
    out["stats_differences"] = same(s_on, s_off)
    out["stats_has_iter_li"] = FW.VERIFY_LI in s_on
    # (b) the same consistent solve from the snapshot with and without the hand-off
    runs = {}
    real = CM.evaluate_geometry
    for name, g in (("handoff", geo), ("plain", None)):
        calls = [0]

        def counting(*a, **k):
            calls[0] += 1
            return real(*a, **k)

        CM.evaluate_geometry = counting
        try:
            with SP.speed_scope(SP.SpeedSettings()):
                fm.restore(snap)
                r = CM.consistent_solve(fm, kin, src, dict(p2, f=_base._F_TRUE * 1.02), cfg, label=name, start=snap,
                                        start_geometry=g)
        finally:
            CM.evaluate_geometry = real
        runs[name] = dict(res=r, calls=calls[0], psi=np.asarray(fm.gs.get_psi(False)).copy())
    h, p = runs["handoff"], runs["plain"]
    out["solve_converged"] = bool(h["res"].converged and p["res"].converged)
    out["solve_differences"] = (same(h["res"].assembly, p["res"].assembly, "assembly")
                                + same(h["res"].solve.stats, p["res"].solve.stats, "stats")
                                + same(h["psi"], p["psi"], "psi")
                                + same([dict((k, v) for k, v in d.items() if not k.endswith("wall_s"))
                                        for d in h["res"].record["passes"]],
                                       [dict((k, v) for k, v in d.items() if not k.endswith("wall_s"))
                                        for d in p["res"].record["passes"]], "passes"))
    out["calls"] = [h["calls"], p["calls"]]
    return out


def _run(tmp_path_factory):
    d = tmp_path_factory.mktemp("speed_handoff")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               TMPDIR=str(d))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return _run(tmp_path_factory)


def test_geometry_of_a_restored_snapshot_equals_the_live_geometry(result):
    assert result["geometry_differences"] == []


def test_consistent_solve_with_the_handoff_equals_the_solve_without(result):
    assert result["solve_converged"]
    assert result["solve_differences"] == []
    assert result["calls"][0] == result["calls"][1] - 1


def test_stats_with_one_get_globals_equal_stats_with_two(result):
    assert result["stats_has_iter_li"] and result["stats_differences"] == []


if __name__ == "__main__":
    with open(sys.argv[1], "w") as fh:
        json.dump(_child(), fh)
