"""The homotopy branch guard (#87, opt-in ``guard_bistability``) on the
live solver (``pytest -m solver``).

Three seeded runs of the same two-draw engine ensemble on the synthetic
D3D-like g-file + p-file, each in its own subprocess (``OFT_env`` is a
per-process singleton; a cross-process pair of seeded runs is bit-identical,
see ``tests/test_seeded_reproducibility.py``):

* ``off`` -- the default: the guard does not exist.  Nothing new is written
  (no per-draw record, no group or scan attr, no baseline setting).
* ``on`` -- the guard armed.  Every draw carries its record and attrs, the
  scan its counts; where the guard did not fire, every archived dataset and
  the draw's numbers are BIT-IDENTICAL to ``off`` (the guard only read the
  solver).  The synthetic equilibrium has one solution, so a firing here is
  a false positive and is reported in the failure message.
* ``forced`` -- DIAGNOSTIC mechanics probe: the guard's step threshold set to
  1e-5 a (6 um) for this probe only, so the continuation, its
  regularisation and bound blends and the state restore run on the real
  solver.  No acceptance is touched: the draw is judged by the stage's own
  tests as always; the probe checks the record is complete and that a
  firing which keeps the jumped state (every action but "held") leaves the
  archived draw bit-identical to ``off`` (the restore is exact).

The fast half (detector and continuation on synthetic iterates, stand-in
solver, stamps on the toy engine) is ``tests/test_homotopy_guard.py``.
"""
import json
import os
import subprocess
import sys
import traceback

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))
#: the forced probe's threshold [fraction of a] (DIAGNOSTIC, this probe only)
_FORCED_FRAC = 1e-5


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
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


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not _files_ok, reason="D3D-like example files missing"),
    pytest.mark.skipif(not _oft_importable(),
                       reason="OpenFUSIONToolkit not importable"),
]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, bytes):
        return o.decode()
    return o


def _probe(outdir, mode):
    import h5py
    import bouquet as bq
    from bouquet.engine import load_baseline_engine
    from bouquet.filtering import _scan_group_path
    from bouquet.utils import _group_path, list_equilibrium_indices

    _harness.assert_bouquet_is_repo_local()
    out = dict(mode=mode)
    path = os.path.join(outdir, f"guard_{mode}.json")
    try:
        if mode == "forced":
            import bouquet.homotopy_guard as H
            _Base = H.HomotopyBranchGuard

            class _Forced(_Base):
                def __init__(self, **kw):
                    kw["threshold"] = _FORCED_FRAC
                    super().__init__(**kw)
            H.HomotopyBranchGuard = _Forced
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                   nthreads=1, n_draws=2,
                                   header=os.path.join(outdir, mode))
        g = b.config.generation
        g.seed = 12345
        g.guard_bistability = (mode != "off")
        b.setup_solver()
        b.prepare_baseline()
        diags = b.generate()
        h5 = b.config.output_header + ".h5"
        sk = g.scan_key
        draws = []
        with h5py.File(h5, "r") as hf:
            for i in list_equilibrium_indices(h5, scan_key=sk):
                grp = hf[_group_path(sk, i)]
                ds = {k: np.asarray(grp[k][()]).tolist() for k in grp.keys()
                      if isinstance(grp[k], h5py.Dataset)
                      and grp[k].dtype.kind in "fiub"}
                at = {k: _jsonable(v) for k, v in grp.attrs.items()
                      if not isinstance(v, (bytes, str))
                      or k.startswith("bistability_guard")}
                draws.append(dict(index=int(i), datasets=ds, attrs=at))
            sat = dict(hf[_scan_group_path(sk)].attrs)
        out["draws"] = draws
        out["scan_attrs"] = {k: _jsonable(v) for k, v in sat.items()
                             if "bistability_guard" in k}
        out["records"] = [_jsonable((d.get("engine") or {}).get(
            "bistability_guard")) for d in diags]
        out["baseline_guard"] = _jsonable((load_baseline_engine(
            b.config.output_header, scan_key=sk).get("draws") or {}).get(
                "bistability_guard"))
        _lv = getattr(b, "engine_bistability_guard", None) or {}
        out["live"] = _jsonable({k: v for k, v in _lv.items()
                                 if k != "records"})
        out["live_records"] = _jsonable(_lv.get("records") or [])
        out["n_archived"] = len(draws)
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    finally:
        with open(path, "w") as fh:
            json.dump(out, fh, default=float)


def _run(tmp_path_factory, mode):
    work = tmp_path_factory.mktemp(f"guard_{mode}")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), mode, str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    p = work / f"guard_{mode}.json"
    if not p.exists():
        pytest.fail(f"{mode} probe wrote no JSON (rc={proc.returncode}):\n"
                    f"{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}")
    with open(str(p)) as fh:
        d = json.load(fh)
    assert "fatal" not in d, d["fatal"]
    return d


@pytest.fixture(scope="module")
def off(tmp_path_factory):
    return _run(tmp_path_factory, "off")


@pytest.fixture(scope="module")
def on(tmp_path_factory):
    return _run(tmp_path_factory, "on")


@pytest.fixture(scope="module")
def forced(tmp_path_factory):
    return _run(tmp_path_factory, "forced")


def test_off_writes_nothing_new(off):
    assert off["n_archived"] >= 1
    assert all(r is None for r in off["records"])
    assert off["scan_attrs"] == {}
    assert off["baseline_guard"] is None and off["live"] == {}
    for d in off["draws"]:
        assert not [k for k in d["attrs"] if k.startswith("bistability_guard")]


def test_on_stamps_every_draw_and_the_scan(on):
    from bouquet.homotopy_guard import GUARD_ACTIONS, GUARD_VERSION
    recs = [r for r in on["records"] if r is not None]
    assert len(recs) == len(on["records"]) >= 1
    for r in recs:
        assert r["version"] == GUARD_VERSION and r["armed"] is True
        assert len(r["stages"]) >= 1 and r["action"] in GUARD_ACTIONS
        assert r["entry_axis"]["displacement_frac_a"] is not None
        for st in r["stages"]:
            assert st["step_displacement_frac_a"] is not None
    for d in on["draws"]:
        assert d["attrs"]["bistability_guard_armed"] is True
        assert d["attrs"]["bistability_guard_action"] in GUARD_ACTIONS
    sa = on["scan_attrs"]
    assert sa["n_bistability_guard_armed"] == on["n_archived"]
    assert json.loads(sa["bistability_guard_counts_json"])["n_armed"] == \
        on["n_archived"]
    assert on["baseline_guard"]["on"] is True
    assert on["live"]["n_armed"] == len(recs)


def test_on_without_firing_is_bit_identical_to_off(off, on):
    fired = [r["draw"] for r in on["records"] if r and r["fired"]]
    # one solution on the synthetic case: a firing is a false positive
    assert not fired, (f"the guard fired on the synthetic (single-solution) "
                       f"case, draws {fired}: a false positive")
    assert [d["index"] for d in on["draws"]] == \
        [d["index"] for d in off["draws"]]
    for a, b in zip(off["draws"], on["draws"]):
        assert set(a["datasets"]) == set(b["datasets"])
        for k in a["datasets"]:
            assert a["datasets"][k] == b["datasets"][k], (a["index"], k)
        common = {k for k in a["attrs"]}
        for k in common:
            assert a["attrs"][k] == b["attrs"].get(k), (a["index"], k)


def test_forced_firing_runs_the_continuation_on_the_live_solver(forced):
    from bouquet.homotopy_guard import GUARD_ACTIONS
    # every attempt's record (the forced draw may be archived or rejected)
    r = forced["live_records"][0]
    assert r is not None and r["armed"] is True
    assert r["threshold_frac_a"] == _FORCED_FRAC
    fired = [st for st in r["stages"] if st["fired"]]
    assert fired, "the forced threshold did not fire (every stage moved " \
        "the axis by less than 6 um)"
    for st in fired:
        assert st["action"] in GUARD_ACTIONS[:-1]
        assert st["n_extra_solves"] == len(st["substeps"]) >= 1
        assert st["walk_outcome"] in ("completed", "persistent", "failed")
        assert st["final_step_displacement_frac_a"] is not None
    assert r["n_extra_solves"] == sum(st["n_extra_solves"] for st in fired)
    assert r["fired"] is True and r["action"] in GUARD_ACTIONS[:-1]
    if forced["n_archived"]:
        assert forced["draws"][0]["attrs"]["bistability_guard_fired"] is True


def test_a_kept_jump_leaves_the_draw_bit_identical_to_off(off, forced):
    """Where the forced guard fired but kept the jumped state
    (legitimate / crossed / failed: every action but "held"), the restore
    is exact: the archived draw is bit-identical to the unguarded one."""
    kept = {r["draw"] for r in forced["live_records"]
            if r and r["fired"] and r["action"] != "held"}
    by_off = {d["index"]: d for d in off["draws"]}
    n = 0
    for d in forced["draws"]:
        if d["index"] not in kept or d["index"] not in by_off:
            continue
        o = by_off[d["index"]]
        assert set(o["datasets"]) == set(d["datasets"])
        for k in o["datasets"]:
            assert o["datasets"][k] == d["datasets"][k], (d["index"], k)
        for k in o["attrs"]:
            assert o["attrs"][k] == d["attrs"].get(k), (d["index"], k)
        n += 1
    if not n:
        pytest.skip("no forced firing kept the jumped state on an archived "
                    "draw (every firing held the branch)")


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _probe(sys.argv[2], sys.argv[1])
