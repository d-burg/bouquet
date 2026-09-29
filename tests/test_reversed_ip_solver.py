"""Reversed-current sources reproduce the normal-current baseline -- live solver.

The IMAS reader brings every current profile of a dd into bouquet's
positive-current frame (``Baseline.source_current_sign``), because the
TokaMaker anchor is always solved to ``|Ip|`` with ``F0 = |r0*b0|`` and every
bootstrap bouquet recomputes on it is therefore positive.  Before that, a dd
with ``ip < 0`` kept its NEGATIVE inductive/fixed currents and had a POSITIVE
recomputed bootstrap added against them.

This is the end-to-end regression for that: the shipped synthetic D3D-like dd
is mirrored into all four (Ip, B0) orientations (``tests/_mirror_dd.py``) and
each mirror's baseline is forward-solved -- the full ``prepare_baseline`` --
under every closure path that recomputes a bootstrap:

  * ``diff``                -- the ``from_imas`` default (full-profile SWB);
  * ``diff_floor_isolate``  -- ``floor_j_BS`` + ``isolate_edge_jBS``: the SWB
    clip at >= 0 and the edge-spike isolation, the two places a wrongly signed
    bootstrap would be silently zeroed or mislabelled;
  * ``ohmic_bootstrap``     -- the hybrid closure (bootstrap channel);
  * ``ohmic_structured``    -- the structured SOFT closure with an l_i row and
    the sawtooth q0 gate (q0 target, radial multipliers, Ip posterior).

and, on builds that carry the self-consistent bootstrap loop, each of those
again with the loop both OFF (legacy frozen SWB) and ON (``evaluate_jBS``).

**The bar is BITWISE identity**, not a tolerance: every array on the Baseline
(``j_phi``, ``j_inductive``, ``j_BS``, the fixed components, ``jBS_diff``), the
solved flux, q profile, l_i, the closure multipliers and the whole Ip-closure
record must equal the original orientation's exactly.  That is the right bar,
and the only honest one: the reader's normalisation is a multiplication by
+-1, which IEEE arithmetic performs exactly, every operation the reader then
applies to a current is odd in it (so the mirrored Baseline is bit-identical
-- ``test_reversed_ip_current_sign`` pins that without a solver), and from
there the solver sees IDENTICAL inputs, runs single-threaded (bouquet's
forward solve is deterministic at ``nthreads=1``, see
``Bouquet._forward_solve_imas_baseline``) and executes the identical sequence
in each mirror's process.  Any difference at all means some code path still
reads an orientation it should not.

The only fields allowed to differ are the ones that RECORD the source's
orientation: ``source_current_sign`` / ``source_b0_sign`` (checked against the
mirror that was written) and the dd's own ``q0_dd`` (the dd's own COCOS
estimator, recorded raw; checked in magnitude, which is all the gate uses) --
plus wall-clock timings (the bootstrap loop's ``wall_s``), which measure the
machine, not the equilibrium.

Every live-solver call runs in a subprocess (``OFT_env`` is a per-process
singleton), one per orientation, four at a time, each at ``nthreads=1``.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

# Must precede any `bouquet` import: puts the repo root on sys.path so a probe
# launched as a script imports the tree under test (see _harness).
import _harness
import _mirror_dd as mdd

_HERE = os.path.dirname(os.path.abspath(__file__))
_files_ok = all(os.path.isfile(p) for p in (mdd.EXAMPLE_DD, mdd.EXAMPLE_MESH))

#: Closure paths exercised.  Each entry is (name, GenerationConfig overrides)
#: applied on top of the ``Bouquet.from_imas`` defaults.
_BASE_CONFIGS = (
    ("diff", {}),
    ("diff_floor_isolate", {"floor_j_BS": True, "isolate_edge_jBS": True}),
    ("ohmic_bootstrap", {"jBS_baseline_mode": "ohmic",
                         "closure_channel": "bootstrap"}),
    ("ohmic_structured", {"jBS_baseline_mode": "ohmic",
                          "closure_channel": "structured",
                          "structured_soft": True,
                          "structured_li_target": 1.20,
                          "structured_li_sigma": 0.05,
                          "structured_li_kind": "li_1",
                          "structured_ip_sigma_frac": 0.005}),
)

#: Fields that record the SOURCE orientation and so legitimately differ.
_ORIENTATION_KEYS = ("source_current_sign", "source_b0_sign")
#: Wall-clock timings recorded alongside the physics (the bootstrap loop's
#: ``wall_s``): measurements of the machine, not of the equilibrium.
_WALL_CLOCK_KEYS = ("wall_s",)


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


def _configs():
    """The closure paths, doubled with the bootstrap loop OFF/ON on builds that
    carry ``GenerationConfig.jbs_self_consistent``."""
    from bouquet.config import GenerationConfig
    if not hasattr(GenerationConfig(), "jbs_self_consistent"):
        return list(_BASE_CONFIGS)
    out = []
    for name, ov in _BASE_CONFIGS:
        out.append((name + "_legacy", dict(ov, jbs_self_consistent=False)))
        out.append((name + "_loop", dict(ov, jbs_self_consistent=True)))
    return out


# ---------------------------------------------------------------------------
#  probe (subprocess entry point)
# ---------------------------------------------------------------------------
def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _probe(dd_path, outdir):
    """Forward-solve the dd's baseline under every closure path; dump results.

    Writes ``<outdir>/<config>.npz`` (arrays) and ``<outdir>/<config>.json``
    (scalars + records, or the refusal text).  One Bouquet/TokaMaker is reused
    across configs (``setup_solver`` is idempotent); the generation config is
    rebuilt from scratch for each, so no config leaks into the next.
    """
    import copy
    import bouquet as bq

    os.makedirs(outdir, exist_ok=True)
    b = bq.Bouquet.from_imas(dd_path, mesh=mdd.EXAMPLE_MESH,
                             time=mdd.EXAMPLE_TIME, nthreads=1,
                             header=os.path.join(outdir, "rev"), n_draws=1)
    gen0 = copy.deepcopy(b.config.generation)
    b.setup_solver()
    for name, overrides in _configs():
        g = copy.deepcopy(gen0)
        for k, v in overrides.items():
            setattr(g, k, v)
        b.config.generation = g
        rec = {"config": name, "overrides": overrides}
        arrays = {}
        try:
            bl = b.prepare_baseline()
        except Exception as exc:                     # a refusal is a RESULT
            rec["error"] = f"{type(exc).__name__}: {exc}"
        else:
            for f in ("psi_N", "j_phi", "j_inductive", "j_BS", "j_NBI", "j_RF",
                      "jBS_diff", "jphi_diff", "ne", "te", "ni", "ti", "Zeff",
                      "p_fast"):
                v = getattr(bl, f, None)
                if v is not None:
                    arrays["bl_" + f] = np.asarray(v, dtype=float)
            mygs = b.mygs
            psi_N = np.asarray(bl.psi_N, dtype=float)
            _psi_q, qvals = mygs.get_q(psi=np.clip(psi_N, 1e-3, 1 - 1e-3).copy())[:2]
            arrays["eq_q"] = np.asarray(qvals, dtype=float)
            try:
                arrays["eq_psi"] = np.asarray(mygs.get_psi(False), dtype=float)
            except Exception:
                pass
            glob = mygs.get_globals()
            arrays["eq_globals"] = np.asarray(
                [float(np.asarray(x).ravel()[0]) for x in glob], dtype=float)
            rec.update(
                Ip_target=float(bl.Ip_target), l_i_target=float(bl.l_i_target),
                bs_scale=float(bl.bs_scale), ohm_scale=float(bl.ohm_scale),
                source_current_sign=float(getattr(bl, "source_current_sign", 1.0)),
                source_b0_sign=getattr(bl, "source_b0_sign", None),
            )
            flat = _flatten(bl.li_metrics, "li_metrics.")
            flat.update(_flatten(bl.ip_closure, "ip_closure."))
            for k, v in flat.items():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) and \
                        all(isinstance(x, (int, float, np.floating, np.integer))
                            for x in np.asarray(v, dtype=object).ravel()):
                    arrays[k] = np.asarray(v, dtype=float)
                else:
                    rec[k] = _jsonable(v)
        np.savez(os.path.join(outdir, name + ".npz"), **arrays)
        with open(os.path.join(outdir, name + ".json"), "w") as fh:
            json.dump(rec, fh, default=str)


# ---------------------------------------------------------------------------
#  fixture: all four orientations, one subprocess each, four at a time
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    from concurrent.futures import ThreadPoolExecutor

    work = tmp_path_factory.mktemp("revip")
    with open(mdd.EXAMPLE_DD) as fh:
        dd = json.load(fh)
    paths = mdd.write_mirrors(dd, str(work / "dd"))

    def _run(key):
        out = str(work / mdd.tag(*key))
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), paths[key], out],
            env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
            capture_output=True, text=True, stdin=subprocess.DEVNULL)
        return key, out, proc

    results = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for key, out, proc in ex.map(_run, list(paths)):
            if proc.returncode != 0:
                pytest.fail(f"probe {mdd.tag(*key)} failed (rc={proc.returncode}):"
                            f"\n{proc.stderr[-4000:]}")
            results[key] = out
    return results


def _same(a, b):
    """Exact equality, with NaN == NaN (a recorded-NaN diagnostic must match
    a recorded NaN; json round-trips it as float('nan'))."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _load(outdir, name):
    with open(os.path.join(outdir, name + ".json")) as fh:
        rec = json.load(fh)
    with np.load(os.path.join(outdir, name + ".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    return rec, arrays


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like example dd/mesh; skipped when unavailable")

_CONFIG_NAMES = [n for n, _ in _configs()]
_MIRRORS = [o for o in mdd.ORIENTATIONS if o != (1.0, 1.0)]


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("name", _CONFIG_NAMES)
def test_the_source_orientation_delivers_a_solved_baseline(solved, name):
    """Guard against a vacuous pass: the reference orientation must actually
    SOLVE on every closure path (two identical refusals would compare equal)."""
    rec, arrays = _load(solved[(1.0, 1.0)], name)
    assert "error" not in rec, rec.get("error")
    assert arrays["bl_j_phi"].size > 0
    # every delivered current is in the positive-Ip frame
    assert rec["source_current_sign"] == 1.0
    assert float(np.trapezoid(arrays["bl_j_phi"], arrays["bl_psi_N"])) > 0.0


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("name", _CONFIG_NAMES)
@pytest.mark.parametrize("orient", _MIRRORS, ids=[mdd.tag(*o) for o in _MIRRORS])
def test_mirrored_orientation_reproduces_the_baseline_bitwise(solved, name,
                                                              orient):
    """Every delivered array and every closure record of the mirrored source
    equals the original's EXACTLY (see the module docstring for why bitwise is
    the right bar); only the orientation records differ, and they say what the
    mirror was."""
    ref_rec, ref_arr = _load(solved[(1.0, 1.0)], name)
    rec, arr = _load(solved[orient], name)
    s_ip, s_b0 = orient

    # the orientation records: what the reader saw and did
    if "error" not in rec:
        assert rec["source_current_sign"] == s_ip * ref_rec["source_current_sign"]
        if ref_rec.get("source_b0_sign") is not None:
            assert rec["source_b0_sign"] == s_b0 * ref_rec["source_b0_sign"]

    assert set(arr) == set(ref_arr), (set(arr) ^ set(ref_arr))
    for k in sorted(ref_arr):
        a, b = ref_arr[k], arr[k]
        assert a.shape == b.shape and np.array_equal(a, b, equal_nan=True), (
            f"{name}/{mdd.tag(*orient)}: {k} differs from the original "
            f"orientation (max |diff| "
            f"{float(np.max(np.abs(a - b))) if a.shape == b.shape else 'shape'})")

    assert set(rec) == set(ref_rec), (set(rec) ^ set(ref_rec))
    n_checked = 0
    for k in sorted(ref_rec):
        # the orientation records, at any nesting (Baseline, li_metrics,
        # ip_closure, li_metrics.ip_closure): checked against the mirror
        if k.rsplit(".", 1)[-1] in _ORIENTATION_KEYS:
            leaf = k.rsplit(".", 1)[-1]
            want = (s_ip if leaf == "source_current_sign" else s_b0)
            if ref_rec[k] is not None:
                assert rec[k] == want * ref_rec[k], (k, rec[k], ref_rec[k])
            continue
        if k.rsplit(".", 1)[-1] in _WALL_CLOCK_KEYS:
            continue
        a, b = ref_rec[k], rec[k]
        if k.endswith("q0_dd") and a is not None:
            # the dd's own COCOS q estimator, recorded raw; the gate reads |q0_dd|
            assert abs(a) == abs(b), (k, a, b)
            continue
        assert _same(a, b), f"{name}/{mdd.tag(*orient)}: {k}: {a!r} != {b!r}"
        n_checked += 1
    assert n_checked > 10, n_checked      # the record was actually compared


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("name", [n for n in _CONFIG_NAMES if "structured" in n])
def test_q0_target_is_positive_in_bouquet_frame(solved, name):
    """The q0 target is un-renormalised with the source's axis current; with
    every current in the positive frame it is positive in every orientation
    (it came out NEGATIVE for a reversed-current source before the reader's
    normalisation: anchor q0 > 0 times a dd-frame axis-current ratio < 0)."""
    for orient in mdd.ORIENTATIONS:
        rec, _ = _load(solved[orient], name)
        assert "error" not in rec, rec.get("error")
        assert rec["ip_closure.q0_target"] > 0.0, (mdd.tag(*orient), rec["ip_closure.q0_target"])
        assert rec["ip_closure.current_direction_sign"] == 1.0


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1], sys.argv[2])
