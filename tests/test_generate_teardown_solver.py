"""Two sequential in-process sigma=0 runs are bit-identical (issue #13) --
the LIVE-SOLVER half (``pytest -m solver``).

``generate_bouquet`` installs a strong coil regularisation and coil-drift
stashes on the solver object for its draws; since 1.4.1 its exit teardown
puts back the reconstruction's regularisation, releases the bounds and
clears the stashes.  The issue's regression test: one prepared baseline on
the live solver, ``generate()`` twice in the same interpreter with every
perturbation zero (the uncertainty envelope zeroed, no bootstrap scale
range) and the same seed, into two archives.  The two archives must be
identical: every dataset bit for bit and every attribute equal, except the
wall-clock records and the run's own output header (:data:`_CLOCK`).  The
solver state after each run must also be the reconstruction's (its coil
regularisation back, no stash).

On the unified engine the class API was already mitigated before the
teardown (the engine's entry restores the reconstruction's delivered
state, and each engine draw installs the reconstruction's regularisation),
so this test guards that path; the fast half shows the teardown itself.

The synthetic g-file + IDA-layout profile pair (the g-file golden's
inputs), the unified engine, one thread.  One interpreter (``OFT_env`` is a
per-process singleton).  Synthetic inputs only.  The fast half is
``tests/test_generate_teardown.py``.
"""
import json
import os
import re
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest  # noqa: E402

_DATA = os.path.join(_harness.REPO_ROOT, "tests", "data")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_PROF = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_MESH = os.path.join(_harness.REPO_ROOT, "examples", "D3D-like",
                     "DIIID_mesh.h5")
_N = 2

#: attribute / JSON-key names that record wall-clock time, not results
#: (measured: the root's created / updated, every wall_s of the loop and
#: cost records), and the run's own output header (in the stored config)
_CLOCK = re.compile(r"^(created|updated|wall_s|output_header)$")


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_harness.REPO_ROOT, "..", "OpenFUSIONToolkit",
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


_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PROF, _MESH))
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not (_files_ok and _oft_importable()),
                       reason="needs OFT + the synthetic fixtures"),
]


def _strip_clock(obj):
    if isinstance(obj, dict):
        return {k: _strip_clock(v) for k, v in obj.items()
                if not _CLOCK.search(str(k))}
    if isinstance(obj, list):
        return [_strip_clock(v) for v in obj]
    return obj


def _attr_value(v):
    """An attribute as comparable data (JSON attrs parsed, clocks dropped)."""
    import numpy as np
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return ("json", _strip_clock(json.loads(v)))
        except ValueError:
            return ("str", v)
    if isinstance(v, np.ndarray):
        return ("array", v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (np.generic, float, int, bool)):
        return ("scalar", np.asarray(v).dtype.str, np.asarray(v).tobytes())
    return ("other", repr(v))


def _digest(path):
    """{object path: comparable content} of every group and dataset."""
    import h5py
    import numpy as np
    out = {}

    def visit(name, obj):
        attrs = {k: _attr_value(obj.attrs[k]) for k in sorted(obj.attrs)
                 if not _CLOCK.search(k)}
        if isinstance(obj, h5py.Dataset):
            a = np.asarray(obj[()])
            if a.dtype.kind in "OSU":          # strings: by value, not bytes
                v = a.tolist()
                v = [x.decode() if isinstance(x, bytes) else x
                     for x in (v if isinstance(v, list) else [v])]
                v = [_attr_value(x) for x in v]
                out[name] = ("dataset-str", a.shape, v, attrs)
            else:
                out[name] = ("dataset", a.dtype.str, a.shape, a.tobytes(),
                             attrs)
        else:
            out[name] = ("group", attrs)
    with h5py.File(path, "r") as hf:
        out["/"] = ("group", {k: _attr_value(hf.attrs[k])
                              for k in sorted(hf.attrs)
                              if not _CLOCK.search(k)})
        hf.visititems(visit)
    return out


def child(work):
    """One baseline, two sigma=0 generate() calls in THIS process."""
    import pickle
    import bouquet as bq
    import bouquet.baseline as _bl
    from bouquet.run import _zero_perturbation_env
    _harness.assert_bouquet_is_repo_local()
    rec = {}
    try:
        real = _bl.resolve_uncertainty
        _bl.resolve_uncertainty = (
            lambda *a, **k: _zero_perturbation_env(real(*a, **k)))
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PROF, mesh=_MESH,
                                   nthreads=1, n_draws=_N,
                                   header=os.path.join(work, "run1"),
                                   reconstruction_engine="unified")
        gc = b.config.generation
        gc.seed = 12345
        gc.n_inspec_target = None
        gc.jBS_scale_range = None
        b.setup_solver()
        b.prepare_baseline()
        recon_reg = list(b.mygs._recon_coil_reg)
        rec["runs"] = []
        for tag in ("run1", "run2"):
            b.output_header = os.path.join(work, tag)
            diags = b.generate()
            m = b.mygs
            rec["runs"].append(dict(
                n_diags=len(diags),
                strong_stash=hasattr(m, "_strong_coil_reg"),
                bounds_stash=hasattr(m, "_coil_drift_bounds"),
                recon_reg_kept=(list(getattr(m, "_recon_coil_reg", []))
                                == recon_reg)))
        for tag in ("run1", "run2"):
            with open(os.path.join(work, tag + ".pkl"), "wb") as fh:
                pickle.dump(_digest(os.path.join(work, tag + ".h5")), fh)
    except Exception as exc:                       # recorded, never lost
        import traceback
        rec["error"] = f"{type(exc).__name__}: {exc}\n" \
            + traceback.format_exc()[-3000:]
    with open(os.path.join(work, "teardown.json"), "w") as fh:
        json.dump(rec, fh)


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    import pickle
    work = str(tmp_path_factory.mktemp("teardown"))
    pr = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", work],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(os.path.join(work, "child.log"), "w") as fh:
        fh.write(pr.stdout)
    path = os.path.join(work, "teardown.json")
    assert os.path.isfile(path), pr.stdout[-4000:]
    with open(path) as fh:
        rec = json.load(fh)
    assert "error" not in rec, rec.get("error")
    dig = []
    for tag in ("run1", "run2"):
        with open(os.path.join(work, tag + ".pkl"), "rb") as fh:
            dig.append(pickle.load(fh))
    return rec, dig


def test_each_run_archives_its_draws_and_leaves_no_stash(twin):
    rec, _ = twin
    for r in rec["runs"]:
        assert r["n_diags"] == _N, r
        assert r["strong_stash"] is False and r["bounds_stash"] is False, r
        assert r["recon_reg_kept"] is True, r


def test_two_sequential_sigma0_runs_are_bit_identical(twin):
    _, (a, b) = twin
    assert sorted(a) == sorted(b)
    diff = [k for k in a if a[k] != b[k]]
    assert not diff, f"{len(diff)} objects differ, e.g. {diff[:10]}"


if __name__ == "__main__":
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1])
