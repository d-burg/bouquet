"""emit_slurm_script must carry its own guidance (Copilot review on PR #39).

Three pieces of context used to reach sbatch users only through the Python
API's docstring/warnings -- which someone copying or re-running the committed
scripts never sees:

1. the compute-node environment hint (without OFT importable, every shard
   dies at ``import OpenFUSIONToolkit``) -- this hint shipped in the
   committed example scripts for months and was silently lost when they were
   regenerated under 1.3.1;
2. the ``threads_per_worker > 1`` caveat (reductions stop being
   bit-reproducible; DLSODE hangs under oversubscription);
3. the bundle's embedded config stores the serial TEMPLATE
   (``solver.nthreads = 1``) while the shard runner overwrites it with
   ``threads_per_worker`` at run time.

Solve-free: the emitter only serialises and writes text files.
"""
import json
import warnings

import pytest

from bouquet.config import (BouquetConfig, ReconstructionSource,
                            SolverConfig)
from bouquet.parallel import emit_slurm_script


def _read(path):
    with open(path) as fh:
        return fh.read()


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _mini_config():
    """A structurally-valid config; the emitter never opens these paths."""
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g.geqdsk",
                                    profiles_path="p.peqdsk"),
        solver=SolverConfig(mesh_path="mesh.h5"),
        output_header="emit_test",
    )


def test_emit_carries_the_env_hint_when_no_setup_given(tmp_path):
    paths = emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                              threads_per_worker=1, out_dir=str(tmp_path),
                              job_name="t")
    for key in ("array", "merge"):
        txt = _read(paths[key])
        assert "compute-node environment" in txt, key
        assert "OFT_PYTHONPATH" in txt, key


def test_explicit_setup_lines_replace_the_hint():
    """Passing real setup lines must not stack the placeholder under them."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        paths = emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                                  threads_per_worker=1, out_dir=d,
                                  job_name="t2", setup=["module load x"])
        txt = _read(paths["array"])
        assert "module load x" in txt
        assert "compute-node environment" not in txt


def test_emit_warns_in_script_when_multithreaded(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi = emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                                  threads_per_worker=4,
                                  out_dir=str(tmp_path), job_name="tm")
        single = emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                                   threads_per_worker=1,
                                   out_dir=str(tmp_path), job_name="ts")
    assert "no longer bit-reproducible" in _read(multi["array"])
    assert "no longer bit-reproducible" not in _read(single["array"])


def test_bundle_notes_the_nthreads_overwrite(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                                  threads_per_worker=4,
                                  out_dir=str(tmp_path), job_name="tb")
    b = _read_json(paths["bundle"])
    assert "_shard_note" in b
    assert "threads_per_worker" in b["_shard_note"]
    assert "(= 4)" in b["_shard_note"]


def test_multithread_python_warning_still_fires(tmp_path):
    """The in-script note supplements the API warning; it must not replace
    it."""
    with pytest.warns(UserWarning, match="threads_per_worker"):
        emit_slurm_script(_mini_config(), n_workers=2, seed=1,
                          threads_per_worker=4, out_dir=str(tmp_path),
                          job_name="tw")
