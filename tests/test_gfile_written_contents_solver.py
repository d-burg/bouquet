"""What a g-file written by the LIVE solver contains (``pytest -m solver``).

The solver-marked twin of
``tests/test_edge_pressure_baseline_gfile.py::
test_a_written_gfile_parses_back_to_the_delivered_frame`` (finding 6 of the
2026-10-06 second-pass review: nothing asserted the CONTENTS of a written
g-file, so a change in the solver's ``save_eqdsk(lcfs_pressure=)`` would
have passed every suite).

Each arm runs ``tests/probes/probe_baseline_gfile_frame.py`` in its own
interpreter (``OFT_env`` is a per-process singleton) on the synthetic g-file
+ p-file example.  The probe builds the baseline, writes the
reconstruction's g-file with ``Bouquet.save_baseline_eqdsk``, a BARE
``save_eqdsk`` of the same state (the solver frame), and runs ``generate()``
with one draw.  This test parses the written files back with the
repository's reader (``bouquet.io.geqdsk._read_geqdsk``) and asserts:

* under ``"offset"``: ``PRES`` of the reconstruction's file is the bare
  file's plus the DELIVERED ``p_sep`` (the record's ``p_sep_applied``) at
  every point, the edge included; under ``"legacy"``: ``p_sep_applied`` is
  0 and ``PRES`` is the bare file's.  The bare file's own edge value is the
  solver's pressure on the last (``lcfs_pad``-truncated) surface, not zero
  (4.8 Pa at d874822 on this example, SOLVER_RESULTS figure 10), so the edge
  is compared as ``PRES_edge = p_sep + PRES_bare_edge``;
* ``PPRIME``, ``QPSI``, ``FPOL``, ``FFPRIM`` and the boundary of the two
  files agree within the file's precision (one float32 ulp, see
  ``_parse_tol``; the offset reaches ``PRES`` alone) -- the q, F and boundary of the same solver state through two
  saves;
(That the archive's ``_baseline`` g-file is written by the same save call
as the reconstruction's is pinned solver-free in
``tests/test_edge_pressure_baseline_gfile.py``; the archive is a re-solved
state, so its ``PRES`` is not compared point by point here.)

First solver run 2026-10-06 (cluster, one thread): the two offset arms
failed the original ten-digit bound by up to 3.0e-3 Pa, exactly the
single-precision quantisation above; the bound was corrected to the file's
precision (owner-approved 2026-10-06), nothing in the solver changed.  Synthetic inputs
only; one thread.
"""
import os
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "probes"))
import measure_engine as ME  # noqa: E402

pytestmark = pytest.mark.solver

_files_ok = all(os.path.isfile(p) for p in (ME._GEQ, ME._PF, ME._MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and ME.oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")

ARMS = [("legacy", "offset"), ("legacy", "legacy"),
        ("unified", "offset"), ("unified", "legacy")]


def _parse_tol(a):
    """The g-file's REAL precision: one float32 ulp at the field's largest
    magnitude.  OFT's writer (``gs_save_eqdsk``) casts every field to single
    precision before the ``5e16.9`` write, so a value is carried to seven
    significant digits, not the ten the format prints (measured on the
    2026-10-06 solver run: the PRES difference of two saves was quantised at
    2^-8 Pa for pressures in 32768-65536 Pa, i.e. the two saves round the
    offset and bare frames to float32 independently)."""
    m = max(1.0, float(np.max(np.abs(np.asarray(a, dtype=float)))))
    return float(np.spacing(np.float32(m)))


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    from probe_baseline_gfile_frame import run_probe
    out = {}
    for eng, sep in ARMS:
        work = str(tmp_path_factory.mktemp(f"gfile_{eng}_{sep}"))
        rc, rec, log, files = run_probe(work, "recon", eng, sep)
        out[(eng, sep)] = dict(rc=rc, rec=rec, log=log, files=files)
    return out


def _parsed(written, eng, sep):
    from bouquet.io.geqdsk import _read_geqdsk
    w = written[(eng, sep)]
    assert w["rc"] == 0 and w["rec"] is not None, w["log"]
    return w["rec"], {k: _read_geqdsk(p) for k, p in w["files"].items()}


@solver_only
@pytest.mark.parametrize("eng, sep", ARMS)
def test_the_written_pres_is_the_solver_frame_plus_the_delivered_p_sep(
        written, eng, sep):
    rec, g = _parsed(written, eng, sep)
    p_sep = float(rec["p_sep_applied"])
    if sep == "offset":
        assert p_sep > 0.0
    else:
        assert p_sep == 0.0
    pr, pb = (np.asarray(g[k]["PRES"], dtype=float) for k in ("recon",
                                                                "bare"))
    tol = _parse_tol(pr)
    np.testing.assert_allclose(pr - pb, p_sep, rtol=0, atol=2 * tol)
    assert abs((pr[-1] - pb[-1]) - p_sep) <= 2 * tol


@solver_only
@pytest.mark.parametrize("eng, sep", ARMS)
def test_everything_but_pres_round_trips_unchanged(written, eng, sep):
    _, g = _parsed(written, eng, sep)
    r, b = g["recon"], g["bare"]
    for name in ("PPRIME", "QPSI", "FPOL", "FFPRIM", "RBBBS", "ZBBBS"):
        np.testing.assert_allclose(r[name], b[name], rtol=0,
                                   atol=_parse_tol(b[name]), err_msg=name)
    for name in ("SIMAG", "SIBRY", "CURRENT", "BCENTR", "RMAXIS"):
        assert float(r[name]) == pytest.approx(float(b[name]), rel=1e-9,
                                               abs=0.0), name
