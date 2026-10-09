#!/usr/bin/env python
"""Probe (live solver; pytest does not collect it): the RECONSTRUCTION's
written g-file and the archive's ``_baseline`` g-file carry the same
pressure frame.

On the repository's synthetic example (``examples/D3D-like``: the g-file +
p-file, or the OMAS source at t = 2.3043 s), one interpreter per run
(``OFT_env`` is a per-process singleton)::

    python tests/probes/probe_baseline_gfile_frame.py OUTDIR \
        [--source recon|imas] [--engine unified|legacy] \
        [--sep offset|legacy]

builds the baseline, writes the reconstruction's g-file with
``Bouquet.save_baseline_eqdsk`` (and, for contrast, a bare
``mygs.save_eqdsk`` from the same state), runs ``generate()`` with one draw
and compares ``PRES`` / ``PPRIME`` of the three files: the baseline's
``p_sep_applied``, each file's edge ``PRES``, ``max|PRES_recon -
PRES_archive|`` and the mean of that difference (a constant offset shows as
mean ~ max), ``max|PPRIME_recon - PPRIME_archive|`` (the re-solve
difference only), and the draw's edge ``PRES`` against its own ``p_sep``.
Writes ``OUTDIR/baseline_gfile_frame_<source>_<engine>_<sep>.json``.

What the output means.  The bare save is the solver frame: it ends on the
last (``lcfs_pad``-truncated) flux surface, so its edge ``PRES`` is the
solver's pressure there -- a few Pa on this example (4.8 Pa at d874822),
NOT zero.  Under ``--sep offset`` the reconstruction's file is the bare
file plus the delivered ``p_sep`` at every point, so its edge is
``p_sep_applied + PRES_bare_edge`` (not exactly ``p_sep_applied``) and
``pres_recon_minus_bare_mean`` is ``p_sep_applied`` to the file's float32
precision.  The archive's ``_baseline`` file is written by the same save
call from a RE-SOLVED state: its edge is ``p_sep_applied`` plus that
state's own truncated-surface pressure, and the recon - archive ``PRES``
difference is re-solve noise (mean well below max), not a constant offset.
Under ``--sep legacy``: ``p_sep_applied`` is 0, every file's edge is the
bare truncated-surface value of its own state (the reconstruction's equals
the bare file's), not zero.  The draw's edge reads the same way against
its own pressure frame (``p_sep_own`` is the last point of the draw's
stored ``pressure`` profile).  The written-contents assertions on these
files live in ``tests/test_gfile_written_contents_solver.py``.
Single-threaded, no network, synthetic data only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.dirname(_HERE)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import _harness  # noqa: E402

_harness.ensure_repo_on_syspath()

from measure_engine import (_GEQ, _MESH, _OMAS, _PF, _TIME,  # noqa: E402
                            oft_importable)


def run_probe(outdir, source="recon", engine="unified", sep="offset",
              timeout=None):
    """Run this probe in its own interpreter (``OFT_env`` is a per-process
    singleton); return ``(returncode, json_or_None, log_tail, files)`` with
    ``files`` the written g-files by kind (``recon``, ``bare``,
    ``archive``).  Used by ``tests/test_gfile_written_contents_solver.py``."""
    import json
    import subprocess
    os.makedirs(outdir, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), outdir,
         "--source", source, "--engine", engine, "--sep", sep],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1",
                                    MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1",
                                    MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=timeout)
    tag = f"{source}_{engine}_{sep}"
    js = os.path.join(outdir, f"baseline_gfile_frame_{tag}.json")
    rec = None
    if os.path.exists(js):
        with open(js) as fh:
            rec = json.load(fh)
    files = {k: os.path.join(outdir, f"frame_{tag}_{k}.geqdsk")
             for k in ("recon", "bare", "archive")}
    return (proc.returncode, rec,
            proc.stdout[-3000:] + "\n" + proc.stderr[-3000:], files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--source", choices=("recon", "imas"), default="recon")
    ap.add_argument("--engine", choices=("unified", "legacy"),
                    default="unified")
    ap.add_argument("--sep", choices=("offset", "legacy"), default="offset")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    _harness.assert_bouquet_is_repo_local()
    if not oft_importable():
        raise SystemExit("OpenFUSIONToolkit is not importable")
    import h5py
    import numpy as np
    import bouquet as bq
    from bouquet.io.geqdsk import _read_geqdsk
    tag = f"{a.source}_{a.engine}_{a.sep}"
    header = os.path.join(a.outdir, f"frame_{tag}")
    if a.source == "recon":
        b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                   nthreads=1, n_draws=1, header=header,
                                   reconstruction_engine=a.engine)
    else:
        b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                                 nthreads=1, header=header,
                                 reconstruction_engine=a.engine)
    g = b.config.generation
    g.separatrix_pressure = a.sep
    b.setup_solver()
    bl = b.prepare_baseline()
    p_sep = float(bl.edge_pressure["p_sep_applied"])
    f_recon = header + "_recon.geqdsk"
    f_bare = header + "_bare.geqdsk"
    b.save_baseline_eqdsk(f_recon)
    from bouquet.utils import safe_save_eqdsk
    safe_save_eqdsk(b.mygs, f_bare, nr=257, nz=257, truncate_eq=True,
                    lcfs_pad=float(getattr(b.config.source, "psi_pad",
                                           1e-3)))
    b.generate()
    with h5py.File(header + ".h5", "r") as hf:
        root = hf["scan/0"] if "scan" in hf else hf
        arch = bytes(root["_baseline"]["eqdsk"][()])
        draws = [k for k in root if k.isdigit()]
        draw = (bytes(root[draws[0]]["eqdsk"][()]),
                float(np.asarray(root[draws[0]]["pressure"])[-1])) \
            if draws else None
    f_arch = header + "_archive.geqdsk"
    with open(f_arch, "wb") as fh:
        fh.write(arch)
    r, ar, br = (_read_geqdsk(f) for f in (f_recon, f_arch, f_bare))
    d = np.asarray(r["PRES"]) - np.asarray(ar["PRES"])
    out = dict(
        source=a.source, engine=a.engine, separatrix_pressure=a.sep,
        p_sep_applied=p_sep, p_sep_input=bl.edge_pressure.get("p_sep"),
        edge_pres=dict(recon=float(r["PRES"][-1]),
                       archive=float(ar["PRES"][-1]),
                       bare=float(br["PRES"][-1])),
        pres_recon_minus_archive=dict(max_abs=float(np.max(np.abs(d))),
                                      mean=float(np.mean(d))),
        pprime_recon_minus_archive_max_abs=float(np.max(np.abs(
            np.asarray(r["PPRIME"]) - np.asarray(ar["PPRIME"])))),
        pres_recon_minus_bare_mean=float(np.mean(
            np.asarray(r["PRES"]) - np.asarray(br["PRES"]))))
    if draw is not None:
        f_d = header + "_draw.geqdsk"
        with open(f_d, "wb") as fh:
            fh.write(draw[0])
        out["draw"] = dict(edge_pres=float(_read_geqdsk(f_d)["PRES"][-1]),
                           p_sep_own=draw[1])
    path = os.path.join(a.outdir, f"baseline_gfile_frame_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
