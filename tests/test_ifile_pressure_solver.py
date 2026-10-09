"""An archived i-file carries the SAME pressure frame as its g-file (#74
review B1) -- on the live solver (``pytest -m solver``).

One probe (this file run as a script, in its own interpreter: ``OFT_env`` is
a per-process singleton) on the synthetic D3D-like g-file + p-file example,
default engine and ``separatrix_pressure="offset"``, ``write_ifile=True``,
one draw:

* from the baseline state, the i-file and the g-file are written through
  the SAME helpers the draw loop uses (``utils.save_state_ifile`` and
  ``save_eqdsk(**edge_pressure.lcfs_kwargs(p_sep))``), each twice -- with
  the delivered ``p_sep`` and bare.  The i-file's pressure offset equals the
  g-file's at every point (both are ``p_sep``): before the fix the i-file's
  was 0;
* ``generate()`` archives a baseline and a draw i-file, each stamped
  ``ifile_written`` with ``ifile_lcfs_pressure`` = that state's
  ``p_sep_applied``; the i-file's edge pressure sits with its g-file's, not
  ``p_sep`` below it.

Synthetic inputs only; one thread.  Expected wall time a few minutes (list
for the cluster when the laptop budget is tight).
"""
import json
import os
import subprocess
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


def _probe(out):
    _harness.assert_bouquet_is_repo_local()
    if not ME.oft_importable():
        raise SystemExit("OpenFUSIONToolkit is not importable")
    import h5py
    import bouquet as bq
    from bouquet.edge_pressure import lcfs_kwargs
    from bouquet.io.geqdsk import _read_geqdsk
    from bouquet.utils import read_ifile, safe_save_eqdsk, save_state_ifile
    header = os.path.join(out, "ifile_frame")
    # BQ_IFILE_PROBE_LEGACY=1: the legacy engine with the frozen SWB
    # bootstrap (an OFT build without sauter_fc(return_eps) cannot run the
    # default engine before the #60 integration fix)
    legacy = os.environ.get("BQ_IFILE_PROBE_LEGACY") == "1"
    b = bq.Bouquet.from_geqdsk(
        ME._GEQ, profiles=ME._PF, mesh=ME._MESH, nthreads=1, n_draws=1,
        header=header,
        **(dict(reconstruction_engine="legacy") if legacy else {}))
    g = b.config.generation
    if legacy:
        g.jbs_self_consistent = False
    g.write_ifile = True
    b.setup_solver()
    bl = b.prepare_baseline()
    p_sep = float(bl.edge_pressure["p_sep_applied"])
    pad = float(getattr(b.config.source, "psi_pad", 1e-3))
    res = dict(p_sep=p_sep)
    for tag, ps in (("full", p_sep), ("bare", 0.0)):
        path, rec = save_state_ifile(b.mygs, f"{header}_{tag}.ifile",
                                     npsi=g.ifile_npsi, ntheta=g.ifile_ntheta,
                                     psi_pad=pad, p_sep=ps)
        with open(path, "rb") as fh:
            res[f"ifile_p_{tag}"] = read_ifile(fh.read())["p"].tolist()
        safe_save_eqdsk(b.mygs, f"{header}_{tag}.geqdsk", nr=257, nz=257,
                        truncate_eq=True, lcfs_pad=pad, **lcfs_kwargs(ps))
        res[f"gfile_p_{tag}"] = np.asarray(
            _read_geqdsk(f"{header}_{tag}.geqdsk")["PRES"], float).tolist()
    b.generate()
    with h5py.File(header + ".h5", "r") as hf:
        root = hf["scan/0"] if "scan" in hf else hf
        groups = ["_baseline"] + [k for k in root if k.isdigit()]
        for k in groups:
            grp = root[k]
            if "ifile" not in grp:
                res[f"arch_{k}"] = dict(missing=True,
                                        attrs={a: str(v) for a, v in grp.attrs.items()
                                               if a.startswith("ifile")})
                continue
            gpath = f"{header}_arch_{k}.geqdsk"
            with open(gpath, "wb") as fh:
                fh.write(bytes(grp["eqdsk"][()]))
            ep = _read_json_attr(grp, "edge_pressure_json")
            res[f"arch_{k}"] = dict(
                ifile_p=read_ifile(bytes(grp["ifile"][()]))["p"].tolist(),
                gfile_p=np.asarray(_read_geqdsk(gpath)["PRES"], float).tolist(),
                written=bool(grp.attrs.get("ifile_written")),
                lcfs_pressure=float(grp.attrs.get("ifile_lcfs_pressure", np.nan)),
                frame=str(grp.attrs.get("ifile_frame", "")),
                p_sep_applied=(None if ep is None else ep.get("p_sep_applied")))
    with open(os.path.join(out, "ifile_frame.json"), "w") as fh:
        json.dump(res, fh)


def _read_json_attr(grp, name):
    v = grp.attrs.get(name)
    if v is None:
        return None
    return json.loads(v if isinstance(v, str) else v.decode())


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    if not (_files_ok and ME.oft_importable()):
        pytest.skip("needs OFT + the D3D-like example files")
    out = str(tmp_path_factory.mktemp("ifile_frame"))
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), out],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                    OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    js = os.path.join(out, "ifile_frame.json")
    assert proc.returncode == 0 and os.path.exists(js), (
        proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
    with open(js) as fh:
        return json.load(fh)


def _ulp(a):
    """One float32 ulp at the field's largest magnitude: the g-file's real
    precision (OFT casts to single before the write)."""
    return float(np.spacing(np.float32(max(1.0, float(np.max(np.abs(a)))))))


def test_the_ifile_and_the_gfile_carry_the_same_separatrix_pressure(probed):
    p_sep = probed["p_sep"]
    assert p_sep > 0.0                       # the comparison discriminates
    d_if = np.asarray(probed["ifile_p_full"]) - np.asarray(probed["ifile_p_bare"])
    d_g = np.asarray(probed["gfile_p_full"]) - np.asarray(probed["gfile_p_bare"])
    # the i-file is double precision: exactly p_sep at every surface
    np.testing.assert_allclose(d_if, p_sep, rtol=0, atol=1e-9 * max(1.0, p_sep))
    # the g-file: p_sep to its float32 precision (two independent roundings)
    tol = 2.0 * _ulp(probed["gfile_p_full"])
    np.testing.assert_allclose(d_g, p_sep, rtol=0, atol=tol)


def test_generate_archives_stamped_ifiles_in_the_gfile_frame(probed):
    arch = {k: v for k, v in probed.items() if k.startswith("arch_")}
    assert "arch__baseline" in arch and len(arch) >= 2, sorted(arch)
    for k, r in arch.items():
        assert not r.get("missing"), (k, r)
        assert r["written"] and "positive-Ip" in r["frame"]
        if r["p_sep_applied"] is not None:
            assert r["lcfs_pressure"] == pytest.approx(r["p_sep_applied"],
                                                       rel=1e-12, abs=0.0)
        p_if, p_g = np.asarray(r["ifile_p"]), np.asarray(r["gfile_p"])
        edge_gap = abs(float(p_if.min()) - float(p_g.min()))
        if r["lcfs_pressure"] > 0.0:
            # with the fix the edges agree far better than the p_sep the
            # i-file used to lack
            assert edge_gap < 0.5 * r["lcfs_pressure"], (k, edge_gap)


if __name__ == "__main__":
    _probe(sys.argv[1])
