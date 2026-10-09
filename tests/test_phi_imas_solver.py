"""ida_hybrid kinetics held on Φ_N vs ψ_N: where they land in real space.

IDA fits its profiles on one equilibrium's flux surfaces (``g_synthetic``);
bouquet re-solves with the dd's current profile, which differs from that
equilibrium's, so the surfaces move.  Profiles held on ψ_N move with them;
profiles held on Φ_N (the IDA file's own q map) stay tied to the geometry.

Fixtures (tests/data, one anonymised time slice): ``dd_synthetic.json.gz``
(currents), ``diiid_profs_synthetic.cdf`` (kinetics, on g_synthetic's ψ_N and q) and
``g_synthetic.geqdsk`` (the equilibrium IDA was fitted on; also the LCFS).

Each run's outboard-midplane R of the IDA nodes is compared with their R on
g_synthetic.  Every solve runs in a subprocess (tests/_harness.py).  Marked
``solver``; a demonstration, run only with ``BOUQUET_RUN_DEMOS=1``.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

from test_phi_solver_draws import _oft_phi_ready

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))
_TIME = 1.013
_METHODS = ("legacy",)
_NODES = (0.1, 0.98)                      # IDA ψ_N range compared

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(os.environ.get("BOUQUET_RUN_DEMOS") != "1",
                       reason="demonstration: BOUQUET_RUN_DEMOS=1 pytest -m solver "
                              "tests/test_phi_imas_solver.py"),
    pytest.mark.skipif(not (all(os.path.isfile(p) for p in (_DD, _IDA, _GEQ, _MESH))
                            and _oft_phi_ready()),
                       reason="needs OFT with toroidal-flux support"),
]


# ---------------------------------------------------------------------------
#  probes (subprocess)
# ---------------------------------------------------------------------------
def _r_out(geq, psi_N):
    """Outboard-midplane R [m] of the surfaces ``psi_N`` of a parsed g-file."""
    from scipy.interpolate import RectBivariateSpline
    s = RectBivariateSpline(geq.Z_grid, geq.R_grid, geq.psi_N_RZ)
    r = np.linspace(geq.R_mag, geq.R_grid[-1], 4000)
    pn = s(geq.Z_mag, r)[0]
    k = int(np.argmax(pn >= 1.0)) + 1
    return np.interp(psi_N, np.maximum.accumulate(pn[:k]), r[:k])


def _probe(work, coord, method):
    import warnings
    import bouquet as bq
    from bouquet import coords
    from bouquet.io.geqdsk import read_geqdsk
    ddp = os.path.join(work, "dd.json")
    with gzip.open(_DD, "rb") as fi, open(ddp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    kw = ({"reconstruction_engine": "legacy"} if method == "legacy"
          else {"solve_method": method})
    run = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, n_draws=1, ida_path=_IDA,
                               LCFS_geqdsk=_GEQ, impurity_Z=6.0,
                               header=os.path.join(work, "run"), **kw)
    run.source.coord = coord
    run.config.generation.seed = 42
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    mygs = run.mygs
    gsolved = os.path.join(work, "solved.geqdsk")
    mygs.save_eqdsk(gsolved, nr=257, nz=257, lcfs_pad=1e-3)
    g_ref, g_run = read_geqdsk(_GEQ), read_geqdsk(gsolved)

    ida = bq.read_ida(_IDA, time=_TIME)
    psi_ida = np.asarray(ida.psi_N, float)
    inside, phi_ida = coords.phi_n_from_q(psi_ida, ida.q, bracket=True)
    psi_in = psi_ida[inside]
    sel = (psi_in >= _NODES[0]) & (psi_in <= _NODES[1])
    psi_nodes = psi_in[sel]
    # where the run holds each IDA node: at its ψ_N, or at the solved ψ_N of its Φ_N
    psi_run = (psi_nodes if coord == "psi_n"
               else coords.psi_at(mygs, phi_ida[sel], coords.PHI))
    r_ref = _r_out(g_ref, psi_nodes)
    dr = 1e3 * (_r_out(g_run, psi_run) - r_ref)
    # the run's own n_e in R against IDA's, at the compared nodes
    bl = run.baseline
    r_bl = _r_out(g_run, coords.psi_at(mygs, np.asarray(bl.psi_N, float), coord))
    ne_run = np.interp(r_ref, r_bl, np.asarray(bl.ne, float))
    ne_ida = np.asarray(ida.ne, float)[inside][sel]
    i = int(np.argmax(np.abs(dr)))
    out = {"max_dr_mm": float(np.abs(dr[i])), "at_psi_N": float(psi_nodes[i]),
           "dr_mm": dr.tolist(), "psi_N": psi_nodes.tolist(),
           "ne_misfit": float(np.max(np.abs(ne_run - ne_ida)) / np.max(ne_ida))}
    with open(os.path.join(work, "probe.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    env = _harness.subprocess_env(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                                  MPLBACKEND="Agg")
    keys = [(m, c) for m in _METHODS for c in ("psi_n", "phi_n")]
    works = {k: str(tmp_path_factory.mktemp(f"ida_{k[0]}_{k[1]}")) for k in keys}
    procs = {k: subprocess.Popen([sys.executable, os.path.abspath(__file__), works[k], k[1], k[0]],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in keys}
    out = {}
    for k, pr in procs.items():
        _, err = pr.communicate()
        path = os.path.join(works[k], "probe.json")
        out[k] = (json.load(open(path)) if pr.returncode == 0 and os.path.isfile(path)
                  else {"_error": f"rc={pr.returncode}\n{err[-4000:]}"})
    return out


def _pair(probes, method):
    p = {c: probes[(method, c)] for c in ("psi_n", "phi_n")}
    for c, v in p.items():
        if "_error" in v:
            pytest.fail(f"{method}/{c} probe failed: {v['_error']}")
    return p["psi_n"], p["phi_n"]


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", _METHODS)
def test_phi_held_kinetics_stay_nearer_their_ida_placement(probes, method):
    # measured: ψ_N-held up to 25 mm (core), Φ_N-held up to 5 mm (near the axis)
    psi, phi = _pair(probes, method)
    assert psi["max_dr_mm"] > 10.0, psi["max_dr_mm"]        # the case moves the surfaces
    assert phi["max_dr_mm"] < min(10.0, 0.4 * psi["max_dr_mm"]), (psi["max_dr_mm"], phi["max_dr_mm"])


@pytest.mark.parametrize("method", _METHODS)
def test_phi_held_ne_matches_ida_in_real_space(probes, method):
    psi, phi = _pair(probes, method)
    # measured: 1.7 % of peak (ψ_N-held) vs 0.45 % (Φ_N-held)
    assert phi["ne_misfit"] < 0.5 * psi["ne_misfit"], (psi["ne_misfit"], phi["ne_misfit"])


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    _probe(*sys.argv[1:4])
