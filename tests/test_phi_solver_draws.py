"""Toroidal-flux (coord="phi_n") draw pipeline, end to end.

  * g-file + p-file: prepare -> sigma=0 check -> generate; the archive records
    profile_coord="phi_n" on the baseline and every draw, draws carry a
    p-file, and the run grid is the g-file's rhovn**2;
  * IMAS (the D3D-like dd with a real rho_tor_norm injected): forward solve ->
    sigma=0 check -> draws -> write_imas_draw back onto the template's
    Phi_N nodes.

Every solve runs in a subprocess (tests/_harness.py).  Marked ``solver``.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_DD = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.2
_SEED = 20260926


def _oft_phi_ready():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        from bouquet import coords
        coords.check_backend(coords.PHI)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(
        not (all(os.path.isfile(p) for p in (_GEQ, _PEQ, _DD, _MESH)) and _oft_phi_ready()),
        reason="needs OFT with toroidal-flux support + the D3D-like example"),
]


# ---------------------------------------------------------------------------
#  probes (subprocess)
# ---------------------------------------------------------------------------
def _dd_with_rho(work):
    """The example dd with core_profiles rho_tor_norm from its equilibrium's q."""
    from scipy.integrate import cumulative_trapezoid
    with open(_DD) as fh:
        dd = json.load(fh)
    eqs = dd["equilibrium"]["time_slice"]
    for it, p in enumerate(dd["core_profiles"]["profiles_1d"]):
        e = eqs[min(it, len(eqs) - 1)]["profiles_1d"]
        pe = np.asarray(e["psi"], float)
        pn_e = (pe - pe[0]) / (pe[-1] - pe[0])
        phi = cumulative_trapezoid(np.abs(np.asarray(e["q"], float)), pn_e, initial=0.0)
        e["rho_tor_norm"] = np.sqrt(phi / phi[-1]).tolist()
        psi = np.asarray(p["grid"]["psi"], float)
        pn = (psi - psi[0]) / (psi[-1] - psi[0])
        p["grid"]["rho_tor_norm"] = np.sqrt(np.interp(pn, pn_e, phi / phi[-1])).tolist()
    ddp = os.path.join(work, "dd_phi.json")
    with open(ddp, "w") as fh:
        json.dump(dd, fh)
    return ddp


def _scan_key(h5):
    """The archive's one scan key (the IMAS path writes a scan layout), else None."""
    from bouquet.utils import discover_scan_keys
    keys = discover_scan_keys(h5) or [None]
    return keys[0]


def _archive(h5, bl_x):
    import bouquet as bq
    from bouquet.utils import profile_coord
    sk = _scan_key(h5)
    sc = bq.BouquetArchive(h5).scan(sk)
    draws = sc.all
    return {"profile_coord": profile_coord(h5, sk),
            "n_draws": len(draws),
            "draw_coords": [d.attrs.get("profile_coord") for d in draws],
            "draw_pfile": [d.pfile_bytes is not None for d in draws],
            "draw_grid_is_run_grid": [bool(np.allclose(d.profiles["psi_N"], bl_x))
                                      for d in draws]}


def _probe_gfile(work):
    import warnings
    import bouquet as bq
    header = os.path.join(work, "gfile")
    run = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PEQ, mesh=_MESH, n_draws=2, header=header,
                                 reconstruction_engine="legacy")
    run.source.coord = "phi_n"
    run.config.generation.jbs_self_consistent = False   # SWB bootstrap (coords.check_run)
    run.config.generation.seed = _SEED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    bl = run.baseline
    from bouquet.io.geqdsk import read_geqdsk
    out = {"coord": bl.coord,
           "grid_is_rhovn2": bool(np.allclose(bl.psi_N, np.asarray(read_geqdsk(_GEQ).rhovn) ** 2))}
    s0 = run.verify_sigma0_consistency(tol_frac=0.02)
    out.update(sigma0_passed=bool(s0["passed"]), sigma0_dev=float(s0["max_dev_frac"]))
    run.generate()
    out.update(_archive(header + ".h5", bl.psi_N))
    with open(os.path.join(work, "probe.json"), "w") as fh:
        json.dump(out, fh)


def _probe_imas(work):
    import warnings
    import bouquet as bq
    from bouquet.io.imas import write_imas_draw
    ddp = _dd_with_rho(work)
    header = os.path.join(work, "imas")
    # A single IMAS draw on this example can fail in the bounded homotopy (in
    # either coordinate); with this seed, 3 draws store at least one.
    run = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, n_draws=3, header=header,
                               LCFS_geqdsk=_GEQ, reconstruction_engine="legacy")
    run.source.coord = "phi_n"
    run.config.generation.jbs_self_consistent = False   # SWB bootstrap (coords.check_run)
    run.config.generation.seed = _SEED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    bl = run.baseline
    if not hasattr(run.source, "psi_pad"):
        run.source.psi_pad = 1e-3
    s0 = run.verify_sigma0_consistency(tol_frac=0.02)
    run.generate()
    out = {"coord": bl.coord, "sigma0_passed": bool(s0["passed"]),
           "sigma0_dev": float(s0["max_dev_frac"])}
    out.update(_archive(header + ".h5", bl.psi_N))
    # the dd's own Φ_N nodes
    with open(ddp) as fh:
        dd = json.load(fh)
    cp = dd["core_profiles"]
    ic = int(np.argmin(np.abs(np.asarray(cp["time"], float) - _TIME)))
    rho = np.asarray(cp["profiles_1d"][ic]["grid"]["rho_tor_norm"], float)
    out["grid_is_rho2"] = bool(np.allclose(bl.psi_N, (rho ** 2 - rho[0] ** 2) / (rho[-1] ** 2 - rho[0] ** 2)))
    if out["n_draws"]:
        import bouquet as bq
        sk = _scan_key(header + ".h5")
        d0 = bq.BouquetArchive(header + ".h5").scan(sk).all[0]
        wp = os.path.join(work, "draw.json")
        write_imas_draw(header + ".h5", d0.count, ddp, wp, scan_key=sk, time=_TIME)
        with open(wp) as fh:
            cpw = json.load(fh)["core_profiles"]["profiles_1d"][ic]
        x_t = (rho ** 2 - rho[0] ** 2) / (rho[-1] ** 2 - rho[0] ** 2)
        out["written_ne_on_rho2"] = bool(np.allclose(
            cpw["electrons"]["density_thermal"],
            np.interp(x_t, d0.profiles["psi_N_kinetic"], d0.profiles["n_e"]), rtol=1e-10))
        out["written_rho_kept"] = bool(np.allclose(cpw["grid"]["rho_tor_norm"], rho))
        psi_w = np.asarray(cpw["grid"]["psi"], float)
        out["written_psi_monotone"] = bool(np.all(np.diff(psi_w) * (psi_w[-1] - psi_w[0]) > 0))
    with open(os.path.join(work, "probe.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------
def _launch(work, kind):
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), work, kind],
        env=_harness.subprocess_env(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                                    MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    works = {k: str(tmp_path_factory.mktemp(f"phi_{k}")) for k in ("gfile", "imas")}
    procs = {k: _launch(w, k) for k, w in works.items()}
    out = {}
    for k, pr in procs.items():
        _, err = pr.communicate()
        path = os.path.join(works[k], "probe.json")
        out[k] = (json.load(open(path)) if pr.returncode == 0 and os.path.isfile(path)
                  else {"_error": f"rc={pr.returncode}\n{err[-4000:]}"})
    return out


def _get(probes, k):
    p = probes[k]
    if "_error" in p:
        pytest.fail(f"{k} probe failed: {p['_error']}")
    return p


# ---------------------------------------------------------------------------
#  g-file draws
# ---------------------------------------------------------------------------
class TestGfileDraws:
    def test_run_grid_is_the_gfile_phi_n(self, probes):
        p = _get(probes, "gfile")
        assert p["coord"] == "phi_n" and p["grid_is_rhovn2"]

    def test_sigma0_reproduces_the_baseline(self, probes):
        p = _get(probes, "gfile")
        assert p["sigma0_passed"], f"max dev {100 * p['sigma0_dev']:.2f}% of peak"

    def test_draws_are_archived_on_phi_n(self, probes):
        p = _get(probes, "gfile")
        assert p["n_draws"] >= 1
        assert p["profile_coord"] == "phi_n"
        assert all(c == "phi_n" for c in p["draw_coords"])
        assert all(p["draw_grid_is_run_grid"])

    def test_every_draw_carries_a_pfile(self, probes):
        p = _get(probes, "gfile")
        assert all(p["draw_pfile"])


# ---------------------------------------------------------------------------
#  IMAS forward solve, one draw, write-back
# ---------------------------------------------------------------------------
class TestImasDraw:
    def test_run_grid_is_the_dd_phi_n(self, probes):
        p = _get(probes, "imas")
        assert p["coord"] == "phi_n" and p["grid_is_rho2"]

    def test_sigma0_reproduces_the_baseline(self, probes):
        p = _get(probes, "imas")
        assert p["sigma0_passed"], f"max dev {100 * p['sigma0_dev']:.2f}% of peak"

    def test_the_draw_is_archived_on_phi_n(self, probes):
        p = _get(probes, "imas")
        assert p["n_draws"] >= 1 and p["profile_coord"] == "phi_n"
        assert all(c == "phi_n" for c in p["draw_coords"])

    def test_write_back_lands_on_the_template_phi_n_nodes(self, probes):
        p = _get(probes, "imas")
        assert p["written_ne_on_rho2"] and p["written_rho_kept"] and p["written_psi_monotone"]


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    work, kind = sys.argv[1], sys.argv[2]
    if kind == "imas":
        _probe_imas(work)
    else:
        _probe_gfile(work)
