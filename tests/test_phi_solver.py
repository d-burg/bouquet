"""Toroidal-flux (coord="phi_n") solver checks.

  * one p(ψ), j_phi solved ψ-tagged and Φ-tagged gives the same equilibrium;
  * Φ round trip on the D3D-like g-file: the σ=0 prepare in phi_n
    reproduces the g-file's Ip, q0, q95 and l_i;
  * the phi_n run (windows in ψ_N) keeps the psi_n classifier mode.

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_PAD = 1e-3


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
        not (all(os.path.isfile(p) for p in (_GEQ, _PEQ, _MESH)) and _oft_phi_ready()),
        reason="needs OFT with toroidal-flux support + the D3D-like example"),
]


# ---------------------------------------------------------------------------
#  probes (subprocess)
# ---------------------------------------------------------------------------
def _q_phi_map(mygs, psi):
    """Φ_N at ``psi`` from ∫q dψ_N on a fine grid (q clamped outside the pad)."""
    from scipy.integrate import cumulative_trapezoid
    fine = np.linspace(_PAD, 1.0 - _PAD, 801)
    q = np.abs(np.asarray(mygs.get_q(psi=fine.copy())[1], float))
    xe = np.concatenate(([0.0], fine, [1.0]))
    qe = np.concatenate(([q[0]], q, [q[-1]]))
    phi = cumulative_trapezoid(qe, xe, initial=0.0)
    return np.interp(psi, xe, phi / phi[-1])


def _state(mygs, psi):
    stats = mygs.get_stats(lcfs_pad=_PAD, li_normalization="iter")
    return {"Ip": float(mygs.get_globals()[0]), "q0": float(stats["q_0"]),
            "q95": float(stats["q_95"]), "li": float(stats["l_i"]),
            "p": np.asarray(mygs.get_profiles(psi=psi.copy())[3], float).tolist()}


def _probe_direct(work):
    """Test 1: one p, j_phi solved ψ-tagged, then Φ-tagged."""
    import bouquet as bq
    from bouquet import coords
    from bouquet.edge_pressure import solver_pp_profile
    from bouquet.io.geqdsk import read_geqdsk

    run = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PEQ, mesh=_MESH, n_draws=1,
                                 reconstruction_engine="legacy",
                                 header=os.path.join(work, "direct"))
    run.setup_solver()
    mygs = run.mygs
    eq = read_geqdsk(_GEQ)
    x = np.asarray(eq.psi_N, float)
    p = np.asarray(eq.pres, float)
    j = np.abs(np.asarray(eq.j_tor_averaged_direct, float))
    Ip = abs(float(eq.Ip))
    g = eq.geometry
    mygs.init_psi(g["R"][-1], g["Z"][-1], g["a"][-1], g["kappa"][-1], g["delta"][-1])

    def solve(xg, coord):
        mygs.set_targets(Ip=Ip, pax=float(p[0]))
        mygs.set_profiles(pp_prof=solver_pp_profile(
                              xg, p, mygs.psi_bounds[1] - mygs.psi_bounds[0],
                              {"edge_pprime_pin": False}, coord),
                          ffp_prof=coords.oft_prof("jphi-linterp", xg, j, coord))
        mygs.solve()

    ps = np.clip(x, _PAD, 1.0 - _PAD)
    solve(x, coords.PSI)
    out = {"psi": _state(mygs, ps)}
    x_phi = _q_phi_map(mygs, x)                 # Φ_N of the ψ nodes, 1st solve
    x_phi[0], x_phi[-1] = 0.0, 1.0
    out["x_phi"] = x_phi.tolist()

    solve(x_phi, coords.PHI)
    out["phi"] = _state(mygs, ps)
    with open(os.path.join(work, "direct.json"), "w") as fh:
        json.dump(out, fh)


def _probe_prepare(work, coord):
    """Tests 2-3: σ=0 prepare on the g-file + p-file in ``coord``."""
    import warnings
    import bouquet as bq
    from bouquet import TokaMaker_interface as ti

    modes = []
    _orig = ti.classify_jphi_profile

    def _spy(*a, **k):
        r = _orig(*a, **k)
        modes.append(r[0])
        return r

    ti.classify_jphi_profile = _spy
    run = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PEQ, mesh=_MESH, n_draws=1,
                                 reconstruction_engine="legacy",
                                 header=os.path.join(work, f"prep_{coord}"))
    run.source.coord = coord
    run.config.generation.jbs_self_consistent = False   # SWB bootstrap (coords.check_run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    bl = run.baseline
    m = bl.reconstruction_metrics
    stats = run.mygs.get_stats(lcfs_pad=run.source.psi_pad, li_normalization="iter")
    out = {"coord": bl.coord, "modes": modes, "l_i_target": float(bl.l_i_target),
           "Ip_target": float(bl.Ip_target),
           "Ip": float(m["Ip_MA"]) * 1e6, "q0": float(m["q0"]), "q95": float(m["q95"]),
           "li_post": float(stats["l_i"]), "q0_post": float(stats["q_0"]),
           "q95_post": float(stats["q_95"])}
    with open(os.path.join(work, f"prep_{coord}.json"), "w") as fh:
        json.dump(out, fh)


# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------
def _launch(work, *args):
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), work, *args],
        env=_harness.subprocess_env(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                                    MPLBACKEND="Agg"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait(proc, name):
    _, err = proc.communicate()
    if proc.returncode != 0:
        pytest.fail(f"{name} probe failed (rc={proc.returncode}):\n{err[-4000:]}")


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("phi_direct"))
    _wait(_launch(work, "direct"), "direct")
    with open(os.path.join(work, "direct.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("phi_prepare"))
    procs = {c: _launch(work, "prepare", c) for c in ("psi_n", "phi_n")}
    out = {}
    for c, pr in procs.items():
        _wait(pr, f"prepare {c}")
        with open(os.path.join(work, f"prep_{c}.json")) as fh:
            out[c] = json.load(fh)
    return out


# ---------------------------------------------------------------------------
#  1. ψ-tagged vs Φ-tagged: the same equilibrium
# ---------------------------------------------------------------------------
class TestTaggedEquivalence:
    def test_pressure_profile(self, direct):
        a, b = np.asarray(direct["psi"]["p"]), np.asarray(direct["phi"]["p"])
        np.testing.assert_allclose(b, a, rtol=0, atol=2e-3 * np.max(np.abs(a)))

    @pytest.mark.parametrize("k,tol", [("Ip", 1e-4), ("q0", 3e-3), ("q95", 3e-3), ("li", 3e-3)])
    def test_globals(self, direct, k, tol):
        a, b = direct["psi"][k], direct["phi"][k]
        assert abs(b - a) <= tol * abs(a), f"{k}: psi {a:.6g} vs phi {b:.6g}"


# ---------------------------------------------------------------------------
#  2. Φ round trip on the g-file example: the phi_n prepare reproduces the g-file
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfile():
    from bouquet.io.geqdsk import read_geqdsk
    g = read_geqdsk(_GEQ)
    q = np.abs(np.asarray(g.qpsi, float))
    return {"Ip": abs(float(g.Ip)), "q0": float(q[0]),
            "q95": float(np.interp(0.95, np.asarray(g.psi_N, float), q))}


class TestGfileRoundTrip:
    def test_coord(self, prepared):
        assert prepared["phi_n"]["coord"] == "phi_n"

    # measured: Ip 4e-6, q0 1.3 % (on axis), q95 0.08 %
    @pytest.mark.parametrize("k,tol", [("Ip", 1e-4), ("q0", 2.5e-2), ("q95", 3e-3)])
    def test_scalars_match_the_gfile(self, prepared, gfile, k, tol):
        a, b = gfile[k], prepared["phi_n"][k]
        assert abs(b - a) <= tol * abs(a), f"{k}: g-file {a:.6g} vs phi_n {b:.6g}"

    def test_li_matches_its_gfile_target(self, prepared):
        # l_i_target is the g-file's l_i as bouquet measures it; measured 5e-4
        p = prepared["phi_n"]
        assert abs(p["li_post"] - p["l_i_target"]) <= 3e-3 * p["l_i_target"], p


# ---------------------------------------------------------------------------
#  3. the windows in ψ_N keep the classifier's mode
# ---------------------------------------------------------------------------
class TestClassifierWindows:
    def test_classifier_mode_matches_psi_run(self, prepared):
        a, b = prepared["psi_n"]["modes"], prepared["phi_n"]["modes"]
        assert a and b, "classify_jphi_profile was not called"
        assert b[0] == a[0], f"psi_n {a} vs phi_n {b}"


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_phi_ready()
    if sys.argv[2] == "direct":
        _probe_direct(sys.argv[1])
    else:
        _probe_prepare(sys.argv[1], sys.argv[3])
