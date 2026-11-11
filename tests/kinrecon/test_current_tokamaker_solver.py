"""Live-solver test of ``bouquet.kinrecon.current.from_tokamaker`` (``pytest -m solver``).

The public synthetic free-boundary equilibrium of ``test_score_solver.py`` is
solved on the repository's public D3D-like mesh from the public synthetic
g-file (shape and profiles only), at OFT default solver settings, and its
geometry is read with the domain-edge rule (default ``psi_n_edge="auto"``).
Checked:

(a) the edge choice: 0.995 with a stated reason (no 2-D grid; route agreement
    not checked unless given); never beyond 0.999; the previous behavior is
    available as ``psi_n_edge=None``;
(b) internal consistency: TokaMaker q vs the geometric q at the domain samples
    (same trace, round-off) and the Phi identity of the resampled geometry;
(c) independent routes: domain + layer volume and toroidal flux against
    TokaMaker's own volume and toroidal-flux integrals (``get_globals``), and
    the layer's Ampere current at its outermost sample against Ip;
(d) a lumped-layer steady state on the 0.995 and 0.99 geometries: total Ip
    conserved, layer current small and positive, interior q independent of
    the cutoff.

``OFT_env`` is a per-process singleton, so all live work runs in ONE child
interpreter (one thread) and writes JSON; the tests assert on it.  Set
``OFT_PYTHONPATH`` to the OFT build's ``python`` directory if OFT is not
importable by default.  All inputs are synthetic and public.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_MESH = os.path.join(_REPO, "examples", "D3D-like", "DIIID_mesh.h5")
_GEQ = os.path.join(_REPO, "examples", "D3D-like", "D3Dlike_Hmode_baseline.geqdsk")


def _oft_path():
    cand = os.environ.get("OFT_PYTHONPATH")
    if cand and os.path.isdir(cand):
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import OpenFUSIONToolkit
        return os.path.dirname(os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__)))
    except Exception:
        return None


_OFT = _oft_path()
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_OFT is None or not (os.path.isfile(_MESH) and os.path.isfile(_GEQ)),
                       reason="needs OpenFUSIONToolkit and the public D3D-like mesh/g-file"),
]


# ---------------------------------------------------------------------------
# child (runs in its own interpreter)
# ---------------------------------------------------------------------------
def _build_equilibrium():
    """As in test_score_solver.py (same public inputs, OFT defaults)."""
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    g = read_eqdsk(_GEQ)
    oft = OFT_env(nthreads=1)
    gs = TokaMaker(oft)
    pts, lc, reg, coil_dict, cond_dict = load_gs_mesh(_MESH)
    gs.setup_model(pts, lc, reg, cond_dict=cond_dict, coil_dict=coil_dict, order=2,
                   F0=g["rcentr"] * g["bcentr"])
    gs.set_coil_vsc({"F9A": 1.0, "F9B": -1.0})
    gs.set_targets(Ip=abs(g["ip"]), pax=g["pres"][0])
    gs.set_isoflux_constraints(g["rzout"][::4].copy())
    regs = [gs.coil_reg_term({n: 1.0}, target=0.0, weight=1.e-2) for n in gs.coil_sets]
    regs.append(gs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1.e-2))
    gs.set_coil_reg(reg_terms=regs)
    x = np.linspace(0.0, 1.0, 50)
    xe = np.linspace(0.0, 1.0, np.size(g["ffprim"]))
    gs.set_profiles(ffp_prof={"type": "linterp", "x": x, "y": np.interp(x, xe, g["ffprim"])},
                    pp_prof={"type": "linterp", "x": x, "y": np.interp(x, xe, g["pprime"])})
    gs.init_psi(1.7, 0.0, 0.5, 1.5, 0.3)
    gs.solve()        # OFT defaults: nl_tol 1e-6, urf 0.2, maxits 40 (raises if not converged)
    return gs, g


def _sigma(x):
    """Synthetic conductivity on psi_N (pedestal-like edge drop)."""
    return 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - np.asarray(x) ** 2, 0.0, None) ** 1.5)


def _jni(x):
    """Synthetic <J_NI.B> on psi_N with an edge bootstrap-like peak."""
    x = np.asarray(x)
    return 2.0 * (1.5e5 * np.exp(-(x / 0.35) ** 2) + 4.0e5 * np.exp(-((x - 0.975) / 0.02) ** 2))


def child(out):
    from scipy.interpolate import PchipInterpolator
    from bouquet.kinrecon.current import (CurrentDiffusion, CurrentState, IpBoundary, OuterLayer,
                                          choose_psi_n_edge, from_tokamaker)
    rec = {}
    gs, g = _build_equilibrium()
    rec["settings"] = dict(nl_tol=gs.settings.nl_tol, urf=gs.settings.urf, maxits=gs.settings.maxits)
    ip_tm, _, vol_tm, _, _, tflux_tm, _ = gs.get_globals()
    R0 = float(g["rcentr"])
    geos = {}
    for key, kw in (("auto", {}), ("0.99", dict(psi_n_edge=0.99))):
        geo = from_tokamaker(gs, R0, **kw)
        lay = geo.meta["outer_layer_geometry"]
        phi_layer = lay.integral(lay.fpol * lay.avg_inv_r2 / (2.0 * np.pi))
        vol_dom = float(np.trapezoid(geo.dvol_drho, geo.rho))
        rec[key] = dict(psi_n_edge=geo.meta["psi_n_edge"], edge_choice=geo.meta["edge_choice"],
                        q_geo_vs_tm=geo.meta["q_geometric_vs_tokamaker_max_rel"],
                        phi_identity=geo.consistency()["max_rel_error"],
                        volume_rel=(vol_dom + lay.total_volume) / abs(vol_tm) - 1.0,
                        tflux_rel=(geo.phi_b + phi_layer) / abs(tflux_tm) - 1.0,
                        ampere_last_rel=float(lay.ampere_current[-1] / abs(ip_tm) - 1.0),
                        layer_last_psi_n=float(lay.psi_n[-1]), tail_fraction=lay.meta["tail_volume_fraction"],
                        n_layer=int(lay.psi_n.size))
        geos[key] = geo
    rec["too_far_raises"] = False
    try:
        from_tokamaker(gs, R0, psi_n_edge=0.9995)
    except ValueError:
        rec["too_far_raises"] = True
    leg = from_tokamaker(gs, R0, psi_n_edge=None)
    rec["legacy"] = dict(psi_n_edge=leg.meta["psi_n_edge"], has_layer="outer_layer_geometry" in leg.meta,
                         edge_choice=leg.meta["edge_choice"])
    rec["rule_default"] = choose_psi_n_edge().psi_n_edge
    # lumped steady state on both cutoffs
    xs = np.linspace(0.02, 0.98, 300)
    qs = {}
    for key, geo in geos.items():
        lay = geo.meta["outer_layer_geometry"]
        rs = np.concatenate(([0.0], geo.meta["rho_samples"]))
        xsamp = np.concatenate(([0.0], geo.meta["psi_n_samples"]))
        layer = OuterLayer.from_profiles(lay, (lay.psi_n, _sigma(lay.psi_n)), (lay.psi_n, _jni(lay.psi_n)))
        sol = CurrentDiffusion(geo, (rs, _sigma(xsamp)), IpBoundary(abs(ip_tm)), j_ni=(rs, _jni(xsamp)),
                               outer_layer=layer)
        ss = sol.steady_state(CurrentState.uniform_faces(400))
        p = sol.profiles(ss)
        r2x = PchipInterpolator(rs, xsamp)
        qs[key] = PchipInterpolator(r2x(p["rho_faces"]), p["q"])(xs)
        rec[key].update(ip_rel=p["ip"] / abs(ip_tm) - 1.0, layer_fraction=p["outer_layer_current"] / abs(ip_tm),
                        q95=p["q95"], li3=p["li3"], surfaces=p["surfaces"])
    rec["interior_q_cutoff_diff"] = float(np.max(np.abs(qs["0.99"] / qs["auto"] - 1.0)))
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_from_tokamaker")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (_REPO, _OFT, os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_default_solver_settings(result):
    assert result["settings"] == dict(nl_tol=1e-6, urf=0.2, maxits=40)


def test_edge_default_is_r29(result):
    assert result["auto"]["psi_n_edge"] == result["rule_default"] == 0.995
    assert "default" in result["auto"]["edge_choice"] and "route_disagreement" in result["auto"]["edge_choice"]
    assert result["0.99"]["psi_n_edge"] == 0.99 and "requested" in result["0.99"]["edge_choice"]
    assert result["too_far_raises"]
    assert result["legacy"]["psi_n_edge"] == 1.0 and not result["legacy"]["has_layer"]
    assert "previous behavior" in result["legacy"]["edge_choice"]


def test_internal_consistency(result):
    for key in ("auto", "0.99"):
        r = result[key]
        # same TokaMaker trace for q and the averages: agreement to round-off (measured 4e-16)
        assert r["q_geo_vs_tm"] < 1e-10, r
        # Phi identity of the resampled geometry: the construction accuracy bounded in
        # test_from_psi_coordinates_recovers_rho_geometry (1e-4 at 200 samples)
        assert r["phi_identity"] < 1e-4, r


def test_independent_routes_agree(result):
    """Domain + layer volume and toroidal flux vs TokaMaker's own integrals: two geometry routes, bound 1e-3
    (the domain-edge rule's route-agreement tolerance)."""
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["volume_rel"]) < 1e-3, r
        assert abs(r["tflux_rel"]) < 1e-3, r


@pytest.mark.xfail(strict=True, reason="measured 2026-11-10: the traced Ampere current (dV/dpsi <B_p^2>/(2 pi mu0)) "
                   "levels off 1.08e-3 below TokaMaker's finite-element Ip from psi_N 0.9999 to 0.99999 on this "
                   "public mesh -- a route offset inside TokaMaker, larger than the 1e-3 bound; the bound is kept "
                   "and the discrepancy is an open item, not a tolerance change")
def test_layer_ampere_current_matches_ip(result):
    """The layer's Ampere current at its outermost sample (psi_N = 1 - 1e-2 (1 - c)) vs Ip; bound 1e-3 (the
    domain-edge rule's route-agreement tolerance)."""
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["ampere_last_rel"]) < 1e-3, r


def test_lumped_steady_state_on_tokamaker_geometry(result):
    for key in ("auto", "0.99"):
        r = result[key]
        assert abs(r["ip_rel"]) < 1e-14
        assert 0.0 < r["layer_fraction"] < 0.01
        assert "last closed surface" in r["surfaces"]["ip"]
        assert np.isfinite(r["q95"]) and np.isfinite(r["li3"])
    # interior q independent of the cutoff: <= 1e-3
    assert result["interior_q_cutoff_diff"] <= 1e-3


if __name__ == "__main__":
    child(sys.argv[1])
