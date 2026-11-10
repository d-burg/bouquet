"""Live-solver tests of ``bouquet.kinrecon.score`` (``pytest -m solver``).

A synthetic free-boundary equilibrium is solved on the repository's public
D3D-like mesh (``examples/D3D-like/DIIID_mesh.h5``) from the public synthetic
g-file ``examples/D3D-like/D3Dlike_Hmode_baseline.geqdsk`` (shape and
profiles only), at OFT default solver settings.  A synthetic sensor set
(flux loops and probes along the limiter, Ip, diamagnetic flux, all coil
sets, a midplane MSE fan with made-up geometric coefficients, a vertical
Thomson chord) is evaluated on it and the measurement set is built directly
from arrays.  Checked:

(a) closure: measurements with known Gaussian noise give chi^2/N within the
    statistical band, per group and in total;
(b) zero noise: residuals at machine precision;
(c) sensitivity: a perturbed equilibrium (Ip +3 %, re-solved) scores worse;
(d) no side effects: psi, coil currents, axis and psi bounds bit-identical
    before and after ``score``; repeated scoring is bit-identical;
plus a cross-check of the magnetics diagnostics against OFT's own
reconstruction constraint evaluation (``reconstruction.eval_error``).

``OFT_env`` is a per-process singleton, so all live work runs in ONE child
interpreter (one thread) and writes JSON; the tests assert on it.  Set
``OFT_PYTHONPATH`` to the OFT build's ``python`` directory if OFT is not
importable by default.
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
_SEED = 20261110


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


def _sensors(gs):
    """Synthetic sensor geometry tied to the limiter of the mesh."""
    lim = np.asarray(gs.lim_contour, dtype=float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lim, axis=0).T))]
    def resample(n, inset):
        s = np.linspace(0.0, seg[-1], n, endpoint=False)
        p = np.column_stack((np.interp(s, seg, lim[:, 0]), np.interp(s, seg, lim[:, 1])))
        t = np.column_stack((np.interp(s + 1e-3, seg, lim[:, 0]), np.interp(s + 1e-3, seg, lim[:, 1]))) - p
        t /= np.hypot(t[:, 0], t[:, 1])[:, None]
        c = np.array([1.75, 0.0])
        d = c - p
        d /= np.hypot(d[:, 0], d[:, 1])[:, None]
        return p + inset * d, t
    fl, _ = resample(40, 0.03)
    pr, tan = resample(60, 0.015)
    n_mse = 16
    rm = np.linspace(1.78, 2.20, n_mse)
    a1 = np.linspace(0.70, 0.95, n_mse)
    coef = np.column_stack((a1, np.zeros(n_mse), np.zeros(n_mse), np.full(n_mse, 1.5e-6),
                            np.full(n_mse, 0.05), np.linspace(0.2, 0.4, n_mse),
                            -np.linspace(0.6, 0.8, n_mse), np.full(n_mse, 1.0e-7), np.zeros(n_mse)))
    geo = {
        "flux_loop": {"r": fl[:, 0], "z": fl[:, 1]},
        "bpol_probe": {"r": pr[:, 0], "z": pr[:, 1], "direction_r": tan[:, 0],
                       "direction_z": tan[:, 1]},
        "ip": {}, "diamagnetic_flux": {},
        "pf_current": {"coil_names": list(gs.coil_sets)},
        "mse": {"r": rm, "z": np.full(n_mse, 0.04), "coefficients": coef},
        "thomson_ne": {"r": np.full(40, 1.94), "z": np.linspace(-0.95, 0.95, 40)},
        "thomson_te": {"r": np.full(40, 1.94), "z": np.linspace(-0.95, 0.95, 40)},
    }
    return geo


def _ne(x):
    return 4.5e19 * (1.0 - 0.6 * np.asarray(x) ** 2) + 0.5e19


def _te(x):
    return 2.5e3 * (1.0 - np.asarray(x)) ** 1.5 + 80.0


def _er(r, z):
    return 2.0e4 * np.exp(-((np.asarray(r) - 2.05) / 0.12) ** 2)


def _sigma(name, v):
    """Synthetic sigma model (an EFIT-like 'relative with floor' form)."""
    a = np.abs(v)
    floors = {"flux_loop": (0.03, 2e-3), "bpol_probe": (0.03, 1e-3), "ip": (0.005, 1e3),
              "diamagnetic_flux": (0.03, 1e-5), "pf_current": (0.005, 100.0),
              "mse": (0.0, 3.5e-3), "thomson_ne": (0.05, 1e18), "thomson_te": (0.07, 20.0)}
    rel, floor = floors[name]
    return np.maximum(rel * a, floor)


def _measurement_set(eq, geo, rng=None):
    from bouquet.kinrecon import diagnostics as diag
    from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
    funcs = {"flux_loop": diag.flux_loop, "bpol_probe": diag.bpol_probe, "ip": diag.ip,
             "diamagnetic_flux": diag.diamagnetic_flux, "pf_current": diag.pf_current,
             "mse": lambda e, g: diag.mse(e, g, e_r=_er),
             "thomson_ne": lambda e, g: diag.thomson_ne(e, g, _ne),
             "thomson_te": lambda e, g: diag.thomson_te(e, g, _te)}
    groups = {}
    for name, f in funcs.items():
        syn = f(eq, geo[name])
        valid = np.isfinite(syn.value)
        v = np.where(valid, syn.value, 0.0)
        s = _sigma(name, v)
        if rng is not None:
            v = v + s * rng.standard_normal(v.shape)
        groups[name] = ConstraintGroup(name=name, labels=[f"{name}_{i}" for i in range(v.size)],
                                       measured=v, sigma=s, valid=valid, geometry=geo[name],
                                       meta={"sigma_model": "synthetic_test", "er_corrected": False})
    return MeasurementSet("synthetic", 1.0, 0.01, groups)


def _state(gs):
    cur, _ = gs.get_coil_currents()
    return dict(psi=np.array(gs.get_psi(False)), coils=np.array([cur[k] for k in sorted(cur)]),
                o_point=np.array(gs.o_point), psi_bounds=np.array(gs.psi_bounds))


def child(out):
    from bouquet.kinrecon import diagnostics as diag
    from bouquet.kinrecon.score import chi2_summary, score
    import OpenFUSIONToolkit
    rec = {"oft": os.path.abspath(OpenFUSIONToolkit.__file__)}
    gs, g = _build_equilibrium()
    rec["settings"] = dict(nl_tol=gs.settings.nl_tol, urf=gs.settings.urf, maxits=gs.settings.maxits)
    geo = _sensors(gs)
    kin = {"n_e": _ne, "t_e": _te, "e_r": _er}

    clean = _measurement_set(gs, geo)
    noisy = _measurement_set(gs, geo, rng=np.random.default_rng(_SEED))

    # (d) side effects
    before = _state(gs)
    t0 = score(gs, noisy, kinetics=kin)
    t1 = score(gs, noisy, kinetics=kin)
    after = _state(gs)
    rec["side_effect_identical"] = {k: bool(np.array_equal(before[k], after[k])) for k in before}
    rec["repeat_identical"] = all(np.array_equal(t0.model[k], t1.model[k], equal_nan=True)
                                  for k in t0.model)

    # (a) closure
    summ = chi2_summary(t0)
    rec["closure"] = {k: list(v) for k, v in summ.items()}
    rec["excluded"] = {k: len(gr.meta.get("excluded", [])) for k, gr in t0.measurements.groups.items()}
    rec["n_thomson_inside"] = int(np.count_nonzero(clean.groups["thomson_ne"].valid))

    # (b) zero noise
    tz = score(gs, clean, kinetics=kin)
    rec["zero_noise_max_abs_residual"] = {k: float(np.max(np.abs(tz.residual(k)))) for k in tz.model}

    # MSE without E_r differs from the E_r-including measurements
    tn = score(gs, clean, kinetics={"n_e": _ne, "t_e": _te}, groups=["mse"])
    rec["mse_chi2_without_er"] = tn.chi2("mse")

    # cross-check against OFT's own constraint evaluation (reconstruction.eval_error)
    from OpenFUSIONToolkit.TokaMaker.reconstruction import reconstruction
    m = {k: clean.groups[k] for k in ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current")}
    recon = reconstruction(gs)
    for i in range(m["flux_loop"].measured.size):
        recon.add_flux_loop([geo["flux_loop"]["r"][i], geo["flux_loop"]["z"][i]],
                            m["flux_loop"].measured[i], m["flux_loop"].sigma[i])
    for i in range(m["bpol_probe"].measured.size):
        recon.add_Mirnov([geo["bpol_probe"]["r"][i], geo["bpol_probe"]["z"][i]],
                         np.array([geo["bpol_probe"]["direction_r"][i], 0.0,
                                   geo["bpol_probe"]["direction_z"][i]]),
                         m["bpol_probe"].measured[i], m["bpol_probe"].sigma[i])
    names = geo["pf_current"]["coil_names"]
    recon.set_coil_currents({n: float(v) for n, v in zip(names, m["pf_current"].measured)},
                            {n: float(s) for n, s in zip(names, m["pf_current"].sigma)})
    recon.set_Ip(float(m["ip"].measured[0]), float(m["ip"].sigma[0]))
    recon.set_DFlux(float(m["diamagnetic_flux"].measured[0]), float(m["diamagnetic_flux"].sigma[0]))
    recon.setup_constraints()
    err = recon.eval_error()
    nfl, npr, nco = m["flux_loop"].measured.size, m["bpol_probe"].measured.size, len(names)
    blocks = {"flux_loop": err[:nfl, 1], "bpol_probe": err[nfl:nfl + npr, 1],
              "pf_current": err[nfl + npr:nfl + npr + nco, 1],
              "ip": err[nfl + npr + nco:nfl + npr + nco + 1, 1],
              "diamagnetic_flux": err[nfl + npr + nco + 1:, 1]}
    # OFT's flux constraint evaluates +2*pi*psi_TM; ours is the IMAS -2*pi*psi_TM
    blocks["flux_loop"] = -blocks["flux_loop"]
    xc = {}
    for k, v in blocks.items():
        ours = m[k].measured
        xc[k] = dict(max_abs=float(np.max(np.abs(v - ours))),
                     max_rel=float(np.max(np.abs(v - ours) / np.maximum(np.abs(ours), 1e-30))))
    rec["oft_crosscheck"] = xc
    # OFT passes constraints through a text file written with '{:E}' (7
    # significant digits).  Evaluate ours at those rounded locations/normals.
    rnd = np.vectorize(lambda v: float(f"{v:E}"))
    g_fl = {"r": rnd(geo["flux_loop"]["r"]), "z": rnd(geo["flux_loop"]["z"])}
    g_pr = {k: rnd(v) for k, v in geo["bpol_probe"].items()}
    rec["oft_crosscheck_rounded_geometry"] = {
        "flux_loop": float(np.max(np.abs(blocks["flux_loop"] - diag.flux_loop(gs, g_fl).value)
                                  / np.abs(blocks["flux_loop"]))),
        # OFT uses the normal as written (not renormalized); ours normalizes it
        "bpol_probe": float(np.max(np.abs(blocks["bpol_probe"] - diag.bpol_probe(gs, g_pr).value
                                          * np.hypot(g_pr["direction_r"], g_pr["direction_z"]))
                                   / np.abs(blocks["bpol_probe"])))}
    # coil currents: OFT's coil constraint omits the virtual VSC contribution
    diff = blocks["pf_current"] - m["pf_current"].measured
    rec["pf_current_diff_by_coil"] = {n: float(d) for n, d in zip(names, diff)}
    recon.destroy_constraints()
    rec["state_after_oft_eval_identical"] = {k: bool(np.array_equal(before[k], v))
                                             for k, v in _state(gs).items()}

    # (c) sensitivity: re-solve with Ip +3 %, score against the noisy set of the base
    gs.set_targets(Ip=1.03 * abs(g["ip"]), pax=g["pres"][0])
    gs.solve()
    tp = score(gs, noisy, kinetics=kin)
    rec["perturbed"] = {k: list(v) for k, v in chi2_summary(tp).items()}
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_score")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (_REPO, _OFT, os.environ.get("PYTHONPATH", "")) if p))
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_child_used_default_solver_settings(result):
    assert result["settings"] == dict(nl_tol=1e-6, urf=0.2, maxits=40)


def test_closure_chi2_per_point_in_band(result):
    from scipy.stats import chi2
    for name, (c, n, _) in result["closure"].items():
        lo, hi = chi2.ppf(1e-4, n), chi2.ppf(1 - 1e-4, n)
        assert lo <= c <= hi, f"{name}: chi2 {c:.3f} for {n} points outside [{lo:.2f}, {hi:.2f}]"
    c, n, _ = result["closure"]["total"]
    assert n >= 150    # 40 + 60 + 1 + 1 + 20 + 16 + 2 x (Thomson channels inside the LCFS)


def test_zero_noise_residuals_machine_precision(result):
    for name, r in result["zero_noise_max_abs_residual"].items():
        assert r < 1e-9, f"{name}: {r}"


def test_thomson_outside_lcfs_excluded(result):
    assert 0 < result["n_thomson_inside"] < 40


def test_mse_er_term_matters(result):
    assert result["mse_chi2_without_er"] > 1.0


def test_perturbed_equilibrium_scores_worse(result):
    base, pert = result["closure"], result["perturbed"]
    assert pert["total"][0] > 10.0 * base["total"][0]
    for name in ("flux_loop", "bpol_probe", "ip"):
        assert pert[name][0] > base[name][0]


def test_score_has_no_side_effects(result):
    assert all(result["side_effect_identical"].values()), result["side_effect_identical"]
    assert result["repeat_identical"]


def test_magnetics_match_oft_constraint_evaluation(result):
    xc = result["oft_crosscheck"]
    # Ip and diamagnetic flux: same Fortran integrals, bit-identical.
    for name in ("ip", "diamagnetic_flux"):
        assert xc[name]["max_abs"] == 0.0, (name, xc[name])
    # Flux loops and probes: OFT reads sensor locations from a '{:E}' text
    # file (7 significant digits); at those rounded locations they agree to
    # round-off, at the exact locations to the rounding's effect (~1e-6).
    for name, rel in result["oft_crosscheck_rounded_geometry"].items():
        assert rel < 1e-10, (name, rel)
    for name in ("flux_loop", "bpol_probe"):
        assert xc[name]["max_rel"] < 1e-4, (name, xc[name])


def test_coil_currents_differ_from_oft_only_by_vsc(result):
    # OFT's coil constraint (fit_coil_eval) omits the virtual VSC part that
    # get_coil_currents includes; the fixture puts the VSC on F9A/F9B only.
    for name, d in result["pf_current_diff_by_coil"].items():
        if name in ("F9A", "F9B"):
            continue
        assert d == 0.0, (name, d)
    assert result["pf_current_diff_by_coil"]["F9A"] == pytest.approx(
        -result["pf_current_diff_by_coil"]["F9B"], rel=1e-9)


if __name__ == "__main__":
    child(sys.argv[1])
