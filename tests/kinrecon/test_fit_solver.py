"""Live-solver tests of the single-slice reconstruction (``pytest -m solver``): synthetic closure.

Public synthetic fixture only: the sawtoothing L-mode slice ``tests/data/g_synthetic_lmode.geqdsk``
with its IDA-layout kinetics ``tests/data/ida_synthetic_lmode.cdf``, on the public D3D-like mesh
``examples/D3D-like/DIIID_mesh.h5``, OFT default solver settings.

Truth: the current model of ``bouquet.kinrecon.current_model`` (ohmic + Redl bootstrap, sawtooth
periodic state at a known strength) iterated to consistency with a suite-like generation solve
(isoflux on the g-file boundary, coils regularized toward zero).  Synthetic measurements are
evaluated on the truth state (40 flux loops in the absolute gauge, 60 finite-length probes, coils,
Ip, a 16-chord MSE fan with made-up coefficients, declared E_r-corrected), with Gaussian noise at
stated sigmas.  The reconstruction uses the anchored forward solve with the truth's own LCFS as the
reference boundary (so the truth state is an exact zero of every anchor row), E-coils held,
two F-coils free, the rest fitted.

Checked:
(a) noisy closure: |theta_fit - theta_true| <= 3 Laplace sigma, and the data chi^2 inside the
    two-sided 1e-4 band of chi^2(N - n_params - inner_dof) (N data rows; the free parameters of the fit,
    log_strength and the plasma-current target; and the effective number of coil parameters the forward
    solve fits to the same data rows, the trace of its coil solve's hat matrix over them, from the fit's
    own record);
(b) zero noise: theta recovered to < 0.5 Laplace sigma (the Gaussian prior on log_strength pulls the
    MAP point toward the prior mean; with zero noise the data still dominate), data chi^2 < 0.1;
(c) sigma = 0 identity: the consistent solve of the reconstruction configuration at the TRUE
    parameters on noiseless data reproduces the truth state (axis 0.5 mm; q0, l_i, Ip 1e-3, the
    forward model's verified-convergence criterion) with data chi^2 < 1e-2;
(d) failure handling: a consistency loop that cannot converge in its pass budget is reported as
    such and the fit refuses to start from it;
(e) settings-record completeness and the solver settings (OFT defaults, read back);
(f) the inner degrees of freedom: computed without changing the solver's equilibrium (psi and coil currents
    bit-identical before and after), between 0 and the number of coil unknowns, and the boundary
    displacement from the anchor recorded in mm.

``OFT_env`` is a per-process singleton, so all live work runs in ONE child interpreter (one thread)
that writes JSON; the tests assert on it.  Set ``OFT_PYTHONPATH`` to the OFT build's ``python``
directory if OFT is not importable by default.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_MESH = os.path.join(_REPO, "examples", "D3D-like", "DIIID_mesh.h5")
_GEQ = os.path.join(_REPO, "tests", "data", "g_synthetic_lmode.geqdsk")
_IDA = os.path.join(_REPO, "tests", "data", "ida_synthetic_lmode.cdf")
_SEED = 20261110
_F_TRUE, _RATE_TRUE = 0.37, 20.0          # truth: 50 ms period at f = 0.37


def _oft_path():
    cand = os.environ.get("OFT_PYTHONPATH")
    if cand and os.path.isdir(cand):
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import OpenFUSIONToolkit
        return os.path.dirname(os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__)))
    except Exception:  # noqa: BLE001
        return None


_OFT = _oft_path()
pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_OFT is None or not all(os.path.isfile(p) for p in (_MESH, _GEQ, _IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


# ---------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------
def _setup():
    import bouquet as bq
    from bouquet.adapters import gfile_parallel_current
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import forward as FW
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    g = read_eqdsk(_GEQ)
    ida = bq.read_ida(_IDA)
    pn = np.asarray(ida.psi_N)
    kin = CM.KineticProfiles(pn, ida.ne, ida.te, ida.ni, ida.ti, ida.Zeff, source="public synthetic L-mode IDA")
    src = CM.SourceProfiles(pn, source="none (no driven current, no fast ions)")
    st = FW.ForwardSettings(vsc={"F9A": 1.0, "F9B": -1.0})
    ip = abs(float(g["ip"]))
    fm = FW.ForwardModel(load_gs_mesh(_MESH), g["rcentr"] * g["bcentr"], None, np.asarray(g["rzout"], float),
                         settings=st, case_id="truth", ip=ip, generation_coil_reg=1e-2)
    nw, nh = g["nr"], g["nz"]
    rg = g["rdim"] * np.arange(nw) / (nw - 1) + g["rleft"]
    zg = g["zmid"] - 0.5 * g["zdim"] + g["zdim"] * np.arange(nh) / (nh - 1)
    fm.set_psi(FW.psi_on_nodes(np.asarray(fm.gs.r)[:, :2], rg, zg, np.asarray(g["psirz"], float),
                               -np.sign(g["ip"])))
    eqg = bq.read_geqdsk(_GEQ)
    _, parts = gfile_parallel_current(eqg)
    xj = np.linspace(0.0, 1.0, 129)
    init = (xj, np.interp(xj, np.asarray(eqg.psi_N), np.asarray(parts["jphi_in"])),
            np.interp(xj, np.linspace(0.0, 1.0, len(g["pres"])), g["pres"]))
    cfg = CM.CurrentModelSettings(r0=float(g["rcentr"]))
    return fm, kin, src, cfg, init, ip


def _sensors(gs):
    lim = np.asarray(gs.lim_contour, float)
    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(lim, axis=0).T))]

    def resample(n, inset):
        s = np.linspace(0.0, seg[-1], n, endpoint=False)
        p = np.column_stack((np.interp(s, seg, lim[:, 0]), np.interp(s, seg, lim[:, 1])))
        t = np.column_stack((np.interp(s + 1e-3, seg, lim[:, 0]), np.interp(s + 1e-3, seg, lim[:, 1]))) - p
        t /= np.hypot(t[:, 0], t[:, 1])[:, None]
        d = np.array([1.75, 0.0]) - p
        d /= np.hypot(d[:, 0], d[:, 1])[:, None]
        return p + inset * d, t

    fl, _ = resample(40, 0.03)
    pr, tan = resample(60, 0.03)
    n = 16
    coef = np.column_stack((np.linspace(0.70, 0.95, n), np.zeros(n), np.zeros(n), np.full(n, 1.5e-6),
                            np.full(n, 0.05), np.linspace(0.2, 0.4, n), -np.linspace(0.6, 0.8, n),
                            np.full(n, 1.0e-7), np.zeros(n)))
    return {"flux_loop": {"r": fl[:, 0], "z": fl[:, 1]},
            "bpol_probe": {"r": pr[:, 0], "z": pr[:, 1], "direction_r": tan[:, 0], "direction_z": tan[:, 1],
                           "length": np.where(np.arange(60) % 2 == 0, 0.04, -0.03)},
            "pf_current": {"coil_names": list(gs.coil_sets)},
            "mse": {"r": np.linspace(1.78, 2.20, n), "z": np.full(n, 0.04), "coefficients": coef}}


def _measurements(gs, geo, roles, ip, rng):
    from bouquet.kinrecon import diagnostics as D
    from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
    model = {"flux_loop": D.flux_loop(gs, geo["flux_loop"]).value,
             "bpol_probe": D.bpol_probe(gs, geo["bpol_probe"]).value,
             "pf_current": D.pf_current(gs, geo["pf_current"]).value, "ip": D.ip(gs).value,
             "mse": D.mse(gs, geo["mse"], quantity="tan_polarisation_angle").value}
    sig = {"flux_loop": np.maximum(0.01 * np.abs(model["flux_loop"]), 2e-3),
           "bpol_probe": np.maximum(0.02 * np.abs(model["bpol_probe"]), 2e-3),
           "pf_current": np.maximum(0.01 * np.abs(model["pf_current"]), 200.0),
           "ip": np.array([5e-3 * ip]), "mse": np.full(16, 3.5e-3)}
    meta = {"flux_loop": dict(coils_form="absolute", absolute_offset=0.0),
            "mse": dict(er_corrected=True, quantity="tan_polarisation_angle")}
    groups = {}
    for name, v in model.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        m = np.where(ok, v, 0.0) + (rng.standard_normal(v.size) * sig[name] if rng is not None else 0.0)
        if name == "pf_current":          # held coils are held at the measurement: give them the true value
            held = np.array([roles[n] == "held" for n in geo["pf_current"]["coil_names"]])
            m = np.where(held, v, m)
        groups[name] = ConstraintGroup(name=name, labels=[f"{name}_{i}" for i in range(v.size)], measured=m,
                                       sigma=sig[name], valid=ok, geometry=geo.get(name, {}),
                                       meta=dict(sigma_model="synthetic_test", **meta.get(name, {})))
    return MeasurementSet("synthetic", 1.0, 0.01, groups)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def child(out):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.current import sawtooth as SAW
    import OpenFUSIONToolkit
    rec = {"oft": os.path.abspath(OpenFUSIONToolkit.__file__)}
    fm, kin, src, cfg, init, ip = _setup()
    rec["settings"] = fm.defaults
    ls_true = float(np.log(SAW.crash_strength(_F_TRUE, _RATE_TRUE)))
    truth_p = dict(log_strength=ls_true, f=_F_TRUE)
    tr = CM.consistent_solve(fm, kin, src, truth_p, cfg, label="truth", initial_profiles=init)
    rec["truth"] = dict(status=tr.status, passes=tr.passes, stats=tr.solve.stats if tr.converged else None,
                        sawtooth_active=bool(tr.assembly["sawtooth"]["active"]) if tr.converged else None)
    if not tr.converged:
        json.dump(_jsonable(rec), open(out, "w"))
        return
    gs = fm.gs
    truth_stats = dict(tr.solve.stats)
    truth_snap = fm.snapshot()
    geo = _sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    noisy = _measurements(gs, geo, roles, ip, np.random.default_rng(_SEED))
    clean = _measurements(gs, geo, roles, ip, None)
    ref = np.asarray(gs.trace_surf(0.999), float)
    rec["ls_true"] = ls_true
    # (c) identity
    fm.reconfigure(clean, ref, roles, case_id="identity")
    r = CM.consistent_solve(fm, kin, src, truth_p, cfg, start=truth_snap, label="identity")
    from bouquet.kinrecon.score import chi2_summary, score
    tab = score(gs, clean, groups=["flux_loop", "bpol_probe", "pf_current", "ip", "mse"])
    rec["identity"] = dict(status=r.status, stats=r.solve.stats if r.converged else None,
                           chi2=chi2_summary(tab)["total"][0], truth_stats=truth_stats,
                           verification=r.solve.record["policy"].get("verification") if r.converged else None)
    # (d) failure handling
    import dataclasses
    tight = dataclasses.replace(cfg, max_passes=1)
    rf = CM.consistent_solve(fm, kin, src, dict(log_strength=ls_true + 0.5, f=_F_TRUE), tight, start=truth_snap,
                             label="budget")
    rec["budget"] = dict(status=rf.status, converged=rf.converged, reason=rf.record.get("reason"))
    fr = FIT.reconstruct(fm, kin, src, tight, FIT.FitSettings(free=("log_strength",)), params0=dict(log_strength=ls_true + 0.5,
                                                                              f=_F_TRUE), start=truth_snap)
    rec["fit_budget"] = dict(status=fr.status, sigma=fr.sigma)
    # (b) zero noise
    fz = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=("log_strength",)), params0=dict(log_strength=ls_true + 0.5, f=_F_TRUE),
                         start=truth_snap, label="zero")
    rec["zero"] = dict(status=fz.status, theta=fz.theta, sigma=fz.sigma, chi2=fz.chi2,
                       n_solves=fz.record["n_solves"], wall_s=fz.record["wall_s"])
    # (a) noisy
    fm.reconfigure(noisy, ref, roles, case_id="noisy")
    fn = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=("log_strength",)), params0=dict(log_strength=ls_true + 0.5, f=_F_TRUE),
                         start=truth_snap, label="noisy")
    rec["noisy"] = dict(status=fn.status, theta=fn.theta, sigma=fn.sigma, chi2=fn.chi2,
                        n_solves=fn.record["n_solves"], wall_s=fn.record["wall_s"])
    # (f) inner degrees of freedom leave the equilibrium untouched; boundary displacement recorded
    psi0, cc0 = np.array(gs.get_psi(False)), dict(gs.get_coil_currents()[0])
    fm._inner_dof = None
    t0 = time.perf_counter()
    inner = fm.inner_dof()
    rec["inner"] = dict(record=inner, wall_s=time.perf_counter() - t0,
                        psi_identical=bool(np.array_equal(psi0, np.array(gs.get_psi(False)))),
                        coils_identical=bool(cc0 == dict(gs.get_coil_currents()[0])))
    rec["boundary"] = fn.record.get("boundary")
    # (e) record keys
    rr = fn.record
    rec["record_keys"] = sorted(rr)
    rec["forward_keys"] = sorted(rr["forward"])
    rec["forward_settings"] = rr["forward"]["settings"]
    rec["forward_rows_keys"] = sorted(rr["forward"]["rows"])
    rec["constraint_groups"] = sorted(rr["constraints"])
    rec["settings_after"] = fm._read_settings()
    with open(out, "w") as fh:
        json.dump(_jsonable(rec), fh, indent=1)


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_fit")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=3000)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_truth_generated_with_active_sawtooth(result):
    assert result["truth"]["status"] == "consistent", result["truth"]
    assert result["truth"]["sawtooth_active"] is True


def test_default_solver_settings(result):
    assert result["settings"] == dict(nl_tol=1e-6, urf=0.2, maxits=40)
    assert result["settings_after"] == result["settings"]


def test_sigma0_identity(result):
    idr = result["identity"]
    assert idr["status"] == "consistent", idr
    a, b = idr["stats"], idr["truth_stats"]
    assert np.hypot(a["R_axis"] - b["R_axis"], a["Z_axis"] - b["Z_axis"]) <= 5e-4
    for k in ("q_0", "l_i", "Ip"):
        assert abs(a[k] - b[k]) <= 1e-3 * abs(b[k]), k
    assert idr["chi2"] < 1e-2
    assert idr["verification"]["verdict"] == "PASS"


def test_zero_noise_recovery(result):
    z = result["zero"]
    assert z["status"].startswith("converged"), z["status"]
    assert abs(z["theta"][0] - result["ls_true"]) < 0.5 * z["sigma"][0]
    assert z["chi2"]["data"] < 0.1


def test_noisy_closure_within_laplace_and_chi2_band(result):
    from scipy.stats import chi2
    n = result["noisy"]
    assert n["status"].startswith("converged"), n["status"]
    assert abs(n["theta"][0] - result["ls_true"]) <= 3.0 * n["sigma"][0]
    N = n["chi2"]["n_data"]
    # every fitted parameter counted: log_strength and the Ip target, and the coil solve's effective parameters on
    # the same data rows (trace of its hat matrix, from the fit's own record)
    dof = N - n["chi2"]["n_params"] - n["chi2"]["inner_dof"]
    assert n["chi2"]["dof"] == pytest.approx(dof, rel=1e-12) and n["chi2"]["inner_dof"] > 0
    lo, hi = chi2.ppf(1e-4, dof), chi2.ppf(1 - 1e-4, dof)
    assert lo <= n["chi2"]["data"] <= hi, (n["chi2"]["data"], lo, hi)


def test_inner_dof_record_and_boundary_displacement(result):
    inn = result["inner"]
    assert inn["psi_identical"] and inn["coils_identical"]
    r = inn["record"]
    data = r["by_kind"]["flux_loop"] + r["by_kind"]["bpol_probe"] + r["by_kind"]["fitted_coil"]
    assert 0.0 < data < r["n_unknowns"] and r["rank"] == r["n_unknowns"]
    assert sum(r["by_kind"].values()) == pytest.approx(r["rank"], rel=1e-9)        # trace of a projector
    assert result["noisy"]["chi2"]["inner_dof"] == pytest.approx(data, rel=1e-12)
    d = result["boundary"]["displacement_mm"]
    assert "error" not in d, d
    assert np.isfinite(d["rms"]) and d["rms"] == pytest.approx(1e3 * result["boundary"]["rms_m"], rel=1e-12)
    for k in ("max", "outboard_midplane", "inboard_midplane", "x_points", "strike_points", "anchor_factor"):
        assert k in d, k


def test_failure_handling(result):
    assert result["budget"]["converged"] is False and result["budget"]["status"] == "max_passes"
    assert result["fit_budget"]["status"] == "failed_initial_consistency"
    assert all(s is None or np.isnan(s) for s in result["fit_budget"]["sigma"])


def test_settings_record_complete(result):
    for k in ("free", "specs", "settings", "model_settings", "fit_groups", "iterations", "forward", "n_solves",
              "wall_s", "constraints", "chi2_detail", "assembly", "final_center", "boundary"):
        assert k in result["record_keys"], k
    for k in ("settings", "oft_defaults", "oft_settings_now", "rows", "edge_taper_case", "oft_python"):
        assert k in result["forward_keys"], k
    for k in ("flux_loops", "probes", "coils", "anchor", "ip_target"):
        assert k in result["forward_rows_keys"], k
    fs = result["forward_settings"]
    for k in ("anchor_factor", "coil_lambda", "vsc", "held_halfwidth_A", "flux_gauge", "probe_in_solve",
              "edge_taper", "verify_criterion"):
        assert k in fs, k
    assert set(result["constraint_groups"]) >= {"flux_loop", "bpol_probe", "ip", "pf_current", "mse_polarisation_angle"}


if __name__ == "__main__":
    child(sys.argv[1])
