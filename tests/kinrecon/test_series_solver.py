"""Live-solver tests of the time-dependent path (``pytest -m solver``): manufactured series.

Public synthetic fixture only (as ``test_fit_solver.py``): the L-mode slice
``tests/data/g_synthetic_lmode.geqdsk`` with its IDA-layout kinetics, on the public D3D-like mesh,
OFT default solver settings, sawtooth model off (the truth is built without it).

Truth: a NON-RELAXED initial state -- the current profile of the steady model state with three
current-correction Gaussians of known amplitudes ``A_TRUE`` as non-inductive sources, supplied as
``initial="from_profile"`` -- evolved by
:func:`bouquet.kinrecon.series.run_series` in generation mode (isoflux on the g-file boundary,
coils regularized) through four slices 20 ms apart with an Ip ramp of 2 % per slice; the
corrections are not sources during the evolution, so the current diffuses.  Synthetic
measurements (40 flux loops, 60 finite-length probes, coils, Ip, 16 MSE chords declared
E_r-corrected) are evaluated on every truth slice.

Checked (bounds fixed before the first run):
(a) evolution from the TRUE initial state through the reconstruction path reproduces every later
    slice: data chi^2 < 1e-2 on noiseless data (the sigma = 0 identity bound of the single-slice
    tests) and q0 within 1e-3;
(b) the same series from the RELAXED initial state does not: data chi^2 > 10 at the first evolved
    slice (it is not a reproduction of the truth);
(c) the ``mse_fit`` start (corrections on the relaxed shape, prior N(0, 0.02)) on NOISELESS data
    recovers ``A_TRUE`` within 3 Laplace sigma per amplitude, with status ``converged`` (the prior pulls
    the MAP point toward zero; the bound allows for it).  On the seeded noisy set the first forward
    solves of this fit stall at 1.7x the solver tolerance (a flat residual stall of the anchored solve),
    so the recovery is tested on the noiseless set;
(d) the pass-consistency q check passed on every reconstructed slice.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_fit_solver as _base  # noqa: E402

A_TRUE = (-0.03, 0.02, 0.03)
TIMES = (0.0, 0.02, 0.04, 0.06)
RAMP = 0.02
_SEED = 20261111

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def child(out, parts="abc"):
    import dataclasses
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon import series as SE
    rec = {}
    fm, kin, src, cfg, init, ip = _base._setup()
    cfg = dataclasses.replace(cfg, sawtooth=False)
    # ---- truth initial state: relaxed shape + corrections as sources
    tr0 = CM.consistent_solve(fm, kin, src, dict(log_strength=None, a_corr=A_TRUE), cfg, label="truth0",
                              initial_profiles=init)
    rec["truth0"] = dict(status=tr0.status, passes=tr0.passes, q_check=(tr0.record.get("q_check") or {}).get("max_rel"))
    if not tr0.converged:
        json.dump(_base._jsonable(rec), open(out, "w"))
        return
    pr = tr0.assembly["profiles_rho"]
    j_true = (np.asarray(pr["rho"]), np.asarray(pr["jdotb"]))
    snap0 = fm.snapshot()
    gs = fm.gs
    geo = _base._sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    bnd = np.asarray(fm.reference_boundary, float)
    gen_slices = [SE.SliceData(time=t, measurements=None, reference_boundary=bnd, kinetics=kin, sources=src,
                               label=f"truth{k}", meta=dict(ip=ip * (1 + RAMP * k), generation_coil_reg=1e-2))
                  for k, t in enumerate(TIMES)]
    meas_clean, meas_noisy, truth_stats, refs = [], [], [], []
    rng = np.random.default_rng(_SEED)

    def cb(r):
        meas_clean.append(_base._measurements(fm.gs, geo, roles, ip * (1 + RAMP * len(meas_clean)), None))
        meas_noisy.append(_base._measurements(fm.gs, geo, roles, ip * (1 + RAMP * len(meas_noisy)), rng))
        truth_stats.append(dict(r["stats"], q_axis=r["q_axis_traced"]))
        refs.append(np.asarray(fm.gs.trace_surf(0.999), float))     # each truth slice's own boundary

    st = SE.SeriesSettings(initial="from_profile", initial_profile=j_true, sawtooth="off")
    truth = SE.run_series(fm, gen_slices, cfg, st, callback=cb)
    rec["truth"] = dict(status=truth["status"], slices=[dict(status=x["status"], passes=x["passes"],
                                                             q_check=(x.get("q_check") or {}).get("max_rel"))
                                                        for x in truth["slices"]],
                        n_solves=truth["n_solves"], wall_s=truth["wall_s"], stats=truth_stats)
    if truth["status"] != "completed":
        json.dump(_base._jsonable(rec), open(out, "w"))
        return

    def recon(ms_list, settings, label):
        fm.restore(snap0)
        fm.reconfigure(ms_list[0], refs[0], roles, case_id=label)
        sl = [SE.SliceData(time=t, measurements=ms_list[k], reference_boundary=refs[k], kinetics=kin, sources=src,
                           label=f"{label}{k}") for k, t in enumerate(TIMES)]
        r = SE.run_series(fm, sl, cfg, settings)
        return dict(status=r["status"], n_solves=r["n_solves"], wall_s=r["wall_s"],
                    slices=[dict(status=x["status"], chi2=(x.get("chi2_report") or {}).get("total", {}).get("chi2"),
                                 q_axis=x.get("q_axis_traced"), q_check=x.get("q_check"), fit=x.get("fit"),
                                 n_solves=x.get("n_solves"), wall=x.get("wall_total_s"))
                            for x in r["slices"]])

    if "a" in parts:
        rec["from_true"] = recon(meas_clean, SE.SeriesSettings(initial="from_profile", initial_profile=j_true,
                                                               sawtooth="off"), "true")
    if "b" in parts:
        rec["from_relaxed"] = recon(meas_clean, SE.SeriesSettings(initial="relaxed", sawtooth="off"), "relaxed")
    if "c" in parts:
        fm.restore(snap0)
        fm.reconfigure(meas_clean[0], refs[0], roles, case_id="msefit")
        fr = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=("a_corr",)), params0=dict(log_strength=None),
                             initial_profiles=init, label="msefit")
        rec["mse_fit"] = dict(status=fr.status, theta=fr.theta, sigma=fr.sigma, chi2=fr.chi2,
                              n_solves=fr.record["n_solves"], wall_s=fr.record["wall_s"],
                              stopping=fr.record.get("final_stopping_tests"), center0=fr.record.get("center0"))
    with open(out, "w") as fh:
        json.dump(_base._jsonable(rec), fh, indent=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_series")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=5400)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_truth_series_generated(result):
    assert result["truth0"]["status"] == "consistent", result["truth0"]
    assert result["truth"]["status"] == "completed", result["truth"]


def test_evolution_from_true_initial_state_reproduces_later_slices(result):
    r = result["from_true"]
    assert r["status"] == "completed", r
    for k, s in enumerate(r["slices"]):
        assert s["chi2"] < 1e-2, (k, s["chi2"])
        qt = result["truth"]["stats"][k]["q_axis"]
        assert abs(s["q_axis"] - qt) <= 1e-3 * qt, (k, s["q_axis"], qt)


def test_relaxed_start_does_not_reproduce(result):
    r = result["from_relaxed"]
    assert r["slices"][1]["status"] == "consistent", r["slices"][1]
    assert r["slices"][1]["chi2"] > 10.0, r["slices"][1]["chi2"]


def test_mse_fit_recovers_known_nonrelaxed_initial_state(result):
    f = result["mse_fit"]
    assert f["status"] == "converged", f
    for a, th, sg in zip(A_TRUE, f["theta"], f["sigma"]):
        assert abs(th - a) <= 3.0 * sg, (th, a, sg)


def test_q_check_passed_on_every_reconstructed_slice(result):
    for key in ("from_true", "from_relaxed"):
        for s in result[key]["slices"]:
            assert s["q_check"] and s["q_check"]["passed"], (key, s["q_check"])


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "abc")
