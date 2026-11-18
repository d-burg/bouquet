"""Live-solver tests (``pytest -m solver``) of the steady-state input check (``steady_state_check``, ON by default; a
record that steers nothing).  Public synthetic fixture only (as ``test_fit_solver.py``); each run is a separate
process; package defaults otherwise (no ``speed`` argument).

Checked, every saved value compared bit for bit (wall times excepted), check ON against check OFF:
(a) a single-slice fit (default free set, noisy data, default core): every fitted number, the fit's whole record but
    the check's own entry, and the final psi are identical; the fit's own ``n_solves`` is identical; the forward
    model's counter differs by exactly the check's solves, which its record states, and the solver's residual
    histories of the OFF run are the leading part of the ON run's; the check was computed (status, l_i, q0, the MSE
    chords by radius of both states);
(b) a series of three slices (the time-dependent path's synthetic truth, noiseless data, ``initial="mse_fit"`` so the
    first slice is a fit, filter fits at the later slices, sawtooth model off): every slice's record but the check's
    own entry -- the first-slice fit, each filter fit, the scored states, the slice solve counts -- and the final psi
    are identical; the series totals differ by exactly the sum of the checks' solves; every slice carries a computed
    check (the first from its fit).
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

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def _fit(on):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon.current import sawtooth as SAW
    fm, kin, src, cfg, init, ip = _base._setup()
    ls_true = float(np.log(SAW.crash_strength(_base._F_TRUE, _base._RATE_TRUE)))
    tr = CM.consistent_solve(fm, kin, src, dict(log_strength=ls_true, f=_base._F_TRUE), cfg, label="truth",
                             initial_profiles=init)
    assert tr.converged, tr.status
    snap = fm.snapshot()
    gs = fm.gs
    geo = _base._sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    ms = _base._measurements(gs, geo, roles, ip, np.random.default_rng(_base._SEED))
    fm.reconfigure(ms, np.asarray(gs.trace_surf(0.999), float), roles, case_id="fit")
    n_before = fm.n_solves
    fr = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=cfg.r0), FIT.FitSettings(), start=snap,
                         label="fx", steady_state_check=on)
    rec = {k: v for k, v in fr.record.items() if k != "center"}
    return dict(status=fr.status, theta=fr.theta, sigma=fr.sigma, covariance=fr.covariance, chi2=fr.chi2,
                params=fr.params, record=rec, psi=np.asarray(gs.get_psi(False)), n_solves=fm.n_solves - n_before,
                history=fm.history, case_history=fm.case.history)


def _series(on):
    import dataclasses
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import series as SE
    import test_series_solver as TS
    fm, kin, src, cfg, init, ip = _base._setup()
    cfg = dataclasses.replace(cfg, core_relaxation="none")
    tr0 = CM.consistent_solve(fm, kin, src, dict(log_strength=None, a_corr=TS.A_TRUE), cfg, label="truth0",
                              initial_profiles=init)
    assert tr0.converged, tr0.status
    pr = tr0.assembly["profiles_rho"]
    j_true = (np.asarray(pr["rho"]), np.asarray(pr["jdotb"]))
    snap0 = fm.snapshot()
    geo = _base._sensors(fm.gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted"))
             for n in fm.gs.coil_sets}
    times = TS.TIMES[:3]
    bnd = np.asarray(fm.reference_boundary, float)
    gen = [SE.SliceData(time=t, measurements=None, reference_boundary=bnd, kinetics=kin, sources=src,
                        label=f"truth{k}", meta=dict(ip=ip * (1 + TS.RAMP * k), generation_coil_reg=1e-2))
           for k, t in enumerate(times)]
    meas, refs = [], []

    def cb(r):
        meas.append(_base._measurements(fm.gs, geo, roles, ip * (1 + TS.RAMP * len(meas)), None))
        refs.append(np.asarray(fm.gs.trace_surf(0.999), float))

    truth = SE.run_series(fm, gen, cfg, SE.SeriesSettings(initial="from_profile", initial_profile=j_true,
                                                          sawtooth="off", steady_state_check=False), callback=cb)
    assert truth["status"] == "completed", truth["status"]
    fm.restore(snap0)
    fm.reconfigure(meas[0], refs[0], roles, case_id="filtered")
    n_before = fm.n_solves
    sl = [SE.SliceData(time=t, measurements=meas[k], reference_boundary=refs[k], kinetics=kin, sources=src,
                       label=f"f{k}") for k, t in enumerate(times)]
    st = SE.SeriesSettings(initial="mse_fit", sawtooth="off", filter=True, steady_state_check=on)
    r = SE.run_series(fm, sl, cfg, st, initial_profiles=init)
    return dict(series=r, psi=np.asarray(fm.gs.get_psi(False)), n_solves=fm.n_solves - n_before)


def child(kind, on, out):
    res = _series(on) if kind == "series" else _fit(on)
    with open(out, "w") as fh:
        json.dump(_base._jsonable(res), fh)


def _run(tmp_path_factory, kind, on):
    d = tmp_path_factory.mktemp(f"steady_{kind}_{on}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), kind, "on" if on else "off", out], cwd=str(d),
                          env=env, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def _computed(chk):
    assert chk["computed"] is True, chk
    assert isinstance(chk["status"], str) and chk["n_solves"] > 0
    for key in ("own",) + (("steady_state",) if chk["steady_state"] is not None else ()):
        assert set(chk[key]) >= {"core", "mid", "outer", "all", "r_axis_m"}
    return int(chk["n_solves"])


def test_single_slice_fit_is_bitwise_identical_with_the_check_on_and_off(tmp_path_factory):
    from test_speed_parallel_solver import differences
    on, off = _run(tmp_path_factory, "fit", True), _run(tmp_path_factory, "fit", False)
    assert off["status"] in ("converged", "converged_within_noise"), off["status"]
    chk = on["record"].pop("steady_state_check")
    assert off["record"].pop("steady_state_check") == dict(computed=False,
                                                            reason="switched off (steady_state_check=False)")
    n_chk = _computed(chk)
    assert on["n_solves"] - off["n_solves"] == n_chk               # the counter is not reset; nothing is hidden
    assert on["history"][:len(off["history"])] == off["history"] and len(on["history"]) > len(off["history"])
    assert on["case_history"][:len(off["case_history"])] == off["case_history"]
    strip = ("n_solves", "history", "case_history")
    assert differences({k: v for k, v in on.items() if k not in strip},
                       {k: v for k, v in off.items() if k not in strip}) == []
    assert on["record"]["n_solves"] == off["record"]["n_solves"]  # the fit's own count


def test_series_of_three_slices_is_bitwise_identical_with_the_check_on_and_off(tmp_path_factory):
    from test_speed_parallel_solver import differences
    on, off = _run(tmp_path_factory, "series", True), _run(tmp_path_factory, "series", False)
    so, sf = on["series"], off["series"]
    assert sf["status"] == "completed", sf["status"]
    assert len(so["slices"]) == len(sf["slices"]) == 3
    assert so["slices"][0]["initial_mode"] == "mse_fit" and all(x.get("filter") for x in so["slices"][1:])
    n_chk = 0
    for x, y in zip(so["slices"], sf["slices"]):
        n_chk += _computed(x.pop("steady_state_check"))
        assert "steady_state_check" not in y
    assert so["settings"].pop("steady_state_check") is True and sf["settings"].pop("steady_state_check") is False
    assert so["n_solves"] - sf["n_solves"] == n_chk == on["n_solves"] - off["n_solves"]
    its_on, its_off = so.pop("solver_iterations"), sf.pop("solver_iterations")
    assert its_on["n_solves"] - its_off["n_solves"] == n_chk
    for d in (so, sf, on, off):
        d.pop("n_solves")
    assert differences(on, off) == []                              # every slice's record and the final psi


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2] == "on", sys.argv[3])
