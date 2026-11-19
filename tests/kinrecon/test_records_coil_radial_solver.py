"""Live-solver tests (``pytest -m solver``) of two records that steer nothing, on the public synthetic fixture (as
``test_fit_solver.py``; one child process; package defaults otherwise).

(a) The vertical-stability coil (VSC) record (``ForwardModel.coil_current_record``): after a consistent solve in
    reconstruction mode (fitted coils with VSC-aware rows), the recorded VSC current equals the value the solver
    itself printed for that state -- column 7 of the last iteration line of the last captured solver output (the
    warm verification re-solve), parsed here independently of the package's parser -- and equals the solve record's
    verification entry; the totals are ``get_coil_currents``; own = total - gain x VSC and share / total follow
    exactly; without a solve record the VSC current is not stated.
(b) The radial-coordinate record (``ForwardModel.midplane_radius_record``) on a constructed displacement: a table of
    the solved state's own outboard-midplane radius against psi_N (psi_N at chosen radii on the line through the
    axis, from ``diagnostics.psi_n_at``, the same interpolator) shifted outward by d(psi) = 2 mm + 3 mm psi^2 gives
    ``dR = -d`` at every table point, the largest |dR| at the last table point inside the plasma, and ``-d(1) = -5 mm``
    at the edge; the unshifted table gives |dR| at round-off.  Bound 5e-3 mm, fixed before running: the record
    interpolates our radius linearly in sqrt(psi_N) on a 0.25 mm scan (error ~ (ds)^2/8 |R''(s)| ~ 1e-6 mm here) and
    the table's edge value linearly in psi_N between table points 0.25 mm apart (error <= h^2/8 |psi''/psi'| <= 1e-3
    mm for |psi''/psi'| <= 100 /m).  Without a table the record says "not supplied".
(c) The fit record carries both: a fit with no free parameter and the Ip target held (the shortest path through
    ``reconstruct``) records the VSC current of its final center's solve and the radial-coordinate record of the
    table on its kinetic inputs.
"""
import json
import os
import re
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
BOUND_MM = 5e-3


def _printed_vsc(text):
    """Column 7 of the last iteration line after the last solver banner (independent of forward.parse_history)."""
    block = text.split("Starting non-linear GS solver")[-1]
    rows = [ln.split() for ln in block.splitlines() if re.match(r"^\s*\d+(\s+[-+]?\d\.\d{4}E[-+]\d{2}){6}\s*$", ln)]
    return float(rows[-1][6])


def _shift(p):
    return 2e-3 + 3e-3 * p ** 2


def child(out):
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon import diagnostics as D
    from bouquet.kinrecon import fit as FIT
    from bouquet.kinrecon import forward as FW
    from bouquet.kinrecon.current import sawtooth as SAW
    fm, kin, src, cfg, init, ip = _base._setup()
    truth_p = dict(log_strength=float(np.log(SAW.crash_strength(_base._F_TRUE, _base._RATE_TRUE))), f=_base._F_TRUE)
    tr = CM.consistent_solve(fm, kin, src, truth_p, cfg, label="truth", initial_profiles=init)
    rec = dict(truth=tr.status)
    gs = fm.gs
    snap = fm.snapshot()
    geo = _base._sensors(gs)
    roles = {n: ("held" if n.startswith("E") else ("free" if n in ("F8A", "F8B") else "fitted")) for n in gs.coil_sets}
    clean = _base._measurements(gs, geo, roles, ip, None)
    fm.reconfigure(clean, np.asarray(gs.trace_surf(0.999), float), roles, case_id="records")
    # (a)
    FW.CAPTURE_LOG = []
    try:
        r = CM.consistent_solve(fm, kin, src, truth_p, cfg, start=snap, label="records")
        texts = [t for t in FW.CAPTURE_LOG if "Starting non-linear GS solver" in t]
    finally:
        FW.CAPTURE_LOG = None
    cc = fm.coil_current_record(r.solve)
    rec["a"] = dict(status=r.status, record=cc, printed=_printed_vsc(texts[-1]),
                    verification_vsc=r.solve.record["policy"]["verification"]["vsc_A"],
                    solve_vsc=r.solve.record["vsc_A"], gains=dict(fm.settings.vsc),
                    totals={str(k): float(v) for k, v in dict(gs.get_coil_currents()[0]).items()},
                    without_solve=fm.coil_current_record(None))
    # (b)
    ra, za = (float(v) for v in gs.o_point)
    rr = ra + np.arange(0.03, 0.8, 2.5e-4)
    pn = np.asarray(D.psi_n_at(gs, np.column_stack((rr, np.full_like(rr, za))))[0], float)
    n = int(np.argmax(pn > 1.01))
    tp, tr_ = pn[:n], rr[:n]
    rec["b"] = dict(monotone=bool(n > 2 and np.all(np.diff(tp) > 0)), psi=tp, d=_shift(tp),
                    shifted=fm.midplane_radius_record(dict(psi_n=tp, r_m=tr_ + _shift(tp), source="constructed")),
                    unshifted=fm.midplane_radius_record(dict(psi_n=tp, r_m=tr_, source="constructed")),
                    none=fm.midplane_radius_record(None))
    # (c)
    kin.midplane_radius = dict(psi_n=tp, r_m=tr_ + _shift(tp), source="constructed")
    fr = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=(), ip_target="measured"), params0=truth_p,
                         start=snap, label="rec", steady_state_check=False)
    R = fr.record
    rec["c"] = dict(status=fr.status, coil=R.get("coil_currents"), radial=R.get("radial_coordinate"),
                    center_vsc=((R.get("final_center") or {}).get("verification") or {}).get("vsc_A"))
    with open(out, "w") as fh:
        json.dump(_base._jsonable(rec), fh)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_records")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env, capture_output=True,
                          text=True, timeout=3600)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


def test_vsc_record_is_the_solvers_own_value(result):
    a = result["a"]
    assert result["truth"] == "consistent" and a["status"] == "consistent", (result["truth"], a["status"])
    c = a["record"]
    assert c["computed"] is True
    assert c["vsc_A"] == a["printed"] == a["verification_vsc"] == a["solve_vsc"]
    assert c["totals_A"] == a["totals"]
    assert set(c["vsc_coils"]) == {k for k, g in a["gains"].items() if g != 0.0}
    for name, e in c["vsc_coils"].items():
        g = a["gains"][name]
        assert e["gain"] == g and e["total_A"] == a["totals"][name]
        assert e["vsc_share_A"] == g * c["vsc_A"] and e["own_A"] == e["total_A"] - e["vsc_share_A"]
        assert e["share_of_total"] == e["vsc_share_A"] / e["total_A"]
    w = a["without_solve"]
    assert w["computed"] is True and w["vsc_A"] is None
    assert all(e["own_A"] is None for e in w["vsc_coils"].values())


def test_radial_coordinate_record_on_a_constructed_displacement(result):
    b = result["b"]
    assert b["monotone"]
    psi, d = np.asarray(b["psi"]), np.asarray(b["d"])
    inside = (psi >= 0.0) & (psi <= 1.0)
    s, u = b["shifted"], b["unshifted"]
    assert s["computed"] is True and u["computed"] is True, (s, u)
    assert s["n_points"] == int(inside.sum()) and np.array_equal(np.asarray(s["psi_n"]), psi[inside])
    assert np.max(np.abs(np.asarray(s["d_mm"]) + 1e3 * d[inside])) <= BOUND_MM
    assert s["max_psi_n"] == psi[inside][-1]
    assert abs(s["max_signed_mm"] + 1e3 * d[inside][-1]) <= BOUND_MM
    assert abs(s["edge_mm"] + 1e3 * _shift(1.0)) <= BOUND_MM
    assert u["max_abs_mm"] <= BOUND_MM and abs(u["edge_mm"]) <= BOUND_MM
    assert b["none"] == dict(supplied=False, note="not supplied")


def test_fit_record_carries_both_records(result):
    c = result["c"]
    assert c["status"] == "no_free_parameters", c["status"]
    assert c["coil"]["computed"] is True and c["coil"]["vsc_A"] == c["center_vsc"] is not None
    assert c["radial"]["supplied"] is True and c["radial"]["computed"] is True
    assert c["radial"]["source"] == "constructed"
    assert abs(c["radial"]["edge_mm"] + 1e3 * _shift(1.0)) <= BOUND_MM


if __name__ == "__main__":
    child(sys.argv[1])
