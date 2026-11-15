"""Live-solver checks of the cut-cell setting and the opt-in taper (``pytest -m solver``), on whichever solver
build is loaded (with or without the setting).

Public synthetic fixture only (as ``test_fit_solver.py``), generation-mode forward solves with the g-file current
and a pressure whose gradient is finite at the boundary: ``p = p_gfile + c p_gfile(0) (1 - psi_N)`` with
``c = 0.2``, the g-file's own pressure (``c = 0``) as a control.  Checked:

(a) the setting is detected and recorded: on a build that provides it the forward model turns it on by default
    (value in effect True, read back before every solve); without it the value is None and nothing is set;
(b) the edge gradient case converges with the remedy :func:`test_fit_solver.edge_remedy_for_this_build` selects:
    on a build with the setting WITHOUT any taper (status ``converged``, no taper applied); on a build without it
    with the explicit taper (status ``converged_with_edge_taper``).  The untapered outcome on a build without the
    setting is recorded, not asserted;
(c) the off-switch ``lcfs_cut_cells="off"``: the value in effect is False on a build with the setting (None
    without), recorded in the settings record and in each solve record, and the control case still converges;
    and the value reaches the solver: on a build with the setting the edge gradient case, which converges with cut
    cells and no taper (b), does NOT converge under ``"off"`` (the observable behavior of the solver without the
    cut-cell quadrature; the recorded value alone is the Python-side setting, which the solver cannot report back).
    Skipped on a build without the setting, where ``"off"`` changes nothing that could be observed;
(d) the taper is off unless requested: the default forward settings never taper;
(e) ``q_90`` is q on the psi_N = 0.9 surface: on the converged control state it equals the package's q routine
    (TokaMaker ``get_q``) at psi_N 0.9 to round-off, differs from q at psi_N 0.1 and 0.95, lies between ``q_0``
    and ``q_95``, and ``get_q`` at psi_N 0.95 reproduces ``q_95`` to 1e-3 (relative; same trace, sign and
    convention).
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

EDGE_C = 0.2

pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(_base._OFT is None
                       or not all(os.path.isfile(p) for p in (_base._MESH, _base._GEQ, _base._IDA)),
                       reason="needs OpenFUSIONToolkit and the public synthetic fixtures"),
]


def _solve(fm, init, ip, c, label):
    xj, jj, pg = init
    r = fm.solve((xj, jj), (xj, pg + c * pg[0] * (1.0 - xj)), ip=ip, init=fm._snap0, label=label)
    att = r.record["policy"]["attempts"]
    return dict(status=r.status, converged=r.converged, taper_applied=r.record["policy"]["taper_applied"],
                solver=r.record["solver"], cut_at_solve=[a["info"].get("lcfs_cut_cells_at_solve") for a in att],
                cut_check=[a["info"].get("lcfs_cut_cells_check") for a in att],
                verdicts=[(a.get("detector") or {}).get("verdict") for a in att],
                q_90=r.stats.get("q_90"), q_95=r.stats.get("q_95"), q_0=r.stats.get("q_0"))


def child(out, kind):
    from bouquet.kinrecon import forward as FW
    rec = dict(kind=kind, build=FW.solver_build())
    if kind == "remedy":
        label, kw = _base.edge_remedy_for_this_build()
        rec["remedy"] = label
    elif kind == "default":
        kw = {}
    else:
        kw = dict(lcfs_cut_cells="off")
    fm, kin, src, cfg, init, ip = _base._setup(**kw)
    fm._snap0 = fm.snapshot()
    rec["forward_solver"] = fm.settings_record()["solver"]
    rec["forward_settings"] = fm.settings.as_dict()
    if kind == "off":
        rec["control"] = _solve(fm, init, ip, 0.0, "control")
        if rec["control"]["converged"]:
            # the package's q routine on the converged control state, at q_90's label and at two other labels
            qs = np.asarray(fm.gs.get_q(psi=np.array([0.1, FW.Q_90_PSIN, 0.95]))[1], float)
            rec["q_labels"] = dict(q_at_0p1=float(qs[0]), q_at_0p9=float(qs[1]), q_at_0p95=float(qs[2]),
                                   q_0=rec["control"].get("q_0"))
        rec["edge"] = _solve(fm, init, ip, EDGE_C, "edge_off")
    else:
        rec["edge"] = _solve(fm, init, ip, EDGE_C, "edge")
    with open(out, "w") as fh:
        json.dump(_base._jsonable(rec), fh, indent=1)


def _run(tmp_path_factory, kind):
    d = tmp_path_factory.mktemp(f"kinrecon_cut_{kind}")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, _base._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out, kind], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def remedy(tmp_path_factory):
    return _run(tmp_path_factory, "remedy")


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    return _run(tmp_path_factory, "default")


@pytest.fixture(scope="module")
def off(tmp_path_factory):
    return _run(tmp_path_factory, "off")


def test_setting_detected_and_recorded(default):
    provided = default["build"]["provides_lcfs_cut_cells"]
    c = default["forward_solver"]["lcfs_cut_cells"]
    assert c["provided"] is provided and c["option"] == "if_available"
    assert c["in_effect"] is (True if provided else None) and c["now"] is c["in_effect"]
    assert default["forward_solver"]["build"]["liboftpy_sha256"] == default["build"]["liboftpy_sha256"]
    e = default["edge"]
    assert e["solver"]["lcfs_cut_cells"] is c["in_effect"] and e["solver"]["provides_lcfs_cut_cells"] is provided
    assert all(v is c["in_effect"] for v in e["cut_at_solve"]) and set(e["cut_check"]) == {"PASS"}
    assert np.isfinite(e["q_90"]) if e["converged"] else True


def test_edge_gradient_case_converges_with_this_builds_remedy(remedy):
    e = remedy["edge"]
    if remedy["build"]["provides_lcfs_cut_cells"]:
        assert remedy["remedy"] == "cut_cells"
        assert e["status"] == "converged" and not e["taper_applied"], e
        assert e["solver"]["lcfs_cut_cells"] is True and e["solver"]["edge_taper_mode"] == "off"
    else:
        assert remedy["remedy"] == "taper_on"
        assert e["status"] == "converged_with_edge_taper" and e["taper_applied"], e
        assert e["solver"]["lcfs_cut_cells"] is None and e["solver"]["edge_taper_mode"] == "on"
    assert np.isfinite(e["q_90"]) and np.isfinite(e["q_95"])


def test_taper_off_unless_requested(default):
    assert default["forward_settings"]["edge_taper"]["mode"] == "off"
    assert not default["edge"]["taper_applied"] and default["edge"]["solver"]["edge_taper_mode"] == "off"
    if default["build"]["provides_lcfs_cut_cells"]:
        # a build with the setting needs no taper for this case
        assert default["edge"]["status"] == "converged", default["edge"]


def test_off_switch(off):
    provided = off["build"]["provides_lcfs_cut_cells"]
    c = off["forward_solver"]["lcfs_cut_cells"]
    assert off["forward_settings"]["lcfs_cut_cells"] == "off" and c["option"] == "off"
    assert c["in_effect"] is (False if provided else None) and c["requested"] is (False if provided else None)
    k = off["control"]
    assert k["status"] == "converged", k
    assert k["solver"]["lcfs_cut_cells"] is c["in_effect"] and k["solver"]["lcfs_cut_cells_option"] == "off"
    assert all(v is c["in_effect"] for v in k["cut_at_solve"])


def test_off_reaches_the_solver(off, default):
    if not off["build"]["provides_lcfs_cut_cells"]:
        pytest.skip("this solver build has no cut-cell setting: 'off' changes nothing observable on it")
    # with cut cells (default) the edge gradient case converges untapered; with the setting turned off it must not
    assert default["edge"]["status"] == "converged" and not default["edge"]["taper_applied"], default["edge"]
    e = off["edge"]
    assert not e["converged"] and not e["taper_applied"], e
    assert e["solver"]["lcfs_cut_cells"] is False and all(v is False for v in e["cut_at_solve"])


def test_q_90_is_q_at_psin_0p9(off):
    k, q = off["control"], off.get("q_labels")
    assert k["converged"] and q is not None, k
    assert k["q_90"] == pytest.approx(q["q_at_0p9"], rel=1e-12, abs=0.0)
    assert abs(k["q_90"] - q["q_at_0p1"]) > 1e-3 * abs(k["q_90"])
    assert abs(k["q_90"] - q["q_at_0p95"]) > 1e-3 * abs(k["q_90"])
    lo, hi = sorted((abs(k["q_0"]), abs(k["q_95"])))
    assert lo < abs(k["q_90"]) < hi
    assert q["q_at_0p95"] == pytest.approx(k["q_95"], rel=1e-3)


if __name__ == "__main__":
    child(sys.argv[1], sys.argv[2])
