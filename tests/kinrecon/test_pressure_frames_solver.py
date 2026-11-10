"""Live-solver round trip of ``bouquet.kinrecon.pressure`` delivery (``pytest -m solver``).

The public synthetic equilibrium of ``test_score_solver.py`` is solved (OFT defaults) in a
child interpreter; a separatrix pressure of 5 % of the axis target is then attached at
delivery: the g-file written by ``save_full_pressure_eqdsk`` must carry PRES = the bare
write's PRES + p_sep (PPRIME unchanged), and the IMAS profiles_1d block built from
``get_profiles`` must be in the full frame and round-trip through its JSON writer.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import test_score_solver as TS  # noqa: E402

pytestmark = TS.pytestmark

_CHILD = textwrap.dedent('''
    import importlib.util, json, sys
    import numpy as np
    sys.path.insert(0, {repo!r})
    spec = importlib.util.spec_from_file_location("ts", {ts!r})
    ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
    from bouquet.kinrecon import pressure as P
    gs, g = ts._build_equilibrium()
    st = gs.get_stats()
    p_sep = 0.05 * float(st["P_ax"])
    fa, fb = {out!r} + "/full.geqdsk", {out!r} + "/bare.geqdsk"
    rec = P.save_full_pressure_eqdsk(gs, fa, p_sep, nr=65, nz=65)
    gs.save_eqdsk(fb, nr=65, nz=65)
    a, b = read_eqdsk(fa), read_eqdsk(fb)
    x = np.linspace(0.0, 1.0, 129)
    psi_n, f, fp, p, pp = gs.get_profiles(psi=x.copy())
    pb = gs.get_psi(False)
    blk = P.equilibrium_profiles_1d(psi_n, p, pp, p_sep, psi_axis=float(gs.psi_bounds[1]),
                                    psi_boundary=float(gs.psi_bounds[0]), dpsi_to_ids=1.0)
    P.write_profiles_1d_json({out!r} + "/eq.json", blk)
    back = P.read_profiles_1d_json({out!r} + "/eq.json")
    fr = P.pressure_frames(st, p_sep)
    json.dump(dict(rec=rec, dpres=(np.asarray(a["pres"]) - np.asarray(b["pres"])).tolist(),
                   pprime_equal=bool(np.array_equal(a["pprime"], b["pprime"])),
                   pres_edge_full=float(a["pres"][-1]), pres_edge_bare=float(b["pres"][-1]),
                   p_sep=p_sep, solver_p_edge=float(p[-1]), back_edge=float(back["pressure"][-1]),
                   back_meta=back["meta"], consistency=P.check_consistency(back["psi"], back["pressure"],
                                                                           back["dpressure_dpsi"]),
                   W_solver=fr["solver"]["W_MHD"], W_full=fr["full"]["W_MHD"], vol=fr["volume"]),
              open({out!r} + "/out.json", "w"))
''')


def test_full_pressure_delivery_round_trip(tmp_path):
    code = _CHILD.format(repo=_REPO, ts=os.path.join(_HERE, "test_score_solver.py"), out=str(tmp_path))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (_REPO, TS._OFT, os.environ.get("PYTHONPATH", "")) if p))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.load(open(tmp_path / "out.json"))
    p_sep = d["p_sep"]
    np.testing.assert_allclose(d["dpres"], p_sep, rtol=1e-6)            # PRES(full) = PRES(bare) + p_sep
    assert d["pprime_equal"]
    # the g-file's last point is psi_N = 1 - lcfs_pad (0.99 by default), so the bare write's
    # edge PRES is the solver pressure there (not 0); the full write adds exactly p_sep
    assert d["pres_edge_full"] == pytest.approx(d["pres_edge_bare"] + p_sep, rel=1e-8)   # g-file E16.9: 9 digits
    assert d["rec"]["pressure_frame"] == "full" and d["rec"]["p_sep"] == p_sep
    assert d["back_edge"] == pytest.approx(p_sep + d["solver_p_edge"], rel=1e-12)
    assert d["back_meta"]["pressure_frame"] == "full"
    assert d["W_full"] == pytest.approx(d["W_solver"] + 1.5 * p_sep * d["vol"], rel=1e-12)
    # get_profiles' P' and P are mutually consistent to the trapezoid on 129 points
    assert d["consistency"] < 1e-2
