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
                   back_psi=back["psi"].tolist(), back_p=back["pressure"].tolist(),
                   back_dp=back["dpressure_dpsi"].tolist(),
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
    # Bound derived from the g-file's storage: PRES is written in Fortran (5e16.9), i.e. v = m 10^e with a
    # 9-digit mantissa m in [0.1, 1) and e = floor(log10|v|) + 1, so each stored value is within half a unit of
    # its 9th significant digit, h(v) = 0.5 10^(e - 9), of the double it came from.  The full write's edge value
    # is the bare one plus p_sep (p_sep exact in double), so the two stored values can differ from
    # stored_bare + p_sep by at most h(full) + h(bare), plus the double-precision rounding of the sum
    # (2 eps |full|, negligible).  That sum is the bound.
    def _half_unit_9(v):
        return 0.5 * 10.0 ** (np.floor(np.log10(abs(v))) + 1 - 9)
    full, bare = d["pres_edge_full"], d["pres_edge_bare"]
    bound = _half_unit_9(full) + _half_unit_9(bare) + 2 * np.finfo(float).eps * abs(full)
    assert abs(full - (bare + p_sep)) <= bound, (full - (bare + p_sep), bound)
    assert d["rec"]["pressure_frame"] == "full" and d["rec"]["p_sep"] == p_sep
    assert d["back_edge"] == pytest.approx(p_sep + d["solver_p_edge"], rel=1e-12)
    assert d["back_meta"]["pressure_frame"] == "full"
    assert d["W_full"] == pytest.approx(d["W_solver"] + 1.5 * p_sep * d["vol"], rel=1e-12)
    # get_profiles' P' and P are mutually consistent to the trapezoid on 129 points.  Bound derived from the
    # profile representation: the solve's P' is piecewise LINEAR in psi_N on the 50 uniform knots of
    # test_score_solver._build_equilibrium (checked below: linear between knots to round-off), so the trapezoid
    # rule on the 129 samples is exact on every sample interval without a knot inside, and on an interval
    # [a, b] with a knot xi inside (slope jump ds of P') it errs by ds (xi - a)(b - xi) / 2.  The check's
    # cumulative error is at most the sum of these over all knots, relative to max |p|.
    psi, pr, dp = (np.asarray(d[k], float) for k in ("back_psi", "back_p", "back_dp"))
    span = psi[-1] - psi[0]
    xs = (psi - psi[0]) / span
    knots = np.linspace(0.0, 1.0, 50)
    slopes = []
    for j in range(knots.size - 1):
        i = np.nonzero((xs >= knots[j] - 1e-12) & (xs <= knots[j + 1] + 1e-12))[0]
        sl = (dp[i[-1]] - dp[i[0]]) / (psi[i[-1]] - psi[i[0]])
        assert np.max(np.abs(dp[i[0]] + sl * (psi[i] - psi[i[0]]) - dp[i])) <= 1e-12 * np.max(np.abs(dp))
        slopes.append(sl)
    err = 0.0
    for j in range(1, knots.size - 1):
        k = int(np.searchsorted(xs, knots[j])) - 1
        a, b = xs[k], xs[k + 1]
        if a + 1e-12 < knots[j] < b - 1e-12:
            err += abs(slopes[j] - slopes[j - 1]) * (knots[j] - a) * (b - knots[j]) * span ** 2 / 2
    assert d["consistency"] <= err / np.max(np.abs(pr)), (d["consistency"], err / np.max(np.abs(pr)))
