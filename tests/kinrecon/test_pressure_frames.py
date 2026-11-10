"""``bouquet.kinrecon.pressure``: separatrix pressure, the two pressure frames, delivery.

Synthetic only.  The solver-side g-file round trip is in ``test_pressure_frames_solver.py``.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import edge as E  # noqa: E402
from bouquet.kinrecon import pressure as P  # noqa: E402
from bouquet.kinrecon.score import SCORED_GROUPS  # noqa: E402


def _profile(n=101, p_sep=5.0e3, s=0.5):
    # full-frame p = 1e5 (1 - x^2) + p_sep; dp/dpsi_N = -2e5 x; P' = dp/dpsi = -2e5 x / s
    x = np.linspace(0.0, 1.0, n)
    p = 1.0e5 * (1.0 - x ** 2) + p_sep
    return x, p, -2.0e5 * x / s, s


def test_offset_keeps_pprime_and_has_no_inflation():
    x, p, pp, s = _profile()                      # p_sep = 5e3 = 4.8 % of p_axis
    r = P.solver_inputs(x, p, pprime=pp, psi_scale=s)
    assert r["pprime"].tobytes() == pp.tobytes() and r["pprime_source"] == "given (unchanged)"
    assert r["p_sep"] == 5.0e3 and r["p_sep_applied"] == 5.0e3 and r["pax"] == 1.0e5
    assert r["p_sep_over_p_axis"] == pytest.approx(5.0e3 / 1.05e5)
    # the solver's uniform rescale pax / (-s int P') is 1: linear P' integrates exactly
    assert r["implied_solver_rescale"] == pytest.approx(1.0, abs=1e-12)
    assert r["legacy_pprime_inflation"] == pytest.approx(1.05e5 / 1.0e5)
    leg = P.solver_inputs(x, p, pprime=pp, psi_scale=s, separatrix_pressure="legacy")
    assert leg["pax"] == 1.05e5 and leg["p_sep_applied"] == 0.0
    assert leg["implied_solver_rescale"] == pytest.approx(1.05, rel=1e-12)     # the inflation
    built = P.solver_inputs(x, p, psi_scale=s)                                 # PCHIP-built P'
    # PCHIP's one-sided end slopes differ from the analytic derivative near the axis by up
    # to 1e3 here (0.25 % of max |P'|); interior agreement is tighter
    np.testing.assert_allclose(built["pprime"], pp, rtol=0, atol=3e-3 * abs(pp).max())
    np.testing.assert_allclose(built["pprime"][5:], pp[5:], rtol=1e-2)


def test_refusals():
    x, p, pp, s = _profile()
    with pytest.raises(P.NegativeSeparatrixPressure):
        P.solver_inputs(x, p - 1.0e4, pprime=pp)
    with pytest.raises(ValueError):
        P.solver_inputs(x, np.full_like(x, 3.0), pprime=pp)           # pax = 0
    with pytest.raises(ValueError):
        P.solver_inputs(x, p, pprime=pp, separatrix_pressure="bogus")
    with pytest.raises(ValueError):
        P.solver_inputs(x, p)                                           # neither pprime nor psi_scale
    assert P.solver_pax(np.array([10.0, -1.0]), "legacy") == 10.0      # legacy never reads the edge


def test_pressure_frames_formulas():
    st = dict(vol=20.0, W_MHD=3.0e5, beta_pol=1.2, beta_tor=2.0, beta_n=1.8, P_ax=1.0e5)
    f = P.pressure_frames(st, 5.0e3)
    pvol = 2.0e5
    fac = 1.0 + 5.0e3 * 20.0 / pvol
    assert f["factor"] == pytest.approx(fac)
    assert f["full"]["W_MHD"] == pytest.approx(3.0e5 + 1.5 * 5.0e3 * 20.0)
    for k in ("beta_pol", "beta_tor", "beta_n"):
        assert f["full"][k] == pytest.approx(st[k] * fac)
    assert f["full"]["P_ax"] == 1.05e5 and f["solver"]["W_MHD"] == 3.0e5
    z = P.pressure_frames(st, 0.0)
    assert z["factor"] == 1.0 and z["full"] == z["solver"]
    with pytest.raises(ValueError):
        P.pressure_frames(dict(vol=1.0, W_MHD=0.0), 1.0)


def test_profiles_1d_writer_round_trip(tmp_path):
    x, p, pp, s = _profile()
    p_sep = P.separatrix_pressure_of(p)
    blk = P.equilibrium_profiles_1d(x, p - p_sep, pp, p_sep, psi_axis=0.0, psi_boundary=s,
                                    dpsi_to_ids=-2.0 * np.pi)
    path = tmp_path / "eq.json"
    P.write_profiles_1d_json(path, blk)
    back = P.read_profiles_1d_json(path)
    np.testing.assert_allclose(back["pressure"], p, rtol=1e-15)        # full frame, p_sep preserved
    assert back["pressure"][-1] == p_sep and back["meta"]["p_sep"] == p_sep
    assert back["meta"]["pressure_frame"] == "full"
    # the written pressure differentiates to the written dpressure_dpsi (linear P': exact)
    assert P.check_consistency(back["psi"], back["pressure"], back["dpressure_dpsi"]) < 1e-12
    with pytest.raises(ValueError):
        P.equilibrium_profiles_1d(x, p - p_sep, pp, p_sep, 0.0, s)       # no flux convention stated


@pytest.mark.parametrize("variant,kw", [("edge_pinned", {}), ("last_node", {}),
                                        ("pressure_conserving", {"band_width": 0.02})])
def test_taper_with_nonzero_p_sep(variant, kw):
    x, p, pp, s = _profile(n=257)
    p_sep = p[-1]
    spec = E.EdgeTaperSpec(variant=variant, **kw)
    xt, ppt, pax, rec = P.solver_inputs_with_taper(x, p, pp, s, spec)
    assert rec["p_sep"] == p_sep and rec["p_sep_after_taper"] == p_sep and rec["taper_variant"] == variant
    t = rec["taper"]
    assert pax == pytest.approx(rec["pax_untapered"] * t["pax_ratio"], rel=1e-15)
    if variant == "pressure_conserving":
        assert abs(t["dp_inside"]) < 1e-6 * p[0] and pax == pytest.approx(1.0e5, rel=1e-12)
    else:
        # full-frame interior drops by one constant; p_sep unchanged
        assert t["dp_inside"] < 0 and pax < 1.0e5
        assert t["dp_inside"] == pytest.approx(-(1.0e5 - pax), rel=1e-9)
    assert rec["frames"]["pax"] == "solver" and rec["frames"]["p_sep"] == "full"


def test_nothing_scored_is_pressure_like():
    # kinrecon scores no pressure / beta / stored-energy constraint; Thomson is n_e / T_e.
    # A future pressure-like group must compare in the full frame (pressure.pressure_frames).
    bad = [g for g in SCORED_GROUPS if any(w in g for w in ("pressure", "beta", "w_mhd", "energy"))]
    assert not bad
    assert set(SCORED_GROUPS) >= {"thomson_ne", "thomson_te"}
