"""A kinetic sigma larger than the profile it perturbs is REPORTED at
baseline time -- one line, nothing clipped, the sampling untouched.

A Gaussian draw with sigma > value goes non-positive; the draw is then
rejected (``kinetics_nonphysical``) and the batch yields little.  The report
names the channel, the fraction of the radius and the largest ratio.
Synthetic inputs only; no solver.
"""
import numpy as np
import pytest

from bouquet.baseline import (SIGMA_EXCEEDS_PROFILE_MIN_FRACTION,
                              sigma_exceeds_profile,
                              sigma_exceeds_profile_line)

X = np.linspace(0.0, 1.0, 101)
PROF = dict(ne=5e19 * (1 - 0.8 * X ** 2), te=2e3 * (1 - 0.9 * X ** 2) + 50,
            ni=4e19 * (1 - 0.8 * X ** 2), ti=3e3 * (1 - 0.9 * X ** 2) + 50)


def _sig(frac):
    return {k: frac * v for k, v in PROF.items()}


def test_ordinary_sigmas_report_nothing():
    assert sigma_exceeds_profile(X, PROF, _sig(0.2)) == []
    # sigma == value is not "exceeds"
    assert sigma_exceeds_profile(X, PROF, _sig(1.0)) == []


def test_a_sigma_above_the_profile_everywhere_is_reported():
    s = _sig(0.1)
    s["ti"] = 2.3 * PROF["ti"]
    s["ti"][0] = 6.1 * PROF["ti"][0]
    rec = sigma_exceeds_profile(X, PROF, s)
    assert [r["channel"] for r in rec] == ["ti"]
    r = rec[0]
    assert r["fraction"] == pytest.approx(1.0)
    assert r["psi_N_range"] == [0.0, 1.0]
    assert r["max_ratio"] == pytest.approx(6.1)
    assert r["max_ratio_psi_N"] == 0.0
    line = sigma_exceeds_profile_line(rec)
    assert "\n" not in line
    assert "sigma_ti > ti over 100% of psi_N" in line
    assert "kinetics_nonphysical" in line and "nothing is clipped" in line


def test_the_stated_fraction_is_the_threshold():
    f = SIGMA_EXCEEDS_PROFILE_MIN_FRACTION
    assert 0.0 < f < 1.0
    s = _sig(0.1)
    # exceeds over ~2 % of the radius: below the threshold, not reported
    s["te"] = np.where(X >= 0.985, 1.5, 0.1) * PROF["te"]
    assert sigma_exceeds_profile(X, PROF, s) == []
    # ... over ~10 %: reported, with the range
    s["te"] = np.where(X >= 0.90, 1.5, 0.1) * PROF["te"]
    rec = sigma_exceeds_profile(X, PROF, s)
    assert [r["channel"] for r in rec] == ["te"]
    assert rec[0]["fraction"] == pytest.approx(0.105, abs=0.01)
    assert rec[0]["psi_N_range"][0] == pytest.approx(0.90)
    assert rec[0]["min_fraction"] == f


def test_the_fraction_is_of_the_radius_not_of_the_node_count():
    """A grid dense near the axis: half the NODES sit inside psi_N < 0.25."""
    x = np.linspace(0.0, 1.0, 101) ** 2
    p = dict(ti=1e3 * (1 - 0.9 * x) + 50)
    s = dict(ti=np.where(x < 0.04, 3.0, 0.1) * p["ti"])     # 20 % of nodes
    assert sigma_exceeds_profile(x, p, s) == []             # 4 % of psi_N
    s = dict(ti=np.where(x < 0.25, 3.0, 0.1) * p["ti"])
    rec = sigma_exceeds_profile(x, p, s)
    assert rec and rec[0]["fraction"] == pytest.approx(0.25, abs=0.02)


def test_nothing_is_changed():
    s = _sig(3.0)
    before = {k: v.copy() for k, v in s.items()}
    pb = {k: v.copy() for k, v in PROF.items()}
    sigma_exceeds_profile(X, PROF, s)
    for k in s:
        np.testing.assert_array_equal(s[k], before[k])
        np.testing.assert_array_equal(PROF[k], pb[k])


def test_resolve_uncertainty_reports_and_returns_the_same_sigmas(
        capsys, tmp_path):
    from bouquet.baseline import Baseline, resolve_uncertainty
    from bouquet.config import (BouquetConfig, ReconstructionSource,
                                SolverConfig, UncertaintyConfig)
    j = 1e6 * (1 - X ** 2)
    bl = Baseline(psi_N=X, j_phi=j, j_inductive=0.85 * j, j_BS=0.15 * j,
                  psi_N_kinetic=X, Zeff=np.full(X.size, 1.8),
                  Ip_target=1.2e6, l_i_target=0.85,
                  provenance="reconstruction", **PROF)

    def env(ti_sigma):
        unc = UncertaintyConfig(
            sigma_profiles=dict(ne=0.1 * PROF["ne"], te=0.1 * PROF["te"],
                                ni=0.1 * PROF["ni"], ti=ti_sigma))
        cfg = BouquetConfig(
            source=ReconstructionSource(
                geqdsk_path=str(tmp_path / "g.geqdsk"),
                profiles_path=str(tmp_path / "p.peqdsk"), time=3.0),
            solver=SolverConfig(mesh_path=str(tmp_path / "mesh.h5")),
            output_header=str(tmp_path / "out"), uncertainty=unc)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return resolve_uncertainty(cfg, bl)

    big = 2.0 * PROF["ti"]
    out = env(big)
    txt = capsys.readouterr().out
    assert txt.count("[sigma-check] WARNING") == 1
    assert [r["channel"] for r in out["sigma_exceeds_profile"]] == ["ti"]
    # report only: the resolved sigma is the input, untouched
    np.testing.assert_array_equal(out["sigma_ti"], big)
    out = env(0.1 * PROF["ti"])
    assert "[sigma-check]" not in capsys.readouterr().out
    assert out["sigma_exceeds_profile"] == []
