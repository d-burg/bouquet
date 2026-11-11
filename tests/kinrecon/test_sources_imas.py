"""IMAS provider: round trip, sign and pressure-component conventions, refusals (synthetic IDS only)."""
import copy

import numpy as np
import pytest

from bouquet.kinrecon.sources import AmbiguousConventionError, NonInductiveSources, Provenance, from_imas, to_imas

RHO = np.linspace(0.0, 1.0, 41)


def _sources():
    prov = Provenance(provider="test", source_code="synthetic", source_version="0", inputs={}, conventions={},
                      settings={}, assumptions=["a"], limitations=["l"])
    p_par = 2e4 * (1 - RHO ** 2) ** 2
    p_perp = 1e4 * (1 - RHO ** 2) ** 2
    return NonInductiveSources(rho=RHO, time=2.5, time_window=0.0, B0=1.8, R0=1.7,
                               jdotb={"nbi_a": 3e5 * np.exp(-(RHO / 0.3) ** 2), "ec_1": 1e5 * np.exp(-((RHO - .4) / .05) ** 2)},
                               p_fast={"D": (p_par + 2 * p_perp) / 3}, p_par={"D": p_par}, p_perp={"D": p_perp},
                               n_fast={"D": 2e18 * (1 - RHO ** 2)}, provenance=prov)


@pytest.mark.parametrize("ip_sign,bt_sign", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_round_trip_all_sign_combinations(ip_sign, bt_sign):
    s = _sources()
    dd = to_imas(s, ip_sign=ip_sign, bt_sign=bt_sign)
    # co-Ip drive: j_parallel = <J.B>/B0_signed has the sign of Ip (DD definition)
    assert np.sign(np.max(dd["core_sources"]["source"][0]["profiles_1d"][0]["j_parallel"])) == ip_sign
    r = from_imas(dd, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    for k in s.jdotb:
        np.testing.assert_allclose(r.jdotb[k], s.jdotb[k], rtol=1e-12)
    np.testing.assert_allclose(r.p_fast["D"], s.p_fast["D"], rtol=1e-12)
    np.testing.assert_allclose(r.p_par["D"], s.p_par["D"], rtol=1e-12)
    np.testing.assert_allclose(r.n_fast["D"], s.n_fast["D"], rtol=1e-12)
    assert r.B0 == pytest.approx(1.8) and r.R0 == pytest.approx(1.7)


def test_true_pressure_declaration_changes_by_factor_three():
    dd = to_imas(_sources(), ip_sign=1, bt_sign=1)
    a = from_imas(dd, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    b = from_imas(dd, 2.5, RHO, fast_pressure_components="true_pressures", max_time_offset=1e-6)
    np.testing.assert_allclose(a.p_fast["D"], 3 * b.p_fast["D"], rtol=1e-12)


def test_refusals():
    dd = to_imas(_sources(), ip_sign=1, bt_sign=1)
    with pytest.raises(AmbiguousConventionError):
        from_imas(dd, 2.5, RHO, fast_pressure_components=None, max_time_offset=1e-6)
    with pytest.raises(AmbiguousConventionError):                      # nearest slice without a tolerance
        from_imas(dd, 2.5, RHO, fast_pressure_components="imas_one_third")
    with pytest.raises(ValueError):                                    # too far from the only slice
        from_imas(dd, 3.0, RHO, fast_pressure_components="imas_one_third", max_time_offset=0.01)
    d2 = copy.deepcopy(dd)
    d2["core_sources"]["source"].append({"identifier": {"index": 999, "name": "mystery"},
                                         "profiles_1d": [{"time": 2.5, "grid": {"rho_tor_norm": RHO.tolist()},
                                                          "j_parallel": np.ones(41).tolist()}]})
    with pytest.raises(AmbiguousConventionError):
        from_imas(d2, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    r = from_imas(d2, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6,
                  classify={999: "exclude"})
    assert "mystery" not in r.jdotb
    d3 = copy.deepcopy(dd)
    del d3["core_sources"]["vacuum_toroidal_field"]["b0"]
    with pytest.raises(AmbiguousConventionError):
        from_imas(d3, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    d4 = copy.deepcopy(dd)
    del d4["core_profiles"]["global_quantities"]
    with pytest.raises(AmbiguousConventionError):
        from_imas(d4, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    with pytest.raises(ValueError):                                    # beyond the producer's grid
        from_imas(dd, 2.5, np.linspace(0, 1.1, 5), fast_pressure_components="imas_one_third", max_time_offset=1e-6)


def test_bootstrap_and_ohmic_excluded_and_window_mean():
    dd = to_imas(_sources(), ip_sign=1, bt_sign=1)
    grid = {"rho_tor_norm": RHO.tolist()}
    for idx, name in ((13, "bootstrap_current"), (7, "ohmic"), (1, "total")):
        dd["core_sources"]["source"].append({"identifier": {"index": idx, "name": name},
                                             "profiles_1d": [{"time": 2.5, "grid": grid,
                                                              "j_parallel": (1e6 * np.ones(41)).tolist()}]})
    # a second NB slice at t = 2.6 with double current: window mean = 1.5x
    src0 = dd["core_sources"]["source"][0]
    src0["profiles_1d"].append(dict(src0["profiles_1d"][0], time=2.6,
                                    j_parallel=(2 * np.asarray(src0["profiles_1d"][0]["j_parallel"])).tolist()))
    r = from_imas(dd, 2.55, RHO, fast_pressure_components="imas_one_third", time_window=0.2)
    assert set(r.jdotb) == {"nbi_a", "ec_1"}
    s = _sources()
    np.testing.assert_allclose(r.jdotb["nbi_a"], 1.5 * s.jdotb["nbi_a"], rtol=1e-12)
    assert "bootstrap_current (index 13)" in r.provenance.inputs["sources_excluded"]
