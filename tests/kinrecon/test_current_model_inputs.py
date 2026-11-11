"""Input containers of ``bouquet.kinrecon.current_model`` (no OFT): units, frames, refusals."""
import numpy as np
import pytest

from bouquet.kinrecon.current.bootstrap import EC
from bouquet.kinrecon.current_model import KineticProfiles, SourceProfiles, full_params


def _kin(**kw):
    x = np.linspace(0.0, 1.1, 23)
    d = dict(ne=4e19 * (1.2 - 0.5 * x), te=2e3 * (1.2 - x) + 50, ni=3.4e19 * (1.2 - 0.5 * x),
             ti=1.5e3 * (1.2 - x) + 60, zeff=np.full_like(x, 1.8))
    d.update(kw)
    return KineticProfiles(x, d["ne"], d["te"], d["ni"], d["ti"], d["zeff"], nimp=d.get("nimp"),
                           p_thermal=d.get("p_thermal"))


def test_thermal_pressure_formula_and_impurities():
    k = _kin()
    x = np.array([0.0, 0.5, 1.0])
    d = k.at(x)
    np.testing.assert_allclose(k.thermal_pressure(x), EC * (d["ne"] * d["te"] + d["ni"] * d["ti"]), rtol=1e-14)
    xx = np.linspace(0.0, 1.1, 23)
    nimp = 1e18 * np.ones_like(xx)
    k2 = _kin(nimp=nimp)
    np.testing.assert_allclose(k2.thermal_pressure(x) - k.thermal_pressure(x), EC * 1e18 * d["ti"], rtol=1e-12)
    k3 = _kin(p_thermal=np.full_like(xx, 7.0))
    np.testing.assert_allclose(k3.thermal_pressure(x), 7.0)


def test_kinetic_profiles_refuse_bad_input():
    x = np.linspace(0.0, 0.9, 10)                        # does not reach the LCFS
    with pytest.raises(ValueError, match="cover"):
        KineticProfiles(x, *(np.ones(10) for _ in range(5)))
    x = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="positive"):
        KineticProfiles(x, np.ones(10), -np.ones(10), np.ones(10), np.ones(10), np.ones(10))


def test_source_forms_and_conversions():
    x = np.linspace(0.0, 1.0, 11)
    s = SourceProfiles.from_j_parallel(x, np.full(11, 2.0e5), B0=-2.0, p_fast=np.full(11, 1e3))
    np.testing.assert_allclose(s.jdotb_at([0.3]), [-4.0e5])
    assert "j_parallel" in s.convention["jdotb_cd"]
    # ONETWO form <J_phi R0/R> -> <J.B> = J <B^2> / (F R0 <1/R^2>) for a parallel current
    t = SourceProfiles(x, jtor_r0_cd=np.full(11, 1.0e5), r0_cd=1.7)
    b2, F, g1 = np.array([4.1]), np.array([3.4]), np.array([0.36])
    np.testing.assert_allclose(t.jdotb_at([0.5], b2=b2, fpol=F, inv_r2=g1), 1.0e5 * 4.1 / (3.4 * 1.7 * 0.36))
    with pytest.raises(ValueError):
        t.jdotb_at([0.5])
    with pytest.raises(ValueError):
        SourceProfiles(x, jdotb_cd=np.ones(11), jtor_r0_cd=np.ones(11), r0_cd=1.0)
    with pytest.raises(ValueError):
        SourceProfiles(x, p_fast=-np.ones(11))
    z = SourceProfiles(x)
    np.testing.assert_array_equal(z.p_fast_at([0.2, 0.4]), [0.0, 0.0])
    with pytest.raises(ValueError, match="cover"):
        SourceProfiles(np.linspace(0, 0.9, 5), p_fast=np.ones(5)).p_fast_at([0.95])


def test_parameter_defaults():
    p = full_params({"s_bs": 1.2})
    assert p["s_bs"] == 1.2 and p["log_strength"] is None and p["f"] == 0.37 and p["a_corr"] == (0.0, 0.0, 0.0)
