"""The pressure factor ``s_press``: the pressure of the force balance is ``s_press`` times the kinetic pressure
(thermal + scaled fast-ion); the kinetic profiles, the bootstrap current and the conductivity do not follow it; the
factor is recorded with the assembly, the fit, the series slice and every written pressure.  No solver."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_correction_local_return as LR  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import pressure as P  # noqa: E402


def test_default_and_parameter_specification():
    assert CM.DEFAULT_PARAMS["s_press"] == 1.0
    spec = FIT.PARAMETERS["s_press"]
    assert spec["prior"] == (1.0, FIT.PRESSURE_FACTOR_SIGMA) and spec["bounds"] == (0.5, 2.0)
    assert FIT.FitSettings().free == ("a_corr", "s_press")       # free by default in a single-slice fit
    from bouquet.kinrecon import series as SE
    st = SE.SeriesSettings()                                     # the series' fits name their own free sets
    assert "s_press" not in st.initial_fit.free and "s_press" not in st.filter_fit.free
    assert FIT.FitSettings().max_step["s_press"] == 0.05


def test_the_record_names_what_is_scaled_and_what_is_not():
    r1 = CM.pressure_factor_record(1.0)
    assert r1["value"] == 1.0 and r1["applied"] is False
    r = CM.pressure_factor_record(1.07)
    assert r["value"] == 1.07 and r["applied"] is True
    assert "bootstrap current" in r["not_scaled"] and "T_e" in r["not_scaled"] and "n_e" in r["not_scaled"]
    assert "pressure of the force balance" in r["scaled"]
    assert "differs from the pressure of the kinetic profiles" in r["note"]
    # the flag: outside 1 +- 0.10 (provisional), a record that steers nothing
    assert CM.PRESSURE_FACTOR_FLAG_RANGE == (0.90, 1.10)
    assert r1["flagged"] is False and r["flagged"] is False and r["flag_range"] == [0.90, 1.10]
    assert "PROVISIONAL" in r["flag_note"]
    for v, flag in ((0.90, False), (1.10, False), (0.899, True), (1.101, True), (1.15, True), (0.85, True)):
        assert CM.pressure_factor_record(v)["flagged"] is flag, v
    assert "provisional" in FIT.PARAMETERS["s_press"]["note"]
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(CM.ModelFailure, match="pressure factor"):
            CM.pressure_factor_record(bad)


def test_the_assembly_scales_the_pressure_and_its_gradient_term_only():
    lag, kin, src = LR._fixture_inputs()
    # a thermal pressure with a nonzero boundary value and a fast-ion pressure with its own scale
    kin.thermal_pressure_at = lambda psi_n=None, rho=None: 2.0e4 * (1.0 - np.asarray(psi_n, float)) + 800.0
    src.p_fast_at = lambda x, rho=None: 3.0e3 * (1.0 - np.asarray(x, float)) ** 2
    cfg = CM.CurrentModelSettings(r0=LR.GEO.R0)
    a1 = CM.assemble(lag, kin, src, dict(s_pfast=1.2), cfg, LR.IP0)
    a2 = CM.assemble(lag, kin, src, dict(s_pfast=1.2, s_press=1.08), cfg, LR.IP0)
    x = np.asarray(a1["x"], float)
    np.testing.assert_allclose(a1["p"], 2.0e4 * (1.0 - x) + 800.0 + 1.2 * 3.0e3 * (1.0 - x) ** 2, rtol=1e-12)
    assert a1["pressure_factor"]["value"] == 1.0 and a1["pressure_factor"]["applied"] is False
    assert a2["pressure_factor"]["value"] == 1.08 and a2["pressure_factor"]["applied"] is True
    assert a2["params"]["s_press"] == 1.08
    np.testing.assert_array_equal(a1["x"], a2["x"])
    np.testing.assert_allclose(a2["p"], 1.08 * a1["p"], rtol=1e-12)
    np.testing.assert_allclose(a2["p_kinetic"], a1["p"], rtol=1e-12)        # the kinetic pressure is unchanged
    np.testing.assert_allclose(a1["p_kinetic"], a1["p"], rtol=0, atol=0)
    np.testing.assert_allclose(a2["jphi_pressure_term"], 1.08 * a1["jphi_pressure_term"], rtol=1e-10, atol=1e-6)
    # the parallel current (ohmic, bootstrap, driven) does not follow the factor
    np.testing.assert_allclose(a2["jdotb"], a1["jdotb"], rtol=1e-12)
    np.testing.assert_allclose(a2["jdotb_ni"], a1["jdotb_ni"], rtol=1e-12)
    for k in a1["currents_A"]:
        assert np.isclose(a2["currents_A"][k], a1["currents_A"][k], rtol=1e-12), k
    assert np.isclose(a2["p_sep"], 1.08 * a1["p_sep"]) and np.isclose(a2["pax"], 1.08 * a1["pax"])


def test_written_pressures_carry_the_factor():
    x = np.linspace(0.0, 1.0, 11)
    p = 1.0e4 * (1.0 - x) + 500.0
    blk = P.equilibrium_profiles_1d(x, p - 500.0, -1.0e4 * np.ones_like(x), 500.0, psi_axis=0.0, psi_boundary=1.0,
                                    dpsi_to_ids=1.0, pressure_factor=CM.pressure_factor_record(1.08))
    assert blk["meta"]["pressure_factor"] == 1.08
    blk = P.equilibrium_profiles_1d(x, p - 500.0, -1.0e4 * np.ones_like(x), 500.0, psi_axis=0.0, psi_boundary=1.0,
                                    dpsi_to_ids=1.0, pressure_factor=1.0)
    assert blk["meta"]["pressure_factor"] == 1.0
    blk = P.equilibrium_profiles_1d(x, p - 500.0, -1.0e4 * np.ones_like(x), 500.0, psi_axis=0.0, psi_boundary=1.0,
                                    dpsi_to_ids=1.0)
    assert blk["meta"]["pressure_factor"] is None                # not stated by the caller: never an assumed 1

    class _Eq:
        def save_eqdsk(self, filename, **kw):
            self.kw = kw

    rec = P.save_full_pressure_eqdsk(_Eq(), "unused.geqdsk", 500.0, pressure_factor=dict(value=1.08))
    assert rec["pressure_factor"] == 1.08
    assert P.save_full_pressure_eqdsk(_Eq(), "unused.geqdsk", 500.0)["pressure_factor"] is None
