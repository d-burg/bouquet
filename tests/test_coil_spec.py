"""chi^2 coil-drift metric (bouquet.coil_spec)."""
import numpy as np
import pytest

from bouquet.coil_spec import (MIN_ABS_MEASURED_A, coil_chi2,
                               coil_sigma_in_base_units)

BASE = {"F1A": -100000.0, "F7B": -330000.0, "ECOILA": -6000.0}
# measured circuit amps + absolute sigma (DIII-D-like: ~7 A on F, ~69 A on E)
MEAS = {"F1A": (-2000.0, 7.0), "F7B": (-5900.0, 7.0), "ECOILA": (-2400.0, 69.0)}


def _sig():
    return coil_sigma_in_base_units(BASE, MEAS)


class TestSigmaTransfer:
    def test_carries_fractional_precision_into_base_units(self):
        s = _sig()
        for k in BASE:
            assert s[k] / abs(BASE[k]) == pytest.approx(
                MEAS[k][1] / abs(MEAS[k][0]))

    def test_drops_coils_absent_from_the_measurement(self):
        s = coil_sigma_in_base_units({**BASE, "F9Z": 1.0}, MEAS)
        assert "F9Z" not in s

    def test_drops_coils_below_the_low_current_floor(self):
        meas = {**MEAS, "F1A": (0.5 * MIN_ABS_MEASURED_A, 7.0)}
        assert "F1A" not in coil_sigma_in_base_units(BASE, meas)

    def test_rejects_nonpositive_sigma(self):
        assert "F1A" not in coil_sigma_in_base_units(
            BASE, {**MEAS, "F1A": (-2000.0, 0.0)})


class TestChi2:
    def test_zero_drift_is_zero(self):
        r = coil_chi2(dict(BASE), BASE, _sig())
        assert r["chi2_nu"] == pytest.approx(0.0)
        assert r["nu"] == len(BASE)

    def test_one_sigma_everywhere_gives_unity(self):
        s = _sig()
        draw = {k: BASE[k] + s[k] for k in BASE}
        assert coil_chi2(draw, BASE, s)["chi2_nu"] == pytest.approx(1.0)

    def test_weights_by_precision_not_by_percentage(self):
        """Equal FRACTIONAL drift must not score equally: F7B is measured ~10x
        better than ECOILA, so the same 1% costs it far more chi^2."""
        s = _sig()
        z = {k: coil_chi2({**BASE, k: BASE[k] * 1.01}, BASE, s)["max_abs_z"]
             for k in ("F7B", "ECOILA")}
        assert z["F7B"] > 5.0 * z["ECOILA"]

    def test_invariant_under_a_shared_unit_rescale(self):
        """Baseline and draw in ampere-turns vs amps must score identically."""
        s = _sig()
        draw = {k: BASE[k] * 1.001 for k in BASE}
        a = coil_chi2(draw, BASE, s)["chi2_nu"]
        turns = 57.0
        base_t = {k: v * turns for k, v in BASE.items()}
        s_t = coil_sigma_in_base_units(base_t, MEAS)
        b = coil_chi2({k: v * turns for k, v in draw.items()}, base_t, s_t)["chi2_nu"]
        assert a == pytest.approx(b, rel=1e-9)

    def test_pooled_not_max(self):
        """One bad coil among many must not dominate the way `max` does."""
        s = _sig()
        draw = {**BASE, "F1A": BASE["F1A"] + 3.0 * s["F1A"]}
        r = coil_chi2(draw, BASE, s)
        assert r["max_abs_z"] == pytest.approx(3.0)
        assert r["chi2_nu"] == pytest.approx(9.0 / len(BASE))

    def test_reports_the_worst_coil(self):
        s = _sig()
        draw = {**BASE, "ECOILA": BASE["ECOILA"] + 5.0 * s["ECOILA"]}
        assert coil_chi2(draw, BASE, s)["worst_coil"] == "ECOILA"

    def test_no_usable_coil_is_nan_not_a_pass(self):
        r = coil_chi2({"X": 1.0}, {"X": 0.0}, {})
        assert r["nu"] == 0 and np.isnan(r["chi2_nu"])


class TestFixedSigma:
    """sigma held fixed across a discharge (bouquet.coil_spec.coil_sigma_fixed)."""

    @staticmethod
    def _samples(n=10, turns=55.0, sigma=7.0, i_meas=2000.0):
        return [(i_meas, turns * i_meas, sigma) for _ in range(n)]

    def test_recovers_the_conversion_factor(self):
        from bouquet.coil_spec import coil_sigma_fixed
        sig, fac = coil_sigma_fixed({"F7B": self._samples()})
        assert fac["F7B"] == pytest.approx(55.0)
        assert sig["F7B"] == pytest.approx(7.0 * 55.0)

    def test_immune_to_a_near_zero_crossing(self):
        """A slice whose measured current is small -- but still ABOVE the
        MIN_ABS_MEASURED_A floor -- must not inflate sigma.

        Real case: DIII-D 174823 F5B at t=5.706 s carries I_meas = -67.8 A
        (clears the 50 A floor) while its baseline is 2.3e4 A-t, so the
        per-slice rescale hands it sigma = 2426 A-t, ~4x its typical value.
        """
        from bouquet.coil_spec import (coil_sigma_fixed,
                                       coil_sigma_in_base_units)
        i_base = 55.0 * 2000.0
        rows = self._samples(n=9)
        rows.append((67.8, i_base, 7.0))                 # clears the 50 A floor
        fixed, _ = coil_sigma_fixed({"F5B": rows})
        per_slice = coil_sigma_in_base_units(
            {"F5B": i_base}, {"F5B": (67.8, 7.0)})
        assert "F5B" in per_slice, "the floor should NOT have excluded this coil"
        assert per_slice["F5B"] > 20.0 * fixed["F5B"]
        assert fixed["F5B"] == pytest.approx(7.0 * 55.0)

    def test_falls_back_when_few_slices_are_well_measured(self):
        from bouquet.coil_spec import coil_sigma_fixed
        rows = [(300.0, 55.0 * 300.0, 7.0)] * 4          # all below 1000 A
        sig, fac = coil_sigma_fixed({"E1": rows})
        assert fac["E1"] == pytest.approx(55.0)

    def test_drops_a_coil_with_no_usable_slice(self):
        from bouquet.coil_spec import coil_sigma_fixed
        sig, fac = coil_sigma_fixed({"E1": [(1.0, 100.0, 7.0)]})
        assert "E1" not in sig


def test_with_sigma_ref_removes_daq_epoch_dependence():
    """Same drift, 2012-epoch (8x) vs 2017-epoch sigma: chi2 differs by ~64x
    on the dd values and is identical once referenced to the fixed table."""
    from bouquet.coil_spec import (SIGMA_REF_D3D_A, coil_chi2,
                                   coil_sigma_in_base_units, with_sigma_ref)
    base = {"F1A": -86300.0, "ECOILA": 18370.0}
    draw = {"F1A": -86300.0 + 400.0, "ECOILA": 18370.0 + 200.0}
    meas_2017 = {"F1A": (-1609.0, 7.0), "ECOILA": (18842.0, 69.0)}
    meas_2012 = {"F1A": (-1609.0, 55.3), "ECOILA": (18842.0, 551.0)}
    c17 = coil_chi2(draw, base, coil_sigma_in_base_units(base, meas_2017))["chi2_nu"]
    c12 = coil_chi2(draw, base, coil_sigma_in_base_units(base, meas_2012))["chi2_nu"]
    assert c17 / c12 > 50.0
    r17 = coil_chi2(draw, base, coil_sigma_in_base_units(base, with_sigma_ref(meas_2017)))["chi2_nu"]
    r12 = coil_chi2(draw, base, coil_sigma_in_base_units(base, with_sigma_ref(meas_2012)))["chi2_nu"]
    assert r12 == pytest.approx(r17) == pytest.approx(c17)
    # coils with no family entry keep their own sigma
    out = with_sigma_ref({"F1A": (1.0, 55.0), "XYZ": (1.0, 3.0)}, {"F": 7.0})
    assert out["F1A"][1] == 7.0 and out["XYZ"][1] == 3.0
    assert SIGMA_REF_D3D_A == {"F": 7.0, "E": 69.0}


class TestEfitResidualSigma:
    def test_floor_plus_fraction_values(self):
        from bouquet.coil_spec import coil_sigma_efit_residual
        s = coil_sigma_efit_residual({"a": 20e3, "b": 100e3, "c": 270e3, "d": -270e3, "e": 0.0})
        assert s["a"] == pytest.approx(1065, rel=0.01)
        assert s["b"] == pytest.approx(1370, rel=0.01)
        assert s["c"] == pytest.approx(2598, rel=0.01)
        assert s["d"] == s["c"]                      # sign-blind
        assert s["e"] == pytest.approx(1050.0)       # floor at zero current

    def test_every_baseline_coil_gets_a_sigma(self):
        from bouquet.coil_spec import coil_sigma_efit_residual
        base = {"F5B": 127.8, "F6A": -266e3, "ECOILA": 18371.0}
        assert set(coil_sigma_efit_residual(base)) == set(base)

    def test_high_ip_fractional_drift_is_plausible_under_efit_sigma(self):
        """171317 H: 0.57 % drift on a 266 kA-t coil is ~4 sigma on the
        digitizer table and <1 sigma on the machine tolerance."""
        from bouquet.coil_spec import coil_chi2, coil_sigma_efit_residual
        base = {"F6A": -266479.0}; draw = {"F6A": -266479.0 - 1518.0}
        z_efit = coil_chi2(draw, base, coil_sigma_efit_residual(base))["max_abs_z"]
        z_digi = coil_chi2(draw, base, {"F6A": 7.0 * 55.0})["max_abs_z"]
        assert z_efit < 1.0 and z_digi > 3.5

    def test_filter_default_needs_no_dd_and_dd_modes_require_one(self):
        import inspect
        from bouquet.filtering import filter_coil_chi2
        sig = inspect.signature(filter_coil_chi2)
        assert sig.parameters["sigma_ref"].default is None
        assert sig.parameters["sigma"].default is None
        assert sig.parameters["dd_path"].default is None
        with pytest.raises(ValueError):
            filter_coil_chi2("nonexistent.h5", None, sigma_ref="d3d")



class TestDeviceRegistryAndResolution:
    D3D = {**{f"F{i}{s}": 1e5 for i in range(1, 10) for s in "AB"}, "ECOILA": 2e4, "ECOILB": 2e4}

    def test_detects_diiid_from_the_exact_coil_signature_only(self):
        from bouquet.devices import detect_device
        assert detect_device(self.D3D.keys()) == "DIII-D"
        assert detect_device(list(self.D3D)[:-1]) is None            # one coil missing
        assert detect_device(list(self.D3D) + ["PF7"]) is None        # one coil extra
        assert detect_device(["PF1", "PF2", "CS1"]) is None
        xia = list(self.D3D) + ["E567UP", "E567DN", "E89UP", "E89DN"]
        assert detect_device(xia) == "DIII-D"                          # finer DIII-D mesh
        assert detect_device(xia[:-1]) is None

    def test_explicit_floor_fraction_wins_over_device(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s, model = resolve_coil_sigma({"F6A": -270e3}, sigma={"floor": 500.0, "fraction": 0.01}, device="DIII-D")
        assert model["kind"] == "floor_fraction"
        assert s["F6A"] == pytest.approx((500.0**2 + 2700.0**2) ** 0.5)

    def test_per_coil_table_judges_only_named_coils(self):
        from bouquet.coil_spec import resolve_coil_sigma, CoilSigmaUnavailable
        s, model = resolve_coil_sigma({"F1A": 1.0, "F2A": 1.0}, sigma={"F1A": 300.0, "XYZ": 1.0})
        assert model["kind"] == "per_coil" and set(s) == {"F1A"}
        with pytest.raises(CoilSigmaUnavailable):
            resolve_coil_sigma({"F1A": 1.0}, sigma={"XYZ": 1.0})

    def test_callable(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s, model = resolve_coil_sigma({"F1A": 2.0}, sigma=lambda b: {k: 10.0 * abs(v) for k, v in b.items()})
        assert model["kind"] == "callable" and s["F1A"] == 20.0

    def test_device_model_named_or_detected(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s1, m1 = resolve_coil_sigma(self.D3D)                       # detected
        s2, m2 = resolve_coil_sigma(self.D3D, device="DIII-D")     # named
        assert m1["kind"] == m2["kind"] == "device" and m1["device"] == "DIII-D"
        assert s1 == s2 and "provenance" in m1
        assert s1["ECOILA"] == pytest.approx((325.0**2 + (0.0035 * 2e4) ** 2) ** 0.5)

    def test_unknown_device_is_loud_not_silent(self):
        from bouquet.coil_spec import resolve_coil_sigma, CoilSigmaUnavailable
        with pytest.raises(CoilSigmaUnavailable, match="BouquetConfig.device"):
            resolve_coil_sigma({"PF1": 1e5, "PF2": 1e5})
        with pytest.raises(KeyError, match="unknown device"):
            resolve_coil_sigma({"PF1": 1e5}, device="NOT-A-TOKAMAK")

    def test_config_validation(self):
        from bouquet.config import FilterConfig
        FilterConfig(coil_sigma={"floor": 1000.0, "fraction": 0.01})
        FilterConfig(coil_sigma={"F1A": 300.0})
        with pytest.raises(ValueError):
            FilterConfig(coil_sigma={"floor": -1.0, "fraction": 0.01})
        with pytest.raises(ValueError):
            FilterConfig(coil_sigma=3.0)


class TestEraTolerance:
    D3D = {**{f"F{i}{s}": 1e5 for i in range(1, 10) for s in "AB"}, "ECOILA": 2e4, "ECOILB": 2e4}

    def test_random_part_is_the_default_and_floor_follows_the_era(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s_new, m_new = resolve_coil_sigma(self.D3D, shot=204441)
        s_old, m_old = resolve_coil_sigma(self.D3D, shot=153072)
        s_unk, m_unk = resolve_coil_sigma(self.D3D)
        assert (m_new["floor"], m_new["fraction"]) == (325.0, 0.0035)
        assert (m_old["floor"], m_old["fraction"]) == (825.0, 0.0035)
        assert m_unk["floor"] == 325.0 and m_unk["shot"] is None
        assert s_old["F1A"] > s_new["F1A"]

    def test_named_model_selects_the_rms_option(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s, m = resolve_coil_sigma(self.D3D, sigma="rms_incl_offset", shot=204441)
        assert (m["floor"], m["fraction"]) == (1050.0, 0.0088) and m["model"] == "rms_incl_offset"
        with pytest.raises(KeyError, match="no sigma model"):
            resolve_coil_sigma(self.D3D, sigma="nonsense")

    def test_vsc_swing_rejected_under_random_tolerance_but_not_under_rms(self):
        """189392 F9A: 4.0 kA-t swing on 62.6 kA-t (legacy-rejected draw)."""
        from bouquet.coil_spec import coil_chi2, resolve_coil_sigma
        base = {"F9A": 62600.0}; draw = {"F9A": 62600.0 + 4001.0}
        z_rand = coil_chi2(draw, base, resolve_coil_sigma(base, device="DIII-D", shot=189392)[0])["max_abs_z"]
        z_rms = coil_chi2(draw, base, resolve_coil_sigma(base, device="DIII-D", sigma="rms_incl_offset")[0])["max_abs_z"]
        assert z_rand > 5.0 and z_rms < 4.0      # caught by the z_max=5 guard; not by the rms model


class TestPerCoilFloorsAndZGuard:
    D3D = {**{f"F{i}{s}": 1e5 for i in range(1, 10) for s in "AB"}, "ECOILA": 2e4, "ECOILB": 2e4}

    def test_per_coil_floor_applies_with_era_and_min_clip(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s, m = resolve_coil_sigma(self.D3D, device="DIII-D", shot=204441)
        assert m["era"] == "modern"
        assert s["F9A"] == pytest.approx((580.0**2 + (0.0035 * 1e5) ** 2) ** 0.5)   # table value
        assert s["F1A"] == pytest.approx((100.0**2 + (0.0035 * 1e5) ** 2) ** 0.5)   # 0 in table -> clipped to 100
        assert s["ECOILA"] == pytest.approx((325.0**2 + (0.0035 * 2e4) ** 2) ** 0.5)  # not in table -> era floor
        s2, m2 = resolve_coil_sigma(self.D3D, device="DIII-D", shot=153072)
        assert m2["era"] == "pre2014"
        assert s2["F1A"] > s["F1A"] and s2["ECOILA"] > s["ECOILA"]      # era floors differ
        assert abs(s2["F9A"] - s["F9A"]) < 0.05 * s["F9A"]                 # F9A: ~575 A-t in both eras

    def test_named_model_has_no_per_coil_table(self):
        from bouquet.coil_spec import resolve_coil_sigma
        s, m = resolve_coil_sigma(self.D3D, device="DIII-D", sigma="rms_incl_offset", shot=204441)
        assert m["per_coil_floors"] == {} and s["F9A"] == s["F1A"]

    def test_z_guard_catches_a_single_bad_coil(self, tmp_path):
        """chi2/nu can hide one coil at 7 sigma among 17 quiet ones; z_max cannot."""
        import h5py, numpy as np
        from bouquet.filtering import filter_coil_chi2
        names = list(self.D3D); base = np.array([self.D3D[n] for n in names])
        sig = {n: 1.0 for n in names}
        h5 = str(tmp_path / "t.h5")
        with h5py.File(h5, "w") as hf:
            g = hf.create_group("scan/1"); b = g.create_group("_baseline")
            b.create_dataset("coil_names", data=np.array(names, dtype="S")); b.create_dataset("coil_currents", data=base)
            d = g.create_group("0"); d.create_dataset("coil_names", data=np.array(names, dtype="S"))
            draw = base.copy(); draw[names.index("F9B")] += 7.0     # 7 sigma on one coil
            d.create_dataset("coil_currents", data=draw)
        r = filter_coil_chi2(h5, None, scan_key=1, apply=False, sigma=sig, z_max=False)
        assert r["draws"][0]["chi2_nu"] < 4.0 and r["draws"][0]["passes"] and r["z_max"] is None
        r = filter_coil_chi2(h5, None, scan_key=1, apply=False, sigma=sig)        # explicit sigma -> generic 4 / 5
        assert not r["draws"][0]["passes"] and r["z_max"] == 5.0 and r["chi2_max"] == 4.0
        assert r["sigma_model"]["acceptance"]["source"] == "generic"
        from bouquet.config import FilterConfig
        assert FilterConfig().z_max is None and FilterConfig().chi2_max is None

    def test_device_acceptance_is_the_calibrated_quantile(self, tmp_path):
        import h5py, numpy as np
        from bouquet.filtering import filter_coil_chi2
        from bouquet.devices import get_device
        names = list(self.D3D); base = np.array([self.D3D[n] for n in names])
        h5 = str(tmp_path / "d.h5")
        with h5py.File(h5, "w") as hf:
            g = hf.create_group("scan/1"); b = g.create_group("_baseline")
            b.create_dataset("coil_names", data=np.array(names, dtype="S")); b.create_dataset("coil_currents", data=base)
            d = g.create_group("0"); d.create_dataset("coil_names", data=np.array(names, dtype="S")); d.create_dataset("coil_currents", data=base)
        r = filter_coil_chi2(h5, None, scan_key=1, apply=False, shot=204441)
        acc = get_device("DIII-D").acceptance
        assert r["chi2_max"] == acc["chi2_max"] == 6.1 and r["z_max"] == acc["z_max"] == 6.3
        assert r["sigma_model"]["acceptance"]["source"] == "device q95"
        r2 = filter_coil_chi2(h5, None, scan_key=1, apply=False, shot=204441, chi2_max=4.0, z_max=5.0)
        assert r2["chi2_max"] == 4.0 and r2["z_max"] == 5.0
