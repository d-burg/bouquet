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
