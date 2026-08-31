"""Measured Z_eff uncertainty: reader tiers + the envelope ladder.

The IDA files carry a measured Zeff uncertainty in BOTH modern vintages --
posterior samples (ensemble layout) and a ``Zeff_err`` dataset (newer direct
layout) -- which the pipeline previously discarded in favour of an assumed
5 % scalar (measured values run ~8-9 % median in-core).  The reader now
reports the highest-fidelity tier the file can support, and
``resolve_zeff_envelope`` picks it with explicit provenance:

    measured IDA sigma_Zeff  >  zeff_scalar_sigma * |Zeff|

with the measured tier eligible only when the Z_eff baseline itself is the
IDA one (recon path) -- a FUSE baseline (IMAS/ida_hybrid) must not be paired
with an IDA envelope.

Synthetic .cdf-shaped HDF5 files exercise all three vintages; no solver.
"""
import warnings

import h5py
import numpy as np
import pytest

from bouquet.baseline import resolve_zeff_envelope
from bouquet.io.ida import read_ida

_NPSI = 40


def _grids():
    psi = np.linspace(0.0, 1.2, _NPSI)
    ne = 4e19 * (1.0 - 0.7 * np.clip(psi, 0, 1) ** 2) + 1e18
    te = 2e3 * (1.0 - 0.8 * np.clip(psi, 0, 1) ** 2) + 30.0
    zeff = 1.8 + 0.3 * psi
    return psi, ne, te, zeff


def _write_direct(path, with_zeff_err, with_carbon=True):
    psi, ne, te, zeff = _grids()
    n_c = (zeff - 1.0) * ne / 30.0          # exactly consistent carbon
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])       # ms
        f["psi_n"] = psi
        for k, v in (("n_e", ne), ("T_e", te), ("T_12C6", te * 0.9)):
            f[k] = v[None, :]
            f[k + "_err"] = (0.05 * v)[None, :]
        f["Zeff"] = zeff[None, :]
        if with_zeff_err:
            f["Zeff_err"] = (0.09 * zeff)[None, :]     # 9 % measured
        if with_carbon:
            f["n_12C6"] = n_c[None, :]
            f["n_12C6_err"] = (0.25 * n_c)[None, :]
    return psi, zeff


def _write_ensemble(path, spread=0.08, nsamp=64):
    psi, ne, te, zeff = _grids()
    rng = np.random.default_rng(7)
    def samp(v, frac):
        return v[None, None, :] * (1.0 + frac * rng.standard_normal(
            (1, nsamp, 1)))
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])
        f["psi_n"] = np.broadcast_to(psi, (1, nsamp, _NPSI)).copy()
        f["samples"] = np.arange(nsamp)
        f["n_e"] = samp(ne, 0.04)
        f["T_e"] = samp(te, 0.05)
        f["T_12C6"] = samp(te * 0.9, 0.05)
        f["Zeff"] = samp(zeff, spread)
        f["n_12C6"] = samp((zeff - 1.0) * ne / 30.0, 0.05)
    return psi, zeff, spread


class TestReaderTiers:
    def test_direct_with_zeff_err_uses_it(self, tmp_path):
        p = str(tmp_path / "new_direct.cdf")
        psi, zeff = _write_direct(p, with_zeff_err=True)
        r = read_ida(p)
        assert r.sigma_Zeff_source == "Zeff_err"
        np.testing.assert_allclose(r.sigma_Zeff, 0.09 * zeff, rtol=1e-12)

    def test_old_direct_without_zeff_err_reports_none(self, tmp_path):
        p = str(tmp_path / "old_direct.cdf")
        _write_direct(p, with_zeff_err=False)
        r = read_ida(p)
        assert r.sigma_Zeff is None
        assert r.sigma_Zeff_source == "none"

    def test_ensemble_uses_sample_spread(self, tmp_path):
        p = str(tmp_path / "ens.cdf")
        psi, zeff, spread = _write_ensemble(p)
        r = read_ida(p, sigma_method="std")
        assert r.sigma_Zeff_source == "ensemble-samples"
        frac = np.median(r.sigma_Zeff / r.Zeff)
        assert frac == pytest.approx(spread, rel=0.25)   # 64 samples

    def test_carbon_crosscheck_reports_zero_on_a_consistent_file(self, tmp_path, capsys):
        p = str(tmp_path / "cons.cdf")
        _write_direct(p, with_zeff_err=True, with_carbon=True)
        r = read_ida(p)
        assert r.zeff_carbon_dev is not None
        assert abs(r.zeff_carbon_dev["median"]) < 1e-10
        assert "Zeff(VB) vs" in capsys.readouterr().out

    def test_carbon_crosscheck_measures_an_injected_inconsistency(self, tmp_path):
        p = str(tmp_path / "incons.cdf")
        psi, ne, te, zeff = _grids()
        _write_direct(p, with_zeff_err=True, with_carbon=False)
        with h5py.File(p, "a") as f:
            # carbon implying Zeff 10 % LOWER than reported
            f["n_12C6"] = ((zeff / 1.1 - 1.0) * ne / 30.0)[None, :]
        r = read_ida(p)
        assert r.zeff_carbon_dev["median"] == pytest.approx(0.10, rel=0.02)

    def test_direct_carbon_tier_propagates_nc_and_ne_errors(self, tmp_path):
        p = str(tmp_path / "carb.cdf")
        psi, zeff = _write_direct(p, with_zeff_err=True, with_carbon=True)
        r = read_ida(p)
        assert r.sigma_Zeff_carbon_source == "n_12C6_err"
        # fixture: s_nC/nC = 0.25, s_ne/ne = 0.05, dilution = zeff - 1
        expect = (zeff - 1.0) * np.sqrt(0.25 ** 2 + 0.05 ** 2)
        np.testing.assert_allclose(r.sigma_Zeff_carbon, expect, rtol=1e-6)

    def test_ensemble_carbon_tier_uses_the_dilution_posterior(self, tmp_path):
        p = str(tmp_path / "enscarb.cdf")
        _write_ensemble(p)
        r = read_ida(p, sigma_method="std")
        assert r.sigma_Zeff_carbon_source == "ensemble-samples"
        assert np.all(np.isfinite(r.sigma_Zeff_carbon))
        assert np.any(r.sigma_Zeff_carbon > 0)

    def test_absent_carbon_channel_skips_the_check(self, tmp_path):
        p = str(tmp_path / "nocarb.cdf")
        _write_direct(p, with_zeff_err=True, with_carbon=False)
        assert read_ida(p).zeff_carbon_dev is None


class TestEnvelopeLadder:
    _base = np.linspace(1.8, 2.1, 10)
    _meas = np.full(10, 0.17)

    def test_measured_wins_on_the_ida_path(self):
        env, label = resolve_zeff_envelope("auto", 0.05, self._base, True,
                                           self._meas, "Zeff_err")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA (Zeff_err)" in label

    def test_scalar_fallback_when_the_file_has_no_measurement(self):
        env, label = resolve_zeff_envelope("auto", 0.05, self._base, True,
                                           None, "none")
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert label.startswith("scalar")

    def test_fuse_baseline_never_pairs_with_an_ida_envelope(self):
        """IMAS/ida_hybrid: zeff_is_ida=False must force the scalar even
        though a measured envelope exists."""
        env, label = resolve_zeff_envelope("auto", 0.05, self._base, False,
                                           self._meas, "Zeff_err")
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert label.startswith("scalar")

    def test_forced_scalar_ignores_the_measurement(self):
        env, label = resolve_zeff_envelope("scalar", 0.05, self._base, True,
                                           self._meas, "Zeff_err")
        np.testing.assert_allclose(env, 0.05 * self._base)

    def test_demanding_measured_warns_loudly_when_unavailable(self):
        with pytest.warns(UserWarning, match="falling back"):
            env, label = resolve_zeff_envelope("measured", 0.05, self._base,
                                               True, None, "none")
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert "no measured tier available" in label

    # ---- the carbon tier (dilution's direct measurement) ------------------
    _carb = np.full(10, 0.04)

    def test_auto_prefers_carbon_over_vb(self):
        env, label = resolve_zeff_envelope(
            "auto", 0.05, self._base, True, self._meas, "Zeff_err",
            carbon_sigma=self._carb, carbon_source="n_12C6_err")
        np.testing.assert_array_equal(env, self._carb)
        assert "carbon-propagated (n_12C6_err)" in label

    def test_forced_carbon_falls_back_to_vb_with_a_warning(self):
        with pytest.warns(UserWarning, match="carbon"):
            env, label = resolve_zeff_envelope(
                "carbon", 0.05, self._base, True, self._meas, "Zeff_err",
                carbon_sigma=None, carbon_source="none")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA" in label

    def test_forced_measured_still_means_the_vb_tier(self):
        env, label = resolve_zeff_envelope(
            "measured", 0.05, self._base, True, self._meas, "Zeff_err",
            carbon_sigma=self._carb, carbon_source="n_12C6_err")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA" in label

    def test_implausibly_large_envelope_draws_the_report_warning(self):
        huge = 0.8 * self._base                      # 80 % of Zeff
        with pytest.warns(UserWarning, match="implausibly large"):
            env, label = resolve_zeff_envelope(
                "measured", 0.05, self._base, True, huge, "Zeff_err")
        np.testing.assert_array_equal(env, huge)     # report-only, not clipped

    def test_unusable_measurement_falls_back_with_a_warning(self):
        bad = np.full(3, 0.1)                      # wrong shape
        with pytest.warns(UserWarning, match="unusable"):
            env, label = resolve_zeff_envelope("auto", 0.05, self._base, True,
                                               bad, "Zeff_err")
        np.testing.assert_allclose(env, 0.05 * self._base)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="zeff_sigma_source"):
            resolve_zeff_envelope("magic", 0.05, self._base, True, None, "none")
