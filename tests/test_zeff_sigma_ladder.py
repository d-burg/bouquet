"""Measured Z_eff uncertainty: reader tiers + the three-tier envelope ladder.

The IDA files carry a measured Zeff uncertainty in BOTH modern vintages --
posterior samples (ensemble layout) and a ``Zeff_err`` dataset (newer direct
layout) -- and, where CER carbon is present, the dilution's own propagated
uncertainty, which the pipeline previously discarded in favour of an assumed
5 % scalar (measured values run ~8-9 % median in-core).  The reader reports
the highest-fidelity tier the file can support, and ``resolve_zeff_envelope``
picks it with explicit provenance -- THREE tiers, not two:

    carbon-propagated  >  VB-measured sigma_Zeff  >  zeff_scalar_sigma*|Zeff|

with both measured tiers eligible only when the Z_eff baseline itself is the
IDA one (recon path, and the SAME file that supplies the sigmas) -- a FUSE
baseline (IMAS/ida_hybrid) must not be paired with an IDA envelope.  That
file-identity test compares RESOLVED paths, and every step down the ladder
warns once and is recorded in ``resolve_uncertainty``'s ``zeff_sigma_tier``
metadata; none of them may be silent.

Synthetic .cdf-shaped HDF5 files exercise all three vintages; no solver.
"""
import os
import warnings

import h5py
import numpy as np
import pytest

from bouquet.baseline import resolve_zeff_envelope, zeff_sigma_eligibility
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
        env, label, meta = resolve_zeff_envelope(
            "auto", 0.05, self._base, True, self._meas, "Zeff_err")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA (Zeff_err)" in label
        assert meta["tier"] == "VB-measured"
        assert meta["provenance"] == "Zeff_err"

    def test_scalar_fallback_when_the_file_has_no_measurement(self):
        with pytest.warns(UserWarning, match="skipped"):
            env, label, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, True, None, "none",
                ida_in_play=True)
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert label.startswith("scalar")
        assert meta["tier"] == "scalar" and meta["fell_back"]
        assert [s["tier"] for s in meta["skipped"]] == ["carbon-propagated",
                                                        "VB-measured"]
        assert all("missing dataset" in s["reason"] for s in meta["skipped"])

    def test_fuse_baseline_never_pairs_with_an_ida_envelope(self):
        """IMAS/ida_hybrid: zeff_is_ida=False must force the scalar even
        though a measured envelope exists."""
        env, label, meta = resolve_zeff_envelope(
            "auto", 0.05, self._base, False, self._meas, "Zeff_err")
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert label.startswith("scalar")
        assert meta["tier"] == "scalar" and not meta["eligible"]

    def test_forced_scalar_ignores_the_measurement(self):
        env, label, meta = resolve_zeff_envelope(
            "scalar", 0.05, self._base, True, self._meas, "Zeff_err")
        np.testing.assert_allclose(env, 0.05 * self._base)
        # explicitly asked for: recorded, but not a fallback and not warned
        assert meta["tier"] == "scalar" and not meta["fell_back"]
        assert not meta["warned"]

    def test_demanding_measured_warns_loudly_when_unavailable(self):
        with pytest.warns(UserWarning, match="skipped"):
            env, label, meta = resolve_zeff_envelope(
                "measured", 0.05, self._base, True, None, "none")
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert "FALLBACK" in label
        assert meta["warned"] and meta["skipped"][0]["tier"] == "VB-measured"

    # ---- the carbon tier (dilution's direct measurement) ------------------
    _carb = np.full(10, 0.04)

    def test_auto_prefers_carbon_over_vb(self):
        env, label, meta = resolve_zeff_envelope(
            "auto", 0.05, self._base, True, self._meas, "Zeff_err",
            carbon_sigma=self._carb, carbon_source="n_12C6_err")
        np.testing.assert_array_equal(env, self._carb)
        assert "carbon-propagated (n_12C6_err)" in label
        assert meta["tier"] == "carbon-propagated"
        assert meta["skipped"] == [] and not meta["warned"]

    def test_forced_carbon_falls_back_to_vb_with_a_warning(self):
        with pytest.warns(UserWarning, match="carbon"):
            env, label, meta = resolve_zeff_envelope(
                "carbon", 0.05, self._base, True, self._meas, "Zeff_err",
                carbon_sigma=None, carbon_source="none")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA" in label
        assert meta["tier"] == "VB-measured"
        assert meta["skipped"] == [
            {"tier": "carbon-propagated",
             "reason": "missing dataset: this file provides no n_12C6 "
                       "uncertainty"}]

    def test_forced_measured_still_means_the_vb_tier(self):
        env, label, meta = resolve_zeff_envelope(
            "measured", 0.05, self._base, True, self._meas, "Zeff_err",
            carbon_sigma=self._carb, carbon_source="n_12C6_err")
        np.testing.assert_array_equal(env, self._meas)
        assert "measured IDA" in label
        # 'measured' never attempts carbon, so it is not reported as skipped
        assert meta["skipped"] == []

    def test_implausibly_large_envelope_draws_the_report_warning(self):
        huge = 0.8 * self._base                      # 80 % of Zeff
        with pytest.warns(UserWarning, match="implausibly large"):
            env, label, meta = resolve_zeff_envelope(
                "measured", 0.05, self._base, True, huge, "Zeff_err")
        np.testing.assert_array_equal(env, huge)     # report-only, not clipped
        assert meta["median_fraction_of_zeff"] == pytest.approx(0.8)

    def test_unusable_measurement_falls_back_with_a_warning(self):
        bad = np.full(3, 0.1)                      # wrong shape
        with pytest.warns(UserWarning, match="invalid data"):
            env, label, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, True, bad, "Zeff_err",
                ida_in_play=True)
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert any("shape" in s["reason"] for s in meta["skipped"])

    def test_negative_sigma_entries_are_refused(self):
        """These are 1-sigma MAGNITUDES.  ``all finite and any > 0`` admitted
        an array with negative entries as long as one entry was positive, and
        a negative sigma propagates a sign into the draw scales."""
        bad = self._meas.copy()
        bad[4] = -0.17
        with pytest.warns(UserWarning, match="negative"):
            env, label, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, True, bad, "Zeff_err",
                ida_in_play=True)
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert any("negative" in s["reason"] for s in meta["skipped"])

    def test_all_zero_sigma_is_still_refused(self):
        with pytest.warns(UserWarning, match="all-zero"):
            env, _, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, True, np.zeros(10), "Zeff_err",
                ida_in_play=True)
        np.testing.assert_allclose(env, 0.05 * self._base)

    def test_ineligible_source_records_and_warns_when_an_ida_is_in_play(self):
        """The ida_hybrid case: an IDA file IS configured and its measured
        envelope is refused by design.  Refusing it silently is what this
        ladder must never do."""
        with pytest.warns(UserWarning, match="source ineligible"):
            env, label, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, False, self._meas, "Zeff_err",
                ineligible_reason="the Z_eff baseline comes from the "
                                  "IMAS/FUSE source",
                ida_in_play=True)
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert meta["ineligible_reason"].startswith("the Z_eff baseline")
        assert all("source ineligible" in s["reason"] for s in meta["skipped"])

    def test_no_ida_file_at_all_is_recorded_but_not_warned(self):
        """No IDA file configured: there is no ladder to fall down, so the
        scalar is not a fallback event -- still fully recorded."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env, label, meta = resolve_zeff_envelope(
                "auto", 0.05, self._base, False, None, "none",
                ineligible_reason="no IDA sigma file is configured",
                ida_in_play=False)
        np.testing.assert_allclose(env, 0.05 * self._base)
        assert meta["tier"] == "scalar" and not meta["warned"]
        assert meta["skipped"], "the un-attempted tiers must still be recorded"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="zeff_sigma_source"):
            resolve_zeff_envelope("magic", 0.05, self._base, True, None, "none")


class TestCarbonDataValidity:
    """Garbage carbon data must drop the tier LOUDLY, never corrupt it.

    Negative n_12C6 (SOL spline undershoot) and netCDF fill values survive
    the clip floors -- a negative nC keeps its sign while its magnitude is
    floored out of the relative-error denominator, producing ~1e6-scale
    sigmas that pass _usable (finite, some > 0) and silently corrupt the
    whole Zeff/ni ensemble.  NaN holes previously became sigma=0 radii via
    nan_to_num.  All three now skip the tier with a printed reason.
    """

    def _direct_with(self, path, mutate):
        psi, zeff = _write_direct(path, with_zeff_err=True, with_carbon=True)
        with h5py.File(path, "a") as f:
            mutate(f)
        return psi, zeff

    def test_negative_carbon_drops_the_tier(self, tmp_path, capsys):
        p = str(tmp_path / "negc.cdf")

        def mutate(f):
            nc = f["n_12C6"][0]
            nc[-3:] = -2e17                     # SOL undershoot
            f["n_12C6"][0] = nc
        self._direct_with(p, mutate)
        r = read_ida(p)
        assert r.sigma_Zeff_carbon is None
        assert r.sigma_Zeff_carbon_source == "none"
        assert "carbon-tier sigma skipped" in capsys.readouterr().out
        # the VB tier is untouched and still wins 'auto'
        assert r.sigma_Zeff is not None

    def test_fill_values_drop_the_tier(self, tmp_path, capsys):
        p = str(tmp_path / "fill.cdf")

        def mutate(f):
            err = f["n_12C6_err"][0]
            err[5] = 9.96921e36                 # netCDF default fill
            f["n_12C6_err"][0] = err
        self._direct_with(p, mutate)
        r = read_ida(p)
        assert r.sigma_Zeff_carbon is None
        assert "carbon-tier sigma skipped" in capsys.readouterr().out

    def test_nan_holes_drop_the_tier_instead_of_zeroing(self, tmp_path):
        """The old nan_to_num turned masked channels into sigma=0 radii --
        never-perturbed points reported under provenance 'n_12C6_err'."""
        p = str(tmp_path / "nanhole.cdf")

        def mutate(f):
            err = f["n_12C6_err"][0]
            err[7] = np.nan
            f["n_12C6_err"][0] = err
        self._direct_with(p, mutate)
        r = read_ida(p)
        assert r.sigma_Zeff_carbon is None
        assert r.sigma_Zeff_carbon_source == "none"

    def test_edge_only_grid_does_not_crash_the_crosscheck(self, tmp_path, capsys):
        """A pedestal-only fit vintage (no psi_N <= 0.9) crashed read_ida
        via argmax([]) inside the report-only cross-check."""
        p = str(tmp_path / "edge.cdf")
        psi = np.linspace(0.92, 1.05, _NPSI)
        ne = 1e19 * np.ones(_NPSI)
        zeff = np.full(_NPSI, 2.0)
        n_c = (zeff - 1.0) * ne / 30.0
        with h5py.File(p, "w") as f:
            f["time"] = np.array([3000.0])
            f["psi_n"] = psi
            for k, v in (("n_e", ne), ("T_e", 100 * np.ones(_NPSI)),
                         ("T_12C6", 90 * np.ones(_NPSI))):
                f[k] = v[None, :]
                f[k + "_err"] = (0.05 * v)[None, :]
            f["Zeff"] = zeff[None, :]
            f["n_12C6"] = n_c[None, :]
        r = read_ida(p)                          # must not raise
        assert r.zeff_carbon_dev is None
        assert "cross-check skipped" in capsys.readouterr().out


def _mk_bl(psi_kin):
    from bouquet.baseline import Baseline
    psi_N = np.linspace(0.0, 1.0, 33)
    _, ne, te, zeff = _grids()
    j = 8.0e5 * (1.0 - psi_N ** 2)
    return Baseline(
        psi_N=psi_N, j_phi=j, j_inductive=0.85 * j, j_BS=0.15 * j,
        psi_N_kinetic=psi_kin, ne=ne, te=te, ni=0.9 * ne, ti=0.9 * te,
        Zeff=zeff, Ip_target=1.2e6, l_i_target=0.85,
        provenance="reconstruction",
    )


def _mk_cfg(profiles_path, tmp_path, unc=None, impurity_Z=6.0):
    from bouquet.config import (BouquetConfig, ReconstructionSource,
                                SolverConfig, UncertaintyConfig)
    return BouquetConfig(
        source=ReconstructionSource(
            geqdsk_path=str(tmp_path / "g.geqdsk"),
            profiles_path=profiles_path, time=3.0,
            impurity_Z=impurity_Z),
        solver=SolverConfig(mesh_path=str(tmp_path / "mesh.h5")),
        output_header=str(tmp_path / "out"),
        uncertainty=unc or UncertaintyConfig(),
    )


class TestEnvelopeWiring:
    """resolve_uncertainty-level wiring: eligibility + impurity_Z."""

    _bl = staticmethod(_mk_bl)
    _cfg = staticmethod(_mk_cfg)

    def test_own_cdf_gets_the_carbon_tier(self, tmp_path):
        from bouquet.baseline import resolve_uncertainty
        cdf = str(tmp_path / "own.cdf")
        psi, zeff = _write_direct(cdf, with_zeff_err=True, with_carbon=True)
        env = resolve_uncertainty(self._cfg(cdf, tmp_path), self._bl(psi))
        expect = (zeff - 1.0) * np.sqrt(0.25 ** 2 + 0.05 ** 2)
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"], expect,
                                   rtol=1e-6)

    def test_pfile_baseline_never_pairs_with_an_ida_envelope(self, tmp_path):
        """A p-file Zeff baseline + unc.ida_path must resolve the SCALAR:
        pairing it with the .cdf's absolute measured envelope is the
        cross-channel mixing the gate forbids."""
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        cdf = str(tmp_path / "sig.cdf")
        _write_direct(cdf, with_zeff_err=True, with_carbon=True)
        cfg = self._cfg(str(tmp_path / "p.peqdsk"), tmp_path,
                        unc=UncertaintyConfig(ida_path=cdf))
        _, _, _, zeff = _grids()
        bl = self._bl(np.linspace(0.0, 1.2, _NPSI))
        env = resolve_uncertainty(cfg, bl)
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   0.05 * np.abs(bl.Zeff), rtol=1e-6)

    def test_mismatched_ida_path_falls_back_to_scalar(self, tmp_path):
        """unc.ida_path naming a DIFFERENT .cdf than the source's own file
        (another vintage) must not pair its envelope with this baseline."""
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        own = str(tmp_path / "own.cdf")
        other = str(tmp_path / "other_vintage.cdf")
        psi, zeff = _write_direct(own, with_zeff_err=True, with_carbon=True)
        _write_direct(other, with_zeff_err=True, with_carbon=True)
        cfg = self._cfg(own, tmp_path, unc=UncertaintyConfig(ida_path=other))
        env = resolve_uncertainty(cfg, self._bl(psi))
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   0.05 * np.abs(zeff), rtol=1e-6)

    def test_impurity_Z_reaches_the_carbon_tier(self, tmp_path):
        """The sigma-envelope read must carry source.impurity_Z: the carbon
        propagation is Z(Z-1)-quadratic, and the old call site pinned it to
        carbon (Z=6) for every machine."""
        from bouquet.baseline import resolve_uncertainty
        cdf = str(tmp_path / "z5.cdf")
        psi, zeff = _write_direct(cdf, with_zeff_err=True, with_carbon=True)
        env = resolve_uncertainty(self._cfg(cdf, tmp_path, impurity_Z=5.0),
                                  self._bl(psi))
        # fixture carbon: nc = (zeff-1)*ne/30, so dil(Z=5) = 20*nc/ne
        expect = (zeff - 1.0) * (20.0 / 30.0) * np.sqrt(0.25 ** 2 + 0.05 ** 2)
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"], expect,
                                   rtol=1e-6)


class TestSigmaPathSpelling:
    """A path SPELLING must never cost a run its measured Z_eff tiers.

    The first version of the gate compared RAW strings
    (``ida_path == src.profiles_path``).  That answers "different file" for a
    relative path, a ``~`` prefix, a trailing ``/`` or ``./`` segment and a
    symlink -- and the consequence was a silent drop to the ASSUMED scalar
    envelope, the same failure class this feature's own end-to-end A/B
    caught.  Every spelling below names the SAME file and must keep the
    carbon tier; a genuinely different file must still be refused, and must
    say so.

    Each case asserts the two spellings are UNEQUAL AS STRINGS, so this
    module fails loudly if the comparison ever reverts to string equality.
    """

    def _own(self, tmp_path):
        own = tmp_path / "own.cdf"
        psi, zeff = _write_direct(str(own), with_zeff_err=True,
                                  with_carbon=True)
        return own, psi, zeff

    def _expect_carbon(self, zeff):
        # the fixture's carbon propagation, identical to the wiring tests
        return (zeff - 1.0) * np.sqrt(0.25 ** 2 + 0.05 ** 2)

    # ---- unit level: the eligibility predicate itself ---------------------

    def _eligible(self, src_spelling, sigma_spelling, tmp_path):
        from bouquet.config import ReconstructionSource
        src = ReconstructionSource(geqdsk_path=str(tmp_path / "g.geqdsk"),
                                   profiles_path=str(src_spelling), time=3.0)
        return zeff_sigma_eligibility(src, str(sigma_spelling))

    def test_relative_vs_absolute_is_the_same_file(self, tmp_path,
                                                   monkeypatch):
        own, _, _ = self._own(tmp_path)
        monkeypatch.chdir(tmp_path)
        rel = "own.cdf"
        assert rel != str(own)              # raw equality would say "differ"
        ok, why = self._eligible(rel, own, tmp_path)
        assert ok, why
        assert self._eligible(own, rel, tmp_path)[0]

    def test_dot_segment_is_the_same_file(self, tmp_path):
        own, _, _ = self._own(tmp_path)
        dotted = os.path.join(str(tmp_path), ".", "own.cdf")
        assert dotted != str(own)
        assert self._eligible(dotted, own, tmp_path)[0]

    def test_trailing_separator_is_the_same_file(self, tmp_path):
        own, _, _ = self._own(tmp_path)
        trailing = str(own) + "/"
        assert trailing != str(own)
        # note: the raw-string `.endswith(".cdf")` screen also fails on this
        assert self._eligible(trailing, own, tmp_path)[0]

    def test_tilde_prefix_is_the_same_file(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        own = home / "own.cdf"
        _write_direct(str(own), with_zeff_err=True, with_carbon=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        tilde = os.path.join("~", "own.cdf")
        assert tilde != str(own)
        assert self._eligible(tilde, own, tmp_path)[0]

    def test_symlinked_file_is_the_same_file(self, tmp_path):
        own, _, _ = self._own(tmp_path)
        link = tmp_path / "link.cdf"
        link.symlink_to(own)
        assert str(link) != str(own)
        assert self._eligible(link, own, tmp_path)[0]

    def test_symlinked_parent_directory_is_the_same_file(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        own = real / "own.cdf"
        _write_direct(str(own), with_zeff_err=True, with_carbon=True)
        alias = tmp_path / "alias"
        alias.symlink_to(real, target_is_directory=True)
        via_alias = alias / "own.cdf"
        assert str(via_alias) != str(own)
        assert self._eligible(via_alias, own, tmp_path)[0]

    def test_a_genuinely_different_file_is_still_refused_and_says_why(
            self, tmp_path):
        own, _, _ = self._own(tmp_path)
        other = tmp_path / "other_vintage.cdf"
        _write_direct(str(other), with_zeff_err=True, with_carbon=True)
        ok, why = self._eligible(own, other, tmp_path)
        assert not ok
        assert "genuinely different file" in why
        assert why, "a refusal must carry a reason, never be silent"

    def test_a_non_cdf_profiles_file_is_refused_and_says_why(self, tmp_path):
        own, _, _ = self._own(tmp_path)
        ok, why = self._eligible(tmp_path / "p.peqdsk", own, tmp_path)
        assert not ok and "not an IDA .cdf" in why

    def test_no_ida_file_is_refused_and_says_why(self, tmp_path):
        from bouquet.config import ReconstructionSource
        own, _, _ = self._own(tmp_path)
        ok, why = zeff_sigma_eligibility(
            ReconstructionSource(geqdsk_path=str(tmp_path / "g.geqdsk"),
                                 profiles_path=str(own), time=3.0), None)
        assert not ok and "no IDA sigma file" in why

    # ---- end to end: the tier that actually reaches the sampler ----------

    def test_relative_spelling_keeps_the_carbon_tier_end_to_end(
            self, tmp_path, monkeypatch):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        own, psi, zeff = self._own(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = _mk_cfg("own.cdf", tmp_path,
                      unc=UncertaintyConfig(ida_path=str(own)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # no fallback may occur
            env = resolve_uncertainty(cfg, _mk_bl(psi))
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   self._expect_carbon(zeff), rtol=1e-6)
        assert env["zeff_sigma_tier"]["tier"] == "carbon-propagated"
        assert env["zeff_sigma_tier"]["skipped"] == []

    def test_symlink_spelling_keeps_the_carbon_tier_end_to_end(self, tmp_path):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        own, psi, zeff = self._own(tmp_path)
        link = tmp_path / "link.cdf"
        link.symlink_to(own)
        cfg = _mk_cfg(str(own), tmp_path,
                      unc=UncertaintyConfig(ida_path=str(link)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env = resolve_uncertainty(cfg, _mk_bl(psi))
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   self._expect_carbon(zeff), rtol=1e-6)
        assert env["zeff_sigma_tier"]["tier"] == "carbon-propagated"

    def test_a_different_file_falls_back_loudly_and_is_recorded(
            self, tmp_path):
        """The refusal is correct -- but it must WARN and be RECORDED, not
        silently hand the run an assumed 5 %-class envelope."""
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        own, psi, zeff = self._own(tmp_path)
        other = tmp_path / "other_vintage.cdf"
        _write_direct(str(other), with_zeff_err=True, with_carbon=True)
        cfg = _mk_cfg(str(own), tmp_path,
                      unc=UncertaintyConfig(ida_path=str(other)))
        with pytest.warns(UserWarning, match="genuinely different file"):
            env = resolve_uncertainty(cfg, _mk_bl(psi))
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   0.05 * np.abs(zeff), rtol=1e-6)
        meta = env["zeff_sigma_tier"]
        assert meta["tier"] == "scalar" and meta["warned"]
        assert not meta["eligible"]
        assert "genuinely different file" in meta["ineligible_reason"]
        assert [s["tier"] for s in meta["skipped"]] == ["carbon-propagated",
                                                        "VB-measured"]

    def test_a_pfile_baseline_falls_back_loudly_and_is_recorded(
            self, tmp_path):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import UncertaintyConfig
        own, psi, zeff = self._own(tmp_path)
        cfg = _mk_cfg(str(tmp_path / "p.peqdsk"), tmp_path,
                      unc=UncertaintyConfig(ida_path=str(own)))
        with pytest.warns(UserWarning, match="not an IDA .cdf"):
            env = resolve_uncertainty(cfg, _mk_bl(psi))
        meta = env["zeff_sigma_tier"]
        assert meta["tier"] == "scalar" and meta["warned"]
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                                   0.05 * np.abs(_mk_bl(psi).Zeff), rtol=1e-6)

    def test_missing_dataset_fallback_is_warned_and_recorded(self, tmp_path):
        """Falling back because the FILE lacks a tier is reported with a
        different reason class than a path mismatch."""
        from bouquet.baseline import resolve_uncertainty
        own = tmp_path / "nocarb.cdf"
        psi, zeff = _write_direct(str(own), with_zeff_err=True,
                                  with_carbon=False)
        cfg = _mk_cfg(str(own), tmp_path)
        with pytest.warns(UserWarning, match="missing dataset"):
            env = resolve_uncertainty(cfg, _mk_bl(psi))
        meta = env["zeff_sigma_tier"]
        assert meta["tier"] == "VB-measured"
        assert meta["skipped"][0]["tier"] == "carbon-propagated"
        assert "missing dataset" in meta["skipped"][0]["reason"]
        assert "path" not in meta["skipped"][0]["reason"]
