"""
Basic test suite for bouquet.

These tests exercise the pure-Python components (GPR sampling,
uncertainty envelopes, HDF5 round-trip, CLI validation) without
requiring OpenFUSIONToolkit / TokaMaker.
"""

import os
import tempfile

import numpy as np
import pytest

from bouquet.uncertainties import new_uncertainty_profiles
from bouquet.sampling import (
    GPRProfilePerturber,
    generate_perturbed_GPR,
    _draw_monotonic_perturbation,
)
from bouquet.utils import (
    initialize_equilibrium_database,
    store_equilibrium,
    load_equilibrium,
    store_baseline_profiles,
    load_baseline_profiles,
    discover_scan_values,
    count_equilibria,
    Hmode_profiles,
)


# ====================================================================
#  Default kinetic profiles (257-point H-mode baselines)
#  Used as canonical test inputs throughout this module.
# ====================================================================
PSI_N = np.linspace(0, 1, 257)

NE = Hmode_profiles(
    rgrid=257,
    edge=9.0e+18, ped=3.0e+19, core=5.5e+19,
    expin=1.0e+00, expout=1.5e+00,
    widthp=6.0e-02, xphalf=9.75e-01,
)

NI = Hmode_profiles(
    rgrid=257,
    edge=6.0e+18, ped=2.4e+19, core=4.5e+19,
    expin=1.4e+00, expout=2.3e+00,
    widthp=6.7e-02, xphalf=9.75e-01,
)

TE = Hmode_profiles(
    rgrid=257,
    edge=7.0e+01, ped=8.3e+02, core=2.5e+03,
    expin=1.1e+00, expout=9.0e-01,
    widthp=4.8e-02, xphalf=9.75e-01,
)

TI = Hmode_profiles(
    rgrid=257,
    edge=3.0e+02, ped=1.4e+03, core=5.0e+03,
    expin=9.5e-01, expout=2.0e+00,
    widthp=2.0e-01, xphalf=9.5e-01,
)


# ====================================================================
#  Uncertainty envelopes
# ====================================================================
class TestUncertaintyProfiles:
    """Tests for new_uncertainty_profiles."""

    def test_power_law_at_axis(self):
        psi_N = np.linspace(0, 1, 101)
        u = new_uncertainty_profiles(psi_N, uncertainty=0.1, falloff_exp=2.0)
        # At psi_N=0, (1-0)^2 * 0.1 = 0.1
        assert np.isclose(u[0], 0.1)

    def test_power_law_at_edge(self):
        psi_N = np.linspace(0, 1, 101)
        u = new_uncertainty_profiles(psi_N, uncertainty=0.1, falloff_exp=2.0)
        # At psi_N=1, (1-1)^2 * 0.1 = 0.0
        assert np.isclose(u[-1], 0.0, atol=1e-14)

    def test_flat_plus_tail_core_value(self):
        psi_N = np.linspace(0, 1, 201)
        u = new_uncertainty_profiles(psi_N, uncertainty=0.05)
        # In the flat region (all psi_N < 0.8 default), value = 0.05
        core_mask = psi_N < 0.5
        np.testing.assert_allclose(u[core_mask], 0.05, atol=1e-14)

    def test_output_shape(self):
        psi_N = np.linspace(0, 1, 50)
        u = new_uncertainty_profiles(psi_N, uncertainty=0.1, falloff_exp=1.0)
        assert u.shape == psi_N.shape

    def test_non_negative(self):
        psi_N = np.linspace(0, 1, 201)
        u = new_uncertainty_profiles(psi_N, uncertainty=0.1)
        assert np.all(u >= 0.0)


# ====================================================================
#  GPR sampling
# ====================================================================
class TestGPRProfilePerturber:
    """Tests for the GPR sampling engine."""

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="not in"):
            GPRProfilePerturber(kernel_func="invalid_kernel")

    def test_output_shape_single(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        result = p.generate_profiles(psi_N, profile, sigma, n_samples=1)
        assert result.shape == (1, len(psi_N))

    def test_output_shape_multi(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        result = p.generate_profiles(psi_N, profile, sigma, n_samples=10)
        assert result.shape == (10, len(psi_N))

    def test_zero_sigma_returns_mean(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = np.zeros_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        rng = np.random.default_rng(0)
        result = p.generate_profiles(psi_N, profile, sigma, n_samples=5, rng=rng)
        # Every row should match the input profile
        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i], profile, atol=1e-10)

    def test_marginal_std_matches_sigma(self):
        """Empirical pointwise std should match the input sigma."""
        rng = np.random.default_rng(42)
        psi_N = np.linspace(0, 1, 41)
        profile = np.ones_like(psi_N)
        sigma = 0.1 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        samples = p.generate_profiles(psi_N, profile, sigma,
                                      n_samples=5000, rng=rng)
        empirical_std = np.std(samples, axis=0)
        np.testing.assert_allclose(empirical_std, sigma, rtol=0.1)

    def test_matern52_runs(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="matern52", length_scale=0.2)
        result = p.generate_profiles(psi_N, profile, sigma, n_samples=3)
        assert result.shape == (3, len(psi_N))

    def test_output_shape_single_redraw(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        rng = np.random.default_rng(0)
        result = p.draw_from_factor(profile, 1, rng)
        assert result.shape == (1, len(psi_N))

    def test_output_shape_multi_redraw(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        rng = np.random.default_rng(0)
        result = p.draw_from_factor(profile, 10, rng)
        assert result.shape == (10, len(psi_N))

    def test_zero_sigma_returns_mean_redraw(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = np.zeros_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        rng = np.random.default_rng(0)
        result = p.draw_from_factor(profile, 5, rng)
        # Every row should match the input profile
        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i], profile, atol=1e-10)

    def test_marginal_std_matches_sigma_redraw(self):
        """Empirical pointwise std should match the input sigma."""
        rng = np.random.default_rng(42)
        psi_N = np.linspace(0, 1, 41)
        profile = np.ones_like(psi_N)
        sigma = 0.1 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        samples = p.draw_from_factor(profile, 5000, rng)
        empirical_std = np.std(samples, axis=0)
        np.testing.assert_allclose(empirical_std, sigma, rtol=0.1)

    def test_redraw_matches_generate_profiles_redraw(self):
        """Re-draw and generate_profiles must produce the same marginal std."""
        psi_N = np.linspace(0, 1, 41)
        profile = np.ones_like(psi_N)
        sigma = 0.1 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)

        rng_a = np.random.default_rng(7)
        samples_a = p.draw_from_factor(profile, 3000, rng_a)

        rng_b = np.random.default_rng(7)
        samples_b = p.generate_profiles(psi_N, profile, sigma, n_samples=3000, rng=rng_b)

        std_a = np.std(samples_a, axis=0)
        std_b = np.std(samples_b, axis=0)
        # Empirical stds agree to within 1 % (Monte-Carlo noise)
        np.testing.assert_allclose(std_a, std_b, rtol=0.01)

    def test_matern52_runs_redraw(self):
        psi_N = np.linspace(0, 1, 51)
        profile = 1.0 - psi_N
        sigma = 0.05 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="matern52", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        rng = np.random.default_rng(0)
        result = p.draw_from_factor(profile, 3, rng)
        assert result.shape == (3, len(psi_N))

    def test_precompute_reuse_consistent(self):
        """Multiple draw_from_factor calls with the same factor must be i.i.d."""
        psi_N = np.linspace(0, 1, 31)
        profile = np.ones_like(psi_N)
        sigma = 0.1 * np.ones_like(psi_N)
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.2)
        p.precompute_factor(psi_N, sigma)
        rng = np.random.default_rng(99)
        # Draw in two separate calls and stack
        batch1 = p.draw_from_factor(profile, 3000, rng)
        batch2 = p.draw_from_factor(profile, 3000, rng)
        # Verify batch1 and batch2 are actually different (independent)
        assert not np.allclose(batch1, batch2), "batch1 and batch2 should be different"
        combined = np.vstack([batch1, batch2])
        empirical_std = np.std(combined, axis=0)
        # Check thath their standard deviation is similar
        np.testing.assert_allclose(empirical_std, sigma, rtol=0.01)


class TestDrawMonotonicPerturbation:
    """Quick tests for _draw_monotonic_perturbation."""

    # A strictly decreasing parabola — nearly every GPR draw is monotone
    PSI = np.linspace(0, 1, 32)
    PROFILE = 1.0 - PSI ** 2
    SIGMA = 0.02 * np.ones(32)

    def test_returns_monotone_array(self):
        result = _draw_monotonic_perturbation(
            self.PSI, self.PROFILE, self.SIGMA, length_scale=0.3,
        )
        assert result.shape == (len(self.PSI),)
        assert np.all(np.diff(result) <= 0.0)

    def test_pre_built_perturber_accepted(self):
        """Passing a pre-built perturber must still return a valid monotone draw."""
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.3)
        p.precompute_factor(self.PSI, self.SIGMA)
        rng = np.random.default_rng(7)
        result = _draw_monotonic_perturbation(
            self.PSI, self.PROFILE, self.SIGMA, length_scale=0.3,
            perturber=p, rng=rng,
        )
        assert np.all(np.diff(result) <= 0.0)

    def test_shared_rng_advances_state(self):
        """Two calls with the same rng object produce different draws."""
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.3)
        p.precompute_factor(self.PSI, self.SIGMA)
        rng = np.random.default_rng(42)
        r1 = _draw_monotonic_perturbation(
            self.PSI, self.PROFILE, self.SIGMA, length_scale=0.3,
            perturber=p, rng=rng,
        )
        r2 = _draw_monotonic_perturbation(
            self.PSI, self.PROFILE, self.SIGMA, length_scale=0.3,
            perturber=p, rng=rng,
        )
        assert not np.allclose(r1, r2)

    def test_exhaustion_raises_runtime_error(self):
        """RuntimeError must be raised when no monotone draw is found in max_draws."""
        # Flat profile + huge sigma: draws will not be monotone in 1 attempt.
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.3)
        psi = np.linspace(0, 1, 8)
        sigma = 10.0 * np.ones(8)
        p.precompute_factor(psi, sigma)
        rng = np.random.default_rng(0)
        with pytest.raises(RuntimeError, match="monotonically"):
            _draw_monotonic_perturbation(
                psi, np.ones(8), sigma, length_scale=0.3,
                max_draws=1, batch_size=1, perturber=p, rng=rng,
            )


class TestGeneratePerturbedGPR:
    """Tests for the convenience wrapper."""

    def test_single_sample_is_1d(self):
        x = np.linspace(0, 1, 51)
        profile = np.ones_like(x)
        result = generate_perturbed_GPR(x, profile, n_samples=1)
        assert result.ndim == 1

    def test_multi_sample_is_2d(self):
        x = np.linspace(0, 1, 51)
        profile = np.ones_like(x)
        result = generate_perturbed_GPR(x, profile, n_samples=5)
        assert result.ndim == 2
        assert result.shape[0] == 5

    def test_none_sigma_no_perturbation(self):
        x = np.linspace(0, 1, 51)
        profile = np.sin(x)
        result = generate_perturbed_GPR(x, profile, sigma_profile=None, n_samples=1)
        np.testing.assert_allclose(result, profile, atol=1e-12)


# ====================================================================
#  HDF5 round-trip
# ====================================================================
class TestHDF5RoundTrip:
    """Test store/load cycle for equilibria and baselines."""

    @pytest.fixture()
    def tmp_db(self, tmp_path):
        """Create a temporary database and return (header, db_path)."""
        header = str(tmp_path / "test_eq")
        db_path = initialize_equilibrium_database(header)
        return header, db_path

    def _dummy_profiles(self, n=51):
        psi_N = np.linspace(0, 1, n)
        ones = np.ones(n)
        return psi_N, ones

    def test_init_creates_file(self, tmp_db):
        _, db_path = tmp_db
        assert os.path.isfile(db_path)

    def test_store_and_load_baseline(self, tmp_db):
        header, db_path = tmp_db
        n = 51
        psi_N, ones = self._dummy_profiles(n)

        store_baseline_profiles(
            header, psi_N,
            ne=ones, te=ones * 2, ni=ones * 3, ti=ones * 4,
            pressure=ones * 5, j_phi=ones * 6,
            sigma_ne=ones * 0.1, sigma_te=ones * 0.2,
            sigma_ni=ones * 0.3, sigma_ti=ones * 0.4,
            sigma_jphi=ones * 0.5,
            Ip_target=1e6, l_i_target=0.8,
        )

        bl = load_baseline_profiles(db_path)
        np.testing.assert_array_equal(bl["psi_N"], psi_N)
        np.testing.assert_array_equal(bl["n_e [m^-3]"], ones)
        assert bl["Ip_target"] == 1e6

    def test_store_and_load_equilibrium(self, tmp_db):
        header, db_path = tmp_db
        n = 51
        psi_N, ones = self._dummy_profiles(n)

        # Write a fake eqdsk file
        eqdsk_path = db_path.replace(".h5", ".eqdsk")
        with open(eqdsk_path, "wb") as f:
            f.write(b"fake eqdsk data")

        store_equilibrium(
            header, count=0, eqdsk_filepath=eqdsk_path,
            psi_N=psi_N, j_phi=ones, j_BS=ones, j_inductive=ones,
            n_e=ones, T_e=ones, n_i=ones, T_i=ones, w_ExB=ones,
            li1=0.5, li3=0.7,
        )

        result = load_equilibrium(header, count=0)
        np.testing.assert_array_equal(result["psi_N"], psi_N)
        assert result["l_i(1)"] == 0.5
        assert result["l_i(3)"] == 0.7
        assert result["eqdsk_bytes"] == b"fake eqdsk data"

        os.remove(eqdsk_path)

    def test_count_equilibria(self, tmp_db):
        header, db_path = tmp_db
        n = 51
        psi_N, ones = self._dummy_profiles(n)

        eqdsk_path = db_path.replace(".h5", ".eqdsk")
        with open(eqdsk_path, "wb") as f:
            f.write(b"fake")

        # count before any stores
        assert count_equilibria(db_path) == 0

        for i in range(3):
            store_equilibrium(
                header, count=i, eqdsk_filepath=eqdsk_path,
                psi_N=psi_N, j_phi=ones, j_BS=ones, j_inductive=ones,
                n_e=ones, T_e=ones, n_i=ones, T_i=ones, w_ExB=ones,
                li1=0.5, li3=0.7,
            )

        assert count_equilibria(db_path) == 3
        os.remove(eqdsk_path)


# ====================================================================
#  discover_scan_values sorting
# ====================================================================
class TestDiscoverScanValues:
    """Test that scan values are sorted numerically when possible."""

    def test_numeric_sort(self, tmp_path):
        import h5py
        db_path = str(tmp_path / "sort_test.h5")
        with h5py.File(db_path, "w") as hf:
            scan = hf.create_group("scan")
            for key in ["10", "2", "1", "20"]:
                scan.create_group(key)

        result = discover_scan_values(db_path)
        assert result == ["1", "2", "10", "20"]

    def test_string_sort_fallback(self, tmp_path):
        import h5py
        db_path = str(tmp_path / "sort_test2.h5")
        with h5py.File(db_path, "w") as hf:
            scan = hf.create_group("scan")
            for key in ["beta", "alpha", "gamma"]:
                scan.create_group(key)

        result = discover_scan_values(db_path)
        assert result == ["alpha", "beta", "gamma"]

    def test_flat_layout_returns_none(self, tmp_path):
        import h5py
        db_path = str(tmp_path / "flat.h5")
        with h5py.File(db_path, "w") as hf:
            hf.create_group("_baseline")

        result = discover_scan_values(db_path)
        assert result is None


# ====================================================================
#  CLI validation
# ====================================================================
class TestCLIValidation:
    """Test the HDF5 validation in gui._validate_h5."""

    def test_missing_file_exits(self):
        from bouquet.gui import _validate_h5
        with pytest.raises(SystemExit, match="file not found"):
            _validate_h5("/nonexistent/path/fake.h5")

    def test_invalid_h5_exits(self, tmp_path):
        from bouquet.gui import _validate_h5
        bad = tmp_path / "bad.h5"
        bad.write_text("not an hdf5 file")
        with pytest.raises(SystemExit, match="cannot open"):
            _validate_h5(str(bad))

    def test_missing_groups_exits(self, tmp_path):
        import h5py
        from bouquet.gui import _validate_h5
        empty = tmp_path / "empty.h5"
        with h5py.File(str(empty), "w"):
            pass
        with pytest.raises(SystemExit, match="does not look like"):
            _validate_h5(str(empty))

    def test_valid_flat_layout(self, tmp_path):
        import h5py
        from bouquet.gui import _validate_h5
        good = tmp_path / "good.h5"
        with h5py.File(str(good), "w") as hf:
            hf.create_group("_baseline")
        # Should not raise
        _validate_h5(str(good))

    def test_valid_scan_layout(self, tmp_path):
        import h5py
        from bouquet.gui import _validate_h5
        good = tmp_path / "good_scan.h5"
        with h5py.File(str(good), "w") as hf:
            hf.create_group("scan/1.0/_baseline")
        _validate_h5(str(good))
