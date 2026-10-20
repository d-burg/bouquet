"""Unit tests for bouquet.utils.pchip_derivative -- pure numpy/scipy, no OFT.

Covers the PCHIP-based dy/dx helper that replaces bouquet's
``np.gradient(y) / np.gradient(x)`` pattern for P'(psi_N), and its two-tier
data-sanity guard system:

  * Tier 1 (always ``ValueError``): non-finite input/output, a degenerate
    (<2 unique points) grid, and the shape-guarantee tripwire.
  * Tier 2 (``DerivativeSanityWarning`` by default, ``ValueError`` when
    ``strict=True``): dropped duplicate x points, unsorted input, and the
    derivative spike detector.
"""

import numpy as np
import pytest

from bouquet.utils import DerivativeSanityWarning, pchip_derivative


class TestMonotoneClean:
    """Well-behaved monotone input: correct slope, no Tier-2 noise."""

    def test_linear_no_warning_correct_slope(self, recwarn):
        x = np.linspace(0.0, 1.0, 101)
        y = 3.5 * x - 2.0
        d = pchip_derivative(x, y)
        sanity_warnings = [w for w in recwarn.list
                           if issubclass(w.category, DerivativeSanityWarning)]
        assert not sanity_warnings, [str(w.message) for w in sanity_warnings]
        assert d.shape == x.shape
        assert np.allclose(d, 3.5, atol=1e-8)

    def test_monotone_increasing_pedestal_like_no_tripwire(self):
        # Smooth monotone-increasing tanh profile (pedestal-like shape);
        # exercises the Tier-1.4 tripwire's "clean" path -- the invariant
        # holds, so no exception should fire.
        x = np.linspace(0.0, 1.0, 201)
        y = np.tanh(5 * (x - 0.5))
        d = pchip_derivative(x, y)
        assert np.all(np.isfinite(d))
        # PCHIP's shape-preservation: monotone-increasing y => d >= 0 everywhere
        assert np.all(d >= -1e-10)

    def test_x_eval_defaults_to_native_grid_shape(self):
        # Fine grid so PCHIP's shape-preserving slope estimate converges
        # tightly to the analytic derivative of the smooth quadratic (PCHIP
        # is not an exact-cubic interpolant -- it deliberately limits slopes
        # for shape-preservation, so this is an O(h^2)-converging check, not
        # an exact-reproduction one).
        x = np.linspace(0.0, 2.0, 2001)
        y = x ** 2
        d = pchip_derivative(x, y)
        assert d.shape == x.shape
        assert np.allclose(d, 2 * x, atol=1e-3)

    def test_x_eval_explicit_grid(self):
        x = np.linspace(0.0, 1.0, 2001)
        y = x ** 2
        x_eval = np.linspace(0.1, 0.9, 9)
        d = pchip_derivative(x, y, x_eval=x_eval)
        assert d.shape == x_eval.shape
        assert np.allclose(d, 2 * x_eval, atol=1e-3)


class TestDuplicateX:
    """Duplicated grid points: warned about (Tier 2), still fit smoothly."""

    def test_duplicate_x_warns_and_stays_smooth(self):
        x = np.linspace(0.0, 1.0, 101)
        y = 2.0 * x + 1.0
        # duplicate one interior point (as if two draws' psi_N grids got
        # concatenated with an overlap -- the suspected jphi_bs stepwise-artifact
        # trigger this guard exists for)
        x_dup = np.insert(x, 51, x[50])
        y_dup = np.insert(y, 51, y[50])

        with pytest.warns(DerivativeSanityWarning, match="duplicate"):
            d = pchip_derivative(x_dup, y_dup)

        # shape preserved (matches input grid, not the de-duplicated one)
        assert d.shape == x_dup.shape
        # still smooth / correct for this linear profile
        assert np.allclose(d, 2.0, atol=1e-6)

    def test_many_duplicates_over_5pct_uses_corrupted_wording(self):
        x = np.linspace(0.0, 1.0, 20)
        y = 2.0 * x
        # duplicate more than 5% of the points
        x_dup = np.concatenate([x, x[:3]])
        y_dup = np.concatenate([y, y[:3]])
        with pytest.warns(DerivativeSanityWarning, match="corrupted"):
            pchip_derivative(x_dup, y_dup)

    def test_duplicate_locations_reported_in_message(self):
        x = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
        y = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
        with pytest.warns(DerivativeSanityWarning) as record:
            pchip_derivative(x, y)
        msgs = " ".join(str(w.message) for w in record.list)
        assert "0.1" in msgs


class TestUnsortedX:
    def test_unsorted_x_warns_and_is_sorted_before_fit(self):
        x = np.array([0.0, 0.3, 0.1, 0.2, 0.4])
        y = np.array([0.0, 0.3, 0.1, 0.2, 0.4])  # y == x, slope should be 1
        with pytest.warns(DerivativeSanityWarning, match="not sorted"):
            d = pchip_derivative(x, y)
        assert np.allclose(d, 1.0, atol=1e-6)


class TestAxisSpike:
    """Synthetic axis-mapping-artifact spike: flat-ish profile, sharp jump
    over the innermost/outermost couple percent of the grid."""

    def test_edge_spike_triggers_spike_warning(self):
        n = 101
        x = np.linspace(0.0, 1.0, n)
        # Flat-ish (small nonzero slope) everywhere...
        y = 0.001 * x
        # ...except a sharp jump concentrated in the last ~2% of the grid,
        # several times steeper than the background slope and, because the
        # background is so flat, enormously larger than the *median* secant
        # slope used as the detector's reference scale.
        y = y.copy()
        y[-2:] = y[-3] + np.array([0.4, 0.8])

        with pytest.warns(DerivativeSanityWarning, match="unphysical"):
            d = pchip_derivative(x, y)
        assert np.all(np.isfinite(d))
        # the spike should show up near the edge where it was injected
        assert np.abs(d[-2:]).max() > 10 * np.abs(d[:80]).max()


class TestNonFiniteRaises:
    def test_nan_in_x_raises(self):
        x = np.array([0.0, 0.5, np.nan, 1.0])
        y = np.array([0.0, 0.5, 0.7, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            pchip_derivative(x, y)

    def test_inf_in_y_raises(self):
        x = np.array([0.0, 0.5, 0.75, 1.0])
        y = np.array([0.0, 0.5, np.inf, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            pchip_derivative(x, y)

    def test_degenerate_grid_raises(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="unique x point"):
            pchip_derivative(x, y)


class TestStrictMode:
    def test_strict_promotes_duplicate_warning_to_raise(self):
        x = np.linspace(0.0, 1.0, 21)
        y = 2.0 * x
        x_dup = np.insert(x, 10, x[9])
        y_dup = np.insert(y, 10, y[9])
        with pytest.raises(ValueError, match="duplicate"):
            pchip_derivative(x_dup, y_dup, strict=True)

    def test_strict_promotes_spike_warning_to_raise(self):
        n = 101
        x = np.linspace(0.0, 1.0, n)
        y = 0.001 * x
        y[-2:] = y[-3] + np.array([0.4, 0.8])
        with pytest.raises(ValueError, match="unphysical"):
            pchip_derivative(x, y, strict=True)

    def test_strict_leaves_clean_input_unaffected(self):
        x = np.linspace(0.0, 1.0, 51)
        y = 2.0 * x + 1.0
        d = pchip_derivative(x, y, strict=True)
        assert np.allclose(d, 2.0, atol=1e-8)
