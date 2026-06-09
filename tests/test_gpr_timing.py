"""
Timing benchmark: re-draw path vs repeated generate_perturbed_GPR calls.

Run with::

    pytest tests/test_gpr_timing.py -v -s

or directly::

    python -m pytest tests/test_gpr_timing.py -v -s --tb=short

The benchmark is not a correctness test, it always passes.  Its
purpose is to show the wall-clock speedup from amortising the O(n³)
eigendecomposition with ``precompute_factor`` + ``draw_from_factor``
versus calling ``generate_perturbed_GPR`` (which does a fresh linalg.eigh on
every call).
"""

import time
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any other plt import
import numpy as np
import pytest

from bouquet.sampling import GPRProfilePerturber, generate_perturbed_GPR, verify_gpr_statistics

# ====================================================================
#  Shared fixtures
# ====================================================================

N_GRID   = 257        # profile grid points
N_DRAWS  = 200       # number of samples to generate in each benchmark
LENGTH   = 0.25      # GPR correlation length
SIGMA    = 0.05      # flat fractional uncertainty

N_VERIFY = 2000      # samples for verify_gpr_statistics (reduced from default 5000
                     # to keep the test fast; increase for publication-quality checks)


@pytest.fixture(scope="module")
def grid():
    psi_N   = np.linspace(0, 1, N_GRID)
    profile = 1.0 - psi_N
    sigma   = SIGMA * np.ones_like(psi_N)
    return psi_N, profile, sigma


# ====================================================================
#  Benchmarks
# ====================================================================

class TestGPRTiming:
    """Wall-clock comparison between the two sampling strategies."""

    def test_timing_comparison(self, grid, capsys):
        """Print a timing table; always passes."""
        psi_N, profile, sigma = grid

        # ---- Method A: re-draw (precompute_factor + draw_from_factor) ----
        perturber = GPRProfilePerturber(kernel_func="rbf", length_scale=LENGTH)

        rng_a = np.random.default_rng(0)
        t0 = time.perf_counter()
        perturber.precompute_factor(psi_N, sigma)   # O(n³) — included in total
        for _ in range(N_DRAWS):
            perturber.draw_from_factor(profile, 1, rng_a)
        t_redraw = time.perf_counter() - t0

        # ---- Method B: repeated generate_perturbed_GPR (fresh eigh each call) ----
        rng_b = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(N_DRAWS):
            generate_perturbed_GPR(
                psi_N, profile,
                sigma_profile=sigma,
                length_scale=LENGTH,
                n_samples=1,
                rng=rng_b,
            )
        t_repeated = time.perf_counter() - t0

        speedup = t_repeated / t_redraw if t_redraw > 0 else float("inf")

        with capsys.disabled():
            print(f"\n{'='*55}")
            print(f"  GPR timing  ({N_DRAWS} draws, {N_GRID}-point grid)")
            print(f"{'='*55}")
            print(f"  Re-draw  (precompute_factor + {N_DRAWS}× draw_from_factor):")
            print(f"    total : {t_redraw*1e3:.1f} ms")
            print(f"    per   : {t_redraw/N_DRAWS*1e3:.3f} ms/draw (amortised)")
            print(f"  Repeated generate_perturbed_GPR (fresh eigh each):")
            print(f"    total : {t_repeated*1e3:.1f} ms")
            print(f"    per   : {t_repeated/N_DRAWS*1e3:.3f} ms/draw")
            print(f"  Speedup : {speedup:.1f}×")
            print(f"{'='*55}")

        # Sanity: re-draw should be at least 2× faster for N_DRAWS >= 10
        if N_DRAWS >= 10:
            assert speedup >= 2.0, (
                f"Expected re-draw to be ≥2× faster; got {speedup:.2f}×. "
                "This may indicate the eigendecomposition is not being cached."
            )

    def test_timing_batch_vs_loop(self, grid, capsys):
        """Batch draw (n_samples > 1) vs loop of single draws for pre_computed factor (minor speedup)"""
        psi_N, profile, sigma = grid

        perturber = GPRProfilePerturber(kernel_func="rbf", length_scale=LENGTH)

        rng_a = np.random.default_rng(1)
        t0 = time.perf_counter()
        perturber.precompute_factor(psi_N, sigma)   # O(n³) — included in total
        _ = perturber.draw_from_factor(profile, N_DRAWS, rng_a)
        t_batch = time.perf_counter() - t0

        perturber2 = GPRProfilePerturber(kernel_func="rbf", length_scale=LENGTH)
        rng_b = np.random.default_rng(1)
        t0 = time.perf_counter()
        perturber2.precompute_factor(psi_N, sigma)  # O(n³) — included in total
        for _ in range(N_DRAWS):
            perturber2.draw_from_factor(profile, 1, rng_b)
        t_loop = time.perf_counter() - t0

        with capsys.disabled():
            print(f"\n{'='*55}")
            print(f"  Batch vs loop  ({N_DRAWS} draws, {N_GRID}-point grid)")
            print(f"{'='*55}")
            print(f"  Batch draw_from_factor(n={N_DRAWS}) + precompute: {t_batch*1e3:.2f} ms")
            print(f"  Loop  draw_from_factor(n=1)×{N_DRAWS} + precompute: {t_loop*1e3:.2f} ms")
            print(f"  Ratio batch/loop: {t_batch/t_loop:.2f}")
            print(f"{'='*55}")

    def test_timing_grid_scaling(self, capsys):
        """Show how timing scales with grid size for precomputed factor vs generate_perturbed_GPR."""
        grid_sizes = [32, 64, 128, 256]
        n_draws = 1000

        rows = []
        for n in grid_sizes:
            psi_N   = np.linspace(0, 1, n)
            profile = 1.0 - psi_N
            sigma   = SIGMA * np.ones(n)

            # Re-draw
            p = GPRProfilePerturber(kernel_func="rbf", length_scale=LENGTH)
            rng = np.random.default_rng(0)
            t0 = time.perf_counter()
            p.precompute_factor(psi_N, sigma)  # O(n³) — included in total
            for _ in range(n_draws):
                p.draw_from_factor(profile, 1, rng)
            t_rd = (time.perf_counter() - t0) / n_draws * 1e3  # ms/draw

            # Repeated
            rng2 = np.random.default_rng(0)
            t0 = time.perf_counter()
            for _ in range(n_draws):
                generate_perturbed_GPR(psi_N, profile,
                                       sigma_profile=sigma,
                                       length_scale=LENGTH,
                                       n_samples=1, rng=rng2)
            t_rep = (time.perf_counter() - t0) / n_draws * 1e3

            rows.append((n, t_rd, t_rep, t_rep / t_rd if t_rd > 0 else float("inf")))

        with capsys.disabled():
            print(f"\n{'='*55}")
            print(f"  Grid-size scaling  ({n_draws} draws each)")
            print(f"  {'n':>5}  {'redraw ms':>10}  {'repeated ms':>12}  {'speedup':>8}")
            print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*8}")
            for n, rd, rep, sp in rows:
                print(f"  {n:>5}  {rd:>10.3f}  {rep:>12.3f}  {sp:>8.1f}×")
            print(f"{'='*55}")


# ====================================================================
#  Statistical verification (both sampling paths)
# ====================================================================

class TestVerifyGPRStatistics:
    """Run verify_gpr_statistics to cross-check re-draw vs generate_profiles.

    ``plt.show`` is patched to a no-op so this runs non-interactively.
    The test passes only if the two paths agree to within 2 %.
    """

    def test_verify_prints_and_agrees(self, capsys):
        psi_N   = np.linspace(0, 1, N_GRID)
        profile = 1.0 - psi_N
        sigma   = SIGMA * np.ones_like(psi_N)

        with patch("matplotlib.pyplot.show"):
            result = verify_gpr_statistics(
                psi_N, profile, sigma,
                length_scale=LENGTH,
                n_verification=N_VERIFY,
                confidence_band=2.0,
            )

        # Check return structure
        assert "stats_a" in result
        assert "stats_b"  in result
        assert "sigma_theory" in result
        assert "theoretical_exceedance" in result

        # Both paths must have avg_exceedance close to the Gaussian prediction
        from scipy.stats import norm
        theory = 2.0 * norm.sf(2.0)
        for key in ("stats_a", "stats_b"):
            exc = result[key]["avg_exceedance"]
            assert abs(exc - theory)/theory < 0.02, (
                f"{key} avg_exceedance {exc:.4f} deviates from theory "
                f"{theory:.4f} by more than 0.02"
            )

        with capsys.disabled():
            print()  # verify_gpr_statistics already printed its own report
