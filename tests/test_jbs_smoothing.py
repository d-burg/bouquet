"""Unit tests for smooth_jbs_transition -- pure numpy, no OFT/solver required.

The helper is the single shared repair for solve_with_bootstrap's numerically
fragile innermost-surface point: it must (a) lift a collapsed axis point
toward its neighbours, (b) leave everything beyond the +/-10-index transition
window bit-untouched, (c) preserve a flat shelf exactly, and -- the reason it
exists as a shared helper at all -- (d) give identical output for identical
input regardless of caller, so the recon j_BS/j_inductive split and every
per-draw spike stay sigma=0-consistent (the 2026-07 hollow-core/q0-offset bug
was the recon applying this smoothing while the draw path did not).
"""

import numpy as np

from bouquet.TokaMaker_interface import smooth_jbs_transition


def _sauter_like_profile(npsi=129, axis_collapse=0.55):
    """Sauter-hump-shaped j_BS with a collapsed first point (the SWB axis
    artifact: ~2x below its neighbours), core hump near psi_N~0.05 and an
    edge spike."""
    psi = np.linspace(0.0, 1.0, npsi)
    hump = 5.5e5 * np.exp(-0.5 * ((psi - 0.05) / 0.10) ** 2)
    tail = 1.5e5 * np.exp(-(psi - 0.05) / 0.35) * (psi > 0.05)
    spike = 4.0e5 * np.exp(-0.5 * ((psi - 0.97) / 0.015) ** 2)
    j = hump + tail + spike + 5.0e4
    j[0] *= axis_collapse
    return psi, j


def test_axis_point_lifted_toward_neighbours():
    _, j = _sauter_like_profile()
    out = smooth_jbs_transition(j)
    # the collapsed point must be pulled up, most of the way to its neighbours
    assert out[0] > j[0]
    assert abs(out[0] - j[1]) < abs(j[0] - j[1])
    # and the repair must not overshoot the local maximum
    assert out[0] < j[1:11].max()


def test_untouched_beyond_window():
    _, j = _sauter_like_profile()
    out = smooth_jbs_transition(j)
    # window is indices 0..10 (shelf end 0, half-width 10): beyond it,
    # including the edge spike, the profile is bit-identical
    assert np.array_equal(out[11:], j[11:])


def test_flat_shelf_preserved():
    psi = np.linspace(0.0, 1.0, 129)
    j = np.full(129, 2.0e5)
    j += 4.0e5 * np.exp(-0.5 * ((psi - 0.97) / 0.015) ** 2)  # edge spike
    j[:40] = 2.0e5                                            # exact flat shelf
    out = smooth_jbs_transition(j)
    # a flat shelf Gaussian-filters to itself: shelf values survive exactly
    # (this is the isolate_edge_jBS+shelf recon configuration)
    np.testing.assert_allclose(out[:30], 2.0e5, rtol=1e-12)


def test_classifier_insensitive_to_axis_treatment():
    """With the valley height reference (min of the profile between the
    edge-window start and the edge peak, mirroring
    analyze_bootstrap_edge_spike), classification no longer depends on the
    numerically fragile axis point -- raw and axis-smoothed profiles must
    classify IDENTICALLY.  (Before the hardening, spike[0] was the height
    bar: the smoothed profile's lifted axis hid comparable-height edge peaks
    -- an L_mode flip on a real discharge -- and raw-axis jitter flipped
    weak-pedestal discharges.)"""
    from bouquet.TokaMaker_interface import classify_jphi_profile

    psi = np.linspace(0.0, 1.0, 129)
    # edge peak (~0.28) comparable to the core hump, collapsed raw axis point
    core = 3.6e5 * np.exp(-0.5 * ((psi - 0.05) / 0.12) ** 2) + 4.0e4
    edge = 2.4e5 * np.exp(-0.5 * ((psi - 0.93) / 0.02) ** 2)
    raw = core + edge
    raw[0] *= 0.45
    smoothed = smooth_jbs_transition(raw)
    # g-file WITH a matching pedestal peak -> H_mode
    jphi = (0.75e6 * (1.0 - psi) ** 1.5 + 5.0e4
            + 2.0e5 * np.exp(-0.5 * ((psi - 0.93) / 0.02) ** 2))

    mode_raw, _ = classify_jphi_profile(psi, jphi, raw)
    mode_sm, _ = classify_jphi_profile(psi, jphi, smoothed)
    assert mode_raw == "H_mode"
    assert mode_sm == mode_raw           # axis treatment no longer matters


def test_weak_pedestal_peak_detected_via_valley():
    """A weak-pedestal geometry: pedestal peak (~0.108 MA/m^2) comparable to
    the collapsed axis point (~0.110), but prominent above the pedestal-foot
    VALLEY (~0.05).  The valley reference must detect it as a real Sauter
    edge spike (metrics carry its position) instead of falling through to
    the no-edge-peak fallback."""
    from bouquet.TokaMaker_interface import classify_jphi_profile

    psi = np.linspace(0.0, 1.0, 257)
    # monotone mantle decay + weak pedestal bump, collapsed axis point
    prof = (3.5e5 * np.exp(-0.5 * ((psi - 0.05) / 0.20) ** 2) + 4.0e4
            + 6.0e4 * np.exp(-0.5 * ((psi - 0.97) / 0.015) ** 2))
    prof[0] *= 0.5
    jphi = 1.1e6 * (1.0 - psi) ** 1.5 + 5.0e4   # no geqdsk edge peak

    mode, met = classify_jphi_profile(psi, jphi, prof)
    assert mode == "Lmode_like_jphi"
    assert met["spike_psiN_sauter"] is not None
    assert met["spike_psiN_sauter"] > 0.9        # the pedestal bump, found


def test_core_hump_only_profile_is_not_L_mode():
    """A Sauter profile with real bootstrap current but no DETECTED edge peak
    must NOT classify as L_mode: that zeroes the whole split and
    double-counts the bootstrap when the per-draw SWB recompute adds the
    profile back. (Observed on a weak-pedestal discharge, whose peak sits
    within ~2 permille of the height threshold -- the fragile collapsed axis
    point -- so detection flips on run-to-run jitter.) It should fall through
    to Lmode_like_jphi (full profile kept in the split). A truly negligible
    profile still classifies L_mode."""
    from bouquet.TokaMaker_interface import classify_jphi_profile

    psi = np.linspace(0.0, 1.0, 129)
    jphi = 1.0e6 * (1.0 - psi) ** 1.5 + 5.0e4
    # core-hump-only: monotone decay through the edge window, no pedestal peak
    hump = 4.2e5 * np.exp(-0.5 * ((psi - 0.05) / 0.15) ** 2) + 2.0e4
    mode, met = classify_jphi_profile(psi, jphi, hump)
    assert mode == "Lmode_like_jphi"
    assert met["spike_height_sauter"] > 0

    # negligible bootstrap -> still L_mode
    mode2, _ = classify_jphi_profile(psi, jphi, hump * 1e-3)
    assert mode2 == "L_mode"


def test_pchip_regrid_removes_staircase():
    """pchip_interp is the single shared kin->eq regrid. Requirements:
    (1) passes through the source knot values exactly (no data alteration),
    (2) clamps to endpoint values outside the source range,
    (3) its derivative is dramatically smoother than the linear regrid's --
    the linear version's slope kinks at every kinetic knot are what Sauter
    inherited as the stepped j_BS on a real weak-pedestal discharge."""
    from bouquet.utils import pchip_interp
    from scipy.interpolate import PchipInterpolator

    rng = np.random.default_rng(0)
    x_kin = np.sort(np.concatenate([[0.0], rng.uniform(0.0, 1.2, 88), [1.2]]))
    y_kin = 5.0e3 * np.exp(-2.2 * x_kin) + 40.0           # smooth Te-like
    x_eq = np.linspace(0.0, 1.0, 257)

    out = pchip_interp(x_kin, y_kin, x_eq)
    # knot passthrough
    np.testing.assert_allclose(pchip_interp(x_kin, y_kin, x_kin), y_kin,
                               rtol=1e-12)
    # endpoint clamping
    assert pchip_interp(x_kin, y_kin, np.array([-0.5]))[0] == y_kin[0]
    assert pchip_interp(x_kin, y_kin, np.array([2.0]))[0] == y_kin[-1]
    # gradient fidelity vs the analytic derivative: the linear regrid's
    # staircase derivative carries ~4x the RMS error of the PCHIP regrid
    d_true = -2.2 * 5.0e3 * np.exp(-2.2 * x_eq)
    lin = np.interp(x_eq, x_kin, y_kin)
    err = lambda p: np.sqrt(np.mean((np.gradient(p, x_eq) - d_true) ** 2))
    assert err(out) < 0.5 * err(lin)


def test_deterministic_across_callers():
    # the sigma=0 contract: identical input -> identical output, no state
    _, j = _sauter_like_profile()
    a = smooth_jbs_transition(j)
    b = smooth_jbs_transition(j.copy())
    assert np.array_equal(a, b)
    # and input is never mutated
    _, j2 = _sauter_like_profile()
    np.testing.assert_array_equal(j, j2)
