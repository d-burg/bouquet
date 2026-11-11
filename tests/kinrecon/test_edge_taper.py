"""``bouquet.kinrecon.edge``: explicit edge P' taper, limit-cycle detector, auto policy.

Synthetic inputs only; no solver.  Hand calculations are written out next to each check.
"""
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import edge as E  # noqa: E402


# ------------------------------------------------------------------ taper geometry
def _hand_case():
    # P' = -2, -4, -10, -10 on psi_N = 0, 0.5, 0.99, 1 (piecewise linear); psi_scale = 2.
    # int_0^1 P' = 0.5*(-3) + 0.49*(-7) + 0.01*(-10) = -1.5 - 3.43 - 0.1 = -5.03
    # p(1) = 1; p(0.99) = 1 + 0.2 = 1.2; p(0.5) = 1.2 + 6.86 = 8.06; p(0) = 8.06 + 3 = 11.06
    x = np.array([0.0, 0.5, 0.99, 1.0])
    y = np.array([-2.0, -4.0, -10.0, -10.0])
    p = np.array([11.06, 8.06, 1.2, 1.0])
    return x, y, p


def test_knot_insertion_and_ramp():
    x, y, _ = _hand_case()
    t = E.apply_edge_taper(x, y, start=0.995)
    np.testing.assert_array_equal(t.psi_n, [0.0, 0.5, 0.99, 0.995, 1.0])
    np.testing.assert_array_equal(t.pprime, [-2.0, -4.0, -10.0, -10.0, 0.0])
    r = t.record
    assert r["applied"] and r["knot_inserted"] and r["convention"] == "edge_pinned"
    assert r["n_knots_in"] == 4 and r["n_knots_out"] == 5 and r["interior_knots_untouched"] == 3
    assert r["start"] == 0.995 and r["start_source"] == "given"
    assert r["width"] == pytest.approx(0.005, abs=1e-15)
    # removed = int_0.995^1 (-10 - (-10 (1-x)/0.005)) = -0.05 + 0.025 = -0.025
    assert r["integral_removed_in_layer"] == pytest.approx(-0.025, rel=1e-12)
    assert r["integral_pprime_dpsin_input"] == pytest.approx(-5.03, rel=1e-12)
    assert r["integral_pprime_dpsin_tapered"] == pytest.approx(-5.005, rel=1e-12)
    assert r["pax_ratio"] == pytest.approx(5.005 / 5.03, rel=1e-12)
    assert r["edge_over_max_abs_input"] == pytest.approx(1.0)


def test_existing_knot_is_not_duplicated():
    x = np.array([0.0, 0.5, 0.995, 0.998, 1.0])
    y = np.array([-1.0, -2.0, -6.0, -8.0, -9.0])
    t = E.apply_edge_taper(x, y, start=0.995)
    assert not t.record["knot_inserted"]
    np.testing.assert_array_equal(t.psi_n, x)
    # ramp from P'(0.995) = -6: at 0.998 -6 * 0.002/0.005 = -2.4, at 1 exactly 0
    np.testing.assert_allclose(t.pprime, [-1.0, -2.0, -6.0, -2.4, 0.0], rtol=1e-12, atol=0)
    assert t.pprime[-1] == 0.0


def test_interior_untouched_bitwise():
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 1.0, 257)
    y = -np.abs(rng.normal(size=257)) * (1 + 10 * x ** 8)
    p = 1.0 + np.cumsum(rng.random(257))[::-1]
    for start in (0.99, 0.995, 0.998):
        t = E.apply_edge_taper(x, y, p, start=start, domain_edge=0.99)
        k = int(np.sum(x < start))
        assert t.pprime[:k].tobytes() == y[:k].tobytes()
        assert t.psi_n[:k].tobytes() == x[:k].tobytes()
        assert np.all(t.psi_n[k:] >= start) and t.psi_n[-1] == 1.0 and t.pprime[-1] == 0.0
        # interior pressure: the input shifted by one constant
        d = t.pressure[:k] - p[:k]
        assert np.all(d == d[0]) or np.ptp(d) <= 1e-15 * np.max(np.abs(p))
    # the inputs themselves are never modified
    assert y.flags.writeable and np.all(np.isfinite(y))


def test_pressure_bookkeeping_hand_values():
    x, y, p = _hand_case()
    t = E.apply_edge_taper(x, y, p, start=0.995, psi_scale=2.0, psi_pedestal_top=0.99)
    r = t.record
    # edge-pinned: dp = psi_scale * removed = 2 * (-0.025) = -0.05 at every psi_N <= 0.995
    assert r["dp_inside"] == pytest.approx(-0.05, rel=1e-12)
    np.testing.assert_allclose(t.pressure, [11.01, 8.01, 1.15, 1.05, 1.0], rtol=1e-12)
    assert t.pressure[-1] == 1.0
    assert r["p_axis_before"] == 11.06
    assert r["dp_axis_rel"] == pytest.approx(-0.05 / 11.06, rel=1e-12)
    assert r["p_start_before"] == pytest.approx(1.1, rel=1e-12)     # 1 - 2*(-10*0.005)
    assert r["dp_start_rel"] == pytest.approx(-0.05 / 1.1, rel=1e-12)
    assert r["dp_pedestal_top"] == pytest.approx(-0.05) and r["p_pedestal_top_before"] == pytest.approx(1.2)
    assert r["dp_pedestal_top_rel"] == pytest.approx(-0.05 / 1.2, rel=1e-12)
    assert r["pressure_consistency_rel"] < 1e-14
    assert r["psi_scale_source"] == "given"


def test_psi_scale_inferred_from_pressure_and_recorded():
    x, y, p = _hand_case()
    t = E.apply_edge_taper(x, y, p, start=0.995)
    assert t.record["psi_scale"] == pytest.approx(2.0, rel=1e-12)       # (1 - 11.06) / (-5.03)
    assert t.record["psi_scale_source"].startswith("inferred")
    assert any("pedestal-top location not supplied" in n for n in t.record["notes"])


def test_ip_and_stored_energy_hand_values():
    x, y, p = _hand_case()
    v = 10.0 * x                                    # V(psi_N) = 10 psi_N -> 9.95 at the start knot
    t = E.apply_edge_taper(x, y, p, start=0.995, psi_scale=2.0, volume=v, ip=1.0)
    r = t.record
    # dI = psi_scale/(2 pi) int dP' dV: one interval, dV = 0.05, mean dP' = (0 + (-10))/2 = -5
    assert r["ip_removed"] == pytest.approx(2.0 / (2 * np.pi) * (-0.25), rel=1e-12)
    assert r["ip_fraction"] == pytest.approx(0.25 / np.pi, rel=1e-12)
    # dW = 1.5 [ -0.05 * 9.95 + trapz over the layer of dp (-0.05 -> 0) dV ] = 1.5 (-0.4975 - 0.00125)
    assert r["stored_energy_change"] == pytest.approx(-0.748125, rel=1e-12)
    assert r["stored_energy_rel"] == pytest.approx(r["stored_energy_change"] / r["stored_energy_before"])


def test_record_states_what_was_not_computed():
    x, y, _ = _hand_case()
    r = E.apply_edge_taper(x, y).record
    assert r["start"] == E.TAPER_START_DEFAULT and r["start_source"].startswith("default")
    assert r["domain_edge_source"] == "default"
    assert r["dp_inside"] is None and r["ip_fraction"] is None and r["stored_energy_rel"] is None
    assert any("no psi_scale and no pressure" in n for n in r["notes"])
    json.dumps(r)
    assert E.no_taper_record("converged without it") == dict(applied=False, reason="converged without it")


@pytest.mark.parametrize("start,edge", [(0.99, 0.995), (1.0, 0.995), (0.9995, 0.9995), (0.995, 0.98)])
def test_start_must_lie_in_the_outer_layer(start, edge):
    x, y, _ = _hand_case()
    with pytest.raises(ValueError):
        E.apply_edge_taper(x, y, start=start, domain_edge=edge)


def test_bad_knots_refused():
    with pytest.raises(ValueError):
        E.apply_edge_taper([0.0, 0.5, 0.99], [1.0, 1.0, 1.0])            # does not end at 1
    with pytest.raises(ValueError):
        E.apply_edge_taper([0.0, 0.7, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])  # not increasing


# ------------------------------------------------------------------ detector
def _transient(n=8, r0=1e-1, rate=0.3):
    return list(r0 * rate ** np.arange(n))


def test_detector_monotone_and_converged():
    r = 1e-2 * 0.5 ** np.arange(30)
    assert E.classify_residual_history(r)["verdict"] == "monotone_contraction"
    assert E.classify_residual_history(r, tol=1e-9)["verdict"] == "converged"


def test_detector_slow_is_not_a_cycle():
    slow = 1e-3 * 0.98 ** np.arange(40)          # 0.0088 decades / iteration
    d = E.classify_residual_history(slow, tol=1e-6)
    assert d["verdict"] == "slow_contraction"
    assert d["projected_iterations_to_tol"] > 0
    very_slow = 1e-4 * 0.995 ** np.arange(40)    # 0.0022 decades / iteration
    assert E.classify_residual_history(very_slow)["verdict"] in ("stalled", "slow_contraction")


def test_detector_period_2_and_3():
    p2 = _transient() + [1e-5, 3e-5] * 16
    d = E.classify_residual_history(p2[:40])
    assert d["verdict"] == "limit_cycle" and d["period"] == 2
    # the measured period-3 pattern shape (residual cycling ~1.2e-5 / 3.8e-5 / 3.4e-5)
    p3 = _transient(6) + [1.155e-5, 3.836e-5, 3.388e-5] * 12
    d = E.classify_residual_history(p3[:40])
    assert d["verdict"] == "limit_cycle" and d["period"] == 3
    assert d["amplitude_decades"] == pytest.approx(np.log10(3.836 / 1.155), rel=1e-9)


def test_detector_periods_4_to_6_and_smallest_period_wins():
    for p in (4, 5, 6):
        pat = list(1e-5 * 10 ** (0.4 * np.sin(2 * np.pi * np.arange(p) / p)))
        d = E.classify_residual_history(_transient(4) + pat * 8)
        assert d["verdict"] == "limit_cycle" and d["period"] == p, (p, d)
    # a period-2 pattern is also 4- and 6-periodic: the smallest period is reported
    assert E.classify_residual_history([1e-5, 4e-5] * 15)["period"] == 2


def test_detector_decaying_oscillation_is_not_a_cycle():
    k = np.arange(40)
    damped_trend = 1e-3 * 0.85 ** k * (1 + 0.5 * (-1.0) ** k)     # oscillating while contracting
    d = E.classify_residual_history(damped_trend)
    assert d["verdict"] == "decaying_oscillation"
    damped_amp = 1e-5 * 10 ** (0.3 * 0.7 ** k * (-1.0) ** k)     # amplitude dying out, flat mean
    d = E.classify_residual_history(damped_amp)
    assert d["verdict"] != "limit_cycle"
    # slowly contracting oscillation: still net progress beyond the drift bound
    slow_osc = 1e-4 * 0.96 ** k * (1 + 0.5 * (-1.0) ** k)
    assert E.classify_residual_history(slow_osc)["verdict"] != "limit_cycle"


def test_detector_noisy_histories():
    rng = np.random.default_rng(20261110)
    cyc = np.array(_transient(6) + [1.155e-5, 3.836e-5, 3.388e-5] * 12)[:40]
    noisy_cycle = cyc * (1 + 0.02 * rng.standard_normal(cyc.size))
    d = E.classify_residual_history(noisy_cycle)
    assert d["verdict"] == "limit_cycle" and d["period"] == 3
    noisy_slow = 1e-3 * 0.97 ** np.arange(40) * (1 + 0.03 * rng.standard_normal(40))
    assert E.classify_residual_history(noisy_slow)["verdict"] in ("slow_contraction", "decaying_oscillation")
    noise_only = 1e-5 * (1 + 0.03 * rng.standard_normal(40))
    assert E.classify_residual_history(noise_only)["verdict"] != "limit_cycle"


def test_detector_flat_residual_with_alternating_state_is_stalled():
    # residual flat to 1e-4 relative while the solver state alternates: not a residual cycle
    r = 3.979e-3 * (1 + 1e-4 * (-1.0) ** np.arange(40))
    assert E.classify_residual_history(r)["verdict"] == "stalled"


def test_detector_edge_cases():
    assert E.classify_residual_history([1e-3, 1e-4, 1e-5])["verdict"] == "insufficient_history"
    assert E.classify_residual_history([1e-3, np.nan, 1.0] + [1.0] * 10)["verdict"] == "diverging"
    assert E.classify_residual_history(1e-6 * 1.3 ** np.arange(30))["verdict"] == "diverging"
    with pytest.raises(TypeError):
        E.classify_residual_history([1.0] * 10, bogus=1)
    json.dumps(E.classify_residual_history(_transient() + [1e-5, 3e-5] * 16))


# ------------------------------------------------------------------ policy
CYCLE = _transient(6) + [1.155e-5, 3.836e-5, 3.388e-5] * 12
SLOW = list(1e-3 * 0.98 ** np.arange(40))
CONV = list(1e-2 * 0.4 ** np.arange(14))


class FakeSolve:
    """Returns a scripted outcome for untapered (None) and tapered (spec) calls."""

    def __init__(self, untapered, tapered):
        self.out = {False: untapered, True: tapered}
        self.calls = []

    def __call__(self, taper):
        self.calls.append(taper)
        conv, hist = self.out[taper is not None]
        return E.SolveAttempt(converged=conv, residuals=hist[:40], payload=("state", taper is not None))


def test_policy_auto_converged_untapered():
    f = FakeSolve((True, CONV), (True, CONV))
    o = E.solve_with_edge_taper(f)            # default setting: auto
    assert o.record["mode"] == "auto" and o.record["status"] == E.STATUS_CONVERGED
    assert f.calls == [None] and not o.record["taper_applied"] and o.payload == ("state", False)
    json.dumps(o.record)


def test_policy_auto_limit_cycle_then_tapered():
    f = FakeSolve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(f, E.EdgeTaperSetting("auto"))
    assert o.record["status"] == E.STATUS_CONVERGED_TAPER and o.record["taper_applied"]
    # the policy's default taper is the pressure-conserving one with the default band
    default = E.EdgeTaperSpec(variant="pressure_conserving")
    assert E.EdgeTaperSetting().spec == default and E.BAND_WIDTH_DEFAULT == 0.02
    assert f.calls[0] is None and f.calls[1] == default
    assert o.record["detector_verdict_untapered"] == "limit_cycle"
    assert len(o.record["attempts"]) == 2 and o.payload == ("state", True)


def test_policy_auto_limit_cycle_tapered_also_fails_keeps_both_histories():
    f = FakeSolve((False, CYCLE), (False, SLOW))
    o = E.solve_with_edge_taper(f)
    assert o.record["status"] == E.STATUS_FAILED and o.record["taper_applied"]
    a, b = o.record["attempts"]
    assert a["history"]["n"] == 40 and b["history"]["n"] == 40
    assert a["detector"]["verdict"] == "limit_cycle" and b["detector"]["verdict"] == "slow_contraction"


@pytest.mark.parametrize("hist", [SLOW, [3.979e-3] * 40])
def test_policy_auto_other_failure_is_not_tapered(hist):
    f = FakeSolve((False, hist), (True, CONV))
    o = E.solve_with_edge_taper(f)
    assert o.record["status"] == E.STATUS_FAILED and not o.record["taper_applied"]
    assert f.calls == [None] and "taper not applied" in o.record["reason"]


def test_policy_off_and_on():
    f = FakeSolve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(f, E.EdgeTaperSetting(E.EdgeTaperMode.OFF))
    assert o.record["status"] == E.STATUS_FAILED and f.calls == [None]
    g = FakeSolve((True, CONV), (True, CONV))
    spec = E.EdgeTaperSpec(start=0.998)
    o = E.solve_with_edge_taper(g, E.EdgeTaperSetting("on", spec))
    assert o.record["status"] == E.STATUS_CONVERGED_TAPER and g.calls == [spec]
    assert o.record["setting"]["start"] == 0.998
    h = FakeSolve((True, CONV), (False, SLOW))
    assert E.solve_with_edge_taper(h, E.EdgeTaperSetting("on")).record["status"] == E.STATUS_FAILED


def test_case_uniformity_tapered_lock():
    case = E.EdgeTaperCase("caseA")
    ref = FakeSolve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(ref, case=case, label="reference")
    assert o.record["status"] == E.STATUS_CONVERGED_TAPER
    assert case.locked is True and case.decided_by == "reference" and case.taper_for_later_solves() == E.EdgeTaperSetting().spec
    # a later draw that would converge untapered still gets the same taper, with no untapered attempt
    draw = FakeSolve((True, CONV), (True, CONV))
    o2 = E.solve_with_edge_taper(draw, case=case, label="draw_001")
    assert draw.calls == [E.EdgeTaperSetting().spec] and o2.record["status"] == E.STATUS_CONVERGED_TAPER
    assert o2.record["case_lock"]["locked"] is True
    assert [h["label"] for h in case.history] == ["reference", "draw_001"]
    json.dumps(case.as_dict())


def test_case_uniformity_untapered_lock_and_undecided_after_failure():
    case = E.EdgeTaperCase("caseB")
    E.solve_with_edge_taper(FakeSolve((True, CONV), (True, CONV)), case=case, label="reference")
    assert case.locked is False and case.taper_for_later_solves() is None
    draw = FakeSolve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(draw, case=case, label="draw_001")
    assert draw.calls == [None] and o.record["status"] == E.STATUS_FAILED and not o.record["taper_applied"]
    # a failed reference leaves the case undecided
    c2 = E.EdgeTaperCase("caseC")
    E.solve_with_edge_taper(FakeSolve((False, SLOW), (True, CONV)), case=c2, label="reference")
    assert c2.locked is None
    with pytest.raises(ValueError):
        E.solve_with_edge_taper(FakeSolve((True, CONV), (True, CONV)), E.EdgeTaperSetting("on"), case=c2)


def test_policy_accepts_mapping_results():
    def solve(taper):
        return dict(converged=taper is not None, residuals=CYCLE[:40] if taper is None else CONV, info={"its": 7})
    o = E.solve_with_edge_taper(solve)
    assert o.record["status"] == E.STATUS_CONVERGED_TAPER
    assert o.record["attempts"][1]["info"] == {"its": 7}


# ------------------------------------------------------------------ variants: last_node
def test_last_node_is_pprime_last_zero_and_equals_explicit_taper_at_second_to_last_node():
    x, y, p = _hand_case()
    t = E.apply_edge_taper(x, y, p, variant="last_node", psi_scale=2.0)
    ref = y.copy()
    ref[-1] = 0.0
    assert t.pprime.tobytes() == ref.tobytes() and t.psi_n.tobytes() == x.tobytes()
    r = t.record
    assert r["variant"] == "last_node" and r["start"] == 0.99 and r["effective_width"] == pytest.approx(0.01)
    assert not r["knot_inserted"] and not r["within_outer_layer"]
    assert any("not confined to the outer layer" in n for n in r["notes"])
    # removed = int_0.99^1 (-10 - (-10 (1-x)/0.01)) = -0.1 + 0.05 = -0.05; dp = 2 * -0.05
    assert r["integral_removed_in_layer"] == pytest.approx(-0.05, rel=1e-12)
    assert r["dp_inside"] == pytest.approx(-0.1, rel=1e-12)
    e = E.apply_edge_taper(x, y, p, start=0.99, domain_edge=0.99, psi_scale=2.0)
    np.testing.assert_array_equal(e.pprime, t.pprime)
    for k in ("integral_removed_in_layer", "pax_ratio", "dp_inside", "dp_axis_rel"):
        assert e.record[k] == pytest.approx(r[k], rel=1e-14), k
    with pytest.raises(ValueError):
        E.apply_edge_taper(x, y, variant="last_node", start=0.995)


def test_last_node_strength_depends_on_grid_spacing():
    removed = []
    for n in (65, 129, 257):
        x = np.linspace(0.0, 1.0, n)
        removed.append(E.apply_edge_taper(x, -np.ones(n), variant="last_node").record["integral_removed_in_layer"])
    # constant P' = -1: removed = -(last interval)/2
    np.testing.assert_allclose(removed, [-0.5 / 64, -0.5 / 128, -0.5 / 256], rtol=1e-12)


# ------------------------------------------------------------------ variants: pressure_conserving
def _flat_case(n=201):
    # P' = -10 everywhere, psi_scale = 1, p = 1 + 10 (1 - x): p_sep = 1
    x = np.linspace(0.0, 1.0, n)
    y = np.full(n, -10.0)
    return x, y, 1.0 + 10.0 * (1.0 - x)


@pytest.mark.parametrize("w", [0.005, 0.01, 0.02])
def test_pressure_conserving_hat_and_conservation(w):
    x, y, p = _flat_case()
    t = E.apply_edge_taper(x, y, p, start=0.995, psi_scale=1.0, variant="pressure_conserving", band_width=w,
                           volume=10.0 * x, ip=1.0, r_avg=np.full_like(x, 2.0), psi_pedestal_top=0.9)
    r, b = t.record, t.record["band"]
    # R = int_0.995^1 (-10 + 10 (1-x)/0.005) = -0.025; hat peak A = 2R/w; integral added = R
    assert r["integral_removed_in_layer"] == pytest.approx(-0.025, rel=1e-9)
    assert b["hat_peak"] == pytest.approx(-0.05 / w, rel=1e-9)
    assert b["integral_added_in_band"] == pytest.approx(-0.025, rel=1e-9)
    assert b["hat_peak_over_max_abs_pprime_in_band"] == pytest.approx((0.05 / w) / 10.0, rel=1e-9)
    assert not b["pprime_changes_sign_in_band"]
    b0, bm = 0.995 - w, 0.995 - w / 2
    assert np.isclose(t.psi_n, b0).any() and np.isclose(t.psi_n, bm).any()
    np.testing.assert_allclose(np.interp(bm, t.psi_n, t.pprime), -10.0 - 0.05 / w, rtol=1e-9)
    # untouched inside the band, bit for bit; continuous single-signed P'
    k = r["interior_knots_untouched"]
    assert t.pprime[:k].tobytes() == y[:k].tobytes() and np.all(t.pprime[:-1] < 0) and t.pprime[-1] == 0.0
    # pressure inside the band, on axis and at psi_N = 1 unchanged (to round-off); pax_ratio 1
    assert abs(r["dp_inside"]) < 1e-12 and abs(r["dp_pedestal_top_rel"]) < 1e-12
    assert t.pressure[-1] == p[-1] and abs(r["pax_ratio"] - 1.0) < 1e-12
    np.testing.assert_allclose(t.pressure[:k], p[:k], rtol=0, atol=1e-12)
    # W changes only through the band + layer: 1.5 * 10 * (R w/2 + int over the layer)
    #   = 1.5 * 10 * (-0.0125 w - 8.333e-5); the trapezoid on the knots approximates the
    #   piecewise-quadratic pressure change, hence the 25 % comparison (bookkeeping, not physics)
    expect = 15.0 * (-0.0125 * w - (5 * 0.005 ** 2 - 1000 * 0.005 ** 3 / 3))
    assert r["stored_energy_change"] == pytest.approx(expect, rel=0.25)
    # <R> P' change: largest at psi_N = 1 (2 * 10), before |<R> P'| max 2 * 10
    assert r["edge_current"]["max_abs_change"] == pytest.approx(20.0, rel=1e-12)
    assert r["band"]["band_width_source"] == "given"


def test_pressure_conserving_default_band_and_spec():
    x, y, p = _flat_case()
    t = E.apply_edge_taper(x, y, p, variant="pressure_conserving")
    assert t.record["band"]["band_width"] == E.BAND_WIDTH_DEFAULT
    assert t.record["band"]["band_width_source"].startswith("default")
    s = E.EdgeTaperSpec(variant="pressure_conserving", band_width=0.01)
    assert s.as_dict()["band_width"] == 0.01 and s.kwargs()["band_width"] == 0.01
    assert E.EdgeTaperSpec(variant="last_node").as_dict()["start"].startswith("grid")
    with pytest.raises(ValueError):
        E.EdgeTaperSpec(band_width=0.01)                      # band only for pressure_conserving
    with pytest.raises(ValueError):
        E.EdgeTaperSpec(variant="bogus")
    with pytest.raises(ValueError):
        E.apply_edge_taper(x, y, variant="pressure_conserving", band_width=0.0)


def test_policy_passes_the_variant_through():
    f = FakeSolve((False, CYCLE), (True, CONV))
    spec = E.EdgeTaperSpec(variant="pressure_conserving", band_width=0.01)
    o = E.solve_with_edge_taper(f, E.EdgeTaperSetting("auto", spec))
    assert f.calls[1] == spec and o.record["attempts"][1]["taper"]["variant"] == "pressure_conserving"
