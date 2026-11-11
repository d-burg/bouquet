"""Named limit-cycle detector versions and the stall policy of ``bouquet.kinrecon.edge`` (synthetic histories).

``v1`` is the default and is pinned here unchanged, including its known weaknesses (an independent review's
adversarial histories); ``v2`` is the corrected classification (non-default, awaiting a decision): a slowly
diverging oscillation is not a cycle, and periods up to 12 are tested.  ``stall_policy='taper'`` (non-default)
treats a flat stall as taper-eligible.
"""
import numpy as np
import pytest

from bouquet.kinrecon import edge as E

K = np.arange(40)
H = {
    "slow_converging_p3": 10 ** (-5 - 0.004 * K + 0.15 * np.cos(2 * np.pi * K / 3)),
    "slow_diverging_p3": 10 ** (-5 + 0.004 * K + 0.15 * np.cos(2 * np.pi * K / 3)),
    "p7": 10 ** (-5 + 0.3 * np.sin(2 * np.pi * K / 7)),
    "p8": 10 ** (-5 + 0.3 * np.sin(2 * np.pi * K / 8)),
    "quasi_3_4": 10 ** (-5 + 0.3 * np.array(([1, -1, 0] + [1, -1, 0, 0.5]) * 6)[:40]),
    "growing_p3": 10 ** (-5 + 0.1 * 1.3 ** (K / 3) * np.cos(2 * np.pi * K / 3)),
    "flat_stall": np.r_[10 ** (-3 - 0.1 * np.arange(25)), np.full(15, 10 ** -5.4)],
}
V1 = {"slow_converging_p3": "limit_cycle", "slow_diverging_p3": "limit_cycle", "p7": "stalled", "p8": "stalled",
      "quasi_3_4": "slow_contraction", "growing_p3": "diverging", "flat_stall": "stalled"}
V2 = {"slow_converging_p3": "limit_cycle", "slow_diverging_p3": "diverging", "p7": "limit_cycle", "p8": "limit_cycle",
      "quasi_3_4": "limit_cycle", "growing_p3": "diverging", "flat_stall": "stalled"}


@pytest.mark.parametrize("name", sorted(H))
def test_v1_default_is_unchanged(name):
    d = E.classify_residual_history(H[name], tol=1e-6)
    assert d["version"] == "v1" and d["verdict"] == V1[name]
    assert E.classify_residual_history(H[name], tol=1e-6, version="v1")["verdict"] == V1[name]


@pytest.mark.parametrize("name", sorted(H))
def test_v2_corrected_classification(name):
    d = E.classify_residual_history(H[name], tol=1e-6, version="v2")
    assert d["verdict"] == V2[name], (name, d["verdict"])
    if name in ("p7", "p8"):
        assert d["period"] == int(name[1:])
    if name == "quasi_3_4":
        assert d["period"] == 7


def test_unknown_version_refused():
    with pytest.raises(ValueError):
        E.classify_residual_history(H["p7"], version="v9")


def _policy(history, converged_tapered, **kw):
    def solve(taper):
        if taper is None:
            return E.SolveAttempt(converged=False, residuals=history)
        return E.SolveAttempt(converged=converged_tapered, residuals=[1e-7])
    return E.solve_with_edge_taper(solve, E.EdgeTaperSetting(), tol=1e-6, **kw).record


def test_stall_policy_default_reports_and_option_tapers():
    rec = _policy(H["flat_stall"], True)
    assert rec["status"] == E.STATUS_FAILED and not rec["taper_applied"] and "nondefault_options" not in rec
    rec = _policy(H["flat_stall"], True, stall_policy="taper")
    assert rec["status"] == E.STATUS_CONVERGED_TAPER and rec["taper_applied"]
    assert rec["nondefault_options"]["stall_policy"] == "taper" and "stalled" in rec["reason"]
    with pytest.raises(ValueError):
        _policy(H["flat_stall"], True, stall_policy="maybe")


def test_v2_never_tapers_a_diverging_oscillation():
    rec = _policy(H["slow_diverging_p3"], True)                          # v1: tapered (the reviewed defect)
    assert rec["taper_applied"]
    rec = _policy(H["slow_diverging_p3"], True, detector_kw=dict(version="v2"))
    assert not rec["taper_applied"] and rec["status"] == E.STATUS_FAILED
    assert rec["nondefault_options"]["detector_version"] == "v2"
