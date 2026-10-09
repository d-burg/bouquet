"""The self-consistent bootstrap kernel refuses what it cannot iterate.

* A non-finite initial guess or evaluated bootstrap raises
  :class:`bouquet.jbs_loop.JBSNonFinite` AT ONCE -- with the pass number and
  the location -- before it can be blended into the next iterate and handed
  to a GS solve, whatever the ``"flag"`` policy says.
* A pass that can never count toward convergence (a gated l_i or q0 the step
  did not return; an identically zero Redl bootstrap against a non-zero
  iterate, ``r_j = inf``) stops the loop at that pass instead of burning the
  pass ceiling, and then follows the ordinary failure policy.
* The unrelaxed closure-half residual ``||jc_k - js_k-1|| / ||jc_k||`` is
  recorded next to the relaxed gap (record only; nothing gates on it).

Synthetic fixed-point maps only; no solver, no device data.
"""
import json

import numpy as np
import pytest

from bouquet.jbs_loop import (JBSNonFinite, JBSNotConverged, jbs_settings,
                              jsonable, run_jbs_loop)
from test_jbs_loop import _GC, _IP, _X, _affine_problem, _two_state_problem


def _counting(step):
    n = {"c": 0}

    def st(jbs, k):
        n["c"] += 1
        return step(jbs, k)
    return st, n


# ---------------------------------------------------------------------------
#  non-finite input / evaluation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_initial_guess_is_refused_before_any_solve(bad):
    Jstar, step, ev = _affine_problem(-0.1)
    st, n = _counting(step)
    j0 = 0.5 * Jstar
    j0[12] = bad
    with pytest.raises(JBSNonFinite, match=r"initial bootstrap guess.*before "
                                           r"pass 1.*index 12") as ei:
        run_jbs_loop(j0, st, ev, jbs_settings(_GC()), Ip=_IP,
                     meas0=dict(li=0.0))
    assert n["c"] == 0
    assert ei.value.pass_number == 0 and ei.value.index == 12
    assert isinstance(ei.value, JBSNotConverged)


@pytest.mark.parametrize("policy", ["raise", "flag"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_evaluation_raises_at_that_pass_with_its_location(
        policy, bad):
    """NaN J on pass 2: raised on pass 2 (the loop used to count it as a
    non-growing, non-passing pass, blend the NaN into the next iterate and
    run to the ceiling), even under the flag policy."""
    Jstar, step, ev = _affine_problem(-0.1)
    st, n = _counting(step)
    calls = {"c": 0}

    def ev_bad(meas):
        calls["c"] += 1
        J = np.array(ev(meas), dtype=float)
        if calls["c"] == 2:
            J[37] = bad
        return J
    with pytest.raises(JBSNonFinite, match=r"evaluated bootstrap J.*pass "
                                           r"2/8.*index 37, psi_N=0\.37") \
            as ei:
        run_jbs_loop(0.5 * Jstar, st, ev_bad,
                     jbs_settings(_GC(jbs_loop_on_fail=policy)), Ip=_IP,
                     meas0=dict(li=0.0))
    e = ei.value
    assert (e.pass_number, e.index) == (2, 37)
    assert e.psi_N == pytest.approx(_X[37])
    assert n["c"] == 2                          # no third solve
    assert e.record["n_passes"] == 2 and e.record["converged"] is False
    json.dumps(jsonable(e.record))


# ---------------------------------------------------------------------------
#  passes that can never count
# ---------------------------------------------------------------------------
def test_an_identically_zero_evaluation_stops_at_once():
    Jstar, step, ev = _affine_problem(-0.1)
    st, n = _counting(step)
    with pytest.raises(JBSNotConverged, match="identically zero") as ei:
        run_jbs_loop(0.5 * Jstar, st, lambda m: np.zeros_like(_X),
                     jbs_settings(_GC()), Ip=_IP, meas0=dict(li=0.0))
    assert n["c"] == 1 and ei.value.record["n_passes"] == 1
    assert not isinstance(ei.value, JBSNonFinite)
    # flag policy: the (finite) last iterate is returned, flagged
    st2, n2 = _counting(step)
    out = run_jbs_loop(0.5 * Jstar, st2, lambda m: np.zeros_like(_X),
                       jbs_settings(_GC(jbs_loop_on_fail="flag")), Ip=_IP,
                       meas0=dict(li=0.0))
    assert out["converged"] is False and n2["c"] == 1
    assert "identically zero" in out["record"]["stop_reason"]


def test_no_bootstrap_at_all_is_a_legitimate_fixed_point():
    """J = 0 AND jbs = 0 (a profile with no bootstrap): r_j = 0, converges."""
    _J, step, _ev = _affine_problem(-0.1)
    out = run_jbs_loop(np.zeros_like(_X), step, lambda m: np.zeros_like(_X),
                       jbs_settings(_GC()), Ip=_IP, meas0=dict(li=1.0))
    assert out["converged"] and out["record"]["n_passes"] == 2


@pytest.mark.parametrize("gate", ["li", "q0"])
def test_a_gated_measurement_the_step_does_not_return_stops_at_once(gate):
    Jstar, step, ev = _affine_problem(-0.1)

    def step_missing(jbs, k):
        m = step(jbs, k)
        m.pop(gate)
        return m
    st, n = _counting(step_missing)
    kw = dict(gate_li=(gate == "li"), gate_q0=(gate == "q0"))
    name = {"li": "l_i", "q0": "q0"}[gate]
    with pytest.raises(JBSNotConverged, match=f"no finite {name}") as ei:
        run_jbs_loop(0.5 * Jstar, st, ev, jbs_settings(_GC()), Ip=_IP,
                     meas0=dict(li=0.0, q0=1.0), **kw)
    assert n["c"] == 1 and ei.value.record["n_passes"] == 1
    # a NaN reading is the same as a missing one
    def step_nan(jbs, k):
        m = step(jbs, k)
        m[gate] = float("nan")
        return m
    with pytest.raises(JBSNotConverged, match=f"no finite {name}"):
        run_jbs_loop(0.5 * Jstar, step_nan, ev, jbs_settings(_GC()), Ip=_IP,
                     meas0=dict(li=0.0, q0=1.0), **kw)


def test_an_ungated_measurement_may_be_absent():
    Jstar, step, ev = _affine_problem(-0.1)

    def step_no_li(jbs, k):
        m = step(jbs, k)
        m.pop("li")
        return m
    out = run_jbs_loop(0.5 * Jstar, step_no_li, ev, jbs_settings(_GC()),
                       Ip=_IP, meas0=None, gate_li=False)
    assert out["converged"]


# ---------------------------------------------------------------------------
#  the unrelaxed closure-half residual (record only)
# ---------------------------------------------------------------------------
def test_the_unrelaxed_residual_is_the_gap_over_one_minus_beta():
    step, ev, st, Lstar, Jstar = _two_state_problem()
    s = jbs_settings(_GC(jbs_relax_current=0.7, jbs_max_passes=40))
    o = run_jbs_loop(Jstar * 0.6, step, ev, s, Ip=_IP,
                     meas0=dict(li=st["L"]), gate_li=True)
    rec = o["record"]
    u, g, bl = (rec["current_residual_unrelaxed"], rec["current_gap"],
                rec["current_blended"])
    assert len(u) == rec["n_passes"]
    assert u[0] is None                         # no previous solved current
    for uk, gk, b in zip(u[1:], g[1:], bl[1:]):
        assert b
        assert uk == pytest.approx(gk / (1.0 - 0.7), rel=1e-12)
    assert rec["final"]["current_residual_unrelaxed"] == u[-1]
    assert "NOT a convergence criterion" in \
        rec["current_residual_unrelaxed_definition"]
    # record only: the criteria are unchanged
    assert set(rec["criteria"]) == {"r_j", "r_I", "dl_i", "dq0"}
    json.dumps(jsonable(rec))


def test_the_unrelaxed_residual_is_defined_without_relaxation():
    """beta = 1: the gap is identically 0, gap/(1-beta) is 0/0, and the
    unrelaxed residual is still measured."""
    step, ev, st, Lstar, Jstar = _two_state_problem()
    s = jbs_settings(_GC(jbs_relax_current=1.0, jbs_max_passes=40))
    o = run_jbs_loop(Jstar * 0.6, step, ev, s, Ip=_IP,
                     meas0=dict(li=st["L"]), gate_li=True)
    rec = o["record"]
    assert all(gk == 0.0 for gk in rec["current_gap"])
    assert all(uk is not None and uk > 0.0
               for uk in rec["current_residual_unrelaxed"][1:])


def test_a_step_without_the_relaxer_records_no_unrelaxed_residual():
    Jstar, step, ev = _affine_problem(-0.1)
    out = run_jbs_loop(0.5 * Jstar, step, ev, jbs_settings(_GC()), Ip=_IP,
                       meas0=dict(li=0.0))
    rec = out["record"]
    assert rec["current_residual_unrelaxed"] == [None] * rec["n_passes"]
