"""r_j / r_I are measured against the bootstrap the equilibrium was SOLVED
with (2026-10-06).

With the solved current relaxed (``jbs_relax_current`` beta < 1, the default
0.7) a pass solves ``(1 - beta) js_k-1 + beta jc_k``, not the closure's
current, so the bootstrap the delivered equilibrium carries is the same
blend of the iterates.  The kernel used to judge the ITERATE against Redl on
the solved state: on the review's toy it reported converged at r_j = 2.4e-4
while Redl against the bootstrap actually solved was 1.09e-3, above
``jbs_rtol_j`` = 1e-3 (the current gate, normalised by the total current,
does not catch it).  Now a converged loop delivers a state whose bootstrap
is within the tolerances of its own Redl evaluation; the iterate residual is
kept as ``r_j_iterate`` (record only).  Mutant: the old residual (against
the iterate) fails :func:`test_a_converged_toy_satisfies_the_solved_residual`
at c = 0.

Synthetic fixed-point maps only; no solver, no device data.
"""
import contextlib
import io

import numpy as np
import pytest

from bouquet.config import GenerationConfig
from bouquet.jbs_loop import (CurrentRelaxer, jbs_settings,
                              profile_residuals, run_jbs_loop)

_X = np.linspace(0.0, 1.0, 101)
_W = np.ones_like(_X)
_A = 1e5 * np.exp(-0.5 * ((_X - 0.9) / 0.05) ** 2)   # the Redl "pedestal"
_JIND = 1e6 * (1.0 - _X ** 2)
_IP = 1e6


def _toy(c, max_passes=60):
    """The review's toy: the step composes ``jind + jbs`` and solves its
    relaxed blend; Redl on the solved state is ``a + c * (bootstrap the
    state carries)``.  Returns the loop output and the bootstrap the last
    solve actually carried."""
    s = jbs_settings(GenerationConfig(), draw=True)
    st = {}

    def step(jbs, k, relax=None):
        jc = _JIND + jbs
        js = jc if relax is None else relax(jc)
        st["b"] = js - _JIND
        return dict(w=_W, x=_X, li=0.0)

    def evaluate(meas):
        return _A + c * st["b"]

    with contextlib.redirect_stdout(io.StringIO()):
        out = run_jbs_loop(0.9 * _A, step, evaluate, s, Ip=_IP,
                           meas0=dict(li=0.0), gate_li=True,
                           raise_on_fail=False, max_passes=max_passes)
    return out, st["b"], s


@pytest.mark.parametrize("c", [0.0, 0.3, 0.6])
def test_a_converged_toy_satisfies_the_solved_residual(c):
    out, solved, s = _toy(c)
    assert s["relax_current"] < 1.0           # the default relaxes
    true = profile_residuals(out["J_final"], solved, _W, _X, _IP)
    if out["converged"]:
        assert true["r_j"] <= s["rtol_j"]
        assert true["r_I"] <= s["rtol_Ip"]
    rec = out["record"]
    # the recorded (gated) residual IS the solved one, and the returned
    # jbs_solved is the bootstrap the last solve carried
    np.testing.assert_allclose(out["jbs_solved"], solved, rtol=1e-12,
                               atol=1e-6)
    assert rec["r_j"][-1] == pytest.approx(true["r_j"], rel=1e-9)
    assert rec["r_I"][-1] == pytest.approx(true["r_I"], rel=1e-9,
                                           abs=1e-15)
    assert rec["residual_definition"].startswith("r_j = ||J - bs_k||")
    assert len(rec["r_j_iterate"]) == rec["n_passes"]


def test_the_iterate_residual_alone_would_have_accepted_the_toy():
    """The defect, kept visible: on a blended pass the iterate residual is
    below tolerance while the solved one is not -- the old criterion's false
    acceptance -- and the loop now runs on until the SOLVED state meets
    it (the path, the omega schedule included, is unchanged: the iterate
    residual history matches the old kernel's on the passes both take)."""
    out, _solved, s = _toy(0.0)
    rec = out["record"]
    it, sv = rec["r_j_iterate"], rec["r_j"]
    false_ok = [k for k in range(len(it))
                if it[k] <= s["rtol_j"] < sv[k]]
    assert false_ok, (it, sv)
    assert out["converged"]
    assert sv[-1] <= s["rtol_j"] and sv[-2] <= s["rtol_j"]
    # pass 1 never blends (no previous solved current): both agree there
    assert it[0] == sv[0] and rec["bootstrap_blended"][0] is False
    assert all(rec["bootstrap_blended"][1:])


def test_beta_one_records_no_iterate_keys_and_is_unchanged():
    s = jbs_settings(GenerationConfig(jbs_relax_current=1.0), draw=True)
    st = {}

    def step(jbs, k, relax=None):
        js = relax(_JIND + jbs)
        st["b"] = js - _JIND
        return dict(w=_W, x=_X, li=0.0)

    with contextlib.redirect_stdout(io.StringIO()):
        out = run_jbs_loop(0.9 * _A, step, lambda m: _A + 0.3 * st["b"], s,
                           Ip=_IP, meas0=dict(li=0.0), gate_li=True,
                           raise_on_fail=False, max_passes=60)
    rec = out["record"]
    assert "r_j_iterate" not in rec and "residual_definition" not in rec
    assert np.array_equal(out["jbs_solved"], out["jbs_used"])


def test_the_relaxer_reports_whether_it_blended():
    r = CurrentRelaxer(0.7)
    r.begin_pass()
    assert not r.called and not r.last_blended
    r(np.ones(3))
    assert r.called and not r.last_blended          # nothing to blend with
    r.commit()
    r.begin_pass()
    assert not r.called and not r.last_blended
    out = r(2.0 * np.ones(3))
    assert r.last_blended
    np.testing.assert_allclose(out, 0.3 * 1.0 + 0.7 * 2.0)
    r1 = CurrentRelaxer(1.0)
    r1(np.ones(3))
    r1.commit()
    r1(np.zeros(3))
    assert not r1.last_blended
