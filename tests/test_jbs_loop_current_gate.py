"""The opt-in closure-half current gate (``jbs_gate_current_residual``).

With the flag ON the loop additionally requires, on the same two consecutive
passes as its other criteria, that the closure-half residual
``||jc_k - js_k-1||_w / ||jc_k||_w`` -- the current the previous pass SOLVED
against the current the closure composes on the equilibrium that solve
produced -- is within ``jbs_rtol_j``.  With the flag OFF (the default) the
kernel is bit-identical to the one before the flag existed.

The toy maps are the ones ``test_jbs_loop.py`` uses.  The positive-coupling
cases (``g > 0``: a slow, monotone closure <-> geometry mode) are the ones
the loop review used: there the ungated loop declares convergence while the
delivered current is not the current its own closure composes on it, and on
slower modes while l_i is off its fixed point by more than ``jbs_tol_li``.

Synthetic fixed-point maps only; no solver, no device data.
"""
import contextlib
import io
import json

import numpy as np
import pytest

from bouquet.jbs_loop import (CURRENT_GATE_DEFINITION, JBSNotConverged,
                              criteria_timeline, jbs_settings, jsonable,
                              profile_residuals, run_jbs_loop,
                              validate_jbs_settings, weighted_norm)
from test_jbs_loop import (_GC, _IP, _W, _X, _affine_problem,
                           _growing_problem, _two_state_problem)

import _jbs_loop_prechange_kernel as _PRE


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k), buf.getvalue()


def _closed_form(g):
    """The two-state map's own closure pieces (as in ``_two_state_problem``):
    ``closure(L, J) = s(L) j_ind0 + J`` with ``s(L) = 1 + kappa (1 - L)``."""
    x = _X
    j_ind0 = 8e5 * (1 - x) ** 1.5 + 8e4
    A = float(np.trapezoid(_W * x * j_ind0, x)) / _IP
    kappa = -g / A
    return j_ind0, kappa


def _run_two_state(g, beta, K, gate, fail="raise", with_relaxer=True):
    """Run the review's toy; returns (out or exception, state, L*, J*)."""
    step, ev, st, Lstar, Jstar = _two_state_problem(g=g)
    if not with_relaxer:
        _inner = step

        def step(jbs, k):                  # noqa: F811 -- beta not applied
            m = _inner(jbs, k)
            m["j_solved"] = st["js"].copy()
            return m
    s = jbs_settings(_GC(jbs_relax_current=beta, jbs_max_passes=K,
                         jbs_loop_on_fail=fail,
                         jbs_gate_current_residual=gate))
    try:
        out, _ = _quiet(run_jbs_loop, Jstar * 0.6, step, ev, s, Ip=_IP,
                        meas0=dict(li=st["L"]), gate_li=True)
    except JBSNotConverged as e:
        out = e
    return out, st, ev, Lstar, Jstar, s


def _delivered_closure_residual(g, st, ev):
    """EXACT closure-half residual of the delivered equilibrium: the closure
    composed on the delivered geometry with the bootstrap evaluated on it,
    against the current that equilibrium was solved with."""
    j_ind0, kappa = _closed_form(g)
    jc = (1.0 + kappa * (1.0 - st["L"])) * j_ind0 + ev(None)
    return weighted_norm(jc - st["js"], _W, _X) / weighted_norm(jc, _W, _X)


# ---------------------------------------------------------------------------
#  configuration
# ---------------------------------------------------------------------------
#: The keys of ``jbs_settings`` before the flag existed.
_PRE_SETTINGS_KEYS = {
    "enabled", "init", "rtol_j", "rtol_Ip", "tol_li", "tol_q0", "max_passes",
    "relax", "relax_current", "relax_halve_on", "on_fail", "relax_floor",
    "required_consecutive", "growth_abort_passes", "post_homotopy_passes"}


def test_the_flag_defaults_off_and_leaves_the_settings_unchanged():
    from bouquet.config import GenerationConfig
    g = GenerationConfig()
    assert g.jbs_gate_current_residual is False
    for draw in (False, True):
        assert set(jbs_settings(g, draw=draw)) == _PRE_SETTINGS_KEYS
        # an object without the field (an old config) reads as OFF
        assert set(jbs_settings(_GC(), draw=draw)) == _PRE_SETTINGS_KEYS
    g.jbs_gate_current_residual = True
    s = jbs_settings(g)
    assert set(s) == _PRE_SETTINGS_KEYS | {"gate_current_residual"}
    assert s["gate_current_residual"] is True
    # nothing else moves with it: same tolerances, same ceilings
    g0 = GenerationConfig()
    assert {k: v for k, v in s.items() if k != "gate_current_residual"} \
        == jbs_settings(g0)


@pytest.mark.parametrize("bad", [1, 0, "yes", None, 1.0])
def test_a_non_bool_flag_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="jbs_gate_current_residual"):
        validate_jbs_settings(_GC(jbs_gate_current_residual=bad))


def test_the_flag_round_trips_and_an_old_config_loads_it_off():
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    cfg = BouquetConfig(source=ImasSource(ids_path="x.json"),
                        solver=SolverConfig(mesh_path="m.h5"),
                        output_header="t",
                        generation=GenerationConfig(
                            jbs_gate_current_residual=True))
    assert BouquetConfig.from_json(cfg.to_json()).generation\
        .jbs_gate_current_residual is True
    d = BouquetConfig(source=ImasSource(ids_path="x.json"),
                      solver=SolverConfig(mesh_path="m.h5"),
                      output_header="t",
                      generation=GenerationConfig()).to_dict()
    del d["generation"]["jbs_gate_current_residual"]
    assert BouquetConfig.from_dict(d).generation\
        .jbs_gate_current_residual is False


# ---------------------------------------------------------------------------
#  flag OFF: bit-identical to the kernel before the flag
# ---------------------------------------------------------------------------
def _scenario(kind, a, beta, K, fail):
    kw = dict(jbs_max_passes=K, jbs_loop_on_fail=fail)
    if beta is not None:
        kw["jbs_relax_current"] = beta
    if kind in ("two_state", "two_state_j_solved"):
        step, ev, st, Lstar, Jstar = _two_state_problem(g=a)
        if kind == "two_state_j_solved":         # a step without the relaxer
            _inner = step                        # that reports j_solved

            def step(jbs, k):                    # noqa: F811
                m = _inner(jbs, k)
                m["j_solved"] = st["js"].copy()
                return m
        return (Jstar * 0.6, step, ev,
                dict(Ip=_IP, meas0=dict(li=st["L"]), gate_li=True)), kw
    if kind == "affine":
        Jstar, step, ev = _affine_problem(a)
        return (0.5 * Jstar, step, ev,
                dict(Ip=_IP, meas0=dict(li=0.0, q0=0.0), gate_q0=True)), kw
    J0, step, ev = _growing_problem()
    return (J0, step, ev, dict(Ip=_IP, meas0=dict(li=1.0))), kw


def _outcome(kernel, kind, a, beta, K, fail, gc_extra):
    (j0, step, ev, kws), kw = _scenario(kind, a, beta, K, fail)
    s = jbs_settings(_GC(**kw, **gc_extra))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            o = kernel(j0, step, ev, s, **kws)
        rec, arrays, err = o["record"], (o["jbs_used"], o["J_final"],
                                         o["converged"]), None
    except JBSNotConverged as e:
        rec, arrays, err = e.record, None, (type(e).__name__, str(e))
    rec = dict(rec)
    rec.pop("wall_s")
    return (s, json.dumps(jsonable(rec), sort_keys=True), arrays, err,
            buf.getvalue())


_OFF_SCENARIOS = (
    [("two_state", g, b, K, f) for g in (-0.55, 0.3, 0.5, 0.8)
     for b in (0.7, 1.0) for K in (8, 12, 40) for f in ("flag", "raise")]
    + [("two_state_j_solved", g, 1.0, 12, "flag") for g in (-0.55, 0.5)]
    + [("affine", c, None, 8, f) for c in (-0.1, 0.0, 0.5)
       for f in ("flag", "raise")]
    + [("growing", None, None, 12, "raise")])


@pytest.mark.parametrize("gc_extra", [{}, {"jbs_gate_current_residual":
                                           False}])
@pytest.mark.parametrize("sc", _OFF_SCENARIOS)
def test_flag_off_is_bit_identical_to_the_kernel_before_the_flag(sc,
                                                                  gc_extra):
    """Settings, the whole record (every float compared through its exact
    JSON repr), the delivered arrays, the exception and every printed line."""
    new = _outcome(run_jbs_loop, *sc, gc_extra)
    pre = _outcome(_PRE.run_jbs_loop, *sc, {})
    assert new[0] == pre[0]
    assert new[1] == pre[1]
    assert new[3] == pre[3]
    assert new[4] == pre[4]
    if pre[2] is None:
        assert new[2] is None
    else:
        assert np.array_equal(new[2][0], pre[2][0])
        assert np.array_equal(new[2][1], pre[2][1])
        assert new[2][2] == pre[2][2]


# ---------------------------------------------------------------------------
#  flag ON: what the gate measures
# ---------------------------------------------------------------------------
def test_the_gated_residual_is_computed_directly_and_matches_the_estimate():
    """On a blended pass the direct ``||jc_k - js_k-1|| / ||jc_k||`` equals
    ``gap / (1 - beta)``; both are recorded with their ratio."""
    # pass ceiling 9 for this toy only: with r_j measured against the
    # bootstrap actually solved (a769882) the blended toy needs one more
    # pass.  Product ceilings unchanged (owner-approved 2026-10-07).
    o, st, ev, Lstar, Jstar, s = _run_two_state(-0.55, 0.7, 9, True)
    rec = o["record"]
    assert o["converged"]
    cur, est = rec["current_residual"], rec["current_residual_estimate"]
    ratio = rec["current_residual_direct_over_estimate"]
    assert cur[0] is None and est[0] is None and ratio[0] is None
    # the gated value IS the record-only unrelaxed residual, bit for bit
    assert cur == rec["current_residual_unrelaxed"]
    for c, e, r in zip(cur[1:], est[1:], ratio[1:]):
        assert e == pytest.approx(c, rel=1e-12)
        assert r == pytest.approx(1.0, rel=1e-12)


def test_the_gated_residual_of_a_step_without_the_relaxer_is_direct():
    """A step that does not take the relaxer reports ``j_solved``; the gate
    measures ``||js_k - js_k-1||_w / ||js_k||_w`` (beta not applied, so the
    closure's current IS the solved one).  No estimate exists there."""
    step, ev, st, Lstar, Jstar = _two_state_problem(g=-0.55)
    solved = []

    def plain(jbs, k):
        m = step(jbs, k)
        solved.append(st["js"].copy())
        m["j_solved"] = solved[-1]
        return m

    s = jbs_settings(_GC(jbs_max_passes=40,
                         jbs_gate_current_residual=True))
    o, _ = _quiet(run_jbs_loop, Jstar * 0.6, plain, ev, s, Ip=_IP,
                  meas0=dict(li=st["L"]), gate_li=True)
    rec = o["record"]
    assert o["converged"]
    assert "j_solved" in rec["current_gate"]["source"]
    assert rec["current_residual"][0] is None
    for k in range(1, rec["n_passes"]):
        want = (weighted_norm(solved[k] - solved[k - 1], _W, _X)
                / weighted_norm(solved[k], _W, _X))
        assert rec["current_residual"][k] == pytest.approx(want, rel=1e-14)
    assert all(e is None for e in rec["current_residual_estimate"])
    assert rec["current_residual"][-1] <= s["rtol_j"]


def test_a_step_that_cannot_report_the_residual_stops_at_once():
    """No relaxer and no ``j_solved``: the gate can never be evaluated, so
    the loop stops at pass 1 under the failure policy -- never passes it."""
    Jstar, step, ev = _affine_problem(-0.1)
    s = jbs_settings(_GC(jbs_gate_current_residual=True))
    with pytest.raises(JBSNotConverged, match="cannot be evaluated") as ei:
        _quiet(run_jbs_loop, 0.5 * Jstar, step, ev, s, Ip=_IP,
               meas0=dict(li=0.0))
    assert ei.value.record["n_passes"] == 1
    sf = jbs_settings(_GC(jbs_gate_current_residual=True,
                          jbs_loop_on_fail="flag"))
    o, _ = _quiet(run_jbs_loop, 0.5 * Jstar, step, ev, sf, Ip=_IP,
                  meas0=dict(li=0.0))
    assert not o["converged"] and o["record"]["n_passes"] == 1


def test_a_relaxer_step_that_never_relaxes_stops_on_pass_two():
    step, ev, st, Lstar, Jstar = _two_state_problem()

    def lazy(jbs, k, relax=None):               # takes it, never calls it
        return step(jbs, k)

    s = jbs_settings(_GC(jbs_gate_current_residual=True,
                         jbs_max_passes=40))
    with pytest.raises(JBSNotConverged, match="relaxer") as ei:
        _quiet(run_jbs_loop, Jstar * 0.6, lazy, ev, s, Ip=_IP,
               meas0=dict(li=st["L"]), gate_li=True)
    assert ei.value.record["n_passes"] == 2


def test_pass_one_cannot_count_under_the_gate():
    """Started AT the fixed point the ungated loop needs 2 passes; the gate
    has no solved current before pass 1, so it needs 3 -- never fewer."""
    for gate, n in ((False, 2), (True, 3)):
        step, ev, st, Lstar, Jstar = _two_state_problem()
        st["L"] = Lstar
        s = jbs_settings(_GC(jbs_gate_current_residual=gate))
        o, _ = _quiet(run_jbs_loop, Jstar.copy(), step, ev, s, Ip=_IP,
                      meas0=dict(li=Lstar), gate_li=True)
        assert o["converged"] and o["record"]["n_passes"] == n
        assert abs(st["L"] - Lstar) <= 1e-12


def test_the_gate_record():
    o, st, ev, Lstar, Jstar, s = _run_two_state(0.3, 1.0, 12, True)
    rec = o["record"]
    assert o["converged"]
    n = rec["n_passes"]
    assert rec["criteria"]["current_residual"] is True
    gate = rec["current_gate"]
    assert gate["active"] is True
    assert gate["tolerance"] == s["rtol_j"]
    assert gate["tolerance_setting"] == "jbs_rtol_j"
    assert gate["definition"] == CURRENT_GATE_DEFINITION
    for key in ("current_residual", "current_residual_estimate",
                "current_residual_direct_over_estimate",
                "current_residual_ok"):
        assert len(rec[key]) == n, key
    assert rec["final"]["current_residual"] == rec["current_residual"][-1]
    assert rec["current_residual_ok"][-2:] == [True, True]
    assert rec["pass_ok"][-2:] == [True, True]
    # every pass that counted met the gate
    assert all(c for c, p in zip(rec["current_residual_ok"], rec["pass_ok"])
               if p)
    tl = rec["criteria_timeline"]
    assert rec["last_criterion_met"] == tl["last_met"]
    assert "current_residual" in tl["gated"]
    last = tl["met_since_pass"]
    assert max(last[c] for c in tl["gated"]) == last[tl["last_met"][0]]
    json.dumps(jsonable(rec))


def test_the_criteria_timeline_reads_an_ungated_record_too():
    """On a flag-OFF record the closure-half residual is reported (from the
    record-only field) but never named as a gating criterion."""
    o, st, ev, Lstar, Jstar, s = _run_two_state(0.3, 0.7, 12, False)
    rec = o["record"]
    assert o["converged"]
    tl = criteria_timeline(rec)
    assert "current_residual" not in tl["gated"]
    assert "current_residual" in tl["met_since_pass"]
    assert tl["last_met"] and "current_residual" not in tl["last_met"]
    assert set(tl["gated"]) == {"r_j", "r_I", "dl_i"}


# ---------------------------------------------------------------------------
#  flag ON: what the gate guarantees, on the review's cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K", [8, 12])        # the shipped ceilings
@pytest.mark.parametrize("beta", [0.7, 1.0])
@pytest.mark.parametrize("g", [0.3, 0.5, 0.65, 0.8])
def test_at_the_shipped_ceilings_a_gated_loop_is_on_its_fixed_point_or_fails(
        g, beta, K):
    """Positive coupling: with the gate ON a declared convergence is within
    the loop's own tolerances of the closed-form fixed point (l_i, j_BS) and
    delivers a current its own closure reproduces -- or the loop raises."""
    o, st, ev, Lstar, Jstar, s = _run_two_state(g, beta, K, True)
    if isinstance(o, JBSNotConverged):
        assert o.record["converged"] is False
        assert "current_residual=" in str(o)
        return
    assert o["converged"]
    assert abs(st["L"] - Lstar) <= s["tol_li"]
    assert profile_residuals(Jstar, o["jbs_used"], _W, _X,
                             _IP)["r_j"] <= s["rtol_j"]
    assert _delivered_closure_residual(g, st, ev) <= s["rtol_j"]


def test_the_reviews_positive_coupling_case_is_closed_by_the_gate():
    """g = +0.5, beta = 0.7, 40 passes (the review's case): ungated, the loop
    declares convergence with l_i off its fixed point by more than
    jbs_tol_li and a delivered current its closure does not reproduce;
    gated, both are inside tolerance."""
    off, st0, ev0, Lstar, Jstar, s = _run_two_state(0.5, 0.7, 40, False)
    assert off["converged"]
    assert abs(st0["L"] - Lstar) > s["tol_li"]
    assert _delivered_closure_residual(0.5, st0, ev0) > s["rtol_j"]
    on, st1, ev1, Lstar, Jstar, s = _run_two_state(0.5, 0.7, 40, True)
    assert on["converged"]
    assert on["record"]["n_passes"] > off["record"]["n_passes"]
    assert abs(st1["L"] - Lstar) <= s["tol_li"]
    assert _delivered_closure_residual(0.5, st1, ev1) <= s["rtol_j"]
    assert profile_residuals(Jstar, on["jbs_used"], _W, _X,
                             _IP)["r_j"] <= s["rtol_j"]


def test_at_the_draw_ceiling_an_inconsistent_convergence_becomes_a_failure():
    """g = +0.3, beta = 0.7, the draw ceiling 12: ungated, the loop
    converges while the delivered current differs from the one its closure
    composes on it by ~3x jbs_rtol_j; gated, it raises at the ceiling."""
    off, st0, ev0, Lstar, Jstar, s = _run_two_state(0.3, 0.7, 12, False)
    assert off["converged"]
    assert _delivered_closure_residual(0.3, st0, ev0) > 2.0 * s["rtol_j"]
    on, st1, ev1, Lstar, Jstar, s = _run_two_state(0.3, 0.7, 12, True)
    assert isinstance(on, JBSNotConverged)
    assert on.record["n_passes"] == 12
    assert "pass ceiling 12" in on.record["stop_reason"]


@pytest.mark.parametrize("beta", [0.7, 1.0])
def test_every_gated_convergence_delivers_a_consistent_current(beta):
    """Across the coupling scan (40-pass ceiling, "flag"), whenever the
    gated loop declares convergence, the delivered equilibrium's current is
    reproduced by its own closure (evaluated exactly) to jbs_rtol_j, and its
    bootstrap by Redl to jbs_rtol_j."""
    n_conv = 0
    for g in np.round(np.arange(-0.8, 0.81, 0.05), 3):
        o, st, ev, Lstar, Jstar, s = _run_two_state(float(g), beta, 40,
                                                    True, fail="flag")
        if not o["converged"]:
            continue
        n_conv += 1
        assert _delivered_closure_residual(float(g), st, ev) <= s["rtol_j"]
        assert profile_residuals(ev(None), o["jbs_used"], _W, _X,
                                 _IP)["r_j"] <= s["rtol_j"]
    assert n_conv >= 25


def test_the_gate_bounds_the_residual_not_the_distance_on_a_slow_mode():
    """A LIMIT of the gate, pinned so it is not over-claimed: on a slow
    monotone mode (g = +0.8, beta = 1, 40 passes) the gated loop converges
    with a delivered current consistent to jbs_rtol_j, yet l_i is still more
    than jbs_tol_li from the fixed point (distance ~ residual / (1 - rho)).
    Only an a-posteriori contraction bound would catch this; at the shipped
    ceilings the case fails loudly instead (test above)."""
    o, st, ev, Lstar, Jstar, s = _run_two_state(0.8, 1.0, 40, True)
    assert o["converged"]
    assert _delivered_closure_residual(0.8, st, ev) <= s["rtol_j"]
    assert abs(st["L"] - Lstar) > s["tol_li"]
