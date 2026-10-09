"""The kernel's opt-in ``extra`` criteria hook (used by the unified engine).

``run_jbs_loop(..., extra=None)`` -- the default, and every legacy caller --
must be bit-identical to the kernel before the hook existed: settings, the
whole record (every float through its exact JSON repr), the delivered arrays,
the exception and every printed line.  The reference is the frozen verbatim
copy ``tests/_jbs_loop_pre_engine_kernel.py``; the scenarios are the ones the
current-gate bit-identity test uses, with the gate OFF and ON and with the
acting q0 pin.

With a hook, its verdict is ANDed into the pass verdict (an added criterion,
never a replacement), its ``never`` reason stops the loop at once, and its
record is stored.

Synthetic fixed-point maps only; no solver, no device data.
"""
import contextlib
import io
import json

import numpy as np
import pytest

from bouquet.jbs_loop import (AxisRowPin, JBSNotConverged, jbs_settings,
                              jsonable, run_jbs_loop)
from test_jbs_loop import _GC, _IP, _W, _X, _affine_problem
from test_jbs_loop_current_gate import _OFF_SCENARIOS, _scenario

import _jbs_loop_pre_engine_kernel as _PRE


#: The settings dict the scenarios produced when the hook was added, FROZEN
#: (literal values; ``_GC``'s fields plus the kernel constants of the time):
#: the pre-hook kernel is run with these, the current one with the live
#: ``jbs_settings(...)`` -- so ``new[0] == pre[0]`` compares the live
#: settings with the frozen ones instead of a value with itself.
_FROZEN_BASE = {"enabled": True, "init": "anchor", "rtol_j": 0.001,
                "rtol_Ip": 0.0001, "tol_li": 0.001, "tol_q0": 0.002,
                "max_passes": 8, "relax": 0.7, "relax_current": 0.7,
                "relax_halve_on": 3, "on_fail": "raise", "relax_floor": 0.25,
                "required_consecutive": 2, "growth_abort_passes": 3,
                "post_homotopy_passes": 4}


def _frozen_settings(kw, gc_extra):
    s = dict(_FROZEN_BASE)
    names = {"jbs_max_passes": "max_passes", "jbs_loop_on_fail": "on_fail",
             "jbs_relax_current": "relax_current"}
    for k, v in kw.items():
        s[names[k]] = v
    if gc_extra.get("jbs_gate_current_residual"):
        s["gate_current_residual"] = True
    return s


def _outcome(kernel, sc, gc_extra, pin=False):
    (j0, step, ev, kws), kw = _scenario(*sc)
    s = (_frozen_settings(kw, gc_extra) if kernel is _PRE.run_jbs_loop
         else jbs_settings(_GC(**kw, **gc_extra)))
    kws = dict(kws)
    if pin:
        kws["q0_pin"] = AxisRowPin(1.0, 0.01, 2.0e5, label="t")
        _inner = step

        def step(jbs, k, relax=None):          # noqa: F811
            m = (_inner(jbs, k) if relax is None
                 else _inner(jbs, k, relax=relax))
            m.setdefault("q0", 1.0 + 1e-9 * float(np.asarray(jbs)[0]))
            m["axis_current_solved"] = 2.0e5
            return m
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


def _same(new, pre):
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


@pytest.mark.parametrize("gc_extra", [{}, {"jbs_gate_current_residual": True}])
@pytest.mark.parametrize("sc", _OFF_SCENARIOS)
def test_hook_absent_is_bit_identical_to_the_kernel_before_it(sc, gc_extra):
    _same(_outcome(run_jbs_loop, sc, gc_extra),
          _outcome(_PRE.run_jbs_loop, sc, gc_extra))


@pytest.mark.parametrize("sc", [s for s in _OFF_SCENARIOS
                                if s[0] == "two_state"][:8])
def test_hook_absent_is_bit_identical_with_the_acting_q0_pin(sc):
    _same(_outcome(run_jbs_loop, sc, {}, pin=True),
          _outcome(_PRE.run_jbs_loop, sc, {}, pin=True))


class _Extra:
    names = ("my_row",)

    def __init__(self, oks, never_at=None):
        self.oks = list(oks)
        self.never_at = never_at
        self.seen = []

    def observe(self, k, meas):
        self.seen.append(k)
        if self.never_at is not None and k == self.never_at:
            return False, "my row cannot be measured", "my_row=n/a"
        ok = self.oks[k] if k < len(self.oks) else True
        return ok, None, f"my_row={'ok' if ok else 'miss'}"

    def record(self):
        return dict(seen=list(self.seen))

    def history_text(self):
        return f"my_row passes seen={self.seen}"


def test_the_hook_adds_a_criterion_and_is_recorded():
    Jstar, step, ev = _affine_problem(0.0)
    s = jbs_settings(_GC(jbs_max_passes=12))
    base = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                        meas0=dict(li=0.0, q0=0.0), gate_q0=True)
    n0 = base["record"]["n_passes"]
    # the hook misses on the passes the loop alone would have converged on
    ex = _Extra([False] * (n0 + 1))
    Jstar, step, ev = _affine_problem(0.0)
    o = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                     meas0=dict(li=0.0, q0=0.0), gate_q0=True, extra=ex)
    assert o["converged"]
    assert o["record"]["n_passes"] > n0
    assert o["record"]["criteria"]["my_row"] is True
    assert o["record"]["extra_criteria"]["seen"] == list(
        range(o["record"]["n_passes"]))


def test_the_hooks_never_reason_stops_at_once_and_raises():
    Jstar, step, ev = _affine_problem(0.0)
    s = jbs_settings(_GC(jbs_max_passes=12))
    with pytest.raises(JBSNotConverged, match="my row cannot be measured"):
        run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                     meas0=dict(li=0.0, q0=0.0), extra=_Extra([], 1))


def test_a_hook_that_never_holds_fails_at_the_ceiling_with_its_history():
    Jstar, step, ev = _affine_problem(0.0)
    s = jbs_settings(_GC(jbs_max_passes=5, jbs_loop_on_fail="flag"))
    o = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                     meas0=dict(li=0.0, q0=0.0), extra=_Extra([False] * 9))
    assert not o["converged"]
    assert "my_row passes seen" in o["record"]["fail_message"]
    assert o["record"]["n_passes"] == 5
