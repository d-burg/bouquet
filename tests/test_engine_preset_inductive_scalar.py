"""The engine preset ``inductive_scalar`` -- fast, on the toy Grad-Shafranov
stand-in (no solver).

``bootstrap_scalar`` with the roles swapped: the bootstrap is the evaluator's
own (``s_bs = 1``, pinned by an infinite weight) and ONE scalar on the
inductive current meets Ip exactly.  The closure for comparing bootstrap
models (``GenerationConfig.bootstrap_model``): no closure coefficient can
absorb the difference between two models.  It adds no number: the constant
basis, the sigma and every tolerance are the existing ones.  Synthetic
inputs only.
"""
import contextlib
import io

import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import ENGINE_PRESETS, PRESET_ROWS, reconstruct


def _run(adapter, **gc):
    b = T.ToyGS()
    adapter.read()
    with contextlib.redirect_stdout(io.StringIO()):
        return reconstruct(adapter, b, T.settings(**gc), label="toy") + (b,)


def test_registered_with_the_Ip_row_only():
    assert "inductive_scalar" in ENGINE_PRESETS
    assert PRESET_ROWS["inductive_scalar"] == frozenset(("Ip",))
    with pytest.raises(ValueError, match="admits"):
        T.settings(engine_preset="inductive_scalar",
                   engine_rows=["Ip", "l_i"])


def test_bootstrap_pinned_and_Ip_met_by_one_inductive_scalar():
    eng, res, rec, b = _run(T.ToyAdapter(), engine_preset="inductive_scalar",
                            engine_rows=["Ip"])
    assert res["converged"] and res["loop_converged"]
    assert rec["delivered"]["ok"], rec["delivered"]["misses"]
    assert rec["settings"]["preset_in_force"] == "inductive_scalar"
    for p in eng.passes:
        assert tuple(p["s_bs_range"]) == (1.0, 1.0)
        lo, hi = p["s_ind_range"]
        assert lo == hi
    out = eng.delivered_closure["out"]
    np.testing.assert_array_equal(np.asarray(out["s_bs"], float), 1.0)
    # the solved request carries the bootstrap iterate exactly once
    gate = eng.delivered_closure["ip_gate"]
    assert gate is None or gate.get("ok", True)


def test_the_closure_is_the_one_scalar_Ip_equation():
    """Hard rows, one free scalar: ``s_ind lin(j_ind) + lin(j_bs) + lin(j_fix)
    + c = Ip`` exactly on the closure's geometry (the bootstrap enters
    unscaled)."""
    from bouquet.engine import UnifiedEngine, _lin
    ad = T.ToyAdapter()
    c = ad.read()
    b = T.ToyGS()
    b.solve(c.anchor_request, n_passes=2)
    e = UnifiedEngine(c, b, T.settings(engine_preset="inductive_scalar",
                                       engine_rows=["Ip"]),
                      anchor=b.measure())
    g = e.state.geom
    cl = e.close(g, e.state.lambda_bs)
    p = cl["parts"]
    s_ind = float(np.asarray(cl["out"]["s_ind"], float)[0])
    lhs = (s_ind * _lin(g, p["ind"]) + _lin(g, p["bs"])
           + _lin(g, p["fix"] + p["pressure"]) + cl["c_signed"])
    assert lhs == pytest.approx(cl["Ip_signed"], rel=1e-12)
    np.testing.assert_array_equal(np.asarray(cl["out"]["s_bs"], float), 1.0)
