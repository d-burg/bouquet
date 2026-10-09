"""The engine presets ``two_scalar_li`` and ``structured_uniform`` -- fast
half, on the toy Grad-Shafranov stand-in (no solver).

``two_scalar_li``: the legacy secant's l_i family as a named closure -- ONE
scalar on the inductive and ONE on the bootstrap (the constant basis the
scalar presets already use), rows Ip + l_i.  With hard rows (g-file) it is a
2 x 2 system: no prior enters, and it is exactly what the q95 study reached
by patching the settings' preset to the constant two-scalar basis with rows
Ip + l_i.  With soft rows (IDS) the soft solver serves it, the constant
basis's sigma = 1 being the documented uniform prior.

``structured_uniform``: the shipped four-Gaussian basis under the ONE
documented alternative prior, ``utils.STRUCTURED_WEIGHTS_UNIFORM`` (every
coefficient sigma = 1, no one-sided up-ladder): the prior-sensitivity run.

Neither adds a number: the basis, the sigmas, the rows and every tolerance
are the existing ones.  Synthetic inputs only.
"""
import contextlib
import io

import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import (ENGINE_PRESETS, PRESET_ROWS, UnifiedEngine,
                            gfile_li_row_tol, reconstruct)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _anchor_li():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"]


LI0 = _anchor_li()


def _run(adapter, **gc):
    b = T.ToyGS()
    adapter.read()
    return _quiet(reconstruct, adapter, b, T.settings(**gc),
                  label="toy") + (b,)


def _engine(adapter, settings):
    c = adapter.read()
    b = T.ToyGS()
    b.solve(c.anchor_request, n_passes=2)
    return UnifiedEngine(c, b, settings, anchor=b.measure())


def test_the_presets_are_registered_with_their_rows():
    assert "two_scalar_li" in ENGINE_PRESETS
    assert "structured_uniform" in ENGINE_PRESETS
    assert PRESET_ROWS["two_scalar_li"] == frozenset(("Ip", "l_i"))
    assert PRESET_ROWS["structured_uniform"] == PRESET_ROWS["structured"]


@pytest.mark.parametrize("gen, match", [
    (dict(engine_preset="two_scalar_li", engine_rows=["Ip"]), "'l_i'"),
    (dict(engine_preset="two_scalar_li", engine_rows=["Ip", "l_i", "q0"]),
     "admits"),
    (dict(engine_preset="two_scalar_li", engine_rows=["Ip", "l_i", "mse"],
          mse_data=dict(R=[1.8] * 4, Z=[0.0] * 4, tgamma=[0.1] * 4,
                        sigma=[0.01] * 4, weight=[1.0] * 4, A1=[1.0] * 4,
                        A2=[1.0] * 4, A3=[0.0] * 4, A4=[0.0] * 4,
                        ip_sign=1, bt_sign=1, er_corrected=True)),
     "admits"),
])
def test_two_scalar_li_refuses_other_rows(gen, match):
    with pytest.raises(ValueError, match=match):
        T.settings(**gen)


@pytest.mark.parametrize("f", [0.99, 1.01, 1.03])
def test_two_scalar_li_converges_and_meets_both_rows_on_the_delivered_state(
        f):
    eng, res, rec, b = _run(T.ToyAdapter(li_target=LI0 * f),
                            engine_preset="two_scalar_li",
                            engine_rows=["Ip", "l_i"])
    assert res["converged"] and res["loop_converged"]
    d = rec["delivered"]
    assert d["ok"], d["misses"]
    li = d["checks"]["l_i"]
    assert li["hard"]
    assert abs(li["delivered"] - LI0 * f) <= gfile_li_row_tol()
    assert d["checks"]["loop"]["ok"] and d["checks"]["current_residual"]["ok"]
    assert rec["settings"]["preset_in_force"] == "two_scalar_li"
    # two scalars: every pass's multipliers are constants
    for p in eng.passes:
        lo, hi = p["s_ind_range"]
        assert lo == hi
        lo, hi = p["s_bs_range"]
        assert lo == hi


def test_two_scalar_li_is_the_patched_two_scalar_state():
    """The q95 study's state: the settings' preset patched to the constant
    two-scalar basis with rows Ip + l_i.  The named preset gives the same
    closure bit for bit, and the same reconstruction."""
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    s_named = T.settings(engine_preset="two_scalar_li",
                         engine_rows=["Ip", "l_i"])
    s_patch = dict(T.settings(engine_rows=["Ip", "l_i"]),
                   preset="sawtooth_two_scalar")
    e1, e2 = _engine(ad, s_named), _engine(ad, s_patch)
    c1 = e1.close(e1.state.geom, e1.state.lambda_bs)
    c2 = e2.close(e2.state.geom, e2.state.lambda_bs)
    np.testing.assert_array_equal(c1["jc"], c2["jc"])
    np.testing.assert_array_equal(c1["x"], c2["x"])
    b1, b2 = T.ToyGS(), T.ToyGS()
    ad.read()
    r1 = _quiet(reconstruct, ad, b1, s_named, label="toy")
    r2 = _quiet(reconstruct, ad, b2, s_patch, label="toy")
    np.testing.assert_array_equal(r1[1]["state"].request,
                                  r2[1]["state"].request)
    assert r1[2]["delivered"]["checks"]["l_i"]["delivered"] \
        == r2[2]["delivered"]["checks"]["l_i"]["delivered"]


def test_two_scalar_li_is_the_2x2_solve():
    """Hard rows: the closure meets Ip exactly and l_i at its predicted
    value with the two scalars -- the weights (sigma = 1) do not enter."""
    ad = T.ToyAdapter(li_target=LI0 * 1.01)
    s = T.settings(engine_preset="two_scalar_li", engine_rows=["Ip", "l_i"])
    e = _engine(ad, s)
    c1 = e.close(e.state.geom, e.state.lambda_bs)
    e.sigma_ind, e.sigma_bs = np.array([7.0]), np.array([0.3])
    from bouquet.utils import _weights_from_sigma
    e.weights = dict(name="x", ind=tuple(_weights_from_sigma(e.sigma_ind)),
                     bs=tuple(_weights_from_sigma(e.sigma_bs)))
    c2 = e.close(e.state.geom, e.state.lambda_bs)
    np.testing.assert_allclose(c1["x"], c2["x"], rtol=1e-9, atol=1e-12)
    assert c1["li_predicted"] == pytest.approx(LI0 * 1.01, abs=1e-9)


def test_two_scalar_li_with_soft_rows_uses_the_soft_solver():
    ad = T.ToyAdapter(soft=True, li_target=LI0 * 1.01)
    eng, res, rec, b = _run(ad, engine_preset="two_scalar_li",
                            engine_rows=["Ip", "l_i"])
    assert eng.soft
    assert res["converged"]
    d = rec["delivered"]
    assert d["ok"], d["misses"]
    li = d["checks"]["l_i"]
    assert not li["hard"]
    assert abs(li["error"]) <= T.settings()["structured_li_tol"]


@pytest.mark.parametrize("soft", [False, True])
def test_structured_uniform_converges_with_the_uniform_prior(soft):
    from bouquet.utils import STRUCTURED_WEIGHTS_UNIFORM
    ad = T.ToyAdapter(soft=soft, li_target=LI0 * 1.01)
    eng, res, rec, b = _run(ad, engine_preset="structured_uniform")
    assert res["converged"]
    assert rec["delivered"]["ok"], rec["delivered"]["misses"]
    assert eng.sigma_up is None
    np.testing.assert_array_equal(
        eng.weights["ind"], STRUCTURED_WEIGHTS_UNIFORM["ind"])
    np.testing.assert_array_equal(
        eng.weights["bs"], STRUCTURED_WEIGHTS_UNIFORM["bs"])
    assert "uniform" in eng.prior_name
    assert eng.soft is soft
    # a different prior, a different routing of the same rows
    e2, r2, rec2, b2 = _run(T.ToyAdapter(soft=soft, li_target=LI0 * 1.01))
    assert not np.array_equal(res["state"].request, r2["state"].request)
