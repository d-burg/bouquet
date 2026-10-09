"""``GenerationConfig.engine_li_row_relaxation``: an under-relaxation of the
unified engine's l_i-row discrepancy update -- fast half (no GS solver).

The l_i row's discrepancy is updated between passes as

    d_k = (1 - r w) d_k-1 + r w [l_i(E_k+1) - l_i_model(js_k; G_k+1)]

with ``w`` the loop's bootstrap omega (the first update, from ``d = 0``,
takes ``w = 1``).  ``r = 1`` (the default) is the update before the setting
existed.  Checked on the toy Grad-Shafranov stand-in (``tests/_engine_toy``):

* the default is 1.0, it reaches the engine settings and the record, and the
  setting is validated by name (``0 < r <= 1``; refused under ``"legacy"``);
* at the default the reconstruction is bit-identical to one run with the
  setting absent from the settings dict (the code path before it existed);
* every update follows the formula above, at ``r = 1`` and ``r = 0.5``;
* ``r = 0.5`` changes the path only: the loop still converges under the SAME
  criteria and the delivered l_i meets the same hard-row tolerance.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import io

import numpy as np
import pytest

import _engine_toy as T
from bouquet.config import GenerationConfig
from bouquet.engine import (ENGINE_FIELD_DEFAULTS, UnifiedEngine,
                            engine_settings, gfile_li_row_tol, reconstruct,
                            validate_engine_settings)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _li0():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"]


LI0 = _li0()


def _run(settings, li_factor=1.01):
    ad = T.ToyAdapter(li_target=LI0 * li_factor)
    b = T.ToyGS()
    ad.read()
    eng, res, rec = _quiet(reconstruct, ad, b, settings, label="toy")
    return eng, res, rec, b


# ---------------------------------------------------------------------------
#  default, settings, validation
# ---------------------------------------------------------------------------
def test_the_default_is_one_and_reaches_the_settings_and_the_record():
    assert GenerationConfig().engine_li_row_relaxation == 1.0
    assert ENGINE_FIELD_DEFAULTS["engine_li_row_relaxation"] == 1.0
    s = T.settings()
    assert s["li_row_relaxation"] == 1.0
    assert T.settings(engine_li_row_relaxation=0.5)["li_row_relaxation"] \
        == 0.5
    _eng, _res, rec, _b = _run(T.settings(engine_li_row_relaxation=0.5))
    assert rec["settings"]["li_row_relaxation"] == 0.5
    assert "under-relaxed" in rec["row_update"]
    _eng, _res, rec1, _b = _run(T.settings())
    assert rec1["settings"]["li_row_relaxation"] == 1.0
    assert "under-relaxed" not in rec1["row_update"]


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan"), float("inf"),
                                 True, "0.5", None])
def test_out_of_range_or_non_numeric_values_are_refused_by_name(bad):
    with pytest.raises(ValueError, match="engine_li_row_relaxation"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_li_row_relaxation=bad))


@pytest.mark.parametrize("ok", [1.0, 1, 0.5, 1e-3])
def test_values_in_range_are_accepted(ok):
    validate_engine_settings(GenerationConfig(
        reconstruction_engine="unified", engine_li_row_relaxation=ok))


def test_a_set_value_with_the_legacy_engine_is_refused():
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", engine_li_row_relaxation=0.5))
    # the default (as 1.0 or 1) is not a change
    validate_engine_settings(GenerationConfig(
        reconstruction_engine="legacy", engine_li_row_relaxation=1.0))
    validate_engine_settings(GenerationConfig(
        reconstruction_engine="legacy", engine_li_row_relaxation=1))


# ---------------------------------------------------------------------------
#  the default is the code path before the setting existed
# ---------------------------------------------------------------------------
def test_the_default_is_bit_identical_to_the_settings_without_the_field():
    s1 = T.settings()
    s0 = {k: v for k, v in s1.items() if k != "li_row_relaxation"}
    _e1, res1, rec1, b1 = _run(s1)
    _e0, res0, rec0, b0 = _run(s0)
    assert res1["converged"] and res0["converged"]
    assert rec1["phases"][0]["record"]["n_passes"] \
        == rec0["phases"][0]["record"]["n_passes"]
    assert np.asarray(rec1["state"]["request"]).tobytes() \
        == np.asarray(rec0["state"]["request"]).tobytes()
    assert rec1["state"]["li_discrepancy"] == rec0["state"]["li_discrepancy"]
    for p1, p0 in zip(rec1["passes"], rec0["passes"]):
        assert p1.get("li_discrepancy_next") == p0.get("li_discrepancy_next")
        assert p1["li"] == p0["li"]


# ---------------------------------------------------------------------------
#  the update law
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [1.0, 0.5])
def test_every_update_follows_the_relaxed_formula(monkeypatch, r):
    seen = []
    orig = UnifiedEngine.on_pass

    def spy(self, k, meas, J, entry):
        d_before = float(self.state.li_discrepancy)
        first = bool(self._d_first)
        p = self._pending
        raw = float(p["m"]["li"]) - float(p["li_model_solved_new_geom"])
        out = orig(self, k, meas, J, entry)
        om = entry.get("omega_next")
        if om is not None:
            seen.append((first, d_before, raw, float(om),
                         float(self.state.li_discrepancy)))
        return out

    monkeypatch.setattr(UnifiedEngine, "on_pass", spy)
    _eng, res, _rec, _b = _run(T.settings(engine_li_row_relaxation=r))
    assert res["converged"]
    assert len(seen) >= 2
    for first, d, raw, om, d_new in seen:
        w = r * (1.0 if first else om)
        want = (raw if (first and r == 1.0) else (1.0 - w) * d + w * raw)
        assert d_new == pytest.approx(want, rel=0.0, abs=1e-15)
    assert seen[0][0] and not any(s[0] for s in seen[1:])


def test_under_relaxation_changes_the_path_not_the_criteria():
    _e1, res1, rec1, _b1 = _run(T.settings())
    _e5, res5, rec5, _b5 = _run(T.settings(engine_li_row_relaxation=0.5))
    for res, rec in ((res1, rec1), (res5, rec5)):
        assert res["converged"] and rec["delivered"]["ok"]
        li = rec["delivered"]["checks"]["l_i"]
        # the same hard-row tolerance on the delivered state
        assert abs(li["delivered"] - LI0 * 1.01) <= gfile_li_row_tol()
        assert li["tol"] == gfile_li_row_tol()
    # the convergence table (every criterion and ceiling) is unchanged
    assert rec5["convergence"] == rec1["convergence"]
    assert rec5["settings"]["loop"] == rec1["settings"]["loop"]
    # the first discrepancy step is halved
    d1 = rec1["passes"][0]["li_discrepancy_next"]
    d5 = rec5["passes"][0]["li_discrepancy_next"]
    assert d5 == pytest.approx(0.5 * d1, rel=1e-12)


def test_engine_settings_of_a_gc_without_the_field_default_to_one():
    class _GC:
        pass
    g = GenerationConfig(reconstruction_engine="unified")
    gc = _GC()
    for k, v in vars(g).items():
        if k != "engine_li_row_relaxation":
            setattr(gc, k, v)
    assert engine_settings(gc)["li_row_relaxation"] == 1.0
