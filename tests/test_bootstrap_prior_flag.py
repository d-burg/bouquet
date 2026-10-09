"""The +/-50 % bootstrap prior is a FLAG on every path (2026-10-06).

``|s_bs - 1| > 0.5`` is a closure failure, not a finding (the owner's
rule): :func:`bouquet.utils.closure_health` flags it as
``bootstrap_scale_out_of_prior`` -- both sides; until 2026-10-06 only
``bs_scale < 0.5`` was flagged and a scale above 1.5 passed silently -- and
prints / warns it, never clamping.  The g-file engine path (which never
called ``closure_health``) and the legacy g-file path now record it too.

Synthetic records only; no solver, no device data.
"""
import os
import sys
import warnings

import numpy as np
import pytest

from bouquet.utils import (BOOTSTRAP_PRIOR_FLAG, BS_SCALE_PRIOR_HALFWIDTH,
                           bootstrap_prior_reason, closure_health,
                           merge_closure_flags)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_engine_draws as TD  # noqa: E402

IP = 1.0e6


def _health(bs):
    # raw components that close Ip at s = 1 (no mismatch flag)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = closure_health(1.0, bs, IP, 0.0, 7.0e5, 2.0e5, 1.0e5)
    return h, w


@pytest.mark.parametrize("bs, flagged", [(0.4, True), (0.6, False),
                                         (1.4, False), (1.6, True)])
def test_the_prior_flags_both_sides_and_never_clamps(bs, flagged, capsys):
    h, w = _health(bs)
    prior = [r for r in h["closure_limited_reasons"]
             if r.startswith(BOOTSTRAP_PRIOR_FLAG)]
    assert bool(prior) is flagged
    assert h["closure_limited"] is flagged
    # never clamped: the closed fraction is the scale actually used
    assert h["f_BS_closed"] == pytest.approx(bs * 2.0e5 / IP)
    th = h["closure_limited_thresholds"]
    # the record's keys are unchanged (pinned by test_structured_mse_optin)
    assert set(th) == {"mismatch_max_pct", "bs_scale_min"}
    assert th["bs_scale_min"] == 1.0 - BS_SCALE_PRIOR_HALFWIDTH == 0.5
    out = capsys.readouterr().out
    if flagged:
        assert "bs_scale" in prior[0] and f"{bs:.3f}" in prior[0]
        assert "closure failure" in prior[0]
        assert "WARNING" in out and BOOTSTRAP_PRIOR_FLAG in out
        assert any(BOOTSTRAP_PRIOR_FLAG in str(x.message) for x in w)
    else:
        assert BOOTSTRAP_PRIOR_FLAG not in out
        assert not w


def test_the_reason_helper_and_the_merge():
    assert bootstrap_prior_reason(1.5) is None          # on the edge: inside
    assert bootstrap_prior_reason(0.5) is None
    assert bootstrap_prior_reason(float("nan")) is None  # unreadable elsewhere
    assert bootstrap_prior_reason(1.6).startswith(BOOTSTRAP_PRIOR_FLAG)
    m = dict(closure_limited_reasons=("earlier",), closure_limited=True)
    h, _ = _health(1.6)
    merge_closure_flags(m, h)
    merge_closure_flags(m, h)                            # no repeat
    assert m["closure_limited"] is True
    assert m["closure_limited_reasons"][0] == "earlier"
    assert sum(r.startswith(BOOTSTRAP_PRIOR_FLAG)
               for r in m["closure_limited_reasons"]) == 1
    clean = {}
    merge_closure_flags(clean, _health(1.0)[0])
    assert "closure_limited" not in clean


class _FakeEngine:
    def __init__(self, bs_eff):
        self.delivered_closure = dict(
            out=dict(ohm_scale_eff=1.0, bs_scale_eff=bs_eff,
                     Ip_lin_ind=7.0e5, Ip_lin_bs=2.0e5, Ip_lin_fix=1.0e5,
                     s_bs=np.array([bs_eff - 0.1, bs_eff, bs_eff + 0.1])),
            Ip_signed=IP, c_signed=0.0)


@pytest.mark.parametrize("bs, flagged", [(0.4, True), (0.6, False),
                                         (1.4, False), (1.6, True)])
def test_the_engine_evaluates_it_on_its_delivered_closure(bs, flagged):
    from bouquet.engine import engine_closure_health
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ch = engine_closure_health(_FakeEngine(bs), "test")
    assert ch["bs_scale"] == bs
    assert ch["bs_scale_basis"].startswith("bs_scale_eff")
    assert ch["s_bs_range"] == pytest.approx([bs - 0.1, bs + 0.1])
    th = ch["closure_limited_thresholds"]
    assert (th["bs_scale_min"], th["bs_scale_max"]) == (0.5, 1.5)
    assert any(r.startswith(BOOTSTRAP_PRIOR_FLAG)
               for r in ch["closure_limited_reasons"]) is flagged


toy_bouquet_solver = TD.toy_bouquet_solver


@pytest.mark.parametrize("bs, flagged", [(0.4, True), (0.6, False),
                                         (1.4, False), (1.6, True)])
def test_the_gfile_engine_baseline_records_it(tmp_path, toy_bouquet_solver,
                                              monkeypatch, bs, flagged):
    """The g-file engine baseline: closure_health on the delivered closure,
    in reconstruction_metrics["closure_health"], its flag folded into the
    baseline's closure_limited (the toy reconstruction, its delivered
    closure's effective scale set to *bs*)."""
    import bouquet.engine as be
    real = be.engine_closure_health

    def forced(eng, where):
        eng.delivered_closure["out"] = dict(eng.delivered_closure["out"],
                                            bs_scale_eff=bs)
        return real(eng, where)
    monkeypatch.setattr(be, "engine_closure_health", forced)
    b = TD._bq(tmp_path)
    b.setup_solver()
    bl = TD._quiet(b.prepare_baseline)
    m = bl.reconstruction_metrics
    assert m["closure_health"]["bs_scale"] == bs
    has = any(str(r).startswith(BOOTSTRAP_PRIOR_FLAG)
              for r in (m.get("closure_limited_reasons") or ()))
    assert has is flagged
    if flagged:
        assert m["closure_limited"] is True
    # only the prior is folded into the g-file baseline's flags
    assert m["closure_health"]["folded_into_baseline_flags"] == [
        r for r in m["closure_health"]["closure_limited_reasons"]
        if r.startswith(BOOTSTRAP_PRIOR_FLAG)]


@pytest.mark.parametrize("bs, flagged", [(0.4, True), (0.6, False),
                                         (1.4, False), (1.6, True)])
def test_the_legacy_gfile_record_of_the_fit_scale(bs, flagged, capsys):
    """The legacy g-file path: the inductive fit's bootstrap scale
    (rescale_j_BS) against the same prior, recorded and flagged
    (baseline.py builds reconstruction_metrics["closure_health"] with
    utils.bootstrap_prior_record from the reconstruction's bs_scale_fit)."""
    from bouquet.utils import bootstrap_prior_record
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec = bootstrap_prior_record(bs, "fit", "g-file reconstruction")
    assert rec["bs_scale"] == bs and rec["closure_limited"] is flagged
    assert bool(w) is flagged
    assert (BOOTSTRAP_PRIOR_FLAG in capsys.readouterr().out) is flagged
    m = {}
    merge_closure_flags(m, rec)
    assert m.get("closure_limited", False) is flagged
    import inspect
    from bouquet import baseline as B
    assert 'result.get("bs_scale_fit", 1.0)' in inspect.getsource(B)
