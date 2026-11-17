"""The solver iteration-limit setting :attr:`bouquet.kinrecon.forward.ForwardSettings.solver_maxits` and the
per-solve iteration record, without OFT (the live checks are in ``test_solver_maxits_solver.py``).

* The setting: default None (OFT's own limit, never set), recorded in ``ForwardSettings.as_dict()``; only None or a
  whole number >= 1 is accepted (:func:`forward.check_solver_maxits`).
* :meth:`forward.ForwardModel.iteration_record` on a stated log: counts above OFT's own limit, solves stopped at the
  limit, unknown counts, the start index of a fit or a series.
* The fit record (``record["solver_iterations"]``) and the series record (``out["solver_iterations"]``) carry the
  iteration record of their own solves, one entry per solve counted in ``n_solves``; a forward model without the
  log leaves both records without the key.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import forward as FW  # noqa: E402

OFT_LIMIT = 40          # OFT's own default (read back from the solver in the live test; stated here for the stub)


def test_default_is_none_and_recorded():
    s = FW.ForwardSettings(vsc={"A": 1.0})
    assert s.solver_maxits is None
    assert "solver_maxits" in s.as_dict() and s.as_dict()["solver_maxits"] is None


@pytest.mark.parametrize("value, expected", [(None, None), (1, 1), (80, 80), (120.0, 120), (np.int64(60), 60)])
def test_check_accepts_none_and_whole_numbers(value, expected):
    assert FW.check_solver_maxits(value) == expected


@pytest.mark.parametrize("value", [0, -1, 2.5, float("nan"), float("inf"), True, False, "80", b"80", [80]])
def test_check_refuses_everything_else(value):
    with pytest.raises(ValueError):
        FW.check_solver_maxits(value)


def _stub(limit=None, log=None):
    """An object with what ``ForwardModel.iteration_record`` reads."""
    return SimpleNamespace(iteration_log=list(log or []),
                           iteration_limit=dict(setting=limit, in_effect=OFT_LIMIT if limit is None else limit,
                                                oft_default=OFT_LIMIT, rule="stub"))


def _entry(its, kind="solve", at_limit=False, label="x"):
    return dict(label=label, its=its, kind=kind, stopped_at_limit=at_limit)


def test_iteration_record_counts():
    log = [_entry(12), _entry(3, "verify"), _entry(41, label="col"), _entry(80, at_limit=True, label="trial"),
           _entry(None), _entry(40)]
    stub = _stub(80, log)
    r = FW.ForwardModel.iteration_record(stub)
    assert r["n_solves"] == 6 and r["its"] == [12, 3, 41, 80, None, 40]
    assert r["kind"] == ["solve", "verify", "solve", "solve", "solve", "solve"]
    assert r["n_unknown"] == 1 and r["max_its"] == 80
    assert r["n_over_oft_default"] == 2          # 41 and 80; 40 is not above OFT's own limit
    assert r["n_stopped_at_limit"] == 1
    assert [e["label"] for e in r["over_oft_default"]] == ["col", "trial"]
    assert r["limit"] == dict(setting=80, in_effect=80, oft_default=OFT_LIMIT, rule="stub")
    # from a start index: only the later solves
    r2 = FW.ForwardModel.iteration_record(stub, start=4)
    assert r2["its"] == [None, 40] and r2["n_over_oft_default"] == 0 and r2["max_its"] == 40


def test_iteration_record_default_limit_flags_solves_stopped_at_it():
    r = FW.ForwardModel.iteration_record(_stub(None, [_entry(40, at_limit=True), _entry(7)]))
    assert r["limit"]["setting"] is None and r["limit"]["in_effect"] == OFT_LIMIT
    assert r["n_over_oft_default"] == 0 and r["n_stopped_at_limit"] == 1 and len(r["over_oft_default"]) == 1
    empty = FW.ForwardModel.iteration_record(_stub(None, []))
    assert empty["n_solves"] == 0 and empty["max_its"] is None


# ------------------------------------------------------------------ fit record
def _fit_fm(with_log):
    import test_fit_unit as U
    fm = U.FakeFM()
    if with_log:
        fm.iteration_log = [_entry(5, label="before the fit"), _entry(6, label="before the fit")]
        fm.iteration_limit = dict(setting=90, in_effect=90, oft_default=OFT_LIMIT, rule="stub")
        fm.iteration_record = lambda start=0: FW.ForwardModel.iteration_record(fm, start=start)
    return U, fm


@pytest.mark.parametrize("with_log", [True, False])
def test_fit_record_carries_the_fits_own_solves(monkeypatch, with_log):
    U, fm = _fit_fm(with_log)
    U._install(monkeypatch, fm)
    n_high = [0]

    def logged(inner):
        def call(fm_, *a, **kw):
            n = fm_.n_solves
            out = inner(fm_, *a, **kw)
            if with_log:   # one log entry per solve the stubs count; every third one above OFT's limit
                for _ in range(fm_.n_solves - n):
                    high = len(fm_.iteration_log) % 3 == 2
                    n_high[0] += high
                    fm_.iteration_log.append(_entry(55 if high else 9, label="in the fit"))
            return out
        return call

    monkeypatch.setattr(FIT, "_consistent", logged(FIT._consistent))
    monkeypatch.setattr(FIT._cm, "lagged_solve", logged(FIT._cm.lagged_solve))
    kin, src = U._kin()
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit"),
                          FIT.FitSettings(free=U.SAW_FREE), params0=dict(log_strength=0.5))
    rec = out.record
    if not with_log:
        assert "solver_iterations" not in rec
        return
    it = rec["solver_iterations"]
    assert it["n_solves"] == rec["n_solves"] == fm.n_solves     # the two entries before the fit are not its own
    assert all(e["label"] == "in the fit" for e in it["over_oft_default"])
    assert it["n_over_oft_default"] == n_high[0] > 0 and it["max_its"] == 55
    assert it["limit"]["setting"] == 90


# ------------------------------------------------------------------ series record
@pytest.mark.parametrize("with_log", [True, False])
def test_series_record_carries_the_series_solves(monkeypatch, with_log):
    import test_series_driver as SD
    from bouquet.kinrecon import series as SE
    lag = SD._lag()
    fm = SD.FakeFM(dict(pred=lag))
    if with_log:
        fm.iteration_log = [_entry(5, label="earlier")]
        fm.iteration_limit = dict(setting=None, in_effect=OFT_LIMIT, oft_default=OFT_LIMIT, rule="stub")
        fm.iteration_record = lambda start=0: FW.ForwardModel.iteration_record(fm, start=start)
    SD._install(monkeypatch, fm, SD._steady(lag, SD.IP0), lambda f: f.lags["pred"])
    inner = SE._cm.consistent_solve

    def consistent(fm_, *a, **kw):
        n = fm_.n_solves
        out = inner(fm_, *a, **kw)
        if with_log:
            fm_.iteration_log.extend(_entry(11) for _ in range(fm_.n_solves - n))
        return out

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, SD.IP0)),
                        CM.CurrentModelSettings(r0=SD.GEO.R0), SE.SeriesSettings(initial="relaxed", sawtooth="off"))
    assert out["status"] == "completed"
    if not with_log:
        assert "solver_iterations" not in out
        return
    it = out["solver_iterations"]
    assert it["n_solves"] == out["n_solves"] > 0 and it["its"] == [11] * out["n_solves"]
    assert it["limit"]["setting"] is None and it["n_over_oft_default"] == 0


@pytest.mark.parametrize("case", ["fit_without_center", "fit_not_converged_refused", "relaxed_start_failed"])
def test_series_early_returns_carry_the_iteration_record(monkeypatch, case):
    """Every early return of run_series (the first-slice fit without a center, a non-converged first-slice fit
    refused by the default policy, a failed first-slice solve) carries solver_iterations for its own solves."""
    import test_series_driver as SD
    from bouquet.kinrecon import series as SE
    lag = SD._lag()
    st0 = SD._steady(lag, SD.IP0)
    fm = SD.FakeFM(dict(pred=lag))
    fm.iteration_log = [_entry(5, label="earlier")]
    fm.iteration_limit = dict(setting=70, in_effect=70, oft_default=OFT_LIMIT, rule="stub")
    fm.iteration_record = lambda start=0: FW.ForwardModel.iteration_record(fm, start=start)

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 2
        fm_.iteration_log.extend([_entry(61, label="first"), _entry(1, kind="verify")])
        return SD._result(lag, st0, ok=(case != "relaxed_start_failed"))

    def reconstruct(fm_, kin, src, cfg, settings, params0=None, initial_profiles=None, label="", **kw):
        fm_.n_solves += 3
        fm_.iteration_log.extend([_entry(61, label="fit"), _entry(1, kind="verify"), _entry(9, label="fit")])
        if case == "fit_without_center":
            rec = dict(center=None, center0=dict(status="model_failure", reason="stub"), iterations=[])
            return SimpleNamespace(status="failed_initial_consistency", record=rec, free=[])
        center = dict(res=SD._result(lag, st0), snap=dict(key="pred"), probe_correction=np.zeros(0))
        rec = dict(center=center, iterations=[], stopping_summary=dict(status="stalled_no_significant_descent"))
        return SimpleNamespace(status="stalled_no_significant_descent", record=rec, free=[])

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    monkeypatch.setattr(SE._fit, "reconstruct", reconstruct)
    initial = "relaxed" if case == "relaxed_start_failed" else "mse_fit"
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, SD.IP0)),
                        CM.CurrentModelSettings(r0=SD.GEO.R0), SE.SeriesSettings(initial=initial, sawtooth="off"))
    expected = dict(fit_without_center="failed_initial", fit_not_converged_refused="refused_initial_fit_not_converged",
                    relaxed_start_failed="failed_initial")[case]
    assert out["status"] == expected
    it = out["solver_iterations"]
    assert it["n_solves"] == out["n_solves"] > 0          # the entry logged before the series is not its own
    assert it["limit"]["setting"] == 70 and it["max_its"] == 61 and it["n_over_oft_default"] >= 1
    assert all(e["label"] != "earlier" for e in it["over_oft_default"])
