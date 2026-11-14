"""Series defaults: no prescribed sawtooth train (measured crash tables only, a recorded warning when the evolved q
falls below 1 without crashes), the crash-table reader, and one Z_eff source per series.

Mocked forward model and synthetic geometry as in ``test_series_driver.py`` (no OFT).
"""
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_series_driver as SD  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon import zeff_sources as ZR  # noqa: E402
from bouquet.kinrecon.crash_table import CrashTable, read_crash_table  # noqa: E402

TABLE = """# made-up crash table
shot,t_s,t_err_s,rho_inv,rho_inv_err,conf
1,0.0500,0.0002,0.20,0.01,A
1,0.0100,0.0002,0.18,0.02,B
1,0.0300,0.0002,,0.02,C
"""


def test_crash_table_reader():
    t = read_crash_table(TABLE, record_start=-1.0, record_end=0.2, unobserved=[(0.12, 0.15)])
    np.testing.assert_allclose(t.time, [0.01, 0.03, 0.05])                  # sorted by time
    assert t.confidence == ["B", "C", "A"] and np.isnan(t.rho_inv[1])
    assert t.covered(0.0, 0.1) is True and t.covered(0.1, 0.13) is False and t.covered(0.15, 0.3) is False
    assert t.meta["comments"] == ["made-up crash table"]
    ab = read_crash_table(TABLE, min_confidence={"A", "B"})
    np.testing.assert_allclose(ab.time, [0.01, 0.05])
    assert ab.meta["n_dropped_by_confidence"] == 1 and ab.covered(0.0, 0.1) is None
    assert "not stated" in ab.meta["coverage_note"]
    renamed = read_crash_table(TABLE.replace("t_s", "time"), columns=dict(time="time"))
    assert renamed.time.size == 3
    with pytest.raises(ValueError, match="time column"):
        read_crash_table(TABLE.replace("t_s", "time"))
    with pytest.raises(ValueError, match="duplicate"):
        read_crash_table(TABLE + "1,0.0500,0.0002,0.2,0.01,A\n")
    with pytest.raises(ValueError, match="unknown column roles"):
        read_crash_table(TABLE, columns=dict(bogus="x"))


def test_default_sawtooth_handling_is_no_prescribed_train():
    s = SE.SeriesSettings()
    assert s.sawtooth == "off" and s.sawtooth_requested == "auto"
    d = s.as_dict()
    assert d["sawtooth_resolved"] == "off" and "warning" in d["sawtooth_rule"]
    tab = CrashTable(time=[-0.5, -0.4, 0.01], time_err=[0, 0, 0], rho_inv=[0.2] * 3, rho_inv_err=[0.0] * 3,
                     confidence="AAA", record_start=-2.0, record_end=1.0)
    s2 = SE.SeriesSettings(crash_table=tab)
    assert s2.sawtooth == "crash_times" and s2.crash_times == (-0.5, -0.4, 0.01) and s2.crash_record_start == -2.0
    assert s2.as_dict()["crash_table"]["n"] == 3
    p = SE.SeriesSettings(sawtooth="periodic")
    assert p.sawtooth == "periodic" and "EXPLICIT OPTION" in p.as_dict()["sawtooth_rule"]
    with pytest.raises(ValueError, match="not both"):
        SE.SeriesSettings(crash_table=tab, crash_times=[0.1])


def test_q_below_one_without_crashes_is_recorded(monkeypatch):
    ip = SD.IP0
    lag = SD._lag(q0=0.9)                                    # the solved equilibrium's q < 1 on axis
    fm = SD.FakeFM(dict(pred=lag))
    SD._install(monkeypatch, fm, SD._steady(lag, ip), lambda f: f.lags["pred"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), CM.CurrentModelSettings(r0=SD.GEO.R0),
                            SE.SeriesSettings(initial="relaxed"))
    assert out["status"] == "completed" and out["settings"]["sawtooth_resolved"] == "off"
    assert len(out["warnings"]) == 3 and all("no crash record" in x["sawtooth"] for x in out["warnings"])
    assert all("sawtooth_warning" in s for s in out["slices"])
    assert any("q_min" in str(x.message) for x in w)
    # no warning when q >= 1
    lag1 = SD._lag(q0=1.1)
    fm2 = SD.FakeFM(dict(pred=lag1))
    SD._install(monkeypatch, fm2, SD._steady(lag1, SD.IP0), lambda f: f.lags["pred"])
    out2 = SE.run_series(fm2, SD._slices(np.arange(3) * 0.02, np.full(3, SD.IP0)),
                         CM.CurrentModelSettings(r0=SD.GEO.R0), SE.SeriesSettings(initial="relaxed"))
    assert "warnings" not in out2


def test_crash_table_coverage_gap_is_recorded(monkeypatch):
    ip = SD.IP0
    lag = SD._lag(q0=0.9)
    fm = SD.FakeFM(dict(pred=lag))
    SD._install(monkeypatch, fm, SD._steady(lag, ip), lambda f: f.lags["pred"])
    ct = [round(-1.6 + 0.05 * k, 6) for k in range(30)]                 # last recorded crash at -0.15 s
    tab = CrashTable(time=ct, time_err=np.zeros(30), rho_inv=np.zeros(30), rho_inv_err=np.zeros(30),
                     confidence=["A"] * 30, record_start=-1.7, record_end=1.0, unobserved=[(0.015, 0.05)])
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), CM.CurrentModelSettings(r0=SD.GEO.R0),
                        SE.SeriesSettings(initial="relaxed", crash_table=tab))
    assert out["settings"]["sawtooth_resolved"] == "crash_times"
    # slice 0 lies in an observed stretch 0.15 s after the last recorded crash, three times the recorded crash
    # interval: recorded as a model-data inconsistency; 1 and 2 overlap the unobserved interval
    assert [w["slice"] for w in out["warnings"]] == [0, 1, 2]
    kinds = [s_["sawtooth_warning"]["kind"] for s_ in out["slices"]]
    assert kinds == ["covered_no_crash", "not_covered", "not_covered"]
    assert "model-data inconsistency" in out["warnings"][0]["sawtooth"]
    assert all("did not cover" in w["sawtooth"] for w in out["warnings"][1:])


def _zrec(source, imp_state="qualified", line_state="qualified"):
    return dict(zeff_source="checked_candidate",
                zeff_rule=dict(candidates={ZR.IMPURITY: dict(state=imp_state, failed=[]),
                                           ZR.LINE: dict(state=line_state, failed=[])},
                               decision=dict(rule_source=source, source=source)))


def _zslices(recs):
    return [SimpleNamespace(time=0.02 * k, kinetics=SimpleNamespace(record=r)) for k, r in enumerate(recs)]


def test_one_zeff_source_per_series():
    assert SE.check_zeff_series(_zslices([{}, {}]))["applied"] is False
    named = SE.check_zeff_series(_zslices([dict(zeff_source="carbon")] * 3))
    assert named["applied"] and named["candidate"] == "carbon"
    with pytest.raises(ValueError, match="switches"):
        SE.check_zeff_series(_zslices([dict(zeff_source="carbon"), dict(zeff_source="file")]))
    # declared intervals: the switch is allowed only where declared, and the per-slice checks stay active inside
    sched = [dict(start=-1.0, end=0.01, candidate="const"), dict(start=0.01, end=1.0, candidate="checked_candidate",
                                                                 source=ZR.IMPURITY)]
    mixed = [dict(zeff_source="const"), _zrec(ZR.IMPURITY, imp_state=ZR.FAILED), _zrec(ZR.IMPURITY)]
    out = SE.check_zeff_series(_zslices(mixed), schedule=sched)
    assert out["failing_slices"] == [0.02] and out["schedule"][1]["source"] == ZR.IMPURITY
    assert [iv["n_slices"] for iv in out["intervals"]] == [1, 2]
    with pytest.raises(ValueError, match="used another source"):           # a candidate switch inside an interval
        SE.check_zeff_series(_zslices([dict(zeff_source="const"), _zrec(ZR.IMPURITY), _zrec(ZR.LINE)]),
                             schedule=sched)
    with pytest.raises(ValueError, match="declared"):                        # an undeclared switch
        SE.check_zeff_series(_zslices([dict(zeff_source="const"), dict(zeff_source="const"), _zrec(ZR.IMPURITY)]),
                             schedule=sched)
    with pytest.raises(ValueError, match="no declared interval"):
        SE.check_zeff_series(_zslices(mixed), schedule=sched[1:])
    recs = [_zrec(ZR.IMPURITY), _zrec(ZR.IMPURITY, imp_state=ZR.FAILED), _zrec(ZR.IMPURITY, imp_state=ZR.UNVERIFIED)]
    with pytest.raises(ZR.ZeffSourceRefused, match="needs one Z_eff source named"):
        SE.check_zeff_series(_zslices(recs))
    out = SE.check_zeff_series(_zslices(recs), ZR.IMPURITY)
    assert out["failing_slices"] == [0.02] and out["unverified_slices"] == [0.04]
    with pytest.raises(ValueError, match="used another source"):
        SE.check_zeff_series(_zslices([_zrec(ZR.IMPURITY), _zrec(ZR.LINE)]), ZR.IMPURITY)


def test_crash_record_with_no_crash_in_the_covered_history_starts_crash_free():
    s = SE.SeriesSettings(initial="relaxed", sawtooth="crash_times", crash_times=[0.03, 0.05],
                          crash_record_start=-2.0)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0)
    cfg0, p0, rec = SE._measured_crash_start(s, cfg, np.array([0.0, 0.02]))
    assert cfg0 is cfg and p0["log_strength"] is None and rec["n_crashes_history"] == 0
    one = SE.SeriesSettings(initial="relaxed", sawtooth="crash_times", crash_times=[-0.3, 0.05], crash_record_start=-2.0)
    with pytest.raises(ValueError, match="one crash in the history"):
        SE._measured_crash_start(one, cfg, np.array([0.0, 0.02]))
    uncovered = SE.SeriesSettings(initial="relaxed", sawtooth="crash_times", crash_times=[0.03, 0.05])
    with pytest.raises(ValueError, match="cannot start the series"):
        SE._measured_crash_start(uncovered, cfg, np.array([0.0, 0.02]))


def test_crash_free_start_respects_unobserved_history():
    """No crash recorded in a history the record did not observe is not 'no crash': the start is refused."""
    tab = CrashTable(time=[0.03, 0.05], time_err=[0, 0], rho_inv=[0.2, 0.2], rho_inv_err=[0, 0], confidence="AA",
                     record_start=-2.0, record_end=1.0, unobserved=[(-1.6, 0.5)])
    s = SE.SeriesSettings(initial="relaxed", crash_table=tab)
    with pytest.raises(ValueError, match="not 'no crash'"):
        SE._measured_crash_start(s, CM.CurrentModelSettings(r0=SD.GEO.R0), np.array([0.0, 0.02]))
    ok = CrashTable(time=[0.03, 0.05], time_err=[0, 0], rho_inv=[0.2, 0.2], rho_inv_err=[0, 0], confidence="AA",
                    record_start=-2.0, record_end=1.0)
    cfg0, p0, rec = SE._measured_crash_start(SE.SeriesSettings(initial="relaxed", crash_table=ok),
                                             CM.CurrentModelSettings(r0=SD.GEO.R0), np.array([0.0, 0.02]))
    assert rec["n_crashes_history"] == 0 and p0["log_strength"] is None


class _Ev(SimpleNamespace):
    pass


def _q_res(q0):
    lag = SD._lag(q0=q0)
    return SimpleNamespace(converged=True, lag=lag)


def test_q_below_one_warning_fires_only_for_the_inconsistency_it_names():
    """Sawtoothing covered by the record: q < 1 within the recorded crash spacing since the last crash is normal
    (recorded, no warning); a crash-free covered stretch longer than any recorded interval is the inconsistency;
    a recorded crash that did not act is named as such (never 'shows no crash')."""
    ct = [0.0, 0.05, 0.11, 0.16]                                   # recorded intervals 0.05, 0.06, 0.05 s
    tab = CrashTable(time=ct, time_err=np.zeros(4), rho_inv=np.zeros(4), rho_inv_err=np.zeros(4), confidence="AAAA",
                     record_start=-1.0, record_end=1.0)
    s = SE.SeriesSettings(initial="relaxed", crash_table=tab)
    res = _q_res(0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normal = SE._sawtooth_warning(s, res, 0.18, 0.20, [])     # 0.04 s after the last crash
        overdue = SE._sawtooth_warning(s, res, 0.20, 0.22, [])    # 0.06 s: equal to the longest interval
        late = SE._sawtooth_warning(s, res, 0.22, 0.24, [])       # 0.08 s: longer than any recorded interval
        ev = [_Ev(time=0.21, record=SimpleNamespace(active=False))]
        inactive = SE._sawtooth_warning(s, res, 0.20, 0.22, ev)
        early = SE._sawtooth_warning(s, res, -0.1, -0.08, [])     # covered, before any recorded crash
    assert normal["kind"] == "between_crashes" and normal["warn"] is False
    assert normal["gap_since_last_crash_s"] == pytest.approx(0.04) and normal["longest_recorded_interval_s"] == \
        pytest.approx(0.06)
    assert overdue["kind"] == "between_crashes" and overdue["warn"] is False
    assert late["kind"] == "covered_no_crash" and late["warn"] is True and "longer than the longest" in late["text"]
    assert inactive["kind"] == "crash_recorded_not_active" and inactive["warn"] is True
    assert "shows no crash" not in inactive["text"] and "0.2100" in inactive["text"]
    assert early["kind"] == "covered_no_crash" and "shows no crash at all" in early["text"]
    assert SE._sawtooth_warning(s, _q_res(1.1), 0.22, 0.24, []) is None          # q >= 1: nothing


def test_q_below_one_between_crashes_is_recorded_but_not_warned_in_a_series(monkeypatch):
    ip = SD.IP0
    lag = SD._lag(q0=0.9)
    fm = SD.FakeFM(dict(pred=lag))
    SD._install(monkeypatch, fm, SD._steady(lag, ip), lambda f: f.lags["pred"])
    ct = [round(-1.6 + 0.05 * k, 6) for k in range(32)] + [0.01, 0.06]     # a crash train through the series
    tab = CrashTable(time=ct, time_err=np.zeros(34), rho_inv=np.zeros(34), rho_inv_err=np.zeros(34),
                     confidence=["A"] * 34, record_start=-1.7, record_end=1.0)
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), CM.CurrentModelSettings(r0=SD.GEO.R0),
                        SE.SeriesSettings(initial="relaxed", crash_table=tab))
    for s_ in out["slices"]:
        if "q_below_one" in s_:
            assert s_["q_below_one"]["kind"] in ("between_crashes", "crash_recorded_not_active")
    assert not any(w_["sawtooth"].find("model-data inconsistency") >= 0 for w_ in out.get("warnings", []))


def test_crash_start_refuses_an_unobserved_history_whatever_the_crash_count():
    """Crashes listed in a history the record did not fully observe do not set the starting crash rate."""
    tab = CrashTable(time=[-1.45, -1.40, 0.03], time_err=[0, 0, 0], rho_inv=[0.2] * 3, rho_inv_err=[0.0] * 3,
                     confidence="AAA", record_start=-2.0, record_end=1.0, unobserved=[(-1.3, 0.0)])
    s = SE.SeriesSettings(initial="relaxed", crash_table=tab)
    with pytest.raises(ValueError, match="did not observe all of it"):
        SE._measured_crash_start(s, CM.CurrentModelSettings(r0=SD.GEO.R0), np.array([0.0, 0.02]))
    ok = CrashTable(time=[-1.45, -1.40, 0.03], time_err=[0, 0, 0], rho_inv=[0.2] * 3, rho_inv_err=[0.0] * 3,
                    confidence="AAA", record_start=-2.0, record_end=1.0)
    cfg0, p0, rec = SE._measured_crash_start(SE.SeriesSettings(initial="relaxed", crash_table=ok),
                                             CM.CurrentModelSettings(r0=SD.GEO.R0), np.array([0.0, 0.02]))
    assert rec["n_crashes_history"] >= 2


def test_overlapping_zeff_intervals_are_refused():
    sched = [dict(start=-1.0, end=0.03, candidate="const"), dict(start=0.01, end=1.0, candidate="const")]
    with pytest.raises(ValueError, match="overlap"):
        SE.check_zeff_series(_zslices([dict(zeff_source="const")] * 3), schedule=sched)
    touching = [dict(start=-1.0, end=0.01, candidate="const"), dict(start=0.01, end=1.0, candidate="const")]
    assert SE.check_zeff_series(_zslices([dict(zeff_source="const")] * 3), schedule=touching)["applied"]
