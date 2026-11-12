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
    ct = [round(-1.6 + 0.05 * k, 6) for k in range(32)]
    tab = CrashTable(time=ct, time_err=np.zeros(32), rho_inv=np.zeros(32), rho_inv_err=np.zeros(32),
                     confidence=["A"] * 32, record_start=-1.7, record_end=1.0, unobserved=[(0.015, 0.05)])
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), CM.CurrentModelSettings(r0=SD.GEO.R0),
                        SE.SeriesSettings(initial="relaxed", crash_table=tab))
    assert out["settings"]["sawtooth_resolved"] == "crash_times"
    assert [w["slice"] for w in out["warnings"]] == [1, 2]
    assert all("did not cover" in w["sawtooth"] for w in out["warnings"])


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
