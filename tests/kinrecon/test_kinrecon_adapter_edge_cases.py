"""Edge cases of the IMAS JSON adapter and the measurement file that used to pass silently (made-up data).

* time / data / data_error_upper / validity_timed of different lengths: refused (were truncated, and a short
  validity_timed was ignored so a flagged sample entered the mean);
* a scalar data_error_upper is broadcast to every sample (it crashed);
* a channel without data_error_upper is invalid with the reason recorded (only n_samples = 0 was a trace);
* strict JSON: non-finite floats tagged, the file has no NaN/Infinity token and round-trips exactly.
"""
import json

import numpy as np
import pytest

from bouquet.kinrecon import measurements as M
from bouquet.kinrecon.adapters.imas_json import measurements_from_imas
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet

T = list(np.linspace(0, 1, 11))


def _ids(blk):
    return {"magnetics": {"flux_loop": [{"name": "L0", "position": [{"r": 1, "z": 0}], "flux": blk}]}}


def _loops(blk):
    return measurements_from_imas(_ids(blk), 0.5, 1.0, groups=["flux_loop"]).groups["flux_loop"]


def test_length_mismatches_are_refused():
    with pytest.raises(ValueError, match="time"):
        _loops({"time": T[:5], "data": [1.0] * 11, "data_error_upper": [0.1] * 11})
    with pytest.raises(ValueError, match="validity_timed"):
        _loops({"time": T, "data": [1.0] * 10 + [99.0], "data_error_upper": [0.1] * 11, "validity_timed": [0] * 10})
    with pytest.raises(ValueError, match="data_error_upper"):
        _loops({"time": T, "data": [1.0] * 11, "data_error_upper": [0.1] * 7})
    # full-length validity_timed is honored: the flagged sample stays out of the mean
    g = _loops({"time": T, "data": [1.0] * 10 + [99.0], "data_error_upper": [0.1] * 11,
                "validity_timed": [0] * 10 + [-1]})
    assert g.measured[0] == pytest.approx(1.0)


def test_scalar_error_and_missing_error():
    g = _loops({"time": T, "data": [1.0] * 11, "data_error_upper": 0.1})
    assert g.valid[0] and g.sigma[0] == pytest.approx(0.1)
    g = _loops({"time": T, "data": [1.0] * 11})
    assert not g.valid[0] and "no data_error_upper" in g.meta["invalid_reason"]["L0"]


def test_strict_json_round_trip(tmp_path):
    ms = MeasurementSet("toy", 1.0, 0.01, meta={"x": float("nan"), "y": [float("inf"), 1.0]})
    ms.groups["ip"] = ConstraintGroup("ip", ["ip"], [1.0e6], [float("nan")], [False])
    loose, strict = tmp_path / "a.json", tmp_path / "b.json"
    M.write_measurements(loose, ms)
    M.write_measurements(strict, ms, strict_json=True)
    assert "NaN" in loose.read_text()
    txt = strict.read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))     # strict parse
    for path in (loose, strict):
        back = M.read_measurements(path)
        assert np.isnan(back.groups["ip"].sigma[0]) and back.groups["ip"].measured[0] == 1.0e6
        assert np.isnan(back.meta["x"]) and back.meta["y"][0] == float("inf")
