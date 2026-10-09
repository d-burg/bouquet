"""ImasSource.ida_time and the ida_hybrid timing guards (io.imas._hybrid_timing,
_check_replay_pairing)."""
import json
import warnings

import pytest

from bouquet.config import ImasSource
from bouquet.io.imas import _check_replay_pairing, _hybrid_timing


def test_ida_time_defaults_to_time():
    src = ImasSource(ids_path="x.json", time=1.04)
    assert src.ida_time is None
    assert _hybrid_timing(src, 1.04, 1.04, 1.04, {}) == 1.04
    src = ImasSource(ids_path="x.json", time=1.04, ida_time=1.033)
    aux = {}
    assert _hybrid_timing(src, 1.04, 1.04, 1.04, aux) == 1.033
    assert aux == {"fuse_time_cp": 1.04, "fuse_time_eq": 1.04}


def test_non_macro_step_warns():
    src = ImasSource(ids_path="x.json", time=1.02)
    with pytest.warns(UserWarning, match="not a dd macro step"):
        _hybrid_timing(src, 1.02, 1.02, 1.04, {})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _hybrid_timing(src, 1.04, 1.04 + 1e-9, 1.04, {})


def test_replay_pairing_check(tmp_path):
    ids = tmp_path / "dd_sim.json"
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033}
    _check_replay_pairing(str(ids), aux)
    assert aux["pairing_consistent"] is None          # no table beside the dd
    (tmp_path / "ida_provenance.json").write_text(json.dumps({"ida_path": "x.cdf"}))
    _check_replay_pairing(str(ids), aux)
    assert aux["pairing_consistent"] is None          # an older table: no replay_pairing
    rows = [{"t_sim": 1.04, "tick": 1.04 - 1e-9, "ida_time": 1.033, "outcome": "ida"},
            {"t_sim": 1.08, "tick": 1.08 - 1e-9, "ida_time": 1.074, "outcome": "ida"}]
    (tmp_path / "ida_provenance.json").write_text(json.dumps({"replay_pairing": rows}))
    _check_replay_pairing(str(ids), aux)
    assert aux["pairing_consistent"] is True
    aux = {"fuse_time_cp": 1.08, "ida_time_used": 1.054}
    with pytest.warns(UserWarning, match="computed on IDA 1.074"):
        _check_replay_pairing(str(ids), aux)
    assert aux["pairing_consistent"] is False and aux["replayed_ida_time"] == 1.074
