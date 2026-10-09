"""The legacy IMAS reader reads the sawteeth entry AT the slice time.

Owner-approved fix (2026-10-05), the rule the engine IDS adapter has had
since its own fix: a ``core_sources`` entry is read at the core_sources
slice TIME, not at its list index.  A model's sawteeth entry (identifier
701) that starts one slice after the IDS time base was read one slice late
by the reader's sawtooth gate (``Baseline.sawtooth``: ``present`` /
``j_par_max_abs`` / ``active``), and at the last slice from its FIRST
slice.  An entry with no per-slice time and a different slice count cannot
be aligned and is refused -- never its first slice in place of the missing
one.

Synthetic inputs only (the shipped D3D-like OMAS example, modified here).
"""
import json
import os
import warnings

import numpy as np
import pytest

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "examples", "D3D-like")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")


def _dd_with_late_sawteeth(with_times=True):
    """The example dd plus a sawteeth entry that starts one slice late; its
    j_parallel at core_sources time t_k is the constant 1e3*(k+1) A/m^2, so
    the slice read is visible in ``j_par_max_abs``."""
    import copy
    with open(_OMAS) as fh:
        dd = json.load(fh)
    t = list(dd["core_sources"]["time"])
    base = next(s for s in dd["core_sources"]["source"]
                if s["identifier"]["index"] == 2)
    saw = copy.deepcopy(base)
    saw["identifier"] = dict(name="sawteeth", index=701)
    n = len(saw["profiles_1d"][0]["j_parallel"])
    for k, q in enumerate(saw["profiles_1d"]):
        q["j_parallel"] = [1.0e3 * (k + 1)] * n
        if with_times:
            q["time"] = t[k]
    saw["profiles_1d"] = saw["profiles_1d"][1:]          # starts one late
    dd["core_sources"]["source"].append(saw)
    return dd, t


def _read(tmp_path, dd, time, name):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    p = tmp_path / name
    with open(p, "w") as fh:
        json.dump(dd, fh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=str(p), time=time))


def test_the_last_slice_reads_its_own_sawtooth_slice(tmp_path):
    """At the last slice the entry's OWN last slice is read (3e3), not its
    first (2e3, what the list-index rule fell back to)."""
    dd, t = _dd_with_late_sawteeth()
    bl = _read(tmp_path, dd, t[-1], "last.json")
    sw = bl.sawtooth
    assert sw["present"] is True and sw["active"] is True
    assert sw["j_par_max_abs"] == pytest.approx(1.0e3 * len(t), rel=0,
                                                abs=0)
    assert sw["slice"] == "matched by time"


def test_a_middle_slice_is_not_read_one_slice_late(tmp_path):
    dd, t = _dd_with_late_sawteeth()
    bl = _read(tmp_path, dd, t[1], "mid.json")
    assert bl.sawtooth["j_par_max_abs"] == 2.0e3        # not 3e3 (index 1)
    assert bl.sawtooth["slice"] == "matched by time"


def test_a_slice_before_the_entry_starts_is_not_active(tmp_path):
    """At the first time the entry has no slice: present, NOT active (the
    list-index rule read the entry's first slice -- the NEXT time's -- and
    admitted the slice as sawtoothing)."""
    dd, t = _dd_with_late_sawteeth()
    bl = _read(tmp_path, dd, t[0], "first.json")
    sw = bl.sawtooth
    assert sw["present"] is True
    assert sw["active"] is False and sw["j_par_max_abs"] == 0.0
    # 2026-10-06: the entry's nearest own slice (t_1) is more than half its
    # local time-step away -- recorded (a gate flag, not a current: not
    # refused, unlike a beam entry)
    assert sw["slice"].startswith("no profiles_1d slice within half a "
                                  "time-step of t =")


def test_an_entry_that_cannot_be_aligned_is_refused(tmp_path):
    dd, t = _dd_with_late_sawteeth(with_times=False)
    with pytest.raises(ValueError, match="cannot be aligned"):
        _read(tmp_path, dd, t[-1], "notime.json")


def test_an_entry_on_the_full_time_base_without_times_reads_by_index(
        tmp_path):
    """The common case -- every slice present, no per-slice time -- is
    read by index, exactly as before."""
    dd, t = _dd_with_late_sawteeth(with_times=False)
    saw = dd["core_sources"]["source"][-1]
    saw["profiles_1d"].insert(0, dict(saw["profiles_1d"][0],
                                      j_parallel=[1.0e3] *
                                      len(saw["profiles_1d"][0]
                                          ["j_parallel"])))
    for k, tk in enumerate(t):
        bl = _read(tmp_path, dd, tk, f"full{k}.json")
        assert bl.sawtooth["j_par_max_abs"] == 1.0e3 * (k + 1)
        assert bl.sawtooth["slice"] == "by index"


def test_the_shipped_example_has_no_sawteeth_entry(tmp_path):
    """Nothing moves for the shipped example: no 701 entry, no 'slice'."""
    with open(_OMAS) as fh:
        dd = json.load(fh)
    bl = _read(tmp_path, dd, dd["core_sources"]["time"][-1], "ex.json")
    assert bl.sawtooth["present"] is False and "slice" not in bl.sawtooth
