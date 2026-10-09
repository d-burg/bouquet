"""The legacy IMAS reader reads each NBI entry AT the slice time.

Owner-approved fix (2026-10-05), the rule of the sawteeth entry next to it
(``tests/test_imas_reader_sawtooth_slice.py``) and of the engine IDS
adapter: a ``core_sources`` beam entry (identifier 2) is read at the
core_sources slice TIME, not at its list index.  An entry that starts one
slice after the IDS time base was read one slice late, and past its last
slice from its FIRST slice.  An entry with no per-slice time and a different
slice count cannot be aligned and is refused -- never its first slice in
place of the missing one.

The time match (owner-approved 2026-10-06, replacing the 1e-6 s absolute
match, under which a beam entry a few microseconds off the time base was
dropped to ZERO with a warning): each entry is matched to its NEAREST own
slice and accepted within HALF its local time-step (the core_profiles step
for a single-time entry; 10 us -- ``IMAS_SINGLE_TIME_WINDOW_S``,
owner-approved 2026-10-07, it was float precision -- when neither grid has
a step);
otherwise the read is REFUSED -- a beam is never silently zeroed.

The slice read is visible in ``Baseline.j_NBI``: the beam entry's
j_parallel at core_sources time t_k is the constant 1e3*(k+1) A/m^2, and the
reference is the same entry on the full time base (the parallel -> toroidal
ratio depends only on the slice's core_profiles, so a correct read is
bit-identical to the reference).

Synthetic inputs only (the shipped D3D-like OMAS example, modified here).
"""
import copy
import json
import os
import warnings

import numpy as np
import pytest

_EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "examples", "D3D-like")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")


def _example():
    with open(_OMAS) as fh:
        return json.load(fh)


def _dd_with_tagged_nbi(with_times=True, drop_first=False):
    """The example dd whose beam entry carries 1e3*(k+1) at time t_k;
    *drop_first* makes it start one slice after the IDS time base."""
    dd = _example()
    t = list(dd["core_sources"]["time"])
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    n = len(nbi["profiles_1d"][0]["j_parallel"])
    for k, q in enumerate(nbi["profiles_1d"]):
        q["j_parallel"] = [1.0e3 * (k + 1)] * n
        if with_times:
            q["time"] = t[k]
        else:
            q.pop("time", None)
    if drop_first:
        nbi["profiles_1d"] = nbi["profiles_1d"][1:]
    return dd, t


def _read(tmp_path, dd, time, name, record=None):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    p = tmp_path / name
    with open(p, "w") as fh:
        json.dump(dd, fh)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(ImasSource(ids_path=str(p), time=time))
    if record is not None:
        record.extend(str(x.message) for x in w)
    return bl


def _reference(tmp_path, t, k):
    dd, _ = _dd_with_tagged_nbi()
    return np.asarray(_read(tmp_path, dd, t[k], f"ref{k}.json").j_NBI,
                      dtype=float)


def test_the_last_slice_reads_its_own_nbi_slice(tmp_path):
    """At the last slice the entry's OWN last slice is read, not its first
    (the list-index rule's fallback past the end of a late entry)."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    k = len(t) - 1
    got = np.asarray(_read(tmp_path, dd, t[k], "last.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
    assert np.max(np.abs(got)) > 0.0


def test_a_middle_slice_is_not_read_one_slice_late(tmp_path):
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    got = np.asarray(_read(tmp_path, dd, t[1], "mid.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, 1))


def test_a_slice_before_the_entry_starts_is_off_before_record(tmp_path):
    """At the first time the late entry has no record yet (its first own
    time is t_1, which carries current).  Owner decision 2026-10-06: the
    beam is OFF there -- zero, stamped off_before_record with its first own
    time on Baseline.source_time_match, and announced (it was REFUSED; the
    list-index rule read the NEXT time's)."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    msgs = []
    bl = _read(tmp_path, dd, t[0], "first.json", msgs)
    assert np.all(np.asarray(bl.j_NBI) == 0.0)
    e = [x for x in bl.source_time_match["entries"] if x["index"] == 2][0]
    assert e["status"] == "off_before_record"
    assert e["first_own_time"] == t[1]
    assert any("'nbi_synthetic' (index 2) has no record before its first "
               "own time" in m for m in msgs)



def test_a_late_beam_entry_idle_before_it_starts_is_off_not_refused(
        tmp_path):
    """Refinement of 2026-10-06: the late entry carrying NO current on its
    first own slice (the one bracketing t_0) is off at t_0, not missing --
    the beam reads zero there, nothing is refused; the next times read
    their own slices as before."""
    dd, t = _dd_with_tagged_nbi(drop_first=True)
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nbi["profiles_1d"][0]["j_parallel"] = [0.0] * len(
        nbi["profiles_1d"][0]["j_parallel"])
    got = np.asarray(_read(tmp_path, copy.deepcopy(dd), t[0],
                           "idle.json").j_NBI)
    assert np.all(got == 0.0)
    got = np.asarray(_read(tmp_path, copy.deepcopy(dd), t[-1],
                           "idle_last.json").j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, len(t) - 1))

def _shifted(dd, shift):
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    for q in nbi["profiles_1d"]:
        q["time"] = q["time"] + shift
    return dd


def test_a_two_microsecond_offset_reads_the_nearest_slice_not_zero(
        tmp_path):
    """The 2026-10-06 review's case: the entry's own times 2 us off the
    core_sources base.  Under the 1e-6 s match the beam was ZERO (one
    warning); under the half-step rule each slice reads its nearest own
    slice, bit-identical to the on-grid read, and nothing is warned."""
    for k in range(3):
        dd, t = _dd_with_tagged_nbi()
        msgs = []
        got = np.asarray(_read(tmp_path, _shifted(dd, 2e-6), t[k],
                               f"us{k}.json", msgs).j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
        assert np.max(np.abs(got)) > 0.0
        assert not any("NBI" in m for m in msgs)


def _single_time(dd, t, k):
    """The tagged example reduced to its slice *k* on every time base (no
    local step on either the core_profiles base or the beam entry)."""
    for blk in ("core_sources", "core_profiles", "equilibrium"):
        dd[blk]["time"] = [t[k]]
    dd["core_profiles"]["profiles_1d"] = [dd["core_profiles"]["profiles_1d"][k]]
    dd["equilibrium"]["time_slice"] = [dd["equilibrium"]["time_slice"][k]]
    for blk in ("vacuum_toroidal_field",):
        vt = dd["equilibrium"].get(blk)
        if vt and "b0" in vt:
            vt["b0"] = [vt["b0"][k]]
    for src in dd["core_sources"]["source"]:
        src["profiles_1d"] = [src["profiles_1d"][k]]
    return dd


def _nbi_match(bl):
    return [x for x in bl.source_time_match["entries"]
            if x["index"] == 2][0]


@pytest.mark.parametrize("shift", [2e-6, -2e-6])
def test_single_time_bases_match_within_the_ten_microsecond_floor(
        tmp_path, shift):
    """With no step on either grid (a single-time IDS and a single-time beam
    entry) the window is IMAS_SINGLE_TIME_WINDOW_S = 10 us (owner-approved
    2026-10-07; it was a few float ulp).  An entry 2 us AFTER the slice was
    off before its record and one 2 us BEFORE it refused; both are now
    MATCHED -- the beam read, bit-identical to the on-time read, its dt
    recorded, nothing announced."""
    from bouquet.io.imas import IMAS_SINGLE_TIME_WINDOW_S
    assert IMAS_SINGLE_TIME_WINDOW_S == 1e-5
    dd, t = _dd_with_tagged_nbi()
    k = 1
    dd = _single_time(dd, t, k)
    # on the time: read, dt = 0
    bl = _read(tmp_path, copy.deepcopy(dd), t[k], "one.json")
    np.testing.assert_array_equal(np.asarray(bl.j_NBI),
                                  _reference(tmp_path, t, k))
    assert _nbi_match(bl)["dt"] == 0.0
    msgs = []
    bl = _read(tmp_path, _shifted(dd, shift), t[k], "one_us.json", msgs)
    got = np.asarray(bl.j_NBI)
    np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
    assert np.max(np.abs(got)) > 0.0
    e = _nbi_match(bl)
    assert e["status"] == "matched"
    assert e["status"] != "off_before_record"
    assert e["matched_time"] == t[k] + shift
    assert e["dt"] == pytest.approx(shift, rel=1e-6)
    assert e["window_own"] == IMAS_SINGLE_TIME_WINDOW_S
    assert not any("off_before_record" in m or "NBI" in m for m in msgs)


def test_single_time_bases_beyond_the_floor_keep_the_record_rules(tmp_path):
    """Beyond the 10 us floor the before-record / after-record rules apply
    as before: an entry 20 us AFTER the slice (the slice precedes its only,
    current-carrying, own time) is off before its record -- zero, stamped
    and announced; one 20 us BEFORE it (the slice is past its only own
    time) is REFUSED -- never a zero beam."""
    dd, t = _dd_with_tagged_nbi()
    k = 1
    dd = _single_time(dd, t, k)
    with pytest.raises(ValueError, match="within half a time-step.*Refusing"):
        _read(tmp_path, _shifted(copy.deepcopy(dd), -2e-5), t[k],
              "twenty_us_early.json")
    msgs = []
    bl = _read(tmp_path, _shifted(dd, 2e-5), t[k], "twenty_us_late.json",
               msgs)
    assert np.all(np.asarray(bl.j_NBI) == 0.0)
    e = _nbi_match(bl)
    assert e["status"] == "off_before_record"
    assert e["first_own_time"] == t[k] + 2e-5
    assert any("off_before_record" in m for m in msgs)


def test_an_entry_exactly_on_a_slice_time_matches(tmp_path):
    dd, t = _dd_with_tagged_nbi()
    for k, tk in enumerate(t):
        got = np.asarray(_read(tmp_path, copy.deepcopy(dd), tk,
                               f"on{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))
        assert np.max(np.abs(got)) > 0.0


def test_a_coarser_entry_grid_is_refused_between_its_own_slices(tmp_path):
    """An entry on a coarser grid (its own times t_0 and t_2 only): at the
    middle slice t_1 its nearest own slice lies within half the ENTRY's
    local step but a full core_profiles step from t_1.  Owner decision
    2026-10-06: the matched own slice must also lie within half the local
    core_profiles step, so t_1 is REFUSED (it read the nearest own slice,
    0.1 s away, before); the end slices read their own."""
    dd, t = _dd_with_tagged_nbi()
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nbi["profiles_1d"] = [nbi["profiles_1d"][0], nbi["profiles_1d"][2]]
    with pytest.raises(ValueError, match=r"within half the local "
                                         r"core_profiles time-step of "
                                         r"t = 2\.2 s.*Refusing"):
        _read(tmp_path, copy.deepcopy(dd), t[1], "coarse.json")
    for k in (0, 2):
        got = np.asarray(_read(tmp_path, copy.deepcopy(dd), t[k],
                               f"coarse{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))


def test_past_the_end_by_more_than_half_a_step_is_refused(tmp_path):
    """The entry stops one slice early: at the last time its nearest own
    slice is a full step away -- refused (never read from another time)."""
    dd, t = _dd_with_tagged_nbi()
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    nbi["profiles_1d"] = nbi["profiles_1d"][:-1]
    with pytest.raises(ValueError, match="within half a time-step.*Refusing"):
        _read(tmp_path, dd, t[-1], "early_end.json")
    # the window edge: the entry's two own times (step 0.1 s) shifted so
    # the last slice time is 0.049 s (inside half a step) or 0.051 s
    # (outside) past its last own time
    for off, ok in ((0.049, True), (0.051, False)):
        dd, t = _dd_with_tagged_nbi()
        nbi = next(s for s in dd["core_sources"]["source"]
                   if s["identifier"]["index"] == 2)
        nbi["profiles_1d"] = nbi["profiles_1d"][:-1]
        last = t[-1] - off
        nbi["profiles_1d"][0]["time"] = last - 0.1
        nbi["profiles_1d"][1]["time"] = last
        if ok:
            got = np.asarray(_read(tmp_path, dd, t[-1], "in.json").j_NBI)
            assert np.max(np.abs(got)) > 0.0
        else:
            with pytest.raises(ValueError, match="within half a time-step"):
                _read(tmp_path, dd, t[-1], "out.json")


def test_an_nbi_entry_that_cannot_be_aligned_is_refused(tmp_path):
    dd, t = _dd_with_tagged_nbi(with_times=False, drop_first=True)
    with pytest.raises(ValueError, match="cannot be aligned"):
        _read(tmp_path, dd, t[-1], "notime.json")


def test_an_nbi_entry_on_the_full_time_base_without_times_reads_by_index(
        tmp_path):
    """The common case -- every slice present, no per-slice time -- is read
    by index, exactly as before (= the time-tagged reference)."""
    dd, t = _dd_with_tagged_nbi(with_times=False)
    for k, tk in enumerate(t):
        got = np.asarray(_read(tmp_path, dd, tk, f"full{k}.json").j_NBI)
        np.testing.assert_array_equal(got, _reference(tmp_path, t, k))


def test_the_shipped_example_reads_the_same_nbi_as_the_index_rule(tmp_path):
    """Nothing moves for the shipped example (its beam entry is on the full
    time base, without per-slice times): at every slice the reader's j_NBI
    is the entry's slice k converted, i.e. what the list-index rule read."""
    dd = _example()
    t = list(dd["core_sources"]["time"])
    nbi = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 2)
    assert len(nbi["profiles_1d"]) == len(t)
    assert all(q.get("time") is None for q in nbi["profiles_1d"])
    for k, tk in enumerate(t):
        msgs = []
        bl = _read(tmp_path, dd, tk, f"ex{k}.json", msgs)
        # the same entry, slice k pinned explicitly by time
        dd2 = copy.deepcopy(dd)
        nb2 = next(s for s in dd2["core_sources"]["source"]
                   if s["identifier"]["index"] == 2)
        for j, q in enumerate(nb2["profiles_1d"]):
            q["time"] = t[j]
        bl2 = _read(tmp_path, dd2, tk, f"ex_t{k}.json")
        np.testing.assert_array_equal(np.asarray(bl.j_NBI),
                                      np.asarray(bl2.j_NBI))
        assert not any("NBI entry" in m for m in msgs)
