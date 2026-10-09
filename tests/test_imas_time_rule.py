"""The IMAS core_sources time rule (owner decision 2026-10-06).

Both readers -- the legacy ``read_imas_baseline`` and the engine's
``IdsAdapter`` -- share one rule (``bouquet.io.imas.core_sources_slice``,
``_source_slice_at``):

1. the core_sources slice is the one NEAREST the core_profiles slice actually
   read and must lie within HALF the local core_profiles time-step of it,
   else the read is REFUSED naming both times (a single-time core_sources was
   read at any requested time);
2. an entry's matched own slice must lie within half ITS own local step AND
   within half the local core_profiles step of the core_profiles slice time
   (a coarse own grid or a constant offset passed on its own step alone);
3. the match is recorded: both slice times, dt, the windows and their basis,
   each entry's matched own time, dt, bracketing own times and status;
4. a driven entry whose own record begins AFTER the slice time is OFF there
   (zero), stamped ``off_before_record`` with its first own time and
   announced (print + warning) once per source file and entry; past its last
   own time it is still refused when that slice carries current; idle on its
   bracketing own slices it is off, as before.  Never interpolated.

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
_MESH = os.path.join(_EX, "DIIID_mesh.h5")

pytestmark = pytest.mark.skipif(not os.path.isfile(_OMAS),
                                reason="synthetic IMAS example absent")


def _example():
    with open(_OMAS) as fh:
        return json.load(fh)


def _nbi(dd):
    return next(s for s in dd["core_sources"]["source"]
                if s["identifier"]["index"] == 2)


def _tagged(dd):
    """The beam carries 1e3 (k+1) at core_sources time t_k, time-tagged."""
    t = [float(x) for x in dd["core_sources"]["time"]]
    nb = _nbi(dd)
    n = len(nb["profiles_1d"][0]["j_parallel"])
    for k, q in enumerate(nb["profiles_1d"]):
        q["j_parallel"] = [1.0e3 * (k + 1)] * n
        q["time"] = t[k]
    return t


def _write(tmp_path, dd, name):
    p = tmp_path / name
    with open(p, "w") as fh:
        json.dump(dd, fh)
    return str(p)


def _read(path, time, msgs=None):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(ImasSource(ids_path=path, time=time))
    if msgs is not None:
        msgs.extend(str(x.message) for x in w)
    return bl


def _adapter(path, time):
    from bouquet.adapters import IdsAdapter
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=path, time=time),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
        return IdsAdapter(cfg.source, cfg, bl).read()


# ---------------------------------------------------------------------------
#  (1) the core_sources slice itself is windowed
# ---------------------------------------------------------------------------
def test_a_single_time_core_sources_at_another_time_is_refused(tmp_path):
    """core_sources reduced to its first (2.1 s) slice, read at the last
    core_profiles slice (2.3043 s): refused, naming both times.  Before,
    the single-time core_sources was index 0 whatever the time -- the
    2.1 s beam was read at 2.3043 s with no trace."""
    dd = _example()
    t = [float(x) for x in dd["core_profiles"]["time"]]
    cs = dd["core_sources"]
    cs["time"] = [cs["time"][0]]
    for s in cs["source"]:
        s["profiles_1d"] = [s["profiles_1d"][0]]
    p = _write(tmp_path, dd, "cs_one.json")
    with pytest.raises(ValueError, match=r"IMAS reader: the core_sources "
                       r"slice nearest the core_profiles slice read "
                       r"\(t = 2\.3043 s\) is at t = 2\.1 s: \|dt\| = 0\.204 "
                       r"s > 0\.0521 s, half the local core_profiles "
                       r"time-step.*Refusing"):
        _read(p, t[-1])
    # at its own time it is read, dt = 0
    bl = _read(p, t[0])
    assert bl.source_time_match["core_sources"]["dt"] == 0.0
    assert np.max(np.abs(np.asarray(bl.j_NBI))) > 0.0


def test_the_slice_helper_refuses_for_both_readers():
    from bouquet.adapters import EngineInputRefused  # noqa: F401
    from bouquet.io.imas import core_sources_slice
    src = dict(time=[2.1])
    cpt = [2.1, 2.2, 2.3043]
    isrc, t_src, rec = core_sources_slice(src, cpt, 0)
    assert (isrc, t_src, rec["dt"]) == (0, 2.1, 0.0)
    assert rec["window_basis"] == "half the local core_profiles time-step"
    for who in ("IMAS reader", "IDS adapter"):
        with pytest.raises(ValueError, match=f"^{who}: the core_sources "
                                             "slice nearest"):
            core_sources_slice(src, cpt, 2, who=who)
    # within half a step: accepted, dt recorded (signed, core_sources minus
    # core_profiles)
    isrc, t_src, rec = core_sources_slice(dict(time=[2.16, 2.31]), cpt, 1)
    assert isrc == 0 and rec["dt"] == pytest.approx(-0.04)
    assert rec["window"] == pytest.approx(0.05)
    # no core_sources time base: by index, as before
    isrc, t_src, rec = core_sources_slice(dict(), cpt, 2)
    assert isrc == 2 and t_src is None and rec["rule"].startswith("by index")


# ---------------------------------------------------------------------------
#  (2) the entry window is also bounded by the core_profiles step
# ---------------------------------------------------------------------------
def test_two_single_time_bases_use_the_ten_microsecond_floor():
    """Owner-approved 2026-10-07: with no local step on either base the
    window is IMAS_SINGLE_TIME_WINDOW_S = 10 us (it was a few float ulp),
    for the core_sources slice (core_sources_slice) and for an entry
    (_entry_time_window) alike -- shared by both readers.  With a step on
    either base the half-step window is unchanged."""
    from bouquet.io.imas import (IMAS_SINGLE_TIME_WINDOW_S, _entry_time_window,
                                 core_sources_slice)
    assert IMAS_SINGLE_TIME_WINDOW_S == 1e-5
    # the core_sources slice 2 us either side of a single-time core_profiles
    for d in (2e-6, -2e-6):
        isrc, t_src, rec = core_sources_slice(dict(time=[2.2 + d]), [2.2], 0)
        assert isrc == 0 and rec["dt"] == pytest.approx(d, rel=1e-6)
        assert rec["window"] == IMAS_SINGLE_TIME_WINDOW_S
        assert "IMAS_SINGLE_TIME_WINDOW_S" in rec["window_basis"]
        assert "IMAS_SINGLE_TIME_WINDOW_S" in rec["rule"]
    # 20 us: refused, as before
    with pytest.raises(ValueError, match="the single-time floor"):
        core_sources_slice(dict(time=[2.2 + 2e-5]), [2.2], 0)
    # an entry: single-time own grid on a single-time base (or none)
    for base in ([2.2], None):
        k, dt, half = _entry_time_window([2.2 + 2e-6], 2.2, base)
        assert (k, half) == (0, IMAS_SINGLE_TIME_WINDOW_S)
        assert dt == pytest.approx(2e-6, rel=1e-6) and dt <= half
        k, dt, half = _entry_time_window([2.2 + 2e-5], 2.2, base)
        assert dt > half
    # a step on either base: half that step, the floor plays no part
    assert _entry_time_window([2.2], 2.2, [2.1, 2.2, 2.3])[2] == \
        pytest.approx(0.05)
    assert _entry_time_window([2.1, 2.2], 2.2, [2.2])[2] == \
        pytest.approx(0.05)
    assert core_sources_slice(dict(time=[2.2]), [2.1, 2.2], 1)[2][
        "window"] == pytest.approx(0.05)


def test_a_coarse_entry_grid_offset_is_refused_on_both_paths():
    """An entry on a grid ten times coarser than core_profiles, read
    between its own samples: within its own half-step, a full core_profiles
    step away -- refused (it was read).  On the core_profiles grid it is
    read; an idle bracketing pair is still off."""
    from bouquet.adapters import EngineInputRefused, _ids_driven_currents
    n = 3
    base = [round(1.0 + 0.1 * k, 10) for k in range(31)]     # 1.0 .. 4.0
    srcs = dict(time=base, source=[dict(
        identifier=dict(name="nbi", index=2),
        profiles_1d=[dict(time=float(tk), j_parallel=[1.0e3 * (j + 1)] * n)
                     for j, tk in enumerate((1.0, 2.0, 3.0, 4.0))])])
    k = base.index(1.4)                      # 0.4 s from own 1.0: own half 0.5
    with pytest.raises(EngineInputRefused, match="within half the local "
                                                 "core_profiles time-step"):
        _ids_driven_currents(srcs, k, n, 1.0, base)
    matches = []
    parts, used, _ = _ids_driven_currents(srcs, base.index(2.0), n, 1.0,
                                          base, matches=matches)
    np.testing.assert_array_equal(parts["nbi"], 2.0e3)
    assert used[0]["dt"] == 0.0 and used[0]["matched_time"] == 2.0
    assert matches[0]["status"] == "matched"
    assert matches[0]["window_core_profiles"] == pytest.approx(0.05)
    assert matches[0]["window_own"] == pytest.approx(0.5)


def test_a_constant_offset_inside_half_the_core_profiles_step_is_read_with_dt(
        tmp_path):
    """Own times offset by +0.02 s on the IDS grid (inside half the 0.1 s
    core_profiles step): read, and dt = +0.02 s recorded on the baseline and
    in the adapter's provenance (dt was never recorded).  (An offset on a
    grid AS FINE as core_profiles always lands within half a step of some
    own sample -- the rule cannot tell which sample was meant; the recorded
    dt is what shows it.)"""
    dd = _example()
    t = _tagged(dd)
    for q in _nbi(dd)["profiles_1d"]:
        q["time"] = q["time"] + 0.02
    p = _write(tmp_path, dd, "off02.json")
    bl = _read(p, t[1])
    e = bl.source_time_match["entries"][0]
    assert e["status"] == "matched" and e["dt"] == pytest.approx(0.02)
    assert e["matched_time"] == pytest.approx(t[1] + 0.02)
    assert e["bracketing_own_times"] == [pytest.approx(t[0] + 0.02),
                                         pytest.approx(t[1] + 0.02)]
    c = _adapter(p, t[-1])
    st = c.provenance["source_time_match"]
    assert st["core_sources"]["dt"] == 0.0
    nb = [x for x in st["entries"] if x["index"] == 2][0]
    assert nb["status"] == "matched" and nb["dt"] == pytest.approx(0.02)
    used = [x for x in c.provenance["driven_sources"] if x["index"] == 2][0]
    assert used["dt"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
#  (4) a beam starting mid-shot is OFF before its record
# ---------------------------------------------------------------------------
def test_a_beam_starting_mid_shot_is_off_before_its_record(tmp_path,
                                                            capsys):
    """The beam's own record starts at the second slice (and carries
    current there).  At the first slice: zero, stamped off_before_record
    with its first own time, announced ONCE (print + warning) however often
    the slice is read; at the later slices read at |dt| = 0.  Same on the
    engine adapter (off_sources)."""
    dd = _example()
    t = _tagged(dd)
    nb = _nbi(dd)
    nb["profiles_1d"] = nb["profiles_1d"][1:]
    p = _write(tmp_path, dd, "midshot.json")
    msgs = []
    bl = _read(p, t[0], msgs)
    assert np.all(np.asarray(bl.j_NBI) == 0.0)
    e = bl.source_time_match["entries"][0]
    assert e["status"] == "off_before_record" and e["first_own_time"] == t[1]
    ann = [m for m in msgs if "has no record before its first own time" in m]
    assert len(ann) == 1 and "'nbi_synthetic' (index 2)" in ann[0]
    assert "off_before_record" in capsys.readouterr().out
    msgs2 = []
    bl2 = _read(p, t[0], msgs2)                    # again: not re-announced
    assert np.all(np.asarray(bl2.j_NBI) == 0.0)
    assert not any("no record before" in m for m in msgs2)
    assert "no record before" not in capsys.readouterr().out
    for k in (1, 2):
        blk = _read(p, t[k])
        ek = blk.source_time_match["entries"][0]
        assert ek["status"] == "matched" and ek["dt"] == 0.0
        assert np.max(np.abs(np.asarray(blk.j_NBI))) > 0.0
    c = _adapter(p, t[0])
    assert np.all(c.jB_fix_parts["nbi"] == 0.0)
    assert c.provenance["off_sources"] == [dict(
        name="nbi_synthetic", index=2, reason="off_before_record",
        first_own_time=t[1])]


def test_past_the_entrys_last_time_with_current_is_still_refused(tmp_path):
    dd = _example()
    t = _tagged(dd)
    nb = _nbi(dd)
    nb["profiles_1d"] = nb["profiles_1d"][:-1]
    p = _write(tmp_path, dd, "ends_early.json")
    with pytest.raises(ValueError, match=r"'nbi_synthetic' \(index 2\) "
                                         r"carries a non-zero j_parallel.*"
                                         r"Refusing"):
        _read(p, t[-1])
    # idle on that last own slice: off, as before (stamped off_idle)
    nb["profiles_1d"][-1]["j_parallel"] = [0.0] * len(
        nb["profiles_1d"][-1]["j_parallel"])
    p = _write(tmp_path, copy.deepcopy(dd), "ends_idle.json")
    bl = _read(p, t[-1])
    assert np.all(np.asarray(bl.j_NBI) == 0.0)
    assert bl.source_time_match["entries"][0]["status"] == "off_idle"
