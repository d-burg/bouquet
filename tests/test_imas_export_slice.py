"""The single-slice IMAS export (``_slice_in_time`` / ``write_imas_draw``).

Two properties (PR #71 review):

1. What is cut is keyed on the IMAS structure -- the IDS ``time``, time-tagged
   arrays of structures, signals, the homogeneous-time arrays -- never on a
   list being as long as a time base: three sources on a three-time base stay
   three sources, a 4-point coil outline on a 4-time pf_active stays 4 points.
2. The export re-reads as the archive it came from.  The cut is taken at the
   core_profiles slice the reader reads (not the caller's time), core_sources
   with the reader's own rule (each entry keeps its own slices bracketing the
   time, and its first and last), and the windows of that read are written
   under ``IMAS_EXPORT_TIME_WINDOW_KEY`` so the re-read's half-step windows
   do not collapse to the 10 us single-time floor (which turned an
   offset-matched beam into "off" and an offset core_sources base into a
   refusal).

Synthetic inputs only (the shipped D3D-like OMAS example, modified here).
"""
import copy
import json
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_EX = os.path.join(os.path.dirname(_HERE), "examples", "D3D-like")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")
_GOLD = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")

T = 2.2


def _src(index, name, times, amps, n=4):
    return dict(identifier=dict(index=index, name=name),
                profiles_1d=[dict(time=float(t), j_parallel=[float(a)] * n)
                             for t, a in zip(times, amps)])


# ---------------------------------------------------------------------------
#  (1) structure-keyed cut
# ---------------------------------------------------------------------------
def test_three_sources_on_a_three_time_base_stay_three_sources():
    from bouquet.io.imas import _get_export_window, _slice_in_time
    t = [0.5, 1.0, 1.5]
    dd = {"core_profiles": {"time": t, "profiles_1d": [
              {"time": x, "grid": {"psi": [0.0, 1.0, 2.0]}} for x in t]},
          "core_sources": {"time": t, "source": [
              _src(2, "nbi", t, [1.0, 2.0, 3.0]),
              _src(3, "ec", t, [4.0, 5.0, 6.0]),
              _src(13, "bootstrap", t, [7.0, 8.0, 9.0])]}}
    assert _slice_in_time(dd, 1.0) == 1.0
    cs = dd["core_sources"]
    assert cs["time"] == [1.0]
    assert [s["identifier"]["name"] for s in cs["source"]] == \
        ["nbi", "ec", "bootstrap"]
    for s, amps in zip(cs["source"], ([1, 2, 3], [4, 5, 6], [7, 8, 9])):
        # the slice read (bracketing = itself at dt 0) + first + last
        assert [q["time"] for q in s["profiles_1d"]] == t
        assert _get_export_window(s)["window_own"] == pytest.approx(0.25)
    # entries on a finer own grid keep only the slices the rule consults
    dd["core_sources"]["source"][0] = _src(2, "nbi", [0.5, 0.75, 1.0, 1.25,
                                                      1.5], [1, 2, 3, 4, 5])
    _slice_in_time(dd, 1.0)
    assert [q["time"] for q in dd["core_sources"]["source"][0][
        "profiles_1d"]] == [0.5, 1.0, 1.5]
    # the radial profile of the kept core_profiles slice is whole
    assert dd["core_profiles"]["profiles_1d"] == [
        {"time": 1.0, "grid": {"psi": [0.0, 1.0, 2.0]}}]


def test_a_coil_outline_on_a_four_time_pf_active_is_kept_whole():
    from bouquet.io.imas import _slice_in_time
    outline = {"r": [1.0, 1.1, 1.1, 1.0], "z": [0.0, 0.0, 0.1, 0.1]}
    dd = {"core_profiles": {"time": [0.0, 1.0, 2.0, 3.0]},
          "pf_active": {"time": [0.0, 1.0, 2.0, 3.0],
                        "ids_properties": {"homogeneous_time": 1},
                        "coil": [
                            {"element": [{"geometry": {"outline":
                                                       copy.deepcopy(outline)}}],
                             "current": {"data": [0.0, 10.0, 20.0, 30.0],
                                         "time": []}},
                            {"element": [{"geometry": {"outline":
                                                       copy.deepcopy(outline)}}],
                             "current": {"data": [5.0, 6.0, 7.0, 8.0],
                                         "time": [0.0, 0.9, 2.2, 3.1]}},
                            {"element": [], "current": {"data": [1.0] * 4}},
                            {"element": [], "current": {"data": [1.0] * 4}}]},
          "wall": {"time": [0.0, 1.0, 2.0, 3.0], "description_2d": [
              {"limiter": {"unit": [{"outline": copy.deepcopy(outline)}]}}]}}
    _slice_in_time(dd, 2.1)                     # core_profiles slice: 2.0
    pf = dd["pf_active"]
    assert pf["time"] == [2.0]
    assert len(pf["coil"]) == 4                  # the coil list is not cut
    for c in pf["coil"][:2]:
        assert c["element"][0]["geometry"]["outline"] == outline
    assert pf["coil"][0]["current"] == {"data": [20.0], "time": []}
    assert pf["coil"][1]["current"] == {"data": [7.0], "time": [2.2]}
    assert dd["wall"]["description_2d"][0]["limiter"]["unit"][0][
        "outline"] == outline


def test_the_cut_is_at_the_core_profiles_slice_read_not_the_callers_time():
    """time = 1.1 reads the core_profiles slice at 1.0; every IDS is cut
    there (a beam with own times [0.75, 0.95, 1.15] keeps the 0.95 slice the
    reader matched, not the 1.15 one nearest 1.1)."""
    from bouquet.io.imas import _slice_in_time
    t = [0.5, 1.0, 1.5]
    dd = {"core_profiles": {"time": t, "profiles_1d": [{"time": x} for x in t]},
          "equilibrium": {"time": [0.5, 1.0, 1.08, 1.5], "time_slice": [
              {"time": x} for x in (0.5, 1.0, 1.08, 1.5)]},
          "core_sources": {"time": t, "source": [
              _src(2, "nbi", [0.75, 0.95, 1.15], [1.0, 2.0, 3.0])]}}
    assert _slice_in_time(dd, 1.1) == 1.0
    assert dd["equilibrium"]["time"] == [1.0]
    assert dd["equilibrium"]["time_slice"] == [{"time": 1.0}]
    nb = dd["core_sources"]["source"][0]["profiles_1d"]
    # bracketing 1.0: 0.95 and 1.15; first 0.75
    assert [q["time"] for q in nb] == [0.75, 0.95, 1.15]


# ---------------------------------------------------------------------------
#  (2) the export re-reads as the archive it came from
# ---------------------------------------------------------------------------
def _multi_slice_template():
    """The example with core_sources offset +30 ms from core_profiles (inside
    the 50 ms half-step), a beam on its own grid matched at 2.24 s (an own
    time != the core_profiles 2.2 s), an EC entry starting after the slice
    (off_before_record), and a fusion entry idle on its bracketing slices
    (off_idle)."""
    with open(_OMAS) as fh:
        dd = json.load(fh)
    n = len(dd["core_profiles"]["profiles_1d"][0]["grid"]["psi"])
    x = np.linspace(0.0, 1.0, n)
    cs = dd["core_sources"]
    cs["time"] = [2.13, 2.23, 2.3343]
    nb = next(s for s in cs["source"] if s["identifier"]["index"] == 2)
    for q, tq in zip(nb["profiles_1d"], (2.12, 2.24, 2.35)):
        q["time"] = tq

    def ent(index, name, times, amps):
        sh = np.exp(-((x - 0.4) / 0.1) ** 2)
        return dict(identifier=dict(index=index, name=name), profiles_1d=[
            dict(time=t, j_parallel=(a * sh).tolist())
            for t, a in zip(times, amps)])
    cs["source"] += [ent(3, "ec", [2.30, 2.40], [5e3, 6e3]),
                     ent(6, "fusion", [2.0, 2.4, 2.6], [0.0, 0.0, 4e3]),
                     ent(5, "ic", [2.10, 2.21, 2.33], [1e3, 2e3, 3e3])]
    return dd


def _read(path, **src):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=path, time=T, **src),
                                  anchor_jtor_to_equilibrium=False)


def _assert_same_match(a, b):
    for k in ("core_sources", "entries", "off_sources", "driven_sources",
              "ignored_sources", "sawteeth_hold"):
        assert a[k] == b[k], k


@pytest.mark.skipif(not os.path.isfile(_OMAS),
                    reason="synthetic IMAS example absent")
def test_a_single_slice_cut_re_reads_identically(tmp_path):
    from bouquet.io.imas import (_drop_export_window, _get_export_window,
                                 _slice_in_time)
    dd = _multi_slice_template()
    p0 = tmp_path / "multi.json"
    p0.write_text(json.dumps(dd))
    ref = _read(str(p0))
    st = ref.source_time_match
    by = {e["index"]: e for e in st["entries"]}
    assert by[2]["status"] == "matched" and by[2]["matched_time"] == 2.24
    assert by[3]["status"] == "off_before_record"
    assert by[6]["status"] == "off_idle"
    assert by[5]["status"] == "matched" and by[5]["matched_time"] == 2.21
    assert st["core_sources"]["dt"] == pytest.approx(0.03)
    cut = copy.deepcopy(dd)
    _slice_in_time(cut, T)
    assert cut["core_profiles"]["time"] == [T]
    assert cut["core_sources"]["time"] == [2.23]
    assert _get_export_window(cut["core_sources"]) is not None
    p1 = tmp_path / "one.json"
    p1.write_text(json.dumps(cut))
    bl = _read(str(p1))
    _assert_same_match(bl.source_time_match, st)
    for k in ("j_NBI", "j_RF", "j_other", "j_sawteeth", "j_BS",
              "j_inductive", "j_phi"):
        np.testing.assert_array_equal(getattr(bl, k), getattr(ref, k), k)
    assert np.max(np.abs(ref.j_NBI)) > 0 and np.max(np.abs(ref.j_RF)) > 0
    # without the recorded windows the same file re-reads differently: the
    # core_sources base 30 ms off the single core_profiles time is refused
    bare = copy.deepcopy(cut)
    _drop_export_window(bare["core_sources"])
    for s in bare["core_sources"]["source"]:
        _drop_export_window(s)
    p2 = tmp_path / "bare.json"
    p2.write_text(json.dumps(bare))
    with pytest.raises(ValueError, match="the single-time floor"):
        _read(str(p2))


@pytest.mark.skipif(not os.path.isfile(_OMAS),
                    reason="synthetic IMAS example absent")
def test_recorded_windows_are_ignored_away_from_the_slice_they_describe(
        tmp_path):
    """A block whose times are not the slice read plays no part."""
    from bouquet.io.imas import (_get_export_window, _set_export_window,
                                 _slice_in_time)
    cut = _multi_slice_template()
    _slice_in_time(cut, T)
    _w = dict(_get_export_window(cut["core_sources"]))
    _w["core_profiles_time"] = T + 1e-3
    _set_export_window(cut["core_sources"], _w)
    p = tmp_path / "moved.json"
    p.write_text(json.dumps(cut))
    with pytest.raises(ValueError, match="the single-time floor"):
        _read(str(p))


@pytest.mark.skipif(not (os.path.isfile(_OMAS) and os.path.isfile(_GOLD)),
                    reason="synthetic IMAS example or golden archive absent")
def test_write_imas_draw_re_reads_as_the_archive_on_both_readers(tmp_path):
    """The full export (write_imas_draw) of a draw into the multi-slice
    template: the reader's match record is the template's, and the engine
    adapter's driven parts (<j.B>, unaffected by the draw's geometry) are
    identical to the template's."""
    from bouquet.adapters import IdsAdapter
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    from bouquet.io.imas import write_imas_draw
    import test_imas_export_roundtrip as rt
    dd = _multi_slice_template()
    tmpl = tmp_path / "multi.json"
    tmpl.write_text(json.dumps(dd))
    # a draw archive on the example's grid (the round-trip module's recipe)
    import h5py
    with h5py.File(rt._GOLD, "r") as hf:
        eqb = bytes(hf[rt._GP]["eqdsk"][()])
        fsa = {k: np.asarray(hf[rt._GP]["eq_fsa"][k][()], dtype=float)
               for k in hf[rt._GP]["eq_fsa"]}

    def contract(path, anchor):
        cfg = BouquetConfig(
            source=ImasSource(ids_path=str(path), time=T),
            solver=SolverConfig(mesh_path=_MESH), output_header="t",
            generation=GenerationConfig(reconstruction_engine="unified",
                                        anchor_jtor_to_equilibrium=anchor))
        from bouquet.baseline import resolve_baseline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = resolve_baseline(cfg, None)
            return bl, IdsAdapter(cfg.source, cfg, bl).read()
    bl0, c0 = contract(tmpl, True)
    psi = np.asarray(c0.psi_N, dtype=float)
    s = dict(bl=bl0, psi=psi, eqb=eqb, fsa=fsa)
    j_phi = np.asarray(bl0.j_phi, dtype=float)
    arc = str(tmp_path / "draw.h5")
    rt._write_archive(arc, s, j_phi, np.asarray(bl0.j_inductive, float),
                      np.asarray(bl0.j_BS, float))
    out = tmp_path / "draw0.json"
    write_imas_draw(arc, 0, str(tmpl), str(out), scan_key=0, time=T)
    bl1, c1 = contract(out, False)
    _assert_same_match(bl1.source_time_match, bl0.source_time_match)
    assert c1.provenance["source_time_match"] == \
        c0.provenance["source_time_match"]
    assert c1.provenance["off_sources"] == c0.provenance["off_sources"]
    for k in ("nbi", "rf", "other"):
        np.testing.assert_array_equal(c1.jB_fix_parts[k], c0.jB_fix_parts[k])


def test_the_window_lives_in_code_parameters_and_keeps_the_templates():
    """Owner recommendation (integration): the export window is recorded in
    the schema-legal ``code.parameters`` string (JSON), never as a
    non-schema key; a template's JSON parameters keep their keys, its other
    text is kept verbatim; an export written with the old direct key still
    re-reads."""
    import json
    from bouquet.io.imas import (IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY,
                                 IMAS_EXPORT_TIME_WINDOW_KEY,
                                 _get_export_window, _set_export_window,
                                 _slice_in_time)
    t = [0.5, 1.0, 1.5]
    dd = {"core_profiles": {"time": t, "profiles_1d": [
              {"time": x, "grid": {"psi": [0.0, 1.0, 2.0]}} for x in t]},
          "core_sources": {"time": t,
                           "code": {"name": "m", "parameters": "<p>x</p>"},
                           "source": [_src(2, "nbi", t, [1.0, 2.0, 3.0])]}}
    dd["core_sources"]["source"][0]["code"] = {
        "parameters": json.dumps({"model": "a"})}
    _slice_in_time(dd, 1.0)
    cs = dd["core_sources"]
    assert IMAS_EXPORT_TIME_WINDOW_KEY not in cs
    top = json.loads(cs["code"]["parameters"])
    assert top[IMAS_EXPORT_TEMPLATE_PARAMETERS_KEY] == "<p>x</p>"
    assert top[IMAS_EXPORT_TIME_WINDOW_KEY]["core_sources_time"] == 1.0
    assert cs["code"]["name"] == "m"
    s0 = cs["source"][0]
    ent = json.loads(s0["code"]["parameters"])
    assert ent["model"] == "a" and IMAS_EXPORT_TIME_WINDOW_KEY in ent
    assert _get_export_window(s0) == ent[IMAS_EXPORT_TIME_WINDOW_KEY]
    # an export from before the move (a direct key) is still honoured
    old = {IMAS_EXPORT_TIME_WINDOW_KEY: {"window": 0.1}}
    assert _get_export_window(old) == {"window": 0.1}
    _set_export_window(old, {"window": 0.2})        # and moved when rewritten
    assert IMAS_EXPORT_TIME_WINDOW_KEY not in old
    assert _get_export_window(old) == {"window": 0.2}
