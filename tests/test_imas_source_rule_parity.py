"""One source-time rule for every driven core_sources channel, on both readers.

The legacy reader (``read_imas_baseline``) and the unified engine's IDS adapter
(``IdsAdapter.read``, which runs the reader first) read every driven entry --
beams, EC/LH/IC, fusion, runaways, sawteeth, unknown indices -- through
``adapters._ids_driven_currents`` with the same arguments: the core_profiles
slice time and half-step window, the off list, the match records and the
announcement key.  So a refusal, an ``off_idle`` / ``off_before_record`` stamp
and the announcement are the same on both.  Before, the reader's
RF / other / sawteeth read passed none of them: it refused an EC entry the
engine accepted (and the default engine runs the reader first), stamped no
off entry, held a sawteeth current matched only on its own coarse step and
zeroed one past its record.

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

T = 2.2                              # the middle core_profiles slice


def _example():
    with open(_OMAS) as fh:
        return json.load(fh)


def _n(dd):
    return len(dd["core_profiles"]["profiles_1d"][0]["grid"]["psi"])


def _entry(dd, index, name, times, amps):
    n = _n(dd)
    x = np.linspace(0.0, 1.0, n)
    shape = np.exp(-((x - 0.4) / 0.1) ** 2)
    return dict(identifier=dict(index=index, name=name),
                profiles_1d=[dict(time=float(t), j_parallel=(a * shape).tolist())
                             for t, a in zip(times, amps)])


def _with(dd, *entries, cs_times=None):
    dd = copy.deepcopy(dd)
    cs = dd["core_sources"]
    if cs_times is not None:
        cs["time"] = list(cs_times)
    cs["source"].extend(entries)
    return dd


def _write(tmp_path, dd, name):
    p = tmp_path / name
    p.write_text(json.dumps(dd))
    return str(p)


def _read(path, time=T, **src):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=path, time=time, **src))


def _engine_read(path, bl, time=T):
    """The engine's own IDS read of *path* (``IdsAdapter.read``) on the
    reader's baseline *bl* (same grid)."""
    from bouquet.adapters import IdsAdapter
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=path, time=time),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IdsAdapter(cfg.source, cfg, bl).read()


def _same_records(bl, c):
    """The reader's records are the engine's."""
    pv = c.provenance
    st = bl.source_time_match
    assert st["core_sources"] == pv["source_time_match"]["core_sources"]
    assert st["entries"] == pv["source_time_match"]["entries"]
    assert st["off_sources"] == pv["off_sources"]
    assert st["driven_sources"] == pv["driven_sources"]
    assert st["ignored_sources"] == pv["ignored_sources"]


def test_an_ec_entry_matched_within_the_core_profiles_window_is_read_on_both(
        tmp_path):
    """core_sources offset +45 ms from core_profiles (inside the 50 ms
    half-step); an EC entry on its own 0.2 s grid whose nearest own time
    (2.17 s) is 75 ms from the core_sources slice but 30 ms from the
    core_profiles slice actually read.  The engine matched it; the reader
    judged it against the core_sources time and REFUSED -- and the default
    engine runs the reader first, so it refused a dd it accepts."""
    dd = _example()
    cs_t = [2.145, 2.245, 2.3493]
    ec = _entry(dd, 3, "ec", [2.17, 2.37], [5.0e3, 7.0e3])
    p = _write(tmp_path, _with(dd, ec, cs_times=cs_t), "ec_offset.json")
    bl = _read(p)
    c = _engine_read(p, bl)
    _same_records(bl, c)
    e = [x for x in bl.source_time_match["entries"] if x["index"] == 3][0]
    assert e["status"] == "matched" and e["matched_time"] == 2.17
    assert e["dt"] == pytest.approx(2.17 - T)
    assert np.max(np.abs(bl.j_RF)) > 0.0
    assert np.max(np.abs(c.jB_fix_parts["rf"])) > 0.0
    # it is the 2.17 s slice that is read: changing the other one changes
    # nothing on either reader
    ec2 = _entry(dd, 3, "ec", [2.17, 2.37], [5.0e3, 0.0])
    p2 = _write(tmp_path, _with(dd, ec2, cs_times=cs_t), "ec_offset2.json")
    bl2 = _read(p2)
    np.testing.assert_array_equal(bl.j_RF, bl2.j_RF)
    np.testing.assert_array_equal(c.jB_fix_parts["rf"],
                                  _engine_read(p2, bl2).jB_fix_parts["rf"])


def test_an_ec_entry_starting_after_the_slice_is_off_before_record_on_both(
        tmp_path, capsys):
    dd = _example()
    ec = _entry(dd, 3, "ec", [2.27, 2.37], [5.0e3, 7.0e3])
    p = _write(tmp_path, _with(dd, ec), "ec_late.json")
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(ImasSource(ids_path=p, time=T))
        c = _engine_read(p, bl)
    assert np.all(np.asarray(bl.j_RF) == 0.0)
    assert np.all(c.jB_fix_parts["rf"] == 0.0)
    off = dict(name="ec", index=3, reason="off_before_record",
               first_own_time=2.27)
    assert bl.source_time_match["off_sources"] == [off]
    assert c.provenance["off_sources"] == [off]
    e = [x for x in bl.source_time_match["entries"] if x["index"] == 3][0]
    assert e["status"] == "off_before_record" and e["first_own_time"] == 2.27
    _same_records(bl, c)
    # announced ONCE across the reader and the engine (one key: the file)
    ann = [str(x.message) for x in w
           if "has no record before its first own time" in str(x.message)]
    assert len(ann) == 1 and "'ec' (index 3)" in ann[0]
    assert capsys.readouterr().out.count("off_before_record") == 1


def test_an_entry_idle_on_its_bracketing_slices_is_off_and_stamped_on_both(
        tmp_path):
    """A fusion entry (index 6, 'other') idle on its own slices bracketing
    the time, carrying current later: off there (zero), stamped
    off_idle in the reader's records as in the engine's."""
    dd = _example()
    fus = _entry(dd, 6, "fusion", [2.0, 2.4, 2.6], [0.0, 0.0, 4.0e3])
    p = _write(tmp_path, _with(dd, fus), "fusion_idle.json")
    bl = _read(p)
    c = _engine_read(p, bl)
    assert np.all(np.asarray(bl.j_other) == 0.0)
    assert np.all(c.jB_fix_parts["other"] == 0.0)
    e = [x for x in bl.source_time_match["entries"] if x["index"] == 6][0]
    assert e["status"] == "off_idle"
    assert [o["index"] for o in bl.source_time_match["off_sources"]] == [6]
    assert bl.source_time_match["off_sources"][0]["reason"].startswith(
        "off near the slice")
    _same_records(bl, c)


def test_an_offset_core_sources_base_is_refused_on_both(tmp_path):
    from bouquet.adapters import EngineInputRefused
    dd = _example()
    cs_t = [2.13, 2.27, 2.37]          # 70 ms either side of 2.2 s
    p = _write(tmp_path, _with(dd, cs_times=cs_t), "cs_offset.json")
    with pytest.raises(ValueError, match=r"IMAS reader: the core_sources "
                                         r"slice nearest.*Refusing"):
        _read(p)
    ok = _read(_write(tmp_path, _example(), "ok.json"))
    with pytest.raises(EngineInputRefused, match=r"IDS adapter: the "
                                                 r"core_sources slice nearest"):
        _engine_read(p, ok)


@pytest.mark.parametrize("own, why", [
    ([2.0, 2.4], "matched on its own 0.2 s half-step only, 0.2 s from the "
                 "core_profiles slice"),
    ([1.8, 1.9], "past its last own time, carrying current there"),
])
def test_a_sawteeth_entry_is_windowed_and_refused_like_any_other(tmp_path,
                                                                 own, why):
    """The sawteeth entry is held like every other driven entry, so it
    follows the same rule: no own slice within half the core_profiles step
    and current on its bracketing slices is REFUSED (the reader held the
    0.2 s-stale current / zeroed the past-the-record one)."""
    from bouquet.adapters import EngineInputRefused
    dd = _example()
    saw = _entry(dd, 701, "sawteeth", own, [3.0e3, 2.0e3])
    p = _write(tmp_path, _with(dd, saw), "saw.json")
    with pytest.raises(EngineInputRefused, match=r"IMAS reader: core_sources "
                       r"'sawteeth' \(index 701\) carries a non-zero "
                       r"j_parallel.*Refusing"):
        _read(p)
    ok = _read(_write(tmp_path, _example(), "ok.json"))
    with pytest.raises(EngineInputRefused, match=r"IDS adapter: core_sources "
                                                 r"'sawteeth' \(index 701\)"):
        _engine_read(p, ok)


def test_a_matched_sawteeth_entry_is_held_and_the_opt_out_leaves_it_inductive(
        tmp_path):
    dd = _example()
    saw = _entry(dd, 701, "sawteeth", dd["core_sources"]["time"],
                 [3.0e3, 3.0e3, 3.0e3])
    p = _write(tmp_path, _with(dd, saw), "saw_ok.json")
    held = _read(p)
    c = _engine_read(p, held)
    _same_records(held, c)
    assert np.max(np.abs(held.j_sawteeth)) > 0.0
    np.testing.assert_array_equal(held.j_other, held.j_sawteeth)
    assert held.source_time_match["sawteeth_hold"]["held"] is True
    out = _read(p, hold_sawteeth=False)
    assert out.source_time_match["sawteeth_hold"]["held"] is False
    assert not np.any(out.j_sawteeth) and not np.any(out.j_other)
    # the current moves to the residual inductive, nothing else changes
    np.testing.assert_allclose(out.j_inductive, held.j_inductive + held.j_other,
                               rtol=0, atol=1e-9 * np.max(np.abs(held.j_phi)))
    np.testing.assert_array_equal(out.j_phi, held.j_phi)
    np.testing.assert_array_equal(out.j_NBI, held.j_NBI)
    # the gate still reads it (a flag) and its match is still recorded
    assert out.sawtooth["active"] and held.sawtooth["active"]
    assert [e["index"] for e in out.source_time_match["entries"]
            if e["index"] == 701] == [701]


def test_the_engine_refuses_the_sawteeth_opt_out(tmp_path):
    import types
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    from bouquet.engine import prepare_engine_baseline
    cfg = BouquetConfig(
        source=ImasSource(ids_path=_OMAS, time=T, hold_sawteeth=False),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    bq = types.SimpleNamespace(config=cfg, mygs=object())
    with pytest.raises(ValueError, match="hold_sawteeth=False is a "
                                         "legacy-reader setting"):
        prepare_engine_baseline(bq)


def test_hold_sawteeth_must_be_a_bool():
    from bouquet.config import ImasSource
    with pytest.raises(ValueError, match="hold_sawteeth"):
        ImasSource(ids_path="x.json", hold_sawteeth="no")


@pytest.mark.parametrize("s_ip, s_b0", [(-1.0, 1.0), (1.0, -1.0),
                                        (-1.0, -1.0)])
def test_rf_and_other_channels_read_in_the_positive_ip_frame(tmp_path, s_ip,
                                                             s_b0):
    """Every driven channel, not only the beams, is brought into the
    positive-Ip frame: a mirrored copy (tests/_mirror_dd.py) with EC, IC,
    fusion and sawteeth entries reads bit-identically, records included."""
    import _mirror_dd as mdd
    dd = _example()
    cs_t = dd["core_sources"]["time"]
    dd = _with(dd,
               _entry(dd, 3, "ec", cs_t, [5.0e3, 6.0e3, 7.0e3]),
               _entry(dd, 5, "ic", cs_t, [1.0e3, 2.0e3, 3.0e3]),
               _entry(dd, 6, "fusion", cs_t, [4.0e2, 5.0e2, 6.0e2]),
               _entry(dd, 701, "sawteeth", cs_t, [3.0e3, 3.0e3, 3.0e3]))
    ref = _read(_write(tmp_path, dd, "pos.json"))
    bl = _read(_write(tmp_path, mdd.mirror_dd(dd, s_ip, s_b0), "mir.json"))
    assert bl.source_current_sign == s_ip
    for k in ("j_NBI", "j_RF", "j_other", "j_sawteeth", "j_inductive",
              "j_BS", "j_phi"):
        np.testing.assert_array_equal(getattr(bl, k), getattr(ref, k), k)
    assert np.all(np.asarray(ref.j_RF) > 0) and np.all(
        np.asarray(ref.j_other) > 0)
    assert bl.source_time_match["entries"] == ref.source_time_match["entries"]
