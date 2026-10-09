"""Archive provenance for error bars (solve-free).

* the LCFS boundary cut is DEVICE-calibrated by default (8.5 mm on DIII-D
  from its boundary-UQ study; generic 5.0 mm), explicit ``rms_max_mm`` wins,
  and the resolved cut + its source are stamped on the archive;
* ``filter_boundaries`` records the cut and each draw's metric;
* generation provenance (requested / attempted / stored, per-attempt
  outcomes, generating version) is stamped and read back, never inferred;
* a refused slice is a recorded fact, not a gap;
* the parallel merge aggregates the shards' provenance.
"""
import inspect
import json

import h5py
import numpy as np
import pytest

from bouquet.config import (BouquetConfig, ImasSource, SolverConfig,
                            FilterConfig)
from bouquet.devices import (DEVICES, boundary_cut_for, GENERIC_BOUNDARY_RMS_MM,
                             DeviceSpec)
from bouquet.filtering import filter_boundaries, read_filter_flags
from bouquet.utils import (stamp_generation_provenance, read_generation_provenance,
                           write_refused_scan, GENERATION_PROVENANCE_KEYS)


def _circle(r=1.0, n=400, R0=1.7):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.c_[R0 + r * np.cos(th), r * np.sin(th)]


# ---------------------------------------------------------------------------
# device calibration
# ---------------------------------------------------------------------------

def test_diiid_boundary_cut_is_calibrated_and_generic_is_five():
    spec = DEVICES["DIII-D"]
    assert boundary_cut_for(spec) == (8.5, "device:DIII-D")
    assert spec.boundary_provenance            # a stated origin, not a bare number
    assert boundary_cut_for(None) == (GENERIC_BOUNDARY_RMS_MM, "generic")
    bare = DeviceSpec(name="X", coil_signature=frozenset({"C1"}), sigma_floor=1.0,
                      sigma_fraction=0.0, sigma_provenance="")
    assert boundary_cut_for(bare) == (5.0, "generic")


def test_filterconfig_default_is_unresolved_not_five():
    # "auto": resolved per device at run time, never a hard-coded 5.0
    assert FilterConfig().rms_max_mm == "auto"


@pytest.mark.parametrize("bad", ["none", "Auto", True, float("nan"), float("inf"), [5.0]])
def test_filterconfig_refuses_an_unknown_boundary_cut_setting(bad):
    with pytest.raises(ValueError, match="rms_max_mm"):
        FilterConfig(rms_max_mm=bad)


@pytest.mark.parametrize("ok", ["auto", "off", None, 5.0, 3, np.float32(2.5)])
def test_filterconfig_accepts_the_documented_settings(ok):
    assert FilterConfig(rms_max_mm=ok).rms_max_mm is ok


def _cfg(tmp_path, device=None, rms="auto"):
    cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                        solver=SolverConfig(mesh_path="unused.h5"),
                        output_header=str(tmp_path / "run"),
                        filtering=FilterConfig(rms_max_mm=rms), device=device)
    cfg.generation.scan_key = 0
    return cfg


def test_bouquet_resolves_explicit_then_device_then_generic(tmp_path, capsys):
    from bouquet.run import Bouquet
    b = Bouquet(_cfg(tmp_path, rms=3.0))
    assert b._boundary_cut() == (3.0, "explicit")
    b._boundary_cut()                                   # second call: no repeat
    out = capsys.readouterr().out
    assert out.count("[boundary cut]") == 1 and "3 mm (explicit" in out
    assert Bouquet(_cfg(tmp_path, device="DIII-D"))._boundary_cut() == (8.5, "device:DIII-D")
    out = capsys.readouterr().out
    assert "8.5 mm" in out and "overrides" in out       # announced once, loudly
    assert Bouquet(_cfg(tmp_path))._boundary_cut() == (5.0, "generic")   # no mesh, no archive


def test_the_boundary_cut_can_be_switched_off_explicitly(tmp_path, capsys):
    from bouquet.run import Bouquet
    b = Bouquet(_cfg(tmp_path, device="DIII-D", rms="off"))
    assert b._boundary_cut() == (None, "disabled")
    assert b.boundary_cut() == (None, "disabled")          # public, quiet
    out = capsys.readouterr().out
    assert out.count("[boundary cut]") == 1 and "DISABLED" in out and "'off'" in out


def test_none_keeps_its_historical_meaning_no_cut(tmp_path, capsys):
    """Before the device-calibrated default, rms_max_mm=None meant "no LCFS
    cut" in both the loop and filter(). A caller who set None deliberately
    still gets no cut, and is told so -- never a device cut instead."""
    from bouquet.run import Bouquet
    assert Bouquet(_cfg(tmp_path, device="DIII-D", rms=None))._boundary_cut() == \
        (None, "disabled")
    out = capsys.readouterr().out
    assert "DISABLED" in out and "historical meaning" in out
    # the same through a stored config JSON written with null
    cfg = _cfg(tmp_path, device="DIII-D", rms=None)
    back = BouquetConfig.from_dict(json.loads(cfg.to_json()))
    assert back.filtering.rms_max_mm is None
    assert Bouquet(back)._boundary_cut(quiet=True) == (None, "disabled")
    # ...and a config that does not mention it gets "auto"
    d = json.loads(cfg.to_json())
    d["filtering"].pop("rms_max_mm")
    assert BouquetConfig.from_dict(d).filtering.rms_max_mm == "auto"


def test_the_livelock_guard_suggests_a_setting_that_disables_the_bound():
    from bouquet.TokaMaker_interface import generate_bouquet
    src = inspect.getsource(generate_bouquet)
    assert "filtering.rms_max_mm='off'" in src
    assert "filtering.rms_max_mm=None) to target" not in src
    assert FilterConfig(rms_max_mm="off").rms_max_mm == "off"


def test_disabled_cut_clears_an_earlier_boundary_verdict(tmp_path):
    h5 = str(tmp_path / "b.h5")
    _boundary_archive(h5)
    filter_boundaries(h5, scan_key=0, rms_max_mm=5.0, plot=False)
    assert read_filter_flags(h5, scan_key=0)[1]["selected"] is False
    filter_boundaries(h5, scan_key=0, plot=False, cut_source="disabled")
    flags = read_filter_flags(h5, scan_key=0)
    assert all("passes_boundary_filter" not in f for f in flags.values())
    assert all(f["selected"] is True for f in flags.values())
    assert flags[1]["boundary_rms_mm"] == pytest.approx(10.0, abs=0.2)
    with h5py.File(h5, "r") as hf:
        a = hf["scan/0"].attrs
        assert a["boundary_cut_source"] == "disabled"
        assert "boundary_rms_max_mm" not in a
    with pytest.raises(ValueError, match="disabled"):
        filter_boundaries(h5, scan_key=0, rms_max_mm=5.0, plot=False,
                          cut_source="disabled")


def test_plots_refuse_a_setting_word_as_a_threshold(tmp_path):
    from bouquet.plotting import plot_traces, plot_spec_summary
    with pytest.raises(ValueError, match="boundary_cut"):
        plot_traces(str(tmp_path / "x.h5"), rms_max_mm="auto")
    with pytest.raises(ValueError, match="boundary_cut"):
        plot_spec_summary(str(tmp_path / "x"), rms_max_mm="off")


def test_bouquet_detects_the_device_from_the_archived_coil_names(tmp_path):
    """A filter-only session has no live solver: the baseline's coil names
    (the same ones the chi2 filter reads) identify the device."""
    from bouquet.run import Bouquet
    cfg = _cfg(tmp_path)
    names = sorted(DEVICES["DIII-D"].coil_signature)
    with h5py.File(f"{cfg.output_header}.h5", "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.create_dataset("coil_names", data=np.array(names, dtype=h5py.string_dtype()))
    assert Bouquet(cfg)._boundary_cut(quiet=True) == (8.5, "device:DIII-D")


# The loop-vs-filter identity of the cut is tested BEHAVIOURALLY in
# tests/test_until_n_inspec.py (test_the_loop_and_the_filter_resolve_the_cut_
# the_same_way, test_the_default_device_cut_is_the_same_number_in_loop_stamp_
# and_filter); the source grep that stood here could not see a wrong value.


# ---------------------------------------------------------------------------
# filter_boundaries stamps the cut and the per-draw metric
# ---------------------------------------------------------------------------

def _boundary_archive(path):
    offsets = [0.000, 0.010, 0.003]         # 0, 10, 3 mm
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.create_dataset("recon_lcfs_ref", data=_circle())
        for i, dr in enumerate(offsets):
            d = hf.create_group(f"scan/0/{i}")
            d.create_dataset("perturbed_lcfs_ref", data=_circle(r=1.0 + dr))
    return offsets


def test_apply_stamps_the_cut_its_source_and_each_draw(tmp_path):
    h5 = str(tmp_path / "b.h5")
    _boundary_archive(h5)
    summ, _ = filter_boundaries(h5, scan_key=0, rms_max_mm=8.5, plot=False,
                                cut_source="device:DIII-D")
    assert summ["n_pass"] == 2
    with h5py.File(h5, "r") as hf:
        a = hf["scan/0"].attrs
        assert a["boundary_rms_max_mm"] == 8.5
        assert a["boundary_cut_source"] == "device:DIII-D"
        assert "boundary_max_max_mm" not in a
    flags = read_filter_flags(h5, scan_key=0)
    assert flags[1]["passes_boundary_filter"] is False
    assert flags[1]["boundary_rms_mm"] == pytest.approx(10.0, abs=0.2)
    assert flags[0]["boundary_rms_mm"] == pytest.approx(0.0, abs=0.05)
    assert "boundary_max_mm" in flags[2]


def test_apply_false_or_no_cut_writes_nothing(tmp_path):
    h5 = str(tmp_path / "b.h5")
    _boundary_archive(h5)
    filter_boundaries(h5, scan_key=0, rms_max_mm=8.5, apply=False, plot=False)
    filter_boundaries(h5, scan_key=0, plot=False)                 # diagnostic only
    with h5py.File(h5, "r") as hf:
        assert "boundary_rms_max_mm" not in hf["scan/0"].attrs
        assert "boundary_rms_mm" not in hf["scan/0/0"].attrs


# ---------------------------------------------------------------------------
# generation provenance
# ---------------------------------------------------------------------------

def test_provenance_round_trip_and_unrecorded_is_none(tmp_path):
    h = str(tmp_path / "p")
    with h5py.File(h + ".h5", "w") as hf:
        hf.create_group("scan/0/_baseline")
    rec = read_generation_provenance(h + ".h5", scan_key=0)
    assert set(rec) == set(GENERATION_PROVENANCE_KEYS)
    assert all(v is None for v in rec.values())          # never inferred
    stamp_generation_provenance(
        h, scan_key=0, n_requested=15, n_requested_source="n_inspec_target",
        generation_mode="until_n", n_attempted=22, n_stored=19,
        attempt_outcomes_json={0: "stored", 1: "solve_failed", 2: "post_align_failed"},
        bouquet_version="1.4.0")
    rec = read_generation_provenance(h, scan_key=0)
    assert rec["n_requested"] == 15 and rec["n_attempted"] == 22 and rec["n_stored"] == 19
    assert rec["generation_mode"] == "until_n"
    assert rec["attempt_outcomes_json"] == {"0": "stored", "1": "solve_failed",
                                            "2": "post_align_failed"}
    assert rec["bouquet_version"] == "1.4.0"


def test_generate_bouquet_records_every_exit_of_an_attempt():
    """Each way an attempt can end is labelled, and the stamp follows the loop."""
    from bouquet.TokaMaker_interface import generate_bouquet
    src = inspect.getsource(generate_bouquet)
    for label in ("solve_failed", "post_align_failed", "stored"):
        assert f'_attempt_outcomes[int(count)] = "{label}"' in src
    assert src.index("stamp_generation_provenance(") > src.index('= "stored"')
    assert "n_attempted=int(_n_att)" in src


def test_refused_slice_is_recorded_not_a_gap(tmp_path):
    h = str(tmp_path / "r")
    write_refused_scan(h, 4400, "no l_i reference within 30 ms")
    with h5py.File(h + ".h5", "a") as hf:
        assert hf["scan/4400"].attrs["refused_reason"] == "no l_i reference within 30 ms"
        hf.create_group("scan/4400/0")
    with pytest.raises(ValueError, match="already holds draws"):
        write_refused_scan(h, 4400, "again")


# ---------------------------------------------------------------------------
# parallel merge aggregates the shards' provenance
# ---------------------------------------------------------------------------

def _shard(path, worker, n_draws, outcomes, cut=None):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["l_i_target"] = 0.9
        g.attrs["Ip_target"] = 1.0e6
        for i in range(n_draws):
            hf.create_group(f"scan/0/{i}").attrs["count"] = i
        sg = hf["scan/0"]
        sg.attrs["parallel_worker_json"] = json.dumps(
            dict(worker_id=worker, n=n_draws, n_attempts=len(outcomes),
                 n_inspec=n_draws, seed=7 + worker, shared_target=3,
                 local_target=3, attempt_cap=4, total_cap=8))
        sg.attrs["attempt_outcomes_json"] = json.dumps(outcomes)
        sg.attrs["n_attempted"] = len(outcomes)
        sg.attrs["bouquet_version"] = "1.4.0"
        if cut is not None:                  # the until-N loop's recorded cut
            sg.attrs["inspec_rms_max_mm"] = cut[0]
            sg.attrs["inspec_cut_source"] = cut[1]


def test_merge_sums_attempts_and_keys_outcomes_by_worker(tmp_path):
    from bouquet.parallel import merge_archives
    s0, s1 = str(tmp_path / "w0.h5"), str(tmp_path / "w1.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "solve_failed", 2: "stored"})
    _shard(s1, 1, 1, {0: "stored"})
    out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    assert n == 3
    rec = read_generation_provenance(out, scan_key=0)
    assert rec["n_attempted"] == 4 and rec["n_stored"] == 3
    assert rec["generation_mode"] == "until_n" and rec["n_requested"] == 3
    assert rec["attempt_outcomes_json"]["0"]["1"] == "solve_failed"
    assert rec["attempt_outcomes_json"]["1"] == {"0": "stored"}
    assert rec["bouquet_version"] == "1.4.0"


def test_merge_carries_the_loops_boundary_cut(tmp_path):
    from bouquet.parallel import merge_archives
    s0, s1 = str(tmp_path / "w0.h5"), str(tmp_path / "w1.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "stored"}, cut=(8.5, "device:DIII-D"))
    _shard(s1, 1, 1, {0: "stored"}, cut=(8.5, "device:DIII-D"))
    out, _ = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    rec = read_generation_provenance(out, scan_key=0)
    assert rec["inspec_rms_max_mm"] == 8.5 and rec["inspec_cut_source"] == "device:DIII-D"
    with h5py.File(out, "r") as hf:
        man = json.loads(hf["scan/0"].attrs["parallel_manifest_json"])
    assert man["until_n"]["inspec_rms_max_mm"] == 8.5
    assert man["until_n"]["inspec_cut_source"] == "device:DIII-D"
    assert all(w["inspec_rms_max_mm"] == 8.5 for w in man["workers"])


def test_merge_refuses_shards_counted_against_different_cuts(tmp_path):
    from bouquet.parallel import merge_archives
    s0, s1 = str(tmp_path / "w0.h5"), str(tmp_path / "w1.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "stored"}, cut=(8.5, "device:DIII-D"))
    _shard(s1, 1, 1, {0: "stored"}, cut=(5.0, "generic"))
    with pytest.raises(RuntimeError, match="different boundary cuts"):
        merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    assert not (tmp_path / "m.h5").exists()           # nothing was merged


# ---------------------------------------------------------------------------
# refused slices: written by the pipeline, superseded by a later generation
# ---------------------------------------------------------------------------

def _store_slice(header, key, n_draws, tmp_path):
    """A baseline + n draws through the real archive writers."""
    from bouquet.utils import (store_equilibrium, store_baseline_profiles,
                               initialize_equilibrium_database)
    initialize_equilibrium_database(header)
    psi, one = np.linspace(0, 1, 9), np.ones(9)
    eq_path = str(tmp_path / "in.eqdsk")
    with open(eq_path, "wb") as fh:
        fh.write(b"GEQDSK-BYTES")
    store_baseline_profiles(header, psi, one, one, one, one, one, one,
                            one, one, one, one, one, 1e6, 1.0, scan_key=key)
    for c in range(n_draws):
        store_equilibrium(header, c, eq_path, psi, one, one, one,
                          one, one, one, one, one, 1.0, 0.8, scan_key=key)
    with h5py.File(header + ".h5", "a") as hf:        # a stamped filter
        hf[f"scan/{key}"].attrs["coil_filter"] = "chi2"
        for c in range(n_draws):
            g = hf[f"scan/{key}/{c}"]
            g.attrs["passes_coil_filter"] = True
            g.attrs["selected"] = True
            g.attrs["val"] = float(c)


def _slice_bouquet(tmp_path, monkeypatch, refuse_at):
    """A Bouquet whose solver stages are stubbed: prepare_baseline raises for
    the times in *refuse_at*; generate writes 5 draws through the writers."""
    from types import SimpleNamespace
    from bouquet.run import Bouquet
    b = Bouquet(_cfg(tmp_path))
    state = {}
    monkeypatch.setattr(b, "setup_solver", lambda: None)
    monkeypatch.setattr(b, "set_slice", lambda time: state.update(t=time))
    monkeypatch.setattr(b, "filter", lambda *a, **k: None)

    def prep():
        if state["t"] in refuse_at:
            raise RuntimeError("closure gate: scale out of bounds")
        b.baseline = SimpleNamespace(l_i_target=1.0, Ip_target=1e6)

    monkeypatch.setattr(b, "prepare_baseline", prep)
    monkeypatch.setattr(b, "generate", lambda *a, **k: _store_slice(
        b.config.output_header, b.config.generation.scan_key, 5, tmp_path))
    return b


def test_run_slices_records_a_refused_slice_and_carries_on(tmp_path, monkeypatch, capsys):
    import bouquet as bq
    b = _slice_bouquet(tmp_path, monkeypatch, refuse_at={2.0})
    res = b.run_slices([1.0, 2.0, 3.0], on_refusal="record")
    assert "closure gate" in res[2000]["refused"] and res[2000]["n_all"] == 0
    assert res[1000]["n_all"] == 5 and res[3000]["n_all"] == 5
    assert "[refused] scan 2000" in capsys.readouterr().out
    h5 = b.config.output_header + ".h5"
    t = bq.draw_bands([(h5, k) for k in (1000, 2000, 3000)],
                      lambda v: {"x": float(v.attrs.get("val", 0.0))})
    assert [(r.scan_key, r.status) for r in t] == [
        ("1000", "ok"), ("2000", "refused"), ("3000", "ok")]
    assert "RuntimeError: closure gate" in t.records[1].refused_reason
    assert bq.BouquetArchive(h5)["2000"].refused_reason.startswith("prepare_baseline raised")


def test_run_slices_raise_still_raises_but_the_refusal_is_on_disk(tmp_path, monkeypatch):
    """on_refusal="raise" (the default until 2026-10-06; now opt-in) re-raises
    at the first refused slice, the refusal written first."""
    b = _slice_bouquet(tmp_path, monkeypatch, refuse_at={2.0})
    with pytest.raises(RuntimeError, match="closure gate"):
        b.run_slices([1.0, 2.0], on_refusal="raise")
    with h5py.File(b.config.output_header + ".h5", "r") as hf:
        assert "closure gate" in hf["scan/2000"].attrs["refused_reason"]
    with pytest.raises(ValueError, match="on_refusal"):
        b.run_slices([1.0], on_refusal="skip")


def test_run_slices_records_by_default_and_summarises(tmp_path, monkeypatch,
                                                     capsys):
    """Owner decision 2026-10-06: run_slices defaults to on_refusal="record"
    -- one refused baseline no longer ends a series.  The refused slice is
    in the summary (reason + time) and in the archive (refused_reason,
    refused_time), the run carries on, and the count and reasons are printed
    and warned once at the end.  Mutant: the old default raises here."""
    import bouquet as bq
    b = _slice_bouquet(tmp_path, monkeypatch, refuse_at={2.0, 4.0})
    with pytest.warns(UserWarning, match=r"run_slices: 2 of 4 slices REFUSED"):
        res = b.run_slices([1.0, 2.0, 3.0, 4.0])
    assert [res[k]["n_all"] for k in (1000, 2000, 3000, 4000)] == [5, 0, 5, 0]
    for k, t in ((2000, 2.0), (4000, 4.0)):
        assert "closure gate" in res[k]["refused"] and res[k]["time"] == t
    out = capsys.readouterr().out
    assert "[run_slices] run_slices: 2 of 4 slices REFUSED" in out
    assert "scan 2000 (t = 2 s): prepare_baseline raised RuntimeError: " \
        "closure gate" in out
    h5 = b.config.output_header + ".h5"
    with h5py.File(h5, "r") as hf:
        assert hf["scan/2000"].attrs["refused_time"] == 2.0
        assert "closure gate" in hf["scan/4000"].attrs["refused_reason"]
        assert "refused_reason" not in hf["scan/3000"].attrs
    t = bq.draw_bands([(h5, k) for k in (1000, 2000, 3000, 4000)],
                      lambda v: {"x": float(v.attrs.get("val", 0.0))})
    assert [r.status for r in t] == ["ok", "refused", "ok", "refused"]


def test_a_series_config_stored_by_run_slices_loads_unchanged(tmp_path,
                                                              monkeypatch):
    """The default change is a run_slices argument, not a config field: the
    config each slice stores loads back unchanged (and without a warning),
    the refused slices in between notwithstanding."""
    import warnings
    from bouquet.utils import (load_config, stamp_coil_solve_mode,
                               write_provenance)
    b = _slice_bouquet(tmp_path, monkeypatch, refuse_at={2.0})
    gen = b.generate

    def gen_and_stamp(*a, **k):
        # what Bouquet.generate stamps: the config and the coil-solve mode
        gen()
        write_provenance(b.config.output_header, config=b.config,
                         scan_key=b.config.generation.scan_key)
        stamp_coil_solve_mode(b.config.output_header,
                              scan_key=b.config.generation.scan_key,
                              mode="bounded")
    monkeypatch.setattr(b, "generate", gen_and_stamp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.run_slices([1.0, 2.0, 3.0])
    want = b.config.to_dict()
    for k in (1000, 3000):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = load_config(b.config.output_header, scan_key=k)
        d = got.to_dict()
        assert d["generation"].pop("scan_key") == k
        w = dict(want, generation=dict(want["generation"]))
        w["generation"].pop("scan_key")
        assert d == w


def test_run_records_the_refusal_then_reraises(tmp_path, monkeypatch):
    from bouquet.run import Bouquet
    b = Bouquet(_cfg(tmp_path))
    monkeypatch.setattr(b, "setup_solver", lambda: None)

    def boom():
        raise RuntimeError("no l_i reference")
    monkeypatch.setattr(b, "prepare_baseline", boom)
    with pytest.raises(RuntimeError, match="no l_i reference"):
        b.run()
    with h5py.File(b.config.output_header + ".h5", "r") as hf:
        assert "no l_i reference" in hf["scan/0"].attrs["refused_reason"]


def test_a_regenerated_slice_supersedes_its_refusal(tmp_path):
    import bouquet as bq
    header = str(tmp_path / "regen")
    bq.write_refused_scan(header, 7, "closure gate rejected")
    ev = lambda v: {"x": float(v.attrs.get("val", 0.0))}
    assert bq.draw_band(header + ".h5", "7", ev)[None].status == "refused"
    _store_slice(header, 7, 6, tmp_path)              # a later generation
    with h5py.File(header + ".h5", "r") as hf:
        a = hf["scan/7"].attrs
        assert "refused_reason" not in a
        assert a["refused_reason_superseded"] == "closure gate rejected"
    r = bq.draw_band(header + ".h5", "7", ev)["x"]
    assert r.status == "ok" and r.n_used == 6 and r.refused_reason is None
    assert bq.BouquetArchive(header + ".h5")["7"].refused_reason is None


def test_a_refused_scan_that_holds_draws_is_refused_by_the_reader(tmp_path):
    import bouquet as bq
    header = str(tmp_path / "bad")
    _store_slice(header, 7, 5, tmp_path)
    with h5py.File(header + ".h5", "a") as hf:        # hand-edited, inconsistent
        hf["scan/7"].attrs["refused_reason"] = "stale"
    with pytest.raises(ValueError, match="inconsistent"):
        bq.draw_band(header + ".h5", "7", lambda v: {"x": 1.0})
    with pytest.raises(KeyError, match="refused"):
        bq.BouquetArchive(header + ".h5")["7"].baseline


def test_merge_refuses_shards_from_different_generation_setups(tmp_path):
    from bouquet.parallel import merge_archives
    s0, s1 = str(tmp_path / "w0.h5"), str(tmp_path / "w1.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "stored"})
    _shard(s1, 1, 1, {0: "stored"})
    with h5py.File(s1, "a") as hf:                    # a fixed-N worker record
        hf["scan/0"].attrs["parallel_worker_json"] = json.dumps(
            dict(worker_id=1, n=1, n_attempts=1, n_inspec=0, seed=8))
    with pytest.raises(RuntimeError, match="different generation set-ups"):
        merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    assert not (tmp_path / "m.h5").exists()


def test_merge_does_not_sum_attempts_over_shards_without_a_record(tmp_path):
    from bouquet.parallel import merge_archives
    s0, s1 = str(tmp_path / "w0.h5"), str(tmp_path / "w1.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "stored"})
    _shard(s1, 1, 3, {0: "stored", 1: "stored", 2: "stored"})
    with h5py.File(s1, "a") as hf:
        del hf["scan/0"].attrs["parallel_worker_json"]
    out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    rec = read_generation_provenance(out, scan_key=0)
    assert n == 5 and rec["n_stored"] == 5
    assert rec["n_attempted"] is None          # 2 recorded < 5 stored: unknown, not 2
    with h5py.File(out, "r") as hf:
        assert json.loads(hf["scan/0"].attrs["parallel_manifest_json"])[
            "n_shards_without_record"] == 1


def test_a_partial_merge_is_marked(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = str(tmp_path / "w0.h5")
    _shard(s0, 0, 2, {0: "stored", 1: "stored"})
    out, _ = merge_archives([s0], str(tmp_path / "m"), scan_key=0,
                            missing_workers=[1])
    rec = read_generation_provenance(out, scan_key=0)
    assert rec["merge_partial_json"] == {"missing_workers": [1], "merged_workers": [0]}
    assert "PARTIAL merge" in rec["n_requested_source"]
    with h5py.File(out, "r") as hf:
        assert json.loads(hf["scan/0"].attrs["parallel_manifest_json"])["partial"][
            "missing_workers"] == [1]
