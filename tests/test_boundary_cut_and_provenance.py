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
    assert FilterConfig().rms_max_mm is None


def _cfg(tmp_path, device=None, rms=None):
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


def test_the_loop_and_the_filter_resolve_the_cut_the_same_way():
    from bouquet.run import Bouquet
    assert "self._boundary_cut()[0]" in inspect.getsource(Bouquet.generate)
    flt = inspect.getsource(Bouquet.filter)
    assert "self._boundary_cut()" in flt and "cut_source=rms_source" in flt


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

def _shard(path, worker, n_draws, outcomes):
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
