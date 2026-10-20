"""The parsed dd_sim.json is cached across slices (``_cached_dd``) and shared
by the readers, which must not mutate it.

Synthetic inputs only (the shipped D3D-like OMAS example).
"""
import json
import os
import shutil

import numpy as np
import pytest

from bouquet.config import ImasSource
from bouquet.io import imas

_OMAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "examples", "D3D-like", "D3Dlike_baseline_omas.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(_OMAS),
                                reason="synthetic IMAS example absent")


def _state(bl):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
            for k, v in vars(bl).items()
            if isinstance(v, (np.ndarray, float, int, str, bool))}


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            assert a[k] == b[k], k


@pytest.fixture()
def dd_path(tmp_path):
    p = str(tmp_path / "dd_sim.json")
    shutil.copy(_OMAS, p)
    imas._cached_dd.cache_clear()
    yield p
    imas._cached_dd.cache_clear()


def test_a_slice_sweep_parses_once_and_rereads_identically(dd_path):
    first = _state(imas.read_imas_baseline(ImasSource(ids_path=dd_path,
                                                      time=2.3043)))
    imas.read_imas_baseline(ImasSource(ids_path=dd_path, time=2.2))
    imas.read_imas_geometry(ImasSource(ids_path=dd_path, time=2.2))
    again = _state(imas.read_imas_baseline(ImasSource(ids_path=dd_path,
                                                      time=2.3043)))
    _same(first, again)
    info = imas._cached_dd.cache_info()
    assert (info.misses, info.hits) == (1, 3)
    with open(dd_path) as fh:
        assert imas._load_dd(dd_path) == json.load(fh)


def test_a_rewritten_file_is_parsed_again(dd_path):
    imas._load_dd(dd_path)
    with open(dd_path) as fh:
        dd = json.load(fh)
    dd["edited"] = True
    with open(dd_path, "w") as fh:
        json.dump(dd, fh)
    assert imas._load_dd(dd_path)["edited"] is True
    assert imas._cached_dd.cache_info().misses == 2


# ---------------------------------------------------------------------------
#  #72 review: the engine adapter and the plotting readers share the cache,
#  and the shared object is never written into (digest-pinned)
# ---------------------------------------------------------------------------
def _digest(dd):
    import hashlib
    return hashlib.sha256(json.dumps(dd, sort_keys=True).encode()).hexdigest()


def _engine_read(dd_path, t):
    import warnings
    from bouquet.adapters import IdsAdapter
    from bouquet.baseline import resolve_baseline
    from bouquet.config import BouquetConfig, GenerationConfig, SolverConfig
    cfg = BouquetConfig(
        source=ImasSource(ids_path=dd_path, time=t),
        solver=SolverConfig(mesh_path=None), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
        return IdsAdapter(cfg.source, cfg, bl).read()


def test_the_engine_adapter_reads_a_second_slice_without_re_parsing(dd_path):
    _engine_read(dd_path, 2.3043)
    assert imas._cached_dd.cache_info().misses == 1
    _engine_read(dd_path, 2.2)
    assert imas._cached_dd.cache_info().misses == 1     # no second parse


def test_slice_reads_leave_the_shared_dd_unchanged(dd_path):
    from bouquet.plotting import _imas_input_profiles
    shared = imas._load_dd(dd_path)
    before = _digest(shared)
    for t in (2.3043, 2.2):
        imas.read_imas_baseline(ImasSource(ids_path=dd_path, time=t))
        imas.read_imas_geometry(ImasSource(ids_path=dd_path, time=t))
        _engine_read(dd_path, t)
        _imas_input_profiles(ImasSource(ids_path=dd_path, time=t))
    assert imas._load_dd(dd_path) is shared             # one object, served
    assert _digest(shared) == before                    # ... never written
    with open(dd_path) as fh:
        assert _digest(json.load(fh)) == before


def test_path_spellings_share_one_entry(dd_path, monkeypatch):
    import pathlib
    monkeypatch.chdir(os.path.dirname(dd_path))
    a = imas._load_dd(dd_path)
    b = imas._load_dd("dd_sim.json")
    c = imas._load_dd(pathlib.Path(dd_path))
    assert a is b is c
    assert imas._cached_dd.cache_info().misses == 1


def test_a_same_size_rewrite_is_parsed_again(dd_path):
    """mtime alone (same size) and a rename-replace with the SAME size and
    mtime (a new inode) both invalidate."""
    first = imas._load_dd(dd_path)
    with open(dd_path) as fh:
        text = fh.read()
    st = os.stat(dd_path)
    edited = text.replace('"time"', '"tIme"', 1)
    assert len(edited) == len(text)
    with open(dd_path, "w") as fh:                       # same size, new mtime
        fh.write(edited)
    os.utime(dd_path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    second = imas._load_dd(dd_path)
    assert second is not first and "tIme" in json.dumps(second)
    st2 = os.stat(dd_path)
    tmp = dd_path + ".new"
    with open(tmp, "w") as fh:                           # same size and mtime,
        fh.write(text)                                   # replaced by rename
    os.utime(tmp, ns=(st2.st_atime_ns, st2.st_mtime_ns))
    os.replace(tmp, dd_path)
    third = imas._load_dd(dd_path)
    assert third is not second and "tIme" not in json.dumps(third)


def test_clear_dd_cache_releases_the_parse(dd_path):
    imas._load_dd(dd_path)
    imas.clear_dd_cache()
    assert imas._cached_dd.cache_info().currsize == 0
    imas._load_dd(dd_path)
    assert imas._cached_dd.cache_info().misses == 1


# ---------------------------------------------------------------------------
#  lean read: given a time, the time-sliced lists are whole only near it
# ---------------------------------------------------------------------------
def _sliced_dd(path, ensure_ascii=True):
    ts = (0.0, 1.0, 2.0, 3.0, 4.0)
    dd = {"equilibrium": {"time": list(ts), "time_slice": [
              {"time": t, "boundary": {"r": [t] * 3}} for t in ts]},
          "core_sources": {"time": list(ts), "source": [{"identifier": {"index": 2},
              "profiles_1d": [{"time": t, "j_parallel": [t, 0.0], "torque": [1.0]}
                              for t in ts]}]},
          "summary": {"code": {"name": "FUSE é"}}}
    with open(path, "w") as fh:
        json.dump(dd, fh, ensure_ascii=ensure_ascii)
    return dd


def test_a_time_keeps_its_slices_whole_and_cuts_the_rest(tmp_path):
    p = str(tmp_path / "dd_sim.json")
    whole = _sliced_dd(p)
    lean = imas._load_dd(p, 2.4)       # nearest 2.0, outside the window
    eq, src = lean["equilibrium"]["time_slice"], lean["core_sources"]["source"][0]
    assert eq[1:4] == whole["equilibrium"]["time_slice"][1:4]   # with its neighbours
    assert (eq[0], eq[4]) == ({"time": 0.0}, {"time": 4.0})
    assert src["profiles_1d"][0] == {"time": 0.0, "j_parallel": [0.0, 0.0]}
    assert lean["summary"] == whole["summary"]
    imas.clear_dd_cache()


def test_a_non_ascii_dd_and_lean_dd_off_read_it_whole(tmp_path):
    p = str(tmp_path / "dd_sim.json")
    whole = _sliced_dd(p, ensure_ascii=False)
    assert imas._load_dd(p, 1.0) == whole
    assert imas._source_dd(ImasSource(ids_path=p, time=1.0)) == whole
    assert imas._source_dd(ImasSource(ids_path=p, time=1.0, lean_dd=True)) is \
        imas._load_dd(p, 1.0)
    imas.clear_dd_cache()


def test_a_time_far_from_every_core_profiles_slice_reads_it_whole(tmp_path):
    p = str(tmp_path / "dd_sim.json")
    whole = _sliced_dd(p)
    whole["core_profiles"] = {"time": [0.0], "profiles_1d": [{"time": 0.0}]}
    with open(p, "w") as fh:
        json.dump(whole, fh)
    assert imas._load_dd(p, 2.4) == whole
    imas.clear_dd_cache()
