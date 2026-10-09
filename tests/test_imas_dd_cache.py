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
