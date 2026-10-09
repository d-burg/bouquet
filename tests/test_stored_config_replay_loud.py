"""Stored-config replay is loud (2026-10-06 review, report D3).

* A loop config stored 2026-09-25..27 lacks ``jbs_relax_current`` /
  ``jbs_relax_halve_on`` (introduced 2026-09-27).  It ran with no current
  relaxation and omega halved on every growth (1.0 / 1); it loaded
  SILENTLY at today's 0.7 / 3.  Now it loads with 1.0 / 1 and ONE warning
  naming the fields.
* The "loud" :data:`~bouquet.config.FIELD_PRE_INTRODUCTION` entries
  (``l_i_tolerance``, ``jBS_scale_range``, ``homotopy_passes``,
  ``floor_j_BS``, ``jbs_max_passes_draw``) were consulted for unified
  configs only, so they never fired for a (legacy) config bouquet itself
  stored.  Now for every stored config.
* ``coil_solve_mode`` was written but never read back.  ``load_config``
  now reads it and warns when the stored run predates the canonical
  (bounded) mode or records another.

The fixture is the stored ``to_dict`` output of the synthetic config at
554edfb (both engines; the stripping is what the owner's 2026-10-06
session did by hand).  Solver-free.
"""
import copy
import json
import os
import warnings

import h5py
import numpy as np
import pytest

from bouquet.config import BouquetConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIX = os.path.join(_HERE, "data", "stored_configs", "config_554edfb.json")
RELAX = ("jbs_relax_current", "jbs_relax_halve_on")
LOUD = ("l_i_tolerance", "jBS_scale_range", "homotopy_passes", "floor_j_BS",
        "jbs_max_passes_draw")


def _stored(eng):
    with open(_FIX) as fh:
        return copy.deepcopy(json.load(fh)[eng])


def _load(d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = BouquetConfig.from_dict(d)
    return c, [str(x.message) for x in w]


@pytest.mark.parametrize("eng", ["legacy", "unified"])
def test_a_loop_config_without_the_relax_fields_loads_what_it_ran(eng):
    d = _stored(eng)
    assert d["generation"]["jbs_self_consistent"] is True
    c0, m0 = _load(copy.deepcopy(d))
    assert (c0.generation.jbs_relax_current,
            c0.generation.jbs_relax_halve_on) == (0.7, 3)
    assert not any("jbs_relax" in m for m in m0)
    for f in RELAX:
        d["generation"].pop(f)
    c, msgs = _load(d)
    assert c.generation.jbs_relax_current == 1.0
    assert c.generation.jbs_relax_halve_on == 1
    relax = [m for m in msgs if "jbs_relax" in m]
    assert len(relax) == 1                                   # warned ONCE
    assert "jbs_relax_current=1.0" in relax[0]
    assert "jbs_relax_halve_on=1" in relax[0]
    assert "today's default 0.7" in relax[0] and "today's default 3" in relax[0]


def test_one_missing_relax_field_is_back_filled_alone():
    d = _stored("legacy")
    d["generation"].pop("jbs_relax_halve_on")
    c, msgs = _load(d)
    assert c.generation.jbs_relax_halve_on == 1
    assert c.generation.jbs_relax_current == 0.7          # as stored
    relax = [m for m in msgs if "jbs_relax" in m]
    assert len(relax) == 1 and "jbs_relax_current" not in relax[0]


def test_a_config_that_did_not_run_the_loop_is_not_back_filled():
    """With the loop off the relax fields had no effect: today's defaults,
    no warning about them."""
    d = _stored("legacy")
    d["generation"]["jbs_self_consistent"] = False
    for f in RELAX:
        d["generation"].pop(f)
    c, msgs = _load(d)
    assert (c.generation.jbs_relax_current,
            c.generation.jbs_relax_halve_on) == (0.7, 3)
    assert not any("jbs_relax" in m for m in msgs)


def test_the_loud_entries_fire_for_a_stored_legacy_config():
    """The five entries whose earlier value is not knowable from the config
    alone fire, LOUDLY, on a stored legacy (loop) config missing them --
    each named, today's default used."""
    d = _stored("legacy")
    for f in LOUD:
        d["generation"].pop(f)
    c, msgs = _load(d)
    for f in LOUD:
        hit = [m for m in msgs if f"STORED CONFIG LACKS generation.{f}," in m]
        assert len(hit) == 1, f
        assert "results may differ" in hit[0]
    # and only for a missing field: the intact config raises none
    _, m0 = _load(_stored("legacy"))
    assert not any("LACKS" in m for m in m0)


def test_the_loop_loud_entry_waits_for_the_loop():
    """jbs_max_passes_draw is the loop's field: a legacy config that did
    not run the loop gets no warning for it; the other four still fire."""
    d = _stored("legacy")
    d["generation"]["jbs_self_consistent"] = False
    for f in LOUD:
        d["generation"].pop(f)
    _, msgs = _load(d)
    assert not any("generation.jbs_max_passes_draw" in m for m in msgs)
    for f in LOUD[:-1]:
        assert any(f"LACKS generation.{f}," in m for m in msgs), f


# ---------------------------------------------------------------------------
#  coil_solve_mode read back on load
# ---------------------------------------------------------------------------
def _archive(tmp_path, name, mode=None, key=0):
    from bouquet.utils import (initialize_equilibrium_database,
                               stamp_coil_solve_mode,
                               store_baseline_profiles, write_provenance)
    header = str(tmp_path / name)
    initialize_equilibrium_database(header)
    psi, one = np.linspace(0, 1, 9), np.ones(9)
    store_baseline_profiles(header, psi, one, one, one, one, one, one,
                            one, one, one, one, one, 1e6, 1.0, scan_key=key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = BouquetConfig.from_dict(_stored("legacy"))
    cfg.generation.scan_key = key
    write_provenance(header, config=cfg, scan_key=key)
    stamp_coil_solve_mode(header, scan_key=key, mode=mode)
    return header


def _load_cfg(header, key=0):
    from bouquet.utils import load_config
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_config(header, scan_key=key)
    return [str(x.message) for x in w if "coil_solve_mode" in str(x.message)]


def test_an_archive_without_coil_solve_mode_warns_on_load(tmp_path):
    from bouquet.utils import load_coil_solve_mode
    h = _archive(tmp_path, "old")
    assert load_coil_solve_mode(h, 0) == (None, None)
    msgs = _load_cfg(h)
    assert len(msgs) == 1
    assert "records no coil_solve_mode: it predates the canonical" in msgs[0]
    assert "<= 5e-4 on archived draws" in msgs[0]


def test_an_archive_in_the_canonical_mode_loads_quietly(tmp_path):
    from bouquet.utils import load_coil_solve_mode
    h = _archive(tmp_path, "new", mode="bounded")
    assert load_coil_solve_mode(h, 0) == ("bounded",
                                          "_baseline attr coil_solve_mode")
    assert _load_cfg(h) == []
    h2 = _archive(tmp_path, "other", mode="unknown")
    msgs = _load_cfg(h2)
    assert len(msgs) == 1 and "coil_solve_mode='unknown'" in msgs[0]


def test_the_engine_record_is_read_when_the_attr_is_absent(tmp_path):
    from bouquet.engine import store_baseline_engine
    from bouquet.utils import load_coil_solve_mode
    h = _archive(tmp_path, "eng")
    store_baseline_engine(h, dict(coil_solve_mode="bounded"), scan_key=0)
    assert load_coil_solve_mode(h, 0) == ("bounded", "baseline engine record")
    assert _load_cfg(h) == []
