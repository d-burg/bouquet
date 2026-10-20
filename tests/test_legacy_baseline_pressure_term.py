"""The legacy baseline archives the p'G of its OWN archived state as
``jphi_pd`` (review of #78, B2; owner decision: the schema's "p'G of the
archived state", consistent with the draws).

Before, the legacy IMAS baseline archived the dd source's p'G (the reader's
``Baseline.jphi_pd``, on the dd's geometry and pressure) while every
legacy draw archived its own state's, so a sigma = 0 draw split
``jphi_ind`` / ``jphi_pd`` differently from its baseline.

``generate_bouquet`` runs end to end on the mock solver of
``tests/test_draw_rejections.py``; the mock has no flux-surface geometry, so
``archived_pressure_term`` (the p'G of the state ``mygs`` holds) is replaced
by a stand-in returning a known profile.  Synthetic inputs only.
"""
from types import SimpleNamespace

import h5py
import numpy as np

from test_draw_rejections import _X, _run

import bouquet.schema as sc
from bouquet.run import _archive_baseline_split

_P_DD = 3.0e4 * _X ** 2            # the dd source's p'G (reader)
_P_TM = 5.0e4 * _X ** 3            # the archived state's own p'G


def _imas_like_baseline():
    return SimpleNamespace(
        jphi_pd=_P_DD, jphi_pd_source=_P_DD,
        current_split_convention=sc.SPLIT_PRESSURE_IN_INDUCTIVE)


def test_the_legacy_method_hands_no_source_pressure_term_to_the_archive():
    bl = _imas_like_baseline()
    leg = _archive_baseline_split(bl, "legacy")
    assert leg["jphi_pd"] is None          # evaluated on the state
    assert leg["inductive_includes_pressure"] is True
    # the dd's value is kept, under its own name, on the baseline
    np.testing.assert_array_equal(bl.jphi_pd_source, _P_DD)
    # engine and swb hand their solved state's own p'G through
    for m in ("engine", "swb"):
        np.testing.assert_array_equal(
            _archive_baseline_split(bl, m)["jphi_pd"], _P_DD)
    sep = SimpleNamespace(jphi_pd=_P_TM, current_split_convention=sc.
                          SPLIT_PRESSURE_SEPARATE)
    assert _archive_baseline_split(sep, "engine")[
        "inductive_includes_pressure"] is False


def test_the_legacy_baseline_archives_its_own_states_pressure_term(
        tmp_path, monkeypatch):
    import bouquet.TokaMaker_interface as TI
    calls = []

    def _state_term(mygs, psi_N, psi_pad=1e-3, coord="psi_n",
                    what="archive"):
        calls.append(what)
        return _P_TM.copy()
    monkeypatch.setattr(TI, "archived_pressure_term", _state_term)
    split = _archive_baseline_split(_imas_like_baseline(), "legacy")
    out, rej, header = _run(tmp_path, monkeypatch, ["ok"],
                            baseline_split=split)
    assert "baseline archive" in calls
    groups = {}
    with h5py.File(f"{header}.h5", "r") as hf:
        def _grab(name, obj):
            if isinstance(obj, h5py.Group) and "jphi_pd" in obj:
                groups[name] = (obj["jphi_pd"][()],
                                obj["jphi_ind"][()]
                                if "jphi_ind" in obj else None,
                                sc.read_current_split_convention(obj))
        hf.visititems(_grab)
    bl = [k for k in groups if k.endswith("_baseline")]
    assert len(bl) == 1, sorted(groups)
    jp, jind, conv = groups[bl[0]]
    np.testing.assert_array_equal(jp, _P_TM)       # the state's, not the dd's
    assert not np.array_equal(jp, _P_DD)
    assert conv == sc.SPLIT_PRESSURE_SEPARATE
    # the archived inductive excludes the SAME p'G: jphi_total - jphi_bs - P_TM
    Jb = 1.0e5 * (1 - _X)
    jphi = 1.0e6 * (1 - _X ** 2) + Jb
    np.testing.assert_allclose(jind, jphi - Jb - _P_TM, rtol=0, atol=1e-6)
    # (each legacy draw evaluates p'G on its own archived state through the
    # same archived_pressure_term; the mock solver writes no draw eqdsk, so
    # no draw group is archived here)
