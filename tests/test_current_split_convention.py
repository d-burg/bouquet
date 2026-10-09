"""Where the pressure-driven p'G sits in an archived current split
(schema.CURRENT_SPLIT_CONVENTION_ATTR; owner decision D2, 2026-10-09: its own
``j_pressure`` bucket).  Old archives carry no attr: before PR #64 the
residual ``j_inductive`` carried it; PR #64's evaluator (evaluate_jBS/3) put
it in ``j_BS``.  The IDS exporter branches on this.  Synthetic, solver-free.
"""
import json

import h5py
import numpy as np
import pytest

import bouquet.schema as sc


def test_the_separate_convention_writes_j_pressure_and_the_attr(tmp_path):
    P = np.linspace(1.0, 2.0, 7)
    with h5py.File(tmp_path / "a.h5", "w") as hf:
        g = hf.create_group("d")
        sc.write_current_split(g, P)
        assert g.attrs[sc.CURRENT_SPLIT_CONVENTION_ATTR] == \
            sc.SPLIT_PRESSURE_SEPARATE
        np.testing.assert_array_equal(g["j_pressure"][()], P)
        assert g["j_pressure"].attrs["units"] == "A m^-2"
        assert sc.read_current_split_convention(g) == sc.SPLIT_PRESSURE_SEPARATE
        sc.write_current_split(g, 2 * P)            # replaces
        np.testing.assert_array_equal(g["j_pressure"][()], 2 * P)
        with pytest.raises(ValueError, match="j_pressure"):
            sc.write_current_split(g, None)
        with pytest.raises(ValueError, match="convention"):
            sc.write_current_split(g, P, "somewhere")


def test_old_archives_are_read_by_what_they_carry(tmp_path):
    with h5py.File(tmp_path / "b.h5", "w") as hf:
        pre64 = hf.create_group("pre64")
        assert (sc.read_current_split_convention(pre64)
                == sc.SPLIT_PRESSURE_IN_INDUCTIVE)
        pr64 = hf.create_group("pr64")
        pr64.attrs[sc.JBS_LOOP_JSON_ATTR] = json.dumps(
            {"evaluate_jBS_version": "evaluate_jBS/3 (... plus p'G)"})
        assert (sc.read_current_split_convention(pr64)
                == sc.SPLIT_PRESSURE_IN_BOOTSTRAP)
        ds = hf.create_group("ds")
        ds.create_dataset("j_pressure", data=np.ones(3))
        assert sc.read_current_split_convention(ds) == sc.SPLIT_PRESSURE_SEPARATE
        # the _baseline stamp applies to a draw without its own
        d = hf.create_group("d")
        assert sc.read_current_split_convention(d, {
            sc.CURRENT_SPLIT_CONVENTION_ATTR: b"pressure_in_bootstrap"}) == \
            sc.SPLIT_PRESSURE_IN_BOOTSTRAP
        d.attrs[sc.CURRENT_SPLIT_CONVENTION_ATTR] = "nonsense"
        with pytest.raises(ValueError, match="unknown"):
            sc.read_current_split_convention(d)
