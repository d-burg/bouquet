"""Where the pressure-driven p'G sits in an archived current split
(schema.CURRENT_SPLIT_CONVENTION_ATTR; owner decision D2, 2026-10-09: its own
``jphi_pd`` bucket).  Old archives carry no attr: before PR #64 the
residual ``jphi_ind`` carried it; PR #64's evaluator (evaluate_jbs/3) put
it in ``jphi_bs``.  The IDS exporter branches on this.  Synthetic, solver-free.
"""
import json

import h5py
import numpy as np
import pytest

import bouquet.schema as sc


def test_the_separate_convention_writes_jphi_pd_and_the_attr(tmp_path):
    P = np.linspace(1.0, 2.0, 7)
    with h5py.File(tmp_path / "a.h5", "w") as hf:
        g = hf.create_group("d")
        sc.write_current_split(g, P)
        assert g.attrs[sc.CURRENT_SPLIT_CONVENTION_ATTR] == \
            sc.SPLIT_PRESSURE_SEPARATE
        np.testing.assert_array_equal(g["jphi_pd"][()], P)
        assert g["jphi_pd"].attrs["units"] == "A m^-2"
        assert sc.read_current_split_convention(g) == sc.SPLIT_PRESSURE_SEPARATE
        sc.write_current_split(g, 2 * P)            # replaces
        np.testing.assert_array_equal(g["jphi_pd"][()], 2 * P)
        with pytest.raises(ValueError, match="jphi_pd"):
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
            {"evaluate_jbs_version": "evaluate_jbs/3 (... plus p'G)"})
        assert (sc.read_current_split_convention(pr64)
                == sc.SPLIT_PRESSURE_IN_BOOTSTRAP)
        ds = hf.create_group("ds")
        ds.create_dataset("jphi_pd", data=np.ones(3))
        assert sc.read_current_split_convention(ds) == sc.SPLIT_PRESSURE_SEPARATE
        # the _baseline stamp applies to a draw without its own
        d = hf.create_group("d")
        assert sc.read_current_split_convention(d, {
            sc.CURRENT_SPLIT_CONVENTION_ATTR: b"pressure_in_bootstrap"}) == \
            sc.SPLIT_PRESSURE_IN_BOOTSTRAP
        d.attrs[sc.CURRENT_SPLIT_CONVENTION_ATTR] = "nonsense"
        with pytest.raises(ValueError, match="unknown"):
            sc.read_current_split_convention(d)


@pytest.mark.parametrize(
    "eps", ["r_over_R_geo", "half_width_over_fsa_R", "a_over_R"])
def test_a_v4_record_of_either_eps_is_never_read_as_pressure_in_bootstrap(
        tmp_path, eps):
    """``evaluate_jbs/4`` is ONE convention (p'G separate as jphi_pd, D2)
    whatever its eps (E4/E7): each opt-in tag must classify exactly like
    the default -- separate where the archive carries ``jphi_pd``,
    never as PR #64's in-bootstrap ``/3``."""
    from bouquet.physics import evaluate_jbs_version
    v = evaluate_jbs_version(eps)
    assert v.startswith("evaluate_jbs/4") and "p'G separate as jphi_pd" in v
    with h5py.File(tmp_path / "v4.h5", "w") as hf:
        g = hf.create_group("rec_only")
        g.attrs[sc.JBS_LOOP_JSON_ATTR] = json.dumps({"evaluate_jbs_version": v})
        assert (sc.read_current_split_convention(g)
                != sc.SPLIT_PRESSURE_IN_BOOTSTRAP)
        g2 = hf.create_group("with_jp")
        g2.attrs[sc.JBS_LOOP_JSON_ATTR] = json.dumps({"evaluate_jbs_version": v})
        g2.create_dataset("jphi_pd", data=np.ones(3))
        assert (sc.read_current_split_convention(g2)
                == sc.SPLIT_PRESSURE_SEPARATE)
