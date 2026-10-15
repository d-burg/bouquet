"""The v4 names (OpenFUSIONToolkit boot_profs: jphi_total, jphi_bs, scale_jbs, ...): the old spellings still work.

Old keywords, attributes and function names warn and map; stored configs and pre-v4 archives load.
"""
import h5py
import numpy as np
import pytest

from bouquet import schema
from bouquet._renamed import accepts_old_names
from bouquet.baseline import Baseline
from bouquet.config import FixedComponentsConfig, GenerationConfig, _decode_source, _renamed_fields


def test_an_old_keyword_maps_with_a_warning_and_both_are_refused():
    @accepts_old_names
    def f(jphi_total, scale_jbs=1.0):
        return jphi_total, scale_jbs
    with pytest.warns(DeprecationWarning, match="'scale_jBS' is renamed 'scale_jbs'"):
        assert f(1.0, scale_jBS=2.0) == (1.0, 2.0)
    with pytest.raises(TypeError, match="both"):
        f(1.0, scale_jBS=2.0, scale_jbs=3.0)


def test_old_attribute_names_read_and_write_the_new_fields():
    x = np.linspace(0.0, 1.0, 5)
    bl = Baseline(psi_N=x, jphi_total=x, jphi_ind=x, jphi_bs=x, psi_N_kinetic=x, ne=x, te=x, ni=x, ti=x,
                  Zeff=x, Ip_target=1.0, l_i_target=1.0, provenance="test")
    with pytest.warns(DeprecationWarning):
        assert bl.j_BS is bl.jphi_bs
    g = GenerationConfig()
    with pytest.warns(DeprecationWarning):
        g.jBS_scale_range = (0.9, 1.1)
    assert g.jbs_scale_range == (0.9, 1.1)


def test_evaluate_jBS_is_a_deprecated_alias():
    from bouquet.physics import evaluate_jBS, evaluate_jbs
    assert evaluate_jBS.__wrapped__ is evaluate_jbs


def test_a_stored_config_loads_the_old_field_names():
    fx = _renamed_fields({"j_NBI": [1.0], "psi_N": [0.0]}, "fixed_components")
    assert FixedComponentsConfig(**fx).jphi_nbi == [1.0]
    src = _decode_source({"source_type": "reconstruction", "geqdsk_path": "g", "profiles_path": "p",
                          "rescale_j_BS": True})
    assert src.rescale_jbs is True


def test_a_pre_v4_archive_reads_and_migrates(tmp_path):
    p = str(tmp_path / "v3.h5")
    with h5py.File(p, "w") as hf:
        g = hf.create_group("scan/0/0")
        g.create_dataset("j_BS", data=np.arange(3.0))
        g.create_group("jB_parallel").create_dataset("jB_NBI", data=np.ones(2))
    with h5py.File(p, "r") as hf:
        g = hf["scan/0/0"]
        np.testing.assert_array_equal(schema.read_profile(g, "jphi_bs"), np.arange(3.0))
        assert set(schema.read_jdotB(g)) == {"jdotB_nbi"}
    with h5py.File(p, "r+") as hf:
        assert schema.migrate_names(hf) == 3
        assert "jphi_bs" in hf["scan/0/0"] and "jdotB/jdotB_nbi" in hf["scan/0/0"]
