"""``GenerationConfig.engine_ids_inductive``: the IDS adapter's inductive
choice, passed through the unified engine -- fast half (no GS solver).

* the default is ``"residual"`` (the inductive current is the parallel
  residual by definition; owner decision 2026-10-02); it reaches the engine
  settings and the record; it is validated by name (one of
  ``adapters.IDS_INDUCTIVE_CHOICES``) and a non-default value is refused
  when set with ``reconstruction_engine="legacy"``;
* ``prepare_engine_baseline`` constructs the IDS adapter with exactly the
  configured choice (the default reproduces the adapter's own default), and
  the contract's provenance says which inductive was used;
* a non-default value set with a g-file source is refused (it would have
  no effect).

Synthetic inputs only (the D3D-like example fixtures); no solver, no device data.
"""
import contextlib
import io
import os
import types
import warnings

import numpy as np
import pytest

import _engine_toy as T
from bouquet.adapters import IDS_INDUCTIVE_CHOICES
from bouquet.config import GenerationConfig
from bouquet.engine import (ENGINE_FIELD_DEFAULTS, engine_settings,
                            reconstruct, validate_engine_settings)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")
_TIME = 2.3043


def _paths_exist():
    return all(os.path.exists(p) for p in (_OMAS, _GEQ, _PF, _MESH))


def _icfg(**gen):
    from bouquet.config import (BouquetConfig, ImasSource, SolverConfig)
    return BouquetConfig(
        source=ImasSource(ids_path=_OMAS, time=_TIME),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


def _gcfg(**gen):
    from bouquet.config import (BouquetConfig, ReconstructionSource,
                                SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=_GEQ, profiles_path=_PF),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


# ---------------------------------------------------------------------------
#  default, settings, record, validation
# ---------------------------------------------------------------------------
def test_the_default_is_residual_and_reaches_the_settings_and_the_record():
    assert GenerationConfig().engine_ids_inductive == "residual"
    assert ENGINE_FIELD_DEFAULTS["engine_ids_inductive"] == "residual"
    assert T.settings()["ids_inductive"] == "residual"
    for v in IDS_INDUCTIVE_CHOICES:
        assert T.settings(engine_ids_inductive=v)["ids_inductive"] == v
    ad = T.ToyAdapter()
    b = T.ToyGS()
    ad.read()
    with contextlib.redirect_stdout(io.StringIO()):
        _eng, _res, rec = reconstruct(
            ad, b, T.settings(engine_ids_inductive="auto",
                              engine_rows=("Ip",)), label="toy")
    assert rec["settings"]["ids_inductive"] == "auto"


@pytest.mark.parametrize("bad", ["ohmic", "", None, 1, True, "RESIDUAL"])
def test_unknown_values_are_refused_by_name(bad):
    with pytest.raises(ValueError, match="engine_ids_inductive"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_ids_inductive=bad))


@pytest.mark.parametrize("v", ["j_ohmic", "auto"])
def test_a_set_value_is_refused_with_the_legacy_engine(v):
    with pytest.raises(ValueError, match="engine_ids_inductive"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", engine_ids_inductive=v))
    # the default passes with legacy
    validate_engine_settings(GenerationConfig(reconstruction_engine="legacy"))


def test_the_settings_dict_carries_every_choice_unchanged():
    for v in IDS_INDUCTIVE_CHOICES:
        s = engine_settings(GenerationConfig(reconstruction_engine="unified",
                                             engine_ids_inductive=v))
        assert s["ids_inductive"] == v


# ---------------------------------------------------------------------------
#  the choice reaches the IDS adapter; refused with a g-file source
# ---------------------------------------------------------------------------
class _Stop(Exception):
    pass


def _prepare_until_the_contract(monkeypatch, cfg):
    """Run ``prepare_engine_baseline`` up to the adapter's contract and
    return ``(inductive passed, contract)``; stops before any solve."""
    import bouquet.adapters as A
    import bouquet.baseline as B
    from bouquet.engine import prepare_engine_baseline
    seen = {}
    real_rb = B.resolve_baseline

    def resolve_baseline(c, _mygs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return real_rb(c, None)

    class Spy(A.IdsAdapter):
        def __init__(self, *a, **k):
            seen["inductive"] = k.get("inductive", "<adapter default>")
            super().__init__(*a, **k)

        def read(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seen["contract"] = super().read()
            raise _Stop()

    monkeypatch.setattr(B, "resolve_baseline", resolve_baseline)
    monkeypatch.setattr(A, "IdsAdapter", Spy)
    bq = types.SimpleNamespace(config=cfg, mygs=object())
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(_Stop):
            prepare_engine_baseline(bq)
    return seen["inductive"], seen["contract"]


@pytest.mark.skipif(not _paths_exist(), reason="synthetic example missing")
@pytest.mark.parametrize("v", IDS_INDUCTIVE_CHOICES)
def test_prepare_hands_the_adapter_the_configured_choice(monkeypatch, v):
    ind, c = _prepare_until_the_contract(monkeypatch,
                                         _icfg(engine_ids_inductive=v))
    assert ind == v
    prov = c.provenance["inductive"]
    if v == "residual":
        assert prov.startswith("residual")
        # the j_ohmic cross-check is stamped, with no threshold
        k = c.provenance["inductive_consistency"]
        assert k["action"] == "residual_by_definition" and k["tol"] is None
        assert np.isfinite(k["net_frac"]) and np.isfinite(k["rms_frac"])
    elif v == "j_ohmic":
        assert prov.startswith("j_ohmic")
    else:
        # the synthetic source is self-consistent: auto keeps j_ohmic
        assert prov.startswith("j_ohmic")
        assert c.provenance["inductive_consistency"]["action"] == \
            "kept_j_ohmic"


@pytest.mark.skipif(not _paths_exist(), reason="synthetic example missing")
def test_the_default_contract_equals_the_adapters_own_default(monkeypatch):
    import bouquet.adapters as A
    from bouquet.baseline import resolve_baseline
    cfg = _icfg()
    ind, c = _prepare_until_the_contract(monkeypatch, cfg)
    assert ind == "residual"
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
        c0 = A.IdsAdapter(cfg.source, cfg, bl).read()
    np.testing.assert_array_equal(c.jB_ind, c0.jB_ind)
    np.testing.assert_array_equal(c.jB_fix, c0.jB_fix)
    assert c.provenance["inductive"] == c0.provenance["inductive"]
    assert c.provenance["inductive"].startswith("residual")


@pytest.mark.skipif(not _paths_exist(), reason="synthetic example missing")
@pytest.mark.parametrize("v", ["j_ohmic", "auto"])
def test_a_set_value_is_refused_with_a_gfile_source(v):
    from bouquet.engine import prepare_engine_baseline
    bq = types.SimpleNamespace(config=_gcfg(engine_ids_inductive=v),
                               mygs=object())
    with pytest.raises(ValueError, match="engine_ids_inductive"):
        prepare_engine_baseline(bq)
