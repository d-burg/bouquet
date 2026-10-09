"""Engine-dependent defaults resolve when the run starts, not at
construction (2026-10-07).

``isolate_edge_jBS`` and ``perturb_jind_in_anchor`` were validated with
different values on the two reconstruction engines (legacy: the full-profile
decomposition, and diff+C on an IDS source; unified: the former dataclass
defaults, which the engine never reads).  Until 2026-10-07 the factories
applied the values at construction, so ``Bouquet.from_imas(...)`` followed by
``config.generation.reconstruction_engine = "legacy"`` ran the legacy paths
on the ENGINE's values (flat core bootstrap, I_BS/I_p off by tens of
percent).  Now both default to ``None`` and ``Bouquet.prepare_baseline()``
resolves them for the engine configured THEN
(:func:`bouquet.engine.resolve_engine_defaults`), recording the resolution on
the baseline and in the archive.  An explicit value is kept, with a warning
when it contradicts the engine's validated value.

Solver-free (the baseline builders are stubbed; the archive test runs the
engine on the toy solver of ``tests/test_engine_draws.py``).
"""
import contextlib
import io
import json
import os
import warnings

import pytest

import bouquet as bq
from bouquet.config import BouquetConfig, GenerationConfig
from bouquet.engine import (ENGINE_DEPENDENT_DEFAULTS, engine_validated_value,
                            resolve_engine_defaults)

from test_engine_draws import _bq, _quiet, toy_bouquet_solver  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_STORED = os.path.join(_HERE, "data", "stored_configs")

#: what each engine was validated with, per input type
_LEGACY = {"gfile": (False, False), "imas": (False, True)}
_UNIFIED = (True, False)


def _factory(kind, **kw):
    if kind == "imas":
        return bq.Bouquet.from_imas(
            os.path.join(_EX, "D3Dlike_baseline_omas.json"),
            mesh=os.path.join(_EX, "DIIID_mesh.h5"), time=2.3043,
            n_draws=1, **kw)
    return bq.Bouquet.from_geqdsk(
        os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(_EX, "DIIID_mesh.h5"), n_draws=1, **kw)


class _Stop(Exception):
    pass


def _what_the_baseline_builder_sees(b, monkeypatch):
    """Run ``b.prepare_baseline()`` up to the baseline builder of the
    configured engine and return the two settings it was handed."""
    import bouquet.baseline as BL
    import bouquet.engine as BE
    seen = {}

    def _capture(*a, **k):
        g = b.config.generation
        seen["vals"] = (g.isolate_edge_jBS, g.perturb_jind_in_anchor)
        raise _Stop

    monkeypatch.setattr(BL, "resolve_baseline", _capture)
    monkeypatch.setattr(BE, "prepare_engine_baseline", _capture)
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(_Stop):
            b.prepare_baseline()
    return seen["vals"]


def test_every_engine_dependent_field_defaults_to_none():
    g = GenerationConfig()
    assert set(ENGINE_DEPENDENT_DEFAULTS) == {"isolate_edge_jBS",
                                              "perturb_jind_in_anchor"}
    for name in ENGINE_DEPENDENT_DEFAULTS:
        assert getattr(g, name) is None


@pytest.mark.parametrize("kind", ["imas", "gfile"])
def test_a_factory_config_switched_to_legacy_runs_the_legacy_values(
        kind, monkeypatch):
    """The finding: build with the factory (the engine is the default),
    THEN set reconstruction_engine="legacy" -- the legacy baseline builder
    must be handed the values the legacy workflow was validated with, not
    the engine's.  (Before the fix it was handed True / False.)"""
    b = _factory(kind)
    b.config.generation.reconstruction_engine = "legacy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # a resolution is not a warning
        assert _what_the_baseline_builder_sees(b, monkeypatch) == \
            _LEGACY[kind]
    rec = b._engine_resolved_defaults
    for name, val in zip(("isolate_edge_jBS", "perturb_jind_in_anchor"),
                         _LEGACY[kind]):
        assert rec[name] == {"value": val,
                             "origin": "resolved from engine=legacy"}


@pytest.mark.parametrize("kind", ["imas", "gfile"])
def test_the_order_does_not_matter(kind, monkeypatch):
    """Named at construction or set afterwards, the same engine runs the
    same values -- both ways round."""
    a = _factory(kind, reconstruction_engine="legacy")
    assert _what_the_baseline_builder_sees(a, monkeypatch) == _LEGACY[kind]
    c = _factory(kind, reconstruction_engine="legacy")
    c.config.generation.reconstruction_engine = "unified"
    assert _what_the_baseline_builder_sees(c, monkeypatch) == _UNIFIED
    assert _what_the_baseline_builder_sees(_factory(kind), monkeypatch) \
        == _UNIFIED


@pytest.mark.parametrize("kind", ["imas", "gfile"])
def test_a_switch_after_a_previous_resolution_is_re_resolved(kind,
                                                             monkeypatch):
    """prepare_baseline() under one engine, then the engine switched and
    prepare_baseline() again: the values the first call FILLED are
    re-resolved for the new engine, not mistaken for explicit ones."""
    b = _factory(kind)
    assert _what_the_baseline_builder_sees(b, monkeypatch) == _UNIFIED
    b.config.generation.reconstruction_engine = "legacy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _what_the_baseline_builder_sees(b, monkeypatch) == \
            _LEGACY[kind]
    b.config.generation.reconstruction_engine = "unified"
    assert _what_the_baseline_builder_sees(b, monkeypatch) == _UNIFIED


@pytest.mark.parametrize("kind", ["imas", "gfile"])
def test_an_explicit_contradicting_value_is_kept_and_warned(kind,
                                                            monkeypatch):
    """An explicit value the engine was not validated with is KEPT (never
    overridden) and the run warns, naming the field; the record says it
    was explicit and what the engine was validated with."""
    b = _factory(kind)
    g = b.config.generation
    g.reconstruction_engine = "legacy"
    g.isolate_edge_jBS = True                 # an edge-spike study
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seen = _what_the_baseline_builder_sees(b, monkeypatch)
    assert seen == (True, _LEGACY[kind][1])
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    hits = [m for m in msgs if "isolate_edge_jBS=True" in m]
    assert len(hits) == 1
    assert "KEPT" in hits[0] and "reconstruction_engine='legacy'" in hits[0]
    assert not any("perturb_jind_in_anchor" in m for m in msgs)
    rec = b._engine_resolved_defaults
    assert rec["isolate_edge_jBS"] == {"value": True, "origin": "explicit",
                                       "engine_validated": False}
    assert rec["perturb_jind_in_anchor"]["origin"] == \
        "resolved from engine=legacy"


def test_an_explicit_agreeing_value_is_kept_silently(monkeypatch):
    b = _factory("imas")
    g = b.config.generation
    g.reconstruction_engine = "legacy"
    g.perturb_jind_in_anchor = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _what_the_baseline_builder_sees(b, monkeypatch) == \
            (False, True)
    assert b._engine_resolved_defaults["perturb_jind_in_anchor"] == \
        {"value": True, "origin": "explicit"}


def test_the_resolver_itself():
    """resolve_engine_defaults on a bare config, both engines, both kinds."""
    for kind in ("gfile", "imas"):
        for eng, want in (("legacy", _LEGACY[kind]), ("unified", _UNIFIED)):
            cfg = _factory(kind).config
            cfg.generation.reconstruction_engine = eng
            rec = resolve_engine_defaults(cfg)
            g = cfg.generation
            assert (g.isolate_edge_jBS, g.perturb_jind_in_anchor) == want
            assert {k: v["origin"] for k, v in rec.items()} == {
                n: f"resolved from engine={eng}"
                for n in ENGINE_DEPENDENT_DEFAULTS}
            assert resolve_engine_defaults(cfg) == rec     # idempotent
            src = "imas" if kind == "imas" else "reconstruction"
            for n in ENGINE_DEPENDENT_DEFAULTS:
                assert getattr(g, n) == engine_validated_value(n, eng, src)


def _round_trip(cfg):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        back = BouquetConfig.from_json(cfg.to_json())
    return back, [str(x.message) for x in w]


@pytest.mark.parametrize("kind", ["imas", "gfile"])
def test_stored_configs_round_trip_unchanged(kind):
    """Unset stays unset; explicit (or already resolved) values load as
    stored, on both engines, with no warning about them."""
    for eng in ("legacy", "unified"):
        cfg = _factory(kind, reconstruction_engine=eng).config
        back, msgs = _round_trip(cfg)
        assert back.generation.isolate_edge_jBS is None
        assert back.generation.perturb_jind_in_anchor is None
        resolve_engine_defaults(cfg)
        back, msgs = _round_trip(cfg)
        for n in ENGINE_DEPENDENT_DEFAULTS:
            assert getattr(back.generation, n) == getattr(cfg.generation, n)
            assert not any(n in m for m in msgs)
    cfg = _factory(kind, reconstruction_engine="legacy").config
    cfg.generation.isolate_edge_jBS = True            # explicit, legacy
    back, _ = _round_trip(cfg)
    assert back.generation.isolate_edge_jBS is True


@pytest.mark.parametrize("fname", sorted(os.listdir(_STORED)))
def test_the_stored_fixture_configs_load_their_explicit_values(fname):
    """Every config stored before 2026-10-07 carries both fields
    explicitly: they load as stored (a unified one carries the engine's
    own values, which it accepts)."""
    with open(os.path.join(_STORED, fname)) as fh:
        blob = json.load(fh)
    for eng, d in blob.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = BouquetConfig.from_dict(d).generation
        for n in ENGINE_DEPENDENT_DEFAULTS:
            if n in d["generation"]:
                assert getattr(g, n) == d["generation"][n], (eng, n)


def test_the_archive_records_the_resolution(tmp_path, toy_bouquet_solver):
    """A run on the engine (toy solver): the baseline, the engine record and
    the archive's ``_baseline`` attr say what ran and where it came from;
    the archived config carries the resolved values."""
    from bouquet.utils import load_config, load_engine_resolved_defaults
    b = _bq(tmp_path)
    assert b.config.generation.isolate_edge_jBS is None
    b.setup_solver()
    _quiet(b.prepare_baseline)
    want = {"isolate_edge_jBS": {"value": True,
                                 "origin": "resolved from engine=unified"},
            "perturb_jind_in_anchor": {
                "value": False, "origin": "resolved from engine=unified"}}
    assert b.baseline.engine_resolved_defaults == want
    assert b.baseline.engine["engine_resolved_defaults"] == want
    _quiet(b.generate)
    hdr = b.config.output_header
    assert load_engine_resolved_defaults(hdr, 0) == want
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = load_config(hdr, scan_key=0).generation
    assert stored.isolate_edge_jBS is True
    assert stored.perturb_jind_in_anchor is False


def test_an_archive_without_the_record_reads_none(tmp_path):
    import h5py
    from bouquet.utils import load_engine_resolved_defaults
    p = tmp_path / "old.h5"
    with h5py.File(p, "w") as hf:
        hf.require_group("scan/0/_baseline")
    assert load_engine_resolved_defaults(str(p), 0) is None
