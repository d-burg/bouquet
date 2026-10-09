"""bouquet.experimental: the registry of opt-in, unvalidated features.

Owner decision 2026-10-09: the PR #56 kinetic combination, the swb solve
method and the PR #60 convergence override are EXPERIMENTAL -- reachable only
by an explicit option, warned at prepare_baseline(), stamped on the baseline
record and the archive, and listed in the docs.  Solver-free.
"""
import os
import warnings

import h5py
import numpy as np
import pytest

from bouquet import experimental as X
from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                            ReconstructionSource, SolverConfig,
                            UncertaintyConfig)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = os.path.join(os.path.dirname(HERE), "docs", "workflows.md")


def _recon(tmp_path, profiles="x.cdf", **kw):
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=str(tmp_path / "g"),
                                    profiles_path=str(tmp_path / profiles),
                                    **kw),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out"))


def _imas(tmp_path, gen=None, unc=None, **kw):
    return BouquetConfig(
        source=ImasSource(ids_path=str(tmp_path / "dd.json"), **kw),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out"),
        generation=gen or GenerationConfig(),
        uncertainty=unc or UncertaintyConfig())


# ---------------------------------------------------------------------------
#  the registry itself
# ---------------------------------------------------------------------------
def test_every_record_is_complete():
    want = {"description", "enabled_by", "introduced", "validation_todo",
            "status"}
    for k, r in X.REGISTRY.items():
        assert set(r) == want, k
        assert r["status"] == "experimental", k
        assert isinstance(r["validation_todo"], list) and r["validation_todo"], k
        assert all(isinstance(t, str) and t for t in r["validation_todo"]), k
        assert r["enabled_by"] and r["description"] and r["introduced"], k


def test_the_registered_features():
    assert list(X.REGISTRY) == [
        "ida_ion_route", "fuse_zeff_fast_ions", "ida_ni_beam_subtraction",
        "kinetic_sampler_clips", "swb_solve_method",
        "bootstrap_convergence_override"]
    # the geometric eps default (PR #60 E4) is a declared physics change,
    # not an experimental feature
    assert not [k for k in X.REGISTRY if "eps" in k]


def test_docs_and_registry_agree():
    """docs/workflows.md carries exactly the block generated from the
    registry (so the keys, todos and switches cannot drift apart)."""
    text = open(WORKFLOWS, encoding="utf-8").read()
    assert "## Experimental features and their validation status" in text
    i, j = text.index(X.DOCS_BEGIN), text.index(X.DOCS_END)
    assert text[i:j + len(X.DOCS_END)] == X.docs_markdown()
    keys = {line.split("`")[1] for line in text[i:j].splitlines()
            if line.startswith("#### `")}
    assert keys == set(X.REGISTRY)


# ---------------------------------------------------------------------------
#  which configurations enable what
# ---------------------------------------------------------------------------
def test_defaults_enable_nothing(tmp_path):
    assert X.experimental_features_enabled(_recon(tmp_path)) == []
    assert X.experimental_features_enabled(_imas(tmp_path)) == []
    assert X.experimental_features_enabled(
        _imas(tmp_path, ida_path=str(tmp_path / "i.cdf"))) == []


@pytest.mark.parametrize("route", ["Zeff", "CER", "all"])
def test_an_ida_route_enables_the_route_and_the_clips(tmp_path, route):
    on = X.experimental_features_enabled(_recon(tmp_path, ni_source=route))
    assert on == ["ida_ion_route", "kinetic_sampler_clips"]


def test_an_ida_route_without_an_ida_file_is_inert(tmp_path):
    cfg = _recon(tmp_path, profiles="p.peqdsk", ni_source="all")
    assert X.experimental_features_enabled(cfg) == []
    cfg.uncertainty.ida_path = str(tmp_path / "sig.cdf")   # now an IDA is in play
    assert "ida_ion_route" in X.experimental_features_enabled(cfg)


def test_each_imas_switch(tmp_path):
    cdf = str(tmp_path / "i.cdf")
    assert X.experimental_features_enabled(
        _imas(tmp_path, zeff_fast_ions=True)) == [
        "fuse_zeff_fast_ions", "kinetic_sampler_clips"]
    assert X.experimental_features_enabled(
        _imas(tmp_path, ida_path=cdf, ni_source="all",
              ni_subtract_fast=True)) == [
        "ida_ion_route", "ida_ni_beam_subtraction", "kinetic_sampler_clips"]


def test_explicit_clips_win_over_auto(tmp_path):
    assert X.experimental_features_enabled(_imas(
        tmp_path, unc=UncertaintyConfig(kinetic_clips=True))) == [
        "kinetic_sampler_clips"]
    on = X.experimental_features_enabled(_imas(
        tmp_path, unc=UncertaintyConfig(kinetic_clips=False),
        zeff_fast_ions=True))
    assert on == ["fuse_zeff_fast_ions"]


def test_swb_and_the_convergence_override(tmp_path):
    assert X.experimental_features_enabled(_imas(
        tmp_path, gen=GenerationConfig(solve_method="swb"))) == [
        "swb_solve_method"]
    assert X.experimental_features_enabled(_imas(
        tmp_path, gen=GenerationConfig(imas_baseline="swb"))) == [
        "swb_solve_method"]
    assert X.experimental_features_enabled(_imas(
        tmp_path, gen=GenerationConfig(
            reconstruction_engine="legacy",          # refused under the engine
            bootstrap_convergence_override=True))) == [
        "bootstrap_convergence_override"]


def test_ni_source_is_validated(tmp_path):
    with pytest.raises(ValueError, match="ni_source"):
        _recon(tmp_path, ni_source="mean")
    with pytest.raises(ValueError, match="ni_source"):
        ImasSource(ids_path="x.json", ni_source="VB+CER")


# ---------------------------------------------------------------------------
#  warning, baseline record, archive stamp
# ---------------------------------------------------------------------------
def test_the_warning_names_the_feature_and_its_todo(tmp_path):
    cfg = _recon(tmp_path, ni_source="all")
    with pytest.warns(X.ExperimentalFeatureWarning) as rec:
        on = X.warn_experimental_features(cfg)
    assert on == ["ida_ion_route", "kinetic_sampler_clips"]
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, X.ExperimentalFeatureWarning)]
    assert len(msgs) == 2
    for k, m in zip(on, msgs):
        assert repr(k) in m
        assert X.REGISTRY[k]["validation_todo"][0] in m


def test_no_warning_on_defaults(tmp_path, recwarn):
    assert X.warn_experimental_features(_recon(tmp_path)) == []
    assert not [w for w in recwarn.list
                if issubclass(w.category, X.ExperimentalFeatureWarning)]


class _StubBaseline:
    experimental_features = None
    engine_resolved_defaults = None

    def __init__(self):
        self.engine = {}


@pytest.mark.parametrize("opt_in", [False, True])
def test_prepare_baseline_warns_and_records(tmp_path, monkeypatch, opt_in):
    """The unified-engine prepare_baseline() route: one warning per
    feature, the list on the baseline record and in its engine record."""
    import bouquet.engine as E
    from bouquet.run import Bouquet
    cfg = _recon(tmp_path, ni_source="all" if opt_in else "standard")
    stub = _StubBaseline()
    monkeypatch.setattr(E, "prepare_engine_baseline", lambda run: stub)
    b = Bouquet(cfg)
    monkeypatch.setattr(b, "_report_sigma_exceeds_profile", lambda bl: None)
    monkeypatch.setattr(b, "_remember_baseline_state", lambda: None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert b.prepare_baseline() is stub
    got = [str(w.message) for w in rec
           if issubclass(w.category, X.ExperimentalFeatureWarning)]
    want = ["ida_ion_route", "kinetic_sampler_clips"] if opt_in else []
    assert len(got) == len(want)
    assert stub.experimental_features == want
    assert stub.engine["experimental_features"] == want


def test_the_archive_stamp_round_trips(tmp_path):
    from bouquet.utils import (load_experimental_features,
                               stamp_experimental_features)
    p = str(tmp_path / "a.h5")
    with h5py.File(p, "w") as f:
        f.create_group("_baseline")
    assert load_experimental_features(p) is None          # predates the record
    stamp_experimental_features(p, features=[])
    assert load_experimental_features(p) == []
    stamp_experimental_features(p, features=["swb_solve_method"])
    assert load_experimental_features(p) == ["swb_solve_method"]


# ---------------------------------------------------------------------------
#  end to end on the engine (toy solver): the archive carries the list
# ---------------------------------------------------------------------------
from test_engine_draws import _bq, _quiet, toy_bouquet_solver  # noqa: E402,F401


@pytest.mark.parametrize("clips", [None, True])
def test_the_archive_records_the_enabled_features(tmp_path, toy_bouquet_solver,
                                                  clips):
    from bouquet.utils import load_experimental_features
    b = _bq(tmp_path)
    b.config.uncertainty.kinetic_clips = clips
    b.setup_solver()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b.prepare_baseline()
    want = ["kinetic_sampler_clips"] if clips else []
    assert b.baseline.experimental_features == want
    assert b.baseline.engine["experimental_features"] == want
    assert len([w for w in rec if issubclass(
        w.category, X.ExperimentalFeatureWarning)]) == len(want)
    _quiet(b.generate)
    assert load_experimental_features(b.config.output_header, 0) == want


# ---------------------------------------------------------------------------
#  the readers surface the list (stats.draw_band, archive views)
# ---------------------------------------------------------------------------
from test_stats_draw_band import _by_attr, _simple  # noqa: E402


@pytest.mark.parametrize("stamp, want", [
    (None, "unrecorded"), ([], []), (["swb_solve_method"], ["swb_solve_method"])])
def test_draw_band_carries_the_features(tmp_path, stamp, want):
    import json
    import bouquet as bq
    attrs = ({} if stamp is None else
             {"experimental_features_json": json.dumps(stamp)})
    p = _simple(tmp_path / "a.h5", 6, baseline_attrs=attrs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.provenance["experimental_features"] == want
    xw = [w for w in rec if issubclass(w.category, X.ExperimentalFeatureWarning)]
    lim = [m for m in r.provenance["limitations"] if "EXPERIMENTAL" in m]
    if want and want != "unrecorded":
        assert len(xw) == 1 and "swb_solve_method" in str(xw[0].message)
        assert len(lim) == 1 and "swb_solve_method" in lim[0]
        assert "EXPERIMENTAL[swb_solve_method]" in repr(r)
    else:
        assert not xw and not lim and "EXPERIMENTAL" not in repr(r)


def test_archive_views_show_the_features(tmp_path, capsys):
    import json
    from bouquet.archive import BouquetArchive
    p = _simple(tmp_path / "a.h5", 3, baseline_attrs={
        "experimental_features_json": json.dumps(["ida_ion_route"])})
    ar = BouquetArchive(p)
    sv = ar.scan("1")
    assert sv.experimental_features == ["ida_ion_route"]
    assert "EXPERIMENTAL=['ida_ion_route']" in repr(sv)
    assert "ida_ion_route" in repr(ar)
    p0 = _simple(tmp_path / "b.h5", 3)
    sv0 = BouquetArchive(p0).scan("1")
    assert sv0.experimental_features is None
    assert "EXPERIMENTAL" not in repr(sv0)
