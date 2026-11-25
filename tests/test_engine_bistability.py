"""The bistability check (``GenerationConfig.check_bistability``) -- fast,
on the toy stand-in solver of ``tests/_engine_fake_gs.py``.

After the reconstruction, ONE extra solve of the same inputs from a COLD
start is compared with the delivered state (l_i(3) and q0 at the q-row radius
at the sigma=0 gates; beta_pol and the axis reported).  A miss stamps the
baseline ``bistable=True`` with the deltas and warns; the delivered
equilibrium is restored.  ON by default under the unified engine since
2026-10-15 (owner decision): ``None`` resolves per engine at
prepare_baseline() (True under "unified", False -- not applicable, never
refused -- under "legacy"); an explicit value, a stored config's included,
is kept.

Synthetic inputs only; no device data.
"""
import os
import sys
import warnings

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_engine_draws as TD  # noqa: E402

toy_bouquet_solver = TD.toy_bouquet_solver
_quiet = TD._quiet


def _prepared(tmp_path, monkeypatch, *, check, fake_cls=None,
              pressure_scale=None):
    """A toy Bouquet on the unified engine, prepared with
    ``check_bistability=check``; ``fake_cls`` the stand-in solver class;
    ``pressure_scale`` set on the generation config when given."""
    import bouquet.run as br
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.solver_state import enter_bounded_coil_mode
    cls = fake_cls or FakeTokaMaker

    def _setup(self):
        gs = cls(None)
        enter_bounded_coil_mode(gs)
        return self._install_solver(gs)
    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    b = TD._bq(tmp_path)
    b.config.generation.check_bistability = check
    if pressure_scale is not None:
        b.config.generation.pressure_scale = pressure_scale
    b.setup_solver()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b.prepare_baseline()
    return b, rec


def _n(calls, kind):
    return sum(1 for c in calls if c[0] == kind)


def test_on_by_default_under_the_unified_engine(tmp_path,
                                                toy_bouquet_solver,
                                                monkeypatch):
    from bouquet.config import GenerationConfig
    from bouquet.engine import ENGINE_DEPENDENT_DEFAULTS
    assert GenerationConfig().check_bistability is None
    assert ENGINE_DEPENDENT_DEFAULTS["check_bistability"] == {
        "unified": True, "legacy": False}
    b, rec = _prepared(tmp_path, monkeypatch, check=None)
    gc = b.config.generation
    assert gc.check_bistability is True
    assert b.baseline.engine_resolved_defaults["check_bistability"] == {
        "value": True, "origin": "resolved from engine=unified"}
    r = b.baseline.engine["bistability"]
    assert r["checked"] is True and r["extra_solves"] == 1
    assert _n(b.mygs.calls, "init_psi") == 2 and _n(b.mygs.calls, "solve") == 1
    # the resolution goes into the archive's record with the others
    assert b.baseline.engine["engine_resolved_defaults"][
        "check_bistability"]["value"] is True


def test_an_explicit_false_is_kept_noted_and_runs_no_extra_solve(
        tmp_path, toy_bouquet_solver, monkeypatch, capsys):
    b, rec = _prepared(tmp_path, monkeypatch, check=False)
    assert b.config.generation.check_bistability is False
    r = b.baseline.engine_resolved_defaults["check_bistability"]
    assert r == {"value": False, "origin": "explicit",
                 "engine_validated": True}
    # a diagnostic opt-out: recorded, not warned as an unvalidated config
    assert not [w for w in rec if "check_bistability" in str(w.message)]
    assert "bistability" not in b.baseline.engine
    # the reconstruction's own start only; no cold re-solve on the solver
    assert _n(b.mygs.calls, "init_psi") == 1
    assert _n(b.mygs.calls, "solve") == 0
    assert not [w for w in rec if "BISTABLE" in str(w.message)]


def test_a_single_branch_baseline_is_stamped_not_bistable(
        tmp_path, toy_bouquet_solver, monkeypatch):
    # pressure_scale="none": this test is about the bistability stamp, and
    # its deltas are asserted exactly 0.0.  With s != 1 the toy solver's
    # cold re-solve can differ from the warm one in the last bit on some
    # hosts (l_i delta 1.1e-16 seen on one Linux build); "none" keeps the
    # exact bar instead of loosening it.  The scale has its own tests.
    b, rec = _prepared(tmp_path, monkeypatch, check=True,
                       pressure_scale="none")
    r = b.baseline.engine["bistability"]
    assert r["checked"] is True and r["bistable"] is False
    assert r["outcome"] == "single" and r["extra_solves"] == 1
    # exactly ONE extra, cold solve: a second init_psi, then one solve
    assert _n(b.mygs.calls, "init_psi") == 2 and _n(b.mygs.calls, "solve") == 1
    assert r["deltas"]["l_i"] == 0.0 and r["deltas"]["q0"] == 0.0
    g = r["gates"]
    s = b._engine_run["engine"].delivered_meas
    assert g["dl_i"]["setting"] == "jbs_tol_li"
    assert g["dq0"]["setting"] == "jbs_tol_q0"
    assert g["dl_i"]["bound"] == b.config.generation.jbs_tol_li
    assert g["dq0"]["bound"] == b.config.generation.jbs_tol_q0
    assert r["delivered"]["l_i"] == s["li"]
    assert not [w for w in rec if "BISTABLE" in str(w.message)]


def test_a_bistable_baseline_is_stamped_and_warned(tmp_path,
                                                   toy_bouquet_solver,
                                                   monkeypatch):
    """The stand-in whose solve right after init_psi lands 0.5 % off in
    l_i(3) -- a different equilibrium of the same inputs: stamped
    bistable=True with the deltas, warned by name, and the delivered state
    is what the baseline and the solver keep."""
    b, rec = _prepared(tmp_path, monkeypatch, check=True,
                       fake_cls=TD._cold_branch_fake())
    r = b.baseline.engine["bistability"]
    li_del = b._engine_run["engine"].delivered_meas["li"]
    assert r["bistable"] is True and r["outcome"] == "bistable"
    assert r["deltas"]["l_i"] == pytest.approx(0.005 * li_del, rel=1e-9)
    assert r["gates"]["dl_i"]["passed"] is False
    assert r["cold"]["l_i"] == pytest.approx(1.005 * li_del, rel=1e-12)
    msgs = [str(w.message) for w in rec if "BISTABLE" in str(w.message)]
    assert len(msgs) == 1 and "Baseline.engine['bistability']" in msgs[0]
    # the delivered equilibrium is restored: the solver's state is the
    # reconstruction's again, and the baseline is unchanged
    assert b.mygs.toy.state["li"] == li_del
    assert b.baseline.l_i_target == li_del


def test_a_failed_cold_solve_is_recorded_not_hidden(tmp_path,
                                                    toy_bouquet_solver,
                                                    monkeypatch):
    from _engine_fake_gs import FakeTokaMaker

    class Failing(FakeTokaMaker):
        def solve(self, *a, **k):
            raise RuntimeError("synthetic cold-solve failure")
    b, rec = _prepared(tmp_path, monkeypatch, check=True, fake_cls=Failing)
    r = b.baseline.engine["bistability"]
    assert r["bistable"] is None and r["outcome"] == "cold_solve_failed"
    assert "synthetic cold-solve failure" in r["error"]
    assert [w for w in rec if "bistability unknown" in str(w.message)]
    assert b.mygs.toy.state["li"] == b.baseline.l_i_target


def test_the_check_is_engine_only(tmp_path):
    from bouquet.config import GenerationConfig
    from bouquet.engine import validate_engine_settings
    gc = GenerationConfig(reconstruction_engine="legacy",
                          check_bistability=True)
    with pytest.raises(ValueError, match="check_bistability"):
        validate_engine_settings(gc)


def _cfg(engine, value):
    """A from_geqdsk configuration (the synthetic example) on *engine* with
    ``check_bistability=value``."""
    import bouquet as bq
    ex = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
    cfg = bq.Bouquet.from_geqdsk(
        os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1,
        reconstruction_engine=engine).config
    cfg.generation.check_bistability = value
    return cfg


@pytest.mark.parametrize("value", [None, False])
def test_the_legacy_default_is_not_refused(value):
    """Under "legacy" the check does not exist: unset and False (what it
    resolves to there) are its default -- never refused -- and resolution
    stamps False."""
    from bouquet.engine import resolve_engine_defaults, \
        validate_engine_settings
    cfg = _cfg("legacy", value)
    validate_engine_settings(cfg.generation)         # no refusal
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec = resolve_engine_defaults(cfg)
    assert not [x for x in w if "check_bistability" in str(x.message)]
    assert cfg.generation.check_bistability is False
    want = ("resolved from engine=legacy" if value is None else "explicit")
    assert rec["check_bistability"] == {"value": False, "origin": want}
    validate_engine_settings(cfg.generation)         # still no refusal


def test_the_unified_resolution_and_explicit_values():
    from bouquet.config import GenerationConfig
    from bouquet.engine import (bistability_check_on, resolve_engine_defaults,
                                validate_engine_settings)
    for v, want, origin in ((None, True, "resolved from engine=unified"),
                            (True, True, "explicit"),
                            (False, False, "explicit")):
        cfg = _cfg("unified", v)
        assert bistability_check_on(cfg.generation) is want  # None: unified
        rec = resolve_engine_defaults(cfg)
        assert cfg.generation.check_bistability is want
        assert rec["check_bistability"]["origin"] == origin
        assert resolve_engine_defaults(cfg) == rec   # idempotent
    with pytest.raises(ValueError, match="check_bistability"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", check_bistability="yes"))


@pytest.mark.parametrize("stored, want", [(None, True), (True, True),
                                          (False, False)])
def test_a_stored_config_keeps_its_value(stored, want):
    """JSON round trip (an archive's config_json): an explicit stored value
    is kept; an unset one resolves to the engine's."""
    from bouquet.config import BouquetConfig
    from bouquet.engine import resolve_engine_defaults
    cfg = _cfg("unified", stored)
    d = cfg.to_dict()
    assert d["generation"]["check_bistability"] is stored
    back = BouquetConfig.from_json(cfg.to_json())
    assert back.generation.check_bistability is stored
    resolve_engine_defaults(back)
    assert back.generation.check_bistability is want
    # a config stored AFTER resolution carries the resolved value
    again = BouquetConfig.from_json(back.to_json())
    assert again.generation.check_bistability is want
