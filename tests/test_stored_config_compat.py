"""Configs stored earlier load as they were produced (finding 5 of the
2026-10-04 review).

``to_dict`` writes every field, so a stored config carries the defaults of
the day.  Fixtures: ``tests/data/stored_configs/config_<sha>.json``, the
``to_dict`` output of the SAME synthetic config (legacy and unified) at five
commits of the engine stack, produced by that commit's own code:

* a LEGACY config carrying an engine field's historical default
  (``engine_mse_jacobian="fd_broyden"``, ``engine_ids_inductive="auto"``)
  loads -- it was refused before (the field has no effect there and is
  loaded as today's default, with a warning);
* a UNIFIED config loads with the values it ran with: present fields as
  stored; a field it predates with the value it was produced with where
  that is knowable (``engine_ids_inductive`` -> "auto";
  ``engine_draw_solve_maxits`` -> the ``draw_solve_maxits`` the engine read
  then), with a warning; where it is not knowable, today's default with a
  loud warning naming the field.

Solver-free.
"""
import json
import os
import warnings

import pytest

from bouquet.config import BouquetConfig, GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(_HERE, "data", "stored_configs")
SHAS = ["554edfb", "5f720ef", "ab6b95f", "de9ff62", "d874822"]


def _stored(sha, eng):
    with open(os.path.join(_DIR, f"config_{sha}.json")) as fh:
        return json.load(fh)[eng]


#: stored keys translated on load (checked by their own tests): swb_iterations
#: is retired for bootstrap_kwargs["iterations"]
_TRANSLATED = ("swb_iterations", "bootstrap_kwargs")


def _load(d):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = BouquetConfig.from_dict(d)
    return c.generation, [str(x.message) for x in w]


@pytest.mark.parametrize("sha", SHAS)
def test_every_stored_legacy_config_loads(sha):
    d = _stored(sha, "legacy")
    g, msgs = _load(d)
    assert g.reconstruction_engine == "legacy"
    stored = d["generation"]
    for name, val in (("engine_mse_jacobian", "fd_broyden"),
                      ("engine_ids_inductive", "auto")):
        if stored.get(name) == val:
            # the historical default: loaded as today's, said so
            assert getattr(g, name) == getattr(GenerationConfig(), name)
            assert any(name in m and "no effect" in m for m in msgs)
    # every legacy-path setting it ran with is as stored
    for k, v in stored.items():
        if k.startswith("engine_") or k in ("separatrix_pressure",) \
                or k in _TRANSLATED:
            continue
        if isinstance(v, list):
            continue
        assert getattr(g, k) == v, k


@pytest.mark.parametrize("sha", SHAS)
def test_every_stored_unified_config_loads_with_what_it_ran_with(sha):
    d = _stored(sha, "unified")
    stored = d["generation"]
    g, msgs = _load(d)
    assert g.reconstruction_engine == "unified"
    # present engine fields: exactly as stored (incl. the old MSE default)
    for k, v in stored.items():
        if k.startswith("engine_") and not isinstance(v, list):
            assert getattr(g, k) == v, k
    if "engine_ids_inductive" not in stored:
        assert g.engine_ids_inductive == "auto"
        assert any("engine_ids_inductive" in m and "predates" in m
                   for m in msgs)
    if "engine_draw_solve_maxits" not in stored:
        assert g.engine_draw_solve_maxits == stored["draw_solve_maxits"]
        assert g.draw_solve_maxits is None
        assert any("engine_draw_solve_maxits" in m for m in msgs)
    if sha == "d874822":
        assert msgs == []                     # a current config: silent


def test_a_unified_config_drops_a_non_default_swb_iterations():
    """The unified engine never read it: dropped, said so, not translated
    into an SWB-only key the engine would refuse."""
    d = _stored(SHAS[-1], "unified")
    d["generation"]["swb_iterations"] = 2
    g, msgs = _load(d)
    assert "iterations" not in g.bootstrap_kwargs
    assert any("swb_iterations=2" in m and "dropped" in m for m in msgs)


def test_a_config_with_the_retired_engine_split_pressure_loads():
    d = _stored(SHAS[-1], "unified")
    d["generation"]["engine_split_pressure"] = "inductive"
    g, msgs = _load(d)
    assert not hasattr(g, "engine_split_pressure")
    assert len([m for m in msgs if "engine_split_pressure" in m]) == 1


def test_a_unified_config_that_capped_its_draws_the_old_way():
    """Written before engine_draw_solve_maxits (the engine read
    draw_solve_maxits then): the cap moves over -- it was REFUSED before."""
    d = _stored("554edfb", "unified")
    d["generation"]["draw_solve_maxits"] = 40
    g, msgs = _load(d)
    assert g.engine_draw_solve_maxits == 40 and g.draw_solve_maxits is None


def test_an_unknowable_missing_field_warns_loudly_naming_it():
    d = _stored("d874822", "unified")
    del d["generation"]["engine_mse_jacobian"]
    g, msgs = _load(d)
    assert g.engine_mse_jacobian == GenerationConfig().engine_mse_jacobian
    assert any(m.startswith("STORED UNIFIED CONFIG LACKS "
                            "generation.engine_mse_jacobian") for m in msgs)


def test_a_loop_config_before_the_post_homotopy_field():
    d = _stored("d874822", "legacy")
    del d["generation"]["jbs_max_passes_post_homotopy"]
    g, msgs = _load(d)
    assert g.jbs_max_passes_post_homotopy == 2
    assert any("jbs_max_passes_post_homotopy" in m for m in msgs)


def test_a_live_config_is_still_strict():
    """The tolerance is for STORED configs only: a config built today with a
    non-default engine field under "legacy" is still refused."""
    with pytest.raises(ValueError, match="no effect"):
        BouquetConfig.from_dict(dict(_stored("d874822", "legacy"),
                                     generation=dict(
                                         _stored("d874822", "legacy")[
                                             "generation"],
                                         engine_mse_jacobian="fd_chord",
                                         engine_rows=["Ip"])))
    from bouquet.engine import validate_engine_settings
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", engine_mse_jacobian="fd_broyden"))


@pytest.mark.parametrize("name, val", [("isolate_edge_jBS", False),
                                       ("perturb_jind_in_anchor", True)])
def test_a_stored_unified_factory_config_loads_at_the_default(name, val):
    """Until 2026-10-05 the factories set these legacy-path fields whatever
    the engine; the engine never read them and now refuses a non-default
    value.  A stored unified config carrying one loads at the default (the
    run it recorded is the same), with a warning -- never refused, never
    silently changed."""
    d = _stored(SHAS[-1], "unified")
    d["generation"][name] = val
    g, msgs = _load(d)
    assert g.reconstruction_engine == "unified"
    from bouquet.engine import engine_validated_value
    assert getattr(g, name) == engine_validated_value(name, "unified",
                                                      "reconstruction")
    assert any(name in m and "never read by the unified engine" in m
               for m in msgs)
    # under "legacy" the same stored value is kept (it is read there)
    dl = _stored(SHAS[-1], "legacy")
    dl["generation"][name] = val
    gl, _ = _load(dl)
    assert getattr(gl, name) == val


#: one non-default value per legacy-path field the engine never reads (the
#: same values ``test_engine_refuses_unread_settings`` refuses on a NEW
#: config)
_UNREAD_SET = dict(
    closure_channel="structured", jBS_baseline_mode="ohmic",
    structured_preset="li_soft_onesided", structured_basis="gaussian",
    structured_weights="flat", structured_sigma_ind_up=0.5,
    structured_li_target=0.90, structured_li_sigma=0.05,
    structured_li_kind="li_3", structured_ip_sigma=1e4,
    structured_ip_sigma_frac=0.005, structured_soft=True,
    structured_li_max_corrector_steps=3,
    anchor_pressure_to_equilibrium=True, imas_corrective_jphi=True,
    jbs_loop_q0_corrector=True, floor_j_BS=True,
    accept_anchor_inband=True, diagnostic_plots=True,
    isolate_edge_jBS=False, perturb_jind_in_anchor=True,
    draw_solve_maxits=40, draw_solve_retry_urf=(0.1,),
    draw_solve_loose_tol=2e-5, bootstrap_convergence_override=True)


def test_every_unread_field_has_a_stored_load_case():
    from bouquet.engine import ENGINE_UNREAD_LEGACY_FIELDS
    assert set(_UNREAD_SET) == set(ENGINE_UNREAD_LEGACY_FIELDS)


@pytest.mark.parametrize("name", sorted(_UNREAD_SET))
def test_a_stored_unified_config_with_any_unread_field_loads_at_default(
        name):
    """Finding 1 of the 2026-10-06 review: 3779b51 refused 20 legacy-path
    fields under the engine, and a stored unified config carrying one at a
    non-default value (accepted and IGNORED by the engine when it was
    written) became unloadable.  It loads at the default -- what the run
    actually used -- with a warning naming the field; a NEW config with the
    same value is still refused."""
    d = _stored(SHAS[-1], "unified")
    d["generation"][name] = _UNREAD_SET[name]
    g, msgs = _load(d)
    assert g.reconstruction_engine == "unified"
    from bouquet.engine import (ENGINE_DEPENDENT_DEFAULTS,
                                engine_validated_value)
    # an engine-dependent field (default None since 2026-10-07) loads at
    # the engine's own value -- what the stored run used
    assert getattr(g, name) == (
        engine_validated_value(name, "unified", "reconstruction")
        if name in ENGINE_DEPENDENT_DEFAULTS
        else getattr(GenerationConfig(), name))
    assert any(f"generation.{name}=" in m
               and "never read by the unified engine" in m for m in msgs)
    # every other field is as stored
    for k, v in _stored(SHAS[-1], "unified")["generation"].items():
        if k == name or isinstance(v, list) or k in _TRANSLATED:
            continue
        assert getattr(g, k) == v, k
    # a NEW unified config with the same value is still refused
    from bouquet.engine import validate_engine_settings
    gn = GenerationConfig(reconstruction_engine="unified")
    setattr(gn, name, _UNREAD_SET[name])
    with pytest.raises(ValueError, match="never reads"):
        validate_engine_settings(gn)


def test_a_stored_unified_config_with_unread_homotopy_passes_loads():
    """The same rule for ``homotopy_passes`` stored with
    ``engine_draw_homotopy=False`` (no homotopy ran, the value was ignored)."""
    d = _stored(SHAS[-1], "unified")
    d["generation"]["engine_draw_homotopy"] = False
    d["generation"]["homotopy_passes"] = [[0.05, 0.1]]
    g, msgs = _load(d)
    assert g.engine_draw_homotopy is False
    assert [tuple(p) for p in g.homotopy_passes] == \
        [tuple(p) for p in GenerationConfig().homotopy_passes]
    assert any("homotopy_passes" in m and "never read" in m for m in msgs)
    # with engine_draw_homotopy=True the stored passes are read: kept
    d = _stored(SHAS[-1], "unified")
    d["generation"]["homotopy_passes"] = [[0.05, 0.1]]
    g, _ = _load(d)
    assert [list(p) for p in g.homotopy_passes] == [[0.05, 0.1]]
    from bouquet.engine import validate_engine_settings
    gn = GenerationConfig(reconstruction_engine="unified")
    gn.engine_draw_homotopy = False
    gn.homotopy_passes = [(0.05, 0.1)]
    with pytest.raises(ValueError, match="homotopy_passes"):
        validate_engine_settings(gn)


# ---------------------------------------------------------------------------
#  the default flip (2026-10-06): a config that PREDATES the engine field
# ---------------------------------------------------------------------------
_ENGINE_ERA_KEYS = ("reconstruction_engine",)


def _predating_the_engine(gen):
    """A stored generation dict as written before the engine existed
    (2026-09-29): no reconstruction_engine and no engine_* field."""
    return {k: v for k, v in gen.items()
            if k not in _ENGINE_ERA_KEYS and not k.startswith("engine_")}


@pytest.mark.parametrize("sha", SHAS)
def test_a_stored_config_without_the_engine_field_replays_as_legacy(sha):
    """The default is "unified" since 2026-10-06; a stored config that
    predates the field was produced by the legacy paths and must keep
    replaying on them, with a warning that says how to opt in."""
    d = _stored(sha, "legacy")
    d = dict(d, generation=_predating_the_engine(d["generation"]))
    g, msgs = _load(d)
    assert g.reconstruction_engine == "legacy"
    assert any("predates the unified engine" in m
               and "reconstruction_engine='legacy'" in m for m in msgs)
    # every legacy-path setting it ran with is as stored
    for k, v in d["generation"].items():
        if k in ("jbs_max_passes_post_homotopy",) + _TRANSLATED:
            continue
        got = getattr(g, k)
        if isinstance(v, (list, tuple)) or isinstance(got, (list, tuple)):
            continue
        assert got == v or (v is None and got is None), k


def test_an_archived_config_without_the_engine_field_replays_as_legacy():
    """The same through an archive's own stored config_json (the golden
    fixture's, with the field removed as an archive written before
    2026-09-29 has it)."""
    h5py = pytest.importorskip("h5py")
    path = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
    with h5py.File(path, "r") as hf:
        raw = hf["scan/0/config_json"][()]
    d = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    d["generation"] = _predating_the_engine(d["generation"])
    g, msgs = _load(d)
    assert g.reconstruction_engine == "legacy"
    assert any("predates the unified engine" in m for m in msgs)


def test_a_new_config_defaults_to_the_engine_and_round_trips_it():
    """... while a config built today is "unified" and to_dict() writes the
    field, so it reloads as "unified" with no engine warning."""
    from bouquet.config import ImasSource, SolverConfig
    c = BouquetConfig(source=ImasSource(ids_path="x.json"),
                      solver=SolverConfig(mesh_path="m.h5"),
                      output_header="t")
    assert c.generation.reconstruction_engine == "unified"
    d = c.to_dict()
    assert d["generation"]["reconstruction_engine"] == "unified"
    g, msgs = _load(d)
    assert g.reconstruction_engine == "unified"
    assert not [m for m in msgs if "reconstruction_engine" in m]
