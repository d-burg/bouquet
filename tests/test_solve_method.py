"""GenerationConfig.solve_method: the one switch between the legacy, swb and
engine solve methods, its older spellings, and the settings both comparable
methods must share.  Solver-free; no data."""
import pytest

from bouquet.config import (GenerationConfig, SOLVE_METHODS,
                            resolve_solve_method)


@pytest.mark.parametrize("kw, method, imas, eng", [
    ({}, "engine", "closure", "unified"),
    (dict(reconstruction_engine="legacy"), "legacy", "closure", "legacy"),
    (dict(solve_method="legacy"), "legacy", "closure", "legacy"),
    (dict(solve_method="swb"), "swb", "swb", "legacy"),
    (dict(solve_method="engine"), "engine", "closure", "unified"),
    (dict(imas_baseline="swb"), "swb", "swb", "legacy"),
    (dict(reconstruction_engine="unified"), "engine", "closure", "unified"),
    (dict(solve_method="swb", imas_baseline="swb"), "swb", "swb", "legacy"),
])
def test_solve_method_and_its_aliases(kw, method, imas, eng):
    g = GenerationConfig(**kw)
    assert resolve_solve_method(g) == method
    assert (g.imas_baseline, g.reconstruction_engine) == (imas, eng)


@pytest.mark.parametrize("kw", [
    dict(solve_method="engine", imas_baseline="swb"),
    dict(solve_method="legacy", imas_baseline="swb"),
    dict(solve_method="closure"),
])
def test_contradicting_methods_are_refused(kw):
    with pytest.raises(ValueError):
        GenerationConfig(**kw)


def test_switching_method_after_construction():
    g = GenerationConfig(solve_method="swb")
    g.solve_method = "engine"
    assert resolve_solve_method(g) == "engine"
    assert (g.imas_baseline, g.reconstruction_engine) == ("closure", "unified")
    g.solve_method = "swb"
    assert resolve_solve_method(g) == "swb"
    assert (g.imas_baseline, g.reconstruction_engine) == ("swb", "legacy")
    assert set(SOLVE_METHODS) == {"legacy", "swb", "engine"}



# ---------------------------------------------------------------------------
#  review PR69 B4: nothing is written at construction; a switch back restores
# ---------------------------------------------------------------------------
def test_construction_writes_neither_field():
    from bouquet.config import solve_method_of
    g = GenerationConfig(imas_baseline="swb")
    assert (g.imas_baseline, g.reconstruction_engine) == ("swb", "unified")
    assert solve_method_of(g) == "swb"
    g = GenerationConfig(solve_method="swb")
    assert (g.imas_baseline, g.reconstruction_engine) == ("closure", "unified")
    assert solve_method_of(g) == "swb"


def test_switching_imas_baseline_back_restores_the_engine():
    """The review's scenario: an swb config resolved once (it wrote
    reconstruction_engine="legacy"), then imas_baseline set back: the user's
    own reconstruction_engine (the default "unified") comes back."""
    g = GenerationConfig(imas_baseline="swb")
    assert resolve_solve_method(g) == "swb"
    assert (g.imas_baseline, g.reconstruction_engine) == ("swb", "legacy")
    g.imas_baseline = "closure"
    assert resolve_solve_method(g) == "engine"
    assert (g.imas_baseline, g.reconstruction_engine) == ("closure", "unified")
    # an explicit legacy choice is kept
    g = GenerationConfig(imas_baseline="swb", reconstruction_engine="legacy")
    resolve_solve_method(g)
    g.imas_baseline = "closure"
    assert resolve_solve_method(g) == "legacy"


def test_a_contradicting_engine_pair_is_refused_too():
    with pytest.raises(ValueError, match="reconstruction_engine"):
        GenerationConfig(solve_method="engine", reconstruction_engine="legacy")
    g = GenerationConfig(solve_method="swb")
    resolve_solve_method(g)                 # wrote ("swb", "legacy")
    g.solve_method = "engine"               # a switch, not a contradiction
    assert resolve_solve_method(g) == "engine"


def test_construction_checks_judge_the_effective_method():
    """An swb config is validated as the legacy-engine run it is (the
    engine's own refusals do not fire on it), without being rewritten."""
    from bouquet.config import BouquetConfig, ImasSource, SolverConfig
    cfg = BouquetConfig(
        source=ImasSource(ids_path="dd.json", time=1.0),
        solver=SolverConfig(mesh_path="m.h5"), output_header="x",
        generation=GenerationConfig(solve_method="swb",
                                    jbs_self_consistent=False))
    assert cfg.generation.reconstruction_engine == "unified"   # untouched
    with pytest.raises(ValueError):
        BouquetConfig(
            source=ImasSource(ids_path="dd.json", time=1.0),
            solver=SolverConfig(mesh_path="m.h5"), output_header="x",
            generation=GenerationConfig(jbs_self_consistent=False))


def test_the_default_engine_path_is_untouched_when_swb_is_not_selected():
    """With every swb option at its default the engine path is the base
    commit's: the default config resolves to the engine with both fields as
    they were, its draw method is the engine's, and the swb-only fields stay
    out of it (the engine reads bootstrap_kwargs alone)."""
    import dataclasses
    from bouquet.config import solve_method_of
    from bouquet.draw_methods import method_hooks
    from bouquet.engine_draws import GenerateEngineDraws
    g = GenerationConfig()
    before = dataclasses.asdict(g)
    assert solve_method_of(g) == "engine"
    assert resolve_solve_method(g) == "engine"
    assert dataclasses.asdict(g) == before          # nothing rewritten
    assert method_hooks(g) is GenerateEngineDraws
    assert g.swb_edge_taper_psi0 is None and g.swb_ip_tol == 5e-3
    assert g.bootstrap_kwargs == {}
