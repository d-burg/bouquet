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

