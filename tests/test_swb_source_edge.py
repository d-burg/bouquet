"""swb_taper_source: the read-in SWB currents are zeroed on the LCFS node (the edge
taper's weight is 0 there) and the seed's other negative nodes clipped, so a source
seed negative at the LCFS can still be redrawn."""
import numpy as np
import pytest

from bouquet import coords
from bouquet import swb_draws as SD
from bouquet.config import swb_config_problems
from bouquet.swb import swb_source_edge, swb_jind_sigma

from _swb import swb_cfg


@pytest.fixture(autouse=True)
def _grid_arg(monkeypatch):
    monkeypatch.setattr(coords, "_swb_params", lambda: frozenset({"x", "jphi_fixed", "p_fixed"}))


X = np.r_[np.linspace(0.0, 0.98, 15), 0.99, 1.0]


def _seed():
    s = 1e6 * (1.0 - X ** 1.5) ** 1.5 + 1e4
    s[-1] = -6.6e3                                   # FUSE's inductive current at the LCFS
    return s


def test_the_lcfs_node_is_zeroed_and_the_rest_kept():
    seed, fixed, saw = _seed(), np.full(X.size, -2e3), np.full(X.size, 5e2)
    s, (f, w), rec = swb_source_edge(X, seed, fixed, saw)
    for new, old in ((s, seed), (f, fixed), (w, saw)):
        assert new[-1] == 0.0
        np.testing.assert_array_equal(new[:-1], old[:-1])      # fixed may stay negative
    assert rec == {"swb_source_edge_zeroed": 1, "swb_source_clipped": 0}


def test_an_interior_negative_seed_node_is_clipped():
    seed = _seed()
    seed[8] = -50.0
    with pytest.warns(UserWarning, match="clipped"):
        s, rest, rec = swb_source_edge(X, seed)
    assert rest == [] and s[8] == 0.0 and s.min() == 0.0
    assert rec["swb_source_clipped"] == 1 and rec["swb_source_clip_min"] == -50.0
    np.testing.assert_array_equal(np.delete(s, [8, X.size - 1]), np.delete(seed, [8, X.size - 1]))


def test_the_redraw_refused_on_the_source_seed_runs_on_the_edged_one():
    prof = 1.0 - X ** 2 + 0.1
    jt = np.r_[1.2 * _seed()[:-1], 0.0]              # sigma is 0 at the LCFS (jphi_total 0 there)
    calls = []

    def recipe(kin, j_seed):
        calls.append(j_seed)
        return {"jphi_ind": j_seed, "jphi_bs": 0 * X, "jphi_total": j_seed}

    def draw(seed, rng=0, clip=False):
        return SD.swb_draw(None, X, None, prof, prof, prof, prof, None, None, None,
                           None, swb_jind_sigma(0.1 * np.abs(jt), jt, seed), 0.5, 0.4, 0.25,
                           1.5 + 0 * X, seed, recipe, coord="psi_n", rng=rng, p_thresh=0.05,
                           jind_clip=clip)[6]
    with pytest.raises(SD.SwbSeedRedrawRefused, match="negative on 1"):
        draw(_seed())
    edged = swb_source_edge(X, _seed())[0]
    draw(edged)
    assert len(calls) == 1 and calls[0].min() >= 0.0 and abs(calls[0][-1]) < 1e-4 * calls[0].max()   # GPR jitter at sigma 0
    for r in range(20):                               # clipped: one try, never negative
        d = draw(edged, rng=r, clip=True)
        assert d["jind_resamples"] == 1 and d["jind_clipped"] in (0, 1) and calls[-1].min() >= 0.0
    assert any(c[-1] == 0.0 for c in calls[1:])      # the jitter's negative half clipped at the edge


def test_it_needs_the_edge_taper():
    assert any("swb_taper_source" in m
               for m in swb_config_problems(swb_cfg(swb_taper_source=True, swb_edge_taper_psi0=None)))
    assert not any("swb_taper_source" in m for m in swb_config_problems(
        swb_cfg(swb_taper_source=True, swb_edge_taper_psi0=0.999)))
