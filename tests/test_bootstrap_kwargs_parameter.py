"""Review PR60 B5 (integration hook): the three public TokaMaker_interface
entry points take ``bootstrap_kwargs`` as an explicit parameter instead of
``**kwargs``, so a mistyped keyword raises ``TypeError`` at the call again,
and the options still reach ``solve_with_bootstrap``.  Solver-free."""
import numpy as np
import pytest

import bouquet.TokaMaker_interface as TI
from test_sigma0_identity_stages import toy  # noqa: F401  (fixture)


@pytest.mark.parametrize("fn", [TI.perturb_kinetic_equilibrium,
                                TI.generate_bouquet,
                                TI.reconstruct_equilibrium])
def test_a_mistyped_keyword_raises_at_the_call(fn):
    with pytest.raises(TypeError, match="isolate_edge_jbs|unexpected"):
        fn(None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, isolate_edge_jbs=True)


def test_the_options_reach_every_solve_with_bootstrap_call(toy, monkeypatch):
    """perturb_kinetic_equilibrium forwards bootstrap_kwargs to each of its
    SWB calls (toy GS, recording SWB), and leaves the caller's dict alone."""
    import sys
    import _swb
    from bouquet.sampling import make_rng
    from test_bootstrap_multiplier import _fake_swb, _reconstruct
    from test_sigma0_identity_stages import (_EC, _IP, _N, _NE, _NI, _PAD,
                                             _TE, _TI, _X, _ZEFF, _li)
    calls, seen = [], []
    base = _fake_swb(calls)

    def swb(*a, **kw):
        seen.append(dict(kw))
        return base(*a, **kw)

    _swb.fake_oft(monkeypatch)
    monkeypatch.setattr(sys.modules["OpenFUSIONToolkit.TokaMaker.bootstrap"],
                        "solve_with_bootstrap", swb)
    req, _jbs, _fx = _reconstruct(toy)
    z = np.zeros(_N)
    bk = {"iterations": 3}
    d = TI.perturb_kinetic_equilibrium(
        toy, _X, _EC * (_NE * _TE + _NI * _TI), _NE, _TE, _NI, _TI,
        req, z, z, z, z, z, 0.5, 0.4, 0.25, _IP, _li(toy.achieved), _ZEFF,
        _N, input_jinductive=0.6 * req, l_i_tolerance=0.05, psi_pad=_PAD,
        constrain_sawteeth=False, recalculate_j_BS=True,
        isolate_edge_jBS=False, scale_jBS=1.0, floor_j_BS=False,
        max_proxy_draws=5, p_thresh=0.05, rng=make_rng(7),
        bootstrap_kwargs=bk)[6]
    assert seen and all(k.get("iterations") == 3 for k in seen)
    # review PR64 B1 (integration hook): the draw says how SWB's bootstrap
    # was converted (this fake toolkit takes the grid "x": TokaMaker jphi)
    from bouquet.physics import SWB_JBS_TOROIDAL
    assert d["swb_conversion"]["swb_jbs_convention"] == SWB_JBS_TOROIDAL
    assert bk == {"iterations": 3}


def _cfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g.geqdsk",
                                    profiles_path="p.cdf", time=1.0),
        solver=SolverConfig(mesh_path="m.h5"), output_header="H",
        generation=GenerationConfig(**gen))


@pytest.mark.parametrize("entry", ["prepare_baseline", "generate"])
def test_an_in_place_edit_is_refused_where_the_run_is_resolved(entry):
    """Review PR60 B3/B6 (integration hook): gc.bootstrap_kwargs["k"] = v
    bypasses the reassignment check; prepare_baseline() and generate()
    validate again, before any solve."""
    from bouquet.run import Bouquet
    b = Bouquet.__new__(Bouquet)
    b.config = _cfg(reconstruction_engine="legacy")
    b.baseline, b.mygs = object(), object()
    b.config.generation.bootstrap_kwargs["iteratoins"] = 3   # typo, in place
    with pytest.raises(ValueError, match="iteratoins"):
        getattr(b, entry)()
