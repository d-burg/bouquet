"""The engine's SWB edge taper: physics.edge_taper_weight is OFT's
apply_edge_taper, and engine.compose applies it to every component.
Solver-free."""
import numpy as np
import pytest

from bouquet.engine import compose, composed_factor, conversion_factor
from bouquet.physics import edge_taper_weight


@pytest.mark.parametrize("shape, mid", [(1, 0.5), (2, 0.5), (3, 0.125)])
def test_the_weight_is_ofts_apply_edge_taper(shape, mid):
    psi0 = 0.99
    psi = np.array([0.0, 0.5, psi0, psi0 + 0.005, 1.0])
    w = edge_taper_weight(psi, psi0, shape)
    assert np.array_equal(w[:3], [1.0, 1.0, 1.0])
    assert w[3] == pytest.approx(mid, abs=1e-12)
    assert w[4] == pytest.approx(0.0, abs=1e-15)


def test_no_taper_when_the_span_is_below_ofts_floor():
    psi = np.linspace(0.0, 1.0, 11)
    assert np.array_equal(edge_taper_weight(psi, 1.0 - 1e-7), np.ones(11))


def _geom(n=21, taper=None):
    psi = np.linspace(0.0, 1.0, n)
    g = dict(psi_N=psi, F=np.full(n, 3.4), inv_R=np.full(n, 0.6),
             B2=np.full(n, 4.0), R_avg=np.full(n, 1.7),
             pprime=-1e4 * (1.0 - psi))
    if taper is not None:
        g["edge_taper"] = taper
    return g


def test_compose_tapers_every_component():
    psi = np.linspace(0.0, 1.0, 21)
    w = edge_taper_weight(psi, 0.9, 2)
    jb = (np.ones(21), 0.5 * np.ones(21), 0.2 * np.ones(21))
    J0, p0 = compose(_geom(), *jb)
    J1, p1 = compose(_geom(taper=w), *jb)
    np.testing.assert_allclose(J1, w * J0, rtol=0, atol=1e-12)
    for k in ("ind", "bs", "fix", "pressure"):
        np.testing.assert_allclose(p1[k], w * p0[k], rtol=0, atol=1e-12)
    assert J1[-1] == 0.0
    np.testing.assert_array_equal(composed_factor(_geom(taper=w)),
                                  w * conversion_factor(_geom()))


def test_taper_off_is_the_original_engine():
    """The default (taper off): no weight on the backend, so compose /
    composed_factor are the untapered ones."""
    from bouquet.config import GenerationConfig
    from bouquet.engine import TokaMakerBackend, engine_edge_taper
    from types import SimpleNamespace
    gc = GenerationConfig(reconstruction_engine="unified")
    tap = engine_edge_taper(gc)
    assert tap["on"] is False
    be = TokaMakerBackend(None, SimpleNamespace(psi_N=np.linspace(0, 1, 21),
                                                pressure=np.zeros(21)),
                          edge_taper=tap)
    assert be.edge_taper is None


def test_the_taper_refuses_the_unpinned_edge():
    """Taper on with edge_pprime_pin=False does not converge on the D3D-like
    example (the solve exceeds maxits): refused, as swb refuses it."""
    from bouquet.config import GenerationConfig
    from bouquet.engine import validate_engine_settings
    gc = GenerationConfig(reconstruction_engine="unified",
                          edge_pprime_pin=False,
                          bootstrap_kwargs={"taper_edge_jBS": True})
    with pytest.raises(ValueError, match="edge_pprime_pin=False"):
        validate_engine_settings(gc)
    gc.bootstrap_kwargs = {}
    validate_engine_settings(gc)
