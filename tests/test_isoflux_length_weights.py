"""Isoflux weights per unit length: the constraint strength does not depend on how densely the contour is sampled."""
import numpy as np

from bouquet.utils import isoflux_length_weights


def _ellipse(n, cluster=0.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    t = t + cluster * np.sin(t)                  # non-uniform spacing
    return np.column_stack([1.7 + 0.6 * np.cos(t), 1.2 * np.sin(t)])


def test_strength_is_independent_of_sampling():
    ref = np.sum(isoflux_length_weights(_ellipse(256), 200.0, 256) ** 2)
    for n, c in ((1000, 0.0), (9000, 0.0), (9000, 0.9), (1000, 0.5)):
        w2 = np.sum(isoflux_length_weights(_ellipse(n, c), 200.0, 256) ** 2)
        assert abs(w2 / ref - 1.0) < 1e-12


def test_evenly_spaced_reference_points_get_the_plain_weight():
    t = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    circle = np.column_stack([1.7 + 0.6 * np.cos(t), 0.6 * np.sin(t)])
    np.testing.assert_allclose(isoflux_length_weights(circle, 200.0, 256), 200.0, rtol=1e-12)


def test_clustered_points_share_their_length():
    w = isoflux_length_weights(_ellipse(2000, 0.9), 200.0, 256)
    assert w.max() / w.min() > 3.0


def test_a_zero_length_contour_is_refused_not_given_nan_weights():
    import pytest
    with pytest.raises(ValueError, match="zero length"):
        isoflux_length_weights(np.ones((8, 2)), 200.0, 256)


def test_the_config_default_is_the_tracer_point_count():
    from bouquet.config import GenerationConfig
    from bouquet.TokaMaker_interface import ISO_UPDATE_REF_POINTS
    assert GenerationConfig().iso_update_ref_points == ISO_UPDATE_REF_POINTS
    assert GenerationConfig(iso_update_ref_points=None).iso_update_ref_points is None
    import pytest
    with pytest.raises(ValueError, match="iso_update_ref_points"):
        GenerationConfig(iso_update_ref_points=0)
