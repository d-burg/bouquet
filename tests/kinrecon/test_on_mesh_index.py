"""The bucket-indexed point-in-triangle test (``diagnostics._on_mesh_compute``) gives exactly the result of the scan of
every triangle (``diagnostics._on_mesh_compute_scan``): on a random Delaunay mesh and, when OpenFUSIONToolkit is
available, on the public D3D-like mesh, for random points, every vertex, edge midpoints, points a hair inside and
outside edges, far-away and non-finite points."""
import os
import sys

import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as D

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_MESH = os.path.join(_REPO, "examples", "D3D-like", "DIIID_mesh.h5")


def _probe_points(xy, tri, rng, n_random=20000, n_tri=None):
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    pts = [lo - 0.1 * span + rng.random((n_random, 2)) * 1.2 * span, xy.copy()]
    sub = tri if n_tri is None else tri[rng.choice(tri.shape[0], n_tri, replace=False)]
    a, b, c = xy[sub[:, 0]], xy[sub[:, 1]], xy[sub[:, 2]]
    for p, q in ((a, b), (b, c), (c, a)):
        mid = 0.5 * (p + q)
        nrm = np.column_stack((-(q - p)[:, 1], (q - p)[:, 0]))
        nrm /= np.maximum(np.hypot(nrm[:, 0], nrm[:, 1]), 1e-300)[:, None]
        pts += [mid, mid + 1e-11 * nrm, mid - 1e-11 * nrm, mid + 1e-9 * nrm, mid - 1e-9 * nrm]
    pts.append((a + b + c) / 3.0)
    pts.append(np.array([[np.nan, 0.0], [np.inf, 1.0], [1e9, -1e9], [-1e9, 1e9]]))
    return np.vstack(pts)


def _check(xy, tri, pts):
    D._MESH_INDEX.clear()
    ref = D._on_mesh_compute_scan(xy, tri, pts)
    new = D._on_mesh_compute(xy, tri, pts)
    assert np.array_equal(ref, new)
    return ref


def test_random_delaunay_mesh():
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(7)
    xy = np.vstack((rng.random((600, 2)) * [2.0, 3.0] + [1.0, -1.5], [[1.0, -1.5], [3.0, -1.5], [1.0, 1.5],
                                                                      [3.0, 1.5]]))
    tri = Delaunay(xy).simplices.astype(np.int64)
    pts = _probe_points(xy, tri, rng, n_random=5000)
    inside = _check(xy, tri, pts)
    assert inside.any() and (~inside).any()


@pytest.mark.skipif(not os.path.isfile(_MESH), reason="needs the public D3D-like mesh")
def test_public_mesh():
    cand = os.environ.get("OFT_PYTHONPATH")
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    meshing = pytest.importorskip("OpenFUSIONToolkit.TokaMaker.meshing")
    pts_mesh, lc, *_ = meshing.load_gs_mesh(_MESH)
    xy = np.asarray(pts_mesh, float)[:, :2]
    tri = np.asarray(lc, np.int64)
    if tri.min() == 1:
        tri = tri - 1
    pts = _probe_points(xy, tri, np.random.default_rng(11), n_random=5000, n_tri=1500)
    inside = _check(xy, tri, pts)
    assert inside.any() and (~inside).any()
