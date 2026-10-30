"""The orientation of the exported ``equilibrium.time_slice[].profiles_2d.psi``
(issue #101).

The data dictionary indexes a rectangular 2-D field dim1 first:
``psi[i_R][i_Z]`` under ``grid.dim1 = R``, ``grid.dim2 = Z``.  The g-file
array is ``[Z][R]`` (``PSIRZ = flat.reshape((NH, NW))``), and before 1.4.1
``write_imas_draw`` wrote it unchanged under that grid.  The shipped
fixtures use square grids, so the transpose was invisible; here the g-file is
resampled onto a NON-square 33 (R) x 65 (Z) grid.

* read with the declared dim1/dim2, the exported psi is [R][Z] and its
  extremum (the magnetic axis) lies within one grid cell of the stored axis;
* bouquet's own reader (``_fsa_from_profiles_2d``, which applies ``.T``)
  gets back the g-file array, bit for bit up to the export's COCOS factor:
  export -> re-read is the identity;
* a transposed (pre-1.4.1) array on a non-square grid is refused by the
  reader instead of being traced the wrong way round.
"""
import json
import os

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

import test_imas_export as TE
from bouquet.io.geqdsk import (GEQDSKEquilibrium, _read_geqdsk,
                               _write_geqdsk)
from bouquet.io.imas import _fsa_from_profiles_2d, write_imas_draw

NW, NH = 33, 65                    # R and Z points: deliberately not square

pytestmark = pytest.mark.skipif(not os.path.isfile(TE._GEQ),
                                reason="d3dlike.geqdsk absent")


def _non_square_gfile(path):
    """The shipped D3D-like g-file resampled onto NW x NH (R x Z)."""
    g = _read_geqdsk(TE._GEQ)
    nw0, nh0 = int(g["NW"]), int(g["NH"])
    R0 = g["RLEFT"] + np.linspace(0.0, g["RDIM"], nw0)
    Z0 = g["ZMID"] + np.linspace(-0.5, 0.5, nh0) * g["ZDIM"]
    R = g["RLEFT"] + np.linspace(0.0, g["RDIM"], NW)
    Z = g["ZMID"] + np.linspace(-0.5, 0.5, NH) * g["ZDIM"]
    f = RegularGridInterpolator((Z0, R0), g["PSIRZ"])
    ZZ, RR = np.meshgrid(Z, R, indexing="ij")
    g["PSIRZ"] = f(np.stack([ZZ, RR], axis=-1))          # [Z][R], (NH, NW)
    x0, x = np.linspace(0, 1, nw0), np.linspace(0, 1, NW)
    for k in ("FPOL", "PRES", "FFPRIM", "PPRIME", "QPSI"):
        g[k] = np.interp(x, x0, g[k])
    g["NW"], g["NH"] = NW, NH
    _write_geqdsk(g, path)
    return path


def _export(tmp_path, monkeypatch):
    geq_path = _non_square_gfile(str(tmp_path / "g_nonsquare.geqdsk"))
    monkeypatch.setattr(TE, "_GEQ", geq_path)
    arc = str(tmp_path / "run.h5")
    TE._make_archive(arc, with_fsa=True)
    tmpl = str(tmp_path / "tmpl.json")
    TE._make_template(tmpl)
    out = str(tmp_path / "draw.json")
    write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")
    ts = json.load(open(out))["equilibrium"]["time_slice"][0]
    return ts, GEQDSKEquilibrium(geq_path, cocos=1)


def test_the_fixture_is_not_square(tmp_path):
    geq = GEQDSKEquilibrium(_non_square_gfile(str(tmp_path / "g.geqdsk")))
    assert np.shape(geq.psi_RZ) == (NH, NW) and NH != NW


def test_the_exported_psi_is_dim1_first_and_the_axis_is_where_it_says(
        tmp_path, monkeypatch):
    ts, geq = _export(tmp_path, monkeypatch)
    p2 = ts["profiles_2d"][0]
    R = np.asarray(p2["grid"]["dim1"], dtype=float)
    Z = np.asarray(p2["grid"]["dim2"], dtype=float)
    psi = np.asarray(p2["psi"], dtype=float)
    assert R.size == NW and Z.size == NH
    assert psi.shape == (R.size, Z.size)               # [i_R][i_Z]
    # the magnetic axis: the grid point where psi is nearest its axis value
    gq = ts["global_quantities"]
    i, j = np.unravel_index(
        np.argmin(np.abs(psi - float(gq["psi_axis"]))), psi.shape)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    assert abs(R[i] - float(gq["magnetic_axis"]["r"])) <= dR
    assert abs(Z[j] - float(gq["magnetic_axis"]["z"])) <= dZ
    assert abs(R[i] - geq.R_mag) <= dR and abs(Z[j] - geq.Z_mag) <= dZ


def test_export_then_re_read_is_the_identity(tmp_path, monkeypatch):
    """bouquet's reader rebuilds the g-file array from the export: the same
    numbers, [Z][R], up to the export's sign / 2 pi factor (one multiply
    there, one sign here, so exact)."""
    import bouquet.io.geqdsk as G
    ts, geq = _export(tmp_path, monkeypatch)
    seen = {}
    real = G.GEQDSKEquilibrium.from_raw.__func__

    def spy(cls, raw, *a, **k):
        seen["PSIRZ"] = np.asarray(raw["PSIRZ"], dtype=float)
        return real(cls, raw, *a, **k)
    monkeypatch.setattr(G.GEQDSKEquilibrium, "from_raw", classmethod(spy))
    _fsa_from_profiles_2d(ts)
    back = seen["PSIRZ"]
    assert back.shape == (NH, NW)
    exported = np.asarray(ts["profiles_2d"][0]["psi"], dtype=float)
    np.testing.assert_array_equal(np.abs(back), np.abs(exported.T))
    # and the exported values are the g-file's times one common factor
    ratio = exported.T / np.asarray(geq.psi_RZ, dtype=float)
    ok = np.isfinite(ratio) & (np.abs(np.asarray(geq.psi_RZ)) > 1e-12)
    np.testing.assert_allclose(ratio[ok], ratio[ok][0], rtol=1e-12)
    assert abs(abs(ratio[ok][0]) - 2.0 * np.pi) < 1e-12


def test_a_transposed_array_on_a_non_square_grid_is_refused(
        tmp_path, monkeypatch):
    ts, _ = _export(tmp_path, monkeypatch)
    p2 = ts["profiles_2d"][0]
    p2["psi"] = np.asarray(p2["psi"], dtype=float).T.tolist()   # [Z][R]
    with pytest.raises(ValueError, match="dim1"):
        _fsa_from_profiles_2d(ts)
