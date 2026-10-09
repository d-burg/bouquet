"""Φ_N archives: profile_coord metadata and the plots that read it (no OFT)."""

import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bouquet.utils import (profile_coord, store_baseline_profiles,
                           store_equilibrium)

GEQDSK = os.path.join(os.path.dirname(__file__), "data", "d3dlike.geqdsk")
N = 21


def _store(path, coord, scan_key="900", n_draws=1, eqdsk=None, **kw):
    stem = os.path.splitext(str(path))[0]
    x = np.linspace(0, 1, N)
    j = 1e6 * (1 - x ** 2)
    one = np.ones(N)
    eq_path = eqdsk or stem + "_in.eqdsk"
    if eqdsk is None:
        with open(eq_path, "wb") as fh:
            fh.write(b"GEQDSK-BYTES")
    with open(eq_path, "rb") as fh:
        eb = fh.read()
    store_baseline_profiles(stem, x, one, one, one, one, one, j,
                            one, one, one, one, one, 1e6, 1.0,
                            scan_key=scan_key, eqdsk_bytes=eb,
                            profile_coord=coord)
    for c in range(n_draws):
        store_equilibrium(stem, c, eq_path, x, j, 0 * j, j,
                          one, one, one, one, one, 1.0, 0.8,
                          scan_key=scan_key, profile_coord=coord, **kw)
    return stem + ".h5"


class TestProfileCoord:
    def test_an_old_archive_is_psi_n(self, tmp_path):
        p = tmp_path / "a.h5"
        with h5py.File(p, "w") as f:
            f.create_group("_baseline")
        assert profile_coord(str(p)) == "psi_n"
        with h5py.File(p, "a") as f:
            f["_baseline"].attrs["profile_coord"] = "phi_n"
        assert profile_coord(str(p)) == "phi_n"

    def test_round_trip_under_scan_key(self, tmp_path):
        p = _store(tmp_path / "a.h5", "phi_n")
        assert profile_coord(p, "900") == "phi_n"

    def test_scan_key_none_falls_back_to_scan_points(self, tmp_path):
        p = _store(tmp_path / "a.h5", "phi_n")
        assert profile_coord(p) == "phi_n"
        assert profile_coord(_store(tmp_path / "b.h5", "psi_n")) == "psi_n"

    def test_mixed_scan_points_raise_without_key(self, tmp_path):
        p = _store(tmp_path / "a.h5", "phi_n", scan_key="1")
        _store(tmp_path / "a.h5", "psi_n", scan_key="2")
        with pytest.raises(ValueError):
            profile_coord(p)
        assert profile_coord(p, "2") == "psi_n"

    def test_draw_group_attr(self, tmp_path):
        p = _store(tmp_path / "a.h5", "phi_n", n_draws=2)
        with h5py.File(p, "r") as hf:
            assert hf["scan/900/0"].attrs["profile_coord"] == "phi_n"
        store_equilibrium(os.path.splitext(p)[0], 1, str(tmp_path / "a_in.eqdsk"),
                          np.linspace(0, 1, N), *([np.ones(N)] * 8), 1.0, 0.8,
                          scan_key="900", profile_coord="psi_n")
        with h5py.File(p, "r") as hf:
            assert hf["scan/900/1"].attrs["profile_coord"] == "psi_n"

    def test_profiles_doc(self, tmp_path):
        import bouquet as bq
        p = _store(tmp_path / "a.h5", "phi_n")
        assert bq.BouquetArchive(p)["900"][0].profiles_doc()["profile_coord"] == "phi_n"


class TestPlots:
    def test_relabel_x(self, tmp_path):
        from bouquet.plotting import _relabel_x
        for coord, want in (("phi_n", r"$\Phi_N$"), ("psi_n", r"$\psi_N$")):
            p = _store(tmp_path / f"{coord}.h5", coord)
            fig, ax = plt.subplots()
            ax.set_xlabel(r"$\psi_N$")
            _relabel_x(fig, p)                      # scan_key=None resolves
            assert ax.get_xlabel() == want
            plt.close(fig)

    def test_flux_functions_on_phi(self, tmp_path):
        from bouquet.io import read_geqdsk
        from bouquet.plotting import _load_flux_functions
        eq = read_geqdsk(GEQDSK)
        p = _store(tmp_path / "a.h5", "phi_n", eqdsk=GEQDSK)
        bl, draws = _load_flux_functions(p, "900")
        np.testing.assert_allclose(bl["x"], np.asarray(eq.rhovn) ** 2)
        np.testing.assert_allclose(draws[0]["x"], np.asarray(eq.rhovn) ** 2)
        p2 = _store(tmp_path / "b.h5", "psi_n", eqdsk=GEQDSK)
        bl2, _ = _load_flux_functions(p2, "900")
        np.testing.assert_allclose(bl2["x"], np.linspace(0, 1, len(bl2["q"])))

    @pytest.mark.parametrize("coord", ["psi_n", "phi_n"])
    def test_plot_jphi_geqdsk_overlay(self, tmp_path, coord):
        from bouquet.io.geqdsk import read_geqdsk
        from bouquet.plotting import plot_jphi
        eq = read_geqdsk(GEQDSK)
        jg = getattr(eq, "j_tor_averaged", None)
        if jg is None:
            jg = eq.j_tor_averaged_direct
        xg = (np.asarray(eq.rhovn) ** 2 if coord == "phi_n"
              else np.asarray(eq.psi_N))
        p = _store(tmp_path / "a.h5", coord)
        fig, ax = plot_jphi(p, source=GEQDSK, source_kind="geqdsk")
        src = [l for l in ax[0].lines if l.get_label().startswith("geqdsk")][0]
        x = np.linspace(0, 1, N)
        np.testing.assert_allclose(src.get_ydata(),
                                   np.interp(x, np.ravel(xg), np.ravel(jg)) / 1e6)
        want = r"$\Phi_N$" if coord == "phi_n" else r"$\psi_N$"
        assert ax[0].get_xlabel() == want
        plt.close(fig)

    def test_timeseries_warns_on_mixed_coords(self, tmp_path):
        from bouquet.plotting import plot_bouquet_timeseries
        a = _store(tmp_path / "a.h5", "phi_n")
        b = _store(tmp_path / "b.h5", "psi_n")
        with pytest.warns(UserWarning, match="profile_coord"):
            fig, _ = plot_bouquet_timeseries({1.0: a, 2.0: b}, scan_key="900")
        plt.close(fig)
        fig, ax = plot_bouquet_timeseries({1.0: a}, scan_key="900")
        assert ax.ravel()[3].get_xlabel() == r"$\Phi_N$"
        plt.close(fig)


def test_merge_refuses_mixed_coords(tmp_path):
    from bouquet.parallel import merge_archives
    a = _store(tmp_path / "w0.h5", "phi_n")
    b = _store(tmp_path / "w1.h5", "psi_n")
    with pytest.raises(RuntimeError, match="profile_coord"):
        merge_archives([a, b], str(tmp_path / "out"), scan_key="900")
    assert not os.path.exists(tmp_path / "out.h5")
