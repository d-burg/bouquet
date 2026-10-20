"""Per-equilibrium chi_perp / tau_E (bouquet.transport) and FUSE's summed
sources (io.imas.dd_source_totals) with their archive group."""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.io.imas import dd_source_totals
from bouquet.transport import E_CHARGE, transport_record

RHO = np.linspace(0.0, 1.0, 41)


def _entry(index, name, t, rho=RHO, **fields):
    pr = {"time": t, "grid": {"rho_tor_norm": list(rho),
                              "volume": list(2.0 * np.asarray(rho) ** 2)}}
    for k, v in fields.items():
        part, key = k.split("__") if "__" in k else (None, k)
        (pr.setdefault(part, {}) if part else pr)[key] = list(v)
    return {"identifier": {"index": index, "name": name}, "profiles_1d": [pr]}


def test_sources_are_summed_as_imas_total_sources():
    rho_own = np.linspace(0.0, 1.0, 11)
    src = {"time": [1.0], "source": [
        _entry(1, "total", 1.0, electrons__power_inside=9e9 * RHO),       # skipped: index 1
        _entry(2, "nbi", 1.0, rho=rho_own, electrons__power_inside=1e6 * rho_own,
               total_ion_power_inside=2e6 * rho_own,
               electrons__particles_inside=1e20 * rho_own),
        _entry(8, "brem", 1.0, electrons__power_inside=-1e5 * RHO),
        _entry(3, "ec", 1.0, electrons__energy=np.full(RHO.size, 5e5)),  # only the density
    ]}
    d = dd_source_totals(src, 0, 1, 1.0, RHO)
    assert [s["index"] for s in d["sources"]] == [2, 8, 3]
    nbi, brem, ec = d["sources"]
    assert (nbi["P_e"], nbi["P_i"]) == pytest.approx((1e6, 2e6))
    assert brem["P_e"] == pytest.approx(-1e5) and "P_i" not in brem
    assert ec["P_e"] == pytest.approx(1e6)
    np.testing.assert_allclose(d["Q_i"], 2e6 * RHO)
    np.testing.assert_allclose(d["Gamma_e"], 1e20 * RHO)
    # ec: cumtrapz(volume, energy) = 5e5 * 2 rho^2
    np.testing.assert_allclose(d["Q_e"], 1e6 * RHO - 1e5 * RHO + 1e6 * RHO ** 2, rtol=1e-12)
    assert d["P_net"] == pytest.approx(1e6 - 1e5 + 1e6 + 2e6)
    assert d["P_rad"] == pytest.approx(1e5)


def _cylinder(a=0.5, R0=1.7):
    x = np.linspace(0.0, 1.0, 201)                 # psi_N = phi_N = rho_N^2
    r = a * np.sqrt(x)
    return dict(psi_N=x, phi_N=x, rho_tor=r, V=2 * np.pi * R0 * np.pi * r ** 2,
                S=2 * np.pi * R0 * 2 * np.pi * r, R_min=R0 - r, R_max=R0 + r,
                a=a, R_geo=R0, kappa=1.0)


def test_chi_perp_matches_the_analytic_cylinder():
    """n flat, T = T0 (1 - rho^2), Q = Q_a rho^2, no particle flux:
    chi = Q_a / (8 pi^2 R0 n e T0) on every surface."""
    a, R0, n, T0, Qa = 0.5, 1.7, 5e19, 2e3, 3e6
    geom = _cylinder(a, R0)
    x = np.linspace(0.0, 1.0, 101)                 # kinetic nodes in psi_N
    rho = np.sqrt(x)
    src = dict(rho_tor_norm=rho, Q_e=Qa * rho ** 2, Q_i=Qa * rho ** 2,
               Gamma_e=np.zeros_like(rho), P_net=2 * Qa, b0=2.0, r0=R0)
    T = T0 * (1 - rho ** 2) + 1.0
    ne = np.full_like(rho, n)
    rec = transport_record(geom, x, ne, T, 1 + 0 * ne, T, x, np.full_like(x, 1e4), src, 1e6)
    want = Qa / (8 * np.pi ** 2 * R0 * n * E_CHARGE * T0)
    mid = slice(20, 80)
    np.testing.assert_allclose(rec["chi_perp_e"][mid], want, rtol=2e-2)
    np.testing.assert_allclose(rec["chi_perp_i"][mid], want, rtol=2e-2)
    assert rec["W_th"] == pytest.approx(1.5 * 1e4 * geom["V"][-1], rel=1e-3)
    assert rec["tau_E_th"] == pytest.approx(rec["W_th"] / (2 * Qa))
    assert rec["tau_E"] == rec["tau_E_th"]
    assert rec["H_98"] == pytest.approx(rec["tau_E_th"] / rec["tau_98"])
    rec0 = transport_record(geom, x, ne, T, 1 + 0 * ne, T, x, np.full_like(x, 1e4),
                            dict(src, P_net=0.0), 1e6)
    assert np.isinf(rec0["tau_E_th"]) and np.isinf(rec0["tau_98"])



def test_chi_perp_i_takes_the_main_ions_as_imas_jl():
    """n_i = n_e/Z_eff and Gamma_i = Gamma_e/Z_eff (IMAS.jl 7.4 calculate_diffusivities)."""
    geom, x = _cylinder(), np.linspace(0.0, 1.0, 101)
    rho = np.sqrt(x)
    T, ne, z = 2e3 * (1 - rho ** 2) + 1.0, np.full_like(rho, 5e19), np.full_like(rho, 2.0)
    src = dict(rho_tor_norm=rho, Q_e=3e6 * rho ** 2, Q_i=3e6 * rho ** 2,
               Gamma_e=1e21 * rho ** 2, P_net=6e6, b0=2.0, r0=1.7)
    one, two = (transport_record(geom, x, ne, T, zz, T, x, np.full_like(x, 1e4), src, 1e6)
                for zz in (np.ones_like(z), z))
    np.testing.assert_array_equal(two["chi_perp_e"], one["chi_perp_e"])
    num = lambda G: src["Q_i"] - 2.5 * E_CHARGE * T * G       # noqa: E731
    mid = slice(20, 80)
    np.testing.assert_allclose(two["chi_perp_i"][mid] / one["chi_perp_i"][mid],
                               (2.0 * num(src["Gamma_e"] / 2.0) / num(src["Gamma_e"]))[mid],
                               rtol=1e-12)

def test_the_archive_groups_round_trip(tmp_path):
    from bouquet.utils import (load_baseline_profiles, load_dd_sources,
                               store_dd_sources, store_transport)
    h5 = str(tmp_path / "a.h5")
    with h5py.File(h5, "w") as hf:
        hf.create_group("scan/1500/_baseline").create_dataset("n_e", data=np.ones(3))
        hf.create_group("scan/1500/0")
    src = dict(rho_tor_norm=RHO, Q_e=RHO, Q_i=RHO, Gamma_e=RHO, P_net=2.0,
               P_rad=0.1, b0=2.0, r0=1.7, core_profiles_time=1.5,
               sources=[{"index": 2, "name": "nbi", "time": 1.5}], skipped=[],
               rule="r")
    store_dd_sources(h5, 1500, src)
    rec = dict(chi_perp_e=RHO, chi_perp_i=2 * RHO, tau_E_th=0.1, tau_E=0.12,
               tau_98=0.1, H_98=1.0, tau_DS03=0.09, W_th=1e5, W_fast=2e4, P_net=1e6)
    store_transport(h5, 1500, None, rec)
    store_transport(h5, 1500, 0, rec)
    d = load_dd_sources(h5, 1500)
    np.testing.assert_array_equal(d["Q_e"], RHO)
    assert d["P_net"] == 2.0 and d["sources_json"][0]["name"] == "nbi"
    bl = load_baseline_profiles(h5, scan_key=1500)
    assert "dd_sources" not in bl and bl["tau_E_th"] == 0.1
    with h5py.File(h5) as hf:
        assert hf["scan/1500/0"].attrs["H_98"] == 1.0
        assert hf["scan/1500/0/chi_perp_i"].attrs["units"] == "m^2 s^-1"
