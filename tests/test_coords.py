"""bouquet.coords: the run coordinate at the OpenFUSIONToolkit boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet import coords
from bouquet.edge_pressure import solver_pp_profile


class _Eq:
    """Stand-in solver: psi_bounds plus a toroidal-flux map ψ_N = Φ_N**0.8."""
    psi_bounds = (-0.4, 0.6)

    def __init__(self):
        self.calls = []

    def get_torflux_map(self, x, inverse=False):
        self.calls.append(inverse)
        x = np.asarray(x, float)
        return (x ** 0.8 if inverse else x ** 1.25), np.ones_like(x)


def _pp(eq, x, p, coord=coords.PSI):
    """The solver's P' profile of ``p`` on ``x`` (no edge pin)."""
    return solver_pp_profile(x, p, eq.psi_bounds[1] - eq.psi_bounds[0],
                             {"edge_pprime_pin": False}, coord)


@pytest.fixture
def swb_params(monkeypatch):
    def _set(names):
        monkeypatch.setattr(coords, "_SWB_PARAMS", frozenset(names), raising=False)
    return _set


X = np.array([0.0, 0.05, 0.2, 0.5, 0.9, 1.0])


def test_psi_profile_dicts_carry_no_coord():
    d = coords.oft_prof("jphi-linterp", X, X ** 2)
    assert set(d) == {"type", "x", "y"}
    pp = _pp(_Eq(), X, 1.0 - X ** 2)
    assert "coord" not in pp and pp["type"] == "linterp"


def test_phi_profile_tags():
    assert coords.oft_prof("jphi-linterp", X, X, coords.PHI)["coord"] == "phi_n_relabel"
    assert coords.oft_prof("linterp", X, X, coords.PHI)["coord"] == "phi_n"


def test_pp_scale_is_the_same_in_both_coordinates():
    p = 1.0 - X ** 2
    a = _pp(_Eq(), X, p)["y"]
    b = _pp(_Eq(), X, p, coords.PHI)["y"]
    np.testing.assert_array_equal(a, b)
    from bouquet.utils import pchip_derivative
    np.testing.assert_array_equal(a, pchip_derivative(X, p) / 1.0)


def test_phi_pp_times_dphi_dpsi_is_psi_pp():
    """P'(Φ)·dΦ/dψ ≈ P'(ψ) on the same nodes (PCHIP tolerance)."""
    eq = _Eq()
    x_phi = np.linspace(0.0, 1.0, 401)
    psi = coords.psi_at(eq, x_phi, coords.PHI)
    p = (1.0 - psi) ** 2 * (1.0 + psi)            # p(ψ) sampled on the Φ nodes
    pp_phi = _pp(eq, x_phi, p, coords.PHI)["y"]
    pp_psi = _pp(eq, psi, p)["y"]
    dphi_dpsi = 1.25 * psi ** 0.25                 # Φ = ψ**1.25
    inner = slice(5, -2)   # dψ/dΦ singular at the axis; one-sided PCHIP ends
    np.testing.assert_allclose(pp_phi[inner] * dphi_dpsi[inner], pp_psi[inner],
                               rtol=5e-3, atol=2e-3)
    # analytic dp/dψ / psi_range
    exact = (-2 * (1 - psi) * (1 + psi) + (1 - psi) ** 2) / 1.0
    np.testing.assert_allclose(pp_psi[inner], exact[inner], rtol=5e-3, atol=2e-3)


def test_psi_of():
    eq = _Eq()
    np.testing.assert_array_equal(coords.psi_of(eq, X, psi_pad=0.01),
                                  np.clip(X, 0.01, 0.99))
    assert eq.calls == []
    got = coords.psi_of(eq, X, coords.PHI, psi_pad=0.01)
    np.testing.assert_allclose(got, np.clip(X ** 0.8, 0.01, 0.99))
    assert eq.calls == [True]


def test_psi_at_is_the_identity_object_in_a_psi_run():
    assert coords.psi_at(_Eq(), X) is X


def test_rho_tor_runs_as_phi_n():
    assert coords.run_coord("rho_tor") == coords.PHI
    assert coords.run_coord("psi_n") == coords.PSI
    with pytest.raises(ValueError):
        coords.run_coord("psi")


# ---- the OFT-capability detection of solve_with_bootstrap's grid argument
# (restored from the pre-cleanup PR #64 branch, review PR64 B13: on a
# toolkit without x it decides the default-path SWB call) ----
@pytest.mark.parametrize("arg", ["x", "psi_N"])
def test_swb_grid_on_a_toolkit_with_a_grid_argument(swb_params, arg):
    swb_params({"mygs", "ne", arg})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), xi)
    assert list(coords.swb_grid_kwargs(xi)) == [arg]
    assert coords.swb_grid_kwargs(xi, coords.PHI)["coord"] == coords.PHI
    np.testing.assert_array_equal(coords.swb_seed(xi), (1 - xi ** 1.5) ** 1.5)


def test_swb_grid_prefers_x(swb_params):
    swb_params({"x", "psi_N"})
    assert list(coords.swb_grid_kwargs(X)) == ["x"]


def test_swb_grid_on_a_legacy_toolkit(swb_params):
    swb_params({"mygs", "ne"})
    xi = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_array_equal(coords.swb_grid(xi), np.linspace(0, 1, 4))
    np.testing.assert_array_equal(coords.swb_seed(xi),
                                  (1 - np.linspace(0, 1, 4) ** 1.5) ** 1.5)
    assert coords.swb_grid_kwargs(xi) == {}


def test_seed_is_the_same_physical_profile_in_a_phi_run(swb_params):
    swb_params({"x"})
    xphi = np.array([0.0, 0.1, 0.4, 1.0])
    psi = xphi ** 0.8
    np.testing.assert_array_equal(coords.swb_seed(xphi, psi), (1 - psi ** 1.5) ** 1.5)
    swb_params(set())                       # legacy toolkit: its own uniform grid
    np.testing.assert_array_equal(coords.swb_seed(xphi, psi),
                                  (1 - np.linspace(0, 1, 4) ** 1.5) ** 1.5)


def test_check_backend():
    coords.check_backend(coords.PSI)
    with pytest.raises(ValueError):
        coords.check_backend("rho_tor")


class TestCheckRun:
    def _cfg(self, coord="psi_n"):
        from bouquet.config import (BouquetConfig, GenerationConfig,
                                    ImasSource, SolverConfig)
        return BouquetConfig(source=ImasSource(ids_path="x.json", coord=coord),
                             solver=SolverConfig(mesh_path="m.h5"),
                             generation=GenerationConfig(), output_header="t")

    def test_psi_run_passes(self):
        assert coords.check_run(self._cfg()) == coords.PSI

    def test_rho_tor_is_a_phi_run(self, monkeypatch):
        monkeypatch.setattr(coords, "check_backend", lambda c: None)
        assert coords.check_run(self._cfg("rho_tor")) == coords.PHI


def test_phi_n_from_q():
    psi = np.linspace(0.0, 1.2, 61)
    q = 1.0 + 3.0 * psi ** 2
    inside, phi = coords.phi_n_from_q(psi, q)
    assert inside.sum() == 51 and phi[0] == 0.0 and phi[-1] == pytest.approx(1.0)
    x = psi[inside]
    np.testing.assert_allclose(phi, (x + x ** 3) / 2.0, atol=2e-4)
    # a grid that stops short of 1 is closed at 1 by interpolation
    _, phi2 = coords.phi_n_from_q(psi[:-2] * 1.0, q[:-2])
    assert phi2[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        coords.phi_n_from_q(psi[1:], q[1:])


def test_phi_n_from_q_bracket():
    psi = np.linspace(0.0, 1.2, 64)                  # does not hit 1.0
    q = 1.0 + 3.0 * psi ** 2
    inside, phi = coords.phi_n_from_q(psi, q)
    inb, phib = coords.phi_n_from_q(psi, q, bracket=True)
    n, b = inside.sum(), int(inside.sum())
    assert inb.sum() == n + 1 and inb[b] and np.array_equal(phib[:n], phi)
    # ida_fuse ida_phi_n: trapezoid to the bracket over the integral to 1
    x, y = psi[:b + 1], q[:b + 1]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
    q1 = y[b - 1] + (y[b] - y[b - 1]) * (1.0 - x[b - 1]) / (x[b] - x[b - 1])
    den = cum[b - 1] + 0.5 * (y[b - 1] + q1) * (1.0 - x[b - 1])
    assert phib[-1] > 1.0 and phib[-1] == pytest.approx(cum[b] / den, rel=1e-12)
    # a grid hitting 1.0, or a non-finite q at the bracket: no extra node
    psi1 = np.linspace(0.0, 1.2, 61)
    assert coords.phi_n_from_q(psi1, 1.0 + psi1, bracket=True)[0].sum() == 51
    qn = q.copy()
    qn[b] = np.nan
    assert coords.phi_n_from_q(psi, qn, bracket=True)[0].sum() == n


class TestGfileRunGrids:
    """g-file nodes by rhovn², IDA nodes by the IDA's q."""

    def test_ida_uses_its_own_q(self):
        psi = np.linspace(0.0, 1.2, 61)
        q = 1.0 + 3.0 * psi ** 2
        psi_eq = np.linspace(0.0, 1.0, 11)
        eqdsk = SimpleNamespace(psi_N=psi_eq, rhovn=psi_eq ** 0.25)
        kin = {"psi_N": psi, "q": q, "ne": 1.0 + psi}
        x_run, x_kin, inside, kin2 = coords.gfile_run_grids(
            eqdsk, kin, "ida.cdf", coords.PHI)
        np.testing.assert_allclose(x_run, psi_eq ** 0.5)
        np.testing.assert_array_equal(x_kin, coords.phi_n_from_q(psi, q)[1])
        assert inside.sum() == 51 and "q" not in kin2
        np.testing.assert_array_equal(kin2["ne"], 1.0 + psi[inside])

    def test_an_ida_without_q_is_refused(self):
        psi_eq = np.linspace(0.0, 1.0, 11)
        eqdsk = SimpleNamespace(psi_N=psi_eq, rhovn=psi_eq ** 0.5)
        with pytest.raises(ValueError, match="no q"):
            coords.gfile_run_grids(eqdsk, {"psi_N": psi_eq}, "ida.cdf",
                                   coords.PHI)

    def test_the_ida_loader_keeps_q(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from test_ni_fast_subtraction import _ida_cdf
        from bouquet.baseline import _load_kinetic_profiles
        from bouquet.config import ReconstructionSource
        cdf = str(tmp_path / "ida.cdf")
        _ida_cdf(cdf)
        with h5py.File(cdf, "a") as f:
            f["q"] = (1.0 + np.linspace(0, 1, 33) ** 2)[None, :]
        kin = _load_kinetic_profiles(ReconstructionSource(
            geqdsk_path="g", profiles_path=cdf, time=1.0))
        np.testing.assert_allclose(kin["q"], 1.0 + np.linspace(0, 1, 33) ** 2)


def test_r2_ip_scale_maps_psi_only_on_the_legacy_branch(monkeypatch):
    """The anchor path never evaluates the ψ map; the legacy one maps Φ → ψ."""
    from bouquet import TokaMaker_interface as ti

    class _Anchor:
        def solve_scale(self, j_ind, j_other):
            return 1.5

    eq = _Eq()
    x = np.linspace(0.0, 1.0, 11)
    j = np.ones_like(x)
    assert ti._r2_ip_scale(_Anchor(), eq, j, j, x, 1.0, coords.PHI) == 1.5
    assert eq.calls == []

    seen = {}

    def _f(alpha, mygs, j_ind, j_other, psi_N, Ip_target):
        seen["psi"] = psi_N
        return alpha - 2.0 * Ip_target

    monkeypatch.setattr(ti, "Ip_flux_integral_vs_target", _f)
    s = ti._r2_ip_scale(None, eq, j, j, x, 1.0, coords.PHI)
    assert abs(s - 2.0) < 1e-5
    np.testing.assert_allclose(seen["psi"], x ** 0.8)
    ti._r2_ip_scale(None, eq, j, j, x, 1.0)
    assert seen["psi"] is x                        # psi_n: the grid itself
