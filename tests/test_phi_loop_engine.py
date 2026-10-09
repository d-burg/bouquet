"""coord="phi_n" through the self-consistent loop and the unified engine.

The run grid x is Phi_N; every equilibrium maps its nodes to psi_N on its own
toroidal-flux map (``get_torflux_map``), and the Redl gradients, the FSA
geometry and the integrals work on those psi_N -- as OFT's Fortran bootstrap
does.  A mock equilibrium with an analytic map Phi(psi) checks that a Phi_N
evaluation on x equals the psi_N evaluation on psi(x), and that every engine
solve tags its profiles.  Solver-free.
"""
import numpy as np
import pytest

from test_jbs_loop import _MockEq, _kin


def _phi_of_psi(psi):
    """A monotone Phi(psi) with Phi(0) = 0, Phi(1) = 1 (q-like stretching)."""
    psi = np.asarray(psi, dtype=float)
    return (psi + 0.6 * psi ** 3) / 1.6


def _psi_of_phi(phi):
    grid = np.linspace(0.0, 1.0, 20001)
    return np.interp(np.asarray(phi, dtype=float), _phi_of_psi(grid), grid)


class _MapEq(_MockEq):
    """The mock geometry plus a toroidal-flux map."""

    def get_torflux_map(self, x, inverse=False):
        x = np.asarray(x, dtype=float)
        if inverse:
            return _psi_of_phi(x), None
        return _phi_of_psi(x), None


def test_residual_weights_on_phi_are_the_psi_weights_at_the_mapped_nodes():
    from bouquet.jbs_loop import residual_weights
    x = np.linspace(0.0, 1.0, 41)
    w_phi, x_phi, k_phi = residual_weights(_MapEq(), x, 1e-3, coord="phi_n")
    w_psi, x_psi, k_psi = residual_weights(_MapEq(), _psi_of_phi(x), 1e-3)
    np.testing.assert_array_equal(w_phi, w_psi)
    np.testing.assert_array_equal(x_phi, x_psi)
    assert k_phi == k_psi


def test_evaluate_jBS_on_phi_is_the_psi_evaluation_at_the_mapped_nodes():
    pytest.importorskip("OpenFUSIONToolkit.TokaMaker.bootstrap",
                        reason="evaluate_jBS wraps OFT's Redl")
    from bouquet.physics import evaluate_jBS
    x = np.linspace(0.0, 1.0, 81)
    psi = _psi_of_phi(x)
    kin = _kin(psi)                    # the same node values either way
    j_phi, d_phi = evaluate_jBS(_MapEq(), x, *kin, coord="phi_n")
    j_psi, d_psi = evaluate_jBS(_MapEq(), psi, *kin)
    np.testing.assert_array_equal(j_phi, j_psi)
    assert d_phi["I_BS"] == d_psi["I_BS"]
    np.testing.assert_array_equal(d_phi["psi_N"], psi)
    np.testing.assert_array_equal(d_phi["x"], x)
    assert d_phi["coord"] == "phi_n"


def test_the_structured_closure_with_its_basis_on_the_run_grid():
    """basis_x = psi_N is the closure without it, bit for bit."""
    from bouquet.utils import close_ip_structured
    psi = np.linspace(0.0, 1.0, 41)
    w = 1.0 + psi
    j_ind, j_bs, j_fix = 1e6 * (1 - psi ** 2), 2e5 * psi ** 4, 0 * psi
    a = close_ip_structured(psi, w, 0.0, 1.2e6, j_ind, j_bs, j_fix)
    b = close_ip_structured(psi, w, 0.0, 1.2e6, j_ind, j_bs, j_fix,
                            basis_x=psi.copy())
    np.testing.assert_array_equal(a["s_ind"], b["s_ind"])
    np.testing.assert_array_equal(a["s_bs"], b["s_bs"])
    # on a Phi_N basis the profiles follow x, the Ip closure still holds
    x = _phi_of_psi(psi)
    c = close_ip_structured(psi, w, 0.0, 1.2e6, j_ind, j_bs, j_fix,
                            basis_x=x)
    from scipy.integrate import trapezoid
    jc = c["s_ind"] * j_ind + c["s_bs"] * j_bs + j_fix
    assert abs(trapezoid(w * jc, psi) - 1.2e6) < 1e-6 * 1.2e6


class _Settings:
    maxits = 50


class _FakeSolver(_MapEq):
    """Records what the engine backend hands the solver."""

    def __init__(self):
        super().__init__()
        self.profiles = []
        self.settings = _Settings()

    def set_targets(self, **kw):
        pass

    def set_profiles(self, **kw):
        self.profiles.append(kw)

    def solve(self):
        pass


@pytest.mark.parametrize("coord", ["psi_n", "phi_n"])
def test_every_engine_solve_is_tagged_with_the_run_coordinate(coord):
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend
    x = np.linspace(0.0, 1.0, 21)
    c = SimpleNamespace(psi_N=x, pressure=2e4 * (1 - x ** 2), Ip=1e6)
    mg = _FakeSolver()
    TokaMakerBackend(mg, c, coord=coord).solve(1e6 * (1 - x ** 2))
    p = mg.profiles[-1]
    if coord == "phi_n":
        assert p["ffp_prof"]["coord"] == "phi_n_relabel"
        assert p["pp_prof"]["coord"] == "phi_n"
    else:
        assert "coord" not in p["ffp_prof"] and "coord" not in p["pp_prof"]
    np.testing.assert_array_equal(p["ffp_prof"]["x"], x)


def test_the_backends_flux_integral_maps_the_run_grid():
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend
    x = np.linspace(0.0, 1.0, 21)
    seen = {}

    class _S(_FakeSolver):
        def flux_integral(self, psi, prof):
            seen["psi"] = np.asarray(psi)
            return 0.0

    c = SimpleNamespace(psi_N=x, pressure=0 * x, Ip=1e6)
    TokaMakerBackend(_S(), c, coord="phi_n").flux_integral(x, 0 * x)
    np.testing.assert_allclose(seen["psi"], _psi_of_phi(x))
