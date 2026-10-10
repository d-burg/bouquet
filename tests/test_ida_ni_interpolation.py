"""ida_hybrid: the run-grid ni is main_ion_density_from_zeff of the run-grid
(ne, Z_eff), so the baseline stays quasineutral between IDA's nodes.

Synthetic inputs only.
"""
from types import SimpleNamespace

import numpy as np

import bouquet.io.imas as I
from bouquet.physics import main_ion_density_from_zeff

_Z = 6.0


def _ida():
    x = np.linspace(0.0, 1.0, 9)                      # coarse IDA grid
    ne = 5e19 * (1.0 - 0.6 * x ** 2) + 1e18
    zeff = 1.6 + 0.8 * x ** 3
    ni = main_ion_density_from_zeff(ne, zeff, _Z)
    s = 0.05 * ne
    return SimpleNamespace(psi_N=x, ne=ne, te=1e3 * (1 - x) + 50, ti=1e3 * (1 - x) + 50,
                           ni=ni, Zeff=zeff, sigma_ne=s, sigma_te=s, sigma_ni=s,
                           sigma_ti=s, q=None)


def _merge(monkeypatch, ida):
    import bouquet.io.ida as IDA
    monkeypatch.setattr(IDA, "read_ida", lambda *a, **k: ida)
    monkeypatch.setattr(I, "_read_ida_omega", lambda *a, **k: None)
    psi = np.linspace(0.0, 1.0, 101)                  # finer run grid
    # the EXPERIMENTAL IDA route (PR #56): ni rebuilt from IDA's (ne, Z_eff)
    return psi, I._merge_ida_kinetics(psi, None, None, None, "x.cdf", 1.0, _Z,
                                      ni_source="all")


def test_the_run_grid_ni_is_the_formula_of_the_run_grid_ne_and_zeff(monkeypatch):
    psi, out = _merge(monkeypatch, _ida())
    ne, ni, zeff = out[0], out[3], out[4]
    np.testing.assert_allclose(ni, main_ion_density_from_zeff(ne, zeff, _Z),
                               rtol=1e-14, atol=0.0)
    # ... and on IDA's own nodes it is IDA's ni
    xi = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(ni[np.isin(psi, xi)],
                               _ida().ni[np.isin(xi, psi)], rtol=1e-13)
