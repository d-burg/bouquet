"""``bouquet.kinrecon.current.bootstrap`` (Redl 2021, jboot1 form).

Synthetic profiles only.  The transcription is checked against OFT's own Python implementation
(``OpenFUSIONToolkit.TokaMaker.bootstrap.redl_bootstrap``, which agrees with TokaMaker's in-solve
Fortran Redl) when OFT is importable; the structural checks run
without OFT.
"""
import os
import sys

import numpy as np
import pytest

from bouquet.kinrecon.current.bootstrap import EC, coulomb_logs, collisionalities, redl_jdotb


def _profiles(n=60):
    x = np.linspace(0.02, 0.98, n)
    te = 3.0e3 * (1 - x ** 1.5) ** 1.5 + 100.0
    ti = 2.5e3 * (1 - x ** 1.4) ** 1.4 + 120.0
    ne = 5.0e19 * (1 - 0.6 * x ** 2) + 0.5e19
    zeff = 1.8 + 0.2 * x
    ni = ne * (6.0 - zeff) / 5.0             # one carbon impurity, quasineutrality
    q = 1.0 + 3.0 * x ** 2
    R = 1.7 + 0.02 * (1 - x)
    eps = np.maximum(0.35 * np.sqrt(x), 1e-3)
    ft = 1.46 * np.sqrt(eps) - 0.46 * eps
    F = 3.4 - 0.05 * x
    return x, te, ti, ne, ni, zeff, q, R, eps, ft, F


def test_zero_gradients_give_zero_and_linear_in_F():
    x, te, ti, ne, ni, zeff, q, R, eps, ft, F = _profiles()
    flat = [np.full_like(x, v[len(x) // 2]) for v in (te, ti, ne, ni)]
    jb, _ = redl_jdotb(x, *flat[:2], *flat[2:], zeff, q, R, eps, ft, F, 0.4)
    j1, _ = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.4)
    # numpy.gradient of a constant on a linspace grid is round-off, not exactly 0
    assert np.max(np.abs(jb)) < 1e-12 * np.max(np.abs(j1))
    j2, _ = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, 2.0 * F, 0.4)
    np.testing.assert_allclose(j2, 2.0 * j1, rtol=1e-14)
    # profiles falling outward in the positive frame give a co-current bootstrap
    assert np.all(j1[3:-3] > 0)
    # 1/dpsi scaling of every gradient term
    j3, _ = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.8)
    np.testing.assert_allclose(j3, 0.5 * j1, rtol=1e-14)


def test_collisionality_and_coulomb_log_forms():
    x, te, ti, ne, ni, zeff, q, R, eps, ft, F = _profiles(5)
    le, li = coulomb_logs(te, ti, ne, ni)
    exp_e = 23.5 - np.log(np.sqrt(ne / 1e6) * te ** -1.25) - np.sqrt(1e-5 + (np.log(te) - 2) ** 2 / 16)
    np.testing.assert_allclose(le, np.maximum(exp_e, 10.0), rtol=1e-14)
    nue, nui = collisionalities(q, R, eps, te, ti, ne, ni, zeff, le, li)
    np.testing.assert_allclose(nue, 6.921e-18 * q * R * ne * zeff * le / (te ** 2 * eps ** 1.5), rtol=1e-14)
    np.testing.assert_allclose(nui, 4.90e-18 * q * R * ni * (ne / ni * zeff) * li / (ti ** 2 * eps ** 1.5),
                               rtol=1e-14)
    with pytest.raises(ValueError):
        collisionalities(q, R, eps, te, ti, ne, ni, zeff, le, li, ion_collisionality="koh")


def _oft_bootstrap():
    cand = os.environ.get("OFT_PYTHONPATH")
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        from OpenFUSIONToolkit.TokaMaker.bootstrap import redl_bootstrap
        return redl_bootstrap
    except Exception:  # noqa: BLE001
        return None


def test_matches_oft_python_redl_to_round_off():
    redl_bootstrap = _oft_bootstrap()
    if redl_bootstrap is None:
        pytest.skip("OpenFUSIONToolkit not importable")
    x, te, ti, ne, ni, zeff, q, R, eps, ft, F = _profiles()
    dpsi = 0.45
    ours, parts = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, dpsi)
    d = lambda y: np.gradient(y, x, edge_order=2) / dpsi  # noqa: E731
    theirs, _ = redl_bootstrap(psi_N=x, Te=te, Ti=ti, ne=ne, ni=ni, pe=EC * ne * te, pi=EC * ni * ti, Zeff=zeff,
                               R=R, q=q, eps=eps, fT=ft, I_psi=F, dT_e_dpsi=d(te), dT_i_dpsi=d(ti),
                               dn_e_dpsi=d(ne), dn_i_dpsi=d(ni), ln_lambda_e=parts["ln_lambda_e"],
                               ln_lambda_ii=parts["ln_lambda_ii"], nu_e_star_override=parts["nu_e_star"],
                               nu_i_star_override=parts["nu_i_star"], use_legacy_L34=False, use_sign_q=False,
                               formula_form="jboot1")
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12 * np.max(np.abs(theirs)))


def test_impurity_enters_the_ion_pressure_only():
    """With an impurity density the ion pressure (and its gradient) is (n_i + n_imp) T_i, the same thermal
    pressure as the equilibrium's; the collisionality still uses the main ions.  Without it the earlier
    result is reproduced bit for bit."""
    x, te, ti, ne, ni, zeff, q, R, eps, ft, F = _profiles()
    nimp = ne * (zeff - 1.0) / 30.0
    j0, p0 = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.45)
    jn, _ = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.45, nimp=None)
    np.testing.assert_array_equal(j0, jn)
    jz, pz = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.45, nimp=nimp)
    np.testing.assert_array_equal(pz["nu_i_star"], p0["nu_i_star"])
    np.testing.assert_array_equal(pz["ln_lambda_ii"], p0["ln_lambda_ii"])
    assert "impurity" in pz["settings"]["ion_pressure"] and "only" in p0["settings"]["ion_pressure"]
    # the same as the main-ion formula with the ion pressure built from n_i + n_imp, written out
    from bouquet.kinrecon.current.bootstrap import redl_coefficients
    c = redl_coefficients(ft, zeff, pz["nu_e_star"], pz["nu_i_star"])
    d = lambda y: np.gradient(y, x, edge_order=2) / 0.45  # noqa: E731
    pe, pi = EC * ne * te, EC * (ni + nimp) * ti
    dp = EC * (ne * d(te) + te * d(ne) + (ni + nimp) * d(ti) + ti * d(ni + nimp))
    ref = -F * pe * (c["L31"] * dp / pe + c["L32"] * d(te) / te + c["L31"] * c["alpha"] * pi / pe * d(ti) / ti)
    np.testing.assert_allclose(jz, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref)))
    # more ion pressure at the same gradients: a larger bootstrap here (falling profiles)
    assert np.median(jz[5:-5] / j0[5:-5]) > 1.0
    with pytest.raises(ValueError):
        redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, 0.45, nimp=-nimp)


def test_impurity_ion_pressure_matches_oft_with_its_pressure_inputs():
    redl_bootstrap = _oft_bootstrap()
    if redl_bootstrap is None:
        pytest.skip("OpenFUSIONToolkit not importable")
    x, te, ti, ne, ni, zeff, q, R, eps, ft, F = _profiles()
    nimp = ne * (zeff - 1.0) / 30.0
    dpsi = 0.45
    ours, parts = redl_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, dpsi, nimp=nimp)
    d = lambda y: np.gradient(y, x, edge_order=2) / dpsi  # noqa: E731
    nit = ni + nimp
    theirs, _ = redl_bootstrap(psi_N=x, Te=te, Ti=ti, ne=ne, ni=ni, pe=EC * ne * te, pi=EC * nit * ti, Zeff=zeff,
                               R=R, q=q, eps=eps, fT=ft, I_psi=F, dT_e_dpsi=d(te), dT_i_dpsi=d(ti),
                               dn_e_dpsi=d(ne), dn_i_dpsi=d(ni),
                               dp_dpsi=EC * (ne * d(te) + te * d(ne) + nit * d(ti) + ti * d(nit)),
                               ln_lambda_e=parts["ln_lambda_e"], ln_lambda_ii=parts["ln_lambda_ii"],
                               nu_e_star_override=parts["nu_e_star"], nu_i_star_override=parts["nu_i_star"],
                               use_legacy_L34=False, use_sign_q=False, formula_form="jboot1")
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12 * np.max(np.abs(theirs)))
