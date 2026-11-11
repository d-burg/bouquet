"""Classical slowing-down physics of kinrecon.sources (analytic limits, conservation, units).

Synthetic inputs only.  Every check is against an independent route: textbook coefficients,
closed forms vs quadrature, or a direct integration of the single-particle slowing-down law.
"""
import numpy as np
import pytest

from bouquet.kinrecon.sources import slowing_down as sd
from bouquet.kinrecon.sources.conventions import E_CHARGE, M_AMU

NE = np.array([2e19, 5e19, 8e19])
TE = np.array([500.0, 3000.0, 8000.0])


def _ions(n=NE, frac_c=0.0):
    nz = frac_c * n
    return [sd.IonSpecies("D", 2.0, 1.0, n - 6.0 * nz), sd.IonSpecies("C", 12.0, 6.0, nz)] if frac_c else \
        [sd.IonSpecies("D", 2.0, 1.0, n)]


def test_tau_s_reproduces_textbook_coefficient():
    # tau_s = 6.27e8 A T_e[eV]^1.5 / (Z^2 n_e[cm^-3] lnL)  (Stix/Spitzer practical form)
    lle = sd.ln_lambda_electron(NE, TE)
    tau = sd.slowing_down_time(NE, TE, 2.0, 1.0, lle)
    ref = 6.27e8 * 2.0 * TE ** 1.5 / (NE * 1e-6 * lle)
    np.testing.assert_allclose(tau, ref, rtol=2e-3)
    # dimensional scaling: tau_s ~ A_b T_e^1.5 / (Z_b^2 n_e)
    np.testing.assert_allclose(sd.slowing_down_time(NE, TE, 4.0, 2.0, lle), tau * 2.0 / 4.0, rtol=1e-12)


def test_critical_energy_reproduces_14p8_coefficient():
    ions = _ions(frac_c=0.02)
    lle = sd.ln_lambda_electron(NE, TE)
    Ec = sd.critical_energy(NE, TE, ions, 2.0, lle, [lle, lle])
    zsum = sum(i.n * i.Z ** 2 / (NE * i.A) for i in ions)
    np.testing.assert_allclose(Ec, 14.8 * 2.0 * TE * zsum ** (2.0 / 3.0), rtol=3e-3)


def test_energy_integral_limits_and_closed_form():
    w = np.array([1e-5, 1e-3, 0.05, 0.3, 1.0, 2.0, 5.0, 19.9, 20.1, 60.0])
    J = sd.energy_integral(w)
    Jq = sd.pitch_integral(w, 0.0, 4)
    np.testing.assert_allclose(J, Jq, rtol=1e-9)
    # E >> E_c: electron drag only, J -> 1/2;  E << E_c: ion drag only, J -> 1/(5 w^3)
    assert abs(sd.energy_integral(np.array([1e-6]))[0] - 0.5) < 1e-10
    big = np.array([200.0, 1000.0])
    np.testing.assert_allclose(sd.energy_integral(big) * 5 * big ** 3, 1.0, rtol=1e-6)


def test_density_integral_closed_form():
    w = np.array([0.01, 0.2, 1.0, 3.0])
    nq = sd.pitch_integral(w, 0.0, 2)          # int x^2/(x^3+w^3)
    np.testing.assert_allclose(sd.density_integral(w), nq, rtol=1e-9)


def test_current_integral_closed_form_and_limit():
    w = np.array([0.02, 0.4, 1.0, 2.5])
    y = 1.0 / w
    F = np.log(1 + y) / 3 - np.log(y ** 2 - y + 1) / 6 + (np.arctan((2 * y - 1) / np.sqrt(3)) + np.pi / 6) / np.sqrt(3)
    np.testing.assert_allclose(sd.pitch_integral(w, 0.0, 3), 1.0 - w * F, rtol=1e-9)
    assert abs(sd.pitch_integral(np.array([1e-6]), 0.7, 3)[0] - 1.0) < 1e-5


def test_moments_against_direct_single_particle_integration():
    """W and n from the closed forms equal S * time integrals of the slowing-down law dE/dt (S1)."""
    ne, te = np.array([4e19]), np.array([2500.0])
    ions = _ions(ne, 0.01)
    E0 = 80e3
    m = sd.classical_moments(np.array([1.0]), E0, ne, te, ions, 2.0, 1.0, ln_lambda_i="equal_to_electron")
    tau, Ec = m["tau_s"][0], m["E_c"][0]
    E = np.geomspace(1e-6 * E0, E0, 200001)
    dEdt = 2 * E / tau * (1 + (Ec / E) ** 1.5)          # |dE/dt|
    W = np.trapezoid(E * E_CHARGE / dEdt, E)            # S = 1: int E dt
    n = np.trapezoid(1.0 / dEdt, E)                     # int dt
    assert abs(W / m["W"][0] - 1) < 2e-4
    assert abs(n / m["n_fast"][0] - 1) < 2e-3          # tail below 1e-6 E0 is negligible


def test_energy_conservation_and_local_identity():
    S = np.array([1e18, 3e18, 5e17])
    ions = _ions(frac_c=0.015)
    for E0 in (80e3, 40e3, 26.7e3):
        m = sd.classical_moments(S, E0, NE, TE, ions, 2.0, 1.0)
        np.testing.assert_allclose(m["q_e"] + m["q_i"], S * E0 * E_CHARGE, rtol=1e-12)   # injected = e + i
        np.testing.assert_allclose(m["W"], 0.5 * m["tau_s"] * m["q_e"], rtol=1e-12)       # S13
        assert np.all(m["q_i"] >= 0) and np.all(m["q_e"] >= 0)


def test_pressure_anisotropy_trace_and_isotropic_birth():
    S = np.array([1e18, 2e18, 1e18])
    ions = _ions(frac_c=0.01)
    m = sd.classical_moments(S, 80e3, NE, TE, ions, 2.0, 1.0, xi0=0.6)
    np.testing.assert_allclose((m["p_par"] + 2 * m["p_perp"]) / 3, m["p_fast"], rtol=1e-12)
    assert np.all(m["p_par"] > m["p_perp"])                     # tangential birth: P_par > P_perp
    iso = sd.classical_moments(S, 80e3, NE, TE, ions, 2.0, 1.0, xi0=0.6, p2_xi0=0.0)
    np.testing.assert_allclose(iso["p_par"], iso["p_perp"], rtol=1e-12)   # <P_2> = 0: isotropic


def test_fast_current_absolute_normalization_electron_drag_limit():
    """w -> 0: J_fast = Z_b e S tau_s v0 xi0 exactly (momentum decays as exp(-t/tau_s)); no numerical factor."""
    ne, te = np.array([1e19]), np.array([300.0])          # algebraic limit E0 >> E_c (w ~ 0.02)
    ions = _ions(ne)
    E0, xi0, S = 1.0e7, 0.5, np.array([1e17])
    m = sd.classical_moments(S, E0, ne, te, ions, 2.0, 1.0, xi0=xi0)
    v0 = np.sqrt(2 * E0 * E_CHARGE / (2.0 * M_AMU))
    ref = E_CHARGE * S * m["tau_s"] * v0 * xi0
    assert m["w"][0] < 0.12
    np.testing.assert_allclose(m["j_fast"], ref * sd.pitch_integral(m["w"], m["R_D"] / 3, 3), rtol=1e-12)
    assert abs(m["j_fast"][0] / ref[0] - 1) < 2.0 * m["w"][0]     # 1 - I_1 = O(w)


def test_moments_linear_in_source_and_zero_source():
    ions = _ions()
    m1 = sd.classical_moments(np.array([1., 2., 3.]) * 1e17, 50e3, NE, TE, ions, 2.0, 1.0, xi0=0.4)
    m2 = sd.classical_moments(np.array([2., 4., 6.]) * 1e17, 50e3, NE, TE, ions, 2.0, 1.0, xi0=0.4)
    for k in ("n_fast", "W", "j_fast", "q_e", "p_par"):
        np.testing.assert_allclose(m2[k], 2 * m1[k], rtol=1e-12)
    m0 = sd.classical_moments(np.zeros(3), 50e3, NE, TE, ions, 2.0, 1.0, xi0=0.4)
    assert np.all(m0["p_fast"] == 0) and np.all(m0["j_fast"] == 0)


def test_coulomb_log_models_and_refusals():
    lle = sd.ln_lambda_electron(NE, TE, "nrl_ei")
    lls = sd.ln_lambda_electron(NE, TE, "sauter")
    np.testing.assert_allclose(lls - lle, 31.3 - 24.0 - np.log(1e3), atol=1e-12)   # same form, constants differ
    li = sd.ln_lambda_beam_ion(NE, TE, 2.0, 1.0, 80e3, _ions()[0])
    assert np.all(li > lle)          # beam-ion lnL exceeds the electron one at beam speeds
    with pytest.raises(ValueError):
        sd.ln_lambda_electron(NE, TE, "made_up")
    with pytest.raises(ValueError):
        sd.classical_moments(np.array([-1., 0, 0]), 50e3, NE, TE, _ions(), 2.0, 1.0)
    with pytest.raises(ValueError):
        sd.classical_moments(np.ones(3), 50e3, NE, TE, _ions(), 2.0, 1.0, xi0=1.5)
    with pytest.raises(ValueError):
        sd.slowing_down_time(NE, -TE, 2.0, 1.0, lle)


def test_pitch_exponent_matches_single_mass_reduction():
    """S5: for one ion species R_D = A_i/A_b, the single-mass exponent A_i Z_eff/(A_b Z_b) with Z_eff = Z_b = 1."""
    ions = _ions()
    lle = sd.ln_lambda_electron(NE, TE)
    np.testing.assert_allclose(sd.deflection_ratio(ions, 2.0, [lle]), 1.0, rtol=1e-12)
    np.testing.assert_allclose(sd.deflection_ratio(ions, 1.0, [lle]), 2.0, rtol=1e-12)   # H beam in D
