"""Simple ECCD estimate and D-D neutron-rate estimates (synthetic; constants and self-consistency)."""
import dataclasses

import numpy as np
import pytest

from bouquet.kinrecon.current.geometry import FluxSurfaceGeometry
from bouquet.kinrecon.sources import ECLauncher, SimpleECModel
from bouquet.kinrecon.sources import ec, neutrons
from bouquet.kinrecon.sources import slowing_down as sd
from bouquet.kinrecon.sources.conventions import E_CHARGE, M_AMU, toroidal_frames

RHO = np.linspace(0, 1, 201)


def test_eccd_coefficient_from_si_constants():
    assert abs(ec.eccd_efficiency_coefficient() - 3.27) < 0.01


def test_ec_current_integral_and_zeta_models():
    g = FluxSurfaceGeometry.cylinder(0.6, 1.7, 2.0, n=201)
    g = dataclasses.replace(g, avg_b2=np.full(201, 4.0))
    ne, te, z = 4e19 * (1 - 0.5 * RHO ** 2), 3000 * (1 - 0.8 * RHO ** 2) + 50, np.full(201, 2.0)
    L = ECLauncher("g1", 1.0e6, 0.4, 0.05, 1)
    m = SimpleECModel([L], zeta0=0.3, zeta_model="constant")
    s = m.compute(RHO, time=0, ne=ne, te_ev=te, zeff=z, B0=2.0, R0=1.7, dvol_drho=g.dvol_drho, geometry=g)
    I = s.meta["launchers"]["g1"]["I_EC"]
    zeta = 3.27 * I * 1.7 * np.interp(0.4, RHO, ne) / 1e19 / (1e6 * np.interp(0.4, RHO, te) / 1e3)
    assert abs(zeta / 0.3 - 1) < 2e-3                                   # E1 reproduced
    I_int = toroidal_frames(s.jdotb["g1"], RHO, g)["current_enclosed"][-1]
    assert abs(I_int / I - 1) < 1e-3                                    # profile carries I_EC (C7)
    fb = SimpleECModel([L], zeta0=0.3, zeta_model="fisch_boozer")
    assert abs(fb.current(L, 4e19, 2000, 2.0, 1.7) / m.current(L, 4e19, 2000, 2.0, 1.7) - 6 / 7) < 1e-12
    # FUSE formula I = c 0.09 T_keV P / ((5 + Z) R n20)  <=>  fisch_boozer with zeta0 = 0.4905 c
    c = 0.37
    fuse = c * 0.09 * 2.0 * 1e6 / ((5 + 2.0) * 1.7 * 0.4)
    ours = SimpleECModel([L], zeta0=32.7 * 0.09 * c / 6, zeta_model="fisch_boozer").current(L, 4e19, 2000, 2.0, 1.7)
    assert abs(ours / fuse - 1) < 2e-3
    with pytest.raises(ValueError):
        SimpleECModel([L], zeta0=0.3, zeta_model="guess")


def test_bosch_hale_cross_section_and_reactivity_are_mutually_consistent():
    """The reactivity fit (D2) equals the Maxwellian average of the cross-section fit (D1) to ~1 %."""
    mr = 2.0141 * M_AMU / 2
    E = np.linspace(0.5, 600.0, 300001)                                  # keV, CM
    for T in (2.0, 5.0, 10.0, 20.0, 50.0):
        Ej, Tj = E * 1e3 * E_CHARGE, T * 1e3 * E_CHARGE
        sv = np.sqrt(8 / (np.pi * mr)) * Tj ** -1.5 * np.trapezoid(neutrons.dd_n_cross_section(E) * Ej * np.exp(-Ej / Tj), Ej)
        assert abs(sv / neutrons.dd_n_reactivity(T) - 1) < 0.02
    # a magnitude anchor: <sigma v>_DD,n at 10 keV ~ 6e-25 m^3/s (textbook order)
    assert 4e-25 < neutrons.dd_n_reactivity(10.0) < 8e-25


def test_beam_target_rate_against_direct_quadrature_and_thermonuclear_factor():
    ne, te = np.array([4e19]), np.array([3000.0])
    ions = [sd.IonSpecies("D", 2.0141, 1.0, ne)]
    m = sd.classical_moments(np.array([1e18]), 80e3, ne, te, ions, 2.0141, 1.0)
    v0 = np.sqrt(2 * 80e3 * E_CHARGE / (2.0141 * M_AMU))
    vc = v0 * m["w"]
    rate = neutrons.beam_target_rate(np.array([1e18]), 80e3, 2.0141, m["tau_s"], vc, ne)
    v = np.linspace(1e-6, v0, 400001)
    sig = neutrons.dd_n_cross_section(0.5 * (2.0141 * M_AMU / 2) * v ** 2 / E_CHARGE / 1e3)
    ref = ne * 1e18 * m["tau_s"] * np.trapezoid(sig * v ** 3 / (v ** 3 + vc ** 3), v)
    assert abs(rate[0] / ref[0] - 1) < 1e-4
    np.testing.assert_allclose(neutrons.thermonuclear_rate(2e19, 5.0), 0.5 * 4e38 * neutrons.dd_n_reactivity(5.0))


def test_warm_target_reduces_to_cold_and_raises_rate():
    ne, te = np.array([4e19, 4e19]), np.array([3000.0, 3000.0])
    ions = [sd.IonSpecies("D", 2.0141, 1.0, ne)]
    S = np.array([1e18, 1e18])
    m = sd.classical_moments(S, 60e3, ne, te, ions, 2.0141, 1.0)
    vc = np.sqrt(2 * 60e3 * E_CHARGE / (2.0141 * M_AMU)) * m["w"]
    cold = neutrons.beam_target_rate(S, 60e3, 2.0141, m["tau_s"], vc, ne)
    tiny = neutrons.beam_target_rate(S, 60e3, 2.0141, m["tau_s"], vc, ne, ti_kev=np.array([1e-4, 1e-4]))
    warm = neutrons.beam_target_rate(S, 60e3, 2.0141, m["tau_s"], vc, ne, ti_kev=np.array([3.0, 3.0]))
    np.testing.assert_allclose(tiny, cold, rtol=2e-3)
    assert np.all(warm > cold)            # sigma v is convex in v over the beam range: target motion adds rate
