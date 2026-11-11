"""Built-in beam model: attenuation, conservation, fractions, and the FUSE bugs as differences.

The FUSE behaviors are re-implemented ONLY here, as deliberately buggy references, so the
tests show what the bug-fixed model changes.  Synthetic circular geometry; no device data.
"""
import numpy as np
import pytest

from bouquet.kinrecon.sources import IonSpecies, NeutralBeam, SimpleBeamModel, classical_moments
from bouquet.kinrecon.sources import beam as bm
from bouquet.kinrecon.sources import slowing_down as sd
from bouquet.kinrecon.sources.conventions import E_CHARGE, M_AMU, M_PROTON

R0, A = 1.7, 0.6
RHO = np.linspace(0.0, 1.0, 101)
DV = 4 * np.pi ** 2 * R0 * A ** 2 * RHO          # dV/drho of concentric circles (rho = r/a)


def rho_of_rz(R, Z):
    r = np.sqrt((R - R0) ** 2 + Z ** 2) / A
    return np.where(r <= 1.0, r, np.nan)


def _plasma(ne0=5e19, te0=3000.0, uniform=False):
    if uniform:
        ne, te = np.full(RHO.size, ne0), np.full(RHO.size, te0)
    else:
        ne, te = ne0 * (1 - 0.8 * RHO ** 2), te0 * (1 - 0.9 * RHO ** 2) + 50
    nc = 0.015 * ne
    ions = [IonSpecies("D", 2.0, 1.0, ne - 6 * nc), IonSpecies("C", 12.0, 6.0, nc)]
    zeff = (ions[0].n + 36 * nc) / ne
    return ne, te, ions, zeff


def _beam(**kw):
    d = dict(name="b1", power=2.0e6, energy=80e3, A=2.0, Z=1.0, direction=1, power_fractions=(0.7, 0.2, 0.1),
             tangency_radius=1.2, source_r=5.0, source_z=0.0, vertical_angle=0.0)
    d.update(kw)
    return NeutralBeam(**d)


def test_fraction_conversions_roundtrip():
    c = np.array([0.5, 0.3, 0.2])
    p = bm.power_fractions_from_current(c)
    np.testing.assert_allclose(p, (c / [1, 2, 3]) / np.sum(c / [1, 2, 3]))
    np.testing.assert_allclose(bm.current_fractions_from_power(p), c)
    b = _beam(power_fractions=None, current_fractions=tuple(c))
    np.testing.assert_allclose(b.p_frac, p)
    np.testing.assert_allclose(np.sum(b.particle_rates * b.energies * E_CHARGE), b.power)    # power accounted
    with pytest.raises(ValueError):
        _beam(power_fractions=(0.7, 0.2, 0.2))
    with pytest.raises(ValueError):
        _beam(current_fractions=(0.5, 0.3, 0.2))           # both given


def test_straight_beam_attenuation_matches_exp_minus_n_sigma_L():
    ne, te, ions, _ = _plasma(uniform=True)
    b = _beam(tangency_radius=1.5)
    dep = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, [(ions[0].n, 1.0)], [(ions[1].n, 6.0)],
                                    n_steps=40000)
    L = 2 * np.sqrt((R0 + A) ** 2 - 1.5 ** 2)           # midplane chord outside R0 - a < R_t
    xs = bm.FreyaCrossSections()
    for k, Ek in enumerate(b.energies):
        vb = np.sqrt(2 * Ek * E_CHARGE / (2.0 * M_AMU))
        lam = xs.inverse_mfp(Ek / 2.0, vb, ne[:1], te[:1], [(ions[0].n[:1], 1.0)], [(ions[1].n[:1], 6.0)])[0]
        assert abs(np.log(dep.shine_through[k]) / (-lam * L) - 1) < 2e-3


def test_particle_conservation_and_pitch():
    ne, te, ions, _ = _plasma()
    b = _beam()
    for width in (0.0, 0.08):
        dep = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, [(ions[0].n, 1.0)], [(ions[1].n, 6.0)],
                                        n_steps=20000, smoothing_width=width)
        for k in range(3):
            born = bm.volume_integral(RHO, DV, dep.births[k])
            assert abs(born + dep.shine_through[k] * b.particle_rates[k] - b.particle_rates[k]) \
                < 1e-9 * b.particle_rates[k]
        # xi0 = R_t / R at birth: bounded by R_t/(R0+a) and 1, and largest where R ~ R_t
        live = dep.births[0] > 1e-3 * dep.births[0].max()
        assert np.all(dep.xi0[0][live] >= 1.2 / (R0 + A) - 1e-9) and np.all(dep.xi0[0][live] <= 1 + 1e-12)


def test_model_energy_conservation_and_outputs():
    ne, te, ions, zeff = _plasma()
    ft = 1.46 * np.sqrt(A * RHO / R0)
    model = SimpleBeamModel([_beam(), _beam(name="b2", direction=-1, tangency_radius=0.8, power=1e6)],
                            smoothing_width=0.06)
    s = model.compute(RHO, time=1.0, ne=ne, te_ev=te, ions=ions, zeff=zeff, ft=ft, B0=2.0, R0=R0, dvol_drho=DV,
                      rho_of_rz=rho_of_rz)
    for b in model.beams:
        qe, qi = s.heating[b.name]
        births = s.meta["beams"][b.name]["births_per_s"]
        absorbed = np.sum(np.array(births) * b.energies * E_CHARGE)
        assert abs(bm.volume_integral(RHO, DV, qe + qi) / absorbed - 1) < 1e-9
    assert np.trapezoid(s.jdotb["b1"] * DV, RHO) > 0 > np.trapezoid(s.jdotb["b2"] * DV, RHO)   # co / counter
    np.testing.assert_allclose((s.p_par["beam_A2_Z1"] + 2 * s.p_perp["beam_A2_Z1"]) / 3, s.p_fast["beam_A2_Z1"],
                               rtol=1e-10)
    assert s.provenance.limitations and "no anomalous fast-ion transport" in " ".join(s.provenance.limitations)
    assert s.provenance.settings["smoothing_width"] == 0.06
    assert s.completeness()                               # no priors given: reported, not filled in


def test_given_deposition_route_requires_shine_through():
    ne, te, ions, zeff = _plasma()
    shape = np.exp(-(RHO / 0.4) ** 2)
    dep = {"rho": RHO, "shape": shape, "xi0": 0.6}
    with pytest.raises(ValueError):
        SimpleBeamModel([_beam(tangency_radius=None, source_r=None, source_z=None, vertical_angle=None,
                               deposition=dep)], smoothing_width=0.0).compute(
            RHO, time=0, ne=ne, te_ev=te, ions=ions, zeff=zeff, ft=0.2 * RHO, B0=2, R0=R0, dvol_drho=DV)
    dep["shine_through"] = [0.05, 0.02, 0.01]
    b = _beam(tangency_radius=None, source_r=None, source_z=None, vertical_angle=None, deposition=dep)
    s = SimpleBeamModel([b], smoothing_width=0.0).compute(RHO, time=0, ne=ne, te_ev=te, ions=ions, zeff=zeff,
                                                          ft=0.2 * RHO, B0=2, R0=R0, dvol_drho=DV)
    born = s.meta["beams"]["b1"]["births_per_s"]
    np.testing.assert_allclose(born, b.particle_rates * (1 - np.array([0.05, 0.02, 0.01])), rtol=1e-12)


# --------------------------------------------------------------------------- FUSE bugs as differences
@pytest.mark.parametrize("te0", [500.0, 1500.0, 4000.0])
def test_fuse_bug_all_components_at_full_energy(te0):
    """Buggy reference (PR #1201): the E/2 and E/3 ions slow down from the full E.  Same births."""
    ne, te = np.full(5, 4e19), np.full(5, te0)
    ions = [IonSpecies("D", 2.0, 1.0, ne)]
    b = _beam(power_fractions=(0.55, 0.30, 0.15))
    S = b.particle_rates / 10.0                                       # per m^3, uniform
    fixed = sum(classical_moments(np.full(5, S[k]), b.energies[k], ne, te, ions, 2.0, 1.0)["p_fast"] for k in range(3))
    buggy = sum(classical_moments(np.full(5, S[k]), b.energy, ne, te, ions, 2.0, 1.0)["p_fast"] for k in range(3))
    ratio = buggy[0] / fixed[0]
    # exact lower bound: with electron drag only (J = 1/2 for every component) the ratio is
    # sum c_k / sum (c_k / k); ion drag lowers J more for the slower components, so the ratio is larger
    c = b.c_frac
    assert ratio >= np.sum(c) / np.sum(c / np.arange(1, 4)) - 1e-12
    assert ratio > 1.2                # PR #1201 quotes 1.3-1.6 for 80 keV D at T_e 0.5-4 keV (its own mix)
    nf = sum(classical_moments(np.full(5, S[k]), b.energies[k], ne, te, ions, 2.0, 1.0)["n_fast"] for k in range(3))
    nb = sum(classical_moments(np.full(5, S[k]), b.energy, ne, te, ions, 2.0, 1.0)["n_fast"] for k in range(3))
    assert nb[0] / nf[0] > 1.05


def test_fuse_bug_full_energy_cross_sections_change_deposition():
    ne, te, ions, _ = _plasma()
    b = _beam()
    xs = bm.FreyaCrossSections()

    class FullEnergyXS:                       # buggy reference: ion impact and CX at the full energy
        name = "buggy"

        def inverse_mfp(self, e_per_amu, vb, ne_, te_, hyd, imp):
            lam_e = xs.inverse_mfp(1e9, vb, ne_, te_, [], [])            # electron part only (E -> inf kills ions)
            return lam_e + (xs.inverse_mfp(b.energy / b.A, vb, ne_ * 0, te_, hyd, imp) -
                            xs.inverse_mfp(b.energy / b.A, vb, ne_ * 0, te_, [], []))
    good = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, [(ions[0].n, 1.0)], [(ions[1].n, 6.0)])
    bad = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, [(ions[0].n, 1.0)], [(ions[1].n, 6.0)],
                                    cross_sections=FullEnergyXS())
    np.testing.assert_allclose(bad.shine_through[0], good.shine_through[0], rtol=1e-12)   # full component same
    assert bad.shine_through[2] > 1.2 * good.shine_through[2]          # E/3 penetrates too far when buggy


def test_fuse_bug_current_fractions_used_as_power_fractions():
    """Buggy reference (PR #1202): beam_current_fraction used as the power split."""
    c = (0.80, 0.15, 0.05)
    right = _beam(power_fractions=None, current_fractions=c)
    wrong = _beam(power_fractions=c)                                  # c taken as power fractions
    assert wrong.particle_rates.sum() > right.particle_rates.sum()    # more (slower) particles
    assert wrong.p_frac[0] < right.p_frac[0]                          # less power at full energy
    ne, te = np.full(3, 4e19), np.full(3, 2500.0)
    ions = [IonSpecies("D", 2.0, 1.0, ne)]

    def W(b):
        return sum(classical_moments(np.full(3, b.particle_rates[k] / 10), b.energies[k], ne, te, ions, 2.0,
                                     1.0)["W"][0] for k in range(3))
    assert abs(W(wrong) / W(right) - 1) > 0.01


def test_fuse_nbcd_normalization_is_not_a_current_density():
    """FUSE: j = 0.1 e (S m v R xi) tau_mom / (A m_p).  Ours: Z e S tau_s v0 xi0 I_1.  Ratio = 0.1 R (in m)."""
    ne, te = np.array([4e19]), np.array([3000.0])
    ions = [IonSpecies("D", 2.0, 1.0, ne)]
    E0, xi0, S = 80e3, 0.6, np.array([1e18])
    m = classical_moments(S, E0, ne, te, ions, 2.0, 1.0, xi0=xi0)
    v0 = np.sqrt(2 * E0 * E_CHARGE / (2.0 * M_AMU))
    tau_mom = m["tau_s"] * sd.pitch_integral(m["w"], m["R_D"] / 3, 3)
    for R in (1.0, 1.7, 6.2):
        mom = S * (2.0 * M_AMU) * v0 * R * xi0                         # angular-momentum source (FUSE mombeam)
        j_fuse = 0.1 * E_CHARGE * mom * tau_mom / (2.0 * M_PROTON)
        np.testing.assert_allclose(j_fuse / m["j_fast"], 0.1 * R * M_AMU / M_PROTON, rtol=1e-12)


def test_fuse_axis_cut_vs_mirrored_smoothing():
    """Buggy reference (PR #1200): a Gaussian cut at rho = 0 has a non-zero slope on axis; mirrored has none."""
    rho = np.linspace(0, 1, 1001)
    dv = 4 * np.pi ** 2 * R0 * A ** 2 * rho
    pts, w = np.array([0.05]), np.array([[1.0]])
    good, _ = bm._bin(rho, dv, pts, w, 0.12, True)
    bad, _ = bm._bin(rho, dv, pts, w, 0.12, False)
    slope = lambda y: (y[0, 1] - y[0, 0]) / (rho[1] - rho[0])
    assert abs(slope(good)) < 0.05 * abs(slope(bad))
    np.testing.assert_allclose(bm.volume_integral(rho, dv, good[0]), 1.0, rtol=1e-12)
    np.testing.assert_allclose(bm.volume_integral(rho, dv, bad[0]), 1.0, rtol=1e-12)


def test_shielding_models():
    from bouquet.kinrecon.sources import nbcd
    eps = np.array([1e-4])
    for z in (1.0, 2.0):
        g_s = nbcd.trapped_electron_correction(np.array([z]), "sauter_l31", ft=1.46 * np.sqrt(eps))
        g_c = nbcd.trapped_electron_correction(np.array([z]), "start_cordey_fit", eps=eps)
        assert abs(g_c[0] / g_s[0] - 1) < 0.10          # large-aspect-ratio fit ~ L31 (Sauter) at small eps
    for model in ("redl_l31", "sauter_l31"):
        np.testing.assert_allclose(nbcd.trapped_electron_correction(np.array([1.5]), model, ft=np.array([1.0])), 1.0)
    np.testing.assert_allclose(nbcd.shielding_factor(np.array([1.0]), 1.0, "none"), 0.0)   # Z_eff = Z_b: full shielding
    np.testing.assert_allclose(nbcd.shielding_factor(np.array([2.0]), 1.0, "none"), 0.5)
    with pytest.raises(ValueError):
        nbcd.trapped_electron_correction(np.array([2.0]), "redl_l31")
