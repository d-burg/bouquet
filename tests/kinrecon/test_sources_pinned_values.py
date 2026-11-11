"""Built-in source models pinned against independently computed values, and the not-validated NBCD flag.

Each test below fails under one of the mutations an independent review found surviving the earlier suite:
the returned NBCD x 2, the l = 2 anisotropy exponent / 3, P_par - P_perp halved, the trapped-electron
correction G halved, the proton-impact ionization fit x 2.7.  Independent references:

* fast-ion current and anisotropy: the Legendre moments of the Gaffey/Cordey steady state, integrated here with
  ``scipy.integrate.quad`` from the distribution itself (P_par - P_perp = m int v^2 P_2(xi) f d^3v ->
  2 E0 P_2(xi0) S tau_s int x^4/(x^3+w^3) g^R_D dx; J = Z e S tau_s v0 xi0 int x^3/(x^3+w^3) g^(R_D/3) dx),
  using only tau_s, w = v_c/v0 and R_D from the package (those have their own tests);
* G: the Redl L31 of ``bouquet.kinrecon.current.bootstrap`` (a separate transcription, itself tested against
  OFT's implementation) at zero collisionality;
* proton-impact ionization of H(1s): peak ~1.45e-16 cm^2 near 50-60 keV/u (Shah and Gilbody, J. Phys. B 14,
  2361 (1981); recalled value) -- the FREYA fit must lie within a factor 1.5 of it at 50 keV/u (the fit gives
  1.76e-16; x 2.7 would give 4.7e-16).

Agreement bound for the quadrature comparisons: 1e-6 relative (the package's composite Gauss-Legendre is
accurate to ~1e-7 at w >= 1e-3; quad to 1e-10).
"""
import numpy as np
import pytest
from scipy.integrate import quad

from bouquet.kinrecon.current.bootstrap import redl_coefficients
from bouquet.kinrecon.sources import IonSpecies, NeutralBeam, SimpleBeamModel, classical_moments
from bouquet.kinrecon.sources import beam as bm
from bouquet.kinrecon.sources.conventions import E_CHARGE, M_AMU
from bouquet.kinrecon.sources.nbcd import trapped_electron_correction
from bouquet.kinrecon.sources.provider import NonInductiveSources, NotValidatedError

R0, A = 1.7, 0.6
RHO = np.linspace(0.0, 1.0, 51)
DV = 4 * np.pi ** 2 * R0 * A ** 2 * RHO
NE = np.full(RHO.size, 4e19)
TE = np.full(RHO.size, 2500.0)
NC = 0.01 * NE
IONS = [IonSpecies("D", 2.0, 1.0, NE - 6 * NC), IonSpecies("C", 12.0, 6.0, NC)]
ZEFF = (IONS[0].n + 36 * NC) / NE
FT = np.clip(1.46 * np.sqrt(A * RHO / R0), 0, 0.9)
XI0 = 0.55


def _g_int(w, alpha, power):
    def f(x):
        g = x ** 3 * (1 + w ** 3) / (x ** 3 + w ** 3)
        return x ** power / (x ** 3 + w ** 3) * g ** alpha
    return quad(f, 0.0, 1.0, epsabs=0, epsrel=1e-11, limit=200)[0]


def test_anisotropy_against_direct_moment():
    S = np.full(RHO.size, 1e18)
    E0 = 80e3
    for xi0 in (0.3, 0.8):
        m = classical_moments(S, E0, NE, TE, IONS, 2.0, 1.0, xi0=xi0)
        w, rd, tau = float(m["w"][0]), float(m["R_D"][0]), float(m["tau_s"][0])
        p2 = 0.5 * (3 * xi0 ** 2 - 1)
        dp_ref = 2.0 * p2 * S[0] * E0 * E_CHARGE * tau * _g_int(w, rd, 4)        # alpha_2 = 2*3/6 R_D = R_D
        assert (m["p_par"][0] - m["p_perp"][0]) / dp_ref == pytest.approx(1.0, abs=1e-6)
        j_ref = E_CHARGE * S[0] * tau * np.sqrt(2 * E0 * E_CHARGE / (2.0 * M_AMU)) * xi0 * _g_int(w, rd / 3.0, 3)
        assert m["j_fast"][0] / j_ref == pytest.approx(1.0, abs=1e-6)


def test_trapped_electron_correction_is_l31():
    for z in (1.0, 1.6, 3.0):
        for ft in (0.1, 0.4, 0.7):
            ref = redl_coefficients(np.array([ft]), np.array([z]), np.array([0.0]), np.array([0.0]))["L31"][0]
            assert trapped_electron_correction(np.array([z]), "redl_l31", ft=np.array([ft]))[0] == \
                pytest.approx(ref, rel=1e-12)


def test_proton_impact_cross_section_near_literature_peak():
    xs = bm.FreyaCrossSections()
    e = 50e3                                             # eV/u
    lam = xs.inverse_mfp(e, 1.0, np.array([0.0]), np.array([1000.0]), [(np.array([1.0]), 1.0)], [])[0]
    sig_cx = 0.6937e-14 * (1 - 0.155 * np.log10(e)) ** 2 / (1 + 0.1112e-14 * e ** 3.3) * 1e-4   # eq. B7, m^2
    sig_p_cm2 = (lam - sig_cx) * 1e4
    assert 1.45e-16 / 1.5 < sig_p_cm2 < 1.45e-16 * 1.5


def _given_beam():
    shape = np.exp(-(RHO / 0.35) ** 2)
    dep = {"rho": RHO, "shape": shape, "xi0": XI0, "shine_through": [0.0, 0.0, 0.0]}
    return NeutralBeam(name="nb", power=2.0e6, energy=80e3, A=2.0, Z=1.0, direction=1,
                       power_fractions=(0.7, 0.2, 0.1), deposition=dep)


def _compute(**kw):
    model = SimpleBeamModel([_given_beam()], smoothing_width=0.0, **kw)
    return model, model.compute(RHO, time=0.0, ne=NE, te_ev=TE, ions=IONS, zeff=ZEFF, ft=FT, B0=2.0, R0=R0,
                                dvol_drho=DV)


def test_nbcd_is_off_by_default_and_banana_shift_declared():
    _, s = _compute()
    assert len(s.jdotb) == 0 and not s.not_validated
    assert s.provenance.inputs["nbcd"].startswith("not computed")
    assert any("banana" in d for d in bm.FUSE_DIFFERENCES)
    assert any("banana" in d for d in s.provenance.limitations)
    assert s.p_fast and s.n_fast                         # fast-ion pressure and density still returned


def test_nbcd_opt_in_is_flagged_and_refused_until_acknowledged(tmp_path):
    model, s = _compute(compute_nbcd=True)
    assert "jdotb:nb" in s.not_validated and s.provenance.limitations[0].startswith("NOT VALIDATED")
    for read in (lambda: s.jdotb["nb"], lambda: s.jdotb_total, lambda: s.j_parallel(), lambda: list(s.jdotb.items()),
                 lambda: s.jdotb.get("nb")):
        with pytest.raises(NotValidatedError):
            read()
    assert "nb" in s.jdotb                               # existence is visible
    assert any("NOT validated" in g for g in s.completeness())
    j = s.acknowledge_not_validated(["nb"], reason="test")["nb"]
    assert s.meta["acknowledged_not_validated"][0]["source"] == "nb"
    # the flag survives serialization
    path = tmp_path / "src.json"
    s.to_json(path)
    back = NonInductiveSources.from_json(path)
    with pytest.raises(NotValidatedError):
        back.jdotb["nb"]
    # absolute value against an independent computation: births from the given deposition, the Gaffey current
    # by quadrature, the shielding with the independent L31
    shape = np.exp(-(RHO / 0.35) ** 2)
    norm = bm.volume_integral(RHO, DV, shape)
    b = model.beams[0]
    ref = np.zeros_like(RHO)
    for k, Ek in enumerate(b.energies):
        S = shape / norm * b.particle_rates[k]
        m = classical_moments(S, Ek, NE, TE, IONS, 2.0, 1.0, xi0=XI0)
        v0 = np.sqrt(2 * Ek * E_CHARGE / (2.0 * M_AMU))
        w, rd, tau = float(m["w"][0]), float(m["R_D"][0]), float(m["tau_s"][0])
        ref += E_CHARGE * S * tau * v0 * XI0 * _g_int(w, rd / 3.0, 3)
    L31 = redl_coefficients(FT, ZEFF, np.zeros_like(FT), np.zeros_like(FT))["L31"]
    F = 1.0 - (1.0 / ZEFF) * (1.0 - L31)
    ref = ref * F * 2.0
    sel = shape > 1e-6
    np.testing.assert_allclose(j[sel], ref[sel], rtol=1e-6)


def test_zeff_inconsistent_with_ions_is_recorded_or_refused():
    _, s = _compute()
    assert s.provenance.inputs["zeff_vs_ions_max_rel_difference"] == pytest.approx(0.0, abs=1e-12)
    model = SimpleBeamModel([_given_beam()], smoothing_width=0.0)
    s = model.compute(RHO, time=0.0, ne=NE, te_ev=TE, ions=IONS, zeff=ZEFF * 1.2, ft=FT, B0=2.0, R0=R0, dvol_drho=DV)
    assert s.provenance.inputs["zeff_vs_ions_max_rel_difference"] == pytest.approx(abs(1 / 1.2 - 1), abs=1e-12)
    with pytest.raises(ValueError, match="Z_eff"):
        SimpleBeamModel([_given_beam()], smoothing_width=0.0, zeff_check_rtol=0.01).compute(
            RHO, time=0.0, ne=NE, te_ev=TE, ions=IONS, zeff=ZEFF * 1.2, ft=FT, B0=2.0, R0=R0, dvol_drho=DV)
