"""Conventions, frames, the array provider's refusals and the provider container (synthetic only)."""
import dataclasses

import numpy as np
import pytest

from bouquet.kinrecon.current.geometry import FluxSurfaceGeometry
from bouquet.kinrecon.sources import (AmbiguousConventionError, NonInductiveSources, Provenance, ScalePrior,
                                      ShapePrior, from_arrays)
from bouquet.kinrecon.sources import conventions as cv

RHO = np.linspace(0.0, 1.0, 51)
B0, R0, A = 2.0, 1.7, 0.6


def _geo_cyl():
    g = FluxSurfaceGeometry.cylinder(A, R0, B0, n=51)
    return dataclasses.replace(g, avg_b2=np.full(51, B0 ** 2))


def _geo_torus(bp2=0.02):
    """Circular torus with a stated poloidal-field contribution to <B^2> (geometric test case)."""
    g = FluxSurfaceGeometry.circular(A, R0, B0, n=51)
    return dataclasses.replace(g, avg_b2=g.fpol ** 2 * g.avg_inv_r2 * (1 + bp2 * RHO ** 2))


def _prov():
    return Provenance(provider="test", source_code="synthetic", source_version="0", inputs={}, conventions={},
                      settings={}, assumptions=["a"], limitations=["l"])


def test_pressure_convention_table():
    p_par, p_perp = np.array([3.0, 6.0]), np.array([1.5, 0.0])
    iso = cv.p_fast_from_true(p_par, p_perp)
    np.testing.assert_allclose(iso, [2.0, 2.0])
    c_par, c_perp = cv.imas_components_from_true(p_par, p_perp)
    np.testing.assert_allclose(cv.p_fast_from_imas_components(c_par, c_perp), iso)       # components = true/3
    np.testing.assert_allclose(cv.p_fast_from_energy_density(1.5 * iso), iso)          # W = (3/2) p
    c1, c2 = cv.imas_components_from_isotropic(iso)
    np.testing.assert_allclose(c1 + 2 * c2, iso)


def test_frames_cylinder_exact():
    g = _geo_cyl()
    jb = 1e6 * (1 - RHO ** 2)
    fr = cv.toroidal_frames(jb, RHO, g)
    np.testing.assert_allclose(fr["j_r0"], jb / B0, rtol=1e-12)      # in a cylinder j_R0 = <J.B>/B0
    np.testing.assert_allclose(fr["j_tor"], jb / B0, rtol=1e-12)
    # enclosed current vs the analytic integral: <J_phi/R> = jb/(B0 R0); dV = 4 pi^2 R0 a^2 rho drho
    I_ref = (1e6 / (B0 * R0)) * 4 * np.pi ** 2 * R0 * A ** 2 / (2 * np.pi) * (RHO ** 2 / 2 - RHO ** 4 / 4)
    np.testing.assert_allclose(fr["current_enclosed"], I_ref, rtol=2e-3, atol=1e-6 * I_ref[-1])


def test_frames_torus_roundtrip_and_ratio():
    g = _geo_torus(0.03)
    jb = 5e5 * np.exp(-((RHO - 0.3) / 0.15) ** 2)
    jr = cv.jphi_over_r_from_jdotb(jb, RHO, g)
    np.testing.assert_allclose(cv.jdotb_from_jphi_over_r(jr, RHO, g), jb, rtol=1e-12)
    # <J.B>/B0 over R0<J_phi/R> = <B^2>/(B0 R0 F <1/R^2>) = 1 + Bp term here
    ratio = cv.jpar_from_jdotb(jb, B0) / (R0 * jr)
    np.testing.assert_allclose(ratio, 1 + 0.03 * RHO ** 2, rtol=1e-9)


def test_rho_psi_n_mapping():
    psi_tab = np.linspace(0, 1, 41)
    rho_tab = np.sqrt(psi_tab) ** 0.9
    r = cv.psi_n_to_rho([0.0, 0.25, 1.0], psi_tab, rho_tab)
    np.testing.assert_allclose(cv.rho_to_psi_n(r, psi_tab, rho_tab), [0.0, 0.25, 1.0], atol=1e-6)
    with pytest.raises(ValueError):
        cv.psi_n_to_rho(0.5, psi_tab[::-1], rho_tab)


def _base(**kw):
    d = dict(radial_coordinate="rho_tor_norm", time=1.0, time_window=0.0, B0=B0, R0=R0,
             source_code="synthetic", source_version="0")
    d.update(kw)
    return d


def test_from_arrays_refuses_undeclared_conventions():
    j = {"nbi": 1e5 * (1 - RHO ** 2)}
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(current=j))                                           # no frame
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(current=j, current_frame="jpar_b0", current_sign="relative_to_ip"))   # no units
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(current=j, current_frame="jpar_b0", current_units="A/m^2"))           # no sign
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(current=j, current_frame="j_r0", current_units="A/m^2",
                                 current_sign="relative_to_ip"))                       # toroidal frame, no geometry
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(current=j, current_frame="jpar_b0", current_units="A/m^2", current_sign="absolute"))
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(pressure={"D": np.ones(51)}))                         # no pressure form
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(pressure={"D": np.ones(51)}, pressure_form="imas_components", pressure_units="Pa"))
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(density={"D": np.ones(51)}))                          # no density unit
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(pressure={"D": np.ones(51)}, pressure_form="isotropic_equivalent",
                                 pressure_units="psi"))
    with pytest.raises(AmbiguousConventionError):
        from_arrays(RHO, **_base(radial_coordinate="psi_n", density={"D": np.ones(51)}, density_units="m^-3"))


def test_from_arrays_frames_units_signs():
    g = _geo_torus(0.02)
    j = 1e5 * (1 - RHO ** 2)
    s1 = from_arrays(RHO, **_base(current={"nbi": j}, current_frame="j_r0", current_units="A/m^2",
                                  current_sign="relative_to_ip", geometry=g))
    np.testing.assert_allclose(R0 * cv.jphi_over_r_from_jdotb(s1.jdotb["nbi"], RHO, g), j, rtol=1e-12)
    s2 = from_arrays(RHO, **_base(current={"nbi": j / 1e3}, current_frame="j_r0", current_units="kA/m^2",
                                  current_sign="relative_to_ip", frame_approximation="large_aspect_ratio"))
    np.testing.assert_allclose(s2.jdotb["nbi"], B0 * j, rtol=1e-12)
    assert any("large-aspect-ratio" in a for a in s2.provenance.assumptions)
    # absolute signs: Ip < 0, Bt > 0, co-Ip toroidal current is negative in the producer frame
    s3 = from_arrays(RHO, **_base(current={"nbi": -j}, current_frame="j_r0", current_units="A/m^2",
                                  current_sign="absolute", ip_sign=-1, bt_sign=1, geometry=g))
    np.testing.assert_allclose(s3.jdotb["nbi"], s1.jdotb["nbi"], rtol=1e-12)
    # <J.B> of co-Ip current has sign ip*bt
    s4 = from_arrays(RHO, **_base(current={"x": -B0 * j}, current_frame="jdotb", current_units="A.T/m^2",
                                  current_sign="absolute", ip_sign=1, bt_sign=-1))
    np.testing.assert_allclose(s4.jdotb["x"], B0 * j)
    # keV/m^3 energy density -> Pa: p = (2/3) W
    W_kev = np.full(51, 1e19)
    s5 = from_arrays(RHO, **_base(pressure={"D": W_kev}, pressure_form="energy_density", pressure_units="keV/m^3"))
    np.testing.assert_allclose(s5.p_fast["D"], (2 / 3) * 1e19 * 1.602176634e-16)


def test_from_arrays_negative_policy():
    p = np.linspace(-0.01, 1.0, 51)
    with pytest.raises(ValueError):
        from_arrays(RHO, **_base(pressure={"D": p}, pressure_form="isotropic_equivalent", pressure_units="kPa"))
    s = from_arrays(RHO, **_base(pressure={"D": p}, pressure_form="isotropic_equivalent", pressure_units="kPa",
                                 negative_policy="clip"))
    assert s.p_fast["D"].min() == 0 and s.provenance.settings["clipped"]


def test_container_validation_and_completeness():
    p_par, p_perp = np.full(51, 3.0), np.full(51, 1.5)
    s = NonInductiveSources(rho=RHO, time=1.0, time_window=0.0, B0=B0, R0=R0, jdotb={"nbi": np.ones(51)},
                            p_fast={"D": np.full(51, 2.0)}, n_fast={"D": np.ones(51)}, p_par={"D": p_par},
                            p_perp={"D": p_perp}, provenance=_prov())
    assert any("no uncertainty" in g for g in s.completeness())
    with pytest.raises(ValueError):
        s.require_complete()
    s.uncertainty = [ScalePrior("jdotb", "*", 1.0, 0.2, "test", "proposed"),
                     ScalePrior("p_fast", "D", 1.0, 0.15, "test", "proposed")]
    assert s.completeness() == ["no scale prior for n_fast['D']"]
    s.uncertainty.append(ScalePrior("n_fast", "*", 1.0, 0.15, "test", "proposed"))
    s.uncertainty.append(ShapePrior("jdotb", "nbi", "shift", 0.05, "test", "proposed"))
    s.require_complete()
    s2 = NonInductiveSources.from_dict(s.to_dict())
    np.testing.assert_allclose(s2.p_par["D"], p_par)
    assert len(s2.uncertainty) == 4 and s2.provenance.source_code == "synthetic"
    with pytest.raises(ValueError):          # true components inconsistent with p_fast
        NonInductiveSources(rho=RHO, time=1.0, time_window=0.0, B0=B0, R0=R0, jdotb={}, p_fast={"D": np.ones(51)},
                            n_fast={}, p_par={"D": p_par}, p_perp={"D": p_perp}, provenance=_prov())
    with pytest.raises(ValueError):
        NonInductiveSources(rho=RHO, time=1.0, time_window=0.0, B0=-B0, R0=R0, jdotb={}, p_fast={}, n_fast={},
                            provenance=_prov())
    with pytest.raises(ValueError):
        ScalePrior("p_fast", "*", 1.0, 0.2, "", "proposed")            # basis required
    with pytest.raises(ValueError):
        Provenance(provider="", source_code="x", source_version="0", inputs={}, conventions={}, settings={},
                   assumptions=[], limitations=[])


def test_on_psi_n_resampling():
    s = NonInductiveSources(rho=RHO, time=1.0, time_window=0.0, B0=B0, R0=R0, jdotb={}, p_fast={"D": 1 - RHO ** 2},
                            n_fast={}, provenance=_prov())
    psi_tab = np.linspace(0, 1, 101)
    rho_tab = np.sqrt(psi_tab)
    out = s.on_psi_n(np.array([0.0, 0.25, 0.81]), psi_tab, rho_tab)
    np.testing.assert_allclose(out, [1.0, 0.75, 0.19], atol=2e-3)


def test_proposed_priors_are_explicit_and_complete():
    from bouquet.kinrecon.sources import PRIOR_KINDS, proposed_priors
    for kind in PRIOR_KINDS:
        pri = proposed_priors(kind)
        assert pri and all(p.status == "proposed" and p.basis for p in pri)
    s = NonInductiveSources(rho=RHO, time=1.0, time_window=0.0, B0=B0, R0=R0, jdotb={"nbi": np.ones(51)},
                            p_fast={"D": np.ones(51)}, n_fast={"D": np.ones(51)}, provenance=_prov(),
                            uncertainty=proposed_priors("simple_beam"))
    assert s.completeness() == []
    with pytest.raises(ValueError):
        proposed_priors("default")
