"""Built-in beam model options: finite beams from beamlets (beam_optics) and fast-ion trapping (trapping).

Synthetic circular geometry and synthetic beam optics only (round numbers, no device description).  The
defaults (pencil deposition, no trapping) are checked against values pinned at the commit before the options
existed, and against an explicit-default construction bit for bit.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from bouquet.kinrecon.sources import (Aperture, BeamOptics, DivergenceComponent, FluxSurfaceField, IonSpecies,
                                      NeutralBeam, SimpleBeamModel, optics_from_ids)
from bouquet.kinrecon.sources import beam as bm
from bouquet.kinrecon.sources import beam_optics as bo
from bouquet.kinrecon.sources import trapping as tr
from bouquet.kinrecon.sources.conventions import E_CHARGE

R0, A, B0, Q = 1.7, 0.6, 2.0, 1.5
RHO = np.linspace(0.0, 1.0, 101)
DV = 4 * np.pi ** 2 * R0 * A ** 2 * RHO          # dV/drho of concentric circles (rho = r/a)
THETA = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)   # includes theta = pi (inboard midplane, B_max)


def rho_of_rz(R, Z):
    r = np.sqrt((R - R0) ** 2 + Z ** 2) / A
    return np.where(r <= 1.0, r, np.nan)


def b_of_rz(R, Z):
    return B0 * R0 / np.asarray(R, float)        # vacuum toroidal field: the large-aspect-ratio |B|


def _field(n_surf=200):
    rs = np.linspace(0.005, 1.0, n_surf)
    Bs, ws = [], []
    for x in rs:
        R = R0 + x * A * np.cos(THETA)
        Bs.append(B0 * R0 / R)
        bp = B0 * (x * A) / (Q * R)              # circular, constant q: B_p = B0 r / (q R)
        ws.append(x * A * (THETA[1] - THETA[0]) / bp)   # dl / B_p
    return FluxSurfaceField(rs, Bs, ws, b_of_rz)


def _plasma():
    ne, te = 5e19 * (1 - 0.8 * RHO ** 2), 3000.0 * (1 - 0.9 * RHO ** 2) + 50
    nc = 0.015 * ne
    ions = [IonSpecies("D", 2.0, 1.0, ne - 6 * nc), IonSpecies("C", 12.0, 6.0, nc)]
    return ne, te, ions, (ions[0].n + 36 * nc) / ne


def _beam(**kw):
    d = dict(name="b1", power=2.0e6, energy=80e3, A=2.0, Z=1.0, direction=1, power_fractions=(0.7, 0.2, 0.1),
             tangency_radius=1.2, source_r=5.0, source_z=0.0, vertical_angle=0.0)
    d.update(kw)
    return NeutralBeam(**d)


def _optics(**kw):
    d = dict(width_h=0.12, width_v=0.40, focal_h=None, focal_v=8.0,
             divergence=(DivergenceComponent(1.0, 0.01, 0.02),))
    d.update(kw)
    return BeamOptics(**d)


def _hyd_imp(ions):
    return [(ions[0].n, 1.0)], [(ions[1].n, 6.0)]


def _compute(model, **kw):
    ne, te, ions, zeff = _plasma()
    ft = 1.46 * np.sqrt(A * RHO / R0)
    return model.compute(RHO, time=1.0, ne=ne, te_ev=te, ions=ions, zeff=zeff, ft=ft, B0=B0, R0=R0, dvol_drho=DV,
                         rho_of_rz=rho_of_rz, **kw)


# --------------------------------------------------------------------------- defaults unchanged
# Values of the default model (pencil, no trapping) at the commit before the options existed: the second beam
# set of test_sources_beam.test_model_energy_conservation_and_outputs (smoothing 0.06, NBCD on).
PINNED_DEFAULT = {"p_fast_axis_Pa": 42522.52015408095, "W_fast_J": 100029.13059761471,
                  "I_cd_b1_A": 119850.57319413946, "I_cd_b2_A": -39954.60756356674}


def _default_numbers(model):
    s = _compute(model)
    j = s.acknowledge_not_validated("*", reason="pinned default values in a unit test")
    pf = s.p_fast["beam_A2_Z1"]
    return s, {"p_fast_axis_Pa": float(pf[0]), "W_fast_J": float(1.5 * bm.volume_integral(RHO, DV, pf)),
               "I_cd_b1_A": float(bm.volume_integral(RHO, DV, j["b1"] / B0) / (2 * np.pi * R0)),
               "I_cd_b2_A": float(bm.volume_integral(RHO, DV, j["b2"] / B0) / (2 * np.pi * R0))}


def _two_beams():
    return [_beam(), _beam(name="b2", direction=-1, tangency_radius=0.8, power=1e6)]


def test_defaults_match_the_values_before_the_options():
    _, got = _default_numbers(SimpleBeamModel(_two_beams(), smoothing_width=0.06, compute_nbcd=True))
    for k, v in PINNED_DEFAULT.items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=0.0), k


def test_explicit_defaults_are_bitwise_identical_and_add_no_record_keys():
    s0, _ = _default_numbers(SimpleBeamModel(_two_beams(), smoothing_width=0.06, compute_nbcd=True))
    s1, _ = _default_numbers(SimpleBeamModel(_two_beams(), smoothing_width=0.06, compute_nbcd=True,
                                             deposition="pencil", trapping="none"))
    # a field given with trapping off changes nothing
    s2 = _compute(SimpleBeamModel(_two_beams(), smoothing_width=0.06, compute_nbcd=True), field=_field())
    s2.acknowledge_not_validated("*", reason="unit test")
    for s in (s1, s2):
        for a, b in ((s0.p_fast, s.p_fast), (s0.n_fast, s.n_fast), (s0.p_par, s.p_par)):
            for k in a:
                assert np.array_equal(a[k], b[k])
        for k in s0.jdotb.raw():
            assert np.array_equal(s0.jdotb.raw()[k], s.jdotb.raw()[k])
        assert s.provenance.settings == s0.provenance.settings
        assert s.provenance.limitations == s0.provenance.limitations
        assert set(s.meta) == set(s0.meta) and s.meta["beams"] == s0.meta["beams"]
        assert s.not_validated == s0.not_validated
    for key in ("deposition", "n_beamlets", "trapping"):
        assert key not in s0.provenance.settings
    with pytest.raises(ValueError):
        SimpleBeamModel(_two_beams(), smoothing_width=0.06, n_beamlets=8)            # option args without the option
    with pytest.raises(ValueError):
        SimpleBeamModel(_two_beams(), smoothing_width=0.06, deposition="beamlets")   # no n_beamlets
    with pytest.raises(ValueError):
        SimpleBeamModel(_two_beams(), smoothing_width=0.06, trapping="banana")


# --------------------------------------------------------------------------- beamlets
def test_halton_is_deterministic_and_known():
    h = bo.halton(4)
    np.testing.assert_array_equal(h[:, 0], [0.5, 0.25, 0.75, 0.125])
    np.testing.assert_allclose(h[:, 1], [1 / 3, 2 / 3, 1 / 9, 4 / 9], rtol=1e-15)
    a, b = bo.beamlet_sample(_optics(), 64), bo.beamlet_sample(_optics(), 64)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_single_point_beamlet_equals_the_pencil():
    ne, te, ions, _ = _plasma()
    hyd, imp = _hyd_imp(ions)
    b = _beam()
    point = BeamOptics(width_h=0.0, width_v=0.0, focal_h=None, focal_v=None, divergence=())
    assert any("divergence missing" in m for m in point.missing)
    for width in (0.0, 0.06):
        p = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, hyd, imp, smoothing_width=width)
        q = bm.beamlet_deposition(b, point, rho_of_rz, RHO, DV, ne, te, hyd, imp, n_beamlets=1, smoothing_width=width)
        np.testing.assert_allclose(q.births, p.births, rtol=1e-10, atol=1e-10 * p.births.max())
        np.testing.assert_allclose(q.shine_through, p.shine_through, rtol=1e-10)
        live = p.births > 1e-6 * p.births.max()
        np.testing.assert_allclose(q.xi0[live], p.xi0[live], rtol=1e-10)


def _axis_value(n, width=0.0, optics=None):
    ne, te, ions, _ = _plasma()
    hyd, imp = _hyd_imp(ions)
    dep = bm.beamlet_deposition(_beam(), optics or _optics(), rho_of_rz, RHO, DV, ne, te, hyd, imp, n_beamlets=n,
                                smoothing_width=width)
    return dep


def test_beam_through_the_axis_has_a_finite_axis_density_that_converges_and_conserves():
    ne, te, ions, _ = _plasma()
    hyd, imp = _hyd_imp(ions)
    b = _beam()                                    # midplane beam, R_t < R0: crosses the magnetic axis
    pencil = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, hyd, imp, smoothing_width=0.0)
    # without smoothing the pencil's line through the axis piles births up in the axis cell; the finite beam does not
    assert _axis_value(256, width=0.0).births.sum(0)[0] < 0.1 * pencil.births.sum(0)[0]
    ns = (128, 256, 512, 1024)
    axis = []
    for n in ns:
        dep = _axis_value(n, width=0.05)
        axis.append(dep.births.sum(0)[0])
        for k in range(3):                         # particle rate conserved: births + shine-through = launched
            born = bm.volume_integral(RHO, DV, dep.births[k])
            assert born + dep.shine_through[k] * b.particle_rates[k] == pytest.approx(b.particle_rates[k], rel=1e-9)
        # power conserved: deposited power = launched power minus the shine-through
        p_dep = sum(bm.volume_integral(RHO, DV, dep.births[k]) * b.energies[k] * E_CHARGE for k in range(3))
        p_st = sum(dep.shine_through[k] * b.particle_rates[k] * b.energies[k] * E_CHARGE for k in range(3))
        assert p_dep + p_st == pytest.approx(b.power, rel=1e-9)
    axis = np.array(axis)
    assert np.all(np.isfinite(axis)) and np.all(axis > 0)
    change = np.abs(np.diff(axis)) / axis[1:]
    assert change[-1] < 0.02                       # doubling 512 -> 1024 moves the axis value by < 2 %
    assert change[-1] < change[0]
    p05 = bm.pencil_beam_deposition(b, rho_of_rz, RHO, DV, ne, te, hyd, imp, smoothing_width=0.05)
    assert axis[-1] < 0.5 * p05.births.sum(0)[0]   # still below the smoothed pencil's axis value


def test_aperture_clipping_partial_and_full():
    ne, te, ions, _ = _plasma()
    hyd, imp = _hyd_imp(ions)
    b = _beam()
    narrow = _optics(apertures=(Aperture(distance=2.0, shape="rectangle", half_width_h=0.05, half_width_v=0.10),))
    dep = bm.beamlet_deposition(b, narrow, rho_of_rz, RHO, DV, ne, te, hyd, imp, n_beamlets=256, smoothing_width=0.05)
    tr_ = dep.info["transmission"]
    assert 0.0 < tr_ < 1.0
    for k in range(3):        # the launched rate is shared by the transmitted beamlets
        born = bm.volume_integral(RHO, DV, dep.births[k])
        assert born + dep.shine_through[k] * b.particle_rates[k] == pytest.approx(b.particle_rates[k], rel=1e-9)
    wide = _axis_value(256, width=0.05)
    assert dep.births.sum(0)[0] > wide.births.sum(0)[0]      # clipping the halo narrows the beam: more on axis
    blocked = _optics(apertures=(Aperture(distance=2.0, shape="circle", radius=0.01, offset_v=1.0),))
    dep = bm.beamlet_deposition(b, blocked, rho_of_rz, RHO, DV, ne, te, hyd, imp, n_beamlets=64)
    assert dep.info["transmission"] == 0.0 and dep.info["n_transmitted"] == 0
    assert not dep.births.any() and not dep.shine_through.any()
    model = SimpleBeamModel([b], smoothing_width=0.0, deposition="beamlets", optics={"b1": blocked}, n_beamlets=64)
    s = _compute(model)
    assert not s.p_fast["beam_A2_Z1"].any() and s.meta["beams"]["b1"]["births_per_s"] == [0.0, 0.0, 0.0]


def test_model_beamlets_records_missing_inputs_and_is_deterministic():
    beams = _two_beams()
    opt = {"b1": BeamOptics(width_h=0.12, width_v=None, focal_h=None, focal_v=8.0, divergence=())}
    kw = dict(smoothing_width=0.05, compute_nbcd=True, deposition="beamlets", optics=opt, n_beamlets=32)
    s1 = _compute(SimpleBeamModel(beams, **kw))
    s2 = _compute(SimpleBeamModel(beams, **kw))
    assert np.array_equal(s1.p_fast["beam_A2_Z1"], s2.p_fast["beam_A2_Z1"])
    j1 = s1.acknowledge_not_validated("*", reason="unit test")
    j2 = s2.acknowledge_not_validated("*", reason="unit test")
    assert np.array_equal(j1["b1"], j2["b1"])
    m = s1.meta["beams"]
    assert m["b1"]["deposition"] == "beamlets"
    missing = m["b1"]["beamlets"]["optics"]["missing"]
    assert any("source height missing" in x for x in missing) and any("divergence missing" in x for x in missing)
    assert any("apertures: none given" in x for x in missing)
    assert m["b2"]["deposition"].startswith("pencil: no finite-beam description")
    st = s1.provenance.settings
    assert st["deposition"] == "beamlets" and st["n_beamlets"] == 32
    assert s1.provenance.limitations[1] == bm.BEAMLET_LIMITATION
    # heating integrates to the absorbed power in beamlet mode too
    for b in beams:
        qe, qi = s1.heating[b.name]
        absorbed = np.sum(np.array(m[b.name]["births_per_s"]) * b.energies * E_CHARGE)
        assert bm.volume_integral(RHO, DV, qe + qi) == pytest.approx(absorbed, rel=1e-9)


def test_optics_from_ids_places_apertures_and_states_missing_items():
    Rs, Rt = 5.0, 1.2
    group = {"width_horizontal": 0.12, "width_vertical": 0.40,
             "focus": {"focal_length_horizontal": 1e31, "focal_length_vertical": 8.0},
             "divergence_component": [{"particles_fraction": 1.0, "horizontal": 0.01, "vertical": 0.02}]}
    r_ap = 4.0
    c = bo.IMAS_APERTURE_CENTER                                                         # the schema's leaf name
    aps = [{c: {"r": r_ap, "z": 0.03, "phi": 0.3}, "geometry_type": 3, "x1_width": 0.2, "x2_width": 0.5},
           {c: {"r": 3.5, "z": 0.0, "phi": 0.4}},                                       # no dimensions
           {c: {"r": 3.0, "z": 0.0}, "geometry_type": 2, "radius": -9e40}]              # IMAS-empty radius
    o = optics_from_ids(group, aps, source_r=Rs, source_z=0.0, tangency_radius=Rt, vertical_angle=0.0)
    assert len(o.apertures) == 1
    a = o.apertures[0]
    s_exp = np.sqrt(Rs ** 2 - Rt ** 2) - np.sqrt(r_ap ** 2 - Rt ** 2)      # midplane chord to radius r_ap
    assert a.distance == pytest.approx(s_exp, rel=1e-12)
    assert a.half_width_h == pytest.approx(0.1) and a.half_width_v == pytest.approx(0.25)
    assert a.offset_v == pytest.approx(0.03) and a.offset_h == 0.0
    assert any("aperture 1" in m and "not applied" in m for m in o.missing)
    assert any("aperture 2" in m and "not applied" in m for m in o.missing)
    assert any("horizontal center offset taken as 0" in m for m in o.missing)
    o2 = optics_from_ids({"width_horizontal": 0.12}, None, source_r=Rs, source_z=0.0, tangency_radius=Rt,
                         vertical_angle=0.0)
    assert o2.width_v is None and o2.divergence == () and o2.apertures == ()
    assert any("apertures: none given" in m for m in o2.missing)


# --------------------------------------------------------------------------- trapping
def test_trapped_fraction_matches_the_large_aspect_ratio_limit_and_vanishes_on_axis():
    f = _field()
    xi = np.linspace(-1, 1, 40001)
    xi = 0.5 * (xi[1:] + xi[:-1])                  # isotropic pitch: uniform in xi
    fracs = []
    for x in (0.005, 0.02, 0.1, 0.4, 1.0):
        i = int(np.argmin(np.abs(f.rho - x)))
        rho = f.rho[i]
        eps = rho * A / R0
        R = np.full(xi.size, R0 + rho * A)         # outboard midplane: B_min of the surface
        trapped, xi_cd = tr.classify_births(f, np.full(xi.size, rho), R, np.zeros(xi.size), xi)
        frac = trapped.mean()
        assert frac == pytest.approx(np.sqrt(2 * eps / (1 + eps)), abs=2e-4)                 # T5, exact for B ~ 1/R
        assert frac == pytest.approx(float(tr.isotropic_trapped_fraction(b_of_rz(R[0], 0.0), f.B_max[i])), abs=2e-4)
        assert np.all(xi_cd[trapped] == 0.0) and np.all(np.sign(xi_cd[~trapped]) == np.sign(xi[~trapped]))
        fracs.append(frac)
    assert np.all(np.diff(fracs) > 0) and fracs[0] < 0.07              # goes to zero toward the axis
    eps = 0.005 * A / R0
    assert fracs[0] / np.sqrt(2 * eps) == pytest.approx(1.0, abs=0.01)   # large-aspect-ratio limit sqrt(2 eps)


def test_transit_averaged_pitch_against_quadrature():
    f = _field()
    i = int(np.argmin(np.abs(f.rho - 0.5)))
    rho = f.rho[i]
    r = rho * A
    for xi0 in (0.95, 0.8, 0.6):
        Rb = R0 + r                                 # outboard midplane birth
        lam = (1 - xi0 ** 2) / (B0 * R0 / Rb)
        if lam * B0 * R0 / (R0 - r) >= 1:
            continue
        Bt = lambda t: B0 * R0 / (R0 + r * np.cos(t))  # noqa: E731
        wt = lambda t: (R0 + r * np.cos(t))            # dl / B_p up to a constant  # noqa: E731
        num = quad(lambda t: Bt(t) * wt(t), 0, 2 * np.pi)[0]
        den = quad(lambda t: Bt(t) * wt(t) / np.sqrt(1 - lam * Bt(t)), 0, 2 * np.pi)[0]
        _, xc = tr.classify_births(f, [rho], [Rb], [0.0], [xi0])
        assert xc[0] == pytest.approx(num / den, rel=1e-6)
        assert xc[0] < xi0                           # the outboard birth pitch overstates the transit average


def test_model_trapping_changes_only_the_current_and_flags_counter_injection():
    beams = _two_beams()
    off = _compute(SimpleBeamModel(beams, smoothing_width=0.06, compute_nbcd=True))
    on_model = SimpleBeamModel(beams, smoothing_width=0.06, compute_nbcd=True, trapping="zero_orbit_width")
    with pytest.raises(ValueError, match="field"):
        _compute(on_model)
    on = _compute(on_model, field=_field())
    assert np.array_equal(off.p_fast["beam_A2_Z1"], on.p_fast["beam_A2_Z1"])       # pressure unaffected
    assert np.array_equal(off.n_fast["beam_A2_Z1"], on.n_fast["beam_A2_Z1"])
    jo = off.acknowledge_not_validated("*", reason="unit test")
    jn = on.acknowledge_not_validated("*", reason="unit test")
    I = lambda j: bm.volume_integral(RHO, DV, j)  # noqa: E731
    assert 0 < I(jn["b1"]) < I(jo["b1"])                   # co beam: trapped births removed
    tf = on.meta["beams"]["b1"]["trapped_birth_fraction"]
    assert all(0.0 < x < 1.0 for x in tf)
    assert on.not_validated["jdotb:b1"] == SimpleBeamModel.NBCD_NOT_VALIDATED_TRAPPING
    assert on.meta["counter_injection"]["flag"] and on.meta["counter_injection"]["counter_power_W"] == 1e6
    assert any(x.startswith("counter-injected power present") for x in on.provenance.limitations)
    assert bm.TRAPPING_LIMITATION in on.provenance.limitations
    assert "trapping" in on.provenance.settings
    co_only = _compute(SimpleBeamModel([_beam()], smoothing_width=0.06, compute_nbcd=True,
                                       trapping="zero_orbit_width"), field=_field())
    assert not co_only.meta["counter_injection"]["flag"]
    # both options together, deterministic
    kw = dict(smoothing_width=0.06, compute_nbcd=True, trapping="zero_orbit_width", deposition="beamlets",
              optics={"b1": _optics()}, n_beamlets=32)
    a = _compute(SimpleBeamModel([_beam()], **kw), field=_field())
    b = _compute(SimpleBeamModel([_beam()], **kw), field=_field())
    ja = a.acknowledge_not_validated("*", reason="unit test")["b1"]
    jb = b.acknowledge_not_validated("*", reason="unit test")["b1"]
    assert np.array_equal(ja, jb) and np.all(np.isfinite(ja))
    assert 0.0 < a.meta["beams"]["b1"]["trapped_birth_fraction"][0] < 1.0
