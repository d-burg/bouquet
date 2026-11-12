"""EFIT stored-array adapter: weight convention, sigma models, flux gauge, refusals, EFIT's own chi^2.

The rules under test are documented in ``bouquet.kinrecon.adapters.efit_tree``.  Synthetic numbers only.
"""
import os
import tempfile

import numpy as np
import pytest

from bouquet.kinrecon.adapters import efit_tree as ET
from bouquet.kinrecon.adapters.efit_kfile import kersil_loop_sigma
from bouquet.kinrecon.measurements import MachineBundle, read_measurements, write_measurements

TWO_PI = 2.0 * np.pi


def _machine():
    sensors = {"flux_loop": {"labels": ["L0", "L1", "L2", "L3"], "r": [1.0, 1.1, 1.2, 2.3],
                             "z": [0.0, 0.5, -0.5, 0.0]},
               "bpol_probe": {"labels": ["P0", "P1", "P2"], "r": [1.0, 2.4, 1.7], "z": [0.0, 0.0, 1.3],
                              "direction_r": [0.0, 0.0, 1.0], "direction_z": [1.0, 1.0, 0.0]},
               "pf_current": {"labels": ["F0", "F1", "E0"]}}
    return MachineBundle(machine="toy", sensors=sensors, channel_maps={"BRSP": ["F0", "F1"], "ECURRT": ["E0"]})


def _arrays(chi2_form="inverse_sigma"):
    rng = np.random.default_rng(3)
    sig_l = np.array([1e-3, 2e-3, 3e-3, 4e-3])
    sil = np.array([2e-5, 0.10, -0.05, 0.20])             # relative storage: reference entry near 0
    csil = sil + rng.normal(0, 1, 4) * sig_l
    fwtsi = np.array([1.0 / sig_l[0], 1.0 / sig_l[1], 0.0, 1.0 / sig_l[3]])   # L2 not fitted
    sig_p = np.array([0.01, 0.02, 0.015])
    exp = np.array([0.3, -0.2, 0.1])
    cmp_ = exp + np.array([0.5, -1.0, 2.0]) * sig_p
    fwtp = 1.0 / sig_p
    sig_f = np.array([500.0, 800.0])
    fc = np.array([1.0e4, -2.0e4])
    r = lambda m, c: m - c  # noqa: E731
    if chi2_form == "inverse_sigma":
        saisil = (fwtsi * r(sil, csil)) ** 2
        saimpi = (fwtp * r(exp, cmp_)) ** 2
    else:
        saisil = fwtsi * r(sil, csil) ** 2
        saimpi = fwtp * r(exp, cmp_) ** 2
    return dict(SILOPT=sil, CSILOP=csil, FWTSI=fwtsi, SAISIL=saisil, EXPMPI=exp, CMPR2=cmp_, FWTMP2=fwtp,
                SAIMPI=saimpi, FCCURT=fc, CCBRSP=fc + 100.0, FWTFC=1.0 / sig_f, ECCURT=np.array([3000.0]),
                CECURR=np.array([3000.0]), FWTEC=np.array([0.0]), PLASMA=np.array([1.0e6]),
                CPASMA=np.array([0.999e6])), dict(loop=sig_l, probe=sig_p, fcoil=sig_f)


def _ms(arrays=None, **kw):
    a = arrays if arrays is not None else _arrays()[0]
    base = dict(sigma_model="weight_inverse", input_weight=1.0, flux_storage="relative", reference_flux=0.5,
                reference_convention="fitted", reference_flux_source="made-up", flux_reference="L0")
    base.update(kw)
    return ET.measurements_from_efit_arrays(a, _machine(), 1.5, 0.02, **base)


def test_weight_inverse_sigma_validity_and_fit_values():
    a, s = _arrays()
    ms = _ms(a)
    fl, pr = ms.groups["flux_loop"], ms.groups["bpol_probe"]
    np.testing.assert_allclose(fl.sigma[[0, 1, 3]], s["loop"][[0, 1, 3]] * TWO_PI, rtol=1e-14)
    assert np.isnan(fl.sigma[2]) and fl.valid.tolist() == [True, True, False, True]
    assert fl.meta["not_fitted_by_efit"] == ["L2"]
    np.testing.assert_allclose(fl.measured, -TWO_PI * a["SILOPT"], rtol=1e-15)
    np.testing.assert_allclose(fl.meta["source_fit"], -TWO_PI * a["CSILOP"], rtol=1e-15)
    np.testing.assert_allclose(pr.sigma, s["probe"], rtol=1e-14)
    assert fl.meta["weight_check"]["supports"] == "inverse_sigma"
    assert pr.meta["weight_check"]["frac_inverse_sigma"] == 1.0
    assert fl.weight.tolist() == [1.0] * 4
    assert ms.meta["sigma_models"]["flux_loop"] == "efit_arrays:weight_inverse"


def test_flux_gauge_is_stated_and_recorded():
    a, _ = _arrays()
    fl = _ms(a).groups["flux_loop"]
    assert fl.meta["coils_form"] == "relative" and fl.meta["coils_form_rule"] == "caller"
    assert fl.meta["absolute_offset"] == pytest.approx(-TWO_PI * 0.5, rel=1e-15)
    assert fl.meta["reference_convention"] == "fitted" and fl.meta["flux_reference"] == "L0"
    assert "2e-05" in fl.meta["coils_form_evidence"]
    fa = _ms(a, flux_storage="absolute", flux_reference=None).groups["flux_loop"]
    assert fa.meta["absolute_offset"] == 0.0
    with pytest.raises(ValueError, match="only with flux_storage='relative'"):
        _ms(a, flux_storage="absolute")
    with pytest.raises(ValueError, match="flux_storage must be stated"):
        _ms(a, flux_storage=None)
    with pytest.raises(ValueError, match="reference_flux"):
        _ms(a, reference_flux=None)
    with pytest.raises(ValueError, match="reference_convention"):
        _ms(a, reference_convention=None)


def test_weight_check_refuses_a_contradicted_convention_only_when_asked():
    a, _ = _arrays(chi2_form="inverse_variance")
    ms = _ms(a)                               # recorded, not refused
    chk = ms.groups["flux_loop"].meta["weight_check"]
    assert chk["supports"] == "inverse_variance" and chk["frac_inverse_sigma"] < 0.5
    with pytest.raises(ValueError, match="does not support the inverse-sigma reading"):
        _ms(a, require_weight_check=True)
    a2, _ = _arrays()
    a2.pop("SAISIL")
    with pytest.raises(ValueError, match="no stored chi"):
        _ms(a2, require_weight_check=True)


def test_weight_convention_check_values():
    m, c, w = np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.8, 3.0]), np.array([10.0, 5.0, 2.0])
    chk = ET.weight_convention_check(m, c, w, (w * (m - c)) ** 2)
    assert chk["n"] == 2 and chk["supports"] == "inverse_sigma"      # zero residual excluded
    chk2 = ET.weight_convention_check(m, c, w, w * (m - c) ** 2)
    assert chk2["supports"] == "inverse_variance"
    assert ET.weight_convention_check(m, c, np.zeros(3), np.ones(3))["supports"] is None


def test_required_statements():
    a, _ = _arrays()
    with pytest.raises(ValueError, match="sigma_model must be one of"):
        ET.measurements_from_efit_arrays(a, _machine(), 1.5, 0.02, sigma_model=None)
    with pytest.raises(ValueError, match="input_weight"):
        _ms(a, input_weight=None)
    with pytest.raises(ValueError, match="rule_inputs"):
        _ms(a, sigma_model="rule")
    with pytest.raises(ValueError, match="requested but their measured arrays are absent"):
        _ms(a, groups=["mse"])


def test_unfitted_coils_and_groups_without_weights():
    a, s = _arrays()
    ms = _ms(a)
    pf, ip = ms.groups["pf_current"], ms.groups["ip"]
    assert pf.labels == ["F0", "F1", "E0"]
    np.testing.assert_allclose(pf.sigma[:2], s["fcoil"], rtol=1e-14)
    assert np.isnan(pf.sigma[2]) and not pf.valid[2]                       # no sigma for the unfitted E-coil
    assert "weight 0 and no sigma available" in pf.meta["invalid_reason_unfitted_by_array"]["ECURRT"]
    assert pf.meta["coil_roles"] == {"F0": "fitted", "F1": "fitted", "E0": "undetermined"}
    assert not ip.valid[0] and "no stored weight" in ip.meta["invalid_reason"]
    rule = dict(SERROR=0.03, VBIT=10.0, BITEC=[50.0], BITIP=1000.0, BITFC=[1.0, 1.0], BITMPI=[1e-4] * 3)
    ms2 = _ms(a, rule_inputs=rule)
    pf2, ip2 = ms2.groups["pf_current"], ms2.groups["ip"]
    assert pf2.sigma[2] == pytest.approx(max(0.03 * 3000.0, 10.0 * 50.0)) and pf2.valid[2]
    assert ip2.sigma[0] == pytest.approx(max(0.03 * 1.0e6, 10.0 * 1000.0)) and ip2.valid[0]
    a3 = dict(a, SIGPASMA=np.array([7.0e3]))
    assert _ms(a3).groups["ip"].sigma[0] == 7.0e3
    ms4 = _ms(a, score_unfitted_coils=False, rule_inputs=rule)
    assert not ms4.groups["pf_current"].valid[2]


def test_stored_and_rule_models():
    a, s = _arrays()
    a = dict(a, SIGSIL=s["loop"] * 1.1, SIGMPI=s["probe"] * 0.9, SIGFCC=s["fcoil"], SIGECC=np.array([90.0]))
    ms = _ms(a, sigma_model="stored", input_weight=None)
    np.testing.assert_allclose(ms.groups["flux_loop"].sigma, s["loop"] * 1.1 * TWO_PI, rtol=1e-14)
    np.testing.assert_allclose(ms.groups["bpol_probe"].sigma, s["probe"] * 0.9, rtol=1e-14)
    assert ms.groups["pf_current"].sigma[2] == 90.0
    rule = dict(SERROR=0.03, VBIT=80.0, BITMPI=[1e-4, 2e-4, 3e-4], BITFC=[2.0, 2.0], BITEC=[1.0], BITIP=100.0,
                PSIBIT=[1e-5] * 4, KERSIL=3, ERRSIL=0.03, SICONT=1e-9, NSLREF=1)
    mr = _ms(a, sigma_model="rule", input_weight=None, rule_inputs=rule)
    np.testing.assert_allclose(mr.groups["bpol_probe"].sigma,
                               np.maximum(0.03 * np.abs(a["EXPMPI"]), 80.0 * np.array([1e-4, 2e-4, 3e-4])), rtol=1e-14)
    ref, _ = kersil_loop_sigma({"KERSIL": [3.0], "NSLREF": [1.0], "SIREF": [0.5]}, a["SILOPT"], np.full(4, 1e-5),
                               [1.0, 1.1, 1.2, 2.3], 1.0e6, 80.0, 0.03, dict(ERRSIL=0.03, SICONT=1e-9), "siref")
    np.testing.assert_allclose(mr.groups["flux_loop"].sigma, ref * TWO_PI, rtol=1e-14)


def test_source_fit_chi2_and_case_blind_names():
    a, s = _arrays()
    ms = _ms({k.lower(): v for k, v in a.items()})                  # measurement-file style names
    out = ET.source_fit_chi2(ms)
    use = np.array([0, 1, 3])
    z = (a["CSILOP"] - a["SILOPT"])[use] / s["loop"][use]
    assert out["flux_loop"][1] == 3 and out["flux_loop"][0] == pytest.approx(np.sum(z ** 2), rel=1e-12)
    zp = (a["CMPR2"] - a["EXPMPI"]) / s["probe"]
    assert out["bpol_probe"][2] == pytest.approx(np.mean(zp ** 2), rel=1e-12)
    # EFIT's own stored chi^2 equals the per-channel terms under the inverse-sigma convention
    assert out["flux_loop"][0] == pytest.approx(np.sum(a["SAISIL"][use]), rel=1e-12)


def test_padding_and_mse():
    a, _ = _arrays()
    pad = dict(a, EXPMPI=np.concatenate((a["EXPMPI"], [0.0, 0.0])), FWTMP2=np.concatenate((a["FWTMP2"], [0.0, 0.0])))
    assert _ms(pad).groups["bpol_probe"].measured.size == 3
    bad = dict(a, EXPMPI=np.concatenate((a["EXPMPI"], [0.5])))
    with pytest.raises(ValueError, match="nonzero entries beyond"):
        _ms(bad)
    m = dict(a, TANGAM=np.array([0.1, -0.05, 0.0, 0.0]), CMGAM=np.array([0.11, -0.04, 0.0, 0.0]),
             FWTGAM=np.array([100.0, 50.0, 0.0, 0.0]), SIGGAM=np.array([0.01, 0.02, 0.0, 0.0]),
             RRGAM=np.array([1.8, 2.0, 0.0, 0.0]), ZZGAM=np.zeros(4), A1GAM=np.ones(4), A2GAM=np.ones(4),
             A3GAM=np.zeros(4), A4GAM=np.zeros(4))
    with pytest.raises(ValueError, match="er_corrected must be stated"):
        _ms(m)
    g = _ms(m, er_corrected=True).groups["mse"]
    assert g.measured.size == 4 and g.valid.tolist() == [True, True, False, False]    # no-data chords kept, invalid
    np.testing.assert_allclose(g.sigma[:2], [0.01, 0.02], rtol=1e-14)
    assert g.meta["er_corrected"] is True and g.meta["quantity"] == "tan_polarisation_angle"


def test_round_trip_through_the_measurement_file():
    ms = _ms()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ms.json")
        write_measurements(p, ms)
        back = read_measurements(p)
    for name, g in ms.groups.items():
        b = back.groups[name]
        np.testing.assert_array_equal(b.measured, g.measured)
        np.testing.assert_array_equal(np.isnan(b.sigma), np.isnan(g.sigma))
        np.testing.assert_array_equal(b.valid, g.valid)


def test_fitted_flux_offset_in_the_scorer_and_for_efits_own_fit():
    from bouquet.kinrecon.interfaces import ResidualTable
    from bouquet.kinrecon.score import _fit_flux_offset
    a, s = _arrays()
    with pytest.raises(ValueError, match="two readings of the gauge"):
        _ms(a, flux_offset="fitted")
    ms = _ms(a, flux_offset="fitted", flux_reference=None)
    g = ms.groups["flux_loop"]
    assert g.meta["flux_offset"] == "fitted"
    use = g.valid
    # a model that differs from the data by a constant plus known residuals: the offset absorbs the constant
    resid = np.array([0.3, -0.5, 9.0, 0.2]) * g.sigma
    resid[2] = 0.0
    model = g.measured - 0.7 + np.where(use, resid, 0.0)
    tab = ResidualTable(measurements=ms, model={"flux_loop": model.copy()},
                        model_valid={"flux_loop": np.isfinite(model)})
    _fit_flux_offset(tab)
    w = 1.0 / g.sigma[use] ** 2
    c = np.sum(w * (g.measured[use] - model[use])) / np.sum(w)
    assert tab.frame["flux_offset"] == pytest.approx(c, rel=1e-12)
    r = tab.residual("flux_loop")
    assert abs(np.sum(r / g.sigma[use])) < 1e-9 * np.sum(np.abs(r / g.sigma[use]))   # weighted mean residual 0
    _fit_flux_offset(tab)                                                           # idempotent
    assert tab.frame["flux_offset"] == pytest.approx(c, rel=1e-12)
    # EFIT's own fitted values: a constant shift of all fitted loops leaves its chi2 unchanged under the offset
    a2 = dict(a, CSILOP=a["CSILOP"] + 0.05)
    c1 = ET.source_fit_chi2(_ms(a, flux_offset="fitted", flux_reference=None))["flux_loop"]
    c2 = ET.source_fit_chi2(_ms(a2, flux_offset="fitted", flux_reference=None))["flux_loop"]
    assert c2[0] == pytest.approx(c1[0], rel=1e-9) and c1[1] == 3


def test_fitted_flux_offset_with_the_reference_prior_row():
    from bouquet.kinrecon.interfaces import ResidualTable
    from bouquet.kinrecon.score import _fit_flux_offset
    a, s = _arrays()
    with pytest.raises(ValueError, match="applies to flux_offset='fitted'"):
        _ms(a, reference_prior=(0.5, 1e-3))
    ms = _ms(a, flux_offset="fitted", flux_reference=None, reference_prior=(0.5, 1e-3))
    g = ms.groups["flux_loop"]
    c0, s0 = g.meta["flux_offset_prior"]
    assert c0 == pytest.approx(TWO_PI * 0.5) and s0 == pytest.approx(TWO_PI * 1e-3)
    use = g.valid
    model = g.measured - 3.0 + np.where(use, 0.2 * g.sigma, 0.0)
    tab = ResidualTable(measurements=ms, model={"flux_loop": model.copy()}, model_valid={"flux_loop": np.isfinite(model)})
    _fit_flux_offset(tab)
    w = 1.0 / g.sigma[use] ** 2
    c = (np.sum(w * (g.measured[use] - model[use])) + c0 / s0 ** 2) / (np.sum(w) + 1.0 / s0 ** 2)
    assert tab.frame["flux_offset"] == pytest.approx(c, rel=1e-12)
    assert tab.frame["flux_offset_prior_chi2"] == pytest.approx(((c - c0) / s0) ** 2, rel=1e-12)
    out = ET.source_fit_chi2(ms)
    assert out["flux_loop_offset_prior"][1] == 1 and out["flux_loop_offset_prior"][0] >= 0.0
