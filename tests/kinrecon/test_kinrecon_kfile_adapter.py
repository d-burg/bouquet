"""bouquet.kinrecon.adapters.efit_kfile on a small made-up k-file."""
import numpy as np
import pytest

from bouquet.kinrecon.adapters.efit_kfile import (measurements_from_kfile, parse_namelists,
                                                  read_kfile)
from bouquet.kinrecon.measurements import MachineBundle, efit_to_imas_coefficients

# A made-up k-file: 3 flux loops, 4 probes, 2 + 1 coils, 3 MSE chords.
# BRSP is padded past the coil count, FWTGAM past the chord count, the way
# EFIT dimensions its namelist arrays.
KFILE = """\
 &IN1
 ISHOT = 1
 ITIME = 1500
 PLASMA = 1.0D+06
 EXPMP2 = 0.5 -0.002 2*0.25
 COILS = 0.1 -0.2
 0.3
 BTOR = -2.0
 FWTSI = 1.0 0.0 1.0
 FWTMP2 = 4*1.0
 SERROR = 0.03
 PSIBIT = 3*1.0E-3
 BITMPI = 4*1.0E-3
 FWTCUR = 1.0
 BITIP = 100.0
 FWTFC = 2*1.0
 BRSP = 1000.0 -2000.0 6*0.0
 BITFC = 10.0 10.0
 FWTEC = 0.0
 ECURRT = 50.0
 BITEC = 1.0
 DFLUX = 2.5
 SIGDLC = 0.1
 FWTDLC = 1.0
 IAVEM = 10
 fwtsi = 3*0.0
 /
 &INWANT
 PSIWANT = 0.99
 /
 &INS
 TGAMMA = 0.1 -0.05 0.0
 SGAMMA = 3*0.002
 FWTGAM = 1.0 1.0 1.0 7*0.0
 RRRGAM = 1.8 1.9 2.0
 ZZZGAM = 3*0.0
 AA1GAM = 3*0.7
 AA2GAM = 3*-0.6
 AA3GAM = 3*0.1
 AA4GAM = 3*0.05
 AA5GAM = 3*0.4
 AA6GAM = 3*0.0
 AA7GAM = 3*0.02
 TGAMMAUNCOR = 0.09 -0.04 0.0
 MSE_USECER = 1
 DTMSEFULL = 5.0
 /
                     free text between blocks is ignored
 &EFITIN
 fwtmp2 = 4*0.0
 /
"""


def _machine(with_maps=False):
    sensors = {
        "flux_loop": {"labels": ["L0", "L1", "L2"], "r": [1.0, 1.1, 1.2], "z": [0.0, 0.5, -0.5]},
        "bpol_probe": {"labels": ["P0", "P1", "P2", "P3"], "r": [1.0] * 4, "z": [0.0] * 4,
                       "direction_r": [0.0] * 4, "direction_z": [1.0] * 4},
        "pf_current": {"labels": ["E0", "F0", "F1"], "turns": [1.0, 10.0, 20.0]},
    }
    maps = {"BRSP": ["F0", "F1"], "ECURRT": ["E0"]} if with_maps else {}
    return MachineBundle(machine="toy", sensors=sensors, channel_maps=maps)


@pytest.fixture
def kfile(tmp_path):
    p = tmp_path / "k_toy.01500"
    p.write_text(KFILE)
    return str(p)


def test_namelist_parser():
    nl = parse_namelists(KFILE)
    assert set(nl) == {"IN1", "INWANT", "INS", "EFITIN"}
    assert nl["IN1"]["PLASMA"] == [1.0e6]
    assert nl["IN1"]["EXPMP2"] == [0.5, -0.002, 0.25, 0.25]
    assert nl["IN1"]["COILS"] == [0.1, -0.2, 0.3]          # continuation line
    assert nl["IN1"]["FWTSI"] == [0.0, 0.0, 0.0]           # later lowercase repeat wins
    assert nl["EFITIN"]["FWTMP2"] == [0.0] * 4             # stays in its own block
    assert nl["IN1"]["FWTMP2"] == [1.0] * 4
    more = parse_namelists(" &X\n A = 'a=b', .true. F\n B(3) = 7 8\n /\n")
    assert more["X"]["A"] == ["a=b", True, False]
    assert more["X"]["B"] == [None, None, 7.0, 8.0]


def test_magnetics_sigma_and_weights(kfile):
    nl = read_kfile(kfile)
    nl["IN1"]["FWTSI"] = [1.0, 0.0, 1.0]  # undo the lowercase override for this test
    ms = measurements_from_kfile(nl, _machine())
    fl = ms.groups["flux_loop"]
    two_pi = 2 * np.pi
    # default flux_scale is -2 pi (the k-files tested store COILS in the solver's psi sign)
    np.testing.assert_allclose(fl.measured, -np.array([0.1, -0.2, 0.3]) * two_pi, rtol=1e-15)
    # sigma = max(SERROR |COILS|, VBIT PSIBIT) x 2 pi, VBIT absent -> 10
    np.testing.assert_allclose(fl.sigma, np.maximum(0.03 * np.array([0.1, 0.2, 0.3]), 1e-2) * two_pi)
    assert fl.valid.tolist() == [True, False, True]
    assert fl.weight.tolist() == [1.0, 0.0, 1.0]
    assert fl.meta["sigma_model"] == "efit_vbit"
    assert fl.meta["VBIT"] == 10.0 and "absent" in fl.meta["VBIT_source"]
    pr = ms.groups["bpol_probe"]
    np.testing.assert_allclose(pr.sigma, [0.015, 0.01, 0.01, 0.01])
    np.testing.assert_array_equal(pr.geometry["direction_z"], [1.0] * 4)
    ip = ms.groups["ip"]
    np.testing.assert_allclose(ip.sigma, [3.0e4])
    dfl = ms.groups["diamagnetic_flux"]
    np.testing.assert_allclose(dfl.measured, [2.5e-3])
    np.testing.assert_allclose(dfl.sigma, [1e-4])
    assert (ms.time, ms.time_window) == (1.5, 0.02)       # 2 x IAVEM (ITIME +/- IAVEM)
    assert "IAVEM" in ms.meta["time_window_source"]


def test_flux_sign_and_reference(kfile):
    nl = read_kfile(kfile)
    nl["IN1"]["FWTSI"] = [1.0, 1.0, 1.0]
    # this slice stores ABSOLUTE fluxes (no SIREF, reference entry nonzero): a reference is refused
    with pytest.raises(ValueError, match="not in relative form"):
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="L0")
    g = measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_scale=+2 * np.pi).groups["flux_loop"]
    np.testing.assert_allclose(g.measured, 2 * np.pi * np.array([0.1, -0.2, 0.3]))   # a stated scale is applied
    # relative storage (reference entry 0, SIREF set): the reference is accepted
    nl["IN1"]["COILS"] = [0.0, -0.2, 0.3]
    nl["IN1"]["SIREF"] = [0.4]
    g = measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="L0").groups["flux_loop"]
    np.testing.assert_allclose(g.measured, -2 * np.pi * np.array([0.0, -0.2, 0.3]))
    assert (g.sigma > 0).all()                     # sigma uses |scale|
    assert g.meta["flux_reference"] == "L0" and g.meta["coils_form"] == "relative"
    with pytest.raises(ValueError, match="flux_reference"):
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="nope")


def test_lowercase_repeat_disables_loops(kfile):
    # Fortran semantics: the later 'fwtsi' in the same block overrides FWTSI
    ms = measurements_from_kfile(kfile, _machine(), groups=["flux_loop"])
    assert not ms.groups["flux_loop"].valid.any()


def test_coils_split_and_per_turn(kfile):
    ms = measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"])
    c = ms.groups["pf_current"]
    assert c.labels == ["F0", "F1", "E0"]
    np.testing.assert_array_equal(c.measured, [1000.0, -2000.0, 50.0])
    np.testing.assert_allclose(c.sigma, [100.0, 100.0, 10.0])
    assert c.valid.tolist() == [True, True, True]      # FWTEC = 0 coil scored (R04)
    assert c.weight.tolist() == [1.0, 1.0, 1.0]
    assert c.meta["not_fitted_by_efit"] == ["E0"]
    np.testing.assert_array_equal(c.geometry["turns"], [10.0, 20.0, 1.0])
    pt = measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"],
                                 coil_current_per_turn=True).groups["pf_current"]
    np.testing.assert_allclose(pt.measured, [100.0, -100.0, 50.0])
    np.testing.assert_allclose(pt.sigma, [10.0, 5.0, 10.0])
    only_f = measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"],
                                     coil_current_per_turn=("BRSP",)).groups["pf_current"]
    np.testing.assert_allclose(only_f.measured, [100.0, -100.0, 50.0 / 1.0])
    assert only_f.meta["current_convention"]["ECURRT"] == "as written in the k-file"
    with pytest.raises(ValueError, match="unknown arrays"):
        measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"],
                                coil_current_per_turn=("XYZ",))
    # without maps the bundle order is used: BRSP coils then ECURRT coils
    nomap = measurements_from_kfile(kfile, _machine(), groups=["pf_current"]).groups["pf_current"]
    assert nomap.labels == ["E0", "F0", "F1"]
    np.testing.assert_array_equal(nomap.measured, [1000.0, -2000.0, 50.0])


def test_mse_block(kfile):
    ms = measurements_from_kfile(kfile, _machine(), groups=["mse"])
    g = ms.groups["mse"]
    assert len(g.labels) == 3                       # FWTGAM padding dropped
    np.testing.assert_array_equal(g.measured, [0.1, -0.05, 0.0])
    np.testing.assert_array_equal(g.sigma, [0.002] * 3)
    assert g.valid.tolist() == [True, True, False]  # zero TGAMMA = no data
    assert g.meta["er_corrected"] is True and g.meta["quantity"] == "tan_polarisation_angle"
    np.testing.assert_array_equal(g.meta["tan_gamma_uncorrected"], [0.09, -0.04, 0.0])
    assert g.meta["er_status"] == "pre_corrected" and g.meta["er_route"] == "efit_internal"
    want = efit_to_imas_coefficients([0.7] * 3, [-0.6] * 3, [0.1] * 3, [0.05] * 3,
                                     [0.4] * 3, [0.0] * 3, [0.02] * 3)
    np.testing.assert_array_equal(g.geometry["coefficients"], want)
    np.testing.assert_array_equal(g.geometry["coefficients"][0],
                                  [0.7, 0, 0, 0.4, 0.05, 0.1, -0.6, 0.02, 0.0])


def test_mse_er_flag_rules(kfile):
    # MSE_USECER absent no longer decides anything: the D rule still applies
    # (D != 0 and E_r-shaped -> pre-corrected, route 'external')
    nl = read_kfile(kfile)
    del nl["INS"]["MSE_USECER"]
    g = measurements_from_kfile(nl, _machine(), groups=["mse"]).groups["mse"]
    assert g.meta["er_corrected"] is True and g.meta["er_route"] == "external"
    g = measurements_from_kfile(nl, _machine(), groups=["mse"], er_corrected=False).groups["mse"]
    assert g.meta["er_corrected"] is False
    assert g.meta["er_corrected_source"] == "caller override"
    assert g.meta["er_route"] == "caller_stated"
    assert g.meta["er_status"] == "pre_corrected"      # evidence still recorded


def test_bundle_size_mismatch(kfile):
    m = _machine()
    m.sensors["bpol_probe"] = {"labels": [f"P{i}" for i in range(6)]}
    with pytest.raises(ValueError, match="6 sensors"):
        measurements_from_kfile(kfile, m, groups=["bpol_probe"])


# --- EFIT-aligned sigma (VBIT), zero-weight coils, +/-IAVEM, E_r status -------
def _with_in1(text, **kv):
    """Insert (or replace) &IN1 entries in the made-up k-file text."""
    lines = [ln for ln in text.splitlines()
             if not any(ln.strip().upper().startswith(k + " =") for k in kv)]
    i = lines.index(" &IN1") + 1
    for k, v in kv.items():
        lines.insert(i, f" {k} = {v}")
    return "\n".join(lines) + "\n"


def test_vbit_read_from_kfile_sets_the_floor():
    nl = parse_namelists(_with_in1(KFILE, VBIT="40.0", BITIP="1.0E+4"))
    nl["IN1"]["FWTSI"] = [1.0, 1.0, 1.0]
    # relative storage (reference entry 0, SIREF set): a reference may be named (report 32 F3)
    nl["IN1"]["COILS"] = [0.0, -2.0, 0.3]
    nl["IN1"]["SIREF"] = [0.7]
    ms = measurements_from_kfile(nl, _machine(with_maps=True), flux_reference="L0")
    two_pi = 2 * np.pi
    assert ms.meta["VBIT"] == 40.0 and ms.meta["VBIT_source"] == "k-file &IN1"
    pr = ms.groups["bpol_probe"]
    np.testing.assert_allclose(pr.sigma, np.maximum(0.03 * np.abs([0.5, -0.002, 0.25, 0.25]),
                                                    40 * 1e-3), rtol=1e-15)
    assert pr.meta["VBIT"] == 40.0 and pr.meta["sigma_model"] == "efit_vbit"
    fl = ms.groups["flux_loop"]
    # reference loop L0: VBIT PSIBIT only; L1: SERROR |COILS| = 0.06 > 0.04 binds
    np.testing.assert_allclose(fl.sigma, np.array([0.04, 0.06, 0.04]) * two_pi, rtol=1e-15)
    co = ms.groups["pf_current"]
    np.testing.assert_allclose(co.sigma, [400.0, 400.0, 40.0])       # F0, F1, E0
    # Ip: SERROR |PLASMA|, no bit floor (VBIT BITIP = 4e5 would dominate)
    np.testing.assert_allclose(ms.groups["ip"].sigma, [3.0e4])
    assert "no bit floor" in ms.groups["ip"].meta["sigma_rule"]
    np.testing.assert_allclose(ms.groups["diamagnetic_flux"].sigma, [1e-4])  # SIGDLC x 1e-3
    np.testing.assert_array_equal(ms.groups["mse"].sigma, [0.002] * 3)      # SGAMMA


def test_vbit_absent_defaults_to_ten():
    ms = measurements_from_kfile(parse_namelists(_with_in1(KFILE, BITMPI="4*2.0E-3")),
                                 _machine(), groups=["bpol_probe"])
    np.testing.assert_allclose(ms.groups["bpol_probe"].sigma, [0.02, 0.02, 0.02, 0.02])
    assert ms.meta["VBIT"] == 10.0 and "absent" in ms.meta["VBIT_source"]


def test_fixed_ten_bit_model_still_available():
    text = _with_in1(KFILE, VBIT="40.0", BITIP="1.0E+4")
    old = measurements_from_kfile(parse_namelists(text), _machine(with_maps=True),
                                  sigma_model="efit_serror_bit")
    np.testing.assert_allclose(old.groups["bpol_probe"].sigma, [0.015, 0.01, 0.01, 0.01])
    np.testing.assert_allclose(old.groups["ip"].sigma, [1.0e5])      # 10 BITIP floor
    assert old.groups["bpol_probe"].meta["sigma_model"] == "efit_serror_bit"
    assert "VBIT" not in old.groups["bpol_probe"].meta
    with pytest.raises(ValueError, match="sigma_model"):
        measurements_from_kfile(parse_namelists(text), _machine(), sigma_model="relative")


def test_zero_weight_coils_old_behavior_option(kfile):
    c = measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"],
                                score_unfitted_coils=False).groups["pf_current"]
    assert c.valid.tolist() == [True, True, False]
    assert c.weight.tolist() == [1.0, 1.0, 0.0]
    assert "not_fitted_by_efit" not in c.meta
    d = measurements_from_kfile(kfile, _machine(with_maps=True),
                                groups=["pf_current"]).groups["pf_current"]
    # the note no longer claims a zero-weight coil is "held fixed" (wrong for F-coils, P1 report)
    assert "is not a constraint" in d.meta["note"] and "held fixed" not in d.meta["note"]
    np.testing.assert_array_equal(d.meta["efit_fit_weight"], [1.0, 1.0, 0.0])


def test_iavem_is_a_half_width(kfile):
    ms = measurements_from_kfile(kfile, _machine(), groups=["bpol_probe"])
    assert ms.time_window == pytest.approx(0.02)                     # IAVEM = 10 ms
    np.testing.assert_allclose(ms.meta["time_window_s"], [1.49, 1.51])
    assert "+/- IAVEM" in ms.meta["time_window_source"]
    caller = measurements_from_kfile(kfile, _machine(), groups=["bpol_probe"], time_window=0.05)
    assert caller.time_window == 0.05 and caller.meta["time_window_source"] == "caller"
    np.testing.assert_allclose(caller.meta["time_window_s"], [1.475, 1.525])


def _ins_kfile(tg, unc, r, usecer=None, extra_in1=""):
    n = len(tg)
    f = lambda a: " ".join(repr(float(x)) for x in a)  # noqa: E731
    lines = [" &IN1", " ITIME = 1500", " IAVEM = 10", extra_in1, " /", " &INS",
             f" TGAMMA = {f(tg)}", f" SGAMMA = {n}*0.002", f" FWTGAM = {n}*1.0",
             f" RRRGAM = {f(r)}", f" ZZZGAM = {n}*0.0", f" AA1GAM = {n}*0.7",
             f" AA2GAM = {n}*-0.6", f" AA3GAM = {n}*0.1", f" AA4GAM = {n}*0.05",
             f" AA5GAM = {n}*0.4", f" AA6GAM = {n}*0.0", f" AA7GAM = {n}*0.02"]
    if unc is not None:
        lines.append(f" TGAMMAUNCOR = {f(unc)}")
    if usecer is not None:
        lines.append(f" MSE_USECER = {usecer}")
    lines.append(" /")
    return parse_namelists("\n".join(lines) + "\n")


def _mse_meta(nl, **kw):
    return measurements_from_kfile(nl, _machine(), groups=["mse"], **kw).groups["mse"].meta


def test_er_status_uncorrected_when_d_is_zero():
    tg = np.array([0.1, 0.12, -0.05, -0.08])
    r = np.array([1.8, 1.9, 2.0, 2.1])
    for usecer in (0, 1):
        m = _mse_meta(_ins_kfile(tg, tg.copy(), r, usecer=usecer))
        assert m["er_corrected"] is False and m["er_status"] == "uncorrected"
        assert m["er_route"] == "none"
        assert ("flag" in m["er_evidence"]) == (usecer == 1)   # requested, not applied


def test_er_status_pre_corrected_and_route():
    r = np.array([2.1, 1.8, 2.0, 1.9])                  # deliberately unsorted in R
    tg = np.array([-0.08, 0.1, -0.05, 0.12])
    d = 0.05 * tg                                       # |tan gamma| increased on every chord
    for usecer, route in ((1, "efit_internal"), (0, "external"), (None, "external")):
        m = _mse_meta(_ins_kfile(tg, tg - d, r, usecer=usecer))
        assert m["er_corrected"] is True and m["er_status"] == "pre_corrected"
        assert m["er_route"] == route
        assert m["er_evidence"]["sign_changes"] == 1 and m["er_evidence"]["sign_consistency"] == 1.0


def test_er_status_inconsistent_refused_unless_stated():
    r = np.linspace(1.8, 2.2, 8)
    tg = np.full(8, 0.1)
    d = 0.01 * np.array([1, -1, 1, -1, 1, -1, 1, -1])  # 7 sign changes, half the signs
    nl = _ins_kfile(tg, tg - d, r, usecer=0)
    with pytest.raises(ValueError, match="inconsistent"):
        _mse_meta(nl)
    m = _mse_meta(nl, er_corrected=True)
    assert m["er_corrected"] is True and m["er_status"] == "inconsistent"
    assert m["er_route"] == "caller_stated" and m["er_evidence"]["sign_changes"] == 7
    # D zero on some active chords but not all
    d2 = np.where(np.arange(8) < 4, 0.0, 0.005)
    with pytest.raises(ValueError, match="some active chords"):
        _mse_meta(_ins_kfile(tg, tg - d2, r, usecer=1))
    # in-fit E_r route with an altered TGAMMA
    with pytest.raises(ValueError, match="KEECUR"):
        _mse_meta(_ins_kfile(tg, tg - 0.01, r, usecer=0, extra_in1=" KEECUR = 2"))


def test_er_status_unknown_without_tgammauncor():
    tg = np.array([0.1, 0.12, -0.05])
    nl = _ins_kfile(tg, None, np.array([1.8, 1.9, 2.0]), usecer=1)
    with pytest.raises(ValueError, match="unknown"):
        _mse_meta(nl)
    m = _mse_meta(nl, er_corrected=False)
    assert m["er_status"] == "unknown" and m["er_corrected"] is False


def test_er_status_thresholds():
    from bouquet.kinrecon.adapters.efit_kfile import mse_er_status
    a2, a5, act = np.full(10, -0.6), np.full(10, 0.4), np.ones(10, bool)
    r = np.linspace(1.8, 2.25, 10)

    def status(tg, d):
        return mse_er_status(tg, tg - d, a2, a5, r, act)["status"]

    # sign changes of D A2/A5 along R: 2 accepted, 3 refused (T has D's sign)
    two = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, 1], float)
    three = np.array([1, 1, -1, -1, 1, 1, -1, -1, -1, -1], float)
    assert status(0.1 * two, 0.01 * two) == "pre_corrected"
    assert status(0.1 * three, 0.01 * three) == "inconsistent"
    # sign consistency of sign(D) sign(TGAMMA): 9/10 accepted, 8/10 refused
    d = np.full(10, 0.01)
    tg9 = np.where(np.arange(10) == 4, -0.1, 0.1)
    tg8 = np.where(np.isin(np.arange(10), (4, 7)), -0.1, 0.1)
    assert status(tg9, d) == "pre_corrected"
    assert status(tg8, d) == "inconsistent"
    # no active chord: nothing to decide, nothing refused
    out = mse_er_status(tg9, tg9, a2, a5, r, np.zeros(10, bool))
    assert out["status"] == "no_active_chords"
    assert mse_er_status(tg9, None, a2, a5, r, np.zeros(10, bool))["status"] == "no_active_chords"
    assert mse_er_status(tg9, None, a2, a5, r, act)["status"] == "unknown"
