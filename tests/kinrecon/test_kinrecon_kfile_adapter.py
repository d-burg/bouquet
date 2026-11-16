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
 TGAMMAUNCOR = 0.11 -0.04 0.0
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
    # VBIT is absent from the made-up k-file: the bundle states a made-up piecewise default by shot
    return MachineBundle(machine="toy", sensors=sensors, channel_maps=maps, meta={"vbit_by_shot": [[0, 10.0]]})


# The fixture's MSE block is labeled E_r-corrected (MSE_USECER = 1) but its D = TGAMMA - TGAMMAUNCOR has
# sign(D) sign(TGAMMA) uniform on only half of the active chords, which the E_r status rule refuses
# (test_mse_block); tests about other groups state the status (er_corrected=True) instead of relying on it.
VBIT_MODEL = {"sigma_model": "efit_vbit"}


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
    ms = measurements_from_kfile(nl, _machine(), er_corrected=True, **VBIT_MODEL)
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
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="L0", **VBIT_MODEL)
    g = measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_scale=+2 * np.pi,
                                **VBIT_MODEL).groups["flux_loop"]
    np.testing.assert_allclose(g.measured, 2 * np.pi * np.array([0.1, -0.2, 0.3]))   # a stated scale is applied
    # relative storage (reference entry 0, SIREF set): the reference is accepted
    nl["IN1"]["COILS"] = [0.0, -0.2, 0.3]
    nl["IN1"]["SIREF"] = [0.4]
    g = measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="L0",
                                **VBIT_MODEL).groups["flux_loop"]
    np.testing.assert_allclose(g.measured, -2 * np.pi * np.array([0.0, -0.2, 0.3]))
    assert (g.sigma > 0).all()                     # sigma uses |scale|
    assert g.meta["flux_reference"] == "L0" and g.meta["coils_form"] == "relative"
    with pytest.raises(ValueError, match="flux_reference"):
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_reference="nope", **VBIT_MODEL)


def test_lowercase_repeat_disables_loops(kfile):
    # Fortran semantics: the later 'fwtsi' in the same block overrides FWTSI
    ms = measurements_from_kfile(kfile, _machine(), groups=["flux_loop"], **VBIT_MODEL)
    assert not ms.groups["flux_loop"].valid.any()


def test_coils_split_and_per_turn(kfile):
    ms = measurements_from_kfile(kfile, _machine(with_maps=True), groups=["pf_current"])
    c = ms.groups["pf_current"]
    assert c.labels == ["F0", "F1", "E0"]
    np.testing.assert_array_equal(c.measured, [1000.0, -2000.0, 50.0])
    np.testing.assert_allclose(c.sigma, [100.0, 100.0, 10.0])
    assert c.valid.tolist() == [True, True, True]      # FWTEC = 0 coil scored (scoring convention)
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
    # the E_r status is stated by the caller: without it the MSE group is refused (no default classification)
    with pytest.raises(ValueError, match="state er_corrected"):
        measurements_from_kfile(kfile, _machine(), groups=["mse"])
    ms = measurements_from_kfile(kfile, _machine(), groups=["mse"], er_corrected=True)
    g = ms.groups["mse"]
    assert len(g.labels) == 3                       # FWTGAM padding dropped
    np.testing.assert_array_equal(g.measured, [0.1, -0.05, 0.0])
    np.testing.assert_array_equal(g.sigma, [0.002] * 3)
    assert g.valid.tolist() == [True, True, False]  # zero TGAMMA = no data
    assert g.meta["er_corrected"] is True and g.meta["quantity"] == "tan_polarisation_angle"
    np.testing.assert_array_equal(g.meta["tan_gamma_uncorrected"], [0.11, -0.04, 0.0])
    assert g.meta["er_corrected_source"] == "stated by the caller" and "er_status" not in g.meta
    assert g.meta["er_evidence"]["n_active"] == 2 and g.meta["er_evidence"]["n_tgamma_differs_from_uncorrected"] == 2
    want = efit_to_imas_coefficients([0.7] * 3, [-0.6] * 3, [0.1] * 3, [0.05] * 3,
                                     [0.4] * 3, [0.0] * 3, [0.02] * 3)
    np.testing.assert_array_equal(g.geometry["coefficients"], want)
    np.testing.assert_array_equal(g.geometry["coefficients"][0],
                                  [0.7, 0, 0, 0.4, 0.05, 0.1, -0.6, 0.02, 0.0])


def test_mse_er_flag_rules(kfile):
    # MSE_USECER decides nothing: the status is the caller's, the flags are recorded as evidence only
    nl = read_kfile(kfile)
    del nl["INS"]["MSE_USECER"]
    with pytest.raises(ValueError, match="state er_corrected"):
        measurements_from_kfile(nl, _machine(), groups=["mse"])
    with pytest.raises(ValueError, match="state er_corrected"):
        measurements_from_kfile(nl, _machine(), groups=["mse"], er_corrected="auto")
    g = measurements_from_kfile(nl, _machine(), groups=["mse"], er_corrected=False).groups["mse"]
    assert g.meta["er_corrected"] is False
    assert g.meta["er_corrected_source"] == "stated by the caller"
    assert g.meta["er_evidence"]["MSE_USECER"] is None
    # a set built without MSE needs no statement
    assert "mse" not in measurements_from_kfile(nl, _machine(), groups=["ip"]).groups


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
    # relative storage (reference entry 0, SIREF set): a reference may be named
    nl["IN1"]["COILS"] = [0.0, -2.0, 0.3]
    nl["IN1"]["SIREF"] = [0.7]
    ms = measurements_from_kfile(nl, _machine(with_maps=True), flux_reference="L0", er_corrected=True,
                                 **VBIT_MODEL)
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


def test_vbit_absent_taken_from_the_bundle_table():
    ms = measurements_from_kfile(parse_namelists(_with_in1(KFILE, BITMPI="4*2.0E-3")),
                                 _machine(), groups=["bpol_probe"], **VBIT_MODEL)
    np.testing.assert_allclose(ms.groups["bpol_probe"].sigma, [0.02, 0.02, 0.02, 0.02])
    assert ms.meta["VBIT"] == 10.0 and "absent" in ms.meta["VBIT_source"]
    bare = _machine()
    bare.meta = {}
    with pytest.raises(ValueError, match="vbit_by_shot"):           # no file value, no caller, no bundle table
        measurements_from_kfile(parse_namelists(KFILE), bare, groups=["bpol_probe"], **VBIT_MODEL)


def test_fixed_ten_bit_model_still_available():
    text = _with_in1(KFILE, VBIT="40.0", BITIP="1.0E+4")
    old = measurements_from_kfile(parse_namelists(text), _machine(with_maps=True),
                                  sigma_model="efit_serror_bit", er_corrected=True)
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
    # the note no longer claims a zero-weight coil is "held fixed" (wrong for F-coils)
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


def test_er_status_is_stated_never_classified():
    """Whatever TGAMMA - TGAMMAUNCOR looks like (zero, E_r-shaped, absent), the adapter takes the stated status and
    records the evidence only; without a statement the MSE group is refused.  (An E_r correction on only some chords
    is refused as malformed input: test_mse_input_consistency_refusals.)"""
    r = np.array([2.1, 1.8, 2.0, 1.9])
    tg = np.array([-0.08, 0.1, -0.05, 0.12])
    cases = dict(zero=tg.copy(), shaped=tg - 0.05 * tg, absent=None)
    for name, unc in cases.items():
        for usecer in (0, 1, None):
            nl = _ins_kfile(tg, unc, r, usecer=usecer)
            with pytest.raises(ValueError, match="state er_corrected"):
                _mse_meta(nl)
            for stated in (True, False):
                m = _mse_meta(nl, er_corrected=stated)
                assert m["er_corrected"] is stated and m["er_corrected_source"] == "stated by the caller"
                ev = m["er_evidence"]
                assert ev["MSE_USECER"] == (None if usecer is None else float(usecer)) and ev["n_active"] == 4
                assert ev["tgammauncor_present"] is (unc is not None)
                if unc is not None:
                    assert ev["n_tgamma_differs_from_uncorrected"] == int(np.count_nonzero(tg != unc))
                assert not any(k in m for k in ("er_status", "er_route"))
    # the in-fit E_r route on the raw measurement is recorded with the stated status
    m = _mse_meta(_ins_kfile(tg, tg.copy(), r, usecer=0, extra_in1=" KEECUR = 2"), er_corrected=False)
    assert m["er_in_fit"] is True and m["er_evidence"]["KEECUR"] == 2.0


def test_mse_input_consistency_refusals():
    """Two input checks (they check the block, they do not classify the plasma): an E_r correction on only some of
    the active chords, and the in-fit E_r route (KEECUR > 0) with an already-altered TGAMMA -- differing from
    TGAMMAUNCOR, or stated E_r-corrected -- are refused whatever the stated status; the well-formed blocks pass."""
    r = np.array([2.1, 1.8, 2.0, 1.9])
    tg = np.array([-0.08, 0.1, -0.05, 0.12])
    mixed = tg - np.array([0.0, 0.0, 0.004, 0.006])
    for usecer in (0, 1, None):
        for stated in (True, False):
            with pytest.raises(ValueError, match="only some chords"):
                _mse_meta(_ins_kfile(tg, mixed, r, usecer=usecer), er_corrected=stated)
    # an inactive chord does not count: the same difference on a chord with zero fit weight is accepted
    nl = _ins_kfile(tg, mixed, r)
    nl["INS"]["FWTGAM"] = [1.0, 1.0, 0.0, 0.0]
    assert _mse_meta(nl, er_corrected=False)["er_evidence"]["n_tgamma_differs_from_uncorrected"] == 0
    # in-fit E_r with an altered TGAMMA (differing on every chord), or stated E_r-corrected: refused
    for stated in (True, False):
        with pytest.raises(ValueError, match="counted twice"):
            _mse_meta(_ins_kfile(tg, tg - 0.01, r, usecer=0, extra_in1=" KEECUR = 2"), er_corrected=stated)
    with pytest.raises(ValueError, match="counted twice"):
        _mse_meta(_ins_kfile(tg, tg.copy(), r, usecer=0, extra_in1=" KEECUR = 2"), er_corrected=True)
    with pytest.raises(ValueError, match="counted twice"):
        _mse_meta(_ins_kfile(tg, None, r, usecer=0, extra_in1=" KEECUR = 2"), er_corrected=True)
    # well formed: a correction on every active chord, or none; KEECUR > 0 on the raw TGAMMA
    assert _mse_meta(_ins_kfile(tg, tg - 0.05 * tg, r), er_corrected=True)["er_corrected"] is True
    assert _mse_meta(_ins_kfile(tg, tg.copy(), r), er_corrected=False)["er_corrected"] is False
    assert _mse_meta(_ins_kfile(tg, None, r, extra_in1=" KEECUR = 2"), er_corrected=False)["er_in_fit"] is True

