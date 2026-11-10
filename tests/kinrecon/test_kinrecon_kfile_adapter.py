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
    np.testing.assert_allclose(fl.measured, np.array([0.1, -0.2, 0.3]) * two_pi, rtol=1e-15)
    # sigma = max(SERROR |COILS|, 10 PSIBIT) x 2 pi
    np.testing.assert_allclose(fl.sigma, np.maximum(0.03 * np.array([0.1, 0.2, 0.3]), 1e-2) * two_pi)
    assert fl.valid.tolist() == [True, False, True]
    assert fl.weight.tolist() == [1.0, 0.0, 1.0]
    assert fl.meta["sigma_model"] == "efit_serror_bit"
    pr = ms.groups["bpol_probe"]
    np.testing.assert_allclose(pr.sigma, [0.015, 0.01, 0.01, 0.01])
    np.testing.assert_array_equal(pr.geometry["direction_z"], [1.0] * 4)
    ip = ms.groups["ip"]
    np.testing.assert_allclose(ip.sigma, [3.0e4])
    dfl = ms.groups["diamagnetic_flux"]
    np.testing.assert_allclose(dfl.measured, [2.5e-3])
    np.testing.assert_allclose(dfl.sigma, [1e-4])
    assert (ms.time, ms.time_window) == (1.5, 0.01)
    assert "IAVEM" in ms.meta["time_window_source"]


def test_flux_sign_and_reference(kfile):
    nl = read_kfile(kfile)
    nl["IN1"]["FWTSI"] = [1.0, 1.0, 1.0]
    g = measurements_from_kfile(nl, _machine(), groups=["flux_loop"], flux_scale=-2 * np.pi,
                                flux_reference="L0").groups["flux_loop"]
    np.testing.assert_allclose(g.measured, -2 * np.pi * np.array([0.1, -0.2, 0.3]))
    assert (g.sigma > 0).all()                     # sigma uses |scale|
    assert g.meta["flux_reference"] == "L0"
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
    assert c.valid.tolist() == [True, True, False]
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
    np.testing.assert_array_equal(g.meta["tan_gamma_uncorrected"], [0.11, -0.04, 0.0])
    want = efit_to_imas_coefficients([0.7] * 3, [-0.6] * 3, [0.1] * 3, [0.05] * 3,
                                     [0.4] * 3, [0.0] * 3, [0.02] * 3)
    np.testing.assert_array_equal(g.geometry["coefficients"], want)
    np.testing.assert_array_equal(g.geometry["coefficients"][0],
                                  [0.7, 0, 0, 0.4, 0.05, 0.1, -0.6, 0.02, 0.0])


def test_mse_er_flag_rules(kfile):
    nl = read_kfile(kfile)
    del nl["INS"]["MSE_USECER"]
    with pytest.raises(ValueError, match="MSE_USECER"):
        measurements_from_kfile(nl, _machine(), groups=["mse"])
    g = measurements_from_kfile(nl, _machine(), groups=["mse"], er_corrected=False).groups["mse"]
    assert g.meta["er_corrected"] is False
    assert g.meta["er_corrected_source"] == "caller override"


def test_bundle_size_mismatch(kfile):
    m = _machine()
    m.sensors["bpol_probe"] = {"labels": [f"P{i}" for i in range(6)]}
    with pytest.raises(ValueError, match="6 sensors"):
        measurements_from_kfile(kfile, m, groups=["bpol_probe"])
