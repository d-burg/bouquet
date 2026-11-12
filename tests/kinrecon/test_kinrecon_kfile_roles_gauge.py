"""k-file adapter: coil roles in EFIT's fit and the flux-loop gauge (SIREF), on made-up k-files.

The rules under test are documented in ``bouquet.kinrecon.adapters.efit_kfile`` ("Coil roles",
"Flux-loop gauge").  Synthetic numbers only.
"""
import numpy as np
import pytest

from bouquet.kinrecon.adapters.efit_kfile import flux_gauge, measurements_from_kfile, parse_namelists
from bouquet.kinrecon.measurements import MachineBundle

_BASE = """\
 &IN1
 ISHOT = 1
 ITIME = 1500
 PLASMA = 1.0D+06
 FWTCUR = 1.0
 BITIP = 100.0
 COILS = {coils}
 FWTSI = 3*1.0
 PSIBIT = 3*1.0E-3
 SERROR = 0.03
 {siref}
 FWTFC = 1.0 0.0
 BRSP = 1000.0 -2000.0 6*0.0
 BITFC = 10.0 10.0
 FWTEC = 0.0 0.0
 ECURRT = 50.0 -60.0
 BITEC = 1.0 1.0
 {iecurr}
 IAVEM = 10
 /
"""


def _machine():
    sensors = {"flux_loop": {"labels": ["L0", "L1", "L2"], "r": [1.0, 1.1, 1.2], "z": [0.0, 0.5, -0.5]},
               "pf_current": {"labels": ["F0", "F1", "E0", "E1"]}}
    return MachineBundle(machine="toy", sensors=sensors,
                         channel_maps={"BRSP": ["F0", "F1"], "ECURRT": ["E0", "E1"]},
                         meta={"vbit_by_shot": [[0, 10.0]]})          # made-up default (VBIT not in the file)


def _ms(coils="0.0 -0.2 0.3", siref="SIREF = 0.5", iecurr="IECURR = 1", **kw):
    text = _BASE.format(coils=coils, siref=siref, iecurr=iecurr)
    kw.setdefault("sigma_model", "efit_vbit")
    return measurements_from_kfile(parse_namelists(text), _machine(), flux_reference="L0", **kw)


def test_coil_roles_follow_fit_weights_and_iecurr():
    pf = _ms().groups["pf_current"]
    assert pf.meta["coil_roles"] == {"F0": "fitted", "F1": "free", "E0": "held", "E1": "held"}
    assert pf.meta["IECURR"] == 1.0
    assert pf.meta["coil_arrays"] == {"BRSP": ["F0", "F1"], "ECURRT": ["E0", "E1"]}
    assert "'free' (free current, measurement not a constraint)" in pf.meta["coil_role_rule"]
    # scoring is unchanged: zero-weight coils are still scored with weight 1 (scoring convention)
    assert pf.valid.tolist() == [True] * 4 and pf.weight.tolist() == [1.0] * 4
    pf2 = _ms(iecurr="IECURR = 2").groups["pf_current"]
    assert pf2.meta["coil_roles"]["E0"] == "free" and pf2.meta["coil_roles"]["F0"] == "fitted"
    pf3 = _ms(iecurr="").groups["pf_current"]
    assert pf3.meta["coil_roles"]["E0"] == "undetermined" and pf3.meta["IECURR"] is None


def test_relative_storage_with_siref():
    ms = _ms()
    fl = ms.groups["flux_loop"]
    assert fl.meta["coils_form"] == "relative" and fl.meta["siref"] == 0.5
    assert ms.meta["SIREF"] == 0.5
    two_pi = 2 * np.pi
    assert fl.meta["absolute_offset"] == pytest.approx(-0.5 * two_pi, rel=1e-15)     # default flux_scale -2 pi
    # absolute flux = (COILS + SIREF) x flux_scale = measured + absolute_offset
    np.testing.assert_allclose(fl.measured + fl.meta["absolute_offset"],
                               -(np.array([0.0, -0.2, 0.3]) + 0.5) * two_pi, rtol=1e-15)


def _ms_noref(coils, siref):
    text = _BASE.format(coils=coils, siref=siref, iecurr="IECURR = 1")
    return measurements_from_kfile(parse_namelists(text), _machine(), sigma_model="efit_vbit")


def test_absolute_storage_and_undetermined():
    # absolute slices: a named reference is refused, the form is recorded without one
    for siref in ("SIREF = 0.0", "", "SIREF = 0.5"):
        with pytest.raises(ValueError, match="not in relative form"):
            _ms(coils="0.4 -0.2 0.3", siref=siref)
    fl = _ms_noref("0.4 -0.2 0.3", "SIREF = 0.0").groups["flux_loop"]
    assert fl.meta["coils_form"] == "absolute"              # SIREF = 0: absolute storage, decided per slice
    assert fl.meta["absolute_offset"] == 0.0
    fl = _ms_noref("0.4 -0.2 0.3", "SIREF = 0.5").groups["flux_loop"]
    assert fl.meta["coils_form"] == "undetermined"          # SIREF set but no reference to check
    g = flux_gauge({"SIREF": [0.5]}, np.array([0.0, 1.0]), None, 1.0)
    assert g["coils_form"] == "undetermined" and "no flux_reference" in g["coils_form_evidence"]
    assert flux_gauge({}, np.array([0.3, 1.0]), None, 1.0)["coils_form"] == "absolute"
