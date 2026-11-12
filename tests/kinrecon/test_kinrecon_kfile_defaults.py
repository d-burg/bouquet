"""k-file adapter defaults: sigma-model selection, the VBIT default, E-coil and Ip floors, the flux-loop storage
precedence, recorded exclusions and the probe averaging count (made-up k-files).

Expected values are computed by hand from the rules stated in ``bouquet.kinrecon.adapters.efit_kfile``.
"""
import numpy as np
import pytest

from bouquet.kinrecon.adapters.efit_kfile import (flux_gauge, flux_storage_series, measurements_from_kfile,
                                                  parse_namelists)
from bouquet.kinrecon.adapters.imas_json import measurements_from_imas
from bouquet.kinrecon.measurements import MachineBundle

R = np.array([1.0, 2.0, 1.5])
CONST = {"ERRSIL": 0.05, "SICONT": 2.0e-9, "NSLREF": 1}

_TEXT = """\
 &IN1
 {ishot}
 ITIME = 1500
 IAVEM = 10
 PLASMA = {ip}
 FWTCUR = 1.0
 BITIP = 100.0
 COILS = {coils}
 FWTSI = 3*1.0
 PSIBIT = 1.0E-4 2.0E-4 1.0E-4
 EXPMP2 = 0.5 -0.002
 FWTMP2 = 2*1.0
 BITMPI = 2*1.0E-3
 FWTFC = 1.0 0.0
 BRSP = 1000.0 -2000.0
 BITFC = 10.0 10.0
 FWTEC = 1.0
 ECURRT = 50.0
 BITEC = 1.0
 SERROR = 0.03
 {vbit}
 SIREF = {siref}
 KERSIL = 3
 {fitsiref}
 IECURR = {iecurr}
 /
"""


def _nl(coils="0.0 -0.2 0.3", siref="0.5", ip="1.0D+06", ishot="ISHOT = 1", vbit="VBIT = 10.0",
        fitsiref="FITSIREF = T", iecurr=1):
    return parse_namelists(_TEXT.format(coils=coils, siref=siref, ip=ip, ishot=ishot, vbit=vbit,
                                        fitsiref=fitsiref, iecurr=iecurr))


def _machine(meta=None):
    sensors = {"flux_loop": {"labels": ["L0", "L1", "L2"], "r": R.tolist()},
               "bpol_probe": {"labels": ["P0", "P1"], "r": [1.0, 1.0], "z": [0.0, 0.1],
                              "direction_r": [0.0, 0.0], "direction_z": [1.0, 1.0]},
               "pf_current": {"labels": ["F0", "F1", "E0"]}}
    return MachineBundle(machine="toy", sensors=sensors, channel_maps={"BRSP": ["F0", "F1"], "ECURRT": ["E0"]},
                         meta=dict(meta or {}))


MAG = ["flux_loop", "bpol_probe", "ip", "pf_current"]


# ----------------------------------------------------------------------------- default sigma model
def test_default_uses_stored_sigma_when_given():
    stored = {"flux_loop": [1e-3, 2e-3, 3e-3], "bpol_probe": [0.02, 0.03], "ip": [4e4],
              "pf_current": [5.0, 6.0, 7.0]}
    ms = measurements_from_kfile(_nl(), _machine(), groups=MAG, flux_scale=1.0, stored_sigma=stored)
    assert ms.meta["sigma_model"] == "efit_stored" and ms.meta["sigma_model_requested"] == "default"
    assert "stored sigma supplied" in ms.meta["sigma_model_selection"]
    assert ms.meta["sigma_models"] == {g: "efit_stored" for g in MAG}
    np.testing.assert_allclose(ms.groups["flux_loop"].sigma, [1e-3, 2e-3, 3e-3], rtol=1e-15)
    np.testing.assert_allclose(ms.groups["ip"].sigma, [4e4], rtol=1e-15)


def test_default_without_stored_sigma_is_the_inferred_loop_rule():
    ms = measurements_from_kfile(_nl(), _machine({"efit_sigma_constants": CONST}), groups=MAG, flux_scale=1.0)
    assert ms.meta["sigma_model"] == "efit_kersil" and ms.meta["sigma_model_requested"] == "default"
    assert set(ms.meta["sigma_models"].values()) == {"efit_kersil:siref"}
    fl = ms.groups["flux_loop"]
    assert fl.meta["sigma_model_id"] == "efit_kersil:siref" and fl.meta["kersil"]["reading"] == "siref"
    assert fl.meta["kersil"]["NSLREF_source"].startswith("machine bundle")
    c, bit = np.array([0.0, -0.2, 0.3]), 10.0 * np.array([1e-4, 2e-4, 1e-4])
    want = np.maximum.reduce([0.05 * np.abs(c), 2e-9 * R * 1e6, bit])
    want[0] = max(0.05 * 0.5, 2e-9 * R[0] * 1e6, bit[0])               # reference loop: |SIREF|
    np.testing.assert_allclose(fl.sigma, want, rtol=1e-14)
    np.testing.assert_allclose(ms.groups["bpol_probe"].sigma, [0.015, 0.01], rtol=1e-14)
    np.testing.assert_allclose(ms.groups["pf_current"].sigma, [100.0, 100.0, 10.0], rtol=1e-14)


def test_default_inputs_are_required_and_named():
    with pytest.raises(ValueError, match="ERRSIL"):
        measurements_from_kfile(_nl(), _machine({"efit_sigma_constants": {"NSLREF": 1}}), groups=["flux_loop"])
    with pytest.raises(ValueError, match="NSLREF"):
        measurements_from_kfile(_nl(), _machine(), groups=["flux_loop"], efit_constants=CONST)
    with pytest.raises(ValueError, match="efit_kersil"):
        measurements_from_kfile(_nl(), _machine({"efit_sigma_constants": CONST}), groups=["flux_loop"],
                                kersil_reference="source")
    # the retired default stays selectable by name and is recorded as such
    ms = measurements_from_kfile(_nl(), _machine(), groups=["flux_loop"], sigma_model="efit_vbit")
    assert ms.meta["sigma_models"] == {"flux_loop": "efit_vbit"}
    assert ms.meta["sigma_model_selection"] == "named by the caller"


def test_ip_floor_binds_under_the_default():
    ms = measurements_from_kfile(_nl(ip="1.0D+03"), _machine({"efit_sigma_constants": CONST}), groups=["ip"])
    assert ms.groups["ip"].sigma[0] == pytest.approx(10.0 * 100.0, rel=1e-15)     # VBIT BITIP > SERROR |Ip|
    ms = measurements_from_kfile(_nl(ip="1.0D+06"), _machine({"efit_sigma_constants": CONST}), groups=["ip"])
    assert ms.groups["ip"].sigma[0] == pytest.approx(0.03 * 1e6, rel=1e-15)


def test_ecoil_floor_and_its_basis():
    # ECURRT = 50, SERROR |ECURRT| = 1.5 < VBIT BITEC = 10: the floor binds
    for iecurr, own in ((2, True), (1, False)):
        pf = measurements_from_kfile(_nl(iecurr=iecurr), _machine(), groups=["pf_current"],
                                     sigma_model="efit_kersil", kersil_reference="siref").groups["pf_current"]
        assert pf.sigma[2] == pytest.approx(10.0, rel=1e-15)
        assert pf.meta["ecoil_sigma_basis"].startswith("EFIT's own") is own


# ----------------------------------------------------------------------------- VBIT default
# a made-up piecewise default by shot: 10 from shot 0, 80 from shot 1001 (boundary shot 1000 -> 10)
TABLE = {"vbit_by_shot": [[0, 10.0], [1001, 80.0]], "vbit_by_shot_source": "made-up test table"}


@pytest.mark.parametrize("shot, want", [(1000, 10.0), (1001, 80.0), (1, 10.0)])
def test_vbit_default_from_the_bundle_table(shot, want):
    ms = measurements_from_kfile(_nl(ishot=f"ISHOT = {shot}", vbit=""), _machine(TABLE), groups=["bpol_probe"],
                                 sigma_model="efit_vbit")
    assert ms.meta["VBIT"] == want and "absent" in ms.meta["VBIT_source"]
    assert "made-up test table" in ms.meta["VBIT_source"]
    np.testing.assert_allclose(ms.groups["bpol_probe"].sigma, np.maximum(0.03 * np.array([0.5, 0.002]),
                                                                         want * 1e-3), rtol=1e-15)


def test_vbit_default_is_never_assumed():
    # no file value, no caller value, no bundle table: refused for a model that needs VBIT
    with pytest.raises(ValueError, match="vbit_by_shot"):
        measurements_from_kfile(_nl(vbit=""), _machine(), groups=["bpol_probe"], sigma_model="efit_vbit")
    # the table needs the shot number, and the shot must not precede its first entry
    with pytest.raises(ValueError, match="ISHOT"):
        measurements_from_kfile(_nl(ishot="", vbit=""), _machine(TABLE), groups=["bpol_probe"],
                                sigma_model="efit_vbit")
    m = _machine({"vbit_by_shot": [[500, 10.0]]})
    with pytest.raises(ValueError, match="precedes"):
        measurements_from_kfile(_nl(ishot="ISHOT = 1", vbit=""), m, groups=["bpol_probe"], sigma_model="efit_vbit")
    ms = measurements_from_kfile(_nl(ishot="", vbit=""), _machine(), groups=["bpol_probe"], sigma_model="efit_vbit",
                                 vbit=40.0)
    assert ms.meta["VBIT"] == 40.0 and ms.meta["VBIT_source"].startswith("caller")
    # a model that does not use VBIT is not blocked by it
    ms = measurements_from_kfile(_nl(ishot="", vbit=""), _machine(), groups=["bpol_probe"],
                                 stored_sigma={"bpol_probe": [0.1, 0.1]})
    assert "VBIT" not in ms.meta
    with pytest.raises(ValueError, match="remove one"):
        measurements_from_kfile(_nl(), _machine(), groups=["bpol_probe"], sigma_model="efit_vbit", vbit=40.0)


# ----------------------------------------------------------------------------- flux-loop storage precedence
def test_storage_precedence_fitsiref_then_exact_values():
    # FITSIREF = T decides relative although the reference entry is not zero
    g = flux_gauge(_nl(coils="-2.0e-4 -0.2 0.3")["IN1"], np.array([-2.0e-4, -0.2, 0.3]), 0, 1.0)
    assert (g["coils_form"], g["coils_form_rule"]) == ("relative", "fitsiref")
    # without the flag: an exactly zero reference entry with SIREF set -> relative; SIREF = 0 -> absolute
    in1 = _nl(fitsiref="")["IN1"]
    g = flux_gauge(in1, np.array([0.0, -0.2, 0.3]), 0, 1.0)
    assert (g["coils_form"], g["coils_form_rule"]) == ("relative", "reference_entry_zero")
    g = flux_gauge(_nl(fitsiref="", siref="0.0")["IN1"], np.array([0.4, -0.2, 0.3]), 0, 1.0)
    assert (g["coils_form"], g["coils_form_rule"]) == ("absolute", "siref_zero")
    # a small but nonzero entry is not decided by one slice (no ratio threshold)
    g = flux_gauge(in1, np.array([1e-12, -0.2, 0.3]), 0, 1.0)
    assert g["coils_form"] == "undetermined" and g["coils_form_rule"] is None
    with pytest.raises(ValueError, match="not in relative form"):
        measurements_from_kfile(_nl(coils="1e-12 -0.2 0.3", fitsiref=""), _machine(), groups=["flux_loop"],
                                sigma_model="efit_vbit", flux_reference="L0")


def test_storage_series_rule():
    # relative storage: the reference entry stays constant while SIREF swings
    rel = [_nl(coils=f"-2.3e-4 {-0.2 - k * 0.01} 0.3", siref=str(0.1 * (k + 1)), fitsiref="") for k in range(4)]
    out = flux_storage_series(rel, 0)
    assert out["coils_form"] == "relative" and out["n_slices"] == 4
    ms = measurements_from_kfile(rel[1], _machine(), groups=["flux_loop"], sigma_model="efit_vbit",
                                 flux_reference="L0", flux_storage=out)
    assert ms.groups["flux_loop"].meta["coils_form_rule"] == "series"
    # absolute storage: the reference entry follows SIREF
    ab = [_nl(coils=f"{0.1 * k + 1e-3} -0.2 0.3", siref=str(0.1 * k + 1.1e-3), fitsiref="") for k in range(4)]
    assert flux_storage_series(ab, 0)["coils_form"] == "absolute"
    # one slice, or SIREF not changing: undetermined
    assert flux_storage_series(rel[:1], 0)["coils_form"] == "undetermined"
    same = [_nl(coils=f"{1e-3 * k} -0.2 0.3", siref="0.5", fitsiref="") for k in range(1, 4)]
    assert flux_storage_series(same, 0)["coils_form"] == "undetermined"
    # slices the file decides are left out of the series and keep their own decision
    mixed = rel + [_nl(siref="0.0", coils="0.4 -0.2 0.3", fitsiref="")]
    assert flux_storage_series(mixed, 0)["n_slices"] == 4
    g = measurements_from_kfile(mixed[-1], _machine(), groups=["flux_loop"], sigma_model="efit_vbit",
                                flux_storage="relative").groups["flux_loop"]
    assert (g.meta["coils_form"], g.meta["coils_form_rule"]) == ("absolute", "siref_zero")
    g = measurements_from_kfile(rel[1], _machine(), groups=["flux_loop"], sigma_model="efit_vbit",
                                flux_reference="L0", flux_storage="relative").groups["flux_loop"]
    assert g.meta["coils_form_rule"] == "caller"
    with pytest.raises(ValueError, match="flux_storage"):
        flux_gauge(rel[1]["IN1"], np.array([-2.3e-4, -0.2, 0.3]), 0, 1.0, series="maybe")


# ----------------------------------------------------------------------------- exclusions, probe averaging
def test_exclusions_are_recorded_not_dropped():
    m = _machine({"channel_exclusions": {"bpol_probe": {"P1": "fit weight 0 in the device's own fits"}}})
    ms = measurements_from_kfile(_nl(), m, groups=["bpol_probe", "ip"], sigma_model="efit_vbit",
                                 exclude={"ip": {"ip": "caller"}})
    pr = ms.groups["bpol_probe"]
    assert pr.valid.tolist() == [True, False] and pr.labels == ["P0", "P1"]
    assert np.isfinite(pr.measured[1]) and np.isfinite(pr.sigma[1])        # kept, only invalid
    assert pr.meta["excluded"] == [("P1", "fit weight 0 in the device's own fits")]
    assert ms.groups["ip"].meta["excluded"] == [("ip", "caller")]
    assert ms.meta["channel_exclusions_applied"] == {"bpol_probe": ["P1"], "ip": ["ip"]}
    with pytest.raises(ValueError, match="not a channel"):
        measurements_from_kfile(_nl(), m, groups=["bpol_probe"], sigma_model="efit_vbit",
                                exclude={"bpol_probe": {"P9": "typo"}})


def test_imas_route_exclusions():
    ids = {"magnetics": {"b_field_pol_probe": [
        {"name": n, "position": {"r": 1.0, "z": 0.0}, "poloidal_angle": 0.0,
         "field": {"time": [1.0], "data": [0.1], "data_error_upper": [0.01]}} for n in ("A", "B")]}}
    ms = measurements_from_imas(ids, 1.0, 0.01, groups=["bpol_probe"], exclude={"bpol_probe": {"B": "fwt 0"}})
    assert ms.groups["bpol_probe"].valid.tolist() == [True, False]
    assert ms.groups["bpol_probe"].meta["excluded"] == [("B", "fwt 0")]


def test_probe_averaging_count_is_recorded():
    pr = measurements_from_kfile(_nl(), _machine(), groups=["bpol_probe"],
                                 sigma_model="efit_vbit").groups["bpol_probe"]
    assert pr.meta["probe_n_avg"] == 25 and pr.meta["probe_n_avg_source"].startswith("default 25")
    m = _machine({"probe_n_avg": 13, "probe_n_avg_source": "device file NSMP2"})
    pr = measurements_from_kfile(_nl(), m, groups=["bpol_probe"], sigma_model="efit_vbit").groups["bpol_probe"]
    assert pr.meta["probe_n_avg"] == 13 and "NSMP2" in pr.meta["probe_n_avg_source"]
