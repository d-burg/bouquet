"""k-file adapter: the named EFIT flux-loop sigma models and the refusal of silent defaults (made-up k-files).

Expected sigma values are computed by hand in each test from the rule stated in the module docstring
(``efit_kersil``: max(ERRSIL |psi_m - psi_ref|, SICONT R_m |Ip|, VBIT PSIBIT_m), reference loop by KERSIL),
not by calling the adapter's helpers.  Exact equality up to round-off (rtol 1e-14).
"""
import numpy as np
import pytest

from bouquet.kinrecon.adapters.efit_kfile import measurements_from_kfile, parse_namelists
from bouquet.kinrecon.measurements import MachineBundle

R = np.array([1.0, 2.0, 1.5])
CONST = {"ERRSIL": 0.05, "SICONT": 2.0e-9, "ERSIL8": 7.0e-3}

_TEXT = """\
 &IN1
 ITIME = 1500
 IAVEM = 10
 PLASMA = {ip}
 FWTCUR = 1.0
 BITIP = 100.0
 COILS = {coils}
 FWTSI = 3*1.0
 PSIBIT = 1.0E-4 2.0E-4 1.0E-4
 SERROR = 0.03
 VBIT = 10.0
 SIREF = {siref}
 KERSIL = {kersil}
 FITSIREF = {fitsiref}
 IECURR = 1
 {extra}
 /
"""


def _nl(coils="0.0 -0.2 0.3", siref="0.5", kersil=3, fitsiref="T", ip="1.0D+06", extra=""):
    return parse_namelists(_TEXT.format(coils=coils, siref=siref, kersil=kersil, fitsiref=fitsiref, ip=ip,
                                        extra=extra))


def _machine(meta=None):
    return MachineBundle(machine="toy", sensors={"flux_loop": {"labels": ["L0", "L1", "L2"], "r": R.tolist()}},
                         meta=meta or {})


def _loops(nl, **kw):
    kw.setdefault("sigma_model", "efit_kersil")
    return measurements_from_kfile(nl, _machine(kw.pop("meta", None)), groups=["flux_loop"], flux_scale=1.0,
                                   **kw).groups["flux_loop"]


def test_kersil3_siref_reading_relative_storage():
    g = _loops(_nl(), kersil_reference="siref", efit_constants=CONST, nslref=1)
    c = np.array([0.0, -0.2, 0.3])
    bit = 10.0 * np.array([1e-4, 2e-4, 1e-4])
    want = np.maximum.reduce([0.05 * np.abs(c), 2e-9 * R * 1e6, bit])
    want[0] = max(0.05 * 0.5, 2e-9 * R[0] * 1e6, bit[0])             # reference: |SIREF|
    np.testing.assert_allclose(g.sigma, want, rtol=1e-14)
    k = g.meta["kersil"]
    assert k["reading"] == "siref" and k["NSLREF_source"].startswith("caller")
    assert k["constants_source"]["ERRSIL"] == "caller (efit_constants)"


def test_kersil3_siref_reading_absolute_storage():
    # SIREF = 0: absolute storage; loops enter relative to the stored reference entry, the reference with |SIREF|
    g = _loops(_nl(coils="0.4 -0.2 0.3", siref="0.0"), kersil_reference="siref", efit_constants=CONST, nslref=1)
    c = np.array([0.4, -0.2, 0.3])
    bit = 10.0 * np.array([1e-4, 2e-4, 1e-4])
    want = np.maximum.reduce([0.05 * np.abs(c - 0.4), 2e-9 * R * 1e6, bit])
    want[0] = max(0.0, 2e-9 * R[0] * 1e6, bit[0])
    np.testing.assert_allclose(g.sigma, want, rtol=1e-14)


def test_kersil3_source_reading_adds_siref_when_not_fitsiref():
    g = _loops(_nl(coils="0.01 -0.2 0.3", fitsiref="F"), kersil_reference="source", efit_constants=CONST, nslref=1)
    psi = np.array([0.01, -0.2, 0.3]) + 0.5                          # FITSIREF false, IECURR > 0: SIREF added
    bit = 10.0 * np.array([1e-4, 2e-4, 1e-4])
    want = np.maximum.reduce([0.05 * np.abs(psi - psi[0]), 2e-9 * R * 1e6, bit])
    want[0] = max(0.05 * abs(psi[0]), 2e-9 * R[0] * 1e6, bit[0])
    np.testing.assert_allclose(g.sigma, want, rtol=1e-14)
    g2 = _loops(_nl(coils="0.01 -0.2 0.3", fitsiref="T"), kersil_reference="source", efit_constants=CONST, nslref=1)
    psi2 = np.array([0.01, -0.2, 0.3])                                # FITSIREF true: nothing added
    want2 = np.maximum.reduce([0.05 * np.abs(psi2 - psi2[0]), 2e-9 * R * 1e6, bit])
    want2[0] = max(0.05 * abs(psi2[0]), 2e-9 * R[0] * 1e6, bit[0])
    np.testing.assert_allclose(g2.sigma, want2, rtol=1e-14)


def test_kersil2_reference_is_ersil8_and_other_kersil_uses_serror():
    g = _loops(_nl(kersil=2), kersil_reference="siref", efit_constants=CONST, nslref=1)
    assert g.sigma[0] == pytest.approx(7.0e-3, rel=1e-15)
    g = _loops(_nl(kersil=1, coils="0.0 -0.2 0.3"), kersil_reference="siref", efit_constants=CONST, nslref=1)
    np.testing.assert_allclose(g.sigma, np.maximum(0.03 * np.abs([0.0, -0.2, 0.3]), 10 * np.array([1e-4, 2e-4, 1e-4])),
                               rtol=1e-14)


def test_kersil_constants_and_reading_are_required_not_assumed():
    with pytest.raises(ValueError, match="kersil_reference"):
        _loops(_nl(), efit_constants=CONST, nslref=1)
    with pytest.raises(ValueError, match="ERRSIL"):
        _loops(_nl(), kersil_reference="siref", nslref=1)
    with pytest.raises(ValueError, match="NSLREF"):
        _loops(_nl(), kersil_reference="siref", efit_constants=CONST)
    # from the machine bundle, recorded as such; from the file when present
    g = _loops(_nl(), kersil_reference="siref", nslref=1, meta={"efit_sigma_constants": CONST})
    assert g.meta["kersil"]["constants_source"]["SICONT"].startswith("machine bundle")
    g = _loops(_nl(extra="ERRSIL = 0.04\n NSLREF = 1"), kersil_reference="siref", efit_constants=CONST)
    assert g.meta["kersil"]["constants"]["ERRSIL"] == 0.04
    assert g.meta["kersil"]["constants_source"]["ERRSIL"] == "k-file &IN1"
    assert g.meta["kersil"]["NSLREF_source"] == "k-file &IN1"


def test_efit_stored_sigma_model():
    stored = {"flux_loop": [1e-3, 2e-3, 0.0]}
    g = measurements_from_kfile(_nl(), _machine(), groups=["flux_loop"], flux_scale=-2 * np.pi,
                                sigma_model="efit_stored", stored_sigma=stored).groups["flux_loop"]
    np.testing.assert_allclose(g.sigma[:2], np.array([1e-3, 2e-3]) * 2 * np.pi, rtol=1e-15)
    assert not g.valid[2] and np.isnan(g.sigma[2])                    # below EFIT's floor: weight zeroed
    with pytest.raises(ValueError, match="stored_sigma"):
        measurements_from_kfile(_nl(), _machine(), groups=["flux_loop"], sigma_model="efit_stored")
    with pytest.raises(ValueError, match="no entry"):
        measurements_from_kfile(_nl(), _machine(), groups=["flux_loop", "ip"], sigma_model="efit_stored",
                                stored_sigma=stored)


def test_kersil_ip_has_the_bit_floor():
    nl = _nl(ip="1.0D+03")                                          # small Ip: VBIT BITIP = 1000 binds
    ip = measurements_from_kfile(nl, _machine(), groups=["ip"], sigma_model="efit_kersil", kersil_reference="siref",
                                 efit_constants=CONST, nslref=1).groups["ip"]
    assert ip.sigma[0] == pytest.approx(1000.0, rel=1e-15)
    ip = measurements_from_kfile(nl, _machine(), groups=["ip"]).groups["ip"]          # efit_vbit: no floor
    assert ip.sigma[0] == pytest.approx(30.0, rel=1e-15)


# ----------------------------------------------------------------------------- silent defaults refused
def test_missing_serror_itime_iavem_bits_and_weights_are_refused():
    base = _TEXT.format(coils="0.0 -0.2 0.3", siref="0.5", kersil=3, fitsiref="T", ip="1e6", extra="")

    def drop(key):
        return parse_namelists("\n".join(ln for ln in base.splitlines() if not ln.strip().startswith(key)) + "\n")

    with pytest.raises(ValueError, match="SERROR"):
        measurements_from_kfile(drop("SERROR"), _machine(), groups=["flux_loop"])
    g = measurements_from_kfile(drop("SERROR"), _machine(), groups=["flux_loop"], serror=0.03)
    assert g.meta["SERROR_source"].startswith("caller")
    with pytest.raises(ValueError, match="ITIME"):
        measurements_from_kfile(drop("ITIME"), _machine(), groups=["flux_loop"])
    assert measurements_from_kfile(drop("ITIME"), _machine(), groups=["flux_loop"], time=1.5).meta["time_source"] \
        .startswith("caller")
    with pytest.raises(ValueError, match="IAVEM"):
        measurements_from_kfile(drop("IAVEM"), _machine(), groups=["flux_loop"])
    with pytest.raises(ValueError, match="PSIBIT"):
        measurements_from_kfile(drop("PSIBIT"), _machine(), groups=["flux_loop"])
    g = measurements_from_kfile(drop("PSIBIT"), _machine(), groups=["flux_loop"], allow_missing_bits=True)
    assert "allow_missing_bits" in g.groups["flux_loop"].meta["bit_source"]
    with pytest.raises(ValueError, match="FWTSI"):
        measurements_from_kfile(drop("FWTSI"), _machine(), groups=["flux_loop"])


def test_arrays_longer_than_the_bundle():
    nl = _nl(coils="0.0 -0.2 0.3 0.0 0.0")
    nl["IN1"]["FWTSI"] = [1.0, 1.0, 1.0, 0.0, 0.0]
    ms = measurements_from_kfile(nl, _machine(), groups=["flux_loop"])
    assert ms.meta["truncated_trailing_zeros"]["COILS"] == 2
    nl = _nl(coils="0.0 -0.2 0.3 0.7")
    with pytest.raises(ValueError, match="beyond"):
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"])


def test_null_entries_are_positional_and_recorded():
    nl = _nl(coils="0.0, , 0.3")
    with pytest.raises(ValueError, match="2 values for 3 sensors"):
        measurements_from_kfile(nl, _machine(), groups=["flux_loop"])
    nl = _nl(extra="AA1GAM = 1.0, , 3.0")
    ms = measurements_from_kfile(nl, _machine(), groups=["flux_loop"])
    assert ms.meta["namelist_null_entries"] == {"IN1.AA1GAM": [1]}


def test_parser_legal_forms():
    p = parse_namelists
    assert p(" &A\n X = 1.0, , 3.0\n /\n")["A"]["X"] == [1.0, None, 3.0]
    assert p(" &A\n X = 2*, 3.0\n /\n")["A"]["X"] == [None, None, 3.0]
    assert p(" &A\n X(2:3) = 5 6\n /\n")["A"]["X"] == [None, 5.0, 6.0]
    assert p(" &A\n X = 1 2 3\n X(2) = , 9\n /\n")["A"]["X"] == [1.0, 2.0, 9.0]    # null leaves element 2
    assert p(" &A\n S = 'it''s, x=1'\n X = 1\n /\n")["A"] == {"S": ["it's, x=1"], "X": [1.0]}
    assert p(" &A\n X = 1 2 ! comment with Y = 3\n /\n")["A"] == {"X": [1.0, 2.0]}
    for bad, msg in ((" &A\n C(1,2) = 5\n X = 1\n /\n", "multi-dimensional"),
                     (" &A\n X = 1\n C(1,2) = 5\n /\n", "multi-dimensional"),
                     (" &A\n junk X = 1\n /\n", "before the first key"),
                     (" &A\n X(2:3) = 1 2 3\n /\n", "range"),
                     (" &A\n X = 1.0 **** 2.0\n /\n", "count")):
        with pytest.raises(ValueError, match=msg):
            p(bad)
