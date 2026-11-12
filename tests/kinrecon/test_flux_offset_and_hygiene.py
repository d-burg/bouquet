"""Fitted flux-loop offset in the scorer (``score`` and ``score_common``), its counts and records, the k-file
reading of a fitted reference, the solve gauge, and the uncertainty-model / vessel switches of a k-file.

The scorer's diagnostics are replaced by stated model values (no OFT), so the tests check exactly the
offset logic of ``bouquet.kinrecon.score``.  Synthetic numbers only.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon.adapters.efit_kfile import measurements_from_kfile, parse_namelists, vessel_switches
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
from bouquet.kinrecon.measurements import MachineBundle

SC = importlib.import_module("bouquet.kinrecon.score")      # the module (the package exports the function)

SIG = np.array([1.0, 2.0, 1.5, 0.5, 1.0])
MEAS = np.array([10.0, 11.0, 9.0, 10.5, 12.0])


def _set(prior=None, ref=None, valid=None, weight=None):
    meta = dict(flux_offset="fitted", sigma_model="test")
    if prior is not None:
        meta["flux_offset_prior"] = list(prior)
    if ref is not None:
        meta["flux_offset_reference"] = ref
    g = ConstraintGroup("flux_loop", ["L0", "L1", "L2", "L3", "L4"], MEAS.copy(), SIG.copy(),
                        np.ones(5, bool) if valid is None else np.asarray(valid, bool),
                        weight=None if weight is None else np.asarray(weight, float), meta=meta)
    return MeasurementSet("toy", 1.0, 0.01, {"flux_loop": g})


def _install(monkeypatch, models):
    """``score`` evaluates ``models[eq]`` for the equilibrium object ``eq`` (an index)."""
    def model_group(eq, name, group, kinetics, notes, frame=None):
        v = np.asarray(models[eq], float)
        return SimpleNamespace(value=v, reason=["off mesh" if not np.isfinite(x) else "" for x in v])
    monkeypatch.setattr(SC, "_model_group", model_group)
    monkeypatch.setattr(SC, "resolve_orientation", lambda eq, ms, o=None: dict(sign_ip=1.0, sign_bt=1.0,
                                                                                  source="test"))


def _closed_form(model, use, prior=None):
    w = 1.0 / SIG[use] ** 2
    num, den = np.sum(w * (MEAS[use] - model[use])), np.sum(w)
    if prior is not None:
        num, den = num + prior[0] / prior[1] ** 2, den + 1.0 / prior[1] ** 2
    return num / den


def test_score_applies_the_fitted_offset(monkeypatch):
    """The hook in ``score``: an absolute model shifted by a constant is scored with the closed-form offset."""
    model = MEAS - 3.0 + np.array([0.2, -0.4, 0.3, 0.1, -0.2])
    _install(monkeypatch, {0: model})
    tab = SC.score(0, _set(), groups=["flux_loop"])
    c = _closed_form(model, np.ones(5, bool))
    assert tab.frame["flux_offset"] == pytest.approx(c, rel=1e-12)
    np.testing.assert_allclose(tab.model["flux_loop"], model + c, rtol=1e-14)
    # without the offset the chi2 would carry the constant 3: it must be far smaller
    assert tab.chi2("flux_loop") < 0.5
    rec = SC.flux_offset_record(tab)
    assert rec["offset"] == pytest.approx(c) and rec["n_extra_params"] == 1 and rec["n_extra_data"] == 0


def test_score_common_refits_the_offset_on_the_common_channels(monkeypatch):
    """The re-fit in ``score_common``: one equilibrium cannot model L4, so the offset of BOTH tables is
    re-solved on L0-L3 (differs from the per-table offset on all channels)."""
    m0 = MEAS - 1.0 + np.array([0.0, 0.5, -0.5, 0.2, 5.0])     # L4 far off: moves the offset when included
    m1 = MEAS - 2.0 + np.array([0.1, -0.1, 0.2, 0.0, np.nan])
    _install(monkeypatch, {0: m0, 1: m1})
    alone = SC.score(0, _set(), groups=["flux_loop"])
    tabs, rec = SC.score_common([0, 1], _set(), groups=["flux_loop"])
    common = np.array([True, True, True, True, False])
    for t, m in zip(tabs, (m0, m1)):
        assert t.frame["flux_offset"] == pytest.approx(_closed_form(m, common), rel=1e-12)
    assert abs(alone.frame["flux_offset"] - tabs[0].frame["flux_offset"]) > 0.1
    assert rec["flux_loop"]["n_common"] == 4


def test_prior_row_counts_and_report(monkeypatch):
    model = MEAS - 3.0 + np.array([0.2, -0.4, 0.3, 0.1, -0.2])
    _install(monkeypatch, {0: model})
    prior = (2.5, 0.8)
    tab = SC.score(0, _set(prior=prior, ref="L0"), groups=["flux_loop"])
    c = _closed_form(model, np.ones(5, bool), prior)
    assert tab.frame["flux_offset"] == pytest.approx(c, rel=1e-12)
    pc = ((c - prior[0]) / prior[1]) ** 2
    assert tab.frame["flux_offset_prior_chi2"] == pytest.approx(pc, rel=1e-12)
    rep = SC.chi2_report(tab)
    # one extra datum (the prior row) and one extra parameter (the offset); the group chi2/N stays loops-only
    assert rep["total"]["n_channels"] == 6 and rep["total"]["n_params"] == 1 and rep["total"]["dof"] == 5
    assert rep["total"]["chi2"] == pytest.approx(tab.chi2() + pc, rel=1e-12)
    assert rep["groups"]["flux_loop"]["n_channels"] == 5
    assert rep["flux_offset"]["prior_chi2"] == pytest.approx(pc)


@pytest.mark.parametrize("off", ["invalid", "weight0"])
def test_no_prior_row_when_the_reference_loop_is_switched_off(monkeypatch, off):
    model = MEAS - 3.0 + np.array([0.2, -0.4, 0.3, 0.1, -0.2])
    _install(monkeypatch, {0: model})
    valid = [False, True, True, True, True] if off == "invalid" else None
    weight = [0.0, 1.0, 1.0, 1.0, 1.0] if off == "weight0" else None
    tab = SC.score(0, _set(prior=(2.5, 0.8), ref="L0", valid=valid, weight=weight), groups=["flux_loop"])
    use = np.array([False, True, True, True, True])
    assert tab.frame["flux_offset"] == pytest.approx(_closed_form(model, use), rel=1e-12)
    assert "flux_offset_prior_chi2" not in tab.frame and "switched off" in tab.frame["flux_offset_prior_dropped"]
    assert SC.flux_offset_record(tab)["n_extra_data"] == 0


# ---------------------------------------------------------------- k-file reading of a fitted reference
_K = """\
 &IN1
 ISHOT = 1
 ITIME = 1500
 PLASMA = 1.0D+06
 FWTCUR = 1.0
 BITIP = 100.0
 COILS = 0.001 -0.2 0.3
 FWTSI = {fwtsi}
 PSIBIT = 3*1.0E-3
 SERROR = 0.03
 SIREF = 0.5
 FITSIREF = T
 NSLREF = 1
 FWTFC = 1.0 0.0
 BRSP = 1000.0 -2000.0 6*0.0
 BITFC = 10.0 10.0
 FWTEC = 0.0 0.0
 ECURRT = 50.0 -60.0
 BITEC = 1.0 1.0
 IECURR = 1
 IAVEM = 10
 {extra}
 /
"""


def _machine():
    sensors = {"flux_loop": {"labels": ["L0", "L1", "L2"], "r": [1.0, 1.1, 1.2], "z": [0.0, 0.5, -0.5]},
               "pf_current": {"labels": ["F0", "F1", "E0", "E1"]}}
    return MachineBundle(machine="toy", sensors=sensors, channel_maps={"BRSP": ["F0", "F1"], "ECURRT": ["E0", "E1"]},
                         meta={"vbit_by_shot": [[0, 10.0]]})


def _kms(fwtsi="3*1.0", extra="", **kw):
    kw.setdefault("sigma_model", "efit_vbit")
    return measurements_from_kfile(parse_namelists(_K.format(fwtsi=fwtsi, extra=extra)), _machine(), **kw)


def test_fitsiref_file_is_scored_with_the_fitted_offset_and_prior():
    fl = _kms(flux_reference="L0").groups["flux_loop"]
    assert fl.meta["coils_form_rule"] == "fitsiref" and fl.meta["flux_offset"] == "fitted"
    assert fl.meta["flux_reference"] is None and fl.meta["flux_reference_requested"] == "L0"
    assert fl.meta["flux_offset_reference"] == "L0"
    c0, s0 = fl.meta["flux_offset_prior"]
    assert c0 == pytest.approx(2 * np.pi * 0.5, rel=1e-14)               # -flux_scale x SIREF (measured)
    assert s0 == pytest.approx(fl.sigma[0], rel=1e-14)
    assert fl.meta["absolute_offset"] == pytest.approx(-2 * np.pi * 0.5)  # the solve gauge: measured SIREF
    # the reference loop from NSLREF when no flux_reference is named
    assert _kms().groups["flux_loop"].meta["flux_offset_reference"] == "L0"


def test_fitsiref_reference_loop_switched_off_has_no_prior_and_old_reading_is_an_option():
    fl = _kms(fwtsi="0.0 1.0 1.0").groups["flux_loop"]
    assert "flux_offset_prior" not in fl.meta and "switched off" in fl.meta["flux_offset_prior_note"]
    old = _kms(flux_reference="L0", fitsiref_scoring="reference_loop").groups["flux_loop"]
    assert old.meta.get("flux_offset") is None and old.meta["flux_reference"] == "L0"
    with pytest.raises(ValueError, match="fitsiref_scoring"):
        _kms(fitsiref_scoring="bogus")


def test_calibration_sigma_option_is_refused_for_rule_models_and_recorded():
    with pytest.raises(ValueError, match="IMAGSIGMA"):
        _kms(extra="IMAGSIGMA = 1")
    ms = _kms(extra="IMAGSIGMA = 1\n ERRMAG = 1.0E-3", allow_calibration_sigma_option=True)
    cal = ms.meta["calibration_sigma_option"]
    assert cal["on"] and cal["IMAGSIGMA"] == 1.0 and cal["ERRMAG"] == 1.0e-3 and "override" in cal
    assert _kms().meta["calibration_sigma_option"]["on"] is False


def test_vessel_switches_recorded():
    v = _kms(extra="IFITVS = 0").meta["vessel"]
    assert v["IFITVS"] == 0.0 and v["IVESEL"] is None and v["prescription_known"] is False
    v2 = vessel_switches(parse_namelists(" &IN1\n IVESEL = 3\n /\n &EFITIN\n VLOOP = 1.2\n /\n"))
    assert v2["IVESEL"] == 3.0 and v2["prescription_known"] and v2["VLOOP"] == 1.2


def test_forward_solve_refuses_a_gauge_from_a_fitted_reference():
    from bouquet.kinrecon import forward as FW
    ms = _set()
    ms.groups["flux_loop"].meta.update(reference_convention="fitted", absolute_offset=0.0, coils_form="relative")
    ms.groups["ip"] = ConstraintGroup("ip", ["ip"], np.array([1e6]), np.array([3e4]), np.array([True]))
    fm = object.__new__(FW.ForwardModel)
    fm.ms, fm.settings, fm.gs, fm.F0 = ms, FW.ForwardSettings(vsc={"F0": 1.0}), None, 1.0
    with pytest.raises(ValueError, match="FITTED reference"):
        fm._setup_rows(np.zeros((10, 2)), None)
