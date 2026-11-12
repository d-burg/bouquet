"""The vendored checked-source Z_eff rule (:mod:`bouquet.kinrecon.zeff_sources`).

The same tests as the bouquet reader's own rule module (the module is a verbatim copy): every
qualification test passes and fails on either side of its threshold and is pinned exactly AT it
(inclusive or exclusive as the rule states); the three candidate states (qualified / unverified /
failed), the ranking, refusal and the explicit override, the > 2 sigma tension warning and the
record's contents, on synthetic profiles.  The module is also imported on its own to hold it
self-contained.  The kinetic-input glue (assemble_kinetic_inputs(..., zeff_source="checked")) is
tested in test_kinetics_zeff_checked.py.
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from bouquet.kinrecon import zeff_sources as Z
from bouquet.kinrecon.zeff_sources import (AUTOMATIC, FAIL, FAILED, HAND_FIT,
                                           IMPURITY, LINE, NOT_APPLICABLE,
                                           NOT_EVALUATED, PASS, QUALIFIED,
                                           UNVERIFIED, MeasurementPoints,
                                           ZeffSourceRefused, ZeffSourceWarning)

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bouquet", "kinrecon",
                      "zeff_sources.py")

PSI = np.round(np.linspace(0.0, 1.2, 121), 10)     # psi_N = 0.9 is a node
NE = 5e19 * (1 - 0.6 * (PSI / 1.2) ** 2) + 1e18
RHO_MAP = (PSI[PSI <= 1.0], np.sqrt(PSI[PSI <= 1.0]))   # a simple rho(psi)
NOFAST = np.zeros_like(NE)
#: a power of two: scaling by it is exact, so ratios built from it are too
P2 = 2.0 ** 60


def _nimp(fc0=0.01, slope=0.5):
    """n_imp with n_imp/n_e = fc0 (1 + slope psi_N)."""
    return fc0 * (1.0 + slope * PSI) * NE


def _pts(rho, value=1e17, sigma=None, **kw):
    rho = np.asarray(rho, dtype=float)
    v = np.full(rho.shape, value) if np.isscalar(value) else np.asarray(value)
    return MeasurementPoints(value=v, rho=rho, sigma=sigma, **kw)


def _covered(n=10, inner=0.05, outer=0.95):
    return np.linspace(inner, outer, n)


def _imp(n_imp=None, sigma=None, **kw):
    n_imp = _nimp() if n_imp is None else n_imp
    sigma = 0.1 * n_imp if sigma is None else sigma
    kw.setdefault("rho_map", RHO_MAP)
    return Z.impurity_tests(PSI, NE, n_imp, sigma, impurity_Z=6.0, **kw)


def _profile_points(rho, scale=1.0, rel_sigma=0.1, n_imp=None, **kw):
    """Points at ``rho`` that sit at ``scale`` x the profile."""
    n_imp = _nimp() if n_imp is None else n_imp
    psi = np.interp(rho, RHO_MAP[1], RHO_MAP[0])
    v = scale * np.interp(psi, PSI, n_imp)
    return _pts(rho, value=v, sigma=rel_sigma * v, **kw)


RAW_RHO = np.linspace(0.05, 0.85, 17)


def _raw(scale=1.0, rho=RAW_RHO, kind=AUTOMATIC, label="auto", t=1.0,
         rel_sigma=0.1):
    """A raw analysis at ``scale`` x the profile, timed at ``t``."""
    p = _profile_points(rho, scale, rel_sigma=rel_sigma, kind=kind,
                        label=label)
    p.time = np.full(np.size(p.value), t)
    return p


def _raw_imp(raw, **kw):
    return _imp(raw_points=raw, slice_time=1.0, **kw)


# ---------------------------------------------------------------------------
#  I-1 coverage
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_inner_point_on_either_side_of_the_line(self):
        ok = _imp(fit_points=_pts(_covered(inner=0.30 - 1e-6)))["tests"]["I-1"]
        bad = _imp(fit_points=_pts(_covered(inner=0.30 + 1e-6)))["tests"]["I-1"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL
        assert "innermost" in bad["detail"]

    def test_outer_point_on_either_side_of_the_line(self):
        ok = _imp(fit_points=_pts(_covered(outer=0.80 + 1e-6)))["tests"]["I-1"]
        bad = _imp(fit_points=_pts(_covered(outer=0.80 - 1e-6)))["tests"]["I-1"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL
        assert "outermost" in bad["detail"]

    def test_point_count_on_either_side_of_the_line(self):
        ok = _imp(fit_points=_pts(_covered(n=8)))["tests"]["I-1"]
        bad = _imp(fit_points=_pts(_covered(n=7)))["tests"]["I-1"]
        assert ok["verdict"] == PASS and ok["value"]["n_inside"] == 8
        assert bad["verdict"] == FAIL and bad["value"]["n_inside"] == 7

    def test_points_outside_the_boundary_do_not_count(self):
        rho = np.append(_covered(n=7), [1.05, 1.1])
        t = _imp(fit_points=_pts(rho))["tests"]["I-1"]
        assert t["verdict"] == FAIL and t["value"]["n_inside"] == 7

    def test_no_points_is_not_evaluated_and_leaves_the_source_unverified(self):
        c = _imp(n_fast=NOFAST)
        assert c["tests"]["I-1"]["verdict"] == NOT_EVALUATED
        assert c["state"] == UNVERIFIED and c["qualifies"] is False
        assert c["failed"] == []
        r = Z.zeff_source_record(c, None)
        assert any("I-1" in s and "no measurement points" in s
                   for s in r["not_evaluated"])
        assert any("not evaluated" in w for w in r["warnings"])

    def test_a_fit_that_used_no_point_fails(self):
        t = _imp(fit_points=_pts([]))["tests"]["I-1"]
        assert t["verdict"] == FAIL and t["value"]["n_inside"] == 0

    def test_fit_points_are_preferred_over_raw_points(self):
        t = _imp(fit_points=_pts(_covered(inner=0.7)),
                 raw_points=[_raw(rho=_covered())],
                 slice_time=1.0)["tests"]["I-1"]
        assert t["verdict"] == FAIL and t["points"].startswith("fit points")

    def test_raw_points_window_is_plus_minus_25_ms(self):
        rho = _covered()
        inside = _pts(rho, time=np.full(10, 1.0 + 0.0249), label="a",
                      kind=AUTOMATIC)
        outside = _pts(rho, time=np.full(10, 1.0 + 0.0251), label="a",
                       kind=AUTOMATIC)
        assert _imp(raw_points=[inside], slice_time=1.0)["tests"]["I-1"][
            "verdict"] == PASS
        t = _imp(raw_points=[outside], slice_time=1.0)["tests"]["I-1"]
        assert t["verdict"] == FAIL and t["value"]["n_inside"] == 0

    def test_the_files_own_window_wins_over_25_ms(self):
        p = _pts(_covered(), time=np.full(10, 1.02), label="a",
                 kind=AUTOMATIC)
        t = _imp(raw_points=[p], slice_time=1.0,
                 slice_window=(0.99, 1.01))["tests"]["I-1"]
        assert t["verdict"] == FAIL and "own window" in t["points"]

    def test_the_label_not_the_order_picks_the_reference(self):
        t1 = np.full(10, 1.0)
        auto = _pts(_covered(inner=0.7), time=t1, label="a", kind=AUTOMATIC)
        hand = _pts(_covered(), time=t1, label="h", kind=HAND_FIT)
        quick = _pts(_covered(), time=t1, label="q", kind="other")
        for order in ([auto, hand, quick], [hand, quick, auto],
                      [quick, auto, hand]):
            t = _imp(raw_points=order, slice_time=1.0)["tests"]["I-1"]
            assert t["reference_analysis"] == "h" and t["verdict"] == PASS
            assert set(t["all_analyses"]) == {"a", "h", "q"}
            assert t["all_analyses"]["a"]["verdict"] == FAIL
        # without a hand-fit analysis: the automatic one, wherever it sits
        for order in ([quick, auto], [auto, quick]):
            t = _imp(raw_points=order, slice_time=1.0)["tests"]["I-1"]
            assert t["reference_analysis"] == "a" and t["verdict"] == FAIL

    def test_unlabeled_or_ambiguous_analyses_are_refused(self):
        t1 = np.full(10, 1.0)
        quick = _pts(_covered(), time=t1, label="q")      # kind "other"
        with pytest.raises(ValueError, match="labeled"):
            _imp(raw_points=[quick], slice_time=1.0)
        a1 = _pts(_covered(), time=t1, label="a1", kind=AUTOMATIC)
        a2 = _pts(_covered(), time=t1, label="a2", kind=AUTOMATIC)
        with pytest.raises(ValueError, match="exactly one"):
            _imp(raw_points=[a1, a2], slice_time=1.0)
        with pytest.raises(ValueError, match="kind"):
            MeasurementPoints(value=np.ones(2), kind="hand")

    def test_psi_points_without_a_rho_map_are_not_evaluated(self):
        p = MeasurementPoints(value=np.full(10, 1e17),
                              psi_N=np.linspace(0.01, 0.95, 10))
        t = _imp(fit_points=p, rho_map=None)["tests"]["I-1"]
        assert t["verdict"] == NOT_EVALUATED and t["not_evaluated_parts"]


# ---------------------------------------------------------------------------
#  I-2 placeholder
# ---------------------------------------------------------------------------
class TestPlaceholder:
    @staticmethod
    def _spread_profile(s):
        # n_imp/n_e = c (1 + eps psi) over psi <= 0.9: relative ptp
        # = 0.9 eps / (1 + 0.45 eps) = s
        eps = s / (0.9 - 0.45 * s)
        return 0.01 * (1 + eps * PSI) * NE

    def test_relative_spread_on_either_side_of_the_line(self):
        ok = _imp(self._spread_profile(1.0e-3 * 1.01))["tests"]["I-2"]
        bad = _imp(self._spread_profile(1.0e-3 * 0.99))["tests"]["I-2"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL
        assert ok["value"]["rel_spread"] == pytest.approx(1.01e-3, rel=1e-6)
        assert "flat" in bad["detail"]

    def test_relative_sigma_on_either_side_of_the_line(self):
        n = _nimp()
        ok = _imp(n, sigma=1.0 * (1 - 1e-6) * n)["tests"]["I-2"]
        bad = _imp(n, sigma=1.0 * (1 + 1e-6) * n)["tests"]["I-2"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL

    def test_a_placeholder_slice_is_detected(self):
        """n_imp/n_e = 0.0100 radially flat with a several-hundred-percent
        sigma: the placeholder signature."""
        n = 0.01 * NE
        t = _imp(n, sigma=3.0 * n)["tests"]["I-2"]
        assert t["verdict"] == FAIL
        assert Z.is_placeholder_slice(PSI, NE, n)
        assert not Z.is_placeholder_slice(PSI, NE, _nimp())

    def test_no_sigma_is_a_part_not_evaluated(self):
        c = Z.impurity_tests(PSI, NE, _nimp(), None, rho_map=RHO_MAP)
        t = c["tests"]["I-2"]
        assert t["verdict"] == PASS and t["sigma_evaluated"] is False
        assert any(p.startswith("I-2: the relative-sigma half")
                   for p in c["not_evaluated"])
        assert c["state"] == UNVERIFIED

    def test_a_zero_sigma_counts_as_no_uncertainty(self):
        c = _imp(sigma=np.zeros_like(NE))
        t = c["tests"]["I-2"]
        assert t["sigma_evaluated"] is False
        assert t["value"]["median_rel_sigma"] is None
        assert t["value"]["n_nodes_without_sigma"] > 0
        assert any("relative-sigma" in p for p in c["not_evaluated"])


# ---------------------------------------------------------------------------
#  I-3 physical
# ---------------------------------------------------------------------------
class TestPhysical:
    @staticmethod
    def _with_min_fraction(fmin):
        # n_i/n_e = 1 - 6 n_imp/n_e reaches fmin on axis
        fc = np.full(PSI.shape, 0.05)
        fc[0] = (1.0 - fmin) / 6.0
        return fc * NE

    def test_main_ion_density_on_either_side_of_zero(self):
        ok = _imp(self._with_min_fraction(+1e-4))["tests"]["I-3"]
        bad = _imp(self._with_min_fraction(-1e-4))["tests"]["I-3"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL
        zero = _imp(self._with_min_fraction(0.0))["tests"]["I-3"]
        assert zero["verdict"] == FAIL      # n_i > 0 is required
        assert bad["cause"] == "dilution"

    def test_warning_below_a_tenth(self):
        c = _imp(self._with_min_fraction(0.1 - 1e-3))
        assert c["tests"]["I-3"]["verdict"] == PASS and c["tests"]["I-3"]["warn"]
        assert any("strongly diluted" in w for w in c["findings"])
        c = _imp(self._with_min_fraction(0.1 + 1e-3))
        assert not c["tests"]["I-3"]["warn"]

    def test_fast_ions_enter_quasineutrality(self):
        n = self._with_min_fraction(0.05)
        assert _imp(n)["tests"]["I-3"]["verdict"] == PASS
        t = _imp(n, n_fast=0.06 * NE)["tests"]["I-3"]
        assert t["verdict"] == FAIL and t["fast_ions_included"] is True

    def test_without_fast_ions_it_is_recorded_not_unverified(self):
        """No fast-ion density: I-3 runs without fast ions, recorded on the
        entry and in evaluated_without -- not a part not evaluated (the
        impurity-derived Z_eff does not depend on the fast-ion density)."""
        c = _qualified_imp(n_fast=None)
        t = c["tests"]["I-3"]
        assert t["verdict"] == PASS and not t["fast_ions_included"]
        assert t["evaluated_without_fast_ions"] is True
        assert t["fast_ions"] == Z.FAST_IONS_NOT_AVAILABLE
        assert not any(p.startswith("I-3") for p in c["not_evaluated"])
        assert c["evaluated_without"] == [
            "I-3: fast ions (no fast-ion density available: evaluated "
            "without fast ions)"]
        assert c["state"] == QUALIFIED
        r = Z.zeff_source_record(c, None)
        assert r["evaluated_without"] == [f"{IMPURITY} " +
                                          c["evaluated_without"][0]]
        assert r["not_evaluated"] == [] and r["warnings"] == []

    def test_a_declared_absence_of_fast_ions_is_recorded_as_such(self):
        c = _imp(n_fast="none")
        t = c["tests"]["I-3"]
        assert t["fast_ions"] == Z.FAST_IONS_DECLARED_NONE
        assert t["evaluated_without_fast_ions"] is False
        assert t["fast_ions_included"] is True and c["evaluated_without"] == []
        assert t["value"] == _imp(n_fast=NOFAST)["tests"]["I-3"]["value"]
        s = _imp(n_fast=NOFAST)["tests"]["I-3"]
        assert s["fast_ions"] == Z.FAST_IONS_SUPPLIED
        with pytest.raises(ValueError, match="'none'"):
            _imp(n_fast="zero")

    def test_the_dilution_warning_still_uses_what_is_known(self):
        n = self._with_min_fraction(0.12)
        assert not _imp(n)["tests"]["I-3"]["warn"]          # without
        t = _imp(n, n_fast=0.05 * NE)["tests"]["I-3"]        # with: 0.07
        assert t["verdict"] == PASS and t["warn"]

    def test_only_inside_the_boundary(self):
        n = _nimp()
        n[PSI > 1.0] = 0.5 * NE[PSI > 1.0]          # unphysical in the SOL
        assert _imp(n)["tests"]["I-3"]["verdict"] == PASS

    def test_invalid_data_outside_the_boundary_is_excluded_and_counted(self):
        n, s = _nimp(), 0.1 * _nimp()
        n[PSI > 1.07] = np.nan                       # SOL holes
        s[PSI > 1.07] = np.nan
        c = _imp(n, sigma=s, n_fast=NOFAST)
        t = c["tests"]["I-3"]
        assert t["verdict"] == PASS
        assert t["value"]["ni_over_ne_min"] is not None
        k = int(np.count_nonzero(PSI > 1.07))
        assert t["excluded"]["n_invalid_outside"] == k
        assert "excluded" in t["excluded"]["note"]
        n[PSI > 1.07] = -1e17                        # a SOL undershoot
        assert _imp(n)["tests"]["I-3"]["verdict"] == PASS

    def test_invalid_data_inside_the_boundary_fails_as_missing_data(self):
        n = _nimp()
        n[30] = np.nan                               # psi_N = 0.3
        t = _imp(n)["tests"]["I-3"]
        assert t["verdict"] == FAIL and t["cause"] == "invalid_data"
        assert "invalid inside the boundary" in t["detail"]
        assert "not a dilution" in t["detail"]
        n = _nimp()
        n[30] = 1e36                                  # a fill value
        assert _imp(n)["tests"]["I-3"]["cause"] == "invalid_data"

    def test_the_validity_screen_matches_the_reader_criteria(self):
        v = Z.validity_inside_boundary(PSI, NE, _nimp())
        assert v["ok"] and v["n_invalid_inside"] == 0
        samples = np.stack([_nimp(), _nimp()])
        samples[1, 10] = -1.0                         # one sample, one node
        v = Z.validity_inside_boundary(PSI, np.stack([NE, NE]), samples)
        assert not v["ok"] and v["n_invalid_inside"] == 1
        s = 0.1 * _nimp()
        s[5] = -1.0                                   # a negative sigma
        assert not Z.validity_inside_boundary(PSI, NE, _nimp(), s)["ok"]


# ---------------------------------------------------------------------------
#  I-4 consistency with supplied raw analyses
# ---------------------------------------------------------------------------
class TestConsistency:
    def _i4(self, pts):
        pts.time = np.full(np.size(pts.value), 1.0)
        pts.label, pts.kind = "auto", AUTOMATIC
        return _raw_imp([pts])["tests"]["I-4"]

    @pytest.mark.parametrize("scale, verdict", [
        (2.0 / 3.0 * (1 + 1e-6), PASS), (2.0 / 3.0 * (1 - 1e-6), FAIL),
        (1.5 * (1 - 1e-6), PASS), (1.5 * (1 + 1e-6), FAIL)])
    def test_ratio_on_either_side_of_its_bounds(self, scale, verdict):
        # a large point sigma keeps z small: the ratio alone decides
        t = self._i4(_profile_points(RAW_RHO, scale, rel_sigma=10.0))
        assert t["verdict"] == verdict

    @pytest.mark.parametrize("zmed, verdict", [(2.0 - 1e-6, PASS),
                                               (2.0 + 1e-6, FAIL)])
    def test_median_z_on_either_side_of_its_bound(self, zmed, verdict):
        n = _nimp()
        sp = 0.1 * n                                  # profile sigma
        psi = np.interp(RAW_RHO, RHO_MAP[1], RHO_MAP[0])
        prof = np.interp(psi, PSI, n)
        sprof = np.interp(psi, PSI, sp)
        s_pt = sprof.copy()                           # point sigma = profile's
        v = prof + zmed * np.hypot(s_pt, sprof)       # ratio 1 + 0.28: inside
        pts = MeasurementPoints(value=v, sigma=s_pt, rho=RAW_RHO,
                                time=np.full(v.size, 1.0), label="auto",
                                kind=AUTOMATIC)
        t = _imp(n, sigma=sp, raw_points=[pts], slice_time=1.0)["tests"]["I-4"]
        assert t["verdict"] == verdict
        assert t["value"]["all"]["median_z"] == pytest.approx(zmed, rel=1e-6)

    def test_an_extrapolated_core_fails_the_core_region_only(self):
        """Points agree outside rho 0.5 and sit at 0.4 x the profile inside:
        the overall median passes, the core region fails."""
        rho = np.linspace(0.3, 0.88, 30)         # 11 of 30 inside rho 0.5
        scale = np.where(rho < 0.5, 0.4, 1.0)
        pts = _profile_points(rho, 1.0, rel_sigma=10.0)
        pts.value = pts.value * scale
        t = self._i4(pts)
        assert t["regions"]["all"]["verdict"] == PASS
        assert t["regions"]["core"]["verdict"] == FAIL
        assert t["verdict"] == FAIL

    def test_a_region_without_points_is_a_part_not_evaluated(self):
        pts = _raw(rho=np.linspace(0.6, 0.85, 9))
        c = _raw_imp([pts], n_fast=NOFAST)
        t = c["tests"]["I-4"]
        assert t["regions"]["core"]["verdict"] == NOT_EVALUATED
        assert t["verdict"] == PASS
        assert any("core region has no points" in p for p in c["not_evaluated"])

    def test_no_point_sigma_leaves_the_z_half_not_evaluated(self):
        pts = _raw()
        pts.sigma = None
        c = _raw_imp([pts])
        assert c["tests"]["I-4"]["verdict"] == PASS
        assert any("I-4: the z half" in p for p in c["not_evaluated"])

    def test_no_raw_analyses_is_not_evaluated(self):
        c = _imp()
        assert c["tests"]["I-4"]["verdict"] == NOT_EVALUATED
        assert any(p == "I-4: no raw analyses supplied"
                   for p in c["not_evaluated"])

    def test_never_against_the_fit_points(self):
        """The file's own fit points serve I-1 only: points at 2 x the
        profile leave I-4 not evaluated, not failed."""
        c = _imp(fit_points=_profile_points(_covered(), 2.0))
        assert c["tests"]["I-1"]["verdict"] == PASS
        assert c["tests"]["I-4"]["verdict"] == NOT_EVALUATED
        assert c["state"] == UNVERIFIED

    def test_reference_by_label_and_every_analysis_recorded(self):
        good = _raw(1.0, kind=HAND_FIT, label="hand")
        bad = _raw(2.0, kind=AUTOMATIC, label="auto")
        t = _raw_imp([bad, good])["tests"]["I-4"]
        assert t["verdict"] == PASS and t["reference_analysis"] == "hand"
        assert t["reference_kind"] == HAND_FIT
        assert t["all_analyses"]["auto"]["all"]["verdict"] == FAIL


# ---------------------------------------------------------------------------
#  every threshold exactly AT its value (inclusive / exclusive as stated)
# ---------------------------------------------------------------------------
class TestAtTheThreshold:
    """Values constructed so that the statistic equals the threshold in
    floating point exactly (checked in each test)."""

    def test_coverage_inner_and_outer_are_inclusive(self):
        rho = np.array([0.30, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.80])
        t = _imp(fit_points=_pts(rho))["tests"]["I-1"]
        assert t["value"]["rho_inner"] == Z.COVERAGE_RHO_INNER_MAX
        assert t["value"]["rho_outer"] == Z.COVERAGE_RHO_OUTER_MIN
        assert t["value"]["n_inside"] == Z.COVERAGE_MIN_POINTS
        assert t["verdict"] == PASS

    def test_window_edge_is_inclusive(self):
        t0, t1 = 0.0, 0.025
        assert abs(t1 - t0) == Z.COVERAGE_WINDOW_S
        p = _pts(_covered(), time=np.full(10, t1), label="a", kind=AUTOMATIC)
        t = _imp(raw_points=[p], slice_time=t0)["tests"]["I-1"]
        assert t["value"]["n_inside"] == 10 and t["verdict"] == PASS

    @staticmethod
    def _flat_grid():
        psi = np.array([0.0, 0.3, 0.6, 0.9])
        return psi, (psi, np.sqrt(psi))

    def test_spread_is_inclusive(self):
        psi = np.array([0.0, 0.5, 0.9])
        ne = np.full(3, P2)
        n = np.array([1000.0, 1000.0, 1001.0]) * 2.0 ** 40
        c = Z.impurity_tests(psi, ne, n, 0.1 * n)
        assert c["tests"]["I-2"]["value"]["rel_spread"] == \
            Z.PLACEHOLDER_REL_SPREAD_MIN
        assert c["tests"]["I-2"]["verdict"] == PASS
        n2 = np.array([1000.0, 1000.0, np.nextafter(1001.0, 0)]) * 2.0 ** 40
        assert Z.impurity_tests(psi, ne, n2, 0.1 * n2)["tests"]["I-2"][
            "verdict"] == FAIL

    def test_placeholder_sigma_is_inclusive(self):
        n = _nimp()
        t = _imp(n, sigma=n.copy())["tests"]["I-2"]
        assert t["value"]["median_rel_sigma"] == Z.PLACEHOLDER_REL_SIGMA_MAX
        assert t["verdict"] == PASS

    def test_dilution_warning_is_exclusive(self):
        psi = np.array([0.0, 0.5, 1.0])
        ne = np.full(3, 6.0 * P2)
        n = np.full(3, 0.875 * P2)
        nf = 0.02499999999999999 * ne
        assert (1.0 - 6.0 * n / ne - nf / ne)[0] == Z.NI_FRACTION_WARN
        t = Z.impurity_tests(psi, ne, n, 0.1 * n, n_fast=nf)["tests"]["I-3"]
        assert t["value"]["ni_over_ne_min"] == Z.NI_FRACTION_WARN
        assert t["verdict"] == PASS and t["warn"] is False

    @pytest.mark.parametrize("ratio", [Z.POINTS_RATIO_MIN, Z.POINTS_RATIO_MAX])
    def test_ratio_bounds_are_inclusive(self, ratio):
        psi, rmap = self._flat_grid()
        prof = np.full(psi.shape, P2)
        rho = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
        v = np.full(rho.shape, ratio * P2)
        pts = MeasurementPoints(value=v, sigma=np.full(rho.shape, 1e30),
                                rho=rho, label="a", kind=AUTOMATIC)
        t = Z.impurity_tests(psi, np.full(psi.shape, 100 * P2), prof,
                             np.zeros_like(prof), rho_map=rmap,
                             raw_points=[pts])["tests"]["I-4"]
        assert t["value"]["all"]["median_ratio"] == ratio
        assert t["value"]["core"]["median_ratio"] == ratio
        assert t["verdict"] == PASS

    def test_median_z_bound_is_inclusive(self):
        psi, rmap = self._flat_grid()
        prof = np.full(psi.shape, P2)
        rho = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
        s = 2.0 ** 56
        v = np.full(rho.shape, P2 + 2.0 * s)           # z = 2 exactly
        pts = MeasurementPoints(value=v, sigma=np.full(rho.shape, s),
                                rho=rho, label="a", kind=AUTOMATIC)
        t = Z.impurity_tests(psi, np.full(psi.shape, 100 * P2), prof,
                             np.zeros_like(prof), rho_map=rmap,
                             raw_points=[pts])["tests"]["I-4"]
        assert t["value"]["all"]["median_z"] == Z.POINTS_ABS_MEDIAN_Z_MAX
        assert t["verdict"] == PASS
        pts.value = np.full(rho.shape, P2 + 2.0 * s + 2.0 ** 8)
        t = Z.impurity_tests(psi, np.full(psi.shape, 100 * P2), prof,
                             np.zeros_like(prof), rho_map=rmap,
                             raw_points=[pts])["tests"]["I-4"]
        assert t["verdict"] == FAIL

    def test_region_edges(self):
        """rho = 0.9 is inside the "all" region (<=), rho = 0.5 is outside
        the core region (<)."""
        psi, rmap = self._flat_grid()
        prof = np.full(psi.shape, P2)

        def run(rho, ratio):
            pts = MeasurementPoints(value=np.asarray(ratio) * P2,
                                    sigma=np.full(len(rho), 1e30),
                                    rho=np.asarray(rho), label="a",
                                    kind=AUTOMATIC)
            return Z.impurity_tests(psi, np.full(psi.shape, 100 * P2), prof,
                                    None, rho_map=rmap,
                                    raw_points=[pts])["tests"]["I-4"]
        t = run([0.2, 0.9, 0.9, 0.9], [1.0, 10.0, 10.0, 10.0])
        assert t["regions"]["all"]["verdict"] == FAIL     # 0.9 counted
        t = run([0.2, 0.5, 0.5, 0.7], [1.0, 10.0, 10.0, 1.0])
        assert t["regions"]["core"]["verdict"] == PASS    # 0.5 not counted
        assert t["regions"]["core"]["n"] == 1

    def test_line_sigma_is_inclusive(self):
        z = np.array([2.0, 2.5, 3.0])
        psi = np.array([0.0, 0.5, 0.9])
        t = Z.line_tests(psi, z, 0.25 * z)["tests"]["L-2"]
        assert t["value"] == Z.LINE_REL_SIGMA_MAX and t["verdict"] == PASS

    def test_copy_difference_is_exclusive(self):
        psi = np.array([0.0, 0.5, 0.9])
        zi = np.full(3, 1000.0)
        zl = np.full(3, 1001.0)
        t = Z.line_tests(psi, zl, 0.1 * zl, zeff_impurity=zi)["tests"]["L-3"]
        assert t["value"] == Z.COPY_REL_DIFF_MIN and t["verdict"] == FAIL

    def test_flatness_is_inclusive(self):
        psi = np.array([0.0, 0.5, 0.9])
        zl = np.array([0.0, 0.01, 0.02])
        t = Z.line_tests(psi, zl, np.full(3, 0.001))["tests"]["L-4"]
        assert t["value"] == Z.FLAT_PTP_MIN and t["verdict"] == PASS

    def test_advisory_is_exclusive(self):
        z = _zline()
        t = Z.line_tests(PSI, z, 0.1 * z, core_median_series=[10.5, 9.5])
        assert t["tests"]["advisory"]["value"] == Z.CORE_CV_ADVISORY
        assert t["tests"]["advisory"]["verdict"] == PASS

    def test_tension_is_exclusive(self):
        psi = np.array([0.0, 0.5, 0.9])
        ten = Z.tension(psi, np.full(3, 3.0), np.full(3, 1.0),
                        np.full(3, 1.0), np.zeros(3))
        assert ten["median_z"] == Z.TENSION_Z_MAX and ten["exceeds"] is False


# ---------------------------------------------------------------------------
#  L-1 ... L-4 and the advisory
# ---------------------------------------------------------------------------
def _zline(ptp=0.5):
    # 1.8 + ptp psi/0.9 over psi <= 0.9: peak-to-peak = ptp
    return 1.8 + ptp * PSI / 0.9


class TestLine:
    def test_no_uncertainty_fails_l1(self):
        c = Z.line_tests(PSI, _zline(), None)
        assert c["tests"]["L-1"]["verdict"] == FAIL
        assert c["tests"]["L-2"]["verdict"] == NOT_EVALUATED
        assert c["state"] == FAILED

    @pytest.mark.parametrize("sig", [0.0, -0.1, np.nan])
    def test_a_zero_or_invalid_uncertainty_fails_l1(self, sig):
        c = Z.line_tests(PSI, _zline(), np.full(PSI.shape, sig))
        assert c["tests"]["L-1"]["verdict"] == FAIL
        assert c["tests"]["L-1"]["value"]["n_nodes_with_sigma"] == 0
        assert c["state"] == FAILED

    def test_relative_sigma_on_either_side_of_the_line(self):
        z = _zline()
        ok = Z.line_tests(PSI, z, 0.25 * (1 - 1e-6) * z)["tests"]["L-2"]
        bad = Z.line_tests(PSI, z, 0.25 * (1 + 1e-6) * z)["tests"]["L-2"]
        assert ok["verdict"] == PASS and bad["verdict"] == FAIL

    def test_copy_detection_on_either_side_of_the_line(self):
        zi = _zline()
        for d, verdict in ((1e-3 * 1.01, PASS), (1e-3 * 0.99, FAIL)):
            zl = zi * (1 + d)
            t = Z.line_tests(PSI, zl, 0.1 * zl, zeff_impurity=zi)["tests"]["L-3"]
            assert t["verdict"] == verdict, d
        t = Z.line_tests(PSI, zi.copy(), 0.1 * zi, zeff_impurity=zi)
        assert t["tests"]["L-3"]["verdict"] == FAIL
        assert "copy" in t["tests"]["L-3"]["detail"]

    def test_copy_test_without_an_impurity_density_is_not_applicable(self):
        c = Z.line_tests(PSI, _zline(), 0.1 * _zline())
        assert c["tests"]["L-3"]["verdict"] == NOT_APPLICABLE
        assert c["state"] == QUALIFIED

    def test_copy_test_without_valid_impurity_values_is_not_evaluated(self):
        c = Z.line_tests(PSI, _zline(), 0.1 * _zline(),
                         zeff_impurity=np.full(PSI.shape, np.nan))
        assert c["tests"]["L-3"]["verdict"] == NOT_EVALUATED
        assert c["state"] == UNVERIFIED

    def test_flat_detection_on_either_side_of_the_line(self):
        ok = Z.line_tests(PSI, _zline(0.02 * 1.01), 0.1 * _zline())
        bad = Z.line_tests(PSI, _zline(0.02 * 0.99), 0.1 * _zline())
        assert ok["tests"]["L-4"]["verdict"] == PASS
        assert bad["tests"]["L-4"]["verdict"] == FAIL
        assert "flat" in bad["tests"]["L-4"]["detail"]
        # a default: exactly flat
        z = np.full(PSI.shape, 2.0)
        assert Z.line_tests(PSI, z, 0.1 * z)["tests"]["L-4"]["verdict"] == FAIL

    def test_the_advisory_warns_and_never_disqualifies(self):
        z = _zline()
        steady = Z.line_tests(PSI, z, 0.1 * z,
                              core_median_series=[2.0, 2.01, 1.99, 2.0])
        assert steady["tests"]["advisory"]["verdict"] == "warn"
        assert any("advisory" in w for w in steady["findings"])
        assert steady["state"] == QUALIFIED
        r = Z.zeff_source_record(None, steady)
        assert r["decision"]["source"] == LINE
        assert any("advisory" in w for w in r["warnings"])
        moving = Z.line_tests(PSI, z, 0.1 * z,
                              core_median_series=[1.5, 2.0, 2.5, 3.0])
        assert moving["tests"]["advisory"]["verdict"] == PASS
        single = Z.line_tests(PSI, z, 0.1 * z, core_median_series=[2.0])
        assert single["tests"]["advisory"]["verdict"] == NOT_EVALUATED
        assert single["state"] == QUALIFIED     # not a qualification test

    def test_advisory_series_leaves_out_placeholder_slices(self):
        z = [_zline(), _zline() + 1.0, _zline() + 2.0]
        nc = [_nimp(), 0.01 * NE, _nimp()]
        ser = Z.advisory_core_series(PSI, z, [NE] * 3, nc)
        assert len(ser) == 2
        assert ser == [Z.core_median(PSI, z[0]), Z.core_median(PSI, z[2])]
        assert len(Z.advisory_core_series(PSI, z, [NE] * 3, None)) == 3


# ---------------------------------------------------------------------------
#  states, ranking, tension, override, record
# ---------------------------------------------------------------------------
def _qualified_imp(n_fast=NOFAST):
    return _imp(fit_points=_profile_points(_covered()), raw_points=[_raw()],
                slice_time=1.0, n_fast=n_fast)


def _unverified_imp():
    return _imp()                          # no points, no fast ions


def _failed_imp():
    n = 0.01 * NE
    return _imp(n, sigma=3.0 * n)          # a placeholder


def _qualified_line():
    z = _zline()
    return Z.line_tests(PSI, z, 0.1 * z)


def _unverified_line():
    z = _zline()
    return Z.line_tests(PSI, z, 0.1 * z,
                        zeff_impurity=np.full(PSI.shape, np.nan))


def _failed_line():
    z = np.full(PSI.shape, 2.0)
    return Z.line_tests(PSI, z, 0.1 * z)   # flat


def _absent():
    return None


class TestDecision:
    def test_the_helpers_states(self):
        assert _qualified_imp()["state"] == QUALIFIED
        assert _unverified_imp()["state"] == UNVERIFIED
        assert _failed_imp()["state"] == FAILED
        assert _qualified_line()["state"] == QUALIFIED
        assert _unverified_line()["state"] == UNVERIFIED
        assert _failed_line()["state"] == FAILED

    @pytest.mark.parametrize("imp, line, source", [
        # a qualified impurity source first
        (_qualified_imp, _qualified_line, IMPURITY),
        (_qualified_imp, _unverified_line, IMPURITY),
        (_qualified_imp, _failed_line, IMPURITY),
        (_qualified_imp, _absent, IMPURITY),
        # then a qualified file value -- ahead of an unverified impurity
        (_unverified_imp, _qualified_line, LINE),
        (_failed_imp, _qualified_line, LINE),
        (_absent, _qualified_line, LINE),
        # then an unverified impurity source -- ahead of an unverified line
        (_unverified_imp, _unverified_line, IMPURITY),
        (_unverified_imp, _failed_line, IMPURITY),
        (_unverified_imp, _absent, IMPURITY),
        # else refused: an unverified file value is never used on its own
        (_failed_imp, _unverified_line, None),
        (_absent, _unverified_line, None),
        (_failed_imp, _failed_line, None),
        (_absent, _absent, None),
    ])
    def test_the_ranking(self, imp, line, source):
        i, li = imp(), line()
        d = Z.decide(i, li)
        assert d["source"] == source and d["rule_source"] == source
        assert d["refused"] is (source is None)
        si = "absent" if i is None else i["state"]
        sl = "absent" if li is None else li["state"]
        assert d["row"] == f"impurity {si}, line-integrated {sl}"
        assert d["states"] == {IMPURITY: si, LINE: sl}
        if source is not None:
            assert d["source_state"] == (si if source == IMPURITY else sl)

    def test_an_unverified_source_in_use_is_labeled_and_warned(self):
        r = Z.zeff_source_record(_unverified_imp(), _failed_line())
        assert r["decision"]["source_state"] == UNVERIFIED
        assert "unverified" in r["decision"]["reason"]
        assert r["warnings"][0].startswith("not evaluated: I-1")
        msg = Z.checked_warning(r)
        assert msg.startswith("Z_eff from the impurity source, UNVERIFIED")

    def test_a_qualified_source_is_not_warned(self):
        r = Z.zeff_source_record(_qualified_imp(), _failed_line())
        assert r["warnings"] == [] and Z.checked_warning(r) is None

    def test_the_not_evaluated_list_names_every_part(self):
        pts = _raw()
        pts.sigma = None
        c = Z.impurity_tests(PSI, NE, _nimp(), None, rho_map=RHO_MAP,
                             raw_points=[pts], slice_time=1.0)
        r = Z.zeff_source_record(c, _unverified_line())
        ne = r["not_evaluated"]
        for part in ("I-2: the relative-sigma half", "I-4: the z half"):
            assert any(s.startswith(f"{IMPURITY} {part}") for s in ne), part
        assert any(s.startswith(f"{LINE} L-3") for s in ne)
        # a missing fast-ion density is listed apart, not as unevaluated
        assert not any("fast ions" in s for s in ne)
        assert r["evaluated_without"] == [
            f"{IMPURITY} I-3: fast ions (no fast-ion density available: "
            "evaluated without fast ions)"]
        c = _imp()
        r = Z.zeff_source_record(c, None)
        assert any(s.startswith(f"{IMPURITY} I-1: coverage")
                   for s in r["not_evaluated"])
        assert any(s.startswith(f"{IMPURITY} I-4: no raw analyses")
                   for s in r["not_evaluated"])

    @pytest.mark.parametrize("zmed, warns", [(2.0 + 1e-6, True),
                                             (2.0 - 1e-6, False)])
    def test_tension_warning_above_two_sigma(self, zmed, warns):
        zl = _zline()
        sl = 0.05 * zl
        si = 0.05 * zl
        zi = zl + zmed * np.hypot(si, sl)
        ten = Z.tension(PSI, zi, si, zl, sl)
        assert ten["median_z"] == pytest.approx(zmed, rel=1e-6)
        assert ten["exceeds"] is warns
        d = Z.decide(_qualified_imp(), _qualified_line(), tension_rec=ten)
        assert d["source"] == IMPURITY            # never switched by tension
        assert bool(d["warnings"]) is warns
        # only in the both-qualified row
        d = Z.decide(_qualified_imp(), _unverified_line(), tension_rec=ten)
        assert d["warnings"] == []

    def test_override_is_honored_and_recorded(self):
        d = Z.decide(_failed_imp(), _failed_line(), override=LINE)
        assert d["source"] == LINE and d["rule_source"] is None
        assert d["override"]["named_failed_tests"] == ["L-4"]
        assert d["override"]["named_state"] == FAILED
        assert not d["override"]["named_qualifies"]
        assert any("override" in w for w in d["warnings"])
        d = Z.decide(_failed_imp(), _unverified_line(), override=LINE)
        assert d["source"] == LINE and d["source_state"] == UNVERIFIED
        assert any("unverified" in w for w in d["warnings"])
        with pytest.raises(ZeffSourceRefused):
            Z.decide(None, _qualified_line(), override=IMPURITY)
        with pytest.raises(ValueError):
            Z.decide(_qualified_imp(), _qualified_line(), override="VB")

    def test_an_override_cannot_make_a_failed_i3_usable(self):
        n = TestPhysical._with_min_fraction(-1e-3)
        c = _imp(n)
        assert c["tests"]["I-3"]["verdict"] == FAIL
        with pytest.raises(ZeffSourceRefused, match="cannot make"):
            Z.decide(c, _failed_line(), override=IMPURITY)
        n = _nimp()
        n[3] = np.nan
        with pytest.raises(ZeffSourceRefused, match="cannot make"):
            Z.decide(_imp(n), _failed_line(), override=IMPURITY)

    def test_an_unusable_impurity_density_says_why(self):
        u = Z.unusable_impurity("impurity_Z=5, but the density is carbon")
        d = Z.decide(u, _failed_line())
        assert d["source"] is None and "impurity_Z=5" in d["reason"]
        assert "no impurity density" not in d["reason"]
        with pytest.raises(ZeffSourceRefused, match="not usable"):
            Z.decide(u, _failed_line(), override=IMPURITY)

    def test_the_record(self):
        z = _zline()
        r = Z.zeff_source_record(_qualified_imp(), _qualified_line(),
                                 psi_N=PSI, zeff_imp=z,
                                 sigma_zeff_imp=0.05 * z, zeff_line=z,
                                 sigma_zeff_line=0.05 * z, time=1.25,
                                 crosschecks={"brightness_ratio": 0.8})
        assert r["schema_version"] == Z.SCHEMA_VERSION == 2
        assert r["thresholds"] == Z.THRESHOLDS
        assert r["thresholds"]["coverage_rho_inner_max"] == 0.30
        assert set(r["candidates"]) == {IMPURITY, LINE}
        for c in r["candidates"].values():
            assert c["state"] == QUALIFIED
            for t in c["tests"].values():
                assert {"verdict", "value", "threshold", "detail",
                        "not_evaluated_parts"} <= set(t)
        assert r["tension"]["median_z"] == pytest.approx(0.0)
        assert r["decision"]["source"] == IMPURITY and r["override"] is None
        assert r["crosschecks"] == {"brightness_ratio": 0.8}
        assert r["time"] == 1.25
        json.dumps(r)                       # archivable as JSON
        # cross-checks never select
        r2 = Z.zeff_source_record(_qualified_imp(), _qualified_line(),
                                  crosschecks={"favors": LINE})
        assert r2["decision"]["source"] == IMPURITY


# ---------------------------------------------------------------------------
#  the reader-independent glue
# ---------------------------------------------------------------------------
class TestGlue:
    def test_rho_map_failures_name_the_reason(self):
        q = 1.0 + PSI ** 2
        q[10] = np.nan
        with pytest.raises(ValueError, match="q is not finite"):
            Z.rho_map_from_q(PSI, q)
        with pytest.raises(ValueError, match="rise from 0"):
            Z.rho_map_from_q(PSI + 0.1, 1.0 + PSI ** 2)

    def test_slice_window_unit(self):
        a, b = np.array([990.0, 1090.0]), np.array([1010.0, 1110.0])
        assert Z.slice_window(a, b, 0, 1.0) == pytest.approx((0.99, 1.01))
        assert Z.slice_window(a / 1e3, b / 1e3, 1, 1.1) == \
            pytest.approx((1.09, 1.11))
        assert Z.slice_window(a, b, 0, 2.0) is None
        assert Z.slice_window(None, b, 0, 1.0) is None

    def test_fit_points_from_padded_arrays(self):
        p = Z.fit_points_from_arrays([1e17, np.nan, 9.97e36, 2e17],
                                     [1e16, 1e16, 1e16, np.nan],
                                     [0.1, 0.2, 0.3, 0.4])
        assert p.value.tolist() == [1e17, 2e17]
        assert p.psi_N.tolist() == [0.1, 0.4]
        assert np.isnan(p.sigma[1])


def test_the_rule_module_imports_without_bouquet(tmp_path):
    """The vendored module is numpy + standard library only: loaded by file path in a fresh
    interpreter (isolated mode, run from an empty directory) it imports no part of bouquet and runs."""
    path = os.path.abspath(MODULE)
    tree = ast.parse(open(path).read())
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            tops.add(node.module.split(".")[0])
    assert tops <= {"__future__", "dataclasses", "typing", "numpy"}, tops
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('zs_alone', {path!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['zs_alone'] = m\n"
        "spec.loader.exec_module(m)\n"
        "import numpy as np\n"
        "psi = np.linspace(0, 1.2, 61); ne = np.full(61, 5e19)\n"
        "c = m.impurity_tests(psi, ne, 0.01 * (1 + psi) * ne)\n"
        "r = m.zeff_source_record(c, None)\n"
        "assert r['decision']['source'] == m.IMPURITY\n"
        "bad = [k for k in sys.modules if k == 'bouquet' or k.startswith('bouquet.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
