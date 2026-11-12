"""The checked-source Z_eff rule (bouquet.io.zeff_sources; docs/zeff-source-rule.md).

Every qualification test passes and fails on either side of its threshold
and is pinned exactly AT it (inclusive or exclusive as the rule states);
the three candidate states (qualified / unverified / failed) and the
ranking, refusal and the explicit override, the > 2 sigma tension warning,
the series helper and the record's contents are checked on synthetic
profiles and synthetic IDA-layout files built here.  ``ni_source="standard"``
is checked bit for bit against stored values of the reader at the branch
point of this change (``tests/data/ida_standard_reference.npz``, written by
``tests/data/make_ida_standard_reference.py``), and also against that
reader taken from git when the history is available.  The rule module is
imported on its own, without ``bouquet``, to hold it self-contained.
"""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import _harness

h5py = pytest.importorskip("h5py")

from bouquet.io import zeff_sources as Z
from bouquet.io.ida import read_ida, zeff_rule_series
from bouquet.io.zeff_sources import (AUTOMATIC, FAIL, FAILED, HAND_FIT,
                                     IMPURITY, LINE, NOT_APPLICABLE,
                                     NOT_EVALUATED, PASS, QUALIFIED,
                                     UNVERIFIED, MeasurementPoints,
                                     ZeffSourceRefused, ZeffSourceWarning)

REPO = _harness.REPO_ROOT
DATA = os.path.join(REPO, "tests", "data")
#: the branch point of this change: "standard" must read as it did there
BASE_REF = "d311694"
FIXTURES = ["diiid_profs_synthetic.cdf", "diiid_profs_synthetic_pfast.cdf",
            "ida_synthetic_lmode.cdf"]

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
    def test_rho_map_matches_phi_n_from_q_bit_for_bit(self):
        from bouquet.coords import phi_n_from_q
        for psi in (PSI, np.linspace(0.0, 1.05, 37)):     # 1.0 hit or not
            q = 1.0 + 2.5 * psi ** 2 + 0.3 * psi ** 5
            inside, phi = phi_n_from_q(psi, q)
            x, rho = Z.rho_map_from_q(psi, q)
            np.testing.assert_array_equal(x, psi[inside])
            np.testing.assert_array_equal(rho,
                                          np.sqrt(np.clip(phi, 0.0, None)))

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
    """bouquet/io/zeff_sources.py is numpy + standard library only, so it
    can be vendored: loaded by file path in a fresh interpreter (isolated
    mode, no PYTHONPATH, run from an empty directory) it imports no part of
    bouquet and runs."""
    path = os.path.join(REPO, "bouquet", "io", "zeff_sources.py")
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
        "bad = [k for k in sys.modules if k == 'bouquet' or "
        "k.startswith('bouquet.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n")
    # -I (isolated mode) ignores PYTHONPATH and the script directory, so the
    # repository root the harness puts on PYTHONPATH is not importable here
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                         env=_harness.subprocess_env(), capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
#  read_ida: the "checked" route on synthetic IDA-layout files
# ---------------------------------------------------------------------------
def _write(path, *, n_time=1, zeff="good", zeff_err=0.08, carbon="good",
           carbon_err=0.1, placeholder_slices=(), points=None, window=None,
           q=True, sol_nan_from=None, carbon_nan_at=None, dilute_axis=False):
    """A direct-layout file.  ``zeff``: "good" (radially varying), "flat",
    "copy" (= the carbon-derived value) or None; ``carbon``: "good" or None;
    ``placeholder_slices``: slices whose carbon is the flat placeholder;
    ``points``: per-slice lists of (psi_N, value-scale) for the file's own
    fit-point datasets; ``window``: per-slice (start, end) [s]; ``q``: True,
    False, or an array; ``sol_nan_from``: n_12C6, n_12C6_err and Zeff are
    NaN at psi_N above this value (holes outside the boundary);
    ``carbon_nan_at``: a psi_N node where n_12C6 is NaN; ``dilute_axis``:
    n_12C6 = 0.2 n_e near the axis (n_i < 0 there)."""
    psi = PSI
    ne = NE
    nc = _nimp()
    if dilute_axis:
        nc = np.where(psi < 0.1, 0.2 * ne, nc)
    zc = 1.0 + 30.0 * nc / ne
    sol = (np.zeros(psi.shape, bool) if sol_nan_from is None
           else psi > sol_nan_from)
    with h5py.File(path, "w") as f:
        f["time"] = 1000.0 * (1.0 + 0.1 * np.arange(n_time))     # ms
        f["psi_n"] = psi
        te = 3000.0 * (1 - 0.9 * (psi / 1.2) ** 2) + 50.0

        def put(k, v):
            f[k] = np.stack([np.asarray(v(i) if callable(v) else v,
                                        dtype=float) for i in range(n_time)])
        put("n_e", ne)
        put("n_e_err", 0.05 * ne)
        put("T_e", te)
        put("T_e_err", 0.04 * te)
        put("T_12C6", 0.9 * te)
        put("T_12C6_err", 0.06 * te)
        if q is True:
            put("q", 1.0 + 2.5 * psi ** 2)
        elif q is not False:
            put("q", q)
        if carbon == "good":
            def ncs(i):
                a = (0.01 * ne if i in placeholder_slices else nc).copy()
                a[sol] = np.nan
                if carbon_nan_at is not None:
                    a[np.argmin(np.abs(psi - carbon_nan_at))] = np.nan
                return a
            put("n_12C6", ncs)
            if carbon_err is not None:
                put("n_12C6_err", lambda i: (3.0 if i in placeholder_slices
                                             else carbon_err) * ncs(i))
        if zeff is not None:
            zv = {"good": 1.5 + 0.6 * psi, "flat": np.full(psi.shape, 2.0),
                  "copy": zc}[zeff].copy()
            zv[sol] = np.nan
            put("Zeff", zv)
            if zeff_err is not None:
                put("Zeff_err", zeff_err * zv)
        if points is not None:
            npt = max(len(p) for p in points)
            raw = np.full((n_time, npt), np.nan)
            err = np.full((n_time, npt), np.nan)
            pn = np.full((n_time, npt), np.nan)
            for i, pl in enumerate(points):
                for j, (x, s) in enumerate(pl):
                    raw[i, j] = s * np.interp(x, psi, nc)
                    err[i, j] = 0.1 * raw[i, j]
                    pn[i, j] = x
            f["raw_n_12C6"], f["raw_n_12C6_err"] = raw, err
            f["psin_n_12C6"] = pn
        if window is not None:
            f["time_win_start"] = np.array([w[0] for w in window])
            f["time_win_end"] = np.array([w[1] for w in window])
    return str(path)


def _file_raw(scale=1.0, t=1.0, kind=AUTOMATIC, label="auto"):
    """A raw analysis of the synthetic file's carbon, on psi_N."""
    psi = np.linspace(0.01, 0.95, 12)
    v = scale * np.interp(psi, PSI, _nimp())
    return MeasurementPoints(value=v, sigma=0.1 * v, psi_N=psi,
                             time=np.full(psi.size, t), label=label,
                             kind=kind)


def _no_fast(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _quiet(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZeffSourceWarning)
        return read_ida(**kw)


def _rule_warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [x for x in w if issubclass(x.category, ZeffSourceWarning)]


class TestReader:
    def test_a_qualified_impurity_takes_the_cer_route_bit_for_bit(
            self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        r, ws = _rule_warnings(lambda: read_ida(
            p, ni_source="checked", zeff_raw_points=[_file_raw()],
            zeff_fast_density=_no_fast))
        cer = _quiet(path=p, ni_source="CER")
        assert r.zeff_rule["decision"]["source"] == IMPURITY
        assert r.zeff_rule["decision"]["row"] == \
            "impurity qualified, line-integrated qualified"
        for k in ("Zeff", "ni", "sigma_ni", "sigma_Zeff", "zeff_dne"):
            np.testing.assert_array_equal(getattr(r, k), getattr(cer, k), k)
        assert r.zeff_provenance["ni_source"] == "checked"
        assert r.zeff_provenance["source"] == "CER"
        assert r.zeff_provenance["zeff_rule"] is r.zeff_rule
        # the only finding here is the > 2 sigma tension of the two
        assert all("disagree" in str(w.message) for w in ws)

    def test_raw_points_without_a_fast_ion_density_qualify_the_carbon(
            self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        r = _quiet(path=p, ni_source="checked",
                   zeff_raw_points=[_file_raw()])
        rec = r.zeff_rule
        assert rec["candidates"][IMPURITY]["state"] == QUALIFIED
        assert rec["decision"]["source"] == IMPURITY
        assert rec["evaluated_without"] and rec["not_evaluated"] == []
        t3 = rec["candidates"][IMPURITY]["tests"]["I-3"]
        assert t3["evaluated_without_fast_ions"] is True
        d = _quiet(path=p, ni_source="checked",
                   zeff_raw_points=[_file_raw()], zeff_fast_density="none")
        assert d.zeff_rule["candidates"][IMPURITY]["tests"]["I-3"][
            "fast_ions"] == Z.FAST_IONS_DECLARED_NONE
        assert d.zeff_rule["evaluated_without"] == []
        with pytest.raises(ValueError, match="psi_N grid"):
            _quiet(path=p, ni_source="checked",
                   zeff_fast_density=np.zeros(5))

    def test_an_unverified_impurity_yields_to_a_qualified_file_value(
            self, tmp_path):
        """No points: the impurity source is unverified (coverage and raw
        consistency), the file value qualified -- the file value is used."""
        p = _write(tmp_path / "a.cdf")
        r, ws = _rule_warnings(lambda: read_ida(p, ni_source="checked"))
        vb = _quiet(path=p, ni_source="Zeff")
        rec = r.zeff_rule
        assert rec["candidates"][IMPURITY]["state"] == UNVERIFIED
        assert rec["candidates"][LINE]["state"] == QUALIFIED
        assert rec["decision"]["source"] == LINE
        assert ws == []
        for k in ("Zeff", "ni", "sigma_ni", "sigma_Zeff"):
            np.testing.assert_array_equal(getattr(r, k), getattr(vb, k), k)

    def test_line_qualifies_alone_takes_the_zeff_route_bit_for_bit(
            self, tmp_path):
        p = _write(tmp_path / "a.cdf", placeholder_slices=(0,))
        r = _quiet(path=p, ni_source="checked")
        vb = _quiet(path=p, ni_source="Zeff")
        assert r.zeff_rule["decision"]["source"] == LINE
        assert "I-2" in r.zeff_rule["candidates"][IMPURITY]["failed"]
        for k in ("Zeff", "ni", "sigma_ni", "sigma_Zeff"):
            np.testing.assert_array_equal(getattr(r, k), getattr(vb, k), k)
        assert r.zeff_provenance["source"] == "VB"

    def test_a_copy_and_an_unverified_carbon_take_the_carbon_flagged(
            self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="copy")
        r, ws = _rule_warnings(lambda: read_ida(p, ni_source="checked"))
        rec = r.zeff_rule
        assert rec["candidates"][LINE]["failed"] == ["L-3"]
        assert rec["decision"]["source"] == IMPURITY
        assert rec["decision"]["source_state"] == UNVERIFIED
        assert len(ws) == 1
        msg = str(ws[0].message)
        assert "UNVERIFIED" in msg and "I-1" in msg and "I-4" in msg
        assert len(msg) < 400                   # short and specific

    def test_neither_is_refused_naming_the_opt_out(self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="flat", placeholder_slices=(0,))
        with pytest.raises(ZeffSourceRefused,
                           match="ni_source='standard'") as e:
            _quiet(path=p, ni_source="checked")
        rec = e.value.record
        assert rec["decision"]["refused"] is True
        assert "I-2" in rec["candidates"][IMPURITY]["failed"]
        assert "L-4" in rec["candidates"][LINE]["failed"]

    def test_invalid_only_outside_the_boundary_is_not_refused(self, tmp_path):
        """A converted file with NaN carbon and Z_eff past psi_N 1.07 and a
        file Z_eff that copies the carbon value: the forced CER route is
        unchanged (it refuses: the whole-grid screen drops the envelope),
        the checked route tests inside the boundary and takes the carbon
        with its envelope screened inside the boundary."""
        p = _write(tmp_path / "a.cdf", zeff="copy", zeff_err=None,
                   sol_nan_from=1.07)
        with pytest.raises(KeyError, match="ni_source='CER'"):
            _quiet(path=p, ni_source="CER")
        r = _quiet(path=p, ni_source="checked")
        std = read_ida(p)
        rec = r.zeff_rule
        t3 = rec["candidates"][IMPURITY]["tests"]["I-3"]
        assert t3["verdict"] == PASS and t3["value"]["ni_over_ne_min"] > 0
        k = int(np.count_nonzero(PSI > 1.07))
        assert t3["excluded"]["n_invalid_outside"] == k
        assert rec["validity_screen"]["ok"] is True
        assert rec["decision"]["source"] == IMPURITY
        assert "inside the boundary" in r.sigma_Zeff_carbon_source
        assert "carbon_envelope_screen" in r.zeff_provenance
        m = PSI <= 1.0
        np.testing.assert_allclose(r.Zeff[m], std.Zeff[m], rtol=1e-12)
        assert np.all(np.isnan(r.Zeff[PSI > 1.07]))

    def test_invalid_inside_the_boundary_is_refused_as_missing_data(
            self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="copy", zeff_err=None,
                   carbon_nan_at=0.5)
        with pytest.raises(ZeffSourceRefused,
                           match="invalid inside the boundary") as e:
            _quiet(path=p, ni_source="checked")
        assert "dilution result" in str(e.value)
        t3 = e.value.record["candidates"][IMPURITY]["tests"]["I-3"]
        assert t3["cause"] == "invalid_data"

    def test_explicit_override_reads_and_records(self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="flat", placeholder_slices=(0,))
        with pytest.warns(ZeffSourceWarning, match="override"):
            r = read_ida(p, ni_source="checked", zeff_override="Zeff")
        ov = r.zeff_rule["override"]
        assert ov["source"] == LINE and ov["rule_source"] is None
        assert ov["named_failed_tests"] == ["L-4"]
        assert r.zeff_provenance["zeff_override"] == "Zeff"
        assert r.zeff_provenance["source"] == "VB"
        r = _quiet(path=p, ni_source="checked", zeff_override="CER")
        assert r.zeff_provenance["source"] == "CER"

    def test_an_override_onto_unphysical_carbon_is_refused(self, tmp_path):
        """n_i < 0 near the axis: the checked route refuses an override
        onto the carbon; the pre-existing forced CER route still clamps
        there (unchanged), now recorded and warned."""
        p = _write(tmp_path / "a.cdf", dilute_axis=True)
        r = _quiet(path=p, ni_source="checked")
        assert r.zeff_rule["decision"]["source"] == LINE
        with pytest.raises(ZeffSourceRefused, match="cannot make"):
            _quiet(path=p, ni_source="checked", zeff_override="CER")
        cer, ws = _rule_warnings(lambda: read_ida(p, ni_source="CER"))
        assert len(ws) == 1 and "clamps" in str(ws[0].message)
        assert cer.zeff_rule["forced"]["n_clipped"]["CER"] > 0
        assert cer.zeff_provenance["n_clipped_cer"] == \
            cer.zeff_rule["forced"]["n_clipped"]["CER"]

    def test_override_only_with_checked(self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        with pytest.raises(ValueError, match="checked"):
            read_ida(p, ni_source="CER", zeff_override="CER")
        with pytest.raises(ValueError, match="'CER'"):
            read_ida(p, ni_source="checked", zeff_override="VB")

    def test_no_carbon_and_no_zeff_sigma_is_refused(self, tmp_path):
        p = _write(tmp_path / "a.cdf", carbon=None, zeff_err=None)
        with pytest.raises(ZeffSourceRefused) as e:
            _quiet(path=p, ni_source="checked")
        rec = e.value.record
        assert rec["candidates"][IMPURITY] == {"present": False,
                                               "state": "absent"}
        assert rec["candidates"][LINE]["failed"] == ["L-1"]
        assert "no impurity density in the file" in rec["decision"]["reason"]
        # overridden: the file's Z_eff with the ne-fraction sigma
        r = _quiet(path=p, ni_source="checked", zeff_override="Zeff")
        assert r.sigma_Zeff_source == "VB (ne-fraction sigma)"
        with pytest.raises(ZeffSourceRefused, match="does not carry"):
            _quiet(path=p, ni_source="checked", zeff_override="CER")

    def test_no_carbon_and_a_qualified_zeff_takes_the_zeff(self, tmp_path):
        p = _write(tmp_path / "a.cdf", carbon=None)
        r = _quiet(path=p, ni_source="checked")
        assert r.zeff_rule["candidates"][LINE]["tests"]["L-3"]["verdict"] \
            == NOT_APPLICABLE
        assert r.zeff_rule["decision"]["source"] == LINE

    def test_another_impurity_charge_says_why(self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        r = _quiet(path=p, ni_source="checked", impurity_Z=5.0)
        c = r.zeff_rule["candidates"][IMPURITY]
        assert c["present"] and not c["usable"] and c["state"] == FAILED
        assert "impurity_Z=5" in c["reason"]
        assert "impurity_Z=5" in r.zeff_rule["decision"]["reason"]
        assert "no impurity density" not in r.zeff_rule["decision"]["reason"]
        assert r.zeff_rule["decision"]["source"] == LINE

    def test_a_failed_rho_map_names_its_reason(self, tmp_path):
        q = 1.0 + 2.5 * PSI ** 2
        q[20] = np.nan
        p = _write(tmp_path / "a.cdf", q=q)
        r = _quiet(path=p, ni_source="checked")
        assert "could not be built" in r.zeff_rule["rho_map"]
        assert "q is not finite" in r.zeff_rule["rho_map"]
        p = _write(tmp_path / "b.cdf", q=False)
        r = _quiet(path=p, ni_source="checked")
        assert r.zeff_rule["rho_map"] == "none (the file carries no q)"

    def test_the_files_fit_points_and_window_are_read(self, tmp_path):
        core = [(x, 1.0) for x in np.linspace(0.01, 0.95, 12)]
        edge = [(x, 1.0) for x in np.linspace(0.7, 0.95, 12)]
        p = _write(tmp_path / "a.cdf", n_time=2, points=[core, edge],
                   window=[(0.99, 1.01), (1.09, 1.11)])
        r0 = _quiet(path=p, time=1.0, ni_source="checked")
        r1 = _quiet(path=p, time=1.1, ni_source="checked")
        t0 = r0.zeff_rule["candidates"][IMPURITY]["tests"]["I-1"]
        t1 = r1.zeff_rule["candidates"][IMPURITY]["tests"]["I-1"]
        assert t0["verdict"] == PASS and t1["verdict"] == FAIL
        # fit points serve I-1 only: I-4 stays not evaluated without raw
        assert r0.zeff_rule["candidates"][IMPURITY]["tests"]["I-4"][
            "verdict"] == NOT_EVALUATED
        assert r0.zeff_rule["rho_map"].startswith("rho_tor from the file's q")
        # raw points with times use the file's own window (in seconds here)
        rr = _quiet(path=p, time=1.0, ni_source="checked",
                    zeff_raw_points=[_file_raw(t=1.015)])
        t4 = rr.zeff_rule["candidates"][IMPURITY]["tests"]["I-4"]
        assert "own window" in t4["all_analyses"]["auto"]["window"]
        assert t4["regions"]["all"]["n"] == 0       # 1.015 s is outside it

    def test_fast_ion_density_reaches_i3(self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        r = _quiet(path=p, ni_source="checked",
                   zeff_fast_density=lambda x: 0.95 * np.interp(x, PSI, NE))
        t = r.zeff_rule["candidates"][IMPURITY]["tests"]["I-3"]
        assert t["verdict"] == FAIL and t["fast_ions_included"]
        assert r.zeff_rule["decision"]["source"] == LINE

    def test_the_record_is_json_and_carries_the_thresholds(self, tmp_path):
        p = _write(tmp_path / "a.cdf")
        r = _quiet(path=p, ni_source="checked",
                   zeff_crosschecks={"radiated_power": "not run"})
        json.dumps(r.zeff_provenance)
        assert r.zeff_rule["thresholds"] == Z.THRESHOLDS
        assert r.zeff_rule["schema_version"] == 2
        assert r.zeff_rule["ni_source"] == "checked"
        assert r.zeff_rule["tension"] is not None
        assert r.zeff_rule["crosschecks"] == {"radiated_power": "not run"}
        assert r.zeff_rule["route_taken"]["zeff_source"] == "VB"

    def test_a_reused_record_is_not_evaluated_or_warned_again(self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="copy")
        first = _quiet(path=p, ni_source="checked")
        again, ws = _rule_warnings(lambda: read_ida(
            p, ni_source="checked", zeff_rule_from=first.zeff_rule))
        assert ws == []
        for k in ("Zeff", "ni", "sigma_ni", "sigma_Zeff", "sigma_ne"):
            np.testing.assert_array_equal(getattr(again, k),
                                          getattr(first, k), k)
        rec = dict(again.zeff_rule)
        assert "reused" in rec
        rec.pop("reused")
        assert rec == first.zeff_rule
        with pytest.raises(ValueError, match="same ni_source"):
            read_ida(p, ni_source="CER", zeff_rule_from=first.zeff_rule)

    def test_a_separate_sigma_file_takes_the_profile_files_decision(
            self, tmp_path):
        """The profile file decides the carbon (its Z_eff is a copy); a
        sigma file whose own rule would take its Z_eff reads the carbon."""
        prof = _write(tmp_path / "prof.cdf", zeff="copy")
        sig = _write(tmp_path / "sig.cdf")
        rec = _quiet(path=prof, ni_source="checked").zeff_rule
        assert rec["decision"]["source"] == IMPURITY
        assert _quiet(path=sig, ni_source="checked").zeff_rule[
            "decision"]["source"] == LINE
        s = _quiet(path=sig, ni_source="checked", zeff_rule_from=rec)
        assert s.zeff_provenance["source"] == "CER"


class TestForcedRoutes:
    """"CER" / "Zeff" / "all" keep their behavior; the rule is recorded,
    and only a source the route takes that FAILS a test is warned about,
    once per read."""

    @pytest.mark.parametrize("route, cands", [
        ("CER", [IMPURITY]), ("Zeff", [LINE]), ("all", [IMPURITY, LINE])])
    def test_recorded_beside_the_route_without_a_warning(self, tmp_path,
                                                         route, cands):
        p = _write(tmp_path / "a.cdf")
        r, ws = _rule_warnings(lambda: read_ida(p, ni_source=route))
        assert r.zeff_rule["forced"]["ni_source"] == route
        assert r.zeff_rule["forced"]["candidates"] == cands
        assert r.zeff_provenance["ni_source"] == route
        assert "zeff_rule" not in r.zeff_provenance
        # the impurity source is unverified here: recorded, not warned
        assert r.zeff_rule["candidates"][IMPURITY]["state"] == UNVERIFIED
        assert ws == []

    def test_a_failing_forced_source_warns_once_not_refuses(self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff="flat")
        r, ws = _rule_warnings(lambda: read_ida(p, ni_source="Zeff"))
        assert len(ws) == 1 and "fails L-4" in str(ws[0].message)
        assert r.zeff_provenance["source"] == "VB"
        assert r.zeff_rule["forced_warnings"]

    def test_all_records_a_tension_without_warning(self, tmp_path):
        p = _write(tmp_path / "a.cdf", zeff_err=0.01, carbon_err=0.01)
        r, ws = _rule_warnings(lambda: read_ida(p, ni_source="all"))
        assert r.zeff_rule["tension"]["exceeds"] is True
        assert ws == []

    def test_the_forced_routes_output_is_unchanged_by_the_rule(self,
                                                               tmp_path):
        """The route's arrays do not depend on anything the rule receives
        (points, fast ions)."""
        p = _write(tmp_path / "a.cdf", zeff="flat")
        a = _quiet(path=p, ni_source="all")
        b = _quiet(path=p, ni_source="all", zeff_raw_points=[_file_raw(3.0)],
                   zeff_fast_density=_no_fast)
        for k in ("Zeff", "ni", "sigma_ni", "sigma_Zeff", "zeff_dne"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k), k)


class TestSeries:
    def test_an_unnamed_series_is_refused_listing_the_slices(self, tmp_path):
        p = _write(tmp_path / "s.cdf", n_time=2)
        with pytest.raises(ZeffSourceRefused, match="name it") as e:
            zeff_rule_series(p, [1.0, 1.1])          # even when they agree
        assert "1.0: rule -> line_integrated" in str(e.value)
        assert "independent per-slice reads" in str(e.value)

    def test_one_named_source_failing_slices_flagged(self, tmp_path):
        p = _write(tmp_path / "s.cdf", n_time=3, placeholder_slices=(1,))
        times = [1.0, 1.1, 1.2]
        profs, ser = zeff_rule_series(p, times, source="CER")
        assert ser["source"] == IMPURITY and ser["named"] is True
        assert ser["failing_slices"] == [pytest.approx(1.1)]
        assert ser["unverified_slices"] == [pytest.approx(1.0),
                                            pytest.approx(1.2)]
        assert [pr.zeff_provenance["source"] for pr in profs] == ["CER"] * 3
        assert all(pr.zeff_rule["override"]["source"] == IMPURITY
                   for pr in profs)
        assert profs[1].zeff_rule["override"]["named_state"] == FAILED

    def test_series_source_helper(self):
        recs = [Z.zeff_source_record(_qualified_imp(), _qualified_line(),
                                     time=1.0),
                Z.zeff_source_record(_failed_imp(), _qualified_line(),
                                     time=1.1)]
        with pytest.raises(ZeffSourceRefused, match="name it"):
            Z.series_source(recs)
        s = Z.series_source(recs, source=LINE)
        assert s["failing_slices"] == [] and s["source"] == LINE
        s = Z.series_source(recs, source=IMPURITY)
        assert s["failing_slices"] == [1.1]


# ---------------------------------------------------------------------------
#  configuration
# ---------------------------------------------------------------------------
def test_the_config_takes_checked_and_validates_the_override():
    from bouquet.config import ImasSource, ReconstructionSource
    s = ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf",
                             ni_source="checked", zeff_override="CER")
    assert s.zeff_override == "CER"
    with pytest.raises(ValueError, match="checked"):
        ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf",
                             ni_source="CER", zeff_override="CER")
    with pytest.raises(ValueError, match="zeff_override"):
        ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf",
                             ni_source="checked", zeff_override="VB")
    ImasSource(ids_path="x", ni_source="checked")
    from bouquet.experimental import IDA_ION_ROUTES, NI_SOURCES
    assert "checked" in IDA_ION_ROUTES and "checked" in NI_SOURCES


class TestFastIonConfig:
    """ReconstructionSource.zeff_fast_density: the g-file route's fast-ion
    input to the rule (validated, stored, round-tripped, read)."""

    @staticmethod
    def _src(v, **kw):
        from bouquet.config import ReconstructionSource
        kw.setdefault("profiles_path", "p.cdf")
        kw.setdefault("ni_source", "checked")
        return ReconstructionSource(geqdsk_path="g", zeff_fast_density=v,
                                    **kw)

    def test_accepted_forms(self):
        assert self._src(None).zeff_fast_density is None
        assert self._src("none").zeff_fast_density == "none"
        a = self._src([0.0, 1e18, 2e18]).zeff_fast_density
        assert isinstance(a, np.ndarray) and a.tolist() == [0.0, 1e18, 2e18]
        d = self._src({"psi_N": [0, 0.5, 1], "n_fast": [2e18, 1e18, 0]})
        assert d.zeff_fast_density["n_fast"].tolist() == [2e18, 1e18, 0.0]
        assert self._src("none", ni_source="CER").zeff_fast_density == "none"

    @pytest.mark.parametrize("v, kw, match", [
        ("zero", {}, "'none'"),
        (lambda x: x, {}, "callable"),
        ([1e18, -1.0], {}, "< 0"),
        ([1e18, np.nan], {}, "finite"),
        ({"psi_N": [0, 1]}, {}, "keys"),
        ({"psi_N": [1, 0], "n_fast": [0, 0]}, {}, "rise"),
        ("none", {"ni_source": "standard"}, "standard"),
        ("none", {"profiles_path": "p.peqdsk"}, ".cdf"),
    ])
    def test_refused_forms(self, v, kw, match):
        with pytest.raises(ValueError, match=match):
            self._src(v, **kw)

    def test_round_trips_through_the_config_dict(self):
        from bouquet.config import _decode, _decode_source, _encode_source
        src = self._src({"psi_N": [0, 0.5, 1], "n_fast": [2e18, 1e18, 0]})
        back = _decode_source(_decode(json.loads(json.dumps(
            _encode_source(src)))))
        np.testing.assert_array_equal(back.zeff_fast_density["n_fast"],
                                      src.zeff_fast_density["n_fast"])
        src = self._src("none")
        assert _decode_source(_decode(json.loads(json.dumps(
            _encode_source(src))))).zeff_fast_density == "none"

    def test_the_kinetics_loader_passes_it_to_the_rule(self, tmp_path):
        from bouquet.baseline import _load_kinetic_profiles
        p = _write(tmp_path / "a.cdf", zeff="copy")
        rec = lambda v: _load_kinetic_profiles(self._src(
            v, profiles_path=p))["zeff_provenance"]["zeff_rule"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ZeffSourceWarning)
            t_none = rec(None)["candidates"][IMPURITY]["tests"]["I-3"]
            t_decl = rec("none")["candidates"][IMPURITY]["tests"]["I-3"]
            big = {"psi_N": PSI, "n_fast": 0.95 * NE}
            # n_i < 0 with these fast ions, and the file Z_eff is a copy:
            # refused, the record says why
            with pytest.raises(ZeffSourceRefused) as e:
                rec(big)
        t_big = e.value.record["candidates"][IMPURITY]["tests"]["I-3"]
        assert t_none["fast_ions"] == Z.FAST_IONS_NOT_AVAILABLE
        assert t_decl["fast_ions"] == Z.FAST_IONS_DECLARED_NONE
        assert t_big["fast_ions"] == Z.FAST_IONS_SUPPLIED
        assert t_big["verdict"] == FAIL


# ---------------------------------------------------------------------------
#  "standard" is unchanged, bit for bit, on the public fixtures
# ---------------------------------------------------------------------------
def _same(a, b, field):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=field)
        assert np.asarray(a).dtype == np.asarray(b).dtype, field
    elif isinstance(a, dict):
        assert set(a) == set(b), field
        for k in a:
            _same(a[k], b[k], f"{field}[{k}]")
    else:
        assert a == b, (field, a, b)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_standard_matches_the_stored_branch_point_values(fixture):
    """Against tests/data/ida_standard_reference.npz (no git needed): every
    field of what the branch-point reader returned, arrays bit for bit with
    their dtype, raw_bytes by SHA-256, the rest through JSON."""
    ref = np.load(os.path.join(DATA, "ida_standard_reference.npz"))
    assert str(ref["__base__"]) == BASE_REF
    meta = json.loads(str(ref["__meta__"]))[fixture]
    path = os.path.join(DATA, fixture)
    for new in (read_ida(path), read_ida(path, ni_source="standard")):
        names = {f.name for f in dataclasses.fields(new)}
        assert set(meta) == names - {"zeff_rule"}
        assert new.zeff_rule is None
        assert "zeff_rule" not in new.zeff_provenance
        for name, m in meta.items():
            v = getattr(new, name)
            if m["kind"] == "array":
                _same(v, ref[f"{fixture}/{name}"], name)
            elif m["kind"] == "bytes_sha256":
                assert hashlib.sha256(v).hexdigest() == m["value"], name
            elif m["kind"] == "none":
                assert v is None, name
            else:
                assert type(v).__name__ == m["kind"], name
                assert json.loads(json.dumps(v)) == m["value"], name


def _load_base_reader():
    try:
        src = subprocess.run(
            ["git", "-C", REPO, "show", f"{BASE_REF}:bouquet/io/ida.py"],
            check=True, capture_output=True, text=True,
            env=_harness.subprocess_env()).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASE_REF} unavailable (the stored-values "
                    "test above covers this)")
    name = "bouquet.io._ida_base_reference"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "bouquet.io"
    sys.modules[name] = mod
    exec(compile(src, f"{BASE_REF}:bouquet/io/ida.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("fixture", FIXTURES)
def test_standard_is_bit_identical_to_the_branch_point(fixture):
    path = os.path.join(DATA, fixture)
    ref = _load_base_reader().read_ida(path)
    new = read_ida(path)
    explicit = read_ida(path, ni_source="standard")
    for fld in dataclasses.fields(ref):
        _same(getattr(new, fld.name), getattr(ref, fld.name), fld.name)
        _same(getattr(explicit, fld.name), getattr(ref, fld.name), fld.name)
    assert new.zeff_rule is None and "zeff_rule" not in new.zeff_provenance


@pytest.mark.parametrize("fixture", FIXTURES)
def test_what_the_rule_decides_on_the_public_fixtures(fixture):
    """The fixtures' Z_eff is the carbon-derived value (a copy, L-3 fails)
    and they carry no measurement points (I-1, I-4 not evaluated; I-3 runs
    without fast ions, recorded apart): the rule takes the carbon density,
    flagged unverified, whose Z_eff equals the stored one to rounding."""
    path = os.path.join(DATA, fixture)
    r = _quiet(path=path, ni_source="checked")
    rec = r.zeff_rule
    assert rec["decision"]["source"] == IMPURITY
    assert rec["decision"]["source_state"] == UNVERIFIED
    assert rec["candidates"][LINE]["failed"] == ["L-3"]
    tests = rec["candidates"][IMPURITY]["tests"]
    assert tests["I-1"]["verdict"] == NOT_EVALUATED
    assert tests["I-4"]["verdict"] == NOT_EVALUATED
    assert tests["I-3"]["evaluated_without_fast_ions"] is True
    assert not any("I-3" in s for s in rec["not_evaluated"])
    std = read_ida(path)
    np.testing.assert_allclose(r.Zeff, std.Zeff, rtol=1e-12)
