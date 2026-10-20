"""The per-draw axis record, its displacement flag and the per-scan
bimodality stamp of the engine draws -- fast, on the toy stand-in solver of
``tests/_engine_fake_gs.py``.

Every archived engine draw records its archived state's magnetic axis
against the reconstruction's delivered axis (``engine["axis"]``) and
``axis_shift_large`` = whether the displacement exceeds
``engine.AXIS_BRANCH_FRAC`` x a -- a displacement, not a branch label.  The
flag and the displacement are stamped on the draw group and counted in the
summary line, in ``ScanView.spread()``, in the ``draw_scalars`` provenance
and on the scan (``n_axis_shift_large``); the sigma=0 draw carries the
record too.  Whether the ensemble sits on two branches is the per-scan
bimodality test (``engine.axis_bimodality``), stamped on the scan, recounted
by merge_archives, and -- under a "single" reconstruction stamp -- put on the
baseline as ``bistability_source="draws"``.  Nothing filters on any of it.

Synthetic inputs only; no device data.
"""
import contextlib
import io
import os
import sys
import warnings

import h5py
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _engine_toy as T  # noqa: E402
import test_engine_draws as TD  # noqa: E402

toy_bouquet_solver = TD.toy_bouquet_solver
_quiet = TD._quiet


# ---------------------------------------------------------------------------
#  the record itself
# ---------------------------------------------------------------------------
REF = dict(R=1.70, Z=0.0, minor_radius_m=0.60)


@pytest.mark.parametrize("frac, large", [(0.005, False), (0.02, True)])
def test_the_flag_at_half_and_two_percent_of_a(frac, large):
    from bouquet.engine import AXIS_BRANCH_FRAC, axis_shift_record
    assert AXIS_BRANCH_FRAC == 0.01
    a = REF["minor_radius_m"]
    r = axis_shift_record((1.70 + 0.6 * frac * a, 0.8 * frac * a), REF)
    assert r["axis_shift_large"] is large
    assert "axis_branch" not in r         # the per-draw name says "shift"
    assert r["displacement_frac_a"] == pytest.approx(frac, rel=1e-9)
    assert r["displacement_mm"] == pytest.approx(1e3 * frac * a, rel=1e-9)
    assert r["dR_mm"] == pytest.approx(1e3 * 0.6 * frac * a, rel=1e-9)
    assert r["dZ_mm"] == pytest.approx(1e3 * 0.8 * frac * a, rel=1e-9)
    assert (r["reference_R"], r["reference_Z"]) == (1.70, 0.0)
    assert r["minor_radius_m"] == a
    assert r["threshold_frac_a"] == AXIS_BRANCH_FRAC


def test_the_discriminator_is_the_mse_branch_checks_constant():
    import bouquet.engine as be
    assert be.MSE_FD_BRANCH_AXIS_FRAC is be.AXIS_BRANCH_FRAC


def test_an_unknown_axis_or_radius_records_no_flag():
    from bouquet.engine import axis_shift_record
    r = axis_shift_record(None, REF)
    assert r["axis_shift_large"] is None and r["R"] is None
    r = axis_shift_record((1.71, 0.0), dict(REF, minor_radius_m=None))
    assert r["axis_shift_large"] is None and r["displacement_frac_a"] is None
    assert r["dR_mm"] == pytest.approx(10.0)
    r = axis_shift_record((1.71, 0.0), None)
    assert r["axis_shift_large"] is None and r["dR_mm"] is None
    assert r["R"] == 1.71


def test_the_count_and_summary_text():
    from bouquet.engine_draws import axis_shift_counts, axis_shift_text
    diags = [dict(engine=dict(axis=dict(axis_shift_large=False, dZ_mm=0.1,
                                        minor_radius_m=0.6))),
             dict(engine=dict(axis=dict(axis_shift_large=True, dZ_mm=-9.0,
                                        minor_radius_m=0.6))),
             dict(engine=dict(axis=dict(axis_shift_large=None))),
             # a record written before the rename: "other" = large
             dict(engine=dict(axis=dict(axis_branch="other", dZ_mm=-8.0,
                                        minor_radius_m=0.6))),
             dict(jphi_total=None)]                       # a legacy draw
    c = axis_shift_counts(diags)
    assert {k: c[k] for k in ("n_recorded", "n_large", "n_small",
                              "large_draws")} == dict(
        n_recorded=3, n_large=2, n_small=1, large_draws=[1, 3])
    assert c["bimodality"]["axis_bimodal"] is None      # 3 draws: too few
    assert axis_shift_text(c).startswith(
        "axis shift: 2 of 3 archived engine draw(s) displaced > 0.01 x a")
    assert "branch label" in axis_shift_text(c)
    assert axis_shift_text(axis_shift_counts([dict(jphi_total=None)])) == ""
    from bouquet.TokaMaker_interface import _rejection_summary
    assert "axis shift: 2 of 3" in _rejection_summary([], 5, 5,
                                                      diagnostics=diags)
    assert "axis shift" not in _rejection_summary([], 4, 4)


# ---------------------------------------------------------------------------
#  end to end on the stand-in
# ---------------------------------------------------------------------------
def _shifted_fake():
    from _engine_fake_gs import FakeTokaMaker

    class ShiftedAxis(FakeTokaMaker):
        """The axis moves by ``offset`` once the test sets it (after the
        reconstruction): the draws land that far from its axis."""
        offset = (0.0, 0.0)

        @property
        def o_point(self):
            return [T.R0 + self.offset[0], self.offset[1]]
    return ShiftedAxis


def _prepared(tmp_path, monkeypatch):
    import bouquet.run as br
    from bouquet.solver_state import enter_bounded_coil_mode
    cls = _shifted_fake()

    def _setup(self):
        gs = cls(None)
        enter_bounded_coil_mode(gs)
        return self._install_solver(gs)
    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    b = TD._bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    return b


@pytest.mark.parametrize("frac, large", [(0.005, False), (0.02, True)])
def test_every_archived_draw_records_its_axis_and_shift(
        tmp_path, toy_bouquet_solver, monkeypatch, frac, large):
    from bouquet.engine_draws import AXIS_SHIFT_ATTR, read_draw_engine
    from bouquet.schema import (AXIS_BIMODAL_ATTR, AXIS_BIMODAL_JSON_ATTR,
                                AXIS_BRANCH_ATTR_LEGACY, AXIS_DR_ATTR,
                                AXIS_DZ_ATTR, AXIS_MINOR_RADIUS_ATTR,
                                AXIS_OTHER_SCAN_ATTR_LEGACY,
                                AXIS_RECORDED_SCAN_ATTR, AXIS_SHIFT_LARGE_ATTR,
                                AXIS_SHIFT_SCAN_ATTR)
    from bouquet.utils import _group_path
    b = _prepared(tmp_path, monkeypatch)
    a = T.A_MIN
    ref = b._engine_run["delivered_axis"]
    assert (ref["R"], ref["Z"], ref["minor_radius_m"]) == (T.R0, 0.0, a)
    assert b.baseline.engine["delivered_axis"] == ref
    b.mygs.offset = (0.0, frac * a)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diags = b.generate()
    assert len(diags) == 2
    n_large = 2 if large else 0
    for d in diags:
        ax = d["engine"]["axis"]
        assert ax["axis_shift_large"] is large
        assert "axis_branch" not in ax
        assert ax["displacement_frac_a"] == pytest.approx(frac, rel=1e-9)
        assert ax["dZ_mm"] == pytest.approx(1e3 * frac * a, rel=1e-9)
        assert ax["dR_mm"] == 0.0
        assert (ax["R"], ax["Z"]) == (T.R0, pytest.approx(frac * a))
        assert (ax["reference_R"], ax["reference_Z"]) == (T.R0, 0.0)
        # never a filter: the verdict is the coil test AND the bands
        ph = d["engine"]["post_hoc"]
        assert ph["in_spec"] == (ph["coil_in_spec"] and ph["in_band"])
        assert not [r for r in ph["reasons"] if "axis" in r]
    # the summary line, the Bouquet attributes (2 draws: not testable)
    assert f"axis shift: {n_large} of 2 archived engine draw(s)" \
        in buf.getvalue()
    assert b.engine_axis_shift["n_large"] == n_large
    assert b.engine_axis_shift["bimodality"]["axis_bimodal"] is None
    assert b.engine_axis_bimodality["axis_bimodal"] is None
    # archived: the record, the group attrs, the scan count
    h, sk = b.config.output_header, b.config.generation.scan_key
    for i in range(2):
        rec = read_draw_engine(h, i, scan_key=sk)
        assert rec["axis"]["axis_shift_large"] is large
    with h5py.File(h + ".h5", "r") as hf:
        for i in range(2):
            g = hf[_group_path(sk, i)]
            assert bool(g.attrs[AXIS_SHIFT_LARGE_ATTR]) is large
            assert AXIS_BRANCH_ATTR_LEGACY not in g.attrs
            assert g.attrs[AXIS_SHIFT_ATTR] == pytest.approx(frac)
            assert g.attrs[AXIS_DZ_ATTR] == pytest.approx(1e3 * frac * a)
            assert g.attrs[AXIS_DR_ATTR] == 0.0
            assert g.attrs[AXIS_MINOR_RADIUS_ATTR] == pytest.approx(a)
        from bouquet.filtering import _scan_group_path
        s = hf[_scan_group_path(sk)].attrs
        assert int(s[AXIS_SHIFT_SCAN_ATTR]) == n_large
        assert int(s[AXIS_RECORDED_SCAN_ATTR]) == 2
        assert AXIS_OTHER_SCAN_ATTR_LEGACY not in s
        # two draws: the bimodality test is not testable -- no bool stamp,
        # the record says why
        assert AXIS_BIMODAL_ATTR not in s
        import json
        bj = json.loads(s[AXIS_BIMODAL_JSON_ATTR])
        assert bj["axis_bimodal"] is None and "not testable" in bj["reason"]
    # the baseline's engine record carries the reference
    from bouquet.engine import load_baseline_engine
    brec = load_baseline_engine(h, scan_key=sk)
    assert brec["draws"]["axis_reference"]["minor_radius_m"] == a
    # the stats tables
    sp = b.archive.scan(sk).spread("all", print_table=False)
    assert sp["axis_shift"] == {"n": 2, "n_large": n_large,
                                "n_small": 2 - n_large}
    assert sp["axis_bimodality"]["axis_bimodal"] is None
    assert "Lambda" in sp
    from bouquet.stats import draw_scalars
    _quiet(b.filter)
    recs = _quiet(draw_scalars, b.archive, sk, selection="all", min_n=1,
                  hard_min=1)
    # every record carries the count (Lambda's too: the toy stores no
    # beta_p, so its record is empty and counts no used draw)
    assert recs["Lambda"].provenance["axis_shift"]["n_recorded"] == 0
    for q in ("beta_N", "q95"):
        pa = recs[q].provenance["axis_shift"]
        assert len(recs[q].values) == 2, recs[q].status
        assert pa["n_large"] == n_large and pa["n_recorded"] == 2
        # a large displacement alone is no branch claim: no limitation
        lim = " ".join(recs[q].provenance.get("limitations") or [])
        assert "bimodal" not in lim and "branch" not in lim
        assert recs[q].provenance["axis_bimodality"]["axis_bimodal"] is None


@pytest.mark.parametrize("frac, large", [(0.005, False), (0.02, True)])
def test_the_sigma0_draw_carries_the_axis_record_ungated(
        tmp_path, toy_bouquet_solver, monkeypatch, frac, large):
    b = _prepared(tmp_path, monkeypatch)
    b.mygs.offset = (0.0, frac * T.A_MIN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = _quiet(b.verify_sigma0_consistency)
    assert v["axis"]["axis_shift_large"] is large
    assert v["axis"] == v["record"]["axis"]
    assert v["axis"]["displacement_frac_a"] == pytest.approx(frac)
    # reported, not gated: no gate names the axis
    for stage in v["gates"].values():
        assert not [k for k in (stage or {}) if "axis" in k]


def test_a_legacy_scan_carries_no_count(tmp_path):
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.schema import AXIS_BIMODAL_JSON_ATTR, AXIS_SHIFT_SCAN_ATTR
    from bouquet.utils import initialize_equilibrium_database
    h = str(tmp_path / "leg")
    initialize_equilibrium_database(h)
    with h5py.File(h + ".h5", "a") as hf:
        hf.require_group("scan/0/0")
    out = stamp_axis_scan(h, "0")
    assert (out["n_large"], out["n_recorded"], out["bimodality"]) \
        == (0, 0, None)
    with h5py.File(h + ".h5", "r") as hf:
        assert AXIS_SHIFT_SCAN_ATTR not in hf["scan/0"].attrs
        assert AXIS_BIMODAL_JSON_ATTR not in hf["scan/0"].attrs


# ---------------------------------------------------------------------------
#  the per-scan bimodality test
# ---------------------------------------------------------------------------
A_TEST = 0.60          # m: the separation threshold is 0.01 x a = 6 mm


def test_a_clearly_bimodal_set_is_bimodal():
    import numpy as np
    from bouquet.engine import (AXIS_BIMODAL_GAP_RATIO, AXIS_BIMODAL_MIN_COUNT,
                                AXIS_BRANCH_FRAC, axis_bimodality)
    rng = np.random.default_rng(3)
    dz = np.r_[rng.normal(0.0, 0.05, 6), rng.uniform(-24.0, -8.0, 30)]
    r = axis_bimodality(list(dz), A_TEST, reference_Z=0.02)
    assert r["axis_bimodal"] is True
    assert r["counts"] == [30, 6]
    assert r["centres_dZ_mm"][1] == pytest.approx(0.0, abs=0.1)
    assert r["centres_dZ_mm"][0] < -12.0
    assert r["gap_mm"] > r["gap_threshold_mm"]
    assert r["separation_mm"] > r["separation_threshold_mm"] \
        == pytest.approx(1e3 * AXIS_BRANCH_FRAC * A_TEST)
    assert r["centres_Z_m"][1] == pytest.approx(0.02, abs=1e-4)
    # the method and its constants are recorded
    assert r["constants"] == dict(min_count=AXIS_BIMODAL_MIN_COUNT,
                                  gap_ratio=AXIS_BIMODAL_GAP_RATIO,
                                  separation_frac_a=AXIS_BRANCH_FRAC)
    assert r["version"] == "axis_bimodality/1" and r["method"]
    from bouquet.engine import axis_bimodality_text
    assert "BIMODAL" in axis_bimodality_text(r)
    assert "mixtures" in axis_bimodality_text(r)


def test_a_continuous_spread_is_not_bimodal():
    import numpy as np
    from bouquet.engine import axis_bimodality
    # evenly spread displacements, -16 .. +16 mm (no gap above the spacing)
    r = axis_bimodality(list(np.linspace(-16.0, 16.0, 38)), A_TEST)
    assert r["axis_bimodal"] is False
    assert "gap" in r["reason"]
    # a tight sub-threshold scatter split in two is no branch pair either
    r = axis_bimodality([0.0] * 12 + [0.5] * 12, A_TEST)
    assert r["axis_bimodal"] is False
    assert "apart" in r["reason"]


def test_a_tiny_set_is_not_testable():
    from bouquet.engine import AXIS_BIMODAL_MIN_COUNT, axis_bimodality
    n = 2 * AXIS_BIMODAL_MIN_COUNT - 1
    r = axis_bimodality([0.0] * (n - 2) + [-20.0, -21.0], A_TEST)
    assert r["axis_bimodal"] is None and "not testable" in r["reason"]
    assert r["n"] == n
    # None / non-finite entries are skipped, an unknown a is not testable
    r = axis_bimodality([0.0] * 12 + [-20.0] * 12 + [None, float("nan")],
                        None)
    assert r["axis_bimodal"] is None and r["n"] == 24
    assert "minor radius" in r["reason"]
    # the verdict needs AXIS_BIMODAL_MIN_N (24) draws: two clean clusters of
    # 23 draws are not testable, with the count; 24 get the verdict, and the
    # smallest cluster may still hold AXIS_BIMODAL_MIN_COUNT draws
    from bouquet.engine import AXIS_BIMODAL_MIN_N
    assert AXIS_BIMODAL_MIN_N == 24
    two = [0.0, 0.1, 0.2] + [-20.0 - 0.01 * k for k in range(21)]
    r = axis_bimodality(two[:23], A_TEST)
    assert r["axis_bimodal"] is None and r["n"] == 23 and r["min_n"] == 24
    assert "not testable: 23 draw(s)" in r["reason"] and "24" in r["reason"]
    assert "counts" not in r
    r = axis_bimodality(two, A_TEST)
    assert r["axis_bimodal"] is True and r["counts"] == [21, 3]


# ---------------------------------------------------------------------------
#  the scan stamp, the baseline's bistability source, merges, old archives
# ---------------------------------------------------------------------------
def _set_axis(path, sk, dz_by_draw, a=T.A_MIN, legacy=False):
    """Overwrite archived draws' axis attrs (dZ only; a legacy archive's
    form when *legacy*)."""
    from bouquet.schema import (AXIS_BRANCH_ATTR_LEGACY, AXIS_DR_ATTR,
                                AXIS_DZ_ATTR, AXIS_MINOR_RADIUS_ATTR,
                                AXIS_SHIFT_FRAC_ATTR, AXIS_SHIFT_LARGE_ATTR)
    from bouquet.utils import _group_path
    with h5py.File(path, "a") as hf:
        for i, dz in dz_by_draw.items():
            g = hf[_group_path(sk, i)]
            frac = abs(dz) * 1e-3 / a
            g.attrs[AXIS_SHIFT_FRAC_ATTR] = frac
            if legacy:
                for k in (AXIS_SHIFT_LARGE_ATTR, AXIS_DZ_ATTR, AXIS_DR_ATTR,
                          AXIS_MINOR_RADIUS_ATTR):
                    if k in g.attrs:
                        del g.attrs[k]
                g.attrs[AXIS_BRANCH_ATTR_LEGACY] = (
                    "other" if frac > 0.01 else "same")
            else:
                g.attrs[AXIS_SHIFT_LARGE_ATTR] = bool(frac > 0.01)
                g.attrs[AXIS_DZ_ATTR] = float(dz)
                g.attrs[AXIS_DR_ATTR] = 0.0
                g.attrs[AXIS_MINOR_RADIUS_ATTR] = float(a)


#: two clusters of 12 draws: ~0 mm and ~-15 mm (AXIS_BIMODAL_MIN_N = 24)
_BIMODAL24 = {i: (0.1 * (i % 3 - 1) if i < 12 else -15.0 - 0.2 * (i % 5))
              for i in range(24)}


def _generated(tmp_path, monkeypatch, n):
    b = _prepared(tmp_path, monkeypatch)
    b.config.generation.n_equils = n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diags = _quiet(b.generate)
    assert len(diags) == n
    return b


def test_a_bimodal_scan_under_a_single_stamp_gets_the_draws_source(
        tmp_path, toy_bouquet_solver, monkeypatch):
    import json
    from bouquet.engine import load_baseline_engine
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.filtering import _scan_group_path
    from bouquet.schema import (AXIS_BIMODAL_ATTR, AXIS_BIMODAL_JSON_ATTR,
                                BISTABILITY_SOURCE_ATTR)
    from bouquet.utils import _baseline_group_path
    b = _generated(tmp_path, monkeypatch, 24)
    h, sk = b.config.output_header, b.config.generation.scan_key
    bis = load_baseline_engine(h, scan_key=sk)["bistability"]
    assert bis["outcome"] == "single" and bis["bistability_source"] is None
    _set_axis(h + ".h5", sk, _BIMODAL24)
    out = stamp_axis_scan(h, sk)
    assert out["bimodality"]["axis_bimodal"] is True
    assert out["bimodality"]["counts"] == [12, 12]
    assert out["n_large"] == 12 and out["bistability_source"] == "draws"
    with h5py.File(h + ".h5", "r") as hf:
        s = hf[_scan_group_path(sk)].attrs
        assert bool(s[AXIS_BIMODAL_ATTR]) is True
        assert json.loads(s[AXIS_BIMODAL_JSON_ATTR])["counts"] == [12, 12]
        bl = hf[_baseline_group_path(sk)].attrs
        v = bl[BISTABILITY_SOURCE_ATTR]
        assert (v.decode() if isinstance(v, bytes) else v) == "draws"
    bis = load_baseline_engine(h, scan_key=sk)["bistability"]
    assert bis["outcome"] == "single"                  # the check's verdict
    assert bis["bistability_source"] == "draws"        # + the draws'
    assert bis["draws_evidence"]["counts"] == [12, 12]
    # the session record follows (generate's own stamp path)
    with pytest.warns(RuntimeWarning, match="BIMODAL"), \
            contextlib.redirect_stdout(io.StringIO()):
        b._note_axis_scan(out)
    assert b.baseline.engine["bistability"]["bistability_source"] == "draws"
    # the stats name it: spread and draw_scalars provenance (limitation)
    sp = b.archive.scan(sk).spread("all", print_table=False)
    assert sp["axis_bimodality"]["axis_bimodal"] is True
    from bouquet.stats import draw_scalars
    _quiet(b.filter)
    recs = _quiet(draw_scalars, b.archive, sk, selection="all", min_n=1,
                  hard_min=1)
    lim = " ".join(recs["beta_N"].provenance["limitations"])
    assert ("the ensemble is bimodal in the axis position: two equilibrium "
            "branches; the bands are mixtures") in lim
    assert recs["beta_N"].provenance["axis_bimodality"]["axis_bimodal"]
    # a recount that no longer supports it removes the draws' source
    _set_axis(h + ".h5", sk, {i: 0.05 * (i % 3 - 1) for i in range(12, 24)})
    out = stamp_axis_scan(h, sk)
    assert out["bimodality"]["axis_bimodal"] is False
    assert out["bistability_source"] is None
    with h5py.File(h + ".h5", "r") as hf:
        assert bool(hf[_scan_group_path(sk)].attrs[AXIS_BIMODAL_ATTR]) \
            is False
        assert BISTABILITY_SOURCE_ATTR not in hf[_baseline_group_path(
            sk)].attrs
    assert "bistability_source" not in \
        load_baseline_engine(h, scan_key=sk)["bistability"]


def test_another_scans_draws_source_never_lands_in_this_scans_baseline(
        tmp_path, toy_bouquet_solver, monkeypatch):
    """Scan A is bimodal, so the session's live baseline carries
    bistability_source="draws".  Scan B archives its baseline from that live
    record and then has no axis entry (0 archived draws): B's archived record
    must not carry A's evidence, its group attr and its JSON agree, and the
    live record is cleared."""
    import bouquet.engine_draws as ED
    from bouquet.engine import load_baseline_engine
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.schema import BISTABILITY_SOURCE_ATTR
    from bouquet.utils import _baseline_group_path, _group_path
    b = _generated(tmp_path, monkeypatch, 24)
    h, sk_a = b.config.output_header, b.config.generation.scan_key
    _set_axis(h + ".h5", sk_a, _BIMODAL24)
    out_a = stamp_axis_scan(h, sk_a)
    assert out_a["bistability_source"] == "draws"
    with pytest.warns(RuntimeWarning, match="BIMODAL"), \
            contextlib.redirect_stdout(io.StringIO()):
        b._note_axis_scan(out_a)
    assert b.baseline.engine["bistability"]["bistability_source"] == "draws"
    # scan B: its baseline is archived from the live record; its own scan
    # stamp is held back here so the archived record is store_baseline's
    sk_b = "B"
    b.config.generation.scan_key = sk_b
    real_stamp = ED.stamp_axis_scan
    monkeypatch.setattr(ED, "stamp_axis_scan",
                        lambda *a, **k: dict(n_large=0, n_recorded=0,
                                             bimodality=None,
                                             bistability_source="draws"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quiet(b.generate)
    monkeypatch.setattr(ED, "stamp_axis_scan", real_stamp)
    bis_b = load_baseline_engine(h, scan_key=sk_b)["bistability"]
    assert bis_b.get("bistability_source") != "draws"
    assert "draws_evidence" not in bis_b
    assert bis_b["outcome"] == "single"                 # the check's own verdict
    # B with 0 archived draws, and a stale "draws" put on its baseline: the
    # recount (no axis entry) removes it from the attr and the JSON alike
    with h5py.File(h + ".h5", "a") as hf:
        for i in range(24):
            del hf[_group_path(sk_b, i)]
        hf[_baseline_group_path(sk_b)].attrs[BISTABILITY_SOURCE_ATTR] = "draws"
    from bouquet.engine import read_engine_json, write_engine_json
    with h5py.File(h + ".h5", "a") as hf:
        g = hf[_baseline_group_path(sk_b)]
        rec = read_engine_json(g)
        rec["bistability"].update(bistability_source="draws",
                                  draws_evidence={"counts": [3, 3]})
        write_engine_json(g, rec)
    out_b = stamp_axis_scan(h, sk_b)
    assert (out_b["n_recorded"], out_b["bimodality"]) == (0, None)
    assert out_b["bistability_source"] is None
    with h5py.File(h + ".h5", "r") as hf:
        assert BISTABILITY_SOURCE_ATTR not in hf[_baseline_group_path(
            sk_b)].attrs
    bis_b = load_baseline_engine(h, scan_key=sk_b)["bistability"]
    assert bis_b.get("bistability_source") is None
    assert "draws_evidence" not in bis_b
    # ... and the live record follows, though B's n is untestable
    b._note_axis_scan(out_b)
    live = b.baseline.engine["bistability"]
    assert live.get("bistability_source") != "draws"
    assert "draws_evidence" not in live
    # scan A's own archived stamp is untouched
    assert load_baseline_engine(h, scan_key=sk_a)["bistability"][
        "bistability_source"] == "draws"


def test_a_scan_below_the_minimum_gets_no_verdict_and_loses_a_stale_source(
        tmp_path, toy_bouquet_solver, monkeypatch):
    """24 archived draws in two clusters: bimodal, the draws' source stamped.
    One draw fewer (23): not testable, with the count -- no verdict, and the
    now unsupported "draws" source is removed from the archive and the live
    record without a warning."""
    import json
    from bouquet.engine import load_baseline_engine
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.filtering import _scan_group_path
    from bouquet.schema import (AXIS_BIMODAL_ATTR, AXIS_BIMODAL_JSON_ATTR,
                                BISTABILITY_SOURCE_ATTR)
    from bouquet.utils import _baseline_group_path, _group_path
    b = _generated(tmp_path, monkeypatch, 24)
    h, sk = b.config.output_header, b.config.generation.scan_key
    _set_axis(h + ".h5", sk, _BIMODAL24)
    out = stamp_axis_scan(h, sk)
    assert out["bimodality"]["n"] == 24
    assert out["bimodality"]["axis_bimodal"] is True
    with pytest.warns(RuntimeWarning, match="BIMODAL"), \
            contextlib.redirect_stdout(io.StringIO()):
        b._note_axis_scan(out)
    assert b.baseline.engine["bistability"]["bistability_source"] == "draws"
    with h5py.File(h + ".h5", "a") as hf:
        del hf[_group_path(sk, 23)]
    out = stamp_axis_scan(h, sk)
    bim = out["bimodality"]
    assert out["n_recorded"] == 23 and bim["n"] == 23
    assert bim["axis_bimodal"] is None
    assert "not testable: 23 draw(s)" in bim["reason"]
    assert out["bistability_source"] is None
    with h5py.File(h + ".h5", "r") as hf:
        s = hf[_scan_group_path(sk)].attrs
        assert AXIS_BIMODAL_ATTR not in s
        assert json.loads(s[AXIS_BIMODAL_JSON_ATTR])["axis_bimodal"] is None
        assert BISTABILITY_SOURCE_ATTR not in hf[_baseline_group_path(
            sk)].attrs
    assert "bistability_source" not in \
        load_baseline_engine(h, scan_key=sk)["bistability"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        b._note_axis_scan(out)
    assert b.baseline.engine["bistability"].get("bistability_source") is None
    assert b.engine_axis_bimodality["axis_bimodal"] is None


def test_a_bistable_stamp_is_not_overwritten_by_the_draws(
        tmp_path, toy_bouquet_solver, monkeypatch):
    from bouquet.engine import (load_baseline_engine, read_engine_json,
                                write_engine_json)
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.schema import BISTABILITY_SOURCE_ATTR
    from bouquet.utils import _baseline_group_path
    b = _generated(tmp_path, monkeypatch, 24)
    h, sk = b.config.output_header, b.config.generation.scan_key
    with h5py.File(h + ".h5", "a") as hf:
        g = hf[_baseline_group_path(sk)]
        rec = read_engine_json(g)
        rec["bistability"].update(bistable=True, outcome="bistable",
                                  bistability_source="reconstruction")
        write_engine_json(g, rec)
    _set_axis(h + ".h5", sk, _BIMODAL24)
    out = stamp_axis_scan(h, sk)
    assert out["bimodality"]["axis_bimodal"] is True
    assert out["bistability_source"] == "reconstruction"
    bis = load_baseline_engine(h, scan_key=sk)["bistability"]
    assert bis["bistability_source"] == "reconstruction"
    with h5py.File(h + ".h5", "r") as hf:
        assert BISTABILITY_SOURCE_ATTR not in hf[_baseline_group_path(
            sk)].attrs


def test_an_archive_with_the_old_attr_names_is_read_and_migrated(
        tmp_path, toy_bouquet_solver, monkeypatch):
    """Archives written between the axis record's introduction and its
    rename carry ``axis_branch`` per draw and ``n_axis_other`` on the scan:
    both are read (the displacement from the draw's engine record), and a
    re-stamp writes the current names."""
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.filtering import _scan_group_path
    from bouquet.schema import (AXIS_OTHER_SCAN_ATTR_LEGACY,
                                AXIS_RECORDED_SCAN_ATTR, AXIS_SHIFT_SCAN_ATTR,
                                read_axis_entry, read_scan_axis_counts)
    from bouquet.utils import _group_path
    b = _prepared(tmp_path, monkeypatch)
    b.mygs.offset = (0.0, 0.02 * T.A_MIN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _quiet(b.generate)
    h, sk = b.config.output_header, b.config.generation.scan_key
    _set_axis(h + ".h5", sk, {0: 12.0, 1: 12.0}, legacy=True)
    with h5py.File(h + ".h5", "a") as hf:
        s = hf[_scan_group_path(sk)].attrs
        del s[AXIS_SHIFT_SCAN_ATTR]
        s[AXIS_OTHER_SCAN_ATTR_LEGACY] = 2
        assert read_scan_axis_counts(s) == (2, 2)
        e = read_axis_entry(hf[_group_path(sk, 0)])
        assert e["shift_large"] is True and "legacy" in e["source"]
        # the components come from the draw's engine record
        assert e["dZ_mm"] == pytest.approx(1e3 * 0.02 * T.A_MIN)
        assert e["minor_radius_m"] == pytest.approx(T.A_MIN)
    sp = b.archive.scan(sk).spread("all", print_table=False)
    assert sp["axis_shift"]["n_large"] == 2
    out = stamp_axis_scan(h, sk)
    assert (out["n_large"], out["n_recorded"]) == (2, 2)
    with h5py.File(h + ".h5", "r") as hf:
        s = hf[_scan_group_path(sk)].attrs
        assert AXIS_OTHER_SCAN_ATTR_LEGACY not in s
        assert int(s[AXIS_SHIFT_SCAN_ATTR]) == 2
        assert int(s[AXIS_RECORDED_SCAN_ATTR]) == 2


def test_merge_archives_recounts_the_bimodality(tmp_path, toy_bouquet_solver):
    """Two shards, each too small to test (12 draws), one on each axis
    position: the merged scan (24 draws) is recounted -- bimodal -- and its
    baseline gains the draws' source."""
    import test_engine_draws as TDm
    from bouquet.engine import load_baseline_engine
    from bouquet.engine_draws import stamp_axis_scan
    from bouquet.filtering import _scan_group_path
    from bouquet.parallel import merge_archives, run_shard
    from bouquet.schema import AXIS_BIMODAL_ATTR, AXIS_SHIFT_SCAN_ATTR
    b = TDm._bq(tmp_path, n=24)
    out = str(tmp_path / "par")
    metas = [_quiet(run_shard, b.config, w, 2, n_equils_total=24,
                    seed_base=7, out_header=out, scan_key=0,
                    threads_per_worker=1, verbose=True) for w in (0, 1)]
    assert [m["n"] for m in metas] == [12, 12]
    for m, dz in zip(metas, ([0.05 * (k % 3 - 1) for k in range(12)],
                             [-15.0 - 0.2 * (k % 5) for k in range(12)])):
        _set_axis(m["path"], 0, dict(enumerate(dz)))
        r = stamp_axis_scan(m["path"], 0)
        assert r["bimodality"]["axis_bimodal"] is None      # 12: too few
    path, n = merge_archives([m["path"] for m in metas], out, scan_key=0)
    assert n == 24
    with h5py.File(path, "r") as hf:
        s = hf[_scan_group_path(0)].attrs
        assert bool(s[AXIS_BIMODAL_ATTR]) is True
        assert int(s[AXIS_SHIFT_SCAN_ATTR]) == 12
    bis = load_baseline_engine(out, scan_key=0)["bistability"]
    assert bis["bistability_source"] == "draws"
