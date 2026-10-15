"""The per-draw axis record and branch flag of the engine draws -- fast, on
the toy stand-in solver of ``tests/_engine_fake_gs.py``.

Every archived engine draw records its archived state's magnetic axis
against the reconstruction's delivered axis (``engine["axis"]``) and
``axis_branch`` = "same" within ``engine.AXIS_BRANCH_FRAC`` x a, "other"
beyond it (owner decision 2026-10-15).  The flag is stamped on the draw
group, counted in the summary line, in ``ScanView.spread()``, in the
``draw_scalars`` provenance and on the scan (``n_axis_other``); the sigma=0
draw carries it too.  It never filters.

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


@pytest.mark.parametrize("frac, branch", [(0.005, "same"), (0.02, "other")])
def test_the_flag_at_half_and_two_percent_of_a(frac, branch):
    from bouquet.engine import AXIS_BRANCH_FRAC, axis_branch_record
    assert AXIS_BRANCH_FRAC == 0.01
    a = REF["minor_radius_m"]
    r = axis_branch_record((1.70 + 0.6 * frac * a, 0.8 * frac * a), REF)
    assert r["axis_branch"] == branch
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


def test_an_unknown_axis_or_radius_records_no_branch():
    from bouquet.engine import axis_branch_record
    r = axis_branch_record(None, REF)
    assert r["axis_branch"] is None and r["R"] is None
    r = axis_branch_record((1.71, 0.0), dict(REF, minor_radius_m=None))
    assert r["axis_branch"] is None and r["displacement_frac_a"] is None
    assert r["dR_mm"] == pytest.approx(10.0)
    r = axis_branch_record((1.71, 0.0), None)
    assert r["axis_branch"] is None and r["dR_mm"] is None
    assert r["R"] == 1.71


def test_the_count_and_summary_text():
    from bouquet.engine_draws import axis_branch_counts, axis_branch_text
    diags = [dict(engine=dict(axis=dict(axis_branch="same"))),
             dict(engine=dict(axis=dict(axis_branch="other"))),
             dict(engine=dict(axis=dict(axis_branch=None))),
             dict(jphi_total=None)]                       # a legacy draw
    c = axis_branch_counts(diags)
    assert c == dict(n_recorded=2, n_other=1, n_same=1, other_draws=[1])
    assert axis_branch_text(c).startswith(
        "axis branch: 1 of 2 archived engine draw(s) 'other'")
    assert axis_branch_text(axis_branch_counts([dict(jphi_total=None)])) == ""
    from bouquet.TokaMaker_interface import _rejection_summary
    assert "axis branch: 1 of 2" in _rejection_summary([], 4, 4,
                                                       diagnostics=diags)
    assert "axis branch" not in _rejection_summary([], 4, 4)


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


@pytest.mark.parametrize("frac, branch", [(0.005, "same"), (0.02, "other")])
def test_every_archived_draw_records_its_axis_and_branch(
        tmp_path, toy_bouquet_solver, monkeypatch, frac, branch):
    from bouquet.engine_draws import (AXIS_BRANCH_ATTR, AXIS_OTHER_SCAN_ATTR,
                                      AXIS_RECORDED_SCAN_ATTR, AXIS_SHIFT_ATTR,
                                      read_draw_engine)
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
    n_other = 2 if branch == "other" else 0
    for d in diags:
        ax = d["engine"]["axis"]
        assert ax["axis_branch"] == branch
        assert ax["displacement_frac_a"] == pytest.approx(frac, rel=1e-9)
        assert ax["dZ_mm"] == pytest.approx(1e3 * frac * a, rel=1e-9)
        assert ax["dR_mm"] == 0.0
        assert (ax["R"], ax["Z"]) == (T.R0, pytest.approx(frac * a))
        assert (ax["reference_R"], ax["reference_Z"]) == (T.R0, 0.0)
        # never a filter: the verdict is the coil test AND the bands
        ph = d["engine"]["post_hoc"]
        assert ph["in_spec"] == (ph["coil_in_spec"] and ph["in_band"])
        assert not [r for r in ph["reasons"] if "axis" in r]
    # the summary line, the Bouquet attribute
    assert f"axis branch: {n_other} of 2 archived engine draw(s)" \
        in buf.getvalue()
    assert b.engine_axis_branch["n_other"] == n_other
    # archived: the record, the group attrs, the scan count
    h, sk = b.config.output_header, b.config.generation.scan_key
    for i in range(2):
        rec = read_draw_engine(h, i, scan_key=sk)
        assert rec["axis"]["axis_branch"] == branch
    with h5py.File(h + ".h5", "r") as hf:
        for i in range(2):
            g = hf[_group_path(sk, i)]
            v = g.attrs[AXIS_BRANCH_ATTR]
            assert (v.decode() if isinstance(v, bytes) else v) == branch
            assert g.attrs[AXIS_SHIFT_ATTR] == pytest.approx(frac)
        from bouquet.filtering import _scan_group_path
        s = hf[_scan_group_path(sk)].attrs
        assert int(s[AXIS_OTHER_SCAN_ATTR]) == n_other
        assert int(s[AXIS_RECORDED_SCAN_ATTR]) == 2
    # the baseline's engine record carries the reference
    from bouquet.engine import load_baseline_engine
    brec = load_baseline_engine(h, scan_key=sk)
    assert brec["draws"]["axis_reference"]["minor_radius_m"] == a
    # the stats tables
    sp = b.archive.scan(sk).spread("all", print_table=False)
    assert sp["axis_branch"] == {"n": 2, "n_other": n_other,
                                 "n_same": 2 - n_other}
    assert "Lambda" in sp
    from bouquet.stats import draw_scalars
    _quiet(b.filter)
    recs = _quiet(draw_scalars, b.archive, sk, selection="all", min_n=1,
                  hard_min=1)
    # every record carries the count (Lambda's too: the toy stores no
    # beta_p, so its record is empty and counts no used draw)
    assert recs["Lambda"].provenance["axis_branch"]["n_recorded"] == 0
    for q in ("beta_N", "q95"):
        pa = recs[q].provenance["axis_branch"]
        assert len(recs[q].values) == 2, recs[q].status
        assert pa["n_other"] == n_other and pa["n_recorded"] == 2
        lim = " ".join(recs[q].provenance["limitations"])
        assert ("OTHER free-boundary solution branch" in lim) \
            == (n_other > 0)


@pytest.mark.parametrize("frac, branch", [(0.005, "same"), (0.02, "other")])
def test_the_sigma0_draw_carries_the_axis_record_ungated(
        tmp_path, toy_bouquet_solver, monkeypatch, frac, branch):
    b = _prepared(tmp_path, monkeypatch)
    b.mygs.offset = (0.0, frac * T.A_MIN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = _quiet(b.verify_sigma0_consistency)
    assert v["axis"]["axis_branch"] == branch
    assert v["axis"] == v["record"]["axis"]
    assert v["axis"]["displacement_frac_a"] == pytest.approx(frac)
    # reported, not gated: no gate names the axis
    for stage in v["gates"].values():
        assert not [k for k in (stage or {}) if "axis" in k]


def test_a_legacy_scan_carries_no_count(tmp_path):
    from bouquet.engine_draws import (AXIS_OTHER_SCAN_ATTR,
                                      stamp_axis_branch_count)
    from bouquet.utils import initialize_equilibrium_database
    h = str(tmp_path / "leg")
    initialize_equilibrium_database(h)
    with h5py.File(h + ".h5", "a") as hf:
        hf.require_group("scan/0/0")
    assert stamp_axis_branch_count(h, "0") == (0, 0)
    with h5py.File(h + ".h5", "r") as hf:
        assert AXIS_OTHER_SCAN_ATTR not in hf["scan/0"].attrs
