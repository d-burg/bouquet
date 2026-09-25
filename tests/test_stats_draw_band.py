"""bouquet.stats: the standard error-bar recipe (draw_band / draw_bands /
draw_scalars). Fast: hand-built h5 archives + the committed golden fixture,
no solver."""

import json
import math
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq
from bouquet.stats import UnfilteredArchiveError, BandTable

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")


# --------------------------------------------------------------------------
#  hand-built archives
# --------------------------------------------------------------------------
def _make_archive(path, scans, *, stamp=True, scan_attrs=None,
                  baseline_attrs=None, root_version="9.9.9"):
    """``scans = {key: {draw: dict(val=, selected=, coil=, bnd=)}}``.

    Writes scan/<key>/_baseline (attr ``val`` = the baseline's value) and one
    group per draw carrying the filter flags + a ``val`` attr the callable
    evaluator reads back through the view.
    """
    with h5py.File(path, "w") as hf:
        hf.attrs["schema_version"] = 2
        hf.attrs["bouquet_version"] = root_version
        for key, draws in scans.items():
            sg = hf.require_group(f"scan/{key}")
            if stamp:
                sg.attrs["coil_filter"] = "chi2"
                sg.attrs["coil_sigma_model"] = json.dumps(
                    {"kind": "device", "device": "generic", "era": "e1",
                     "acceptance": {"chi2_max": 6.1, "z_max": 6.3, "source": "device"}})
            for k, v in (scan_attrs or {}).items():
                sg.attrs[k] = v
            bl = sg.create_group("_baseline")
            bl.attrs["l_i_scale"] = "iter(li3)"
            bl.attrs["val"] = 5.0
            for k, v in (baseline_attrs or {}).items():
                bl.attrs[k] = v
            for d, rec in draws.items():
                g = sg.create_group(str(d))
                g.attrs["val"] = float(rec.get("val", 1.0))
                g.attrs["l_i(1)"] = 0.9
                g.attrs["l_i(3)"] = 0.7
                if stamp:
                    coil = bool(rec.get("coil", True))
                    bnd = bool(rec.get("bnd", True))
                    g.attrs["passes_coil_filter"] = coil
                    g.attrs["passes_boundary_filter"] = bnd
                    g.attrs["selected"] = coil and bnd
    return str(path)


def _simple(path, n, *, vals=None, stamp=True, **kw):
    vals = list(range(n)) if vals is None else list(vals)
    return _make_archive(path, {"1": {i: {"val": vals[i]} for i in range(n)}},
                         stamp=stamp, **kw)


def _ok(v, regular=True, status="ok"):
    return {"value": v, "status": status, "regular": regular}


def _by_attr(view):
    """Callable evaluator: reads the ``val`` attr through the view."""
    return {"x": _ok(float(view.attrs["val"]))}


# --------------------------------------------------------------------------
#  population + dropped reasons
# --------------------------------------------------------------------------
def test_population_and_every_dropped_reason(tmp_path):
    draws = {i: {"val": float(i)} for i in range(10)}
    draws[0]["coil"] = False                      # -> not_selected
    p = _make_archive(tmp_path / "a.h5", {"1": draws})
    ev = {d: {"x": _ok(float(d))} for d in range(10) if d != 3}   # 3 -> not_evaluated
    ev[2] = {"x": _ok(2.0, status="solver_failed")}
    ev[4] = {"x": _ok(float("nan"), regular=False)}
    ev[4]["x"]["label"] = "no_q2_surface"
    ev[5] = {"x": _ok(float("inf"))}
    ev["_baseline"] = {"x": _ok(4.5)}
    r = bq.draw_band(p, "1", ev, exclude={1: "bad coil read"})["x"]
    reasons = dict(r.dropped)
    assert reasons == {0: "not_selected", 1: "user:bad coil read",
                       2: "status:solver_failed", 3: "status:not_evaluated",
                       4: "irregular:no_q2_surface", 5: "non_finite"}
    assert r.n_stored == 10 and r.n_selected == 9
    assert r.n_evaluated == 7                     # 9 selected - 1 excluded - 1 missing
    assert r.n_ok == 6                            # draws 4..9 (2 failed status)
    assert r.n_regular == 5 and r.n_used == 4
    assert sorted(r.values) == [6, 7, 8, 9]
    assert r.regular_fraction == pytest.approx(5 / 6)
    assert r.status == "ok"


def test_exclude_unknown_draw_raises(tmp_path):
    p = _simple(tmp_path / "a.h5", 6)
    with pytest.raises(KeyError):
        bq.draw_band(p, "1", _by_attr, exclude={99: "typo"})


def test_require_filter_raises_on_unstamped_archive(tmp_path):
    p = _simple(tmp_path / "u.h5", 6, stamp=False)
    with pytest.raises(UnfilteredArchiveError, match="coil_filter"):
        bq.draw_band(p, "1", _by_attr)
    r = bq.draw_band(p, "1", _by_attr, require_filter=False)["x"]
    assert r.n_used == 6 and r.provenance["coil_filter"] == "none"
    assert any("No coil_filter stamp" in s for s in r.provenance["limitations"])


def test_callable_runs_on_draws_and_baseline(tmp_path):
    seen = []

    def ev(view):
        seen.append(view.is_baseline)
        return _by_attr(view)

    p = _simple(tmp_path / "a.h5", 6)
    r = bq.draw_band(p, "1", ev)["x"]
    assert seen.count(True) == 1 and seen.count(False) == 6
    assert r.baseline_value == 5.0 and r.baseline_status == "ok"
    # baseline=False: never evaluated, never reported
    seen.clear()
    r = bq.draw_band(p, "1", ev, baseline=False)["x"]
    assert True not in seen and r.baseline_value is None


def test_per_quantity_status(tmp_path):
    p = _simple(tmp_path / "a.h5", 6)
    ev = {d: {"dp21": _ok(float(d)), "dp31": _ok(10.0 + d)} for d in range(6)}
    ev[2]["dp21"] = _ok(float("nan"), status="no_q2_surface")
    out = bq.draw_band(p, "1", ev)
    assert out["dp21"].n_used == 5 and (2, "status:no_q2_surface") in out["dp21"].dropped
    assert out["dp31"].n_used == 6 and out["dp31"].dropped == []


# --------------------------------------------------------------------------
#  statistic, floors, pole rule
# --------------------------------------------------------------------------
def test_percentiles_match_numpy_linear(tmp_path):
    rng = np.random.default_rng(0)
    vals = rng.lognormal(size=23)
    p = _simple(tmp_path / "a.h5", 23, vals=vals)
    r = bq.draw_band(p, "1", _by_attr)["x"]
    lo, hi = np.percentile(vals, (16, 84), method="linear")
    assert r.p16 == lo and r.p84 == hi
    assert r.median == np.median(vals)
    assert r.min == vals.min() and r.max == vals.max()
    assert r.percentile_method == "linear" and tuple(r.percentiles) == (16, 84)
    assert not r.below_floor and not r.no_band and r.show
    assert r.std == pytest.approx(np.std(vals, ddof=1))
    assert isinstance(r.skew_flag, bool)


def test_hard_min_gives_nan_band_but_median(tmp_path):
    p = _simple(tmp_path / "a.h5", 4, vals=[1.0, 2.0, 3.0, 10.0])
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.n_used == 4 and r.no_band and r.below_floor
    assert math.isnan(r.p16) and math.isnan(r.p84)
    assert r.median == 2.5 and r.min == 1.0 and r.max == 10.0
    assert r.values == {0: 1.0, 1: 2.0, 2: 3.0, 3: 10.0}


def test_min_n_flags_but_keeps_band(tmp_path):
    p = _simple(tmp_path / "a.h5", 10)
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.below_floor and not r.no_band and r.show
    assert np.isfinite(r.p16) and np.isfinite(r.p84)
    assert any("covers about" in s for s in r.provenance["limitations"])


def test_majority_regular_gate(tmp_path):
    p = _simple(tmp_path / "a.h5", 12)
    ev = {d: {"x": _ok(float(d), regular=(d % 2 == 0))} for d in range(12)}
    r = bq.draw_band(p, "1", ev)["x"]
    assert r.n_ok == 12 and r.n_regular == 6 and r.regular_fraction == 0.5
    assert r.gated and not r.show                 # <= 1/2 -> hidden
    assert r.n_used == 6 and np.isfinite(r.p16)   # but the band is still computed
    # one more regular draw -> majority -> shown
    ev[1]["x"]["regular"] = True
    r = bq.draw_band(p, "1", ev)["x"]
    assert not r.gated and r.show


def test_no_magnitude_cut_extreme_stays_in(tmp_path):
    vals = [1.0 + 0.01 * i for i in range(19)] + [1.0e6]
    p = _simple(tmp_path / "a.h5", 20, vals=vals)
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.n_used == 20 and r.max == 1.0e6 and r.values[19] == 1.0e6
    assert r.n_extreme == 1
    assert r.p84 == np.percentile(vals, 84, method="linear")


def test_baseline_quantile_and_outside_range(tmp_path):
    p = _simple(tmp_path / "a.h5", 10)            # draws 0..9, baseline attr 5.0
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.baseline_value == 5.0
    assert r.baseline_quantile == 0.5 and r.baseline_outside_range is False
    ev = {d: {"x": _ok(float(d))} for d in range(10)}
    ev["_baseline"] = {"x": _ok(14.28)}
    r = bq.draw_band(p, "1", ev)["x"]
    assert r.baseline_quantile == 1.0 and r.baseline_outside_range is True
    assert r.median == 4.5                        # never centred on the baseline


# --------------------------------------------------------------------------
#  provenance: recorded vs unrecorded
# --------------------------------------------------------------------------
def test_unrecorded_fields_are_reported_not_guessed(tmp_path):
    p = _simple(tmp_path / "a.h5", 6)
    r = bq.draw_band(p, "1", _by_attr, evaluator_meta={"code": "X", "grid": "g1"})["x"]
    assert r.n_requested is None and r.n_requested_source == "unrecorded"
    assert r.n_attempted is None
    assert r.closure_limited is None and r.closure_channel is None
    pv = r.provenance
    assert pv["rms_max_mm"] == "unrecorded" and pv["max_max_mm"] == "unrecorded"
    assert pv["draw_boundary_rms_mm"] == "unrecorded"
    assert pv["generation_mode"] == "unrecorded"
    assert pv["bouquet_version"]["scan"] == "unrecorded"
    assert pv["bouquet_version"]["file"] == "9.9.9"
    assert pv["bouquet_version"]["reader"] == bq.__version__
    assert pv["coil_filter"] == "chi2"
    assert pv["coil_sigma_model"]["chi2_max"] == 6.1
    assert pv["coil_sigma_model"]["z_max"] == 6.3
    assert pv["coil_sigma_model"]["era"] == "e1"
    assert pv["evaluator_meta"] == {"code": "X", "grid": "g1"}
    assert pv["pole_rule"] == "majority_regular"
    assert (pv["min_n"], pv["hard_min"]) == (15, 5)
    assert any("Only the perturbations bouquet samples" in s for s in pv["limitations"])


def test_recorded_fields_are_read(tmp_path):
    meta = {"ip_closure": {"closure_limited": True,
                           "closure_limited_reasons": ["bs_scale 0.2 < 0.3"],
                           "closure_channel": "structured"},
            "closure_limited": True}
    p = _make_archive(
        tmp_path / "a.h5", {"1": {i: {"val": i} for i in range(6)}},
        scan_attrs={"n_requested": 20, "n_attempted": 31, "generation_mode": "until_n",
                    "boundary_rms_max_mm": 8.5, "boundary_max_max_mm": 20.0,
                    "boundary_cut_source": "device:generic-test",
                    "bouquet_version": "1.4.0"},
        baseline_attrs={"li_metrics_json": json.dumps(meta)})
    with h5py.File(p, "a") as hf:
        hf["scan/1/2"].attrs["boundary_rms_mm"] = 3.25
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert (r.n_requested, r.n_requested_source) == (20, "archive:n_requested")
    assert r.n_attempted == 31
    assert r.closure_limited is True and r.closure_channel == "structured"
    assert r.closure_limited_reasons == ("bs_scale 0.2 < 0.3",)
    pv = r.provenance
    assert (pv["rms_max_mm"], pv["max_max_mm"]) == (8.5, 20.0)
    assert pv["boundary_cut_source"] == "device:generic-test"
    assert pv["draw_boundary_rms_mm"] == {2: 3.25}
    assert pv["generation_mode"] == "until_n"
    assert pv["bouquet_version"]["scan"] == "1.4.0"
    assert any("closure_limited" in s for s in pv["limitations"])


def test_n_requested_from_config(tmp_path):
    p = _simple(tmp_path / "a.h5", 6)
    cfg = bq.BouquetConfig(source=bq.ImasSource(ids_path="unused.json"),
                           solver=bq.SolverConfig(mesh_path="unused.h5"),
                           output_header="t")
    cfg.generation.n_equils = 12
    bq.write_provenance(p, cfg, scan_key="1")
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert (r.n_requested, r.n_requested_source) == (12, "config_json:generation.n_equils")
    cfg.generation.n_inspec_target = 8
    bq.write_provenance(p, cfg, scan_key="1")
    r = bq.draw_band(p, "1", _by_attr)["x"]
    assert r.n_requested == 8
    assert r.n_requested_source == "config_json:generation.n_inspec_target"


# --------------------------------------------------------------------------
#  many keys, table, CSV, plot
# --------------------------------------------------------------------------
def test_draw_bands_missing_and_refused(tmp_path):
    p = _make_archive(tmp_path / "a.h5",
                      {"1": {i: {"val": i} for i in range(6)},
                       "2": {i: {"val": 2 * i} for i in range(6)}})
    with h5py.File(p, "a") as hf:
        hf.create_group("scan/3").attrs["refused_reason"] = "closure gate rejected"
    t = bq.draw_bands([(p, "1"), (p, "2"), (p, "3"), (p, "4"),
                       (str(tmp_path / "absent.h5"), "1")], _by_attr)
    assert isinstance(t, BandTable) and t.quantities == ["x"]
    st = {(r.scan_key, r.status) for r in t}
    assert st == {("1", "ok"), ("2", "ok"), ("3", "refused"),
                  ("4", "no_archive"), ("1", "no_archive")}
    ref = [r for r in t if r.status == "refused"][0]
    assert ref.refused_reason == "closure gate rejected" and ref.quantity == "x"
    assert [r.median for r in t if r.status == "ok"] == [2.5, 5.0]
    # refused / no_archive records honour the provenance contract too: the
    # recipe settings, the reader version, and a limitation saying why
    for r in t:
        if r.status in ("refused", "no_archive"):
            pv = r.provenance
            assert pv["pole_rule"] == "majority_regular" and pv["min_n"] == 15
            assert pv["boundary_cut_source"] == "unrecorded"
            assert pv["bouquet_version"]["reader"] == bq.__version__
            assert any(f"status={r.status}" in l for l in pv["limitations"])
            assert r.show is False


def test_status_records_carry_the_requested_recipe_settings(tmp_path):
    p = _make_archive(tmp_path / "a.h5", {"1": {i: {} for i in range(6)}})
    with h5py.File(p, "a") as hf:
        hf.create_group("scan/9").attrs["refused_reason"] = "no reference"
    r = bq.draw_band(p, "9", _by_attr, quantities=["x"], min_n=7, hard_min=3,
                     percentiles=(25, 75))["x"]
    assert r.status == "refused" and r.provenance["min_n"] == 7
    assert r.provenance["percentiles"] == [25, 75]
    assert "no reference" in r.provenance["limitations"][-1]


def test_draw_bands_mapping_and_per_key_exclude(tmp_path):
    p = _make_archive(tmp_path / "a.h5",
                      {"1": {i: {} for i in range(6)}, "2": {i: {} for i in range(6)}})
    ev = {"1": {d: {"x": _ok(float(d))} for d in range(6)},
          "2": {d: {"x": _ok(10.0 + d)} for d in range(6)}}
    t = bq.draw_bands([(p, "1"), (p, "2")], ev, exclude={"2": {0: "hand"}})
    r1, r2 = t.records
    assert r1.n_used == 6 and r2.n_used == 5 and (0, "user:hand") in r2.dropped


def test_to_csv_round_trip(tmp_path):
    p = _simple(tmp_path / "a.h5", 17, vals=np.random.default_rng(3).normal(size=17))
    t = bq.draw_bands([(p, "1"), (p, "9")], _by_attr)
    out = t.to_csv(tmp_path / "bands.csv")
    rows = BandTable.read_csv(out)
    assert len(rows) == 2
    for rec, row in zip(t.records, rows):
        for k in ("median", "p16", "p84", "min", "max", "baseline_value"):
            a, b = getattr(rec, k), row[k]
            assert (a is None and b is None) or (math.isnan(a) and math.isnan(b)) or a == b
        assert row["status"] == rec.status and row["n_used"] == rec.n_used
        assert row["below_floor"] == rec.below_floor
        assert {int(k): v for k, v in row["values"].items()} == rec.values
        assert [tuple(x) for x in row["dropped"]] == [tuple(x) for x in rec.dropped]
    json.dumps(t.records[0].to_dict())           # JSON-able
    assert "BandRecord" in repr(t.records[0])


def test_plot_band_smoke(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    p = _make_archive(tmp_path / "a.h5",
                      {"1": {i: {"val": i} for i in range(20)},
                       "2": {i: {"val": i} for i in range(6)}})
    ev = {"1": {d: {"x": _ok(float(d))} for d in range(20)},
          "2": {d: {"x": _ok(float(d), regular=d < 2)} for d in range(6)}}
    t = bq.draw_bands([(p, "1"), (p, "2"), (p, "7")], ev)
    ax = bq.plot_band(t, "x")
    texts = [tx.get_text() for tx in ax.texts]
    assert "2/6" in texts and "no_archive" in texts


def test_flat_layout(tmp_path):
    p = str(tmp_path / "flat.h5")
    with h5py.File(p, "w") as hf:
        hf.attrs["schema_version"] = 2
        hf.attrs["bouquet_version"] = "9.9.9"
        hf.attrs["coil_filter"] = "legacy"
        for i in range(6):
            g = hf.create_group(str(i))
            g.attrs["val"] = float(i)
            g.attrs["passes_coil_filter"] = i != 0
            g.attrs["selected"] = i != 0
    r = bq.draw_band(p, None, _by_attr)["x"]
    assert r.scan_key is None and r.n_stored == 6 and r.n_selected == 5
    assert r.provenance["coil_filter"] == "legacy"
    assert r.provenance["bouquet_version"]["scan"] == "unrecorded"
    assert r.baseline_status == "no_baseline" and r.baseline_value is None


def test_bad_arguments(tmp_path):
    p = _simple(tmp_path / "a.h5", 6)
    with pytest.raises(ValueError):
        bq.draw_band(p, "1", _by_attr, pole_rule="magnitude")
    with pytest.raises(ValueError):
        bq.draw_band(p, "1", _by_attr, hard_min=20)
    with pytest.raises(KeyError):
        bq.draw_band(p, "nope", _by_attr)


# --------------------------------------------------------------------------
#  draw_scalars on the golden fixture + spread docstring
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def golden_scalars():
    if not os.path.isfile(_GOLDEN):
        pytest.skip("golden fixture absent")
    # the slim golden predates the coil_filter stamp -> knowingly unfiltered
    return bq.draw_scalars(_GOLDEN, "0", rational=((2, 1), (10, 1)),
                           require_filter=False)


def test_draw_scalars_golden_values(golden_scalars):
    s = golden_scalars
    for q in ("q0", "q95", "rho(q=2/1)", "l_i", "beta_N", "<P> [kPa]"):
        r = s[q]
        assert r.status == "ok" and r.n_used == r.n_stored > 0, q
        assert all(np.isfinite(v) for v in r.values.values()), q
        assert r.p16 <= r.median <= r.p84, q
    assert 0.0 < s["rho(q=2/1)"].median < 1.0
    assert s["q0"].median < s["q95"].median


def test_forced_eq_fsa_source_does_not_fall_back_for_the_baseline():
    if not os.path.exists(_GOLDEN):
        pytest.skip("golden fixture absent")
    ar = bq.BouquetArchive(_GOLDEN)
    key = ar.scan_keys[0]
    s = bq.draw_scalars(ar, key, require_filter=False, q_source="eq_fsa")
    r = s["q0"]
    # draws have eq_fsa; the baseline has none -> its overlay is absent and
    # SAYS so, rather than being quietly taken from the g-file
    assert np.isfinite(r.median)
    assert r.baseline_status == "status:no_eq_fsa"
    assert r.baseline_value is None


def test_draw_scalars_rational_flag_matches_q_range(golden_scalars):
    ar = bq.BouquetArchive(_GOLDEN)
    has2 = has10 = 0
    for d in ar["0"].all:
        q = np.abs(np.asarray(d.equilibrium().qpsi, dtype=float))
        has2 += int(q.min() <= 2.0 <= q.max())
        has10 += int(q.min() <= 10.0 <= q.max())
    r2, r10 = golden_scalars["rho(q=2/1)"], golden_scalars["rho(q=10/1)"]
    assert r2.n_regular == has2
    assert r10.n_regular == has10
    if has10 * 2 <= r10.n_ok:
        assert r10.gated and not r10.show
        assert all(why == "irregular:no_q=10/1_surface"
                   for _, why in r10.dropped)


def test_draw_scalars_names_li_estimator(golden_scalars):
    li = golden_scalars["l_i"]
    meta = li.provenance["evaluator_meta"]
    assert meta["l_i_estimator"] == "iter(li3)" and meta["l_i_attr"] == "l_i(3)"
    sc = bq.BouquetArchive(_GOLDEN)["0"]
    assert li.values == {d.count: d.li3 for d in sc.all}


def test_spread_docstring_points_to_draw_scalars():
    assert "draw_scalars" in bq.ScanView.spread.__doc__
