"""Readers can exclude draws whose solve was rescued by the opt-in draw-solve
rescue (#75, owner decision D5 condition 3): ``draw_band(rescued=...)`` and
``merge_archives(rescued=...)``.  Default: include, counted and warned.

Hand-built archives only (no solver); synthetic values.
"""
import json
import warnings

import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq


def _archive(path, rescued=()):
    with h5py.File(path, "w") as hf:
        hf.attrs["schema_version"] = 2
        sg = hf.require_group("scan/1")
        sg.create_group("_baseline").attrs["val"] = 5.0
        for d in range(4):
            g = sg.create_group(str(d))
            g.attrs["val"] = float(d)
            if rescued:
                g.attrs["solve_recovered"] = d in rescued
                if d in rescued:
                    g.attrs["solve_recovered_by"] = "nl_tol=2e-05"
                    g.attrs["solve_nl_tol_accepted"] = 2e-5
    return str(path)


def _ev(view):
    return {"x": float(view.attrs["val"])}


def _band(path, **kw):
    return bq.draw_band(path, "1", _ev, selection="all", require_filter=False,
                        min_n=1, hard_min=1, **kw)["x"]


def test_by_default_rescued_draws_are_kept_counted_and_warned(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={2})
    with pytest.warns(UserWarning, match=r"1 draw\(s\) whose solve was RESCUED"):
        r = _band(p)
    assert sorted(r.values) == [0, 1, 2, 3]
    assert r.provenance["rescued"] == "include"
    assert r.provenance["rescued_draws"] == {2: "nl_tol=2e-05"}


def test_excluded_on_request_with_the_reason(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _band(p, rescued="exclude")
    assert sorted(r.values) == [0, 1, 3]
    assert (2, "rescued:nl_tol=2e-05") in r.dropped


def test_an_archive_without_rescued_draws_is_untouched(tmp_path):
    p = _archive(tmp_path / "a.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _band(p)
    assert sorted(r.values) == [0, 1, 2, 3]
    assert "rescued" not in r.provenance and "rescued_draws" not in r.provenance
    with pytest.raises(ValueError, match="rescued must be one of"):
        _band(p, rescued="drop")


def test_draw_bands_passes_the_policy_through(tmp_path):
    p = _archive(tmp_path / "a.h5", rescued={0, 3})
    t = bq.draw_bands([(p, "1")], _ev, selection="all", require_filter=False,
                      min_n=1, hard_min=1, rescued="exclude")
    assert sorted(t.records[0].values) == [1, 2]


def _shard(path, worker, rescued=()):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["l_i_target"] = 0.9
        g.attrs["Ip_target"] = 1.0e6
        for i in range(2):
            d = hf.create_group(f"scan/0/{i}")
            d.attrs["count"] = i
            if rescued:
                d.attrs["solve_recovered"] = i in rescued
                if i in rescued:
                    d.attrs["solve_recovered_by"] = "urf=0.1"
        hf["scan/0"].attrs["parallel_worker_json"] = json.dumps(
            dict(worker_id=worker, n=2, n_attempts=2, seed=7 + worker))
    return str(path)


def test_merge_includes_by_default_and_records_it(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0, rescued={1})
    s1 = _shard(tmp_path / "w1.h5", 1)
    with pytest.warns(UserWarning, match=r"1 merged draw\(s\) had their solve"):
        out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0)
    assert n == 4
    with h5py.File(out, "r") as hf:
        rec = json.loads(hf["scan/0"].attrs["merge_rescued_json"])
        assert bool(hf["scan/0/1"].attrs["solve_recovered"])
    assert rec["policy"] == "include" and rec["n_rescued"] == 1
    assert rec["draws"] == [dict(shard="w0.h5", index=1,
                                 recovered_by="urf=0.1", merged_as=1)]


def test_merge_excludes_on_request(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0, rescued={1})
    s1 = _shard(tmp_path / "w1.h5", 1, rescued={0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, n = merge_archives([s0, s1], str(tmp_path / "m"), scan_key=0,
                                rescued="exclude")
    assert n == 2
    with h5py.File(out, "r") as hf:
        rec = json.loads(hf["scan/0"].attrs["merge_rescued_json"])
        assert not any(bool(hf[f"scan/0/{k}"].attrs.get("solve_recovered"))
                       for k in ("0", "1"))
    assert rec["policy"] == "exclude" and rec["n_rescued"] == 2
    assert all(d["merged_as"] is None for d in rec["draws"])


def test_merge_without_rescued_draws_writes_nothing_new(tmp_path):
    from bouquet.parallel import merge_archives
    s0 = _shard(tmp_path / "w0.h5", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, n = merge_archives([s0], str(tmp_path / "m"), scan_key=0)
    with h5py.File(out, "r") as hf:
        assert "merge_rescued_json" not in hf["scan/0"].attrs
    with pytest.raises(ValueError, match="rescued must be"):
        merge_archives([s0], str(tmp_path / "m2"), scan_key=0, rescued="no")
