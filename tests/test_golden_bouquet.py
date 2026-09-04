"""Golden-value regression tests for the bouquet pipeline.

These run against a slimmed, git-tracked fixture
(``tests/golden/D3Dlike_Hmode_golden_slim.h5``) plus a JSON manifest of
expected values (``tests/golden/golden_manifest.json``).  Both are produced by
``tests/golden/make_golden_fixture.py`` from a real bouquet run.

The tests:
  * assert every per-draw + baseline scalar in the fixture matches the
    manifest within tight tolerances (l_i, Ip, coil drifts, boundary
    deviation, coil currents, TokaMaker X-points);
  * exercise the postprocessing filters and selection helpers on the golden
    data (coil filter must reproduce the stored ``in_spec`` decision);
  * confirm the stored X-points drive ``_extract_boundary_points``.

To intentionally update the golden set: regenerate the full ``.h5`` (re-run
the notebook), run ``make_golden_fixture.py``, and commit the new fixture +
manifest.  The manifest git diff shows exactly which values moved.
"""
import json
import os
import shutil

import numpy as np
import pytest

import h5py

import bouquet
from bouquet import (filter_coil_currents, filter_boundaries,
                     select_indices, read_filter_flags, export_filtered)
from bouquet.plotting import _extract_boundary_points
from bouquet.utils import read_eqdsk_from_bytes  # noqa: F401  (import smoke)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
_SLIM = os.path.join(_GOLDEN_DIR, "D3Dlike_Hmode_golden_slim.h5")
_MANIFEST = os.path.join(_GOLDEN_DIR, "golden_manifest.json")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_SLIM) and os.path.isfile(_MANIFEST)),
    reason="golden fixture not built; run tests/golden/make_golden_fixture.py")


@pytest.fixture(scope="module")
def manifest():
    with open(_MANIFEST) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tol(manifest):
    return manifest["tolerances"]


def _scan_args(bkey):
    """Map a manifest scan key back to a scan_key argument."""
    return None if bkey == "None" else bkey


def _iter_scans(manifest):
    for bkey, entry in manifest["scans"].items():
        yield bkey, _scan_args(bkey), entry


# ---------------------------------------------------------------------------
#  structural
# ---------------------------------------------------------------------------
def test_fixture_structure(manifest):
    for bkey, sv, entry in _iter_scans(manifest):
        idxs = bouquet.utils.list_equilibrium_indices(_SLIM, scan_key=sv)
        assert idxs == entry["draw_indices"]
        assert len(idxs) == entry["n_draws"]


def test_no_pfile_blobs_in_fixture():
    """The slim fixture drops p-file byte blobs (kept in the example artifact)."""
    bad = []
    with h5py.File(_SLIM, "r") as hf:
        def _check(name, obj):
            if isinstance(obj, h5py.Dataset) and (name.endswith(".pfile") or name == "pfile"):
                bad.append(name)
        hf.visititems(_check)
    assert not bad, f"slim fixture still has p-file blobs: {bad}"


def test_geqdsks_retained_per_manifest(manifest):
    """geqdsks the manifest says were kept are present (and gzip-compressed)."""
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            for i in entry["eqdsk_indices"]:
                grp = hf[f"{prefix}{i}"]
                eqk = (["eqdsk"] if "eqdsk" in grp else [])
                assert eqk, f"draw {i} missing its retained geqdsk"
                # stored gzip-compressed to keep the fixture small
                assert grp[eqk[0]].compression == "gzip"


# ---------------------------------------------------------------------------
#  per-draw scalar golden values
# ---------------------------------------------------------------------------
def test_draw_scalars_match_manifest(manifest, tol):
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            for sidx, exp in entry["draws"].items():
                a = hf[f"{prefix}{sidx}"].attrs
                assert a["l_i(1)"] == pytest.approx(
                    exp["l_i(1)"], abs=tol["l_i_atol"])
                assert a["l_i(3)"] == pytest.approx(
                    exp["l_i(3)"], abs=tol["l_i_atol"])
                assert a["Ip"] == pytest.approx(
                    exp["Ip"], rel=tol["Ip_rtol"])
                assert a["max_F_drift_pct"] == pytest.approx(
                    exp["max_F_drift_pct"], abs=tol["drift_atol"])
                assert a["max_VSC_drift_pct"] == pytest.approx(
                    exp["max_VSC_drift_pct"], abs=tol["drift_atol"])
                assert bool(a["in_spec"]) == exp["in_spec"]


def test_coil_currents_match_manifest(manifest, tol):
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            for sidx, exp in entry["draws"].items():
                if "coil_currents" not in exp:
                    continue
                grp = hf[f"{prefix}{sidx}"]
                names = [n.decode() if isinstance(n, bytes) else str(n) for n in grp["coil_names"][()]]
                vals = np.asarray(grp["coil_currents"][()], dtype=float)
                got = {n: float(v) for n, v in zip(names, vals)}
                assert set(got) == set(exp["coil_currents"])
                for n, v in exp["coil_currents"].items():
                    assert got[n] == pytest.approx(v, abs=tol["coil_atol_A"])


def test_xpoints_match_manifest(manifest, tol):
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            # baseline
            if "x_points" in entry["baseline"]:
                xp = np.asarray(hf[f"{prefix}_baseline/x_points"][()],
                                dtype=float)
                np.testing.assert_allclose(
                    xp, np.asarray(entry["baseline"]["x_points"]),
                    atol=tol["xpoint_atol_m"])
            for sidx, exp in entry["draws"].items():
                if "x_points" not in exp:
                    continue
                xp = np.asarray(hf[f"{prefix}{sidx}/x_points"][()], dtype=float)
                np.testing.assert_allclose(
                    xp, np.asarray(exp["x_points"]), atol=tol["xpoint_atol_m"])


def test_boundary_deviations_match_manifest(manifest, tol):
    """filter_boundaries (apply=False) must reproduce the manifest RMS/max."""
    for bkey, sv, entry in _iter_scans(manifest):
        summ, _ = filter_boundaries(_SLIM, scan_key=sv, apply=False,
                                    plot=False)
        draws = summ["draws"]
        for sidx, exp in entry["draws"].items():
            got = draws[int(sidx)]
            assert got["rms_mm"] == pytest.approx(
                exp["bnd_rms_mm"], abs=tol["bnd_atol_mm"])
            assert got["max_mm"] == pytest.approx(
                exp["bnd_max_mm"], abs=tol["bnd_atol_mm"])


# ---------------------------------------------------------------------------
#  geqdsk handling (the retained, gzipped g-files)
# ---------------------------------------------------------------------------
def _iter_stored_geqdsks(manifest):
    """Yield (raw_bytes, expected_Ip, label) for every retained geqdsk."""
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            # baseline
            if entry["baseline"].get("has_eqdsk"):
                grp = hf[f"{prefix}_baseline"]
                eqk = "eqdsk"
                yield (bytes(grp[eqk][()]), entry["baseline"]["Ip"],
                       f"{bkey}/baseline")
            for i in entry["eqdsk_indices"]:
                grp = hf[f"{prefix}{i}"]
                eqk = "eqdsk"
                yield (bytes(grp[eqk][()]),
                       entry["draws"][str(i)]["Ip"], f"{bkey}/{i}")


def test_geqdsk_parse_and_ip(manifest, tol):
    """Every retained geqdsk parses and its Ip matches the manifest.

    Validates the full geqdsk write->store->read path on real (coarse)
    g-files, not just the cached scalar attrs.
    """
    from bouquet import read_geqdsk
    from bouquet.utils import read_eqdsk_from_bytes
    n = 0
    for raw, exp_ip, label in _iter_stored_geqdsks(manifest):
        eq = read_eqdsk_from_bytes(raw, read_geqdsk)
        assert abs(eq.Ip) == pytest.approx(abs(exp_ip), rel=1e-4), label
        # closed, non-trivial boundary contour
        assert len(eq.boundary_R) == len(eq.boundary_Z)
        assert len(eq.boundary_R) > 10, label
        # magnetic axis sits inside the boundary's radial extent
        assert eq.boundary_R.min() < eq.R_mag < eq.boundary_R.max(), label
        # normalised flux is well-formed
        psi_N = np.asarray(eq.psi_N, dtype=float)
        assert psi_N[0] == pytest.approx(0.0, abs=1e-6), label
        assert np.all(np.diff(psi_N) > 0), label
        n += 1
    assert n >= 1, "no retained geqdsks were exercised"


def test_geqdsk_separatrix_is_coarse(manifest):
    """Document the known geqdsk limitation: the stored boundary is far
    coarser than the high-res LCFS reference we keep alongside it.

    This is *why* the pipeline stores ``perturbed_lcfs_ref`` separately;
    the test guards that the coarse g-file boundary and the fine
    trace-surf reference stay distinct sources.
    """
    from bouquet import read_geqdsk
    from bouquet.utils import read_eqdsk_from_bytes
    checked = 0
    with h5py.File(_SLIM, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            for i in entry["eqdsk_indices"]:
                grp = hf[f"{prefix}{i}"]
                if "perturbed_lcfs_ref" not in grp:
                    continue
                eqk = "eqdsk"
                eq = read_eqdsk_from_bytes(bytes(grp[eqk][()]), read_geqdsk)
                fine = np.asarray(grp["perturbed_lcfs_ref"][()])
                assert len(fine) > 5 * len(eq.boundary_R), (
                    f"{bkey}/{i}: expected the trace-surf reference to be "
                    f"much finer than the coarse geqdsk separatrix")
                checked += 1
    if checked == 0:
        pytest.skip("no retained geqdsks with a high-res LCFS reference")


# ---------------------------------------------------------------------------
#  filtering behaviour on golden data
# ---------------------------------------------------------------------------
def test_coil_filter_reproduces_in_spec(tmp_path, manifest):
    """The postprocess coil filter (stored thresholds) == stored in_spec."""
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    summ, _ = filter_coil_currents(work, apply=True, plot=False)
    flags = read_filter_flags(work)
    for sv, drect in flags.items():
        for i, rec in drect.items():
            if "in_spec" in rec:
                assert rec["passes_coil_filter"] == rec["in_spec"]
    # counts line up with the manifest
    for bkey, sv, entry in _iter_scans(manifest):
        assert summ[sv]["n_pass"] == entry["n_in_spec"]


def test_selection_partition(tmp_path, manifest):
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    filter_coil_currents(work, apply=True, plot=False)
    for bkey, sv, entry in _iter_scans(manifest):
        alli = select_indices(work, scan_key=sv, selection="all")
        sel = select_indices(work, scan_key=sv, selection="selected")
        exc = select_indices(work, scan_key=sv, selection="excluded")
        assert sorted(sel + exc) == sorted(alli)        # partition
        assert set(sel).isdisjoint(exc)
        assert len(sel) == entry["n_in_spec"]


def test_selection_unfiltered_defaults():
    """Before any filter is applied, 'selected' == all, 'excluded' == none."""
    assert select_indices(_SLIM, scan_key="0", selection="selected") == \
        select_indices(_SLIM, scan_key="0", selection="all")
    assert select_indices(_SLIM, scan_key="0", selection="excluded") == []


def test_export_filtered_keeps_selected(tmp_path, manifest):
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    filter_coil_currents(work, apply=True, plot=False)
    out = str(tmp_path / "selected.h5")
    kept = export_filtered(work, out)
    total_sel = sum(e["n_in_spec"] for _, _, e in _iter_scans(manifest))
    assert kept == total_sel
    for bkey, sv, entry in _iter_scans(manifest):
        got = bouquet.utils.list_equilibrium_indices(out, scan_key=sv)
        assert got == select_indices(work, scan_key=sv, selection="selected")
        # baseline preserved
        with h5py.File(out, "r") as hf:
            bl = f"scan/{bkey}/_baseline" if sv is not None else "_baseline"
            assert bl in hf


def test_boundary_filter_cut(tmp_path):
    """A boundary RMS cut marks the expected draws and is reversible."""
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    # diagnostic-only first: no flags written
    summ0, _ = filter_boundaries(work, apply=True, plot=False)
    assert summ0["0"]["cutting"] is False
    flags0 = read_filter_flags(work)
    assert all("passes_boundary_filter" not in r
               for r in flags0["0"].values())
    # now cut at a tight RMS so some draws fail
    summ, _ = filter_boundaries(work, rms_max_mm=2.0, apply=True, plot=False)
    assert summ["0"]["cutting"] is True
    assert 0 < summ["0"]["n_pass"] < summ["0"]["n_total"]


# ---------------------------------------------------------------------------
#  X-point drives boundary-point extraction
# ---------------------------------------------------------------------------
def test_stored_xpoint_used_for_bottom(manifest):
    """_extract_boundary_points anchors 'bottom' on the stored lower X-point."""
    with h5py.File(_SLIM, "r") as hf:
        bl = hf["scan/0/_baseline"]
        if "recon_lcfs_ref" not in bl or "x_points" not in bl:
            pytest.skip("baseline lacks LCFS ref or x_points")
        ref = np.asarray(bl["recon_lcfs_ref"][()], dtype=float)
        R, Z = ref[:, 0], ref[:, 1]
        xp = np.asarray(bl["x_points"][()], dtype=float)
        R_axis = float(np.mean(R))
        Z_axis = float(np.mean(Z))
        pts, has_xpt = _extract_boundary_points(
            R, Z, R_axis, Z_axis, x_points=xp)
        # the lower X-point sits below the axis -> drives 'bottom'
        lower = xp[xp[:, 1] < Z_axis]
        if len(lower):
            assert has_xpt["lower"]
            np.testing.assert_allclose(
                pts["bottom"], lower[np.argmin(lower[:, 1])], atol=1e-9)


def test_chi2_filter_is_the_default_and_needs_no_dd(tmp_path):
    """Bouquet.filter() default = chi2 with the EFIT-residual sigma: writes
    passes_coil_filter for every draw from the archive alone and refreshes
    'selected'; the legacy rule stays selectable."""
    from bouquet.config import FilterConfig
    from bouquet import filter_coil_chi2
    assert FilterConfig().coil_filter == "chi2"
    assert FilterConfig(coil_filter="legacy").coil_filter == "legacy"
    with pytest.raises(ValueError):
        FilterConfig(coil_filter="bogus")
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    summ = filter_coil_chi2(work, None, apply=True)
    flags = read_filter_flags(work)
    for sv, drect in flags.items():
        assert summ[sv]["n_total"] == len(drect)
        for i, rec in drect.items():
            assert "passes_coil_filter" in rec
            assert rec["selected"] == (rec["passes_coil_filter"] and rec.get("passes_boundary_filter", True))



def test_chi2_filter_stamps_provenance_and_falls_back_loudly(tmp_path, monkeypatch):
    """Device detected from the golden (DIII-D) coil names -> model stamped on the
    scan group; with detection defeated and no sigma, the filter raises
    CoilSigmaUnavailable (Bouquet.filter turns that into a warning + legacy rule)."""
    import json, h5py, warnings
    from bouquet import filter_coil_chi2
    from bouquet.coil_spec import CoilSigmaUnavailable
    import bouquet.devices as dev
    work = str(tmp_path / "work.h5")
    shutil.copy(_SLIM, work)
    summ = filter_coil_chi2(work, None, apply=True)
    for sv, v in summ.items():
        assert v["sigma_model"]["kind"] == "device" and v["sigma_model"]["device"] == "DIII-D"
    with h5py.File(work) as hf:
        sk = list(hf["scan"].keys())[0]
        assert hf["scan"][sk].attrs["coil_filter"] == "chi2"
        assert json.loads(hf["scan"][sk].attrs["coil_sigma_model"])["device"] == "DIII-D"
    monkeypatch.setattr(dev, "detect_device", lambda names: None)
    with pytest.raises(CoilSigmaUnavailable):
        filter_coil_chi2(work, None, apply=False)
    # explicit floor/fraction rescues it without a device
    summ2 = filter_coil_chi2(work, None, apply=False, sigma={"floor": 325.0, "fraction": 0.0035})
    for sv in summ:
        assert summ2[sv]["sigma_model"]["kind"] == "floor_fraction" and summ2[sv]["n_total"] == summ[sv]["n_total"]
    # the looser rms-including-offset model never rejects more
    monkeypatch.undo()
    summ3 = filter_coil_chi2(work, None, apply=False, sigma="rms_incl_offset")
    for sv in summ:
        assert summ3[sv]["n_pass"] >= summ[sv]["n_pass"]


def test_infer_shot_from_geqdsk_name_and_header():
    from bouquet.run import Bouquet
    class S: geqdsk_path = "/x/y/g169510.03000"
    class C: source = S(); output_header = "D3D_169510_3000"
    b = Bouquet.__new__(Bouquet); b.config = C()
    assert b._infer_shot() == 169510
    class S2: pass
    class C2: source = S2(); output_header = "D3D_204441_4400"
    b.config = C2(); assert b._infer_shot() == 204441
    class C3: source = S2(); output_header = "nothing_here"
    b.config = C3(); assert b._infer_shot() is None
