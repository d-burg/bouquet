"""Golden-value regression tests for the bouquet pipeline.

These run against the slimmed, git-tracked h5 goldens plus their JSON
manifests of expected values, every test once per golden (``GOLDENS``):

* ``gfile``: ``tests/golden/D3Dlike_Hmode_golden_slim.h5`` +
  ``golden_manifest.json`` -- the unified-engine run of the synthetic g-file
  + IDA-layout profile fixtures (``tests/data/g_synthetic.geqdsk`` +
  ``tests/data/diiid_profs_synthetic.cdf``);
* ``ids``: ``tests/golden/D3Dlike_ids_golden_slim.h5`` +
  ``ids_golden_manifest.json`` -- the unified-engine IDA-hybrid run of the
  synthetic dd's 1.2 s slice (``tests/data/dd_synthetic.json.gz`` with the
  same profile file); what only that golden is, ``tests/test_golden_ids.py``.

Both are produced by ``tests/golden/regenerate_golden_run.py`` (the full run,
at the library defaults) and ``tests/golden/make_golden_fixture.py`` (the
slim fixture + manifests); see ``tests/golden/README.md``.

The tests:
  * assert every per-draw + baseline scalar in the fixture matches the
    manifest within tight tolerances (l_i, Ip, coil drifts, boundary
    deviation, coil currents, TokaMaker X-points);
  * exercise the postprocessing filters and selection helpers on the golden
    data (filter() under the stamped in-spec rule must reproduce the stored
    ``in_spec`` decision);
  * confirm the stored X-points drive ``_extract_boundary_points``.

To intentionally update the golden set: regenerate the full ``.h5``, run
``make_golden_fixture.py``, and commit the new fixture + manifest.  The
manifest git diff shows exactly which values moved.
"""
import json
import os
import shutil

import numpy as np
import pytest

import h5py

import _harness

import bouquet
from bouquet import (filter_boundaries,
                     select_indices, read_filter_flags, export_filtered)
from bouquet.plotting import _extract_boundary_points
from bouquet.utils import read_eqdsk_from_bytes  # noqa: F401  (import smoke)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")

#: golden -> (slim fixture, manifest); the names
#: ``tests/golden/make_golden_fixture.GOLDEN_FILES`` writes
GOLDENS = {
    "gfile": ("D3Dlike_Hmode_golden_slim.h5", "golden_manifest.json"),
    "ids": ("D3Dlike_ids_golden_slim.h5", "ids_golden_manifest.json"),
}


class Golden:
    """One h5 golden: its name, slim fixture path and manifest path."""

    def __init__(self, name):
        slim, man = GOLDENS[name]
        self.name = name
        self.slim = os.path.join(_GOLDEN_DIR, slim)
        self.manifest_path = os.path.join(_GOLDEN_DIR, man)

    def available(self):
        return (os.path.isfile(self.slim)
                and os.path.isfile(self.manifest_path))


@pytest.fixture(scope="module", params=sorted(GOLDENS))
def golden(request):
    g = Golden(request.param)
    if not g.available():
        pytest.skip(f"golden {g.name!r} not built; run "
                    "tests/golden/make_golden_fixture.py")
    return g


@pytest.fixture(scope="module")
def manifest(golden):
    with open(golden.manifest_path) as fh:
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


def test_the_goldens_are_the_builders_files():
    """The names here are the ones the builder writes."""
    import sys
    sys.path.insert(0, _GOLDEN_DIR)
    import make_golden_fixture as mgf
    for name, (slim, man) in GOLDENS.items():
        assert mgf.GOLDEN_FILES[name][:2] == (slim, man), name


def test_the_golden_is_its_recipe(golden, manifest):
    """Each golden is its regeneration recipe's run
    (``tests/golden/regenerate_golden_run.py``): the synthetic fixtures by
    basename, the unified engine, 20 attempts at seed 12345, one thread,
    everything else at the defaults of the code that made it."""
    import sys
    sys.path.insert(0, _GOLDEN_DIR)
    import regenerate_golden_run as rgr
    from bouquet.config import ImasSource, ReconstructionSource
    from bouquet.utils import load_config
    cfg = load_config(golden.slim, scan_key=0)
    gc = cfg.generation
    assert manifest.get("golden") == golden.name
    assert gc.reconstruction_engine == "unified"
    assert gc.n_equils == rgr.N_DRAWS and gc.seed == rgr.SEED
    assert gc.n_inspec_target is None
    assert cfg.solver.nthreads == rgr.NTHREADS
    assert cfg.solver.mesh_path == rgr.MESH
    if golden.name == "gfile":
        assert isinstance(cfg.source, ReconstructionSource)
        assert cfg.source.geqdsk_path == rgr.GFILE
        assert cfg.source.profiles_path == rgr.PROFILES
    else:
        assert isinstance(cfg.source, ImasSource)   # tests/test_golden_ids.py
        assert cfg.source.ids_path == rgr.DD


# ---------------------------------------------------------------------------
#  structural
# ---------------------------------------------------------------------------
def test_fixture_structure(golden, manifest):
    for bkey, sv, entry in _iter_scans(manifest):
        idxs = bouquet.utils.list_equilibrium_indices(golden.slim, scan_key=sv)
        assert idxs == entry["draw_indices"]
        assert len(idxs) == entry["n_draws"]


def test_no_pfile_blobs_in_fixture(golden):
    """The slim fixture drops p-file byte blobs (kept in the example artifact)."""
    bad = []
    with h5py.File(golden.slim, "r") as hf:
        def _check(name, obj):
            if isinstance(obj, h5py.Dataset) and (name.endswith(".pfile") or name == "pfile"):
                bad.append(name)
        hf.visititems(_check)
    assert not bad, f"slim fixture still has p-file blobs: {bad}"


def test_geqdsks_retained_per_manifest(golden, manifest):
    """geqdsks the manifest says were kept are present (and gzip-compressed)."""
    with h5py.File(golden.slim, "r") as hf:
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
def test_draw_scalars_match_manifest(golden, manifest, tol):
    prov = _harness.golden_provenance_banner(golden.slim)
    with h5py.File(golden.slim, "r") as hf:
        for bkey, sv, entry in _iter_scans(manifest):
            prefix = f"scan/{bkey}/" if sv is not None else ""
            for sidx, exp in entry["draws"].items():
                a = hf[f"{prefix}{sidx}"].attrs
                where = f"draw {sidx}\n{prov}"
                assert a["l_i(1)"] == pytest.approx(
                    exp["l_i(1)"], abs=tol["l_i_atol"]), where
                assert a["l_i(3)"] == pytest.approx(
                    exp["l_i(3)"], abs=tol["l_i_atol"]), where
                assert a["Ip"] == pytest.approx(
                    exp["Ip"], rel=tol["Ip_rtol"]), where
                assert a["max_F_drift_pct"] == pytest.approx(
                    exp["max_F_drift_pct"], abs=tol["drift_atol"]), where
                assert a["max_VSC_drift_pct"] == pytest.approx(
                    exp["max_VSC_drift_pct"], abs=tol["drift_atol"]), where
                assert bool(a["in_spec"]) == exp["in_spec"], where


def test_the_fixture_says_what_built_it(golden, manifest):
    """A golden with no provenance cannot be told apart from a current one.

    That is not a hypothetical: the 2026-08 fixture recorded its bouquet
    version and nothing about the solver, and when an OFT bootstrap change
    moved l_i(1) by 3.7 % the failure was indistinguishable from a bouquet
    regression until the two were bisected by hand.  The generator now stamps
    the OFT build (stated commit/branch where given, content digests always,
    plus the feature probes that separate the OFT lines) into both the fixture
    and the manifest; this asserts it survived.
    """
    prov = _harness.golden_provenance(golden.slim)
    assert prov, (
        "the slim fixture carries no prov_* attrs -- regenerate it with "
        "tests/golden/make_golden_fixture.py so the next staleness is "
        "diagnosable")
    for key in ("prov_created", "prov_bouquet_version", "prov_oft_available"):
        assert key in prov, f"missing {key}; have {sorted(prov)}"
    assert "provenance" in manifest, "the manifest lost its provenance block"
    oft = manifest["provenance"]["oft"]
    assert oft.get("available"), \
        "the fixture was built without OFT importable -- it cannot be a " \
        "record of a solver run"
    assert oft.get("sources_sha256") or oft.get("library_sha256"), \
        "no measured OFT identity was recorded (a stated commit alone can " \
        "be wrong about the build that actually ran)"


def test_the_fixture_is_a_self_consistent_bootstrap_run(golden):
    """The golden records the DEFAULT pipeline: the self-consistent bootstrap
    loop (jbs_self_consistent=True).  Its stored config says so, the baseline
    and every draw carry a converged loop record, and no loop record names a
    filesystem path (a public fixture).

    The g-file golden's baseline carries the schema-v3 ``jbs_loop`` block.
    The IDS route archives none on its baseline (``read_jbs_loop`` is None
    there): its reconstruction's loop verdict is in the baseline's engine
    record (``engine_json``: ``converged`` and ``loop_converged``), which is
    read instead.  Every draw carries the block on both routes."""
    from bouquet.utils import load_config
    from bouquet.schema import read_jbs_loop
    import sys
    sys.path.insert(0, _GOLDEN_DIR)
    import make_golden_fixture as mgf
    cfg = load_config(golden.slim, scan_key=0)
    assert cfg.generation.jbs_self_consistent is True
    with h5py.File(golden.slim, "r") as hf:
        g = hf["scan/0"]
        bl = read_jbs_loop(g["_baseline"])
        if golden.name == "ids":
            assert bl is None, "the IDS baseline now archives a jbs_loop " \
                "block: check it here as on the g-file golden"
            raw = g["_baseline/engine_json"][()]
            eng = json.loads(raw.decode() if isinstance(raw, bytes)
                             else bytes(raw).decode())
            assert eng["converged"] and eng["loop_converged"], \
                {k: eng.get(k) for k in ("converged", "loop_converged")}
        else:
            assert bl is not None and bl["converged"], bl
        draws = [k for k in g if k.isdigit()]
        assert draws
        for k in draws:
            rec = read_jbs_loop(g[k])
            assert rec is not None, f"draw {k} carries no jbs_loop block"
            assert bool(g[k].attrs["jbs_converged"]), f"draw {k}: {rec}"
    mgf.assert_no_filesystem_paths(golden.slim)


def test_the_fixture_archives_the_input_current(golden, manifest):
    """The systematics replay feeds ``_baseline/jphi_total`` back as the input of
    its baseline solve, so the fixture must archive the INPUT current
    (``regenerate_golden_run.py``, ``store_achieved_jphi=False``).  A fixture
    regenerated through plain ``Bouquet.generate()`` archives the ACHIEVED
    current instead and fails mode 1 on the coils (tests/golden/README.md,
    "mode-1 coil drift").  The archive stamp, the fixture's provenance and the
    manifest must all say "input"."""
    with h5py.File(golden.slim, "r") as hf:
        assert hf.attrs.get("golden_jphi_archival") == "input", \
            dict(hf.attrs).get("golden_jphi_archival")
    prov = _harness.golden_provenance(golden.slim)
    assert prov.get("prov_generator_args_jphi_archival") == "input", prov
    assert manifest["provenance"]["generator_args"].get(
        "jphi_archival") == "input", manifest["provenance"]["generator_args"]


def test_coil_currents_match_manifest(golden, manifest, tol):
    with h5py.File(golden.slim, "r") as hf:
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


def test_xpoints_match_manifest(golden, manifest, tol):
    with h5py.File(golden.slim, "r") as hf:
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


def test_boundary_deviations_match_manifest(golden, manifest, tol):
    """filter_boundaries (apply=False) must reproduce the manifest RMS/max."""
    for bkey, sv, entry in _iter_scans(manifest):
        summ, _ = filter_boundaries(golden.slim, scan_key=sv, apply=False,
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
def _iter_stored_geqdsks(golden, manifest):
    """Yield (raw_bytes, expected_Ip, label) for every retained geqdsk."""
    with h5py.File(golden.slim, "r") as hf:
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


def test_geqdsk_parse_and_ip(golden, manifest, tol):
    """Every retained geqdsk parses and its Ip matches the manifest.

    Validates the full geqdsk write->store->read path on real (coarse)
    g-files, not just the cached scalar attrs.
    """
    from bouquet import read_geqdsk
    from bouquet.utils import read_eqdsk_from_bytes
    n = 0
    for raw, exp_ip, label in _iter_stored_geqdsks(golden, manifest):
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


def test_geqdsk_separatrix_is_coarse(golden, manifest):
    """Document the known geqdsk limitation: the stored boundary is far
    coarser than the high-res LCFS reference we keep alongside it.

    This is *why* the pipeline stores ``perturbed_lcfs_ref`` separately;
    the test guards that the coarse g-file boundary and the fine
    trace-surf reference stay distinct sources.
    """
    from bouquet import read_geqdsk
    from bouquet.utils import read_eqdsk_from_bytes
    checked = 0
    with h5py.File(golden.slim, "r") as hf:
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
def _filter_under_the_stamped_rule(golden, tmp_path):
    """A copy of the golden filtered as ``Bouquet.filter()`` filters it, on
    the golden's own stored config: the configured coil filter (chi2 by
    default) and the boundary cut, then the in-spec rule the archive was
    stamped with.  Returns ``(work_path, filter() summary)``."""
    from bouquet.utils import load_config
    work = str(tmp_path / "work.h5")
    shutil.copy(golden.slim, work)
    cfg = load_config(work, scan_key=0)
    cfg.output_header = work[:-len(".h5")]
    return work, bouquet.Bouquet(cfg).filter()


def test_filter_under_the_stamped_rule_reproduces_in_spec(golden, tmp_path,
                                                          manifest):
    """The archived ``in_spec`` is the in-spec rule's verdict:
    ``generate_bouquet`` judged every draw by the rule before archiving it
    (the configured coil test AND the boundary cut AND an engine draw's band
    flag; ``in_spec_source = "inspec_rule"``).  ``filter()`` under the
    stamped rule reproduces it draw for draw, and ``refilter_inspec`` under
    the stamped bands leaves the selection unchanged.  The legacy +/-2 %
    drift verdict is kept as ``coil_drift_in_spec``; it is not the rule and
    is not compared here."""
    from bouquet.filtering import read_inspec_rule, refilter_inspec
    rule, _ = read_inspec_rule(golden.slim, scan_key="0")
    assert rule is not None
    work, sel = _filter_under_the_stamped_rule(golden, tmp_path)
    assert sel["inspec"]["rule"] == rule
    assert sel["inspec"]["refiltered"] is False     # the stamp is this rule
    flags = read_filter_flags(work)
    for bkey, sv, entry in _iter_scans(manifest):
        drect = flags[sv]
        for i, rec in drect.items():
            assert rec["in_spec_source"] == "inspec_rule", (sv, i, rec)
            assert rec["selected"] == rec["in_spec"], (sv, i, rec)
        assert sum(bool(r["in_spec"]) for r in drect.values()) == \
            entry["n_in_spec"]
        before = select_indices(work, scan_key=sv, selection="selected")
        refilter_inspec(work, read_inspec_rule(work, scan_key=sv)[1],
                        scan_key=sv, verbose=False)
        assert select_indices(work, scan_key=sv, selection="selected") == before


def test_selection_partition(golden, tmp_path, manifest):
    work, _ = _filter_under_the_stamped_rule(golden, tmp_path)
    for bkey, sv, entry in _iter_scans(manifest):
        alli = select_indices(work, scan_key=sv, selection="all")
        sel = select_indices(work, scan_key=sv, selection="selected")
        exc = select_indices(work, scan_key=sv, selection="excluded")
        assert sorted(sel + exc) == sorted(alli)        # partition
        assert set(sel).isdisjoint(exc)
        assert len(sel) == entry["n_in_spec"]


def test_selection_unfiltered_defaults(golden):
    """Before any filter is applied, 'selected' == all, 'excluded' == none."""
    assert select_indices(golden.slim, scan_key="0", selection="selected") == \
        select_indices(golden.slim, scan_key="0", selection="all")
    assert select_indices(golden.slim, scan_key="0", selection="excluded") == []


def test_export_filtered_keeps_selected(golden, tmp_path, manifest):
    work, _ = _filter_under_the_stamped_rule(golden, tmp_path)
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


def test_boundary_filter_cut(golden, tmp_path):
    """A boundary RMS cut marks the expected draws and is reversible."""
    work = str(tmp_path / "work.h5")
    shutil.copy(golden.slim, work)
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
def test_stored_xpoint_used_for_bottom(golden, manifest):
    """_extract_boundary_points anchors 'bottom' on the stored lower X-point."""
    with h5py.File(golden.slim, "r") as hf:
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


def test_chi2_filter_is_the_default_and_needs_no_dd(golden, tmp_path):
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
    shutil.copy(golden.slim, work)
    summ = filter_coil_chi2(work, None, apply=True)
    flags = read_filter_flags(work)
    for sv, drect in flags.items():
        assert summ[sv]["n_total"] == len(drect)
        for i, rec in drect.items():
            assert "passes_coil_filter" in rec
            # every flag of filtering._FILTER_FLAGS present is ANDed in; the
            # engine's post-hoc band rides along on a unified-engine draw
            assert rec["selected"] == (
                rec["passes_coil_filter"]
                and rec.get("passes_boundary_filter", True)
                and rec.get("passes_draw_band", True))



def test_chi2_filter_stamps_provenance_and_falls_back_loudly(golden, tmp_path, monkeypatch):
    """Device detected from the golden (DIII-D) coil names -> model stamped on the
    scan group; with detection defeated and no sigma, the filter raises
    CoilSigmaUnavailable (Bouquet.filter turns that into a warning + legacy rule)."""
    import json, h5py, warnings
    from bouquet import filter_coil_chi2
    from bouquet.coil_spec import CoilSigmaUnavailable
    import bouquet.devices as dev
    work = str(tmp_path / "work.h5")
    shutil.copy(golden.slim, work)
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


def test_daq_era_is_stated_never_scraped_from_a_name():
    """The acquisition era picks the coil sigma FLOOR, i.e. an acceptance
    criterion, so it may only come from something the user stated.

    Regression: the old ``_infer_shot`` matched ANY six consecutive digits in the
    run header, so a run named after a mesh resolution or a date silently bought
    a 2.5x looser floor on every coil.
    """
    from bouquet.devices import D3D_DAQ_UPGRADE_PULSE, era_for_pulse, get_device
    from bouquet.run import Bouquet

    class Filt:
        coil_daq_era = None
    class S: pass
    class C:
        source = S(); output_header = "mesh_262144_conv"; device = "DIII-D"
        filtering = Filt()

    b = Bouquet.__new__(Bouquet); b.config = C()
    # a six-digit number in the header (or in a mesh/geqdsk path) buys nothing
    assert b._coil_daq_era() is None
    C.output_header = "run_20260904_case"
    assert b._coil_daq_era() is None
    S.geqdsk_path = "/x/y/g262144.03000"
    assert b._coil_daq_era() is None

    # the explicit knob wins
    Filt.coil_daq_era = "pre2014"
    assert b._coil_daq_era() == "pre2014"
    Filt.coil_daq_era = None

    # ... and so does an explicit pulse field on the source, via the era bands
    S.pulse = D3D_DAQ_UPGRADE_PULSE + 10
    assert b._coil_daq_era() == "modern"
    S.pulse = D3D_DAQ_UPGRADE_PULSE - 10
    assert b._coil_daq_era() == "pre2014"

    spec = get_device("DIII-D")
    assert era_for_pulse(spec, None) is None
    assert era_for_pulse(spec, "not-a-pulse") is None


def test_a_six_digit_mesh_name_cannot_change_the_coil_sigma():
    """End-to-end companion to the above: the same archive, judged with a header
    that happens to contain six digits, must get IDENTICAL sigmas."""
    import warnings
    from bouquet.coil_spec import resolve_coil_sigma
    base = {**{f"F{i}{s}": 1e5 for i in range(1, 10) for s in "AB"},
            "ECOILA": 2e4, "ECOILB": 2e4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_default, m_default = resolve_coil_sigma(base, device="DIII-D")
    s_modern, _ = resolve_coil_sigma(base, device="DIII-D", era="modern")
    s_pre, _ = resolve_coil_sigma(base, device="DIII-D", era="pre2014")
    assert s_default == s_modern                 # unknown era -> tightest floor
    assert s_pre["F1A"] > s_modern["F1A"]         # the eras really do differ
    assert m_default["era_given"] is None
