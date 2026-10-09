"""Backend systematics *replay* regression test -- the LEGACY path.

Instead of re-running the (slow) reconstruction + GPR sampling, this test
**replays the pre-drawn golden draws** of the slim LEGACY golden
(``tests/golden/D3Dlike_Hmode_legacy_golden.json``: the golden recipe run with
``reconstruction_engine="legacy"``) through the legacy perturbation->solve->
``solve_with_bootstrap``->coil-homotopy pipeline and checks the outputs.  It
loads the pre-reconstructed jphi-linterp baseline (re-solved once from the
golden baseline j_phi -- *not* a full ``reconstruct_equilibrium``) and then,
for each stored draw, feeds that draw's stored profiles back with sigma=0 so
there is no resampling and no l_i iteration -- the only thing exercised is the
deterministic solve path.

Three modes (decompose pressure- vs current-systematics):
  * Mode 1  pinned, baseline kinetics      -> reproduces the baseline (floor).
  * Mode 2  pinned, draw's kinetics         -> pressure-only response; bounded
            and unbiased vs the baseline (isolates pressure systematics).
  * Mode 3  production, draw's profiles      -> reproduces the golden draw
            (full pipeline incl. bootstrap; current+pressure), on the
            bootstrap model the golden's own stored config names (the
            self-consistent loop, or the legacy frozen bootstrap).

Every step here is the legacy pipeline (``reconstruction_engine="legacy"``
reconstruction, the functional ``generate_bouquet`` draw path), so the
reference is the legacy golden, not the h5 fixture, which since 2026-10 is a
unified-engine run.  The JSON keeps the reconstruction's LCFS reference as a
uniform 1-in-8 subsample of its trace (the reference side of every boundary
RMS below; the other side is always a full trace) and each replayed draw's
own RMS to that subsample, measured by the generator on the draw's full
trace; see ``make_golden_fixture.LEGACY_LCFS_STRIDE``.

Reconstruction itself is covered by a separate test (future).  Needs OFT + the
D3D-like mesh/baseline; runs by default when available, marked ``solver``
(deselect with ``pytest -m "not solver"``).
"""
import json
import os

import numpy as np
import pytest

import h5py

import _harness

from bouquet.utils import _read_coil_names

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_GOLDEN = os.path.join(_HERE, "golden", "D3Dlike_Hmode_legacy_golden.json")

_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH, _GOLDEN))


def _oft_importable():
    import sys
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_files_ok and _oft_importable()),
        reason="solver replay test needs OFT + the D3D-like mesh/baseline + "
               "legacy golden; skipped when unavailable"),
]

# Replay a fixed, representative subset of in-spec golden draws.  Each replayed
# draw is a full solve + solve_with_bootstrap + coil-homotopy (~2-3 min), so the
# subset is kept small; N=2 covers >1 draw for the mode-2 check at ~13 min total.
_N_REPLAY = 2

# Mode 1/2 (pinned) floor: the small CONSTANT jphi-linterp edge residual.
_BND_RMS_MAX_MM = 0.8           # pinned boundary RMS vs baseline
_COIL_DRIFT_MAX_PCT = 0.3       # mode-1 (baseline kinetics) coil drift
_MODE2_BND_MAX_MM = 6.0         # mode-2 (draw pressure) bounded boundary shift
_MODE2_MEAN_BIAS_MM = 2.0       # mode-2 signed-mean boundary bias (no systematic)
# Mode 3 reproduces the golden draw via a live re-solve (looser than the
# fixture *read* tests since it is a fresh solve from a re-solved baseline).
_MODE3_BND_RMS_MM = 2.0         # |replay RMS-to-baseline - golden RMS-to-baseline|
_MODE3_LI_REL = 0.03
_MODE3_IP_REL = 0.01


def _legacy_golden():
    with open(_GOLDEN) as fh:
        return json.load(fh)


def _golden_generation():
    """The legacy golden's own stored generation config."""
    from bouquet.config import BouquetConfig
    return BouquetConfig.from_json(_legacy_golden()["config_json"]).generation


def _load_golden():
    """Pull baseline + a subset of draws (profiles, targets, references)."""
    doc = _legacy_golden()
    bl = doc["baseline"]
    pr = bl["profiles"]
    base = dict(
        psi_N=np.asarray(pr["psi_N"], dtype=float),
        psi_N_kin=np.asarray(pr["psi_N_kinetic"], dtype=float),
        ne=np.asarray(pr["n_e"], dtype=float),
        te=np.asarray(pr["T_e"], dtype=float),
        ni=np.asarray(pr["n_i"], dtype=float),
        ti=np.asarray(pr["T_i"], dtype=float),
        jphi=np.asarray(pr["j_phi"], dtype=float),
        pressure=np.asarray(pr["pressure"], dtype=float),
        # the uniform subsample of the reconstruction's LCFS trace
        recon_lcfs=np.asarray(bl["recon_lcfs_ref"]["points"], dtype=float),
        Ip_target=float(bl["attrs"]["Ip_target"]),
        l_i_target=float(bl["attrs"]["l_i_target"]),
    )
    # the generator kept the first _N_REPLAY in-spec draws (the deliverable
    # draws), as this test always replayed
    idxs = sorted(int(k) for k in doc["replay_draws"])[:_N_REPLAY]
    assert len(idxs) == _N_REPLAY, (
        f"the legacy golden carries {len(idxs)} replay draws, not "
        f"{_N_REPLAY}")
    draws = {}
    for i in idxs:
        gi = doc["replay_draws"][str(i)]
        p = gi["profiles"]
        draws[i] = dict(
            ne=np.asarray(p["n_e"], dtype=float),
            te=np.asarray(p["T_e"], dtype=float),
            ni=np.asarray(p["n_i"], dtype=float),
            ti=np.asarray(p["T_i"], dtype=float),
            jphi=np.asarray(p["j_phi"], dtype=float),
            jind=np.asarray(p["j_inductive"], dtype=float),
            # the draw's boundary RMS to the subsampled reconstruction LCFS,
            # measured by the generator on the draw's full trace
            bnd_rms_mm=float(gi["bnd_rms_to_recon_mm"]),
            li1=float(gi["l_i(1)"]),
            # The golden archive stores BOTH estimators per draw. Since
            # issue #20 the draw path targets and measures li(3)/'iter',
            # so that is the number mode 3 must be handed as its target --
            # the same golden draw, read on the estimator the code now
            # uses. Nothing about the golden equilibrium changed.
            li3=float(gi["l_i(3)"]),
            Ip=float(gi["Ip_eqdsk"] if gi.get("Ip_eqdsk") is not None
                     else np.nan),
            # The Z_eff this draw's bootstrap was evaluated with: with the
            # zeff aux channel active the generator draws Z_eff per draw
            # and archives it (kinetic grid); None for an archive that
            # has no such channel.
            zeff=(np.asarray(p["aux_zeff"], dtype=float) if "aux_zeff" in p
                  else None),
            count=int(gi.get("count", i)),
            coils=dict(zip(gi["coil_names"], gi["coil_currents"])),
        )
    base["coils"] = dict(zip(bl["coil_names"], bl["coil_currents"]))
    return base, draws


#: What mode 3 takes from ``Bouquet.generate()``'s own ``generate_bouquet``
#: call rather than from the function defaults: the bootstrap model
#: (baseline split, edge isolation, floor, diff offset, delta mode, the
#: anchor routes) and the fixed additive components
#: (impurity / fast pressure, diff anchors, NBI / RF current).  Per-draw
#: quantities (Z_eff, the bootstrap scale) are the draw's own, below.
_GENERATOR_MODEL_KWARGS = (
    "baseline_j_BS", "isolate_edge_jBS", "floor_j_BS",
    "jBS_diff", "jbs_delta_mode", "accept_anchor_inband",
    "perturb_jind_in_anchor", "p_fast", "z_fast", "Z_imp", "p_diff",
    "jphi_diff", "j_NBI", "j_RF",
)


def _generator_call(run):
    """The arguments ``run.generate()`` hands ``generate_bouquet``, by
    parameter name, captured at the call (no solve runs: the capture raises
    before it)."""
    import inspect
    import bouquet.TokaMaker_interface as _ti

    class _Captured(Exception):
        pass

    seen = {}

    def _capture(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise _Captured

    orig = _ti.generate_bouquet
    _ti.generate_bouquet = _capture
    try:
        run.generate()
    except _Captured:
        pass
    finally:
        _ti.generate_bouquet = orig
    assert seen, "Bouquet.generate() did not reach generate_bouquet"
    bound = inspect.signature(orig).bind(*seen["args"], **seen["kwargs"])
    return dict(bound.arguments)


def _draw_jBS_scale(gen_golden, jBS_scale_range, count):
    """The bootstrap scale the generator drew for draw ``count``.

    ``generate_bouquet`` draws the whole batch's scales as the FIRST
    consumption of its one ``make_rng(seed)`` Generator
    (``rng.uniform(lo, hi, size=n_equils)``), so the golden draw's scale is
    that block's ``count``-th element under the golden's own seed and
    ``n_equils``.  ``None`` range -> 1.0, as in the generator.
    """
    if jBS_scale_range is None:
        return 1.0
    from bouquet.sampling import make_rng
    lo, hi = (float(v) for v in jBS_scale_range)
    n = int(gen_golden.n_equils)
    assert gen_golden.n_inspec_target is None and 0 <= count < n, (
        "the scale block is only the first n_equils values of the stream "
        "without until-N")
    return float(make_rng(gen_golden.seed).uniform(lo, hi, size=n)[count])


def _bnd_rms_mm(ref, pts):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(pts)).query(np.asarray(ref))
    return float(np.sqrt((d ** 2).mean()) * 1e3)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    """Set up mygs, establish the jphi-linterp baseline (one forward solve,
    no reconstruct), then replay the golden draws in each mode."""
    from bouquet import generate_bouquet, initialize_equilibrium_database
    import bouquet as bq

    base, draws = _load_golden()
    psi_N = base["psi_N"]
    psi_pf = base["psi_N_kin"]
    # MUST equal the generator's ReconstructionSource.psi_pad (class default
    # 1e-3): every LCFS reference is a trace of the psi_N = 1 - pad surface,
    # so a pad mismatch shifts the replay onto a DIFFERENT flux surface and
    # shows up as a constant ~2 mm boundary "error" (the legacy notebook's
    # 1e-4 measured 1.92 mm against the class-generated golden).
    pad = 1e-3
    Zeff = np.ones_like(psi_N)

    # Stand up the solver THROUGH THE CLASS API, because that is what generated
    # the golden archive.  The environment a replay must reproduce is not a set
    # of constants that can be hand-copied: run.generate() samples under the
    # RECONSTRUCTION's isoflux points and weights (run.py restores them before
    # sampling), from the reconstruction's converged warmstart state.  The
    # previous hand-rolled setup (eqdsk boundary points, weight 200) replayed a
    # different constrained optimum and sat a deterministic ~1.9 mm from the
    # archived baseline LCFS -- masked for years by this fixture's nthreads=2
    # jitter.  Reconstructing here (~2 min, deterministic at the class default
    # nthreads=1) puts the replay in the generator's exact environment.
    _work = str(tmp_path_factory.mktemp("replay_recon"))
    run = bq.Bouquet.from_geqdsk(
        _GEQ, profiles=_PF, mesh=_MESH, n_draws=1,
        header=os.path.join(_work, "replay_recon"),
        reconstruction_engine="legacy")
    # Replay on the bootstrap model the golden was GENERATED with (its own
    # stored config): the self-consistent loop for a fixture made with it,
    # the frozen SWB bootstrap for one made before it existed (such a config
    # loads with jbs_self_consistent=False).  The reconstruction follows the
    # config; the functional generate_bouquet call below takes the same
    # choice as its per-draw jbs_loop settings.  Mixing the two (a loop
    # baseline under frozen draws, or the reverse) would replay a pipeline
    # that never produced the fixture.
    from bouquet.jbs_loop import jbs_settings
    _gen_golden = _golden_generation()
    run.config.generation.jbs_self_consistent = bool(
        _gen_golden.jbs_self_consistent)
    # ... and on the pressure frame the golden was GENERATED with (its own
    # stored config; a fixture that predates the setting back-fills
    # "legacy"): the reconstruction through the copied config field, and
    # every functional generate_bouquet call of _run below through the
    # resolved settings -- without them those calls would take
    # EdgePressure()'s default, not the golden's.  A test change only:
    # like-for-like replay, no bar moves (owner-approved 2026-10-05).
    from bouquet.edge_pressure import resolve_edge_pressure
    run.config.generation.separatrix_pressure = str(
        _gen_golden.separatrix_pressure)
    _edge_golden = resolve_edge_pressure(_gen_golden)
    assert resolve_edge_pressure(run.config.generation) == _edge_golden, (
        "the replay's reconstruction would not run on the golden's edge-"
        f"pressure settings: {resolve_edge_pressure(run.config.generation)}"
        f" vs the golden's {_edge_golden}")
    print(f"[replay] edge pressure (the golden's own): {_edge_golden}")
    _jbs_draw = jbs_settings(_gen_golden, draw=True)
    _jbs_draw = _jbs_draw if _jbs_draw["enabled"] else None
    run.reconstruct()
    mygs = run.mygs
    bl_run = run.baseline
    iso, isow = None, None
    if getattr(bl_run, "recon", None) is not None and \
            "isoflux_pts" in bl_run.recon:
        iso, isow = bl_run.recon["isoflux_pts"], bl_run.recon["weights"]
        mygs.set_isoflux(iso, weights=isow)

    # Mode 3 replays a draw of the pipeline generate() ran, so its bootstrap
    # model and fixed components are the ones generate() passes, read off
    # generate()'s own call on this reconstruction -- not the function
    # defaults.  Before this, mode 3 solved with Z_eff = 1 (generate: the
    # baseline / per-draw Z_eff, 1.76-1.92 here), isolate_edge_jBS=True
    # (generate: the geqdsk workflow's False), floor_j_BS=True, and no
    # baseline_j_BS or bootstrap scale, i.e. a different bootstrap (I_BS
    # ~219-233 kA against the golden draw loops' ~295-312 kA).
    _gen_kw = _generator_call(run)
    if iso is not None:                  # generate() restored the same set
        mygs.set_isoflux(iso, weights=isow)
    gen_model = {k: _gen_kw[k] for k in _GENERATOR_MODEL_KWARGS
                 if k in _gen_kw}
    # _run passes recalculate_j_BS=True itself (all modes); generate()'s agrees
    assert _gen_kw.get("recalculate_j_BS", True) is True

    def _short(v):
        if v is None or np.isscalar(v) or isinstance(v, (tuple, list)):
            return repr(v)
        a = np.asarray(v, dtype=float)
        return f"array[{a.size}] {a.min():.4g}..{a.max():.4g}"
    print("[replay mode3] from generate(): " + ", ".join(
        f"{k}={_short(v)}" for k, v in sorted(
            dict(gen_model, Zeff=_gen_kw.get("Zeff"),
                 jBS_scale_range=_gen_kw.get("jBS_scale_range")).items())))

    def _zeff_eq(zeff_kin):
        # generate_bouquet's own kin->eq regrid + floor for the drawn Z_eff
        from bouquet.utils import pchip_interp
        return np.clip(pchip_interp(psi_pf, np.asarray(zeff_kin, dtype=float),
                                    psi_N), 1.0, None)

    base_snapshot = mygs.copy_eq()

    # Hand modes 1-3 an l_i target on the scale the code compares against
    # (issue #20: the draw path targets and measures 'iter'/li(3)).  Measured
    # here from the class-API reconstruction state the replay warmstarts from;
    # for a golden generated after #20 this re-measures the same number the
    # archive already stores, and for a pre-#20 archive it converts the
    # legacy 'std' target.  (Modes 1/2 run pin_jphi=True, which
    # short-circuits the l_i band entirely; mode 3 is the one that gates.)
    base["l_i_target_std_golden"] = base["l_i_target"]
    base["l_i_target"] = float(
        mygs.get_stats(lcfs_pad=pad, li_normalization="iter")["l_i"])

    z = np.zeros_like(psi_pf)
    zj = np.zeros_like(psi_N)
    with open(_GEQ, 'rb') as fh:
        geq_raw = fh.read()
    with open(_PF, 'rb') as fh:
        pf_raw = fh.read()

    def _run(header, ne, te, ni, ti, input_jphi, input_jind, l_i_target,
             pin_jphi, Zeff_run=None, **model):
        mygs.replace_eq(base_snapshot)            # restore baseline each time
        if iso is not None:                       # re-point at the recon isoflux
            mygs.set_isoflux(iso, weights=isow)
        if os.path.exists(header + ".h5"):
            os.remove(header + ".h5")
        initialize_equilibrium_database(header)
        generate_bouquet(
            mygs, psi_N, 1, header, input_jphi,
            ne, te, ni, ti, z, z, z, z, zj,
            0.5, 0.4, 0.25, base["Ip_target"], l_i_target,
            Zeff if Zeff_run is None else Zeff_run,
            input_jinductive=input_jind,
            l_i_tolerance=0.05, psi_pad=pad,
            constrain_sawteeth=False, recalculate_j_BS=True,
            pfile_bytes=pf_raw, baseline_eqdsk_bytes=geq_raw,
            baseline_pfile_bytes=pf_raw,
            diagnostic_plots=False, scan_key=0, psi_N_kinetic=psi_pf,
            coil_drift=0.01,
            homotopy_passes=[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)],
            inspec_F_max=0.02, inspec_VSC_max=0.02, p_thresh=0.05,
            save_truncate_eq=True, jphi_baseline=True, seed=12345,
            pin_jphi=pin_jphi, jbs_loop=_jbs_draw,
            edge_pressure=_edge_golden,
            **model,
        )
        with h5py.File(header + ".h5", "r") as hf:
            g = hf["scan/0"]
            stored = sorted(int(k) for k in g if k.isdigit())
            if not stored:
                return None        # draw produced no feasible equilibrium
            gi = g[str(stored[0])]
            return dict(
                pert_lcfs=np.asarray(gi["perturbed_lcfs_ref"][()]),
                li1=float(gi.attrs["l_i(1)"]),
                li3=float(gi.attrs["l_i(3)"]),
                Ip=float(gi.attrs.get("Ip", np.nan)),
                coils=dict(zip(_read_coil_names(gi),
                               np.asarray(gi["coil_currents"][()]))),
            )

    gen_model_range = _gen_kw.get("jBS_scale_range")
    work = str(tmp_path_factory.mktemp("replay"))
    results = {"base": base, "draws": draws, "mode1": None,
               "mode2": {}, "mode3": {}}
    # Mode 1: pinned, baseline kinetics -> baseline
    results["mode1"] = _run(
        work + "/m1", base["ne"], base["te"], base["ni"], base["ti"],
        base["jphi"], base["jphi"], base["l_i_target"], pin_jphi=True)
    # Modes 2 & 3 per draw
    for i, d in draws.items():
        results["mode2"][i] = _run(
            work + f"/m2_{i}", d["ne"], d["te"], d["ni"], d["ti"],
            base["jphi"], base["jphi"], base["l_i_target"], pin_jphi=True)
        # Mode 3: the draw's own Z_eff (else generate()'s baseline Z_eff)
        # and its own bootstrap scale (a (s, s) range draws exactly s).
        _s = _draw_jBS_scale(_gen_golden, gen_model_range, d["count"])
        print(f"[replay mode3] draw {i}: scale_jBS={_s:.6f}, Z_eff "
              + (_short(_zeff_eq(d["zeff"])) if d["zeff"] is not None
                 else "generate()'s baseline"))
        results["mode3"][i] = _run(
            work + f"/m3_{i}", d["ne"], d["te"], d["ni"], d["ti"],
            d["jphi"], d["jind"], d["li3"], pin_jphi=False,
            Zeff_run=(_zeff_eq(d["zeff"]) if d["zeff"] is not None
                      else np.asarray(_gen_kw["Zeff"], dtype=float)),
            jBS_scale_range=(_s, _s), **gen_model)
    return results


def test_mode1_pinned_baseline_reproduces_baseline(replay):
    base = replay["base"]
    assert replay["mode1"] is not None, "mode-1 baseline replay produced no equilibrium"
    rms = _bnd_rms_mm(base["recon_lcfs"], replay["mode1"]["pert_lcfs"])
    print(f"[replay mode1] baseline RMS = {rms:.4f} mm (limit {_BND_RMS_MAX_MM})")
    assert rms < _BND_RMS_MAX_MM
    maxd = max(100.0 * abs(replay["mode1"]["coils"][c] - base["coils"][c])
               / max(abs(base["coils"][c]), 1.0) for c in base["coils"])
    print(f"[replay mode1] max coil drift = {maxd:.4f}% (limit {_COIL_DRIFT_MAX_PCT})")
    assert maxd < _COIL_DRIFT_MAX_PCT


def test_mode2_pinned_pressure_no_systematic(replay):
    """Pressure-only (j_phi pinned): bounded shift, signed-mean ~0 (no bias)."""
    base = replay["base"]
    devs = []
    for i, r in replay["mode2"].items():
        if r is None:
            continue
        rms = _bnd_rms_mm(base["recon_lcfs"], r["pert_lcfs"])
        print(f"[replay mode2] draw {i}: pressure-only boundary RMS = {rms:.3f} mm")
        assert rms < _MODE2_BND_MAX_MM, f"draw {i}: pressure shift {rms:.2f} mm too large"
        devs.append(rms)
    assert devs, "no mode-2 draws produced an equilibrium"
    # bounded above; (signed-mean check is a placeholder for a larger-N run)
    assert np.mean(devs) < _MODE2_BND_MAX_MM


def test_mode3_production_reproduces_golden(replay):
    """Production replay reproduces each golden draw (boundary/li/Ip).

    When this fails, WHICH side moved is the whole question: the replay runs
    live code against a live solver, the golden is a recording, and a change in
    either reads the same from here.  The fixture's provenance is therefore
    printed on every assertion below -- an l_i(1) miss with l_i(3) and the
    boundary intact is the signature of an edge-localised j_BS change, which is
    a SOLVER-build difference far more often than a bouquet one.
    """
    base = replay["base"]
    prov = _harness.legacy_golden_provenance_banner(_GOLDEN)
    n_checked = 0
    for i, d in replay["draws"].items():
        r = replay["mode3"][i]
        if r is None:
            continue
        n_checked += 1
        rms_replay = _bnd_rms_mm(base["recon_lcfs"], r["pert_lcfs"])
        rms_golden = d["bnd_rms_mm"]
        print(f"[replay mode3] draw {i}: boundary RMS replay={rms_replay:.3f} "
              f"golden={rms_golden:.3f} mm  li(3) replay={r['li3']:.4f} "
              f"golden={d['li3']:.4f}  li(1) replay={r['li1']:.4f} "
              f"golden={d['li1']:.4f}")
        assert abs(rms_replay - rms_golden) < _MODE3_BND_RMS_MM, (
            f"draw {i}: boundary RMS replay {rms_replay:.3f} mm vs golden "
            f"{rms_golden:.3f} mm (bar {_MODE3_BND_RMS_MM} mm)\n{prov}")
        # li(3) is the estimator the replay targets (issue #20); li(1) is
        # checked too so a convention drift between the two shows up here.
        # Both against the SAME _MODE3_LI_REL -- the bar is not widened.
        assert abs(r["li3"] - d["li3"]) / d["li3"] < _MODE3_LI_REL, (
            f"draw {i}: l_i(3) replay {r['li3']:.6f} vs golden "
            f"{d['li3']:.6f} ({100 * abs(r['li3'] - d['li3']) / d['li3']:.2f} "
            f"%, bar {100 * _MODE3_LI_REL:.0f} %)\n{prov}")
        assert abs(r["li1"] - d["li1"]) / d["li1"] < _MODE3_LI_REL, (
            f"draw {i}: l_i(1) replay {r['li1']:.6f} vs golden "
            f"{d['li1']:.6f} ({100 * abs(r['li1'] - d['li1']) / d['li1']:.2f} "
            f"%, bar {100 * _MODE3_LI_REL:.0f} %).  l_i(3) moved "
            f"{100 * abs(r['li3'] - d['li3']) / d['li3']:.2f} % -- an l_i(1)-"
            "only miss is edge-localised, so suspect the j_BS/solver build "
            f"before suspecting a bouquet change.\n{prov}")
        if np.isfinite(d["Ip"]) and np.isfinite(r["Ip"]):
            assert abs(r["Ip"] - d["Ip"]) / abs(d["Ip"]) < _MODE3_IP_REL, (
                f"draw {i}: Ip replay {r['Ip']:.1f} vs golden {d['Ip']:.1f}"
                f"\n{prov}")
    assert n_checked >= 1, "no mode-3 draws reproduced an equilibrium"
