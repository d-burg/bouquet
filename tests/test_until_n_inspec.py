"""until-N-in-spec: draw until N draws pass the filters, not exactly N draws.

The feature's whole correctness claim is an IDENTITY: a run that stops after
counting N in-spec draws must show N ``selected`` draws once ``.filter()``
runs over the archive it wrote.  That holds only because the in-loop verdict
and the postprocess verdict are the SAME code
(``filtering.passes_all_filters`` composing ``passes_coil_spec`` /
``passes_boundary_spec`` over ``boundary_deviation_mm``), so most of what is
tested here is that identity and the ways it could quietly rot:

  * the LCFS metric's DIRECTION.  The in-loop ``[bnd-diag]`` print queries the
    baseline tree with the perturbed points; the filter queries the perturbed
    tree with the baseline points.  Those disagree on the same contour pair.
    Counting with the wrong one would deliver 20 "in-spec" draws that filter
    down to 18, which is exactly the bug this feature exists to avoid.
  * the THRESHOLDS.  The loop must read them from the same FilterConfig
    ``.filter()`` later cuts on.
  * the RNG STREAM.  With the feature off nothing may change, so the block
    ``jBS_scales`` draw stays one ``size=n_equils`` call and the lazy
    extension is only reachable past that block.

Solve-free throughout: the predicate takes raw numbers and contours, and the
archive fixtures are hand-built HDF5.
"""
import inspect
import os
import shutil

import h5py
import numpy as np
import pytest

from bouquet.config import (BouquetConfig, GenerationConfig, ReconstructionSource,
                            SolverConfig)
from bouquet.filtering import (boundary_deviation_mm, filter_boundaries,
                               filter_coil_currents, passes_all_filters,
                               passes_boundary_spec, passes_coil_spec,
                               select_indices)


# --------------------------------------------------------------------------
#  contour helpers
# --------------------------------------------------------------------------
def _circle(n=360, r=1.0, cx=1.7, cz=0.0):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(th), cz + r * np.sin(th)])


def _mini_config(**gen):
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g.geqdsk",
                                    profiles_path="p.peqdsk"),
        solver=SolverConfig(mesh_path="mesh.h5"),
        output_header="until_n_test",
        generation=GenerationConfig(**gen),
    )


# ==========================================================================
#  1. the metric
# ==========================================================================
def test_uniform_radial_offset_gives_that_offset_in_mm():
    """A concentric radius change is a pure 1 mm shift for every point."""
    rms, mx = boundary_deviation_mm(_circle(r=1.0), _circle(r=1.001))
    assert rms == pytest.approx(1.0, abs=1e-3)
    assert mx == pytest.approx(1.0, abs=1e-3)


def test_direction_is_load_bearing_and_is_the_filter_s_direction():
    """Swapping the arguments must change the answer -- else the choice of
    direction is untested and a refactor could flip it unnoticed.

    A dense baseline against a sparse perturbed contour: querying the sparse
    tree with dense points hits the sampling gaps (large max), while the
    reverse finds every sparse point close to some dense point (small max).
    """
    dense = _circle(n=720)
    sparse = _circle(n=8)
    fwd = boundary_deviation_mm(dense, sparse)       # filter direction
    rev = boundary_deviation_mm(sparse, dense)
    assert fwd[1] > 10.0 * rev[1], (fwd, rev)

    # ...and the filter direction is the one that builds the tree on the
    # PERTURBED contour and queries it with the BASELINE points.
    from scipy.spatial import cKDTree
    devs, _ = cKDTree(sparse).query(dense)
    assert fwd[0] == pytest.approx(float(np.sqrt(np.mean(devs ** 2)) * 1e3))


@pytest.mark.parametrize("bl,pt", [
    (None, _circle()), (_circle(), None), (None, None),
    (_circle()[:1], _circle()), (_circle(), _circle()[:1]),
])
def test_degenerate_contours_give_no_verdict_not_a_pass(bl, pt):
    rms, mx = boundary_deviation_mm(bl, pt)
    assert np.isnan(rms) and np.isnan(mx)
    # and a NaN deviation must FAIL a supplied bound rather than pass it
    assert passes_boundary_spec(rms, mx, rms_max_mm=5.0) is False


# ==========================================================================
#  2. the channel predicates
# ==========================================================================
def test_coil_spec_is_inclusive_at_the_threshold():
    assert passes_coil_spec(2.0, 2.0, 2.0, 2.0) is True
    assert passes_coil_spec(2.001, 1.0, 2.0, 2.0) is False
    assert passes_coil_spec(1.0, 2.001, 2.0, 2.0) is False


def test_coil_chi2_is_inclusive_at_the_threshold_and_fails_the_unjudgeable():
    """The chi2 sibling of the test above -- and the expression
    ``filter_coil_chi2`` itself cuts on, so this pins both sides at once."""
    from bouquet.filtering import passes_coil_chi2
    assert passes_coil_chi2(6.1, 6.3, 6.1, 6.3) is True
    assert passes_coil_chi2(6.11, 1.0, 6.1, 6.3) is False
    assert passes_coil_chi2(1.0, 6.31, 6.1, 6.3) is False
    assert passes_coil_chi2(1.0, 99.0, 6.1, None) is True     # guard disabled
    # unjudgeable (no usable coil) is NaN and must FAIL, never pass
    assert passes_coil_chi2(np.nan, np.nan, 6.1, 6.3) is False
    assert passes_coil_chi2(1.0, np.nan, 6.1, 6.3) is False


@pytest.mark.parametrize("F,V", [(np.nan, 1.0), (1.0, np.nan)])
def test_unmeasured_coil_drift_is_not_silently_in_spec(F, V):
    assert passes_coil_spec(F, V, 2.0, 2.0) is False


def test_missing_threshold_fails_rather_than_waves_through():
    """An archive without ``inspec_F_max`` yields a NaN threshold; that draw
    must not be counted as in spec."""
    assert passes_coil_spec(1.0, 1.0, np.nan, 2.0) is False


def test_boundary_bounds_left_none_are_not_applied():
    # report-only mode: filter_boundaries' documented no-threshold default
    assert passes_boundary_spec(1e9, 1e9) is True
    assert passes_boundary_spec(np.nan, np.nan) is True
    # a supplied bound is applied to its own channel only
    assert passes_boundary_spec(4.0, 999.0, rms_max_mm=5.0) is True
    assert passes_boundary_spec(4.0, 999.0, rms_max_mm=5.0,
                                max_max_mm=100.0) is False


def test_passes_all_filters_names_every_failing_channel():
    bl, pt = _circle(r=1.0), _circle(r=1.010)          # 10 mm out
    ok, rms, mx, why = passes_all_filters(9.9, 0.1, bl, pt, 2.0, 2.0,
                                          rms_max_mm=5.0)
    assert ok is False and set(why) == {"coil", "boundary"}
    assert rms == pytest.approx(10.0, abs=1e-2)

    ok, _, _, why = passes_all_filters(0.1, 0.1, bl, pt, 2.0, 2.0,
                                       rms_max_mm=5.0)
    assert ok is False and why == ("boundary",)

    ok, _, _, why = passes_all_filters(9.9, 0.1, bl, _circle(r=1.0),
                                       2.0, 2.0, rms_max_mm=5.0)
    assert ok is False and why == ("coil",)

    ok, _, _, why = passes_all_filters(0.1, 0.1, bl, _circle(r=1.0),
                                       2.0, 2.0, rms_max_mm=5.0)
    assert ok is True and why == ()


# ==========================================================================
#  3. THE identity: in-loop count == what .filter() marks selected
# ==========================================================================
#  A hand-built archive of draws that straddle both thresholds in every
#  combination, run through the real postprocess filters, compared against
#  the predicate the generation loop counts with.
_DRAWS = [
    # (F_pct, V_pct, radius_offset_m)   -> pass/fail per channel
    (0.5, 0.5, 0.000),      # both pass
    (0.5, 0.5, 0.010),      # boundary fails (10 mm)
    (9.0, 0.5, 0.000),      # coil F fails
    (0.5, 9.0, 0.000),      # coil VSC fails
    (9.0, 9.0, 0.010),      # both fail
    (2.0, 2.0, 0.005),      # exactly on all three thresholds -> passes
]
_F_MAX_PCT, _V_MAX_PCT, _RMS_MAX_MM = 2.0, 2.0, 5.0


def _build_archive(path, scan_key=0):
    bl = _circle()
    with h5py.File(path, "w") as hf:
        g = hf.create_group(f"scan/{scan_key}/_baseline")
        g.create_dataset("recon_lcfs_ref", data=bl)
        for i, (F, V, dr) in enumerate(_DRAWS):
            d = hf.create_group(f"scan/{scan_key}/{i}")
            d.create_dataset("perturbed_lcfs_ref", data=_circle(r=1.0 + dr))
            d.attrs["max_F_drift_pct"] = float(F)
            d.attrs["max_VSC_drift_pct"] = float(V)
            d.attrs["inspec_F_max"] = _F_MAX_PCT / 100.0
            d.attrs["inspec_VSC_max"] = _V_MAX_PCT / 100.0
    return bl


def test_inloop_verdict_matches_the_postprocess_filters_draw_by_draw(tmp_path):
    h5 = str(tmp_path / "run.h5")
    bl = _build_archive(h5)

    filter_coil_currents(h5, scan_key=0, F_max_pct=_F_MAX_PCT,
                         VSC_max_pct=_V_MAX_PCT, apply=True, plot=False)
    filter_boundaries(h5, scan_key=0, rms_max_mm=_RMS_MAX_MM,
                      apply=True, plot=False)
    post = set(select_indices(h5, scan_key=0, selection="selected"))

    inloop = {i for i, (F, V, dr) in enumerate(_DRAWS)
              if passes_all_filters(F, V, bl, _circle(r=1.0 + dr),
                                    _F_MAX_PCT, _V_MAX_PCT,
                                    rms_max_mm=_RMS_MAX_MM)[0]}
    assert inloop == post, f"in-loop {sorted(inloop)} != selected {sorted(post)}"
    # the fixture must actually exercise both outcomes, or the equality is vacuous
    assert 0 < len(post) < len(_DRAWS)


def test_the_identity_would_break_under_the_wrong_tree_direction(tmp_path):
    """Negative control for the test above: counting with the bnd-diag
    direction on a sampling-mismatched pair gives a DIFFERENT verdict, which
    is why the shared helper (and not a second implementation) is used."""
    dense, sparse = _circle(n=720), _circle(n=8, r=1.0005)
    filt = boundary_deviation_mm(dense, sparse)              # filter direction
    diag = boundary_deviation_mm(sparse, dense)              # bnd-diag direction
    assert passes_boundary_spec(*filt, rms_max_mm=5.0) is False
    assert passes_boundary_spec(*diag, rms_max_mm=5.0) is True


_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "golden", "D3Dlike_Hmode_golden_slim.h5")


@pytest.mark.skipif(not os.path.isfile(_GOLDEN),
                    reason="golden fixture not built")
def test_the_identity_holds_on_the_real_golden_archive(tmp_path):
    """Same identity, on 20 REAL draws with real ~10k-point LCFS contours --
    where sampling density, contour closure and near-threshold drifts are all
    as they actually come out of a run, not as a circle fixture stages them."""
    h5 = str(tmp_path / "golden.h5")
    shutil.copy(_GOLDEN, h5)

    rms_max_mm = 5.0
    filter_coil_currents(h5, scan_key=0, apply=True, plot=False)
    filter_boundaries(h5, scan_key=0, rms_max_mm=rms_max_mm,
                      apply=True, plot=False)
    post = set(select_indices(h5, scan_key=0, selection="selected"))

    inloop = set()
    with h5py.File(h5, "r") as hf:
        g = hf["scan/0"]
        bl = np.asarray(g["_baseline"]["recon_lcfs_ref"][()], dtype=float)
        for k in sorted(int(x) for x in g if str(x).isdigit()):
            d = g[str(k)]
            ok, _, _, _ = passes_all_filters(
                float(d.attrs["max_F_drift_pct"]),
                float(d.attrs["max_VSC_drift_pct"]),
                bl, np.asarray(d["perturbed_lcfs_ref"][()], dtype=float),
                float(d.attrs["inspec_F_max"]) * 100.0,
                float(d.attrs["inspec_VSC_max"]) * 100.0,
                rms_max_mm=rms_max_mm)
            if ok:
                inloop.add(k)

    assert inloop == post, f"in-loop {sorted(inloop)} != selected {sorted(post)}"
    assert len(post) >= 1


# ==========================================================================
#  4. config surface
# ==========================================================================
def test_default_is_off_and_unchanged():
    gc = _mini_config().generation
    assert gc.n_inspec_target is None and gc.max_total_draws is None


def test_target_and_cap_round_trip_through_json():
    c = _mini_config(n_inspec_target=7, max_total_draws=33)
    g = BouquetConfig.from_dict(c.to_dict()).generation
    assert (g.n_inspec_target, g.max_total_draws) == (7, 33)


@pytest.mark.parametrize("kw,frag", [
    (dict(n_inspec_target=0), "must be >= 1"),
    (dict(n_inspec_target=-1), "must be >= 1"),
    (dict(n_inspec_target=20, max_total_draws=10), "below n_inspec_target"),
    (dict(max_total_draws=50), "only applies with"),
])
def test_incoherent_until_n_settings_are_rejected_at_construction(kw, frag):
    with pytest.raises(ValueError, match=frag):
        _mini_config(**kw)


# ==========================================================================
#  5. parallel launchers: a target is honoured through a shared ledger
# ==========================================================================
def test_run_shard_refuses_a_target_without_a_ledger(tmp_path):
    """N workers each chasing the target alone would deliver N*target draws,
    so a bare run_shard (no shared ledger) must refuse rather than ignore."""
    from bouquet.parallel import run_shard
    cfg = _mini_config(n_inspec_target=5)
    with pytest.raises(ValueError, match="shared ledger"):
        run_shard(cfg, 0, 2, n_equils_total=4, seed_base=1,
                  out_header=str(tmp_path / "h"), scan_key=0,
                  threads_per_worker=1)


def test_emit_slurm_script_carries_the_shared_ledger(tmp_path):
    """The SLURM bundle records the target and the ledger file, and submit.sh
    truncates that file before the array starts."""
    import json
    from bouquet.parallel import emit_slurm_script
    cfg = _mini_config(n_inspec_target=5)
    cfg.output_header = str(tmp_path / "run")
    out = emit_slurm_script(cfg, n_workers=2, seed=1, threads_per_worker=1,
                            out_dir=str(tmp_path), job_name="j")
    b = json.load(open(out["bundle"]))
    assert b["n_inspec_target"] == 5
    assert b["ledger"].endswith("run_inspec.ledger")
    submit = open(out["submit"]).read()
    assert f': > "{b["ledger"]}"' in submit
    assert submit.index(': > "') < submit.index("sbatch --parsable")


# ==========================================================================
#  6. structural guards on the generation loop (solve-free)
# ==========================================================================
def _gb_source():
    from bouquet.TokaMaker_interface import generate_bouquet
    return inspect.getsource(generate_bouquet)


def test_generate_bouquet_defaults_the_feature_off():
    from bouquet.TokaMaker_interface import generate_bouquet
    sig = inspect.signature(generate_bouquet)
    for p in ("n_inspec_target", "max_total_draws", "inspec_rms_max_mm",
              "inspec_max_max_mm"):
        assert sig.parameters[p].default is None, p


def test_the_loop_counts_with_the_shared_predicate():
    """Not a re-implementation: the loop routes through _until_n_verdict,
    whose whole body is a passes_all_filters call -- if either link is ever
    inlined, the in-loop and postprocess verdicts can drift apart silently."""
    import bouquet.TokaMaker_interface as tmi
    assert "_until_n_verdict(" in _gb_source()
    helper = inspect.getsource(tmi._until_n_verdict)
    assert "passes_all_filters(" in helper


def test_the_jbs_block_draw_is_untouched_when_the_feature_is_off():
    """The rng-stream guarantee: one ``size=n_equils`` block draw as before,
    and the lazy extension reachable only past that block (``i >= len``)."""
    src = _gb_source()
    assert src.count("rng.uniform(lo, hi, size=n_equils)") == 1
    assert "while i >= len(jBS_scales):" in src
    # the loop must go through the accessor, not index the array directly
    assert "scale_jBS = _jBS_scale_for(count)" in src
    assert "float(jBS_scales[count])" not in src


def test_a_boundary_bounded_target_refuses_to_run_without_a_reference():
    """BNDDIAG=0 leaves recon_lcfs_ref None, which would NaN every boundary
    verdict -- the loop could then never terminate before the attempt cap.
    That must fail up front, not after N solves."""
    src = _gb_source()
    assert "_ref_arr.ndim != 2 or len(_ref_arr) < 2" in src
    assert "BNDDIAG=0" in src
    # and the check must precede the draw loop, not sit inside it
    assert src.index("BNDDIAG=0 disables") < src.index("for count in eq_iter:")


def test_a_missing_perturbed_trace_undercounts_rather_than_overcounts():
    """The one place the in-loop and postprocess verdicts can disagree: a
    failed high-res trace. The loop must call that out of spec (postprocess
    falls back to the coarse eqdsk contour and may pass it), so the run
    delivers at least N selected draws, never fewer."""
    ok, rms, mx, why = passes_all_filters(0.1, 0.1, _circle(), None,
                                          2.0, 2.0, rms_max_mm=5.0)
    assert ok is False and why == ("boundary",)
    assert np.isnan(rms) and np.isnan(mx)
    assert "never overcount" in _gb_source()


def test_missing_the_target_is_warned_not_swallowed():
    """Hitting the attempt cap means the requested ensemble was NOT delivered;
    it must not look like a completed run -- and the warning must be emitted
    OUTSIDE the quiet-mode output capture, which swallows stdout AND stderr
    into generation_log (where a failure signal may not live alone)."""
    src = _gb_source()
    assert "RuntimeWarning" in src
    assert "_inspec_hit_target" in src
    from bouquet.run import Bouquet
    import inspect as _i
    gen = _i.getsource(Bouquet.generate)
    assert "until_n_delivered" in gen
    # the re-derivation must sit after the capture block closes
    assert gen.index('_cap["text"]') < gen.index("until_n_delivered(")


def test_coil_channel_failfast_precedes_the_loop():
    """NaN coil drifts (SKIP_HARD=1 / coil_drift=None) fail every verdict by
    design; with a target set that is a livelock to the attempt cap, so it
    must raise before the first solve, like the boundary guard."""
    src = _gb_source()
    assert "needs a measurable coil channel" in src
    assert src.index("needs a measurable coil channel") \
        < src.index("for count in eq_iter:")


def test_archived_in_spec_flag_uses_the_shared_predicate():
    """The pre-existing in-loop [in-spec] print is the one other place the
    counted-vs-selected identity could rot; it must route through
    passes_coil_spec, not re-implement it."""
    assert "_in_spec = passes_coil_spec(" in _gb_source()


def test_the_loop_and_the_filter_read_the_same_thresholds():
    """``Bouquet.generate`` must source the in-loop bounds from the same
    FilterConfig ``Bouquet.filter`` later cuts on -- otherwise the run stops
    on a count the postprocess disagrees with."""
    from bouquet.run import Bouquet
    gen = inspect.getsource(Bouquet.generate)
    assert "inspec_rms_max_mm=fc.rms_max_mm" in gen
    assert "n_inspec_target=gc.n_inspec_target" in gen
    assert "max_total_draws=gc.max_total_draws" in gen
    flt = inspect.getsource(Bouquet.filter)
    assert "rms_max_mm=rms" in flt and "fc.rms_max_mm" in flt


def test_the_summary_states_delivered_vs_requested():
    """A short bouquet must not read as a completed run in the summary line."""
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet._print_generation_summary)
    assert "SHORT of target by" in src and "target met" in src
    assert "requested in-spec" in src


def test_boundary_devs_routes_through_the_shared_metric():
    from bouquet import filtering
    assert "boundary_deviation_mm(bl_boundary, perturbed)" in \
        inspect.getsource(filtering._boundary_devs)


def test_orphan_cap_set_after_construction_warns_at_generate():
    """The documented notebook idiom mutates config fields post-construction,
    past __post_init__'s validation."""
    from bouquet.run import Bouquet
    gen = inspect.getsource(Bouquet.generate)
    assert "max_total_draws has no effect without n_inspec_target" in gen
    assert "n_inspec_target must be >= 1" in gen


# ---------------------------------------------------------------------------
#  7. the extracted helpers -- the loop's arithmetic, executed
# ---------------------------------------------------------------------------
#  These replace the source-text placeholders above with behavioral coverage:
#  the budget arithmetic, the verdict glue (keys + percent conversion +
#  keyword wiring), the lazy scale extension's rng semantics, and the
#  delivered-count the caller re-derives outside the output capture.

class TestAttemptBudget:
    def _budget(self, *a):
        from bouquet.TokaMaker_interface import _resolve_attempt_budget
        return _resolve_attempt_budget(*a)

    def test_feature_off_is_exactly_n_equils(self):
        assert self._budget(20, None, None) == (None, 20)

    def test_default_cap_is_five_targets_floored_at_n_equils(self):
        assert self._budget(20, 3, None) == (3, 20)     # floor wins
        assert self._budget(20, 10, None) == (10, 50)   # 5x wins

    def test_explicit_cap_is_a_hard_ceiling_even_below_n_equils(self):
        assert self._budget(50, 10, 30) == (10, 30)

    def test_cap_below_target_raises(self):
        with pytest.raises(ValueError, match="could never be met"):
            self._budget(20, 10, 5)

    def test_nonpositive_target_raises_for_direct_callers(self):
        """generate_bouquet(n_inspec_target=0) used to 'meet' its target
        after one draw -- a silently truncated ensemble."""
        for bad in (0, -3):
            with pytest.raises(ValueError, match=">= 1"):
                self._budget(20, bad, None)

    def test_non_integral_and_bool_targets_raise(self):
        with pytest.raises(ValueError, match="integer"):
            self._budget(20, 2.5, None)
        with pytest.raises(ValueError, match="integer"):
            self._budget(20, True, None)
        assert self._budget(20, 3.0, None)[0] == 3   # int-valued float is fine

    def test_non_integral_and_bool_caps_raise_too(self):
        """int(60.9) -> 60 silently shortened the attempt budget, and
        int(True) -> 1 turned a cap into 'stop after one attempt'."""
        with pytest.raises(ValueError, match="max_total_draws"):
            self._budget(20, 10, 60.9)
        with pytest.raises(ValueError, match="max_total_draws"):
            self._budget(20, 10, True)
        assert self._budget(20, 10, 60.0)[1] == 60   # int-valued float is fine


class TestConfigLayerRejectsTheSameThings:
    """Copilot's comment was about the CONSTRUCTION layer: the budget resolver
    already refused a non-integral target, but only once generate_bouquet ran."""

    def _gen(self, **kw):
        from bouquet.config import GenerationConfig
        return GenerationConfig(**kw)

    def _cfg(self, **kw):
        from bouquet.config import (BouquetConfig, ReconstructionSource,
                                    SolverConfig)
        return BouquetConfig(
            source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
            solver=SolverConfig(mesh_path="m.h5"),
            output_header="hdr",
            generation=self._gen(**kw))

    def test_a_non_integral_target_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="n_inspec_target"):
            self._cfg(n_inspec_target=7.9)

    def test_a_bool_target_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="n_inspec_target"):
            self._cfg(n_inspec_target=True)

    def test_a_non_integral_or_bool_cap_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="max_total_draws"):
            self._cfg(n_inspec_target=5, max_total_draws=60.9)
        with pytest.raises(ValueError, match="max_total_draws"):
            self._cfg(n_inspec_target=5, max_total_draws=True)

    def test_integer_valued_floats_still_pass(self):
        c = self._cfg(n_inspec_target=5.0, max_total_draws=60.0)
        assert c.generation.n_inspec_target == 5.0     # not coerced in place

    def test_generate_rechecks_after_the_notebook_mutation_idiom(self):
        """The fields are mutated after __post_init__ in the documented idiom,
        so the same refusal must exist at generate()."""
        from bouquet.config import require_integer_count
        import inspect as _i
        from bouquet.run import Bouquet
        src = _i.getsource(Bouquet.generate)
        assert "require_integer_count" in src
        with pytest.raises(ValueError, match="integer count"):
            require_integer_count(7.9, "generation.n_inspec_target")
        assert require_integer_count(None, "x") is None


class TestUntilNVerdict:
    """The verdict glue, on the LEGACY coil predicate.

    The coil half is now a PREDICATE parameter (the configured filter --
    chi2 by default) rather than two hard-coded percent thresholds, so these
    build the legacy predicate explicitly through the same factory
    ``generate_bouquet`` calls.  The chi2 path is covered below.
    """

    def _verdict(self, diag, **kw):
        from bouquet.TokaMaker_interface import _until_n_verdict
        from bouquet.filtering import make_coil_predicate
        kw.setdefault("recon_lcfs_ref", _circle())
        kw.setdefault("perturbed_lcfs_ref", _circle())
        pred, kind, model = make_coil_predicate(
            "legacy", inspec_F_max=kw.pop("inspec_F_max", 0.02),
            inspec_VSC_max=kw.pop("inspec_VSC_max", 0.10))
        assert (kind, model) == ("legacy", None)
        return _until_n_verdict(diag, kw.pop("recon_lcfs_ref"),
                                kw.pop("perturbed_lcfs_ref"), pred, **kw)

    def test_fraction_thresholds_convert_to_percent(self):
        """diagnostics carry PERCENT drifts; the config carries fractions.
        A dropped x100 fails a 1.8%-drift draw against a 2% spec."""
        ok, *_ = self._verdict({"max_F_drift_pct": 1.8,
                                "max_VSC_drift_pct": 5.0})
        assert ok is True
        ok, *_ = self._verdict({"max_F_drift_pct": 2.2,
                                "max_VSC_drift_pct": 5.0})
        assert ok is False

    def test_argument_order_cannot_be_transposed_silently(self):
        """F and VSC have different limits (2% vs 10%): a draw with F=1%,
        VSC=9% passes only if the channels are wired the right way round.
        With both thresholds equal (the default FilterConfig) a swap is
        invisible in production -- this is the test that sees it."""
        ok, *_ = self._verdict({"max_F_drift_pct": 1.0,
                                "max_VSC_drift_pct": 9.0})
        assert ok is True
        ok, _, _, why, _ = self._verdict({"max_F_drift_pct": 9.0,
                                          "max_VSC_drift_pct": 1.0})
        assert ok is False and why == ("coil",)

    def test_missing_diagnostics_keys_fail_not_pass(self):
        ok, _, _, why, _ = self._verdict({})
        assert ok is False and "coil" in why

    def test_boundary_bound_flows_through(self):
        far = _circle(r=1.02)                 # ~20 mm off a 1 m circle
        ok, rms, mx, why, _ = self._verdict(
            {"max_F_drift_pct": 1.0, "max_VSC_drift_pct": 1.0},
            perturbed_lcfs_ref=far, rms_max_mm=5.0)
        assert ok is False and why == ("boundary",) and rms > 5.0

    def test_matches_the_postprocess_predicate_exactly(self):
        from bouquet.filtering import passes_all_filters
        diag = {"max_F_drift_pct": 1.7, "max_VSC_drift_pct": 8.0}
        got = self._verdict(diag, rms_max_mm=5.0)
        want = passes_all_filters(1.7, 8.0, _circle(), _circle(),
                                  2.0, 10.0, rms_max_mm=5.0)
        assert got[:4] == want
        # the drift percentages ride along for the log line / diagnostics
        assert got[4] == {"max_F_drift_pct": 1.7, "max_VSC_drift_pct": 8.0}


class TestConfiguredCoilPredicate:
    """The until-N loop stops on the CONFIGURED coil filter, not on the
    legacy band, and on the same numbers the postprocess cuts with.

    ``filtering.coil_filter`` defaults to the measurement-referenced chi2
    test, so a loop still counting ``|dI/I| <= 2%`` would grind toward its
    attempt cap chasing legacy-in-spec draws while ``.filter()`` marked a
    different (and larger) subset selected -- the identity this whole
    feature rests on, broken in both the count and the cost.
    """

    #: a 20-circuit signature the device registry resolves (18 F + 2 E)
    COILS = {**{f"F{i}{s}": 1e5 for i in range(1, 10) for s in "AB"},
             "ECOILA": 2e4, "ECOILB": 2e4}

    def _sigma(self, era="modern"):
        from bouquet.coil_spec import resolve_coil_sigma
        sig, model = resolve_coil_sigma(dict(self.COILS), era=era)
        return sig, model

    def _predicate(self, **kw):
        from bouquet.filtering import make_coil_predicate
        kw.setdefault("era", "modern")
        return make_coil_predicate("chi2", dict(self.COILS), **kw)

    def test_the_thresholds_are_the_devices_calibrated_acceptance(self):
        """Not the legacy band, and not a number of this module's own: the
        same ``resolve_coil_acceptance`` the postprocess filter calls."""
        _, _, model = self._predicate()
        acc = model["acceptance"]
        from bouquet.coil_spec import resolve_coil_acceptance
        _, sig_model = self._sigma()
        assert (acc["chi2_max"], acc["z_max"]) == \
            resolve_coil_acceptance(sig_model)[:2]

    def test_a_draw_inside_the_measured_precision_passes(self):
        pred, kind, _ = self._predicate()
        sig, _ = self._sigma()
        assert kind == "chi2"
        draw = {c: v + 0.5 * sig[c] for c, v in self.COILS.items()}
        ok, info = pred({}, draw)
        assert ok is True and info["chi2_nu"] == pytest.approx(0.25)
        # ...and the legacy drift percentages are NOT what decided it
        assert "max_F_drift_pct" not in info

    def test_a_draw_beyond_the_worst_coil_guard_fails(self):
        """One coil far out hides behind nineteen quiet ones in chi2/nu; the
        max|z| guard is what rejects it, and the loop must apply it too."""
        pred, _, model = self._predicate()
        sig, _ = self._sigma()
        zm = model["acceptance"]["z_max"]
        draw = dict(self.COILS)
        draw["F1A"] = self.COILS["F1A"] + (zm + 1.0) * sig["F1A"]
        ok, info = pred({}, draw)
        assert ok is False
        assert info["chi2_nu"] < model["acceptance"]["chi2_max"]   # pooled: quiet
        assert info["max_abs_z"] > zm and info["worst_coil"] == "F1A"

    def test_a_draw_with_no_coil_currents_is_a_FAILURE_not_a_pass(self):
        """Matches the postprocess fix: unjudgeable is not a pass.  In the
        loop it must also not be counted toward the target."""
        pred, _, _ = self._predicate()
        for empty in (None, {}):
            ok, info = pred({"max_F_drift_pct": 0.0,
                             "max_VSC_drift_pct": 0.0}, empty)
            assert ok is False
            assert np.isnan(info["chi2_nu"]) and info["coil_nu"] == 0

    def test_an_unresolvable_sigma_falls_back_to_legacy_loudly(self, monkeypatch):
        """Exactly where Bouquet.filter() falls back -- and with the same
        words, which is a fact about one shared function."""
        import bouquet.devices as dev
        monkeypatch.setattr(dev, "detect_device", lambda names: None)
        with pytest.warns(UserWarning, match="COIL FILTER FALLBACK"):
            pred, kind, model = self._predicate()
        assert kind == "legacy(fallback)" and model is None
        # and it really is the legacy rule now: a 3% drift fails a 2% band
        assert pred({"max_F_drift_pct": 3.0, "max_VSC_drift_pct": 0.1},
                    dict(self.COILS))[0] is False

    def test_a_baseline_without_coil_currents_falls_back_too(self):
        with pytest.warns(UserWarning, match="COIL FILTER FALLBACK"):
            from bouquet.filtering import make_coil_predicate
            _, kind, _ = make_coil_predicate("chi2", None, era="modern")
        assert kind == "legacy(fallback)"

    def test_the_filter_wrapper_and_the_loop_share_one_fallback_message(self):
        from bouquet import filtering, run
        assert "_coil_fallback_message" in inspect.getsource(run.Bouquet.filter)
        assert "_coil_fallback_message" in \
            inspect.getsource(filtering.make_coil_predicate)

    def test_generate_hands_the_loop_the_filter_config_coil_settings(self):
        """The wiring that makes the identity possible at all: same filter,
        same sigma, same acceptance, same DAQ era as .filter() resolves."""
        from bouquet.run import Bouquet
        gen = inspect.getsource(Bouquet.generate)
        for frag in ("coil_filter=fc.coil_filter", "coil_sigma=fc.coil_sigma",
                     "coil_device=self.config.device",
                     "coil_daq_era=self._coil_daq_era()",
                     "coil_chi2_max=fc.chi2_max", "coil_z_max=fc.z_max"):
            assert frag in gen, frag

    def test_the_postprocess_resolves_the_era_through_the_same_accessor(
            self, tmp_path, monkeypatch):
        """Behavioural form of the last line of the wiring above.

        ``generate`` hands the loop ``self._coil_daq_era()``; whatever that
        accessor returns must also be what ``.filter()`` hands the postprocess,
        or the two sides resolve different sigma FLOORS -- an acceptance
        criterion -- and the identity dies without a symptom.  Checked by
        driving a real ``.filter()`` rather than by matching its source text,
        which broke the moment the era announcement split the call in two.
        """
        from bouquet import filtering, run
        seen = {}
        real = filtering.filter_coil_chi2

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        monkeypatch.setattr(filtering, "filter_coil_chi2", spy)
        monkeypatch.setattr(run.Bouquet, "_coil_daq_era", lambda self: "pre2014")
        header = str(tmp_path / "era")
        _write_coil_archive(header + ".h5", [dict(TestConfiguredCoilPredicate.COILS)])
        _filter_selected(header)
        assert seen["era"] == "pre2014"

    def test_the_postprocess_cuts_with_the_shared_predicate(self):
        from bouquet import filtering
        src = inspect.getsource(filtering.filter_coil_chi2)
        assert "passes_coil_chi2(" in src
        assert "resolve_coil_acceptance(" in src


# ==========================================================================
#  THE identity, on the configured (chi2) coil filter
# ==========================================================================
def _write_coil_archive(path, draws, scan_key=1, radius=None):
    """One archive: baseline + per-draw coil currents and LCFS contours.

    *draws* is a list of ``{coil: current}``; *radius* an optional list of
    perturbed-LCFS radii (default: all identical to the baseline contour).
    """
    names = list(TestConfiguredCoilPredicate.COILS)
    base = np.array([TestConfiguredCoilPredicate.COILS[n] for n in names])
    bl = _circle()
    with h5py.File(path, "w") as hf:
        g = hf.create_group(f"scan/{scan_key}")
        b = g.create_group("_baseline")
        b.create_dataset("coil_names", data=np.array(names, dtype="S"))
        b.create_dataset("coil_currents", data=base)
        b.create_dataset("recon_lcfs_ref", data=bl)
        for i, cur in enumerate(draws):
            d = g.create_group(str(i))
            d.create_dataset("coil_names", data=np.array(names, dtype="S"))
            d.create_dataset("coil_currents",
                             data=np.array([cur[n] for n in names]))
            r = 1.0 if radius is None else radius[i]
            d.create_dataset("perturbed_lcfs_ref", data=_circle(r=r))
            # legacy drift percentages, deliberately DISAGREEING with the chi2
            # verdict: every draw is inside the +/-2% band, so a loop still
            # counting the legacy rule would pass draws the chi2 filter cuts.
            d.attrs["max_F_drift_pct"] = 0.5
            d.attrs["max_VSC_drift_pct"] = 0.5
            d.attrs["inspec_F_max"] = 0.02
            d.attrs["inspec_VSC_max"] = 0.02
    return bl


def _filter_selected(header, scan_key=1, **filt):
    """``Bouquet.filter()`` over an archive, as a run would call it."""
    from bouquet.config import FilterConfig
    from bouquet.run import Bouquet

    class Gen:
        scan_key = None
        n_inspec_target = None
    Gen.scan_key = scan_key

    class Cfg:
        output_header = header
        filtering = FilterConfig(**filt)
        generation = Gen()
        device = None
        source = type("S", (), {})()
    b_ = Bouquet.__new__(Bouquet)
    b_.config = Cfg()
    b_.filter(plot=False)
    return set(select_indices(header, scan_key=scan_key, selection="selected"))


def _loop_selected(bl_contour, draws, radii, coil_filter="chi2", **pred_kw):
    """The in-loop stopping rule's verdicts over the same draws."""
    from bouquet.TokaMaker_interface import _until_n_verdict
    from bouquet.filtering import make_coil_predicate
    pred, _, _ = make_coil_predicate(
        coil_filter, dict(TestConfiguredCoilPredicate.COILS), **pred_kw)
    out = set()
    for i, cur in enumerate(draws):
        ok, *_ = _until_n_verdict(
            {"max_F_drift_pct": 0.5, "max_VSC_drift_pct": 0.5},
            bl_contour, _circle(r=radii[i]), pred, draw_currents=cur,
            rms_max_mm=5.0)
        if ok:
            out.add(i)
    return out


def _straddling_draws():
    """Draws that straddle chi2/nu, max|z| and the LCFS bound, in units of
    the resolved per-coil sigma -- so the fixture cannot silently stop
    exercising both outcomes when a floor changes."""
    from bouquet.coil_spec import resolve_coil_acceptance, resolve_coil_sigma
    base = dict(TestConfiguredCoilPredicate.COILS)
    sig, model = resolve_coil_sigma(dict(base), era="modern")
    cm, zm, _, _ = resolve_coil_acceptance(model)
    n = np.sqrt(cm)                      # per-coil z that sits exactly at chi2_max
    draws, radii = [], []
    #  0: well inside          -> passes both channels
    draws.append({c: v + 0.3 * n * sig[c] for c, v in base.items()}); radii.append(1.0)
    #  1: pooled chi2 over     -> coil fails
    draws.append({c: v + 1.4 * n * sig[c] for c, v in base.items()}); radii.append(1.0)
    #  2: one coil past z_max  -> coil fails on the worst-coil guard alone
    d = dict(base); d["F3B"] = base["F3B"] + (zm + 2.0) * sig["F3B"]
    draws.append(d); radii.append(1.0)
    #  3: coils fine, LCFS 10 mm out -> boundary fails
    draws.append({c: v + 0.3 * n * sig[c] for c, v in base.items()}); radii.append(1.010)
    #  4: just INSIDE chi2_max (and inside z_max) -> passes, narrowly.
    #     Deliberately 0.98n rather than exactly n: a draw sitting on the
    #     bound is a float-equality coin toss, and the point of this fixture
    #     is a near-threshold draw both sides agree on, not the inclusivity
    #     of the comparison (covered by test_coil_chi2_is_inclusive...).
    draws.append({c: v + 0.98 * n * sig[c] for c, v in base.items()}); radii.append(1.0)
    return draws, radii


def test_the_identity_holds_on_the_chi2_coil_filter(tmp_path):
    """THE test: the set the loop would stop on IS the set .filter() selects,
    with the default (chi2) coil filter -- neither branch had this."""
    header = str(tmp_path / "chi2id")
    draws, radii = _straddling_draws()
    bl = _write_coil_archive(header + ".h5", draws, radius=radii)

    with pytest.warns(UserWarning):        # nu 20 vs calibrated 18 (recorded)
        post = _filter_selected(header, coil_daq_era="modern",
                                coil_filter="chi2", rms_max_mm=5.0)
    inloop = _loop_selected(bl, draws, radii, era="modern")

    assert inloop == post, f"in-loop {sorted(inloop)} != selected {sorted(post)}"
    # the fixture must exercise both outcomes, or the equality is vacuous
    assert 0 < len(post) < len(draws)
    # ...and it must NOT be the legacy verdict: every draw is inside +/-2%,
    # so a loop on the legacy rule would have counted all five.
    legacy = _loop_selected(bl, draws, radii, coil_filter="legacy",
                            inspec_F_max=0.02, inspec_VSC_max=0.02)
    assert legacy != post and len(legacy) > len(post)


def test_the_identity_holds_on_the_legacy_fallback_path(tmp_path, monkeypatch):
    """Same identity when the sigma cannot be resolved: both sides fall back
    to the legacy rule, so both sides must still agree."""
    import bouquet.devices as dev
    header = str(tmp_path / "legacyid")
    draws, radii = _straddling_draws()      # needs the registry, before the patch
    bl = _write_coil_archive(header + ".h5", draws, radius=radii)
    monkeypatch.setattr(dev, "detect_device", lambda names: None)
    # make the legacy channel discriminate: draw 2 is out of the +/-2% band
    with h5py.File(header + ".h5", "a") as hf:
        hf["scan/1/2"].attrs["max_F_drift_pct"] = 9.0

    with pytest.warns(UserWarning, match="COIL FILTER FALLBACK"):
        post = _filter_selected(header, coil_filter="chi2", rms_max_mm=5.0)
    with pytest.warns(UserWarning, match="COIL FILTER FALLBACK"):
        pred, kind, _ = __import__(
            "bouquet.filtering", fromlist=["x"]).make_coil_predicate(
                "chi2", dict(TestConfiguredCoilPredicate.COILS))
    assert kind == "legacy(fallback)"

    from bouquet.TokaMaker_interface import _until_n_verdict
    inloop = set()
    for i, cur in enumerate(draws):
        diag = {"max_F_drift_pct": 9.0 if i == 2 else 0.5,
                "max_VSC_drift_pct": 0.5}
        ok, *_ = _until_n_verdict(diag, bl, _circle(r=radii[i]), pred,
                                  draw_currents=cur, rms_max_mm=5.0)
        if ok:
            inloop.add(i)
    assert inloop == post, f"in-loop {sorted(inloop)} != selected {sorted(post)}"
    assert 0 < len(post) < len(draws)


class TestScaleBlockExtension:
    def test_no_range_extends_with_ones(self):
        from bouquet.TokaMaker_interface import _extend_scale_block
        out = _extend_scale_block(np.ones(4), None, None, 4)
        np.testing.assert_array_equal(out, np.ones(8))

    def test_extension_is_deterministic_and_leaves_the_block_alone(self):
        """The rng-stream contract, executed: the initial block is identical
        with and without the feature (the extension consumes the generator
        only AFTER the block draw), and the extension itself is
        reproducible under the run's seed."""
        from bouquet.TokaMaker_interface import _extend_scale_block
        lo, hi, n = 0.6, 0.8, 5
        g1 = np.random.default_rng(42)
        block_only = g1.uniform(lo, hi, size=n)
        g2 = np.random.default_rng(42)
        block = g2.uniform(lo, hi, size=n)
        extended = _extend_scale_block(block, g2, (lo, hi), n)
        np.testing.assert_array_equal(block_only, extended[:n])
        assert len(extended) == 2 * n
        assert np.all((extended >= lo) & (extended <= hi))
        g3 = np.random.default_rng(42)
        g3.uniform(lo, hi, size=n)
        again = _extend_scale_block(block, g3, (lo, hi), n)
        np.testing.assert_array_equal(extended, again)


class TestDeliveredCount:
    def test_counts_only_the_stored_verdicts(self):
        from bouquet.filtering import until_n_delivered
        diags = [{"until_n_inspec": True}, {"until_n_inspec": False},
                 {}, {"until_n_inspec": True}]
        assert until_n_delivered(diags) == 2
        assert until_n_delivered([]) == 0
        assert until_n_delivered(None) == 0
