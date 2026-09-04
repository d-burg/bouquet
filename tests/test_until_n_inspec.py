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
#  5. serial-only guard
# ==========================================================================
def test_every_parallel_entry_point_rejects_a_target(tmp_path):
    """Shards cannot see each other's yield: N workers each chasing the target
    would deliver N*target draws. All three entry points must refuse."""
    from bouquet.parallel import (emit_slurm_script, parallel_generate,
                                  run_shard)
    cfg = _mini_config(n_inspec_target=5)
    for call in (
        lambda: parallel_generate(cfg, n_workers=2, seed=1),
        lambda: emit_slurm_script(cfg, n_workers=2, seed=1,
                                  threads_per_worker=1,
                                  out_dir=str(tmp_path), job_name="j"),
        lambda: run_shard(cfg, 0, 2, n_equils_total=4, seed_base=1,
                          out_header="h", scan_key=0, threads_per_worker=1),
    ):
        with pytest.raises(ValueError, match="serial-only"):
            call()


def test_parallel_still_runs_the_guard_before_anything_expensive():
    """The rejection must precede shard sizing / process spawn, so a bad
    config fails in milliseconds rather than after N baselines solve."""
    from bouquet import parallel
    src = inspect.getsource(parallel.run_shard)
    body = src.split(':\n', 1)[1]
    assert body.index("_reject_until_n") < body.index("_shard_size")


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


class TestUntilNVerdict:
    def _verdict(self, diag, **kw):
        from bouquet.TokaMaker_interface import _until_n_verdict
        kw.setdefault("recon_lcfs_ref", _circle())
        kw.setdefault("perturbed_lcfs_ref", _circle())
        kw.setdefault("inspec_F_max", 0.02)
        kw.setdefault("inspec_VSC_max", 0.10)
        return _until_n_verdict(diag, kw.pop("recon_lcfs_ref"),
                                kw.pop("perturbed_lcfs_ref"),
                                kw.pop("inspec_F_max"),
                                kw.pop("inspec_VSC_max"), **kw)

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
        ok, _, _, why = self._verdict({"max_F_drift_pct": 9.0,
                                       "max_VSC_drift_pct": 1.0})
        assert ok is False and why == ("coil",)

    def test_missing_diagnostics_keys_fail_not_pass(self):
        ok, _, _, why = self._verdict({})
        assert ok is False and "coil" in why

    def test_boundary_bound_flows_through(self):
        far = _circle(r=1.02)                 # ~20 mm off a 1 m circle
        ok, rms, mx, why = self._verdict(
            {"max_F_drift_pct": 1.0, "max_VSC_drift_pct": 1.0},
            perturbed_lcfs_ref=far, rms_max_mm=5.0)
        assert ok is False and why == ("boundary",) and rms > 5.0

    def test_matches_the_postprocess_predicate_exactly(self):
        from bouquet.filtering import passes_all_filters
        diag = {"max_F_drift_pct": 1.7, "max_VSC_drift_pct": 8.0}
        got = self._verdict(diag, rms_max_mm=5.0)
        want = passes_all_filters(1.7, 8.0, _circle(), _circle(),
                                  2.0, 10.0, rms_max_mm=5.0)
        assert got == want


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
