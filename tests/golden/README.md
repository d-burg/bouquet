# Golden bouquet test fixtures

Git-tracked regression fixtures for `tests/test_golden_bouquet.py`.

| file | what it is |
|------|------------|
| `D3Dlike_Hmode_golden_slim.h5` | a slimmed real bouquet run (~11.3 MB): `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles). |
| `golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points) with tolerances. |
| `rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the slim fixture's baseline profiles + sigma envelopes. |
| `regenerate_golden_run.py` | produces that full run: the recipe (stored config, class API, INPUT-current archival). |
| `make_golden_fixture.py` | regenerates the three files above from a full run. |
| `D3Dlike_Hmode_legacy_golden.json` | the slim LEGACY golden (~0.2 MB): the same recipe run with `reconstruction_engine="legacy"` -- stored config, reconstruction scalars/profiles/coils, a 1-in-8 subsample of its LCFS trace, every draw's scalars, and the profiles + coils + boundary RMS of the first two in-spec draws (what `tests/test_systematics.py` replays). Built by `make_golden_fixture.py --legacy-json`. |

## The draw-stream golden

`rng_stream_manifest.json` is the only draw-*level* golden here, and it became
possible only when `GenerationConfig.seed` started reaching the GPR: before
that, every draw site re-seeded from OS entropy and no drawn value was
reproducible. It replays what `perturb_kinetic_equilibrium` does for one draw
— ne, Te, ni, Ti through `_draw_monotonic_perturbation`, then the `j_phi` GPR
candidate — off one `make_rng(seed)` Generator. Pure NumPy: no solver, no
mesh, so it is bitwise identical on any machine.

Re-pin it on its own (no full run needed) with

```bash
python tests/golden/make_golden_fixture.py --rng-stream-only
```

The manifest carries sampled values and per-channel min/max alongside each
hash, so the git diff shows roughly *where* a stream moved, not just that it
did. A changed hash with unchanged samples means the change is elsewhere in
the profile.

The geqdsks are deliberately retained: geqdsk is a coarse-at-the-separatrix
format and exercising its read/parse path on real files (see the
`test_geqdsk_*` tests) is worthwhile. They are stored as gzipped `uint8`
under their original `.eqdsk` dataset names, so every reader
(`bytes(grp[k][()])`) is unaffected. `make_golden_fixture.py --eqdsk` chooses
retention: `all` (default, ~11.3 MB), `subset` (baseline + representative
draws, ~5 MB), or `none` (~3.7 MB, no geqdsk-handling coverage). Eventually,
when the default interchange migrates to IMAS/OMAS, the fixture can store
those instead.

The full-fidelity 30 MB run (with p-file bytes and uncompressed eqdsks) stays
as the shareable example artifact under
`bouquet/examples/D3D-like/D3Dlike_Hmode_golden.h5` (not tracked here).

## Updating the golden set (on purpose)

**The recipe** (every systematics golden: 0f92d28, bc7a49a, 060bc1f, and the
input-current rebuild of the self-consistent-bootstrap refresh):

* the fixture's **own stored config** (`scan/0/config_json`), verbatim:
  20 draws, seed 12345, the synthetic-IDA sigmas, the `jbs_*` loop settings
  and pass ceilings, `solver.nthreads=1`; only the archive name and the log
  verbosity are set;
* the **class API**: `Bouquet(cfg) -> setup_solver() -> prepare_baseline() ->
  generate()`;
* **input-current archival** (`store_achieved_jphi=False`): `_baseline/j_phi`
  and each draw's `j_phi` / `j_inductive` hold the current the generator
  handed the solver, not the achieved flux-surface average. The systematics
  replay feeds `_baseline/j_phi` back as the input of its baseline solve, so
  this is its premise ("archived current reproduces archived LCFS").
  **`Bouquet.generate()` hard-wires `store_achieved_jphi=True`**, so a plain
  notebook run does NOT follow the recipe; the refresh in 5296720 was built
  that way and mode 1 failed on it (see "mode-1 coil drift" below). The
  current fixture (bc85d46) is that refresh rebuilt with this recipe, and
  mode 1 passes on it.
  `store_achieved_jphi` changes only what is written, never what is solved;
* one thread (`OMP_NUM_THREADS=1`), on the OFT build the fixture pins, stated
  through `BOUQUET_OFT_COMMIT` / `BOUQUET_OFT_BRANCH` / `BOUQUET_OFT_BUILD_ID`
  when slimming.

The earlier goldens switched the archival off the same way: their
regeneration script wrapped `bouquet.TokaMaker_interface.generate_bouquet`
(which `generate()` imports at call time) to inject
`store_achieved_jphi=False`. That script was never committed; the commits
stated the convention, not the mechanism, and the refresh lost it.
`regenerate_golden_run.py` is that recipe, in the tree. It flips only
`store_achieved_jphi` on `generate()`'s own `generate_bouquet` call, refuses
to run if `generate()` stops passing it as `True`, and stamps the archive root
with `golden_jphi_archival = "input"`. The builder copies that attr into the
fixture and records it in the manifest's provenance
(`generator_args.jphi_archival`), so a fixture says how it was archived.

1. Produce the full run (~6.5 h at one thread for this example; run it on a
   machine with the disk and the time for it):
   ```bash
   OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR --verbose
   ```
2. Regenerate the fixture + manifest:
   ```bash
   python tests/golden/make_golden_fixture.py \
       --source RUN_DIR/D3Dlike_Hmode_golden.h5
   ```
   (defaults to the D3D-like example artifact path if `--source` is omitted;
   `rng_stream_manifest.json` is re-pinned from the new slim fixture in the
   same command).
3. Review the `golden_manifest.json` git diff — it shows exactly which physics
   values moved — then commit the new fixture + manifests together.

The builder refuses to write a fixture that names a filesystem path anywhere
a reader sees text: string attrs, string datasets (`config_json`), string
ARRAYS (attrs and datasets, element by element), and geqdsk headers. A path is
an absolute path under a well-known root (`/Users`, `/home`, `/usr`,
`/Volumes`, `/opt`, `/mnt`, ...), any absolute path of two or more components
at a token boundary, a `~/` or `~user/` path, a `../` path, or a Windows
drive path; units and ratios such as `A/m^2` or `1/R` are not
(`tests/test_no_paths_in_records.py` tries to defeat the guard). It reduces
the paths inside the self-consistent bootstrap records (`jbs_loop_json`) to
basenames: records written by earlier builds of the loop carried the OFT
package path in `oft_build.path`; current builds record a path-free
`oft_build = {version, git_hash, build_id}`. This repository is public, and
the OFT build is identified by the content digests in the provenance stamp.
The committed fixture passes the extended guard as it stands (it predates the
path-free record, so its loop records carry the scrubbed basename
`oft_build.path = "OpenFUSIONToolkit"`).

## The self-consistent-bootstrap refresh

The fixture is a loop-on run: `GenerationConfig.jbs_self_consistent=True`
(the default), regenerated from the previous fixture's own stored config with
only the code and the bootstrap model changed, on the OFT line that carries
the bootstrap stencil change (build identity in the provenance stamp). Pass
ceilings 12 per draw loop and 4 post-homotopy; no tolerance moved. Of 20
requested draws, 17 are archived (10 in spec): one draw was skipped when an
l_i-match candidate's solve exhausted `maxits`, and two were rejected at the
post-homotopy stage (see the known limitation below). Every archived draw's
loops converged, the longest in 9 passes. `test_the_fixture_is_a_self_consistent_bootstrap_run`
asserts the stored config, a converged `jbs_loop` block on the baseline and on
every draw, and the path guard.

`rng_stream_manifest.json` was re-pinned on the same machine class as before:
the ne / Te / ni / Ti stream hashes are unchanged, and only the `jphi` hash
moved, because that channel is drawn from the fixture's baseline j_phi, which
now carries the self-consistent bootstrap (baseline j_phi moved by ~1 % of its
peak, sigma_jphi by ~1.5 %).

**Rebuilt with input-current archival (bc85d46).** The 5296720 refresh was
made through plain `Bouquet.generate()` and so archived the ACHIEVED current
(see "mode-1 coil drift" below). The fixture was then rebuilt from a full run
of `regenerate_golden_run.py`: the same stored config (seed 12345, 20 draws,
ceilings 8 reconstruction / 12 per draw loop / 4 post-homotopy, as stored and
as applied), the same OFT build, one thread, ~6.2 h. Archival changes only
what is written, and the two runs agree on that. Every group, attr and dataset
of the new archive is identical to the 5296720 one (coil currents, X-points,
kinetic profiles, geqdsks, l_i(1) / l_i(3), in-spec flags, the `jbs_loop`
records apart from wall time, `eq_fsa`), except `j_phi` and `j_inductive` on
`_baseline` and on every draw. Those now hold the solver input and differ by
1.2-1.9 % of peak. The run log's loop-pass, homotopy, in-spec and rejection
lines are identical line for line, so the same draws were skipped and rejected
(count 2 skipped; counts 1 and 17 rejected post-homotopy). `golden_manifest.json`
changed only in its provenance. In `rng_stream_manifest.json` the `jphi` hash
moved once more, because that channel is drawn from `_baseline/j_phi`, now the
input current; the four kinetic hashes are unchanged.
`test_the_fixture_archives_the_input_current` asserts the archival stamp in the
fixture, its provenance and the manifest.

## Regenerated on the unified-engine default (2026-10-07)

The h5 fixture is now a run of the unified engine, the default reconstruction
engine: the same recipe and stored config (seed 12345, 20 draws, one thread,
OFT `fix/jphi-update-ravgs-and-nonfinite-abort` 7da4f18, build
`20260929_7da4f18`), run with `regenerate_golden_run.py
--reconstruction-engine unified`, which sets the field and puts the
legacy-only fields the engine never reads back to their defaults (printed and
recorded in the run's `summary.json`; here only `isolate_edge_jBS`
False -> True). No bar changed.

* Unified engine: 20 attempts, **17 archived** (3 rejected:
  `homotopy_maxits`), **4 in spec** (draws 0, 9, 13, 17). An engine draw's
  `in_spec` is the coil verdict AND its post-hoc l_i band
  (`passes_draw_band`, one of `filtering._FILTER_FLAGS`): 5 of 17 pass the
  coil rule, 11 of 17 the band. The three tests that restated the selection
  rule as "coil (and boundary)" now AND the band where a draw carries it
  (`test_coil_filter_reproduces_in_spec`,
  `test_chi2_filter_is_the_default_and_needs_no_dd`,
  `test_the_identity_holds_on_the_real_golden_archive`), as
  `engine_draws.until_n` and `filtering._recompute_selected` do. Wall time:
  reconstruction 66 s, draws 42.4 min (123.6 s per equilibrium).
* The legacy path keeps its numeric record in
  `D3Dlike_Hmode_legacy_golden.json`, made from a legacy-engine run of the
  same recipe at the same code: 20 archived, 12 in spec -- the in-spec flags
  of the 2026-10-05 fixture on every draw; l_i(1) / l_i(3) move by <= 0.61 %
  and the baseline j_BS peak by -6.4 %, the declared bootstrap conversion
  (`<j.B> F <1/R> / <B^2>`) now applied on the legacy path. Wall time:
  reconstruction 217 s, draws 153.6 min (457.7 s per equilibrium). `tests/test_systematics.py` (legacy
  replay) reads it; `tests/test_legacy_golden.py` checks what it is.
* `rng_stream_manifest.json`: re-pinned from the new baseline (the `jphi`
  stream's edge sample moves most, 1.57e5 -> 7.10e4 A/m^2).
* **Validated at `1a15685` on the owner's cluster** (aggregate; see
  [docs/validation-provenance.md](../../docs/validation-provenance.md)):
  fast suite 2897 passed; solver suite 165 passed, 2 failed, 1 skipped --
  the two failures are the q0-pinned structured loop test, which needs about
  one pass more than the default reconstruction ceiling under the
  solved-state residual (owner decision pending; nothing changed), the skip
  a comparison that needs an older OFT build. `tests/test_systematics.py`
  against the new legacy JSON golden, bars unchanged: 3 passed (mode 1
  baseline RMS 0.4189 mm, limit 0.8; max coil drift 0.0345 %, limit 0.3;
  mode 2 draws 0 / 3 0.426 / 0.511 mm; mode 3 draw 0 li(3) replay 0.6287 vs
  golden 0.6267, li(1) 0.8194 vs 0.8197). Fixture numbers as above: engine
  17 archived / 4 in spec, reconstruction 66 s, draws 124 s per
  equilibrium; legacy 20 archived / 12 in spec, draws 458 s per
  equilibrium.
* **The bouquet stamp reads dirty.** The fixture and both manifests stamp
  bouquet commit `8285201 (the branch was re-ordered after generation: the stamp inside the fixture and its manifests reads `7bd48fb`, the pre-reorder name of the commit whose tree is now `8285201`; the trees are identical)` with `dirty: true`: the generator edits it ran
  with (`--reconstruction-engine`, `--legacy-json`) were not yet committed
  when the run was made, and were committed together with the fixture in
  `1a15685`. The fixture is regenerable from `1a15685`'s tree. (`3d974e6`
  later changes how the generator's engine switch records
  `isolate_edge_jBS` -- that field now defaults to `None` and is resolved
  per engine -- not what runs: it still records and applies False -> True.)

## Regenerated on the fixed OFT build at today's defaults (2026-10-05, owner-approved)

An owner-approved change of an acceptance BASELINE (no bar changed). From: the
fixture above (OFT `fix/bootstrap-pchip-derivatives` build `20260919_abbfc6f`,
the jphi_update `<1/R>` row shift unfixed; stored config with
`separatrix_pressure` back-filled `"legacy"`, post-homotopy ceiling 4). To:
the same recipe (`regenerate_golden_run.py`, the same stored config, seed
12345, 20 draws, one thread) with only the defaults that changed since moved to
today's -- `separatrix_pressure="offset"` (`edge_pprime_pin=True`),
`jbs_max_passes_post_homotopy=6` -- on the fixed OFT line
`fix/jphi-update-ravgs-and-nonfinite-abort` (commit 7da4f18, build
`20260929_7da4f18`); bouquet commit and OFT build/branch/commit are stamped in
the fixture and both manifests.

* Archived / in spec: 17 / 10 -> **20 / 12** (draws 1, 2 and 17 now archived;
  draw 8 now in spec).
* Per draw (old -> new): l_i(1) mostly ~ -0.3 %, l_i(3) ~ -0.25 %; draw 3
  l_i(3) +2.4 % (homotopy pass 1 -> 2); draw 19 l_i(3) +10 % (0.6257 -> 0.6886),
  out of spec both times.
* `test_systematics` against it, bars unchanged: 3 passed. Mode 1
  `[replay mode1] max coil drift = 0.0190% (limit 0.3)` (1.357 % against the
  previous fixture: its separatrix part is gone with the like-for-like replay
  settings), `baseline RMS = 0.4193 mm (limit 0.8)`; mode 2 draw 0 0.445 mm;
  mode 3 draw 0 `boundary RMS replay=0.992 golden=2.166 mm li(3) replay=0.6275
  golden=0.6259 li(1) replay=0.8172 golden=0.8175`, draw 3 `boundary RMS
  replay=0.818 golden=1.324 mm li(3) replay=0.6379 golden=0.6372 li(1)
  replay=0.8313 golden=0.8296`.
* The regeneration was made twice (two code states of the engine line that
  differ only off the legacy draw path); the second reproduced the first's
  in-spec flags and l_i(1) / l_i(3) on every draw exactly (largest relative
  difference 0). `rng_stream_manifest.json`: all five stream hashes moved --
  `jphi` because it is drawn from the new `_baseline/j_phi` (sampled values
  move by <= 0.3 % except the edge sample), the four kinetic streams because
  the baseline kinetic envelopes the stream is drawn from differ in their last
  digits (e.g. the n_e minimum 6.981469218e18 -> 6.981469216e18).

## `test_systematics` against the refreshed fixture

**Against the input-current rebuild (bc85d46): all three modes pass** on the
Linux production build and OFT line of the fixture, at one thread
(`3 passed, 53 warnings in 770.30s (0:12:50)`). No bar was changed.

* Mode 1: `[replay mode1] baseline RMS = 0.4418 mm (limit 0.8)`,
  `[replay mode1] max coil drift = 0.0189% (limit 0.3)`. The replay's
  jphi-baseline solve lands at l_i(3) = 0.65387, the regeneration's own value.
  The drift is about twice the 0.0092 % that run B-on predicted below, and
  16× inside the bar.
* Mode 2: `draw 0: pressure-only boundary RMS = 0.446 mm`,
  `draw 3: pressure-only boundary RMS = 0.953 mm` (bar 6 mm).
* Mode 3 (with the c75576e replay of `generate()`'s bootstrap model): both
  replayed draws now produce an equilibrium and reproduce within every bar.
  `draw 0: boundary RMS replay=0.689 golden=2.083 mm  li(3) replay=0.6287
  golden=0.6275  li(1) replay=0.8189 golden=0.8200`;
  `draw 3: boundary RMS replay=0.687 golden=2.046 mm  li(3) replay=0.6234
  golden=0.6224  li(1) replay=0.8151 golden=0.8174`. Draw 3's replay no
  longer exhausts `maxits`, so draw 3 is checked again. The boundary-RMS
  difference sits at 1.36-1.39 mm against the 2 mm bar, as it did before the
  rebuild (draw 0: 0.719 vs 2.083 mm).

What follows is the history against the 5296720 fixture (achieved-current
archival).

**Mode 3 (`test_mode3_production_reproduces_golden`) passes** on the Linux
production build. Before the refresh it missed the recorded l_i(1) of draw 3
by 3.74 % against its 3 % bar (`l_i(1) replay 0.8240 vs golden 0.8560`), the
end-stencil signature of the OFT bootstrap-gradient change the old fixture
predated. Against the refreshed fixture the replayed draws are the first two
in-spec ones (0 and 3): draw 0 reproduces within every bar (`boundary RMS
replay=0.719 golden=2.083 mm  li(3) replay=0.6274 golden=0.6275  li(1)
replay=0.8182 golden=0.8200`), but draw 3's mode-3 replay produced no
equilibrium (`STOPPED: ValueError: Error in solve: Exceeded "maxits"`), and
the test skips a replay that produced none, so draw 3 was NOT re-checked.
No bar was changed.

**Mode 1 (`test_mode1_pinned_baseline_reproduces_baseline`) now fails** --
new with the refresh (it passed at 0.0059 % against the frozen fixture):
`[replay mode1] max coil drift = 1.2292% (limit 0.3)` (`assert
np.float64(1.2292165342528374) < 0.3`), boundary RMS 0.5440 mm inside its
0.8 mm bar. The drift is on one coil (F9B) and is already present between the
replay's own loop-on reconstruction and the fixture's loop-on baseline
(1.2278 %); the pinned draw adds only 0.005 %. The fixture's baseline comes
from `Bouquet(stored config) -> prepare_baseline`, the replay's from
`Bouquet.from_geqdsk(...) -> reconstruct()`; with the frozen bootstrap the two
agreed to 0.006 %, with the loop on they do not. The bar is not widened.
Diagnosed in "Resolved: mode-1 coil drift after the refresh" below.
The two entry points reconstruct bit-identically. The drift comes from the
refresh's archival convention (achieved rather than input baseline current),
not from the loop.

## Test scope: what the tests around this fixture do not cover

Stated so that a pass is not read as more than it is. None of these is a
loosened bar; each narrows what a passing suite shows.

* **The axis-collapse regression test is a build-dependent skip.**
  `tests/test_fsa_current_integral.py::test_get_q_collapses_silently_on_an_unclipped_grid`
  was a hard assertion (`unclipped <R> span < 1e-9`, absolute, over all
  surfaces) and is now a measured skip: it asserts only on a build whose
  unclipped grid collapses (at least 2 traced surfaces with a relative `<R>`
  span <= 1e-9) and skips, with a reason, on every other build. It can
  therefore skip with the defect present (a PARTIAL collapse that pins only
  some surfaces, or a build that traces fewer than 2 surfaces). What runs on
  every build instead: the unconditional invariant that the CLIPPED geometry
  is real (`<R>` spans more than 10 % of its mean) and two solver-free tests
  of the guard. The production guard in `utils.fsa_current_geometry` uses the
  raw `<R>` span, zero (untraced) rows included, so an untraced row can hide
  a collapse from it; `physics.evaluate_jBS` now refuses a zero (failed-trace)
  row on its own surfaces.
* **Six zero-perturbation tests run on the legacy path only.** The route-R2
  sigma=0 tests of `tests/test_seeded_reproducibility.py`
  (`test_sigma0_r2_reproduces_the_baseline_jbs`,
  `test_sigma0_r2_exact_measure_lands_in_its_own_budget`,
  `test_sigma0_r2_exact_measure_reports_a_plausible_inductive_share`,
  `test_sigma0_r2_exact_measure_still_recovers_the_recon_li`,
  `test_sigma0_r2_exact_measure_is_bit_reproducible`,
  `test_sigma0_r2_exact_measure_leaves_the_bootstrap_alone`) build their
  baseline with `jbs_self_consistent=False`: they test the legacy frozen-SWB
  path, not the shipped default. The default path's counterpart,
  `tests/test_jbs_loop_solver.py::test_f_sigma0_route_r2_draw_converges_near_the_baseline`,
  checks only that the loop converges, the `|s-1|*f_ind <= 3.86e-3` budget
  and the init-source bookkeeping -- not the other five properties.
* **The structured solver tests run with a pass ceiling of 12; the default
  is 8.** `tests/test_jbs_loop_solver.py` sets
  `jbs_max_passes = _STRUCTURED_TEST_PASSES = 12` for the structured closure
  and its MSE stage, because that synthetic case needed exactly the default 8
  passes on the development build. The shipped default therefore has no
  headroom on that channel, and the tests do not show that it converges
  within 8 on another build. The default was not changed.

## Known limitation: a standard draw's post-homotopy re-solve can diverge slowly

Seen in the loop-on regeneration of this example (config seed 12345, 20
requested draws, pass ceilings 12 per draw loop / 4 post-homotopy): archive
counts **1** and **17** were rejected this way, both before and after the
ceilings were raised, and again in the input-current rebuild (bc85d46; the
stage took ~20-40 min for count 1 and ~40-60 min for count 17, bounded by the
run's 10-min stack dumps). Both runs, whose logs match line for line, also
show a slow failure one stage earlier: count 15's homotopy pass 3 of 3
(F/VSC +/-1 %) exhausted `maxits` after ~30-50 min and was rolled back to pass 2, and the draw was kept
(in spec). That step is the homotopy's own infeasible-pass rollback, not this
limitation. The limitation itself ends in a rejected draw, loudly, never an
accepted one; nothing here caps, retries or falls back.

**What happens.** The draw's own loops converge (count 1: anchor 5 passes,
l_i-match candidate 6; count 17: anchor 3, then four l_i-match candidates of
7-8 passes each). The coil homotopy then delivers the draw (count 1 stops at
pass 2 of 3, F +/-2 %, VSC +/-5 %, because its natural VSC drift of 4.17 %
already exceeds the next pass's +/-1 %; count 17 reaches pass 3, F/VSC
+/-1 %, with 0.54 % / 0.29 % drift). Redl on the delivered equilibrium misses
the bootstrap the draw carries (count 1: `r_j = 2.0e-3`, `r_I = 2.4e-4`;
count 17: `r_j = 9.1e-4` inside, `r_I = 1.8e-4` outside), so the post-homotopy
stage takes its first pass: the delivered inductive held, the relaxed
bootstrap swapped in, the target renormalised to I_p (x1.0035 and x0.9957)
and handed to `_corrective_jphi_iteration`. The corrective iteration's
`jphi-linterp` solve then does not converge. A `jphi-linterp` solve
flux-surface-averages every nonlinear iterate, so once the iterate loses its
nested surfaces every iteration's surface trace fails (the log fills with
`gs_get_qprof: Trace did not complete` and `DLSODE ... R1 = NaN`) and the
solver spends its whole iteration budget: roughly 25-40 min for count 1 and
33-43 min for count 17 at one thread, inside `TokaMaker.solve` in every
periodic stack dump. For count 1 the log shows `[jphi_corr] WARNING: the
FIRST corrective solve failed (Error in solve: Exceeded "maxits")`; for
count 17 no such line appears (the post-homotopy call runs with
`verbose=False`, so a later-iteration failure, or a solve that returned a
degenerate state, is not distinguishable in the log). The call runs without
`protect_state`, so the solver is left in that state; the loop then measures
it, `fsa_current_geometry`'s guard refuses it (`get_q returned a CONSTANT <R>
across all 257 surfaces ... the surface tracer collapsed onto the magnetic
axis`), and the draw is rejected as OUT_OF_SPEC with a NaN VSC drift. This is
the broken state's collapse, not the build-dependent near-axis collapse
described under "Test scope" below; it occurs on the Linux build.

**What the standard-draw post-homotopy fix changed, and why it did not remove
this.** Before that fix the pass solved `j_ind_used + j_BS` as ONE
`jphi-linterp` request, where `j_ind_used` is derived from the draw's stored
j_phi -- the ACHIEVED current of its corrective iteration, not the solver
input that achieved it; on count 1 that solve exhausted `maxits` (a scratch
60-iteration cap made it fail in 34 s; uncapped it ran for tens of minutes).
The fix routes the pass through the draw's own I_p renormalisation +
corrective iteration. But the corrective iteration's first input IS the
target (`j_phi_input = target_jphi.copy()`), so its first solve is handed the
same achieved-derived profile, rescaled by <0.5 %: not the input that
produced the delivered state (the draw's corrective iteration had moved its
input off the target by its Newton edge corrections, edge RMS
~0.014 -> ~0.001 MA/m^2). The failure moved from "the single solve exhausts
maxits" to "the corrective iteration's first solve exhausts maxits"; wall
time and outcome are the same.

**Why these draws.** Not established. Count 1 sits at a loose coil stage with
the VSC near its bound, which suggested a basin-escape under tight bounds;
count 17 falsifies that as the whole story (tightest stage, 0.29 % VSC
drift, a smaller first residual). What the two share is a new
`jphi-linterp` request solved from a converged free-boundary state under
homotopy-tightened coil bounds with I_p and p_axis pinned. 17 of 19 draws that
reached this stage passed it (3 accepted without passes, the rest in 2-4
passes).

**Options (none implemented; each needs a decision):**

1. *Fail fast on a diverging solve* (robustness guard, not a tolerance
   change). Stop a corrective solve whose nonlinear residual grows over
   consecutive iterations, or whose surface trace fails, and fail the pass at
   once: the same rejection in seconds instead of ~30-45 min. Needs a
   per-iteration residual / trace-status hook from the solver, or a
   per-solve iteration budget for this stage (the latter changes a solver
   limit and needs its own approval).
2. *Start the post-homotopy corrective iteration from the draw's last
   corrective INPUT* (plus the bootstrap change) instead of from the target,
   so the solver is asked for a near neighbour of a state it just reached.
   Path-only (the fixed point is unchanged); may rescue these draws rather
   than reject them faster. Requires carrying that input in the draw context.
3. *`protect_state=True` for the post-homotopy corrective call*, so a failed
   solve restores the pre-solve state and the pass fails as a non-converged
   loop with the solver's own message rather than through the axis-collapse
   guard. Clearer reason; wall time unchanged.
4. *`verbose=True` (or a recorded failure field) for that call*, so the log
   and the draw record say which corrective iteration failed and how.
   Diagnostic only.
5. *Status quo*: a rejected draw, ~30-60 min of wall time per occurrence
   (2 of 20 draws, ~1.6 h of a 6.5 h single-thread run here).

## Resolved: mode-1 coil drift after the refresh

**Resolved by the input-current recipe; no bar changed.** Option 1 below was
taken. `regenerate_golden_run.py` (f5d440f) regenerates the run with
`store_achieved_jphi=False`, and the fixture was rebuilt from that run
(bc85d46). Mode 1 now passes at 0.0189 % max coil drift (bar 0.3 %),
boundary RMS 0.4418 mm (bar 0.8 mm); see "`test_systematics` against the
refreshed fixture" above. Option 4 (mode 3 replays `generate()`'s bootstrap
model) landed separately in c75576e. The diagnosis below is kept as history,
written when the 5296720 fixture was current.

**Diagnosed (history).** The mode-1 failure is not
caused by the self-consistent bootstrap loop, and the two entry points do not
reconstruct differently. It is a change in how the refreshed fixture
archives the baseline current. The replay then solves its baseline from a
different j_phi than the generator used.

**The mechanism.** `test_systematics` builds its mode-1 baseline inside
`generate_bouquet`: the unperturbed `jphi-linterp` baseline solve
(`jphi_baseline=True`) is handed `input_j_phi` and leaves the coils free
under the reconstruction's isoflux set. The coil currents after that solve
are what `_baseline/coil_currents` records and what mode 1 compares against.
The generator hands that solve the reconstruction's own delivered current
(`Baseline.j_phi`). The replay hands it the fixture's `_baseline/j_phi`. The
replay is therefore a faithful replay only if `_baseline/j_phi` holds the
INPUT current. That is the documented recipe of every earlier systematics
golden: "input-current archival (`store_achieved_jphi=False`)", from commits
0f92d28, bc7a49a and 060bc1f and `docs/CHANGES_SUMMARY.md`, "because a
replay premise 'archived current reproduces archived LCFS' requires the
input, not the achieved, current". The refresh was regenerated with
`Bouquet(stored config) -> setup_solver -> prepare_baseline -> generate()`.
`Bouquet.generate()` hard-wires `store_achieved_jphi=True`, so the refreshed
fixture archives the ACHIEVED FSA current of that solve (the regeneration log
prints `[archive] baseline j_phi = achieved FSA current`). The per-draw
`j_phi` / `j_inductive` are archived as achieved currents too. The loop was
switched on in the same regeneration, so the change looked like a
loop-on/loop-off difference.

### Evidence

Every replay below ran on the Linux production build and OFT line of the
refresh, at one thread, through the test's own call sequence
(`Bouquet.from_geqdsk -> reconstruct()`, recon isoflux restored, then
`generate_bouquet` with the test's arguments). Only what each row names was
varied; for the mode-1 rows that is just the `input_j_phi` of the pinned
call. The replay scripts are diagnosis scratch and are not committed.

**(1) The two entry points run the same reconstruction.** `reconstruct()` is
`setup_solver(); prepare_baseline()`, the same two calls the regeneration
made. The fixture's stored config carries the full `jbs_*` set it was built
with (schema v3): `jbs_self_consistent=True`, `jbs_init="anchor"`, `rtol_j
1e-3`, `rtol_Ip 1e-4`, `tol_li 1e-3`, `tol_q0 2e-3`, ceilings 8 / 12 / 4,
`relax 0.7`, `relax_current 0.7`, `relax_halve_on 3`, `loop_on_fail "raise"`.
The test copies `jbs_self_consistent` from that stored config and takes the
other `jbs_*` fields from the defaults, which are identical. A field-by-field
diff of the stored config against the test's `from_geqdsk` config finds only
`n_equils` (20 vs 1), `seed` (12345 vs None), `sigma_profiles` (explicit vs
empty), `fixed_components.p_fast_reduction` (`trace` vs `auto`, with no fast
pressure in this case), and list-vs-tuple spellings of `jBS_scale_range` and
`homotopy_passes`. None of these reaches the reconstruction. There is no
difference in coil regularisation, bootstrap model, `recalculate_j_BS`,
`isolate_edge_jBS` or `perturb_jind_in_anchor`. The loop records are
bit-identical:

| reconstruction `jbs_loop` | fixture (refresh) | replay (test path) |
|---|---|---|
| init / init_source | anchor / none | anchor / none |
| main loop n_passes, converged | 4, True | 4, True |
| r_j per pass | 4.5367e-3, 1.3796e-3, 4.2161e-4, 1.3085e-4 | identical to every digit |
| r_I, dl_i, omega per pass | (recorded) | identical to every digit |
| post-corrective check | r_j 1.8125e-3, r_I 4.898e-4, not ok | identical |
| post-corrective passes, r_j | 3: 5.4434e-4, 1.6541e-4, 5.5847e-5 | identical |
| tolerances, criteria blocks | (recorded) | equal |

Both paths record the delivered state after the same final solve, the last
post-corrective pass. The reconstruction summaries agree to every printed
digit.

**(2) The difference is the mode-1 input current.**

| run | bootstrap | reference fixture | mode-1 `input_j_phi` | jphi-baseline l_i(3) | baseline coils vs fixture, max | test metric (mode-1 draw vs fixture) |
|---|---|---|---|---|---|---|
| A-on (= the test) | loop | refreshed | archived `_baseline/j_phi` (achieved) | 0.65331 | 1.2278 % (F9B) | **1.2292 %** (fails 0.3 %) |
| B-on | loop | refreshed | the replay's own `Baseline.j_phi` (what `generate()` passes) | 0.65387 (regeneration: 0.65387) | 0.0006 % | **0.0092 %** |
| A-off | frozen | pre-refresh (input archival) | archived `_baseline/j_phi` (input) | 0.65369 | 0.0003 % | 0.0059 % |
| B-off | frozen | pre-refresh | the replay's own `Baseline.j_phi` | 0.65384 | 0.4841 % (F8A) | 0.4842 % |
| C-off | frozen | B-off's own archive, achieved archival | B-off's archived achieved j_phi | 0.65334 (B-off: 0.65384) | 2.4842 % (F9B) | **2.5015 %** (fails 0.3 %) |

B-on reproduces the refreshed fixture: all 20 baseline coils within 0.0006 %,
and the achieved current it archives matches the fixture's `_baseline/j_phi`
to 2.8e-6 of peak. The loop-on baseline is therefore exactly reproducible
from the test's own entry point once the solve gets the generator's input.
C-off is the counterfactual: with the frozen bootstrap, a baseline archived
the refresh's way and replayed the test's way fails mode 1 in the same way.
The drift comes from the archival convention; the bootstrap model plays no
part. The frozen counterfactual drifts further (2.50 %) than the loop-on
fixture (1.23 %), and F9B again carries the maximum. B-off adds only a
side fact. Today's frozen reconstruction current sits 0.28 %-of-peak
(edge) away from the pre-refresh fixture's archived input, because the
code moved after that fixture was pinned. The test never sees this, since
it replays the archive.

The input-vs-achieved gap on the refreshed baseline, as (`Baseline.j_phi` −
archived achieved `j_phi`) / peak:

| psi_N band | max \|d\| / peak | mean d / peak |
|---|---|---|
| [0, 0.5) | 5.2e-3 | +1.7e-3 |
| [0.5, 0.9) | 8.3e-4 | +3.9e-4 |
| [0.9, 0.98) | 1.25e-2 | +3.4e-3 |
| [0.98, 1] | 1.74e-2 | −8.5e-3 |

This is the `jphi-linterp` edge realisation gap that the jphi-baseline solve
exists to absorb. It is a property of the representation, not of the loop:
the frozen analogue (B-off's achieved current against its input) is about
1.4e-2 of peak.

**(3) Not a path dependence of the loop's fixed point.** The loop is
bit-reproducible across the two entry points (table in (1)), so the
init-independence tests are not contradicted. For the record, those tests
(`test_d_the_fixed_point_does_not_depend_on_the_initial_guess`,
`test_draw_fixed_point_does_not_depend_on_its_start`) compare only j_BS
(within 5 `rtol_j`) and l_i (within 2 `tol_li`). They compare no coil
currents and no boundary.

**(4) Is F9B weakly constrained?** The shift is not one coil's. It is a
redistribution over the whole coil set, carried mostly by the B-set coils
and ECOILB. F9B's small current makes its share the largest in relative
terms. The frozen counterfactual C-off shows the same pattern: F9B +2.48 %,
ECOILB +1.04 %, F3B +0.61 %, F5B −0.54 %. Here is A-on against the fixture
(A-t = ampere-turns):

| coil | fixture [A-t] | Δ [A-t] | Δ [%] |
|---|---|---|---|
| F9B | −45 398 | +557 | +1.228 |
| F3B | −57 322 | +498 | +0.869 |
| F3A | 26 816 | −157 | −0.586 |
| ECOILB | −30 592 | +151 | +0.492 |
| F4B | 114 438 | −544 | −0.475 |
| F8A | 41 780 | −182 | −0.436 |
| F5B | 175 092 | −584 | −0.334 |
| F8B | 182 945 | −385 | −0.210 |

In this synthetic case the coils are only loosely pinned. `from_geqdsk` sets
no `coil_reg` targets, so every coil is pulled toward zero at weight 1, with
the VSC at 1e-2, against the reconstruction's isoflux set. F9B is also one
half of the default VSC pair (`coil_vsc = {F9A: +1, F9B: -1}`), whose channel
carries the weakest regularisation term. The A-on shift is not a pure VSC
mode, though: F9A moved −95 A-t where F9B moved +557 A-t. The reconstruction's
own inverse-mode coil set and the jphi-baseline coil set of the same plasma
differ by up to 3.1–3.6 % (B-on / A-on, F4B). Commit 060bc1f documented the
same degeneracy: a psi re-initialisation moved the baseline coils 0.41 % (F8A)
for a 1.5e-5 change in l_i. Measured sensitivity here: a 1.7e-2-of-peak edge
change (5e-3 core) moves the coils up to 1.23 %, l_i(3) by −0.086 % and the
boundary by 0.54 mm RMS. A 2.8e-3-of-peak edge change (B-off) moves them up to
0.48 %. So the 0.3 % bar is meaningful only when the replay's input current is
the generator's, bit for bit. That was true under input archival and is not
true now.

Coil currents per loop pass (loop on, test path, drift against the
reconstruction's final coils): the four main passes sit at F9B −0.41, −0.29,
−0.26, −0.25 % (max over coils 3.15 → 3.11 %, on F4B). The post-corrective
passes then converge to 0.016 %, 0.003 % and 0. The loop's delivered coil
state is stable. The baseline coil record is set by the jphi-baseline solve
that follows, not by the loop.

**(5) Draw 3's mode-3 `Exceeded "maxits"`.** It is not the coil stage. The
replay failed on the solve of pass 2 of the draw's anchor loop (`[jbs-loop
draw anchor] pass 1/12: r_j=6.531e-03 ... I_BS=219.36 kA`, then `STOPPED:
... Exceeded "maxits"`), before any coil homotopy. The golden's own draw 3
converged that loop in 6 passes. The failure is input-dependent. Mode 3's
`generate_bouquet` call differs from the one `Bouquet.generate()` makes in
several ways, and none of them is new with the refresh except the first:

* it feeds the draw's archived `j_phi` / `j_inductive`, which the refresh
  archives as ACHIEVED currents (the per-draw half of the archival change);
* it passes `Zeff ≡ 1`, where `generate()` passes the baseline Z_eff
  (1.76–1.92 here);
* it leaves `isolate_edge_jBS` at the function default `True`, where
  `generate()` passes the geqdsk workflow's `False` (`from_geqdsk`), and it
  passes no `baseline_j_BS` and no `jBS_scale_range`.

The replay's draw loops therefore carry a different bootstrap from the
generator's: I_BS ≈ 219–233 kA, against ≈ 295–312 kA in the golden's draw
loops. Z_eff alone accounts for only 219 → 233 kA; the edge-isolated
decomposition is the likely remainder, but that was not run separately.
The same replay was rerun with only Z_eff set to the baseline value. The
anchor solve that had exhausted `maxits` now succeeded: the anchor loop
converged in 6 passes (I_BS 232.5 → 232.8 kA) and the l_i-match loop in 6
(l_i error 2.10 %, inside the 5 % band). The coil homotopy then stopped at
pass 2 of 3 on a natural VSC drift of 2.55 %, and the post-homotopy check
came out at r_j = 1.45e-3 / r_I = 1.67e-4, outside tolerance. The draw was
therefore in the post-homotopy corrective stage, which is the pattern of the
known limitation above; that stage had not finished when this was written,
and the run was stopped. So the `maxits` is a property of what the replay
is asked to solve, not of the coil stage. The mode-3 replay is not a
faithful replay of the generator's draw in the first place.

*Later:* c75576e made the mode-3 replay use `generate()`'s bootstrap model,
and bc85d46 made the fixture archive input currents. With both, draw 3's
replay converges its anchor loop in 3 passes (I_BS ≈ 313 kA, in line with
the golden's draw loops) and its l_i-match loop in 3 (l_i error 0.17 %),
solves all three homotopy stages, is accepted post-homotopy without passes
and reproduces the golden draw (see above).

### Options (as written before the resolution; 1 and 4 were taken)

1. *Regenerate the fixture under the documented recipe.* Use input-current
   archival (`store_achieved_jphi=False`) with the loop on, everything else
   unchanged. B-on predicts mode 1 at about 0.009 %. This is an
   acceptance-artifact update (a new approved golden commit, a manifest
   re-pin, about 6.5 h at one thread) and it restores the premise for modes
   2 and 3 too. The `jphi` hash in `rng_stream_manifest.json` would move
   again, because that channel is drawn from `_baseline/j_phi`.
   `Bouquet.generate()` has no switch for this today; the regeneration
   script would have to supply it. How the earlier recipe did
   so is not recorded in the tree, and that gap should be closed whichever
   route is taken.
2. *Keep the achieved-archived fixture and feed the replay the generator's
   input.* Mode 1 and mode 2 would pass `run.baseline.j_phi` from the
   replay's own reconstruction. The reconstruction is reproduced exactly
   (identical loop records), and B-on shows this puts the baseline coils
   within 0.0006 % of the fixture. This is a test-code change only, with
   the same bars. Mode 1 then also depends on the reconstruction reproducing,
   which it already implicitly did. Mode 3 has no archived per-draw input
   current, so it would need option 3 or stay the looser check it is.
3. *Archive both currents.* An additive dataset (e.g. the input current next
   to the achieved one) on `_baseline` and on every draw. Replays use the
   input; plots and consumers keep the achieved current that matches the
   eqdsk. This is a schema addition.
4. *Make the replay's `generate_bouquet` call match `generate()`'s.* Pass
   the baseline Z_eff, `isolate_edge_jBS=False`, `baseline_j_BS` and
   `jBS_scale_range` as the class API does, and Z_imp / p_fast / j_NBI /
   j_RF where set, instead of the function defaults. The simplest way is to
   derive the arguments from the reconstructed `Bouquet` rather than
   hand-copy them. This is a test-code change, and it concerns mode 3's
   fidelity (and the draw-3 `maxits`), not mode 1. It is the same class of
   drift the fixture's docstring already warns about for isoflux, psi_pad
   and warm start.
5. *Treat the coil set as weakly constrained and report it.* For example,
   normalise the drift per coil by a coil-current scale or use the coil χ²
   filter's sigma. That changes the acceptance criterion and needs explicit
   approval. It is not recommended as the fix here: the cause is an input
   mismatch, and B-on passes the existing bar with a 30× margin.
6. *Diagnostics, optional.* Record coil currents in the `jbs_loop` pass
   history, and add coil currents (or boundary RMS) to the fixed-point
   tests. Neither would have caught this failure, since the loop is
   bit-reproducible.

The `*.h5` glob in `.gitignore` is negated for `tests/golden/*.h5` so the slim
fixture is tracked while ad-hoc run outputs elsewhere stay ignored.
