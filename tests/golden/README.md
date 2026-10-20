# Golden bouquet test fixtures

Git-tracked regression fixtures. Every golden is a real bouquet run, made by
`regenerate_golden_run.py` and slimmed by `make_golden_fixture.py`. The recipe
runs the synthetic test fixtures in `tests/data/` at the library's defaults.
**Status (2026-10-17):** all three goldens are such runs, at main `27979d1`
(1.4.0: the homotopy's VSC bound floor, the boundary-partition stamp, the
per-draw wall budget, the kinetic clips on by default), the g-file and legacy
ones on the thermal-only g-file (see "The current goldens" below). The IDS
golden was regenerated after A1 / B / C (2026-10-17, commit `35c75f0`): it is
IDA-hybrid (the measured IDA Z_eff), which A1 does not change, so every draw
and every manifest number is identical to the 27979d1 run; the archive adds
the Z_eff definition stamps (`zeff_definition`: total, from the IDA file).
The g-file golden (thermal g-file + IDA `.cdf`, no p-file) is not affected by
B's p-file `pb` default and was not regenerated. The legacy JSON golden was
regenerated at `35c75f0` with the per-draw wall budget off (decision C, below).

| golden | source | engine | files | read by |
|--------|--------|--------|-------|---------|
| g-file (h5) | `g_synthetic.geqdsk` + `diiid_profs_synthetic.cdf` (the synthetic D3D-like g-file, thermal pressure only, and its IDA-layout profile file) | unified (default) | `D3Dlike_Hmode_golden_slim.h5`, `golden_manifest.json`, `rng_stream_manifest.json` | `tests/test_golden_bouquet.py`, `tests/test_rng_reproducibility.py`, and the archive/filter/stats tests that read a real archive |
| IDS (h5) | `dd_synthetic.json.gz` at 1.2 s, IDA-hybrid with `diiid_profs_synthetic.cdf` (the production IDS route) | unified (default) | `D3Dlike_ids_golden_slim.h5`, `ids_golden_manifest.json`, `ids_rng_stream_manifest.json` | `tests/test_golden_bouquet.py` (every test runs on both h5 goldens), `tests/test_golden_ids.py`, `tests/test_rng_reproducibility.py` |
| legacy (JSON) | the g-file golden's source pair | legacy | `D3Dlike_Hmode_legacy_golden.json` | `tests/test_systematics.py` (the legacy replay, `-m solver`), `tests/test_legacy_golden.py` |

| file | what it is |
|------|------------|
| `*_golden_slim.h5` | a slimmed real bouquet run: `*.pfile` byte blobs dropped, the `*.eqdsk` geqdsks **kept but gzip-compressed** (~3x; all of them on the g-file golden, 12.7 MB; a representative subset of 4 on the IDS golden, 7.5 MB), `Ip` also extracted into an attr, everything the assertions need kept (attrs, `coil_currents`, `x_points`, both LCFS refs, profiles, the stored config). |
| `golden_manifest.json`, `ids_golden_manifest.json` | expected per-draw + baseline values (l_i, Ip, coil drifts, boundary RMS/max, coil currents, X-points, `in_spec`) with tolerances, and the provenance (bouquet commit, OFT build, numerics). |
| `rng_stream_manifest.json`, `ids_rng_stream_manifest.json` | the **seeded GPR draw stream**, pinned bitwise (SHA-256 per channel + sampled values), drawn from the golden's baseline profiles + sigma envelopes. |
| `D3Dlike_Hmode_legacy_golden.json` | the slim LEGACY golden (~0.2 MB): the g-file golden's recipe run with `reconstruction_engine="legacy"` -- stored config, reconstruction scalars/profiles/coils, a 1-in-8 subsample of its LCFS trace, every draw's scalars, and the profiles + coils + boundary RMS of the first two in-spec draws (what `tests/test_systematics.py` replays). Built by `make_golden_fixture.py --legacy-json`. |
| `regenerate_golden_run.py` | produces a full run: the recipe (the synthetic fixtures at the defaults, class API, INPUT-current archival), and the sigma=0 probe of the same reconstruction. |
| `make_golden_fixture.py` | regenerates the slim fixture + manifests of one golden from a full run (`--golden gfile|ids`, `--legacy-json`). |
| `structured_closure_pre_mse.json` | a separate record of the structured closure (not a run golden). |

## The draw-stream goldens

`rng_stream_manifest.json` and `ids_rng_stream_manifest.json` are the
draw-*level* goldens, and they became possible only when
`GenerationConfig.seed` started reaching the GPR: before that, every draw site
re-seeded from OS entropy and no drawn value was reproducible. Each replays
what `perturb_kinetic_equilibrium` does for one draw -- ne, Te, ni, Ti through
`_draw_monotonic_perturbation`, then the `jphi_total` GPR candidate -- off one
`make_rng(seed)` Generator, from its golden's baseline profiles. Pure NumPy: no
solver, no mesh, so it is bitwise identical on the machine that pinned it and
equal to `1e-7` elsewhere.

Re-pin one on its own (no full run needed) with

```bash
python tests/golden/make_golden_fixture.py --rng-stream-only [--golden ids]
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
retention: `all` (default), `subset` (baseline + representative draws: the
first in spec, the first out of spec, the min and max boundary deviation), or
`none` (no geqdsk-handling coverage).

## Updating the golden set (on purpose)

**The recipe** (`regenerate_golden_run.py`):

* the **config is built fresh** by the class constructor
  (`Bouquet.from_geqdsk(g_synthetic.geqdsk, profiles=diiid_profs_synthetic.cdf,
  ...)`; `Bouquet.from_imas(dd_synthetic.json, time=1.2,
  ida_path=diiid_profs_synthetic.cdf, ...)` for the IDS golden), **every
  setting at its default** except the run's identity: 20 attempts (no
  until-N), seed 12345, `nthreads=1`, the engine. So a golden embodies the
  defaults of the code that made it; when a default moves, the goldens are
  regenerated, not edited. (`--config-from FIXTURE` replays a golden's stored
  config verbatim instead.)
* the **inputs are linked** into the run directory under their basenames (the
  dd unpacked from its `.gz`), so the stored config names no path;
* the **class API**: `Bouquet(cfg) -> setup_solver() -> prepare_baseline() ->
  generate()`;
* **input-current archival** (`store_achieved_jphi=False`): `_baseline/jphi_total`
  and each draw's `jphi_total` / `jphi_ind` hold the current the generator
  handed the solver, not the achieved flux-surface average. The systematics
  replay feeds `_baseline/jphi_total` back as the input of its baseline solve, so
  this is its premise ("archived current reproduces archived LCFS").
  **`Bouquet.generate()` hard-wires `store_achieved_jphi=True`**, so a plain
  notebook run does NOT follow the recipe (an earlier refresh built that way
  failed the mode-1 replay of `tests/test_systematics.py`; see "Why
  input-current archival: the mode-1 coil drift" below).
  `store_achieved_jphi` changes only what is written, never what is solved.
  `regenerate_golden_run.py` flips only `store_achieved_jphi` on
  `generate()`'s own `generate_bouquet` call, refuses to run if `generate()`
  stops passing it as `True`, and stamps the archive root with
  `golden_jphi_archival = "input"`; the builder copies that attr into the
  fixture and records it in the manifest's provenance
  (`generator_args.jphi_archival`);
* one thread (`OMP_NUM_THREADS=1`), on the OFT build the fixture pins, stated
  through `BOUQUET_OFT_COMMIT` / `BOUQUET_OFT_BRANCH` / `BOUQUET_OFT_BUILD_ID`
  when slimming;
* the **sigma=0 probe** (`--sigma0-probe OUT.json`, same arguments): the same
  config reconstructed in its own process and
  `Bouquet.verify_sigma0_consistency()` run on it (the zero-perturbation draw
  must reproduce the reconstruction). It makes no draws: the probe leaves the
  solver re-anchored, and the golden run must be the production sequence.

The commands (each run ~45-60 min at one thread on the engine, ~4 h on the
legacy path; run them on a machine with the disk and the time):

```bash
# the g-file h5 golden (+ golden_manifest.json, rng_stream_manifest.json)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_G --golden gfile --reconstruction-engine unified --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --golden gfile --source RUN_G/D3Dlike_Hmode_golden.h5
# the IDS h5 golden (+ ids_golden_manifest.json, ids_rng_stream_manifest.json)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_I --golden ids --reconstruction-engine unified --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --golden ids --eqdsk subset --source RUN_I/D3Dlike_ids_golden.h5
# the legacy JSON golden (the recipe switches the per-draw wall budget off: draw_wall_budget_s=None)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_L --golden gfile --reconstruction-engine legacy --verbose
BOUQUET_OFT_BUILD_ID=<build> python tests/golden/make_golden_fixture.py --legacy-json --source RUN_L/D3Dlike_Hmode_golden.h5
# the sigma=0 probe of any of them (same arguments, its own run directory)
OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_S --golden ids --reconstruction-engine unified --sigma0-probe sigma0.json
```

Slim each golden from a clean checkout of the commit that ran it (the builder
records `git status` in the provenance: build one golden, move its files
aside, restore the tree, build the next), then review the manifest / JSON git
diff -- it shows exactly which physics values moved -- and commit the new
fixtures and manifests together. The builder refuses a run whose stored
source or engine is not the golden's (an h5 golden is a unified-engine run of
its own source class; the legacy JSON a legacy-path g-file run).

The builder refuses to write a fixture that names a filesystem path anywhere
a reader sees text: string attrs, string datasets (`config_json`), string
ARRAYS (attrs and datasets, element by element), and geqdsk headers. A path is
an absolute path under a well-known root (`/Users`, `/home`, `/usr`,
`/Volumes`, `/opt`, `/mnt`, ...), any absolute path of two or more components
at a token boundary, a `~/` or `~user/` path, a `../` path, or a Windows
drive path; units and ratios such as `A/m^2` or `1/R` are not
(`tests/test_no_paths_in_records.py` tries to defeat the guard). It reduces
the paths inside the self-consistent bootstrap records (`jbs_loop_json`) to
basenames (records written by earlier builds of the loop carried the OFT
package path in `oft_build.path`; current builds record a path-free
`oft_build = {version, git_hash, build_id}`). This repository is public, and
the OFT build is identified by the content digests in the provenance stamp.

## Build-specific goldens

The goldens record the OFT build they were generated with, and are valid
for that build only. The record is `provenance.oft` of
`golden_manifest.json`, `ids_golden_manifest.json` and
`D3Dlike_Hmode_legacy_golden.json`:
`build_id` as stated through `BOUQUET_OFT_BUILD_ID` when slimming,
`library_sha256` and `sources_sha256` as measured. The policy (owner
decision, 2026-10-09):

* Every golden comparison runs at its existing bar, on every build.
  `tests/_harness.golden_build_check` compares the installed library's
  SHA-256 (`bouquet.jbs_loop.oft_build_info()`) with the fixture's
  `library_sha256`; the `build_id` comparison is informational. A fixture
  with no `library_sha256` is "unstamped" and counts as a mismatch.
* Same build: a pass is a pass, a failure is a regression.
* Different build: a pass passes with a `GoldenBuildMismatchWarning` that
  names both builds; a failure FAILS (never xfail, never skip), its message
  starting with `OFT build mismatch: fixture generated with
  <build_id>/<sha8>, installed <build_id>/<sha8>` and the regeneration
  commands, followed by the original failure.
* No band and no tolerance depends on the build.

Known cross-build deviation (measured on the earlier D3D-like example
golden): the mode-1 replay of the legacy golden in
`tests/test_systematics.py` (pinned baseline, bar 0.3 %) gave a maximum
coil drift of 0.027 % on the build the golden was generated with, and
1.45 % on OFT builds that evaluate the edge FF' from an exact per-node
`<R>` at the diverted LCFS node: the lower coils move, the boundary stays
within its bar. Which edge convention is right is an open physics
question; until it is answered, regenerate on the build in use, never widen
the bar.

Regenerating on the installed build: the commands of "Updating the golden
set" above (`tests/_harness.py` prints them in a mismatch message), with
`BOUQUET_OFT_BUILD_ID` naming the installed build; review the git diff of
the manifests / JSON before committing.

## The current goldens (2026-10-17)

**The g-file h5 golden and the IDS golden** are the recipe's runs at bouquet
main `27979d1` (a clean tree), OFT `fix/jphi-update-ravgs-and-nonfinite-abort`
7da4f18 (build `20260929_7da4f18`, lib `7885fedc1e62`), one thread; the
g-file golden on the corrected, thermal-only `g_synthetic.geqdsk` +
`diiid_profs_synthetic.cdf`. The defaults they embody: the in-spec rule is
the magnetics (no l_i band), IDA-hybrid takes the IDA file's Z_eff, the
radial sigma templates (the profile file's own envelopes here), the warm
baseline re-solve in `generate()`, engine draws held on the reconstruction's
isoflux with the boundary constraint weighted per unit length, `check_bistability`
on, the coil-saturation test against the installed bound, and the Redl epsilon
`(R_max - R_min)/(R_max + R_min)` (`evaluate_jbs/4`); since `27979d1` also the
homotopy's VSC bound floor (10 kA), the report-only boundary-partition probe
(stamped in the baseline's `solver_geometry_json`: g-file 1.16, IDS 0.95), the
600 s per-draw wall budget (stamped in the generation provenance; 0 rejections
here) and the kinetic clips on by default (no clip fired on any draw of
either golden). Their draws, counts and manifests' numbers are those of the
`49ebebb` regeneration (only the provenance and the new stamps differ).

**`in_spec` is the in-spec rule's verdict.** Since main `49ebebb`,
`generate_bouquet` judges every draw by the rule before archiving it (the
configured coil test, chi2 by default, AND the boundary cut, AND an engine
draw's band flag; `in_spec_source = "inspec_rule"`). The old +/-2 % coil-drift
verdict is kept beside it as `coil_drift_in_spec`; it is not the rule, and on
these goldens it differs (the g-file golden: 9 of its 18 in-spec draws pass
the drift test; the IDS golden: 18 of 20).

| golden | attempts | archived | in spec (`coil_drift_in_spec`) | reconstruction | draws | bistability | sigma=0 probe |
|--------|----------|----------|-------------------------------|----------------|-------|-------------|---------------|
| g-file, unified | 20 | 18 (2 homotopy_saturated) | 18 (9) | 76 s | 32 min | single | PASS (archived dl_i -1.5e-5, dq0 +4.1e-5, r_j 5.6e-5, r_I 7.8e-6) |
| IDS, unified | 20 | 20 | 20 (18) | 73 s | 31 min | single | PASS (archived dl_i -9.6e-6, dq0 -4.7e-5, r_j 1.4e-4, r_I 1.4e-5) |

* On the integration branch both h5 goldens' datasets and the legacy JSON
  golden's keys carry the v4 names (`schema.migrate_names`,
  `schema.LEGACY_NAMES`), and both manifests' boundary numbers are distances
  to the perturbed LCFS polyline (`filtering.contour_distance`), recomputed
  from the same slim h5s: RMS -0.05…-0.13 mm, max up to -4.5 mm against the
  nearest-vertex values. Nothing else changed.
* The IDS route archives no schema `jbs_loop` block on its `_baseline` (the
  g-file route does); its reconstruction's loop verdict is in the baseline's
  engine record (`engine_json`: `converged`, `loop_converged`), which the
  golden test reads there.

**The legacy JSON golden** is the same recipe on `reconstruction_engine="legacy"`
(same commit, build and pair) **with the per-draw wall budget off**
(`draw_wall_budget_s=None`, `regenerate_golden_run.LEGACY_DRAW_WALL_BUDGET_S`;
owner decision 2026-10-17). Why: legacy draws take up to about ten minutes
each on the validation host (2.5-10 min), so under the default 600 s budget
which draws survive depended on the machine and its load (the 27979d1 run with
the budget archived 15 of 20, 4 rejected by it; the `49ebebb` run, before the
budget existed, 19); a golden must be host-independent, and the legacy route
is deprecated (#90). The current legacy golden is that recipe at `35c75f0`
(the A1 / B / C code; the g-file + IDA `.cdf` pair, so neither A1 nor B acts
on it): 20 attempts, **19 archived** (1 perturb_failed), 19 in spec, draws
177 min at one thread; its replayed draws are the first two in-spec ones,
0 and 1. The h5 goldens keep the default budget. The replay
(`tests/test_systematics.py`) calls `generate_bouquet` directly, outside the
class API's `DrawSolveGuard`, so it runs with no budget too -- the same
setting. The legacy replay reconstructs from the pair its stored config names.

* The legacy record archives the pressure-driven current as its own
  `jphi_pd` beside a `jphi_ind` that does not carry it, and says so
  (`current_split_convention = "pressure_separate"`); the legacy replay
  composes the legacy path's carried (in-memory) inductive, `jphi_ind +
  jphi_pd`, before handing it to `generate_bouquet` as `jphi_ind_input`
  -- a record from before that convention replays as is.

## Why input-current archival: the mode-1 coil drift

Mode 1 of `tests/test_systematics.py` builds its baseline from the fixture's
`_baseline/jphi_total`, handed to the unperturbed `jphi-linterp` baseline solve,
and compares the coil currents after that solve with
`_baseline/coil_currents`. The generator handed the same solve the
reconstruction's own delivered current, so the replay is faithful only if
`_baseline/jphi_total` holds that INPUT current. A refresh made through plain
`Bouquet.generate()` (which archives the ACHIEVED flux-surface-averaged
current) failed mode 1 at 1.23 % max coil drift on one coil against the
0.3 % bar; the same run rebuilt with `regenerate_golden_run.py`
(`store_achieved_jphi=False`) passed at 0.019 %. No bar was changed.
`test_the_fixture_archives_the_input_current` asserts the archival stamp in
the fixture, its provenance and the manifest.

The refresh history of earlier fixtures is in the messages of the commits
that changed them (`git log -- tests/golden/`).

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
  a collapse from it; `physics.evaluate_jbs` now refuses a zero (failed-trace)
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
* **The structured solver tests and the shipped default both use a pass
  ceiling of 12.** `tests/test_jbs_loop_solver.py` sets
  `jbs_max_passes = _STRUCTURED_TEST_PASSES = 12`, which is now the default
  (raised from 8 on 2026-10-07 with the owner's approval, after the synthetic
  structured case needed exactly 8 passes on the development build). The
  tests therefore show convergence within the default on this build only;
  they do not show headroom on another build.

## Known limitation: a standard draw's post-homotopy re-solve can diverge slowly

In the loop-on regeneration of the earlier D3D-like example golden, two of
twenty draws were
rejected after their coil homotopy: Redl on the delivered equilibrium missed
the bootstrap the draw carried by `r_j` of order 1e-3, the post-homotopy
stage handed the renormalised target to the corrective `jphi-linterp`
iteration, and that solve did not converge. Once the iterate loses its nested
surfaces every nonlinear iteration's surface trace fails and the solver spends
its whole iteration budget, roughly 20-60 min at one thread. The call runs
without `protect_state`, so the loop then measures the broken state, the
`fsa_current_geometry` guard refuses it, and the draw is rejected as
OUT_OF_SPEC with a NaN coil drift. The outcome is always a loud rejection,
never an accepted draw; nothing caps, retries or falls back. The cause (why
these draws, and why the corrective iteration's first solve diverges from a
target rescaled by less than 0.5 %) is not established; the full
investigation is kept in the private validation record.
