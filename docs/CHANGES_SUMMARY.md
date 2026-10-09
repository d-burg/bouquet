# Bouquet — change summaries

## Unreleased — PR #56 (IDA/FUSE ion coupling) and PR #60 (bootstrap options), integrated

### Redl ε: the geometric `(R_max − R_min)/(2⟨R⟩)` by default (`evaluate_jBS/4`, PR #60, owner decision E4)

- **Default physics change, declared.** `evaluate_jBS`'s inverse aspect ratio
  is now `ε = (R_max − R_min)/(2⟨R⟩)` on every OpenFUSIONToolkit build (from
  `sauter_fc(return_eps=True)` where the build has it, else from `get_fsa`'s
  `R_min`/`R_max` over the `sauter_fc` `⟨R⟩`). PR #60 as submitted required
  the fork-only `return_eps` and raised `RuntimeError` on every other build
  (the default engine and the legacy loop could not prepare a baseline); it
  never raises for that now. `ε = ⟨a⟩/⟨R⟩` (versions `/1`-`/3`) is the opt-in
  `eps_definition="a_over_R"`.
- **Version.** `EVALUATE_JBS_VERSION` is `evaluate_jBS/4 (..., geometric eps =
  (R_max-R_min)/(2<R>), ...)`; `/3` keeps its meaning (p′G with the bootstrap,
  PR #64), and an opt-in run records `evaluate_jbs_version("a_over_R")`, which
  names `OPT-IN eps = <a>/<R>`. `diag["eps_definition"]`/`["eps_route"]` per
  evaluation.
- **What moves** (synthetic D3D-like, same equilibrium and kinetics): ν* ×1.27
  at ψ_N 0.1 rising to ×1.72 at 0.98 (`ν* ∝ ε^-3/2`); j_BS +0.2 % core, +2.1 %
  at 0.9, −1.0 % at 0.95, −8.2 % at 0.98; peak −2.0 %; I_BS −0.46 %. Every
  default-path bootstrap (unified engine and legacy loop) changes accordingly;
  goldens that pin `evaluate_jBS` output need regeneration.
  [physics-notes.md](physics-notes.md#the-evaluator-physicsevaluate_jbs).

### Kinetic draws: `kinetic_sampler/2` (PR #56)

- **One sampler for every path** (`bouquet.kinetic_sampler`, version
  `KINETIC_SAMPLER_VERSION = "kinetic_sampler/2 ..."`). A main-ion density
  derived from a Z_eff draw is now an **increment on the baseline**,
  `ni = bl.ni + ni_of(ne_d, Zeff_d) - ni_of(ne, Zeff)`, instead of the
  absolute `ni_of(ne_d, Zeff_d)`.
- **Why:** the absolute form does not return `bl.ni` at sigma = 0 whenever the
  baseline is not single-impurity quasineutral at the median `Z_imp`
  (multi-species or beam p-files, IDA) -- a sigma = 0 violation of the legacy
  path (pinned by
  `tests/test_kinetic_sampler.py::test_sigma0_returns_the_baseline_ni_on_a_non_quasineutral_baseline`).
- **What moves:** legacy draws with the Z_eff channel on (the default
  channel) differ seed-for-seed from `/1`; each profile is now
  `base + (sample - mean) b0` (ULP-level differences on ne, Te, Ti). Engine
  draws are unchanged except where one of the clips below binds.
- **Clips, unchanged in value, now counted per draw** (`KineticDraw.clips`,
  `KineticDraw.record()`; a log line names every clip that fires):
  `zeff_floor_1` (Z_eff lifted to 1 where `zeff_bounds` alone allowed less),
  `zeff_window_lo/hi`, the same three for a passive Z_eff aux draw,
  `ni_floor_0` (thermal n_i floored at 0) and `ni_ceiling` (n_i capped at
  `ne - z_fast`). The assumptions and their literature basis:
  [physics-notes.md, "Kinetic assumptions"](physics-notes.md#kinetic-assumptions-z_eff-n_i-and-the-clips-pr-56).

## Unreleased — the unified engine becomes the default; one current conversion (owner decisions, 2026-10-06)

**Both change results by default.**

Where each number below was measured, and how this branch's chapter commits
map to the original history on the archival tag
`archive/engine-unified-2116923`: [validation-provenance.md](validation-provenance.md).

### Reproducing a run made before this release; what moves on the default path

- **Recipe.** `reconstruction_engine="legacy"` (the legacy reconstruction and
  draws) + `jbs_self_consistent=False` (the frozen SWB bootstrap) +
  `separatrix_pressure="legacy"` (the full axis pressure as the solver's
  target). A stored config that predates these fields gets all three on
  load. Two moves remain that are NOT switchable: the canonical coil-solve
  mode (one bounded coil solve entered at `setup_solver`; measured <= 5e-7
  relative on reconstructions and <= 5e-4 on archived draws, yields and
  in-spec flags unchanged) and the one current conversion below (the frozen
  bootstrap's `_swb_jbs_to_toroidal` is now kappa: -6.4 to -6.8 % at the
  pedestal of the synthetic example). `jbs_self_consistent=False` on its own
  is NOT "bit for bit legacy" (an earlier entry said so; corrected).
- **On the default path (loop on, engine or legacy), not opt-in:**
  - the structured soft closure's **noise-floor acceptance** is on (every
    loop closure call passes `accept_noise_floor=True`); a rounding-level
    effect -- an iterate stationary within the objective's rounding noise is
    accepted instead of raising;
  - the **exact elementary charge** (`physics.ELEMENTARY_CHARGE`,
    1.602176634e-19, -1.46e-5 relative to the 1.6022e-19 the frozen path
    keeps) in the thermal pressure of the reconstruction and the draws;
  - the **g-file anchor's axis-pressure target is the FULL pressure**
    (thermal + fast + impurity), the draws' convention; the frozen path
    inherits SWB's thermal-only target;
  - under the loop the legacy **IMAS q0 corrector and the structured-l_i
    corrector record and flag instead of correcting**: the loop re-solves
    with the axis row and the l_i row held at their targets, and the
    residuals a corrector step would have reduced are measured and flagged
    against `q0_tol` / `structured_li_tol`, not corrected
    (`jbs_loop_q0_corrector=True` makes the q0 pin act per pass; the l_i row
    stays held either way). The unified engine instead has the q0 and l_i
    rows inside its loop.

### At a glance: the review fixes of 2026-10-06 / 07 (one line each; details below or in the commit)

- **sigma=0 gate widened (fix).** The engine draws (the sigma=0 draw
  included) now solve under the reconstruction's OWN coil regularisation
  (they installed the legacy exploratory list, identical only with an empty
  `SolverConfig.coil_reg`); `verify_sigma0_consistency` additionally gates
  |dq0| at the q-row radius (`jbs_tol_q0`), the archived coil drift against
  the hard coil bound and the LCFS rms against the in-spec boundary cut
  (existing bounds; each recorded with its value and setting).
- **Solved-residual criterion (stricter).** With the solved current relaxed
  (`jbs_relax_current` 0.7) `r_j` / `r_I` are measured against the bootstrap
  the pass SOLVED (the relaxer's blend), not the iterate; the iterate's are
  recorded (`r_j_iterate`). Can take more passes within the same ceilings
  (two fast tests that converged only on the iterate are left failing,
  owner decision pending).
- **IDS export from the stored `<j.B>` parts.** Engine draws archive a
  `jB_parallel/` subgroup; the exporter writes it as it is, and subtracts
  the pressure-driven term from older / legacy `j_inductive` (below).
- **IMAS time rule.** The core_sources slice is windowed at half the
  core_profiles step, dt and the bracketing pair recorded, and a driven
  entry whose record starts after the slice is OFF there (zero, stamped
  `off_before_record`, announced) instead of refused (below).
- **`run_slices(on_refusal="record")` is the default** (was `"raise"`).
- **Stored-config replay warns** on missing `jbs_relax_*` keys (loaded at
  the pre-field 1.0 / 1), the loud entries fire for every stored config, and
  `coil_solve_mode` is read back (below).
- **±50 % bootstrap prior flagged on every path.** `|s_bs - 1| > 0.5` is
  flagged `bootstrap_scale_out_of_prior` both ways (only `< 0.5` was), on
  both engine paths (the g-file engine path never called `closure_health`)
  and the legacy g-file path; printed and warned, never clamped.
- **`imas_li3_radius` (new, IDS l_i row).** The source `li_3`'s
  normalisation radius (`R_geo` from IMAS.jl / FUSE vs TokaMaker's
  `R_axis`) is resolved (`"auto"`) or stated, the target rescaled by
  `R_src/R_axis` and the factor recorded; a stored IMAS unified config
  loads as `"axis"`.
- **Engine MSE: Jacobian refresh, no-MSE fallback, chord chi^2/N** (below;
  `mse_chi2n_flag = 10.0`, owner-approved 2026-10-07, flag only;
  `MSE_JACOBIAN_MAX_REFRESHES = 3`, owner-approved 2026-10-07,
  non-converged only).
- **`p_scale` recorded (new record).** The solver's uniform `P'` rescale
  (OFT `p_scale` = `pax / P(psi_axis)`; ~1.02-1.03 on a pedestal with the
  edge pin, the size of the separatrix correction) is recorded on every
  delivered state: the engine record's and the legacy reconstruction's
  `edge_pressure` blocks (so `delivered_state_json`), every
  `edge_pressure_json` (baseline: from that record; each draw: its own
  solved state) and the engine draw records (`delivered` / `archived`).
- **Reconstruction pass ceiling 8 → 12 (owner-approved 2026-10-07).** Under
  the solved-state residual a healthy q0-pinned structured loop needed nine
  passes (pass 8 met every criterion, pass 7 missed `r_I` by 17 %); the
  reconstruction ceiling now matches the draw ceiling. A ceiling is a limit,
  not a tolerance: runs that converge sooner are unchanged.
- **Growth abort (documented).** At the default relaxation the earliest
  `JBS_GROWTH_ABORT_PASSES` abort is pass 10: never within the post-homotopy
  ceiling 6 (nor within the former 8-pass reconstruction ceiling); reachable
  within the reconstruction and draw ceilings of 12 (and from pass 6 with
  `jbs_relax_halve_on = 1`). No value changed;
  `tests/test_jbs_growth_abort_reach.py` pins it.
- **Engine-dependent defaults resolve at `prepare_baseline()` (fix).**
  `isolate_edge_jBS` / `perturb_jind_in_anchor` default to `None` and are
  resolved for the engine configured when the run starts, so setting
  `reconstruction_engine` after construction runs that engine's validated
  values; the resolution is archived (below).
- **What an archive field MEANS changed on the default path** (names and
  units did not). An earlier entry called schema v3 "additive"; it is
  additive in its names only. On an engine archive `j_inductive` / `j_BS`
  are the engine's UNFLOORED split (`j_inductive` the residual, may be
  negative, carries the pressure-driven term; legacy floors it at zero and
  moves the sliver into `j_BS`); `in_spec` is the coil verdict AND the
  post-hoc band (legacy: coil only); and under `separatrix_pressure=
  "offset"` the stored g-files' `PRES` is the solver's pressure + `p_sep`
  (zero at the boundary before). See archive-schema.md.

### Validation of this branch (the owner's cluster, at `1a15685`)

Aggregate numbers only; where each was measured:
[validation-provenance.md](validation-provenance.md).

- **Fast suite:** 2897 passed at `1a15685` (2919 with the tests `3d974e6`
  adds), no failures.
- **Solver suite** (every solver-marked file, one thread, OFT build
  `20260929_7da4f18`): **165 passed, 2 failed, 1 skipped.** Both failures
  are the q0-pinned structured variant of the bootstrap-loop q0-pin test
  (and the comparison that needs its residual): under the solved-state
  residual (`a769882`) that loop needs about one pass more than the former
  reconstruction ceiling of 8 (pass 8 meets every criterion, pass 7 misses
  `r_I` by 17 %, so the two-consecutive rule is not met). Resolved by the
  ceiling change 8 → 12 above (owner-approved 2026-10-07); the suite re-run at
  the tip gives 167 passed, 1 skipped (build-dependent), 0 failed. The skip is a q95
  comparison that needs a specific older OFT build (its identity asserts
  ran).
- **Golden fixture** (synthetic g-file example, 20 draws, seed fixed):
  unified engine 20 attempts, **17 archived, 4 in spec** (coil verdict AND
  the post-hoc l_i band), reconstruction 66 s, draws 124 s per equilibrium;
  legacy golden **20 archived, 12 in spec** (the previous fixture's in-spec
  flags on every draw), draws 458 s per equilibrium.
- **Re-validation against the previous validation** (11 single slices and a
  30-slice time series of the owner's private cases, engine default): all
  converged; quantities agree to 1e-7 -- 2e-4 relative (the series' largest,
  a boundary rms, 3e-4). The deltas are explained by two declared changes:
  the IMAS time rule (`ce10dec`) aligned the driven-current reads with the
  kinetic-profile slice, moving the beam (NBI) input by 8--21 % on four
  slices; the solved-state residual (`a769882`) added one loop pass on
  three. Kinetic inputs are identical.
- **Legacy path:** moved only by the one current conversion (`9f6abed`;
  bootstrap peak -5 to -7 %), and is now within 0.8 % of the engine on
  I_BS/I_p on 10 of 11 slices (3.9--7.3 % apart before). A first legacy
  arm built with the factory and switched to `"legacy"` afterwards was not
  the validated legacy configuration; that trap is fixed (`3d974e6`,
  engine-dependent defaults resolve at `prepare_baseline()`), and the arm
  above was built with the engine named at construction.
- **σ=0:** 11/11 pass the widened gate; the check now takes 50--56 s
  (was 15--18 s) because the σ=0 draw runs the full draw route
  (`eee55bf`).
- **Wall time, engine reconstruction:** 53--113 s on 8 slices, 125--136 s
  on 3 (as before); the time series 52--59 s.
- **Network:** zero network attempts in every re-validation run.

### `reconstruction_engine` default `"legacy"` -> `"unified"`

- **What moves.** `GenerationConfig.reconstruction_engine` now defaults to
  `"unified"`: `Bouquet.from_geqdsk` / `from_imas` (and any config built
  without the field) reconstruct with the unified engine and run the draws on
  it ([engine.md](engine.md)). The factories set none of the legacy-path
  workflow flags (`isolate_edge_jBS`, `perturb_jind_in_anchor`; see the
  engine-dependent defaults below). Reconstructions,
  draws, yields and every recorded quantity follow the engine's definitions
  (composition identity, rows, delivery check; see engine.md).
- **Runtime** (from the engine PR's validation, the shipped synthetic cases):
  engine reconstructions 38–76 s and draws 60–144 s, against 150–660 s and
  405–1407 s on the legacy path with the bootstrap loop on (the legacy loop
  is the cost; with the loop off the legacy path is faster but frozen).
- **Legacy-only settings are refused under the engine** (the 22 fields of
  `engine.ENGINE_UNREAD_LEGACY_FIELDS`, `homotopy_passes` with
  `engine_draw_homotopy=False`, `draw_solve_maxits`, the MSE knobs the engine
  does not read, and the engine's requirements `jbs_self_consistent=True`,
  `recalculate_j_BS=True`, `single_profile_jphi=False`,
  `jbs_init="anchor"`). Every such refusal now ends with how to get the
  legacy paths back. `workflow="custom"` still downgrades the
  unread-field refusals to a printed WARN.
- **How to get legacy back:** `reconstruction_engine="legacy"` -- in the
  factory call (`Bouquet.from_geqdsk(..., reconstruction_engine="legacy")`),
  in `GenerationConfig(...)`, or set on the configuration afterwards; the
  validated legacy workflow flags are applied at `prepare_baseline()`
  either way (below). Its results are those of the legacy path of this
  release, which includes the conversion change below.
- **Engine-dependent defaults resolve when the run starts (fix,
  2026-10-07).** `isolate_edge_jBS` and `perturb_jind_in_anchor` were
  validated with different values on the two engines (legacy:
  `isolate_edge_jBS=False`, and `perturb_jind_in_anchor=True` for an IDS
  source; unified: `True` / `False`, never read). The factories applied
  them at construction, so a factory configuration switched to `"legacy"`
  afterwards ran the legacy paths on the engine's values (an isolated-edge
  bootstrap with a flat core, I_BS/I_p off by tens of percent), silently.
  Both fields now default to `None` (`engine.ENGINE_DEPENDENT_DEFAULTS`),
  the factories set neither, and `prepare_baseline()` resolves them once for
  the engine configured then; the order no longer matters. The resolution
  is recorded (`Baseline.engine_resolved_defaults`, the engine record, the
  archive's `_baseline` attr `engine_resolved_defaults_json`): value and
  origin, `"resolved from engine=<x>"` or `"explicit"`. An explicit value is
  kept; one contradicting the engine's validated value warns naming the
  field (and is refused under the engine, as before). Stored configurations
  carry both fields explicitly and load unchanged. A hand-built
  `GenerationConfig(reconstruction_engine="legacy")` without the fields now
  runs `isolate_edge_jBS=False` (it ran the dataclass default `True`, the
  edge-spike mode); set `True` explicitly for an edge-spike study. Tests:
  `tests/test_engine_resolved_defaults.py`; the factory tests now expect
  `None` at construction.
- **Stored configurations.** A stored config (dict / JSON / an archive's
  `config_json`) that LACKS the field predates the engine (2026-09-29) and
  loads as `"legacy"` with a warning, so it replays the paths it was produced
  with (`BouquetConfig.from_dict`; tested on the five stored fixtures and an
  archive's own config). A stored config that carries the field keeps it.
- **Tests.** Tests of legacy behaviour now construct
  `reconstruction_engine="legacy"` explicitly (construction only; no
  expectation moved). Tests of DEFAULT behaviour now exercise the engine;
  converted expectations: the default is `"unified"`
  (`test_engine.py::test_the_engine_is_the_default_and_old_configs_load_as_legacy`)
  and the factories without the keyword build the engine's configuration
  (`test_engine_refuses_unread_settings.py::test_the_factories_default_to_the_engine`).
- **Owed:** the example notebooks are not re-executed here (their stored
  outputs are legacy-path runs); the golden fixtures are regenerated on the
  engine default separately (with a slim legacy golden).

### One `<j.B>` -> `<j_phi>` conversion in the package (declared physics change)

- **What moves.** Every field-aligned conversion is now
  `kappa = F<1/R>/<B^2>` (`physics.field_aligned_conversion`; the engine's
  `engine.conversion_factor` calls it): `<j_phi> = kappa <j.B>`, the plain
  flux-surface average OFT's `jphi-linterp` consumes, so
  `<j_phi> = kappa<j.B> + p'(<R> - F^2<1/R>/<B^2>)` is exact. The legacy
  bootstrap (`evaluate_jBS`, `_swb_jbs_to_toroidal`) used `<j.B>/(F<1/R>)`;
  the IDS export (`toroidal_to_parallel`) used the `<1/R^2>` form
  `<j.B>F<1/R^2>/(<B^2><1/R>)`. `parallel_to_toroidal` / `toroidal_to_parallel`
  keep their names and are now kappa and its exact inverse; a `geom` carrying
  `avg_inv_R2` is refused by name. `EVALUATE_JBS_VERSION` is `evaluate_jBS/2`.
- **Size.** The old legacy factor exceeds kappa by
  `[<B^2>/<B_phi^2>] x [<1/R^2>/<1/R>^2]` -- the bracket (~1.5 %) times the
  Jensen ratio (~5 %), +6.8 % at psi_N ~ 0.97 on the synthetic example
  (+1.0 % at 0.1, +4.3 % at 0.5, +6.4 % at 0.9). (The docstrings had quoted
  the bracket alone, "1.4 %".) Measured on the synthetic g-file example,
  legacy engine with the loop on: bootstrap peak 563.7 -> 527.8 kA/m^2
  (-6.4 %), I_BS/I_p 0.247 -> 0.236; l_i and Ip still meet their targets.
  The unified engine is unchanged by construction: its reconstruction of the
  same example is bit-identical before and after.
- **Not switchable** (one conversion is the point). The IDS export's
  `j_bootstrap` / `j_ohmic` / `j_total` now invert exactly what bouquet
  converts in with.

### IDS export: no pressure-driven current in any parallel field (fix)

- **What was wrong (pre-dates the engine).** `write_imas_draw` converted the
  archived toroidal `j_inductive` -- the residual `j_phi - j_BS - fixed`,
  which carries the pressure-driven term `P = p'(<R> - F^2<1/R>/<B^2>)`
  (engine draws by construction; legacy draws freeze it there) -- to
  `<j.B>/B0` as it was, so `P/kappa` sat inside the exported `j_ohmic` and
  `j_total`; the engine's `IdsAdapter` re-adds `P` from the pressure, so a
  re-read counted it twice (+0.3 / +3.9 / +8.6 / +10.4 / +17.7 % of `j_phi`
  at psi_N 0.5 / 0.9 / 0.95 / 0.97 / 0.99 on the synthetic example).
- **Now.** Engine draws archive their `<j.B>` parts (`jB_parallel/`, schema:
  `jB_inductive` the field-aligned inductive only, `jB_BS`, `jB_NBI`,
  `jB_RF`, with `kappa` and `j_pressure`); the exporter writes them as they
  are. Draws without them (legacy; engine archives before this change) have
  `P` -- from the archived eqdsk's own flux surfaces,
  `io.imas.archived_pressure_term` -- subtracted from `j_inductive` before
  the conversion. `j_total = j_ohmic + j_bootstrap + driven`. Export ->
  `IdsAdapter.read` returns the archived parts and `<j_phi>` to 1e-9
  (`tests/test_imas_export_roundtrip.py`, engine and legacy archives).
  The exported `j_ohmic` / `j_total` change by `-P/kappa` (as above); `j_tor`
  and `j_bootstrap` do not.

### IMAS time matching: the core_sources slice is windowed, dt recorded, OFF before an entry's record (owner decision 2026-10-06)

- **The core_sources slice itself.** It was `_nearest_index(core_sources.time,
  T)` with no window: a single-time core_sources was index 0 whatever the
  time (the synthetic example reduced to its 2.1 s core_sources slice read
  its 2.1 s beam at 2.3043 s, with no trace). Now (both readers,
  `io.imas.core_sources_slice`): the core_sources time nearest the
  core_profiles slice READ, within half the local core_profiles step of it
  (a single-time core_profiles uses the core_sources step; two single-time
  bases, the 10 us floor below), else refused naming both times.
- **Ten-microsecond floor with no time step (owner-approved 2026-10-07).**
  When neither time base has a local step (a single-time entry on a
  single-time core_profiles, or single-time core_sources and core_profiles)
  the window is `io.imas.IMAS_SINGLE_TIME_WINDOW_S = 1e-5` s; it was a few
  float ulp, so an entry 2 us after the slice was off before its record and
  one 2 us before it was refused. Rounding-level mismatches of
  millisecond-stored times are now matches with their dt recorded; the
  off_before_record and refusal rules apply only beyond 10 us. With a local
  step on either base nothing changes.
- **Entries.** The match stays half the entry's OWN local step, and the
  matched own slice must ALSO lie within half the local core_profiles step
  of the core_profiles slice time: a coarse own grid read between its
  samples is refused (it read the nearest sample, e.g. 0.1 s away, before).
- **OFF before an entry's record.** A driven entry whose own record starts
  AFTER the slice time (and whose first own slice carries current; an idle
  one is off by the bracketing rule, as before) is OFF there: zero, stamped
  `off_before_record` with its first own time, announced once per source
  file and entry (print + `UserWarning`). It was refused. Past its LAST own
  time an entry carrying current there is still refused. Never interpolated
  (the review's "interpolate inside the range" is rejected).
- **Recorded.** `Baseline.source_time_match` (archived in the baseline's
  `li_metrics`) and the engine's `provenance["source_time_match"]`: both
  slice times, dt (core_sources minus core_profiles), the window and its
  basis, and per entry its matched own time, dt (own minus core_profiles),
  both windows, the bracketing own times, first / last own time and status
  (`matched`, `off_before_record`, `off_idle`, `zero`). `driven_sources`
  entries carry `matched_time` and `dt`.
- **Tests whose expectation moved (declared):** the entries starting a step
  late WITH current (adapter and reader) and a single-time entry 0.1 s or
  2 us after the slice are now `off_before_record` (were refused; the
  refusal is now checked past the entry's end); the coarse-grid NBI entry
  read between its samples is refused (was read).
  `tests/test_imas_time_rule.py` is new.
- **Note for the owner.** With two single-time bases the window is float
  precision, so an entry stored a few microseconds AFTER the slice is now
  off before its record (zero, announced) where it was refused; a few
  microseconds before it is still refused.

### `run_slices(on_refusal=...)` default `"raise"` -> `"record"` (owner decision 2026-10-06)

One refused baseline (a closure refusal, a negative separatrix pressure, a
time-matching refusal) ended a whole series. Now the refused slice is
recorded -- the summary carries `refused=<reason>` and its `time`, the
archive `scan/<key>` carries `refused_reason` and `refused_time` (moved to
`refused_time_superseded` when a later run writes the slice) -- and the sweep
continues; after the last slice the count and the reasons are printed and
warned once. `on_refusal="raise"` keeps the old behaviour. Not covered (as
before): a refusal raised inside `generate()` (e.g. an engine-draw context
refusal) still ends the sweep unrecorded.

### Stored-config replay is loud (2026-10-06 review, D3)

- **`jbs_relax_current` / `jbs_relax_halve_on`.** A loop config stored
  2026-09-25..27 lacks both (introduced 2026-09-27 at 0.7 / 3); it ran with
  no current relaxation and omega halved on every growth, and loaded
  SILENTLY at 0.7 / 3. Now a config that ran the loop and lacks either loads
  with 1.0 / 1 (`FIELD_PRE_INTRODUCTION`) and ONE warning naming the fields;
  with the loop off they had no effect and are not back-filled.
- **The loud entries fire for every stored config.** `FIELD_PRE_INTRODUCTION`
  was consulted for stored unified configs only, so its loud entries
  (`l_i_tolerance`, `jBS_scale_range`, `homotopy_passes`, `floor_j_BS`,
  `jbs_max_passes_draw`) never fired for a config bouquet itself stored
  (all legacy before the engine). Now: for every stored config; the
  engine-only entries (`engine_ids_inductive`, `engine_mse_jacobian`) for
  unified ones; the loop's own (`jbs_max_passes_draw`, the relax fields,
  `jbs_max_passes_post_homotopy`) for a config that ran the loop (or a
  unified one).
- **`coil_solve_mode` is read back.** `generate()` stamps it on `_baseline`
  (both paths; `utils.stamp_coil_solve_mode`; the engine record already
  carried it); `load_config` reads it (`utils.load_coil_solve_mode`: the
  attr, else the baseline engine record) and warns when the stored run
  records none (it predates the canonical mode) or another one -- a replay
  runs bounded; measured <= 5e-7 relative on reconstructions and <= 5e-4 on
  archived draws.

### Engine MSE: the Jacobian re-taken at convergence; the no-MSE fallback; chord chi^2/N (2026-10-07 review, item 6)

- **Jacobian refresh (changes the delivered MSE fit).** The engine's MSE stage
  took the chord Jacobian once at the no-MSE convergence and refreshed only
  the offset, so it stopped at `J0^T W r + grad(prior) = 0`, not at the chi^2
  minimum (the legacy chord stage re-takes it). Now, at each MSE loop's
  convergence, the Jacobian is re-taken by the same finite differences
  (1 + n_free solves, `solves["mse_refresh"]`) and the closure's
  Gauss-Newton step with it is judged by the stage's own criterion
  (`MSE_CHORD_OFFSET_TOL_SIGMA`, 0.1 sigma_eff, unchanged); above it the loop
  continues with the refreshed Jacobian (same pass ceiling), at most
  `engine.MSE_JACOBIAN_MAX_REFRESHES = 3` refreshes (a NEW cost ceiling,
  owner-approved 2026-10-07: non-converged only -- it can only make a stage
  non-converged). Recorded:
  `jacobian_refresh_rel_change`, `refresh_step_norm`, the old-J step and the
  predicted tan(gamma) moves (MSE phase `jacobian["refresh"]`). Within the
  criterion the delivery is the one before; on the toys the step is < 0.01
  sigma (17-19 % Jacobian change), and 0.3 sigma -> 0.003 sigma after one
  continuation on a toy with a non-linear pitch-angle response. The record
  now names the Jacobian method in use (it said "Broyden" on `fd_chord`).
- **No-MSE fallback (fix).** A failure inside the MSE stage (or in the
  delivery of its fit) set `bq.baseline = None` and re-raised even with
  `structured_mse_required=False`, discarding a converged reconstruction.
  Now the converged reconstruction without MSE is restored
  (`solver_state.SolverState` + the engine state), re-delivered and
  re-checked, and flagged `mse_stage_failed` with the exception text
  (closure-limited reason, print, `RuntimeWarning`,
  `Baseline.engine["mse"]["failure"]`). `structured_mse_required=True` still
  raises.
  The failed phase keeps the Jacobian record the stage took (`n_free`, the
  FD's `n_solves`, scheme, Broyden updates, refresh rounds;
  `jacobian_taken`, `n_solves_spent`); it used to be reduced to
  `{applied, failed, where, reason, n_solves}`, so a reader of `n_free` on
  a stage that failed in its passes got `KeyError`
  (`tests/test_engine_mse_fallback.py::
  test_a_failed_mse_stage_keeps_the_jacobian_record_it_took`).
- **Chord chi^2 records and two flags (flag only, never acceptance).**
  Against the raw E_r-corrected chords with the stage's own weights:
  delivered chi^2, N, chi^2/N and the no-MSE reconstruction's chi^2
  (`checks["mse"]`, `Baseline.engine["mse"]`); `mse_worse_than_without`
  when the delivered chi^2 exceeds the no-MSE one; `mse_chi2_per_chord_high`
  when chi^2/N exceeds the NEW `GenerationConfig.mse_chi2n_flag` (default
  10.0 -- owner-approved 2026-10-07, flag only; refused non-default under
  `"legacy"`). Under `jbs_loop_on_fail="flag"` a non-converged MSE fit is
  delivered with `mse_converged=False` and the same records.
- **Tests.** `tests/test_engine_mse_refresh.py`,
  `tests/test_engine_mse_fallback.py` (mutants: no continuation leaves the
  fresh-J step above the criterion; the code before the fallback raises and
  leaves no baseline). Expectation moved (declared):
  `test_engine.py::test_a_stated_orientation_the_data_contradict_is_kept_and_flagged`
  -- the closure refusal that test provokes is now a failed MSE stage, so it
  falls back (stated orientation still kept and flagged, on
  `engine["mse"]["orientation_stated"]`). The deliberately failing
  `test_ip_li_and_mse_converge[fd_broyden-True]` (owner decision pending)
  now fails at its record assertions: its MSE loop's non-convergence falls
  back to the no-MSE state instead of raising.

## Decisions on record (owner, 2026-10-05) -- no value changes

Approvals given on 2026-10-05 for settings that were already in force but had
no recorded approval. Nothing below changes a value; it records that the value
in force is approved.

- **Draw-loop pass ceiling `jbs_max_passes_draw` 6 -> 12: approved.** A pass
  LIMIT, not a tolerance (convergence is still every active criterion on two
  consecutive passes). Measured need: in the passes-to-convergence study (six
  12-draw batches, 72 draw attempts, run with the ceiling raised to 30 for
  diagnosis only) the draw loops needed 4-10 passes -- 4: 3, 5: 12, 6: 21,
  7: 11, 8: 13, 9: 9, 10: 3 attempts -- so 25 of 72 needed more than the old
  ceiling of 6, and 12 was never reached; on the synthetic g-file example the
  standard draw's l_i-match coupling contracts at about 0.38 per pass and
  needs 7-8 passes.
- **The bootstrap loop's noise-floor acceptance factor
  (`utils.NOISE_FLOOR_FACTOR = 2.0`) and its relaxation halve-on rule
  (`jbs_relax_halve_on = 3`: omega halved only after r_j grows on 3
  consecutive passes, floor 0.25): kept as implemented.**
- **The MSE closure's defaults: kept as implemented** --
  `structured_mse_fd_step = 0.02`, `structured_mse_steps = 1`,
  `structured_mse_sigma_sys = 0.0`, `structured_mse_min_chords = 4`, and the
  MSE convergence tolerance of 0.1 sigma per chord
  (`jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA`).
- **The engine draws' l_i band (`l_i_tolerance = 0.05`, +/-5 % relative to
  the delivered reconstruction's l_i, applied post hoc): confirmed.**
- **Stored unified configurations carrying the factory values of
  `isolate_edge_jBS` / `perturb_jind_in_anchor`: accepted as implemented**
  (6e052d0, `config.STORED_UNIFIED_UNREAD_FIELDS`). A stored `"unified"`
  configuration carrying a non-default value of either field (what the
  factories set for the legacy path until 2026-10-05) loads with that field
  at its default and a warning, because the engine never read either field
  -- the defaults are what that run actually ran. A live configuration
  switched to `"unified"` with non-default values is still refused; only the
  stored-configuration load path relaxes to defaults-plus-warning, so
  archives written before the refusal stay loadable.

## Unreleased — second-pass review fixes (2026-10-06)

Fixes for the second adversarial review of the engine fix commits
(d874822..f36b03a). Nothing here changes a default value, a tolerance or a
test bar.

- **Stored unified configurations load again when they carry any
  engine-unread legacy field.** 3779b51 made the engine refuse 20 legacy-path
  fields it never reads (`closure_channel`, `structured_li_target`, ...); a
  stored `"unified"` configuration written before then with one of them at a
  non-default value could no longer be loaded. The stored-load rule that
  6e052d0 gave `isolate_edge_jBS` / `perturb_jind_in_anchor` now covers all
  22 fields of `engine.ENGINE_UNREAD_LEGACY_FIELDS`, plus `homotopy_passes`
  stored with `engine_draw_homotopy=False`. On `BouquetConfig.from_dict`
  only, such a value is loaded as the field's default with a warning naming
  it. The engine ignored the value, so the default is what the stored run
  used. A NEW configuration with a non-default value is still refused.
- **Beam and driven-current entries: matched by nearest time within half a
  step, otherwise refused** (owner-recorded 2026-10-06: the owner approved
  the stricter failure mode; the window value is to be confirmed). Since
  807fd93 / 1ff7ccc a `core_sources` entry with its own per-slice times was
  matched to the slice time within an absolute 1e-6 s. A beam entry a few
  microseconds off the time base was dropped to ZERO with a warning (legacy
  reader) or a provenance stamp (engine adapter): its current moved silently
  into the inductive residual and into every draw. The rule now, in both the
  legacy reader (`io.imas._source_slice_at`) and the engine adapter
  (`adapters._ids_source_slice`), sharing `io.imas._entry_time_window`:
  - match the entry to the NEAREST of its own times;
  - accept the match within HALF the local time-step of the entry's own
    grid (the interval the slice time lies in, or the end interval past
    either end). A single-time entry uses the core_profiles grid's local
    step. With no step on either grid the window was float precision
    (10 us since 2026-10-07, `IMAS_SINGLE_TIME_WINDOW_S`);
  - otherwise REFUSE: `ValueError` from the reader, `EngineInputRefused`
    from the adapter. The error names the entry and its index, the slice
    time, the nearest own time, |dt| and the half-step.

  An entry carrying no non-zero current has nothing to drop and is skipped.
  Refinement (2026-10-06): "carries current" is judged on the entry's own
  slices BRACKETING the slice time (its nearest own slice on each side; only
  the nearest end slice when the time lies outside its range,
  `io.imas._entry_bracketing_slices`), not on its whole history. An entry
  with no current there is OFF at that time, not missing: it contributes
  zero, is not warned about, and the adapter stamps it in
  `provenance["off_sources"]` (a model's sawteeth entry whose grid starts a
  step after the IDS time base, idle for its first slices, was refused at
  the first two times). An entry carrying current on a bracketing slice is
  still refused.
  An aggregate or bootstrap-like entry is never added, so it is still only
  stamped in `provenance["ignored_sources"]`. The legacy reader's sawteeth
  entry feeds a gate flag, not a current: it uses the same window and is
  recorded "not active" when there is no match, not refused.
  **What changes:**
  - an entry inside the time span of its own grid is always matched now:
    the review's 2 µs case reads the beam bit-identically to the on-grid
    read, where it was zero before;
  - a slice time more than half a step outside the entry's span (for
    example a model's entry that starts later than the IDS time base, read
    at an earlier slice) is refused where it was zeroed before.
  The shipped example reads bit-identically: its beam entry has no
  per-slice times and is read by index.
  **Not measured:** the effect on the real-data FUSE testbeds. A FUSE
  source whose sawteeth or beam entry starts after the first time-base
  slice will now refuse at those early slices on the engine path. That
  needs an owner check on real data, offline from this package.

- **Disclosed: a negative separatrix pressure also refuses the BASELINE.**
  c709aae refuses a negative pressure at psi_N = 1 under
  `separatrix_pressure="offset"`. Its commit message and this summary named
  only the legacy draws (rejected as `perturb_failed`), but the refusal is
  in the shared `edge_pressure.applied_offset`. It therefore also stops
  `prepare_baseline()` when the input's own pressure at the separatrix is
  negative, for example a fit that dips below zero at the edge. That
  applies on the default legacy path (the g-file reconstruction and the
  IMAS forward solve) and on the unified engine. Before c709aae such an
  input ran, with a RAISED axis-pressure target and a negative boundary
  PRES. **Kept by the owner's rule:** failures are loud, and a negative
  p_sep is unphysical input. The refusal is now named:
  - the error is `edge_pressure.NegativeSeparatrixPressure`, a
    `ValueError`;
  - `prepare_baseline` re-raises it as "prepare_baseline REFUSED THE
    BASELINE (source ..., reconstruction_engine=...)", saying why and how
    to proceed: correct the edge profiles, or set
    `generation.separatrix_pressure="legacy"`, which never reads the edge
    value, to build it as before 2026-10-04.

  Test: `test_a_negative_separatrix_pressure_refuses_the_legacy_baseline`
  (a legacy IMAS baseline; refused before any solve).

- **Tests that pin claims which had none.** A mutation pass over the fix
  commits found three claims that no test could fail.
  - `test_an_archived_stage_miss_fails_the_sigma0_check`: a loop stage
    that passes but an archived state that misses must fail the sigma=0
    check.
  - `test_the_archived_split_uses_the_archived_states_kappa_and_redl`:
    the archived bootstrap and fixed parts are on the archived state's
    `F<1/R>/<B^2>` and Redl, so a 1 % error in either fails.
  - `test_after_a_failure_the_next_secant_step_uses_two_good_points`: a
    behavioural test of the l_i secant's pairing after a failed solve.
  The pass was re-run on this tree. 24 of the review's 27 mutants still
  apply; the other 3 targeted code that later commits replaced, and they
  are re-expressed among 8 new mutants. All 32 are killed.
- **What a written g-file contains, parsed back.**
  `test_a_written_gfile_parses_back_to_the_delivered_frame` (solver-free)
  checks:
  - edge `PRES` is the delivered p_sep under `"offset"` and 0 under
    `"legacy"`;
  - `PPRIME`, q, F and the boundary are unchanged within the format's
    precision.

  Its live-solver twin, `tests/test_gfile_written_contents_solver.py`
  (`-m solver`, 8 tests), runs the existing probe and checks that `PRES`
  minus a bare save of the same state is p_sep at every point. The bare
  edge is the solver's pressure on the truncated last surface, 4.8 Pa on
  the example, not zero. The twin has not been run yet: it is owed on the
  next solver-suite run. The solver suite grows from 158 to 166 tests.
- **Not changed, recorded** (the review's partial-close notes):
  - a negative residual inductive current on an IDS source is RECORDED
    (`residual_negative_nodes`), not refused, by the owner's rule;
  - an unknown `core_sources` identifier index is held fixed as a driven
    current under "other", with a warning.
  - `test_get_q_collapses_silently_on_an_unclipped_grid` now skips by its
    documented build-dependent rule (see below). It is on the pending list
    for retirement and is not retired here.

## Unreleased — owner-approved decision (2026-10-06)

- **One coil solver for the whole run: OpenFUSIONToolkit's bounded coil
  mode is entered ONCE, at solver setup, on both paths** (owner-approved
  2026-10-06; consistency between the reconstruction and the draws).
  OpenFUSIONToolkit solves the coil least-squares problem by the normal
  equations until `set_coil_bounds` is first called and by bounded least
  squares (BVLS) from then on, for the life of the solver object; every
  `generate()` makes that call (its homotopy). So a process's reconstruction
  and its first `generate()`'s baseline re-solve ran unbounded and every
  later solve bounded: results depended on call order (the first engine
  sigma=0 check differed from the second by 3.8e-7 in the inductive
  amplitude until the check entered the mode itself, below). Now
  `Bouquet.setup_solver` calls `bouquet.solver_state.enter_bounded_coil_mode`
  (the one place: after the VSC and the coil regularisation, before any
  solve), installing +/-1e98, which never binds. The reconstruction, the
  sigma=0 check and every draw run the same coil solve;
  `Baseline.coil_solve_mode` (both paths) and the engine record's
  `coil_solve_mode` say `"bounded"`. The sigma=0 check's own entry is
  removed -- a no-op on a solver set up this way; its restore still
  re-installs the recorded bounds. **What moves:** everything computed in
  the old unbounded mode -- the reconstruction and the first `generate()` of
  a process -- at the round-off-carried level (first-call results are no
  longer bit-identical to before). Measured on the synthetic examples (fixed
  build, one thread, old -> canonical): reconstruction l_i(3) and q0 by
  2e-8 to 5e-7 relative (l_i(3): g-file engine -2.0e-7, legacy +3.1e-7;
  IDS engine +4.8e-7, legacy +2.0e-8), q95 by 2e-6 to 1.3e-5, coil currents by at most 2.1e-5 relative; every sigma=0 check
  (both paths, both sources) still passes, its residuals moving within the
  same decade (largest ratio to a tolerance 0.31, IDS engine r_I). No
  tolerance, bar or default changed.
  The draws inherit the moved reconstruction and stop at the loop's own
  tolerances, so they move further, but within those tolerances: on the
  four seeded 12-draw engine batches of the synthetic examples (inductive
  sigma 0.10 and 0.05) the yields, every in-spec flag and every loop,
  homotopy and post-homotopy pass count are unchanged, and the archived
  l_i(3), l_i(1), beta_N, q0 and q95 move by at most 5e-4 relative. The
  golden replay stays inside its bars without regeneration (mode-1 coil
  drift 0.0190 -> 0.0191 %, boundary RMS 0.419 -> 0.421 mm). One solver
  test's documented build-dependent skip now fires:
  `test_get_q_collapses_silently_on_an_unclipped_grid` asks whether the
  solved state reproduces the axis collapse of an unclipped surface grid,
  and on the canonical state it does not (257/257 surfaces traced); the
  clipping it guards stays, and its guard is still covered without a
  solver.

## Unreleased — owner-approved decisions (2026-10-05)

- **Legacy draws: a failed homotopy rollback re-solve rejects the draw**
  (`homotopy_rollback_failed`, the engine draws' code), whatever the cause
  and whether or not `draw_solve_maxits` is set. Before, a legacy draw
  printed "rollback re-solve failed; stats may be stale" and went on from the
  failed solve -- the post-homotopy check measured that state and the draw
  could be archived, in spec or not. **This changes legacy yields** in runs
  where such a re-solve failed (those draws are now rejected attempts, never
  archived or counted toward until-N); runs in which every rollback re-solve
  converged are unchanged.
- **Engine sigma=0 check: it puts ALL solver state back, and a second check
  is bit-identical to the first** (owner-approved fix of the restore,
  2026-10-05). `verify_sigma0_consistency` under `"unified"` restored psi
  and the isoflux only; the first call then differed from every later one
  (3.8e-7 in the inductive amplitude on the synthetic g-file example, Δl_i
  9e-6 -- 30-100x inside every tolerance, but not bitwise). Cause, measured:
  OpenFUSIONToolkit's coil solve switches from the normal equations to
  bounded least squares on the first `set_coil_bounds` call and never
  switches back (`set_coil_bounds(None)` installs +/-1e98 and stays bounded);
  the route's `generate()` makes that call (the homotopy's bounds), so only
  the first check ran unbounded. It also left the strong coil-regularisation
  stash (`_strong_coil_reg`) on the solver object. Now one helper,
  `bouquet.solver_state.SolverState`, captures and restores the equilibrium
  object (psi, coils, coil regularisation, targets, profiles, constraints),
  the settings, VSC gains, Vcoils, the recorded coil bounds and bouquet's
  stashes, and enters the one-way bounded mode at capture (re-installing the
  bounds on record -- none: +/-1e98, which never binds; since 2026-10-06 the
  mode is entered at solver setup instead, above). **What moves:** the
  check's own numbers, once, to what the second call already gave (the
  first call now runs in the mode every later solve runs in); nothing else
  -- `generate()` itself and the legacy path are unchanged (the legacy check
  calls `set_coil_bounds` only to swap a bound stash a previous `generate()`
  left, and leaves no stash of its own; its twins were bitwise).
- **Engine draws: the archived current split is evaluated on the draw's own
  archived (final) state, and the inductive is never clipped**
  (owner-approved, engine only, 2026-10-05). Before, an engine draw's
  archived `j_BS` was the bootstrap composed on the geometry of the PREVIOUS
  solve, its fixed beam/RF parts were the RECONSTRUCTION's (at its
  `F<1/R>/<B^2>`), and `j_inductive` was clipped at zero twice (post-homotopy
  re-split and archival) with the sliver moved into `j_BS`. Now: `j_phi` the
  archived state's achieved FSA current (as before); `j_BS` = `s_bs (1 +
  d_bs) x scale x Redl` of the archived state times its `F<1/R>/<B^2>`; the
  fixed parts the contract's `<j.B>` times the same factor; `j_inductive` the
  residual, never clipped -- a negative value is RECORDED
  (`engine.archived.split`: `n_negative_inductive`, `min_inductive`,
  `negative_inductive_psi_N`, plus a console note), not altered and not
  filtered. The post-homotopy re-split uses the draw's own solved fixed
  parts, unclipped. The draws' reference flux range (and q-row radius) is
  the DELIVERED measurement's, not the last loop pass's geometry `G*`, so a
  zero-perturbation draw's flux-range delta measures only its own
  reproduction. **What moves:** archived `j_BS` / `j_inductive` of engine
  draws (on the 16 stored synthetic draws: no clip sliver was ever active;
  fixed-part conversion <= 0.012 % of peak; j_BS within the final
  post-homotopy r_j, 5e-5 ... 1.8e-4) and the engine σ=0 flux-range delta.
  No solve, l_i, q, beta, coil or in-spec verdict changes. Legacy draws keep
  their clips exactly as they are.
- **Legacy IMAS reader: the sawtooth gate reads the sawteeth entry at the
  slice TIME** (the engine IDS adapter's rule since its own fix). The gate
  input `Baseline.sawtooth` (`present` / `j_par_max_abs` / `active`, archived
  as `li_metrics["sawtooth"]`) read the `core_sources` sawteeth entry (701)
  at its LIST index: an entry that starts one slice after the IDS time base
  (as a model's sawteeth entry can) was read one slice late at every slice,
  and at the last slice from its FIRST slice. Now an entry carrying per-slice
  times is matched by time; at a time it does not cover it is present but
  NOT active; an entry with no per-slice time and a different slice count is
  refused (`ValueError`, "cannot be aligned") instead of read at slice 0. A
  new key `slice` records how it was read. **What moves for existing users:**
  only dds with such a late-starting (or misaligned) sawteeth entry -- the
  `sawtooth_bootstrap` gate's `active` flag and `j_par_max_abs` at each
  slice (a slice before the entry starts is no longer admitted as
  sawtoothing; the last slice now reads its own amplitude), and through
  `Baseline.sawtooth` the engine's q0-row admission on the same sources.
  Dds whose sawteeth entry is on the full time base, or that have none (the
  shipped example), read the same values as before. The NBI read just above
  it was fixed the same way afterwards (next item).
- **Legacy IMAS reader: each NBI entry is read at the slice TIME** (the same
  defect, the same rule as the sawteeth entry above; owner-approved
  2026-10-05). The beam current `Baseline.j_NBI` summed the `core_sources`
  NBI entries (identifier 2) at their LIST index (`pr[isrc]`, or `pr[0]` past
  the end of a short entry). Now an entry carrying per-slice times is matched
  by time to the core_sources slice time; at a time it does not cover it
  carries no current at that slice (a `UserWarning` says so); an entry with
  no per-slice time and a different slice count is refused (`ValueError`,
  "cannot be aligned"). **What moves for existing users:** only dds whose
  NBI entry's own slice times are not the core_sources time base -- an entry
  that starts late (before its start: no beam current instead of the next
  slice's; afterwards its own slice instead of one late; past a short
  entry's end no longer its first slice) or one written on another grid
  (e.g. the core_profiles times where those differ from core_sources': now
  matched, or warned and zero where nothing matches). There `j_NBI` changes,
  and with it the legacy path's fixed beam current, the inductive residual
  and every draw built on them. An NBI entry with no per-slice time and a
  different slice count is now refused. Dds whose NBI entries are on the full
  core_sources time base, with or without per-slice times (the shipped
  example), read the same values as before, bit for bit.
- **Engine: `isolate_edge_jBS` and `perturb_jind_in_anchor` are refused
  under `reconstruction_engine="unified"`** when not at their defaults (the
  rule and message of the other unread legacy settings); the engine never
  read either. `Bouquet.from_geqdsk` / `from_imas` take a new keyword,
  `reconstruction_engine` (default `None` = the config default, `"legacy"`):
  with `"unified"` they no longer set these legacy-path workflow values. **A
  legacy factory configuration switched to `"unified"` afterwards is now
  refused** (build it with the keyword instead, or reset both fields); the
  message says so. Stored unified configurations carrying the factory values
  load at the defaults with a warning (the stored run is unchanged). Legacy
  configurations, and what the factories set for them, are unchanged.

## Unreleased — review fixes to the unified engine (2026-10-04)

- **Legacy g-file reconstruction (DEFAULT path): a failed last l_i-secant
  solve hands on the restored state's inductive factor.** When the secant's
  last solve failed, psi was restored to the last good state but the FAILED
  `ind_factor` was handed on (as the inductive profile the corrective
  iteration starts from). It is now the factor of the state actually held,
  with that state's profile restored too, and the secant's previous point is
  always a good evaluation. Runs whose secant solves all converge are
  unchanged (tested against the frozen copy); runs with a failed last secant
  solve change.
- **Engine draws: a failed homotopy rollback re-solve rejects the draw**
  (`homotopy_rollback_failed`), whatever the cause and with or without a cap.
  Before, only a capped one did; any other failure printed "stats may be
  stale" and the draw went on (and could be archived) from a failed solve.
  Legacy draws unchanged.
- **Engine IDS adapter: driven currents by an explicit IMAS identifier
  classification.** It held EVERY `core_sources` entry except ohmic (7) and
  bootstrap (13) as a fixed driven current, so a "total" entry, a combination
  entry (100-107) or a bootstrap published as "neoclassical" (401) would have
  been counted twice (the residual inductive current goes negative). Now:
  driven primaries by index (nbi; ec/lh/ic; fusion, runaways, sawteeth);
  aggregates and bootstrap-like entries ignored, stamped in
  `provenance["ignored_sources"]` and warned about; an unknown index held
  fixed under `other` with a warning. Also: an entry carrying its own
  per-slice times is read AT the slice time, not at its list index -- a model
  sawteeth entry that starts one slice after the IDS time base was read one
  slice late (and at the last slice, from its FIRST slice). **This changes
  engine IDS results on sources with such a sawteeth entry** (FUSE
  `dd_sim.json`: the sawteeth current of the slice itself instead of the
  next one's); sources whose entries are all nbi/ec/lh/ic on the full time
  base are unchanged bit for bit. (When this entry was written the legacy
  reader was unchanged. Since then it reads its sawteeth entry (d381951)
  and its beam entries (807fd93) at the slice time too. Since 2026-10-06
  both the reader and this adapter match an entry to its nearest own slice
  within half a local time-step and REFUSE a driven entry outside that
  window that carries current on its own slices bracketing the time; see "second-pass review fixes (2026-10-06)" above.)

- **Engine: legacy settings it never reads are refused** (they were accepted
  and silently ignored): `closure_channel`, `jBS_baseline_mode`, the
  `structured_*` closure fields (except `structured_li_tol`),
  `anchor_pressure_to_equilibrium`, `imas_corrective_jphi`,
  `jbs_loop_q0_corrector`, `floor_j_BS`, `swb_iterations`,
  `accept_anchor_inband`, `diagnostic_plots`, and `homotopy_passes` with
  `engine_draw_homotopy=False` -- each message names the engine setting that
  replaces it. Defaults and the factories' configs are unaffected. Legacy
  path unchanged.

- **Engine: `verify_sigma0_consistency()` runs the draw's own route.** It
  ran only the engine loop (`engine_draws.verify_zero_perturbation`) from
  whatever state the solver held -- not the warm start, coil
  regularisation, isoflux re-point, homotopy and post-homotopy stage a
  `generate()` draw runs -- while the docs said it ran "exactly this draw".
  It now calls `generate(n=1)` itself with every perturbation zero and the
  bootstrap scale 1.0 (into a temporary archive) and judges both the loop
  stage and the archived state at the unchanged loop tolerances
  (`stages`); a rejected zero-perturbation draw fails. On the stand-in the
  numbers are unchanged (no coils to move). On the live solver (rounds 2
  and 3 of the 2026-10-05 fix work, synthetic examples) the generate()
  route PASSES on every path and source; the worst residual-to-tolerance
  ratio is 0.30. After the canonical coil-solve mode (2026-10-06) it still
  passes, with a worst ratio of 0.31. The archived-stage gate is pinned by
  `test_an_archived_stage_miss_fails_the_sigma0_check` (2026-10-06).
- **Engine: an MSE row skipped by a non-converged loop is flagged or
  refused.** Under `jbs_loop_on_fail="flag"` a loop that did not converge
  delivered with no MSE stage and no word about it; now an `MSE: ... NOT
  applied` flag and phase record, and `structured_mse_required=True`
  refuses. A flagged engine IDS baseline is also warned about (as a g-file
  one already was).
- **Edge pressure: loud where it was silent.** A negative separatrix
  pressure is refused under `"offset"` (it raised the axis target and wrote a
  negative boundary PRES -- reachable by a legacy draw whose perturbed edge
  n_e or T_e goes below zero: such a draw is now rejected instead of
  archived); `plot_input_vs_recon` shows the solved pressure in the reported
  (full) frame; a failed full-frame computation warns that beta / W_MHD stay
  in the solver frame; changing `separatrix_pressure` / `edge_pprime_pin`
  after `prepare_baseline()` is refused at `generate()` /
  `verify_sigma0_consistency()`.
- **Stored configs load as they were produced.** A legacy config written
  while `engine_mse_jacobian="fd_broyden"` or `engine_ids_inductive="auto"`
  was the default (to_dict writes every field) was refused by `from_dict`;
  it now loads (the value has no effect on the legacy path; loaded as
  today's default, with a warning). A stored unified config that predates a
  field whose default changed is loaded with the value it ran with where
  that is knowable (`engine_ids_inductive` -> `"auto"`;
  `engine_draw_solve_maxits` -> the `draw_solve_maxits` the engine draws
  read then; a loop config without `jbs_max_passes_post_homotopy` -> 2),
  with a warning, and with today's default plus a loud warning naming the
  field where it is not. Fixtures: the to_dict output at five commits of
  the engine stack (`tests/data/stored_configs/`).

## Unreleased — `engine_ids_inductive` (unified engine, IDS sources)

*The default is now `"residual"` (owner-approved; see "IDS inductive: the
parallel residual by definition" in the unified-engine section below). As
introduced, the setting defaulted to `"auto"`, as described here.*

New `GenerationConfig.engine_ids_inductive` (default then `"auto"`): passes the
IDS adapter's inductive choice (`IdsAdapter(inductive=...)`) through the
unified engine. `"auto"` is the adapter's own default (the source's
`j_ohmic`, or the parallel residual `j_total − j_bootstrap − Σ driven` when
`j_ohmic` is absent or fails the adapter's consistency check);
`"j_ohmic"` and `"residual"` force one or the other. At the default the
contract is the one before the setting existed (tested). No tolerance,
criterion, ceiling or target moves; the consistency numbers are stamped in
`provenance["inductive_consistency"]` whichever choice is used. Validated by
name, refused when set with `reconstruction_engine="legacy"` and with a
g-file source (no effect there). Recorded in the engine record
(`settings.ids_inductive`).

## Unreleased — `engine_li_row_relaxation` (unified engine; default unchanged)

New `GenerationConfig.engine_li_row_relaxation` (default `1.0`): an
under-relaxation `r` (0 < r <= 1) of the unified engine's l_i-row
discrepancy update between reconstruction passes,
`d_k = (1 − rω) d_k−1 + rω [measured − model]` (the first update, from
`d = 0`, takes `r`). At the default the update is the one before the setting
existed, bit for bit (tested). It is a solver-side remedy for a row whose
per-pass gain exceeds the stability limit of 2 (a growing period-2
oscillation of the l_i row was measured on a high-bootstrap-fraction case
under the two-scalar preset, gain 2.07 at ω = 0.7); `r = 0.5` halves that
gain. It changes the path only: no tolerance, convergence criterion, pass
ceiling or target moves. Validated by name (refused outside `0 < r <= 1`,
and when set with `reconstruction_engine="legacy"`). Recorded in the engine
record (`settings.li_row_relaxation`). Draws carry no l_i row and are not
affected; nor are the q0 row and the MSE chords.

## Unreleased — approved change of a physics default: `separatrix_pressure` `"legacy"` → `"offset"`

**This is a change of a physics default, approved by the package owner on
2026-10-02.** `GenerationConfig.separatrix_pressure` (below, "The pressure
handed to the solver") now defaults to `"offset"`; `edge_pprime_pin` stays
`True`. Both engines and every legacy path follow it. No tolerance,
convergence criterion or ceiling moved.

- **Why.** Under `"legacy"` the solver is handed the full axis pressure as
  its target; its own pressure is zero at the boundary, so with a non-zero
  separatrix pressure `p_sep` it inflates `P'` everywhere by
  `p_axis / (p_axis - p_sep)` and reports the `beta` / `W_MHD` of a
  different profile. `"offset"` hands it `p_axis - p_sep` (the target the input's own
  `P'` integrates to; the solver's remaining rescale is its discretisation of
  that integral, not exactly 1) and adds `p_sep` back wherever pressure,
  `beta` or `W_MHD` is reported or delivered.
- **Measured basis** (real g-file and IDS cases, four arms each, pin on
  vs off × `"legacy"` vs `"offset"`): `"offset"` brought the full-frame
  `beta_N` / `W_MHD` closer to the input on 8 of 8 comparable g-file cases,
  by 1.4-8.0 points (median 3.5), and by about 0.5 points on the IDS slices;
  every pin-on `"offset"` arm converged (and the separatrix setting never
  changed whether a case converged, pin on or off); `l_i`, `q` and the current distances to the
  input did not move (l_i to 1e-3, current distances to 0.1 point); passes,
  solves, GS iterations and wall time identical. Solver-frame gaps grow
  (median 1.4 points on g-files) because under `"legacy"` the inflated `P'`
  was compensating a deficit in that frame. The large gaps where the input
  pressures themselves disagree are not closed by either setting.
- **What changes for existing users** when the solve pressure is not zero
  at `psi_N = 1`: `P'` in the solve is scaled by `(p_axis - p_sep) / p_axis`
  (with it the pressure-driven current and the Shafranov shift, slightly);
  the reported pressure, `beta_N` / `beta_p` / `W_MHD` (headline values are
  the full-frame ones) and the delivered g-files' `PRES` move toward the
  input's full-pressure values. With `p_sep = 0` nothing changes.
- **To get the old numbers:** `separatrix_pressure="legacy"`. At the
  pre-change settings (`edge_pprime_pin=True, separatrix_pressure="legacy"`,
  `edge_pressure.PRE_CHANGE_EDGE_PRESSURE`) every path is bit for bit what
  it was; the frozen-copy tests now prove that at those settings explicitly
  (`EDGE_PRESSURE_DEFAULTS` is the new default, a separate constant). A
  stored config that predates the setting (an old archive's `config_json`)
  reloads with `"legacy"`, with a warning, so it replays what it recorded.
- **Also:** the legacy non-loop routes' note that the solver's
  `solve_with_bootstrap` helper is not reached by the settings is now
  printed whenever the settings are not the pre-change ones, so it appears
  at the default.

## Unreleased — unified engine (introduced opt-in; the default since 2026-10-06, see the first entry)

*Introduced opt-in; with `reconstruction_engine="legacy"` nothing changes
(the legacy paths are untouched; the loop kernel's new hook is proven
bit-identical when absent) -- except the separatrix-pressure default above,
which applies to every path.*

- **`GenerationConfig.reconstruction_engine`** (`"legacy"` | `"unified"`,
  default `"legacy"`) with `engine_preset` (`"structured"` |
  `"structured_uniform"` | `"bootstrap_scalar"` | `"sawtooth_two_scalar"` |
  `"two_scalar_li"`), `engine_rows` (`"Ip"`,
  `"l_i"`, `"q0"`, `"mse"`), `engine_delivery_correction` (default `False`)
  and `engine_mse_jacobian` (`"fd_chord"` | `"fd_broyden"`; default `"fd_chord"`
  since 2026-10-02, owner-approved: the fixed finite-difference Jacobian
  converged every MSE validation case where Broyden's rank-one update, fed a
  second pass whose tan γ change was mostly the relaxing bootstrap and
  geometry, cost 4-5 extra passes and two pass-ceiling failures). Validated by
  name; engine options set under `"legacy"` are refused. Stored configs
  without the field load as `"legacy"`.
- **`bouquet/engine.py`**: one reconstruction loop for g-file and IDS inputs
  on the existing kernel -- parallel components composed on the latest
  solved geometry (identity I2: field-aligned conversion + the
  pressure-driven term from each pass's own p'), the structured closure with
  row discrepancies (scalar presets as restricted bases), one GS solve per
  pass, the MSE Jacobian by finite differences once at convergence then
  Broyden (or held), a checked two-pass delivery solve whose state a draw
  will inherit. Every convergence constant is an existing one
  (`engine.convergence_table`).
- **`bouquet/adapters.py`**: the source-adapter contract and the g-file / IDS
  adapters (the only place, with the exporters, where a current convention is
  converted). Raw-E_r MSE is refused.
- **IDS inductive: the parallel residual by definition (owner-approved
  default change, 2026-10-02).** `IdsAdapter(inductive=...)` and
  `GenerationConfig.engine_ids_inductive` default to `"residual"`: the
  inductive current is `j_total − j_bootstrap − Σ driven`; the source's
  `j_ohmic` becomes a cross-check, compared and stamped in
  `provenance["inductive_consistency"]` (action `"residual_by_definition"`)
  with no threshold and no warning. A source without `j_total` /
  `j_bootstrap` is refused (`"j_ohmic"` uses its `j_ohmic` explicitly).
  Evidence: on self-consistent sources the two are indistinguishable
  (`|Δl_i| ≤ 1.8e-3`); where a source's own split is locally inconsistent the
  residual is closer to its `<j_phi>`. `"auto"` and `"j_ohmic"` remain as
  explicit options with their semantics unchanged. No tolerance, criterion,
  ceiling or target moved. The IDS inductive changes on every source whose
  `j_ohmic` does not equal the residual exactly.
- `Bouquet.prepare_baseline()` dispatches to the engine when selected, for
  both inputs, and returns the usual `Baseline` plus `Baseline.engine` (the
  full record: contract, settings, convergence constants with their origins,
  per-pass log, delivery checks, state, solve counts). (Stage 2 refused
  engine baselines in `generate()`; Stage 3 below runs their draws.)
- `run_jbs_loop(..., extra=None)`: an opt-in hook for criteria the kernel
  does not own; absent, the kernel is bit-identical (frozen-copy test).
- Archive: `engine.store_baseline_engine` / `load_baseline_engine` write /
  read an ADDED `_baseline@engine_json` attribute (schema stays v3; wired
  into `generate()` with the draws in Stage 3).
- Tests: a toy Grad-Shafranov stand-in (`tests/_engine_toy.py`) drives the
  engine through Ip + l_i, + q0, + MSE, the presets, the delivery
  correction and the failure modes; the adapters run on the synthetic
  examples; identity (I2) is checked on the golden fixture's stored
  geometry; solver tests (`tests/test_engine_solver.py`, `-m solver`) and the
  probe `tests/probes/measure_engine.py` write the distance-to-input table.
- **The draws on the engine (Stage 3, `bouquet/engine_draws.py`).** With
  `reconstruction_engine="unified"`, `generate()` and
  `verify_sigma0_consistency()` run on the engine: a draw starts from the
  reconstruction's delivered state, perturbs the kinetics (the legacy
  pressure-matched stream), the parallel inductive (today's toroidal
  `sigma_jphi` / `j_ls`) and the bootstrap scale, holds `x*` and closes
  only the Ip row -- a scalar amplitude on the inductive in the exact
  measure, zero extra solves (optionally the q0 row too,
  `engine_draw_q0_row`, default off) -- and runs ONE bootstrap loop (draw
  ceiling, current gate standing), then the coil homotopy
  (`engine_draw_homotopy`, default on) and the existing post-homotopy check
  with its saturation guard. The first request of a zero-perturbation draw
  is the stored request bit for bit, by construction. l_i and beta_N drift
  and are recorded, with the change of the poloidal flux range
  `psi_b - psi_a` against the reconstruction (the per-part linear l_i
  attribution of an earlier version was removed: its model holds the flux
  range fixed, so its remainder was the geometry's response); the
  l_i band (`l_i_tolerance`) and `constrain_sawteeth` are post-hoc filters
  (out-of-band draws archived with `in_spec=False`, not counted by until-N,
  not `selected`). Cost is recorded per draw by stage. `draw_solve_maxits`
  now caps every engine solve. Archive: ADDED `engine_json` on draws and a
  `draws` block on the baseline record, `passes_draw_band` (a filter flag),
  engine MSE arrays as `structured_mse/engine_mse_*` datasets (schema stays
  v3). New rejection codes `jbs_non_finite`, `engine_closure_refused`
  (engine draws only). The legacy draw path is unchanged: every hook is a
  gated block, and the functions minus those blocks are the frozen code
  (`tests/test_engine_draws_legacy_ast.py`). Solver tests add the
  zero-perturbation draw on both examples and a seeded 6-draw batch
  (`measure_engine.py --draws 6 --seed 12345`).
- **Engine draws: bootstrap refresh and the homotopy solve cap (both
  default to the behaviour before them).**
  `GenerationConfig.engine_draw_bootstrap_refresh` (default `False`): after
  a draw's first loop solve the anchor's kinetic Redl increment is
  re-evaluated on that solved geometry and the loop restarts from it
  (`run_jbs_loop(start_refresh=...)`, a kernel hook that is a no-op when
  `None`) -- the path only, zero extra solves, no criterion or tolerance
  changed, the zero-perturbation request still bit-identical and the
  refreshed bootstrap `lambda_BS*` up to the re-solve's reproduction of
  the stored equilibrium (rounding on the toy, r_j 1.9e-5 on the live
  g-file example); recorded as
  `loop.bootstrap_refresh`. `draw_solve_maxits` (default `None`) now also
  governs every homotopy solve of an engine draw (installed for the stage
  when the solver does not already carry it), and a capped homotopy or
  post-homotopy solve that does not converge rejects the draw with the new
  codes `homotopy_maxits` / `post_homotopy_maxits` (never rolled back and
  archived); with `None` nothing is re-classified. The legacy draw path is
  unchanged (the new hooks are gated blocks; the frozen-code AST test
  passes). The solver probe takes `BQ_ENGINE_PROBE_GC` to run the solver
  tests with a draw setting on.
- **Engine draws: the solve cap defaults to 100, with rollback for a capped
  homotopy stage (owner's decision, 2026-09-30).** New
  `GenerationConfig.engine_draw_solve_maxits` (default `100`; `None` = the
  solver's own cap) caps every GS solve inside an engine draw (loop,
  homotopy stages, rollback re-solve, post-homotopy passes) and the
  zero-perturbation draw; the reconstruction runs under the solver's own
  cap. **Changed rule:** a capped homotopy STAGE is now a failed stage like
  any other -- it rolls back to the last good stage, and the draw is
  rejected (`homotopy_maxits`) only when there is none (it was rejected
  outright before); a capped rollback re-solve, post-homotopy pass or loop
  solve still rejects with its code. Every capped solve is recorded (stage,
  iterations, seconds, outcome) on `Bouquet.engine_draw_cap_events` and the
  draw's `homotopy.cap_events`. `draw_solve_maxits` is refused under the
  engine; the legacy draws are unchanged (`draw_solve_maxits` default
  `None`; frozen-code AST test passes). The fast test that asserted the old
  rule at homotopy pass 2 now asserts the rollback.
- **Engine presets `two_scalar_li` and `structured_uniform` (not defaults).**
  `two_scalar_li`: one scalar on the inductive, one on the bootstrap
  (constant basis), rows Ip + l_i -- a 2 × 2 system on the g-file's hard
  rows, the legacy secant's l_i family and the q95 attribution study's
  "2-scalar" state as a named preset; soft IDS rows go to the soft solver
  with the constant basis's σ = 1 as the documented uniform prior.
  `structured_uniform`: the shipped basis under
  `utils.STRUCTURED_WEIGHTS_UNIFORM` (the design's prior-sensitivity run).
  Neither adds a number. Solver test `tests/test_engine_two_scalar_solver.py`.
- **The pressure handed to the solver: one helper, two settings (both
  defaulted to the behaviour before they existed when introduced;
  `separatrix_pressure` has since moved to `"offset"`, see the approved
  change above; PHYSICS changes when moved).** `bouquet/edge_pressure.py` now builds every `P'` profile and
  axis-pressure target, replacing the inline `pp["y"][-1] = 0.0` /
  `pax = p[0]` statements of the legacy reconstruction and draws, the
  modelling-source forward solve, the zero-perturbation checks, the engine
  backend and the engine draws. At the pre-change settings (pin on,
  `"legacy"`) every array is bit for bit what it was (frozen-copy tests:
  each solve path, as an AST, against its pre-change code with the helper
  written back inline; the helper at those settings against the inline
  statements). Applies to BOTH engines.
  - `GenerationConfig.edge_pprime_pin` (default `True`). `False` keeps the
    profile's own `P'` at `psi_N = 1` instead of zeroing the last node.
    **Physics change when off:** the pressure-driven current is no longer
    forced to zero at the boundary and the edge current moves between the
    `P'` and `FF'` terms. Measured on the synthetic g-file example (unified
    engine): pressure-driven current at the boundary 0.006 -> 0.018 MA/m^2,
    `<j_phi>` at the last node 0.076 -> 0.115 MA/m^2, the `FF'` term there
    changes sign, `q95` +0.002, `l_i` and the core unchanged; no extra
    passes, solves or GS iterations.
  - `GenerationConfig.separatrix_pressure` (introduced with default
    `"legacy"`; default `"offset"` since 2026-10-02).
    `"offset"` passes `p_axis - p_sep` as the solver's axis target (`p_sep`:
    the solve pressure at `psi_N = 1`, thermal + impurity + fast; each draw
    its own) and adds `p_sep` back wherever pressure, beta or `W_MHD` is
    reported or delivered. **Physics change when on and `p_sep != 0`:** `P'`
    moves by the factor `(p_axis - p_sep) / p_axis` -- under `"legacy"` the
    solver inflates `P'` by the inverse to reach the full axis pressure with
    a pressure that is zero at the boundary. Measured on a constructed
    variant of the synthetic g-file example with `p_sep` = 5.4 % of the axis
    pressure: `"legacy"` solves `P'` x 1.058 and reports `beta_N` / `W_MHD`
    4.8 % / 5.2 % above the input's `p - p_edge` values and 7.7 % / 7.3 %
    below its full-pressure values; `"offset"` solves `P'` x 1.002 and
    reports -0.8 % / -0.4 % (solver frame) and +0.9 % / +1.2 % (full frame).
  - **Reporting:** `Baseline.edge_pressure`, the engine record's
    `edge_pressure` block, `reconstruction_metrics["pressure_like_for_like"]`
    and every draw record carry `p_sep` and beta / `W_MHD` in two frames --
    the solver's (from `p - p_sep`) and the full one (`W_MHD + 1.5 p_sep V`,
    each beta times `(int p dV + p_sep V) / int p dV`, `V` and `int p dV`
    the solved equilibrium's own) -- each compared with the input's
    same-definition quantity. Archived as `edge_pressure_json` on
    `_baseline` and on every draw.
  - **Delivery:** under `"offset"` written g-files (and the IMAS export built
    from them) carry the full pressure: `PRES` + that equilibrium's `p_sep`,
    `PPRIME` unchanged.
  - **Where the model stops:** a pressure jump at the boundary is not
    physical; the real separatrix pressure continues into the scrape-off
    layer, which a vacuum-outside free-boundary model cannot represent
    ([physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)).
  - **Not reached:** the solver's own `solve_with_bootstrap` helper (legacy
    non-loop routes; a printed note says so whenever the settings are not
    the pre-change ones, the default included).
  - Probe: `tests/probes/measure_engine.py` reports beta / `W_MHD` both ways
    (`distance.pressure_frames`), an `edge` stage, an opt-in per-solve
    iteration log, part `recon_legacy`, and checks of the delivered g-files.
    Solver tests: `tests/test_edge_pressure_solver.py`.
- **Probe fix:** `tests/probes/measure_engine.py::_distance_ids` takes the
  slice time from the source (it used the synthetic example's constant for
  every dd); the table records the slice it used.

## Unreleased, intended for the release after 1.4.0 — self-consistent bootstrap current (default ON)

### Approved change of a pass ceiling: `jbs_max_passes_post_homotopy` 4 → 6

**This is a change of a default pass ceiling, approved by the package owner
on 2026-10-01.** It is the only ceiling that changed: the reconstruction
ceiling stays 8 (`jbs_max_passes`) and the draw-loop ceiling stays 12
(`jbs_max_passes_draw`). No tolerance, no convergence criterion and not the
two-consecutive-pass rule moved; "converged" means what it meant.

- What it is: the number of further bootstrap passes a draw may take at the
  tight coil stage after its coil homotopy, when Redl on the post-homotopy
  equilibrium misses the bootstrap the draw carries. A draw that is not back
  inside the loop tolerances within the ceiling is rejected
  (`jbs_post_homotopy`).
- Measured basis ("bouquet unified engine: measurement fixes and the
  passes-to-convergence study", 2026-10-01): with the ceiling raised for
  measurement only, no draw of 72 needed more than **5** post-homotopy
  passes; the 10 draws that needed 5 are exactly the ones a ceiling of 4
  rejects, and all 8 of them that could be compared attempt for attempt with
  a run at the ceiling of 4 converged on the next pass and were archived.
  The new default is the measured need plus one pass of margin.
- Consequence: draws that were rejected as `jbs_post_homotopy` one pass
  short of convergence are now archived (each costs one or two more solves);
  a draw that does not converge still fails, two passes later.
- What does not change: with `jbs_self_consistent=False` (the frozen legacy
  path) the post-homotopy stage does not exist and nothing reads the field,
  so that path is bit-identical. A configuration that sets the field keeps
  its value.
- Results obtained with the loop on and the old default can differ in which
  draws are archived (never in an archived draw that needed 4 passes or
  fewer). The stored reference run of the test suite was produced at the
  ceiling of 4 and has not been regenerated here.

*Everything about the self-consistent bootstrap loop sits under this heading,
so it can become its own release after 1.4.0. The version string is still
1.3.1 and is not bumped here.*

**This changes results.** `GenerationConfig.jbs_self_consistent` now defaults
to `True`: every run re-evaluates the bootstrap on the delivered equilibrium
and iterates it to self-consistency. The bootstrap/inductive split moves, and
with it l_i, q0 and every per-draw bootstrap response. The golden regression
fixture has been regenerated with the loop on and input-current archival
(17 of 20 draws archived, 10 in spec; `tests/golden/README.md`, "The
self-consistent-bootstrap refresh"). `jbs_self_consistent=False` alone does
NOT reproduce a run made before this release (corrected 2026-10-06; this
paragraph used to say it was the legacy frozen bootstrap "bit for bit"). The
reproduction recipe is in "Reproducing a run made before this release" at the
top of this file: loop off AND `separatrix_pressure="legacy"` AND
`reconstruction_engine="legacy"`, with two residual moves that are not
switchable (the canonical coil-solve mode, the one current conversion). The
noise-floor acceptance of the structured soft closure (below) is passed only
by the loop's closure calls (`close_ip_structured_soft(...,
accept_noise_floor=True)`); the frozen path's calls keep the historical strict
solver -- but since the loop is the DEFAULT, the noise-floor acceptance is on
the default path (see that section). Also:

- A stored config without the field (an old archive's `config_json`, or any
  dict/JSON without it) loads with `jbs_self_consistent=False` and a warning
  -- it replays the bootstrap model it was produced with; the warning says how
  to opt in (`"jbs_self_consistent": true` in the `generation` section).
- `single_profile_jphi=True` and `recalculate_j_BS=False` have no bootstrap to
  iterate and are refused unless `jbs_self_consistent=False` is set.
- **Archive schema v3** (additive in its NAMES; field meanings changed on
  the default path later -- see the first entry's "At a glance"): the
  `jbs_loop` block (`jbs_converged`,
  `jbs_n_passes`, `jbs_loop_json`) on every loop draw and now also on
  `_baseline`; `DrawView.jbs_loop`, `ScanView.baseline_jbs_loop` /
  `bootstrap_model`. No migration: a v2 archive reads as frozen everywhere
  ([archive-schema.md](archive-schema.md#v2--v3-the-self-consistent-bootstrap-record)).
- Plots label the bootstrap "self-consistent Redl bootstrap" or "frozen SWB
  bootstrap (legacy)" from what the archive records.

What the loop is (unchanged from its opt-in introduction below): the joint
relaxation of the bootstrap (ω = 0.7) and of the solved current (β = 0.7), ω
halved only on sustained growth, convergence = every active residual on two
consecutive passes (`r_j ≤ 1e-3`, `r_I ≤ 1e-4 I_p`, `Δl_i ≤ 1e-3`,
`Δq0 ≤ 2e-3`), hard failure by default or a flagged slice with
`jbs_loop_on_fail="flag"`, and the soft closure's noise-aware stop test with
one logged retry. Pass ceilings (limits, not tolerances): 8 for the baseline /
reconstruction, 12 for each loop of a draw (`jbs_max_passes_draw`, was 6) and
6 post-homotopy passes (`jbs_max_passes_post_homotopy`, a config field; was a
hard-coded 2, then 4 -- see "Approved change of a pass ceiling" below). The draw ceilings were raised when the golden refresh
showed the standard draw's l_i-match coupling needing 7–8 passes (it contracts
at ≈0.38/pass from r_j ≈ 2e-2…1.2e-1) and a post-homotopy stage whose first
pass misses needing 3 under the two-consecutive rule; no tolerance moved.

Where a draw's loop starts (recorded per loop as `init_source`): Redl at the
draw's state anchor on the draw's OWN perturbed kinetics for its first loop,
warm from its previous converged bootstrap for later ones -- never the
unperturbed baseline bootstrap. The large first residual of an l_i-match
candidate's loop is geometric (the candidate's new inductive shape moves q and
the flux range; Redl with the same kinetics on that geometry differs by a few
to ~10 % in I_BS), not a kinetic mismatch. A fast and a solver test run the
same draw loop from the baseline bootstrap and reach the same fixed point.

### One reconstruction state; an unperturbed draw reproduces it

**The rule.** There are three levels: the INPUT (a g-file, or a
modelling-source IDS); the bouquet RECONSTRUCTION, as close to the input as it
can be while physically valid and carrying a Redl bootstrap -- allowed to
differ from the input, since most inputs carry no Redl/Sauter bootstrap; and
the DRAWS, perturbations of the reconstruction. With the loop on, the
reconstruction is ONE equilibrium -- the saved baseline g-file, `l_i_target`
and every recorded l_i / q0 / q95, the archived baseline profiles, the centre
of the draws' l_i band and the reference of the zero-perturbation check -- and
a draw with every perturbation at zero reproduces it (its equilibrium, current
profile, bootstrap, l_i, q0, q95) at the loop's existing tolerances. Nothing
here applies with `jbs_self_consistent=False`: the legacy path is unchanged
bit for bit (`tests/probes/legacy_bitwise_ab.py` runs the out-of-tree A/B).

**What changed in the reconstruction.**
- g-file path: before, the baseline carried three states (the step-6
  l_i-matched value 0.653864 that `l_i_target` came from, the post-corrective
  state 0.656455, and the saved g-file, a single re-solve of the stored
  achieved current, 0.653866). Now every loop pass ends on the corrective
  iteration's landed REQUEST re-matched in l_i (the step-5 secant on the
  inductive amplitude of that request); that state is delivered, measured,
  saved and archived, and keeps the designed l_i match to the input.
  Alternative, not taken (the owner's choice to make): deliver the
  post-corrective state and give up the l_i match (band centre +2.59e-3 in
  l_i, q95 reference moves ≈0.024, about one ensemble σ).
- Both paths: the stored split is the delivered state's jphi-linterp request,
  normalised to I_p in the 'exact' FSA current measure on it (on the
  modelling-source example the source total read 0.965 I_p there; the
  solver had made the rest up uniformly), `j_BS` (+ `jBS_diff`) the draws'
  σ=0 bootstrap composition on it, the fixed parts as read, `j_inductive` the
  residual. In diff mode `j_BS + jBS_diff` is still the source bootstrap.
  New: `Baseline.delivered_state`, `Baseline.jphi_request_offset`, the
  archive attr `_baseline@delivered_state_json`
  ([archive-schema.md](archive-schema.md)).

**What changed in the draws (loop on).** Every stage is the identity at zero
perturbation: the state anchor carries `jphi_diff`; route R2's scale reads 1;
the standard route perturbs the achieved-convention inductive
(`j_inductive - jphi_request_offset`), roots its amplitude in the exact
measure on the live geometry instead of the limiter-area flux integral, and
starts every corrective iteration from `target + jphi_request_offset`; the
delta-mode reference anchor carries `jphi_diff`. **This moves draws with
non-zero perturbation** (their reference moved): modelling-source R2 draws
lose the 1.046–1.049 inductive over-scaling (expected at the band centre:
l_i −0.9 %, q0 +0.9 %, q95 +0.3 %, from the diagnosis' s = 1 replay);
g-file draws are centred on the re-matched state (its l_i within the
secant's 1e-3 of the input's, as the old `l_i_target` was; its q0/q95 are
expected of the order of the old single-solve baseline g-file's, which sat
+0.024 in q95 above the post-corrective state -- not yet measured);
standard-route draws get a different amplitude root (the limiter measure read
10–44 % high; `find_optimal_scale` compensated partly, j0 scale 1.153 on the
g-file example, 1.035 on the modelling-source one) and a warm corrective
start; the j_φ GPR envelope (`jphi_scalar_sigma·|j_phi|`) follows the
normalised request (+3.6 % in amplitude on the modelling-source example,
edge-shape changes of order 1 % on the g-file one).

**`verify_sigma0_consistency` (loop on).** `passed` now REQUIRES the draw's own
route -- every route the configuration can use (both on the g-file path; R2
on the modelling-source path unless `workflow="custom"`) -- to reproduce the
reconstruction state at `jbs_rtol_j` / `jbs_rtol_Ip` / `jbs_tol_li`, with q0,
q95 and the total-current profile reported beside them; `draw_route=False`
leaves it False (unverified). The loop "solved the baseline's way" is kept as
`passed_baseline_way` and no longer decides `passed`; it now solves the way
the reconstruction's final state is solved (one jphi-linterp solve per pass).
Two existing test assertions that pinned the old scoping ("the draw route
gates nothing"; `passed` is the baseline-way conjunction) were re-scoped by
this decision, their checks kept on the renamed result. New tests: the
stage-wise identity on a toy solver (`tests/test_sigma0_identity_stages.py`),
live probes on both examples and both routes
(`tests/test_sigma0_identity_solver.py`), and loop-ON twins of the six
zero-perturbation tests in `tests/test_seeded_reproducibility.py` (the legacy
ones are kept).

### Opt-in introduction (earlier on this branch)

`GenerationConfig.jbs_self_consistent=True` replaces the once-computed, frozen
`solve_with_bootstrap` bootstrap with a Redl bootstrap re-evaluated on the
delivered equilibrium inside a relaxed outer loop (closure ↔ GS solve ↔ Redl)
that runs to a convergence test. New: `physics.evaluate_jBS` (Redl on the
caller's own ψ_N grid, geometry of the current equilibrium, gradients on the
true grid, direct toroidal conversion; bit-identical to SWB's first pass on a
uniform grid), `bouquet/jbs_loop.py` (residuals, two-consecutive-pass
convergence, relaxation, `JBSNotConverged` / `"flag"`), and the loop in the
IMAS baseline (every `jBS_baseline_mode` and closure channel; the correctors'
steps not taken, their bookkeeping and acceptance flags kept -- for q0 that
leaves a residual only recorded, see `jbs_loop_q0_corrector` below), the structured
closure's MSE stage (Jacobian once, chord steps with j_BS re-evaluated, one
final Jacobian refresh), every draw (Fix C and the standard l_i loop, a
post-homotopy check), the σ=0 check and the geqdsk reconstruction. In diff mode
`jBS_diff` becomes a pure model offset on the delivered baseline geometry.
Records: `li_metrics["jbs_loop"]`, `ip_closure["jbs_loop"]`,
`reconstruction_metrics["jbs_loop"]`, per-draw archive attrs `jbs_converged` /
`jbs_n_passes` / `jbs_loop_json`. Config: `jbs_self_consistent`, `jbs_init`,
`jbs_rtol_j`, `jbs_rtol_Ip`, `jbs_tol_li`, `jbs_tol_q0`, `jbs_max_passes`,
`jbs_max_passes_draw`, `jbs_max_passes_post_homotopy`, `jbs_relax`,
`jbs_relax_halve_on`, `jbs_relax_current`, `jbs_loop_on_fail`;
`swb_iterations` is now documented as legacy. With the flag off the legacy
code path is the historical one. No existing tolerance value moved; one
acceptance criterion did (the soft closure's noise-floor acceptance, below,
loop only). See [physics-notes.md](physics-notes.md#self-consistent-bootstrap-jbs_self_consistent).

Follow-up (loop iteration path and closure stop test): the loop relaxes the
solved current as well as the bootstrap (`jbs_relax_current = 0.7`) and halves
ω only on sustained growth (`jbs_relax_halve_on = 3`) — path only, the fixed
point is unchanged (tested against the closed-form fixed point of a two-state
model). **The structured soft closure's acceptance criterion changed, for
the loop only.** Old: when no damped step descends, accept iff the scaled
gradient is below `rtol·max|J|·max(√F, 1)`, else refuse. New, with
`accept_noise_floor=True`: also accept an iterate stationary to within the
objective's rounding noise (`stop_reason="noise_floor"`, recorded with the
gradient, predicted decrease and noise estimate, and printed). It accepts
points the old test refused, and only those (every result the old test
returned is unchanged). The loop's closure calls pass the flag; the default,
and every frozen-path call, is the old criterion. `close_ip_structured_soft`
also takes an optional start `x0`; inside the loop a refused soft closure is
retried once from the previous pass's coefficients (`closure_retry`, logged).

### The q0 pin under the loop: `jbs_loop_q0_corrector` (opt-in, default off)

With the loop on, the two channels that pin the on-axis safety factor
(`closure_channel="sawtooth_bootstrap"`, and `"structured"` when the sawtooth
gate admits the axis row) re-solve their predictor every pass with the axis
row HELD at the anchor's requested axis current; the corrector only records
the q0 residual on the delivered equilibrium and flags it against `q0_tol`.
The legacy path's Newton corrector step, which removes that residual, does not
act under the loop. That remains the default, bit for bit.

`GenerationConfig.jbs_loop_q0_corrector=True` makes the pin act inside the
loop. Once per pass, the axis row is moved from the q0 measured on that pass's
solved equilibrium: `j_ref0 ← j0_solved · q0 / q0_target`. This is the legacy
structured corrector's update, applied per pass, with `j0_solved` the axis
value of the current actually solved (the β-relaxed blend). Convergence then
also requires `|q0 − q0_target| ≤ q0_tol` on two consecutive passes, beside
the `jbs_tol_q0` step criterion. This adds a condition and relaxes nothing:
`q0_tol` (0.01), every loop tolerance and every pass ceiling are unchanged.
The MSE chord stage keeps the pin acting (every chord step is a pass). A
joint iteration that does not converge within the ceiling fails exactly as
the loop fails (raise, or flag under `jbs_loop_on_fail="flag"`), with the q0
residuals in the message and in the record. The delivered equilibrium is
checked against `q0_tol` once more, so a delivered state outside it is never
marked converged.

Records:
- `jbs_loop["q0_pin"]`, per pass: the axis row, the solved axis current,
  q0, the residual, the residual / `q0_tol`, and the next row.
- `ip_closure`: `q0_pin_acted`, `q0_pin_n_row_updates`, the initial and final
  axis row, `q0_residual_over_tol`, and `q0_pin_delivered_within_tol`.

The run-time NOTICE says which mode is in force. The l_i row is not covered:
it stays held at its target, because its log-gain update is a separate
design. With `jbs_self_consistent=False` the flag has no effect, because the
legacy corrector already takes its step.

### Review fixes (2026-09-29)

- **Noise-floor acceptance is opt-in at the function** (`accept_noise_floor`,
  default `False` = the historical strict solver, raising exactly where and
  with exactly the message it did before the loop) -- and ON on the default
  path, because every closure call of the self-consistent loop (the default)
  passes `True`; only the frozen path keeps the strict solver. Its effect is
  rounding-level (it accepts an iterate stationary within the objective's
  rounding noise instead of raising). It had been applied to every caller,
  the frozen path included. A non-finite or non-positive noise
  estimate now accepts nothing; every acceptance is printed as well as
  recorded. `NOISE_FLOOR_FACTOR` (2) and every tolerance are unchanged.
- **`evaluate_jBS` never returns a silently zeroed bootstrap.** Non-physical
  input (`n_e`, `n_i`, `T_e`, `T_i` not strictly positive, `Z_eff < 1`) and a
  failed-trace geometry row raise `physics.JBSEvaluationError` (a
  `ValueError`) naming the quantity and ψ_N; the historical `nan_to_num`
  survives only at the clipped axis / separatrix nodes (counted in
  `diag["n_nonfinite_zeroed_at_ends"]`). Bit-identical for every accepted
  input. A draw whose kinetics are refused is a failed draw.
- **The loop kernel checks finiteness.** A non-finite initial guess or Redl
  evaluation raises `jbs_loop.JBSNonFinite` (a `JBSNotConverged`) at once,
  with the pass and ψ_N, whatever the `"flag"` policy; a pass that can never
  count (a gated l_i / q0 not returned, J ≡ 0 against a non-zero iterate)
  stops the loop at that pass instead of at the ceiling.
- **Record only:** every pass records the unrelaxed closure-half residual
  `current_residual_unrelaxed = ‖jc_k − js_k−1‖_w / ‖jc_k‖_w`
  (= `current_gap / (1 − β)`); the convergence gate is unchanged.
- **No filesystem path in records.** `oft_build` is now `{version, git_hash,
  build_id}` (was the OFT install path). The golden fixture's path guard also
  catches `/usr`, `/Volumes`, any absolute path at a token boundary, `~`,
  `../` and Windows paths, including inside string arrays.
- **Config validation.** `swb_iterations` set with the loop on raises a
  `DeprecationWarning` (ignored under the loop, honoured only with
  `jbs_self_consistent=False`); bools are refused as tolerances, relaxation
  factors and ceilings; an unknown `generation` key in `from_dict` /
  `from_json` is refused with the nearest valid key (retired fields are
  dropped with a warning).
- **Documented, not changed:** the evaluator's toroidal conversion drops the
  `⟨B_φ²⟩/⟨B²⟩` bracket, exactly as the legacy path does. The neglected
  `⟨B_p²⟩/⟨B²⟩` is 1.0–2.1 % across a D3D-like plasma (1.4 % at the bootstrap
  peak, 1.4–1.5 % of I_BS), not the "sub-1 %" the docstrings claimed, and the
  IDS export (exact `⟨1/R²⟩`) round-trips `⟨j·B⟩` high by that fraction.

### Engine base: loud refusals and consistency fixes (Stage 0)

- **The inductive-amplitude fallback is loud.** When `fit_inductive_profile`
  cannot bracket the cylindrical l_i-proxy root it still falls back to 1.0
  (bracket unchanged), but now prints the reason, the bracket and the
  residuals at its ends and records them in the reconstruction metrics
  (`ind_scale_fallback`, `ind_scale_fallback_n`,
  `ind_scale_fallback_records`). Outputs are bit-identical otherwise.
- **A failed draw state-anchor solve rejects the draw.** Under the
  self-consistent loop the anchor solve's failure was swallowed and the draw
  continued on whatever equilibrium the solver held; it now raises
  `DrawAnchorSolveFailed` and the draw is rejected, printed and recorded
  with the new reason code `anchor_solve_failed`. The frozen legacy draw
  route (loop off) is unchanged.
- **One electron-charge constant** (`physics.ELEMENTARY_CHARGE =
  1.602176634e-19`), read by every pressure site. **Deliberate consistency
  fix:** under the self-consistent loop the draws' (and the g-file
  reconstruction's) thermal pressure moves from 1.6022e-19 to it, i.e. by
  −1.46e-5 relative (−1.3e-5 of the total with fast-ion and impurity
  pressure), which removes the σ=0 draw's pressure offset against the
  modelling-source forward solve. No fast-test number moved beyond rounding
  (the toy-solver draws are bit-identical apart from the pressure itself).
  The frozen legacy path keeps its value as `ELEMENTARY_CHARGE_LEGACY`
  (selected by `physics.thermal_pressure_charge`), bit for bit;
  `sampling.EC` remains as that legacy value for back-compatibility.
- **q0 at like radii.** The solver's reported q0 (`get_stats()['q_0']`) is q
  at ψ_N = 0.02 (`physics.SOLVER_Q0_PSI_N`), not on axis. The g-file
  reconstruction metrics now read the g-file's q at that radius (`q0_efit`,
  `q0_err_pct`), keep its axis value and the old solver-vs-axis error as
  `q0_efit_axis` / `q0_err_pct_vs_axis`, and record `q0_psi_N`; the
  delivered states, the σ=0 check's reference and the baseline re-solve
  record carry `q0_psi_N`; the sawtooth-gate messages name the ψ_N they read.
  No q0 target, gate, tolerance or verdict changed (the verdict never read
  q0); the IMAS q0 closure was already like-for-like at `psi_q[0]`.
- **The legacy path stays bit for bit** under all of the above: its pressure
  factor is the historical 1.6022e-19 exactly, the amplitude fit is
  bit-identical to its frozen copy, the anchor rejection is loop-only and
  the q0 changes are labels (`tests/test_legacy_path_stage0_bitwise.py`, a
  fast form of the out-of-tree legacy A/B probe).
- **Optional draw-loop iteration cap** (`draw_solve_maxits`, default `None` =
  the solver's own cap, so nothing changes unless it is set), with a record
  of every draw solve that raises (`diagnostics['solve_failures']`,
  `Bouquet.solve_failures`, one `[draw-solves]` line). Ported from the
  collaborator's pull request with the same field name; its re-solve of a
  capped solve at a looser tolerance is NOT ported (not approved).

## Unreleased — MSE pitch angles on the structured closure (opt-in)

`closure_channel="structured"` accepts measured MSE pitch angles
(`GenerationConfig.mse_data`, schema in `bouquet/mse.py`) as a third
measurement next to Ip and l_i: `chi2_MSE` joins the objective of both the hard
and the soft solver. Because tanγ is read off the **re-solved** equilibrium,
the forward model is linearised by forward differences on solved equilibria
(`utils.structured_mse_jacobian`, one GS solve per free coefficient) and the
closure re-solved (`utils.structured_mse_outer`); the q0/l_i corrector keeps
the term in every re-solve it takes. Recorded per slice: chords used, E_r
treatment, the field orientation (stated by the block's `ip_sign`/`bt_sign` and mapped onto the equilibrium's own directions — never chosen by fit; a better-fitting alternative is flagged), chi² before/after/delivered, per-chord residuals,
the linearisation residual and the achieved objective, which the linear step
cannot raise — if it does, the slice is flagged closure-limited. The q0/l_i
corrector may re-solve after the MSE stage, so the verdict is also taken on
the **delivered** equilibrium: its per-chord residuals (in sigma) are
recorded, and the slice is flagged when the delivered chi² — or, where
comparable, the delivered objective — is worse than the pre-MSE closure's
(reporting only; no iteration count or tolerance changed). `n_extra_solves`
counts the MSE stage's solves; after an applied MSE stage the corrector's
entry readbacks are recorded as `*_mse_stage`, and `*_predictor` keeps the
predictor's values.
Without `Er` or `er_corrected=True` the model takes E_R = 0, which biases the
fit in a rotating plasma (to first order B_Z is read as B_Z + (A5/A1)E_R —
a systematic reshaping of the fitted current profile); this is warned and
recorded. The forward model is the standard A1..A7 form with E_Z = 0 and no
denominator E_R term (a block applying E_r with non-zero A7 is refused).
`structured_mse_required=True` refuses instead of running without the
constraint. **Nothing changes without `mse_data`**: every default is inert and
no tolerance or acceptance criterion moved.

Under the self-consistent bootstrap loop the MSE chord stage gets the same
review fixes as the stage above: an off-mesh chord is excluded with its reason
(fewer than `structured_mse_min_chords` left refuses) and a chord lost on a
later read is a refusal; the orientation is the stated one, audited and
flagged, never fitted; every GS solve it spends (restore re-solve included) is
counted into `n_extra_solves`; per-chord arrays and the Jacobian go to
`Baseline.mse_record`; and the delivered-equilibrium chi² judgement runs on the
state it delivers.

## Unreleased — reversed-current IMAS sources (hotfix)

**A dd with `ip < 0` is now read into bouquet's positive-current frame.** Before
this, `read_imas_baseline` kept the dd's (negative) current profiles while every
bootstrap bouquet recomputes on its positive-current anchor is positive, so on
a reversed-current source the bootstrap was added **against** Ip — in the
legacy `solve_with_bootstrap` path, in the draws, and (on builds that have it)
in the self-consistent loop. The Ip closure, the q0 target (negative),
`fuse_total_err_pct` (off by 2c) and `swb_over_fuse_jBS_peak` were all wrong for
such a source.

- **Changes results only for sources with `ip < 0`.** For `ip ≥ 0` the reader
  multiplies by exactly `+1.0`; baselines, closures, draws and archives are
  bit-identical to before (verified bitwise A/B against the pre-fix build: the
  synthetic IMAS example's forward-solved baseline under four closure paths,
  and a seeded g-file run of the golden-fixture example including its draws). **Any bouquet result built on a reversed-current dd before
  this change is invalid and must be regenerated.**
- New records: `Baseline.source_current_sign` / `source_current_sign_origin`
  / `source_b0_sign`, the same keys in `li_metrics`,
  `ip_closure.source_current_sign` / `source_current_sign_origin`, and
  `_baseline` attrs `source_current_sign` / `source_current_sign_origin` /
  `source_b0_sign` / `current_frame` on IMAS archives.
- **A dd whose current profiles disagree in sign with its own `ip` is now
  refused** (`ValueError` naming each quantity and its sign) instead of being
  warned about and closed in a mixed frame. The test is on the net,
  area-weighted toroidal current of `core_profiles.j_tor` and of the
  equilibrium `j_tor` the anchor uses. New
  `ImasSource.current_orientation` (`"auto"` default, `+1`, `-1`) names the
  factor explicitly for a file whose convention the user knows.
- **Measured coil-current targets are in the solve frame.**
  `coil_targets.measured_from_pf_active` multiplies the `pf_active` circuit
  currents by the source's orientation factor (new `current_orientation`
  argument, `"auto"` default) and returns a `MeasuredCoilCurrents` dict that
  records it; `coil_reg_from_measured` stamps each term with
  `"source_current_sign"`, and `_apply_coil_reg` refuses a term whose factor
  disagrees with the IMAS baseline's. Unchanged for `ip > 0`.
- A user-supplied `FixedComponentsConfig.j_NBI` / `j_RF` is defined in
  bouquet's positive-Ip frame (co-current positive) and used as given on both
  source paths; the IMAS reader does not multiply it by `sign(ip)`.
- **The IMAS export is written in the source's frame.** `write_imas_draw`
  used to write positive-frame `ip`, ψ, P′, FF′, `f`, q and currents into a
  template that keeps source-frame `core_sources`, `pf_active` and `b0`, so a
  re-read of a reversed-source export was off by 2|j_NBI| in `j_inductive`. It
  now restores the source orientation (archive `source_current_sign`, template
  `b0` and q-sign convention) on every field it writes, and refuses a template
  whose orientation contradicts the archive. For `ip > 0` sources only `f`
  changes, taking `b0`'s sign.
- Delivered g-files are unchanged in convention (`CURRENT > 0`, `BCENTR > 0`,
  so always `Ip·Bt > 0`, TokaMaker's COCOS 7, for every source); they do not
  carry the experiment's orientation or its field-line helicity, which matters
  for 3D-field (error-field / coil-coupling / NTV) work. Restoring the source
  orientation flips P′ and FF′ with ψ. See
  [physics-notes](physics-notes.md#current-and-field-orientation).
- g-file-path plot overlays (`plot_input_vs_recon`, the reconstruction
  diagnostic's FF′, `plot_jphi`'s geqdsk total) are now drawn in the solve's
  positive frame (`× sign(CURRENT)`); `plot_input_vs_recon` reads the g-file in
  the source's declared COCOS.

## Unreleased — a report-only core-pressure hollowness record

Every baseline now records `core_pressure_hollow`, which describes whether and
by how much the core pressure rises above its axis value. It **changes no
results**. No profile, solve, filter decision, in-spec count or until-N count
reads it. The pressure handed to the solver is bit-identical with and without
it, and a test checks this on the real composition code.

- **Measured:** `rise_frac = (max p over psi_N ≤ 0.5 − p_axis) / p_axis`, where
  the maximum sits and the radial extent of the climb, and the summed width of
  the positive-gradient core intervals. `is_hollow` is `rise_frac > 1 %`. That
  bar is a reporting choice, not an acceptance criterion, and it is stored with
  the numbers.
- **Where:** on the input pressure (total, and thermal species alone) and on
  the achieved pressure of the converged baseline, on both source paths. Bad
  input gives "not evaluated" with a reason, never "not hollow".
- **Stored:** `Baseline.core_pressure_hollow` and `li_metrics_json` on the
  archived `_baseline` group, next to `ip_closure`. Older archives read
  unchanged. Draws do not carry the record.
- It describes the profile only. A hollow core can be physical, and the
  record does not say why one is there. See
  [physics-notes.md](physics-notes.md#core-pressure-hollowness-record).

## Unreleased — the default LCFS boundary cut is now device-calibrated

**For DIII-D the default boundary cut changes from 5.0 mm to 8.5 mm. This
LOOSENS that acceptance criterion for DIII-D:** a draw whose LCFS deviates from
the reconstruction by between 5.0 and 8.5 mm rms was rejected before and is
accepted now, both by `Bouquet.filter()` and by the until-N loop's in-spec
count. The 8.5 mm value is the across-slice 90th percentile (8.34 mm, rounded)
of the LCFS rms displacement under the magnetics' stated noise in a
pre-registered boundary-UQ study (`bouquet.devices.DEVICES["DIII-D"]`). Every
other device, and any mesh whose coil set is not recognised, keeps the generic
5.0 mm: a run that sets nothing on an unregistered device selects exactly what
it did before.

`FilterConfig.rms_max_mm` now takes:

| setting | cut | stamped `boundary_cut_source` |
|---|---|---|
| `"auto"` *(default)* | the device's calibrated cut (DIII-D 8.5 mm), else 5.0 mm | `device:<name>` / `generic` |
| a number | that number | `explicit` |
| `"off"` | none | `disabled` |
| `None` | none — its historical meaning, kept so configs written with `None` behave as before | `disabled` |

- **To keep 5.0 mm on DIII-D**, set `filtering.rms_max_mm = 5.0` explicitly.
- The device is taken from `config.device`, else detected from the mesh's (or
  the archived baseline's) coil names; only an exact signature match counts.
- The resolved cut is printed once per run, on screen (`[boundary cut] …`),
  before the solver output is captured; an explicit `filter(rms_max_mm=…)` is
  announced too. It is stamped on the scan group (`boundary_rms_max_mm`,
  `boundary_cut_source`), and an until-N run also archives the bound its count
  was taken against (`inspec_rms_max_mm`, `inspec_cut_source`, carried
  through the parallel worker record and merge manifest). `filter()` warns if
  it cuts at a different bound than the loop counted against.
- Re-running `filter()` with default settings on a DIII-D archive you filtered
  before this change can select **more** draws than it did. Quote the cut that
  produced any in-spec fraction you report — `draw_band` provenance carries it.

## Unreleased — the default coil acceptance criterion changed

**`Bouquet.filter()` now judges coil currents with a measurement-referenced χ²
test instead of the ±2% band.** `FilterConfig.coil_filter` is a new field and its
default is `"chi2"`. This is a change to an **acceptance criterion**, so read the
whole section before regenerating a figure from an existing archive.

### What changed

| | before | now (default) |
|---|---|---|
| Rule | `max_F_drift_pct ≤ inspec_F_max` (2%) and `max_VSC_drift_pct ≤ inspec_VSC_max` (2%) | χ²/ν ≤ `chi2_max` over every modelled coil, plus a worst-coil \|z\| ≤ `z_max` guard |
| σ | none — a flat fraction of each coil's own baseline | per-coil, from the device tolerance model (floor + fractional part, floor set by the acquisition era) |
| DIII-D thresholds | 2% / 2% | χ²/ν ≤ 6.1, \|z\| ≤ 6.3 (the 95th percentile of what real machine states score); generic 4 / 5 where no device calibration applies |

Only **which draws are marked `selected`** moves. Generation, the baseline
solve, sampling and the archive contents are untouched, and the per-draw
`in_spec` attribute still carries the legacy verdict (see
[`coil-constraints.md`](coil-constraints.md)).

### This changes results you may already have

- **Any `Bouquet.filter()` call, and any saved config replayed through
  `load_config` that predates `FilterConfig.coil_filter`, now selects a
  different subset of the same archive.** A config written before this release
  has no `coil_filter` key, so it takes the new default; nothing about it looks
  different. Re-running `filter()` on an archive you filtered last month can
  legitimately give a different in-spec fraction, with no other input changed.
- **The new default is not uniformly stricter — on some ensembles it is more
  permissive than the band it replaces.** A flat ±2% is many σ on a high-current
  F-coil and a fraction of one σ on a low-current coil, so it rejects a large
  and state-dependent share of an L-mode ensemble for reasons that have nothing
  to do with measurement precision. The χ² rule is calibrated instead to a 5%
  nominal false-rejection rate against the machine's own residual distribution.
  Better-founded, but **not** a conservative swap: quote which rule produced any
  in-spec fraction you report.
- **The calibration caveat is real and is already stated in the code.** The
  quantiles were measured at ν = 18 coils; the shipped DIII-D signature judges
  20. `max|z|` is an order statistic, so the true false-rejection rate is
  **above** the nominal 5%, and it grows with ν. The thresholds were **not**
  moved to compensate: `filter_coil_chi2` warns whenever ν ≠ the calibrated ν
  and stamps both counts into `coil_sigma_model["acceptance"]`
  (`calibrated_nu`, `nu_used`). See the "Scope of the calibration — READ BEFORE
  TRUSTING THE QUANTILES" block in `bouquet/devices.py`.

### Getting the old behaviour back

One knob — **`filtering.coil_filter = "legacy"`**:

```python
config.filtering.coil_filter = "legacy"     # restores the ±percent band
config.filtering.inspec_F_max = 0.02        # the band itself, unchanged
config.filtering.inspec_VSC_max = 0.02
```

`"legacy"` is bit-identical to the pre-release rule; `inspec_F_max` /
`inspec_VSC_max` keep their old meanings and defaults and are read by nothing
else. The standalone `filtering.filter_coil_currents` is also unchanged and
still applies the band directly.

The χ² filter also falls back to the band **loudly** (a `UserWarning` naming the
reason) when no per-coil σ can be resolved — a schema-v1 archive, an archive
stored without coil data, or a mesh whose coil names match no registered device
and no `filtering.coil_sigma` given.

### New knobs

`filtering.chi2_max`, `filtering.z_max`, `filtering.coil_sigma`,
`filtering.coil_daq_era` and the top-level `device` field; all default to
`None`, i.e. "use the device model". `coil_daq_era` chooses the σ **floor** and
is therefore itself an acceptance criterion: it is never inferred from a run
header, mesh name or file path, and `filter()` prints the era it resolved, the
floor that buys and which route it came from, once per call. Full table in
[`workflows.md`](workflows.md#configuration-reference).

## Unreleased — fast-ion pressure: the reduction rule now comes from dd provenance

**Behaviour change on the IMAS path.** `FixedComponentsConfig.p_fast_reduction`
and `read_imas_baseline(..., p_fast_reduction=...)` now default to `"auto"`
instead of `"trace"`. On a FUSE/IMAS.jl-written dd this makes `p_fast` **3×
larger** than ≤1.3.1 produced; on a dictionary-convention dd it is unchanged.
Every downstream `beta_N`, `W_MHD` and `p'` on the IMAS path moves with it. The
g-file / `ReconstructionSource` path is untouched — `p_fast_reduction` is read
only on the IMAS path.

### What was wrong

`pressure_fast_parallel` / `pressure_fast_perpendicular` are written with two
incompatible meanings, and no dd field records which one is in use:

| producer | what the two fields hold | scalar `p_fast` | rule |
|---|---|---|---|
| IMAS.jl / FUSE | the pressure **per degree of freedom** (`pressa/3` in each) | `p_par + 2·p_perp` | `"sum"` |
| IMAS data dictionary, OMAS-written dds | the **full** directional pressures | `(p_par + 2·p_perp)/3` | `"trace"` |

Verified upstream: IMAS.jl's `pressure` expression is `pressure_thermal +
pressure_fast_parallel + 2·pressure_fast_perpendicular`
(`src/expressions/dynamic.jl`) and its `src/physics/fast.jl` adds `pressa/3` to
*each* directional field. So the old fixed `"trace"` default returned **one
third** of the fast-ion pressure on every FUSE dd — a shortfall measured at
8–35 % of the total pressure across a set of beam-heated discharges, closing to
<2 % under `"sum"`. A fixed `"sum"` default would have been wrong the other way,
by exactly 3×, on any standards-compliant dd — including this repo's own
synthetic `examples/D3D-like/D3Dlike_baseline_omas.json`.

### What changed

* **`p_fast_reduction="auto"` (new default)** picks the rule from the dd's own
  recorded provenance, most specific first:
  1. an explicit convention stamp in a provenance comment, e.g.
     `core_profiles.ids_properties.comment = "... p_fast_reduction=trace ..."`;
  2. IMASdd.jl-only top-level keys (`global_time`, `requirements`, `build`,
     `balance_of_plant`, `solid_mechanics`, `costing`) ⇒ `"sum"`. This
     identifies the *writer of the file*, which is what sets the convention, so
     it outranks per-IDS producer names — a FUSE dd legitimately carries IDSes
     imported from other codes;
  3. producer names in `{dataset_description, core_profiles, equilibrium,
     summary}` × `{ids_properties.{comment,provider,source},
     code.{name,description,repository}}` — FUSE/IMAS.jl ⇒ `"sum"`,
     OMAS/OMFIT/IMASPy ⇒ `"trace"`.
* **Undeterminable provenance falls back to `"sum"` and warns loudly, once**,
  naming both conventions, the factor-of-3 stake, and how to set the rule
  explicitly. It is never applied silently. The warning is raised only where the
  choice actually moved a number: it is held back when the dd's fast pressure is
  absent or identically zero, and when `FixedComponentsConfig.p_fast` supplies
  `p_fast` instead of the dd. `Baseline.p_fast_meta` records the resolution
  either way (`basis="undetermined-fallback"`, `warned=False`).
* **A convention stamp whose value is unrecognised warns**, naming the slot and
  the value, and the convention is inferred from structure/producer instead — a
  typo in a stamp is no longer silently discarded.
* **An explicit `"sum"` / `"trace"` / `"mean"` / `"perp"` always wins and is
  silent.**
* The rule used, the basis for it and the evidence are recorded on the new
  **`Baseline.p_fast_meta`**.
* **The missing-`pressure_fast_parallel` fallback is explicit.** The reader still
  closes with `p_par := p_perp`, but now warns and states what that means under
  the rule in force: `p_perp` under the full-pressure rules (unchanged from
  ≤1.3.1), `3·p_perp` under `"sum"` — where `2·p_perp` is the competing reading
  if the producer simply omitted an all-zero parallel field. Under a bare `"sum"`
  default this path would have tripled silently.
* **`physics.isotropize_fast_pressure(p_perp, p_par, method)` now requires
  `method`.** No default is safe for both conventions; a direct caller gets a
  `TypeError` rather than a silent 3×.
* The completeness backstop (`equilibrium.pressure` vs the reconstruction) now
  reports the **signed** direction, names the reduction rule in force, and claims
  the `p_diff` anchor absorbs the gap only when
  `anchor_pressure_to_equilibrium` is actually on (it is off by default).

### What this changes for you

* **Configs that omit `p_fast_reduction`** resolve to `"auto"`. On a FUSE dd that
  is `"sum"` — 3× the ≤1.3.1 `p_fast`. Results produced before and after this
  change are not comparable on the IMAS path unless the rule was pinned.
* **Configs that pin `"trace"`** keep `"trace"`. The shipped
  `examples/D3D-like/slurm_jobs/bouquet_2000ms_bundle.json` pins it on purpose —
  it runs against the synthetic dictionary-convention dd — and now says so in a
  `_p_fast_note`.
* **A dd with no recorded provenance warns once per process**, on reads where
  its own fast pressure is non-zero, until the rule is pinned or the dd is
  stamped. Stamping is one line:
  `core_profiles.ids_properties.comment = "... p_fast_reduction=sum ..."`.
* `examples/D3D-like/D3Dlike_baseline_omas.json` carries that stamp now. The
  local (gitignored) generator that produces it should emit it too.

## Unreleased — the Z_eff envelope is measured, not assumed

**Behaviour change on the reconstruction/IDA path.** The Z_eff perturbation
width used to be `zeff_scalar_sigma · |Z_eff|` — an *assumed* 5 % — even when
the IDA `.cdf` carried a measured uncertainty. It is now taken from the file
wherever the file supports it, through a three-tier ladder selected by the new
`UncertaintyConfig.zeff_sigma_source` (default `"auto"`):

| Tier | Source | Typical width |
|---|---|---|
| carbon-propagated | `n_12C6_err` (direct) or the dilution posterior (ensemble), propagated with `sigma_ne` | ~2 % of Z_eff in-core |
| VB-measured | `Zeff_err` (direct) or the `Zeff` sample spread (ensemble) | ~8–9 % in-core, much wider in the SOL |
| scalar | `zeff_scalar_sigma` × abs(Z_eff) | the assumed 5 % |

Because Z_eff is the primary density channel, this rescales **every `n_i` /
`n_z` band in the ensemble** — by up to a factor of ~4 either way, depending on
which tier a given file reaches. Runs before and after this change are not
comparable on the recon+IDA path unless `zeff_sigma_source="scalar"` was set.
Users with no IDA file, or with `zeff_sigma_source="scalar"`, are bit-identical
to ≤1.3.1: the scalar expression is unchanged and no channel is added or
removed, so the sampler sees the same `user_sigmas` in the same order.

* **Both measured tiers are eligible only when the Z_eff baseline is itself the
  IDA one** — the recon path, and the *same* file that supplies the sigmas.
  Pairing a FUSE (IMAS/`ida_hybrid`) or p-file baseline with an IDA envelope
  would mix channels, so it falls back to the scalar. The file test compares
  **resolved** paths (`expandvars` → `expanduser` → `abspath` → `realpath`, then
  `os.path.samefile`), so a relative, `~`-prefixed, trailing-separator or
  symlinked spelling of the same file stays eligible.
* **No step down the ladder is silent.** Each emits a single `UserWarning`
  naming the tier chosen, every tier skipped, and the reason class (*source
  ineligible* / *missing dataset* / *invalid data*); the same record comes back
  as `resolve_uncertainty()`'s `"zeff_sigma_tier"` metadata and is printed under
  `[sigma]`.
* **Non-physical carbon data drops the carbon tier rather than corrupting it**,
  in **both** IDA layouts. Negative `n_12C6` (SOL spline undershoot), netCDF
  fill values (~1e36) and NaN holes all survive the downstream clip floors and
  produce enormous-but-valid-looking sigmas, so any bad radius (direct) or bad
  sample point (ensemble) drops the tier with a counted, printed reason.
* A 1-sigma array with **negative entries** is refused as corrupt rather than
  read as a wide band, with a counted reason; shape, non-finite, negative and
  all-zero are four distinct recorded refusals.
* `uncertainty.zeff_sigma_source` is validated in `BouquetConfig.__post_init__`,
  so a typo raises at construction rather than after the baseline GS solve.

### Also in this change: `impurity_Z` now reaches the IDA reader

`resolve_uncertainty` previously called `read_ida()` without `impurity_Z`, so
the sigma read always used the default **Z = 6** regardless of the source's own
`impurity_Z`. It now passes `source.impurity_Z` through, matching what the
kinetics loader has always done.

**This moves `sigma_ni`, not only the new Z_eff channel**, for any user whose
source sets `impurity_Z != 6.0`: `ni = n_e − Z·n_C` is re-derived at the
source's real charge, and the ion-density sigma follows. Users on the default
carbon `impurity_Z = 6.0` are unaffected.

## Unreleased — the structured closure defaults to its validated preset

### The structured closure's default prior is now the validated one

`closure_channel="structured"` with no `structured_preset` used to resolve to
the raw shipped fields — the symmetric physics ladder (`W_ind` 100/10/3/1,
`W_bs` 1/3/10/100) on the hard KKT solver, I_p imposed exactly. That is the
configuration this channel's own l_i study **superseded**, and the validated one
was reachable only by naming `structured_preset="li_soft_onesided"`. A default
nobody is expected to want is a trap, so the validated preset **is** the default
of the channel now.

* **Default.** `closure_channel="structured"` and no preset → `li_soft_onesided`
  (`utils.STRUCTURED_PRESET_DEFAULT`): σ_bs `(0.50, 0.30, 0.15, 0.10)`,
  σ_ind,down `(0.10, 0.40, 0.40, 0.40)`, σ_ind,up `(0.10, 0.10, 0.10, 0.40)`,
  `structured_soft=True`, `structured_ip_sigma_frac=0.005`, and σ_li `0.04`
  **only** when `structured_li_target` is set. Without an l_i target it degrades
  exactly as the named preset always has: soft I_p plus the one-sided prior, no
  l_i row. The `UserWarning` names every field filled and says `BY DEFAULT`.
* **Opt-out: `structured_preset="none"`** (`utils.STRUCTURED_PRESET_NONE`) —
  declines the default and leaves every structured field as shipped, reproducing
  the previous behaviour exactly. Every explicit configuration that worked
  before still works, unchanged.
* **Nothing changes off the structured channel.** The code-wide default is still
  `closure_channel="bootstrap"`; with any other channel and no preset named, no
  field is touched and no warning fires.
* **Explicit settings still win**, by the same rule as for a named preset (a
  field set to its own default value remains indistinguishable from an unset
  one, and is still named in the warning). Two guards keep a *default* from
  breaking a configuration that ran before: an explicit `structured_basis`
  declines the default outright (the ladders are widths at the shipped basis's
  radii, meaningless on another basis), and an explicit absolute
  `structured_ip_sigma` suppresses the `structured_ip_sigma_frac` fill it would
  otherwise clash with. A preset named explicitly behaves as it always did in
  both cases.
* **Recorded.** `structured_preset_in_force` / `structured_preset_source` /
  `structured_preset_fields` on the config, and `structured_preset`,
  `structured_preset_source` (`default` | `explicit` | `opt-out` |
  `default-declined-custom-basis` | `unset`) and `structured_preset_filled` in
  `Baseline.ip_closure`. Resolution happens at construction **and** at the
  closure's entry point (`config.resolve_structured_preset`, idempotent), so a
  `closure_channel` set after the config was built is resolved too.

**Scope, stated honestly.** These σ values were set from a study on **one device
with one integrated-modelling source** for the inductive current. They are
**priors in relative units** — fractions of the component profiles, on
normalised flux — not device constants; that is why they transfer at all, and it
is not a claim that they are right elsewhere. On another device or another
`j_ind` source they are a **starting point**: check the recorded closure-health
flags (the `0.2 < s < 5` scale bounds, `|s_bs − 1| > 0.5`, the q0 miss, the l_i
z-score), and run `STRUCTURED_WEIGHTS_UNIFORM` as the no-prior sensitivity.

No tolerance, bound, gate or acceptance criterion moved. A preset is a prior: it
changes which exactly-closing profile is chosen, never what "closed" means.

## 1.3.0 — the seeded draw is now machine-independent; find_ida (2026-08-05)

1.2.0 shipped the contract "same seed → bitwise-identical archives". True on
one machine; **not** true across machines: the CI golden's j_phi[0] differed by
**1.3%** between the macOS build that pinned it and the Linux runners, and the
draw-stream golden failed on CI from the hour it was written. This closes that
gap.

### What was wrong

`GPRProfilePerturber.generate_profiles` factorised the kernel with
`np.linalg.eigh` and used the eigenvectors raw. A squared-exponential kernel is
numerically rank-deficient — on the golden's 257-point j_phi grid, 242 of 257
eigenvalues sit at the double-precision floor (cond(K) ~ 8e19) — so LAPACK
returns two things it is free to choose: an **arbitrary basis** for that null
subspace, and an **arbitrary sign** per eigenvector. Both leave `V Λ Vᵀ`
exactly unchanged — the sampling law was never wrong — but both change the
individual draw. Under an eigensolver A/B (LAPACK syev/syevd/syevr) the same
seed moved by up to ~20% on flat-sigma kernels.

An intermediate fix (null-mode truncation + per-vector sign canonicalisation)
was built and adversarially reviewed, and is worth recording because its
failure mode is instructive: a flat sigma on a uniform grid makes K
mirror-symmetric, its eigenvectors then peak at mirrored index pairs with
equal magnitude, and for the antisymmetric modes those two carry opposite
sign — so "the largest component" is decided by floating-point noise that
grows as `eps·λ_max/λ` and defeats any fixed tie window. 35 of 72
production-plausible configs still moved by >1e-7 across eigensolvers, and no
per-vector rule can fix a rotation *within* a (near-)degenerate eigenspace at
all. Per-vector canonicalisation of `eigh` is a dead end; it never shipped.

### The fix

The draw is now `L z` with `L` the **fixed-order Cholesky factor** of
`K + jitter·I` (jitter = 1e-10 · n · max diag(K)). Cholesky has **no discrete
choices** — no eigenvector signs, no null-space basis, no pivots, no ties — so
there is nothing for a different LAPACK build to decide differently, for any
kernel, length scale, or sigma shape; cross-build variation collapses to
rounding accumulation. `z` keeps its full length on every path (including
sigma = 0), so RNG consumption and channel sequencing never depend on the data.

Measured (`tests/test_rng_reproducibility.py::TestDrawIsFactorizationStable`,
sub-ulp kernel perturbation as the stand-in for a different LAPACK/libm build):

| property | raw eigh | Cholesky |
|---|---|---|
| golden channels, cross-build residue | 1.3% (observed on CI) | ~1e-9 |
| flat-sigma matern52 (worst prior config) | up to ~20% (driver A/B) | 3.6e-10 |
| factorisation success, 151-config sweep | — | 151/151 |
| marginal std vs sigma envelope | exact | +3e-8 relative (jitter) |
| where sigma_i = 0 exactly | exactly 0 | residual std ~2e-4·max(sigma); all-zero sigma stays exactly 0 |

The full covariance was re-verified empirically (20k draws match `K` to the MC
floor), and the law tests assert it in-suite.

### What this changes for you

**Drawn values differ from ≤1.2.0 for the same seed** — up to ~21% pointwise on
the golden fixture's Te channel — a different draw from the *same*
distribution, not a correction to any one draw. Seeded ensembles generated
under ≤1.2.0 do not reproduce under 1.3.0; regenerate them. Nothing about the
physics, the solve, or the sigma semantics moves.

### The golden's claim, corrected

The 1.2.0 golden asserted the stream was "bitwise identical on any machine".
It cannot be — rounding inside LAPACK/libm is build-dependent — so
`rng_stream_manifest.json` now records the machine that pinned it
(`pinned_on`: platform/arch/numpy/BLAS), and the test asserts

* **numerically everywhere** — `rtol = 1e-7`, two decades above the ~1e-9
  residue measured on the golden's own channels, so a real sampler change
  still trips it;
* **bitwise on the pinning machine** — the SHA-256, which is the contract
  seeded `generate()` runs actually rely on.

### The systematics golden, regenerated — and its fixture made deterministic

`test_systematics`'s mode-3 golden (l_i 0.8521) was **unreachable by released
code**: pristine 1.2.0 at `nthreads=1` lands 0.8810, bit-stable, under any
sampler and any Ip-measure mode. The stored values dated to `0b00eb7`, with
several intentional physics changes landed since and never re-pinned; the
fixture's `nthreads=2` solver jitter (±1%, amplified through the l_i-matching
loop) let the test sometimes pass anyway, so the staleness was invisible —
solver tests do not run in CI.

The golden archive is regenerated with current code (class API, 20 draws, seed
12345, ψ-dependent synthetic-IDA sigmas, flat 10% j_φ σ, input-current
archival — `store_achieved_jphi=False`, because a replay premise "archived
current reproduces archived LCFS" requires the input, not the achieved,
current). The replay fixture now runs at `nthreads=1` and stands its solver up
**through the class API's own reconstruction**, inheriting the generator's
isoflux set, weights, warmstart state, and `psi_pad` — four hand-copied
constants (weight 200 vs 500, pad 1e-4 vs 1e-3, boundary point set, warmstart)
each produced a deterministic ~2 mm replay offset when they drifted from the
generator. All three modes now pass bit-identically across repeated runs.
`make_golden_fixture` additionally learned the schema-v2 dataset names
(`eqdsk`/`coil_currents`), which it silently mishandled before.

### New: `find_ida` — locate kinetic-profile data that isn't in the repo

`bouquet.find_ida(name, start=..., extra=...)` resolves an IDA `.cdf` at run
time, so analysis repos stop carrying machine data (current vintages reach
190 MB — past GitHub's 100 MB hard limit — and are regenerated as IDA-lite
evolves). Precedence: `BOUQUET_IDA` naming a **file** → `extra=` → a walk-up
from the notebook (the sibling copy wins) → `BOUQUET_IDA` naming a
**directory**, searched recursively with a depth cap. Because two vintages of
one shot commonly share a basename and both load silently, a directory search
that finds differing copies **raises** listing them rather than guessing, and
a miss raises listing every path tried.

## 1.2.0 — reproducibility contract + trustworthy R2 Ip renormalisation (2026-08-04)

Four fixes found while driving a β-scan through `generate()` at σ=0. All four
are backwards compatible; production `generate()` defaults are unchanged.
(Fix 4 supersedes the *diagnosis* in fix 2, but not its behaviour, which stays
the default — read them in order.)

### 1. The seed now reaches the GPR — *the reproducibility contract*

`GenerationConfig.seed` was consumed by `np.random.seed()`
(`TokaMaker_interface.py:2717-2719`), but every one of the nine GPR draw sites
called `generate_perturbed_GPR(..., rng=None)` and `sampling.py:186-187`
answers that with `np.random.default_rng()` — **fresh OS entropy per draw**.
Only `np.random.uniform` (`scale_jBS`) and `np.random.normal` (the per-draw
`l_i` target) honoured the seed. Seeded ensembles were not regenerable, and no
draw-level value could be pinned as a golden. `parallel.py:158` inherited the
same defect through `GenerationConfig.seed`.

Fixed as parameter plumbing, not global state:

* **`sampling.make_rng(seed)`** is the single seed → `numpy.random.Generator`
  entry point (an existing Generator passes through; `None` = OS entropy).
* `generate_bouquet` consumes `seed` **exactly once** into that Generator and
  threads it into every draw: `rng=` on the `perturb_kinetic_equilibrium`
  call, and `rng.uniform` / `rng.normal` replacing the two legacy calls. **One
  seed governs everything.**
* `perturb_kinetic_equilibrium` grew an `rng` argument and passes it to all
  nine sites; `_draw_monotonic_perturbation` too, so its rejection loop — which
  consumes a *variable* number of draws — stays on the run's stream instead of
  desynchronising every later channel.
* `np.random.seed(seed)` is retained only so third-party code in the solve path
  stays deterministic; bouquet's own draws no longer read global state.
* Parallel shards are unchanged in behaviour and now documented: `_derive_seed`
  already derives each shard deterministically from `(seed, worker_id,
  scan_key)` via `SeedSequence`, and that int becomes the shard's Generator, so
  a parallel run is reproducible for a fixed `n_workers`.

**Contract:** same seed + same inputs + same solver → **bitwise-identical
archive**. `seed=None` keeps the OS-entropy behaviour.

### 2. The R2 `I_p` renormalisation is evaluated on the anchor geometry

Route R2 (`perturb_jind_in_anchor=True`) sets the inductive **amplitude** from
`Ip_flux_integral_vs_target` while holding the bootstrap fixed — the correct
bookkeeping, since an `I_p` constraint should move the ohmic drive only. Two
defects made it untrustworthy. Both measured at σ=0 on the synthetic D3D-like
example, where the archived split *is* the answer and the root must return
`1.000`:

1. **Geometry.** `TokaMaker_interface.py:1810-1815` rooted *after*
   `solve_with_bootstrap`, so `mygs.flux_integral` saw SWB's landed
   equilibrium. Anchor geometry gives `0.8524` vs the landed geometry's
   `0.8373` — 1.80 % of inductive amplitude for no physical reason.
2. **Convention normalisation** (found while validating 1).
   `compute_flux_integral` is a faithful `∫f dA` — verified `FI(1)` == plasma
   area and `compute_area_integral(calc_jtor_plasma)` == `I_p` to 1e-7 relative
   — but bouquet's currents are the FSA toroidal density `<j_φ/R>/<1/R>`, whose
   area integral is **not** `I_p`. The archived total integrates to **+12.92 %**
   of `I_p`, in *any* geometry — by far the larger part of the R2 error.

`_AnchorIpRenorm` fixes both: `copy_eq()` pins the anchor equilibrium
immediately after the state-anchor solve and every flux integral runs on that
frozen snapshot (`mygs` is never mutated), and the demand is calibrated as
`FI(archived total) * Ip_target / Ip_anchor` instead of raw `Ip_target`, which
cancels the representation bias and makes the golden invariant exact by
construction while preserving `I_p` retargeting.

| σ=0, D3D-like | scale `s` | `l_i` vs recon |
|---|---|---|
| before | 0.837339 | −2.008 % |
| after  | 0.999150 | +0.100 % |

Bit-identical across repeats. Cost: one `copy_eq` (0.1 ms) plus an analytic
root (2 flux integrals + a linearity check, ~62 ms, replacing brentq's
~125 ms) against a ~26 s perturb call. `BOUQUET_R2_IP_MODE=legacy` restores
the old behaviour for A/B.

**Scope: route R2 only.** The standard `l_i` loop — the production ensemble
path, `perturb_jind_in_anchor=False` — is untouched and bit-identical; its root
is followed by `find_optimal_scale` + the corrective iteration, which re-derive
the amplitude from the solved equilibrium. The anchor snapshot is not even
captured off R2.

Also: `perturb_kinetic_equilibrium` diagnostics carry `r2_ip_scale` (and
`generate_bouquet`'s per-draw diagnostics carry `scale_jBS`), and
`run.py:_validate_workflow` no longer hard-errors on geqdsk +
`perturb_jind_in_anchor` — that guard existed because of this defect. It prints
a one-line note instead. R2 remains opt-in.

### 3. The kinetic-sigma precedence is no longer silent

`resolve_uncertainty` resolves each channel as `sigma_profiles` > IDA `.cdf` >
`<chan>_scalar_sigma`, and `baseline.py:199-201` auto-adopts
`ReconstructionSource.profiles_path` as the IDA source whenever it ends in
`.cdf`. A winning source shadows the ones below it, silently — so zeroing
`*_scalar_sigma` to get a deterministic run is a **no-op** against an IDA
source, and every "deterministic" point is a full-σ draw.

The precedence is the intended design, so this makes it loud rather than
changing it:

* one `[sigma-source]` line per channel naming the winner and the resolved
  peak, flagging `ALL ZERO` explicitly — gated on the new
  `UncertaintyConfig.log_sigma_sources` (default `True`);
* a `UserWarning` when a `<chan>_scalar_sigma` moved off its dataclass default
  but lost to an active IDA file, naming the file and giving the
  `sigma_profiles` recipe that actually works. Untouched defaults do not warn;
* `UncertaintyConfig`'s docstring states the precedence as a table.

### 4. A real `I_p` measure: `utils.Ip_fsa_integral`

Fix 2 cancelled a "+12.9 % convention bias" with a ratio calibration. Chasing
where that 12.9 % actually comes from turned up two separate errors, neither of
them the one fix 2 named:

* **`compute_flux_integral` is not `∫_plasma f dA`.** It integrates over the
  whole `reg == 1` limiter region, and off the plasma the flux-function
  interpolator returns the profile's **LCFS value** (`gs_prof_interp_apply`
  CASE(4) returns 0 — the LCFS end of the internal flux coordinate — and
  `gs_flux_int` then reads the profile there). `FI(1) = 2.83853 m²` is the
  limiter area, **not** the plasma cross-section, which is `1.79005 m²`; fix
  2's note to the contrary is wrong. For the archived total the 1.05 m² excess
  is charged at the edge value, `1.36e5 A/m² × 1.05 m² = +1.43e5 A` = **+11.9 %
  of `I_p`** — essentially the entire bias.
* **The remaining ~1 % is a convention, but not the assumed one.** bouquet's
  arrays are TokaMaker `jphi-linterp` values, `J = <R> P' + <1/R> FF'/µ0`
  (`jphi_update`), not the FSA density `<j_φ/R>/<1/R>`. The two differ by
  `<R><1/R²>/<1/R>`, up to 11 % per surface at the edge.

`utils.Ip_fsa_integral` replaces the mesh integral with the textbook
axisymmetric current integral, `I_p = ∫ dψ (V'/2π) <j_φ/R>`, taking `V'`,
`<R>`, `<1/R>` and `<1/R²>` from `get_q`'s `ravgs` dict and folding in the
`jphi-linterp` conversion. Supporting helpers: `fsa_current_geometry` (the
per-surface arrays), `Ip_fsa_weights` (`I_p[J] = trapezoid(w·J) + c` — the
measure is **affine**, the `P'` term is −3.3 % of `I_p` and lands in `c`), and
`eq_jphi_profile` (the equilibrium's own profile in either convention).

**Validation** (D3D-like, `tests/test_fsa_current_integral.py`): integrating the
solved equilibrium's *own* current profile returns its true `I_p` to
**+0.0071 %** against a required 0.1 %, in both conventions. On the *archived*
total the `jphi-linterp` reading gives +0.068 % and the `fsa` reading +0.927 %;
the former agrees to 0.011 % with the solver's own internal `jphi_norm`.

Two getter traps are now closed in code rather than by luck:

* `get_q(psi=…)` **silently collapses onto the magnetic axis** if the sample
  grid contains `psi_N = 0` — `<R>` constant to 2e-15 across all 257 surfaces,
  no exception. `fsa_current_geometry` clips to `[psi_pad, 1-psi_pad]` and
  raises if it sees the collapse anyway.
* `dV/dPsi` is per **dimensional** ψ (`∫ dV/dPsi dψ` recovers the volume to
  −0.25 %; the `dψ_N` reading is out by +291 %).

`_AnchorIpRenorm` gains the measure as `BOUQUET_R2_IP_MODE=exact` (and the
literal FSA-density reading as `fsa`), alongside `ratio` (fix 2's calibration,
also spelled `anchor`) and `legacy`. Every FSA getter runs on the frozen
`copy_eq` snapshot — verified bit-identical to the live solver, and verified
not to perturb it — and the weights are cached as arrays at capture time, so
after `__init__` the root needs no solver call at all. The class self-checks
the measure against the anchor's own profile at runtime (+0.014 % here) and
prints it.

**The default is still `ratio`, deliberately.** Measured at σ=0:

| mode | `s` | `\|s−1\|` | `l_i` vs recon |
|---|---|---|---|
| `exact` | 0.996750 | 3.3e-3 | +0.130 % |
| `fsa` | 0.985600 | 1.4e-2 | −0.093 % |
| `ratio` | 0.999150 | 8.5e-4 | +0.100 % |
| `legacy` | 0.837339 | 1.6e-1 | −2.008 % |

The correct measure reproduces `s == 1.000` **less** closely, for an understood
reason: `ratio` is exact by construction, because it asks the draw to carry the
same mis-measured integral as the archived total, so every representation error
cancels. `exact` asks for `Ip_target` in real amperes and therefore also
charges the draw for the reconstruction's own `j_φ` residual (the archived
total differs from the anchor's own profile by 1.6 % of peak *in shape*, worth
+0.193 % of `I_p` at the R2 state anchor → −0.25 % of inductive amplitude) on
top of the σ=0 SWB residual (−0.085 %). −0.335 % predicted, −0.325 % measured.
Both terms are real. Even a perfectly self-consistent archive would leave
~1.1e-3, so the pinned `|s−1| ≤ 1e-3` invariant is **not attainable by any
honest measure** on this case — flipping the default is an acceptance-criterion
decision, not a code change, and is left to the author
(`_R2_IP_MODE_DEFAULT`).

**The production `l_i` loop is untouched** (see below), but now measured: with
`perturb_jind_in_anchor=False` on a seeded 2-draw `generate()`, the loop's root
returns `a = 0.785003` and `0.928871` where the FSA measure gives `0.920811`
and `1.078281` — the loop absorbs a **+16.1 % to +17.3 %** bias in inductive
amplitude (+14.9 % / +15.7 % under the `fsa` reading), which
`find_optimal_scale` + the corrective iteration then re-derive away. The
measure self-checks to +0.015 % at those states.

### Tests

* `tests/test_fsa_current_integral.py` — fast half: the affine algebra, a
  circular large-aspect-ratio geometry with an analytic answer, and that every
  unsupported combination raises instead of returning a plausible number.
  `solver` half (subprocess): the 0.1 % self-consistency validation, snapshot
  ≡ live, the `dV/dPsi` Jacobian, the silent `get_q` collapse, and that
  `compute_flux_integral(1)` is still the limiter area (so the rationale is
  re-checked if OFT changes the interpolator).
* `tests/test_seeded_reproducibility.py` also A/Bs `BOUQUET_R2_IP_MODE=exact`:
  its own derived bar `_S_ATOL_EXACT = 5e-3` (measured 3.25e-3) — a **new pin
  on new behaviour, not a widening of `_S_ATOL`**, which still governs the
  default path — plus the same 0.5 % `l_i` acceptance, bit-reproducibility, and
  that changing the measure leaves `j_BS` untouched.
* `tests/test_rng_reproducibility.py` (fast) — `make_rng`; samplers honour an
  injected Generator; an AST check that **every** draw call site passes `rng=`
  (the defect was invisible at runtime, so only a structural assertion prevents
  its return); and a committed bitwise golden of the seeded draw stream.
* `tests/golden/rng_stream_manifest.json` — the first draw-level golden the
  package can hold at all. Pure NumPy, so it is bitwise-portable. Re-pin with
  `python tests/golden/make_golden_fixture.py --rng-stream-only`.
* `tests/test_seeded_reproducibility.py` (`solver`) — two seeded runs produce
  bitwise-identical archives (with `jBS_scale_range` and `l_i_uncertainty` on,
  so the two previously-seeded streams cannot regress while the GPR is fixed),
  and the σ=0 R2 invariant `s == 1.000`, `l_i` within 0.5 %, `j_BS` within the
  σ0-guard bar, bit-reproducible.
* `tests/test_sigma_precedence.py` (fast) — resolution, the warning, the log.

No existing golden needed re-pinning: `golden_manifest.json` and the slim `.h5`
are a frozen artifact read by the tests, and `test_systematics.py` replays at
σ=0, so nothing depended on the draw stream (it could not have — an unseeded
stream would have made such a test flaky).

## 1.1.0 — machine-neutral API + comment hygiene (2026-07-31)

Ships the untagged 1.0.1 fix as well (`verify_sigma0_consistency` raised on the
IMAS path: it read `psi_pad`, a `ReconstructionSource`-only field).

1. **BREAKING — `ImasSource.efit01_geqdsk` -> `LCFS_geqdsk`** (also the
   `Bouquet.from_imas` keyword). The field is an *optional* external separatrix,
   not a specific EFIT tree: supply a g-file whose LCFS replaces the source dd
   boundary outline, or omit it and keep the dd's own. No alias — the old keyword
   now raises `TypeError` rather than being silently ignored, which would have
   changed the boundary the draws are held to. All EFIT01/EFIT02 tree names are
   gone from the code, docs and flowchart.
2. **`radial_field_from_impurity_force_balance`** replaces
   `radial_field_from_cer` (`n_imp`/`t_imp`/`Z_imp`/`sigma_*_imp` instead of the
   carbon-specific spellings). The physics is generic impurity force balance;
   only the diagnostic was DIII-D. `radial_field_from_cer` is kept as a
   forwarding alias, so existing scripts keep working.
3. **Discharge identifiers removed from code comments** (12 sites). Each is now
   a descriptor of why the case mattered — "stiff high-l_i case",
   "strong-pedestal case", "low-current case" — with the measured numbers
   (0/500 candidates, ~2 permille, 0.347 vs 0.313 MA/m^2) kept intact.
4. **Examples pinned to `nthreads=1`**, matching the doctrine the docs already
   state: `_run_omas_timeseries.py`, `generate_baseline.py`,
   `bouquet_D3Dlike_systematics.ipynb` and the legacy example no longer default
   to 2 or 4 threads. Bit-reproducible solves, no BLAS oversubscription.

`kinetic_source="ida_hybrid"` is deliberately unchanged: it names a specific
workflow and input file format, unlike a tree name standing in for any g-file.

## Class API + HDF5 schema v2 round (2026-07, PR #8)

Decisions and outcomes folded from the (deleted) working document
`docs/ux-review-feat-bouquet-class-api.md`:

1. **Config serialization + provenance**: `BouquetConfig.to_dict/from_dict`
   (JSON); every archive stores `schema_version` / `bouquet_version` /
   `created` (stamped at file creation) + per-scan `config_json`
   (`write_provenance` / `load_config`).
2. **`BouquetArchive`** reader class (ScanView/DrawView, lazy, cached attrs,
   fixed-name + legacy suffix-scan eqdsk/pfile lookup) — replaces downstream
   hand-rolled h5 traversal.
3. **De-threaded readers**: module-level plot/filter/select functions accept a
   `Bouquet` / `BouquetArchive` / header / path uniformly; a missing explicit
   `scan_key` raises listing available keys (was silent-empty).
4. **Schema v2 — clean break, no legacy readers** (decision: no external users
   yet): bare dataset names + `units` attrs, fixed `eqdsk`/`pfile` names,
   coil names as a string dataset everywhere, `scan/<key>/` layout only.
   Legacy files: `BouquetArchive` warns, `load_equilibrium` raises clearly.
   Spec: `docs/archive-schema.md`; source of truth: `bouquet/schema.py`.
5. **Config/API simplifications**: `run.describe()` (non-default knobs),
   `workflow` preset enum (`auto` / `geqdsk-standard` / `imas-diff-c` /
   `custom`), source-agnostic `run.prepare()`.
6. **Sweeps + plotting**: `run.run_slices()` (one archive, one scan_key per
   slice); `plot_bouquet` dispatches on stored `source_kind`
   (`plot_imas_bouquet` demoted from `__all__`).
7. **Parallel hardening**: fresh shards, merge-side baseline guard +
   expected-shard accounting (`--allow-missing`), nthreads=1 doctrine warning,
   `SeedSequence(seed, worker, scan_key)` slice-decorrelated seeding, JSON
   SLURM bundles, CWD-independent submit scripts, physical-core default.
8. **Post-review fixes** (8-angle adversarial review of the final diff):
   solver-marked tests un-broken (coil_names dataset read), merged archives
   get run-level `config_json`, multi-scan `load_config` disambiguation,
   `DrawView` O(N) flags + cached attrs, `schema.find_bytes_dataset`
   consolidation (8 duplicated lookups), unique eqdsk extraction filenames,
   legacy-archive warnings/errors.
9. **Won't-do (user decisions)**: `jphi_scalar_sigma` default stays 0.10;
   IDA_run per-shot notebooks stay split (no templating).
10. **Deferred**: example-notebook rewrite to run-object idioms;
    reconstruction-style IMAS baseline summary block.

Also in this round: scipy≥1.18/numpy≥2.5 compatibility (`.item()` at the
axis-point `.ev()` call), and the CER/E_r feature (`read_ida_cer`,
`radial_field_from_cer`).

---

# Bouquet + OFT — change summary (golden suite, filtering, Ip-secant removal, systematics)

> **Historical document** (2026-05): a snapshot of the coil-bounds/golden-suite
> round of work, kept for context. Branch/repo layout below reflects the
> author's working setup at that time; PR #3 (feat/coil-bounds) has since
> merged to `main`.

| repo | path | branch | remote |
|---|---|---|---|
| OpenFUSIONToolkit (fork) | `OpenFUSIONToolkit/` | `feat/jphi-linterp-Ip-cutcell-fix` | `d-burg/OpenFUSIONToolkit` |
| bouquet backend (package + tests) | `bouquet_coil_bounds/` | `feat/coil-bounds` | `d-burg/bouquet` |
| bouquet examples (D3D-like) | `bouquet/` | `feat/lock-coils-pr1` | `d-burg/bouquet` |

> The two bouquet clones are the same repo on different branches; the backend
> path-insert in the notebooks is temporary until they're merged into one
> shipped `bouquet`.

---

## 1. OpenFUSIONToolkit — native Ip hold + bootstrap cleanup

**Already committed** on `feat/jphi-linterp-Ip-cutcell-fix` (the Ip "hot fixes"):
- `5b06f66` jphi-linterp Ip-correction **outer iteration** in the GS solve.
- `328163f` restore `Itor_target` at outer-loop exit.
- `97436b2` opt-in trace prints (`oft_debug_print`).
- `b032e4f` safety-bail on corrupted `ip_phys`.
- (plus the earlier cut-cell / `jphi_update` structural fix and the
  `#248` `TokaMaker_equilibrium` merge.)

These make the Fortran backend hold `Ip` to target natively (≈0.05 %),
which is what lets us delete the Python Ip workarounds below.

**New, uncommitted** (`src/python/OpenFUSIONToolkit/TokaMaker/bootstrap.py`):
- Removed the **`find_j0=False` (Ip-scale) secant** call from
  `solve_with_bootstrap` → `final_scale_Ip = 1.0` (Ip held natively).
- Stripped the now-dead `find_j0=False` branch + `get_Ip_error` + the
  `find_j0`/`scale_j0` params from `find_optimal_scale`; it is now a clean
  **core-j0-only** scaler. Callers updated.
- (Mirrored into `build_release/` + `install_release/` so the running env
  matches; only `src/` is tracked.)

→ **PR target:** `feat/jphi-linterp-Ip-cutcell-fix` → upstream (or fork main).
See `OpenFUSIONToolkit` PR body draft.

---

## 2. bouquet backend (`bouquet_coil_bounds/`, `feat/coil-bounds`)

### X-point detection (TokaMaker, not geometric)
- Capture `mygs.get_xpoints()` at generation time (baseline + per draw) at the
  same solver state the eqdsk is saved from; store `x_points` dataset +
  `diverted` attr in the H5 (`utils.store_equilibrium` / `store_baseline_profiles`).
- `plot_boundary_point_traces` now uses the stored true B_p=0 saddles
  (`_xpoints_on_lcfs`) instead of the erratic geometric corner finder; clean
  fallback to the axis-line intersection for pre-existing H5s.
- Added `utils.list_equilibrium_indices()`; fixed a `KeyError` in
  `plot_coil_currents` (and silent last-draw drop in other plots) on
  band-rejection index gaps.

### Postprocessing filters (`bouquet/filtering.py`, new)
- `filter_coil_currents()` and `filter_boundaries()` — **non-destructive**:
  write `passes_coil_filter` / `passes_boundary_filter` + derived `selected`
  flags, return `(summary, distribution-figure)`. Coil thresholds default to
  the stored `inspec_F_max/VSC_max` (reproduce in-loop in_spec); boundary filter
  is diagnostic-only until a `rms_max_mm`/`max_max_mm` is given.
- `read_filter_flags()`, `select_indices('all'|'selected'|'excluded')`,
  `export_filtered()` (pruned copy, baseline preserved, source untouched).
- `plot_bouquet(..., selection=...)` honours the flags.

### Units, flags, RNG (no env vars / no percentages in the user API)
- `l_i_tolerance` and `p_thresh` are now **fractions** (e.g. `0.05`), converted
  to percent internally; defaults updated.
- `jphi_baseline=True` flag replaces the `JPHI_BASELINE` env var.
- `pin_jphi=False` flag replaces the `PIN_JPHI` env var.
- `seed=None` parameter seeds the RNG inside `generate_bouquet`.
- (env vars retained only as back-compat overrides.)

### Removed all Python Ip-rescaling secants (kept core-j0)
- Per-draw post-perturb Ip-secant (was a no-op; dead block deleted).
- `reconstruct`/`fit_inductive_profile` **§6 Ip-correction secant** deleted
  (kept `Ip_desired` + `j_ind_li` that §7 consumes).
- OFT `find_j0=False` call (see §1).
- Inert `final_scale_Ip = 1.0` vestige in the l_i loop.
- **Kept:** all `find_j0=True` core-j0 secants + the l_i-match inductive secant.

### Tests + golden suite (`tests/`)
- `tests/golden/`: `D3Dlike_Hmode_golden_slim.h5` (~12 MB, **geqdsks retained
  gzip-compressed** for g-file handling tests, p-files dropped) +
  `golden_manifest.json` + `make_golden_fixture.py` (`--eqdsk all|subset|none`)
  + `README.md`. `.gitignore` negation `!tests/golden/*.h5`.
- `tests/test_golden_bouquet.py` (15 tests): scalars/coils/x-points/boundary
  vs manifest, geqdsk parse + separatrix-coarseness, filtering/selection/export.
- `tests/test_systematics.py` (2 tests, **opt-in** via
  `pytest -m solver`): σ=0 pinned → baseline (RMS<0.8 mm, drift<0.3 %).
- `tests/test_synthetic_sigma.py` (11 tests): the sine-basis IDA σ helper.
- **Protected proprietary data:** added `*.cdf`/`*.nc` to `.gitignore`
  (fixed a latent inline-comment bug) — operational `IDA_*.cdf` files must never
  be committed.

**Fast suite: 123 passed, 17 skipped (the 2 solver tests skip by default).**

---

## 3. bouquet examples (`bouquet/`, `feat/lock-coils-pr1`)

- **Non-proprietary baseline:** `D3Dlike_Hmode_baseline.geqdsk` / `.peqdsk` +
  `D3Dlike_Hmode_baseline_RECIPE.md` + overview PNG (147131-derived, COCOS 1,
  Ip +1.20 MA, Bt −2.0 T, physical bootstrap, smooth edge).
- **Updated example notebook** `bouquet_D3Dlike_example.ipynb`:
  top-level `REGENERATE` toggle (load golden vs rebuild), decimal knobs,
  `jphi_baseline`/`seed` flags, §9 filtering demo, **no shot-number references**.
- **New systematics notebook** `bouquet_D3Dlike_systematics.ipynb`: runs 3 modes
  (pinned σ=0 / pinned IDA-σ / production, n=10) and checks all traces + coil
  currents with signed-drift summaries to expose any bias.
- `legacy/` folder for the superseded example; builder/helper scripts.
- The full 30 MB golden run stays **untracked** (`*.h5`); only the slim fixture
  in the backend repo is tracked.

---

## 4. Validation

- **Recon Ip (native):** baseline **0.0000 %**, per-draw median +0.03 %, max
  0.04 % — confirms the OFT native hold; Python Ip secants were redundant.
- **No-systematic floor (σ=0 pinned, live solve):** boundary RMS **0.525 mm**
  (deterministic across draws), max coil drift **0.054 %**.
- **VSC drift (production golden):** signed F9A/F9B means −0.19 % / +0.24 %
  (vs σ≈3 %), non-VSC F-coils ±0.005 % → symmetric scatter, **no systematic
  bias**. The 10/20 in-spec yield is honest spread straddling the ±2 % spec.
- **New golden:** 20 draws, 10 in-spec, recon `l_i` 0.842; manifest diff vs the
  prior golden is small (l_i_target +0.2 %, slightly wider boundary/VSC spread).

---

## 5. New / changed public API

```python
generate_bouquet(..., l_i_tolerance=0.01, p_thresh=0.05,   # now fractions
                 jphi_baseline=True, seed=None, pin_jphi=False)
from bouquet import (filter_coil_currents, filter_boundaries,
                     select_indices, read_filter_flags, export_filtered)
plot_bouquet(..., selection='all'|'selected'|'excluded')
from bouquet import list_equilibrium_indices, synthetic_ida_sigma
# OFT: find_optimal_scale(...) is now core-j0 only (no find_j0/scale_j0 args)
```

---

## 6. Follow-ups

- **Open issue:** jphi-linterp edge / separatrix `j_φ` handling in reconstruction
  (the ~0.5 mm σ=0 floor) — see `docs/ISSUE_jphi_edge_reconstruction.md`
  (backend task #41).
- Modes 3–4 (free-jphi ±homotopy) broader validation (task #32).
- Eventually merge the two bouquet branches into one shipped package (removes the
  notebook path-insert).
