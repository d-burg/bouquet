# The bouquet HDF5 archive — schema v3

Authoritative description of the on-disk layout written by bouquet ≥ 1.0.0
(schema v2 first shipped in the 1.0.0 release; v3 adds the self-consistent
bootstrap record — see [v2 → v3](#v2--v3-the-self-consistent-bootstrap-record)).
The single source of truth in code is [`bouquet/schema.py`](../bouquet/schema.py)
(`SCHEMA_VERSION`, `PROFILE_UNITS`, fixed dataset names, `write_profile` /
`find_bytes_dataset`); this document mirrors it for human readers. Prefer
reading archives through [`bouquet.BouquetArchive`](../bouquet/archive.py) or
the functional readers (`load_equilibrium`, `load_baseline_profiles`,
`select_indices`, `load_config`) rather than raw `h5py` — see
[workflows.md](workflows.md#reading-an-archive-back) for worked examples.

## Layout

```
{header}.h5                            file attrs: schema_version (=3),
│                                      bouquet_version, created, updated
├── config_json                        JSON dump of the run BouquetConfig
│                                      (root copy = most recent write; the
│                                      per-scan copies below are authoritative)
└── scan/<scan_key>/                   one group per scan point / time slice
    ├── config_json                    this slice's exact config
    │   attrs: [coil_filter]           'chi2' | 'legacy' -- which coil filter wrote
    │          [coil_sigma_model]      passes_coil_filter last (chi2: the sigma model JSON)
    │          [boundary_rms_max_mm, boundary_max_max_mm, boundary_cut_source]
    │                                  the LCFS cut filter_boundaries applied and where it
    │                                  came from ('explicit' | 'device:<name>' | 'generic'
    │                                  | 'disabled' = cut switched off, no threshold)
    │          [n_requested, n_requested_source, generation_mode, n_attempted,
    │           n_stored, attempt_outcomes_json, bouquet_version,
    │           inspec_rms_max_mm, inspec_max_max_mm, inspec_cut_source,
    │           merge_partial_json]
    │                                  generation provenance (absent from older archives;
    │                                  readers return None): requested vs attempted
    │                                  vs stored; per-attempt outcome (stored |
    │                                  solve_failed | post_align_failed); the version that
    │                                  GENERATED the draws (read_generation_provenance());
    │                                  until-N runs only: the LCFS bound the in-spec count
    │                                  was taken against and its source (absent bound =
    │                                  none; a merge refuses shards that disagree);
    │                                  merge_partial_json marks a merge that knowingly
    │                                  left workers out (--allow-missing)
    │          [parallel_manifest_json] process-parallel runs: per-worker record
    │          [refused_reason]        a slice refused before any draw (write_refused_scan;
    │                                  run() / run_slices() write it when prepare_baseline
    │                                  raises); a later baseline/draw write moves it to
    │                                  [refused_reason_superseded]
    ├── _baseline/                     written once per scan point
    │   ├── eqdsk, [pfile]             raw byte-perfect g-file / p-file
    │   ├── psi_N, psi_N_kinetic         run grids, in the `profile_coord` coordinate
    │   ├── n_e, T_e, n_i, T_i         kinetic profiles
    │   ├── pressure[, pressure_thermal]
    │   ├── j_phi[, j_BS, j_inductive] separated toroidal currents
    │   ├── [j_NBI, j_RF, j_other, j_sawteeth]  held-fixed driven channels
    │   │                              (2026-10-09; as on each draw)
    │   ├── [j_pressure]               the pressure-driven p'G, its own bucket
    │   │                              (owner decision D2, 2026-10-09; with the
    │   │                              attr current_split_convention -- see
    │   │                              "The current split" below)
    │   ├── sigma_ne/te/ni/ti/jphi     the uncertainty envelope used
    │   ├── [aux_<name>, sigma_aux_<name>]   switchboard channels
    │   ├── [recon_lcfs_ref]           10k-pt LCFS reference (boundary metric)
    │   ├── [x_points], [coil_currents, coil_names]
    │   ├── [structured_mse/]          structured closure + mse_data only:
    │   │                              per-chord arrays as DATASETS (chord_*,
    │   │                              tgamma_meas, sigma_eff, residual_sigma_*,
    │   │                              tgamma_pred_*, jacobian, excluded_*;
    │   │                              unified engine: engine_mse_*)
    │   ├── [engine_json]              the engine record as a string DATASET
    │   │                              when too large for an attribute
    │   └── attrs: Ip_target, l_i_target, l_i_scale, source_kind, profile_coord,
    │              [diverted],
    │              [source_current_sign, source_b0_sign,
    │               source_current_sign_origin, current_frame],
    │              [coil_solve_mode]   the run's coil-solve mode ("bounded"
    │              since 2026-10-06; utils.load_coil_solve_mode, read back
    │              and checked by load_config),
    │              [engine_resolved_defaults_json]   how the engine-dependent
    │              settings (isolate_edge_jBS, perturb_jind_in_anchor) were
    │              resolved: field -> {value, origin "resolved from
    │              engine=<x>" | "explicit"[, engine_validated]} (since
    │              2026-10-07; utils.load_engine_resolved_defaults),
    │              [li_metrics_json, closure_limited]   baseline provenance (absent
    │              from older archives):
    │              Baseline.li_metrics as JSON, incl. the ip_closure health
    │              record on hybrid baselines and the report-only
    │              core_pressure_hollow record; load_baseline_profiles()
    │              decodes it to li_metrics / ip_closure / closure_limited /
    │              core_pressure_hollow (each only when present).  IMAS
    │              legacy path: source_time_match (the core_sources slice,
    │              every entry's match, driven_sources / ignored_sources /
    │              off_sources, sawteeth_hold) and swb_seed (requested /
    │              resolved / oft_jphi_fixed).  Since 2026-10-09 also:
    │              swb_conversion (how SWB's bootstrap was converted,
    │              legacy SWB baselines), zeff_provenance (how IDA's Z_eff /
    │              n_i were resolved: rung, VB/CER weights, window, clamps),
    │              zeff_dd_provenance (IMAS: how the dd's Z_eff numerator
    │              convention was decided), imas_current_conversion,
    │              ida_time_match (ida_hybrid)
    │              [current_split_convention]  where p'G sits (below)
    │              [imas_baseline="swb", swb_ip_tol, swb_ip_rel_err,
    │               swb_ip_rel_err_solve_A, swb_edge_taper_psi0,
    │               swb_jbs_convention, swb_jbs_conversion, swb_alpha*,
    │               swb_li_3_solve_A, coil_reg_target_*, swb_saw_*,
    │               swb_j_saw, swb_jphi_saw]   solve_method="swb" baselines
    │              [jbs_converged, jbs_n_passes, jbs_loop_json]
    │                                  ← the baseline's jbs_loop block (v3)
    │              [delivered_state_json]
    │                                  ← the ONE reconstruction state (loop)
    │              [engine_json]      ← the unified engine's record (added;
    │                                    reconstruction_engine="unified"),
    │                                    with its "draws" settings block
    │              [edge_pressure_json] ← the edge-pressure record (added):
    │                                    edge_pprime_pin, separatrix_pressure,
    │                                    p_sep, p_sep_applied, pax_target,
    │                                    p_scale (added 2026-10-07)
    └── <count>/                       one group per accepted draw
        │                              (integer; gaps = rejected draws)
        ├── eqdsk, [pfile]             raw bytes, fixed names
        ├── psi_N[, psi_N_kinetic]       run grids, in `profile_coord`
        ├── j_phi, j_BS, j_inductive[, j_BS,edge]
        ├── [j_pressure]               p'G, its own bucket (owner decision D2;
        │                              with current_split_convention)
        ├── [j_NBI, j_RF, j_other, j_sawteeth]  the held-fixed driven channels
        │                              (2026-10-09; engine: its own on the
        │                              archived state, j_RF = rf + other;
        │                              legacy / swb: the baseline's) -- so
        │                              j_phi = j_inductive + j_BS + j_NBI +
        │                              j_RF + j_other (+ j_pressure); j_sawteeth
        │                              is the sawteeth share OF j_other
        ├── n_e, T_e, n_i, T_i, w_ExB[, Zeff]
        ├── [pressure, pressure_thermal]
        ├── [aux_<name>]               perturbed switchboard channels
        ├── [coil_currents, coil_names]
        ├── [perturbed_lcfs_ref], [x_points]
        ├── [eq_fsa/]                  live-equilibrium flux-surface averages
        │   ├── psi_N                  always ψ_N (subgroup; see below)
        │   ├── F, avg_inv_R, avg_inv_R2, avg_B2
        │   └── q, dV_dpsi, f_trap, B_avg
        ├── [jB_parallel/]             engine draws (added 2026-10-06): the
        │   ├── psi_N                  <j.B> parts of the archived split
        │   ├── jB_inductive, jB_BS    (see "The current split of an
        │   ├── jB_NBI, jB_RF          engine draw group" below)
        │   └── kappa, j_pressure
        ├── [engine_json]              engine draw record as a string DATASET
        │                              when too large for an attribute
        └── attrs: l_i(1), l_i(3), count, profile_coord, homotopy_*, max_F_drift_pct,
                   max_VSC_drift_pct, in_spec, inspec_*, l_i_target_used, [jbs_delta_active],
                   [diverted], [passes_coil_filter, passes_boundary_filter,
                   selected]           ← filter flags, written post-hoc
                   [jbs_converged, jbs_n_passes, jbs_loop_json]
                                       ← the draw's jbs_loop block (v3)
                   [engine_json, passes_draw_band]
                                       ← engine draws only (added; see below)
                   [current_split_convention]  where p'G sits (below)
                   [kinetic_sampler_json]  legacy / swb draws: the shared
                                       kinetic sampler's record (version
                                       kinetic_sampler/2, pressure match,
                                       clip counters; engine draws carry it
                                       in engine_json's inputs)
                   [swb_jbs_convention, swb_jbs_conversion]  draws whose
                                       bootstrap came from solve_with_bootstrap
                   [swb_alpha, swb_ip_rel_err, swb_jind_resamples,
                    swb_j_saw, swb_saw_*]  solve_method="swb" draws
                   [edge_pressure_json] ← the draw's edge-pressure record
                                       (added): its own p_sep, the offset
                                       applied, beta / W_MHD in both
                                       pressure frames ("frames") and its
                                       p_scale (added 2026-10-07)
                   [boundary_rms_mm, boundary_max_mm]  ← the draw's LCFS metric,
                                       written when filter_boundaries applies a cut
```

`edge_pressure_json` is read with `bouquet.edge_pressure.load_record(header,
count=None, scan_key=None)` (`count=None`: the baseline's). `p_scale`
(added 2026-10-07; absent in records written before, `null` when not known)
is the solver's uniform `P'` rescale of that state,
`bouquet.edge_pressure.P_SCALE_DEFINITION`: TokaMaker integrates the handed
`P'` inward from the boundary from zero and rescales it uniformly so the axis
value is `pax_target`; the edge `P'` pin, the flux range the profile was built
with and (under `"legacy"`) `p_sep` move it off 1 -- about 1.02-1.03 on a
pedestal, the size of the separatrix correction. A draw's is read on its own
solved state; the baseline's is the one its reconstruction recorded (the
engine record's `edge_pressure` block, also in `delivered_state_json`; the
legacy reconstruction's `edge_pressure`). Engine draw records carry it too
(`engine_json` `delivered.p_scale`, `archived.p_scale`). See
[physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin).
Under `separatrix_pressure="offset"` (the default since 2026-10-02; archives
written before then were made with `"legacy"`)
the stored `eqdsk` bytes carry the full pressure (`PRES` = the solver's
pressure + that equilibrium's `p_sep`).

## Conventions

- **Bare dataset names, units in attrs.** Profile datasets carry plain names
  (`j_phi`, `n_e`, …) with the unit string in `ds.attrs["units"]`
  (`PROFILE_UNITS` in `schema.py`). v1 archives embedded units in the name
  (`"j_phi [A m^-2]"`).
- **Fixed byte-blob names.** The g-file / p-file bytes are stored as `eqdsk` /
  `pfile` inside each group — the group path carries the coordinates. Bytes
  are stored opaque (`np.void`) and round-trip bit-perfect.
- **Always `scan/<key>/`.** The scan key is a user-chosen label
  (`GenerationConfig.scan_key`, default `0`) — a time in ms, a beta value, …
  Several bouquets can share one file under different keys.
- **Profile coordinate.** `profile_coord` (`"psi_n"` or `"phi_n"`, on
  `_baseline` and each draw; absent = `"psi_n"`) names the coordinate of the
  `psi_N` / `psi_N_kinetic` grids despite their names. `eq_fsa/psi_N` is
  always ψ_N. Read it with `bouquet.utils.profile_coord`; `merge_archives`
  refuses shards that differ.
- **Gap-tolerant indices.** Rejected draws leave gaps; iterate with
  `list_equilibrium_indices` / `BouquetArchive`, never `range(n)`.
- **Filtering is non-destructive.** Filters write boolean attrs
  (`passes_*`, `selected` = AND of applied flags); `export_filtered` produces
  a pruned copy, the source is never modified.
- **Provenance.** `schema_version` / `bouquet_version` / `created` are stamped
  at file creation; `config_json` is added by `write_provenance` (called from
  `Bouquet.generate`, `run_shard`, and `merge_archives`). Recover the exact
  run configuration with `bq.load_config(path, scan_key=...)`.
- **Current orientation.** Every archived current and eqdsk is in bouquet's
  positive-current frame (TokaMaker native: `Ip > 0`, `F0 > 0`). IMAS-path
  archives record the source's own orientation on `_baseline`:
  `source_current_sign` (the factor the reader multiplied every source current
  by; `-1.0` for a reversed-current source), `source_current_sign_origin`
  (whether that factor was `sign(ip)` or set by
  `ImasSource.current_orientation`), `source_b0_sign` (the source's
  vacuum-field sign; absent when `b0` is zero or unreadable) and
  `current_frame` (a plain statement of the frame).
  Absent on g-file-path archives and on IMAS archives written before the
  reader's normalisation. See
  [physics-notes](physics-notes.md#current-and-field-orientation).
- **Structured-closure MSE record (`_baseline/structured_mse/`).** Written
  only when `closure_channel="structured"` ran with `mse_data`
  (`Baseline.mse_record`). The per-chord arrays and the `n_chords × 2K`
  Jacobian grow with the chord count, so they are datasets here and never
  part of the closure record's single JSON attribute (HDF5 caps an attribute
  at 64 kB); the closure record carries only chord-count-independent
  summaries and points here. `load_baseline_profiles` returns the subgroup as
  a dict under `"structured_mse"`.
- **Live-equilibrium FSA (`eq_fsa/`).** Optional per-draw subgroup of
  flux-surface averages captured directly from the live TokaMaker object at
  generate time (`GenerationConfig.capture_live_eq`, on by default), on a
  ψ_N grid of `capture_npsi` points (ψ_N whatever the `profile_coord`). Keys and units are `EQ_FSA_GROUP` /
  `EQ_FSA_UNITS` in `schema.py`: `F` (T m), `avg_R` (⟨R⟩, m), `avg_inv_R`
  (⟨1/R⟩, m⁻¹), `avg_inv_R2` (⟨1/R²⟩, m⁻²), `avg_B2` (⟨B²⟩, T²), `pprime`
  (p′, Pa Wb⁻¹, signed so `jphi_eq` > 0), `jphi_eq` (the equilibrium's own
  TokaMaker jphi, A m⁻²), `q`, `dV_dpsi` (m³ Wb⁻¹), `f_trap`, `B_avg`
  (⟨B⟩, T); `avg_R`/`pprime`/`jphi_eq` are absent from older archives.
  `⟨1/R²⟩` comes from `get_q` when the toolkit exposes it, else exact FSA
  quadrature (`capture_exact_inv_R2`, default). This is what enables the exact
  TokaMaker-jphi → IMAS current conversion in the IMAS/OMAS exporter
  (`write_imas_draw(..., fidelity="exact")`); read it back with
  `bq.load_eq_fsa`.

- **Self-consistent bootstrap record (`jbs_loop` block, schema v3).** When
  the bootstrap came from the self-consistent loop
  (`GenerationConfig.jbs_self_consistent`, **the default**) every draw group
  carries `jbs_converged` (bool), `jbs_n_passes` (int, all loops of the draw)
  and the full loop record as JSON in `jbs_loop_json` (residual histories,
  relaxation factors, the solved-vs-closure gap and the record-only
  unrelaxed closure residual `current_residual_unrelaxed`, tolerances, the
  post-homotopy check, the evaluator version, the OFT build as a path-free
  identifier `oft_build = {version, git_hash, library_sha256,
  sources_sha256, build_id}` -- `git_hash` is `None` outside a git checkout
  (every installed build), so the SHA-256 of the loaded `liboftpy` and the
  digest of the package's Python sources identify the build (added
  2026-10-06; earlier records carry `{version, git_hash, build_id}`, and
  archives written by earlier builds of this branch carry `oft_build.path`,
  the install location, instead), and `init_source` -- what each loop started from,
  per loop under `loops` and for the draw's first loop at the top level); the `_baseline` group carries the same three
  attrs for the baseline's own loop (`jbs_n_passes` = its pass count). Names
  in `schema.JBS_LOOP_ATTRS`; write/read with `schema.write_jbs_loop` /
  `schema.read_jbs_loop`, or read with
  `bouquet.utils.load_jbs_loop(header, count, scan_key)` (`count="_baseline"`
  for the baseline), `DrawView.jbs_loop` / `DrawView.jbs_converged`,
  `ScanView.baseline_jbs_loop` and `ScanView.bootstrap_model`. A group
  **without** the block carries a frozen (`solve_with_bootstrap`) bootstrap.

- **The one reconstruction state (`_baseline@delivered_state_json`, loop
  only).** The design rule: the input (g-file or modelling-source IDS), the
  bouquet reconstruction (as close to the input as it can be while physically
  valid and carrying a Redl bootstrap -- allowed to differ from the input),
  and the draws (perturbations of the reconstruction; at zero perturbation
  they reproduce it). With the loop on, the reconstruction is ONE
  equilibrium, and `_baseline` records it and says whether the run's
  baseline re-solve -- the saved `eqdsk` above and every draw's warm start --
  is it. JSON keys (`utils.DELIVERED_STATE_ATTR`, written by
  `utils.store_baseline_state`): `convention` (what the in-memory split is,
  below), `path`, `l_i` (= `l_i_target`, l_i(3)/'iter'), `q0`, `q95`
  (`get_stats` on the delivered state), `Ip_target`,
  `request_normalisation` / `achieved_normalisation` (the uniform factors
  that put the stored request / the achieved current at `Ip_target` in the
  'exact' FSA current measure), `n_floored_inductive`,
  `n_floored_target_inductive` (points where a zero-perturbation draw cannot
  reproduce the state), on the g-file path `li_corrective_state`,
  `li_step6_matched` and `li_input`, `how`, then `l_i_target`,
  `baseline_resolve` (`l_i`, `q0`, `q95`, `Ip` of the run's baseline re-solve
  and their differences from the recorded values) and
  `archived_j_phi_rel_l2_vs_delivered` (the archived `j_phi` against the
  delivered state's achieved current). Absent with
  `jbs_self_consistent=False` (legacy archives are unchanged bit for bit).
  With the loop on, `_baseline/j_phi` is the delivered state's ACHIEVED FSA
  current (as before: `store_achieved_jphi`), `j_BS` the draws' σ=0
  bootstrap composition on it (+ `jBS_diff`) and `j_inductive` their
  residual; the in-memory `Baseline` split the draws consume is the
  Ip-normalised jphi-linterp REQUEST of the same state (one solve of it is
  the state), with `Baseline.jphi_request_offset` = request − achieved
  (not archived).

- **The unified engine's records (added fields; schema stays v3).** Written
  only with `reconstruction_engine="unified"`; a legacy archive carries none
  of them and reads exactly as before.
  - `engine_json` (on `_baseline` and on every draw group): the JSON record,
    as the attribute when it fits in 60 000 bytes
    (`engine.ENGINE_ATTR_MAX_BYTES`), else a string DATASET of the same name
    with the attribute holding the pointer `{"stored_as": "dataset"}` (HDF5
    caps a group's object header at 64 KiB). Read with
    `bouquet.engine.read_engine_json(group)`,
    `engine.load_baseline_engine(header, scan_key)` or
    `engine_draws.read_draw_engine(header, count, scan_key)`.
  - The baseline record is the reconstruction (contract, settings,
    convergence constants and their origins, per-pass log, delivery checks,
    the state, solves, and `coil_solve_mode`: `"bounded"` -- the solver's
    coil least-squares mode, entered once at `Bouquet.setup_solver`, see
    `bouquet.solver_state`; absent in records written before 2026-10-06),
    `coil_reg` (the coil regularisation the reconstruction solved under:
    `source` -- `"configured"`, `"default"`, `"strong (left installed by
    generate())"` or `"unknown"` -- and its `terms` as `coils` / `target` /
    `weight`; every engine draw's loop installs exactly that list; absent
    before 2026-10-06) plus, once `generate()` ran, a `draws` block (the
    draws' loop settings, `rng_stream`, `q0_row`, `homotopy`,
    `l_i_tolerance`, the Ip-row target `Ip_target_A` in the exact measure,
    and the `reference` values every draw is compared with). Its per-chord
    MSE arrays and Jacobians are NOT in the JSON: they are datasets
    `engine_mse_*` under `_baseline/structured_mse/`, and the JSON holds
    `"mse_record[<key>]"` in their place.
  - The per-draw record (`engine_draws`, version `unified-engine-draw/1`):
    `identity` (whether the first request was the stored one, bit for bit),
    `coil_reg` (the coil regularisation the draw's loop solved under:
    `source` -- `"reconstruction (configured)"` / `"reconstruction
    (default)"`, the reconstruction's own term list, recorded on the
    baseline's engine record as `coil_reg` with its terms; or a historical
    fallback named as such -- `n_terms`, `installed`; absent before
    2026-10-06),
    `inputs` (bootstrap scale, pressure-match iterations, inductive tries),
    `rng_stream`, `amplitude` (per pass: `a_ind`, `a_bs` with the q0 row,
    the Ip increment and the Ip in the exact measure against its target),
    `loop` (the kernel record), `passes`, `q0_row`, `delivered` (the loop's
    delivered draw: `l_i_3`, `l_i_1`, `beta_n`, `q0` at `q0_psi_N`,
    `q0_stats` at `q0_stats_psi_N`, `q95`, `Ip`, the delivery check,
    request − achieved, `flux_range`), `archived` (the same after the
    homotopy stage, plus `split`: see the next item),
    `reference` (the reconstruction's DELIVERED measurement -- every value,
    its `flux_range` and the q-row radius included), `deltas`
    (against the reconstruction: `l_i_3`, `l_i_1`, `beta_n`, `q0`, `q95`,
    and the poloidal flux range `psi_b - psi_a` -- `flux_range` [Wb/rad]
    and `flux_range_rel`; `archived.deltas` the same for the archived
    state; `delivered.flux_range` / `archived.flux_range` the values),
    `post_hoc`
    (the l_i band and `constrain_sawteeth` verdicts, `coil_in_spec`,
    `in_spec`), `homotopy`, `post_homotopy`, and `cost` (solves, passes and
    wall time for the stages `anchor`, `loop`, `homotopy`, `post_homotopy`,
    `filters`, `archive`, and their `total`).
  - **The current split of an engine draw group** (`j_phi`, `j_BS`,
    `j_inductive`) is evaluated on the draw's own ARCHIVED (final,
    post-homotopy) equilibrium: `j_phi` its achieved FSA current; `j_BS`
    the draw's bootstrap model on that state, `s_bs (1 + d_bs) x scale x
    Redl(final) x F<1/R>/<B^2>(final)`; the fixed beam / RF parts the
    contract's `<j.B>` times the same final-state factor (recorded in
    `archived.split.j_NBI` / `j_RF`); `j_inductive` the residual `j_phi -
    j_BS - j_NBI - j_RF - j_pressure` (since 2026-10-09 the pressure-driven
    term is the separate `j_pressure`, owner decision D2) and NEVER
    clipped -- a negative value is recorded in `archived.split`
    (`n_negative_inductive`, `min_inductive`, `negative_inductive_psi_N`)
    and printed, never altered or filtered. (Legacy draws keep their split:
    the residual floored at zero with the sliver moved into `j_BS`.)
  - **`jB_parallel/`** (engine draws, added 2026-10-06; keys and units
    `JB_PARALLEL_GROUP` / `JB_PARALLEL_UNITS` in `schema.py`, read with
    `bouquet.schema.read_jB_parallel(group)`): the PARALLEL parts the
    toroidal split above was converted from, raw `<j.B>` [T A m⁻²] in the
    positive frame on the subgroup's `psi_N` (the draw's grid) --
    `jB_BS` (the bootstrap model on the archived state), `jB_NBI`,
    `jB_RF` (rf + other driven), and `jB_inductive` the FIELD-ALIGNED
    inductive only; with `kappa =
    F<1/R>/<B^2>` and `j_pressure = p'(<R> - F^2<1/R>/<B^2>)` [A m⁻²] of
    the archived state, `j_phi = kappa (jB_inductive + jB_BS + jB_NBI +
    jB_RF) + j_pressure` to round-off. Neither the toroidal `j_BS` nor
    any parallel part carries `j_pressure` (it is the group's own
    `j_pressure` dataset since 2026-10-09). The IDS
    exporter (`write_imas_draw`) writes these parts as they are, so no
    exported parallel current (`j_ohmic`, `j_bootstrap`, `j_total`)
    carries the pressure-driven term and export -> `IdsAdapter.read`
    returns the archived `<j.B>` parts and `<j_phi>`. For draws without
    the subgroup (every legacy draw; engine draws archived before
    2026-10-06) the exporter subtracts `j_pressure` computed from the
    archived eqdsk's own flux surfaces (`io.imas.archived_pressure_term`,
    COCOS 7) from `j_BS` before converting with the `eq_fsa` geometry.
  - `passes_draw_band` (bool attr, engine draws only): the post-hoc band
    verdict. It is one of the filter flags ANDed into `selected`
    (`filtering._FILTER_FLAGS`), so `.filter()` selects what the until-N
    ledger counted; `in_spec` is the coil verdict AND this band.
  - **Fields whose MEANING differs on an engine archive** (names and units
    unchanged; a reader that predates the engine reads them without error
    but must not assume the legacy meaning): `j_inductive` / `j_BS` -- the
    engine's split is NOT floored (`j_inductive` is the residual and may be
    negative; legacy draws floor it at zero and move the sliver into
    `j_BS`); `in_spec` -- on an engine draw
    the coil verdict AND the post-hoc band (`passes_draw_band`), on a legacy
    draw the coil verdict alone. (An older package that RE-FILTERS an engine
    archive would not AND `passes_draw_band` into `selected`; `in_spec`
    still carries the band.) Independently of the engine, under
    `separatrix_pressure="offset"` (the default since 2026-10-02) the stored
    `eqdsk` bytes carry `PRES` = the solver's pressure + `p_sep` (it was
    zero at the boundary before).

## The current split: where the pressure-driven `p′G` sits (2026-10-09)

The toroidal split of every group satisfies `j_phi = j_inductive + j_BS +
j_NBI + j_RF [+ j_other] (+ j_pressure)`, where `j_pressure = p′(⟨R⟩ −
F²⟨1/R⟩/⟨B²⟩)` (A7 of docs/current-conventions.md) is the pressure-driven
current, whose `⟨j·B⟩` is zero. The group attr `current_split_convention`
(`schema.CURRENT_SPLIT_CONVENTION_ATTR`, read with
`schema.read_current_split_convention(group, baseline_attrs)`) names where it
is:

- `"pressure_separate"` (owner decision D2; every path since 2026-10-09):
  the `j_pressure` dataset; neither `j_BS` nor `j_inductive` carries it.
  Engine draws and baselines take it from their own composition; swb from
  the SWB split; legacy draws evaluate it on the archived state
  (`TokaMaker_interface.archived_pressure_term`) and the archive writer
  takes it off the legacy in-memory `j_inductive`.
- `"pressure_in_inductive"`: the residual `j_inductive` carries it -- every
  archive before PR #64, and any group without the attr or a `j_pressure`
  dataset (also a legacy group whose state could not be evaluated, warned).
- `"pressure_in_bootstrap"`: `j_BS` carries it -- PR #64's evaluator
  (`evaluate_jBS/3`, never on main), inferred from a `/3` loop record.

The IDS exporter takes `p′G` off whichever bucket carries it before it
converts to parallel currents (and groups it with the non-inductive
currents, never `j_ohmic`).

## v2 → v3: the self-consistent bootstrap record

Schema v3 is **additive in its names**: it adds the `jbs_loop` block above
and renames or removes no v2 dataset or attr. It is NOT free of changes of
meaning on the default path of this release: the engine's unfloored
`j_inductive` / `j_BS` split, `in_spec` = coil AND band on engine draws, and
`PRES` + `p_sep` in the stored g-files under `separatrix_pressure="offset"`
(see "The unified engine's records" above). The schema version does not
encode these; the engine record (`engine_json`) and the edge-pressure record
(`edge_pressure_json`) say which applies.

- **No migration.** A v2 archive reads as a v3 archive whose bootstrap is
  frozen everywhere (no `jbs_loop` block); every v3 reader accepts it
  unchanged, and every v2 reader ignores the new attrs. Do not gate readers on
  `schema_version == 2`.
- **Which bootstrap is in an archive** is decided by the block, never by the
  version number: a v3 archive written with `jbs_self_consistent=False`
  carries no block either (it is the frozen path, bit for bit: the one solver
  change the loop needed, the soft closure's noise-floor acceptance, is
  opt-in via `close_ip_structured_soft(accept_noise_floor=True)` and passed by
  the loop's closure calls only), and appending
  to a v2 file with a current bouquet restamps `schema_version` to 3 while its
  old draws keep reading as frozen. `ScanView.bootstrap_model` gives the label
  ("self-consistent Redl bootstrap" / "frozen SWB bootstrap (legacy)").
- **Replaying an old archive's config.** A `config_json` written before the
  loop existed has no `jbs_self_consistent` field; `load_config` /
  `BouquetConfig.from_dict` rebuild it with `jbs_self_consistent=False` (and
  warn), the behaviour it was produced with.
- **Comparisons across the change.** The self-consistent bootstrap moves the
  bootstrap/inductive split (and with it l_i, q0 and the per-draw responses);
  compare a v2/frozen archive with a v3 loop archive as two bootstrap models,
  not as a regression.

## Legacy (pre-v2) archives

Schema v2 was a clean break (2026-07). Files without the `schema_version`
attr are pre-v2: `BouquetArchive` opens them with a warning (byte blobs still
resolve via a suffix scan; profile keys keep their v1 bracketed names), and
`load_equilibrium` raises a clear error. Regenerate old archives with the
current package for full support.
