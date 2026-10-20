# API reference

Everything exported from the top-level `bouquet` namespace, grouped by role.
Docstrings in the source are authoritative; this page is a map.

## Class API (recommended)

| Class / Method | Description |
|------------------|-------------|
| `Bouquet` | Stateful driver: `setup_solver` → `prepare`/`reconstruct` → `generate` → `filter` → `export` (or just `.run()`) |
| `Bouquet.from_geqdsk()` / `Bouquet.from_imas()` | Minimal constructors for the two baseline sources; auto-apply the validated workflow preset for each path |
| `Bouquet.describe()` | Print the configuration, non-default knobs only |
| `Bouquet.verify_sigma0_consistency()` | σ=0 regression guard on the draw-pipeline bootstrap split (one solve) |
| `Bouquet.run_slices()` | Multi-slice IMAS sweep into one archive, one `scan_key` per slice; a refused slice is recorded as such (reason + time) and the sweep continues (`on_refusal="record"`, the default since 2026-10-06; `"raise"` stops at the first) |
| `Bouquet.save_baseline_eqdsk(path)` | Write the reconstruction's own equilibrium (the live state `prepare_baseline()` left; refused after a later solve) as a g-file in the same pressure frame as the archive's: `PRES` carries the baseline's `p_sep` under `separatrix_pressure="offset"`. A bare `mygs.save_eqdsk` writes the solver frame (`PRES` zero at the boundary) |
| `Bouquet.export_bundle()` / `Bouquet.export_ids()` | Per-draw file bundle (geqdsk / pfile / profiles JSON), or one IMAS/OMAS IDS per draw (`fidelity="exact"` uses the captured `eq_fsa` geometry) |
| `Bouquet.selected_indices()` / `Bouquet.output_spread()` | Post-generation introspection |
| `Bouquet.plot_baseline()` / `.plot_bouquet()` / `.plot_traces()` / `.plot_coil_currents()` / `.plot_spec_summary()` | Bound plotting |
| `Bouquet.archive` | The run's `BouquetArchive` |

## Configuration and sources

| Class / Function | Description |
|---|---|
| `BouquetConfig` | Top-level typed config; JSON-serializable via `to_dict` / `from_dict` |
| `SolverConfig`, `UncertaintyConfig`, `GenerationConfig`, `FilterConfig`, `FixedComponentsConfig` | The sub-objects — see [workflows.md](workflows.md#configuration-reference) |
| `ReconstructionSource` / `ImasSource` | Baseline source definitions (g-file + profiles, or IMAS/OMAS JSON) |
| `Baseline` / `resolve_baseline()` | The common separated-current product every source resolves to |
| `resolve_uncertainty()` | Resolve sigma envelopes + GPR length scales from the config |

## Archive

| Class / Function | Description |
|------------------|-------------|
| `BouquetArchive` | High-level reader: `ar[scan_key]` → `ScanView` → `DrawView` (profiles, flags, parsed equilibria, extraction) |
| `ScanView` / `DrawView` / `BaselineView` | The view objects those return (`ScanView.baseline_view()` exposes the baseline through the draw accessors) |
| `draw_band()` | Standard across-draw band (median, p16/p84, min/max) for a quantity computed per draw by the caller's evaluator; counts, dropped reasons, floors, pole gate, baseline overlay and provenance — see [workflows.md](workflows.md#error-bars-from-an-archive) |
| `draw_bands()` | `draw_band` over a requested list of `(archive, scan_key)`; absent keys become `no_archive` records, refused slices `refused`, keys where no draw reaches the statistic (e.g. every draw rejected) `empty` records with an `empty_reason` -- never a silent gap |
| `draw_scalars()` | `draw_band` for bouquet's own scalars: q0, q95, rho(q=m/n), l_i (archive estimator), beta_N, <P>, beta_p, Lambda = beta_p + l_i(3)/2 (reported, never a filter) |
| `bands_summary()` | The standard error bars of one slice in one call: per quantity the median, p16–p84, full range and n over the in-spec draws, for scalars (`beta_N`, `l_i_3`, `Lambda`, ...) and gradients (`dp_dpsi`, `djphi_dpsi`: d/dψ_N of each draw's own pressure / j_φ at its OWN q = 2 surface, draws without one counted and excluded, a named ψ_N fallback stated in the record); returns a `BandSummary` (`.table()`, `.plot()`) — see [workflows.md](workflows.md#error-bars-from-an-archive) |
| `BandSummary` / `plot_bands_summary()` | Its result (records + where the surface was found) and its one band figure (n stated) |
| `BandRecord` / `BandTable` | The per-(key, quantity) record and its container (`to_dataframe()`, `to_csv()`) |
| `plot_band()` | Median + p16–p84 band across keys; hollow markers below the floor, regular fraction where gated, baseline dashed |
| `write_provenance()` / `load_config()` | Stamp / recover the exact `BouquetConfig` stored in an archive |
| `write_refused_scan()` | Record a slice refused before any draw (`refused_reason` on an empty `scan/<key>`); `run()` / `run_slices()` call it when `prepare_baseline` raises |
| `initialize_equilibrium_database()` | Create/open an archive (stamps `schema_version`) |
| `load_equilibrium()` / `load_equilibrium_by_path()` | Read one draw |
| `load_baseline_profiles()` | Read the per-scan baseline |
| `load_eq_fsa()` | Read the per-draw live-equilibrium flux-surface-average block |
| `discover_scan_keys()` / `count_equilibria()` / `list_equilibrium_indices()` | Archive introspection |
| `select_indices()` / `read_filter_flags()` | Filter-flag queries |
| `filter_coil_currents()` / `filter_boundaries()` / `export_filtered()` | Post-generation selection of the machine-realizable subset |
| `refilter_inspec()` / `InspecBands` | Re-judge an archive's opt-in in-spec bands (l_i, beta_p, q0 gate) from its stored numbers; default `bands=None` = the magnetics rule (owner decision 2026-10-14; see [workflows.md](workflows.md#the-in-spec-rule)) |
| `bouquet.filtering.passes_all_filters()` (+ `passes_coil_spec` / `passes_coil_chi2` / `passes_boundary_spec` / `boundary_deviation_mm`) | The selection predicate itself, on raw numbers rather than an archive. Shared with `generate_bouquet`'s [until-N](workflows.md#until-n-in-spec-draws) stopping rule so the two agree by construction. Module-level, not re-exported at the package root |
| `bouquet.filtering.make_coil_predicate()` (+ `coil_spec.resolve_coil_acceptance()`) | Builds the **configured** coil verdict (chi2 or legacy, with the loud `CoilSigmaUnavailable` fallback) once per run from the same `FilterConfig` fields `Bouquet.filter()` cuts on — the reason the until-N count and the postprocess selection are the same set |

Layout spec: [archive-schema.md](archive-schema.md). Code source of truth:
[`bouquet/schema.py`](../bouquet/schema.py).

## Added or changed in 1.4.0

The fields, functions and records 1.4.0 added or whose default it changed.
Dates are the owner decisions recorded in
[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) (1.4.0 entry); the linked page
explains each one. Defaults are read from `bouquet/config.py`.

### Configuration fields

| Field | Default | What it does |
|---|---|---|
| `SolverConfig.isoflux_weight_total` | `None` (= `solver_geometry.ISOFLUX_WEIGHT_TOTAL`, 2e5) | The TOTAL weight of the unified engine's standard boundary (isoflux) set, spread per unit length; a number is an explicit override, installed as given and stamped as mode `"override"`. Must be `None` or a finite number > 0; ignored (with a `UserWarning`) when `isoflux_weights` is given or on the legacy route. Decided 2026-10-15 (per unit length, #84) and 2026-10-16 (the constant total). See [engine.md](engine.md#the-boundary-constraint-a-constant-total-inside-a-measured-window) |
| `SolverConfig.boundary_partition_probe` | `True` | After the unified reconstruction, one capped re-solve with the pressure x 1.02 measures how a perturbation splits between the boundary cut and the coil cut at the installed total; stamped (`Baseline.engine["boundary_balance"]`, the solver-geometry record's `isoflux_balance["partition"]`) and warned outside [1/3, 3]. Report only: it never changes the total. `False` stamps `"off"`. See [engine.md](engine.md#the-boundary-constraint-a-constant-total-inside-a-measured-window) |
| `GenerationConfig.check_bistability` | `None` (resolved: `True` under the unified engine, `False` under legacy) | One extra cold-start solve after the reconstruction, compared at the σ=0 gates (l_i(3), q0); a miss stamps `bistable=True` and warns, nothing is refused. On by default since 2026-10-15; an explicit `True` is refused on the legacy route. See [workflows.md](workflows.md#generationconfig-bgeneration) |
| `GenerationConfig.draw_wall_budget_s` | `600.0` | Per-draw wall-clock budget [s] on both routes, checked between solves (a running GS solve is never interrupted); an over-budget draw is rejected as `draw_wall_budget`. `None` switches it off; the σ=0 verification runs without it. Wall-clock, so host-dependent. Decided 2026-10-16. See [engine.md](engine.md#cost) |
| `GenerationConfig.l_i_tolerance` | `None` (was `0.05` until 2026-10-14) | An explicit fractional l_i(3) band around the reconstruction's l_i; unset, no l_i band (the in-spec rule is the magnetics, 2026-10-14). The legacy routes still construct their draws with 0.05 when unset. A stored config at the old default 0.05 loads with the band off and a one-line note. See [workflows.md](workflows.md#the-in-spec-rule) |
| `GenerationConfig.l_i_sigma` | `None` | The stated 1σ of l_i(3) (absolute) of the source's MSE / diamagnetic fit; band `\|l_i - l_i*\| <= inspec_band_nsigma * l_i_sigma`. Exclusive with `l_i_tolerance`; unified engine only (refused on the legacy routes at `generate()`). See [workflows.md](workflows.md#the-in-spec-rule) |
| `GenerationConfig.beta_p_target` / `beta_p_sigma` | `None` / `None` | A diamagnetic β_p and its 1σ (both or neither); band `\|beta_p - beta_p_target\| <= inspec_band_nsigma * beta_p_sigma`. See [workflows.md](workflows.md#the-in-spec-rule) |
| `GenerationConfig.inspec_band_nsigma` | `2.0` | Half-width of a σ band (`l_i_sigma`, `beta_p_sigma`) in σ. A stored config without this key predates the 2026-10-14 rule (the stored-config compatibility above keys on it). See [workflows.md](workflows.md#the-in-spec-rule) |
| `UncertaintyConfig.kinetic_clips` | `None` (resolved by `experimental.kinetic_clips_on`: ON on every route) | The kinetic sampler's clips: drawn Z_eff floored at 1, a derived n_i held in [0, n_e − z_fast], an independent n_i capped there when Z_imp is declared, a passive Z_eff aux draw clipped to the window; every clip that fires is counted per draw and archived. On every route since 2026-10-17 (graduated 2026-10-14 on the measured route). `False`: only the `zeff_bounds` window, and a draw with Z_eff < 1 is refused (`kinetics_nonphysical`). See [workflows.md](workflows.md#uncertaintyconfig-buncertainty) |
| `UncertaintyConfig.ne_scalar_sigma` / `te_` / `ni_` / `ti_` / `zeff_scalar_sigma` | `None` | The assumed σ of a channel with no measured σ, as a fraction of \|baseline\|; `None` takes the H-mode flat-top radial template (`sigma_template`), 2026-10-14. See [workflows.md](workflows.md#uncertaintyconfig-buncertainty) |
| `ImasSource.ni_source` | `"auto"` | The Z_eff / n_i route on `ida_hybrid`: `"auto"` resolves to the measured IDA route (`"all"`) when an IDA file supplies the kinetics, else `"standard"` (the dd's Z_eff). An IDA file with no Z_eff information is refused on `ida_hybrid` unless `zeff_from_fuse=True`. Decided 2026-10-14. See [workflows.md](workflows.md#ion-and-z_eff-fields-of-the-sources-bsource-pr-56) |
| `ImasSource.zeff_from_fuse` | `False` | On a measured `ida_hybrid` route, `True` keeps the dd's TOTAL Z_eff (its validated `zeff` dataset, else the all-ion sum; 2026-10-18) for the bootstrap while n_e/T_e/T_i/n_i come from IDA; `False` takes IDA's resolved Z_eff. Under `ni_source="standard"` the Z_eff is the dd's either way. See [workflows.md](workflows.md#ion-and-z_eff-fields-of-the-sources-bsource-pr-56) |
| `ImasSource.ni_subtract_fast` | `None` (follows `include_pfast` on the measured route, `False` elsewhere) | Subtract the dd's fast-ion density equivalent from the IDA (total) n_i on a measured `ida_hybrid` route; `True` with `ni_source="standard"` is refused. Graduated 2026-10 (the #79 rule: fast ions in `p_fast` with the subtraction, or both off). See [workflows.md](workflows.md#ion-and-z_eff-fields-of-the-sources-bsource-pr-56) |
| `FixedComponentsConfig.p_fast` | `None` → `ImasSource`: the dd's reduced fast pressure; `ReconstructionSource`: the p-file's `pb` block (x 1e3, Pa), else zero (an IDA file) | The p-file `pb` default is decision B, 2026-10-17; an explicit array wins (a p-file `pb` beside it is warned about once). The source is stamped `p_fast_source`, beside the report-only `pressure_fidelity`. See [workflows.md](workflows.md#fixedcomponentsconfig-bfixed_components) |

`reconstruction_engine="legacy"` / `solve_method="legacy"` emit one
`DeprecationWarning` per run (text `config.LEGACY_ROUTE_DEPRECATION`); the
route is removed in 1.5 (#90). See [engine.md](engine.md).

### Functions and classes

| Name | What it does |
|---|---|
| `bands_summary(archive, scan_key=None, *, scalars=("beta_N", "l_i_3", "Lambda"), gradients=("dp_dpsi", "djphi_dpsi"), at=None, fallback_psi_N=0.5, **kw) -> BandSummary` | The standard error bars of one slice over the in-spec draws (median, p16/p84, min/max, n), built on `draw_band` (`**kw` goes to it: `selection`, `require_filter`, `min_n`, `hard_min`, `percentiles`, ...). Scalars from `stats.SUMMARY_SCALARS` (`beta_N`, `beta_p`, `Lambda`, `l_i_3`, `l_i_1`, `q0`, `q95`); gradients from `stats.SUMMARY_GRADIENTS`, d/dψ_N of each draw's archived `pressure` / `j_phi` at `at` (`None` = `{"q": 2.0}`, each draw's own outermost crossing; or `{"psi_N": x}`). Draws without the surface are counted and excluded; when no in-spec draw has one, every draw is evaluated at `fallback_psi_N` and the record says so (`None` disables the fallback). See [workflows.md](workflows.md#error-bars-from-an-archive) |
| `BandSummary` | Fields: `records` (`{quantity: BandRecord}`), `surface` (`mode` = `"own_surface"` / `"fallback_psi_N"` / `"fixed_psi_N"`, the rule, located and no-surface draws, ψ_N and q source per draw, the baseline's evaluation point), `scan_key`, `archive`, `n_selected`, `units`. Properties `population`, `where`; methods `rows()`, `table()`, `plot()`; indexable by quantity |
| `plot_bands_summary(summary, ax=None)` | One band figure: per quantity the deviation from its median in % (p16–p84 bar, min–max line, median dot, hollow below the floor, baseline cross); n in the row labels and the title |
| `refilter_inspec(h5path_or_header, bands=None, scan_key=None, *, verbose=True)` | Re-judge an archive's opt-in bands from the stored per-draw numbers (no physics re-run): band flags on are written, flags off removed, `passes_draw_band` and `selected` recomputed with the coil and boundary flags already there, the scan stamped with the rule. `bands` is an `InspecBands`, a `GenerationConfig`, or `None` (the magnetics alone). The in-loop until-N stamps are not rewritten. Returns `{scan_key: {"rule", "n_draws", "n_in_band", "n_selected", "draws"}}`. See [workflows.md](workflows.md#the-in-spec-rule) |
| `InspecBands` | The opt-in bands (`l_i_tolerance`, `l_i_sigma`, `beta_p_target`, `beta_p_sigma`, `nsigma`, `constrain_sawteeth`), built by `filtering.resolve_inspec_bands` from a `GenerationConfig`; every band off by default; a missing or non-finite banded quantity fails that band |
| `sigma_template()` | Relative 1σ radial templates (the assumed default envelope, 2026-10-14); listed above under Sampling |
| `Bouquet.draw_rejections` | After `generate()`: one dict per REJECTED attempt (`draw`, `reason` from `TokaMaker_interface.DRAW_REJECTION_REASONS`, `stage`, `error_type`, `message`; `masked_reason`, `elapsed_s` and `wall_budget_s` when the wall budget ran out first; `info` for a coil saturation). Never archived, never counted toward until-N; summarised in the generate summary line |
| `Bouquet.engine_draw_cap_events` | After an engine `generate()`: every draw solve that stopped at `engine_draw_solve_maxits` (`draw`, `where`, `stage`, `maxits`, `iterations`, `seconds`, `outcome` = rolled back / rejected, `error`); a homotopy stage also carries `vertical_runaway` (the record of `engine.vertical_runaway_record`) and `label`. See [engine.md](engine.md#l_i-controllability) |
| `Bouquet.engine_axis_shift` | After an engine `generate()`: the archived draws' `axis_shift_large` count and the scan's `axis_bimodality` record (`engine_draws.axis_shift_counts`) |

### Stamps and rejection codes

| Name | Where | Meaning |
|---|---|---|
| `axis_shift_large` | each engine draw's axis record (`engine.axis_shift_record`) | `True` when the draw's magnetic axis sits more than `AXIS_BRANCH_FRAC` = 0.01 x a from the reconstruction's delivered axis; a displacement, not a branch label (2026-10-15). Replaces `axis_branch`, which old archives still answer as `axis_shift_large`. Reported, never a filter |
| `axis_bimodal` | the scan's `axis_bimodality` record (`engine.axis_bimodality`) | Largest-gap two-cluster test of the draws' axis Z; a verdict only with at least `AXIS_BIMODAL_MIN_N` = 24 draws with an axis, else `None` (2026-10-15). A bimodal scan sets `bistability_source="draws"`. Reported, never a filter. See [workflows.md](workflows.md#the-in-spec-rule) |
| `draw_wall_budget` | rejection reason | The draw exceeded `draw_wall_budget_s` (checked before each solve and before archiving); the rejection it pre-empted is kept as `masked_reason` |
| `vertical_runaway` | label on a capped homotopy stage's cap event | The vertical-control channel at its homotopy `#VSC` bound AND the axis displaced vertically by more than `VERTICAL_RUNAWAY_AXIS_FRAC` = 0.1 x a (2026-10-16). A label, never a gate. See [engine.md](engine.md#the-boundary-constraint-a-constant-total-inside-a-measured-window) |

Where each is archived: [archive-schema.md](archive-schema.md#records-added-in-140).

## Parallel generation

| Function | Description |
|---|---|
| `parallel_generate()` | Process-parallel driver (`backend="laptop"` pool or `"slurm"` job array) |
| `parallel_cases()` / `run_case()` / `merge_cases()` | Many bouquets on one node: one standing solver per worker, cases swapped on (`Bouquet.set_case`), merged per input group |
| `CaseSpec` / `GeqdskProfilePairs` / `IdaTimeslices` / `ImasTimeslices` | The cases of a sweep (one baseline source each) and the sources that expand into them |
| `units.make_units()` / `units.units_from_cases()` / `units.add_units()` / `python -m bouquet.units work` | Many bouquets (one per baseline) on a shared-filesystem queue, across processes, nodes and jobs (`bouquet.workqueue`) |
| `run_shard()` | The per-worker serial pipeline on one shard |
| `merge_archives()` | Concatenate shards, verifying every shard hit the same baseline |
| `emit_slurm_script()` | Write the SLURM job-array + submit scripts |

## I/O

| Class / Function | Description |
|------------------|-------------|
| `GEQDSKEquilibrium` / `read_geqdsk()` | Full-featured COCOS-aware GEQDSK reader with flux-surface analysis |
| `read_eqdsk_from_bytes()` | Parse a g-file from in-memory bytes |
| `bouquet.io.write_geqdsk()` | Write a raw g-file dict to disk (import from `bouquet.io`) |
| `PFile` / `read_pfile()` | Osborne p-file reader/writer with rotation computation |
| `read_ida()` / `IDAProfiles` | IDA netCDF kinetic fits (direct + ensemble layouts) |
| `read_ida_cer()` / `IDACERProfiles` | Impurity CER channels |
| `read_imas_baseline()` / `read_imas_geometry()` | IMAS/OMAS data-dictionary JSON reader |
| `write_imas_draw()` / `export_imas_drawset()` | Reconstruct one / all perturbed IMAS/OMAS IDS from an archive |

Details: [io-and-plotting.md](io-and-plotting.md).

## Physics helpers

| Function | Description |
|---|---|
| `parallel_to_toroidal()` / `toroidal_to_parallel()` | Current-convention conversion `⟨j_φ⟩ = κ⟨j·B⟩`, `κ = F⟨1/R⟩/⟨B²⟩` (`field_aligned_conversion`), and its exact inverse; or the source's own `j_tor/j_total` ratio |
| `jpar_to_jphi_tokamaker()` / `jphi_tokamaker_to_jpar()` / `jphi_tokamaker_pressure_term()` | ⟨J·B⟩ ↔ TokaMaker `jphi` (field-aligned part) and the pressure term p′G ([current-conventions.md](current-conventions.md) A7) |
| `jtor_imas_to_jphi_tokamaker()` / `jphi_tokamaker_to_jtor_imas()` | IMAS `j_tor` ↔ TokaMaker `jphi` (A5) |
| `isotropize_fast_pressure()` | Anisotropic fast-pressure reduction for the isotropic GS solve (`method` is **required** — the two dd conventions differ by 3×) |
| `detect_p_fast_convention()` / `resolve_p_fast_reduction()` | Which fast-pressure storage convention a dd was written in, and the reduction rule that follows (`bouquet.io.imas`) |
| `fast_pressure_residual()` / `infer_fast_pressure()` | Fast-ion pressure accounting |
| `core_pressure_health()` / `core_pressure_hollow_record()` | Report-only description of core-pressure hollowness (rise above the axis value, its extent); gates nothing |
| `radial_field_from_impurity_force_balance()` | Impurity radial-force-balance E_r with propagated uncertainty |
| `Hmode_profiles()` | Synthetic H-mode profile generator |

## Sampling and uncertainty

| Function / Class | Description |
|------------------|-------------|
| `GPRProfilePerturber` | Gaussian process profile perturbation engine (Gibbs non-stationary kernels) |
| `generate_perturbed_GPR()` | One-call wrapper for perturbing a 1-D profile |
| `sigmoid_length_scale()` | Spatially varying correlation length for Gibbs kernels |
| `verify_gpr_statistics()` | Monte Carlo validation of the GPR sampling statistics |
| `calc_cylindrical_li_proxy()` | Fast cylindrical l_i proxy (no GS solve required) |
| `new_uncertainty_profiles()` | Build 1-D uncertainty envelopes (power-law or flat+tail) |
| `sigma_template()` | Relative 1-sigma radial templates (H-mode flat top / ramp / L-mode) of n_e, T_e, T_i, Z_eff, n_C, n_i; the assumed default envelope |
| `synthetic_ida_sigma()` | The H-mode flat-top template as a fractional sigma envelope for synthetic studies |

## Plotting

`plot_bouquet`, `plot_bouquet_timeseries`, `plot_traces`,
`plot_boundary_point_traces`, `plot_geqdsk_bouquet`, `plot_pfile_bouquet`,
`plot_kinetic_profiles`, `plot_jphi_profiles`, `plot_jphi`,
`plot_aux_profiles`, `plot_transport_profiles`, `plot_coil_currents`,
`plot_spec_summary`, `plot_tokamaker_comparison`, `plot_input_vs_recon`, the
`draw_*` axes-level variants, `set_plot_style`, and `WONG`. Signatures and
selection semantics: [io-and-plotting.md](io-and-plotting.md#plotting).

## Environment

| Function | Description |
|----------|-------------|
| `add_oft_to_path()` | Make OpenFUSIONToolkit importable (`OFT_PYTHONPATH` → known locations → walk-up) |
| `find_mesh()` | Locate the TokaMaker mesh (`BOUQUET_MESH` → walk-up → bundled example) |
| `find_ida()` | Locate an IDA `.cdf` (`BOUQUET_IDA`-as-file → `extra` → walk-up → `BOUQUET_IDA`-as-directory, searched recursively; differing same-named vintages raise rather than guess). Data lives outside the repo — nothing is bundled |

All three raise with the full list of locations tried, so failures are actionable on
a new machine.

## Functional API (pre-class, still supported)

Existing scripts written against the functional API keep working; new work
should use `Bouquet`.

| Function | Description |
|----------|-------------|
| `generate_bouquet()` | Batch driver: draw N perturbations, solve GS, archive to HDF5. Supports `psi_N_kinetic` for SOL-aware profiles |
| `perturb_kinetic_equilibrium()` | Single perturbation: draw profiles, match pressure and l_i, corrective jphi_total iteration |
| `reconstruct_equilibrium()` | Reconstruct one GS equilibrium from a g-file + profiles, with corrective iteration |
| `classify_jphi_profile()` | Classify the edge current profile (`H_mode` / `Lmode_like_jphi` / `L_mode`) |
| `fit_inductive_profile()` | Smoothing-spline + PCHIP fit of the inductive current, scaled to a target l_i |

A few knobs are reachable only here — notably `p_thresh` (pressure acceptance
band, default `0.05`) and `coil_drift_floor_A` (absolute coil-bound floor,
default 50 A).

> All tolerance arguments across both APIs are **fractions**
> (`l_i_tolerance=0.01` is 1%), never percentages.
