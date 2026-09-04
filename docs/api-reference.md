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
| `Bouquet.run_slices()` | Multi-slice IMAS sweep into one archive, one `scan_key` per slice |
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
| `ScanView` / `DrawView` | The view objects those return |
| `write_provenance()` / `load_config()` | Stamp / recover the exact `BouquetConfig` stored in an archive |
| `initialize_equilibrium_database()` | Create/open an archive (stamps `schema_version`) |
| `load_equilibrium()` / `load_equilibrium_by_path()` | Read one draw |
| `load_baseline_profiles()` | Read the per-scan baseline |
| `load_eq_fsa()` | Read the per-draw live-equilibrium flux-surface-average block |
| `discover_scan_keys()` / `count_equilibria()` / `list_equilibrium_indices()` | Archive introspection |
| `select_indices()` / `read_filter_flags()` | Filter-flag queries |
| `filter_coil_currents()` / `filter_boundaries()` / `export_filtered()` | Post-generation selection of the machine-realizable subset |
| `bouquet.filtering.passes_all_filters()` (+ `passes_coil_spec` / `passes_boundary_spec` / `boundary_deviation_mm`) | The selection predicate itself, on raw numbers rather than an archive. Shared with `generate_bouquet`'s [until-N](workflows.md#until-n-in-spec-draws) stopping rule so the two agree by construction. Module-level, not re-exported at the package root |

Layout spec: [archive-schema.md](archive-schema.md). Code source of truth:
[`bouquet/schema.py`](../bouquet/schema.py).

## Parallel generation

| Function | Description |
|---|---|
| `parallel_generate()` | Process-parallel driver (`backend="laptop"` pool or `"slurm"` job array) |
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
| `parallel_to_toroidal()` / `toroidal_to_parallel()` | Current-convention conversion, both directions, FSA-geometry aware |
| `isotropize_fast_pressure()` | Anisotropic fast-pressure reduction for the isotropic GS solve |
| `fast_pressure_residual()` / `infer_fast_pressure()` | Fast-ion pressure accounting |
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
| `synthetic_ida_sigma()` | IDA-shaped fractional sigma envelopes for synthetic studies |

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
| `perturb_kinetic_equilibrium()` | Single perturbation: draw profiles, match pressure and l_i, corrective j_phi iteration |
| `reconstruct_equilibrium()` | Reconstruct one GS equilibrium from a g-file + profiles, with corrective iteration |
| `classify_jphi_profile()` | Classify the edge current profile (`H_mode` / `Lmode_like_jphi` / `L_mode`) |
| `fit_inductive_profile()` | Smoothing-spline + PCHIP fit of the inductive current, scaled to a target l_i |

A few knobs are reachable only here — notably `p_thresh` (pressure acceptance
band, default `0.05`) and `coil_drift_floor_A` (absolute coil-bound floor,
default 50 A).

> All tolerance arguments across both APIs are **fractions**
> (`l_i_tolerance=0.01` is 1%), never percentages.
