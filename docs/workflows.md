# Workflows

How a bouquet is actually produced, stage by stage, and every knob that
controls it. See [`../README.md`](../README.md) for the short version and
[`../architecture.md`](../architecture.md) for the physics derivations.

## Contents

- [The pipeline](#the-pipeline)
- [Stage reference](#stage-reference)
- [What is perturbed vs. held fixed](#what-is-perturbed-vs-held-fixed)
- [Configuration reference](#configuration-reference)
- [until-N in-spec draws](#until-n-in-spec-draws)
- [Workflow presets and the guard](#workflow-presets-and-the-guard)
- [Reading an archive back](#reading-an-archive-back)
- [Exporting draws](#exporting-draws)
- [Timeseries sweeps](#timeseries-sweeps)
- [Process-parallel generation](#process-parallel-generation)

---

## The pipeline

```
Baseline: g-file + profiles (p-file / IDA), or IMAS/OMAS JSON
        │
        ▼
┌────────────────────────┐
│  Define uncertainties  │  IDA sigmas / synthetic_ida_sigma()
│  σ_ne, σ_Te, σ_Zeff, … │  or flat fractional envelopes
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Draw GPR perturbation │  GPRProfilePerturber
│  ne±δne, Te±δTe, Zeff… │  (Gibbs kernel, monotonicity enforced)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Derive n_i (quasi-    │  Z_eff-primary density scheme
│  neutrality) + p_total │  + fixed p_fast / j_NBI / j_RF
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Rebuild j_phi         │  j_ind (GPR) + j_BS (Sauter, per-draw)
│  Match pressure & l_i  │  + fixed anchors; secant l_i iteration
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Solve Grad–Shafranov  │  TokaMaker + coil-bound homotopy
│  Export g-file bytes   │
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Store to HDF5 (v2)    │  profiles + raw bytes + diagnostics
│  + provenance          │  + config_json / schema_version
└────────────────────────┘
```

The same flow is available as a rendered diagram, and as a full 550-node
logic map with a `file:line` anchor on every node:
**[interactive flowchart](https://d-burg.github.io/bouquet/flowchart/)**
([source + regeneration](flowchart/)).

## Stage reference

| Stage | Method | What it does |
|---|---|---|
| Solver | `setup_solver()` | Stands up the TokaMaker object from `SolverConfig` (mesh, order, isoflux/saddle constraints, VSC definition). Idempotent. |
| Baseline | `prepare()` / `reconstruct()` / `prepare_baseline()` | Resolves the baseline from `config.source`. `reconstruct()` is the g-file-path alias (`setup_solver()` + `prepare_baseline()`) and prints the reconstruction-fidelity summary; `prepare()` is the source-agnostic form. The IMAS path does a single forward solve instead of a reconstruction. |
| Guard | `verify_sigma0_consistency()` | Optional but recommended: one bootstrap solve confirming the *draw* pipeline reproduces the *baseline* j_BS split at σ=0. See [physics-notes.md](physics-notes.md#the-0-consistency-guard). |
| Draws | `generate(n=None)` | Draws `n` (default `GenerationConfig.n_equils`) perturbations, solves each, archives to `{header}.h5`. Returns the per-draw diagnostics list. With `generation.n_inspec_target` set, keeps drawing until that many pass the filters — see [until-N](#until-n-in-spec-draws). |
| Selection | `filter(rms_max_mm=None, plot=False)` | Applies the coil-drift and boundary-RMS filters, writing non-destructive pass flags into the archive. Returns a summary dict. |
| Export | `export()` / `export_bundle()` / `export_ids()` | Pruned `{header}_selected.h5`, a per-draw file bundle, or one IMAS/OMAS IDS per draw. |
| All of it | `run()` | `setup_solver → prepare_baseline → generate → filter → export`, idempotent on the early stages. |

Introspection helpers on the run object: `describe()` (prints only the
non-default knobs), `archive` (its `BouquetArchive`), `selected_indices()`,
`output_spread()`, `plot_baseline()`, `plot_bouquet()`, `plot_traces()`,
`plot_coil_currents()`, `plot_spec_summary()`.

## What is perturbed vs. held fixed

| Quantity | Perturbed? | Notes |
|----------|:----------:|-------|
| n_e, T_e, T_i | ✓ | Drawn from the GPR posterior |
| Z_eff | ✓ | Active channel (default on via `zeff_scalar_sigma`) |
| n_i, n_z (impurity) | derived | Quasi-neutrality with the drawn Z_eff (`impurity_Z`) |
| Total pressure (p_tot) | ✓ | Recomputed from the perturbed kinetics |
| Bootstrap current (j_BS) | ✓ | Sauter model recomputed per draw (`recalculate_j_BS`) |
| Inductive current (j_ind) | ✓ | GPR-perturbed, then scaled to match l_i |
| Coil currents | ✓ | Adjusted by TokaMaker within the homotopy bounds |
| Aux channels (ω_tor, E_r, χ_e, χ_i) | optional | Switchboard: perturbed + stored when sigmas are supplied (passive) |
| p_fast, j_NBI, j_RF | ✗ | Fixed additive components, never perturbed |
| Equilibrium anchors (p_diff, jphi_diff, jBS_diff) | ✗ | Fixed offsets applied to the baseline **and** every draw |

## Configuration reference

Every knob lives on a `BouquetConfig` sub-object, reachable from the run
object as `b.solver`, `b.source`, `b.uncertainty`, `b.generation`,
`b.filtering`, `b.fixed_components`. The dataclass docstrings in
[`bouquet/config.py`](../bouquet/config.py) are the source of truth; this table
is a navigational summary of the defaults.

### `SolverConfig` (`b.solver`)

| Knob | Default | Meaning |
|---|---|---|
| `mesh_path` | *(required)* | TokaMaker mesh; `bq.find_mesh()` resolves it |
| `nthreads` | `1` | **Keep at 1.** OpenMP reduction order is non-deterministic and jitters `l_i(1)` by ~1%; parallelise across processes instead |
| `order` | `3` | FE order |
| `F0` | `None` | Vacuum `R·B_t`; defaults from the g-file / IDS |
| `isoflux_pts`, `isoflux_weights` | `None` | Boundary constraint points; default from the source boundary |
| `saddle_targets`, `saddle_weights` | `None` | Opt-in X-point pins. Without them a diverted forward solve typically rounds the boundary corner by a few cm |
| `coil_vsc` | `{"F9A": 1.0, "F9B": -1.0}` | Antisymmetric vertical-stability channel definition |
| `region_overrides` | `None` | Special-case mesh cond/coil dict edits |

### `UncertaintyConfig` (`b.uncertainty`)

| Knob | Default | Meaning |
|---|---|---|
| `ida_path` | `None` | IDA `.cdf` supplying measured sigma envelopes (overrides the scalars below) |
| `sigma_mode` / `sigma_method` | `"auto"` / `"percentile"` | IDA layout dispatch (direct `*_err` vs. ensemble posterior) and ensemble reduction |
| `log_sigma_sources` | `True` | Log one line per kinetic channel naming the source that actually won the precedence |
| `ne_scalar_sigma` | `0.05` | Flat fractional envelope used when no IDA sigmas are supplied |
| `te_scalar_sigma` | `0.05` | " |
| `ni_scalar_sigma` | `0.10` | " |
| `ti_scalar_sigma` | `0.10` | " |
| `jphi_scalar_sigma` | `0.10` | Inductive-current envelope. **Must be > 0** — setting it to 0 freezes `j_inductive` and trips the workflow guard |
| `zeff_scalar_sigma` | `0.05` | One Z_eff perturbation per draw; n_i / n_z follow from quasi-neutrality |
| `sigma_profiles` | `{}` | Explicit `{name: sigma(psi_N)}` envelopes, highest precedence |
| `n_ls` / `t_ls` / `j_ls` | `0.5` / `0.4` / `0.25` | GPR correlation lengths for density / temperature / current |
| `aux_sigmas`, `aux_baselines`, `aux_length_scales` | `{}` | The passive switchboard: any extra channel gets perturbed and archived alongside the physics |

**Precedence, per kinetic channel:** `sigma_profiles[chan]` > an IDA `.cdf` >
`<chan>_scalar_sigma`. A `.cdf` handed to `ReconstructionSource.profiles_path`
is adopted as an IDA source automatically, so it counts here even when
`ida_path` is unset. A winning source **shadows** the ones below it rather than
combining with them — which means zeroing `*_scalar_sigma` to get a
deterministic σ=0 point is a **no-op** against an IDA source, and every such
"deterministic" point is a full-σ draw. The only setting that wins is an
explicit profile:

```python
n_kin = len(b.baseline.psi_N_kinetic)
b.uncertainty.sigma_profiles = {ch: np.zeros(n_kin)
                                for ch in ("ne", "te", "ni", "ti")}
```

`resolve_uncertainty` logs the winning source per channel (disable with
`log_sigma_sources=False`) and warns when a scalar you moved off its default is
being ignored. `sigma_jphi` and the aux channels have no `.cdf` branch, so
their scalars always apply.

### `GenerationConfig` (`b.generation`)

| Knob | Default | Meaning |
|---|---|---|
| `n_equils` | `20` | Draws to attempt — or, with `n_inspec_target` set, the initial allocation |
| `n_inspec_target` | `None` | Draw **until N draws pass both filters** rather than exactly `n_equils`. See [until-N](#until-n-in-spec-draws) below |
| `max_total_draws` | `None` | Attempt cap for `n_inspec_target` (default `5 ×` the target). Hitting it is a loud, non-fatal outcome |
| `seed` | `None` | The run's one RNG seed. Consumed once into a single `numpy.random.Generator` threaded into every draw (GPR kinetic/aux/j_φ, the per-draw `scale_jBS`, the per-draw l_i target), so the same seed + inputs + solver gives a **bitwise-identical** archive on one machine (across machines the draws agree to ~1e-9, not bitwise -- LAPACK/BLAS). `None` = OS entropy |
| `scan_key` | `0` | Label for this bouquet within the archive (`scan/<key>/`) — a time in ms, a beta value, … |
| `l_i_tolerance` | `0.05` | l_i acceptance band, as a **fraction** of target |
| `constrain_sawteeth` | `False` | Gate draws on q0 |
| `recalculate_j_BS` | `True` | Recompute the Sauter bootstrap per draw (vs. reusing the baseline's) |
| `jbs_delta_mode` | `False` | Opt-in differential bootstrap composition — see [physics-notes.md](physics-notes.md#differential-bootstrap-jbs_delta_mode) |
| `isolate_edge_jBS` | `True` (dataclass) | Both `from_geqdsk` and `from_imas` set this **`False`**: the unified forward decomposition (pure-ohmic `j_inductive`, full bootstrap in `j_BS`) closes exactly and yields better. Flip to `True` only for dedicated edge-spike studies |
| `jBS_baseline_mode` | `"diff"` | IMAS path: how the SWB bootstrap is reconciled with the source (`"diff"` / `"rescale"`) |
| `kinetic_source` | `"fuse"` | IMAS path: `"ida_hybrid"` takes ne/Te/Ti/ω_tor from an IDA `.cdf` while keeping FUSE Z_eff / currents / equilibrium. `from_imas(ida_path=…)` selects it automatically |
| `anchor_jtor_to_equilibrium` | `True` | IMAS path: anchor total j_phi to `equilibrium.profiles_1d.j_tor` rather than `core_profiles.j_tor` |
| `anchor_pressure_to_equilibrium` | `False` | IMAS path: add the fixed `p_diff = equilibrium.pressure − p_reconstructed` offset |
| `imas_corrective_jphi` | `False` | Opt-in corrective j_phi iteration on the IMAS baseline solve (still being validated) |
| `floor_j_BS` | `False` | Clip negative bootstrap excursions; only needed with `isolate_edge_jBS=False` on sources that carry an inner negative lobe |
| `swb_iterations` | `3` | `solve_with_bootstrap` self-consistency iterations per draw |
| `coil_drift` | `0.01` | Soft coil-drift target |
| `coil_drift_hard_factor` | `None` | Optional hard inequality bounds at `± factor·coil_drift` in every solve |
| `homotopy_passes` | `[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]` | Progressive `(F_tol, VSC_tol)` schedule — see [coil-constraints.md](coil-constraints.md) |
| `workflow` | `"auto"` | Named preset; see below |
| `allow_unsafe_workflow` | `False` | Deprecated alias for `workflow="custom"` |
| `allow_incomplete_pressure` | `False` | IMAS path: bypass the fail-fast pressure-accounting check |
| `capture_live_eq` | `True` | Snapshot each draw's converged flux-surface averages into `eq_fsa/` — what makes `fidelity="exact"` IDS export possible |
| `capture_npsi` | `257` | FSA grid for that block |
| `capture_exact_inv_R2` | `True` | Compute ⟨1/R²⟩ by exact quadrature rather than the ⟨B_φ²⟩≈⟨B²⟩ bracket |
| `diagnostic_plots` | `False` | Per-draw diagnostic figures |

### `FilterConfig` (`b.filtering`)

| Knob | Default | Meaning |
|---|---|---|
| `rms_max_mm` | `5.0` | Boundary-RMS acceptance threshold |
| `inspec_F_max` | `0.02` | Max non-VSC F-coil drift (fraction) for `in_spec` |
| `inspec_VSC_max` | `0.02` | Max VSC-channel drift (fraction) for `in_spec` |

### `FixedComponentsConfig` (`b.fixed_components`)

`p_fast`, `j_NBI`, `j_RF` on their own `psi_N` grid — additive components that
are never perturbed. `p_fast_reduction` (default `"trace"`) selects the
anisotropic fast-pressure reduction applied before the isotropic GS solve.

> **Tolerances are fractions, not percentages.** `l_i_tolerance=0.05` means
> 5%. This applies to every tolerance argument in the package.

> **`p_thresh`** (the volume-averaged pressure acceptance band, default `0.05`,
> calibrated to a realistic `<P>` measurement uncertainty) is currently a
> `generate_bouquet` keyword only — it is not surfaced on `GenerationConfig`,
> so the class API always uses the default.

## until-N in-spec draws

By default a run draws exactly `n_equils` times and you get whatever yield the
equilibrium gives — 20 draws might leave 12 selected on a stiff case and 19 on
an easy one, so ensembles are not comparable in size across shots. Setting
`generation.n_inspec_target` inverts that: the loop keeps drawing until **N
draws pass both filters**, then stops.

```python
b.generation.n_equils = 20            # initial allocation, not the total
b.generation.n_inspec_target = 20     # what you actually want delivered
b.generation.max_total_draws = 60     # optional; default is 5 x the target
b.filtering.rms_max_mm = 5.0          # the bounds the loop stops on
b.generate()
b.filter()                            # -> >= 20 selected
```

Points worth knowing:

- **The count is the one `filter()` will agree with.** The loop's per-draw
  verdict comes from `filtering.passes_all_filters`, which is the same
  `passes_coil_spec` / `passes_boundary_spec` pair the postprocess filters use,
  over the same `boundary_deviation_mm` metric and the same two archived LCFS
  contours. It reads its thresholds from `config.filtering`, so stopping at N
  and then filtering to fewer than N is not a state this can reach. (Where the
  two *can* differ — a draw whose high-res LCFS trace failed — the loop calls
  it out of spec while the postprocess falls back to the coarse eqdsk contour,
  so the run over-delivers rather than under-delivers.)
- **Re-cutting afterwards is still your call.** The identity is against the
  thresholds in `config.filtering` at generation time. Passing a different
  bound to `filter(rms_max_mm=…)` later re-cuts the archive at the new
  criterion, and the selected count moves accordingly — that is the filters
  working as designed, not the loop having miscounted.
- **`run_slices` chases the target per slice.** Each slice gets its own N
  in-spec draws, which is usually what a timeseries sweep wants; budget the
  wall-clock as N-per-slice divided by the worst slice's yield.
- **Nothing is discarded.** Out-of-spec draws are archived exactly as before;
  the run just doesn't stop until N have passed. The yield is still visible in
  `filter()`'s summary and `plot_spec_summary()`.
- **The cost scales with the inverse yield.** A 40 %-yield equilibrium spends
  ~2.5× the solves of a 100 %-yield one for the same delivered ensemble. Budget
  wall-clock accordingly; the per-draw log prints the running tally and a
  yield-projected ETA.
- **Hitting `max_total_draws` is a failure, not an answer.** It warns
  (`RuntimeWarning`) and says how far it got. The fix is more attempts, or a
  deliberate decision about the thresholds — not a quietly short bouquet.
- **Serial only.** `parallel_generate`, `run_shard` and `emit_slurm_script` all
  reject a config carrying `n_inspec_target`: shards cannot see each other's
  yield, so N workers each chasing the target would deliver N×target draws.
- **The RNG stream is untouched when the feature is off.** The `jBS_scales`
  block draw is unchanged for the first `n_equils` draws and only extends past
  it when until-N actually runs on, so `n_inspec_target=None` runs are bitwise
  what they were.

## Workflow presets and the guard

`from_geqdsk` and `from_imas` each auto-apply the flag combination validated
for their path, and `generate()` raises on a known-bad combination:

| Preset | Applied by | What it sets |
|---|---|---|
| `geqdsk-standard` | `from_geqdsk` | Standard flagship l_i loop (`perturb_jind_in_anchor=False`), unified decomposition (`isolate_edge_jBS=False`) |
| `imas-diff-c` | `from_imas` | Bootstrap anchored to the source via the fixed diff (`jBS_baseline_mode="diff"`), inductive perturbed in the recon-anchor (`perturb_jind_in_anchor=True`), unified decomposition |
| `auto` *(default)* | — | Resolve per source type at `generate()` |
| `custom` | you | Leave the flags as set and downgrade the guard to a warning. For deliberate backend experiments only |

Known-bad combinations the guard rejects: geqdsk + `perturb_jind_in_anchor`
(drops draws on stiff, high-l_i baselines via band-conditioning rejection),
IMAS without it (the matching loop homogenizes the draws), and
`jphi_scalar_sigma <= 0` (freezes `j_inductive`, so the draws carry no
current-profile uncertainty at all).

## Reading an archive back

```python
ar = bq.BouquetArchive("my_run.h5")     # or bq.BouquetArchive(b)
ar.scan_keys                            # e.g. ['0']
sc = ar["0"]
sc.indices, sc.baseline                 # draw indices (gap-tolerant), baseline dict
for d in sc.selected:                   # DrawViews passing the filters
    print(d.count, d.li1, d.flags)
eq = sc[3].equilibrium()                # parsed GEQDSKEquilibrium from stored bytes
sc[3].extract("out/", formats=("geqdsk", "pfile"))   # write the raw files

cfg = bq.load_config("my_run")          # the exact BouquetConfig that made it
```

Functional readers are available for scripted access — `load_equilibrium`,
`load_baseline_profiles`, `load_eq_fsa`, `discover_scan_keys`,
`count_equilibria`, `list_equilibrium_indices`, `select_indices`,
`read_filter_flags`, `export_filtered`, `write_provenance`, `load_config`.
Prefer any of these over raw `h5py`; the on-disk layout is specified in
[archive-schema.md](archive-schema.md).

```python
for sk in bq.discover_scan_keys("run.h5"):
    print(sk, bq.count_equilibria("run.h5", scan_key=sk))
```

Pre-v2 archives (written before 2026-07) are detected by the missing
`schema_version` attr: `BouquetArchive` opens them with a warning and
`load_equilibrium` raises a clear error. Regenerate them with the current
package.

## Exporting draws

Two targets for handing the ensemble to codes that don't read the HDF5
archive: a per-draw **file bundle** (g-file / p-file / self-describing
profiles JSON) or **one IMAS/OMAS `equilibrium` + `core_profiles` IDS per
draw**. `selection` is `"selected"` (the in-spec subset, default) or `"all"`.

```python
b.export_bundle("bundle/", formats=("geqdsk", "profiles"))   # -> {draw: {fmt: path}}
# equivalently from an archive on disk, honouring the same selection:
bq.BouquetArchive("my_run.h5")["0"].extract("bundle/", formats=("geqdsk", "pfile"))

b.export_ids("ids/", fidelity="exact")            # IMAS/OMAS source only
```

The profiles JSON is source-agnostic and carries everything needed to rebuild
the state elsewhere: the perturbed profiles and their units, scalar diagnostics
(`l_i`, `I_p`, …), coil currents by name, and the captured flux-surface-averaged
geometry (`eq_fsa`).

### IDS current-split fidelity

The toroidal current `j_tor` in the IDS is always exact. The *parallel* split
IMAS stores (`j_total` / `j_ohmic` / `j_bootstrap` = ⟨**j**·**B**⟩/B₀) needs a
flux-surface geometry factor to convert from bouquet's toroidal components, and
`fidelity` picks where that factor comes from:

| `fidelity` | Parallel split uses | When |
|---|---|---|
| `"exact"` | the draw's **own** captured `eq_fsa` geometry (`toroidal_to_parallel`) | draws deviate from the baseline; the split must track each perturbed equilibrium |
| `"reconstruct"` | the baseline template ratio `c = j_tor/j_total` | exact only when a draw's flux geometry matches the baseline's |
| `"auto"` *(default)* | exact when the `eq_fsa` block is present, else reconstruct | — |

`eq_fsa` is captured at generate time from the live TokaMaker object
(`GenerationConfig.capture_live_eq`, on by default), so a freshly generated
archive supports `"exact"` out of the box. Across an ensemble the two paths
differ by a few percent per draw — which is the point of capturing the live
geometry rather than reusing the baseline's.

## Timeseries sweeps

One IMAS/OMAS file often holds many time slices. `run_slices` sweeps them into
a single archive, one `scan_key` per slice, reusing the solver:

```python
b = bq.Bouquet.from_imas("dd_sim.json", mesh=bq.find_mesh(), n_draws=20,
                         header="my_sweep")
b.setup_solver()
metrics = b.run_slices(times=[2.10, 2.20, 2.30],
                       scan_keys=[2100, 2200, 2300])
# -> {2100: {time, n_all, n_sel, l_i, Ip}, ...} all in my_sweep.h5
```

`scan_keys` defaults to the time in ms. Reconstruction sources have no time
axis — build one `Bouquet` per g-file instead.

## Process-parallel generation

Draws are embarrassingly parallel, and `OFT_env` is a per-process singleton —
so parallelism is across **processes**, one single-threaded TokaMaker per
physical core (`nthreads=1` is the validated regime: bit-reproducible
baselines, no OpenMP l_i jitter, no DLSODE hangs).

```python
cfg = b.config                       # any BouquetConfig

summary = bq.parallel_generate(      # laptop: ProcessPoolExecutor (spawn)
    cfg, n_workers=None,             # None -> physical core count
    threads_per_worker=1, seed=1234,
    backend="laptop",
)

paths = bq.parallel_generate(        # cluster: emit a SLURM job-array + merge
    cfg, n_workers=32, seed=1234, threads_per_worker=1,
    backend="slurm",
    slurm=dict(out_dir="slurm_jobs", job_name="my_run",
               setup=["export OFT_PYTHONPATH=/path/to/OFT/python"]),
)
# then: bash slurm_jobs/my_run_submit.sh   (works from any CWD)
```

Each worker runs the ordinary serial pipeline on its shard and writes
`{header}_w{i}.h5`; the merge concatenates them into `{header}.h5`,
**verifying every shard converged to the same baseline** before copying, and
stamps the run-level config provenance. Worker seeds derive from
`SeedSequence(seed, worker_id, scan_key)`, so timeseries slices swept with one
seed are decorrelated. Parallel draws are statistically equivalent to — but not
bit-identical with — a serial run of the same seed.
