# Workflows

How a bouquet is actually produced, stage by stage, and every knob that
controls it. See [`../README.md`](../README.md) for the short version and
[`../architecture.md`](../architecture.md) for the physics derivations.

## Contents

- [The pipeline](#the-pipeline)
- [Stage reference](#stage-reference)
- [What is perturbed vs. held fixed](#what-is-perturbed-vs-held-fixed)
- [Configuration reference](#configuration-reference)
- [Experimental features and their validation status](#experimental-features-and-their-validation-status)
- [Graduated features and their known limitations](#graduated-features-and-their-known-limitations)
- [The in-spec rule](#the-in-spec-rule)
- [until-N in-spec draws](#until-n-in-spec-draws)
- [Workflow presets and the guard](#workflow-presets-and-the-guard)
- [Reading an archive back](#reading-an-archive-back)
- [Error bars from an archive](#error-bars-from-an-archive)
- [Exporting draws](#exporting-draws)
- [Timeseries sweeps](#timeseries-sweeps)
- [Process-parallel generation](#process-parallel-generation)
- [Many bouquets on one node](#many-bouquets-on-one-node)
- [Many bouquets on a cluster](#many-bouquets-on-a-cluster)
- [Running bouquet on another device](#running-bouquet-on-another-device)

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
│  neutrality) + p_total │  + fixed p_fast / jphi_nbi / jphi_rf
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│  Rebuild jphi_total         │  j_ind (GPR) + jphi_bs (Sauter, per-draw)
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
| Guard | `verify_sigma0_consistency()` | Optional but recommended: one bootstrap solve confirming the *draw* pipeline reproduces the *baseline* jphi_bs split at σ=0. See [physics-notes.md](physics-notes.md#the-0-consistency-guard). |
| Draws | `generate(n=None)` | Draws `n` (default `GenerationConfig.n_equils`) perturbations, solves each, archives to `{header}.h5`. Returns the per-draw diagnostics list. With `generation.n_inspec_target` set, keeps drawing until that many pass the filters — see [until-N](#until-n-in-spec-draws). |
| Selection | `filter(rms_max_mm=None, plot=False)` | Applies the coil-drift and boundary-RMS filters, writing non-destructive pass flags into the archive. `rms_max_mm=None` applies `filtering.rms_max_mm`; a number, `"auto"` or `"off"` overrides it for this call (announced and stamped). Returns a summary dict. |
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
| Z_eff | ✓ | Active channel (on by default; `zeff_scalar_sigma=0.0` turns it off) |
| n_i, n_z (impurity) | derived | Quasi-neutrality with the drawn Z_eff (`impurity_Z`) |
| Total pressure (p_tot) | ✓ | Recomputed from the perturbed kinetics |
| Bootstrap current (jphi_bs) | ✓ | Sauter model recomputed per draw (`recalculate_jbs`) |
| Inductive current (j_ind) | ✓ | GPR-perturbed, then scaled to match l_i |
| Coil currents | ✓ | Adjusted by TokaMaker within the homotopy bounds |
| Aux channels (ω_tor, E_r, χ_e, χ_i) | optional | Switchboard: perturbed + stored when sigmas are supplied (passive) |
| p_fast, jphi_nbi, jphi_rf, jphi_other | ✗ | Fixed additive components, never perturbed (`jphi_other`: fusion, runaways, sawteeth and unknown-index core_sources entries on the IMAS path) |
| Equilibrium anchors (p_diff, jphi_diff, jphi_bs_diff) | ✗ | Fixed offsets applied to the baseline **and** every draw |

## Configuration reference

Every knob lives on a `BouquetConfig` sub-object, reachable from the run
object as `b.solver`, `b.source`, `b.uncertainty`, `b.generation`,
`b.filtering`, `b.fixed_components`. The dataclass docstrings in
[`bouquet/config.py`](../bouquet/config.py) are the source of truth; this table
is a navigational summary of the defaults.

### `SolverConfig` (`b.solver`)

| Knob | Default | Meaning |
|---|---|---|
| `mesh_path` | *(required)* | TokaMaker mesh; `bq.find_mesh()` resolves it. Loaded by `utils.load_mesh`: the detected device's turn-table coils (DIII-D F coils) at `nturns` 1, so their currents are ampere-turns whatever turns the mesh carries (OFT #268's DIII-D mesh: 58 / 55) |
| `nthreads` | `1` | **Keep at 1.** OpenMP reduction order is non-deterministic and jitters `l_i(1)` by ~1%; parallelise across processes instead |
| `order` | `3` | FE order |
| `nl_tol` | `None` | TokaMaker's Picard tolerance for every solve; `None`: OpenFUSIONToolkit's default (1e-6) |
| `F0` | `None` | Vacuum `R·B_t`; defaults from the g-file / IDS |
| `isoflux_pts`, `isoflux_weights` | `None` | Boundary (isoflux) constraint points and weights, **used as given** (never resampled; points without weights get per-length weights summing to `ISOFLUX_WEIGHT_TOTAL` under the unified engine, with a UserWarning). Default (unified engine, #84): the source boundary resampled to `solver_geometry.N_ISOFLUX` = 256 points by arc length, X-points kept exactly, each point weighted by the arc length it represents, the weights summing to `solver_geometry.ISOFLUX_WEIGHT_TOTAL` = 2e5 (a constant inside a measured window: engine.md "The boundary constraint: a constant total inside a measured window"), or `isoflux_weight_total` when set -- one set for the reconstruction and the draws, whatever the source's point count. Legacy route: the source boundary as given at `ISOFLUX_WEIGHT` = 500 per point. See [The boundary constraint](#the-boundary-constraint-per-unit-length-84) |
| `isoflux_weight_total` | `None` | The total weight of the unified engine's standard boundary set. `None`: the constant `solver_geometry.ISOFLUX_WEIGHT_TOTAL` = 2e5, inside the measured window (engine.md, "The boundary constraint: a constant total inside a measured window"; stamped as `isoflux_weight_mode` "constant"). A number: an explicit OVERRIDE, installed as given, stamped "override". Also the total of an `isoflux_pts` set given without weights |
| `boundary_partition_probe` | `True` | The report-only boundary partition probe: ONE capped solve after the reconstruction measures how a 2 % pressure step splits between the boundary cut and the coil cut at the installed total (vertical-control pair excluded); stamped under `isoflux_balance.partition` in `solver_geometry_json` and `boundary_balance` in the engine record, warned outside [1/3, 3]; never changes the total. `False` skips it |
| `saddle_targets`, `saddle_weights` | `None` | Opt-in X-point pins. Without them a diverted forward solve typically rounds the boundary corner by a few cm |
| `coil_vsc` | `{"F9A": 1.0, "F9B": -1.0}` | Antisymmetric vertical-stability channel definition |

### The boundary constraint per unit length (#84)

Under the unified engine every reconstruction and every draw holds the
boundary with ONE standard isoflux set, whatever the source carried: the
source boundary (g-file boundary, IDS `boundary.outline`, or `LCFS_geqdsk`)
resampled to `solver_geometry.N_ISOFLUX` = **256** points by arc length
(closed, counter-clockwise, the X-points kept as exact points), each point
weighted by the arc length it represents, the weights summing to the
case's total: `solver_geometry.ISOFLUX_WEIGHT_TOTAL` = 2e5, a constant inside
a measured window (engine.md, "The boundary constraint: a constant total
inside a measured window"; `SolverConfig.isoflux_weight_total` overrides it). The first point -- the one
TokaMaker measures the other points' flux against -- is the outermost point,
never the X-point. Before (1.3.x and the 1.4.0 candidates), the IDS route held
its ~200 outline points at 500 each and the g-file route its 100-257 boundary
points at 200 each, so the strength followed the file's point count. **Every
reconstruction and draw therefore moves slightly**: on the validation cases
l_i(3) by at most 1.4e-4 and q by at most 3.7e-4 (under a fifth of the sigma=0
gates), with the source boundary held as well or better; the g-file case whose
sigma=0 draw failed before now passes, and its perturbed draws are in spec
(numbers: [engine.md, "The boundary constraint per unit length"](engine.md)).
`SolverConfig.isoflux_pts` / `isoflux_weights` are used as given (never
resampled; also on the engine's g-file route); `saddle_targets` are installed
as configured and mark their X-point for the resampling. The legacy route is
unchanged (the source boundary at `ISOFLUX_WEIGHT` = 500 per point; its g-file
reconstruction's own set at 200). The archive's `solver_geometry_json` stamps
`isoflux_rule`, `isoflux_n`, `isoflux_weight_total`, `isoflux_source_n`,
`boundary_length_m` and `isoflux_resampling` (method, N, the X-points kept).

### Ion and Z_eff fields of the sources (`b.source`, PR #56)

| Knob | Default | Meaning |
|---|---|---|
| `ReconstructionSource.ni_source` | `"standard"` | IDA `.cdf` profiles: where Z_eff and n_i come from. `"standard"` (the route before PR #56): Z_eff is the file's visible-bremsstrahlung value, n_i = n_e(Z − Z_eff)/(Z − 1) at Z_eff clamped to `[1, Z]`, σ_ni the n_e fraction; the Z_eff envelope comes from the carbon > VB > scalar ladder below. The measured routes (no longer experimental since 2026-10; [known limitations](#graduated-features-and-their-known-limitations), `ida_ion_route`): `"Zeff"` = the VB Z_eff with a propagated σ_ni; `"CER"` = `Z_eff = 1 + Z(Z−1)·n_C/n_e` from the measured `n_12C6` (needs `impurity_Z=6`); `"all"` = the equal-weight mean of the two, each clamped to `[1, Z]` first, their disagreement beyond σ added to σ_ni / σ_Zeff (a file lacking one route's envelope uses the other **alone**: an older vintage without `Zeff_err` gets the CER value). Recorded as `IDAProfiles.zeff_provenance`. Ignored for p-files |
| `ImasSource.ni_source` | `"auto"` | The same choice for the IMAS path. **`"auto"` (default since 2026-10): ONE thermal-kinetics source per run** -- on `ida_hybrid` it resolves to the measured IDA route `"all"` (Z_eff and n_i from the IDA file: the VB+CER mean where both routes have an envelope, else whichever does, each clamped to `[1, Z]` and counted); with FUSE kinetics to `"standard"`. An IDA file with no Z_eff information (no `Zeff`, no `n_12C6`) is **refused** on `ida_hybrid`, naming the explicit choices. `"standard"` (the default before 2026-10, an explicit choice now): Z_eff stays the dd's and n_i = n_e,IDA(Z − Z_eff,dd)/(Z − 1). `"Zeff"` / `"CER"` / `"all"`: that measured route explicitly. The route taken is stamped warning-free: `li_metrics["zeff_source"]` (`"ida"` / `"dd"`) and `li_metrics["kinetics_route"]` |
| `ImasSource.zeff_from_fuse` | `False` | `ida_hybrid` on a measured route: `True` keeps the dd's Z_eff (an explicit choice of the dd's Z_eff: its total, as on the plain dd route, `zeff_includes_fast` following the beam; owner decision 2026-10-18, before it the thermal numerator) while n_e/T_e/T_i/n_i come from IDA (on a file with no Z_eff information the route resolves to `"standard"`, and the Z_eff is the dd's total). Under `"standard"` the Z_eff is the dd's total anyway |
| `ImasSource.ni_subtract_fast` | `None` (auto) | `ida_hybrid` on a measured route: subtract the dd's fast-ion density equivalent from IDA's (total) n_i (no longer experimental since 2026-10; [known limitations](#graduated-features-and-their-known-limitations), `ida_ni_beam_subtraction`). `None`: follows `include_pfast` on that route (the #79 rule: the fast ions in `p_fast` with the subtraction, or both off), `False` elsewhere. `True` with `ni_source="standard"` (or on FUSE kinetics) is refused |
| `ImasSource.include_pfast` | `True` | Add the dd's fast-ion pressure `p_fast` to the equilibrium pressure (an explicit `FixedComponentsConfig.p_fast` still wins). On `ida_hybrid` the **one declared cross-source input** (the IDA fits carry no fast ions). On a measured `ni_source` match it to `ni_subtract_fast` (the default does): both `True` puts the fast ions in `p_fast`, both `False` leaves them inside the IDA n_i; on a beam slice the reader warns when they differ (fast ions counted twice, or left out) |
| `ImasSource.zeff_fast_ions` | `False` | **EXPERIMENTAL** (`fuse_zeff_fast_ions`): classify the dd's stored Z_eff against the thermal-only and thermal+fast numerators (`io.imas._dd_zeff`) and give the bootstrap `Z_eff,th + z2_fast/n_e` where it counts the fast ions. `False` (default): on the plain dd route the dd's own **total** Z_eff -- Σ_s Z_s² (n_s^th + n_s^fast)/n_e from the dd's densities, or the `zeff` dataset when it agrees with that within 5 % (`io.imas.ZEFF_DATASET_RTOL`; a dataset matching neither definition is warned about and not used) (owner decision A1, 2026-10-17; `io.imas._dd_total_zeff`); the same on the `ida_hybrid` `ni_source="standard"` sub-route (owner decision 2026-10-18; its n_i / Z_imp keep the thermal numerator); and with `zeff_from_fuse=True` on a measured `ida_hybrid` route (owner decision 2026-10-18; its n_i comes from the IDA Z_eff). The decision is archived as `li_metrics["zeff_dd_provenance"]` |
| `Baseline.z2_fast` | — (read) | Σ_s Z_s² n_s^fast on the kinetic grid, from the dd's fast-ion species (IMAS path); `None` without a fast population. Archived per draw with `z_fast` |
| `Baseline.zeff_includes_fast` | — (read) | Whether the baseline Z_eff's numerator counts the fast ions. On the plain dd route and the `ida_hybrid` `ni_source="standard"` sub-route `True` with a beam (A1 and the 2026-10-18 decision: the total Z_eff; `False` without one, where the numerators coincide); with `ImasSource.zeff_fast_ions=True` classified from the dd's stored `zeff` against both numerators (`io.imas._dd_zeff`; with no stored Z_eff and a beam, `True`), and `True` for a measured (IDA) Z_eff on the `ida_hybrid` measured route (the default; with `zeff_from_fuse=True` the dd's total, `True` with a beam). When `True`, the bootstrap sees `Z_eff,th + z2_fast/n_e` (since PR #56; see [physics-notes.md](physics-notes.md#kinetic-assumptions-z_eff-n_i-and-the-clips-pr-56)) and the draw window is `physics.zeff_bounds`' includes-fast one |

`read_ida(sigma_ni_from_ne=...)` is accepted as a deprecated no-op (it warns);
`UncertaintyConfig.sigma_ni_from_ne` was removed (it was already a no-op), and a
stored config carrying it still loads.

### `UncertaintyConfig` (`b.uncertainty`)

| Knob | Default | Meaning |
|---|---|---|
| `ida_path` | `None` | IDA `.cdf` supplying measured sigma envelopes (overrides the IMAS error fields and the assumed defaults below) |
| `sigma_mode` / `sigma_method` | `"auto"` / `"percentile"` | IDA layout dispatch (direct `*_err` vs. ensemble posterior) and ensemble reduction. An `ImasSource` with no IDA sigma file reads the dd's own core_profiles `*_error_upper` under `"auto"` / `"direct"` (`io.imas.read_imas_sigmas`; not on `ida_hybrid`, not under `"ensemble"`) |
| `log_sigma_sources` | `True` | Log one line per kinetic channel naming the source that actually won the precedence |
| `ne_scalar_sigma` | `None` | The **assumed** envelope of a channel with no measured sigma (no IDA file, no IMAS error field). `None` = the H-mode flat-top radial template, `bouquet.sigma_template(psi_N, chan)` × \|profile\| (relative σ vs ψ_N, evaluated at the nodes' ψ_N in a Φ_N run); a number = an explicit flat fraction, which wins. Default changed 2026-10-14 (was `0.05`; see below) |
| `te_scalar_sigma` | `None` | " (was `0.05`) |
| `ni_scalar_sigma` | `None` | " (the derived-n_i template row; was `0.10`; used only when `ni_from_zeff` resolves False) |
| `ti_scalar_sigma` | `None` | " (was `0.10`) |
| `jphi_scalar_sigma` | `0.10` | Inductive-current envelope. **Must be > 0** — setting it to 0 freezes `jphi_ind` and trips the workflow guard. `solve_method="swb"` applies it as a relative width on the inductive seed (`swb.swb_jind_sigma`), not on `|jphi_total|` |
| `zeff_scalar_sigma` | `None` | One Z_eff perturbation per draw; n_i / n_z follow from quasi-neutrality. The width of the bottom (assumed) tier below: `None` = the Z_eff template (was `0.05`), a number = a flat fraction, `0.0` = the channel off |
| `zeff_sigma_source` | `"auto"` | Which tier supplies the Z_eff envelope's **magnitude**: `"auto"` / `"carbon"` / `"measured"` / `"scalar"` — see the ladder below |
| `ni_from_zeff` | `None` (auto) | With the Z_eff channel on: `True` derives n_i per draw from the drawn (n_e, Z_eff) as an increment on the baseline n_i; `False` draws n_i from its own σ and Z_eff passively after T_i (different RNG order). Auto: `True` (n_i always derived, as before PR #56). On a measured `ni_source` (the `ida_hybrid` default) auto is the PR #56 rule: `True` when n_i's σ is the scalar fallback, or n_i and the Z_eff envelope are the same IDA resolution (tier `"IDA-resolved"`); `False` for any other real n_i envelope. That rule follows the Z_eff tier: pin it for an A/B of Z_eff envelopes |
| `kinetic_clips` | `None` (auto) | The PR #56 sampler clips (graduated 2026-10-14, `kinetic_sampler_clips`) -- a drawn Z_eff floored at 1 where `physics.zeff_bounds` allows less, a derived n_i held in `[0, n_e − z_fast]`, an independent n_i capped at `n_e − z_fast` when `Z_imp` is declared, a passive Z_eff aux draw clipped to the window. Auto: on by default on every route (owner decision 2026-10-17; the Z_eff floor at 1 is a physical constraint, and without the clip a draw below 1 is refused before any solve). Every clip that fires is counted per draw and archived; on a clean plasma (Z_eff near 1) watch the clipped fraction, since the clip truncates the lower tail and the Z_eff, n_i and I_BS bands become one-sided. `False`: only the `physics.zeff_bounds` window; `True`: on. |
| `sigma_profiles` | `{}` | Explicit `{name: sigma(psi_N)}` envelopes on the kinetic run grid (`psi_N_kinetic`; Φ_N in a `"phi_n"` run), highest precedence |
| `n_ls` / `t_ls` / `j_ls` | `0.5` / `0.4` / `0.25` | GPR correlation lengths for density / temperature / current, in units of the run coordinate (Φ_N lengths in a `"phi_n"` run; the defaults are not converted) |
| `aux_sigmas`, `aux_baselines`, `aux_length_scales` | `{}` | The passive switchboard: any extra channel gets perturbed and archived alongside the physics. Arrays on the kinetic run grid; length scales in the run coordinate |

### Radial coordinate (`b.source.coord`)

`"psi_n"` (default) keeps every profile on normalised poloidal flux.
`"phi_n"` puts the whole run on normalised toroidal flux: the reader relabels
the source's nodes with their Φ_N (the IMAS `core_profiles` `grid.rho_tor_norm²`;
the g-file's own q-integral `rhovn²`, with the p-file/IDA nodes inside the LCFS
mapped through it), and every profile, envelope and GPR draw then stays on that
grid. TokaMaker receives the profiles as-is, tagged `phi_n`, and remaps them to
ψ each nonlinear step with the equilibrium's own q; bouquet samples readbacks at
the ψ_N the solver's map gives for each node. `"rho_tor"` is accepted as an
input spelling and runs as `"phi_n"` on ρ². A `"phi_n"` run needs an
OpenFUSIONToolkit with toroidal-flux profiles (`TokaMaker.get_torflux_map`) and
the internal bootstrap solve; both are checked in `prepare()`. The archive's
baseline group records the coordinate as the `profile_coord` attr. Fields and
datasets named `psi_N` / `psi_N_kinetic` keep that name but hold the run grid:
Φ_N in a `"phi_n"` run.

The self-consistent bootstrap loop (`jbs_self_consistent=True`) and the unified
engine run Φ_N as well.
- Every solve tags its profiles with the run coordinate.
- After each solve the run nodes are mapped to ψ_N on that equilibrium's own
  toroidal-flux map (`get_torflux_map`, inverse), and the Redl evaluation
  (`physics.evaluate_jbs(..., coord=)`), the loop's residual weights and the
  FSA current integrals work on those ψ_N. This is what OFT's Fortran bootstrap
  does: kinetic values at the mapped nodes, gradients numerical in ψ.
- The engine's structured basis lives on the run grid, so the closure
  coefficients describe the same profile on every pass and draw; its integrals
  are over each geometry's own ψ_N.
- The g-file engine labels the g-file's nodes with `rhovn²` and the kinetic
  nodes by their own map, as the legacy reconstruction does. Its inductive
  basis and q-row radius stay in ψ_N.

With IDA-hybrid kinetics (`kinetic_source="ida_hybrid"`) the IDA fits, their
sigmas and ω_tor are placed on the run nodes by their own Φ_N, integrated from
the IDA file's `q`, not by the dd's map; a `"phi_n"` run refuses an IDA file
without `q`. The g-file path does the same for an IDA `.cdf` (a p-file, which
carries no q, goes through the g-file's map; an IDA `.cdf` without q is
refused). An `UncertaintyConfig.ida_path` other than the source's IDA file is
placed by its own q when it has one. q95 stays in ψ_N.
Window-type helpers (`sampling.sigmoid_length_scale`,
`uncertainties.new_uncertainty_profiles`, `synthetic_ida_sigma`) take the grid
they are given: pass the run grid and their widths/positions are Φ_N in a
`"phi_n"` run. Fixed radial windows (edge > 0.9, pedestal 0.85) and the SWB
inductive seed shape are in ψ_N in either run. `source.coord` is checked when
the config is built; the toolkit is checked before the baseline and the draws. If the
solver's toroidal-flux map cannot be built (surfaces fail to trace), the solve
fails like any other and the draw is rejected.

**Precedence, per kinetic channel:** `sigma_profiles[chan]` > an IDA `.cdf` >
an `ImasSource`'s own core_profiles `*_error_upper` > the assumed default
(`<chan>_scalar_sigma` when set, else the H-mode flat-top template). A `.cdf`
handed to `ReconstructionSource.profiles_path`
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
`log_sigma_sources=False`), returns it warning-free as `"sigma_source"`
(`{chan: {"kind": "explicit" | "ida" | "imas" | "assumed", "source", "assumed_default"}}`,
Z_eff included, with its tier), and warns when a scalar you set is being
ignored.

**Every sigma-less run's widths moved on 2026-10-14.** The assumed default is
the H-mode flat-top template shape (the median relative envelope of IDA-lite
deliveries on DIII-D H-mode flat-top slices; 14 ψ_N nodes, 2 s.f.,
`bouquet.uncertainties.SIGMA_TEMPLATES`) instead of the flat scalars. As
uniform-in-ψ_N means (about the volume average):

| channel | before | now | at ψ_N 0 / 0.5 / 0.9 / 1.0 now |
|---|---|---|---|
| n_e | 5 % | 4.0 % | 5.7 / 2.8 / 5.8 / 15 % |
| T_e | 5 % | 8.4 % | 5.2 / 7.8 / 13 / 19 % |
| n_i | 10 % | 9.6 % | 17 / 8.1 / 9.7 / 22 % |
| T_i | 10 % | 9.7 % | 8.9 / 8.7 / 14 / 31 % |
| Z_eff | 5 % | 11 % | 16 / 8.7 / 12 / 41 % |

(The design note's 9.7 % / 9.8 % for n_i / T_i are the means of the unrounded
median; the shipped 2-s.f. nodes give 9.6 % / 9.7 %.) Which default applied is
stamped (`sigma_source[chan]["assumed_default"]`: `"template hmode_flattop"` or
`"scalar <x>"`). A stored config carries the old scalars explicitly and
reproduces what it ran; set them to get the flat envelopes back. `sigma_jphi` and the aux channels have no `.cdf` branch, so
their scalars always apply.

**The Z_eff envelope has its own ladder (`zeff_sigma_source`).** Z_eff is the
primary density channel — each draw perturbs it and *derives* n_i / n_z — so the
width of its envelope sets the width of every dilution band in the ensemble.
With `ni_source="standard"` (the `ReconstructionSource` default; on the IMAS
path the explicit dd route), `"auto"` takes the highest-fidelity tier the file
supports:

| Tier | Where the magnitude comes from | Forced by |
|---|---|---|
| carbon-propagated | `n_12C6_err` + `n_e_err` (direct layout) or the dilution posterior (ensemble), propagated through `Z_eff = 1 + Z(Z−1)·n_C/n_e` | `"carbon"` |
| VB-measured | the file's own `Zeff_err` (direct) or `Zeff` sample spread (ensemble) | `"measured"` |
| IMAS-measured | an `ImasSource`'s own `zeff_error_upper`, for the dd's own Z_eff baseline (no IDA tier chosen; not on `ida_hybrid`; skipped, warned, as "dataset rejected" when the stored `zeff` failed the consistency check) | — (`"auto"` / `"measured"`) |
| scalar | the Z_eff template × abs(Z_eff) (`zeff_scalar_sigma=None`, the default) or `zeff_scalar_sigma` × abs(Z_eff) — an **assumed** width, not a measured one | `"scalar"` |

(On `ida_hybrid` with `ni_source="standard"` the two IDA tiers apply to the
dd's total Z_eff, see below.)

CER carbon is the direct measurement of the dilution the draw actually moves,
which is why it outranks the visible-bremsstrahlung sigma. Both measured tiers
require the Z_eff baseline to measure the same total Z_eff of the discharge the
**same** file measured (compared as resolved paths, so a relative,
`~`-prefixed or symlinked spelling is still the same file): the reconstruction
path's own `.cdf`, and on `ida_hybrid` with `ni_source="standard"` the
`ImasSource.ida_path` too. That sub-route's baseline is the dd's **total**
Z_eff (owner decision 2026-10-18, as A1 on the plain dd route), so the file's
Z_eff envelope is the measured σ of the same quantity; the label then reads
e.g. "carbon-propagated (n_12C6_err) on a dd-derived total-Z_eff baseline" and
`zeff_sigma_tier["pairing"]` records the base definition, the draw centre, the
width source and that the draw is conditional at the dd's fixed beam fraction
(the assumed template stays the loud fallback for a file with no Z_eff
envelope). A plain dd baseline (FUSE kinetics) gets the dd's own
`zeff_error_upper` where the dd carries one (the synthetic fixtures do; a real
FUSE dd carries none), else the assumed tier; a p-file baseline gets the
assumed tier. A different `.cdf` (another vintage, another discharge) is
refused on every route. The definitions on every route are tabulated in
[physics-notes.md](physics-notes.md#z_eff-which-definition-feeds-what-on-every-route).

With a measured `ni_source` (the `ida_hybrid` default, `ida_ion_route`) the measured tier is
instead `read_ida`'s own resolution, **IDA-resolved**: VB+CER > CER > VB over
what the file supports, equal-weight combined, with the routes' disagreement
beyond their combined σ added one-sidedly (`"auto"` / `"measured"`; `"carbon"`
is then a single-route override), and `ImasSource.ida_path` on `ida_hybrid` is
eligible (also with `zeff_from_fuse=True`, the envelope then carried absolute
on the dd's total Z_eff, labelled "... on a dd-derived total-Z_eff baseline"
and recorded as its own `zeff_sigma_tier["pairing"]`).
The assumptions behind the VB/CER combination are stated and cited in
[physics-notes.md](physics-notes.md#kinetic-assumptions-z_eff-n_i-and-the-clips-pr-56).

**No step down this ladder is silent.** Each one emits a single warning naming
the tier chosen, each tier skipped and its reason class (*source ineligible* /
*missing dataset* / *invalid data*), and the same record is returned as
`resolve_uncertainty()`'s `"zeff_sigma_tier"`. A forced `"carbon"` or
`"measured"` that cannot be honoured falls back **loudly** rather than raising.
Non-physical carbon data (negative `n_12C6`, netCDF fill values, NaN holes) drops
the carbon tier with a counted reason in both layouts rather than riding through
as an enormous sigma.

### `GenerationConfig` (`b.generation`)

| Knob | Default | Meaning |
|---|---|---|
| `n_equils` | `20` | Draws to attempt — or, with `n_inspec_target` set, the initial allocation |
| `n_inspec_target` | `None` | Draw **until N draws are in spec** ([the in-spec rule](#the-in-spec-rule): by default the coil test and the boundary cut) rather than exactly `n_equils`. See [until-N](#until-n-in-spec-draws) below |
| `max_total_draws` | `None` | Attempt cap for `n_inspec_target` (default `5 ×` the target, never below `n_equils`). Hitting it is a loud, non-fatal outcome |
| `seed` | `None` | The run's one RNG seed. Consumed once into a single `numpy.random.Generator` threaded into every draw (GPR kinetic/aux/j_φ, the per-draw `scale_jbs`, the per-draw l_i target), so the same seed + inputs + solver gives a **bitwise-identical** archive on one machine (across machines the draws agree to ~1e-9, not bitwise -- LAPACK/BLAS). `None` = OS entropy |
| `scan_key` | `0` | Label for this bouquet within the archive (`scan/<key>/`) — a time in ms, a beta value, … |
| `l_i_tolerance` | `None` | **Opt-in** fractional l_i(3) band of the in-spec rule, `\|l_i - l_i*\| <= l_i_tolerance * l_i*` around the reconstruction's l_i (a **fraction**: `0.05` = ±5 %). `None` = no l_i band (**default since 2026-10-14**, owner decision; it was `0.05` and applied to every engine draw). The legacy routes also *construct* their draws with it (rejection sampling inside the draw, not a filter): unset, they use the old `0.05`. See [the in-spec rule](#the-in-spec-rule) |
| `l_i_sigma` | `None` | **Opt-in** l_i(3) band from a **stated measurement uncertainty** (absolute 1σ of an MSE / diamagnetic fit): `\|l_i - l_i*\| <= inspec_band_nsigma * l_i_sigma`. Exclusive with `l_i_tolerance`; unified engine only (refused on the legacy routes at `generate()`; `refilter_inspec` applies it to any archive afterwards) |
| `beta_p_target` / `beta_p_sigma` | `None` / `None` | **Opt-in** β_p band for a diamagnetic measurement (both or neither): `\|β_p - beta_p_target\| <= inspec_band_nsigma * beta_p_sigma`, β_p of the full-pressure frame (see [Λ](physics-notes.md#lambda--beta_p--l_i2-reported-never-filtered)). Unified engine only, as `l_i_sigma` |
| `inspec_band_nsigma` | `2.0` | Half-width of a σ band (`l_i_sigma`, `beta_p_sigma`) in σ |
| `constrain_sawteeth` | `False` | Gate draws on q0 ≥ 1 (unified engine: a post-hoc gate of the in-spec rule, `+q0` in its name; legacy routes: inside the draw construction) |
| `jphi_band_nsigma` / `core_gradient_ratio` | `None` / `None` | swb only: the core-current checks of the in-spec rule (`+jphi`, `+jgrad`), judged on each draw's solved `jphi_total` inside the pedestal top (`filtering.pedestal_top`: the baseline's current minimum between x = 0.5 and its edge peak): within `jphi_band_nsigma` × the baseline's `sigma_jphi` of the baseline's current, and the steepest d`jphi_total`/d√x at most `core_gradient_ratio` × the baseline's. They remove the draws whose single-solve sawtooth reset leaves an unrelaxed current step. Counted by until-N; `refilter_inspec` applies them to any archive |
| `recalculate_jbs` | `True` | Recompute the Sauter bootstrap per draw (vs. reusing the baseline's) |
| `single_profile_jphi` | `False` | Legacy path: perturb the TOTAL `jphi_total` as one profile (no inductive / bootstrap split; no per-draw Sauter call). `jphi_scalar_sigma` then applies to the total, a larger absolute perturbation -- re-tune it. Needs `jbs_self_consistent=False` (refused otherwise); the unified engine refuses it |
| `jbs_scale_range` | `(0.99, 1.01)` | Legacy draws: the per-draw multiplicative spread of the bootstrap (uniform in the range; default `None` -> `(0.99, 1.01)` on 2026-06-04). The baseline's multiplier (`bs_scale` / `s_bs(ψ)`) is applied after SWB on SWB draws and re-centres this range on loop draws -- see [physics-notes.md](physics-notes.md) ("The bootstrap multiplier") |
| `swb_seed` | `None` | IMAS path, every legacy SWB call (baseline split, draws, σ=0 check): `"source"` seeds SWB with the source's `jphi_ind` and holds its NBI + RF + other current fixed (`jphi_fixed`); `"generic"` the `(1 - s^1.5)^1.5` seed. `None` resolves at `prepare_baseline()` to `"source"` when the installed OFT's `solve_with_bootstrap` takes `jphi_fixed`, else `"generic"` -- never an error; an explicit `"source"` on a toolkit without `jphi_fixed` is refused at the first SWB call only. Stamped in `li_metrics["swb_seed"]` |
| `solve_method` | `None` | The internal solve method: `"legacy"` (**deprecated in 1.4.0, removed in 1.5, #90**; one `DeprecationWarning` per run), `"swb"` (**EXPERIMENTAL**, [`swb_solve_method`](#experimental-features-and-their-validation-status); `solve_with_bootstrap` is the baseline and every draw; IMAS sources) or `"engine"` (the unified engine). `None` derives it from `imas_baseline` / `reconstruction_engine`. Nothing is rewritten at construction; the effective fields are set where the run is resolved (`prepare_baseline()`, the draw method, the solver setup) and a later switch back restores the user's own fields. A contradicting pair (`imas_baseline="swb"` with another method, `reconstruction_engine="legacy"` with `"engine"`) is refused |
| `imas_baseline` | `"closure"` | IMAS path: `"swb"` (**EXPERIMENTAL**, [`swb_solve_method`](#experimental-features-and-their-validation-status)) selects the swb method (solve A at the setup coil reg, solve B with a strong reg toward A's coils is the baseline, every draw is solve B with resampled kinetics and a GPR redraw of the inductive seed; 20 negative redraws refuse the draw, `swb_jind_redraw_refused`) |
| `swb_coil_reg_weight` | `1e3` | swb only: weight of solve B's (and every draw's) coil reg toward solve A's coils |
| `swb_edge_taper_psi0` | `None` | swb only: taper `jphi_total` to zero from this ψ_N to the LCFS in every SWB solve (OFT `taper_edge_jphi`). **Opt-in** (owner decision D4; was 0.999): `None` is off and is sent as `taper_edge_jphi=False` where the toolkit has the option (a toolkit whose own default is taper-on keeps it on otherwise; such a taper with the setting off warns). The channel taper factor is measured, not assumed |
| `swb_taper_source` | `False` | swb only, needs `swb_edge_taper_psi0`: the read-in SWB currents (inductive seed, `jphi_fixed`, sawteeth input) are zeroed on the LCFS node, where the taper weight is 0 (the solve is unchanged), and any negative node left in the inductive seed is clipped to 0 (warned); each draw's seed redraw is clipped to 0 too (`swb_jind_clipped`) instead of redrawn. Lets a source seed negative at the LCFS be drawn. Stamped on `_baseline`: `swb_taper_source`, `swb_source_edge_zeroed`, `swb_source_clipped` (`swb_source_clip_min` when clipped) |
| `swb_ip_tol` | `5e-3` | swb only: the largest `|Ip/Ip_target - 1|` an swb solve (solve A, solve B, every draw) is accepted at; beyond it the solve raises. Validated in (0, 1); stamped on `_baseline` (`swb_ip_tol`, `swb_ip_rel_err`, `swb_ip_rel_err_solve_A`) and per draw (`swb_ip_rel_err`); a solve accepted above 1e-4 (the engine's acceptance) warns. Value pending owner decision E6 |
| `swb_saw_q` / `swb_saw_dq` / `swb_saw_tol` / `swb_saw_ramp` / `swb_saw_rule` | `None` / `0.03` / `1e-4` / `0.01` / `"local"` | swb only: the sawtooth q reset inside SWB (OFT `saw_q_s` ...; `None` = off). Set from these fields, never from `bootstrap_kwargs` |
| `jbs_delta_mode` | `False` | Opt-in differential bootstrap composition — see [physics-notes.md](physics-notes.md#differential-bootstrap-jbs_delta_mode) |
| `isolate_edge_jbs` | `None` | Legacy path only. `None` is resolved per engine at `prepare_baseline()`: **`False`** under `reconstruction_engine="legacy"` (both input types; the unified forward decomposition -- pure-ohmic `jphi_ind`, full bootstrap in `jphi_bs` -- closes exactly and yields better), `True` under `"unified"` (never read; the engine refuses `False`). Set `True` explicitly only for dedicated edge-spike studies (kept, with a warning). Recorded in the archive ([engine.md](engine.md)) |
| `jbs_baseline_mode` | `"diff"` | IMAS path: how the SWB bootstrap is reconciled with the source (`"diff"` / `"rescale"`) |
| `closure_channel` | `"bootstrap"` | IMAS `jbs_baseline_mode="ohmic"` path: which component absorbs the Ip closure. `"bootstrap"` / `"ohmic"` (deprecated bracket) rescale one component by a scalar; `"sawtooth_bootstrap"` adds a q0 pin where sawteeth justify it; `"structured"` replaces the scalars with minimal-norm radial multiplier **profiles** `s_ind(ψ)`, `s_bs(ψ)` — see [physics-notes.md](physics-notes.md#the-structured-closure-and-its-l_i-constraint) |
| `structured_preset` | `None` → **`"li_soft_onesided"` on the `"structured"` channel** | A **named one-switch configuration**, resolved at construction (and again at the closure's entry point, for a channel set afterwards). `None` = no preference: with `closure_channel="structured"` that now resolves to the validated preset `"li_soft_onesided"` — σ_bs `(0.50, 0.30, 0.15, 0.10)`, σ_ind,down `(0.10, 0.40, 0.40, 0.40)`, σ_ind,up `(0.10, 0.10, 0.10, 0.40)`, `structured_soft=True`, `structured_ip_sigma_frac=0.005`, σ_li `0.04` **only** when `structured_li_target` is set; the raw shipped fields it replaces are the configuration that study superseded. **`"none"` is the opt-out**: it declines the default and reproduces those raw fields exactly. Explicit settings win unless they equal the field's own default, which a dataclass cannot distinguish from unset (a `UserWarning` names every field the preset filled and says `BY DEFAULT` when it was not named); an unknown name is refused; it does **not** change `closure_channel`, which stays `"bootstrap"`. The default steps aside for an explicit `structured_basis`, and does not fill `structured_ip_sigma_frac` next to an explicit `structured_ip_sigma`. The σ values are **relative-unit priors from one device and one `j_ind` source**, not device constants — elsewhere, a starting point to check against the recorded closure-health flags. Recorded as `structured_preset` / `structured_preset_source` (`default` vs `explicit`) — see [physics-notes.md](physics-notes.md#recommended-configuration) |
| `structured_preset_in_force` / `structured_preset_source` / `structured_preset_fields` | (recorded) | Not settable (`init=False`): which preset is in force after resolution, how it got there (`"explicit"`, `"default"`, `"opt-out"`, `"default-declined-custom-basis"`, `"unset"`) and the fields it filled |
| `structured_basis` / `structured_weights` | `None` | `"structured"`: the multiplier basis (default: four peak-normalised Gaussians at ψ_N 0.15/0.45/0.75/0.95) and the trust weights that pick among the exactly-closing profiles (default: the physics prior, or the preset's ladder where the preset applies). Weights are **W = 1/σ²**; `utils.sigma_from_weights` is the bridge. A **prior**, not a tolerance |
| `structured_sigma_ind_up` | `None` | `"structured"`: the **up** side of a one-sided prior on the inductive multipliers, as K σ values. `None` = symmetric (byte-identical to runs made before the field existed). A tight up-side σ lets `s_ind` **fall** freely at that radius while resisting a **rise** — empirically motivated, with an edge-resistivity mechanism as support. Convex piecewise-quadratic objective, solved by sign iteration at **zero** extra GS solves; a cycling pattern is refused |
| `structured_li_target` | `None` | `"structured"`: the slice's reconstructed l_i, as a **plain input** — bouquet neither fetches it nor applies any cross-code definition offset. `None` = Ip is the only global constraint, exactly as before |
| `structured_li_sigma` | `None` | `"structured"` + `structured_soft`: 1σ on that l_i. Required there; ignored by the hard solver, which imposes the target exactly |
| `structured_li_kind` | `"li_1"` | Which normalisation the target is in — `"li_1"` (EFIT-like) or `"li_3"` (ITER) |
| `structured_ip_sigma` | `None` | `"structured"` + `structured_soft`: 1σ on Ip [A]. `None` = Ip imposed exactly (every other channel's behaviour). With a finite σ the closure delivers a **posterior** Ip slightly off the measurement by design; the post-closure round-trip check compares against that posterior (`structured_ip_posterior`) at its unchanged 0.05 % budget, and the distance from the measurement is recorded (`structured_ip_measured_residual_pct`, `structured_residual_sigma_Ip`) and flagged past 1σ — never refused |
| `structured_soft` | `False` | `"structured"`: solve the **posterior mode** (Ip and l_i as Gaussian measurements against the same trust prior, σ = W<sup>−1/2</sup>) instead of the hard KKT system. Still zero GS solves |
| `structured_li_tol` | `0.005` | `"structured"`: absolute l_i acceptance for the single post-solve correction — the l_i analogue of `q0_tol`, and it shares that one extra solve |
| `structured_li_max_corrector_steps` | `1` | `"structured"`: how many l_i corrector steps a slice may spend. The first inverts the parameter-free log-gain `utils.LI_GAIN_EXPONENT = 2` (l_i enters the achieved equilibrium twice, so `achieved ∝ row²`). `2` enables a **conditional** second step that reads the slice's own log-gain off its two measured `(row, achieved)` pairs — changes the cost, never `structured_li_tol` |
| `mse_data` | `None` | `"structured"`: measured MSE pitch angles as a **third measurement** — a dict of per-chord `R`, `Z`, `tgamma`, `sigma`, `weight` (fit weight, folded into `sigma_eff = sigma/sqrt(weight)`), `A1`..`A4`, optional `A5` + `Er`, or `er_corrected=True` when `tgamma` is already E_r-corrected upstream, and the required scalars `ip_sign` / `bt_sign` (±1: the directions of Ip and B<sub>t</sub> in the A-coefficients' right-handed (R, φ, Z) frame — the field orientation is **stated, never fitted**; a better-fitting orientation is flagged, not adopted) (schema in `bouquet/mse.py`). Adds `chi2_MSE = Σ ((tanγ_pred − tanγ_meas)/sigma_eff)²` to the structured objective. Forward model `(A1 B_Z + A5 E_R)/(A2 B_φ + A3 B_R + A4 B_Z)` — the standard coefficient form with E<sub>Z</sub> = 0 (A6 unused) and no denominator E<sub>R</sub> term (non-zero A7 with an applied E<sub>r</sub> is refused). **Without `Er` or `er_corrected=True`, E<sub>R</sub> is taken as 0**: in a rotating plasma that biases the fit systematically (to first order B<sub>Z</sub> is read as B<sub>Z</sub> + (A5/A1)E<sub>R</sub>, reshaping the fitted current profile); warned and recorded (`structured_mse_er_neglected`). tanγ is the field of the **re-solved** equilibrium, so it is linearised in the coefficients by forward differences on solved equilibria (one GS solve per free coefficient) and the closure re-solved; the linearisation residual, chi² before/after and the achieved objective are recorded per slice (`structured_mse_*` in `ip_closure`). With MSE on, the structured trust weights are an **absolute** σ⁻² prior (their overall scale now moves the answer; the uniform ladder is σ = 1, not "no prior"); the ladder in force is recorded as `structured_mse_prior_sigma_*`. Refused (not ignored) on any configuration that never runs the structured closure; `workflow='custom'` downgrades that to a WARN. `None` = the channel exactly as before |
| `structured_mse_required` | `False` | `"structured"`: **refuse** when `mse_data` is absent/unusable (fewer than `structured_mse_min_chords` usable chords, malformed, E_r double-counted) or the MSE-constrained closure cannot be delivered. `False`: no block → no term; an unusable block → no term, **warned and recorded** (`structured_mse_status`). Refused outright on any configuration that never runs the structured closure |
| `structured_mse_fd_step` / `structured_mse_steps` | `0.02` / `1` | `"structured"` + `mse_data`: the forward-difference step (coefficient units — a numerical-differentiation step, not a tolerance) and the number of re-solve steps (2 re-centres the linearisation on the first step's solve, same Jacobian). Cost only: `free coefficients + steps` GS solves per slice |
| `structured_mse_sigma_sys` / `structured_mse_min_chords` | `0.0` / `4` | `"structured"` + `mse_data`: an optional caller-stated systematic added in quadrature to every chord's `sigma_eff` (default 0: nothing inflated), and the fewest usable chords a block may carry. Every `structured_mse_*` knob is validated when the config is built (steps an integer ≥ 1, `fd_step` finite > 0, `sigma_sys` finite ≥ 0, `min_chords` an integer ≥ 1), and an unknown key in `mse_data` is refused; a dropped chord (weight ≤ 0, a non-finite value, a NaN E<sub>r</sub> inside a supplied E<sub>r</sub> profile, off the solver mesh) is recorded with its reason |
| `q0_gate` | `1.1` | `closure_channel="sawtooth_bootstrap"` (and the engine's `"q0"` row): the axis row is admitted only where the source's sawtooth model is active at the slice OR its own axis `|q0_dd|` is at/below this value; otherwise the slice falls back to `"bootstrap"` with a printed note (`q0_gate_basis` recorded) |
| `accept_anchor_inband` | `False` | Legacy draws (Fix B): when the reconstruction anchor's l_i is already in the band, accept the anchor and skip the scale search and the corrective iteration. Refused non-default under the unified engine |
| `kinetic_source` | `"fuse"` | IMAS path: `"ida_hybrid"` takes n_e/T_e/T_i/ω_tor -- and by default (`ImasSource.ni_source="auto"`) Z_eff and n_i -- from an IDA `.cdf` while keeping FUSE currents / equilibrium / p_fast: one thermal-kinetics source per run, with the dd's fast-ion pressure the one declared cross-source input and its beam density subtracted from the IDA n_i (`ni_subtract_fast` follows `include_pfast`). The dd's Z_eff is an explicit choice (`ni_source="standard"`, or `zeff_from_fuse=True`); an IDA file with no Z_eff information is refused. `from_imas(ida_path=…)` selects it automatically |
| `anchor_jtor_to_equilibrium` | `True` | IMAS path: anchor total jphi_total to `equilibrium.profiles_1d.j_tor` rather than `core_profiles.j_tor` |
| `anchor_pressure_to_equilibrium` | `False` | IMAS path: add the fixed `p_diff = equilibrium.pressure − p_reconstructed` offset |
| `imas_corrective_jphi` | `False` | Opt-in corrective jphi_total iteration on the IMAS baseline solve (still being validated) |
| `floor_jbs` | `False` | Clip negative bootstrap excursions; only needed with `isolate_edge_jbs=False` on sources that carry an inner negative lobe |
| `bootstrap_kwargs` | `{}` | Keyword options passed through to `solve_with_bootstrap` on the **legacy** paths (e.g. `{"iterations": 2}`; replaces `swb_iterations`). Checked when the config is built **and whenever the attribute is reassigned**: a key outside the explicit allow-list `config.BOOTSTRAP_KWARGS_ALLOWED` is refused (a typo is refused with or without the toolkit), a key the call sites already set is refused, and on the legacy path a key the **installed** toolkit lacks (the internal-solve options `use_python_solve`, `use_sauter_eps`, `diagnose_bs`, `djbs_tol`, `saw_relax`, `taper_edge_*` exist only on a toolkit with the internal Fortran bootstrap solve) is refused -- warned instead while a stored config is loaded, so an archive written on another build reloads. Under the self-consistent loop SWB runs only for `jbs_init="swb"` and the jBS-delta / `DIFF_BS` caches (a non-empty dict is warned). It reaches **every** SWB call (reconstruction / IMAS baseline, caches, σ=0 check, draws): a stored `swb_iterations=n` (which reached only the draws) loads as `{"iterations": n}` with a warning that the baseline SWB now runs with it too. With `reconstruction_engine="unified"` only the engine's own keys are read -- the edge taper (`taper_edge_jphi`, `taper_edge_psi0`, `taper_edge_shape`; **off** by default, 0.999, quintic; implemented by bouquet, so no toolkit capability is needed) and `use_sauter_eps=True` (an accepted no-op: the engine's ε is `eps_definition`; `False` is refused, pointing at `eps_definition="a_over_R"`); any other key is refused (and dropped with a warning from a stored unified config, which never ran it) |
| `eps_definition` | `"r_over_R_geo"` | The inverse aspect ratio of the Redl collisionalities **and** the major radius R in ν\*_e, ν\*_i (Sauter Eqs. 18b/18c), at every bouquet Redl evaluation (unified engine reconstruction and draws, legacy self-consistent loop, IMAS baseline loop, delta caches); engine-independent. `"r_over_R_geo"`: ε = (R_max − R_min)/(R_max + R_min), R_geo = (R_max + R_min)/2 in ν\* (Sauter 1999 / Redl 2021 "ε = r/R0"; OMFIT, FUSE). Opt-ins: `"half_width_over_fsa_R"`, (R_max − R_min)/(2⟨R⟩) with ⟨R⟩ in ν\*; `"a_over_R"`, ⟨a⟩/⟨R⟩ with ⟨R⟩ in ν\* (the pre-2026-10-09 evaluator, bit for bit). R_min/R_max/⟨R⟩ from `get_fsa` on every OFT build. Recorded in every loop record's `evaluate_jbs_version` and the `_baseline` attr `bootstrap_eps_json`. A stored config without the field replays with the default (warned); toolkit-internal `solve_with_bootstrap` calls do not read it: on `solve_method="swb"` and the frozen legacy route (`jbs_self_consistent=False`), whose bootstrap is the toolkit's, `bootstrap_eps_json` records the toolkit's own definition (`bootstrap: "toolkit solve_with_bootstrap"`; upstream OFT `<a>/<R>`; on the internal-solve toolkit what `bootstrap_kwargs["use_sauter_eps"]` selects, or "the toolkit's own default"), with the configured value as `eps_definition_configured`. See [physics-notes](physics-notes.md) |
| `bootstrap_convergence_override` | `False` | **EXPERIMENTAL** ([`bootstrap_convergence_override`](#experimental-features-and-their-validation-status)). The explicit opt-in for the `bootstrap_kwargs` keys that change a **convergence criterion** of the toolkit's bootstrap solve (`config.BOOTSTRAP_CONVERGENCE_KWARGS`: `djbs_tol`, the jphi_bs freeze threshold; `saw_relax`). Without it such a key is refused; with it the key is accepted with a warning, and the archive's `config_json` records the flag and the values. Legacy paths only (the unified engine refuses those keys). A stored config that carries such a key from before the flag existed loads with it on, warned |
| `draw_solve_maxits` / `draw_solve_retry_urf` / `draw_solve_loose_tol` | `"auto"` / `()` / `None` | The legacy and swb draws' GS iteration cap and its OPT-IN rescue. `"auto"` resolves at `prepare_baseline()` (recorded in `engine_resolved_defaults`): 100 for the legacy draws; with `reconstruction_engine="unified"` all three are refused (the engine draws use `engine_draw_solve_maxits`, see engine.md "The solve cap", and are never rescued). `None` keeps the solver's setup cap (800) for the legacy draws, which is also what a stored config without the field, or with `null`, replays; the swb draws always run with at least `SWB_DRAW_MAXITS` = 100 iterations (`max(100, draw_solve_maxits)`, so `None` gives 100 there). The cap and the rescue act on DRAW solves only: the cold baseline re-solve and the σ=0 reference solve before the draw loop keep the setup cap. Draw solves converge in ≤ ~25 iterations; one that does not sits in a limit cycle just above `nl_tol`. The rescue (off by default) re-solves a capped draw solve from where it stopped at each `draw_solve_retry_urf` (the same criterion), then, if `draw_solve_loose_tol` is set, accepts it at that LOOSER `nl_tol`. With the rescue on every stored draw is stamped `solve_recovered`, and a rescued one `solve_recovered_by`, `solve_nl_tol_accepted`, `solve_residual_upper` / `solve_residual_lower` (bounds: OFT reports no residual value), `solve_strict_nl_tol` and `solve_rescue_its`; `draw_band(rescued="exclude")` and `merge_archives(rescued="exclude")` leave rescued draws out (the default keeps them and warns with their count). Failed solves, and what recovered each, are listed per draw in `diagnostics['solve_failures']`, on `Bouquet.solve_failures`, and in one printed `[draw-solves]` line |
| `draw_wall_budget_s` | `600.0` | The per-draw WALL-CLOCK budget [s], both routes; `None` = off. Checked before each of a draw's solves (a running GS solve is never interrupted) and once before archiving; a draw over it is rejected as `draw_wall_budget` (recorded like every rejection; the reason it pre-empted kept as `masked_reason`), and the budget and its rejection count are stamped in the generation provenance. It ends stalled draws; the structural bound (every solve capped at `engine_draw_solve_maxits`, a bounded number of solves per draw) is separate. See engine.md "Cost" |
| `jbs_self_consistent` | `True` | Iterate the bootstrap to self-consistency with the delivered equilibrium (Redl on the caller's own ψ_N grid, re-evaluated after every solve; joint under-relaxation of the bootstrap and the solved current) in the baseline, every closure channel, the MSE stage, every draw and the reconstruction -- see [physics-notes.md](physics-notes.md#self-consistent-bootstrap-jbs_self_consistent). `False` = the **legacy** frozen-SWB bootstrap (legacy engine only; the unified engine refuses it); required with `single_profile_jphi=True` or `recalculate_jbs=False` (refused otherwise). Not by itself a pre-release reproduction: that needs `reconstruction_engine="legacy"` + `separatrix_pressure="legacy"` too, and the coil-solve mode and the current conversion still differ (CHANGES_SUMMARY, "Reproducing a run made before this release"). A stored config without the field (pre-loop archive) loads as `False` |
| `jbs_init` | `"anchor"` | The loop's initial guess: Redl on the anchor equilibrium, or `"swb"` (legacy result; A/B only). The fixed point does not depend on it |
| `jbs_rtol_j` / `jbs_rtol_Ip` | `1e-3` / `1e-4` | Loop convergence: current-weighted L2 residual of the jphi_bs profile, and its current integral over I_p |
| `jbs_tol_li` / `jbs_tol_q0` | `1e-3` / `2e-3` | Loop convergence: pass-to-pass change of the solved l_i, and of q0 where an axis row is active |
| `jbs_max_passes` / `jbs_max_passes_draw` / `jbs_max_passes_post_homotopy` | `12` / `12` / `6` | Pass ceilings (limits, not tolerances): baseline / reconstruction; each loop of a draw; passes after a draw's coil homotopy when its Redl check misses (6 since the approved change from 4: measured need 5, plus one pass). Convergence = every active criterion on **two consecutive** passes |
| `jbs_relax` / `jbs_relax_halve_on` | `0.7` / `3` | Under-relaxation ω of the bootstrap; halved (floor 0.25) only when `r_j` grows on `jbs_relax_halve_on` consecutive passes (`1` = on every growth); three growing passes at the floor abort |
| `jbs_relax_current` | `0.7` | Under-relaxation β of the SOLVED current (`(1−β)` previous solved + `β` closure), damping the closure ↔ geometry oscillation; path only, gap recorded per pass. `1` = off |
| `jbs_gate_current_residual` | `False` | Legacy loop, opt-in: ADD the closure-half current residual `‖jc_k − js_k−1‖_w/‖jc_k‖_w ≤ jbs_rtol_j` to the loop's criteria (same two consecutive passes; nothing relaxed; pass 1 cannot count). The unified engine applies it always (a standing criterion) |
| `jbs_loop_on_fail` | `"raise"` | Non-convergence: raise `JBSNotConverged` (with the residual history), or `"flag"` the slice closure-limited and keep the last iterate. A non-converged draw is always a failed draw |
| `jbs_loop_q0_corrector` | `False` | The q0 pin under the loop (`"sawtooth_bootstrap"`, or `"structured"` with the axis row admitted). `False`: record-only, with the axis row held and the q0 residual flagged against `q0_tol`. `True`: the axis row is moved once per pass from the measured q0 (`j_ref0 ← j0_solved·q0/q0_target`), and convergence also requires `|q0 − q0_target| ≤ q0_tol` (unchanged); failure raises or flags like the loop. The l_i row is not covered. See [physics-notes.md](physics-notes.md#the-q0-pin-under-the-loop-jbs_loop_q0_corrector) |
| `reconstruction_engine` | `"unified"` | **Default `"unified"` since 2026-10-06** (was `"legacy"`): ONE reconstruction loop for g-file and IDS inputs, which also runs the draws ([engine.md](engine.md)). `"legacy"` (**deprecated in 1.4.0, removed in 1.5, #90**: one `DeprecationWarning` per run, where the run is resolved) = the existing g-file reconstruction / IMAS baseline paths and the legacy draws, opt-in by name (`from_geqdsk` / `from_imas(..., reconstruction_engine="legacy")`). Legacy-path settings the engine never reads are refused under `"unified"`, naming the field and saying to set `"legacy"`. A stored config without the field (written before 2026-09-29) loads as `"legacy"` with a warning. `generate()` / `verify_sigma0_consistency()` under `"unified"` run the engine draws from the engine baseline `prepare_baseline()` built IN THIS SESSION (its live state); a baseline built by the other engine, or one loaded without that state, is refused (`Bouquet._refuse_unified_engine_draws`) |
| `engine_preset` | `"structured"` | `"unified"` only: `"structured"` (4 Gaussians, `li_soft_onesided` priors), `"structured_uniform"` (the same basis under the uniform σ = 1 ladder: the prior-sensitivity run), `"bootstrap_scalar"` (rows Ip), `"sawtooth_two_scalar"` (rows Ip, q0; falls back to `"bootstrap_scalar"` with a notice when the sawtooth gate rejects q0) or `"two_scalar_li"` (rows Ip, l_i: the legacy secant's two-scalar family). Refused when changed under `"legacy"` |
| `engine_rows` | `("Ip", "l_i")` | `"unified"` only: the measurement rows -- `"Ip"` (mandatory), `"l_i"` (g-file: hard at 1e-3; IDS: soft σ 0.04), `"q0"` (the source's own q at the row radius, only where the sawtooth gate admits it), `"mse"` (needs `mse_data` with `er_corrected=True`; raw E_r is refused) |
| `engine_delivery_correction` | `False` | `"unified"` only: add `request − achieved/c` of the previous pass to each request (one Newton step per pass; part of the state a draw inherits). Default to be decided after the solver's `jphi-linterp` defect is fixed |
| `engine_mse_jacobian` | `"fd_chord"` | `"unified"` + `"mse"`: the tan γ Jacobian by finite differences once at convergence, held fixed (`"fd_chord"`, the legacy chord stage's treatment; the default since 2026-10-02 -- the Broyden update's second pass mis-attributed the relaxing bootstrap and geometry to the coefficients and cost 4-5 extra passes) or with Broyden updates every pass (`"fd_broyden"`). Either way the Jacobian is re-taken by the same finite differences at the MSE loop's convergence and the loop continues with it while the refreshed Gauss–Newton step moves tan γ by more than 0.1 σ (since 2026-10-07; [engine.md](engine.md)) |
| `mse_chi2n_flag` | `10.0` | `"unified"` + `"mse"`: the delivered fit's chord χ²/N (raw E_r-corrected chords, the stage's own weights) above which it is FLAGGED `mse_chi2_per_chord_high` -- a flag only, never acceptance; the value is owner-approved 2026-10-07 (flag only). The delivered χ² above the no-MSE reconstruction's is flagged `mse_worse_than_without` whatever this is. Finite > 0; refused non-default under `"legacy"` |
| `engine_li_row_relaxation` | `1.0` | `"unified"` + `"l_i"`: under-relaxation `r` (0 < r <= 1) of the l_i row's discrepancy update between passes, `d_k = (1 - r w) d_k-1 + r w [measured - model]` (`w` the loop's omega; the first update takes `w = 1`). `1.0` is the update before the setting existed, bit for bit; `r < 1` multiplies the row's per-pass gain by `r` (a remedy for a row that overshoots in a period-2 oscillation). Path only: no tolerance, criterion, ceiling or target changes. Reconstruction only (draws carry no l_i row); the q0 row and the MSE chords are not affected |
| `engine_ids_inductive` | `"residual"` | `"unified"`, IDS sources only: the IDS adapter's inductive current. `"residual"` (default, owner decision 2026-10-02): the parallel residual `j_total − j_bootstrap − Σ driven` by definition; the source's `j_ohmic` is a stamped cross-check (no threshold, no warning); a source without `j_total` / `j_bootstrap` is refused. `"j_ohmic"`: the source's, warning when it misses the residual by more than the adapter's consistency tolerance (2 % of the total current). `"auto"`: `j_ohmic`, or the residual with a warning when `j_ohmic` is absent or misses by more than that tolerance. The consistency numbers are stamped either way. A non-default value is refused with a g-file source |
| `imas_li3_radius` | `"auto"` | `"unified"`, IDS sources only: the normalisation radius of the l_i row's target (the source's `li_3`; the IMAS data dictionary gives none, IMAS.jl / FUSE write it with the geometric radius `R_geo`, TokaMaker measures with `R_axis`). `"auto"`: recompute `li_3` from the source's own equilibrium with each radius and take the one reproducing the stored value within 0.5 % (neither: refused naming both; not recomputable: used unrescaled, recorded `"undetermined"`); `"geometric"` / `"axis"`: stated. The target is rescaled to the measurement's radius by `R_src/R_axis`; choice, ratios and factor recorded. A stored IMAS unified config without the field loads as `"axis"` with a warning; refused non-default with a g-file source |
| `engine_draw_q0_row` | `False` | `"unified"` draws: also keep the reconstruction's q0 row acting in every draw (`AxisRowPin` on the bootstrap amplitude) -- for sawtoothing discharges; needs `"q0"` in `engine_rows` |
| `engine_draw_homotopy` | `True` | `"unified"` draws: run the coil homotopy after a draw's loop (then the post-homotopy bootstrap check with its saturation guard); `False` measures the coil drift of the loop's own delivered draw |
| `engine_draw_bootstrap_refresh` | `False` | `"unified"` draws: after a draw's first loop solve, restart the bootstrap iterate from the anchor's Redl increment re-evaluated on that solved geometry. Path only (no criterion, tolerance or ceiling; zero extra solves) |
| `check_bistability` | `None` (on) | `"unified"` only. `None` resolves at `prepare_baseline()` (recorded in `engine_resolved_defaults`): **`True` under `"unified"`** (on by default since 2026-10-15, owner decision -- ONE extra GS solve per reconstruction), `False` under `"legacy"` (not applicable there: the default is never refused, an explicit `True` is). An explicit value, a stored config's included, is kept (an explicit `False` under `"unified"` prints a note). After the reconstruction, ONE extra solve of the same inputs from a COLD start (psi re-initialised as the reconstruction's own start) is compared with the delivered state -- l_i(3) and q0 at the q-row radius at the σ=0 gates (`jbs_tol_li`, `jbs_tol_q0`), β_pol and the magnetic axis reported. A miss stamps `Baseline.engine["bistability"]` with `bistable=True` and the deltas (archived with the baseline's engine record) and warns; the delivered equilibrium is restored and nothing is refused. See [Bistable equilibria](#the-in-spec-rule) and [engine.md](engine.md) |
| `edge_pprime_pin` | `True` | Every solve: `True` zeroes the last node of `P'` (ψ_N = 1), so `P'` ramps to zero across the final interval (the behaviour before the setting); `False` keeps the profile's own derivative there -- a physics change (edge current and q95 move; see [physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)). The solver's resulting uniform `P'` rescale is recorded as `p_scale` |
| `separatrix_pressure` | `"offset"` | Every solve: `"offset"` (default since 2026-10-02, owner-approved physics change) hands the solver `p_axis − p_sep` and adds `p_sep` back wherever pressure, β or W_MHD is reported or delivered (written g-files carry the full `PRES`); `"legacy"` hands the full axis pressure (a non-zero `p_sep` then inflates `P'` by `p_axis/(p_axis − p_sep)`). A negative `p_sep` is refused under `"offset"` |
| `coil_drift` | `0.01` | Soft coil-drift target |
| `coil_drift_hard_factor` | `None` | Optional hard inequality bounds at `± factor·coil_drift` in every solve |
| `homotopy_passes` | `[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]` | Progressive `(F_tol, VSC_tol)` schedule — see [coil-constraints.md](coil-constraints.md) |
| `iso_update_ref_points` | `9000` | Legacy draws' iso-update: the isoflux constraint on the perturbed LCFS has the strength of this many evenly spaced points at weight 200, spread per unit length of the traced LCFS, so it does not depend on `trace_surf`'s sampling (the field-line tracer gives ~9000 points, OFT's cut-cell trace 1000). `None`: weight 200 per traced point, as before |
| `workflow` | `"auto"` | Named preset; see below |
| `allow_unsafe_workflow` | `False` | Deprecated alias for `workflow="custom"` |
| `allow_incomplete_pressure` | `False` | IMAS path: bypass the fail-fast pressure-accounting check |
| `capture_live_eq` | `True` | Snapshot each draw's converged flux-surface averages into `eq_fsa/` — what makes `fidelity="exact"` IDS export possible |
| `capture_npsi` | `257` | FSA grid for that block |
| `capture_exact_inv_R2` | `True` | Record ⟨1/R²⟩ in the draw's `eq_fsa` block (read from `get_q`, else by flux-surface quadrature). Archived geometry only: since 2026-10-06 the current conversion is the one field-aligned factor `F⟨1/R⟩/⟨B²⟩`, which does not read it |
| `diagnostic_plots` | `False` | Per-draw diagnostic figures |
| `write_ifile` / `ifile_npsi` / `ifile_ntheta` | `False` / `129` / `257` | Also archive an OFT i-file (`TokaMaker.save_ifile`: ψ, F, p, q per flux surface and R, Z on the surfaces; GPEC `eq_type="ldp_i"`) per stored draw and for the baseline (`DrawView.ifile_bytes`, `extract(formats=("ifile",))`), traced from the same state and with the same `lcfs_pressure` (the separatrix pressure) and `lcfs_pad` as that state's g-file, so its pressure is the g-file's. Every route writes the baseline i-file (with `coil_drift=None` from the recon-converged state). Each group is stamped `ifile_written`, the grid, `ifile_lcfs_pressure` and `ifile_frame`: like every bouquet output it is in the positive-Ip frame (issue #68); the scan's `_baseline` `source_current_sign` / `source_b0_sign` map it back. Refused at `generate()` on a solver without `save_ifile`. A failed save rejects the draw (`ifile_save_failed`; the baseline's refuses the run), as does a failed state restore after it (`ifile_restore_failed`). Grid sizes are integers ≥ 2. About 0.5 MB per draw at the defaults |
| `ifile_profile_derivs` / `ifile_psi_bounds` | `False` / `False` | Append OFT `save_ifile`'s extended records to every i-file (readers of the standard format stop before them): FF′ and p′ after Z, which GPEC uses with `profile_source = "integrate"` (Julia GPEC `"derivatives"`) instead of differentiating the tabulated F and P, and the unpadded [ψ_axis, ψ_LCFS] last, to which GPEC and Julia GPEC then normalize ψ_N (the last surface, at 1 − `psi_pad`, sits below ψ_N = 1). Stamped `ifile_profile_derivs` / `ifile_psi_bounds`; on a `save_ifile` without them the baseline i-file fails and the run is refused |
| `store_eqdsk` | `True` | Archive each group's g-file. `False` (needs `write_ifile` and `store_tokamaker_state`): none is stored; the run still writes and uses it, and `DrawView.eqdsk_bytes` (so `equilibrium()`, `extract`, the stats, the IDS writer) rebuilds it from the TokaMaker state |
| `ifile_axis_pack` | `0.0` | OFT `save_ifile`'s axis zone: ψ_N of the innermost surface (0: none, today's file); the spacing grows outward until it joins the packed grid, and the zone's surfaces count in `ifile_npsi`. Stamped `ifile_axis_pack`; handed to `save_ifile` only when set (needs an OFT with `axis_pack`) |
| `store_tokamaker_state` | `False` | Also archive the TokaMaker equilibrium each i-file is written from (`save_TokaMaker`, ~0.9 MB per group on the D3D mesh) as the group's `tokamaker_state` blob, stamped `tokamaker_state_written`. Any i-file or g-file can then be rebuilt later at any grid (`DrawView.rebuild_ifile` / `rebuild_eqdsk`, on the run's mesh by default), so the archived i-files and g-files can be dropped once used (`bouquet.filtering.drop_file_blobs`); `DrawView.eqdsk_bytes` (and so `equilibrium()`, `extract`, the stats and the IDS writer) then rebuilds the g-file. Only with `write_ifile` (no i-file, no state); a failed state save fails the i-file |
| `eqdsk_nr` / `eqdsk_nz` | `257` / `257` | The archived g-files' R × Z grid (`TokaMaker.save_eqdsk` `nr` / `nz`), for the baseline and every draw; the 1-D profiles come on `nr` uniform ψ_N points. Integers ≥ 2. |

**Timing the IDA slice (`ImasSource.ida_time`, ida_hybrid only).** By default
the IDA slice is read at the dd slice `time`; `ida_time` reads it at another
time while `time` still picks the dd slices (equilibrium, core_profiles,
core_sources) — for example the IDA slice FUSE paired with a macro step when it
computed the dd's bootstrap. The IDA slice is the file's own slice nearest the
requested time, never interpolated, accepted within half the IDA file's local
time-step (a single-slice IDA file: half the dd step when paired with `time`,
the 10 µs floor for an explicit `ida_time`); paired with `time` it must also
sit within half the dd step of the core_profiles slice. Anything else is
refused. **EXPERIMENTAL** (see [Experimental features](#experimental-features-and-their-validation-status),
`ida_fuse_timing`): `ImasSource.ida_timing="ida_fuse"` takes the nearest IDA slice at any
distance instead, without the window acceptance, as IDA_fuse pairs its own dd and IDA slices (this applies to a
separate `uncertainty.ida_path` too). The match — requested and used IDA time, `dt`, the window and its
basis, the dd slice times, and the verdict of FUSE's `ida_provenance.json`
replay pairing when that table names this run's IDA file and holds the dd
slice — is archived in the baseline's `li_metrics["ida_time_match"]`.
On ida_hybrid, a dd step `time` outside the window the run simulated
(`time_grid.json`: t0 < t ≤ `reached`; a FUSE run that stopped early still
lists its planned steps in that table) raises `IdaTimeNotSimulated`: the IDA time
is not attempted, so `run()` / `run_slices()` do not archive it as refused.
With `ida_timing="ida_fuse"`, the pair must also be one of the run's consistent pairs
(`bq.consistent_pairs(ida_provenance.json)`: each IDA slice a reached macro step replayed, with
the nearest such step), else the same `IdaTimeNotSimulated`; the check is skipped (and stamped
in `li_metrics["ida_time_match"]["consistent_pair"]`) when the two json files are absent or
the table names another IDA file.
`ida_time` outside ida_hybrid is refused. `set_slice(time=t)` keeps a
configured `ida_time` (with a warning), `set_slice(ida_time=x)` applies it
alone, and a series takes one per slice: `run_slices(times, ida_times=[…])`.

### `FilterConfig` (`b.filtering`)

| Knob | Default | Meaning |
|---|---|---|
| `rms_max_mm` | `"auto"` | Boundary-RMS acceptance threshold [mm]. `"auto"` → the device's calibrated cut (DIII-D: **8.5 mm**, looser than the generic 5.0 mm), else the generic **5.0 mm**; a number is an explicit cut; `"off"` disables it (`None` = `"off"`, its historical meaning). `b.boundary_cut()` returns the resolved `(mm, source)`; the cut is printed once and stamped on the archive |
| `coil_filter` | `"chi2"` | `"chi2"` = measurement-referenced coil filter; `"legacy"` = the ±`inspec_*` band |
| `chi2_max`, `z_max` | `None` | `None` → the device's calibrated thresholds (DIII-D: χ²/ν ≤ 6.1, worst-coil \|z\| ≤ 6.3), else the generic 4 / 5 |
| `coil_sigma` | `None` | Per-coil σ override (`{"floor","fraction"}`, `{coil: σ}`, callable, or a named device model); `None` → the device model |
| `coil_daq_era` | `None` | Acquisition era whose σ **floor** applies (DIII-D: `"pre2014"` 825 A-t / `"modern"` 325 A-t). `None` → mapped from an explicit `source.pulse`/`source.shot`, else the device's default (tightest) band |
| `inspec_F_max` | `0.02` | Max non-VSC F-coil drift (fraction) for `in_spec` |
| `inspec_VSC_max` | `0.02` | Max VSC-channel drift (fraction) for `in_spec` |

> **The era is an acceptance criterion, so `filter()` states it out loud.** The
> pulse number is only a *date proxy* for the coil-current acquisition upgrade,
> so the band boundary is approximate. `Bouquet.filter()` prints one line per
> call naming the era, the floor it buys and the route it came from: an
> `[automatic: ... set filtering.coil_daq_era to override]` line for the pulse
> mapping, an `[explicit; filtering.coil_daq_era]` line when it was stated, and
> an `era undetermined (...) -> default band ..., the TIGHTEST floor` line when
> none could be resolved.

### `FixedComponentsConfig` (`b.fixed_components`)

**g-file + p-file: the p-file's `pb` is the default `p_fast`** (decision B,
2026-10-17). When the profiles file is an Osborne p-file carrying a `pb`
block and `p_fast` is unset, the reconstruction and every draw take
`p_fast = 1e3 * pb` (kPa → Pa) on `pb`'s own psinorm (coord `"psi_n"`),
stamped `li_metrics["p_fast_source"] = {"p_fast_source": "pfile_pb", ...}`
(and `Baseline.p_fast_source`; `Baseline.p_fast_meta` stays the IMAS route's
reduction record).
An explicit `p_fast` still wins (stamped `"fixed_components"`; a p-file with
`pb` beside it warns once per run, naming the field); a p-file without `pb`, and an
IDA `.cdf`, give zero fast pressure as before (`"none"`). Every g-file-route
reconstruction also carries the **report-only** `reconstruction_metrics
["pressure_fidelity"]`: the full pressure (the solver's, separatrix pressure
added back) against the input g-file's, as a relative RMS and the on-axis
relative error, with a `UserWarning` when the on-axis error exceeds 10 %
either way (the band that would have caught, with wide margin, the −63 % of a
dropped `pb`; an excess warns too, since a default `pb` can only add pressure:
a g-file whose pressure is thermal only, or a `pb` not in kPa). It is never
part of the verdict. The `pb` block's header unit is checked: anything but
kPa warns and is **not** converted (stamped `pb_header_units`,
`pb_units_ok`).

`p_fast`, `jphi_nbi`, `jphi_rf`, `jphi_other` on their own `psi_N` grid — additive
components that are never perturbed (an explicit `jphi_other` replaces every
fusion / runaways / sawteeth / unknown-index entry the IMAS reader would hold,
and zeroes `jphi_saw`). `coord` (default `"run"`) is the coordinate of that grid:
`"run"` (Φ_N in a `"phi_n"` run) or `"psi_n"`, mapped to the run coordinate
through the source equilibrium's ψ_N → Φ_N map. `jphi_nbi` / `jphi_rf` are given in bouquet's **positive-Ip
frame** — co-current drive positive — on both source paths and for either
orientation of the source; unlike the dd's own currents they are *not*
multiplied by `sign(ip)` on the IMAS path (see
[physics-notes](physics-notes.md#current-and-field-orientation)). `p_fast_reduction` (default `"auto"`) selects the
anisotropic fast-pressure reduction applied before the isotropic GS solve.

> **`p_fast_reduction` — a factor-of-3 convention, chosen from dd provenance.**
> `pressure_fast_parallel` / `pressure_fast_perpendicular` are written with two
> incompatible meanings and no dd field records which one is in use:
>
> | producer | what the two fields hold | scalar `p_fast` | rule |
> |---|---|---|---|
> | IMAS.jl / FUSE | the pressure **per degree of freedom** (`pressa/3` in each) | `p_par + 2·p_perp` | `"sum"` |
> | IMAS data dictionary, OMAS-written dds | the **full** directional pressures | `(p_par + 2·p_perp)/3` | `"trace"` |
>
> Getting it wrong is a clean 3× (or ⅓×) error in `p_fast`, and therefore in
> `beta_N`, `W_MHD` and `p'`. `"auto"` reads the dd's own recorded provenance, in
> this order: an explicit convention stamp in an `ids_properties.comment`
> (`... p_fast_reduction=trace ...`); then IMASdd.jl-only top-level keys
> (`global_time`, `requirements`, `build`, `balance_of_plant`, `solid_mechanics`,
> `costing`); then producer names in
> `{dataset_description,core_profiles,equilibrium,summary}` ×
> `{ids_properties.{comment,provider,source}, code.{name,description,repository}}`.
> If nothing identifies the producer it falls back to `"sum"` **and warns loudly,
> once**. An explicit `"sum"` / `"trace"` / `"mean"` / `"perp"` always wins and is
> silent. The rule used and the grounds for it are recorded on
> `Baseline.p_fast_meta`.
>
> `bouquet.physics.isotropize_fast_pressure(p_perp, p_par, method)` takes
> `method` as a **required** argument for the same reason — no default is safe
> for both conventions.

> **Tolerances are fractions, not percentages.** `l_i_tolerance=0.05` means
> 5%. This applies to every tolerance argument in the package (`l_i_sigma`
> and `beta_p_sigma` are absolute 1σ values, not fractions).

> **`p_thresh`** (the volume-averaged pressure acceptance band, default `0.05`,
> calibrated to a realistic `<P>` measurement uncertainty) is currently a
> `generate_bouquet` keyword only — it is not surfaced on `GenerationConfig`,
> so the class API always uses the default.

## Experimental features and their validation status

A feature listed here is reachable only by an explicit configuration value and
is not validated on real data, so it is never a default. The list is
`bouquet.experimental.REGISTRY` (the block below is generated from it by
`bouquet.experimental.docs_markdown()`, and a test holds the two in step).
When a run enables one:

- `Bouquet.prepare_baseline()` emits one `bouquet.experimental.ExperimentalFeatureWarning`
  per feature, naming it and its open validation items;
- the baseline record carries the list (`Baseline.experimental_features`) and the
  archive carries it as the `_baseline` attr `experimental_features_json` on every
  solve method (engine, legacy, swb), next to `engine_resolved_defaults_json`
  (`[]` when none; absent on an archive that predates the record;
  `utils.load_experimental_features`);
- `stats.draw_band` adds it to every record's provenance and limitations, and
  `BouquetArchive` / `ScanView` print it with their summaries.

`bouquet.experimental.experimental_features_enabled(config)` lists the keys a
configuration enables. The PR #56 dd fast-ion Z_eff term and the sampler clips
are here pending validation. The measured IDA Z_eff / n_i route and the beam
subtraction LEFT the registry in 2026-10 (owner decision): they are the
`ida_hybrid` default, with their open items kept as known limitations (next
section). (The geometric Redl ε default of PR #60 is a declared physics change,
not an experimental feature.)

<!-- experimental-registry:begin (generated by bouquet.experimental.docs_markdown) -->

#### `fuse_zeff_fast_ions` (experimental; introduced PR #56, 1.4.0)

On the FUSE (IMAS) path, classify the dd's stored Z_eff against the thermal-only and the thermal+fast numerators (io.imas._dd_zeff) and, where it counts the fast ions (or when no Z_eff is stored and the dd carries a beam), hand the bootstrap Z_eff_th + z2_fast/n_e with Baseline.zeff_includes_fast=True (a stored Z_eff classified thermal-only: the thermal-only Z_eff).  Default (owner decision A1, 2026-10-17): the dd's own TOTAL Z_eff -- the zeff dataset when it agrees with the all-ion sum over n_e from the dd's densities (within 5 %), else that sum -- with no classification (io.imas._dd_total_zeff), on the plain dd route and (owner decision 2026-10-18) the ida_hybrid ni_source='standard' sub-route.

Enabled by: `ImasSource.zeff_fast_ions=True`

Validation to do:

- on >= 5 real beam-heated dd's: the classification against the dd's own j_bootstrap (which numerator reproduces it)
- I_BS and l_i sensitivity to the fast-ion term on the same dd's

#### `swb_solve_method` (experimental; introduced PR #64 / #69 / #70, 1.4.0)

solve_with_bootstrap as the baseline and every draw on the IMAS path (bouquet.swb): solve A at the setup coil regularisation, solve B regularised toward A's coils, the draws solve B with resampled kinetics; I_p accepted within swb_ip_tol; the edge taper (swb_edge_taper_psi0) opt-in.

Enabled by: `GenerationConfig.solve_method='swb'`; `GenerationConfig.imas_baseline='swb'`

Validation to do:

- real-data arms on an OpenFUSIONToolkit build whose solve_with_bootstrap takes x / jphi_fixed / p_fixed, on dd's from a current IDA_fuse (so far: 6 of 8 IDA-hybrid reconstructions converged on the PR author's build; dd vintage unrecorded)
- swb_ip_tol statistics: the distribution of each solve's |I_p/I_p,target - 1| (swb_ip_rel_err) over real draws, against the 5e-3 acceptance
- taper off (swb_edge_taper_psi0=None) verified on real slices: the Picard 2-cycle the taper was added for does not occur, or is caught

#### `bootstrap_convergence_override` (experimental; introduced PR #60, 1.4.0)

Accept the bootstrap_kwargs keys that change a convergence criterion of the toolkit's internal bootstrap solve (BOOTSTRAP_CONVERGENCE_KWARGS: djbs_tol, saw_relax).  Legacy and swb paths only; the unified engine refuses it.

Enabled by: `GenerationConfig.bootstrap_convergence_override=True`

Validation to do:

- for each key, the converged jphi_bs, l_i and q0 against the toolkit's default criterion on >= 5 real slices; a value that changes a converged result beyond the default criterion's own residual is a loosened criterion and must be flagged
- the effective values stamped per run are read back by load_config and shown with the archive summary

#### `ida_fuse_timing` (experimental; introduced #79, 1.4.0)

Read the IDA slice (ida_hybrid's ImasSource.ida_path, or a separate uncertainty.ida_path) at the file's nearest own slice to the requested time AT ANY DISTANCE, as IDA_fuse pairs its own dd and IDA slices (io.imas.IDA_FUSE_TIME_RULE, the #73 timing before review).  It removes the IDA time-window acceptance: an IDA slice far from the dd slice is read without a refusal (the match, dt included, is still archived in li_metrics['ida_time_match']).  Default 'window': the nearest slice within half the IDA file's local time-step, else refused (io.imas.IDA_TIME_RULE).

Enabled by: `ImasSource.ida_timing='ida_fuse' with an ida_path or uncertainty.ida_path`

Validation to do:

- on >= 5 real IDA_fuse runs: the |dt| distribution of the slices it accepts beyond the window rule's half-step, and the kinetic profiles of each against the in-window slice
- I_BS, l_i(3), q0 and beta_N of a run timed by 'ida_fuse' against the same slice timed by 'window', where both accept it, and wherever 'window' refuses: show the accepted slice is the one FUSE's replay pairing used (ida_provenance.json)

<!-- experimental-registry:end -->

## Graduated features and their known limitations

A feature listed here was experimental and is now a default, on validation
evidence (owner decision); the open validation items of its registry record are
kept as KNOWN LIMITATIONS. It is not warned about; the route a run took is
stamped on the baseline record (`li_metrics["zeff_source"]`,
`li_metrics["kinetics_route"]`, `IDAProfiles.zeff_provenance`). The list is
`bouquet.experimental.GRADUATED` (the block below is generated from it by
`bouquet.experimental.graduated_markdown()`, and a test holds the two in step).

<!-- graduated-features:begin (generated by bouquet.experimental.graduated_markdown) -->

#### `ida_ion_route` (graduated 2026-10 (owner decision): default on ida_hybrid; left the experimental registry)

Z_eff and n_i from the IDA file's measured routes: the visible-bremsstrahlung Z_eff ('Zeff'), the CER carbon density ('CER'), or the clamped equal-weight mean of the two ('all'), with the route disagreement folded into sigma_ni / sigma_Zeff, the 'IDA-resolved' Z_eff envelope and the PR #56 auto rule of UncertaintyConfig.ni_from_zeff.  The DEFAULT on ida_hybrid (ImasSource.ni_source='auto' -> 'all'; ONE thermal-kinetics source per run); on a g-file + IDA run an explicit ReconstructionSource.ni_source.

Evidence:

- the sigma=0 draw reproduced the reconstruction on a reversed-Ip near-double-null IDA-hybrid slice with the measured route, where the dd-Z_eff route missed the archived-stage gates (before the engine draws held the reconstruction's isoflux)
- on real IDA-hybrid slices the dd's Z_eff differed from the IDA file's own by factors 0.78 and 3.9 (median over psi_N <= 0.9), with no signal of the disagreement on the dd route

Known limitations (open validation items, kept):

- the IDA-derived n_i has not been compared against the dd's total n_i on >= 5 real slices (agreement within the combined 1-sigma envelope, or each disagreement explained)
- Z_eff on axis: the VB+CER mean has not been compared against the CER-only and VB-only routes on the same slices with the route tension (zeff_route_chi) reported
- the I_BS, l_i(3), q0 and beta_N sensitivity to the route has not been measured on >= 5 real H-mode slices against the dd route and the measurement uncertainties
- old IDA vintages without Zeff_err ('all' falls to CER alone): not shown that the CER-only Z_eff is no worse than VB on those files

#### `kinetic_sampler_clips` (graduated 2026-10-14 (owner decision): on with the measured route, default included; left the experimental registry with it. 2026-10-17 (owner decision): on by default on every route -- the floor is physical, every clip is counted)

The PR #56 clips in the kinetic sampler: a drawn Z_eff floored at 1 even where physics.zeff_bounds allows less (thermal-numerator Z_eff with a beam), a derived n_i held in [0, n_e - z_fast], an independent n_i capped at n_e - z_fast when Z_imp is declared, and a passive Z_eff aux draw clipped to the same window.  Every clip that fires is counted per draw (KineticDraw.clips).  On by default on EVERY route (UncertaintyConfig.kinetic_clips=None, auto); an explicit bool wins.  The Z_eff floor at 1 is a physical constraint: without the clip a draw below 1 is refused before any solve (kinetics_nonphysical), which cost 7 of 10 attempts on the synthetic dd fixture (core Z_eff 1.7, template sigma 0.4).

Evidence:

- the clips change no draw unless they bind, and every clip that fires is counted per draw and archived
- graduated 2026-10 with the measured IDA route on the sigma=0 evidence of a reversed-Ip near-double-null slice; on every route since 2026-10-17 (a sigma=0 verification counts any clip that fires on the baseline and records it)

Known limitations (open validation items, kept):

- per-clip counts on a real ensemble (KineticDraw.clips): the fraction of draws and nodes each clip moves
- the bias each clip introduces in the drawn Z_eff, n_i and I_BS distributions against the unclipped draws, same seed

#### `ida_ni_beam_subtraction` (graduated 2026-10 (owner decision): default with the measured route and include_pfast=True (the #79 rule: the fast ions in p_fast with the subtraction, or both off))

ida_hybrid on a measured route: subtract the dd's fast-ion density equivalent from the IDA (total) n_i to give a thermal n_i (io.imas._subtract_fast_ni), so the fast ions are counted once -- in the dd's p_fast, the one declared cross-source input.  The IDA n_i and the dd's total n_i are compared at psi_N 0, 0.2, ..., 0.8 and a disagreement above 1 % warns.  The DEFAULT there (ImasSource.ni_subtract_fast=None follows include_pfast=True).

Evidence:

- the sigma=0 draw reproduced the reconstruction on a reversed-Ip near-double-null IDA-hybrid slice with the measured route and the subtraction

Known limitations (open validation items, kept):

- the IDA n_i has not been compared against the dd's total n_i on >= 5 real beam slices (the gate the subtraction prints); the subtracted density is trustworthy only where that disagreement is within the measurement uncertainty
- the thermal n_i floor at 0 (counted, ni_fast_meta.n_floored) is not shown not to bind on real slices

<!-- graduated-features:end -->

## The in-spec rule

**A draw is in spec when it passes the magnetics: the configured coil test
and the boundary cut** (and was not rejected before it was archived). Owner
decision, 2026-10-14. The external magnetics constrain I_p, the boundary and
the Shafranov combination β_p + l_i/2; bouquet applies that constraint exactly
through the free-boundary draw solve (the coils carry the boundary), the coil
χ² test (χ²/ν and the worst-coil |z| against the device's coil uncertainties)
and the boundary-deviation cut (DIII-D: 8.5 mm). l_i(3) on its own is an
equilibrium *inference*, not a magnetics measurement, so the l_i band that was
applied to every unified-engine draw until then (`l_i_tolerance = 0.05`)
conditioned every ensemble on information the magnetics do not provide -- and
it was the binding filter on the synthetic ensembles. It is now opt-in.

| filter | default | per-draw flag | applied by |
|---|---|---|---|
| coil χ² test (`filtering.coil_filter="chi2"`; the legacy ±% band on request) | **on** | `passes_coil_filter` | `filter()`; the until-N loop in flight |
| boundary-deviation cut (`filtering.rms_max_mm="auto"`) | **on** | `passes_boundary_filter` | `filter()`; the until-N loop in flight |
| l_i band (`l_i_tolerance` fraction, or `l_i_sigma` × `inspec_band_nsigma`) | off | `passes_l_i_band` | the engine's post-hoc stage; `refilter_inspec` |
| β_p band (`beta_p_target`, `beta_p_sigma` × `inspec_band_nsigma`) | off | `passes_beta_p_band` | the engine's post-hoc stage; `refilter_inspec` |
| q0 ≥ 1 gate (`constrain_sawteeth`) | off | `passes_q0_gate` | the engine's post-hoc stage; `refilter_inspec` (engine draws) |
| core current band (`jphi_band_nsigma`): the draw's `jphi_total` inside the pedestal top within this many of the baseline's `sigma_jphi` | off | `passes_jphi_band` (+ `jphi_band_z`) | the swb draws in flight; `refilter_inspec` |
| core gradient (`core_gradient_ratio`): the draw's steepest d`jphi_total`/d√x inside the pedestal top over the baseline's | off | `passes_core_gradient` (+ `core_gradient_ratio`) | the swb draws in flight; `refilter_inspec` |

`passes_draw_band` is the AND of the opt-in bands that are on (True when none
is: every engine draw carries it), and `selected` = coil AND boundary AND
`passes_draw_band`. A banded quantity that is missing or non-finite fails its
band; with the band off it is not judged. **Use an opt-in band only when the
source states the uncertainty** (an MSE-constrained fit's l_i error, a
diamagnetic loop's β_p error): the band then conditions the ensemble on a
measurement, not on the tolerance.

**Recorded.** The scan group carries `inspec_rule` (`"magnetics"`,
`"magnetics+li"`, `"magnetics+beta_p"`, `"magnetics+li+beta_p"`, with `"+q0"`
when the gate is on), `inspec_bands_json` (the band settings and the route;
on the legacy/swb routes also the draw-construction tolerance, which is not a
filter) and `lambda_definition`. The rule stamped is the one APPLIED: when
`constrain_sawteeth` is set but the reconstruction's q(0) < 1,
`generate_bouquet` switches the gate off for the scan, the stamp carries no
`"+q0"`, and `inspec_bands_json["q0_gate"]` (also the baseline engine record's
`draws.q0_gate`) records `configured`, `applied`, the q0 and the reason;
`filter()` and a re-filter under the stamped rule then do not apply the gate
either. Every draw carries `beta_p` and `Lambda`
(β_p + l_i(3)/2, [physics-notes](physics-notes.md#lambda--beta_p--l_i2-reported-never-filtered));
reported in `draw_scalars`, `ScanView.spread()` and `read_filter_flags`, never
a filter. The until-N target counts in-spec draws by the rule in force.
`merge_archives` refuses shards judged under different rules and records the
one rule in the merged scan and its manifest (`inspec_rule`, `inspec_bands`).

**Re-filtering.** `bq.refilter_inspec(path, bands=None, scan_key=None)`
re-judges the opt-in bands from the stored numbers (an engine draw's own
post-hoc record, else the archived `l_i(3)` / `beta_p`); no physics is re-run.
`Bouquet.filter()` does it automatically when the archive's stamped rule is
missing (an archive made before the rule, whose engine draws carry the old
unconditional 5 % band) or differs from the config's, and says so.

```python
bq.refilter_inspec("old_run.h5")                         # the magnetics rule
bq.refilter_inspec("run.h5", bq.InspecBands(l_i_sigma=0.03))   # a stated 1-sigma
```

**Stored configs.** A config stored before the rule (no
`generation.inspec_band_nsigma`) with `l_i_tolerance` at the old default
`0.05` loads with the band **off** and a one-line note; any other stored value
was explicit and keeps its band; a config without the field loads loud, band
off. A config written since keeps exactly what it says.

**What moves.** Yields rise wherever the l_i band was binding, and every
reported l_i band becomes a result of the ensemble rather than a reflection of
the 5 % tolerance. The coil and boundary thresholds are unchanged.

**Bistable equilibria.** On a near-double-null slice the free-boundary problem
can have two neighbouring solutions of the same inputs. On the one known case
(a reversed-Ip slice) their magnetic axes sit 17 mm (2.8 % of a) apart, and
l_i(3) differs by 2e-3 and q0 by 4e-3 -- an order of magnitude below the
ensemble's band widths. Three records say where a run stands; none refuses
or filters anything:

* the reconstruction's **bistability stamp** (`check_bistability`, on by
  default under the unified engine, one extra solve per reconstruction):
  `Baseline.engine["bistability"]` with `bistable=True`, the deltas and a
  warning when a cold re-solve of the same inputs lands outside the σ=0
  gates. The delivered equilibrium is kept.
* every engine draw's **axis record and displacement flag**: the archived
  state's magnetic axis against the reconstruction's delivered one
  (`engine["axis"]`: `R`, `Z`, `reference_R`, `reference_Z`, `dR_mm`,
  `dZ_mm`, `displacement_mm`, `displacement_frac_a`, `minor_radius_m`) and
  `axis_shift_large` = whether the displacement exceeds
  `engine.AXIS_BRANCH_FRAC` (1 %) of a. It is a DISPLACEMENT flag, not a
  branch label: the kinetic draws of an ordinary ensemble move the axis
  continuously by up to several % of a (in the validation runs: 0-28 of
  24-43 draws per synthetic case and 10-37 per real case beyond 1 % of a, all
  but one case one continuous spread), so a large shift of one draw says
  nothing about a second solution. The draw group carries `axis_shift_large`,
  `axis_shift_frac_a`, `axis_dR_mm`, `axis_dZ_mm` and `axis_minor_radius_m`;
  the σ=0 draw of `verify_sigma0_consistency()` carries the record too
  (`["axis"]`, not gated); the count is in the `[generate]` /
  `[draw summary]` line, in `ScanView.spread()["axis_shift"]`, in every
  `draw_scalars` record's provenance (`axis_shift`) and on the scan
  (`n_axis_shift_large`, `n_axis_recorded`).
* the per-scan **axis bimodality stamp** (`engine.axis_bimodality`): whether
  the archived draws' axis positions form TWO clusters -- the evidence of two
  equilibrium branches. Method (`engine.AXIS_BIMODAL_METHOD`, version
  `axis_bimodality/1`): sort the draws' axis Z (relative to the
  reconstruction's axis), split at the largest gap that leaves at least
  `AXIS_BIMODAL_MIN_COUNT` = 3 draws on each side, and call the scan bimodal
  when the gap exceeds `AXIS_BIMODAL_GAP_RATIO` = 1.0 x the larger of the two
  clusters' sample standard deviations AND the cluster centres sit more than
  `AXIS_BRANCH_FRAC` x a apart; fewer than `AXIS_BIMODAL_MIN_N` = 24 draws
  with an axis, or an unknown a: not testable (`None`, with the count and
  the reason; no verdict, no draws' source, no warning, no limitation). The scan carries `axis_bimodal` (when
  testable) and `axis_bimodal_json` (method, constants, both centres and
  counts, the gap, the spreads); it is printed on the `[generate]` /
  `[draw summary]` line and as an `[axis]` line with a `RuntimeWarning` when
  bimodal, reported in `ScanView.spread()["axis_bimodality"]` and in every
  `draw_scalars` record's provenance (`axis_bimodality`, over that
  statistic's used draws) with the limitation "the ensemble is bimodal in the
  axis position: two equilibrium branches; the bands are mixtures", and
  recounted by `merge_archives`. When a scan is bimodal and the
  reconstruction's own stamp said "single" (or the check did not run), the
  baseline is stamped `bistability_source="draws"` (`_baseline` attr and
  `Baseline.engine["bistability"]`, with the clusters as `draws_evidence`):
  the draws' own evidence meets the release rule "no bimodal band without a
  bistability stamp". A recount that no longer finds two clusters removes
  it; a reconstruction stamp "bistable" (`bistability_source=
  "reconstruction"`) is never overwritten. Why at least 24 draws (owner decision 2026-10-15),
  measured on random unimodal normal samples (a = 0.6 m): at n = 6-12 the
  rule calls a single Gaussian "bimodal" in about 20-37 % of samples when its
  spread is 6-20 mm (about 5-9 % at 3 mm); at n = 24 (36) about 10-11 %
  (3 %), a sub-threshold scatter never. Until 2026-10-15 the verdict was
  issued from n = 6. It is a report, and a false "bimodal" only adds a
  limitation and the draws' source. Archives written before the rename carry `axis_branch`
  (`"other"` / `"same"`) per draw and `n_axis_other` on the scan: both are
  read (`schema.read_axis_entry`, `schema.read_scan_axis_counts`; the
  displacement from the draw's engine record), and a re-stamp writes the
  current names.

The magnetics do not distinguish the branches on the known case: a σ=0 draw
that landed on the other branch passed the coil test (F 0.023 %, VSC 0.012 %)
and the boundary cut (0.50 mm). So both branches pass the in-spec rule, and a
band over both may be bimodal: that is the physics of the slice, not a defect
to filter away. The reconstruction's cold re-solve can miss a second branch
the draws find: on the same slice's no-MSE arm the stamp said "single" while
30 of 36 archived draws sat 6-24 mm below the delivered axis -- the case the
per-scan bimodality stamp exists for (applied after the fact it calls that
ensemble bimodal: 30 and 6 draws, centres 15.8 mm apart, gap 6.4 mm against
a 4.5 mm within-cluster spread; and none of thirteen ordinary ensembles of
the same validation runs). A continuity guard for the draws (keeping a draw on the
reconstruction's branch through its homotopy) is tracked as #87.

## until-N in-spec draws

By default a run draws exactly `n_equils` times and you get whatever yield the
equilibrium gives — 20 draws might leave 12 selected on a stiff case and 19 on
an easy one, so ensembles are not comparable in size across shots. Setting
`generation.n_inspec_target` inverts that: the loop keeps drawing until **N
draws are in spec** -- by [the in-spec rule](#the-in-spec-rule) in force: the
coil test and the boundary cut, plus any opt-in band -- then stops.

```python
b.generation.n_equils = 24            # initial allocation, not the total
b.generation.n_inspec_target = 24     # what you actually want delivered
b.generation.max_total_draws = 60     # optional; default is 5 x the target
b.filtering.rms_max_mm = 5.0          # the bounds the loop stops on
b.generate()
b.filter()                            # -> >= 20 selected
```

**How many.** 24 in-spec draws is the default of every example and launcher;
use 30 for 5/95 bands or band-against-band comparisons and 50 for widths
quoted in a paper.

Points worth knowing:

- **The stopping rule and the filter are one predicate.** The loop's per-draw
  verdict comes from `filtering.passes_all_filters`, composing the **configured**
  coil filter with `passes_boundary_spec` over the same `boundary_deviation_mm`
  metric and the same two archived LCFS contours the postprocess reads back.
  The coil half is built once per run by `filtering.make_coil_predicate` from
  the very `config.filtering` fields `filter()` cuts on:

  | `filtering.coil_filter` | in the loop | in `filter()` |
  |---|---|---|
  | `"chi2"` *(default)* | `passes_coil_chi2(chi2/nu, max\|z\|)` against the per-coil sigma from `coil_spec.resolve_coil_sigma(coil_sigma, device, coil_daq_era)` | `filter_coil_chi2`, same sigma, same era |
  | `"legacy"` | `passes_coil_spec` on the ±`inspec_F_max` / ±`inspec_VSC_max` band | `filter_coil_currents`, same band |

  The acceptance numbers come from `coil_spec.resolve_coil_acceptance` on both
  sides (`chi2_max` / `z_max`, else the device's calibrated quantiles, else the
  generic rule of thumb), and the DAQ era from the same `_coil_daq_era()` the
  filter uses — so the two cut at identical thresholds with identical sigma
  floors, and stopping at N and then filtering to fewer than N is not a state
  this can reach. If the per-coil sigma cannot be resolved (an unregistered
  mesh, an archive with no stored coil names) **both** sides fall back to the
  legacy rule, loudly and together. A draw with no coil currents is
  unjudgeable and **fails** in the loop, exactly as it fails in the filter.
  (Where the two *can* differ — a draw whose high-res LCFS trace failed — the
  loop calls it out of spec while the postprocess falls back to the coarse
  eqdsk contour, so the run over-delivers rather than under-delivers.)
- **What the per-draw log shows.** On the chi2 path each `[until-N]` line
  carries `chi2/nu`, `max|z|` and the worst coil (also stored in the returned
  diagnostics as `chi2_nu` / `max_abs_z` / `worst_coil` / `coil_nu`); on the
  legacy path, the F and VSC drift percentages.
- **Re-cutting afterwards is still your call.** The identity is against the
  thresholds in `config.filtering` at generation time. Passing a different
  bound to `filter(rms_max_mm=…)` later re-cuts the archive at the new
  criterion, and the selected count moves accordingly — that is the filters
  working as designed, not the loop having miscounted. The loop's LCFS bound
  is archived with the generation counts (`inspec_rms_max_mm` /
  `inspec_cut_source`), so `filter()` warns when it cuts at a different one,
  and a band's provenance carries both.
- **The boundary cut is announced once per run, on screen.** `generate()`
  resolves `filtering.rms_max_mm` before the solver output is captured and
  prints `[boundary cut] …` with its value and source (`explicit`,
  `device:<name>`, `generic` or `disabled`); an explicit `filter(rms_max_mm=…)`
  argument is announced too.
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
- **Both launchers honour it, through a shared ledger.** `parallel_generate`
  and the SLURM array pool their in-spec count (a `Manager` counter on the
  laptop, an append-only `{header}_inspec.ledger` file on the cluster); every
  worker checks the pooled count at the top of each attempt and stops once the
  run's ONE target is met. The stop is cooperative, at the attempt boundary,
  so a draw in flight is finished, never killed: up to one extra in-spec draw
  per worker can land after the threshold and is kept. `max_total_draws` is
  split across workers, so a zero-yield configuration still terminates. The
  merged archive carries a manifest (`parallel_manifest_json`) with each
  worker's attempt count -- the replay key, since *which* draws exist in a
  shared-stop run depends on worker timing even though each draw is bitwise
  reproducible. A bare `run_shard` without a ledger refuses the target.
- **The merged archive is filtered.** `parallel_generate(apply_filters=True)`
  (the default) and the SLURM merge job run the configured filters on the
  merged archive, exactly as a serial `run.filter()`; the parallel path used to
  leave only the in-loop legacy band on it.
- **The RNG stream is untouched when the feature is off.** The `jbs_scales`
  block draw is unchanged for the first `n_equils` draws and only extends past
  it when until-N actually runs on, so `n_inspec_target=None` runs are bitwise
  what they were.

## Workflow presets and the guard

These presets are the LEGACY paths' (`reconstruction_engine="legacy"`). The
factories set none of these flags: `isolate_edge_jbs` and
`perturb_jind_in_anchor` default to `None` and `prepare_baseline()` resolves
them for the engine configured when it runs, so the engine may be named at
construction or set afterwards with the same result (the unified engine
reads none of them and refuses them when changed). With
`reconstruction_engine="legacy"` each input type gets the flag combination
validated for its path, and `generate()` raises on a known-bad combination:

| Preset | Input | What it resolves to |
|---|---|---|
| `geqdsk-standard` | g-file | Standard flagship l_i loop (`perturb_jind_in_anchor=False`), unified decomposition (`isolate_edge_jbs=False`) |
| `imas-diff-c` | IDS | Bootstrap anchored to the source via the fixed diff (`jbs_baseline_mode="diff"`, the default), inductive perturbed in the recon-anchor (`perturb_jind_in_anchor=True`), unified decomposition |
| `auto` *(default)* | — | Resolve per source type at `generate()` |
| `custom` | you | Leave the flags as set and downgrade the guard to a warning. For deliberate backend experiments only |

Known-bad combinations the guard rejects: geqdsk + `perturb_jind_in_anchor`
(drops draws on stiff, high-l_i baselines via band-conditioning rejection),
IMAS without it (the matching loop homogenizes the draws), and
`jphi_scalar_sigma <= 0` (freezes `jphi_ind`, so the draws carry no
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

## Error bars from an archive

One recipe for every across-draw band (`bouquet.stats`): the population is the
draws the stamped filters mark `selected` (an unfiltered archive raises unless
`require_filter=False`), then per quantity status `"ok"`, `regular` and a finite
value -- one number per draw (a one-element array counts as one; an
array-valued quantity raises `NonScalarQuantityError` instead of vanishing, so
band a profile as one named quantity per point); the statistic is the median with p16/p84 (`np.percentile`,
`method="linear"`), min and max. The baseline is overlaid, never the centre.

```python
ar = bq.BouquetArchive("run.h5"); key = ar.scan_keys[0]
sc = bq.draw_scalars(ar, key)             # q0, q95, rho(q=2/1), rho(q=3/1), l_i, beta_N, <P>, beta_p, Lambda
def my_dprime(view):                      # user code; bouquet never sees the external code
    res = run_my_code(view.eqdsk_bytes)   # same accessor on draws and on the baseline
    return {"dp21": {"value": res.dp21, "status": res.status, "regular": res.has_q2}}
band = bq.draw_band(ar, key, my_dprime, evaluator_meta={"code": "...", "grid": "..."})
r = band["dp21"]; print(r.median, r.p16, r.p84, r.n_used, r.n_stored, r.n_requested, r.below_floor)
t = bq.draw_bands([(ar, k) for k in ar.scan_keys], my_dprime)
bq.plot_band(t, "dp21"); t.to_csv("dp21_bands.csv")
```

`evaluate` may also be a mapping `{draw: result}` (baseline under
`"_baseline"`) to ingest results computed elsewhere. Every record carries the
counts at each stage (`n_requested` … `n_used`), every dropped draw with its
reason (`not_selected`, `user:<reason>`, `status:<code>`, `irregular:<label>`,
`non_finite`), and a `provenance` block (filters, thresholds, versions,
limitations; `inspec_rule` / `inspec_bands` name [the in-spec rule](#the-in-spec-rule)
the population was selected under -- an archive that predates the record
gets a limitation saying its engine-draw selection includes the old 5 % l_i
band, and a rule with an opt-in band one saying that a band of the banded
quantity reflects the band's width). Below `min_n=15` a record is flagged `below_floor` (shown hollow);
below `hard_min=5` p16/p84 are NaN. If half or fewer of the ok draws are regular
the record is `gated` (`show=False`); there is no magnitude cut. Whenever any ok
draw is irregular the band is conditional on the regular outcome: the record
carries `n_irregular` and the irregular draws' values (`irregular_values`), a
limitation line says so, and `plot_band` annotates `k/n reg.`. `regular` must
be a real boolean (`"False"` or NaN raises `TypeError`). Fields an older
archive does not record come back `None` or `"unrecorded"`. A requested key or
quantity is never silently dropped: a key where no draw reaches the statistic
(for example, the filters rejected every draw) comes back as `status="empty"`
with `n_used=0`, no band and an `empty_reason` (`all_draws_rejected`,
`none_evaluated`, `no_finite_values`, ...); a refused slice as `"refused"`; a
missing archive or key as `"no_archive"`; a file that exists but cannot be
opened (locked, still being written, corrupt) as `"unreadable"`. For
`draw_scalars` q0/q95 the source of every used draw is in
`provenance["q_source_by_draw"]`; a statistic mixing `eq_fsa` and g-file
values, or a g-file baseline overlaid on `eq_fsa` draws (axis q0 vs
innermost-surface q0), is stated in the limitations and marks
`baseline_status` `"…:source_mismatch"`. `ScanView.spread()` remains the
quick-look mean/std summary.

**The standard summary in one call** (what every example notebook ends with):

```python
s = bq.bands_summary("run.h5", key, scalars=("beta_N", "l_i_3", "Lambda"),
                     gradients=("dp_dpsi", "djphi_dpsi"), at={"q": 2.0},
                     fallback_psi_N=0.5)
print(s.table())        # median, p16-p84, full range, n per quantity
s.plot()                # one band figure, n stated
s["dp_dpsi"].provenance["surface"]     # where each draw was evaluated
```

It is `draw_band` with one evaluator, so the population, floors, counts and
provenance are the recipe's. Scalars: `beta_N` (the draw's g-file),
`beta_p`, `Lambda`, `l_i_3`, `l_i_1`, `q0`, `q95`. Gradients: `dp_dpsi` and
`djphi_dpsi`, d/dψ_N of the draw's own archived `pressure` [Pa] and `jphi_total`
[A m⁻²] (central differences on the archived grid; a Φ_N-coordinate archive
is mapped to ψ_N through the draw's own q), evaluated on each draw at **its
own** `at` surface: the outermost q = 2 crossing of its archived q profile
(`eq_fsa/q`, else its g-file; the source per draw is recorded). A draw without
that surface is counted and excluded (`regular=False`, `no_q=2_surface`; the
majority gate applies). When no in-spec draw has one, every draw is evaluated
at `fallback_psi_N` and the summary and every gradient record say so
(`surface["mode"] = "fallback_psi_N"` and a limitation line);
`fallback_psi_N=None` returns the gradients `empty` instead. `at={"psi_N":
x}` evaluates every draw at a fixed ψ_N.

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

Exported g-files, profiles and archive currents are in bouquet's
positive-current frame (g-file `CURRENT > 0`, `BCENTR > 0`), whatever the
source's orientation; the source's own signs are on the archive
(`source_current_sign`, `source_b0_sign` on `_baseline`) for a consumer that
needs to restore them. The **IMAS export** (`export_ids`) is the exception: it
restores the source's orientation on every field it writes, so the exported
dd is self-consistent with the template fields it keeps (`core_sources`,
`pf_active`, `b0`) — see
[physics-notes](physics-notes.md#current-and-field-orientation).

The profiles JSON is source-agnostic and carries everything needed to rebuild
the state elsewhere: the perturbed profiles and their units, scalar diagnostics
(`l_i`, `I_p`, …), coil currents by name, and the captured flux-surface-averaged
geometry (`eq_fsa`).

### IDS current-split fidelity

bouquet's currents are TokaMaker `jphi`; the IDS `j_tor` (IMAS convention) and
the parallel split (`j_total` / `j_ohmic` / `j_bootstrap` = ⟨**j**·**B**⟩/B₀)
are converted with flux-surface geometry ([current-conventions.md](current-conventions.md)),
and `fidelity` picks where that geometry comes from:

| `fidelity` | Geometry | When |
|---|---|---|
| `"exact"` | an engine draw's stored `<j.B>` parts (`jdotB/`, no conversion); otherwise the draw's **own** captured `eq_fsa` geometry (needs `avg_R`, `avg_inv_R2`, `pprime`) | draws deviate from the baseline; the split must track each perturbed equilibrium |
| `"reconstruct"` | the template's baseline equilibrium (`gm1/gm5/gm8/gm9/f/dpressure_dpsi`) | exact only when a draw's flux geometry matches the baseline's |
| `"auto"` *(default)* | stored parts, else exact when a complete `eq_fsa` block is present, else reconstruct | — |

No exported parallel current carries the pressure-driven term
`P = p'(<R> - F^2<1/R>/<B^2>)` (its `<j.B>` is zero; a reader recovers it from
the pressure): `j_ohmic` is the field-aligned inductive only and `j_total =
j_ohmic + j_bootstrap + driven`. The archived toroidal `jphi_bs` carries `P`
(FUSE's convention); when the draw has no stored
`<j.B>` parts the exporter subtracts `P` (from the archived eqdsk's own flux
surfaces) before converting. Export -> `IdsAdapter.read` returns the archived
`<j.B>` parts and `<jphi_total>` (2026-10-06; before, `P/kappa` sat inside the
exported `j_ohmic` / `j_total` and a re-read counted it twice).

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

A slice whose `prepare_baseline` raises (a closure refusal, a failed gate, a
time-matching refusal) is written into the archive as **refused**
(`bq.write_refused_scan`, with the exception as `refused_reason` and the slice
time as `refused_time`), so `draw_bands` reports it as `status="refused"`
rather than a gap. By default (`on_refusal="record"`, since 2026-10-06) the
sweep records it in the summary (`refused=<reason>`, with its `time`) and
moves on to the next slice; after the last slice it prints and warns the
count and the reasons. `run_slices(..., on_refusal="raise")` re-raises at the
first refusal (the default before). `Bouquet.run()` records a
refusal the same way before re-raising. A later baseline or draw written into
the same scan supersedes the refusal (kept as `refused_reason_superseded`).
Parallel shards do not write refused records (a refused worker raises).

The parsed dd is **cached per file** (`bouquet.io.imas._load_dd`, keyed on the
real path, mtime, size and inode) and shared by the legacy reader, the unified
engine's IDS adapter, the geometry reader and the plotting readers, so a sweep
(or a `plot_jphi` loop over its scan keys) parses the file once. The cost is
memory: up to two parsed files, about twice the file size each, stay resident
for the life of the process — in a `parallel_generate` pool, in every worker
for its whole `generate()`. `bouquet.io.imas.clear_dd_cache()` releases them;
call it too after rewriting a dd in place with the same size within the
filesystem's mtime granule, or across hosts on NFS within its attribute-cache
window, where the stat key cannot see the change.

## Process-parallel generation

`OFT_env` is a per-process singleton, so parallelism is across **processes**, and every parallel path
runs one thread per process: TokaMaker at `nthreads=1` (bit-reproducible baselines, no OpenMP l_i
jitter, no DLSODE hangs) and every BLAS/OpenMP pool at 1 (`bouquet.threads`: the environment plus
`threadpoolctl`; `HDF5_USE_FILE_LOCKING=FALSE` too, since shared inputs are only read).  A config or
`threads_per_worker` asking for more is refused.

| | splits | use when | entry point |
|---|---|---|---|
| **draws** | one bouquet's draws across workers | one case, finished sooner | `bq.parallel_generate` |
| **cases, one node** | whole bouquets, one standing solver per worker | many cases | `bq.parallel_cases` |
| **cases, many jobs** | whole bouquets on a shared-filesystem queue, a fresh process each | many cases, many nodes or jobs | `bouquet.units` |

Never nest them.  One solver serves many baselines: `prepare_baseline` points it at each baseline's
own F0, boundary and isoflux (`Bouquet.point_solver`; under the unified engine the isoflux is each
baseline's own boundary resampled per unit length, #84), and every reset returns to that baseline's
geometry; the mesh, solve method and thread count are fixed per solver (`Bouquet.set_case` refuses a
change).

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

## Many bouquets on one node

One complete bouquet per case (g-file/p-file pair, IDA time slice, IMAS time).  Each worker stands up its
solver once, in its own directory (TokaMaker's scratch files) with its own copy of the mesh, and swaps
case after case onto it (`Bouquet.set_case`).  A failed case returns its traceback (`errors`,
`errors.pkl`) and the sweep goes on; the merge records it in its group's archive as a refused scan
(`refused_reason` names the case and its exception, so `draw_bands` reports `status="refused"`) and marks
the archive as a PARTIAL merge (root attrs `merge_partial`, `merge_expected_cases`,
`merge_failed_cases_json`, `merge_partial_json`).  An existing merged archive is not replaced unless
`overwrite=True`.  Each case runs at its own seed (`bouquet.parallel.case_seed`, from
the sweep's base seed and the case's group and scan_key), the one `units.units_from_cases` gives it too.
The base is the run seed; with the default `seed=None` one base is drawn from OS entropy when the sweep
starts (`units.sweep_seed_base`), so two default-seed sweeps differ.  Each case record, unit record and
merged archive carries `seed_base` and `seed_rule` (`"config"` or `"entropy"`; root attrs of the
archive, plus `merge_case_seeds_json` per case).  Input
paths may be relative to the launch directory.  A sweep is one run-level config plus the cases:

```python
src = bq.ImasTimeslices(header="sweep", ids_path="dd_sim.json", times=[3.163, 3.263, 3.363],
                        ida_path="IDA_synthetic.cdf", LCFS_geqdsk="g_synthetic.geqdsk")  # or one g-file per time
src = bq.IdaTimeslices(header="sweep", inputs=[("IDA_synthetic.cdf", ["g_synthetic.geqdsk", ...])])
src = bq.GeqdskProfilePairs(header="sweep", pairs=[(geqdsk, pfile), ...])

if __name__ == "__main__":           # required: workers are spawned
    summary = bq.parallel_cases(src, cfg, "work_dir")
```

A hand-built `list[CaseSpec]` works as well.  swb takes IDS sources only (`ImasTimeslices`); the engine
and legacy take all three.  Cases are keyed `scan_key` = time in ms (or the g-file stem) and grouped by
input (one `.cdf`, one IDS): the merge writes one archive per group, its cases under `scan/<key>/`, the
layout `bq.plot_bouquet_timeseries` reads.

```
work_dir/
  worker_0/ ...            per-worker cwd and mesh copy
  worker_0.log ...         per-worker output (fd level, so OFT's too)
  cases/sweep_<stem>_idx0.h5 ...   one archive per case
  sweep_<group>.h5         merged
  map_object.pkl           idx -> CaseSpec
  errors.pkl               {idx: traceback} (also recorded in the merged archives)
```

`group_by=None` merges into one archive, `merge=False` keeps only the per-case files, `cleanup=True`
deletes them once merged.  The worker count follows the affinity mask and the scheduler's CPU count
(`SLURM_CPUS_PER_TASK`, `PBS_NUM_PPN`, `LSB_DJOB_NUMPROC`, `NSLOTS`); `use_logical_cpus=False` puts one
worker per physical core.  A worker that fails to start (a bad mesh, a missing OFT) fails the run with
its traceback.  A worker that dies mid-sweep (a crash, an out-of-memory kill) ends the sweep: the finished
cases are kept and merged, the unfinished ones listed in `errors`.  Each case record carries its F0 and
the thread count of every loaded pool.

Per-case setup a `CaseSpec` cannot express goes in two hooks, `f(bouquet, case)`, run on the worker:
`before_baseline` (after `set_case`; e.g. solver targets from this slice's g-file) and `after_baseline`
(after `prepare_baseline`; anything on this baseline's grid, e.g. `uncertainty.aux_baselines`).  Workers
are spawned, so a hook must be a module-level function; a lambda or closure is refused before the pool
starts.  Each worker keeps up to two parsed dd files cached (#72); for a sweep over many dd files,
`after_baseline=bouquet.parallel.release_dd_cache` releases them after each baseline (a slice sweep over
one dd would then re-parse it per case).  `bouquet.units` releases them in every unit.

## Many bouquets on a cluster

`parallel_generate` splits the draws of one bouquet. To run many bouquets (one per baseline: time
slices, shots, variants) across processes, nodes and jobs, `bouquet.units` puts them on a
shared-filesystem queue (`bouquet.workqueue`; `mkdir` claims, heartbeat leases, correct on NFSv4 and
Lustre). Each unit runs in a fresh single-threaded process, in its worker's directory, on its worker's
copy of the mesh.  `units.units_from_cases(cfg, src)` queues the same cases `parallel_cases` takes.

```python
from bouquet import units
us = units.make_units(cfg, [{"source.time": t, "source.ida_time": None, "generation.scan_key": round(1e3 * t)}
                            for t in times], ids=[f"{shot}_{round(1e3 * t)}" for t in times])
units.add_units("queue", us)            # idempotent: ids already queued are skipped; add more any time
```

```bash
python -m bouquet.units work --queue queue --out-dir out --workers 64   # one per node: srun --ntasks-per-node=1
python -m bouquet.units status --queue queue                            # counts, failures grouped by signature
python -m bouquet.units stop|drain|undrain --queue queue
python -m bouquet.units retry-failed --queue queue --out-dir out       # reopens failed units, removes their records
```

- **Outputs.** `out/<id>.h5` (moved into place when complete), then `out/<id>.json`, the unit record
  (`status` `done`/`failed`, `selected_counts`, `profile_coord`, `seed`, ...). Readers wait for the record:
  a unit that raised gets it at once, a unit the queue failed (a crash or preemption on every attempt)
  when `work` ends. Relative paths are fine: inputs are stored absolute, queue and output resolve from
  where `work` starts.
- **Seeds.** Each unit's `generation.seed` is derived from the config seed and the unit id (from the case,
  for `units_from_cases`): reproducible, independent of which worker runs it, and different across units.
- **Failures.** A unit that raises is retried once more (`MAX_FAILURES=2`), then gets a `failed` record;
  `MAX_ATTEMPTS=4` bounds crashes and preemptions. A worker stops after 3 failures in a row; when workers
  of 2 jobs break within an hour the queue writes `STOP`. Leases of dead workers (OOM, node loss,
  preemption) expire after 300 s and are taken over. Policy overrides: `queue/config.json`.
- **Jobs.** Single-node or multi-node, or both at once: every launcher claims from the same queue, and
  units added mid-run are picked up. On Perlmutter prefer multi-node `regular` (or `preempt`) jobs sized
  to the backlog; set `--workers` to min(cores, memory / peak RSS per unit) and `--margin`/`--min-left`
  so no unit starts that cannot finish before the walltime.

## Running bouquet on another device

bouquet's defaults were measured on one device (the registered entry in
`bouquet/devices.py`) and on synthetic fixtures built on its geometry. The
physics carries over. The numbers listed below do not: each is either a
property of that device or a value measured on it. This section explains
what each number means, how it is set, and how to re-measure it. A helper
that turns a calibration campaign into a registry entry is planned for 1.5
(#86); until then, the steps below are manual.

### What is device-specific

1. **The mesh and the coil set.** `SolverConfig.mesh_path` is the device's
   TokaMaker mesh (coils, vessel, limiter). Its coil names select the device
   entry (`devices.detect_device`, an exact match of the coil set); an
   unrecognised set gets the generic path below. `SolverConfig.coil_vsc`
   names the vertical-control pair and its anti-series signs. Its default is
   the registered device's pair, so on any other device you must set it to
   your own pair, or the vertical-control channel acts on the wrong coils.
2. **The coil bounds and the 10 kA VSC floor.** The draw homotopy tightens
   the coil bounds in stages: `GenerationConfig.homotopy_passes`, default
   (F, VSC) fractions (0.05, 0.10), (0.02, 0.05), (0.01, 0.01).
   - Each F coil's bound is max(fraction × |I_base|, 50 A).
   - The vertical-control channel's bound is the VSC fraction ×
     max(min(|I_A|, |I_B|), 10 kA).

   Without the floor, a pair with one nearly idle coil would get a channel
   bound of a few hundred A, too tight to hold the plasma's vertical
   position. The axis then runs away and the draws are rejected; on a
   validated reversed-I_p case the floor raised the archived yield from 1
   to 9 of 10 attempts. The 10 kA is the same constant as the legacy in-spec
   VSC metric's offset. It is tokamak-scale, and it is a module constant
   (`TokaMaker_interface._COIL_DRIFT_DENOM_FLOOR_A`), not a configuration
   field.
   - If your vertical-control coils carry far less than 10 kA, the floor
     dominates and the channel bound is looser than the fraction suggests.
   - If they carry far more, the floor never binds.

   Either way, watch the `vertical_runaway` labels
   (`Bouquet.engine_draw_cap_events`) and the `homotopy_*` rejections; a
   device that needs a different floor is a case for #86 / #93.
3. **The in-spec coil test (χ²/ν and worst |z|).** The registered device has
   two calibrated parts:
   - a per-coil current-uncertainty model (floor + fraction, per-coil
     overrides, acquisition-era floors);
   - an acceptance of χ²/ν ≤ 6.1 and |z| ≤ 6.3, the 95 % quantiles over real
     fit residuals with 18 degrees of freedom.

   Another device gets the generic path:
   - With `FilterConfig.coil_sigma` supplied (a `{"floor", "fraction"}`
     dict, a per-coil table or a callable), the χ² test runs with the
     uncalibrated `devices.GENERIC_ACCEPTANCE` (χ²/ν 4.0, |z| 5.0), stamped
     `generic`.
   - With nothing supplied, it falls back to the legacy ± % band with a
     warning.

   To calibrate the acceptance:
   1. Build the sigma model from your coil-current measurement uncertainty
      (power-supply or shunt specification).
   2. Reconstruct a set of your device's discharges and compute each fit's
      χ²/ν and worst |z| against that model.
   3. Take the 95 % quantiles, and record the sample size and the degrees of
      freedom.
   4. Add a `DeviceSpec` entry with the provenance of each number.

   For a design with no measured residuals, take the sigma from the design
   specification and declare the generic acceptance.
4. **The boundary cut.** The registered device's cut is 8.5 mm: the 90th
   percentile of the LCFS RMS displacement under its magnetics' stated noise,
   from a boundary-UQ study. Every other device gets the generic 5.0 mm
   (`devices.GENERIC_BOUNDARY_RMS_MM`), and `FilterConfig.rms_max_mm`
   overrides either. Measure your own cut the same way from your magnetics'
   noise, or declare the generic one as uncalibrated.
5. **The boundary-constraint total (2e5) and its window.** The isoflux set
   is 256 points, resampled by arc length and weighted per unit length.
   Their weights sum to `solver_geometry.ISOFLUX_WEIGHT_TOTAL` = 2e5, the
   same for the reconstruction and every draw. What was measured on six
   equilibria (three experimental, three synthetic):
   - the window in which the σ=0 reproduction and the draw yield hold:
     about 1.5e5 to 4e5;
   - the band in which a boundary miss at its cut and a coil drift at its
     cut cost the solver equally (the vertical-control pair excluded):
     1.8e5 to 3.8e5.

   Re-measure it on a few representative slices of your device with the
   two probes 1.4.0 ships:
   - **The partition stamp.** After every reconstruction, one capped solve
     (pressure × 1.02) measures the partition: boundary RMS over its cut,
     divided by coil drift over its cut. It is stamped in
     `Baseline.engine["boundary_balance"]["partition"]` and in the solver
     geometry record (`isoflux_balance`), and warned outside [1/3, 3]. On
     the validated cases it read 0.95–1.38 (synthetic), 0.97–1.10 and 2.56
     (real), with one synthetic slice at 3.2–3.3. If your slices sit well
     inside the band, keep the constant. If they sit persistently outside
     it, scan `SolverConfig.isoflux_weight_total` (for example 1e5, 2e5,
     4e5) and find where the partition crosses 1.
   - **The fixed-seed yield test.** At each candidate total, run 10
     fixed-seed draw attempts. At least 5 must reach a solve, and the
     archived count must be at least 0.8 × the solved count (the template
     is `tests/test_boundary_balance_solver.py`,
     `test_the_default_total_keeps_the_draw_yield`). Also check that
     `Bouquet.verify_sigma0_consistency()` passes.

   Choose a total where both probes hold, and set it with
   `SolverConfig.isoflux_weight_total`; it is stamped as mode `"override"`.
   Please report the measured window on #93, which collects windows for a
   device-independent rule.
6. **The sigma templates.** When no measured sigma exists (no IDA file, no
   IMAS error field), each kinetic channel's envelope is the H-mode
   flat-top radial template times |baseline|. The templates
   (`bouquet.SIGMA_TEMPLATES`: `hmode_flattop`, `ramp`, `lmode`; 14 ψ_N
   nodes, read with `bouquet.sigma_template`) are the median relative
   envelopes of one device's IDA-style profile fits. To use your device's
   own envelopes, in order of preference:
   - an IDA-style profile file with your fits' uncertainties
     (`UncertaintyConfig.ida_path`, or the source's own IDA file);
   - absolute σ profiles on the kinetic grid
     (`UncertaintyConfig.sigma_profiles`, keys `ne`, `te`, `ni`, `ti`);
   - a flat fraction (`UncertaintyConfig.ne_scalar_sigma`, ...,
     `zeff_scalar_sigma`).

   To build device templates, take the median of σ/|X| over a set of your
   fits per regime on the same 14-node grid, as the shipped templates were
   built. Each run logs and stamps which source won for each channel
   (`log_sigma_sources`).
7. **The Z_eff definitions.** The rule is the same on every route: the
   bootstrap coefficients take the total Z_eff; the species derivations
   take the thermal numerator; each draw is centred on the base and
   perturbs it with the measured absolute width. Its inputs are
   device-specific:
   - the impurity charge (`impurity_Z` on the source, default 6, carbon);
   - the measured Z_eff tiers (carbon CER through n_C, visible
     bremsstrahlung), which assume those diagnostics;
   - the beam's charge moments, which come from the dd on the IMAS routes
     and are not yet carried on the g-file + p-file route (#97).

   Set `impurity_Z` to your dominant impurity's charge, and read the route
   table ([physics-notes.md](physics-notes.md#z_eff-which-definition-feeds-what-on-every-route))
   for what each route takes.

### Checklist

1. A mesh with your coil set, `coil_vsc` set to your pair, and `impurity_Z`
   set.
2. A coil sigma model (`coil_sigma`). Either calibrate the acceptance (95 %
   quantiles over your fits) and register a `DeviceSpec`, or accept the
   generic acceptance, stamped as such.
3. A boundary cut measured from your magnetics' noise, or the generic 5 mm
   declared.
4. The partition stamp and the fixed-seed yield test on a few slices at the
   default total. Override the total only if they say so, and report the
   window on #93.
5. Measured kinetic σ (an IDA-style file or `sigma_profiles`) rather than
   another device's templates.
6. Before quoting a band: σ=0 passes, the in-spec count is at least 24
   (until-N), and the stamps (partition, clips, `vertical_runaway`,
   bistability, bimodality) have been read.
