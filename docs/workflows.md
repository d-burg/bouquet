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
- [Error bars from an archive](#error-bars-from-an-archive)
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
| Z_eff | ✓ | Active channel (default on via `zeff_scalar_sigma`) |
| n_i, n_z (impurity) | derived | Quasi-neutrality with the drawn Z_eff (`impurity_Z`) |
| Total pressure (p_tot) | ✓ | Recomputed from the perturbed kinetics |
| Bootstrap current (j_BS) | ✓ | Sauter model recomputed per draw (`recalculate_j_BS`) |
| Inductive current (j_ind) | ✓ | GPR-perturbed, then scaled to match l_i |
| Coil currents | ✓ | Adjusted by TokaMaker within the homotopy bounds |
| Aux channels (ω_tor, E_r, χ_e, χ_i) | optional | Switchboard: perturbed + stored when sigmas are supplied (passive) |
| p_fast, j_NBI, j_RF, j_other | ✗ | Fixed additive components, never perturbed (`j_other`: fusion, runaways, sawteeth and unknown-index core_sources entries on the IMAS path) |
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
| `zeff_scalar_sigma` | `0.05` | One Z_eff perturbation per draw; n_i / n_z follow from quasi-neutrality. Also the width of the bottom tier below |
| `zeff_sigma_source` | `"auto"` | Which tier supplies the Z_eff envelope's **magnitude**: `"auto"` / `"carbon"` / `"measured"` / `"scalar"` — see the ladder below |
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
  (`physics.evaluate_jBS(..., coord=)`), the loop's residual weights and the
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

**The Z_eff envelope has its own ladder (`zeff_sigma_source`).** Z_eff is the
primary density channel — each draw perturbs it and *derives* n_i / n_z — so the
width of its envelope sets the width of every dilution band in the ensemble.
`"auto"` takes the highest-fidelity tier the file supports:

| Tier | Where the magnitude comes from | Forced by |
|---|---|---|
| carbon-propagated | `n_12C6_err` + `n_e_err` (direct layout) or the dilution posterior (ensemble), propagated through `Z_eff = 1 + Z(Z−1)·n_C/n_e` | `"carbon"` |
| VB-measured | the file's own `Zeff_err` (direct) or `Zeff` sample spread (ensemble) | `"measured"` |
| scalar | `zeff_scalar_sigma` × abs(Z_eff) — an **assumed** width, not a measured one | `"scalar"` |

CER carbon is the direct measurement of the dilution the draw actually moves,
which is why it outranks the visible-bremsstrahlung sigma. Both measured tiers
require the Z_eff baseline to be the IDA one — the reconstruction path, and the
**same** file that supplies the sigmas (compared as resolved paths, so a
relative, `~`-prefixed or symlinked spelling is still the same file). An
IMAS/`ida_hybrid` or p-file baseline therefore always gets the scalar: pairing a
FUSE Z_eff with an IDA envelope would mix channels.

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
| `n_inspec_target` | `None` | Draw **until N draws pass both filters** rather than exactly `n_equils`. See [until-N](#until-n-in-spec-draws) below |
| `max_total_draws` | `None` | Attempt cap for `n_inspec_target` (default `5 ×` the target, never below `n_equils`). Hitting it is a loud, non-fatal outcome |
| `seed` | `None` | The run's one RNG seed. Consumed once into a single `numpy.random.Generator` threaded into every draw (GPR kinetic/aux/j_φ, the per-draw `scale_jBS`, the per-draw l_i target), so the same seed + inputs + solver gives a **bitwise-identical** archive on one machine (across machines the draws agree to ~1e-9, not bitwise -- LAPACK/BLAS). `None` = OS entropy |
| `scan_key` | `0` | Label for this bouquet within the archive (`scan/<key>/`) — a time in ms, a beta value, … |
| `l_i_tolerance` | `0.05` | l_i acceptance band, as a **fraction** of target |
| `constrain_sawteeth` | `False` | Gate draws on q0 |
| `recalculate_j_BS` | `True` | Recompute the Sauter bootstrap per draw (vs. reusing the baseline's) |
| `single_profile_jphi` | `False` | Legacy path: perturb the TOTAL `j_phi` as one profile (no inductive / bootstrap split; no per-draw Sauter call). `jphi_scalar_sigma` then applies to the total, a larger absolute perturbation -- re-tune it. Needs `jbs_self_consistent=False` (refused otherwise); the unified engine refuses it |
| `jBS_scale_range` | `(0.99, 1.01)` | Legacy draws: the per-draw multiplicative spread of the bootstrap (uniform in the range; default `None` -> `(0.99, 1.01)` on 2026-06-04). The baseline's multiplier (`bs_scale` / `s_bs(ψ)`) is applied after SWB on SWB draws and re-centres this range on loop draws -- see [physics-notes.md](physics-notes.md) ("The bootstrap multiplier") |
| `jbs_delta_mode` | `False` | Opt-in differential bootstrap composition — see [physics-notes.md](physics-notes.md#differential-bootstrap-jbs_delta_mode) |
| `isolate_edge_jBS` | `None` | Legacy path only. `None` is resolved per engine at `prepare_baseline()`: **`False`** under `reconstruction_engine="legacy"` (both input types; the unified forward decomposition -- pure-ohmic `j_inductive`, full bootstrap in `j_BS` -- closes exactly and yields better), `True` under `"unified"` (never read; the engine refuses `False`). Set `True` explicitly only for dedicated edge-spike studies (kept, with a warning). Recorded in the archive ([engine.md](engine.md)) |
| `jBS_baseline_mode` | `"diff"` | IMAS path: how the SWB bootstrap is reconciled with the source (`"diff"` / `"rescale"`) |
| `closure_channel` | `"bootstrap"` | IMAS `jBS_baseline_mode="ohmic"` path: which component absorbs the Ip closure. `"bootstrap"` / `"ohmic"` (deprecated bracket) rescale one component by a scalar; `"sawtooth_bootstrap"` adds a q0 pin where sawteeth justify it; `"structured"` replaces the scalars with minimal-norm radial multiplier **profiles** `s_ind(ψ)`, `s_bs(ψ)` — see [physics-notes.md](physics-notes.md#the-structured-closure-and-its-l_i-constraint) |
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
| `kinetic_source` | `"fuse"` | IMAS path: `"ida_hybrid"` takes ne/Te/Ti/ω_tor from an IDA `.cdf` while keeping FUSE Z_eff / currents / equilibrium. `from_imas(ida_path=…)` selects it automatically |
| `anchor_jtor_to_equilibrium` | `True` | IMAS path: anchor total j_phi to `equilibrium.profiles_1d.j_tor` rather than `core_profiles.j_tor` |
| `anchor_pressure_to_equilibrium` | `False` | IMAS path: add the fixed `p_diff = equilibrium.pressure − p_reconstructed` offset |
| `imas_corrective_jphi` | `False` | Opt-in corrective j_phi iteration on the IMAS baseline solve (still being validated) |
| `floor_j_BS` | `False` | Clip negative bootstrap excursions; only needed with `isolate_edge_jBS=False` on sources that carry an inner negative lobe |
| `bootstrap_kwargs` | `{}` | Keyword options passed through to `solve_with_bootstrap` (e.g. `{"iterations": 2}`; replaces `swb_iterations`). Checked at config time: a key the call sites already set, or one the installed toolkit does not accept, is refused. Under the self-consistent loop SWB runs only for `jbs_init="swb"` and the jBS-delta / `DIFF_BS` caches (a non-empty dict is warned). With `reconstruction_engine="unified"` only the edge-taper keys (`taper_edge_jBS`, `taper_edge_psi0`, `taper_edge_shape`; on by default, 0.999, quintic) and `use_sauter_eps=True` are read; any other key is refused. |
| `draw_solve_maxits` | `100` | GS iteration cap inside `generate()`. Draw solves converge in ≤ ~25 iterations; one that does not is stuck in a limit cycle just above `nl_tol` and would burn the setup cap (800, ~200–350 s). It is re-solved from where it stopped at each `draw_solve_retry_urf` (default none), then, if `draw_solve_loose_tol` is set (default unset; `2e-5` rescues the cycle), at that `nl_tol`, which accepts it only if the residual really is that small. Failed solves, and what recovered each, are listed per draw in `diagnostics['solve_failures']`, on `Bouquet.solve_failures`, and in one printed `[draw-solves]` line with the largest iteration count seen. `None` keeps the setup cap |
| `jbs_self_consistent` | `True` | Iterate the bootstrap to self-consistency with the delivered equilibrium (Redl on the caller's own ψ_N grid, re-evaluated after every solve; joint under-relaxation of the bootstrap and the solved current) in the baseline, every closure channel, the MSE stage, every draw and the reconstruction -- see [physics-notes.md](physics-notes.md#self-consistent-bootstrap-jbs_self_consistent). `False` = the **legacy** frozen-SWB bootstrap (legacy engine only; the unified engine refuses it); required with `single_profile_jphi=True` or `recalculate_j_BS=False` (refused otherwise). Not by itself a pre-release reproduction: that needs `reconstruction_engine="legacy"` + `separatrix_pressure="legacy"` too, and the coil-solve mode and the current conversion still differ (CHANGES_SUMMARY, "Reproducing a run made before this release"). A stored config without the field (pre-loop archive) loads as `False` |
| `jbs_init` | `"anchor"` | The loop's initial guess: Redl on the anchor equilibrium, or `"swb"` (legacy result; A/B only). The fixed point does not depend on it |
| `jbs_rtol_j` / `jbs_rtol_Ip` | `1e-3` / `1e-4` | Loop convergence: current-weighted L2 residual of the j_BS profile, and its current integral over I_p |
| `jbs_tol_li` / `jbs_tol_q0` | `1e-3` / `2e-3` | Loop convergence: pass-to-pass change of the solved l_i, and of q0 where an axis row is active |
| `jbs_max_passes` / `jbs_max_passes_draw` / `jbs_max_passes_post_homotopy` | `12` / `12` / `6` | Pass ceilings (limits, not tolerances): baseline / reconstruction; each loop of a draw; passes after a draw's coil homotopy when its Redl check misses (6 since the approved change from 4: measured need 5, plus one pass). Convergence = every active criterion on **two consecutive** passes |
| `jbs_relax` / `jbs_relax_halve_on` | `0.7` / `3` | Under-relaxation ω of the bootstrap; halved (floor 0.25) only when `r_j` grows on `jbs_relax_halve_on` consecutive passes (`1` = on every growth); three growing passes at the floor abort |
| `jbs_relax_current` | `0.7` | Under-relaxation β of the SOLVED current (`(1−β)` previous solved + `β` closure), damping the closure ↔ geometry oscillation; path only, gap recorded per pass. `1` = off |
| `jbs_gate_current_residual` | `False` | Legacy loop, opt-in: ADD the closure-half current residual `‖jc_k − js_k−1‖_w/‖jc_k‖_w ≤ jbs_rtol_j` to the loop's criteria (same two consecutive passes; nothing relaxed; pass 1 cannot count). The unified engine applies it always (a standing criterion) |
| `jbs_loop_on_fail` | `"raise"` | Non-convergence: raise `JBSNotConverged` (with the residual history), or `"flag"` the slice closure-limited and keep the last iterate. A non-converged draw is always a failed draw |
| `jbs_loop_q0_corrector` | `False` | The q0 pin under the loop (`"sawtooth_bootstrap"`, or `"structured"` with the axis row admitted). `False`: record-only, with the axis row held and the q0 residual flagged against `q0_tol`. `True`: the axis row is moved once per pass from the measured q0 (`j_ref0 ← j0_solved·q0/q0_target`), and convergence also requires `|q0 − q0_target| ≤ q0_tol` (unchanged); failure raises or flags like the loop. The l_i row is not covered. See [physics-notes.md](physics-notes.md#the-q0-pin-under-the-loop-jbs_loop_q0_corrector) |
| `reconstruction_engine` | `"unified"` | **Default `"unified"` since 2026-10-06** (was `"legacy"`): ONE reconstruction loop for g-file and IDS inputs, which also runs the draws ([engine.md](engine.md)). `"legacy"` = the existing g-file reconstruction / IMAS baseline paths and the legacy draws, opt-in by name (`from_geqdsk` / `from_imas(..., reconstruction_engine="legacy")`). Legacy-path settings the engine never reads are refused under `"unified"`, naming the field and saying to set `"legacy"`. A stored config without the field (written before 2026-09-29) loads as `"legacy"` with a warning. `generate()` / `verify_sigma0_consistency()` under `"unified"` run the engine draws from the engine baseline `prepare_baseline()` built IN THIS SESSION (its live state); a baseline built by the other engine, or one loaded without that state, is refused (`Bouquet._refuse_unified_engine_draws`) |
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
| `edge_pprime_pin` | `True` | Every solve: `True` zeroes the last node of `P'` (ψ_N = 1), so `P'` ramps to zero across the final interval (the behaviour before the setting); `False` keeps the profile's own derivative there -- a physics change (edge current and q95 move; see [physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)). The solver's resulting uniform `P'` rescale is recorded as `p_scale` |
| `separatrix_pressure` | `"offset"` | Every solve: `"offset"` (default since 2026-10-02, owner-approved physics change) hands the solver `p_axis − p_sep` and adds `p_sep` back wherever pressure, β or W_MHD is reported or delivered (written g-files carry the full `PRES`); `"legacy"` hands the full axis pressure (a non-zero `p_sep` then inflates `P'` by `p_axis/(p_axis − p_sep)`). A negative `p_sep` is refused under `"offset"` |
| `coil_drift` | `0.01` | Soft coil-drift target |
| `coil_drift_hard_factor` | `None` | Optional hard inequality bounds at `± factor·coil_drift` in every solve |
| `homotopy_passes` | `[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]` | Progressive `(F_tol, VSC_tol)` schedule — see [coil-constraints.md](coil-constraints.md) |
| `workflow` | `"auto"` | Named preset; see below |
| `allow_unsafe_workflow` | `False` | Deprecated alias for `workflow="custom"` |
| `allow_incomplete_pressure` | `False` | IMAS path: bypass the fail-fast pressure-accounting check |
| `capture_live_eq` | `True` | Snapshot each draw's converged flux-surface averages into `eq_fsa/` — what makes `fidelity="exact"` IDS export possible |
| `capture_npsi` | `257` | FSA grid for that block |
| `capture_exact_inv_R2` | `True` | Record ⟨1/R²⟩ in the draw's `eq_fsa` block (read from `get_q`, else by flux-surface quadrature). Archived geometry only: since 2026-10-06 the current conversion is the one field-aligned factor `F⟨1/R⟩/⟨B²⟩`, which does not read it |
| `diagnostic_plots` | `False` | Per-draw diagnostic figures |

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

`p_fast`, `j_NBI`, `j_RF`, `j_other` on their own `psi_N` grid — additive
components that are never perturbed (an explicit `j_other` replaces every
fusion / runaways / sawteeth / unknown-index entry the IMAS reader would hold,
and zeroes `j_sawteeth`). `coord` (default `"run"`) is the coordinate of that grid:
`"run"` (Φ_N in a `"phi_n"` run) or `"psi_n"`, mapped to the run coordinate
through the source equilibrium's ψ_N → Φ_N map. `j_NBI` / `j_RF` are given in bouquet's **positive-Ip
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
- **The RNG stream is untouched when the feature is off.** The `jBS_scales`
  block draw is unchanged for the first `n_equils` draws and only extends past
  it when until-N actually runs on, so `n_inspec_target=None` runs are bitwise
  what they were.

## Workflow presets and the guard

These presets are the LEGACY paths' (`reconstruction_engine="legacy"`). The
factories set none of these flags: `isolate_edge_jBS` and
`perturb_jind_in_anchor` default to `None` and `prepare_baseline()` resolves
them for the engine configured when it runs, so the engine may be named at
construction or set afterwards with the same result (the unified engine
reads none of them and refuses them when changed). With
`reconstruction_engine="legacy"` each input type gets the flag combination
validated for its path, and `generate()` raises on a known-bad combination:

| Preset | Input | What it resolves to |
|---|---|---|
| `geqdsk-standard` | g-file | Standard flagship l_i loop (`perturb_jind_in_anchor=False`), unified decomposition (`isolate_edge_jBS=False`) |
| `imas-diff-c` | IDS | Bootstrap anchored to the source via the fixed diff (`jBS_baseline_mode="diff"`, the default), inductive perturbed in the recon-anchor (`perturb_jind_in_anchor=True`), unified decomposition |
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
sc = bq.draw_scalars(ar, key)             # q0, q95, rho(q=2/1), rho(q=3/1), l_i, beta_N, <P>
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
limitations). Below `min_n=15` a record is flagged `below_floor` (shown hollow);
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
| `"exact"` | an engine draw's stored `<j.B>` parts (`jB_parallel/`, no conversion); otherwise the draw's **own** captured `eq_fsa` geometry (needs `avg_R`, `avg_inv_R2`, `pprime`) | draws deviate from the baseline; the split must track each perturbed equilibrium |
| `"reconstruct"` | the template's baseline equilibrium (`gm1/gm5/gm8/gm9/f/dpressure_dpsi`) | exact only when a draw's flux geometry matches the baseline's |
| `"auto"` *(default)* | stored parts, else exact when a complete `eq_fsa` block is present, else reconstruct | — |

No exported parallel current carries the pressure-driven term
`P = p'(<R> - F^2<1/R>/<B^2>)` (its `<j.B>` is zero; a reader recovers it from
the pressure): `j_ohmic` is the field-aligned inductive only and `j_total =
j_ohmic + j_bootstrap + driven`. The archived toroidal `j_BS` carries `P`
(FUSE's convention); when the draw has no stored
`<j.B>` parts the exporter subtracts `P` (from the archived eqdsk's own flux
surfaces) before converting. Export -> `IdsAdapter.read` returns the archived
`<j.B>` parts and `<j_phi>` (2026-10-06; before, `P/kappa` sat inside the
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
