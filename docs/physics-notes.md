# Physics notes

Behavioural notes on the parts of the pipeline where the physics, not the
plumbing, determines the answer. The full derivations, sign conventions, and
numerical floors live in [`../architecture.md`](../architecture.md); this page
covers the guarantees a user should know about and the knobs that change them.

## Contents

- [What the ensemble is (and isn't)](#what-the-ensemble-is-and-isnt)
- [The σ=0 consistency guard](#the-0-consistency-guard)
- [Bootstrap current treatment](#bootstrap-current-treatment)
- [Differential bootstrap (`jbs_delta_mode`)](#differential-bootstrap-jbs_delta_mode)
- [Self-consistent bootstrap (`jbs_self_consistent`)](#self-consistent-bootstrap-jbs_self_consistent)
- [The unified reconstruction engine (`reconstruction_engine`, default off)](#the-unified-reconstruction-engine-reconstruction_engine-default-off)
- [The pressure handed to the solver: separatrix pressure and the edge P′ pin](#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)
- [Kinetics regridding](#kinetics-regridding)
- [Edge-profile classification](#edge-profile-classification)
- [Hybrid kinetics on the IMAS path](#hybrid-kinetics-on-the-imas-path)
- [Current and field orientation](#current-and-field-orientation)
- [Z_eff-primary density scheme](#z_eff-primary-density-scheme)
- [Corrective j_phi iteration](#corrective-j_phi-iteration)
- [The structured closure and its l_i constraint](#the-structured-closure-and-its-l_i-constraint)
- [Core-pressure hollowness record](#core-pressure-hollowness-record)

---

## What the ensemble is (and isn't)

The selected (in-spec) draws are a **realizability-filtered sensitivity
ensemble**, not a calibrated Bayesian posterior. Read them as *"equilibria
consistent with the stated kinetic-profile uncertainty that remain broadly
machine-realizable"* — useful for sensitivity and what-if analysis. They are
**not** a posterior you should quote calibrated probabilistic confidence
intervals from. Specifically:

- Only the **kinetic profiles** are sampled (the prior); the equilibrium is
  forward-solved with the **boundary held** and the **coils left to drift**.
  Pinning the coils instead (and letting the boundary move) gives a *different*
  ensemble — neither is "the" posterior; the choice privileges the
  magnetics-measured boundary.
- Selection is a **hard threshold** on coil drift and boundary RMS (approximate
  Bayesian computation), **not** a likelihood weighting by the real measurement
  covariances, and there is no joint correlation structure between the
  perturbed quantities and the constraints.
- The coil thresholds (especially the VSC channel metric) are **engineering
  heuristics**, not the true measurement/control uncertainties — see
  [coil-constraints.md](coil-constraints.md) for the specific assumptions and
  their limits.

A genuinely calibrated posterior would replace the hard cut with soft
likelihood weighting using the joint coil/magnetics/kinetics covariances; that
is future work.

## The σ=0 consistency guard

The pipeline is designed so that at σ=0 (no kinetic perturbation) the output
reproduces the reconstructed equilibrium to within reconstruction's own
residual: l_i within ~0.5% of target, X-point within ~2 mm, boundary RMS
~3–4 mm against the input g-file. This requires the recon-anchor solve in
`perturb_kinetic_equilibrium` (replacing the seed-based inductive with
reconstruction's `j_inductive_fit`) and a post-anchor l_i tolerance gate; see
[architecture.md §3.3](../architecture.md#33-li-matching-recon-anchor--adaptive-gate).

`Bouquet.verify_sigma0_consistency()` turns that design intent into a runnable
check. It replays the exact per-draw pre-bootstrap sequence — state-anchor solve
at the baseline j_phi/pressure, `solve_with_bootstrap` on the baseline
kinetics, toroidal conversion, axis-transition smoothing — and compares the
resulting spike to `baseline.j_BS`:

```python
b.reconstruct()                          # or b.prepare()
res = b.verify_sigma0_consistency(tol_frac=0.02)
assert res["passed"], res["max_dev_frac"]
b.generate()
```

It returns `spike0`, `max_dev`, `rms_dev`, `max_dev_frac`, `psi_worst`, and
`passed`, and costs one bootstrap solve (~1 min). That is the **legacy**
(`jbs_self_consistent=False`) check; by default (the self-consistent loop)
`passed` means that an unperturbed draw, on every route the configuration
can use, reproduces the one reconstruction state at the loop's tolerances
(the `draw_route` block, with q0, q95 and the total-current profile reported
beside them), and the loop "solved the baseline's way" is reported as
`passed_baseline_way` -- see
[the self-consistent bootstrap](#one-reconstruction-state-and-what-an-unperturbed-draw-reproduces). Call it after
`reconstruct()` / `prepare_baseline()` and before `generate()`; it leaves the
solver re-anchored on the baseline equilibrium.

**Why it exists.** Any systematic deviation at σ=0 is inherited by *every* draw
as a j_phi target bias. The 2026-07 hollow-core bug was exactly such an
inconsistency: near-axis Gaussian smoothing of j_BS was applied on the
reconstruction path only, so every draw got the raw collapsed innermost-surface
point instead. The resulting grid-point-scale axis deficit hollowed every
draw's core by ~9% and shifted the q0 distribution wholesale by +12% — while
leaving l_i unbiased, and therefore invisible to every existing diagnostic.
The fix mirrors the smoothing into the draw path so both share it; this guard
is what keeps it from silently regressing.

## Bootstrap current treatment

The per-draw bootstrap comes from TokaMaker's Sauter/Redl
`solve_with_bootstrap`, whose output is already TokaMaker `jphi` (field-aligned
part plus the pressure term p′G; see [current-conventions.md](current-conventions.md))
and is used as is.

Two composition modes:

- **Shared near-axis smoothing** (default). Every spike — baseline and draws
  alike — passes through the same `smooth_jbs_transition` axis treatment. This
  is the fix for the hollow-core bug above and is what
  `verify_sigma0_consistency` exercises.
- **Differential composition** (`jbs_delta_mode=True`), below.

`recalculate_j_BS=False` skips the per-draw recompute entirely and keeps the
baseline j_BS — useful for isolating pressure-driven from current-driven
responses, not for production.

Both `from_geqdsk` and `from_imas` set `isolate_edge_jBS=False`, i.e. the
unified forward decomposition: `j_inductive` is pure ohmic and `j_BS` carries
the full physical Sauter profile (core hump + edge spike). It closes exactly,
is non-negative, and yields better than the older isolated-edge-spike split.
Set `isolate_edge_jBS=True` only for dedicated edge-spike studies.

## Differential bootstrap (`jbs_delta_mode`)

Opt-in alternative to the shared smoothing. Each draw's spike is composed as

```
j_BS(draw) = baseline_j_BS + [ SWB_raw(perturbed) − SWB_raw(σ=0) ]
```

with both solver terms **raw** (no smoothing of the perturbed profiles). Any
common-mode evaluation artifact — the collapsed innermost-surface point being
the motivating example — cancels exactly, and the per-draw Sauter response
passes through unfiltered. The l_i conditioning machinery is unchanged.

Cost: one extra `solve_with_bootstrap` call per run for the σ=0 reference,
computed in the same pre-draw anchor context. Under this mode the σ=0 draw
reproduces the baseline split exactly *by construction*, so
`verify_sigma0_consistency` becomes a pure bootstrap-context-reproducibility
probe rather than an independent check.

## Self-consistent bootstrap (`jbs_self_consistent`)

**On by default** (`GenerationConfig.jbs_self_consistent=True`): the bootstrap
is re-evaluated on the delivered equilibrium inside a relaxed outer loop
(closure ↔ GS solve ↔ Redl) that runs to a convergence test, in every path
that builds a j_phi containing a bootstrap. `jbs_self_consistent=False` is the
**legacy frozen bootstrap** (with `reconstruction_engine="legacy"`; the
unified engine is the loop and refuses it), kept for A/B comparisons and for
reproducing archives made before the loop existed. It is NOT by itself the
pre-release code path bit for bit: reproduction also needs
`reconstruction_engine="legacy"` and `separatrix_pressure="legacy"`, and two
moves are not switchable -- the canonical coil-solve mode (<= 5e-4 relative on
draws) and the one current conversion (the frozen bootstrap is ~6.4-6.8 %
lower at the pedestal on the synthetic example). The structured soft
closure's noise-floor acceptance, below, is passed only by the loop's closure
calls (`utils.close_ip_structured_soft(..., accept_noise_floor=True)`); every
frozen-path call keeps the historical strict solver. Since the loop is the
default, that acceptance is on the DEFAULT path (a rounding-level effect).
Two consequences of the default:

- `single_profile_jphi=True` and `recalculate_j_BS=False` have no bootstrap to
  iterate; they are refused unless `jbs_self_consistent=False` is set (never
  silently downgraded -- the error says so).
- A stored config that predates the field (an old archive's `config_json`,
  or any dict/JSON without it) loads with `jbs_self_consistent=False` and a
  warning, i.e. it replays the bootstrap model it was produced with; the
  warning says how to opt in (`"jbs_self_consistent": true` in the
  `generation` section). A current config always carries the field. An
  unknown (e.g. misspelt) `generation` key is refused, naming the nearest
  valid key, so a typo can no longer land on this legacy default.
- `bootstrap_kwargs` configures `solve_with_bootstrap`; under the loop SWB
  runs only for `jbs_init="swb"` and the jBS-delta / `DIFF_BS` caches, so a
  non-empty dict raises a `DeprecationWarning` saying where it acts.  (It
  replaced `swb_iterations`; a stored value loads as `{"iterations": n}`.)

The archive says which model a group carries: the schema-v3 `jbs_loop` block
(below) is present exactly where the loop ran; plots label the bootstrap
"self-consistent Redl bootstrap" or "frozen SWB bootstrap (legacy)"
accordingly.

### What the legacy path does, and why it is not enough

With the legacy flag, the bootstrap is computed **once** per baseline and per draw
by OFT's `solve_with_bootstrap` (SWB) and then frozen: the closures, the
correctors, the MSE stage and the draws only *rescale* it. Two properties of
that single call matter:

- **It runs on SWB's own auxiliary equilibrium** -- a generic power-law
  inductive seed, thermal main-ion pressure only (no impurity, no fast ions),
  a fixed number of Picard passes with no convergence test. The equilibrium
  bouquet finally delivers (the source's inductive current, the full pressure,
  the closure's multipliers, the post-homotopy coils) is never the one the
  bootstrap was evaluated on. Since `j_BS ∝ (dp/dψ_N)/Δψ` and `Δψ` scales with
  `√l_i`, a different l_i alone moves the bootstrap amplitude by several
  percent; the inductive seed alone moves it by more.
- **It assumes evenly sampled `ψ_N`** unless the OFT build accepts `psi_N=`
  (and bouquet never passed it). A source grid uniform in ρ_tor -- the usual
  IMAS/integrated-modelling grid -- is then read as ψ_N-uniform: the profiles'
  gradients are mis-scaled by `dψ_N/du` (for `ψ_N ≈ ρ²` that factor is
  `≈ 2ρ`: far below 1 near the axis, above 1 beyond mid-radius) and the
  geometry is sampled at the wrong surfaces.
  The geqdsk path is unaffected (its grid is uniform).

### The evaluator: `physics.evaluate_jBS`

`evaluate_jBS(mygs, psi_N, ne, te, ni, ti, zeff)` is a faithful port of SWB's
inner Redl evaluation (NRL electron / `Zavg` ion Coulomb logarithms, Koh ion
collisionality, `redl_bootstrap(formula_form='jboot1', use_sign_q=True)`) run
**once on the equilibrium it is handed -- no solve inside** -- with three
differences, each the point of the helper:

1. geometry (`F`, `f_T = 1 − f_c`, `ε = ⟨a⟩/⟨R⟩`, `q`, `⟨R⟩`) is sampled on the
   **caller's** surfaces, `clip(ψ_N, psi_pad, 1 − psi_pad)`;
2. gradients are taken on the **true** grid, `numpy.gradient(y, ψ_N,
   edge_order=2)`, divided by the **current** flux range;
3. Redl's `⟨j·B⟩` is converted to TokaMaker `jphi = ⟨j_φ⟩` exactly, by (A7) of
   [current-conventions](current-conventions.md): the field-aligned
   `F⟨1/R⟩⟨j·B⟩/⟨B²⟩` plus the pressure-driven `p′(⟨R⟩ − F²⟨1/R⟩/⟨B²⟩)`, all of
   the same surfaces. The bootstrap component carries `p′G`, as IMAS
   `j_bootstrap` and OFT's own SWB output do, and the IDS export inverts the
   same relations.

   **Declared default physics change (2026-10-06, owner-approved).** Until then
   the legacy sites converted with `⟨j·B⟩/(F⟨1/R⟩)` (`⟨1/R²⟩` not passed) and
   the IDS export with `⟨j·B⟩F⟨1/R²⟩/(⟨B²⟩⟨1/R⟩)`. The legacy factor exceeds κ
   by `⟨B²⟩/(F²⟨1/R⟩²)` = the bracket `⟨B²⟩/⟨B_φ²⟩` (the poloidal-field
   content: ≈ 1.5 % at ψ_N ≈ 0.97 on the synthetic D3D-like example) × the
   Jensen ratio `⟨1/R²⟩/⟨1/R⟩²` (≈ 5 % there) -- **+6.8 %** at the pedestal
   (+1.0 % at ψ_N 0.1, +4.3 % at 0.5, +6.4 % at 0.9). The legacy bootstrap
   drops by that fraction (before `p′G` is added); the unified engine,
   which already used κ, is unchanged (`tests/test_one_conversion.py`). (The bracket alone, ~1.4 % at
   the peak, is what this page used to quote; the Jensen term was missed.)

**Refusals, never a silent zero.** The historical evaluation mapped every NaN
of the Redl expressions to `j_BS = 0` at that node. `evaluate_jBS` now raises
`physics.JBSEvaluationError` (a `ValueError`), naming the quantity, the first
ψ_N and the grid index, for non-physical input (`n_e`, `n_i`, `T_e`, `T_i`
not strictly positive, or `Z_eff < 1`, at any node), for a surface the tracer
failed on (the all-zero row: a non-positive `⟨R⟩`, `⟨1/R⟩`, `⟨a⟩`, `⟨B²⟩` or
`dV/dψ`, `F = 0`, `q = 0`, `f_T ≥ 1`; also `f_T ≤ 0` away from the axis), and
for a non-finite Redl value anywhere but the END nodes. The END nodes -- those
whose geometry is the clipped axis or separatrix surface, `ψ_N ≤ psi_pad` or
`ψ_N ≥ 1 − psi_pad`, identified by coordinate -- keep the historical
treatment exactly (a non-finite value there is zeroed and counted in
`diag["n_nonfinite_zeroed_at_ends"]`). For every accepted input the output is
bit-identical to the evaluator before the refusals (a fast test compares it
against a verbatim copy). The production callers clip `Z_eff` at 1 before
calling, as they always did.

On a uniform grid it reproduces SWB's first-pass `⟨j·B⟩` **bit for bit** (on a
build whose SWB accepts `psi_N=`). Grids whose first intervals are finer than
`psi_pad` (a ρ-uniform grid near the axis) are handled without changing
`psi_pad` and without merging surfaces: every point keeps its own profile value
and gradient; only the geometry of the points inside the pad is looked up at
`psi_pad`, once. It uses only primitives present on every supported OFT build
(`get_profiles`, `sauter_fc`, `get_q`, `psi_bounds`, `redl_bootstrap`,
`calculate_ln_lambda`), in either of their return layouts.

### The loop

```
E_0   = anchor equilibrium (the source's total current, full pressure)
jBS_0 = evaluate_jBS(E_0)                      # jbs_init="anchor" (default)
for k:
    E_k+1   = solve( closure on E_k's geometry, assembled with jBS_k )
    J       = evaluate_jBS(E_k+1)
    residuals(J, jBS_k, E_k+1, E_k)
    jBS_k+1 = (1 − ω) jBS_k + ω J
```

Residuals, **all logged every pass**:

| residual | definition | tolerance |
|---|---|---|
| `r_j` | `‖J − jBS_k‖_w / ‖J‖_w`, `‖f‖_w² = ∫ |w| f² dψ_N` with the pass's own Ip weights `w` | `jbs_rtol_j = 1e-3` |
| `r_I` | `|∫ w (J − jBS_k) dψ_N| / I_p` (linear part of the closure's measure) | `jbs_rtol_Ip = 1e-4` |
| `Δl_i` | `|l_i(E_k+1) − l_i(E_k)|` | `jbs_tol_li = 1e-3` |
| `Δq0` | `|q0(E_k+1) − q0(E_k)|`, only where an axis row is active | `jbs_tol_q0 = 2e-3` |
| `q0 − q0_target` | the true q0 residual on the pass's solved equilibrium; a criterion only with `jbs_loop_q0_corrector=True` on an axis-row channel (below) | `q0_tol = 0.01` (unchanged) |

`r_j` and `r_I` are the **unrelaxed** fixed-point residual -- the distance
between the bootstrap an equilibrium was solved with and the Redl bootstrap of
that equilibrium. That is `1/ω` times the relaxed step `‖jBS_k+1 − jBS_k‖`, so
the criterion is never looser than a step-size test. Converged means **every
active criterion on two consecutive passes**. Ceilings (limits, not
tolerances): `jbs_max_passes = 12` (baseline / reconstruction; 8 before 2026-10-07),
`jbs_max_passes_draw = 12` for each loop of a draw (its anchor loop, each
l_i-match candidate's coupling, each Fix C resample), and
`jbs_max_passes_post_homotopy = 6` passes after a draw's coil homotopy when
Redl on the delivered equilibrium misses (with the two-consecutive rule a
stage whose first pass misses needs at least 3). The draw ceilings were
raised from 6 / 2 once the golden case showed the standard draw's l_i-match
coupling contracting at ≈0.38/pass from r_j ≈ 2e-2…1.2e-1 (7–8 passes) --
a limit change; no tolerance moved. The post-homotopy ceiling was then
raised from 4 to 6, an owner-approved change of a pass ceiling: in the
passes-to-convergence study no draw needed more than 5 post-homotopy
passes, and every draw a ceiling of 4 had rejected converged on its next
pass; 6 is that measured need plus one pass. Again a limit change; no
tolerance and no criterion moved.

**Where a draw's loop starts.** Every draw's first loop starts from
`evaluate_jBS` on the draw's **state anchor** (the archived total current at
the draw's full pressure) with the draw's **own perturbed kinetics**
(n_e, T_e, n_i, T_i, Z_eff; in delta mode composed as baseline + (that Redl −
the σ=0 Redl reference); `jBS_diff` added in diff mode) -- never from the
unperturbed baseline bootstrap. Later loops of the same draw (the next
l_i-match candidate, a Fix C resample) start warm from the draw's previous
converged bootstrap; the post-homotopy stage starts from the relaxed blend of
the bootstrap the draw carries and Redl on the delivered equilibrium. Each loop
record says which (`init_source`), and the per-draw block repeats the first
one. The start changes the path only (a test runs the same draw loop from the
baseline bootstrap and from the anchor Redl and gets the same fixed point).
The large first residual of an l_i-match candidate's loop is geometric: the
candidate's new inductive shape moves q and the flux range, and Redl with the
**same** kinetics on that geometry differs by a few to ~10 % in I_BS.
Relaxation, on two quantities, both of the **path** only (at the fixed point
both blends are the identity):

* **the bootstrap**, `jBS_k+1 = (1 − ω) jBS_k + ω J` with `ω = jbs_relax = 0.7`,
  held fixed and halved (floor 0.25) only on **sustained** growth of `r_j` —
  growth on `jbs_relax_halve_on = 3` consecutive passes (`1` restores the
  earlier halve-on-every-growth schedule). Three growing passes at the floor
  abort early.
* **the solved current**, `js_k = (1 − β) js_k−1 + β jc_k` (from the second pass
  on) with `β = jbs_relax_current = 0.7`: each pass closes on the *previous*
  equilibrium's geometry, so the closure's current `jc` and the geometry it
  produces form an oscillating two-state mode (l_i swings back by a fraction
  g ≈ −0.5 per pass) that ω does not act on; `β ≈ 1/(1 − g)` damps it. The
  record carries β, the per-pass gap `‖js − jc‖_w / ‖jc‖_w` and, next to it,
  the **unrelaxed** closure-half residual `‖jc_k − js_k−1‖_w / ‖jc_k‖_w`
  (`current_residual_unrelaxed`; it equals gap/(1 − β) on a blended pass, so
  the gap understates it by the factor 1 − β, and it is also defined at
  β = 1). Both are recorded only, **not gated**: the convergence gate is the
  four criteria above. Whether to gate the unrelaxed residual (a stricter
  "converged") is a pending decision, and the record exists so it can be
  made with numbers. Applied where a pass solves one assembled j_phi (the IMAS baseline
  loop in every mode and channel, the σ=0 check, the draws' anchor loops); not
  in the standard draw's l_i-match coupling, the geqdsk reconstruction or the
  MSE chord steps (whose linearisation is centred on the closure's own
  current) — their records say so.

A single growth of `r_j` is the forced response of that mode, not divergence,
which is why ω is no longer halved on it. **None of the existing solver or closure tolerances
(`nl_tol`, `maxits`, `structured_li_tol`, `q0_tol`, the soft solver's
`rtol`/`max_iter`, `SIGN_ITER_MAX`) is touched**; these numbers define what
"j_BS converged" means and are initial values, to be revisited with data.

Failure is never silent. `jbs_loop_on_fail="raise"` (default) raises
`jbs_loop.JBSNotConverged` carrying the whole residual history;
`"flag"` delivers the last iterate with `jbs_converged=False` and a
`"j_BS loop: …"` reason in `closure_limited_reasons`. A draw whose loop does
not converge is a **failed draw** in either mode. A pass that can never count
-- a gated l_i or q0 the step did not return (or returned non-finite), or an
identically zero Redl bootstrap against a non-zero iterate -- ends the loop at
that pass (then raise or flag as above) instead of running to the ceiling. A
non-finite initial guess or evaluated bootstrap raises `jbs_loop.JBSNonFinite`
(a `JBSNotConverged`) at once, with the pass number and the ψ_N location,
whatever the policy: it is never blended into the next iterate or handed to a
GS solve.

### The q0 pin under the loop (`jbs_loop_q0_corrector`)

This applies to the channels that pin the on-axis safety factor: IMAS
baseline, `jBS_baseline_mode="ohmic"`, with `closure_channel=
"sawtooth_bootstrap"`, or `"structured"` when the sawtooth gate admits the
axis row.

- **Default (`False`): record-only.** Every pass closes with the axis row
  held at the anchor's requested axis current. The delivered equilibrium's
  `q0 − q0_target` is recorded and flagged against `q0_tol`.
- **`True`: the pin acts.** After pass k is solved, the q0 measured on its
  equilibrium moves the row for pass k+1:

  ```
  j_ref0(k+1) = j0_solved(k) · q0(E_k+1) / q0_target
  ```

  This is the structured corrector's `j_ref0' = j_ref0 · q0_solved/q0_target`,
  applied once per pass; the scalar corrector's Newton step is its
  first-order expansion. `j0_solved` is the axis value of the current the pass
  actually solved: under `jbs_relax_current = β` the equilibrium sees
  `(1 − β) js_k−1(0) + β j_ref0(k)`, not the row. With β = 1 the two are the
  same.

  In the `q0 ~ 1/j0` model the update lands the row in one step, and the
  solved axis current then follows it at the β rate. At the joint fixed point
  the row stops moving exactly when `q0 = q0_target`, so, like ω and β, the
  update changes the path and not the answer. The pin's criterion
  `|q0 − q0_target| ≤ q0_tol` is **added** to the pass criteria next to
  `Δq0 ≤ jbs_tol_q0`. So the loop converges on the bootstrap, I_p, l_i and q0
  together, and the delivered equilibrium (the last one solved, whose
  bootstrap was the last evaluated) meets all of them.
- **MSE chord stage.** The chord steps are passes of the loop, so the stage's
  refresh moves the row from each step's q0. Every chord step, the final
  step and the refusal re-solve carry the same added criterion.
- **Unchanged.** `q0_tol`, every loop tolerance and the pass ceilings do not
  move. A joint iteration that does not converge within the ceiling raises
  `JBSNotConverged`, or flags the slice under `jbs_loop_on_fail="flag"`. The
  q0 residual history is in the message and in `jbs_loop["q0_pin"]`, and the
  reason is in `closure_limited_reasons`. The delivered equilibrium is
  checked against `q0_tol` once more after the correctors' readback, so no
  path delivers a q0 outside `q0_tol` as converged.
- **Records.** Per pass, in `jbs_loop["q0_pin"]`: the axis row, the solved
  axis current, q0, the residual, the residual / `q0_tol` and the next row.
  In `ip_closure`: `q0_pin_acted`, `q0_pin_n_row_updates`,
  `q0_pin_axis_row_initial` / `_final`, `q0_residual_over_tol` and
  `q0_pin_delivered_within_tol`. The run-time NOTICE names the mode.
- **Not covered: the l_i row.** It stays held at its target either way; its
  log-gain row update is a separate design, not part of this flag.

### Where it runs

- **IMAS baseline, `jBS_baseline_mode="ohmic"`** (every `closure_channel`):
  each pass re-builds the closure geometry (FSA geometry, ⟨1/R²⟩, `w_lin`, the
  affine `c`, the probe profile, `li_geom`) on the current iterate and re-solves
  the channel's closure there. Held fixed: I_p and σ_Ip, the l_i target and
  σ_li, the MSE chords, the prior ladders, and `q0_target` / the axis-current
  row -- computed once from the source's total on the ORIGINAL anchor, because
  they are data-derived targets, not forward-model quantities. The l_i model is
  exact given its geometry, so with `li_geom` refreshed every pass its
  frozen-geometry error vanishes at the fixed point, and the l_i corrector
  *step* is not needed (no row rescaling). **The q0 residual is different.**
  `q0_target = q0_anchor · j_ach0 / j_req0` is a first-order `q0 ~ 1/j_φ(0)`
  mapping at the anchor, and at the loop's fixed point the held row enforces
  `j_φ(0) = j_req0`, not `q0 = q0_target`. Refreshing the geometry therefore
  does not remove the residual that the legacy Newton step removes. By default
  (`jbs_loop_q0_corrector=False`) that residual is only recorded and flagged
  against `q0_tol`; see "The q0 pin under the loop" below. The correctors
  still run, in record-only mode on the delivered equilibrium, so every
  bookkeeping field and every acceptance flag (`structured_li_tol`, `q0_tol`,
  the round-trip gate) is written as before -- "predictor" = the first pass,
  "corrected" = the delivered equilibrium.
- **`"rescale"`**: the l_i-proxy root is re-solved on every iterate.
- **`"diff"`**: the baseline total stays pinned to the source, so the baseline
  itself needs no loop; `jBS_diff` is redefined as
  `source j_BS − evaluate_jBS(delivered baseline, source kinetics)` -- a pure
  model offset on the baseline geometry, frozen in ψ_N labels. A σ=0 draw
  whose equilibrium returns to the baseline therefore reproduces the source
  bootstrap exactly. Only the draws iterate. The stored split is then
  normalised to the one reconstruction state (see "One reconstruction
  state" below); the bootstrap and `jBS_diff` are untouched by that.
- **MSE** (`closure_channel="structured"` with `mse_data`): converge the loop
  WITHOUT the MSE term; the forward-difference Jacobian of tanγ once at that
  state (j_BS held fixed during the differences -- an approximation of the loop
  map's Jacobian, recorded as such); chord steps -- closure with the linearised
  MSE term on the current geometry and bootstrap → solve → `evaluate_jBS`
  (+ relaxation) → refreshed offset and closure state -- until the j_BS
  residuals hold on two consecutive steps **and** tanγ moved by less than
  `jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA = 0.1` σ on every chord between steps
  (at most `jbs_max_passes` steps -- each chord step is also a pass of the
  bootstrap loop); then the Jacobian is recomputed ONCE at
  the converged state and one final step taken (a stale Jacobian biases the
  stationary point of a chord iteration, not only its rate). The Jacobian
  change and the objective change of that step are recorded.
- **Draws** (`perturb_kinetic_equilibrium`): the per-draw composition is
  unchanged (scale, smoothing or delta mode, floor, `jBS_diff`) with
  `evaluate_jBS` on the draw's own equilibrium in place of SWB. Fix C: per GPR
  candidate, {Redl → Ip renormalisation of the inductive with the R2 measure
  rebuilt on the current iterate → solve}; every band-conditioning resample
  converges its own loop. Standard l_i loop: the anchor bootstrap is converged
  first; each candidate is then re-solved (root, `find_optimal_scale`,
  corrective) while its bootstrap still moves beyond tolerance (Gauss–Seidel)
  before the l_i band judges it. After the post-perturb coil homotopy, Redl on
  the delivered equilibrium is checked against the bootstrap it carries;
  outside tolerance, up to two passes at the tight coil stage, else the draw is
  rejected. With `jbs_delta_mode` the σ=0 reference is `evaluate_jBS` on the
  cache anchor, so delta mode and the shared mode coincide to loop tolerance
  (the flag is kept for back-compatibility). The `PIN_JPHI` / `DIFF_BS`
  diagnostics are unchanged.
- **geqdsk reconstruction**: `E_0` is the g-file's own current at the full
  pressure; the loop wraps the inductive fit + l_i secant; the corrective
  iteration runs once afterwards and is followed by the **l_i re-match** of
  its landed request (below), then a post-corrective check (up to
  `jbs_max_passes` further passes of fit + l_i match + corrective +
  re-match). Every pass ends on the re-matched state.
- **`verify_sigma0_consistency`**: `passed` requires the draw's OWN route(s)
  -- every route the configuration can use -- to reproduce the one
  reconstruction state at zero perturbation, at the loop's unchanged
  tolerances (below). The loop "solved the baseline's way" (the baseline
  inductive held, one jphi-linterp solve per pass, which is how the
  reconstruction's final state is solved) is kept beside it as
  `passed_baseline_way`. Route R2's inductive Ip renormalisation keeps its
  own, separately budgeted σ=0 invariant as well.

### One reconstruction state, and what an unperturbed draw reproduces

There are three levels: the INPUT (a g-file, or a modelling-source IDS); the
bouquet RECONSTRUCTION, as close to the input as it can be while physically
valid and carrying a neoclassical bootstrap current -- it is allowed to
differ from the input (the input usually carries no Redl/Sauter bootstrap);
and the DRAWS, perturbations of the reconstruction. With the loop on:

- **The reconstruction is one equilibrium F.** The saved baseline g-file,
  `l_i_target` and every recorded l_i / q0 / q95 (`Baseline.delivered_state`,
  `reconstruction_metrics`), the archived baseline profiles, the centre of the
  draws' l_i band and the reference of the zero-perturbation check are all F.
  Its bootstrap is Redl evaluated on F (loop-converged).
- **On the g-file path F keeps the l_i match to the input.** The corrective
  iteration shapes the current toward the fitted target but moved l_i(3)
  +0.40 % off the step-6 match on the synthetic example, and before this
  change the baseline carried three states: the l_i-matched target
  (0.653864), the post-corrective state (0.656455) and the saved g-file, a
  single re-solve of the stored achieved current (0.653866). F is now the
  corrective iteration's landed *request* re-matched in l_i by the same
  secant step 5 uses, on the inductive amplitude of that request (the shape
  the corrective iteration gave the inductive is kept; only its amplitude
  relative to the bootstrap moves). F is a single jphi-linterp solve of a
  known request -- the route every draw pass takes. (Alternative, not
  taken: F = the post-corrective state, `l_i_target` = its l_i. That gives
  up the l_i match -- the band centre would move +2.59e-3 in l_i, and q95 of
  the reference by ≈0.024, about one ensemble σ -- and a single-solve route
  cannot reach it.)
- **The stored split is F in the form the draws consume it.** `j_phi`
  (+ `jphi_diff`) is F's jphi-linterp request, normalised to I_p in the
  'exact' FSA current measure ON F (the measure route R2 roots in; a uniform
  factor, so F is unchanged); `j_BS` (+ `jBS_diff`) is the draws' own σ=0
  bootstrap composition on F (Redl, scale, floor, `jBS_diff`); the fixed
  parts are as read; `j_inductive` is the residual, so it carries the whole
  normalisation. On the modelling-source example the source total read
  0.965 I_p in that measure and the solver made it up uniformly; the stored
  split no longer hands the draws a total 3.5 % short. In diff mode
  `j_BS + jBS_diff` is still the source bootstrap exactly. On the g-file path
  the inductive is floored at zero by the usual convention; a floored point
  is one where a zero-perturbation draw cannot reproduce F, and is counted
  (`delivered_state["n_floored_inductive"]`).
- **Every draw stage is the identity at zero perturbation.** The state anchor
  is one solve of the stored request (`jphi_diff` included) -- F. Route R2's
  scale is 1 to rounding (same measure, normalised split). The standard
  route's l_i stage targets ACHIEVED currents: it perturbs
  `j_inductive - jphi_request_offset` (F's achieved current minus the σ=0
  bootstrap and fixed parts), roots its inductive amplitude in the same
  exact measure on the live geometry (was: the limiter-area flux integral,
  which read the total 10–44 % high), so the root is 1; `find_optimal_scale`
  then accepts its first trial (scale 1); and the corrective iteration starts
  from `target + jphi_request_offset`, which at zero perturbation is F's own
  request, so its first iterate IS F. The only residual departure there is
  the corrective target's exact-measure I_p normalisation against the
  solver's own (the measure's self-check, 1e-5–4e-4 of I_p), second order in
  the state. The loop passes and the post-homotopy passes then find F's
  bootstrap already self-consistent. Every sampled perturbation (kinetics,
  inductive GPR, bootstrap scale, l_i target) enters as a departure from the
  reconstruction's value.
- **What is left non-identity by construction:** for an asymmetric
  `jBS_scale_range` the draws' centre scale is not the reconstruction's;
  `jBS_baseline_mode="ohmic"` (baseline-only; the draws refuse it) keeps its
  split as before. (The electron-charge mismatch that used to sit here -- the
  modelling-source forward solve at 1.602176634e-19 against the draws'
  1.6022e-19, a σ=0 draw pressure 1.28e-5 relative high -- is gone under the
  loop: every pressure there uses `physics.ELEMENTARY_CHARGE`; only the
  frozen legacy path keeps `ELEMENTARY_CHARGE_LEGACY` for its thermal terms.)

`jbs_init="swb"` starts from the legacy SWB result instead (A/B only; `psi_N=`
is passed when the OFT build accepts it and the grid allows it, and the record
says whether it was). The fixed point does not depend on the initial guess.

The loop is refused with `single_profile_jphi=True` (no bootstrap component)
and with `recalculate_j_BS=False`; set `jbs_self_consistent=False` for either.

**Closure stop test inside the loop.** Each pass re-solves the closure on
the new geometry, so the structured soft closure is called several times per
slice. **Its acceptance criterion is changed for the loop only**: old
criterion -- when no damped step descends, accept iff the scaled gradient is
below `rtol·max|J|·max(√F, 1)`, else refuse; new criterion (with the loop on)
-- the same, plus accept an iterate stationary to within the objective's
rounding noise (`stop_reason="noise_floor"`). It therefore accepts points the
old test refused (and only those: every result the old test returned is
unchanged). It is opt-in (`accept_noise_floor=True`, passed by the loop's
closure calls only); every acceptance is recorded (gradient, predicted
decrease, noise estimate, `n_noise_floor_accepts`) and printed, and a noise
estimate that is not finite and positive accepts nothing. A refusal inside
the loop is retried once from the previous pass's coefficients
(`closure_retry=1`, logged) -- see the soft closure under
[the structured closure](#the-structured-closure-and-its-l_i-constraint).

### What is recorded

`li_metrics["jbs_loop"]` (and `ip_closure["jbs_loop"]` in ohmic mode;
`reconstruction_metrics["jbs_loop"]` on the geqdsk path; in the archive, the
schema-v3 `jbs_loop` block -- attrs `jbs_converged`, `jbs_n_passes`,
`jbs_loop_json` -- on every loop draw and on `_baseline`, read with
`bouquet.utils.load_jbs_loop`, `DrawView.jbs_loop` or
`ScanView.baseline_jbs_loop`; see
[archive-schema.md](archive-schema.md#v2--v3-the-self-consistent-bootstrap-record)): `enabled, init, grid, n_passes, converged,
stop_reason, tolerances, omega[], r_j[], r_I[], dl_i[], dq0[], I_BS[],
jBS_peak_psiN[], jBS_peak[], wall_s, evaluate_jBS_version, oft_build`
(`version`, 12-character `git_hash`, `library_sha256` of the loaded
`liboftpy`, `sources_sha256` of the package's Python sources, `build_id`; no
filesystem path -- records
written by earlier builds of this branch carried the OFT package path there),
`current_gap[]` and, next to it, `current_residual_unrelaxed[]` (record only,
see above), plus `jBS_diff_definition` in diff mode, the per-pass closure log
in ohmic mode, the MSE chord-stage block, and the post-move check on draws and
reconstructions.

### What it does not change

The kinetic profiles stay pinned to their ψ_N labels: the loop makes the
geometry Redl sees the delivered equilibrium's and the gradients true-grid, but
it does not move a measurement to a different flux surface when the current
redistributes. The Redl drive is main-ion + electron (the model definition);
impurity and fast-ion pressure enter the GS pressure, not the bootstrap drive.

**Cost.** SWB's generic-seed cold solves are replaced by warm-started passes;
on the synthetic D3D-like IMAS example the loop converged in 2 passes
(`rescale`), 5 (`bootstrap`), 6 (`sawtooth_bootstrap`) and 8 (`structured`, soft
preset with the axis row -- its one-sided prior makes the closure map
non-smooth, which costs relaxation), and a Fix-C draw at 5 % kinetic / j_φ σ
needed 7 passes -- one more than the draw default.

## The unified reconstruction engine (`reconstruction_engine`, default off)

`GenerationConfig.reconstruction_engine="unified"` (default `"legacy"`; with
the factories, `Bouquet.from_geqdsk/from_imas(..., reconstruction_engine=
"unified")`, which leave the legacy-path workflow settings at their defaults)
replaces the g-file reconstruction and the IMAS baseline with ONE loop for
both inputs ([engine.md](engine.md)). Every current component is stored as a
parallel current `<j.B>`: the g-file's from identity (I0) on its own surfaces
(minus Redl on the anchor, smoothed with the existing inductive basis, no
amplitude search), the IDS's as `|B0|` times its `<j.B>/B0` fields. Each pass
composes the solver's `<j_phi>` on the latest SOLVED geometry with the
field-aligned conversion `<j.B> F<1/R>/<B^2>` plus the pressure-driven term
`p'(<R> - F^2<1/R>/<B^2>)` recomputed from that pass's own `p'` (identity
I2) -- so the Redl bootstrap enters in the solver-consistent convention, not
the legacy `<j.B>/(F<1/R>)` (about 6 % high at the peak), and a pressure change
is never booked as inductive current. The structured closure then meets the
rows (Ip; l_i hard at 1e-3 for a g-file, soft σ 0.04 for an IDS; optional q0
at like radii; optional E_r-corrected MSE chords), each carrying the
discrepancy measured on the previous solved equilibrium, and ONE GS solve is
taken per pass. Convergence uses only the existing tolerances (plus the
closure-half current gate as a standing criterion); the delivery solve is
checked on every row and is the reconstruction. Stage 2 builds the baseline
only: the draws refuse an engine baseline until they run on the engine.

## The pressure handed to the solver: separatrix pressure and the edge P′ pin

Every Grad-Shafranov solve in the package hands the solver two things about
the pressure: a `P'` profile on `psi_N` and an axis-pressure target. One
module builds both (`bouquet/edge_pressure.py`), for every path -- the legacy
reconstruction and draws, the modelling-source forward solve, the
zero-perturbation checks, the unified engine and its draws.

**The convention of the solver.** The solver builds the pressure by
integrating `P'` inward from the plasma boundary starting at ZERO, then
rescales `P'` so that the axis value equals the target. Its `P'` is a
piecewise-linear function of `psi_N` that is zero outside the plasma and may
take any value at `psi_N = 1` (`P'` then jumps to zero across the boundary).
So a non-zero `P'` at the boundary is representable; a non-zero pressure
there is not, and need not be: only `P'` enters the Grad-Shafranov equation.

Two settings follow, both on `GenerationConfig`. At the PRE-CHANGE settings
(`edge_pprime_pin=True`, `separatrix_pressure="legacy"`) every path hands
the solver what it did before the settings existed, bit for bit (proven by
frozen-copy tests). Since 2026-10-02 the default of `separatrix_pressure` is
`"offset"`, so the defaults are no longer the pre-change settings:

| setting | default | other value |
|---|---|---|
| `edge_pprime_pin` | `True` (pre-change): the last node of `P'` (`psi_N = 1`) is set to zero, so `P'` ramps linearly to zero across the final grid interval | `False`: the last node keeps the profile's own derivative |
| `separatrix_pressure` | `"offset"`: the axis target is `p_axis - p_sep`, and `p_sep` is added back wherever pressure, beta or stored energy is reported or delivered | `"legacy"` (pre-change): the axis target is the full axis pressure `p_axis` |

`p_sep` is the pressure handed to the solver at its last node -- the TOTAL
solve pressure (thermal + impurity + fast, and the pressure anchor where a
path uses one), defined once in `edge_pressure.separatrix_pressure_of`. Each
draw uses its own, from its own perturbed pressure.

**What `"legacy"` does when `p_sep` is not zero.** The solver's pressure is
zero at the boundary, so it reaches the full axis target only by inflating
`P'` everywhere by `p_axis / (p_axis - p_sep)`. The equilibrium is then that
of the pressure `p_axis (p - p_sep) / (p_axis - p_sep)`: too steep by that
factor, and its `beta` and `W_MHD` are neither the input's full-pressure
values nor its `p - p_sep` values. `"offset"` hands the solver the axis
target `p_axis - p_sep` that the input's own `P'` integrates to, so the
solver's rescaling of `P'` to meet its target is reduced to the
discretisation of that integral (not exactly 1: estimated 1.017 at 129
nodes and 1.005 at 257 on a smooth synthetic profile) instead of
`p_axis / (p_axis - p_sep)`. That is a PHYSICS change relative to
`"legacy"`: `P'`, the pressure-driven current and the Shafranov shift move
by about the factor `(p_axis - p_sep) / p_axis`.

**Why `"offset"` is the default (owner-approved change, 2026-10-02).** On
real g-file and IDS cases (pin on), `"offset"` brought the full-frame
`beta_N` and `W_MHD` closer to the input on every comparable g-file case, by
1.4-8 points (median 3.5), and by about 0.5 points on IDS slices. Every
case converged under it, with the same passes, solves and wall time, and
`l_i`, `q` and the current distances did not move. The solver-frame gaps
grew (median 1.4 points on g-files): under `"legacy"` the inflated `P'` had
been compensating a deficit in that frame. For an existing run with
`p_sep != 0` the change means: `P'` in the solve is scaled by
`(p_axis - p_sep) / p_axis`, and the reported pressure, `beta`, `W_MHD` and
the delivered `PRES` move toward the input's full-pressure values. With
`p_sep = 0` nothing changes. `separatrix_pressure="legacy"` reproduces the
pre-change numbers; a stored config that predates the setting reloads with
`"legacy"`.

**What the edge pin does.** With the pin on, the pressure gradient is
truncated in the last grid interval, and the pressure-driven part of the
current, `P' (<R> - F^2 <1/R> / <B^2>)`, is forced to zero at the boundary.
With it off the profile keeps its pedestal gradient to the separatrix. For
the same requested `<j_phi>` this changes how the edge current is split
between the `P'` and `FF'` terms in the last interval (`FF'` carries less,
and can change sign there), the pressure-driven current at the boundary, and
with them the edge current and `q95`. It is a PHYSICS change when turned off.

**Reporting under `"offset"`: two frames, compared like for like.**

- *solver frame* -- the solver's own statistics, built from `p - p_sep`: what
  the equilibrium responds to. Compare with an input's `p - p_edge`
  quantities (a magnetics-only input usually has zero edge pressure already).
- *full frame* -- with `p_sep` added back: what a kinetic input reports.
  With `V` the plasma volume and `int p dV` of the solved equilibrium (both
  the solver's own numbers: `vol` and `W_MHD / 1.5`),

  ```
  W_MHD(full)  = W_MHD(solver) + 1.5 p_sep V
  beta_X(full) = beta_X(solver) * (int p dV + p_sep V) / int p dV      X = p, t, N
  P_ax(full)   = P_ax(solver) + p_sep
  ```

  (every beta of the solver is `2 mu0 <p> / B_ref^2` with the same reference
  field, so the constant adds `2 mu0 p_sep / B_ref^2`). With nothing added
  back the two frames ARE the solver's numbers.

`Baseline.edge_pressure`, the engine record (`edge_pressure`), the
reconstruction summary (`pressure_like_for_like`: each frame against the
input's same-definition quantity, a g-file's edge pressure read from its own
`PRES`) and every draw record carry `p_sep` and both frames; the archive
stores the record on `_baseline` and on every draw (`edge_pressure_json`).
The headline `beta_N` / `beta_p` / `W_MHD` of a summary are the full-frame
values under `"offset"` and the solver's own under `"legacy"`.

**Delivery under `"offset"`.** Every g-file bouquet writes carries the FULL
pressure -- the archive's `_baseline` and each draw (`generate()`), and the
reconstruction's own (`Bouquet.save_baseline_eqdsk`): `PRES` is the solver's
pressure plus that equilibrium's own `p_sep`, `PPRIME` is unchanged, so
`PRES` still differentiates to `PPRIME` and equals the input pressure at the
edge. The IMAS export is built from the delivered g-file and so carries the
same pressure; nothing downstream adds `p_sep` to a written `PRES` again. A
bare `mygs.save_eqdsk` bypasses this and writes the solver frame (`PRES`
zero at the boundary).

**Where the model stops.** A pressure that is `p_sep` just inside the
boundary and zero just outside is not physical: the real separatrix pressure
continues into the scrape-off layer, which a vacuum-outside free-boundary
equilibrium cannot represent. The constant `p_sep` exerts no force in the
model -- the equilibrium inside the boundary is the one the input's `P'` asks
for -- and is bookkeeping for readers of the pressure. The same holds for the
`P'` jump at the boundary with the pin off: it is the truncation of a
gradient that in reality continues outward.

**Not covered.** The solver's own bootstrap helper (`solve_with_bootstrap`,
used by the legacy non-loop routes for their intermediate bootstrap
evaluation) builds its own `P'` and axis target inside the solver package
and is not reached by either setting (it keeps the pre-change ones; a
printed note says so whenever the settings are not the pre-change ones --
the default included); the states a run delivers are solved by bouquet's
own calls, which are. The g-file READER's edge extrapolation of
`PPRIME` / `FFPRIM` is a separate, unchanged option.

## Kinetics regridding

Measurement-grid kinetic profiles are resampled onto the equilibrium `psi_N`
grid with **knot-passthrough PCHIP**, not linear interpolation. Linear
interpolation put staircase steps into the profiles, which the Sauter model
differentiates — producing a ~10× larger step-energy artifact in j_BS. PCHIP is
shape-preserving (no new extrema) and passes exactly through the measurement
knots, so the gradient inputs are clean without smoothing away real structure.

The same regridding applies wherever a kinetic quantity crosses grids
(`psi_N_kinetic` → `psi_N`), including the dual-grid case where the kinetic
profiles extend past ψ_N = 1 into the SOL: GPR sampling includes the SOL,
equilibrium solving uses the confined region only.

## Edge-profile classification

`classify_jphi_profile` labels the baseline current profile `H_mode`,
`Lmode_like_jphi`, or `L_mode` and reports edge-spike alignment metrics. Peak
detection uses a **two-pass valley height reference** rather than the
innermost-surface value, which is numerically fragile (it is exactly the
collapsed axis point that caused the hollow-core bug). A profile with
significant bootstrap but no detected edge peak now retains the full Sauter
split instead of having it zeroed.

## Hybrid kinetics on the IMAS path

`GenerationConfig.kinetic_source` selects where the IMAS-path baseline kinetics
come from:

| Value | ne / Te / Ti / ω_tor | Z_eff, Z_imp, n_i dilution | Currents, equilibrium, p_fast, anchors |
|---|---|---|---|
| `"fuse"` *(default)* | FUSE `core_profiles` | FUSE | FUSE |
| `"ida_hybrid"` | IDA `.cdf` fits, PCHIP-resampled onto the FUSE ψ_N grid | FUSE | FUSE |

`Bouquet.from_imas(..., ida_path=…)` selects `"ida_hybrid"` automatically and
also points `UncertaintyConfig.ida_path` at the same file, so the sigma
envelopes come from the measured fits rather than flat scalars. The optional
`LCFS_geqdsk` argument replaces the source boundary outline with the separatrix
from a supplied g-file — use it when you have a better boundary for the slice
than the dd carries, typically a magnetics-only equilibrium reconstruction.

Note that `anchor_pressure_to_equilibrium` defaults to `False` for exactly this
reason: with IDA-hybrid kinetics, anchoring to `equilibrium.pressure` would
force the trusted IDA pressure back onto the FUSE total.

## Current and field orientation

bouquet works in **one positive-current frame**: every TokaMaker anchor is
solved to `|Ip|` with `F0 = |R·B|` (`read_imas_geometry`, and `abs(R_center·B_center)`
on the g-file path), whatever orientation the source was written in. Every
bootstrap bouquet recomputes on that anchor — the legacy `solve_with_bootstrap`
path and the self-consistent loop's `evaluate_jBS` alike — therefore comes out
positive, and so does everything built from the solve.

**IMAS sources.** A dd written for a reversed-current discharge (`ip < 0` in the
dd's own COCOS) carries *negative* current profiles. `read_imas_baseline`
multiplies every current it reads by `sign(equilibrium ip)`:

| multiplied by `sign(ip)` | read unchanged |
|---|---|
| `core_profiles` `j_total`, `j_tor`, `j_ohmic`, `j_bootstrap` | kinetics (`n`, `T`, `Z_eff`), fast and equilibrium pressure |
| every beam-source `j_parallel` (→ `j_NBI`) | rotation (`omega_tor`), `E_r`, transport coefficients |
| `equilibrium.profiles_1d.j_tor` (→ `jphi_diff`) | the dd's own `q` (`q0_dd`, recorded raw; the sawtooth gate reads `|q0_dd|`) |
| `pf_active` coil currents read as coil-regularisation targets (`coil_targets.measured_from_pf_active`) | `pf_active` per-coil sigma (`data_error_upper`, used through `abs()` by the χ² coil filter) |
| | a user-supplied `FixedComponentsConfig.j_NBI` / `j_RF` — defined in bouquet's positive-Ip frame (co-current positive), exactly as on the g-file path |
| | the boundary outline, `F0 = |r0·b0|` |

The factor is recorded as `Baseline.source_current_sign` (with where it came
from as `Baseline.source_current_sign_origin`, and the source's B0 sign as
`Baseline.source_b0_sign`), in `li_metrics` and `ip_closure`, and on the
archive's `_baseline` group; a reversed source is also logged. For `ip ≥ 0` the
factor is exactly `+1.0` and the read is bit-identical to what it always was.
For a mirrored source the Baseline is **bit-identical** to the original's — the
factor is exact in IEEE arithmetic and every operation the reader applies to a
current afterwards (the `j_tor/j_total` conversion ratio, the inductive
residual, the `jphi_diff` difference) is odd in it — so the whole downstream
chain (closure, q0 target, bootstrap loop, draws) sees identical inputs.

Before this normalisation the reader kept the dd's sign: on a reversed source
the inductive and fixed currents stayed negative while the recomputed bootstrap
was positive, so the bootstrap was added *against* Ip.
`utils.closure_sign_convention` re-signed the Ip target and the affine P′
constant but never the bootstrap, and `unrenormalise_q0` mixed the two frames
into a negative q0 target. `closure_sign_convention` stays in place as a guard
(`current_direction_sign` in the closure record is read *after* the
normalisation and is `+1` on any dd whose currents agree with its own `ip`; a
negative value is printed as a warning).

**Coil-current targets.** The positive-frame solve of a reversed-Ip discharge
is the mirror image of the lab plasma, so its equilibrium coil currents are the
lab ones with the sign reversed. `coil_targets.measured_from_pf_active`
therefore multiplies every measured circuit current by the same factor
(`sign(ip)` of the same dd at the same slice the reader takes it from, or its own explicit
`current_orientation=+1/-1`) and returns it with the factor recorded;
`coil_reg_from_measured` copies the factor onto each `SolverConfig.coil_reg`
term as `"source_current_sign"` without applying it again, so it reaches the
archived `config_json`. `Bouquet._apply_coil_reg` refuses a term whose recorded
factor disagrees with the IMAS baseline's `source_current_sign` (e.g. an
`ImasSource.current_orientation` override the coil read did not share): pinning
at W0 = 100 toward the mirror-image field would distort or fail the solve.
Hand-built terms and plain-dict targets claim no orientation and are not
checked.

**Mixed-orientation sources are refused.** A dd whose current profiles and
plasma current were written in *different* orientations (for example an IDS
conversion that mixed COCOS between the equilibrium and core_profiles IDSs)
cannot be put into one frame by `sign(ip)`: it would flip currents that were
already co-Ip. The reader therefore raises `ValueError` — naming each quantity,
its stored sign, and `ip` — when the *net* toroidal current of
`core_profiles.j_tor`, or of the equilibrium `j_tor` that the `jphi_diff`
anchor uses, disagrees in sign with the orientation factor. The net current is
the area-weighted integral on the quantity's own grid: the file's `area`
(`core_profiles.grid.area`, or `equilibrium.profiles_1d.area` interpolated in
ψ_N) when present, else `rho_tor_norm²` as an area proxy, else ψ_N (no
geometry on file). Every one of these weights is monotone in the enclosed area,
so a single-signed profile — the only kind a whole-profile orientation mismatch
produces — is classified identically by all of them; they can differ only for a
profile with a genuine sign reversal of comparable weight, and the refusal
quotes the weighting used. A profile with a local counter-current region (a
current hole) is accepted as long as its net current is co-Ip.

A user who knows the file's convention names the factor with
`ImasSource.current_orientation` — `"auto"` (default: `sign(ip)`), `+1` (keep
the stored currents) or `-1` (reverse them). The normalised currents must still
integrate co-Ip; an override that leaves them counter-Ip is refused the same
way. A dd whose equilibrium and core_profiles currents disagree with *each
other* has no single factor: set
`GenerationConfig.anchor_jtor_to_equilibrium=False` so the equilibrium `j_tor`
is not used, or fix the file. The factor's origin is recorded as
`Baseline.source_current_sign_origin`, in `li_metrics` and `ip_closure`, and
on the archive's `_baseline` attrs.

The orientation is read at the equilibrium slice nearest the core_profiles
slice the currents come from (the two IDSs choose their slices independently;
on a common time base, as in FUSE output, it is the requested slice). A zero or
unreadable vacuum `b0` carries no orientation and is recorded as
`source_b0_sign = None`.

**What bouquet delivers.** Every delivered equilibrium is in the positive frame,
for every source, normal or reversed:

- **g-files** are written by TokaMaker's `save_eqdsk` with its default COCOS 7
  (bouquet passes no `cocos`) from the positive-frame solve, so they carry
  `CURRENT > 0` and `BCENTR > 0` for every source — and therefore **always
  `Ip·Bt > 0`**. They carry **neither** the experiment's Ip sign **nor** its
  toroidal-field sign: a normal-orientation source with `B0 < 0` (`Ip·B0 < 0`)
  is also delivered with `Ip·Bt > 0`, the opposite field-line helicity to the
  lab; a reversed-Ip source with `B0 < 0` happens to match the lab. This is
  what the writer does today, unchanged by this work; which convention the
  delivered g-file *should* carry is an open decision, not settled here. Being
  COCOS 7, ψ decreases outward (`SIMAG > SIBRY`), which a reader assuming
  COCOS 1 must detect.
- **archived currents** (`j_phi`, `j_BS`, `j_inductive`) are positive-frame.
- **plots** overlay raw source currents in the same positive frame as the
  solve: on the IMAS path the dd `j_tor` times the reader's factor (the
  configured `ImasSource.current_orientation`, or `sign(ip)` under `"auto"`;
  from an archive, its stamped `source_current_sign`), on the
  g-file path the g-file `<j_tor>` (read in the source's declared COCOS) and
  `FF′` times `sign(CURRENT)` — so a reversed-Ip input is not drawn upside
  down. The reconstruction itself fits `abs(Ip)` and `abs(<j_tor>)`; `abs()`
  also folds any genuine local sign change, which the overlays keep.
- **IMAS export** (`write_imas_draw` / `export_imas_drawset`) is written in the
  **source's** frame. The template (the source dd) keeps its own
  `core_sources`, `pf_active`, `vacuum_toroidal_field`, rotation and
  `core_profiles.global_quantities`, so every field the writer overwrites is
  taken back to the source orientation: `ip`, ψ (1-D, 2-D, axis, boundary), P′,
  FF′ and the `core_profiles` currents by `s_I` (the archive's
  `source_current_sign`, or `sign(template ip)` for an unstamped archive), `f`
  by the sign of the template's `b0`, and q in the template's own q-sign
  convention (`s_I·s_B` when it carries no q). A template whose ip or b0 sign
  contradicts the archive's stamp is refused. Re-reading an export therefore
  gives the same currents as the un-mirrored source's export. For `ip > 0`
  nothing changes except that `f` now takes `b0`'s sign. Two pre-existing
  limits remain: ψ / P′ / FF′ are TokaMaker's COCOS-7 eqdsk values written
  without a COCOS conversion, and the written `profiles_1d` has no `j_tor`, so
  a re-read needs `anchor_jtor_to_equilibrium=False`.

Intrinsic axisymmetric MHD quantities (the equilibrium, l_i, q magnitude, and
the Δ′ / δW of the plasma on its own) do not depend on the frame: flipping Ip
alone is a mirror reflection (φ → −φ) of the plasma and flipping both is a full
field reversal, and both are symmetries of the MHD equations. What does
depend on the frame is anything with a *direction relative to the lab*:

- toroidal rotation, `E_r`, diamagnetic and E×B frequencies — passed through in
  the source's own lab signs and **not** transformed into the delivered frame;
- the **field-line helicity**, sign(Ip·Bt), relative to external coils. The
  response to 3D fields — error-field and 3D-coil coupling computed with real
  coil geometry and phasing, resonant field penetration, NTV — depends on it.
  Since a delivered g-file always has Ip·Bt > 0 (above), a 3D-response
  calculation on it describes the lab only for sources whose own Ip·B0 > 0.

A consumer that combines a delivered equilibrium with flows or with lab-frame
3D coils must therefore either restore the source orientation or transform
those inputs into the delivered frame. Restoring means, with
`s_I = source_current_sign` and `s_B = source_b0_sign`:
`Ip → s_I·Ip`; ψ → s_I·ψ (1-D ψ, `PSIRZ`, `SIMAG`, `SIBRY`); **P′ → s_I·P′ and
FF′ → s_I·FF′** (both are ψ-derivatives: P′ = dp/dψ flips with ψ; FF′ = F dF/dψ
flips with ψ and, F and dF flipping together, not with B0); `F → s_B·F`,
`Bt → s_B·Bt`; `q → s_I·s_B·q`. bouquet does this automatically only in the
IMAS export (above); archived g-files are left in the positive frame.

## Z_eff-primary density scheme

Each draw perturbs {n_e, T_e, T_i, Z_eff} and *derives* the main-ion density
from quasi-neutrality with a single effective impurity charge (`impurity_Z`,
carbon 6.0 by default — set it for your device). n_i, n_z, and Z_eff therefore
stay mutually consistent in every sample, which a naive independent-perturbation
scheme cannot guarantee. One Z_eff value per draw. See
[architecture.md §4](../architecture.md#4-quasi-neutrality-and-impurity-handling).

## Corrective j_phi iteration

TokaMaker's `jphi-linterp` mode imposes the requested `j_phi(psi_N)` using
pre-solve geometry, so the *achieved* flux-surface-averaged j_phi drifts from
the request once ψ converges. On the reconstruction path an adaptive Newton
iteration (2–8 steps) drives the achieved profile back onto the target.

On the IMAS path the same correction is available as
`imas_corrective_jphi=True` but is **off by default** while it is validated:
the single-pass solve there measures a +3.2–3.4% achieved-j_phi offset, which
surfaces as l_i +3.4% and q(ψ_N) −5% against the source IDS. The archived
j_phi is always the *achieved* TokaMaker profile on both paths, not the
requested one.

A related, accepted artifact: a localized ~8–10% dip in core j_phi relative to
the input g-file, which is an l_i-versus-peakedness tradeoff intrinsic to
matching both. Pinning the core has been tried and is unstable. See
[architecture.md §16](../architecture.md#16-known-limitations-and-future-work).

**Known error, kept for legacy bit-identity: index-for-index ψ_N readbacks.** On a uniform ψ_N grid (every g-file run), the legacy path samples the solver at its own padded points, `linspace(psi_pad, 1 - psi_pad, n)`, and pairs those samples index for index with profiles on the nodes, `linspace(0, 1, n)`. Each pairing is misplaced by up to `psi_pad`, most of all at the edge.

Sites:
- the corrective iteration's measurement (`_corrective_output_jphi`, `coords.readback_kw`'s uniform branch);
- the cylindrical l_i proxy (`calc_cylindrical_li_proxy`);
- the self-consistent loop's delivered state (`_deliver_request_split`).

This is wrong, and it is kept only so that legacy ψ_N results and their goldens stay bit-identical with main. On the D3D-like g-file it costs q95 −0.48% against the g-file's own q; with the readbacks moved to the nodes, the same run is −0.036% off, and l_i(3) moves from 0.65594 to 0.65397. The unified engine, swb, Φ_N runs and the archived achieved current all sample at, or interpolate onto, the nodes, and are not affected.

A separate known issue — the small constant boundary offset from `jphi-linterp`
edge/separatrix handling that sets the ~0.5 mm σ=0 floor — is written up in
[ISSUE_jphi_edge_reconstruction.md](ISSUE_jphi_edge_reconstruction.md).

---

## The structured closure and its l_i constraint

On the IMAS hybrid path (`jBS_baseline_mode="ohmic"`) the three current
components — the source's inductive current, the recomputed Sauter bootstrap
and the fixed auxiliary drive — generally do not add up to the measured I_p,
and something has to absorb the difference. The scalar channels multiply one
component by one number. `closure_channel="structured"` instead makes each
multiplier a smooth radial **profile** on a small basis,

```
s_ind(ψ) = 1 + Σ_k a_k φ_k(ψ),    s_bs(ψ) = 1 + Σ_k b_k φ_k(ψ),
```

and returns, among all coefficient vectors that satisfy the constraints
exactly, the one that departs least from "trust the sources" (s ≡ 1) in a
trust-weighted norm. Everything is linear algebra on the same affine
flux-surface I_p measure the scalar channels use, so it costs one small dense
solve and **zero** Grad–Shafranov solves.

### Why l_i

I_p is *one* number against 2K coefficients. It constrains the integral of the
correction and says nothing about where in radius it goes — which is exactly
what the closure-cloud campaign found the scalar channels get wrong (they pay
for I_p closure out of the pedestal bootstrap). `l_i` is the other global
number a magnetics reconstruction reports, and it is sensitive to precisely
that redistribution. Set `structured_li_target` and it becomes a second
constraint row.

The target is a **plain input**. bouquet does not fetch it, does not know which
code produced it, and applies no cross-code definition offset — whatever offset
the caller's l_i carries must already be in the number handed in.

### The discrete form is exact, not a proxy

With ψ the per-radian poloidal flux (B_p = |∇ψ|/R, TokaMaker's ψ), Ampère's law
on a flux surface reads

```
μ0 I(ψ) = ∮ (|∇ψ|/R) dl = (V'/2π) ⟨|∇ψ|²/R²⟩ = (V'/2π) ⟨B_p²⟩,
```

so ⟨B_p²⟩ = 2π μ0 I(ψ)/V'(ψ) and

```
∫ B_p² dV = ∫ ⟨B_p²⟩ V' dψ = 2π μ0 ∫ I(ψ) dψ  ≡  2π μ0 S.
```

The flux-surface geometry cancels identically: the poloidal field energy of any
axisymmetric equilibrium is fixed by its enclosed-current profile alone. With
`dl` the LCFS poloidal perimeter, `vol` the plasma volume and `R_axis` the
magnetic-axis major radius — the three numbers TokaMaker's own `get_stats`
uses, read off the same calls at the same `lcfs_pad` —

```
l_i(1) = 2π dl² S / (μ0 · vol · I_p²)        ("std" in get_stats)
l_i(3) = 4π S / (μ0 · I_p² · R_axis)         ("iter" in get_stats)
```

Both are (prefactor) × S / I_p², i.e. linear in the enclosed-current profile
over the square of the total — and S is linear in the structured coefficients
at frozen anchor geometry, exactly as I_p is. That is what makes the constraint
cost nothing.

**Validation** (`tests/test_li_closure.py`, `pytest -m solver`, on the
synthetic D3D-like anchor): integrating the equilibrium's own GS current
profile, the identity reproduces TokaMaker's `Bp_vol` to **+0.0208 %** and both
`get_stats` l_i normalisations to **+0.0215 %**, against a 1 % acceptance bar.

### Which perimeter — the one calibration that is not optional

`l_i(1) ∝ L²`, so the LCFS perimeter is not a detail. TokaMaker's `get_stats`
takes `dl` from `get_q`; that reading sits **above** the perimeter of the traced
`1 − psi_pad` contour — the surface `save_eqdsk` writes as `RBBBS`/`ZBBBS` — by
+0.8 % on the synthetic anchor and +1.6 % on real diverted DIII-D equilibria,
i.e. **0.013 to 0.032 of l_i(1)**. An EFIT a-file `LI` is l_i(1) normalised
through the Ampère-law mean field μ0 Ip / L_LCFS, so a target taken from a
reconstruction lives on the contour perimeter; closing on `get_stats`' `dl`
would charge the closure with core current peaking it did not do.

So `utils.lcfs_perimeter` measures the contour itself and `utils.li_achieved`
reads the solved equilibrium back the same way (TokaMaker's exact `Bp_vol`,
`vol`, `Ip` from `get_globals`, normalised with that perimeter) — never
`get_stats('std')`. Both are recorded alongside `get_stats`' `dl` and their
ratio, so the offset stays visible.

Calibrated against an independent out-of-tree g-file l_i estimator — one
estimator run on both equilibria — on the anchor's saved g-file:

| quantity | value | vs the g-file estimator |
|---|---|---|
| g-file estimator l_i(1) | 0.838671 | — |
| closure l_i(1) (this identity) | 0.839054 | **+0.00038** |
| `li_achieved` l_i(1) readback | 0.838873 | **+0.00020** |
| `get_stats('std')` | 0.852039 | +0.01337 |
| g-file estimator l_i(3) (R₀ = R_axis) | 0.655075 | — |
| closure l_i(3) | 0.656377 | +0.00130 |
| `get_stats('iter')` = `li_achieved` l_i(3) | 0.656236 | +0.00116 |

l_i(3) carries no perimeter, so it needs no correction: `get_stats('iter')` is
used unchanged. The archived baseline total — a slightly different current
profile from the equilibrium's own — lands at −0.1506 % of `get_stats`.

### Hard and soft

Two statements about the data, one prior:

* **hard** (`structured_soft=False`, the default): I_p, the optional on-axis
  current row and l_i are imposed *exactly*, through one bordered KKT system.
  Because the I_p row pins I_p, l_i is linear in the coefficients and the l_i
  row is **exact**, not merely linearised — the linearisation is taken along
  the I_p-closed manifold. (Linearising about the anchor's own I_p instead
  leaves an (I_p⁰/I_p)² factor in the row, worth ~0.3 % of l_i on a 4 % I_p
  deficit.)
* **soft** (`structured_soft=True`): I_p and l_i become Gaussian measurements
  with `structured_ip_sigma` / `structured_li_sigma` and the answer is the
  posterior mode of

  ```
  Σ a_k²/σ_ind,k² + Σ b_k²/σ_bs,k² + (I_p[x]−I_p^t)²/σ_Ip² + (l_i[x]−l_i^t)²/σ_li²
  ```

  over the same eight unknowns — Gauss–Newton with Levenberg damping, no GS
  solves, converged to 1e-10 relative. When no damped step can be verified
  downhill, the iterate is accepted only if the scaled gradient is below
  `rtol·max|J|·max(√F, 1)` (`stop_reason="gradient_floor"`), or if it is
  stationary to within the objective's **rounding noise**
  (`stop_reason="noise_floor"`): the gradient below the floor that noise implies
  and every Levenberg trial's predicted decrease below
  `noise_F = 2 ε Σ_i (2|r_i| m_i + r_i²)` (`m_i` = the magnitude of the terms
  row i is computed from, in σ units — an MA-scale I_p difference over a
  kA-scale σ_Ip is what makes it exceed the `F·1e-14` acceptance slack).
  The noise-floor acceptance applies **only with
  `accept_noise_floor=True`**, which the self-consistent bootstrap loop's
  closure calls pass; the default (`False`, every frozen-path call) is the
  historical strict solver. An acceptance is printed as well as recorded, and
  a non-finite or non-positive noise estimate accepts nothing.
  Otherwise it refuses, as before. Inside the self-consistent loop a refused
  soft closure is retried ONCE from the previous pass's coefficients
  (`closure_retry=1`, logged); a second refusal is a real one. The prior is the *same* one: σ = W^(−1/2)
  of `structured_weights`, so a hard/soft pair differs only in what is claimed
  about the data. The on-axis-current row stays hard in both (a q0 pin is a
  topological statement, not a measurement with a σ). `structured_ip_sigma=None`
  keeps I_p exact inside the soft solver.

  With a finite σ_Ip the closed hybrid integrates to a **posterior** I_p a
  little away from the measurement — that is the channel working, not an
  error. The post-closure round-trip check is an *algebra* check (does the
  assembled hybrid carry the current the closure said it would?), so on this
  channel it compares against that posterior, recorded as
  `structured_ip_posterior`; its budget (0.05 %, `utils.IP_ROUNDTRIP_TOL_PCT`)
  is unchanged. The distance from the measurement is reported instead —
  `structured_ip_measured_residual_pct` and, in σ units,
  `structured_residual_sigma_Ip` (achieved, and refreshed by the corrector,
  which re-solves the same posterior rather than snapping I_p back) — and past
  1 σ_Ip the slice picks up the closure-health flag *"soft Ip beyond 1 σ_Ip"*.
  It is never refused and never retried on that basis. At σ_Ip = 0.5 % of I_p
  the posterior typically lands 0.02–0.1 % from the measurement.

Both refuse — never clamp — when a multiplier would leave `0.2 < s < 5`
(strictly: a multiplier landing exactly on a bound is refused) anywhere on the
grid, when the constraint system is degenerate against a relative floor (which
includes carrying more constraint rows than the basis has free coefficients —
e.g. the `{"kind": "constant"}` one-liner with both an axis row and an l_i
target), or (soft) when the posterior mode does not converge.

### One correction, shared

The predictor is exact in the coefficient algebra but runs on the *frozen*
anchor geometry. The closed-hybrid GS solve is the first time the real q0 and
l_i are knowable, and reading them costs nothing. If both are inside their
tolerances (`q0_tol`, `structured_li_tol`) the slice is done at **zero** extra
solves; otherwise the constraint rows' right-hand sides are corrected by models
calibrated on the measured pair and inverted exactly,

```
j_ref0' = j_ref0 · (q0_solved / q0_target)                  (q0 ~ 1/j_φ(0))
li_row' = li_row · (li_target / li_achieved)^(1/p),  p = 2
```

and the whole minimal-norm (or posterior-mode) problem is re-solved once. When
both miss, they are corrected *together* in that single re-solve: the ceiling is
one extra GS solve per slice, not one per constraint. There is deliberately no
*open-ended* loop — a residual that is reported is worth more than a residual
iterated away invisibly. The ceiling is a user-visible count
(`structured_li_max_corrector_steps`, default **1**; the conditional second
step is described below), never an "iterate until it converges", and every
predictor/achieved/corrected value, every residual in σ units and
`n_extra_solves` land in `Baseline.ip_closure`. A q0 or l_i that the delivered
equilibrium still misses is **flagged** there (`closure_limited_reasons`), and
so is a readback that came back non-finite — flagged, never retried, and
`q0_tol` / `structured_li_tol` are untouched by either.

**Why the square root** (`utils.LI_GAIN_EXPONENT = 2`). The obvious update
inverts a *proportionality* — assume the achieved l_i follows its row with gain
≈1 and step by the ratio. It does not: `li_closure_geometry` freezes
`dψ/dψ_N = |ψ_axis − ψ_bnd|` at the anchor, but the solved equilibrium's own
`Δψ` moves with the internal inductance, `li_achieved/li_model =
Δψ_solved/Δψ_anchor` to ~1 %. With `Δψ ∝ √l_i`, l_i enters the achieved
equilibrium **twice** — once through the enclosed-current shape integral the
closure prescribes, once more through the flux span that responds to it — so
`achieved ∝ row²` and the correct inverse is the square root of the ratio. `p`
is parameter-free and comes from that argument; a campaign-calibrated exponent
was measured (≈2.17 hard) and deliberately **not** adopted, since it is a
constant fitted to one device's ohmic slices. The proportional update, assuming
a gain of ~0.96 where the measured gain is ~2.14, overshot by a factor ~2.2 and
flipped the residual's *sign* on every corrected slice of the campaign.
Multiplicative rather than additive because l_i is positive and the correction
must stay scale-free. `structured_li_tol` is untouched by any of this — the
gain law changes the model, never the acceptance criterion.

`structured_li_max_corrector_steps` (default **1**, the shipped one-solve
ceiling) may be raised to 2 for a *conditional* second step on slices whose
corrected l_i still misses tolerance. By then two measured `(row, achieved)`
pairs exist on that slice, so the second step reads the slice's **own** log-gain
off them (`utils.li_gain_exponent_secant`) instead of assuming one — a secant
iteration with no fitted constant, i.e. a solver-side remedy that changes the
cost and not the criterion.

### The prior, as σ

`structured_weights` are Tikhonov weights, but the quantity to reason about is
the width: **W = 1/σ²**, and `utils.sigma_from_weights` / its inverse are the
bridge, so the hard and soft solvers are handed the *same* prior and a
hard/soft pair differs only in what is claimed about the data. `σ = 0` hard-pins
a coefficient to zero; `σ = ∞` leaves it unpenalised.

A σ ladder says, radius by radius, how far each source is allowed to move
before the objective objects. The shipped physics prior and the recommended
preset both run the two ladders in **opposite directions**, for different
reasons:

| entry | σ_ind (inductive) | why |
|---|---|---|
| core (ψ_N ≈ 0.15) | tight (0.10) | wherever the sawtooth gate admits an on-axis current row, the core inductive current is already *pinned by that row*; leaving it loose would only let the objective relitigate a constraint |
| mid-radius (≈ 0.45, 0.75) | loose (0.40) | this is where I_p is blind and where the closure is meant to work |
| mantle (≈ 0.95) | loose (0.40) | same, and the edge inductive current is small enough that a wide multiplier moves little current |

| entry | σ_bs (bootstrap) | why |
|---|---|---|
| core (≈ 0.15) | loose (0.50) | the trapped fraction vanishes on axis, the Sauter/Redl collisionality expansion is at its worst there, and the bootstrap current is negligible — a wide multiplier costs almost no current and buys freedom where the formula is least trustworthy |
| ≈ 0.45 | 0.30 | interpolating |
| ≈ 0.75 | 0.15 | tightening into the gradient region |
| pedestal (≈ 0.95) | tight (0.10) | the steep-gradient pedestal is exactly where Redl/Sauter has been validated against drift-kinetic (NEO) calculations; this is the part of the bootstrap profile the closure should not be rewriting |

The inductive current is what a transport model actually *diffuses*, and is
most trustworthy where the axis row already constrains it; the bootstrap is a
local formula whose accuracy is a function of collisionality and trapped
fraction. Hence the opposite ladders. `utils.STRUCTURED_WEIGHTS_UNIFORM` (all
1) is the one documented alternative and exists to be run as a **sensitivity**:
the difference between the two answers is the part of the result the prior —
not the data — is holding up, and it is meant to be reported.

None of this is a tolerance. A prior changes *which* exactly-closing profile is
returned; it cannot change what "closed" means.

### The one-sided mid-radius inductive prior

`structured_sigma_ind_up` gives the inductive coefficients a **second** prior
width, used while a coefficient is *positive*; the ordinary ladder keeps the
negative side. A tight up-side σ therefore lets `s_ind` **fall** freely at that
radius while resisting a **rise**. `None` (the default) is the symmetric prior,
byte-identical to every run made before the field existed.

This prior is **empirically motivated**, with a supporting mechanism — not
derived from one, and the docstrings say so as plainly as this page does.

* *The empirical part.* Over the closure-cloud campaign the l_i-informed
  channels win on the over-shoot slices and **lose** on the under-shoot ones.
  The MSE chords are what says so: they penalise inductive current *added* at
  mid-radius. The sign of the mismatch, not its size, is the gate.
* *The mechanism that makes that asymmetry expected rather than a fluke.* The
  transport model's edge electron temperature collapses relative to the
  reference kinetics (ratios of 2 to 40 on the campaign). A cold edge is a
  resistive edge, so the edge resistivity runs high, current diffuses inward
  faster than it should, and the source's mid-radius inductive current is
  accordingly more likely to be **over**- than under-estimated. A prior that
  lets mid-radius `s_ind` fall freely while resisting a rise encodes exactly
  that.
* *The caveat, carried with every result.* The asymmetry was tuned on the same
  MSE chords that then judge it. That circularity is a property of the result,
  not a defect the mechanism repairs.

**How it is solved.** The objective becomes piecewise-quadratic. It stays
**convex** — the penalty is continuous at 0 with a non-decreasing derivative —
so its minimiser is unique, and a sign iteration reaches it *exactly* once the
sign pattern is stable, rather than approximately. Each step is the same linear
algebra the symmetric channel does once: **zero extra GS solves**. The iteration
starts from the all-down pattern and is capped at 8; a pattern that cycles or
will not settle is **refused loudly**, never returned. Applied identically in
both solvers (`close_ip_structured`, `close_ip_structured_soft`).

Because `σ = 0` (pinned) and `σ = ∞` (unpenalised) decide which coefficients
*exist*, the up ladder must pin and un-penalise in the same places as the down
ladder — the set of unknowns must not flip with a sign — and a mismatch is
refused. The scale-bound and finiteness checks sit deliberately **outside** the
sign iteration, so an intermediate pattern, which is not an answer, cannot
refuse a slice whose actual minimiser is in bounds.

This too is a prior. No acceptance criterion, convergence threshold, scale
bound or test bound is touched by it.

### When a slice is flagged: `closure_health`

`utils.closure_health` records the per-slice honesty of the closure on every
ohmic-mode channel and sets `closure_limited` with a tuple of reasons. It is a
**flag**, never a retry and never a refusal — refusals (a multiplier outside
`0.2 < s < 5`, a singular constraint system, a cycling sign pattern) raise before
it is reached. Downstream consumers should treat a closure-limited slice as
unvalidated.

| reason | what it means |
|---|---|
| `raw components miss Ip by …% (> …%)` | the *unscaled* components are far from I_p, so the closure is being asked for a large reconciliation however it distributes it |
| `bootstrap_scale_out_of_prior: bs_scale … outside 1 +/- 0.5 …` | the closure's bootstrap scale left the ±50 % bootstrap prior (`utils.BS_SCALE_PRIOR_HALFWIDTH`), either way: a **closure failure, not a finding**. Printed and warned loudly, never clamped. Evaluated on every path since 2026-10-06 -- the legacy IMAS channels, both engine paths (the effective scale `bs_scale_eff`, the `s_bs(ψ)` range recorded beside it; the g-file engine path records it in `reconstruction_metrics["closure_health"]`, the IDS path in `ip_closure` and `li_metrics["bootstrap_prior"]`) and the legacy g-file path (the inductive fit's scale, 1.0 unless `rescale_j_BS`). Until then only `bs_scale < 0.5` was flagged, and only on the IMAS paths |
| `soft Ip beyond 1 sigma_Ip (z_Ip = …)` | soft channel only: the **delivered** hybrid's I_p sits more than 1 σ_Ip from the measurement. A small offset is the channel working; past 1 σ it is worth seeing. The corrector *replaces* this flag rather than stacking a stale one |
| `l_i misses its hard row by … (> tol …) after … corrector solve(s)` | hard channel only: the corrected l_i is still outside `structured_li_tol`. Before this existed, `closure_health` did not look at l_i at all |

On the structured channel `ohm_scale` / `bs_scale` are the I_p-weighted **means**
of the multiplier profiles. Those satisfy the scalar closure equation exactly,
which is what keeps `closure_health` applicable to a channel that no longer has
scalars — and the record says so.

### The posterior-I_p round-trip gate

After the closure, the ohmic block re-integrates the hybrid it just assembled
and checks that it carries the current the closure said it would. That gate
(`utils.ip_roundtrip_gate`, budget `IP_ROUNDTRIP_TOL_PCT = 0.05 %`) is an
**algebra** check — a wrong component, a double-counted affine term, a
misapplied sign — and its budget has never been a statement about the data.

On every hard channel the closure's own I_p *is* the measurement, so the two
references coincide and nothing changes. On the soft channel they do not: the
posterior mode lands a little off the measurement *by design*. Comparing it
against the measurement with an arithmetic budget tested the data statement
rather than the algebra. So **the reference moved and the tolerance did not**:
the gate takes the closure's own posterior where there is one
(`structured_ip_posterior`) and keeps the same 0.05 %. A soft hybrid that misses
its *own* posterior by 0.06 % is still refused, with the same message. The
distance from the measurement is reported instead
(`structured_ip_measured_residual_pct`, `structured_residual_sigma_Ip`) and
flagged past 1 σ. Nothing anywhere snaps I_p back to the measurement.

### Recommended configuration

The shipped default remains `closure_channel="bootstrap"`. The structured
channel is **opt-in** — naming a preset does not switch the channel on.

The configuration this work converged on **is the default of that channel**:

```python
cfg.generation.closure_channel = "structured"       # -> preset li_soft_onesided
cfg.generation.structured_li_target = li_from_the_reconstruction   # optional
```

Selecting the structured channel with no `structured_preset` resolves to
`li_soft_onesided` (`utils.STRUCTURED_PRESET_DEFAULT`), because the raw shipped
fields — the symmetric physics ladder on the hard solver — are the configuration
this study *superseded*, and a default nobody is expected to want is a trap
rather than a default. Naming the preset explicitly does exactly the same thing;
the record says which of the two happened (`structured_preset_source`:
`"default"` or `"explicit"`).

**The opt-out is `structured_preset="none"`** (`utils.STRUCTURED_PRESET_NONE`):
it declines the default and leaves every structured field at its shipped value,
which reproduces, exactly, what a bare `closure_channel="structured"` did before
the default existed. Every explicit configuration that worked before remains
reachable, and the old default is one spelling away.

```python
cfg.generation.closure_channel = "structured"
cfg.generation.structured_preset = "none"           # the raw shipped fields
```

**Scope — read this before carrying the σ values elsewhere.** They were set from
a study on **one device with one integrated-modelling source** for the inductive
current. They are **priors in relative units** — fractions of the component
profiles themselves, on normalised flux — not device constants, which is why
they transfer at all; that is not a claim that they are right anywhere else. On
another device, or another source of `j_ind`, treat them as a **starting
point**: run the channel, then read the recorded closure-health flags before
trusting the answer —

* the scale bounds (a multiplier leaving `0.2 < s < 5` anywhere is refused);
* `|s_bs − 1| > 0.5`, i.e. the bootstrap rescaled past its own ±50 % uncertainty
  prior, which is a closure failure and not a finding;
* the q0 miss against `q0_tol`, where the sawtooth gate admits an axis row;
* the l_i z-score (`structured_residual_sigma_li`) and the σ_Ip residual.

`STRUCTURED_WEIGHTS_UNIFORM` is the no-prior sensitivity: the difference between
the two answers is the part of the result the prior, not the data, is holding
up, and off the original device it is the first thing to run. "No prior" holds
only without MSE data: with `mse_data` the objective gains the chords' χ², the
weights become an absolute σ⁻² that trades against it, and the uniform ladder
is a σ = 1 prior on every coefficient.

Two guards keep the default from being worse than no default:

* an explicit `structured_basis` **declines** it, with a warning and no fills —
  the ladders are widths at the shipped basis's radii and mean nothing on
  another basis (`structured_preset_source="default-declined-custom-basis"`);
  naming the preset applies it anyway, since then the caller asked;
* an explicit absolute `structured_ip_sigma` suppresses the default's
  `structured_ip_sigma_frac` fill, because the two are mutually exclusive
  downstream and a *default* may not turn a configuration that ran yesterday
  into a refusal. A preset **named** explicitly still fills the fraction and
  lets the closure refuse the clash loudly.

The preset — named, or applied by default — fills, in σ terms:

| field | value |
|---|---|
| σ_bs | `(0.50, 0.30, 0.15, 0.10)` |
| σ_ind (down) | `(0.10, 0.40, 0.40, 0.40)` |
| σ_ind,up | `(0.10, 0.10, 0.10, 0.40)` |
| `structured_soft` | `True` |
| `structured_ip_sigma_frac` | `0.005` (the magnetics' own accuracy) |
| `structured_li_sigma` | `0.04` — **only** when `structured_li_target` is set |

The σ ladders are recorded as `structured_weights` (W = σ⁻²) and
`structured_sigma_ind_up`, so the campaign record reads on the usual scale.

Three rules govern it:

1. **An explicit setting that differs from the default wins.** A preset fills
   only fields still *holding* their dataclass default value — and a field
   explicitly set to that same value is indistinguishable from an unset one,
   so it is overridden too. `structured_soft=False` alongside a soft preset
   comes back `True`; the same applies to any field set to its own default.
   Every field a preset fills is named in a `UserWarning` at construction, so
   the override is visible rather than silent. A caller who wants these
   ladders on the *hard* solver should set the σ fields directly rather than
   naming the preset, or set the held field **after** construction. This holds
   identically for the preset applied *by default* — the warning then says
   `applied BY DEFAULT`, names the same fields and gives the opt-out.
2. **A preset with no l_i target simply omits the l_i term.** No σ is recorded
   for a measurement that was never supplied — so a default-preset run without
   an l_i target degrades to exactly what naming the preset without one gives:
   soft I_p plus the one-sided prior, no l_i row.
3. **An unknown preset name is refused at construction**, never ignored: a
   preset that does not exist is a typo, and a typo that quietly left the
   shipped prior in place would be invisible in the record. `"none"` is the one
   extra spelling accepted, and it is the opt-out above.

`utils.STRUCTURED_PRESETS` is the registry and
`utils.structured_preset_settings` resolves one; it returns a fresh dict, so a
campaign runner's edit cannot reconfigure the next slice.
`config.resolve_structured_preset` applies the rules, at construction and again
at the closure's own entry point — so a config whose `closure_channel` is set
*after* the `GenerationConfig` was built (the one-liner above does exactly that)
is resolved too, rather than being left silently on the superseded fields. It is
idempotent, and it records `structured_preset_in_force`,
`structured_preset_source` and the fields it filled onto the config and into the
closure record (`structured_preset`, `structured_preset_source`,
`structured_preset_filled` in `Baseline.ip_closure`).

---

## Core-pressure hollowness record

Every baseline carries `core_pressure_hollow`, a **report-only** description
of the core shape of its pressure profile (`physics.core_pressure_hollow_record`
on top of `physics.core_pressure_health`). It sits on
`Baseline.core_pressure_hollow` and inside `Baseline.li_metrics`, so it is
archived in `li_metrics_json` next to the `ip_closure` closure-health record,
and `load_baseline_profiles` lifts it to top level. Archives written before it
existed read as before, without the key. Draws do not carry it: the per-draw
pressure is archived, and the same function can be applied to it offline.

It describes what the profile does, not why. A pressure that rises off-axis can
be physical (off-axis heating, an off-axis fast-ion population, impurity
accumulation) or can come from how the inputs were fitted or composed; the
record does not distinguish them. Nothing in bouquet reads it back: no
profile, solve, filter decision, in-spec count or until-N count depends on it.

**What is measured.** The innermost grid node stands in for the axis
(`p_ref` at `psi_N_ref`). Over the core window `psi_N ≤ 0.5`:

| field | meaning |
|---|---|
| `rise_frac` | `(max p − p_ref) / p_ref`: how far the core maximum exceeds the axis value, as a fraction of it. 0 when the axis is the core maximum |
| `psi_N_of_max`, `rise_extent` | where the maximum sits, and `psi_N_of_max − psi_N_ref`, the radial distance over which the pressure climbs to it |
| `positive_gradient_extent`, `positive_gradient_psi_N_max` | the summed psi_N width of the core intervals whose secant slope is positive, and the outer edge of the outermost one. Counts every positive secant, including grid-scale noise |
| `component_shares` | each additive component's share of the total's positive core rise (they add to 1). Descriptive only |
| `is_hollow` | `rise_frac > 0.01` — a convenience boolean |

**Where it is evaluated.** `input.total` is the pressure handed to the
solver, including fast ions and, on the IMAS path, the anchor offset.
`input.thermal` is the thermal species alone (electrons + main ions +
impurity). Note that this differs from the archived `pressure_thermal`, which
holds electrons + main ions only. `achieved.total` is the pressure the converged
equilibrium carries, read back on the same grid. `achieved.thermal` is always
"not evaluated", because the solver holds one total pressure.

**The threshold is a reporting choice, not an acceptance criterion.** The 1 %
bar (`CORE_HOLLOW_RISE_FRAC`) sits an order of magnitude above 1e-3-level
grid-scale wiggles, so a single noisy node does not set `is_hollow`. It is
stored in the record's `definition` block, and the numbers it is derived from
are always stored with it, so a reader can apply a different bar.

**Not evaluated is never "not hollow".** Non-finite values, a grid that is not
strictly monotone, fewer than 3 core nodes, a non-positive axis pressure, or an
innermost node beyond `psi_N = 0.05` give `evaluated: False`, `is_hollow: None`
and a `reason`. A failure while building the record is recorded, never raised.
When an evaluated total is hollow, a `CorePressureHollowWarning` is emitted
once. It says that nothing was modified.
