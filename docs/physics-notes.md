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
- [Kinetics regridding](#kinetics-regridding)
- [Edge-profile classification](#edge-profile-classification)
- [Hybrid kinetics on the IMAS path](#hybrid-kinetics-on-the-imas-path)
- [Z_eff-primary density scheme](#z_eff-primary-density-scheme)
- [Corrective j_phi iteration](#corrective-j_phi-iteration)
- [The structured closure and its l_i constraint](#the-structured-closure-and-its-l_i-constraint)

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
`passed`, and costs one bootstrap solve (~1 min). Call it after
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
`solve_with_bootstrap`, whose parallel output is converted to toroidal with the
flux-surface geometry factor `c = 1/(⟨R⟩⟨1/R⟩)` (`bouquet.physics.parallel_to_toroidal`).

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
  solves, converged to 1e-10 relative. The prior is the *same* one: σ = W^(−1/2)
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
| `bs_scale … < …` | the closure paid for I_p by scaling the bootstrap down past `bs_scale_min` |
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
channel and every preset of it are **opt-in** — naming a preset does not switch
the channel on.

The configuration this work converged on is one switch:

```python
cfg.generation.closure_channel = "structured"
cfg.generation.structured_preset = "li_soft_onesided"
cfg.generation.structured_li_target = li_from_the_reconstruction   # optional
```

`structured_preset="li_soft_onesided"` fills, in σ terms:

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
   naming the preset, or set the held field **after** construction.
2. **A preset with no l_i target simply omits the l_i term.** No σ is recorded
   for a measurement that was never supplied.
3. **An unknown preset name is refused at construction**, never ignored: a
   preset that does not exist is a typo, and a typo that quietly left the
   shipped prior in place would be invisible in the record.

`utils.STRUCTURED_PRESETS` is the registry and
`utils.structured_preset_settings` resolves one; it returns a fresh dict, so a
campaign runner's edit cannot reconfigure the next slice.
