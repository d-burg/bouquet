# Coil constraint handling

Forward-mode perturbed equilibria must keep coil currents close to the
reconstructed baseline to be engineering-realistic. Bouquet enforces this with
per-coil hard bounds on TokaMaker's QP, applied through a **progressive
homotopy** that warm-starts each tighter pass from the prior pass's converged
ψ. Each draw is then tagged `in_spec` against class-specific tolerances; out-of-
spec draws are still archived.

The full source-cited tolerance budget, bound construction, and homotopy
failure/rollback semantics are in
[architecture.md §15](../architecture.md#15-coil-constraint-handling-diii-d-reference).
This page is the user-facing summary plus the VSC drift metric, which is
specific enough to warrant its own writeup.

> **Which rule `Bouquet.filter()` applies changed.** The bounds and the
> `in_spec` tag described on this page are unchanged, but the *postprocess*
> acceptance is now the measurement-referenced χ² test by default
> (`FilterConfig.coil_filter = "chi2"`), not the ±`inspec_*` band described
> below. The band is still one setting away — `filtering.coil_filter =
> "legacy"` — and `filter_coil_currents` still applies it directly. Read
> [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) before comparing an in-spec fraction
> against one produced before that release: the two rules select different
> subsets of the same archive, and on an L-mode ensemble the χ² rule is often
> the *more permissive* of the two.

## Three coil classes (DIII-D reference)

| Class | Members | Baseline range | Spec interpretation |
|---|---|---|---|
| Non-VSC F-coils | F1A–F8A, F1B–F8B | 35–180 kA | Tight relative drift (`FilterConfig.inspec_F_max`, default `0.02`) |
| VSC pair | F9A, F9B | 35–50 kA | Channel-decomposed drift (`FilterConfig.inspec_VSC_max`, default `0.02`) — see below |
| E-coils | ECOILA, ECOILB | 0.5–1 kA | Bounded by an absolute floor (`coil_drift_floor_A`, default 50 A) — excluded from `in_spec` because their small baselines make relative bounds engineering-meaningless |

The **50 A floor** is calibrated to a realistic current-control tolerance
budget: ~30–50 A power-supply ripple + ~30–50 A Rogowski/integrator noise +
~10–100 A un-modelled vessel coupling. `coil_drift_floor_A` is a
`generate_bouquet` keyword rather than a `BouquetConfig` field.

## VSC drift metric (anti-series pair)

F9A/F9B are an **anti-series VSC pair**: they move together as a common-mode
current `±ΔI` that controls the plasma's vertical position. On vertically
sensitive equilibria one of the pair routinely sits near a **current
zero-crossing** (e.g. F9A ≈ −16 kA while F9B ≈ +93 kA). The naive per-coil
metric `|ΔI| / |I_baseline|` then divides a *benign* common-mode current by
F9A's near-zero baseline and reports a spurious large drift — a 400 A
common-mode reads as 2.4% on F9A but only 0.4% on F9B, falsely rejecting the
draw. This is the same small-baseline pathology that excludes the E-coils
above, and it silently tanks the in-spec yield on any peaked or
vertically-unstable baseline.

Bouquet therefore scores the VSC **selection** metric by decomposing the pair
into its two physical degrees of freedom and gating their changes against the
**error-propagated** measurement uncertainty (`_vsc_channel_drift_pct` in
`bouquet/TokaMaker_interface.py`):

```
I_cm = (F9A − F9B)/2   # common-mode = vertical control
I_df = (F9A + F9B)/2   # differential = shaping residual

# both channels are linear combinations of two independently measured coils,
# so their uncertainty propagates IN QUADRATURE from the per-coil uncertainties:
sigma_VSC = sqrt(|F9A|² + |F9B|²) / 2  +  denom_floor      # (= offset/gain)
drift_VSC = 100 · max(|ΔI_cm|, |ΔI_df|) / sigma_VSC
```

- **The tolerance is built from the coil magnitudes, not the channel
  baselines**, so it never goes near zero. Gating a channel against *its own*
  baseline blows up whenever that baseline is small, and there are **two** such
  cases on real equilibria: a near-zero *coil* (F9A ≈ −16 kA, which breaks the
  naive per-coil `|ΔI|/|I_base|`) **and** a near-zero *common-mode baseline* (a
  co-current pair, e.g. F9A ≈ F9B ≈ −65 kA, where `(F9A−F9B)/2 ≈ 7 kA`). The
  propagated `sigma_VSC` is immune to both, because a small difference of two
  large, imprecisely known currents is itself imprecisely known: σ comes from
  the coil errors, not the (small) channel value. An earlier larger-pair
  version and a channel-baseline version each fixed only one of the two cases;
  the quadrature fixes both.
- **The quadrature** (sum of squares, ÷2) is exactly the propagation of two
  *independent* transducer errors through the ±½ channel coefficients; the
  differential channel still catches a genuinely asymmetric/same-sign excursion.
- **`denom_floor`** (`_COIL_DRIFT_DENOM_FLOOR_A`, default 10 kA = offset/gain)
  carries the additive **measurement + eddy-current** uncertainty: real
  coil-current noise is `gain·|I| + offset`, where the offset (sensor zero +
  vessel/passive-structure eddy contribution) is *current-independent*. At the
  2% spec this floor is roughly a 200 A additive tolerance.
  **ASSUMPTION:** no published DIII-D coil-current measurement-noise figure was
  found (Rogowski/PF measurement is ~0.1–1% in the literature; eddy adds a
  current-independent term), so this is a deliberately conservative
  placeholder — replace the constant with the real transducer offset +
  eddy-equivalent figure when one is available.

This is the *measurement-uncertainty* reading: "states consistent with the
measured VSC currents". The non-VSC F-coils keep the relative
`|ΔI|/|I_base|` metric, since their uncertainty is gain-dominated. If the
binding constraint is instead the VSC power-supply **control authority** rather
than measurement uncertainty, gate `|ΔI_cm|` against that amp budget directly —
same structure, different number.

> Note: [architecture.md §15.6](../architecture.md#15-coil-constraint-handling-diii-d-reference)
> still describes the older naive per-coil formula for the VSC pair. This page
> is the current behaviour.

## Progressive homotopy

`GenerationConfig.homotopy_passes` is a list of `(drift_F, drift_VSC)` tuples
tried in order, each warm-starting from the prior pass's converged ψ. The
default schedule is:

```python
homotopy_passes = [
    (0.05, 0.10),   # Pass 1: loose start
    (0.02, 0.05),   # Pass 2: intermediate
    (0.01, 0.01),   # Pass 3: strict — total F9 drift ≤ 2% (bare 1% + VSC 1%)
]
```

Total F9A/F9B drift is `drift_F + drift_VSC`, so pick passes such that their
sum is the desired total F9 tolerance. A single direct solve at the tight
target is usually QP-infeasible from a cold start — the schedule is what makes
±1% coils reachable.

On infeasibility the homotopy rolls back to the last successful pass and
re-solves to restore the solver's internal flux-surface state (without that
re-solve `get_stats()` returns `l_i = inf`). If pass 1 itself fails, the draw
is rejected.

Related knobs: `GenerationConfig.coil_drift` (soft target, default `0.01`),
`coil_drift_hard_factor` (optional hard bounds at `± factor·coil_drift` in
*every* solve, default `None` = soft only), and `SolverConfig.coil_vsc` (which
coils form the antisymmetric channel).

## Per-draw attributes

Every stored draw carries these HDF5 attributes, so you can filter downstream
without re-running anything:

| Attribute | Meaning |
|---|---|
| `homotopy_pass` | Index of the last successful pass (0-based; −1 if no hard bounds ran) |
| `homotopy_F_lim`, `homotopy_VSC_lim` | The `(drift_F, drift_VSC)` of that pass |
| `max_F_drift_pct` | Max non-VSC F-coil drift, percent |
| `max_VSC_drift_pct` | Max VSC channel drift, percent (the metric above) |
| `in_spec` | `max_F_drift_pct ≤ inspec_F_max` **and** `max_VSC_drift_pct ≤ inspec_VSC_max` |
| `inspec_F_max`, `inspec_VSC_max` | The thresholds that were applied |

`in_spec` is the **legacy** verdict and stays that way whichever filter runs, so
on a default (χ²) run it means something different from `selected`: `in_spec`
answers "inside the ±`inspec_*` band", `selected` answers "χ²/ν ≤ `chi2_max` and
worst \|z\| ≤ `z_max`". Say which one a quoted in-spec fraction came from.

`Bouquet.filter()` writes the non-destructive `passes_coil_filter` /
`passes_boundary_filter` / `selected` flags. Its coil rule is
`filtering.coil_filter`: `"chi2"` (default) goes through `filter_coil_chi2` with
the per-coil σ from `filtering.coil_sigma` / the device model, and `"legacy"`
re-applies the thresholds in the table above. `filter_coil_currents`,
`filter_coil_chi2` and `filter_boundaries` are the standalone forms.
