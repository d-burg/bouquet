# Bouquet — change summaries

## 1.4.0 — hybrid Ip closures, measurement-referenced coil filter, provenance-read fast pressure (2026-09-19)

Ten PRs landed together. Four of them change results for existing configs;
they are listed first. Everything else is opt-in or a bug fix.

### Behaviour changes — check these before re-running old configs

1. **Fast-ion pressure convention is now read from the data file.**
   `p_fast_reduction` defaults to `"auto"`. Integrated-modelling codes disagree on
   whether `pressure_fast_perpendicular/parallel` are full pressures or
   per-degree-of-freedom values — a factor of 3 in p_fast. `"auto"` uses an explicit
   stamp in the file if present, else identifies the producing code from provenance,
   else falls back with a warning. For per-degree-of-freedom producers this raises
   p_fast (and so beta_N, W_MHD, P') relative to 1.3.x. Set `"sum"` or `"trace"`
   explicitly to pin it. The rule used is recorded on `Baseline.p_fast_meta`.
2. **Coil-current acceptance is now a chi-squared test, not a ± percent band.**
   Each draw's coil currents are compared to the baseline using per-coil
   uncertainties from a device registry (`bouquet/devices.py`); a draw must pass
   both chi2/nu and worst-coil |z|. A given ensemble will select a *different*
   subset than before. `filtering.coil_filter = "legacy"` restores the old band
   exactly. Devices not in the registry fall back to the legacy band with a
   warning; supply `filtering.coil_sigma` (floor + fraction, or a per-coil table)
   to use the new test on another machine. The acquisition era that sets the
   uncertainty floor is chosen from the pulse number and printed on every
   `filter()` call; `filtering.coil_daq_era` overrides it.
3. **Fast ions no longer inflate the impurity density (IMAS path).** Quasi-neutrality
   removes the fast-ion charge before the main-ion / impurity split. Sources with no
   fast ions are bit-identical to before. Assumes the source Z_eff is defined with
   thermal species over the full n_e — stated in the docstring, not verified.
4. **Z_eff uncertainty is measured when the data allow it.**
   `uncertainty.zeff_sigma_source` walks a ladder: propagated from measured carbon
   density → measured visible-bremsstrahlung sigma → the old scalar envelope. The
   tier used is recorded; fallbacks warn. Users still on the scalar tier are
   unchanged. Side effect: `impurity_Z` now reaches the kinetics reader, which moves
   sigma_ni for non-carbon impurities.

Also fixed: in `jbs_delta_mode` (and DIFF_BS), the sigma=0 bootstrap reference was
built at scale 1 while draws used the configured scale, biasing every delta.

### New capabilities (all opt-in)

- **Hybrid "ohmic" baseline with explicit Ip closure.** Baseline current =
  s_ind·j_inductive (from an integrated-modelling source) + s_bs·j_bootstrap
  (neoclassical model on measured kinetics) + fixed sources, closed on the measured
  Ip. `closure_channel` chooses what absorbs the mismatch:
  - `"bootstrap"` — scalar rescale of j_BS (remains the code default);
  - `"sawtooth_bootstrap"` — two scalars, Ip exact and q0 pinned when sawtoothing;
  - `"structured"` — smooth radial scale functions (four Gaussians in psi_N) for both
    components, minimum-norm under trust weights, with **Ip and l_i as soft
    measurements** and a one-sided prior on mid-radius inductive current. Selecting
    this channel applies the validated preset `li_soft_onesided` by default
    (`structured_preset="none"` opts out). In our study it matched Ip within 0.5 % on
    all slices and l_i within its uncertainty on 95 %, and beat both scalar channels
    about 2:1 on agreement with MSE-constrained q profiles, without rescaling the
    bootstrap current beyond ±50 %. The preset's widths come from one device and one
    modelling source: a starting point elsewhere — check the recorded
    closure-health flags.
  - `"ohmic"` is deprecated.
  Every closure records a health block (bound hits, |s_bs−1| > 0.5, missed q0,
  l_i / Ip residuals in sigma, refusals). Refusals are loud, never silent.
- **`SolverConfig.coil_reg`** — regularise the baseline inverse solve toward measured
  coil currents with inverse-variance weights (helpers in `bouquet/coil_targets.py`).
  Removes the large null-space wander of an unregularised baseline. Turn-count
  conventions are checked against the mesh at solve time. `coil_init` seeds the solve.
- **`generation.n_inspec_target`** — draw until N draws pass the in-spec filters
  (bounded by `max_total_draws`) instead of drawing exactly N. The in-loop test and
  the post-hoc filter are the same code on the same stored vectors.

### Compatibility

- `isotropize_fast_pressure(..., method)` signature changed (hence the minor bump).
- Counts such as `n_inspec_target` / `max_total_draws` must be true integers.
- `closure_channel="structured"` with no other settings now means the validated
  preset, not the raw weight ladder.
- No acceptance tolerance was loosened anywhere in this release.

The four "1.4.0 detail" sections below give the full account of each behaviour change.

## 1.4.0 detail — the default coil acceptance criterion changed

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

## 1.4.0 detail — fast-ion pressure: the reduction rule now comes from dd provenance

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

## 1.4.0 detail — the Z_eff envelope is measured, not assumed

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

## 1.4.0 detail — the structured closure defaults to its validated preset

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
