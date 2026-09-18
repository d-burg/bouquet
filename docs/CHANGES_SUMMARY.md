# Bouquet — change summaries

## Unreleased — the default coil acceptance criterion changed

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
