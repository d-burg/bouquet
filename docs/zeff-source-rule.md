# The Z_eff source rule (`ni_source="checked"`)

`bouquet.io.zeff_sources` (numpy and the standard library only) and `read_ida(ni_source="checked")`.
Default: `ni_source="standard"` (no rule; unchanged). See also
[physics-notes.md](physics-notes.md#kinetic-assumptions-z_eff-n_i-and-the-clips-pr-56),
[api-reference.md](api-reference.md) and [workflows.md](workflows.md).

## 1. Problem

The effective charge enters a kinetic equilibrium through the main-ion density (dilution, thermal pressure), the
neoclassical conductivity and the bootstrap current. A kinetic profile file can carry two candidates: a Z_eff labeled
as a line-integrated (visible-bremsstrahlung) measurement, and an impurity density from charge-exchange spectroscopy
from which Z_eff follows by single-impurity quasineutrality, Z_eff = 1 + Z (Z − 1) n_imp/n_e. A reader that takes one
of them blindly is unsafe in both directions:

- the file Z_eff can be a constrained measurement, a prior-dominated default with no information, or a copy of the
  impurity-derived value, and nothing in the dataset name distinguishes these;
- the impurity density can be a placeholder on slices where no channel was available, or an extrapolation where the
  channels do not cover the core.

With everything else fixed, the choice changes the neoclassical conductivity by tens of percent, the pedestal
bootstrap current by up to about 20 %, and the stored thermal energy by up to about 20 % on a strongly diluted plasma.

## 2. The rule

Each candidate is tested per time slice. Nothing is chosen by fit quality, by agreement with a reconstruction or by a
cross-check. The rule is evaluated for one impurity species (the file's `n_12C6`, carbon, Z = 6).

**Tests of an impurity-density source**

| test | criterion |
|---|---|
| I-1 coverage | the measurement points behind the profile (those the fit used; else raw points within the slice's time window; else within ± 25 ms, inclusive) include one at ρ ≤ 0.30, one at ρ ≥ 0.80, and at least 8 inside the boundary |
| I-2 not a placeholder | relative radial spread (peak-to-peak over median) of n_imp/n_e over ψ_N ≤ 0.9 ≥ 1e-3, and median relative uncertainty of n_imp over ψ_N ≤ 0.9 ≤ 1 |
| I-3 physical | inside the boundary (ψ_N ≤ 1): the data are valid (finite, positive, not fill values) and the main-ion density is positive, n_i/n_e = 1 − Z n_imp/n_e − Σ Z n_fast/n_e > 0 (fast ions included when a fast-ion density is supplied; see "Fast ions" below), and Z_eff ≤ Z; a warning when n_i/n_e < 0.1 |
| I-4 consistent with supplied raw analyses | over ρ ≤ 0.9 and separately over ρ < 0.5: \|median z\| ≤ 2 and median point/profile ratio within [2/3, 3/2], with z = (point − profile)/sqrt(σ_point² + σ_profile²). Evaluated only against raw analyses the caller supplies, never against the points the profile was fitted to |

ρ is ρ_tor,norm; the reader places points through the file's own `q` (`zeff_sources.rho_map_from_q`, the same
arithmetic as `bouquet.coords.phi_n_from_q`).

**Tests of a file-supplied line-integrated Z_eff**

| test | criterion |
|---|---|
| L-1 | it carries its own uncertainty (`Zeff_err`, or the sample spread of a posterior file); a zero, negative or non-finite σ counts as none |
| L-2 | median relative uncertainty over ψ_N ≤ 0.9 ≤ 0.25 |
| L-3 | it is not a copy of the impurity-derived value: maximum relative difference over ψ_N ≤ 0.9 > 1e-3 (not applicable to a file without an impurity density: there is nothing to copy) |
| L-4 | it is not radially flat: peak-to-peak over ψ_N ≤ 0.9 ≥ 0.02 |
| advisory | coefficient of variation of the slice core median (ψ_N ≤ 0.5) over the file's non-placeholder slices < 0.05 is recorded as a warning (never disqualifies) |

### Three states

Each candidate ends in one of three states:

- **qualified**: every test was evaluated in full and none failed;
- **unverified**: no evaluated test failed, but a test, or part of one, could not be evaluated;
- **failed**: an evaluated test failed (or the candidate cannot be used at all, e.g. `impurity_Z ≠ 6` for the carbon
  density).

What can leave a candidate unverified, each listed by name in the record's `not_evaluated` entry:

| part | when |
|---|---|
| I-1 coverage | no measurement points (no fit points in the file, no raw analyses supplied), or ψ_N-only points with no usable ρ map |
| I-2 relative-σ half | no usable impurity-density uncertainty |
| I-4 | no raw analyses supplied |
| I-4 z half | a raw analysis without point uncertainties, or a profile without one |
| I-4 region | the core (ρ < 0.5) or overall (ρ ≤ 0.9) region has no raw points |
| L-3 | the file has an impurity density but no valid impurity-derived values to compare with |

### Fast ions

A missing fast-ion density does **not** by itself make the impurity source unverified. With singly charged main and
fast ions the impurity-derived Z_eff = 1 + Z (Z − 1) n_imp/n_e is independent of the fast-ion density; fast ions change
the thermal main-ion density and pressure, so supplying them matters for n_i, not for the choice of Z_eff source. I-3
therefore runs in one of three ways, recorded on its entry (`fast_ions`, `evaluated_without_fast_ions`):

| `fast_ions` | input | I-3 |
|---|---|---|
| `"supplied"` | a fast-ion charge density (`ida_hybrid`: the dd's; the g-file route: `ReconstructionSource.zeff_fast_density`, or `read_ida(zeff_fast_density=)`) | evaluated with it |
| `"none"` | the caller declares there are no fast ions (`zeff_fast_density="none"`) | evaluated with fast ions = 0, by declaration |
| `"not_available"` | nothing supplied | evaluated without fast ions; listed in the record's `evaluated_without` (kept apart from `not_evaluated`) |

The n_i/n_e < 0.1 warning uses whatever is known: a supplied fast-ion density lowers n_i/n_e and can trigger it.

**Ranking**

| order | used when | action |
|---|---|---|
| 1 | the impurity source is qualified | impurity source (a > 2σ disagreement with a qualified file value is warned about) |
| 2 | the file value is qualified | file value |
| 3 | the impurity source is unverified | impurity source, labeled unverified in the record and warned about |
| 4 | otherwise | refused, unless an explicit, recorded override names a source |

The rule names only the impurity source among the unverified ones. An **unverified file value** is ranked
conservatively: it is never used ahead of an unverified impurity source and is never used without an explicit
override (open question to the maintainers: whether it should rank between 3 and 4). A file with no impurity density
and a Z_eff without uncertainty is refused unless overridden.

The tension (both qualified) is the \|median z\| over ψ_N ≤ 0.9, z = (Z_imp − Z_line)/sqrt(σ_imp² + σ_line²), warned
above 2.

**Arithmetic.** The chosen source is taken through the arithmetic of the corresponding forced route: the impurity
source as `ni_source="CER"`, the file value as `ni_source="Zeff"`. Where that route has its envelope the result is the
same Z_eff, n_i and envelopes, bit for bit. Two cases differ from the forced route: (a) when the reader's whole-grid
carbon screen dropped the carbon envelope only because of values outside the boundary, the checked route uses the
envelope screened inside the boundary (`sigma_Zeff_carbon_source` and `zeff_provenance["carbon_envelope_screen"]` say
so; the forced `"CER"` route still refuses such a file, as before); (b) an override of a source without its envelope,
or a carbon source without `n_12C6_err`, takes the reader's bottom rung (the source's value with an n_e-fraction σ).
Outside the boundary the impurity-derived values are whatever the data give (NaN where the density is missing).

**Overrides.** `zeff_override="CER"` or `"Zeff"` names the source whatever the tests say, recorded and warned when
the named source is not qualified or differs from the rule's choice. It is refused for a source the file does not
carry or cannot use, and for an impurity density that fails I-3: an override cannot make invalid data or n_i ≤ 0
usable.

**Thresholds** are module constants of `bouquet.io.zeff_sources` (`COVERAGE_RHO_INNER_MAX`, `LINE_REL_SIGMA_MAX`, …,
collected in `THRESHOLDS`). Each was set from its measured distribution over the files of a validation set, in the gap
between the slices it is meant to reject and those it is meant to accept. They are part of the public behavior: a
change of any value changes which source a file gets, and every record carries the values it used. The tests pin each
one's inclusive or exclusive side exactly at its value.

### Validity outside the boundary

The rule's tests are defined inside the boundary, so the rule screens the impurity data (with the reader's carbon
criteria: finite, positive, not a fill value, finite non-negative σ) over ψ_N ≤ 1 only. Invalid values outside the
boundary (SOL holes, spline undershoot) are excluded and counted in the I-3 record (`excluded`), and the slice is not
refused for them. Invalid values inside the boundary fail I-3 with a message that names missing or bad data, not
dilution. The reader's own screen for the pre-existing routes is unchanged.

### Measurement points

- **Fit points** (I-1 only): the points the profile was fitted to at the slice. The reader takes them from the file's
  `raw_n_12C6` / `raw_n_12C6_err` / `psin_n_12C6` datasets when it carries all three (padding and fill values dropped;
  a slice whose fit used no point gives an empty set, which fails I-1), or from `zeff_fit_points`.
- **Raw analyses** (`zeff_raw_points`): one or more `MeasurementPoints` (or dicts with `value`, `sigma`, `rho` or
  `psi_N`, `time` in s, `label`, `kind`), selected in the file's own window for the slice (`time_win_start` /
  `time_win_end`; s or ms, whichever contains the slice time) when it states one, else within ± 25 ms. Each analysis
  is labeled `kind="hand_fit"`, `"automatic"` or `"other"`: the hand-fit analysis is the reference for I-1 (when there
  are no fit points) and I-4 when one is supplied, else the automatic one. The label decides, never the order; two
  analyses of the deciding kind, or none of either kind, is an error. Every analysis is evaluated and recorded.

## 3. What is recorded

`IDAProfiles.zeff_rule` (and, on `"checked"`, `zeff_provenance["zeff_rule"]`, which the baseline archives with
`li_metrics`) is a JSON-safe dict (`schema_version` 2):

- `candidates`: for `impurity_density` and `line_integrated`, `present`, `state`, a value summary (median, min, max of
  the candidate's Z_eff over ψ_N ≤ 0.9), every test with its `value`, `threshold`, `verdict`, `detail` and
  `not_evaluated_parts` (I-1 and I-4 per raw analysis, I-4 per region; I-3 with the excluded outside values),
  `failed`, `not_evaluated`, `findings`, and, for an unusable candidate, `reason`;
- `tension`: median and maximum \|z\|, median ratio Z_imp/Z_line, `exceeds`;
- `decision`: `source` and its `source_state`, `rule_source` (what the rule alone decides), the candidates' `states`,
  the ranking `rank`, the `row` (e.g. "impurity unverified, line-integrated failed"), `reason`, `refused`;
- `override`: the named source, what the rule would have chosen, the named source's state, failed tests and
  not-evaluated parts;
- `forced`: on `"CER"` / `"Zeff"` / `"all"`, the route, its candidates, the candidates it used and its clamp counts;
- `not_evaluated` (every unevaluated part of both candidates), `evaluated_without` (I-3 run without fast ions because
  none was available), `warnings` (those for the decided source),
  `crosschecks` (optional cross-checks supplied with `zeff_crosschecks`; reported, never selecting), `thresholds`,
  `time`, `ni_source`, `validity_screen`, how ρ was obtained (`rho_map`, with the reason when the map could not be
  built), and on `"checked"` the route taken.

A refusal raises `ZeffSourceRefused` (a `ValueError`) whose `record` attribute holds the same dict.

**Warnings.** A `"checked"` read emits at most one `ZeffSourceWarning`, naming the decided source and its state and
listing only what concerns it: the not-evaluated parts of an unverified source, a tension, an override, a strongly
diluted plasma, an advisory. A qualified source with no finding is silent.

## 4. Options

| `ni_source` | Z_eff | the rule |
|---|---|---|
| `"standard"` (default) | the file's Z_eff as stored | not applied; output identical to earlier releases bit for bit (`zeff_rule` is `None`). **The explicit opt-out.** |
| `"checked"` | the source the rule names | applied; refused per the ranking; `zeff_override="CER"` or `"Zeff"` names a source explicitly |
| `"CER"` | the impurity density | evaluated and recorded; one warning per read only when the impurity source **fails** a test (with the clamps the route applies), never for an unverified one; not refused |
| `"Zeff"` | the file's Z_eff | the same, for the file value |
| `"all"` | the equal-weight mean of the two | the same, for the sources the mean uses; a tension is recorded, not warned. Not part of the rule |

`ImasSource.ni_source="auto"` resolves to `"all"` on `ida_hybrid`, so default hybrid runs evaluate the rule (the
advisory reads every slice of the IDA file) and warn when a source the mean uses fails a test, for example a file
Z_eff that copies the carbon value (L-3).

`zeff_substitute` (the IDA-hybrid route's dd Z_eff for a file without `Zeff`) applies to `"standard"` only, as before.

Configuration: `ReconstructionSource.ni_source="checked"` / `ImasSource.ni_source="checked"`, with `zeff_override` on
either source (allowed only with `"checked"`). On `ida_hybrid` the dd's fast-ion charge density is passed to I-3 on
`"checked"`. On the g-file route `ReconstructionSource.zeff_fast_density` supplies the fast-ion charge density
(`"none"` to declare there are none; an array on the IDA file's ψ_N grid; or `{"psi_N": ..., "n_fast": ...}`, kept
JSON-round-trippable, so a callable is accepted by the reader keyword but not by the config); it applies to an IDA file
on a route that applies the rule. Point data for the g-file route are reader keywords only
(`read_ida(..., zeff_fit_points=, zeff_raw_points=)`); without them a g-file read's impurity source is unverified
(coverage and raw consistency not evaluated), and a qualified file value is then preferred.

**Uncertainty re-read.** The rule is evaluated once, on the profile file's own read. The uncertainty step's re-read
(the same file, or a separate `UncertaintyConfig.ida_path`) reuses that record (`read_ida(..., zeff_rule_from=)`): it
is not evaluated or warned again, and a separate σ file takes the source the profile file decided.

## 5. Time series

One source for the whole series, named by the user, never switched silently.
`bouquet.io.ida.zeff_rule_series(path, times, source="CER" | "Zeff")` reads every slice from the named source (an
override recorded on each slice) and returns the slices on which that source fails its tests
(`series["failing_slices"]`) or is unverified (`series["unverified_slices"]`), flagged, not switched. Without `source`
the call is refused with the rule's per-slice outcomes listed, even when every slice agrees, so that the user names
one. The pure helper is `bouquet.io.zeff_sources.series_source(records, source)`.

**Independent per-slice reads do not enforce this.** `read_ida(ni_source="checked", time=t)` or one
`ReconstructionSource(ni_source="checked", time=t)` per slice applies the rule to each slice on its own, and across a
campaign different slices can take different sources with nothing tying the runs together. Use `zeff_rule_series`
(or pass the same `zeff_override` to every slice) for a series.

## 6. Limitations and assumptions

- **Single impurity species.** Z_eff,imp assumes one fully stripped impurity (carbon, Z = 6); other impurities make it
  a lower bound. Multi-impurity plasmas are outside the rule. With `impurity_Z ≠ 6` the carbon candidate is recorded as
  present but not usable, with that reason.
- The fast-ion term is a charge density, Σ Z n_fast (for singly charged beam ions, their density); the impurity is
  assumed to be at the ion temperature where that matters.
- **Without a fast-ion density** I-3 is evaluated without fast ions (recorded); the source state does not depend on it,
  but the n_i/n_e value and its < 0.1 warning do, so supply the fast-ion density when n_i matters.
- **The thresholds were set on one validation set** (files of a single carbon-wall device). Other devices or analysis
  chains may need different values; a change must be proposed and reviewed, never made silently.
- **Cases can sit near thresholds.** On the validation set some files sat close to the lines: innermost measurement
  points near ρ 0.22–0.26 (coverage line 0.30) and an outermost point near ρ 0.88 (line 0.80); slices with only 6–13
  points inside the boundary; median relative Z_eff uncertainties of 0.20–0.23 (line 0.25); file Z_eff peak-to-peak of
  0.026–0.039 (line 0.02); a core point/profile ratio of 0.673 with median z −1.997 (lines 2/3 and 2); two candidates at
  a tension of 1.97σ (line 2). For files without their own points the coverage verdict depends on the time window used.
- **Cross-checks never select.** Bremsstrahlung brightness, radiated power and similar checks may be recorded beside
  the decision (`crosschecks`) but never choose a source; a z-test alone does not catch an extrapolated core (a large
  extrapolated σ hides it), which is why I-4 also bounds the ratio.
- Whether a file's line-integrated Z_eff is prior-dominated cannot be read from the file; L-2 and L-4 are proxies.
  The cleanest fix is for the kinetic file to record what constrained its Z_eff.
- I-1 and I-4 depend on the mapping of channel positions (the file's own `q`); a file without a usable `q` cannot
  place ψ_N-only points in ρ_tor, and those tests are not evaluated (the record names why).
- L-3 assumes the file Z_eff and the impurity density are independent once they differ by more than 1e-3.
- The not-evaluated parts are a conservative reading: a part is counted even when it could not change the verdict.
