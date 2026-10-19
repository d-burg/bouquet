# Known limitations (1.4.0)

These are the limitations bouquet 1.4.0 ships with. Each entry gives the
symptom, its consequence for a user, and the tracking issue. None of them was
closed by loosening a bar. The measurements behind them are in
[validation-provenance.md](validation-provenance.md).

## Solver and reconstruction

- **Bistable equilibria: the continuity guard is opt-in since 1.4.1;
  default-on pending the false-positive exercise (#87).** A near-double-null
  case can have two neighbouring free-boundary solutions for the same
  inputs; the bistability check stamps such a reconstruction and the
  per-scan stamp marks an axis-bimodal ensemble. `guard_bistability=True`
  (experimental `bistability_guard`) keeps a draw's coil homotopy on the
  branch each stage starts from: a stage that moves the axis more than 1 %
  of a is re-solved by continuation, and the continuous state is kept when
  the jump was branch hopping. It does not guard the draw's bootstrap loop:
  on the known reversed-I_p case 22 of 28 archived draws already sat
  16.5-17.6 mm below the reconstruction's axis when they entered the
  homotopy, so that ensemble stays bimodal with the guard on. There it
  fired on 7 of 28 archived draws: three continuous responses (kept), and
  four crossings between the two solutions in the first stage (two each
  way) that persisted down to 1/8 of the stage or whose sub-steps did not
  solve -- the starting branch ends inside the stage, so the guard keeps
  and stamps them; it changed no draw. On two single-solution slices it
  fired on 3 of 30 and 2 of 32 draws, each a continuous response of
  1.0-1.6 % of a, kept unchanged at two extra solves each; on the
  reversed-I_p control case it never fired. Until a loop-stage guard exists,
  read an ensemble's `axis_shift` record and bimodality stamp before
  quoting its bands.
- **MSE-stage runtime (#88).** Reconstructions with the structured MSE stage
  take 149–213 s (legacy route 178–192 s), against the 90–120 s
  target; without MSE they take 41–103 s. A study that needs MSE on many
  slices costs about twice the target per slice; the MSE stage still fits the
  raw chords best on every validated slice.
- **Series refusals (no issue).** A slice of a time series that the engine
  refuses (a closure refusal, a non-converged reconstruction, the IDA
  time-matching rule) is recorded per slice and superseded on regeneration,
  but it is not replaced. A series can have gaps: 5 of 30 slices of the
  validated series were refused by the time rule.
- **Δ′ at σ=0 is ill-conditioned on some slices (no issue).** The
  zero-perturbation repeat solve moved Δ′(2/1) by up to 1.88 on one validated
  slice (about a tenth of its draw band). Treat the reference solve's Δ′ as
  one sample with that scatter, not as an exact centre for the draws'
  distribution.
- **The boundary-constraint total is a constant, not a prescriptive rule
  (#93).** The total 2e5 sits inside a window measured on one device and the
  synthetic fixtures; the reported partition warns outside [1/3, 3] but
  never changes the total. On another device the window must be re-measured
  ([Running bouquet on another device](workflows.md#running-bouquet-on-another-device)).
  A per-case rule is deferred to 1.5.
- **One synthetic slice's partition sits outside the band (#93).** The
  synthetic dd at 0.7 s stamps a partition of 3.17 at the default total on
  the release tree (3.34 at `27979d1`) and the library warns; its draws do not suffer (24 in spec in 24–25
  attempts). The warning is informative there, not a failure; it stays as
  evidence for the 1.5 rule.

## Draws and statistics

- **The bimodality stamp's false-positive rate, and its n ≥ 24 floor (no
  issue).** On unimodal samples the axis-bimodality rule still says
  "bimodal" about 10–11 % of the time at n = 24 and about 3 % at n = 36;
  below n = 24 it reports the gap and spread but no verdict. A single
  "bimodal" stamp at n ≈ 24 is a prompt to look at the axis scatter, not
  proof of two branches.
- **The per-draw wall budget depends on the host (no issue).** The default
  600 s budget is wall-clock, so on a slow or loaded host an until-N run can
  archive a different draw set than on a fast one. Unified-engine draws took
  47–168 s in validation (no rejection); the legacy route's take
  156–598 s (2.6–10 min), so legacy runs should set `draw_wall_budget_s=None`. The σ=0
  verification always runs without the budget.
- **The beam fraction's own uncertainty is not drawn (1.5 plan).** Every
  draw holds the beam's charge moments and `p_fast` fixed and samples the
  total Z_eff conditional on them. A real beam-fraction uncertainty, which
  would move the thermal Z_eff and `p_fast` in opposite directions, is
  missing from the bands.
- **g-file + p-file: the beam's density is not carried (#97).** Since 1.4.0
  the p-file's `pb` is the default `p_fast`, but its `nb` never reaches the
  draws as `z_fast`. The Z_imp derivation, the Z_eff window and the n_i
  ceiling therefore ignore the beam: on the shipped total-pressure pair 15 of
  24 in-spec draws carry clips that come from this beam-blind window.
- **g-file + p-file: post-align failures (#97).** On the same pair, 10 of 35
  attempts were rejected as `post_align_failed` (none on the thermal g-file
  routes). The ensemble still reaches 24 in spec, but it costs about 1.5x the
  attempts. The cause is not established; #97 is the first suspect.
- **IDA versus dd consistency is reported only as warnings (#89).** When an
  IDA file and a dd describe different plasmas (on one validated case their
  n_i differed by far more than their uncertainties), the advisory warnings fire but
  there is no pre-run summary. Read the warnings before spending an
  ensemble on an IDA-hybrid pair.

## Experimental and deprecated routes

- **swb is experimental, and its I_p closure is red until #81.** The swb
  production ensemble's worst draw closes I_p to 1.06e-4 against the 1e-4
  bar, on the fork build the route needs. Use swb for comparison only; the
  bar is not moved, and the test stays red until the toolkit build switch
  (#81, with #82).
- **The legacy route is deprecated (#90).** `reconstruction_engine="legacy"`
  and `solve_method="legacy"` raise a `DeprecationWarning` and will be
  removed in 1.5. New work should use the unified engine; the legacy route
  stays only for comparison and its golden.
- **The legacy route is not replay-deterministic (#90).** The legacy golden's
  second stored draw replays at 0.86 mm boundary RMS against 2.95 mm stored
  (bar 2.0 mm), on the same build and draw set. A legacy draw cannot be
  reproduced to that bar, so the legacy replay test stays red; no fix is
  planned before removal.
