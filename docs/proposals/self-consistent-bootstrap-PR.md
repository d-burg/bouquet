# PR: self-consistent bootstrap current (default ON)

*Draft pull-request description for `feat/jbs-self-consistent-loop`. The PR is
not opened yet; this file is the text it will carry. Stacked on
`feat/structured-mse-term` (the structured closure's MSE term), which it
includes. The branch also carries the reversed-Ip current-sign commits
(the IMAS reader brings a `ip < 0` source into bouquet's positive-current
frame, records the orientation, and the matching tests and docs;
`docs/CHANGES_SUMMARY.md`, "reversed-current IMAS sources"), cherry-picked
from their own branch; they are not part of the loop and are described
there, not here.*

## Summary

Until now bouquet computed the bootstrap current **once** — OFT's
`solve_with_bootstrap` (SWB) on its own auxiliary equilibrium — and then only
rescaled it. This PR replaces that with a Redl bootstrap re-evaluated on the
equilibrium bouquet actually delivers, iterated to self-consistency with the
current closure and the Grad–Shafranov solve, and makes it the **default**.

Two defects of the frozen bootstrap motivate it:

1. **Grid.** SWB assumes an evenly sampled ψ_N. bouquet never passed `psi_N=`,
   so on the IMAS path — whose integrated-modelling grid is uniform in ρ_tor —
   profile gradients were mis-scaled by dψ_N/du (well below 1 near the axis,
   above 1 beyond mid-radius) and the geometry sampled at the wrong surfaces.
   The geqdsk path (uniform grid) is unaffected.
2. **Frozen geometry.** The bootstrap was evaluated on SWB's generic-seed,
   thermal-only equilibrium, never on the delivered one (source inductive
   current, full pressure, closure multipliers, post-homotopy coils). Since
   j_BS ∝ (dp/dψ_N)/Δψ and Δψ scales with √l_i, the amplitude error is
   several percent, and the closures, correctors and draws inherited it.

## What changes

- **`physics.evaluate_jBS`** — a port of SWB's inner Redl evaluation run once
  on the equilibrium it is handed (no solve inside), with geometry sampled on
  the caller's surfaces, gradients on the true grid, and a direct
  ⟨j·B⟩ → toroidal conversion. Bit-identical to SWB's first pass on a uniform
  grid; grid-independent on a non-uniform one (the defect-1 regression test).
- **`bouquet/jbs_loop.py`** — the fixed-point kernel: closure on the current
  geometry → GS solve → Redl, with joint under-relaxation of the bootstrap
  (ω = 0.7) and of the solved current (β = 0.7; damps the closure ↔ geometry
  oscillation), ω halved only on sustained growth, convergence = every active
  residual on **two consecutive passes** (`r_j ≤ 1e-3`, `r_I ≤ 1e-4 I_p`,
  `Δl_i ≤ 1e-3`, `Δq0 ≤ 2e-3`). Pass ceilings (limits, not tolerances):
  8 for the baseline / reconstruction, 12 for each loop of a draw
  (`jbs_max_passes_draw`) and 4 post-homotopy passes
  (`jbs_max_passes_post_homotopy`). Non-convergence raises `JBSNotConverged` with the full history,
  or flags the slice (`jbs_loop_on_fail="flag"`); a non-converged draw is a
  failed draw.
- **Everywhere a bootstrap enters j_φ:** the IMAS baseline in every
  `jBS_baseline_mode` and closure channel (structured closure incl. its MSE
  stage: Jacobian once, chord steps with j_BS re-evaluated, one final Jacobian
  refresh), every draw (Fix C and the standard l_i loop, plus a post-homotopy
  check), `verify_sigma0_consistency` (the loop's own σ=0 invariant) and the
  geqdsk reconstruction. In diff mode `jBS_diff` becomes a pure model offset on
  the delivered baseline geometry, so a σ=0 draw still reproduces the source
  bootstrap exactly.
- **Structured soft closure: a changed acceptance criterion, loop only.**
  The soft closure is called once per loop pass. *Old criterion:* when no
  Levenberg-damped step can be verified downhill, accept the iterate iff its
  scaled gradient is below `rtol·max|J|·max(√F, 1)`; otherwise refuse.
  *New criterion (with the loop on):* the same, **plus** accept an iterate
  stationary to within the objective's rounding noise
  (`stop_reason="noise_floor"`: every trial's predicted decrease below the
  noise estimate `NOISE_FLOOR_FACTOR · ε · Σ(2|r_i|m_i + r_i²)`, factor 2, and
  the gradient below the floor that noise implies). So it **accepts points
  the old test refused** -- only those; every result the old test returned is
  unchanged. It is opt-in: `close_ip_structured_soft(...,
  accept_noise_floor=True)`, passed only by the loop's closure calls; the
  default (`False`, every frozen-path call) is the historical strict solver.
  Every acceptance is recorded (`gn_stop`, `n_noise_floor_accepts`) and
  printed; a non-finite or non-positive noise estimate accepts nothing.
  Inside the loop a refusal is retried once from the previous pass's
  coefficients (logged).
- **Default ON** (`GenerationConfig.jbs_self_consistent=True`).
  `jbs_self_consistent=False` is the legacy frozen path, bit for bit -- which
  requires the noise-floor acceptance above to stay opt-in: the frozen path's
  closure calls leave `accept_noise_floor` at its default `False`. A stored
  config (dict or JSON) without the field loads with the loop off and a
  warning that says how to opt in, so an old archive's `config_json` replays
  the model it was produced with; an unknown (misspelt) `generation` key is
  refused, naming the nearest valid key, so a typo cannot land there.
  `swb_iterations` is ignored under the loop and a non-default value raises a
  `DeprecationWarning`.
  `single_profile_jphi=True` / `recalculate_j_BS=False` (no bootstrap to
  iterate) are refused unless the legacy flag is set.
- **Archive schema v3** (additive): the `jbs_loop` block
  (`jbs_converged`, `jbs_n_passes`, `jbs_loop_json`) on every loop draw and on
  `_baseline`; readers `DrawView.jbs_loop`, `ScanView.baseline_jbs_loop` /
  `bootstrap_model`. No migration: a v2 archive reads as frozen everywhere.
- **Plots** label the bootstrap "self-consistent Redl bootstrap" or "frozen
  SWB bootstrap (legacy)" from what the archive records.
- **Where a draw's loop starts** (recorded per loop as `init_source`): Redl
  at the draw's state anchor on the draw's OWN perturbed kinetics for its
  first loop, warm from its previous converged bootstrap for later ones, and
  a relaxed blend with Redl on the delivered equilibrium post-homotopy --
  never the unperturbed baseline bootstrap. Initialisation only: a fast and a
  solver test start the same draw loop from the baseline bootstrap and reach
  the same fixed point.
- **Post-homotopy check of a standard draw** re-solves through the draw's own
  Ip renormalisation + corrective iteration instead of handing the achieved
  current back to one jphi-linterp solve (which exhausted `maxits`).

No existing solver tolerance VALUE (`nl_tol`, `maxits`, `structured_li_tol`,
`q0_tol`, the soft solver's `rtol`/`max_iter`, …) and no test bar changed.
**One acceptance criterion did change:** the structured soft closure's
noise-floor acceptance above (approved for the loop; opt-in, so the legacy
path keeps the old criterion).

**Review fixes on top of the loop (2026-09-29).** `evaluate_jBS` refuses
(`physics.JBSEvaluationError`, naming the quantity and ψ_N) non-physical
input and a failed-trace geometry row instead of returning a silently zeroed
bootstrap there, keeping the historical treatment only at the clipped axis /
separatrix nodes; bit-identical for every accepted input. The kernel raises
`JBSNonFinite` at once on a non-finite initial guess or Redl evaluation, and
stops at the first pass that can never count (a gated l_i / q0 not returned,
J ≡ 0 against a non-zero iterate). Every loop record also carries the
unrelaxed closure-half residual `current_residual_unrelaxed`
(`= current_gap / (1 − β)`; record only, not gated -- whether to gate it is
an open decision). The OFT build is recorded as `{version, git_hash,
build_id}`, never an install path. The `⟨B_φ²⟩/⟨B²⟩` bracket the evaluator
drops is the legacy convention; its size is documented (1.0–2.1 % across a
D3D-like plasma, 1.4 % at the bootstrap peak), not changed.

## Evidence

- **Legacy flag is the legacy path, bit for bit.** Out-of-tree A/B against the
  pre-loop tree on the synthetic D3D-like example (same machine, same OFT
  build, one thread): geqdsk reconstruction + σ=0 check + two draws, IMAS
  diff baseline + one draw, and the IMAS structured (ohmic) closure — every
  array, attr and archived dataset identical (276 / 156 / 177 compared items
  respectively); the only
  differences are the new closure stop-test bookkeeping keys (additive).
  None of those cases hit a soft-closure refusal, so this A/B could not show
  the noise-floor acceptance reaching the legacy path (it did, at the time);
  with the acceptance now opt-in, a fast test shows the default solver
  refusing exactly where, and with exactly the message, the pre-loop solver
  did (and a scratch comparison against the pre-loop function agreed on 850
  of 850 synthetic cases: 794 bit-identical returns, 56 identical refusals).
  In-tree, a solver test tripwires the loop kernel and the Redl evaluator on
  the legacy flag and asserts neither is entered while `solve_with_bootstrap`
  is.
- **Grid independence** (fast + live): the same physical profiles on a
  uniform and a strongly non-uniform ψ_N grid give the same j_BS to
  interpolation accuracy; the legacy evenly-sampled reading does not.
- **Fixed point** (fast + live): init-independent (anchor vs legacy init
  converge to the same baseline); a loop started at its fixed point returns
  it in one pass; relaxation changes the path, not the fixed point (tested
  against the closed-form fixed point of a two-state model).
- **Real-data A/B (summarised, device-agnostic).** On a multi-shot
  tokamak campaign (hundreds of slice-channels, two closure channels) the
  loop converged on ≥ 99 % of slices in 4–5 passes at roughly neutral cost;
  the remaining slices failed loudly (flagged, excluded) rather than silently.
  The bootstrap fraction of I_p moved down by a few percent of I_p on IMAS-path
  hybrids and a spurious mid-radius inductive cut the frozen bootstrap forced
  on the closure disappeared; the q = 2 location moved by a few 10⁻³ in ψ_N.

## Figures

All five figures come from the public synthetic example only: the
D3D-like g-file, p-file and mesh in `examples/D3D-like`, configured from the
golden fixture's stored config (read the way
`tests/golden/regenerate_golden_run.py` reads it). Each is a single-slice
reconstruction on the geqdsk path, run one thread at a time, with no draws and
no IMAS. `docs/figures/make_bootstrap_loop_figures.py` regenerates them. Each
figure's computation is its own subcommand, taking about 1–3 min per
reconstruction; `all` runs every subcommand. The numbers quoted below come
from the development laptop build. The loop-on reconstruction reproduces the
refreshed fixture's baseline j_BS to 5×10⁻⁵ of its peak, and its l_i target
to 2×10⁻⁷. Bootstrap fractions are `I_BS / I_tor`: both are integrals of the
delivered profiles, weighted by the loop's own current measure
(`residual_weights`), so the normalisation of that measure cancels.

**Figure 1 — what the flag changes** (`docs/figures/jbs_loop_fig1_profiles.pdf`).
Solid lines show the delivered reconstruction with the loop on
(self-consistent Redl bootstrap). Dashed lines show the same reconstruction
with `jbs_self_consistent=False` (frozen SWB, legacy). Orange is j_BS, blue is
the inductive current and black is the total j_tor, all against ψ_N. The left
panel covers the full radius; the right panel zooms on the pedestal. The two
totals nearly coincide, because the reconstruction fits j_tor to the g-file
either way. What moves is the split between bootstrap and inductive current.
The self-consistent pedestal bootstrap peak is 5 % higher (0.534 →
0.562 MA m⁻²), and the inductive current gives up the same current (−11 kA) in the
edge, beyond ψ_N ≈ 0.85. I_BS/I_p goes from 0.2293 to 0.2385 (I_BS 284 → 296 kA). The
l_i target barely moves (0.653833 → 0.653864), and neither does q0
(1.2306 → 1.2311).

**Figure 2 — convergence of the loop** (`docs/figures/jbs_loop_fig2_residuals.pdf`).
This plots the per-pass residuals of the loop-on reconstruction on a log
scale:
- r_j (current-weighted L2 distance between the Redl bootstrap of the new
  equilibrium and the one it was solved with);
- r_I (the same distance as a fraction of I_p);
- Δl_i;
- Δq0.

Each has its tolerance as a horizontal line. Dashed means the criterion is
gated; dotted means it is not. Δq0 is only logged here: it is not a
criterion on the reconstruction path. The figure script records q0 per pass
through a logging wrapper and never gates on it. Relaxation is ω = 0.7 on the
bootstrap and β = 0.7 on the solved current, and ω is never halved.

Passes 1–4 are the main loop, where each pass is an inductive fit plus the
l_i secant. It converges in 4 passes: passes 3 and 4 are the two consecutive
"ok" passes. r_j falls by about 0.3 per pass (1 − ω): 4.5×10⁻³ → 1.4×10⁻³ →
4.2×10⁻⁴ → 1.3×10⁻⁴. The corrective iteration then moves the equilibrium. The
open symbols are the post-corrective check, r_j = 1.8×10⁻³ and
r_I = 4.9×10⁻⁴, which is outside tolerance. Three post-corrective passes
(5–7) bring it back to r_j = 5.9×10⁻⁵ and r_I = 1.4×10⁻⁵.

**Figure 3 — defect A, the grid** (`docs/figures/jbs_loop_fig3_grid.pdf`).
Top panel: Redl j_BS on the delivered equilibrium, from the same physical
profiles sampled in three ways:
- black: on the uniform ψ_N grid (257 points, the reference);
- blue dashed: on a strongly non-uniform grid (513 points uniform in √ψ_N,
  ρ-like), with that grid passed to `evaluate_jBS`;
- vermillion: the same non-uniform arrays read as if they were evenly
  sampled, which is the silent assumption SWB made and bouquet never
  corrected.

Bottom panel: the difference from the reference, as a percentage of peak
j_BS. With the grid passed, the result is grid-independent to interpolation
accuracy. The current-weighted error is 0.17 %, and in the shaded mid-radius
band (0.3 ≤ ψ_N ≤ 0.7) it is below 3×10⁻⁵ of the peak. The legacy reading is
wrong everywhere. The gradient is mis-scaled by dψ_N/du = 2√ψ_N and the
geometry is sampled on the wrong surfaces. The result is low inside
ψ_N ≈ 0.14, 8–12 % of peak high across mid-radius, and nearly doubled (+92 %
of peak) at the pedestal. Its weighted error is 79 %.

**Figure 4 — the fixed point does not depend on the start** (`docs/figures/jbs_loop_fig4_init.pdf`).
The same loop-on reconstruction was started from four initial bootstraps:
- the anchor evaluation (the default);
- the legacy SWB profile (`jbs_init="swb"`);
- the anchor evaluation × 0.8;
- the anchor evaluation × 1.2.

The two scaled starts are made by a figure-only wrapper that multiplies the
loop's first iterate.

Left panel: the delivered j_BS of all four, with the pedestal in the inset.
The curves lie on top of each other. The largest pairwise current-weighted
distance is 2.6×10⁻⁵, 40× below r_j's tolerance, and the l_i targets agree to
6×10⁻⁷. I_BS/I_p is 0.2385 in all four cases.

Right panel: r_j per pass, with the main loop filled and the post-corrective
loop open. A worse start costs passes, not accuracy:
- anchor: 4 + 3 passes;
- SWB: 6 + 3;
- ×0.8: 8 + 3;
- ×1.2: 8 + 3.

Every start contracts at the same rate. The ±20 % starts converge exactly at
the baseline ceiling of 8 passes. That is a deliberately bad start, not the
production one, but it shows how much headroom that ceiling leaves.

**Figure 5 — σ = 0 reproduces the baseline** (`docs/figures/jbs_loop_fig5_sigma0.pdf`).
This is `verify_sigma0_consistency` under the loop on the geqdsk path, with
no IMAS. The σ = 0 draw loop runs from the state anchor with the unperturbed
kinetics and converges in 4 passes.
- Top panel: its j_BS (orange dashed) over the baseline's (black).
- Bottom panel: the difference as a percentage of peak.
- Text box: the invariant's own measures.

The differences are:
- r_j against the baseline: 4.3×10⁻⁵ (tolerance 10⁻³);
- r_I: 8.3×10⁻⁶ (tolerance 10⁻⁴);
- |Δl_i|: 2.0×10⁻⁶ (tolerance 10⁻³), against the delivered baseline
  equilibrium's post-corrective l_i of 0.656455;
- largest pointwise difference: 5.4×10⁻⁵ of peak, at ψ_N ≈ 0.11.

The σ = 0 draw therefore lands on the baseline 12–500× inside the loop's own
tolerances.

## Golden refresh

**Done.** The geqdsk-path golden fixture is regenerated loop-on from its own
stored config (20 requested draws, one thread, the current OFT line; the
build identity is stamped into the fixture and manifest):

- reconstruction loop converged in 4 passes;
- 17 draws archived, 10 in spec; one skipped (an l_i-match candidate's solve
  exhausted `maxits`, the pre-existing failure mode of that solve) and two
  rejected at the post-homotopy stage (known limitation below);
- every archived draw's loops converged: anchor loops 3–6 passes, l_i-match
  candidate loops 4–9 (mostly 7–8; 1–5 candidates per draw), post-homotopy
  0 (accepted as delivered, 3 draws) or 2–4 passes. One loop used its whole
  allowance: draw 4's post-homotopy stage took all 4 of its 4 passes and
  converged on the last one allowed; no other loop reached its ceiling;
- recorded physics: baseline l_i target 0.65384 → 0.65386, baseline
  I_BS/I_p 0.2285 → 0.2380; draw l_i(1) mostly 1–10 % lower and I_BS/I_p
  higher (the draws are different realisations of the same seed, since the
  l_i-match paths differ).

The seeded draw-stream golden (`rng_stream_manifest.json`) is unchanged for
the kinetic channels; only the `jphi` channel's hash moved, because it is
drawn from the baseline j_φ, which now carries the self-consistent bootstrap.

**Rebuilt with input-current archival.** The first refresh went through
plain `Bouquet.generate()`, which archives the achieved flux-surface-average
current, and `test_systematics` mode 1 failed on it (Tests below). The fixture
is now built from a full run of the committed recipe
`tests/golden/regenerate_golden_run.py`: the same stored config, draw
ceilings 8 / 12 / 4 as stored and as applied, the same OFT build, one thread,
~6.2 h. The difference is `store_achieved_jphi=False`, stamped as
`golden_jphi_archival = "input"` on the archive, the fixture and the
manifest. The flag changes only what is written. Draws, coil currents, l_i,
in-spec flags, geqdsks and loop records are identical to the first refresh,
and so are the same skip and the same two rejections. Only `j_phi` /
`j_inductive` differ (1.2–1.9 % of peak, now the solver input).
`golden_manifest.json` changed only in its provenance. The `jphi` stream
hash moved again; the kinetic hashes are unchanged.

## Known limitation: slow post-homotopy divergence of a standard draw

Two of the 20 golden draws were rejected after ~30–60 min each: the
post-homotopy corrective re-solve (at the homotopy's coil bounds, I_p and
p_axis pinned) diverges -- every flux-surface trace fails, the solve spends
its whole iteration budget -- and the diverged state is refused by the
`get_q` axis-collapse guard. The draw is rejected loudly, never accepted.
The corrective iteration's first input is the achieved-derived target
itself, so routing the pass through the corrective iteration did not change
the request that fails. Options, none implemented (details in
`tests/golden/README.md`): fail fast on a growing solve residual or failed
trace; start the corrective iteration from the draw's last corrective input;
`protect_state` for a clearer failure reason; a recorded failure field;
status quo.

## Tests

**Status as recorded at commit 5fd9713** (the fixture itself was built at
1332caf; hashes are pre-rebase). It is NOT claimed for anything later: the
branch has since gained the cherry-picked reversed-Ip commits, a solver-test
commit (the bitwise record comparison skips wall-clock timings) and the
2026-09-29 review fixes, and the solver suites below were not rerun on them.
At the review-fix commits only the fast suite was run (laptop, no solver).

Against the input-current fixture, at 5fd9713:

- fast suite (no solver): 1166 passed on the laptop, including the new
  `test_the_fixture_archives_the_input_current`;
- golden tests (`test_golden_bouquet.py`): 22 passed on the laptop. The
  Linux production build ran 21 passed, before that test was added;
- `test_rng_reproducibility.py`: 23 passed on the laptop;
- `test_systematics.py` (`-m solver`, Linux production build, one thread):
  **3 passed**, no bar changed.
  - Mode 1: max coil drift 0.0189 % (bar 0.3 %), boundary RMS 0.4418 mm
    (bar 0.8 mm).
  - Mode 2: 0.446 / 0.953 mm (bar 6 mm).
  - Mode 3: draws 0 and 3 both replayed and reproduce within every bar
    (l_i(3) within 0.2 %, l_i(1) within 0.3 %, boundary-RMS difference
    1.36–1.39 mm against 2 mm). Draw 3's replay no longer exhausts `maxits`,
    now that mode 3 replays `generate()`'s bootstrap model (Z_eff,
    edge-isolation and floor flags, baseline j_BS, bootstrap scale).

The mode-1 failure against the first refresh (max coil drift 1.2292 %,
F9B) is resolved. Its diagnosis is kept in `tests/golden/README.md`,
"Resolved: mode-1 coil drift after the refresh": the cause was the
archival convention, not the loop, and the fix is the recipe above, not a
code or bar change.

Solver suites on the Linux production build before the refresh; they do
not read the fixture and were not rerun: fsa 6 passed / 1 skipped
(build-aware collapse demonstration), harness 1, loop solver 19 (incl. the
legacy-flag tripwire test), l_i closure 10, seeded reproducibility 12.

At 5fd9713 no test status blocked the PR; that statement is not made for
the later commits until the solver suites are rerun on them.

## Reviewer notes

- The behaviour change is intended: archives made with this release are not
  comparable to earlier ones draw-for-draw; compare them as two bootstrap
  models (`ScanView.bootstrap_model`), or rerun with
  `jbs_self_consistent=False`.
- Drivers that relied on the loop being OFF by default for a "frozen"
  reference arm must now set `jbs_self_consistent=False` explicitly.
- Not in this PR: iterating the diff-mode *baseline* (pinned to the source
  total by design) and a ψ_N-label remap of the kinetic profiles.
- Follow-ups discussed but **not approved or implemented**: a fallback for a
  GS failure inside a blended pass (such a pass fails loudly today); the
  fail-fast guard on a diverging corrective solve (known limitation above);
  and starting each l_i-match candidate's loop from Redl on that candidate's
  own geometry (today later candidates start warm from the draw's previous
  converged bootstrap; their larger first residual is geometric).
