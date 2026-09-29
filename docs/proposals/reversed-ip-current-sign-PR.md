# Reversed-current IMAS sources: bring every source current into bouquet's positive-Ip frame (hotfix)

Branch `fix/reversed-ip-current-sign`, from `main`. Independent of the
self-consistent-bootstrap branch (the same commits cherry-pick onto it; see
"Loop branch" below).

## The bug

bouquet solves every TokaMaker anchor in a **positive-current frame**: the
target is `|Ip|` and the vacuum field is `F0 = |r0·b0|`. Every bootstrap it
recomputes on that anchor is therefore positive. That holds for the legacy
`solve_with_bootstrap` path, the per-draw SWB, and (on the loop branch) the
self-consistent loop's `evaluate_jBS`.

`read_imas_baseline` read a dd's current profiles **with the dd's own sign**. A
dd written for a reversed-current discharge (`ip < 0` in its COCOS) has
negative `j_tor`, `j_total`, `j_ohmic` and `j_bootstrap`. For such a source:

- `j_inductive` and `j_fixed` stayed negative (dd frame), while the recomputed
  bootstrap was positive (anchor frame). The hybrid
  `s_ind·j_ind + s_bs·j_BS + j_fixed` therefore added the bootstrap **against**
  Ip. TokaMaker then rescaled that shape to +|Ip|, which left an edge-current
  hole where the pedestal bootstrap belongs.
- `utils.closure_sign_convention` re-signed the Ip target and the affine P′
  constant with the dd's sign, but never the recomputed bootstrap.
- `utils.unrenormalise_q0` multiplied an anchor-frame q0 (> 0) by a dd-frame
  axis-current ratio (< 0). The q0 target came out negative, the q0 corrector's
  ratio was ≈ −1, and it was refused.
- `fuse_total_err_pct` was off by 2c: it took `|lin + c|` with a negative linear
  part and a positive c.
- `swb_over_fuse_jBS_peak` read ~10⁵, because the FUSE peak was ≤ 0 and got
  floored at 1.
- The `floor_j_BS` clip (≥ 0), `floor_inductive_split` (`j_inductive ≥ 0`) and
  the draw path's `np.maximum(·, 0)` floors all assume the positive frame. On
  dd-frame currents they would zero or relabel whole components.

The symptoms were:

- The closure's posterior asked for `s_bs ≈ −1`, which the strict `s_bs` bound
  correctly refused.
- With a bootstrap prior that forbids `s_bs < 0`, the raw components missed Ip
  by about −2·I_BS/Ip, and `s_ind` was pushed high.
- Grad–Shafranov solves failed with `maxits` on the edge-hole profile.

Only sources with `ip < 0` are affected. The reconstruction (g-file) path is
not affected, because its current split comes from a TokaMaker fit in the
positive frame.

## The fix (at the reader)

`read_imas_baseline` multiplies **every current profile it reads** by
`s = sign(equilibrium ip)` (new helper `io.imas.source_current_sign`; `+1.0`
for `ip ≥ 0` and for a zero or non-finite `ip`):

| multiplied by `s` | read unchanged |
|---|---|
| `core_profiles` `j_total`, `j_tor`, `j_ohmic`, `j_bootstrap` | kinetics, `Z_eff`, fast and equilibrium pressure |
| every beam-source `j_parallel` (→ `j_NBI`) | rotation, `E_r`, transport coefficients |
| `equilibrium.profiles_1d.j_tor` (→ `jphi_diff`) | the dd's own `q` (`q0_dd`, recorded raw; the gate reads `\|q0_dd\|`) |
| user `FixedComponentsConfig.j_NBI` / `j_RF` (they replace dd quantities, so they are in the dd's orientation) | boundary outline, `F0 = \|r0·b0\|` (already orientation-free) |

The parallel→toroidal ratio `j_tor/j_total` does not depend on the sign, so
flipping the inputs is bitwise the same as flipping every derived component.

**Recorded, never silent:**

- New fields `Baseline.source_current_sign` and `Baseline.source_b0_sign`. The
  B0 sign is recorded only; nothing is flipped on it.
- The same two keys in `li_metrics` on every IMAS forward solve.
- `ip_closure.source_current_sign`.
- The IMAS archive's `_baseline` group gets the attrs `source_current_sign`,
  `source_b0_sign` and `current_frame` (`utils.stamp_source_orientation`,
  called from `Bouquet.generate`, so it also reaches shards and merged
  archives).
- A log line for a reversed source.
- A `UserWarning` for a dd whose `core_profiles` total integrates against its
  own `ip`. No sign convention can repair that dd.
- In the ohmic closure, a printed warning if `current_direction_sign` is still
  −1 after the normalisation.

`closure_sign_convention` is unchanged and stays in place as a guard for
callers that pass in their own components. On reader output it now returns +1.
The raw-source plot overlays (`plot_input_vs_recon`, `plot_jphi`) apply the
same factor, so a reversed source is not drawn upside down against its own
solve.

**Positive-Ip behaviour is unchanged.** The factor is exactly `+1.0`, so every
array is the same object arithmetic as before. See "A/B identity" below.

## Sign audit (every site that meets a source current, main line numbers)

| site | before | after | changed? |
|---|---|---|---|
| `io/imas.py` `read_imas_geometry`: `F0 = abs(r0*b0)` | positive frame | same | no (it defines the frame) |
| `io/imas.py` `Ip_target = abs(ip)` | magnitude | same | no |
| `io/imas.py` currents `j_total/j_tor/j_ohmic/j_bootstrap`, NBI `j_parallel`, eq `j_tor` (`jphi_diff`), user `j_NBI/j_RF` | dd sign | × `sign(ip)` | **yes**: the fix |
| `io/imas.py` `j_inductive = j_phi − j_BS − j_NBI − j_RF` | dd frame | positive frame (follows from the inputs) | via inputs |
| `io/imas.py` `sawtooth.q0_dd` | dd COCOS | same (recorded raw; gate uses `\|q0_dd\|`) | no, documented |
| `io/imas.py` `aux` rotation / `E_r` | lab signs | same | no, flagged (see "Not in this PR") |
| `run.py` `_forward_solve_imas_baseline` first `solve_jphi(bl.j_phi)` | negative shape renormalised to +\|Ip\| (same equilibrium) | positive shape; bitwise-identical equilibrium | via inputs |
| `run.py` SWB `solve_with_bootstrap(..., bl.Ip_target)` → `_swb_jbs_to_toroidal` | anchor frame (+) | same | no (anchor frame is the frame) |
| `run.py` `floor_j_BS` clip of `j_BS_swb` | clips a + bootstrap; consistent only for + data | same, now consistent for every source | via inputs |
| `run.py` diff mode `jBS_diff = j_BS_src − j_BS_swb` | **mixed frames** on ip < 0 | one frame | via inputs |
| `run.py` rescale mode li-proxy closure `j_ind + s·j_BS_swb + j_fixed` | **mixed frames** | one frame | via inputs |
| `run.py` `ratio = j_BS_swb.max()/max(j_BS_src.max(),1)` (`swb_over_fuse_jBS_peak`) | ~10⁵ on ip < 0 | O(1) | via inputs |
| `run.py` `fuse_tot_err_pct = (\|lin(FUSE_tot)+c\| − Ip)/Ip` | off by 2c on ip < 0 | correct | via inputs |
| `run.py` `closure_sign_convention` → `sgn`, `_Ip_signed`, `_c_signed` | −1 on ip < 0, bootstrap left unsigned | +1; warns if still −1 | guard kept; warning added |
| `run.py` `close_ip` / `close_ip_q0` / `close_ip_structured(_soft)` (all take `sgn·Ip`, `c_signed`) | mixed-frame inputs | one frame | via inputs |
| `run.py` q0 predictors: `unrenormalise_q0(q0_anchor, j_achieved0, j_requested0=axis(FUSE_tot))` | q0_target < 0 on ip < 0 | > 0 | via inputs |
| `run.py` `q0_gate_admits` | magnitudes | same | no |
| `run.py` q0 / structured correctors (`ip_of = lin + c_signed`, ratios `q0_cur/q0_target`) | ratio ≈ −1 → refusal | consistent | via inputs |
| `run.py` `ip_roundtrip_gate`, `closure_health` (`Ip_target_signed`) | signed consistently with the wrong frame | +1 frame | via inputs |
| `run.py` recorded-only `_would_be` scales (`np.sign(tot)`) | per-integrator sign | same | no |
| `run.py` `Ip_achieved` check (`abs`) | magnitude | same | no |
| `run.py` `imas_corrective_jphi` (`abs(Ip_target)`; `TokaMaker_interface` `np.maximum(j_phi_input, 0)`) | would floor a negative request | consistent | via inputs |
| `run.py` `verify_sigma0_consistency` (`floored = j_inductive <= 0`) | whole profile "floored" on ip < 0 | correct | via inputs |
| `baseline.floor_inductive_split` (`j_inductive ≥ 0`) | would move the whole negative inductive into j_BS | consistent | via inputs |
| `TokaMaker_interface` draw path (`perturb_kinetic_equilibrium`, SWB spike floor `floor_j_BS`, `np.maximum` floors, `_renormalize_target_to_Ip` / `_R2` sign from `dot(probe, ·)`) | mixed frames on ip < 0 | one frame | via inputs |
| `utils.li_value` (`sgn = sign(Ip)`), `structured_li_model` (`Ip_target_signed`) | sign-aware | same | no |
| g-file path `F0 = abs(R_center·B_center)`, `Ip_target = abs(eqdsk.Ip)` | positive frame, split fitted in it | same | no |
| writer: TokaMaker `save_eqdsk` (default COCOS 7) | `CURRENT > 0`, `BCENTR > 0` for every source | same | no, documented (below) |
| `io/imas.py` `write_imas_draw` (IMAS export) | positive-frame `ip` and currents into the template | same | no, documented; orientation recorded on the archive |
| `plotting._imas_input_profiles`, `plot_jphi` raw FUSE overlay | dd sign against a positive solve | × `sign(ip)` | **yes** (cosmetic) |

## What sign the delivered g-file carries

All delivered g-files use one convention, for every source, normal or reversed.
They are written by TokaMaker's `save_eqdsk` (its default COCOS 7) from the
positive-frame solve, with `CURRENT > 0`, `BCENTR > 0` and `SIMAG > SIBRY`. For
example, the shipped synthetic example's input g-file has `BCENTR < 0`, and its
bouquet baseline g-file has `BCENTR > 0`.

- The delivered g-file carries **neither** the experiment's Ip sign **nor** its
  Bt sign. That was already true for normal-orientation DIII-D-like sources
  (Bt < 0) before this change, and this PR does not change it.
- For a reversed-Ip source the delivered g-file is `CURRENT > 0`, like every
  other bouquet g-file. It does not carry the experiment's negative Ip.

**Recommendation: keep the writer as is in this hotfix**, and treat restoring
the orientation as a separate, opt-in decision:

- Static MHD quantities do not depend on the orientation. These are the
  equilibrium, l_i, |q|, Δ′ and δW. Flipping Ip alone is a mirror reflection
  φ → −φ, and flipping Ip and Bt together is a full field reversal. Both are
  symmetries of the MHD equations, so an EFIT-style consumer that only needs
  the equilibrium gets the same answer from the positive-frame g-file.
- Quantities that carry a **direction** relative to the field are passed
  through in the source's lab signs and are **not** transformed. These are
  toroidal rotation, `E_r`, and the E×B and diamagnetic frequencies. A consumer
  that combines a delivered g-file with flows needs the two to share an
  orientation. For example, a resistive-layer code needs the sign of
  ω_E relative to ω_*. This was already the situation for every source; the
  hotfix only makes the currents consistent.
- **Proposed follow-up (not in this PR):** an opt-in
  `export_bundle(..., orientation="source")` / `write_imas_draw(..., orientation="source")`.
  It would restore the recorded orientation with
  `Ip → s_I·Ip`, `ψ → s_I·ψ`, `F → s_B·F`, `q → s_I·s_B·q`, where
  `s_I = source_current_sign` and `s_B = source_b0_sign`. A matching option would
  let the flows be transformed into the delivered frame instead. Choosing
  between the two, and making one of them the default, is a physics decision
  for the downstream codes. It is out of scope for a hotfix.

## Tests

All inputs are synthetic: the shipped D3D-like example dd and dds built
in-process. `tests/_mirror_dd.py` mirrors a dd into all four (Ip, B0)
orientations. It multiplies the currents by `s_ip`, `b0` by `s_b0` and `q` by
`s_ip·s_b0`, and touches nothing else.

- `tests/test_reversed_ip_current_sign.py` (fast, no solver, 33 tests):
  - Every orientation of the example dd (all 3 slices) and of a minimal dd reads
    to a **bit-identical** Baseline. The only difference is the dd's own
    `q0_dd`, in sign.
  - The sign is recorded and logged. An `ip > 0` read equals the pre-fix
    formulas bit for bit. An inconsistent dd warns. User fixed currents follow
    the dd.
  - `closure_sign_convention`, `close_ip`, `unrenormalise_q0` (q0_target > 0),
    the FUSE-total error, the SWB/FUSE peak ratio, the `floor_j_BS` clip and
    `floor_inductive_split` behave identically on mirrored reads. Each check
    comes with a **witness** that the pre-fix dd-frame inputs did not.
  - The plot overlay and the archive stamp.
- `tests/test_reversed_ip_solver.py` (`-m solver`): the full `prepare_baseline`
  of all four orientations, run under four closure paths:
  - `diff`;
  - `diff` with `floor_j_BS` and `isolate_edge_jBS`, which exercises the SWB
    clip and the edge-spike isolation;
  - `ohmic` with the bootstrap channel;
  - `ohmic` with the structured-soft closure, with an l_i row and the sawtooth
    q0 gate.

  On builds with the bootstrap loop, each path runs with the loop off and on.
  **The bar is bitwise identity** of every Baseline array, the solved ψ, q,
  globals, l_i, the multipliers and the whole Ip-closure record. The only
  exceptions are the orientation records. This is the right bar because the
  normalisation is exact in IEEE arithmetic, every reader operation is odd in
  the current, and the forward solve is deterministic at `nthreads=1`.
  Machine-precision *tolerance* would be weaker and is not needed.

### Results

These results use the production Linux build of OpenFUSIONToolkit (abbfc6f), single-threaded solver processes,
and the fast suite on both Linux and macOS. Counts are verbatim.

| suite | build | result |
|---|---|---|
| fast (`pytest`), before this branch | `main` 135bf69 | `1025 passed, 49 deselected, 68 warnings` |
| fast (`pytest`) | this branch | `1058 passed, 49 deselected, 69 warnings` (Linux and macOS) |
| `pytest -m solver tests/test_reversed_ip_solver.py` | this branch | `17 passed in 279.95s (0:04:39)` |
| `pytest -m solver` (all other solver tests) | this branch | `2 failed, 30 passed, 1075 deselected, 91 warnings in 5541.25s (1:32:21)` |
| `tests/test_systematics.py::test_mode3_production_reproduces_golden` | this branch | fails with `l_i(1)` replay 0.82397 vs golden 0.85596 (3.74 % vs its 3 % bar). This is the documented `main`-fixture failure on this OFT build, with the same numbers recorded before this change (the loop branch's golden refresh fixes it). It is a g-file path the change does not touch, and the g-file A/B below is bitwise identical. No bar was changed. |
| `tests/test_fsa_current_integral.py::test_get_q_collapses_silently_on_an_unclipped_grid` | `main` 135bf69 **and** this branch | fails on both: a known build-dependent guard (the OFT build no longer collapses the tracer), unrelated to this change |

The first run of the new solver test failed 4 of 17. The cause was the test itself: a nested record key
(`li_metrics.ip_closure.source_current_sign`) was not in its skip list. Every array had already compared
bitwise. The test was fixed to check orientation records at any nesting against the mirror, and the rerun
gave `17 passed`.

**A/B identity for positive Ip (this branch vs `main` 135bf69, bitwise):**

- **Synthetic IMAS baseline.** The shipped D3D-like dd was forward-solved under `diff`, `diff`+floor/isolate,
  `ohmic`/bootstrap and `ohmic`/structured-soft. That compared 88 arrays (every Baseline current, kinetics,
  solved ψ, q, globals, and every numeric closure record) and 472 scalars. **0 differences.** The only
  additions are the new orientation records (`source_current_sign`, `source_b0_sign`).
- **The IMAS archive's `_baseline` group** from `generate()`: 23 datasets and 20 attrs, **0 differences**. The
  draws of that run could not be compared, because on this example both builds stall identically in the first
  draw. The q-profile tracer fails, `DLSODE` returns NaN, and the stall reproduces on `main`. It is not related
  to this change and is flagged separately.
- **Golden-fixture example, g-file path.** A seeded `from_geqdsk` run (reconstruction + 2 draws, `nthreads=1`)
  compared 82 datasets and 82 attrs, including every draw's eqdsk bytes and currents. **0 differences.**

**Pre-fix witness (`main` 135bf69, reversed-current mirror of the same dd):**

- `ohmic`/bootstrap is refused: `bs_scale -1.086 is outside (0.2, 5)`.
- `ohmic`/structured-soft is refused: `s_bs(psi) reaches -0.448`.
- `diff` solves, but to a different baseline.

With the fix, all four orientations give the positive-orientation result bit for bit.

### Loop branch

The same commits were cherry-picked onto `feat/jbs-self-consistent-loop` with no code conflicts. The docs
conflicted only in `CHANGES_SUMMARY.md` and `archive-schema.md`, and both entries were kept. The loop's
`evaluate_jBS` path is covered by the reader fix: every loop site consumes the reader's currents
(`bl.j_BS`, `bl.j_inductive`, `FUSE_tot = bl.j_phi`), and the loop's own `q0_target` comes from the same
`unrenormalise_q0`. The solver test runs every closure path with the loop both OFF and ON.

| suite | build | result |
|---|---|---|
| fast, before | loop head 5fd9713 | `1166 passed, 87 deselected, 77 warnings` |
| fast | loop + hotfix | `1199 passed, 87 deselected, 78 warnings` (Linux and macOS) |
| `-m solver tests/test_reversed_ip_solver.py` | loop + hotfix | `34 passed in 472.85s (0:07:52)` |
| `-m solver tests/test_jbs_loop_solver.py` | loop + hotfix | `20 passed in 1850.76s (0:30:50)` |
| `-m solver` (all other solver tests) | loop + hotfix | `32 passed, 1 skipped, 1253 deselected, 53 warnings in 2571.32s (0:42:51)` (the skip is the build-aware get_q axis-collapse guard) |

On the loop branch, the first run of the new solver test failed 6 of 34. Both causes were in the test: the same
nested-key issue, and the loop record's wall-clock `wall_s`. The test now excludes that key by name, as a
measurement of the machine and not of the equilibrium. The rerun gave `34 passed`.

The positive-Ip A/B against loop head 5fd9713 compared 244 arrays and 1298 scalars across 8 configurations
(4 closure paths × loop off/on). **0 differences**, apart from `wall_s` and the new orientation records.

The pre-fix witness on the loop is the same as on `main`. The loop-ON bootstrap channel is refused with
`bs_scale -1.171`, and structured-soft is refused with `s_bs -0.358`.

**Validation on real reversed-current data (private, not in this repo).** The fixed build reproduces an earlier
in-process diagnostic of the same fix bit for bit. That covers every closure scalar, the delivered profiles and
the g-file bytes, on four slices of two reversed-current discharges. All four now deliver:

- the loop converges in 4–5 passes;
- the raw Ip miss is +0.5…+6.3 %;
- `s_bs` stays at 0.99–1.08;
- there are no closure-limited flags.

Before the fix, the same slices were refused (`s_bs → -0.27`), failed with `maxits`, or delivered with the
bootstrap opposing Ip.

## Not in this PR (flagged)

- **Flow orientation for downstream codes.** Rotation, `E_r`, and the E×B and
  diamagnetic directions stay in the source's lab signs. See the g-file section
  above.
- **IMAS export orientation.** `write_imas_draw` writes positive-frame `ip` and
  currents into the source's template. The source orientation is recorded on
  the archive but is not restored.
- **Existing results on reversed-current sources are invalid** and must be
  regenerated. Every bouquet baseline, closure, draw and archive built from an
  `ip < 0` dd before this fix is affected.
