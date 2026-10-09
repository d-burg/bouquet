# Validation provenance of the unified-engine branch

The unified-engine work was developed as a long series of commits and then
rebuilt for review as twelve chapter commits on top of `main`, followed by
the review-driven commits. The original history is kept, unchanged, on the
archival tag **`archive/engine-unified-2116923`** (tip `2116923`). This page
maps each chapter to the original commits it squashes and says where the
validation numbers quoted for the branch were measured.

## Chapters and the original commits they squash

Each chapter commit's message lists the original commits it squashes (first
and last below, in the order listed), and its merge resolution: hunks that
duplicate work already on `main` take `main`'s reviewed form, frozen-copy
snapshot / AST tests are retired rather than added, and every
`tests/golden/**` change is moved to chapter 12 so that only the final
fixture blob enters history.

| chapter | commit | original commits (first … last, on the tag) | count | notes |
|---|---|---|---|---|
| 1. MSE pitch angles on the structured closure (opt-in) | `e095347` | `47d6f9c` | 1 | |
| 2. Self-consistent bootstrap loop kernel (opt-in) | `c6cd587` | `940d8a0` … `adc2db1` | 10 | |
| 3. Self-consistent bootstrap on by default; archive schema v3 | `34cf405` | `3c26048` … `14b0c2b` | 28 | golden changes moved to ch 12; `14b0c2b` pulled forward from the post-merge range |
| 4. Bootstrap-loop review fixes; one reconstruction state | `b1b11e6` | `5913559` … `0f68543` | 29 | the in-line reversed-Ip copy `5fd9713..82e1344` dropped (replaced by `main`'s) |
| 5. Structured-MSE review fixes; engine base | `f6cdf86` | `a70abec` … `2ce3209` | 15 | |
| 6. Unified reconstruction engine (opt-in) | `1f10f42` | `194677a` … `373d2fb` | 10 | |
| 7. Draws on the unified engine | `9227f05` | `554edfb` … `5f720ef` | 14 | frozen-copy snapshots retired |
| 8. Edge-pressure helper; validation fixes | `01b92b4` | `e0b5e01` … `f4c62d8` | 15 | of the orientation range `5f720ef..60a609f` only `f54ea16` kept (the rest duplicates `main`) |
| 9. Approved default changes (offset, IDS residual inductive, `fd_chord`) | `92c8cba` | `a95617c` … `d874822` | 7 | frozen-copy snapshots retired |
| 10. Review-fix rounds 2–3; canonical coil-solve mode; figures | `83a0f88` | `4ee1869` … `c651bb1` | 41 | golden regeneration moved to ch 12 |
| 11. Post-merge fixes | `4fc2b79` | `e7d7b5c` … `1cc541d` | 4 | |
| 12. Golden fixture (final blob) and golden-consuming tests | `1a15685` | `3c26048` … `2116923` | 16 | collects `tests/golden/**` from every range; intermediate fixture regenerations not replayed |

The full list for each chapter, and its merge resolution, is in the chapter
commit's message (`git log origin/main..HEAD`).

## Commits after the chapters

Owner decisions and the fixes from the review rounds of 2026-10-06 / 07,
each its own commit with its tests and the fast-suite line in the message:

| commit | change |
|---|---|
| `9f6abed` | one `<j.B>` -> `<j_phi>` conversion in the package (declared physics change) |
| `4eaba56` | the unified engine becomes the default (`reconstruction_engine="unified"`) |
| `2c8c2c2` | hygiene and provenance: no discharge numbers, OFT build digests, anchor doc |
| `0a0cf2a` | CHANGES_SUMMARY / README: what reproduces legacy, what moves by default |
| `eee55bf` | σ=0: draws solve under the reconstruction's coil regularisation; the gate widened |
| `a769882` | bootstrap loop: `r_j` / `r_I` measured against the bootstrap the pass solved |
| `2671a49` | closure health: the ±50 % bootstrap prior flagged on every path |
| `fe2f967` | IDS l_i row: the source `li_3`'s normalisation radius resolved |
| `f3cebe1` | IDS export: no pressure-driven current in any parallel field; round trip |
| `ce10dec` | IMAS time matching: windowed core_sources slice, dt recorded, OFF before an entry's record |
| `6098a23` | `run_slices`: `on_refusal="record"` is the default |
| `2375b18` | stored-config replay is loud |
| `51279b1` | engine MSE: the chord Jacobian re-taken at convergence and recorded |
| `a7231f7` | engine MSE: restore-and-flag on stage failure; chord chi^2/N flags |
| `4d555e2` | the solver's `P'` rescale recorded; docs catch-up; growth abort documented inert |
| `b8abd83` | engine MSE: a failed stage keeps the Jacobian record it took |
| `9edbdc6` | tests: two toy kernel tests take their own pass ceilings under the solved-state residual |
| `f3d6799` | IMAS time match: a 10 µs floor when neither time base has a local step |
| `8285201` | `mse_chi2n_flag = 10.0` and `MSE_JACOBIAN_MAX_REFRESHES = 3` marked owner-approved |
| `1a15685` | golden fixtures regenerated on the unified-engine default; slim legacy JSON golden |
| `3d974e6` | engine-dependent defaults resolve at `prepare_baseline()`, not at construction |

## Where the validation numbers were measured

- **The original validation** (the engine against the legacy path on the
  synthetic examples and on the owner's private cases: single slices, a
  time series, the σ=0 checks, the solver suite) was run on the archival
  tag's history, at `c651bb1` (the tip of the range squashed into chapter
  10) and at `bab3902` (the canonical coil-solve mode, in the same range).
  Numbers quoted from that validation refer to those commits.
- **The rebuild's re-validation** was run on the owner's cluster at
  `1a15685` (the tip of this branch before `3d974e6`): the fast and solver
  suites, the golden fixtures, and the re-run of the single slices, the time
  series and the legacy comparison against the original validation. Its
  aggregate results are recorded in [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
  (the first Unreleased entry) and in
  [`tests/golden/README.md`](../tests/golden/README.md). `3d974e6` changes
  nothing a factory configuration that names its engine at construction,
  or a stored configuration, runs with, so those numbers stand for it.
- The private cases are not part of the repository; only aggregate numbers
  are quoted, with no case identifiers. Everything measured on the
  repository's synthetic examples can be re-run from this tree
  (`pytest -m solver`, `tests/golden/regenerate_golden_run.py`).
