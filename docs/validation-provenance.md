# Validation provenance of the unified-engine branch

What the unified-engine branch was validated on, where, at which code
revision, and with what result. Aggregate numbers only; the private cases
carry no identifiers here.

## History

The work was developed as a long series of commits and rebuilt for review
as twelve chapter commits on top of `main`, followed by one commit per
review fix or owner decision. The original history is kept, unchanged, on
the archival tag **`archive/engine-unified-2116923`**. Each chapter
commit's message lists the original commits it squashes and its merge
resolution (`git log origin/main..HEAD`).

## Where and at which revision

- **Original validation:** on the archival tag's history, at `c651bb1` and
  at `bab3902` (the canonical coil-solve mode). Numbers quoted from it
  refer to those commits.
- **Re-validation of the rebuilt branch:** on the owner's compute cluster,
  one thread, OFT build `20260929_7da4f18`, at `1a15685`. `3d974e6`
  (engine-dependent defaults resolved at `prepare_baseline()`) changes
  nothing that a configuration naming its engine at construction, or a
  stored configuration, runs with, so those numbers stand for it.
- **Goldens are build-specific:** each golden fixture records the OFT build
  that generated it; the golden tests compare at unchanged bars on every
  build and name both builds on a mismatch (warning on a pass, failure
  message on a failure; [`tests/golden/README.md`](../tests/golden/README.md),
  "Build-specific goldens").
- **Solver suite at the tip:** re-run after the reconstruction pass ceiling
  change 8 -> 12 (`fe53920`, owner-approved).

## Results

| check | revision | result |
|---|---|---|
| fast suite | `1a15685` | 2897 passed, 0 failed (2919 with the tests `3d974e6` adds) |
| solver suite | `1a15685` | 165 passed, 2 failed, 1 skipped; both failures were the q0-pinned structured loop test and the comparison that needs its residual, one pass short of the former ceiling of 8 |
| solver suite | `fe53920` | 167 passed, 1 skipped (a comparison that needs an older OFT build), 0 failed |
| golden fixture, unified engine | `1a15685` | 20 attempts, 17 archived, 4 in spec; reconstruction 66 s, draws 124 s per equilibrium |
| golden fixture, legacy engine | `1a15685` | 20 archived, 12 in spec (the previous fixture's flags on every draw); draws 458 s per equilibrium |
| `tests/test_systematics.py` vs the legacy golden | `1a15685` | 3 passed, bars unchanged |
| private cases: 11 single slices + a 30-slice time series | `1a15685` | all converged; agree with the original validation to 1e-7 to 2e-4 relative (3e-4 on one boundary rms), deltas explained by two declared changes (`ce10dec`, `a769882`) |
| legacy vs engine, I_BS/I_p | `1a15685` | within 0.8 % on 10 of 11 slices (3.9 to 7.3 % apart before `9f6abed`) |
| σ=0 gate (as widened in `eee55bf`) | `1a15685` | 11/11 pass |
| engine reconstruction wall time | `1a15685` | 53 to 113 s on 8 slices, 125 to 136 s on 3; time series 52 to 59 s |
| network | `1a15685` | zero network attempts in every run |

Details of each item are in [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) (the
first Unreleased entry) and [`tests/golden/README.md`](../tests/golden/README.md).
Everything measured on the repository's synthetic examples can be re-run
from this tree (`pytest -m solver`, `tests/golden/regenerate_golden_run.py`).
