# Continuous integration / test policy

Two tiers:

| tier | what | when | where | time |
|------|------|------|-------|------|
| **fast** | unit suite (golden-fixture reads, filtering, io, etc.) — no OFT | every push + PR | GitHub-hosted `ubuntu-latest` (automated) | ~seconds |
| **solver** | live OpenFUSIONToolkit GS solve: reconstruction-free replay + systematics (`-m solver`) | **pre-merge** | **manual** (see below) | ~13 min |

`pytest` is **fast-by-default** (`addopts = -m "not solver"` in `pyproject.toml`).
Run the comprehensive suite explicitly with `pytest -m solver`.

## Why the solver suite is manual (for now)
The solver tests need the **OFT fork** built (they depend on the jphi-linterp Ip
fixes — the Python Ip secants were removed in favour of the native hold).  OFT is
not on PyPI, and running a self-hosted runner for a **public** repo would put
PR-controlled code on lab infrastructure.  So until OFT publishes to PyPI we keep
the comprehensive suite off automated CI and run it deliberately.

## Workflows
- `.github/workflows/tests.yml` — **fast** suite on `push`/`pull_request`
  (automated), run on a 2-way dependency matrix: **pinned** (the validated
  numpy/scipy floor — bump deliberately) and **latest** (unpinned canary).
  An upstream numpy/scipy release then breaks only the `latest` leg, as its
  own signal, instead of failing unrelated feature PRs (this bit us when
  scipy 1.18 + numpy 2.5 changed scalar-conversion semantics).
- `.github/workflows/solver.yml` — **solver** suite, `workflow_dispatch` only
  (manual "Run workflow" button / `gh workflow run solver.yml`).  Not a required
  check, so it never blocks a merge.  It already contains the `runs-on`, env, and
  `merge_group` wiring (commented) for when automation is set up.

## Pre-merge checklist (the manual gate)
Before merging a branch that touches the perturbation/solve pipeline into `main`:
1. On a machine with OFT built from [`LCFS_grid_pack`](https://github.com/StuartBenjamin/OpenFUSIONToolkit/tree/LCFS_grid_pack) (open OpenFUSIONToolkit [PR #365](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/pull/365)): `pytest -m solver` (or trigger
   `solver.yml` on a self-hosted runner via `gh workflow run`).
2. Confirm all solver tests pass; note it in the PR.
3. The automated `fast` check must also be green.

## Branch protection (`main`)
Require the **`fast (pinned)`** status check (+ optionally `fast (latest)` and
a review).  **Do not** require `solver` while it's manual — a required check
with no runner would block merges forever.

## Future: automate the solver gate
Once the OFT Ip fixes are merged upstream **and** OpenFUSIONToolkit publishes to
PyPI, flip `solver.yml` to fully automated:
1. `runs-on: ubuntu-latest`
2. add a `pip install OpenFUSIONToolkit` step (drop the `OFT_PYTHONPATH` /
   self-hosted bits)
3. re-enable the `merge_group:` trigger
4. add `solver` to the required status checks + enable the merge queue for `main`

That makes the comprehensive suite a pre-merge gate that runs **once per merge,
not per commit**, on GitHub-hosted runners — no fork hosting, no self-hosted
runner.  (Self-hosted on feynman or a fork-built wheel remain fallback options
if you need automation before OFT is on PyPI.)
