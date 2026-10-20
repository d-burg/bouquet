# Proposal — export IDS and/or geqdsk+profiles from the *live* TokaMaker equilibrium

> **Status:** scoping / design (not yet implemented).
> **Branch:** `feat/live-equilibrium-export` off `main`.
> **Date:** 2026-07-03.
> **Supersedes:** the `TODO(backend)` interim note in `bouquet/io/imas.py`.

## 1. Problem

Today, at generate time bouquet stores **only a geqdsk** per draw (the raw
`save_eqdsk` bytes) plus the perturbed profiles and scalars, for *both* the
geqdsk and IMAS/OMAS source paths. IMAS/OMAS output is produced later, on
demand, by `io.imas.write_imas_draw` / `export_imas_drawset`, which
**reconstruct** an IDS from the stored g-file. That reconstruction is
explicitly interim (`io/imas.py` module TODO):

- the `equilibrium` IDS is faithful (lossless to the archived eqdsk grid,
  machine-precision GS), and `core_profiles.j_tor` is exact; **but**
- the **parallel-current split** (`j_ohmic` / `j_bootstrap` / `j_total`) is
  reconstructed with the *baseline* ratio `c(ψ)=j_tor/j_total` taken from the
  template IDS — exact **only** when the draw's flux geometry equals the
  baseline's. Every perturbed draw moves the geometry, so the split carries a
  systematic error that grows toward the edge / at low aspect ratio.

The live TokaMaker equilibrium that produced each draw holds the exact
flux-surface geometry, but we throw that information away when we serialise to
a g-file and later re-read it.

## 2. Goal

Give the user, at generate time, the ability to export — per draw — **either or
both** of:

- **(A) an IMAS/OMAS IDS** built from the *live* equilibrium, with an **exact**
  toroidal↔parallel current conversion (no baseline-ratio approximation); and
- **(B) a geqdsk + profiles bundle** (the g-file we already write, plus the
  matching kinetic/current profiles in a portable form — p-file / i-file /
  profiles JSON), as a self-contained hand-off to codes that don't consume IDS.

…without a second GS solve, and without depending on an OFT feature that does
not yet exist.

## 3. Key finding — this is buildable now (no OFT change required)

The `io/imas.py` TODO says the exact IDS should come "once OFT exposes an
IMAS/ODS export." OFT does **not** expose an ODS export — but it already
exposes, on the live `mygs`, every flux-surface-averaged (FSA) metric the exact
conversion needs. Verified against the installed OFT
(`OpenFUSIONToolkit/TokaMaker/_core.py`):

| Need | TokaMaker method (live `mygs`) | Returns |
|---|---|---|
| Flux functions | `get_profiles(psi, npsi)` | `ψ̂, F, F′, P, P′` |
| q + FSA geometry | `get_q(psi, npsi, compute_geo=True)` | `ψ̂, q, [⟨R⟩, ⟨1/R⟩, dV/dψ], L_lcfs, …` |
| Sauter / FSA fields | `sauter_fc(psi, npsi)` | `f_c, [⟨R⟩,⟨1/R⟩,⟨a⟩], [⟨\|B\|⟩,⟨\|B\|²⟩]` |
| Exact flux integrals | `compute_flux_integral(psi, field)` | `∫ f dA` |
| Globals | `get_globals()` | `Ip, [R_Ip,Z_Ip], ∫dV, ∫P dV, Φ_diamag, Φ_tor` |
| ψ map / boundary | `get_psi()`, `trace_surf(psi)`, `get_xpoints()` | ψ(R,Z); LCFS contour; X-points |

Crucially, `bouquet.physics.parallel_to_toroidal` **already accepts a `geom`
dict** for a per-surface exact conversion — today it is fed the baseline ratio;
the feature simply feeds it the **draw's own** live FSA metrics
(⟨R⟩, ⟨1/R⟩, ⟨|B|⟩, ⟨|B|²⟩, dV/dψ, F). So the physics plumbing exists; what is
missing is (i) capturing those metrics per draw and (ii) an exact
parallel⇄toroidal inverse for the IDS write-back.

**Conclusion:** implementable entirely within bouquet against the current OFT.
A native OFT ODS export remains a nice-to-have (Phase 3), not a blocker.

## 4. Design

### 4.1 Architecture — capture at generate, export on demand (recommended)

Two candidate shapes:

- **Eager:** write IDS/geqdsk files for every draw *inside* `generate_bouquet`.
  Rejected as the default — bloats generation with file I/O, couples the solve
  loop to output formats, and duplicates work when the user only wants a subset.
- **Capture-then-export (recommended):** at generate time, capture the small
  **live-equilibrium FSA payload** the exact conversion needs and store it in
  the per-draw HDF5 group. Exports (`export_imas_drawset`, a new
  `export_geqdsk_profiles`) then run on demand from the archive and are
  **exact** because the geometry travelled with the draw. This preserves the
  archive-as-source-of-truth design and keeps generation lean. An **eager**
  convenience (`export_formats=[...]` on generation) can be layered on top for
  users who want files immediately.

### 4.2 What to capture per draw (the new payload)

A compact FSA block on the equilibrium ψ grid, written as optional datasets
under `scan/<key>/<draw>/eq_fsa/`. Sampled at **npsi = 257 by default**
(configurable; **do not go below 129** — see §8): this matches the archived
257² eqdsk so no radial resolution is lost relative to the g-file, and it is
what actually resolves the edge bootstrap spike, where the parallel⇄toroidal
correction is largest and a coarse grid would alias the pedestal current.

- `psi_N`, `F` (=R·Bt), `Fprime`, `P`, `Pprime`  — from `get_profiles`
- `q`, `R_avg` (⟨R⟩), `one_over_R_avg` (⟨1/R⟩), `dV_dpsi` — from `get_q(compute_geo=True)`
- `f_trap` (f_c), `B_avg` (⟨|B|⟩), `B2_avg` (⟨|B|²⟩) — from `sauter_fc`
- global scalars as attrs: `Ip`, `∫dV`, `∫P dV`, `Phi_diamag`, `Phi_tor`, F0.

Size: ~10 float64 arrays × 257 pts ≈ 21 kB/draw — still negligible vs the
eqdsk bytes.

Provenance: a per-draw attr `eq_fsa_source = "live_tokamaker"` (vs
`"eqdsk_reconstruct"`) so exporters and downstream users know which fidelity
they have. Legacy archives (no `eq_fsa/`) fall back to the current
baseline-ratio path with a warning — no breakage.

### 4.3 Export target A — IMAS/OMAS IDS (exact)

`write_imas_draw` / `export_imas_drawset` gain an exact path: when `eq_fsa/` is
present, build `core_profiles.{j_total, j_ohmic, j_bootstrap}` from the stored
toroidal components using the **draw's** FSA metrics via
`physics.parallel_to_toroidal` (inverse direction), instead of the baseline
`c(ψ)`. `equilibrium` IDS is unchanged (already faithful). Same public
signatures; a new `fidelity={"auto"|"exact"|"reconstruct"}` kwarg lets the user
force or inspect the path. `"auto"` = exact when `eq_fsa/` present, else
reconstruct.

### 4.4 Export target B — geqdsk + profiles bundle

New `io` helpers (mirroring the IMAS ones):

- `export_geqdsk_drawset(header, out_dir, scan_key=, selection=)` — write the
  stored g-file per draw (`{header}_draw{idx}.geqdsk`). (The bytes already
  exist; this is a thin, exact dump — partially covered today by
  `load_equilibrium(eqdsk_out_dir=)`, to be unified.)
- profiles companion per draw, format-selectable:
  - **p-file** (`PFile`) — kinetics (ne/Te/ni/Ti/…),
  - **i-file** (`mygs.save_ifile`, captured at generate time) — inverse-equil
    profile file some transport/stability codes prefer, and/or
  - **profiles JSON** — psi_N + all current components + kinetics in bouquet's
    own schema, the least lossy option.

Bundle writer `export_draw_bundle(header, out_dir, formats=("geqdsk","pfile"))`
gives one call for "g-file + profiles for every selected draw."

### 4.5 API surface

- `GenerationConfig.capture_live_eq: bool = True` — capture the FSA payload
  (cheap; default on so exports are exact by default).
- `GenerationConfig.capture_npsi: int = 257` — radial resolution of the
  captured FSA block (matches the 257² eqdsk). **Do not set below 129** — the
  edge bootstrap / pedestal current needs it; coarser aliases the very feature
  the exact conversion is meant to get right.
- `GenerationConfig.eager_export: tuple = ()` — e.g. `("ids","geqdsk")` to also
  write files during generation (default off).
- `Bouquet.export_ids(out_dir, selection="selected", fidelity="auto")`,
  `Bouquet.export_geqdsks(out_dir, ...)`,
  `Bouquet.export_bundle(out_dir, formats=(...))` — thin run-object wrappers
  over the `io` drawset functions (consistent with the de-threaded reader API).
- New `io` functions: `export_geqdsk_drawset`, `export_draw_bundle`,
  `write_profiles_json`; `write_imas_draw`/`export_imas_drawset` gain
  `fidelity=`.

### 4.6 Schema / compatibility

- **Additive-optional**: the `eq_fsa/` group and any `ifile`/`profiles_json`
  datasets are new optional members. No `schema_version` bump required (v2
  readers skip unknown members); bump to v2.1 only if we want provenance to
  advertise the capability.
- Backward compatible both ways: new code reads old archives (falls back to
  reconstruct + warns); old readers ignore `eq_fsa/`.
- One cleanup to fold in: `write_imas_draw` still uses the inline
  `["eqdsk"] if "eqdsk" in g` lookup — switch to `schema.find_bytes_dataset`
  (the helper the docs-refresh consolidation introduced).

## 5. Phasing

- **Phase 1 — exact IDS (highest value). ✅ DONE.** Captured `eq_fsa/` in
  `generate_bouquet`/`store_equilibrium` (incl. exact `<1/R^2>` by
  quadrature, forward-compatible with native `sauter_fc` per #312); added the
  exact `fidelity=` conversion to `write_imas_draw`/`export_imas_drawset`.
  Live-verified on the OMAS example (exact vs baseline-ratio bootstrap differs
  2–10%/draw). Conversion checked vs Wesson + IMAS + FSA-quadrature benchmark.
- **Phase 2 — geqdsk+profiles bundle. ✅ DONE.** `DrawView.extract` /
  `ScanView.extract` write per-draw geqdsk / pfile / profiles JSON
  (`profiles_doc`: profiles + units + scalars + coils + eq_fsa, source-agnostic);
  `Bouquet.export_bundle` + `Bouquet.export_ids` run-object wrappers.
  Deferred: `save_ifile` (no confirmed consumer -- see §8), `eager_export`.
- **Phase 3 — native OFT ODS export (optional, upstream).** If/when OFT grows a
  direct ODS/IDS export from the equilibrium object, swap the Phase-1 assembler
  for it behind the same `fidelity="exact"` API. No user-facing change.

## 6. Scope boundaries

**In:** `equilibrium` + `core_profiles` IDS at exact fidelity; geqdsk +
kinetics/current profiles bundle; per-draw capture; on-demand + optional eager
export; selection-aware (selected/all).

**Out (v1):** `core_sources` beam/RF detail beyond the existing `jphi_nbi`/`jphi_rf`
carry-over; `pressure_fast` anisotropy round-trip; MSE / diagnostic synthetics;
multi-time-slice IDS assembly (one IDS per draw per slice, as today); writing
back into a live IMAS database (JSON/OMAS files only).

## 7. Validation plan

- **FSA sanity:** ⟨R⟩, ⟨1/R⟩, ⟨|B|²⟩ from `get_q`/`sauter_fc` vs analytic on a
  circular large-aspect-ratio equilibrium (known closed forms).
- **Conversion round-trip:** toroidal → parallel → toroidal through
  `parallel_to_toroidal` with the captured `geom` returns the input to ~1e-10.
- **Exact vs interim:** on the D3D-like OMAS fixture, quantify the
  baseline-ratio error the exact path removes (expect largest near the edge);
  add a `solver`-marked test asserting `j_total = j_ohmic + j_bootstrap` and
  `j_tor` exactness in the written IDS.
- **Bundle round-trip:** re-read exported geqdsk + profiles JSON → same arrays
  as the archive (bit/‰-level).
- **Golden:** extend the fixture with an `eq_fsa/` block on ≥1 draw so the exact
  path is covered by the fast (non-solver) suite.

## 8. Risks / open questions

- **Per-draw capture cost:** MEASURED on the OMAS example — ~**2.7 s/draw** at
  npsi = 257 (`get_profiles` + `sauter_fc` + `get_q(compute_geo=True)` plus the
  exact `<1/R^2>` flux-surface quadrature over 257 traced contours). Negligible
  next to the multi-minute GS solve, so both `capture_live_eq` and the exact
  `<1/R^2>` default on at full resolution. `inv_R2_npsi` can trace `<1/R^2>`
  coarsely + spline if ever needed on very large bouquets (not the default).
- **COCOS:** the archived eqdsk is COCOS-tagged; the FSA metrics come straight
  from OFT's internal convention. Pin the sign/2π conventions where the FSA
  block meets `parallel_to_toroidal` and the IDS (IMAS is COCOS 11/17); add an
  explicit conversion test.
- **OFT version floor:** `sauter_fc` / `get_field_eval` / `get_q(compute_geo=True)`
  must exist in the pinned OFT build — record a minimum OFT commit in
  `docs/CI.md` and guard with a clear error.
- **Native ⟨1/R²⟩ (READY):** [OpenFUSIONToolkit#312](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit/issues/312)
  / hansenc's branch adds ⟨1/R²⟩ (and ⟨B_φ²⟩) to `sauter_fc`. `capture_equilibrium_fsa`
  already **auto-detects and uses** them (`physics._native_fsa_inv_R2`),
  validated by Jensen + the exact `⟨B_φ²⟩=F²⟨1/R²⟩` identity, and skips the
  trace quadrature when present — so the branch is a transparent speed-up with
  no code change. A build without it (or a layout mismatch) safely falls back
  to the quadrature. Confirm/simplify the array indices against the merged
  signature.
- **Deprecation:** `flux_integral` is deprecated in favour of
  `compute_flux_integral` — use the latter.
- **i-file necessity:** confirm any consumer actually wants `save_ifile`
  output; if not, drop it from Phase 2 and ship p-file + profiles JSON only.

## 9. Rough effort

- Phase 1: ~1–2 focused days (capture + exact conversion + tests). Highest ROI.
- Phase 2: ~1–2 days (bundle writers + wrappers + tests).
- Phase 3: gated on OFT; small adapter once available.

## 10. Files likely touched

- `bouquet/TokaMaker_interface.py` — capture `eq_fsa` after each converged
  solve (near the existing `save_eqdsk`/`store_equilibrium` call).
- `bouquet/utils.py` — `store_equilibrium` writes the `eq_fsa/` group; loader.
- `bouquet/io/imas.py` — exact `fidelity` path; `schema.find_bytes_dataset`.
- `bouquet/io/__init__.py`, `bouquet/io/pfile.py` — profiles JSON / p-file export.
- `bouquet/physics.py` — inverse (toroidal→parallel) helper if not already covered.
- `bouquet/run.py` — `Bouquet.export_ids/export_geqdsks/export_bundle`;
  `GenerationConfig` knobs in `bouquet/config.py`.
- `bouquet/schema.py` — `eq_fsa` dataset names/units.
- `tests/` — FSA sanity, conversion round-trip, exact-vs-interim (solver),
  bundle round-trip; golden `eq_fsa` block.
- `docs/archive-schema.md` — document the `eq_fsa/` group.
