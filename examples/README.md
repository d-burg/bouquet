# Bouquet examples

`bouquet` generates a *bouquet* — a family of perturbed Grad–Shafranov
equilibria — from a single baseline, so you can propagate kinetic/current
uncertainty into the equilibrium reconstruction. These examples get you running
on fully synthetic, non-proprietary DIII-D-like data.

## Prerequisites

The package needs the **OpenFUSION Toolkit** (TokaMaker) importable — either
call `bq.add_oft_to_path()` (resolves `OFT_PYTHONPATH`, known install
locations, or a sibling checkout) or set the path yourself:

```bash
export OFT_PYTHONPATH=/path/to/OpenFUSIONToolkit/build_release/python
```

`import bouquet` itself works without OFT (all TokaMaker imports are lazy); a
missing path surfaces later, at `setup_solver()` / `reconstruct()`, as
`ModuleNotFoundError: OpenFUSIONToolkit`. Set the path above before running
the solver sections.

## Where to start

Both notebooks are **layered**: a minimal "set inputs, press run" example at the
top, then sections that each add one axis of complexity (inputs → filtering →
switchboard extras → … → tuning). Run section 1, then stop wherever you have
what you need. The two share a skeleton — the only real difference is the
**baseline source**, so learning one teaches the other.

| Notebook | Baseline source | What it shows | Full run |
|---|---|---|---|
| [`D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb`](D3D-like/bouquet_D3Dlike_geqdsk_example.ipynb) | EFIT **g-file** + Osborne **p-file** (reconstructed) | reconstruction fidelity report, perturbed LCFS/profile overlays, manual switchboard extras | ~7–10 min |
| [`D3D-like/bouquet_D3Dlike_omas_example.ipynb`](D3D-like/bouquet_D3Dlike_omas_example.ipynb) | FUSE **IMAS/OMAS** IDS (pre-separated) | pre-separated currents, switchboard extras from the IDS, 3-slice L→H time evolution | ~10 min/slice |
| [`D3D-like/bouquet_D3Dlike_parallel_IMAS_example.ipynb`](D3D-like/bouquet_D3Dlike_parallel_IMAS_example.ipynb) | same OMAS IDS, **process-parallel** | `parallel_generate` (laptop pool) + SLURM job-array emission, shard merge + baseline guard | scales with cores |
| [`D3D-like/bouquet_D3Dlike_systematics.ipynb`](D3D-like/bouquet_D3Dlike_systematics.ipynb) | committed golden fixture | bias/systematics decomposition: jphi_total pinned σ=0 / pinned IDA-σ / full production | solver-marked |

**New to bouquet?** Start with the **geqdsk** notebook — it's fully
self-contained (reconstructs from the committed fixtures) and the reconstruction
fidelity summary is a good first confidence check.

The MVP cell is nearly identical across the two:

```python
# geqdsk
run = bq.Bouquet.from_geqdsk('…baseline.geqdsk', profiles='…baseline.peqdsk',
                             mesh='DIIID_mesh.h5', n_draws=5, header='…')
run.reconstruct()                 # reconstruct + fidelity summary
run.generate(); run.filter(); run.export()

# IMAS
run = bq.Bouquet.from_imas('…omas.json', mesh='DIIID_mesh.h5', time=2.3043,
                           n_draws=5, header='…')
run.run()                         # setup → baseline → generate → filter → export
```

## In-spec counts

A draw is **in spec** when it passes the magnetics: the coil χ² test and the
boundary-deviation cut (`run.filter()` prints the counts and names the rule;
see [the in-spec rule](../docs/workflows.md#the-in-spec-rule)). Since
2026-10-14 (owner decision) the l_i band is **not** part of that default; it
and a β_p band are opt-in for a source that states their uncertainty
(`generation.l_i_sigma`, `generation.beta_p_target` / `beta_p_sigma`, or an
explicit `generation.l_i_tolerance`). Rendered outputs saved before that date
count in-spec draws under the old unconditional ±5 % l_i band
(`l_i_tolerance=0.05`), so a fresh run reports a higher yield and wider l_i
bands; `bq.refilter_inspec(path)` (or `run.filter()`) re-judges an old archive
under the current rule. β_p and Λ = β_p + l_i(3)/2 are reported per draw
(`bq.draw_scalars`), never filtered on.

## Regenerating vs. loading

Each notebook has a `REGENERATE` flag. The HDF5 outputs (`*.h5`) are gitignored,
so a fresh checkout regenerates them:

- **geqdsk** is self-contained — `REGENERATE=True` reconstructs and solves from
  the committed g-file/p-file.
- **IMAS** single slices regenerate from the committed OMAS json; the 3-slice
  **time series** is produced by `D3D-like/_run_omas_timeseries.py` (one
  subprocess per slice, because `OFT_env` is a per-process singleton) — or,
  within one process, via `run.run_slices(times=[...], scan_keys=[...])`.

Rendered outputs are saved in the committed notebooks, so you can preview results
on GitHub without running anything.

## Fixtures (all synthetic, non-proprietary)

- `D3Dlike_Hmode_baseline.geqdsk` / `.peqdsk` — H-mode equilibrium + kinetic profiles
- `D3Dlike_baseline_omas.json` — synthetic FUSE OMAS data dictionary (3-slice L→H)
- `DIIID_mesh.h5` — TokaMaker mesh

Real machine data (`*.cdf`, proprietary g-files, FUSE `dd_sim.json`) is
gitignored and never committed.
