# bouquet documentation

Start at the [top-level README](../README.md) for what bouquet is, how to
install it, and a working quickstart. These pages carry the depth.

## User guides

| Page | Contents |
|---|---|
| [workflows.md](workflows.md) | The pipeline stage by stage, what is perturbed vs. held fixed, the **full configuration reference**, workflow presets, archive reading, export, timeseries sweeps, process-parallel generation |
| [engine.md](engine.md) | The unified reconstruction engine (`reconstruction_engine="unified"`, the default since 2026-10-06; `"legacy"` opt-in): the adapter contract, the composition, the rows, convergence, delivery, presets, draws |
| [physics-notes.md](physics-notes.md) | What the ensemble is (and isn't), the σ=0 consistency guard, bootstrap treatment and `jbs_delta_mode`, PCHIP kinetics regridding, edge classification, IDA-hybrid kinetics, Z_eff-primary densities, corrective j_phi |
| [coil-constraints.md](coil-constraints.md) | Coil classes, the VSC channel drift metric, progressive homotopy, the `in_spec` criterion and per-draw attributes |
| [io-and-plotting.md](io-and-plotting.md) | GEQDSK / p-file / IDA / IMAS readers and writers, COCOS conversion, the plotting catalogue |
| [api-reference.md](api-reference.md) | Every public name, grouped by role |
| [archive-schema.md](archive-schema.md) | The HDF5 archive layout (schema v2) |
| [gui-display-guide.md](gui-display-guide.md) | In-notebook plotting vs. the `plot-family` CLI |

## Reference and process

| Page | Contents |
|---|---|
| [../architecture.md](../architecture.md) | Physics assumptions, numerical approximations, coordinate/sign conventions, tolerance budgets, known limitations |
| [flowchart/](flowchart/) | The rendered physics workflow and the 550-node interactive logic map ([view](https://d-burg.github.io/bouquet/flowchart/)) |
| [CI.md](CI.md) | Fast vs. solver test tiers, the manual pre-merge gate, branch protection |
| [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) | Historical change summaries by development round |
| [validation-provenance.md](validation-provenance.md) | What the unified-engine branch was validated on, at which code revision, and the aggregate results |
| [proposals/](proposals/) | Design proposals |

## Examples

Runnable notebooks on synthetic, non-proprietary D3D-like fixtures are in
[`../examples/`](../examples/README.md).
