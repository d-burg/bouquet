# Draw methods: legacy, unified engine and swb

bouquet can make a bouquet's draws in three ways, selected by `GenerationConfig.solve_method`:

| `solve_method` | One draw is | Code |
|---|---|---|
| `"legacy"` | `perturb_kinetic_equilibrium`: GPR inductive + SWB/Redl bootstrap, l_i loop or Fix C, homotopy | `bouquet.draw_methods.DrawMethod` |
| `"engine"` (default) | the unified engine's draw: x* held, the Ip row as an inductive amplitude, one bootstrap loop from the reconstruction state | `bouquet.engine_draws.GenerateEngineDraws` |
| `"swb"` | one `solve_with_bootstrap`: the baseline's solve B with resampled kinetics and inductive seed | `bouquet.swb_draws.SwbDraws` (draws), `bouquet.swb` (baseline) |

## One strategy object

`generate_bouquet` runs the same draw loop for all three methods: sampling, archiving, filters and until-N are shared. Where the methods differ, it calls one **draw method** object, `generate_bouquet(draw_method=...)`, built by `Bouquet._draw_method` from `solve_method`. There are no per-method `if` branches in the loop. Each method is a subclass of `DrawMethod` and overrides the hooks it needs. `DrawMethod` itself is the legacy behaviour.

### Why a strategy object
- Each method's code lives in its own module. None of swb's code sits in the legacy draw loop.
- A new method is a new subclass, not a fourth set of branches through a 3300-line function.

### The legacy path
The legacy draws are `DrawMethod`'s own hook bodies: the legacy code, moved, not changed. `tests/test_legacy_golden.py` and the legacy behavioural tests pin it.

## The hooks

Stage order follows one draw through `generate_bouquet`.

| Hook (kind) | Stage | Legacy | Engine | swb |
|---|---|---|---|---|
| `validate` (noop) | setup | — | refuses PIN_JPHI, DIFF_BS, delta mode, l_i σ, loop off; prints the banner | — |
| `jphi_baseline` (pass) | setup | as configured | as configured | `False`: solve B is the baseline |
| `solve_pressure`, `edge` (pass) | setup | the legacy assembly | the contract's pressure and edge settings | legacy |
| `strong_coil_reg` (flag) / `announce_coil_reg` | setup | strong reg toward the baseline coils | same | none; the per-draw recipe installs its own reg |
| `pin_bounds_on_skip` (flag) | per draw | SKIP_HOMOTOPY pin | same | no |
| `draw` (legacy) | per draw | `perturb_kinetic_equilibrium(...)` | `GenerateEngineDraws.draw` | `swb_draw` |
| `rejection_reason`, `annotate_rejection` | per draw | `_draw_rejection_reason` | adds the engine codes (caps, closure refused, non-finite) | legacy |
| `iso_update` (flag) | after Ip align | isoflux update | same | no |
| `drift_without_homotopy` (none) | coils | None: homotopy | `engine_draw_homotopy=False`: measured drift | always measured drift (no bounds) |
| `homotopy` (flag), `cap_solver`/`uncap_solver`, `hit_cap`, `announce_cap`, `announce_rollback_failed` | homotopy | homotopy, no cap | homotopy under `engine_draw_solve_maxits`, cap events recorded | no homotopy |
| `post_homotopy_jbs`, `post_homotopy_split` (legacy) | post-homotopy | `_post_homotopy_jbs`, `_decompose_draw_currents` | the engine draw's own passes and split | not reached |
| `post_hoc` (pass) | acceptance | coil verdict | AND the l_i / sawtooth band | legacy |
| `lcfs_pressure`, `stored_pressures`, `archived_split` (pass) | archive | legacy assembly | the engine's own pressures and split on the archived state | legacy; `swb_draw` records its own p_sep |
| `store_draw` (noop) | archive | — | the engine block | `swb_alpha` and the saw attrs |
| `until_n` (pass) | until-N | coil + boundary | AND the band | legacy |
| `draw_env`, `scale_settings`, `loop_settings_for`, `solve_maxits`, `achieved_jphi`, `draw_jbs_loop`, `loop_codes` | `Bouquet.generate` | as configured | engine loop settings, its scale range, its codes | no scale range, cap ≥ 100, SWB's own split, no Redl loop |
| `store_baseline`, `summarize` (noop) | after the loop | — | the engine record, cap events | the solve A / B and saw stamp |
| `verify_sigma0`, `workflow_problems` (class-level, none) | checks | the legacy σ=0 check and route rules | `_verify_sigma0_engine`; one route for both inputs | one σ=0 draw through `SwbDraws.draw`, equal to solve B to the bit; `swb_config_problems` |

## Where the engine leaves the legacy path

The main differences are:
- **Per draw:** the engine draws with x* held and the Ip row as an inductive amplitude. This replaces the GPR candidate, the l_i loop and Fix C (`draw`).
- **Bootstrap:** the engine runs one self-consistent bootstrap loop from the reconstruction state, then its own post-homotopy passes (`post_homotopy_jbs`).
- **Acceptance:** the engine adds a post-hoc l_i / sawtooth band to acceptance and until-N (`post_hoc`, `until_n`).
- **Archive:** the engine archives its own pressure assembly and its split on the archived equilibrium, with no clipping (`stored_pressures`, `archived_split`).
- **Solver:** every homotopy solve runs under a GS iteration cap, with every capped solve recorded (cap hooks).

The swb draw differs more. It has no anchor route, no bouquet Ip closure, no strong reg, no isoflux update and no homotopy: one SWB solve per draw, with the coil drift measured rather than bounded.

## Adding a method
1. Subclass `DrawMethod` and override only the hooks that differ.
2. Build it in `Bouquet._draw_method`, and `draw_methods.method_hooks` for the class-level hooks.
3. A hook that needs a new kind of legacy behaviour goes into `HOOKS` / `FLAGS`, so the legacy-path test can write it back.
