# The unified reconstruction engine (`reconstruction_engine="unified"`)

One reconstruction loop for both input types -- a g-file (+ p-file / IDA
profiles) and a modelling source (IMAS / OMAS IDS) -- behind
`GenerationConfig.reconstruction_engine`. **Default `"unified"` since
2026-10-06** (owner decision; it was `"legacy"`): this page describes what
`Bouquet.prepare_baseline()`, `generate()` and `verify_sigma0_consistency()`
run unless a configuration says `reconstruction_engine="legacy"` -- the
existing g-file reconstruction / IMAS baseline paths and the legacy draws,
still available by name (`Bouquet.from_geqdsk` / `from_imas(...,
reconstruction_engine="legacy")`) but **deprecated in 1.4.0 and removed in
1.5 (#90)**: a run that resolves to the legacy route (`reconstruction_engine=
"legacy"` or `solve_method="legacy"`) emits one `DeprecationWarning` where it
is resolved (`prepare_baseline()`, the draw method). A stored configuration that predates the
field (written before 2026-09-29) loads as `"legacy"`, with a warning. The
legacy-path settings the engine never reads are refused under `"unified"`
(see [Settings the engine does not read](#settings-the-engine-does-not-read-refused)), and the refusal
says how to get the legacy paths back. Runtime on the shipped synthetic
cases: engine reconstructions 38–76 s and draws 60–144 s, against 150–660 s
and 405–1407 s on the legacy path with the bootstrap loop on.

**Solve methods.** `GenerationConfig.solve_method` is the one switch:
`"legacy"`, `"swb"` (OFT `solve_with_bootstrap` is the baseline and every
draw; IMAS only) or `"engine"` (this page). `imas_baseline` /
`reconstruction_engine` remain as its older spellings: with
`solve_method=None` the method is derived from them. Nothing is rewritten at
construction; `Bouquet.prepare_baseline()` (like the engine-dependent
defaults) sets both fields to the method's effective values and remembers
the user's own, so switching back (e.g. `imas_baseline` from `"swb"` to
`"closure"`) restores the engine. A contradicting pair is refused:
`imas_baseline="swb"` with another `solve_method`, and
`reconstruction_engine="legacy"` with `solve_method="engine"`. `"swb"` and
`"engine"` share the kinetic sampler (`bouquet.kinetic_sampler`), the P' edge
pin and the separatrix-pressure offset (swb refuses `edge_pprime_pin=False`
and `separatrix_pressure="legacy"`, which OFT's SWB cannot honour). The SWB
edge taper is opt-in for both: swb's `swb_edge_taper_psi0` (default `None`,
off; sent as `taper_edge_jphi=False` to a toolkit that has the option, since
its own default may be on), the engine's `bootstrap_kwargs`. swb-only
settings (`swb_*`, including the Ip acceptance `swb_ip_tol`, default 5e-3,
with a warning for any solve accepted above 1e-4) never reach the engine.
swb's bootstrap is the toolkit's SWB output converted to the field-aligned
`kappa <j.B>` for the installed toolkit's own output convention
(`physics._swb_jbs_to_toroidal`), or taken as returned from a toolkit that
reports `p'G` apart (`jphi_pd`, OFT PSFC_dev); the pressure-driven `p'G` is
the third bucket `jphi_pd`, archived as such. Known asymmetry: the SWB sawtooth
reset (`swb_saw_*`) has no engine counterpart.
The three methods share one draw loop and differ only through a draw-method
object (`bouquet.draw_methods`); docs/draw-methods.md has the hook-by-hook
comparison and how the legacy path is kept bit for bit.

**Status (Stage 3).** The engine builds the baseline (`Bouquet.prepare_baseline()`
returns the same `Baseline` the rest of the package consumes, plus
`Baseline.engine`, the full record) and runs the draws: `generate()` and
`verify_sigma0_consistency()` run on the engine when it built the baseline in
the same session ([Draws](#draws) below). A mismatched pair -- `"unified"`
with a baseline the engine did not build, or `"legacy"` with an engine
baseline -- is refused: the legacy draw routes keep the pressure-driven term
frozen in the inductive (and close the current differently), so they would
not reproduce an engine reconstruction at zero perturbation (and vice versa).
(Since 2026-10-06 both use the same `<j.B>` -> `<jphi_total>` conversion,
`physics.field_aligned_conversion`.)

Code: `bouquet/engine.py` (the engine, the TokaMaker backend, the wiring),
`bouquet/engine_draws.py` (the draws), `bouquet/adapters.py` (the source
adapters), the kernel `bouquet.jbs_loop.run_jbs_loop` (unchanged except an
opt-in hook for added criteria). Tests: `tests/test_engine.py` (a toy
Grad-Shafranov stand-in), `tests/test_engine_adapters.py`,
`tests/test_engine_wiring.py`, `tests/test_engine_draws.py` (the draws, on
the toy and a TokaMaker stand-in over it), `tests/test_one_conversion.py`
(the one `<j.B>` -> `<jphi_total>` conversion), `tests/test_engine_solver.py`
(`-m solver`), probe `tests/probes/measure_engine.py`.

## The picture

```
 g-file adapter                         IDS adapter
 <j.B>_in - Redl(anchor), smoothed      |B0| (j_tot - jphi_bs - driven), driven fixed
 rows: Ip, l_i(3) hard, (q0)            rows: Ip, li_3 soft, (q0), (MSE)
            \                               /
             v                             v
   +---------------------------------------------------------------+
   | compose on G_k (the latest SOLVED geometry):                  |
   |   J = F<1/R>/<B^2> [s_ind <j.B>_ind + s_bs <j.B>_BS + <j.B>_fix]|
   |       + p'(<R> - F^2<1/R>/<B^2>)                  (identity I2)|
   +---------------------------------------------------------------+
             |
             v
   closure (0 solves): rows + discrepancies, prior / preset
             |
             v
   relax (beta) -> ONE GS solve ----------------------+
             |                                        |
             v                                        |
   measure on the solved state: Redl, l_i, Ip and c,  |  pass
   q0, tan(gamma), request - achieved                 |
             |                                        |
             v                                        |
   update lambda_bs (omega), discrepancies, MSE J ----+
   criteria on 2 consecutive passes
             |
             v
   delivery solve (two passes) + checks on every row
   = THE reconstruction (Baseline, metrics, draw reference)
```

## The contract (`adapters.EngineContract`)

| item | g-file adapter | IDS adapter |
|---|---|---|
| kinetics n_e, T_e, n_i, T_i, Z_eff | p-file / IDA, PCHIP onto the g-file ψ_N (as the legacy reconstruction) | core_profiles (or the IDA hybrid), as `read_imas_baseline` resolves them |
| pressure | thermal + impurity + fast (fixed) | thermal + impurity + fast, **no `p_diff`** |
| inductive (parallel) | `<j.B>_in = F p' + F'<B^2>/mu0` on the g-file's own traced surfaces (identity I0), minus Redl `<j.B>` on the anchor, minus the fixed parts; smoothed with a verbatim copy of `fit_inductive_profile`'s basis (spline + PCHIP, zero edge anchor, ≥ 0), **no amplitude search** | **the parallel residual by definition** (owner decision, 2026-10-02): `|B0| (j_total − j_bootstrap − Σ driven)` (IMAS `<j.B>/B0`), every driven entry the one held fixed below. The source's `j_ohmic` is a **cross-check**, not a choice: it is compared with the residual and the net (fraction of the total current) and rms (fraction of rms `j_total`) differences are stamped in `provenance["inductive_consistency"]` (action `"residual_by_definition"`; `"unchecked"` when the source has no `j_ohmic`), with no threshold and no warning. A source without `j_total` or `j_bootstrap` is refused (`inductive="j_ohmic"` uses its `j_ohmic` explicitly; there is no silent fallback). Evidence: on self-consistent sources the residual and `j_ohmic` are indistinguishable (`|Δl_i| ≤ 1.8e-3`); on sources whose own split is locally inconsistent the residual is closer to the source's `<j_phi>`. Explicit options (`IdsAdapter(inductive=...)`, passed by the engine from `GenerationConfig.engine_ids_inductive`, default `"residual"`): `"j_ohmic"` takes the source's, warning when the net mismatch exceeds `IdsAdapter(inductive_tol=0.02)` (2 % of the total current, owner-set); `"auto"` takes `j_ohmic` unless it is absent or misses by more than that tolerance, then the residual with a warning. Nothing in the source is altered |
| fixed driven (parallel) | user `jphi_nbi` / `jphi_rf` (toroidal inputs), converted with the anchor's `F<1/R>/<B^2>` | the **driven** core_sources entries by their IMAS identifier (the data dictionary's `core_sources.source[:].identifier` enumeration; `adapters.IDS_DRIVEN_SOURCE_PARTS`), × `|B0|`, held fixed, by part: nbi (2) → `nbi`; ec / lh / ic (3, 4, 5) → `rf`; fusion (6), runaways (501), a model's sawteeth entry (701) → `other`. **Never added**: ohmic (7) and bootstrap (13) (the core_profiles `j_ohmic` / `j_bootstrap` stand for them), AGGREGATES -- total (1), auxiliary (100), the combinations 101-107, radiation (200), 202, 203 -- which would double-count their constituents, and a bootstrap-like `neoclassical` (401); each such entry carrying a non-zero `j_parallel` is listed in `provenance["ignored_sources"]` with its reason and warned about. An unknown index carrying a non-zero `j_parallel` is held fixed under `other` WITH a warning (`unclassified` in its `driven_sources` entry). The core_sources slice is the one nearest the core_profiles slice read and must lie within half the local core_profiles time-step of it, else the read is refused naming both times (`io.imas.core_sources_slice`, shared with the legacy reader; owner decision 2026-10-06). An entry carrying its own per-slice `time` is read at the core_sources slice time (a model's entry may start later than the IDS time base), not at the list index: its nearest own slice, accepted within half its own local time-step AND within half the local core_profiles step of the core_profiles slice time -- never interpolated; when neither time base has a local step (single-time bases, for the core_sources slice and for an entry alike) the window is 10 us (`io.imas.IMAS_SINGLE_TIME_WINDOW_S`, owner-approved 2026-10-07: a rounding-level mismatch of millisecond-stored times is a match, its dt recorded, and the rules below apply only beyond it); outside that window a driven entry carrying current on its own slices bracketing the time is refused, one carrying none there is off at that time (zero, listed in `provenance["off_sources"]`), and one whose own record begins AFTER the slice time is off before its record (zero, listed with `reason="off_before_record"` and its `first_own_time`, announced once per source file and entry); past its last own time an entry carrying current there is refused; one with a different slice count and no times is refused. Every match (both slice times, dt, the windows, the bracketing own times, the status) is in `provenance["source_time_match"]`. `provenance["driven_sources"]` lists what was added; a user override is converted as on the left. The delivered split reports `other` inside `j_RF`. The legacy reader (`read_imas_baseline`, which the engine runs first) reads every driven entry through this same call with the same arguments, so its refusals, `off_sources` stamps and announcements are the engine's; it records `driven_sources` / `ignored_sources` / `off_sources` / `sawteeth_hold` in `Baseline.source_time_match`. `ImasSource.hold_sawteeth=False` (legacy reader only: the sawteeth entry left in the residual inductive) is refused here |
| boundary | g-file LCFS | equilibrium boundary outline (or `LCFS_geqdsk`) -- both held as the standard per-unit-length isoflux set (#84, below) |
| rows | Ip (exact); l_i(3) = the reader's `li(2)` key, **hard**, absolute tolerance 1e-3; q0 (optional) | Ip (soft, σ = 0.5 % of Ip); li_3 (soft, σ = 0.04); q0 (optional); MSE chords (optional) |
| signs | positive frame: `sign(Ip)`, `|F|`; a file whose `<jphi_total>` disagrees in sign with its Ip is refused (wrong COCOS) | `source_current_sign`, `|B0|`; `b0_sign` recorded |

**The IDS li_3 target's radius (`GenerationConfig.imas_li3_radius`, default
`"auto"`, 2026-10-06).** The data dictionary gives `li_3` no normalisation
radius: IMAS.jl (and so FUSE) writes it with the boundary's geometric radius
`R_geo = (R_out + R_in)/2`, the measurement (`utils.li_achieved`) normalises
by the magnetic axis -- a few per cent apart on a shifted axis.
`adapters.resolve_li3_radius`: `"auto"` recomputes `li_3 = 2 ∫B_p² dV /
((μ0 Ip)² R)` from the source's own flux-surface averages (`profiles_1d`
`gm2`, `dpsi_drho_tor`, `dvolume_dpsi`; COCOS 11) with each radius and takes
the one reproducing the stored value within `adapters.LI3_RADIUS_MATCH_TOL`
(0.5 %); neither is REFUSED, naming both (never rescaled silently); a source
without those averages (the shipped example) keeps the stored value
unrescaled, as before the setting, printed and recorded `"undetermined"`.
`"geometric"` / `"axis"` state it. The row's target is the stored li_3 times
`R_src / R_axis` (the source's axis); the choice, both ratios and the factor
are recorded on the row (`radius_*`), in the contract's provenance
(`li3_radius`) and in `Baseline.li_metrics["li3_radius"]`. A non-default value
is refused with a g-file source; a stored IMAS unified config without the
field loads as `"axis"`, what it ran with. On the shipped example the
TokaMaker and IDS li_3 agree within 0.1 %; no bias is assumed for a file --
it is measured.

The IDS σ values are the `li_soft_onesided` preset's
(`utils.STRUCTURED_PRESETS`); the g-file l_i tolerance is the legacy step-5 /
re-match secant's (`_rematch_li_request`'s `li_tol` default, read from its
signature). The q0 target is the source's own q **at the measurement radius**
(the ψ_pad-clipped axis sample the closure's axis row uses), admitted by
`utils.q0_gate_admits` (sawtooth model active, or `|q0_source| ≤ q0_gate`).
MSE rows accept **E_r-corrected** pitch angles only (`er_corrected=True`, no
`Er`); a raw-E_r modelling request is refused.

Pressures use `physics.ELEMENTARY_CHARGE`. The anchor E_0 is one (two-pass) solve of the source's own total current as
the legacy path solves it (g-file: `|j_tor_averaged_direct|`; IDS: `j_tor`);
it seeds the geometry and the first Redl bootstrap. **For a g-file the anchor
also defines the fixed inductive, once:** a g-file carries only the total
current, so the adapter (`GFileAdapter.finalize`) forms
`jB_ind = smooth(<j.B>_in − Redl(E_0) − <j.B>_fix)` (the `fit_inductive_profile`
basis, no amplitude search) from the Redl bootstrap ON THE ANCHOR, and that
`jB_ind` is held fixed for the whole reconstruction (the closure then scales
it, `s_ind`). Redl is re-evaluated on every pass's solved equilibrium
thereafter; only the inductive's SHAPE remembers the anchor. So the g-file
fixed point does depend on the anchor, through that one subtraction: it is
the self-consistent state for the inductive the source's total current
implies at the anchor's bootstrap, not for an inductive re-derived from the
converged bootstrap (which would make the inductive a moving target of the
loop it is meant to anchor). For an IDS source the inductive is the source's
own (`j_total − j_bootstrap − Σ driven`, read once); beyond the seed, the
anchor there only converts user-supplied toroidal fixed parts with its
`F<1/R>/<B^2>` (as it does for a g-file's).

## Composition (identity I2)

Components are stored as parallel currents `<j.B>` [T A/m²]. The solver's
variable -- the plain flux-surface average `<jphi_total>` TokaMaker's
`jphi-linterp` consumes -- is formed in one place, `engine.compose`, on the
geometry of the latest solve:

```
<jphi_total> = <j.B> F<1/R>/<B^2>  +  p' (<R> - F^2 <1/R>/<B^2>)
```

The first term is the field-aligned conversion (c) of the verification report
(the legacy `evaluate_jbs` output uses (a), `<j.B>/(F<1/R>)`, about 6 % high
at the bootstrap peak; the engine takes the Redl `<j.B>` from the evaluator's
diagnostics and converts it itself). The second term is the pressure-driven
toroidal current (diamagnetic + Pfirsch–Schlüter, zero `<j.B>`), recomputed
every pass from that equilibrium's own `p'` and geometry (13–22 % of `<jphi_total>`
at ψ_N 0.9–0.97 on the synthetic examples). `F`, `<B^2>` come from the
evaluator's surfaces (`sauter_fc`), `<R>`, `<1/R>`, `<1/R^2>`, `V'`, `p'` from
`utils.fsa_current_geometry`. The Redl `<j.B>` receives the shared
innermost-surface repair (`smooth_jbs_transition`) every SWB-derived profile
receives.

**The archived split** puts the pressure-driven term
`p'(<R> - F^2<1/R>/<B^2>)` on `jphi_bs`, as IMAS `j_bootstrap`, the IMAS reader
and `evaluate_jbs` do; `jphi_ind` is the residual.

**Toroidal-flux runs (`coord="phi_n"`).** The contract's grid is the run grid
(Φ_N). The backend tags every solve with it and samples each measurement's
geometry at the nodes' ψ_N on that solve's own toroidal-flux map, so
`geom["psi_N"]` is ψ_N and every integral, interpolation and residual uses it.
The structured basis stays on the run grid (`close_ip_structured(...,
basis_x=)`). In a ψ_N run all of this is the identity, bit for bit. See
docs/workflows.md for the source side.

**Edge taper (off by default).** With `bootstrap_kwargs={"taper_edge_jphi":
True}`, as `solve_with_bootstrap(taper_edge_jphi=True)` does for the swb
method, every term above is multiplied by OFT's edge taper
(`physics.edge_taper_weight`, a port of `apply_edge_taper`): 1 below
`taper_edge_psi0` (default 0.999), falling to 0 at the LCFS (`taper_edge_shape`
2, quintic smoothstep).  The backend puts the factor on every geometry it
measures, so the closure rows, the draws and the archived split all see the
tapered composition.  Off, no weight is built and `composed_factor` is
`conversion_factor`.

## A pass

1. Compose on `G_k` with the bootstrap iterate `lambda_bs,k`.
2. Closure, zero solves: `utils.close_ip_structured` (hard rows: the g-file)
   or `close_ip_structured_soft` via `soft_closure_with_retry` (soft rows:
   the IDS), with the rows and their discrepancies; the Ip round trip checked
   by `ip_roundtrip_gate` (0.05 %). A refusal (scale bounds 0.2 < s < 5, a
   degenerate row, a failed round trip) raises `EngineClosureRefused` with
   its reason.
3. Relax: `js = (1−β) js_prev + β jc` (`CurrentRelaxer`, `jbs_relax_current`).
4. **One GS solve** of `js` (+ the delivery correction, when on).
5. Measure on the solved state: Redl `<j.B>`, l_i (`li_achieved`, li_3), Ip,
   the uniform factor `c = achieved / request` (linear Ip measure), q at the
   row radius, tan γ at the chords, request − achieved (core/edge, % of peak).
6. Update (between passes, never after the last): `lambda_bs` by the kernel
   (ω); the row discrepancies; the MSE linearisation point and Jacobian; the
   delivery correction.

The kernel is `run_jbs_loop` with `step` = 1–5; the engine's rows enter
through its `extra` hook and the existing `AxisRowPin`.

Every GS solve -- the anchor, each pass, the delivery, the σ=0 check and
every draw -- uses one coil solve: OpenFUSIONToolkit's bounded (BVLS) coil
least squares, entered once at `Bouquet.setup_solver`
(`bouquet.solver_state.enter_bounded_coil_mode`; ±1e98, never binding) and
recorded as `coil_solve_mode` on the Baseline and in the engine record. The
mode is one-way and every `generate()` enters it, so entering it before the
reconstruction is what keeps the reconstruction and its draws on the same
coil solver whatever the call order (the unbounded and bounded solves agree
to round-off per solve, not bit for bit).

### Rows and their update

| row | model in the closure (on `G_k`) | measurement on `E_k+1` | update |
|---|---|---|---|
| Ip | `Ip_fsa_weights` affine exact measure | the solver imposes Ip; `c` recorded | — |
| l_i | `structured_li_model` (li_3) | `li_achieved` | `d_k = (1−rω) d_k−1 + rω [l_i(E_k+1) − l_i_model(js_k; G_k+1)]`, `r = engine_li_row_relaxation` (default 1); the closure's target is `T − d` |
| q0 | the axis-current row | q at the row radius | `AxisRowPin`: `j_ref0 ← j0_solved · q0 / q0_target` |
| MSE | `tan γ ≈ tg0 + J (x − x0)` | `mse_tan_gamma` of the solved field (`mse_field_at` -> `(B, found)`) | offset refreshed from every solve; `J` by finite differences once at convergence, then Broyden (`"fd_broyden"`) or held (`"fd_chord"`); re-taken at the MSE loop's convergence (below) |

MSE is a second run of the loop after the no-MSE loop has converged (not a
row of the first one), with the chords as an added criterion: at the first read (the finite-difference
base) chords OFF the solver mesh are excluded with their reason; fewer than
`structured_mse_min_chords` left leaves the MSE term NOT applied and the
slice flagged (refused with `structured_mse_required=True`); a chord missing
on a later read is a refusal. The field orientation is STATED, never fitted:
`sign_pol = ip_sign(data)·ip_sign(equilibrium)`, `sign_tor =
bt_sign(data)·bt_sign(equilibrium)`, the equilibrium's read off its field;
if another orientation fits the chords better by Δχ² >
`mse.MSE_ORIENTATION_DCHI2` the slice is flagged and the stated one KEPT.
The adapters complete a block that states no `ip_sign`/`bt_sign` from the
source's declared orientation in the (R, φ, Z) frame (g-file: CURRENT and
BCENTR signs times its COCOS's σ_RpZ; IDS: `ip`, `b0` signs, COCOS 11) and
refuse a block the source contradicts. MSE knobs the engine never reads
(any, without the `"mse"` row; `structured_mse_steps` with it) are refused
as on the legacy path.

On a **hard** row the closure imposes `model + d = T`, so at the fixed point
the delivered l_i is on the target. On a **soft** row the fit weighs
`(model + d − T)/σ`, which at the fixed point is the delivered residual
`(l_i − T)/σ` (the `LiRowPin` soft semantics). The l_i discrepancy is taken
against the model of the current that was actually SOLVED, evaluated on the
NEW geometry -- see "Deviations" below.

**Under-relaxing the l_i row (`engine_li_row_relaxation`, default 1.0).**
The update above moves `d` toward the measured discrepancy by `rω` per pass
(the first update, from `d = 0`, by `r`). Linearised, the row error obeys
`e_k+1 = (1 − G) e_k` with `G = rω·c·(1 + s)` (`c`: the fraction of the row
error the update measures on the new geometry; `1 + s`: how far the delivered
l_i moves per unit move of the model's target), so a row whose `G` exceeds 2
oscillates with a growing period-2 amplitude. `r < 1` scales `G` by `r`. It
changes the path only: the fixed point (`d` = the measured discrepancy), the
targets, and every tolerance, criterion and ceiling are unchanged, and
`r = 1` is the update before the setting existed, bit for bit. Reconstruction
only; the q0 row (`AxisRowPin`) and the MSE chords are not affected.

The MSE linearisation point is the coefficient vector whose current was
solved: with the current relaxed it is the same β-blend of the closure's
coefficients (the composition is affine in them).

### Convergence (existing constants only, two consecutive passes)

`engine.convergence_table()` returns this, with each value read from its home:

| criterion | value | where it lives |
|---|---|---|
| r_j | 1e-3 | `GenerationConfig.jbs_rtol_j` |
| r_I | 1e-4 | `GenerationConfig.jbs_rtol_Ip` |
| \|Δl_i\| | 1e-3 | `GenerationConfig.jbs_tol_li` |
| \|Δq0\| (q0 row) | 2e-3 | `GenerationConfig.jbs_tol_q0` |
| \|q0 − q0_target\| (q0 row) | 0.01 | `GenerationConfig.q0_tol` (via `AxisRowPin`) |
| l_i, g-file hard row | 1e-3 | `TokaMaker_interface._rematch_li_request` `li_tol` default |
| l_i, IDS hard row / soft-row discrepancy | 0.005 | `GenerationConfig.structured_li_tol` |
| MSE tan γ change | 0.1 σ_eff | `jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA` |
| closure-half current residual (standing, decision 9) | 1e-3 | `jbs_rtol_j`, the loop's current gate |
| consecutive passes / ceiling | 2 / 12 | `JBS_REQUIRED_CONSECUTIVE` / `jbs_max_passes` (12 since 2026-10-07, was 8) |
| ω floor / growth abort | 0.25 / 3 | `JBS_RELAX_FLOOR` / `JBS_GROWTH_ABORT_PASSES` (earliest abort pass 10 at the default relaxation: inert within the post-homotopy ceiling 6, reachable within the reconstruction / MSE and draw ceilings of 12) |
| closure scale bounds | 0.2 < s < 5 | `close_ip_structured` default |
| Ip round trip | 0.05 % | `utils.IP_ROUNDTRIP_TOL_PCT` |

r_j and r_I are measured against the bootstrap the pass SOLVED (2026-10-06,
`jbs_loop` "Residuals"): with the solved current relaxed (β < 1) a pass
solves the β-blend of its closure current with the previous solved one, so
the bootstrap that equilibrium carries is the same blend of the iterates,
and "converged" means the solved state's bootstrap is within the tolerances
of its own Redl evaluation. The residual against the iterate is recorded as
`r_j_iterate` / `r_I_iterate` (record only; the ω schedule follows it, so
the path is unchanged). Until then the iterate residual was the criterion,
which on a blended pass can read converged while the solved state is not (a
toy: 2.4e-4 reported, 1.09e-3 against the solved bootstrap); the stricter
reading can take more passes within the same ceilings (on the fast suite's
toys: the gated two-state toy 8 -> 9 passes, the soft fd_broyden MSE-stage
toy 8 -> 12, beyond the 8-pass ceiling that was then the default; the reconstruction ceiling is 12 since 2026-10-07). The engine draw's delivery check and
the bootstrap it carries into the post-homotopy stage are the solved one.

The MSE stage runs the loop again after the Jacobian, with its own ceiling
`jbs_max_passes` (as the legacy chord stage).

**Jacobian refresh at convergence (2026-10-07).** A chord iteration with a
Jacobian taken once stops where `J0ᵀ W r + ∇prior = 0`, not at the χ²
minimum. At each MSE loop's convergence the engine therefore re-takes the
Jacobian by the same finite differences (1 base solve of the last closure's
coefficients + one per free coefficient), runs the closure once with the old
and once with the refreshed Jacobian (zero solves) and converts the
refreshed Gauss–Newton step to the tan γ move it predicts. Within the
stage's own criterion (`MSE_CHORD_OFFSET_TOL_SIGMA`, 0.1 σ_eff on every
chord) the stage is converged and the delivery is unchanged; above it the
loop CONTINUES from the refresh's base state with the refreshed Jacobian
(same pass ceiling), and the Jacobian is re-taken again at its convergence,
at most `engine.MSE_JACOBIAN_MAX_REFRESHES` (3, a cost ceiling, owner-approved 2026-10-07: non-converged only) times -- a
stage whose fresh-Jacobian step never settles is NOT converged (raised, or
flagged under `jbs_loop_on_fail="flag"`). Recorded per refresh in the MSE
phase's `jacobian["refresh"]["rounds"]`: `jacobian_refresh_rel_change`
(`|J_new − J_old|_F/|J_old|_F`), `refresh_step_norm` (max |Δx| of the
refreshed step), the old-Jacobian step, their relative change and the
predicted tan γ moves; the solves are `solves["mse_refresh"]`. On the fast
suite's toy the Jacobian moves 17–19 % between the no-MSE state and the fit
and the refreshed step stays below 0.01 σ (data the profile family fits);
on a toy whose pitch angles respond non-linearly to the current the first
refreshed step is 0.3 σ and one continuation brings it to 0.003 σ.

**Branch-consistent finite differences (2026-10-14).** Every finite-
difference Jacobian of the stage (the first and each refresh) checks that
its perturbed solves stay on the base equilibrium's solution branch: a
perturbed solve whose magnetic axis lands farther than
`engine.MSE_FD_BRANCH_AXIS_FRAC` (0.01) × the base state's minor radius
from the base axis has jumped to the free-boundary problem's other solution,
and its column is re-taken from the base state with the step halved, at most
`engine.MSE_FD_MAX_HALVINGS` (4) times. The first step that stays gives the
column; a column whose solves never stay keeps its original step and is
recorded unresolved. Why: on a slice with two flux extrema of nearly equal
depth the free-boundary solve has two solutions for nearly the same current
(measured on a reversed-Ip slice: axes 17 mm apart, Δl_i 2e-3, Δq0 4e-3). A
column taken across that jump made the refreshed Jacobian's Gauss–Newton
step send the stage to the other branch (χ²/N 0.745 → 0.790) and the next
refresh into a period-3 cycle between the two that ended at the pass ceiling
(`JBSNotConverged`); with the check every column stayed on its branch at
h/2 … h/8 and the stage converged at its first refresh. A numerical-method
parameter only: it chooses which difference step a column uses; no
criterion, tolerance or ceiling is touched, and a state without a jump is
differenced exactly as before (bit for bit; the extra solves are counted in
`solves["mse_fd"]` / `solves["mse_refresh"]`). Recorded as `branch_check`
on the MSE phase's `jacobian` record and on each refresh round (threshold,
minor radius, the refined columns with their steps and axis shifts). The
backend reports its minor radius (`TokaMakerBackend.minor_radius()`, the
solver's `a_geo` on a copy of the state); a backend without it runs the
Jacobian unchecked.

**When the MSE stage fails (2026-10-07).** With `structured_mse_required=
False` (the default) ANY exception inside the stage (a failed solve in the
finite differences or a pass, a closure refusal, a non-converged MSE loop
under `jbs_loop_on_fail="raise"`, a failed refresh) or in the delivery of its
fit restores the converged reconstruction WITHOUT MSE (backend state through
`solver_state.SolverState`, the engine state, the last pass, the q0 pin, the
chord set), delivers it through the ordinary delivery (whose solve and
checks re-verify it; on the toy it is the run without the `"mse"` row bit
for bit) and flags `mse_stage_failed` with the exception text (closure-
limited reason, printed, `RuntimeWarning`; `engine_record()["mse"]["failure"]`).
With `structured_mse_required=True` it raises, as before. The failed phase
(`phases[...]["jacobian"]`, `failed=True`) keeps the stage's Jacobian record
as far as it got: when the finite-difference Jacobian was taken, every key a
delivered MSE phase carries (`n_free`, the FD's `n_solves`, the scheme, the
orientation, `J_initial`, the passes' `n_broyden_updates` / `n_pass_solves`
and the refresh rounds), with `jacobian_taken=True`; otherwise `n_free` and
`n_solves` (= the solves spent). `n_solves_spent` is always every solve
since the snapshot (`solves["mse_failed"]`). Every MSE phase record (applied,
not applied, failed) carries `n_free`.

**The delivered fit's chord χ² (2026-10-07).** Against the raw E_r-corrected
chords of the stage with its own weights (`mse.mse_chi2`; never against an
EFIT q profile): `checks["mse"]` and `engine_record()["mse"]` carry χ², N,
χ²/N and the χ² of the reconstruction without MSE (the finite-difference
base solve, same chords), and two FLAGS (flag only, never acceptance; each a
closure-limited reason): `mse_worse_than_without` (delivered χ² above the
pre-MSE χ²) and `mse_chi2_per_chord_high` (χ²/N above
`GenerationConfig.mse_chi2n_flag`, default 10.0 -- owner-approved 2026-10-07, flag only).
Under `jbs_loop_on_fail="flag"` a non-converged fit is delivered with
`mse_converged=False` and the same records.

### Failure

Exactly the loop's: `JBSNotConverged` raises (or, with
`jbs_loop_on_fail="flag"`, the result is delivered flagged
`closure_limited`); `JBSNonFinite` raises at once whatever the policy; a
closure refusal raises `EngineClosureRefused` with its reason; a failed GS
solve raises `EngineSolveError`. A failed build leaves no baseline
(`Bouquet._failed_baseline` keeps the half-built one). The one exception is
a failure inside the MSE stage with `structured_mse_required=False`: the
reconstruction without MSE is delivered, flagged `mse_stage_failed` (see
the MSE stage above).

### Delivery

After the loop (and the MSE stage), the closure is run once more on the last
solved geometry with the last bootstrap and discrepancies, **unrelaxed**, and
solved with two passes. It is checked with `jbs_loop.check_delivered` (r_j,
r_I) and on every row: the l_i row error, q0 against `q0_tol` and Δq0, the MSE
tan γ change, Δl_i and the closure-half current residual against the last
pass. A miss raises `JBSNotConverged` (or flags). That solve, its request,
its geometry snapshot, `x*`, `lambda_bs*` and the discrepancies are the
reconstruction: `EngineState`, recorded in `Baseline.engine["state"]`.

**Closure health.** `engine.engine_closure_health` evaluates
`utils.closure_health` on the delivered closure on BOTH input types
(2026-10-06; the g-file path never called it): the effective bootstrap scale
`bs_scale_eff` against the ±50 % bootstrap prior -- `|s_bs - 1| > 0.5` is
flagged `bootstrap_scale_out_of_prior`, printed and warned loudly, never
clamped (a closure failure, not a finding) -- with the `s_bs(ψ)` range
recorded beside it. IDS: on `ip_closure` (every reason folded into
`closure_limited`, as before) and `li_metrics["bootstrap_prior"]`; g-file:
`reconstruction_metrics["closure_health"]`, of which only the prior flag is
folded into the baseline's `closure_limited` (the raw-component Ip mismatch
is recorded, not a g-file flag).

**What a draw inherits.** Composing on the stored geometry snapshot with the
stored components, `x*` and `lambda_bs*` (+ the delivery correction)
reproduces the stored request bit for bit (tested, and re-checked by every
draw context before it draws); the state also carries the q0 row and target,
so a draw can keep the q0 row as an option for sawtoothing discharges.

## Settings the engine does not read: refused

A setting that is accepted is honoured or refused, never silently ignored.
Under `reconstruction_engine="unified"` each legacy-path setting below is
REFUSED when it holds anything but its default (`engine.
ENGINE_UNREAD_LEGACY_FIELDS`; `workflow='custom'` downgrades the refusal to
a printed WARN, as for the MSE knobs), with what replaces it under the
engine. Since the engine is the default (2026-10-06), a configuration written
for the legacy paths reaches these refusals unless it names its engine; every
refusal ends by saying so: set `reconstruction_engine="legacy"` (or build with
`Bouquet.from_geqdsk` / `from_imas(..., reconstruction_engine="legacy")`) to
run the legacy reconstruction and draws instead. The same holds for the
engine's requirements (`jbs_self_consistent=True`, `recalculate_jbs=True`,
`single_profile_jphi=False`, `jbs_init="anchor"`, no `draw_solve_maxits`):
`jbs_self_consistent=False` on a default configuration is refused with that
instruction, because the frozen bootstrap exists on the legacy paths only.

| setting | under the engine |
|---|---|
| `closure_channel`, `jbs_baseline_mode` | the engine's closure: `engine_preset` / `engine_rows` |
| `structured_preset`, `structured_basis`, `structured_weights`, `structured_sigma_ind_up` | `engine_preset` |
| `structured_li_target` | nothing: the l_i row targets the source's own l_i |
| `structured_li_sigma`, `structured_ip_sigma`, `structured_ip_sigma_frac` | nothing: the IDS soft rows use the preset's σ |
| `structured_li_kind` | nothing: the l_i row is li_3 |
| `structured_soft` | nothing: hard rows for a g-file, soft for an IDS source |
| `structured_li_max_corrector_steps` | `engine_li_row_relaxation` |
| `anchor_pressure_to_equilibrium` | nothing: no `p_diff` in the engine's pressure |
| `imas_corrective_jphi` | `engine_delivery_correction` |
| `jbs_loop_q0_corrector` | `engine_rows` with `"q0"` (`engine_draw_q0_row` for the draws) |
| `floor_jbs`, `accept_anchor_inband`, `diagnostic_plots` | nothing: legacy draw / SWB mechanics |
| `bootstrap_kwargs` keys other than `taper_edge_jphi` / `taper_edge_psi0` / `taper_edge_shape` and `use_sauter_eps=True` | nothing: the engine never runs `solve_with_bootstrap` |
| `homotopy_passes` with `engine_draw_homotopy=False` | no homotopy runs |
| `isolate_edge_jbs` (default `None`: resolved per engine; `True` under the engine) | nothing: the engine never isolates the edge bootstrap (Redl on the whole profile) |
| `perturb_jind_in_anchor` (default `None`: resolved per engine; `False` under the engine) | nothing: one engine draw route replaces Fix C and the standard l_i loop |

Already refused elsewhere: the MSE knobs without the `"mse"` row (and
`structured_mse_steps` with it), `draw_solve_maxits`
(`engine_draw_solve_maxits`), `jbs_self_consistent=False`,
`recalculate_jbs=False`, `single_profile_jphi`, `jbs_init != "anchor"`,
and in the draws `jbs_delta_mode`, `PIN_JPHI`, `DIFF_BS`,
`l_i_uncertainty > 0`.

`isolate_edge_jbs` and `perturb_jind_in_anchor` joined the refused set on
2026-10-05 (owner-approved). Their validated values depend on the engine,
so since 2026-10-07 both default to `None` ("resolve per engine",
`engine.ENGINE_DEPENDENT_DEFAULTS`) and the factories set neither.
`prepare_baseline()` resolves them once, for the engine configured when it
runs:

| setting | `"legacy"`, g-file | `"legacy"`, IDS | `"unified"` |
|---|---|---|---|
| `isolate_edge_jbs` | `False` | `False` | `True` (not read) |
| `perturb_jind_in_anchor` | `False` | `True` (diff+C) | `False` (not read) |

So it no longer matters whether the engine is named at construction or set
afterwards: these three give the same legacy run.

```python
bq = Bouquet.from_imas(dd, mesh=mesh, time=t, reconstruction_engine="legacy")

bq = Bouquet.from_imas(dd, mesh=mesh, time=t)
bq.generation.reconstruction_engine = "legacy"

cfg.generation.reconstruction_engine = "legacy"      # a hand-built config
```

(Until 2026-10-07 the factories applied the values at construction, and a
factory configuration switched to `"legacy"` afterwards ran the legacy
paths on the engine's values: an isolated-edge bootstrap with a flat core,
I_BS/I_p off by tens of percent.) The resolution is recorded on the
baseline (`Baseline.engine_resolved_defaults`, also in the engine record)
and in the archive (`_baseline` attr `engine_resolved_defaults_json`,
`utils.load_engine_resolved_defaults`): each field's value and its origin,
`"resolved from engine=<engine>"` or `"explicit"`. An explicit value is
always kept; one that contradicts the configured engine's validated value
warns, naming the field (and under the engine is refused besides, as
above). A configuration that has been through `prepare_baseline()` and is
then switched to the other engine is re-resolved, not taken for explicit
values. A stored configuration carries both fields explicitly and loads
unchanged; a stored unified configuration that carries a legacy value
(written before 2026-10-05) loads at the engine's value with a warning --
the engine never read it, so the stored run is unchanged.

## Presets

| `engine_preset` | basis / prior | rows admitted |
|---|---|---|
| `"structured"` (default) | 4 Gaussians (ψ_N 0.15/0.45/0.75/0.95, width 0.20), `li_soft_onesided` σ ladders | Ip, l_i, q0, MSE |
| `"bootstrap_scalar"` | constant basis, `s_ind` pinned | Ip |
| `"structured_uniform"` | the same 4 Gaussians under `utils.STRUCTURED_WEIGHTS_UNIFORM` (σ = 1 on every coefficient, no one-sided up-ladder) | Ip, l_i, q0, MSE |
| `"bootstrap_scalar"` | constant basis, `s_ind` pinned | Ip |
| `"sawtooth_two_scalar"` | constant basis, two scalars | Ip, q0 (falls back to `"bootstrap_scalar"` with a notice when the gate rejects q0, as the legacy sawtooth channel) |
| `"two_scalar_li"` | constant basis, two scalars | Ip, l_i (both required) |

The scalar presets reduce exactly to `close_ip("bootstrap")` and
`close_ip_q0` (tested).

**`"two_scalar_li"`** is the legacy secant's l_i family as a named closure:
one scalar on the inductive and one on the bootstrap, rows Ip + l_i. With
the g-file's hard rows it is a 2 × 2 system and no prior enters (the
weights provably do not change the answer; tested). It is exactly the state
the q95 attribution study reached by patching the settings' preset to the
constant two-scalar basis with rows Ip + l_i (tested bit for bit on the toy;
the solver test `tests/test_engine_two_scalar_solver.py` checks the two
routes agree within `jbs_tol_li` / `jbs_tol_q0` on the synthetic g-file
example, and on the build the study used that q95 is the study's 4.5956
within `jbs_tol_q0`). With soft rows (the IDS default) the soft solver
serves it, and the constant basis's σ = 1 is then an absolute prior on each
scalar -- the documented meaning of the uniform ladder; it enters the answer
weakly. It adds no number: the basis, the σ, the rows and every tolerance
are existing ones. It is not a default.

**`"structured_uniform"`** is the prior-sensitivity run of design decision
8: the shipped basis under the ONE documented alternative prior. On the hard
closure only the weights' ratios matter, so it expresses no preference; with
soft rows or MSE chords it is an absolute σ = 1 prior
(`utils.STRUCTURED_WEIGHTS_UNIFORM`). The difference from `"structured"` is
the part of the answer the `li_soft_onesided` prior -- not the data -- holds
up; it is meant to be reported.

## The delivery correction (`engine_delivery_correction`, default off)

With the solver's `jphi-linterp` defect present, the achieved current differs
from the request by a local edge footprint (~2.8 % of peak at ψ_N 0.996 on
the synthetic g-file). With the option on, each pass adds
`Δ = request − achieved/c` of the previous pass to the request (one Newton
step with a unit Jacobian), after the relaxation of the intended current, so
the delivered current approaches the intended composition. `Δ` carries no
net current in the linear Ip measure and is part of the state a draw
inherits. Its default is to be decided after the solver fix, by re-running
the distance-to-input table (`tests/probes/measure_engine.py`, part
`recon_dc`).

## Draws

`bouquet/engine_draws.py`; `Bouquet.generate()` builds a
`GenerateEngineDraws` from the live reconstruction and hands it to
`generate_bouquet(draw_method=...)`, whose per-draw loop then calls it in
place of the legacy `perturb_kinetic_equilibrium` (everything else --
the warm start, the strong coil regularisation of the post-loop phase, the
homotopy, the archive, the until-N ledger -- is the same code). The parallel
launchers need nothing new: every worker runs `prepare_baseline()` +
`generate()`.

**The loop solves under the reconstruction's own coil regularisation.** The
reconstruction records the term list it solved under
(`Bouquet._engine_run["coil_reg"]`, `engine.reconstruction_coil_reg`: the
terms `Bouquet._apply_coil_reg` installed -- `SolverConfig.coil_reg`'s
measured-coil targets at their CONFIGURED weights and its `#VSC` term as
configured, or the toward-zero default -- and their JSON record, also on
the baseline's engine record as `coil_reg`), and every draw installs exactly
that list for its loop (`GenerateEngineDraws.install_coil_reg`; recorded per
draw as `coil_reg`), the zero-perturbation draw included: a draw is the
reconstruction's closure perturbed in its inputs, not re-regularised.
`generate_bouquet` puts its strong regularisation back after the loop, as
before. (Until 2026-10-06 the loop ran under the legacy exploratory list --
every configured weight replaced by 1.0, a configured `#VSC` term replaced
by a 1e-2 pull toward zero -- identical to the reconstruction's only when
`coil_reg` is empty, as on every example; with measured targets the two
solves differed ~100x in weight.) Only on a solver object `setup_solver`
did not prepare (nothing on record) does the draw fall back to that
historical list, and its record says so.

```
 reconstruction state: G*, x*, lambda_bs*, (Delta*), request R*
            |
            v
 sample (legacy stream): kinetics (pressure-matched), aux, ONE inductive
 candidate (toroidal sigma_jphi, j_ls); the run's bootstrap scale
            |
            v
 anchor (0 solves): lambda_0 = scale [lambda_bs* + Redl(draw kin) - Redl(recon kin)]
            |
            v
 ONE loop (run_jbs_loop; ceiling jbs_max_passes_draw; current gate standing)
   compose on G_k with x* HELD, p' from the draw's own pressure
   close the Ip row: J + d_ind * (inductive term), exact measure
   [+ q0 row: the two-scalar increments, AxisRowPin moves the axis row]
   relax (beta) -> ONE solve -> measure (Redl with the draw's kinetics)
            |
            v
 coil homotopy (engine_draw_homotopy, default on) -> post-homotopy check
 (the existing _post_homotopy_jbs, the engine draw's own passes, the
 saturation guard)
            |
            v
 post-hoc filters on the archived draw: the OPT-IN bands (l_i, beta_p,
 constrain_sawteeth; all off by default); coil + boundary as always
   ->  in_spec / selected
```

**Inputs.** Every perturbed quantity is formed as `base + (drawn - base)`:
the kinetics on the kinetic grid (then PCHIP'd as the adapter does), the
solve pressure from the adapter's own assembly (thermal + impurity + fast),
the auxiliary channels, the parallel inductive. The random stream is the
legacy one through the first inductive candidate (`engine_draws.RNG_STREAM`):
the kinetic channels `ne, Te, (Zeff -> ni | ni), Ti` redrawn together until
the flux-integrated thermal pressure matches within `p_thresh`, the
auxiliary channels in their order (both by `bouquet.kinetic_sampler`, shared
with the legacy and swb draws), then one inductive candidate drawn IN
TOROIDAL UNITS with the legacy call on `s_ind(x*) kappa* lambda_ind` (so its
toroidal perturbation is the legacy draw's for the same normals: today's
`sigma_jphi` and `j_ls`) and mapped back to `lambda_ind`; it is redrawn only
while negative where its mean exceeds its sigma, and clipped to zero in the
floor zone (the standard route's rule). The legacy routes then draw further
candidates (Fix C band resampling, the standard route's l_i pre-screen); the
engine draw does not, so a seeded engine run and a seeded legacy run share
their streams up to the first legacy draw that resampled. The bootstrap
scale is the run's `jbs_scale_range` sample, used as is (1.0 is the
reconstruction's value -- `s_bs(x*)` already carries the reconstruction's
scaling, so the legacy `bs_scale` centring is not applied).

**The Ip row.** The one row a draw closes: an increment on the inductive
term, `J = J0 + d_ind s_ind kappa lambda_ind'`, with
`d_ind = [(Ip_lin* - Ip_lin(J0; G_k)) + (c* - c(G_k))] / Ip_lin(ind; G_k)`
in the exact (`jphi-linterp`) measure, the target being the Ip the
reconstruction's delivered composition carries in that measure on `G*`
(`Ip` itself on the hard g-file row, the posterior on the soft IDS row).
Zero extra solves; `a_ind = 1 + d_ind` is recorded per pass. With
`engine_draw_q0_row=True` (needs the reconstruction's active q0 row) the
sawtooth two-scalar system is solved in the same increment form on the
inductive and bootstrap terms, and `AxisRowPin` moves the axis row once per
pass from the measured q0 (the q0 criteria are then loop criteria).

**Zero-perturbation identity by construction.** With every perturbation
zero and the scale 1.0: the inputs are the base exactly, the first pass
composes on `G*` (its `p'` shifted by the draw's pressure change, zero), the
anchor increment is zero, and the Ip increment is formed from differences
that vanish exactly -- so the first request IS the stored request, bit for
bit (recorded per draw as `identity.pass1_request_bit_identical`). The solve
of it from the warm state reproduces the reconstruction, the loop's pass-1
residuals are the reconstruction's delivered ones, and the draw delivers the
reconstruction to the loop tolerances. With the current gate standing (it is
measured one pass late) the loop takes `JBS_REQUIRED_CONSECUTIVE + 1 = 3`
passes.

**The re-solve starts from the reconstruction.** "The warm state" is
generate_bouquet's baseline re-solve: the reconstruction's request solved
once more through the draws' jphi-linterp machinery, whose result is every
draw's warm start, the target of the strong coil regularisation and the
reference of the coil drift and the homotopy. On the engine route it starts
from the reconstruction's DELIVERED equilibrium -- restored from the
snapshot `prepare_baseline()` takes (`Bouquet._engine_run["delivered_eq"]`)
-- with the reconstruction's own settings and inputs, warm
(`GenerateEngineDraws.baseline_resolve_start`). It never re-initialises psi
first: that is the legacy route's guard (issue #24), for a reconstruction
solved in inverse mode, and a free-boundary solve from a cold analytic psi
can converge to a DIFFERENT equilibrium of identical inputs. Measured on a
reversed-Ip, near-double-null IDS slice (2026-10): the cold re-solve put the
magnetic axis 1.7 cm lower, with q0 -1.5e-3, l_i -0.2 % and beta_pol
+0.9 % against the reconstruction, at nl_tol 1e-6 and still at 1e-9, while
one more solve from that state, or a warm re-solve from the reconstruction,
returns to the reconstruction to ~1e-7 in q0. The legacy route keeps the
re-init (`DrawMethod.baseline_resolve_start` returns `None`).

**The homotopy holds the reconstruction's isoflux.** On that same slice the
homotopy's first solve -- with the isoflux re-pointed to the draw's traced
boundary (every traced point, weight 200, the legacy route's pattern) under
the strong coil regularisation -- selected the other equilibrium too,
starting from the reproduced reconstruction, so the zero-perturbation draw
missed the archived-stage gates (Delta q0 -3.8e-3 against 2e-3) although its
loop stage reproduced the reconstruction. The selection needed both the
re-pointed pattern and the strong regularisation, and was not monotone in
the isoflux weight: a branch-selection sensitivity of a bistable
equilibrium. Since 2026-10 (owner decision) every ENGINE draw's homotopy
holds the boundary with the RECONSTRUCTION's own isoflux pattern -- the very
points and weights the reconstruction was pointed at
(`GenerateEngineDraws.homotopy_isoflux`): on both routes the solver
geometry's (`CaseGeometry.iso_pts` / `iso_w`, the standard boundary
constraint below, whose fingerprint the solver geometry record carries as
`isoflux_sha256`). `prepare_baseline()` records it
(`Bouquet._engine_run["recon_isoflux"]`, stamped as
`Baseline.engine["recon_isoflux"]`), and every draw stamps the pattern it
installed under `engine["homotopy"]["isoflux"]` (`n`, `weight_min` /
`weight_max`, `origin`, `sha256`, `installed`); the baseline's engine record
carries it as `draws.homotopy_isoflux`. With no pattern on record the isoflux
in place is left alone and the stamp says so -- never a silent re-trace. The
legacy and swb routes keep the re-trace (`DrawMethod.homotopy_isoflux`
returns `None`). This changes every engine draw's homotopy -- its boundary
constraint density and weight, hence coil drift, LCFS deviation and yield of
perturbed draws -- and it fixes the branch selection on the slice above; it
does not prove robustness on other near-double-null slices.

**The boundary constraint per unit length (#84, owner decision for 1.4.0).**
Before, the strength with which the boundary was held followed the number of
isoflux points the source happened to carry: the IDS route held the slice's
boundary outline at `solver_geometry.ISOFLUX_WEIGHT` = 500 per point (about
200 points, total ~1e5), the g-file route installed its own set, the g-file
boundary at 200 per point (100-257 points, total 2e4-5e4), and the draws held
whichever set their reconstruction had. Since #84 the unified engine holds
ONE standard set on both routes, for the reconstruction and every draw
(`solver_geometry.standard_isoflux`, installed by the reset through
`CaseGeometry`; the g-file route no longer installs a set of its own):

- the source boundary (g-file boundary, IDS `boundary.outline` or the
  `LCFS_geqdsk` boundary) is resampled to `solver_geometry.N_ISOFLUX` = 256
  points by arc length (`resample_boundary`): an interpolating cubic spline
  through the source points, broken at the X-points and at any vertex where
  the source polygon turns by more than `SPLINE_BREAK_DEG` = 20 deg AND by
  more than `SPLINE_BREAK_MEDIAN_RATIO` = 3x the median turn of the vertices
  within `SPLINE_BREAK_MEDIAN_WINDOW` = 3 of it (so a corner is never
  smoothed or overshot, while a coarse smooth source -- 16-30 points, whose
  tips turn 29-46 deg per vertex -- keeps one smooth spline instead of
  becoming a polygon 6-24 mm inside its curve), spaced uniformly in arc length
  between X-points; a closed contour (no repeated point), counter-clockwise;
- the source's X-points are kept as EXACT points (the source's own
  vertices): a vertex whose turning angle over a window of
  `XPOINT_WINDOW_FRAC` = 1 % of the boundary length on each side exceeds
  `XPOINT_TURN_DEG` = 45 deg (the sharpest within two windows), or the vertex
  a saddle target (`SolverConfig.saddle_targets`) marks within
  `XPOINT_SADDLE_MATCH_FRAC` = 1 % of the length -- the sharpest vertex in
  that radius when it turns by more than `XPOINT_TURN_DEG` (so a target off
  the X-point still marks the X-point), else the nearest (stamped as
  `xpoint_match`); the saddle constraints
  themselves are installed exactly as configured;
- each point is weighted by the arc length it represents (half the spacing to
  each neighbour), the weights summing to the case's total (`per_length_weights`):
  `solver_geometry.ISOFLUX_WEIGHT_TOTAL` = 2e5, a constant inside a measured
  window (next section), or `SolverConfig.isoflux_weight_total` when set. A total is
  nominal: TokaMaker's `set_isoflux` does not apply the
  reference point's (the first point's) weight, so the installed total is
  the nominal total minus `w[0]` (199 219 for a set without X-points); the
  record stamps both (`isoflux_weight_total`,
  `isoflux_weight_installed_total`; the isoflux stamps `weight_total`,
  `weight_installed_total`). The
  constraint strength then depends on the boundary, not on the writer of the
  file or the mesh: a 100-point and a 10 000-point sampling of one contour
  give the same total and the same points to micrometres;
- the set's first point -- the point TokaMaker measures every other isoflux
  point's flux against -- is the outermost (largest-R) point, never the
  X-point. With the X-point first, the reversed-Ip near-double-null IDS slice
  of the ladder stopped at maxits in its reconstruction (also at a total of
  1e5); with the same points and weights and the outermost point first it
  converged as before.

A `SolverConfig.isoflux_pts` (with `isoflux_weights`) set is used AS GIVEN,
never resampled, on every route (also on the engine's g-file route, which
before #84 replaced it with its own set). Given WITHOUT `isoflux_weights`,
the unified engine does not leave it at TokaMaker's default of 1 per point
(about 780x weaker than the standard set): each point is weighted by the arc
length it represents along the points in the order given (a closed contour
when the closing gap is no longer than the longest other spacing, else an
open arc), the weights summing to `ISOFLUX_WEIGHT_TOTAL`
(`solver_geometry.user_isoflux_weights`), and a UserWarning says so; the
record stamps those weights and the rule. On the legacy route such a set is
installed at 1 per point as before, and stamped as 1 per point. The legacy and swb routes are
unchanged: the reset installs the source boundary as given at
`ISOFLUX_WEIGHT` per point, the legacy g-file reconstruction its own set at
200 per point, and the legacy draws re-trace (`DrawMethod.homotopy_isoflux`
returns `None`). The solver geometry record (`solver_geometry_json`) stamps
`isoflux_rule`, `isoflux_n`, `isoflux_weight_total`, `isoflux_source_n`,
`boundary_length_m` and `isoflux_resampling` (method
`arc_length_resample/1`, N, the X-points kept with how each was found, the
spline breaks, the source's point count, the resampled and source polygon
lengths, the first point); every isoflux stamp (the reconstruction's
`recon_isoflux`, each draw's `engine["homotopy"]["isoflux"]`) carries
`weight_total`. The legacy draws' iso-update normalised per unit length of
the traced LCFS (#81) is the same form on that route.

**Every reconstruction and draw moves slightly** (measured at the total 2e5 of
#84). The boundary is held at
resampled points with a per-length weight, at a total 2x the IDS route's
former one and 4x-10x the g-file route's, so every unified-engine number
moves. Measured (one thread each, the same tree before and after #84;
`l_i(3)` and q at the q row are the engine's delivered measurement; the
boundary RMS is the solved LCFS's distance from the SOURCE boundary polygon,
the same measure before and after):

| case | isoflux before -> after | l_i(3) | q (q row) | I_BS (Redl), relative change | boundary RMS [mm] | reconstruction wall [s] |
|---|---|---|---|---|---|---|
| g-file + IDA profiles + p-file fast pressure (diverted) | 100 @ 200 (2e4) -> 256, 2e5 | 0.825765 -> 0.825773 | 1.030095 -> 1.030043 | +5.6e-4 | 2.21 -> 1.38 | 90 -> 93 |
| IDA-hybrid IDS slice (ordinary) | 187 @ 500 (9.4e4) -> 256, 2e5 | 0.722319 -> 0.722335 | 0.987696 -> 0.987791 | +3.4e-4 | 0.82 -> 0.67 | 44 -> 43 |
| IDA-hybrid IDS slice, reversed Ip, near double null | 197 @ 500 (9.9e4) -> 256, 2e5 | 0.838506 -> 0.838477 | 0.957527 -> 0.957725 | +4.1e-4 | 1.49 -> 1.24 | 53 -> 54 |
| synthetic dd, 1.2 s (dd kinetics) | 257 @ 500 (1.3e5) -> 256, 2e5 | 0.855757 -> 0.855873 | 1.216590 -> 1.216761 | +3.2e-4 | 0.64 -> 0.53 | 58 -> 58 |
| synthetic H-mode g-file | 257 @ 200 (5.1e4) -> 256, 2e5 | 0.835025 -> 0.834882 | 1.412861 -> 1.413233 | +8.9e-4 | 1.35 -> 0.83 | 59 -> 66 |

Every reconstruction moved by at most 1.4e-4 in l_i(3) and 3.7e-4 in q --
under a fifth of the sigma=0 gates (1e-3, 2e-3) -- and held the source
boundary as well or better. The sigma=0 gates, all with margin:

| case | before: r_j / r_I / dl_i / dq0, VSC drift | after: r_j / r_I / dl_i / dq0, VSC drift |
|---|---|---|
| g-file + p-file (diverted) | **FAIL**: 3.2e-4 / **1.06e-4** / -2.3e-4 / +6.6e-4, 0.51 % | PASS: 2.0e-5 / 1.5e-6 / +4.0e-5 / +8.1e-5, 0.087 % |
| IDS slice (ordinary) | PASS: 3.5e-5 / 2.1e-6 / -2.8e-5 / -2.4e-5, 0.029 % | PASS: 4.0e-5 / 2.6e-6 / -1.7e-5 / +3.7e-6, 0.007 % |
| IDS slice, reversed Ip, near double null | PASS: 7.6e-5 / 7.7e-6 / +2.0e-4 / -5.1e-6, 0.001 % | PASS: 9.5e-5 / 1.0e-5 / -2.2e-5 / -2.8e-7, 0.008 % |
| synthetic dd, 1.2 s | PASS: 1.6e-4 / 1.6e-5 / +1.9e-5 / -5.5e-5, 0.003 % | PASS: 1.3e-4 / 1.3e-5 / -6.4e-6 / -5.1e-5, 0.010 % |
| synthetic H-mode g-file | PASS: 2.6e-5 / 1.1e-6 / -5.8e-5 / -5.5e-5, 0.020 % | PASS: 2.7e-5 / 1.5e-7 / -2.2e-5 / +1.2e-5, 0.023 % |

(gates: r_j 1e-3, r_I 1e-4, l_i 1e-3, q0 2e-3; the archived stage, the binding one.)

The g-file case is the one #84 was decided on: before, the draws' homotopy
(every coil pinned by the strong regularisation, only the VSC channel free)
found a different fixed point under the g-file route's weak boundary (axis
0.3 mm away, the VSC pair +/-0.5 %) and the sigma=0 draw missed its
archived-stage bootstrap gate; spreading the SAME total per unit length had
no effect, five to ten times that total held the state. At 2e5 the sigma=0
draw holds the reconstruction (VSC 0.09 %) and its perturbed draws:

| case | before | after |
|---|---|---|
| g-file + p-file, 6 attempts (that arm's in-spec: the coil verdict and an l_i band) | 4 archived, **0 in spec**, 2 rejected (homotopy maxits, post-homotopy loop); VSC 2.5-8.4 %, F <= 0.07 %; 173 s per draw | 6 archived, **4 in spec**, 0 rejected; VSC 0.3-4.6 %, F 0.1-1.5 %; 132 s per draw. The two not in spec miss the l_i band (the same draws as before) |
| synthetic dd 1.2 s, until 6 in spec | 6 in 12 attempts (6 refused as non-physical kinetics); VSC 0.06-1.3 %, F 0.06-0.7 %; 112 s per draw | 6 in 12 attempts (the same 6 refused; the in-draw homotopy coil verdict flags one of the six, F 3.9 %, which the run's coil test passes); VSC 0.01-0.6 %, F 0.5-3.9 %; 95 s per draw |
| synthetic H-mode g-file, until 6 in spec | 6 in 7 attempts (one archived out of spec, VSC 2.0 %); VSC 0.2-2.0 %, F 0.02-0.25 %; 128 s per draw | 6 in 6 attempts; VSC 0.1-1.1 %, F 0.3-1.4 %; 106 s per draw |

The stronger boundary moves part of the draws' coil response from the VSC pair
to the shaping coils: their F-coil drifts grow (still inside the homotopy's
bound), their VSC drifts shrink.

The goldens are not regenerated here: they are regenerated on the merged
code (the solver-marked golden tests move with every unified-engine number).

### The boundary constraint: a constant total inside a measured window

**The model choice.** The boundary (isoflux) constraint and the coil
regularisation are both least-squares penalties in TokaMaker's coil solve:
every constraint row is scaled by its weight, the isoflux rows act on the flux
difference `psi(x_j) - psi(x_ref)` [Wb/rad] and the coil rows on `mu0 x` the
current's departure from its target [A]. Only their RATIO shapes the solution: a
perturbation of the plasma is split between moving the boundary and moving the
coils in proportion to it. The in-spec rule judges every draw on both -- the
boundary RMS cut and the coil cut -- so the total isoflux weight `W` should sit
where a boundary miss equal to the boundary cut and a coil drift equal to the
coil cut cost the solver comparably.

**The set** is the standard set of #84 (above): the source boundary resampled
to 256 points by arc length, per-length weights, the outermost point first.
**The strength** is ONE constant, `solver_geometry.ISOFLUX_WEIGHT_TOTAL` =
2e5 (owner decision 2026-10-16), for the reconstruction and every draw, inside
two measured ranges:

- the **window**, about 1.5e5-4e5: below it a diverted g-file case's
  vertical-control drift grows as the boundary weakens and its in-spec draws
  fall (7 of 10 at 2e5, 3 at 1e5, 1 at 5e4; at the pre-#84 g-file route's
  ~2e4 its sigma=0 draw missed the archived-stage bootstrap gate); at 4e5 the
  boundary is already the stiffer side (partition 0.3-0.9) and that case keeps
  5 of 10 in spec. Every sigma=0 draw passes its gates from 5e4 to 4e5;
- the **balance-crossing band**, 1.8e5-3.8e5: the total at which the partition
  below equals 1, measured on six equilibria (three experimental, three
  synthetic), at 2 % and 10 % probe amplitude, with and without the
  homotopy's coil bounds (where a bounded probe converges unsaturated it
  reproduces the unbounded partition to a few %).

`SolverConfig.isoflux_weight_total` (default `None` = the constant) is the
explicit override, installed as given and stamped "override"; a
`SolverConfig.isoflux_pts` set is used as given; the legacy and swb routes are
unchanged.

**Why not a per-case calibration.** A root find that brings the partition to 1
on each case was built and rejected (2026-10-16). On cases whose boundary
and vertical-control pair move as one rigid vertical displacement (the pair's
`#VSC` channel is nearly unregularised on purpose, below), the partition is
flat in `W` over most of the window, so `rho = 1` does not fix `W` there: the
root find landed a decade below the window, where the draws' vertical-control
drift took most of them out of the coil cut. With the pair excluded the
partition is monotone, but its crossing then lands inside the band above,
which a constant already sits in.

**The partition, reported** (`bouquet.boundary_balance`, report only). After
the reconstruction has converged, ONE extra solve, capped at
`engine_draw_solve_maxits`: the delivered request re-solved from the
delivered equilibrium with the pressure scaled by 1.02
(`PARTITION_PRESSURE_STEP`; a beta_p perturbation at fixed current request and
I_p, deterministic; 2 % is the amplitude at which the probe was feasible on
every case measured, at 10 % some experimental cases stopped at the cap),
under the draws' strong coil regularisation (1e4, `#VSC` 1) toward the
delivered currents and the installed set. The partition

    rho = (boundary RMS / boundary cut) / (coil fraction)

uses the in-spec rule's own measures (the boundary RMS as
`boundary_deviation_mm` measures a draw against the boundary cut the rule
resolves, the generic 5 mm when the cut is off, stamped; the coil fraction the
binding one of the coil test's criteria, each exactly 1 at its cut:
`sqrt(chi2_nu / chi2_max)` and `max|z| / z_max`, or the legacy F-coil drift
over `F_max`), with the vertical-control pair EXCLUDED from the coil fraction
(it moves with the boundary as above, and would pin the ratio near 1). The
delivered equilibrium and the reconstruction's coil regularisation are
restored afterwards; the baseline is unchanged and the total is never
changed. Outside `PARTITION_BAND` = [1/3, 3] a UserWarning names it (one cut
costing the solver more than three times the other; the band holds the
0.8-2.6 the six cases measured at 2e5). `SolverConfig.boundary_partition_probe`
(default `True`) switches it off (stamped `status` "off").

**Stamped** in the solver geometry record (`isoflux_weight_mode`: "constant",
"override", "as given"; `isoflux_balance`, with the probe under `partition`)
and the baseline's engine record (`boundary_balance`, the same): the nominal
and installed totals (`total_nominal`, `total_installed`: the reference point's
weight is not applied), `partition`, `boundary_frac`, `coil_frac`, the
boundary RMS and max, the coil test and its acceptance, the excluded coils,
`in_band`, `n_solves`, `solve_s`, `wall_s`, `status` and any error.

**The vertical-control bound of the homotopy.** The draws' homotopy bounds the
`#VSC` channel at `drift_VSC x` a scale. Before 2026-10-16 the scale was the
smaller coil of the anti-series pair, `min(|I_F9A|, |I_F9B|)`; on a
reversed-I_p case whose pair carried 66 kA and 2.8 kA that bound was 279 A in
the first pass: TokaMaker's vertical-control column sat on it from the first
iteration, the axis ran away vertically (Z from -0.02 m to +/-0.7-1.0 m) and 9
of 10 draws stopped at the iteration cap -- an infeasible solve, not a slow
one (a cap of 400 or 1000 reached a non-finite state instead), while the
in-spec rule accepts several hundred A on that pair. The scale is now
`max(min|I_pair|, _COIL_DRIFT_DENOM_FLOOR_A)` (10 kA;
`TokaMaker_interface._vsc_bound_scale`), and the homotopy's VSC saturation
measure is taken on the same denominator (`_coil_bound_max_vsc`; the pass
loop, the early-stop check and the post-homotopy jphi_bs guard). With it that
case archived 9 of 10 (9 in spec) at 2e5. It is inactive for a pair whose
smaller coil carries more than 10 kA (every synthetic fixture, the diverted
g-file case).

*What the floor shares with the acceptance rules, exactly.* It reuses the
legacy in-spec VSC metric's CONSTANT, not that metric's scale. The legacy
metric (`_vsc_channel_drift_pct`; `coil_filter="legacy"` and the chi2
rule's fallback) judges the pair's common-mode and differential channels
against `sigma_VSC = hypot(I_A, I_B)/2 + 10 kA`: the 10 kA is ADDED to the
propagated coil scale, so on the 66 kA / 2.8 kA pair that scale is 43 kA,
while the homotopy's is `max(2.8 kA, 10 kA)` = 10 kA. The default acceptance
(`coil_filter="chi2"`) judges each coil against its own measurement sigma
(`bouquet.coil_spec`) and does not use the constant at all. The floor
therefore does not put the homotopy on the scale an acceptance rule judges
the pair on: it lifts a bound that was far tighter than any acceptance rule
needs to one that is still tighter than the legacy metric's scale. **This
changes a homotopy coil bound; it changes no acceptance gate**
(owner-approved 2026-10-16): the in-spec rule and
the sigma=0 coil gate (the archived per-coil percentage against the tightest
stage) are unchanged. The hard-factor bounds (`coil_drift_hard_factor`, not
default) build their `#VSC` bound on the same floored scale and their guard
measures on it: an outer ceiling tighter than the homotopy's own bound would
bring the infeasibility back in the draw's jphi_bs loop.

**The vertical-control weight** stays `vsc_soft_reg_weight` = 1.0 (its
docstring records the scans): weights 1, 10 and 100 give identical draws;
1e3 starves the vertical control (0 of 10 at a total of 5e4, where the sigma=0
draw fails too); 1e4 fails the sigma=0 draw on every case measured, the axis
running away with the channel on its bound. A heavier weight does not move
the pair's drift onto the shaping coils; it removes vertical control.

**The vertical-runaway label.** A homotopy stage whose solve stops at the
iteration cap with the vertical-control channel at its `#VSC` bound (the
homotopy's saturation fraction 0.99; OpenFUSIONToolkit does not report the
channel's current, so it is read as the pair's anti-series drift `|dI_A -
dI_B|/2` over the installed half-width -- a bare drift of one coil or a
same-sign drift does not count) and the magnetic axis more than
`VERTICAL_RUNAWAY_AXIS_FRAC` = 0.1 a from the delivered axis VERTICALLY
(`|dZ|`; a radial excursion does not count) is labelled `"vertical_runaway"` in its
cap event (below). 0.1 a is ten times the axis-displacement flag and above the
several % of a that kinetic draws move the axis; the runaway it names moved the
axis by about one minor radius. A label: nothing is accepted, refused or
filtered on it.

**Tests.** OFT-free (`tests/test_boundary_balance.py`,
`tests/test_homotopy_vsc_floor.py`): the constant is 2e5 with its window and
band in its docstring; the VSC weight is 1.0 with its scans recorded; the
bound builder uses the floor for a near-zero coil and the coil otherwise,
saturation sits exactly at the bound; the label; the partition's coil fraction
leaves the pair out and is 1 at the cut; the probe solves once, restores the
delivered equilibrium and the coil regularisation, warns outside the band and
records a failed solve; the stamps. Live (`tests/test_boundary_balance_solver.py`,
`-m solver`), on the synthetic H-mode and L-mode g-file pairs and the
synthetic dd at 1.2 s: the stamped partition at the default total lies in
[1/3, 3] at no more than 2 probe solves -- a change of TokaMaker's conventions
(the row scaling, the `mu0` of the coil rows, the flux units) moves it by that
convention's factor, far outside the band; and 10 fixed-seed draw attempts
archive at least 80 % of the attempts that reach a solve. The sigma=0
reproduction (`tests/test_sigma0_identity_solver.py`) is unchanged. No fixture
with a vertical-control coil below 10 kA ships with the repository: the
floor's yield on such a case is verified on the owner's private ladder, its
arithmetic by the fast test.

**History.** Before #84 the strength followed the source's point count (about 2e4
on the g-file route); #84 standardised the set per unit length and fixed the
total at 2e5. A reversed-I_p case's draws then stopped at the iteration cap;
a per-case calibration was tried (above) and the cause turned out to be the
homotopy's vertical-control bound, now floored.

**Is this baseline bistable? (`check_bistability`, on by default.)**
`GenerationConfig.check_bistability` defaults to `None`, resolved at
`prepare_baseline()` like the other engine-dependent settings
(`engine.ENGINE_DEPENDENT_DEFAULTS`, recorded in `engine_resolved_defaults`):
`True` under the unified engine (owner decision 2026-10-15), `False` under
`"legacy"`, which has no such check (its default is never refused; an
explicit `True` is). An explicit value, a stored config's included, is kept.
With it on, `prepare_baseline()` spends ONE extra solve
after the reconstruction (`engine.check_bistability`): the solver is put
back on the reconstruction's delivered equilibrium (its targets, profiles,
isoflux and coil regularisation -- the same inputs), psi is re-initialised
COLD as the reconstruction's own start (g-file: `init_psi` from its LCFS
shape; IDS: from the boundary's shape, with the `coil_init` seed), and the
solver solves once. The cold state is measured by the engine backend's own
final measurement and compared with the delivered one: l_i(3) and q0 at the
q-row radius against the sigma=0 gates (`jbs_tol_li`, `jbs_tol_q0` -- the
zero-perturbation draw's own), beta_pol and the magnetic axis reported (the
sigma=0 rule gates neither). A miss stamps `Baseline.engine["bistability"]`
with `bistable=True`, the deltas, both states and the gates, and a
`RuntimeWarning` names it; a cold solve that fails is recorded
(`outcome="cold_solve_failed"`, `bistable=None`) and warned, never hidden.
The delivered equilibrium is restored afterwards: the baseline, its archive
and the draws are unchanged. The record is archived with the baseline's
engine record. A cold start converging to the delivered state does not prove
there is no other branch; a miss shows there is one.

**Where is each draw's axis? (`engine["axis"]`, always recorded.)** Every
archived engine draw records its archived state's magnetic axis against the
reconstruction's delivered axis (`Baseline.engine["delivered_axis"]`, the
reference, with the delivered minor radius a): `R`, `Z`, `reference_R`,
`reference_Z`, `dR_mm`, `dZ_mm`, `displacement_mm`, `displacement_frac_a`
and `axis_shift_large` -- whether the displacement exceeds
`engine.AXIS_BRANCH_FRAC` (0.01) x a (`engine.axis_shift_record`; the same
constant is the MSE stage's branch-check discriminator,
`MSE_FD_BRANCH_AXIS_FRAC`, where it judges ONE coefficient step). On a draw
it is a displacement, not a branch label: kinetic draws move the axis
continuously by several % of a. The flag, the displacement and its
components are stamped on the draw group (`axis_shift_large`,
`axis_shift_frac_a`, `axis_dR_mm`, `axis_dZ_mm`, `axis_minor_radius_m`;
names in `bouquet.schema`) and counted on the scan (`n_axis_shift_large`).

**Is the ensemble on two branches? (`engine.axis_bimodality`, per scan.)**
After a scan is archived (and again after `merge_archives`),
`engine_draws.stamp_axis_scan` recounts the scan's draws from the archive and
tests their axis Z for two clusters: split at the largest gap leaving at
least `AXIS_BIMODAL_MIN_COUNT` (3) draws per side; bimodal when the gap
exceeds `AXIS_BIMODAL_GAP_RATIO` (1.0) x the larger within-cluster standard
deviation and the centres sit more than `AXIS_BRANCH_FRAC` x a apart; not
testable below `AXIS_BIMODAL_MIN_N` = 24 draws (owner decision 2026-10-15:
on unimodal normal samples the rule says "bimodal" in about 20-37 % of
samples at n = 6-12 for axis spreads of 6-20 mm, about 5-9 % at 3 mm, about
10-11 % at n = 24 and 3 % at n = 36; until then the verdict was issued from
n = 6). The scan carries `axis_bimodal` and
`axis_bimodal_json` (method `AXIS_BIMODAL_METHOD`, version
`axis_bimodality/1`, the constants, both centres and counts); a bimodal scan
whose reconstruction stamp is not "bistable" stamps the baseline
`bistability_source="draws"` (`_baseline` attr and the archived and live
`Baseline.engine["bistability"]`, with `draws_evidence`); the reconstruction
check's own "bistable" carries `bistability_source="reconstruction"`. A
`"draws"` source belongs to ONE scan: the session's live record never
carries it into another scan's archived `_baseline`
(`GenerateEngineDraws.store_baseline` drops it), and a scan with no axis
record or too few draws to test clears it from its baseline and from the
live record. The
summaries, `ScanView.spread()` and `draw_scalars` report it
([workflows.md](workflows.md#the-in-spec-rule), "Bistable equilibria"). All
of it is a report: nothing is accepted, refused or filtered on it. Archives
written with the earlier per-draw `axis_branch` / scan `n_axis_other` names
are read (`schema.read_axis_entry`, `schema.read_scan_axis_counts`) and
migrated when re-stamped.

`verify_sigma0_consistency()` under the engine runs ONE draw through the
route `generate()` runs -- it calls `generate(n=1)` itself, with every
perturbation zero and the bootstrap scale 1.0, archiving into a temporary
file (the configured archive is never touched): generate_bouquet's baseline
re-solve and warm start, its strong coil regularisation and the
reconstruction's own one installed for the loop (above), the
reconstruction's own isoflux held by the homotopy (above), the homotopy and
the post-homotopy
stage, under `engine_draw_solve_maxits`. (Until 2026-10-04 it ran only the
loop, `engine_draws.verify_zero_perturbation`, from whatever state the
solver held, under whatever regularisation was installed -- not the draw's
route; that function is kept, documented as the loop stage only.) `passed`
needs every gate of every stage; each gated quantity is recorded with its
value, its bound and the setting the bound comes from (`gates`, per stage):

| stage | gated | bound |
|---|---|---|
| LOOP (`stages["loop"]`) | request bit-identical, loop converged; `r_j`, `r_I` of the draw's bootstrap against `lambda_bs*`; `|dl_i|`; `|dq0|` at the q-row radius | `jbs_rtol_j`, `jbs_rtol_Ip`, `jbs_tol_li`, `jbs_tol_q0` |
| ARCHIVED (`stages["archived"]`, after the homotopy and the post-homotopy stage) | `r_j`, `r_I` of the bootstrap the draw carries there against `lambda_bs*` on its geometry; `|dl_i|`; `|dq0|` at the q-row radius | the same |
| ARCHIVED GEOMETRY (`stages["archived_geometry"]`, read from the temporary archive) | max F-coil and max VSC-coil drift [%] of the archived coil currents against the reconstruction's (`_baseline/coil_currents`; the homotopy's own per-coil measure); LCFS rms deviation [mm] of the archived boundary from the reconstruction's (`recon_lcfs_ref`; the boundary filter's own metric) | the HARD coil bound the route rejects at -- `coil_drift_hard_factor x coil_drift` when configured, else the tightest homotopy stage `homotopy_passes[-1]`, else (`engine_draw_homotopy=False`) `coil_drift` -- and the in-spec boundary cut (`filtering.rms_max_mm` as resolved by `_boundary_cut`; recorded as not gated when the cut is `"off"`) |

`dq95`, the flux-range change, the boundary max deviation and the homotopy
drifts are reported; a rejected draw fails; an unreadable gated quantity
fails. (Until 2026-10-06 `passed` gated r_j, r_I and |dl_i| only; q0, the
coils and the boundary were reported.) No tolerance is new: every bound is
the loop's, the homotopy's or the in-spec filter's own number. The
top-level `r_j` / `r_I` / `dl_i` / `dq0` / `dq95` are the archived state's;
`coil_reg` is the regularisation record of the draw's loop. The solver
state, the isoflux targets and every attribute `generate()` sets are
restored afterwards.

**Post-hoc filters, not matching.** A draw matches no l_i, q0 or MSE row;
l_i, beta_p, beta_N and Lambda = beta_p + l_i(3)/2 drift and are
recorded. Since 2026-10-14 (owner decision) a draw is in spec on the
magnetics alone ([workflows.md](workflows.md#the-in-spec-rule)); the
OPT-IN bands -- an l_i band (`|l_i - l_i*| <= l_i_tolerance l_i*`, or
`<= inspec_band_nsigma l_i_sigma`, around the reconstruction's l_i; the
old default `l_i_tolerance = 0.05` applied it to every draw), a beta_p band
(`beta_p_target`, `beta_p_sigma`) and `constrain_sawteeth` (`q0 >= 1` at
the q-row radius, psi_N = psi_pad, the legacy gate's radius) -- are applied
to the ARCHIVED draw when on: a draw outside a band is archived with `in_spec=False` and
`passes_draw_band=False`, never dropped; the until-N ledger and
`.filter()`'s `selected` both AND the band into the coil + boundary verdict.
Rejected (never archived, never counted), with their
`DRAW_REJECTION_REASONS` code: non-physical DRAWN kinetics
(`kinetics_nonphysical`: n_e, n_i, T_e or T_i not strictly positive, Z_eff
below 1 or a non-finite value at any node -- found before any solve, with the
quantity, the value and psi_N in the rejection record's `info`; nothing is
clipped), a loop that does not converge
(`jbs_not_converged`), a non-finite bootstrap (`jbs_non_finite`, at once), a
failed first solve (`anchor_solve_failed`, the anchor's analog: the stored
state composed with the draw's components), a refused amplitude closure
(`engine_closure_refused`), a coil saturation (`coil_saturation_jbs_loop` /
`_post_homotopy`), the homotopy and post-homotopy codes as before, and --
with a cap set (`engine_draw_solve_maxits`, default 100) --
`homotopy_maxits` / `post_homotopy_maxits` (below).

**Bootstrap refresh after the first solve (`engine_draw_bootstrap_refresh`,
default off).** The loop's start is computed on the reconstruction's
geometry `G*`; the first solve moves the geometry (the flux range by
several per cent for a typical inductive sample) and with it the Redl
bootstrap, so a relaxed blend toward the stale start costs passes. With the
setting on, after the loop's FIRST solve the anchor's form is re-evaluated
with the draw-kinetics Redl taken on that solved geometry,
`scale [lambda_bs* + Redl(draw kin; G_1) - Redl(recon kin; G*)]`
(`engine_draws.REFRESH_SOURCE`), and pass 2 restarts from it instead of the
blend `(1 - omega) lambda_0 + omega J_1`; every later pass blends as usual
(`jbs_loop.run_jbs_loop(start_refresh=...)`). Zero extra solves and zero
extra Redl evaluations (pass 1's Redl is the loop's own). It changes the
PATH only: the first request (still bit-identical at zero perturbation),
every criterion, tolerance and the ceiling are untouched, and every pass is
judged against the bootstrap it was solved with. At zero perturbation the
refreshed bootstrap is `lambda_bs*` plus the change of Redl between the
stored and the re-solved equilibrium: rounding on the toy stand-in, and on
the live solver the re-solve's own reproduction of `G*` (measured on the
g-file example: a refresh step of r_j = 1.9e-5, against `jbs_rtol_j` =
1e-3; the pass count at zero perturbation is unchanged, 3). The loop
record carries `bootstrap_refresh` (pass 1's residuals before it, pass 2's
after it, the refresh step, `I_BS` start / evaluated / refreshed).

### l_i controllability

How much the draws' l_i scatters is recorded per draw: `deltas` carries
the change of l_i(3) and l_i(1) against the reconstruction's delivered
values and the change of the poloidal flux range `psi_b - psi_a` (the
geometry's `dpsi_dpsiN`), absolute [Wb/rad] and relative
(`flux_range`, `flux_range_rel`); `archived.deltas` the same for the
post-homotopy state. No extra solves.

Where the scatter comes from was established by a channel-off measurement
(the report "bouquet unified engine: why the six-draw batch looks the way
it does"): on the g-file example, with the same realisations, the
inductive-shape sample alone gives about 10 % RMS in l_i of the 11 % RMS
with every channel on; the kinetics give about 2 %, the bootstrap scale
0.1 %. The flux range moves with l_i in sign and size (-8 % to +10 % on
those draws): it is the solved equilibrium's first-order response to the
current shape.

An earlier version recorded a per-draw linear split of the l_i change into
inductive / bootstrap / pressure-term / amplitude parts plus a remainder,
from the closure's l_i model on the reconstruction's geometry. That model
holds the flux range fixed, so the remainder -- 50-86 % of the change on
the measured draws -- was the geometry's response, and the split was
misleading as a variance budget; it was removed.

**Cost.** Every draw records solves, passes and wall time by stage
(`anchor` -- no solve --, `loop`, `homotopy`, `post_homotopy`, `filters`,
`archive`), counted by the draw loop's `DrawSolveGuard` (every GS solve of
the draw). On the toy a draw takes 3-5 loop passes (3 at zero perturbation)
and one solve per pass; the homotopy adds one solve per stage it runs.
`python tests/probes/measure_engine.py OUTDIR --draws 6 --seed 12345` writes
the live numbers for the g-file example (the legacy batch measured 405-1407 s
per draw, 4 archived / 2 rejected / 1 in spec at that seed).

**The solve cap.** `engine_draw_solve_maxits` (default **100**, the owner's
value of 2026-09-30; `None` = the solver's own cap) caps EVERY
Grad-Shafranov solve inside an engine draw: the loop (its first solve and
every pass) and the post-homotopy passes through the engine's solve wrapper
(`TokaMakerBackend.solve`, set and restored per solve), and every homotopy
stage and rollback re-solve (installed on the solver for the homotopy stage
and restored after it). The zero-perturbation draw of
`verify_sigma0_consistency` runs under it too. The reconstruction runs
under the solver's own cap. The legacy draws' `draw_solve_maxits` (any
value but `"auto"` / `None`) and their opt-in rescue
(`draw_solve_retry_urf`, `draw_solve_loose_tol`) are REFUSED under the
engine by the unread-settings rule (they would be silently ignored); the
engine draws are never rescued. The legacy path never reads
`engine_draw_solve_maxits`.

A solve that converges under the cap is untouched (measured on the
synthetic g-file example, fixed build: loop ≤ 15, post-homotopy ≤ 18,
homotopy stages 13-21 iterations for every archived draw; the two slow
homotopy solves at 264 and 384 iterations belonged to draws rejected for
saturation anyway). A solve that stops at the cap (the solver's own
`Exceeded "maxits"`, anywhere in the exception chain):

| where | what happens | code |
|---|---|---|
| a homotopy STAGE | a failed stage like any other: roll back to the last good (looser) stage and re-solve there, as a non-converging stage does; with no earlier good stage the draw is rejected | `homotopy_maxits` (only when rejected) |
| the rollback re-solve | the draw is rejected | `homotopy_maxits` |
| a post-homotopy pass | the draw is rejected | `post_homotopy_maxits` |
| a loop solve | the draw is rejected with the loop's code | `anchor_solve_failed` (pass 1) / `perturb_failed` |

A rollback re-solve that fails for ANY other reason (a non-finite abort,
a lost plasma, ...) rejects the draw too, with its own code
`homotopy_rollback_failed` -- with a cap set or with
`engine_draw_solve_maxits=None` alike: the state such a failure leaves
behind is not a converged solve, so an engine draw never goes on from it
(recorded on `GenerateEngineDraws.rollback_failures`: stage, what triggered
the rollback -- `saturation` or `failed stage` --, the error). Legacy
draws follow the same rule since the owner-approved change of 2026-10-05:
any failed rollback re-solve (capped by `draw_solve_maxits` or not) rejects
a legacy draw with `homotopy_rollback_failed` (before, they printed "stats
may be stale" and went on).

Every capped solve is recorded on `GenerateEngineDraws.cap_events` and
`Bouquet.engine_draw_cap_events` (draw, stage, `iterations` = the cap,
`seconds`, `outcome` = `rolled_back` | `rejected`, the error), in the
archived draw's `homotopy.cap_events`, and summarised in one printed
`[generate]` line. A homotopy stage's event also carries `vertical_runaway`
(the vertical-control channel's excursion as a fraction of the stage's `#VSC`
bound, the axis displacement and its vertical part `axis_dZ_frac_a` in units
of a, the threshold) and `label` (`"vertical_runaway"` when the channel is at
its bound and `|dZ|` is more than `VERTICAL_RUNAWAY_AXIS_FRAC` = 0.1 a, else
`None`; "The boundary
constraint" above) -- a label, never a gate.

## The pressure handed to the solver (`edge_pprime_pin`, `separatrix_pressure`)

The backend builds every `P'` profile and axis target through one helper
(`bouquet/edge_pressure.py`; the physics is in
[physics-notes.md](physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)).
Two `GenerationConfig` settings, shared with the legacy paths:

- `edge_pprime_pin` (default `True`, the behaviour before the setting): the
  last `P'` node is zeroed. `False` keeps the profile's own derivative at
  `psi_N = 1`.
- `separatrix_pressure` (default `"offset"` since 2026-10-02; `"legacy"` is
  the behaviour before the setting): `"offset"` passes `p_axis - p_sep` as
  the axis target and adds `p_sep` back at reporting and delivery.
  `"legacy"` passes the full axis pressure; with a non-zero `p_sep` the
  solver then inflates `P'` by `p_axis / (p_axis - p_sep)`.

**The default changed (owner-approved physics change, 2026-10-02).**
`separatrix_pressure` moved from `"legacy"` to `"offset"`. On real g-file
and IDS cases (both engines' paths, pin on), `"offset"` brought the
full-frame `beta_N` and `W_MHD` closer to the input on every comparable
g-file case, by 1.4-8 points (median 3.5), and by about 0.5 points on IDS
slices; every case converged, with the same passes, solves and wall time,
and `l_i`, `q` and the current distances unchanged. What it changes for an
existing run whose solve pressure is not zero at `psi_N = 1`: `P'` in the
solve is scaled by `(p_axis - p_sep) / p_axis`; the reported pressure,
`beta` and `W_MHD` (and the delivered g-files' `PRES`) move toward the
input's full-pressure values. With `p_sep = 0` nothing changes. Set
`separatrix_pressure="legacy"` to reproduce the pre-change numbers (bit for
bit, proven by the frozen-copy tests); a stored config that predates the
setting reloads with `"legacy"` and says so.

How they meet the engine:

- **The contract's pressure is unchanged.** `p_sep` is its last node
  (thermal + impurity + fast); a draw's is that of its own perturbed
  pressure (`TokaMakerBackend.p_sep()` follows `set_inputs`).
- **The composition needs nothing new.** The pressure-driven term
  `p'(<R> - F^2<1/R>/<B^2>)` is recomputed every pass from the `p'` READ BACK
  from the solved equilibrium, so it follows whatever the solver was handed:
  with the pin on it goes to zero at the boundary, with the pin off it does
  not; under `"offset"` it is smaller by `(p_axis - p_sep) / p_axis`. The
  inductive component absorbs the difference through the rows. A draw's
  first-pass shift of that term uses the helper's `d p / d psi_N`.
- **The zero-perturbation identity holds by construction** under every
  combination: the stored state composes the stored request bit for bit
  whatever the settings, and a zero-perturbation draw's pressure (hence its
  `p_sep`) is the contract's.
- **Records.** `engine_record["edge_pressure"]` (and
  `Baseline.edge_pressure`, `delivered_state["edge_pressure"]`): the
  settings, `p_sep`, `p_axis`, `p_sep_applied`, `pax_target` and `frames` --
  beta / `W_MHD` of the delivered equilibrium in the solver's frame and with
  `p_sep` added back. A draw's `delivered` / `archived` blocks carry
  `pressure_frames`; their `beta_n` is the full-frame value (the solver's
  own when nothing is added back). The reconstruction summary and the probe
  (`tests/probes/measure_engine.py`, `distance.pressure_frames`) compare
  each frame with the input's same-definition quantity.
- **Delivery.** Under `"offset"` EVERY g-file bouquet writes carries the
  full pressure: the archive's `_baseline` g-file and every draw's g-file
  (`generate()`), and the reconstruction's own g-file written with
  `Bouquet.save_baseline_eqdsk(path)` (the live state `prepare_baseline()`
  left; refused once a later solve has moved it), each with that
  equilibrium's own `p_sep` (`edge_pressure["p_sep_applied"]`) as the
  boundary pressure; `PPRIME` is unchanged. The reconstruction's g-file and
  the archive baseline's are the same save call, so for the same state they
  carry the same `PRES`. A bare `mygs.save_eqdsk(...)` writes the SOLVER
  frame (`PRES` zero at the boundary, lower by `p_sep` everywhere): do not
  use it to deliver an equilibrium.

Both settings change the physics when moved (and `separatrix_pressure`
did, when its default changed); the measurements are in the change
summary.

## Cost

Measured on the toy: Ip + l_i converges in 5–7 passes (the toy's flux range
responds to l_i with the measured log-gain of 2); anchor 2 + passes + delivery
2 solves. The MSE stage adds 1 + 8 finite-difference solves and 4–8 passes,
plus 1 + 8 solves per Jacobian refresh at its convergence (one when the
refreshed step is within the criterion).
A draw: 3-5 loop passes of one solve each (see [Draws](#draws)). The live
numbers are written by the solver probe. The report-only boundary-partition
probe adds ONE capped solve to every unified-engine reconstruction (its wall
is stamped as `boundary_balance.partition.wall_s`).

**Two bounds on a draw's cost.** Structural: every engine-draw solve is capped
at `engine_draw_solve_maxits` (default 100) iterations and a draw runs a bounded
number of solves (the loop's passes, the homotopy stages and a rollback, the
post-homotopy passes), so no draw runs forever. Wall clock:
`GenerationConfig.draw_wall_budget_s` (default 600 s, `None` = off) -- checked
before each of a draw's solves (a running GS solve is never interrupted) and
once before the draw is archived; a draw over it is rejected as
`draw_wall_budget`, recorded like every rejection (attempt record, counts, the
summary line; the reason it pre-empted kept as `masked_reason`), and the budget
and its rejection count are stamped in the generation provenance
(`draw_wall_budget_s`, `draw_wall_budget_rejections`). 600 s is five times the
120 s per-draw target and about three times the slowest normal draws (150-190 s);
it ends only stalled draws (one was measured at 55 min). Owner decision
2026-10-16. It is wall-clock, so it is machine-dependent: on a slower or busier
host a draw near it can be rejected where a faster one archives it, and an
until-N run with a fixed seed can then archive a different set; the stamp and
the rejection count say when it bit. The sigma=0 verification draw
(`verify_sigma0_consistency`) runs with the budget OFF and records so
(`wall_budget`): it verifies reproduction, and a budget rejection there would be
a false failure set by machine load. Its kinetic clips stay on (the same path as every
draw); their counters are recorded (`sigma0_clips`), and a clip that fired at
sigma=0 -- the baseline itself outside `Z_eff >= 1` or `0 <= n_i <= n_e - z_fast`
there -- is recorded as `baseline_outside_physical_window` with the counts: a
finding on the baseline, not a changed verdict.

## Deviations from the design note

See the Stage 2 report; in short: the l_i discrepancy is measured against the
model on the NEW geometry (the note's `model(x_k; G_k)` form is unstable on
the gain-2 geometry mode); the MSE linearisation is centred on the solved
(β-blended) coefficients; `"fd_chord"` is offered beside `"fd_broyden"`;
the q0 row starts from the anchor's achieved axis current; the IDS q0 target
is the source's own q at the row radius. The draws: the inductive is drawn
in toroidal units and its negative excursions follow the standard route's
floor-zone rule (not Fix C's all-positive retry); the loop starts from
`lambda_bs*` plus the Redl kinetic increment on the starting equilibrium
(zero at identity); the first pass shifts `G*`'s `p'` by the draw's pressure
change; the Ip-row target is the Ip the delivered composition carries in the
exact measure (the posterior on a soft row); the optional q0 row is the
two-scalar closure in increment form; the delivery correction, when on,
keeps being updated per pass in the draw as in the reconstruction.
